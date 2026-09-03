#include "jeff/Translation/Deserialize.hpp"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/serialize.h>
#include <jeff.capnp.h>
#include <kj/common.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace {

struct DeserializationContext {
    llvm::DenseMap<uint32_t, mlir::Value> values;
    capnp::List<jeff::Value>::Reader jeffValues;
    llvm::DenseMap<uint16_t, mlir::func::FuncOp> funcs;
    llvm::SmallVector<std::string> strings;

    mlir::Value getValue(uint32_t id) {
        auto it = values.find(id);
        if (it == values.end()) {
            llvm::errs() << "Value " << id << " not found\n";
            llvm::report_fatal_error("Value not found");
        }
        return it->second;
    }

    void setValue(uint32_t id, mlir::Value value) {
        if (values.contains(id)) {
            llvm::errs() << "Value " << id << " already exists\n";
            llvm::report_fatal_error("Value already exists");
        }
        values[id] = value;
    }

    mlir::func::FuncOp getFunc(uint16_t id) {
        auto it = funcs.find(id);
        if (it == funcs.end()) {
            llvm::errs() << "Function " << id << " not found\n";
            llvm::report_fatal_error("Function not found");
        }
        return it->second;
    }

    void setFunc(uint16_t id, mlir::func::FuncOp func) {
        if (funcs.contains(id)) {
            llvm::errs() << "Function " << id << " already exists\n";
            llvm::report_fatal_error("Function already exists");
        }
        funcs[id] = func;
    }

    [[nodiscard]] jeff::Type::Reader getJeffType(uint32_t id) const {
        return jeffValues[id].getType();
    }

    [[nodiscard]] int64_t getLength(uint32_t id) const {
        auto type = getJeffType(id);
        switch (type.which()) {
        case jeff::Type::QUREG:
            if (type.getQureg().isStatic()) {
                return type.getQureg().getStatic();
            } else {
                return mlir::ShapedType::kDynamic;
            }
        case jeff::Type::INT_ARRAY:
            if (type.getIntArray().getLength().isStatic()) {
                return type.getIntArray().getLength().getStatic();
            } else {
                return mlir::ShapedType::kDynamic;
            }
        case jeff::Type::FLOAT_ARRAY:
            if (type.getFloatArray().getLength().isStatic()) {
                return type.getFloatArray().getLength().getStatic();
            } else {
                return mlir::ShapedType::kDynamic;
            }
        default:
            llvm::errs() << "Value " << id << " does not have a length\n";
            llvm::report_fatal_error("Value does not have a length");
        }
    }
};

//===----------------------------------------------------------------------===//
// Qubit operations
//===----------------------------------------------------------------------===//

void deserializeQubitAlloc(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                           DeserializationContext& ctx) {
    auto allocOp = mlir::jeff::QubitAllocOp::create(builder);
    ctx.setValue(operation.getOutputs()[0], allocOp.getResult());
}

void deserializeQubitFree(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                          DeserializationContext& ctx) {
    mlir::jeff::QubitFreeOp::create(builder, ctx.getValue(operation.getInputs()[0]));
}

void deserializeQubitFreeZero(mlir::ImplicitLocOpBuilder& builder,
                              const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    mlir::jeff::QubitFreeZeroOp::create(builder, ctx.getValue(operation.getInputs()[0]));
}

void deserializeMeasure(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                        DeserializationContext& ctx) {
    auto op = mlir::jeff::QubitMeasureOp::create(builder, ctx.getValue(operation.getInputs()[0]));
    ctx.setValue(operation.getOutputs()[0], op.getResult());
}

void deserializeMeasureNd(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                          DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::QubitMeasureNDOp::create(builder, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getOutQubit());
    ctx.setValue(outputs[1], op.getResult());
}

void deserializeReset(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                      DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::QubitResetOp::create(builder, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getOutQubit());
}

template <typename OpType>
void deserializeOneTargetZeroParameter(mlir::ImplicitLocOpBuilder& builder,
                                       const jeff::Op::Reader& operation,
                                       DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto numControls = gate.getControlQubits();
    llvm::SmallVector<mlir::Value> controls;
    for (uint8_t i = 1; i < 1 + numControls; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    auto op =
        OpType::create(builder, ctx.getValue(inputs[0]), controls,
                       static_cast<uint8_t>(controls.size()), gate.getAdjoint(), gate.getPower());
    ctx.setValue(outputs[0], op.getOutQubit());
    for (uint8_t i = 1; i < 1 + numControls; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i - 1]);
    }
}

template <typename OpType>
void deserializeOneTargetOneParameter(mlir::ImplicitLocOpBuilder& builder,
                                      const jeff::Op::Reader& operation,
                                      DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto numControls = gate.getControlQubits();
    llvm::SmallVector<mlir::Value> controls;
    for (uint8_t i = 1; i < 1 + numControls; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    auto rotation = ctx.getValue(inputs[1 + numControls]);
    auto op =
        OpType::create(builder, ctx.getValue(inputs[0]), rotation, controls,
                       static_cast<uint8_t>(controls.size()), gate.getAdjoint(), gate.getPower());
    ctx.setValue(outputs[0], op.getOutQubit());
    for (uint8_t i = 1; i < 1 + numControls; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i - 1]);
    }
}

void deserializeU(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                  DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto numControls = gate.getControlQubits();
    llvm::SmallVector<mlir::Value> controls;
    for (uint8_t i = 1; i < 1 + numControls; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    auto theta = ctx.getValue(inputs[1 + numControls]);
    auto phi = ctx.getValue(inputs[2 + numControls]);
    auto lambda = ctx.getValue(inputs[3 + numControls]);
    auto op = mlir::jeff::UOp::create(builder, ctx.getValue(inputs[0]), theta, phi, lambda,
                                      controls, static_cast<uint8_t>(controls.size()),
                                      gate.getAdjoint(), gate.getPower());
    ctx.setValue(outputs[0], op.getOutQubit());
    for (uint8_t i = 1; i < 1 + numControls; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i - 1]);
    }
}

void deserializeSwap(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                     DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto numControls = gate.getControlQubits();
    llvm::SmallVector<mlir::Value> controls;
    for (uint8_t i = 2; i < 2 + numControls; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    auto op = mlir::jeff::SwapOp::create(builder, ctx.getValue(inputs[0]), ctx.getValue(inputs[1]),
                                         controls, static_cast<uint8_t>(controls.size()),
                                         gate.getAdjoint(), gate.getPower());
    ctx.setValue(outputs[0], op.getOutQubitOne());
    ctx.setValue(outputs[1], op.getOutQubitTwo());
    for (uint8_t i = 2; i < 2 + numControls; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i - 2]);
    }
}

void deserializeGPhase(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                       DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto numControls = gate.getControlQubits();
    llvm::SmallVector<mlir::Value> controls;
    for (uint8_t i = 0; i < numControls; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    auto rotation = ctx.getValue(inputs[numControls]);
    auto op = mlir::jeff::GPhaseOp::create(builder, rotation, controls,
                                           static_cast<uint8_t>(controls.size()), gate.getAdjoint(),
                                           gate.getPower());
    for (uint8_t i = 0; i < numControls; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i]);
    }
}

void deserializeWellKnown(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                          DeserializationContext& ctx) {
    switch (const auto wellKnown = operation.getInstruction().getQubit().getGate().getWellKnown()) {
    case jeff::WellKnownGate::X:
        deserializeOneTargetZeroParameter<mlir::jeff::XOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::Y:
        deserializeOneTargetZeroParameter<mlir::jeff::YOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::Z:
        deserializeOneTargetZeroParameter<mlir::jeff::ZOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::S:
        deserializeOneTargetZeroParameter<mlir::jeff::SOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::T:
        deserializeOneTargetZeroParameter<mlir::jeff::TOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::R1:
        deserializeOneTargetOneParameter<mlir::jeff::R1Op>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::RX:
        deserializeOneTargetOneParameter<mlir::jeff::RxOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::RY:
        deserializeOneTargetOneParameter<mlir::jeff::RyOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::RZ:
        deserializeOneTargetOneParameter<mlir::jeff::RzOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::H:
        deserializeOneTargetZeroParameter<mlir::jeff::HOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::U:
        deserializeU(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::SWAP:
        deserializeSwap(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::I:
        deserializeOneTargetZeroParameter<mlir::jeff::IOp>(builder, operation, ctx);
        break;
    case jeff::WellKnownGate::GPHASE:
        deserializeGPhase(builder, operation, ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize well-known gate " << static_cast<int>(wellKnown)
                     << "\n";
        llvm::report_fatal_error("Unknown well-known gate");
    }
}

void deserializeCustom(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                       DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto custom = gate.getCustom();
    const auto name = ctx.strings[custom.getName()];
    const auto numControls = gate.getControlQubits();
    const auto numTargets = custom.getNumQubits();
    const auto numQubits = static_cast<uint32_t>(numTargets + numControls);
    const auto numParams = custom.getNumParams();
    llvm::SmallVector<mlir::Value> targets;
    for (uint32_t i = 0; i < numTargets; ++i) {
        targets.push_back(ctx.getValue(inputs[i]));
    }
    llvm::SmallVector<mlir::Value> controls;
    for (uint32_t i = numTargets; i < numQubits; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    llvm::SmallVector<mlir::Value> params;
    for (uint32_t i = numQubits; i < numQubits + numParams; ++i) {
        params.push_back(ctx.getValue(inputs[i]));
    }
    auto op = mlir::jeff::CustomOp::create(builder, targets, controls, params, numControls,
                                           gate.getAdjoint(), gate.getPower(), name, numTargets,
                                           numParams);
    for (uint32_t i = 0; i < numTargets; ++i) {
        ctx.setValue(outputs[i], op.getOutTargetQubits()[i]);
    }
    for (uint32_t i = numTargets; i < numQubits; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i - numTargets]);
    }
}

void deserializePpr(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                    DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto gate = operation.getInstruction().getQubit().getGate();
    const auto pauliString = gate.getPpr().getPauliString();
    const auto numTargets = pauliString.size();
    const auto numControls = gate.getControlQubits();
    const auto numQubits = static_cast<uint32_t>(numTargets + numControls);
    llvm::SmallVector<mlir::Value> targets;
    for (uint32_t i = 0; i < numTargets; ++i) {
        targets.push_back(ctx.getValue(inputs[i]));
    }
    llvm::SmallVector<mlir::Value> controls;
    for (uint32_t i = numTargets; i < numQubits; ++i) {
        controls.push_back(ctx.getValue(inputs[i]));
    }
    auto rotation = ctx.getValue(inputs[numQubits]);
    llvm::SmallVector<int32_t> pauliStringVector;
    for (auto pauli : pauliString) {
        pauliStringVector.push_back(static_cast<int32_t>(pauli));
    }
    auto pauliStringArrayAttr =
        mlir::DenseI32ArrayAttr::get(builder.getContext(), pauliStringVector);
    auto op = mlir::jeff::PPROp::create(builder, targets, controls, rotation, numControls,
                                        gate.getAdjoint(), gate.getPower(), pauliStringArrayAttr);
    for (uint32_t i = 0; i < numTargets; ++i) {
        ctx.setValue(outputs[i], op.getOutQubits()[i]);
    }
    for (uint32_t i = numTargets; i < numQubits; ++i) {
        ctx.setValue(outputs[i], op.getOutCtrlQubits()[i - numTargets]);
    }
}

void deserializeGate(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                     DeserializationContext& ctx) {
    switch (const auto gate = operation.getInstruction().getQubit().getGate(); gate.which()) {
    case jeff::QubitGate::WELL_KNOWN:
        deserializeWellKnown(builder, operation, ctx);
        break;
    case jeff::QubitGate::CUSTOM:
        deserializeCustom(builder, operation, ctx);
        break;
    case jeff::QubitGate::PPR:
        deserializePpr(builder, operation, ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize gate instruction " << gate.which() << "\n";
        llvm::report_fatal_error("Unknown gate instruction");
    }
}

void deserializeQubit(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                      DeserializationContext& ctx) {
    switch (const auto qubit = operation.getInstruction().getQubit(); qubit.which()) {
    case jeff::QubitOp::ALLOC:
        deserializeQubitAlloc(builder, operation, ctx);
        break;
    case jeff::QubitOp::FREE:
        deserializeQubitFree(builder, operation, ctx);
        break;
    case jeff::QubitOp::FREE_ZERO:
        deserializeQubitFreeZero(builder, operation, ctx);
        break;
    case jeff::QubitOp::MEASURE:
        deserializeMeasure(builder, operation, ctx);
        break;
    case jeff::QubitOp::MEASURE_ND:
        deserializeMeasureNd(builder, operation, ctx);
        break;
    case jeff::QubitOp::RESET:
        deserializeReset(builder, operation, ctx);
        break;
    case jeff::QubitOp::GATE:
        deserializeGate(builder, operation, ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize qubit instruction " << qubit.which() << "\n";
        llvm::report_fatal_error("Unknown qubit instruction");
    }
}

//===----------------------------------------------------------------------===//
// Qureg operations
//===----------------------------------------------------------------------===//

void deserializeQuregAlloc(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                           DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto qregType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[0]));
    auto allocOp = mlir::jeff::QuregAllocOp::create(builder, qregType, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], allocOp.getResult());
}

void deserializeQuregFreeZero(mlir::ImplicitLocOpBuilder& builder,
                              const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    mlir::jeff::QuregFreeZeroOp::create(builder, ctx.getValue(operation.getInputs()[0]));
}

void deserializeQuregExtractIndex(mlir::ImplicitLocOpBuilder& builder,
                                  const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::QuregExtractIndexOp::create(builder, ctx.getValue(inputs[0]),
                                                      ctx.getValue(inputs[1]));
    ctx.setValue(outputs[0], op.getOutQreg());
    ctx.setValue(outputs[1], op.getOutQubit());
}

void deserializeQuregInsertIndex(mlir::ImplicitLocOpBuilder& builder,
                                 const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::QuregInsertIndexOp::create(
        builder, ctx.getValue(inputs[0]), ctx.getValue(inputs[1]), ctx.getValue(inputs[2]));
    ctx.setValue(outputs[0], op.getOutQreg());
}

void deserializeQuregExtractSlice(mlir::ImplicitLocOpBuilder& builder,
                                  const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto outQregType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[0]));
    auto newQregType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[1]));
    auto op = mlir::jeff::QuregExtractSliceOp::create(
        builder, outQregType, newQregType, ctx.getValue(inputs[0]), ctx.getValue(inputs[1]),
        ctx.getValue(inputs[2]));
    ctx.setValue(outputs[0], op.getOutQreg());
    ctx.setValue(outputs[1], op.getNewQreg());
}

void deserializeQuregInsertSlice(mlir::ImplicitLocOpBuilder& builder,
                                 const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::QuregInsertSliceOp::create(
        builder, ctx.getValue(inputs[0]), ctx.getValue(inputs[1]), ctx.getValue(inputs[2]));
    ctx.setValue(outputs[0], op.getOutQreg());
}

void deserializeQuregLength(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                            DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::QuregLengthOp::create(builder, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getOutQreg());
    ctx.setValue(outputs[1], op.getLength());
}

void deserializeQuregSplit(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                           DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto qregOneType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[0]));
    auto qregTwoType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[1]));
    auto op = mlir::jeff::QuregSplitOp::create(builder, qregOneType, qregTwoType,
                                               ctx.getValue(inputs[0]), ctx.getValue(inputs[1]));
    ctx.setValue(outputs[0], op.getOutQregOne());
    ctx.setValue(outputs[1], op.getOutQregTwo());
}

void deserializeQuregJoin(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                          DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto qregType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[0]));
    auto op = mlir::jeff::QuregJoinOp::create(builder, qregType, ctx.getValue(inputs[0]),
                                              ctx.getValue(inputs[1]));
    ctx.setValue(outputs[0], op.getOutQreg());
}

void deserializeQuregCreate(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                            DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    llvm::SmallVector<mlir::Value> inQreg;
    inQreg.reserve(inputs.size());
    for (auto input : inputs) {
        inQreg.push_back(ctx.getValue(input));
    }
    auto qregType = mlir::jeff::QuregType::get(builder.getContext(), ctx.getLength(outputs[0]));
    auto op = mlir::jeff::QuregCreateOp::create(builder, qregType, inQreg);
    ctx.setValue(outputs[0], op.getOutQreg());
}

void deserializeQuregFree(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                          DeserializationContext& ctx) {
    mlir::jeff::QuregFreeOp::create(builder, ctx.getValue(operation.getInputs()[0]));
}

void deserializeQureg(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                      DeserializationContext& ctx) {
    switch (const auto qureg = operation.getInstruction().getQureg(); qureg.which()) {
    case jeff::QuregOp::ALLOC:
        deserializeQuregAlloc(builder, operation, ctx);
        break;
    case jeff::QuregOp::FREE_ZERO:
        deserializeQuregFreeZero(builder, operation, ctx);
        break;
    case jeff::QuregOp::EXTRACT_INDEX:
        deserializeQuregExtractIndex(builder, operation, ctx);
        break;
    case jeff::QuregOp::INSERT_INDEX:
        deserializeQuregInsertIndex(builder, operation, ctx);
        break;
    case jeff::QuregOp::EXTRACT_SLICE:
        deserializeQuregExtractSlice(builder, operation, ctx);
        break;
    case jeff::QuregOp::INSERT_SLICE:
        deserializeQuregInsertSlice(builder, operation, ctx);
        break;
    case jeff::QuregOp::LENGTH:
        deserializeQuregLength(builder, operation, ctx);
        break;
    case jeff::QuregOp::SPLIT:
        deserializeQuregSplit(builder, operation, ctx);
        break;
    case jeff::QuregOp::JOIN:
        deserializeQuregJoin(builder, operation, ctx);
        break;
    case jeff::QuregOp::CREATE:
        deserializeQuregCreate(builder, operation, ctx);
        break;
    case jeff::QuregOp::FREE:
        deserializeQuregFree(builder, operation, ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize qureg instruction " << qureg.which() << "\n";
        llvm::report_fatal_error("Unknown qureg instruction");
    }
}

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

#define DESERIALIZE_INT_CONST(BIT_WIDTH)                                                           \
    void deserializeIntConst##BIT_WIDTH(mlir::ImplicitLocOpBuilder& builder,                       \
                                        const jeff::Op::Reader& operation,                         \
                                        DeserializationContext& ctx) {                             \
        const auto value = operation.getInstruction().getInt().getConst##BIT_WIDTH();              \
        auto intType = builder.getI##BIT_WIDTH##Type();                                            \
        auto intAttr = mlir::IntegerAttr::get(intType, value);                                     \
        auto op = mlir::jeff::IntConst##BIT_WIDTH##Op::create(builder, intType, intAttr);          \
        ctx.setValue(operation.getOutputs()[0], op.getConstant());                                 \
    }

DESERIALIZE_INT_CONST(1)
DESERIALIZE_INT_CONST(8)
DESERIALIZE_INT_CONST(16)
DESERIALIZE_INT_CONST(32)
DESERIALIZE_INT_CONST(64)

#undef DESERIALIZE_INT_CONST

void deserializeIntUnaryOp(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                           mlir::jeff::IntUnaryOperation unaryOperation,
                           DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::IntUnaryOp::create(builder, ctx.getValue(inputs[0]), unaryOperation);
    ctx.setValue(outputs[0], op.getB());
}

void deserializeIntBinaryOp(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                            mlir::jeff::IntBinaryOperation binaryOperation,
                            DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::IntBinaryOp::create(builder, ctx.getValue(inputs[0]),
                                              ctx.getValue(inputs[1]), binaryOperation);
    ctx.setValue(outputs[0], op.getC());
}

void deserializeIntComparisonOp(mlir::ImplicitLocOpBuilder& builder,
                                const jeff::Op::Reader& operation,
                                mlir::jeff::IntComparisonOperation comparisonOperation,
                                DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::IntComparisonOp::create(builder, ctx.getValue(inputs[0]),
                                                  ctx.getValue(inputs[1]), comparisonOperation);
    ctx.setValue(outputs[0], op.getC());
}

#define ADD_CONST_CASE(BIT_WIDTH)                                                                  \
    case jeff::IntOp::CONST##BIT_WIDTH:                                                            \
        deserializeIntConst##BIT_WIDTH(builder, operation, ctx);                                   \
        break;

#define ADD_UNARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                          \
    case jeff::IntOp::JEFF_ENUM_VALUE:                                                             \
        deserializeIntUnaryOp(builder, operation,                                                  \
                              mlir::jeff::IntUnaryOperation::_##MLIR_ENUM_SUFFIX, ctx);            \
        break;

#define ADD_BINARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                         \
    case jeff::IntOp::JEFF_ENUM_VALUE:                                                             \
        deserializeIntBinaryOp(builder, operation,                                                 \
                               mlir::jeff::IntBinaryOperation::_##MLIR_ENUM_SUFFIX, ctx);          \
        break;

#define ADD_COMPARISON_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                     \
    case jeff::IntOp::JEFF_ENUM_VALUE:                                                             \
        deserializeIntComparisonOp(builder, operation,                                             \
                                   mlir::jeff::IntComparisonOperation::_##MLIR_ENUM_SUFFIX, ctx);  \
        break;

void deserializeInt(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                    DeserializationContext& ctx) {
    switch (const auto intInstr = operation.getInstruction().getInt(); intInstr.which()) {
        ADD_CONST_CASE(1)
        ADD_CONST_CASE(8)
        ADD_CONST_CASE(16)
        ADD_CONST_CASE(32)
        ADD_CONST_CASE(64)
        ADD_UNARY_CASE(NOT, not)
        ADD_UNARY_CASE(ABS, abs)
        ADD_BINARY_CASE(ADD, add)
        ADD_BINARY_CASE(SUB, sub)
        ADD_BINARY_CASE(MUL, mul)
        ADD_BINARY_CASE(DIV_S, divS)
        ADD_BINARY_CASE(DIV_U, divU)
        ADD_BINARY_CASE(POW, pow)
        ADD_BINARY_CASE(AND, and)
        ADD_BINARY_CASE(OR, or)
        ADD_BINARY_CASE(XOR, xor)
        ADD_BINARY_CASE(MIN_S, minS)
        ADD_BINARY_CASE(MIN_U, minU)
        ADD_BINARY_CASE(MAX_S, maxS)
        ADD_BINARY_CASE(MAX_U, maxU)
        ADD_BINARY_CASE(REM_S, remS)
        ADD_BINARY_CASE(REM_U, remU)
        ADD_BINARY_CASE(SHL, shl)
        ADD_BINARY_CASE(SHR, shr)
        ADD_COMPARISON_CASE(EQ, eq)
        ADD_COMPARISON_CASE(LT_S, ltS)
        ADD_COMPARISON_CASE(LTE_S, lteS)
        ADD_COMPARISON_CASE(LT_U, ltU)
        ADD_COMPARISON_CASE(LTE_U, lteU)
    default:
        llvm::errs() << "Cannot deserialize int instruction " << intInstr.which() << "\n";
        llvm::report_fatal_error("Unknown int instruction");
    }
}

#undef ADD_CONST_CASE
#undef ADD_UNARY_CASE
#undef ADD_BINARY_CASE
#undef ADD_COMPARISON_CASE

//===----------------------------------------------------------------------===//
// IntArray operations
//===----------------------------------------------------------------------===//

void deserializeIntArrayConst1(mlir::ImplicitLocOpBuilder& builder,
                               const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto outputs = operation.getOutputs();
    const auto values = operation.getInstruction().getIntArray().getConst1();
    llvm::SmallVector<bool> inArray;
    inArray.reserve(values.size());
    for (auto value : values) {
        inArray.push_back(static_cast<bool>(value));
    }
    auto inArrayAttr = mlir::DenseBoolArrayAttr::get(builder.getContext(), inArray);
    auto tensorType = mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, builder.getI1Type());
    auto op = mlir::jeff::IntArrayConst1Op::create(builder, tensorType, inArrayAttr);
    ctx.setValue(outputs[0], op.getOutArray());
}

#define DESERIALIZE_INT_ARRAY_CONST(BIT_WIDTH)                                                     \
    void deserializeIntArrayConst##BIT_WIDTH(mlir::ImplicitLocOpBuilder& builder,                  \
                                             const jeff::Op::Reader& operation,                    \
                                             DeserializationContext& ctx) {                        \
        const auto outputs = operation.getOutputs();                                               \
        const auto values = operation.getInstruction().getIntArray().getConst##BIT_WIDTH();        \
        llvm::SmallVector<int##BIT_WIDTH##_t> inArray;                                             \
        inArray.reserve(values.size());                                                            \
        for (auto value : values) {                                                                \
            inArray.push_back(static_cast<int##BIT_WIDTH##_t>(value));                             \
        }                                                                                          \
        auto inArrayAttr = mlir::DenseI##BIT_WIDTH##ArrayAttr::get(builder.getContext(), inArray); \
        auto tensorType = mlir::RankedTensorType::get({ctx.getLength(outputs[0])},                 \
                                                      builder.getI##BIT_WIDTH##Type());            \
        auto op =                                                                                  \
            mlir::jeff::IntArrayConst##BIT_WIDTH##Op::create(builder, tensorType, inArrayAttr);    \
        ctx.setValue(outputs[0], op.getOutArray());                                                \
    }

DESERIALIZE_INT_ARRAY_CONST(8)
DESERIALIZE_INT_ARRAY_CONST(16)
DESERIALIZE_INT_ARRAY_CONST(32)
DESERIALIZE_INT_ARRAY_CONST(64)

#undef DESERIALIZE_INT_ARRAY_CONST

void deserializeIntArrayZero(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                             DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto zero = operation.getInstruction().getIntArray().getZero();
    auto tensorType =
        mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, builder.getIntegerType(zero));
    auto op = mlir::jeff::IntArrayZeroOp::create(builder, tensorType, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeIntArrayGetIndex(mlir::ImplicitLocOpBuilder& builder,
                                 const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto tensorType = ctx.getValue(inputs[0]).getType();
    auto entryType = llvm::cast<mlir::RankedTensorType>(tensorType).getElementType();
    auto op = mlir::jeff::IntArrayGetIndexOp::create(builder, entryType, ctx.getValue(inputs[0]),
                                                     ctx.getValue(inputs[1]));
    ctx.setValue(outputs[0], op.getValue());
}

void deserializeIntArraySetIndex(mlir::ImplicitLocOpBuilder& builder,
                                 const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto tensorType = ctx.getValue(inputs[0]).getType();
    auto op =
        mlir::jeff::IntArraySetIndexOp::create(builder, tensorType, ctx.getValue(inputs[0]),
                                               ctx.getValue(inputs[1]), ctx.getValue(inputs[2]));
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeIntArrayLength(mlir::ImplicitLocOpBuilder& builder,
                               const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::IntArrayLengthOp::create(builder, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getLength());
}

void deserializeIntArrayCreate(mlir::ImplicitLocOpBuilder& builder,
                               const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    llvm::SmallVector<mlir::Value> inArray;
    inArray.reserve(inputs.size());
    for (auto input : inputs) {
        inArray.push_back(ctx.getValue(input));
    }
    auto tensorType =
        mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, ctx.getValue(inputs[0]).getType());
    auto op = mlir::jeff::IntArrayCreateOp::create(builder, tensorType, inArray);
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeIntArray(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                         DeserializationContext& ctx) {
    switch (const auto intArray = operation.getInstruction().getIntArray(); intArray.which()) {
    case jeff::IntArrayOp::CONST1:
        deserializeIntArrayConst1(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::CONST8:
        deserializeIntArrayConst8(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::CONST16:
        deserializeIntArrayConst16(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::CONST32:
        deserializeIntArrayConst32(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::CONST64:
        deserializeIntArrayConst64(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::ZERO:
        deserializeIntArrayZero(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::GET_INDEX:
        deserializeIntArrayGetIndex(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::SET_INDEX:
        deserializeIntArraySetIndex(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::LENGTH:
        deserializeIntArrayLength(builder, operation, ctx);
        break;
    case jeff::IntArrayOp::CREATE:
        deserializeIntArrayCreate(builder, operation, ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize int array instruction " << intArray.which() << "\n";
        llvm::report_fatal_error("Unknown int array instruction");
    }
}

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

#define DESERIALIZE_FLOAT_CONST(BIT_WIDTH)                                                         \
    void deserializeFloatConst##BIT_WIDTH(mlir::ImplicitLocOpBuilder& builder,                     \
                                          const jeff::Op::Reader& operation,                       \
                                          DeserializationContext& ctx) {                           \
        const auto value = operation.getInstruction().getFloat().getConst##BIT_WIDTH();            \
        auto floatType = builder.getF##BIT_WIDTH##Type();                                          \
        auto floatAttr = mlir::FloatAttr::get(floatType, value);                                   \
        auto op = mlir::jeff::FloatConst##BIT_WIDTH##Op::create(builder, floatType, floatAttr);    \
        ctx.setValue(operation.getOutputs()[0], op.getConstant());                                 \
    }

DESERIALIZE_FLOAT_CONST(32)
DESERIALIZE_FLOAT_CONST(64)

#undef DESERIALIZE_FLOAT_CONST

void deserializeFloatUnaryOp(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                             mlir::jeff::FloatUnaryOperation unaryOperation,
                             DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::FloatUnaryOp::create(builder, ctx.getValue(inputs[0]), unaryOperation);
    ctx.setValue(outputs[0], op.getB());
}

void deserializeFloatBinaryOp(mlir::ImplicitLocOpBuilder& builder,
                              const jeff::Op::Reader& operation,
                              mlir::jeff::FloatBinaryOperation binaryOperation,
                              DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::FloatBinaryOp::create(builder, ctx.getValue(inputs[0]),
                                                ctx.getValue(inputs[1]), binaryOperation);
    ctx.setValue(outputs[0], op.getC());
}

void deserializeFloatComparisonOp(mlir::ImplicitLocOpBuilder& builder,
                                  const jeff::Op::Reader& operation,
                                  mlir::jeff::FloatComparisonOperation comparisonOperation,
                                  DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::FloatComparisonOp::create(builder, ctx.getValue(inputs[0]),
                                                    ctx.getValue(inputs[1]), comparisonOperation);
    ctx.setValue(outputs[0], op.getC());
}

void deserializeFloatIsOp(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                          mlir::jeff::FloatIsOperation isOperation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::FloatIsOp::create(builder, ctx.getValue(inputs[0]), isOperation);
    ctx.setValue(outputs[0], op.getResult());
}

#define ADD_CONST_CASE(BIT_WIDTH)                                                                  \
    case jeff::FloatOp::CONST##BIT_WIDTH:                                                          \
        deserializeFloatConst##BIT_WIDTH(builder, operation, ctx);                                 \
        break;

#define ADD_UNARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                          \
    case jeff::FloatOp::JEFF_ENUM_VALUE:                                                           \
        deserializeFloatUnaryOp(builder, operation,                                                \
                                mlir::jeff::FloatUnaryOperation::_##MLIR_ENUM_SUFFIX, ctx);        \
        break;

#define ADD_BINARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                         \
    case jeff::FloatOp::JEFF_ENUM_VALUE:                                                           \
        deserializeFloatBinaryOp(builder, operation,                                               \
                                 mlir::jeff::FloatBinaryOperation::_##MLIR_ENUM_SUFFIX, ctx);      \
        break;

#define ADD_COMPARISON_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                     \
    case jeff::FloatOp::JEFF_ENUM_VALUE:                                                           \
        deserializeFloatComparisonOp(                                                              \
            builder, operation, mlir::jeff::FloatComparisonOperation::_##MLIR_ENUM_SUFFIX, ctx);   \
        break;

#define ADD_IS_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                                             \
    case jeff::FloatOp::JEFF_ENUM_VALUE:                                                           \
        deserializeFloatIsOp(builder, operation,                                                   \
                             mlir::jeff::FloatIsOperation::_is##MLIR_ENUM_SUFFIX, ctx);            \
        break;

void deserializeFloat(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                      DeserializationContext& ctx) {
    switch (const auto floatInstr = operation.getInstruction().getFloat(); floatInstr.which()) {
        ADD_CONST_CASE(32)
        ADD_CONST_CASE(64)
        ADD_UNARY_CASE(SQRT, sqrt)
        ADD_UNARY_CASE(ABS, abs)
        ADD_UNARY_CASE(CEIL, ceil)
        ADD_UNARY_CASE(FLOOR, floor)
        ADD_UNARY_CASE(EXP, exp)
        ADD_UNARY_CASE(LOG, log)
        ADD_UNARY_CASE(SIN, sin)
        ADD_UNARY_CASE(COS, cos)
        ADD_UNARY_CASE(TAN, tan)
        ADD_UNARY_CASE(ASIN, asin)
        ADD_UNARY_CASE(ACOS, acos)
        ADD_UNARY_CASE(ATAN, atan)
        ADD_UNARY_CASE(SINH, sinh)
        ADD_UNARY_CASE(COSH, cosh)
        ADD_UNARY_CASE(TANH, tanh)
        ADD_UNARY_CASE(ASINH, asinh)
        ADD_UNARY_CASE(ACOSH, acosh)
        ADD_UNARY_CASE(ATANH, atanh)
        ADD_BINARY_CASE(ADD, add)
        ADD_BINARY_CASE(SUB, sub)
        ADD_BINARY_CASE(MUL, mul)
        ADD_BINARY_CASE(POW, pow)
        ADD_BINARY_CASE(ATAN2, atan2)
        ADD_BINARY_CASE(MAX, max)
        ADD_BINARY_CASE(MIN, min)
        ADD_COMPARISON_CASE(EQ, eq)
        ADD_COMPARISON_CASE(LT, lt)
        ADD_COMPARISON_CASE(LTE, lte)
        ADD_IS_CASE(IS_NAN, Nan)
        ADD_IS_CASE(IS_INF, Inf)
    default:
        llvm::errs() << "Cannot deserialize float instruction " << floatInstr.which() << "\n";
        llvm::report_fatal_error("Unknown float instruction");
    }
}

#undef ADD_CONST_CASE
#undef ADD_UNARY_CASE
#undef ADD_BINARY_CASE
#undef ADD_COMPARISON_CASE
#undef ADD_IS_CASE

//===----------------------------------------------------------------------===//
// FloatArray operations
//===----------------------------------------------------------------------===//

void deserializeFloatArrayConst32(mlir::ImplicitLocOpBuilder& builder,
                                  const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto outputs = operation.getOutputs();
    const auto values = operation.getInstruction().getFloatArray().getConst32();
    llvm::SmallVector<float> inArray;
    inArray.reserve(values.size());
    for (auto value : values) {
        inArray.push_back(static_cast<float>(value));
    }
    auto inArrayAttr = mlir::DenseF32ArrayAttr::get(builder.getContext(), inArray);
    auto tensorType =
        mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, builder.getF32Type());
    auto op = mlir::jeff::FloatArrayConst32Op::create(builder, tensorType, inArrayAttr);
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeFloatArrayConst64(mlir::ImplicitLocOpBuilder& builder,
                                  const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto outputs = operation.getOutputs();
    const auto values = operation.getInstruction().getFloatArray().getConst64();
    llvm::SmallVector<double> inArray;
    inArray.reserve(values.size());
    for (auto value : values) {
        inArray.push_back(static_cast<double>(value));
    }
    auto inArrayAttr = mlir::DenseF64ArrayAttr::get(builder.getContext(), inArray);
    auto tensorType =
        mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, builder.getF64Type());
    auto op = mlir::jeff::FloatArrayConst64Op::create(builder, tensorType, inArrayAttr);
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeFloatArrayZero(mlir::ImplicitLocOpBuilder& builder,
                               const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto zero = operation.getInstruction().getFloatArray().getZero();
    mlir::Type floatType;
    switch (zero) {
    case jeff::FloatPrecision::FLOAT32:
        floatType = builder.getF32Type();
        break;
    case jeff::FloatPrecision::FLOAT64:
        floatType = builder.getF64Type();
        break;
    default:
        llvm::report_fatal_error("Invalid bit width");
    }
    auto tensorType = mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, floatType);
    auto op = mlir::jeff::FloatArrayZeroOp::create(builder, tensorType, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeFloatArrayGetIndex(mlir::ImplicitLocOpBuilder& builder,
                                   const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto tensorType = ctx.getValue(inputs[0]).getType();
    auto entryType = llvm::cast<mlir::RankedTensorType>(tensorType).getElementType();
    auto op = mlir::jeff::FloatArrayGetIndexOp::create(builder, entryType, ctx.getValue(inputs[0]),
                                                       ctx.getValue(inputs[1]));
    ctx.setValue(outputs[0], op.getValue());
}

void deserializeFloatArraySetIndex(mlir::ImplicitLocOpBuilder& builder,
                                   const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto tensorType = ctx.getValue(inputs[0]).getType();
    auto op =
        mlir::jeff::FloatArraySetIndexOp::create(builder, tensorType, ctx.getValue(inputs[0]),
                                                 ctx.getValue(inputs[1]), ctx.getValue(inputs[2]));
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeFloatArrayLength(mlir::ImplicitLocOpBuilder& builder,
                                 const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    auto op = mlir::jeff::FloatArrayLengthOp::create(builder, ctx.getValue(inputs[0]));
    ctx.setValue(outputs[0], op.getLength());
}

void deserializeFloatArrayCreate(mlir::ImplicitLocOpBuilder& builder,
                                 const jeff::Op::Reader& operation, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    llvm::SmallVector<mlir::Value> inArray;
    inArray.reserve(inputs.size());
    for (auto input : inputs) {
        inArray.push_back(ctx.getValue(input));
    }
    auto tensorType =
        mlir::RankedTensorType::get({ctx.getLength(outputs[0])}, ctx.getValue(inputs[0]).getType());
    auto op = mlir::jeff::FloatArrayCreateOp::create(builder, tensorType, inArray);
    ctx.setValue(outputs[0], op.getOutArray());
}

void deserializeFloatArray(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                           DeserializationContext& ctx) {
    switch (const auto floatArray = operation.getInstruction().getFloatArray();
            floatArray.which()) {
    case jeff::FloatArrayOp::CONST32:
        deserializeFloatArrayConst32(builder, operation, ctx);
        break;
    case jeff::FloatArrayOp::CONST64:
        deserializeFloatArrayConst64(builder, operation, ctx);
        break;
    case jeff::FloatArrayOp::ZERO:
        deserializeFloatArrayZero(builder, operation, ctx);
        break;
    case jeff::FloatArrayOp::GET_INDEX:
        deserializeFloatArrayGetIndex(builder, operation, ctx);
        break;
    case jeff::FloatArrayOp::SET_INDEX:
        deserializeFloatArraySetIndex(builder, operation, ctx);
        break;
    case jeff::FloatArrayOp::LENGTH:
        deserializeFloatArrayLength(builder, operation, ctx);
        break;
    case jeff::FloatArrayOp::CREATE:
        deserializeFloatArrayCreate(builder, operation, ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize float array instruction " << floatArray.which() << "\n";
        llvm::report_fatal_error("Unknown float array instruction");
    }
}

//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

// Forward declaration
void deserializeOperations(mlir::ImplicitLocOpBuilder& builder,
                           const capnp::List<jeff::Op>::Reader& operations,
                           DeserializationContext& ctx);

void deserializeBlock(mlir::ImplicitLocOpBuilder& builder, mlir::Block& block,
                      mlir::TypeRange argTypes, const jeff::Region::Reader& region,
                      DeserializationContext& ctx) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    // Add sources to map
    for (size_t i = 0; i < region.getSources().size(); ++i) {
        auto arg = block.addArgument(argTypes[i], builder.getUnknownLoc());
        ctx.setValue(region.getSources()[i], arg);
    }

    deserializeOperations(builder, region.getOperations(), ctx);

    // Retrieve targets from map
    llvm::SmallVector<mlir::Value> targetValues;
    targetValues.reserve(region.getTargets().size());
    for (size_t i = 0; i < region.getTargets().size(); ++i) {
        targetValues.push_back(ctx.getValue(region.getTargets()[i]));
    }

    mlir::jeff::YieldOp::create(builder, targetValues);
}

void deserializeSwitch(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                       DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto switchInstr = operation.getInstruction().getScf().getSwitch();
    const auto branches = switchInstr.getBranches();

    llvm::SmallVector<mlir::Value> inValues;
    llvm::SmallVector<mlir::Type> outTypes;
    inValues.reserve(inputs.size() - 1);
    outTypes.reserve(inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); ++i) {
        auto value = ctx.getValue(inputs[i]);
        inValues.push_back(value);
        outTypes.push_back(value.getType());
    }

    auto op = mlir::jeff::SwitchOp::create(builder, outTypes, ctx.getValue(inputs[0]), inValues,
                                           branches.size());

    for (size_t i = 0; i < branches.size(); ++i) {
        deserializeBlock(builder, op.getBranches()[i].emplaceBlock(), outTypes, branches[i], ctx);
    }

    if (switchInstr.hasDefault()) {
        deserializeBlock(builder, op.getDefault().emplaceBlock(), outTypes,
                         switchInstr.getDefault(), ctx);
    }

    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(operation.getOutputs().size());
    for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
        ctx.setValue(operation.getOutputs()[i], op.getResults()[i]);
    }
}

void deserializeFor(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                    DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto forInstr = operation.getInstruction().getScf().getFor();

    llvm::SmallVector<mlir::Value> inValues;
    llvm::SmallVector<mlir::Type> outTypes;
    inValues.reserve(inputs.size() - 3);
    outTypes.reserve(inputs.size() - 3);
    for (size_t i = 3; i < inputs.size(); ++i) {
        auto value = ctx.getValue(inputs[i]);
        inValues.push_back(value);
        outTypes.push_back(value.getType());
    }

    auto op = mlir::jeff::ForOp::create(builder, outTypes, ctx.getValue(inputs[0]),
                                        ctx.getValue(inputs[1]), ctx.getValue(inputs[2]), inValues);

    llvm::SmallVector<mlir::Type> argTypes;
    argTypes.reserve(inputs.size() - 2);
    argTypes.push_back(ctx.getValue(inputs[0]).getType());
    argTypes.append(outTypes.begin(), outTypes.end());
    deserializeBlock(builder, op.getBody().emplaceBlock(), argTypes, forInstr, ctx);

    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(operation.getOutputs().size());
    for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
        ctx.setValue(operation.getOutputs()[i], op.getResults()[i]);
    }
}

void deserializeWhile(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                      jeff::ScfOp::While::Reader reader, DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();

    llvm::SmallVector<mlir::Value> inValues;
    llvm::SmallVector<mlir::Type> outTypes;
    inValues.reserve(inputs.size());
    outTypes.reserve(inputs.size());
    for (const auto input : inputs) {
        auto value = ctx.getValue(input);
        inValues.push_back(value);
        outTypes.push_back(value.getType());
    }

    auto op = mlir::jeff::WhileOp::create(builder, outTypes, inValues);

    deserializeBlock(builder, op.getBefore().emplaceBlock(), outTypes, reader.getBefore(), ctx);
    deserializeBlock(builder, op.getAfter().emplaceBlock(), outTypes, reader.getAfter(), ctx);

    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(operation.getOutputs().size());
    for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
        ctx.setValue(operation.getOutputs()[i], op.getResults()[i]);
    }
}

void deserializeScf(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                    DeserializationContext& ctx) {
    switch (const auto scf = operation.getInstruction().getScf(); scf.which()) {
    case jeff::ScfOp::SWITCH:
        deserializeSwitch(builder, operation, ctx);
        break;
    case jeff::ScfOp::FOR:
        deserializeFor(builder, operation, ctx);
        break;
    case jeff::ScfOp::WHILE:
        deserializeWhile(builder, operation, scf.getWhile(), ctx);
        break;
    default:
        llvm::errs() << "Cannot deserialize scf instruction " << scf.which() << "\n";
        llvm::report_fatal_error("Unknown scf instruction");
    }
}

//===----------------------------------------------------------------------===//
// Func operations
//===----------------------------------------------------------------------===//

void deserializeFunc(mlir::ImplicitLocOpBuilder& builder, const jeff::Op::Reader& operation,
                     DeserializationContext& ctx) {
    const auto inputs = operation.getInputs();
    const auto outputs = operation.getOutputs();
    const auto func = operation.getInstruction().getFunc();
    auto mlirFunc = ctx.getFunc(func.getFuncCall());
    llvm::SmallVector<mlir::Value> mlirInputs;
    mlirInputs.reserve(inputs.size());
    for (const auto input : inputs) {
        mlirInputs.push_back(ctx.getValue(input));
    }
    auto op = mlir::func::CallOp::create(builder, mlirFunc, mlirInputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
        ctx.setValue(outputs[i], op.getResult(i));
    }
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

mlir::Type deserializeQuregType(const mlir::ImplicitLocOpBuilder& builder,
                                const jeff::Type::Reader& type) {
    const auto quregType = type.getQureg();
    auto length = mlir::ShapedType::kDynamic;
    if (quregType.isStatic()) {
        length = quregType.getStatic();
    }
    return mlir::jeff::QuregType::get(builder.getContext(), length);
}

mlir::Type deserializeIntType(mlir::ImplicitLocOpBuilder& builder, const jeff::Type::Reader& type) {
    switch (type.getInt()) {
    case 1:
        return builder.getI1Type();
    case 8:
        return builder.getI8Type();
    case 16:
        return builder.getI16Type();
    case 32:
        return builder.getI32Type();
    case 64:
        return builder.getI64Type();
    default:
        llvm::errs() << "Cannot deserialize int type " << static_cast<int>(type.getInt()) << "\n";
        llvm::report_fatal_error("Unknown int type");
    }
}

mlir::Type deserializeIntArrayType(mlir::ImplicitLocOpBuilder& builder,
                                   const jeff::Type::Reader& type) {
    const auto intArrayType = type.getIntArray();
    auto length = mlir::ShapedType::kDynamic;
    if (intArrayType.getLength().isStatic()) {
        length = intArrayType.getLength().getStatic();
    }
    switch (intArrayType.getBitwidth()) {
    case 1:
        return mlir::RankedTensorType::get({length}, builder.getI1Type());
    case 8:
        return mlir::RankedTensorType::get({length}, builder.getI8Type());
    case 16:
        return mlir::RankedTensorType::get({length}, builder.getI16Type());
    case 32:
        return mlir::RankedTensorType::get({length}, builder.getI32Type());
    case 64:
        return mlir::RankedTensorType::get({length}, builder.getI64Type());
    default:
        llvm::errs() << "Cannot deserialize int array type with bit width "
                     << static_cast<int>(intArrayType.getBitwidth()) << "\n";
        llvm::report_fatal_error("Unknown int array type");
    }
}

mlir::FloatType deserializeFloatType(mlir::ImplicitLocOpBuilder& builder,
                                     const jeff::Type::Reader& type) {
    switch (type.getFloat()) {
    case jeff::FloatPrecision::FLOAT32:
        return builder.getF32Type();
    case jeff::FloatPrecision::FLOAT64:
        return builder.getF64Type();
    default:
        llvm::errs() << "Cannot deserialize float type " << static_cast<int>(type.getFloat())
                     << "\n";
        llvm::report_fatal_error("Unknown float type");
    }
}

mlir::Type deserializeFloatArrayType(mlir::ImplicitLocOpBuilder& builder,
                                     const jeff::Type::Reader& type) {
    const auto floatArrayType = type.getFloatArray();
    auto length = mlir::ShapedType::kDynamic;
    if (floatArrayType.getLength().isStatic()) {
        length = floatArrayType.getLength().getStatic();
    }
    switch (floatArrayType.getPrecision()) {
    case jeff::FloatPrecision::FLOAT32:
        return mlir::RankedTensorType::get({length}, builder.getF32Type());
    case jeff::FloatPrecision::FLOAT64:
        return mlir::RankedTensorType::get({length}, builder.getF64Type());
    default:
        llvm::errs() << "Cannot deserialize float array type with precision "
                     << static_cast<int>(floatArrayType.getPrecision()) << "\n";
        llvm::report_fatal_error("Unknown float array type");
    }
}

mlir::Type deserializeType(mlir::ImplicitLocOpBuilder& builder, const jeff::Type::Reader& type) {
    switch (type.which()) {
    case jeff::Type::QUBIT:
        return mlir::jeff::QubitType::get(builder.getContext());
    case jeff::Type::QUREG:
        return deserializeQuregType(builder, type);
    case jeff::Type::INT:
        return deserializeIntType(builder, type);
    case jeff::Type::INT_ARRAY:
        return deserializeIntArrayType(builder, type);
    case jeff::Type::FLOAT:
        return deserializeFloatType(builder, type);
    case jeff::Type::FLOAT_ARRAY:
        return deserializeFloatArrayType(builder, type);
    default:
        llvm::errs() << "Cannot deserialize type " << type.which() << "\n";
        llvm::report_fatal_error("Unknown type");
    }
}

auto deserializeOperations(mlir::ImplicitLocOpBuilder& builder,
                           const capnp::List<jeff::Op>::Reader& operations,
                           DeserializationContext& ctx) -> void {
    for (auto operation : operations) {
        switch (const auto instruction = operation.getInstruction(); instruction.which()) {
        case jeff::Op::Instruction::QUBIT:
            deserializeQubit(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::QUREG:
            deserializeQureg(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::INT:
            deserializeInt(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::INT_ARRAY:
            deserializeIntArray(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::FLOAT:
            deserializeFloat(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::FLOAT_ARRAY:
            deserializeFloatArray(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::SCF:
            deserializeScf(builder, operation, ctx);
            break;
        case jeff::Op::Instruction::FUNC:
            deserializeFunc(builder, operation, ctx);
            break;
        default:
            llvm::errs() << "Cannot deserialize instruction " << instruction.which() << "\n";
            llvm::report_fatal_error("Unknown instruction");
        }
    }
}

void deserializeFunctionSignature(mlir::ImplicitLocOpBuilder& builder,
                                  const jeff::Function::Reader& function, uint16_t functionId,
                                  DeserializationContext& ctx) {
    // Get function definition
    const auto definition = function.getDefinition();

    // Get values
    const auto jeffValues = definition.getValues();

    // Get function body
    if (!definition.hasBody()) {
        llvm::report_fatal_error("Function definition has no body");
    }
    const auto body = definition.getBody();

    // Get sources
    const auto sources = body.getSources();

    // Get source types
    llvm::SmallVector<mlir::Type> sourceTypes;
    sourceTypes.reserve(sources.size());
    for (const auto source : sources) {
        const auto jeffType = jeffValues[source].getType();
        sourceTypes.push_back(deserializeType(builder, jeffType));
    }

    // Get targets
    const auto targets = body.getTargets();

    // Get target types
    llvm::SmallVector<mlir::Type> targetTypes;
    targetTypes.reserve(targets.size());
    for (auto target : targets) {
        const auto jeffType = jeffValues[target].getType();
        targetTypes.push_back(deserializeType(builder, jeffType));
    }

    // Create function
    const auto funcName = ctx.strings[function.getName()];
    auto funcType = builder.getFunctionType(sourceTypes, targetTypes);
    auto func = mlir::func::FuncOp::create(builder, funcName, funcType);
    ctx.setFunc(functionId, func);
}

void deserializeFunctionBody(mlir::ImplicitLocOpBuilder& builder,
                             const jeff::Function::Reader& function, mlir::func::FuncOp func,
                             DeserializationContext& ctx) {
    ctx.values.clear();
    ctx.jeffValues = function.getDefinition().getValues();
    ctx.values.reserve(ctx.jeffValues.size());

    const auto body = function.getDefinition().getBody();
    if (!body.hasOperations()) {
        llvm::report_fatal_error("Function body has no operations");
    }
    const auto operations = body.getOperations();
    const auto sources = body.getSources();

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    for (auto i = 0; i < sources.size(); ++i) {
        ctx.setValue(sources[i], entryBlock.getArgument(i));
    }

    deserializeOperations(builder, operations, ctx);

    llvm::SmallVector<mlir::Value> results;
    results.reserve(body.getTargets().size());
    for (const auto target : body.getTargets()) {
        results.push_back(ctx.getValue(target));
    }

    // Create return statement
    mlir::func::ReturnOp::create(builder, results);
}

mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              const jeff::Module::Reader& jeffModule) {
    DeserializationContext ctx;

    // Create MLIR builder
    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(context), context);

    // Create MLIR module
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto mlirModule = mlir::ModuleOp::create(builder);
    builder.setInsertionPointToStart(mlirModule.getBody());

    // Get strings
    auto strings = jeffModule.getStrings();
    ctx.strings.reserve(strings.size());
    for (auto string : strings) {
        ctx.strings.push_back(std::string(string.cStr()));
    }

    // Get functions
    if (!jeffModule.hasFunctions()) {
        llvm::report_fatal_error("No functions found in module");
    }
    const auto functions = jeffModule.getFunctions();
    ctx.funcs.reserve(functions.size());

    // Register every signature before reading bodies so calls can reference later functions.
    uint16_t functionId = 0;
    for (const auto function : functions) {
        deserializeFunctionSignature(builder, function, functionId++, ctx);
    }

    functionId = 0;
    for (const auto function : functions) {
        deserializeFunctionBody(builder, function, ctx.getFunc(functionId++), ctx);
    }

    // Set metadata
    auto uint16Type = builder.getIntegerType(16, false);

    const auto entryPoint = jeffModule.getEntrypoint();
    mlirModule->setAttr("jeff.entrypoint", builder.getIntegerAttr(uint16Type, entryPoint));

    llvm::SmallVector<llvm::StringRef> stringRefs;
    stringRefs.reserve(ctx.strings.size());
    for (const auto& str : ctx.strings) {
        stringRefs.push_back(str);
    }
    mlirModule->setAttr("jeff.strings", builder.getStrArrayAttr(stringRefs));

    const auto tool = std::string_view(jeffModule.getTool().cStr());
    mlirModule->setAttr("jeff.tool", builder.getStringAttr(tool));

    const auto toolVersion = std::string_view(jeffModule.getToolVersion().cStr());
    mlirModule->setAttr("jeff.toolVersion", builder.getStringAttr(toolVersion));

    const auto jeffVersion = jeffModule.getVersion();
    mlirModule->setAttr("jeff.version", builder.getIntegerAttr(uint16Type, jeffVersion));

    const auto jeffVersionMinor = jeffModule.getVersionMinor();
    mlirModule->setAttr("jeff.versionMinor", builder.getIntegerAttr(uint16Type, jeffVersionMinor));

    const auto jeffVersionPatch = jeffModule.getVersionPatch();
    mlirModule->setAttr("jeff.versionPatch", builder.getIntegerAttr(uint16Type, jeffVersionPatch));

    if (mlir::verify(mlirModule).failed()) {
        llvm::errs() << "Verification of MLIR module failed\n";
        mlirModule->print(llvm::errs());
        llvm::report_fatal_error("Verification of MLIR module failed");
    }

    return mlirModule;
}
} // namespace

mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              kj::ArrayPtr<capnp::word> buffer) {
    DeserializationContext ctx;

    capnp::FlatArrayMessageReader message(buffer);
    return deserialize(context, message.getRoot<jeff::Module>());
}

mlir::OwningOpRef<mlir::ModuleOp> deserializeFromFile(mlir::MLIRContext* context,
                                                      llvm::StringRef path) {
    // The buffer is read as an array of Cap'n Proto words, so request a buffer
    // that is aligned accordingly.
    auto file = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/true, /*IsVolatile=*/false,
                                            llvm::Align(alignof(capnp::word)));
    if (!file) {
        llvm::errs() << "Failed to open " << path << ": " << file.getError().message() << "\n";
        return {};
    }

    // Get jeff module from buffer
    const auto bytes = (*file)->getBuffer();
    if (bytes.size() % sizeof(capnp::word) != 0) {
        llvm::errs() << "Failed to read " << path
                     << ": size must be a multiple of the Cap'n Proto word size\n";
        return {};
    }

    assert(reinterpret_cast<uintptr_t>(bytes.data()) % alignof(capnp::word) == 0 &&
           "Serialized module buffer must be aligned to capnp::word alignment");
    const auto words = kj::ArrayPtr(reinterpret_cast<const capnp::word*>(bytes.data()),
                                    bytes.size() / sizeof(capnp::word));

    capnp::FlatArrayMessageReader message(words);
    const jeff::Module::Reader jeffModule = message.getRoot<jeff::Module>();
    return deserialize(context, jeffModule);
}
