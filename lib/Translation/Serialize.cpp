#include "jeff/Translation/Serialize.hpp"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffInterfaces.h"
#include "jeff/IR/JeffOps.h"

#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <jeff.capnp.h>
#include <kj/array.h>
#include <kj/io.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>

static void checkRank(mlir::RankedTensorType tensorType) {
    if (tensorType.getRank() != 1) {
        llvm::report_fatal_error("Only one-dimensional tensors are supported");
    }
}

namespace {

struct SerializationContext {
    llvm::DenseMap<mlir::Value, uint32_t> values;
    llvm::StringMap<uint16_t> funcs;
    llvm::StringMap<uint16_t> strings;

    uint32_t getValueId(mlir::Value value) {
        auto it = values.find(value);
        if (it != values.end()) {
            return it->second;
        }
        auto id = values.size();
        values[value] = id;
        return id;
    }

    uint16_t getFuncId(llvm::StringRef funcName) {
        auto it = funcs.find(funcName);
        if (it == funcs.end()) {
            llvm::errs() << "Function " << funcName << " not found\n";
            llvm::report_fatal_error("Function not found");
        }
        return it->second;
    }

    uint16_t getStringId(llvm::StringRef str) {
        auto it = strings.find(str);
        if (it == strings.end()) {
            llvm::errs() << "String " << str << " not found\n";
            llvm::report_fatal_error("String not found");
        }
        return it->second;
    }
};

//===----------------------------------------------------------------------===//
// Qubit operations
//===----------------------------------------------------------------------===//

void serializeQubitAlloc(jeff::Op::Builder builder, mlir::jeff::QubitAllocOp op,
                         SerializationContext& ctx) {
    auto qubitBuilder = builder.initInstruction().initQubit();
    qubitBuilder.setAlloc();

    builder.initInputs(0);

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeQubitFree(jeff::Op::Builder builder, mlir::jeff::QubitFreeOp op,
                        SerializationContext& ctx) {
    auto qubitBuilder = builder.initInstruction().initQubit();
    qubitBuilder.setFree();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInQubit()));

    builder.initOutputs(0);
}

void serializeQubitFreeZero(jeff::Op::Builder builder, mlir::jeff::QubitFreeZeroOp op,
                            SerializationContext& ctx) {
    auto qubitBuilder = builder.initInstruction().initQubit();
    qubitBuilder.setFreeZero();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInQubit()));

    builder.initOutputs(0);
}

void serializeMeasure(jeff::Op::Builder builder, mlir::jeff::QubitMeasureOp op,
                      SerializationContext& ctx) {
    auto qubitBuilder = builder.initInstruction().initQubit();
    qubitBuilder.setMeasure();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInQubit()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeMeasureNd(jeff::Op::Builder builder, mlir::jeff::QubitMeasureNDOp op,
                        SerializationContext& ctx) {
    auto qubitBuilder = builder.initInstruction().initQubit();
    qubitBuilder.setMeasureNd();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInQubit()));

    auto outputs = builder.initOutputs(2);
    outputs.set(0, ctx.getValueId(op.getOutQubit()));
    outputs.set(1, ctx.getValueId(op.getResult()));
}

void serializeReset(jeff::Op::Builder builder, mlir::jeff::QubitResetOp op,
                    SerializationContext& ctx) {
    auto qubitBuilder = builder.initInstruction().initQubit();
    qubitBuilder.setReset();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInQubit()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutQubit()));
}

template <typename OpType>
void serializeOneTargetZeroParameter(jeff::Op::Builder builder, OpType op, jeff::WellKnownGate gate,
                                     SerializationContext& ctx) {
    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setWellKnown(gate);
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());

    auto numControls = op.getNumCtrls();
    auto inputs = builder.initInputs(1 + numControls);
    inputs.set(0, ctx.getValueId(op.getInQubit()));
    for (uint8_t i = 0; i < numControls; ++i) {
        inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
    }

    auto outputs = builder.initOutputs(1 + numControls);
    outputs.set(0, ctx.getValueId(op.getOutQubit()));
    for (uint8_t i = 0; i < numControls; ++i) {
        outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
    }
}

template <typename OpType>
void serializeOneTargetOneParameter(jeff::Op::Builder builder, OpType op, jeff::WellKnownGate gate,
                                    SerializationContext& ctx) {
    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setWellKnown(gate);
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());

    auto numControls = op.getNumCtrls();
    auto inputs = builder.initInputs(1 + numControls + 1);
    inputs.set(0, ctx.getValueId(op.getInQubit()));
    for (uint8_t i = 0; i < numControls; ++i) {
        inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
    }
    inputs.set(1 + numControls, ctx.getValueId(op.getRotation()));

    auto outputs = builder.initOutputs(1 + numControls);
    outputs.set(0, ctx.getValueId(op.getOutQubit()));
    for (uint8_t i = 0; i < numControls; ++i) {
        outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
    }
}

void serializeU(jeff::Op::Builder builder, mlir::jeff::UOp op, SerializationContext& ctx) {

    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setWellKnown(jeff::WellKnownGate::U);
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());

    auto numControls = op.getNumCtrls();
    auto inputs = builder.initInputs(1 + numControls + 3);
    inputs.set(0, ctx.getValueId(op.getInQubit()));
    for (uint8_t i = 0; i < numControls; ++i) {
        inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
    }
    inputs.set(1 + numControls, ctx.getValueId(op.getTheta()));
    inputs.set(2 + numControls, ctx.getValueId(op.getPhi()));
    inputs.set(3 + numControls, ctx.getValueId(op.getLambda()));

    auto outputs = builder.initOutputs(1 + numControls);
    outputs.set(0, ctx.getValueId(op.getOutQubit()));
    for (uint8_t i = 0; i < numControls; ++i) {
        outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
    }
}

void serializeSwap(jeff::Op::Builder builder, mlir::jeff::SwapOp op, SerializationContext& ctx) {
    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setWellKnown(jeff::WellKnownGate::SWAP);
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());

    auto numControls = op.getNumCtrls();
    auto inputs = builder.initInputs(2 + numControls);
    inputs.set(0, ctx.getValueId(op.getInQubitOne()));
    inputs.set(1, ctx.getValueId(op.getInQubitTwo()));
    for (uint8_t i = 0; i < numControls; ++i) {
        inputs.set(2 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
    }

    auto outputs = builder.initOutputs(2 + numControls);
    outputs.set(0, ctx.getValueId(op.getOutQubitOne()));
    outputs.set(1, ctx.getValueId(op.getOutQubitTwo()));
    for (uint8_t i = 0; i < numControls; ++i) {
        outputs.set(2 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
    }
}

void serializeGPhase(jeff::Op::Builder builder, mlir::jeff::GPhaseOp op,
                     SerializationContext& ctx) {
    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setWellKnown(jeff::WellKnownGate::GPHASE);
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());

    auto numControls = op.getNumCtrls();
    auto inputs = builder.initInputs(numControls + 1);
    for (uint8_t i = 0; i < numControls; ++i) {
        inputs.set(i, ctx.getValueId(op.getInCtrlQubits()[i]));
    }
    inputs.set(numControls, ctx.getValueId(op.getRotation()));

    auto outputs = builder.initOutputs(numControls);
    for (uint8_t i = 0; i < numControls; ++i) {
        outputs.set(i, ctx.getValueId(op.getOutCtrlQubits()[i]));
    }
}

void serializeWellKnownGate(jeff::Op::Builder builder, mlir::jeff::WellKnownGate operation,
                            SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::XOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::X, ctx);
        })
        .Case<mlir::jeff::YOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::Y, ctx);
        })
        .Case<mlir::jeff::ZOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::Z, ctx);
        })
        .Case<mlir::jeff::SOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::S, ctx);
        })
        .Case<mlir::jeff::TOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::T, ctx);
        })
        .Case<mlir::jeff::R1Op>([&](auto op) {
            serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::R1, ctx);
        })
        .Case<mlir::jeff::RxOp>([&](auto op) {
            serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::RX, ctx);
        })
        .Case<mlir::jeff::RyOp>([&](auto op) {
            serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::RY, ctx);
        })
        .Case<mlir::jeff::RzOp>([&](auto op) {
            serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::RZ, ctx);
        })
        .Case<mlir::jeff::HOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::H, ctx);
        })
        .Case<mlir::jeff::UOp>([&](auto op) { serializeU(builder, op, ctx); })
        .Case<mlir::jeff::SwapOp>([&](auto op) { serializeSwap(builder, op, ctx); })
        .Case<mlir::jeff::IOp>([&](auto op) {
            serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::I, ctx);
        })
        .Case<mlir::jeff::GPhaseOp>([&](auto op) { serializeGPhase(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize well-known gate " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown well-known gate");
        });
}

void serializeCustom(jeff::Op::Builder builder, mlir::jeff::CustomOp op,
                     SerializationContext& ctx) {
    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());
    auto customBuilder = gateBuilder.initCustom();
    customBuilder.setName(ctx.getStringId(op.getName().str()));
    customBuilder.setNumQubits(op.getNumTargets());
    customBuilder.setNumParams(op.getNumParams());

    auto inputs = builder.initInputs(op.getNumOperands());
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
        inputs.set(i, ctx.getValueId(op.getOperand(i)));
    }

    auto outputs = builder.initOutputs(op.getNumResults());
    for (size_t i = 0; i < op.getNumResults(); ++i) {
        outputs.set(i, ctx.getValueId(op.getResult(i)));
    }
}

void serializePpr(jeff::Op::Builder builder, mlir::jeff::PPROp op, SerializationContext& ctx) {
    auto gateBuilder = builder.initInstruction().initQubit().initGate();
    gateBuilder.setControlQubits(op.getNumCtrls());
    gateBuilder.setAdjoint(op.getIsAdjoint());
    gateBuilder.setPower(op.getPower());
    auto pprBuilder = gateBuilder.initPpr();

    auto pauliString = op.getPauliGates();
    capnp::List<jeff::Pauli>::Builder pauliStringBuilder =
        pprBuilder.initPauliString(pauliString.size());
    for (size_t i = 0; i < pauliString.size(); ++i) {
        switch (pauliString[i]) {
        case 0:
            pauliStringBuilder.set(i, jeff::Pauli::I);
            break;
        case 1:
            pauliStringBuilder.set(i, jeff::Pauli::X);
            break;
        case 2:
            pauliStringBuilder.set(i, jeff::Pauli::Y);
            break;
        case 3:
            pauliStringBuilder.set(i, jeff::Pauli::Z);
            break;
        default:
            llvm::report_fatal_error("Unknown Pauli gate");
        }
    }

    auto inputs = builder.initInputs(op.getNumOperands());
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
        inputs.set(i, ctx.getValueId(op.getOperand(i)));
    }

    auto outputs = builder.initOutputs(op.getNumResults());
    for (size_t i = 0; i < op.getNumResults(); ++i) {
        outputs.set(i, ctx.getValueId(op.getResult(i)));
    }
}

void serializeGate(jeff::Op::Builder builder, mlir::jeff::MultipleCtrlQubitsGate operation,
                   SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::WellKnownGate>([&](auto op) { serializeWellKnownGate(builder, op, ctx); })
        .Case<mlir::jeff::CustomOp>([&](auto op) { serializeCustom(builder, op, ctx); })
        .Case<mlir::jeff::PPROp>([&](auto op) { serializePpr(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize gate operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown gate operation");
        });
}

void serializeQubit(jeff::Op::Builder builder, mlir::jeff::QubitOperation operation,
                    SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::QubitAllocOp>([&](auto op) { serializeQubitAlloc(builder, op, ctx); })
        .Case<mlir::jeff::QubitFreeOp>([&](auto op) { serializeQubitFree(builder, op, ctx); })
        .Case<mlir::jeff::QubitFreeZeroOp>(
            [&](auto op) { serializeQubitFreeZero(builder, op, ctx); })
        .Case<mlir::jeff::QubitMeasureOp>([&](auto op) { serializeMeasure(builder, op, ctx); })
        .Case<mlir::jeff::QubitMeasureNDOp>([&](auto op) { serializeMeasureNd(builder, op, ctx); })
        .Case<mlir::jeff::QubitResetOp>([&](auto op) { serializeReset(builder, op, ctx); })
        .Case<mlir::jeff::MultipleCtrlQubitsGate>([&](auto op) { serializeGate(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize qubit operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown qubit operation");
        });
}

//===----------------------------------------------------------------------===//
// Qureg operations
//===----------------------------------------------------------------------===//

void serializeQuregAlloc(jeff::Op::Builder builder, mlir::jeff::QuregAllocOp op,
                         SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setAlloc();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getNumQubits()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeQuregFreeZero(jeff::Op::Builder builder, mlir::jeff::QuregFreeZeroOp op,
                            SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setFreeZero();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getQreg()));

    builder.initOutputs(0);
}

void serializeQuregExtractIndex(jeff::Op::Builder builder, mlir::jeff::QuregExtractIndexOp op,
                                SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setExtractIndex();

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getInQreg()));
    inputs.set(1, ctx.getValueId(op.getIndex()));

    auto outputs = builder.initOutputs(2);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
    outputs.set(1, ctx.getValueId(op.getOutQubit()));
}

void serializeQuregInsertIndex(jeff::Op::Builder builder, mlir::jeff::QuregInsertIndexOp op,
                               SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setInsertIndex();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getInQreg()));
    inputs.set(1, ctx.getValueId(op.getIndex()));
    inputs.set(2, ctx.getValueId(op.getInQubit()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregExtractSlice(jeff::Op::Builder builder, mlir::jeff::QuregExtractSliceOp op,
                                SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setExtractSlice();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getInQreg()));
    inputs.set(1, ctx.getValueId(op.getStart()));
    inputs.set(2, ctx.getValueId(op.getLength()));

    auto outputs = builder.initOutputs(2);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
    outputs.set(1, ctx.getValueId(op.getNewQreg()));
}

void serializeQuregInsertSlice(jeff::Op::Builder builder, mlir::jeff::QuregInsertSliceOp op,
                               SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setInsertSlice();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getInQreg()));
    inputs.set(1, ctx.getValueId(op.getStart()));
    inputs.set(2, ctx.getValueId(op.getNewQreg()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregLength(jeff::Op::Builder builder, mlir::jeff::QuregLengthOp op,
                          SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setLength();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInQreg()));

    auto outputs = builder.initOutputs(2);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
    outputs.set(1, ctx.getValueId(op.getLength()));
}

void serializeQuregSplit(jeff::Op::Builder builder, mlir::jeff::QuregSplitOp op,
                         SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setSplit();

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getInQreg()));
    inputs.set(1, ctx.getValueId(op.getIndex()));

    auto outputs = builder.initOutputs(2);
    outputs.set(0, ctx.getValueId(op.getOutQregOne()));
    outputs.set(1, ctx.getValueId(op.getOutQregTwo()));
}

void serializeQuregJoin(jeff::Op::Builder builder, mlir::jeff::QuregJoinOp op,
                        SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setJoin();

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getInQregOne()));
    inputs.set(1, ctx.getValueId(op.getInQregTwo()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregCreate(jeff::Op::Builder builder, mlir::jeff::QuregCreateOp op,
                          SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setCreate();

    auto inputs = builder.initInputs(op.getInQubits().size());
    for (size_t i = 0; i < op.getInQubits().size(); ++i) {
        inputs.set(i, ctx.getValueId(op.getInQubits()[i]));
    }

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregFree(jeff::Op::Builder builder, mlir::jeff::QuregFreeOp op,
                        SerializationContext& ctx) {
    auto quregBuilder = builder.initInstruction().initQureg();
    quregBuilder.setFree();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getQreg()));

    builder.initOutputs(0);
}

void serializeQureg(jeff::Op::Builder builder, mlir::jeff::QuregOperation operation,
                    SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::QuregAllocOp>([&](auto op) { serializeQuregAlloc(builder, op, ctx); })
        .Case<mlir::jeff::QuregFreeZeroOp>(
            [&](auto op) { serializeQuregFreeZero(builder, op, ctx); })
        .Case<mlir::jeff::QuregExtractIndexOp>(
            [&](auto op) { serializeQuregExtractIndex(builder, op, ctx); })
        .Case<mlir::jeff::QuregInsertIndexOp>(
            [&](auto op) { serializeQuregInsertIndex(builder, op, ctx); })
        .Case<mlir::jeff::QuregExtractSliceOp>(
            [&](auto op) { serializeQuregExtractSlice(builder, op, ctx); })
        .Case<mlir::jeff::QuregInsertSliceOp>(
            [&](auto op) { serializeQuregInsertSlice(builder, op, ctx); })
        .Case<mlir::jeff::QuregLengthOp>([&](auto op) { serializeQuregLength(builder, op, ctx); })
        .Case<mlir::jeff::QuregSplitOp>([&](auto op) { serializeQuregSplit(builder, op, ctx); })
        .Case<mlir::jeff::QuregJoinOp>([&](auto op) { serializeQuregJoin(builder, op, ctx); })
        .Case<mlir::jeff::QuregCreateOp>([&](auto op) { serializeQuregCreate(builder, op, ctx); })
        .Case<mlir::jeff::QuregFreeOp>([&](auto op) { serializeQuregFree(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize qureg operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown qureg operation");
        });
}

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

#define SERIALIZE_INT_CONST(BIT_WIDTH)                                                             \
    void serializeIntConst##BIT_WIDTH(jeff::Op::Builder builder,                                   \
                                      mlir::jeff::IntConst##BIT_WIDTH##Op op,                      \
                                      SerializationContext& ctx) {                                 \
        auto intBuilder = builder.initInstruction().initInt();                                     \
        intBuilder.setConst##BIT_WIDTH(op.getVal());                                               \
                                                                                                   \
        builder.initInputs(0);                                                                     \
                                                                                                   \
        auto outputs = builder.initOutputs(1);                                                     \
        outputs.set(0, ctx.getValueId(op.getConstant()));                                          \
    }

SERIALIZE_INT_CONST(1)
SERIALIZE_INT_CONST(8)
SERIALIZE_INT_CONST(16)
SERIALIZE_INT_CONST(32)
SERIALIZE_INT_CONST(64)

#undef SERIALIZE_INT_CONST

void serializeIntUnary(jeff::Op::Builder builder, mlir::jeff::IntUnaryOp op,
                       SerializationContext& ctx) {
    auto intBuilder = builder.initInstruction().initInt();
    switch (op.getOp()) {
    case mlir::jeff::IntUnaryOperation::_not:
        intBuilder.setNot();
        break;
    case mlir::jeff::IntUnaryOperation::_abs:
        intBuilder.setAbs();
        break;
    default:
        llvm::report_fatal_error("Unknown int unary operation");
    }

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getA()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getB()));
}

void serializeIntBinary(jeff::Op::Builder builder, mlir::jeff::IntBinaryOp op,
                        SerializationContext& ctx) {
    auto intBuilder = builder.initInstruction().initInt();
    switch (op.getOp()) {
    case mlir::jeff::IntBinaryOperation::_add:
        intBuilder.setAdd();
        break;
    case mlir::jeff::IntBinaryOperation::_sub:
        intBuilder.setSub();
        break;
    case mlir::jeff::IntBinaryOperation::_mul:
        intBuilder.setMul();
        break;
    case mlir::jeff::IntBinaryOperation::_divS:
        intBuilder.setDivS();
        break;
    case mlir::jeff::IntBinaryOperation::_divU:
        intBuilder.setDivU();
        break;
    case mlir::jeff::IntBinaryOperation::_pow:
        intBuilder.setPow();
        break;
    case mlir::jeff::IntBinaryOperation::_and:
        intBuilder.setAnd();
        break;
    case mlir::jeff::IntBinaryOperation::_or:
        intBuilder.setOr();
        break;
    case mlir::jeff::IntBinaryOperation::_xor:
        intBuilder.setXor();
        break;
    case mlir::jeff::IntBinaryOperation::_minS:
        intBuilder.setMinS();
        break;
    case mlir::jeff::IntBinaryOperation::_minU:
        intBuilder.setMinU();
        break;
    case mlir::jeff::IntBinaryOperation::_maxS:
        intBuilder.setMaxS();
        break;
    case mlir::jeff::IntBinaryOperation::_maxU:
        intBuilder.setMaxU();
        break;
    case mlir::jeff::IntBinaryOperation::_remS:
        intBuilder.setRemS();
        break;
    case mlir::jeff::IntBinaryOperation::_remU:
        intBuilder.setRemU();
        break;
    case mlir::jeff::IntBinaryOperation::_shl:
        intBuilder.setShl();
        break;
    case mlir::jeff::IntBinaryOperation::_shr:
        intBuilder.setShr();
        break;
    default:
        llvm::report_fatal_error("Unknown int binary operation");
    }

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getA()));
    inputs.set(1, ctx.getValueId(op.getB()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeIntComparison(jeff::Op::Builder builder, mlir::jeff::IntComparisonOp op,
                            SerializationContext& ctx) {
    auto intBuilder = builder.initInstruction().initInt();
    switch (op.getOp()) {
    case mlir::jeff::IntComparisonOperation::_eq:
        intBuilder.setEq();
        break;
    case mlir::jeff::IntComparisonOperation::_ltS:
        intBuilder.setLtS();
        break;
    case mlir::jeff::IntComparisonOperation::_lteS:
        intBuilder.setLteS();
        break;
    case mlir::jeff::IntComparisonOperation::_ltU:
        intBuilder.setLtU();
        break;
    case mlir::jeff::IntComparisonOperation::_lteU:
        intBuilder.setLteU();
        break;
    default:
        llvm::report_fatal_error("Unknown int comparison operation");
    }

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getA()));
    inputs.set(1, ctx.getValueId(op.getB()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeIntSelect(jeff::Op::Builder builder, mlir::jeff::IntSelectOp op,
                        SerializationContext& ctx) {
    auto intBuilder = builder.initInstruction().initInt();
    intBuilder.setSelect();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getCondition()));
    inputs.set(1, ctx.getValueId(op.getA()));
    inputs.set(2, ctx.getValueId(op.getB()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeFloatConversion(jeff::Op::Builder builder, mlir::jeff::IntOperation operation,
                              SerializationContext& ctx) {
    auto intBuilder = builder.initInstruction().initInt();
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::IntExtSOp>([&](auto op) { intBuilder.setExtS(); })
        .Case<mlir::jeff::IntExtUOp>([&](auto op) { intBuilder.setExtU(); })
        .Case<mlir::jeff::IntTruncOp>([&](auto op) { intBuilder.setTrunc(); })
        .Case<mlir::jeff::IntToFloatSOp>([&](auto op) { intBuilder.setToFloatS(); })
        .Case<mlir::jeff::IntToFloatUOp>([&](auto op) { intBuilder.setToFloatU(); });

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(operation->getOperand(0)));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(operation->getResult(0)));
}

void serializeInt(jeff::Op::Builder builder, mlir::jeff::IntOperation operation,
                  SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::IntConst1Op>([&](auto op) { serializeIntConst1(builder, op, ctx); })
        .Case<mlir::jeff::IntConst8Op>([&](auto op) { serializeIntConst8(builder, op, ctx); })
        .Case<mlir::jeff::IntConst16Op>([&](auto op) { serializeIntConst16(builder, op, ctx); })
        .Case<mlir::jeff::IntConst32Op>([&](auto op) { serializeIntConst32(builder, op, ctx); })
        .Case<mlir::jeff::IntConst64Op>([&](auto op) { serializeIntConst64(builder, op, ctx); })
        .Case<mlir::jeff::IntUnaryOp>([&](auto op) { serializeIntUnary(builder, op, ctx); })
        .Case<mlir::jeff::IntBinaryOp>([&](auto op) { serializeIntBinary(builder, op, ctx); })
        .Case<mlir::jeff::IntComparisonOp>(
            [&](auto op) { serializeIntComparison(builder, op, ctx); })
        .Case<mlir::jeff::IntSelectOp>([&](auto op) { serializeIntSelect(builder, op, ctx); })
        .Case<mlir::jeff::IntExtSOp, mlir::jeff::IntExtUOp, mlir::jeff::IntTruncOp,
              mlir::jeff::IntToFloatSOp, mlir::jeff::IntToFloatUOp>(
            [&](auto) { serializeFloatConversion(builder, operation, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize int operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown int operation");
        });
}

//===----------------------------------------------------------------------===//
// IntArray operations
//===----------------------------------------------------------------------===//

void serializeIntArrayConst1(jeff::Op::Builder builder, mlir::jeff::IntArrayConst1Op op,
                             SerializationContext& ctx) {
    auto intArrayBuilder = builder.initInstruction().initIntArray();
    auto values = op.getInArray();
    auto listBuilder = intArrayBuilder.initConst1(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        listBuilder.set(i, static_cast<bool>(values[i]));
    }

    builder.initInputs(0);

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

#define SERIALIZE_INT_ARRAY_CONST(BIT_WIDTH)                                                       \
    void serializeIntArrayConst##BIT_WIDTH(jeff::Op::Builder builder,                              \
                                           mlir::jeff::IntArrayConst##BIT_WIDTH##Op op,            \
                                           SerializationContext& ctx) {                            \
        auto intArrayBuilder = builder.initInstruction().initIntArray();                           \
        auto values = op.getInArray();                                                             \
        auto listBuilder = intArrayBuilder.initConst##BIT_WIDTH(values.size());                    \
        for (size_t i = 0; i < values.size(); ++i) {                                               \
            listBuilder.set(i, static_cast<uint##BIT_WIDTH##_t>(values[i]));                       \
        }                                                                                          \
                                                                                                   \
        builder.initInputs(0);                                                                     \
                                                                                                   \
        auto outputs = builder.initOutputs(1);                                                     \
        outputs.set(0, ctx.getValueId(op.getOutArray()));                                          \
    }

SERIALIZE_INT_ARRAY_CONST(8)
SERIALIZE_INT_ARRAY_CONST(16)
SERIALIZE_INT_ARRAY_CONST(32)
SERIALIZE_INT_ARRAY_CONST(64)

#undef SERIALIZE_INT_ARRAY_CONST

void serializeIntArrayZero(jeff::Op::Builder builder, mlir::jeff::IntArrayZeroOp op,
                           SerializationContext& ctx) {
    auto tensorType = llvm::cast<mlir::RankedTensorType>(op.getOutArray().getType());
    checkRank(tensorType);
    auto elementType = llvm::cast<mlir::IntegerType>(tensorType.getElementType());

    auto intArrayBuilder = builder.initInstruction().initIntArray();
    intArrayBuilder.setZero(elementType.getWidth());

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getLength()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeIntArrayGetIndex(jeff::Op::Builder builder, mlir::jeff::IntArrayGetIndexOp op,
                               SerializationContext& ctx) {
    auto intArrayBuilder = builder.initInstruction().initIntArray();
    intArrayBuilder.setGetIndex();

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getInArray()));
    inputs.set(1, ctx.getValueId(op.getIndex()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getValue()));
}

void serializeIntArraySetIndex(jeff::Op::Builder builder, mlir::jeff::IntArraySetIndexOp op,
                               SerializationContext& ctx) {
    auto intArrayBuilder = builder.initInstruction().initIntArray();
    intArrayBuilder.setSetIndex();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getInArray()));
    inputs.set(1, ctx.getValueId(op.getIndex()));
    inputs.set(2, ctx.getValueId(op.getValue()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeIntArrayLength(jeff::Op::Builder builder, mlir::jeff::IntArrayLengthOp op,
                             SerializationContext& ctx) {
    auto intArrayBuilder = builder.initInstruction().initIntArray();
    intArrayBuilder.setLength();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInArray()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getLength()));
}

void serializeIntArrayCreate(jeff::Op::Builder builder, mlir::jeff::IntArrayCreateOp op,
                             SerializationContext& ctx) {
    auto intArrayBuilder = builder.initInstruction().initIntArray();
    intArrayBuilder.setCreate();

    auto inputs = builder.initInputs(op.getInArray().size());
    for (size_t i = 0; i < op.getInArray().size(); ++i) {
        inputs.set(i, ctx.getValueId(op.getInArray()[i]));
    }

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeIntArray(jeff::Op::Builder builder, mlir::jeff::IntArrayOperation operation,
                       SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::IntArrayConst1Op>(
            [&](auto op) { serializeIntArrayConst1(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayConst8Op>(
            [&](auto op) { serializeIntArrayConst8(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayConst16Op>(
            [&](auto op) { serializeIntArrayConst16(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayConst32Op>(
            [&](auto op) { serializeIntArrayConst32(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayConst64Op>(
            [&](auto op) { serializeIntArrayConst64(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayZeroOp>([&](auto op) { serializeIntArrayZero(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayGetIndexOp>(
            [&](auto op) { serializeIntArrayGetIndex(builder, op, ctx); })
        .Case<mlir::jeff::IntArraySetIndexOp>(
            [&](auto op) { serializeIntArraySetIndex(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayLengthOp>(
            [&](auto op) { serializeIntArrayLength(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayCreateOp>(
            [&](auto op) { serializeIntArrayCreate(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize int array operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown int array operation");
        });
}

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

void serializeFloatConst32(jeff::Op::Builder builder, mlir::jeff::FloatConst32Op op,
                           SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    auto value = static_cast<float>(op.getVal().convertToDouble());
    floatBuilder.setConst32(value);

    builder.initInputs(0);

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getConstant()));
}

void serializeFloatConst64(jeff::Op::Builder builder, mlir::jeff::FloatConst64Op op,
                           SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    auto value = op.getVal().convertToDouble();
    floatBuilder.setConst64(value);

    builder.initInputs(0);

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getConstant()));
}

void serializeFloatUnary(jeff::Op::Builder builder, mlir::jeff::FloatUnaryOp op,
                         SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    switch (op.getOp()) {
    case mlir::jeff::FloatUnaryOperation::_sqrt:
        floatBuilder.setSqrt();
        break;
    case mlir::jeff::FloatUnaryOperation::_abs:
        floatBuilder.setAbs();
        break;
    case mlir::jeff::FloatUnaryOperation::_ceil:
        floatBuilder.setCeil();
        break;
    case mlir::jeff::FloatUnaryOperation::_floor:
        floatBuilder.setFloor();
        break;
    case mlir::jeff::FloatUnaryOperation::_exp:
        floatBuilder.setExp();
        break;
    case mlir::jeff::FloatUnaryOperation::_log:
        floatBuilder.setLog();
        break;
    case mlir::jeff::FloatUnaryOperation::_sin:
        floatBuilder.setSin();
        break;
    case mlir::jeff::FloatUnaryOperation::_cos:
        floatBuilder.setCos();
        break;
    case mlir::jeff::FloatUnaryOperation::_tan:
        floatBuilder.setTan();
        break;
    case mlir::jeff::FloatUnaryOperation::_asin:
        floatBuilder.setAsin();
        break;
    case mlir::jeff::FloatUnaryOperation::_acos:
        floatBuilder.setAcos();
        break;
    case mlir::jeff::FloatUnaryOperation::_atan:
        floatBuilder.setAtan();
        break;
    case mlir::jeff::FloatUnaryOperation::_sinh:
        floatBuilder.setSinh();
        break;
    case mlir::jeff::FloatUnaryOperation::_cosh:
        floatBuilder.setCosh();
        break;
    case mlir::jeff::FloatUnaryOperation::_tanh:
        floatBuilder.setTanh();
        break;
    case mlir::jeff::FloatUnaryOperation::_asinh:
        floatBuilder.setAsinh();
        break;
    case mlir::jeff::FloatUnaryOperation::_acosh:
        floatBuilder.setAcosh();
        break;
    case mlir::jeff::FloatUnaryOperation::_atanh:
        floatBuilder.setAtanh();
        break;
    default:
        llvm::report_fatal_error("Unknown float unary operation");
    }

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getA()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getB()));
}

void serializeFloatBinary(jeff::Op::Builder builder, mlir::jeff::FloatBinaryOp op,
                          SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    switch (op.getOp()) {
    case mlir::jeff::FloatBinaryOperation::_add:
        floatBuilder.setAdd();
        break;
    case mlir::jeff::FloatBinaryOperation::_sub:
        floatBuilder.setSub();
        break;
    case mlir::jeff::FloatBinaryOperation::_mul:
        floatBuilder.setMul();
        break;
    case mlir::jeff::FloatBinaryOperation::_div:
        floatBuilder.setDiv();
        break;
    case mlir::jeff::FloatBinaryOperation::_pow:
        floatBuilder.setPow();
        break;
    case mlir::jeff::FloatBinaryOperation::_atan2:
        floatBuilder.setAtan2();
        break;
    case mlir::jeff::FloatBinaryOperation::_max:
        floatBuilder.setMax();
        break;
    case mlir::jeff::FloatBinaryOperation::_min:
        floatBuilder.setMin();
        break;
    default:
        llvm::report_fatal_error("Unknown float binary operation");
    }

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getA()));
    inputs.set(1, ctx.getValueId(op.getB()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeFloatComparison(jeff::Op::Builder builder, mlir::jeff::FloatComparisonOp op,
                              SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    switch (op.getOp()) {
    case mlir::jeff::FloatComparisonOperation::_eq:
        floatBuilder.setEq();
        break;
    case mlir::jeff::FloatComparisonOperation::_lt:
        floatBuilder.setLt();
        break;
    case mlir::jeff::FloatComparisonOperation::_lte:
        floatBuilder.setLte();
        break;
    default:
        llvm::report_fatal_error("Unknown float comparison operation");
    }

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getA()));
    inputs.set(1, ctx.getValueId(op.getB()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeFloatIs(jeff::Op::Builder builder, mlir::jeff::FloatIsOp op,
                      SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    switch (op.getOp()) {
    case mlir::jeff::FloatIsOperation::_isNan:
        floatBuilder.setIsNan();
        break;
    case mlir::jeff::FloatIsOperation::_isInf:
        floatBuilder.setIsInf();
        break;
    default:
        llvm::report_fatal_error("Unknown float is operation");
    }

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getA()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeFloatSelect(jeff::Op::Builder builder, mlir::jeff::FloatSelectOp op,
                          SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    floatBuilder.setSelect();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getCondition()));
    inputs.set(1, ctx.getValueId(op.getA()));
    inputs.set(2, ctx.getValueId(op.getB()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeFloatConversion(jeff::Op::Builder builder, mlir::jeff::FloatOperation operation,
                              SerializationContext& ctx) {
    auto floatBuilder = builder.initInstruction().initFloat();
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::FloatExtOp>([&](auto op) { floatBuilder.setExt(); })
        .Case<mlir::jeff::FloatTruncOp>([&](auto op) { floatBuilder.setTrunc(); })
        .Case<mlir::jeff::FloatToSIntOp>([&](auto op) { floatBuilder.setToSInt(); })
        .Case<mlir::jeff::FloatToUIntOp>([&](auto op) { floatBuilder.setToUInt(); });

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(operation->getOperand(0)));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(operation->getResult(0)));
}

void serializeFloat(jeff::Op::Builder builder, mlir::jeff::FloatOperation operation,
                    SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::FloatConst32Op>([&](auto op) { serializeFloatConst32(builder, op, ctx); })
        .Case<mlir::jeff::FloatConst64Op>([&](auto op) { serializeFloatConst64(builder, op, ctx); })
        .Case<mlir::jeff::FloatUnaryOp>([&](auto op) { serializeFloatUnary(builder, op, ctx); })
        .Case<mlir::jeff::FloatBinaryOp>([&](auto op) { serializeFloatBinary(builder, op, ctx); })
        .Case<mlir::jeff::FloatComparisonOp>(
            [&](auto op) { serializeFloatComparison(builder, op, ctx); })
        .Case<mlir::jeff::FloatIsOp>([&](auto op) { serializeFloatIs(builder, op, ctx); })
        .Case<mlir::jeff::FloatSelectOp>([&](auto op) { serializeFloatSelect(builder, op, ctx); })
        .Case<mlir::jeff::FloatExtOp, mlir::jeff::FloatTruncOp, mlir::jeff::FloatToSIntOp,
              mlir::jeff::FloatToUIntOp>(
            [&](auto) { serializeFloatConversion(builder, operation, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize float operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown float operation");
        });
}

//===----------------------------------------------------------------------===//
// FloatArray operations
//===----------------------------------------------------------------------===//

void serializeFloatArrayConst32(jeff::Op::Builder builder, mlir::jeff::FloatArrayConst32Op op,
                                SerializationContext& ctx) {
    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    auto values = op.getInArray();
    auto listBuilder = floatArrayBuilder.initConst32(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        listBuilder.set(i, static_cast<float>(values[i]));
    }

    builder.initInputs(0);

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayConst64(jeff::Op::Builder builder, mlir::jeff::FloatArrayConst64Op op,
                                SerializationContext& ctx) {
    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    auto values = op.getInArray();
    auto listBuilder = floatArrayBuilder.initConst64(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        listBuilder.set(i, static_cast<double>(values[i]));
    }

    builder.initInputs(0);

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayZero(jeff::Op::Builder builder, mlir::jeff::FloatArrayZeroOp op,
                             SerializationContext& ctx) {
    auto tensorType = llvm::cast<mlir::RankedTensorType>(op.getOutArray().getType());
    checkRank(tensorType);
    auto elementType = llvm::cast<mlir::FloatType>(tensorType.getElementType());

    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    if (elementType.getWidth() == 32) {
        floatArrayBuilder.setZero(jeff::FloatPrecision::FLOAT32);
    } else if (elementType.getWidth() == 64) {
        floatArrayBuilder.setZero(jeff::FloatPrecision::FLOAT64);
    } else {
        llvm::report_fatal_error("Unknown float array element type");
    }

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getLength()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayGetIndex(jeff::Op::Builder builder, mlir::jeff::FloatArrayGetIndexOp op,
                                 SerializationContext& ctx) {
    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    floatArrayBuilder.setGetIndex();

    auto inputs = builder.initInputs(2);
    inputs.set(0, ctx.getValueId(op.getInArray()));
    inputs.set(1, ctx.getValueId(op.getIndex()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getValue()));
}

void serializeFloatArraySetIndex(jeff::Op::Builder builder, mlir::jeff::FloatArraySetIndexOp op,
                                 SerializationContext& ctx) {
    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    floatArrayBuilder.setSetIndex();

    auto inputs = builder.initInputs(3);
    inputs.set(0, ctx.getValueId(op.getInArray()));
    inputs.set(1, ctx.getValueId(op.getIndex()));
    inputs.set(2, ctx.getValueId(op.getValue()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayLength(jeff::Op::Builder builder, mlir::jeff::FloatArrayLengthOp op,
                               SerializationContext& ctx) {
    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    floatArrayBuilder.setLength();

    auto inputs = builder.initInputs(1);
    inputs.set(0, ctx.getValueId(op.getInArray()));

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getLength()));
}

void serializeFloatArrayCreate(jeff::Op::Builder builder, mlir::jeff::FloatArrayCreateOp op,
                               SerializationContext& ctx) {
    auto floatArrayBuilder = builder.initInstruction().initFloatArray();
    floatArrayBuilder.setCreate();

    auto inputs = builder.initInputs(op.getInArray().size());
    for (size_t i = 0; i < op.getInArray().size(); ++i) {
        inputs.set(i, ctx.getValueId(op.getInArray()[i]));
    }

    auto outputs = builder.initOutputs(1);
    outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArray(jeff::Op::Builder builder, mlir::jeff::FloatArrayOperation operation,
                         SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::FloatArrayConst32Op>(
            [&](auto op) { serializeFloatArrayConst32(builder, op, ctx); })
        .Case<mlir::jeff::FloatArrayConst64Op>(
            [&](auto op) { serializeFloatArrayConst64(builder, op, ctx); })
        .Case<mlir::jeff::FloatArrayZeroOp>(
            [&](auto op) { serializeFloatArrayZero(builder, op, ctx); })
        .Case<mlir::jeff::FloatArrayGetIndexOp>(
            [&](auto op) { serializeFloatArrayGetIndex(builder, op, ctx); })
        .Case<mlir::jeff::FloatArraySetIndexOp>(
            [&](auto op) { serializeFloatArraySetIndex(builder, op, ctx); })
        .Case<mlir::jeff::FloatArrayLengthOp>(
            [&](auto op) { serializeFloatArrayLength(builder, op, ctx); })
        .Case<mlir::jeff::FloatArrayCreateOp>(
            [&](auto op) { serializeFloatArrayCreate(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize float array operation " << operation->getName()
                         << "\n";
            llvm::report_fatal_error("Unknown float array operation");
        });
}

//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

// Forward declaration
void serializeOperation(jeff::Op::Builder builder, mlir::Operation* operation,
                        SerializationContext& ctx);

void serializeRegion(jeff::Region::Builder builder, mlir::Region& region,
                     SerializationContext& ctx) {
    const auto numSources = region.getNumArguments();
    auto sources = builder.initSources(numSources);
    for (size_t i = 0; i < numSources; ++i) {
        sources.set(i, ctx.getValueId(region.getArgument(i)));
    }

    const auto numOperations = region.front().getOperations().size() - 1;
    auto operationBuilders = builder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : region.front().without_terminator()) {
        serializeOperation(operationBuilders[i], &operation, ctx);
        ++i;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(region.front().back());
    const auto numTargets = yieldOp.getNumOperands();
    auto targets = builder.initTargets(numTargets);
    for (size_t t = 0; t < numTargets; ++t) {
        targets.set(t, ctx.getValueId(yieldOp.getOperand(t)));
    }
}

void serializeSwitch(jeff::Op::Builder builder, mlir::jeff::SwitchOp op,
                     SerializationContext& ctx) {
    const auto numInputs = op.getNumOperands();
    auto inputs = builder.initInputs(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        inputs.set(i, ctx.getValueId(op.getOperand(i)));
    }

    const auto numOutputs = op.getNumResults();
    auto outputs = builder.initOutputs(numOutputs);
    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.set(i, ctx.getValueId(op.getResult(i)));
    }

    auto switchBuilder = builder.initInstruction().initScf().initSwitch();

    auto branches = op.getBranches();
    const auto numBranches = branches.size();
    auto branchBuilders = switchBuilder.initBranches(numBranches);
    for (size_t i = 0; i < numBranches; ++i) {
        serializeRegion(branchBuilders[i], branches[i], ctx);
    }

    serializeRegion(switchBuilder.initDefault(), op.getDefault(), ctx);
}

void serializeFor(jeff::Op::Builder builder, mlir::jeff::ForOp op, SerializationContext& ctx) {
    const auto numInputs = op.getNumOperands();
    auto inputs = builder.initInputs(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        inputs.set(i, ctx.getValueId(op.getOperand(i)));
    }

    const auto numOutputs = op.getNumResults();
    auto outputs = builder.initOutputs(numOutputs);
    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.set(i, ctx.getValueId(op.getResult(i)));
    }

    serializeRegion(builder.initInstruction().initScf().initFor(), op.getBody(), ctx);
}

void serializeWhile(jeff::Op::Builder builder, mlir::jeff::WhileOp op, SerializationContext& ctx) {
    const auto numInputs = op.getNumOperands();
    auto inputs = builder.initInputs(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        inputs.set(i, ctx.getValueId(op.getOperand(i)));
    }

    const auto numOutputs = op.getNumResults();
    auto outputs = builder.initOutputs(numOutputs);
    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.set(i, ctx.getValueId(op.getResult(i)));
    }

    auto whileBuilder = builder.initInstruction().initScf().initWhile();
    serializeRegion(whileBuilder.initBefore(), op.getBefore(), ctx);
    serializeRegion(whileBuilder.initAfter(), op.getAfter(), ctx);
}

void serializeSCF(jeff::Op::Builder builder, mlir::jeff::SCFOperation operation,
                  SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::SwitchOp>([&](auto op) { serializeSwitch(builder, op, ctx); })
        .Case<mlir::jeff::ForOp>([&](auto op) { serializeFor(builder, op, ctx); })
        .Case<mlir::jeff::WhileOp>([&](auto op) { serializeWhile(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize SCF operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown SCF operation");
        });
}

//===----------------------------------------------------------------------===//
// Func operations
//===----------------------------------------------------------------------===//

void serializeCall(jeff::Op::Builder builder, mlir::func::CallOp op, SerializationContext& ctx) {
    auto funcBuilder = builder.initInstruction().initFunc();
    funcBuilder.setFuncCall(ctx.getFuncId(op.getCallee()));

    const auto numInputs = op.getNumOperands();
    auto inputs = builder.initInputs(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        inputs.set(i, ctx.getValueId(op.getOperand(i)));
    }

    const auto numOutputs = op.getNumResults();
    auto outputs = builder.initOutputs(numOutputs);
    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.set(i, ctx.getValueId(op.getResult(i)));
    }
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

void serializeQubitType(jeff::Type::Builder builder) { builder.setQubit(); }

void serializeQuregType(jeff::Type::Builder builder, mlir::jeff::QuregType quregType) {
    auto quregBuilder = builder.initQureg();
    auto length = quregType.getLength();
    if (length == mlir::ShapedType::kDynamic) {
        quregBuilder.setDynamic();
    } else {
        quregBuilder.setStatic(length);
    }
}

void serializeIntType(jeff::Type::Builder builder, mlir::IntegerType intType) {
    builder.setInt(intType.getWidth());
}

void serializeIntArrayType(jeff::Type::Builder builder, mlir::RankedTensorType tensorType) {
    auto intArrayBuilder = builder.initIntArray();
    auto elementType = llvm::cast<mlir::IntegerType>(tensorType.getElementType());
    intArrayBuilder.setBitwidth(elementType.getWidth());
    auto length = tensorType.getShape()[0];
    if (length == mlir::ShapedType::kDynamic) {
        intArrayBuilder.initLength().setDynamic();
    } else {
        intArrayBuilder.initLength().setStatic(length);
    }
}

void serializeFloatType(jeff::Type::Builder builder, mlir::FloatType floatType) {
    if (floatType.getWidth() == 32) {
        builder.setFloat(jeff::FloatPrecision::FLOAT32);
    } else if (floatType.getWidth() == 64) {
        builder.setFloat(jeff::FloatPrecision::FLOAT64);
    } else {
        llvm::errs() << "Cannot serialize floats with bit width " << floatType.getWidth() << "\n";
        llvm::report_fatal_error("Unknown float type");
    }
}

void serializeFloatArrayType(jeff::Type::Builder builder, mlir::RankedTensorType tensorType) {
    auto floatArrayBuilder = builder.initFloatArray();
    auto elementType = llvm::cast<mlir::FloatType>(tensorType.getElementType());
    if (elementType.getWidth() == 32) {
        floatArrayBuilder.setPrecision(jeff::FloatPrecision::FLOAT32);
    } else if (elementType.getWidth() == 64) {
        floatArrayBuilder.setPrecision(jeff::FloatPrecision::FLOAT64);
    } else {
        llvm::errs() << "Cannot serialize float arrays with bit width " << elementType.getWidth()
                     << "\n";
        llvm::report_fatal_error("Unknown float array type");
    }
    auto length = tensorType.getShape()[0];
    if (length == mlir::ShapedType::kDynamic) {
        floatArrayBuilder.initLength().setDynamic();
    } else {
        floatArrayBuilder.initLength().setStatic(length);
    }
}

void serializeRankedTensorType(jeff::Type::Builder builder, mlir::RankedTensorType tensorType) {
    checkRank(tensorType);
    llvm::TypeSwitch<mlir::Type, void>(tensorType.getElementType())
        .Case<mlir::IntegerType>([&](auto) { serializeIntArrayType(builder, tensorType); })
        .Case<mlir::FloatType>([&](auto) { serializeFloatArrayType(builder, tensorType); })
        .Default([&](auto elementType) {
            llvm::errs() << "Cannot serialize ranked tensor with element type " << elementType
                         << "\n";
            llvm::report_fatal_error("Unknown ranked tensor type");
        });
}

void serializeType(jeff::Type::Builder builder, mlir::Type type) {
    llvm::TypeSwitch<mlir::Type, void>(type)
        .Case<mlir::jeff::QubitType>([&](auto) { serializeQubitType(builder); })
        .Case<mlir::jeff::QuregType>(
            [&](auto quregType) { serializeQuregType(builder, quregType); })
        .Case<mlir::IntegerType>([&](auto intType) { serializeIntType(builder, intType); })
        .Case<mlir::RankedTensorType>(
            [&](auto tensorType) { serializeRankedTensorType(builder, tensorType); })
        .Case<mlir::FloatType>([&](auto floatType) { serializeFloatType(builder, floatType); })
        .Default([&](auto) { llvm::report_fatal_error("Unknown type"); });
}

void serializeOperation(jeff::Op::Builder builder, mlir::Operation* operation,
                        SerializationContext& ctx) {
    llvm::TypeSwitch<mlir::Operation*, void>(operation)
        .Case<mlir::jeff::QubitOperation>([&](auto op) { serializeQubit(builder, op, ctx); })
        .Case<mlir::jeff::QuregOperation>([&](auto op) { serializeQureg(builder, op, ctx); })
        .Case<mlir::jeff::IntOperation>([&](auto op) { serializeInt(builder, op, ctx); })
        .Case<mlir::jeff::IntArrayOperation>([&](auto op) { serializeIntArray(builder, op, ctx); })
        .Case<mlir::jeff::FloatOperation>([&](auto op) { serializeFloat(builder, op, ctx); })
        .Case<mlir::jeff::FloatArrayOperation>(
            [&](auto op) { serializeFloatArray(builder, op, ctx); })
        .Case<mlir::jeff::SCFOperation>([&](auto op) { serializeSCF(builder, op, ctx); })
        .Case<mlir::func::CallOp>([&](auto op) { serializeCall(builder, op, ctx); })
        .Default([&](auto) {
            llvm::errs() << "Cannot serialize operation " << operation->getName() << "\n";
            llvm::report_fatal_error("Unknown operation");
        });
}

void serializeFunction(jeff::Function::Builder functionBuilder, mlir::func::FuncOp func,
                       SerializationContext& ctx) {
    ctx.values.clear();

    auto defBuilder = functionBuilder.initDefinition();
    auto& entryBlock = func.getRegion().front();

    // Build body
    auto bodyBuilder = defBuilder.initBody();

    // Set sources
    const auto numSources = entryBlock.getNumArguments();
    auto sourcesBuilder = bodyBuilder.initSources(numSources);
    for (unsigned i = 0; i < numSources; ++i) {
        sourcesBuilder.set(i, ctx.getValueId(entryBlock.getArgument(i)));
    }

    // Serialize operations
    const auto numOperations = entryBlock.getOperations().size() - 1;
    auto operationBuilders = bodyBuilder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : entryBlock.getOperations()) {
        if (llvm::isa<mlir::func::ReturnOp>(operation)) {
            continue;
        }
        serializeOperation(operationBuilders[i], &operation, ctx);
        ++i;
    }

    // Set targets
    auto returnOp = llvm::cast<mlir::func::ReturnOp>(entryBlock.back());
    const auto numTargets = returnOp.getNumOperands();
    auto targetsBuilder = bodyBuilder.initTargets(numTargets);
    for (unsigned t = 0; t < numTargets; ++t) {
        targetsBuilder.set(t, ctx.getValueId(returnOp.getOperand(t)));
    }

    // Build values
    const auto numValues = ctx.values.size();
    auto valuesBuilder = defBuilder.initValues(numValues);
    llvm::SmallVector<mlir::Value> values(numValues);
    for (auto& pair : ctx.values) {
        values[pair.second] = pair.first;
    }
    for (size_t v = 0; v < numValues; ++v) {
        auto valueBuilder = valuesBuilder[v];
        auto typeBuilder = valueBuilder.initType();
        serializeType(typeBuilder, values[v].getType());
    }
}

void writeMessage(mlir::ModuleOp module, capnp::MallocMessageBuilder& message) {
    SerializationContext ctx;

    auto moduleBuilder = message.initRoot<jeff::Module>();

    // Get strings
    auto stringsAttr = llvm::cast<mlir::ArrayAttr>(module->getAttr("jeff.strings"));
    const auto numStrings = stringsAttr.size();
    auto stringsBuilder = moduleBuilder.initStrings(numStrings);
    for (int32_t i = 0; i < numStrings; ++i) {
        const auto str = llvm::cast<mlir::StringAttr>(stringsAttr[i]).getValue().str();
        ctx.strings[str] = i;
        stringsBuilder.set(i, str);
    }

    // Build functions
    uint16_t id = 0;
    llvm::SmallVector<mlir::func::FuncOp> functions;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        ctx.funcs[func.getSymName()] = id++;
        functions.push_back(func);
    }

    const auto numFunctions = functions.size();
    auto functionBuilders = moduleBuilder.initFunctions(numFunctions);

    for (size_t i = 0; i < numFunctions; ++i) {
        auto function = functions[i];
        auto functionBuilder = functionBuilders[i];
        functionBuilder.setName(ctx.getStringId(function.getName().str()));
        serializeFunction(functionBuilder, function, ctx);
    }

    // Set metadata
    moduleBuilder.setEntrypoint(
        llvm::cast<mlir::IntegerAttr>(module->getAttr("jeff.entrypoint")).getUInt());

    moduleBuilder.setTool(
        llvm::cast<mlir::StringAttr>(module->getAttr("jeff.tool")).getValue().str());

    moduleBuilder.setToolVersion(
        llvm::cast<mlir::StringAttr>(module->getAttr("jeff.toolVersion")).getValue().str());

    moduleBuilder.setVersion(
        llvm::cast<mlir::IntegerAttr>(module->getAttr("jeff.version")).getUInt());

    moduleBuilder.setVersionMinor(
        llvm::cast<mlir::IntegerAttr>(module->getAttr("jeff.versionMinor")).getUInt());

    moduleBuilder.setVersionPatch(
        llvm::cast<mlir::IntegerAttr>(module->getAttr("jeff.versionPatch")).getUInt());
}

} // namespace

kj::Array<capnp::word> serialize(mlir::ModuleOp module) {
    capnp::MallocMessageBuilder message;
    writeMessage(module, message);
    return capnp::messageToFlatArray(message);
}

mlir::LogicalResult serializeToFile(mlir::ModuleOp module, llvm::StringRef path) {
    capnp::MallocMessageBuilder message;
    writeMessage(module, message);

    auto file = llvm::sys::fs::openNativeFileForWrite(path, llvm::sys::fs::CD_CreateAlways,
                                                      llvm::sys::fs::OF_None);
    if (!file) {
        llvm::errs() << "Failed to open " << path << ": " << llvm::toString(file.takeError())
                     << "\n";
        return mlir::failure();
    }

#ifdef _WIN32
    kj::AutoCloseHandle handle(*file);
    kj::HandleOutputStream output(kj::mv(handle));
    capnp::writeMessage(output, message);
#else
    const kj::AutoCloseFd autoCloseFd(*file);
    capnp::writeMessageToFd(autoCloseFd, message);
#endif
    return mlir::success();
}
