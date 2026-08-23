#include "jeff/Conversion/NativeToJeff/NativeToJeff.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_NATIVETOJEFF
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

struct ConvertArithConstOp final : OpConversionPattern<arith::ConstantOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        auto value = op.getValue();
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType())
            .Case<IntegerType>([&](auto type) {
                auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
                if (!intAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected IntegerAttr");
                }
                switch (type.getWidth()) {
                case 1:
                    rewriter.replaceOpWithNewOp<jeff::IntConst1Op>(op, intAttr);
                    return success();
                case 8:
                    rewriter.replaceOpWithNewOp<jeff::IntConst8Op>(op, intAttr);
                    return success();
                case 16:
                    rewriter.replaceOpWithNewOp<jeff::IntConst16Op>(op, intAttr);
                    return success();
                case 32:
                    rewriter.replaceOpWithNewOp<jeff::IntConst32Op>(op, intAttr);
                    return success();
                case 64:
                    rewriter.replaceOpWithNewOp<jeff::IntConst64Op>(op, intAttr);
                    return success();
                default:
                    return rewriter.notifyMatchFailure(op, "Unsupported integer type");
                }
            })
            .Case<FloatType>([&](auto type) {
                auto floatAttr = llvm::dyn_cast<FloatAttr>(value);
                if (!floatAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected FloatAttr");
                }
                switch (type.getWidth()) {
                case 32:
                    rewriter.replaceOpWithNewOp<jeff::FloatConst32Op>(op, floatAttr);
                    return success();
                case 64:
                    rewriter.replaceOpWithNewOp<jeff::FloatConst64Op>(op, floatAttr);
                    return success();
                default:
                    return rewriter.notifyMatchFailure(op, "Unsupported float type");
                }
            })
            .Case<RankedTensorType>([&](auto type) {
                if (type.getRank() != 1) {
                    return rewriter.notifyMatchFailure(
                        op, "Only one-dimensional tensors are supported");
                }
                auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(value);
                if (!denseAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected DenseElementsAttr");
                }
                auto elementType = type.getElementType();
                auto* ctx = op.getContext();
                auto loc = op.getLoc();
                return llvm::TypeSwitch<Type, LogicalResult>(elementType)
                    .template Case<IntegerType>([&](auto intType) {
                        switch (intType.getWidth()) {
                        case 1: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<bool>());
                            auto inArrayAttr = mlir::DenseBoolArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst1Op>(op, type,
                                                                                inArrayAttr);
                            return success();
                        }
                        case 8: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int8_t>());
                            auto inArrayAttr = mlir::DenseI8ArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst8Op>(op, type,
                                                                                inArrayAttr);
                            return success();
                        }
                        case 16: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int16_t>());
                            auto inArrayAttr = mlir::DenseI16ArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst16Op>(op, type,
                                                                                 inArrayAttr);
                            return success();
                        }
                        case 32: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int32_t>());
                            auto inArrayAttr = mlir::DenseI32ArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst32Op>(op, type,
                                                                                 inArrayAttr);
                            return success();
                        }
                        case 64: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int64_t>());
                            auto inArrayAttr = mlir::DenseI64ArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst64Op>(op, type,
                                                                                 inArrayAttr);
                            return success();
                        }
                        default:
                            return rewriter.notifyMatchFailure(op, "Unsupported integer type");
                        }
                    })
                    .template Case<FloatType>([&](auto floatType) {
                        switch (floatType.getWidth()) {
                        case 32: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<float>());
                            auto inArrayAttr = mlir::DenseF32ArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::FloatArrayConst32Op>(op, type,
                                                                                   inArrayAttr);
                            return success();
                        }
                        case 64: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<double>());
                            auto inArrayAttr = mlir::DenseF64ArrayAttr::get(ctx, inArray);
                            rewriter.replaceOpWithNewOp<jeff::FloatArrayConst64Op>(op, type,
                                                                                   inArrayAttr);
                            return success();
                        }
                        default:
                            return rewriter.notifyMatchFailure(op, "Unsupported float type");
                        }
                    })
                    .Default([&](auto) {
                        return rewriter.notifyMatchFailure(op, "Unsupported element type");
                    });
            })
            .Case<IndexType>([&](auto) {
                auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
                if (!intAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected IntegerAttr");
                }
                const auto value = intAttr.getInt();
                if (value > std::numeric_limits<int32_t>::max()) {
                    return rewriter.notifyMatchFailure(op, "Index value out of range");
                }
                rewriter.replaceOpWithNewOp<jeff::IntConst32Op>(
                    op, rewriter.getI32IntegerAttr(static_cast<int32_t>(value)));
                return success();
            })
            .Default([&](auto) { return rewriter.notifyMatchFailure(op, "Unsupported type"); });
    }
};

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

struct ConvertMathAbsIOp final : OpConversionPattern<math::AbsIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::AbsIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntUnaryOp>(op, adaptor.getOperand(),
                                                      jeff::IntUnaryOperation::_abs);
        return success();
    }
};

template <typename ArithOp, jeff::IntBinaryOperation JeffOp>
struct ConvertArithIntBinaryOp final : OpConversionPattern<ArithOp> {
    using OpConversionPattern<ArithOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                       JeffOp);
        return success();
    }
};

struct ConvertMathIPowIOp final : OpConversionPattern<math::IPowIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::IPowIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                       jeff::IntBinaryOperation::_pow);
        return success();
    }
};

struct ConvertArithCmpIOpToJeff final : OpConversionPattern<arith::CmpIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getLhs();
        auto b = adaptor.getRhs();
        switch (op.getPredicate()) {
        case arith::CmpIPredicate::eq:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_eq);
            break;
        case arith::CmpIPredicate::slt:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_ltS);
            break;
        case arith::CmpIPredicate::sle:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_lteS);
            break;
        case arith::CmpIPredicate::ult:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_ltU);
            break;
        case arith::CmpIPredicate::ule:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_lteU);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

template <typename MathOp, jeff::FloatUnaryOperation JeffOp>
struct ConvertMathFloatUnaryOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatUnaryOp>(op, adaptor.getOperand(), JeffOp);
        return success();
    }
};

template <typename ArithOp, jeff::FloatBinaryOperation JeffOp>
struct ConvertArithFloatBinaryOp final : OpConversionPattern<ArithOp> {
    using OpConversionPattern<ArithOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                         JeffOp);
        return success();
    }
};

template <typename MathOp, jeff::FloatBinaryOperation JeffOp>
struct ConvertMathFloatBinaryOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                         JeffOp);
        return success();
    }
};

struct ConvertArithCmpFOp final : OpConversionPattern<arith::CmpFOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getLhs();
        auto b = adaptor.getRhs();
        switch (op.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
            rewriter.replaceOpWithNewOp<jeff::FloatComparisonOp>(
                op, a, b, jeff::FloatComparisonOperation::_eq);
            break;
        case arith::CmpFPredicate::OLT:
            rewriter.replaceOpWithNewOp<jeff::FloatComparisonOp>(
                op, a, b, jeff::FloatComparisonOperation::_lt);
            break;
        case arith::CmpFPredicate::OLE:
            rewriter.replaceOpWithNewOp<jeff::FloatComparisonOp>(
                op, a, b, jeff::FloatComparisonOperation::_lte);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

template <typename MathOp, jeff::FloatIsOperation JeffOp>
struct ConvertMathFloatIsOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatIsOp>(op, adaptor.getOperand(), JeffOp);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Mixed operations
//===----------------------------------------------------------------------===//

struct ConvertArithSelectOp final : OpConversionPattern<arith::SelectOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        if (llvm::isa<IntegerType>(op.getType())) {
            rewriter.replaceOpWithNewOp<jeff::IntSelectOp>(op, op.getType(), adaptor.getCondition(),
                                                           adaptor.getTrueValue(),
                                                           adaptor.getFalseValue());
        } else if (llvm::isa<FloatType>(op.getType())) {
            rewriter.replaceOpWithNewOp<jeff::FloatSelectOp>(
                op, op.getType(), adaptor.getCondition(), adaptor.getTrueValue(),
                adaptor.getFalseValue());
        } else {
            return rewriter.notifyMatchFailure(op, "Unsupported result type");
        }
        return success();
    }
};

template <typename ArithOp, typename JeffOp>
struct ConvertArithConversionOp final : OpConversionPattern<ArithOp> {
    using OpConversionPattern<ArithOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<JeffOp>(op, op.getType(), adaptor.getIn());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// IntArray/FloatArray operations
//===----------------------------------------------------------------------===//

struct ConvertTensorEmptyOp final : OpConversionPattern<tensor::EmptyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto sizes = adaptor.getDynamicSizes();
        if (sizes.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Only one-dimensional tensors are supported");
        }
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType().getElementType())
            .Case<IntegerType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::IntArrayZeroOp>(op, op.getType(), sizes[0]);
                return success();
            })
            .Case<FloatType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayZeroOp>(op, op.getType(), sizes[0]);
                return success();
            })
            .Default(
                [&](auto) { return rewriter.notifyMatchFailure(op, "Unsupported element type"); });
    }
};

struct ConvertTensorExtractOp final : OpConversionPattern<tensor::ExtractOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto indices = adaptor.getIndices();
        if (indices.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Only one-dimensional tensors are supported");
        }
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType())
            .Case<IntegerType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::IntArrayGetIndexOp>(
                    op, op.getType(), adaptor.getTensor(), indices[0]);
                return success();
            })
            .Case<FloatType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayGetIndexOp>(
                    op, op.getType(), adaptor.getTensor(), indices[0]);
                return success();
            })
            .Default(
                [&](auto) { return rewriter.notifyMatchFailure(op, "Unsupported element type"); });
    }
};

struct ConvertTensorInsertOp final : OpConversionPattern<tensor::InsertOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto indices = adaptor.getIndices();
        if (indices.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Only one-dimensional tensors are supported");
        }
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType().getElementType())
            .Case<IntegerType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::IntArraySetIndexOp>(
                    op, op.getType(), adaptor.getDest(), indices[0], adaptor.getScalar());
                return success();
            })
            .Case<FloatType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::FloatArraySetIndexOp>(
                    op, op.getType(), adaptor.getDest(), indices[0], adaptor.getScalar());
                return success();
            })
            .Default(
                [&](auto) { return rewriter.notifyMatchFailure(op, "Unsupported element type"); });
    }
};

struct ConvertTensorDimOp final : OpConversionPattern<tensor::DimOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        return llvm::TypeSwitch<Type, LogicalResult>(op.getSource().getType().getElementType())
            .Case<IntegerType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::IntArrayLengthOp>(op, adaptor.getSource());
                return success();
            })
            .Case<FloatType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayLengthOp>(op, adaptor.getSource());
                return success();
            })
            .Default(
                [&](auto) { return rewriter.notifyMatchFailure(op, "Unsupported element type"); });
    }
};

struct ConvertTensorFromElementsOp final : OpConversionPattern<tensor::FromElementsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto type = op.getType();
        auto elementType = type.getElementType();
        return llvm::TypeSwitch<Type, LogicalResult>(elementType)
            .Case<IntegerType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::IntArrayCreateOp>(op, type,
                                                                    adaptor.getElements());
                return success();
            })
            .Case<FloatType>([&](auto) {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayCreateOp>(op, type,
                                                                      adaptor.getElements());
                return success();
            })
            .Default(
                [&](auto) { return rewriter.notifyMatchFailure(op, "Unsupported element type"); });
    }
};

struct ConvertArithIndexCastOp final : OpConversionPattern<arith::IndexCastOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOp(op, adaptor.getIn());
        return success();
    }
};

/**
 * @brief Pass for converting built-in MLIR operations to jeff operations
 */
struct NativeToJeff final : impl::NativeToJeffBase<NativeToJeff> {
    using NativeToJeffBase::NativeToJeffBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addIllegalDialect<arith::ArithDialect, math::MathDialect, tensor::TensorDialect>();
        target.addLegalDialect<jeff::JeffDialect>();

        RewritePatternSet patterns(context);
        jeff::populateNativeToJeffConversionPatterns(patterns);
        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace

namespace jeff {

void populateNativeToJeffConversionPatterns(RewritePatternSet& patterns) {
    patterns.add<
        // Constants
        ConvertArithConstOp,
        // Int operations
        ConvertMathAbsIOp, ConvertArithIntBinaryOp<arith::AddIOp, jeff::IntBinaryOperation::_add>,
        ConvertArithIntBinaryOp<arith::SubIOp, jeff::IntBinaryOperation::_sub>,
        ConvertArithIntBinaryOp<arith::MulIOp, jeff::IntBinaryOperation::_mul>,
        ConvertArithIntBinaryOp<arith::DivSIOp, jeff::IntBinaryOperation::_divS>,
        ConvertArithIntBinaryOp<arith::DivUIOp, jeff::IntBinaryOperation::_divU>,
        ConvertMathIPowIOp, ConvertArithIntBinaryOp<arith::AndIOp, jeff::IntBinaryOperation::_and>,
        ConvertArithIntBinaryOp<arith::OrIOp, jeff::IntBinaryOperation::_or>,
        ConvertArithIntBinaryOp<arith::XOrIOp, jeff::IntBinaryOperation::_xor>,
        ConvertArithIntBinaryOp<arith::MinSIOp, jeff::IntBinaryOperation::_minS>,
        ConvertArithIntBinaryOp<arith::MinUIOp, jeff::IntBinaryOperation::_minU>,
        ConvertArithIntBinaryOp<arith::MaxSIOp, jeff::IntBinaryOperation::_maxS>,
        ConvertArithIntBinaryOp<arith::MaxUIOp, jeff::IntBinaryOperation::_maxU>,
        ConvertArithIntBinaryOp<arith::RemSIOp, jeff::IntBinaryOperation::_remS>,
        ConvertArithIntBinaryOp<arith::RemUIOp, jeff::IntBinaryOperation::_remU>,
        ConvertArithIntBinaryOp<arith::ShLIOp, jeff::IntBinaryOperation::_shl>,
        ConvertArithIntBinaryOp<arith::ShRSIOp, jeff::IntBinaryOperation::_shr>,
        ConvertArithCmpIOpToJeff,
        // Float operations
        ConvertMathFloatUnaryOp<math::SqrtOp, jeff::FloatUnaryOperation::_sqrt>,
        ConvertMathFloatUnaryOp<math::AbsFOp, jeff::FloatUnaryOperation::_abs>,
        ConvertMathFloatUnaryOp<math::CeilOp, jeff::FloatUnaryOperation::_ceil>,
        ConvertMathFloatUnaryOp<math::FloorOp, jeff::FloatUnaryOperation::_floor>,
        ConvertMathFloatUnaryOp<math::ExpOp, jeff::FloatUnaryOperation::_exp>,
        ConvertMathFloatUnaryOp<math::LogOp, jeff::FloatUnaryOperation::_log>,
        ConvertMathFloatUnaryOp<math::SinOp, jeff::FloatUnaryOperation::_sin>,
        ConvertMathFloatUnaryOp<math::CosOp, jeff::FloatUnaryOperation::_cos>,
        ConvertMathFloatUnaryOp<math::TanOp, jeff::FloatUnaryOperation::_tan>,
        ConvertMathFloatUnaryOp<math::AsinOp, jeff::FloatUnaryOperation::_asin>,
        ConvertMathFloatUnaryOp<math::AcosOp, jeff::FloatUnaryOperation::_acos>,
        ConvertMathFloatUnaryOp<math::AtanOp, jeff::FloatUnaryOperation::_atan>,
        ConvertMathFloatUnaryOp<math::SinhOp, jeff::FloatUnaryOperation::_sinh>,
        ConvertMathFloatUnaryOp<math::CoshOp, jeff::FloatUnaryOperation::_cosh>,
        ConvertMathFloatUnaryOp<math::TanhOp, jeff::FloatUnaryOperation::_tanh>,
        ConvertMathFloatUnaryOp<math::AsinhOp, jeff::FloatUnaryOperation::_asinh>,
        ConvertMathFloatUnaryOp<math::AcoshOp, jeff::FloatUnaryOperation::_acosh>,
        ConvertMathFloatUnaryOp<math::AtanhOp, jeff::FloatUnaryOperation::_atanh>,
        ConvertArithFloatBinaryOp<arith::AddFOp, jeff::FloatBinaryOperation::_add>,
        ConvertArithFloatBinaryOp<arith::SubFOp, jeff::FloatBinaryOperation::_sub>,
        ConvertArithFloatBinaryOp<arith::MulFOp, jeff::FloatBinaryOperation::_mul>,
        ConvertArithFloatBinaryOp<arith::DivFOp, jeff::FloatBinaryOperation::_div>,
        ConvertMathFloatBinaryOp<math::Atan2Op, jeff::FloatBinaryOperation::_atan2>,
        ConvertMathFloatBinaryOp<math::PowFOp, jeff::FloatBinaryOperation::_pow>,
        ConvertArithFloatBinaryOp<arith::MaxNumFOp, jeff::FloatBinaryOperation::_max>,
        ConvertArithFloatBinaryOp<arith::MinNumFOp, jeff::FloatBinaryOperation::_min>,
        ConvertArithCmpFOp, ConvertMathFloatIsOp<math::IsNaNOp, jeff::FloatIsOperation::_isNan>,
        ConvertMathFloatIsOp<math::IsInfOp, jeff::FloatIsOperation::_isInf>,
        // Mixed operations
        ConvertArithSelectOp, ConvertArithConversionOp<arith::ExtSIOp, jeff::IntExtSOp>,
        ConvertArithConversionOp<arith::ExtUIOp, jeff::IntExtUOp>,
        ConvertArithConversionOp<arith::TruncIOp, jeff::IntTruncOp>,
        ConvertArithConversionOp<arith::SIToFPOp, jeff::IntToFloatSOp>,
        ConvertArithConversionOp<arith::UIToFPOp, jeff::IntToFloatUOp>,
        ConvertArithConversionOp<arith::ExtFOp, jeff::FloatExtOp>,
        ConvertArithConversionOp<arith::TruncFOp, jeff::FloatTruncOp>,
        ConvertArithConversionOp<arith::FPToSIOp, jeff::FloatToSIntOp>,
        ConvertArithConversionOp<arith::FPToUIOp, jeff::FloatToUIntOp>,
        // IntArray/FloatArray operations
        ConvertTensorEmptyOp, ConvertTensorExtractOp, ConvertTensorInsertOp, ConvertTensorDimOp,
        ConvertTensorFromElementsOp,
        // Cast operations
        ConvertArithIndexCastOp>(patterns.getContext());
}

} // namespace jeff

} // namespace mlir
