#include "jeff/Conversion/JeffToNative/JeffToNative.h"

#include "jeff/IR/JeffOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>

namespace mlir {

#define GEN_PASS_DEF_JEFFTONATIVE
#include "jeff/Conversion/JeffToNative/JeffToNative.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

template <typename ConstOp> struct ConvertJeffIntConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

struct ConvertJeffIntUnaryOp final : OpConversionPattern<jeff::IntUnaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntUnaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        switch (op.getOp()) {
        case jeff::IntUnaryOperation::_not: {
            auto one = arith::ConstantOp::create(rewriter, op.getLoc(),
                                                 mlir::IntegerAttr::get(a.getType(), -1));
            rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, a, one.getResult());
            break;
        }
        case jeff::IntUnaryOperation::_abs:
            rewriter.replaceOpWithNewOp<math::AbsIOp>(op, a);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown unary operation");
        }
        return success();
    }
};

struct ConvertJeffIntBinaryOp final : OpConversionPattern<jeff::IntBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntBinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::IntBinaryOperation::_add:
            rewriter.replaceOpWithNewOp<arith::AddIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_sub:
            rewriter.replaceOpWithNewOp<arith::SubIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_mul:
            rewriter.replaceOpWithNewOp<arith::MulIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_divS:
            rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_divU:
            rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_pow:
            rewriter.replaceOpWithNewOp<math::IPowIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_and:
            rewriter.replaceOpWithNewOp<arith::AndIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_or:
            rewriter.replaceOpWithNewOp<arith::OrIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_xor:
            rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_minS:
            rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_minU:
            rewriter.replaceOpWithNewOp<arith::MinUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_maxS:
            rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_maxU:
            rewriter.replaceOpWithNewOp<arith::MaxUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_remS:
            rewriter.replaceOpWithNewOp<arith::RemSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_remU:
            rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_shl:
            rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_shr:
            rewriter.replaceOpWithNewOp<arith::ShRSIOp>(op, a, b);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

struct ConvertJeffIntComparisonOp final : OpConversionPattern<jeff::IntComparisonOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntComparisonOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::IntComparisonOperation::_eq:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::eq, a, b);
            break;
        case jeff::IntComparisonOperation::_ltS:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::slt, a, b);
            break;
        case jeff::IntComparisonOperation::_lteS:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::sle, a, b);
            break;
        case jeff::IntComparisonOperation::_ltU:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ult, a, b);
            break;
        case jeff::IntComparisonOperation::_lteU:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ule, a, b);
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

template <typename ConstOp> struct ConvertJeffFloatConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

struct ConvertJeffFloatUnaryOp final : OpConversionPattern<jeff::FloatUnaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatUnaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        switch (op.getOp()) {
        case jeff::FloatUnaryOperation::_sqrt:
            rewriter.replaceOpWithNewOp<math::SqrtOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_abs:
            rewriter.replaceOpWithNewOp<math::AbsFOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_ceil:
            rewriter.replaceOpWithNewOp<math::CeilOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_floor:
            rewriter.replaceOpWithNewOp<math::FloorOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_exp:
            rewriter.replaceOpWithNewOp<math::ExpOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_log:
            rewriter.replaceOpWithNewOp<math::LogOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_sin:
            rewriter.replaceOpWithNewOp<math::SinOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_cos:
            rewriter.replaceOpWithNewOp<math::CosOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_tan:
            rewriter.replaceOpWithNewOp<math::TanOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_asin:
            rewriter.replaceOpWithNewOp<math::AsinOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_acos:
            rewriter.replaceOpWithNewOp<math::AcosOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_atan:
            rewriter.replaceOpWithNewOp<math::AtanOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_sinh:
            rewriter.replaceOpWithNewOp<math::SinhOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_cosh:
            rewriter.replaceOpWithNewOp<math::CoshOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_tanh:
            rewriter.replaceOpWithNewOp<math::TanhOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_asinh:
            rewriter.replaceOpWithNewOp<math::AsinhOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_acosh:
            rewriter.replaceOpWithNewOp<math::AcoshOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_atanh:
            rewriter.replaceOpWithNewOp<math::AtanhOp>(op, a);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown unary operation");
        }
        return success();
    }
};

struct ConvertJeffFloatBinaryOp final : OpConversionPattern<jeff::FloatBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatBinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::FloatBinaryOperation::_add:
            rewriter.replaceOpWithNewOp<arith::AddFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_sub:
            rewriter.replaceOpWithNewOp<arith::SubFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_mul:
            rewriter.replaceOpWithNewOp<arith::MulFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_div:
            rewriter.replaceOpWithNewOp<arith::DivFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_pow:
            rewriter.replaceOpWithNewOp<math::PowFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_atan2:
            rewriter.replaceOpWithNewOp<math::Atan2Op>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_max:
            rewriter.replaceOpWithNewOp<arith::MaxNumFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_min:
            rewriter.replaceOpWithNewOp<arith::MinNumFOp>(op, a, b);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

struct ConvertJeffFloatComparisonOp final : OpConversionPattern<jeff::FloatComparisonOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatComparisonOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::FloatComparisonOperation::_eq:
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OEQ, a, b);
            break;
        case jeff::FloatComparisonOperation::_lt:
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OLT, a, b);
            break;
        case jeff::FloatComparisonOperation::_lte:
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OLE, a, b);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

struct ConvertJeffFloatIsOp final : OpConversionPattern<jeff::FloatIsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatIsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        switch (op.getOp()) {
        case jeff::FloatIsOperation::_isNan:
            rewriter.replaceOpWithNewOp<math::IsNaNOp>(op, a);
            break;
        case jeff::FloatIsOperation::_isInf:
            rewriter.replaceOpWithNewOp<math::IsInfOp>(op, a);
            break;
        default:
            return rewriter.notifyMatchFailure(op, "Unknown is operation");
        }
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Mixed operations
//===----------------------------------------------------------------------===//

template <typename JeffOp> struct ConvertJeffSelectOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getCondition(), adaptor.getA(),
                                                     adaptor.getB());
        return success();
    }
};

template <typename JeffOp, typename ArithOp>
struct ConvertJeffConversionOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<ArithOp>(op, op.getB().getType(), adaptor.getA());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// IntArray/FloatArray operations
//===----------------------------------------------------------------------===//

template <typename ConstOp> struct ConvertJeffIntArrayConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        if (op.getType().getShape()[0] == mlir::ShapedType::kDynamic) {
            return rewriter.notifyMatchFailure(
                op, "IntArray*ConstOps with dynamic length are not supported");
        }
        auto denseAttr = DenseElementsAttr::get(op.getType(), op.getInArrayAttr().asArrayRef());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseAttr);
        return success();
    }
};

template <typename ConstOp>
struct ConvertJeffFloatArrayConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        if (op.getType().getShape()[0] == mlir::ShapedType::kDynamic) {
            return rewriter.notifyMatchFailure(
                op, "IntArray*ConstOps with dynamic length are not supported");
        }
        auto denseAttr = DenseElementsAttr::get(op.getType(), op.getInArrayAttr().asArrayRef());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseAttr);
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayZeroOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto length = adaptor.getLength();
        auto lengthIndex =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), length);
        rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, op.getType(), lengthIndex.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayGetIndexOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto index = adaptor.getIndex();
        auto indexIndex =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), index);
        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, adaptor.getInArray(),
                                                       indexIndex.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArraySetIndexOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto index = adaptor.getIndex();
        auto indexIndex =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), index);
        rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getValue(), adaptor.getInArray(),
                                                      indexIndex.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayLengthOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        auto zero = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
        auto length = tensor::DimOp::create(rewriter, loc, adaptor.getInArray(), zero.getResult());
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op.getType(), length.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayCreateOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        if (op.getType().getShape()[0] == mlir::ShapedType::kDynamic) {
            return rewriter.notifyMatchFailure(
                op, "*ArrayCreateOps with dynamic length are not supported");
        }
        rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, adaptor.getInArray());
        return success();
    }
};

/**
 * @brief Pass for converting jeff operations to built-in MLIR operations
 */
struct JeffToNative final : impl::JeffToNativeBase<JeffToNative> {
    using JeffToNativeBase::JeffToNativeBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addLegalDialect<arith::ArithDialect, math::MathDialect, tensor::TensorDialect>();
        target.addIllegalOp<
            // Int operations
            jeff::IntConst1Op, jeff::IntConst8Op, jeff::IntConst16Op, jeff::IntConst32Op,
            jeff::IntConst64Op, jeff::IntUnaryOp, jeff::IntBinaryOp, jeff::IntComparisonOp,
            // Float operations
            jeff::FloatConst32Op, jeff::FloatConst64Op, jeff::FloatUnaryOp, jeff::FloatBinaryOp,
            jeff::FloatComparisonOp, jeff::FloatIsOp,
            // IntArray operations
            jeff::IntArrayConst1Op, jeff::IntArrayConst8Op, jeff::IntArrayConst16Op,
            jeff::IntArrayConst32Op, jeff::IntArrayConst64Op, jeff::IntArrayZeroOp,
            jeff::IntArrayGetIndexOp, jeff::IntArraySetIndexOp, jeff::IntArrayLengthOp,
            jeff::IntArrayCreateOp,
            // FloatArray operations
            jeff::FloatArrayConst32Op, jeff::FloatArrayConst64Op, jeff::FloatArrayZeroOp,
            jeff::FloatArrayGetIndexOp, jeff::FloatArraySetIndexOp, jeff::FloatArrayLengthOp,
            jeff::FloatArrayCreateOp>();

        RewritePatternSet patterns(context);
        jeff::populateJeffToNativeConversionPatterns(patterns);
        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace

namespace jeff {

void populateJeffToNativeConversionPatterns(RewritePatternSet& patterns) {
    patterns.add<
        // Int operations
        ConvertJeffIntConstOp<jeff::IntConst1Op>, ConvertJeffIntConstOp<jeff::IntConst8Op>,
        ConvertJeffIntConstOp<jeff::IntConst16Op>, ConvertJeffIntConstOp<jeff::IntConst32Op>,
        ConvertJeffIntConstOp<jeff::IntConst64Op>, ConvertJeffIntUnaryOp, ConvertJeffIntBinaryOp,
        ConvertJeffIntComparisonOp,
        // Float operations
        ConvertJeffFloatConstOp<jeff::FloatConst32Op>,
        ConvertJeffFloatConstOp<jeff::FloatConst64Op>, ConvertJeffFloatUnaryOp,
        ConvertJeffFloatBinaryOp, ConvertJeffFloatComparisonOp, ConvertJeffFloatIsOp,
        // Mixed operations
        ConvertJeffSelectOp<jeff::IntSelectOp>, ConvertJeffSelectOp<jeff::FloatSelectOp>,
        ConvertJeffConversionOp<jeff::IntExtSOp, arith::ExtSIOp>,
        ConvertJeffConversionOp<jeff::IntExtUOp, arith::ExtUIOp>,
        ConvertJeffConversionOp<jeff::IntTruncOp, arith::TruncIOp>,
        ConvertJeffConversionOp<jeff::IntToFloatSOp, arith::SIToFPOp>,
        ConvertJeffConversionOp<jeff::IntToFloatUOp, arith::UIToFPOp>,
        ConvertJeffConversionOp<jeff::FloatExtOp, arith::ExtFOp>,
        ConvertJeffConversionOp<jeff::FloatTruncOp, arith::TruncFOp>,
        ConvertJeffConversionOp<jeff::FloatToSIntOp, arith::FPToSIOp>,
        ConvertJeffConversionOp<jeff::FloatToUIntOp, arith::FPToUIOp>,
        // IntArray operations
        ConvertJeffIntArrayConstOp<jeff::IntArrayConst1Op>,
        ConvertJeffIntArrayConstOp<jeff::IntArrayConst8Op>,
        ConvertJeffIntArrayConstOp<jeff::IntArrayConst16Op>,
        ConvertJeffIntArrayConstOp<jeff::IntArrayConst32Op>,
        ConvertJeffIntArrayConstOp<jeff::IntArrayConst64Op>,
        ConvertJeffArrayZeroOp<jeff::IntArrayZeroOp>,
        ConvertJeffArrayGetIndexOp<jeff::IntArrayGetIndexOp>,
        ConvertJeffArraySetIndexOp<jeff::IntArraySetIndexOp>,
        ConvertJeffArrayLengthOp<jeff::IntArrayLengthOp>,
        ConvertJeffArrayCreateOp<jeff::IntArrayCreateOp>,
        // FloatArray operations
        ConvertJeffFloatArrayConstOp<jeff::FloatArrayConst32Op>,
        ConvertJeffFloatArrayConstOp<jeff::FloatArrayConst64Op>,
        ConvertJeffArrayZeroOp<jeff::FloatArrayZeroOp>,
        ConvertJeffArrayGetIndexOp<jeff::FloatArrayGetIndexOp>,
        ConvertJeffArraySetIndexOp<jeff::FloatArraySetIndexOp>,
        ConvertJeffArrayLengthOp<jeff::FloatArrayLengthOp>,
        ConvertJeffArrayCreateOp<jeff::FloatArrayCreateOp>>(patterns.getContext());
}

} // namespace jeff

} // namespace mlir
