#include "jeff/Conversion/JeffToNative/JeffToNative.h"
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h"
#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <filesystem>
#include <utility>

namespace fs = std::filesystem;

TEST(NativeConversionTest, ZeroArrayValues) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();
    for (const auto* filename : {"unit_int_array_zero.jeff", "unit_float_array_zero.jeff"}) {
        SCOPED_TRACE(filename);
        auto module =
            deserializeFromFile(&context, (fs::path(TEST_INPUTS_DIR) / filename).string());
        ASSERT_TRUE(module);
        auto function = module->lookupSymbol<mlir::func::FuncOp>("main");
        auto result = llvm::cast<mlir::func::ReturnOp>(function.getBody().front().getTerminator());
        auto value = result.getOperand(0);
        auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
        // Exercise static and dynamic result dimensions using the existing fixtures.
        if (llvm::isa<mlir::IntegerType>(type.getElementType())) {
            value.setType(mlir::RankedTensorType::get({5}, type.getElementType()));
        }
        mlir::OpBuilder builder(value.getDefiningOp());
        auto duplicate = builder.clone(*value.getDefiningOp())->getResult(0);
        builder.clone(*value.getDefiningOp()); // An unused zero value.
        result->setOperands({value, duplicate});
        function.setType(builder.getFunctionType({}, {value.getType(), value.getType()}));
        mlir::PassManager pm(&context);
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        ASSERT_TRUE(pm.run(*module).succeeded());
        EXPECT_EQ(result.getOperand(0), result.getOperand(1));
        for (auto& op : function.getBody().front()) {
            EXPECT_FALSE(op.getNumResults() && op.use_empty());
        }

        mlir::PassManager conversion(&context);
        conversion.addPass(mlir::createJeffToNative());
        ASSERT_TRUE(conversion.run(*module).succeeded());
        mlir::PassManager canonicalizer(&context);
        canonicalizer.addPass(mlir::createCanonicalizerPass());
        ASSERT_TRUE(canonicalizer.run(*module).succeeded());
        auto nativeValue = result.getOperand(0);
        if (auto splat = nativeValue.getDefiningOp<mlir::tensor::SplatOp>()) {
            nativeValue = splat.getInput();
        }
        auto constant = nativeValue.getDefiningOp<mlir::arith::ConstantOp>();
        ASSERT_TRUE(constant);
        EXPECT_EQ(constant.getValue(), builder.getZeroAttr(constant.getType()));
    }
}

TEST(NativeArrayConstantConversion, PreservesOrderedBitPatterns) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::arith::ArithDialect, mlir::jeff::JeffDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(module {
      %0 = arith.constant dense<[[-1, 0, 1], [-4, 2, 3]]> : tensor<2x3xi3>
      %1 = arith.constant dense<[[-0.0, 0.0], [0x7ff8000000000042, 1.25]]> : tensor<2x2xf64>
    })mlir",
                                                          &context);
    auto expected = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(module {
      %0 = jeff.int_array_const8([7, 0, 1, 4, 2, 3]) : tensor<6xi8>
      %1 = jeff.float_array_const64([-0.0, 0.0, 0x7ff8000000000042, 1.25]) : tensor<4xf64>
    })mlir",
                                                            &context);
    ASSERT_TRUE(module && expected);
    mlir::TypeConverter converter;
    converter.addConversion([&](mlir::Value value) -> mlir::Type {
        auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
        auto element = type.getElementType();
        if (element.isInteger(3)) {
            element = mlir::IntegerType::get(&context, 8);
        }
        return mlir::RankedTensorType::get({type.getNumElements()}, element);
    });
    mlir::RewritePatternSet patterns(&context);
    mlir::jeff::populateNativeToJeffConversionPatterns(patterns, &converter);
    mlir::ConversionTarget target(context);
    target.addIllegalOp<mlir::arith::ConstantOp>();
    target.addLegalDialect<mlir::jeff::JeffDialect>();
    ASSERT_TRUE(
        mlir::succeeded(mlir::applyPartialConversion(*module, target, std::move(patterns))));
    EXPECT_TRUE(mlir::OperationEquivalence::isEquivalentTo(
        *module, *expected, mlir::OperationEquivalence::IgnoreLocations));
}

TEST(NativeArrayConstantConversion, RejectsLossyConversion) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::arith::ArithDialect, mlir::jeff::JeffDialect>();
    for (auto converted : {
             mlir::RankedTensorType::get({2}, mlir::IntegerType::get(&context, 8)),
             mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context, 16)),
         }) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            "module { %0 = arith.constant dense<[256, -1]> : tensor<2xi16> }", &context);
        ASSERT_TRUE(module);
        mlir::TypeConverter converter;
        converter.addConversion([&](mlir::RankedTensorType) { return converted; });
        mlir::RewritePatternSet patterns(&context);
        mlir::jeff::populateNativeToJeffConversionPatterns(patterns, &converter);
        mlir::ConversionTarget target(context);
        target.addIllegalOp<mlir::arith::ConstantOp>();
        target.addLegalDialect<mlir::jeff::JeffDialect>();
        EXPECT_TRUE(
            mlir::failed(mlir::applyPartialConversion(*module, target, std::move(patterns))));
    }
}
