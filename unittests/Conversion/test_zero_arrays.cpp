#include "jeff/Conversion/JeffToNative/JeffToNative.h"
#include "jeff/IR/JeffDialect.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

TEST(ZeroArrayTest, GenericOptimizations) {
    MLIRContext context;
    context.loadDialect<jeff::JeffDialect, func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @zeros(%length: i32) ->
                (tensor<?xi32>, tensor<?xi32>, tensor<?xf64>, tensor<?xf64>) {
            %a = jeff.int_array_zero(%length) : tensor<?xi32>
            %b = jeff.int_array_zero(%length) : tensor<?xi32>
            %c = jeff.float_array_zero(%length) : tensor<?xf64>
            %d = jeff.float_array_zero(%length) : tensor<?xf64>
            %unused_i = jeff.int_array_zero(%length) : tensor<?xi8>
            %unused_f = jeff.float_array_zero(%length) : tensor<?xf32>
            return %a, %b, %c, %d :
                tensor<?xi32>, tensor<?xi32>, tensor<?xf64>, tensor<?xf64>
        }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    PassManager pm(&context);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto function = module->lookupSymbol<func::FuncOp>("zeros");
    auto result = cast<func::ReturnOp>(function.getBody().front().getTerminator());
    EXPECT_EQ(result.getOperand(0), result.getOperand(1));
    EXPECT_EQ(result.getOperand(2), result.getOperand(3));
    EXPECT_EQ(function.getBody().front().getOperations().size(), 3U);
}

TEST(ZeroArrayTest, NativeConversionPreservesValues) {
    MLIRContext context;
    context.loadDialect<jeff::JeffDialect, func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @zeros() -> (i8, i8, f64) {
            %length = jeff.int_const32(3) : i32
            %index = jeff.int_const32(1) : i32
            %value = jeff.int_const8(7) : i8
            %a = jeff.int_array_zero(%length) : tensor<3xi8>
            %b = jeff.int_array_zero(%length) : tensor<3xi8>
            %f = jeff.float_array_zero(%length) : tensor<3xf64>
            %updated = jeff.int_array_set_index(%index) %a %value :
                i32, tensor<3xi8>, i8 -> tensor<3xi8>
            %changed = jeff.int_array_get_index(%index) %updated :
                i32, tensor<3xi8> -> i8
            %original = jeff.int_array_get_index(%index) %b :
                i32, tensor<3xi8> -> i8
            %zero = jeff.float_array_get_index(%index) %f :
                i32, tensor<3xf64> -> f64
            return %changed, %original, %zero : i8, i8, f64
        }
        func.func @dynamic(%length: i32) -> (tensor<?xi8>, tensor<?xf64>) {
            %a = jeff.int_array_zero(%length) : tensor<?xi8>
            %f = jeff.float_array_zero(%length) : tensor<?xf64>
            return %a, %f : tensor<?xi8>, tensor<?xf64>
        }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    PassManager pm(&context);
    pm.addPass(createCSEPass());
    pm.addPass(createJeffToNative());
    pm.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto function = module->lookupSymbol<func::FuncOp>("zeros");
    auto result = cast<func::ReturnOp>(function.getBody().front().getTerminator());
    auto changed = result.getOperand(0).getDefiningOp<arith::ConstantOp>();
    auto original = result.getOperand(1).getDefiningOp<arith::ConstantOp>();
    auto zero = result.getOperand(2).getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(changed && original && zero);
    EXPECT_EQ(cast<IntegerAttr>(changed.getValue()).getInt(), 7);
    EXPECT_EQ(cast<IntegerAttr>(original.getValue()).getInt(), 0);
    EXPECT_TRUE(cast<FloatAttr>(zero.getValue()).getValue().isPosZero());

    auto dynamic = module->lookupSymbol<func::FuncOp>("dynamic");
    auto dynamicResult = cast<func::ReturnOp>(dynamic.getBody().front().getTerminator());
    auto integerSplat = dynamicResult.getOperand(0).getDefiningOp<tensor::SplatOp>();
    auto floatSplat = dynamicResult.getOperand(1).getDefiningOp<tensor::SplatOp>();
    ASSERT_TRUE(integerSplat && floatSplat);
    auto integer = integerSplat.getInput().getDefiningOp<arith::ConstantOp>();
    auto floating = floatSplat.getInput().getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(integer && floating);
    EXPECT_EQ(cast<IntegerAttr>(integer.getValue()).getInt(), 0);
    EXPECT_TRUE(cast<FloatAttr>(floating.getValue()).getValue().isPosZero());
}
