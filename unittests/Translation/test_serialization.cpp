#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include <filesystem>

namespace fs = std::filesystem;

TEST(SerializeToFileTest, WritesFileThatRoundTrips) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    const fs::path inputsDir = TEST_INPUTS_DIR;
    const auto& input = inputsDir / "bell_pair.jeff";
    auto mlirModule = deserializeFromFile(&context, input.string());
    ASSERT_TRUE(mlirModule);

    const auto output = fs::path(::testing::TempDir()) / "serialize_to_file.jeff";
    ASSERT_TRUE(mlir::succeeded(serializeToFile(*mlirModule, output.string())));

    EXPECT_EQ(jeff::test::readJeffFileToText(input.string()),
              jeff::test::readJeffFileToText(output.string()));
}

TEST(SerializeToFileTest, FailsForUnwritablePath) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    const fs::path inputsDir = TEST_INPUTS_DIR;
    auto mlirModule = deserializeFromFile(&context, (inputsDir / "bell_pair.jeff").string());
    ASSERT_TRUE(mlirModule);

    const auto output = fs::path(::testing::TempDir()) / "missing" / "serialize_to_file.jeff";
    EXPECT_TRUE(mlir::failed(serializeToFile(*mlirModule, output.string())));
}
