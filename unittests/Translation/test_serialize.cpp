#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

TEST(SerializeToFileTest, WritesFileThatRoundTrips) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

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
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    const fs::path inputsDir = TEST_INPUTS_DIR;
    auto mlirModule = deserializeFromFile(&context, (inputsDir / "bell_pair.jeff").string());
    ASSERT_TRUE(mlirModule);

    const auto output = inputsDir / "bell_pair.jeff" / "serialize_to_file.jeff";
    EXPECT_TRUE(mlir::failed(serializeToFile(*mlirModule, output.string())));
}

#ifdef __linux__
TEST(SerializeToFileTest, DiagnosesWriteFailure) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();
    auto moduleOp =
        deserializeFromFile(&context, (fs::path(TEST_INPUTS_DIR) / "bell_pair.jeff").string());
    ASSERT_TRUE(moduleOp);
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });
    EXPECT_TRUE(mlir::failed(serializeToFile(*moduleOp, "/dev/full")));
    EXPECT_NE(diagnostic.find("Failed to serialize jeff"), std::string::npos);
}
#endif
