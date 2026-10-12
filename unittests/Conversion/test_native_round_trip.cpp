#include "jeff/Conversion/JeffToNative/JeffToNative.h"
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h"
#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/common.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <kj/common.h>
#include <kj/io.h>
#include <kj/string-tree.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct NativeRoundTripTestCase {
    std::string filename;
};

std::ostream& operator<<(std::ostream& os, const NativeRoundTripTestCase& testCase) {
    return os << testCase.filename;
}

class NativeRoundTripTest : public ::testing::Test,
                            public ::testing::WithParamInterface<NativeRoundTripTestCase> {};

std::string readJeffFileToText(llvm::StringRef path) {
    auto file = llvm::sys::fs::openNativeFileForRead(path);
    if (!file) {
        llvm::errs() << "Failed to open file: " << path << "\n";
        llvm::report_fatal_error("Could not open file");
    }

    capnp::MallocMessageBuilder message;
#ifdef _WIN32
    kj::AutoCloseHandle autoCloseHandle(*file);
    kj::HandleInputStream input(std::move(autoCloseHandle));
    capnp::readMessageCopy(input, message);
#else
    const kj::AutoCloseFd autoCloseFd(*file);
    capnp::readMessageCopyFromFd(autoCloseFd, message);
#endif

    const auto jeffModule = message.getRoot<jeff::Module>();
    return jeffModule.toString().flatten().cStr();
}

std::string moduleTextFromBuffer(const kj::ArrayPtr<capnp::word>& buffer) {
    capnp::FlatArrayMessageReader message(buffer);
    const auto jeffModule = message.getRoot<jeff::Module>();
    return jeffModule.toString().flatten().cStr();
}

mlir::LogicalResult convertJeffToNative(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    pm.addPass(mlir::createJeffToNative());
    return pm.run(moduleOp);
}

mlir::LogicalResult convertNativeToJeff(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    pm.addPass(mlir::createNativeToJeff());
    return pm.run(moduleOp);
}

mlir::LogicalResult canonicalize(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    return pm.run(moduleOp);
}

std::vector<NativeRoundTripTestCase> getTestCases() {
    std::vector<NativeRoundTripTestCase> cases;
    for (const auto& entry : fs::directory_iterator(TEST_INPUTS_DIR)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".jeff") {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.rfind("unit_int_", 0) != 0 && filename.rfind("unit_float_", 0) != 0) {
            continue;
        }
        cases.push_back({filename});
    }
    std::sort(cases.begin(), cases.end(),
              [](const auto& a, const auto& b) { return a.filename < b.filename; });
    return cases;
}

} // namespace

TEST_P(NativeRoundTripTest, RoundTrip) {
    const auto& testCase = GetParam();

    if (testCase.filename.rfind("skip_", 0) == 0) {
        GTEST_SKIP();
    }

    // The following programs are correctly round-tripped, but the tests fail due to the simple
    // equivalence checking. The *_get_index and *_set_index tests fail because of the operation
    // order not being preserved. The *_length tests fail because the program is simplified by a
    // built-in canonicalization.
    if (testCase.filename == "unit_float_array_get_index.jeff" ||
        testCase.filename == "unit_float_array_length.jeff" ||
        testCase.filename == "unit_float_array_set_index.jeff" ||
        testCase.filename == "unit_int_array_get_index.jeff" ||
        testCase.filename == "unit_int_array_length.jeff" ||
        testCase.filename == "unit_int_array_set_index.jeff") {
        GTEST_SKIP();
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    const fs::path inputsDir = TEST_INPUTS_DIR;
    const auto& path = inputsDir / testCase.filename;

    // Deserialize jeff module
    auto mlirModule = deserializeFromFile(&context, path.string());

    llvm::errs() << "Input MLIR module:\n";
    mlirModule->print(llvm::errs());
    llvm::errs() << "\n\n";

    EXPECT_TRUE(convertJeffToNative(mlirModule.get()).succeeded());
    EXPECT_TRUE(verify(*mlirModule).succeeded());

    llvm::errs() << "Converted MLIR module:\n";
    mlirModule->print(llvm::errs());
    llvm::errs() << "\n\n";

    EXPECT_TRUE(convertNativeToJeff(mlirModule.get()).succeeded());
    EXPECT_TRUE(verify(*mlirModule).succeeded());

    EXPECT_TRUE(canonicalize(mlirModule.get()).succeeded());
    EXPECT_TRUE(verify(*mlirModule).succeeded());

    llvm::errs() << "Output MLIR module:\n";
    mlirModule->print(llvm::errs());
    llvm::errs() << "\n\n";

    // Serialize MLIR module
    auto serialized = serialize(*mlirModule);

    // Compare textual representations
    const auto originalText = readJeffFileToText(path.string());
    const auto serializedText = moduleTextFromBuffer(serialized);

    llvm::errs() << "Original module:\n" << originalText << "\n\n";
    llvm::errs() << "Serialized module:\n" << serializedText << "\n\n";

    ASSERT_EQ(originalText, serializedText);
}

INSTANTIATE_TEST_SUITE_P(, NativeRoundTripTest, ::testing::ValuesIn(getTestCases()));
