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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct RoundTripTestCase {
    std::string filename;
};

std::ostream& operator<<(std::ostream& os, const RoundTripTestCase& testCase) {
    return os << testCase.filename;
}

class RoundTripTest : public ::testing::Test,
                      public ::testing::WithParamInterface<RoundTripTestCase> {};

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

    const auto module = message.getRoot<jeff::Module>();
    return module.toString().flatten().cStr();
}

std::string moduleTextFromBuffer(const kj::ArrayPtr<capnp::word>& buffer) {
    capnp::FlatArrayMessageReader message(buffer);
    const auto module = message.getRoot<jeff::Module>();
    return module.toString().flatten().cStr();
}

std::vector<RoundTripTestCase> getTestCases() {
    std::vector<RoundTripTestCase> cases;
    for (const auto& entry : fs::directory_iterator(TEST_INPUTS_DIR)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".jeff") {
            continue;
        }
        cases.push_back({entry.path().filename().string()});
    }
    std::sort(cases.begin(), cases.end(),
              [](const auto& a, const auto& b) { return a.filename < b.filename; });
    return cases;
}

} // namespace

TEST_P(RoundTripTest, RoundTrip) {
    const auto& testCase = GetParam();

    if (testCase.filename.rfind("skip_", 0) == 0) {
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

    llvm::errs() << "Deserialized MLIR module:\n";
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

INSTANTIATE_TEST_SUITE_P(, RoundTripTest, ::testing::ValuesIn(getTestCases()));

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

    EXPECT_EQ(readJeffFileToText(input.string()), readJeffFileToText(output.string()));
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

TEST(DeserializeFromFileTest, ReturnsNullForMissingFile) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    const auto input = fs::path(::testing::TempDir()) / "missing" / "does_not_exist.jeff";
    EXPECT_FALSE(deserializeFromFile(&context, input.string()));
}

TEST(DeserializeFromFileTest, ReturnsNullForTruncatedFile) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    // Drop a single byte so that the size is no longer a multiple of the word size.
    const fs::path inputsDir = TEST_INPUTS_DIR;
    const auto input = inputsDir / "bell_pair.jeff";
    const auto truncated = fs::path(::testing::TempDir()) / "truncated.jeff";
    fs::copy_file(input, truncated, fs::copy_options::overwrite_existing);
    fs::resize_file(truncated, fs::file_size(truncated) - 1);

    EXPECT_FALSE(deserializeFromFile(&context, truncated.string()));
}

TEST(DeserializeTest, IndependentControlFlowTuples) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    for (const auto* source : {
             R"MLIR(
      func.func @main(%value: i32) -> (i1, i32) {
        %flag, %result = jeff.while : (i32) -> (i1, i32) args(%before = %value) {
          %false = jeff.int_const1(false) : i1
          jeff.yield %false, %false, %before : i1, i1, i32
        } args(%after_flag, %after_value) {
          jeff.yield %after_value : i32
        }
        return %flag, %result : i1, i32
      }
    )MLIR",
             R"MLIR(
      func.func @main(%selector: i32, %value: i32, %flag: i1) -> i1 {
        %result = jeff.switch (%selector, %value, %flag) : (i32, i32, i1) -> (i1)
        case 0 args(%x, %p) {
          jeff.yield %p : i1
        }
        default args(%x, %p) {
          jeff.yield %p : i1
        }
        return %result : i1
      }
    )MLIR"}) {
        SCOPED_TRACE(source);
        auto original =
            mlir::parseSourceString<mlir::ModuleOp>(std::string(R"MLIR(module attributes {
              jeff.strings = ["main"], jeff.entrypoint = 0 : ui16,
              jeff.tool = "test", jeff.toolVersion = "test",
              jeff.version = 0 : ui16, jeff.versionMinor = 3 : ui16,
              jeff.versionPatch = 0 : ui16
            } {)MLIR") + source + "}",
                                                    &context);
        ASSERT_TRUE(original);
        auto bytes = serialize(*original);
        auto decoded = deserialize(&context, bytes);
        ASSERT_TRUE(decoded);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*decoded)));
        auto reserialized = serialize(*decoded);
        EXPECT_EQ(moduleTextFromBuffer(bytes), moduleTextFromBuffer(reserialized));
    }
}
