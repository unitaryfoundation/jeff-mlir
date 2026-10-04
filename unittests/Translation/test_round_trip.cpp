#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/any.h>
#include <capnp/blob.h>
#include <capnp/common.h>
#include <capnp/list.h>
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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
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

TEST(DeserializeTest, DiagnosesMalformedBuffers) {
    mlir::MLIRContext context;
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });

    std::array<capnp::word, 1> words{};
    EXPECT_FALSE(deserialize(&context, kj::arrayPtr(words.data(), words.size())));
    EXPECT_FALSE(diagnostic.empty());
}

TEST(DeserializeTest, DiagnosesRejectedModulesAndContinuesImporting) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });

    const auto input = fs::path(::testing::TempDir()) / "rejected_module.jeff";
    const auto checkRejected = [&](capnp::MallocMessageBuilder& message, const char* expected) {
        auto words = capnp::messageToFlatArray(message);
        diagnostic.clear();
        EXPECT_FALSE(deserialize(&context, words));
        EXPECT_NE(diagnostic.find(expected), std::string::npos) << diagnostic;

        auto bytes = words.asBytes();
        std::ofstream output(input, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.begin()),
                     static_cast<std::streamsize>(bytes.size()));
        output.close();
        ASSERT_TRUE(output.good());
        diagnostic.clear();
        EXPECT_FALSE(deserializeFromFile(&context, input.string()));
        EXPECT_NE(diagnostic.find(expected), std::string::npos) << diagnostic;

        auto valid =
            deserializeFromFile(&context, (fs::path(TEST_INPUTS_DIR) / "bell_pair.jeff").string());
        ASSERT_TRUE(valid);
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*valid)));
    };

    capnp::MallocMessageBuilder missingFunctions;
    auto missing = missingFunctions.initRoot<jeff::Module>();
    missing.setVersionMinor(3);
    checkRejected(missingFunctions, "No functions found");

    capnp::MallocMessageBuilder undefinedValue;
    auto serializedModule = undefinedValue.initRoot<jeff::Module>();
    serializedModule.setVersionMinor(3);
    serializedModule.initStrings(1).set(0, "main");
    auto function = serializedModule.initFunctions(1)[0];
    function.setName(0);
    auto definition = function.initDefinition();
    definition.initValues(1)[0].initType().setQubit();
    auto body = definition.initBody();
    body.initSources(0);
    body.initTargets(0);
    auto free = body.initOperations(1)[0];
    free.initInputs(1).set(0, 0);
    free.initOutputs(0);
    free.initInstruction().initQubit().setFree();
    checkRejected(undefinedValue, "Value not found");

    capnp::MallocMessageBuilder duplicateFunctions;
    auto duplicates = duplicateFunctions.initRoot<jeff::Module>();
    duplicates.setVersionMinor(3);
    duplicates.initStrings(1).set(0, "main");
    for (auto duplicate : duplicates.initFunctions(2)) {
        duplicate.setName(0);
        auto empty = duplicate.initDefinition();
        empty.initValues(0);
        auto region = empty.initBody();
        region.initSources(0);
        region.initTargets(0);
        region.initOperations(0);
    }
    checkRejected(duplicateFunctions, "Verification of MLIR module failed");

    capnp::MallocMessageBuilder invalidInputName;
    auto namedModule = invalidInputName.initRoot<jeff::Module>();
    namedModule.initStrings(2).set(0, "main");
    namedModule.getStrings().set(1, "jeff.input_name");
    auto named = namedModule.initFunctions(1)[0].initDefinition();
    auto value = named.initValues(1)[0];
    value.initType().setFloat(jeff::FloatPrecision::FLOAT64);
    auto region = named.initBody();
    region.initSources(1).set(0, 0);
    region.initTargets(0);
    region.initOperations(0);
    auto metadata = value.initMetadata(1)[0];
    metadata.setName(2);
    checkRejected(invalidInputName, "Input metadata name index out of bounds");
    metadata.setName(1);
    checkRejected(invalidInputName, "Missing input name metadata value");
    metadata.getValue().initAs<capnp::List<uint32_t>>(1);
    checkRejected(invalidInputName, "Failed to deserialize jeff");
    for (auto duplicate : value.initMetadata(2)) {
        duplicate.setName(1);
        duplicate.getValue().setAs<capnp::Text>("theta");
    }
    checkRejected(invalidInputName, "Duplicate input name metadata");
    fs::remove(input);
}

TEST(DeserializeTest, ReadsInputNamesBySourceIndex) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    capnp::MallocMessageBuilder message;
    auto module = message.initRoot<jeff::Module>();
    auto strings = module.initStrings(3);
    strings.set(0, "main");
    strings.set(1, "jeff.input_name");
    strings.set(2, std::string("jeff.input_name") + '\0' + "ignored");
    auto definition = module.initFunctions(1)[0].initDefinition();
    auto values = definition.initValues(3);
    for (auto value : values) {
        value.initType().setFloat(jeff::FloatPrecision::FLOAT64);
    }
    auto metadata = values[0].initMetadata(2);
    metadata[0].setName(2);
    metadata[0].getValue().initAs<capnp::List<uint32_t>>(1);
    metadata[1].setName(1);
    metadata[1].getValue().setAs<capnp::Text>("theta");
    auto body = definition.initBody();
    auto sources = body.initSources(3);
    sources.set(0, 2);
    sources.set(1, 0);
    sources.set(2, 1);
    body.initTargets(0);
    body.initOperations(0);

    auto words = capnp::messageToFlatArray(message);
    auto decoded = deserialize(&context, words);
    ASSERT_TRUE(decoded);
    auto func = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_FALSE(func.getArgAttr(0, "jeff.input_name"));
    EXPECT_EQ(func.getArgAttr(1, "jeff.input_name"), mlir::StringAttr::get(&context, "theta"));
    EXPECT_FALSE(func.getArgAttr(2, "jeff.input_name"));
}

TEST(SerializeTest, PreservesNamedFunctionInputs) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    for (const auto* strings :
         {R"(["main", "helper"])", R"(["main", "helper", "jeff.input_name"])"}) {
        SCOPED_TRACE(strings);
        auto original = mlir::parseSourceString<mlir::ModuleOp>(
            std::string("module attributes { jeff.strings = ") + strings + R"MLIR(,
              jeff.entrypoint = 0 : ui16, jeff.tool = "test", jeff.toolVersion = "test",
              jeff.version = 0 : ui16, jeff.versionMinor = 3 : ui16,
              jeff.versionPatch = 1 : ui16
            } {
              func.func @main(%flag: i1, %theta: f64 {jeff.input_name = "theta"},
                              %phi: f64 {jeff.input_name = "φ"}) -> f64 {
                %result = func.call @helper(%theta) : (f64) -> f64
                return %result : f64
              }
              func.func @helper(%x: f64 {jeff.input_name = "x"}) -> f64 {
                return %x : f64
              }
            })MLIR",
            &context);
        ASSERT_TRUE(original);
        auto originalStrings = (*original)->getAttr("jeff.strings");
        auto words = serialize(*original);
        capnp::FlatArrayMessageReader message(words);
        auto serialized = message.getRoot<jeff::Module>();
        EXPECT_EQ(serialized.getStrings().size(), 3);
        auto definition = serialized.getFunctions()[0].getDefinition();
        auto metadata = definition.getValues()[definition.getBody().getSources()[1]].getMetadata();
        ASSERT_EQ(metadata.size(), 1);
        EXPECT_EQ(metadata[0].getName(), 2);
        EXPECT_EQ(metadata[0].getValue().getAs<capnp::Text>(), "theta");

        auto decoded = deserialize(&context, words);
        ASSERT_TRUE(decoded);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*decoded)));
        auto originalFunc = original->getOps<mlir::func::FuncOp>().begin();
        for (auto func : decoded->getOps<mlir::func::FuncOp>()) {
            EXPECT_EQ(func.getAllArgAttrs(), (*originalFunc).getAllArgAttrs());
            ++originalFunc;
        }
        auto reserialized = serialize(*decoded);
        EXPECT_EQ(moduleTextFromBuffer(words), moduleTextFromBuffer(reserialized));
        EXPECT_EQ((*original)->getAttr("jeff.strings"), originalStrings);
    }
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
