#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"
#include "test_utils.hpp"

#include <capnp/any.h>
#include <capnp/blob.h>
#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <kj/common.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>

namespace fs = std::filesystem;

namespace {

class DeserializeErrorTest : public ::testing::Test {
  protected:
    mlir::MLIRContext context;
    std::string diagnostic;
    fs::path input;
    mlir::ScopedDiagnosticHandler handler{&context, [this](mlir::Diagnostic& error) {
                                              diagnostic = error.str();
                                              return mlir::success();
                                          }};

    void SetUp() override {
        context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();
        input = fs::path(::testing::TempDir()) /
                (std::string("jeff_deserialize_") +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".jeff");
    }

    void TearDown() override { fs::remove(input); }

    void checkRejected(capnp::MallocMessageBuilder& message, const char* expected) {
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
    }
};

jeff::Value::Builder initNamedInputModule(capnp::MallocMessageBuilder& message) {
    auto module = message.initRoot<jeff::Module>();
    auto strings = module.initStrings(2);
    strings.set(0, "main");
    strings.set(1, "jeff.input_name");
    auto definition = module.initFunctions(1)[0].initDefinition();
    auto value = definition.initValues(1)[0];
    value.initType().setFloat(jeff::FloatPrecision::FLOAT64);
    auto body = definition.initBody();
    body.initSources(1).set(0, 0);
    body.initTargets(0);
    body.initOperations(0);
    auto metadata = value.initMetadata(1)[0];
    metadata.setName(1);
    metadata.getValue().setAs<capnp::Text>("theta");
    return value;
}

} // namespace

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

TEST_F(DeserializeErrorTest, RejectsMissingFunctions) {
    capnp::MallocMessageBuilder missingFunctions;
    auto missing = missingFunctions.initRoot<jeff::Module>();
    missing.setVersionMinor(3);
    checkRejected(missingFunctions, "No functions found");
}

TEST_F(DeserializeErrorTest, RejectsUndefinedValues) {
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
}

TEST_F(DeserializeErrorTest, RejectsDuplicateFunctions) {
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
}

TEST_F(DeserializeErrorTest, RejectsOutOfBoundsInputMetadataNames) {
    capnp::MallocMessageBuilder message;
    auto value = initNamedInputModule(message);
    value.getMetadata()[0].setName(2);
    checkRejected(message, "Input metadata name index out of bounds");
}

TEST_F(DeserializeErrorTest, RejectsMissingInputNames) {
    capnp::MallocMessageBuilder message;
    auto value = initNamedInputModule(message);
    value.getMetadata()[0].getValue().clear();
    checkRejected(message, "Missing input name metadata value");
}

TEST_F(DeserializeErrorTest, RejectsNonTextInputNames) {
    capnp::MallocMessageBuilder message;
    auto value = initNamedInputModule(message);
    value.getMetadata()[0].getValue().initAs<capnp::List<uint32_t>>(1);
    checkRejected(message, "Failed to deserialize jeff");
}

TEST_F(DeserializeErrorTest, RejectsDuplicateInputNames) {
    capnp::MallocMessageBuilder message;
    auto value = initNamedInputModule(message);
    for (auto duplicate : value.initMetadata(2)) {
        duplicate.setName(1);
        duplicate.getValue().setAs<capnp::Text>("theta");
    }
    checkRejected(message, "Duplicate input name metadata");
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
    )MLIR",
         }) {
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
        EXPECT_EQ(jeff::test::moduleTextFromBuffer(bytes),
                  jeff::test::moduleTextFromBuffer(reserialized));
    }
}
