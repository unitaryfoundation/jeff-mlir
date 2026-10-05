#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"
#include "test_utils.hpp"

#include <capnp/any.h>
#include <capnp/blob.h>
#include <capnp/list.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <filesystem>
#include <string>

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
        EXPECT_EQ(jeff::test::moduleTextFromBuffer(words),
                  jeff::test::moduleTextFromBuffer(reserialized));
        EXPECT_EQ((*original)->getAttr("jeff.strings"), originalStrings);
    }
}
