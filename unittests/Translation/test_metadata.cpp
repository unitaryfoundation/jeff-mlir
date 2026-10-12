#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Metadata.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/any.h>
#include <capnp/blob.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

class MetadataTest : public ::testing::Test {
  protected:
    mlir::MLIRContext context;

    void SetUp() override {
        context.loadDialect<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();
    }

    mlir::OwningOpRef<mlir::ModuleOp> makeModule() {
        return mlir::parseSourceString<mlir::ModuleOp>(R"MLIR(
          module attributes {
            jeff.strings = ["main", "helper", "empty"], jeff.entrypoint = 0 : ui16,
            jeff.tool = "test", jeff.toolVersion = "test", jeff.version = 0 : ui16,
            jeff.versionMinor = 3 : ui16, jeff.versionPatch = 1 : ui16
          } {
            func.func @main(%theta: f64, %phi: f64) -> (f64, f64) {
              %result = jeff.while : (f64, f64) -> (f64) args(%a = %theta, %b = %phi) {
                %false = jeff.int_const1(false) : i1
                jeff.yield %false, %a : i1, f64
              } args(%next) {
                jeff.yield %next, %next : f64, f64
              }
              %x, %y = func.call @helper(%result, %phi) : (f64, f64) -> (f64, f64)
              func.call @empty() : () -> ()
              return %x, %y : f64, f64
            }
            func.func @helper(%theta: f64, %phi: f64) -> (f64, f64) {
              return %theta, %phi : f64, f64
            }
            func.func @empty() {
              return
            }
          }
        )MLIR",
                                                       &context);
    }

    mlir::ArrayAttr textMetadata(llvm::StringRef name) {
        capnp::MallocMessageBuilder message;
        auto entry = message.initRoot<capnp::List<jeff::Meta>>(1)[0];
        entry.setName(0);
        entry.getValue().setAs<capnp::Text>(name.str());
        const std::array<std::string, 1> names{"mqt.input_name"};
        return mlir::jeff::deserializeMetadata(&context, message.getRoot<capnp::List<jeff::Meta>>(),
                                               names);
    }

    mlir::ArrayAttr arbitraryMetadata() {
        capnp::MallocMessageBuilder message(1);
        auto entries = message.initRoot<capnp::List<jeff::Meta>>(6);
        const std::array<std::string, 5> names{
            std::string("vendor.text") + '\0' + "suffix",
            "vendor.data",
            "vendor.list",
            "vendor.struct",
            "vendor.null",
        };
        for (unsigned i = 0; i < 5; ++i) {
            entries[i].setName(i);
        }
        entries[0].getValue().setAs<capnp::Text>(std::string("φ") + '\0' + "suffix");
        auto bytes = entries[1].getValue().initAs<capnp::Data>(3);
        bytes[0] = 0;
        bytes[1] = 128;
        bytes[2] = 255;
        auto list = entries[2].getValue().initAs<capnp::List<uint32_t>>(2);
        list.set(0, 7);
        list.set(1, 42);
        auto structure = entries[3].getValue().initAs<jeff::Meta>();
        structure.setName(17);
        structure.getValue().setAs<capnp::Text>("nested");
        entries[5].setName(0);
        entries[5].getValue().setAs<capnp::Text>("duplicate key");
        return mlir::jeff::deserializeMetadata(&context, entries.asReader(), names);
    }
};

} // namespace

TEST_F(MetadataTest, PreservesAllAttachmentPointsAndOpaquePayloads) {
    auto original = makeModule();
    ASSERT_TRUE(original);
    auto metadata = arbitraryMetadata();
    auto strings = (*original)->getAttr("jeff.strings");
    original->walk([&](mlir::Operation* operation) {
        if (!operation->hasTrait<mlir::OpTrait::IsTerminator>()) {
            mlir::jeff::setMetadata(operation, metadata);
        }
        for (auto result : operation->getResults()) {
            mlir::jeff::setMetadata(result, metadata);
        }
        if (!llvm::isa<mlir::ModuleOp>(operation)) {
            for (auto& region : operation->getRegions()) {
                mlir::jeff::setMetadata(region, metadata);
                for (auto argument : region.getArguments()) {
                    mlir::jeff::setMetadata(argument, metadata);
                }
            }
        }
    });

    auto words = serialize(*original);
    ASSERT_GT(words.size(), 0);
    auto decoded = deserialize(&context, words);
    ASSERT_TRUE(decoded);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*decoded)));
    decoded->walk([&](mlir::Operation* operation) {
        if (!operation->hasTrait<mlir::OpTrait::IsTerminator>()) {
            EXPECT_EQ(mlir::jeff::getMetadata(operation), metadata);
        }
        for (auto result : operation->getResults()) {
            EXPECT_EQ(mlir::jeff::getMetadata(result), metadata);
        }
        if (!llvm::isa<mlir::ModuleOp>(operation)) {
            for (auto& region : operation->getRegions()) {
                EXPECT_EQ(mlir::jeff::getMetadata(region), metadata);
                for (auto argument : region.getArguments()) {
                    EXPECT_EQ(mlir::jeff::getMetadata(argument), metadata);
                }
            }
        }
    });
    EXPECT_EQ((*original)->getAttr("jeff.strings"), strings);

    // The opaque payload also survives printing and parsing the MLIR representation.
    std::string text;
    llvm::raw_string_ostream output(text);
    decoded->print(output);
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(parsed);
    auto reserialized = serialize(*parsed);
    capnp::FlatArrayMessageReader before(words);
    capnp::FlatArrayMessageReader after(reserialized);
    auto beforeEntries = before.getRoot<jeff::Module>().getMetadata();
    auto afterEntries = after.getRoot<jeff::Module>().getMetadata();
    ASSERT_EQ(afterEntries.size(), beforeEntries.size());
    for (unsigned i = 0; i < beforeEntries.size(); ++i) {
        EXPECT_EQ(afterEntries[i].getName(), beforeEntries[i].getName());
        EXPECT_TRUE(afterEntries[i].getValue() == beforeEntries[i].getValue());
    }
}

TEST_F(MetadataTest, ReadsInputMetadataBySourceIndex) {
    capnp::MallocMessageBuilder message;
    auto jeffModule = message.initRoot<jeff::Module>();
    auto strings = jeffModule.initStrings(2);
    strings.set(0, "main");
    strings.set(1, "mqt.input_name");
    auto definition = jeffModule.initFunctions(1)[0].initDefinition();
    auto values = definition.initValues(3);
    for (auto value : values) {
        value.initType().setFloat(jeff::FloatPrecision::FLOAT64);
    }
    auto theta = values[0].initMetadata(1)[0];
    theta.setName(1);
    theta.getValue().setAs<capnp::Text>("theta");
    auto phi = values[1].initMetadata(1)[0];
    phi.setName(1);
    phi.getValue().setAs<capnp::Text>("phi");
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
    auto function = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_FALSE(mlir::jeff::getMetadata(function.getArgument(0)));
    EXPECT_EQ(mlir::jeff::getMetadata(function.getArgument(1)), textMetadata("theta"));
    EXPECT_EQ(mlir::jeff::getMetadata(function.getArgument(2)), textMetadata("phi"));
    EXPECT_FALSE(function.getArgAttr(1, "jeff.input_name"));

    auto reserialized = serialize(*decoded);
    capnp::FlatArrayMessageReader restored(reserialized);
    auto result = restored.getRoot<jeff::Module>().getFunctions()[0].getDefinition();
    EXPECT_EQ(result.getValues()[result.getBody().getSources()[1]]
                  .getMetadata()[0]
                  .getValue()
                  .getAs<capnp::Text>(),
              "theta");
    EXPECT_EQ(result.getValues()[result.getBody().getSources()[2]]
                  .getMetadata()[0]
                  .getValue()
                  .getAs<capnp::Text>(),
              "phi");
}

TEST_F(MetadataTest, ReplacesAndClearsMetadata) {
    auto moduleOp = makeModule();
    ASSERT_TRUE(moduleOp);
    auto function = *moduleOp->getOps<mlir::func::FuncOp>().begin();
    auto theta = function.getArgument(0);
    auto phi = function.getArgument(1);
    mlir::jeff::setMetadata(theta, textMetadata("theta"));
    mlir::jeff::setMetadata(phi, textMetadata("phi"));
    mlir::jeff::setMetadata(theta, textMetadata("new_theta"));
    mlir::jeff::setMetadata(phi, {});
    auto& region = function.getBody();
    auto& operation = region.front().front();
    mlir::jeff::setMetadata(&operation, textMetadata("operation"));
    mlir::jeff::setMetadata(&operation, {});
    mlir::jeff::setMetadata(region, textMetadata("region"));
    mlir::jeff::setMetadata(region, {});
    mlir::jeff::setMetadata(operation.getResult(0), textMetadata("result"));
    mlir::jeff::setMetadata(operation.getResult(0), {});
    auto& nested = operation.getRegion(0);
    mlir::jeff::setMetadata(nested.getArgument(0), textMetadata("argument"));
    mlir::jeff::setMetadata(nested.getArgument(0), {});

    auto words = serialize(*moduleOp);
    ASSERT_GT(words.size(), 0);
    auto decoded = deserialize(&context, words);
    ASSERT_TRUE(decoded);
    auto restored = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_EQ(mlir::jeff::getMetadata(restored.getArgument(0)), textMetadata("new_theta"));
    EXPECT_FALSE(mlir::jeff::getMetadata(restored.getArgument(1)));
    EXPECT_FALSE(mlir::jeff::getMetadata(restored.getBody()));
    auto& restoredOperation = restored.getBody().front().front();
    EXPECT_FALSE(mlir::jeff::getMetadata(&restoredOperation));
    EXPECT_FALSE(mlir::jeff::getMetadata(restoredOperation.getResult(0)));
    EXPECT_FALSE(mlir::jeff::getMetadata(restoredOperation.getRegion(0).getArgument(0)));
}

TEST_F(MetadataTest, DiagnosesMalformedPayloadsAndContinuesSerializing) {
    auto moduleOp = makeModule();
    ASSERT_TRUE(moduleOp);
    auto function = *moduleOp->getOps<mlir::func::FuncOp>().begin();
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });
    for (unsigned size : {0U, 1U, 8U}) {
        SCOPED_TRACE(size);
        llvm::SmallVector<int8_t> bytes(size, 0);
        auto metadata = mlir::ArrayAttr::get(
            &context,
            {
                mlir::DictionaryAttr::get(&context,
                                          {
                                              {
                                                  mlir::StringAttr::get(&context, "name"),
                                                  mlir::StringAttr::get(&context, "vendor.payload"),
                                              },
                                              {
                                                  mlir::StringAttr::get(&context, "value"),
                                                  mlir::DenseI8ArrayAttr::get(&context, bytes),
                                              },
                                          }),
            });
        mlir::jeff::setMetadata(function.getArgument(0), metadata);
        diagnostic.clear();
        EXPECT_EQ(serialize(*moduleOp).size(), 0);
        EXPECT_NE(diagnostic.find("Failed to serialize jeff"), std::string::npos) << diagnostic;
    }
    mlir::jeff::setMetadata(function.getArgument(0), textMetadata("theta"));
    auto words = serialize(*moduleOp);
    EXPECT_GT(words.size(), 0);
    EXPECT_TRUE(deserialize(&context, words));
}

TEST_F(MetadataTest, DiagnosesInvalidStorage) {
    auto moduleOp = makeModule();
    ASSERT_TRUE(moduleOp);
    auto function = *moduleOp->getOps<mlir::func::FuncOp>().begin();
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });
    function->setAttr("jeff.metadata", mlir::StringAttr::get(&context, "wrong type"));
    EXPECT_EQ(serialize(*moduleOp).size(), 0);
    EXPECT_NE(diagnostic.find("Metadata storage must be an array"), std::string::npos);
    function->removeAttr("jeff.metadata");

    function->setAttr("jeff.result_metadata",
                      mlir::ArrayAttr::get(&context, {textMetadata("extra")}));
    EXPECT_EQ(serialize(*moduleOp).size(), 0);
    function->removeAttr("jeff.result_metadata");
}

TEST_F(MetadataTest, RejectsMetadataWithoutAnExchangeAttachmentPoint) {
    auto moduleOp = makeModule();
    ASSERT_TRUE(moduleOp);
    auto function = *moduleOp->getOps<mlir::func::FuncOp>().begin();
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });
    auto& terminator = function.getBody().front().back();
    mlir::jeff::setMetadata(&terminator, textMetadata("terminator"));
    EXPECT_EQ(serialize(*moduleOp).size(), 0);
    EXPECT_NE(diagnostic.find("Terminator metadata has no jeff operation"), std::string::npos);
    mlir::jeff::setMetadata(&terminator, {});
    mlir::jeff::setMetadata(moduleOp->getBodyRegion(), textMetadata("module body"));
    EXPECT_EQ(serialize(*moduleOp).size(), 0);
    EXPECT_NE(diagnostic.find("Module body metadata has no jeff region"), std::string::npos);
}

TEST_F(MetadataTest, RejectsInvalidNamesAtEveryAttachmentPoint) {
    auto original = makeModule();
    ASSERT_TRUE(original);
    auto words = serialize(*original);
    capnp::FlatArrayMessageReader reader(words);
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });
    for (unsigned site = 0; site < 5; ++site) {
        SCOPED_TRACE(site);
        capnp::MallocMessageBuilder message;
        message.setRoot(reader.getRoot<jeff::Module>());
        auto jeffModule = message.getRoot<jeff::Module>();
        auto function = jeffModule.getFunctions()[0];
        auto definition = function.getDefinition();
        auto body = definition.getBody();
        std::array<jeff::Meta::Builder, 5> entries{
            jeffModule.initMetadata(1)[0],
            function.initMetadata(1)[0],
            body.initMetadata(1)[0],
            body.getOperations()[0].initMetadata(1)[0],
            definition.getValues()[0].initMetadata(1)[0],
        };
        entries[site].setName(jeffModule.getStrings().size());
        auto invalid = capnp::messageToFlatArray(message);
        diagnostic.clear();
        EXPECT_FALSE(deserialize(&context, invalid));
        EXPECT_NE(diagnostic.find("Metadata name index out of bounds"), std::string::npos)
            << diagnostic;
    }
}

TEST_F(MetadataTest, InvalidPayloadDoesNotTruncateOutputFile) {
    auto moduleOp = makeModule();
    ASSERT_TRUE(moduleOp);
    auto metadata = mlir::ArrayAttr::get(
        &context,
        {
            mlir::DictionaryAttr::get(&context,
                                      {
                                          {
                                              mlir::StringAttr::get(&context, "name"),
                                              mlir::StringAttr::get(&context, "vendor.payload"),
                                          },
                                          {
                                              mlir::StringAttr::get(&context, "value"),
                                              mlir::DenseI8ArrayAttr::get(&context, {1}),
                                          },
                                      }),
        });
    mlir::jeff::setMetadata(*moduleOp, metadata);
    const auto path = std::filesystem::path(::testing::TempDir()) / "invalid_metadata_output.jeff";
    std::ofstream(path) << "previous contents";
    EXPECT_TRUE(mlir::failed(serializeToFile(*moduleOp, path.string())));
    std::ifstream input(path);
    std::string contents;
    std::getline(input, contents);
    EXPECT_EQ(contents, "previous contents");
    input.close();
    std::filesystem::remove(path);
}

TEST_F(MetadataTest, PreservesMetadataThroughDirectMessageExchange) {
    auto original = makeModule();
    ASSERT_TRUE(original);
    auto metadata = arbitraryMetadata();
    mlir::jeff::setMetadata(*original, metadata);
    auto function = *original->getOps<mlir::func::FuncOp>().begin();
    mlir::jeff::setMetadata(function.getArgument(0), textMetadata("theta"));

    mlir::OwningOpRef<mlir::ModuleOp> decoded;
    {
        capnp::MallocMessageBuilder message(1, capnp::AllocationStrategy::FIXED_SIZE);
        serialize(*original, message);
        capnp::SegmentArrayMessageReader reader(message.getSegmentsForOutput());
        decoded = deserialize(&context, reader.getRoot<jeff::Module>());
    }
    ASSERT_TRUE(decoded);
    EXPECT_EQ(mlir::jeff::getMetadata(*decoded), metadata);
    auto decodedFunction = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_EQ(mlir::jeff::getMetadata(decodedFunction.getArgument(0)), textMetadata("theta"));
    auto words = serialize(*decoded);
    EXPECT_GT(words.size(), 0);
}
