#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Metadata.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/any.h>
#include <capnp/blob.h>
#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
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
#include <utility>

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

    template <typename Owner> mlir::ArrayAttr getMetadata(Owner&& owner) {
        auto metadata = mlir::jeff::getMetadata(std::forward<Owner>(owner));
        EXPECT_TRUE(mlir::succeeded(metadata));
        return mlir::failed(metadata) ? mlir::ArrayAttr{} : *metadata;
    }

    mlir::ArrayAttr textMetadata(llvm::StringRef name) {
        capnp::MallocMessageBuilder message;
        auto entry = message.initRoot<capnp::List<jeff::Meta>>(1)[0];
        entry.setName(0);
        entry.getValue().setAs<capnp::Text>(name.str());
        const std::array<std::string, 1> names{"mqt.input_name"};
        auto metadata = mlir::jeff::deserializeMetadata(
            mlir::UnknownLoc::get(&context), message.getRoot<capnp::List<jeff::Meta>>(), names);
        EXPECT_TRUE(mlir::succeeded(metadata));
        return mlir::failed(metadata) ? mlir::ArrayAttr{} : *metadata;
    }

    mlir::ArrayAttr payloadMetadata(llvm::ArrayRef<int8_t> bytes) {
        mlir::Builder builder(&context);
        return builder.getArrayAttr({
            builder.getDictionaryAttr({
                builder.getNamedAttr("name", builder.getStringAttr("vendor.payload")),
                builder.getNamedAttr("value", builder.getDenseI8ArrayAttr(bytes)),
            }),
        });
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
        auto metadata = mlir::jeff::deserializeMetadata(mlir::UnknownLoc::get(&context),
                                                        entries.asReader(), names);
        EXPECT_TRUE(mlir::succeeded(metadata));
        return mlir::failed(metadata) ? mlir::ArrayAttr{} : *metadata;
    }
};

} // namespace

TEST_F(MetadataTest, TextPayloadRoundTrip) {
    auto location = mlir::UnknownLoc::get(&context);
    for (auto text : {
             llvm::StringRef(""),
             llvm::StringRef("theta suffix").take_front(5),
             llvm::StringRef("φ\0suffix", 9),
         }) {
        auto payload = mlir::jeff::serializeTextMetadata(location, text);
        ASSERT_TRUE(mlir::succeeded(payload));
        auto decoded = mlir::jeff::deserializeTextMetadata(location, *payload);
        ASSERT_TRUE(mlir::succeeded(decoded));
        EXPECT_EQ(decoded->getValue(), text);
        capnp::MallocMessageBuilder message;
        auto entry = message.initRoot<capnp::List<jeff::Meta>>(1);
        auto metadata = payloadMetadata(payload->asArrayRef());
        llvm::StringMap<uint16_t> strings{{"vendor.payload", 0}};
        ASSERT_TRUE(
            mlir::succeeded(mlir::jeff::serializeMetadata(location, metadata, entry, strings)));
        auto nativeText = entry[0].getValue().getAs<capnp::Text>();
        EXPECT_EQ(llvm::StringRef(nativeText.begin(), nativeText.size()), text);
    }
}

TEST_F(MetadataTest, RejectsInvalidTextPayloads) {
    auto location = mlir::UnknownLoc::get(&context);
    mlir::Builder builder(&context);
    auto entries = arbitraryMetadata();
    auto nonText =
        llvm::cast<mlir::DictionaryAttr>(entries[2]).getAs<mlir::DenseI8ArrayAttr>("value");
    auto text = mlir::jeff::serializeTextMetadata(location, "theta");
    ASSERT_TRUE(mlir::succeeded(text));
    llvm::SmallVector<int8_t> trailing(text->asArrayRef());
    trailing.resize(trailing.size() + sizeof(capnp::word), 0);
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& error) {
        diagnostic = error.str();
        return mlir::success();
    });
    for (auto payload : {
             builder.getDenseI8ArrayAttr({}),
             builder.getDenseI8ArrayAttr({0}),
             nonText,
             builder.getDenseI8ArrayAttr(trailing),
         }) {
        diagnostic.clear();
        EXPECT_TRUE(mlir::failed(mlir::jeff::deserializeTextMetadata(location, payload)));
        EXPECT_FALSE(diagnostic.empty());
    }
}

TEST_F(MetadataTest, PreservesAllAttachmentPointsAndOpaquePayloads) {
    auto original = makeModule();
    ASSERT_TRUE(original);
    auto metadata = arbitraryMetadata();
    auto strings = (*original)->getAttr("jeff.strings");
    original->walk([&](mlir::Operation* operation) {
        if (!operation->hasTrait<mlir::OpTrait::IsTerminator>()) {
            ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(operation, metadata)));
        }
        for (auto result : operation->getResults()) {
            ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(result, metadata)));
        }
        if (!llvm::isa<mlir::ModuleOp>(operation)) {
            for (auto& region : operation->getRegions()) {
                ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(region, metadata)));
                for (auto argument : region.getArguments()) {
                    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(argument, metadata)));
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
            EXPECT_EQ(getMetadata(operation), metadata);
        }
        for (auto result : operation->getResults()) {
            EXPECT_EQ(getMetadata(result), metadata);
        }
        if (!llvm::isa<mlir::ModuleOp>(operation)) {
            for (auto& region : operation->getRegions()) {
                EXPECT_EQ(getMetadata(region), metadata);
                for (auto argument : region.getArguments()) {
                    EXPECT_EQ(getMetadata(argument), metadata);
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

TEST_F(MetadataTest, PreservesStringTableIndicesWhenAddingMetadataNames) {
    auto moduleOp = makeModule();
    ASSERT_TRUE(moduleOp);
    const std::array<llvm::StringRef, 6> originalStrings{
        "main", "unused", "helper", "empty", "main", "mqt.input_name",
    };
    mlir::Builder builder(&context);
    auto stringsAttr = builder.getStrArrayAttr(originalStrings);
    (*moduleOp)->setAttr("jeff.strings", stringsAttr);
    auto metadata = arbitraryMetadata();
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(*moduleOp, metadata)));
    auto function = *moduleOp->getOps<mlir::func::FuncOp>().begin();
    ASSERT_TRUE(
        mlir::succeeded(mlir::jeff::setMetadata(function.getArgument(0), textMetadata("theta"))));

    auto words = serialize(*moduleOp);
    ASSERT_GT(words.size(), 0);
    capnp::FlatArrayMessageReader reader(words);
    auto jeffModule = reader.getRoot<jeff::Module>();
    auto strings = jeffModule.getStrings();
    ASSERT_EQ(strings.size(), originalStrings.size() + 5);
    for (unsigned i = 0; i < originalStrings.size(); ++i) {
        EXPECT_EQ(std::string(strings[i].begin(), strings[i].size()), originalStrings[i].str());
    }
    auto entries = jeffModule.getMetadata();
    ASSERT_EQ(entries.size(), 6);
    EXPECT_EQ(entries[0].getName(), originalStrings.size());
    EXPECT_EQ(entries[5].getName(), entries[0].getName());
    auto definition = jeffModule.getFunctions()[0].getDefinition();
    auto argument = definition.getValues()[definition.getBody().getSources()[0]];
    EXPECT_EQ(argument.getMetadata()[0].getName(), 5);
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
    body.initOperations(0);

    auto words = capnp::messageToFlatArray(message);
    auto decoded = deserialize(&context, words);
    ASSERT_TRUE(decoded);
    auto function = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_FALSE(getMetadata(function.getArgument(0)));
    EXPECT_EQ(getMetadata(function.getArgument(1)), textMetadata("theta"));
    EXPECT_EQ(getMetadata(function.getArgument(2)), textMetadata("phi"));
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
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(theta, textMetadata("theta"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(phi, textMetadata("phi"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(theta, textMetadata("new_theta"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(phi, {})));
    auto& region = function.getBody();
    auto& operation = region.front().front();
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(&operation, textMetadata("operation"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(&operation, {})));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(region, textMetadata("region"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(region, {})));
    ASSERT_TRUE(
        mlir::succeeded(mlir::jeff::setMetadata(operation.getResult(0), textMetadata("result"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(operation.getResult(0), {})));
    auto& nested = operation.getRegion(0);
    ASSERT_TRUE(
        mlir::succeeded(mlir::jeff::setMetadata(nested.getArgument(0), textMetadata("argument"))));
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(nested.getArgument(0), {})));

    auto words = serialize(*moduleOp);
    ASSERT_GT(words.size(), 0);
    auto decoded = deserialize(&context, words);
    ASSERT_TRUE(decoded);
    auto restored = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_EQ(getMetadata(restored.getArgument(0)), textMetadata("new_theta"));
    EXPECT_FALSE(getMetadata(restored.getArgument(1)));
    EXPECT_FALSE(getMetadata(restored.getBody()));
    auto& restoredOperation = restored.getBody().front().front();
    EXPECT_FALSE(getMetadata(&restoredOperation));
    EXPECT_FALSE(getMetadata(restoredOperation.getResult(0)));
    EXPECT_FALSE(getMetadata(restoredOperation.getRegion(0).getArgument(0)));
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
        ASSERT_TRUE(mlir::succeeded(
            mlir::jeff::setMetadata(function.getArgument(0), payloadMetadata(bytes))));
        diagnostic.clear();
        EXPECT_EQ(serialize(*moduleOp).size(), 0);
        EXPECT_FALSE(diagnostic.empty());
    }
    ASSERT_TRUE(
        mlir::succeeded(mlir::jeff::setMetadata(function.getArgument(0), textMetadata("theta"))));
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
    EXPECT_TRUE(mlir::failed(mlir::jeff::getMetadata(function.getOperation())));
    EXPECT_EQ(serialize(*moduleOp).size(), 0);
    EXPECT_NE(diagnostic.find("Metadata storage must be an array"), std::string::npos);
    function->removeAttr("jeff.metadata");

    function->setAttr("jeff.result_metadata",
                      mlir::ArrayAttr::get(&context, {textMetadata("extra")}));
    EXPECT_EQ(serialize(*moduleOp).size(), 0);
    EXPECT_TRUE(mlir::failed(mlir::jeff::getMetadata(function.getOperation())));
    function->removeAttr("jeff.result_metadata");

    auto metadata = getMetadata(function.getOperation());
    EXPECT_TRUE(mlir::failed(mlir::jeff::setMetadata(
        function.getOperation(),
        mlir::ArrayAttr::get(&context, {mlir::StringAttr::get(&context, "invalid")}))));
    EXPECT_EQ(getMetadata(function.getOperation()), metadata);
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
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(*moduleOp, payloadMetadata({1}))));
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
    ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(*original, metadata)));
    auto function = *original->getOps<mlir::func::FuncOp>().begin();
    ASSERT_TRUE(
        mlir::succeeded(mlir::jeff::setMetadata(function.getArgument(0), textMetadata("theta"))));

    mlir::OwningOpRef<mlir::ModuleOp> decoded;
    {
        capnp::MallocMessageBuilder message(1, capnp::AllocationStrategy::FIXED_SIZE);
        ASSERT_TRUE(mlir::succeeded(serialize(*original, message)));
        capnp::SegmentArrayMessageReader reader(message.getSegmentsForOutput());
        decoded = deserialize(&context, reader.getRoot<jeff::Module>());
    }
    ASSERT_TRUE(decoded);
    EXPECT_EQ(getMetadata(*decoded), metadata);
    auto decodedFunction = *decoded->getOps<mlir::func::FuncOp>().begin();
    EXPECT_EQ(getMetadata(decodedFunction.getArgument(0)), textMetadata("theta"));
    auto words = serialize(*decoded);
    EXPECT_GT(words.size(), 0);
}
