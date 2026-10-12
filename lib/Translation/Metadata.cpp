#include "jeff/Translation/Metadata.hpp"

#include <capnp/any.h>
#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <jeff.capnp.h>
#include <kj/array.h>
#include <kj/debug.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace mlir::jeff {
namespace {

ArrayAttr getArray(Operation* operation, llvm::StringRef name) {
    auto attribute = operation->getAttr(name);
    KJ_REQUIRE(!attribute || llvm::isa<ArrayAttr>(attribute), "Metadata storage must be an array") {
    }
    return llvm::dyn_cast_or_null<ArrayAttr>(attribute);
}

ArrayAttr getElement(ArrayAttr array, unsigned index, unsigned size) {
    if (!array || array.empty()) {
        return {};
    }
    KJ_REQUIRE(array.size() == size, "Metadata storage size does not match its owner") {}
    auto element = llvm::dyn_cast<ArrayAttr>(array[index]);
    KJ_REQUIRE(static_cast<bool>(element), "Metadata storage entries must be arrays") {}
    return element;
}

ArrayAttr replaceElement(MLIRContext* context, ArrayAttr array, unsigned index, unsigned size,
                         ArrayAttr element) {
    llvm::SmallVector<Attribute> elements(size, ArrayAttr::get(context, {}));
    if (array && !array.empty()) {
        KJ_REQUIRE(array.size() == size, "Metadata storage size does not match its owner") {}
        elements.assign(array.begin(), array.end());
    }
    elements[index] = element;
    return ArrayAttr::get(context, elements);
}

void checkMetadata(ArrayAttr metadata) {
    if (!metadata) {
        return;
    }
    for (auto entry : metadata) {
        auto fields = llvm::dyn_cast<DictionaryAttr>(entry);
        KJ_REQUIRE(static_cast<bool>(fields), "Metadata entries must be dictionaries") {}
        KJ_REQUIRE(fields.size() == 2, "Metadata entries must contain only name and value") {}
        KJ_REQUIRE(static_cast<bool>(fields.getAs<StringAttr>("name")),
                   "Metadata names must be strings") {}
        KJ_REQUIRE(static_cast<bool>(fields.getAs<DenseI8ArrayAttr>("value")),
                   "Metadata payloads must be byte arrays") {}
    }
}

DenseI8ArrayAttr deserializeMetadataPayload(MLIRContext* context, capnp::AnyPointer::Reader value) {
    capnp::MallocMessageBuilder message;
    message.getRoot<capnp::AnyPointer>().set(value);
    auto words = capnp::messageToFlatArray(message);
    auto bytes = words.asBytes();
    return DenseI8ArrayAttr::get(
        context, llvm::ArrayRef(reinterpret_cast<const int8_t*>(bytes.begin()), bytes.size()));
}

void serializeMetadataPayload(DenseI8ArrayAttr value, capnp::AnyPointer::Builder builder) {
    auto bytes = value.asArrayRef();
    KJ_REQUIRE(bytes.size() % sizeof(capnp::word) == 0,
               "Metadata payload size must be a multiple of the Cap'n Proto word size") {}
    auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
    if (!bytes.empty()) {
        std::memcpy(words.begin(), bytes.data(), bytes.size());
    }
    capnp::FlatArrayMessageReader message(words);
    KJ_REQUIRE(message.getEnd() == words.end(), "Metadata payload contains trailing data") {}
    builder.set(message.getRoot<capnp::AnyPointer>());
}

} // namespace

ArrayAttr getMetadata(Operation* operation) {
    auto results = getArray(operation, "jeff.result_metadata");
    KJ_REQUIRE(!results || results.empty() || results.size() == operation->getNumResults(),
               "Metadata storage size does not match its owner") {}
    auto regions = getArray(operation, "jeff.region_metadata");
    KJ_REQUIRE(!regions || regions.empty() || regions.size() == operation->getNumRegions(),
               "Metadata storage size does not match its owner") {}
    auto arguments = getArray(operation, "jeff.argument_metadata");
    KJ_REQUIRE(!arguments || arguments.empty() || arguments.size() == operation->getNumRegions(),
               "Metadata storage size does not match its owner") {}
    KJ_REQUIRE(!llvm::isa<func::FuncOp>(operation) || !arguments || arguments.empty(),
               "Function arguments must use their argument metadata dictionaries") {}
    for (auto& region : operation->getRegions()) {
        auto entries = getElement(arguments, region.getRegionNumber(), operation->getNumRegions());
        const auto size = region.empty() ? 0 : region.getNumArguments();
        KJ_REQUIRE(!entries || entries.empty() || entries.size() == size,
                   "Metadata storage size does not match its owner") {}
    }
    auto metadata = getArray(operation, "jeff.metadata");
    checkMetadata(metadata);
    return metadata;
}

ArrayAttr getMetadata(Region& region) {
    auto* owner = region.getParentOp();
    auto metadata = getElement(getArray(owner, "jeff.region_metadata"), region.getRegionNumber(),
                               owner->getNumRegions());
    checkMetadata(metadata);
    return metadata;
}

ArrayAttr getMetadata(Value value) {
    ArrayAttr metadata;
    if (auto result = llvm::dyn_cast<OpResult>(value)) {
        auto* owner = result.getOwner();
        metadata = getElement(getArray(owner, "jeff.result_metadata"), result.getResultNumber(),
                              owner->getNumResults());
    } else {
        auto argument = llvm::cast<BlockArgument>(value);
        auto& region = *argument.getOwner()->getParent();
        auto* owner = region.getParentOp();
        if (auto function = llvm::dyn_cast<func::FuncOp>(owner)) {
            auto attribute = function.getArgAttr(argument.getArgNumber(), "jeff.metadata");
            KJ_REQUIRE(!attribute || llvm::isa<ArrayAttr>(attribute),
                       "Metadata storage must be an array") {}
            metadata = llvm::dyn_cast_or_null<ArrayAttr>(attribute);
        } else {
            auto arguments = getElement(getArray(owner, "jeff.argument_metadata"),
                                        region.getRegionNumber(), owner->getNumRegions());
            metadata = getElement(arguments, argument.getArgNumber(),
                                  argument.getOwner()->getNumArguments());
        }
    }
    checkMetadata(metadata);
    return metadata;
}

void setMetadata(Operation* operation, ArrayAttr metadata) {
    if (metadata && !metadata.empty()) {
        operation->setAttr("jeff.metadata", metadata);
    } else {
        operation->removeAttr("jeff.metadata");
    }
}

void setMetadata(Region& region, ArrayAttr metadata) {
    auto* owner = region.getParentOp();
    if ((!metadata || metadata.empty()) && !owner->hasAttr("jeff.region_metadata")) {
        return;
    }
    if (!metadata) {
        metadata = ArrayAttr::get(owner->getContext(), {});
    }
    owner->setAttr("jeff.region_metadata",
                   replaceElement(owner->getContext(), getArray(owner, "jeff.region_metadata"),
                                  region.getRegionNumber(), owner->getNumRegions(), metadata));
}

void setMetadata(Value value, ArrayAttr metadata) {
    if (!metadata) {
        metadata = ArrayAttr::get(value.getContext(), {});
    }
    if (auto result = llvm::dyn_cast<OpResult>(value)) {
        auto* owner = result.getOwner();
        if (metadata.empty() && !owner->hasAttr("jeff.result_metadata")) {
            return;
        }
        owner->setAttr("jeff.result_metadata",
                       replaceElement(owner->getContext(), getArray(owner, "jeff.result_metadata"),
                                      result.getResultNumber(), owner->getNumResults(), metadata));
    } else {
        auto argument = llvm::cast<BlockArgument>(value);
        auto& region = *argument.getOwner()->getParent();
        auto* owner = region.getParentOp();
        if (auto function = llvm::dyn_cast<func::FuncOp>(owner)) {
            if (metadata.empty()) {
                function.removeArgAttr(argument.getArgNumber(), "jeff.metadata");
            } else {
                function.setArgAttr(argument.getArgNumber(), "jeff.metadata", metadata);
            }
        } else {
            if (metadata.empty() && !owner->hasAttr("jeff.argument_metadata")) {
                return;
            }
            auto regions = getArray(owner, "jeff.argument_metadata");
            auto arguments = replaceElement(
                owner->getContext(),
                getElement(regions, region.getRegionNumber(), owner->getNumRegions()),
                argument.getArgNumber(), argument.getOwner()->getNumArguments(), metadata);
            owner->setAttr("jeff.argument_metadata",
                           replaceElement(owner->getContext(), regions, region.getRegionNumber(),
                                          owner->getNumRegions(), arguments));
        }
    }
}

ArrayAttr deserializeMetadata(MLIRContext* context, capnp::List<::jeff::Meta>::Reader metadata,
                              llvm::ArrayRef<std::string> strings) {
    llvm::SmallVector<Attribute> entries;
    for (auto entry : metadata) {
        KJ_REQUIRE(entry.getName() < strings.size(), "Metadata name index out of bounds") {}
        entries.push_back(DictionaryAttr::get(
            context, {
                         {
                             StringAttr::get(context, "name"),
                             StringAttr::get(context, strings[entry.getName()]),
                         },
                         {
                             StringAttr::get(context, "value"),
                             deserializeMetadataPayload(context, entry.getValue()),
                         },
                     }));
    }
    return ArrayAttr::get(context, entries);
}

void serializeMetadata(ArrayAttr metadata, capnp::List<::jeff::Meta>::Builder builder,
                       const llvm::StringMap<uint16_t>& strings) {
    checkMetadata(metadata);
    if (!metadata) {
        return;
    }
    for (unsigned i = 0; i < metadata.size(); ++i) {
        auto fields = llvm::cast<DictionaryAttr>(metadata[i]);
        auto name = fields.getAs<StringAttr>("name").getValue();
        auto string = strings.find(name);
        KJ_REQUIRE(string != strings.end(), "Metadata name missing from string table") {}
        builder[i].setName(string->second);
        serializeMetadataPayload(fields.getAs<DenseI8ArrayAttr>("value"), builder[i].getValue());
    }
}

} // namespace mlir::jeff
