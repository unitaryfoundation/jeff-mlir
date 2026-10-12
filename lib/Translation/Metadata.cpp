#include "jeff/Translation/Metadata.hpp"

#include <capnp/any.h>
#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <jeff.capnp.h>
#include <kj/array.h>
#include <kj/common.h>
#include <kj/exception.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace mlir::jeff {
namespace {

template <typename Callback>
auto withCapnpDiagnostics(Location location, llvm::StringRef action, Callback callback) {
    using Result = decltype(callback());
    Result result = failure();
    auto exception = kj::runCatchingExceptions([&] { result = callback(); });
    KJ_IF_MAYBE (error, exception) {
        return Result(emitError(location)
                      << "Failed to " << action << " metadata: " << error->getDescription().cStr());
    }
    return result;
}

FailureOr<ArrayAttr> getArray(Operation* operation, llvm::StringRef name) {
    auto attribute = operation->getAttr(name);
    if (attribute && !llvm::isa<ArrayAttr>(attribute)) {
        return operation->emitError("Metadata storage must be an array");
    }
    return llvm::dyn_cast_or_null<ArrayAttr>(attribute);
}

FailureOr<ArrayAttr> getElement(Location location, ArrayAttr array, unsigned index, unsigned size) {
    if (!array || array.empty()) {
        return ArrayAttr{};
    }
    if (array.size() != size) {
        return emitError(location, "Metadata storage size does not match its owner");
    }
    auto element = llvm::dyn_cast<ArrayAttr>(array[index]);
    if (!element) {
        return emitError(location, "Metadata storage entries must be arrays");
    }
    return element;
}

FailureOr<ArrayAttr> replaceElement(Location location, ArrayAttr array, unsigned index,
                                    unsigned size, ArrayAttr element) {
    llvm::SmallVector<Attribute> elements(size, ArrayAttr::get(location.getContext(), {}));
    if (array && !array.empty()) {
        if (array.size() != size) {
            return emitError(location, "Metadata storage size does not match its owner");
        }
        elements.assign(array.begin(), array.end());
    }
    elements[index] = element;
    return ArrayAttr::get(location.getContext(), elements);
}

LogicalResult checkMetadata(Location location, ArrayAttr metadata) {
    if (!metadata) {
        return success();
    }
    for (auto entry : metadata) {
        auto fields = llvm::dyn_cast<DictionaryAttr>(entry);
        if (!fields) {
            return emitError(location, "Metadata entries must be dictionaries");
        }
        if (fields.size() != 2) {
            return emitError(location, "Metadata entries must contain only name and value");
        }
        if (!fields.getAs<StringAttr>("name")) {
            return emitError(location, "Metadata names must be strings");
        }
        if (!fields.getAs<DenseI8ArrayAttr>("value")) {
            return emitError(location, "Metadata payloads must be byte arrays");
        }
    }
    return success();
}

FailureOr<DenseI8ArrayAttr> deserializeMetadataPayload(Location location,
                                                       capnp::AnyPointer::Reader value) {
    return withCapnpDiagnostics(location, "deserialize", [&]() -> FailureOr<DenseI8ArrayAttr> {
        capnp::MallocMessageBuilder message;
        message.getRoot<capnp::AnyPointer>().set(value);
        auto words = capnp::messageToFlatArray(message);
        auto bytes = words.asBytes();
        return DenseI8ArrayAttr::get(
            location.getContext(),
            llvm::ArrayRef(reinterpret_cast<const int8_t*>(bytes.begin()), bytes.size()));
    });
}

LogicalResult serializeMetadataPayload(Location location, DenseI8ArrayAttr value,
                                       capnp::AnyPointer::Builder builder) {
    auto bytes = value.asArrayRef();
    if (bytes.size() % sizeof(capnp::word) != 0) {
        return emitError(location,
                         "Metadata payload size must be a multiple of the Cap'n Proto word size");
    }
    return withCapnpDiagnostics(location, "serialize", [&]() -> LogicalResult {
        auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
        if (!bytes.empty()) {
            std::memcpy(words.begin(), bytes.data(), bytes.size());
        }
        capnp::FlatArrayMessageReader message(words);
        if (message.getEnd() != words.end()) {
            return emitError(location, "Metadata payload contains trailing data");
        }
        builder.set(message.getRoot<capnp::AnyPointer>());
        return success();
    });
}

} // namespace

FailureOr<ArrayAttr> getMetadata(Operation* operation) {
    auto results = getArray(operation, "jeff.result_metadata");
    auto regions = getArray(operation, "jeff.region_metadata");
    auto arguments = getArray(operation, "jeff.argument_metadata");
    if (failed(results) || failed(regions) || failed(arguments)) {
        return failure();
    }
    if ((*results && !results->empty() && results->size() != operation->getNumResults()) ||
        (*regions && !regions->empty() && regions->size() != operation->getNumRegions()) ||
        (*arguments && !arguments->empty() && arguments->size() != operation->getNumRegions())) {
        return operation->emitError("Metadata storage size does not match its owner");
    }
    if (llvm::isa<func::FuncOp>(operation) && *arguments && !arguments->empty()) {
        return operation->emitError(
            "Function arguments must use their argument metadata dictionaries");
    }
    for (auto& region : operation->getRegions()) {
        auto entries = getElement(operation->getLoc(), *arguments, region.getRegionNumber(),
                                  operation->getNumRegions());
        if (failed(entries)) {
            return failure();
        }
        const auto size = region.empty() ? 0 : region.getNumArguments();
        if (*entries && !entries->empty() && entries->size() != size) {
            return operation->emitError("Metadata storage size does not match its owner");
        }
    }
    auto metadata = getArray(operation, "jeff.metadata");
    if (failed(metadata) || failed(checkMetadata(operation->getLoc(), *metadata))) {
        return failure();
    }
    return *metadata;
}

FailureOr<ArrayAttr> getMetadata(Region& region) {
    auto* owner = region.getParentOp();
    auto array = getArray(owner, "jeff.region_metadata");
    if (failed(array)) {
        return failure();
    }
    auto metadata =
        getElement(owner->getLoc(), *array, region.getRegionNumber(), owner->getNumRegions());
    if (failed(metadata) || failed(checkMetadata(owner->getLoc(), *metadata))) {
        return failure();
    }
    return *metadata;
}

FailureOr<ArrayAttr> getMetadata(Value value) {
    FailureOr<ArrayAttr> metadata = ArrayAttr{};
    if (auto result = llvm::dyn_cast<OpResult>(value)) {
        auto* owner = result.getOwner();
        auto array = getArray(owner, "jeff.result_metadata");
        if (failed(array)) {
            return failure();
        }
        metadata =
            getElement(value.getLoc(), *array, result.getResultNumber(), owner->getNumResults());
    } else {
        auto argument = llvm::cast<BlockArgument>(value);
        auto& region = *argument.getOwner()->getParent();
        auto* owner = region.getParentOp();
        if (auto function = llvm::dyn_cast<func::FuncOp>(owner)) {
            auto attribute = function.getArgAttr(argument.getArgNumber(), "jeff.metadata");
            if (attribute && !llvm::isa<ArrayAttr>(attribute)) {
                return emitError(value.getLoc(), "Metadata storage must be an array");
            }
            metadata = llvm::dyn_cast_or_null<ArrayAttr>(attribute);
        } else {
            auto array = getArray(owner, "jeff.argument_metadata");
            if (failed(array)) {
                return failure();
            }
            auto arguments = getElement(value.getLoc(), *array, region.getRegionNumber(),
                                        owner->getNumRegions());
            if (failed(arguments)) {
                return failure();
            }
            metadata = getElement(value.getLoc(), *arguments, argument.getArgNumber(),
                                  argument.getOwner()->getNumArguments());
        }
    }
    if (failed(metadata) || failed(checkMetadata(value.getLoc(), *metadata))) {
        return failure();
    }
    return *metadata;
}

LogicalResult setMetadata(Operation* operation, ArrayAttr metadata) {
    if (failed(checkMetadata(operation->getLoc(), metadata))) {
        return failure();
    }
    if (metadata && !metadata.empty()) {
        operation->setAttr("jeff.metadata", metadata);
    } else {
        operation->removeAttr("jeff.metadata");
    }
    return success();
}

LogicalResult setMetadata(Region& region, ArrayAttr metadata) {
    auto* owner = region.getParentOp();
    if (failed(checkMetadata(owner->getLoc(), metadata))) {
        return failure();
    }
    if ((!metadata || metadata.empty()) && !owner->hasAttr("jeff.region_metadata")) {
        return success();
    }
    auto array = getArray(owner, "jeff.region_metadata");
    if (failed(array)) {
        return failure();
    }
    if (!metadata) {
        metadata = ArrayAttr::get(owner->getContext(), {});
    }
    auto updated = replaceElement(owner->getLoc(), *array, region.getRegionNumber(),
                                  owner->getNumRegions(), metadata);
    if (failed(updated)) {
        return failure();
    }
    owner->setAttr("jeff.region_metadata", *updated);
    return success();
}

LogicalResult setMetadata(Value value, ArrayAttr metadata) {
    if (failed(checkMetadata(value.getLoc(), metadata))) {
        return failure();
    }
    if (!metadata) {
        metadata = ArrayAttr::get(value.getContext(), {});
    }
    if (auto result = llvm::dyn_cast<OpResult>(value)) {
        auto* owner = result.getOwner();
        if (metadata.empty() && !owner->hasAttr("jeff.result_metadata")) {
            return success();
        }
        auto array = getArray(owner, "jeff.result_metadata");
        if (failed(array)) {
            return failure();
        }
        auto updated = replaceElement(value.getLoc(), *array, result.getResultNumber(),
                                      owner->getNumResults(), metadata);
        if (failed(updated)) {
            return failure();
        }
        owner->setAttr("jeff.result_metadata", *updated);
    } else {
        auto argument = llvm::cast<BlockArgument>(value);
        auto& region = *argument.getOwner()->getParent();
        auto* owner = region.getParentOp();
        if (auto function = llvm::dyn_cast<func::FuncOp>(owner)) {
            if (!metadata.empty()) {
                function.setArgAttr(argument.getArgNumber(), "jeff.metadata", metadata);
            } else {
                function.removeArgAttr(argument.getArgNumber(), "jeff.metadata");
            }
        } else {
            if (metadata.empty() && !owner->hasAttr("jeff.argument_metadata")) {
                return success();
            }
            auto array = getArray(owner, "jeff.argument_metadata");
            if (failed(array)) {
                return failure();
            }
            auto arguments = getElement(value.getLoc(), *array, region.getRegionNumber(),
                                        owner->getNumRegions());
            if (failed(arguments)) {
                return failure();
            }
            auto updated = replaceElement(value.getLoc(), *arguments, argument.getArgNumber(),
                                          argument.getOwner()->getNumArguments(), metadata);
            if (failed(updated)) {
                return failure();
            }
            updated = replaceElement(value.getLoc(), *array, region.getRegionNumber(),
                                     owner->getNumRegions(), *updated);
            if (failed(updated)) {
                return failure();
            }
            owner->setAttr("jeff.argument_metadata", *updated);
        }
    }
    return success();
}

FailureOr<ArrayAttr> deserializeMetadata(Location location,
                                         capnp::List<::jeff::Meta>::Reader metadata,
                                         llvm::ArrayRef<std::string> strings) {
    return withCapnpDiagnostics(location, "deserialize", [&]() -> FailureOr<ArrayAttr> {
        llvm::SmallVector<Attribute> entries;
        for (auto entry : metadata) {
            if (entry.getName() >= strings.size()) {
                return emitError(location, "Metadata name index out of bounds");
            }
            auto payload = deserializeMetadataPayload(location, entry.getValue());
            if (failed(payload)) {
                return failure();
            }
            entries.push_back(DictionaryAttr::get(
                location.getContext(),
                {
                    {
                        StringAttr::get(location.getContext(), "name"),
                        StringAttr::get(location.getContext(), strings[entry.getName()]),
                    },
                    {StringAttr::get(location.getContext(), "value"), *payload},
                }));
        }
        return ArrayAttr::get(location.getContext(), entries);
    });
}

LogicalResult serializeMetadata(Location location, ArrayAttr metadata,
                                capnp::List<::jeff::Meta>::Builder builder,
                                const llvm::StringMap<uint16_t>& strings) {
    if (failed(checkMetadata(location, metadata))) {
        return failure();
    }
    if (!metadata) {
        return success();
    }
    if (builder.size() != metadata.size()) {
        return emitError(location, "Metadata builder size does not match its entries");
    }
    return withCapnpDiagnostics(location, "serialize", [&]() -> LogicalResult {
        for (unsigned i = 0; i < metadata.size(); ++i) {
            auto fields = llvm::cast<DictionaryAttr>(metadata[i]);
            auto name = fields.getAs<StringAttr>("name").getValue();
            auto string = strings.find(name);
            if (string == strings.end()) {
                return emitError(location, "Metadata name missing from string table");
            }
            builder[i].setName(string->second);
            if (failed(serializeMetadataPayload(location, fields.getAs<DenseI8ArrayAttr>("value"),
                                                builder[i].getValue()))) {
                return failure();
            }
        }
        return success();
    });
}

} // namespace mlir::jeff
