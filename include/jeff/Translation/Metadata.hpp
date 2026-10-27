#pragma once

#include <capnp/list.h>
#include <jeff.capnp.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <string>

namespace mlir::jeff {

/**
 * @brief Encode text as a metadata payload with a Cap'n Proto Text root.
 *
 * Returns an owned dense i8 array for a metadata entry's `value` field. Empty text
 * and embedded NUL bytes are preserved. Errors are diagnosed and return failure.
 */
FailureOr<DenseI8ArrayAttr> serializeTextMetadata(Location location, llvm::StringRef text);

/**
 * @brief Decode a text metadata payload into an owned string attribute.
 *
 * The payload must contain one complete Cap'n Proto message with a Text root.
 * Invalid framing or incompatible payload types are diagnosed and return failure.
 * A null root decodes as empty text.
 */
FailureOr<StringAttr> deserializeTextMetadata(Location location, DenseI8ArrayAttr value);

/**
 * @brief Get metadata attached to a jeff program element.
 *
 * Entries are dictionaries with a string `name` and a dense i8 array `value` holding a
 * complete Cap'n Proto message with an `AnyPointer` root. Missing metadata yields a null
 * attribute. Invalid storage is diagnosed and yields failure.
 */
FailureOr<ArrayAttr> getMetadata(Operation* operation);
FailureOr<ArrayAttr> getMetadata(Region& region);
FailureOr<ArrayAttr> getMetadata(Value value);

/**
 * @brief Set metadata attached to a jeff program element.
 *
 * An empty array clears the metadata. Invalid storage is diagnosed and yields failure.
 */
LogicalResult setMetadata(Operation* operation, ArrayAttr metadata);
LogicalResult setMetadata(Region& region, ArrayAttr metadata);
LogicalResult setMetadata(Value value, ArrayAttr metadata);

/**
 * @brief Deserialize metadata entries without interpreting their keys or payload schemas.
 *
 * Invalid metadata is diagnosed and yields failure. Cap'n Proto errors do not escape
 * as exceptions.
 */
FailureOr<ArrayAttr> deserializeMetadata(Location location,
                                         capnp::List<::jeff::Meta>::Reader metadata,
                                         llvm::ArrayRef<std::string> strings);

/**
 * @brief Serialize metadata entries without interpreting their keys or payload schemas.
 *
 * Invalid metadata is diagnosed and yields failure. Discard the builder on failure.
 * Cap'n Proto errors do not escape as exceptions.
 */
LogicalResult serializeMetadata(Location location, ArrayAttr metadata,
                                capnp::List<::jeff::Meta>::Builder builder,
                                const llvm::StringMap<uint16_t>& strings);

} // namespace mlir::jeff
