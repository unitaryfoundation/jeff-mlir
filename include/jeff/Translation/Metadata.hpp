#pragma once

#include <capnp/list.h>
#include <jeff.capnp.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <string>

namespace mlir::jeff {

/**
 * @brief Get metadata attached to a jeff program element.
 *
 * Entries are dictionaries with a string `name` and a dense i8 array `value` holding a
 * complete Cap'n Proto message with an `AnyPointer` root.
 */
ArrayAttr getMetadata(Operation* operation);
ArrayAttr getMetadata(Region& region);
ArrayAttr getMetadata(Value value);

/**
 * @brief Set metadata attached to a jeff program element.
 *
 * An empty array clears the metadata.
 */
void setMetadata(Operation* operation, ArrayAttr metadata);
void setMetadata(Region& region, ArrayAttr metadata);
void setMetadata(Value value, ArrayAttr metadata);

/**
 * @brief Deserialize metadata entries without interpreting their keys or payload schemas.
 */
ArrayAttr deserializeMetadata(MLIRContext* context, capnp::List<::jeff::Meta>::Reader metadata,
                              llvm::ArrayRef<std::string> strings);

/**
 * @brief Serialize metadata entries without interpreting their keys or payload schemas.
 */
void serializeMetadata(ArrayAttr metadata, capnp::List<::jeff::Meta>::Builder builder,
                       const llvm::StringMap<uint16_t>& strings);

} // namespace mlir::jeff
