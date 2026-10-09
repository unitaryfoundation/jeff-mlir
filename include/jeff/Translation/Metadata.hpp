#pragma once

#include <capnp/any.h>
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

/// Get or set metadata on the corresponding jeff program element.
ArrayAttr getMetadata(Operation* operation);
ArrayAttr getMetadata(Region& region);
ArrayAttr getMetadata(Value value);
void setMetadata(Operation* operation, ArrayAttr metadata);
void setMetadata(Region& region, ArrayAttr metadata);
void setMetadata(Value value, ArrayAttr metadata);

/// Copy an opaque Cap'n Proto payload into owned MLIR storage or back into a message.
DenseI8ArrayAttr deserializeMetadataPayload(MLIRContext* context, capnp::AnyPointer::Reader value);
void serializeMetadataPayload(DenseI8ArrayAttr value, capnp::AnyPointer::Builder builder);

/// Translate metadata entries without interpreting their keys or payload schemas.
ArrayAttr deserializeMetadata(MLIRContext* context, capnp::List<::jeff::Meta>::Reader metadata,
                              llvm::ArrayRef<std::string> strings);
void serializeMetadata(ArrayAttr metadata, capnp::List<::jeff::Meta>::Builder builder,
                       const llvm::StringMap<uint16_t>& strings);

} // namespace mlir::jeff
