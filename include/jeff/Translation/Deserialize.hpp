#pragma once

#include <capnp/common.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

/**
 * @brief Deserialize a memory buffer containing a serialized .jeff module into an MLIR module.
 * @param context The MLIR context to use for the deserialization.
 * @param buffer A memory buffer containing the serialized jeff module.
 * @return An owning reference to the deserialized MLIR module, or null after an import error has
 * been diagnosed. Deserializer and Cap'n Proto errors do not escape this boundary as exceptions.
 */
mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              kj::ArrayPtr<capnp::word> buffer);

/**
 * @brief Deserialize a .jeff file into an MLIR module.
 * @param context The MLIR context to use for the deserialization.
 * @param path The path to the .jeff file.
 * @return An owning reference to the deserialized MLIR module, or null after an import error has
 * been diagnosed. Deserializer and Cap'n Proto errors do not escape this boundary as exceptions.
 */
mlir::OwningOpRef<mlir::ModuleOp> deserializeFromFile(mlir::MLIRContext* context,
                                                      llvm::StringRef path);
