#pragma once

#include <capnp/common.h>
#include <kj/array.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

/**
 * @brief Serialize an MLIR module containing a jeff program into a memory buffer.
 * @param module The MLIR module to serialize.
 * @return An owned memory buffer containing the serialized jeff module.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
kj::Array<capnp::word> serialize(mlir::ModuleOp module);

/**
 * @brief Serialize an MLIR module containing a jeff program into a .jeff file.
 * @param module The MLIR module to serialize.
 * @param path The path to the .jeff file.
 * @return Success if the file was written, failure otherwise.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
mlir::LogicalResult serializeToFile(mlir::ModuleOp module, llvm::StringRef path);
