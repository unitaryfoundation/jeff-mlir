#pragma once

#include <capnp/common.h>
#include <capnp/message.h>
#include <kj/array.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

/**
 * @brief Serialize an MLIR module containing a jeff program into a fresh Cap'n Proto message
 * @param moduleOp The MLIR module to serialize.
 * @param message The fresh message to encode into.
 *
 * @details
 * The caller owns the message and its segments. Keep the message alive and unchanged while reading
 * its root or the views returned by `getSegmentsForOutput()`. Discard the message if encoding
 * fails.
 *
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
void serialize(mlir::ModuleOp moduleOp, capnp::MessageBuilder& message);

/**
 * @brief Serialize an MLIR module containing a jeff program into a memory buffer.
 * @param moduleOp The MLIR module to serialize.
 * @return An owned memory buffer containing the serialized jeff module.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
kj::Array<capnp::word> serialize(mlir::ModuleOp moduleOp);

/**
 * @brief Serialize an MLIR module containing a jeff program into a .jeff file.
 * @param moduleOp The MLIR module to serialize.
 * @param path The path to the .jeff file.
 * @return Success if the file was written, failure otherwise.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
mlir::LogicalResult serializeToFile(mlir::ModuleOp moduleOp, llvm::StringRef path);
