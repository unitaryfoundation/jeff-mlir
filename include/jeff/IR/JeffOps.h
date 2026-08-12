// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "jeff/IR/JeffInterfaces.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Bufferization/IR/AllocationOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

//===----------------------------------------------------------------------===//
// jeff trait declarations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// jeff ops declarations.
//===----------------------------------------------------------------------===//

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "jeff/IR/JeffAttributes.h.inc"
#define GET_OP_CLASSES
#include "jeff/IR/JeffOps.h.inc"
