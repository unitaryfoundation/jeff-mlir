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

#ifndef JEFF_DIALECT
#define JEFF_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// jeff dialect.
//===----------------------------------------------------------------------===//

def JeffDialect : Dialect {
    let summary = "A jeff dialect.";
    let description = [{
        The `jeff` dialect represents the `jeff` quantum program exchange format
        in MLIR. It provides quantum types and operations alongside classical
        computation and structured control flow.

        Function arguments may carry a `jeff.input_name` string attribute for
        their source-level name. It does not change execution semantics.
        Serialization stores it on the function body's source value as metadata
        named `jeff.input_name` with a Cap'n Proto `Text` payload. Deserialization
        restores it on each defined function's arguments. Files without this
        metadata remain readable and their arguments remain unnamed.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "jeff";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::mlir::jeff";

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// jeff dialect types.
//===----------------------------------------------------------------------===//

class Jeff_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<JeffDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : Jeff_Type<"Qubit", "qubit"> {
    let summary = "A value-semantic qubit (state).";
}
def QuregType : Jeff_Type<"Qureg", "qureg"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";
    let parameters = (ins DefaultValuedParameter<"int64_t", "::mlir::ShapedType::kDynamic">:$length);
    let hasCustomAssemblyFormat = 1;
}

def SupportedIntType : AnyTypeOf<[I1, I8, I16, I32, I64]>;

def SupportedFloatType : AnyTypeOf<[F32, F64]>;

def SupportedFloatArrayAttrType : AnyAttrOf<[DenseF32ArrayAttr, DenseF64ArrayAttr]>;

//===----------------------------------------------------------------------===//
// jeff dialect base operation.
//===----------------------------------------------------------------------===//

class Jeff_Op<string mnemonic, list<Trait> traits = []> :
        Op<JeffDialect, mnemonic, traits>;

#endif // JEFF_DIALECT
