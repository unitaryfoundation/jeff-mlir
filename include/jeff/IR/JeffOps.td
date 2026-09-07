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

#ifndef JEFF_OPS
#define JEFF_OPS

include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "jeff/IR/JeffDialect.td"
include "jeff/IR/JeffInterfaces.td"

//===----------------------------------------------------------------------===//
// Qubit operations.
//===----------------------------------------------------------------------===//

class Qubit_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [QubitOperation]>;

def QubitAllocOp : Qubit_Op<"qubit_alloc", []> {
    let summary = "Allocates a new qubit in the |0> state.";
    let description = [{}];

    let results = (outs
        QubitType:$out_qubit
    );

    // %q = jeff.qubit_alloc : !jeff.qubit
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
}

def QubitFreeOp : Qubit_Op<"qubit_free", []> {
    let summary = "Frees a qubit.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    // jeff.qubit_free %q : !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type($in_qubit)
    }];
}

def QubitFreeZeroOp : Qubit_Op<"qubit_free_zero", []> {
    let summary = "Frees a qubit in the |0> state.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    // jeff.qubit_free_zero %q : !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type($in_qubit)
    }];
}

def QubitMeasureOp : Qubit_Op<"qubit_measure", []> {
    let summary = "Perform a destructive measurement of a qubit in the computational basis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        I1:$result
    );

    // %result = jeff.qubit_measure %q : i1
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

def QubitMeasureNDOp : Qubit_Op<"qubit_measure_nd", []> {
    let summary = "Perform a non-destructive measurement of a qubit in the computational basis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        QubitType:$out_qubit,
        I1:$result
    );

    // %q_out, %result = jeff.qubit_measure_nd %q_in : !jeff.qubit, i1
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

def QubitResetOp : Qubit_Op<"qubit_reset", [SameOperandsAndResultType]> {
    let summary = "Resets a qubit to the |0> state.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        QubitType:$out_qubit
    );

    // %q_out = jeff.qubit_reset %q_in : !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// Gate operations.
//===----------------------------------------------------------------------===//

class Gate_Op<string mnemonic, list<Trait> traits = []> : Qubit_Op<mnemonic, traits # [MultipleCtrlQubitsGate]>;

def PPROp : Gate_Op<"ppr", [AttrSizedOperandSegments, AttrSizedResultSegments, MultiplePauliGate]> {
    let summary = "Apply an arbitrary Pauli-product rotation gate.";
    let description = [{}];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits,
        Variadic<QubitType>:$in_ctrl_qubits,
        F64:$rotation,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power,
        DenseI32ArrayAttr:$pauli_gates
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::ValueRange":$in_qubits,
             "mlir::ValueRange":$in_ctrl_qubits,
             "mlir::Value":$rotation,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power,
             "mlir::DenseI32ArrayAttr":$pauli_gates
        ),[{
            PPROp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubits=*/ in_qubits,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*rotation=*/ rotation,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power,
            /*pauli_gates=*/ pauli_gates
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.ppr(%rotation, %pauli_gates) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $rotation `,` $pauli_gates `)` attr-dict $in_qubits ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def CustomOp : Gate_Op<"custom", [AttrSizedOperandSegments, AttrSizedResultSegments, MultipleQubitsGate, MultipleParamsGate]> {
    let summary = "Apply an arbitrary quantum gate.";
    let description = [{}];

    let arguments = (ins
        Variadic<QubitType>:$in_target_qubits,
        Variadic<QubitType>:$in_ctrl_qubits,
        Variadic<F64>:$params,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power,
        StrAttr:$name,
        I8Attr:$num_targets,
        I8Attr:$num_params
    );

    let results = (outs
        Variadic<QubitType>:$out_target_qubits,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_target_qubits = number of in_target_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::ValueRange":$in_target_qubits,
             "mlir::ValueRange":$in_ctrl_qubits,
             "mlir::ValueRange":$params,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power,
             "mlir::StringRef":$name,
             "uint8_t":$num_targets,
             "uint8_t":$num_params
        ),[{
            CustomOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_target_qubits),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_target_qubits=*/ in_target_qubits,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*params=*/ params,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power,
            /*name=*/ name,
            /*num_targets=*/ num_targets,
            /*num_params=*/ num_params
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.custom "Rot"(%params) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        $name `(` $params `)` attr-dict $in_target_qubits ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_target_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

//===----------------------------------------------------------------------===//
// Well Known Gate operations.
//===----------------------------------------------------------------------===//

class WellKnownGate_Op<string mnemonic, list<Trait> traits = []> : Gate_Op<mnemonic, traits # [WellKnownGate]>;

def GPhaseOp : WellKnownGate_Op<"gphase", []> {
    let summary = "Global phase operation on the 'vacuum' state (no qubits).";
    let description = [{}];

    let arguments = (ins
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            GPhaseOp::build($_builder, $_state,
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q_out = jeff.gphase(%rotation) ctrls(%q_in) : ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict ( `ctrls` `(` $in_ctrl_qubits^ `)` )? (`:` `ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def IOp : WellKnownGate_Op<"i", []> {
    let summary = "Identity (no-op) gate. Pi radians X rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            IOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.i %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def XOp : WellKnownGate_Op<"x", []> {
    let summary = "Pauli-X (NOT) gate. Pi radians X rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            XOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.x %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def YOp : WellKnownGate_Op<"y", []> {
    let summary = "Pauli-Y gate. Pi radians Y rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            YOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.y %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def ZOp : WellKnownGate_Op<"z", []> {
    let summary = "Pauli-Z gate. Pi radians Z rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            ZOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.z %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def SOp : WellKnownGate_Op<"s", []> {
    let summary = "Root-Z gate. Pi/2 radians Z rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            SOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.s %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def TOp : WellKnownGate_Op<"t", []> {
    let summary = "T gate. Pi/4 radians Z rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            TOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.t %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def R1Op : WellKnownGate_Op<"r1", []> {
    let summary = "Rotation about the |1> state.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            R1Op::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.r1(%rotation) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def RxOp : WellKnownGate_Op<"rx", []> {
    let summary = "Rotation about the X axis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            RxOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.rx(%rotation) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def RyOp : WellKnownGate_Op<"ry", []> {
    let summary = "Rotation about the Y axis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            RyOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.ry(%rotation) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def RzOp : WellKnownGate_Op<"rz", []> {
    let summary = "Rotation about the Z axis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            RzOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.rz(%rotation) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def HOp : WellKnownGate_Op<"h", []> {
    let summary = "Hadamard gate.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            HOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.h %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def UOp : WellKnownGate_Op<"u", []> {
    let summary = "Euler gate.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$theta,
        F64:$phi,
        F64:$lambda,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$theta,
             "mlir::Value":$phi,
             "mlir::Value":$lambda,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            UOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*theta=*/ theta,
            /*phi=*/ phi,
            /*lambda=*/ lambda,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %q0_out, %q1_out = jeff.u(%theta, %phi, %lambda) %q0_in ctrls(%q1_in) : !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        `(` $theta `,` $phi `,` $lambda `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def SwapOp : WellKnownGate_Op<"swap", []> {
    let summary = "Swap gate. Swaps the state of two qubits.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit_one,
        QubitType:$in_qubit_two,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit_one,
             "mlir::Value":$in_qubit_two,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            SwapOp::build($_builder, $_state,
            /*out_qubit_one=*/ in_qubit_one.getType(),
            /*out_qubit_two=*/ in_qubit_two.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit_one=*/ in_qubit_one,
            /*in_qubit_two=*/ in_qubit_two,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    let results = (outs
        QubitType:$out_qubit_one,
        QubitType:$out_qubit_two,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    // %q0_out, %q1_out, %q2_out = jeff.u(%theta, %phi, %lambda) %q0_in %q1_in ctrls(%q2_in) : !jeff.qubit !jeff.qubit ctrls !jeff.qubit
    let assemblyFormat = [{
        attr-dict $in_qubit_one $in_qubit_two ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit_one) type($out_qubit_two) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

//===----------------------------------------------------------------------===//
// Qureg operations.
//===----------------------------------------------------------------------===//

class Qureg_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [QuregOperation]>;

def QuregAllocOp : Qureg_Op<"qureg_alloc", []> {
    let summary = "Allocates a new qubit register given a number of qubits in the |0> state.";
    let description = [{}];

    let arguments = (ins
        I32:$num_qubits
    );

    let results = (outs
        QuregType:$qreg
    );

    // %qreg = jeff.qureg_alloc(%num_qubits) : !jeff.qureg
    let assemblyFormat = [{
        `(` $num_qubits `)` attr-dict `:` type(results)
    }];
}

def QuregFreeOp : Qureg_Op<"qureg_free", []> {
    let summary = "Frees a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$qreg
    );

    // jeff.qureg_free %qreg : !jeff.qureg
    let assemblyFormat = [{
        $qreg attr-dict `:` type($qreg)
    }];
}

def QuregFreeZeroOp : Qureg_Op<"qureg_free_zero", []> {
    let summary = "Frees a qubit register, assuming that all qubits are in the |0> state.";
    let description = [{}];

    let arguments = (ins
        QuregType:$qreg
    );

    // jeff.qureg_free_zero %qreg : !jeff.qureg
    let assemblyFormat = [{
        $qreg attr-dict `:` type($qreg)
    }];
}

def QuregExtractIndexOp : Qureg_Op<"qureg_extract_index", [TypesMatchWith<"in_qreg type matches out_qreg type", "in_qreg", "out_qreg", "$_self">]> {
    let summary = "Extracts a single qubit from a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32:$index
    );

    let results = (outs
        QuregType:$out_qreg,
        QubitType:$out_qubit
    );

    // %qreg_in, %q = jeff.qureg_extract_index(%index) %qreg_in : !jeff.qureg, !jeff.qubit
    let assemblyFormat = [{
        `(` $index `)` $in_qreg attr-dict `:` functional-type(operands, results)
    }];
}

def QuregInsertIndexOp : Qureg_Op<"qureg_insert_index", [TypesMatchWith<"in_qreg type matches out_qreg type", "in_qreg", "out_qreg", "$_self">]> {
    let summary = "Insert a single qubit into a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32:$index,
        QubitType:$in_qubit
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %qreg_out = jeff.qureg_insert_index(%index) %qreg_in %q : !jeff.qureg
    let assemblyFormat = [{
        `(` $index `)` $in_qreg $in_qubit attr-dict `:` functional-type(operands, results)
    }];
}

def QuregExtractSliceOp : Qureg_Op<"qureg_extract_slice", [TypesMatchWith<"in_qreg type matches out_qreg type", "in_qreg", "out_qreg", "$_self">]> {
    let summary = "Extract a slice of qubits from a qubit register given a range of indices.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32:$start,
        I32:$length
    );

    let results = (outs
        QuregType:$out_qreg,
        QuregType:$new_qreg
    );

    // %qreg_out, %qreg = jeff.qureg_extract_slice(%start, %length) %qreg_in : !jeff.qureg, !jeff.qureg
    let assemblyFormat = [{
        `(` $start `,` $length `)` $in_qreg attr-dict `:` functional-type(operands, results)
    }];
}

def QuregInsertSliceOp : Qureg_Op<"qureg_insert_slice", [TypesMatchWith<"in_qreg type matches out_qreg type", "in_qreg", "out_qreg", "$_self">]> {
    let summary = "Insert a slice of qubits into a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32:$start,
        QuregType:$new_qreg
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %qreg_out = jeff.qureg_insert_slice(%start) %qreg_in %qreg : !jeff.qureg
    let assemblyFormat = [{
        `(` $start `)` $in_qreg $new_qreg attr-dict `:` functional-type(operands, results)
    }];
}

def QuregLengthOp : Qureg_Op<"qureg_length", [TypesMatchWith<"in_qreg type matches out_qreg type", "in_qreg", "out_qreg", "$_self">]> {
    let summary = "Returns the length of the qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg
    );

    let results = (outs
        QuregType:$out_qreg,
        I32:$length
    );

    // %qreg_out, %length = jeff.qureg_length %qreg_in : !jeff.qureg, i32
    let assemblyFormat = [{
        $in_qreg attr-dict `:` functional-type(operands, results)
    }];
}

def QuregSplitOp : Qureg_Op<"qureg_split", []> {
    let summary = "Splits a qubit register into two qubit registers at a given index.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32:$index
    );

    let results = (outs
        QuregType:$out_qreg_one,
        QuregType:$out_qreg_two
    );

    // %qreg1, %qreg2 = jeff.qureg_split(%index) %qreg0 : !jeff.qureg, !jeff.qureg
    let assemblyFormat = [{
        `(` $index `)` $in_qreg attr-dict `:` functional-type(operands, results)
    }];
}

def QuregJoinOp : Qureg_Op<"qureg_join", []> {
    let summary = "Joins together two qubit registers into a single qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg_one,
        QuregType:$in_qreg_two
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %qreg2 = jeff.qureg_join %qreg0 %qreg1 : !jeff.qureg
    let assemblyFormat = [{
        $in_qreg_one $in_qreg_two attr-dict `:` functional-type(operands, results)
    }];
}

def QuregCreateOp : Qureg_Op<"qureg_create", []> {
    let summary = "Creates a qubit register from a variable number of input qubits.";
    let description = [{}];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %qreg = jeff.qureg_create %q0, %q1 : !jeff.qureg
    let assemblyFormat = [{
        $in_qubits attr-dict `:` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// jeff dialect enums for numeric ops.
//===----------------------------------------------------------------------===//

def IntUnaryOperation : I32EnumAttr<"IntUnaryOperation",
    "Integer unary operations",
    [
        I32EnumAttrCase<"_not",   0>,
        I32EnumAttrCase<"_abs",   1>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def IntBinaryOperation : I32EnumAttr<"IntBinaryOperation",
    "Integer binary operations",
    [
        I32EnumAttrCase<"_add",   0>,
        I32EnumAttrCase<"_sub",   1>,
        I32EnumAttrCase<"_mul",   2>,
        I32EnumAttrCase<"_divS",  3>,
        I32EnumAttrCase<"_divU",  4>,
        I32EnumAttrCase<"_pow",   5>,
        I32EnumAttrCase<"_and",   6>,
        I32EnumAttrCase<"_or",    7>,
        I32EnumAttrCase<"_xor",   8>,
        I32EnumAttrCase<"_minS",  9>,
        I32EnumAttrCase<"_minU",  10>,
        I32EnumAttrCase<"_maxS",  11>,
        I32EnumAttrCase<"_maxU",  12>,
        I32EnumAttrCase<"_remS",  13>,
        I32EnumAttrCase<"_remU",  14>,
        I32EnumAttrCase<"_shl",   15>,
        I32EnumAttrCase<"_shr",   16>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def IntComparisonOperation : I32EnumAttr<"IntComparisonOperation",
    "Integer comparison operations",
    [
        I32EnumAttrCase<"_eq",   0>,
        I32EnumAttrCase<"_ltS",   1>,
        I32EnumAttrCase<"_lteS",   2>,
        I32EnumAttrCase<"_ltU",  3>,
        I32EnumAttrCase<"_lteU",  4>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatUnaryOperation : I32EnumAttr<"FloatUnaryOperation",
    "Float unary operations",
    [
        I32EnumAttrCase<"_sqrt",  0>,
        I32EnumAttrCase<"_abs",   1>,
        I32EnumAttrCase<"_ceil",  2>,
        I32EnumAttrCase<"_floor", 3>,
        I32EnumAttrCase<"_exp",   4>,
        I32EnumAttrCase<"_log",   5>,
        I32EnumAttrCase<"_sin",   6>,
        I32EnumAttrCase<"_cos",   7>,
        I32EnumAttrCase<"_tan",   8>,
        I32EnumAttrCase<"_asin",  9>,
        I32EnumAttrCase<"_acos",  10>,
        I32EnumAttrCase<"_atan",  11>,
        I32EnumAttrCase<"_sinh",  12>,
        I32EnumAttrCase<"_cosh",  13>,
        I32EnumAttrCase<"_tanh",  14>,
        I32EnumAttrCase<"_asinh", 15>,
        I32EnumAttrCase<"_acosh", 16>,
        I32EnumAttrCase<"_atanh", 17>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatBinaryOperation : I32EnumAttr<"FloatBinaryOperation",
    "Float binary operations",
    [
        I32EnumAttrCase<"_add",   0>,
        I32EnumAttrCase<"_sub",   1>,
        I32EnumAttrCase<"_mul",   2>,
        I32EnumAttrCase<"_div",   7>,
        I32EnumAttrCase<"_pow",   3>,
        I32EnumAttrCase<"_atan2", 4>,
        I32EnumAttrCase<"_max",   5>,
        I32EnumAttrCase<"_min",   6>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatComparisonOperation : I32EnumAttr<"FloatComparisonOperation",
    "Float comparison operations",
    [
        I32EnumAttrCase<"_eq",   0>,
        I32EnumAttrCase<"_lt",   1>,
        I32EnumAttrCase<"_lte",   2>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatIsOperation : I32EnumAttr<"FloatIsOperation",
    "Float Is operations",
    [
        I32EnumAttrCase<"_isNan",  0>,
        I32EnumAttrCase<"_isInf",  1>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// jeff dialect attributes for numeric ops.
//===----------------------------------------------------------------------===//

def IntUnaryOperationAttr : EnumAttr<JeffDialect, IntUnaryOperation, "int_unary_operation">;

def IntBinaryOperationAttr : EnumAttr<JeffDialect, IntBinaryOperation, "int_binary_operation">;

def IntComparisonOperationAttr : EnumAttr<JeffDialect, IntComparisonOperation, "int_comparison_operation">;

def FloatUnaryOperationAttr : EnumAttr<JeffDialect, FloatUnaryOperation, "float_unary_operation">;

def FloatBinaryOperationAttr : EnumAttr<JeffDialect, FloatBinaryOperation, "float_binary_operation">;

def FloatComparisonOperationAttr : EnumAttr<JeffDialect, FloatComparisonOperation, "float_comparison_operation">;

def FloatIsOperationAttr : EnumAttr<JeffDialect, FloatIsOperation, "float_is_operation">;

//===----------------------------------------------------------------------===//
// Int operations.
//===----------------------------------------------------------------------===//

class Int_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [IntOperation]>;

def IntConst1Op : Int_Op<"int_const1", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I1Attr:$val
    );

    let results = (outs
        I1:$constant
    );

    // %0 = jeff.int_const1(true) : i1
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst8Op : Int_Op<"int_const8", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I8Attr:$val
    );

    let results = (outs
        I8:$constant
    );

    // %0 = jeff.int_const8(3) : i8
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst16Op : Int_Op<"int_const16", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I16Attr:$val
    );

    let results = (outs
        I16:$constant
    );

    // %0 = jeff.int_const16(3) : i16
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst32Op : Int_Op<"int_const32", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I32Attr:$val
    );

    let results = (outs
        I32:$constant
    );

    // %0 = jeff.int_const32(3) : i32
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst64Op : Int_Op<"int_const64", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I64Attr:$val
    );

    let results = (outs
        I64:$constant
    );

    // %0 = jeff.int_const64(3) : i64
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntUnaryOp : Int_Op<"int_unary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a,
        IntUnaryOperationAttr:$op
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %1 = jeff.int_unary_op [0] (%0) : i64 -> i64
    let assemblyFormat = [{
        `[` $op `]` `(` $a `)` attr-dict `:` type($a) `->` type($b)
    }];
}

def IntBinaryOp : Int_Op<"int_binary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a,
        SupportedIntType:$b,
        IntBinaryOperationAttr:$op
    );

    let results = (outs
        SupportedIntType:$c
    );

    // %2 = jeff.int_binary_op [0] (%0, %1) : i64, i64 -> i64
    let assemblyFormat = [{
        `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) `,` type($b) `->` type($c)
    }];

    let hasCanonicalizer = 1;
}

def IntComparisonOp : Int_Op<"int_comp_op", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a,
        SupportedIntType:$b,
        IntComparisonOperationAttr:$op
    );

    let results = (outs
        I1:$c
    );

    // %2 = jeff.int_comp_op [0] (%0, %1) : i64, i64 -> i1
    let assemblyFormat = [{
        `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) `,` type($b) `->` type($c)
    }];
}

def IntSelectOp : Int_Op<"int_select", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I1:$condition,
        SupportedIntType:$a,
        SupportedIntType:$b
    );

    let results = (outs
        SupportedIntType:$c
    );

    // %c = jeff.int_select $condition, $a, %b : i64, i64 -> i64
    let assemblyFormat = [{
        $condition `,` $a `,` $b attr-dict `:` type($a) `,` type($b) `->` type($c)
    }];
}

def IntExtSOp : Int_Op<"int_ext_s", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %b = jeff.int_ext_s $a : i32 -> i64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def IntExtUOp : Int_Op<"int_ext_u", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %b = jeff.int_ext_u $a : i32 -> i64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def IntTruncOp : Int_Op<"int_trunc", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %b = jeff.int_trunc $a : i64 -> i32
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def IntToFloatSOp : Int_Op<"int_to_float_s", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a
    );

    let results = (outs
        SupportedFloatType:$b
    );

    // %b = jeff.int_to_float_s $a : i64 -> f64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def IntToFloatUOp : Int_Op<"int_to_float_u", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a
    );

    let results = (outs
        SupportedFloatType:$b
    );

    // %b = jeff.int_to_float_u $a : i64 -> f64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

//===----------------------------------------------------------------------===//
// IntArray operations.
//===----------------------------------------------------------------------===//

class IntArray_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [IntArrayOperation]>;

def IntArrayConst1Op : IntArray_Op<"int_array_const1", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseBoolArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I1]>:$out_array
    );

    // %0 = jeff.int_array_const1([false, false, true]) : tensor<?xi1>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst8Op : IntArray_Op<"int_array_const8", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI8ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I8]>:$out_array
    );

    // %0 = jeff.int_array_const8([3, 5, 7]) : tensor<?xi8>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst16Op : IntArray_Op<"int_array_const16", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI16ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I16]>:$out_array
    );

    // %0 = jeff.int_array_const16([3, 5, 7]) : tensor<?xi16>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst32Op : IntArray_Op<"int_array_const32", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI32ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I32]>:$out_array
    );

    // %0 = jeff.int_array_const32([3, 5, 7]) : tensor<?xi32>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst64Op : IntArray_Op<"int_array_const64", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI64ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I64]>:$out_array
    );

    // %0 = jeff.int_array_const64([3, 5, 7]) : tensor<?xi64>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayZeroOp : IntArray_Op<"int_array_zero", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I32:$length
    );

    let results = (outs
        1DTensorOf<[I1, I8, I16, I32, I64]>:$out_array
    );

    // %0 = jeff.int_array_zero(%length) : tensor<?xi64>
    let assemblyFormat = [{
        `(` $length `)` attr-dict `:` type(results)
    }];
}

def IntArrayGetIndexOp : IntArray_Op<"int_array_get_index", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[I1, I8, I16, I32, I64]>:$in_array,
        I32:$index
    );

    let results = (outs
        SupportedIntType:$value
    );

    // %value = jeff.int_array_get_index(%index) %array : i1, tensor<?xi32> -> i32
    let assemblyFormat = [{
        `(` $index `)` $in_array attr-dict `:` type($index) `,` type($in_array) `->` type(results)
    }];
}

def IntArraySetIndexOp : IntArray_Op<"int_array_set_index", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[I1, I8, I16, I32, I64]>:$in_array,
        I32:$index,
        SupportedIntType:$value
    );

    let results = (outs
        1DTensorOf<[I1, I8, I16, I32, I64]>:$out_array
    );

    // %array_out = jeff.int_array_set_index(%index) %array_in %value : i1, tensor<?xi32>, i32 -> tensor<?xi32>
    let assemblyFormat = [{
        `(` $index `)` $in_array $value attr-dict `:` type($index) `,` type($in_array) `,` type($value) `->` type($out_array)
    }];
}

def IntArrayLengthOp : IntArray_Op<"int_array_length", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[I1, I8, I16, I32, I64]>:$in_array
    );

    let results = (outs
        I32:$length
    );

    // %length = jeff.int_array_length %array_on : tensor<?xi16> -> i32
    let assemblyFormat = [{
        $in_array attr-dict `:` type($in_array) `->` type($length)
    }];
}

def IntArrayCreateOp : IntArray_Op<"int_array_create", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        Variadic<SupportedIntType>:$in_array
    );

    let results = (outs
        1DTensorOf<[I1, I8, I16, I32, I64]>:$out_array
    );

    // %array = jeff.int_array_create %0, %1, %2 : i64, i64, i64 -> tensor<?xi64>
    let assemblyFormat = [{
        $in_array attr-dict `:` type($in_array) `->` type($out_array)
    }];
}

//===----------------------------------------------------------------------===//
// Float operations.
//===----------------------------------------------------------------------===//

class Float_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [FloatOperation]>;

def FloatConst32Op : Float_Op<"float_const32", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        F32Attr:$val
    );

    let results = (outs
        F32:$constant
    );

    // %0 = jeff.float_const32(0.3) : f32
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def FloatConst64Op : Float_Op<"float_const64", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        F64Attr:$val
    );

    let results = (outs
        F64:$constant
    );

    // %0 = jeff.float_const64(0.3) : f64
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def FloatUnaryOp : Float_Op<"float_unary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        FloatUnaryOperationAttr:$op
    );

    let results = (outs
        SupportedFloatType:$b
    );

    // %1 = jeff.float_unary_op [0] (%0) : f64 -> f64
    let assemblyFormat = [{
       `[` $op `]` `(` $a `)` attr-dict `:` type($a) `->` type($b)
    }];
}

def FloatBinaryOp : Float_Op<"float_binary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        SupportedFloatType:$b,
        FloatBinaryOperationAttr:$op
    );

    let results = (outs
        SupportedFloatType:$c
    );

    // %2 = jeff.float_binary_op [0] (%0, %1) : f64, f64 -> f64
    let assemblyFormat = [{
       `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) `,` type($b) `->` type(results)
    }];
}

def FloatComparisonOp : Float_Op<"float_comp_op", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        SupportedFloatType:$b,
        FloatComparisonOperationAttr:$op
    );

    let results = (outs
        I1:$c
    );

    // %2 = jeff.float_comp_op [0] (%0, %1) : f64, f64 -> i1
    let assemblyFormat = [{
       `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) `,` type($b) `->` type($c)
    }];
}

def FloatIsOp : Float_Op<"float_is_op", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        FloatIsOperationAttr:$op
    );

    let results = (outs
        I1:$b
    );

    // %1 = jeff.float_is_op [0] (%0) : f64 -> i1
    let assemblyFormat = [{
       `[` $op `]` `(` $a `)` attr-dict `:` type($a) `->` type($b)
    }];
}

def FloatSelectOp : Float_Op<"float_select", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I1:$condition,
        SupportedFloatType:$a,
        SupportedFloatType:$b
    );

    let results = (outs
        SupportedFloatType:$c
    );

    // %c = jeff.float_select $condition, $a, %b : f64, f64 -> f64
    let assemblyFormat = [{
        $condition `,` $a `,` $b attr-dict `:` type($a) `,` type($b) `->` type($c)
    }];
}

def FloatExtOp : Float_Op<"float_ext", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a
    );

    let results = (outs
        SupportedFloatType:$b
    );

    // %b = jeff.float_ext $a : f32 -> f64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def FloatTruncOp : Float_Op<"float_trunc", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a
    );

    let results = (outs
        SupportedFloatType:$b
    );

    // %b = jeff.float_trunc $a : f64 -> f32
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def FloatToSIntOp : Float_Op<"float_to_s_int", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %b = jeff.float_to_s_int $a : f64 -> i64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

def FloatToUIntOp : Float_Op<"float_to_u_int", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %b = jeff.float_to_u_int $a : f64 -> i64
    let assemblyFormat = [{
        $a attr-dict `:` type($a) `->` type($b)
    }];
}

//===----------------------------------------------------------------------===//
// FloatArray operations.
//===----------------------------------------------------------------------===//

class FloatArray_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [FloatArrayOperation]>;

def FloatArrayConst32Op : FloatArray_Op<"float_array_const32", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseF32ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[F32]>:$out_array
    );

    // %0 = jeff.float_array_const32([0.1, 0.2, 0.3]) : tensor<?xf32>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def FloatArrayConst64Op : FloatArray_Op<"float_array_const64", [Pure]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseF64ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[F64]>:$out_array
    );

    // %0 = jeff.float_array_const64([0.1, 0.2, 0.3]) : tensor<?xf64>
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def FloatArrayZeroOp : FloatArray_Op<"float_array_zero", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I32:$length
    );

    let results = (outs
        1DTensorOf<[F32, F64]>:$out_array
    );

    // %0 = jeff.float_array_zero(%length) : tensor<?xf64>
    let assemblyFormat = [{
        `(` $length `)` attr-dict `:` type(results)
    }];
}

def FloatArrayGetIndexOp : FloatArray_Op<"float_array_get_index", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[F32, F64]>:$in_array,
        I32:$index
    );

    let results = (outs
        SupportedFloatType:$value
    );

    // %value = jeff.float_array_get_index(%index) %array : i32, tensor<?xf64> -> f64
    let assemblyFormat = [{
        `(` $index `)` $in_array attr-dict `:` type($index) `,` type($in_array) `->` type($value)
    }];
}

def FloatArraySetIndexOp : FloatArray_Op<"float_array_set_index", []> {
    let summary = "Set the value of an integer array at a given index.";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[F32, F64]>:$in_array,
        I32:$index,
        SupportedFloatType:$value
    );

    let results = (outs
        1DTensorOf<[F32, F64]>:$out_array
    );

    // %array_out = jeff.float_array_set_index(%index) %array_in %value : i32, tensor<?xf64>, f64 -> tensor<?xf64>
    let assemblyFormat = [{
        `(` $index `)` $in_array $value attr-dict `:` type($index) `,` type($in_array) `,` type($value) `->` type($out_array)
    }];
}

def FloatArrayLengthOp : FloatArray_Op<"float_array_length", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[F32, F64]>:$in_array
    );

    let results = (outs
        I32:$length
    );

    // %1 = jeff.float_array_length %0 : tensor<?xf64> -> i32
    let assemblyFormat = [{
        $in_array attr-dict `:` type($in_array) `->` type($length)
    }];
}

def FloatArrayCreateOp : FloatArray_Op<"float_array_create", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        Variadic<SupportedFloatType>:$in_array
    );

    let results = (outs
        1DTensorOf<[F32, F64]>:$out_array
    );

    // %array = jeff.float_array_create %0, %1, %2 : f64, f64, f64 -> tensor<?xf64>
    let assemblyFormat = [{
        $in_array attr-dict `:` type($in_array) `->` type($out_array)
    }];
}

//===----------------------------------------------------------------------===//
// Structured control flow operations.
//===----------------------------------------------------------------------===//

class SCF_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [SCFOperation]>;

def SwitchOp : SCF_Op<"switch", [IsolatedFromAbove, SingleBlockImplicitTerminator<"jeff::YieldOp">]> {
    let summary = "Multi-way branch on an integer selector";
    let description = [{
      The `jeff.switch` operation dispatches on an integer selector to one of
      several branch regions or, if no case matches, an optional default
      region. Branch labels are positional: the i-th branch region runs when
      `selection` equals `i`. If `selection` does not match any branch label,
      control flows to the `default` region.

      The selector operand `selection` must be a `SupportedIntType` (one of
      `i1`, `i8`, `i16`, `i32`, `i64`).

      The op carries a list of in-values (`in_values`) shared across all
      branches and the default region: each region declares its own local
      block arguments, bound positionally to the in-values on entry. Each
      region must terminate with a `jeff.yield` whose operands provide the
      op's results for the selected branch.

      The default region is optional. When present, it covers any selector
      value not matched by an explicit branch.

      Unlike a `for` or `while` loop, a `switch` has no iteration semantics:
      it dispatches once. The op's result count and types are therefore not
      related to the in-value count and types — branches may yield any
      number of values of any type, provided every branch (and the default,
      if present) yields the same shape.

      Schematic syntax:
      ```
      jeff.switch ( $selection, $in_values ) : ( $selection_type, $in_value_types )
                                            -> ( $result_types )
      ( case $i args ( $local_names ) { $branch_body } )*
      ( default args ( $local_names ) { $default_body } )?
      ```

      Example:
      ```mlir
      %r1, %r2 = jeff.switch (%sel, %a, %b) : (i32, i32, i64) -> (i32, i64)
      case 0 args(%x, %y) {
        jeff.yield %x, %y : i32, i64
      }
      case 1 args(%x, %y) {
        // ... compute %next_x, %next_y ...
        jeff.yield %next_x, %next_y : i32, i64
      }
      default args(%x, %y) {
        jeff.yield %x, %y : i32, i64
      }
      ```

      Implicit inferences in the syntax:
      - Each region's `args(%x, %y)` lists *only the local block-argument
        names*, not assignments. The block arguments are bound positionally
        to the op's `in_values` (in the example above, `%x` to `%a`, `%y`
        to `%b`) — the operands are already named in the op's header.
      - Block-argument types are not written: they are inferred from the
        in-value types in the header.

      Invariants:
      - `selection` is a `SupportedIntType`.
      - Case labels are contiguous integers starting at 0: the i-th
        branch region is labeled `case i`. The parser enforces this so
        that print-then-parse is faithful.
      - Every region's block argument count and types match `in_values`.
      - Every region's yielded value count and types match the op's
        result types.

      Underspecified:
      - The semantics when `selection` does not match any branch label and
        the default region is absent.
      - Structurally, the op admits zero branches and an empty default;
        such a form is legal but meaningless and is not constrained out.

      Differences from `scf.index_switch`:
      - Selector type is restricted to `SupportedIntType` (no `index`).
      - Case labels are implicit and positional (`case 0`, `case 1`, ...)
        rather than arbitrary integer literals.
      - The default region is optional rather than mandatory.
      - Supports in-values shared across regions, decoupled from the
        op's results (`scf.index_switch` has no shared-value mechanism).
    }];

    let arguments = (ins
        SupportedIntType:$selection,
        Variadic<AnyType>:$in_values
    );

    let regions = (region
        MaxSizedRegion<1>:$default,
        VariadicRegion<SizedRegion<1>>:$branches
    );

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let hasCustomAssemblyFormat = 1;
    let hasRegionVerifier = 1;
}

def ForOp : SCF_Op<"for", [IsolatedFromAbove, SingleBlockImplicitTerminator<"jeff::YieldOp">]> {
    let summary = "Counted for loop with iteration arguments";
    let description = [{
      The `jeff.for` operation represents a counted loop over a half-open
      integer range `[start, stop)`, advancing by `step` on each iteration.
      The body executes once for every value the induction variable takes
      while it remains less than `stop`, starting at `start`.

      The op takes three SSA integer operands — `start`, `stop`, and
      `step`, each of which must be a `SupportedIntType` (one of `i1`,
      `i8`, `i16`, `i32`, `i64`). The induction variable has the same
      type as `start`, `stop`, and `step`.

      The op may optionally carry a list of loop-carried values
      (`in_values`). These are bound to the corresponding body block
      arguments on entry to each iteration and replaced at the end of
      each iteration by the operands of the terminating `jeff.yield`.
      The op's results (`out_values`) hold the final values of the
      carried variables after the last iteration; their types must match
      the types of `in_values`.

      The body is a single block whose arguments are the induction
      variable followed by one argument per loop-carried value. It must
      terminate with a `jeff.yield` whose operand types and count match
      `in_values`. When no values are carried, the terminator is inserted
      implicitly by the parser.

      Schematic syntax:
      ```
      jeff.for $induction_variable = $start to $stop step $step
               ( args ( $assignments ) -> ( $result_types ) )?
               : $induction_variable_type
               { $body }
      ```

      The body's block-argument types are not printed explicitly: they
      are recovered from `-> ( $result_types )` and must equal the iter-arg types.

      Example without carried values:
      ```mlir
      jeff.for %i = %lo to %hi step %s : i32 {
        // body
      }
      ```

      Example with carried values:
      ```mlir
      %r1, %r2 = jeff.for %i = %lo to %hi step %s
                          args(%x = %a, %y = %b) -> (i32, i64) : i32 {
        // ... compute %next_x, %next_y ...
        jeff.yield %next_x, %next_y : i32, i64
      }
      ```

      Invariants enforced by the verifier:
      - `start`, `stop`, and `step` are each a `SupportedIntType`.
      - The induction variable has the same type as `start`, `stop`, and `step`.
      - The number of `in_values` equals the number of results.
      - Each `in_value`, the corresponding body block argument, and the
        corresponding result have matching types.

      Underspecified (not currently enforced):
      - Comparison signedness (the operand types are signless MLIR
        integers).
      - That `step` is non-zero and positive.

      Differences from `scf.for`:
      - Bounds and step are restricted to `SupportedIntType`
        (no `index`, no widths outside `{1, 8, 16, 32, 64}`).
      - Uses the keyword `args` instead of `iter_args`, with a trailing
        `-> ( result_types )` instead of the SCF arrow placement.
      - The induction-variable type annotation `:` is always printed,
        even when no values are carried.
      - Does not implement `LoopLikeOpInterface`,
        `RegionBranchOpInterface`, `RecursiveMemoryEffects`, or
        `AutomaticAllocationScope`, so upstream loop-aware passes
        (LICM, peeling, generic unrolling) do not apply.
    }];

    let arguments = (ins
        SupportedIntType:$start,
        SupportedIntType:$stop,
        SupportedIntType:$step,
        Variadic<AnyType>:$in_values
    );

    let regions = (region SizedRegion<1>:$body);

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasRegionVerifier = 1;
}

def WhileOp : SCF_Op<"while", [IsolatedFromAbove, SingleBlockImplicitTerminator<"jeff::YieldOp">]> {
    let summary = "While loop with iteration arguments";
    let description = [{
      The `jeff.while` operation represents a loop that repeats while a condition holds.
      It has two regions:

      - The `before` region evaluates whether the condition is met.
        It must terminate with a `jeff.yield` operation that yields the condition followed by the input values for the `after` region.
        If the condition is not met, the latter values represent the op's results.
      - The `after` region is only executed if the condition is `true`.
        It must terminate with a `jeff.yield` operation that yields the input values for the next iteration.

      Syntax:
      ```
      [ $out_values = ] jeff.while : ( $in_value_types ) -> ( $out_value_types )
        args ( $before_assignments ) $before
        args ( $after_assignments ) $after
      ```

      Implicit inferences in the syntax:
      - Block-argument types inside the `args(...)` clauses are not written.
        They are inferred from the operation's type annotation.

      Example:
      ```mlir
      %r1, %r2 = jeff.while : (i32, i64) -> (i32, i64) args(%before_a = %a, %before_b = %b) {
        // Evaluate condition
        jeff.yield %condition, %result_a, %result_b : i1, i32, i64
      } args(%after_a, %after_b) {
        // Evaluate body
        jeff.yield %next_a, %next_b : i32, i64
      }
      ```
    }];

    let arguments = (ins
        Variadic<AnyType>:$in_values
    );

    let regions = (region
        SizedRegion<1>:$before,
        SizedRegion<1>:$after
    );

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let hasCustomAssemblyFormat = 1;
    let hasRegionVerifier = 1;
}

def YieldOp : SCF_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["SwitchOp", "ForOp", "WhileOp"]>]> {
  let summary = "Return results from SCF regions";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$out_values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($out_values^ `:` type($out_values))? }];
}

#endif // JEFF_OPS
