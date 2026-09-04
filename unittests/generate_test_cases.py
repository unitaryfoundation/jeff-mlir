#!/usr/bin/python3

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jeff-format~=0.1.0",
# ]
# ///

from collections.abc import Callable
from pathlib import Path

from jeff import (
    FloatArrayType,
    FloatType,
    ForSCF,
    FunctionDef,
    IntArrayType,
    IntType,
    JeffModule,
    JeffOp,
    JeffRegion,
    JeffValue,
    QubitType,
    QuregType,
    WhileSCF,
    pauli_rotation,
    quantum_gate,
    qubit_alloc,
    qubit_free,
    schema,
    switch_case,
)

INPUTS_DIR = Path(__file__).parent / "inputs"

# Registry for generator functions
_generators = []


def register_generator(function: Callable[[], None]) -> Callable[[], None]:
    """Decorator for registering generator functions."""
    _generators.append(function)
    return function


def _create_and_write_module(operations: list[JeffOp], output_filename: str) -> None:
    body = JeffRegion(
        sources=[],
        targets=[],
        operations=operations,
    )
    function = FunctionDef(name="main", body=body)
    module = JeffModule([function])

    output_file = INPUTS_DIR / output_filename
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


# ===----------------------------------------------------------------------=== #
# Qubit operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_qubit_alloc() -> None:
    alloc = qubit_alloc()
    _create_and_write_module([alloc], "unit_qubit_alloc.jeff")


@register_generator
def generate_qubit_free() -> None:
    alloc = qubit_alloc()
    free = qubit_free(alloc.outputs[0])
    _create_and_write_module([alloc, free], "unit_qubit_free.jeff")


@register_generator
def generate_qubit_free_zero() -> None:
    alloc = qubit_alloc()
    free_zero = JeffOp("qubit", "freeZero", [alloc.outputs[0]], [])
    _create_and_write_module([alloc, free_zero], "unit_qubit_free_zero.jeff")


@register_generator
def generate_qubit_measure() -> None:
    alloc = qubit_alloc()
    measure = JeffOp(
        "qubit",
        "measure",
        [alloc.outputs[0]],
        [JeffValue(IntType(1))],
    )
    _create_and_write_module([alloc, measure], "unit_qubit_measure.jeff")


@register_generator
def generate_qubit_measure_nd() -> None:
    alloc = qubit_alloc()
    measure = JeffOp(
        "qubit",
        "measureNd",
        [alloc.outputs[0]],
        [JeffValue(QubitType()), JeffValue(IntType(1))],
    )
    free = qubit_free(measure.outputs[0])
    _create_and_write_module([alloc, measure, free], "unit_qubit_measure_nd.jeff")


@register_generator
def generate_qubit_reset() -> None:
    alloc = qubit_alloc()
    reset = JeffOp("qubit", "reset", [alloc.outputs[0]], [JeffValue(QubitType())])
    free = qubit_free(reset.outputs[0])
    _create_and_write_module([alloc, reset, free], "unit_qubit_reset.jeff")


# ===----------------------------------------------------------------------=== #
# Gate operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_one_qubit_zero_parameter() -> None:
    for gate_name in ["x", "y", "z", "h", "s", "t", "h", "i"]:
        alloc = qubit_alloc()
        gate = quantum_gate(gate_name, qubits=[alloc.outputs[0]])
        free = qubit_free(gate.outputs[0])
        _create_and_write_module([alloc, gate, free], f"unit_gate_{gate_name}.jeff")


@register_generator
def generate_multi_controlled_one_qubit_zero_parameter() -> None:
    for gate_name in ["x", "y", "z", "h", "s", "t", "h", "i"]:
        alloc1 = qubit_alloc()
        alloc2 = qubit_alloc()
        alloc3 = qubit_alloc()
        gate = quantum_gate(
            gate_name,
            qubits=[alloc1.outputs[0]],
            control_qubits=[alloc2.outputs[0], alloc3.outputs[0]],
        )
        free1 = qubit_free(gate.outputs[0])
        free2 = qubit_free(gate.outputs[1])
        free3 = qubit_free(gate.outputs[2])
        _create_and_write_module(
            [alloc1, alloc2, alloc3, gate, free1, free2, free3],
            f"unit_gate_mc{gate_name}.jeff",
        )


@register_generator
def generate_one_qubit_one_parameter() -> None:
    for gate_name in ["r1", "rx", "ry", "rz"]:
        alloc = qubit_alloc()
        rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
        gate = quantum_gate(
            gate_name,
            qubits=[alloc.outputs[0]],
            params=[rotation.outputs[0]],
        )
        free = qubit_free(gate.outputs[0])
        _create_and_write_module(
            [alloc, rotation, gate, free],
            f"unit_gate_{gate_name}.jeff",
        )


@register_generator
def generate_controlled_one_qubit_one_parameter() -> None:
    for gate_name in ["r1", "rx", "ry", "rz"]:
        alloc1 = qubit_alloc()
        alloc2 = qubit_alloc()
        rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
        gate = quantum_gate(
            gate_name,
            qubits=[alloc1.outputs[0]],
            params=[rotation.outputs[0]],
            control_qubits=[alloc2.outputs[0]],
        )
        free1 = qubit_free(gate.outputs[0])
        free2 = qubit_free(gate.outputs[1])
        _create_and_write_module(
            [alloc1, alloc2, rotation, gate, free1, free2],
            f"unit_gate_c{gate_name}.jeff",
        )


@register_generator
def generate_u() -> None:
    alloc = qubit_alloc()
    theta = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.1)
    phi = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.2)
    lambda_ = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.3)
    gate = quantum_gate(
        "u",
        qubits=[alloc.outputs[0]],
        params=[theta.outputs[0], phi.outputs[0], lambda_.outputs[0]],
    )
    free = qubit_free(gate.outputs[0])
    _create_and_write_module(
        [alloc, theta, phi, lambda_, gate, free], "unit_gate_u.jeff"
    )


@register_generator
def generate_cu() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    theta = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.1)
    phi = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.2)
    lambda_ = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.3)
    gate = quantum_gate(
        "u",
        qubits=[alloc1.outputs[0]],
        params=[theta.outputs[0], phi.outputs[0], lambda_.outputs[0]],
        control_qubits=[alloc2.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, theta, phi, lambda_, gate, free1, free2],
        "unit_gate_cu.jeff",
    )


@register_generator
def generate_swap() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    gate = quantum_gate("swap", qubits=[alloc1.outputs[0], alloc2.outputs[0]])
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, gate, free1, free2], "unit_gate_swap.jeff"
    )


@register_generator
def generate_cswap() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    gate = quantum_gate(
        "swap",
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
        control_qubits=[alloc3.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    free3 = qubit_free(gate.outputs[2])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, gate, free1, free2, free3],
        "unit_gate_cswap.jeff",
    )


@register_generator
def generate_gphase() -> None:
    alloc = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = quantum_gate(
        "gphase",
        qubits=[],
        params=[rotation.outputs[0]],
    )
    free = qubit_free(alloc.outputs[0])
    _create_and_write_module([alloc, rotation, gate, free], "unit_gate_gphase.jeff")


@register_generator
def generate_cgphase() -> None:
    alloc = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = quantum_gate(
        "gphase",
        qubits=[],
        params=[rotation.outputs[0]],
        control_qubits=[alloc.outputs[0]],
    )
    free = qubit_free(gate.outputs[0])
    _create_and_write_module([alloc, rotation, gate, free], "unit_gate_cgphase.jeff")


@register_generator
def generate_custom_1() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    gate = quantum_gate("custom", qubits=[alloc1.outputs[0], alloc2.outputs[0]])
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, gate, free1, free2], "unit_gate_custom_1.jeff"
    )


@register_generator
def generate_custom_2() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = quantum_gate(
        "custom",
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
        params=[rotation.outputs[0]],
        control_qubits=[alloc3.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    free3 = qubit_free(gate.outputs[2])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, rotation, gate, free1, free2, free3],
        "unit_gate_custom_2.jeff",
    )


@register_generator
def generate_ppr_rxx() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = pauli_rotation(
        angle=rotation.outputs[0],
        pauli_string=["x", "x"],
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, rotation, gate, free1, free2],
        "unit_gate_ppr_rxx.jeff",
    )


@register_generator
def generate_ppr_crxy() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = pauli_rotation(
        angle=rotation.outputs[0],
        pauli_string=["x", "y"],
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
        control_qubits=[alloc3.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    free3 = qubit_free(gate.outputs[2])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, rotation, gate, free1, free2, free3],
        "unit_gate_ppr_crxy.jeff",
    )


# ===----------------------------------------------------------------------=== #
# Qureg operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_qureg_alloc() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    _create_and_write_module([num_qubits, alloc], "unit_qureg_alloc.jeff")


@register_generator
def generate_qureg_free_zero() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    free_zero = JeffOp("qureg", "freeZero", [alloc.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, alloc, free_zero], "unit_qureg_free_zero.jeff"
    )


@register_generator
def generate_qureg_extract_index() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType(5))],
    )
    extract_index = JeffOp(
        "qureg",
        "extractIndex",
        [alloc.outputs[0], index.outputs[0]],
        [JeffValue(QuregType(5)), JeffValue(QubitType())],
    )
    free1 = JeffOp("qureg", "free", [extract_index.outputs[0]], [])
    free2 = qubit_free(extract_index.outputs[1])
    _create_and_write_module(
        [num_qubits, index, alloc, extract_index, free1, free2],
        "unit_qureg_extract_index.jeff",
    )


@register_generator
def generate_qureg_insert_index() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType(5))],
    )
    extract_index = JeffOp(
        "qureg",
        "extractIndex",
        [alloc.outputs[0], index.outputs[0]],
        [JeffValue(QuregType(5)), JeffValue(QubitType())],
    )
    insert_index = JeffOp(
        "qureg",
        "insertIndex",
        [extract_index.outputs[0], index.outputs[0], extract_index.outputs[1]],
        [JeffValue(QuregType(5))],
    )
    free = JeffOp("qureg", "free", [insert_index.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, index, alloc, extract_index, insert_index, free],
        "unit_qureg_insert_index.jeff",
    )


@register_generator
def generate_qureg_extract_slice() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    start = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    length = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType(5))],
    )
    extract_slice = JeffOp(
        "qureg",
        "extractSlice",
        [alloc.outputs[0], start.outputs[0], length.outputs[0]],
        [JeffValue(QuregType(5)), JeffValue(QuregType(2))],
    )
    free1 = JeffOp("qureg", "free", [extract_slice.outputs[0]], [])
    free2 = JeffOp("qureg", "free", [extract_slice.outputs[1]], [])
    _create_and_write_module(
        [num_qubits, start, length, alloc, extract_slice, free1, free2],
        "unit_qureg_extract_slice.jeff",
    )


@register_generator
def generate_qureg_insert_slice() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    length = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType(5))],
    )
    extract_slice = JeffOp(
        "qureg",
        "extractSlice",
        [alloc.outputs[0], index.outputs[0], length.outputs[0]],
        [JeffValue(QuregType(5)), JeffValue(QuregType(2))],
    )
    insert_slice = JeffOp(
        "qureg",
        "insertSlice",
        [extract_slice.outputs[0], index.outputs[0], extract_slice.outputs[1]],
        [JeffValue(QuregType(5))],
    )
    free = JeffOp("qureg", "free", [insert_slice.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, index, length, alloc, extract_slice, insert_slice, free],
        "unit_qureg_insert_slice.jeff",
    )


@register_generator
def generate_qureg_length() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    length = JeffOp(
        "qureg",
        "length",
        [alloc.outputs[0]],
        [JeffValue(QuregType()), JeffValue(IntType(32))],
    )
    free = JeffOp("qureg", "free", [length.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, alloc, length, free],
        "unit_qureg_length.jeff",
    )


@register_generator
def generate_qureg_split() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType(5))],
    )
    split = JeffOp(
        "qureg",
        "split",
        [alloc.outputs[0], index.outputs[0]],
        [JeffValue(QuregType(3)), JeffValue(QuregType(2))],
    )
    free1 = JeffOp("qureg", "free", [split.outputs[0]], [])
    free2 = JeffOp("qureg", "free", [split.outputs[1]], [])
    _create_and_write_module(
        [num_qubits, index, alloc, split, free1, free2],
        "unit_qureg_split.jeff",
    )


@register_generator
def generate_qureg_join() -> None:
    num_qubits1 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    num_qubits2 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    alloc1 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits1.outputs[0]],
        [JeffValue(QuregType(3))],
    )
    alloc2 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits2.outputs[0]],
        [JeffValue(QuregType(2))],
    )
    join = JeffOp(
        "qureg",
        "join",
        [alloc1.outputs[0], alloc2.outputs[0]],
        [JeffValue(QuregType(5))],
    )
    free = JeffOp("qureg", "free", [join.outputs[0]], [])
    _create_and_write_module(
        [num_qubits1, num_qubits2, alloc1, alloc2, join, free],
        "unit_qureg_join.jeff",
    )


@register_generator
def generate_qureg_create() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    qureg_create = JeffOp(
        "qureg",
        "create",
        [alloc1.outputs[0], alloc2.outputs[0], alloc3.outputs[0]],
        [JeffValue(QuregType(3))],
    )
    free = JeffOp("qureg", "free", [qureg_create.outputs[0]], [])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, qureg_create, free],
        "unit_qureg_create.jeff",
    )


@register_generator
def generate_qureg_free() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    free = JeffOp("qureg", "free", [alloc.outputs[0]], [])
    _create_and_write_module([num_qubits, alloc, free], "unit_qureg_free.jeff")


# ===----------------------------------------------------------------------=== #
# Int operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_int_const1() -> None:
    const = JeffOp("int", "const1", [], [JeffValue(IntType(1))], True)

    body = JeffRegion(
        sources=[],
        targets=[const.outputs[0]],
        operations=[const],
    )
    function = FunctionDef(name="main", body=body)
    module = JeffModule([function])

    output_file = INPUTS_DIR / "unit_int_const1.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


@register_generator
def generate_int_const() -> None:
    for bit_width in [8, 16, 32, 64]:
        const = JeffOp(
            "int", f"const{bit_width}", [], [JeffValue(IntType(bit_width))], 3
        )

        body = JeffRegion(
            sources=[],
            targets=[const.outputs[0]],
            operations=[const],
        )
        function = FunctionDef(name="main", body=body)
        module = JeffModule([function])

        output_file = INPUTS_DIR / f"unit_int_const{bit_width}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_int_unary() -> None:
    operations = ["not", "abs"]
    for operation in operations:
        value = JeffValue(IntType(32))
        unary = JeffOp(
            "int",
            operation,
            [value],
            [JeffValue(IntType(32))],
        )
        compute_body = JeffRegion(
            sources=[value],
            targets=[unary.outputs[0]],
            operations=[unary],
        )
        compute_function = FunctionDef(name="compute", body=compute_body)

        const = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
        call = JeffOp(
            "func",
            "funcCall",
            [const.outputs[0]],
            [JeffValue(IntType(32))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([compute_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_int_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_int_binary() -> None:
    operations = [
        "add",
        "sub",
        "mul",
        "divS",
        "divU",
        "pow",
        "and",
        "or",
        "xor",
        "minS",
        "minU",
        "maxS",
        "maxU",
        "remS",
        "remU",
        "shl",
        "shr",
    ]
    for operation in operations:
        lhs = JeffValue(IntType(32))
        rhs = JeffValue(IntType(32))
        binary = JeffOp(
            "int",
            operation,
            [lhs, rhs],
            [JeffValue(IntType(32))],
        )
        compute_body = JeffRegion(
            sources=[lhs, rhs],
            targets=[binary.outputs[0]],
            operations=[binary],
        )
        compute_function = FunctionDef(name="compute", body=compute_body)

        const1 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
        const2 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
        call = JeffOp(
            "func",
            "funcCall",
            [const1.outputs[0], const2.outputs[0]],
            [JeffValue(IntType(32))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const1, const2, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([compute_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_int_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_int_comparison() -> None:
    operations = ["eq", "ltS", "lteS", "ltU", "lteU"]
    for operation in operations:
        lhs = JeffValue(IntType(32))
        rhs = JeffValue(IntType(32))
        comparison = JeffOp(
            "int",
            operation,
            [lhs, rhs],
            [JeffValue(IntType(1))],
        )
        compute_body = JeffRegion(
            sources=[lhs, rhs],
            targets=[comparison.outputs[0]],
            operations=[comparison],
        )
        compute_function = FunctionDef(name="compute", body=compute_body)

        const1 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
        const2 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
        call = JeffOp(
            "func",
            "funcCall",
            [const1.outputs[0], const2.outputs[0]],
            [JeffValue(IntType(1))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const1, const2, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([compute_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_int_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


# ===----------------------------------------------------------------------=== #
# IntArray operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_int_array_const1() -> None:
    const = JeffOp(
        "intArray", "const1", [], [JeffValue(IntArrayType(1, 3))], [True, False, True]
    )

    body = JeffRegion(
        sources=[],
        targets=[const.outputs[0]],
        operations=[const],
    )
    function = FunctionDef(name="main", body=body)
    module = JeffModule([function])

    output_file = INPUTS_DIR / "unit_int_array_const1.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


@register_generator
def generate_int_array_const() -> None:
    for bit_width in [8, 16, 32, 64]:
        const = JeffOp(
            "intArray",
            f"const{bit_width}",
            [],
            [JeffValue(IntArrayType(bit_width, 3))],
            [1, 2, 3],
        )

        body = JeffRegion(
            sources=[],
            targets=[const.outputs[0]],
            operations=[const],
        )
        function = FunctionDef(name="main", body=body)
        module = JeffModule([function])

        output_file = INPUTS_DIR / f"unit_int_array_const{bit_width}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_int_array_zero() -> None:
    length = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    zero = JeffOp(
        "intArray",
        "zero",
        [length.outputs[0]],
        [JeffValue(IntArrayType(32))],
        32,
    )
    _create_and_write_module([length, zero], "unit_int_array_zero.jeff")


@register_generator
def generate_int_array_get_index() -> None:
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    array = JeffOp(
        "intArray",
        "const32",
        [],
        [JeffValue(IntArrayType(32, 3))],
        [1, 2, 3],
    )
    get_index = JeffOp(
        "intArray",
        "getIndex",
        [array.outputs[0], index.outputs[0]],
        [JeffValue(IntType(32))],
    )
    _create_and_write_module(
        [index, array, get_index],
        "unit_int_array_get_index.jeff",
    )


@register_generator
def generate_int_array_set_index() -> None:
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    value = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    array = JeffOp(
        "intArray",
        "const32",
        [],
        [JeffValue(IntArrayType(32, 3))],
        [1, 2, 3],
    )
    set_index = JeffOp(
        "intArray",
        "setIndex",
        [array.outputs[0], index.outputs[0], value.outputs[0]],
        [JeffValue(IntArrayType(32, 3))],
    )
    _create_and_write_module(
        [index, value, array, set_index],
        "unit_int_array_set_index.jeff",
    )


@register_generator
def generate_int_array_length() -> None:
    value = JeffValue(IntArrayType(32, 3))
    length = JeffOp(
        "intArray",
        "length",
        [value],
        [JeffValue(IntType(32))],
    )
    get_length_body = JeffRegion(
        sources=[value],
        targets=[length.outputs[0]],
        operations=[length],
    )
    get_length_function = FunctionDef(name="get_length", body=get_length_body)

    array = JeffOp(
        "intArray",
        "const32",
        [],
        [JeffValue(IntArrayType(32, 3))],
        [1, 2, 3],
    )
    call = JeffOp(
        "func",
        "funcCall",
        [array.outputs[0]],
        [JeffValue(IntType(32))],
        0,
    )
    main_body = JeffRegion(
        sources=[],
        targets=[],
        operations=[array, call],
    )
    main_function = FunctionDef(name="main", body=main_body)

    module = JeffModule([get_length_function, main_function], entrypoint=1)

    output_file = INPUTS_DIR / "unit_int_array_length.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


@register_generator
def generate_int_array_create() -> None:
    value1 = JeffValue(IntType(32))
    value2 = JeffValue(IntType(32))
    value3 = JeffValue(IntType(32))
    create = JeffOp(
        "intArray",
        "create",
        [value1, value2, value3],
        [JeffValue(IntArrayType(32, 3))],
    )
    create_body = JeffRegion(
        sources=[value1, value2, value3],
        targets=[create.outputs[0]],
        operations=[create],
    )
    create_function = FunctionDef(name="create_array", body=create_body)

    const1 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    const2 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    const3 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    call = JeffOp(
        "func",
        "funcCall",
        [const1.outputs[0], const2.outputs[0], const3.outputs[0]],
        [JeffValue(IntArrayType(32, 3))],
        0,
    )
    main_body = JeffRegion(
        sources=[],
        targets=[],
        operations=[const1, const2, const3, call],
    )
    main_function = FunctionDef(name="main", body=main_body)

    module = JeffModule([create_function, main_function], entrypoint=1)

    output_file = INPUTS_DIR / "unit_int_array_create.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


# ===----------------------------------------------------------------------=== #
# Float operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_float_const() -> None:
    for bit_width in [32, 64]:
        const = JeffOp(
            "float", f"const{bit_width}", [], [JeffValue(FloatType(bit_width))], 0.3
        )

        body = JeffRegion(
            sources=[],
            targets=[const.outputs[0]],
            operations=[const],
        )
        function = FunctionDef(name="main", body=body)
        module = JeffModule([function])

        output_file = INPUTS_DIR / f"unit_float_const{bit_width}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_float_unary() -> None:
    operations = [
        "sqrt",
        "abs",
        "ceil",
        "floor",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
    ]
    for operation in operations:
        value = JeffValue(FloatType(32))
        unary = JeffOp(
            "float",
            operation,
            [value],
            [JeffValue(FloatType(32))],
        )
        compute_body = JeffRegion(
            sources=[value],
            targets=[unary.outputs[0]],
            operations=[unary],
        )
        compute_function = FunctionDef(name="compute", body=compute_body)

        const = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.3)
        call = JeffOp(
            "func",
            "funcCall",
            [const.outputs[0]],
            [JeffValue(FloatType(32))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([compute_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_float_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_float_binary() -> None:
    operations = ["add", "sub", "mul", "pow", "atan2", "max", "min"]
    for operation in operations:
        lhs = JeffValue(FloatType(32))
        rhs = JeffValue(FloatType(32))
        binary = JeffOp(
            "float",
            operation,
            [lhs, rhs],
            [JeffValue(FloatType(32))],
        )
        compute_body = JeffRegion(
            sources=[lhs, rhs],
            targets=[binary.outputs[0]],
            operations=[binary],
        )
        compute_function = FunctionDef(name="compute", body=compute_body)

        const1 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.3)
        const2 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.5)
        call = JeffOp(
            "func",
            "funcCall",
            [const1.outputs[0], const2.outputs[0]],
            [JeffValue(FloatType(32))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const1, const2, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([compute_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_float_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_float_comparison() -> None:
    operations = ["eq", "lt", "lte"]
    for operation in operations:
        lhs = JeffValue(FloatType(32))
        rhs = JeffValue(FloatType(32))
        comparison = JeffOp(
            "float",
            operation,
            [lhs, rhs],
            [JeffValue(IntType(1))],
        )
        compute_body = JeffRegion(
            sources=[lhs, rhs],
            targets=[comparison.outputs[0]],
            operations=[comparison],
        )
        compute_function = FunctionDef(name="compute", body=compute_body)

        const1 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.3)
        const2 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.5)
        call = JeffOp(
            "func",
            "funcCall",
            [const1.outputs[0], const2.outputs[0]],
            [JeffValue(IntType(1))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const1, const2, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([compute_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_float_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_float_is() -> None:
    operations = ["isNan", "isInf"]
    for operation in operations:
        value = JeffValue(FloatType(32))
        is_op = JeffOp(
            "float",
            operation,
            [value],
            [JeffValue(IntType(1))],
        )
        check_body = JeffRegion(
            sources=[value],
            targets=[is_op.outputs[0]],
            operations=[is_op],
        )
        check_function = FunctionDef(name="check", body=check_body)

        const = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.3)
        call = JeffOp(
            "func",
            "funcCall",
            [const.outputs[0]],
            [JeffValue(IntType(1))],
            0,
        )
        main_body = JeffRegion(
            sources=[],
            targets=[],
            operations=[const, call],
        )
        main_function = FunctionDef(name="main", body=main_body)

        module = JeffModule([check_function, main_function], entrypoint=1)

        output_file = INPUTS_DIR / f"unit_float_{operation}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


# ===----------------------------------------------------------------------=== #
# FloatArray operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_float_array_const() -> None:
    for bit_width in [32, 64]:
        const = JeffOp(
            "floatArray",
            f"const{bit_width}",
            [],
            [JeffValue(FloatArrayType(bit_width, 3))],
            [0.1, 0.2, 0.3],
        )

        body = JeffRegion(
            sources=[],
            targets=[const.outputs[0]],
            operations=[const],
        )
        function = FunctionDef(name="main", body=body)
        module = JeffModule([function])

        output_file = INPUTS_DIR / f"unit_float_array_const{bit_width}.jeff"
        output_file.unlink(missing_ok=True)
        module.write_out(output_file)


@register_generator
def generate_float_array_zero() -> None:
    length = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    zero = JeffOp(
        "floatArray",
        "zero",
        [length.outputs[0]],
        [JeffValue(FloatArrayType(32))],
        schema.FloatPrecision.float32,
    )
    _create_and_write_module([length, zero], "unit_float_array_zero.jeff")


@register_generator
def generate_float_array_get_index() -> None:
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    array = JeffOp(
        "floatArray",
        "const32",
        [],
        [JeffValue(FloatArrayType(32, 3))],
        [0.1, 0.2, 0.3],
    )
    get_index = JeffOp(
        "floatArray",
        "getIndex",
        [array.outputs[0], index.outputs[0]],
        [JeffValue(FloatType(32))],
    )
    _create_and_write_module(
        [index, array, get_index],
        "unit_float_array_get_index.jeff",
    )


@register_generator
def generate_float_array_set_index() -> None:
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    value = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.5)
    array = JeffOp(
        "floatArray",
        "const32",
        [],
        [JeffValue(FloatArrayType(32, 3))],
        [0.1, 0.2, 0.3],
    )
    set_index = JeffOp(
        "floatArray",
        "setIndex",
        [array.outputs[0], index.outputs[0], value.outputs[0]],
        [JeffValue(FloatArrayType(32, 3))],
    )
    _create_and_write_module(
        [index, value, array, set_index],
        "unit_float_array_set_index.jeff",
    )


@register_generator
def generate_float_array_length() -> None:
    value = JeffValue(FloatArrayType(32, 3))
    length = JeffOp(
        "floatArray",
        "length",
        [value],
        [JeffValue(IntType(32))],
    )
    get_length_body = JeffRegion(
        sources=[value],
        targets=[length.outputs[0]],
        operations=[length],
    )
    get_length_function = FunctionDef(name="get_length", body=get_length_body)

    array = JeffOp(
        "floatArray",
        "const32",
        [],
        [JeffValue(FloatArrayType(32, 3))],
        [0.1, 0.2, 0.3],
    )
    call = JeffOp(
        "func",
        "funcCall",
        [array.outputs[0]],
        [JeffValue(IntType(32))],
        0,
    )
    main_body = JeffRegion(
        sources=[],
        targets=[],
        operations=[array, call],
    )
    main_function = FunctionDef(name="main", body=main_body)

    module = JeffModule([get_length_function, main_function], entrypoint=1)

    output_file = INPUTS_DIR / "unit_float_array_length.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


@register_generator
def generate_float_array_create() -> None:
    value1 = JeffValue(FloatType(32))
    value2 = JeffValue(FloatType(32))
    value3 = JeffValue(FloatType(32))
    create = JeffOp(
        "floatArray",
        "create",
        [value1, value2, value3],
        [JeffValue(FloatArrayType(32, 3))],
    )
    create_body = JeffRegion(
        sources=[value1, value2, value3],
        targets=[create.outputs[0]],
        operations=[create],
    )
    create_function = FunctionDef(name="create_array", body=create_body)

    const1 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.1)
    const2 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.2)
    const3 = JeffOp("float", "const32", [], [JeffValue(FloatType(32))], 0.3)
    call = JeffOp(
        "func",
        "funcCall",
        [const1.outputs[0], const2.outputs[0], const3.outputs[0]],
        [JeffValue(FloatArrayType(32, 3))],
        0,
    )
    main_body = JeffRegion(
        sources=[],
        targets=[],
        operations=[const1, const2, const3, call],
    )
    main_function = FunctionDef(name="main", body=main_body)

    module = JeffModule([create_function, main_function], entrypoint=1)

    output_file = INPUTS_DIR / "unit_float_array_create.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


# ===----------------------------------------------------------------------=== #
# SCF operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_scf_switch() -> None:
    alloc = qubit_alloc()
    true = JeffOp("int", "const1", [], [JeffValue(IntType(1))], True)

    h = quantum_gate("h", qubits=[JeffValue(QubitType())])
    then = JeffRegion(
        sources=[h.inputs[0]],
        targets=[h.outputs[0]],
        operations=[h],
    )

    qubit = JeffValue(QubitType())
    else_ = JeffRegion(
        sources=[qubit],
        targets=[qubit],
        operations=[],
    )

    switch = switch_case(
        index=true.outputs[0],
        region_args=[alloc.outputs[0]],
        branches=[then],
        default=else_,
    )

    free = qubit_free(switch.outputs[0])

    _create_and_write_module([alloc, true, switch, free], "unit_scf_switch.jeff")


@register_generator
def generate_scf_for() -> None:
    alloc = qubit_alloc()
    start = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 0)
    end = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    step = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)

    h = quantum_gate("h", qubits=[JeffValue(QubitType())])
    body = JeffRegion(
        sources=[JeffValue(IntType(32)), h.inputs[0]],
        targets=[h.outputs[0]],
        operations=[h],
    )

    for_scf = ForSCF(body=body)
    for_ = JeffOp(
        "scf",
        "for",
        [start.outputs[0], end.outputs[0], step.outputs[0], alloc.outputs[0]],
        [JeffValue(QubitType())],
        for_scf,
    )

    free = qubit_free(for_.outputs[0])

    _create_and_write_module([alloc, start, end, step, for_, free], "unit_scf_for.jeff")


@register_generator
def generate_scf_while() -> None:
    alloc = qubit_alloc()
    counter = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 0)

    qubit = JeffValue(QubitType())
    three = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    int_lt = JeffOp(
        "int",
        "ltS",
        [JeffValue(IntType(32)), three.outputs[0]],
        [JeffValue(IntType(1))],
    )
    before = JeffRegion(
        sources=[qubit, int_lt.inputs[0]],
        targets=[int_lt.outputs[0], qubit, int_lt.outputs[0]],
        operations=[three, int_lt],
    )

    h = quantum_gate("h", qubits=[JeffValue(QubitType())])
    one = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    int_add = JeffOp(
        "int", "add", [JeffValue(IntType(32)), one.outputs[0]], [JeffValue(IntType(32))]
    )
    after = JeffRegion(
        sources=[h.inputs[0], int_add.inputs[0]],
        targets=[h.outputs[0], int_add.outputs[0]],
        operations=[h, one, int_add],
    )

    while_scf = WhileSCF(before=before, after=after)
    while_ = JeffOp(
        "scf",
        "while",
        [alloc.outputs[0], counter.outputs[0]],
        [JeffValue(QubitType()), JeffValue(IntType(32))],
        while_scf,
    )

    free = qubit_free(while_.outputs[0])

    _create_and_write_module([alloc, counter, while_, free], "unit_scf_while.jeff")


# ===----------------------------------------------------------------------=== #
# Func operations
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_function_calls() -> None:
    for callee_index in [0, 1]:
        qubit = JeffValue(QubitType())
        angle = JeffValue(FloatType(64))
        rotation = quantum_gate("rz", qubits=[qubit], params=[angle])
        callee = FunctionDef(
            name="rotate",
            body=JeffRegion(
                sources=[qubit, angle],
                targets=rotation.outputs,
                operations=[rotation],
            ),
        )

        alloc = qubit_alloc()
        const = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
        call = JeffOp(
            "func",
            "funcCall",
            [alloc.outputs[0], const.outputs[0]],
            [JeffValue(QubitType())],
            callee_index,
        )
        free = qubit_free(call.outputs[0])
        caller = FunctionDef(
            name="main",
            body=JeffRegion(
                sources=[], targets=[], operations=[alloc, const, call, free]
            ),
        )

        functions = [caller]
        functions.insert(callee_index, callee)
        program = JeffModule(functions, entrypoint=1 - callee_index)
        direction = "forward" if callee_index else "backward"
        output_file = INPUTS_DIR / f"unit_func_call_{direction}.jeff"
        output_file.unlink(missing_ok=True)
        program.write_out(output_file)


# ===----------------------------------------------------------------------=== #
# Integration tests
# ===----------------------------------------------------------------------=== #


@register_generator
def generate_bell_pair() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    h = quantum_gate("h", qubits=[alloc1.outputs[0]])
    cx = quantum_gate("x", qubits=[alloc2.outputs[0]], control_qubits=[h.outputs[0]])
    free1 = qubit_free(cx.outputs[1])
    free2 = qubit_free(cx.outputs[0])
    _create_and_write_module(
        [alloc1, alloc2, h, cx, free1, free2],
        "bell_pair.jeff",
    )


if __name__ == "__main__":
    for generator in _generators:
        generator()
