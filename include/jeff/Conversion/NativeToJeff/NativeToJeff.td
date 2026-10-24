include "mlir/Pass/PassBase.td"

def NativeToJeff : Pass<"native-to-jeff"> {
  let summary = "Convert built-in MLIR operations to `jeff` operations";

  let description = [{
    This pass converts operations from built-in MLIR dialects to their equivalent operations in the `jeff` dialect.

    Supported `arith` operations:

    - `addf`
    - `addi`
    - `cmpf`
    - `cmpi`
    - `constant`
    - `divsi`
    - `divui`
    - `index_cast`
    - `maxnumf`
    - `maxsi`
    - `maxui`
    - `minnumf`
    - `minsi`
    - `minui`
    - `mulf`
    - `muli`
    - `ori`
    - `remsi`
    - `remui`
    - `shli`
    - `shrsi`
    - `subf`
    - `subi`
    - `xori`

    Supported `math` operations:

    - `absf`
    - `absi`
    - `acos`
    - `acosh`
    - `asin`
    - `asinh`
    - `atan`
    - `atan2`
    - `atanh`
    - `ceil`
    - `cos`
    - `cosh`
    - `exp`
    - `floor`
    - `ipowi`
    - `isinf`
    - `isnan`
    - `log`
    - `powf`
    - `sin`
    - `sinh`
    - `sqrt`
    - `tan`
    - `tanh`

    Supported `tensor` operations:

    - `cast`
    - `dim`
    - `empty`
    - `extract`
    - `from_elements`
    - `insert`
    - `splat` of zero into a one-dimensional dynamically sized tensor
      with a length that converts to `i32` (positive zero for floating-point elements)

    Known limitations:

    - `arith.constant` operations of type `index` are converted to `jeff.int_const_32` operations.
    - `arith.index_cast` operations are removed during conversion.
      This assumes that functions have a `jeff`-compatible signature (e.g., a function must not return an `index`).
      The conversion ensures that the created operations are compatible with each other.
  }];

  let dependentDialects = [
    "mlir::jeff::JeffDialect",
  ];
}
