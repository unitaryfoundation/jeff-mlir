# Changelog

This file tracks the changes to `jeff-mlir`.

The project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

`serializeToFile()` now returns `mlir::LogicalResult` instead of `void` and
reports a failure to open the output file through that result. Previously, it
called `llvm::report_fatal_error()`, which aborts the process and leaves callers
no way to handle the error. For the same reason, `deserializeFromFile()` now
returns a null module instead of aborting when the input file cannot be read.

## [0.3.0] - 2026-07-13

This release fixes the linearity of `WhileOp` (see
<https://github.com/unitaryfoundation/jeff/issues/4> for more information). As
the new implementation covers the functionalit of a do-while operation,
`DoWhileOp` has been removed.

Furthermore, this release marks `SwitchOp`, `ForOp`, and `WhileOp` as
`IsolatedFromAbove`, ensuring that their regions are checked to be isolated from
above. All three operations now also have a parser.

This release is compatible with `jeff-v0.3.0`.

## [0.2.0] - 2026-05-11

This release renames `serialize()` and `deserialize()` to `serializeToFile()`
and `deserializeFromFile()`, respectively. The new `serialize()` and
`deserialize()` functions serialize to and from a memory buffer instead.

Furthermore, this release fixes the deserialization of functions. The function
index had incorrectly been retrieved from list of strings and not the list of
functions.

This release is compatible with `jeff-v0.2.0`.

## [0.1.0] - 2026-04-14

Initial release.

This release is compatible with `jeff-v0.2.0`.

<!-- Version links -->

[unreleased]: https://github.com/PennyLaneAI/jeff-mlir/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/PennyLaneAI/jeff-mlir/tree/v0.3.0
[0.2.0]: https://github.com/PennyLaneAI/jeff-mlir/tree/v0.2.0
[0.1.0]: https://github.com/PennyLaneAI/jeff-mlir/tree/v0.1.0
