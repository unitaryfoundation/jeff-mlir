# An MLIR dialect for the `jeff` exchange format

This repository is contains an MLIR dialect for the
[`jeff` exchange format](https://github.com/unitaryfoundation/jeff). The purpose
is to facilitate conversion to and from `jeff` for MLIR-based quantum compilers,
and reuse certain components like the jeff serialization & deserialization.

## Build instructions

The dialect sources can be incorporated directly into an existing project
(in-tree), or built separately (out-of-tree). At the moment, the CMake script
supports building the dialect as a standalone project which generates a custom
`opt` tool.

To do so, make sure you have an existing MLIR build.

<details>
<summary>If you don't ...</summary>

You can get pre-built binaries from the
[munich-quantum-software/portable-mlir-toolchain](https://github.com/munich-quantum-software/portable-mlir-toolchain)
project via the installers from
[munich-quantum-software/setup-mlir](https://github.com/munich-quantum-software/setup-mlir):

On Linux and macOS, use the following Bash command:

```bash
curl -LsSf https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.sh | bash -s -- -v 23.1.0 -p /path/to/installation
```

On Windows, use the following PowerShell command:

```powershell
powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.ps1))) -llvm_version 23.1.0 -install_prefix /path/to/installation"
```

Replace `/path/to/installation` with the path to your preferred installation
directory. Then, set `MLIR_DIR` to `/path/to/installation/lib/cmake/mlir` in
your CMake configuration step. The project hosts a variety of pre-built binaries
for different platforms and LLVM versions. Check out the
[project page](https://github.com/munich-quantum-software/setup-mlir) for more
details.

Alternatively, you can build MLIR from source following the instructions in the
[MLIR documentation](https://mlir.llvm.org/getting_started/), which are
summarized below. First, clone the repository:

```sh
git clone https://github.com/llvm/llvm-project.git external/llvm-project
```

Then, build and run the MLIR test suite:

```sh
cd external/llvm-project
cmake -B build -S . -G Ninja            \
      -DCMAKE_BUILD_TYPE=Release        \
      -DLLVM_ENABLE_PROJECTS="mlir"     \
      -DLLVM_BUILD_EXAMPLES=OFF         \
      -DLLVM_ENABLE_ASSERTIONS=ON       \
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
cmake --build build --target check-mlir
```

Finally, you can point the `MLIR_DIR` variable to
`external/llvm-project/build/lib/cmake/mlir` in your CMake configuration step.

</details>
<br>

From here, we can build the `jeff` dialect via the provided CMake presets:

```sh
cmake --preset release && cmake --build --preset release
```

The available presets are `debug` and `release` on Linux and macOS, and
`debug-windows` and `release-windows` on Windows. Each preset builds into its
own directory under `./build/<preset>`. The tests can be run with
`ctest --preset <preset>`.

If CMake can't find the MLIR build, it can be specified via the `MLIR_DIR`
environment variable, which the presets pick up, or via `-DMLIR_DIR=...` during
the config step. It must point to the cmake files in the build or installation
directory, for example `/path/to/installation/lib/cmake/mlir`.

The `jeff-opt` tool will be located in the build directory under
`./build/<preset>/lib/opt/jeff-opt`.

## Contributions

`jeff-mlir` is hosted by the [Unitary Foundation](https://unitary.foundation/)
and is a collaboration between developers and researchers at
[Xanadu](https://www.xanadu.ai), [MQSC](https://mq.sc/), and the
[Chair for Design Automation at TUM](https://www.cda.cit.tum.de/).

Your contributions help improve the tool for everyone! There are many ways you
can contribute, such as:

- Opening issues to share bugs or request features.
- Taking part in the development discussion and helping us shape the roadmap of
  `jeff-mlir`.
- Integrating `jeff-mlir` with quantum compilers.

If you have questions about contributing, please ask on the Unitary Foundation
Discord.

## License

This `jeff` dialect is **free** and **open source**, released under the Apache
License, Version 2.0.
