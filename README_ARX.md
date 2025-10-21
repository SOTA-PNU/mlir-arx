# MLIR-ARX: ONNX-MLIR with ARX Accelerator Support

MLIR-ARX is an enhanced version of ONNX-MLIR that provides specialized support for the ARX (Advanced RISC eXtended) hardware accelerator. This project enables efficient compilation and execution of ONNX models on ARX-enabled systems through dedicated dialect and optimization passes.

## Table of Contents

- [Overview](#overview)
- [ARX Accelerator Features](#arx-accelerator-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [ARX-Specific Usage](#arx-specific-usage)
- [HARX Dialect](#harx-dialect)
- [Compilation Pipeline](#compilation-pipeline)
- [Testing](#testing)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

MLIR-ARX extends the standard ONNX-MLIR compiler with:

- **HARX Dialect**: High-level ARX-specific MLIR dialect for hardware acceleration
- **ARX Accelerator Backend**: Specialized code generation for ARX hardware
- **Optimized Compilation Paths**: Multiple compilation strategies for different use cases
- **Quantization Support**: INT8 quantization optimized for ARX hardware
- **Comprehensive Testing**: Validation across different compilation pipelines

## ARX Accelerator Features

### Hardware Accelerated Operations

The ARX accelerator provides hardware acceleration for:

- **Convolution Operations**: `harx.ConvolutionShift` with integrated quantization
- **Matrix Multiplication**: `harx.MatMul` with bias and scaling support
- **Pooling Operations**: `harx.MaxPool` with configurable kernels
- **Quantization/Dequantization**: `harx.Quantization`, `harx.Dequantization`
- **Data Layout**: `harx.Transpose`, `harx.Constant` for efficient memory access

### Key Benefits

- **Hardware-Specific Optimization**: Direct mapping to ARX instruction set
- **Quantized Inference**: Native INT8 support for reduced memory and faster execution
- **Efficient Memory Management**: Optimized data layouts and memory access patterns
- **Low Latency**: Direct hardware execution without software emulation

---

## Installation

### Build Instructions

### Checking LLVM Version Dependencies

The LLVM version used by ONNX-MLIR is determined based on the file located at the following path:
`./third_party/stablehlo/build_tools/llvm_version.txt`

### Environment Setup: Docker and Visual Studio Code Dev Container

This project can be built and run in two environments.

#### 1. Docker

1. Repository Clone
```bash
$ git clone --recursive https://gitlab.com/ones-ai/mlir-arx
$ cd mlir-arx
```
2. Image Build
```bash
$ docker build -f ./docker/Dockerfile .
```
3. Run Container  
```bash
$ docker run -it <container_id> /bin/bash
```

#### 2. Visual Studio Code Dev Container (in Devcontainer)

Using Dev Containers makes it easy to set up your development environment.

1. Open Container in VS Code
   - Open VS Code and navigate to your project folder via `File → Open Folder...`.
   - Open the Command Palette (`Ctrl+Shift+P`) and select "Dev Containers: rebuild without cache and reopen in container".
2. Working Inside the Container  
   - Once the container starts, run the following command in the embedded terminal to keep the repository up-to-date:
```bash
$ git pull
```

You can now edit, build, and debug your code via VS Code within the Dev Container environment.

All prerequisite environment setup for ONNX-MLIR builds is complete. The LLVM and MLIR environments are now configured, and builds proceed using the base image defined in `.devcontainer/Dockerfile`.

### Process of MLIR-ARX Build

Detailed information is defined in the [`.gitlab-ci.yml`](.gitlab-ci.yml) file, which describes how to configure the build environment using Docker files and Dev Containers. The main steps of the build process, which runs automatically within the CI/CD pipeline.

#### 1. Requirment: before build

```sh
$ cd /workdir
$ mkdir -p build && cd build
```

#### 2. Build
Use the command below to generate the CMake configuration file for the Ninja build system, then proceed with the build. The key point is the `CMAKE_BUILD_TYPE` option. Building in Release mode may cause issues with the program, so please ensure you build in Debug mode.

```sh
$ cmake -G Ninja -DLLVM_DIR=/build/lib/cmake/llvm \
         -DCMAKE_CXX_COMPILER=${LLVM_PROJECT_ROOT}/build/bin/clang++ \
         -DCMAKE_C_COMPILER=${LLVM_PROJECT_ROOT}/build/bin/clang-20 \
         -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=OFF \
         -DMLIR_DIR=${LLVM_PROJECT_ROOT}/build/lib/cmake/mlir \
         -DONNX_MLIR_ACCELERATORS=ARX \
         -DCMAKE_LIBRARY_PATH=${LLVM_PROJECT_ROOT}/build/lib ..
$ cmake --build .
```

`LLVM_PROJECT_ROOT`는 `./.devcontainer/Dockerfile`에 `ENV`로 정의가 되어 있습니다. 

---

## Quick Start
### Want to test right away?

If you have Ubuntu 22.04 and protobuf installed, you can download the `artifacts` from the `build` pipeline at `https://gitlab.com/ones-ai/mlir-arx/-/artifacts` and run the binaries directly.

### Looking for example usage code?

The script `./test/mlir-arx/run.py` contains automated validation routines. The code is simple Python, making it easy to understand. While integration with CTEST or GTEST was considered, the `ONNX-MLIR` project uses its own unique test code structure, which increased implementation complexity. Therefore, independent test scripts were created.

The workflow is straightforward: for example, when testing lowering from onnx to harx, the script reads files from each folder and compares the compiler output.



### Test with MNIST Example

```bash
cd test/mlir-arx

# Run all ARX test pipelines
python run.py

# Expected output:
# ✅ MNIST : C++ to ARX C++
# ✅ MNIST : EmitC to C++  
# ✅ MNIST : Onnx to EmitC
# ✅ MNIST : Onnx to ARX
# ✅ MNIST : Onnx to ONNX-MLIR
```

## ARX-Specific Usage

### Compilation Options

#### Target ARX Hardware
```bash
# Specify ARX accelerator
onnx-mlir --maccel=ARX model.onnx
```

#### Compilation Modes
```bash
# Generate HARX dialect (direct ARX mapping)
onnx-mlir --maccel=ARX --EmitARXIR --EmitMLIR model.onnx

# Generate EmitC for C++ code generation
onnx-mlir --maccel=ARX --EmitEmitCIR --EmitMLIR model.onnx

# Standard ONNX MLIR with ARX optimizations
onnx-mlir --maccel=ARX --EmitMLIR model.onnx
```

## HARX Dialect

The HARX (Hardware ARX) dialect provides high-level operations that map directly to ARX hardware capabilities.

### Core Operations

#### Convolution with Quantization
```mlir
%result = "harx.ConvolutionShift"(%input, %weights, %shift_scale, %bias) {
  do_bias = 1 : ui8,
  do_relu = 0 : ui8,
  output_offset = 0 : ui8,
  pad_size = 2 : ui8,
  stride_size = 1 : ui8
} : (tensor<1x1x28x28xui8>, tensor<8x1x5x5xui8>, tensor<8xsi32>, tensor<8xsi32>) 
  -> tensor<1x8x28x28xui8>
```

#### Matrix Multiplication
```mlir
%output = "harx.MatMul"(%input, %weights, %bias, %shift_scale) {
  do_bias = 1 : ui8,
  output_offset = 112 : i32
} : (tensor<1x256xui8>, tensor<256x10xui8>, tensor<1x10xsi32>, tensor<1x10xsi32>) 
  -> tensor<1x10xui8>
```

#### Pooling
```mlir
%pooled = "harx.MaxPool"(%input) {
  kernel_height = 2 : ui8,
  kernel_width = 2 : ui8,
  pad_size = 0 : ui8,
  stride_size = 2 : ui8
} : (tensor<1x8x28x28xui8>) -> tensor<1x8x14x14xui8>
```

#### Quantization
```mlir
%quantized = "harx.Quantization"(%input) {
  offset = 0 : ui8,
  scale = 1.000000e+00 : f32
} : (tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xui8>
```

## Compilation Pipeline

MLIR-ARX supports multiple compilation strategies:

### 1. Direct ARX Pipeline (Recommended)
```
ONNX Model → HARX Dialect → ARX Assembly → Binary
```
**Command**: `onnx-mlir --maccel=ARX --EmitARXIR model.onnx`

**Benefits**: 
- Maximum performance
- Direct hardware mapping
- Optimized memory layout

### 2. EmitC Pipeline (Portable)
```
ONNX Model → EmitC Dialect → C++ Code → ARX Binary
```
**Command**: `onnx-mlir --maccel=ARX --EmitEmitCIR model.onnx`

**Benefits**:
- Cross-platform compatibility
- Easier debugging
- Standard C++ toolchain

### 3. Standard ONNX Pipeline (Fallback)
```
ONNX Model → ONNX MLIR → LLVM IR → Binary
```
**Command**: `onnx-mlir --maccel=ARX --EmitMLIR model.onnx`

**Benefits**:
- Full ONNX compatibility
- Gradual ARX adoption
- Debugging reference

## Testing

### Using Docker (Recommended)

The easiest way to build and test MLIR-ARX is using the provided Docker scripts:

1. **Build the project in a container**:
   ```bash
   # Build the Docker container with all dependencies
   ./docker/build-in-container.sh
   ```

2. **Run tests in the container**:
   ```bash
   # Execute the complete test suite
   ./docker/run-tests.sh
   ```

3. **Access container for debugging**:
   ```bash
   # Start an interactive container session
   ./docker/run-container.sh
   # Then SSH into the container: ssh -p 44123 root@localhost
   ```

The Docker scripts provide:
- Automated dependency installation
- Consistent build environment
- Isolated testing environment
- Test result artifacts in `artifacts/test_results/`

### Manual Testing

If you prefer to run tests manually:

```bash
cd test/mlir-arx
python run.py
```

### Individual Test Categories

```bash
# Test ARX C++ code generation
./code_to_arx.py -i ./code/mnist_onnx.h -o ./arx/mnist_onnx.h

# Test EmitC to C++ conversion
mlir-translate -mlir-to-cpp ./emitc/onnx_to_emitc.mlir -o output.h

# Test ONNX to ARX compilation
onnx-mlir --maccel=ARX --EmitARXIR --EmitMLIR ./onnx/mnist-12-int8.onnx
```

### Test Files Structure
```
test/mlir-arx/
├── onnx/           # Input ONNX models
├── emitc/          # EmitC MLIR files
├── harx/           # HARX MLIR files
├── mlir/           # Standard ONNX MLIR
├── code/           # Generated C++ code
├── arx/            # ARX-optimized C++ code
└── test_code/      # Test utilities
```

## Examples

### MNIST Classification

Complete MNIST example with ARX acceleration:

1. **Prepare the model**:
   ```bash
   cd docs/mnist_example
   python gen_mnist_onnx.py --export-onnx
   ```

2. **Compile for ARX**:
   ```bash
   onnx-mlir --maccel=ARX -O3 mnist.onnx
   ```

3. **Run inference**:
   ```c++
   #include "OnnxMlirRuntime.h"
   
   // Load ARX-compiled model
   OMExecutionSession session("mnist.so");
   
   // Run inference
   auto outputs = session.run({input_tensor});
   ```

### Custom Model Integration

For your own models:

```bash
# Convert PyTorch/TensorFlow model to ONNX
python convert_to_onnx.py --model your_model --output model.onnx

# Quantize for ARX (optional but recommended)
python quantize_model.py --input model.onnx --output model_int8.onnx

# Compile with ARX acceleration
onnx-mlir --maccel=ARX -O3 model_int8.onnx

# Integrate in your application
# See docs/mnist_example/ for complete integration examples
```

## Performance Tips

### Optimization Best Practices

1. **Use Quantization**: INT8 models perform significantly better on ARX
   ```bash
   # Always prefer quantized models
   onnx-mlir --maccel=ARX -O3 model_int8.onnx
   ```

2. **Choose Right Pipeline**: Use direct ARX pipeline for maximum performance
   ```bash
   # Best performance
   onnx-mlir --maccel=ARX --EmitARXIR model.onnx
   ```

3. **Optimize Data Layout**: Ensure compatible tensor layouts
   ```bash
   # Check model compatibility
   onnx-mlir --maccel=ARX --check-compatibility model.onnx
   ```

### Memory Optimization

- **Batch Size**: Optimize batch size for ARX memory architecture
- **Tensor Sizes**: Align tensor dimensions with ARX requirements  
- **Memory Layout**: Use ARX-optimized memory layouts when possible

## Troubleshooting

### Common Issues

#### Compilation Errors
```bash
# Check ARX compatibility
onnx-mlir --maccel=ARX --verify model.onnx

# Debug with verbose output
onnx-mlir --maccel=ARX --debug-level=3 model.onnx
```

#### Runtime Issues
```bash
# Check ARX driver installation
arx-info --version

# Verify model format
onnx-mlir --maccel=ARX --validate model.so
```

#### Performance Issues
```bash
# Profile ARX execution
onnx-mlir --maccel=ARX --enable-profiling model.onnx

# Check optimization levels
onnx-mlir --maccel=ARX -O3 --print-optimizations model.onnx
```

## Contributing

### Development Setup

1. **Fork and clone**:
   ```bash
   git clone https://github.com/your-fork/mlir-arx.git
   cd mlir-arx
   ```

2. **Setup development environment**:
   ```bash
   # Install pre-commit hooks
   pip install pre-commit
   pre-commit install
   
   # Build in debug mode
   mkdir build-debug && cd build-debug
   cmake -DCMAKE_BUILD_TYPE=Debug -DONNX_MLIR_ENABLE_ARX=ON ..
   make -j$(nproc)
   ```

### Adding ARX Operations

1. **Define HARX operation** in `src/Accelerators/ARX/Dialect/HARX/HARX.td`
2. **Implement operation logic** in `src/Accelerators/ARX/Dialect/HARX/HARXOps/`
3. **Add ONNX lowering** in `src/Accelerators/ARX/Conversion/ONNXToHARX/`
4. **Add EmitC lowering** in `src/Accelerators/ARX/Conversion/HARXToLLVM/`
5. **Write tests** in `test/mlir-arx/`

### Testing Guidelines

```bash
# Run ARX-specific tests
cd test/mlir-arx
python run.py

# Add new test cases
echo "New test description" >> test_cases.txt

# Validate against reference
./validate_output.py --reference ref.mlir --test new.mlir
```

### Code Style

- Follow MLIR coding conventions
- Use ARX-specific naming: `harx.*` for dialect operations
- Add comprehensive documentation for new features
- Include performance benchmarks for optimizations

## Architecture

### ARX Accelerator Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX-MLIR Frontend                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 ONNX Dialect                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────▼────────┐
              │ ARX Accelerator │
              │   Selection     │
              └───────┬────────┘
                      │
         ┌────────────▼─────────────┐
         │                          │
┌────────▼─────────┐    ┌──────────▼──────────┐
│   HARX Dialect   │    │   EmitC Dialect     │
│  (Direct ARX)    │    │  (Portable C++)     │
└────────┬─────────┘    └──────────┬──────────┘
         │                         │
┌────────▼─────────┐    ┌──────────▼──────────┐
│ ARX Code Gen     │    │  C++ Code Gen       │
└────────┬─────────┘    └──────────┬──────────┘
         │                         │
         └────────────┬────────────┘
                      │
              ┌───────▼────────┐
              │ ARX Hardware   │
              │   Execution    │
              └────────────────┘
```

## License

This project is licensed under Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report bugs via GitHub Issues
- **Community**: Join the discussion on ONNX-MLIR slack channels
- **Examples**: Check `test/mlir-arx/` and `docs/mnist_example/` for working examples

## Acknowledgments

This project builds upon the excellent work of the ONNX-MLIR community and extends it with ARX hardware acceleration capabilities. Special thanks to all contributors who made this enhanced version possible.