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

## Installation

### Prerequisites

Same as ONNX-MLIR:
```bash
python >= 3.8
gcc >= 6.4
protobuf >= 4.21.12
cmake >= 3.13.4
make >= 4.2.1 or ninja >= 1.10.2
```

### Build Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url> mlir-arx
   cd mlir-arx
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build MLIR-ARX**:
   ```bash
   # Build with ARX support enabled
   ./utils/clone-mlir.sh
   mkdir build && cd build
   cmake -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DONNX_MLIR_ENABLE_ARX=ON \
     -DMLIR_DIR=../llvm-project/build/lib/cmake/mlir \
     -DLLVM_DIR=../llvm-project/build/lib/cmake/llvm \
     ..
   cmake --build .
   ```

4. **Verify installation**:
   ```bash
   ./build/Debug/bin/onnx-mlir --help
   ```

## Quick Start

### Basic Usage

Compile an ONNX model for ARX acceleration:

```bash
# Compile to ARX-optimized shared library
./onnx-mlir --maccel=ARX -O3 model.onnx

# Generate ARX-specific MLIR
./onnx-mlir --maccel=ARX --EmitARXIR model.onnx
```

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

#### Optimization Levels
```bash
# Maximum optimization for ARX
onnx-mlir --maccel=ARX -O3 model.onnx
```

### Environment Variables

Configure ARX-specific behavior:
```bash
# Enable ARX optimizations
export ONNX_MLIR_FLAGS="--maccel=ARX -O3"

# Custom ARX configurations
export ARX_CONFIG_PATH="/path/to/arx/config"
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

### Comprehensive Test Suite

Run all ARX tests:
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