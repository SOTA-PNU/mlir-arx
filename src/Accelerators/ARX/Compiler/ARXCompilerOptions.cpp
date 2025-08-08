/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerOptions.cpp --------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Options for NNPA
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/ARX/Compiler/ARXCompilerOptions.hpp"

#define DEBUG_TYPE "ARXCompilerOptions"

namespace onnx_mlir {

llvm::cl::opt<bool> arxEnableCuteChaCha("arx-cute-chacha",
    llvm::cl::desc("HAHA 나는 매우 귀엽지!"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

    
} // namespace onnx_mlir
