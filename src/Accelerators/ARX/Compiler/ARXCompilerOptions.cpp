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
   
    llvm::cl::opt<ARXEmissionTargetType> arxEmissionTarget(
        llvm::cl::desc("[Optional] Choose ARX-related target to emit "
                       "(once selected it will cancel the other targets):"),
        llvm::cl::values(
            clEnumVal(EmitARXIR, "Lower model to ARX IR (ARX dialect)"),
            clEnumVal(EmitEmitCIR, "Lower model to EmitCIR IR (EmitCIR dialect)"),
            clEnumVal(EmitNONE, "Do not emit ARX-related target (default)")),
        llvm::cl::init(EmitNONE), llvm::cl::cat(OnnxMlirOptions));
    
} // namespace onnx_mlir