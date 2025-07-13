/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ARXPasses.hpp - ARX Passes Definition ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ARX in
// addition to the passes used by ONNX MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ARX_PASSES_H
#define ONNX_MLIR_ARX_PASSES_H

#include "mlir/Pass/Pass.h"

#include "src/Accelerators/ARX/Compiler/ARXCompilerOptions.hpp"

namespace onnx_mlir {

// Add pass for device placement.
// std::unique_ptr<mlir::Pass> createDevicePlacementPass();
// std::unique_ptr<mlir::Pass> createDevicePlacementPass(
    // std::string loadConfigFile, std::string saveConfigFile,
    // ARXPlacementHeuristic placementHeuristic);

/// Add pass for lowering ONNX ops to HARX ops.
std::unique_ptr<mlir::Pass> createONNXToHARXPass();

void configureOnnxToHARXLoweringPass(bool reportOnARXUnsupportedOps);

/// Add pass for rewriting ONNX ops for HARX.
std::unique_ptr<mlir::Pass> createRewriteONNXForHARXPass();

/// Add pass for re-construct ONNX ops from HARX ops.
std::unique_ptr<mlir::Pass> createHARXToONNXPass();

/// Pass for folding std.alloc.
std::unique_ptr<mlir::Pass> createFoldStdAllocPass();

namespace harx {

/// Pass for layout propagation at HARXIR.
std::unique_ptr<mlir::Pass> createHARXLayoutPropagationPass();

/// Pass for constant propagation at HARXIR.
std::unique_ptr<mlir::Pass> createHARXConstPropagationPass();

/// Pass for clipping values to dlfloat before stickification at HARXIR.
std::unique_ptr<mlir::Pass> createHARXClipToDLFloatPass();

/// Pass for decomposing stick/unstick at HARXIR.
std::unique_ptr<mlir::Pass> createHARXDecomposeStickUnstickPass();

/// Pass for recomposing ops back to stick/unstick at HARXIR.
std::unique_ptr<mlir::Pass> createHARXRecomposeToStickUnstickPass();
} // end namespace HARX

namespace larx {

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createLARXRewritePass();

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createLARXStickExpansionPass(
    bool enableParallel = false);

/// Add pass for rewriting ZLow ops.
std::unique_ptr<mlir::Pass> createLARXDummyOpForMultiDerefPass();

} // namespace zlow
} // namespace onnx_mlir
#endif
