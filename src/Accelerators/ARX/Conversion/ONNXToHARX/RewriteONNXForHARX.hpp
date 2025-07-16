/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- RewriteONNXForHARX.hpp - Rewrite ONNX ops for HARX lowering ----===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements pass for rewriting of ONNX operations to generate
// combination of ONNX and HARX operations.

#ifndef ONNX_MLIR_REWRITE_HARX_H
#define ONNX_MLIR_REWRITE_HARX_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

// Exports RewriteONNXForHARX patterns.
void getRewriteONNXForHARXPatterns(
    mlir::RewritePatternSet &patterns, DimAnalysis *dimAnalysis);

// Exports RewriteONNXForHARX dynamically legal checks.
void getRewriteONNXForHARXDynamicallyLegal(
    mlir::ConversionTarget *target, const DimAnalysis *dimAnalysis);

} // namespace onnx_mlir
#endif
