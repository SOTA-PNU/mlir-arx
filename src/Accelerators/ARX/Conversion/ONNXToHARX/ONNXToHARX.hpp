/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToHARX.hpp - ONNX dialect to HARX lowering -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and HARX operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TO_ZHIGH_H
#define ONNX_MLIR_ONNX_TO_ZHIGH_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

// Exports ONNXtoHARX patterns.
void getONNXToHARXOneOpPatterns(mlir::RewritePatternSet &patterns);
void getONNXToHARXMultipleOpPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
