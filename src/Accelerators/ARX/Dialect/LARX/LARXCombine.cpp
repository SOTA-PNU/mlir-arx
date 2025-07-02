/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ZLowCombine.cpp - ZLow High Level Optimizer ----------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZLow dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Accelerators/ARX/Dialect/LARX/LARXOps.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/ARX/Dialect/LARX/ONNXLARXCombine.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace larx {

/// Register optimization patterns as "canonicalization" patterns on the
/// ZLowDummyOp.
void LARXDummyOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDummyOpPattern>(context);
}

} // namespace larx
} // namespace onnx_mlir
