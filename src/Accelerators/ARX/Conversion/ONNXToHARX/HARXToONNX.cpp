/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ HARXToONNX.cpp - ONNX dialect to HARX lowering -------------===//
//
// Copyright 2023- The IBM Research Authors.
//
// =============================================================================
//
// This file defines patterns to reconstruct ONNX ops from HARX ops.
//
// After all optimizations, if there are still light-weight ops (e.g. add,
// sub, ...) that are of `stick -> light-weight op -> unstick`, it's better to
// use CPU instead of ARX to avoid stick/unstick. CPU is efficient to handle
// these ops, e.g. by vectorizing the computation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// HARX to ONNX Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXHARXToONNX.inc"

struct HARXToONNXLoweringPass
    : public PassWrapper<HARXToONNXLoweringPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HARXToONNXLoweringPass)

  StringRef getArgument() const override { return "convert-harx-to-onnx"; }

  StringRef getDescription() const override {
    return "Reconstruct ONNX ops from HARX ops.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace.

void HARXToONNXLoweringPass::runOnOperation() {
  Operation *function = getOperation();
  ConversionTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  harx::HARXStickOp::getCanonicalizationPatterns(patterns, &getContext());
  harx::HARXUnstickOp::getCanonicalizationPatterns(patterns, &getContext());

  (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
}

std::unique_ptr<Pass> createHARXToONNXPass() {
  return std::make_unique<HARXToONNXLoweringPass>();
}

} // namespace onnx_mlir
