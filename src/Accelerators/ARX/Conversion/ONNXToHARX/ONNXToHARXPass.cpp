/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToHARX.cpp - ONNX dialect to HARX lowering -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and HARX operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARX.hpp"
// #include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARXCommon.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
// #include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/HARXTypeConverter.hpp"

#include <iterator>

using namespace mlir;


namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX to HARX Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXONNXToHARX.inc"

struct ONNXToHARXLoweringPass
    : public PassWrapper<ONNXToHARXLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToHARXLoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-harx"; }

  StringRef getDescription() const override {
    return "Lower ONNX ops to HARX ops.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ONNXToHARXLoweringPass() = default;
  ONNXToHARXLoweringPass(const ONNXToHARXLoweringPass &pass)
      : PassWrapper<ONNXToHARXLoweringPass, OperationPass<ModuleOp>>() {}
  void runOnOperation() final;
};
} 

void ONNXToHARXLoweringPass::runOnOperation() {
  llvm::outs() << "ONNXToHARXLoweringPass::runOnOperation\n";
  ModuleOp module = getOperation();

  ConversionTarget target(getContext());

  target.addLegalDialect<harx::HARXDialect, func::FuncDialect, arith::ArithDialect>();

  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  target.addIllegalOp<ONNXEntryPointOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp fn) {
    // Check all arguments for an "onnx.name" attribute:
    for (unsigned i = 0, e = fn.getFunctionType().getNumInputs(); i < e; ++i)
      if (fn.getArgAttr(i, "onnx.name"))
        return false;
    // Check all results for an "onnx.name" attribute:
    for (unsigned i = 0, e = fn.getFunctionType().getNumResults(); i < e; ++i)
      if (fn.getResultAttr(i, "onnx.name"))
        return false;
    return true;  // no onnx.name attrs → legal
  });
  target.addIllegalOp<ONNXQuantizeLinearOp, ONNXDequantizeLinearOp, ONNXReshapeOp, ONNXMaxPoolSingleOutOp, ONNXQLinearConvOp>();

  target.addDynamicallyLegalOp<ONNXCustomOp>([](ONNXCustomOp op) { 
    if (op.getFunctionName() != "QLinearAdd") {
      llvm::dbgs() << "QLinearAdd is legal, return true.\n";
      return true; // QLinearAdd is not legal.
    }

    auto op2 = op.getInputs()[0].getDefiningOp<ONNXQLinearMatMulOp>();
    if (!op2) {
      llvm::dbgs() << "QLinearMatMulOp not found, return true.\n";
      return true; // If no QLinearMatMulOp, then legal.
    }

    auto size = std::distance(op2.getResult().getUses().begin(), op2.getResult().getUses().end());

    if (size == 1) { 
      return false;
    }else { 
      return true;
    }
  });

  MyTypeConverter krnlTypeConverter;
  
  RewritePatternSet patterns(&getContext());

  onnx_mlir::harx::populateONNXToARXConversionPattern(krnlTypeConverter, patterns, &getContext(), false);


  if (failed(applyPartialConversion(module, target, std::move(patterns)))){
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createONNXToHARXPass() {
  return std::make_unique<ONNXToHARXLoweringPass>();
}

} // namespace onnx_mlir
