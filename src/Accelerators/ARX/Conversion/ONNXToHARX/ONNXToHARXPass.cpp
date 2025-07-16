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

  target.addLegalDialect<ONNXDialect, harx::HARXDialect, func::FuncDialect, arith::ArithDialect>();

  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  // target.addDynamicallyLegalOp<ONNXConstantOp>(
  //   [](ONNXConstantOp op) {
  //     auto resTy = mlir::cast<mlir::TensorType>(op.getResult().getType()).getElementType();
  //     return resTy.isUnsignedInteger(8);
  //   }
  // );

  target.addDynamicallyLegalOp<ONNXCastOp>(
    [](ONNXCastOp op) {
      auto resTy = mlir::cast<mlir::TensorType>(op.getResult().getType()).getElementType();
      return !resTy.isUnsignedInteger(8);
    }
  );

  // target.addIllegalOp<ONNXAddOp>();
  // target.addIllegalDialect<ONNXDialect>();
  
  // llvm::outs() << "1rewritePatternONNXToKrnl CALLED!!!!\n";
  // llvm::outs() << "2rewritePatternONNXToKrnl CALLED!!!!\n";
  MyTypeConverter krnlTypeConverter;
  RewritePatternSet patterns(&getContext());
  onnx_mlir::harx::populateONNXToARXConversionPattern(krnlTypeConverter, patterns, &getContext(), false);

  module.walk([&](Operation *op) {
    llvm::outs() << "Processing operation: " << op->getName() << "\n";
    llvm::outs() << "Found ONNX operation: " << op->getDialect() << "\n";
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createONNXToHARXPass() {
  return std::make_unique<ONNXToHARXLoweringPass>();
}

} // namespace onnx_mlir
