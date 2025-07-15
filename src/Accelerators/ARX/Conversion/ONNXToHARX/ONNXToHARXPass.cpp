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
} // end anonymous namespace.

void ONNXToHARXLoweringPass::runOnOperation() {
  llvm::outs() << "ONNXToHARXLoweringPass::runOnOperation\n";
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // Enable reporting on ARX unsupported ops when specifying
  // `--opt-report=ARXUnsupportedOps`.
  // OnnxToHARXLoweringConfiguration::reportOnARXUnsupportedOps =
      // OnnxToHARXLoweringConfiguration::optReportARXUnsupportedOps;

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, harx::HARXDialect, func::FuncDialect, arith::ArithDialect>();

  // NOTE: if we change the order of calling combinedPatterns and single op
  // patterns, make sure to change the order in DevicePlacement.cpp also to make
  // them synced.

  // Combined ONNX ops to HARX lowering.
  // There are some combinations of ONNX ops that can be lowering into a single
  // HARX op, e.g. ONNXMatMul and ONNXAdd can be lowered to HARXMatmul.
  // The lowering of such combinations should be done before the lowering of
  // a single ONNX Op, because the single op lowering might have conditions that
  // prohibit the combined ops lowering happened.
  // RewritePatternSet combinedPatterns(&getContext());
  // onnx_mlir::getONNXToHARXMultipleOpPatterns(combinedPatterns);

  // It's ok to fail.
  // (void)applyPatternsAndFoldGreedily(module, std::move(combinedPatterns));

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  // Single ONNX to HARX operation lowering.
  // onnx_mlir::getONNXToHARXOneOpPatterns(patterns);
  
  // This is to make sure we don't want to alloc any MemRef at this high-level
  // representation.
  // target.addIllegalOp<mlir::memref::AllocOp>();
  // target.addIllegalOp<mlir::memref::DeallocOp>();
  target.addIllegalOp<ONNXAddOp>();
  // target.addIllegalDialect<ONNXDialect>();
  
  llvm::outs() << "1rewritePatternONNXToKrnl CALLED!!!!\n";
  llvm::outs() << "2rewritePatternONNXToKrnl CALLED!!!!\n";
  MyTypeConverter krnlTypeConverter;
  RewritePatternSet patterns(&getContext());
  onnx_mlir::harx::populateONNXToARXConversionPattern(krnlTypeConverter, patterns, &getContext(), false);

  // ONNX ops to HARX dialect under specific conditions.
  // When adding a new op, need to implement a method, i.e. isSuitableForZDNN,
  // for the op in ONNXLegalityCheck.cpp.
  // getONNXToHARXOneOpDynamicallyLegal(&target, &dimAnalysis);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  module.walk([&](Operation *op) {
    llvm::outs() << "Processing operation: " << op->getName() << "\n";
    llvm::outs() << "Found ONNX operation: " << op->getDialect() << "\n";
      // llvm::outs() << "Found ONNX operation: " << op->getName() << "\n";
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createONNXToHARXPass() {
  return std::make_unique<ONNXToHARXLoweringPass>();
}

} // namespace onnx_mlir
