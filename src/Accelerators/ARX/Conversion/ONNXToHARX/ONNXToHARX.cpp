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
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARXCommon.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"

using namespace mlir;

//
// LSTM/GRU specific functions
//

namespace onnx_mlir {

SmallVector<Value, 4> emitONNXSplitOp(Location loc, PatternRewriter &rewriter,
    Value input, IntegerAttr axis, ArrayAttr split) {
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  SmallVector<Type> outputTypes;
  int64_t splitNum = split.size();
  ArrayRef<int64_t> inputShape =
      mlir::cast<RankedTensorType>(input.getType()).getShape();
  int64_t splitAxis = mlir::cast<IntegerAttr>(axis).getSInt();
  assert(splitAxis >= 0 && "Negative axis");
  for (int i = 0; i < splitNum; i++) {
    SmallVector<int64_t> outputShape;
    for (size_t dim = 0; dim < inputShape.size(); dim++) {
      outputShape.emplace_back(
          (dim == (unsigned int)splitAxis)
              ? mlir::cast<IntegerAttr>(split[dim]).getInt()
              : inputShape[dim]);
    }
    outputTypes.emplace_back(RankedTensorType::get(outputShape, elementType));
  }
  ONNXSplitV11Op splitOp =
      rewriter.create<ONNXSplitV11Op>(loc, outputTypes, input, axis, split);
  return splitOp.getResults();
}

/// Get kernelShapes using shape helper
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
SmallVector<int64_t, 2> getArrayKernelShape(OP op) {
  OPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();

  // Check if kernelShape is literal. Only static value is supported.
  assert((llvm::any_of(shapeHelper.kernelShape, [](IndexExpr val) {
    return val.isLiteral();
  })) && "Only support static kernel_shape ");

  SmallVector<int64_t, 2> kernelShapesRet;
  kernelShapesRet.emplace_back(shapeHelper.kernelShape[0].getLiteral());
  kernelShapesRet.emplace_back(shapeHelper.kernelShape[1].getLiteral());
  return kernelShapesRet;
}

/// Get strides using shape helper
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
SmallVector<int64_t, 2> getArrayStrides(OP op) {
  OPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  return shapeHelper.strides;
}

//===----------------------------------------------------------------------===//
// ONNX to HARX Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXONNXToHARX.inc"

// Enhance 'replaceONNXSumOpPatternRecursion' to allow operating recursively.
struct ONNXSumOpPatternEnhancedRecursion
    : public replaceONNXSumOpPatternRecursion {
  ONNXSumOpPatternEnhancedRecursion(MLIRContext *context)
      : replaceONNXSumOpPatternRecursion(context) {}
  void initialize() {
    // This pattern recursively unpacks one variadic operand at a time. The
    // recursion bounded as the number of variadic operands is strictly
    // decreasing.
    setHasBoundedRewriteRecursion(true);
  }
};

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

void getONNXToHARXOneOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  populateWithGenerated(patterns);
  patterns.insert<ONNXSumOpPatternEnhancedRecursion>(context);
}

void getONNXToHARXOneOpDynamicallyLegal(
    ConversionTarget *target, const DimAnalysis *dimAnalysis) {
  addDynamicallyLegalOpFor<ONNXAddOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSubOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMulOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXDivOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSumOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXMinOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXMaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReluOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXTanhOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXSigmoidOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXLogOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXExpOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSoftmaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMaxPoolSingleOutOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXAveragePoolOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMatMulOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXGemmOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXReduceMeanV13Op>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXLSTMOp>(target, dimAnalysis);
  // addDynamicallyLegalOpFor<ONNXGRUOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXConvOp>(target, dimAnalysis);
}

void getONNXToHARXMultipleOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<replaceONNXMatMulAddPattern1>(context);
  patterns.insert<replaceONNXMatMulAddPattern2>(context);
  patterns.insert<replaceONNXReluConvPattern>(context);
  // patterns.insert<replaceONNXLogSoftmaxPattern>(context);
  // Shape inference for newly-added operations.
  getShapeInferencePatterns(patterns);
}

void ONNXToHARXLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // Enable reporting on ARX unsupported ops when specifying
  // `--opt-report=ARXUnsupportedOps`.
  OnnxToHARXLoweringConfiguration::reportOnARXUnsupportedOps =
      OnnxToHARXLoweringConfiguration::optReportARXUnsupportedOps;

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, harx::HARXDialect, KrnlDialect,
      func::FuncDialect, arith::ArithDialect>();

  // NOTE: if we change the order of calling combinedPatterns and single op
  // patterns, make sure to change the order in DevicePlacement.cpp also to make
  // them synced.

  // Combined ONNX ops to HARX lowering.
  // There are some combinations of ONNX ops that can be lowering into a single
  // HARX op, e.g. ONNXMatMul and ONNXAdd can be lowered to HARXMatmul.
  // The lowering of such combinations should be done before the lowering of
  // a single ONNX Op, because the single op lowering might have conditions that
  // prohibit the combined ops lowering happened.
  RewritePatternSet combinedPatterns(&getContext());
  onnx_mlir::getONNXToHARXMultipleOpPatterns(combinedPatterns);

  // It's ok to fail.
  (void)applyPatternsAndFoldGreedily(module, std::move(combinedPatterns));

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  // Single ONNX to HARX operation lowering.
  RewritePatternSet patterns(&getContext());
  onnx_mlir::getONNXToHARXOneOpPatterns(patterns);

  // This is to make sure we don't want to alloc any MemRef at this high-level
  // representation.
  target.addIllegalOp<mlir::memref::AllocOp>();
  target.addIllegalOp<mlir::memref::DeallocOp>();

  // ONNX ops to HARX dialect under specific conditions.
  // When adding a new op, need to implement a method, i.e. isSuitableForZDNN,
  // for the op in ONNXLegalityCheck.cpp.
  getONNXToHARXOneOpDynamicallyLegal(&target, &dimAnalysis);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createONNXToHARXPass() {
  return std::make_unique<ONNXToHARXLoweringPass>();
}

} // namespace onnx_mlir
