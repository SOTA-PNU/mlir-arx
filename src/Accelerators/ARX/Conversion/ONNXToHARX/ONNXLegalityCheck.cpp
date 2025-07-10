/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ONNXLegalityCheck.cpp - Check legality for ONNX ops -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods to check whether an ONNX op is suitable for
// being lowered to zDNN or not.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXLegalityCheck.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARXCommon.hpp"
#include "src/Accelerators/ARX/Support/ARXLimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

/// Report ARX unsupported case.
bool onnxToHARXUnsupportedReport(Operation *op, const std::string &message) {
  if (OnnxToHARXLoweringConfiguration::reportOnARXUnsupportedOps &&
      !message.empty()) {
    StringAttr opName = op->getName().getIdentifier();
    std::string nodeNameStr = getNodeNameInPresenceOfOpt(op);
    printf("==ARX-UNSUPPORTEDOPS-REPORT==, %s, %s, %s\n", opName.data(),
        nodeNameStr.c_str(), message.c_str());
  }
  return false;
}

/// Report incompatibility with ARX Level.
bool onnxToHARXInCompatibilityReport(
    Operation *op, std::string inputARXLevel) {
  std::string message =
      "onnx-mlir ARX level (" + inputARXLevel +
      ") is not compatible with  ARX level specified by '-mcpu'(" + mcpu +
      ").";
  return onnxToHARXUnsupportedReport(op, message);
}

/// A function to check whether a value's element type is valid for zAIU or not.
/// zAIU supports only F16, F32 and BFLOAT. Since MLIR does not support BFLOAT,
/// we check F16 and F32 here only. zAIU only supports rank in range of (0, 4].
bool isValidElementTypeAndRank(Operation *op, Value val, bool donotCheckRank) {
  if (mlir::isa<NoneType>(val.getType()))
    return true;
  if (auto valueType = mlir::dyn_cast_or_null<ShapedType>(val.getType())) {
    Type elementType = (valueType) ? valueType.getElementType() : val.getType();
    // Element type must be in 16 or F32.
    if (mlir::isa<FloatType>(elementType) &&
        (mlir::cast<FloatType>(elementType).getWidth() == 16 ||
            mlir::cast<FloatType>(elementType).getWidth() == 32)) {
      if (donotCheckRank)
        return true;
      // Rank must be in range of (0, 4].
      if (!valueType.hasRank()) {
        std::string message = "Value does not have rank.";
        return onnxToHARXUnsupportedReport(op, message);
      }
      int64_t rank = valueType.getRank();
      if ((rank == 0) || (rank > 4)) {
        std::string message =
            "Rank " + std::to_string(rank) +
            " is not supported. zAIU only supports rank in range of (0, 4].";
        return onnxToHARXUnsupportedReport(op, message);
      }
      return true;
    } else {
      std::string message = "Element type is not F16 or F32.";
      return onnxToHARXUnsupportedReport(op, message);
    }
  }
  std::string message = "Value is not shaped type.";
  return onnxToHARXUnsupportedReport(op, message);
}

/// Common legality check for pooling ops.
template <typename POOLOP, typename POOLOPAdaptor, typename POOLOPShapeHelper>
bool checkLegalityPoolOpsCommon(
    POOLOP op, Value Y, const DimAnalysis *dimAnalysis) {
  POOLOPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  Value X = op.getX();
  int64_t ceilMode = op.getCeilMode();
  ShapedType inputType = mlir::cast<ShapedType>(X.getType());
  ShapedType outputType = mlir::cast<ShapedType>(Y.getType());
  ArrayRef<int64_t> shapeInput = inputType.getShape();
  ArrayRef<int64_t> shapeOutput = outputType.getShape();

  // 4D tensors(N x C x H x W) are supported as input and output.
  if (shapeInput.size() != 4 || shapeOutput.size() != 4) {
    std::string message = "4D tensors(N x C x H x W) are supported as input "
                          "and output, but the input dim size (" +
                          std::to_string(shapeInput.size()) +
                          ") is not 4, or output dim size (" +
                          std::to_string(shapeOutput.size()) + ") is not 4.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // ceil_mode not supported.
  if (ceilMode != 0) {
    std::string message = "ceil_mode not supported";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // `getStrPaddingType` returns `SAME_PADDING`, `VALID_PADDING`, or empty.
  // zDNN only support padding for `SAME_PADDING` and `VALID_PADDING`.
  // When input has unknown dimension and auto_pad is `NOTSET`, paddingType is
  // empty.
  StringRef paddingType =
      getStrPaddingType<POOLOP, POOLOPAdaptor, POOLOPShapeHelper>(op);
  if (paddingType.empty()) {
    std::string message =
        "Padding type must be `SAME_PADDING` or `VALID_PADDING`, but it is "
        "neither of them. When attribute `auto_pad` is `NOTSET`, padding is "
        "computed from input dimention etc, but when input has unknown "
        "dimensions, it can't be computed.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // Check "MaxPool2D/AvgPool2D Parameter Restrictions". These restrictions are
  // described in "zDNN API Reference". Input tensor N(batchNum) and C(Channel)
  // dimensions must always match the output tensor's respective dimensions.
  if (!dimAnalysis->sameDim(X, 0, Y, 0) || !dimAnalysis->sameDim(X, 1, Y, 1)) {
    std::string message =
        "Batch dimension in input tensor (" + std::to_string(shapeInput[0]) +
        ") and in output tensor (" + std::to_string(shapeOutput[0]) +
        ") are not the same, or channel dimension in input tensor (" +
        std::to_string(shapeInput[1]) + ") and in output tensor (" +
        std::to_string(shapeOutput[1]) + ") are not the same.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // Check if kernelShape is literal. Only static value is supported.
  if (llvm::any_of(shapeHelper.kernelShape,
          [](IndexExpr val) { return !val.isLiteral(); })) {
    std::string message = "The kernel_shape must be static value.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // Check parameter restrictions for maxpool2d/avgpool2d for each axis only
  // when input and output are of static tensor type. When unknown dimensions
  // are included, the restrictions are not checked and error messages are
  // generated at runtime in zDNN.
  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    int64_t inputShapeH = shapeInput[2];
    int64_t inputShapeW = shapeInput[3];
    int64_t outputShapeH = shapeOutput[2];
    int64_t outputShapeW = shapeOutput[3];
    int64_t kernelShapeH = shapeHelper.kernelShape[0].getLiteral();
    int64_t kernelShapeW = shapeHelper.kernelShape[1].getLiteral();
    int64_t stridesH = shapeHelper.strides[0];
    int64_t stridesW = shapeHelper.strides[1];
    bool checkH = meetPoolParamRestrictions(op.getOperation(), inputShapeH,
        kernelShapeH, stridesH, outputShapeH, paddingType);
    if (!checkH)
      return false;
    bool checkW = meetPoolParamRestrictions(op.getOperation(), inputShapeW,
        kernelShapeW, stridesW, outputShapeW, paddingType);
    if (!checkW)
      return false;
  }

  // No check for tensors with unknown dimensions.
  return true;
}

/// Get padding type using shape helper. This returns
/// `SAME_PADDING`, `VALID_PADDING`, or empty.
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
StringRef getStrPaddingType(OP op) {
  IndexExprBuilderForAnalysis createIE(op.getLoc());
  OPShapeHelper shapeHelper(op.getOperation(), {}, &createIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  auto autoPad = op.getAutoPad();
  if (autoPad == "SAME_UPPER")
    return "SAME_PADDING";
  else if (autoPad == "SAME_LOWER")
    // zDNN not support SAME_LOWER.
    return StringRef();
  else if (autoPad == "VALID")
    return "VALID_PADDING";
  else {
    // Only support static pads at this moment being.
    if (llvm::any_of(
            shapeHelper.pads, [](IndexExpr val) { return !val.isLiteral(); }))
      return StringRef();

    // VALID_PADDING.
    if (llvm::all_of(shapeHelper.pads,
            [&](IndexExpr val) { return (val.getLiteral() == 0); }))
      return "VALID_PADDING";

    if (autoPad == "NOTSET") {
      // Pads must be set manually.
      // Check pad values according to SAME_PADDING in zDNN:
      // More information about SAME_PADDING:
      // https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow/
      // https://github.ibm.com/zosdev/AI-on-Z/wiki/SAME_PADDING
      //
      // height_out = math.ceil(height_in / strides[0])
      // width_out = math.ceil(width_in / strides[1])
      // pad_along_height
      //   = max((height_out - 1) * strides[0] + kernel_size[0] - height_in, 0)
      // pad_along_width
      //   = max((width_out - 1) * strides[1] + kernel_size[1] - width_in, 0)
      // pad_top = pad_along_height // 2
      // pad_bottom = pad_along_height - pad_top
      // pad_left = pad_along_width // 2
      // pad_right = pad_along_width - pad_left

      // Input height and width.
      IndexExpr hi = createIE.getShapeAsDim(op.getX(), 2);
      IndexExpr wi = createIE.getShapeAsDim(op.getX(), 3);
      if (!hi.isLiteral() || !wi.isLiteral())
        return StringRef();
      // Output height and width.
      IndexExpr ho = shapeHelper.getOutputDims()[2];
      IndexExpr wo = shapeHelper.getOutputDims()[3];
      if (!ho.isLiteral() || !wo.isLiteral())
        return StringRef();

      // Check if output height and width are correct.
      if (ho.getLiteral() != hi.ceilDiv(shapeHelper.strides[0]).getLiteral())
        return StringRef();
      if (wo.getLiteral() != wi.ceilDiv(shapeHelper.strides[1]).getLiteral())
        return StringRef();

      // Compute pad values according to zDNN.
      LiteralIndexExpr zeroIE(0);
      LiteralIndexExpr twoIE(2);
      IndexExpr padH =
          (ho - 1) * shapeHelper.strides[0] + shapeHelper.kernelShape[0] - hi;
      IndexExpr padW =
          (wo - 1) * shapeHelper.strides[1] + shapeHelper.kernelShape[1] - wi;
      IndexExpr pH = IndexExpr::max(padH, zeroIE);
      IndexExpr pW = IndexExpr::max(padW, zeroIE);
      if (!pH.isLiteral() || !pW.isLiteral())
        return StringRef();

      IndexExpr pHTop = pH.floorDiv(twoIE);
      IndexExpr pHBottom = pH - pHTop;
      IndexExpr pWLeft = pW.floorDiv(twoIE);
      IndexExpr pWRight = pW - pWLeft;
      // Compare ONNX pads and zDNN pads.
      if (pHTop.getLiteral() == shapeHelper.pads[0].getLiteral() &&
          pWLeft.getLiteral() == shapeHelper.pads[1].getLiteral() &&
          pHBottom.getLiteral() == shapeHelper.pads[2].getLiteral() &&
          pWRight.getLiteral() == shapeHelper.pads[3].getLiteral())
        return "SAME_PADDING";
    }
  }

  return StringRef();
}

/// Check if input, output, kernel, strides, and paddingType for each axis meet
/// parameter restrictions for maxpool. See "MaxPool2D Parameter Restrictions"
/// in "zDNN API Reference"
bool meetPoolParamRestrictions(Operation *op, int64_t inputShape,
    int64_t kernelShape, int64_t strides, int64_t outputShape,
    StringRef paddingType) {
  // TODO: Shape inference fails when `strides` is zero.
  // (third_party/onnx-mlir/src/Dialect/ONNX/ONNXOps.cpp:L204). So strides==0
  // case is not tested. Need to investigate how to handle this.
  if (strides == 0) {
    // Both input tensor's Height/Width dimension and the kernel_height/width
    // must match
    if (inputShape != kernelShape) {
      std::string message = "When the strides is zero, both input tensor's "
                            "Height or Width dimension (" +
                            std::to_string(inputShape) +
                            ")  and the kernel_height or width (" +
                            std::to_string(kernelShape) + ") must match.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    // inputShape and kernelShape are less than or equal to 1024.
    if (inputShape > 1024) {
      std::string message =
          "When the strides is zero, the inputShape and kernelShape (" +
          std::to_string(inputShape) + ") must be less than or equal to 1024.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    // Output tensor's height and width dimensions must be 1.
    if (outputShape != 1) {
      std::string message = "When the strides is zero, output tensor's height "
                            "or width dimensions (" +
                            std::to_string(outputShape) + ") must be 1.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    // padding_type must be VALID_PADDING.
    if (!(paddingType == "VALID_PADDING")) {
      std::string message = "When the strides is zero, padding type (" +
                            paddingType.str() + ") must be VALID_PADDING.";
      return onnxToHARXUnsupportedReport(op, message);
    }
  } else {
    // strides are greater than zero
    // kernel_width and kernel_height must be less than or equal to 64.
    if (kernelShape > 64) {
      std::string message = "When the strides (" + std::to_string(strides) +
                            ") are greater than zero, the "
                            "kernel_width and kernel_height (" +
                            std::to_string(kernelShape) +
                            ") must be less than or equal to 64.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    if (paddingType == "SAME_PADDING") {
      int64_t reqOutputShape = ceil(static_cast<float>(inputShape) / strides);
      if (outputShape != reqOutputShape) {
        std::string message =
            "When the strides (" + std::to_string(strides) +
            ") and the padding type is `SAME_PADDING`, output tensor's height "
            "or "
            "width dimensions (" +
            std::to_string(outputShape) +
            ") must be equal with ceil((float)inputShape / strides)) (=" +
            std::to_string(reqOutputShape) + ").";
        return onnxToHARXUnsupportedReport(op, message);
      }
    } else { // VALID_PADDING
      int64_t reqOutputShape =
          ceil(static_cast<float>(inputShape - kernelShape + 1) / strides);
      if (outputShape != reqOutputShape) {
        std::string message = "When the strides (" + std::to_string(strides) +
                              ") and the padding type is VALID_PADDING, output "
                              "tensor's height or width dimensions (" +
                              std::to_string(outputShape) +
                              ") must be equal with ceil((float)(inputShape - "
                              "kernelShape + 1) / strides)) (= " +
                              std::to_string(reqOutputShape) + ").";
        return onnxToHARXUnsupportedReport(op, message);
      }
    }
  }
  return true;
}

/// Default legality check.
template <typename OP_TYPE>
bool isSuitableForZDNN(OP_TYPE op, const DimAnalysis *dimAnalysis) {
  return false;
}

/// Check legality for ONNXAdd.
// zDNN Add, Sub, Mul, Div do not support broadcasting.
template <>
bool isSuitableForZDNN<ONNXAddOp>(
    ONNXAddOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16)) {
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  }
  if (!isValidElementTypeAndRank(op.getOperation(), op.getA()))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), op.getB()))
    return false;
  if (!dimAnalysis->sameShape(op.getA(), op.getB()))
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "The dynamic dimension analysis couldn't identify "
        "input `A` and `B` have the same shape.");
  return true;
}

/// Check legality for ONNXSub.
template <>
bool isSuitableForZDNN<ONNXSubOp>(
    ONNXSubOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  if (!isValidElementTypeAndRank(op.getOperation(), op.getA()))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), op.getB()))
    return false;
  if (!dimAnalysis->sameShape(op.getA(), op.getB()))
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "The dynamic dimension analysis couldn't identify "
        "input `A` and `B` have the same shape.");
  return true;
}

/// Check legality for ONNXMul.
template <>
bool isSuitableForZDNN<ONNXMulOp>(
    ONNXMulOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  if (!isValidElementTypeAndRank(op.getOperation(), op.getA()))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), op.getB()))
    return false;
  if (!dimAnalysis->sameShape(op.getA(), op.getB()))
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "The dynamic dimension analysis couldn't identify "
        "input `A` and `B` have the same shape.");
  return true;
}

/// Check legality for ONNXDiv.
template <>
bool isSuitableForZDNN<ONNXDivOp>(
    ONNXDivOp op, const DimAnalysis *dimAnalysis) {
  Value A = op.getA();
  Value B = op.getB();
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  // Broadcast with a scalar operand.
  // if (isEnableScalarBcastBinary()) {
  //   if (isF32ScalarConstantTensor(A) &&
  //       isValidElementTypeAndRank(op.getOperation(), B))
  //     return true;
  //   if (isF32ScalarConstantTensor(B) &&
  //       isValidElementTypeAndRank(op.getOperation(), A))
  //     return true;
  // }
  // Non-broadcast cases.
  if (!isValidElementTypeAndRank(op.getOperation(), A))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), B))
    return false;
  if (!dimAnalysis->sameShape(A, B))
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "The dynamic dimension analysis couldn't identify "
        "input `A` and `B` have the same shape.");
  return true;
}

/// Check legality for ONNXSum.
template <>
bool isSuitableForZDNN<ONNXSumOp>(
    ONNXSumOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  // Do not support a single input.
  if (op.getData_0().size() < 2)
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "The input `data_0` needs to include multiple tensors.");
  // Check data type.
  if (!isValidElementTypeAndRank(op.getOperation(), op.getData_0()[0]))
    return false;
  // All inputs must have the same static shape.
  for (unsigned int i = 1; i < op.getData_0().size(); ++i) {
    // Check data type.
    if (!isValidElementTypeAndRank(op.getOperation(), op.getData_0()[i]))
      return false;
    if (!dimAnalysis->sameShape(op.getData_0()[0], op.getData_0()[i])) {
      std::string message =
          "The dimension analysis could not identify the shape of the first "
          "tensor in `data_0` is the same as that of " +
          std::to_string(i) + "-th tensor";
      return onnxToHARXUnsupportedReport(op.getOperation(), message);
    }
  }
  return true;
}

/// Check legality for ONNXSoftmax.
/// zDNN softmax only supports axis = rank-1 (or -1) when rank = 2 or 3). If
/// axis is not rank-1 (or -1) when rank = 2/3), keep ONNXSoftmax unchanged.
template <>
bool isSuitableForZDNN<ONNXSoftmaxOp>(
    ONNXSoftmaxOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  if (!isValidElementTypeAndRank(op.getOperation(), op.getInput()))
    return false;
  ShapedType inputType = mlir::cast<ShapedType>(op.getType());
  if (!inputType.hasRank())
    return onnxToHARXUnsupportedReport(
        op.getOperation(), "The `input` tensor doesn't have the rank.");
  int64_t rank = inputType.getRank();
  if ((rank != 2) && (rank != 3))
    return onnxToHARXUnsupportedReport(
        op.getOperation(), "The rank of input tensor (" + std::to_string(rank) +
                               ") needs to be 2 or 3.");
  if ((op.getAxis() != rank - 1) && (op.getAxis() != -1))
    return onnxToHARXUnsupportedReport(
        op.getOperation(), "The `axis` (" + std::to_string(op.getAxis()) +
                               ") needs to be `rank - 1`(" +
                               std::to_string(rank - 1) + ") or -1.");
  return true;
}

/// Check legality for ONNXRelu.
template <>
bool isSuitableForZDNN<ONNXReluOp>(
    ONNXReluOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  if (!isValidElementTypeAndRank(op.getOperation(), op.getX()))
    return false;
  return true;
}

/// Check legality for ONNXTanh.
template <>
bool isSuitableForZDNN<ONNXTanhOp>(
    ONNXTanhOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  if (!isValidElementTypeAndRank(op.getOperation(), op.getInput()))
    return false;
  return true;
}

/// Check legality for ONNXMatMul.
template <>
bool isSuitableForZDNN<ONNXMatMulOp>(
    ONNXMatMulOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);
  int64_t opnum = op.getNumOperands();
  if (opnum != 2)
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "The number of operands is " + std::to_string(opnum) + ", not 2.");
  if (!isValidElementTypeAndRank(op.getOperation(), op.getOperand(0))) {
    return false;
  }
  if (!isValidElementTypeAndRank(op.getOperation(), op.getOperand(1))) {
    return false;
  }
  ShapedType aType = mlir::cast<ShapedType>(op.getOperand(0).getType());
  ShapedType bType = mlir::cast<ShapedType>(op.getOperand(1).getType());

  // Illegal if A or B is unranked.
  if (!aType.hasRank() || !bType.hasRank())
    return onnxToHARXUnsupportedReport(
        op.getOperation(), "A or B is unranked.");

  auto shapeA = aType.getShape();
  auto shapeB = bType.getShape();

  // In case of Tensors with unknown dimension, check only size of matrices.
  // Actual shape is not checked. If actual shape does not meet, get error at
  // runtime.
  // TODO: Support other cases
  // (https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul) on zDNN
  // by using broadcasting etc.
  if ((shapeA.size() == 2) && (shapeB.size() == 2)) {
    // unstacked case
    if (aType.hasStaticShape() && bType.hasStaticShape()) {
      if (shapeA[1] != shapeB[0]) {
        std::string message = "Unstacked case: the 2nd dim of A (" +
                              std::to_string(shapeA[1]) +
                              ") and the 1st dim of B (" +
                              std::to_string(shapeB[0]) + ") are not the same.";
        return onnxToHARXUnsupportedReport(op.getOperation(), message);
      }
    }
    return true;
  } else if ((shapeA.size() == 3) && (shapeB.size() == 3)) {
    // stacked w/o bcast case
    if (aType.hasStaticShape() && bType.hasStaticShape()) {
      if ((shapeA[0] != shapeB[0]) || (shapeA[2] != shapeB[1])) {
        std::string message =
            "Stacked w/o bcast case: the 1st dim of A (" +
            std::to_string(shapeA[0]) + ") and the 1st dim of B (" +
            std::to_string(shapeB[0]) +
            ") are not the same, or the 3rd dim of A (" +
            std::to_string(shapeA[2]) + ") and the 2nd dim of B (" +
            std::to_string(shapeB[1]) + ") are not the same.";
        return onnxToHARXUnsupportedReport(op.getOperation(), message);
      }
    }
    return true;
  } else if ((shapeA.size() == 3) && (shapeB.size() == 2)) {
    // stacked w/ bcast
    if (aType.hasStaticShape() && bType.hasStaticShape()) {
      if (shapeA[2] != shapeB[0]) {
        std::string message = "Stacked w/ bcast case: the 3rd dim of A (" +
                              std::to_string(shapeA[2]) +
                              ") and the 1st dim of B (" +
                              std::to_string(shapeB[0]) + ") are not the same.";
        return onnxToHARXUnsupportedReport(op.getOperation(), message);
      }
    }
    return true;
  }
  std::string message = "Dim size of A(" + std::to_string(shapeA.size()) +
                        ") and B(" + std::to_string(shapeB.size()) +
                        ") is not supported.";
  return onnxToHARXUnsupportedReport(op.getOperation(), message);
}

/// Check legality for ONNXGemm.
template <>
bool isSuitableForZDNN<ONNXGemmOp>(
    ONNXGemmOp op, const DimAnalysis *dimAnalysis) {
  Value A = op.getA();
  Value B = op.getB();
  Value C = op.getC();

  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);

  // Check data type.
  if (!isValidElementTypeAndRank(op.getOperation(), A))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), B))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), C))
    return false;

  ShapedType aType = mlir::cast<ShapedType>(A.getType());
  ShapedType bType = mlir::cast<ShapedType>(B.getType());
  ShapedType cType;
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  ArrayRef<int64_t> cShape;

  bool hasC = !isNoneValue(C);
  if (hasC) {
    cType = mlir::cast<ShapedType>(C.getType());
    cShape = cType.getShape();
  }

  // Element type must be f32.
  if (!aType.getElementType().isF32() || !bType.getElementType().isF32() ||
      (hasC && !cType.getElementType().isF32()))
    return onnxToHARXUnsupportedReport(
        op.getOperation(), "Element type of `A` and `B` must be f32.");
  ;
  // A and B's rank must be 2 and C's rank must be 1 or 2.
  if ((aShape.size() != 2) || (bShape.size() != 2) ||
      (hasC && (cShape.size() != 1) && (cShape.size() != 2))) {
    std::string message = "The rank of A(" + std::to_string(aShape.size()) +
                          ") and B (" + std::to_string(bShape.size()) +
                          ") must be 2. ";
    message += hasC ? "The rank of C(" + std::to_string(cShape.size()) +
                          ") must be 1 or 2."
                    : "";
    return onnxToHARXUnsupportedReport(op.getOperation(), message);
  }

  ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
  float alpha = gemmOp.getAlpha().convertToFloat();
  float beta = gemmOp.getBeta().convertToFloat();
  if (alpha != 1.0 || beta != 1.0) {
    std::string message = "`alpha` (" + std::to_string(alpha) +
                          ")  and `beta` (" + std::to_string(beta) +
                          ") must be 1.";
    return onnxToHARXUnsupportedReport(op.getOperation(), message);
  }
  auto bShape1 = gemmOp.getTransB() ? bShape[0] : bShape[1];
  // If C's rank is 1: Only support B's second dim is the same with C's dim
  // (A(m, n) * B(n, p) + C(p))
  if (hasC && cShape.size() == 1) {
    // Cannot check broadcasting at compile time.
    std::string message = "When the rank of C is 1, ";
    if (ShapedType::isDynamic(cShape[0])) {
      message += "The first dim of `C` need to be static dim.";
      return onnxToHARXUnsupportedReport(op.getOperation(), message);
    }
    if (cShape[0] != bShape1) {
      message += "The 2nd dim of B` must be the same as the first dim of `C`.";
      return onnxToHARXUnsupportedReport(op.getOperation(), message);
    }
  }
  return true;
}

/// Check legality for ONNXMaxPool.
template <>
bool isSuitableForZDNN<ONNXMaxPoolSingleOutOp>(
    ONNXMaxPoolSingleOutOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);

  // Check data type.
  if (!isValidElementTypeAndRank(op.getOperation(), op.getX()))
    return false;

  ONNXMaxPoolSingleOutOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();

  if (!checkLegalityPoolOpsCommon<ONNXMaxPoolSingleOutOp,
          ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>(
          op, op.getO_Y(), dimAnalysis))
    return false;

  // dilations not supported. Only default one is accepted.
  if (shapeHelper.dilations[0] != 1 || shapeHelper.dilations[1] != 1) {
    std::string message =
        "The `dilations` (" + std::to_string(shapeHelper.dilations[0]) + ", " +
        std::to_string(shapeHelper.dilations[1]) +
        ") is not supported. Only default `dilations` (1, 1) is supported.";
    return onnxToHARXUnsupportedReport(op.getOperation(), message);
  }

  return true;
}

/// Check legality for ONNXAveragePool.
template <>
bool isSuitableForZDNN<ONNXAveragePoolOp>(
    ONNXAveragePoolOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);

  // Check data type.
  if (!isValidElementTypeAndRank(op.getOperation(), op.getX()))
    return false;

  // count_include_pad not supported.
  if (op.getCountIncludePad() != 0)
    return onnxToHARXUnsupportedReport(op.getOperation(),
        "`count_include_pad` (" + std::to_string(op.getCountIncludePad()) +
            ") must be default one (0).");

  return checkLegalityPoolOpsCommon<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor,
      ONNXAveragePoolOpShapeHelper>(op, op.getY(), dimAnalysis);
}

/// Check if input, output, kernel, strides, and paddingType for each axis meet
/// parameter restrictions for conv2d. See "Conv2D Parameter Restrictions"
/// in "zDNN API Reference"
static bool checkConv2DParamRestrictions(Operation *op, int64_t inputDim,
    int64_t kernelDim, int64_t stride, int64_t outputDim,
    StringRef paddingType) {
  if (stride == 0) {
    // paddingType must be VALID_PADDING.
    if (!(paddingType == "VALID_PADDING")) {
      std::string message = "When the strides (" + std::to_string(stride) +
                            ") is zero, padding type (" + paddingType.str() +
                            ") must be VALID_PADDING.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    // inputDim must be = kernel dim.
    if (inputDim != kernelDim) {
      std::string message = "When the strides (" + std::to_string(stride) +
                            ") is zero, both input tensor's "
                            "Height or Width dimension (" +
                            std::to_string(inputDim) +
                            ")  and the kernel_height or width (" +
                            std::to_string(kernelDim) + ") must match.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    // inputDim and kernelDim are less than or equal to 448.
    if (inputDim > 448) {
      std::string message =
          "When the strides (" + std::to_string(stride) +
          ") is zero, the input tensor's Height or Width dimension (" +
          std::to_string(inputDim) + ") must be less than or equal to 448.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    // outputDim must be 1.
    if (outputDim != 1) {
      std::string message = "When the strides (" + std::to_string(stride) +
                            ") is zero, output tensor's height "
                            "or width dimensions (" +
                            std::to_string(outputDim) + ") must be 1.";
      return onnxToHARXUnsupportedReport(op, message);
    }
  } else if (stride > 0 && stride <= 13) {
    // stride is greater than zero and less than or equal to 13.
    // kernel dim must be less than or equal to 64.
    if (kernelDim > 64) {
      std::string message =
          "When the strides (" + std::to_string(stride) +
          ") is greater than zero and less than or equal to 13, "
          "kernel_width and kernel_height (" +
          std::to_string(kernelDim) + ") must be less than or equal to 64.";
      return onnxToHARXUnsupportedReport(op, message);
    }
    if (paddingType == "SAME_PADDING") {
      // height_out restriction.
      int64_t reqOutputShape = ceil(static_cast<float>(inputDim) / stride);
      if (outputDim != reqOutputShape) {
        std::string message =
            "When the strides (" + std::to_string(stride) +
            ")  is greater than zero and less than or equal to 13, output "
            "tensor's height or width dimensions (" +
            std::to_string(outputDim) +
            ") must be equal with ceil((float)inputShape / stride)) (=" +
            std::to_string(reqOutputShape) + ").";
        return onnxToHARXUnsupportedReport(op, message);
      }
    } else { // VALID_PADDING
      // inputDim must be >= kernelDim.
      if (inputDim < kernelDim) {
        std::string message =
            "When the strides (" + std::to_string(stride) +
            ") is greater than zero and less than or equal to 13 and the "
            "padding type is VALID_PADDING, input tensor's height or width "
            "dimensions (" +
            std::to_string(inputDim) +
            ") must be less than kernel height or width dimension (" +
            std::to_string(kernelDim) + ").";
        return onnxToHARXUnsupportedReport(op, message);
      }
      // height_out restriction.
      int64_t reqOutputShape =
          ceil(static_cast<float>(inputDim - kernelDim + 1) / stride);
      if (outputDim != reqOutputShape) {
        std::string message =
            "When the strides (" + std::to_string(stride) +
            ") is greater than zero and less than or equal to "
            "13 and the padding type is VALID_PADDING, output "
            "tensor's height or width dimensions (" +
            std::to_string(outputDim) +
            ") must be equal with ceil((float)(inputShape - "
            "kernelShape + 1) / strides)) (= " +
            std::to_string(reqOutputShape) + ").";
        return onnxToHARXUnsupportedReport(op, message);
      }
    }
  } else {
    std::string message = "When the strides (" + std::to_string(stride) +
                          ") must be less than or equal to 13.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  return true;
}

/// Check legality for ONNXConvOp.
template <>
bool isSuitableForZDNN<ONNXConvOp>(
    ONNXConvOp op, const DimAnalysis *dimAnalysis) {
  // Check ARX level.
  if (!isCompatibleWithARXLevel(ARX_Z16))
    return onnxToHARXInCompatibilityReport(op.getOperation(), ARX_Z16);

  // Check data type.
  if (!isValidElementTypeAndRank(op.getOperation(), op.getX()))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), op.getW()))
    return false;
  if (!isValidElementTypeAndRank(op.getOperation(), op.getB()))
    return false;

  ONNXConvOpAdaptor operandAdaptor = ONNXConvOpAdaptor(op);
  ONNXConvOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();

  ShapedType inputType = mlir::cast<ShapedType>(op.getX().getType());
  ShapedType outputType = mlir::cast<ShapedType>(op.getY().getType());
  ArrayRef<int64_t> shapeInput = inputType.getShape();
  ArrayRef<int64_t> shapeOutput = outputType.getShape();

  // 4D tensors(N x C x H x W) are supported as input and output.
  if (shapeInput.size() != 4 || shapeOutput.size() != 4) {
    std::string message = "4D tensors(N x C x H x W) are supported as input "
                          "and output, but the input dim size (" +
                          std::to_string(shapeInput.size()) +
                          ") is not 4, or output dim size (" +
                          std::to_string(shapeOutput.size()) + ") is not 4.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // Do not support dynamic height and width dimensions since we can not check
  // them at compile time.
  if (ShapedType::isDynamic(shapeInput[2]) ||
      ShapedType::isDynamic(shapeInput[3]) ||
      ShapedType::isDynamic(shapeOutput[2]) ||
      ShapedType::isDynamic(shapeOutput[3]))
    return onnxToHARXUnsupportedReport(op,
        "Height and/or width have dynamic dimensions. They are not supported.");

  // Do not support group.
  if (operandAdaptor.getGroup() != 1)
    return onnxToHARXUnsupportedReport(op, "`group` must be 1 (default).");

  // Do not support non-default dilations.
  if (shapeHelper.dilations[0] != 1 || shapeHelper.dilations[1] != 1) {
    std::string message =
        "The `dilations` (" + std::to_string(shapeHelper.dilations[0]) + ", " +
        std::to_string(shapeHelper.dilations[1]) +
        ") is not supported. Only default `dilations` (1, 1) is supported.";
    return onnxToHARXUnsupportedReport(op.getOperation(), message);
  }

  // `getStrPaddingType` returns `SAME_PADDING`, `VALID_PADDING`, or empty.
  // `zdnn_conv2d` only support padding for `SAME_PADDING` and `VALID_PADDING`.
  StringRef paddingType =
      getStrPaddingType<ONNXConvOp, ONNXConvOpAdaptor, ONNXConvOpShapeHelper>(
          op);

  if (paddingType.empty()) {
    std::string message =
        "Padding type must be `SAME_PADDING` or `VALID_PADDING`, but it is "
        "neither of them. When attribute `auto_pad` is `NOTSET`, padding is "
        "computed from input dimention etc, but when input has unknown "
        "dimensions, it can't be computed.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  // Check if kernelShape is literal. Only static value is supported.
  if (llvm::any_of(shapeHelper.kernelShape,
          [](IndexExpr val) { return !val.isLiteral(); })) {
    std::string message = "The kernel_shape must be static value.";
    return onnxToHARXUnsupportedReport(op, message);
  }

  int64_t inputShapeH = shapeInput[2];
  int64_t inputShapeW = shapeInput[3];
  int64_t outputShapeH = shapeOutput[2];
  int64_t outputShapeW = shapeOutput[3];
  int64_t kernelShapeH = shapeHelper.kernelShape[0].getLiteral();
  int64_t kernelShapeW = shapeHelper.kernelShape[1].getLiteral();
  int64_t stridesH = shapeHelper.strides[0];
  int64_t stridesW = shapeHelper.strides[1];

  // Check parameter restrictions for conv2d for each axis.
  bool isHOK = checkConv2DParamRestrictions(
      op, inputShapeH, kernelShapeH, stridesH, outputShapeH, paddingType);
  if (!isHOK)
    return false;
  bool isWOK = checkConv2DParamRestrictions(
      op, inputShapeW, kernelShapeW, stridesW, outputShapeW, paddingType);
  if (!isWOK)
    return false;

  return true;
}

/// Check legality for ONNXReshapeOp.
template <>
bool isSuitableForZDNN<ONNXReshapeOp>(
    ONNXReshapeOp op, const DimAnalysis *dimAnalysis) {
  // Noop Reshape is suitable for zAIU as this pass removes such reshape ops.
  return isIdentityReshape(op, dimAnalysis);
}
