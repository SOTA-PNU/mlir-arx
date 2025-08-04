#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace harx {

LogicalResult HARXConstantOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

LogicalResult HARXQuantizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

LogicalResult HARXDequantizationOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

LogicalResult HARXAddOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

LogicalResult HARXReluOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

// LogicalResult inferShapeForUnaryOps(Operation *op) {
//   Value input = op->getOperand(0);
//   if (!hasShapeAndRank(input))
//     return success();
//   RankedTensorType inputType =
//       mlir::dyn_cast<RankedTensorType>(input.getType());
//   return inferShapeForUnaryOps(
//       op, inputType.getElementType(), inputType.getEncoding());
// }

LogicalResult HARXConvolutionShiftOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (getInput().getType().hasRank() && 
      getKernel().getType().hasRank() && 
      getOutputShift().getType().hasRank() && 
      getOutput().getType().hasRank()) {
      return success();
  }

  // RankedTensorType inputType =
  //     mlir::cast<RankedTensorType>(getInput().getType());
  // HARXConv2DOpShapeHelper shapeHelper(getOperation());
  // return shapeHelper.computeShapeAndUpdateType(
  //     inputType.getElementType(), inputType.getEncoding());
  return success();
}


LogicalResult HARXTransposeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return success();
}

LogicalResult HARXMaxPoolOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return success();
}



// LogicalResult HARXSubOp::inferShapes(
//     std::function<void(Region &)> doShapeInference) {
//   return inferShapeForUnaryOps(this->getOperation());
// }

// //===----------------------------------------------------------------------===//
// // MulOp

// LogicalResult HARXMulOp::inferShapes(
//     std::function<void(Region &)> doShapeInference) {
//   return inferShapeForUnaryOps(this->getOperation());
// }

// //===----------------------------------------------------------------------===//
// // DivOp

// LogicalResult HARXDivOp::inferShapes(
//     std::function<void(Region &)> doShapeInference) {
//   return inferShapeForUnaryOps(this->getOperation());
// }

// //===----------------------------------------------------------------------===//
// // ReluOp

// LogicalResult HARXReluOp::inferShapes(
//     std::function<void(Region &)> doShapeInference) {
//   return inferShapeForUnaryOps(this->getOperation());
// }

// //===----------------------------------------------------------------------===//
// // TanhOp

// LogicalResult HARXTanhOp::inferShapes(
//     std::function<void(Region &)> doShapeInference) {
//   return inferShapeForUnaryOps(this->getOperation());
// }

} // namespace harx
} // namespace onnx_mlir
