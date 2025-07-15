/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Elementwise.cpp - HARX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace harx {

//===----------------------------------------------------------------------===//
// AddOp
LogicalResult HARXAddOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

// //===----------------------------------------------------------------------===//
// // SubOp

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
