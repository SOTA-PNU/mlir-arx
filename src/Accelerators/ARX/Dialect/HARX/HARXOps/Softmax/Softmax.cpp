/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Softmax.cpp - HARX Operations --------------------===//
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

LogicalResult HARXSoftmaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

} // namespace harx
} // namespace onnx_mlir
