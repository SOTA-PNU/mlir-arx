/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ShapeHelper.hpp - shape helpers for HARX ------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains shape computation for HARX operations.
// IndexExp is used in order to handle both static and dynamic shapes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_HARX_SHAPE_HELPER_H
#define ONNX_MLIR_HARX_SHAPE_HELPER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

namespace onnx_mlir {
namespace harx {

//===----------------------------------------------------------------------===//
// Shape helper for MatMulOp.
//===----------------------------------------------------------------------===//

struct HARXMatMulOpShapeHelper : public ONNXOpShapeHelper {
  HARXMatMulOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~HARXMatMulOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Broadcast case: X:3DS - Y:2D
  bool isBroadcasted = false;
  // Stack case: X:3DS - Y:3DS
  bool isStacked = false;
  // Keep original dimensions in this order: m, n, p if 2D or s, m, n, p if 3D.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for Conv2DOp.
//===----------------------------------------------------------------------===//

struct HARXConv2DOpShapeHelper : public ONNXOpShapeHelper {
  HARXConv2DOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~HARXConv2DOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Keep original dimensions in this order: batchsize, channel_in, height_in,
  // weight_in, channel_out, height_out, weight_out.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for PoolingOp.
//===----------------------------------------------------------------------===//

template <typename OP>
struct HARXPoolingOpShapeHelper : public ONNXOpShapeHelper {
  HARXPoolingOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~HARXPoolingOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Keep original dimensions in this order: batchsize, channel_in, height_in,
  // weight_in, height_out, weight_out.
  DimsExpr allOriginalDims;
};

//===----------------------------------------------------------------------===//
// Shape helper for UnaryOp.
//===----------------------------------------------------------------------===//

struct HARXUnaryOpShapeHelper : public ONNXUnaryOpShapeHelper {
public:
  HARXUnaryOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXUnaryOpShapeHelper(op, operands, ieBuilder, scope) {}
};

//===----------------------------------------------------------------------===//
// Shape helper for BinaryOp.
// HARX BinaryOps do not support broadcasting at this moment. Borrow UnaryOp
// shapeHelper.
//===----------------------------------------------------------------------===//

struct HARXBinaryOpShapeHelper : public ONNXUnaryOpShapeHelper {
public:
  HARXBinaryOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands = {},
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : ONNXUnaryOpShapeHelper(op, operands, ieBuilder, scope) {}
};

using HARXFixGRUYOpShapeHelper = ONNXUnaryOpShapeHelper;

} // namespace harx
} // namespace onnx_mlir
#endif
