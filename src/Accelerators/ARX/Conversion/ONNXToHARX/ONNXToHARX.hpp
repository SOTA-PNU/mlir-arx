/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ZHighToZLow.hpp - ZHigh dialect to ZLow lowering -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to ZLow operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_TO_ZLOW_H
#define ONNX_MLIR_ZHIGH_TO_ZLOW_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/Mlir/IndexExpr.hpp"

/// Default 4K alignment for sticked tensors.
static constexpr int64_t gAlignment = 4096;

namespace onnx_mlir {
namespace harx {

/// Populate all conversion patterns for ZHigh Ops.
void populateONNXToARXConversionPattern(mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx, bool enableParallel);

} // namespace harx
} // namespace onnx_mlir
#endif
