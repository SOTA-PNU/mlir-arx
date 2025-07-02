/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------- DialectBuilder.hpp - ZLow Dialect Builder -----------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the ZLow Dialect Builder.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIALECT_BUILDER_H
#define ONNX_MLIR_DIALECT_BUILDER_H

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

// =============================================================================
// IndexExpr Builder for building
// =============================================================================

struct IndexExprBuilderForLARX : IndexExprBuilder {
  IndexExprBuilderForLARX(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForLARX(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForLARX(const DialectBuilder &db) : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForLARX() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// =============================================================================
// MultiDialectBuilder for ZLow
// =============================================================================

// Recursive class specialized for IndexExprBuilderForZLow referred to as
// larxIE.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForLARX, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), larxIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), larxIE(db) {}
  IndexExprBuilderForLARX larxIE;
};

} // namespace onnx_mlir
#endif
