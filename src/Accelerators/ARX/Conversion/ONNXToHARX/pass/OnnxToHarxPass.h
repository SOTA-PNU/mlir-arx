#ifndef ONNX_MLIR_ONNX_TO_HARX_Pass_H
#define ONNX_MLIR_ONNX_TO_HARX_Pass_H

#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "mlir/IR/PatternMatch.h"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARX.hpp"

/// Default 4K alignment for sticked tensors.
using namespace mlir;
using namespace onnx_mlir::harx;

namespace onnx_mlir {
namespace harx {


struct EraseEntryPointPattern
    : public mlir::OpRewritePattern<ONNXEntryPointOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ONNXEntryPointOp op, mlir::PatternRewriter &rewriter) const override;
};

struct EraseFuncOnnxPattern
    : public mlir::OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(func::FuncOp op, mlir::PatternRewriter &rewriter) const override;
};

struct RewriteOnnxFCToArxFCPattern
    : public mlir::OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ONNXCustomOp op, mlir::PatternRewriter &rewriter) const override;
};

struct RewriteOnnxQuantToArxPattern
    : public mlir::OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ONNXQuantizeLinearOp op, mlir::PatternRewriter &rewriter) const override;
};

struct RewriteOnnxDequantToArxPattern
    : public mlir::OpRewritePattern<ONNXDequantizeLinearOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ONNXDequantizeLinearOp op, mlir::PatternRewriter &rewriter) const override;
};


} // namespace harx
} // namespace onnx_mlir
#endif
