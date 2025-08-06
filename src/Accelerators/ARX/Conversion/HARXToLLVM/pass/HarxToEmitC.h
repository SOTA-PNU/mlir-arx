#ifndef ONNX_MLIR_ONNX_TO_HARX_PASS_EMIT_C_H
#define ONNX_MLIR_ONNX_TO_HARX_PASS_EMIT_C_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVM.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace larx {

// struct ReturnToEmitCPattern : public mlir::OpRewritePattern<func::ReturnOp> {
//   using OpRewritePattern::OpRewritePattern;
//   LogicalResult matchAndRewrite(func::ReturnOp op, mlir::PatternRewriter &rewriter) const override;
// };

struct FunctionToEmitCPattern : public mlir::OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op, mlir::PatternRewriter &rewriter) const override;
};

struct QuantizeToEmitCPattern : public mlir::OpRewritePattern<harx::HARXQuantizationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(harx::HARXQuantizationOp op, mlir::PatternRewriter &rewriter) const override;
};

struct DequantizeToEmitCPattern : public mlir::OpRewritePattern<harx::HARXDequantizationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(harx::HARXDequantizationOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ConvolutionShiftToEmitCPattern : public mlir::OpRewritePattern<harx::HARXConvolutionShiftOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(harx::HARXConvolutionShiftOp op, mlir::PatternRewriter &rewriter) const override;
};

struct MaxPoolToEmitCPattern : public mlir::OpRewritePattern<harx::HARXMaxPoolOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(harx::HARXMaxPoolOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReShapeToEmitCPattern : public mlir::OpRewritePattern<harx::HARXTransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(harx::HARXTransposeOp op, mlir::PatternRewriter &rewriter) const override;
};

struct FullyConnectedToEmitCPattern : public mlir::OpRewritePattern<harx::HARXMatMulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(harx::HARXMatMulOp op, mlir::PatternRewriter &rewriter) const override;
};


struct ConstantToEmitCPattern : public ConvertToLLVMPattern {
  ConstantToEmitCPattern(LLVMTypeConverter &converter, MLIRContext *ctx) : ConvertToLLVMPattern(harx::HARXConstantOp::getOperationName(), ctx, converter) {}
  LogicalResult matchAndRewrite(Operation *ops, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override;
};

// struct MaxPoolToEmitC : public ConvertToLLVMPattern {
//   MaxPoolToEmitC(LLVMTypeConverter &converter, MLIRContext *ctx) : ConvertToLLVMPattern(harx::HARXMaxPoolOp::getOperationName(), ctx, converter) {}
//   LogicalResult matchAndRewrite(Operation *ops, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override;
// };

// struct QuantizeToEmitC : public ConvertToLLVMPattern {
//   QuantizeToEmitC(LLVMTypeConverter &converter, MLIRContext *ctx) : ConvertToLLVMPattern(harx::HARXQuantizationOp::getOperationName(), ctx, converter) {}
//   LogicalResult matchAndRewrite(Operation *ops, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override;
// };


} // namespace larx
} // namespace onnx_mlir
#endif
