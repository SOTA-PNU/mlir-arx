/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVM.cpp - Lowering from ZLow to LLVM ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines patterns to lower ZLow operations to LLVM dialect.
//
// Note that once a type is lowed to LLVM, it can be opaque pointer and its
// element type is lost. Thus, to get element type in LLVM, get it from the
// original operation (not via operandAdaptor), then use
// `typeConverter(elementType)` to convert it to LLVM. See `ZLowStickLowering`
// as an example.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVM.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/pass/HarxToEmitC.h"


using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace larx {

// static memref::GlobalOp getOrInsertElemAddGlobal(ModuleOp module, 
//                                                  PatternRewriter &rewriter,
//                                                  MemRefType bufTy) {
//   constexpr llvm::StringLiteral kName("elemadd_scratch");

//   if (auto g = module.lookupSymbol<memref::GlobalOp>(kName))
//     return g;

//   OpBuilder::InsertionGuard g(rewriter);
//   rewriter.setInsertionPointToStart(module.getBody());

//   // mutable 전역: constant = false
//   auto global = rewriter.create<memref::GlobalOp>(
//       module.getLoc(),                    // 위치
//       /*sym_name=*/kName,     // 심볼 이름
//       /*sym_visibility=*/rewriter.getStringAttr("private"),
//       /*type=*/bufTy,
//       /*initial_value=*/Attribute(),   // 초기값 없음
//       /*constant=*/false,
//       /*alignment=*/IntegerAttr());    // 정렬(옵션)

//   return global;
// }



// struct LowerAddWithCall : public ConvertToLLVMPattern {
//   LowerAddWithCall(LLVMTypeConverter &converter, MLIRContext *ctx)
//       : ConvertToLLVMPattern(harx::HARXAddOp::getOperationName(),
//                              ctx, converter) {}

//   LogicalResult
//   matchAndRewrite(Operation *ops, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const override {
    
//     llvm::outs() << "sadfasdfddd" << "??\n";
//     llvm::outs() << "sadfasdfddd" << "??\n";
//     llvm::outs() << "sadfasdfddd" << "??\n";
//     llvm::outs() << "sadfasdfddd" << "??\n";
//     llvm::outs() << "sadfasdfddd" << "??\n";

//     auto op = cast<harx::HARXAddOp>(ops);
//     Location loc = op.getLoc();
//     auto module = ops->getParentOfType<mlir::ModuleOp>();
    
//     auto argType = mlir::cast<RankedTensorType>(op.getOperand(0).getType());
//     auto mTy = MemRefType::get(argType.getShape(), argType.getElementType());
    
//     auto fnArgTy = TypeRange{mTy, mTy, mTy, rewriter.getIntegerType(32)};
//     auto fnTy = rewriter.getFunctionType(fnArgTy, TypeRange{});
    
//     // auto bufTy = MemRefType::get({1, 10, 10}, rewriter.getIntegerType(8, false));
//     auto output_var = getOrInsertElemAddGlobal(module, rewriter, mTy);
//     auto output_catch = rewriter.create<memref::GetGlobalOp>(loc, mTy, output_var.getSymNameAttr()).getResult();
    
    
//     mlir::Value arglhs = op.getOperand(0);
//     if (mlir::isa<mlir::RankedTensorType>(arglhs.getType())) {  
//       llvm::outs() << "arglhs" << "??\n";
//       // auto allocated = rewriter.create<memref::AllocaOp>(loc, mTy);
//       // arglhs = rewriter.create<bufferization::ToMemrefOp>(loc, allocated.getType(), arglhs, allocated).getResult();
//       arglhs = rewriter.create<bufferization::ToMemrefOp>(loc, mTy, arglhs).getResult();
//     }

//     mlir::Value argrhs = op.getOperand(1);
//     if (mlir::isa<mlir::RankedTensorType>(argrhs.getType())) {  
//       llvm::outs() << "argrhs" << "??\n";
//       // auto allocated = rewriter.create<memref::AllocaOp>(loc, mTy);
//       // argrhs = rewriter.create<bufferization::ToMemrefOp>(loc, allocated.getType(), argrhs, allocated).getResult();
//       argrhs = rewriter.create<bufferization::ToMemrefOp>(loc, mTy, argrhs).getResult();
//     }
//     // if (mlir::isa<mlir::RankedTensorType>(argrhs.getType())) { 
//     //   llvm::outs() << "argrhs" << "??\n";
//     //   argrhs = rewriter.create<bufferization::ToMemrefOp>(loc, mTy, op.getOperand(1)).getResult();
//     // }
    
//     auto funcss = getOrInsertElemAdd(module, rewriter, fnTy);
//     auto cstI32 = rewriter.create<arith::ConstantIntOp>(
//         loc,
//         100,
//         rewriter.getI32Type()).getResult();

//     auto callOp = rewriter.create<func::CallOp>(loc, funcss, ValueRange { arglhs, argrhs, output_catch, cstI32});
    
//     auto output_callOp = rewriter.create<bufferization::ToTensorOp>(loc, argType, callOp.getOperand(2));
//     // output_callOp.setRestrict(true);

//     rewriter.replaceOp(ops, output_callOp);
                    
//     return success();
//   }
// };


// static memref::GlobalOp getOrInsertElemAddGlobalScrach(ModuleOp module, 
//                                                  PatternRewriter &rewriter,
//                                                  MemRefType bufTy,
//                                                 mlir::ElementsAttr default_value) {
//   constexpr llvm::StringLiteral kName("elemadd_scratch_val");

//   if (auto g = module.lookupSymbol<memref::GlobalOp>(kName))
//     return g;

//   OpBuilder::InsertionGuard g(rewriter);
//   rewriter.setInsertionPointToStart(module.getBody());

//   // mutable 전역: constant = false
//   auto global = rewriter.create<memref::GlobalOp>(
//       module.getLoc(),                    // 위치
//       /*sym_name=*/kName,     // 심볼 이름
//       /*sym_visibility=*/rewriter.getStringAttr("private"),
//       /*type=*/bufTy,
//       /*initial_value=*/default_value,   // 초기값 없음
//       /*constant=*/true,
//       /*alignment=*/rewriter.getI64IntegerAttr(8));    // 정렬(옵션)

//   return global;
// }

// struct ConstantToGlobal : public ConvertToLLVMPattern {
//   ConstantToGlobal(LLVMTypeConverter &converter, MLIRContext *ctx)
//       : ConvertToLLVMPattern(harx::HARXConstantOp::getOperationName(),
//                              ctx, converter) {}

//   LogicalResult
//   matchAndRewrite(Operation *ops, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const override {

//     auto op = cast<harx::HARXConstantOp>(ops);
//     Location loc = op.getLoc();
//     auto module = ops->getParentOfType<mlir::ModuleOp>();
    
//     auto argType = mlir::cast<RankedTensorType>(op.getValue().getType());
//     auto mTy = MemRefType::get(argType.getShape(), argType.getElementType());

//     auto global_param = getOrInsertElemAddGlobalScrach(module, rewriter, mTy, op.getValue());
//     auto output_catch = rewriter.create<memref::GetGlobalOp>(loc, mTy, global_param.getSymNameAttr()).getResult();
//     auto output_global = rewriter.create<bufferization::ToTensorOp>(loc, argType, output_catch);

//     rewriter.replaceOp(op, output_global.getResult());

//     return success();
//   }
// };




void populateHARXToLLVMConversionPattern(mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  // clang-format off
  
  // patterns.add<QuantizeToEmitC>(typeConverter, ctx);
  patterns.add<ConstantToEmitCPattern>(typeConverter, ctx);
  // patterns.add<FunctionToEmitC>(typeConverter, ctx);
  patterns.add<QuantizeToEmitCPattern>(patterns.getContext());
  // patterns.add<DequantizeToEmitCPattern>(patterns.getContext());
  // patterns.add<ConvolutionShiftToEmitCPattern>(patterns.getContext());
  // patterns.add<MaxPoolToEmitCPattern>(patterns.getContext());
  // patterns.add<FullyConnectedToEmitCPattern>(patterns.getContext());
  // patterns.add<ReShapeToEmitCPattern>(patterns.getContext());
  
  // clang-format on
}

} // namespace larx
} // namespace onnx_mlir
