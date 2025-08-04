/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ZHighToZLow.cpp - ZHigh dialect to ZLow lowering -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to ZLow operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"   

#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARX.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/ShapeHelper.hpp"
// #include "src/Accelerators/ARX/Dialect/LARX/LARXOps.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
// #include "src/Accelerators/ARX/Support/Stickify/Convert.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/pass/OnnxToHarxPass.h"

#define DEBUG_TYPE "harx-to-larx"

using namespace mlir;
using namespace onnx_mlir::harx;

namespace onnx_mlir {
namespace harx {

// struct OnnxToARXAddOpLowering : public OpConversionPattern<ONNXAddOp> {
//   using OpConversionPattern::OpConversionPattern;   // ctor 상속

//   LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {

//     if (!getTypeConverter()) return failure();
//     Type dstTy = getTypeConverter()->convertType(op.getType());
//     if (!dstTy) return failure();                   // 변환 실패 시 bail-out

//     auto ty = op.getType();
//     ty.print(llvm::outs());
//     llvm::outs() << " " << ty.getDialect().getNamespace() << "\n";

//     auto out = rewriter.create<HARXAddOp>(op.getLoc(), dstTy, adaptor.getOperands()[0], adaptor.getOperands()[1], 
//                                             rewriter.getStringAttr("/Add"), 
//                                             rewriter.getF32FloatAttr(1.0f),  // scale
//                                             rewriter.getF32FloatAttr(0.0f),  // offset
//                                             rewriter.getF32FloatAttr(1.0f),  // scale
//                                             rewriter.getF32FloatAttr(0.0f),  // offset
//                                             rewriter.getF32FloatAttr(1.0f),  // scale
//                                             rewriter.getF32FloatAttr(0.0f)); // offset

//     // llvm::outs() << "HELLO! !!!! " << op->getAttrOfType<mlir::StringAttr>("onnx_node_name").getValue() << "\n";
//     rewriter.replaceOp(op, out.getResult());
//     return success();
//   }
// };


// /// 반환: (새 Attr, 새 Type)
// static std::optional<std::pair<DenseElementsAttr, ShapedType>>
// convertDenseToUInt8(DenseElementsAttr src, Builder &b) {
//   ShapedType srcTy = src.getType();
//   auto ctx = b.getContext();

//   // u8 타입 (필요 시 signless i8로 바꿔도 됨)
//   auto u8Ty = IntegerType::get(ctx, /*width=*/8, IntegerType::Unsigned);
//   ShapedType dstTy = srcTy.clone(u8Ty);

//   SmallVector<APInt> data;
//   data.reserve(src.getNumElements());

//   Type elemTy = srcTy.getElementType();
//   if (elemTy.isInteger()) {
//     for (APInt v : src.getValues<APInt>()) {
//       uint64_t z = v.getZExtValue();
//       if (z > 255) z = 255;              // saturate; 정책에 맞게 변경
//       data.emplace_back(8, z);
//     }
//   } else {
//     return std::nullopt; // 지원 안 함
//   }

//   auto newAttr = DenseIntElementsAttr::get(dstTy, data);
//   return std::make_pair(newAttr, dstTy);
// }


// struct ConvertConstI64ToI8 : public OpRewritePattern<ONNXConstantOp> {
//   ConvertConstI64ToI8(MLIRContext *ctx)
//       : OpRewritePattern<ONNXConstantOp>(ctx, /*benefit=*/1) {}

//   LogicalResult matchAndRewrite(ONNXConstantOp op,
//                                 PatternRewriter &rewriter) const override {
//                                   // 1) Only fold dense i64 constants
//     auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(op.getValueAttr());
//     if (!denseAttr) 
//       return failure();
//     auto shapedTy = mlir::dyn_cast<ShapedType>(denseAttr.getType());
//     if (!shapedTy) 
//       return failure();
//     auto eltTy = mlir::dyn_cast<IntegerType>(shapedTy.getElementType());
//     if (!eltTy || eltTy.getWidth() != 64)
//       return failure();
//     // 2) Build new 8-bit elements, clamping to [0,255]
//     SmallVector<APInt> newValues;
//     newValues.reserve(denseAttr.getNumElements());
//     for (APInt v : denseAttr.getValues<APInt>()) {
//       uint64_t z = v.getZExtValue();
//       if (z > 255) z = 255;
//       newValues.emplace_back(/*nbits=*/8, z);
//     }

//     // 3) Create the new tensor<…xi8> type
//     auto ui8Ty = IntegerType::get(op.getContext(), /*width=*/8, mlir::IntegerType::Unsigned);
//     auto newShapedTy = shapedTy.clone(ui8Ty);
//     // 4) Make a new DenseElementsAttr<…xi8>
//     auto newDense = DenseElementsAttr::get(newShapedTy, newValues);
//     // 5) Replace with a new ONNXConstantOp carrying the i8 attr
//     //    Note: ONNXConstantOp builder wants (loc, resultTypes, operands, attributes)
//     // auto loc = op.getLoc();
//     auto newConst = rewriter.replaceOpWithNewOp<ONNXConstantOp>(
//         op,
//         /*resultTypes*/ TypeRange{newShapedTy},
//         /*operands*/     ValueRange{},
//         /*attrs*/        ArrayRef<NamedAttribute>{
//                            rewriter.getNamedAttr("value", newDense)
//                         });
//     newConst.print(llvm::outs());
//     // newConst.getOutput() now has type tensor<1x10x10xi8>.
//     return success();
//   }
// };


void populateONNXToARXConversionPattern(TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx, bool enableParallel) {
  llvm::outs() << "populateONNXToARXConversionPattern CALLED!!!!\n";
  
  patterns.add<EraseEntryPointPattern>(patterns.getContext());
  patterns.add<EraseFuncOnnxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxDequantToArxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxQuantToArxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxFCToArxFCPattern>(patterns.getContext());

}

} // namespace harx
} // namespace onnx_mlir
