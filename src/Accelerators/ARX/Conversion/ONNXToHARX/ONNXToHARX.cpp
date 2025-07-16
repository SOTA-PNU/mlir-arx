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

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"

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

#define DEBUG_TYPE "harx-to-larx"

using namespace mlir;
using namespace onnx_mlir::harx;

namespace onnx_mlir {
namespace harx {

// using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    // MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilder, SCFBuilder>;

// int ZHighToZLowStickifiedConstantOpLowering::constantID = 0;

template <typename OP_TYPE>
struct HARXOpFor {
  using Op = void;
};

//===----------------------------------------------------------------------===//
// Lower ZHigh binary ops to ZLow.
//===----------------------------------------------------------------------===//

template <>
struct HARXOpFor<ONNXAddOp> {
  using Op = HARXAddOp;
};

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


/// 반환: (새 Attr, 새 Type)
static std::optional<std::pair<DenseElementsAttr, ShapedType>>
convertDenseToUInt8(DenseElementsAttr src, Builder &b) {
  ShapedType srcTy = src.getType();
  auto ctx = b.getContext();

  // u8 타입 (필요 시 signless i8로 바꿔도 됨)
  auto u8Ty = IntegerType::get(ctx, /*width=*/8, IntegerType::Unsigned);
  ShapedType dstTy = srcTy.clone(u8Ty);

  SmallVector<APInt> data;
  data.reserve(src.getNumElements());

  Type elemTy = srcTy.getElementType();
  if (elemTy.isInteger()) {
    for (APInt v : src.getValues<APInt>()) {
      uint64_t z = v.getZExtValue();
      if (z > 255) z = 255;              // saturate; 정책에 맞게 변경
      data.emplace_back(8, z);
    }
  } else {
    return std::nullopt; // 지원 안 함
  }

  auto newAttr = DenseIntElementsAttr::get(dstTy, data);
  return std::make_pair(newAttr, dstTy);
}


struct ConvertConstI64ToI8 : public OpRewritePattern<ONNXConstantOp> {
  ConvertConstI64ToI8(MLIRContext *ctx)
      : OpRewritePattern<ONNXConstantOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ONNXConstantOp op,
                                PatternRewriter &rewriter) const override {
    llvm::outs() << "ConvertConstI64ToI8 0001\n";
                                  // 1) Only fold dense i64 constants
    auto denseAttr = op.getValueAttr().dyn_cast<DenseElementsAttr>();
    if (!denseAttr) 
      return failure();
    llvm::outs() << "ConvertConstI64ToI8 0000\n";
    auto shapedTy = denseAttr.getType().dyn_cast<ShapedType>();
    if (!shapedTy) 
      return failure();
    llvm::outs() << "ConvertConstI64ToI8 0002\n";
    auto eltTy = shapedTy.getElementType().dyn_cast<IntegerType>();
    if (!eltTy || eltTy.getWidth() != 64)
      return failure();
    llvm::outs() << "ConvertConstI64ToI8 0003\n";
    // 2) Build new 8-bit elements, clamping to [0,255]
    SmallVector<APInt> newValues;
    newValues.reserve(denseAttr.getNumElements());
    for (APInt v : denseAttr.getValues<APInt>()) {
      uint64_t z = v.getZExtValue();
      if (z > 255) z = 255;
      newValues.emplace_back(/*nbits=*/8, z);
    }

    // 3) Create the new tensor<…xi8> type
    auto ui8Ty = IntegerType::get(op.getContext(), /*width=*/8, mlir::IntegerType::Unsigned);
    auto newShapedTy = shapedTy.clone(ui8Ty);
    llvm::outs() << "ConvertConstI64ToI8 0005\n";
    // 4) Make a new DenseElementsAttr<…xi8>
    auto newDense = DenseElementsAttr::get(newShapedTy, newValues);
    llvm::outs() << "ConvertConstI64ToI8 0006\n";
    // 5) Replace with a new ONNXConstantOp carrying the i8 attr
    //    Note: ONNXConstantOp builder wants (loc, resultTypes, operands, attributes)
    auto loc = op.getLoc();
    auto newConst = rewriter.replaceOpWithNewOp<ONNXConstantOp>(
        op,
        /*resultTypes*/ TypeRange{newShapedTy},
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", newDense)
                        });
    newConst.print(llvm::outs());
    // newConst.getOutput() now has type tensor<1x10x10xi8>.
    return success();
  }
};



struct OnnxToARXFoldAdderOpLowering : public OpConversionPattern<ONNXCastOp> {
  using OpConversionPattern::OpConversionPattern;   // ctor 상속

  LogicalResult matchAndRewrite(ONNXCastOp castOut, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    llvm::outs() << "OnnxToARXCastOpLowering\n";
    // op.print(llvm::outs());
    // llvm::outs() << " :: aaa asdf\n";

    // 1) Find the Add underneath the final cast.
    auto addOp = castOut.getOperand().getDefiningOp<ONNXAddOp>();
    if (!addOp)
      return failure();

    // 2) The Add's LHS must be an ONNXCastOp (ui8→i64).
    auto castIn = addOp.getOperand(0).getDefiningOp<ONNXCastOp>();
    if (!castIn)
      return failure();

    // 3) The Add's RHS must be an ONNXConstantOp<i64>.
    auto constOp = addOp.getOperand(1).getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return failure();
    auto denseAttr = constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
    if (!denseAttr || !denseAttr.getElementType().isInteger(64))
      return failure();

    // 4) Convert the dense<i64> → dense<ui8>.
    auto newAttrAndType = convertDenseToUInt8(denseAttr, rewriter);
    if (!newAttrAndType)
      return failure();
    auto [newDenseAttr, newType] = *newAttrAndType;

    // 5) Materialize a new ui8 constant.
    // auto newConst = rewriter.create<ONNXConstantOp>(
    //     castOut.getLoc(), newType, newDenseAttr);

    auto newConst = rewriter.create<ONNXConstantOp>(
      castOut.getLoc(),
        /*resultTypes*/ TypeRange{newType},
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", newDenseAttr)
                        });

    // 6) Create the fused harx.Add on ui8 (x:uint8 + constant:uint8).
    Value x = castIn.getOperand();             // original ui8 input
    Value c = newConst.getOutput();            // the new ui8 constant
    auto newAdd = rewriter.create<HARXAddOp>(
        castOut.getLoc(),
        /*resultType=*/castOut.getResult().getType(),
        /*lhs=*/x,
        /*rhs=*/c,
        /*node_name*/ rewriter.getStringAttr("/Add"),
        /*out_scale*/    rewriter.getF32FloatAttr(1.0f),
        /*out_offset*/   rewriter.getF32FloatAttr(0.0f),
        /*x_scale*/      rewriter.getF32FloatAttr(1.0f),
        /*x_offset*/     rewriter.getF32FloatAttr(0.0f),
        /*y_scale*/      rewriter.getF32FloatAttr(1.0f),
        /*y_offset*/     rewriter.getF32FloatAttr(0.0f));

    // 7) Replace the final cast with our new Add result.
    rewriter.replaceOp(castOut, newAdd.getResult());

    // 8) Clean up the now-unused original ops.
    if (addOp.use_empty())   rewriter.eraseOp(addOp);
    if (castIn.use_empty())  rewriter.eraseOp(castIn);
    if (constOp.use_empty()) rewriter.eraseOp(constOp);

    return success();
    
  }
};


void populateONNXToARXConversionPattern(TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx, bool enableParallel) {
  llvm::outs() << "populateONNXToARXConversionPattern CALLED!!!!\n";
    llvm::outs() << HARXAddOp::getOperationName() << "\n";
    llvm::outs() << ONNXCastOp::getOperationName() << "\n";  
  // patterns.add<ConvertConstI64ToI8>(patterns.getContext());
  patterns.add<OnnxToARXFoldAdderOpLowering>(tc, patterns.getContext());
  // patterns.add<OnnxToARXCastOpLowering>(tc, patterns.getContext());

  // patterns.insert<OnnxToARXAddOpLowering>(ctx);
  // patterns.insert<OnnxToARXAddOpLowering>(ctx);
}

} // namespace harx
} // namespace onnx_mlir
