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

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilder, SCFBuilder>;

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

struct OnnxToARXAddOpLowering : public OpConversionPattern<ONNXAddOp> {
  using OpConversionPattern::OpConversionPattern;   // ctor 상속

  LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 2) 결과 타입을 TypeConverter로 변환
    llvm::outs() << "HELLO! !!!! " << ONNXAddOp::getOperationName() << "\n";
    llvm::outs() << "HELLO! !!!! " << HARXAddOp::getOperationName() << "\n";

    // getTypeConverter();
    if (!getTypeConverter()) return failure();             // TypeConverter가 없으면 실패
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy) return failure();                   // 변환 실패 시 bail-out
    llvm::outs() << "HELLO! !!!! 1\n";

    // 3) 새 ARX Add 생성 (lhs/rhs는 adaptor에서 이미 변환된 operand)
    auto qLHS = rewriter.create<HARXAddOp>(op.getLoc(), dstTy, adaptor.getOperands()[0], adaptor.getOperands()[0], rewriter.getStringAttr("/Add"));
    auto qRHS = rewriter.create<HARXAddOp>(op.getLoc(), dstTy, adaptor.getOperands()[1], adaptor.getOperands()[1], rewriter.getStringAttr("/Add2"));
    auto add2 = rewriter.create<HARXAddOp>(op.getLoc(), dstTy, qLHS, qRHS, rewriter.getStringAttr("/Ad3"));
    auto out = rewriter.create<HARXAddOp>(op.getLoc(), dstTy, add2, add2, rewriter.getStringAttr("/Add4"));
    llvm::outs() << "HELLO! !!!! " << op->getAttrOfType<mlir::StringAttr>("onnx_node_name").getValue() << "\n";
    // auto nodeName = out->getAttrOfType<mlir::StringAttr>("onnx_node_name");
    // out->setAttr("onnx_node_name", nodeName);

    llvm::outs() << "HELLO! !!!! 2\n";
    rewriter.replaceOp(op, out.getResult());
    return success();
  }
};


void populateONNXToARXConversionPattern(TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx, bool enableParallel) {
  llvm::outs() << "populateONNXToARXConversionPattern CALLED!!!!\n";
      llvm::outs() << HARXAddOp::getOperationName() << "\n";
      llvm::outs() << ONNXAddOp::getOperationName() << "\n";  
  patterns.add<OnnxToARXAddOpLowering>(tc, patterns.getContext());

  // patterns.insert<OnnxToARXAddOpLowering>(ctx);
  // patterns.insert<OnnxToARXAddOpLowering>(ctx);
}

} // namespace harx
} // namespace onnx_mlir
