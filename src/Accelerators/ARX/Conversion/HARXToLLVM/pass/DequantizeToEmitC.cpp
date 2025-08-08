#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVM.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

#include "mlir/Target/LLVMIR/TypeToLLVM.h"   // DataLayout, TypeToLLVM
#include "src/Accelerators/ARX/Conversion/HARXToLLVM/pass/HarxToEmitC.h"

#include <random>

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace larx {


LogicalResult DequantizeToEmitCPattern::matchAndRewrite(harx::HARXDequantizationOp op, PatternRewriter &rewriter) const {
  auto ui8Ty = rewriter.getIntegerType(8, false);
  auto fp32Ty = rewriter.getF32Type();

  auto numElements = op.getInput().getType().getNumElements();
  auto numElemTy = rewriter.getIntegerType(32, false);
  auto numElemVal = rewriter.getIntegerAttr(numElemTy, numElements);
  auto numElemAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), numElemTy, numElemVal).getResult();

  auto channelSizeNum = op.getChannelSize();
  auto channelSizeTy = rewriter.getIntegerType(32, true);
  auto channelSizeVal = rewriter.getIntegerAttr(channelSizeTy, channelSizeNum);
  auto channelSizeAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), channelSizeTy, channelSizeVal).getResult();

  auto scaleVal = op.getScale();
  auto scaleTy = emitc::ArrayType::get({channelSizeNum}, rewriter.getF32Type());
  SmallVector<APFloat> scaleValues;
  scaleValues.reserve(channelSizeNum);
  for (int i = 0; i < (int)channelSizeNum; ++i) {
    scaleValues.push_back(scaleVal);
  }
  DenseFPElementsAttr scaleDenseAttr = DenseFPElementsAttr::get(scaleTy, scaleValues);
  auto scaleAttr = rewriter.create<mlir::emitc::VariableOp>(op.getLoc(), scaleTy, scaleDenseAttr).getResult();

  auto useStrideTy = rewriter.getI1Type();
  auto useStrideVal = rewriter.getIntegerAttr(useStrideTy, op.getUseStride() == 1);
  auto useStrideAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), useStrideTy, useStrideVal).getResult();

  auto harxName = op.getOperation()->getAttrOfType<mlir::StringAttr>("harx_name");
  auto retTy = op.getResult().getType();
  auto outputTy = emitc::ArrayType::get(retTy.getShape(), fp32Ty);
  auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
  auto dequantFn = module.lookupSymbol<emitc::FuncOp>("dequantize");
  
  auto offsetTy = emitc::ArrayType::get({channelSizeNum}, ui8Ty);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      harxName,
      /* type */          outputTy, Attribute(), false, true, false);

    auto offsetVal = op.getOffsetAttr();
    SmallVector<APInt> offsetVec;
    for (int i = 0; i < (int)channelSizeNum; ++i) {
      offsetVec.push_back(offsetVal.getValue());
    }
    DenseIntElementsAttr offsetDenseAttr = DenseIntElementsAttr::get(RankedTensorType::get(offsetTy.getShape(), offsetTy.getElementType()), offsetVec);
    rewriter.create<mlir::emitc::GlobalOp>(op.getLoc(), rewriter.getStringAttr(harxName.str() + "_offset"), offsetTy, offsetDenseAttr, false, true, false);
  }

  auto symRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), harxName);
  auto getGlobal = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), outputTy, symRef).getResult();

  auto offsetSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_offset"));
  auto offsetAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), offsetTy, offsetSymRef).getResult();

  auto fnInputTy = op.getInput().getType();
  auto inputTy = emitc::ArrayType::get(fnInputTy.getShape(), fnInputTy.getElementType());
  // auto inAttr = op.getInput();
  auto inAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), inputTy, op.getInput()).getResult(0);

  auto ui8Ptr = emitc::PointerType::get(ui8Ty);
  auto fp32Ptr = emitc::PointerType::get(fp32Ty);
  auto ptrOp = rewriter.getStringAttr("&");

  auto shape = inputTy.getShape();
  SmallVector<Value> shapeVec;
  auto dim = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)).getResult();
  for (int i = 0; i < (int)shape.size(); ++i) {
    shapeVec.push_back(dim);
  }
  
  auto eui8Ty = mlir::emitc::LValueType::get(ui8Ty);
  auto efp32Ty = mlir::emitc::LValueType::get(fp32Ty);

  // auto inputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, inAttr, ValueRange(shapeVec)).getResult();
  // auto inputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, inputSubAttr).getResult();
  auto inputSubPtrAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ui8Ptr, op.getInput()).getResult(0);

  auto outputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), efp32Ty, getGlobal, ValueRange(shapeVec)).getResult();
  auto outputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), fp32Ptr, ptrOp, outputSubAttr).getResult();

  auto scaleSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), efp32Ty, scaleAttr, ValueRange { dim }).getResult();
  auto scaleSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), fp32Ptr, ptrOp, scaleSubAttr).getResult();

  auto offsetSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, offsetAttr, ValueRange { dim }).getResult();
  auto offsetSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, offsetSubAttr).getResult();

  rewriter.create<mlir::emitc::CallOp>(
                          op.getLoc(),
                          dequantFn,
                          ValueRange{inputSubPtrAttr, outputSubPtrAttr, numElemAttr, scaleSubPtrAttr, offsetSubPtrAttr, channelSizeAttr, useStrideAttr});
  
  auto r = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), op.getOutput().getType(), getGlobal);
  rewriter.replaceOp(op, r.getResults());

  return mlir::success();
}


}
}