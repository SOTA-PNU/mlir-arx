
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


LogicalResult QuantizeToEmitCPattern::matchAndRewrite(harx::HARXQuantizationOp op, PatternRewriter &rewriter) const {
  // auto retTy = rewriter.getI32Type();
  // op.getInput()
  auto ui8Ty = rewriter.getIntegerType(8, false);
  auto fp32Ty = rewriter.getF32Type();

  auto numElements = op.getInput().getType().getNumElements();
  auto numElemTy = rewriter.getIntegerType(32, false);
  auto numElemVal = rewriter.getIntegerAttr(numElemTy, numElements);
  auto numElemAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), numElemTy, numElemVal).getResult();

  auto scaleVal = op.getScaleAttr();
  auto scaleTy = emitc::ArrayType::get({1}, fp32Ty);
  SmallVector<APFloat, 1> scaleVec;
  scaleVec.reserve(1); scaleVec.push_back(scaleVal.getValue());
  DenseFPElementsAttr scaleDenseAttr = DenseFPElementsAttr::get(scaleTy, scaleVec);
  auto scaleAttr = rewriter.create<mlir::emitc::VariableOp>(op.getLoc(), scaleTy, scaleDenseAttr).getResult();

  auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
  auto quantFn = module.lookupSymbol<emitc::FuncOp>("quantize");
  
  auto retTy = op.getResult().getType();

  auto harxName = op.getOperation()->getAttrOfType<mlir::StringAttr>("harx_name");
  auto outputTy = emitc::ArrayType::get(retTy.getShape(), ui8Ty);
  SmallVector<uint8_t> outputVec;
  outputVec.reserve(numElements);
  for (int i = 0; i < (int)numElements; ++i) {
    outputVec.push_back(0); //   APInt(8, 0, false)
  }

  auto offsetTy = emitc::ArrayType::get({1}, ui8Ty);
  {
    OpBuilder::InsertionGuard g(rewriter);
    // DenseIntElementsAttr outputDenseAttr = DenseIntElementsAttr::get(outputTy, outputVec);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      harxName,
      /* type */          outputTy, Attribute(), false, true, false);

    auto offsetVal = op.getOffsetAttr();
    
    SmallVector<APInt, 1> offsetVec;
    offsetVec.reserve(1); offsetVec.push_back(offsetVal.getValue());
    DenseIntElementsAttr offsetDenseAttr = DenseIntElementsAttr::get(RankedTensorType::get({1}, ui8Ty), offsetVec);
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(), rewriter.getStringAttr(harxName.str() + "_offset"), offsetTy, offsetDenseAttr, false, true, false);
  }

  auto offsetSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_offset"));
  auto offsetAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), offsetTy, offsetSymRef).getResult();
  
  auto symRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), harxName);
  auto getGlobal = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), outputTy, symRef).getResult();

  // auto fnInputTy = op.getInput().getType();
  // auto inputTy = emitc::ArrayType::get(fnInputTy.getShape(), fnInputTy.getElementType());
  // auto inAttr = op.getInput();
  // auto inAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), inputTy, op.getInput()).getResult(0);

  auto ui8Ptr = emitc::PointerType::get(ui8Ty);
  auto fp32Ptr = emitc::PointerType::get(fp32Ty);
  
  auto ptrOp = rewriter.getStringAttr("&");

  auto shape = outputTy.getShape();
  SmallVector<Value> shapeVec;
  auto dim = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)).getResult();
  for (int i = 0; i < (int)shape.size(); ++i) {
    shapeVec.push_back(dim);
  }
  
  auto eui8Ty = mlir::emitc::LValueType::get(ui8Ty);
  auto efp32Ty = mlir::emitc::LValueType::get(fp32Ty);

  auto getGlobalSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, getGlobal, ValueRange(shapeVec)).getResult();
  auto getGlobalSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, getGlobalSubAttr).getResult();
  
  // auto inSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), efp32Ty, inAttr, ValueRange(shapeVec)).getResult();
  // auto inSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), fp32Ptr, ptrOp, inSubAttr).getResult();
  auto inSubPtrAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), fp32Ptr, op.getInput()).getResult(0);

  auto scaleSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), efp32Ty, scaleAttr, ValueRange { dim }).getResult();
  auto scaleSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), fp32Ptr, ptrOp, scaleSubAttr).getResult();

  auto offsetSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, offsetAttr, ValueRange { dim }).getResult();
  auto offsetSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, offsetSubAttr).getResult();

  auto args = ValueRange({inSubPtrAttr, getGlobalSubPtrAttr, numElemAttr, scaleSubPtrAttr, offsetSubPtrAttr});
  rewriter.create<mlir::emitc::CallOp>(op.getLoc(), quantFn, args);

  auto r = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), op.getOutput().getType(), getGlobalSubPtrAttr);
  rewriter.replaceOp(op, r.getResults());

  // rewriter.replaceOp(op, getGlobal);

  return mlir::success();
}

}
}