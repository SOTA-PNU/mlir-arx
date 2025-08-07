#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/IRMapping.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVM.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"


#include "src/Accelerators/ARX/Conversion/HARXToLLVM/pass/HarxToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace onnx_mlir {
namespace larx {


LogicalResult FunctionToEmitCPattern::matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const {
  if (op.getName() != "main_graph") {
    return failure();
  }
  auto funcTy = op.getFunctionType();
  if (funcTy.getNumInputs() != 1 || funcTy.getNumResults() != 1) {
    return failure();
  }

  auto inpTy = mlir::cast<RankedTensorType>(op.getArgumentTypes()[0]);
  auto retTy = mlir::cast<RankedTensorType>(op.getResultTypes()[0]);

  auto retElemTy = retTy.getElementType();
  auto bitSize = retElemTy.getIntOrFloatBitWidth() / 8; // byte 단위
  auto elemtSize = retTy.getNumElements();

  SmallVector<Type> newInputs;
  auto f32Ty = rewriter.getF32Type();
  auto ui8Ty = rewriter.getIntegerType(8, false);
  
  auto eui8Ty = mlir::emitc::LValueType::get(ui8Ty);
  auto efp32Ty = mlir::emitc::LValueType::get(f32Ty);
  
  auto f32PtrTy = emitc::PointerType::get(f32Ty);
  auto uiPtr8Ty = emitc::PointerType::get(ui8Ty);
  newInputs.push_back(f32PtrTy);
  newInputs.push_back(uiPtr8Ty);

  auto newFuncType = rewriter.getFunctionType(newInputs, TypeRange {});
  auto emitcFuncOp = rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), newFuncType);
  emitcFuncOp.getBody().takeBody(op.getBody());
  
  emitcFuncOp.getBody().front();

  // emitcFuncOp.front().insertArgument(0, ef32Ty, emitcFuncOp.getLoc());
  emitcFuncOp.front().insertArgument(1, uiPtr8Ty, emitcFuncOp.getLoc());

  emitcFuncOp.getArgument(0).setType(f32PtrTy);
  emitcFuncOp.getArgument(1).setType(uiPtr8Ty);

  Block* entry = &emitcFuncOp.getBody().getBlocks().front();
  
  auto inpRankTy = RankedTensorType::get(inpTy.getShape(), inpTy.getElementType());
  auto retRankTy = RankedTensorType::get(retTy.getShape(), retTy.getElementType());
  auto tmpRetTy = emitc::ArrayType::get(retTy.getShape(), retTy.getElementType());
  
  rewriter.setInsertionPointToStart(entry);
  auto inpAttr = rewriter.create<UnrealizedConversionCastOp>(emitcFuncOp.getBody().getLoc(), inpRankTy, emitcFuncOp.getArgument(0)).getResult(0);

  auto shape = tmpRetTy.getShape();
  SmallVector<Value> shapeVec;
  auto dim = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)).getResult();
  for (int i = 0; i < (int)shape.size(); ++i) {
    shapeVec.push_back(dim);
  }
  auto ptrOp = rewriter.getStringAttr("&");

  emitcFuncOp.getBody().front().getArgument(0).replaceAllUsesExcept(inpAttr, inpAttr.getDefiningOp());

  llvm::dbgs() << "FunctionToEmitCPattern::matchAndRewrite: emitcFuncOp: " << emitcFuncOp << "\n";

  rewriter.modifyOpInPlace(emitcFuncOp, [&]() {
    emitcFuncOp.walk([&](func::ReturnOp returnOp) {
        rewriter.setInsertionPoint(returnOp); 
        auto retVal = returnOp.getOperand(0).getDefiningOp()->getResult(0); //tensor
        
        llvm::dbgs() << "ReturnToEmitC::matchAndRewrite: retVal: " << retVal << "\n";
  
        auto retSize = rewriter.create<emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(elemtSize * bitSize)).getResult();
  
        auto ui8Ty = rewriter.getIntegerType(8, false);
        auto ui8Ptr = emitc::PointerType::get(ui8Ty);
  
        auto resultAttr = rewriter.create<UnrealizedConversionCastOp>(returnOp.getLoc(), tmpRetTy, retVal).getResult(0);
        auto resultSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), efp32Ty, resultAttr, ValueRange(shapeVec)).getResult();
        auto resultSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), f32PtrTy, ptrOp, resultSubAttr).getResult();
        auto resultCastSubPtrAttr = rewriter.create<mlir::emitc::CastOp>(op.getLoc(), ui8Ptr, resultSubPtrAttr).getResult();

        auto outputVal = emitcFuncOp.getArgument(1);
        auto outputAttr = rewriter.create<UnrealizedConversionCastOp>(returnOp.getLoc(), ui8Ptr, outputVal).getResult(0);

        rewriter.create<emitc::CallOpaqueOp>(returnOp.getLoc(), ui8Ptr, "memcpy", ValueRange{ resultCastSubPtrAttr, outputAttr, retSize });
        auto emitRetOp = rewriter.create<emitc::ReturnOp>(returnOp.getLoc(), mlir::Value());
        rewriter.replaceOp(returnOp, emitRetOp);
    });
  });

  rewriter.eraseOp(op);

  return success();
}


} // namespace harx
} // namespace onnx_mlir
