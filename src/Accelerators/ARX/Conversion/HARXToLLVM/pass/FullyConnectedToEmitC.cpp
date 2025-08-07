
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
  
LogicalResult FullyConnectedToEmitCPattern::matchAndRewrite(harx::HARXMatMulOp op,PatternRewriter &rewriter) const {
  auto i1Ty = rewriter.getI1Type();
  auto ui8Ty = rewriter.getIntegerType(8, false);
  auto ui32Ty = rewriter.getIntegerType(32, false);
  auto i32Ty = rewriter.getIntegerType(32, true);

  auto ei1Ty = emitc::LValueType::get(i1Ty);
  auto eui8Ty = emitc::LValueType::get(ui8Ty);
  auto eui32Ty = emitc::LValueType::get(ui32Ty);
  auto ei32Ty = emitc::LValueType::get(i32Ty);
  
  auto ui8Ptr = emitc::PointerType::get(ui8Ty);
  auto i32Ptr = emitc::PointerType::get(i32Ty);
  auto ui32Ptr = emitc::PointerType::get(ui32Ty);
  auto ptrOp = rewriter.getStringAttr("&");
  auto dim = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)).getResult();

  // input Dimension
  auto fnInputTy = op.getInput().getType();
  auto inputMVal = fnInputTy.getShape()[0];
  auto inputKVal = fnInputTy.getShape()[1];
  auto inputMAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui32Ty, rewriter.getIntegerAttr(ui32Ty, inputMVal)).getResult();
  auto inputKAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui32Ty, rewriter.getIntegerAttr(ui32Ty, inputKVal)).getResult();

  // output Dimension
  auto fnKernelTy = op.getWeight().getType();
  auto kernelNVal = fnKernelTy.getShape()[0];
  auto kernelKVal = fnKernelTy.getShape()[1];
  auto kernelNAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui32Ty, rewriter.getIntegerAttr(ui32Ty, kernelNVal)).getResult();
  auto kernelKAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui32Ty, rewriter.getIntegerAttr(ui32Ty, kernelKVal)).getResult();

  // do bias
  auto doBias = op.getDoBias();
  auto doBiasAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, doBias)).getResult();

  // shift
  auto shiftScale = op.getShift();
  auto shiftScaleTy = emitc::ArrayType::get({shiftScale.getType().getNumElements()}, shiftScale.getType().getElementType());
  SmallVector<APInt> shiftScaleVec;
  auto shiftValueAttr = shiftScale.getDefiningOp<harx::HARXConstantOp>().getValueAttr();
  shiftScaleVec.reserve(shiftScale.getType().getNumElements());
  for (auto item : shiftValueAttr.getValues<APInt>()) {
    shiftScaleVec.push_back(APInt(32, item.getSExtValue(), true));
  }
  auto shiftScaleDenseAttr = DenseIntElementsAttr::get(shiftScaleTy, shiftScaleVec);
  auto shiftScaleAttr = rewriter.create<mlir::emitc::VariableOp>(op.getLoc(), shiftScaleTy, shiftScaleDenseAttr).getResult();
  auto shiftScaleSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), ei32Ty, shiftScaleAttr, ValueRange { dim }).getResult();
  auto shiftScaleSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), i32Ptr, ptrOp, shiftScaleSubAttr).getResult();
  
  auto module = op->getParentOfType<ModuleOp>();
  auto harxName = op.getOperation()->getAttrOfType<mlir::StringAttr>("harx_name");

  auto kernelOp = op.getWeight().getDefiningOp<harx::HARXConstantOp>();
  auto kernelEmitTy = mlir::emitc::ArrayType::get(fnKernelTy.getShape(), fnKernelTy.getElementType());
  auto kernelVal = kernelOp.getValueAttr();

  auto biasOp = op.getBias().getDefiningOp<harx::HARXConstantOp>();
  auto biasTy = op.getBias().getType();
  auto biasEmitTy = mlir::emitc::ArrayType::get(biasTy.getShape(), biasTy.getElementType());
  auto biasVal = biasOp.getValueAttr();

  auto outputTy = op.getOutput().getType();
  auto outputEmitTy = mlir::emitc::ArrayType::get(outputTy.getShape(), outputTy.getElementType());

  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      rewriter.getStringAttr(harxName.str() + "_kernel"),
      /* type */          kernelEmitTy, kernelVal, false, true, true);

    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      rewriter.getStringAttr(harxName.str() + "_bias"),
      /* type */          biasEmitTy, biasVal, false, true, true);

    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      rewriter.getStringAttr(harxName.str() + "_output"),
      /* type */          outputEmitTy, Attribute(), false, true, false);
  }

  auto kernelSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_kernel"));
  auto kernelAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), kernelEmitTy, kernelSymRef).getResult();
  auto kernelSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, kernelAttr, ValueRange { dim, dim }).getResult();
  auto kernelSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, kernelSubAttr).getResult();

  auto biasSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_bias"));
  auto biasAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), biasEmitTy, biasSymRef).getResult();
  auto biasSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), ei32Ty, biasAttr, ValueRange { dim, dim }).getResult();
  auto biasSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), i32Ptr, ptrOp, biasSubAttr).getResult();

  auto outputSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_output"));
  auto outputAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), outputEmitTy, outputSymRef).getResult();
  auto outputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, outputAttr, ValueRange { dim, dim }).getResult();
  auto outputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, outputSubAttr).getResult();


  auto outputOffset = op.getOutputOffset();
  auto outputOffsetTy = emitc::ArrayType::get({1}, ui8Ty);
  SmallVector<APInt, 1> outputOffsetVec;
  outputOffsetVec.push_back(APInt(8, outputOffset, false));
  auto outputOffsetDenseAttr = DenseIntElementsAttr::get(outputOffsetTy, outputOffsetVec);
  auto outputOffsetAttr = rewriter.create<mlir::emitc::VariableOp>(op.getLoc(), outputOffsetTy, outputOffsetDenseAttr).getResult();
  auto outputOffsetSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, outputOffsetAttr, ValueRange { dim }).getResult();
  auto outputOffsetSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, outputOffsetSubAttr).getResult();

  // unsigned char* input, unsigned char *kernel, int *bias, unsigned char *output, unsigned char* outputOffset
  auto inputSubPtrAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ui8Ptr, op.getInput()).getResult(0);

  auto fcFn = module.lookupSymbol<emitc::FuncOp>("fullyconnected_i8_shift");


  ValueRange args { 
    inputSubPtrAttr, kernelSubPtrAttr, biasSubPtrAttr, outputSubPtrAttr, outputOffsetSubPtrAttr,
    inputMAttr, inputKAttr, 
    kernelNAttr, kernelKAttr,
    doBiasAttr, shiftScaleSubPtrAttr,
    inputMAttr, kernelKAttr
  };
    rewriter.create<mlir::emitc::CallOp>(
                          op.getLoc(),
                          fcFn,
                          args
  );
  auto r = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), op.getOutput().getType(), outputSubPtrAttr);
  rewriter.replaceOp(op, r.getResults());

  return mlir::success();
}


}
}