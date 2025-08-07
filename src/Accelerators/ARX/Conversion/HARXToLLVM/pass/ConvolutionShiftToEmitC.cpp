
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


LogicalResult ConvolutionShiftToEmitCPattern::matchAndRewrite(harx::HARXConvolutionShiftOp op,PatternRewriter &rewriter) const {
  auto ui8Ty = rewriter.getIntegerType(8, false);
  auto i32Ty = rewriter.getIntegerType(32, true);
  // op.getLoc(), op.getResult().getType(),
    // op.getX().getDefiningOp()->getResult(0),
    // kernel_op,
    // kernel_scale_op,
    // bias_op,
    // do_bias, do_relu, pad_size, stride_size);

  auto module = op->getParentOfType<ModuleOp>();
  auto harxName = op.getOperation()->getAttrOfType<mlir::StringAttr>("harx_name");


  auto kernelOp = op.getKernel().getDefiningOp<harx::HARXConstantOp>();
  auto kernelVal = kernelOp.getValueAttr();
  auto kernelElemTy = kernelVal.getShapedType();
  auto kernelEmitTy = mlir::emitc::ArrayType::get(kernelElemTy.getShape(), kernelElemTy.getElementType());
  
  auto biasOp = op.getBias().getDefiningOp<harx::HARXConstantOp>();
  auto biasVal = biasOp.getValueAttr();
  auto biasElemTy = biasVal.getShapedType();
  auto biasEmitTy = mlir::emitc::ArrayType::get(biasElemTy.getShape(), biasElemTy.getElementType());

  auto retTy = op.getOutput().getType();
  auto outputTy = emitc::ArrayType::get(retTy.getShape(), ui8Ty);
  auto outputOffsetTy = emitc::ArrayType::get({1}, ui8Ty);
  
  auto shiftScale = op.getOutputShift();
  auto shiftScaleTy = emitc::ArrayType::get({shiftScale.getType().getNumElements()}, shiftScale.getType().getElementType());

  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      rewriter.getStringAttr(harxName.str() + "_kernel"),
      /* type */          kernelEmitTy, kernelVal, false, true, false);

    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      rewriter.getStringAttr(harxName.str() + "_bias"),
      /* type */          biasEmitTy, biasVal, false, true, false);


    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      rewriter.getStringAttr(harxName.str() + "_output"),
      /* type */          outputTy, Attribute(), false, true, false);

    // output offset
    auto outputOffset = op.getOutputOffset();
    SmallVector<APInt, 1> outputOffsetVec;
    outputOffsetVec.push_back(APInt(8, outputOffset, false));
    auto outputOffsetDenseAttr = DenseIntElementsAttr::get(RankedTensorType::get({1}, ui8Ty), outputOffsetVec);
    rewriter.create<mlir::emitc::GlobalOp>(op.getLoc(), rewriter.getStringAttr(harxName.str() + "_output_offset"), outputOffsetTy, outputOffsetDenseAttr, false, true, false);

    // shift
    SmallVector<APInt> shiftScaleVec;
    auto shiftValueAttr = shiftScale.getDefiningOp<harx::HARXConstantOp>().getValueAttr();
    shiftScaleVec.reserve(shiftScale.getType().getNumElements());
    for (auto item : shiftValueAttr.getValues<APInt>()) {
      shiftScaleVec.push_back(APInt(32, item.getSExtValue(), true));
    }
    auto shiftScaleDenseAttr = DenseIntElementsAttr::get(RankedTensorType::get(shiftScaleTy.getShape(), shiftScaleTy.getElementType()), shiftScaleVec);
    rewriter.create<mlir::emitc::GlobalOp>(op.getLoc(), rewriter.getStringAttr(harxName.str() + "_shift_scale"), shiftScaleTy, shiftScaleDenseAttr, false, true, false);
  }

  auto kernelSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_kernel"));
  auto kernelAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), kernelEmitTy, kernelSymRef).getResult();

  auto biasSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_bias"));
  auto biasAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), biasEmitTy, biasSymRef).getResult();

  auto outputSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_output"));
  auto outputAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), outputTy, outputSymRef).getResult();

  auto outputOffsetSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_output_offset"));
  auto outputOffsetAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), outputOffsetTy, outputOffsetSymRef).getResult();

  auto shiftScaleSymRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), rewriter.getStringAttr(harxName.str() + "_shift_scale"));
  auto shiftScaleAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), shiftScaleTy, shiftScaleSymRef).getResult();



  auto inputTy = op.getInput().getType();
  auto inputN = rewriter.getIntegerAttr(ui8Ty, inputTy.getDimSize(0));
  auto inputC = rewriter.getIntegerAttr(ui8Ty, inputTy.getDimSize(1));
  auto inputH = rewriter.getIntegerAttr(ui8Ty, inputTy.getDimSize(2));
  auto inputW = rewriter.getIntegerAttr(ui8Ty, inputTy.getDimSize(3));
  auto inputNAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, inputN);
  auto inputCAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, inputC);
  auto inputHAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, inputH);
  auto inputWAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, inputW);

  // unsigned char KN, unsigned char KH, unsigned char KW,
  auto kernelTy = op.getKernel().getType();
  auto kernelN = rewriter.getIntegerAttr(ui8Ty, kernelTy.getDimSize(0));
  auto kernelH = rewriter.getIntegerAttr(ui8Ty, kernelTy.getDimSize(2));
  auto kernelW = rewriter.getIntegerAttr(ui8Ty, kernelTy.getDimSize(3));
  auto kernelNAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, kernelN);
  auto kernelHAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, kernelH);
  auto kernelWAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, kernelW);
  
  // pad_size
  auto padSize = op.getPadSize();
  auto padSizeTy = ui8Ty;
  auto padSizeVal = rewriter.getIntegerAttr(padSizeTy, padSize);
  auto padSizeAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), padSizeTy, padSizeVal).getResult();

  // stride_size
  auto strideSize = op.getStrideSize();
  auto strideSizeTy = ui8Ty;
  auto strideSizeVal = rewriter.getIntegerAttr(strideSizeTy, strideSize);
  auto strideSizeAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), strideSizeTy, strideSizeVal).getResult();

  // do_relu
  auto doRelu = op.getDoRelu() == 1;
  auto doReluTy = rewriter.getI1Type();
  auto doReluVal = rewriter.getIntegerAttr(doReluTy, doRelu);
  auto doReluAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), doReluTy, doReluVal).getResult();

  // do_bais
  auto doBias = op.getDoBias() == 1;
  auto doBiasTy = rewriter.getI1Type();
  auto doBiasVal = rewriter.getIntegerAttr(doBiasTy, doBias);
  auto doBiasAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), doBiasTy, doBiasVal).getResult();
  
  // NCHW
  // out_w, out_h
  auto outHeightVal = rewriter.getIntegerAttr(ui8Ty, retTy.getDimSize(2));
  auto outHeightAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, outHeightVal).getResult();
  auto outWidthVal = rewriter.getIntegerAttr(ui8Ty, retTy.getDimSize(3));
  auto outWidthAttr = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), ui8Ty, outWidthVal).getResult();

  auto convFn = module.lookupSymbol<emitc::FuncOp>("convolution_i8_shift");
    
  auto shape = inputTy.getShape();
  SmallVector<Value> shapeVec;
  auto dim = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)).getResult();
  for (int i = 0; i < (int)shape.size(); ++i) {
    shapeVec.push_back(dim);
  }

  auto eui8Ty = mlir::emitc::LValueType::get(ui8Ty);
  auto ei32Ty = mlir::emitc::LValueType::get(i32Ty);
  auto ui8Ptr = emitc::PointerType::get(ui8Ty);
  auto i32Ptr = emitc::PointerType::get(i32Ty);

  auto ptrOp = rewriter.getStringAttr("&");

  // auto inputFnTy = emitc::ArrayType::get(inputTy.getShape(), inputTy.getElementType());
  // auto inputAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), inputFnTy, op.getInput()).getResult(0);
  
  // auto inputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, inputAttr, ValueRange(shapeVec)).getResult();
  // auto inputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, inputSubAttr).getResult();
  auto inputSubPtrAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ui8Ptr, op.getInput()).getResult(0);

  auto kernelSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, kernelAttr, ValueRange(shapeVec)).getResult();
  auto kernelSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, kernelSubAttr).getResult();

  auto biasSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), ei32Ty, biasAttr, ValueRange{ dim }).getResult();
  auto biasSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), i32Ptr, ptrOp, biasSubAttr).getResult();

  auto outputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, outputAttr, ValueRange(shapeVec)).getResult();
  auto outputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, outputSubAttr).getResult();

  auto outputOffsetSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, outputOffsetAttr, ValueRange { dim }).getResult();
  auto outputOffsetSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, outputOffsetSubAttr).getResult();

  auto shiftScaleSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), ei32Ty, shiftScaleAttr, ValueRange { dim }).getResult();
  auto shiftScaleSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), i32Ptr, ptrOp, shiftScaleSubAttr).getResult();

  rewriter.create<mlir::emitc::CallOp>(
                          op.getLoc(),
                          convFn,
                          ValueRange{
                            inputSubPtrAttr, kernelSubPtrAttr, biasSubPtrAttr, outputSubPtrAttr, outputOffsetSubPtrAttr,
                            inputNAttr, inputCAttr, inputHAttr, inputWAttr,
                            kernelNAttr, kernelHAttr, kernelWAttr,
                            padSizeAttr, strideSizeAttr,
                            doReluAttr, doBiasAttr,
                            shiftScaleSubPtrAttr,
                            outHeightAttr, outWidthAttr
                          });
  
  auto r = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), op.getOutput().getType(), outputSubPtrAttr);
  rewriter.replaceOp(op, r.getResults());


  return mlir::success();
}

}
}