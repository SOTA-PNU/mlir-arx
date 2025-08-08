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


LogicalResult MaxPoolToEmitCPattern::matchAndRewrite(harx::HARXMaxPoolOp op, PatternRewriter &rewriter) const {
  auto ui8Ty = rewriter.getIntegerType(8, false);
  
  auto harxName = op.getOperation()->getAttrOfType<mlir::StringAttr>("harx_name");
  auto retTy = op.getResult().getType();
  auto outputTy = emitc::ArrayType::get(retTy.getShape(), ui8Ty);
  auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!module.lookupSymbol<mlir::emitc::GlobalOp>(harxName)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(module.getLoc(),
      /* sym_name */      harxName,
      /* type */          outputTy, Attribute(), false, true, false);
  }

  auto symRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), harxName);
  auto outputAttr = rewriter.create<mlir::emitc::GetGlobalOp>(op.getLoc(), outputTy, symRef).getResult();


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
  auto kernelH = rewriter.getIntegerAttr(ui8Ty, op.getKernelHeight());
  auto kernelW = rewriter.getIntegerAttr(ui8Ty, op.getKernelWidth());
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

  auto shape = inputTy.getShape();
  SmallVector<Value> shapeVec;
  auto dim = rewriter.create<mlir::emitc::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)).getResult();
  for (int i = 0; i < (int)shape.size(); ++i) {
    shapeVec.push_back(dim);
  }
  
  auto eui8Ty = mlir::emitc::LValueType::get(ui8Ty);
  auto ui8Ptr = emitc::PointerType::get(ui8Ty);
  auto ptrOp = rewriter.getStringAttr("&");

  // auto inputFnTy = emitc::ArrayType::get(inputTy.getShape(), ui8Ty);
  // auto inputAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), inputFnTy, op.getInput()).getResult(0);
  // auto inputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, inputAttr, ValueRange(shapeVec)).getResult();
  // auto inputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, inputSubAttr).getResult();
  // auto inputSubPtrAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ui8Ptr, ptrOp, inputSubAttr).getResult(0);
  auto inputSubPtrAttr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ui8Ptr, op.getInput()).getResult(0);

  auto outputSubAttr = rewriter.create<mlir::emitc::SubscriptOp>(op.getLoc(), eui8Ty, outputAttr, ValueRange(shapeVec)).getResult();
  auto outputSubPtrAttr = rewriter.create<mlir::emitc::ApplyOp>(op.getLoc(), ui8Ptr, ptrOp, outputSubAttr).getResult();


  auto maxpoolFn = module.lookupSymbol<emitc::FuncOp>("maxpool_i8");
  auto args = ValueRange({inputSubPtrAttr, outputSubPtrAttr,
    inputNAttr, inputCAttr, inputHAttr, inputWAttr,
    kernelHAttr, kernelWAttr, padSizeAttr, strideSizeAttr
  });
  rewriter.create<mlir::emitc::CallOp>(op.getLoc(), maxpoolFn, args);
  auto r = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), op.getOutput().getType(), outputSubPtrAttr);
  
  rewriter.replaceOp(op, r.getResults());

    // maxpool_i8(output2, output3, 1, 8, 28, 28, 2, 2, 0, 2);


  return mlir::success();
}


}
}