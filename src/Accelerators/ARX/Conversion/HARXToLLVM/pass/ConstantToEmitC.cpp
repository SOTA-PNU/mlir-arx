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


LogicalResult ConstantToEmitCPattern::matchAndRewrite(Operation *ops, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const { 
  auto module = ops->getParentOfType<mlir::ModuleOp>();
  // auto ctx = ops->getContext();
  auto op = mlir::cast<harx::HARXConstantOp>(ops);

  auto dataValue = op.getValueAttr();
  auto elemDataTy = dataValue.getShapedType();
  auto dataTy = mlir::emitc::ArrayType::get(elemDataTy.getShape(), elemDataTy.getElementType());
  auto harxName = op.getOperation()->getAttrOfType<mlir::StringAttr>("harx_name");

  llvm::dbgs() << dataValue << "\n";

  if (!module.lookupSymbol<mlir::emitc::GlobalOp>(harxName)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::emitc::GlobalOp>(
      module.getLoc(),
      /* sym_name */      harxName,
      /* type */          dataTy,
      /* initializer */   dataValue,
      /* extern? */       false,
      /* static? */       true,
      /* const? */        true);
  }

  auto symRef = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), harxName);
  auto getGlobal = rewriter.create<mlir::emitc::GetGlobalOp>(
      op.getLoc(),
      dataTy,
      symRef);

  rewriter.replaceOp(op, getGlobal.getResult());

  return success();
}

}
}