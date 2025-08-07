
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

#include <sstream>

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace larx {
  
LogicalResult ReShapeToEmitCPattern::matchAndRewrite(harx::HARXTransposeOp op,PatternRewriter &rewriter) const {
  auto ui8Ty = rewriter.getIntegerType(8, false);
  // auto eui8Ty = mlir::emitc::LValueType::get(ui8Ty);
  auto ui8Ptr = emitc::PointerType::get(ui8Ty);

  auto inputOp = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ui8Ptr, op.getInput());
  auto outputOp = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), op.getOutput().getType(), inputOp.getResults());
  rewriter.replaceOp(op, outputOp.getResults());

  return mlir::success();
}


}
}