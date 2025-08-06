
#include "mlir/Conversion/LLVMCommon/Pattern.h"

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


// LogicalResult ReturnToEmitCPattern::matchAndRewrite(func::ReturnOp op,
//                                                     PatternRewriter &rewriter) const {
//   llvm::dbgs() << "ReturnToEmitC::matchAndRewrite\n";
//   llvm::dbgs() << "ReturnToEmitC::matchAndRewrite: " << op << "\n";
//   // 1) 기본 가드: 단일 피연산자 리턴만 처리
//   if (op.getNumOperands() != 1)
//     return failure();

//   auto funcOp = op->getParentOfType<func::FuncOp>();
//   llvm::dbgs() << "ReturnToEmitC::matchAndRewrite: funcOp: " << funcOp << "\n";
//   if (!funcOp)
//     return failure();

//   // 2) 시그니처가 이미 (arg0, out) -> () 인지 확인
//   auto fty = funcOp.getFunctionType();
//   llvm::dbgs() << "ReturnToEmitC::matchAndRewrite: funcTy: " << fty << "\n";
//   // if (fty.getNumResults() != 0 || funcOp.getNumArguments() < 2)
//     // return failure();

//   Value src = op.getOperand(0);      // memcpy source: 리턴하려던 값
//   Value dst = funcOp.getArgument(1); // memcpy dest: 두 번째 인자(out)
//   Location loc = op.getLoc();

//   // 3) 크기 상수 계산: 정적 RankedTensorType만 지원 (동적이면 별도 설계 필요)
//   auto tensTy = dyn_cast<RankedTensorType>(src.getType()); // ⚠ cast → dyn_cast
//   if (!tensTy || !tensTy.hasStaticShape())
//     return failure();

//   int64_t numElems = tensTy.getNumElements();
//   unsigned bitWidth = tensTy.getElementTypeBitWidth();
//   if (bitWidth == 0 || (bitWidth % 8) != 0)
//     return failure();

//   int64_t nBytes = numElems * static_cast<int64_t>(bitWidth / 8);

//   // 4) memcpy를 리턴 앞에 생성
//   rewriter.setInsertionPoint(op);

//   // emitc.constant i32 로 size 생성
//   auto sizeConst = rewriter.create<emitc::ConstantOp>(
//       loc, rewriter.getI32Type(),
//       rewriter.getI32IntegerAttr(static_cast<int32_t>(nBytes)));

//   // emitc.call_opaque "memcpy"(dst, src, size)
//   auto callee = rewriter.getStringAttr("memcpy");
//   auto empty  = rewriter.getArrayAttr({});
//   rewriter.create<emitc::CallOpaqueOp>(
//       loc,
//       /*resultTypes=*/TypeRange{},
//       /*callee=*/callee,
//       /*operands=*/ValueRange{dst, src, sizeConst});

//   // 5) 기존 func.return %v  →  빈 func.return
//   rewriter.replaceOpWithNewOp<func::ReturnOp>(op, ValueRange{});
//   return success();
// }


LogicalResult FunctionToEmitCPattern::matchAndRewrite(func::FuncOp funcOp,
                              PatternRewriter &rewriter) const {
  if (funcOp.getName() != "main_graph") { 
    return failure();
  }

  auto ctx = rewriter.getContext();
  
  //--- 1) 새 함수 시그니처 만들기 ----------------------------
  FunctionType oldTy = funcOp.getFunctionType();

  SmallVector<Type> inputs(oldTy.getInputs().begin(),
                            oldTy.getInputs().end());
  inputs.push_back(oldTy.getResult(0));                 // %arg1 추가
  // SmallVector<Type> results;             // 반환은 void

  auto newTy   = FunctionType::get(ctx, inputs, {});
  funcOp.setType(newTy);

  //--- 2) 엔트리 블록에 arg1 추가 ----------------------------
  Block &entry = funcOp.getBody().front();
  entry.addArgument(oldTy.getResult(0), funcOp.getLoc());   // %arg1 : tensor<…>

  //--- 3) return 변환 ---------------------------------------
  // 가정: 함수 안에 return 1개, 하나의 tensor operand만 반환
  funcOp.walk([&](func::ReturnOp ret) {
    Value retVal = ret.getOperand(0);    // 옛날 결과
    Value outBuf = entry.getArgument(inputs.size() - 1); // %arg1

    OpBuilder b(ret);
    retVal.replaceAllUsesWith(outBuf);
    
    b.create<func::ReturnOp>(ret.getLoc());
    ret.erase();
  });

  return success();
}



// LogicalResult ReturnToEmitCPattern::matchAndRewrite(func::ReturnOp op, mlir::PatternRewriter &rewriter) const { 
//   llvm::dbgs() << "01 ReturnToEmitC::matchAndRewrite: " << "\n";

//   auto module = op->getParentOfType<ModuleOp>();
//   auto funcOp = op->getParentOfType<func::FuncOp>();
//   if (!module || !funcOp) return failure();
//   llvm::dbgs() << "03 ReturnToEmitC::matchAndRewrite: " << "\n";
//   Value retData = op.getOperand(0);
//   auto retTensorTy = mlir::cast<RankedTensorType>(retData.getType());
//   if (!retTensorTy) return failure();

//   auto funcTy = funcOp.getFunctionType();
//   llvm::dbgs() << "40 ReturnToEmitC::matchAndRewrite: " << funcTy << "\n";

//   llvm::dbgs() << "41 ReturnToEmitC::matchAndRewrite: " << funcTy << "\n";
//   auto inTensorTy  = mlir::cast<RankedTensorType>(funcTy.getInput(0));
//   auto resTensorTy = mlir::cast<RankedTensorType>(funcTy.getResult(0));
//   if (!inTensorTy || !resTensorTy) return failure();
//   llvm::dbgs() << "42 ReturnToEmitC::matchAndRewrite: " << funcTy << "\n";

//   auto emitInTy  = mlir::emitc::ArrayType::get(inTensorTy.getShape(),
//                                                inTensorTy.getElementType());
//   auto emitResTy = mlir::emitc::ArrayType::get(resTensorTy.getShape(),
//                                                resTensorTy.getElementType());

//   llvm::dbgs() << "04 ReturnToEmitC::matchAndRewrite: " << emitInTy << "\n\n" << emitResTy << "\n";
//   auto newFuncTy = rewriter.getFunctionType(
//       ArrayRef<Type>{emitInTy, emitResTy}, /*results=*/TypeRange{});


//   auto emitcFuncOp = rewriter.create<emitc::FuncOp>(module.getLoc(), funcOp.getName(), newFuncTy);
//   rewriter.eraseOp(op);

//   rewriter.inlineRegionBefore(funcOp.getBody(), emitcFuncOp.getBody(), emitcFuncOp.getBody().begin());
//   rewriter.eraseOp(funcOp);

//   llvm::dbgs() << "06 ReturnToEmitC::matchAndRewrite: " << newFuncTy << "\n";

//   Block &entry = funcOp.getBody().front();
//   Operation *term = entry.getTerminator();

//   rewriter.setInsertionPointToEnd(&entry);
//   rewriter.create<emitc::ReturnOp>(funcOp.getLoc());


//   return success();
// }



  // int64_t numElems  = resTensorTy.getNumElements();
  // int64_t byteWidth = resTensorTy.getElementTypeBitWidth() / 8;
  // auto sizeConstant = rewriter.create<emitc::ConstantOp>(op->getLoc(), 
  //                                                   rewriter.getI32Type(), 
  //                                                   rewriter.getI32IntegerAttr(static_cast<int32_t>(numElems * byteWidth)));

  // Value dstPtr = emitFunc.getArgument(1);
  // rewriter.create<emitc::CallOpaqueOp>(
  //     op->getLoc(), /*result types=*/TypeRange{}, 
  //     rewriter.getStringAttr("memcpy"),
  //     ValueRange{dstPtr, retData, sizeConstant});

  // llvm::dbgs() << "08 ReturnToEmitC::matchAndRewrite: " << "\n";

//   static memref::GlobalOp getOrInsertElemAddGlobalScrach(ModuleOp module, 
//                                                  PatternRewriter &rewriter,
//                                                  MemRefType bufTy,
//                                                 mlir::ElementsAttr default_value) {
//   constexpr llvm::StringLiteral kName("elemadd_scratch_val");

//   if (auto g = module.lookupSymbol<memref::GlobalOp>(kName))
//     return g;

//   OpBuilder::InsertionGuard g(rewriter);
//   rewriter.setInsertionPointToStart(module.getBody());

//   // mutable 전역: constant = false
//   auto global = rewriter.create<memref::GlobalOp>(
//       module.getLoc(),                    // 위치
//       /*sym_name=*/kName,     // 심볼 이름
//       /*sym_visibility=*/rewriter.getStringAttr("private"),
//       /*type=*/bufTy,
//       /*initial_value=*/default_value,   // 초기값 없음
//       /*constant=*/true,
//       /*alignment=*/rewriter.getI64IntegerAttr(8));    // 정렬(옵션)

//   return global;
// }

// mlir::LogicalResult ConstantToGlobal::matchAndRewrite(mlir::Operation *ops, mlir::ArrayRef<Value> operands, mlir::ConversionPatternRewriter &rewriter) const { 
//   return mlir::success();
// }

} // namespace harx
} // namespace onnx_mlir
