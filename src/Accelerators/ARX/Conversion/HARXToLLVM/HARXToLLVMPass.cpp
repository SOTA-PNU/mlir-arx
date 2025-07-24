/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToHARX.cpp - ONNX dialect to HARX lowering -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and HARX operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARX.hpp"
// #include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARXCommon.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
// #include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/HARXTypeConverter.hpp"
#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVM.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"      // RankedTensorType, MemRefType, IntegerType
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <optional>                    // std::optional / std::nullopt  (C++17)



using namespace mlir;


namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX to HARX Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXONNXToHARX.inc"

struct HARXToLLVMLoweringPass
    : public PassWrapper<HARXToLLVMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HARXToLLVMLoweringPass)

  StringRef getArgument() const override { return "convert-harx-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower HARX ops to LLVM ops.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  HARXToLLVMLoweringPass() = default;
  HARXToLLVMLoweringPass(const HARXToLLVMLoweringPass &pass)
      : PassWrapper<HARXToLLVMLoweringPass, OperationPass<ModuleOp>>() {}
  void runOnOperation() final;
};
} // namespace

void HARXToLLVMLoweringPass::runOnOperation() {
  llvm::outs() << "HARXToLLVMLoweringPass::runOnOperation\n";
  ModuleOp module = getOperation();

  ConversionTarget target(getContext());

  target.addLegalDialect<func::FuncDialect, memref::MemRefDialect, LLVM::LLVMDialect, arith::ArithDialect, bufferization::BufferizationDialect>();
  target.addIllegalOp<harx::HARXConstantOp>();
  target.addLegalOp<func::FuncOp, func::ReturnOp>();


  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();
  mlir::LLVMTypeConverter typeConverter(&getContext());
    // Materialize MemRef → Tensor with ExtractSliceOp

    typeConverter.addConversion([](RankedTensorType type) {
      llvm::outs() << "Converting RankedTensorType to MemRefType\n";
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    
    typeConverter.addConversion([](MemRefType type) {
      llvm::outs() << "Converting MemRefType to RankedTensorType\n";
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
  


  RewritePatternSet patterns(&getContext());
  onnx_mlir::larx::populateHARXToLLVMConversionPattern(patterns, typeConverter, &getContext());

    // module.walk([&](func::FuncOp fn) {
    //   if (fn.getName() != "main_graph")
    //     return;
    //   auto ctx = &getContext();
    //   auto tTy = RankedTensorType::get({1, 10, 10, 8},
    //                                  IntegerType::get(ctx, 8, IntegerType::Unsigned));

    //   //--- 1) 새 함수 시그니처 만들기 ----------------------------
    //   FunctionType oldTy = fn.getFunctionType();

    //   SmallVector<Type> inputs(oldTy.getInputs().begin(),
    //                            oldTy.getInputs().end());
    //   inputs.push_back(tTy);                 // %arg1 추가
    //   SmallVector<Type> results;             // 반환은 void

    //   auto newTy   = FunctionType::get(ctx, inputs, results);
    //   fn.setType(newTy);

    //   //--- 2) 엔트리 블록에 arg1 추가 ----------------------------
    //   Block &entry = fn.getBody().front();
    //   entry.addArgument(tTy, fn.getLoc());   // %arg1 : tensor<…>

    //   //--- 3) return 변환 ---------------------------------------
    //   // 가정: 함수 안에 return 1개, 하나의 tensor operand만 반환
    //   fn.walk([&](func::ReturnOp ret) {
    //     Value retVal = ret.getOperand(0);    // 옛날 결과
    //     Value outBuf = entry.getArgument(inputs.size() - 1); // %arg1

    //     OpBuilder b(ret);
    //     // tensor.store 값 -> %arg1
    //     retVal.replaceAllUsesWith(outBuf);
        
    //     b.create<func::ReturnOp>(ret.getLoc());
    //     ret.erase();
    //   });
    // });

module.walk([&](func::FuncOp fn) {
  if (fn.getName() != "main_graph") return;

  // 1) Original signature
  FunctionType oldType = fn.getFunctionType();

  // 2) Build new input types: change any strided memref to identity layout
  SmallVector<Type, 4> newInputs;
  for (Type t : oldType.getInputs()) {
    if (auto mt = t.dyn_cast<MemRefType>()) {
      auto ctx = fn.getContext();
      // identity layout map of same rank
      AffineMap idMap = AffineMap::getMultiDimIdentityMap(mt.getRank(), ctx);
      // new static memref type
      newInputs.push_back(
        MemRefType::get(
          mt.getShape(),
          mt.getElementType(),
          idMap,
          mt.getMemorySpace()));
    } else {
      newInputs.push_back(t);
    }
  }

  // 3) Results remain unchanged
  SmallVector<Type, 4> newResults(oldType.getResults().begin(),
                                   oldType.getResults().end());

  // 4) Create and apply the new FunctionType
  OpBuilder builder(fn);
  auto newFuncType = builder.getFunctionType(newInputs, newResults);
  fn.setType(newFuncType);  // updates external signature :contentReference[oaicite:2]{index=2}

  // 5) Update the entry block arguments too
  Block &entry = fn.getBody().front();
  assert(entry.getNumArguments() == newInputs.size() &&
         "argument count must match new function type");
  for (unsigned i = 0, e = newInputs.size(); i < e; ++i) {
    entry.getArgument(i).setType(newInputs[i]);  // patch block arg :contentReference[oaicite:3]{index=3}
  }
});


  module.walk([&](Operation *op) {
    llvm::outs() << "Processing operation: " << op->getName() << "\n";
    llvm::outs() << "Found operation: " << op->getDialect() << "\n";
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createHARXToLLVMPass() {
  return std::make_unique<HARXToLLVMLoweringPass>();
}

} // namespace onnx_mlir
