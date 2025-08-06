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

#include "mlir/Dialect/EmitC/IR/EmitC.h"

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

  StringRef getArgument() const override { return "convert-harx-to-emitc"; }

  StringRef getDescription() const override {
    return "Lower HARX ops to EmitC ops.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  HARXToLLVMLoweringPass() = default;
  HARXToLLVMLoweringPass(const HARXToLLVMLoweringPass &pass)
      : PassWrapper<HARXToLLVMLoweringPass, OperationPass<ModuleOp>>() {}
  void runOnOperation() final;
};
} // namespace


void CreateFunctions(ModuleOp module, OpBuilder &builder, FunctionType fnType, std::string fnName) {
  auto c = builder.create<emitc::FuncOp>(module.getLoc(), fnName, fnType);
  c.setPrivate();
  c->setAttr("specifiers", builder.getStrArrayAttr({"extern"}));
}

void HARXToLLVMLoweringPass::runOnOperation() {
  llvm::outs() << "HARXToLLVMLoweringPass::runOnOperation\n";
  ModuleOp module = getOperation();
  ConversionTarget target(getContext());

  target.addLegalDialect<mlir::emitc::EmitCDialect>();
  // target.addIllegalDialect<harx::HARXDialect>();
  target.addIllegalOp<harx::HARXQuantizationOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  
  // target.addIllegalOp<func::FuncOp>();
  // target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
  //   // The return op is legal if it has no operands.
  //   return op.getNumOperands() == 0;
  // });
  // target.addIllegalOp<harx::HARXConstantOp>();

  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();
  mlir::LLVMTypeConverter typeConverter(&getContext());

  typeConverter.addConversion([&](RankedTensorType t) -> emitc::ArrayType {
    return emitc::ArrayType::get(t.getShape(), t.getElementType());
  });

  typeConverter.addConversion([&](emitc::ArrayType t) -> RankedTensorType {
    return RankedTensorType::get(t.getShape(), t.getElementType());
  });

  // 1b) Leave all other types alone
  typeConverter.addConversion([](Type t) -> Type { return t; });

  RewritePatternSet patterns(&getContext());
  onnx_mlir::larx::populateHARXToLLVMConversionPattern(patterns, typeConverter, &getContext());
  
  MLIRContext &ctx = getContext();
  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  builder.create<emitc::IncludeOp>(module.getLoc(), "stdint.h", true);
  builder.create<emitc::IncludeOp>(module.getLoc(), "stdlib.h", true);
  builder.create<emitc::IncludeOp>(module.getLoc(), "string.h", true);
  builder.create<emitc::IncludeOp>(module.getLoc(), "stdbool.h", true);
  
  auto ui8Ty = builder.getIntegerType(8, false);
  auto ui8pTy = mlir::emitc::PointerType::get(ui8Ty);
  auto i32Ty = builder.getIntegerType(32, true);
  auto ui32Ty = builder.getIntegerType(32, false);
  auto i32pTy = mlir::emitc::PointerType::get(i32Ty);
  auto i1Ty = builder.getI1Type();
  auto f32Ty = builder.getF32Type();
  auto f32pTy = mlir::emitc::PointerType::get(f32Ty);

  
  CreateFunctions(module, builder,  builder.getFunctionType(
      /*inputs*/  {
        /*input*/ui8pTy, /*kernel*/ui8pTy, /*bias*/i32pTy, /*output*/ui8pTy, /*outputOffset*/ui8pTy, 
        /*N*/ui8Ty, /*H*/ui8Ty, /*W*/ui8Ty, /*C*/ui8Ty,
        /*KN*/ui8Ty, /*KH*/ui8Ty, /*KW*/ui8Ty, 
        /*pad_size*/ui8Ty, /*stride_size*/ui8Ty, 
        /*doRelu*/i1Ty, /*doBias*/i1Ty,
        /*shift*/i32pTy, /*out_h*/ui8Ty, /*out_w*/ui8Ty
      },
      /*results*/ {i32Ty}), "convolution_i8_shift");

  CreateFunctions(module, builder, builder.getFunctionType(
    /*inputs*/  {
      ui8pTy, ui8pTy, i32pTy, ui8pTy, 
      ui32Ty, ui32Ty, ui32Ty, ui32Ty,
      i1Ty, i32pTy, ui32Ty,ui32Ty
  },
  /*results*/ {i32Ty}), "fullyconnected_i8_shift");

  CreateFunctions(module, builder, builder.getFunctionType(
    /*inputs*/  {
      ui8pTy, ui8pTy, 
      ui8Ty, ui8Ty, ui8Ty, ui8Ty, 
      ui8Ty, ui8Ty,
      ui8Ty, ui8Ty,
  },
  /*results*/ {i32Ty}), "maxpool_i8");
  CreateFunctions(module, builder, builder.getFunctionType(
    /*inputs*/  {
      f32pTy, ui8pTy,
      ui32Ty, f32pTy, ui8pTy
  },
  /*results*/ {i32Ty}), "quantize");

  CreateFunctions(module, builder, builder.getFunctionType(
    /*inputs*/  {
      ui8pTy, f32pTy, ui32Ty,
      f32pTy, ui8pTy, i32Ty, i1Ty
  },
  /*results*/ {i32Ty}), "dequantize");
  CreateFunctions(module, builder, builder.getFunctionType(
    /*inputs*/  {
      ui8pTy, ui8pTy,
      ui32Ty, ui32Ty, ui32Ty, ui32Ty,
      ui32Ty, ui32Ty, ui32Ty, ui32Ty,
      ui8Ty, ui8Ty, ui8Ty, ui8Ty
  },
  /*results*/ {i32Ty}), "transpose_i8");

    // module.walk([&](func::FuncOp fn) {
    //   if (fn.getName() != "main_graph")
    //     return;
    //   auto ctx = &getContext();
      
    //   //--- 1) 새 함수 시그니처 만들기 ----------------------------
    //   FunctionType oldTy = fn.getFunctionType();

    //   SmallVector<Type> inputs(oldTy.getInputs().begin(),
    //                            oldTy.getInputs().end());
    //   inputs.push_back(oldTy.getResult(0));                 // %arg1 추가
    //   // SmallVector<Type> results;             // 반환은 void

    //   auto newTy   = FunctionType::get(ctx, inputs, {});
    //   fn.setType(newTy);

    //   //--- 2) 엔트리 블록에 arg1 추가 ----------------------------
    //   Block &entry = fn.getBody().front();
    //   entry.addArgument(oldTy.getResult(0), fn.getLoc());   // %arg1 : tensor<…>

    //   //--- 3) return 변환 ---------------------------------------
    //   // 가정: 함수 안에 return 1개, 하나의 tensor operand만 반환
    //   fn.walk([&](func::ReturnOp ret) {
    //     Value retVal = ret.getOperand(0);    // 옛날 결과
    //     Value outBuf = entry.getArgument(inputs.size() - 1); // %arg1

    //     OpBuilder b(ret);
    //     retVal.replaceAllUsesWith(outBuf);
        
    //     b.create<func::ReturnOp>(ret.getLoc());
    //     ret.erase();
    //   });
    // });

    module.print(llvm::outs());
    

  // // builder.create<emitc::FuncOp>(module.getLoc(), "fullyconnected_i8_scale", )
  // builder.create<emitc::FuncOp>(module.getLoc(), "maxpool_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "avgpool_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "relu", )
  // builder.create<emitc::FuncOp>(module.getLoc(), "quantize", )
  // builder.create<emitc::FuncOp>(module.getLoc(), "dequantize", )
  // builder.create<emitc::FuncOp>(module.getLoc(), "transpose_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "elemadd_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "elemadd_relu_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "elemsub_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "elemdiv_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "elemmax_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "splat_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "softmax", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "tanh_activation", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "tanh_activation_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "copy_f32", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "copy_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "matmul", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "make_dict", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "ones_abs", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "add_bias", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "add_bias1", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "sigmoid", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "apply_sigmoid", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "apply_tanh", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_hidden_state", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_update_states", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "matmul_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "compute_quant_params_sym", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "dequantize_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "quantize_i8", )
  // builder.create<emitc::FuncOp>(module.getLoc(), "compute_shift", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "matmul_i8_shift", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_input_gate_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_forget_gate_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_candidate_cell_state_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_output_gate_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_cell_state_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_cell_state", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_forward_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "dense_layer_forward_i8", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_input_gate", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_forget_gate", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_candidate_cell_state", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_calculate_output_gate", )
  // // builder.create<emitc::FuncOp>(module.getLoc(), "lstm_layer_forward", )
  

  int counter = 0;
  module.walk([&](Operation *op) {
    if (op->getDialect()->getNamespace() == "harx") {
      llvm::StringRef dialect = op->getDialect()->getNamespace();
      llvm::StringRef fullName = op->getName().getStringRef();
      llvm::StringRef opName = fullName.substr(dialect.size() + 1);
      std::string name = opName.str();
      std::transform(
        name.begin(), name.end(), name.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
      );

      op->setAttr("harx_name", StringAttr::get(&ctx, name + "_" + std::to_string(counter++)));
    }
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();


// module.walk([&](func::FuncOp fn) {
//   if (fn.getName() != "main_graph") return;

//   // 1) Original signature
//   FunctionType oldType = fn.getFunctionType();

//   // 2) Build new input types: change any strided memref to identity layout
//   SmallVector<Type, 4> newInputs;
//   for (Type t : oldType.getInputs()) {
//     if (auto mt = mlir::dyn_cast<MemRefType>(t)) {
//       auto ctx = fn.getContext();
//       // identity layout map of same rank
//       AffineMap idMap = AffineMap::getMultiDimIdentityMap(mt.getRank(), ctx);
//       // new static memref type
//       newInputs.push_back(
//         MemRefType::get(
//           mt.getShape(),
//           mt.getElementType(),
//           idMap,
//           mt.getMemorySpace()));
//     } else {
//       newInputs.push_back(t);
//     }
//   }

//   // 3) Results remain unchanged
//   SmallVector<Type, 4> newResults(oldType.getResults().begin(),
//                                    oldType.getResults().end());

//   // 4) Create and apply the new FunctionType
//   OpBuilder builder(fn);
//   auto newFuncType = builder.getFunctionType(newInputs, newResults);
//   fn.setType(newFuncType);  // updates external signature :contentReference[oaicite:2]{index=2}

//   // 5) Update the entry block arguments too
//   Block &entry = fn.getBody().front();
//   assert(entry.getNumArguments() == newInputs.size() &&
//          "argument count must match new function type");
//   for (unsigned i = 0, e = newInputs.size(); i < e; ++i) {
//     entry.getArgument(i).setType(newInputs[i]);  // patch block arg :contentReference[oaicite:3]{index=3}
//   }
// });


//   module.walk([&](Operation *op) {
//     llvm::outs() << "Processing operation: " << op->getName() << "\n";
//     llvm::outs() << "Found operation: " << op->getDialect() << "\n";
//   });

}

std::unique_ptr<Pass> createHARXToLLVMPass() {
  return std::make_unique<HARXToLLVMLoweringPass>();
}

} // namespace onnx_mlir
