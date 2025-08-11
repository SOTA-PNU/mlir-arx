/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPACompilerUtils.cpp ---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Utilities for  NNPA
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "src/Accelerators/ARX/Compiler/ARXCompilerOptions.hpp"
#include "src/Accelerators/ARX/Compiler/ARXCompilerUtils.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
// #include "src/Accelerators/ARX/Dialect/LARX/LARXOps.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"
// Bufferization
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// MemRef → LLVM
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
// Func → LLVM
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#define DEBUG_TYPE "ARXCompilerUtils"

#include <fstream>


using namespace mlir;
using namespace onnx_mlir;

extern std::string outputBaseName;


namespace onnx_mlir {

  extern llvm::cl::opt<ARXEmissionTargetType> arxEmissionTarget;

void addPassesARX(mlir::OwningOpRef<mlir::ModuleOp> &module,
  mlir::PassManager &pm, EmissionTargetType &emissionTarget,
  std::string outputNameNoExt) {

  llvm::outs() << "configurePasses\n";
  configurePasses();

  llvm::outs() << "addONNXToMLIRPasses\n";
  addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty());

  // onnx-mlir -o 01_mnist -EmitONNXIR ./mnist-12-int8.onnx
  // onnx-mlir-opt -o 01_mnist.harx.mlir -maccel=ARX --canonicalize --convert-onnx-to-harx --canonicalize  ./01_mnist.onnx.mlir
  // onnx-mlir-opt -o 01_mnist.emitc.mlir -maccel=ARX --canonicalize --convert-harx-to-emitc -reconcile-unrealized-casts --canonicalize 01_mnist.harx.mlir
// emissionTarget >= EmitMLIR
  if (arxEmissionTarget >= EmitARXIR) {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(onnx_mlir::createONNXToHARXPass());
    pm.addPass(mlir::createCanonicalizerPass());
  } 

  if (arxEmissionTarget >= EmitEmitCIR) {
    pm.addPass(onnx_mlir::createHARXToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  
  // mlir::emitc::translateToCpp(module.get(), raw_output_stream);
  // llvm::LLVMContext llvmContext;
  // mlir::registerBuiltinDialectTranslation(*(module.get().getContext()));
  // mlir::registerLLVMDialectTranslation(*(module.get().getContext()));
  // std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
}

} // namespace onnx_mlir
