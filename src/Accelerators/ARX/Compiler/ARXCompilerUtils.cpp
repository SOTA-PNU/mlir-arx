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
#include "src/Accelerators/ARX/Dialect/LARX/LARXOps.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#include <iostream>
#define DEBUG_TYPE "ARXCompilerUtils"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
void addPassesARX(mlir::OwningOpRef<mlir::ModuleOp> &module,
  mlir::PassManager &pm, EmissionTargetType &emissionTarget,
  std::string outputNameNoExt) {
  llvm::outs() << "CHACHA! " << arxEnableCuteChaCha << " \n";

  llvm::outs() << "configurePasses\n";
  configurePasses();

  llvm::outs() << "addONNXToMLIRPasses\n";
  addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty());

  // for (unsigned i = 0; i < 3; i++) {
    // pm.addPass(onnx_mlir::createRewriteONNXForHARXPass());
    // pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  // }
  // pm.addPass(onnx_mlir::createRewriteONNXForHARXPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addPass(onnx_mlir::createONNXToHARXPass());
  pm.addPass(mlir::createCanonicalizerPass());

}

} // namespace onnx_mlir
