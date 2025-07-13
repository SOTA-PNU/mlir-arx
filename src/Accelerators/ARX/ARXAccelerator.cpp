
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- ARXAccelerator.cpp -----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for the IBM Telum processor.
//
//===----------------------------------------------------------------------===//
// clang-format on

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/ARX/Compiler/ARXCompilerUtils.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARX.hpp"
// #include "src/Accelerators/ARX/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
// #include "src/Accelerators/ARX/Conversion/ZHighToZLow/ZHighToZLow.hpp"
// #include "src/Accelerators/ARX/Conversion/ZLowToLLVM/ZLowToLLVM.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/LARX/LARXOps.hpp"
#include "src/Accelerators/ARX/ARXAccelerator.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
// #include "src/Accelerators/ARX/Support/ARXLimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include <iostream>

#include <memory>

#define DEBUG_TYPE "ARXAccelerator"

extern llvm::cl::OptionCategory OMARXPassOptions;

namespace onnx_mlir {
namespace accel {

Accelerator *createARX() { return ARXAccelerator::getInstance(); }

ARXAccelerator *ARXAccelerator::instance = nullptr;

ARXAccelerator *ARXAccelerator::getInstance() {
  if (instance == nullptr)
    instance = new ARXAccelerator();
  return instance;
}

ARXAccelerator::ARXAccelerator() : Accelerator(Accelerator::Kind::ARX) {
  llvm::outs() << "Creating an ARX accelerator asdfasdf\n";

  // Print a warning if mcpu is not set or < z16.
  // if (!isCompatibleWithARXLevel(ARX_Z16))
    // llvm::outs() << "Warning: No ARX code is generated because --mcpu is not "
                    // "set or < z16.\n";

  acceleratorTargets.push_back(this);
  // Order is important! libRuntimeARX depends on libzdnn
  // addCompilerConfig("CHACHA I DONT KNOW THAT", {"RuntimeARX", "zdnn"}, true);
};

ARXAccelerator::~ARXAccelerator() { delete instance; }

uint64_t ARXAccelerator::getVersionNumber() const { return 0; }

void ARXAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) const {
  LLVM_DEBUG(llvm::dbgs() << "Adding passes for ARX accelerator\n");
  llvm::outs() << "Adding passes for ARX accelerator\n";
  onnx_mlir::addPassesARX(module, pm, emissionTarget, outputNameNoExt);

}

void ARXAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering dialects for ARX accelerator\n");
  llvm::outs() << "Registering dialects for ARX accelerator\n";
  
  registry.insert<onnx_mlir::harx::HARXDialect>();
  registry.insert<onnx_mlir::larx::LARXDialect>();
}

void ARXAccelerator::registerPasses(int optLevel) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering passes for ARX accelerator\n");
  llvm::outs() << "Registering passes for ARX accelerator\n";
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXToHARXPass();
  });
  // mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
  //   return onnx_mlir::createZHighToONNXPass();
  // });
}

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-to-krnl pass
  //===--------------------------------------------------------------------===//
  mlir::MemRefType ARXAccelerator::convertTensorTypeToMemRefType(
      const mlir::TensorType tensorType) const {
        return nullptr;
      }
  void ARXAccelerator::conversionTargetONNXToKrnl(
      mlir::ConversionTarget &target) const{
      }
  void ARXAccelerator::rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
      mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const{
        

  }

  int64_t ARXAccelerator::getDefaultAllocAlignment(
      const mlir::TensorType tensorType) const {
        return 0;
      }
  //===--------------------------------------------------------------------===//
  // Hooks for krnl-to-llvm pass
  //===--------------------------------------------------------------------===//
  void ARXAccelerator::conversionTargetKrnlToLLVM(
      mlir::ConversionTarget &target) const {
      }
  void ARXAccelerator::rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
      mlir::LLVMTypeConverter &typeConverter,
      mlir::MLIRContext *ctx) const {
      }



} // namespace accel
} // namespace onnx_mlir
