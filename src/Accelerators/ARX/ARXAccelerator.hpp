/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.hpp ----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// ===========================================================================
//
// Accelerator support for the IBM Telum coprocessor.
//
//===---------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ARX_ACCELERATOR_H
#define ONNX_MLIR_ARX_ACCELERATOR_H

#include "mlir/IR/BuiltinTypes.h"
#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

/// Singleton class to construct an ARX accelerator.
class ARXAccelerator final : public Accelerator {
private:
  static ARXAccelerator *instance;
  ARXAccelerator();

public:
  /// Singleton should not be clonable or assignable.
  ARXAccelerator(ARXAccelerator &) = delete;
  void operator=(const ARXAccelerator &) = delete;

  ~ARXAccelerator();

  /// Creates an instance on the first invocation. Subsequent invocations
  /// return the existing instance.
  static ARXAccelerator *getInstance();

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const Accelerator *accel) {
    return accel->getKind() == Accelerator::Kind::ARX;
  }
  static bool classof(const ARXAccelerator *) { return true; }

  uint64_t getVersionNumber() const final;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
      std::string outputNameNoExt) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//
  virtual void registerDialects(mlir::DialectRegistry &registry) const final;
  virtual void registerPasses(int optLevel) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-to-krnl pass
  //===--------------------------------------------------------------------===//
  virtual mlir::MemRefType convertTensorTypeToMemRefType(
      const mlir::TensorType tensorType) const final;
  virtual void conversionTargetONNXToKrnl(
      mlir::ConversionTarget &target) const final;
  virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
      mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const final;
  virtual int64_t getDefaultAllocAlignment(
      const mlir::TensorType tensorType) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for krnl-to-llvm pass
  //===--------------------------------------------------------------------===//
  virtual void conversionTargetKrnlToLLVM(
      mlir::ConversionTarget &target) const final;
  virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
      mlir::LLVMTypeConverter &typeConverter,
      mlir::MLIRContext *ctx) const final;
};

} // namespace accel
} // namespace onnx_mlir
#endif
