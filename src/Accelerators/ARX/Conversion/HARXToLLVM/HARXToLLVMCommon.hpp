/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVMCommon.hpp - Lowering from ZLow to LLVM ---------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in lowering ZLow to LLVM
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZLOW_TO_LLVM_COMMON_H
#define ONNX_MLIR_ZLOW_TO_LLVM_COMMON_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"


namespace onnx_mlir {
namespace larx {

enum class API {
  NULL_API,
  HARX_ADD,
};

// Obtain a zDNN API for an elementwise HARX operation.
template <typename HARXOp>
API APIFor() {
  return API::NULL_API;
}

// API specs to declare external function types.
struct ApiSpec {
  API id;
  std::string name;
  mlir::Type outputTy;
  mlir::SmallVector<mlir::Type, 4> inputTys;
  bool isVarArg;

  ApiSpec(API id, const std::string &name, mlir::Type outputTy,
      mlir::ArrayRef<mlir::Type> inputTys, const bool isVarArg)
      : id(id), name(name), outputTy(outputTy),
        inputTys(inputTys.begin(), inputTys.end()), isVarArg(isVarArg) {}

  mlir::Type funcTy() {
    return mlir::LLVM::LLVMFunctionType::get(outputTy, inputTys,
        /*isVarArg=*/isVarArg);
  }
};

using ApiRegistry = std::map<API, ApiSpec>;
ApiRegistry RegisterAllApis(mlir::MLIRContext *context);


} // namespace zlow
} // namespace onnx_mlir
#endif
