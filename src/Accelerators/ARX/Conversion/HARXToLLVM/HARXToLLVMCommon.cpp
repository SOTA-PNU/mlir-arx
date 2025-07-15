/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVMCommon.hpp - Lowering from ZLow to LLVM ---------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in lowering ZLow to LLVM
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVMCommon.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace larx {

ApiRegistry RegisterAllApis(MLIRContext *context) {
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto opaquePtrTy = krnl::getI8PointerType(context);
  auto int16Ty = IntegerType::get(context, 16);
  auto int32Ty = IntegerType::get(context, 32);
  auto int64Ty = IntegerType::get(context, 64);
  auto float32Ty = FloatType::getF32(context);

  // Declare API type as an enum value, its string name and an LLVM Type
  // specifying its signature.
  //
  // Note: Though zDNN APIs use int32 for their integer parameters, we have to
  // pass them as int64 to avoid segfault when compiling with -O{1,2,3} on
  // s390x. This is an issue about use of MLIR on s390x, where int32 is carried
  // inside a 64-bit register and the higher 4 bytes are not cleared correctly.
  // More info can be found here:
  // https://github.com/onnx/onnx-mlir/pull/567#issuecomment-841061475
  //
  // clang-format off
  std::vector<ApiSpec> apiSpecs = {
    // Elementwise operations
    ApiSpec(API::HARX_ADD, "zdnn_add_ext", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
  };
  // clang-format on

  // Declare APIs in the current module and build an API registry mapping api.
  ApiRegistry registry;
  for (auto &apiSpec : apiSpecs) {
    registry.emplace(apiSpec.id, apiSpec);
  }

  return registry;
}

// Call a registered API, return the return SSA values if only one result is
// returned, otherwise return nullptr.
Value callApi(PatternRewriter &rewriter, Location loc, ModuleOp module,
    ApiRegistry registry, API apiId, ArrayRef<Value> params) {
  MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
  // To be used as parameters in LLVM::CallOp, voidTy must be converted
  // to empty list to avoid emission of an SSA value with voidTy. However,
  // we still keep using LLVM voidTy (as opposed to empty list) when recording
  // API function signatures in API registry because when declaring API
  // functions in LLVM IR, the correct way to indicate an output type for
  // "void" is still LLVM voidTy. Relevant discussion thread:
  // https://github.com/onnx/onnx-mlir/issues/255.
  ApiSpec apiSpec = registry.at(apiId);
  FlatSymbolRefAttr symbolRef =
      create.llvm.getOrInsertSymbolRef(module, StringRef(apiSpec.name),
          apiSpec.outputTy, apiSpec.inputTys, apiSpec.isVarArg);
  SmallVector<Type, 1> outputTys;
  Type outputTy = apiSpec.outputTy;
  if (!mlir::isa<LLVM::LLVMVoidType>(outputTy))
    outputTys.emplace_back(outputTy);
  return create.llvm.call(ArrayRef<Type>(outputTys), symbolRef,
      ArrayRef<Value>(params), apiSpec.isVarArg);
}

} // namespace zlow
} // namespace onnx_mlir
