/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVM.cpp - Lowering from ZLow to LLVM ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines patterns to lower ZLow operations to LLVM dialect.
//
// Note that once a type is lowed to LLVM, it can be opaque pointer and its
// element type is lost. Thus, to get element type in LLVM, get it from the
// original operation (not via operandAdaptor), then use
// `typeConverter(elementType)` to convert it to LLVM. See `ZLowStickLowering`
// as an example.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVM.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "src/Accelerators/ARX/Conversion/HARXToLLVM/pass/HarxToEmitC.h"


using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace larx {

void populateHARXToLLVMConversionPattern(mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  // clang-format off
  
  // patterns.add<QuantizeToEmitC>(typeConverter, ctx);
  // patterns.add<ConstantToEmitCPattern>(typeConverter, ctx);
  patterns.add<FunctionToEmitCPattern>(patterns.getContext());
  patterns.add<QuantizeToEmitCPattern>(patterns.getContext());
  patterns.add<DequantizeToEmitCPattern>(patterns.getContext());
  patterns.add<ConvolutionShiftToEmitCPattern>(patterns.getContext());
  patterns.add<MaxPoolToEmitCPattern>(patterns.getContext());
  patterns.add<FullyConnectedToEmitCPattern>(patterns.getContext());
  patterns.add<ReShapeToEmitCPattern>(patterns.getContext());
  
  // clang-format on
}

} // namespace larx
} // namespace onnx_mlir
