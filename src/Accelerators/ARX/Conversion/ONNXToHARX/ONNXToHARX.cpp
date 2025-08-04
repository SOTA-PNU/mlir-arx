/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ZHighToZLow.cpp - ZHigh dialect to ZLow lowering -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to ZLow operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"   

#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARX.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps/ShapeHelper.hpp"
// #include "src/Accelerators/ARX/Dialect/LARX/LARXOps.hpp"
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
// #include "src/Accelerators/ARX/Support/Stickify/Convert.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/pass/OnnxToHarxPass.h"

#define DEBUG_TYPE "harx-to-larx"

using namespace mlir;
using namespace onnx_mlir::harx;

namespace onnx_mlir {
namespace harx {

void populateONNXToARXConversionPattern(TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx, bool enableParallel) {
  llvm::outs() << "populateONNXToARXConversionPattern CALLED!!!!\n";
  
  patterns.add<EraseEntryPointPattern>(patterns.getContext());
  patterns.add<EraseFuncOnnxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxDequantToArxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxQuantToArxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxFCToArxFCPattern>(patterns.getContext());
  patterns.add<RewriteOnnxReshapeToArxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxMaxPoolToArxPattern>(patterns.getContext());
  patterns.add<RewriteOnnxQConv2DToArxPattern>(patterns.getContext());
}

} // namespace harx
} // namespace onnx_mlir
