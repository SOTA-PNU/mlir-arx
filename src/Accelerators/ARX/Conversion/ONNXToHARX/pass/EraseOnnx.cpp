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
#include "src/Accelerators/ARX/Pass/ARXPasses.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/pass/OnnxToHarxPass.h"

#define DEBUG_TYPE "harx-to-larx"

using namespace mlir;
using namespace onnx_mlir::harx;

namespace onnx_mlir {
namespace harx {


LogicalResult EraseEntryPointPattern::matchAndRewrite(
    ONNXEntryPointOp op, mlir::PatternRewriter &rewriter) const {
    rewriter.eraseOp(op);          // marker 제거
    return mlir::success();
}

LogicalResult EraseFuncOnnxPattern::matchAndRewrite(func::FuncOp fn, PatternRewriter &rewriter) const {
    auto fnType = fn.getFunctionType();
    bool hasAttr = false;

    // Check args
    for (unsigned i = 0, e = fnType.getNumInputs(); i < e; ++i)
        if (fn.getArgAttr(i, "onnx.name")) {
        hasAttr = true;
        break;
        }
    // Check results only if no attr found yet
    if (!hasAttr)
        for (unsigned i = 0, e = fnType.getNumResults(); i < e; ++i)
        if (fn.getResultAttr(i, "onnx.name")) {
            hasAttr = true;
            break;
        }

    // If no onnx.name anywhere, do nothing
    if (!hasAttr)
        return failure();

    // Remove them in-place
    rewriter.startOpModification(fn);
    // strip from args
    for (unsigned i = 0, e = fnType.getNumInputs(); i < e; ++i)
        if (fn.getArgAttr(i, "onnx.name"))
        fn.removeArgAttr(i, "onnx.name");
    // strip from results
    for (unsigned i = 0, e = fnType.getNumResults(); i < e; ++i)
        if (fn.getResultAttr(i, "onnx.name"))
        fn.removeResultAttr(i, rewriter.getStringAttr("onnx.name"));
    rewriter.finalizeOpModification(fn);

    return success();
}

} // namespace harx
} // namespace onnx_mlir
