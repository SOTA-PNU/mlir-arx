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
#include "src/Accelerators/ARX/Conversion/HARXToLLVM/HARXToLLVMCommon.hpp"
#include "src/Accelerators/ARX/Dialect/HARX/HARXOps.hpp"
#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace larx {

// static bool FUNC_CALL_FOR_DLF16_CONVERSION = false;
// static bool SIMD_FOR_DLF16_CONVERSION = true;

// zdnn_data_layouts UNDEFINED_ZDNN_LAYOUT = static_cast<zdnn_data_layouts>(255);

// Obtain a zDNN API for an elementwise ZLow operation.
template <>
API APIFor<harx::HARXAddOp>() {
  return API::HARX_ADD;
}

template <typename BinaryElementwiseOp>
class HARXBinaryElementwiseOpLowering : public ConvertToLLVMPattern {
public:
  explicit HARXBinaryElementwiseOpLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            BinaryElementwiseOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // ModuleOp module = op->getParentOfType<ModuleOp>();
    // Location loc = op->getLoc();
    // BinaryElementwiseOp binaryOp = mlir::cast<BinaryElementwiseOp>(op);
    // typename BinaryElementwiseOp::Adaptor operandAdaptor(operands);

    // Value input1 = operandAdaptor.getX();
    // Value input2 = operandAdaptor.getY();
    // Value shape = operandAdaptor.getShape();
    // Value output = operandAdaptor.getOut();
    // Type llvmElementTy = typeConverter->convertType(
    //     mlir::cast<MemRefType>(op->getOperand(0).getType()).getElementType());

    // ZTensorHelper zTensorHelper =
    //     ZTensorHelper(rewriter, loc, module, apiRegistry);

    // // Get zDNN data type.
    // zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // // Get zDNN data layout.
    // zdnn_data_layouts zDNNDataLayout =
    //     convertLayoutAttrToZDNNDataLayout(0, binaryOp.getLayoutAttr());

    // // Get the dimensions of the original shape (the shape before stickifying)
    // // used for creating a zTensor.
    // std::vector<Value> dims =
    //     getDimsFromShapeMemRef(rewriter, loc, module, shape,
    //         /*layout=*/zDNNDataLayout);

    // // Create the first zTensor input.
    // Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input1);
    // ZTensor inputZTensor1 =
    //     zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
    //         /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
    //         /*isTransformed=*/true);

    // // Create the second zTensor input.
    // stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input2);
    // ZTensor inputZTensor2 = zTensorHelper.getZTensor(
    //     /*preTransformedDescPtr=*/inputZTensor1.preTransformedDescPtr,
    //     /*transformedDescPtr=*/inputZTensor1.transformedDescPtr,
    //     /*bufferSize=*/inputZTensor1.bufferSize,
    //     /*alignedBuffer=*/stickI8Ptr,
    //     /*isTransformed=*/true);

    // // Create an output zTensor.
    // stickI8Ptr = zTensorHelper.getAlignedI8Ptr(output);
    // ZTensor outputZTensor = zTensorHelper.getZTensor(
    //     /*preTransformedDescPtr=*/inputZTensor1.preTransformedDescPtr,
    //     /*transformedDescPtr=*/inputZTensor1.transformedDescPtr,
    //     /*bufferSize=*/inputZTensor1.bufferSize,
    //     /*alignedBuffer=*/stickI8Ptr,
    //     /*isTransformed=*/true);

    // // Ready to call a zDNN elementwise API.
    // callApi(rewriter, loc, module, apiRegistry, APIFor<BinaryElementwiseOp>(),
    //     {toOpaquePtr(rewriter, loc, module, inputZTensor1.val),
    //         toOpaquePtr(rewriter, loc, module, inputZTensor2.val),
    //         toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    // rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

void populateHARXToLLVMConversionPattern(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  ApiRegistry apiRegistry = RegisterAllApis(ctx);
  // clang-format off
  patterns.insert<HARXBinaryElementwiseOpLowering<onnx_mlir::harx::HARXAddOp>>(ctx, typeConverter, apiRegistry);
  // clang-format on
}

} // namespace larx
} // namespace onnx_mlir
