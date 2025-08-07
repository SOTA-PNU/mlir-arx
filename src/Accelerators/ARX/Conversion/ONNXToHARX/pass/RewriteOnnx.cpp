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

#include <iterator>

#define DEBUG_TYPE "harx-to-larx"

using namespace mlir;
using namespace onnx_mlir::harx;

namespace onnx_mlir {
namespace harx {

mlir::LogicalResult RewriteOnnxQuantToArxPattern::matchAndRewrite(ONNXQuantizeLinearOp op, mlir::PatternRewriter &rewriter) const {

    auto yScale = op.getYScale().getDefiningOp<ONNXConstantOp>();
    if (!yScale)
        return rewriter.notifyMatchFailure(op, "y_scale must be constant");

    auto yZeroPoint = op.getYZeroPoint().getDefiningOp<ONNXConstantOp>();
    if (!yZeroPoint)
        return rewriter.notifyMatchFailure(op, "y_zero_point must be constant");
    llvm::dbgs() << "RewriteOnnxQuantToArxPattern: " << op.getOperand(0) << "\n\n\n";
    
    // no func dialect
    if (op.getOperand(0).getDefiningOp()) {
        llvm::dbgs() << "Remove ONNXQuantizeLinearOp, since it is not func dialect.\n";
        rewriter.replaceOp(op, op.getOperand(0));
        return mlir::success();
    }

    auto scaleAttr = mlir::cast<DenseFPElementsAttr>(yScale.getValueAttr());
    auto getYScale = rewriter.getFloatAttr(scaleAttr.getElementType(), *(scaleAttr.getValues<APFloat>().begin()));

    auto yZeroPointAttr = mlir::cast<DenseIntElementsAttr>(yZeroPoint.getValueAttr());
    auto getYZeroPoint = rewriter.getIntegerAttr(yZeroPointAttr.getElementType(), *(yZeroPointAttr.getValues<APInt>().begin()));

    auto ops = rewriter.create<HARXQuantizationOp>(op.getLoc(), op.getType(), op.getX(), getYScale, getYZeroPoint);
    rewriter.replaceOp(op, ops.getResult());

    return mlir::success();
}


mlir::LogicalResult RewriteOnnxDequantToArxPattern::matchAndRewrite(ONNXDequantizeLinearOp op, mlir::PatternRewriter &rewriter) const {
    auto xScale = op.getXScale().getDefiningOp<ONNXConstantOp>();
    if (!xScale)
        return rewriter.notifyMatchFailure(op, "x_scale must be constant");

    auto xZeroPoint = op.getXZeroPoint().getDefiningOp<ONNXConstantOp>();
    if (!xZeroPoint)
        return rewriter.notifyMatchFailure(op, "x_zero_point must be constant");
    
    bool flags = true;
    for (auto &use : op.getResult().getUses()) {
        if (use.getOwner()->getDialect()->getNamespace() == "func") {
            flags = false;
        }
    }
    if (flags) {
        llvm::dbgs() << "Remove ONNXDequantizeLinearOp, since it is not func dialect.\n";
        rewriter.replaceOp(op, op.getOperand(0));
        return mlir::success();
    }

    auto ctx = rewriter.getContext();

    auto scaleAttr = mlir::cast<DenseFPElementsAttr>(xScale.getValueAttr());
    auto getXScale = rewriter.getFloatAttr(scaleAttr.getElementType(), *(scaleAttr.getValues<APFloat>().begin()));

    auto xZeroPointAttr = mlir::cast<DenseIntElementsAttr>(xZeroPoint.getValueAttr());
    auto getXZeroPoint = rewriter.getIntegerAttr(xZeroPointAttr.getElementType(), *(xZeroPointAttr.getValues<APInt>().begin()));

    auto u8Type = IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
    auto s32Type = IntegerType::get(ctx, 32, mlir::IntegerType::Signless);

    auto ops = rewriter.create<HARXDequantizationOp>(op.getLoc(), op.getType(), op.getX(), getXScale, getXZeroPoint, rewriter.getIntegerAttr(s32Type, 1), rewriter.getIntegerAttr(u8Type, 0));
    rewriter.replaceOp(op, ops.getResult());

    return mlir::success(); 
}


// SmallVector<APInt> computeShift(
//     float input_scale,
//     float output_scale,
//     const SmallVector<APInt> &kernel_scale,
//     int channel_size) 
// {
//     // 1) 결과 벡터 초기화
//     std::vector<int32_t> shift_values(channel_size, 0);

//     for (int i = 0; i < channel_size; ++i) {
//         // 2) shift_quantized 계산
//         float shift_quantized = kernel_scale[i] * input_scale / output_scale;

//         // 3) 0 이하 예외 처리
//         if (shift_quantized <= 0.0f) {
//             shift_values[i] = 0;
//             continue;
//         }

//         // 4) M과 shift_cnt 초기화
//         float M = shift_quantized;
//         int shift_cnt = 0;

//         // 5) M > 1.0 → 1 미만이 될 때까지 2로 나누기
//         while (M > 1.0f) {
//             M /= 2.0f;
//             ++shift_cnt;
//         }

//         // 6) M <= 0.5 → 0.5 초과가 될 때까지 2로 곱하기
//         while (M <= 0.5f) {
//             M *= 2.0f;
//             --shift_cnt;
//         }

//         // 7) 음수면 양수로
//         if (shift_cnt < 0)
//             shift_cnt = -shift_cnt;

//         shift_values[i] = static_cast<int32_t>(shift_cnt);
//     }

//     return shift_values;
// }


mlir::LogicalResult RewriteOnnxFCToArxFCPattern::matchAndRewrite(ONNXCustomOp op, mlir::PatternRewriter &rewriter) const {
    llvm::dbgs() << "\n\n\n\n\n";
    llvm::dbgs() << "01 RewriteOnnxFCToArxFCPattern, op : " << op << "\n";
    auto defOp = op.getInputs()[0].getUses();
    auto size = std::distance(defOp.begin(), defOp.end());
    if (size != 1) { 
        llvm::dbgs() << "RewriteOnnxFCToArxFCPattern, due to size is " << size << " value.\n";
        return mlir::failure();
    }
    if (op.getFunctionName() != "QLinearAdd") {
        return mlir::failure();
    }
    
    // get MatmulOp from AddOp from %0
    auto onnx_matmul_op = op.getInputs()[0].getDefiningOp<ONNXQLinearMatMulOp>();

    if (!onnx_matmul_op) {
        onnx_matmul_op.dump();
        return mlir::failure();
    }
    
    llvm::dbgs() << "02 RewriteOnnxFCToArxFCPattern, op : " << op << "\n";

    auto bias = op.getInputs()[3];
    auto biasOp = bias.getDefiningOp<ONNXConstantOp>();
    if (!bias)
        return rewriter.notifyMatchFailure(op, "bias must be constant");

    auto biasAttr = mlir::cast<DenseIntElementsAttr>(biasOp.getValueAttr());

    // Needs : UI8 -> I32 for HARXMatMulOp and bias
    auto ui8_array = mlir::cast<ShapedType>(biasAttr.getType());
    auto i32 = IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Signed);
    auto i32_array = ui8_array.clone(i32);

    SmallVector<APInt> data;
    data.reserve(ui8_array.getNumElements());
    for (auto item : biasAttr.getValues<APInt>()) { 
        auto value = item.getSExtValue();
        data.push_back(APInt(32, value, true, true));
    }
    
    auto denseAttr = DenseElementsAttr::get(i32_array, data);
    auto newConstBias = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ i32_array },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", denseAttr)
                        });

    auto ui8_type = rewriter.getIntegerType(8, false);

    auto C_zero_point_ui8 = op.getInputs()[7];
    auto C_zero_point_value = mlir::cast<DenseIntElementsAttr>(C_zero_point_ui8.getDefiningOp<ONNXConstantOp>().getValueAttr()).getValues<APInt>()[0].getSExtValue();
    auto C_zero_point_i32 = rewriter.getI32IntegerAttr(C_zero_point_value);

    // auto scale3_float32 = onnx_matmul_op.getAScale(); 
    auto scale5_float32 = op.getInputs()[6]; // getCScale(); 
    auto w_scale3_float32 = onnx_matmul_op.getBScale(); // BScale
    // auto bScale = mlir::cast<DenseIntElementsAttr>(scale3_float32.getType());
    // llvm::dbgs() << "bScale : " << bScale << "\n";
    llvm::dbgs() << "w_scale3_float32 : " << w_scale3_float32 << "\n";
    llvm::dbgs() << "scale5_float32 : " << scale5_float32 << "\n";

    // shift3_scale_int32 = compute_shift(scale3_float32, scale5_float32, w_scale3_float32, 10);

    SmallVector<APInt> shift_data;
    shift_data.reserve(ui8_array.getNumElements());
    for (auto item : biasAttr.getValues<APInt>()) { 
        auto value = item.getSExtValue();
        shift_data.push_back(APInt(32, value, true, true));
    }
    auto shiftDenseAttr = DenseElementsAttr::get(i32_array, shift_data);
    auto newShiftAttr = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ i32_array },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", shiftDenseAttr)
                        });
    
    auto mat_mul_kernel_i8_to_ui8 = onnx_matmul_op.getB().getDefiningOp<ONNXConstantOp>();
    if (!mat_mul_kernel_i8_to_ui8)
        return rewriter.notifyMatchFailure(op, "mat_mul_kernel_i8_to_ui8 must be constant");
    
    auto mat_mul_kernel_i8_to_ui8_attr = mlir::cast<DenseIntElementsAttr>(mat_mul_kernel_i8_to_ui8.getValueAttr());
    auto mat_mul_kernel_i8_to_ui8_type = mlir::cast<ShapedType>(mat_mul_kernel_i8_to_ui8_attr.getType());
    auto mat_mul_kernel_ui8_type = mat_mul_kernel_i8_to_ui8_type.clone(rewriter.getIntegerType(8, false));
    SmallVector<APInt> matmul_kernel_ui8;
    matmul_kernel_ui8.reserve(mat_mul_kernel_i8_to_ui8_attr.getNumElements());
    for (auto item : mat_mul_kernel_i8_to_ui8_attr.getValues<APInt>()) {
        auto value = item.getSExtValue();
        matmul_kernel_ui8.push_back(APInt(8, value + 127, false));
    }
    auto matmul_kernel_attr = DenseElementsAttr::get(mat_mul_kernel_ui8_type, matmul_kernel_ui8);
    auto matmul_kernel_op = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ mat_mul_kernel_ui8_type },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", matmul_kernel_attr)
                        });

    llvm::dbgs() << "03 RewriteOnnxFCToArxFCPattern, shiftDenseAttr : " << newShiftAttr << "\n";
    llvm::dbgs() << "03 RewriteOnnxFCToArxFCPattern, matmulOp : " << newShiftAttr << "\n";

    auto matmulOp = rewriter.create<HARXMatMulOp>(op.getLoc(), onnx_matmul_op.getResult().getType(),
        onnx_matmul_op.getA(), matmul_kernel_op, newConstBias, C_zero_point_i32, 
        rewriter.getIntegerAttr(ui8_type, 1), newShiftAttr);
    llvm::dbgs() << "03 RewriteOnnxFCToArxFCPattern, matmulOp : " << matmulOp << "\n";

    rewriter.replaceOp(op, matmulOp.getResult());
    rewriter.replaceOp(onnx_matmul_op, matmulOp.getResult());

    auto d = op.getResult(0).getUses();
    for (auto& use : d) { 
        use.get().setType(matmulOp.getResult().getType());
    }

    llvm::dbgs() << "04 fin : \n";

    return mlir::success();
}



mlir::LogicalResult RewriteOnnxReshapeToArxPattern::matchAndRewrite(ONNXReshapeOp op, mlir::PatternRewriter &rewriter) const {
    llvm::dbgs() << "RewriteOnnxReshapeToArxPattern, op : " << op << "\n";
    
    auto input_type = op.getOperand(0).getType();
    
    auto ranked_input = mlir::cast<RankedTensorType>(input_type);
    auto shape = ranked_input.getShape();

    mlir::SmallVector<APInt> transpose_seq; // 0, 3, 1, 2 -> DenseAttr is End
    mlir::SmallVector<APInt> transpose_shape; // 1,4,4 16
    ArrayRef<int64_t> transpose_input_shape { (int64_t)shape.size() };

    transpose_seq.reserve(shape.size());
    transpose_shape.reserve(shape.size());
    for (auto dim : {0,3,1,2}) {
        transpose_seq.push_back(APInt(32, dim, true));
        transpose_shape.push_back(APInt(32, shape[dim], true));
    }

    auto i32_type = rewriter.getIntegerType(32, true);

    auto four_shape = RankedTensorType::get(transpose_input_shape, i32_type);
    auto seq = DenseElementsAttr::get(mlir::cast<ShapedType>(four_shape), transpose_seq);
    auto reshape = DenseElementsAttr::get(mlir::cast<ShapedType>(four_shape), transpose_shape);

    auto constant_seq_op = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ four_shape },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", seq)
                        });

    auto constant_reshape_op = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ four_shape },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", reshape)
                        });
    
    auto ui8_type = IntegerType::get(rewriter.getContext(), 8, mlir::IntegerType::Unsigned);

    auto input_fp32_type = mlir::cast<ShapedType>(op.getOperand(0).getType());
    auto input_ui8_array = input_fp32_type.clone(ui8_type);
    // auto castedInput = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), input_ui8_array, op.getOperand(0)).getResult(0);
    op.getOperand(0).setType(input_ui8_array);

    auto result_fp32_type = mlir::cast<ShapedType>(op.getResult().getType());
    auto result_ui8_array = result_fp32_type.clone(ui8_type);
    // op.getResult().setType(result_ui8_array);

    auto transpose_op = rewriter.create<HARXTransposeOp>(op.getLoc(), result_ui8_array, op.getOperand(0), constant_seq_op, constant_reshape_op);
    rewriter.replaceOp(op, transpose_op.getResult());

    return mlir::success();
}

// 강제적으로 ui8 형태로 변환함, 입력과 출력 레이어에 대해서.
mlir::LogicalResult RewriteOnnxMaxPoolToArxPattern::matchAndRewrite(ONNXMaxPoolSingleOutOp op, mlir::PatternRewriter &rewriter) const {
    llvm::dbgs() << "RewriteOnnxMaxPoolToArxPattern, op : " << op << "\n";
    
    auto fp32_result_type = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto ui8_type = rewriter.getIntegerType(8, false);
    auto ui8_result_array = fp32_result_type.clone(ui8_type);
    
    auto fp32_input_type = mlir::cast<RankedTensorType>(op.getOperand().getType());
    auto ui8_input_array = fp32_input_type.clone(ui8_type);
    op.getOperand().setType(ui8_input_array);

    auto kernel_shape = op.getKernelShapeAttr().getValue();
    auto pad_size = op.getPadsAttr().getValue();
    auto stride = op.getStridesAttr().getValue();

    auto ui8_kernel_height = rewriter.getIntegerAttr(ui8_type, mlir::cast<IntegerAttr>(kernel_shape[0]).getInt());
    auto ui8_kernel_width = rewriter.getIntegerAttr(ui8_type, mlir::cast<IntegerAttr>(kernel_shape[1]).getInt());
    auto ui8_pad_size = rewriter.getIntegerAttr(ui8_type, mlir::cast<IntegerAttr>(pad_size[0]).getInt());
    auto ui8_stride_size = rewriter.getIntegerAttr(ui8_type, mlir::cast<IntegerAttr>(stride[0]).getInt());
    
    // auto inputTy = mlir::cast<RankedTensorType>(op.getX().getType());
    // auto inputShape = inputTy.getShape();
    // auto inputN = rewriter.getIntegerAttr(ui8_type, inputShape[0]);
    // auto inputC = rewriter.getIntegerAttr(ui8_type, inputShape[1]);
    // auto inputH = rewriter.getIntegerAttr(ui8_type, inputShape[2]);
    // auto inputW = rewriter.getIntegerAttr(ui8_type, inputShape[3]);
    

    auto maxpool_op = rewriter.create<HARXMaxPoolOp>(op.getLoc(), ui8_result_array, op.getOperand(), 
                                                            // inputN, inputC, inputH, inputW,
                                                            ui8_kernel_height, ui8_kernel_width, ui8_pad_size, ui8_stride_size);
    rewriter.replaceOp(op, maxpool_op.getResult());

    return mlir::success();
}


mlir::LogicalResult RewriteOnnxQConv2DToArxPattern::matchAndRewrite(ONNXQLinearConvOp op, mlir::PatternRewriter &rewriter) const {
    llvm::dbgs() << "RewriteOnnxQConv2DToArxPattern, op : " << op << "\n";

    // auto input_attr = op.getX().getDefiningOp<ONNXConstantOp>().getValueAttr();
    auto kernel_attr = op.getW().getDefiningOp<ONNXConstantOp>().getValueAttr();
    auto bias_attr = op.getB().getDefiningOp<ONNXConstantOp>().getValueAttr();

    // auto scale1_float32 = op.getXScale(); 
    // auto scale2_float32 = op.getYScale();
    auto w_scale1_float32 = op.getWScale().getDefiningOp<ONNXConstantOp>().getValueAttr(); // BScale
    // auto shift1_scale_int32 = compute_shift(scale1_float32, scale2_float32, w_scale1_float32, 8)

    auto kernel_elem_attr = mlir::cast<DenseElementsAttr>(kernel_attr);
    auto kernel_i8_shape = mlir::cast<ShapedType>(kernel_elem_attr.getType());
    auto kernel_ui8_shape = kernel_i8_shape.clone(rewriter.getIntegerType(8, false));
    
    SmallVector<APInt> kernel_ui8;
    kernel_ui8.reserve(kernel_elem_attr.getNumElements());
    for (auto item : kernel_elem_attr.getValues<APInt>()) {
        auto value = item.getSExtValue();
        kernel_ui8.push_back(APInt(8, value + 127, false));
    }
    auto kernel_op = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ kernel_ui8_shape },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("value", DenseElementsAttr::get(kernel_ui8_shape, kernel_ui8))
        });
        
    auto bias_elem_attr = mlir::cast<DenseIntElementsAttr>(bias_attr);
    auto bias_i8_shape = mlir::cast<ShapedType>(bias_elem_attr.getType());
    auto bias_ui8_shape = bias_i8_shape.clone(rewriter.getIntegerType(32, true));
    SmallVector<APInt> bias_ui8;
    bias_ui8.reserve(bias_elem_attr.getNumElements());
    for (auto item : bias_elem_attr.getValues<APInt>()) {
        auto value = item.getSExtValue();
        bias_ui8.push_back(APInt(32, value + 127, true));
    }
    auto bias_dense_attr = DenseElementsAttr::get(bias_ui8_shape, bias_ui8);
    auto bias_op = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ bias_ui8_shape },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", bias_dense_attr)
                        });
                        
    auto kernel_scale_attr = mlir::cast<DenseElementsAttr>(w_scale1_float32);
    auto kernel_scale_fp32_shape = mlir::cast<ShapedType>(kernel_scale_attr.getType());
    auto kernel_scale_i32_shape = kernel_scale_fp32_shape.clone(rewriter.getIntegerType(32, true));
    SmallVector<APInt> kernel_scale_i32;
    for (auto item : kernel_scale_attr.getValues<APFloat>()) {
        // auto value = item.convertToDouble();
        kernel_scale_i32.push_back(APInt(32, 0, true));
    }
    auto kernel_scale_dense_attr = DenseElementsAttr::get(kernel_scale_i32_shape, kernel_scale_i32);
    auto kernel_scale_op = rewriter.create<HARXConstantOp>(
        op.getLoc(),
        /*resultTypes*/ TypeRange{ kernel_scale_i32_shape },
        /*operands*/     ValueRange{},
        /*attrs*/        ArrayRef<NamedAttribute>{
                           rewriter.getNamedAttr("value", kernel_scale_dense_attr)
                        });
    auto ui8_type = rewriter.getIntegerType(8, false);
    auto outputOffsetAttr = op.getYZeroPoint().getDefiningOp<ONNXConstantOp>().getValueAttr();
    auto output_offset = rewriter.getIntegerAttr(ui8_type, mlir::cast<DenseElementsAttr>(outputOffsetAttr).getValues<APInt>()[0].getZExtValue());
    auto do_bias = rewriter.getIntegerAttr(ui8_type, 1);
    auto do_relu = rewriter.getIntegerAttr(ui8_type, 0);
    auto pad_size = rewriter.getIntegerAttr(ui8_type, 2);
    auto stride_size = rewriter.getIntegerAttr(ui8_type, 1);
        
    llvm::dbgs() << "04 RewriteOnnxQConv2DToArxPattern, op : " << op.getX() << "\n";
    // llvm::dbgs() << "04 RewriteOnnxQConv2DToArxPattern, op : " << op.getX().getDefiningOp< << "\n";

    // auto inputTy = mlir::cast<RankedTensorType>(op.getX().getType());
    // auto kernelTy = mlir::cast<RankedTensorType>(op.getW().getType());
    // auto outputTy = mlir::cast<RankedTensorType>(op.getResult().getType());
    // auto inputShape = inputTy.getShape();
    // auto kernelShape = kernelTy.getShape();
    // auto outputShape = outputTy.getShape();
    
    // auto inputN = rewriter.getIntegerAttr(ui8_type, inputShape[0]);
    // auto inputC = rewriter.getIntegerAttr(ui8_type, inputShape[1]);
    // auto inputH = rewriter.getIntegerAttr(ui8_type, inputShape[2]);
    // auto inputW = rewriter.getIntegerAttr(ui8_type, inputShape[3]);
    
    // auto kernelN = rewriter.getIntegerAttr(ui8_type, kernelShape[0]);
    // auto kernelC = rewriter.getIntegerAttr(ui8_type, kernelShape[1]);
    // auto kernelH = rewriter.getIntegerAttr(ui8_type, kernelShape[2]);
    // auto kernelW = rewriter.getIntegerAttr(ui8_type, kernelShape[3]);

    // auto outputN = rewriter.getIntegerAttr(ui8_type, outputShape[0]);
    // auto outputC = rewriter.getIntegerAttr(ui8_type, outputShape[1]);
    // auto outputH = rewriter.getIntegerAttr(ui8_type, outputShape[2]);
    // auto outputW = rewriter.getIntegerAttr(ui8_type, outputShape[3]);

    auto conv_op = rewriter.create<HARXConvolutionShiftOp>(op.getLoc(), op.getResult().getType(), 
                                                                op.getX().getDefiningOp()->getResult(0), 
                                                                kernel_op, 
                                                                kernel_scale_op, 
                                                                bias_op, 
                                                                // inputN, inputC, inputH, inputW,
                                                                // kernelN, kernelC, kernelH, kernelW,
                                                                // outputN, outputC, outputH, outputW,
                                                                output_offset, do_bias, do_relu, pad_size, stride_size);

    llvm::dbgs() << "RewriteOnnxQConv2DToArxPattern, conv_op : " << conv_op << "\n";
    
    rewriter.replaceOp(op, conv_op.getResult());

    return mlir::success();
}




} // namespace harx
} // namespace onnx_mlir
