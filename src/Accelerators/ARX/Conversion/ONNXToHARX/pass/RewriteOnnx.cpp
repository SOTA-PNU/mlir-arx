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
    if (auto module = op->getParentOfType<ModuleOp>()){ 
        module.dump();
    }
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
    auto C_zero_point_value = C_zero_point_ui8.getDefiningOp<ONNXConstantOp>().getValueAttr().cast<DenseIntElementsAttr>().getValues<APInt>()[0].getSExtValue();
    auto C_zero_point_i32 = rewriter.getI32IntegerAttr(C_zero_point_value);

    auto scale3_float32 = onnx_matmul_op.getAScale(); 
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

    // op.getResult().replaceAllUsesWith(matmulOp);
    // customOp.getResult(0).replaceAllUsesWith(matmulOp);
    // rewriter.eraseOp(op);
    // rewriter.eraseOp(customOp);
    // for (auto& addOp : customOp.getResult(0).getUses()) {
    //     auto idx = addOp.getOperandNumber();
    //     addOp.get().setType(matmulOp.getResult().getType());
    // }

    if (auto module = op->getParentOfType<ModuleOp>()){ 
        module.dump();
    }
    llvm::dbgs() << "04 fin : \n";

    return mlir::success();
}

} // namespace harx
} // namespace onnx_mlir
