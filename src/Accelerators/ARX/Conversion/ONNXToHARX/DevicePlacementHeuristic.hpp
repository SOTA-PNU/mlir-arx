/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacementHeuristic.hpp - Place ops using model  -------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains heuristics to place operations on CPU or ARX.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_HEURISTICS_H
#define ONNX_MLIR_HEURISTICS_H

#include "mlir/IR/BuiltinOps.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

using OpSetType = mlir::DenseSet<mlir::Operation *>;

/**
 *  Place all ops that qualify for ARX executions on the ARX.
 *
 * @param context Context of the model.
 * @param ops ONNX ops that should be considered in the device assignment.
 * @param cpuOps Set of ops that must execute on CPU.
 */

void PlaceAllLegalOpsOnARX(mlir::MLIRContext *context,
    const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const OpSetType &cpuOps);

/**
 *  Place all ops that qualify for ARX execution on the ARX when the
 * operations are estimated run faster on the ARX.
 *
 * @param context Context of the model.
 * @param ops ONNX ops that should be considered in the device assignment.
 * @param dimAnalysis Pointer to dimension analysis tool for disambiguating
 * dynamic shape dimensions.
 * @param cpuOps Set of ops that must execute on CPU.
 */

void PlaceBeneficialOpsOnARX(mlir::MLIRContext *context,
    const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps);

/**
 * Place all operations that qualify for ARX execution on the ARX when the
 * operations are estimated run faster on the ARX, including the costs of Stick
 * and Unstick necessary for ARX execution. The algorithm starts to place on
 * the CPU/ARX operations that are significantly faster on CPU/ARX. Then it
 * aims to add operations to the ARX when the new operations are faster
 * including the additional (if any) stick/unstick required for these less
 * significantly faster ARX operations. Three factors below can modify the
 * sensitivity at which ops are assigned to the ARX.
 *
 * @param context Context of the model.
 * @param ops ONNX ops that should be considered in the device assignment.
 * @param dimAnalysis Pointer to dimension analysis tool for disambiguating
 * dynamic shape dimensions.
 * @param cpuOps Set of ops that must execute on CPU.
 * @param minFactor ARX (including stick/unstick) has to be at least minFactor
 * times faster than CPU for an op to be assigned to the ARX.
 * @param significantCPUFactor CPU has to be at least significantFactor faster
 * than ARX to seed/force computations on the CPU.
 * @param significantARXFactor ARX has to be at least significantFactor faster
 * than CPU to seed/force computations on the ARX.
 *
 * @note The significantCPUFactor can be smaller, as if it's not looking good
 * for the ARX, we may as well seed the computation on CPU for ops that are
 * much better on the CPU. For significantARXFactor, we may want it much higher
 * as we might want only to send there really beneficial ops on the ARX.
 * Combining a high significantARXFactor with a large minFactor, the heuristic
 * will put only ops that are really beneficial on the ARX.
 */
void PlaceBeneficialOpsOnARXWithStickUnstick(mlir::MLIRContext *context,
    mlir::ModuleOp module, const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps,
    double minFactor = 1.1, double significantCPUFactor = 2.0,
    double significantARXFactor = 3.0);

} // namespace onnx_mlir
#endif
