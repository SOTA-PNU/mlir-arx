// SPDX-License-Identifier: Apache-2.0

//===------- HARXUnstick.td - Pattern Match for HARXUnstick -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for HARX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef UNSTICK_TD 
#define UNSTICK_TD 

#ifndef OP_BASE
include "src/Accelerators/ARX/Dialect/HARX/HARX.td"
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// DRR patterns 
//===----------------------------------------------------------------------===//

// harx.Unstick (harx.Stick (%X)) = %X
def UnstickStickRemovalPattern : Pat<
  (HARXUnstickOp (HARXStickOp $arg, $_, $_)),
  (replaceWithValue $arg)
>;

// onnx.Dim (harx.Unstick (%X)) = onnx.Dim %X
// For NHWC layout.
def DimUnstickNHWCRemovalPattern : Pat<
  (ONNXDimOp (HARXUnstickOp $arg), $axis),
  (ONNXDimOp $arg, (GetAxisNHWC $axis)),
  [(TensorHasNHWCLayout $arg), (HasARXUse $arg)]
>;

// onnx.Dim (harx.Unstick (%X)) = onnx.Dim %X
// For non-NHWC layout.
def DimUnstickRemovalPattern : Pat<
  (ONNXDimOp (HARXUnstickOp $arg), $axis),
  (ONNXDimOp $arg, $axis),
  [(TensorHasNoNHWCLayout $arg), (HasARXUse $arg)]
>;

#endif // UNSTICK_TD
