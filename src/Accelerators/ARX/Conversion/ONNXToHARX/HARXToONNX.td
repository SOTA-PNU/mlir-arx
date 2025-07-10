// SPDX-License-Identifier: Apache-2.0

//===- HARXToONNX.td - Replacing HARX Ops by ONNX Ops -*- tablegen ------===//
//
// Copyright 2023- The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef HARX_TO_ONNX
#define HARX_TO_ONNX

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/ARX/Dialect/HARX/HARX.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def NotBlockArgument : Constraint<CPred<"!isa<BlockArgument>($_self)">>;
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">>;
def CreateONNXMinOp : NativeCodeCall<"$_builder.create<ONNXMinOp>($_loc, $0.getType(), ValueRange({$1, $2}))">;
def CreateONNXMaxOp : NativeCodeCall<"$_builder.create<ONNXMaxOp>($_loc, $0.getType(), ValueRange({$1, $2}))">;

//===----------------------------------------------------------------------===//
// ONNXAddOp %X = HARXUnstickOp (HARXAddOp (HARXStickOp %X),
// (HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceHARXAddPattern : Pat<
  (HARXUnstickOp (HARXAddOp (HARXStickOp:$s_x $x, $_, $_), (HARXStickOp:$s_y $y, $_, $_))),
  (ONNXAddOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXMulOp %X = HARXUnstickOp (HARXMulOp (HARXStickOp %X),
// (HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceHARXMulPattern : Pat<
  (HARXUnstickOp (HARXMulOp (HARXStickOp:$s_x $x, $_, $_), (HARXStickOp:$s_y $y, $_, $_))),
  (ONNXMulOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = HARXUnstickOp (HARXSubOp (HARXStickOp %X),
// (HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceHARXSubPattern : Pat<
  (HARXUnstickOp (HARXSubOp (HARXStickOp:$s_x $x, $_, $_), (HARXStickOp:$s_y $y, $_, $_))),
  (ONNXSubOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = HARXUnstickOp (HARXDivOp (HARXStickOp
// %X),(HARXStickOp %Y))
// Note: turn off this pattern since ARX is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceHARXDivPattern : Pat<
//   (HARXUnstickOp (HARXDivOp (HARXStickOp:$s_x $x, $_, $_), (HARXStickOp:$s_y $y, $_, $_))),
//   (ONNXDivOp $x, $y),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
// >;

//===----------------------------------------------------------------------===//
// ONNXReluOp %X = HARXUnstickOp (HARXReluOp (HARXStickOp %X))
//===----------------------------------------------------------------------===//
def replaceHARXReluPattern : Pat<
  (HARXUnstickOp (HARXReluOp (HARXStickOp:$s_x $x, $_, $_))),
  (ONNXReluOp $x),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)]
>;

//===----------------------------------------------------------------------===//
// ONNXTanhOp %X = HARXUnstickOp (HARXTanhOp (HARXStickOp %X))
// Note: turn off this pattern since ARX is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceHARXTanhPattern : Pat<
//   (HARXUnstickOp (HARXTanhOp (HARXStickOp:$s_x $x, $_))),
//   (ONNXTanhOp $x),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x)]
// >;

//===----------------------------------------------------------------------===//
// ONNXSigmoidOp %X = HARXUnstickOp (HARXSigmoidOp (HARXStickOp %X))
// Note: turn off this pattern since ARX is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceHARXSigmoidPattern : Pat<
//   (HARXUnstickOp (HARXSigmoidOp (HARXStickOp:$s_x $x, $_))),
//   (ONNXSigmoidOp $x),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x)]
// >;

#endif // HARX_TO_ONNX
