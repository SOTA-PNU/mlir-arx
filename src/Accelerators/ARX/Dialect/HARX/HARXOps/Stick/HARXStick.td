// SPDX-License-Identifier: Apache-2.0

//===------- Stick.td - Pattern Match for HARXStick ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for HARX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef STICK_TD 
#define STICK_TD 

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

def NoneTypeStickRemovalPattern : Pat<
  (HARXStickOp:$stick $arg, $layout1, $_),
  (replaceWithValue $arg),
  [(IsNoneType:$arg)]
>;

// harx.Stick (harx.Unstick (%X)) = %X
def StickUnstickSameLayoutRemovalPattern : Pat<
  (HARXStickOp:$stick (HARXUnstickOp:$unstick $arg), $_, $_),
  (replaceWithValue $arg),
  [(SameLayout $arg, $stick)]
>;

// harx.Stick (harx.Unstick (%X)) = onnx.LayoutTransform(%X)
// Does not support NHWC layout because onnx.LayoutTransform requires that
// the input and output must have the same shape, but NHWC stickify/unstickify
// transposes the shape.
def StickUnstickDiffLayoutRemovalPattern : Pat<
  (HARXStickOp:$stick (HARXUnstickOp:$unstick $arg), $_, $_),
  (ONNXLayoutTransformOp $arg, (GetEncodingAttr $stick)),
  [(NotSameLayout $arg, $stick), (NoOneIsOfNHWCLayout $arg, $stick),
   // Do not support 1D and 2DS because of this issue: https://github.com/onnx/onnx-mlir/issues/1940
   (NoOneIsOf1DLayout $arg, $stick), (NoOneIsOf2DSLayout $arg, $stick)]
>;

// The pattern
//   harx.Stick (onnx.LeakyRelu (harx.Unstick (%X)))
// can be replaced by
//   harx.Sub (harx.Relu(%X),
//              harx.Relu(harx.Mul(%X, MinusBcastConst(%alpha))))
//
// Constraints:
//   - %X should have static shape, and %alpha should be constant.
//
def ReplaceONNXLeakyReluPattern: Pat<
  (HARXStickOp:$stickout (ONNXLeakyReluOp:$out (HARXUnstickOp $X), $alpha),
                $layout, $_),
  (HARXSubOp
     (HARXReluOp $X, (returnType $X)),
     (HARXReluOp (HARXMulOp $X,
                              (HARXStickOp (GetMinusBcastConst $alpha,
                                             $out),
                                            $layout,
                                            // Donot saturate since input orignally from NNPA
                                            (NoneIntegerAttr)), 
                              (returnType $X)),
                  (returnType $X))),
  [(IsStaticShapeTensor $X), (IsPlusConstantFloat $alpha),
   (SameLayout $X, $stickout)]
>;

// The folowing pattern was found in bertsquad and GPT models.
// ```
// %0 = "harx.Unstick"(%X) {layout = "2D"} : tensor<?x768xf32, #harx.encoding<{dataLayout = "3DS"}>> -> tensor<?x768xf32>
// %1 = "onnx.Reshape"(%0) : tensor<?x768xf32> -> tensor<?x256x12x64xf32>
// %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]}: tensor<?x256x12x64xf32> -> tensor<?x12x256x64xf32>
// %3 = "onnx.Reshape"(%2) tensor<?x12x256x64xf32> -> tensor<?x256x64xf32>
// %4 = "harx.Stick"(%4) {layout = "3DS"} : tensor<?x256x64xf32> -> tensor<?x256x64xf32, #harx.encoding<{dataLayout = "3DS"}>>
// ```
// where
//  - the 1st Reshape can be expressed by an affine map, i.e. (d0, d1) -> (d0/256, d0%256, d1/64, d1%/64)
//  - the Transpose can be expressed by an affine map, i.e. (d0, d1, d2, d3) -> (d0, d2, d1, d3), and
//  - the 2nd Reshape can be expressed by an affine map, i.e. (d0, d1, d2, d3)  -> (d0*12+d1, d2, d3)
// 
// Thus, we will replace Reshape, Transpose by onnx.ShapeTransform operations
// that can be composed automatically into a single onnx.ShapeTransform.
//
// Note: current implementation only supports tensors with static dimensions.

def ReshapeTransposeReshape2DTo3DSPattern : Pat<
  (HARXStickOp:$stick
    (ONNXReshapeOp:$reshape2
      (ONNXTransposeOp:$transpose
        (ONNXReshapeOp:$reshape1
          (HARXUnstickOp:$unstick $X),
          $shape1, $_),
        $perm),
      $shape2, $_),
    $layout3DS, $saturation),
  (HARXStickOp
    (ONNXShapeTransformOp // reshape
      (ONNXShapeTransformOp // transpose
        (ONNXShapeTransformOp // reshape
          (HARXUnstickOp $X),
          (GetTiling2DTo4DMap $reshape1),
          (returnType (GetResultType $reshape1))),
        (GetTransposeMap $perm),
        (returnType (GetResultType $transpose))),
      (GetCollapsing4DTo3DMap $reshape2),
      (returnType (GetResultType $reshape2))),
    $layout3DS, $saturation),
  [(TensorHas2DLayout $X), (Is3DSLayout $layout3DS),
   (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
   (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
   (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
   (IsTiling2DTo4D $reshape1), // 1st reshape is tiling over each input dimension
   (IsCollapsing4DTo3D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
  ]
>;

// Reversed direction of the above pattern.
def ReshapeTransposeReshape3DSTo2DPattern : Pat<
  (HARXStickOp:$stick
    (ONNXReshapeOp:$reshape2
      (ONNXTransposeOp:$transpose
        (ONNXReshapeOp:$reshape1
          (HARXUnstickOp:$unstick $X),
          $shape1, $_),
        $perm),
      $shape2, $_),
    $layout2D, $saturation),
  (HARXStickOp
    (ONNXShapeTransformOp // reshape
      (ONNXShapeTransformOp // transpose
        (ONNXShapeTransformOp // reshape
          (HARXUnstickOp $X),
          (GetTiling3DTo4DMap $reshape1),
          (returnType (GetResultType $reshape1))),
        (GetTransposeMap $perm),
        (returnType (GetResultType $transpose))),
      (GetCollapsing4DTo2DMap $reshape2),
      (returnType (GetResultType $reshape2))),
    $layout2D, $saturation),
  [(TensorHas3DSLayout $X), (Is2DLayout $layout2D),
   (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
   (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
   (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
   (IsTiling3DTo4D $reshape1), // 1st reshape is tiling over each input dimension
   (IsCollapsing4DTo2D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
  ]
>;

#endif // STICK_TD
