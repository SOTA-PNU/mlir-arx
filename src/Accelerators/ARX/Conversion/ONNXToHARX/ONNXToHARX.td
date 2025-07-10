// SPDX-License-Identifier: Apache-2.0

//===- ONNXToHARX.td - Replacing ONNX Ops by HARX Ops -*- tablegen ------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_TO_ZHIGH
#define ONNX_TO_ZHIGH

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/ARX/Dialect/HARX/HARX.td"
include "src/Accelerators/ARX/Dialect/HARX/HARXOps/OpHelper.td"
include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARXCommon.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def IsEnableScalarBcastBinary: Constraint<CPred<"isEnableScalarBcastBinary()">>;

def IsNotNoneType : Constraint<CPred<"!mlir::isa<NoneType>(($_self).getType())">>;

class HasRankOf<int rank> : Constraint<
  CPred<"mlir::isa<ShapedType>($0.getType()) && "
        "mlir::cast<ShapedType>($0.getType()).hasRank() && "
        "mlir::cast<ShapedType>($0.getType()).getRank() == " # rank>
>;

def IsBiasNoneOr1D : Constraint<
  CPred<"mlir::isa<NoneType>($_self.getType()) || "
        "(mlir::isa<ShapedType>($_self.getType()) && "
	"  mlir::cast<ShapedType>($_self.getType()).hasRank() && "
	"  mlir::cast<ShapedType>($_self.getType()).getRank() == 1)">
>;

class VariadicSizeIs<int N> : Constraint<
  CPred<"$0.size() == " # N>, "The variadic operand's size is N"
>;

class VariadicSizeIsGT<int N> : Constraint<
  CPred<"$0.size() > " # N>, "The variadic operand's size is greater than N"
>;

// Create a unit constant that will be used as none input.
def CreateNoneValue : NativeCodeCall<"$_builder.create<ONNXNoneOp>($_loc)">;

class getNth<int n, string x> : NativeCodeCall<x # "[" # n # "]">;

class GetNthVariadicOperand<int n> : NativeCodeCall<"$0[" # n # "]">;

class getNthVariadicResults<int n, string x> : NativeCodeCall<x # ".getResults()[" # n # "]">;

def GetShape :
	NativeCodeCall<"mlir::cast<ShapedType>($0.getType()).getShape()">;

def GetDynShape:
        NativeCodeCall<"getDynShape($_loc, $_builder, $0)">;

def GetRank :
	NativeCodeCall<"mlir::cast<RankedTensorType>($0.getType()).getRank()">;

def GetSInt :
        NativeCodeCall<"mlir::cast<IntegerAttr>($0).getSInt()">;

class GetStrAttr<string s> :
        NativeCodeCall<"$_builder.getStringAttr(\"" # s # "\")">;

def GetI64Attr :
	NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), $0)">;

class GetI64NAttr<int n> :
	NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), " # n # ")">;

def GetI64ZeroAttr :
        NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), 0)">;

class GetI64ArrayAttr<int n> :
        NativeCodeCall<"$_builder.getI64ArrayAttr(" # n # ")">;

def GetUnrankedTensorTypeOf : NativeCodeCall<
  "UnrankedTensorType::get(mlir::cast<ShapedType>($0.getType()).getElementType())"
>;

class EmitOp3<string s> :
        NativeCodeCall<"$_builder.create<" # s # ">($_loc, $0, $1, $2)">;

class EmitOp4<string s> :
        NativeCodeCall<"$_builder.create<" # s # ">($_loc, $0, $1, $2, $3)">;

def IsF32ScalarConstantTensor: Constraint<
  CPred<"isF32ScalarConstantTensor($0)">,
  "The value is a constant tensor of a single f32 value"
>;

def GetScalarF32AttrFromConstant :
        NativeCodeCall<"getScalarF32AttrFromConstant($0)">;

//===----------------------------------------------------------------------===//
// ONNXReluOp %X = HARXUnstickOp (HARXReluOp (HARXStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXReluPattern : Pat<
  (ONNXReluOp $x),
  (HARXUnstickOp (HARXReluOp (HARXStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                               (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXTanhOp %X = HARXUnstickOp (HARXTanhOp (HARXStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXTanhPattern : Pat<
  (ONNXTanhOp $x),
  (HARXUnstickOp (HARXTanhOp (HARXStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                               (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXAddOp %X = HARXUnstickOp (HARXAddOp (HARXStickOp %X),
// (HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXAddPattern : Pat<
  (ONNXAddOp $x, $y),
  (HARXUnstickOp (HARXAddOp (HARXStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (HARXStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x))
)
>;

//===----------------------------------------------------------------------===//
//
// ONNXSumOp %X = HARXUnstickOp (HARXAddOp (HARXStickOp %X_0),
//                                           (HARXStickOp %X_1))
//
//===----------------------------------------------------------------------===//
def GetONNXSumOpWithoutFirst : NativeCodeCall<
  "$_builder.create<ONNXSumOp>("
  "  $_loc,"
  "  mlir::cast<ShapedType>($0[0].getType()),"
  "  ValueRange(OperandRange{$0.begin() + 1, $0.end()}))"
>;

// Recursive case: ONNXSumOp (N inputs) = ONNXSumOp (N-1 inputs) + HARXAddOp.
def replaceONNXSumOpPatternRecursion : Pat<
  (ONNXSumOp $x),
  (HARXUnstickOp
     (HARXAddOp
        (HARXStickOp:$s_x (GetNthVariadicOperand<0> $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (HARXStickOp (GetONNXSumOpWithoutFirst $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (returnType $s_x))),
  [(VariadicSizeIsGT<2> $x)], [], (addBenefit 1)
>;

// Singleton case: ONNXSumOp (2 inputs) to HARXAddOp.
def replaceONNXSumOpPatternSingleton : Pat<
  (ONNXSumOp $x),
  (HARXUnstickOp
     (HARXAddOp
        (HARXStickOp:$s_x (GetNthVariadicOperand<0> $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (HARXStickOp (GetNthVariadicOperand<1> $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (returnType $s_x))),
  [(VariadicSizeIs<2> $x)], [], (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = HARXUnstickOp (HARXSubOp (HARXStickOp %X),
// (HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXSubPattern : Pat<
  (ONNXSubOp $x, $y),
  (HARXUnstickOp (HARXSubOp (HARXStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (HARXStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXMulOp %X = HARXUnstickOp (HARXMulOp (HARXStickOp %X),
// (HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXMulPattern : Pat<
  (ONNXMulOp $x, $y),
  (HARXUnstickOp (HARXMulOp (HARXStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (HARXStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = HARXUnstickOp (HARXDivOp (HARXStickOp
// %X),(HARXStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXDivPattern : Pat<
  (ONNXDivOp $x, $y),
  (HARXUnstickOp (HARXDivOp (HARXStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (HARXStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x))),
  [], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXSoftmaxOp %X = ONNXSqueezeOp
//                        (HARXUnstickOp
//                            (HARXSoftmaxOp
//                                (HARXStickOp (ONNXUnsqueezeOp %X, 0), (GetDefaultSaturation)),
//                                0))
//===----------------------------------------------------------------------===//

def replaceONNXSoftmax2DPattern : Pattern<
  (ONNXSoftmaxOp:$res $x, $axis),
  [
   // Get unstick result
   (HARXUnstickOp:$unstick
       (HARXSoftmaxOp
           (HARXStickOp:$s_x
               (EmitOp3<"ONNXUnsqueezeV11Op">
                   (GetUnrankedTensorTypeOf $x),
                   $x,
                   (GetI64ArrayAttr<0>)),
               (_3DSLayoutAttr),
               (GetDefaultSaturation)),
           (GetStringAttr<"ACT_NONE">),
           (returnType $s_x))),

   (EmitOp3<"ONNXSqueezeV11Op">
       (GetTypeOf $res),
       $unstick,
       (GetI64ArrayAttr<0>))
  ],
  [(HasRankOf<2> $x)]
>;

def replaceONNXSoftmax3DPattern : Pat<
  (ONNXSoftmaxOp:$res $x, $axis),
  // Get unstick result
  (HARXUnstickOp:$unstick
     (HARXSoftmaxOp
        (HARXStickOp:$s_x $x, (_3DSLayoutAttr), (GetDefaultSaturation)),
        (GetStringAttr<"ACT_NONE">),
        (returnType $s_x))),
  [(HasRankOf<3> $x)]
>;

//===----------------------------------------------------------------------===//
// ONNXLogSoftmaxOp (ONNXSoftmaxOp %X) =
//   ONNXSqueezeOp
//     (HARXUnstickOp
//         (HARXSoftmaxOp
//            (HARXStickOp (ONNXUnsqueezeOp %X, 0), (GetDefaultSaturation)),
//            "ACT_LOG"),
//     0)
//===----------------------------------------------------------------------===//
def IsSoftmaxLegalForZDNN: Constraint<
  CPred<"isSuitableForZDNN<ONNXSoftmaxOp>(" #
             "dyn_cast_or_null<ONNXSoftmaxOp>($0.getDefiningOp()))">,
  "Softmax is legal for zDNN"
>;
def replaceONNXLogSoftmaxPattern : Pattern<
  (ONNXLogOp:$res (ONNXSoftmaxOp:$s $x, $axis)),
  [
   // Get unstick result
   (HARXUnstickOp:$unstick
       (HARXSoftmaxOp
           (HARXStickOp:$s_x
               (EmitOp3<"ONNXUnsqueezeV11Op">
                   (GetUnrankedTensorTypeOf $x),
                   $x,
                   (GetI64ArrayAttr<0>)),
               (_3DSLayoutAttr),
               (GetDefaultSaturation)),
           (GetStringAttr<"ACT_LOG">),
           (returnType $s_x))),

   (EmitOp3<"ONNXSqueezeV11Op">
       (GetTypeOf $res),
       $unstick,
       (GetI64ArrayAttr<0>))
  ],
  [(IsSoftmaxLegalForZDNN $s)]
>;

//===----------------------------------------------------------------------===//
// ONNXMaxPoolSingleOutOp %X =
//                 (HARXUnstickOp
//                     (HARXMaxPoolOp
//                         (HARXStickOp %X),
//                         $krnel_shape,
//                         $strides,
//		           $padding_type))
//===----------------------------------------------------------------------===//

def GetStrAttrPaddingtypeMaxPool : NativeCodeCall<
  "$_builder.getStringAttr(getStrPaddingType<ONNXMaxPoolSingleOutOp,"
  "ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>"
  "($0.getDefiningOp<ONNXMaxPoolSingleOutOp>()))">;

def GetI64ArrayAttrKernelShapeMaxPool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayKernelShape<ONNXMaxPoolSingleOutOp,"
  "ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>"
  "($0.getDefiningOp<ONNXMaxPoolSingleOutOp>()))">;

def GetI64ArrayAttrStridesMaxPool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayStrides<ONNXMaxPoolSingleOutOp,"
  "ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>"
  "($0.getDefiningOp<ONNXMaxPoolSingleOutOp>()))">;

def replaceONNXMaxPoolSingleOutPattern : Pattern<
  (ONNXMaxPoolSingleOutOp:$res $x, $_, $_, $_, $_, $_, $_, $_),
  [
    // Get attributes using shape helper
    (GetStrAttrPaddingtypeMaxPool:$padtype $res),
    (GetI64ArrayAttrKernelShapeMaxPool:$kernel_shape $res),
    (GetI64ArrayAttrStridesMaxPool:$strides $res),

    (HARXUnstickOp
       (HARXMaxPool2DOp
          (HARXStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
          $kernel_shape,
          $strides,
          $padtype))
  ]
>;

//===----------------------------------------------------------------------===//
// ONNXAveragePoolOp %X =
//                 (HARXUnstickOp
//                     (HARXAvgPoolOp
//                         (HARXStickOp %X),
//                         $krnel_shape,
//                         $strides,
//		           $padding_type))
//===----------------------------------------------------------------------===//

def GetStrAttrPaddingtypeAveragePool : NativeCodeCall<
  "$_builder.getStringAttr(getStrPaddingType<ONNXAveragePoolOp,"
  "ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>"
  "($0.getDefiningOp<ONNXAveragePoolOp>()))">;

def GetI64ArrayAttrKernelShapeAveragePool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayKernelShape<ONNXAveragePoolOp,"
  "ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>"
  "($0.getDefiningOp<ONNXAveragePoolOp>()))">;

def GetI64ArrayAttrStridesAveragePool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayStrides<ONNXAveragePoolOp,"
  "ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>"
  "($0.getDefiningOp<ONNXAveragePoolOp>()))">;

def replaceONNXAveragePoolPattern : Pattern<
  (ONNXAveragePoolOp:$res $x, $_, $_, $_, $_, $_, $_, $_),
  [
    // Get attributes using shape helper
    (GetStrAttrPaddingtypeAveragePool:$padtype $res),
    (GetI64ArrayAttrKernelShapeAveragePool:$kernel_shape $res),
    (GetI64ArrayAttrStridesAveragePool:$strides $res),

    (HARXUnstickOp
       (HARXAvgPool2DOp
          (HARXStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
          $kernel_shape,
          $strides,
          $padtype))
  ]
>;

//===----------------------------------------------------------------------===//
// MatMul
//===----------------------------------------------------------------------===//

def GetMatMulLayoutStringAttr : NativeCodeCall<
  "$_builder.getStringAttr(($0 == 2) ? LAYOUT_2D : LAYOUT_3DS)"
>;

def GetMatMulBiasLayoutStringAttr : NativeCodeCall<
  "$_builder.getStringAttr((($0 == 3) && ($1 == 3)) ? LAYOUT_2DS : LAYOUT_1D)"
>;

//===----------------------------------------------------------------------===//
// ONNXMatMulOp %X %Y = HARXUnstickOp
//                          (HARXMatMulOp
//                              (HARXStickOp %X),
//                              (HARXStickOp %Y),
//                              CreateNoneValue)
//===----------------------------------------------------------------------===//

def replaceONNXMatMulPattern : Pat<
  (ONNXMatMulOp:$res $x, $y),
  (HARXUnstickOp
     (HARXMatMulOp
        (HARXStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (HARXStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (CreateNoneValue)))
>;

//===----------------------------------------------------------------------===//
// ONNXAddOp (ONNXMatMulOp %X %Y), %B =
//   HARXUnstickOp
//       (HARXMatMulOp
//           (HARXStickOp %X),
//           (HARXStickOp %Y),
//           (HARXStickOp %B))
//===----------------------------------------------------------------------===//

def IsMatMulLegalForZDNN: Constraint<
  CPred<"isSuitableForZDNN<ONNXMatMulOp>(" #
             "dyn_cast_or_null<ONNXMatMulOp>($0.getDefiningOp()))">,
  "MatMul is legal for zDNN"
>;

// Be careful, this check is very specific to '$0' of rank 2 and '$1' of rank 1.
def HaveSameLastDimR2R1: Constraint<
  CPred<"(!mlir::cast<ShapedType>($0.getType()).isDynamicDim(1))" #
        " && (!mlir::cast<ShapedType>($1.getType()).isDynamicDim(0))" #
        " && (mlir::cast<ShapedType>($0.getType()).getShape()[1]" #
              " == mlir::cast<ShapedType>($1.getType()).getShape()[0])">,
  "Have the same last dimension"
>;

// Only 1D bias is suitable for this transformation since only then
// the semantics of bias addition is the same for both ONNX and zDNN.
def replaceONNXMatMulAddPattern1 : Pat<
  // From Add $b, (MatMul $x, $y)
  (ONNXAddOp $b, (ONNXMatMulOp:$m $x, $y)),
  // To HARXMatMulOp
  (HARXUnstickOp
     (HARXMatMulOp
        (HARXStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (HARXStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (HARXStickOp $b, (GetMatMulBiasLayoutStringAttr (GetRank $x),
                                                         (GetRank $y)), (GetDefaultSaturation)))),
  [(IsMatMulLegalForZDNN $m), (HasRankOf<2> $y), (HasRankOf<1> $b),
   (HaveSameLastDimR2R1 $y, $b)], [],
  (addBenefit 0)
>;

def replaceONNXMatMulAddPattern2 : Pat<
  // From Add (MatMul $x, $y), $b
  (ONNXAddOp (ONNXMatMulOp:$m $x, $y), $b),
  // To HARXMatMulOp
  (HARXUnstickOp
     (HARXMatMulOp
        (HARXStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (HARXStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (HARXStickOp $b, (GetMatMulBiasLayoutStringAttr (GetRank $x),
                                                         (GetRank $y)), (GetDefaultSaturation)))),
  [(IsMatMulLegalForZDNN $m), (HasRankOf<2> $y), (HasRankOf<1> $b),
   (HaveSameLastDimR2R1 $y, $b)], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// GEMM
//===----------------------------------------------------------------------===//
def IsTransposed: Constraint<CPred<"(mlir::cast<IntegerAttr>($_self).getSInt() == 1)">>;

def Transpose2D: NativeCodeCall<
  "emitONNXTranspose($_loc, $_builder, $0, SmallVector<int64_t, 2>({1, 0}))">;

//===----------------------------------------------------------------------===//
// ARX does not directly support cases with alpha != 1.0 or beta != 1.0 or bias's rank == 2.
// Since alpha and beta were legalized by the pass configuration, we check rank only.
//
// If bias's rank is 1, directly lower ONNX GEMM to HARX MatMul.
// ONNXGemmOp %A %B %C = HARXUnstickOp
//                         (HARXMatMulOp
//                            (HARXStickOp %A),
//                            (HARXStickOp %B),
//                            (HARXStickOp %C)),
//
// If bias's rank is 2, lower ONNX GEMM to HARX MatMul and ONNX Add.
// ONNX Add could be lowered to HARX depending on the rule for ONNX Add.
// ONNXGemmOp %A %B %C = ONNXAdd (HARXUnstickOp
//                                  (HARXMatMulOp
//                                     (HARXStickOp %A),
//                                     (HARXStickOp %B),
//                                     (NoValue)),
//                                %C)
//===----------------------------------------------------------------------===//

def normalizeONNXGemmTransAPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $alpha, $beta, $transA, $transB),
  (ONNXGemmOp (Transpose2D $a), $b, $c, $alpha, $beta, (GetZeroI64Attr), $transB),
  [(IsTransposed:$transA)], [],
  (addBenefit 1)
>;

def normalizeONNXGemmTransBPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $alpha, $beta, $transA, $transB),
  (ONNXGemmOp $a, (Transpose2D $b), $c, $alpha, $beta, $transA, (GetZeroI64Attr)),
  [(IsTransposed:$transB)], [],
  (addBenefit 1)
>;


def replaceONNXGemmBiasNoneOr1DPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $_, $_, $_, $_),
  (HARXUnstickOp
     (HARXMatMulOp
        (HARXStickOp $a, (_2DLayoutAttr), (GetDefaultSaturation)),
        (HARXStickOp $b, (_2DLayoutAttr), (GetDefaultSaturation)),
        (HARXStickOp $c, (_1DLayoutAttr), (GetDefaultSaturation)))),
  [(IsBiasNoneOr1D:$c)], [],
  (addBenefit 0)
>;

def replaceONNXGemmBias2DPattern : Pat<
  (ONNXGemmOp:$res $a, $b, $c, $_, $_, $_, $_),
  (ONNXAddOp
    (HARXUnstickOp
       (HARXMatMulOp
          (HARXStickOp $a, (_2DLayoutAttr), (GetDefaultSaturation)),
          (HARXStickOp $b, (_2DLayoutAttr), (GetDefaultSaturation)),
          (CreateNoneValue)),
       (returnType $res)),
    $c),
  [(HasRankOf<2> $c)], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// LSTM
//===----------------------------------------------------------------------===//

def GetLSTMGRUBiasSplitShape : NativeCodeCall<
  "getLSTMGRUBiasSplitShape($_loc, $_builder, $0)"
>;

class GetLSTMGRUZDNNWeightFromONNXWeight<string opName> : NativeCodeCall<
  "getLSTMGRUZDNNWeightFromONNXWeight($_loc, $_builder, $0,"
  "!strcmp(\"" # opName # "\", \"LSTM\"))">;

def GetLSTMGRUReturnAllStepsAttr : NativeCodeCall<
  "$_builder.getIntegerAttr($_builder.getIntegerType(64, true),"
  "(isNoneValue($0) ? 1 : -1))">;

def GetLSTMGRUGetY :
        NativeCodeCall<"getLSTMGRUGetY($_loc, $_builder, $0, $1)">;

def GetLSTMGRUGetYh :
        NativeCodeCall<"getLSTMGRUGetYh($_loc, $_builder, $0, $1, $2, $3, $4)">;

def GetLSTMGRUGetYWithSequenceLens :
        NativeCodeCall<"getLSTMGRUGetYWithSequenceLens($_loc, $_builder, $0, $1, $2, $3)">;

def GetLSTMGRUGetYhWithSequenceLens :
        NativeCodeCall<"getLSTMGRUGetYhWithSequenceLens($_loc, $_builder, $0, $1, $2, $3, $4, $5)">;

def GetLSTMGRUGetYc :
        NativeCodeCall<"getLSTMGRUGetYc($_loc, $_builder, $0, $1)">;

def EmitONNXSplitOp6Results :
        NativeCodeCall<"emitONNXSplitOp($_loc, $_builder, $0, $1, $2)", 6>;

def EmitONNXSplitOp8Results :
        NativeCodeCall<"emitONNXSplitOp($_loc, $_builder, $0, $1, $2)", 8>;

//===----------------------------------------------------------------------===//
// Rewrite
//
// ONNXConvOp %X, %W, %B
//
// to
//
// (HARXUnstickOp
//    (HARXConvOp
//      (HARXStickOp %X, "NHWC"),
//      (HARXStickOp (NCHWtoHWCK %W), "HWCK"),
//      (HARXStickOp %B, "1D"),
//      kernel_shape,
//      strides,
//      GetPaddingType,
//      ACT_NONE)))
//
//===----------------------------------------------------------------------===//

def GetStrAttrPaddingtypeConv: NativeCodeCall<
  "$_builder.getStringAttr(getStrPaddingType<ONNXConvOp, ONNXConvOpAdaptor,"
  "ONNXConvOpShapeHelper>($0.getDefiningOp<ONNXConvOp>()))">;

def GetI64ArrayAttrKernelShapeConv: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayKernelShape<ONNXConvOp, ONNXConvOpAdaptor,"
  "ONNXConvOpShapeHelper>($0.getDefiningOp<ONNXConvOp>()))">;

def GetI64ArrayAttrStridesConv: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayStrides<ONNXConvOp, ONNXConvOpAdaptor,"
  "ONNXConvOpShapeHelper>($0.getDefiningOp<ONNXConvOp>()))">;

def replaceONNXConv2DPattern : Pattern<
  (ONNXConvOp:$res $x, $w, $b, $_, $_, $_, $_, $_, $_),
  [
   // Get attributes using shape helper
   (GetStrAttrPaddingtypeConv:$padtype $res),
   (GetI64ArrayAttrKernelShapeConv:$kernel_shape $res),
   (GetI64ArrayAttrStridesConv:$strides $res),

   (HARXUnstickOp
    (HARXConv2DOp
       (HARXStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
       (HARXStickOp (NCHWtoHWCK $w), (HWCKLayoutAttr), (GetDefaultSaturation)),
       (HARXStickOp $b, (_1DLayoutAttr), (GetDefaultSaturation)),
       $kernel_shape,
       $strides,
       $padtype,
       (ACT_NONEAttr)))
  ]
>;

//===----------------------------------------------------------------------===//
// Rewrite
//
// ONNXReluOp (ONNXConvOp %X, %W, %B)
//
// to
//
// (HARXUnstickOp
//    (HARXConvOp
//      (HARXStickOp %X, "NHWC"),
//      (HARXStickOp (NCHWtoHWCK %W), "HWCK"),
//      (HARXStickOp %B, "1D"),
//      kernel_shape,
//      strides,
//      GetPaddingType,
//      ACT_RELU)))
//
//===----------------------------------------------------------------------===//

def IsConv2DLegalForZDNN: Constraint<
  CPred<"isSuitableForZDNN<ONNXConvOp>(" #
             "dyn_cast_or_null<ONNXConvOp>($0.getDefiningOp()))">,
  "Conv is legal for zDNN"
>;

def replaceONNXReluConvPattern : Pattern<
  (ONNXReluOp (ONNXConvOp:$res $x, $w, $b, $_, $_, $_, $_, $_, $_)),
  [
   // Get attributes using shape helper
   (GetStrAttrPaddingtypeConv:$padtype $res),
   (GetI64ArrayAttrKernelShapeConv:$kernel_shape $res),
   (GetI64ArrayAttrStridesConv:$strides $res),

   (HARXUnstickOp
    (HARXConv2DOp
       (HARXStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
       (HARXStickOp (NCHWtoHWCK $w), (HWCKLayoutAttr), (GetDefaultSaturation)),
       (HARXStickOp $b, (_1DLayoutAttr), (GetDefaultSaturation)),
       $kernel_shape,
       $strides,
       $padtype,
       (ACT_RELUAttr)))
  ],
  [(IsConv2DLegalForZDNN $res)], [],
  (addBenefit 0)
>;

#endif // ONNX_TO_ZHIGH
