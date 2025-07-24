// SPDX-License-Identifier: Apache-2.0

//===----- HARX.td -- HARX Dialect Operation Definitions -*- tablegen ----==//
//
// Copyright 2019-2020 The IBM Research Authors
//
// =============================================================================
//
// Defines HARX Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef HARX_OPS
#define HARX_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/IR/AttrBase.td"


//===----------------------------------------------------------------------===//
// HARX Dialect
//===----------------------------------------------------------------------===//

def HARX_Dialect : Dialect {
  let name = "harx";
  let summary = "A high-level dialect for the ONNX hard accelerator ISA.";
  let cppNamespace = "::onnx_mlir::harx";
  // let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// HARX Attribute
//===----------------------------------------------------------------------===//

// All of the Tensor attributes will extend this class.
class HARX_Attr<string name, list<Trait> traits = []>
	: BaseLayoutAttr<HARX_Dialect, name, traits>;

//===----------------------------------------------------------------------===//
// HARX Operations
//===----------------------------------------------------------------------===//

// Op has the same operand and result layout.
def SameOperandsAndResultLayout: NativeOpTrait<"SameOperandsAndResultLayout">;

//===----------------------------------------------------------------------===//
// Base class for HARX dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.

class HARX_Op<string mnemonic, list<Trait> traits = []> :
    Op<HARX_Dialect, mnemonic, traits>;

def HARXConstantOp : HARX_Op<"Constant", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
      
  let summary = "Constant tensor operation";
  let description = [{
    The constant operation produces a tensor with the same shape and values
    as the DenseElementsAttr attribute 'value'.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results   = (outs TensorOf<[UI8]>:$result);

  // let builders = [OpBuilder<(ins TensorOf<[UI8]>:$value)>];

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXConstantOp::getShapeHelper(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb,
        onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}


def HARXAddOp : HARX_Op<"Add", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Add operation";
  let description = [{
    HARX operation to perform an Add.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$X,
                       TensorOf<[UI8]>:$Y,
                       OptionalAttr<StrAttr>:$node_name,
                       OptionalAttr<F32Attr>:$x_scale,
                       OptionalAttr<F32Attr>:$x_offset,
                       OptionalAttr<F32Attr>:$y_scale,
                       OptionalAttr<F32Attr>:$y_offset,
                       OptionalAttr<F32Attr>:$out_scale,
                       OptionalAttr<F32Attr>:$out_offset);

  let results   = (outs TensorOf<[UI8]>:$Out);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXAddOp::getShapeHelper(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb,
        onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

// def HARXStickOp:HARX_Op<"Stick", [Pure,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Stick operation";
//   let description = [{
//     HARX operation to perform a Stick."

//     If `layout`=`NHWC`, input must be in `NCHW` and output will be in `NHWC`.

//     Optional `saturation` indicates whether the CPU tensor is saturated before stickification
//     or not. If it is saturated, the dlfloat16 range would be used.
//     Saturation if off if `saturation == 0` or it is not given. Otherwise, it is on.
//   }];
//   let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$In,
//                        OptionalAttr<StrAttr>:$layout,
//                        OptionalAttr<SI64Attr>:$saturation);
//   let results = (outs AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
//                                  ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
//                                  ZTensor_NHWC,  ZTensor_NCHW, ZTensor_HWCK,
//                                  NoneType]>:$Out);
//   let builders = [
//     OpBuilder<(ins "::mlir::Value":$In, "::mlir::StringAttr":$layout), [{
//       build($_builder, $_state, In, layout, IntegerAttr());
//     }]>,
//     OpBuilder<(ins "::mlir::Value":$In, "::mlir::StringAttr":$layout,
//                "::mlir::IntegerAttr":$saturation)>
//   ];
//   let hasCanonicalizer = 1;
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXStickOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXStickOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXUnstickOp:HARX_Op<"Unstick", [Pure,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Unstick operation";
//   let description = [{
//     HARX operation to perform a Unstick.
//   }];
//   let arguments = (ins AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
//                                  ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
//                                  ZTensor_NHWC, ZTensor_NCHW, ZTensor_HWCK]>:$In);
//   let results = (outs TensorOf<[F32]>:$Out);
//   let builders = [
//     OpBuilder<(ins "::mlir::Value":$In)>,
//   ];
//   let hasCanonicalizer = 1;
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXUnstickOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnstickOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }


// def HARXAddOp:HARX_Op<"Add", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Add operation";
//   let description = [{
//     HARX operation to perform an Add.
//     This operation does not support broadcasting.
//   }];
//   let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
//                        AnyTypeOf<[AnyZTensor]>:$Y);
//   let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXAddOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }




// def HARXSubOp:HARX_Op<"Sub", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Sub operation";
//   let description = [{
//     HARX operation to perform a Sub.
//     This operation does not support broadcasting.
//   }];
//   let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
//                        AnyTypeOf<[AnyZTensor]>:$Y);
//   let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXSubOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXMulOp:HARX_Op<"Mul", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Mul operation";
//   let description = [{
//     HARX operation to perform a Mul.
//     This operation does not support broadcasting.
//   }];
//   let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
//                        AnyTypeOf<[AnyZTensor]>:$Y);
//   let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXMulOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXDivOp:HARX_Op<"Div", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Div operation";
//   let description = [{
//     HARX operation to perform a Div.
//     This operation does not support broadcasting.
//   }];
//   let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
//                        AnyTypeOf<[AnyZTensor]>:$Y);
//   let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXDivOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXReluOp:HARX_Op<"Relu", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Relu operation";
//   let description = [{
//     "HARX operation to perform a Relu."
//   }];
//   let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
//   let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXReluOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXTanhOp:HARX_Op<"Tanh", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Tanh operation";
//   let description = [{
//     HARX operation to perform a Tanh.
//   }];
//   let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
//   let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXTanhOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXSoftmaxOp:HARX_Op<"Softmax", [Pure, SameOperandsAndResultLayout,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX Softmax operation";
//   let description = [{
//     HARX operation to perform a Softmax.
//     act_func: ACT_NONE or ACT_LOG.
//   }];
//   let arguments = (ins ZTensor_3DS:$X,
//                        DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
//   let results = (outs ZTensor_3DS:$Out);
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXSoftmaxOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnaryOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXMaxPool2DOp:HARX_Op<"MaxPool2D", [Pure,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX 2D max pooling operation";
//   let description = [{
//     HARX operation to perform 2D max pooling.
//   }];
//   let arguments = (ins ZTensor_NHWC:$input,
//                        I64ArrayAttr:$kernel_shape,
//                        I64ArrayAttr:$strides,
//                        DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
//   );
//   let results = (outs ZTensor_NHWC:$output);
//   let builders = [
//     OpBuilder<(ins "::mlir::Value":$input, "::mlir::ArrayAttr":$kernel_shape,
//                    "::mlir::ArrayAttr":$strides, "::mlir::StringAttr":$padding_type), [{
//       Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
//       UnrankedTensorType resType = UnrankedTensorType::get(elementType);
//       build($_builder, $_state, resType, input, kernel_shape, strides, padding_type);
//     }]>
//   ];
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXMaxPool2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXPoolingOpShapeHelper<HARXMaxPool2DOp>(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXAvgPool2DOp:HARX_Op<"AvgPool2D", [Pure,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX 2D average pooling operation";
//   let description = [{
//     HARX operation to perform 2D average pooling.
//   }];
//   let arguments = (ins ZTensor_NHWC:$input,
//                        I64ArrayAttr:$kernel_shape,
//                        I64ArrayAttr:$strides,
//                        DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
//   );
//   let results = (outs ZTensor_NHWC:$output);
//   let builders = [
//     OpBuilder<(ins "::mlir::Value":$input, "::mlir::ArrayAttr":$kernel_shape,
//                    "::mlir::ArrayAttr":$strides, "::mlir::StringAttr":$padding_type), [{
//       Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
//       UnrankedTensorType resType = UnrankedTensorType::get(elementType);
//       build($_builder, $_state, resType, input, kernel_shape, strides, padding_type);
//     }]>
//   ];
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXAvgPool2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXPoolingOpShapeHelper<HARXAvgPool2DOp>(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXMatMulOp:HARX_Op<"MatMul", [Pure,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX MatMul operation";
//   let description = [{
//     HARX operation to perform a MatMul.
//   }];
//   let arguments = (ins AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$X,
//                        AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$Y,
//                        AnyTypeOf<[ZTensor_1D, ZTensor_2DS, NoneType]>:$B);

//   let results = (outs AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$Out);
//   let builders = [
//     OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Y, "::mlir::Value":$B), [{
//       Type elementType = mlir::cast<ShapedType>(X.getType()).getElementType();
//       UnrankedTensorType resType = UnrankedTensorType::get(elementType);
//       build($_builder, $_state, resType, X, Y, B);
//     }]>
//   ];
//   let hasVerifier = 1;
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXMatMulOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXMatMulOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

// def HARXConv2DOp:HARX_Op<"Conv2D", [Pure,
//     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
//   let summary = "HARX 2D convolution operation";
//   let description = [{
//     HARX operation to perform 2D convolution.
//     * input: `[num_batches, height_in, width_in, channels_in]`
//     * input_kernel: `[kernel_height, kernel_width, channels_in, channels_out]`
//     * input_bias: `[channels_out] `
//     * kernel_shape: 1D array of kernel height and width
//     * strides: 1D array of stride height and width
//     * padding_type: SAME_PADDING or VALID_PADDING
//     * act_func: ACT_NONE or ACT_RELU
//     * output: `[num_batches, height_out, width_out, channels_out]`
//   }];
//   let arguments = (ins ZTensor_NHWC:$input,
//                        ZTensor_HWCK:$input_kernel,
//                        AnyTypeOf<[ZTensor_1D, NoneType]>:$input_bias,
//                        I64ArrayAttr:$kernel_shape,
//                        I64ArrayAttr:$strides,
//                        DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type,
//                        DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func
//   );
//   let results = (outs ZTensor_NHWC:$output);
//   let builders = [
//     OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$input_kernel, "::mlir::Value":$input_bias,
//                    "::mlir::ArrayAttr":$kernel_shape, "::mlir::ArrayAttr":$strides,
//                    "::mlir::StringAttr":$padding_type, "::mlir::StringAttr":$act_func), [{
//       Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
//       UnrankedTensorType resType = UnrankedTensorType::get(elementType);
//       build($_builder, $_state, resType, input, input_kernel, input_bias,
//             kernel_shape, strides, padding_type, act_func);
//     }]>
//   ];
//   let hasVerifier = 1;
//   let extraClassDefinition = [{
//     onnx_mlir::ONNXOpShapeHelper * HARXConv2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
//         onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
//       onnx_mlir::ONNXOpShapeHelper *sh = new HARXConv2DOpShapeHelper(op, oper, ieb, scope);
//       assert(sh && "failed to allocate shape helper");
//       return sh;
//     }
//   }];
// }

#endif // HARX_OPS
