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
  let results   = (outs TensorOf<[UI8, I32, SI32]>:$result);

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
                       F32Attr:$x_scale,
                       I32Attr:$x_offset,
                       F32Attr:$y_scale,
                       I32Attr:$y_offset,
                       F32Attr:$out_scale,
                       I32Attr:$out_offset);

  let results   = (outs TensorOf<[UI8]>:$output);

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


def HARXReluOp : HARX_Op<"Relu", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Relu operation";
  let description = [{
    HARX operation to perform a Relu.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$X,
                       TensorOf<[UI8]>:$Y,
                       I32Attr:$offset);

  let results   = (outs TensorOf<[UI8]>:$Out);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXReluOp::getShapeHelper(mlir::Operation *op,
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


def HARXQuantizationOp : HARX_Op<"Quantization", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Quantization operation";
  let description = [{
    HARX operation to perform quantization.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[F32]>:$input,
                       F32Attr:$scale,
                       UI8Attr:$offset);

  let results   = (outs TensorOf<[UI8]>:$output);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXQuantizationOp::getShapeHelper(mlir::Operation *op,
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

def HARXDequantizationOp : HARX_Op<"Dequantization", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Dequantization operation";
  let description = [{
    HARX operation to perform dequantization.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$input,
                       F32Attr:$scale,
                       UI8Attr:$offset,
                       I32Attr:$channel_size,
                       UI8Attr:$use_stride);

  let results   = (outs TensorOf<[F32]>:$output);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXDequantizationOp::getShapeHelper(mlir::Operation *op,
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

def HARXConvolutionShiftOp : HARX_Op<"ConvolutionShift", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX ConvolutionShift operation";
  let description = [{
    HARX operation to perform convolution with shift.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$input,
                       TensorOf<[UI8]>:$kernel,
                       TensorOf<[I32, SI32]>:$output_shift,
                       TensorOf<[I32, SI32]>:$bias,
                       UI8Attr:$output_offset,
                       UI8Attr:$do_bias,
                       UI8Attr:$do_relu,
                       UI8Attr:$pad_size,
                       UI8Attr:$stride_size);

  let results   = (outs TensorOf<[UI8]>:$output);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXConvolutionShiftOp::getShapeHelper(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb,
        onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new HARXConv2DOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXMatMulOp : HARX_Op<"MatMul", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX MatMul operation";
  let description = [{
    HARX operation to perform an MatMul.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$input,
                       TensorOf<[UI8]>:$weight,
                       TensorOf<[I32, SI32]>:$bias,
                       I32Attr:$output_offset,
                       UI8Attr:$do_bias,
                       TensorOf<[I32, SI32]>:$shift);

  let results   = (outs TensorOf<[UI8]>:$output);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXMatMulOp::getShapeHelper(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb,
        onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new HARXMatMulOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXTransposeOp : HARX_Op<"Transpose", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Transpose operation";
  let description = [{
    HARX operation to perform an Transpose.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$input,
                       TensorOf<[I32, SI32]>:$sequence, 
                       TensorOf<[I32, SI32]>:$reshape);

  let results   = (outs TensorOf<[UI8]>:$output);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXTransposeOp::getShapeHelper(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb,
        onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new HARXMatMulOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXMaxPoolOp : HARX_Op<"MaxPool", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX MaxPool operation";
  let description = [{
    HARX operation to perform an MaxPool.
    This operation does not support broadcasting.
  }];

  // Operands and result are restricted to MLIR tensor types only.
  let arguments = (ins TensorOf<[UI8]>:$input,
                       UI8Attr:$kernel_height, 
                       UI8Attr:$kernel_width, 
                       UI8Attr:$pad_size, 
                       UI8Attr:$stride_size);

  let results   = (outs TensorOf<[UI8]>:$output);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper *HARXMaxPoolOp::getShapeHelper(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb,
        onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new HARXMatMulOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}


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
//   let results = (outs TensorOf<UI8>:$output);
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


#endif // HARX_OPS
