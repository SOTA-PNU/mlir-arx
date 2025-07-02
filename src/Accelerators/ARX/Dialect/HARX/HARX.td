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
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// HARX Attribute
//===----------------------------------------------------------------------===//

// All of the Tensor attributes will extend this class.
class HARX_Attr<string name, list<Trait> traits = []>
	: BaseLayoutAttr<HARX_Dialect, name, traits>;

// ztensor encoding attribute.
def ZTensorEncodingAttr : HARX_Attr<"ZTensorEncoding"> {
  let hasCustomAssemblyFormat = 1;

  let description = [{
    An attribute to encode information on properties of ztensors. A tensor with
    this encoding attribute indicates that it is a ztensor whose type is
    the original type of the pre-stickified data.

    `dataLayout` indicates the data layout of the pre-stickified data.

    TODO: should we add an affine_map to describe how the data is stickified?

    Example:

    ```mlir
    #ZAIU_NHWC= #HARX.encoding<{
      dataLayout = "nhwc"
    }>

    ... tensor<8x8xf64, #ZAIU_NHWC> ...
    ```
  }];

  // Data in ztensor encoding.
  let parameters = (ins
    // A original data layout of the pre-stickified data.
    "ZTensorEncodingAttr::DataLayout":$dataLayout
  );

  let extraClassDeclaration = [{
    enum class DataLayout {
      UNDEFINED, _1D, _2D, _2DS, _3D, _3DS, _4D, _4DS,
      NCHW, NHWC, HWCK,
      FICO, ZRH, BFICO, BZRH
    };
  }];

  let cppNamespace = "::onnx_mlir::harx";
}

// Whether a ztensor type has the specified layout.
class DataLayoutOfPred<string layout> : And<[
  CPred<"(mlir::cast<::mlir::RankedTensorType>($_self)) &&"
        "(mlir::dyn_cast_or_null<ZTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding())) &&"
        "(mlir::cast<ZTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding()).getDataLayout()"
        " == ZTensorEncodingAttr::DataLayout::" # layout # ")">
]>;

// So far ZTensor supports only F16 for stickified data.
class ZTensorOf<string layout, list<int> ranks> :
    Type<And<[TensorOf<[F16]>.predicate, HasAnyRankOfPred<ranks>,
              DataLayoutOfPred<layout>]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         TensorOf<[F16]>.summary # " with layout " # layout,
         "::mlir::TensorType">;

def UnrankedZTensor : UnrankedTensorOf<[F16]>;

def ZTensor_1D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_1D", [1]>]>;
def ZTensor_2D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_2D", [2]>]>;
def ZTensor_2DS: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_2DS", [2]>]>;
def ZTensor_3D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_3D", [3]>]>;
def ZTensor_3DS: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_3DS", [3]>]>;
def ZTensor_4D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_4D", [4]>]>;
def ZTensor_4DS: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_4DS", [4]>]>;
def ZTensor_NHWC: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"NHWC", [4]>]>;
def ZTensor_NCHW: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"NCHW", [4]>]>;
def ZTensor_HWCK: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"HWCK", [4]>]>;
def ZTensor_FICO: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"FICO", [2, 3]>]>;
def ZTensor_ZRH: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"ZRH", [2, 3]>]>;
def ZTensor_BFICO: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"BFICO", [2, 3]>]>;
def ZTensor_BZRH: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"BZRH", [2, 3]>]>;

def AnyZTensor: AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
      ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
      ZTensor_NCHW, ZTensor_NHWC, ZTensor_HWCK,
      ZTensor_FICO, ZTensor_ZRH, ZTensor_BFICO, ZTensor_BZRH]>;

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

def HARXAddOp:HARX_Op<"Add", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Add operation";
  let description = [{
    HARX operation to perform an Add.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXAddOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXSubOp:HARX_Op<"Sub", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Sub operation";
  let description = [{
    HARX operation to perform a Sub.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXSubOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXMulOp:HARX_Op<"Mul", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Mul operation";
  let description = [{
    HARX operation to perform a Mul.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXMulOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXDivOp:HARX_Op<"Div", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Div operation";
  let description = [{
    HARX operation to perform a Div.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXDivOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXReluOp:HARX_Op<"Relu", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Relu operation";
  let description = [{
    "HARX operation to perform a Relu."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXReluOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXTanhOp:HARX_Op<"Tanh", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Tanh operation";
  let description = [{
    HARX operation to perform a Tanh.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXTanhOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXSoftmaxOp:HARX_Op<"Softmax", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX Softmax operation";
  let description = [{
    HARX operation to perform a Softmax.
    act_func: ACT_NONE or ACT_LOG.
  }];
  let arguments = (ins ZTensor_3DS:$X,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
  let results = (outs ZTensor_3DS:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXSoftmaxOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXMaxPool2DOp:HARX_Op<"MaxPool2D", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX 2D max pooling operation";
  let description = [{
    HARX operation to perform 2D max pooling.
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::ArrayAttr":$kernel_shape,
                   "::mlir::ArrayAttr":$strides, "::mlir::StringAttr":$padding_type), [{
      Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input, kernel_shape, strides, padding_type);
    }]>
  ];
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXMaxPool2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXPoolingOpShapeHelper<HARXMaxPool2DOp>(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXAvgPool2DOp:HARX_Op<"AvgPool2D", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX 2D average pooling operation";
  let description = [{
    HARX operation to perform 2D average pooling.
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::ArrayAttr":$kernel_shape,
                   "::mlir::ArrayAttr":$strides, "::mlir::StringAttr":$padding_type), [{
      Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input, kernel_shape, strides, padding_type);
    }]>
  ];
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXAvgPool2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXPoolingOpShapeHelper<HARXAvgPool2DOp>(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXMatMulOp:HARX_Op<"MatMul", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX MatMul operation";
  let description = [{
    HARX operation to perform a MatMul.
  }];
  let arguments = (ins AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$X,
                       AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$Y,
                       AnyTypeOf<[ZTensor_1D, ZTensor_2DS, NoneType]>:$B);

  let results = (outs AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Y, "::mlir::Value":$B), [{
      Type elementType = mlir::cast<ShapedType>(X.getType()).getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, X, Y, B);
    }]>
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXMatMulOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXMatMulOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def HARXConv2DOp:HARX_Op<"Conv2D", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "HARX 2D convolution operation";
  let description = [{
    HARX operation to perform 2D convolution.
    * input: `[num_batches, height_in, width_in, channels_in]`
    * input_kernel: `[kernel_height, kernel_width, channels_in, channels_out]`
    * input_bias: `[channels_out] `
    * kernel_shape: 1D array of kernel height and width
    * strides: 1D array of stride height and width
    * padding_type: SAME_PADDING or VALID_PADDING
    * act_func: ACT_NONE or ACT_RELU
    * output: `[num_batches, height_out, width_out, channels_out]`
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       ZTensor_HWCK:$input_kernel,
                       AnyTypeOf<[ZTensor_1D, NoneType]>:$input_bias,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func
  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$input_kernel, "::mlir::Value":$input_bias,
                   "::mlir::ArrayAttr":$kernel_shape, "::mlir::ArrayAttr":$strides,
                   "::mlir::StringAttr":$padding_type, "::mlir::StringAttr":$act_func), [{
      Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input, input_kernel, input_bias,
            kernel_shape, strides, padding_type, act_func);
    }]>
  ];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * HARXConv2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new HARXConv2DOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

#endif // HARX_OPS
