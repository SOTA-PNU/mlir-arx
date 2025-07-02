// SPDX-License-Identifier: Apache-2.0

//===-- ZLowOps.td -- ZLow Dialect Operation Definitions -*- tablegen ------==//
//
// Copyright 2019-2020 The IBM Research Authors
//
// =============================================================================
//
// Defines ZLow Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZLOW_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def LARX_Dialect : Dialect {
  let name = "larx";
  let summary = "A low-level dialect for the ONNX NNPA accelerator ISA.";
  let cppNamespace = "::onnx_mlir::larx";  
  let usePropertiesForAttributes = 0;
}

// Base class for ZLow dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class LARX_Op<string mnemonic, list<Trait> traits = []> :
    Op<LARX_Dialect, mnemonic, traits>;

// DLFloat16 is not natively supported in MLIR.
// F16 is used instead as a container.
def DLF16 : Type<CPred<"$_self.isF16()">, "dlfloat16 type">,
                 BuildableType<"$_builder.getF16Type()">;

// MemRef-like type for zTensor.
def ZMemRef : MemRefOf<[DLF16]>;

//===----------------------------------------------------------------------===//
// LARX Operations
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/SideEffectInterfaces.td"

def LARXAddOp:LARX_Op<"add", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX add operation";
  let description = [{
  LARX operation to perform an add.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def LARXSubOp:LARX_Op<"sub", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX sub operation";
  let description = [{
  LARX operation to perform a sub.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def LARXMulOp:LARX_Op<"mul", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX mul operation";
  let description = [{
  LARX operation to perform a mul.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def LARXDivOp:LARX_Op<"div", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX div operation";
  let description = [{
  LARX operation to perform a div.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def LARXReluOp:LARX_Op<"relu", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX relu operation";
  let description = [{
  LARX operation to perform a relu.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def LARXTanhOp:LARX_Op<"tanh", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX tanh operation";
  let description = [{
  LARX operation to perform a tanh.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def LARXSoftmaxOp:LARX_Op<"softmax", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX softmax operation";
  let description = [{
  LARX operation to perform a softmax.
  work_area: a 4K-aligned buffer.
  act_func: ACT_NONE or ACT_LOG.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I8]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$act_func);
}

def LARXMatMulOp:LARX_Op<"matmul", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX matmul operation";
  let description = [{
  LARX operation to perform a matmul.
  * In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
  shape is a 1D MemRef (memref<3xi64>) whose items are:
    * 1st item: m
    * 2nd item: n
    * 3rd item: p
  * In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
       or broadcasting: X(s, m, n) * Y(n, p) + Bias(p)
  shape is a 1D MemRef (memref<4xi64>) whose items are:
    * 1st item: s
    * 2nd item: m
    * 3rd item: n
    * 4th item: p
  * is_bcast: -1 broadcasting, 0: no broadcasting.
  * is_stacked: -1 stacked, 0: unstacked.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       ZMemRef:$Bias,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_bcast,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_stacked);
}

def LARXConv2DOp:LARX_Op<"conv2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX 2D convolution operation";
  let description = [{
  LARX operation to perform 2D convolution.
  * shape is a 1D MemRef (memref<7xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: channel in
    * 3rd item: height in
    * 4th item: width in
    * 5th item: channel out
    * 6th item: height out
    * 7th item: width out
  * kernel_shape: 1D array of kernel height and width
  * strides: 1D array of stride height and width
  * padding_type: SAME_PADDING or VALID_PADDING.
  * act_func: ACT_NONE or ACT_RELU.
  }];
  let arguments = (ins ZMemRef:$input,
                       ZMemRef:$input_kernel,
                       ZMemRef:$input_bias,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
}

def LARXAvgPool2DOp:LARX_Op<"avgpool2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX 2D average pooling operation";
  let description = [{
  LARX operation to perform 2D average pooling.
  * shape is a 1D MemRef (memref<6xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: channel
    * 3rd item: height in
    * 4th item: width in
    * 5th item: height out
    * 6th item: width out
  * kernel_shape: 1D array of kernel height and width
  * strides: 1D array of stride height and width
  * padding_type: SAME_PADDING or VALID_PADDING.
  }];
  let arguments = (ins ZMemRef:$input,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
                  );
}

def LARXMaxPool2DOp:LARX_Op<"maxpool2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LARX 2D max pooling operation";
  let description = [{
  LARX operation to perform 2D max pooling.
  * shape is a 1D MemRef (memref<6xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: channel
    * 3rd item: height in
    * 4th item: width in
    * 5th item: height out
    * 6th item: width out
  * kernel_shape: 1D array of kernel height and width
  * strides: 1D array of stride height and width
  * padding_type: SAME_PADDING or VALID_PADDING.
  }];
  let arguments = (ins ZMemRef:$input,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
                  );
}

def LARXDummyOp:LARX_Op<"dummy", [MemRefsNormalizable]> {
  let summary = "LARX dummy operation that behaves like identity";
  let description = [{
  LARX operation to forward the input value to the output value.
  It will be removed if canonicalization is called.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      build($_builder, $_state, input.getType(), input);
    }]>
  ];
  let hasCanonicalizer = 1;
}

#endif // ZLOW_OPS
