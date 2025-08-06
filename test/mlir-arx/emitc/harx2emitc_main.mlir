module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "mnist-12-int8"} {
  emitc.include <"stdint.h">
  emitc.include <"stdlib.h">
  emitc.include <"string.h">
  emitc.include <"stdbool.h">
  emitc.func private @convolution_i8_shift(!emitc.ptr<ui8>, !emitc.ptr<ui8>, !emitc.ptr<si32>, !emitc.ptr<ui8>, !emitc.ptr<ui8>, ui8, ui8, ui8, ui8, ui8, ui8, ui8, ui8, ui8, i1, i1, !emitc.ptr<si32>, ui8, ui8) -> si32 attributes {specifiers = ["extern"]}
  emitc.func private @fullyconnected_i8_shift(!emitc.ptr<ui8>, !emitc.ptr<ui8>, !emitc.ptr<si32>, !emitc.ptr<ui8>, ui32, ui32, ui32, ui32, i1, !emitc.ptr<si32>, ui32, ui32) -> si32 attributes {specifiers = ["extern"]}
  emitc.func private @maxpool_i8(!emitc.ptr<ui8>, !emitc.ptr<ui8>, ui8, ui8, ui8, ui8, ui8, ui8, ui8, ui8) -> si32 attributes {specifiers = ["extern"]}
  emitc.func private @quantize(!emitc.ptr<f32>, !emitc.ptr<ui8>, ui32, !emitc.ptr<f32>, !emitc.ptr<ui8>) -> si32 attributes {specifiers = ["extern"]}
  emitc.func private @dequantize(!emitc.ptr<ui8>, !emitc.ptr<f32>, ui32, !emitc.ptr<f32>, !emitc.ptr<ui8>, si32, i1) -> si32 attributes {specifiers = ["extern"]}
  emitc.func private @transpose_i8(!emitc.ptr<ui8>, !emitc.ptr<ui8>, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui8, ui8, ui8, ui8) -> si32 attributes {specifiers = ["extern"]}
  func.func @main_graph(%arg0: tensor<1x1x28x28xf32>, %arg1: tensor<1x1x28x28xf32>) {
    return
  }
}

