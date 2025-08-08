module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "mnist-12-int8"} {
  func.func @main_graph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xf32> {
    return %arg0 : tensor<1x1x28x28xf32>
  }
}

