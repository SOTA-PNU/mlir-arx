module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "uint_model_1x10x10"} {
  func.func @main_graph(%arg0: tensor<1x10x10xui8>) -> tensor<1x10x10xui8> {
    %0 = "harx.Constant"() {value = dense<[[[136, 139, 2, 229, 131, 247, 25, 77, 94, 91], [219, 70, 78, 194, 188, 100, 108, 102, 215, 247], [186, 64, 34, 221, 7, 250, 232, 233, 215, 161], [83, 220, 19, 130, 173, 4, 251, 26, 58, 15], [120, 144, 31, 161, 5, 120, 10, 8, 114, 222], [169, 211, 75, 100, 58, 101, 61, 220, 192, 122], [66, 214, 58, 210, 154, 195, 196, 169, 219, 119], [8, 3, 59, 247, 168, 148, 68, 27, 90, 16], [224, 44, 59, 86, 146, 192, 195, 187, 29, 45], [180, 77, 140, 45, 246, 71, 228, 252, 144, 96]]]> : tensor<1x10x10xui8>} : () -> tensor<1x10x10xui8>
    %1 = "harx.Add"(%arg0, %0) {node_name = "/Add", out_offset = 0.000000e+00 : f32, out_scale = 1.000000e+00 : f32, x_offset = 0.000000e+00 : f32, x_scale = 1.000000e+00 : f32, y_offset = 0.000000e+00 : f32, y_scale = 1.000000e+00 : f32} : (tensor<1x10x10xui8>, tensor<1x10x10xui8>) -> tensor<1x10x10xui8>
    return %1 : tensor<1x10x10xui8>
  }
}

