add_onnx_mlir_dialect(LARX larx)
add_onnx_mlir_dialect_doc(larx LARX.td)

add_onnx_mlir_rewriter(LARXCombine)

add_onnx_mlir_library(OMLARXOps
  LARXOps.cpp
  LARXCombine.cpp
  DialectBuilder.cpp

  DEPENDS
  OMLARXIncGen
  OMONNXLARXCombineIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  OMMlirDialects
)
