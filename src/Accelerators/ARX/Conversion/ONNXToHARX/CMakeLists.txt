add_onnx_mlir_rewriter(ONNXToHARX)
add_onnx_mlir_rewriter(HARXToONNX)

add_onnx_mlir_library(OMONNXToHARX
  ONNXLegalityCheck.cpp
  ONNXToHARX.cpp
  ONNXToHARXCommon.cpp

  DEPENDS
  OMONNXONNXToHARXIncGen

  LINK_LIBS PUBLIC
  OMARXCompilerOptions
  OMONNXOps
  OMONNXToKrnl
  OMShapeInferencePass
  OMHARXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
  )

add_onnx_mlir_rewriter(RewriteONNXForHARX)

add_onnx_mlir_library(OMRewriteONNXForHARX
  ONNXLegalityCheck.cpp
  RewriteONNXForHARX.cpp
  ONNXToHARXCommon.cpp

  DEPENDS
  OMONNXRewriteONNXForHARXIncGen
  
  LINK_LIBS PUBLIC
  OMARXCompilerOptions
  OMONNXOps
  OMONNXToKrnl
  OMHARXOps
  OMLayoutHelper


  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMHARXToONNX
  HARXToONNX.cpp

  DEPENDS
  OMONNXHARXToONNXIncGen

  LINK_LIBS PUBLIC
  OMONNXOps
  OMHARXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMDevicePlacement
  DevicePlacement.cpp
  DevicePlacementHeuristic.cpp
  PerfModel.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  OMONNXOps
  OMONNXToHARX
  OMRewriteONNXForHARX

  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
  )
