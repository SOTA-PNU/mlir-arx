add_subdirectory(HARXOps)

add_onnx_mlir_dialect(HARX harx)
add_onnx_mlir_dialect_doc(harx HARX.td)

add_onnx_mlir_library(OMHARXOps
  # Top files for HARX dialect.
  HARXOps.cpp
  HARXOps/OpHelper.cpp

  # Support for shape inference, verifier, and canonicalization.
  # Please list in the alphabetical order.
  # HARXOps/Conv2D/Conv2D.cpp
  HARXOps/Elementwise/Elementwise.cpp
  # HARXOps/MatMul/MatMul.cpp
  # HARXOps/Pooling/Pooling.cpp
  # HARXOps/Softmax/Softmax.cpp
  # HARXOps/Unstick/Unstick.cpp
  HARXOps/Stick/Stick.cpp

  DEPENDS
  OMLayoutHelper
  OMHARXIncGen
  OMShapeInferenceOpInterfaceIncGen
  # OMONNXHARXStickIncGen
  # OMONNXHARXUnstickIncGen

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  OMLayoutHelper
  OMShapeHelperOpInterface
  OMARXCompilerOptions
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
  )
