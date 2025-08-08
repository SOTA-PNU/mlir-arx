add_onnx_mlir_library(OMLayoutHelper
  LayoutHelper.cpp
  ARXLimit.cpp

  DEPENDS
  OMCompilerOptions

  LINK_LIBS PUBLIC
  MLIRIR
  OMCompilerOptions

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_SRC_ROOT}
  ${ARX_BIN_ROOT}
  ${ARX_INCLUDE_PATH}
)
