add_onnx_mlir_library(OMHARXToLLVM
  HARXToLLVM.cpp
  HARXToLLVMCommon.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMHARXOps
  OMMlirDialects

  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
)
