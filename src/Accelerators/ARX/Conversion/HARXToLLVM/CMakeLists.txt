add_onnx_mlir_library(OMHARXToLLVM
  HARXToLLVM.cpp
  HARXToLLVMPass.cpp
  pass/RewriteHarx.cpp
  pass/EmitSetup.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMHARXOps
  OMMlirDialects
  
  ACCEL_INCLUDE_DIRS PRIVATE
  ${ARX_INCLUDE_PATH}
)
