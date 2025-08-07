add_onnx_mlir_library(OMHARXToLLVM
  HARXToLLVM.cpp
  HARXToLLVMPass.cpp
  pass/EmitSetup.cpp

  pass/QuantizeToEmitC.cpp
  pass/ConstantToEmitC.cpp
  pass/DequantizeToEmitC.cpp
  pass/ConvolutionShiftToEmitC.cpp
  pass/MaxPoolToEmitC.cpp
  pass/ReShapeToEmitC.cpp
  pass/FullyConnectedToEmitC.cpp

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
