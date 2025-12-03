add_onnx_mlir_library(OMNNTargetPython
  TranslateRegistration.cpp
  TranslateToPython.cpp


  INCLUDE_DIRS

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  )
