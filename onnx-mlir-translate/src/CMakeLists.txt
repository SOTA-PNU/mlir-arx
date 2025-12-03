add_subdirectory(Python)



get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)


message(STATUS "OMLibs = ${OMLibs}")



add_onnx_mlir_executable(onnx-mlir-translate
  onnx-mlir-translate.cpp
  
  LINK_LIBS PRIVATE
  ${OMLibs}


  MLIRFuncDialect
  onnx
  onnx_proto


  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROpenMPToLLVM
  MLIROptLib
  MLIRSCFToOpenMP
  MLIRIR

  MLIRShapeDialect

)


target_link_libraries(onnx-mlir-translate
  PRIVATE
  ${translation_libs}
  ${test_libs}
  )

mlir_check_link_libraries(onnx-mlir-translate)
