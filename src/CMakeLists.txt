# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Accelerators)
add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)
add_subdirectory(Tools)
add_subdirectory(Version)
add_subdirectory(Compiler)


add_onnx_mlir_executable(onnx-mlir
  onnx-mlir.cpp

  LINK_LIBS PRIVATE
  MLIROpenMPToLLVMIRTranslation
  OMCompilerOptions
  OMCompilerUtils

  )

add_onnx_mlir_executable(onnx-mlir-part
  onnx-mlir-part.cpp

  LINK_LIBS PRIVATE
  MLIROpenMPToLLVMIRTranslation
  OMCompilerOptions
  OMCompilerUtils
#        yaml-cpp
)

# CMAKE_SOURCE_DIR 값을 정의한 파일을 생성
configure_file(
        ${CMAKE_SOURCE_DIR}/src/Builder/config.h.in
        ${CMAKE_BINARY_DIR}/config.h
        @ONLY
)
