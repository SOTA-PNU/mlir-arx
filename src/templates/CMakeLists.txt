set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")

add_custom_command(OUTPUT
#BIN_OBJ_NAMES#
#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O30.o
#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O31.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o resnet18O3 -emit-folder=ptout_imagenet
		-device-cfg=#MLIR_ROOT#/examples/partition/deviceConfig.yaml
		-partition-plan=#MLIR_ROOT#/examples/partition/PartitionPlan.yaml
		/home/msyu/Dropbox/Demo/models/mxnet_exported_resnet18.onnx

		#	COMMAND ${CMAKE_COMMAND} -E copy
		#        ${CMAKE_CURRENT_SOURCE_DIR}/resnet18O30.o
		#        ${CMAKE_CURRENT_BINARY_DIR}/resnet18O30.o

		#		${CMAKE_CURRENT_SOURCE_DIR}/resnet18O31.o
		#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O31.o
)

add_custom_target(#EXE_NAME#Net DEPENDS
#BIN_OBJ_NAMES#
#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O30.o
#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O31.o
)

add_executable(#EXE_NAME# ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#BIN_OBJ_NAMES#
#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O30.o
#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O31.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-part/include
)
add_dependencies(#EXE_NAME#
		#EXE_NAME#Net
)

target_link_libraries(#EXE_NAME#Exe -L #MLIR_ROOT#/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(#EXE_NAME#Exe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET #EXE_NAME# POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_SOURCE_DIR}/test.png
		${CMAKE_CURRENT_BINARY_DIR}/test.png
)


