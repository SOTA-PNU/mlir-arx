set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")
#
add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_gpt2/gpt2O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_gpt2/gpt2O3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o gpt2O3 -emit-folder=ptout_lm_gpt2
		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
		-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/lm/gpt2_partition/PartitionPlan.yaml
		/home/msyu/Dropbox/Demo/models/gpt2.onnx
)

add_custom_target(gpt2O3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_gpt2/gpt2O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_gpt2/gpt2O3_p1.o
)

add_executable(gpt2O3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_gpt2/gpt2O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_gpt2/gpt2O3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-part/include
)
add_dependencies(gpt2O3PartExe
		gpt2O3PartExeNet
)

target_link_libraries(gpt2O3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(gpt2O3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


#
#add_custom_command(OUTPUT
#		${CMAKE_CURRENT_BINARY_DIR}/gpt2O3.o
#			COMMAND
#			onnx-mlir -O3 -EmitObj -o gpt2O3
#			/home/msyu/Dropbox/Demo/models/gpt2.onnx
##		COMMAND ${CMAKE_COMMAND} -E copy
##		${CMAKE_CURRENT_SOURCE_DIR}/gpt2O3.o
##		${CMAKE_CURRENT_BINARY_DIR}/gpt2O3.o
#)
#
#add_custom_target(gpt2O3ExeNet DEPENDS
#		${CMAKE_CURRENT_BINARY_DIR}/gpt2O3.o
#)
#
#add_executable(gpt2O3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#		${CMAKE_CURRENT_BINARY_DIR}/gpt2O3.o
#)
#
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
#		/home/msyu/CLionProjects/onnx-mlir-part/include
#)
#add_dependencies(gpt2O3Exe
#		gpt2O3ExeNet
#)
#
#target_link_libraries(gpt2O3Exe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib/libcruntime.a png cruntime)
#set_target_properties(gpt2O3Exe
#		PROPERTIES
#		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)
#
#
#
