set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")
#
#add_custom_command(OUTPUT
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_bertsquad/bertsquadO3_p0.o
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_bertsquad/bertsquadO3_p1.o
#		COMMAND
#		onnx-mlir-part -O3 -EmitObj -o bertsquadO3 -emit-folder=ptout_lm_bertsquad
#		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
#		-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/lm/bertsquad_partition/PartitionPlan.yaml
#		/home/msyu/Dropbox/Demo/models/bertsquad.onnx
#)
#
#add_custom_target(bertsquadO3PartExeNet DEPENDS
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_bertsquad/bertsquadO3_p0.o
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_bertsquad/bertsquadO3_p1.o
#)
#
#add_executable(bertsquadO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_bertsquad/bertsquadO3_p0.o
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_bertsquad/bertsquadO3_p1.o
#)
#
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
#		/home/msyu/CLionProjects/onnx-mlir-part/include
#)
#add_dependencies(bertsquadO3PartExe
#		bertsquadO3PartExeNet
#)
#
#target_link_libraries(bertsquadO3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
#set_target_properties(bertsquadO3PartExe
#		PROPERTIES
#		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)


add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/bertsquadO3.o
			COMMAND
			onnx-mlir -O3 -EmitObj -o bertsquadO3
				/home/msyu/Dropbox/Demo/models/bertsquad.onnx
#		COMMAND ${CMAKE_COMMAND} -E copy
#		${CMAKE_CURRENT_SOURCE_DIR}/bertsquadO3.o
#		${CMAKE_CURRENT_BINARY_DIR}/bertsquadO3.o
)

add_custom_target(bertsquadO3ExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/bertsquadO3.o
)

add_executable(bertsquadO3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/bertsquadO3.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-pt/include
)
add_dependencies(bertsquadO3Exe
		bertsquadO3ExeNet
)

target_link_libraries(bertsquadO3Exe -L /home/msyu/CLionProjects/onnx-mlir-pt/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(bertsquadO3Exe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)



