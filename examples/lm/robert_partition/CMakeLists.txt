set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_robertabase/robertabaseO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_robertabase/robertabaseO3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o robertabaseO3 -emit-folder=ptout_lm_robertabase
		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
		-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/lm/robert_partition/PartitionPlan.yaml
		/home/msyu/Dropbox/Demo/models/robertabase.onnx
)

add_custom_target(robertabaseO3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_robertabase/robertabaseO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_robertabase/robertabaseO3_p1.o
)

add_executable(robertabaseO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_robertabase/robertabaseO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_lm_robertabase/robertabaseO3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-part/include
)
add_dependencies(robertabaseO3PartExe
		robertabaseO3PartExeNet
)

target_link_libraries(robertabaseO3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(robertabaseO3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


#
#add_custom_command(OUTPUT
#		${CMAKE_CURRENT_BINARY_DIR}/robertabaseO3.o
#			COMMAND
#			onnx-mlir -O3 -EmitObj -o robertabaseO3
#				/home/msyu/Dropbox/Demo/models/robertabase.onnx
##		COMMAND ${CMAKE_COMMAND} -E copy
##		${CMAKE_CURRENT_SOURCE_DIR}/robertabaseO3.o
##		${CMAKE_CURRENT_BINARY_DIR}/robertabaseO3.o
#)
#
#add_custom_target(robertabaseO3ExeNet DEPENDS
#		${CMAKE_CURRENT_BINARY_DIR}/robertabaseO3.o
#)
#
#add_executable(robertabaseO3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#		${CMAKE_CURRENT_BINARY_DIR}/robertabaseO3.o
#)
#
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
#		/home/msyu/CLionProjects/onnx-mlir-pt/include
#)
#add_dependencies(robertabaseO3Exe
#		robertabaseO3ExeNet
#)
#
#target_link_libraries(robertabaseO3Exe -L /home/msyu/CLionProjects/onnx-mlir-pt/cmake-build-release/Release/lib/libcruntime.a png cruntime)
#set_target_properties(robertabaseO3Exe
#		PROPERTIES
#		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)
