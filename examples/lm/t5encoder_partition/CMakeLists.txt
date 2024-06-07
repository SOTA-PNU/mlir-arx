set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")
#
#add_custom_command(OUTPUT
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_bertsquad-12/t5encoderO3_p0.o
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_bertsquad-12/t5encoderO3_p1.o
#		COMMAND
#		onnx-mlir-part -O3 -EmitObj -o t5encoderO3 -emit-folder=ptout_imagenet_bertsquad-12
#		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
#		-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/imagenet/bertsquad-12_partition/PartitionPlan.yaml
#		/home/msyu/Dropbox/Demo/models/bertsquad-12.onnx
#)
#
#add_custom_target(t5encoderO3PartExeNet DEPENDS
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_bertsquad-12/t5encoderO3_p0.o
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_bertsquad-12/t5encoderO3_p1.o
#)
#
#add_executable(t5encoderO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_bertsquad-12/t5encoderO3_p0.o
#		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_bertsquad-12/t5encoderO3_p1.o
#)
#
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
#		/home/msyu/CLionProjects/onnx-mlir-part/include
#)
#add_dependencies(t5encoderO3PartExe
#		t5encoderO3PartExeNet
#)
#
#target_link_libraries(t5encoderO3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
#set_target_properties(t5encoderO3PartExe
#		PROPERTIES
#		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)
#
#add_custom_command(
#		TARGET t5encoderO3PartExe POST_BUILD
#		COMMAND ${CMAKE_COMMAND} -E copy
#		${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
#		${CMAKE_CURRENT_BINARY_DIR}/cat_285.png
#)


add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/t5encoderO3.o
			COMMAND
			onnx-mlir -O3 -EmitObj -o t5encoderO3
				/home/msyu/Dropbox/Demo/models/t5encoder.onnx
#		COMMAND ${CMAKE_COMMAND} -E copy
#		${CMAKE_CURRENT_SOURCE_DIR}/t5encoderO3.o
#		${CMAKE_CURRENT_BINARY_DIR}/t5encoderO3.o
)

add_custom_target(t5encoderO3ExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/t5encoderO3.o
)

add_executable(t5encoderO3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/t5encoderO3.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-pt/include
)
add_dependencies(t5encoderO3Exe
		t5encoderO3ExeNet
)

target_link_libraries(t5encoderO3Exe -L /home/msyu/CLionProjects/onnx-mlir-pt/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(t5encoderO3Exe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)



