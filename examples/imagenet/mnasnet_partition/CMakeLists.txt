set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_mnasnet/mnasnetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_mnasnet/mnasnetO3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o mnasnetO3 -emit-folder=ptout_imagenet_mnasnet
		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
		-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/imagenet/mnasnet_partition/PartitionPlan.yaml
		/home/msyu/Dropbox/Demo/models/mnasnet.onnx

		#	COMMAND ${CMAKE_COMMAND} -E copy
		#        ${CMAKE_CURRENT_SOURCE_DIR}/resnet18O30.o
		#        ${CMAKE_CURRENT_BINARY_DIR}/resnet18O30.o

		#		${CMAKE_CURRENT_SOURCE_DIR}/resnet18O31.o
		#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O31.o
)

add_custom_target(mnasnetO3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_mnasnet/mnasnetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_mnasnet/mnasnetO3_p1.o
)

add_executable(mnasnetO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_mnasnet/mnasnetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_mnasnet/mnasnetO3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-part/include
)
add_dependencies(mnasnetO3PartExe
		mnasnetO3PartExeNet
)

target_link_libraries(mnasnetO3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(mnasnetO3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET mnasnetO3PartExe POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_SOURCE_DIR}/test.png
		${CMAKE_CURRENT_BINARY_DIR}/test.png
)


