set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_resnet50/resnet50O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_resnet50/resnet50O3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o resnet50O3 -emit-folder=ptout_imagenet_resnet50
		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
		-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/imagenet/resnet50_partition/PartitionPlan.yaml
		/home/msyu/Dropbox/Demo/models/resnet50.onnx

		#	COMMAND ${CMAKE_COMMAND} -E copy
		#        ${CMAKE_CURRENT_SOURCE_DIR}/resnet18O30.o
		#        ${CMAKE_CURRENT_BINARY_DIR}/resnet18O30.o

		#		${CMAKE_CURRENT_SOURCE_DIR}/resnet18O31.o
		#		${CMAKE_CURRENT_BINARY_DIR}/resnet18O31.o
)

add_custom_target(resnet50O3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_resnet50/resnet50O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_resnet50/resnet50O3_p1.o
)

add_executable(resnet50O3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_resnet50/resnet50O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_resnet50/resnet50O3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-part/include
)
add_dependencies(resnet50O3PartExe
		resnet50O3PartExeNet
)

target_link_libraries(resnet50O3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(resnet50O3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET resnet50O3PartExe POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
		${CMAKE_CURRENT_BINARY_DIR}/cat_285.png
)


