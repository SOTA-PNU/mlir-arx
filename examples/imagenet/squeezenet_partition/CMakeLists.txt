set(ONNX_MLIR_ROOT "/home/msyu/CLionProjects/onnx-mlir-part")
set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib")
#set(ONNX_MLIR_RUNTIME_DIR "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib")
#
add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_squeezenet/squeezenetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_squeezenet/squeezenetO3_p1.o
		COMMAND
			onnx-mlir-part -O3 -EmitObj -o squeezenetO3 -emit-folder=ptout_imagenet_squeezenet
			-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/DeviceConfig.yaml
			-partition-plan=/home/msyu/CLionProjects/onnx-mlir-part/examples/imagenet/squeezenet_partition/PartitionPlan.yaml
			/home/msyu/Dropbox/Demo/models/squeezenet.onnx
)

add_custom_target(squeezenetO3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_squeezenet/squeezenetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_squeezenet/squeezenetO3_p1.o
)

add_executable(squeezenetO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_squeezenet/squeezenetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet_squeezenet/squeezenetO3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		/home/msyu/CLionProjects/onnx-mlir-part/include
)
add_dependencies(squeezenetO3PartExe
		squeezenetO3PartExeNet
)

target_link_libraries(squeezenetO3PartExe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(squeezenetO3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

#add_custom_command(
#		TARGET squeezenetO3PartExe POST_BUILD
#		COMMAND ${CMAKE_COMMAND} -E copy
#		${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
#		${CMAKE_CURRENT_BINARY_DIR}/cat_285.png
#)


add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/squeezenetO3.o
		COMMAND
			onnx-mlir -O3 -EmitObj -o squeezenetO3
#				-emit-folder=squeezenetO3_out
#				-device-cfg=/home/msyu/CLionProjects/onnx-mlir-part/examples/partition/deviceConfig.yaml
			/home/msyu/Dropbox/Demo/models/squeezenet.onnx
#		COMMAND ${CMAKE_COMMAND} -E copy
#		${CMAKE_CURRENT_SOURCE_DIR}/squeezenetO3.o
#		${CMAKE_CURRENT_BINARY_DIR}/squeezenetO3.o
)
#
#add_custom_target(squeezenetO3ExeNet DEPENDS
#		${CMAKE_CURRENT_BINARY_DIR}/squeezenetO3.o
#)
#
#add_executable(squeezenetO3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#		${CMAKE_CURRENT_BINARY_DIR}/squeezenetO3.o
#)
#
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
#		/home/msyu/CLionProjects/onnx-mlir-part/include
#)
#add_dependencies(squeezenetO3Exe
#		squeezenetO3ExeNet
#)
#
#target_link_libraries(squeezenetO3Exe -L /home/msyu/CLionProjects/onnx-mlir-part/cmake-build-release/Release/lib/libcruntime.a png cruntime)
#set_target_properties(squeezenetO3Exe
#		PROPERTIES
#		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)
#
#add_custom_command(
#		TARGET squeezenetO3Exe POST_BUILD
#		COMMAND ${CMAKE_COMMAND} -E copy
#		${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
#		${CMAKE_CURRENT_BINARY_DIR}/cat_285.png
#)
#
#
