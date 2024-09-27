
add_custom_command(
		OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/resnet50O3.o

		COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_SOURCE_DIR}/resnet50O3.o
		${CMAKE_CURRENT_BINARY_DIR}/resnet50O3.o
)

add_custom_target(resnet50O3ExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/resnet50O3.o
)

add_executable(resnet50O3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/resnet50O3.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		${CMAKE_SOURCE_DIR}/include
)
add_dependencies(resnet50O3Exe
		resnet50O3ExeNet
)

target_link_libraries(resnet50O3Exe -L ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib/libcruntime.a png cruntime)
set_target_properties(resnet50O3Exe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET resnet50O3Exe
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND} -E echo "Running resnet50O3Exe "
		COMMAND resnet50O3Exe ${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
)



set(ONNX_MLIR_ROOT ${CMAKE_SOURCE_DIR})
set(ONNX_MLIR_RUNTIME_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib)

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/resnet50O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/resnet50O3_p1.o
		COMMAND
        aws s3 cp s3://onnx-mlir-part/ai-models/resnet50.onnx ${CMAKE_SOURCE_DIR}/ai-models/resnet50.onnx
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o resnet50O3 -emit-folder=ptout_imagenet
		-device-cfg=${CMAKE_SOURCE_DIR}/examples/partition/DeviceConfig.yaml
		-partition-plan=${CMAKE_CURRENT_SOURCE_DIR}/PartitionPlan.yaml
		${CMAKE_SOURCE_DIR}/ai-models/resnet50.onnx
)

add_custom_target(resnet50O3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/resnet50O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/resnet50O3_p1.o
)

add_executable(resnet50O3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/resnet50O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/resnet50O3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		${CMAKE_SOURCE_DIR}/include
)
add_dependencies(resnet50O3PartExe
		resnet50O3PartExeNet
)

target_link_libraries(resnet50O3PartExe -L ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib/libcruntime.a png cruntime)
set_target_properties(resnet50O3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET resnet50O3PartExe
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND} -E echo "Running resnet50O3PartExe "
		COMMAND resnet50O3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
)
