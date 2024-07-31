set(ONNX_MLIR_ROOT ${CMAKE_SOURCE_DIR})
set(ONNX_MLIR_RUNTIME_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib)

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/squeezenetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/squeezenetO3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o squeezenetO3 -emit-folder=ptout_imagenet
		-device-cfg=${CMAKE_SOURCE_DIR}/examples/partition/DeviceConfig.yaml
		-partition-plan=${CMAKE_CURRENT_SOURCE_DIR}/PartitionPlan.yaml
		${CMAKE_SOURCE_DIR}/ai-models/squeezenet.onnx
)

add_custom_target(squeezenetO3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/squeezenetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/squeezenetO3_p1.o
)

add_executable(squeezenetO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/squeezenetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/squeezenetO3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		${CMAKE_SOURCE_DIR}/include
)
add_dependencies(squeezenetO3PartExe
		squeezenetO3PartExeNet
)

target_link_libraries(squeezenetO3PartExe -L ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib/libcruntime.a png cruntime)
set_target_properties(squeezenetO3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET squeezenetO3PartExe
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND} -E echo "Running squeezenetO3PartExe "
		COMMAND squeezenetO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
)
