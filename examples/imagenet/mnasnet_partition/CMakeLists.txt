set(ONNX_MLIR_ROOT ${CMAKE_SOURCE_DIR})
set(ONNX_MLIR_RUNTIME_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib)

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/mnasnetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/mnasnetO3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o mnasnetO3 -emit-folder=ptout_imagenet
		-device-cfg=${CMAKE_SOURCE_DIR}/examples/partition/DeviceConfig.yaml
		-partition-plan=${CMAKE_CURRENT_SOURCE_DIR}/PartitionPlan.yaml
		${CMAKE_SOURCE_DIR}/ai-models/mnasnet.onnx

)

add_custom_target(mnasnetO3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/mnasnetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/mnasnetO3_p1.o
)

add_executable(mnasnetO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/mnasnetO3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_imagenet/mnasnetO3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		${CMAKE_SOURCE_DIR}/include
)
add_dependencies(mnasnetO3PartExe
		mnasnetO3PartExeNet
)

target_link_libraries(mnasnetO3PartExe -L ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib/libcruntime.a png cruntime)
set_target_properties(mnasnetO3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET mnasnetO3PartExe
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND} -E echo "Running mnasnetO3PartExe "
		COMMAND mnasnetO3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/cat_285.png
)
