set(ONNX_MLIR_ROOT ${CMAKE_SOURCE_DIR})
set(ONNX_MLIR_RUNTIME_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib)

add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/ptout_detection/yolo3O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_detection/yolo3O3_p1.o
		COMMAND
		onnx-mlir-part -O3 -EmitObj -o yolo3O3 -emit-folder=ptout_detection
		-device-cfg=${CMAKE_SOURCE_DIR}/examples/partition/DeviceConfig.yaml
		-partition-plan=${CMAKE_CURRENT_SOURCE_DIR}/PartitionPlan.yaml
		${CMAKE_SOURCE_DIR}/ai-models/yolov3_neubla_movingaverage.onnx

)

add_custom_target(yolo3O3PartExeNet DEPENDS
		${CMAKE_CURRENT_BINARY_DIR}/ptout_detection/yolo3O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_detection/yolo3O3_p1.o
)

add_executable(yolo3O3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		${CMAKE_CURRENT_BINARY_DIR}/ptout_detection/yolo3O3_p0.o
		${CMAKE_CURRENT_BINARY_DIR}/ptout_detection/yolo3O3_p1.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		${CMAKE_SOURCE_DIR}/include
)
add_dependencies(yolo3O3PartExe
		yolo3O3PartExeNet
)

target_link_libraries(yolo3O3PartExe -L ${CMAKE_CURRENT_BINARY_DIR}/../../../${CMAKE_BUILD_TYPE}/lib/libcruntime.a png cruntime)
set_target_properties(yolo3O3PartExe
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


add_custom_command(
		TARGET yolo3O3PartExe
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND} -E echo "Running yolo3O3PartExe "
		COMMAND yolo3O3PartExe ${CMAKE_CURRENT_SOURCE_DIR}/yolo3_608X608.png
)
