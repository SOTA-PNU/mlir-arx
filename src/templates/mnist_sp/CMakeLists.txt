add_custom_command(OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/#OBJ_NAME#.o
#	COMMAND
#	onnx-mlir-pt -O3 -EmitObj -o #OBJ_NAME# 
#		-emit-folder=#OBJ_NAME#_out 
#		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-pt/examples/partition/deviceConfig.yaml 
#		/home/msyu/CLionProjects/onnx-mlir-pt/examples/mnist/mnist.onnx 
	COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/#OBJ_NAME#.o
        ${CMAKE_CURRENT_BINARY_DIR}/#OBJ_NAME#.o
)

add_custom_target(#EXE_NAME#Net DEPENDS
	${CMAKE_CURRENT_BINARY_DIR}/#OBJ_NAME#.o
)

add_executable(#EXE_NAME# ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_BINARY_DIR}/#OBJ_NAME#.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
 	#ONNX_MLIR_ROOT#/include
)
add_dependencies(#EXE_NAME# 
#EXE_NAME#Net
)

target_link_libraries(#EXE_NAME# -L #ONNX_MLIR_RUNTIME_DIR#/libcruntime.a png cruntime)
set_target_properties(#EXE_NAME#
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)

add_custom_command(
        TARGET #EXE_NAME# POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/0.png
        ${CMAKE_CURRENT_BINARY_DIR}/0.png
	)


