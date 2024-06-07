add_custom_command(OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o
	COMMAND
	onnx-mlir-pt -O3 -EmitObj ../mnist.onnx -o mnistO3
)

add_custom_target(mnistNet DEPENDS
	${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o
)

add_executable(mnistsp ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
 	/home/msyu/CLionProjects/onnx-mlir-pt/include
)
add_dependencies(mnistsp mnistNet)
target_link_libraries(mnistsp -L /home/msyu/CLionProjects/onnx-mlir-pt/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(mnistsp
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)

#add_custom_command(
#        TARGET minstsp POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy
#        ${CMAKE_CURRENT_SOURCE_DIR}/0.png
#        ${CMAKE_CURRENT_BINARY_DIR}/0.png
#	)
