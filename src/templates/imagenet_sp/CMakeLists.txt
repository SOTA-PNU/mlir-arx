add_custom_command(OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/resnet18O3.o
#	COMMAND
#	onnx-mlir-pt -O3 -EmitObj -o resnet18O3 
#		-emit-folder=resnet18O3_out 
#		-device-cfg=/home/msyu/CLionProjects/onnx-mlir-pt/examples/partition/deviceConfig.yaml 
#		/home/msyu/CLionProjects/onnx-mlir-pt/examples/mnist/mnist.onnx 
	COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/resnet18O3.o
        ${CMAKE_CURRENT_BINARY_DIR}/resnet18O3.o
)

add_custom_target(resnet18O3ExeNet DEPENDS
	${CMAKE_CURRENT_BINARY_DIR}/resnet18O3.o
)

add_executable(resnet18O3Exe ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_BINARY_DIR}/resnet18O3.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
 	/home/msyu/CLionProjects/onnx-mlir-pt/include
)
add_dependencies(resnet18O3Exe 
resnet18O3ExeNet
)

target_link_libraries(resnet18O3Exe -L /home/msyu/CLionProjects/onnx-mlir-pt/cmake-build-release/Release/lib/libcruntime.a png cruntime)
set_target_properties(resnet18O3Exe
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)

add_custom_command(
        TARGET resnet18O3Exe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/test.png
        ${CMAKE_CURRENT_BINARY_DIR}/test.png
	)


