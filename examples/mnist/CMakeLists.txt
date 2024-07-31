add_custom_command(
		OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o

		COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_SOURCE_DIR}/mnistO3.o
		${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o
)

add_custom_target(mnistNet DEPENDS
	${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o
)

add_executable(mnistsp ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_BINARY_DIR}/mnistO3.o
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
		${CMAKE_SOURCE_DIR}/include
)
add_dependencies(mnistsp mnistNet)
target_link_libraries(mnistsp -L ${CMAKE_CURRENT_BINARY_DIR}/../../${CMAKE_BUILD_TYPE}/lib/libcruntime.a png cruntime)
set_target_properties(mnistsp
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
		TARGET mnistsp
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND} -E echo "mnistsp "
		COMMAND mnistsp ${CMAKE_CURRENT_SOURCE_DIR}/0.png
)
