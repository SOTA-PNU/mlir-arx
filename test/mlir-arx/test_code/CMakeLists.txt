cmake_minimum_required(VERSION 3.10)
project(chacha LANGUAGES CXX)

# — Require C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# — 외부 정적 라이브러리 등록
add_library(arx_api STATIC IMPORTED)
set_target_properties(arx_api PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libarx_api.a"
)

# — Your executable
add_executable(chacha main.cpp ${UINT_MODEL_O})

# — arx_api 라이브러리 링크
target_link_libraries(chacha
    PRIVATE
        arx_api
)