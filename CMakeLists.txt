cmake_minimum_required(VERSION 3.10)
project(openCLNN LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIRS})

add_compile_definitions(KERNELS_DIR="${CMAKE_SOURCE_DIR}/kernels")

set(
		SOURCES
		src/main.cpp
		src/FC.cpp
		src/KernelUtils.cpp
		src/Module.cpp
		src/Layer.cpp
		src/Sigmoid.cpp
		src/MSE.cpp
)

add_library(openCLNNLib ${SOURCES})

add_executable(openCLNN ${SOURCES})

target_link_libraries(openCLNN ${OpenCL_LIBRARIES})

