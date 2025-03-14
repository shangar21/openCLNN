cmake_minimum_required(VERSION 3.10)
project(openCLNN LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/submodules/OpenCL-CLHPP/include)

add_compile_definitions(KERNELS_DIR="${CMAKE_SOURCE_DIR}/kernels")

add_compile_options("-DCL_MAKE_VERSION(a,b,c)=(((a)<<20)|((b)<<10)|(c))")

add_compile_definitions(
    CL_HPP_TARGET_OPENCL_VERSION=200
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_KHR_COMMAND_BUFFER_VERSION=0
    CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION=0
		CL_HPP_ENABLE_EXCEPTIONS=1
)

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

