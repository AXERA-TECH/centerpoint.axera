cmake_minimum_required(VERSION 3.13)

# Target chip type (default to ax650)
if(NOT DEFINED AXERA_TARGET_CHIP)
    set(AXERA_TARGET_CHIP "ax650" CACHE STRING "AXERA target chip (ax650)")
endif()

project(CenterPoint CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building CenterPoint for chip: ${AXERA_TARGET_CHIP}")

if(NOT DEFINED OpenCV_DIR)
    if(DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "${ENV{OpenCV_DIR}}" CACHE PATH "OpenCV directory")
    else()
        set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv-aarch64-linux/lib/cmake/opencv4" CACHE PATH "OpenCV directory")
    endif()
endif()

message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

find_package(Threads REQUIRED)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

set(SOURCES
    src/main.cpp
    src/centerpoint_common.cpp
    src/data_loader.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/visualization.cpp
    src/utils.cpp
)

set(HEADERS
    include/centerpoint_common.hpp
    include/data_loader.hpp
    include/preprocess.hpp
    include/postprocess.hpp
    include/visualization.hpp
    include/utils.hpp
    include/timer.hpp
)

add_executable(centerpoint_inference ${SOURCES} ${HEADERS})

target_link_libraries(centerpoint_inference
    ${OpenCV_LIBS}
    Threads::Threads
    ax_engine
    ax_interpreter
    ax_sys
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(centerpoint_inference OpenMP::OpenMP_CXX)
endif()

if(DEFINED BSP_MSP_DIR)
    target_include_directories(centerpoint_inference PRIVATE ${BSP_MSP_DIR}/include)
    target_link_directories(centerpoint_inference PRIVATE ${BSP_MSP_DIR}/lib)
else()
    message(WARNING "BSP_MSP_DIR not set, using default path")
    target_include_directories(centerpoint_inference PRIVATE /opt/axera/msp/out/include)
    target_link_directories(centerpoint_inference PRIVATE /opt/axera/msp/out/lib)
endif()

target_compile_options(centerpoint_inference PRIVATE -O3 -Wall)

install(TARGETS centerpoint_inference DESTINATION bin)
