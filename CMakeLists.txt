cmake_minimum_required(VERSION 2.8.3)

set(target SuperResolution)
project(${target})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

file(GLOB sources "*.h" "*.cpp")
add_executable(${target} ${sources})

target_link_libraries(${target} ${OpenCV_LIBS})

