########################################
#
# SuperResolution
#
# Vladislav Vinogradov <vlad.vinogradov47@gmail.com>
#

########### CMake properties ########### 

cmake_minimum_required(VERSION 2.8)

# Avoid repeating the IF statement in the ENDIF and ELSE
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

########## Project properties ##########

project(SuperResolution)

############ Build options #############

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Output directory for libraries")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications")

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
set(BUILD_TEST_APP ON CACHE BOOL "Build test application")

set(LIBRARY_NAME "super_resolution")

############ Find packages #############

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

include_directories("${CMAKE_SOURCE_DIR}/lib/include")

############ Subdirectories ############

add_subdirectory(lib)
	
if(BUILD_TEST_APP)
    add_subdirectory(test)
endif()

