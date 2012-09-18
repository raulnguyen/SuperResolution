########################################
#
# SuperResolution
#
# Vladislav Vinogradov (jet47)
#

########### CMake properties ########### 

cmake_minimum_required(VERSION 2.8)

# Avoid repeating the IF statement in the ENDIF and ELSE
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

########## Project properties ##########

project(SuperResolution)

############ Build options #############

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

################ CPack #################

include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Implementation of different Super Resolution algorithms")
set(CPACK_PACKAGE_VENDOR "Vladislav Vinogradov (jet47)")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "3")

include(CPack)

