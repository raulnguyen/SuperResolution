########################################
#
# SuperResolution
#
# Vladislav Vinogradov (jet47)
#

########### CMake properties ########### 

cmake_minimum_required(VERSION 2.8.8)

########## Project properties ##########

project(SuperResolution)

set(SUPER_RESOLUTION_LIBRARY_NAME super_resolution)

set(SUPER_RESOLUTION_VERSION_MAJOR 0)
set(SUPER_RESOLUTION_VERSION_MINOR 1)
set(SUPER_RESOLUTION_VERSION_PATCH 0)
set(SUPER_RESOLUTION_VERSION ${SUPER_RESOLUTION_VERSION_MAJOR}.${SUPER_RESOLUTION_VERSION_MINOR}.${SUPER_RESOLUTION_VERSION_PATCH})

############ Build options #############

option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
option(BUILD_TEST_APP "Build test application" OFF)

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

install(FILES README.md COPYING DESTINATION .)

include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Implementation of different Super Resolution algorithms")
set(CPACK_PACKAGE_VENDOR "Vladislav Vinogradov (jet47)")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION_MAJOR ${SUPER_RESOLUTION_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${SUPER_RESOLUTION_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${SUPER_RESOLUTION_VERSION_PATCH})

include(CPack)

