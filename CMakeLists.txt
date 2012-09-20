########################################
#
# SuperResolution
#
# Vladislav Vinogradov (jet47)
#

########### CMake properties ########### 

cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

########## Project properties ##########

project(SuperResolution)

set(PROJECT_BASE_NAME "super_resolution")

set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "1")
set(PROJECT_VERSION_PATCH "0")
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

set(PROJECT_VENDOR "Vladislav Vinogradov (jet47)")
set(PROJECT_COPYRIGHT_YEAR "2012")

############ Build options #############

option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(BUILD_SAMPLE_APPS "Build sample applications" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(BIN_INSTALL_DIR "bin")
set(LIB_INSTALL_DIR "lib")
set(INCLUDE_INSTALL_DIR "include")    
set(DOC_INSTALL_DIR "share/doc/${PROJECT_NAME}")

message(STATUS "Configuring ${PROJECT_NAME} in ${CMAKE_BUILD_TYPE} mode")
message(STATUS "Build options :")
message(STATUS "Build shared libraries    : ${BUILD_SHARED_LIBS}")
message(STATUS "Build sample applications : ${BUILD_SAMPLE_APPS}")

############ Find packages #############

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

include_directories("${CMAKE_SOURCE_DIR}/lib/include")
include_directories("${CMAKE_BINARY_DIR}/lib")

############ Subdirectories ############

add_subdirectory(lib)
	
if(BUILD_SAMPLE_APPS)
    add_subdirectory(single_image)
    add_subdirectory(video)
endif()

################ CPack #################

set(README_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(LICENSE_FILE "${CMAKE_SOURCE_DIR}/COPYING")

install(FILES ${LICENSE_FILE} ${README_FILE} DESTINATION ${DOC_INSTALL_DIR})

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${BIN_INSTALL_DIR})
include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Implementation of different Super Resolution algorithms")
set(CPACK_PACKAGE_VENDOR "${PROJECT_VENDOR}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_RESOURCE_FILE_LICENSE "${LICENSE_FILE}")
set(CPACK_RESOURCE_FILE_README "${README_FILE}")
if(WIN32)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}")
    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY "${BIN_INSTALL_DIR}")
    set(CPACK_NSIS_MENU_LINKS "${LICENSE_FILE}" "License" "${README_FILE}" "Readme")
elseif(APPLE)
    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")
    set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
    set(CPACK_SYSTEM_NAME "OSX")
elseif(UNIX)
    set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

include(CPack)

