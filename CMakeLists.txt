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
option(BUILD_SAMPLE_APPS "Build sample applications" ON)
option(BUILD_TESTS "Build tests" OFF)

set(BIN_INSTALL_DIR "bin")
set(LIB_INSTALL_DIR "lib")
set(INCLUDE_INSTALL_DIR "include")
set(DOC_INSTALL_DIR "share/doc/${PROJECT_NAME}")

############ Find packages #############

include_directories("${CMAKE_SOURCE_DIR}/lib/include")
include_directories("${CMAKE_BINARY_DIR}/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

############ Subdirectories ############

add_subdirectory(lib)

if(BUILD_SAMPLE_APPS)
    add_subdirectory(samples)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

############### Install ################

set(README_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(LICENSE_FILE "${CMAKE_SOURCE_DIR}/COPYING")

install(FILES ${LICENSE_FILE} ${README_FILE} DESTINATION ${DOC_INSTALL_DIR})

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${BIN_INSTALL_DIR})
include(InstallRequiredSystemLibraries)

################ CPack #################

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

############### Summary ################

message(STATUS "Configuring ${PROJECT_NAME}")

message(STATUS "")
message(STATUS "================= Platform ==================")
message(STATUS "")
message(STATUS "Host   : ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Target : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
message(STATUS "")
message(STATUS "CMake            : ${CMAKE_VERSION}")
message(STATUS "CMake Generator  : ${CMAKE_GENERATOR}")
message(STATUS "CMake Build Tool : ${CMAKE_BUILD_TOOL}")
if(MSVC)
    message(STATUS "MSVC             : ${MSVC_VERSION}")
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
    message(STATUS "Xcode            : ${XCODE_VERSION}")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
    message(STATUS "Configuration    : ${CMAKE_BUILD_TYPE}")
endif()

message(STATUS "")
message(STATUS "=============== Build options ===============")
message(STATUS "")
message(STATUS "Build shared libraries    : ${BUILD_SHARED_LIBS}")
message(STATUS "Build sample applications : ${BUILD_SAMPLE_APPS}")
message(STATUS "Build tests               : ${BUILD_TESTS}")

message(STATUS "")
message(STATUS "============== C/C++ options ================")
message(STATUS "")
message(STATUS "C++ Compiler               : ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags (Release)        : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug)          : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags (RelWithDebInfo) : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags (MinSizeRel)     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
message(STATUS "")
message(STATUS "C Compiler               : ${CMAKE_C_COMPILER}")
message(STATUS "C flags (Release)        : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "C flags (Debug)          : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "C flags (RelWithDebInfo) : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
message(STATUS "C flags (MinSizeRel)     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_MINSIZEREL}")
