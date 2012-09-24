# Locate Intel Threading Building Blocks include paths and libraries
# TBB can be found at http://www.threadingbuildingblocks.org/ 
# Written by Hannes Hofmann, hannes.hofmann _at_ informatik.uni-erlangen.de
# Adapted by Gino van den Bergen gino _at_ dtecta.com
#         and Vinogradov Vladislav

# This module defines
# TBB_INCLUDE_DIRS, where to find task_scheduler_init.h, etc.
# TBB_LIBRARY_DIRS, where to find libtbb, libtbbmalloc
# TBB_LIBRARIES, the libraries to link against to use TBB.
# TBB_FOUND, If false, don't try to use TBB.


set(TBB_LIB_NAME "tbb")
set(TBB_LIB_MALLOC_NAME "${TBB_LIB_NAME}malloc")
set(TBB_LIB_DEBUG_NAME "${TBB_LIB_NAME}_debug")
set(TBB_LIB_MALLOC_DEBUG_NAME "${TBB_LIB_MALLOC_NAME}_debug")


if(WIN32)
    set(TBB_DEFAULT_INSTALL_DIR "C:/Program Files/Intel/TBB")
    #-- Find TBB install dir and set ${_TBB_INSTALL_DIR} and cached ${TBB_INSTALL_DIR}
    # first: use CMake variable TBB_INSTALL_DIR
    if(TBB_INSTALL_DIR)
        set(_TBB_INSTALL_DIR ${TBB_INSTALL_DIR})
    endif(TBB_INSTALL_DIR)
    # second: use environment variable
    if(NOT _TBB_INSTALL_DIR)
        if(NOT "$ENV{TBB_INSTALL_DIR}" STREQUAL "")
            set (_TBB_INSTALL_DIR $ENV{TBB_INSTALL_DIR})
        endif(NOT "$ENV{TBB_INSTALL_DIR}" STREQUAL "")
        # Intel recommends setting TBB21_INSTALL_DIR
        if(NOT "$ENV{TBB21_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB21_INSTALL_DIR})
        endif(NOT "$ENV{TBB21_INSTALL_DIR}" STREQUAL "")
        if(NOT "$ENV{TBB22_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB22_INSTALL_DIR})
        endif(NOT "$ENV{TBB22_INSTALL_DIR}" STREQUAL "")
        if(NOT "$ENV{TBB30_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB30_INSTALL_DIR})
        endif(NOT "$ENV{TBB30_INSTALL_DIR}" STREQUAL "")
    endif(NOT _TBB_INSTALL_DIR)
    # third: try to find path automatically
    if(NOT _TBB_INSTALL_DIR)
        set(_TBB_INSTALL_DIR ${TBB_DEFAULT_INSTALL_DIR})
    endif(NOT _TBB_INSTALL_DIR)
    # finally: set the cached CMake variable TBB_INSTALL_DIR
    set(TBB_INSTALL_DIR ${_TBB_INSTALL_DIR} CACHE PATH "Intel TBB install directory")
    mark_as_advanced(TBB_INSTALL_DIR)

    if (MSVC71)
        set (TBB_COMPILER "vc7.1")
    endif(MSVC71)
    if (MSVC80)
        set(TBB_COMPILER "vc8")
    endif(MSVC80)
    if (MSVC90)
        set(TBB_COMPILER "vc9")
    endif(MSVC90)
    if(MSVC10)
        set(TBB_COMPILER "vc10")
    endif(MSVC10)
    if(NOT TBB_COMPILER)
        message(WARNING "TBB supports only VC 7.1, 8, 9 and 10 compilers on Windows platforms.")
    endif (NOT TBB_COMPILER)

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        set(X86_64 1)
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(X86_64 1)
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
        set(X86 1)
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
        set(X86 1)
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
        set(X86 1)
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)

    if(X86_64)
        set(_TBB_ARCHITECTURE "intel64")
    endif(X86_64)
    if(X86)
        set(_TBB_ARCHITECTURE "ia32")
    endif(X86)
    set(TBB_ARCHITECTURE ${_TBB_ARCHITECTURE} CACHE STRING "TBB Architecture (ia32 OR intel64)")
    mark_as_advanced(TBB_ARCHITECTURE)

    set(TBB_INC_SEARCH_DIR "${TBB_INSTALL_DIR}/include")
    set(TBB_LIBRARY_DIR "${TBB_INSTALL_DIR}/lib/${TBB_ARCHITECTURE}/${TBB_COMPILER}")
endif (WIN32)


if(UNIX AND NOT APPLE)
    set(TBB_INC_SEARCH_DIR "usr/include")
    set(TBB_LIBRARY_DIR "usr/lib")
endif(UNIX AND NOT APPLE)


#-- Clear the public variables
set (TBB_FOUND "NO")


#-- Look for include directory and set ${TBB_INCLUDE_DIR}
find_path(TBB_INCLUDE_DIRS NAMES tbb/tbb.h PATHS ${TBB_INC_SEARCH_DIR})
mark_as_advanced(TBB_INCLUDE_DIRS)


#-- Look for libraries
find_library(TBB_LIBRARY        NAMES ${TBB_LIB_NAME}        PATHS ${TBB_LIBRARY_DIR})
find_library(TBB_MALLOC_LIBRARY NAMES ${TBB_LIB_MALLOC_NAME} PATHS ${TBB_LIBRARY_DIR})

#Extract path from TBB_LIBRARY name
get_filename_component(TBB_LIBRARY_DIR ${TBB_LIBRARY} PATH)
mark_as_advanced(TBB_LIBRARY TBB_MALLOC_LIBRARY)

#-- Look for debug libraries
if(WIN32)
    find_library(TBB_LIBRARY_DEBUG        NAMES ${TBB_LIB_DEBUG_NAME}        PATHS ${TBB_LIBRARY_DIR})
    find_library(TBB_MALLOC_LIBRARY_DEBUG NAMES ${TBB_LIB_MALLOC_DEBUG_NAME} PATHS ${TBB_LIBRARY_DIR})
    mark_as_advanced(TBB_LIBRARY_DEBUG TBB_MALLOC_LIBRARY_DEBUG)
endif(WIN32)


if(TBB_INCLUDE_DIRS)
    if(TBB_LIBRARY)
        set(TBB_FOUND "YES")
        if(WIN32)
            set(TBB_LIBRARIES optimized ${TBB_LIBRARY} debug ${TBB_LIBRARY_DEBUG} optimized ${TBB_MALLOC_LIBRARY} debug ${TBB_MALLOC_LIBRARY_DEBUG} ${TBB_LIBRARIES})
        else(WIN32)
            set(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})
        endif(WIN32)
        set(TBB_LIBRARY_DIRS ${TBB_LIBRARY_DIR} CACHE PATH "TBB library directory" FORCE)
        mark_as_advanced(TBB_INCLUDE_DIRS TBB_LIBRARY_DIRS)
        message(STATUS "Found Intel TBB")
    endif(TBB_LIBRARY)
endif(TBB_INCLUDE_DIRS)

if(NOT TBB_FOUND)
    message("ERROR: Intel TBB NOT found!")
    # do only throw fatal, if this pkg is REQUIRED
    if(TBB_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find TBB library.")
    endif(TBB_FIND_REQUIRED)
endif(NOT TBB_FOUND)

