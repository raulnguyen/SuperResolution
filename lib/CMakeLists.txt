########################################
#
# SuperResolution library
#
# Vladislav Vinogradov <vlad.vinogradov47@gmail.com>
#

file(GLOB HEADER_FILES "include/*.h" "include/*.hpp")
file(GLOB SOURCE_FILES "src/*.cpp")
file(GLOB INTERNAL_HEADER_FILES "src/*.h" "src/*.hpp")
	
if(BUILD_SHARED_LIBS)
    add_definitions(-DSUPER_RESOLUTION_SHARED)
endif()

set(TARGET_NAME ${LIBRARY_NAME})

add_library(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${INTERNAL_HEADER_FILES})

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

set_target_properties(${TARGET_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

