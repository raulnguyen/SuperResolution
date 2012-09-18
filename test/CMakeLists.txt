########################################
#
# SuperResolution test application
#
# Vladislav Vinogradov <vlad.vinogradov47@gmail.com>
#

set(TARGET_NAME "super_resolution_test")

file(GLOB HEADER_FILES "*.h" "*.hpp")
file(GLOB SOURCE_FILES "*.cpp")

add_executable(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})
	
add_dependencies(${TARGET_NAME} ${LIBRARY_NAME}) 

target_link_libraries(${TARGET_NAME} ${LIBRARY_NAME} ${OpenCV_LIBS})
	
set_target_properties(${TARGET_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

