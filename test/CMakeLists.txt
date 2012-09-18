########################################
#
# SuperResolution test application
#
# Vladislav Vinogradov (jet47)
#

set(TARGET_NAME "super_resolution_test")

file(GLOB HEADER_FILES "*.h" "*.hpp")
file(GLOB SOURCE_FILES "*.cpp")

add_executable(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})
	
add_dependencies(${TARGET_NAME} ${LIBRARY_NAME}) 

target_link_libraries(${TARGET_NAME} ${LIBRARY_NAME} ${OpenCV_LIBS})

install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin)

