########################################
#
# SuperResolution test application
#
# Vladislav Vinogradov (jet47)
#

set(TARGET_NAME ${SUPER_RESOLUTION_LIBRARY_NAME}_test)

file(GLOB HEADER_FILES *.h *.hpp)
file(GLOB SOURCE_FILES *.c *.cpp)

include_directories("${CMAKE_BINARY_DIR}/lib")

add_executable(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})
	
add_dependencies(${TARGET_NAME} ${SUPER_RESOLUTION_LIBRARY_NAME}) 

target_link_libraries(${TARGET_NAME} ${SUPER_RESOLUTION_LIBRARY_NAME} ${OpenCV_LIBS})

install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin)

