########################################
#
# SuperResolution library
#
# Vladislav Vinogradov (jet47)
#

set(TARGET_NAME ${SUPER_RESOLUTION_LIBRARY_NAME})

file(GLOB HEADER_FILES include/*.h include/*.hpp)
file(GLOB SOURCE_FILES src/*.c src/*.cpp)
file(GLOB INTERNAL_HEADER_FILES src/*.h src/*.hpp)

include(GenerateExportHeader)

if(BUILD_SHARED_LIBS)
    add_compiler_export_flags()
endif()

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${INTERNAL_HEADER_FILES})

generate_export_header(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${SUPER_RESOLUTION_VERSION})

install(FILES ${HEADER_FILES} "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_export.h" DESTINATION include)
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

