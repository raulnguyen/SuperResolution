########################################
#
# SuperResolution library
#
# Vladislav Vinogradov (jet47)
#

set(TARGET_NAME ${PROJECT_BASE_NAME})

file(GLOB HEADER_FILES "include/*.h" "include/*.hpp")
file(GLOB SOURCE_FILES "src/*.c" "src/*.cpp")
file(GLOB INTERNAL_HEADER_FILES "src/*.h" "src/*.hpp")

include(GenerateExportHeader)

add_compiler_export_flags()

if(BUILD_TESTS)
    add_definitions(-DWITH_TESTS)
endif()

add_library(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${INTERNAL_HEADER_FILES})

generate_export_header(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

if(BUILD_TESTS)
    target_link_libraries(${TARGET_NAME} ${GTEST_LIBRARIES})
endif()

set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

install(FILES ${HEADER_FILES} "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_export.h" DESTINATION ${INCLUDE_INSTALL_DIR})
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${BIN_INSTALL_DIR} LIBRARY DESTINATION ${LIB_INSTALL_DIR} ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
