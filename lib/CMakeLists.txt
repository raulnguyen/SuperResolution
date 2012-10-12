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

set(ALL_BUILD_FILES ${HEADER_FILES} ${SOURCE_FILES} ${INTERNAL_HEADER_FILES})

include(GenerateExportHeader)

add_compiler_export_flags()

if(BUILD_WITH_CUDA)
    add_definitions(-DWITH_CUDA)
    file(GLOB CUDA_SOURCE_FILES "src/*.cu")
    CUDA_COMPILE(CUDA_OBJECT_FILES ${CUDA_SOURCE_FILES})
    set_source_files_properties(${CUDA_OBJECT_FILES} PROPERTIES GENERATED TRUE)
    list(APPEND ALL_BUILD_FILES ${CUDA_SOURCE_FILES} ${CUDA_OBJECT_FILES})
endif()

add_library(${TARGET_NAME} ${ALL_BUILD_FILES})

generate_export_header(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

if(BUILD_WITH_CUDA)
    target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES})
endif()

set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

install(FILES ${HEADER_FILES} "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_export.h" DESTINATION ${INCLUDE_INSTALL_DIR})
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${BIN_INSTALL_DIR} LIBRARY DESTINATION ${LIB_INSTALL_DIR} ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
