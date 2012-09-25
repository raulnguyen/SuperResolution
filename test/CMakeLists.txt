########################################
#
# SuperResolution test
#
# Vladislav Vinogradov (jet47)
#

set(TARGET_NAME "${PROJECT_BASE_NAME}_test")

file(GLOB HEADER_FILES "*.h" "*.hpp")
file(GLOB SOURCE_FILES "*.c" "*.cpp")

add_executable(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})

add_dependencies(${TARGET_NAME} ${PROJECT_BASE_NAME})

target_link_libraries(${TARGET_NAME} ${PROJECT_BASE_NAME} ${OpenCV_LIBS} ${GTEST_LIBRARIES})

enable_testing()
add_test(AllTests ${TARGET_NAME})
