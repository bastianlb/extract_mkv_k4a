SET(TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/k4a.cpp")

enable_testing()

SET(TARGET_NAME hello_test)
# build gtests

add_executable(${TARGET_NAME} ${TEST_SOURCES})
target_link_libraries(${TARGET_NAME} extract_mkv)
