SET(APP_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/transformation_helpers.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/extract_mkv_k4a.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/timesync.h"
        )

SET(APP_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/transformation_helpers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/extract_mkv_k4a.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/timesync.cpp"
        )

SET(TARGET_NAME extract_mkv_k4a)

add_executable(${TARGET_NAME} ${APP_SOURCES} ${APP_HEADERS} )
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(${TARGET_NAME} PRIVATE
        stdc++fs
        CONAN_PKG::opencv
        CONAN_PKG::magnum
        CONAN_PKG::corrade
        CONAN_PKG::kinect-azure-sensor-sdk
        CONAN_PKG::bzip2
        CONAN_PKG::spdlog
        CONAN_PKG::yaml-cpp
        CONAN_PKG::tbb
        )


target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        )

install(TARGETS ${TARGET_NAME} DESTINATION ${MAGNUM_BINARY_INSTALL_DIR})
