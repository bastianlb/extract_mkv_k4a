SET(APP_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/transformation_helpers.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv_k4a.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/timesync.h"
        )

SET(APP_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/transformation_helpers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/extract_mkv_k4a.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/timesync.cpp"
        )

SET(TARGET_NAME extract_mkv_k4a)
add_executable(${TARGET_NAME} ${APP_SOURCES} ${APP_HEADERS} )
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

SET(MAIN_LIBS
    CONAN_PKG::opencv
    CONAN_PKG::eigen
    CONAN_PKG::magnum
    CONAN_PKG::corrade
    CONAN_PKG::kinect-azure-sensor-sdk
    CONAN_PKG::kinect-azure-bodytracking-sdk
    CONAN_PKG::bzip2
    CONAN_PKG::spdlog
    CONAN_PKG::yaml-cpp
    CONAN_PKG::tbb
    CONAN_PKG::jsoncpp
)

target_link_libraries(${TARGET_NAME} PRIVATE ${MAIN_LIBS})

target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        )

install(TARGETS ${TARGET_NAME} 
        DESTINATION ${MAGNUM_BINARY_INSTALL_DIR})

# PYTHON BINDINGS
SET(LIB_NAME mkv_extractor)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)

set(PYTHON_HOME ${Python3_STDLIB})
set(ENV{LD_LIBRARY_PATH} "${ENV_LD_LIBRARY_PATH_ORG}")

SET(PYBIND_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python_interface.cpp")

pybind11_add_module(${LIB_NAME} ${APP_SOURCES} ${APP_HEADERS} ${PYBIND_SOURCES})
target_link_libraries(${LIB_NAME} PUBLIC ${MAIN_LIBS})
target_include_directories(${LIB_NAME} PUBLIC "include")
set_target_properties(${LIB_NAME} PROPERTIES CXX_STANDARD 17)
# 
# # create in correct destination for inplace use within the build directory
set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/python3.8/lib-dynload)
# 
set(ENV{LD_LIBRARY_PATH} "${ENV_LD_LIBRARY_PATH_ORG}")
set(PYTHON_HOME ${Python3_STDLIB})
