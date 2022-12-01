cmake_minimum_required(VERSION 3.13)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (WITH_PCPD)
  project(extract_mkv VERSION 1.0.0 LANGUAGES C CXX CUDA)
else()
  project(extract_mkv VERSION 1.0.0 LANGUAGES C CXX)
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  conan_basic_setup()
else()
  message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

SET(APP_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/thread_pool.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/transformation_helpers.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/extract_mkv_k4a.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/timesync.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/filesystem.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/h264.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/export.h"
    )

SET(PCPD_HEADERS)

SET(APP_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/transformation_helpers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/extract_mkv_k4a.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/timesync.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/h264.cpp"
    )


SET(PCPD_HEADERS)

SET(TARGET_NAME extract_mkv)

SET(PCPD_LIBS)

if(WITH_PCPD)
    if (NOT DEFINED PCPD_DIR)
        MESSAGE(FATAL_ERROR "PCPD_DIR needs to be set")
        # NOTE: PCPD_DIR needs to be set via command line
        # set(PCPD_DIR /data/develop/pcpd/)
    else()
      message("Linking PCPD...")
    endif ()
    if(NOT DEFINED PCPD_BUILD_DIR)
      set(PCPD_BUILD_DIR ${PCPD_DIR}/build/)
    endif()

    add_compile_definitions(WITH_PCPD)

    enable_language(CUDA)
    cmake_policy(SET CMP0104 NEW)
    cmake_policy(SET CMP0105 NEW)
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
    set(BUILD_SHARED_LIBS YES)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    set(CMAKE_CXX_STANDARD 17)
    add_definitions("-std=c++17")
    find_package(Threads REQUIRED)
    find_package(CUDA REQUIRED)
    find_package(NVRTC REQUIRED)

    SET(PCPD_CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIB})

    include_directories(${PCPD_DIR}/include)
    include_directories(${PCPD_DIR}/include/pcpd)
    include_directories(${PCPD_DIR}/include/pcpd/artekmed)
    include_directories(${PCPD_DIR}/include/pcpd/artekmed/schema)
    include_directories("${CUDA_INCLUDE_DIRS}")
    include_directories(${PCPD_BUILD_DIR}/include)
    link_directories(${PCPD_BUILD_DIR}/lib)

    add_library(ringbuffer STATIC IMPORTED)
    set_property(TARGET ringbuffer PROPERTY IMPORTED_LOCATION
      ${PCPD_BUILD_DIR}/lib/libringbuffer.so)

    message("Directories linked...")

    SET(PCPD_LIBS
        Threads::Threads
        ringbuffer
        kj
        capnp
        pcpd
        pcpd_cuda
        ${PCPD_CUDA_LIBRARIES}
    )

SET(PCPD_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/pcpd_file_exporter.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/extract_mkv/kinect4azure_capture_wrapper.h"
    )

SET(PCPD_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pcpd_file_exporter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kinect4azure_capture_wrapper.cpp"
    )
endif ()

add_library(${TARGET_NAME} ${APP_SOURCES} ${APP_HEADERS} ${PCPD_SOURCES} ${PCPD_HEADERS})

# build main lib
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
SET(MAIN_LIBS
    ${CONAN_LIBS}
    stdc++fs
)
target_link_libraries(${TARGET_NAME} PUBLIC ${MAIN_LIBS} ${PCPD_LIBS})

target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        )

install(TARGETS ${TARGET_NAME}
        DESTINATION ${MAGNUM_BINARY_INSTALL_DIR})

# executable
set(TARGET_NAME extract)
add_executable(${TARGET_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
SET(MAIN_LIBS
    ${CONAN_LIBS}
    stdc++fs
)
target_link_libraries(${TARGET_NAME} PUBLIC extract_mkv ${MAIN_LIBS} ${PCPD_LIBS})

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

SET(PYBIND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/python_interface.cpp")

pybind11_add_module(${LIB_NAME} ${APP_SOURCES} ${APP_HEADERS} ${PYBIND_SOURCES} ${PCPD_SOURCES} ${PCPD_HEADERS})
target_include_directories(${LIB_NAME} PRIVATE "include")
target_link_libraries(${LIB_NAME} PRIVATE ${MAIN_LIBS} ${PCPD_LIBS})
set_target_properties(${LIB_NAME} PROPERTIES CXX_STANDARD 17)
# # create in correct destination for inplace use within the build directory
set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/python3.8/lib-dynload)

set(ENV{LD_LIBRARY_PATH} "${ENV_LD_LIBRARY_PATH_ORG}")
set(PYTHON_HOME ${Python3_STDLIB})

add_subdirectory("tests/")
