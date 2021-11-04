##############################################################################
# - Try to find the Cuda NVTOOLSEXT library
# Once done this will define
# LIBNVTOOLSEXT_FOUND - System has LibNVTOOLSEXT
# LIBNVTOOLSEXT_LIBRARY_DIR - The NVTOOLSEXT library dir
# CUDA_NVTOOLSEXT_LIB - The NVTOOLSEXT lib
##############################################################################
find_package(PkgConfig)

find_library(CUDA_NVTOOLSEXT_LIB libnvToolsExt nvToolsExt HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVTOOLSEXT_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibNVTOOLSEXT DEFAULT_MSG CUDA_NVTOOLSEXT_LIB)

mark_as_advanced(CUDA_NVTOOLSEXT_LIB)

if(NOT LIBNVTOOLSEXT_FOUND)
    message(FATAL_ERROR "Cuda NVTOOLSEXT Library not found: Specify the LIBNVTOOLSEXT_LIBRARY_DIR where libnvrtc is located")
endif()