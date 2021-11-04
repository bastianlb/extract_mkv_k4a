#!/bin/bash

set -e
set -x

rm -rf build
mkdir build

pushd build

CONAN_CPU_COUNT=12 CXX=/usr/bin/g++-7 CC=/usr/bin/gcc-7 CUDACXX=/usr/local/cuda-11.1/bin/nvcc conan install .. -s build_type=Debug --build "missing" --build "outdated" -o with_pcpd=True -o opencv:shared=True -o opencv:with_gtk=True -o opencv:with_cuda=True
CUDACXX=/usr/local/cuda-11.1/bin/nvcc cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_PCPD=ON -DPCPD_WS_DIR=/data/develop/pcp_workspace
cmake --build . --parallel 12

popd
