#!/bin/bash

set -e
set -x

rm -rf build
mkdir build

pushd build

CONAN_CPU_COUNT=12 CXX=/usr/bin/g++-7 CC=/usr/bin/gcc-7 CUDACXX=/usr/local/cuda-11.5/bin/nvcc conan install .. -s build_type=Debug --build "missing" --build "outdated" -o with_pcpd=True -o opencv:shared=True -o opencv:with_gtk=True -o opencv:with_cuda=True -o opencv:with_ffmpeg=True -o cuda_dev_config:cuda_version=11.5 --profile pcpd
CUDACXX=/usr/local/cuda-11.5/bin/nvcc cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_PCPD=ON -DPCPD_DIR=/data/develop/pcpd/
cmake --build . --parallel 12

popd
