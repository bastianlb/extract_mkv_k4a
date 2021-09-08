#!/bin/bash

set -e
set -x

rm -rf build
mkdir build

pushd build

CONAN_CPU_COUNT=12 conan install .. -s build_type=Debug --build "missing" --build "outdated" --profile dev
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --parallel 12

popd
