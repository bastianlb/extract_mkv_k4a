# syntax=docker/dockerfile:1.0.0-experimental
ARG registry=artekmedregistry.campar.in.tum.de
ARG from=${registry}/artekmed_cn_base:v0.3-cuda11
FROM ${from} as pcpd_base

WORKDIR /pcpd

# ignore failed exit code due to unsigned package
RUN apt update || :
RUN apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev libva-dev libvdpau-dev
RUN python3 -m pip install conan --upgrade

# TODO: --mount=type=cache can avoid cache bust? does it work with ssh?
# https://stackoverflow.com/questions/55003297/how-to-get-git-clone-to-play-nice-with-docker-cache
RUN --mount=type=ssh,id=github,required cd /pcpd && \
    git clone -b lennart-mkv-test git@github.com:TUM-CAMP-NARVIS/pcpd.git /pcpd/ && \
    cd /pcpd/ && \
    git pull

RUN git --no-pager log --decorate=short --pretty=oneline -n5

RUN mkdir build && \
    cd build && \
    export CONAN_CPU_COUNT=$(nproc) && export CXX=/usr/bin/g++-7 && export CC=/usr/bin/gcc-7 && export CUDACXX=/usr/local/cuda-11.1/bin/nvcc && \
    conan install .. -s build_type=Release -o pcpd:with_python=True -o python_dev_config:python=/usr/bin/python3 -o opencv:shared=True --build "*" && \
    CUDACXX=/usr/local/cuda-11.1/bin/nvcc cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 && \
    cmake --build . --parallel 12 && \
    conan remove "*/*@*/*" --builds --force

# enables selectively building with --target
# only needs to be built if dependencies change
FROM pcpd_base as extract_mkv_base

WORKDIR /extract_mkv

# note: this busts the cache :(..
# however, it respects the .dockerignore file
COPY . .

RUN mkdir build && \
    cd build && \
    export CONAN_CPU_COUNT=$(nproc) && export CXX=/usr/bin/g++-7 && export CC=/usr/bin/gcc-7 && export CUDACXX=/usr/local/cuda-11.1/bin/nvcc && \
    conan install .. -s build_type=Release --build "missing" --build "outdated" -o with_pcpd=True -o opencv:shared=True -o opencv:with_gtk=True -o opencv:with_cuda=True -o opencv:with_ffmpeg=True && \
    conan remove "*/*@*/*" --builds --force
