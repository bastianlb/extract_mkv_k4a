# syntax=docker/dockerfile:1.0.0-experimental
ARG registry=artekmedregistry.campar.in.tum.de
ARG from=${registry}/export_mkv_base
FROM ${from} as extract_mkv_run

WORKDIR /extract_mkv/

COPY . .

WORKDIR /extract_mkv/build

RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 -DWITH_PCPD=ON -DPCPD_DIR=/pcpd/ && \
    cmake --build . --parallel 12

RUN python3 -m pip install pandas

ENV LD_LIBRARY_PATH /extract_mkv/build/lib:/usr/local/cuda-11.1/lib64/:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}
RUN mkdir /deploy
RUN mkdir -p /data/input /data/export

VOLUME ["/data/export"]
VOLUME ["/deploy"]

ENTRYPOINT ["/deploy/entrypoint.sh"]
