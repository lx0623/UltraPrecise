ARG CUDA_VERSION
FROM rateup/aries:rateup-cuda$CUDA_VERSION-base

WORKDIR /rateup
COPY src /rateup/src
COPY CMakeLists.txt /rateup/CMakeLists.txt
COPY src/parserv2/FlexLexer.h /usr/local/include/FlexLexer.h
COPY dependency/cub-1.8.0/cub /usr/local/include/cub
COPY genversion.sh /rateup/genversion.sh
COPY .build.config.cmake /rateup/.build.config.cmake
WORKDIR /rateup/build
RUN cmake  -DBUILD_TEST=OFF .. 
RUN make -j$(nproc) rateup && make install && cd / && rm -rf /rateup/build

EXPOSE 3306

RUN echo "/usr/local/cuda-$CUDA_VERSION/compat/" >> /etc/ld.so.conf.d/cuda.conf
RUN ldconfig

WORKDIR /usr/sbin
CMD rateup
