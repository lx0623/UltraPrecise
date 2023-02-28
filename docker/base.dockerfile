ARG CUDA_VERSION
#FROM nvidia/cuda:10.1-devel-ubuntu18.04
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04
LABEL maintainer "RATEUP CORPORATION <dev@ratup.com.cn>"

RUN echo "deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse              \\n\
        deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse       \\n\
        deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse        \\n\
        deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse       \\n\
        deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse      \\n\
        deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse            \\n\
        deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse   \\n\
        deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse    \\n\
        deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse   \\n\
        deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse" >  /etc/apt/sources.list


RUN rm /etc/apt/sources.list.d/*
RUN apt-get update
RUN apt-get install  -y --no-install-recommends \
        g++ \
        git \
        uuid-dev \
        libcanberra-gtk-module \
        libevent-dev \
        liblzma-dev \
        binutils-dev \
        libjemalloc-dev \
        libssl-dev \
        pkg-config \
        libaio-dev \
        libunwind8-dev \
        libelf-dev \
        libdwarf-dev \
        libgoogle-glog-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-serialization-dev \
        googletest  

WORKDIR /third_party           
RUN git clone https://gitee.com/mirrors/CMake.git && \
    cd /third_party/CMake/ &&\
    git checkout v3.18.4 &&\
    mkdir build && cd build &&\
    ../bootstrap -- -DCMAKE_USE_OPENSSL=OFF && make -j$(nproc) && make install &&\
    rm -rf /third_party/CMake 
RUN ln -s /usr/local/bin/cmake /usr/bin/cmake
RUN cd /usr/src/googletest && mkdir build && cd build && cmake .. && \
    make -j$(nproc) && make install && rm -rf /usr/src/googletest

EXPOSE 3306

RUN echo "/usr/local/cuda/compat/" >> /etc/ld.so.conf.d/cuda.conf
RUN ldconfig
