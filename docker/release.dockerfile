#ubuntu18.04编译好的rateup，放入cuda的runtime镜像，安装依赖后即可运行
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu18.04
LABEL maintainer "RATEUP CORPORATION <dongyang.li@ratup.com.cn>"

COPY rateup errmsg.sys /usr/sbin/
COPY lib/* /usr/sbin/lib/
COPY include/* /usr/sbin/include/
RUN apt update && apt install -y --no-install-recommends \
    libunwind8=1.2.1-8