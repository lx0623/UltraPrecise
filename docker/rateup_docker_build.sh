#!/bin/bash
usage="Usage: 
	rateup_docker_build.sh [cuda version]
example: 
	build rateup db based on cuda 10.1
	rateup_docker_build.sh 10.1"


if [ $# -eq 1 ]; then
    cd ..
	echo "build image rateup/aries:rateup-cuda$1-base"
        docker build --build-arg CUDA_VERSION=$1 -t rateup/aries:rateup-cuda$1-base -f docker/base.dockerfile .
	if [ ! -f .build.config.cmake ]; then
		echo ".build.config.cmake not found, create an empty .build.config.cmake"
		touch .build.config.cmake
	fi
	docker build --build-arg CUDA_VERSION=$1 -t rateup/aries:rateup-cuda$1-db -f docker/rateup.dockerfile .
else
	echo "$usage"
fi

