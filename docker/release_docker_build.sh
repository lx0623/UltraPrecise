#!/bin/bash
###########################################################
#	编译rateup并构建相应的Docker镜像
#	
###########################################################
set -e
usage="Usage: 
	release_docker_build.sh [rateup version] [cuda version] [compute Capability]
example: 
	build 1.0.0 version of rateup db, based on cuda 11.1 and compute capability 6.1
	release_docker_build.sh 1.0.0 11.1 6.1"


if [ ! $# -eq 3 ]; then
	echo "$usage"
	exit
fi


echo -e "`date`
============================================================
1/3, make rateup
============================================================
"

# sleep 3
cd ..
if [ ! -d "build" ]; then
	mkdir "build"
fi
compute=$3
if [ $3 == "6.0" ]; then
	compute="60"
elif [ $3 == "6.1" ]; then
	compute="61"
elif [ $3 == "7.0" ]; then
	compute="70"
elif [ $3 == "7.5" ]; then
	compute="75"
else
	echo "supported compute capability is 6.0/6.1/7.0/7.5, no $3"
	exit 1
fi
echo "set_property(GLOBAL PROPERTY CUDA_API_LEVEL \"${compute}\")" > .build.config.cmake
cd "build/"
cmake -DBUILD_TEST=OFF .. && make -j2

echo -e "`date`
====================================================================
2/3, build image rateup/rateup:${1}-cuda${2}-compute${3}-ubuntu18.04
====================================================================
"
# sleep 3
cd ..
cp src/server/mysql/share/english/errmsg.sys build/
docker build --build-arg CUDA_VERSION=${2} \
-t rateup/rateup:${1}-cuda${2}-compute${3}-ubuntu18.04 \
-f docker/release.dockerfile ./build \
| tee docker/build.log

line=$(tail -n1 docker/build.log)
if [[ $line == Successfully* ]]; then
	echo "build image SUCCESSFUL"
else
	echo "build image FAILED, check docker/build.log for more information"
	exit
fi


echo -e "`date`
============================================================
3/3, test
============================================================
"
# set -x
rateup_home=`pwd`/docker/initialize/
sudo rm -rf $rateup_home
mkdir $rateup_home
build/rateup --datadir=$rateup_home --initialize << EOF
123456
123456
EOF

container_id=$(docker run -d --rm --gpus all \
-p 3306:3306 \
-v $rateup_home:/var/rateup \
-v /data/tpch/tpch218_1/:/data/tpch/tpch218_1/ \
${line:20:100} \
rateup &)

sleep 5

./partition_test_.sh | tee -a docker/build.log

docker stop $container_id

echo -e "`date`
============================================================
test image SUCCESSFUL
${line:20:100}
============================================================
"