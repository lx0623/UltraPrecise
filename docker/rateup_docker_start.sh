#!/bin/bash
usage="Usage: 
    rateup_docker_start.sh [cuda version]
example: 
    run image rateup/aries:rateup-cuda$1-db
    rateup_docker_start.sh 10.1"

extra_volumns=""

if [ -f extra_volumn.env ]; then
    source ./extra_volumn.env
fi

if [ $# -eq 1 ]; then
	docker run --rm -d --gpus all -v /var/rateup:/var/rateup $extra_volumns -p 3306:3306 rateup/aries:rateup-cuda$1-db
else
	echo "$usage"
fi
