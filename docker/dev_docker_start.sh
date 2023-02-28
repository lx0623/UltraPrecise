cd ..
docker run -d --rm --gpus all -v /var/rateup:/var/rateup -v $(pwd):/rateup -v /usr/local/include/cub/:/usr/local/include/cub/ -p 3308:3306 -p 10022:22 rateup/aries:dev  
