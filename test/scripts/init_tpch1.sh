#!/bin/bash
mysql -h:: -urateup -p'123456' -e "drop database if exists scale_1"
mysql -h:: -urateup -p'123456' -e "create database scale_1"
mysql -h:: -urateup -p'123456' scale_1 < test_resources/schema/tpch/create_table.sql
./load_tpch_data.sh /data/tpch/scale_1 scale_1
