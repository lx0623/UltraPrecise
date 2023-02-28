#!/bin/bash
mysql -h:: -urateup -p'123456' -e "drop database if exists scale_1_null"
mysql -h:: -urateup -p'123456' -e "create database scale_1_null"
mysql -h:: -urateup -p'123456' scale_1_null < test_resources/schema/tpch/create_table_with_null_key.sql
./load_tpch_data.sh /data/tpch/scale_1_null scale_1_null
