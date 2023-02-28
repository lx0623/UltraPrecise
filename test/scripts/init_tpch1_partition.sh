#!/bin/bash
DB_NAME="tpch_1_partition"
mysql -h:: -urateup -p'123456' -e "drop database if exists ${DB_NAME}"
mysql -h:: -urateup -p'123456' -e "create database ${DB_NAME}"
mysql -h:: -urateup -p'123456' ${DB_NAME} < test_resources/schema/test/create_table_partition.sql
./load_tpch_data.sh /data/tpch/scale_1 ${DB_NAME}
