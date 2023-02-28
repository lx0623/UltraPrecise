#!/bin/bash
DB_NAME="new_scale_1"
mysql -h:: -urateup -p'123456' -e "drop database if exists ${DB_NAME}"
mysql -h:: -urateup -p'123456' -e "create database ${DB_NAME}"
mysql -h:: -urateup -p'123456' ${DB_NAME} < test_resources/schema/tpch/create_table.sql
./load_tpch_data.sh /home/server/work/lc/aries/debug/data/data/scale_1 ${DB_NAME}
