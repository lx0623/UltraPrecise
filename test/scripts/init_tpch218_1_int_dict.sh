#!/bin/bash
DB_NAME="tpch218_1_int_dict"
mysql -h:: -urateup -p'123456' -e "drop database if exists ${DB_NAME}"
mysql -h:: -urateup -p'123456' -e "create database ${DB_NAME}"
mysql -h:: -urateup -p'123456' ${DB_NAME} < test_resources/schema/test/create_table_int_dict.sql
./load_tpch_data.sh /data/tpch218_1 ${DB_NAME}
