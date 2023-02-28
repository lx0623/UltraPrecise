#!/bin/bash
./rateup &

sleep 3

./init_tpch1.sh
./init_tpch1_null.sh

./init_tpch1_partition_nodict.sh
./init_tpch1_partition.sh

./init_tpch218_1.sh
./init_tpch218_1_partition.sh
./init_tpch218_1_partition_nodict.sh
./init_tpch218_1_partition_char40.sh

pkill rateup

sleep 3

./rateup --gtest_filter=UT*:tpch_1.*:scale1_null.*:smoke_1.*:bugfix*
./rateup --gtest_filter=tpch_1.* tpch_1_partition
./rateup --gtest_filter=tpch_1.* tpch_1_partition_nodict

./rateup --gtest_filter=tpch218_1.q*
./rateup --gtest_filter=tpch218_1.q* tpch218_1_partition
./rateup --gtest_filter=tpch218_1.q* tpch218_1_partition_nodict
./rateup --gtest_filter=tpch218_1.q* tpch218_1_partition_char40
