#!/bin/bash

executorPath=/home/server/data_for_pie/script/orc-csv-import
sourcePath=/data/tpch/data/scale_5/csv/org
soucePostfix=.tbl
targetPath=/data/tpch/data/scale_5/csv/aries
targetPostfix=
delimiter='|'
#importMode 1: to ORC file, 2: to Aries file, <0: show data
importMode='2'

#nation
tableName=nation
schema="struct<n_nationkey:not:int,n_name:not:char(25),n_regionkey:not:int,n_comment:nul:varchar(152)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#region
tableName=region
schema="struct<r_regionkey:not:int,r_name:not:char(25),r_comment:nul:varchar(152)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#part
tableName=part
schema="struct<p_partkey:not:int,p_name:not:varchar(55),p_mfgr:not:char(25),p_brand:not:char(10),p_type:not:varchar(25),p_size:not:int,p_container:not:char(10),p_retailprice:not:decimal(15,2),p_comment:not:varchar(23)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#supplier
tableName=supplier
schema="struct<s_suppkey:not:int,s_name:not:char(25),s_address:not:varchar(40),s_nationkey:not:int,s_phone:not:char(15),s_acctbal:not:decimal(15,2),s_comment:not:varchar(101)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#partsupp
tableName=partsupp
schema="struct<ps_partkey:not:int,ps_suppkey:not:int,ps_availqty:not:int,ps_supplycost:not:decimal(15,2),ps_comment:not:varchar(199)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#customer
tableName=customer
schema="struct<c_custkey:not:int,c_name:not:varchar(25),c_address:not:varchar(40),c_nationkey:not:int,c_phone:not:char(15),c_acctbal:not:decimal(15,2),c_mktsegment:not:char(10),c_comment:not:varchar(117)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#orders
tableName=orders
schema="struct<o_orderkey:not:int,o_custkey:not:int,o_orderstatus:not:char(1),o_totalprice:not:decimal(15,2),o_orderdate:not:date,o_orderpriority:not:char(15),o_clerk:not:char(15),o_shippriority:not:int,o_comment:not:varchar(79)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix

#lineitem
tableName=lineitem
schema="struct<l_orderkey:not:int,l_partkey:not:int,l_suppkey:not:int,l_linenumber:not:int,l_quantity:not:decimal(15,2),l_extendedprice:not:decimal(15,2),l_discount:not:decimal(15,2),l_tax:not:decimal(15,2),l_returnflag:not:char(1),l_linestatus:not:char(1),l_shipdate:not:date,l_commitdate:not:date,l_receiptdate:not:date,l_shipinstruct:not:char(25),l_shipmode:not:char(10),l_comment:not:varchar(44)>"
$executorPath --delimiter=$delimiter --mode=$importMode $schema $sourcePath/$tableName$soucePostfix $targetPath/$tableName$targetPostfix
