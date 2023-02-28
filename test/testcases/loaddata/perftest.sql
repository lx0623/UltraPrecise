create database if not exists perf_test_tpch_100;
use perf_test_tpch_100;
CREATE TABLE if not exists `partsupp` (
  `ps_partkey` int(10) not null,
  `ps_suppkey` int(10) not null,
  `ps_availqty` int(10) not null,
  `ps_supplycost` decimal(15,2) not null,
  `ps_comment` char(199) not null
);
load data infile '/home/tengjp/git/data/tpch_100/partsupp.tbl' into table partsupp fields terminated by '|';
