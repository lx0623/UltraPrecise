# Synthesized_workload

 "YourPath" represents the location where the data you generated is stored.

If you want to enable multi-threaded computation, you need to uncomment `MakeCalcXmpNode` in line 45 of `AriesCalcTreeGenerator.cpp`, and if you want to modify the `TPI` (thread per instance) of the computed expression, you need to change line 358 of `AriesCalcTreeGenerator.cpp`.

# TPC-H Q1

In the statistics of TPC-H Q1 performance, we count the execution time under in memory. For FastAPA, Heavy.AI, and RateupDB we load the data onto the GPU earlier by a few query statements.

## Create table

```sql
create table lineitem ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(12,2) not null,
                        l_extendedprice  decimal(12,2) not null,
                        l_discount    decimal(12,2) not null,
                        l_tax         decimal(12,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/tpch/scale_10/lineitem.tbl' into table lineitem fields terminated by '|';

create table lineitem_2 ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(15,2) not null,
                        l_extendedprice  decimal(15,2) not null,
                        l_discount    decimal(15,2) not null,
                        l_tax         decimal(15,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/tpch/scale_10/lineitem.tbl' into table lineitem_2 fields terminated by '|';

create table lineitem_t0_LEN_2 ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(12,2) not null,
                        l_extendedprice  decimal(8,4) not null,
                        l_discount    decimal(3,2) not null,
                        l_tax         decimal(3,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/Query_7_tpch_revision/scale_10/LEN_2/lineitem.tbl' into table lineitem_t0_LEN_2 fields terminated by '|';

create table lineitem_t1_LEN_4 ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(31,2) not null,
                        l_extendedprice  decimal(27,11) not null,
                        l_discount    decimal(3,2) not null,
                        l_tax         decimal(3,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/Query_7_tpch_revision/scale_10/LEN_4/lineitem.tbl' into table lineitem_t1_LEN_4 fields terminated by '|';

create table lineitem_t2_LEN_8 ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(68,2) not null,
                        l_extendedprice  decimal(64,19) not null,
                        l_discount    decimal(3,2) not null,
                        l_tax         decimal(3,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/Query_7_tpch_revision/scale_10/LEN_8/lineitem.tbl' into table lineitem_t2_LEN_8 fields terminated by '|';

create table lineitem_t3_LEN_16 ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(140,2) not null,
                        l_extendedprice  decimal(136,59) not null,
                        l_discount    decimal(3,2) not null,
                        l_tax         decimal(3,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/Query_7_tpch_revision/scale_10/LEN_16/lineitem.tbl' into table lineitem_t3_LEN_16 fields terminated by '|';

create table lineitem_t4_LEN_32 ( l_orderkey    integer not null,
                        l_partkey     integer not null,
                        l_suppkey     integer not null,
                        l_linenumber  integer not null,
                        l_quantity    decimal(284,2) not null,
                        l_extendedprice  decimal(280,109) not null,
                        l_discount    decimal(3,2) not null,
                        l_tax         decimal(3,2) not null,
                        l_returnflag  char(1) not null,
                        l_linestatus  char(1) not null,
                        l_shipdate    date not null,
                        l_commitdate  date not null,
                        l_receiptdate date not null,
                        l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
                        l_shipmode     char(10) not null encoding bytedict as l_shipmode,
                        l_comment      varchar(44) not null);
load data infile 'YourPath/Query_7_tpch_revision/scale_10/LEN_32/lineitem.tbl' into table lineitem_t4_LEN_32 fields terminated by '|';
```

## Query

```sql
use scale_10;

select   count(distinct l_returnflag)   from lineitem;

select   count(distinct l_linestatus)   from lineitem;

select   max(l_quantity*l_extendedprice+l_discount*l_tax),   max(l_shipdate)
from lineitem;

select     l_returnflag,     l_linestatus,     sum(l_quantity) as sum_qty,     sum(l_extendedprice) as sum_base_price,     sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,     sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,     avg(l_quantity) as avg_qty,     avg(l_extendedprice) as avg_price,     avg(l_discount) as avg_disc,     count(*) as count_order from     lineitem where     l_shipdate <= date '1998-12-01' - interval '90' day group by     l_returnflag,     l_linestatus order by     l_returnflag,     l_linestatus;

# lineitem_2/lineitem_t0_LEN_2/lineitem_t1_LEN_4/lineitem_t2_LEN_8/lineitem_t3_LEN_16/lineitem_t4_LEN_32
```

# RSA

## Create table

```sql
create table rsa_t1_LEN_4 ( col1 decimal(17,0) not null
                );
load data infile 'YourPath/Query_8_RSA_data/LEN_4/data.tbl' into table rsa_t1_LEN_4 fields terminated by '|';

create table rsa_t2_LEN_8 ( col1 decimal(35,0) not null
                );
load data infile 'YourPath/Query_8_RSA_data/LEN_8/data.tbl' into table rsa_t2_LEN_8 fields terminated by '|';

create table rsa_t3_LEN_16 ( col1 decimal(71,0) not null
                );
load data infile 'YourPath/Query_8_RSA_data/LEN_16/data.tbl' into table rsa_t3_LEN_16 fields terminated by '|';         

create table rsa_t4_LEN_32 ( col1 decimal(143,0) not null
                );
load data infile 'YourPath/Query_8_RSA_data/LEN_32/data.tbl' into table rsa_t4_LEN_32 fields terminated by '|'; 
```

## Query

```sql
select col1 * col1 % 514679589732349351 * col1 % 514679589732349351
from rsa_t1_LEN_4;

select col1 * col1 % 331972734823750661745273085756461571 * col1 % 331972734823750661745273085756461571
from rsa_t2_LEN_8;

select col1 * col1 % 273420351015258185841522860354335753665897876941082396543621156533322691 * col1 % 273420351015258185841522860354335753665897876941082396543621156533322691
from rsa_t3_LEN_16;

select col1 * col1 % 168687844420361814218965240578418106558146045032543735951056173311706452511006631405498089801144629119144418134762919565200448008667563331468801 * col1 % 168687844420361814218965240578418106558146045032543735951056173311706452511006631405498089801144629119144418134762919565200448008667563331468801
from rsa_t4_LEN_32;
```

# Sin

## Create table

```sql
create table sin_one_ans( col1 decimal(9,8) not null,
                      ans decimal(287,286) not null);
load data infile 'YourPath/Query_9_SIN_data/one/ans.tbl' into table sin_one_ans fields terminated by '|';

create table sin_two_ans( col1 decimal(9,8) not null,
                      ans decimal(287,286) not null);
load data infile 'YourPath/Query_9_SIN_data/two/ans.tbl' into table sin_two_ans fields terminated by '|';

create table sin_thr_ans( col1 decimal(9,8) not null,
                      ans decimal(287,286) not null);
load data infile 'YourPath/Query_9_SIN_data/thr/ans.tbl' into table sin_thr_ans fields terminated by '|';
```

## Query

```sql
select 	col1-col1*col1*col1/6 from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+    col1*col1*col1*col1*col1*col1*col1*col1*col1/362880 
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+    col1*col1*col1*col1*col1*col1*col1*col1*col1/362880-col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/39916800
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+     col1*col1*col1*col1*col1*col1*col1*col1*col1/362880-col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/39916800+    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/6227020800
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+  col1*col1*col1*col1*col1*col1*col1*col1*col1/362880-col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/39916800+ col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/6227020800-    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/1307674368000
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+    col1*col1*col1*col1*col1*col1*col1*col1*col1/362880-col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/39916800+    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/6227020800- col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/1307674368000+     col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/355687428096000
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+     col1*col1*col1*col1*col1*col1*col1*col1*col1/362880-col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/39916800+    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/6227020800-   col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/1307674368000+    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/355687428096000-     col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/121645100408832000
from sin_one_ans;

select 	col1-col1*col1*col1/6+col1*col1*col1*col1*col1/120-col1*col1*col1*col1*col1*col1*col1/5040+      col1*col1*col1*col1*col1*col1*col1*col1*col1/362880-col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/39916800+    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/6227020800-    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/1307674368000+    col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/355687428096000-   col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/121645100408832000+  col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1*col1/51090942171709440000
from sin_one_ans;

# /sin_two_ans/sin_thr_ans
```