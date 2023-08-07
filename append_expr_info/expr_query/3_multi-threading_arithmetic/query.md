# Mult-thread Arithmetic

 "YourPath" represents the location where the data you generated is stored.

# expr_evaluation

If you want to enable multi-threaded computation, you need to uncomment `MakeCalcXmpNode` in line 45 of `AriesCalcTreeGenerator.cpp`, and if you want to modify the `TPI` (thread per instance) of the computed expression, you need to change line 358 of `AriesCalcTreeGenerator.cpp`.

## add_or_sub

### Create table

```sql
create table sig_add_t1_LEN_4 ( col1 decimal(35,10) not null, 
                  col2 decimal(35,10) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_4/data.tbl' into table sig_add_t1_LEN_4 fields terminated by '|';

create table sig_add_t2_LEN_8 ( col1 decimal(71,20) not null, 
                  col2 decimal(71,20) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_8/data.tbl' into table sig_add_t2_LEN_8 fields terminated by '|';

create table sig_add_t3_LEN_16 ( col1 decimal(143,30) not null, 
                  col2 decimal(143,30) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_16/data.tbl' into table sig_add_t3_LEN_16 fields terminated by '|';

create table sig_add_t4_LEN_32 ( col1 decimal(287,40) not null, 
                  col2 decimal(287,40) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_32/data.tbl' into table sig_add_t4_LEN_32 fields terminated by '|';
```

### Query

```sql
select 
  col1+col2
from sig_add_t1_LEN_4;
# /sig_add_t2_LEN_8/sig_add_t3_LEN_16/sig_add_t4_LEN_32

select 
  col1-col2
from sig_add_t1_LEN_4;
# /sig_add_t2_LEN_8/sig_add_t3_LEN_16/sig_add_t4_LEN_32
```

## mul

### Create table

```sql
create table sig_mul_t1_LEN_4 ( col1 decimal(20,10) not null, 
                  col2 decimal(16,11) not null
                );
load data infile 'YourPath/Query_5_single_calculation/mul/LEN_4/data.tbl' into table sig_mul_t1_LEN_4 fields terminated by '|';

create table sig_mul_t2_LEN_8 ( col1 decimal(56,20) not null, 
                  col2 decimal(16,11) not null
                );
load data infile 'YourPath/Query_5_single_calculation/mul/LEN_8/data.tbl' into table sig_mul_t2_LEN_8 fields terminated by '|';

create table sig_mul_t3_LEN_16 ( col1 decimal(128,30) not null, 
                  col2 decimal(16,11) not null
                );
load data infile 'YourPath/Query_5_single_calculation/mul/LEN_16/data.tbl' into table sig_mul_t3_LEN_16 fields terminated by '|';

create table sig_mul_t4_LEN_32 ( col1 decimal(272,40) not null, 
                  col2 decimal(16,11) not null
                );
load data infile 'YourPath/Query_5_single_calculation/mul/LEN_32/data.tbl' into table sig_mul_t4_LEN_32 fields terminated by '|';
```

### Query

```sql
select 
  col1*col2
from sig_mul_t1_LEN_4;
# /sig_mul_t2_LEN_8/sig_mul_t3_LEN_16/sig_mul_t4_LEN_32
```

## div

### Create table

```sql
create table sig_div_t1_LEN_4 ( col1 decimal(18,1) not null, 
                  col2 decimal(18,14) not null
                );
load data infile 'YourPath/Query_5_single_calculation/div/LEN_4/data.tbl' into table sig_div_t1_LEN_4 fields terminated by '|';

create table sig_div_t2_LEN_8 ( col1 decimal(54,21) not null, 
                  col2 decimal(18,14) not null
                );
load data infile 'YourPath/Query_5_single_calculation/div/LEN_8/data.tbl' into table sig_div_t2_LEN_8 fields terminated by '|';

create table sig_div_t3_LEN_16 ( col1 decimal(126,41) not null, 
                  col2 decimal(18,14) not null
                );
load data infile 'YourPath/Query_5_single_calculation/div/LEN_16/data.tbl' into table sig_div_t3_LEN_16 fields terminated by '|';

create table sig_div_t4_LEN_32 ( col1 decimal(270,81) not null, 
                  col2 decimal(18,14) not null
                );
load data infile 'YourPath/Query_5_single_calculation/div/LEN_32/data.tbl' into table sig_div_t4_LEN_32 fields terminated by '|';
```

### Query

```sql
select 
  col1/col2
from sig_div_t1_LEN_4;
# /sig_div_t2_LEN_8/sig_div_t3_LEN_16/sig_div_t4_LEN_32
```

# aggregation

The aggregation of Decimal type in UltraPrecise is done by multi-threaded calculation, and the size of `TPI` (thread per instance) can be modified by modifying the code in `decimal.h` in line 61.

## Create table

```sql
create table agg_2_scale_t0_LEN_2( col1 decimal(11,7) not null);
load data infile 'YourPath/Query_6_aggregation_2_scale/LEN_2/data.tbl' into table agg_2_scale_t0_LEN_2 fields terminated by '|';

create table agg_2_scale_t1_LEN_4( col1 decimal(29,11) not null);
load data infile 'YourPath/Query_6_aggregation_2_scale/LEN_4/data.tbl' into table agg_2_scale_t1_LEN_4 fields terminated by '|';

create table agg_2_scale_t2_LEN_8( col1 decimal(65,31) not null);
load data infile 'YourPath/Query_6_aggregation_2_scale/LEN_8/data.tbl' into table agg_2_scale_t2_LEN_8 fields terminated by '|';

create table agg_2_scale_t3_LEN_16( col1 decimal(137,51) not null);
load data infile 'YourPath/Query_6_aggregation_2_scale/LEN_16/data.tbl' into table agg_2_scale_t3_LEN_16 fields terminated by '|';

create table agg_2_scale_t4_LEN_32( col1 decimal(281,101) not null);
load data infile 'YourPath/Query_6_aggregation_2_scale/LEN_32/data.tbl' into table agg_2_scale_t4_LEN_32 fields terminated by '|';
```

## Query

```sql
select 
  sum(col1)
from agg_2_scale_t0_LEN_2;
# agg_2_scale_t1_LEN_4/agg_2_scale_t2_LEN_8/agg_2_scale_t3_LEN_16/agg_2_scale_t4_LEN_32
```