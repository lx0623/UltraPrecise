# Opt_expression

In this experiment, we use the single-threaded operation of Fast-APA. "YourPath" represents the location where the data you generated is stored.

We compare the system by utilizing the optimization policy with the system that does not utilize the optimization policy. If you want to cancel the corresponding optimization policy, you can comment on the corresponding code in `AriesEngineShell.cpp`.

## alignment schdeuling

### Create table

```sql
create table sch_t0_LEN_2_mlt ( col1 decimal(2,1) not null, 
                  col2 decimal(17,11) not null
                );
load data infile 'YourPath/Query_3_alignment_scheduling_mult/LEN_2/data.tbl' into table sch_t0_LEN_2_mlt fields terminated by '|';

create table sch_t1_LEN_4_mlt ( col1 decimal(20,1) not null, 
                  col2 decimal(18,11) not null
                );
load data infile 'YourPath/Query_3_alignment_scheduling_mult/LEN_4/data.tbl' into table sch_t1_LEN_4_mlt fields terminated by '|';

create table sch_t2_LEN_8_mlt ( col1 decimal(56,1) not null, 
                  col2 decimal(18,11) not null
                );
load data infile 'YourPath/Query_3_alignment_scheduling_mult/LEN_8/data.tbl' into table sch_t2_LEN_8_mlt fields terminated by '|';

create table sch_t3_LEN_16_mlt ( col1 decimal(128,1) not null, 
                  col2 decimal(18,11) not null
                );
load data infile 'YourPath/Query_3_alignment_scheduling_mult/LEN_16/data.tbl' into table sch_t3_LEN_16_mlt fields terminated by '|'; 

create table sch_t4_LEN_32_mlt ( col1 decimal(272,1) not null, 
                  col2 decimal(18,11) not null
                );
load data infile 'YourPath/Query_3_alignment_scheduling_mult/LEN_32/data.tbl' into table sch_t4_LEN_32_mlt fields terminated by '|';
```

### Query

```sql
select col1+col2+col1
from sch_t0_LEN_2_mlt
# /sch_t1_LEN_4_mlt/sch_t2_LEN_8_mlt/sch_t3_LEN_16_mlt/sch_t4_LEN_32_mlt;

select col1+col2+col1+col1+col1
from sch_t0_LEN_2_mlt
# /sch_t1_LEN_4_mlt/sch_t2_LEN_8_mlt/sch_t3_LEN_16_mlt/sch_t4_LEN_32_mlt;

select col1+col2+col1+col1+col1+col1+col1
from sch_t0_LEN_2_mlt
# /sch_t1_LEN_4_mlt/sch_t2_LEN_8_mlt/sch_t3_LEN_16_mlt/sch_t4_LEN_32_mlt;
```

## constant_construction

### Create table

```sql
create table const_t0_LEN_2 ( col1 decimal(17,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_2/data.tbl' into table const_t0_LEN_2 fields terminated by '|';

create table const_t1_LEN_4 ( col1 decimal(35,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_4/data.tbl' into table const_t1_LEN_4 fields terminated by '|';

create table const_t2_LEN_8 ( col1 decimal(71,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_8/data.tbl' into table const_t2_LEN_8 fields terminated by '|';

create table const_t3_LEN_16 ( col1 decimal(143,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_16/data.tbl' into table const_t3_LEN_16 fields terminated by '|';         

create table const_t4_LEN_32 ( col1 decimal(287,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_32/data.tbl' into table const_t4_LEN_32 fields terminated by '|';      
```

### Query

```sql
select 1+col1
from const_t0_LEN_2;
# const_t1_LEN_4/const_t2_LEN_8/const_t3_LEN_16/const_t4_LEN_32;
```

## constant_pre_calculation

### Create table

```sql
create table calc_const_t0_LEN_2 ( col1 decimal(17,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_2/data.tbl' into table calc_const_t0_LEN_2 fields terminated by '|';

create table calc_const_t1_LEN_4 ( col1 decimal(35,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_4/data.tbl' into table calc_const_t1_LEN_4 fields terminated by '|';

create table calc_const_t2_LEN_8 ( col1 decimal(71,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_8/data.tbl' into table calc_const_t2_LEN_8 fields terminated by '|';

create table calc_const_t3_LEN_16 ( col1 decimal(143,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_16/data.tbl' into table calc_const_t3_LEN_16 fields terminated by '|';         

create table calc_const_t4_LEN_32 ( col1 decimal(287,10) not null
                );
load data infile 'YourPath/Query_4_constant_calc/LEN_32/data.tbl' into table calc_const_t4_LEN_32 fields terminated by '|';  
```

```sql
create table calc_const_mul_t0_LEN_2 ( col1 decimal(17,10) not null, 
                  col2 decimal(17,10) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_2/data.tbl' into table calc_const_mul_t0_LEN_2 fields terminated by '|';

create table calc_const_mul_t1_LEN_4 ( col1 decimal(35,10) not null, 
                  col2 decimal(35,10) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_4/data.tbl' into table calc_const_mul_t1_LEN_4 fields terminated by '|';

create table calc_const_mul_t2_LEN_8 ( col1 decimal(71,20) not null, 
                  col2 decimal(71,20) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_8/data.tbl' into table calc_const_mul_t2_LEN_8 fields terminated by '|';

create table calc_const_mul_t3_LEN_16 ( col1 decimal(143,30) not null, 
                  col2 decimal(143,30) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_16/data.tbl' into table calc_const_mul_t3_LEN_16 fields terminated by '|';

create table calc_const_mul_t4_LEN_32 ( col1 decimal(287,40) not null, 
                  col2 decimal(287,40) not null
                );
load data infile 'YourPath/Query_5_single_calculation/add_or_sub/LEN_32/data.tbl' into table calc_const_mul_t4_LEN_32 fields terminated by '|';
```

### Query

```sql
select 
  1+col1+2+11
from calc_const_t0_LEN_2;
# /calc_const_t1_LEN_4/calc_const_t2_LEN_8/calc_const_t3_LEN_16/calc_const_t4_LEN_32;

select 
  1+col1+2-3
from calc_const_t0_LEN_2
# /calc_const_t1_LEN_4/calc_const_t2_LEN_8/calc_const_t3_LEN_16/calc_const_t4_LEN_32;

select 
  4*(col1+col2)*0.25
from calc_const_mul_t0_LEN_2;
# /calc_const_mul_t1_LEN_4/calc_const_mul_t2_LEN_8/calc_const_mul_t3_LEN_16/calc_const_mul_t4_LEN_32;
```