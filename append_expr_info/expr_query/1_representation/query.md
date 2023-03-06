# Dec_representation

In this experiment, we use the single-threaded operation of Fast-APA. "YourPath" represents the location where the data you generated is stored.

## Query1

### Create table

```sql
create table str_t0_LEN_2 ( col1 decimal(16,2) not null, 
                  col2 decimal(16,2) not null,
                  col3 decimal(16,2) not null
                );
load data infile 'YourPath/Query_1_data_structure/LEN_2/data.tbl' into table str_t0_LEN_2 fields terminated by '|';

create table str_t1_LEN_4 ( col1 decimal(34,2) not null, 
                  col2 decimal(34,2) not null,
                  col3 decimal(34,2) not null
                );
load data infile 'YourPath/Query_1_data_structure/LEN_4/data.tbl' into table str_t1_LEN_4 fields terminated by '|';

create table str_t2_LEN_8 ( col1 decimal(70,2) not null, 
                  col2 decimal(70,2) not null,
                  col3 decimal(70,2) not null
                );
load data infile 'YourPath/Query_1_data_structure/LEN_8/data.tbl' into table str_t2_LEN_8 fields terminated by '|';

create table str_t3_LEN_16 ( col1 decimal(142,2) not null, 
                  col2 decimal(142,2) not null,
                  col3 decimal(142,2) not null
                );
load data infile 'YourPath/Query_1_data_structure/LEN_16/data.tbl' into table str_t3_LEN_16 fields terminated by '|';

create table str_t4_LEN_32 ( col1 decimal(286,2) not null, 
                  col2 decimal(286,2) not null,
                  col3 decimal(286,2) not null
                );
load data infile 'YourPath/Query_1_data_structure/LEN_32/data.tbl' into table str_t4_LEN_32 fields terminated by '|';
```

### Query

```sql
select col1+col2+col3
from str_t0_LEN_2;
# /str_t1_LEN_4/str_t2_LEN_8/str_t3_LEN_16/str_t4_LEN_32
```

## Query2

### Create table

```sql
create table var_t0_LEN_2 ( col1 decimal(6,2) not null, 
                  col2 decimal(6,2) not null, 
                  col3 decimal(6,2) not null, 
                  col4 decimal(6,2) not null, 
                  col5 decimal(15,2) not null,
                  col6 decimal(15,2) not null,
                  col7 decimal(15,2) not null,
                  col8 decimal(15,2) not null
                );
load data infile 'YourPath/Query_2_variable_calculation/LEN_2/data.tbl' into table var_t0_LEN_2 fields terminated by '|';

create table var_t1_LEN_4 ( col1 decimal(6,2) not null, 
                  col2 decimal(6,2) not null, 
                  col3 decimal(6,2) not null, 
                  col4 decimal(6,2) not null, 
                  col5 decimal(33,2) not null,
                  col6 decimal(33,2) not null,
                  col7 decimal(33,2) not null,
                  col8 decimal(33,2) not null
                );
load data infile 'YourPath/Query_2_variable_calculation/LEN_4/data.tbl' into table var_t1_LEN_4 fields terminated by '|';

create table var_t2_LEN_8 ( col1 decimal(6,2) not null, 
                  col2 decimal(6,2) not null, 
                  col3 decimal(6,2) not null, 
                  col4 decimal(6,2) not null, 
                  col5 decimal(69,2) not null,
                  col6 decimal(69,2) not null,
                  col7 decimal(69,2) not null,
                  col8 decimal(69,2) not null
                );
load data infile 'YourPath/Query_2_variable_calculation/LEN_8/data.tbl' into table var_t2_LEN_8 fields terminated by '|';

create table var_t3_LEN_16 ( col1 decimal(6,2) not null, 
                  col2 decimal(6,2) not null, 
                  col3 decimal(6,2) not null, 
                  col4 decimal(6,2) not null, 
                  col5 decimal(141,2) not null,
                  col6 decimal(141,2) not null,
                  col7 decimal(141,2) not null,
                  col8 decimal(141,2) not null
                );
load data infile 'YourPath/Query_2_variable_calculation/LEN_16/data.tbl' into table var_t3_LEN_16 fields terminated by '|';       

create table var_t4_LEN_32 ( col1 decimal(6,2) not null, 
                  col2 decimal(6,2) not null, 
                  col3 decimal(6,2) not null, 
                  col4 decimal(6,2) not null, 
                  col5 decimal(285,2) not null,
                  col6 decimal(285,2) not null,
                  col7 decimal(285,2) not null,
                  col8 decimal(285,2) not null
                );
load data infile 'YourPath/Query_2_variable_calculation/LEN_32/data.tbl' into table var_t4_LEN_32 fields terminated by '|'; 
```

### Query

```sql
select 
  col1+col2+col3+col4,
  col5+col6+col7+col8
from var_t0_LEN_2;
# /var_t1_LEN_4/var_t2_LEN_8/var_t2_LEN_8/var_t4_LEN_32
```