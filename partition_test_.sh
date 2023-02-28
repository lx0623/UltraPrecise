set -ex

export MYSQL_PWD=123456
export MYSQL_HOST=127.0.0.1
data_dir="/data/tpch/tpch218_1"
database="tpch218_1_partition_char16"
load=true
test=true
result_dir="${database}_result"
answer_dir="tpch218_1_result"















function getMicroTiming(){ 
    start=$1
    end=$2
    start_s=$(echo $start | cut -d '.' -f 1)
    start_ns=$(echo $start | cut -d '.' -f 2)
    end_s=$(echo $end | cut -d '.' -f 1)
    end_ns=$(echo $end | cut -d '.' -f 2)
    time=$(( ( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) ))
    echo "$time ms"
} 
echo "run $database test"
date
if $load; then
mysql -urateup  -e "drop database if exists $database"
mysql -urateup  -e "create database if not exists $database"

mysql -urateup  -D$database -e "
create table region  ( r_regionkey  integer not null primary key,
                       r_name       char(25) not null,
                       r_comment    varchar(152));

create table nation  ( n_nationkey  integer not null primary key,
                       n_name       char(25) not null,
                       n_regionkey  integer not null,
                       n_comment    varchar(152) );

create table part  ( p_partkey     integer not null primary key,
                     p_name        varchar(55) not null,
                     p_mfgr        char(25) not null,
                     p_brand       char(10) not null encoding bytedict as p_brand,
                     p_type        varchar(25) not null encoding shortdict as p_type,
                     p_size        integer not null,
                     p_container   char(10) not null encoding bytedict as p_container,
                     p_retailprice decimal(12,2) not null,
                     p_comment     varchar(23) not null );

create table supplier ( s_suppkey     integer not null primary key,
                        s_name        char(25) not null,
                        s_address     varchar(40) not null,
                        s_nationkey   integer not null,
                        s_phone       char(15) not null,
                        s_acctbal     decimal(12,2) not null,
                        s_comment     varchar(101) not null );

create table partsupp ( ps_partkey     integer not null,
                        ps_suppkey     integer not null,
                        ps_availqty    integer not null,
                        ps_supplycost  decimal(12,2)  not null,
                        ps_comment     varchar(199) not null,
                        primary key ( ps_partkey, ps_suppkey ) );

create table customer ( c_custkey     integer not null primary key,
                        c_name        varchar(25) not null,
                        c_address     varchar(40) not null,
                        c_nationkey   integer not null,
                        c_phone       char(15) not null,
                        c_acctbal     decimal(12,2)   not null,
                        c_mktsegment  char(10) not null encoding bytedict as c_mktsegment,
                        c_comment     varchar(117) not null );

create table orders  ( o_orderkey       char(16) not null primary key,
                       o_custkey        integer not null,
                       o_orderstatus    char(1) not null,
                       o_totalprice     decimal(12,2) not null,
                       o_orderdate      date not null,
                       o_orderpriority  char(15) not null encoding bytedict as o_orderpriority,  
                       o_clerk          char(15) not null, 
                       o_shippriority   integer not null,
                       o_comment        varchar(79) not null );

# create table lineitem (
# l_orderkey    integer not null,
# l_partkey     integer not null,
# l_suppkey     integer not null,
# l_linenumber  integer not null,
# l_quantity    decimal(12,2) not null,
# l_extendedprice  decimal(12,2) not null,
# l_discount    decimal(12,2) not null,
# l_tax         decimal(12,2) not null,
# l_returnflag  char(1) not null,
# l_linestatus  char(1) not null,
# l_shipdate    date not null,
# l_commitdate  date not null,
# l_receiptdate date not null,
# l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
# l_shipmode     char(10) not null encoding bytedict as l_shipmode,
# l_comment      varchar(44) not null ,
# primary key(l_orderkey, l_linenumber)
# );

create table lineitem (
l_orderkey    char(16) not null,
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
l_comment      varchar(44) not null ,
primary key(l_orderkey, l_linenumber)
)
partition by range(l_shipdate) (
partition p0 values less than ('1992-02-01'),
partition p1 values less than ('1994-03-01'),
partition p2 values less than ('1995-03-01'),
partition p3 values less than ('1995-04-01'),
partition p4 values less than ('1995-05-01'),
partition p5 values less than ('1995-06-01'),
partition p6 values less than ('1995-07-01'),
partition p7 values less than ('1995-08-01'),
partition p8 values less than ('1995-09-01'),
partition p9 values less than ('1995-10-01'),
partition p10 values less than ('1995-11-01'),
partition p11 values less than ('1995-12-01'),
partition p12 values less than ('1996-03-01'),
partition p13 values less than ('1996-04-01'),
partition p14 values less than ('1996-05-01'),
partition p15 values less than ('1996-06-01'),
partition p16 values less than ('1996-07-01'),
partition p17 values less than ('1996-08-01'),
partition p18 values less than ('1996-09-01'),
partition p19 values less than ('1996-10-01'),
partition p20 values less than ('1996-11-01'),
partition p21 values less than ('1996-12-01'),
partition p22 values less than ('1997-12-01'),
partition p23 values less than ('1998-06-01'),
partition p24 values less than maxvalue
);
"

date
mysql -urateup  -D$database -e "load data infile '$data_dir/region.tbl' into table region fields terminated by '|';"
mysql -urateup  -D$database -e "load data infile '$data_dir/nation.tbl' into table nation fields terminated by '|';" 
mysql -urateup  -D$database -e "load data infile '$data_dir/part.tbl' into table part fields terminated by '|';"
mysql -urateup  -D$database -e "load data infile '$data_dir/supplier.tbl' into table supplier fields terminated by '|';"
mysql -urateup  -D$database -e "load data infile '$data_dir/partsupp.tbl' into table partsupp fields terminated by '|';"
mysql -urateup  -D$database -e "load data infile '$data_dir/customer.tbl' into table customer fields terminated by '|';"
mysql -urateup  -D$database -e "load data infile '$data_dir/orders.tbl' into table orders fields terminated by '|';"
# start=`date +%s.%N`
mysql -urateup  -D$database -e "load data infile '$data_dir/lineitem.tbl' into table lineitem fields terminated by '|';"
# end=`date +%s.%N`
# q1=$(getMicroTiming $start $end)
date

fi  # load

if $test; then
rm -rf $result_dir
mkdir $result_dir

start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
from
    lineitem
where
    l_shipdate <= date '1998-12-01' - interval '90' day
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus;
" > $result_dir/q1.out
end=`date +%s.%N`
q1=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    s_acctbal,
    s_name,
    n_name,
    p_partkey,
    p_mfgr,
    s_address,
    s_phone,
    s_comment
from
    part,
    supplier,
    partsupp,
    nation,
    region
where
    p_partkey = ps_partkey
    and s_suppkey = ps_suppkey
    and p_size = 15
    and p_type like '%BRASS'
    and s_nationkey = n_nationkey
    and n_regionkey = r_regionkey
    and r_name = 'EUROPE'
    and ps_supplycost = (
    select
        min(ps_supplycost)
    from
        partsupp,
        supplier,
        nation,
        region
    where
        p_partkey = ps_partkey
        and s_suppkey = ps_suppkey
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'EUROPE'
)
order by
    s_acctbal desc,
    n_name,
    s_name,
    p_partkey
limit
    100;
" > $result_dir/q2.out
end=`date +%s.%N`
q2=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    l_orderkey,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    o_orderdate,
    o_shippriority
from
    customer,
    orders,
    lineitem
where
    c_mktsegment = 'BUILDING'
    and c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15'
group by
    l_orderkey,
    o_orderdate,
    o_shippriority
order by
    revenue desc,
    o_orderdate
limit
10;
" > $result_dir/q3.out
end=`date +%s.%N`
q3=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    o_orderpriority,
    count(*) as order_count
from
    orders
where
    o_orderdate >= date '1993-07-01'
    and o_orderdate < date '1993-07-01' + interval '3' month
    and exists (
        select
            *
        from
            lineitem
        where
            l_orderkey = o_orderkey
            and l_commitdate < l_receiptdate
    )
group by
    o_orderpriority
order by
    o_orderpriority;
" > $result_dir/q4.out
end=`date +%s.%N`
q4=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    n_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue
from
    customer,
    orders,
    lineitem,
    supplier,
    nation,
    region
where
    c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and l_suppkey = s_suppkey
    and c_nationkey = s_nationkey
    and s_nationkey = n_nationkey
    and n_regionkey = r_regionkey
    and r_name = 'ASIA'
    and o_orderdate >= date '1994-01-01'
    and o_orderdate < date '1994-01-01' + interval '1' year
group by
    n_name
order by
    revenue desc;
" > $result_dir/q5.out
end=`date +%s.%N`
q5=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    sum(l_extendedprice * l_discount) as revenue
from
    lineitem
where
    l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1994-01-01' + interval '1' year
    and l_discount between .06 - 0.01 and .06 + 0.01
    and l_quantity < 24;
" > $result_dir/q6.out
end=`date +%s.%N`
q6=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    supp_nation,
    cust_nation,
    l_year,
    sum(volume) as revenue
from
    (
        select
            n1.n_name as supp_nation,
            n2.n_name as cust_nation,
            extract(year from l_shipdate) as l_year,
            l_extendedprice * (1 - l_discount) as volume
        from
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2
        where
            s_suppkey = l_suppkey
            and o_orderkey = l_orderkey
            and c_custkey = o_custkey
            and s_nationkey = n1.n_nationkey
            and c_nationkey = n2.n_nationkey
            and (
                (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
            )
            and l_shipdate between date '1995-01-01' and date '1996-12-31'
    ) as shipping
group by
    supp_nation,
    cust_nation,
    l_year
order by
    supp_nation,
    cust_nation,
    l_year;
" > $result_dir/q7.out
end=`date +%s.%N`
q7=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    o_year,
    sum(case
        when nation = 'BRAZIL' then volume
        else 0
    end) / sum(volume) as mkt_share
from
    (
        select
            extract(year from o_orderdate) as o_year,
            l_extendedprice * (1 - l_discount) as volume,
            n2.n_name as nation
        from
            part,
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2,
            region
        where
            p_partkey = l_partkey
            and s_suppkey = l_suppkey
            and l_orderkey = o_orderkey
            and o_custkey = c_custkey
            and c_nationkey = n1.n_nationkey
            and n1.n_regionkey = r_regionkey
            and r_name = 'AMERICA'
            and s_nationkey = n2.n_nationkey
            and o_orderdate between date '1995-01-01' and date '1996-12-31'
            and p_type = 'ECONOMY ANODIZED STEEL'
    ) as all_nations
group by
    o_year
order by
    o_year;
" > $result_dir/q8.out
end=`date +%s.%N`
q8=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    nation,
    o_year,
    sum(amount) as sum_profit
from
    (
        select
            n_name as nation,
            extract(year from o_orderdate) as o_year,
            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
        from
            part,
            supplier,
            lineitem,
            partsupp,
            orders,
            nation
        where
            s_suppkey = l_suppkey
            and ps_suppkey = l_suppkey
            and ps_partkey = l_partkey
            and p_partkey = l_partkey
            and o_orderkey = l_orderkey
            and s_nationkey = n_nationkey
            and p_name like '%green%'
    ) as profit
group by
    nation,
    o_year
order by
    nation,
    o_year desc;
" > $result_dir/q9.out
end=`date +%s.%N`
q9=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    c_custkey,
    c_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    c_acctbal,
    n_name,
    c_address,
    c_phone,
    c_comment
from
    customer,
    orders,
    lineitem,
    nation
where
    c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and o_orderdate >= date '1993-10-01'
    and o_orderdate < date '1993-10-01' + interval '3' month
    and l_returnflag = 'R'
    and c_nationkey = n_nationkey
group by
    c_custkey,
    c_name,
    c_acctbal,
    c_phone,
    n_name,
    c_address,
    c_comment
order by
    revenue desc
limit
    20;
" > $result_dir/q10.out
end=`date +%s.%N`
q10=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    ps_partkey,
    sum(ps_supplycost * ps_availqty) as value
from
    partsupp,
    supplier,
    nation
where
    ps_suppkey = s_suppkey
    and s_nationkey = n_nationkey
    and n_name = 'GERMANY'
group by
    ps_partkey having
        sum(ps_supplycost * ps_availqty) > (
            select
                sum(ps_supplycost * ps_availqty) * 0.0001000000
            from
                partsupp,
                supplier,
                nation
            where
                ps_suppkey = s_suppkey
                and s_nationkey = n_nationkey
                and n_name = 'GERMANY'
        )
order by
    value desc;
" > $result_dir/q11.out
end=`date +%s.%N`
q11=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    l_shipmode,
    sum(case
        when o_orderpriority = '1-URGENT'
            or o_orderpriority = '2-HIGH'
            then 1
        else 0
    end) as high_line_count,
    sum(case
        when o_orderpriority <> '1-URGENT'
            and o_orderpriority <> '2-HIGH'
            then 1
        else 0
    end) as low_line_count
from
    orders,
    lineitem
where
    o_orderkey = l_orderkey
    and l_shipmode in ('MAIL', 'SHIP')
    and l_commitdate < l_receiptdate
    and l_shipdate < l_commitdate
    and l_receiptdate >= date '1994-01-01'
    and l_receiptdate < date '1994-01-01' + interval '1' year
group by
    l_shipmode
order by
    l_shipmode;
" > $result_dir/q12.out
end=`date +%s.%N`
q12=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    c_count,
    count(*) as custdist
from
    (
        select
            c_custkey,
            count(o_orderkey)
        from
            customer left outer join orders on
                c_custkey = o_custkey
                and o_comment not like '%special%requests%'
        group by
            c_custkey
    ) as c_orders (c_custkey, c_count)
group by
    c_count
order by
    custdist desc,
    c_count desc;
" > $result_dir/q13.out
end=`date +%s.%N`
q13=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    100.00 * sum(case
        when p_type like 'PROMO%'
            then l_extendedprice * (1 - l_discount)
        else 0
    end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
    lineitem,
    part
where
    l_partkey = p_partkey
    and l_shipdate >= date '1995-09-01'
    and l_shipdate < date '1995-09-01' + interval '1' month;
" > $result_dir/q14.out
end=`date +%s.%N`
q14=$(getMicroTiming $start $end)


mysql -urateup  -D$database -e "
create view revenue0 (supplier_no, total_revenue) as
    select
        l_suppkey,
        sum(l_extendedprice * (1 - l_discount))
    from
        lineitem
    where
        l_shipdate >= date '1996-01-01'
        and l_shipdate < date '1996-01-01' + interval '3' month
    group by
        l_suppkey;
"
start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    s_suppkey,
    s_name,
    s_address,
    s_phone,
    total_revenue
from
    supplier,
    revenue0
where
    s_suppkey = supplier_no
    and total_revenue = (
        select
            max(total_revenue)
        from
            revenue0
    )
order by
    s_suppkey;
" > $result_dir/q15.out
end=`date +%s.%N`
q15=$(getMicroTiming $start $end)
mysql -urateup  -D$database -e "
drop view revenue0;
"


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    p_brand,
    p_type,
    p_size,
    count(distinct ps_suppkey) as supplier_cnt
from
    partsupp,
    part
where
    p_partkey = ps_partkey
    and p_brand <> 'Brand#45'
    and p_type not like 'MEDIUM POLISHED%'
    and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
    and ps_suppkey not in (
        select
            s_suppkey
        from
            supplier
        where
            s_comment like '%Customer%Complaints%'
    )
group by
    p_brand,
    p_type,
    p_size
order by
    supplier_cnt desc,
    p_brand,
    p_type,
    p_size;
" > $result_dir/q16.out
end=`date +%s.%N`
q16=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    sum(l_extendedprice) / 7.0 as avg_yearly
from
    lineitem,
    part
where
    p_partkey = l_partkey
    and p_brand = 'Brand#23'
    and p_container = 'MED BOX'
    and l_quantity < (
        select
            0.2 * avg(l_quantity)
        from
            lineitem
        where
            l_partkey = p_partkey
    );
" > $result_dir/q17.out
end=`date +%s.%N`
q17=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    sum(l_quantity)
from
    customer,
    orders,
    lineitem
where
    o_orderkey in (
        select
            l_orderkey
        from
            lineitem
        group by
            l_orderkey having
                sum(l_quantity) > 300
    )
    and c_custkey = o_custkey
    and o_orderkey = l_orderkey
group by
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice
order by
    o_totalprice desc,
    o_orderdate
limit
    100;
" > $result_dir/q18.out
end=`date +%s.%N`
q18=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    sum(l_extendedprice* (1 - l_discount)) as revenue
from
    lineitem,
    part
where
    (
        p_partkey = l_partkey
        and p_brand = 'Brand#12'
        and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        and l_quantity >= 1 and l_quantity <= 1 + 10
        and p_size between 1 and 5
        and l_shipmode in ('AIR', 'AIR REG')
        and l_shipinstruct = 'DELIVER IN PERSON'
    )
    or
    (
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        and l_quantity >= 10 and l_quantity <= 10 + 10
        and p_size between 1 and 10
        and l_shipmode in ('AIR', 'AIR REG')
        and l_shipinstruct = 'DELIVER IN PERSON'
    )
    or
    (
        p_partkey = l_partkey
        and p_brand = 'Brand#34'
        and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        and l_quantity >= 20 and l_quantity <= 20 + 10
        and p_size between 1 and 15
        and l_shipmode in ('AIR', 'AIR REG')
        and l_shipinstruct = 'DELIVER IN PERSON'
    );
" > $result_dir/q19.out
end=`date +%s.%N`
q19=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    s_name,
    s_address
from
    supplier,
    nation
where
    s_suppkey in (
        select
            ps_suppkey
        from
            partsupp
        where
            ps_partkey in (
                select
                    p_partkey
                from
                    part
                where
                    p_name like 'forest%'
            )
            and ps_availqty > (
                select
                    0.5 * sum(l_quantity)
                from
                    lineitem
                where
                    l_partkey = ps_partkey
                    and l_suppkey = ps_suppkey
                    and l_shipdate >= date '1994-01-01'
                    and l_shipdate < date '1994-01-01' + interval '1' year
            )
    )
    and s_nationkey = n_nationkey
    and n_name = 'CANADA'
order by
    s_name;
" > $result_dir/q20.out
end=`date +%s.%N`
q20=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    s_name,
    count(*) as numwait
from
    supplier,
    lineitem l1,
    orders,
    nation
where
    s_suppkey = l1.l_suppkey
    and o_orderkey = l1.l_orderkey
    and o_orderstatus = 'F'
    and l1.l_receiptdate > l1.l_commitdate
    and exists (
        select
            *
        from
            lineitem l2
        where
            l2.l_orderkey = l1.l_orderkey
            and l2.l_suppkey <> l1.l_suppkey
    )
    and not exists (
        select
            *
        from
            lineitem l3
        where
            l3.l_orderkey = l1.l_orderkey
            and l3.l_suppkey <> l1.l_suppkey
            and l3.l_receiptdate > l3.l_commitdate
    )
    and s_nationkey = n_nationkey
    and n_name = 'SAUDI ARABIA'
group by
    s_name
order by
    numwait desc,
    s_name
limit
    100;
" > $result_dir/q21.out
end=`date +%s.%N`
q21=$(getMicroTiming $start $end)


start=`date +%s.%N`
mysql -urateup  -D$database -e "
select
    cntrycode,
    count(*) as numcust,
    sum(c_acctbal) as totacctbal
from
    (
        select
            substring(c_phone from 1 for 2) as cntrycode,
            c_acctbal
        from
            customer
        where
            substring(c_phone from 1 for 2) in
                ('13', '31', '23', '29', '30', '18', '17')
            and c_acctbal > (
                select
                    avg(c_acctbal)
                from
                    customer
                where
                    c_acctbal > 0.00
                    and substring(c_phone from 1 for 2) in
                        ('13', '31', '23', '29', '30', '18', '17')
            )
            and not exists (
                select
                    *
                from
                    orders
                where
                    o_custkey = c_custkey
            )
    ) as custsale
group by
    cntrycode
order by
    cntrycode;
" > $result_dir/q22.out
end=`date +%s.%N`
q22=$(getMicroTiming $start $end)



echo "=============================="
# for i in `seq 1 22`;
# do
# echo q$i:$((q$i))
# done
echo -e "
q1:$q1
q2:$q2
q3:$q3
q4:$q4
q5:$q5
q6:$q6
q7:$q7
q8:$q8
q9:$q9
q10:$q10
q11:$q11
q12:$q12
q13:$q13
q14:$q14
q15:$q15
q16:$q16
q17:$q17
q18:$q18
q19:$q19
q20:$q20
q21:$q21
q22:$q22
"
echo "=============================="
diff -w $answer_dir $result_dir
fi

date