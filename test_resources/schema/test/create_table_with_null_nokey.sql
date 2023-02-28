create table region
(
    r_regionkey  integer not null,
    r_name       char(25),
    r_comment    varchar(152)
);

create table nation
(
    n_nationkey  integer not null,
    n_name       char(25),
    n_regionkey  integer not null,
    n_comment    varchar(152)
);

create table part
(
    p_partkey     integer not null,
    p_name        varchar(55),
    p_mfgr        char(25),
    p_brand       char(10) encoding bytedict as p_brand_nullable,
    p_type        varchar(25) encoding shortdict as p_type_nullable,
    p_size        integer,
    p_container   char(10) encoding bytedict as p_container_nullable,
    p_retailprice decimal(12,2),
    p_comment     varchar(23)
);

create table supplier
(
    s_suppkey     integer not null,
    s_name        char(25),
    s_address     varchar(40),
    s_nationkey   integer not null,
    s_phone       char(15),
    s_acctbal     decimal(12,2),
    s_comment     varchar(101)
);

create table partsupp
(
    ps_partkey     integer not null,
    ps_suppkey     integer not null,
    ps_availqty    integer,
    ps_supplycost  decimal(12,2),
    ps_comment     varchar(199)
);

create table customer
(
    c_custkey     integer not null,
    c_name        varchar(25),
    c_address     varchar(40),
    c_nationkey   integer not null,
    c_phone       char(15),
    c_acctbal     decimal(12,2),
    c_mktsegment  char(10) encoding bytedict as c_mktsegment_nullable,
    c_comment     varchar(117)
);

create table orders
(
    o_orderkey       integer not null,
    o_custkey        integer not null,
    o_orderstatus    char(1),
    o_totalprice     decimal(12,2),
    o_orderdate      date,
    o_orderpriority  char(15) encoding bytedict as o_orderpriority_nullable,
    o_clerk          char(15), 
    o_shippriority   integer,
    o_comment        varchar(79)
);

create table lineitem
(
    l_orderkey    integer not null,
    l_partkey     integer not null,
    l_suppkey     integer not null,
    l_linenumber  integer,
    l_quantity    decimal(12,2),
    l_extendedprice  decimal(12,2),
    l_discount    decimal(12,2),
    l_tax         decimal(12,2),
    l_returnflag  char(1),
    l_linestatus  char(1),
    l_shipdate    date,
    l_commitdate  date,
    l_receiptdate date,
    l_shipinstruct char(25) encoding bytedict as l_shipinstruct_nullable,
    l_shipmode     char(10) encoding bytedict as l_shipmode_nullable,
    l_comment      varchar(44)
);
