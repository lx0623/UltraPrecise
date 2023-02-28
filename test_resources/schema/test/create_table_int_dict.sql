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

create table orders  ( o_orderkey       char(40) not null primary key,
                       o_custkey        integer not null,
                       o_orderstatus    char(1) not null,
                       o_totalprice     decimal(12,2) not null,
                       o_orderdate      date not null,
                       o_orderpriority  char(15) not null encoding bytedict as o_orderpriority,  
                       o_clerk          char(15) not null, 
                       o_shippriority   integer not null,
                       o_comment        varchar(79) not null );

create table lineitem ( l_orderkey    char(40) not null encoding intdict as l_orderkey,
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
                        l_comment      varchar(44) not null );
