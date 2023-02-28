( o_orderkey       integer not null primary key,
  o_custkey        integer not null,
  o_orderstatus    char(1) not null,
  o_totalprice     decimal(15,2) not null,
  o_orderdate      date not null,
  o_orderpriority  char(15) not null encoding bytedict shares orders( o_orderpriority ),
  o_clerk          char(15) not null, 
  o_shippriority   integer not null,
  o_comment        varchar(79) not null );
