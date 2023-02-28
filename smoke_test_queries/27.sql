select
    l_orderkey,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    date(o_orderdate),
    o_shippriority,
    count(*)
from
    customer,
    orders,
    lineitem
where
    c_mktsegment = 'FURNITURE'
    and c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and date_add( date(o_orderdate), interval -2 day ) < date_add( date( '2019-08-23' ), interval -8000 day )
    and l_shipdate > '1995-03-20'
group by
    l_orderkey,
    date(o_orderdate),
    o_shippriority
order by
    revenue desc
;
