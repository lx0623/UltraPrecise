
select
    l_orderkey,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    o_orderdate,
    o_shippriority,
    sum(datediff(l_shipdate, o_orderdate)) as latency
from
    customer,
    orders,
    lineitem
where
    c_mktsegment = 'FURNITURE'
    and c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and date_sub( date(o_orderdate), interval 10 day ) < date_sub( date( '2019-9-5' ), interval 20 hour )
    and l_shipdate > '1995-03-20'
group by
    l_orderkey,
    o_orderdate,
    o_shippriority
order by
    revenue desc,
    o_orderdate,
    l_orderkey,
    o_shippriority,
    latency desc

limit 147601,100
;
