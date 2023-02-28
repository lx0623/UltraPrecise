select
    l_returnflag,
    1,
    l_linestatus,
    1.13579,
    sum(l_quantity) as sum_qty,
    "xyz"
from
    lineitem
where
    l_shipdate <= '1998-08-01'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus
;