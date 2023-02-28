select
    1, 1.13579, "abc"
from
    lineitem
where
    l_shipdate <= '1998-08-01'
group by
    l_returnflag,
    l_linestatus
;