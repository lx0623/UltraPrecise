select
    l_returnflag,
    l_linestatus,
    count(distinct l_suppkey)
from
    lineitem
where
    l_shipdate >= '2019-11-16'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus
;

