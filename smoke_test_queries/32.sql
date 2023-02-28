
select
    sum(l_extendedprice * l_discount) as revenue,
    l_shipdate
from
    lineitem
where
    l_shipdate >= '1995-01-01'
    and l_shipdate < '1996-01-01'
    and l_discount between 0.05 - 0.01 and 0.05 + 0.01
    and l_quantity < 25
group by 
    l_shipdate
having 
    abs( sum(l_extendedprice) - sum(l_discount) ) > 10
order by
    l_shipdate
;
