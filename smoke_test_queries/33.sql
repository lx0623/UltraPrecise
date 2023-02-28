
select
    sum(l_extendedprice * l_discount) as revenue,
    l_shipdate
from
    lineitem
where
    l_shipdate >= '1995-01-01'
    and abs(datediff(l_shipdate, '1995-10-1')) < 3
group by
    l_shipdate
having 
    abs( sum(l_extendedprice) - sum(l_discount) ) > 10
order by
    l_shipdate
;
