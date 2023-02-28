
select
    sum(l_extendedprice * l_discount) as revenue,
    l_shipdate
from
    lineitem
where
    l_shipdate >= '1995-1-01'
    and timediff(l_shipdate, '1995-10-1') < 720000
    and timediff(l_shipdate, '1995-10-1') > -720000
group by
    l_shipdate
having
    abs( sum(l_extendedprice) - sum(l_discount) ) > 10
order by
    l_shipdate
;
