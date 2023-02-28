
select
l_partkey,
    count(if
        (p_type like 'PROMO%' and p_type is not null,
             true,
         null
    ) )as promo_revenue
from
    lineitem,
    part
where
    l_partkey = p_partkey
    and l_shipdate >= '1993-04-01'
    and l_shipdate < '1993-05-01'

group by 
l_partkey
order by 
l_partkey
    
;
