
select
    100.00 * sum(if
        (p_type like 'PROMO%' and p_type is not null,
             true,
         null
    )) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
    lineitem,
    part
where
    l_partkey = p_partkey
    and l_shipdate >= '1993-04-01'
    and l_shipdate < '1993-05-01'
;
