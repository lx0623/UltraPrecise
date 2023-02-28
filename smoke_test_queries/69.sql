select
    c_nationkey,
    sum(c_nationkey * (c_acctbal - 500) / 1000)
from
    customer
group by
    c_nationkey
order by
    sum(c_nationkey * (500 - c_acctbal)) desc ,
    c_nationkey,
    sum(c_acctbal * (50 - c_nationkey)) desc
;
