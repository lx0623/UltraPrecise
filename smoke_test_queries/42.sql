select
    C_ACCTBAL,O_TOTALPRICE
from
    orders left outer join customer on
    c_custkey = o_custkey and C_ACCTBAL < O_TOTALPRICE
order by 
    C_ACCTBAL, O_TOTALPRICE
;