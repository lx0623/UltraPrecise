select
    C_ACCTBAL,O_TOTALPRICE
from
    orders full outer join customer on
    c_custkey = o_custkey 
order by 
    C_ACCTBAL, O_TOTALPRICE
;
