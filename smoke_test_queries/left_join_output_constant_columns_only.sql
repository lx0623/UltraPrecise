select 1, 1.24680, "jackpot"
from
    (select * from orders where o_custkey = 1) aaa left join customer on
            c_custkey = aaa.o_custkey
;
