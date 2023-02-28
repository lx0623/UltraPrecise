select 1, 1.24680, "jackpot"
from
    customer right join (select * from orders where o_custkey = 1) aaa on
            c_custkey = aaa.o_custkey
;
