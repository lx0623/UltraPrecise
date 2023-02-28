select 1, 1.24680, "jackpot"
from
    (select * from customer where c_custkey in (1, 2)) aaa full join (select * from orders where o_custkey = 1) bbb on
            aaa.c_custkey = bbb.o_custkey
;
