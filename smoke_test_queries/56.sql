select count(*)
from
     (select * from customer where c_custkey = 123456789) bbb right join (select * from orders where o_custkey = 123456789) aaa on
    bbb.c_custkey = aaa.o_custkey
;
