select count(*)
from
     (select * from orders where o_custkey = 123456789) aaa full outer join customer on
    c_custkey = aaa.o_custkey
;