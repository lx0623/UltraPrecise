select count(*)
from
    (select * from orders where o_custkey = 123456789) aaa left join customer on
    c_custkey = aaa.o_custkey and c_custkey = 123456789
;