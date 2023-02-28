select count(*)
from
     customer inner join (select * from orders where o_custkey = 123456789) aaa on
    c_custkey = aaa.o_custkey
;
