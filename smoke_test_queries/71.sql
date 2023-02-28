select count(*)
from
     customer where not exists (select o_custkey from orders where o_custkey = customer.c_custkey and o_custkey = 123456789)
;