select count(*)
from
     customer, orders
where
    o_custkey <> c_custkey
;
