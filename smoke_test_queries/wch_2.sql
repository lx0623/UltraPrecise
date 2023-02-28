select
    s_suppkey, s_name, s_address, c_custkey, c_name, c_address, c_comment
from
    supplier
join
    customer on s_name=c_name
order by
    s_suppkey desc,
    c_custkey
;
