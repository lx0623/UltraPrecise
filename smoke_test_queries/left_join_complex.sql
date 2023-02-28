select c_phone, o_orderstatus from customer left join orders on c_custkey = o_custkey and c_nationkey * 2 < o_orderkey + 3;
