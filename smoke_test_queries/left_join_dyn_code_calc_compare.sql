select * from region left join nation on r_regionkey = n_regionkey and r_regionkey - 1 = 1 order by r_regionkey, n_nationkey;
