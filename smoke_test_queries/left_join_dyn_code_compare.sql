select * from region left join nation on r_regionkey = n_regionkey and r_regionkey = 2 order by r_regionkey, n_nationkey;
