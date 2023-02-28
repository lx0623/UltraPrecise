select
    concat(substr(n_name, -5, 3), " 牛！牛！牛！")
from
    nation
where
    n_name like '%量%'
;
