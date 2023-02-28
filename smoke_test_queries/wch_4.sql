select
    concat(substr(n_name, 1, 3), " 牛！牛！牛！")
from
    nation
where
    n_name like '%量%'
;
