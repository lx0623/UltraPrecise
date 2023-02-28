select
    substr(n_name, 2)
from
    nation
where
    n_name like '%量%'
;
