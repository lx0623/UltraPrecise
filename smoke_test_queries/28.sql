select
    UNIX_TIMESTAMP(L_COMMITDATE)
from
    lineitem
where
    l_shipdate = '1996-07-02' and UNIX_TIMESTAMP(L_COMMITDATE)-UNIX_TIMESTAMP(L_SHIPDATE) > 0 and extract('year', L_COMMITDATE) - extract('year', L_SHIPDATE) = 0

order by 
    L_COMMITDATE
;

