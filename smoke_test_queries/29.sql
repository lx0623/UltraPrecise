        select
            date(
                if(
                    L_SHIPDATE is not null,
                    L_SHIPDATE,
                    L_COMMITDATE
                )
            ) ,
            count(*) loan_cnt,
            sum(L_EXTENDEDPRICE) loan_amt
from
    lineitem
where
    l_shipdate <= '1996-08-01'
        group by
            date
                (
                    if(
                    L_SHIPDATE is not null,
                    L_SHIPDATE,
                    L_COMMITDATE
                    )
                )
                order by
                loan_amt
                ;
