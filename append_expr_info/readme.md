# Expr_info

This folder records the query statements and the corresponding data for the experiment. First, you need to generate the corresponding data using the script file in the `gen_data` folder and record the location of the data. After that, you can complete the experiment by using the SQL statements in the `expr_query` folder.

In terms of statistical time, we look at the time it takes to compute the expression, so we try to remove the effect of the print cost. Specifically, for Fast-APA, RateupDB, and Heavy.AI, we count the time by checking the `logs`. For MonetDB, PostgreSQL, Cockroach, H2, etc. we use statements such as `TRACE`, `Explain Analyze`, etc.