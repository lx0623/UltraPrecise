1. Make sure you have installed antlr4/c++. Modify ANTLR4_CPP_PATH in the Makefile!

2. Make sure you have installed Facebook folly C++ library. Modify FOLLY_STATIC_LIB in the Makefile!

3. If you want to see call stacks when debugging, keep RICH_DEBUG_CC_OPTIONS and RICH_DEBUG_LINK_OPTIONS; Otherwise, don't use them.


4. Make. Then you can get an exe called justtest.

5. learn from test.py! It should be able to execute all 22 tpc-h queries. You can also check the queries in subdir test_simple_queries.



