#include <gtest/gtest.h>
#include <string>

#include "utils/string_util.h"
#include "../../TestUtils.h"
#include "../../TestCommonBase.h"
using namespace std;
using namespace aries_test;

TEST( optimizer, groupby_multi_key )
{
    string cwd = aries_utils::get_current_work_directory();
    string sql( R"( select l_orderkey, l_linenumber, l_discount, max( l_quantity ), l_comment
                  from lineitem
                  where l_discount >= 0.1
                  group by l_orderkey, l_linenumber, l_discount, l_comment
                      having max( l_quantity ) >= 50
                  order by l_discount desc, max( l_quantity ) desc )" );
    string expectedResultFile = cwd + "/test_resources/Optimizer/multi_primary_key_groupby.result";
    auto result = ExecuteSQL( sql, "scale_1" );

    CheckQueryResult( result, expectedResultFile );
}