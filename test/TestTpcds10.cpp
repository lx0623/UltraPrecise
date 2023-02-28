#include <gtest/gtest.h>
#include <string>

#include "TestCommonBase.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace std;
using std::string;

void TestTpcds10( int arg_query_number )
{
    std::string db_name = "tpcds_10";
    std::string sql = "test_tpcds10_queries/tpcds_10_" + std::to_string( arg_query_number ) + ".sql";
    std::string result = "test_tpcds10_queries/mysql_result/tpcds_10_" + std::to_string( arg_query_number ) + ".sql.result";
    std::cout << "-------------------------------------------------------------------------------------------------------" << arg_query_number
              << "\n\n\n";
    doQueryAndCheckResult(db_name, sql, result, arg_query_number);
}

TEST(query_tpcds10, q1)
{
    TestTpcds10(1);
}

TEST(query_tpcds10, q3)
{
    TestTpcds10(3);
}

TEST(query_tpcds10, q6)
{
    TestTpcds10(6);
}

TEST(query_tpcds10, q7)
{
    TestTpcds10(7);
}

TEST(query_tpcds10, q9)
{
    TestTpcds10(9);
}

TEST(query_tpcds10, q13)
{
    TestTpcds10(13);
}

TEST(query_tpcds10, q15)
{
    TestTpcds10(15);
}

TEST(query_tpcds10, q19)
{
    TestTpcds10(19);
}

TEST(query_tpcds10, q25)
{
    TestTpcds10(25);
}

TEST(query_tpcds10, q26)
{
    TestTpcds10(26);
}

TEST(query_tpcds10, q29)
{
    TestTpcds10(29);
}

TEST(query_tpcds10, q32)
{
    TestTpcds10(32);
}

TEST(query_tpcds10, q34)
{
    TestTpcds10(34);
}

TEST(query_tpcds10, q37)
{
    TestTpcds10(37);
}

TEST(query_tpcds10, q41)
{
    TestTpcds10(41);
}

TEST(query_tpcds10, q42)
{
    TestTpcds10(42);
}

TEST(query_tpcds10, q43)
{
    TestTpcds10(43);
}

TEST(query_tpcds10, q45)
{
    TestTpcds10(45);
}

TEST(query_tpcds10, q46)
{
    TestTpcds10(46);
}

TEST(query_tpcds10, q48)
{
    TestTpcds10(48);
}

TEST(query_tpcds10, q50)
{
    TestTpcds10(50);
}

TEST(query_tpcds10, q52)
{
    TestTpcds10(53);
}

TEST(query_tpcds10, q55)
{
    TestTpcds10(55);
}

TEST(query_tpcds10, q61)
{
    TestTpcds10(61);
}

TEST(query_tpcds10, q62)
{
    TestTpcds10(62);
}

TEST(query_tpcds10, q65)
{
    TestTpcds10(65);
}

TEST(query_tpcds10, q68)
{
    TestTpcds10(68);
}

TEST(query_tpcds10, q71)
{
    TestTpcds10(71);
}

TEST(query_tpcds10, q72)
{
    TestTpcds10(72);
}

TEST(query_tpcds10, q73)
{
    TestTpcds10(73);
}

TEST(query_tpcds10, q79)
{
    TestTpcds10(79);
}

TEST(query_tpcds10, q85)
{
    TestTpcds10(85);
}

TEST(query_tpcds10, q91)
{
    TestTpcds10(91);
}

TEST(query_tpcds10, q93)
{
    TestTpcds10(93);
}

TEST(query_tpcds10, q96)
{
    TestTpcds10(96);
}

TEST(query_tpcds10, q99)
{
    TestTpcds10(99);
}
