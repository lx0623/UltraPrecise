#include <gtest/gtest.h>
#include <string>

#include "TestCommonBase.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace std;
using std::string;

void TestPie( int arg_query_number )
{
    std::string db_name = "analysis";
    std::string sql = "test_pie_queries/pie_" + std::to_string( arg_query_number ) + ".sql";
    std::string result = "test_pie_queries/mysql_result/pie_" + std::to_string( arg_query_number ) + ".sql.result";
    doQueryAndCheckResult(db_name, sql, result, arg_query_number);
}

TEST(pie_query, q1)
{
    TestPie(1);
}

TEST(pie_query, q2)
{
    TestPie(2);
}

TEST(pie_query, q3)
{
    TestPie(3);
}

TEST(pie_query, q4)
{
    TestPie(4);
}

TEST(pie_query, q5)
{
    TestPie(5);
}

TEST(pie_query, q6)
{
    TestPie(6);
}

TEST(pie_query, q7)
{
    TestPie(7);
}

TEST(pie_query, q8)
{
    TestPie(8);
}

TEST(pie_query, q9)
{
    TestPie(9);
}

TEST(pie_query, q10)
{
    TestPie(10);
}

TEST(pie_query, q11)
{
    TestPie(11);
}

TEST(pie_query, q12)
{
    TestPie(12);
}

TEST(pie_query, q13)
{
    TestPie(13);
}

TEST(pie_query, q14)
{
    TestPie(14);
}

TEST(pie_query, q15)
{
    TestPie(15);
}

TEST(pie_query, q16)
{
    TestPie(16);
}

TEST(pie_query, q17)
{
    TestPie(17);
}

TEST(pie_query, q18)
{
    TestPie(18);
}

TEST(pie_query, q19)
{
    TestPie(19);
}

TEST(pie_query, q20)
{
    TestPie(20);
}

TEST(pie_query, q21)
{
    TestPie(21);
}

TEST(pie_query, q22)
{
    TestPie(22);
}

TEST(pie_query, q23)
{
    TestPie(23);
}

TEST(pie_query, q24)
{
    TestPie(24);
}

TEST(pie_query, q25)
{
    TestPie(25);
}

TEST(pie_query, q26)
{
    TestPie(26);
}

TEST(pie_query, q27)
{
    TestPie(27);
}

TEST(pie_query, q28)
{
    TestPie(28);
}

TEST(pie_query, q29)
{
    TestPie(29);
}

TEST(pie_query, q30)
{
    TestPie(30);
}

TEST(pie_query, q31)
{
    TestPie(31);
}

TEST(pie_query, q32)
{
    TestPie(32);
}

TEST(pie_query, q33)
{
    TestPie(33);
}

TEST(pie_query, q34)
{
    TestPie(34);
}

TEST(pie_query, q35)
{
    TestPie(35);
}

TEST(pie_query, q36)
{
    TestPie(36);
}

TEST(pie_query, q37)
{
    TestPie(37);
}

TEST(pie_query, q38)
{
    TestPie(38);
}

TEST(pie_query, q39)
{
    TestPie(39);
}

TEST(pie_query, q40)
{
    TestPie(40);
}

TEST(pie_query, q41)
{
    TestPie(41);
}

TEST(pie_query, q42)
{
    TestPie(42);
}

TEST(pie_query, q43)
{
    TestPie(43);
}

TEST(pie_query, q44)
{
    TestPie(44);
}

TEST(pie_query, q45)
{
    TestPie(45);
}

TEST(pie_query, q46)
{
    TestPie(46);
}

TEST(pie_query, q47)
{
    TestPie(47);
}
