#include <gtest/gtest.h>
#include <string>

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

#include "../../TestUtils.h"

using namespace std;
using namespace aries_engine;
using namespace aries_acc;
using namespace aries_test;

static string TEST_DB_NAME( "test_inner_join" );
static string TEST_LEFT_TABLE_NAME( "t_left" );
static string TEST_RIGHT_TABLE_NAME( "t_right" );
class UT_inner_join: public testing::Test
{
private:
protected:
    void SetUp()
    {
        InitTable( TEST_DB_NAME, TEST_LEFT_TABLE_NAME );
        string sql = "create table " + TEST_LEFT_TABLE_NAME + " ( lf1 int, lf2 char(4) )";
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE(result->IsSuccess());

        InitTable( TEST_DB_NAME, TEST_RIGHT_TABLE_NAME );
        sql = "create table " + TEST_RIGHT_TABLE_NAME + " ( rf1 int, rf2 decimal )";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE(result->IsSuccess());

    }
    static void TearDownTestCase()
    {
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    }
};


TEST_F(UT_inner_join, left_empty)
{
    string sql = "insert into " + TEST_RIGHT_TABLE_NAME + " values( 1, 1.0 )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + TEST_LEFT_TABLE_NAME + " join " + TEST_RIGHT_TABLE_NAME + " where lf1 = rf1;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
    ASSERT_EQ( resTable->GetColumnCount(), 4 );
}
TEST_F(UT_inner_join, right_empty)
{
    string sql = "insert into " + TEST_LEFT_TABLE_NAME + " values( 1, 'abcd' )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + TEST_LEFT_TABLE_NAME + " join " + TEST_RIGHT_TABLE_NAME + " where lf1 = rf1;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
    ASSERT_EQ( resTable->GetColumnCount(), 4 );
}