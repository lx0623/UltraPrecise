#include <gtest/gtest.h>
#include <string>

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

#include "../../TestUtils.h"

using namespace std;
using namespace aries_engine;
using namespace aries_acc;
using namespace aries_test;

static string TEST_DB_NAME( "test_right_join" );
static string TEST_LEFT_TABLE_NAME( "t_left" );
static string TEST_RIGHT_TABLE_NAME( "t_right" );
class UT_right_join: public testing::Test
{
private:
protected:
    void SetUp()
    {
        InitTable( TEST_DB_NAME, TEST_LEFT_TABLE_NAME );
        string sql = "create table " + TEST_LEFT_TABLE_NAME + " ( lf1 int )";
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE(result->IsSuccess());

        InitTable( TEST_DB_NAME, TEST_RIGHT_TABLE_NAME );
        sql = "create table " + TEST_RIGHT_TABLE_NAME + " ( rf1 int not null )";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE(result->IsSuccess());

    }
    static void TearDownTestCase()
    {
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    }
};


TEST_F(UT_right_join, const_false_condition)
{
    string sql = "insert into " + TEST_LEFT_TABLE_NAME + " values( 1 )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + TEST_RIGHT_TABLE_NAME + " values( 2 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + TEST_LEFT_TABLE_NAME + " right join " + TEST_RIGHT_TABLE_NAME + " on false;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    ASSERT_EQ( resTable->GetColumnCount(), 2 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_TRUE( columnBuff->isInt32DataNull( 0 ) );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );

}