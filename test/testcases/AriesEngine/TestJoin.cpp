#include <gtest/gtest.h>
#include <string>

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

using namespace std;
using namespace aries_engine;
using namespace aries_acc;

static string TEST_DB_NAME( "test_hash_join" );
class UT_hash_join: public testing::Test
{
private:
protected:
    static void SetUpTestCase()
    {
        string sql = "create database if not exists " + TEST_DB_NAME + ";";
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );

        sql = "drop table if exists ta;";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
        sql = "create table ta ( value int );";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
        sql = "insert into ta values (-1), (2), (3), (4), (null), (9);";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );

        sql = "drop table if exists tb;";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
        sql = "create table tb ( value int );";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
        sql = "insert into tb values (-1), (null);";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );

        sql = "drop table if exists tc;";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
        sql = "create table tc( value int not null primary key );";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
        sql = "insert into tc values (-1), (2), (3), (4);";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );
    }
    static void TearDownTestCase()
    {
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    }
};

TEST_F(UT_hash_join, half_join)
{
    auto sql = "select * from tc where value in ( select value from ta ) order by value";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 4 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );

    sql = "select * from tc where value not in( select value from ta ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "select * from tc where exists ( select 1 from ta where tc.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 4 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );

    sql = "select * from tc where not exists ( select 1 from ta where tc.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "select * from tc where value in ( select value from tb ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from tc where value not in( select value from tb ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "select * from tc where exists ( select 1 from tb where tc.value = tb.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from tc where not exists ( select 1 from tb where tc.value = tb.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 3 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "4" );

    sql = "select * from ta where value in ( select value from tc ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 4 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );

    sql = "select * from ta where value not in( select value from tc ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "9" );

    sql = "select * from ( select value from ta group by value ) as td where value not in( select value from tc ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "9" );

    sql = "select * from ta where exists ( select 1 from tc where tc.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 4 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );

    sql = "select * from ta where not exists ( select 1 from tc where tc.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 2 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "9" );

    sql = "select * from tb where value in ( select value from ta ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from tb where value not in( select value from ta ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "select * from tb where exists ( select 1 from ta where tb.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from tb where not exists ( select 1 from ta where tb.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "NULL" );

    sql = "select * from ta where value in ( select value from tb ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from ta where value not in( select value from tb ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "select * from ta where exists ( select 1 from tb where tb.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from ta where not exists ( select 1 from tb where tb.value = ta.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 5 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 4 ), "9" );

    sql = "select * from ( select value from ta group by value ) as td where not exists ( select 1 from tb where tb.value = td.value ) order by value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 5 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 4 ), "9" );
}

TEST_F(UT_hash_join, left_join)
{
    auto sql = "select * from ta left join tb on ta.value = tb.value order by ta.value;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 6 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 4 ), "4" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 5 ), "9" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 1 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 2 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 3 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 4 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 5 ), "NULL" );

    sql = "select * from tc left join tb on tc.value = tb.value order by tc.value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 4 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 1 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 2 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 3 ), "NULL" );

    sql = "select * from ( select value from ta group by value ) as td left join tb on td.value = tb.value order by td.value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 6 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 4 ), "4" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 5 ), "9" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 0 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 1 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 2 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 3 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 4 ), "NULL" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 5 ), "NULL" );
}

TEST_F(UT_hash_join, inner_join)
{
    auto sql = "select * from ta inner join tb on ta.value = tb.value order by ta.value;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 0 ), "-1" );


    sql = "select * from tc inner join tb on tc.value = tb.value order by tc.value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 0 ), "-1" );

    sql = "select * from ( select value from ta group by value ) as td inner join tb on td.value = tb.value order by td.value;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] )->GetContent();
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "-1" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( 0 ), "-1" );
}