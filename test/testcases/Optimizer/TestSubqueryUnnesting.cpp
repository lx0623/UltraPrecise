#include <gtest/gtest.h>
#include <string>

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

#include "utils/string_util.h"
#include "../../TestUtils.h"
#include "../../TestCommonBase.h"
using namespace std;
using namespace aries_test;

const string TEST_DB_NAME( "ut_subquery_unnesting" );

class UT_subquery_unnesting: public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        ResetTestDatabase();

        string sql = "create database if not exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    }
    static void TearDownTestCase()
    {
        ResetTestDatabase();
    }
private:
    static void ResetTestDatabase()
    {
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    }
};

TEST_F( UT_subquery_unnesting, IN_subquery_more_than_one_columns )
{
    string tableName1 = "in_subquery_more_than_one_columns";
    InitTable( TEST_DB_NAME, tableName1 );

    string sql( "create table " + tableName1 + "( f1 int not null, f2 int);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName1 +
          " where ( f1, f2 ) in ( select f1 from " + tableName1 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select * from " + tableName1 +
          " where ( f1, f2 ) in ( select f1, f2 from " + tableName1 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select * from " + tableName1 +
          " where f1 in ( select f1 as a, f1 as b from " + tableName1 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_OPERAND_COLUMNS );

    sql = "select * from " + tableName1 +
          " where f1 in (10, ( select f1 as a, f1 as b from " + tableName1 + " ) )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_OPERAND_COLUMNS );

    sql = "select * from " + tableName1 +
          " where f1 in (10, ( select f1 as a, f1 as b from " + tableName1 + " ), " +
                            "( select f1 from " + tableName1 + " ) )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_OPERAND_COLUMNS );
}

TEST_F( UT_subquery_unnesting, IN_subquery_more_than_one_row )
{
    string tableName1 = "in_subquery_more_than_one_row1";
    string tableName2 = "in_subquery_more_than_one_row2";
    InitTable( TEST_DB_NAME, tableName1 );
    InitTable( TEST_DB_NAME, tableName2 );

    string sql( "create table " + tableName1 + "( f1 int not null);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName1 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName2 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName2 + " where f1 in (10, ( select * from " + tableName1 + " where f1 = 3 ) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_SUBQUERY_NO_1_ROW );

    sql = "select * from " + tableName2 +
          " where f1 in (10, ( select * from " + tableName1 + " where f1 = 2 ), " +
                            "( select * from " + tableName1 + " where f1 = 3 ) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_SUBQUERY_NO_1_ROW );
}

TEST_F( UT_subquery_unnesting, IN_subquery_correlated )
{
    string tableName1 = "in_subquery_subquery_correlated1";
    string tableName2 = "in_subquery_subquery_correlated2";
    InitTable( TEST_DB_NAME, tableName1 );
    InitTable( TEST_DB_NAME, tableName2 );

    string sql( "create table " + tableName1 + "( f1 int not null);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select f1 from " + tableName1 + " where f1 IN ( select " + tableName1 + ".f1 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select f1 from " + tableName1 +
          " where f1 IN ( select 1 from " + tableName2 + " where " +
          tableName1 + ".f1 = " + tableName2 + ".f1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select f1 from " + tableName1 +
          " where f1 IN ( select f1 from " + tableName2 + " where " +
          tableName1 + ".f1 = " + tableName2 + ".f1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

TEST_F( UT_subquery_unnesting, IN_subquery_mix_const_and_select )
{
    string tableName1 = "in_subquery_mix_const_and_select1";
    string tableName2 = "in_subquery_mix_const_and_select2";
    InitTable( TEST_DB_NAME, tableName1 );
    InitTable( TEST_DB_NAME, tableName2 );

    string sql( "create table " + tableName1 + "( f1 int not null);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName1 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName2 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName2 +
          " where f1 in (10, ( select * from " + tableName1 + " where f1 = 2 ), " +
                            "( select * from " + tableName1 + " where f1 = 1 ) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );
}

TEST_F( UT_subquery_unnesting, IN_subquery_const_select_expr )
{
    string tableName1 = "in_subquery_const_select_expr1";
    string tableName2 = "in_subquery_const_select_expr2";
    string tableName3 = "in_subquery_const_select_expr3";
    InitTable( TEST_DB_NAME, tableName1 );
    InitTable( TEST_DB_NAME, tableName2 );
    InitTable( TEST_DB_NAME, tableName3 );

    string sql( "create table " + tableName1 + "( f1 int not null);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName3 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName1 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName2 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select f1 from " + tableName1 + " where f1 IN ( select 2 from " + tableName1 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select 4 from " + tableName1 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select 3 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 3 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select 1 + 2 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 3 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select 4 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    // select from empty table
    sql = "select f1 from " + tableName1 + " where f1 IN ( select 3 from " + tableName3 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
}

TEST_F( UT_subquery_unnesting, IN_subquery_calc_select_expr )
{
    string tableName1 = "in_subquery_calc_select_expr1";
    string tableName2 = "in_subquery_calc_select_expr2";
    string tableName3 = "in_subquery_calc_select_expr3";
    InitTable( TEST_DB_NAME, tableName1 );
    InitTable( TEST_DB_NAME, tableName2 );
    InitTable( TEST_DB_NAME, tableName3 );

    string sql( "create table " + tableName1 + "( f1 int not null);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName3 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName1 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName2 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select f1 from " + tableName1 + " where f1 IN ( select f1 + 1 from " + tableName1 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 3 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select f1 + 1 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 3 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select f1 + 3 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select f1 + 4 from " + tableName3 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
}

TEST_F( UT_subquery_unnesting, IN_subquery_aggfunc_select_expr )
{
    string tableName1 = "in_subquery_aggfunc_select_expr1";
    string tableName2 = "in_subquery_aggfunc_select_expr2";
    string tableName3 = "in_subquery_aggfunc_select_expr3";
    InitTable( TEST_DB_NAME, tableName1 );
    InitTable( TEST_DB_NAME, tableName2 );
    InitTable( TEST_DB_NAME, tableName3 );

    string sql( "create table " + tableName1 + "( f1 int not null);" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName3 + "( f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName1 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName2 + " values ( 1 ), ( 2 ), ( 3 ), ( 3 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select f1 from " + tableName1 + " where f1 IN ( select max( f1 ) from " + tableName1 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 3 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select min( f1 ) + 1 from " + tableName1 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select min( f1 ) + max( f1 ) - 2 from " + tableName1 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );

    // different tables
    sql = "select f1 from " + tableName1 + " where f1 IN ( select max( f1 ) from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 3 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select min( f1 ) + 1 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );

    sql = "select f1 from " + tableName1 + " where f1 IN ( select min( f1 ) + max( f1 ) - 2 from " + tableName2 + ")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
}