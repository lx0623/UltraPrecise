#include <gtest/gtest.h>

#include "AriesEngine/AriesInsertNode.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesConstantGenerator.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "../../TestUtils.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace aries_test;

extern bool STRICT_MODE;

ARIES_UNIT_TEST_CLASS( TestAriesInsertNode )
{
protected:
    std::string db_name = "test_insert";
    std::string table_name = "testariesinsertnode";
    void SetUp() override
    {
        InitTable( db_name, table_name );
        auto sql = R"(
 create table testariesinsertnode(
     id int,
     name char(64),
     birth_day date,
     age int not null default 18,
     location char(64)
 );
        )";

        aries_test::ExecuteSQL( "drop table if exists " + table_name, db_name );
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
        EXPECT_TRUE( result->IsSuccess() );
    }

    void TearDown() override
    {
        STRICT_MODE = true;
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( "drop database if exists " + db_name, "" );
        EXPECT_TRUE( result->IsSuccess() );
    }
};

ARIES_UNIT_TEST_F( TestAriesInsertNode, GetNext )
{
    auto transaction = AriesTransManager::GetInstance().NewTransaction();
    auto node = std::make_shared< AriesInsertNode >( transaction, db_name, table_name );

    std::vector< int > column_ids( 3 );
    std::iota( column_ids.begin(), column_ids.end(), 1 );
    node->SetColumnIds( column_ids );

    auto source = GenerateConstNode( db_name, table_name, column_ids );
    node->SetSourceNode( source );

    ASSERT_TRUE( node->Open() );

    auto result = node->GetNext();
    ASSERT_EQ( result.Status, AriesOpNodeStatus::END );
    ASSERT_TRUE( result.TableBlock );

    auto& table = result.TableBlock;
    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnCount(), 1 );

    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt64AsString( 0 ), "2" );

    result = node->GetNext();
    ASSERT_EQ( result.Status, AriesOpNodeStatus::END );
    ASSERT_TRUE( result.TableBlock );

    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnCount(), 1 );

    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt64AsString( 0 ), "0" );
    node->Close();

    AriesTransManager::GetInstance().EndTransaction( transaction, TransactionStatus::COMMITTED );

    auto mvcc_table = AriesMvccTableManager::GetInstance().getMvccTable( db_name, table_name );

    auto read_transaction = AriesTransManager::GetInstance().NewTransaction();
    std::vector< int > columns_id_for_read( 5 );
    std::iota( columns_id_for_read.begin(), columns_id_for_read.end(), 1 );
    auto table_blcok = mvcc_table->GetTable( read_transaction, columns_id_for_read );
    AriesTransManager::GetInstance().EndTransaction( read_transaction, TransactionStatus::COMMITTED );

    ASSERT_TRUE( table_blcok);

    ASSERT_EQ( table_blcok->GetRowCount(), 2 );
    ASSERT_EQ( table_blcok->GetColumnCount(), 5 );

    ASSERT_EQ( table_blcok->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "1" );
    ASSERT_EQ( table_blcok->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );

    ASSERT_EQ( table_blcok->GetColumnBuffer( 2 )->GetString( 0 ), "abc" );
    ASSERT_EQ( table_blcok->GetColumnBuffer( 2 )->GetString( 1 ), "efg" );

    ASSERT_EQ( table_blcok->GetColumnBuffer( 3 )->GetDateAsString( 0 ), "2019-10-10" );
    ASSERT_EQ( table_blcok->GetColumnBuffer( 3 )->GetDateAsString( 1 ), "2019-10-12" );

    ASSERT_EQ( table_blcok->GetColumnBuffer( 4 )->GetInt32AsString( 0 ), "18" );
    ASSERT_EQ( table_blcok->GetColumnBuffer( 4 )->GetInt32AsString( 1 ), "18" );

    ASSERT_EQ( table_blcok->GetColumnBuffer( 5 )->GetString( 0 ), "NULL" );
    ASSERT_EQ( table_blcok->GetColumnBuffer( 5 )->GetString( 1 ), "NULL" );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_values_non_literal )
{
    std::string dbName = "test_insert";
    std::string tableName = "insert_values_non_literal";

    InitTable(dbName, tableName);

    auto sql = "create table " + tableName + "( f1 int )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( a )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_FIELD_ERROR );

    sql = "insert into " + tableName + " values ( 1 + a )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_FIELD_ERROR );

    sql = "insert into " + tableName + " values ( cast( a as signed) )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_FIELD_ERROR );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_literal_wrong_value_count )
{
    std::string dbName = "test_insert";
    std::string tableName = "insert_literal_wrong_value_count";

    InitTable(dbName, tableName);

    auto sql = "create table " + tableName + "( f1 int, f2 char(3) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WRONG_VALUE_COUNT_ON_ROW );

    sql = "insert into " + tableName + "( f1, f2 ) values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WRONG_VALUE_COUNT_ON_ROW );

}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_column_wrong_value_count )
{
    std::string dbName = "test_insert";
    std::string tableName1 = "insert_column_wrong_value_count1";
    std::string tableName2 = "insert_column_wrong_value_count2";

    InitTable(dbName, tableName1);
    InitTable(dbName, tableName2);

    auto sql = "create table " + tableName1 + "( f1 int, f2 char(3) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int, f2 char(3) )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName1 + " select f1 from " + tableName2;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WRONG_VALUE_COUNT_ON_ROW );

    sql = "insert into " + tableName1 + "( f1, f2 ) select f1 from " + tableName2;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WRONG_VALUE_COUNT_ON_ROW );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, not_null_column_default_values )
{
    std::string dbName = "test_insert";
    std::string tableName = "not_null_column_default_values";

    InitTable(dbName, tableName);

    auto sql = "create table " + tableName + "( " +
               "f1 char(1) not null, " +
               "f2 tinyint not null, " +
               "f3 smallint not null, " +
               "f4 int not null, " +
               "f5 bigint not null, " +
               "f6 float not null, " +
               "f7 double not null, " +
               "f8 decimal(10, 6) not null, " +
               "f9 date not null, " +
               "f10 time not null, " +
               "f11 year not null, " +
               "f12 datetime not null, " +
               "f13 timestamp not null, " +
               "f14 bool not null ) ";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    int iValue = 1;
    double fValue = 1.000001;
    string strValue( "a" );
    string intValue = std::to_string( iValue );
    string floatValue = std::to_string( fValue );
    string dateValue( "2020-12-30" );
    string timeValue( "17:00:00" );
    string yearValue( "2021" );
    string datetimeValue( "2020-12-30 17:00:00" );
    string timestampValue( "2020-12-30 17:00:00.000001" );

    // char
    sql = "insert into " + tableName + "( f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 )" +
          " values ( " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_NO_DEFAULT_FOR_FIELD );

    /*
    ASSERT_TRUE( result->IsSuccess() );
    sql = "select f1 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "" );

    // int8
    sql = "insert into " + tableName + "( f1, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 )" +
          " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );
    sql = "select f2 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 0 );

    // int16
    sql = "insert into " + tableName + "( f1, f2, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 )" +
          " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );
    sql = "select f3 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt16( 0 ), 0 );
    */
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, not_null_columns )
{
    std::string dbName = "test_insert";
    std::string tableName = "not_null_columns";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( " +
               "f1 char(1) not null, " +
               "f2 tinyint not null, " +
               "f3 smallint not null, " +
               "f4 int not null, " +
               "f5 bigint not null, " +
               "f6 float not null, " +
               "f7 double not null, " +
               "f8 decimal(10, 6) not null, " +
               "f9 date not null, " +
               "f10 time not null, " +
               "f11 year not null, " +
               "f12 datetime not null, " +
               "f13 timestamp not null, " +
               "f14 bool not null ) ";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    int iValue = 1;
    double fValue = 1.000001;
    string strValue( "a" );
    string intValue = std::to_string( iValue );
    string floatValue = std::to_string( fValue );
    string dateValue( "2020-12-30" );
    string timeValue( "17:00:00" );
    string yearValue( "2021" );
    string datetimeValue( "2020-12-30 17:00:00" );
    string timestampValue( "2020-12-30 17:00:00.000001" );

    sql = "insert into " + tableName + " values ( " +
          "null, " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          "null, " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          "null, " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          "null, " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          "null, " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          "null, " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          "null, " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          "null, " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "null, " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "null, " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          "null, " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "null, " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "null, " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          "null" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto resultRowCount = resTable->GetRowCount();
    auto columnCount = resTable->GetColumnCount();
    ASSERT_EQ( resultRowCount, 1 );
    ASSERT_EQ( columnCount, 14 );

    int colId = 1;
    auto columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetString( 0 ), strValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetInt8( 0 ), iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetInt16( 0 ), iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->GetFloat( 0 ) - fValue < 0.0000001 );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetDouble( 0 ), fValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), floatValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), dateValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetTimeAsString( 0 ), timeValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetYearAsString( 0 ), yearValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetDatetimeAsString( 0 ), datetimeValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetTimestampAsString( 0 ), timestampValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetInt8( 0 ), iValue );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, nullable_columns )
{
    std::string dbName = "test_insert";
    std::string tableName = "nullable_columns";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( " +
               "f1 char(1), " +
               "f2 tinyint, " +
               "f3 smallint, " +
               "f4 int, " +
               "f5 bigint, " +
               "f6 float, " +
               "f7 double, " +
               "f8 decimal(10, 6), " +
               "f9 date, " +
               "f10 time, " +
               "f11 year, " +
               "f12 datetime, " +
               "f13 timestamp, " +
               "f14 bool ) ";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    int iValue = 1;
    double fValue = 1.000001;
    string strValue( "a" );
    string intValue = std::to_string( iValue );
    string floatValue = std::to_string( fValue );
    string dateValue( "2020-12-30" );
    string timeValue( "17:00:00" );
    string yearValue( "2021" );
    string datetimeValue( "2020-12-30 17:00:00" );
    string timestampValue( "2020-12-30 17:00:00.000001" );

    sql = "insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " +
          intValue + ", " + // bigint
          floatValue + ", " +
          floatValue + ", " +
          floatValue + ", " + // decimal
          "'" + dateValue + "', " +
          "'" + timeValue + "', " +
          yearValue + ", " +
          "'" + datetimeValue + "', " +
          "'" + timestampValue + "', " +
          intValue +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto resultRowCount = resTable->GetRowCount();
    auto columnCount = resTable->GetColumnCount();
    ASSERT_EQ( resultRowCount, 1 );
    ASSERT_EQ( columnCount, 14 );

    int colId = 1;
    auto columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableString( 0 ), strValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableInt8( 0 ).value, iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableInt16( 0 ).value, iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ).value, iValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->GetNullableFloat( 0 ).value - fValue < 0.0000001 );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableDouble( 0 ).value, fValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableDecimalAsString( 0 ), floatValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), dateValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableTimeAsString( 0 ), timeValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableYearAsString( 0 ), yearValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableDatetimeAsString( 0 ), datetimeValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableTimestampAsString( 0 ), timestampValue );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_EQ( columnBuff->GetNullableInt8( 0 ).value, iValue );

    // insert null values
    string nullValues;
    for ( int i = 0; i < 13; ++i )
        nullValues.append( "null," );
    nullValues.append( "null" );

    sql = "insert into " + tableName + " values ( " + nullValues + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    resultRowCount = resTable->GetRowCount();
    columnCount = resTable->GetColumnCount();
    ASSERT_EQ( resultRowCount, 2 );
    ASSERT_EQ( columnCount, 14 );

    colId = 1;
    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isStringDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isInt8DataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isInt16DataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isInt32DataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isInt64DataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isFloatDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isDoubleDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isDecimalDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isDateDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isTimeDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isYearDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isDatetimeDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isTimestampDataNull( 1 ) );

    columnBuff = resTable->GetColumnBuffer( colId++ );
    ASSERT_TRUE( columnBuff->isInt8DataNull( 1 ) );
}

const static string maxLenString( ARIES_MAX_CHAR_WIDTH + 1, 'a' );
void TestInsertCharLiteralsExceedLen( const string& dbName, const string& tableName )
{
    string newValue1 = "1111111";
    string newValue2 = "2";

    // strict mode
    STRICT_MODE = true;

    auto sql = "insert into " + tableName + " values ( '" + newValue1 + "' ), ( '" + newValue2 + "' )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 0 );

    sql = "insert into " + tableName + " values ( '" + maxLenString + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_FIELDLENGTH );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 0 );

    // non strict mode
    STRICT_MODE = false;

    sql = "insert into " + tableName + " values ( '" + newValue1 + "' ), ( '" + newValue2 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 2 );
    ASSERT_EQ( resTable->GetColumnCount(), 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" ); // string is truncated
    ASSERT_EQ( columnBuff->GetString( 1 ), "2" );

    STRICT_MODE = true;
}

void TestInsertCharColumnsExceedLen( const string& dbName,
                                     const string& srcTableName,
                                     const string& dstTableName )
{
    // strict mode
    STRICT_MODE = true;
    string insertValue1( "aa" );
    auto sql = "insert into " + srcTableName + " values ( '" + insertValue1 + "')";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    // non strict mode
    STRICT_MODE = false;
    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "a" ); // string is truncated
}

////////////////////////////////////////////////
void TestInsertNullLiteralIntoNotNullColumn( const string& dbName, const string& tableName )
{
    auto sql = "insert into " + tableName + " values ( null )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, null_into_primary_column )
{
    std::string dbName = "test_insert";
    std::string tableName = "null_into_primary_column";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1) primary key );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    
    TestInsertNullLiteralIntoNotNullColumn( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_literal_null_into_notnull_column )
{
    std::string dbName = "test_insert";
    std::string tableName = "insert_literal_null_into_notnull_column";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1) not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    
    TestInsertNullLiteralIntoNotNullColumn( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_literal_null_into_notnull_dict_column )
{
    std::string dbName = "test_insert";
    std::string tableName = "insert_literal_null_into_notnull_dict_column";
    std::string dictName = "TestAriesInsertNode_insert_literal_null_into_notnull_dict_column";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1) not null encoding bytedict as " + dictName + ");";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    
    TestInsertNullLiteralIntoNotNullColumn( dbName, tableName );
}

////////////////////////////////////////////////
void TestInsertNullColumnIntoNotNullColumn( const string& dbName,
                                            const string& srcTableName,
                                            const string& dstTableName )
{
    auto sql = "insert into " + srcTableName + " values ( null )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_null_column_into_primary_column )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "insert_null_column_into_primary_column_src";
    std::string dstTableName = "insert_null_column_into_primary_column_dst";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(1) );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) primary key );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    
    TestInsertNullColumnIntoNotNullColumn( dbName, srcTableName, dstTableName );
}
ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_null_column_into_notnull_column )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "insert_null_column_into_notnull_column_src";
    std::string dstTableName = "insert_null_column_into_notnull_column_dst";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(1) );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    
    TestInsertNullColumnIntoNotNullColumn( dbName, srcTableName, dstTableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, insert_null_column_into_notnull_dict_column )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "insert_null_column_into_notnull_dict_column_src";
    std::string dstTableName = "insert_null_column_into_notnull_dict_column_dst";
    std::string dictName1 = "TestAriesInsertNode_insert_null_column_into_notnull_dict_column1";
    std::string dictName2 = "TestAriesInsertNode_insert_null_column_into_notnull_dict_column2";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(2) encoding bytedict as " + dictName1 + ");";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) not null encoding bytedict as " + dictName2 + ");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    
    TestInsertNullColumnIntoNotNullColumn( dbName, srcTableName, dstTableName );
}

////////////////////////////
ARIES_UNIT_TEST_F( TestAriesInsertNode, not_null_char_literal_exceed_len )
{
    std::string dbName = "test_insert";
    std::string tableName = "not_null_char_literal_exceed_len";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1) not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharLiteralsExceedLen( dbName, tableName );
}
ARIES_UNIT_TEST_F( TestAriesInsertNode, nullable_char_literal_exceed_len )
{
    std::string dbName = "test_insert";
    std::string tableName = "nullable_char_literal_exceed_len";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1));";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharLiteralsExceedLen( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, not_null_char_dict_literal_exceed_len )
{
    std::string dbName = "test_insert";
    std::string tableName = "not_null_char_dict_literal_exceed_len";
    std::string dictName = "TestAriesInsertNode_not_null_char_dict_literal_exceed_len";

    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 char(1) not null encoding bytedict as " + dictName + ");";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharLiteralsExceedLen( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, nullable_char_dict_literal_exceed_len )
{
    std::string dbName = "test_insert";
    std::string tableName = "nullable_char_dict_literal_exceed_len";
    std::string dictName = "TestAriesInsertNode_nullable_char_dict_literal_exceed_len";

    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 char(1) encoding bytedict as " + dictName + ");";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharLiteralsExceedLen( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, not_null_char_column_exceed_len )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "not_null_char_column_exceed_len_src";
    std::string dstTableName = "not_null_char_column_exceed_len_dst";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(2) not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharColumnsExceedLen( dbName, srcTableName, dstTableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, nullable_char_column_exceed_len )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "nullable_char_column_exceed_len_src";
    std::string dstTableName = "nullable_char_column_exceed_len_dst";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(2) );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharColumnsExceedLen( dbName, srcTableName, dstTableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, not_null_char_dict_column_exceed_len )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "not_null_char_dict_column_exceed_len_src";
    std::string dstTableName = "not_null_char_dict_column_exceed_len_dst";
    std::string dictName1 = "TestAriesInsertNode_not_null_char_dict_column_exceed_len1";
    std::string dictName2 = "TestAriesInsertNode_not_null_char_dict_column_exceed_len2";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(2) not null encoding bytedict as " + dictName1 + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) not null encoding bytedict as " + dictName2 + ");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharColumnsExceedLen( dbName, srcTableName, dstTableName );
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, nullable_char_dict_column_exceed_len )
{
    std::string dbName = "test_insert";
    std::string srcTableName = "nullable_char_dict_column_exceed_len_src";
    std::string dstTableName = "nullable_char_dict_column_exceed_len_dst";
    std::string dictName1 = "TestAriesInsertNode_nullable_char_dict_column_exceed_len1";
    std::string dictName2 = "TestAriesInsertNode_nullable_char_dict_column_exceed_len2";

    InitTable( dbName, srcTableName );
    InitTable( dbName, dstTableName );
    string sql = "create table " + srcTableName + "( f1 char(2) encoding bytedict as " + dictName1 + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + dstTableName + "( f1 char(1) encoding bytedict as " + dictName2 + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestInsertCharColumnsExceedLen( dbName, srcTableName, dstTableName );
}


ARIES_UNIT_TEST_F( TestAriesInsertNode, into_decimal_invalid_values )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_insert";
    std::string tableName = "into_decimal_invalid_values";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 36 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string invalidDec1( "0.1111111111111111111111111111111" ); // 31 fraction digits
    string invalidDec2( "1111111.111111111111111111111111111111" ); // 7 ingeter digits and 30 fraction digits
    string maxValue1( "999999999999999999999999999999999999" ); // 36 ingeter digits

    // decimal into decimal
    sql = "insert into " + tableName + " values ( " + invalidDec1 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_SCALE );

    sql = "insert into " + tableName + " values ( " + invalidDec2 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_PRECISION );

    // string into decimal
    sql = "insert into " + tableName + " values ( '" + invalidDec1 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_SCALE );

    sql = "insert into " + tableName + " values ( '" + invalidDec2 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_PRECISION );

    sql = "insert into " + tableName + " values ( " + maxValue1 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), maxValue1 );

    sql = "insert into " + tableName + " values ( '" + maxValue1 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), maxValue1 );

    STRICT_MODE = false;
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, int_into_decimal )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_insert";
    std::string tableName = "int_into_decimal";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 3, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 123 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // big int to decimal
    sql = "insert into " + tableName + " values ( " + std::to_string( INT64_MAX ) + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    STRICT_MODE = false;
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, float_into_decimal )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_insert";
    std::string tableName = "float_into_decimal";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 3, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 1.23 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    // ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto column = resTable->GetColumnBuffer( 1 );
    column->Dump();

    STRICT_MODE = false;
}

ARIES_UNIT_TEST_F( TestAriesInsertNode, string_into_decimal )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_insert";
    std::string tableName = "string_into_decimal";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 3, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( '123' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // big int to decimal
    sql = "insert into " + tableName + " values ( '" + std::to_string( INT64_MAX ) + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    sql = "insert into " + tableName + " values ( '1.23' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    // ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto column = resTable->GetColumnBuffer( 1 );
    column->Dump();

    STRICT_MODE = false;
}