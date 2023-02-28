#include "gtest/gtest.h"
#include "frontend/SQLExecutor.h"
#include "schema/SchemaManager.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "datatypes/AriesCastFunctions.hxx"
#include "../../TestUtils.h"

using namespace aries_test;
using namespace aries_engine;
using namespace aries_acc;
using namespace std;

const string TEST_DB_NAME( "ut_function_cast" );

class UT_function_cast : public testing::Test
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

TEST_F( UT_function_cast, to_int_not_supported )
{
    string tableName = "t_to_int_not_supported";

    ///////////////////////////////////////////
    // not supported cast
    InitTable( TEST_DB_NAME, tableName );
    string sql( "create table " + tableName + "( f1 date, f2 time, f3 datetime, f4 timestamp, f5 year, f6 decimal, f7 bigint )" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    for ( int i = 1; i <= 7; ++i )
    {
        sql = "select cast( f" + std::to_string( i ) + " as signed ) from " + tableName;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
        ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    }
}

TEST_F( UT_function_cast, to_int_nullable )
{
    string tableName = "t_to_int_nullable";

    /* not supported for now
    ///////////////////////////////////////////
    // float and double to int
    InitTable( TEST_DB_NAME, tableName );
    string sql = "create table " + tableName + "( f1 float, f2 double)";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 1.0000001, 1.000000001 ), (1.5, 1.5 ), (1.6, 1.6 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select cast( f1 as signed ), cast( f2 as signed ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 2 ), 1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 2 ), 1 );
    */

    // integer to integer
    InitTable( TEST_DB_NAME, tableName );
    string sql = "create table " + tableName + "( f1 bool, f2 tinyint, f3 smallint, f4 int, f5 char(1024) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    int32_t i8 = INT8_MAX;
    int32_t i16 = INT16_MAX;
    int32_t i32 = INT32_MAX;
    int64_t i64 = INT64_MAX;
    sql = "insert into " + tableName + " values ( " +
          std::to_string( i8 ) + ", " +
          std::to_string( i8 ) + ", " +
          std::to_string( i16 ) + ", " +
          std::to_string( i32 ) + ", '" +
          std::to_string( i32 ) + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt8( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableInt8( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetNullableString( 0 ), std::to_string( i32 ) );

    ///////////////////////////////////////////
    // cast to signed int
    sql = "select cast( f1 as signed ), cast( f2 as signed ), cast( f3 as signed ), cast( f4 as signed ), cast( f5 as signed ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), i32 );

    // string value overflow
    sql = "insert into " + tableName + " values ( " +
          std::to_string( i8 ) + ", " +
          std::to_string( i8 ) + ", " +
          std::to_string( i16 ) + ", " +
          std::to_string( i32 ) + ", '" +
          std::to_string( i64 ) + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    sql = "select cast( f5 as signed ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ), -1 ); // overflow

    ///////////////////////////////////////////
    // cast to signed big int
    sql = "select cast( f1 as bigint ), cast( f2 as bigint ), cast( f3 as bigint ), cast( f4 as bigint ), cast( f5 as bigint ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ), i32 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 1 ), i64 );
}

TEST_F( UT_function_cast, to_int_not_nullable )
{
    string tableName = "t_int_not_nullable";

    /*
    ///////////////////////////////////////////
    // float and double to int
    InitTable( TEST_DB_NAME, tableName );
    string sql = "create table " + tableName + "( f1 float not null, f2 double not null)";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 1.0000001, 1.000000001 ), (1.5, 1.5 ), (1.6, 1.6 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select cast( f1 as signed ), cast( f2 as signed ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 1 );
    */

    // integer to integer
    InitTable( TEST_DB_NAME, tableName );
    string sql = "create table " + tableName + "( f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 char(1024) not null )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    int32_t i8 = INT8_MAX;
    int32_t i16 = INT16_MAX;
    int32_t i32 = INT32_MAX;
    int64_t i64 = INT64_MAX;
    sql = "insert into " + tableName + " values ( " +
          std::to_string( i8 ) + ", " +
          std::to_string( i8 ) + ", " +
          std::to_string( i16 ) + ", " +
          std::to_string( i32 ) + ", '" +
          std::to_string( i32 ) + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt8( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt8( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt16( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetString( 0 ), std::to_string( i32 ) );

    ///////////////////////////////////////////
    // cast to signed int
    sql = "select cast( f1 as signed ), cast( f2 as signed ), cast( f3 as signed ), cast( f4 as signed ), cast( f5 as signed ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), i32 );

    // string value overflow
    sql = "insert into " + tableName + " values ( " +
          std::to_string( i8 ) + ", " +
          std::to_string( i8 ) + ", " +
          std::to_string( i16 ) + ", " +
          std::to_string( i32 ) + ", '" +
          std::to_string( i64 ) + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    sql = "select cast( f5 as signed ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), -1 ); // overflow

    ///////////////////////////////////////////
    // cast to signed big int
    sql = "select cast( f1 as bigint ), cast( f2 as bigint ), cast( f3 as bigint ), cast( f4 as bigint ), cast( f5 as bigint ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), i32 );
    ASSERT_EQ( columnBuff->GetInt64( 1 ), i64 );
}

TEST_F( UT_function_cast, to_char )
{
    string tableName = "t_to_char";
    InitTable( TEST_DB_NAME, tableName );

    // int to char
    string sql = "create table " + tableName +
                 "( f1 bool not null, "
                 "f2 tinyint not null, "
                 "f3 smallint not null, "
                 "f4 int not null, "
                 "f5 bigint not null, "
                 "f6 float not null, "
                 "f7 double not null, "
                 "f8 decimal not null, "
                 "f9 char(4) not null, "
                 "f10 date, "
                 "f11 time, "
                 "f12 datetime, "
                 "f13 timestamp, "
                 "f14 year )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    for ( int i = 1; i <= 14; ++i )
    {
        sql = "select cast( f" + std::to_string( i ) + " as char ) from " + tableName;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
        ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    }
}

TEST_F( UT_function_cast, to_date_not_supported )
{
    string tableName = "t_to_date_not_supported";
    InitTable( TEST_DB_NAME, tableName );

    // int to char
    string sql = "create table " + tableName +
                 "( f1 bool not null, "
                 "f2 tinyint not null, "
                 "f3 smallint not null, "
                 "f4 int not null, "
                 "f5 bigint not null, "
                 "f6 float not null, "
                 "f7 double not null, "
                 "f8 decimal not null, "
                 "f9 time, "
                 "f10 year )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    for ( int i = 1; i <= 10; ++i )
    {
        sql = "select cast( f" + std::to_string( i ) + " as date ) from " + tableName;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
        ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    }
}

TEST_F( UT_function_cast, to_date_nullable )
{
    string tableName = "t_to_date_nullable";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 date, f2 datetime, f3 timestamp, f4 char( 20 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    string date1 = "2021-01-18";
    string datetime1 = "2021-01-18 11:26:11";
    string timestamp1 = "2021-01-18 11:26:12";
    string char1 = "2021-01-18 11:26:12";
    sql = "insert into " + tableName + " values( '" + date1 + "', '" + datetime1 + "', '" + timestamp1 + "', '" + char1 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select cast( f1 as date ), cast( f2 as date ), cast( f3 as date ), cast( f4 as date ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), date1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), date1 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), date1 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), date1 );
}
TEST_F( UT_function_cast, to_date_not_nullable )
{
    string tableName = "t_to_date_not_nullable";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 date not null, f2 datetime not null, f3 timestamp not null, f4 char( 20 ) not null )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    string date1 = "2021-01-18";
    string datetime1 = "2021-01-18 11:26:11";
    string timestamp1 = "2021-01-18 11:26:12";
    string char1 = "2021-01-18 11:26:12";
    sql = "insert into " + tableName + " values( '" + date1 + "', '" + datetime1 + "', '" + timestamp1 + "', '" + char1 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select cast( f1 as date ), cast( f2 as date ), cast( f3 as date ), cast( f4 as date ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), date1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), date1 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), date1 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), date1 );
}

TEST_F( UT_function_cast, to_datetime_not_supported )
{
    string tableName = "t_to_datetime_not_supported";
    InitTable( TEST_DB_NAME, tableName );

    // int to char
    string sql = "create table " + tableName +
                 "( f1 bool not null, "
                 "f2 tinyint not null, "
                 "f3 smallint not null, "
                 "f4 int not null, "
                 "f5 bigint not null, "
                 "f6 float not null, "
                 "f7 double not null, "
                 "f8 decimal not null, "
                 "f9 time, "
                 "f10 year )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    for ( int i = 1; i <= 10; ++i )
    {
        sql = "select cast( f" + std::to_string( i ) + " as datetime ) from " + tableName;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
        ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    }
}
TEST_F( UT_function_cast, to_datetime_nullable )
{
    string tableName = "t_to_datetime_nullable";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 date, f2 datetime, f3 timestamp, f4 char( 20 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    string date1 = "2021-01-18";
    string datetime1 = "2021-01-18 11:26:11";
    string timestamp1 = "2021-01-18 11:26:12";
    string char1 = "2021-01-18 11:26:12";
    sql = "insert into " + tableName + " values( '" + date1 + "', '" + datetime1 + "', '" + timestamp1 + "', '" + char1 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select cast( f1 as datetime ), cast( f2 as datetime ), cast( f3 as datetime ), cast( f4 as datetime ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableDatetimeAsString( 0 ), date1 + " 00:00:00");

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableDatetimeAsString( 0 ), datetime1 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableDatetimeAsString( 0 ), timestamp1 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableDatetimeAsString( 0 ), char1 );
}

TEST_F( UT_function_cast, to_datetime_not_nullable )
{
    string tableName = "t_to_datetime_not_nullable";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 date not null, f2 datetime not null, f3 timestamp not null, f4 char( 30 ) not null )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    string date1 = "2021-01-18";
    string datetime1 = "2021-01-18 11:26:11";
    string timestamp1 = "2021-01-18 11:26:12";
    sql = "insert into " + tableName + " values( '" + date1 + "', '" + datetime1 + "', '" + timestamp1 + "', '" + timestamp1 + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select f4 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), timestamp1 );

    sql = "select cast( f1 as datetime ), cast( f2 as datetime ), cast( f3 as datetime ), cast( f4 as datetime ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDatetimeAsString( 0 ), date1 + " 00:00:00");

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetDatetimeAsString( 0 ), datetime1 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetDatetimeAsString( 0 ), timestamp1 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetDatetimeAsString( 0 ), timestamp1 );
}

TEST_F( UT_function_cast, to_decimal_not_supported )
{
    string tableName = "t_to_decimal_not_supported";
    InitTable( TEST_DB_NAME, tableName );

    // int to char
    string sql = "create table " + tableName +
                 "( f1 float, "
                 "f2 double, "
                 "f3 date, "
                 "f4 time, "
                 "f5 datetime, "
                 "f6 timestamp, "
                 "f7 year )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    for ( int i = 1; i <= 7; ++i )
    {
        sql = "select cast( f" + std::to_string( i ) + " as decimal ) from " + tableName;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
        ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    }
}

TEST_F( UT_function_cast, to_decimal_nullable )
{
    string tableName = "t_to_decimal_nullable";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 bool, f2 tinyint, f3 smallint, f4 int, f5 bigint, f6 char( 40 ), f7 decimal( 35, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    int32_t i8 = INT8_MAX;
    int32_t i16 = INT16_MAX;
    int32_t i32 = INT32_MAX;
    int64_t i64 = INT64_MAX;
    string decStr = "9999999999999999999999999999999999.8";
    sql = "insert into " + tableName + " values ( " +
          std::to_string( i8 ) + ", " +
          std::to_string( i8 ) + ", " +
          std::to_string( i16 ) + ", " +
          std::to_string( i32 ) + ", " +
          std::to_string( i64 ) + ", '" +
          std::to_string( i64 ) + "', '" + decStr + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    sql = "select f5, f6, f7 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt64( 0 ), i64 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableString( 0 ), std::to_string( i64 ) );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableDecimalAsString( 0 ), decStr );

    // cast to less precision
    sql = "select cast( f1 as decimal ), cast( f2 as decimal ), cast( f3 as decimal ), cast( f4 as decimal ), cast( f5 as decimal ), cast( f6 as decimal ), cast( f7 as decimal ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), 9999999999 );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), 9999999999 );

    columnBuff = resTable->GetColumnBuffer( 7 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), 9999999999 );

    // calc result to decimal
    sql = "select cast( f7 + 0.1 as decimal ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), 9999999999 );

    // cast to the same precision
    sql = "select cast( f5 as decimal(19) ), cast( f6 as decimal(19) ), cast( f7 as decimal( 35, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), i64 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableDecimal( 0 ), i64 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), decStr );

    // cast to bigger precision
    sql = "select cast( f7 as decimal( 36, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), decStr );

    // calc result to decimal
    sql = "select cast( f7 + 0.1 as decimal( 35, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), "9999999999999999999999999999999999.9" );

    // calc result to bigger precision
    sql = "select cast( f7 + 0.1 as decimal( 36, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), "9999999999999999999999999999999999.9" );
}

TEST_F( UT_function_cast, to_decimal_not_nullable )
{
    string tableName = "t_to_decimal_nullable";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 bigint not null, f6 char( 40 ) not null, f7 decimal( 35, 1 )  not null)";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    int32_t i8 = INT8_MAX;
    int32_t i16 = INT16_MAX;
    int32_t i32 = INT32_MAX;
    int64_t i64 = INT64_MAX;
    string decStr = "9999999999999999999999999999999999.8";
    sql = "insert into " + tableName + " values ( " +
          std::to_string( i8 ) + ", " +
          std::to_string( i8 ) + ", " +
          std::to_string( i16 ) + ", " +
          std::to_string( i32 ) + ", " +
          std::to_string( i64 ) + ", '" +
          std::to_string( i64 ) + "', '" + decStr + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    sql = "select f5, f6 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), i64 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), std::to_string( i64 ) );

    // cast to less precision
    sql = "select cast( f1 as decimal ), cast( f2 as decimal ), cast( f3 as decimal ), cast( f4 as decimal ), cast( f5 as decimal ), cast( f6 as decimal ), cast( f7 as decimal ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), i8 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), i16 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), i32 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), 9999999999 );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), 9999999999 );

    columnBuff = resTable->GetColumnBuffer( 7 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), 9999999999 );

    // calc result to decimal
    sql = "select cast( f7 + 0.1 as decimal ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), 9999999999 );

    // cast to the same precision
    sql = "select cast( f5 as decimal(19) ), cast( f6 as decimal(19) ), cast( f7 as decimal( 35, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), i64 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), i64 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), decStr );

    // cast to bigger precision
    sql = "select cast( f7 as decimal( 36, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), decStr );

    // calc result to decimal
    sql = "select cast( f7 + 0.1 as decimal( 35, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), "9999999999999999999999999999999999.9" );

    // calc result to bigger precision
    sql = "select cast( f7 + 0.1 as decimal( 36, 1 ) ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess() );

    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetDecimalAsString( 0 ), "9999999999999999999999999999999999.9" );
}