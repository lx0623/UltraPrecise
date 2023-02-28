#include <unistd.h>
#include <gtest/gtest.h>
#include <string>

#include "server/mysql/include/mysqld.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "datatypes/AriesDatetimeTrans.h"
#include "../../TestUtils.h"

using namespace std;
using namespace aries_engine;
using namespace aries_acc;
using namespace aries_test;

static string TEST_DB_NAME( "test_datatype_convert" );
class UT_datatype_convert: public testing::Test
{
private:
protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
        string sql = "drop database if exists " + TEST_DB_NAME;
        ExecuteSQL( sql, "" );
    }
};
TEST_F(UT_datatype_convert, convert_int_error)
{
    string tableName = "t_convert_int_error";
    InitTable( TEST_DB_NAME, tableName );

    // char
    string sql = "create table " + tableName + "(f1 char not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // date
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 date not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1201 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // time
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 time not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1900 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // datetime
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 datetime not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1900 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // timestamp
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 timestamp not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1900 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // year
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 year not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1234567 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
}

TEST_F(UT_datatype_convert, convert_int)
{
    string tableName = "t_convert_int";
    InitTable( TEST_DB_NAME, tableName );

    // convert int8 to all int type
    string sql = "create table " + tableName + "(f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 bigint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1, 1, 1, 1, 1 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = " insert into " + tableName + " values ( " +
          std::to_string( INT8_MAX ) + ", " +
          std::to_string( INT8_MAX ) + ", " +
          std::to_string( INT16_MAX ) + ", " +
          std::to_string( INT32_MAX ) + ", " +
          std::to_string( INT64_MAX ) +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );
    sql = " insert into " + tableName + " values ( " +
          std::to_string( INT8_MIN ) + ", " +
          std::to_string( INT8_MIN ) + ", " +
          std::to_string( INT16_MIN ) + ", " +
          std::to_string( INT32_MIN ) + ", " +
          std::to_string( INT64_MIN ) +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    ASSERT_EQ( column->GetInt8( 1 ), INT8_MAX );
    ASSERT_EQ( column->GetInt8( 2 ), INT8_MIN );

    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    ASSERT_EQ( column->GetInt8( 1 ), INT8_MAX );
    ASSERT_EQ( column->GetInt8( 2 ), INT8_MIN );

    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt16( 0 ), 1 );
    ASSERT_EQ( column->GetInt16( 1 ), INT16_MAX );
    ASSERT_EQ( column->GetInt16( 2 ), INT16_MIN );

    column = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), INT32_MAX );
    ASSERT_EQ( column->GetInt32( 2 ), INT32_MIN );

    column = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
    ASSERT_EQ( column->GetInt64( 1 ), INT64_MAX );
    ASSERT_EQ( column->GetInt64( 2 ), INT64_MIN );

    // bool overflow
    int64_t oorValue = INT8_MAX + 1;
    string oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValue = ( int64_t )INT32_MAX + 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775807";
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775808";
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // bool underflow
    oorValue = INT8_MIN - 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775808";
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);

    oorValueStr = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( " + oorValueStr + ", 127, 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);

    // int8 overflow
    oorValue = INT8_MAX + 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( 127, " + oorValueStr + ", 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775807";
    sql = " insert into " + tableName + " values ( 127, " + oorValueStr + ", 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, " + oorValueStr + ", 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int8 underflow
    oorValue = INT8_MIN - 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( 127, " + oorValueStr + ", 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, " + oorValueStr + ", 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( 127, " + oorValueStr + ", 127, 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int16 overflow
    oorValue = INT16_MAX + 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( 127, 127, " + oorValueStr + ", 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775807";
    sql = " insert into " + tableName + " values ( 127, 127, " + oorValueStr + ", 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, 127, " + oorValueStr + ", 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int16 underflow
    oorValue = INT16_MIN - 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( 127, 127, " + oorValueStr + ", 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, 127, " + oorValueStr + ", 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( 127, 127, " + oorValueStr + ", 127, 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int32 overflow
    oorValue = ( int64_t )INT32_MAX + 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( 127, 127, 127, " + oorValueStr + ", 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775807";
    sql = " insert into " + tableName + " values ( 127, 127, 127, " + oorValueStr + ", 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, 127, 127, " + oorValueStr + ", 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int32 underflow
    oorValue = ( int64_t )INT32_MIN - 1;
    oorValueStr = std::to_string( oorValue );
    sql = " insert into " + tableName + " values ( 127, 127, 127, " + oorValueStr + ", 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, 127, 127, " + oorValueStr + ", 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    oorValueStr = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( 127, 127, 127, " + oorValueStr + ", 127 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int64 overflow
    oorValueStr = "9223372036854775808";
    sql = " insert into " + tableName + " values ( 127, 127, 127, 127, " + oorValueStr + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int64 underflow
    // convert to double can't do the job, why?
    // dOorValue = ( double )( ( double )INT64_MIN - 1 );
    // string tmpStr = std::to_string( INT64_MIN - 1 );
    // sprintf( valueBuff, "%.0f", dOorValue );
    // sql = " insert into " + tableName + " values ( 127, 127, 127, 127, " + valueBuff + " );";
    oorValueStr = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( 127, 127, 127, 127, " + oorValueStr + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int to float, double and decimal
    InitTable( TEST_DB_NAME, tableName );

    sql = "create table " + tableName + "(f1 float not null, f2 double not null, f3 decimal(36) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1, 1, 1 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    // ok to insert int64_max and int64_min to float, double and decimal
    oorValueStr = "9223372036854775808";
    sql = " insert into " + tableName + " values ( " +
            oorValueStr + ", " +
            oorValueStr + ", " +
            oorValueStr +
            " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    oorValueStr = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( " +
            oorValueStr + ", " +
            oorValueStr + ", " +
            oorValueStr +
            " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    column = resTable->GetColumnBuffer( 1 );
    int decPrecision = 36;
    int decScale = 0;
    ASSERT_TRUE( column->GetFloat( 0 ) - 1 < 0.000001 );
    aries_acc::Decimal decValue( decPrecision,
                                 decScale,
                                 ARIES_MODE_STRICT_ALL_TABLES,
                                 "9223372036854775808" );
    ASSERT_TRUE( decValue - column->GetFloat( 1 ) < 0.000001 );
    decValue = aries_acc::Decimal( decPrecision,
                                   decScale,
                                   ARIES_MODE_STRICT_ALL_TABLES,
                                   "-9223372036854775809" );
    ASSERT_TRUE( decValue - column->GetFloat( 2 ) < 0.000001 );

    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );
    decValue = aries_acc::Decimal( decPrecision,
                                   decScale,
                                   ARIES_MODE_STRICT_ALL_TABLES,
                                   "9223372036854775808" );
    ASSERT_EQ( decValue, column->GetDouble( 1 ) );
    decValue = aries_acc::Decimal( decPrecision,
                                   decScale,
                                   ARIES_MODE_STRICT_ALL_TABLES,
                                   "-9223372036854775809" );
    ASSERT_EQ( decValue, column->GetDouble( 2 ) );

    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "1" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "9223372036854775808" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "-9223372036854775809" );

    // year
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 year not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( 1900 );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetYearAsString( 0 ), "1900" );
}

TEST_F(UT_datatype_convert, convert_char_error)
{
    string tableName = "t_convert_char_error";
    InitTable( TEST_DB_NAME, tableName );

    // convert string to all int type
    string sql = "create table " + tableName + "(f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 bigint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // incorrect value
    string valueStr = "xxx";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', '1', '1', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    sql = " insert into " + tableName + " values ( " +
          "'1', '" + valueStr + "', '1', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    sql = " insert into " + tableName + " values ( " +
          "'1', '1', '" + valueStr + "', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    sql = " insert into " + tableName + " values ( " +
          "'1', '1', '1', '" + valueStr + "', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    sql = " insert into " + tableName + " values ( " +
          "'1', '1', '1', '1', '" + valueStr + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // data truncated
    valueStr = "1";
    string valueStr2 = "1xxx";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );

    // out of range
    // bool
    valueStr = "1";
    valueStr2 = std::to_string( INT8_MAX + 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr2 = std::to_string( INT8_MIN - 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int8
    valueStr2 = std::to_string( INT8_MAX + 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr2 = std::to_string( INT8_MIN - 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int16
    valueStr2 = std::to_string( INT16_MAX + 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr2 = std::to_string( INT16_MIN - 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int32
    valueStr2 = std::to_string( (int64_t)INT32_MAX + 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr2 = std::to_string( (int64_t)INT32_MIN - 1 );
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "', " +
          "'" + valueStr + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // int64
    valueStr2 = "9223372036854775808";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr2 = "-9223372036854775809";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr + "', " +
          "'" + valueStr2 + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // convert string to all decimal

    // incorrect value
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 float not null, f2 double not null, f3 decimal(12, 6) not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    valueStr = "xxx";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD  );
    sql = " insert into " + tableName + " values ( " +
          "'1', '" + valueStr + "', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD  );
    sql = " insert into " + tableName + " values ( " +
          "'1', '1', '" + valueStr + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD  );

    // data truncated
    valueStr = "1.1xxx";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );
    sql = " insert into " + tableName + " values ( " +
          "'1', '" + valueStr + "', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), WARN_DATA_TRUNCATED );
    sql = " insert into " + tableName + " values ( " +
          "'1', '1', '" + valueStr + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // out of range
    // float signed min: -3.40282e38, max: 3.40282e38
    valueStr = "3.40282e39";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr = "-3.40282e39";
    sql = " insert into " + tableName + " values ( " +
          "'" + valueStr + "', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // double
    // double max: 1.7976931348623157e308, min?: -1.7976931348623157e308
    valueStr = "1.7976931348623157e309";
    sql = " insert into " + tableName + " values ( " +
          "'1', '" + valueStr + "', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    valueStr = "-1.7986931348623157e308";
    sql = " insert into " + tableName + " values ( " +
          "'1', '" + valueStr + "', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // decimal
    valueStr = "123456789012.1";
    sql = " insert into " + tableName + " values ( " +
          "'1', '1', '" + valueStr + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // char
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 char(3) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    valueStr = "abcd";
    sql = " insert into " + tableName + " values ( '" + valueStr + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    valueStr = string( ARIES_MAX_CHAR_WIDTH + 1, 'a' );
    sql = " insert into " + tableName + " values ( '" + valueStr + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_FIELDLENGTH );

    // temporal
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 date not null, f2 time not null, f3 datetime not null, f4 timestamp not null, f5 year not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // wrong date
    string dateStr( "22020-12-08" );
    string timeStr( "1558" );
    string datetimeStr( "2020-12-08 15:58:00" );
    string timestampStr( "2020-12-08 15:58:00" );
    string yearStr( "2020" );
    sql = "insert into " + tableName + " values ( " +
          "'" + dateStr + "', " +
          "'" + timeStr + "', " +
          "'" + datetimeStr + "', " +
          "'" + timestampStr + "', " +
          "'" + yearStr + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // wrong time
    dateStr = "2020-12-08";
    timeStr = "15580000";
    datetimeStr = "2020-12-08 15:58:00";
    timestampStr = "2020-12-08 15:58:00";
    yearStr = "2020";
    sql = "insert into " + tableName + " values ( " +
          "'" + dateStr + "', " +
          "'" + timeStr + "', " +
          "'" + datetimeStr + "', " +
          "'" + timestampStr + "', " +
          "'" + yearStr + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // wrong dattime
    dateStr = "2020-12-08";
    timeStr = "1558";
    datetimeStr = "22020-12-08 15:58:00";
    timestampStr = "2020-12-08 15:58:00";
    yearStr = "2020";
    sql = "insert into " + tableName + " values ( " +
          "'" + dateStr + "', " +
          "'" + timeStr + "', " +
          "'" + datetimeStr + "', " +
          "'" + timestampStr + "', " +
          "'" + yearStr + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // wrong timestamp
    dateStr = "2020-12-08";
    timeStr = "1558";
    datetimeStr = "2020-12-08 15:58:00";
    timestampStr = "22020-12-08 15:58:00";
    yearStr = "2020";
    sql = "insert into " + tableName + " values ( " +
          "'" + dateStr + "', " +
          "'" + timeStr + "', " +
          "'" + datetimeStr + "', " +
          "'" + timestampStr + "', " +
          "'" + yearStr + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // wrong year
    dateStr = "2020-12-08";
    timeStr = "1558";
    datetimeStr = "2020-12-08 15:58:00";
    timestampStr = "2020-12-08 15:58:00";
    yearStr = "20200";
    sql = "insert into " + tableName + " values ( " +
          "'" + dateStr + "', " +
          "'" + timeStr + "', " +
          "'" + datetimeStr + "', " +
          "'" + timestampStr + "', " +
          "'" + yearStr + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
}

TEST_F(UT_datatype_convert, convert_char)
{
    string tableName = "t_convert_char";
    InitTable( TEST_DB_NAME, tableName );

    // convert string to all int type
    string sql = "create table " + tableName + "(f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 bigint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( '1', '1', '1', '1', '1' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = " insert into " + tableName + " values ( " +
          "'" + std::to_string( INT8_MAX ) + "', " +
          "'" + std::to_string( INT8_MAX ) + "', " +
          "'" + std::to_string( INT16_MAX ) + "', " +
          "'" + std::to_string( INT32_MAX ) + "', " +
          "'" + std::to_string( INT64_MAX ) + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = " insert into " + tableName + " values ( " +
          "'" + std::to_string( INT8_MIN ) + "', " +
          "'" + std::to_string( INT8_MIN ) + "', " +
          "'" + std::to_string( INT16_MIN ) + "', " +
          "'" + std::to_string( INT32_MIN ) + "', " +
          "'" + std::to_string( INT64_MIN ) + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    ASSERT_EQ( column->GetInt8( 1 ), INT8_MAX );
    ASSERT_EQ( column->GetInt8( 2 ), INT8_MIN );

    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    ASSERT_EQ( column->GetInt8( 1 ), INT8_MAX );
    ASSERT_EQ( column->GetInt8( 2 ), INT8_MIN );

    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt16( 0 ), 1 );
    ASSERT_EQ( column->GetInt16( 1 ), INT16_MAX );
    ASSERT_EQ( column->GetInt16( 2 ), INT16_MIN );

    column = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), INT32_MAX );
    ASSERT_EQ( column->GetInt32( 2 ), INT32_MIN );

    column = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
    ASSERT_EQ( column->GetInt64( 1 ), INT64_MAX );
    ASSERT_EQ( column->GetInt64( 2 ), INT64_MIN );

    InitTable( TEST_DB_NAME, tableName );

    // convert string to all decimal
    sql = "create table " + tableName + "(f1 float not null, f2 double not null, f3 decimal(12, 6) not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    double d = 1.123456;
    string strValue = std::to_string( d );
    sql = " insert into " + tableName + " values ( " +
          "'" + strValue + "', " +
          "'" + strValue + "', " +
          "'" + strValue + "'" +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_TRUE( column->GetFloat( 0 ) - d < 0.000001 );

    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDouble( 0 ), d );

    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), strValue );

    // char
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 char(3) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    strValue = "abc";
    sql = " insert into " + tableName + " values ( '" + strValue + "' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), strValue );

    // temporal
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 date not null, f2 time not null, f3 datetime not null, f4 timestamp not null, f5 year not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string dateStr( "2020-12-08" );
    string timeStr( "1558" );
    string datetimeStr( "2020-12-08 15:58:00" );
    string timestampStr( "2020-12-08 15:58:00" );
    string yearStr( "2020" );
    sql = "insert into " + tableName + " values ( " +
          "'" + dateStr + "', " +
          "'" + timeStr + "', " +
          "'" + datetimeStr + "', " +
          "'" + timestampStr + "', " +
          "'" + yearStr + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDateAsString( 0 ), dateStr );
    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetTimeAsString( 0 ), "0:15:58" );
    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), datetimeStr );
    column = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetTimestampAsString( 0 ), timestampStr );
    column = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetYearAsString( 0 ), yearStr );
}

TEST_F(UT_datatype_convert, convert_decimal)
{
    string tableName = "t_convert_decimal";
    InitTable( TEST_DB_NAME, tableName );

    // convert float to decimal
    string sql = "create table " + tableName + "(f1 float not null, f2 double not null, f3 decimal(12, 6) not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    double d = 1.23456;
    string strValue = std::to_string( d );
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_TRUE( column->GetFloat( 0 ) - d < 0.000001 );

    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDouble( 0 ), d );

    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), strValue );

    // scientific float
    strValue = "1.123e1";
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_FALSE( result->IsSuccess() );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    // ASSERT_TRUE( column->GetFloat( 0 ) - d < 0.000001 );
    std::cout << "float column\n";
    column->Dump();

    column = resTable->GetColumnBuffer( 2 );
    // ASSERT_EQ( column->GetDouble( 0 ), d );
    std::cout << "double column\n";
    column->Dump();

    column = resTable->GetColumnBuffer( 3 );
    // ASSERT_EQ( column->GetDecimalAsString( 0 ), strValue );
    std::cout << "decimal column\n";
    column->Dump();

    // max float
    strValue = "3.40282e38";
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_FALSE( result->IsSuccess() );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    // ASSERT_TRUE( column->GetFloat( 0 ) - d < 0.000001 );
    std::cout << "float column\n";
    column->Dump();

    column = resTable->GetColumnBuffer( 2 );
    // ASSERT_EQ( column->GetDouble( 0 ), d );
    std::cout << "double column\n";
    column->Dump();

    column = resTable->GetColumnBuffer( 3 );
    // ASSERT_EQ( column->GetDecimalAsString( 0 ), strValue );
    std::cout << "decimal column\n";
    column->Dump();

    // convert decimal to all int type
    InitTable( TEST_DB_NAME, tableName );

    sql = "create table " + tableName + "(f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 bigint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    d = 1.23456;
    strValue = std::to_string( d );
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), ( int8_t)d );
    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt8( 0 ), ( int8_t)d );
    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt16( 0 ), ( int16_t)d );
    column = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 0 ), ( int32_t)d );
    column = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetInt64( 0 ), ( int64_t)d );
}

TEST_F(UT_datatype_convert, convert_true_false)
{
    string tableName = "t_convert_true_false";
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "(f1 bool not null, f2 tinyint not null, f3 smallint not null, f4 int not null, f5 bigint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string strValue = "true";
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt16( 0 ), 1 );
    column = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    column = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );

    strValue = "false";
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 1 ), 0 );
    column = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt8( 1 ), 0 );
    column = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt16( 1 ), 0 );
    column = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 1 ), 0 );
    column = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetInt64( 1 ), 0 );

    // decimal
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 float not null, f2 double not null, f3 decimal(12, 6) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    strValue = "true";
    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // temporal
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 date not null, f2 time not null, f3 datetime not null, f4 timestamp not null, f5 year not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = " insert into " + tableName + " values ( " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue + ", " +
          strValue +
          " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}


TEST_F(UT_datatype_convert, convert_column_int16_to_smaller)
{
    string srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    string sql = "create table " + srcTableName + "(f1 smallint not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // bool
    string dstTableName = "t_dst_bool";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 bool not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int8
    dstTableName = "t_dst_int8";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 tinyint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int16
    dstTableName = "t_dst_int16";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 smallint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt16( 0 ), 1 );

    // int32
    dstTableName = "t_dst_int32";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );

    // int64
    dstTableName = "t_dst_int64";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 bigint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
}

TEST_F(UT_datatype_convert, convert_column_int32_to_smaller)
{
    string srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    string sql = "create table " + srcTableName + "(f1 int not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // bool
    string dstTableName = "t_dst_bool";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 bool not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int8
    dstTableName = "t_dst_int8";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 tinyint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int16
    dstTableName = "t_dst_int16";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 smallint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int32
    dstTableName = "t_dst_int32";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );

    // int64
    dstTableName = "t_dst_int64";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 bigint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
}

TEST_F(UT_datatype_convert, convert_column_int64_to_smaller)
{
    string srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    string sql = "create table " + srcTableName + "(f1 bigint not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // bool
    string dstTableName = "t_dst_bool";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 bool not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int8
    dstTableName = "t_dst_int8";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 tinyint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int16
    dstTableName = "t_dst_int16";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 smallint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int32
    dstTableName = "t_dst_int32";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 int not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int64
    dstTableName = "t_dst_int64";
    InitTable( TEST_DB_NAME, dstTableName );

    sql = "create table " + dstTableName + "(f1 bigint not null);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
}

TEST_F(UT_datatype_convert, convert_column_to_int8 )
{
    string dstTableName = "t_dst_int8";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 tinyint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );

    // from int16
    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int32
    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int64
    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from float
    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

TEST_F(UT_datatype_convert, convert_column_to_int16 )
{
    string dstTableName = "t_dst_int16";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 smallint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt16( 0 ), 1 );

    // from int16
    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt16( 1 ), 1 );

    // from int32
    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int64
    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from float
    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

TEST_F(UT_datatype_convert, convert_column_to_int32 )
{
    string dstTableName = "t_dst_int32";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 int not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );

    // from int16
    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // from int32
    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 2 ), 1 );

    // from int64
    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from float
    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

TEST_F(UT_datatype_convert, convert_column_to_int64 )
{
    string dstTableName = "t_dst_int64";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 bigint not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );

    // from int16
    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 1 ), 1 );

    // from int32
    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 2 ), 1 );

    // from int64
    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 3 ), 1 );

    // from float
    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

TEST_F(UT_datatype_convert, convert_column_to_float )
{
    string dstTableName = "t_dst_float";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 float not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );

    // from int16
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );

    // from int32
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );

    // from int64
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );

    // from float
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );

    // from double
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    /*
    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );
    */

    // from decimal
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

}

TEST_F(UT_datatype_convert, convert_column_to_double )
{
    string dstTableName = "t_dst_double";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 double not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from int16
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from int32
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from int64
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from float
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from double
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from decimal
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

}

TEST_F(UT_datatype_convert, convert_column_to_decimal )
{
    string dstTableName = "t_dst_decimal";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 decimal(12, 2) not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), 1 );

    // from int16
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), 1 );

    // from int32
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), 1 );

    // from int64
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), 1 );

    // from float
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal(12, 2) not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), 1 );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

TEST_F(UT_datatype_convert, convert_column_to_date )
{
    string dstTableName = "t_dst_date";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 date not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int16
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int32
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int64
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from float
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2020-12-12" );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2020-12-12" );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2020-12-12" );

}

TEST_F(UT_datatype_convert, convert_column_to_datetime )
{
    string dstTableName = "t_dst_datetime";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 datetime not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int16
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int32
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int64
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from float
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), "2020-12-12 00:00:00" );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), "2020-12-12 23:00:00" );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), "2020-12-12 23:00:00" );

}

TEST_F( UT_datatype_convert, convert_column_to_timestamp )
{
    string dstTableName = "t_dst_timestamp";
    InitTable( TEST_DB_NAME, dstTableName );

    string sql = "create table " + dstTableName + "(f1 timestamp not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    // from int8
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string srcTableName = "t_src_int8";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 tinyint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int16
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int16";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 smallint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int32
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int32";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 int not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from int64
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_int64";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 bigint not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from float
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_float";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 float not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from double
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_double";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 double not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from decimal
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_decimal";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 decimal not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( 1 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from date
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_date";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 date not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetTimestampAsString( 0 ), "2020-12-12 00:00:00" );

    // from time
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_time";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 time not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2338' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // from datetime
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    string tsStr = "1970-01-01 00:00:01";
    srcTableName = "t_src_datetime";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 datetime not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '" + tsStr + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    column = resTable->GetColumnBuffer( 1 );

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    auto tmpTs = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTimestamp( tsStr );
    // ASSERT_EQ( column->GetTimestampAsString( 0 ), tsStr );
    auto resultItem = *( column->GetTimestamp( 0 ) );
    ASSERT_EQ( resultItem, tmpTs );

    // from timestamp
    sql = "delete from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    srcTableName = "t_src_timestamp";
    InitTable( TEST_DB_NAME, srcTableName );
    sql = "create table " + srcTableName + "(f1 timestamp not null );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    sql = "insert into " + srcTableName + " values ( '2020-12-12 23:00:00' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + dstTableName + " select * from " + srcTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + dstTableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetTimestampAsString( 0 ), "2020-12-12 23:00:00" );
}