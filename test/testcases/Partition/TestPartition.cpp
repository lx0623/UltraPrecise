
#include <unistd.h>
#include <gtest/gtest.h>
#include <string>
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/mysqld.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "schema/SchemaManager.h"
#include "server/Configuration.h"
#include "AriesEngine/transaction/AriesInitialTable.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"

#include "utils/string_util.h"

#include "../../TestUtils.h"

using namespace std;
using namespace aries_engine;
using namespace aries_utils;
using namespace aries_test;

static const string TEST_DB_NAME( "ut_test_partition" );
static string cwd = get_current_work_directory();
class UT_partition: public testing::Test
{
private:
protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
        cout << "tear down UT_partition\n";
        string sql = "drop database if exists " + TEST_DB_NAME;
        ExecuteSQL( sql, "" );
    }
};

TEST_F(UT_partition, dict )
{
    string tableName( "t1" );
    string dictName( tableName + "_f2" );
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + " ( f1 date, f2 char(40) encoding bytedict as " + dictName + " )";
    sql += " partition by range (f1) ";
    sql += "( ";
    sql += "partition p0 values less than ('2000-01-01'),";
    sql += "partition p1 values less than ('2010-01-01'),";
    sql += "partition p2 values less than MAXVALUE";
    sql += ")";

    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    string csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_dict.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + " where f1 < '2000-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isDateDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), "1996-01-13" );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableString( 0 ), "1996" );

    sql = "select * from " + tableName + " where f1 >= '2000-01-01' order by f1";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isDateDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), "2000-02-10" );
    ASSERT_EQ( columnBuff->isDateDataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 1 ), "2021-06-29" );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableString( 0 ), "2000" );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableString( 1 ), "2021" );
}