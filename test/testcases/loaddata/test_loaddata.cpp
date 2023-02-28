//
// Created by tengjp on 19-12-31.
//
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

#include "CudaAcc/AriesSqlOperator_sort.h"

#include "../../TestUtils.h"

using namespace std;
using namespace aries_engine;
using namespace aries_test;

extern bool STRICT_MODE;

const string dbName = "test_loaddata";
string CWD()
{
    char* ret = nullptr;
    if ( ( ret = getcwd( NULL, 0) ) )
    {
        string retStr( ret );
        free( ret );
        return retStr;
    }
    return string("");
}
string cwd = CWD();

AriesTableBlockUPtr executeSql( const string& sql, const string& dbName )
{
    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    auto art = make_unique< AriesTableBlock >();
    if (results->IsSuccess() && results->GetResults().size() > 0) {
        auto amtp = results->GetResults()[ 0 ];
        auto block = std::move( ( ( ( AriesMemTable * )( amtp.get() ) )->GetContent() ) );
        int count = block->GetRowCount();
        //art->Dump();
        cout << "tupleNum is:" << count << endl;
        return block;
    }
    return art;
}

void LoadData( const string& testcaseName, int expectErrCode = 0, const string& expectErrMsg = "" )
{
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );

    ifstream ifs( csvPath );
    char schemaSql[1024];
    ifs.getline( schemaSql, 1024 );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(schemaSql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    if ( expectErrCode )
    {
        ASSERT_EQ( result->GetErrorCode(), expectErrCode );
        ASSERT_EQ( result->GetErrorMessage(), expectErrMsg );
    }
    else
        ASSERT_TRUE(result->IsSuccess());
}

void VerifyResultCount( const string& tableName, int expectColCnt, int64 expectRowCnt, AriesTableBlockUPtr& table )
{
    string sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, expectColCnt );
    auto tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, expectRowCnt );
}

class UT_loaddata: public testing::Test
{
private:
protected:
    static void SetUpTestCase()
    {
        string sql = "set @@primary_key_checks=1";
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
        ASSERT_TRUE(result->IsSuccess());
    }
    static void TearDownTestCase()
    {
        cout << "tear down UT_loaddata\n";
        string sql = "drop database if exists " + dbName;
        ExecuteSQL( sql, "" );

        sql = "set @@primary_key_checks=0";
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
        ASSERT_TRUE(result->IsSuccess());
    }
};

TEST_F(UT_loaddata, incr_with_primary_key)
{
    string testcaseName = "incr_non_empty_init_table_and_empty_delta_table";
    string tableName = "t_" + testcaseName;
    string dictName = tableName + "_f5";
    InitTable( dbName, tableName );

    string sql = "create table " + tableName + " ( f1 int primary key, f2 char(4), f3 decimal(12,2 ) not null, f4 date, f5 char(4) encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "1.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 5, 7, table );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( dbName, tableName );
    ASSERT_EQ( initTable->GetBlockCount(), 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );
    ASSERT_EQ( column->GetInt32( 5 ), 6 );
    ASSERT_EQ( column->GetInt32( 6 ), 7 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "1" );
    ASSERT_EQ( column->GetNullableString( 1 ), "2" );
    ASSERT_EQ( column->GetNullableString( 2 ), "3" );
    ASSERT_EQ( column->GetNullableString( 3 ), "4" );
    ASSERT_EQ( column->GetNullableString( 4 ), "5" );
    ASSERT_EQ( column->GetNullableString( 5 ), "6" );
    ASSERT_EQ( column->GetNullableString( 6 ), "7" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "1.11" ) );
    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "3.33" ) );
    ASSERT_EQ( column->GetDecimal( 4 ), aries_acc::Decimal( "5.55" ) );
    ASSERT_EQ( column->GetDecimal( 6 ), aries_acc::Decimal( "7.77" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2021-05-23" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "2021-05-26" );
    ASSERT_EQ( column->GetDateAsString( 5 ), "2021-05-28" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbbb" );
    ASSERT_EQ( column->GetNullableString( 2 ), "cccc" );
    ASSERT_EQ( column->GetNullableString( 3 ), "dddd" );
    ASSERT_EQ( column->GetNullableString( 4 ), "eeee" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ffff" );
    ASSERT_EQ( column->GetNullableString( 6 ), "gggg" );

    // load duplicated
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_DUP_ENTRY );

    // load another
    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "2.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 9, table );

    initTable = AriesInitialTableManager::GetInstance().getTable( dbName, tableName );
#ifdef ARIES_INIT_TABLE_TEST
    ASSERT_EQ( initTable->GetBlockCount(), 2 );
#else
    ASSERT_EQ( initTable->GetBlockCount(), 1 );
#endif

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );
    ASSERT_EQ( column->GetInt32( 5 ), 6 );
    ASSERT_EQ( column->GetInt32( 6 ), 7 );
    ASSERT_EQ( column->GetInt32( 7 ), 8 );
    ASSERT_EQ( column->GetInt32( 8 ), 9 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "1" );
    ASSERT_EQ( column->GetNullableString( 1 ), "2" );
    ASSERT_EQ( column->GetNullableString( 2 ), "3" );
    ASSERT_EQ( column->GetNullableString( 3 ), "4" );
    ASSERT_EQ( column->GetNullableString( 4 ), "5" );
    ASSERT_EQ( column->GetNullableString( 5 ), "6" );
    ASSERT_EQ( column->GetNullableString( 6 ), "7" );
    ASSERT_EQ( column->GetNullableString( 7 ), "8" );
    ASSERT_EQ( column->GetNullableString( 8 ), "9" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "1.11" ) );
    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "3.33" ) );
    ASSERT_EQ( column->GetDecimal( 4 ), aries_acc::Decimal( "5.55" ) );
    ASSERT_EQ( column->GetDecimal( 6 ), aries_acc::Decimal( "7.77" ) );
    ASSERT_EQ( column->GetDecimal( 7 ), aries_acc::Decimal( "8.88" ) );
    ASSERT_EQ( column->GetDecimal( 8 ), aries_acc::Decimal( "9.99" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2021-05-23" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "2021-05-26" );
    ASSERT_EQ( column->GetDateAsString( 5 ), "2021-05-28" );
    ASSERT_EQ( column->GetDateAsString( 7 ), "2021-05-30" );
    ASSERT_EQ( column->GetDateAsString( 8 ), "2021-05-31" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbbb" );
    ASSERT_EQ( column->GetNullableString( 2 ), "cccc" );
    ASSERT_EQ( column->GetNullableString( 3 ), "dddd" );
    ASSERT_EQ( column->GetNullableString( 4 ), "eeee" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ffff" );
    ASSERT_EQ( column->GetNullableString( 6 ), "gggg" );
    ASSERT_EQ( column->GetNullableString( 7 ), "hhhh" );
    ASSERT_EQ( column->GetNullableString( 8 ), "iiii" );

    // insert and then load
    sql = "insert into " + tableName + " values (1, '10', 10.00, '2021-05-23', 'jjjj')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_DUP_ENTRY_WITH_KEY_NAME );

    sql = "insert into " + tableName + " values (10, '10', 10.00, '2021-05-23', 'jjjj')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 10, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 10 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "10" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 9 ), aries_acc::Decimal( "10.00" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 9 ), "2021-05-23" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "jjjj" );

    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "3.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 12, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 11 );
    ASSERT_EQ( column->GetInt32( 10 ), 12 );
    ASSERT_EQ( column->GetInt32( 11 ), 10 ); // delta table

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "11" );
    ASSERT_EQ( column->GetNullableString( 10 ), "12" );
    ASSERT_EQ( column->GetNullableString( 11 ), "10" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 9 ), aries_acc::Decimal( "11.01" ) );
    ASSERT_EQ( column->GetDecimal( 10 ), aries_acc::Decimal( "12.01" ) );
    ASSERT_EQ( column->GetDecimal( 11 ), aries_acc::Decimal( "10.00" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 9 ), "2021-06-01" );
    ASSERT_EQ( column->GetDateAsString( 10 ), "2021-06-02" );
    ASSERT_EQ( column->GetDateAsString( 11 ), "2021-05-23" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "kkkk" );
    ASSERT_EQ( column->GetNullableString( 10 ), "llll" );
    ASSERT_EQ( column->GetNullableString( 11 ), "jjjj" );

    // delete and then load
    sql = "delete from " + tableName + " where f1 = 11";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "4.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 12, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 12 );
    ASSERT_EQ( column->GetInt32( 10 ), 13 );
    ASSERT_EQ( column->GetInt32( 11 ), 10 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 9 ), "12" );
    ASSERT_EQ( column->GetNullableString( 10 ), "13" );
    ASSERT_EQ( column->GetNullableString( 11 ), "10" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 9 ), aries_acc::Decimal( "12.01" ) );
    ASSERT_EQ( column->GetDecimal( 10 ), aries_acc::Decimal( "13.01" ) );
    ASSERT_EQ( column->GetDecimal( 11 ), aries_acc::Decimal( "10.00" ) );

    // update and then load
    sql = "update " + tableName + " set f5 = 'LLLL' where f1 = 12";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "5.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 13, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 13 );
    ASSERT_EQ( column->GetInt32( 10 ), 14 );
    ASSERT_EQ( column->GetInt32( 11 ), 10 );
    ASSERT_EQ( column->GetInt32( 12 ), 12 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "13" );
    ASSERT_EQ( column->GetNullableString( 10 ), "14" );
    ASSERT_EQ( column->GetNullableString( 11 ), "10" );
    ASSERT_EQ( column->GetNullableString( 12 ), "12" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "mmmm" );
    ASSERT_EQ( column->GetNullableString( 10 ), "nnnn" );
    ASSERT_EQ( column->GetNullableString( 11 ), "jjjj" );
    ASSERT_EQ( column->GetNullableString( 12 ), "LLLL" );
}

TEST_F(UT_loaddata, incr_no_key)
{
    string testcaseName = "incr_non_empty_init_table_and_empty_delta_table";
    string tableName = "t_incr_no_key";
    string dictName = tableName + "_f5";
    InitTable( dbName, tableName );

    string sql = "create table " + tableName + " ( f1 int not null, f2 char(4), f3 decimal(12,2 ) not null, f4 date, f5 char(4) encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "1.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 5, 7, table );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( dbName, tableName );
    ASSERT_EQ( initTable->GetBlockCount(), 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );
    ASSERT_EQ( column->GetInt32( 5 ), 6 );
    ASSERT_EQ( column->GetInt32( 6 ), 7 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "1" );
    ASSERT_EQ( column->GetNullableString( 1 ), "2" );
    ASSERT_EQ( column->GetNullableString( 2 ), "3" );
    ASSERT_EQ( column->GetNullableString( 3 ), "4" );
    ASSERT_EQ( column->GetNullableString( 4 ), "5" );
    ASSERT_EQ( column->GetNullableString( 5 ), "6" );
    ASSERT_EQ( column->GetNullableString( 6 ), "7" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "1.11" ) );
    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "3.33" ) );
    ASSERT_EQ( column->GetDecimal( 4 ), aries_acc::Decimal( "5.55" ) );
    ASSERT_EQ( column->GetDecimal( 6 ), aries_acc::Decimal( "7.77" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2021-05-23" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "2021-05-26" );
    ASSERT_EQ( column->GetDateAsString( 5 ), "2021-05-28" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbbb" );
    ASSERT_EQ( column->GetNullableString( 2 ), "cccc" );
    ASSERT_EQ( column->GetNullableString( 3 ), "dddd" );
    ASSERT_EQ( column->GetNullableString( 4 ), "eeee" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ffff" );
    ASSERT_EQ( column->GetNullableString( 6 ), "gggg" );

    // load another
    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "2.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 9, table );

    initTable = AriesInitialTableManager::GetInstance().getTable( dbName, tableName );
#ifdef ARIES_INIT_TABLE_TEST
    ASSERT_EQ( initTable->GetBlockCount(), 2 );
#else
    ASSERT_EQ( initTable->GetBlockCount(), 1 );
#endif

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );
    ASSERT_EQ( column->GetInt32( 5 ), 6 );
    ASSERT_EQ( column->GetInt32( 6 ), 7 );
    ASSERT_EQ( column->GetInt32( 7 ), 8 );
    ASSERT_EQ( column->GetInt32( 8 ), 9 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "1" );
    ASSERT_EQ( column->GetNullableString( 1 ), "2" );
    ASSERT_EQ( column->GetNullableString( 2 ), "3" );
    ASSERT_EQ( column->GetNullableString( 3 ), "4" );
    ASSERT_EQ( column->GetNullableString( 4 ), "5" );
    ASSERT_EQ( column->GetNullableString( 5 ), "6" );
    ASSERT_EQ( column->GetNullableString( 6 ), "7" );
    ASSERT_EQ( column->GetNullableString( 7 ), "8" );
    ASSERT_EQ( column->GetNullableString( 8 ), "9" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "1.11" ) );
    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "3.33" ) );
    ASSERT_EQ( column->GetDecimal( 4 ), aries_acc::Decimal( "5.55" ) );
    ASSERT_EQ( column->GetDecimal( 6 ), aries_acc::Decimal( "7.77" ) );
    ASSERT_EQ( column->GetDecimal( 7 ), aries_acc::Decimal( "8.88" ) );
    ASSERT_EQ( column->GetDecimal( 8 ), aries_acc::Decimal( "9.99" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 0 ), "2021-05-23" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "2021-05-26" );
    ASSERT_EQ( column->GetDateAsString( 5 ), "2021-05-28" );
    ASSERT_EQ( column->GetDateAsString( 7 ), "2021-05-30" );
    ASSERT_EQ( column->GetDateAsString( 8 ), "2021-05-31" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbbb" );
    ASSERT_EQ( column->GetNullableString( 2 ), "cccc" );
    ASSERT_EQ( column->GetNullableString( 3 ), "dddd" );
    ASSERT_EQ( column->GetNullableString( 4 ), "eeee" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ffff" );
    ASSERT_EQ( column->GetNullableString( 6 ), "gggg" );
    ASSERT_EQ( column->GetNullableString( 7 ), "hhhh" );
    ASSERT_EQ( column->GetNullableString( 8 ), "iiii" );

    // insert and then load
    sql = "insert into " + tableName + " values (10, '10', 10.00, '2021-05-23', 'jjjj')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 10, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 10 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "10" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 9 ), aries_acc::Decimal( "10.00" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 9 ), "2021-05-23" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "jjjj" );

    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "3.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 12, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 11 );
    ASSERT_EQ( column->GetInt32( 10 ), 12 );
    ASSERT_EQ( column->GetInt32( 11 ), 10 ); // delta table

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "11" );
    ASSERT_EQ( column->GetNullableString( 10 ), "12" );
    ASSERT_EQ( column->GetNullableString( 11 ), "10" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 9 ), aries_acc::Decimal( "11.01" ) );
    ASSERT_EQ( column->GetDecimal( 10 ), aries_acc::Decimal( "12.01" ) );
    ASSERT_EQ( column->GetDecimal( 11 ), aries_acc::Decimal( "10.00" ) );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isDateDataNull( 0 ) );
    ASSERT_EQ( column->GetDateAsString( 9 ), "2021-06-01" );
    ASSERT_EQ( column->GetDateAsString( 10 ), "2021-06-02" );
    ASSERT_EQ( column->GetDateAsString( 11 ), "2021-05-23" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "kkkk" );
    ASSERT_EQ( column->GetNullableString( 10 ), "llll" );
    ASSERT_EQ( column->GetNullableString( 11 ), "jjjj" );

    // delete and then load
    sql = "delete from " + tableName + " where f1 = 11";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "4.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 12, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 12 );
    ASSERT_EQ( column->GetInt32( 10 ), 13 );
    ASSERT_EQ( column->GetInt32( 11 ), 10 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 9 ), "12" );
    ASSERT_EQ( column->GetNullableString( 10 ), "13" );
    ASSERT_EQ( column->GetNullableString( 11 ), "10" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetDecimal( 9 ), aries_acc::Decimal( "12.01" ) );
    ASSERT_EQ( column->GetDecimal( 10 ), aries_acc::Decimal( "13.01" ) );
    ASSERT_EQ( column->GetDecimal( 11 ), aries_acc::Decimal( "10.00" ) );

    // update and then load
    sql = "update " + tableName + " set f5 = 'LLLL' where f1 = 12";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    csvPath = cwd + "/test_resources/loaddata/csv/" + testcaseName + "5.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    VerifyResultCount( tableName, 5, 13, table );

    column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetInt32( 9 ), 13 );
    ASSERT_EQ( column->GetInt32( 10 ), 14 );
    ASSERT_EQ( column->GetInt32( 11 ), 10 );
    ASSERT_EQ( column->GetInt32( 12 ), 12 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "13" );
    ASSERT_EQ( column->GetNullableString( 10 ), "14" );
    ASSERT_EQ( column->GetNullableString( 11 ), "10" );
    ASSERT_EQ( column->GetNullableString( 12 ), "12" );

    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 9 ), "mmmm" );
    ASSERT_EQ( column->GetNullableString( 10 ), "nnnn" );
    ASSERT_EQ( column->GetNullableString( 11 ), "jjjj" );
    ASSERT_EQ( column->GetNullableString( 12 ), "LLLL" );

}

TEST_F(UT_loaddata, empty_csv) {
    string tableName = "t_empty_csv";
    InitTable(dbName, tableName);

    // load empty file into table with only one column
    string sql = "create table " + tableName + "( f1 char(16) );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/empty.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 0 );

    // load empty file into table with two columns
    InitTable(dbName, tableName);
    sql = "create table " + tableName + "( f1 int, f2 char(16) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    csvPath = cwd + "/test_resources/loaddata/csv/empty.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 0 );

    // load file with empty lines into table with a char column
    InitTable(dbName, tableName);
    sql = "create table " + tableName + "( f1 char(16) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    csvPath = cwd + "/test_resources/loaddata/csv/only_empty_lines.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 3 );

    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "" );
    ASSERT_EQ( column->GetString( 1 ), "" );
    ASSERT_EQ( column->GetString( 2 ), "" );

    // load file with empty lines into table with a int column
    InitTable(dbName, tableName);
    sql = "create table " + tableName + "( f1 int );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    csvPath = cwd + "/test_resources/loaddata/csv/only_empty_lines.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 3 );

    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ), 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ), 0 );

    // load file with empty lines into table with two columns
    InitTable(dbName, tableName);
    sql = "create table " + tableName + "( f1 int, f2 char(16) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    csvPath = cwd + "/test_resources/loaddata/csv/only_empty_lines.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 3 );

    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ), 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ), 0 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "" );
    ASSERT_EQ( column->GetString( 1 ), "" );
    ASSERT_EQ( column->GetString( 2 ), "" );

}

TEST_F(UT_loaddata, char_column) {
    string tableName = "t_char_column";
    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 char(16) );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/char.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );

    InitTable(dbName, tableName);
    sql = "create table " + tableName + "( f1 char(16) );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );

    string newValue = "0123456789012345";
    sql = "insert into " + tableName + " values ( '" + newValue + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 3 );

    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );
    ASSERT_EQ( column->GetString( 2 ), newValue );
}

TEST_F(UT_loaddata, int_column) {
    string tableName = "t_int_column";
    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 int );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/int.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ), 2 );
}
TEST_F(UT_loaddata, dict_column) {
    string tableName = "t_dict_column";
    string dictName = "UT_loaddata_dict_column";
    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 char(16) encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/char.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );
}

TEST_F(UT_loaddata, miss_column) {
    string tableName = "t_miss_column";
    InitTable(dbName, tableName);

    string sql = "create table t_miss_column( f1 tinyint not null default -1,"
                 "f2 smallint not null default -1,"
                 "f3 int not null default -1,"
                 "f4 bigint not null default -1,"
                 "f5 float not null default -1,"
                 "f6 double not null default -1,"
                 "f7 decimal not null default -1,"
                 "f8 char(64) not null default 'empty',"
                 "f9 bool not null default -1 );";

    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/miss_column.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 9 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 11 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    ASSERT_EQ( column->GetInt8( 1 ), 0 );
    ASSERT_EQ( column->GetInt8( 2 ), 2 );
    ASSERT_EQ( column->GetInt8( 3 ), 3 );
    ASSERT_EQ( column->GetInt8( 4 ), 4 );
    ASSERT_EQ( column->GetInt8( 5 ), 5 );
    ASSERT_EQ( column->GetInt8( 6 ), 6 );
    ASSERT_EQ( column->GetInt8( 7 ), 7 );
    ASSERT_EQ( column->GetInt8( 8 ), 8 );
    ASSERT_EQ( column->GetInt8( 9 ), 9 );
    ASSERT_EQ( column->GetInt8( 10 ), 10 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt16( 0 ), 1 );
    ASSERT_EQ( column->GetInt16( 1 ), 0 );
    ASSERT_EQ( column->GetInt16( 2 ), 0 );
    ASSERT_EQ( column->GetInt16( 3 ), 3 );
    ASSERT_EQ( column->GetInt16( 4 ), 4 );
    ASSERT_EQ( column->GetInt16( 5 ), 5 );
    ASSERT_EQ( column->GetInt16( 6 ), 6 );
    ASSERT_EQ( column->GetInt16( 7 ), 7 );
    ASSERT_EQ( column->GetInt16( 8 ), 8 );
    ASSERT_EQ( column->GetInt16( 9 ), 9 );
    ASSERT_EQ( column->GetInt16( 10 ), 10 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 0 );
    ASSERT_EQ( column->GetInt32( 2 ), 0 );
    ASSERT_EQ( column->GetInt32( 3 ), 0 );
    ASSERT_EQ( column->GetInt32( 4 ), 4 );
    ASSERT_EQ( column->GetInt32( 5 ), 5 );
    ASSERT_EQ( column->GetInt32( 6 ), 6 );
    ASSERT_EQ( column->GetInt32( 7 ), 7 );
    ASSERT_EQ( column->GetInt32( 8 ), 8 );
    ASSERT_EQ( column->GetInt32( 9 ), 9 );
    ASSERT_EQ( column->GetInt32( 10 ), 10 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
    ASSERT_EQ( column->GetInt64( 1 ), 0 );
    ASSERT_EQ( column->GetInt64( 2 ), 0 );
    ASSERT_EQ( column->GetInt64( 3 ), 0 );
    ASSERT_EQ( column->GetInt64( 4 ), 0 );
    ASSERT_EQ( column->GetInt64( 5 ), 5 );
    ASSERT_EQ( column->GetInt64( 6 ), 6 );
    ASSERT_EQ( column->GetInt64( 7 ), 7 );
    ASSERT_EQ( column->GetInt64( 8 ), 8 );
    ASSERT_EQ( column->GetInt64( 9 ), 9 );
    ASSERT_EQ( column->GetInt64( 10 ), 10 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetFloat( 0 ), 1 );
    ASSERT_EQ( column->GetFloat( 1 ), 0 );
    ASSERT_EQ( column->GetFloat( 2 ), 0 );
    ASSERT_EQ( column->GetFloat( 3 ), 0 );
    ASSERT_EQ( column->GetFloat( 4 ), 0 );
    ASSERT_EQ( column->GetFloat( 5 ), 0 );
    ASSERT_EQ( column->GetFloat( 6 ), 6 );
    ASSERT_EQ( column->GetFloat( 7 ), 7 );
    ASSERT_EQ( column->GetFloat( 8 ), 8 );
    ASSERT_EQ( column->GetFloat( 9 ), 9 );
    ASSERT_EQ( column->GetFloat( 10 ), 10 );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetDouble( 0 ), 1 );
    ASSERT_EQ( column->GetDouble( 1 ), 0 );
    ASSERT_EQ( column->GetDouble( 2 ), 0 );
    ASSERT_EQ( column->GetDouble( 3 ), 0 );
    ASSERT_EQ( column->GetDouble( 4 ), 0 );
    ASSERT_EQ( column->GetDouble( 5 ), 0 );
    ASSERT_EQ( column->GetDouble( 6 ), 0 );
    ASSERT_EQ( column->GetDouble( 7 ), 7 );
    ASSERT_EQ( column->GetDouble( 8 ), 8 );
    ASSERT_EQ( column->GetDouble( 9 ), 9 );
    ASSERT_EQ( column->GetDouble( 10 ), 10 );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetDecimal( 0 ), 1 );
    ASSERT_EQ( column->GetDecimal( 1 ), 0 );
    ASSERT_EQ( column->GetDecimal( 2 ), 0 );
    ASSERT_EQ( column->GetDecimal( 3 ), 0 );
    ASSERT_EQ( column->GetDecimal( 4 ), 0 );
    ASSERT_EQ( column->GetDecimal( 5 ), 0 );
    ASSERT_EQ( column->GetDecimal( 6 ), 0 );
    ASSERT_EQ( column->GetDecimal( 7 ), 0 );
    ASSERT_EQ( column->GetDecimal( 8 ), 8 );
    ASSERT_EQ( column->GetDecimal( 9 ), 9 );
    ASSERT_EQ( column->GetDecimal( 10 ), 10 );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_EQ( column->GetString( 0 ), "1" );
    ASSERT_EQ( column->GetString( 1 ), "" );
    ASSERT_EQ( column->GetString( 2 ), "" );
    ASSERT_EQ( column->GetString( 3 ), "" );
    ASSERT_EQ( column->GetString( 4 ), "" );
    ASSERT_EQ( column->GetString( 5 ), "" );
    ASSERT_EQ( column->GetString( 6 ), "" );
    ASSERT_EQ( column->GetString( 7 ), "" );
    ASSERT_EQ( column->GetString( 8 ), "" );
    ASSERT_EQ( column->GetString( 9 ), "9" );
    ASSERT_EQ( column->GetString( 10 ), "10" );

    // verify column 9
    column = table->GetColumnBuffer( 9 );
    ASSERT_EQ( column->GetInt8( 0 ), 1 );
    ASSERT_EQ( column->GetInt8( 1 ), 0 );
    ASSERT_EQ( column->GetInt8( 2 ), 0 );
    ASSERT_EQ( column->GetInt8( 3 ), 0 );
    ASSERT_EQ( column->GetInt8( 4 ), 0 );
    ASSERT_EQ( column->GetInt8( 5 ), 0 );
    ASSERT_EQ( column->GetInt8( 6 ), 0 );
    ASSERT_EQ( column->GetInt8( 7 ), 0 );
    ASSERT_EQ( column->GetInt8( 8 ), 0 );
    ASSERT_EQ( column->GetInt8( 9 ), 0 );
    ASSERT_EQ( column->GetInt8( 10 ), 10 );
    // ASSERT_TRUE( !result->IsSuccess() );
    // ASSERT_EQ( result->GetErrorCode(), ER_WARN_TOO_FEW_RECORDS );
    // ASSERT_EQ( result->GetErrorMessage(), "Row 4 doesn't contain data for all columns");
}

TEST_F(UT_loaddata, miss_column2) {
    string tableName = "t_miss_column2";
    string dictName1 = "UT_loaddata_miss_column2_1";
    string dictName2 = "UT_loaddata_miss_column2_2";
    string dictName3 = "UT_loaddata_miss_column2_3";
    InitTable(dbName, tableName);

    string sql( "create table t_miss_column2  ( p_partkey     integer not null primary key,"
                "p_name        varchar(128) not null,"
                "p_mfgr        char(25) not null,"
                "p_brand       char(10) not null encoding bytedict as " + dictName1 + " ,"
                "p_type        varchar(25) not null encoding shortdict as " + dictName2 + " ,"
                "p_size        integer not null,"
                "p_container   char(10) not null encoding bytedict as " + dictName3 + " ,"
                "p_retailprice decimal(12,2) not null,"
                "p_comment     varchar(23) not null ); "
               );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    string csvPath = cwd + "/test_resources/loaddata/csv/miss_column2.csv";
    // use wrong fields terminator
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 9 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

}

TEST_F(UT_loaddata, not_null_null_values)
{
    string tableName = "t_not_null_null_values";
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append(tableName);
    sql.append("( f1 tinyint not null, f2 smallint not null,"
               "f3 int not null, f4 bigint not null, f5 float not null, f6 double not null,"
               "f7 decimal not null, f8 char(64) not null, f9 bool not null );");
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/not_null_null_values.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 9 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt8( 0 ), 0 );
    ASSERT_EQ( column->GetInt8( 1 ), 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt16( 0 ), 0 );
    ASSERT_EQ( column->GetInt16( 1 ), 1 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt64( 0 ), 0 );
    ASSERT_EQ( column->GetInt64( 1 ), 1 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetFloat( 0 ), 0 );
    ASSERT_EQ( column->GetFloat( 1 ), 1 );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetDouble( 0 ), 0 );
    ASSERT_EQ( column->GetDouble( 1 ), 1 );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetDecimal( 0 ), 0 );
    ASSERT_EQ( column->GetDecimal( 1 ), 1 );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_EQ( column->GetString( 0 ), "" );
    ASSERT_EQ( column->GetString( 1 ), "1" );

    // verify column 9
    column = table->GetColumnBuffer( 9 );
    ASSERT_EQ( column->GetInt8( 0 ), 0 );
    ASSERT_EQ( column->GetInt8( 1 ), 1 );
}
/*
test empty values
in non-strict mode,
mysql> show warnings;
+---------+------+------------------------------------------------------+
| Level   | Code | Message                                              |
+---------+------+------------------------------------------------------+
| Warning | 1366 | Incorrect integer value: '' for column 'f1' at row 1 |
| Warning | 1366 | Incorrect integer value: '' for column 'f2' at row 1 |
| Warning | 1366 | Incorrect integer value: '' for column 'f3' at row 1 |
| Warning | 1366 | Incorrect integer value: '' for column 'f4' at row 1 |
| Warning | 1265 | Data truncated for column 'f5' at row 1              |
| Warning | 1265 | Data truncated for column 'f6' at row 1              |
| Warning | 1366 | Incorrect decimal value: '' for column 'f7' at row 1 |
| Warning | 1366 | Incorrect integer value: '' for column 'f9' at row 1 |
+---------+------+------------------------------------------------------+
8 rows in set (0.01 sec)
*/
TEST_F(UT_loaddata, nullable_empty_values)
{
    string testcaseName = "nullable_empty_values";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 9, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( !column->isInt8DataNull( 0 ) );
    ASSERT_EQ( column->GetNullableInt8( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt8( 1 ).value, 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( !column->isInt16DataNull( 0 ) );
    ASSERT_EQ( column->GetNullableInt16( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt16( 1 ).value, 1 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_TRUE( !column->isInt32DataNull( 0 ) );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 1 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( !column->isInt64DataNull( 0 ) );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt64( 1 ).value, 1 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_TRUE( !column->isFloatDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableFloat( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableFloat( 1 ).value, 1 );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_TRUE( !column->isDoubleDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableDouble( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableDouble( 1 ).value, 1 );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_TRUE( !column->isDecimalDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableDecimal( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableDecimal( 1 ).value, 1 );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_TRUE( !column->isStringDataNull( 0 ) );
    ASSERT_EQ( column->GetNullableString( 0 ), "" );
    ASSERT_EQ( column->GetNullableString( 1 ), "1" );

    // verify column 9
    column = table->GetColumnBuffer( 9 );
    ASSERT_TRUE( !column->isInt8DataNull( 0 ) );
    ASSERT_EQ( column->GetNullableInt8( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt8( 1 ).value, 1 );
}
/*
test invalid values for number columns
mysql, in non-strict mode:
mysql> show warnings;
+---------+------+-------------------------------------------------------+
| Level   | Code | Message                                               |
+---------+------+-------------------------------------------------------+
| Warning | 1366 | Incorrect integer value: 'a' for column 'f1' at row 1 |
| Warning | 1366 | Incorrect integer value: 'a' for column 'f2' at row 1 |
| Warning | 1366 | Incorrect integer value: 'a' for column 'f3' at row 1 |
| Warning | 1366 | Incorrect integer value: 'a' for column 'f4' at row 1 |
| Warning | 1265 | Data truncated for column 'f5' at row 1               |
| Warning | 1265 | Data truncated for column 'f6' at row 1               |
| Warning | 1366 | Incorrect decimal value: 'a' for column 'f7' at row 1 |
| Warning | 1366 | Incorrect integer value: 'a' for column 'f8' at row 1 |
| Warning | 1264 | Out of range value for column 'f1' at row 3           |
+---------+------+-------------------------------------------------------+
9 rows in set (0.00 sec)

mysql> select * from t_number_invalid_values;
+------+------+------+------+------+------+------+------+
| f1   | f2   | f3   | f4   | f5   | f6   | f7   | f8   |
+------+------+------+------+------+------+------+------+
|    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
|    1 |    1 |    1 |    1 |    1 |    1 |    1 |    1 |
|  127 |    1 |    1 |    1 |    1 |    1 |    1 |    1 |
+------+------+------+------+------+------+------+------+
3 rows in set (0.01 sec)

strict mode:
mysql> show warnings;
+-------+------+-------------------------------------------------------+
| Level | Code | Message                                               |
+-------+------+-------------------------------------------------------+
| Error | 1366 | Incorrect integer value: 'a' for column 'f1' at row 1 |
| Error | 1366 | Incorrect integer value: 'a' for column 'f2' at row 1 |
| Error | 1366 | Incorrect integer value: 'a' for column 'f3' at row 1 |
| Error | 1366 | Incorrect integer value: 'a' for column 'f4' at row 1 |
| Error | 1265 | Data truncated for column 'f5' at row 1               |
| Error | 1265 | Data truncated for column 'f6' at row 1               |
| Error | 1366 | Incorrect decimal value: 'a' for column 'f7' at row 1 |
| Error | 1366 | Incorrect integer value: 'a' for column 'f8' at row 1 |
+-------+------+-------------------------------------------------------+
8 rows in set (0.00 sec)

*/
TEST_F(UT_loaddata, int_invalid_values1)
{
    STRICT_MODE = false;
    string testcaseName = "int_invalid_values1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect integer value: 'a' for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, int_invalid_values2)
{
    STRICT_MODE = false;
    string testcaseName = "int_invalid_values2";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, WARN_DATA_TRUNCATED, "Data truncated for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, int_signed_overflow)
{
    STRICT_MODE = false;
    string testcaseName = "int_signed_overflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 2147483647 );
    ASSERT_EQ( column->GetInt32( 1 ), 2147483647 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, int_signed_underflow)
{
    STRICT_MODE = false;
    string testcaseName = "int_signed_underflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), -2147483648 );
    ASSERT_EQ( column->GetInt32( 1 ), -2147483648 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, bigint_signed_overflow)
{
    STRICT_MODE = false;
    string testcaseName = "bigint_signed_overflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 9223372036854775807 );
    ASSERT_EQ( column->GetInt64( 1 ), 9223372036854775807 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, bigint_signed_underflow)
{
    STRICT_MODE = false;
    string testcaseName = "bigint_signed_underflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), INT64_MIN );
    ASSERT_EQ( column->GetInt64( 1 ), INT64_MIN );
    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, bigint_invalid_values1)
{
    STRICT_MODE = false;
    string testcaseName = "bigint_invalid_values1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 0 );
    ASSERT_EQ( column->GetInt64( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect integer value: 'a' for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, bigint_invalid_values2)
{
    STRICT_MODE = false;
    string testcaseName = "bigint_invalid_values2";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 1 );
    ASSERT_EQ( column->GetInt64( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, WARN_DATA_TRUNCATED, "Data truncated for column 'f1' at row 3" );
}
/*
TEST_F(UT_loaddata, int_unsigned_overflow)
{
    STRICT_MODE = false;
    string testcaseName = "int_unsigned_overflow";
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 int unsigned not null );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 4294967295 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 int not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    ASSERT_EQ( result->GetErrorMessage(), "Out of range value for column 'f1' at row 3");
}
TEST_F(UT_loaddata, int_unsigned_underflow)
{
    STRICT_MODE = false;
    string testcaseName = "int_unsigned_underflow1";
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 int unsigned not null );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 int not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    ASSERT_EQ( result->GetErrorMessage(), "Out of range value for column 'f1' at row 3");

    // case 2
    STRICT_MODE = false;
    testcaseName = "int_unsigned_underflow2";
    tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 int not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 int not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );
    ASSERT_EQ( result->GetErrorMessage(), "Out of range value for column 'f1' at row 3");
}
*/

TEST_F(UT_loaddata, float_signed_overflow)
{
    STRICT_MODE = false;
    string testcaseName = "float_signed_overflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), HUGE_VAL );
    ASSERT_TRUE( column->GetFloat( 1 ) - 3.4e38 < 0.000001 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, float_signed_underflow)
{
    STRICT_MODE = false;
    string testcaseName = "float_signed_underflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetFloat( 0 ), -HUGE_VAL );
    ASSERT_TRUE( column->GetFloat( 1 ) - 3.4e38 < 0.000001 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, double_signed_overflow)
{
    STRICT_MODE = false;
    string testcaseName = "double_signed_overflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), HUGE_VALL );
    ASSERT_TRUE( abs( column->GetDouble( 1 ) - 1.79e308 ) < 0.000001 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, double_signed_underflow)
{
    STRICT_MODE = false;
    string testcaseName = "double_signed_underflow";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDouble( 0 ), -HUGE_VALL );
    ASSERT_TRUE( abs( column->GetDouble( 1 ) + 1.79e308 ) < 0.000001 );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, decimal_overflow_scale1)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_overflow_scale1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "9999999999" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "9999999999" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName );
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "9999999999" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "9999999999" );
}
TEST_F(UT_loaddata, decimal_overflow_scale2)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_overflow_scale2";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName );
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );
}
TEST_F(UT_loaddata, decimal_overflow_scale3)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_overflow_scale3";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 1 );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, decimal_overflow_scale4)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_overflow_scale4";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "100.0" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName );

    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "100.0" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );
}
TEST_F(UT_loaddata, decimal_overflow_int1)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_overflow_int1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "9999999999" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "9999999999" ) );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, decimal_overflow_int2)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_overflow_int2";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "999999999.9" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "999999999.9" ) );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, decimal_underflow_scale1)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_underflow_scale1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-9999999999" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-9999999999" ) );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName );

    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-9999999999" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-9999999999" ) );
}
TEST_F(UT_loaddata, decimal_underflow_scale2)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_underflow_scale2";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-999999999.9" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-999999999.9" ) );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName );

    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-999999999.9" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-999999999.9" ) );
}
TEST_F(UT_loaddata, decimal_underflow_scale3)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_underflow_scale3";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-999999999.9" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-999999999.9" ) );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_WARN_DATA_OUT_OF_RANGE, "Out of range value for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, decimal_underflow_scale4)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_underflow_scale4";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    cout << "decimal under flow value : " << column->GetDecimalAsString(0) << endl;
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-100.0" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-999999999.9" ) );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName );

    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "-100.0" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "-999999999.9" ) );
}
TEST_F(UT_loaddata, decimal_invalid1)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_invalid1";
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal not null );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    auto columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    auto tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "0" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "9999999999" ) );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    ASSERT_EQ(result->GetErrorMessage(), "Incorrect decimal value: 'abc' for column 'f1' at row 3");
}
TEST_F(UT_loaddata, decimal_invalid2)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_invalid2";
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal not null );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    auto columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    auto tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "0" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "9999999999" ) );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    // mysql: Note  | 1265 | Data truncated for column 'f1' at row 1 
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    ASSERT_EQ(result->GetErrorMessage(), "Incorrect decimal value: '123abc' for column 'f1' at row 3");
}
TEST_F(UT_loaddata, decimal_invalid3)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_invalid3";
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal(10, 1) not null );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    auto columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    auto tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 1 );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "0.0" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "999999999.9" );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal(10, 1) not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD);
    ASSERT_EQ(result->GetErrorMessage(), "Incorrect decimal value: 'abc' for column 'f1' at row 3");
}
TEST_F(UT_loaddata, decimal_invalid4)
{
    STRICT_MODE = false;
    string testcaseName = "decimal_invalid4";
    string tableName = "t_" + testcaseName;
    InitTable(dbName, tableName);

    string sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal(10, 1) not null );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/";
    csvPath.append( testcaseName ).append( ".csv" );
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    auto columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    auto tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "0" ) );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "999999999.9" ) );

    // strict mode
    STRICT_MODE = true;
    InitTable(dbName, tableName);

    sql = "create table ";
    sql.append( tableName );
    sql.append( "( f1 decimal(10, 1) not null );" );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    ASSERT_EQ(result->GetErrorMessage(), "Incorrect decimal value: '1.1abc' for column 'f1' at row 3");
}
TEST_F(UT_loaddata, datetime_invalid1)
{
    string zeroDatetime = "0000-00-00 00:00:00";
    STRICT_MODE = false;
    string testcaseName = "datetime_invalid1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), zeroDatetime );
    ASSERT_EQ( column->GetDatetimeAsString( 1 ), "2020-01-01 11:11:11" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect datetime value: 'abc' for column 'f1' at row 3");
}
TEST_F(UT_loaddata, datetime_invalid2)
{
    string zeroDatetime = "0000-00-00 00:00:00";
    string testcaseName = "datetime_invalid2";
    string tableName = "t_" + testcaseName;

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect datetime value: '2020-00-11 11:11:11' for column 'f1' at row 3" );

    STRICT_MODE = false;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), "2020-00-11 11:11:11" );
    ASSERT_EQ( column->GetDatetimeAsString( 1 ), "2020-01-01 11:11:11" );
}
TEST_F(UT_loaddata, datetime_invalid3)
{
    string zeroDatetime = "0000-00-00 00:00:00";
    STRICT_MODE = false;
    string testcaseName = "datetime_invalid3";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDatetimeAsString( 0 ), "2020-11-00 11:11:11" );
    ASSERT_EQ( column->GetDatetimeAsString( 1 ), "2020-01-01 11:11:11" );

    // strict mode
    STRICT_MODE = true;
    LoadData(testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect datetime value: '2020-11-00 11:11:11' for column 'f1' at row 3");
}
TEST_F(UT_loaddata, timestamp_invalid1)
{
    string zeroDatetime = "0000-00-00 00:00:00";
    STRICT_MODE = false;
    string testcaseName = "timestamp_invalid1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetTimestampAsString( 0 ), zeroDatetime );
    ASSERT_EQ( column->GetTimestampAsString( 1 ), "2020-01-01 11:11:11" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect timestamp value: 'abc' for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, date_invalid1)
{
    string zeroDate = "0000-00-00";
    STRICT_MODE = false;
    string testcaseName = "date_invalid1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDateAsString( 0 ), zeroDate );
    ASSERT_EQ( column->GetDateAsString( 1 ), "2020-01-01" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect date value: 'abc' for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, time_invalid1)
{
    string zeroTime = "00:00:00";
    string testcaseName = "time_invalid1";
    string tableName = "t_" + testcaseName;

    STRICT_MODE = false;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    auto timeValue = *( column->GetTime( 0 ) );
    ASSERT_EQ( timeValue, aries_acc::AriesTime() );
    ASSERT_EQ( column->GetTimeAsString( 1 ), "11:11:11" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect time value: 'abc' for column 'f1' at row 3" );
}
TEST_F(UT_loaddata, year_invalid1)
{
    STRICT_MODE = false;
    string testcaseName = "year_invalid1";
    string tableName = "t_" + testcaseName;
    LoadData( testcaseName );

    AriesTableBlockUPtr table;
    VerifyResultCount( tableName, 1, 2, table );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    auto yearValue = *( column->GetYear( 0 ) );
    ASSERT_EQ( yearValue, aries_acc::AriesYear() );
    ASSERT_EQ( column->GetYearAsString( 1 ), "2020" );

    // strict mode
    STRICT_MODE = true;
    LoadData( testcaseName, ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "Incorrect year value: 'abc' for column 'f1' at row 3" );
}
/*
number_invalid_values2.csv
non strict mode:
mysql> show warnings;
+---------+------+---------------------------------------------+
| Level   | Code | Message                                     |
+---------+------+---------------------------------------------+
| Warning | 1265 | Data truncated for column 'f1' at row 1     |
| Warning | 1265 | Data truncated for column 'f2' at row 1     |
| Warning | 1265 | Data truncated for column 'f3' at row 1     |
| Warning | 1265 | Data truncated for column 'f4' at row 1     |
| Warning | 1265 | Data truncated for column 'f5' at row 1     |
| Warning | 1265 | Data truncated for column 'f6' at row 1     |
| Note    | 1265 | Data truncated for column 'f7' at row 1     |
| Warning | 1265 | Data truncated for column 'f8' at row 1     |
| Warning | 1264 | Out of range value for column 'f1' at row 3 |
+---------+------+---------------------------------------------+
9 rows in set (0.00 sec)

mysql> select * from t_number_invalid_values2;
+------+------+------+------+------+------+------+------+
| f1   | f2   | f3   | f4   | f5   | f6   | f7   | f8   |
+------+------+------+------+------+------+------+------+
|    1 |    1 |    1 |    1 |    1 |    1 |    1 |    1 |
|    1 |    1 |    1 |    1 |    1 |    1 |    1 |    1 |
|  127 |    1 |    1 |    1 |    1 |    1 |    1 |    1 |
+------+------+------+------+------+------+------+------+
3 rows in set (0.00 sec)


strict mode:
mysql> show warnings;
+-------+------+--------------------------------------------------------+
| Level | Code | Message                                                |
+-------+------+--------------------------------------------------------+
| Error | 1265 | Data truncated for column 'f1' at row 1                |
| Error | 1265 | Data truncated for column 'f2' at row 1                |
| Error | 1265 | Data truncated for column 'f3' at row 1                |
| Error | 1265 | Data truncated for column 'f4' at row 1                |
| Error | 1265 | Data truncated for column 'f5' at row 1                |
| Error | 1265 | Data truncated for column 'f6' at row 1                |
| Error | 1366 | Incorrect decimal value: '1a' for column 'f7' at row 1 |
| Error | 1265 | Data truncated for column 'f8' at row 1                |
+-------+------+--------------------------------------------------------+
8 rows in set (0.00 sec)

*/

/*
float unsigned:

mysql>  load data infile '/home/tengjp/git/aries-test/test/testcases/test_resources/loaddata/csv/float_unsigned_underflow.csv' into table t_float_unsigned_underflow ignore 2 lines; 
Query OK, 2 rows affected, 2 warnings (0.01 sec)
Records: 2  Deleted: 0  Skipped: 0  Warnings: 2

mysql> show warnings;
+---------+------+---------------------------------------------+
| Level   | Code | Message                                     |
+---------+------+---------------------------------------------+
| Warning | 1264 | Out of range value for column 'f1' at row 1 |
| Warning | 1264 | Out of range value for column 'f1' at row 2 |
+---------+------+---------------------------------------------+
2 rows in set (0.00 sec)

mysql> select * from t_float_unsigned_underflow;
+------+
| f1   |
+------+
|    0 |
|    0 |
+------+
2 rows in set (0.00 sec)

mysql> insert into t_float_unsigned_underflow values (-1);
Query OK, 1 row affected, 1 warning (0.00 sec)

mysql> show warnings;
+---------+------+---------------------------------------------+
| Level   | Code | Message                                     |
+---------+------+---------------------------------------------+
| Warning | 1264 | Out of range value for column 'f1' at row 1 |
+---------+------+---------------------------------------------+
1 row in set (0.00 sec)


*/
/*
If FIELDS ENCLOSED BY is not empty, a field containing the literal word NULL as its value is read as
a NULL value. This differs from the word NULL enclosed within FIELDS ENCLOSED BY characters,
which is read as the string 'NULL'.
*/
TEST_F(UT_loaddata, null_enclosed) {
    string tableName = "t_null_enclosed";
    InitTable(dbName, tableName);

    string sql = "create table t_null_enclosed(f1 int, f2 char(64));";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/null_enclosed.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields enclosed by '\"' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_TRUE( column->isInt32DataNull( 0 ) );
    ASSERT_TRUE( column->isInt32DataNull( 1 ) );
    ASSERT_TRUE( column->isInt32DataNull( 2 ) );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 3 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_TRUE( column->isStringDataNull( 0 ) );
    ASSERT_TRUE( column->isStringDataNull( 1 ) );
    ASSERT_EQ( column->GetNullableString( 2 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 3 ), "aaa" );
    ASSERT_EQ( column->GetNullableString( 4 ), "bbb" );
}
TEST_F(UT_loaddata, empty_value) {
    string tableName = "t_empty_value";
    InitTable(dbName, tableName);

    string sql = "create table t_empty_value(f1 int, f2 char(64));";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/empty_value.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' enclosed by '\"' ignore 2 lines";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from ";
    sql.append( tableName );
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 3 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "" );
    ASSERT_EQ( column->GetNullableString( 1 ), "" );
    ASSERT_EQ( column->GetNullableString( 2 ), "222" );
    ASSERT_EQ( column->GetNullableString( 3 ), "333" );
}

TEST_F(UT_loaddata, char_exceed_schema_len) {
    string tableName = "t_char_exceed_schema_len";
    InitTable(dbName, tableName);

    string sql = "create table " + tableName + "(f1 char(4) not null);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/char_exceed_schema_len.csv";

    // strict mode
    STRICT_MODE = true;
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);

    ASSERT_TRUE( !result->IsSuccess() );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    // non strict mode
    STRICT_MODE = false;
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1234" ); // string is truncated

    tableName = "t_char_dict_exceed_schema_len";
    string dictName = "UT_loaddata_char_exceed_schema_len";
    InitTable(dbName, tableName);
    sql = "create table " + tableName + "(f1 char(4) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    // strict mode
    STRICT_MODE = true;
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);

    ASSERT_TRUE( !result->IsSuccess() );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    // non strict mode
    STRICT_MODE = false;
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 2 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1234" ); // string is truncated

    STRICT_MODE = true;
}

TEST_F(UT_loaddata, char_max_len) {
    string tableName = "t_char_max_len";
    InitTable(dbName, tableName);

    char* oldOptmzCharColumn = getenv("RATEUP_OPTIMIZE_CHAR_COLUMN");
    setenv( "RATEUP_OPTIMIZE_CHAR_COLUMN", "true", true );

    string sql = "create table " + tableName + "(f1 char(1024));";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/char_excceed_max_len.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);

    if ( !oldOptmzCharColumn )
        unsetenv( "RATEUP_OPTIMIZE_CHAR_COLUMN" );
    else
        setenv( "RATEUP_OPTIMIZE_CHAR_COLUMN", oldOptmzCharColumn, true );

    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_FIELDLENGTH);

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
}

/*
TEST_F(UT_loaddata, convert_error_int) {
    STRICT_MODE = true;
    string tableName = "t_int";
    InitTable(dbName, tableName);

    string sql = "create table t_int(f1 int);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/loaddata/csv/convert_error_int.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( !result->IsSuccess() );
    ASSERT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    ASSERT_EQ( result->GetErrorMessage(), "Incorrect integer value: 'abc' for column 'f1' at row 4" );
}
*/

TEST_F(UT_loaddata, int_char)
{
    string tableName = "t_int_char";
    InitTable( dbName, tableName );

    string sql = "create table t_int_char(f1 int, f2 char(128));";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    // empty table
    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 0 );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_int_char-default.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    // select again
    sql = "select * from " + tableName + ";";
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 8 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 7 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 8 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "abc" );
    ASSERT_EQ( column->GetNullableString( 1 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 2 ), "a'b'c" );
    ASSERT_EQ( column->GetNullableString( 3 ), "a'b\"c" );
    ASSERT_EQ( column->GetNullableString( 4 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ab\\c" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ab,c" );
    ASSERT_EQ( column->GetNullableString( 7 ), "ab\tc" );
}

TEST_F(UT_loaddata, int_char_enclosed )
{
    string tableName = "t_int_char";
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table t_int_char(f1 int, f2 char(128));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_int_char-enclosed.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields enclosed by '\"';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 8 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 7 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 8 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "abc" );
    ASSERT_EQ( column->GetNullableString( 1 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 2 ), "a'b'c" );
    ASSERT_EQ( column->GetNullableString( 3 ), "a'b\"c" );
    ASSERT_EQ( column->GetNullableString( 4 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ab\\c" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ab,c" );
    ASSERT_EQ( column->GetNullableString( 7 ), "ab\tc" );
}
TEST_F(UT_loaddata, int_char_fields_terminated_by )
{
    string tableName = "t_int_char";
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table t_int_char(f1 int, f2 char(128));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_int_char-f-termby.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 8 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 7 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 8 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "abc" );
    ASSERT_EQ( column->GetNullableString( 1 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 2 ), "a'b'c" );
    ASSERT_EQ( column->GetNullableString( 3 ), "a'b\"c" );
    ASSERT_EQ( column->GetNullableString( 4 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ab\\c" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ab,c" );
    ASSERT_EQ( column->GetNullableString( 7 ), "ab\tc" );
}
TEST_F(UT_loaddata, int_char_fields_terminated_enclosed )
{
    string tableName = "t_int_char";
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table t_int_char(f1 int, f2 char(128));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_int_char-f-termby-enclosedby.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ',' enclosed by '\"';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 8 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 7 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 8 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "abc" );
    ASSERT_EQ( column->GetNullableString( 1 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 2 ), "a'b'c" );
    ASSERT_EQ( column->GetNullableString( 3 ), "a'b\"c" );
    ASSERT_EQ( column->GetNullableString( 4 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ab\\c" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ab,c" );
    ASSERT_EQ( column->GetNullableString( 7 ), "ab\tc" );
}
TEST_F(UT_loaddata, int_char_fields_terminated_opt_enclosed )
{
    string tableName = "t_int_char";
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table t_int_char(f1 int, f2 char(128));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_int_char-f-termby-opt-enclosedby.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ',' optionally enclosed by '\"';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 8 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 7 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 8 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "abc" );
    ASSERT_EQ( column->GetNullableString( 1 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 2 ), "a'b'c" );
    ASSERT_EQ( column->GetNullableString( 3 ), "a'b\"c" );
    ASSERT_EQ( column->GetNullableString( 4 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ab\\c" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ab,c" );
    ASSERT_EQ( column->GetNullableString( 7 ), "ab\tc" );
}
TEST_F(UT_loaddata, int_char_field_value_include_newline )
{
    string tableName = "t_int_char";
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table t_int_char(f1 int, f2 char(128));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_int_char-has-newline-default.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 9 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 7 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 8 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "abc" );
    ASSERT_EQ( column->GetNullableString( 1 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 2 ), "a'b'c" );
    ASSERT_EQ( column->GetNullableString( 3 ), "a'b\"c" );
    ASSERT_EQ( column->GetNullableString( 4 ), "a\"b\"c" );
    ASSERT_EQ( column->GetNullableString( 5 ), "ab\\c" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ab,c" );
    ASSERT_EQ( column->GetNullableString( 7 ), "ab\tc" );
    ASSERT_EQ( column->GetNullableString( 8 ), "aaa\nbbbb" );
}
TEST_F(UT_loaddata, char_int_bigint_float_double_decimal_datetime )
{
    string tableName = "t_load_char_int_bigint_float_double_decimal_datetime";
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table t_load_char_int_bigint_float_double_decimal_datetime("
          "f1 char(255),"
          "f2 int,"
          "f3 bigint,"
          "f4 float,"
          "f5 double,"
          "f6 decimal(15, 5),"
          "f7 datetime );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/load_char_int_bigint_float_double_decimal_datetime.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    auto table = executeSql( sql, dbName );
    ASSERT_EQ( table->GetRowCount(), 3);

    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaaaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbbbb" );
    ASSERT_EQ( column->GetString( 2 ), "ccccc" );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 3 );

    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 11111 );
    ASSERT_EQ( column->GetNullableInt64( 1 ).value, 22222 );
    ASSERT_EQ( column->GetNullableInt64( 2 ).value, 33333 );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( column->GetNullableFloat( 0 ).value - 1.0001 < 0.00001 );
    ASSERT_TRUE( column->GetNullableFloat( 1 ).value - 2.0002 < 0.00001 );
    ASSERT_TRUE( column->GetNullableFloat( 2 ).value - 3.0003 < 0.00001 );

    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableDouble( 0 ).value, 1.0002 );
    ASSERT_EQ( column->GetNullableDouble( 1 ).value, 2.0002 );
    ASSERT_EQ( column->GetNullableDouble( 2 ).value, 3.0003 );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 6 );
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "1123456789.11111" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "2123456789.22222" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 2, colEntry->numeric_precision, colEntry->numeric_scale ), "3123456789.33333" );

    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 0 ), "2019-12-01 01:01:01" );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 1 ), "2019-12-02 02:02:02" );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 2 ), "2019-12-03 03:03:03" );

    string newCharValue = "0123456789";
    string newIntValue = "1111111111";
    string newBigintValue = "1111111111";
    float newFloatValue = (float)1111111111.1;
    string newFloatValueStr = std::to_string( newFloatValue );
    double newDoubleValue = 1111111111.1;
    string newDoubleValueStr = std::to_string( newDoubleValue );
    string newDecimalValue = "9999999999.99999";
    string newDatetimeValue = "2020-12-03 11:46:00";

    sql = "insert into " + tableName + " values ( " +
          "'" + newCharValue + "', " +
          newIntValue + ", " +
          newBigintValue + ", " +
          newFloatValueStr + ", " +
          newDoubleValueStr + ", " +
          newDecimalValue + ", " +
          "'" + newDatetimeValue + "'" +
          " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + ";";
    table = executeSql( sql, dbName );
    ASSERT_EQ( table->GetRowCount(), 4);

    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaaaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbbbb" );
    ASSERT_EQ( column->GetString( 2 ), "ccccc" );
    ASSERT_EQ( column->GetString( 3 ), newCharValue );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 3 );
    ASSERT_EQ( column->GetInt32AsString( 3 ), newIntValue );

    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 11111 );
    ASSERT_EQ( column->GetNullableInt64( 1 ).value, 22222 );
    ASSERT_EQ( column->GetNullableInt64( 2 ).value, 33333 );
    ASSERT_EQ( column->GetInt64AsString( 3 ), newBigintValue );

    column = table->GetColumnBuffer( 4 );
    ASSERT_TRUE( column->GetNullableFloat( 0 ).value - 1.0001 < 0.00001 );
    ASSERT_TRUE( column->GetNullableFloat( 1 ).value - 2.0002 < 0.00001 );
    ASSERT_TRUE( column->GetNullableFloat( 2 ).value - 3.0003 < 0.00001 );
    ASSERT_TRUE( column->GetNullableFloat( 3 ).value - newFloatValue < 0.00001 );

    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableDouble( 0 ).value, 1.0002 );
    ASSERT_EQ( column->GetNullableDouble( 1 ).value, 2.0002 );
    ASSERT_EQ( column->GetNullableDouble( 2 ).value, 3.0003 );
    ASSERT_EQ( column->GetNullableDouble( 3 ).value, newDoubleValue );

    colEntry = tableEntry->GetColumnById( 6 );
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetCompactDecimalAsString( 0, colEntry->numeric_precision, colEntry->numeric_scale ), "1123456789.11111" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 1, colEntry->numeric_precision, colEntry->numeric_scale ), "2123456789.22222" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 2, colEntry->numeric_precision, colEntry->numeric_scale ), "3123456789.33333" );
    ASSERT_EQ( column->GetCompactDecimalAsString( 3, colEntry->numeric_precision, colEntry->numeric_scale ), newDecimalValue );

    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 0 ), "2019-12-01 01:01:01" );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 1 ), "2019-12-02 02:02:02" );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 2 ), "2019-12-03 03:03:03" );
    ASSERT_EQ( column->GetNullableDatetimeAsString( 3 ), newDatetimeValue );
}

void testCasePrepare( const string& tableName )
{
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );
}

// 重复单主键，int
TEST_F( UT_loaddata, dupSinglePkInt )
{
    string tableName( "t_dupSinglePkInt" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 int primary key, f2 char(128) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_single_pk_int.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复单主键，decimal
TEST_F( UT_loaddata, dupSinglePkDec )
{
    string tableName( "t_dupSinglePkDec" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 decimal(10, 2) primary key, f2 double )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_single_pk_dec.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复单主键，char
TEST_F( UT_loaddata, dupSinglePkChar )
{
    string tableName( "t_dupSinglePkChar" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 char(10) primary key, f2 float )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_single_pk_char.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复单主键，char(1)
TEST_F( UT_loaddata, dupSinglePkChar1 )
{
    string tableName( "t_dupSinglePkChar1" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 char(1) primary key, f2 float )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_single_pk_char1.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复复合主键，int, long
TEST_F( UT_loaddata, dupMultiPkIntLong )
{
    string tableName( "t_dupMkIntLong" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 int, f2 bigint, f3 float, " +
                 "PRIMARY KEY ( f1, f2 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    vector< string > pkNames;
    pkNames.push_back( "f1" );
    pkNames.push_back( "f2" );
    // tableEntry->SetPrimaryKey( pkNames );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_multiple_pk_int_long.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    bool ret = aries::schema::fix_column_length( aries::Configuartion::GetInstance().GetColumnDataDirectory() );
    ASSERT_TRUE( ret );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复复合主键，int, decimal
TEST_F( UT_loaddata, dupMultiPkIntDec )
{
    string tableName( "t_dupMkIntDec" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 int, f2 decimal, f3 float, " +
                 "PRIMARY KEY ( f1, f2 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    vector< string > pkNames;
    pkNames.push_back( "f1" );
    pkNames.push_back( "f2" );
    // tableEntry->SetPrimaryKey( pkNames );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_multiple_pk_int_dec.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    bool ret = aries::schema::fix_column_length( aries::Configuartion::GetInstance().GetColumnDataDirectory() );
    ASSERT_TRUE( ret );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复复合主键，int, char, char
TEST_F( UT_loaddata, dupMultiPkIntCharChar )
{
    string tableName( "t_dupMkIntCharChar" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 int, f2 char(10), f3 char(10), f4 float, " +
                 "PRIMARY KEY ( f1, f2, f3 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    vector< string > pkNames;
    pkNames.push_back( "f1" );
    pkNames.push_back( "f2" );
    pkNames.push_back( "f3" );
    // tableEntry->SetPrimaryKey( pkNames );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_multiple_pk_int_char_char.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    bool ret = aries::schema::fix_column_length( aries::Configuartion::GetInstance().GetColumnDataDirectory() );
    ASSERT_TRUE( ret );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}
// 重复复合主键，decimal, char
TEST_F( UT_loaddata, dupMultiPkDecCharChar )
{
    string tableName( "t_dupMkDecCharChar" );
    testCasePrepare( tableName );

    string sql = "CREATE TABLE " + tableName + "( f1 decimal(15,3), f2 char(10), f3 char(10), f4 float, " +
                 "PRIMARY KEY ( f1, f2, f3 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );

    string csvPath = cwd + "/test_resources/loaddata/csv/dup_multiple_pk_dec_char_char.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    bool ret = aries::schema::fix_column_length( aries::Configuartion::GetInstance().GetColumnDataDirectory() );
    ASSERT_TRUE( ret );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( ER_DUP_ENTRY, result->GetErrorCode() );
}

// test foreign keys
// TEST_F( UT_loaddata, fkIgnoreCheck )
// {
//     string tableNameP( "t_fkIgnoreCheck_P" );
//     string tableNameC( "t_fkIgnoreCheck_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );

//     string sql = "CREATE TABLE " + tableNameP + "( f1 int primary key, f2 char(128) )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );

//     sql = "CREATE TABLE " + tableNameC + "( f1 int, f2 char(128), " +
//           "foreign key ( f1 ) references " + tableNameP + " ( f1 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );

//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_single_int_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );

//     // disable foreign_key_checks
//     THD* thd = current_thd;
//     thd->variables.foreign_key_checks = 0;

//     csvPath = cwd + "/test_resources/loaddata/csv/fk_single_int_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     thd->variables.foreign_key_checks = 1;
//     ASSERT_TRUE( result->IsSuccess() );
//     sql = "SELECT * FROM " + tableNameC;
//     auto table = executeSql( sql, dbName );
//     int columnCount = table->GetColumnCount();
//     ASSERT_EQ( columnCount, 2 );
//     int tupleNum = table->GetRowCount();
//     ASSERT_EQ( tupleNum, 2 );
//     auto column = table->GetColumnBuffer( 1 );
//     ASSERT_EQ( column->GetNullableInt32( 0 ).value, 3 );
//     ASSERT_EQ( column->GetNullableInt32( 1 ).value, 4 );

//     column = table->GetColumnBuffer( 2 );
//     ASSERT_EQ( column->GetNullableString( 0 ), "ccc" );
//     ASSERT_EQ( column->GetNullableString( 1 ), "ddd" );
// }
// 单外键，int
// TEST_F( UT_loaddata, fkSingleInt )
// {
//     string tableNameP( "t_fkSingleInt_p" );
//     string tableNameC( "t_fkSingleInt_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 int primary key, f2 char(128) )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 int, f2 char(128), " +
//           "foreign key ( f1 ) references " + tableNameP + " ( f1 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_single_int_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_single_int_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// //单外键，decimal
// TEST_F( UT_loaddata, fkSingleDec )
// {
//     string tableNameP( "t_fkSingleDec_P" );
//     string tableNameC( "t_fkSingleDec_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 decimal(10, 2) primary key, f2 double )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 decimal(10, 2), f2 double, " +
//           "foreign key ( f1 ) references " + tableNameP + " ( f1 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_single_dec_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_single_dec_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// // 单外键，char
// TEST_F( UT_loaddata, fkSingleChar )
// {
//     string tableNameP( "t_fkSingleChar_p" );
//     string tableNameC( "t_fkSingleChar_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 char(4) primary key, f2 float )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 char(4), f2 float, " +
//           "foreign key ( f1 ) references " + tableNameP + " ( f1 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_single_char_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_single_char_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// // 单外键，char1
// TEST_F( UT_loaddata, fkSingleChar1 )
// {
//     string tableNameP( "t_fkSingleChar1_p" );
//     string tableNameC( "t_fkSingleChar1_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 char(1) primary key, f2 float )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 char(1), f2 float, " +
//           "foreign key ( f1 ) references " + tableNameP + " ( f1 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_single_char1_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_single_char1_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// // 复合外键，int, long
// TEST_F( UT_loaddata, fkMultiIntLong )
// {
//     string tableNameP( "t_fkMultiIntLong_P" );
//     string tableNameC( "t_fkMultiIntLong_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 int, f2 bigint, " +
//                  "PRIMARY KEY ( f1, f2 ) )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 int, f2 bigint, " +
//           "foreign key ( f1, f2 ) references " + tableNameP + " ( f1, f2 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_int_long_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_int_long_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// // 复合外键，int, char
// TEST_F( UT_loaddata, fkMultiIntChar )
// {
//     string tableNameP( "t_fkMultiIntChar_P" );
//     string tableNameC( "t_fkMultiIntChar_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 int, f2 char(2), " +
//                  "PRIMARY KEY ( f1, f2 ) )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 int, f2 char(2), " +
//           "foreign key ( f1, f2 ) references " + tableNameP + " ( f1, f2 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_int_char_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_int_char_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// // 复合外键，int, dec
// TEST_F( UT_loaddata, fkMultiIntDec )
// {
//     string tableNameP( "t_fkMultiIntDec_P" );
//     string tableNameC( "t_fkMultiIntDec_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 int, f2 decimal(20, 2), " +
//                  "PRIMARY KEY ( f1, f2 ) )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 int, f2 decimal(20, 2), " +
//           "foreign key ( f1, f2 ) references " + tableNameP + " ( f1, f2 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_int_dec_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_int_dec_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
// // 复合外键，char, dec
// TEST_F( UT_loaddata, fkMultiCharDec )
// {
//     string tableNameP( "t_fkMultiCharDec_P" );
//     string tableNameC( "t_fkMultiCharDec_C" );
//     testCasePrepare( tableNameC );
//     testCasePrepare( tableNameP );
// 
//     string sql = "CREATE TABLE " + tableNameP + "( f1 char(2), f2 decimal(20, 2), " +
//                  "PRIMARY KEY ( f1, f2 ) )";
//     auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     sql = "CREATE TABLE " + tableNameC + "( f1 char(2), f2 decimal(20, 2), " +
//           "foreign key ( f1, f2 ) references " + tableNameP + " ( f1, f2 ) )";
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     string csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_char_dec_p.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameP;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_TRUE( result->IsSuccess() );
// 
//     csvPath = cwd + "/test_resources/loaddata/csv/fk_multi_char_dec_c.csv";
//     sql = "load data infile '" +  csvPath + "' into table " + tableNameC;
//     result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
//     ASSERT_EQ( ER_NO_REFERENCED_ROW_2, result->GetErrorCode() );
// }
TEST_F( UT_loaddata, clear_table_after_loaddata )
{
    string tableName( "t_clearTableAfterLoadData" );
    testCasePrepare( tableName );
    string sql = "CREATE TABLE " + tableName + "( f1 int, f2 char(3) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvPath = cwd + "/test_resources/loaddata/csv/clear_table_after_loaddata.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "SELECT * FROM " + tableName;
    auto table = executeSql( sql, dbName );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbb" );

    int insertI1 = 3;
    int insertI2 = 4;
    string insertS1( "ccc" );
    string insertS2( "ddd" );
    sql = "INSERT INTO " + tableName + " VALUES ( " + std::to_string( insertI1 ) + ", '" + insertS1 +
           "'), ( " + std::to_string( insertI2 ) + ", '" + insertS2 + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "DELETE FROM " + tableName + " WHERE f1 = 2";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "SELECT * FROM " + tableName;
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 3 );

    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 3 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 4 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "ccc" );
    ASSERT_EQ( column->GetNullableString( 2 ), "ddd" );

    testCasePrepare( tableName );
    sql = "CREATE TABLE " + tableName + "( f1 int, f2 char(3) )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    // load data in replace mode
    sql = "load data infile '" +  csvPath + "' into table " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "SELECT * FROM " + tableName;
    table = executeSql( sql, dbName );
    columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify table content is replaced
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 2 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbb" );
}

TEST_F( UT_loaddata, partition_empty_csv )
{
    string tableName( "partition_empty_csv" );
    InitTable( dbName, tableName );

    string sql = R"(
CREATE TABLE partition_empty_csv(
 date1 date,
 some_data int
)
PARTITION BY RANGE ( date1 ) (
 PARTITION p0 VALUES LESS THAN ('1990-01-01'),
 PARTITION p1 VALUES LESS THAN ('2000-01-01')
);
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "select * from " + tableName + " where date1 < '1990-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    string csvPath = cwd + "/test_resources/loaddata/csv/empty.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "select * from " + tableName + " where date1 < '1990-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

}

TEST_F( UT_loaddata, partition_no_data_partition )
{
    string tableName( "partition_no_data_partition" );
    InitTable( dbName, tableName );

    string sql = R"(
CREATE TABLE partition_no_data_partition(
 date1 date,
 some_data int
)
PARTITION BY RANGE ( date1 ) (
 PARTITION p0 VALUES LESS THAN ('1990-01-01'),
 PARTITION p1 VALUES LESS THAN ('2000-01-01'),
 PARTITION p2 VALUES LESS THAN ('2010-01-01'),
 PARTITION p3 VALUES LESS THAN ('2020-01-01'),
 PARTITION p4 VALUES LESS THAN MAXVALUE
);
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    string csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_no_data_partition.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName + " order by date1";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isDateDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), "1991-01-01" );
    ASSERT_EQ( columnBuff->isDateDataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 1 ), "2019-01-01" );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 1991 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 2019 );

    sql = "select * from " + tableName + " where date1 < '1990-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "select * from " + tableName + " where date1 >= '1990-01-01' and date1 < '2000-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isDateDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), "1991-01-01" );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 1991 );

    sql = "select * from " + tableName + " where date1 >= '2000-01-01' and date1 < '2010-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "select * from " + tableName + " where date1 >= '2010-01-01' and date1 < '2020-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isDateDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableDateAsString( 0 ), "2019-01-01" );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 2019 );

    sql = "select * from " + tableName + " where date1 > '2020-01-01'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
}

TEST_F( UT_loaddata, partition_date )
{
    string tableName( "partition_date" );
    InitTable( dbName, tableName );

    string sql = R"(
CREATE TABLE partition_date(
 date1 date,
 some_data int
)
PARTITION BY RANGE ( date1 ) (
 PARTITION p0 VALUES LESS THAN ('1990-01-01'),
 PARTITION p1 VALUES LESS THAN ('2000-01-01')
);
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    string csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_date_nopartition_for_value.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), ER_NO_PARTITION_FOR_GIVEN_VALUE );

    InitTable( dbName, tableName );
    sql = R"(
CREATE TABLE partition_date(
 date1 date,
 some_data int
)
PARTITION BY RANGE ( date1 ) (
 PARTITION p0 VALUES LESS THAN ('1990-01-01'),
 PARTITION p1 VALUES LESS THAN ('2000-01-01'),
 PARTITION p2 VALUES LESS THAN MAXVALUE
);
    )";

    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    auto tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 0 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 3 );

    auto &partitions = tableEntry->GetPartitions();
    auto partition = partitions[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p0" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "1990-01-01" );

    partition = partitions[ 1 ];
    ASSERT_EQ( partition->m_partitionName, "p1" );
    ASSERT_EQ( partition->m_partOrdPos, 2 );
    ASSERT_EQ( partition->m_partDesc, "2000-01-01" );

    partition = partitions[ 2 ];
    ASSERT_EQ( partition->m_partitionName, "p2" );
    ASSERT_EQ( partition->m_partOrdPos, 3 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_date.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    AriesInitialTable initTable( dbName, tableName );
    auto partitionMetaInfos = initTable.GetPartitionMetaInfo();
    auto partitionCount = partitionMetaInfos.size();
    ASSERT_EQ( partitionCount, 3 );

    auto partitionMetaInfo = partitionMetaInfos[ 0 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 2 ); // null is in this partition
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );

    partitionMetaInfo = partitionMetaInfos[ 1 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 9 );
#ifdef ARIES_INIT_TABLE_TEST
    ASSERT_EQ( partitionMetaInfo.BlockCount, 2 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 2 );
#else
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );
#endif

    partitionMetaInfo = partitionMetaInfos[ 2 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );

    uint32_t partitionIndex = 0;
    auto resultTable = initTable.GetPartitionData( { 1, 2 }, partitionIndex );
    ASSERT_EQ( resultTable->GetRowCount(), 2 );

    auto columnBuff = resultTable->GetColumnBuffer( 1 );
    auto columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->isDateDataNull( 0 ), true );
    ASSERT_EQ( columnBuffSorted->GetNullableDateAsString( 1 ), "1980-11-30" );

    columnBuff = resultTable->GetColumnBuffer( 2 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 1 ).value, 1980 );

    ++partitionIndex;
    resultTable = initTable.GetPartitionData( { 1, 2 }, partitionIndex );
    ASSERT_EQ( resultTable->GetRowCount(), 9 );

    columnBuff = resultTable->GetColumnBuffer( 1 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableDateAsString( 0 ), "1996-01-13" );
    ASSERT_EQ( columnBuffSorted->GetNullableDateAsString( 5 ), "1996-06-13" );
    ASSERT_EQ( columnBuffSorted->GetNullableDateAsString( 8 ), "1996-09-13" );

    columnBuff = resultTable->GetColumnBuffer( 2 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 0 ).value, 1996 );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 5 ).value, 1996 );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 8 ).value, 1996 );

    ++partitionIndex;
    resultTable = initTable.GetPartitionData( { 1, 2 }, partitionIndex );
    ASSERT_EQ( resultTable->GetRowCount(), 1 );

    columnBuff = resultTable->GetColumnBuffer( 1 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableDateAsString( 0 ), "2000-02-10" );

    columnBuff = resultTable->GetColumnBuffer( 2 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 0 ).value, 2000 );

    // with MAXVALUE partition ONLY
    tableName = "partition_date_maxvalue_only";
    InitTable( dbName, tableName );
    sql = R"(
CREATE TABLE partition_date_maxvalue_only(
 date1 date,
 some_data int
)
PARTITION BY RANGE ( date1 ) (
 PARTITION p2 VALUES LESS THAN MAXVALUE
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 0 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 1 );

    auto &partitions2 = tableEntry->GetPartitions();
    partition = partitions2[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p2" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_date.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    AriesInitialTable initTable2( dbName, tableName );
    partitionMetaInfos = initTable2.GetPartitionMetaInfo();
    partitionCount = partitionMetaInfos.size();
    ASSERT_EQ( partitionCount, 1 );

    partitionMetaInfo = partitionMetaInfos[ 0 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 12 );
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );
}
TEST_F( UT_loaddata, partition_datetime )
{
    string tableName( "partition_datetime" );
    InitTable( dbName, tableName );

    string sql = R"(
CREATE TABLE partition_datetime(
 dateTime1 datetime,
 some_data int
)
PARTITION BY RANGE ( dateTime1 ) (
 PARTITION p0 VALUES LESS THAN ('1990-01-01 00:00:01'),
 PARTITION p1 VALUES LESS THAN ('2000-01-01 00:00:01')
);
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    string csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_datetime_nopartition_for_value.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( result->GetErrorCode(), ER_NO_PARTITION_FOR_GIVEN_VALUE );

    InitTable( dbName, tableName );
    sql = R"(
CREATE TABLE partition_datetime(
 dateTime1 datetime,
 some_data int
)
PARTITION BY RANGE ( dateTime1 ) (
 PARTITION p0 VALUES LESS THAN ('1990-01-01 00:00:01'),
 PARTITION p1 VALUES LESS THAN ('2000-01-01 00:00:01'),
 PARTITION p2 VALUES LESS THAN MAXVALUE
);
    )";

    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    auto tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 0 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 3 );

    auto &partitions = tableEntry->GetPartitions();
    auto partition = partitions[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p0" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "1990-01-01 00:00:01" );

    partition = partitions[ 1 ];
    ASSERT_EQ( partition->m_partitionName, "p1" );
    ASSERT_EQ( partition->m_partOrdPos, 2 );
    ASSERT_EQ( partition->m_partDesc, "2000-01-01 00:00:01" );

    partition = partitions[ 2 ];
    ASSERT_EQ( partition->m_partitionName, "p2" );
    ASSERT_EQ( partition->m_partOrdPos, 3 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_datetime.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    AriesInitialTable initTable( dbName, tableName );
    auto partitionMetaInfos = initTable.GetPartitionMetaInfo();
    auto partitionCount = partitionMetaInfos.size();
    ASSERT_EQ( partitionCount, 3 );

    auto partitionMetaInfo = partitionMetaInfos[ 0 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 2 ); // null is in this partition
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );

    partitionMetaInfo = partitionMetaInfos[ 1 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 9 );
#ifdef ARIES_INIT_TABLE_TEST
    ASSERT_EQ( partitionMetaInfo.BlockCount, 2 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 2 );
#else
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );
#endif

    partitionMetaInfo = partitionMetaInfos[ 2 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );

    uint32_t partitionIndex = 0;
    auto resultTable = initTable.GetPartitionData( { 1, 2 }, partitionIndex );
    ASSERT_EQ( resultTable->GetRowCount(), 2 );

    auto columnBuff = resultTable->GetColumnBuffer( 1 );
    auto columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->isDatetimeDataNull( 0 ), true );
    ASSERT_EQ( columnBuffSorted->GetNullableDatetimeAsString( 1 ), "1980-11-30 01:01:59" );

    columnBuff = resultTable->GetColumnBuffer( 2 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 0 ).value, 1 );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 1 ).value, 1980 );

    ++partitionIndex;
    resultTable = initTable.GetPartitionData( { 1, 2 }, partitionIndex );
    ASSERT_EQ( resultTable->GetRowCount(), 9 );

    columnBuff = resultTable->GetColumnBuffer( 1 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableDatetimeAsString( 0 ), "1996-01-13 01:01:59" );
    ASSERT_EQ( columnBuffSorted->GetNullableDatetimeAsString( 5 ), "1996-06-13 01:01:59" );
    ASSERT_EQ( columnBuffSorted->GetNullableDatetimeAsString( 8 ), "1996-09-13 01:01:59" );

    columnBuff = resultTable->GetColumnBuffer( 2 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 0 ).value, 1996 );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 5 ).value, 1996 );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 8 ).value, 1996 );

    ++partitionIndex;
    resultTable = initTable.GetPartitionData( { 1, 2 }, partitionIndex );
    ASSERT_EQ( resultTable->GetRowCount(), 1 );

    columnBuff = resultTable->GetColumnBuffer( 1 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableDatetimeAsString( 0 ), "2000-02-10 01:01:59" );

    columnBuff = resultTable->GetColumnBuffer( 2 );
    columnBuffSorted = aries_acc::SortData( columnBuff, AriesOrderByType::ASC );
    ASSERT_EQ( columnBuffSorted->GetNullableInt32( 0 ).value, 2000 );

    // with MAXVALUE partition ONLY
    tableName = "partition_datetime_maxvalue_only";
    InitTable( dbName, tableName );
    sql = R"(
CREATE TABLE partition_datetime_maxvalue_only(
 dateTime1 datetime,
 some_data int
)
PARTITION BY RANGE ( dateTime1 ) (
 PARTITION p2 VALUES LESS THAN MAXVALUE
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 0 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 1 );

    auto &partitions2 = tableEntry->GetPartitions();
    partition = partitions2[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p2" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    csvPath = cwd + "/test_resources/loaddata/csv/partition/partition_datetime.csv";
    sql = "load data infile '" +  csvPath + "' into table " + tableName + " fields terminated by ','";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    AriesInitialTable initTable2( dbName, tableName );
    partitionMetaInfos = initTable2.GetPartitionMetaInfo();
    partitionCount = partitionMetaInfos.size();
    ASSERT_EQ( partitionCount, 1 );

    partitionMetaInfo = partitionMetaInfos[ 0 ];
    ASSERT_EQ( partitionMetaInfo.RowCount, 12 );
    ASSERT_EQ( partitionMetaInfo.BlockCount, 1 );
    ASSERT_EQ( partitionMetaInfo.BlocksID.size(), 1 );
}