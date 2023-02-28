#include <gtest/gtest.h>

#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/AriesConstantNode.h"
#include "AriesEngine/AriesInsertNode.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "frontend/SQLExecutor.h"
#include "../../../TestUtils.h"
#include "utils/string_util.h"

using namespace aries_engine;
using namespace aries;
using namespace std;

static string cwd = aries_utils::get_current_work_directory();

namespace aries_test
{

ARIES_UNIT_TEST_CLASS( TestAriesXLogRecoveryer )
{
protected:
    std::string db_name = "TestAriesXLogRecoveryer";
    std::string table_name = "test";
    std::vector< TxId > transactions;

protected:
    void SetUp() override
    {
        aries_utils::to_lower( db_name );
        SQLExecutor::GetInstance()->ExecuteSQL( "CREATE database IF NOT EXISTS " + db_name + ";", std::string() );
        SQLExecutor::GetInstance()->ExecuteSQL( "CREATE table IF NOT EXISTS " + table_name + "(id int, name char(64));", db_name );

        prepare();
    }

    void TearDown() override
    {
        SQLExecutor::GetInstance()->ExecuteSQL( "DROP table test;", db_name );
        SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + db_name + ";", std::string() );
    }

    void prepare()
    {
        auto trans = AriesTransManager::GetInstance().NewTransaction();
        auto trans2 = AriesTransManager::GetInstance().NewTransaction();

        std::vector< std::vector< AriesCommonExprUPtr > > data;

        auto expr_id = CreateConstantExpression( int32_t( 1 ) );
        auto expr_name = CreateConstantExpression( std::string( "my name" ) );

        std::vector< AriesCommonExprUPtr > row;
        row.emplace_back( std::move( expr_id ) );
        row.emplace_back( std::move( expr_name ) );

        data.emplace_back( std::move( row ) );

        expr_id = CreateConstantExpression( int32_t( 2 ) );
        expr_name = CreateConstantExpression( std::string( "your name" ) );

        row.clear();
        row.emplace_back( std::move( expr_id ) );
        row.emplace_back( std::move( expr_name ) );
        data.emplace_back( std::move( row ) );

        doInsert( trans, data );

        data.clear();
        expr_id = CreateConstantExpression( int32_t( 3 ) );
        expr_name = CreateConstantExpression( std::string( "his name" ) );

        row.clear();
        row.emplace_back( std::move( expr_id ) );
        row.emplace_back( std::move( expr_name ) );

        data.emplace_back( std::move( row ) );

        expr_id = CreateConstantExpression( int32_t( 4 ) );
        expr_name = CreateConstantExpression( std::string( "her name" ) );

        row.clear();
        row.emplace_back( std::move( expr_id ) );
        row.emplace_back( std::move( expr_name ) );
        data.emplace_back( std::move( row ) );

        doInsert( trans2, data );

        AriesTransManager::GetInstance().EndTransaction( trans, TransactionStatus::COMMITTED );
        AriesTransManager::GetInstance().EndTransaction( trans2, TransactionStatus::COMMITTED );
    }

    void doInsert( AriesTransactionPtr transaction, const std::vector< std::vector< AriesCommonExprUPtr > >& data )
    {
        auto insert_node = std::make_shared< AriesInsertNode >( transaction, db_name, table_name );

        std::vector< int > column_ids;
        column_ids.emplace_back( 1 );
        column_ids.emplace_back( 2 );
        insert_node->SetColumnIds( column_ids );

        auto node = std::make_shared< AriesConstantNode >( db_name, table_name );
        string errorMsg;
        node->SetColumnData( data, column_ids, errorMsg );
        insert_node->SetSourceNode( node );

        ASSERT_TRUE( insert_node->Open() );

        auto table = AriesMvccTableManager::GetInstance().getMvccTable( db_name, table_name );

        auto result = insert_node->GetNext();
        ASSERT_TRUE( result.Status == AriesOpNodeStatus::END );
        ASSERT_TRUE( result.TableBlock );
        ASSERT_EQ( result.TableBlock->GetColumnBuffer( 1 )->GetInt64( 0 ), data.size() );
    }

};

ARIES_UNIT_TEST_F( TestAriesXLogRecoveryer, run )
{
    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    auto result = SQLExecutor::GetInstance()->ExecuteSQL( "select * from " + table_name + " order by id;", db_name );
    ASSERT_TRUE( result->IsSuccess() );

    ASSERT_TRUE( result->GetResults().size() == 1 );

    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
    auto table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 4 );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "1" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 1 ), "2" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 2 ), "3" );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 3 ), "4" );

    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetString( 0 ), "my name" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetString( 1 ), "your name" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetString( 2 ), "his name" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetString( 3 ), "her name" );
}

string ORDERS_UPDATE_TABLE_NAME( "orders_u1" );
string LINEITEM_UPDATE_TABLE_NAME( "lineitem_u1" );
string ORDERS_UPDATE_FILE_NAME( "orders.tbl.u1" );
string LINEITEM_UPDATE_FILE_NAME( "lineitem.tbl.u1" );

string ordersDictName = "TestAriesXLogRecoveryer_orders";
string SQL_CREATE_TABLE_ORDERS( "create table orders "
                       "( o_orderkey       integer not null primary key,"
                       "o_custkey        integer not null,"
                       "o_orderstatus    char(1) not null,"
                       "o_totalprice     decimal(15,2) not null,"
                       "o_orderdate      date not null,"
                       "o_orderpriority  char(15) not null encoding bytedict as " + ordersDictName + ", "
                       "o_clerk          char(15) not null,"
                       "o_shippriority   integer not null,"
                       "o_comment        varchar(79) not null );"
                       );

string lineitemShipinstructDict = "TestAriesXLogRecoveryer_lineitem_l_shipinstruct";
string lineitemShipmodeDict = "TestAriesXLogRecoveryer_lineitem_l_shipinmode";
string SQL_CREATE_TABLE_LINEITEM( "create table lineitem "
                      "( l_orderkey    integer not null,"
                        "l_partkey     integer not null,"
                        "l_suppkey     integer not null,"
                        "l_linenumber  integer not null,"
                        "l_quantity    decimal(15,2) not null,"
                        "l_extendedprice  decimal(15,2) not null,"
                        "l_discount    decimal(15,2) not null,"
                        "l_tax         decimal(15,2) not null,"
                        "l_returnflag  char(1) not null,"
                        "l_linestatus  char(1) not null,"
                        "l_shipdate    date not null,"
                        "l_commitdate  date not null,"
                        "l_receiptdate date not null,"
                        "l_shipinstruct char(25) not null encoding bytedict as " + lineitemShipinstructDict + ","
                        "l_shipmode     char(10) not null encoding bytedict as " + lineitemShipmodeDict + ","
                        "l_comment      varchar(44) not null,"
                        "primary key ( l_orderkey, l_linenumber ) ); "
                        );

void PrepareInitData( const string& testDbName )
{
    string sql = "drop table if exists lineitem";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists orders";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    result = SQLExecutor::GetInstance()->ExecuteSQL( SQL_CREATE_TABLE_ORDERS, testDbName );
    ASSERT_TRUE( result->IsSuccess() );
    string csvPath = cwd + "/test_resources/xlog/orders.tbl";
    sql = "load data infile '" + csvPath + "' into table orders fields terminated by '|';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    result = SQLExecutor::GetInstance()->ExecuteSQL( SQL_CREATE_TABLE_LINEITEM, testDbName );
    ASSERT_TRUE( result->IsSuccess() );
    csvPath = cwd + "/test_resources/xlog/lineitem.tbl";
    sql = "load data infile '" + csvPath + "' into table lineitem fields terminated by '|';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );
}
void PrepareUpdateData( const string& testDbName )
{
    string sql( "drop database if exists " + testDbName );
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create database " + testDbName;
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + LINEITEM_UPDATE_TABLE_NAME;
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + ORDERS_UPDATE_TABLE_NAME;
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    PrepareInitData( testDbName );

    string tableSchema( "( o_orderkey    integer not null primary key,"
                       "o_custkey        integer not null,"
                       "o_orderstatus    char(1) not null,"
                       "o_totalprice     decimal(15,2) not null,"
                       "o_orderdate      date not null,"
                       "o_orderpriority  char(15) not null encoding bytedict as " + ordersDictName + ", "
                       "o_clerk          char(15) not null,"
                       "o_shippriority   integer not null,"
                       "o_comment        varchar(79) not null );" );
    sql = "create table " + ORDERS_UPDATE_TABLE_NAME + tableSchema;
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );
    string csvPath = cwd + "/test_resources/xlog/" + ORDERS_UPDATE_FILE_NAME;
    sql = "load data infile '" + csvPath + "' into table " + ORDERS_UPDATE_TABLE_NAME + " fields terminated by '|';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    tableSchema = "(l_orderkey    integer not null,"
                 "l_partkey     integer not null,"
                 "l_suppkey     integer not null,"
                 "l_linenumber  integer not null,"
                 "l_quantity    decimal(15,2) not null,"
                 "l_extendedprice  decimal(15,2) not null,"
                 "l_discount    decimal(15,2) not null,"
                 "l_tax         decimal(15,2) not null,"
                 "l_returnflag  char(1) not null,"
                 "l_linestatus  char(1) not null,"
                 "l_shipdate    date not null,"
                 "l_commitdate  date not null,"
                 "l_receiptdate date not null,"
                 "l_shipinstruct char(25) not null encoding bytedict as " + lineitemShipinstructDict + ","
                 "l_shipmode     char(10) not null encoding bytedict as " + lineitemShipmodeDict + ","
                 "l_comment      varchar(44) not null,"
                 "primary key ( l_orderkey, l_linenumber ) ); ";
    sql = "create table " + LINEITEM_UPDATE_TABLE_NAME + tableSchema;
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );
    csvPath = cwd + "/test_resources/xlog/" + LINEITEM_UPDATE_FILE_NAME;
    sql = "load data infile '" + csvPath + "' into table " + LINEITEM_UPDATE_TABLE_NAME + " fields terminated by '|';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists delete_orderkeys";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create table delete_orderkeys( orderkey int not null primary key )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );

    csvPath = cwd + "/test_resources/xlog/delete.1";
    sql = "load data infile '" + csvPath + "' into table delete_orderkeys fields terminated by '|';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName );
    ASSERT_TRUE( result->IsSuccess() );
}

void BatchInsert( const string& test_db_name )
{
    string sql = R"(
insert into 
orders 
select 
o_orderkey,  
o_custkey, 
o_orderstatus, 
o_totalprice, 
o_orderdate, 
dict_index( o_orderpriority ), 
o_clerk, 
o_shippriority, 
o_comment 
from 
    )";
    sql.append( " " ).append( ORDERS_UPDATE_TABLE_NAME );
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );

    sql = R"(
insert into 
lineitem 
select 
l_orderkey,  
l_partkey, 
l_suppkey, 
l_linenumber, 
l_quantity, 
l_extendedprice, 
l_discount, 
l_tax, 
l_returnflag, 
l_linestatus, 
l_shipdate, 
l_commitdate, 
l_receiptdate, 
dict_index( l_shipinstruct ), 
dict_index( l_shipmode ), 
l_comment 
from 
    )";
    sql.append( " " ).append( LINEITEM_UPDATE_TABLE_NAME );
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
}

void BatchDelete( const string& test_db_name )
{
    string sql( R"(
delete from 
lineitem 
where 
l_orderkey 
in ( 
select 
orderkey 
from 
    )" );
    sql.append( " delete_orderkeys )" );
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );

    sql = R"(
delete from 
orders 
where 
o_orderkey 
in ( 
select 
orderkey 
from 
    )";
    sql.append( " delete_orderkeys )" );
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
}

// test batch insert
ARIES_UNIT_TEST_F( TestAriesXLogRecoveryer, batch_insert )
{
    string test_db_name = "testariesxlogrecoveryer_test0";
    PrepareUpdateData( test_db_name );

    BatchInsert( test_db_name );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( test_db_name, "orders" );
    auto totalRowCount = initTable->GetTotalRowCount();
    auto capacity = initTable->GetCapacity();
    auto slotBitmaps = initTable->GetBlockBitmaps();
    ASSERT_EQ( capacity, ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( totalRowCount, 20 );
    size_t i = 0;
    for ( i = 0; i < totalRowCount; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, true );
    }
    for ( ; i < capacity; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, false );
    }

    initTable = AriesInitialTableManager::GetInstance().getTable( test_db_name, "lineitem" );
    totalRowCount = initTable->GetTotalRowCount();
    capacity = initTable->GetCapacity();
    slotBitmaps = initTable->GetBlockBitmaps();
    ASSERT_EQ( capacity, ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( totalRowCount, 78 );
    for ( i = 0; i < totalRowCount; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, true );
    }
    for ( ; i < capacity; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, false );
    }

    string sql = "select * from orders";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 20 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 5 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 6 ), 7 );
    ASSERT_EQ( columnBuff->GetInt32( 7 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 8 ), 33 );
    ASSERT_EQ( columnBuff->GetInt32( 9 ), 34 );

    ASSERT_EQ( columnBuff->GetInt32( 10 ), 9 );
    ASSERT_EQ( columnBuff->GetInt32( 11 ), 10 );
    ASSERT_EQ( columnBuff->GetInt32( 12 ), 11 );
    ASSERT_EQ( columnBuff->GetInt32( 13 ), 12 );
    ASSERT_EQ( columnBuff->GetInt32( 14 ), 13 );
    ASSERT_EQ( columnBuff->GetInt32( 15 ), 14 );
    ASSERT_EQ( columnBuff->GetInt32( 16 ), 15 );
    ASSERT_EQ( columnBuff->GetInt32( 17 ), 40 );
    ASSERT_EQ( columnBuff->GetInt32( 18 ), 41 );
    ASSERT_EQ( columnBuff->GetInt32( 19 ), 42 );

    columnBuff = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 4 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 5 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 6 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 7 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 8 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 9 ), "O" );

    ASSERT_EQ( columnBuff->GetString( 10 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 11 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 12 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 13 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 14 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 15 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 16 ), "P" );
    ASSERT_EQ( columnBuff->GetString( 17 ), "O" );
    ASSERT_EQ( columnBuff->GetString( 18 ), "F" );
    ASSERT_EQ( columnBuff->GetString( 19 ), "F" );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), aries_acc::Decimal( "173665.47" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 1 ), aries_acc::Decimal( "46929.18" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 2 ), aries_acc::Decimal( "193846.25" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 3 ), aries_acc::Decimal( "32151.78" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 4 ), aries_acc::Decimal( "144659.20" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 5 ), aries_acc::Decimal( "58749.59" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 6 ), aries_acc::Decimal( "252004.18" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 7 ), aries_acc::Decimal( "208660.75" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 8 ), aries_acc::Decimal( "163243.98" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 9 ), aries_acc::Decimal( "58949.67" ) );

    ASSERT_EQ( columnBuff->GetDecimal( 10 ), aries_acc::Decimal( "182274.73" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 11 ), aries_acc::Decimal( "140081.83" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 12 ), aries_acc::Decimal( "97148.59" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 13 ), aries_acc::Decimal( "165157.10" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 14 ), aries_acc::Decimal( "106083.78" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 15 ), aries_acc::Decimal( "114019.79" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 16 ), aries_acc::Decimal( "95707.32" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 17 ), aries_acc::Decimal( "224722.49" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 18 ), aries_acc::Decimal( "49029.17" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 19 ), aries_acc::Decimal( "216628.37" ) );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), "1996-01-02" );
    ASSERT_EQ( columnBuff->GetDateAsString( 1 ), "1996-12-01" );
    ASSERT_EQ( columnBuff->GetDateAsString( 2 ), "1993-10-14" );
    ASSERT_EQ( columnBuff->GetDateAsString( 3 ), "1995-10-11" );
    ASSERT_EQ( columnBuff->GetDateAsString( 4 ), "1994-07-30" );
    ASSERT_EQ( columnBuff->GetDateAsString( 5 ), "1992-02-21" );
    ASSERT_EQ( columnBuff->GetDateAsString( 6 ), "1996-01-10" );
    ASSERT_EQ( columnBuff->GetDateAsString( 7 ), "1995-07-16" );
    ASSERT_EQ( columnBuff->GetDateAsString( 8 ), "1993-10-27" );
    ASSERT_EQ( columnBuff->GetDateAsString( 9 ), "1998-07-21" );

    ASSERT_EQ( columnBuff->GetDateAsString( 10 ), "1992-11-07" );
    ASSERT_EQ( columnBuff->GetDateAsString( 11 ), "1992-06-12" );
    ASSERT_EQ( columnBuff->GetDateAsString( 12 ), "1997-02-05" );
    ASSERT_EQ( columnBuff->GetDateAsString( 13 ), "1992-08-01" );
    ASSERT_EQ( columnBuff->GetDateAsString( 14 ), "1993-06-11" );
    ASSERT_EQ( columnBuff->GetDateAsString( 15 ), "1994-04-09" );
    ASSERT_EQ( columnBuff->GetDateAsString( 16 ), "1995-03-22" );
    ASSERT_EQ( columnBuff->GetDateAsString( 17 ), "1998-04-23" );
    ASSERT_EQ( columnBuff->GetDateAsString( 18 ), "1993-04-26" );
    ASSERT_EQ( columnBuff->GetDateAsString( 19 ), "1993-02-18" );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 4 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 6 ), "2-HIGH" );
    ASSERT_EQ( columnBuff->GetString( 8 ), "3-MEDIUM" );

    ASSERT_EQ( columnBuff->GetString( 10 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 11 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 12 ), "2-HIGH" );
    ASSERT_EQ( columnBuff->GetString( 13 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 14 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 15 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 16 ), "3-MEDIUM" );
    ASSERT_EQ( columnBuff->GetString( 17 ), "3-MEDIUM" );
    ASSERT_EQ( columnBuff->GetString( 18 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 19 ), "3-MEDIUM" );

    sql = "select * from lineitem";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 78 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 5 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 6 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 9 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 12 ), 3 );
    ASSERT_EQ( columnBuff->GetInt32( 19 ), 7 );
    ASSERT_EQ( columnBuff->GetInt32( 29 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 37 ), 34 );

    ASSERT_EQ( columnBuff->GetInt32( 38 ), 9 );
    ASSERT_EQ( columnBuff->GetInt32( 41 ), 9 );
    ASSERT_EQ( columnBuff->GetInt32( 44 ), 9 );
    ASSERT_EQ( columnBuff->GetInt32( 46 ), 10 );
    ASSERT_EQ( columnBuff->GetInt32( 50 ), 11 );
    ASSERT_EQ( columnBuff->GetInt32( 53 ), 12 );
    ASSERT_EQ( columnBuff->GetInt32( 55 ), 13 );
    ASSERT_EQ( columnBuff->GetInt32( 60 ), 14 );
    ASSERT_EQ( columnBuff->GetInt32( 65 ), 15 );
    ASSERT_EQ( columnBuff->GetInt32( 70 ), 40 );
    ASSERT_EQ( columnBuff->GetInt32( 75 ), 42 );
    ASSERT_EQ( columnBuff->GetInt32( 77 ), 42 );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), aries_acc::Decimal( "17" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 3 ), aries_acc::Decimal( "28" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 5 ), aries_acc::Decimal( "32" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 6 ), aries_acc::Decimal( "38" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 9 ), aries_acc::Decimal( "27" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 12 ), aries_acc::Decimal( "26" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 19 ), aries_acc::Decimal( "9" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 29 ), aries_acc::Decimal( "44" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 37 ), aries_acc::Decimal( "6" ) );

    ASSERT_EQ( columnBuff->GetDecimal( 38 ), aries_acc::Decimal( "14" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 41 ), aries_acc::Decimal( "32" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 44 ), aries_acc::Decimal( "31" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 46 ), aries_acc::Decimal( "28" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 50 ), aries_acc::Decimal( "3" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 53 ), aries_acc::Decimal( "23" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 55 ), aries_acc::Decimal( "3" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 60 ), aries_acc::Decimal( "5" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 65 ), aries_acc::Decimal( "7" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 70 ), aries_acc::Decimal( "6" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 75 ), aries_acc::Decimal( "38" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 77 ), aries_acc::Decimal( "27" ) );

    columnBuff = resTable->GetColumnBuffer( 11 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), "1996-03-13" );
    ASSERT_EQ( columnBuff->GetDateAsString( 3 ), "1996-04-21");
    ASSERT_EQ( columnBuff->GetDateAsString( 5 ), "1996-01-30");
    ASSERT_EQ( columnBuff->GetDateAsString( 6 ), "1997-01-28");
    ASSERT_EQ( columnBuff->GetDateAsString( 9 ), "1994-01-16");
    ASSERT_EQ( columnBuff->GetDateAsString( 12 ), "1993-10-29");
    ASSERT_EQ( columnBuff->GetDateAsString( 19 ), "1996-02-01");
    ASSERT_EQ( columnBuff->GetDateAsString( 29 ), "1995-08-28");
    ASSERT_EQ( columnBuff->GetDateAsString( 37 ), "1998-10-30");

    ASSERT_EQ( columnBuff->GetDateAsString( 38 ), "1992-11-14" );
    ASSERT_EQ( columnBuff->GetDateAsString( 41 ), "1992-12-22");
    ASSERT_EQ( columnBuff->GetDateAsString( 44 ), "1993-01-09");
    ASSERT_EQ( columnBuff->GetDateAsString( 46 ), "1992-07-17");
    ASSERT_EQ( columnBuff->GetDateAsString( 50 ), "1997-03-28");
    ASSERT_EQ( columnBuff->GetDateAsString( 53 ), "1992-11-01");
    ASSERT_EQ( columnBuff->GetDateAsString( 55 ), "1993-06-16");
    ASSERT_EQ( columnBuff->GetDateAsString( 60 ), "1994-06-23");
    ASSERT_EQ( columnBuff->GetDateAsString( 65 ), "1995-05-25");
    ASSERT_EQ( columnBuff->GetDateAsString( 70 ), "1998-06-14");
    ASSERT_EQ( columnBuff->GetDateAsString( 75 ), "1993-03-25");
    ASSERT_EQ( columnBuff->GetDateAsString( 77 ), "1993-03-30");

    columnBuff = resTable->GetColumnBuffer( 14 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "DELIVER IN PERSON" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "NONE");
    ASSERT_EQ( columnBuff->GetString( 5 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 6 ), "TAKE BACK RETURN");
    ASSERT_EQ( columnBuff->GetString( 9 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 12 ), "TAKE BACK RETURN");
    ASSERT_EQ( columnBuff->GetString( 19 ), "TAKE BACK RETURN");
    ASSERT_EQ( columnBuff->GetString( 29 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 37 ), "NONE");

    ASSERT_EQ( columnBuff->GetString( 38 ), "COLLECT COD" );
    ASSERT_EQ( columnBuff->GetString( 41 ), "COLLECT COD");
    ASSERT_EQ( columnBuff->GetString( 44 ), "NONE");
    ASSERT_EQ( columnBuff->GetString( 46 ), "NONE");
    ASSERT_EQ( columnBuff->GetString( 50 ), "COLLECT COD");
    ASSERT_EQ( columnBuff->GetString( 53 ), "TAKE BACK RETURN");
    ASSERT_EQ( columnBuff->GetString( 55 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 60 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 65 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 70 ), "DELIVER IN PERSON");
    ASSERT_EQ( columnBuff->GetString( 75 ), "TAKE BACK RETURN");
    ASSERT_EQ( columnBuff->GetString( 77 ), "DELIVER IN PERSON");

}

// test batch delete
ARIES_UNIT_TEST_F( TestAriesXLogRecoveryer, batch_delete )
{
    string test_db_name = "testariesxlogrecoveryer_test1";
    PrepareUpdateData( test_db_name );

    BatchDelete( test_db_name );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( test_db_name, "orders" );
    auto totalRowCount = initTable->GetTotalRowCount();
    auto capacity = initTable->GetCapacity();
    auto slotBitmaps = initTable->GetBlockBitmaps();
    ASSERT_EQ( capacity, ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( totalRowCount, 5 );
    size_t i = 0;
    for ( i = 0; i < totalRowCount; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, true );
    }
    for ( ; i < capacity; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, false );
    }

    initTable = AriesInitialTableManager::GetInstance().getTable( test_db_name, "lineitem" );
    totalRowCount = initTable->GetTotalRowCount();
    capacity = initTable->GetCapacity();
    slotBitmaps = initTable->GetBlockBitmaps();
    ASSERT_EQ( capacity, ARIES_BLOCK_FILE_ROW_COUNT );
    auto itemCount = 38 - 6 - 6 - 3 - 7 - 4;
    ASSERT_EQ( totalRowCount, itemCount );
    for ( i = 0; i < totalRowCount; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, true );
    }
    for ( ; i < capacity; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, false );
    }

    string sql = "select * from orders order by o_orderkey";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();

    /**
     * mysql> select * from orders order by o_orderkey;
+------------+-----------+---------------+--------------+-------------+-----------------+-----------------+----------------+------------------------------------------------------------------------+
| o_orderkey | o_custkey | o_orderstatus | o_totalprice | o_orderdate | o_orderpriority | o_clerk         | o_shippriority | o_comment                                                              |
+------------+-----------+---------------+--------------+-------------+-----------------+-----------------+----------------+------------------------------------------------------------------------+
|          2 |     78002 | O             |     46929.18 | 1996-12-01  | 1-URGENT        | Clerk#000000880 |              0 |  foxes. pending accounts at the pending, silent asymptot               |
|          4 |    136777 | O             |     32151.78 | 1995-10-11  | 5-LOW           | Clerk#000000124 |              0 | sits. slyly regular warthogs cajole. regular, regular theodolites acro |
|          6 |     55624 | F             |     58749.59 | 1992-02-21  | 4-NOT SPECIFIED | Clerk#000000058 |              0 | ggle. special, final requests are against the furiously specia         |
|         32 |    130057 | O             |    208660.75 | 1995-07-16  | 2-HIGH          | Clerk#000000616 |              0 | ise blithely bold, regular requests. quickly unusual dep               |
|         34 |     61001 | O             |     58949.67 | 1998-07-21  | 3-MEDIUM        | Clerk#000000223 |              0 | ly final packages. fluffily final deposits wake blithely ideas. spe    |
+------------+-----------+---------------+--------------+-------------+-----------------+-----------------+----------------+------------------------------------------------------------------------+
5 rows in set (0.01 sec)
     */

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( resTable->GetRowCount(), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 34 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), aries_acc::Decimal( "46929.18" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 1 ), aries_acc::Decimal( "32151.78" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 2 ), aries_acc::Decimal( "58749.59" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 3 ), aries_acc::Decimal( "208660.75" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 4 ), aries_acc::Decimal( "58949.67" ) );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), "1996-12-01" );
    ASSERT_EQ( columnBuff->GetDateAsString( 1 ), "1995-10-11" );
    ASSERT_EQ( columnBuff->GetDateAsString( 2 ), "1992-02-21" );
    ASSERT_EQ( columnBuff->GetDateAsString( 3 ), "1995-07-16" );
    ASSERT_EQ( columnBuff->GetDateAsString( 4 ), "1998-07-21" );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "4-NOT SPECIFIED" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "2-HIGH" );
    ASSERT_EQ( columnBuff->GetString( 4 ), "3-MEDIUM" );

/*
mysql> select * from lineitem order by l_orderkey;
+------------+-----------+-----------+--------------+------------+-----------------+------------+-------+--------------+--------------+------------+--------------+---------------+-------------------+------------+--------------------------------------------+
| l_orderkey | l_partkey | l_suppkey | l_linenumber | l_quantity | l_extendedprice | l_discount | l_tax | l_returnflag | l_linestatus | l_shipdate | l_commitdate | l_receiptdate | l_shipinstruct    | l_shipmode | l_comment                                  |
+------------+-----------+-----------+--------------+------------+-----------------+------------+-------+--------------+--------------+------------+--------------+---------------+-------------------+------------+--------------------------------------------+
|          2 |    106170 |      1191 |            1 |      38.00 |        44694.46 |       0.00 |  0.05 | N            | O            | 1997-01-28 | 1997-01-14   | 1997-02-02    | TAKE BACK RETURN  | RAIL       | ven requests. deposits breach a            |
|          4 |     88035 |      5560 |            1 |      30.00 |        30690.90 |       0.03 |  0.08 | N            | O            | 1996-01-10 | 1995-12-14   | 1996-01-18    | DELIVER IN PERSON | REG AIR    | - quickly regular packages sleep. idly     |
|          6 |    139636 |      2150 |            1 |      37.00 |        61998.31 |       0.08 |  0.03 | A            | F            | 1992-04-27 | 1992-05-15   | 1992-05-02    | TAKE BACK RETURN  | TRUCK      | p furiously special foxes                  |
|         32 |     11615 |      4117 |            6 |       6.00 |         9159.66 |       0.04 |  0.03 | N            | O            | 1995-07-21 | 1995-09-23   | 1995-07-25    | COLLECT COD       | RAIL       |  gifts cajole carefully.                   |
|         32 |     85811 |      8320 |            5 |      44.00 |        79059.64 |       0.05 |  0.06 | N            | O            | 1995-08-28 | 1995-08-20   | 1995-09-14    | DELIVER IN PERSON | AIR        | symptotes nag according to the ironic depo |
|         32 |      2743 |      7744 |            4 |       4.00 |         6582.96 |       0.09 |  0.03 | N            | O            | 1995-08-04 | 1995-10-01   | 1995-09-03    | NONE              | REG AIR    | e slyly final pac                          |
|         32 |     44161 |      6666 |            3 |       2.00 |         2210.32 |       0.09 |  0.02 | N            | O            | 1995-08-07 | 1995-10-07   | 1995-08-23    | DELIVER IN PERSON | AIR        |  express accounts wake according to the    |
|         32 |    197921 |       441 |            2 |      32.00 |        64605.44 |       0.02 |  0.00 | N            | O            | 1995-08-14 | 1995-10-07   | 1995-08-27    | COLLECT COD       | AIR        | lithely regular deposits. fluffily         |
|         32 |     82704 |      7721 |            1 |      28.00 |        47227.60 |       0.05 |  0.08 | N            | O            | 1995-10-23 | 1995-08-27   | 1995-10-26    | TAKE BACK RETURN  | TRUCK      | sleep quickly. req                         |
|         34 |    169544 |      4577 |            3 |       6.00 |         9681.24 |       0.02 |  0.06 | N            | O            | 1998-10-30 | 1998-09-20   | 1998-11-05    | NONE              | FOB        | ar foxes sleep                             |
|         34 |     89414 |      1923 |            2 |      22.00 |        30875.02 |       0.08 |  0.06 | N            | O            | 1998-10-09 | 1998-10-16   | 1998-10-12    | NONE              | FOB        | thely slyly p                              |
|         34 |     88362 |       871 |            1 |      13.00 |        17554.68 |       0.00 |  0.07 | N            | O            | 1998-10-23 | 1998-09-14   | 1998-11-06    | NONE              | REG AIR    | nic accounts. deposits are alon            |
+------------+-----------+-----------+--------------+------------+-----------------+------------+-------+--------------+--------------+------------+--------------+---------------+-------------------+------------+--------------------------------------------+
12 rows in set (0.04 sec)
*/
    sql = "select * from lineitem order by l_orderkey";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), itemCount );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 5 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 6 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 7 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 8 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 9 ), 34 );
    ASSERT_EQ( columnBuff->GetInt32( 10 ), 34 );
    ASSERT_EQ( columnBuff->GetInt32( 11 ), 34 );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), aries_acc::Decimal( "44694.46") );
    ASSERT_EQ( columnBuff->GetDecimal( 1 ), aries_acc::Decimal( "30690.90") );
    ASSERT_EQ( columnBuff->GetDecimal( 2 ), aries_acc::Decimal( "61998.31") );
    ASSERT_EQ( columnBuff->GetDecimal( 3 ), aries_acc::Decimal( "9159.66") );
    ASSERT_EQ( columnBuff->GetDecimal( 4 ), aries_acc::Decimal( "79059.64") );
    ASSERT_EQ( columnBuff->GetDecimal( 5 ), aries_acc::Decimal( "6582.96") );
    ASSERT_EQ( columnBuff->GetDecimal( 6 ), aries_acc::Decimal( "2210.32") );
    ASSERT_EQ( columnBuff->GetDecimal( 7 ), aries_acc::Decimal( "64605.44") );
    ASSERT_EQ( columnBuff->GetDecimal( 8 ), aries_acc::Decimal( "47227.60") );
    ASSERT_EQ( columnBuff->GetDecimal( 9 ), aries_acc::Decimal( "9681.24") );
    ASSERT_EQ( columnBuff->GetDecimal( 10 ), aries_acc::Decimal( "30875.02") );
    ASSERT_EQ( columnBuff->GetDecimal( 11 ), aries_acc::Decimal( "17554.68") );

    columnBuff = resTable->GetColumnBuffer( 12 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), "1997-01-14" );
    ASSERT_EQ( columnBuff->GetDateAsString( 1 ), "1995-12-14" );
    ASSERT_EQ( columnBuff->GetDateAsString( 3 ), "1995-09-23" );
    ASSERT_EQ( columnBuff->GetDateAsString( 5 ), "1995-10-01" );
    ASSERT_EQ( columnBuff->GetDateAsString( 6 ), "1995-10-07" );
    ASSERT_EQ( columnBuff->GetDateAsString( 8 ), "1995-08-27");
    ASSERT_EQ( columnBuff->GetDateAsString( 10 ), "1998-10-16");
    ASSERT_EQ( columnBuff->GetDateAsString( 11 ), "1998-09-14");

    columnBuff = resTable->GetColumnBuffer( 15 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "RAIL" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "TRUCK" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "RAIL" );
    ASSERT_EQ( columnBuff->GetString( 4 ), "AIR" );
    ASSERT_EQ( columnBuff->GetString( 5 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 6 ), "AIR" );
    ASSERT_EQ( columnBuff->GetString( 7 ), "AIR" );
    ASSERT_EQ( columnBuff->GetString( 8 ), "TRUCK" );
    ASSERT_EQ( columnBuff->GetString( 9 ), "FOB" );
    ASSERT_EQ( columnBuff->GetString( 10 ), "FOB" );
    ASSERT_EQ( columnBuff->GetString( 11 ), "REG AIR" );
}

// test batch insert and delete
ARIES_UNIT_TEST_F( TestAriesXLogRecoveryer, batch_insert_and_delete )
{
    string test_db_name = "testariesxlogrecoveryer_test1";
    PrepareUpdateData( test_db_name );

    BatchInsert( test_db_name );
    BatchDelete( test_db_name );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( test_db_name, "orders" );
    auto totalRowCount = initTable->GetTotalRowCount();
    auto capacity = initTable->GetCapacity();
    auto slotBitmaps = initTable->GetBlockBitmaps();
    ASSERT_EQ( capacity, ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( totalRowCount, 10 );
    size_t i = 0;
    for ( i = 0; i < totalRowCount; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, true );
    }
    for ( ; i < capacity; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, false );
    }

    initTable = AriesInitialTableManager::GetInstance().getTable( test_db_name, "lineitem" );
    totalRowCount = initTable->GetTotalRowCount();
    capacity = initTable->GetCapacity();
    slotBitmaps = initTable->GetBlockBitmaps();
    ASSERT_EQ( capacity, ARIES_BLOCK_FILE_ROW_COUNT );
    auto itemCount = 78 - 43;
    ASSERT_EQ( totalRowCount, itemCount );
    for ( i = 0; i < totalRowCount; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, true );
    }
    for ( ; i < capacity; ++i )
    {
        bool b;
        GET_BIT_FLAG( slotBitmaps[ 0 ], i, b );
        ASSERT_EQ( b, false );
    }

    string sql = "select * from orders order by o_orderkey";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();

    /**
     * mysql> select * from orders order by o_orderkey;
+------------+-----------+---------------+--------------+-------------+-----------------+-----------------+----------------+------------------------------------------------------------------------+
| o_orderkey | o_custkey | o_orderstatus | o_totalprice | o_orderdate | o_orderpriority | o_clerk         | o_shippriority | o_comment                                                              |
+------------+-----------+---------------+--------------+-------------+-----------------+-----------------+----------------+------------------------------------------------------------------------+
|          2 |     78002 | O             |     46929.18 | 1996-12-01  | 1-URGENT        | Clerk#000000880 |              0 |  foxes. pending accounts at the pending, silent asymptot               |
|          4 |    136777 | O             |     32151.78 | 1995-10-11  | 5-LOW           | Clerk#000000124 |              0 | sits. slyly regular warthogs cajole. regular, regular theodolites acro |
|          6 |     55624 | F             |     58749.59 | 1992-02-21  | 4-NOT SPECIFIED | Clerk#000000058 |              0 | ggle. special, final requests are against the furiously specia         |
|         10 |   1776871 | F             |    140081.83 | 1992-06-12  | 1-URGENT        | Clerk#000032123 |              0 | uickly special requests doubt blithely furio                           |
|         12 |   9650911 | F             |    165157.10 | 1992-08-01  | 5-LOW           | Clerk#000093125 |              0 |  against the slyly special deposits. furiously special                 |
|         14 |   6135926 | F             |    114019.79 | 1994-04-09  | 1-URGENT        | Clerk#000099045 |              0 | ly regular instructions boost carefully. instructions according to the |
|         32 |    130057 | O             |    208660.75 | 1995-07-16  | 2-HIGH          | Clerk#000000616 |              0 | ise blithely bold, regular requests. quickly unusual dep               |
|         34 |     61001 | O             |     58949.67 | 1998-07-21  | 3-MEDIUM        | Clerk#000000223 |              0 | ly final packages. fluffily final deposits wake blithely ideas. spe    |
|         40 |   7763176 | O             |    224722.49 | 1998-04-23  | 3-MEDIUM        | Clerk#000070079 |              0 | es. blithely silent ideas                                              |
|         42 |   8250538 | F             |    216628.37 | 1993-02-18  | 3-MEDIUM        | Clerk#000018100 |              0 |  accounts boost except the quickly                                     |
+------------+-----------+---------------+--------------+-------------+-----------------+-----------------+----------------+------------------------------------------------------------------------+

     */

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( resTable->GetRowCount(), 10 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 10 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 12 );
    ASSERT_EQ( columnBuff->GetInt32( 5 ), 14 );
    ASSERT_EQ( columnBuff->GetInt32( 6 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 7 ), 34 );
    ASSERT_EQ( columnBuff->GetInt32( 8 ), 40 );
    ASSERT_EQ( columnBuff->GetInt32( 9 ), 42 );

    columnBuff = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), aries_acc::Decimal( "46929.18" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 1 ), aries_acc::Decimal( "32151.78" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 2 ), aries_acc::Decimal( "58749.59" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 3 ), aries_acc::Decimal( "140081.83" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 4 ), aries_acc::Decimal( "165157.10" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 5 ), aries_acc::Decimal( "114019.79" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 6 ), aries_acc::Decimal( "208660.75" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 7 ), aries_acc::Decimal( "58949.67" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 8 ), aries_acc::Decimal( "224722.49" ) );
    ASSERT_EQ( columnBuff->GetDecimal( 9 ), aries_acc::Decimal( "216628.37" ) );

    columnBuff = resTable->GetColumnBuffer( 5 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), "1996-12-01" );
    ASSERT_EQ( columnBuff->GetDateAsString( 1 ), "1995-10-11" );
    ASSERT_EQ( columnBuff->GetDateAsString( 2 ), "1992-02-21" );
    ASSERT_EQ( columnBuff->GetDateAsString( 3 ), "1992-06-12" );
    ASSERT_EQ( columnBuff->GetDateAsString( 4 ), "1992-08-01" );
    ASSERT_EQ( columnBuff->GetDateAsString( 5 ), "1994-04-09" );
    ASSERT_EQ( columnBuff->GetDateAsString( 6 ), "1995-07-16" );
    ASSERT_EQ( columnBuff->GetDateAsString( 7 ), "1998-07-21" );
    ASSERT_EQ( columnBuff->GetDateAsString( 8 ), "1998-04-23" );
    ASSERT_EQ( columnBuff->GetDateAsString( 9 ), "1993-02-18" );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "4-NOT SPECIFIED" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 4 ), "5-LOW" );
    ASSERT_EQ( columnBuff->GetString( 5 ), "1-URGENT" );
    ASSERT_EQ( columnBuff->GetString( 6 ), "2-HIGH" );
    ASSERT_EQ( columnBuff->GetString( 7 ), "3-MEDIUM" );
    ASSERT_EQ( columnBuff->GetString( 8 ), "3-MEDIUM" );
    ASSERT_EQ( columnBuff->GetString( 9 ), "3-MEDIUM" );

/*
mysql> select * from lineitem order by l_orderkey;
+------------+-----------+-----------+--------------+------------+-----------------+------------+-------+--------------+--------------+------------+--------------+---------------+-------------------+------------+--------------------------------------------+
  | l_orderkey | l_partkey | l_suppkey | l_linenumber | l_quantity | l_extendedprice | l_discount | l_tax | l_returnflag | l_linestatus | l_shipdate | l_commitdate | l_receiptdate | l_shipinstruct    | l_shipmode | l_comment                                  |
  +------------+-----------+-----------+--------------+------------+-----------------+------------+-------+--------------+--------------+------------+--------------+---------------+-------------------+------------+--------------------------------------------+
 0|          2 |    106170 |      1191 |            1 |      38.00 |        44694.46 |       0.00 |  0.05 | N            | O            | 1997-01-28 | 1997-01-14   | 1997-02-02    | TAKE BACK RETURN  | RAIL       | ven requests. deposits breach a            |
 1|          4 |     88035 |      5560 |            1 |      30.00 |        30690.90 |       0.03 |  0.08 | N            | O            | 1996-01-10 | 1995-12-14   | 1996-01-18    | DELIVER IN PERSON | REG AIR    | - quickly regular packages sleep. idly     |
 2|          6 |    139636 |      2150 |            1 |      37.00 |        61998.31 |       0.08 |  0.03 | A            | F            | 1992-04-27 | 1992-05-15   | 1992-05-02    | TAKE BACK RETURN  | TRUCK      | p furiously special foxes                  |
 3|         10 |  18416670 |    666689 |            3 |      26.00 |        41229.50 |       0.05 |  0.02 | R            | F            | 1992-09-08 | 1992-07-12   | 1992-10-05    | DELIVER IN PERSON | REG AIR    | ites. deposit                              |
 4|         10 |   5277469 |     27485 |            2 |      28.00 |        40493.60 |       0.07 |  0.05 | R            | F            | 1992-07-17 | 1992-08-12   | 1992-08-11    | NONE              | TRUCK      |  at the even, unusual escapades are amon   |
 5|         10 |   6347670 |    347671 |            1 |      36.00 |        61824.96 |       0.02 |  0.00 | A            | F            | 1992-07-19 | 1992-07-16   | 1992-07-30    | NONE              | MAIL       | yly silent instructions unwind             |
 6|         12 |    593099 |     93100 |            4 |      19.00 |        22649.33 |       0.00 |  0.07 | R            | F            | 1992-08-28 | 1992-10-04   | 1992-09-08    | DELIVER IN PERSON | FOB        | ter the ironic requests. fluffily even     |
 7|         12 |  19463355 |    713375 |            3 |      23.00 |        30299.74 |       0.10 |  0.08 | A            | F            | 1992-11-01 | 1992-09-29   | 1992-11-18    | TAKE BACK RETURN  | TRUCK      | haggle blithely against the                |
 8|         12 |  11357141 |    107175 |            2 |      46.00 |        55088.68 |       0.01 |  0.04 | A            | F            | 1992-08-13 | 1992-10-07   | 1992-08-27    | DELIVER IN PERSON | REG AIR    | uickly. permanent sentiments around        |
 9|         12 |  16425975 |    425976 |            1 |      29.00 |        55104.35 |       0.08 |  0.08 | A            | F            | 1992-10-02 | 1992-09-13   | 1992-10-17    | NONE              | TRUCK      | ckages cajole. blithely express dolphins   |
10|         14 |   2653174 |    403181 |            5 |      23.00 |        25921.92 |       0.10 |  0.08 | R            | F            | 1994-06-12 | 1994-06-05   | 1994-06-24    | NONE              | AIR        | ic packages. ca                            |
11|         14 |   9540231 |     40250 |            4 |      17.00 |        21602.92 |       0.04 |  0.00 | R            | F            | 1994-07-23 | 1994-06-15   | 1994-07-27    | COLLECT COD       | AIR        | efully regular accounts sleep fina         |
12|         14 |  19173532 |    423552 |            3 |       5.00 |         8022.90 |       0.07 |  0.00 | A            | F            | 1994-06-23 | 1994-05-13   | 1994-07-07    | DELIVER IN PERSON | REG AIR    |  about the quickly dari                    |
13|         14 |  12263889 |    763914 |            2 |      24.00 |        44454.48 |       0.00 |  0.00 | A            | F            | 1994-07-21 | 1994-06-14   | 1994-08-20    | NONE              | TRUCK      | ve the quickly unusual pinto               |
14|         14 |  11123477 |    623500 |            1 |      11.00 |        16499.12 |       0.02 |  0.00 | R            | F            | 1994-06-15 | 1994-05-22   | 1994-07-09    | NONE              | TRUCK      | nag furiou                                 |
15|         32 |     82704 |      7721 |            1 |      28.00 |        47227.60 |       0.05 |  0.08 | N            | O            | 1995-10-23 | 1995-08-27   | 1995-10-26    | TAKE BACK RETURN  | TRUCK      | sleep quickly. req                         |
16|         32 |    197921 |       441 |            2 |      32.00 |        64605.44 |       0.02 |  0.00 | N            | O            | 1995-08-14 | 1995-10-07   | 1995-08-27    | COLLECT COD       | AIR        | lithely regular deposits. fluffily         |
17|         32 |     44161 |      6666 |            3 |       2.00 |         2210.32 |       0.09 |  0.02 | N            | O            | 1995-08-07 | 1995-10-07   | 1995-08-23    | DELIVER IN PERSON | AIR        |  express accounts wake according to the    |
18|         32 |      2743 |      7744 |            4 |       4.00 |         6582.96 |       0.09 |  0.03 | N            | O            | 1995-08-04 | 1995-10-01   | 1995-09-03    | NONE              | REG AIR    | e slyly final pac                          |
19|         32 |     85811 |      8320 |            5 |      44.00 |        79059.64 |       0.05 |  0.06 | N            | O            | 1995-08-28 | 1995-08-20   | 1995-09-14    | DELIVER IN PERSON | AIR        | symptotes nag according to the ironic depo |
20|         32 |     11615 |      4117 |            6 |       6.00 |         9159.66 |       0.04 |  0.03 | N            | O            | 1995-07-21 | 1995-09-23   | 1995-07-25    | COLLECT COD       | RAIL       |  gifts cajole carefully.                   |
21|         34 |    169544 |      4577 |            3 |       6.00 |         9681.24 |       0.02 |  0.06 | N            | O            | 1998-10-30 | 1998-09-20   | 1998-11-05    | NONE              | FOB        | ar foxes sleep                             |
22|         34 |     89414 |      1923 |            2 |      22.00 |        30875.02 |       0.08 |  0.06 | N            | O            | 1998-10-09 | 1998-10-16   | 1998-10-12    | NONE              | FOB        | thely slyly p                              |
23|         34 |     88362 |       871 |            1 |      13.00 |        17554.68 |       0.00 |  0.07 | N            | O            | 1998-10-23 | 1998-09-14   | 1998-11-06    | NONE              | REG AIR    | nic accounts. deposits are alon            |
24|         40 |   3101351 |    851361 |            7 |      18.00 |        24339.60 |       0.03 |  0.07 | N            | O            | 1998-04-28 | 1998-06-11   | 1998-05-26    | DELIVER IN PERSON | SHIP       | pendencies nag quickly about the special   |
25|         40 |    687994 |    937995 |            6 |      28.00 |        55494.88 |       0.00 |  0.07 | N            | O            | 1998-07-09 | 1998-06-30   | 1998-07-10    | NONE              | REG AIR    | its. deposits wake. ironic packages affix  |
26|         40 |  13481329 |    981356 |            5 |       6.00 |         7857.90 |       0.06 |  0.01 | N            | O            | 1998-06-14 | 1998-07-19   | 1998-06-23    | DELIVER IN PERSON | REG AIR    | beans; slyly ironic pinto beans impress    |
27|         40 |   9989498 |    989499 |            4 |      13.00 |        20631.00 |       0.09 |  0.03 | N            | O            | 1998-07-30 | 1998-07-20   | 1998-08-15    | DELIVER IN PERSON | SHIP       | . ironic, bold deposits are. car           |
28|         40 |   8142441 |    892466 |            3 |       6.00 |         8898.24 |       0.07 |  0.06 | N            | O            | 1998-06-02 | 1998-07-05   | 1998-06-12    | COLLECT COD       | FOB        | gage furiously slyly ironic reques         |
29|         40 |   6832162 |     82169 |            2 |      33.00 |        36096.06 |       0.00 |  0.06 | N            | O            | 1998-06-25 | 1998-06-24   | 1998-07-19    | NONE              | FOB        | nstructions haggl                          |
30|         40 |   8030395 |    780420 |            1 |      50.00 |        66249.50 |       0.00 |  0.00 | N            | O            | 1998-08-11 | 1998-06-22   | 1998-08-29    | TAKE BACK RETURN  | TRUCK      | cajole furiously bo                        |
31|         42 |   8518925 |    768934 |            4 |      27.00 |        52474.50 |       0.10 |  0.00 | R            | F            | 1993-03-30 | 1993-04-15   | 1993-04-06    | DELIVER IN PERSON | AIR        | ng the furiously unusual                   |
32|         42 |  16745911 |    495960 |            3 |      32.00 |        62594.56 |       0.09 |  0.05 | A            | F            | 1993-03-05 | 1993-05-12   | 1993-03-26    | DELIVER IN PERSON | FOB        | fully regular d                            |
33|         42 |  11348650 |    348651 |            2 |      38.00 |        64527.42 |       0.06 |  0.08 | R            | F            | 1993-03-25 | 1993-04-18   | 1993-04-14    | TAKE BACK RETURN  | FOB        | egular courts. fluffily unusu              |
34|         42 |  19918567 |    668625 |            1 |      28.00 |        44367.96 |       0.08 |  0.08 | R            | F            | 1993-04-02 | 1993-04-08   | 1993-04-11    | COLLECT COD       | AIR        | tructions cajole above the unusual accou   |
  +------------+-----------+-----------+--------------+------------+-----------------+------------+-------+--------------+--------------+------------+--------------+---------------+-------------------+------------+--------------------------------------------+
35 rows in set (0.24 sec)
*/
    sql = "select * from lineitem order by l_orderkey";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, test_db_name );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), itemCount );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 10 );
    ASSERT_EQ( columnBuff->GetInt32( 5 ), 10 );
    ASSERT_EQ( columnBuff->GetInt32( 6 ), 12 );
    ASSERT_EQ( columnBuff->GetInt32( 8 ), 12 );
    ASSERT_EQ( columnBuff->GetInt32( 10 ), 14 );
    ASSERT_EQ( columnBuff->GetInt32( 12 ), 14 );
    ASSERT_EQ( columnBuff->GetInt32( 14 ), 14 );
    ASSERT_EQ( columnBuff->GetInt32( 16 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 18 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 20 ), 32 );
    ASSERT_EQ( columnBuff->GetInt32( 25 ), 40 );
    ASSERT_EQ( columnBuff->GetInt32( 28 ), 40 );
    ASSERT_EQ( columnBuff->GetInt32( 31 ), 42 );
    ASSERT_EQ( columnBuff->GetInt32( 34 ), 42 );

    columnBuff = resTable->GetColumnBuffer( 6 );
    ASSERT_EQ( columnBuff->GetDecimal( 0 ), aries_acc::Decimal( "44694.46") );
    ASSERT_EQ( columnBuff->GetDecimal( 1 ), aries_acc::Decimal( "30690.90") );
    ASSERT_EQ( columnBuff->GetDecimal( 2 ), aries_acc::Decimal( "61998.31") );
    ASSERT_EQ( columnBuff->GetDecimal( 3 ), aries_acc::Decimal( "41229.50") );
    ASSERT_EQ( columnBuff->GetDecimal( 5 ), aries_acc::Decimal( "61824.96") );
    ASSERT_EQ( columnBuff->GetDecimal( 6 ), aries_acc::Decimal( "22649.33") );
    ASSERT_EQ( columnBuff->GetDecimal( 8 ), aries_acc::Decimal( "55088.68") );
    ASSERT_EQ( columnBuff->GetDecimal( 10 ), aries_acc::Decimal( "25921.92") );
    ASSERT_EQ( columnBuff->GetDecimal( 12 ), aries_acc::Decimal( "8022.90") );
    ASSERT_EQ( columnBuff->GetDecimal( 14 ), aries_acc::Decimal( "16499.12") );
    ASSERT_EQ( columnBuff->GetDecimal( 16 ), aries_acc::Decimal( "64605.44") );
    ASSERT_EQ( columnBuff->GetDecimal( 18 ), aries_acc::Decimal( "6582.96") );
    ASSERT_EQ( columnBuff->GetDecimal( 20 ), aries_acc::Decimal( "9159.66") );
    ASSERT_EQ( columnBuff->GetDecimal( 25 ), aries_acc::Decimal( "55494.88") );
    ASSERT_EQ( columnBuff->GetDecimal( 28 ), aries_acc::Decimal( "8898.24") );
    ASSERT_EQ( columnBuff->GetDecimal( 31 ), aries_acc::Decimal( "52474.50") );
    ASSERT_EQ( columnBuff->GetDecimal( 34 ), aries_acc::Decimal( "44367.96") );

    columnBuff = resTable->GetColumnBuffer( 12 );
    ASSERT_EQ( columnBuff->GetDateAsString( 0 ), "1997-01-14" );
    ASSERT_EQ( columnBuff->GetDateAsString( 1 ), "1995-12-14" );
    ASSERT_EQ( columnBuff->GetDateAsString( 2 ), "1992-05-15" );
    ASSERT_EQ( columnBuff->GetDateAsString( 3 ), "1992-07-12" );
    ASSERT_EQ( columnBuff->GetDateAsString( 5 ), "1992-07-16" );
    ASSERT_EQ( columnBuff->GetDateAsString( 6 ), "1992-10-04" );
    ASSERT_EQ( columnBuff->GetDateAsString( 8 ), "1992-10-07" );
    ASSERT_EQ( columnBuff->GetDateAsString( 10 ), "1994-06-05" );
    ASSERT_EQ( columnBuff->GetDateAsString( 12 ), "1994-05-13");
    ASSERT_EQ( columnBuff->GetDateAsString( 14 ), "1994-05-22");
    ASSERT_EQ( columnBuff->GetDateAsString( 16 ), "1995-10-07");
    ASSERT_EQ( columnBuff->GetDateAsString( 18 ), "1995-10-01");
    ASSERT_EQ( columnBuff->GetDateAsString( 20 ), "1995-09-23");
    ASSERT_EQ( columnBuff->GetDateAsString( 25 ), "1998-06-30" );
    ASSERT_EQ( columnBuff->GetDateAsString( 28 ), "1998-07-05");
    ASSERT_EQ( columnBuff->GetDateAsString( 31 ), "1993-04-15");
    ASSERT_EQ( columnBuff->GetDateAsString( 34 ), "1993-04-08");

    columnBuff = resTable->GetColumnBuffer( 15 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "RAIL" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "TRUCK" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 5 ), "MAIL" );
    ASSERT_EQ( columnBuff->GetString( 6 ), "FOB" );
    ASSERT_EQ( columnBuff->GetString( 8 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 10 ), "AIR" );
    ASSERT_EQ( columnBuff->GetString( 12 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 14 ), "TRUCK" );
    ASSERT_EQ( columnBuff->GetString( 16 ), "AIR" );
    ASSERT_EQ( columnBuff->GetString( 18 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 20 ), "RAIL" );
    ASSERT_EQ( columnBuff->GetString( 25 ), "REG AIR" );
    ASSERT_EQ( columnBuff->GetString( 28 ), "FOB" );
    ASSERT_EQ( columnBuff->GetString( 31 ), "AIR" );
    ASSERT_EQ( columnBuff->GetString( 34 ), "AIR" );
}

// test literal insert
ARIES_UNIT_TEST_F( TestAriesXLogRecoveryer, literal_insert )
{
    string test_db_name = "testariesxlogrecoveryer_test2";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + test_db_name + ";", std::string() );
    InitTable( test_db_name, "orders" );
    InitTable( test_db_name, "lineitem" );

    PrepareInitData( test_db_name );
    string sql( R"(
        insert into orders values ( 100000000, 8567159, "F",284394.47,"1994-05-08", "2-HIGH","Clerk#000033528",0, "ctions boost carefully before the even, ex");
    )");
    
}
} // namespace aries_test
