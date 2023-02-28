#include <gtest/gtest.h>
#include <thread>
#include <iostream>
#include "AriesEngine/AriesDeleteNode.h"
#include "AriesEngine/AriesUpdateNode.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "AriesConstantGenerator.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/transaction/AriesXLogRecoveryer.h"

#include "QueryOpNode.h"
#include "../../TestUtils.h"

using namespace aries;
using namespace aries_test;
using namespace aries_engine;

#define TEST_DB_NAME "test_db"
#define TEST_TABLE_NAME "region"

void DropRegionTable()
{
    string sql = "drop database if exists ";
    sql += TEST_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
}

void SetupTestEnvirement() {
    DropRegionTable();
    string sql = "create database if not exists ";
    sql += TEST_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "use ";
    sql += TEST_DB_NAME;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );

    sql = R"(CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
                            R_NAME       CHAR(25) NOT NULL,
                            R_COMMENT    VARCHAR(152));)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_TABLE_NAME;
    sql += " values(4, \"MIDDLE EAST\", \"uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle "
           "furiousl\");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_TABLE_NAME;
    sql += " values(3, \"EUROPE\", \"ly final courts cajole furiously final excuse\");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_TABLE_NAME;
    sql += " values(2, \"ASIA\", \"ges. thinly even pinto beans ca\");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_TABLE_NAME;
    sql += " values(1, \"AMERICA\", \"hs use ironic, even requests. s\");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_TABLE_NAME;
    sql += " values(0, \"AFRICA\", \"lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according "
           "to \");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto recoveryer = std::make_shared< aries_engine::AriesXLogRecoveryer >();
    recoveryer->SetReader( aries_engine::AriesXLogManager::GetInstance().GetReader() );
    auto r = recoveryer->Recovery();
    ARIES_ASSERT( r, "cannot recovery from xlog" );
}

TEST( AriesDeleteNode, GetNext )
{
    SetupTestEnvirement();
    auto sql = "select 0 - ( R_REGIONKEY + 1 ) from REGION where R_REGIONKEY < 5";
    auto source = std::make_shared< QueryOpNode >( sql, TEST_DB_NAME );

    auto transaction = AriesTransManager::GetInstance().NewTransaction();
    auto node = std::make_shared< AriesDeleteNode >( transaction, TEST_DB_NAME, TEST_TABLE_NAME );

    node->SetColumnId4RowPos( 1 );
    node->SetSourceNode( source );

    ASSERT_TRUE( node->Open() );

    auto result = node->GetNext();
    ASSERT_EQ( result.Status, AriesOpNodeStatus::END );

    ASSERT_TRUE( result.TableBlock );

    auto column = result.TableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 1 );

    ASSERT_EQ( column->GetInt64AsString( 0 ), "5" );

    transaction->AddModifiedTable( schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME ) );
    AriesTransManager::GetInstance().EndTransaction( transaction, TransactionStatus::COMMITTED );

    auto sql1 = " select * from REGION where R_REGIONKEY >= 5 order by R_REGIONKEY";
    auto sql2 = " select * from REGION order by R_REGIONKEY";

    AssertTwoSQLS( sql1, sql2, TEST_DB_NAME );
}

TEST( AriesDeleteNode, GetNext2 )
{
    SetupTestEnvirement();
    auto source_sql_for_delete = "select 0 - ( R_REGIONKEY + 1 ) from REGION where R_REGIONKEY = 4";
    auto source_for_delete = std::make_shared< QueryOpNode >( source_sql_for_delete, TEST_DB_NAME );

    auto source_sql_for_update = "select 0 - ( R_REGIONKEY + 1 ), R_REGIONKEY * 100 from REGION where R_REGIONKEY = 4";
    auto source_for_update = std::make_shared< QueryOpNode >( source_sql_for_update, TEST_DB_NAME );

    auto transaction = AriesTransManager::GetInstance().NewTransaction();
    auto update_transaction = AriesTransManager::GetInstance().NewTransaction();

    auto update_node = std::make_shared< AriesUpdateNode >( update_transaction, TEST_DB_NAME, TEST_TABLE_NAME );
    auto node = std::make_shared< AriesDeleteNode >( transaction, TEST_DB_NAME, TEST_TABLE_NAME );

    update_node->SetSourceNode( source_for_update );
    update_node->SetColumnId4RowPos( 1 );
    std::vector< int > update_ids;
    update_ids.emplace_back( 1 );
    update_node->SetUpdateColumnIds( update_ids );

    node->SetColumnId4RowPos( 1 );
    node->SetSourceNode( source_for_delete );

    ASSERT_TRUE( node->Open() );
    ASSERT_TRUE( update_node->Open() );

    auto update_result = update_node->GetNext();
    ASSERT_EQ( update_result.Status, AriesOpNodeStatus::END );
    ASSERT_TRUE( update_result.TableBlock );

    auto column = update_result.TableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 3 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "1" );
    ASSERT_EQ( column->GetInt64AsString( 1 ), "1" );
    ASSERT_EQ( column->GetInt64AsString( 2 ), "0" );

    std::thread t( [&]
    {
        ::sleep( 5 );
        update_transaction->AddModifiedTable( schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME ) );
        AriesTransManager::GetInstance().EndTransaction( update_transaction, TransactionStatus::COMMITTED );
    });

    auto delete_result = node->GetNext();
    ASSERT_EQ( delete_result.Status, AriesOpNodeStatus::ERROR );

    ASSERT_FALSE( delete_result.TableBlock );
    AriesTransManager::GetInstance().EndTransaction( transaction, TransactionStatus::ABORTED );

    t.join();

    auto sql = "select * from REGION where R_REGIONKEY = 4";

    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
    auto table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "select * from REGION where R_REGIONKEY = 1";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
    table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 1 );
}

TEST( AriesDeleteNode, GetNext3 )
{
    SetupTestEnvirement();
    auto source_sql_for_delete = "select 0 - ( R_REGIONKEY + 1 ) from REGION where R_REGIONKEY = 4";
    auto source_for_delete = std::make_shared< QueryOpNode >( source_sql_for_delete, TEST_DB_NAME );

    auto source_sql_for_update = "select 0 - ( R_REGIONKEY + 1 ), R_REGIONKEY * 100 from REGION where R_REGIONKEY = 4";
    auto source_for_update = std::make_shared< QueryOpNode >( source_sql_for_update, TEST_DB_NAME );

    auto transaction = AriesTransManager::GetInstance().NewTransaction();
    auto update_transaction = AriesTransManager::GetInstance().NewTransaction();

    auto update_node = std::make_shared< AriesUpdateNode >( update_transaction, TEST_DB_NAME, TEST_TABLE_NAME );
    auto node = std::make_shared< AriesDeleteNode >( transaction, TEST_DB_NAME, TEST_TABLE_NAME );

    update_node->SetSourceNode( source_for_update );
    update_node->SetColumnId4RowPos( 1 );
    std::vector< int > update_ids;
    update_ids.emplace_back( 1 );
    update_node->SetUpdateColumnIds( update_ids );

    node->SetColumnId4RowPos( 1 );
    node->SetSourceNode( source_for_delete );

    ASSERT_TRUE( node->Open() );
    ASSERT_TRUE( update_node->Open() );

    auto delete_result = node->GetNext();
    ASSERT_EQ( delete_result.Status, AriesOpNodeStatus::END );
    ASSERT_TRUE( delete_result.TableBlock );

    std::thread t( [&]
    {
        ::sleep( 5 );
        transaction->AddModifiedTable( schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME ) );
        AriesTransManager::GetInstance().EndTransaction( transaction, TransactionStatus::COMMITTED );
    });

    auto update_result = update_node->GetNext();
    ASSERT_EQ( update_result.Status, AriesOpNodeStatus::ERROR );
    ASSERT_FALSE( update_result.TableBlock );

    t.join();

    auto sql = "select * from REGION where R_REGIONKEY = 4";

    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
    auto table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 0 );
}

TEST( AriesDeleteNode, GetNext4 )
{
    SetupTestEnvirement();
    auto source_sql_for_delete = "select 0 - ( R_REGIONKEY + 1 ) from REGION where R_REGIONKEY = 4";
    auto source_for_delete = std::make_shared< QueryOpNode >( source_sql_for_delete, TEST_DB_NAME );

    auto source_sql_for_update = "select 0 - ( R_REGIONKEY + 1 ), R_REGIONKEY * 100 from REGION where R_REGIONKEY = 4";
    auto source_for_update = std::make_shared< QueryOpNode >( source_sql_for_update, TEST_DB_NAME );

    auto transaction = AriesTransManager::GetInstance().NewTransaction();
    auto update_transaction = AriesTransManager::GetInstance().NewTransaction();

    auto update_node = std::make_shared< AriesUpdateNode >( update_transaction, TEST_DB_NAME, TEST_TABLE_NAME );
    auto node = std::make_shared< AriesDeleteNode >( transaction, TEST_DB_NAME, TEST_TABLE_NAME );

    update_node->SetSourceNode( source_for_update );
    update_node->SetColumnId4RowPos( 1 );
    std::vector< int > update_ids;
    update_ids.emplace_back( 1 );
    update_node->SetUpdateColumnIds( update_ids );

    node->SetColumnId4RowPos( 1 );
    node->SetSourceNode( source_for_delete );

    ASSERT_TRUE( node->Open() );
    ASSERT_TRUE( update_node->Open() );

    auto delete_result = node->GetNext();
    ASSERT_EQ( delete_result.Status, AriesOpNodeStatus::END );
    ASSERT_TRUE( delete_result.TableBlock );

    std::thread t( [&]
    {
        ::sleep( 5 );
        AriesTransManager::GetInstance().EndTransaction( transaction, TransactionStatus::ABORTED );
    });

    auto update_result = update_node->GetNext();
    ASSERT_EQ( update_result.Status, AriesOpNodeStatus::ERROR );
    ASSERT_FALSE( update_result.TableBlock );

    AriesTransManager::GetInstance().EndTransaction( update_transaction, TransactionStatus::ABORTED );

    t.join();

    auto sql = "select * from REGION where R_REGIONKEY = 1";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
    auto table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 1 );
}
