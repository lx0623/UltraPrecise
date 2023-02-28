//
// Created by david.shen on 2020/3/26.
//

#include <gtest/gtest.h>
#include <pthread.h>

#include "AriesEngine/AriesUpdateNode.h"
#include "AriesEngine/AriesMvccScanNode.h"
#include "AriesEngine/AriesUpdateCalcNode.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "CudaAcc/DynamicKernel.h"

#include "../../TestUtils.h"
using namespace aries_test;

extern bool STRICT_MODE;

/**
 * |=====================================================================================|
 * | 测试用例名称           |用例描述                         | 期待结果                      |
 * | committedUpdate      |更新并commit                    | 能够正常体现更新结果             |
 * | abortedUpdate        |更新并abort                     | 无任何更新                     |
 * | inprogressUpdate     |更新但不commit和abort            | 无任何更新                     |
 * | waitForAbortedUpdate |tx1 更新时需要等待另外tx2 abort   | 更新成功，结果为tx1的结果        |
 * | waitForComittedUpdate|tx1 更新时需要等待另外tx2 commit  | 更新失败,结果未另外tx2更新的结果  |
*/

BEGIN_ARIES_ENGINE_NAMESPACE

#define TEST_DB_NAME "test_update"
#define TEST_TABLE_NAME "region"

TEST(UT_TestUpdateValue, not_null_to_null)
{
    string tableName( "not_null_to_null" );
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 int not null )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values(1)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "update " + tableName + " set f1 = null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ(result->GetErrorCode(), ER_BAD_NULL_ERROR );

    sql = "update " + tableName + " set f1 = 1 + null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ(result->GetErrorCode(), ER_BAD_NULL_ERROR );

    sql = "update " + tableName + " set f1 = f1 + null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ(result->GetErrorCode(), ER_BAD_NULL_ERROR );
}

TEST(UT_TestUpdateValue, primary_to_null)
{
    string tableName( "primary_to_null" );
    InitTable( TEST_DB_NAME, tableName );

    string sql = "create table " + tableName + "( f1 int primary key )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values(1)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "update " + tableName + " set f1 = null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ(result->GetErrorCode(), ER_BAD_NULL_ERROR );

    sql = "update " + tableName + " set f1 = 1 + null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ(result->GetErrorCode(), ER_BAD_NULL_ERROR );

    sql = "update " + tableName + " set f1 = f1 + null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ(result->GetErrorCode(), ER_BAD_NULL_ERROR );
}

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

void UpdateWithInprogress(AriesTransactionPtr tx, int added, bool success) {
    auto mvccScanNode = make_shared<AriesMvccScanNode>(tx, TEST_DB_NAME, TEST_TABLE_NAME);
    mvccScanNode->SetOutputColumnIds({4,1});

    int exprId = 0;
    auto calc = AriesCommonExpr::Create( AriesExprType::CALC,
                                      static_cast< int >( AriesCalculatorOpType::ADD ),
                                      AriesColumnType{ { AriesValueType::INT32 }, false, false }
                                    );
    auto col1 = AriesCommonExpr::Create( AriesExprType::COLUMN_ID, 1, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto col2 = AriesCommonExpr::Create( AriesExprType::COLUMN_ID, 2, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto const1 = AriesCommonExpr::Create( AriesExprType::INTEGER, added, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    calc->AddChild( std::move( col2) );
    calc->AddChild( std::move( const1 ) );
    calc->SetId( ++exprId );

    auto updateCalcNode = make_shared<AriesUpdateCalcNode>();
    updateCalcNode->SetNodeId( 0 );
    updateCalcNode->SetSourceNode(mvccScanNode);
    updateCalcNode->SetColumnIds({1,2});
    std::vector< AriesCommonExprUPtr > exprs;
    exprs.emplace_back(move(col1));
    exprs.emplace_back(move(calc));
    updateCalcNode->SetCalcExprs(exprs);
    auto updateNode = make_shared<AriesUpdateNode>(tx, TEST_DB_NAME, TEST_TABLE_NAME);
    updateNode->SetNodeId( 1 );
    updateNode->SetSourceNode(updateCalcNode);
    updateNode->SetUpdateColumnIds({1});
    updateNode->SetColumnId4RowPos(1);

    auto code = R"(#include "functions.hxx"
#include "AriesDateFormat.hxx"
#include "aries_char.hxx"
#include "decimal.hxx"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesIntervalTime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"
#include "AriesYear.hxx"
#include "AriesTimeCalc.hxx"
#include "AriesSqlFunctions.hxx"
#include "AriesColumnDataIterator.hxx"
using namespace aries_acc;

)" + updateNode->GetCudaKernelCode();

    AriesDynamicCodeInfo codeInfo;
    codeInfo.KernelCode = code;
    auto modules = AriesDynamicKernelManager::GetInstance().CompileKernels( codeInfo );
    updateNode->SetCuModule( modules->Modules );

    ASSERT_TRUE(updateNode->Open());
    auto res = updateNode->GetNext();
    ASSERT_TRUE(res.Status == (success ? AriesOpNodeStatus::END : AriesOpNodeStatus::ERROR));
}

static void * abortedFunc(void *arg) {
    auto abortTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress(abortTx, 1000, true);
    sleep(2);
    AriesTransManager::GetInstance().EndTransaction( abortTx, TransactionStatus::ABORTED );
    return nullptr;
}

static void * committedFunc(void *arg) {
    auto committedTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress(committedTx, 2000, true);
    sleep(2);
    AriesTransManager::GetInstance().EndTransaction( committedTx, TransactionStatus::COMMITTED );
    return nullptr;
}

void CreateAbortedTx() {
    pthread_t thread;
    pthread_create(&thread, nullptr, abortedFunc, nullptr);
    sleep(1);
}

void CreateCommittedTx() {
    pthread_t thread;
    pthread_create(&thread, nullptr, committedFunc, nullptr);
    sleep(1);
}
void committedUpdate()
{
    SetupTestEnvirement();
    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress(updateTx, 100, true);
    AriesTransManager::GetInstance().EndTransaction( updateTx, TransactionStatus::COMMITTED );
    //check result
    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
    auto table = AriesMvccTableManager::GetInstance().getMvccTable(TEST_DB_NAME, TEST_TABLE_NAME);
    auto tableAfterInsert = table->GetTable(selectTx, {1,2,3});
    ASSERT_TRUE(tableAfterInsert->GetRowCount() == 5);
    //very column 1
    auto columnData = tableAfterInsert->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 100 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 101 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 102 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 103 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 104 );
    //very column 2
    columnData = tableAfterInsert->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterInsert->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
}
void abortedUpdate()
{
    SetupTestEnvirement();
    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress(updateTx, 100, true);
    AriesTransManager::GetInstance().EndTransaction( updateTx, TransactionStatus::ABORTED );
    //check result
    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
    auto table = AriesMvccTableManager::GetInstance().getMvccTable(TEST_DB_NAME, TEST_TABLE_NAME);
    auto tableAfterInsert = table->GetTable(selectTx, {1,2,3});
    ASSERT_TRUE(tableAfterInsert->GetRowCount() == 5);
    //very column 1
    auto columnData = tableAfterInsert->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 4 );
    //very column 2
    columnData = tableAfterInsert->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterInsert->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
}

void inprogressUpdate()
{
    SetupTestEnvirement();
    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress(updateTx, 100, true);

    //check result
    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
    auto table = AriesMvccTableManager::GetInstance().getMvccTable(TEST_DB_NAME, TEST_TABLE_NAME);
    auto tableAfterInsert = table->GetTable(selectTx, {1,2,3});
    ASSERT_TRUE(tableAfterInsert->GetRowCount() == 5);
    //very column 1
    auto columnData = tableAfterInsert->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 4 );
    //very column 2
    columnData = tableAfterInsert->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterInsert->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
}

//TEST(UT_TestAriesUpdateNode, committedUpdate) {
//    committedUpdate();
//}
//
//TEST(UT_TestAriesUpdateNode, abortedUpdate) {
//    abortedUpdate();
//}
//
//TEST(UT_TestAriesUpdateNode, inprogressUpdate) {
//    inprogressUpdate();
//}
//
//TEST(UT_TestAriesUpdateNode, waitForAbortedUpdate) {
//    SetupTestEnvirement();
//    CreateAbortedTx();
//    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
//    UpdateWithInprogress(updateTx, 100, true);
//    AriesTransManager::GetInstance().EndTransaction( updateTx, TransactionStatus::COMMITTED );
//
//    //check result
//    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
//    auto table = AriesMvccTableManager::GetInstance().getTable(TEST_DB_NAME, TEST_TABLE_NAME);
//    auto tableAfterInsert = table->GetTable(selectTx, {1,2,3});
//    ASSERT_TRUE(tableAfterInsert->GetRowCount() == 5);
//    //very column 1
//    auto columnData = tableAfterInsert->GetColumnBuffer(1);
//    ASSERT_EQ( columnData->GetInt32( 0 ), 100 );
//    ASSERT_EQ( columnData->GetInt32( 1 ), 101 );
//    ASSERT_EQ( columnData->GetInt32( 2 ), 102 );
//    ASSERT_EQ( columnData->GetInt32( 3 ), 103 );
//    ASSERT_EQ( columnData->GetInt32( 4 ), 104 );
//    //very column 2
//    columnData = tableAfterInsert->GetColumnBuffer(2);
//    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
//    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
//    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
//    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
//    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
//    //very column 3
//    columnData = tableAfterInsert->GetColumnBuffer(3);
//    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
//    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
//    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
//    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
//    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
//}
//
//TEST(UT_TestAriesUpdateNode, waitForComittedUpdate) {
//    SetupTestEnvirement();
//    CreateCommittedTx();
//    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
//    UpdateWithInprogress(updateTx, 100, false);
//    AriesTransManager::GetInstance().EndTransaction(updateTx, TransactionStatus::ABORTED);
//
//    //check result
//    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
//    auto table = AriesMvccTableManager::GetInstance().getTable(TEST_DB_NAME, TEST_TABLE_NAME);
//    auto tableAfterInsert = table->GetTable(selectTx, {1,2,3});
//    ASSERT_TRUE(tableAfterInsert->GetRowCount() == 5);
//    //very column 1
//    auto columnData = tableAfterInsert->GetColumnBuffer(1);
//    ASSERT_EQ( columnData->GetInt32( 0 ), 2000 );
//    ASSERT_EQ( columnData->GetInt32( 1 ), 2001 );
//    ASSERT_EQ( columnData->GetInt32( 2 ), 2002 );
//    ASSERT_EQ( columnData->GetInt32( 3 ), 2003 );
//    ASSERT_EQ( columnData->GetInt32( 4 ), 2004 );
//    //very column 2
//    columnData = tableAfterInsert->GetColumnBuffer(2);
//    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
//    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
//    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
//    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
//    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
//    //very column 3
//    columnData = tableAfterInsert->GetColumnBuffer(3);
//    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
//    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
//    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
//    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
//    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
//}

// UPDATE to be supported
TEST(UT_TestAriesUpdateNodeFirstWriterWin, committedUpdate)
{
    committedUpdate();
}

TEST(UT_TestAriesUpdateNodeFirstWriterWin, abortedUpdate)
{
    abortedUpdate();
}

TEST(UT_TestAriesUpdateNodeFirstWriterWin, inprogressUpdate)
{
    inprogressUpdate();
}

TEST(UT_TestAriesUpdateNodeFirstWriterWin, notWaitForAbortedUpdateAborted)
{
    SetupTestEnvirement();
    CreateAbortedTx();
    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress( updateTx, 100, false );
    AriesTransManager::GetInstance().EndTransaction( updateTx, TransactionStatus::COMMITTED );

    //check result
    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
    auto table = AriesMvccTableManager::GetInstance().getMvccTable( TEST_DB_NAME, TEST_TABLE_NAME );
    auto tableAfterInsert = table->GetTable( selectTx,
    { 1, 2, 3 } );
    ASSERT_TRUE( tableAfterInsert->GetRowCount() == 5 );
    //very column 1
    auto columnData = tableAfterInsert->GetColumnBuffer( 1 );
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 4 );
    //very column 2
    columnData = tableAfterInsert->GetColumnBuffer( 2 );
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterInsert->GetColumnBuffer( 3 );
    ASSERT_EQ( columnData->GetString( 0 ),
            "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ),
            "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
}

TEST(UT_TestAriesUpdateNodeFirstWriterWin, notWaitForComittedUpdateAborted)
{
    SetupTestEnvirement();
    CreateCommittedTx();
    auto updateTx = AriesTransManager::GetInstance().NewTransaction();
    UpdateWithInprogress( updateTx, 100, false );
    AriesTransManager::GetInstance().EndTransaction( updateTx, TransactionStatus::ABORTED );

    //check result
    auto selectTx = AriesTransManager::GetInstance().NewTransaction();
    auto table = AriesMvccTableManager::GetInstance().getMvccTable( TEST_DB_NAME, TEST_TABLE_NAME );
    auto tableAfterInsert = table->GetTable( selectTx,
    { 1, 2, 3 } );
    ASSERT_TRUE( tableAfterInsert->GetRowCount() == 5 );
    //very column 1
    auto columnData = tableAfterInsert->GetColumnBuffer( 1 );
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 4 );
    //very column 2
    columnData = tableAfterInsert->GetColumnBuffer( 2 );
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterInsert->GetColumnBuffer( 3 );
    ASSERT_EQ( columnData->GetString( 0 ),
            "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ),
            "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
}

ARIES_UNIT_TEST_CLASS( TestAriesUpdateNode )
{
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        STRICT_MODE = true;
    }
};
ARIES_UNIT_TEST_F( TestAriesUpdateNode, decimal_invalid_values )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "");
    std::string tableName = "decimal_invalid_values";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 36 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 1 ), ( 2 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    string invalidDec1( "0.1111111111111111111111111111111" ); // 31 fraction digits
    string invalidDec2( "1111111.111111111111111111111111111111" ); // 7 ingeter digits and 30 fraction digits
    string maxValue1( "999999999999999999999999999999999999" ); // 36 ingeter digits

    sql = "update " + tableName + " set f1 = " + invalidDec1;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_SCALE );

    sql = "update " + tableName + " set f1 = " + invalidDec2;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_PRECISION );

    // string into decimal
    sql = "update " + tableName + " set f1 = '" + invalidDec1 + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_SCALE );

    sql = "update " + tableName + " set f1 = '" + invalidDec2 + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_PRECISION );

    sql = "update " + tableName + " set f1 = " + maxValue1 + " where f1 = 1";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    auto column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "2" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), maxValue1 );

    sql = "update " + tableName + " set f1 = " + maxValue1 + " where f1 = 2";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    column = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), maxValue1 );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), maxValue1 );

    STRICT_MODE = false;
}

void TestUpdateCharColumn( const string& dbName, const string& tableName )
{
    string newValue1 = "222222";

    string insertValue1( "1" );
    auto sql = "insert into " + tableName + " values ( '" + insertValue1 + "')";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" );

    // strict mode
    STRICT_MODE = true;

    sql = "update " + tableName + " set f1 = '" + newValue1 + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" ); // value is not updated

    // non strict mode
    STRICT_MODE = false;

    sql = "update " + tableName + " set f1 = '" + newValue1 + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 1 );
    ASSERT_EQ( resTable->GetColumnCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "2" ); // string is truncated

    STRICT_MODE = true;
}

ARIES_UNIT_TEST_F( TestAriesUpdateNode, not_null_char_exceed_len )
{
    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "not_null_char_exceed_len";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1) not null );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestUpdateCharColumn( dbName, tableName );
}
ARIES_UNIT_TEST_F( TestAriesUpdateNode, nullable_char_exceed_len )
{
    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "nullable_char_exceed_len";

    InitTable(dbName, tableName);
    auto sql = "create table " + tableName + "( f1 char(1));";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestUpdateCharColumn( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesUpdateNode, not_null_char_dict_exceed_len )
{
    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "not_null_char_dict_exceed_len";
    std::string dictName = "dict_TestAriesUpdateNode_not_null_char_dict_exceed_len";

    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 char(1) not null encoding bytedict as " + dictName + ");";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestUpdateCharColumn( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesUpdateNode, nullable_char_dict_exceed_len )
{
    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "nullable_char_dict_exceed_len";
    std::string dictName = "dict_TestAriesUpdateNode_nullable_char_dict_exceed_len";

    InitTable(dbName, tableName);
    string sql = "create table " + tableName + "( f1 char(1) encoding bytedict as " + dictName + ");";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    TestUpdateCharColumn( dbName, tableName );
}

ARIES_UNIT_TEST_F( TestAriesUpdateNode, int_into_decimal )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "int_into_decimal";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 3, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values( 12.3 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "update " + tableName + " set f1 = 123";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // big int to decimal
    sql = "update " + tableName + " set f1 = " + std::to_string( INT64_MAX );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    STRICT_MODE = false;
}

ARIES_UNIT_TEST_F( TestAriesUpdateNode, float_into_decimal )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "float_into_decimal";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 3, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values( 12.3 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "update " + tableName + " set f1 = 1.23";
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

ARIES_UNIT_TEST_F( TestAriesUpdateNode, string_into_decimal )
{
    // strict mode
    STRICT_MODE = true;

    std::string dbName = "test_update";
    SQLExecutor::GetInstance()->ExecuteSQL( "DROP database " + dbName, "" );
    std::string tableName = "string_into_decimal";

    InitTable( dbName, tableName );
    string sql = "create table " + tableName + "( f1 decimal( 3, 1 ) )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values( 12.3 )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    sql = "update " + tableName + " set f1 = '123'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    // big int to decimal
    sql = "update " + tableName + " set f1 = '" + std::to_string( INT64_MAX ) + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_EQ( result->GetErrorCode(), ER_WARN_DATA_OUT_OF_RANGE );

    sql = "update " + tableName + " set f1 = '1.23'";
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

END_ARIES_ENGINE_NAMESPACE
