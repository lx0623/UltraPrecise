#include <gtest/gtest.h>
#include <thread>
#include <mutex>
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "server/mysql/include/sql_class.h"
#include "../../../TestUtils.h"
#include "utils/string_util.h"

using namespace aries_engine;
using namespace std;

extern THD *createPseudoThd();
static string cwd = aries_utils::get_current_work_directory();

namespace aries_test
{

static const string TEST_DB_NAME( "test_mvcc_cache" );
static const string TEST_TABLE_NAME( "t1" );
ARIES_UNIT_TEST_CLASS( TestAriesMvccTableCache )
{
protected:
    void SetUp() override
    {
        InitTable( TEST_DB_NAME, TEST_TABLE_NAME );

        string sql( "create table " + TEST_TABLE_NAME + "( f1 int not null, f2 int not null, f3 int not null ) " );
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE(result->IsSuccess());

        auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );
        string csvPath = cwd + "/test_resources/mvcc_cache/csv/t1.csv";
        sql = "load data infile '" + csvPath + "' into table " + TEST_TABLE_NAME + " fields terminated by '|';";
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );

        auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
        ASSERT_EQ( tableTxMax, INVALID_TX_ID );
        auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
        ASSERT_EQ( tableCache, nullptr );

    }
    void TearDown() override
    {
        string sql = "drop database " + TEST_DB_NAME;
        auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
        ASSERT_TRUE(result->IsSuccess());
    }
};

ARIES_UNIT_TEST_F( TestAriesMvccTableCache, readonly_txs )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    // tx 1, get and cache column f1 
    // auto txId = AriesTransManager::GetInstance().GetNextTxId();
    auto sql = "select f1 from t1";
    auto resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, INVALID_TX_ID );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, START_TX_ID );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // tx 2, get and cache column f1 and f2
    sql = "select f1, f2 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );

    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 22 );
    ASSERT_EQ( column->GetInt32( 2 ), 33 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, INVALID_TX_ID );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, START_TX_ID );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );
}

/**
 * tx1 query, tx2 update, tx3 get
 */
ARIES_UNIT_TEST_F( TestAriesMvccTableCache, cache_and_update_1 )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    // tx 1, get and cache column f1 
    // auto txId1 = AriesTransManager::GetInstance().GetNextTxId();
    string sql = "select f1 from t1";
    auto resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, INVALID_TX_ID );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, START_TX_ID );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // tx 2, update 
    auto txId2 = AriesTransManager::GetInstance().GetNextTxId();
    // int newValue_Row0Col1 = 110;
    // sql = "update " + TEST_TABLE_NAME + " set f2 = " + std::to_string( newValue_Row0Col1 ) +
    //       " where f1=1";
    sql = "insert into " + TEST_TABLE_NAME + " values ( 4, 44, 444 )";
    ExecuteSQL( sql, TEST_DB_NAME );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, txId2 + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, START_TX_ID );
    ASSERT_EQ( tableCache->m_xMax, txId2 + 1 );

    // tx 3, get column f1 
    // auto txId3 = AriesTransManager::GetInstance().GetNextTxId();
    sql = "select f1 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, txId2 + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, txId2 + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );
}

class RateupConn;
using RateupConnSPtr = shared_ptr< RateupConn >;
class RateupConn
{
public:
    // template<typename _Callable, typename... _Args>
    //   explicit
    // RateupConn(_Callable&& __f, _Args&&... __args)
    // {
    //     m_thread = new thread( std::forward<_Callable>(__f), std::forward<_Args>(__args )... );
    // }
    static RateupConnSPtr NewConn()
    {
        RateupConnSPtr conn;
        auto connPtr = new RateupConn();
        conn.reset( connPtr );
        conn->run();
        return conn;
    }
    ~RateupConn()
    {
        if ( m_thread )
            delete m_thread;
        if ( m_thd )
            delete m_thd;
    }

    void Join()
    {
        m_thread->join();
    }

    AriesTableBlockUPtr ExecuteSql( string sql )
    {
        {
            unique_lock<mutex> lock( m_mutex );
            m_sql = sql;
            m_cond.notify_all();
        }
        {
            cout << "connection " << m_tx->GetTxId() << " waiting for sql to end" << endl;
            unique_lock<mutex> lock( m_mutex );
            m_cond.wait( lock,
                         [ & ] { return m_sql.empty(); } );
        }
        return std::move( m_resultTable );
    }

    THD* GetThd() const { return m_thd; }
    AriesTransactionPtr GetTx() const { return m_tx; }

private:
    RateupConn() : m_thd( nullptr ), m_sql( "" ), m_ready( false )
    {
    }

    void run()
    {
        m_thread = new thread( RateupConn::thread_proc, this );
        WaitForReady();
    }

    void Prepare()
    {
        m_thd = createPseudoThd();
        m_tx = AriesTransManager::GetInstance().NewTransaction();
        m_thd->m_tx = m_tx;
        unique_lock<mutex> lock( m_mutex );
        m_ready = true;
        cout << "tx " << m_thd->m_tx->GetTxId() << " is ready" << endl;
        m_cond.notify_all();
    }

    void WaitForReady()
    {
        unique_lock<mutex> lock( m_mutex );
        m_cond.wait( lock,
                     [ & ] { return m_ready; } );
        cout << "tx " << GetTx()->GetTxId() << " is running" << endl;
    }

    bool _executeSql()
    {
        unique_lock<mutex> lock( m_mutex );
        cout << "tx " << m_thd->m_tx->GetTxId() << " waiting for sql" << endl;
        m_cond.wait( lock,
                     [ & ] { return !m_sql.empty(); } );
        cout << "tx " << m_thd->m_tx->GetTxId() << " executing sql " << m_sql << endl;

        m_resultTable = aries_test::ExecuteSQL( m_sql, TEST_DB_NAME );
        m_sql.clear();
        cout << "tx " << GetTx()->GetTxId() << " sql " << m_sql << "ended" << endl;
        m_cond.notify_all();

        return true;
    }

    static void thread_proc( RateupConn* conn )
    {
        conn->Prepare();
        conn->_executeSql();
        cout << "tx " << conn->GetTx()->GetTxId() << " ended" << endl;
    }

private:
    thread* m_thread;
    THD* m_thd;
    aries_engine::AriesTransactionPtr m_tx;
    string m_sql;
    bool m_ready;

    AriesTableBlockUPtr m_resultTable;
    // SQLResultPtr m_sqlResult;
    mutex m_mutex;
    condition_variable m_cond;
};


/**
 * test cases:
 * 
 * test0:
 * tx1 begin and query, create cache: [ start txId, max_id )
 * tx2 begin and update and commit, set tableTxMax: tx2Id + 1, update cache info:[ start txId, tx2Id + 1 )
 * tx3 begin and query, update cache: [ tx2Id + 1, max_id )
 * 
 * test1:
 * 1. RF tx1 update and commit, tableTxMax: tx1Id + 1
 * 2. tx2 begin and query, create cache: [ tx1Id + 1, max_id )
 * 3. tx3 begin and query, use cache: [ tx1Id + 1, max_id )
 * 
 * test2:
 * 1. RF tx1 begin and update and commit, tableTxMax: tx1Id + 1
 * 2. tx2 begin,
 * 3. tx3 begin and query, create cache: [ tx1Id + 1, max_id )
 * 4. tx2 query, use cache 
 * 
 * test3:
 * 1. RF tx1 begin
 * 2. tx2 begin
 * 3. RF tx1 update and commit, tableTxMax: tx2Id + 1
 * 4. tx3 begin
 * 5. tx2 query, cache not created
 * 6. tx3 query, create cache: [ tx2Id + 1, max_id )
 * 
 * test4:
 * 1. RF tx1 begin
 * 2. tx2 begin
 * 3. RF tx1 update and commit, tableTxMax: tx2Id + 1
 * 4. tx3 begin and query, create cache: [ tx2Id + 1, max_id )
 * 5. tx2 query, cache not usable
 * 6. tx4 begin and query, use cache
 * 7. tx5 begin
 * 8. RF tx6 begin
 * 9. tx7 begin
 * 10. RF tx6 update and commit, tableTxMax: tx8Id, update cache info: [ tx2Id, tx8Id )
 * 11. tx8 begin and query, create new cache: [ tx8Id, max_id )
 * 12. tx9 begin
 * 13. RF tx10 begin
 * 14. tx11 begin
 * 15. RF tx10 update and commit, tableTxMax: tx12Id, update cache info: [ tx8Id, tx12Id )
 * 16. tx5 query, cache not usable, don't create new cache
 * 17. tx7 query, cache not usable, don't create new cache
 * 18. tx9 query, use cache
 */

/**
 * tx1 begin and query, create cache: [ start txId, max_id )
 * tx2 begin and update and commit, set tableTxMax: tx2Id + 1, update cache info:[ start txId, tx2Id + 1 )
 * tx3 begin and query, update cache: [ tx2Id + 1, max_id )
 */
ARIES_UNIT_TEST_F( TestAriesMvccTableCache, test0 )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    auto nextTxId = AriesTransManager::GetInstance().GetNextTxId();
    cout << "next tx id: " << nextTxId << endl;

    /***********************************************************************
    1. conn1 query
    ************************************************************************/
    auto conn1 = RateupConn::NewConn();

    cout << "----------------step 1. tx1 query\n";
    string sql = "select f1 from " + TEST_TABLE_NAME;
    auto resultTable = conn1->ExecuteSql( sql );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, INVALID_TX_ID );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, START_TX_ID );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    /***********************************************************************
    tx 2, update 
    ************************************************************************/
    auto conn2 = RateupConn::NewConn();

    cout << "----------------step 2. tx2 update\n";
    // int newValue_Row0Col1 = 110;
    // sql = "update " + TEST_TABLE_NAME + " set f2 = " + std::to_string( newValue_Row0Col1 ) +
    //       " where f1=1";
    sql = "insert into " + TEST_TABLE_NAME + " values ( 4, 44, 444 )";
    conn2->ExecuteSql( sql );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, START_TX_ID );
    ASSERT_EQ( tableCache->m_xMax, conn2->GetTx()->GetTxId() + 1 );

    // tx 3, get column f1 
    auto conn3 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn3->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, conn2->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    conn1->Join();
    conn2->Join();
    conn3->Join();
}

/**
 * 1. RF tx1 update and commit, tableTxMax: tx1Id + 1
 * 2. tx2 begin and query, create cache: [ tx1Id + 1, max_id )
 * 3. tx3 begin and query, use cache: [ tx1Id + 1, max_id )
 */
ARIES_UNIT_TEST_F( TestAriesMvccTableCache, concurrent_test1 )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    auto nextTxId = AriesTransManager::GetInstance().GetNextTxId();
    cout << "next tx id: " << nextTxId << endl;

    /***********************************************************************
    1. tx1 update
    ************************************************************************/
    cout << "----------------step 1. RF tx1 update\n";
    auto conn1 = RateupConn::NewConn();

    string sql = "insert into " + TEST_TABLE_NAME + " values ( 4, 44, 444 )";
    conn1->ExecuteSql( sql );

    ASSERT_EQ( conn1->GetTx()->GetTxInfo()->GetStatus(), TransactionStatus::COMMITTED );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn1->GetTx()->GetTxId() + 1 );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr == tableCache );

    /***********************************************************************
    2. tx2 query
    ************************************************************************/
    cout << "----------------step 2. tx2 query\n";

    auto conn2 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    auto resultTable = conn2->ExecuteSql( sql );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn1->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, conn1->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    /***********************************************************************
    3. tx3 query
    ************************************************************************/
    cout << "----------------step 3. tx3 query\n";

    auto conn3 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn3->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn1->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, conn1->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    conn1->Join();
    conn2->Join();
    conn3->Join();
}

/**
 * 1. RF tx1 begin and update and commit, tableTxMax: tx1Id + 1
 * 2. tx2 begin,
 * 3. tx3 begin and query, create cache: [ tx1Id + 1, max_id )
 * 4. tx2 query, use cache 
 */
ARIES_UNIT_TEST_F( TestAriesMvccTableCache, concurrent_test2 )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    auto nextTxId = AriesTransManager::GetInstance().GetNextTxId();
    cout << "next tx id: " << nextTxId << endl;

    /***********************************************************************
    1. RF tx1 update
    ************************************************************************/
    cout << "----------------step 1. RF tx1 update\n";
    auto conn1 = RateupConn::NewConn();

    string sql = "insert into " + TEST_TABLE_NAME + " values ( 4, 44, 444 )";
    conn1->ExecuteSql( sql );

    ASSERT_EQ( conn1->GetTx()->GetTxInfo()->GetStatus(), TransactionStatus::COMMITTED );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn1->GetTx()->GetTxId() + 1 );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr == tableCache );

    /***********************************************************************
    2. tx2 begin
    ************************************************************************/
    cout << "----------------step 2. tx2 begin\n";
    auto conn2 = RateupConn::NewConn();

    /***********************************************************************
    3. tx3 begin and query
    ************************************************************************/
    cout << "----------------step 3. tx3 begin and execute query\n";
    auto conn3 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    auto resultTable = conn3->ExecuteSql( sql );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn1->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, conn1->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    /***********************************************************************
    4. tx2 execute query
    ************************************************************************/
    cout << "----------------step 4. tx2 execute query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn2->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn1->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );
    ASSERT_EQ( tableCache->m_xMin, conn1->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    conn1->Join();
    conn2->Join();
    conn3->Join();
}

/**
 * 1. RF tx1 begin
 * 2. tx2 begin
 * 3. RF tx1 update and commit, tableTxMax: tx2Id + 1
 * 4. tx2 query, cache not usable
 * 5. tx3 begin and query, create cache: [ tx2Id + 1, max_id )
 */
ARIES_UNIT_TEST_F( TestAriesMvccTableCache, concurrent_test3 )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    auto nextTxId = AriesTransManager::GetInstance().GetNextTxId();
    cout << "next tx id: " << nextTxId << endl;

    string sql;

    /***********************************************************************
    1. RF tx1 begin
    ************************************************************************/
    cout << "----------------step 1. RF tx1 begin\n";
    auto conn1 = RateupConn::NewConn();

    /***********************************************************************
    2. tx2 begin
    ************************************************************************/
    cout << "----------------step 2. tx2 begin\n";
    auto conn2 = RateupConn::NewConn();

    /***********************************************************************
    3. RF tx1 update and commit
    ************************************************************************/
    cout << "----------------step 3. RF tx1 update\n";
    sql = "insert into " + TEST_TABLE_NAME + " values ( 4, 44, 444 )";
    conn1->ExecuteSql( sql );

    ASSERT_EQ( conn1->GetTx()->GetTxInfo()->GetStatus(), TransactionStatus::COMMITTED );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr == tableCache );

    /***********************************************************************
    4. tx3 begin
    ************************************************************************/
    cout << "----------------step 4. tx3 begin\n";
    auto conn3 = RateupConn::NewConn();

    /***********************************************************************
    5. tx2 query
    ************************************************************************/
    cout << "----------------step 5. tx2 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    auto resultTable = conn2->ExecuteSql( sql );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    // cache is not created
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr == tableCache );

    /***********************************************************************
    6. tx3 query
    ************************************************************************/
    cout << "----------------step 6. tx3 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn3->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    conn1->Join();
    conn2->Join();
    conn3->Join();
}

/**
 * 1. RF tx1 begin
 * 2. tx2 begin
 * 3. RF tx1 update and commit, tableTxMax: tx2Id + 1
 * 4. tx3 begin and query, create cache: [ tx2Id + 1, max_id )
 * 5. tx2 query, cache not usable
 * 6. tx4 begin and query, use cache
 * 7. tx5 begin
 * 8. RF tx6 begin
 * 9. tx7 begin
 * 10. RF tx6 update and commit, tableTxMax: tx8Id, update cache info: [ tx2Id, tx8Id )
 * 11. tx8 begin and query, create new cache: [ tx8Id, max_id )
 * 12. tx9 begin
 * 13. RF tx10 begin
 * 14. tx11 begin
 * 15. RF tx10 update and commit, tableTxMax: tx12Id, update cache info: [ tx8Id, tx12Id )
 * 16. tx5 query, cache not usable, don't create new cache
 * 17. tx7 query, cache not usable, don't create new cache
 * 18. tx9 query, use cache
 * 19. tx11 query, use cache
 * 20. tx12 begin and query, cache not usable, create new cache: [ tx12Id, max_id )
 */
ARIES_UNIT_TEST_F( TestAriesMvccTableCache, concurrent_test4 )
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );

    auto nextTxId = AriesTransManager::GetInstance().GetNextTxId();
    cout << "next tx id: " << nextTxId << endl;

    string sql;

    /***********************************************************************
    1. RF tx1 begin
    ************************************************************************/
    cout << "----------------step 1. RF tx1 begin\n";
    auto conn1 = RateupConn::NewConn();

    /***********************************************************************
    2. tx2 begin
    ************************************************************************/
    cout << "----------------step 2. tx2 begin\n";
    auto conn2 = RateupConn::NewConn();

    /***********************************************************************
    3. RF tx1 update and commit
    ************************************************************************/
    cout << "----------------step 3. RF tx1 update\n";
    sql = "insert into " + TEST_TABLE_NAME + " values ( 4, 44, 444 )";
    conn1->ExecuteSql( sql );

    ASSERT_EQ( conn1->GetTx()->GetTxInfo()->GetStatus(), TransactionStatus::COMMITTED );

    // verify cache info
    auto tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr == tableCache );

    /***********************************************************************
    4. tx3 begin and query
    ************************************************************************/
    cout << "----------------step 4. tx3 begin and query\n";
    auto conn3 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    auto resultTable = conn3->ExecuteSql( sql );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_TRUE( nullptr != tableCache );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    /***********************************************************************
    5. tx2 query
    ************************************************************************/
    cout << "----------------step 5. tx2 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn2->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );

    // cache is not changed
    auto tx2TableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tx2TableCache, tableCache );

    /***********************************************************************
    6. tx4 begin and query
    ************************************************************************/
    cout << "----------------step 6. tx4 begin and query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    auto conn4 = RateupConn::NewConn();
    resultTable = conn4->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn2->GetTx()->GetTxId() + 1 );
    // cache is not changed
    auto tx4TableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tx4TableCache, tableCache );

    /***********************************************************************
    7. tx5 begin
    ************************************************************************/
    cout << "----------------step 5. tx5 begin\n";
    auto conn5 = RateupConn::NewConn();

    /***********************************************************************
    8. RF tx6 begin
    ************************************************************************/
    cout << "----------------step 8. RF tx6 begin\n";
    auto conn6 = RateupConn::NewConn();

    /***********************************************************************
    9. tx7 begin
    ************************************************************************/
    cout << "----------------step 9. tx7 begin\n";
    auto conn7 = RateupConn::NewConn();

    /***********************************************************************
    10. RF tx6 udpate and commit
    ************************************************************************/
    cout << "----------------step 10. RF tx6 update and commit\n";
    sql = "insert into " + TEST_TABLE_NAME + " values ( 5, 55, 555 )";
    conn6->ExecuteSql( sql );

    ASSERT_EQ( conn6->GetTx()->GetTxInfo()->GetStatus(), TransactionStatus::COMMITTED );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn7->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache->m_xMin, conn2->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, conn7->GetTx()->GetTxId() + 1 );

    /***********************************************************************
    11. tx8 begin and query
    ************************************************************************/
    cout << "----------------step 11. tx8 begin and query\n";
    auto conn8 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn8->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 5 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn7->GetTx()->GetTxId() + 1 );
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache->m_xMin, conn7->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    /***********************************************************************
    12. tx9 begin
    ************************************************************************/
    cout << "----------------step 12. tx9 begin\n";
    auto conn9 = RateupConn::NewConn();

    /***********************************************************************
    13. RF tx10 begin
    ************************************************************************/
    cout << "----------------step 13. RF tx10 begin\n";
    auto conn10 = RateupConn::NewConn();

    /***********************************************************************
    14. tx11 begin
    ************************************************************************/
    cout << "----------------step 14. tx11 begin\n";
    auto conn11 = RateupConn::NewConn();

    /***********************************************************************
    15. RF tx10 update and commit
    ************************************************************************/
    cout << "----------------step 15. RF tx10 update and commit\n";
    sql = "insert into " + TEST_TABLE_NAME + " values ( 6, 66, 666 )";
    conn10->ExecuteSql( sql );

    ASSERT_EQ( conn10->GetTx()->GetTxInfo()->GetStatus(), TransactionStatus::COMMITTED );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn11->GetTx()->GetTxId() + 1 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache->m_xMin, conn8->GetTx()->GetTxId() );
    ASSERT_EQ( tableCache->m_xMax, conn11->GetTx()->GetTxId() + 1 );

    /***********************************************************************
    16. tx5 query
    ************************************************************************/
    cout << "----------------step 16. tx5 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn5->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn11->GetTx()->GetTxId() + 1 );
    auto tx5TableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tx5TableCache, tableCache );

    /***********************************************************************
    17. tx7 query
    ************************************************************************/
    cout << "----------------step 17. tx7 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn7->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn11->GetTx()->GetTxId() + 1 );
    auto tx7TableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tx7TableCache, tableCache );

    /***********************************************************************
    18. tx9 query
    ************************************************************************/
    cout << "----------------step 18. tx9 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn9->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 5 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn11->GetTx()->GetTxId() + 1 );
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache->m_xMin, conn7->GetTx()->GetTxId() + 1 );
    ASSERT_EQ( tableCache->m_xMax, conn11->GetTx()->GetTxId() + 1 );

    /***********************************************************************
    19. tx11 query
    ************************************************************************/
    cout << "----------------step 19. tx11 query\n";
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn11->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 5 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn11->GetTx()->GetTxId() + 1 );
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache->m_xMin, conn8->GetTx()->GetTxId() );
    ASSERT_EQ( tableCache->m_xMax, conn11->GetTx()->GetTxId() + 1 );

    /***********************************************************************
    20. tx12 begin and query
    ************************************************************************/
    cout << "----------------step 20. tx12 begin and query\n";
    auto conn12 = RateupConn::NewConn();
    sql = "select f1 from " + TEST_TABLE_NAME;
    resultTable = conn12->ExecuteSql( sql );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 6 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );
    ASSERT_EQ( column->GetInt32( 4 ), 5 );
    ASSERT_EQ( column->GetInt32( 5 ), 6 );

    // verify cache info
    tableTxMax = AriesMvccTableManager::GetInstance().getTableTxMax( tableId );
    ASSERT_EQ( tableTxMax, conn11->GetTx()->GetTxId() + 1 );
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache->m_xMin, conn12->GetTx()->GetTxId() );
    ASSERT_EQ( tableCache->m_xMax, INT32_MAX );

    conn1->Join();
    conn2->Join();
    conn3->Join();
    conn4->Join();
    conn5->Join();
    conn6->Join();
    conn7->Join();
    conn8->Join();
    conn9->Join();
    conn10->Join();
    conn11->Join();
    conn12->Join();
}

ARIES_UNIT_TEST_F( TestAriesMvccTableCache, readonly_txs2 )
{
    InitTable( TEST_DB_NAME, "t1" );
    string sql( "create table t1 ( f1 int not null, f2 int not null, f3 char( 64 ) not null ) " );
    ExecuteSQL( sql, TEST_DB_NAME );

    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );
    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    sql = "insert into t1 values ( 1, 11, \"aaa\" ), ( 2, 22, \"bbb\" ), ( 3, 33, \"ccc\" )";
    ExecuteSQL( sql, TEST_DB_NAME );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    // 1. cache column f1 
    sql = "select f1 from t1";
    auto resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // test again
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 1 );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // 2. cache column f2 
    sql = "select f1, f2 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );

    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 22 );
    ASSERT_EQ( column->GetInt32( 2 ), 33 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 2 );

    // verify that column 1 and 2 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // 3. cache column f3 
    sql = "select f3 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );
    ASSERT_EQ( column->GetString( 2 ), "ccc" );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 2 );

    // verify that column 1 and 2 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );
}

ARIES_UNIT_TEST_F( TestAriesMvccTableCache, readwrite_one_tx )
{
    InitTable( TEST_DB_NAME, "t1" );
    string sql( "create table t1 ( f1 int not null, f2 int not null, f3 char( 64 ) not null ) " );
    ExecuteSQL( sql, TEST_DB_NAME );

    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );
    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    sql = "insert into t1 values ( 1, 11, \"aaa\" ), ( 2, 22, \"bbb\" ), ( 3, 33, \"ccc\" )";
    ExecuteSQL( sql, TEST_DB_NAME );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    sql = "start transaction";
    ExecuteSQL( sql, TEST_DB_NAME );

    // 1. cache column f1 
    sql = "select f1 from t1";
    auto resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 3 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // 2. cache column f2 
    sql = "select f1, f2 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 3 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 22 );
    ASSERT_EQ( column->GetInt32( 2 ), 33 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 1 );

    // verify that column 1 and 2 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    // 3. insert
    sql = "insert into t1 values ( 4, 44, \"ddd\")";
    ExecuteSQL( sql, TEST_DB_NAME );

    // 4. select again
    sql = "select f1, f2 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 22 );
    ASSERT_EQ( column->GetInt32( 2 ), 33 );
    ASSERT_EQ( column->GetInt32( 3 ), 44 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 1 ); // cache is not used

    // verify that cache is not touched
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    sql = "commit";
    ExecuteSQL( sql, TEST_DB_NAME );
}

ARIES_UNIT_TEST_F( TestAriesMvccTableCache, read_and_write_txs )
{
    InitTable( TEST_DB_NAME, "t1" );
    string sql( "create table t1 ( f1 int not null, f2 int not null, f3 char( 64 ) not null ) " );
    ExecuteSQL( sql, TEST_DB_NAME );

    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );
    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    sql = "insert into t1 values ( 1, 11, \"aaa\" ), ( 2, 22, \"bbb\" ), ( 3, 33, \"ccc\" )";
    ExecuteSQL( sql, TEST_DB_NAME );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    //////////////////////////////////////////////////////////
    // tx1.0: new
    //////////////////////////////////////////////////////////
    sql = "start transaction";
    ExecuteSQL( sql, TEST_DB_NAME );

    //////////////////////////////////////////////////////////
    // tx1.1 cache column f1 
    //////////////////////////////////////////////////////////
    sql = "select f1 from t1";
    auto resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    auto columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );

    auto column = resultTable->GetColumnBuffer( 1 );
    auto rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 3 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    //////////////////////////////////////////////////////////
    // tx1.2 insert new data
    //////////////////////////////////////////////////////////
    sql = "insert into t1 values ( 4, 44, \"ddd\")";
    ExecuteSQL( sql, TEST_DB_NAME );

    //////////////////////////////////////////////////////////
    // tx1.3 select
    //////////////////////////////////////////////////////////
    sql = "select f1, f2, f3 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 2 );
    ASSERT_EQ( column->GetInt32( 2 ), 3 );
    ASSERT_EQ( column->GetInt32( 3 ), 4 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 22 );
    ASSERT_EQ( column->GetInt32( 2 ), 33 );
    ASSERT_EQ( column->GetInt32( 3 ), 44 );

    column = resultTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );
    ASSERT_EQ( column->GetString( 2 ), "ccc" );
    ASSERT_EQ( column->GetString( 3 ), "ddd" );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 ); // cache is not used

    // verify that cache is not touched
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

    //////////////////////////////////////////////////////////
    // tx2.1 start a new transaction that do select
    //////////////////////////////////////////////////////////
    thread t1( [ & ]
    {
        THD *thd = createPseudoThd();
        sql = "select f1, f2 from t1";
        resultTable = ExecuteSQL( sql, TEST_DB_NAME );
        columnCount = resultTable->GetColumnCount();
        ASSERT_EQ( columnCount, 2 );

        // new data inserted in step tx1.2 is not visible
        column = resultTable->GetColumnBuffer( 1 );
        rowCount = column->GetItemCount();
        ASSERT_EQ( rowCount, 3 );

        ASSERT_EQ( column->GetInt32( 0 ), 1 );
        ASSERT_EQ( column->GetInt32( 1 ), 2 );
        ASSERT_EQ( column->GetInt32( 2 ), 3 );

        column = resultTable->GetColumnBuffer( 2 );
        ASSERT_EQ( column->GetInt32( 0 ), 11 );
        ASSERT_EQ( column->GetInt32( 1 ), 22 );
        ASSERT_EQ( column->GetInt32( 2 ), 33 );

        tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
        ASSERT_NE( tableCache, nullptr );
        // ASSERT_EQ( tableCache->m_hitCount, 1 ); // cache is used

        // verify that column 1 and 2 is cached
        ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
        ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
        ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );

        delete thd;
    });

    t1.join();

    //////////////////////////////////////////////////////////
    // tx1.4: commit
    //////////////////////////////////////////////////////////
    sql = "commit";
    ExecuteSQL( sql, TEST_DB_NAME );

    // verify that cache is deleted
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    //ASSERT_EQ( tableCache, nullptr );

    //////////////////////////////////////////////////////////
    // tx3. start a new transaction that do select
    //////////////////////////////////////////////////////////
    sql = "select f2, f3 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );

    // new data inserted in step tx1.2 is now visible
    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 4 );

    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 22 );
    ASSERT_EQ( column->GetInt32( 2 ), 33 );
    ASSERT_EQ( column->GetInt32( 3 ), 44 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "bbb" );
    ASSERT_EQ( column->GetString( 2 ), "ccc" );
    ASSERT_EQ( column->GetString( 3 ), "ddd" );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 ); // new cache is created

    // verify that column 1 and 2 is cached
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );

    //////////////////////////////////////////////////////////
    // tx4.0: new
    //////////////////////////////////////////////////////////
    sql = "start transaction";
    ExecuteSQL( sql, TEST_DB_NAME );

    //////////////////////////////////////////////////////////
    // tx4.1 delete
    //////////////////////////////////////////////////////////
    sql = "delete from t1 where f1 in ( 2, 4 )";
    ExecuteSQL( sql, TEST_DB_NAME );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 );

    // verify that column 1, 2, 3 is cached
    //ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) ); // column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );

    //////////////////////////////////////////////////////////
    // tx4.2 select
    //////////////////////////////////////////////////////////
    sql = "select f1, f2, f3 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 2 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 3 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 33 );

    column = resultTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "ccc" );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 ); // cache is not used

    // verify that cache is not touched
    //ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );

    // new tx
    thread t2( [ & ]
    {
        THD *thd = createPseudoThd();

        //////////////////////////////////////////////////////////
        // tx5.0: new tx
        //////////////////////////////////////////////////////////
        sql = "start transaction";
        ExecuteSQL( sql, TEST_DB_NAME );

        //////////////////////////////////////////////////////////
        // tx5.1 select
        //////////////////////////////////////////////////////////
        sql = "select f2, f3 from t1";
        resultTable = ExecuteSQL( sql, TEST_DB_NAME );
        columnCount = resultTable->GetColumnCount();
        ASSERT_EQ( columnCount, 2 );

        column = resultTable->GetColumnBuffer( 1 );
        rowCount = column->GetItemCount();
        ASSERT_EQ( rowCount, 4 ); // rows delete in step tx4.1 is not visible to this tx

        ASSERT_EQ( column->GetInt32( 0 ), 11 );
        ASSERT_EQ( column->GetInt32( 1 ), 22 );
        ASSERT_EQ( column->GetInt32( 2 ), 33 );
        ASSERT_EQ( column->GetInt32( 3 ), 44 );

        column = resultTable->GetColumnBuffer( 2 );
        ASSERT_EQ( column->GetString( 0 ), "aaa" );
        ASSERT_EQ( column->GetString( 1 ), "bbb" );
        ASSERT_EQ( column->GetString( 2 ), "ccc" );
        ASSERT_EQ( column->GetString( 3 ), "ddd" );

        tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
        ASSERT_NE( tableCache, nullptr );
        // ASSERT_EQ( tableCache->m_hitCount, 1 ); // cache is used

        // verify that column 1, 2  and 3 is cached
//        ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
        ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
        ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );

        //////////////////////////////////////////////////////////
        // tx5.2 update
        //////////////////////////////////////////////////////////
        sql = "update t1 set f3 = \"ccc_c\" where f1 = 3";
        ExecuteSQL( sql, TEST_DB_NAME );
        //////////////////////////////////////////////////////////

        // tx5.3 select
        //////////////////////////////////////////////////////////
        sql = "select f1, f3 from t1 order by f1";
        resultTable = ExecuteSQL( sql, TEST_DB_NAME );
        columnCount = resultTable->GetColumnCount();
        ASSERT_EQ( columnCount, 2 );

        column = resultTable->GetColumnBuffer( 1 );
        rowCount = column->GetItemCount();
        ASSERT_EQ( rowCount, 4 ); // rows delete in step tx4.1 is not visible to this tx

        ASSERT_EQ( column->GetInt32( 0 ), 1 );
        ASSERT_EQ( column->GetInt32( 1 ), 2 );
        ASSERT_EQ( column->GetInt32( 2 ), 3 );
        ASSERT_EQ( column->GetInt32( 3 ), 4 );

        column = resultTable->GetColumnBuffer( 2 );
        ASSERT_EQ( column->GetString( 0 ), "aaa" );
        ASSERT_EQ( column->GetString( 1 ), "bbb" );
        ASSERT_EQ( column->GetString( 2 ), "ccc_c" ); // row is udpated
        ASSERT_EQ( column->GetString( 3 ), "ddd" );

        tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
        ASSERT_NE( tableCache, nullptr );
        // ASSERT_EQ( tableCache->m_hitCount, 2 ); // cache is used

        // verify that column 1, 2  and 3 is cached
//        ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
        ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
        ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );

        //////////////////////////////////////////////////////////
        // tx5.4: commit
        //////////////////////////////////////////////////////////
        sql = "commit";
        ExecuteSQL( sql, TEST_DB_NAME );

        delete thd;
    });

    t2.join();
    // verify that cache is deleted
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    //ASSERT_EQ( tableCache, nullptr );

    //////////////////////////////////////////////////////////
    // tx4.3 select
    //////////////////////////////////////////////////////////
    sql = "select f1, f2, f3 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 2 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 3 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 33 );

    column = resultTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "ccc" ); // update in step tx5.2 is still invisible 

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
//    ASSERT_EQ( tableCache, nullptr );

    //////////////////////////////////////////////////////////
    // tx4.4 commit
    //////////////////////////////////////////////////////////
    sql = "commit";
    ExecuteSQL( sql, TEST_DB_NAME );

    // verify that cache is deleted
    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
//    ASSERT_EQ( tableCache, nullptr );

    //////////////////////////////////////////////////////////
    // tx6 start a new transaction that do select
    //////////////////////////////////////////////////////////
    sql = "select f1, f2, f3 from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    columnCount = resultTable->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );

    column = resultTable->GetColumnBuffer( 1 );
    rowCount = column->GetItemCount();
    ASSERT_EQ( rowCount, 2 ); // delete tx is committed and is visible now

    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 3 );

    column = resultTable->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 11 );
    ASSERT_EQ( column->GetInt32( 1 ), 33 );

    column = resultTable->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "aaa" );
    ASSERT_EQ( column->GetString( 1 ), "ccc_c" ); // update in step tx5.2 is visible

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );
    // ASSERT_EQ( tableCache->m_hitCount, 0 ); // new cache is created

    // verify that column 1, 2 and 3 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 3 ) );
}

ARIES_UNIT_TEST_F( TestAriesMvccTableCache, delete_txs )
{
    InitTable( TEST_DB_NAME, "t1" );
    string sql( "create table t1 ( f1 int not null, f2 int not null, f3 char( 64 ) not null ) " );
    ExecuteSQL( sql, TEST_DB_NAME );

    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( TEST_DB_NAME, TEST_TABLE_NAME );
    auto tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    sql = "insert into t1 values ( 1, 11, \"aaa\" ), ( 2, 22, \"bbb\" ), ( 3, 33, \"ccc\" )";
    ExecuteSQL( sql, TEST_DB_NAME );

    sql = "delete from t1 where f1 = 2";
    ExecuteSQL( sql, TEST_DB_NAME );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_EQ( tableCache, nullptr );

    // 1. cache column f1
    sql = "select f1 from t1";
    auto resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( resultTable->GetColumnCount(), 1 );
    ASSERT_EQ( resultTable->GetRowCount(), 2 );
    ASSERT_EQ( resultTable->GetAllMaterilizedColumnIds().size(), 1 );
    auto column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 3 );

    tableCache = AriesMvccTableManager::GetInstance().getCache( tableId );
    ASSERT_NE( tableCache, nullptr );

    // verify that only column 1 is cached
    ASSERT_TRUE( tableCache->m_table->ColumnExists( 1 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 2 ) );
    ASSERT_TRUE( !tableCache->m_table->ColumnExists( 3 ) );


    //////////////////////////////////////////////////////////
    // tx2.1 start a new transaction that do delete
    //////////////////////////////////////////////////////////
    thread t1( [ & ]
    {
        THD *thd = createPseudoThd();
        string tmpSql = "delete from t1 where f1 = 1";
        ExecuteSQL( tmpSql, TEST_DB_NAME );
        sleep( 5 );

        delete thd;
    });

    // cache all column can't see the delete
    sql = "select * from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( resultTable->GetColumnCount(), 3 );
    ASSERT_EQ( resultTable->GetRowCount(), 2 );
    ASSERT_EQ( resultTable->GetAllMaterilizedColumnIds().size(), 3 );
    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 3 );
    t1.join();

    // cache all column can see the delete
    sql = "select * from t1";
    resultTable = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( resultTable->GetColumnCount(), 3 );
    ASSERT_EQ( resultTable->GetRowCount(), 1 );
    ASSERT_EQ( resultTable->GetAllMaterilizedColumnIds().size(), 3 );
    column = resultTable->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 3 );
}

}
