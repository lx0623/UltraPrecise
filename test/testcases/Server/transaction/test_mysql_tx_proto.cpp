#include <gtest/gtest.h>
#include <vector>
#include "server/mysql/include/sql_class.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngine/AriesUtil.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "AriesEngineWrapper/AriesMemTable.h"

#include "../../../TestUtils.h"

using namespace aries_engine;
using namespace aries_test;
using namespace std;
TEST(UT_tx, mysqlproto)
{
    string dbName = "test_mysqlproto";
    SQLExecutor::GetInstance()->ExecuteSQL( "drop database if exists " + dbName, "" );
    string tableName = "nation";
    InitTable( dbName, tableName );

    string sql = "create table " + tableName + "( f1 int )";
    auto results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );

    THD* thd = current_thd;
    auto& txManager = AriesTransManager::GetInstance();

    sql = "use " + dbName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    TxId txId = txManager.GetNextTxId();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // implicit tx, commit
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // implicit tx, abort
    sql = "select * from nat";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( !results->IsSuccess() );
    txId += 1;
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::ABORTED );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // explicit tx, commit
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    txId += 1;
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // normal statements
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );
    // commit
    sql = "commit";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // explicit tx, abort
    txId += 1;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // normal statements
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );
    // commit
    sql = "rollback";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::ABORTED );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // explicit tx, error( abort )
    txId += 1;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // normal statements
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // error
    sql = "select * from naton";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( !results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // implicitly commit live tx
    txId += 1;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // normal statements
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // start a new tx
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );

    // verify that the live tx is commited implicity
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );

    txId += 1;
    // verify that a new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // commit
    sql = "commit";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );

    txId += 1;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // normal statements
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    // start a new tx
    sql = "create database if not exists test_tx_mysql_proto";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );

    // verify that the live tx is commited implicity
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );

    txId += 1;
    // verify that a new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // tx end with options chain and release
    txId += 1;
    thd->killed = THD::NOT_KILLED;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    //////////////////////////////////////////////
    // commit with no chain and release
    sql = "commit and no chain release";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    //  verify that no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );
    ASSERT_EQ( thd->killed, THD::KILL_CONNECTION );

    //////////////////////////////////////////////
    // commit with chain
    txId += 1;
    thd->killed = THD::NOT_KILLED;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    sql = "commit and chain";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::COMMITTED );
    // verify that NEW TX is started
    txId += 1;
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    //////////////////////////////////////////////
    // commit with chain and release
    txId += 1;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    sql = "commit and chain release";
    thd->reset_for_next_command();
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_EQ( results->GetErrorCode(), ER_SYNTAX_ERROR );
    // no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 

    //////////////////////////////////////////////
    // commit with no chain and no release
    txId += 1;
    thd->killed = THD::NOT_KILLED;
    sql = "start transaction";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::IN_PROGRESS );

    sql = "rollback and no chain no release";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( thd->killed, THD::NOT_KILLED );
    // no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    ASSERT_EQ( txManager.GetTxStatus( txId ),TransactionStatus::ABORTED );

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    // unexpected command
    sql = "commit";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 

    sql = "rollback";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    // verify no new tx is started
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 


    //set autocommit=1会自动提交
    txId += 1;
    sql = "set autocommit=1";
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 1 ); 
    //begin语句会结束当前连接正在进行中的transaction，并新开启一个transaction。
    results = SQLExecutor::GetInstance()->ExecuteSQL( "begin", dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 2 ); 
    // 在begin显式开启事务后，执行普通select语句不会自动提交
    sql = "select * from " + tableName;
    results = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( results->IsSuccess() );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 2 ); 
    // 在begin显式开启事务后,执行普通select语句失败后不会自动结束事务
    results = SQLExecutor::GetInstance()->ExecuteSQL( "select * from aa.aa", dbName );
    ASSERT_FALSE(results->IsSuccess());
    ASSERT_EQ( txManager.GetNextTxId(), txId + 2 ); 
    // 在begin显式开启事务后,执行command语句(create,drop),会自动结束事务
    SQLExecutor::GetInstance()->ExecuteSQL( "drop table if exists QWERTY", dbName );
    results = SQLExecutor::GetInstance()->ExecuteSQL( "create table QWERTY(a int)", dbName );
    ASSERT_TRUE(results->IsSuccess());
    ASSERT_EQ( txManager.GetNextTxId(), txId + 4 ); 
    // 在begin显式开启事务后,执行command语句(create,drop)失败后也会自动结束事务
    txId = txManager.GetNextTxId();
    SQLExecutor::GetInstance()->ExecuteSQL( "begin", dbName );
    results = SQLExecutor::GetInstance()->ExecuteSQL( "create table QWERTY(a int)", dbName );
    ASSERT_FALSE(results->IsSuccess());
    ASSERT_EQ( txManager.GetTxStatus( txId+1 ),TransactionStatus::ABORTED );
    ASSERT_EQ( txManager.GetNextTxId(), txId + 2 ); 
    SQLExecutor::GetInstance()->ExecuteSQL( "drop table if exists QWERTY", dbName );

    // 当前session的autocommit从0变为1时，会自动提交还在进行中事务
    SQLExecutor::GetInstance()->ExecuteSQL("set autocommit=1", dbName);
    SQLExecutor::GetInstance()->ExecuteSQL("set autocommit=0", dbName);
    SQLExecutor::GetInstance()->ExecuteSQL( "begin", dbName );
    SQLExecutor::GetInstance()->ExecuteSQL( "drop table if exists tt", dbName );
    SQLExecutor::GetInstance()->ExecuteSQL( "create table tt(a int)", dbName );
    txId = txManager.GetNextTxId();
    SQLExecutor::GetInstance()->ExecuteSQL( "insert into tt values(123)", dbName );
    SQLExecutor::GetInstance()->ExecuteSQL("set autocommit=1", dbName);  // commit上一个tx，新建自己的tx并commit
    ASSERT_EQ( txManager.GetNextTxId(), txId+1 );
    results = SQLExecutor::GetInstance()->ExecuteSQL("select a from tt", dbName);
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( results->GetResults()[ 0 ] )->GetContent();
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 123 );
    SQLExecutor::GetInstance()->ExecuteSQL("rollback", dbName);
    results = SQLExecutor::GetInstance()->ExecuteSQL("select a from tt", dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( results->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 123 );
    SQLExecutor::GetInstance()->ExecuteSQL( "drop table if exists tt", dbName );

    // global的autocommit改变时不会影响当前session，只会影响新建的连接。
    SQLExecutor::GetInstance()->ExecuteSQL("set autocommit=1", dbName);
    SQLExecutor::GetInstance()->ExecuteSQL("set autocommit=0", dbName);
    SQLExecutor::GetInstance()->ExecuteSQL( "begin", dbName );
    SQLExecutor::GetInstance()->ExecuteSQL( "drop table if exists tt", dbName );
    SQLExecutor::GetInstance()->ExecuteSQL( "create table tt(a int)", dbName );
    txId = txManager.GetNextTxId();
    SQLExecutor::GetInstance()->ExecuteSQL( "insert into tt values(123)", dbName );
    ASSERT_EQ( txManager.GetNextTxId(), txId+1 );
    SQLExecutor::GetInstance()->ExecuteSQL("set global autocommit=1", dbName);
    ASSERT_EQ( txManager.GetNextTxId(), txId+1 );
    results = SQLExecutor::GetInstance()->ExecuteSQL("select a from tt", dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( results->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 123 );
    SQLExecutor::GetInstance()->ExecuteSQL("rollback", dbName);
    ASSERT_EQ( txManager.GetNextTxId(), txId+1 );
    results = SQLExecutor::GetInstance()->ExecuteSQL("select a from tt", dbName);
    resTable = std::dynamic_pointer_cast< AriesMemTable >( results->GetResults()[ 0 ] )->GetContent();
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetData(), nullptr );
    SQLExecutor::GetInstance()->ExecuteSQL( "drop table if exists tt", dbName );

    SQLExecutor::GetInstance()->ExecuteSQL("set autocommit=1", dbName);

}


