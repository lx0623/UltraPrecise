//
// Created by david.shen on 2020/3/26.
//

#include <gtest/gtest.h>

#include "AriesMvccTestDataGenerator.h"
#include "AriesEngine/transaction/AriesMvccTable.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "AriesEngine/index/AriesIndex.h"
#include "datatypes/decimal.hxx"

BEGIN_ARIES_ENGINE_NAMESPACE

#define REGION_INITIAL_ROW_COUNT 5
#define MODIFY_INITIAL_ROWID -2
#define MODIFY_DELTA_ROWID1 1
#define MODIFY_DELTA_ROWID2 2
#define DELETE_INITIAL_ROWID -5
#define DELETE_DELTA_ROWID MODIFY_DELTA_ROWID1

#define TEST_DB_NAME "test_db"
#define TEST_TABLE_NAME "region"

#define TEST_PK_DB_NAME "test_db_pk"
#define TEST_PK_TABLE_NAME "region_pk"

#define TEST_COMPOSITE_PK_DB_NAME "test_db_composite_pk"
#define TEST_COMPOSITE_PK_TABLE_NAME "region_composite_pk"

void DropCompositePkTable()
{
    string sql = "drop database if exists ";
    sql += TEST_COMPOSITE_PK_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
}

void CreateCompositePkTableAndInsertData()
{
    DropCompositePkTable();
    string sql = "create database if not exists ";
    sql += TEST_COMPOSITE_PK_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "use ";
    sql += TEST_COMPOSITE_PK_DB_NAME;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );

    sql = R"(CREATE TABLE REGION_COMPOSITE_PK  ( R_REGIONKEY INTEGER NOT NULL, R_KEY2 INTEGER NOT NULL );)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_COMPOSITE_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_COMPOSITE_PK_TABLE_NAME;
    sql += " values(0, 5);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_COMPOSITE_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_COMPOSITE_PK_TABLE_NAME;
    sql += " values(0, 4);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_COMPOSITE_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_COMPOSITE_PK_TABLE_NAME;
    sql += " values(0, 3);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_COMPOSITE_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_COMPOSITE_PK_TABLE_NAME;
    sql += " values(0, 2);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_COMPOSITE_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_COMPOSITE_PK_TABLE_NAME;
    sql += " values(0, 1);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_COMPOSITE_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto recoveryer = std::make_shared< aries_engine::AriesXLogRecoveryer >();
    recoveryer->SetReader( aries_engine::AriesXLogManager::GetInstance().GetReader() );
    auto r = recoveryer->Recovery();
    ARIES_ASSERT( r, "cannot recovery from xlog" );
}

void DropPkTable()
{
    string sql = "drop database if exists ";
    sql += TEST_PK_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
}

void CreatePkTableAndInsertData()
{
    DropPkTable();
    string sql = "create database if not exists ";
    sql += TEST_PK_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "use ";
    sql += TEST_PK_DB_NAME;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );

    sql = R"(CREATE TABLE REGION_PK  ( R_REGIONKEY  INTEGER NOT NULL PRIMARY KEY);)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_PK_TABLE_NAME;
    sql += " values(5);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_PK_TABLE_NAME;
    sql += " values(4);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_PK_TABLE_NAME;
    sql += " values(3);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_PK_TABLE_NAME;
    sql += " values(2);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into ";
    sql += TEST_PK_TABLE_NAME;
    sql += " values(1);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto recoveryer = std::make_shared< aries_engine::AriesXLogRecoveryer >();
    recoveryer->SetReader( aries_engine::AriesXLogManager::GetInstance().GetReader() );
    auto r = recoveryer->Recovery();
    ARIES_ASSERT( r, "cannot recovery from xlog" );
}

void DropTable()
{
    string sql = "drop database if exists ";
    sql += TEST_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
}

void CreateTableAndInsertData() {
    DropTable();
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

void insertTuple(AriesMvccTestDataGeneratorSPtr table) {
    auto insert = AriesTransManager::GetInstance().NewTransaction();
    int regionKey = 5;
    string name("RATEUP_INSERT");
    string commnent("This is Rateup Insert Testcase! This is Rateup Insert Testcase! This is Rateup Insert Testcase!");
    TupleDataSPtr newdata = AriesMvccTestDataGenerator::GenerateRegionTupleData(regionKey, name, commnent);
    table->InsertTuple(newdata, 0, insert);
}

void modifyInitialTable(AriesMvccTestDataGeneratorSPtr table, RowPos rowPos) {
    auto modify = AriesTransManager::GetInstance().NewTransaction();
    int regionKey = 5;
    string name("RATEUP_MODIFY");
    string commnent("This is Rateup Modify InitTable Testcase! This is Rateup Modify InitTable Testcase! This is Rateup Modify InitTable Testcase!");
    TupleDataSPtr newdata = AriesMvccTestDataGenerator::GenerateRegionTupleData(regionKey, name, commnent);
    table->ModifyTuple(rowPos, newdata, 0, modify);
}

void modifyDeltaTable(AriesMvccTestDataGeneratorSPtr table, RowPos rowPos) {
    auto modify = AriesTransManager::GetInstance().NewTransaction();
    int regionKey = 500;
    string name("MODIFY_DELTA");
    string commnent("This is Rateup Modify DeltaTable TestCase!This is Rateup Modify DeltaTable TestCase!This is Rateup Modify DeltaTable TestCase!");
    TupleDataSPtr newdata = AriesMvccTestDataGenerator::GenerateRegionTupleData(regionKey, name, commnent);
    table->ModifyTuple(rowPos, newdata, 0, modify);
}

void deleteTuple(AriesMvccTestDataGeneratorSPtr table, RowPos rowPos) {
    auto delete1 = AriesTransManager::GetInstance().NewTransaction();
    table->DeleteTuple(rowPos, delete1);
}

AriesTableBlockUPtr getMvccTable(AriesMvccTableSPtr table)
{
    auto getMvcc = AriesTransManager::GetInstance().NewTransaction();
    return table->GetTable(getMvcc, {1,2,3});
}

TEST(UT_TestAriesMvccTable, insert) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared<AriesMvccTestDataGenerator>(table);
    insertTuple(dataGenerator);
    auto tableAfterInsert = getMvccTable(table);
    ASSERT_EQ( tableAfterInsert->GetRowCount(), REGION_INITIAL_ROW_COUNT + 1 );
    //very column 1
    auto columnData = tableAfterInsert->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 4 );
    ASSERT_EQ( columnData->GetInt32( 5 ), 5 );
    //very column 2
    columnData = tableAfterInsert->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    ASSERT_EQ( columnData->GetString( 5 ), "RATEUP_INSERT" );
    //very column 3
    columnData = tableAfterInsert->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
    ASSERT_EQ( columnData->GetString( 5 ), "This is Rateup Insert Testcase! This is Rateup Insert Testcase! This is Rateup Insert Testcase!" );
    DropTable();
}

/*
// UPDATE to be supported
TEST(UT_TestAriesMvccTable, modify_initialTable) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared< AriesMvccTestDataGenerator >( table );
    //modify initialTable
    modifyInitialTable(dataGenerator, MODIFY_INITIAL_ROWID);
    //verfy result
    auto tableAfterModify = getMvccTable(table);
    ASSERT_TRUE(tableAfterModify->GetRowCount() == REGION_INITIAL_ROW_COUNT);
    //very column 1
    auto columnData = tableAfterModify->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 4 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 5 );
    //very column 2
    columnData = tableAfterModify->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 2 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 3 ), "MIDDLE EAST" );
    ASSERT_EQ( columnData->GetString( 4 ), "RATEUP_MODIFY" );
    //very column 3
    columnData = tableAfterModify->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 2 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 3 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
    ASSERT_EQ( columnData->GetString( 4 ),
               "This is Rateup Modify InitTable Testcase! This is Rateup Modify InitTable Testcase! This is Rateup Modify InitTable Testcase!" );
    DropTable();
}

TEST(UT_TestAriesMvccTable, modify_deltaTable) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared< AriesMvccTestDataGenerator >( table );
    //modify initialTable
    modifyInitialTable(dataGenerator, MODIFY_INITIAL_ROWID);
    //modify deltaTable
    modifyDeltaTable(dataGenerator, MODIFY_DELTA_ROWID1);
    //very result
    auto tableAfterModify = getMvccTable(table);
    ASSERT_TRUE(tableAfterModify->GetRowCount() == REGION_INITIAL_ROW_COUNT);
    //very column 1
    auto columnData = tableAfterModify->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 4 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 500 );
    //very column 2
    columnData = tableAfterModify->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 2 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 3 ), "MIDDLE EAST" );
    ASSERT_EQ( columnData->GetString( 4 ), "MODIFY_DELTA" );
    //very column 3
    columnData = tableAfterModify->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 2 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 3 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
    ASSERT_EQ( columnData->GetString( 4 ),
               "This is Rateup Modify DeltaTable TestCase!This is Rateup Modify DeltaTable TestCase!This is Rateup Modify DeltaTable TestCase!" );
    DropTable();
}
*/

TEST(UT_TestAriesMvccTable, delete_initialTable) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared< AriesMvccTestDataGenerator >( table );
    //delete initialTable
    deleteTuple(dataGenerator, DELETE_INITIAL_ROWID);
    //very result
    auto tableAfterDelete = getMvccTable(table);
    ASSERT_EQ( tableAfterDelete->GetRowCount(), REGION_INITIAL_ROW_COUNT - 1 );
    //very column 1
    auto columnData = tableAfterDelete->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    //very column 2
    columnData = tableAfterDelete->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    //very column 3
    columnData = tableAfterDelete->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    DropTable();
}

/*
TEST(UT_TestAriesMvccTable, delete_deltaTable_afterModify) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared< AriesMvccTestDataGenerator >( table );
    //modify initialTable
    modifyInitialTable(dataGenerator, MODIFY_INITIAL_ROWID);
    //delete deltaTable
    deleteTuple(dataGenerator, DELETE_DELTA_ROWID);
    //verify result
    auto tableAfterDelete = getMvccTable(table);
    ASSERT_EQ( tableAfterDelete->GetRowCount(), REGION_INITIAL_ROW_COUNT - 1 );
    //very column 1
    auto columnData = tableAfterDelete->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 4 );
    //very column 2
    columnData = tableAfterDelete->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 2 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 3 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterDelete->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 2 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 3 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
    DropTable();
}
*/

TEST(UT_TestAriesMvccTable, delete_deltaTable_afterInsert) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared< AriesMvccTestDataGenerator >( table );
    //insert deltaTable
    insertTuple(dataGenerator);
    //delete deltaTable
    deleteTuple(dataGenerator, MODIFY_DELTA_ROWID1);
    //verify result
    auto tableAfterDelete = getMvccTable(table);
    ASSERT_TRUE(tableAfterDelete->GetRowCount() == REGION_INITIAL_ROW_COUNT);
    //very column 1
    auto columnData = tableAfterDelete->GetColumnBuffer(1);
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 3 ), 3 );
    ASSERT_EQ( columnData->GetInt32( 4 ), 4 );
    //very column 2
    columnData = tableAfterDelete->GetColumnBuffer(2);
    ASSERT_EQ( columnData->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( columnData->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( columnData->GetString( 2 ), "ASIA" );
    ASSERT_EQ( columnData->GetString( 3 ), "EUROPE" );
    ASSERT_EQ( columnData->GetString( 4 ), "MIDDLE EAST" );
    //very column 3
    columnData = tableAfterDelete->GetColumnBuffer(3);
    ASSERT_EQ( columnData->GetString( 0 ), "lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to " );
    ASSERT_EQ( columnData->GetString( 1 ), "hs use ironic, even requests. s" );
    ASSERT_EQ( columnData->GetString( 2 ), "ges. thinly even pinto beans ca" );
    ASSERT_EQ( columnData->GetString( 3 ), "ly final courts cajole furiously final excuse" );
    ASSERT_EQ( columnData->GetString( 4 ), "uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl" );
    DropTable();
}

TEST(UT_TestAriesMvccTable, tryLock) {
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared<AriesMvccTestDataGenerator>(table);
    insertTuple(dataGenerator);

    int lockRowId = -1;  // lock rowId in initialTable
    ASSERT_EQ( table->TryLock(lockRowId), true);
    ASSERT_EQ( table->TryLock(lockRowId), false);
    table->Unlock(lockRowId);
    ASSERT_EQ( table->TryLock(lockRowId), true);
    table->Unlock(lockRowId);
    table->Lock(lockRowId);
    ASSERT_EQ( table->TryLock(lockRowId), false);
    table->Unlock(lockRowId);
    ASSERT_EQ( table->TryLock(lockRowId), true);
    table->Unlock(lockRowId);

    lockRowId = 1; // lock rowId in deltaTable
    ASSERT_EQ( table->TryLock(lockRowId), true);
    ASSERT_EQ( table->TryLock(lockRowId), false);
    table->Unlock(lockRowId);
    ASSERT_EQ( table->TryLock(lockRowId), true);
    table->Unlock(lockRowId);
    table->Lock(lockRowId);
    ASSERT_EQ( table->TryLock(lockRowId), false);
    table->Unlock(lockRowId);
    ASSERT_EQ( table->TryLock(lockRowId), true);
    table->Unlock(lockRowId);
    DropTable();
}

TEST(UT_TestAriesMvccTable, empty) {
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( string( "drop database if exists " ) + TEST_DB_NAME, "" );
    ASSERT_TRUE( result->IsSuccess() );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( string( "create database " ) + TEST_DB_NAME, "" );
    ASSERT_TRUE( result->IsSuccess() );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( "create table TestAriesMvccTable_empty ( id int );", TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( "insert into TestAriesMvccTable_empty set id=10;", TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( "select * from TestAriesMvccTable_empty;", TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );

    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] );
    auto table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "10" );

    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( string( "drop database if exists " ) + TEST_DB_NAME, "" );
    ASSERT_TRUE( result->IsSuccess() );
}

/*
TEST( UT_TestAriesMvccTable, GetTableBySlots )
{
    CreateTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_DB_NAME, TEST_TABLE_NAME );
    AriesMvccTestDataGeneratorSPtr dataGenerator = make_shared< AriesMvccTestDataGenerator >( table );
    // modify initialTable
    modifyInitialTable( dataGenerator, MODIFY_INITIAL_ROWID );
    auto insert = AriesTransManager::GetInstance().NewTransaction();
    auto resTable = table->GetTableBySlots( insert, {-1, -2, -3, 1}, 1);
    ASSERT_EQ( resTable->GetRowCount(), 3 );
    // very column 1
    auto columnData = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnData->GetInt32( 0 ), 0 );
    ASSERT_EQ( columnData->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnData->GetInt32( 2 ), 5 );
    DropTable();
}
*/

TupleDataSPtr GenerateRegionPkTupleData( int regionkey )
{
    TupleDataSPtr newdata = make_shared<TupleData>();
    AriesColumnType intColumnType({AriesValueType::INT32}, false);
    AriesDataBufferSPtr dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
    memcpy(dataBuf->GetData(), &regionkey, sizeof(int));
    newdata->data[1] = dataBuf;
    return newdata;
}

TEST( UT_TestAriesMvccTable, XminInprogress_PK )
{
    //initial table has 1, 2, 3, 4, 5
    CreatePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_PK_DB_NAME, TEST_PK_TABLE_NAME );

    vector<int> keyColumnIds;
    keyColumnIds.push_back( 1 );

    auto self = AriesTransManager::GetInstance().NewTransaction();
    auto other = AriesTransManager::GetInstance().NewTransaction();
    TupleDataSPtr newdata;
    int regionKey;
    RowPos rowPos;

    //insert 6
    regionKey = 6;
    rowPos = 1;
    newdata = GenerateRegionPkTupleData( regionKey );
    table->AddTuple( self, newdata );

    //xmin == self and xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    //xmin == other and xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    //delete 6
    table->SetTxMax( rowPos, self->GetTxId() );
    table->ModifyTuple( self, rowPos, nullptr, 0 );

    //xmin == self and xmax == xmin
    ASSERT_FALSE( table->PrimaryKeyMightExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    //xmin == other and xmax == INVALID_TX_ID
    ASSERT_FALSE( table->PrimaryKeyMightExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::COMMITTED );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );
    DropPkTable();
}

TEST( UT_TestAriesMvccTable, XminAbort_PK )
{
    //initial table has 1, 2, 3, 4, 5
    CreatePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_PK_DB_NAME, TEST_PK_TABLE_NAME );

    vector<int> keyColumnIds;
    keyColumnIds.push_back( 1 );

    auto self = AriesTransManager::GetInstance().NewTransaction();

    //insert 6 and abort
    int regionKey = 6;
    RowPos rowPos = 1;
    TupleDataSPtr newdata = GenerateRegionPkTupleData( regionKey );
    table->AddTuple( self, newdata );
    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::ABORTED );

    auto other = AriesTransManager::GetInstance().NewTransaction();
    ASSERT_FALSE( table->PrimaryKeyMightExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );

    DropPkTable();
}

TEST( UT_TestAriesMvccTable, XminCommitted_PK )
{
    //initial table has 1, 2, 3, 4, 5
    CreatePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_PK_DB_NAME, TEST_PK_TABLE_NAME );
    table->RebuildPrimaryKeyIndex();

    vector<int> keyColumnIds;
    keyColumnIds.push_back( 1 );

    auto self = AriesTransManager::GetInstance().NewTransaction();
    auto other = AriesTransManager::GetInstance().NewTransaction();
    table->GetTable( self, { 1 } );

    // data in initial table
    int regionKey = 1;
    RowPos rowPos = -1;

    // xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    // delete 1 by self, xmax == inprogress
    table->SetTxMax( rowPos, self->GetTxId() );
    table->ModifyTuple( self, rowPos, nullptr, 0 );
    ASSERT_FALSE( table->PrimaryKeyMightExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    // delete 1 by other, xmax == inprogress
    RowPos otherRowPost = -2;
    int otherRegionKey = 2;
    table->SetTxMax( otherRowPost, other->GetTxId() );
    table->ModifyTuple( other, otherRowPost, nullptr, 0 );
    ASSERT_TRUE( table->PrimaryKeyMightExists( self->GetTxId(), &otherRegionKey, otherRowPost, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), &otherRegionKey, otherRowPost, keyColumnIds ) );

    // xmax == abort
    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::ABORTED );
    ASSERT_TRUE( table->PrimaryKeyMightExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    // delete 1, xmax == committed
    table->SetTxMax( rowPos, other->GetTxId() );
    table->ModifyTuple( other, rowPos, nullptr, 0 );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );
    auto third = AriesTransManager::GetInstance().NewTransaction();
    ASSERT_FALSE( table->PrimaryKeyMightExists( third->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( third->GetTxId(), &regionKey, rowPos, keyColumnIds ) );


    // data in delta table
    //insert 6, committed
    TupleDataSPtr newdata;
    regionKey = 6;
    rowPos = 1;
    newdata = GenerateRegionPkTupleData( regionKey );
    table->AddTuple( third, newdata );
    AriesTransManager::GetInstance().EndTransaction( third, TransactionStatus::COMMITTED );

    auto fourth = AriesTransManager::GetInstance().NewTransaction();

    // xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( fourth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( fourth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    // delete 6, xmax == inprogress
    table->SetTxMax( rowPos, fourth->GetTxId() );
    table->ModifyTuple( fourth, rowPos, nullptr, 0 );
    ASSERT_FALSE( table->PrimaryKeyMightExists( fourth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( fourth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    auto fifth = AriesTransManager::GetInstance().NewTransaction();
    // xmax == abort
    AriesTransManager::GetInstance().EndTransaction( fourth, TransactionStatus::ABORTED );
    ASSERT_TRUE( table->PrimaryKeyMightExists( fifth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( fifth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    // delete 6, xmax == committed
    table->SetTxMax( rowPos, fifth->GetTxId() );
    table->ModifyTuple( fifth, rowPos, nullptr, 0 );
    AriesTransManager::GetInstance().EndTransaction( fifth, TransactionStatus::COMMITTED );
    auto sixth = AriesTransManager::GetInstance().NewTransaction();
    ASSERT_FALSE( table->PrimaryKeyMightExists( sixth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( sixth->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    AriesTransManager::GetInstance().EndTransaction( sixth, TransactionStatus::COMMITTED );

    DropPkTable();
}

TEST( UT_TestAriesMvccTable, KeyChanged_PK )
{
    //initial table has 1, 2, 3, 4, 5
    CreatePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_PK_DB_NAME, TEST_PK_TABLE_NAME );

    vector<int> keyColumnIds;
    keyColumnIds.push_back( 1 );

    auto self = AriesTransManager::GetInstance().NewTransaction();
    auto other = AriesTransManager::GetInstance().NewTransaction();
    TupleDataSPtr newdata;
    int regionKey;
    RowPos rowPos;

    //insert 6
    regionKey = 6;
    rowPos = 1;
    newdata = GenerateRegionPkTupleData( regionKey );
    table->AddTuple( self, newdata );

    //insert 7
    regionKey = 7;
    rowPos = 2;
    newdata = GenerateRegionPkTupleData( regionKey );
    table->AddTuple( self, newdata );

    rowPos = 1;
    ASSERT_FALSE( table->PrimaryKeyMightExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    ASSERT_FALSE( table->PrimaryKeyMightExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), &regionKey, rowPos, keyColumnIds ) );

    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::COMMITTED );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );

    DropPkTable();
}

TupleDataSPtr GenerateRegionCompositePkTupleData( int regionkey )
{
    int firstKey = 0;
    TupleDataSPtr newdata = make_shared<TupleData>();
    AriesColumnType intColumnType({AriesValueType::INT32}, false);
    AriesDataBufferSPtr dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
    memcpy(dataBuf->GetData(), &firstKey, sizeof(int));
    newdata->data[1] = dataBuf;

    dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
    memcpy(dataBuf->GetData(), &regionkey, sizeof(int));
    newdata->data[2] = dataBuf;

    return newdata;
}

string GenerateCompositeKey( int regionkey )
{
    string data;
    data.resize( sizeof(int) * 2 );
    int* pdata = ( int* )data.data();

    pdata[0] = 0;
    pdata[1] = regionkey;

    return data;
}


TEST( UT_TestAriesMvccTable, XminInprogress_COMPOSITE_PK )
{
    CreateCompositePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_COMPOSITE_PK_DB_NAME, TEST_COMPOSITE_PK_TABLE_NAME );

    vector<int> keyColumnIds{ 1, 2 };

    auto self = AriesTransManager::GetInstance().NewTransaction();
    auto other = AriesTransManager::GetInstance().NewTransaction();
    TupleDataSPtr newdata;
    int regionKey;
    RowPos rowPos;

    //insert 6
    regionKey = 6;
    rowPos = 1;
    newdata = GenerateRegionCompositePkTupleData( regionKey );
    table->AddTuple( self, newdata );

    string primaryKey = GenerateCompositeKey( regionKey );

    //xmin == self and xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    //xmin == other and xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    //delete 6
    table->SetTxMax( rowPos, self->GetTxId() );
    table->ModifyTuple( self, rowPos, nullptr, 0 );

    //xmin == self and xmax == xmin
    ASSERT_FALSE( table->PrimaryKeyMightExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    //xmin == other and xmax == INVALID_TX_ID
    ASSERT_FALSE( table->PrimaryKeyMightExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::COMMITTED );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );
    DropCompositePkTable();
}

TEST( UT_TestAriesMvccTable, XminAbort_COMPOSITE_PK )
{
    CreateCompositePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_COMPOSITE_PK_DB_NAME, TEST_COMPOSITE_PK_TABLE_NAME );

    vector<int> keyColumnIds{ 1, 2 };

    auto self = AriesTransManager::GetInstance().NewTransaction();

    //insert 6 and abort
    int regionKey = 6;
    RowPos rowPos = 1;
    TupleDataSPtr newdata = GenerateRegionCompositePkTupleData( regionKey );
    table->AddTuple( self, newdata );
    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::ABORTED );

    string primaryKey = GenerateCompositeKey( regionKey );

    auto other = AriesTransManager::GetInstance().NewTransaction();
    ASSERT_FALSE( table->PrimaryKeyMightExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );

    DropCompositePkTable();
}

TEST( UT_TestAriesMvccTable, XminCommitted_COMPOSITE_PK )
{
    CreateCompositePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_COMPOSITE_PK_DB_NAME, TEST_COMPOSITE_PK_TABLE_NAME );
    table->RebuildPrimaryKeyIndex();

    vector<int> keyColumnIds{ 1, 2 };

    auto self = AriesTransManager::GetInstance().NewTransaction();
    auto other = AriesTransManager::GetInstance().NewTransaction();
    table->GetTable( self, { 1, 2 } );
    int regionKey = 1;
    // data in initial table
    string primaryKey = GenerateCompositeKey( regionKey );

    RowPos rowPos = -1;

    // xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // delete 1, xmax == inprogress, check myself
    table->SetTxMax( rowPos, self->GetTxId() );
    table->ModifyTuple( self, rowPos, nullptr, 0 );
    ASSERT_FALSE( table->PrimaryKeyMightExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // delete 1, xmax == inprogress, check other
    ASSERT_TRUE( table->PrimaryKeyMightExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // xmax == abort
    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::ABORTED );
    ASSERT_TRUE( table->PrimaryKeyMightExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyMightExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // delete 1, xmax == committed
    table->SetTxMax( rowPos, other->GetTxId() );
    table->ModifyTuple( other, rowPos, nullptr, 0 );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );
    auto third = AriesTransManager::GetInstance().NewTransaction();
    ASSERT_FALSE( table->PrimaryKeyMightExists( third->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( third->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );


    // data in delta table
    //insert 6, committed
    TupleDataSPtr newdata;
    regionKey = 6;
    rowPos = 1;
    newdata = GenerateRegionCompositePkTupleData( regionKey );
    table->AddTuple( third, newdata );
    AriesTransManager::GetInstance().EndTransaction( third, TransactionStatus::COMMITTED );
    primaryKey = GenerateCompositeKey( regionKey );

    auto fourth = AriesTransManager::GetInstance().NewTransaction();
    auto fifth = AriesTransManager::GetInstance().NewTransaction();

    // xmax == INVALID_TX_ID
    ASSERT_TRUE( table->PrimaryKeyMightExists( fourth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( fourth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // delete 6, xmax == inprogress, check self
    table->SetTxMax( rowPos, fourth->GetTxId() );
    table->ModifyTuple( fourth, rowPos, nullptr, 0 );
    ASSERT_FALSE( table->PrimaryKeyMightExists( fourth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( fourth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // delete 6, xmax == inprogress, check other
    ASSERT_TRUE( table->PrimaryKeyMightExists( fifth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( fifth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    
    // xmax == abort
    AriesTransManager::GetInstance().EndTransaction( fourth, TransactionStatus::ABORTED );
    ASSERT_TRUE( table->PrimaryKeyMightExists( fourth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( fourth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyMightExists( fifth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_TRUE( table->PrimaryKeyExists( fifth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    // delete 6, xmax == committed
    table->SetTxMax( rowPos, fifth->GetTxId() );
    table->ModifyTuple( fifth, rowPos, nullptr, 0 );
    AriesTransManager::GetInstance().EndTransaction( fifth, TransactionStatus::COMMITTED );
    auto sixth = AriesTransManager::GetInstance().NewTransaction();
    ASSERT_FALSE( table->PrimaryKeyMightExists( sixth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( sixth->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    AriesTransManager::GetInstance().EndTransaction( sixth, TransactionStatus::COMMITTED );

    DropCompositePkTable();
}

TEST( UT_TestAriesMvccTable, KeyChanged_COMPOSITE_PK )
{
    //initial table has 1, 2, 3, 4, 5
    CreateCompositePkTableAndInsertData();
    AriesMvccTableSPtr table = make_shared< AriesMvccTable >( TEST_COMPOSITE_PK_DB_NAME, TEST_COMPOSITE_PK_TABLE_NAME );

    vector<int> keyColumnIds{ 1, 2 };

    auto self = AriesTransManager::GetInstance().NewTransaction();
    auto other = AriesTransManager::GetInstance().NewTransaction();
    TupleDataSPtr newdata;
    int regionKey;
    RowPos rowPos;

    //insert 6
    regionKey = 6;
    rowPos = 1;
    newdata = GenerateRegionCompositePkTupleData( regionKey );
    table->AddTuple( self, newdata );

    //insert 7
    regionKey = 7;
    rowPos = 2;
    newdata = GenerateRegionCompositePkTupleData( regionKey );
    table->AddTuple( self, newdata );
    string primaryKey = GenerateCompositeKey( regionKey );

    rowPos = 1;
    ASSERT_FALSE( table->PrimaryKeyMightExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( self->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    ASSERT_FALSE( table->PrimaryKeyMightExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );
    ASSERT_FALSE( table->PrimaryKeyExists( other->GetTxId(), primaryKey.data(), rowPos, keyColumnIds ) );

    AriesTransManager::GetInstance().EndTransaction( self, TransactionStatus::COMMITTED );
    AriesTransManager::GetInstance().EndTransaction( other, TransactionStatus::COMMITTED );

    DropCompositePkTable();
}

TEST( UT_TestAriesMvccTable, MakeIndexKey )
{
    // check more than one column
    vector< int > keyColumnIds{ 1, 2 };
    // insert 6
    int regionKey = 6;
    auto newdata = GenerateRegionCompositePkTupleData( regionKey );
    auto key = AriesMvccTable::MakeIndexKey( keyColumnIds, newdata, 0 );

    auto value = key.data();
    ASSERT_EQ( ( (int *) value )[0], 0 );
    ASSERT_EQ( ( (int *) value )[1], 6 );

    //test single string column
    string keyString( "Th" );
    newdata = make_shared< TupleData >();
    AriesColumnType intColumnType( { AriesValueType::CHAR, 2 }, false );
    AriesDataBufferSPtr dataBuf = make_shared< AriesDataBuffer >( intColumnType, 1, true );
    memcpy( dataBuf->GetData(), keyString.data(), keyString.size() );
    newdata->data[1] = dataBuf;
    key = AriesMvccTable::MakeIndexKey( { 1 }, newdata, 0 );

    ASSERT_EQ( memcmp( key.data(), dataBuf->GetItemDataAt( 0 ), dataBuf->GetDataType().GetDataTypeSize() ), 0 );

    // test single nullable string column
    newdata = make_shared< TupleData >();
    AriesColumnType stringColumnType( { AriesValueType::CHAR, 2 }, true );
    dataBuf = make_shared< AriesDataBuffer >( stringColumnType, 1, true );
    auto itemBuf = dataBuf->GetItemDataAt( 0 );
    *itemBuf = 1;
    memcpy( itemBuf + 1, keyString.data(), keyString.size() );
    newdata->data[1] = dataBuf;

    key = AriesMvccTable::MakeIndexKey( { 1 }, newdata, 0 );
    char flag;
    memcpy( &flag, key.data(), 1 );
    ASSERT_EQ( flag, 1 );
    ASSERT_EQ( memcmp( key.data() + 1, keyString.data(), keyString.size() ), 0 );

    // test char(1)
    string keyChar1( "T" );
    newdata = make_shared< TupleData >();
    AriesColumnType char1ColumnType( { AriesValueType::CHAR, 1 }, false );
    dataBuf = make_shared< AriesDataBuffer >( char1ColumnType, 1, true );
    memcpy( dataBuf->GetData(), keyChar1.data(), keyChar1.size() );
    newdata->data[1] = dataBuf;
    key = AriesMvccTable::MakeIndexKey( { 1 }, newdata, 0 );

    ASSERT_EQ( memcmp( key.data(), dataBuf->GetItemDataAt( 0 ), dataBuf->GetDataType().GetDataTypeSize()), 0 );

    // test decimal column
    aries_acc::Decimal d(15,2, "123.00");
    auto len = GetDecimalRealBytes( 15, 2 );
    newdata = make_shared< TupleData >();
    AriesColumnType decimalColumnType( { AriesValueType::COMPACT_DECIMAL, 15, 2 }, true );
    dataBuf = make_shared< AriesDataBuffer >( decimalColumnType, 1, true );
    itemBuf = dataBuf->GetItemDataAt( 0 );
    *itemBuf = 1;
    aries_acc::Decimal( 15, 2 ).cast( d ).ToCompactDecimal( (char *) itemBuf + 1, len );
    newdata->data[1] = dataBuf;
    key = AriesMvccTable::MakeIndexKey( { 1 }, newdata, 0 );

    memcpy( &flag, key.data(), 1 );
    ASSERT_EQ( flag, 1 );
    ASSERT_EQ( memcmp( key.data() + 1, itemBuf + 1, len ), 0 );
}

#define TEST_PK_FK_DB_NAME "test_db_pk_pk"
#define TEST_PK_FK_DB_PK_TABLE_NAME "pk_table"
#define TEST_PK_FK_DB_FK_TABLE_NAME "fk_table"

void DropPkFkTable()
{
    string sql = "drop database if exists ";
    sql += TEST_PK_FK_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
}

void CreatePkFkTable()
{
    DropPkFkTable();
    string sql = "create database if not exists ";
    sql += TEST_PK_FK_DB_NAME;
    sql += ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "use ";
    sql += TEST_PK_FK_DB_NAME;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );

    sql = R"(CREATE TABLE PK_TABLE  ( PK_ID  INTEGER NOT NULL, PRIMARY KEY(PK_ID) );)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_FK_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    sql = R"(CREATE TABLE FK_TABLE  ( FK_ID  INTEGER NOT NULL, FOREIGN KEY(FK_ID) REFERENCES PK_TABLE(PK_ID) );)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_FK_DB_NAME );
    ASSERT_FALSE( result->IsSuccess() );
    // sql = R"(insert into PK_TABLE values(1);)";
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_PK_FK_DB_NAME );
    // ASSERT_TRUE( result->IsSuccess() );
    // auto recoveryer = std::make_shared< aries_engine::AriesXLogRecoveryer >();
    // recoveryer->SetReader( aries_engine::AriesXLogManager::GetInstance().GetReader() );
    // auto r = recoveryer->Recovery();
    // ARIES_ASSERT( r, "cannot recovery from xlog" );
}

TEST( UT_TestAriesMvccTable, CheckPkFk )
{
    // for delay load
    CreatePkFkTable();
    // AriesMvccTableSPtr pkTable = AriesMvccTableManager::GetInstance().getMvccTable( TEST_PK_FK_DB_NAME, TEST_PK_FK_DB_PK_TABLE_NAME );
    // AriesMvccTableSPtr fkTable = AriesMvccTableManager::GetInstance().getMvccTable( TEST_PK_FK_DB_NAME, TEST_PK_FK_DB_FK_TABLE_NAME );
    // auto tx = AriesTransManager::GetInstance().NewTransaction();
    // auto pk = pkTable->GetPrimaryKeyInfo( tx );
    // //auto fk = pkTable->GetForeignKeyInfo( tx );
    // ASSERT_TRUE( pk.index != nullptr );
    // //ASSERT_TRUE( fk.size() == 0 );
    // pk = fkTable->GetPrimaryKeyInfo( tx );
    // //fk = fkTable->GetForeignKeyInfo( tx );
    // ASSERT_TRUE( pk.index == nullptr );
    // //ASSERT_TRUE( fk.size() == 1 );
    // //ASSERT_TRUE( fk[0].indexKey.index != nullptr );
    // AriesMvccTableManager::GetInstance().removeMvccTable( TEST_PK_FK_DB_NAME, TEST_PK_FK_DB_PK_TABLE_NAME );
    // AriesMvccTableManager::GetInstance().removeMvccTable( TEST_PK_FK_DB_NAME, TEST_PK_FK_DB_FK_TABLE_NAME );

    // // for other cases
    // CreatePkFkTable();
    // pkTable = AriesMvccTableManager::GetInstance().getMvccTable( TEST_PK_FK_DB_NAME, TEST_PK_FK_DB_PK_TABLE_NAME );
    // fkTable = AriesMvccTableManager::GetInstance().getMvccTable( TEST_PK_FK_DB_NAME, TEST_PK_FK_DB_FK_TABLE_NAME );

    // // add same key
    // auto newData = make_shared< TupleData >();
    // AriesColumnType intColumnType( { AriesValueType::INT32, 1 }, false );
    // AriesDataBufferSPtr dataBuf = make_shared< AriesDataBuffer >( intColumnType, 1, true );
    // int value = 1;
    // memcpy( dataBuf->GetData(), &value, intColumnType.GetDataTypeSize() );
    // newData->data[1] = dataBuf;
    // try
    // {
    //     auto res = pkTable->AddTuple( tx, 0, newData, 0 );
    //     ASSERT_TRUE( false );
    // }
    // catch ( const std::exception &e )
    // {
    //     std::cerr << e.what() << '\n';
    //     ASSERT_TRUE( true );
    // }
    // AriesTransManager::GetInstance().EndTransaction( tx, TransactionStatus::ABORTED );

    // //add other key
    // tx = AriesTransManager::GetInstance().NewTransaction();
    // newData = make_shared< TupleData >();
    // dataBuf = make_shared< AriesDataBuffer >( intColumnType, 1, true );
    // value = 2;
    // memcpy( dataBuf->GetData(), &value, intColumnType.GetDataTypeSize() );
    // newData->data[1] = dataBuf;
    // auto res = pkTable->AddTuple( tx, 0, newData, 0 );
    // ASSERT_EQ( res, true );
    // AriesTransManager::GetInstance().EndTransaction( tx, TransactionStatus::COMMITTED );

    // // add reference key : exist key in pk_table
    // tx = AriesTransManager::GetInstance().NewTransaction();
    // newData = make_shared< TupleData >();
    // dataBuf = make_shared< AriesDataBuffer >( intColumnType, 1, true );
    // value = 1;
    // memcpy( dataBuf->GetData(), &value, intColumnType.GetDataTypeSize() );
    // newData->data[1] = dataBuf;
    // res = fkTable->AddTuple( tx, 0, newData, 0 );
    // ASSERT_EQ( res, true );
    // AriesTransManager::GetInstance().EndTransaction( tx, TransactionStatus::COMMITTED );

    // // delete pk do NOT exist in fk: 2
    // tx = AriesTransManager::GetInstance().NewTransaction();
    // res = pkTable->ModifyTuple( tx, 0, 1, nullptr, 0 );
    // ASSERT_EQ( res, true );
    // AriesTransManager::GetInstance().EndTransaction( tx, TransactionStatus::COMMITTED );
}

TEST( UT_TestAriesMvccTable, CheckAEExprInNode )
{
    string sql = "create database if not exists test;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    sql = "drop table if exists t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    sql = "create table t1(a int);";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    sql = "SELECT a FROM t1 WHERE a IN(1, (SELECT 2));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into t1 values (1),(2),(3),(4),(5),(6),(7);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "SELECT a FROM t1 WHERE a IN(1, (SELECT 2));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "SELECT a FROM t1 WHERE a IN(1, 3);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "SELECT a FROM t1 WHERE a IN((select 3), (select 4));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );
    ASSERT_EQ( std::dynamic_pointer_cast< AriesMemTable >(result->GetResults()[0])->GetContent()->GetRowCount(), 2 );
    // not supported now fow subquery who's result has mutiple columns
    // sql = "SELECT a FROM t1 WHERE a IN((select 1,2,3),4)";
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    // ASSERT_TRUE( result->IsSuccess() );
    // ASSERT_TRUE( result->GetResults().size()==4 );

    sql = "drop table if exists t1;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
}

END_ARIES_ENGINE_NAMESPACE
