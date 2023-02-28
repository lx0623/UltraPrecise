#include <gtest/gtest.h>
#include <vector>

#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"
#include "schema/SchemaManager.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "AriesEngine/AriesUtil.h"
#include "datatypes/AriesDatetimeTrans.h"
#include "datatypes/decimal.hxx"
#include "datatypes/AriesCastFunctions.hxx"
#include "utils/string_util.h"

#include "../../../TestUtils.h"

using namespace aries_engine;
using namespace std;
using namespace aries_acc;
using namespace aries_test;

/** tpch_100:
 * orders table:
 * 150,000,000 lines
 * block count: ( 150000000  + 20971520 - 1 ) / 20971520 = 8
 * available slot:  8 *  20971520 - 150000000 = 17,772,16
 *
 * part table:
 * 20,000,000 line
 *
 * supplier table:
 * 1,000,00
 *
 * partsupp table:
 * 80,000,000 lines
 * block count: ( 80000000 + 20971520 - 1 ) / 20971520 = 4
 * available slot:  4 *  20971520 - 80000000 = 3,886,080
 */

/**
 * tpch_10:
 * lineitem table:
 * 59,986,052 rows
 * block count: ( 59986052 + 20971520 - 1 ) / 20971520 = 3
 * available slot: 3 * 20971520 - 59986052 = 2,928,508
 */

const static string TEST_DB_NAME = "init_table_test";
// const static string TPCH_100_PARTSUPP_CSV_PATH = "/data/tpch/tpch_100/partsupp.tbl";
// const static string TPCH_10_LINEITEM_CSV_PATH = "/data/tpch/tpch_10/lineitem.tbl";
const static string TPCH_10_20971530_LINEITEM_CSV_PATH = "test_resources/AriesEngine/transaction/TestInitTable/lineitem-tpch10-20971530.tbl";
class UT_TestAriesInitialTable : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "TestAriesInitTable SetUpTestCase...\n";
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLResultPtr result;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
        ASSERT_TRUE( result->IsSuccess() );

        sql = "create database if not exists " + TEST_DB_NAME;
        result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
        ASSERT_TRUE( result->IsSuccess() );

        result = aries::SQLExecutor::GetInstance()->ExecuteSQLFromFile( "./test_resources/schema/tpch/create_table.sql", TEST_DB_NAME );
        ASSERT_TRUE( result->IsSuccess() );

    }
    static void TearDownTestCase()
    {
        cout << "TestAriesInitTable TearDownTestCase...\n";
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    }
    virtual void SetUp() { }
    virtual void TearDown() { }
};

int8_t* GetInsertRowData( const TableEntrySPtr& tableEntry,
                          const int lInsertOrderkey,
                          const int lInsertPartkey,
                          const int lInsertSuppkey,
                          const int lInsertLinenumber,
                          const char* buffQuantity,
                          const char* buffExtendedPrice,
                          const char* buffDiscount,
                          const char* buffTax,
                          const string& lInsertReturnFlag,
                          const string& lInsertLinestatus,
                          const AriesDate& lInsertShipdate,
                          const AriesDate& lInsertCommitDate,
                          const AriesDate& lInsertReceiptDate,
                          const string& lInsertShipInstruct,
                          const string& lInsertShipMode,
                          const string& lInsertComment,
                          const int decLen )
{
    auto columnCount = tableEntry->GetColumnsCount();
    int32_t bitmapLen = DIV_UP( columnCount, 8 );
    size_t lineStoreLen = tableEntry->GetRowStoreSize();
    lineStoreLen += bitmapLen;
    int8_t* rowDataPtr = new int8_t[ lineStoreLen ];
    memset( rowDataPtr, 0, lineStoreLen );

    size_t buffPos = bitmapLen;
    int colId = 1;
    auto colEntry = tableEntry->GetColumnById( colId++ );
    size_t itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertOrderkey, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertPartkey, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertSuppkey, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertLinenumber, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, buffQuantity, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, buffExtendedPrice, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, buffDiscount, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, buffTax, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, lInsertReturnFlag.data(), std::min( itemStoreSize, lInsertReturnFlag.size() ) );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, lInsertLinestatus.data(), std::min( itemStoreSize, lInsertLinestatus.size() ) );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertShipdate, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertCommitDate, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, &lInsertReceiptDate, itemStoreSize );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, lInsertShipInstruct.data(), std::min( itemStoreSize, lInsertShipInstruct.size() ) );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, lInsertShipMode.data(), std::min( itemStoreSize, lInsertShipMode.size() ) );
    buffPos += itemStoreSize;

    colEntry = tableEntry->GetColumnById( colId++ );
    itemStoreSize = colEntry->GetItemStoreSize();
    memcpy( rowDataPtr + buffPos, lInsertComment.data(), std::min( itemStoreSize, lInsertComment.size() ) );

    return rowDataPtr;
}

void BatchVerifyInsertedData( const string& dbName,
                              uint64_t expectedRowCount,
                              int lInsertOrderkey, int lInsertPartkey, int lInsertSuppkey, int lInsertLinenumber,
                              const string& lInsertQuantity, const string& lInsertExtendedPrice, const string& lInsertDiscount, const string& lInsertTax,
                              const string& lInsertReturnFlag, const string& lInsertLinestatus,
                              const AriesDate& lInsertShipdate,
                              const AriesDate& lInsertCommitDate,
                              const AriesDate& lInsertReceiptDate,
                              const string& lInsertShipInstruct,
                              const string& lInsertShipMode,
                              const string& lInsertComment )
{
    string sql = "select * from lineitem where L_ORDERKEY >= " + std::to_string( lInsertOrderkey ) + " ORDER BY L_ORDERKEY ";
    auto table = ExecuteSQL( sql, dbName );
    auto resultRowCount = table->GetRowCount();
    ASSERT_EQ( resultRowCount, expectedRowCount );

    int columnCount = table->GetColumnCount();
    vector< AriesDataBufferSPtr > columns;
    for ( int col = 1; col < columnCount + 1; col++ )
    {
        columns.push_back( table->GetColumnBuffer( col ) );
    }

    for ( int64_t i = 0; i < resultRowCount; i += 100 )
    {
        // verify columns data
        int colIdx = 0;
        auto column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetInt32( i ), lInsertOrderkey );
        lInsertOrderkey += 100;

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetInt32( i ), lInsertPartkey );
        lInsertPartkey += 100;

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetInt32( i ), lInsertSuppkey );
        lInsertSuppkey += 100;

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetInt32( i ), lInsertLinenumber );
        lInsertLinenumber += 100;

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetCompactDecimalAsString( i, 15, 2 ), lInsertQuantity );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetCompactDecimalAsString( i, 15, 2 ), lInsertExtendedPrice );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetCompactDecimalAsString( i, 15, 2 ), lInsertDiscount );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetCompactDecimalAsString( i, 15, 2 ), lInsertTax );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetString( i ), lInsertReturnFlag );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetString( i ), lInsertLinestatus );

        column = columns[ colIdx++ ];
        ASSERT_EQ( *column->GetDate( i ), lInsertShipdate );

        column = columns[ colIdx++ ];
        ASSERT_EQ( *column->GetDate( i ), lInsertCommitDate );

        column = columns[ colIdx++ ];
        ASSERT_EQ( *column->GetDate( i ), lInsertReceiptDate );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetString( i ), lInsertShipInstruct );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetString( i ), lInsertShipMode );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetString( i ), lInsertComment );
    }
}

void BatchVerifyUpdatedData( const string& dbName,
                             vector< index_t > rowIndiceToUpdate,
                             int lUpdateSuppkey, int lUpdateLinenumber,
                             aries_acc::Decimal lUpdateQuantity,
                             const string& lUpdateReturnFlag, const string& lUpdateLinestatus,
                             AriesDate& lUpdateShipdate,
                             AriesDate& lUpdateReceiptDate,
                             const string& lUpdateComment )
{
    for ( auto rowIdx : rowIndiceToUpdate )
    {
        string sql = "select * from lineitem limit " + std::to_string( rowIdx ) + ", 1";
        auto table = ExecuteSQL( sql, dbName );
        auto resultRowCount = table->GetRowCount();
        ASSERT_EQ( resultRowCount, 1 );

        auto column = table->GetColumnBuffer( 3 );
        ASSERT_EQ( column->GetInt32( 0 ), lUpdateSuppkey );
        ++lUpdateSuppkey ;

        column = table->GetColumnBuffer( 4 );
        ASSERT_EQ( column->GetInt32( 0 ), lUpdateLinenumber );
        ++lUpdateLinenumber;

        column = table->GetColumnBuffer( 5 );
        char buffQuantity[ 64 ];
        ASSERT_EQ( column->GetCompactDecimalAsString( 0, 15, 2 ),
                   lUpdateQuantity.GetDecimal( buffQuantity ) );
        lUpdateQuantity += 1;

        column = table->GetColumnBuffer( 9 );
        ASSERT_EQ( column->GetString( 0 ), lUpdateReturnFlag );

        column = table->GetColumnBuffer( 10 );
        ASSERT_EQ( column->GetString( 0 ), lUpdateLinestatus );

        column = table->GetColumnBuffer( 11 );
        ASSERT_EQ( *column->GetDate( 0 ), lUpdateShipdate );
        lUpdateShipdate.day += 1;

        column = table->GetColumnBuffer( 13 );
        ASSERT_EQ( *column->GetDate( 0 ), lUpdateReceiptDate );
        lUpdateReceiptDate.day += 1;

        column = table->GetColumnBuffer( 16 );
        ASSERT_EQ( column->GetString( 0 ), lUpdateComment );
    }
}

TEST_F( UT_TestAriesInitialTable, add_into_empty_table )
{
    string tableName = "empty_table";
    string sql = "drop table if exists " + tableName;
    ExecuteSQL( sql, TEST_DB_NAME );

    sql = "create table " + tableName + "( f1 int not null, f2 char(64) not null, f3 float not null, f4 double not null, f5 decimal(10, 2 ) not null )";
    ExecuteSQL( sql, TEST_DB_NAME );

    AriesInitialTableSPtr initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), 0 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( TEST_DB_NAME );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto columnCount = tableEntry->GetColumnsCount();

    int    baseInt = 1;
    string baseStr( "empty table str" );
    float  baseFloat = 1.01;
    double baseDouble = 2.01;
    int decLen = GetDecimalRealBytes( 10, 2 );
    aries_acc::Decimal baseDecimal( 10, 2, ARIES_MODE_STRICT_ALL_TABLES,
                                    "3.01" );

    int    fInt = baseInt;
    string fStr = baseStr;
    float  fFloat = baseFloat;
    double fDouble = baseDouble;
    aries_acc::Decimal fDecimal = baseDecimal;

    vector< int8_t* > rowsData;
    int32_t bitmapLen = DIV_UP( columnCount, 8 );
    size_t lineStoreLen = tableEntry->GetRowStoreSize();
    lineStoreLen += bitmapLen;
    for ( int i = 0; i < 2; ++i )
    {
        int8_t* rowDataPtr = new int8_t[ lineStoreLen ];
        memset( rowDataPtr, 0, lineStoreLen );

        size_t buffPos = bitmapLen;
        int colId = 1;
        auto colEntry = tableEntry->GetColumnById( colId++ );
        size_t itemStoreSize = colEntry->GetItemStoreSize();
        memcpy( rowDataPtr + buffPos, &fInt, itemStoreSize );
        buffPos += itemStoreSize;
        ++fInt;

        colEntry = tableEntry->GetColumnById( colId++ );
        itemStoreSize = colEntry->GetItemStoreSize();
        string str = fStr + std::to_string( i );
        memcpy( rowDataPtr + buffPos, str.data(), std::min( itemStoreSize, str.size() ) );
        buffPos += itemStoreSize;

        colEntry = tableEntry->GetColumnById( colId++ );
        itemStoreSize = colEntry->GetItemStoreSize();
        memcpy( rowDataPtr + buffPos, &fFloat, itemStoreSize );
        buffPos += itemStoreSize;
        fFloat += 1;

        colEntry = tableEntry->GetColumnById( colId++ );
        itemStoreSize = colEntry->GetItemStoreSize();
        memcpy( rowDataPtr + buffPos, &fDouble, itemStoreSize );
        buffPos += itemStoreSize;
        fDouble += 1;

        colEntry = tableEntry->GetColumnById( colId++ );
        char decBuff[ 128 ] = { 0 };
        aries_acc::Decimal( 10, 2).cast( fDecimal ).ToCompactDecimal( decBuff, decLen );
        itemStoreSize = colEntry->GetItemStoreSize();
        memcpy( rowDataPtr + buffPos, decBuff, itemStoreSize );
        buffPos += itemStoreSize;
        fDecimal += 1;

        rowsData.emplace_back( rowDataPtr );
    }

    vector< index_t > insertedSlotIndice = initialTable->XLogRecoverInsertRows( rowsData );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( initialTable->GetTotalRowCount(), 2 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( insertedSlotIndice.size(), 2 );
    ASSERT_EQ( insertedSlotIndice[ 0 ], 0 );
    ASSERT_EQ( insertedSlotIndice[ 1 ], 1 );
    ASSERT_TRUE( !initialTable->IsSlotFree( 0 ) );
    ASSERT_TRUE( !initialTable->IsSlotFree( 1 ) );

    for ( auto rowDataPtr : rowsData )
        delete[] rowDataPtr;

    initialTable->Sweep();
    initialTable->XLogRecoverDone();
    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->MarkXLogRecoverDone();
    recoveryer->PostXLogRecoverDone();
    initialTable = make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( initialTable->GetTotalRowCount(), 2 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_TRUE( !initialTable->IsSlotFree( 0 ) );
    ASSERT_TRUE( !initialTable->IsSlotFree( 1 ) );

    fInt = baseInt;
    fStr = baseStr;
    fFloat = baseFloat;
    fDouble = baseDouble;
    fDecimal = baseDecimal;
    sql = "select * from " + tableName;
    auto table = ExecuteSQL( sql, TEST_DB_NAME );
    auto resultRowCount = table->GetRowCount();
    ASSERT_EQ( resultRowCount, 2 );

    vector< AriesDataBufferSPtr > columns;
    for ( size_t col = 1; col < columnCount + 1; col++ )
    {
        columns.push_back( table->GetColumnBuffer( col ) );
    }
    for ( int64_t i = 0; i < resultRowCount; ++i )
    {
        // verify columns data
        int colIdx = 0;
        auto column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetInt32( i ), fInt );
        ++fInt;

        column = columns[ colIdx++ ];
        string tmpStr = fStr + std::to_string( i );
        ASSERT_EQ( column->GetString( i ), tmpStr );

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetFloat( i ), fFloat );
        fFloat += 1;

        column = columns[ colIdx++ ];
        ASSERT_EQ( column->GetDouble( i ), fDouble );
        fDouble += 1;

        column = columns[ colIdx++ ];
        char decBuff[ 64 ] = { 0 };
        ASSERT_EQ( column->GetCompactDecimalAsString( i, 10, 2 ), fDecimal.GetDecimal( decBuff ) );
        fDecimal += 1;
    }
}
TEST( TestUpdateInitTable, tpch10_lineitem_update )
{
    string dbName( "init_table_test_tpch10");
    string tableName( "lineitem" );
    uint64_t origTotalRowCount = 59986052;
    int origBlockCount = DIV_UP( origTotalRowCount, ARIES_BLOCK_FILE_ROW_COUNT );
    uint64_t origCapacity = origBlockCount * ARIES_BLOCK_FILE_ROW_COUNT;
    uint64_t lastBlockFreeCount = 2928508;
    AriesInitialTableSPtr initialTable =
        make_shared<AriesInitialTable>( dbName, tableName );

    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetCapacity(), origCapacity );
    ASSERT_EQ( initialTable->GetBlockCount(), origBlockCount );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = dbEntry->GetTableByName( tableName );

    ///////////////////////////////////////
    // test delete
    cout << "test delete...\n";
    vector< index_t > rowsToDelete;
    index_t idx0_0 = 0; // 边缘情况
    index_t idx0_1 = 19999;
    index_t idx0_2 = 199999;
    index_t idx0_3 = ARIES_BLOCK_FILE_ROW_COUNT - 1; // 边缘情况
    rowsToDelete.push_back( idx0_0 );
    rowsToDelete.push_back( idx0_1 );
    rowsToDelete.push_back( idx0_2 );
    rowsToDelete.push_back( idx0_3 );

    index_t idx1_0 = ARIES_BLOCK_FILE_ROW_COUNT; // 边缘情况
    index_t idx1_1 = ARIES_BLOCK_FILE_ROW_COUNT + 18888;
    index_t idx1_2 = ARIES_BLOCK_FILE_ROW_COUNT + 188888;
    index_t idx1_3 = ARIES_BLOCK_FILE_ROW_COUNT + ARIES_BLOCK_FILE_ROW_COUNT - 1; // 边缘情况
    rowsToDelete.push_back( idx1_0 );
    rowsToDelete.push_back( idx1_1 );
    rowsToDelete.push_back( idx1_2 );
    rowsToDelete.push_back( idx1_3 );

    index_t idx2_0 = ARIES_BLOCK_FILE_ROW_COUNT * 2; // 边缘情况
    index_t idx2_1 = ARIES_BLOCK_FILE_ROW_COUNT * 2 + 17777;
    index_t idx2_2 = ARIES_BLOCK_FILE_ROW_COUNT * 2 + 177777;
    index_t idx2_3 = ARIES_BLOCK_FILE_ROW_COUNT * 3 - 1 - lastBlockFreeCount; // 边缘情况
    rowsToDelete.push_back( idx2_0 );
    rowsToDelete.push_back( idx2_1 );
    rowsToDelete.push_back( idx2_2 );
    rowsToDelete.push_back( idx2_3 );

    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }
    // for ( index_t slotIdx = idx2_3 + 1; slotIdx < origCapacity; ++slotIdx )
    // {
    //     ASSERT_TRUE( initialTable->IsSlotFree( slotIdx ) );
    // }

    initialTable->XLogRecoverDeleteRows( rowsToDelete );
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }

    // reopen table meta file
    initialTable = nullptr;
    initialTable = make_shared<AriesInitialTable>( dbName, tableName );
    // verify free slots
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }

    ///////////////////////////////////////
    // test insert
    cout << "test insert...\n";
    int baseInsertOrderkey = 60000001;
    int baseInsertPartkey = 2000001;
    int baseInsertSuppkey = 100001;
    int baseInsertLinenumber = 100;

    int lInsertOrderkey = baseInsertOrderkey;
    int lInsertPartkey = baseInsertPartkey;
    int lInsertSuppkey = baseInsertSuppkey;
    int lInsertLinenumber = baseInsertLinenumber;
    string lInsertQuantity( "100.00" );
    string lInsertExtendedPrice( "1111.11" );
    string lInsertDiscount( "0.05" );
    string lInsertTax( "0.08" );
    string lInsertReturnFlag( "A" );
    string lInsertLinestatus( "F" );
    AriesDate lInsertShipdate = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( "2020-05-04", ARIES_DATE_STRICT_MODE );
    AriesDate lInsertCommitDate = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( "2020-05-04", ARIES_DATE_STRICT_MODE );
    AriesDate lInsertReceiptDate = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( "2020-05-04", ARIES_DATE_STRICT_MODE );
    string lInsertShipInstruct( "INIT TABLE TEST" );
    string lInsertShipMode( "TEST MO" );
    string lInsertComment( "INIT TABLE INSERT TEST" );

    int decLen = GetDecimalRealBytes( 15, 2 );

    aries_acc::Decimal lDecQuantity( 15, 2, ARIES_MODE_STRICT_ALL_TABLES,
                                     lInsertQuantity.data() );
    char buffQuantity[ 128 ];
    lDecQuantity.ToCompactDecimal( buffQuantity, decLen );

    aries_acc::Decimal lDecExtendedPrice( 15, 2, ARIES_MODE_STRICT_ALL_TABLES,
                                          lInsertExtendedPrice.data() );
    char buffExtendedPrice[ 128 ];
    lDecExtendedPrice.ToCompactDecimal( buffExtendedPrice, decLen );

    aries_acc::Decimal lDecDiscount( 15, 2, ARIES_MODE_STRICT_ALL_TABLES,
                                     lInsertDiscount.data() );
    char buffDiscount[ 128 ];
    lDecDiscount.ToCompactDecimal( buffDiscount, decLen );

    aries_acc::Decimal lDecTax( 15, 2, ARIES_MODE_STRICT_ALL_TABLES,
                                lInsertTax.data() );
    char buffTax[ 128 ];
    lDecTax.ToCompactDecimal( buffTax, decLen );

    uint64_t extraRowCount = ARIES_BLOCK_FILE_ROW_COUNT * 2 - 10;
    uint64_t newRowCount = lastBlockFreeCount + extraRowCount;

    vector< int8_t* > rowsData;
    for ( size_t i = 0; i < rowsToDelete.size() + newRowCount; ++i )
    {
        int8_t* rowDataPtr =
            GetInsertRowData( tableEntry,
                              lInsertOrderkey, lInsertPartkey, lInsertSuppkey, lInsertLinenumber,
                              buffQuantity, buffExtendedPrice, buffDiscount, buffTax,
                              lInsertReturnFlag, lInsertLinestatus,
                              lInsertShipdate, lInsertCommitDate, lInsertReceiptDate,
                              lInsertShipInstruct, lInsertShipMode, lInsertComment,
                              decLen );
        rowsData.emplace_back( rowDataPtr );

        lInsertOrderkey++;
        lInsertPartkey++;
        lInsertSuppkey++;
        lInsertLinenumber++;
    }

    uint64_t newTotalRowCount = origTotalRowCount + newRowCount;
    int newBlockCount = origBlockCount + 2;
    uint64_t newCapacity = newBlockCount * ARIES_BLOCK_FILE_ROW_COUNT;
    vector< index_t > insertedSlotIndice = initialTable->XLogRecoverInsertRows( rowsData );
    for ( auto rowData : rowsData )
        delete[] rowData;
    rowsData.clear();

    ASSERT_EQ( initialTable->GetCapacity(), newCapacity );
    ASSERT_EQ( initialTable->GetTotalRowCount(), newTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), newBlockCount );
    ASSERT_EQ( insertedSlotIndice.size(), rowsToDelete.size() + newRowCount );

    // verify holes are filled
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }
    // verify all slots till new total row count are filled
    for ( index_t slotIdx = idx2_3 + 1; ( uint64_t )slotIdx < newTotalRowCount; ++slotIdx )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( slotIdx ) );
    }

    initialTable = make_shared<AriesInitialTable>( dbName, tableName );

    // verify holes are filled
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }
    // verify all slots till new total row count are filled
    for ( index_t slotIdx = idx2_3 + 1; ( uint64_t )slotIdx < newTotalRowCount; ++slotIdx )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( slotIdx ) );
    }
    initialTable = nullptr;

    // verify inserted data
    cout << "verify inserted rows...\n";
    lInsertOrderkey = baseInsertOrderkey;
    lInsertPartkey = baseInsertPartkey;
    lInsertSuppkey = baseInsertSuppkey;
    lInsertLinenumber = baseInsertLinenumber;
    BatchVerifyInsertedData( dbName,
                             rowsToDelete.size() + newRowCount,
                             lInsertOrderkey, lInsertPartkey, lInsertSuppkey, lInsertLinenumber,
                             lInsertQuantity, lInsertExtendedPrice, lInsertDiscount, lInsertTax,
                             lInsertReturnFlag, lInsertLinestatus,
                             lInsertShipdate, lInsertCommitDate, lInsertReceiptDate,
                             lInsertShipInstruct, lInsertShipMode, lInsertComment );

    // test update
    cout << "test update...\n";
    vector< index_t > rowIndiceToUpdate;
    index_t rowIdxToUpdate0_0 = 0;
    index_t rowIdxToUpdate0_1 = 29999;
    index_t rowIdxToUpdate0_2 = 299999;
    index_t rowIdxToUpdate0_3 = ARIES_BLOCK_FILE_ROW_COUNT - 1;
    rowIndiceToUpdate.push_back( rowIdxToUpdate0_0 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate0_1 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate0_2 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate0_3 );

    index_t rowIdxToUpdate1_0 = ARIES_BLOCK_FILE_ROW_COUNT; // 边缘情况
    index_t rowIdxToUpdate1_1 = ARIES_BLOCK_FILE_ROW_COUNT + 28888;
    index_t rowIdxToUpdate1_2 = ARIES_BLOCK_FILE_ROW_COUNT + 288888;
    index_t rowIdxToUpdate1_3 = ARIES_BLOCK_FILE_ROW_COUNT + ARIES_BLOCK_FILE_ROW_COUNT - 1; // 边缘情况
    rowIndiceToUpdate.push_back( rowIdxToUpdate1_0 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate1_1 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate1_2 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate1_3 );

    index_t rowIdxToUpdate2_0 = ARIES_BLOCK_FILE_ROW_COUNT * 2; // 边缘情况
    index_t rowIdxToUpdate2_1 = ARIES_BLOCK_FILE_ROW_COUNT * 2 + 27777;
    index_t rowIdxToUpdate2_2 = ARIES_BLOCK_FILE_ROW_COUNT * 2 + 277777;
    index_t rowIdxToUpdate2_3 = ARIES_BLOCK_FILE_ROW_COUNT * 3 - 1 - lastBlockFreeCount; // 边缘情况
    rowIndiceToUpdate.push_back( rowIdxToUpdate2_0 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate2_1 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate2_2 );
    rowIndiceToUpdate.push_back( rowIdxToUpdate2_3 );

    int baseUpdateSuppkey = 0x7e000000;
    int baseUpdateLinenumber = 0x7f000000;
    double doubleUpdateQuantity = 200.00;
    string lUpdateQuantity( "200.00" );
    string lUpdateReturnFlag( "N" );
    string lUpdateLinestatus( "O" );
    aries_acc::Decimal baseUpdateQuantity( 15, 2, ARIES_MODE_STRICT_ALL_TABLES,
                                           lUpdateQuantity.data() );
    aries_acc::Decimal lUpdateDecQuantity = baseUpdateQuantity;

    AriesDate baseUpdateShipdate = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( "2020-05-01", ARIES_DATE_STRICT_MODE );
    // AriesDate baseUpdateCommitDate = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( "2020-05-01", ARIES_DATE_STRICT_MODE );
    AriesDate baseUpdateReceiptDate = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( "2020-05-01", ARIES_DATE_STRICT_MODE );
    AriesDate lUpdateShipdate = baseUpdateShipdate;
    AriesDate lUpdateReceiptDate = baseUpdateReceiptDate;

    string lUpdateShipInstruct( "INIT TABLE UPDATE" );
    string lUpdateShipMode( "UPDATE" );
    string lUpdateComment( "INIT TABLE UPDATE TEST" );

    vector< UpdateRowDataPtr > updateRowDatas;

    int validFlagLen = DIV_UP( tableEntry->GetColumnsCount(), 8 );
    size_t updateDataLen = 0;
    // l_suppkey
    updateDataLen += tableEntry->GetColumnById( 3 )->GetItemStoreSize();
    // l_linenumber
    updateDataLen += tableEntry->GetColumnById( 4 )->GetItemStoreSize();
    // l_quantity
    updateDataLen += tableEntry->GetColumnById( 5 )->GetItemStoreSize();
    // l_returnflag
    updateDataLen += tableEntry->GetColumnById( 9 )->GetItemStoreSize();
    // l_linestatus
    updateDataLen += tableEntry->GetColumnById( 10 )->GetItemStoreSize();
    // l_shipdate
    updateDataLen += tableEntry->GetColumnById( 11 )->GetItemStoreSize();
    // l_receiptdate
    updateDataLen += tableEntry->GetColumnById( 13 )->GetItemStoreSize();
    // l_comment
    updateDataLen += tableEntry->GetColumnById( 16 )->GetItemStoreSize();

    int lUpdateSuppkey = baseUpdateSuppkey;
    int lUpdateLinenumber = baseUpdateLinenumber;

    for ( auto rowIdx : rowIndiceToUpdate )
    {
        int8_t* updateDataBuff = new int8_t[ validFlagLen + updateDataLen ];
        memset( updateDataBuff, 0, validFlagLen + updateDataLen );
        int8_t* dataStart = updateDataBuff + validFlagLen;
        size_t buffPos = 0;
        size_t itemStoreSize;

        SET_BIT_FLAG( updateDataBuff, 2 );
        itemStoreSize = sizeof( lUpdateSuppkey );
        memcpy( dataStart + buffPos, &lUpdateSuppkey, itemStoreSize );
        buffPos += itemStoreSize;
        ++lUpdateSuppkey;

        SET_BIT_FLAG( updateDataBuff, 3 );
        itemStoreSize = sizeof( lUpdateLinenumber );
        memcpy( dataStart + buffPos, &lUpdateLinenumber, itemStoreSize );
        buffPos += itemStoreSize;
        ++lUpdateLinenumber;

        SET_BIT_FLAG( updateDataBuff, 4 );
        char tmpBuffQuantity[ 128 ];
        if ( !aries_acc::Decimal( 15, 2 ).cast( lUpdateDecQuantity ).ToCompactDecimal( tmpBuffQuantity, decLen ) )
        {
            cout << "To compact decimal failed\n";
            lUpdateDecQuantity = aries_acc::Decimal( 15, 2, ARIES_MODE_STRICT_ALL_TABLES, std::to_string( doubleUpdateQuantity ).data() );
            ASSERT_TRUE( lUpdateDecQuantity.ToCompactDecimal( tmpBuffQuantity, decLen ) );
        }
        itemStoreSize = decLen;
        memcpy( dataStart + buffPos, tmpBuffQuantity, itemStoreSize );
        buffPos += itemStoreSize;
        lUpdateDecQuantity += 1;
        doubleUpdateQuantity += 1;

        SET_BIT_FLAG( updateDataBuff, 8 );
        itemStoreSize = tableEntry->GetColumnById( 9 )->GetItemStoreSize();
        memcpy( dataStart + buffPos, lUpdateReturnFlag.data(), std::min( itemStoreSize, lUpdateReturnFlag.size() ) );
        buffPos += itemStoreSize;

        SET_BIT_FLAG( updateDataBuff, 9 );
        itemStoreSize = tableEntry->GetColumnById( 10 )->GetItemStoreSize();
        memcpy( dataStart + buffPos, lUpdateLinestatus.data(), std::min( itemStoreSize, lUpdateLinestatus.size() ) );
        buffPos += itemStoreSize;

        SET_BIT_FLAG( updateDataBuff, 10 );
        itemStoreSize = tableEntry->GetColumnById( 11 )->GetItemStoreSize();
        memcpy( dataStart + buffPos, &lUpdateShipdate, itemStoreSize );
        buffPos += itemStoreSize;
        lUpdateShipdate.day += 1;

        SET_BIT_FLAG( updateDataBuff, 12 );
        itemStoreSize = tableEntry->GetColumnById( 13 )->GetItemStoreSize();
        memcpy( dataStart + buffPos, &lUpdateReceiptDate, itemStoreSize );
        buffPos += itemStoreSize;
        lUpdateReceiptDate.day += 1;

        SET_BIT_FLAG( updateDataBuff, 15 );
        itemStoreSize = tableEntry->GetColumnById( 16 )->GetItemStoreSize();
        memcpy( dataStart + buffPos, lUpdateComment.data(), std::min( itemStoreSize, lUpdateComment.size() ) );

        UpdateRowDataPtr updateRowData = make_shared< UpdateRowData >();
        updateRowData->m_rowIdx = rowIdx;
        updateRowData->m_colDataBuffs = updateDataBuff;
        updateRowDatas.emplace_back( updateRowData );
    }

    initialTable = make_shared<AriesInitialTable>( dbName, tableName );
    initialTable->UpdateFileRows( updateRowDatas );
    initialTable = nullptr;
    auto mvccTable = AriesMvccTableManager::GetInstance().getMvccTable( dbName, tableName );
    mvccTable->GetInitialTable()->Clear();

    lUpdateSuppkey = baseUpdateSuppkey;
    lUpdateLinenumber = baseUpdateLinenumber;
    lUpdateDecQuantity = baseUpdateQuantity;
    lUpdateShipdate = baseUpdateShipdate;
    lUpdateReceiptDate = baseUpdateReceiptDate;
    BatchVerifyUpdatedData( dbName,
                            rowIndiceToUpdate,
                            lUpdateSuppkey, lUpdateLinenumber,
                            lUpdateDecQuantity,
                            lUpdateReturnFlag, lUpdateLinestatus,
                            lUpdateShipdate,
                            lUpdateReceiptDate,
                            lUpdateComment );

    for ( auto updateRow : updateRowDatas )
    {
        delete[] updateRow->m_colDataBuffs;
    }
}
TEST_F( UT_TestAriesInitialTable, sweep )
{
    string csvPath = "./test_resources/AriesEngine/transaction/TestInitTable/table_without_holes.csv";
    uint64_t origRowCount = 7;
    string tableName = "table_without_holes";
    string sql = "drop table if exists " + tableName;
    ExecuteSQL( sql, TEST_DB_NAME );

    sql = "create table if not exists " + tableName + "( f1 int, f2 varchar( 3 ) )";
    ExecuteSQL( sql, TEST_DB_NAME );

    sql = "load data infile '" + csvPath + "' into table " + tableName;
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    // check initial data
    auto initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    for ( index_t rowIdx = 0; ( uint64_t )rowIdx < origRowCount; ++rowIdx )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }

    sql = "select * from " + tableName;
    auto table = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetRowCount(), origRowCount );
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ), 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ), 7 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbb" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ggg" );

    // sweep table without holes
    initialTable->Sweep();
    // verify meta info
    ASSERT_EQ( initialTable->GetTotalRowCount(), origRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    for ( index_t rowIdx = 0; ( uint64_t )rowIdx < origRowCount; ++rowIdx )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }

    // reload meta info file and verify again
    initialTable->XLogRecoverDone();
    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->MarkXLogRecoverDone();
    recoveryer->PostXLogRecoverDone();
    initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    for ( index_t rowIdx = 0; ( uint64_t )rowIdx < origRowCount; ++rowIdx )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }

    // verify column file content
    // clear loaded column files, ExecuteSQL will re-read column files
    sql = "select * from " + tableName;
    table = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetRowCount(), origRowCount );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 1 );
    ASSERT_EQ( column->GetNullableInt32( 1 ), 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ), 7 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "aaa" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbb" );
    ASSERT_EQ( column->GetNullableString( 6 ), "ggg" );

    // delete rows
    vector< index_t > rowsToDelete;
    rowsToDelete.push_back( 0 );
    rowsToDelete.push_back( 2 );
    rowsToDelete.push_back( 6 ); // delete last row
    initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    initialTable->XLogRecoverDeleteRows( rowsToDelete );
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }
    ASSERT_TRUE( !initialTable->IsSlotFree( 1 ) );
    ASSERT_TRUE( !initialTable->IsSlotFree( 3 ) );
    ASSERT_TRUE( !initialTable->IsSlotFree( 4 ) );
    ASSERT_TRUE( !initialTable->IsSlotFree( 5 ) );

    // sweep table
    initialTable->Sweep();
    // verify meta info
    uint64_t newTotalRowCount = origRowCount - rowsToDelete.size();
    ASSERT_EQ( initialTable->GetTotalRowCount(), newTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    for ( index_t rowIdx = 0; ( uint64_t )rowIdx < newTotalRowCount; ++rowIdx )
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );

    // reload meta info file and verify again
    initialTable->XLogRecoverDone();
    recoveryer->MarkXLogRecoverDone();
    recoveryer->PostXLogRecoverDone();
    initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), newTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    for ( index_t rowIdx = 0; ( uint64_t )rowIdx < newTotalRowCount; ++rowIdx )
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );

    // verify column file content
    // clear loaded column files, ExecuteSQL will re-read column files
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME, tableName );
    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME, tableName );

    sql = "select * from " + tableName;
    table = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetRowCount(), newTotalRowCount );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 6 );
    ASSERT_EQ( column->GetNullableInt32( 1 ), 2 );
    ASSERT_EQ( column->GetNullableInt32( 2 ), 5 );
    ASSERT_EQ( column->GetNullableInt32( 3 ), 4 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "fff" );
    ASSERT_EQ( column->GetNullableString( 1 ), "bbb" );
    ASSERT_EQ( column->GetNullableString( 2 ), "eee" );
    ASSERT_EQ( column->GetNullableString( 3 ), "ddd" );

    // sweep to emtpy
    rowsToDelete.clear();
    rowsToDelete.push_back( 0 );
    rowsToDelete.push_back( 1 );
    rowsToDelete.push_back( 2 );
    rowsToDelete.push_back( 3 );
    initialTable->XLogRecoverDeleteRows( rowsToDelete );
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }

    // sweep table
    initialTable->Sweep();
    // verify meta info
    ASSERT_EQ( initialTable->GetTotalRowCount(), 0 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );

    // reload meta info file and verify again
    initialTable->XLogRecoverDone();
    recoveryer->MarkXLogRecoverDone();
    recoveryer->PostXLogRecoverDone();
    initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), 0 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );

    // verify column file content
    // clear loaded column files, ExecuteSQL will re-read column files
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME, tableName );
    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME, tableName );

    sql = "select * from " + tableName;
    table = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "drop table if exists " + tableName;
    ExecuteSQL( sql, TEST_DB_NAME );
}
TEST_F( UT_TestAriesInitialTable, sweep_empty_table )
{
    string tableName = "empty_table";
    string sql = "drop table if exists " + tableName;
    ExecuteSQL( sql, TEST_DB_NAME );

    sql = "create table if not exists " + tableName + "( f1 int, f2 varchar( 64 ) )";
    ExecuteSQL( sql, TEST_DB_NAME );

    AriesInitialTableSPtr initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), 0 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );

    sql = "select * from " + tableName;
    auto table = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetRowCount(), 0 );

    initialTable->Sweep();
    ASSERT_EQ( initialTable->GetTotalRowCount(), 0 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->MarkXLogRecoverDone();
    recoveryer->PostXLogRecoverDone();
    initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, tableName );
    ASSERT_EQ( initialTable->GetTotalRowCount(), 0 );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );

    auto mvccTable = AriesMvccTableManager::GetInstance().getMvccTable( TEST_DB_NAME, tableName );
    mvccTable->GetInitialTable()->Clear();
    sql = "select * from " + tableName;
    table = ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( table->GetColumnCount(), 2 );
    ASSERT_EQ( table->GetRowCount(), 0 );

    sql = "drop table if exists " + tableName;
    ExecuteSQL( sql, TEST_DB_NAME );
}

/*
TEST_F( UT_TestAriesInitialTable, sweep_tpch10_lineitem )
{
    AriesInitialTableSPtr initialTable =
        make_shared<AriesInitialTable>( TEST_DB_NAME, "lineitem" );

    uint64_t origTotalRowCount = 59986052;
    int origBlockCount = DIV_UP( origTotalRowCount, ARIES_BLOCK_FILE_ROW_COUNT );
    uint64_t origCapacity = origBlockCount * ARIES_BLOCK_FILE_ROW_COUNT;

    // 18043012
    uint64_t lastBlockRowCount = origTotalRowCount % ARIES_BLOCK_FILE_ROW_COUNT;

    vector< index_t > rowsToDelete;
    // block0, delete 10,000,000 rows
    index_t idxDelta = 2;
    uint32_t countToDeleteOfBlock = 10000000;
    uint32_t count = 0;
    for ( index_t i = 0;
          count < countToDeleteOfBlock - 1;
          i += idxDelta, ++count )
    {
        rowsToDelete.push_back( i );
    }
    rowsToDelete.push_back( ARIES_BLOCK_FILE_ROW_COUNT - 1 );

    // block1, delete 8,000,000 rows
    countToDeleteOfBlock = 8000000;
    count = 0;
    for ( index_t i = ARIES_BLOCK_FILE_ROW_COUNT;
          count < countToDeleteOfBlock - 1;
          i += idxDelta, ++count )
    {
        rowsToDelete.push_back( i );
    }
    rowsToDelete.push_back( ARIES_BLOCK_FILE_ROW_COUNT + ARIES_BLOCK_FILE_ROW_COUNT - 1 ); // 边缘情况

    // block2, delete 43012 rows
    countToDeleteOfBlock = 43012;
    count = 0;
    for ( index_t i = ARIES_BLOCK_FILE_ROW_COUNT * 2;
          count < countToDeleteOfBlock - 1;
          i += idxDelta, ++count )
    {
        rowsToDelete.push_back( i );
    }
    rowsToDelete.push_back( ARIES_BLOCK_FILE_ROW_COUNT * 2 + lastBlockRowCount - 1 ); // 边缘情况

    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }

    cout << "Deleting rows...\n";
    initialTable->XLogRecoverDeleteRows( rowsToDelete );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 3 );
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }

    // sweep
    cout << "Sweeping...\n";
    initialTable->Sweep();
    initialTable->XLogRecoverDone();
    origTotalRowCount -= rowsToDelete.size();
    // verify that holes are filled
    ASSERT_EQ( initialTable->GetBlockCount(), 2 );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    for ( int i = 0; i < origTotalRowCount; ++i )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( i ) );
    }

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->MarkXLogRecoverDone();
    recoveryer->PostXLogRecoverDone();
    initialTable = make_shared<AriesInitialTable>( TEST_DB_NAME, "lineitem" );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 2 );
    for ( int i = 0; i < origTotalRowCount; ++i )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( i ) );
    }
    initialTable = nullptr;

    cout << "Verifying...\n";
    string sql = "select count(1) from lineitem";
    auto table = ExecuteSQL( sql, TEST_DB_NAME );
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), origTotalRowCount );

}
*/
namespace aries
{
namespace schema
{
extern bool fix_column_length(const std::string& db_root_path);
}
}
/*
TEST_F( UT_TestAriesInitialTable, sweep_tpch10_lineitem_20971530 )
{
    string dbName = "sweep_tpch10_lineitem_20971530";
    string sql = "drop database if exists " + dbName;
    aries::SQLResultPtr result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "create database if not exists " + dbName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    ASSERT_TRUE( result->IsSuccess() );
    result = aries::SQLExecutor::GetInstance()->ExecuteSQLFromFile( "./test_resources/Schema/create_table.sql", dbName );
    ASSERT_TRUE( result->IsSuccess() );

    string csvFullPath = aries_utils::get_current_work_directory() + "/" + TPCH_10_20971530_LINEITEM_CSV_PATH;
    sql = "load data infile '" + csvFullPath + "' into table lineitem fields terminated by '|'";
    cout << sql << endl;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    bool ret = aries::schema::fix_column_length( aries::SQLExecutor::GetInstance()->getDataDir() );
    ASSERT_TRUE( ret );

    uint64_t origTotalRowCount = 20971530;
    int origBlockCount = DIV_UP( origTotalRowCount, ARIES_BLOCK_FILE_ROW_COUNT );
    uint64_t origCapacity = origBlockCount * ARIES_BLOCK_FILE_ROW_COUNT;

    sql = "select count(1) from lineitem";
    auto table = ExecuteSQL( sql, dbName );
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), origTotalRowCount );

    AriesInitialTableSPtr initialTable =
        make_shared<AriesInitialTable>( dbName, "lineitem" );

    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 2 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT * 2 );

    // 10
    uint64_t lastBlockRowCount = origTotalRowCount % ARIES_BLOCK_FILE_ROW_COUNT;

    vector< index_t > rowsToDelete;
    rowsToDelete.push_back( 0 );
    rowsToDelete.push_back( 10000 );
    rowsToDelete.push_back( ARIES_BLOCK_FILE_ROW_COUNT - 1 );
    int count = 0;
    for ( index_t i = ARIES_BLOCK_FILE_ROW_COUNT; count < 7; ++i, ++count )
        rowsToDelete.push_back( i );

    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( rowIdx ) );
    }

    cout << "Deleting rows...\n";
    initialTable->XLogRecoverDeleteRows( rowsToDelete );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 2 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT * 2 );
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }

    // reopen table meta file
    initialTable = nullptr;
    initialTable = make_shared<AriesInitialTable>( dbName, "lineitem" );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 2 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT * 2 );
    // verify free slots
    for ( auto rowIdx : rowsToDelete )
    {
        ASSERT_TRUE( initialTable->IsSlotFree( rowIdx ) );
    }

    // sweep
    cout << "Sweeping...\n";
    initialTable->Sweep();
    origTotalRowCount -= rowsToDelete.size();
    // verify that holes are filled
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetCapacity(), ARIES_BLOCK_FILE_ROW_COUNT );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    for ( int i = 0; i < origTotalRowCount; ++i )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( i ) );
    }

    initialTable = nullptr;
    initialTable = make_shared<AriesInitialTable>( dbName, "lineitem" );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    ASSERT_EQ( initialTable->GetBlockCount(), 1 );
    ASSERT_EQ( initialTable->GetTotalRowCount(), origTotalRowCount );
    for ( int i = 0; i < origTotalRowCount; ++i )
    {
        ASSERT_TRUE( !initialTable->IsSlotFree( i ) );
    }

    cout << "Verifying...\n";
    sql = "select count(1) from lineitem";
    auto mvccTable = AriesMvccTableManager::GetInstance().getTable( dbName, "lineitem" );
    mvccTable->GetInitialTable()->Clear();
    mvccTable->GetInitialTable()->GetMetaInfo();
    table = ExecuteSQL( sql, dbName );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), origTotalRowCount );

}
*/

TEST_F( UT_TestAriesInitialTable, GetTable )
{
  string dbName = TEST_DB_NAME;
  string tableName = "nation";
  InitTable( dbName, tableName );

  string cwd = aries_utils::get_current_work_directory();

  string sql = R"(
create table nation  ( n_nationkey  integer not null primary key,
                       n_name       char(25) not null,
                       n_regionkey  integer not null,
                       n_comment    varchar(152) );
  )";
  auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
  ASSERT_TRUE( result->IsSuccess() );

  string csvFilePath = cwd + "/test_resources/AriesEngine/transaction/TestInitTable/nation.tbl";
  sql = "load data infile '" + csvFilePath + "' into table " + tableName + " fields terminated by '|'";
  result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
  ASSERT_TRUE( result->IsSuccess() );

  int64_t lineCnt = 25;
  vector< int32_t > columnIds;
  columnIds.push_back( 1 );
  columnIds.push_back( 2 );
  columnIds.push_back( 3 );
  columnIds.push_back( 4 );
  AriesInitialTableSPtr initialTable =
      make_shared<AriesInitialTable>( dbName, tableName );

  auto table = initialTable->GetTable( columnIds );
  ASSERT_EQ( table->GetRowCount(), lineCnt );

  auto column = table->GetColumnBuffer( 1, true );
  ASSERT_EQ( 0, column->GetInt32( 0 ) );
  ASSERT_EQ( 11, column->GetInt32( 11 ) );
  ASSERT_EQ( 24, column->GetInt32( 24 ) );

  column = table->GetColumnBuffer( 2, true );
  ASSERT_EQ( "ALGERIA", column->GetString(0) );
  ASSERT_EQ( "IRAQ", column->GetString(11) );
  ASSERT_EQ( "UNITED STATES", column->GetString(24) );

  column = table->GetColumnBuffer( 3, true );
  ASSERT_EQ( 0, column->GetInt32( 0 ) );
  ASSERT_EQ( 4, column->GetInt32( 11 ) );
  ASSERT_EQ( 1, column->GetInt32( 24 ) );

  column = table->GetColumnBuffer( 4, true );
  ASSERT_EQ(" haggle. carefully final deposits detect slyly agai",
            column->GetString(0));
  ASSERT_EQ(
      "nic deposits boost atop the quickly final requests? quickly regula",
      column->GetString(11));
  ASSERT_EQ(
      "y final packages. slow foxes cajole quickly. quickly silent platelets "
      "breach ironic accounts. unusual pinto be",
      column->GetString(24));

  // tableName = "customer";
  // lineCnt = 150000;
  // columnIds.push_back( 5 );
  // columnIds.push_back( 6 );
  // columnIds.push_back( 7 );
  // columnIds.push_back( 8 );
  // initialTable =
  //     make_shared<AriesInitialTable>( dbName, tableName );
  // table = initialTable->GetTable( columnIds );
  // ASSERT_EQ( table->GetRowCount(), lineCnt );
  // column = table->GetColumnBuffer( 1, true );
  // ASSERT_EQ( 1, column->GetInt32( 0 ) );
  // ASSERT_EQ( 1000, column->GetInt32( 999 ) );
  // ASSERT_EQ( 150000, column->GetInt32( 149999 ) );

  // column = table->GetColumnBuffer( 2, true );
  // ASSERT_EQ("Customer#000000001", column->GetString(0));
  // ASSERT_EQ("Customer#000010000", column->GetString(9999));
  // ASSERT_EQ("Customer#000149999", column->GetString(149998));

  // column = table->GetColumnBuffer( 6, true );
  // ASSERT_EQ(column->GetDecimalAsString(0), "711.56");
  // ASSERT_EQ(column->GetDecimalAsString(99999), "4840.82");
  // ASSERT_EQ(column->GetDecimalAsString(149997), "5952.41");

  // dbName = "tpch_100";
  // tableName = "partsupp";
  // lineCnt = 80000000;
  // initialTable =
  //     make_shared<AriesInitialTable>( dbName, tableName, lineCnt );
  // int64_t readTime;
  // float s;
  // CPU_Timer t;
  // t.begin();
  // columnIds.clear();
  // columnIds.push_back( 1 );
  // columnIds.push_back( 2 );
  // columnIds.push_back( 3 );
  // columnIds.push_back( 4 );
  // columnIds.push_back( 5 );
  // readCnt = initialTable->CacheColumnData( columnIds );
  // ASSERT_EQ( readCnt, lineCnt );
  // readTime = t.end();
  // s = ( readTime + 0.0 ) / 1000;
  // cout << "Read time: " << s << "s\n";

  // tableName = "customer";
  // lineCnt = 15000000;
  // initialTable =
  //     make_shared<AriesInitialTable>( dbName, tableName, lineCnt );
  // t.begin();
  // readCnt = initialTable->CacheColumnData( columnIds );
  // ASSERT_EQ( readCnt, lineCnt );
  // readTime = t.end();
  // s = ( readTime + 0.0 ) / 1000;
  // cout << "Read time: " << s << "s\n";
}

TEST_F(UT_TestAriesInitialTable, Partition)
{
    string dbName = TEST_DB_NAME;
    string tableName = "lineitem";
    InitTable(dbName, tableName);

    string cwd = aries_utils::get_current_work_directory();

    string sql = R"(
create table lineitem
( l_orderkey    char(40),
l_partkey     integer not null,
l_suppkey     integer not null,
l_linenumber  integer not null,
l_quantity    decimal(12,2) not null,
l_extendedprice  decimal(12,2) not null,
l_discount    decimal(12,2) not null,
l_tax         decimal(12,2) not null,
l_returnflag  char(1) not null,
l_linestatus  char(1) not null,
l_shipdate    date not null,
l_commitdate  date not null,
l_receiptdate date not null,
l_shipinstruct char(25),
l_shipmode     char(10),
l_comment      varchar(44) not null )
partition by range(l_shipdate)
(partition p0 values less than ('1992-02-01'),
partition p1 values less than ('1994-03-01'),
partition p2 values less than ('1995-03-01'),
partition p3 values less than ('1995-04-01'),
partition p4 values less than ('1995-05-01'),
partition p5 values less than ('1995-06-01'),
partition p6 values less than ('1995-07-01'),
partition p7 values less than ('1995-08-01'),
partition p8 values less than ('1995-09-01'),
partition p9 values less than ('1995-10-01'),
partition p10 values less than ('1995-11-01'),
partition p11 values less than ('1995-12-01'),
partition p12 values less than ('1996-03-01'),
partition p13 values less than ('1996-04-01'),
partition p14 values less than ('1996-05-01'),
partition p15 values less than ('1996-06-01'),
partition p16 values less than ('1996-07-01'),
partition p17 values less than ('1996-08-01'),
partition p18 values less than ('1996-09-01'),
partition p19 values less than ('1996-10-01'),
partition p20 values less than ('1996-11-01'),
partition p21 values less than ('1996-12-01'),
partition p22 values less than ('1997-12-01'),
partition p23 values less than maxvalue
);
  )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());

    string csvFilePath = cwd + "/test_resources/AriesEngine/transaction/TestInitTable/lineitem.tbl";
    sql = "load data infile '" + csvFilePath + "' into table " + tableName + " fields terminated by '|'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, dbName);
    ASSERT_TRUE(result->IsSuccess());


    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    auto tableEntry = dbEntry->GetTableByName(tableName);
    ASSERT_TRUE(tableEntry->IsPartitioned());
    int partitionColumnId = tableEntry->GetPartitionColumnIndex() + 1;

    ASSERT_EQ(partitionColumnId, 11);

    vector<int32_t> columnIds;
    columnIds.push_back(1); // l_orderkey
    columnIds.push_back(4); // l_linenumber
    columnIds.push_back(partitionColumnId);

    AriesInitialTableSPtr initialTable = make_shared<AriesInitialTable>(dbName, tableName);
    auto partitionInfo = initialTable->GetPartitionMetaInfo();
    auto partitionConds = tableEntry->GetPartitions();

    size_t tablePartitionCount = partitionInfo.size();
    size_t schemaPartitionCount = tableEntry->GetPartitionCount();
    ASSERT_EQ(tablePartitionCount, 24);
    ASSERT_EQ(schemaPartitionCount, 24);
    auto table = initialTable->GetTable(columnIds);
    ASSERT_EQ(table->GetRowCount(), 6001215);

    int index = 0;
    size_t blockCount = table->GetBlockCount();
    for( const auto& part : partitionInfo )
    {
        for( auto blockId : part.BlocksID )
        {
            assert( blockId < blockCount );
            auto tableBlock = table->GetOneBlock( blockId );
            auto column = tableBlock->GetColumnBuffer( 3 );
            AriesDate* pDate = ( AriesDate* )column->GetData();
            for( int i = 0; i < column->GetItemCount(); ++i )
            {
                ASSERT_TRUE( pDate->toTimestamp() < partitionConds[ index ]->m_value );
                if( index > 0 )
                    ASSERT_TRUE( pDate->toTimestamp() >= partitionConds[ index - 1 ]->m_value );
                ++pDate;
            }
        }
        ++index;
    }

    sql = R"(SELECT
  l_shipdate,
  l_shipinstruct,
  SUM(l_quantity) amt
FROM
  lineitem
WHERE
  l_shipinstruct != ''
GROUP BY
  l_shipdate,
  l_shipinstruct
ORDER BY
  amt DESC
limit 10;)";

    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    EXPECT_TRUE( result->IsSuccess() );
}
