//
// Created by david.shen on 2020/4/8.
//

#include <gtest/gtest.h>

#include "schema/SchemaManager.h"
#include "AriesEngine/transaction/AriesTuple.h"
#include "AriesEngine/transaction/AriesDeltaTable.h"
#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "../../../TestUtils.h"

using namespace aries_engine;
using namespace aries_test;

#define TEST_TUPLE_DB "TEST_DB_TUPLE"
#define TEST_TUPLE_TABLE_LINEITEM "LINEITEM"
#define TEST_TUPLE_TABLE_NATION "NATION"

class UT_TestAriesTuple : public ::testing::Test
{
protected:
    std::string db_name = TEST_TUPLE_DB;

protected:
    void SetUp() override
    {
        auto sql = R"(create database if not exists TEST_DB_TUPLE; )";
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
        ASSERT_TRUE( result->IsSuccess() );

        sql = R"(CREATE TABLE if not exists LINEITEM ( L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL);)";
        result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
        ASSERT_TRUE( result->IsSuccess() );
        sql = R"(CREATE TABLE if not exists NATION ( N_NATIONKEY  INTEGER NOT NULL,
                            N_NAME       CHAR(25) NOT NULL,
                            N_REGIONKEY  INTEGER NOT NULL,
                            N_COMMENT    VARCHAR(152));)";
        result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
        ASSERT_TRUE( result->IsSuccess() );
    }

    void TearDown() override
    {
        auto sql = "drop database TEST_DB_TUPLE;";
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
        ASSERT_TRUE( result->IsSuccess() );
    }
};

TEST_F( UT_TestAriesTuple, FillData )
{
    /*
    for lineitem:
    CREATE TABLE LINEITEM ( L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL);
    */
    string dbName = TEST_TUPLE_DB;
    string tableName = TEST_TUPLE_TABLE_LINEITEM;
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = database->GetTableByName( tableName );
    auto tupleParser = make_shared<TupleParser>( tableEntry );
    auto deltaTable = make_shared< AriesDeltaTable >( 1, tupleParser->GetColumnTypes() );
    int columnId = 0;
    // --> int32_t

    // int8 --> int
    auto tupleData = make_shared<TupleData>();
    auto buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT8, 1 }, false }, 1);
    *(buf->GetItemDataAt(0)) = 1;
    tupleData->data[++columnId] = buf;

    // uint8 --> int
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::UINT8, 1 }, false }, 1);
    *(uint8_t *)(buf->GetItemDataAt(0)) = -1;
    tupleData->data[++columnId] = buf;

    // int16_6 --> int
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT16, 1 }, false }, 1);
    *(int16_t *)(buf->GetItemDataAt(0)) = 1;
    tupleData->data[++columnId] = buf;

    // uint32 --> int
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::UINT32, 1 }, false }, 1);
    *(uint32_t *)(buf->GetItemDataAt(0)) = -1;
    tupleData->data[++columnId] = buf;

    // decimal --> compactaries_acc::Decimal
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::DECIMAL, 6, 2 }, false }, 1);
    auto decimal = aries_acc::Decimal(6, 2, "1188.88");
    memcpy(buf->GetItemDataAt(0), &decimal, sizeof(aries_acc::Decimal));
    tupleData->data[++columnId] = buf;

    // compact decimal --> compactaries_acc::Decimal
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::COMPACT_DECIMAL, 6, 2 }, true }, 1);
    auto dataBuf = buf->GetItemDataAt(0);
    *dataBuf++ = 1;
    decimal = aries_acc::Decimal(6, 2, "1188.88").ToCompactDecimal((char *)dataBuf, buf->GetDataType().DataType.Length);
    tupleData->data[++columnId] = buf;

    // nullable int --> compactaries_acc::Decimal
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT32, 1 }, true }, 1);
    dataBuf = buf->GetItemDataAt(0);
    *dataBuf++ = 1;
    *(int32_t *)dataBuf = 1;
    tupleData->data[++columnId] = buf;

    // nullable decimal --> compactaries_acc::Decimal
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::DECIMAL, 6, 2 }, true }, 1);
    decimal = aries_acc::Decimal(6, 2, "1188.88");
    dataBuf = buf->GetItemDataAt(0);
    *dataBuf++ = 1;
    memcpy(dataBuf, &decimal, sizeof(aries_acc::Decimal));
    tupleData->data[++columnId] = buf;

    // nullalbe char --> char
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 1 }, true }, 1);
    dataBuf = buf->GetItemDataAt(0);
    *dataBuf++ = 1;
    *(char *)dataBuf = 'A';
    tupleData->data[++columnId] = buf;

    // char --> char
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 1 }, false }, 1);
    dataBuf = buf->GetItemDataAt(0);
    *(char *)dataBuf = 'A';
    tupleData->data[++columnId] = buf;

    // date --> date
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::DATE, 1 }, false }, 1);
    dataBuf = buf->GetItemDataAt(0);
    auto date = AriesDate(2020, 4, 8);
    memcpy(dataBuf, &date, sizeof(AriesDate));
    tupleData->data[++columnId] = buf;

    // datetime --> date
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::DATETIME, 1 }, false }, 1);
    dataBuf = buf->GetItemDataAt(0);
    auto datetime = AriesDatetime(2020, 4, 8, 0, 0, 0, 0);
    memcpy(dataBuf, &datetime, sizeof(AriesDatetime));
    tupleData->data[++columnId] = buf;

    // timestamp --> date
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::TIMESTAMP, 1 }, false }, 1);
    dataBuf = buf->GetItemDataAt(0);
    auto timestamp = AriesTimestamp(1586304000000001);
    memcpy(dataBuf, &timestamp, sizeof(AriesDatetime));
    tupleData->data[++columnId] = buf;

    // char(19) --> char(25)
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 19 }, false }, 1);
    dataBuf = buf->GetItemDataAt(0);
    memcpy(dataBuf, "This is a testcase.", 19);
    tupleData->data[++columnId] = buf;

    // char(19) --> char(10)
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 10 }, false }, 1);
    dataBuf = buf->GetItemDataAt(0);
    memcpy(dataBuf, "This is a testcase.", 10);
    tupleData->data[++columnId] = buf;

    // nullable char(19) --> char(44)
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 19 }, true }, 1);
    dataBuf = buf->GetItemDataAt(0);
    *dataBuf++ = 1;
    memcpy(dataBuf, "This is a testcase.", 19);
    tupleData->data[++columnId] = buf;

    ASSERT_TRUE(( size_t )columnId == tupleParser->GetColumnsCount());

    //fill data
    bool unused;
    auto rowPos = deltaTable->ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused )[0];
    std::vector< int8_t* > columnBuffers;
    columnBuffers.resize( 16 );
    deltaTable->GetTupleFieldBuffer( rowPos, columnBuffers );
    tupleParser->FillData( columnBuffers, tupleData, 0 );

    // check result
    char r[64];
    columnId = 0;
    // 4 integers
    auto colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(int32_t *)colBuf == 1);
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(int32_t *)colBuf == 255);
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(int32_t *)colBuf == 1);
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(int32_t *)colBuf == -1);
    // 4 decimal(15,2)
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(string(aries_acc::Decimal((CompactDecimal *)colBuf, 15, 2).GetDecimal(r)) == "1188.88");
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(string(aries_acc::Decimal((CompactDecimal *)colBuf, 15, 2).GetDecimal(r)) == "1188.88");
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(string(aries_acc::Decimal((CompactDecimal *)colBuf, 15, 2).GetDecimal(r)) == "1.00");
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(string(aries_acc::Decimal((CompactDecimal *)colBuf, 15, 2).GetDecimal(r)) == "1188.88");
    // 2 char(1)
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(char *)colBuf == 'A');
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(char *)colBuf == 'A');
    // 3 date
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(AriesDate *)colBuf == AriesDate(2020, 4, 8));
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(AriesDate *)colBuf == AriesDate(2020, 4, 8));
    colBuf = columnBuffers[ columnId++ ];

    ASSERT_TRUE(*(AriesDate *)colBuf == AriesDate(2020, 4, 8));

    // 1 char(25)
    int len = 19;
    colBuf = columnBuffers[ columnId++ ];
    memcpy(r, colBuf, len);
    r[len] = 0;
    ASSERT_TRUE(string(r) == "This is a testcase.");

    // 1 char(10)
    len = 10;
    colBuf = columnBuffers[ columnId++ ];
    memcpy(r, colBuf, len);
    r[len] = 0;
    ASSERT_TRUE(string(r) == "This is a ");

    // 1 char(44)
    len = 19;
    colBuf = columnBuffers[ columnId++ ];
    memcpy(r, colBuf, len);
    r[len] = 0;
    ASSERT_TRUE(string(r) == "This is a testcase.");
    ASSERT_TRUE(( size_t )columnId == tupleParser->GetColumnsCount());
}

TEST_F( UT_TestAriesTuple, FillData_Exception_SameType_Null )
{
    string dbName = TEST_TUPLE_DB;
    string tableName = TEST_TUPLE_TABLE_LINEITEM;
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = database->GetTableByName( tableName );
    auto tupleParser = make_shared<TupleParser>(tableEntry);
    auto deltaTable = make_shared< AriesDeltaTable >( 1, tupleParser->GetColumnTypes() );
    int columnId = 0;
    // same type: nullable null --> not nullable
    auto tupleData = make_shared<TupleData>();
    auto buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT32, 1 }, true }, 1);
    *(buf->GetItemDataAt(0)) = 0;
    tupleData->data[++columnId] = buf;
    try {
        //fill data
        bool unused;
        auto rowPos = deltaTable->ReserveSlot(1, AriesDeltaTableSlotType::AddedTuples, unused)[0];
        auto deltaBuf = deltaTable->GetTupleFieldBuffer(rowPos);
        tupleParser->FillData({deltaBuf}, tupleData, 0);
        ASSERT_TRUE(false);
    } catch (...) {
        ASSERT_TRUE(true);
    }
}

TEST_F( UT_TestAriesTuple, FillData_Exception_DiffType_Null )
{
    string dbName = TEST_TUPLE_DB;
    string tableName = TEST_TUPLE_TABLE_LINEITEM;
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = database->GetTableByName( tableName );
    auto tupleParser = make_shared<TupleParser>(tableEntry);
    auto deltaTable = make_shared< AriesDeltaTable >( 1, tupleParser->GetColumnTypes() );
    int columnId = 0;
    // diff type: nullable null --> not nullable
    auto tupleData = make_shared<TupleData>();
    auto buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT8, 1 }, true }, 1);
    *(buf->GetItemDataAt(0)) = 0;
    tupleData->data[++columnId] = buf;
    try {
        //fill data
        bool unused;
        auto rowPos = deltaTable->ReserveSlot(1, AriesDeltaTableSlotType::AddedTuples, unused)[0];
        auto deltaBuf = deltaTable->GetTupleFieldBuffer(rowPos);
        tupleParser->FillData({deltaBuf}, tupleData, 0);
        ASSERT_TRUE(false);
    } catch (...) {
        ASSERT_TRUE(true);
    }
}

TEST_F( UT_TestAriesTuple, FillData_Exception_Cannot_Convert )
{
    string dbName = TEST_TUPLE_DB;
    string tableName = TEST_TUPLE_TABLE_LINEITEM;
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = database->GetTableByName( tableName );
    auto tupleParser = make_shared<TupleParser>(tableEntry);
    auto deltaTable = make_shared< AriesDeltaTable >( 1, tupleParser->GetColumnTypes() ); 
    int columnId = 0;
    // can't convert type: int64_t --> int32_t
    auto tupleData = make_shared<TupleData>();
    auto buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT64, 1 }, false }, 1);
    *(int64_t *)(buf->GetItemDataAt(0)) = 1;
    tupleData->data[++columnId] = buf;
    try {
        //fill data
        bool unused;
        auto rowPos = deltaTable->ReserveSlot(1, AriesDeltaTableSlotType::AddedTuples, unused)[0];
        auto deltaBuf = deltaTable->GetTupleFieldBuffer(rowPos);
        tupleParser->FillData({deltaBuf}, tupleData, 0);
        ASSERT_TRUE(false);
    } catch (...) {
        ASSERT_TRUE(true);
    }
}

class UT_TestAriesTupleClass : public ::testing::Test
{
protected:
    std::string db_name = "test_tuple_class" ;
    std::string table_name = "testariestuple_transferdata";
protected:
    void SetUp() override
    {
        InitTable( db_name, table_name );
        auto sql = R"(
create table TestAriesTuple_TransferData
(
    id int,
    name char(64),
    age tinyint,
    score bigint,
    balance decimal( 15 , 2 ),
    birth_year year,
    birth_day date,
    birth_datetime datetime
);
        )";

        auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
        ASSERT_TRUE( result->IsSuccess() );
    }

    void TearDown() override
    {
        auto sql = "drop database " + db_name;
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
        ASSERT_TRUE( result->IsSuccess() );
    }
};

TEST_F( UT_TestAriesTupleClass, TransferData )
{
    auto sql = R"(
insert into TestAriesTuple_TransferData values(
    '10', 'jack', '28', '1000000000000', '100.99', '2002', '2002-01-23', '2002-01-23 08:12:43'
);
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from TestAriesTuple_TransferData;";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );

    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] );
    auto table = mem_table->GetContent();

    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnCount(), 8 );
    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( 0 ), "10" );
    ASSERT_EQ( table->GetColumnBuffer( 2 )->GetString( 0 ), "jack" );
    ASSERT_EQ( table->GetColumnBuffer( 3 )->GetNullableInt8( 0 ), 28 );
    ASSERT_EQ( table->GetColumnBuffer( 4 )->GetInt64AsString( 0 ), "1000000000000" );
    ASSERT_EQ( table->GetColumnBuffer( 5 )->GetDecimalAsString( 0 ), "100.99" );
    ASSERT_EQ( table->GetColumnBuffer( 6 )->GetYearAsString( 0 ), "2002" );
    ASSERT_EQ( table->GetColumnBuffer( 7 )->GetDateAsString( 0 ), "2002-01-23" );
    ASSERT_EQ( table->GetColumnBuffer( 8 )->GetDatetimeAsString( 0 ), "2002-01-23 08:12:43" );
}

TEST_F( UT_TestAriesTuple, FillData_NULL_Data )
{
    /*
    CREATE TABLE NATION  ( N_NATIONKEY  INTEGER NOT NULL,
                            N_NAME       CHAR(25) NOT NULL,
                            N_REGIONKEY  INTEGER NOT NULL,
                            N_COMMENT    VARCHAR(152));
    */
    string dbName = TEST_TUPLE_DB;
    string tableName = TEST_TUPLE_TABLE_NATION;
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = database->GetTableByName( tableName );
    auto tupleParser = make_shared<TupleParser>(tableEntry);
    auto deltaTable = make_shared< AriesDeltaTable >( 1, tupleParser->GetColumnTypes() );
    int columnId = 0;
    auto tupleData = make_shared<TupleData>();

    // uint16 --> int32_t
    auto buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::UINT16, 1 }, false }, 1);
    *(uint16_t *)(buf->GetItemDataAt(0)) = -1;
    tupleData->data[++columnId] = buf;

    // char(19) --> char(25)
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 19 }, false }, 1);
    auto dataBuf = buf->GetItemDataAt(0);
    memcpy(dataBuf, "This is a testcase.", 19);
    tupleData->data[++columnId] = buf;

    // int32_t --> int32_t
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::INT32, 1 }, false }, 1);
    *(int32_t *)(buf->GetItemDataAt(0)) = -1;
    tupleData->data[++columnId] = buf;

    // null data --> null data
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 10 }, true }, 1);
    *(buf->GetItemDataAt(0)) = 0;
    tupleData->data[++columnId] = buf;

    //fill data
    std::vector< int8_t* > columnBuffers;
    columnBuffers.resize( 4 );
    bool unused;
    auto rowPos = deltaTable->ReserveSlot(1, AriesDeltaTableSlotType::AddedTuples, unused)[0];
    deltaTable->GetTupleFieldBuffer( rowPos, columnBuffers, { 1, 2, 3, 4 } );
    tupleParser->FillData(columnBuffers, tupleData, 0);

    // check result
    char r[64];
    columnId = 0;
    // 4 integers
    auto colBuf = columnBuffers[ columnId++ ];
    ASSERT_TRUE(*(int32_t *)colBuf == 65535);
    int len = 19;
    colBuf = columnBuffers[ columnId++ ];
    memcpy(r, colBuf, len);
    r[len] = 0;
    ASSERT_TRUE(string(r) == "This is a testcase.");
    colBuf = columnBuffers[ columnId++ ];
    ASSERT_TRUE(*(int32_t *)colBuf == -1);
    colBuf = columnBuffers[ columnId++ ];
    ASSERT_TRUE(*colBuf == 0);
}

TEST_F( UT_TestAriesTuple, FillData_Char2String )
{
    string dbName = TEST_TUPLE_DB;
    string tableName = TEST_TUPLE_TABLE_NATION;
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    auto tableEntry = database->GetTableByName( tableName );
    auto tupleParser = make_shared<TupleParser>(tableEntry);
    auto deltaTable = make_shared< AriesDeltaTable >( 1, tupleParser->GetColumnTypes() );
    int columnId = 0;
    auto tupleData = make_shared<TupleData>();

    // bad normal string --> int32_t
    auto buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 4 }, false }, 1);
    memcpy((char *)(buf->GetItemDataAt(0)), "1234", 4);
    tupleData->data[++columnId] = buf;
    
    // char(19) --> char(25)
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 19 }, false }, 1);
    auto dataBuf = buf->GetItemDataAt(0);
    memcpy(dataBuf, "This is a testcase.", 19);
    tupleData->data[++columnId] = buf;

    // bad nullable string --> int32_t
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 4 }, true }, 1);
    dataBuf = buf->GetItemDataAt(0);
    *dataBuf++ = 1;
    memcpy(dataBuf, "1234", 4);
    tupleData->data[++columnId] = buf;

    // null data --> null data
    buf = make_shared<AriesDataBuffer>(AriesColumnType{ { AriesValueType::CHAR, 10 }, true }, 1);
    *(buf->GetItemDataAt(0)) = 0;
    tupleData->data[++columnId] = buf;

    //fill data
    std::vector< int8_t* > columnBuffers;
    columnBuffers.resize( 4 );
    bool unused;
    auto rowPos = deltaTable->ReserveSlot(1, AriesDeltaTableSlotType::AddedTuples, unused)[0];
    deltaTable->GetTupleFieldBuffer( rowPos, columnBuffers, { 1, 2, 3, 4 } );
    tupleParser->FillData(columnBuffers, tupleData, 0);

    // check result
    columnId = 0;
    char r[64];
    // 4 integers
    auto colBuf = columnBuffers[ columnId++ ];
    ASSERT_TRUE(*(int32_t *)colBuf == 1234);
    int len = 19;
    colBuf = columnBuffers[ columnId++ ];
    memcpy(r, colBuf, len);
    r[len] = 0;
    ASSERT_TRUE(string(r) == "This is a testcase.");
    colBuf = columnBuffers[ columnId++ ];
    ASSERT_TRUE(*(int32_t *)colBuf == 1234);
    colBuf = columnBuffers[ columnId++ ];
    ASSERT_TRUE(*colBuf == 0);
}

