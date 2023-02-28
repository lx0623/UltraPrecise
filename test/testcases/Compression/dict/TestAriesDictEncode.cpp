#include "gtest/gtest.h"
#include "frontend/SQLExecutor.h"
#include "schema/SchemaManager.h"
#include "Compression/dict/AriesDictManager.h"
#include "AriesEngine/transaction/AriesInitialTable.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "Compression/dict/AriesDict.h"
#include "utils/string_util.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "server/Configuration.h"

#include "../../../TestUtils.h"

using namespace aries_test;
using namespace aries_engine;
using namespace aries_utils;
using namespace std;

extern bool STRICT_MODE;

const string TEST_DB_NAME( "ut_dict_encode" );
const string TEST_DB_NAME_XLOG( "ut_dict_encode_xlog" );

string testDbName1 = TEST_DB_NAME;
string testDbName2 = "ut_dict_encode2";

class UT_dict_encode : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        ResetTestDatabase();

        string sql = "create database if not exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    }
    static void TearDownTestCase()
    {
        ResetTestDatabase();
    }
private:
    static void ResetTestDatabase()
    {
        string sql = "drop database if exists " + TEST_DB_NAME;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );

        sql = "drop database if exists " + testDbName2;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, testDbName2 );

        sql = "drop database if exists " + TEST_DB_NAME_XLOG;
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    }
};

TEST_F( UT_dict_encode, exceed_len )
{
    string tableName = "t_exceed_len";
    string dictName( "dict_UT_dict_encode_exceed_len" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    string sql( "create table " + tableName + "( f2 char(4) not null encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    // load null into not null columns
    STRICT_MODE = true;
    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_exceed_len.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    sql = "insert into " + tableName + " values ( 'bbbbb' )" ;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    // non strict mode
    STRICT_MODE = false;
    csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_exceed_len.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "aaaa" );

    auto dict = AriesDictManager::GetInstance().GetDict( dictName );
    auto dictBuff = dict->getDictBuffer();
    ASSERT_EQ( dictBuff->GetString( 0 ), "aaaa" );

    sql = "insert into " + tableName + " values ( 'bbbbb' )" ;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "aaaa" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "bbbb" );

    ASSERT_EQ( dictBuff->GetString( 0 ), "aaaa" );
    ASSERT_EQ( dictBuff->GetString( 1 ), "bbbb" );

    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    dict = AriesDictManager::GetInstance().GetDict( dictName );
    ASSERT_EQ( dict->getDictBuffer(), nullptr );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "aaaa" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "bbbb" );

    dict = AriesDictManager::GetInstance().GetDict( dictName );
    dictBuff = dict->getDictBuffer();
    ASSERT_EQ( dictBuff->GetString( 0 ), "aaaa" );
    ASSERT_EQ( dictBuff->GetString( 1 ), "bbbb" );

    STRICT_MODE = true;
    InitTable( TEST_DB_NAME, tableName );
    sql= "create table " + tableName + "( f2 char(4) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    csvPath = cwd + "/test_resources/loaddata/csv/char_excceed_max_len.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_BIG_FIELDLENGTH);

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
}

TEST_F( UT_dict_encode, load_null_into_not_null )
{
    string tableName = "t_load_null_into_not_null";
    string dictName( "dict_UT_dict_encode_load_null_into_not_null" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    string sql( "create table " + tableName + "( f char(4) not null encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    // load null into not null columns
    string csvPath = cwd + "/test_resources/Compression/dict/t_null.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );
}

TEST_F( UT_dict_encode, encode_not_null )
{
    string tableName = "t_encode_not_null";
    string dictName( "dict_UT_dict_encode_encode_not_null" );
    size_t totalRowCount = 10;
    size_t itemLen = 16;
    size_t dictItemCount = 5;
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    string sql( "create table " + tableName + "( f1 int, f2 char(16) not null encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    // load null into not null columns
    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_null.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableName );
    auto tableBlock = initTable->GetTable( { 2 } );
    auto dictCol = tableBlock->GetDictEncodedColumn( 1 );
    auto dictIndices = dictCol->GetIndices()->GetDataBuffer();
    ASSERT_EQ( dictIndices->GetItemCount(), 0 );

    csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( TEST_DB_NAME );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 2 );
    auto dictPtr = colEntry->GetDict();
    ASSERT_EQ( dictPtr->getDictItemCount(), dictItemCount );

    vector< string > expectedDict;
    string tmp( "aaa" );
    tmp.resize( itemLen );
    expectedDict.emplace_back( tmp );
    tmp = "bbb";
    tmp.resize( itemLen );
    expectedDict.emplace_back( tmp );
    tmp = "ccc";
    tmp.resize( itemLen );
    expectedDict.emplace_back( tmp );
    tmp = "aaaa";
    tmp.resize( itemLen );
    expectedDict.emplace_back( tmp );
    tmp = "bbbb";
    tmp.resize( itemLen );
    expectedDict.emplace_back( tmp );

    for ( size_t i = 0; i < dictItemCount; ++i )
    {
        string dictItem( dictPtr->getDictData() + i * itemLen, itemLen );
        ASSERT_EQ( dictItem, expectedDict[ i ] );
    }

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), totalRowCount );

    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableName );
    tableBlock = initTable->GetTable( { 2 } );
    dictCol = tableBlock->GetDictEncodedColumn( 1 );
    dictIndices = dictCol->GetIndices()->GetDataBuffer();
    ASSERT_EQ( dictIndices->GetItemCount(), totalRowCount );

    int expectedDictIndice[] = { 0, 1, 0, 2, 2, 3, 1, 4, 0, 4 };
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        auto dictIdx = dictIndices->GetInt8( i );
        auto expectedDictIdx = expectedDictIndice[ i ];
        ASSERT_EQ( dictIdx, expectedDictIdx );
    }
}

TEST_F( UT_dict_encode, encode_nullable )
{
    string tableName = "t_dict_encode_char_nullable";
    string dictName( "dict_UT_dict_encode_encode_nullable" );
    size_t totalRowCount = 12;
    size_t itemLen = 17;
    size_t dictItemCount = 6;
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    string sql( "create table t_dict_encode_char_nullable( f1 int, f2 char(16) encoding bytedict as " + dictName + ");" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_null.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    auto tableDataDir = aries::Configuartion::GetInstance().GetDataDirectory( TEST_DB_NAME, tableName );
    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( TEST_DB_NAME );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    auto colEntry = tableEntry->GetColumnById( 2 );
    auto dictPtr = colEntry->GetDict();
    ASSERT_EQ( dictPtr->getDictItemCount(), dictItemCount );

    vector< string > expectedDict;
    char* tmpBuff = new char[ itemLen ];
    memset( tmpBuff, 0, itemLen );
    tmpBuff[ 0 ] = 1;

    memcpy( tmpBuff + 1, "aaa", 3 );
    expectedDict.emplace_back( string( tmpBuff, itemLen ) );

    memset( tmpBuff + 1, 0, itemLen - 1 );
    memcpy( tmpBuff + 1, "bbb", 3 );
    expectedDict.emplace_back( string( tmpBuff, itemLen ) );

    memset( tmpBuff + 1, 0, itemLen - 1 );
    memcpy( tmpBuff + 1, "ccc", 3 );
    expectedDict.emplace_back( string( tmpBuff, itemLen ) );

    memset( tmpBuff + 1, 0, itemLen - 1 );
    memcpy( tmpBuff + 1, "aaaa", 4 );
    expectedDict.emplace_back( string( tmpBuff, itemLen ) );

    memset( tmpBuff + 1, 0, itemLen - 1 );
    memcpy( tmpBuff + 1, "bbbb", 4 );
    expectedDict.emplace_back( string( tmpBuff, itemLen ) );

    // NULL value
    memset( tmpBuff, 0, itemLen );
    expectedDict.emplace_back( string( tmpBuff, itemLen ) );

    delete[] tmpBuff;

    for ( size_t i = 0; i < dictItemCount; ++i )
    {
        string dictItem( dictPtr->getDictData() + i * itemLen, itemLen );
        ASSERT_EQ( dictItem, expectedDict[ i ] );
    }

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), totalRowCount );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableName );
    auto tableBlock = initTable->GetTable( { 2 } );
    auto dictCol = tableBlock->GetDictEncodedColumn( 1 );
    auto dictIndices = dictCol->GetIndices()->GetDataBuffer();
    ASSERT_EQ( dictIndices->GetItemCount(), totalRowCount );

    const int8_t* pDictIdx = dictIndices->GetData();
    int expectedDictIndice[] = { 0, 1, 0, 2, 2, 3, 1, 4, 5, 0, 4, 5 };
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        int8_t* tmpIdx = ( int8_t* )pDictIdx + 2 * i;
        if ( 8 == i || 11 == i )
        {
            ASSERT_EQ( tmpIdx[ 0 ], 0 );
        }
        else
        {
            auto expectedDictIdx = expectedDictIndice[ i ];
            ASSERT_EQ( tmpIdx[ 0 ], 1 );
            ASSERT_EQ( tmpIdx[ 1 ], expectedDictIdx );
        }
    }
}

TEST_F( UT_dict_encode, not_nullable_load_and_modify )
{
    string tableName = "t_dict_encode_char";
    string dictName( "dict_UT_dict_encode_not_nullable_load_and_modify" );
    size_t totalRowCount = 10;
    size_t oldDictItemCount = 5;
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    string sql( "create table t_dict_encode_char( f1 int not null, f2 char(4) not null encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    // scan empty table
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    /*
    1,aaa
    2,bbb
    3,aaa
    4,ccc
    5,ccc
    6,aaaa
    7,bbb
    8,bbbb
    9,aaa
    10,bbbb
    */
    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableName );
    vector< int > colIds;
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    auto ariesTable = initTable->GetTable( colIds );

    ASSERT_TRUE( !ariesTable->IsColumnUnMaterilized( 1 ) );

    auto colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    auto columnBuff = ariesTable->GetColumnBuffer( 1 );
    size_t resultRowCount = columnBuff->GetItemCount();
    ASSERT_EQ( resultRowCount, totalRowCount );
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        auto resultItem = columnBuff->GetInt32( i );
        ASSERT_EQ( resultItem, i + 1 );
    }

    auto dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    auto dictBuff = dictColumn->GetDictDataBuffer();
    size_t newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    // check dict
    auto resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );

    // check dict indice
    int expectedDictIndice[] = { 0, 1, 0, 2, 2, 3, 1, 4, 0, 4 };
    auto indice = dictColumn->GetIndices()->GetDataBuffer();
    auto indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, totalRowCount );

    auto indiceData = ( int8_t* )indice->GetData();
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        auto dictIdx = indiceData[ i ];
        auto expectedDictIdx = expectedDictIndice[ i ];
        ASSERT_EQ( dictIdx, expectedDictIdx );
    }

    // test insert into dict encoded column
    string insertValue1( "aaa" );
    string insertValue2New( "dddd" );
    sql = "insert into " + tableName +
           " values (101, \"" + insertValue1 + "\"), " +
                   "(102, \"" + insertValue2New + "\");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    /**
     after insert:
     [0]aaa
     [1]bbb
     [2]aaa
     [3]ccc
     [4]ccc
     [5]aaaa
     [6]bbb
     [7]bbbb
     [8]aaa
     [9]bbbb
     [10]aaa
     [11]dddd
     */

    sql = "select * from " +  tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto expectRowCount = totalRowCount + 2;
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( expectRowCount - 2 ), 101 );
    ASSERT_EQ( columnBuff->GetInt32( expectRowCount - 1 ), 102 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 2 ), insertValue1 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 1 ), insertValue2New );

    auto expectedDictItemCount = oldDictItemCount + 1;
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, insertValue2New );

    string insertValue3New( "eeee" );
    sql = "insert into " + tableName +
           " values (103, \"" + insertValue3New + "\");";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    expectRowCount += 1;

    /**
    after insert2:
    [0]aaa
    [1]bbb
    [2]aaa
    [3]ccc
    [4]ccc
    [5]aaaa
    [6]bbb
    [7]bbbb
    [8]aaa
    [9]bbbb
    [10]aaa
    [11]dddd
    [12]eeee
     */

    sql = "select * from " +  tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( expectRowCount - 1 ), 103 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 1 ), insertValue3New );

    expectedDictItemCount += 1;
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, insertValue2New );
    resultItem = dictBuff->GetString( 6 );
    ASSERT_EQ( resultItem, insertValue3New );

    // test delete: initial table and inserted rows
    sql = "delete from " + tableName + " where f2 in ('bbb', '" + insertValue2New + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    expectRowCount -= 3;

    sql = "select * from " +  tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );

    /**
     * after delete
    [0]aaa
    [1]aaa
    [2]ccc
    [3]ccc
    [4]aaaa
    [5]bbbb
    [6]aaa
    [7]bbbb
    [8]aaa
    [9]eeee
     */
    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "aaa" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "aaa" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "ccc" );
    ASSERT_EQ( columnBuff->GetString( 3 ), "ccc" );
    ASSERT_EQ( columnBuff->GetString( 4 ), "aaaa" );
    ASSERT_EQ( columnBuff->GetString( 5 ), "bbbb" );
    ASSERT_EQ( columnBuff->GetString( 6 ), "aaa" );
    ASSERT_EQ( columnBuff->GetString( 7 ), "bbbb" );
    ASSERT_EQ( columnBuff->GetString( 8 ), "aaa" );
    ASSERT_EQ( columnBuff->GetString( 9 ), "eeee" );

    newDictItemCount = dictBuff->GetItemCount();
    // dict item 'aaa' should not be deleted
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, insertValue2New );
    resultItem = dictBuff->GetString( 6 );
    ASSERT_EQ( resultItem, insertValue3New );

    // inserted data is too long

    // non-strict mode
    STRICT_MODE = false;
    string insertValue4New( "fffff" );

    sql = "insert into " + tableName +
           " values (104, '" + insertValue4New + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    expectRowCount += 1;

    newDictItemCount = dictBuff->GetItemCount();
    expectedDictItemCount += 1;
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    resultItem = dictBuff->GetString( newDictItemCount - 1 );
    ASSERT_EQ( resultItem, "ffff" ); // string is truncated

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );
    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 1 ), "ffff" ); // string is truncated

    // strict mode
    STRICT_MODE = true;
    string insertValue5New( "ggggg" );
    sql = "insert into " + tableName +
           " values (105, '" + insertValue5New + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    /**
     * after insert
    [0]aaa
    [1]aaa
    [2]ccc
    [3]ccc
    [4]aaaa
    [5]bbbb
    [6]aaa
    [7]bbbb
    [8]aaa
    [9]eeee
    [10]ffff
     */

    // test update dict encoded column: update initial table rows and inserted rows
    string updateValue1New( "hhhh" );
    sql = "update " + tableName + " set f2 = '" + updateValue1New + "' where f2 in ( 'aaa', 'ffff' );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    /**
     * after update
    [0]ccc
    [1]ccc
    [2]aaaa
    [3]bbbb
    [4]bbbb
    [5]eeee
    [6]hhhh
    [7]hhhh
    [8]hhhh
    [9]hhhh
    [10]hhhh
     */

    sql = "select * from " +  tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 5 ), updateValue1New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 4 ), updateValue1New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 3 ), updateValue1New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 2 ), updateValue1New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 1 ), updateValue1New );

    newDictItemCount = dictBuff->GetItemCount();
    expectedDictItemCount += 1;
    // dict item 'aaa' should not be deleted and
    // 'hhhh' is added to dict
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, insertValue2New );
    resultItem = dictBuff->GetString( 6 );
    ASSERT_EQ( resultItem, insertValue3New );
    resultItem = dictBuff->GetString( 7 );
    ASSERT_EQ( resultItem, "ffff" ); // truncated string
    resultItem = dictBuff->GetString( 8 );
    ASSERT_EQ( resultItem, updateValue1New );

    // test update dict encoded column: update updated rows
    string updateValue2New( "iiii" );
    sql = "update " + tableName + " set f2 = '" + updateValue2New + "' where f2 = '" + updateValue1New + "';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    /**
     * after update
    [0]ccc
    [1]ccc
    [2]aaaa
    [3]bbbb
    [4]bbbb
    [5]eeee
    [6]iiiii
    [7]iiiii
    [8]iiiii
    [9]iiiii
    [10]iiii
     */

    sql = "select * from " +  tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 5 ), updateValue2New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 4 ), updateValue2New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 3 ), updateValue2New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 2 ), updateValue2New );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 1 ), updateValue2New );

    newDictItemCount = dictBuff->GetItemCount();
    expectedDictItemCount += 1;
    // dict item 'aaa' should not be deleted and
    // 'hhhh' is added to dict
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, insertValue2New );
    resultItem = dictBuff->GetString( 6 );
    ASSERT_EQ( resultItem, insertValue3New );
    resultItem = dictBuff->GetString( 7 );
    ASSERT_EQ( resultItem, "ffff" ); // truncated string
    resultItem = dictBuff->GetString( 8 );
    ASSERT_EQ( resultItem, updateValue1New );
    resultItem = dictBuff->GetString( 9 );
    ASSERT_EQ( resultItem, updateValue2New );

    // test update dict encoded column: string too long

    // strict mode
    STRICT_MODE = true;
    string updateValue3New( "jjjjj" );
    sql = "update " + tableName + " set f2 = '" + updateValue3New + "' where f2 = '" + updateValue2New + "';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    // non strict mode
    STRICT_MODE = false;
    sql = "update " + tableName + " set f2 = '" + updateValue3New + "' where f2 = '" + updateValue2New + "';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    /**
     * after update
    [0]ccc
    [1]ccc
    [2]aaaa
    [3]bbbb
    [4]bbbb
    [5]eeee
    [6]jjjjj
    [7]jjjjj
    [8]jjjjj
    [9]jjjjj
    [10]jjjj
     */

    sql = "select * from " +  tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), expectRowCount );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 5 ), "jjjj" ); // string is truncated
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 4 ), "jjjj" );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 3 ), "jjjj" );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 2 ), "jjjj" );
    ASSERT_EQ( columnBuff->GetString( expectRowCount - 1 ), "jjjj" );

    newDictItemCount = dictBuff->GetItemCount();
    expectedDictItemCount += 1;
    // dict item 'aaa' should not be deleted and
    // 'hhhh' is added to dict
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, insertValue2New );
    resultItem = dictBuff->GetString( 6 );
    ASSERT_EQ( resultItem, insertValue3New );
    resultItem = dictBuff->GetString( 7 );
    ASSERT_EQ( resultItem, "ffff" ); // truncated string
    resultItem = dictBuff->GetString( 8 );
    ASSERT_EQ( resultItem, updateValue1New );
    resultItem = dictBuff->GetString( 9 );
    ASSERT_EQ( resultItem, updateValue2New );
    resultItem = dictBuff->GetString( 10 );
    ASSERT_EQ( resultItem, "jjjj" );
}

TEST_F( UT_dict_encode, nullable_load_and_insert )
{
    string tableName = "t_nullable_load_and_insert";
    string dictName( "dict_UT_dict_encode_nullable_load_and_insert" );
    size_t totalRowCount = 12;
    size_t oldDictItemCount = 6;
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    string sql( "create table " + tableName + "( f1 int not null, f2 char(4) encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    // scan empty table
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_null.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableName );
    vector< int > colIds;
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    auto ariesTable = initTable->GetTable( colIds );

    ASSERT_TRUE( !ariesTable->IsColumnUnMaterilized( 1 ) );

    auto colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    auto columnBuff = ariesTable->GetColumnBuffer( 1 );
    size_t resultRowCount = columnBuff->GetItemCount();
    ASSERT_EQ( resultRowCount, totalRowCount );
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        auto resultItem = columnBuff->GetInt32( i );
        ASSERT_EQ( resultItem, i + 1 );
    }

    auto dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    auto dictBuff = dictColumn->GetDictDataBuffer();
    size_t newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    // check dict
    auto resultItem = dictBuff->GetNullableString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetNullableString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetNullableString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetNullableString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetNullableString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetNullableString( 5 );
    bool b = dictBuff->isStringDataNull( 5 );
    ASSERT_TRUE( b );

    // check dict indice
    int expectedDictIndice[] = { 0, 1, 0, 2, 2, 3, 1, 4, 5, 0, 4, 5 };
    auto indice = dictColumn->GetIndices()->GetDataBuffer();
    auto indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, totalRowCount );

    auto indiceData = ( int8_t* )indice->GetData();
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        int8_t* tmpIdx = indiceData + 2 * i;
        if ( 8 == i || 11 == i )
        {
            ASSERT_EQ( tmpIdx[ 0 ], 0 );
        }
        else
        {
            auto expectedDictIdx = expectedDictIndice[ i ];
            ASSERT_EQ( tmpIdx[ 0 ], 1 );
            ASSERT_EQ( tmpIdx[ 1 ], expectedDictIdx );
        }
    }

    // test insert into dict encoded column
    string newValue( "aaa" );
    sql = "insert into " + tableName + " values (101, \"" + newValue + "\")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    ++resultRowCount;

    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    sql = "insert into " + tableName + " values (102, NULL)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    ++resultRowCount;

    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    newValue = "dddd";
    sql = "insert into " + tableName + " values (103, \"" + newValue + "\")";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    ++resultRowCount;

    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount + 1 );

    resultItem = dictBuff->GetNullableString( newDictItemCount - 1 );
    ASSERT_EQ( resultItem, newValue );

    auto dict = initTable->GetColumnDict( 1 );
    int32_t dictIdx;
    int errorCode;
    string errorMsg;
    b = dict->addDict( newValue.data(), newValue.size(), 0, &dictIdx, errorCode, errorMsg );
    ASSERT_FALSE( b );
    ASSERT_EQ( dictIdx, newDictItemCount - 1 );

    // too long

    // non-strict mode
    STRICT_MODE = false;
    string insertValue4New( "eeeee" );

    sql = "insert into " + tableName +
           " values (104, '" + insertValue4New + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    ++resultRowCount;

    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount + 2 );

    resultItem = dictBuff->GetNullableString( newDictItemCount - 1 );
    ASSERT_EQ( resultItem, "eeee" ); // string is truncated

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), resultRowCount );
    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetNullableString( resultRowCount - 1 ), "eeee" ); // string is truncated

    // strict mode
    STRICT_MODE = true;
    string insertValue5New( "fffff" );
    sql = "insert into " + tableName +
           " values (105, '" + insertValue5New + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_DATA_TOO_LONG );

    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount + 2 );
}

TEST_F( UT_dict_encode, share_dict1 )
{
    string dictName( "dict_UT_dict_encode_share_dict1" );
    string tableNameShared = "t_dict_encode_share_dict_shared";
    string tableNameShare = "t_dict_encode_share_dict_share";
    InitTable( TEST_DB_NAME, tableNameShared );
    InitTable( TEST_DB_NAME, tableNameShare );

    string sql = "create table " + tableNameShared + " ( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    auto dict = AriesDictManager::GetInstance().GetDict( dictName );
    ASSERT_EQ(  dict->GetRefCount(), 1 );

    // column type does not match
    sql = "create table " + tableNameShare +
          "( f1 int not null, f2 char(26) null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_DICT );
    ASSERT_EQ( dict->GetRefCount(), 1 );

    sql = "create table " + tableNameShare +
          "( f1 int not null, f2 char(26) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_DICT );
    ASSERT_EQ( dict->GetRefCount(), 1 );

    // dict index data type does not match
    sql = "create table " + tableNameShare +
          "( f1 int not null, f2 char(16) not null encoding shortdict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_DICT );
    ASSERT_EQ( dict->GetRefCount(), 1 );

    sql = "create table " + tableNameShare +
          "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE( result->IsSuccess() );

    ASSERT_EQ( dict->GetRefCount(), 2 );
}

TEST_F( UT_dict_encode, share_dict_load_csv )
{
    size_t totalRowCount = 10;
    size_t oldDictItemCount = 5;
    string dictName( "dict_UT_dict_encode_share_dict_load_csv" );
    string tableNameShared = "t_dict_encode_share_dict_shared";
    string tableNameShare1 = "t_dict_encode_share_dict_share1";
    string tableNameShare2 = "t_dict_encode_share_dict_share2";
    InitTable( TEST_DB_NAME, tableNameShared );
    InitTable( TEST_DB_NAME, tableNameShare1 );
    InitTable( TEST_DB_NAME, tableNameShare2 );

    string cwd = get_current_work_directory();

    string sql( "create table " + tableNameShared + "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableNameShare1 +
          "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableNameShare2 +
          "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    // load csv into sharing table1
    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_share1.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableNameShare1 + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    ////////////////////////////////////////////////////////////
    // check dict of sharing table1
    ////////////////////////////////////////////////////////////
    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableNameShare1 );
    vector< int > colIds;
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    auto ariesTable = initTable->GetTable( colIds );

    ASSERT_TRUE( !ariesTable->IsColumnUnMaterilized( 1 ) );

    auto colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    auto dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    auto dictBuff = dictColumn->GetDictDataBuffer();
    size_t newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    // check dict
    auto resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );

    // check dict indice
    int expectedDictIndice[] = { 0, 1, 0, 2, 2, 3, 1, 4, 0, 4 };
    auto indice = dictColumn->GetIndices()->GetDataBuffer();
    auto indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, totalRowCount );

    auto indiceData = ( int8_t* )indice->GetData();
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        auto dictIdx = indiceData[ i ];
        auto expectedDictIdx = expectedDictIndice[ i ];
        ASSERT_EQ( dictIdx, expectedDictIdx );
    }

    ////////////////////////////////////////////////////////////
    // check dict of sharing table2
    ////////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableNameShare2 );
    colIds.clear();
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    ariesTable = initTable->GetTable( colIds );

    ASSERT_TRUE( !ariesTable->IsColumnUnMaterilized( 1 ) );

    colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );

    // no dict indice for sharing table2
    indice = dictColumn->GetIndices()->GetDataBuffer();
    indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, 0 );

    ////////////////////////////////////////////////////////////
    // check dict of SHARED table
    ////////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableNameShared );
    colIds.clear();
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    ariesTable = initTable->GetTable( colIds );

    colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, oldDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );

    // no dict indice for shared dict table
    indice = dictColumn->GetIndices()->GetDataBuffer();
    indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, 0 );

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // load csv into sharing table2
    // 1 new dict item will be added
    csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_char_share2.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableNameShare2 + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    ////////////////////////////////////////////////////////////
    // check dict of sharing table1
    ////////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableNameShare1 );
    colIds.clear();
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    ariesTable = initTable->GetTable( colIds );

    ASSERT_TRUE( !ariesTable->IsColumnUnMaterilized( 1 ) );

    colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    auto expectedDictItemCount = oldDictItemCount + 1;

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, "ddd" );

    // check dict indice, not changed
    indice = dictColumn->GetIndices()->GetDataBuffer();
    indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, totalRowCount );

    indiceData = ( int8_t* )indice->GetData();
    for ( size_t i = 0; i < totalRowCount; ++i )
    {
        auto dictIdx = indiceData[ i ];
        auto expectedDictIdx = expectedDictIndice[ i ];
        ASSERT_EQ( dictIdx, expectedDictIdx );
    }

    ////////////////////////////////////////////////////////////
    // check dict of sharing table2
    ////////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableNameShare2 );
    colIds.clear();
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    ariesTable = initTable->GetTable( colIds );

    ASSERT_TRUE( !ariesTable->IsColumnUnMaterilized( 1 ) );

    colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, "ddd" );

    // no dict indice for sharing table2
    indice = dictColumn->GetIndices()->GetDataBuffer();
    indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, 3 );
    indiceData = ( int8_t* )indice->GetData();
    {
        auto dictIdx = indiceData[ 0 ];
        ASSERT_EQ( dictIdx, 0 );
        dictIdx = indiceData[ 1 ];
        ASSERT_EQ( dictIdx, 1 );
        dictIdx = indiceData[ 2 ];
        ASSERT_EQ( dictIdx, 5 );
    }

    ////////////////////////////////////////////////////////////
    // check dict of SHARED table
    ////////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME, tableNameShared );
    colIds.clear();
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    ariesTable = initTable->GetTable( colIds );

    colEncodeType = ariesTable->GetColumnEncodeType( 2 );
    ASSERT_EQ( colEncodeType, EncodeType::DICT );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    newDictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( newDictItemCount, expectedDictItemCount );

    // check dict
    resultItem = dictBuff->GetString( 0 );
    ASSERT_EQ( resultItem, "aaa" );
    resultItem = dictBuff->GetString( 1 );
    ASSERT_EQ( resultItem, "bbb" );
    resultItem = dictBuff->GetString( 2 );
    ASSERT_EQ( resultItem, "ccc" );
    resultItem = dictBuff->GetString( 3 );
    ASSERT_EQ( resultItem, "aaaa" );
    resultItem = dictBuff->GetString( 4 );
    ASSERT_EQ( resultItem, "bbbb" );
    resultItem = dictBuff->GetString( 5 );
    ASSERT_EQ( resultItem, "ddd" );

    // no dict indice for shared dict table
    indice = dictColumn->GetIndices()->GetDataBuffer();
    indiceItemCount = indice->GetItemCount();
    ASSERT_EQ( indiceItemCount, 0 );
}

TEST_F( UT_dict_encode, dict_xlog )
{
    string tableName = "t_dict_encode_xlog";
    string dictName( "dict_UT_dict_encode_dict_xlog" );
    InitTable( TEST_DB_NAME_XLOG, tableName );

    string cwd = get_current_work_directory();

    string sql = "create table " + tableName + "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME_XLOG);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    string dictItemA( "aaa" );
    string dictItemB( "bbb" );
    string dictItemC( "ccc" );
    string dictItemD( "ddd" );
    string dictItemE( "eee" );

    /********************************************************/
    /* test1: insert rows into table                        */
    /********************************************************/

    //////////////////////////////////////////////////////////
    // tx1: insert rows into table
    sql = "insert into " + tableName + " values ( 1, '" + dictItemA + "' ), ( 2, '" + dictItemB + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx2: insert rows into table and rollback
    sql = "start transaction";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "insert into " + tableName + " values ( 3, '" + dictItemC + "' ), ( 4, '" + dictItemD + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "rollback";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx3: insert rows into table
    sql = "insert into " + tableName + " values ( 4, '" + dictItemD + "' ), ( 5, '" + dictItemE + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // verify inserted rows
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemB );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 3 ), dictItemE );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    vector< int > colIds;
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    auto ariesTable = initTable->GetTable( colIds );

    auto dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    auto dictBuff = dictColumn->GetDictDataBuffer();
    size_t dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 5 );
    auto dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName );

    /********************************************************/
    /* test2: reover insert xlog and verify data again      */
    /********************************************************/
    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    auto dict = AriesDictManager::GetInstance().GetDict( dictName );
    ASSERT_EQ( dict->getDictBuffer(), nullptr );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    //////////////////////////////////////////////////////////
    // verify inserted rows
    //////////////////////////////////////////////////////////

    /*
    dict:
    [0]aaa
    [1]bbb
    [2]ccc
    [3]ddd
    [4]eee

    column f1:
    4
    5
    1
    2

    column f2:
    [0]ddd
    [1]eee
    [2]aaa
    [3]bbb
    */
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter insert xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 2 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter insert xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemA );
    ASSERT_EQ( columnBuff->GetString( 3 ), dictItemB );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "after insert xlog recovery, dict:\n");
    // dictBuff->Dump();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 5 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );

    /********************************************************/
    /* test3: update rows                                   */
    /********************************************************/
    string newValueA = "AAAA";
    string newValueD = "DDDD";
    sql = "update " + tableName + " set f2 = '" + newValueA + "' where f2 = '" + dictItemA + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update1, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update1, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemB );
    ASSERT_EQ( columnBuff->GetString( 3 ), newValueA );

    sql = "update " + tableName + " set f2 = '" + newValueD + "' where f2 = '" + dictItemD + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update2, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 4 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update2, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemB );
    ASSERT_EQ( columnBuff->GetString( 2 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 3 ), newValueD );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 7 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueD );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName );

    /**
    dict:
    [0]aaa
    [1]bbb
    [2]ccc
    [3]ddd
    [4]eee
    [5]AAAA
    [6]DDDD

    f1:
    5
    2
    1
    4

    f2:
    [0]eee
    [1]bbb
    [2]AAAA
    [3]DDDD
     */

    /********************************************************/
    /* test4: reover update xlog and verify data again      */
    /********************************************************/
    special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    dict = AriesDictManager::GetInstance().GetDict( dictName );
    ASSERT_EQ( dict->getDictBuffer(), nullptr );

    recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 2 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 2 ), newValueD );
    ASSERT_EQ( columnBuff->GetString( 3 ), dictItemB );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "after update xlog recovery, dict:\n");
    // dictBuff->Dump();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 7 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueD );

    /********************************************************/
    /* test5: delete rows                                   */
    /********************************************************/
    sql = "insert into " + tableName + " values (111, '" + dictItemB + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "delete from " + tableName + " where f2 ='" + dictItemB + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete1, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 4 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete1, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 2 ), newValueD );

    sql = "delete from " + tableName + " where f2 ='" + newValueD + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete1, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete1, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 7 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueD );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName );

    /********************************************************/
    /* test6: reover delete xlog and verify data again      */
    /********************************************************/
    special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    dict = AriesDictManager::GetInstance().GetDict( dictName );
    ASSERT_EQ( dict->getDictBuffer(), nullptr );

    recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 7 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueD );

}

TEST_F( UT_dict_encode, nullable_dict_xlog )
{
    string tableName = "t_nullable_dict_encode_xlog";
    string dictName( "dict_UT_dict_encode_nullable_dict_xlog" );
    InitTable( TEST_DB_NAME_XLOG, tableName );

    string cwd = get_current_work_directory();

    string sql = "create table " + tableName + "( f1 int not null, f2 char(16) encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME_XLOG);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    string dictItemA( "aaa" );
    string dictItemB( "bbb" );
    string dictItemC( "ccc" );
    string dictItemD( "ddd" );
    string dictItemE( "eee" );
    string dictItemNull( "NULL" );

    /********************************************************/
    /* test1: insert rows into table                        */
    /********************************************************/

    //////////////////////////////////////////////////////////
    // tx1: insert rows into table
    sql = "insert into " + tableName + " values ( 1, '" + dictItemA + "' ), ( 2, '" + dictItemB + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx2: insert rows into table and rollback
    sql = "start transaction";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "insert into " + tableName + " values ( 3, '" + dictItemC + "' ), ( 4, '" + dictItemD + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "rollback";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx3: insert rows into table
    sql = "insert into " + tableName + " values ( 4, '" + dictItemD + "' ), ( 5, '" + dictItemE + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx4: insert null into table
    sql = "insert into " + tableName + " values ( 6, " + dictItemNull + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );


    //////////////////////////////////////////////////////////
    // verify inserted rows
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 5 );

    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    vector< int > colIds;
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    auto ariesTable = initTable->GetTable( colIds );

    auto dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    auto dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "dict after insert:\n");
    // dictBuff->Dump();
    auto dictIndices = dictColumn->GetIndices();
    // printf( "dict indices after insert:\n");
    // dictIndices->GetDataBuffer()->Dump();

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 6 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "after insert, dict encoded column:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemB );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 3 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 4 ), dictItemNull );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    colIds.clear();
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    size_t dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 6 );
    auto dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, dictItemNull );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName );

    /********************************************************/
    /* test2: reover insert xlog and verify data again      */
    /********************************************************/
    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    //////////////////////////////////////////////////////////
    // verify inserted rows
    //////////////////////////////////////////////////////////

    /*
    dict:
    [0]aaa
    [1]bbb
    [2]ccc
    [3]ddd
    [4]eee
    [5]NULL

    column f1:
    6
    4
    5
    1
    2

    column f2:
    [0]NULL
    [1]ddd
    [2]eee
    [3]aaa
    [4]bbb
    */
    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 5 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter insert xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 2 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter insert xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 3 ), dictItemA );
    ASSERT_EQ( columnBuff->GetString( 4 ), dictItemB );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "after insert xlog recovery, dict:\n");
    // dictBuff->Dump();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 6 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, dictItemNull );

    /********************************************************/
    /* test3: update rows                                   */
    /********************************************************/
    string newValueA = "AAAA";
    string newValueD = "DDDD";
    sql = "update " + tableName + " set f2 = '" + newValueA + "' where f2 = '" + dictItemA + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "after update1, dict:\n");
    // dictBuff->Dump();

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 5 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update1, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update1, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 3 ), dictItemB );
    ASSERT_EQ( columnBuff->GetString( 4 ), newValueA );

    sql = "update " + tableName + " set f2 = '" + newValueD + "' where f2 = '" + dictItemD + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 5 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update2, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 2 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 4 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update2, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemB );
    ASSERT_EQ( columnBuff->GetString( 3 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 4 ), newValueD );

    // sql = "update " + tableName + " set f2 = null where f2 = '" + dictItemE + "'";
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    // ASSERT_TRUE( result->IsSuccess() );

    // sql = "select * from " + tableName;
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    // ASSERT_TRUE( result->IsSuccess() );
    // resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    // ASSERT_EQ( resTable->GetRowCount(), 5 );

    // columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update3, f1:\n");
    // columnBuff->Dump();
    // // ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    // // ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );
    // // ASSERT_EQ( columnBuff->GetInt32( 2 ), 2 );
    // // ASSERT_EQ( columnBuff->GetInt32( 3 ), 1 );
    // // ASSERT_EQ( columnBuff->GetInt32( 4 ), 4 );

    // columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update3, f2:\n");
    // columnBuff->Dump();
    // ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    // ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );
    // ASSERT_EQ( columnBuff->GetString( 2 ), dictItemB );
    // ASSERT_EQ( columnBuff->GetString( 3 ), newValueA );
    // ASSERT_EQ( columnBuff->GetString( 4 ), newValueD );

    ///////////////////////////////////////////////////////
    // verify dict
    ///////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 8 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, dictItemNull );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 7 );
    ASSERT_EQ( dictItem, newValueD );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName );

    /**
    dict:
    [0]aaa
    [1]bbb
    [2]ccc
    [3]ddd
    [4]eee
    [5]NULL
    [6]AAAA
    [7]DDDD

    f1:
    6
    1
    5
    4
    2

    f2:
    [0]NULL
    [1]AAAA
    [2]eee
    [3]DDDD
    [4]bbb
     */

    /********************************************************/
    /* test4: recover update xlog and verify data again      */
    /********************************************************/
    special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 5 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter update xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 4 ), 2 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter update xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    ASSERT_EQ( columnBuff->GetString( 1 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 3 ), newValueD );
    ASSERT_EQ( columnBuff->GetString( 4 ), dictItemB );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "after update xlog recovery, dict:\n");
    // dictBuff->Dump();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 8 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, dictItemNull );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 7 );
    ASSERT_EQ( dictItem, newValueD );

    /********************************************************/
    /* test5: delete rows                                   */
    /********************************************************/
    sql = "insert into " + tableName + " values (111, '" + dictItemB + "')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "delete from " + tableName + " where f2 ='" + dictItemB + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete1, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 3 ), 4 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete1, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    ASSERT_EQ( columnBuff->GetString( 1 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 3 ), newValueD );

    sql = "delete from " + tableName + " where f2 ='" + newValueD + "'";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete2, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 6 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 2 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete2, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemNull );
    ASSERT_EQ( columnBuff->GetString( 1 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 2 ), dictItemE );

    sql = "delete from " + tableName + " where f2 is null";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete3, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete3, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), newValueA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 8 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, dictItemNull );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 7 );
    ASSERT_EQ( dictItem, newValueD );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName );

    /********************************************************/
    /* test6: recover delete xlog and verify data again      */
    /********************************************************/
    special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter delete xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 5 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter delete xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemE );
    ASSERT_EQ( columnBuff->GetString( 1 ), newValueA );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 8 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
    dictItem = dictBuff->GetString( 5 );
    ASSERT_EQ( dictItem, dictItemNull );
    dictItem = dictBuff->GetString( 6 );
    ASSERT_EQ( dictItem, newValueA );
    dictItem = dictBuff->GetString( 7 );
    ASSERT_EQ( dictItem, newValueD );
}

TEST_F( UT_dict_encode, share_dict_xlog )
{
    string tableName1 = "t_dict_encode_share_dict_xlog1";
    string tableName2 = "t_dict_encode_share_dict_xlog2";
    string dictName( "dict_UT_dict_encode_share_dict_xlog" );
    InitTable( TEST_DB_NAME_XLOG, tableName1 );
    InitTable( TEST_DB_NAME_XLOG, tableName2 );

    string cwd = get_current_work_directory();

    string sql = "create table " + tableName1 + "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME_XLOG);
    ASSERT_TRUE(result->IsSuccess());

    sql = "create table " + tableName2 + "( f1 int not null, f2 char(16) not null encoding bytedict as " + dictName + " );";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME_XLOG);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select * from " + tableName1;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    string dictItemA( "aaa" );
    string dictItemB( "bbb" );
    string dictItemC( "ccc" );
    string dictItemD( "ddd" );
    string dictItemE( "eee" );

    /********************************************************/
    /* test1: insert rows into table                        */
    /********************************************************/

    //////////////////////////////////////////////////////////
    // tx1: insert rows into table1
    sql = "insert into " + tableName1 + " values ( 1, '" + dictItemA + "' ), ( 2, '" + dictItemB + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx2: insert rows into table and rollback
    sql = "start transaction";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "insert into " + tableName1 + " values ( 3, '" + dictItemC + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "insert into " + tableName2 + " values ( 4, '" + dictItemD + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "rollback";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // tx3: insert rows into table2
    sql = "insert into " + tableName2 + " values ( 4, '" + dictItemD + "' ), ( 5, '" + dictItemE + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );

    //////////////////////////////////////////////////////////
    // verify inserted rows
    sql = "select * from " + tableName1;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemB );

    sql = "select * from " + tableName2;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    auto initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName1 );
    vector< int > colIds;
    colIds.emplace_back( 1 );
    colIds.emplace_back( 2 );
    auto ariesTable = initTable->GetTable( colIds );

    auto dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    auto dictBuff = dictColumn->GetDictDataBuffer();
    size_t dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 5 );
    auto dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName1 );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName1 );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName1 );

    AriesMvccTableManager::GetInstance().deleteCache( TEST_DB_NAME_XLOG, tableName2 );
    AriesInitialTableManager::GetInstance().removeTable( TEST_DB_NAME_XLOG, tableName2 );
    AriesMvccTableManager::GetInstance().removeMvccTable( TEST_DB_NAME_XLOG, tableName2 );

    /********************************************************/
    /* test2: reover insert xlog and verify data again      */
    /********************************************************/
    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    auto dict = AriesDictManager::GetInstance().GetDict( dictName );
    ASSERT_EQ( dict->getDictBuffer(), nullptr );

    auto recoveryer = std::make_shared< AriesXLogRecoveryer >();
    recoveryer->SetReader( AriesXLogManager::GetInstance().GetReader() );
    auto recovery_result = recoveryer->Recovery();
    ASSERT_TRUE( recovery_result );

    //////////////////////////////////////////////////////////
    // verify inserted rows
    //////////////////////////////////////////////////////////

    sql = "select * from " + tableName1;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    // printf( "\nafter insert xlog recovery, f1:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    // printf( "\nafter insert xlog recovery, f2:\n");
    // columnBuff->Dump();
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemA );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemB );

    sql = "select * from " + tableName2;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME_XLOG );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 2 );

    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 4 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 5 );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), dictItemD );
    ASSERT_EQ( columnBuff->GetString( 1 ), dictItemE );

    //////////////////////////////////////////////////////////
    // verify dict
    //////////////////////////////////////////////////////////
    initTable = AriesInitialTableManager::GetInstance().getTable( TEST_DB_NAME_XLOG, tableName2 );
    ariesTable = initTable->GetTable( colIds );

    dictColumn = ariesTable->GetDictEncodedColumn( 2 );
    dictBuff = dictColumn->GetDictDataBuffer();
    // printf( "after insert xlog recovery, dict:\n");
    // dictBuff->Dump();
    dictItemCount = dictBuff->GetItemCount();
    ASSERT_EQ( dictItemCount, 5 );
    dictItem = dictBuff->GetString( 0 );
    ASSERT_EQ( dictItem, dictItemA );
    dictItem = dictBuff->GetString( 1 );
    ASSERT_EQ( dictItem, dictItemB );
    dictItem = dictBuff->GetString( 2 );
    ASSERT_EQ( dictItem, dictItemC );
    dictItem = dictBuff->GetString( 3 );
    ASSERT_EQ( dictItem, dictItemD );
    dictItem = dictBuff->GetString( 4 );
    ASSERT_EQ( dictItem, dictItemE );
}

TEST_F( UT_dict_encode, dict_not_nullable_to_nullable )
{
    string dictName( "dict_UT_dict_encode_dict_not_nullable_to_nullable" );
    string tableRegion = "t_dict_encode_region";
    InitTable( TEST_DB_NAME, tableRegion );
    string tableNation = "t_dict_encode_nation";
    InitTable( TEST_DB_NAME, tableNation );

    string cwd = get_current_work_directory();

    string sql( "create table t_dict_encode_region( "
                "r_regionkey integer not null primary key,"
                "r_name char(25) not null encoding bytedict as " + dictName + ","
                "r_comment    varchar(152) );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = R"(create table t_dict_encode_nation(
                 n_nationkey  integer not null primary key,
                 n_name       char(25) not null,
                 n_regionkey  integer not null,
                 n_comment    varchar(152) );)";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_region.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableRegion + " fields terminated by '|';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_nation.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableNation + " fields terminated by '|';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select n_nationkey, n_name, r_regionkey, r_name from " +
          tableNation + " left outer join " + tableRegion +
          " on n_regionkey = r_regionkey order by n_nationkey";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 3 );
    auto columnBuffer = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuffer->GetInt32(0), 0 );
    ASSERT_EQ( columnBuffer->GetInt32(1), 1 );
    ASSERT_EQ( columnBuffer->GetInt32(2), 2 );

    columnBuffer = resTable->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuffer->GetNullableInt32(0).flag, 1);
    ASSERT_EQ( columnBuffer->GetNullableInt32(0).value, 0);
    ASSERT_EQ( columnBuffer->GetNullableInt32(1).flag, 1 );
    ASSERT_EQ( columnBuffer->GetNullableInt32(1).value, 1 );
    ASSERT_EQ( columnBuffer->GetNullableInt32(2).flag, 0 ); // null

    columnBuffer = resTable->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuffer->GetString(0), "AFRICA");
    ASSERT_EQ( columnBuffer->GetString(1), "ASIA");
    ASSERT_TRUE( columnBuffer->isStringDataNull(2) );
    ASSERT_EQ( columnBuffer->GetString(2), "NULL");
}

TEST_F( UT_dict_encode, dict_sort )
{
    string tableName = "t_dict_encode_sort";
    string dictName( "dict_UT_dict_encode_dict_dict_sort" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();

    string sql( "create table t_dict_encode_sort("
                "f1 int not null,"
                "f2 char(16) not null encoding bytedict as " + dictName + " ); " );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_sort.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + " order by f1, f2";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 4 );
    auto columnBuffer = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuffer->GetInt32(0), 1 );
    ASSERT_EQ( columnBuffer->GetInt32(1), 1 );
    ASSERT_EQ( columnBuffer->GetInt32(2), 2 );
    ASSERT_EQ( columnBuffer->GetInt32(3), 2 );

    columnBuffer = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuffer->GetString(0), "AAA" );
    ASSERT_EQ( columnBuffer->GetString(1), "aaa" );
    ASSERT_EQ( columnBuffer->GetString(2), "AAA" );
    ASSERT_EQ( columnBuffer->GetString(3), "aaa" );

}

TEST_F( UT_dict_encode, dict_sort_nullable )
{
    string tableName = "t_dict_encode_nullalbe_sort";
    string dictName( "dict_UT_dict_encode_dict_sort_nullable" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();

    string sql( "create table t_dict_encode_nullalbe_sort("
                "f1 int not null,"
                "f2 char(16) encoding bytedict as " + dictName + " );" );
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/Compression/dict/t_dict_encode_sort_null.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName + " order by f1, f2";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 7 );
    auto columnBuffer = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuffer->GetInt32(0), 1 );
    ASSERT_EQ( columnBuffer->GetInt32(1), 1 );
    ASSERT_EQ( columnBuffer->GetInt32(2), 1 );
    ASSERT_EQ( columnBuffer->GetInt32(3), 2 );
    ASSERT_EQ( columnBuffer->GetInt32(4), 2 );
    ASSERT_EQ( columnBuffer->GetInt32(5), 2 );
    ASSERT_EQ( columnBuffer->GetInt32(6), 3 );

    columnBuffer = resTable->GetColumnBuffer( 2 );
    ASSERT_TRUE( columnBuffer->isStringDataNull(0) );
    ASSERT_EQ( columnBuffer->GetString(1), "AAA" );
    ASSERT_EQ( columnBuffer->GetString(2), "aaa" );
    ASSERT_TRUE( columnBuffer->isStringDataNull(3) );
    ASSERT_EQ( columnBuffer->GetString(4), "AAA" );
    ASSERT_EQ( columnBuffer->GetString(5), "aaa" );
    ASSERT_TRUE( columnBuffer->isStringDataNull(6) );
}

TEST_F( UT_dict_encode, too_many_dict_items )
{
    string tableName = "t_too_many_dict_items";
    string dictName( "dict_UT_dict_encode_too_many_dict_items" );
    string dictName2( "dict_UT_dict_encode_too_many_dict_items_2" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();

    string sql = "create table " + tableName + "(f1 char(16) encoding bytedict as " + dictName +  " )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/Compression/dict/too_many_dict_items.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_MANY_DICT_ITEMS );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    // load again
    InitTable( TEST_DB_NAME, tableName );
    sql = "create table " + tableName + "(f1 char(16) encoding bytedict as " + dictName2 +  " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 127 );

    sql = "insert into " + tableName + " values ( 'aaa' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_MANY_DICT_ITEMS );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 127 );
}

TEST_F( UT_dict_encode, too_many_dict_items2 )
{
    string tableName = "t_too_many_dict_items2";
    string dictName( "dict_UT_dict_encode_too_many_dict_items2" );
    string dictName2( "dict_UT_dict_encode_too_many_dict_items2_2" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    // smallint
    string sql = "create table " + tableName + "(f1 char(16) encoding shortdict as " + dictName + " )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string csvPath = cwd + "/test_resources/Compression/dict/too_many_dict_items_smallint.csv";
    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',';";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_MANY_DICT_ITEMS );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    InitTable( TEST_DB_NAME, tableName );

    sql = "create table " + tableName + "(f1 char(16) encoding shortdict as " + dictName + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName + " fields terminated by ',' ignore 1 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), INT16_MAX );

    sql = "insert into " + tableName + " values ( 'aaa' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_MANY_DICT_ITEMS );

    sql = "select * from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    ASSERT_TRUE( result->IsSuccess() );
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), INT16_MAX );

    //tinyint
    string tableName3 = "t_too_many_dict_items3";
    InitTable(TEST_DB_NAME, tableName3);
    sql = "create table " + tableName3 + " (f2 char(10) encoding bytedict as " + dictName2 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName3 + " fields terminated by ',' ignore 32640 lines;"; // 32768-32640=128
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_TOO_MANY_DICT_ITEMS );

    InitTable(TEST_DB_NAME, tableName3);
    sql = "create table " + tableName3 + " (f2 char(10) encoding bytedict as " + dictName2 + " )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "load data infile '" + csvPath + "' into table " + tableName3 + " fields terminated by ',' ignore 32641 lines;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    sql = "select * from " + tableName3;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0])->GetContent();
    ASSERT_EQ(resTable->GetRowCount(), INT8_MAX);
}

TEST_F( UT_dict_encode, dict_index )
{
    string tableName = "t_dict_index";
    string dictName( "dict_UT_dict_encode_dict_index" );
    InitTable( TEST_DB_NAME, tableName );

    string cwd = get_current_work_directory();
    // smallint
    string sql = "create table " + tableName + "( f1 int, f2 char(16), f3 char(16) encoding shortdict as " + dictName + " )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    string itemA( "aaa" );
    string itemB( "bbb" );
    string itemC( "ccc" );
    sql = "insert into " + tableName + " values" +
          "( 1, 'a', '" + itemA + "' ), " +
          "( 1, 'a', '" + itemA + "' ), " +
          "( 2, 'b', '" + itemB + "' ), " +
          "( 2, 'b', '" + itemB + "' ), " +
          "( 3, 'c', '" + itemC + "' ), " +
          "( 3, 'c', '" + itemC + "' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "select dict_index( 1 ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select dict_index( f1 ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select dict_index( f2 ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select dict_index( f3 + 1 ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select dict_index( f3 ) + 1 from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select dict_index( f3 ) from " + tableName + " where dict_index( f3 ) > 0";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = "select dict_index( f3 ) from " + tableName;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 6 );
    ASSERT_EQ( resTable->GetColumnCount(), 1 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 0 ).flag, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 0 ).value, 0 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 1 ).flag, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 1 ).value, 0 );

    ASSERT_EQ( columnBuff->GetNullableInt16( 2 ).flag, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 2 ).value, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 3 ).flag, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 3 ).value, 1 );

    ASSERT_EQ( columnBuff->GetNullableInt16( 4 ).flag, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 4 ).value, 2 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 5 ).flag, 1 );
    ASSERT_EQ( columnBuff->GetNullableInt16( 5 ).value, 2 );
}

TEST_F( UT_dict_encode, share_dict_insert_dict_index )
{
    string cwd = get_current_work_directory();

    string tableName1( "share_dict_insert_dict_index1" );
    string tableName2( "share_dict_insert_dict_index2" );

    string dictName1( "dict_UT_dict_encode_share_dict_insert_dict_index1" );
    string dictName2( "dict_UT_dict_encode_share_dict_insert_dict_index2" );

    InitTable( testDbName1, tableName1 );
    InitTable( testDbName1, tableName2 );

    // not share dict
    string table1Schema( "create table " + tableName1 + " ( f1 char( 3 ) encoding bytedict as " + dictName1 + " );" );
    string table2Schema( "create table " + tableName2 + " ( f1 char( 3 ) encoding bytedict as " + dictName2 + " );" );

    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(table1Schema, testDbName1);
    ASSERT_TRUE(result->IsSuccess());
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(table2Schema, testDbName1);
    ASSERT_TRUE(result->IsSuccess());

    string sql = "insert into " + tableName1 + " select dict_index( f1 ) from " + tableName2;
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, testDbName1);
    ASSERT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}

// need to hard code partition count to > 1
TEST_F( UT_dict_encode, partitioned_left_join )
{
    string cwd = get_current_work_directory();

    string tableName1( "partitioned_left_join1" );
    string tableName2( "partitioned_left_join2" );

    string dictName1( "dict_UT_partitioned_left_join" );

    InitTable( testDbName1, tableName1 );
    InitTable( testDbName1, tableName2 );

    // not share dict
    string table1Schema( "create table " + tableName1 + " ( f1 char( 3 ) encoding bytedict as " + dictName1 + ", unique key ( f1 ) );" );
    string table2Schema( "create table " + tableName2 + " ( f1 char( 3 ) encoding bytedict as " + dictName1 + " );" );

    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(table1Schema, testDbName1);
    ASSERT_TRUE(result->IsSuccess());
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(table2Schema, testDbName1);
    ASSERT_TRUE(result->IsSuccess());

    string sql = "insert into " + tableName1 + " values ( 'aaa' ), ('bbb')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, testDbName1);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "insert into " + tableName2 + " values ( 'ccc' ), ('bbb')";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, testDbName1);
    ASSERT_EQ( result->GetErrorCode(), 0 );

    sql = "select * from " + tableName1 + " left join " + tableName2 + " ON " + tableName1 + ".f1 = " + tableName2 + ".f1 order by " + tableName1 + ".f1";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, testDbName1);
    ASSERT_EQ( result->GetErrorCode(), 0 );
    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    auto resultRowCount = resTable->GetRowCount();
    ASSERT_EQ( resultRowCount, 2 );

    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetString( 0 ), "aaa" );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetString( 1 ), "bbb" );

    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetString( 1 ), "bbb" );
}
