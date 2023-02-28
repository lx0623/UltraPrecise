#include <gtest/gtest.h>
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"
#include "schema/SchemaManager.h"
#include "Configuration.h"
#include "AriesEngineWrapper/AriesMemTable.h"

#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "AriesEngine/transaction/AriesXLogManager.h"

#include "../../TestUtils.h"
#include "utils/string_util.h"

using namespace std;
using namespace aries_test;
using namespace aries_engine;

const static string TEST_DB_NAME = "schema_test_db1";
class UT_TestSchema : public testing::Test
{
protected:
    std::string database_name = "TestSchema";

protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
        cout << "tear down TestSchema\n";
        string sql = "drop database if exists " + TEST_DB_NAME;
        ExecuteSQL( sql, "" );
    }
    virtual void SetUp() override
    {
        aries_utils::to_lower( database_name );
        ExecuteSQL( "drop database if exists " + database_name + ";", "" );
        ExecuteSQL( "create database " + database_name + ";", "" );
    }

    virtual void TearDown() override
    {
        ExecuteSQL( "drop database " + database_name + ";", "" );
    }
};

TEST_F( UT_TestSchema, t1 )
{
    string sql = "drop database if exists " + TEST_DB_NAME;
    ExecuteSQL( sql, "" );

    // verify database does not exists
    sql = "select schema_name from information_schema.schemata where schema_name = '" + TEST_DB_NAME + "'";
    auto table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 0 );

    sql = "create database " + TEST_DB_NAME;
    ExecuteSQL( sql, "" );

    ExecuteSQLFromFile( "./test_resources/schema/tpch/create_table.sql", TEST_DB_NAME );

    // verify db is created
    sql = "select schema_name from information_schema.schemata where schema_name = '" + TEST_DB_NAME + "'";
    table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 1 );
    ASSERT_EQ( column->GetString( 0 ), TEST_DB_NAME );

    // verify tables are created
    sql = "select table_name from information_schema.tables where table_schema = '" + TEST_DB_NAME + "'"
          " order by table_name";
    table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 8 );
    ASSERT_EQ( column->GetString( 0 ), "customer" );
    ASSERT_EQ( column->GetString( 1 ), "lineitem" );
    ASSERT_EQ( column->GetString( 2 ), "nation" );
    ASSERT_EQ( column->GetString( 3 ), "orders" );
    ASSERT_EQ( column->GetString( 4 ), "part" );
    ASSERT_EQ( column->GetString( 5 ), "partsupp" );
    ASSERT_EQ( column->GetString( 6 ), "region" );
    ASSERT_EQ( column->GetString( 7 ), "supplier" );

    // test drop table
    sql = "drop table nation";
    ExecuteSQL( sql, TEST_DB_NAME );

    // verify table is dropped
    sql = "select table_name from information_schema.tables where table_schema = '" + TEST_DB_NAME + "'"
          " order by table_name";
    table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 7 );
    ASSERT_EQ( column->GetString( 0 ), "customer" );
    ASSERT_EQ( column->GetString( 1 ), "lineitem" );
    ASSERT_EQ( column->GetString( 2 ), "orders" );
    ASSERT_EQ( column->GetString( 3 ), "part" );
    ASSERT_EQ( column->GetString( 4 ), "partsupp" );
    ASSERT_EQ( column->GetString( 5 ), "region" );
    ASSERT_EQ( column->GetString( 6 ), "supplier" );

    // test drop database
    sql = "drop database " + TEST_DB_NAME;
    ExecuteSQL( sql, "" );

    // verify database is dropped
    sql = "select schema_name from information_schema.schemata where schema_name = '" + TEST_DB_NAME + "'";
    table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 0 );
}
TEST_F( UT_TestSchema, upper_case )
{
    std::string db_name = "Schema_Test_Db2";
    ExecuteSQL( "CREATE database if not exists " + db_name + ";", std::string() );
    // verify database is created
    string sql = "select schema_name from information_schema.schemata where schema_name = 'schema_test_db2'";
    auto table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    auto column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 1 );
    ASSERT_EQ( column->GetString( 0 ), "schema_test_db2" );

    ExecuteSQL( "CREATE table test(id int, name char(64));", db_name );
    // verify tables are created
    sql = "select table_name from information_schema.tables where table_schema = 'schema_test_db2'"
          " order by table_name";
    table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 1 );
    ASSERT_EQ( column->GetString( 0 ), "test" );

    ExecuteSQL( "DROP database " + db_name + ";", std::string() );
    sql = "select schema_name from information_schema.schemata where schema_name = 'schema_test_db2'";
    table = ExecuteSQL( sql, "" );
    ASSERT_EQ( table->GetColumnCount(), 1 );
    column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 0 );
}

TEST_F( UT_TestSchema, not_null_default_null )
{
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(f1 int not null default null );", database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_INVALID_DEFAULT );
}

TEST_F( UT_TestSchema, primary )
{
    // explicit default null
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(f1 int default null, primary key(f1));", database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_PRIMARY_CANT_HAVE_NULL );

    // explicit nullable
    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(f1 int null, primary key(f1));", database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_PRIMARY_CANT_HAVE_NULL );

    // nullable primary key, no explicit default null
    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(f1 int, f2 int, primary key (f1));", database_name );
    EXPECT_TRUE( result->IsSuccess() );

    // insert without explicite primary key field
    result = SQLExecutor::GetInstance()->ExecuteSQL( "insert into t1(f2) values(1);", database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NO_DEFAULT_FOR_FIELD );

    // insert explicit null value for primary key field
    result = SQLExecutor::GetInstance()->ExecuteSQL( "insert into t1 values(null, 1);", database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_BAD_NULL_ERROR );

    ExecuteSQL( "drop table t1;", database_name );

    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(id int primary, name char(64), age int);", database_name );
    EXPECT_FALSE( result->IsSuccess() );
    EXPECT_EQ( result->GetErrorCode(), ER_PARSE_ERROR );

    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(id int primary key, name char(64) primary key, age int);", database_name );
    EXPECT_FALSE( result->IsSuccess() );
    EXPECT_EQ( result->GetErrorCode(), ER_MULTIPLE_PRI_KEY );

    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t1(id int primary key, name char(64), age int);", database_name );
    EXPECT_TRUE( result->IsSuccess() );

    ExecuteSQL( "drop table t1;", database_name );

    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t2(id int, name char(64), age int, primary key(id, name, unknown));", database_name );
    EXPECT_FALSE( result->IsSuccess() );
    EXPECT_EQ( result->GetErrorCode(), ER_KEY_COLUMN_DOES_NOT_EXITS );

    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t2(id int, name char(64), age int, primary key(id, name, name));", database_name );
    EXPECT_FALSE( result->IsSuccess() );
    EXPECT_EQ( result->GetErrorCode(), ER_DUP_FIELDNAME );

    result = SQLExecutor::GetInstance()->ExecuteSQL( "create table t2(id int, name char(64), age int, primary key(id, name, age));", database_name );
    EXPECT_TRUE( result->IsSuccess() );

    ExecuteSQL( "drop table t2;", database_name );
}

// TEST_F( UT_TestSchema, foreign )
// {
//     ExecuteSQL( "create table t1(id int, name char(64), age int, primary key(id, name, age), comment char(64));", database_name );

//     auto sql = "create table t2(id int, name char(64), age int, foreign key(id, age) references t1(id, age));";
//     auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
//     EXPECT_FALSE( result->IsSuccess() );
//     // EXPECT_EQ( result->GetErrorCode(), ER_FK_NO_INDEX_PARENT ); // mysql 8.0
//     EXPECT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_FOREIGN ); // mysql 5.7

//     sql = "create table t2(id int, name char(64), age int, foreign key(id, age) references t1(name, id));";
//     result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
//     EXPECT_FALSE( result->IsSuccess() );
//     // EXPECT_EQ( result->GetErrorCode(), ER_FK_NO_INDEX_PARENT ); // mysql 8.0
//     EXPECT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_FOREIGN ); // mysql 5.7

//     sql = "create table t2(id int, name char(64), age int, foreign key(id, name) references t1(id, name2));";
//     result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
//     EXPECT_FALSE( result->IsSuccess() );
//     EXPECT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_FOREIGN );

//     sql = "create table t2(id int, name char(64), age int, foreign key(id, name) references t1(id, comment));";
//     result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
//     EXPECT_FALSE( result->IsSuccess() );
//     EXPECT_EQ( result->GetErrorCode(), ER_CANNOT_ADD_FOREIGN );

//     sql = "create table t2(id int, name char(64), age int, foreign key(id) references t1(id));";
//     ExecuteSQL( sql, database_name );

//     sql = "create table t3(id int, name char(64), age int, foreign key(id, name) references t1(id, name));";
//     ExecuteSQL( sql, database_name );

//     sql = "create table t4(id1 int, name1 char(64), age1 int, foreign key(id1, name1, age1) references t1(id, name, age));";
//     ExecuteSQL( sql, database_name );

//     sql = "create table t5(id int, name char(64), age int, constraint f_name foreign key(id, name, age) references t1(id, name, age));";
//     ExecuteSQL( sql, database_name );

//     sql = "create table t6(id int, name char(64), age int, constraint f_name foreign key(id, name, age) references t1(id, name, age));";
//     result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
//     EXPECT_FALSE( result->IsSuccess() );
//     EXPECT_EQ( result->GetErrorCode(), ER_FK_DUP_NAME );

//     /*
//     sql = "create table t6(id int, name char(64), age int, constraint f_name_2 foreign key(id, name, age) references t1(id, name, age));";
//     ExecuteSQL( sql, database_name );

//     sql = "drop table t1;";
//     result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
//     EXPECT_FALSE( result->IsSuccess() );
//     EXPECT_EQ( result->GetErrorCode(), ER_ROW_IS_REFERENCED );

//     auto table1 = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name )->GetTableByName( "t1" );

//     ASSERT_EQ( table1->GetReferencings().size(), 5 );

//     auto referencings = table1->GetReferencings();

//     std::sort( referencings.begin(), referencings.end(), []( schema::ReferencingContraintSPtr left, schema::ReferencingContraintSPtr right )
//     {
//         return left->referencingTable < right->referencingTable;
//     } );

//     ASSERT_EQ( referencings[ 0 ]->keys.size(), 1 );
//     ASSERT_EQ( referencings[ 0 ]->keys[ 0 ], "id" );
//     ASSERT_EQ( referencings[ 0 ]->referencingKeys[ 0 ], "id" );
//     ASSERT_EQ( referencings[ 0 ]->referencingTable, "t2" );

//     ASSERT_EQ( referencings[ 1 ]->keys.size(), 2 );
//     ASSERT_EQ( referencings[ 1 ]->keys[ 0 ], "id" );
//     ASSERT_EQ( referencings[ 1 ]->referencingKeys[ 0 ], "id" );
//     ASSERT_EQ( referencings[ 1 ]->keys[ 1 ], "name" );
//     ASSERT_EQ( referencings[ 1 ]->referencingKeys[ 1 ], "name" );
//     ASSERT_EQ( referencings[ 1 ]->referencingTable, "t3" );

//     ASSERT_EQ( referencings[ 2 ]->keys.size(), 3 );
//     ASSERT_EQ( referencings[ 2 ]->keys[ 0 ], "id" );
//     ASSERT_EQ( referencings[ 2 ]->referencingKeys[ 0 ], "id1" );
//     ASSERT_EQ( referencings[ 2 ]->keys[ 1 ], "name" );
//     ASSERT_EQ( referencings[ 2 ]->referencingKeys[ 1 ], "name1" );
//     ASSERT_EQ( referencings[ 2 ]->keys[ 2 ], "age" );
//     ASSERT_EQ( referencings[ 2 ]->referencingKeys[ 2 ], "age1" );
//     ASSERT_EQ( referencings[ 2 ]->referencingTable, "t4" );
//     */

//     ExecuteSQL( "drop table t2;", database_name );
//     ExecuteSQL( "drop table t3;", database_name );
//     ExecuteSQL( "drop table t4;", database_name );
//     ExecuteSQL( "drop table t5;", database_name );
//     ExecuteSQL( "drop table if exists t6;", database_name );
//     ExecuteSQL( "drop table t1;", database_name );
// }

TEST_F( UT_TestSchema, unique_key )
{
    string tableName( "t1" );
    InitTable( database_name, tableName );

    string sql = "create table " + tableName + "(f1 int, f2 int, unique key uk1(f1), unique key uk1(f1));";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_DUP_KEYNAME );

    sql = "create table " + tableName + "(f1 int, f2 int, constraint uk1 unique key (f1), constraint uk1 unique key (f1));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_DUP_KEYNAME );

    sql = "create table " + tableName + "(f1 int, f2 int, constraint uk1 unique key (f1), unique key uk1(f2));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_DUP_KEYNAME );

    sql = "create table " +tableName + "(f1 int, f2 int, constraint uk1 unique key (f1), constraint uk2 unique key (f2));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    sql = "select CONSTRAINT_NAME, CONSTRAINT_TYPE from information_schema.table_constraints ";
    sql += "WHERE TABLE_SCHEMA = '" + database_name + "' AND TABLE_NAME ='" + tableName + "';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "uk1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "uk2" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "UNIQUE" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "UNIQUE" );

    sql = "select CONSTRAINT_NAME, COLUMN_NAME, ORDINAL_POSITION from information_schema.key_column_usage ";
    sql += "WHERE TABLE_SCHEMA = '" + database_name + "' AND TABLE_NAME ='" + tableName + "' ORDER BY CONSTRAINT_NAME;";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "uk1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "uk2" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "f1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "f2" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt64( 1 ), 1 );

    // multi
    tableName = "t2";
    InitTable( database_name, tableName );

    sql = "create table " + tableName + "(f1 int, f2 int, f3 int, constraint uk1 unique key (f1, f2), constraint uk2 unique key (f3));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    sql = "select CONSTRAINT_NAME, CONSTRAINT_TYPE from information_schema.table_constraints ";
    sql += "WHERE TABLE_SCHEMA = '" + database_name + "' AND TABLE_NAME ='" + tableName + "';";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "uk1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "uk2" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "UNIQUE" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "UNIQUE" );

    sql = "select CONSTRAINT_NAME, COLUMN_NAME, ORDINAL_POSITION from information_schema.key_column_usage ";
    sql += "WHERE TABLE_SCHEMA = '" + database_name + "' AND TABLE_NAME ='" + tableName + "' ORDER BY CONSTRAINT_NAME, ORDINAL_POSITION;";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 3 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "uk1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "uk1" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "uk2" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "f1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "f2" );
    ASSERT_EQ( columnBuff->GetString( 2 ), "f3" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt64( 1 ), 2 );
    ASSERT_EQ( columnBuff->GetInt64( 2 ), 1 );
}

TEST_F( UT_TestSchema, load )
{
    ExecuteSQL( "create table t1(id int, name char(64), age int, primary key(id, name, age), comment char(64));", database_name );

    auto sql = "create table t2(id int, name char(64), age int);";
    ExecuteSQL( sql, database_name );

    sql = "create table t3(id int, name char(64), age int);";
    ExecuteSQL( sql, database_name );

    sql = "create table t4(id1 int, name1 char(64), age1 int);";
    ExecuteSQL( sql, database_name );

    sql = "create table t5(id int, name char(64), age int);";
    ExecuteSQL( sql, database_name );

    sql = "create table t6(id int, name char(64), age int);";
    ExecuteSQL( sql, database_name );

    sql = "create table t7(id int, name char(64), age int, primary key( id, name, age ));";
    ExecuteSQL( sql, database_name );

    auto new_sql = R"(
create table t8(
    id1 int,
    name1 char(64),
    age1 int,
    id2 int,
    name2 char(64),
    age2 int
);
    )";
    ExecuteSQL( new_sql, database_name );

    auto schema = new aries::schema::Schema();
    schema->LoadSchema();

    auto database = schema->GetDatabaseByName( database_name );
    ASSERT_TRUE( database );
    auto t1 = database->GetTableByName( "t1" );
    ASSERT_TRUE( t1 );

    auto t2 = database->GetTableByName( "t2");
    ASSERT_TRUE( t2 );

    auto t3 = database->GetTableByName( "t3");
    ASSERT_TRUE( t3 );

    auto t4 = database->GetTableByName( "t4");
    ASSERT_TRUE( t4 );

    auto t5 = database->GetTableByName( "t5");
    ASSERT_TRUE( t5 );

    auto t6 = database->GetTableByName( "t6");
    ASSERT_TRUE( t6 );

    auto t8 = database->GetTableByName( "t8");
    ASSERT_TRUE( t8 );

    ASSERT_EQ( t1->GetPrimaryKeyName(), "PRIMARY" );
    ASSERT_EQ( t1->GetPrimaryKey().size(), 3 );
    ASSERT_EQ( t1->GetPrimaryKey()[ 0 ], "id" );
    ASSERT_EQ( t1->GetPrimaryKey()[ 1 ], "name" );
    ASSERT_EQ( t1->GetPrimaryKey()[ 2 ], "age" );

    /*
    // TODO: foreign key to be supported
    ASSERT_EQ( t2->GetForeignKeys().size(), 1 );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->name, "t2_ibfk_1" );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->keys.size(), 1 );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->referencedSchema, database_name );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->referencedTable, "t1" );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->referencedKeys.size(), 1 );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->keys[ 0 ], "id" );
    ASSERT_EQ( t2->GetForeignKeys()[ 0 ]->referencedKeys[ 0 ], "id" );

    ASSERT_EQ( t3->GetForeignKeys().size(), 1 );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->name, "t3_ibfk_1" );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->keys.size(), 2 );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->referencedSchema, database_name );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->referencedTable, "t1" );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->referencedKeys.size(), 2 );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->keys[ 0 ], "id" );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->keys[ 1 ], "name" );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->referencedKeys[ 0 ], "id" );
    ASSERT_EQ( t3->GetForeignKeys()[ 0 ]->referencedKeys[ 1 ], "name" );

    ASSERT_EQ( t4->GetForeignKeys().size(), 1 );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->name, "t4_ibfk_1" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->keys.size(), 3 );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->referencedSchema, database_name );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->referencedTable, "t1" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->referencedKeys.size(), 3 );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->keys[ 0 ], "id1" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->keys[ 1 ], "name1" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->keys[ 2 ], "age1" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->referencedKeys[ 0 ], "id" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->referencedKeys[ 1 ], "name" );
    ASSERT_EQ( t4->GetForeignKeys()[ 0 ]->referencedKeys[ 2 ], "age" );

    ASSERT_EQ( t5->GetForeignKeys().size(), 1 );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->name, "f_name" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->keys.size(), 3 );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->referencedSchema, database_name );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->referencedTable, "t1" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->referencedKeys.size(), 3 );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->keys[ 0 ], "id" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->keys[ 1 ], "name" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->keys[ 2 ], "age" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->referencedKeys[ 0 ], "id" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->referencedKeys[ 1 ], "name" );
    ASSERT_EQ( t5->GetForeignKeys()[ 0 ]->referencedKeys[ 2 ], "age" );

    ASSERT_EQ( t6->GetForeignKeys().size(), 1 );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->name, "f_name_2" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->keys.size(), 3 );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->referencedSchema, database_name );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->referencedTable, "t1" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->referencedKeys.size(), 3 );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->keys[ 0 ], "id" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->keys[ 1 ], "name" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->keys[ 2 ], "age" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->referencedKeys[ 0 ], "id" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->referencedKeys[ 1 ], "name" );
    ASSERT_EQ( t6->GetForeignKeys()[ 0 ]->referencedKeys[ 2 ], "age" );

    ASSERT_EQ( t8->GetForeignKeys().size(), 2 );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->name, "f_name_3" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->name, "f_name_4" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->keys.size(), 3 );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->referencedSchema, database_name );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->referencedTable, "t1" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->referencedKeys.size(), 3 );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->keys[ 0 ], "id1" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->keys[ 1 ], "name1" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->keys[ 2 ], "age1" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->referencedKeys[ 0 ], "id" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->referencedKeys[ 1 ], "name" );
    ASSERT_EQ( t8->GetForeignKeys()[ 0 ]->referencedKeys[ 2 ], "age" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->keys.size(), 3 );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->referencedSchema, database_name );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->referencedTable, "t7" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->referencedKeys.size(), 3 );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->keys[ 0 ], "id2" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->keys[ 1 ], "name2" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->keys[ 2 ], "age2" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->referencedKeys[ 0 ], "id" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->referencedKeys[ 1 ], "name" );
    ASSERT_EQ( t8->GetForeignKeys()[ 1 ]->referencedKeys[ 2 ], "age" );

    ASSERT_EQ( t1->GetReferencings().size(), 6 );

    auto referencings = t1->GetReferencings();

    std::sort( referencings.begin(), referencings.end(), []( schema::ReferencingContraintSPtr left, schema::ReferencingContraintSPtr right )
    {
        return left->referencingTable < right->referencingTable;
    } );

    ASSERT_EQ( referencings[ 0 ]->keys.size(), 1 );
    ASSERT_EQ( referencings[ 0 ]->keys[ 0 ], "id" );
    ASSERT_EQ( referencings[ 0 ]->referencingKeys[ 0 ], "id" );
    ASSERT_EQ( referencings[ 0 ]->referencingTable, "t2" );

    ASSERT_EQ( referencings[ 1 ]->keys.size(), 2 );
    ASSERT_EQ( referencings[ 1 ]->keys[ 0 ], "id" );
    ASSERT_EQ( referencings[ 1 ]->referencingKeys[ 0 ], "id" );
    ASSERT_EQ( referencings[ 1 ]->keys[ 1 ], "name" );
    ASSERT_EQ( referencings[ 1 ]->referencingKeys[ 1 ], "name" );
    ASSERT_EQ( referencings[ 1 ]->referencingTable, "t3" );

    ASSERT_EQ( referencings[ 2 ]->keys.size(), 3 );
    ASSERT_EQ( referencings[ 2 ]->keys[ 0 ], "id" );
    ASSERT_EQ( referencings[ 2 ]->referencingKeys[ 0 ], "id1" );
    ASSERT_EQ( referencings[ 2 ]->keys[ 1 ], "name" );
    ASSERT_EQ( referencings[ 2 ]->referencingKeys[ 1 ], "name1" );
    ASSERT_EQ( referencings[ 2 ]->keys[ 2 ], "age" );
    ASSERT_EQ( referencings[ 2 ]->referencingKeys[ 2 ], "age1" );
    ASSERT_EQ( referencings[ 2 ]->referencingTable, "t4" );
    */

    delete schema;

    ExecuteSQL( "drop table if exists t2;", database_name );
    ExecuteSQL( "drop table if exists t3;", database_name );
    ExecuteSQL( "drop table if exists t4;", database_name );
    ExecuteSQL( "drop table if exists t5;", database_name );
    ExecuteSQL( "drop table if exists t6;", database_name );
    ExecuteSQL( "drop table if exists t8;", database_name );
    ExecuteSQL( "drop table if exists t7;", database_name );
    ExecuteSQL( "drop table if exists t1;", database_name );
}

TEST_F( UT_TestSchema, partition )
{
    string tableName( "t_partition" );
    InitTable( database_name, tableName );

    // NOT SUPPORTED COLUMN TYPE
    string sql = R"(
CREATE TABLE t_partition(
 year_col float,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );

    sql = R"(
CREATE TABLE t_partition(
 year_col int
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // partition by LIST
    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY LIST ( year_col ) (
 PARTITION p0 VALUES IN ( 1991 ),
 PARTITION p1 VALUES IN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // VALUES LESS THAN ( null )
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE ( year_col ) (
 PARTITION p0 VALUES LESS THAN ( null ),
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NULL_IN_VALUES_LESS_THAN );

    // VALUES LESS THAN (1991+1),
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE ( year_col ) (
 PARTITION p0 VALUES LESS THAN (1991+1),
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );

    // PARTITION BY RANGE (year(date1)) (
    sql = R"(
CREATE TABLE t_partition(
 date1 date,
 some_data INT
)
PARTITION BY RANGE (year(date1)) (
 PARTITION p0 VALUES LESS THAN (1991),
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // bad field
    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY RANGE (qqq) (
 PARTITION p0 VALUES LESS THAN (1991),
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_BAD_FIELD_ERROR );

    // bad field
    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY RANGE (aaa.year_col) (
 PARTITION p0 VALUES LESS THAN (1991),
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_BAD_FIELD_ERROR );

    // bad field
    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY RANGE (aaaa.t_partition.year_col) (
 PARTITION p0 VALUES LESS THAN (1991),
 PARTITION p1 VALUES LESS THAN ( 2000 )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_BAD_FIELD_ERROR );

    // duplicated partition name
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p0 VALUES LESS THAN ( "1991-01-01" ),
 PARTITION p1 VALUES LESS THAN ( "2000-01-01" ),
 PARTITION p0 VALUES LESS THAN ( "2010-01-01" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_SAME_NAME_PARTITION );

    // MAXVALUE should be the last one 
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p0 VALUES LESS THAN ( "1991-01-01" ),
 PARTITION p1 VALUES LESS THAN MAXVALUE,
 PARTITION p1 VALUES LESS THAN ( "2000-01-01" ),
 PARTITION p3 VALUES LESS THAN MAXVALUE
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_PARTITION_MAXVALUE_ERROR );

    // wrong date format
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( "abc" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( "2000" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // wrong datetime format
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( "2o00-01 00:00:19" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( "2000-01 00:00:19" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_TRUNCATED_WRONG_VALUE_FOR_FIELD );

    // not ordered date
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( "2000-01-01" ),
 PARTITION p0 VALUES LESS THAN ( "1991-01-01" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_RANGE_NOT_INCREASING_ERROR );

    // not ordered datetime
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p1 VALUES LESS THAN ( "2000-01-01 00:00:19" ),
 PARTITION p0 VALUES LESS THAN ( "1991-01-01 23:00:59" )
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_RANGE_NOT_INCREASING_ERROR );

    // non-int value for int column
    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p0 VALUES LESS THAN ('1991'),
 PARTITION p2 VALUES LESS THAN (1990)
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    // EXPECT_EQ( result->GetErrorCode(), ER_VALUES_IS_NOT_INT_TYPE_ERROR );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p0 VALUES LESS THAN (1991),
 PARTITION p2 VALUES LESS THAN (1.0)
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    // EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

    // int value for date column
    sql = R"(
CREATE TABLE t_partition(
 date_col DATE,
 some_data INT
)
PARTITION BY RANGE (date_col) (
 PARTITION p0 VALUES LESS THAN ('1991-01-01'),
 PARTITION p2 VALUES LESS THAN (1990)
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );

    // float value for date column
    sql = R"(
CREATE TABLE t_partition(
 date_col DATE,
 some_data INT
)
PARTITION BY RANGE (date_col) (
 PARTITION p0 VALUES LESS THAN ('1991-01-01'),
 PARTITION p2 VALUES LESS THAN (1.0)
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );

    // int value for datetime column
    sql = R"(
CREATE TABLE t_partition(
 datetime_col DATETIME,
 some_data INT
)
PARTITION BY RANGE (datetime_col) (
 PARTITION p2 VALUES LESS THAN (1990)
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );

    // float value for datetime column
    sql = R"(
CREATE TABLE t_partition(
 datetime_col DATETIME,
 some_data INT
)
PARTITION BY RANGE (datetime_col) (
 PARTITION p2 VALUES LESS THAN (1.0)
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD );

    // no partition definition
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_PARSE_ERROR );
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_PARTITIONS_MUST_BE_DEFINED_ERROR );

    // ONLY MAXVALUE partition
    sql = R"(
CREATE TABLE t_partition(
 year_col date,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p5 VALUES LESS THAN MAXVALUE
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    // partition by int column
    InitTable( database_name, tableName );
    sql = R"(
CREATE TABLE t_partition(
 year_col INT,
 some_data INT
)
PARTITION BY RANGE (year_col) (
 PARTITION p0 VALUES LESS THAN (1991),
 PARTITION p5 VALUES LESS THAN MAXVALUE
);
    )";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );

/*
    auto tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 0 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 2 );

    auto &partitions = tableEntry->GetPartitions();
    auto partition = partitions[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p0" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "1991" );

    partition = partitions[ 1 ];
    ASSERT_EQ( partition->m_partitionName, "p5" );
    ASSERT_EQ( partition->m_partOrdPos, 2 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 0 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 2 );

    auto &partitions2 = tableEntry->GetPartitions();
    partition = partitions2[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p0" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "1991" );

    partition = partitions2[ 1 ];
    ASSERT_EQ( partition->m_partitionName, "p5" );
    ASSERT_EQ( partition->m_partOrdPos, 2 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );
*/
    // partition by date column
    InitTable( database_name, tableName );
    sql = R"(
CREATE TABLE t_partition(
 some_data INT,
 date_col date
)
PARTITION BY RANGE (date_col) (
 PARTITION p0 VALUES LESS THAN ('1991-01-01'),
 PARTITION p1 VALUES LESS THAN MAXVALUE
);
    )";

    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), 0 );

    auto tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 1 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 2 );

    auto &partitions3 = tableEntry->GetPartitions();
    auto partition = partitions3[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p0" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "1991-01-01" );

    partition = partitions3[ 1 ];
    ASSERT_EQ( partition->m_partitionName, "p1" );
    ASSERT_EQ( partition->m_partOrdPos, 2 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto special_recoverr_result = special_recoveryer->Recovery();
    ARIES_ASSERT(special_recoverr_result, "cannot recovery(special) from xlog");

    aries::schema::SchemaManager::GetInstance()->Load();

    tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name )->GetTableByName( tableName );
    ASSERT_EQ( tableEntry->GetPartitionMethod(), "RANGE" );
    ASSERT_EQ( tableEntry->GetPartitionColumnIndex(), 1 );
    ASSERT_EQ( tableEntry->GetPartitionCount(), 2 );

    auto &partitions4 = tableEntry->GetPartitions();
    partition = partitions4[ 0 ];
    ASSERT_EQ( partition->m_partitionName, "p0" );
    ASSERT_EQ( partition->m_partOrdPos, 1 );
    ASSERT_EQ( partition->m_partDesc, "1991-01-01" );

    partition = partitions4[ 1 ];
    ASSERT_EQ( partition->m_partitionName, "p1" );
    ASSERT_EQ( partition->m_partOrdPos, 2 );
    ASSERT_EQ( partition->m_partDesc, "MAXVALUE" );

    sql = "insert into " + tableName + " values(1, 1)";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    sql = "delete from " + tableName;
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
    sql = "update " + tableName + " set year_col = 100 ";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, database_name );
    EXPECT_EQ( result->GetErrorCode(), ER_NOT_SUPPORTED_YET );
}