#include <gtest/gtest.h>
#include <regex>
#include <string>

#include "AriesEngineWrapper/AriesMemTable.h"
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"
#include <server/mysql/include/mysqld_error.h>
#include "server/mysql/include/mysqld.h"
#include "server/mysql/include/sql_class.h"

#include <schema/SchemaManager.h>
#include "schema/DatabaseEntry.h"
#include "schema/TableEntry.h"

#include "TestUtils.h"

using namespace aries_engine;
using namespace std;
using namespace aries_test;
using std::string;

using aries::schema::SchemaManager;
using aries::schema::DatabaseEntry;
using aries::schema::TableEntry;

extern string DB_NAME;

AriesTableBlockUPtr TestBackdoorV2( const string arg_query_name )
{
    std::string db_name = "scale_1";
    if ( !DB_NAME.empty() )
        db_name = DB_NAME;
    std::string filename = "test_tpch_queries/" + arg_query_name + ".sql";
    cout << "using database " << db_name << endl;
    current_thd->set_db( db_name );
    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQLFromFile( filename, db_name );
    auto art = make_unique< AriesTableBlock >();
    if (results->IsSuccess() && results->GetResults().size() > 0) {
        auto amtp = results->GetResults()[ 0 ];
        auto block = std::move( ( ( ( AriesMemTable * )( amtp.get() ) )->GetContent() ) );
        int count = block->GetRowCount();
        //art->Dump();
        cout << "tupleNum is:" << count << endl;
        return block;
    }
    return art;
}

AriesTableBlockUPtr TestBackdoorV2( const string& sql_dir, const string& arg_query_name )
{
    std::string db_name = "scale_1";
    if ( !DB_NAME.empty() )
        db_name = DB_NAME;
    std::string filename =  sql_dir + "/" + arg_query_name + ".sql";
    cout << "using database " << db_name << endl;
    current_thd->set_db( db_name );
    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQLFromFile( filename, db_name );
    auto art = make_unique< AriesTableBlock >();
    if (results->IsSuccess() && results->GetResults().size() > 0) {
        auto amtp = results->GetResults()[ 0 ];
        auto block = std::move( ( ( ( AriesMemTable * )( amtp.get() ) )->GetContent() ) );
        int count = block->GetRowCount();
        //art->Dump();
        cout << "tupleNum is:" << count << endl;
        return block;
    }
    return art;
}

AriesTableBlockUPtr TestBackdoorV2( int arg_query_number )
{
    return TestBackdoorV2( std::to_string( arg_query_number) );
}

AriesTableBlockUPtr TestBackdoorV2( const string& sql_dir, int arg_query_number )
{
    return TestBackdoorV2( sql_dir, std::to_string( arg_query_number) );
}

// 0.001
TEST(tpch_1, q1)
{
    auto table = TestBackdoorV2( 1 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 10 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "A" );
    ASSERT_EQ( column->GetString( 1 ), "N" );
    ASSERT_EQ( column->GetString( 2 ), "N" );
    ASSERT_EQ( column->GetString( 3 ), "R" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "F" );
    ASSERT_EQ( column->GetString( 1 ), "F" );
    ASSERT_EQ( column->GetString( 2 ), "O" );
    ASSERT_EQ( column->GetString( 3 ), "F" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "37734107.00" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "991417.00" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "72675577.00" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "37719753.00" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "56586554400.73" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "1487504710.38" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "109001350084.26" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "56568041380.90" );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "53758257134.8700" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "1413082168.0541" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "103552520554.2534" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "53741292684.6040" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "55909065222.827692" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "1469649223.194375" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "107698448674.818097" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "55889619119.831932" );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "25.522006" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "25.516472" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "25.502314" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "25.505794" );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "38273.129735" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "38284.467761" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "38249.255056" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "38250.854626" );

    // verify column 9
    column = table->GetColumnBuffer( 9 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.049985" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "0.050093" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "0.049999" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "0.050009" );

    // verify column 10
    column = table->GetColumnBuffer( 10 );
    ASSERT_EQ( column->GetInt64( 0 ), 1478493 );
    ASSERT_EQ( column->GetInt64( 1 ), 38854 );
    ASSERT_EQ( column->GetInt64( 2 ), 2849764 );
    ASSERT_EQ( column->GetInt64( 3 ), 1478870 );
}

TEST(tpch_1, q2)
{
    auto table = TestBackdoorV2( 2 );
    int64_t columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 8 );
    int64_t tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2343 );
    table.reset();
    //FIXME no order by. can't compare by index
}

TEST(tpch_1, q3)
{
    auto table = TestBackdoorV2( 3 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 11439 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1670759 );
    ASSERT_EQ( column->GetInt32( 6269 ), 5356994 );
    ASSERT_EQ( column->GetInt32( 11438 ), 1112002 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "394096.6203" );
    ASSERT_EQ( column->GetDecimalAsString( 6269 ), "71925.6700" );
    ASSERT_EQ( column->GetDecimalAsString( 11438 ), "967.8952" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1995-02-20" );
    ASSERT_EQ( column->GetDateAsString( 6269 ), "1994-12-24" );
    ASSERT_EQ( column->GetDateAsString( 11438 ), "1994-12-11" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 6269 ), 0 );
    ASSERT_EQ( column->GetInt32( 11438 ), 0 );
}

TEST(tpch_1, q4)
{
    auto table = TestBackdoorV2( 4 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "1-URGENT" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 4 ).c_str(), "5-LOW" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 10437 );
    ASSERT_EQ( column->GetInt64( 4 ), 10520 );
}

TEST(tpch_1, q5)
{
    auto table = TestBackdoorV2( 5 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "ALGERIA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 1 ).c_str(), "MOZAMBIQUE" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 2 ).c_str(), "ETHIOPIA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 3 ).c_str(), "KENYA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 4 ).c_str(), "MOROCCO" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "55934306.6140" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "55762583.0723" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "52998703.9473" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "49930581.0204" );
    ASSERT_EQ( column->GetDecimalAsString( 4 ), "49162063.0310" );
}

TEST(tpch_1, q6)
{
    auto table = TestBackdoorV2( 6 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "112296798.0742" );
}

TEST(tpch_1, q7)
{
    auto table = TestBackdoorV2( 7 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "IRAQ" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 1 ).c_str(), "IRAQ" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 2 ).c_str(), "KENYA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 3 ).c_str(), "KENYA" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "KENYA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 1 ).c_str(), "KENYA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 2 ).c_str(), "IRAQ" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 3 ).c_str(), "IRAQ" ), 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32AsString( 0 ), "1995" );
    ASSERT_EQ( column->GetInt32AsString( 1 ), "1996" );
    ASSERT_EQ( column->GetInt32AsString( 2 ), "1995" );
    ASSERT_EQ( column->GetInt32AsString( 3 ), "1996" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "53682901.5202" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "57729548.5892" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "45402679.9204" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "46672599.8198" );
}

TEST(tpch_1, q8)
{
    auto table = TestBackdoorV2( 8 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32AsString( 0 ), "1995" );
    ASSERT_EQ( column->GetInt32AsString( 1 ), "1996" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.04546296");
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "0.03800943");
}

TEST(tpch_1, q9)
{
    auto table = TestBackdoorV2( 9 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 175 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "ALGERIA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 174 ).c_str(), "VIETNAM" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32AsString( 0 ), "1998" );
    ASSERT_EQ( column->GetInt32AsString( 174 ), "1992" );

    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "30767419.2860" );
    ASSERT_EQ( column->GetDecimalAsString( 174 ), "49003297.8053" );
}

TEST(tpch_1, q10)
{
    auto table = TestBackdoorV2( 10 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 8 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 35593 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 51811 );
    ASSERT_EQ( column->GetInt32( 18 ), 109168 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "Customer#000051811" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 18 ).c_str(), "Customer#000109168" ), 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "762516.0781" );
    ASSERT_EQ( column->GetDecimalAsString( 18 ), "481087.0855" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "1016.15" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "-219.55" );
    ASSERT_EQ( column->GetDecimalAsString( 18 ), "2947.40" );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "CHINA" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 18 ).c_str(), "CHINA" ), 0 );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), " iahxUHfAxlJkZWz4iPGDmAQJSWl UlrqXRW" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 18 ).c_str(), "9HqXcL6X4eyYc4OUd" ), 0 );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "28-777-205-1675" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 18 ).c_str(), "28-198-666-9028" ), 0 );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_EQ(
            strcmp( column->GetString( 0 ).c_str(),
                    "ajole slyly fluffily even asymptotes. packages sleep furiously. blithely pending packages cajole: express " ), 0 );
    ASSERT_EQ(
            strcmp( column->GetString( 18 ).c_str(),
                    "efully final, regular asymptotes. quickly ironic packages cajole carefully. blithely final platelets wak" ), 0 );
}

TEST(tpch_1, q11)
{
    auto table = TestBackdoorV2( 11 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 753 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 18488 );
    ASSERT_EQ( column->GetInt32( 18 ), 192862 );
    ASSERT_EQ( column->GetInt32( 752 ), 184454 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "19174541.94" );
    ASSERT_EQ( column->GetDecimalAsString( 18 ), "14196434.62" );
    ASSERT_EQ( column->GetDecimalAsString( 752 ), "8298660.62" );
}

TEST(tpch_1, q12)
{
    auto table = TestBackdoorV2( 12 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "AIR" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 1 ).c_str(), "TRUCK" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "6169" );
    ASSERT_EQ( column->GetInt64AsString( 1 ), "6156" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "9444" );
    ASSERT_EQ( column->GetInt64AsString( 1 ), "9206" );
}

TEST(tpch_1, q13)
{
    auto table = TestBackdoorV2( 13 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 42 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 0 );
    ASSERT_EQ( column->GetInt64( 41 ), 39 );

    // verify column 3
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 50004 );
    ASSERT_EQ( column->GetInt64( 41 ), 1 );
}

TEST(tpch_1, q14)
{
    auto table = TestBackdoorV2( 14 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "16.7286491435" );
}

TEST(tpch_1, q15)
{
    auto table = TestBackdoorV2( 15 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 984 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "Supplier#000000984" ), 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "6H6qqye iYbYzCmwWhj" ), 0 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "31-519-879-5266" ), 0 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "1939192.2484" );
}

TEST(tpch_1, q16)
{
    auto table = TestBackdoorV2( 16 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 18297 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "Brand#12" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 21 ).c_str(), "Brand#52" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "STANDARD POLISHED STEEL" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 21 ).c_str(), "SMALL ANODIZED COPPER" ), 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 15 );
    ASSERT_EQ( column->GetInt32( 21 ), 3 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt64( 0 ), 28 );
    ASSERT_EQ( column->GetInt64( 21 ), 23 );
}

TEST(tpch_1, q17)
{
    auto table = TestBackdoorV2( 17 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "352372.151429" );
}

TEST(tpch_1, q18)
{
    auto table = TestBackdoorV2( 18 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 6 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 10 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "Customer#000128120" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 9 ).c_str(), "Customer#000119989" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 128120 );
    ASSERT_EQ( column->GetInt32( 9 ), 119989 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 4722021 );
    ASSERT_EQ( column->GetInt32( 9 ), 1544643 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( strcmp( column->GetDateAsString( 0 ).c_str(), "1994-04-07" ), 0 );
    ASSERT_EQ( strcmp( column->GetDateAsString( 9 ).c_str(), "1997-09-20" ), 0 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "544089.09" );
    ASSERT_EQ( column->GetDecimalAsString( 9 ), "434568.25" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "323.00" );
    ASSERT_EQ( column->GetDecimalAsString( 9 ), "320.00" );
}

TEST(tpch_1, q19)
{
    auto table = TestBackdoorV2( 19 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "4336969.9274" );
}

TEST(tpch_1, q20)
{
    auto table = TestBackdoorV2( 20 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 190 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "Supplier#000000024" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 189 ).c_str(), "Supplier#000009993" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "C4nPvLrVmKPPabFCj" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 189 ).c_str(), "vwhUkukD cuAjoCNRj,vw,jSFRY5zzfNLO" ), 0 );
}

TEST(tpch_1, q21)
{
    auto table = TestBackdoorV2( 21 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 411 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "Supplier#000002829" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 410 ).c_str(), "Supplier#000008136" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 20 );
    ASSERT_EQ( column->GetInt64( 410 ), 3 );
}

TEST(tpch_1, q22)
{
    auto table = TestBackdoorV2( 22 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 7 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "10" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 6 ).c_str(), "34" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 881 );
    ASSERT_EQ( column->GetInt64( 6 ), 949 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "6601086.63" );
    ASSERT_EQ( column->GetDecimalAsString( 6 ), "7181952.16" );
}


TEST(smoke_1, q23)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 23 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 10 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "A" );
    ASSERT_EQ( column->GetString( 1 ), "N" );
    ASSERT_EQ( column->GetString( 2 ), "N" );
    ASSERT_EQ( column->GetString( 3 ), "R" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "F" );
    ASSERT_EQ( column->GetString( 1 ), "F" );
    ASSERT_EQ( column->GetString( 2 ), "O" );
    ASSERT_EQ( column->GetString( 3 ), "F" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "37734107.00" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "991417.00" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "72675577.00" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "37719753.00" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "56586554400.73" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "1487504710.38" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "109001350084.26" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "56568041380.90" );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "53758257134.8700" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "1413082168.0541" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "103552520554.2534" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "53741292684.6040" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "55909065222.827692" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "1469649223.194375" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "107698448674.818097" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "55889619119.831932" );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "25.522006" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "25.516472" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "25.502314" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "25.505794" );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "38273.129735" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "38284.467761" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "38249.255056" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "38250.854626" );

    // verify column 9
    column = table->GetColumnBuffer( 9 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.049985" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "0.050093" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "0.049999" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "0.050009" );

    // verify column 10
    column = table->GetColumnBuffer( 10 );
    ASSERT_EQ( column->GetInt64( 0 ), 1478493 );
    ASSERT_EQ( column->GetInt64( 1 ), 38854 );
    ASSERT_EQ( column->GetInt64( 2 ), 2849764 );
    ASSERT_EQ( column->GetInt64( 3 ), 1478870 );
}

TEST(smoke_1, q24)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 24 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.000460" );
}

TEST(smoke_1, q25)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 25 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 165120 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 4576548 );
    ASSERT_EQ( column->GetInt32( 6269 ), 3420518 );
    ASSERT_EQ( column->GetInt32( 11438 ), 5761507 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "504144.3558" );
    ASSERT_EQ( column->GetDecimalAsString( 6269 ), "303099.1197" );
    ASSERT_EQ( column->GetDecimalAsString( 11438 ), "278187.3300" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1997-12-26" );
    ASSERT_EQ( column->GetDateAsString( 6269 ), "1997-07-26" );
    ASSERT_EQ( column->GetDateAsString( 11438 ), "1996-07-31" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 6269 ), 0 );
    ASSERT_EQ( column->GetInt32( 11438 ), 0 );
}

TEST(smoke_1, q26)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 26 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 18 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetString( 0 ).c_str(), "12" ), 0 );
    ASSERT_EQ( strcmp( column->GetString( 17 ).c_str(), "33" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 957 );
    ASSERT_EQ( column->GetInt64( 17 ), 1042 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "6904717.51" );
    ASSERT_EQ( column->GetDecimalAsString( 17 ), "7522292.16" );
}

TEST(smoke_1, q27)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 27 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 126601 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetItemCount(), 126601 );
    ASSERT_EQ( column->GetInt32( 0 ), 3233286 );
    ASSERT_EQ( column->GetInt32( 14990 ), 5529570 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetItemCount(), 126601 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "466492.3944" );
    ASSERT_EQ( column->GetDecimalAsString( 14990 ), "250856.9319" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetItemCount(), 126601 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1997-04-01" );
    ASSERT_EQ( column->GetDateAsString( 14990 ), "1995-07-22" );

    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetItemCount(), 126601 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 14990 ), 0 );
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetItemCount(), 126601 );
}

TEST(smoke_1, q28)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 28 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1217 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "836323200" );
    ASSERT_EQ( column->GetDecimalAsString( 1215 ), "843753600" );
    ASSERT_EQ( column->GetDecimalAsString( 1216 ), "843840000" );
}

TEST(smoke_1, q32)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 32 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 365 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "343587.6337" );
    ASSERT_EQ( column->GetDecimalAsString( 364 ), "282257.7760" );
}

TEST(smoke_1, q42)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 42 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1500000 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 7 ), "NULL" );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "857.71" );
    ASSERT_EQ( column->GetDecimalAsString( 7 ), "896.59" );
}

TEST(smoke_1, q43)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 43 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1537756 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "-999.99" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "-999.98" );
    ASSERT_EQ( column->GetDecimalAsString( 18990 ), "-869.12" );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "65957.77" );
    ASSERT_EQ( column->GetDecimalAsString( 18990 ), "162457.18" );
}

TEST(smoke_1, q44)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 44 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1550004 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 12247 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 12248 ), "-999.99" );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "857.71" );
    ASSERT_EQ( column->GetDecimalAsString( 12247 ), "9960.63" );
    ASSERT_EQ( column->GetDecimalAsString( 12248 ), "NULL" );
}

TEST(smoke_1, q46)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 46 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32AsString( 0 ), "NULL" );
}

TEST(smoke_1, q47)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 47 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q48)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 48 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q49)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 49 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000");
}

TEST(smoke_1, q50)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 50 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q51)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 51 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q52)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 52 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q53)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 53 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q54)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 54 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q55)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 55 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q56)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 56 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q57)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 57 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q58)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 58 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q59)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 59 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "0" );
}

TEST(smoke_1, q60)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 60 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 0 );
}

TEST(smoke_1, q61)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 61 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q62)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 62 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q63)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 63 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q64)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 64 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, q65)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 65 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    ASSERT_EQ( table->GetRowCount(), 0 );
}

TEST(smoke_1, q66)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 66 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 0 );

    // verify column 1
    // AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    // ASSERT_EQ( column->GetInt64AsString( 0 ), "NULL" );
}

TEST(smoke_1, q67)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 67 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 25 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    for (int i = 0; i < tupleNum; i ++)
    {
        ASSERT_EQ( column->GetInt32( i ), 1 );
    }

    column = table->GetColumnBuffer( 3 );
    for (int i = 0; i < tupleNum; i ++)
    {
        ASSERT_EQ( column->GetString( i ), "constant string" );
    }

    column = table->GetColumnBuffer( 5 );
    for (int i = 0; i < tupleNum; i ++)
    {
        ASSERT_EQ( column->GetInt32( i ), 6 );
    }
}

TEST(smoke_1, q68)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 68 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetInt32( 0 ), 1 );

    column = table->GetColumnBuffer( 2 );

    ASSERT_EQ( column->GetString( 0 ), "abc" );

    column = table->GetColumnBuffer( 3 );

    ASSERT_EQ( column->GetDateAsString( 0 ), "2019-10-10" );

    column = table->GetColumnBuffer( 4 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "50.0000" );
}

TEST(smoke_1, q69)
{
    //result:
    /*
     * 0	0.000000
1	23810.375680
2	47644.351560
3	72046.032480
4	96411.416200
5	118069.026400
6	144058.082880
7	163469.658030
8	194181.019840
9	223649.842500
10	238397.674900
11	263319.062600
12	287093.624520
13	310373.258650
14	341747.941080
15	354975.182550
16	384574.503040
17	400669.461790
18	427107.818340
19	468798.245910
20	470073.866600
21	505637.951070
22	537214.820780
23	555038.300690
24	583795.172400
     * */

    std::vector<string> result = {
            "0.000000",
            "23810.375680",
            "47644.351560",
            "72046.032480",
            "96411.416200",
            "118069.026400",
            "144058.082880",
            "163469.658030",
            "194181.019840",
            "223649.842500",
            "238397.674900",
            "263319.062600",
            "287093.624520",
            "310373.258650",
            "341747.941080",
            "354975.182550",
            "384574.503040",
            "400669.461790",
            "427107.818340",
            "468798.245910",
            "470073.866600",
            "505637.951070",
            "537214.820780",
            "555038.300690",
            "583795.172400"
    };
    auto table = TestBackdoorV2( "smoke_test_queries", 69 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 25 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    for (int i = 0; i < 25; ++i)
    {
        ASSERT_EQ( column->GetInt32(i), i );
    }

    column = table->GetColumnBuffer( 2 );
    for (int i = 0; i < 25; ++i)
    {
        ASSERT_EQ( column->GetDecimalAsString(i), result[i] );
    }
}

//for limit node offset over row count
TEST(smoke_1, q70)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 70 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 0 );
}

TEST(smoke_1, q_filter_output_const_only)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "filter_output_const_only" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum,  1);

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal("1.13579") );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "abc" );
}

TEST(smoke_1, q_filter_output_const_only_and_empty)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "filter_output_const_only_and_empty" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum,  0);
}

TEST(smoke_1, q_filter_output_empty)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "filter_output_empty" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 0);
}

TEST(smoke_1, q_group_empty_result)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "group_empty_result" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    ASSERT_EQ( table->GetRowCount(), 0 );
}
TEST(smoke_1, q_group_distinct_empty_result)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "group_distinct_empty_result" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    ASSERT_EQ( table->GetRowCount(), 0 );
}
TEST(smoke_1, q_group_output_constant_columns_only)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "group_output_constant_columns_only" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );
    ASSERT_EQ( column->GetInt32( 2 ), 1 );
    ASSERT_EQ( column->GetInt32( 3 ), 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal("1.13579") );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal("1.13579") );
    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal("1.13579") );
    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal("1.13579") );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "abc" );
    ASSERT_EQ( column->GetString( 1 ), "abc" );
    ASSERT_EQ( column->GetString( 2 ), "abc" );
    ASSERT_EQ( column->GetString( 3 ), "abc" );
}

TEST(smoke_1, q_group_output_any_constant_columns)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "group_output_any_constant_columns" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 6 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "A" );
    ASSERT_EQ( column->GetString( 1 ), "N" );
    ASSERT_EQ( column->GetString( 2 ), "N" );
    ASSERT_EQ( column->GetString( 3 ), "R" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );
    ASSERT_EQ( column->GetInt32( 2 ), 1 );
    ASSERT_EQ( column->GetInt32( 3 ), 1 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "F" );
    ASSERT_EQ( column->GetString( 1 ), "F" );
    ASSERT_EQ( column->GetString( 2 ), "O" );
    ASSERT_EQ( column->GetString( 3 ), "F" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal("1.13579") );
    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal("1.13579") );
    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal("1.13579") );
    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal("1.13579") );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "37734107.00" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "991417.00" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "72675577.00" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "37719753.00" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetString( 0 ), "xyz" );
    ASSERT_EQ( column->GetString( 1 ), "xyz" );
    ASSERT_EQ( column->GetString( 2 ), "xyz" );
    ASSERT_EQ( column->GetString( 3 ), "xyz" );
}

TEST(smoke_1, q_left_join_output_constant_columns_only)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "left_join_output_constant_columns_only" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 6 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 3 ), 1 );
    ASSERT_EQ( column->GetInt32( 5 ), 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 5 ), aries_acc::Decimal("1.24680") );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "jackpot" );
    ASSERT_EQ( column->GetString( 3 ), "jackpot" );
    ASSERT_EQ( column->GetString( 5 ), "jackpot" );
}

TEST(smoke_1, q_left_join_right_table_empty1)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "left_join_right_table_empty1" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString(  0 ), "150000" );
}

TEST(smoke_1, q_left_join_right_table_empty2)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "left_join_right_table_empty2" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString(  0 ), "150000" ) ;
}

TEST(smoke_1, q_right_join_output_constant_columns_only)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "right_join_output_constant_columns_only" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 6 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 3 ), 1 );
    ASSERT_EQ( column->GetInt32( 5 ), 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 5 ), aries_acc::Decimal("1.24680") );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "jackpot" );
    ASSERT_EQ( column->GetString( 3 ), "jackpot" );
    ASSERT_EQ( column->GetString( 5 ), "jackpot" );
}

TEST(smoke_1, q_full_join_output_constant_columns_only)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "full_join_output_constant_columns_only" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 7 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 1 );
    ASSERT_EQ( column->GetInt32( 3 ), 1 );
    ASSERT_EQ( column->GetInt32( 5 ), 1 );
    ASSERT_EQ( column->GetInt32( 6 ), 1 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 5 ), aries_acc::Decimal("1.24680") );
    ASSERT_EQ( column->GetDecimal( 6 ), aries_acc::Decimal("1.24680") );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "jackpot" );
    ASSERT_EQ( column->GetString( 3 ), "jackpot" );
    ASSERT_EQ( column->GetString( 5 ), "jackpot" );
    ASSERT_EQ( column->GetString( 6 ), "jackpot" );
}

TEST(smoke_1, q_full_join_left_table_empty1)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "full_join_left_table_empty1" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString(  0 ), "150000" );
}

TEST(smoke_1, q_full_join_left_table_empty2)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "full_join_left_table_empty2" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString(  0 ), "150000" );
}

TEST(smoke_1, q_full_join_right_table_empty1)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "full_join_right_table_empty1" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString(  0 ), "150000" );
}

TEST(smoke_1, q_full_join_right_table_empty2)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "full_join_right_table_empty2" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString(  0 ), "150000" );
}

TEST(smoke_1, q71)
{
    auto table = TestBackdoorV2( "smoke_test_queries", 71 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64AsString( 0 ), "150000" );
}

TEST(smoke_1, left_join_dyn_code_compare)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "left_join_dyn_code_compare" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 7 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 9 );

    // verify column r_regionkey
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );
    ASSERT_EQ( column->GetInt32( 2 ), 2 );
    ASSERT_EQ( column->GetInt32( 3 ), 2 );
    ASSERT_EQ( column->GetInt32( 4 ), 2 );
    ASSERT_EQ( column->GetInt32( 5 ), 2 );
    ASSERT_EQ( column->GetInt32( 6 ), 2 );
    ASSERT_EQ( column->GetInt32( 7 ), 3 );
    ASSERT_EQ( column->GetInt32( 8 ), 4 );

    // verify column r_name
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( column->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( column->GetString( 2 ), "ASIA" );
    ASSERT_EQ( column->GetString( 3 ), "ASIA" );
    ASSERT_EQ( column->GetString( 4 ), "ASIA" );
    ASSERT_EQ( column->GetString( 5 ), "ASIA" );
    ASSERT_EQ( column->GetString( 6 ), "ASIA" );
    ASSERT_EQ( column->GetString( 7 ), "EUROPE" );
    ASSERT_EQ( column->GetString( 8 ), "MIDDLE EAST" );

    // verify column n_nationkey
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 8 );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 9 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 12 );
    ASSERT_EQ( column->GetNullableInt32( 5 ).value, 18 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 21 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 8 ).flag, 0 );

    // verify column n_name
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 1 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 2 ), "INDIA" );
    ASSERT_EQ( column->GetNullableString( 3 ), "INDONESIA" );
    ASSERT_EQ( column->GetNullableString( 4 ), "JAPAN" );
    ASSERT_EQ( column->GetNullableString( 5 ), "CHINA" );
    ASSERT_EQ( column->GetNullableString( 6 ), "VIETNAM" );
    ASSERT_EQ( column->GetNullableString( 7 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 8 ), "NULL" );

    // verify column n_regionkey
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 5 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 8 ).flag, 0 );
}

TEST(smoke_1, left_join_dyn_code_and_or)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "left_join_dyn_code_and_or" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 7 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 13 );

    // verify column r_regionkey
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );
    ASSERT_EQ( column->GetInt32( 2 ), 2 );
    ASSERT_EQ( column->GetInt32( 4 ), 2 );
    ASSERT_EQ( column->GetInt32( 6 ), 2 );
    ASSERT_EQ( column->GetInt32( 7 ), 3 );
    ASSERT_EQ( column->GetInt32( 9 ), 3 );
    ASSERT_EQ( column->GetInt32( 11 ), 3 );
    ASSERT_EQ( column->GetInt32( 12 ), 4 );

    // verify column r_name
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( column->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( column->GetString( 2 ), "ASIA" );
    ASSERT_EQ( column->GetString( 4 ), "ASIA" );
    ASSERT_EQ( column->GetString( 6 ), "ASIA" );
    ASSERT_EQ( column->GetString( 7 ), "EUROPE" );
    ASSERT_EQ( column->GetString( 9 ), "EUROPE" );
    ASSERT_EQ( column->GetString( 11 ), "EUROPE" );
    ASSERT_EQ( column->GetString( 12 ), "MIDDLE EAST" );

    // verify column n_nationkey
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 8 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 12 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 21 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 6 );
    ASSERT_EQ( column->GetNullableInt32( 9 ).value, 19 );
    ASSERT_EQ( column->GetNullableInt32( 11 ).value, 23 );
    ASSERT_EQ( column->GetNullableInt32( 12 ).flag, 0 );

    // verify column n_name
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 1 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 2 ), "INDIA" );
    ASSERT_EQ( column->GetNullableString( 4 ), "JAPAN" );
    ASSERT_EQ( column->GetNullableString( 6 ), "VIETNAM" );
    ASSERT_EQ( column->GetNullableString( 7 ), "FRANCE" );
    ASSERT_EQ( column->GetNullableString( 9 ), "ROMANIA" );
    ASSERT_EQ( column->GetNullableString( 11 ), "UNITED KINGDOM" );
    ASSERT_EQ( column->GetNullableString( 12 ), "NULL" );

    // verify column n_regionkey
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).value, 3 );
    ASSERT_EQ( column->GetNullableInt32( 9 ).value, 3 );
    ASSERT_EQ( column->GetNullableInt32( 11 ).value, 3 );
    ASSERT_EQ( column->GetNullableInt32( 12 ).flag, 0 );
}
TEST(smoke_1, left_join_dyn_code_calc_compare)
{
    auto table = TestBackdoorV2( "smoke_test_queries", "left_join_dyn_code_calc_compare" );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 7 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 9 );

    // verify column r_regionkey
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 0 );
    ASSERT_EQ( column->GetInt32( 1 ), 1 );
    ASSERT_EQ( column->GetInt32( 2 ), 2 );
    ASSERT_EQ( column->GetInt32( 3 ), 2 );
    ASSERT_EQ( column->GetInt32( 4 ), 2 );
    ASSERT_EQ( column->GetInt32( 5 ), 2 );
    ASSERT_EQ( column->GetInt32( 6 ), 2 );
    ASSERT_EQ( column->GetInt32( 7 ), 3 );
    ASSERT_EQ( column->GetInt32( 8 ), 4 );

    // verify column r_name
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "AFRICA" );
    ASSERT_EQ( column->GetString( 1 ), "AMERICA" );
    ASSERT_EQ( column->GetString( 2 ), "ASIA" );
    ASSERT_EQ( column->GetString( 3 ), "ASIA" );
    ASSERT_EQ( column->GetString( 4 ), "ASIA" );
    ASSERT_EQ( column->GetString( 5 ), "ASIA" );
    ASSERT_EQ( column->GetString( 6 ), "ASIA" );
    ASSERT_EQ( column->GetString( 7 ), "EUROPE" );
    ASSERT_EQ( column->GetString( 8 ), "MIDDLE EAST" );

    // verify column n_nationkey
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 8 );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 9 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 12 );
    ASSERT_EQ( column->GetNullableInt32( 5 ).value, 18 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 21 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 8 ).flag, 0 );

    // verify column n_name
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 1 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 2 ), "INDIA" );
    ASSERT_EQ( column->GetNullableString( 3 ), "INDONESIA" );
    ASSERT_EQ( column->GetNullableString( 4 ), "JAPAN" );
    ASSERT_EQ( column->GetNullableString( 5 ), "CHINA" );
    ASSERT_EQ( column->GetNullableString( 6 ), "VIETNAM" );
    ASSERT_EQ( column->GetNullableString( 7 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 8 ), "NULL" );

    // verify column n_regionkey
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 4 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 5 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 6 ).value, 2 );
    ASSERT_EQ( column->GetNullableInt32( 7 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 8 ).flag, 0 );
}

static const string TEST_DB( "smoke" );
TEST(smoke_1, left_hash_join_single_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 int primary key, f2 int unique key);";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 int , f2 int);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // hash join using primary key
    auto joinPriLeftAsHash = "select * from t1 left join t2 on t1.f1 = t2.f1 order by t1.f1;";
    auto joinPriRightAsHash = "select * from t2 left join t1 on t1.f1 = t2.f1 order by t2.f1;";

    auto joinUniqLeftAsHash = "select * from t1 left join t2 on t1.f2 = t2.f2 order by t1.f1;";
    auto joinUniqRightAsHash = "select * from t2 left join t1 on t1.f2 = t2.f2 order by t2.f1;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values(1,11),"
                               "(2,22);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values(3,33),"
                               "(4,44);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 11 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 22 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).flag, 0 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).flag, 0 );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ), 44 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    // unique key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 11 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 22 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).flag, 0 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).flag, 0 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).flag, 0 );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ), 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ), 44 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );
}

TEST(smoke_1, left_hash_join_multi_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 int, f2 int, f3 int, f4 int, primary key(f1, f2), unique key( f3, f4) );";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 int , f2 int);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    auto joinPriLeftAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t1 left join t2 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t1.f1, t1.f2;";
    auto joinPriRightAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t2 left join t1 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t2.f1, t2.f2;";

    auto joinUniqLeftAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t1 left join t2 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t1.f3, t1.f4;";
    auto joinUniqRightAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t2 left join t1 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t2.f1, t2.f2;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values(1, 11, 111, 1111),"
                               "(1, 12, 111, 1112);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values(3,33),"
                               "(4,44);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // multi primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 11 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 12 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 44 );

    // multi unique key
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 111 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 111 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 1111 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ), 1112 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 44 );
}

TEST(smoke_1, right_hash_join_single_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 int primary key, f2 int unique key);";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 int , f2 int);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // hash join using primary key
    auto joinPriLeftAsHash = "select * from t1 right join t2 on t1.f1 = t2.f1 order by t2.f1;";
    auto joinPriRightAsHash = "select * from t2 right join t1 on t1.f1 = t2.f1 order by t1.f1;";

    auto joinUniqLeftAsHash = "select * from t1 right join t2 on t1.f2 = t2.f2 order by t2.f1;";
    auto joinUniqRightAsHash = "select * from t2 right join t1 on t1.f2 = t2.f2 order by t1.f1;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values(1,11),"
                               "(2,22);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values(3,33),"
                               "(4,44);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 44 );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );
    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 11 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 22 );

    // unique key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 33 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 44 );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );
    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 11 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 22 );
}

TEST(smoke_1, right_hash_join_multi_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 int, f2 int, f3 int, f4 int, primary key(f1, f2), unique key( f3, f4) );";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 int , f2 int);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    auto joinPriLeftAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t1 right join t2 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t2.f1, t2.f2;";
    auto joinPriRightAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t2 right join t1 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t1.f1, t1.f2;";

    auto joinUniqLeftAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t1 right join t2 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t2.f1, t2.f2;";
    auto joinUniqRightAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t2 right join t1 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t1.f3, t1.f4;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values(1, 11, 111, 1111),"
                               "(1, 12, 111, 1112);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values(3,33),"
                               "(4,44);";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // multi primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 44 );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 1 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 11 );
    ASSERT_EQ( columnBuff->GetInt32( 1 ), 12 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    // multi unique key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 4 );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 33 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 44 );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 111 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 111 );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), false );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), false );
    ASSERT_EQ( columnBuff->GetNullableInt32( 0 ).value, 1111 );
    ASSERT_EQ( columnBuff->GetNullableInt32( 1 ).value, 1112 );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isInt32DataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isInt32DataNull( 1 ), true );
}

TEST(smoke_1, char_left_hash_join_single_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 char(40) primary key, f2 char(40) unique key);";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 char(40) , f2 char(40));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // hash join using primary key
    auto joinPriLeftAsHash = "select * from t1 left join t2 on t1.f1 = t2.f1 order by t1.f1;";
    auto joinPriRightAsHash = "select * from t2 left join t1 on t1.f1 = t2.f1 order by t2.f1;";

    auto joinUniqLeftAsHash = "select * from t1 left join t2 on t1.f2 = t2.f2 order by t1.f1;";
    auto joinUniqRightAsHash = "select * from t2 left join t1 on t1.f2 = t2.f2 order by t2.f1;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values('1', '11'),"
                               "('2', '22');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values('3', '33'),"
                               "('4', '44');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "2" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "11" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "22" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );
    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "3" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "4" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "33" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "44" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    // unique key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "2" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "11" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "22" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );
    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "3" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "4" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "33" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "44" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );
}

TEST(smoke_1, char_left_hash_join_multi_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 char(40), f2 char(40), f3 char(40), f4 char(40), primary key(f1, f2), unique key( f3, f4) );";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 char(40) , f2 char(40));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    auto joinPriLeftAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t1 left join t2 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t1.f1, t1.f2;";
    auto joinPriRightAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t2 left join t1 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t1.f1, t1.f2;";

    auto joinUniqLeftAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t1 left join t2 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t1.f3, t1.f4;";
    auto joinUniqRightAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t2 left join t1 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t1.f3, t1.f4;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values('1', '11', '111', '1111'),"
                               "('1', '12', '111', '1112');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values('3', '33'),"
                               "('4', '44');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // multi primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "1" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "11" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "12" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "3" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "4" );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "33" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "44" );

    // multi unique key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "111" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "111" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1111" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "1112" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "3" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "4" );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "33" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "44" );
}

TEST(smoke_1, char_right_hash_join_single_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 char(40) primary key, f2 char(40) unique key);";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 char(40) , f2 char(40));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // hash join using primary key
    auto joinPriLeftAsHash = "select * from t1 right join t2 on t1.f1 = t2.f1 order by t2.f1;";
    auto joinPriRightAsHash = "select * from t2 right join t1 on t1.f1 = t2.f1 order by t1.f1;";

    auto joinUniqLeftAsHash = "select * from t1 right join t2 on t1.f2 = t2.f2 order by t2.f1;";
    auto joinUniqRightAsHash = "select * from t2 right join t1 on t1.f2 = t2.f2 order by t1.f1;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values('1', '11'),"
                               "('2', '22');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values('3', '33'),"
                               "('4', '44');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );
    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "3" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "4" );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "33" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "44" );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );
    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "2" );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "11" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "22" );
}

TEST(smoke_1, char_right_hash_join_multi_column_inner_result_0)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1(f1 char(40), f2 char(40), f3 char(40), f4 char(40), primary key(f1, f2), unique key( f3, f4) );";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2(f1 char(40) , f2 char(40));";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    auto joinPriLeftAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t1 right join t2 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t1.f1, t1.f2;";
    auto joinPriRightAsHash = "select t1.f1, t1.f2, t2.f1, t2.f2 from t2 right join t1 on t1.f1 = t2.f1 and t1.f2 = t2.f2 order by t1.f1, t1.f2;";

    auto joinUniqLeftAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t1 right join t2 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t1.f3, t1.f4;";
    auto joinUniqRightAsHash = "select t1.f3, t1.f4, t2.f1, t2.f2 from t2 right join t1 on t1.f3 = t2.f1 and t1.f4 = t2.f2 order by t1.f3, t1.f4;";

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    result = SQLExecutor::GetInstance()->ExecuteSQL( joinUniqRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 0 );

    sql = "insert into t1 values('1', '11', '111', '1111'),"
                               "('1', '12', '111', '1112');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values('3', '33'),"
                               "('4', '44');";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    // multi primary key
    // left as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriLeftAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "3" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "4" );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "33" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "44" );

    // right as hash
    result = SQLExecutor::GetInstance()->ExecuteSQL( joinPriRightAsHash, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );
    tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 2 );

    columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "1" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "1" );

    columnBuff = tableBlock->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetString( 0 ), "11" );
    ASSERT_EQ( columnBuff->GetString( 1 ), "12" );

    columnBuff = tableBlock->GetColumnBuffer( 3 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );

    columnBuff = tableBlock->GetColumnBuffer( 4 );
    ASSERT_EQ( columnBuff->isStringDataNull( 0 ), true );
    ASSERT_EQ( columnBuff->isStringDataNull( 1 ), true );
}


TEST(smoke_1, char_dict_encode_after_group_leftjoin)
{
    InitTable( TEST_DB, "t1" );
    InitTable( TEST_DB, "t2" );

    auto sql = "create table t1 ( f1 char(40) encoding bytedict as dict_join_test_1, f2 int );";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "create table t2 ( f1 char(40) encoding bytedict as dict_join_test_1, f2 int );";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t1 values ( 'a', 1 );";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = "insert into t2 values ( 'b', 2 );";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    sql = R"(select count(*)
from
    ( select f1, f2 from t1  group by f1, f2 ) a
left join
    ( select f1,f2 from t2 group by f1, f2 ) b
on a.f1 = b.f1 and a.f2 = b.f2;)";
    result = SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB );
    EXPECT_TRUE( result->IsSuccess() );

    auto tableBlock = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( tableBlock->GetRowCount(), 1 );

    auto columnBuff = tableBlock->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32AsString( 0 ), "1" );
}