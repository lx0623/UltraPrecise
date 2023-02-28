#include <gtest/gtest.h>
#include <string>

#include "AriesEngineWrapper/AriesMemTable.h"
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"

using namespace aries_engine;
using namespace std;
using std::string;

AriesTableBlockUPtr TestTpchWithWch( int arg_query_number )
{
    std::cout << "-------------------------------------------------------------------------------------------------------" << arg_query_number
            << "\n\n\n";
    std::string db_name = "scale_1_wch";
    std::string filename = "test_tpch_queries/wch_" + std::to_string( arg_query_number ) + ".sql";
    
    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQLFromFile( filename, db_name );

    if (results->IsSuccess() && results->GetResults().size() > 0) {
        auto amtp = results->GetResults()[0];
        auto art = std::move( ( ( ( AriesMemTable * )( amtp.get() ) )->GetContent() ) );
        int count = art->GetRowCount();
        cout << "tupleNum is:" << count << endl;
        return art;
    }
    return make_unique< AriesTableBlock >();
}

TEST(scale1_wch, q1)
{
    auto table = TestTpchWithWch( 1 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32(0), 18 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "量" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 2);

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetString( 0 ), "c dependencies. furiously express notornis sleep slyly regular accounts. ideas sleep. depos");
}

TEST(scale1_wch, q2)
{
    AriesTableBlockUPtr table = TestTpchWithWch( 2 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 7 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 3750000 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32(0), 9999 );
    ASSERT_EQ( column->GetInt32(1000000), 7339 );
    ASSERT_EQ( column->GetInt32(3750000 - 1), 19 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "量子象牛" );
    ASSERT_EQ( column->GetString( 1000000 ), "量子象牛" );
    ASSERT_EQ( column->GetString( 3750000 - 1 ), "量子象牛" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "mX37oAzqsBPhN1LWdzV p");
    ASSERT_EQ( column->GetString( 1000000 ), "N8lH6VcOyzGl,O7DogZA,VM008ORQcDdD9j4");
    ASSERT_EQ( column->GetString( 3750000 - 1 ), "edZT3es,nBFD8lBXTGeTl");

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt32(0), 19 );
    ASSERT_EQ( column->GetInt32( 1000000 ), 50019 );
    ASSERT_EQ( column->GetInt32( 3750000 - 1 ), 149999 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetString(0), "量子象牛" );
    ASSERT_EQ( column->GetString(1000000), "量子象牛" );
    ASSERT_EQ( column->GetString(3750000 - 1), "量子象牛" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetString(0), "uc,3bHIx84H,wdrmLOjVsiqXCq2tr" );
    ASSERT_EQ( column->GetString(1000000), "7in3vGDjnHWbtAsjF jLp3v0 wtLqW7QiHKxs" );
    ASSERT_EQ( column->GetString(3750000 - 1), "nBpZoYhCPFKZqSunxdeHtRN08x3RE8hqh" );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetString(0), " nag. furiously careful packages are slyly at the accounts. furiously regular in" );
    ASSERT_EQ( column->GetString(1000000), "ly about the quickly regular packages. slyly regular pains impress quickly slyly special theodolites. quiet theodol" );
    ASSERT_EQ( column->GetString(3750000 - 1), "s haggle about the final foxes. carefully special dependencies use carefully bold patterns-- final, regular instruct" );
}

