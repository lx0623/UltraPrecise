#include <gtest/gtest.h>
#include <string>

#include "AriesEngineWrapper/AriesMemTable.h"
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"
#include "server/mysql/include/mysqld.h"
#include "server/mysql/include/sql_class.h"

using namespace aries_engine;
using namespace std;
using std::string;

extern string DB_NAME;

AriesTableBlockUPtr TestTpchWithNull( int arg_query_number )
{
    std::cout << "-------------------------------------------------------------------------------------------------------" << arg_query_number
            << "\n\n\n";
    std::string db_name = "scale_1_null";
    if ( !DB_NAME.empty() )
        db_name = DB_NAME;
    std::string filename = "test_tpch_queries/" + std::to_string( arg_query_number ) + ".sql";
    cout << "using database " << db_name << endl;
    current_thd->set_db( db_name );
    //std::string filename = "/home/lichi/rateup/aries/test_tpch_queries/" + std::to_string( arg_query_number ) + ".sql";
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

TEST(scale1_null, q1)
{
    auto table = TestTpchWithNull( 1 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 10 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 9 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 1 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 2 ), "A" );
    ASSERT_EQ( column->GetNullableString( 3 ), "A" );
    ASSERT_EQ( column->GetNullableString( 4 ), "N" );
    ASSERT_EQ( column->GetNullableString( 5 ), "N" );
    ASSERT_EQ( column->GetNullableString( 6 ), "N" );
    ASSERT_EQ( column->GetNullableString( 7 ), "R" );
    ASSERT_EQ( column->GetNullableString( 8 ), "R" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "F" );
    ASSERT_EQ( column->GetNullableString( 1 ), "O" );
    ASSERT_EQ( column->GetNullableString( 2 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 3 ), "F" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "3822019.00" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "3641672.00" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "1893110.00" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "30172151.00" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "5734116583.04" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "5463539510.91" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "2838153295.88" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "45242566643.30" );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "5447146329.7225" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "5191159476.5861" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "2697174259.3672" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "40296298202.0537" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "5664552512.942016" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "5399106801.561359" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "2805179325.080777" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "39117646363.561432" );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    cout << column->GetDecimalAsString( 0 ) << endl;
    cout << column->GetDecimalAsString( 1 ) << endl;
    cout << column->GetDecimalAsString( 2 ) << endl;
    cout << column->GetDecimalAsString( 3 ) << endl;
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "25.516190" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "25.542329" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "25.553561" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "25.516057" );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "25.5220058532573370" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "25.5164719205229835" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "25.5023142267219321" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "25.5057936126907707" ) );

// verify column 8
    column = table->GetColumnBuffer( 8 );
    cout << column->GetDecimalAsString( 0 ) << endl;
    cout << column->GetDecimalAsString( 1 ) << endl;
    cout << column->GetDecimalAsString( 2 ) << endl;
    cout << column->GetDecimalAsString( 3 ) << endl;
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "38281.548475" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "38320.728260" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "38309.935963" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "38270.293265" );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "38273.129734621672" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "38284.467760848304" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "38249.255055597586" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "38250.854626099657" ) );

// verify column 9
    column = table->GetColumnBuffer( 9 );
    cout << column->GetDecimalAsString( 0 ) << endl;
    cout << column->GetDecimalAsString( 1 ) << endl;
    cout << column->GetDecimalAsString( 2 ) << endl;
    cout << column->GetDecimalAsString( 3 ) << endl;
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.050070" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "0.049925" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "0.049846" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "0.049987" );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "0.04998529583839761162" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "0.05009342667421629691" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "0.04999909817093625999" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "0.05000940583012705647" ) );
    // verify column 10
    column = table->GetColumnBuffer( 10 );
    ASSERT_EQ( column->GetInt64( 0 ), 149788 );
    ASSERT_EQ( column->GetInt64( 1 ), 142574 );
    ASSERT_EQ( column->GetInt64( 2 ), 74084 );
    ASSERT_EQ( column->GetInt64( 3 ), 1256283 );
}

TEST(scale1_null, q2)
{
    auto table = TestTpchWithNull( 2 );
    int64_t columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 8 );
    int64_t tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2149 );

    //FIXME no order by. can't compare by index
}

TEST(scale1_null, q3)
{
    auto table = TestTpchWithNull( 3 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 10157 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 4613219 );
    ASSERT_EQ( column->GetInt32( 6264 ), 1889317 );
    ASSERT_EQ( column->GetInt32( 9776 ), 1112002 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "388942.5848" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 6264 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "53274.8496" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 9776 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "967.8952" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 9777 ).flag, 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableDateAsString( 0 ), "1995-03-10" );
    ASSERT_EQ( column->GetNullableDateAsString( 6264 ), "1994-12-09" );
    ASSERT_EQ( column->GetNullableDateAsString( 9776 ), "1994-12-11" );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ), 0 );
    ASSERT_EQ( column->GetNullableInt32( 6264 ).flag, 0 );
    ASSERT_EQ( column->GetNullableInt32( 9776 ), 0 );
}

TEST(scale1_null, q4)
{
    auto table = TestTpchWithNull( 4 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 6 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 5 ), "5-LOW" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 2513 );
    ASSERT_EQ( column->GetInt64( 5 ), 9164 );
}

TEST(scale1_null, q5)
{
    auto table = TestTpchWithNull( 5 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableString( 0 ), "ALGERIA" );
    ASSERT_EQ( column->GetNullableString( 1 ), "MOZAMBIQUE" );
    ASSERT_EQ( column->GetNullableString( 2 ), "ETHIOPIA" );
    ASSERT_EQ( column->GetNullableString( 3 ), "KENYA" );
    ASSERT_EQ( column->GetNullableString( 4 ), "MOROCCO" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
//    cout << column->GetDecimalAsString( 0 ) << endl;
//    cout << column->GetDecimalAsString( 1 ) << endl;
//    cout << column->GetDecimalAsString( 2 ) << endl;
//    cout << column->GetDecimalAsString( 3 ) << endl;
//    cout << column->GetDecimalAsString( 4 ) << endl;
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "46884192.4084" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 1 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "46537924.8319" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 2 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "44945883.3856" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 3 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "42118006.4471" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 4 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "40714760.2182" ) ) );
}

TEST(scale1_null, q6)
{
    auto table = TestTpchWithNull( 6 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "89690407.2491" ) ) );
}

TEST(scale1_null, q7)
{
    auto table = TestTpchWithNull( 7 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 4 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableString( 0 ), "IRAQ" );
    ASSERT_EQ( column->GetNullableString( 1 ), "IRAQ" );
    ASSERT_EQ( column->GetNullableString( 2 ), "KENYA" );
    ASSERT_EQ( column->GetNullableString( 3 ), "KENYA" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "KENYA" );
    ASSERT_EQ( column->GetNullableString( 1 ), "KENYA" );
    ASSERT_EQ( column->GetNullableString( 2 ), "IRAQ" );
    ASSERT_EQ( column->GetNullableString( 3 ), "IRAQ" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1995 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 1996 );
    ASSERT_EQ( column->GetNullableInt32( 2 ).value, 1995 );
    ASSERT_EQ( column->GetNullableInt32( 3 ).value, 1996 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "44798500.9932" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 1 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "49140174.2027" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 2 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "38936353.2984" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 3 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "39935368.7557" ) ) );
}

TEST(scale1_null, q8)
{
    auto table = TestTpchWithNull( 8 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1995 );
    ASSERT_EQ( column->GetNullableInt32( 1 ).value, 1996 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.04468643" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "0.03626766" );
}

TEST(scale1_null, q9)
{
    auto table = TestTpchWithNull( 9 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 200 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 191 ), "UNITED STATES" );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 1998 );
    ASSERT_EQ( column->GetNullableInt32( 191 ).flag, 0 );

    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "18986302.2249" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 191 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "11044877.1207" ) ) );
}

TEST(scale1_null, q10)
{
    auto table = TestTpchWithNull( 10 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 8 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 33711 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 51811 );
    ASSERT_EQ( column->GetInt32( 81 ), 110839 );
    ASSERT_EQ( column->GetInt32( 32267 ), 5348 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableString( 0 ), "Customer#000051811" );
    ASSERT_EQ( column->GetNullableString( 81 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 32267 ), "Customer#000005348" );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "707782.0057" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 81 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "377874.9346" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 32267 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "889.6768" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 32268 ).flag, 0 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "1016.15" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 81 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "-237.25" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 32267 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "-86.67" ) ) );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableString( 0 ), "NULL" );
    ASSERT_EQ( column->GetNullableString( 81 ), "ETHIOPIA" );
    ASSERT_EQ( column->GetNullableString( 32267 ), "BRAZIL" );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetNullableString( 0 ), " iahxUHfAxlJkZWz4iPGDmAQJSWl UlrqXRW" );
    ASSERT_EQ( column->GetNullableString( 81 ), "GzvLu1IfKgmYN4CiEaysVErpg" );
    ASSERT_EQ( column->GetNullableString( 32267 ), "FUyt365fT3kK00AOQ72BPJ,oSK4s85 mr4G0t1G" );

    // verify column 7
    column = table->GetColumnBuffer( 7 );
    ASSERT_EQ( column->GetNullableString( 0 ), "28-777-205-1675" );
    ASSERT_EQ( column->GetNullableString( 81 ), "15-612-195-4742" );
    ASSERT_EQ( column->GetNullableString( 32267 ), "12-835-192-6690" );

    // verify column 8
    column = table->GetColumnBuffer( 8 );
    ASSERT_EQ( column->GetNullableString( 0 ),
            "ajole slyly fluffily even asymptotes. packages sleep furiously. blithely pending packages cajole: express " );
    ASSERT_EQ( column->GetNullableString( 81 ),
            "uffily carefully bold accounts. special, regular requests dazzle furiously blithely special accounts" );
    ASSERT_EQ( column->GetNullableString( 32267 ),
            "are furiously across the slyly close requests. furiously special accounts wake carefully alongside of the blithel" );
}

TEST(scale1_null, q11)
{
    auto table = TestTpchWithNull( 11 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1267 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 18488 );
    ASSERT_EQ( column->GetInt32( 999 ), 48453 );
    ASSERT_EQ( column->GetInt32( 1266 ), 34004 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "19174541.94" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 999 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "7791279.70" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 1266 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "7442517.10" ) ) );
}

TEST(scale1_null, q12)
{
    auto table = TestTpchWithNull( 12 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 2 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "AIR" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 1 ).c_str(), "TRUCK" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 4675 );
    ASSERT_EQ( column->GetNullableInt64( 1 ).value, 4619 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 7157 );
    ASSERT_EQ( column->GetNullableInt64( 1 ).value, 7006 );
}

TEST(scale1_null, q13)
{
    auto table = TestTpchWithNull( 13 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 41 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt64( 0 ), 0 );
    ASSERT_EQ( column->GetInt64( 40 ), 40 );

    // verify column 3
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 50006 );
    ASSERT_EQ( column->GetInt64( 40 ), 1 );
}

TEST(scale1_null, q14)
{
    auto table = TestTpchWithNull( 14 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "15.9874074789" );
}

TEST(scale1_null, q15)
{
    auto table = TestTpchWithNull( 15 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 984 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "Supplier#000000984" ), 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "6H6qqye iYbYzCmwWhj" ), 0 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "31-519-879-5266" ), 0 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "1754584.8300" ) ) );
}

TEST(scale1_null, q16)
{
    auto table = TestTpchWithNull( 16 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 16557 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "Brand#41" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 16556 ).c_str(), "Brand#55" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "MEDIUM BRUSHED TIN" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 16556 ).c_str(), "STANDARD PLATED TIN" ), 0 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 3 );
    ASSERT_EQ( column->GetNullableInt32( 16556 ).value, 49 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt64( 0 ), 28 );
    ASSERT_EQ( column->GetInt64( 16556 ), 3 );
}

TEST(scale1_null, q17)
{
    auto table = TestTpchWithNull( 17 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );

    ASSERT_EQ( column->GetDecimalAsString( 0 ), "278884.315714" );
}

TEST(scale1_null, q18)
{
    auto table = TestTpchWithNull( 18 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 6 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 6 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "Customer#000128120" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 3 ).c_str(), "NULL" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt32( 0 ), 128120 );
    ASSERT_EQ( column->GetInt32( 3 ), 15619 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetInt32( 0 ), 4722021 );
    ASSERT_EQ( column->GetInt32( 3 ), 3767271 );

    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( strcmp( column->GetNullableDateAsString( 0 ).c_str(), "1994-04-07" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableDateAsString( 3 ).c_str(), "1996-08-07" ), 0 );

    // verify column 5
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "544089.09" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 3 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "480083.96" ) ) );

    // verify column 6
    column = table->GetColumnBuffer( 6 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "323.00" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 3 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "318.00" ) ) );
}

TEST(scale1_null, q19)
{
    auto table = TestTpchWithNull( 19 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "2596103.8967" ) ) );
}

TEST(scale1_null, q20)
{
    auto table = TestTpchWithNull( 20 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 172 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetNullableString( 8 ).c_str(), "Supplier#000000024" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 165 ).c_str(), "Supplier#000009518" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( strcmp( column->GetNullableString( 8 ).c_str(), "C4nPvLrVmKPPabFCj" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 165 ).c_str(), "NULL" ), 0 );
}

TEST(scale1_null, q21)
{
    auto table = TestTpchWithNull( 21 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 392 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "NULL" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 391 ).c_str(), "Supplier#000007847" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 228 );
    ASSERT_EQ( column->GetInt64( 391 ), 4 );
}

TEST(scale1_null, q22)
{
    auto table = TestTpchWithNull( 22 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 7 );

    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( strcmp( column->GetNullableString( 0 ).c_str(), "10" ), 0 );
    ASSERT_EQ( strcmp( column->GetNullableString( 6 ).c_str(), "34" ), 0 );

    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetInt64( 0 ), 779 );
    ASSERT_EQ( column->GetInt64( 6 ), 862 );

    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "5842070.85" ) ) );
    ASSERT_EQ( column->GetNullableDecimal( 6 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "6536145.02" ) ) );
}

//TEST(scale1_null, q0)
//{
//    auto table = TestTpchWithNull( 0 );
//    int columnCount = table->GetColumnCount();
//    ASSERT_EQ( columnCount, 1 );
//    int tupleNum = table->GetRowCount();
//    ASSERT_EQ( tupleNum, 1 );
//    // verify column 1
//    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
//
//    ASSERT_EQ( column->GetNullableDecimal( 0 ), aries_acc::nullable_type< aries_acc::Decimal >( 1, aries_acc::Decimal( "16.7286491435" ) ) );
//}
//
//TEST(scale1_null, q23)
//{
//    auto table = TestTpchWithNull( 23 );
//    int columnCount = table->GetColumnCount();
//    ASSERT_EQ( columnCount, 10 );
//    int tupleNum = table->GetRowCount();
//    ASSERT_EQ( tupleNum, 4 );
//
//    // verify column 1
//    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
//    ASSERT_EQ( column->GetString( 0 ), "A" );
//    ASSERT_EQ( column->GetString( 1 ), "N" );
//    ASSERT_EQ( column->GetString( 2 ), "N" );
//    ASSERT_EQ( column->GetString( 3 ), "R" );
//
//    // verify column 2
//    column = table->GetColumnBuffer( 2 );
//    ASSERT_EQ( column->GetString( 0 ), "F" );
//    ASSERT_EQ( column->GetString( 1 ), "F" );
//    ASSERT_EQ( column->GetString( 2 ), "O" );
//    ASSERT_EQ( column->GetString( 3 ), "F" );
//
//    // verify column 3
//    column = table->GetColumnBuffer( 3 );
//
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "37734107.0" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "991417.0" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "72675577.0" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "37719753.0" ) );
//
//    // verify column 4
//    column = table->GetColumnBuffer( 4 );
//
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "56586554400.73" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "1487504710.38" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "109001350084.26" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "56568041380.90" ) );
//
//    // verify column 5
//    column = table->GetColumnBuffer( 5 );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "53758257134.8700" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "1413082168.0541" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "103552520554.2534" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "53741292684.6040" ) );
//
//    // verify column 6
//    column = table->GetColumnBuffer( 6 );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "55909065222.827692" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "1469649223.194375" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "107698448674.818097" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "55889619119.831932" ) );
//
//    // verify column 7
//    column = table->GetColumnBuffer( 7 );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "25.522005" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "25.516471" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "25.502314" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "25.505793" ) );
////    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "25.5220058532573370" ) );
////    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "25.5164719205229835" ) );
////    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "25.5023142267219321" ) );
////    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "25.5057936126907707" ) );
//
//    // verify column 8
//    column = table->GetColumnBuffer( 8 );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "38273.129734" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "38284.467760" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "38249.255055" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "38250.854626" ) );
////    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "38273.129734621672" ) );
////    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "38284.467760848304" ) );
////    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "38249.255055597586" ) );
////    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "38250.854626099657" ) );
//
//    // verify column 9
//    column = table->GetColumnBuffer( 9 );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "0.049985" ) );
//    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "0.050093" ) );
//    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "0.049999" ) );
//    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "0.050009" ) );
////    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "0.04998529583839761162" ) );
////    ASSERT_EQ( column->GetDecimal( 1 ), aries_acc::Decimal( "0.05009342667421629691" ) );
////    ASSERT_EQ( column->GetDecimal( 2 ), aries_acc::Decimal( "0.04999909817093625999" ) );
////    ASSERT_EQ( column->GetDecimal( 3 ), aries_acc::Decimal( "0.05000940583012705647" ) );
//    // verify column 10
//    column = table->GetColumnBuffer( 10 );
//    ASSERT_EQ( column->GetInt64( 0 ), 1478493 );
//    ASSERT_EQ( column->GetInt64( 1 ), 38854 );
//    ASSERT_EQ( column->GetInt64( 2 ), 2849764 );
//    ASSERT_EQ( column->GetInt64( 3 ), 1478870 );
//}

TEST(scale1_null, q24)
{
    auto table = TestTpchWithNull( 24 );
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 1 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "0.000490" );
}

//TEST(scale1_null, q25)
//{
//    auto table = TestTpchWithNull( 25 );
//    int columnCount = table->GetColumnCount();
//    ASSERT_EQ( columnCount, 4 );
//    int tupleNum = table->GetRowCount();
//    ASSERT_EQ( tupleNum, 165120 );
//
//    // verify column 1
//    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
//    ASSERT_EQ( column->GetInt32( 0 ), 4576548 );
//    ASSERT_EQ( column->GetInt32( 6269 ), 3420518 );
//    ASSERT_EQ( column->GetInt32( 11438 ), 5761507 );
//
//    // verify column 2
//    column = table->GetColumnBuffer( 2 );
//    ASSERT_EQ( column->GetDecimal( 0 ), aries_acc::Decimal( "504144.3558" ) );
//    ASSERT_EQ( column->GetDecimal( 6269 ), aries_acc::Decimal( "303099.1197" ) );
//    ASSERT_EQ( column->GetDecimal( 11438 ), aries_acc::Decimal( "278187.3300" ) );
//
//    // verify column 3
//    column = table->GetColumnBuffer( 3 );
//    ASSERT_EQ( column->GetDateAsString( 0 ), "1997-12-26" );
//    ASSERT_EQ( column->GetDateAsString( 6269 ), "1997-07-26" );
//    ASSERT_EQ( column->GetDateAsString( 11438 ), "1996-07-31" );
//
//    // verify column 4
//    column = table->GetColumnBuffer( 4 );
//    ASSERT_EQ( column->GetInt32( 0 ), 0 );
//    ASSERT_EQ( column->GetInt32( 6269 ), 0 );
//    ASSERT_EQ( column->GetInt32( 11438 ), 0 );
//}

//TEST(scale1_null, q26)
//{
//    auto table = TestTpchWithNull( 26 );
//}

TEST(scale1_null, q29)
{
    auto table = TestTpchWithNull( 29 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 3 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1674 );
}

TEST(scale1_null, q30)
{
    auto table = TestTpchWithNull( 30 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 59822 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 3 );
    ASSERT_EQ( column->GetInt32( 59821 ), 199998 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt32( 59821 ).value, 0 );
}

TEST(scale1_null, q31)
{
    auto table = TestTpchWithNull( 31 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 147602 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 5489475 );
    ASSERT_EQ( column->GetInt32( 147601 ), 4799525 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "466412.6229" );
    ASSERT_EQ( column->GetDecimalAsString( 147601 ), "NULL" );
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1997-05-23" );
    ASSERT_EQ( column->GetDateAsString( 147601 ), "1998-08-02" );
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt32( 147601 ).flag, 1 );
    ASSERT_EQ( column->GetNullableInt32( 147601 ).value, 0 );
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 352 );
    ASSERT_EQ( column->GetNullableInt64( 147601 ).value, 43 );
}

TEST(scale1_null, q33)
{
    auto table = TestTpchWithNull( 33 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "4247535.2209" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "4009004.6381" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "3918344.8494" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "4028546.8842" );
    ASSERT_EQ( column->GetDecimalAsString( 4 ), "4175639.1376" );
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1995-09-29" );
    ASSERT_EQ( column->GetDateAsString( 1 ), "1995-09-30" );
    ASSERT_EQ( column->GetDateAsString( 2 ), "1995-10-01" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "1995-10-02" );
    ASSERT_EQ( column->GetDateAsString( 4 ), "1995-10-03" );
}

TEST(scale1_null, q34)
{
    auto table = TestTpchWithNull( 34 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "4247535.2209" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "4009004.6381" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "3918344.8494" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "4028546.8842" );
    ASSERT_EQ( column->GetDecimalAsString( 4 ), "4175639.1376" );
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1995-09-29" );
    ASSERT_EQ( column->GetDateAsString( 1 ), "1995-09-30" );
    ASSERT_EQ( column->GetDateAsString( 2 ), "1995-10-01" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "1995-10-02" );
    ASSERT_EQ( column->GetDateAsString( 4 ), "1995-10-03" );
}

TEST(scale1_null, q35)
{
    auto table = TestTpchWithNull( 35 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 5 );
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "4247535.2209" );
    ASSERT_EQ( column->GetDecimalAsString( 1 ), "4009004.6381" );
    ASSERT_EQ( column->GetDecimalAsString( 2 ), "3918344.8494" );
    ASSERT_EQ( column->GetDecimalAsString( 3 ), "4028546.8842" );
    ASSERT_EQ( column->GetDecimalAsString( 4 ), "4175639.1376" );
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1995-09-29" );
    ASSERT_EQ( column->GetDateAsString( 1 ), "1995-09-30" );
    ASSERT_EQ( column->GetDateAsString( 2 ), "1995-10-01" );
    ASSERT_EQ( column->GetDateAsString( 3 ), "1995-10-02" );
    ASSERT_EQ( column->GetDateAsString( 4 ), "1995-10-03" );
}

TEST(scale1_null, q36)
{
    auto table = TestTpchWithNull(36);
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ(columnCount, 5);
    int tupleNum = table->GetRowCount();
    ASSERT_EQ(tupleNum, 1);

    AriesDataBufferSPtr column = table->GetColumnBuffer(1);
    ASSERT_EQ(column->GetInt32(0), 5489475);

    column = table->GetColumnBuffer(2);
    ASSERT_EQ(column->GetDecimalAsString(0), "466412.6229");
    column = table->GetColumnBuffer(3);
    ASSERT_EQ(column->GetDateAsString(0), "1997-05-23");
    column = table->GetColumnBuffer(4);
    ASSERT_EQ(column->GetNullableInt32(0).flag, 1);
    ASSERT_EQ(column->GetNullableInt32(0).value, 0);
    column = table->GetColumnBuffer(5);
    ASSERT_EQ(column->GetNullableInt64(0).value, 352);
}

TEST(scale1_null, q37)
{
    auto table = TestTpchWithNull( 37 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 4799525 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1998-08-02" );
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 0 );
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 43 );
}
TEST(scale1_null, q38)
{
    auto table = TestTpchWithNull( 38 );
    table->VerifyContent();
    ASSERT_EQ( table->GetColumnCount(), 4 );
    ASSERT_EQ( table->GetRowCount(), 0 );
}

TEST(scale1_null, q39)
{
    auto table = TestTpchWithNull( 39 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    auto rowCount = table->GetRowCount();
    ASSERT_EQ( rowCount, 0 );
}

TEST(scale1_null, q40)
{
    auto table = TestTpchWithNull( 40 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 4 );
    auto rowCount = table->GetRowCount();
    ASSERT_EQ( rowCount, 0 );
}

TEST(scale1_null, q41)
{
    auto table = TestTpchWithNull( 41 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 5 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 147602 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetInt32( 0 ), 2104992 );
    ASSERT_EQ( column->GetInt32( 147601 ), 5202049 );

    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "26645108.365079" );
    ASSERT_EQ( column->GetDecimalAsString( 147601 ), "NULL" );
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetDateAsString( 0 ), "1997-01-10" );
    ASSERT_EQ( column->GetDateAsString( 147601 ), "1998-08-02" );
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).flag, 1 );
    ASSERT_EQ( column->GetNullableInt32( 0 ).value, 0 );
    ASSERT_EQ( column->GetNullableInt32( 147601 ).flag, 0 );
    column = table->GetColumnBuffer( 5 );
    ASSERT_EQ( column->GetNullableInt64( 0 ).value, 537 );
    ASSERT_EQ( column->GetNullableInt64( 147601 ).value, 6 );
}

TEST(scale1_null, q42)
{
    auto table = TestTpchWithNull( 42 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1500000 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "4216.95" );
    ASSERT_EQ( column->GetDecimalAsString( 1500000 - 1 ), "9999.99" );
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "183299.15" );
    ASSERT_EQ( column->GetDecimalAsString( 1500000 - 1 ), "424918.30" );
}

TEST(scale1_null, q43)
{
    auto table = TestTpchWithNull( 43 );
    table->VerifyContent();
    int columnCount = table->GetColumnCount();
    ASSERT_EQ( columnCount, 2 );
    int tupleNum = table->GetRowCount();
    ASSERT_EQ( tupleNum, 1397196 );

    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "5240.73" );
    ASSERT_EQ( column->GetDecimalAsString( 1397196 - 1 ), "9999.99" );
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "77279.36" );
    ASSERT_EQ( column->GetDecimalAsString( 1397196 - 1 ), "424918.30" );
}

//TEST(scale1_null, q44)
//{
//    auto table = TestTpchWithNull( 44 );
//    table->VerifyContent();
//    int columnCount = table->GetColumnCount();
//    ASSERT_EQ( columnCount, 2 );
//    int tupleNum = table->GetRowCount();
//    ASSERT_EQ( tupleNum, 1550004 );
//
//    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
//    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
//    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "4581.98" );
//    ASSERT_EQ( column->GetDecimalAsString( 1342191 - 1 ), "9999.99" );
//        column = table->GetColumnBuffer( 2 );
//    ASSERT_EQ( column->GetDecimalAsString( 0 ), "857.71" );
//    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "158260.18" );
//    ASSERT_EQ( column->GetDecimalAsString( 1342191 - 1 ), "424918.30" );
//}
//
//TEST(scale1_null, q45)
//{
//    auto table = TestTpchWithNull( 45 );
//    table->VerifyContent();
//    int columnCount = table->GetColumnCount();
//    ASSERT_EQ( columnCount, 2 );
//    int tupleNum = table->GetRowCount();
//    ASSERT_EQ( tupleNum, 1550004 );
//
//    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
//    ASSERT_EQ( column->GetDecimalAsString( 0 ), "-999.99" );
//    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "4656.33" );
//    ASSERT_EQ( column->GetDecimalAsString( 1550004 - 1 ), "9999.99" );
//    column = table->GetColumnBuffer( 2 );
//    ASSERT_EQ( column->GetDecimalAsString( 0 ), "NULL" );
//    ASSERT_EQ( column->GetDecimalAsString( 800000 - 1 ), "22470.52" );
//    ASSERT_EQ( column->GetDecimalAsString( 1550004 - 1 ), "424918.30" );
//}
