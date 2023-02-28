#include <gtest/gtest.h>

#include <thread>
#include <mutex>
#include <condition_variable>

#include "AriesEngineWrapper/AriesMemTable.h"
#include "frontend/SQLExecutor.h"
#include "TestUtils.h"

using namespace aries;
using namespace aries_engine;

extern THD *createPseudoThd();

namespace aries_test
{

class TestTPCHIsolation : public ::testing::Test
{
protected:
    static std::string db_name;
    int random_a;
    int random_b;

    aries_acc::Decimal old_totalprice;
    aries_acc::Decimal new_totalprice;

    int32_t O_KEY;
    int32_t L_KEY;
    int32_t DELTA;

    int M;

    std::string O_KEY_SQL;
    std::string M_SQL;

    std::mutex update_mutex;
    std::mutex query_mutex;

    std::condition_variable update_condition;
    std::condition_variable query_condition;

    bool update_should_commit = true;
    bool query_should_commit = true;

    bool update_ready_to_commit = true;
    bool query_ready_to_commit = true;

    bool update_committed_or_aborted = false;
    bool query_committed_or_aborted = false;

protected:
    void SetUp() override
    {

        ExecuteSQL( "CREATE TABLE HISTORY ( H_P_KEY int, H_S_KEY int, H_O_KEY int, H_L_KEY int, H_DELTA int, H_DATE_T datetime )", db_name );


        srand( time( NULL ) );
        random_a = rand();

        auto table = ExecuteSQL( "SELECT MAX(L_ORDERKEY) FROM LINEITEM;", db_name );
        EXPECT_EQ( table->GetRowCount(), 1 );
        EXPECT_EQ( table->GetColumnCount(), 1 );

        auto value = table->GetColumnBuffer( 1 )->GetNullableInt32( 0 );
        EXPECT_FALSE( value.is_null() );

        auto max = value.value;

        random_b = random_a % max;
        table = ExecuteSQL( "SELECT L_ORDERKEY AS O_KEY FROM LINEITEM WHERE L_ORDERKEY >= " + std::to_string( random_b ) + " LIMIT 1;", db_name );
        EXPECT_EQ( table->GetRowCount(), 1 );
        EXPECT_EQ( table->GetColumnCount(), 1 );

        O_KEY = table->GetColumnBuffer( 1 )->GetInt32( 0 );

        table = ExecuteSQL( "SELECT MAX(L_LINENUMBER) FROM LINEITEM WHERE L_ORDERKEY = " + std::to_string( O_KEY ), db_name );
        EXPECT_EQ( table->GetRowCount(), 1 );
        EXPECT_EQ( table->GetColumnCount(), 1 );

        value = table->GetColumnBuffer( 1 )->GetNullableInt32( 0 );
        EXPECT_FALSE( value.is_null() );
        M = value.value;
    
        L_KEY = rand() % M + 1;
        DELTA = rand() % 100 + 1;

        table = ExecuteSQL( "SELECT SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT) * (1 + L_TAX)) FROM LINEITEM WHERE L_ORDERKEY = "
                    + std::to_string( O_KEY ) + ";", db_name );
        auto v = table->GetColumnBuffer( 1 )->GetNullableDecimal( 0 );
        old_totalprice = v.value;
        {
            char tmp[ 64 ];
            std::cout << "old_totalprice: " << old_totalprice.GetDecimal( tmp ) << std::endl;
        }

        std::cout << "M: " << M << ", O_KEY: " << O_KEY << ", L_KEY: " << L_KEY << ", DELTA: " << DELTA << std::endl;

    }

    void TearDown() override
    {
        ExecuteSQL( "DROP TABLE HISTORY", db_name );
    }

    static void UpdateThreadWorker( TestTPCHIsolation* instance )
    {
        instance->UpdateTransaction();
    }

    static void QueryThreadWorker( TestTPCHIsolation* instance )
    {
        instance->QueryTransaction();
    }

    void UpdateTransaction()
    {
        createPseudoThd();
        ExecuteSQL( "start transaction;", db_name );

        auto table = ExecuteSQL( "SELECT O_TOTALPRICE FROM ORDERS WHERE O_ORDERKEY = " + std::to_string( O_KEY ) + ";", db_name );
        EXPECT_EQ( table->GetRowCount(), 1 );
        auto ototal = table->GetColumnBuffer( 1 )->GetDecimal( 0 );

        table = ExecuteSQL( "SELECT L_QUANTITY, L_EXTENDEDPRICE, L_PARTKEY, L_SUPPKEY, L_TAX, L_DISCOUNT FROM LINEITEM WHERE L_ORDERKEY = " + std::to_string( O_KEY ) + " AND L_LINENUMBER = " + std::to_string( L_KEY ), db_name );
        EXPECT_EQ( table->GetRowCount(), 1 );
        auto quantity = table->GetColumnBuffer( 1 )->GetDecimal( 0 );
        auto extprice = table->GetColumnBuffer( 2 )->GetDecimal( 0 );
        auto pkey = table->GetColumnBuffer( 3 )->GetInt32( 0 );
        auto skey = table->GetColumnBuffer( 4 )->GetInt32( 0 );
        auto tax = table->GetColumnBuffer( 5 )->GetDecimal( 0 );
        auto disc = table->GetColumnBuffer( 6 )->GetDecimal( 0 );
        auto rprice = extprice / quantity;
        auto cost = rprice * DELTA;
        auto new_extprice = extprice + cost;
        auto new_quantity = quantity + DELTA;
        auto new_ototal = new_extprice * ( aries_acc::Decimal( "1.0" ) - disc );
        new_ototal = new_ototal * ( aries_acc::Decimal( "1.0" ) + tax );
        new_ototal = ototal + new_ototal;

        char tmp[ 64 ];
        std::string new_extprice_str( new_extprice.GetDecimal( tmp ) );
        std::string new_quantity_str( new_quantity.GetDecimal( tmp ) );
        std::string new_ototal_str( new_ototal.GetDecimal( tmp ) );
        auto set_content = "L_EXTENDEDPRICE = " + new_extprice_str + ", L_QUANTITY = " + new_quantity_str;

        ExecuteSQL( "UPDATE LINEITEM SET " + set_content + " WHERE L_ORDERKEY = " + std::to_string( O_KEY ) + " AND L_LINENUMBER = " + std::to_string( L_KEY ) + ";", db_name );

        set_content = "O_TOTALPRICE = " + new_ototal_str;
        ExecuteSQL( "UPDATE ORDERS SET " + set_content + " WHERE  O_ORDERKEY = " + std::to_string( O_KEY ), db_name );

        if ( !update_ready_to_commit )
        {
            std::unique_lock< std::mutex > lock( update_mutex );
            update_condition.wait( lock, [ = ]{ return update_ready_to_commit; } );
        }

        if ( update_should_commit )
        {
            ExecuteSQL( "commit;", db_name );
        }
        else
        {
            ExecuteSQL( "rollback;", db_name );
        }

        update_committed_or_aborted = true;
        update_condition.notify_all();
    }

    void QueryTransaction()
    {
        createPseudoThd();
        ExecuteSQL( "start transaction;", db_name );
    
        auto tb = ExecuteSQL( "SELECT SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT) * (1 + L_TAX)) FROM LINEITEM WHERE L_ORDERKEY = "
                    + std::to_string( O_KEY ) + ";", db_name );

        auto v = tb->GetColumnBuffer( 1 )->GetNullableDecimal( 0 );
        new_totalprice = v.value;

        if ( !query_ready_to_commit )
        {
            std::unique_lock< std::mutex > lock( query_mutex );
            query_condition.wait( lock, [ = ]{ return query_ready_to_commit; } );
        }

        if ( query_should_commit )
        {
            ExecuteSQL( "commit;", db_name );
        }
        else
        {
            ExecuteSQL( "rollback;", db_name );
        }

        query_committed_or_aborted = true;
        query_condition.notify_all();
    }

    void Reset()
    {
        update_should_commit = true;
        query_should_commit = true;

        update_ready_to_commit = false;
        query_ready_to_commit = false;

        update_committed_or_aborted = false;
        query_committed_or_aborted = false;
    }
};

std::string TestTPCHIsolation::db_name( "scale_1" );

TEST_F( TestTPCHIsolation, test1 )
{
    Reset();
    std::thread update_thread( UpdateThreadWorker, this );
    std::thread query_thread( QueryThreadWorker, this );

    update_ready_to_commit = true;
    update_condition.notify_all();
    {
        std::unique_lock< std::mutex > lock( update_mutex );
        if ( !update_committed_or_aborted )
        {
            update_condition.wait( lock );
        }
    }

    query_ready_to_commit = true;
    query_condition.notify_all();
    {
        std::unique_lock< std::mutex > lock( query_mutex );
        if ( !query_committed_or_aborted )
        {
            query_condition.wait( lock );
        }
    }

    ASSERT_EQ( new_totalprice, old_totalprice );

    update_thread.join();
    query_thread.join();
}

TEST_F( TestTPCHIsolation, test2 )
{
    Reset();

    update_should_commit = false;

    std::thread update_thread( UpdateThreadWorker, this );
    std::thread query_thread( QueryThreadWorker, this );

    update_ready_to_commit = true;
    update_condition.notify_all();
    {
        std::unique_lock< std::mutex > lock( update_mutex );
        if ( !update_committed_or_aborted )
        {
            update_condition.wait( lock );
        }
    }

    query_ready_to_commit = true;
    query_condition.notify_all();
    {
        std::unique_lock< std::mutex > lock( query_mutex );
        if ( !query_committed_or_aborted )
        {
            query_condition.wait( lock );
        }
    }

    ASSERT_EQ( new_totalprice, old_totalprice );

    update_thread.join();
    query_thread.join();
}

} // namespace aries_test