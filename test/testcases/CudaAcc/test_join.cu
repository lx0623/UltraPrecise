#include "test_common.h"
#include "CudaAcc/AriesEngineAlgorithm.h"
#include <algorithm>
#include <typeinfo>
static const char* DB_NAME = "scale_1";
TEST(UT_join, string_inner)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_inner_join( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 66626 );
    printf( "result count is: %ld\n", join_count );
    for( int i = 0; i < join_count; ++i )
        ASSERT_EQ(
                strncmp( ( const char* )customerPhone->GetData() + right[i] * colSize, ( const char* )supplierPhone->GetData() + left[i] * colSize,
                        colSize ), 0 );
}

TEST(UT_join, string_left)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_left_join( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    int* left = result.LeftIndices->GetData();
    int* right = result.RightIndices->GetData();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 66643 );
}

TEST(UT_join, string_right)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_right_join( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    int* left = result.LeftIndices->GetData();
    int* right = result.RightIndices->GetData();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 162930 );
}

TEST(UT_join, string_full)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_full_join( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    int* left = result.LeftIndices->GetData();
    int* right = result.RightIndices->GetData();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 162947 );
}

TEST(UT_join, nullstring_inner)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < supplierPhone->GetItemCount(); i++ )
    {
        if( i % 3 == 0 )
            *( supplierPhone->GetData() + i * colSize ) = 0;
        else
            *( supplierPhone->GetData() + i * colSize ) = 1;
    }
    for( int i = 0; i < customerPhone->GetItemCount(); i++ )
    {
        if( i % 10 == 0 )
            *( customerPhone->GetData() + i * colSize ) = 0;
        else
            *( customerPhone->GetData() + i * colSize ) = 1;
    }

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_inner_join_has_null( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 104035 );
    printf( "result count is: %ld\n", join_count );

    for( int i = 0; i < join_count; ++i )
    {
        ASSERT_EQ( *( customerPhone->GetData() + right[i] * colSize ), 1 );
        ASSERT_EQ( *( supplierPhone->GetData() + left[i] * colSize ), 1 );
        ASSERT_EQ(
                strncmp( ( const char* )customerPhone->GetData() + right[i] * colSize, ( const char* )supplierPhone->GetData() + left[i] * colSize,
                        colSize ), 0 );
    }
}

TEST(UT_join, nullstring_left)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < supplierPhone->GetItemCount(); i++ )
    {
        if( i % 3 == 0 )
            *( supplierPhone->GetData() + i * colSize ) = 0;
        else
            *( supplierPhone->GetData() + i * colSize ) = 1;
    }
    for( int i = 0; i < customerPhone->GetItemCount(); i++ )
    {
        if( i % 10 == 0 )
            *( customerPhone->GetData() + i * colSize ) = 0;
        else
            *( customerPhone->GetData() + i * colSize ) = 1;
    }

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_left_join_has_null( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    int* left = result.LeftIndices->GetData();
    int* right = result.RightIndices->GetData();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 107369 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, nullstring_right)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < supplierPhone->GetItemCount(); i++ )
    {
        if( i % 3 == 0 )
            *( supplierPhone->GetData() + i * colSize ) = 0;
        else
            *( supplierPhone->GetData() + i * colSize ) = 1;
    }
    for( int i = 0; i < customerPhone->GetItemCount(); i++ )
    {
        if( i % 10 == 0 )
            *( customerPhone->GetData() + i * colSize ) = 0;
        else
            *( customerPhone->GetData() + i * colSize ) = 1;
    }

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_right_join_has_null( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    int* left = result.LeftIndices->GetData();
    int* right = result.RightIndices->GetData();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 182352 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, nullstring_full)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < supplierPhone->GetItemCount(); i++ )
    {
        if( i % 3 == 0 )
            *( supplierPhone->GetData() + i * colSize ) = 0;
        else
            *( supplierPhone->GetData() + i * colSize ) = 1;
    }
    for( int i = 0; i < customerPhone->GetItemCount(); i++ )
    {
        if( i % 10 == 0 )
            *( customerPhone->GetData() + i * colSize ) = 0;
        else
            *( customerPhone->GetData() + i * colSize ) = 1;
    }

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinPair result = sort_based_full_join_has_null( ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(),
            ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(), colSize, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    int* left = result.LeftIndices->GetData();
    int* right = result.RightIndices->GetData();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 185686 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, nullint_inner)
{
    standard_context_t context;

    int num_needles = 100000000;
    int num_haystack = 100000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< nullable_type< long > > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i].value = i;        //d( get_mt19937() );
        if( i % 3 == 0 )
        {
            haystack.data()[i].flag = 0;
        }
        else
            haystack.data()[i].flag = 1;
    }

    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< nullable_type< long > > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinPair result = sort_based_inner_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 60000000 );
    printf( "result count is: %ld\n", join_count );
    {
        for( int i = 0; i < join_count; ++i )
        {
            //printf( "lIndex=%d, rIndex=%d,   haystack is:%10.10s, needles is:%10.10s\n", pair->x, pair->y, haystack + pair->x * colSize, needles + pair->y * colSize );
            ASSERT_TRUE( left[i] >= 0 && right[i] >= 0 );
            ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
        }
    }
}

TEST(UT_join, nullint_left)
{
    standard_context_t context;

    int num_needles = 100000000;
    int num_haystack = 100000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< nullable_type< long > > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i].value = i;        //d( get_mt19937() );
        if( i % 3 == 0 )
        {
            haystack.data()[i].flag = 0;
        }
        else
            haystack.data()[i].flag = 1;
    }
    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< nullable_type< long > > clone_haystack = haystack.clone();

    context.timer_begin();
    JoinPair result = sort_based_left_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 100000000 );
    printf( "result count is: %ld\n", join_count );
    {
        int count = 0;
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( left[i] >= 0 );
            if( right[i] >= 0 )
            {
                ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
            }
            else
                ++count;
        }
        ASSERT_EQ( count, 40000000 );
        printf( "result null count is: %d\n", count );
    }
}

TEST(UT_join, nullint_right)
{
    standard_context_t context;

    int num_needles = 100000000;
    int num_haystack = 100000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< nullable_type< long > > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i].value = i;        //d( get_mt19937() );
        if( i % 3 == 0 )
        {
            haystack.data()[i].flag = 0;
        }
        else
            haystack.data()[i].flag = 1;
    }

    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< nullable_type< long > > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinPair result = sort_based_right_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 100000000 );
    printf( "result count is: %ld\n", join_count );
    {
        int count = 0;
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( right[i] >= 0 );
            if( left[i] >= 0 )
                ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
            else
                ++count;
        }
        ASSERT_EQ( count, 40000000 );
        printf( "result null count is: %d\n", count );
    }
}

TEST(UT_join, nullint_full)
{
    standard_context_t context;

    int num_needles = 100000000;
    int num_haystack = 100000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< nullable_type< long > > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i].value = i;        //d( get_mt19937() );
        if( i % 3 == 0 )
        {
            haystack.data()[i].flag = 0;
        }
        else
            haystack.data()[i].flag = 1;
    }

    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< nullable_type< long > > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinPair result = sort_based_full_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 140000000 );
    printf( "result count is: %ld\n", join_count );
    {
        int count = 0;
        for( int i = 0; i < join_count; ++i )
        {
            if( left[i] >= 0 && right[i] >= 0 )
                ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
            else
                ++count;
        }
        ASSERT_EQ( count, 80000000 );
        printf( "result null count is: %d\n", count );
    }
}

TEST(UT_join, int_inner)
{
    standard_context_t context;

    AriesDataBufferSPtr l_orderkey = ReadColumn( DB_NAME, "lineitem", 1 );
    AriesDataBufferSPtr o_orderkey = ReadColumn( DB_NAME, "orders", 1 );

    AriesDataBufferSPtr clone_l_orderkey = l_orderkey->Clone();
    AriesDataBufferSPtr clone_o_orderkey = o_orderkey->Clone();

    context.timer_begin();
    JoinPair result = sort_based_inner_join( ( int32_t* )clone_l_orderkey->GetData(), clone_l_orderkey->GetItemCount(),
            ( int32_t* )clone_o_orderkey->GetData(), clone_o_orderkey->GetItemCount(), context, nullptr, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 6001215 );
    printf( "result count is: %ld\n", join_count );
    for( int i = 0; i < join_count; ++i )
        ASSERT_TRUE( ( ( int32_t* )l_orderkey->GetData() )[left[i]] == ( ( int32_t* )o_orderkey->GetData() )[right[i]] );
}

TEST(UT_join, int_left)
{
    standard_context_t context;

    AriesDataBufferSPtr ps_avail = ReadColumn( DB_NAME, "partsupp", 3 );
    AriesDataBufferSPtr p_size = ReadColumn( DB_NAME, "part", 6 );

    AriesDataBufferSPtr clone_ps_avail = ps_avail->Clone();
    AriesDataBufferSPtr clone_p_size = p_size->Clone();

    context.timer_begin();
    JoinPair result = sort_based_left_join( ( int32_t* )clone_ps_avail->GetData(), clone_ps_avail->GetItemCount(),
            ( int32_t* )clone_p_size->GetData(), clone_p_size->GetItemCount(), context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 17057080 );

    printf( "result count is: %ld\n", join_count );
    {
        int count = 0;
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( left[i] >= 0 );
            if( right[i] >= 0 )
                ASSERT_TRUE( ( ( int32_t* )ps_avail->GetData() )[left[i]] == ( ( int32_t* )p_size->GetData() )[right[i]] );
            else
                ++count;
        }
        ASSERT_EQ( count, 795936 );
        printf( "result null count is: %d\n", count );
    }
}

TEST(UT_join, int_right)
{
    standard_context_t context;

    AriesDataBufferSPtr ps_avail = ReadColumn( DB_NAME, "partsupp", 3 );
    AriesDataBufferSPtr p_size = ReadColumn( DB_NAME, "part", 6 );

    AriesDataBufferSPtr clone_ps_avail = ps_avail->Clone();
    AriesDataBufferSPtr clone_p_size = p_size->Clone();

    context.timer_begin();
    JoinPair result = sort_based_right_join( ( int32_t* )clone_ps_avail->GetData(), clone_ps_avail->GetItemCount(),
            ( int32_t* )clone_p_size->GetData(), clone_p_size->GetItemCount(), context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 16261144 );

    printf( "result count is: %ld\n", join_count );
    {
        int count = 0;
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( left[i] >= 0 );
            if( right[i] >= 0 )
                ASSERT_TRUE( ( ( int32_t* )ps_avail->GetData() )[left[i]] == ( ( int32_t* )p_size->GetData() )[right[i]] );
            else
                ++count;
        }
        ASSERT_EQ( count, 0 );
        printf( "result null count is: %d\n", count );
    }
}

TEST(UT_join, int_full)
{
    standard_context_t context;

    AriesDataBufferSPtr ps_avail = ReadColumn( DB_NAME, "partsupp", 3 );
    AriesDataBufferSPtr p_size = ReadColumn( DB_NAME, "part", 6 );

    AriesDataBufferSPtr clone_ps_avail = ps_avail->Clone();
    AriesDataBufferSPtr clone_p_size = p_size->Clone();

    context.timer_begin();
    JoinPair result = sort_based_full_join( ( int32_t* )clone_ps_avail->GetData(), clone_ps_avail->GetItemCount(),
            ( int32_t* )clone_p_size->GetData(), clone_p_size->GetItemCount(), context, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 17057080 );

    printf( "result count is: %ld\n", join_count );
    {
        int count = 0;
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( left[i] >= 0 );
            if( right[i] >= 0 )
                ASSERT_TRUE( ( ( int32_t* )ps_avail->GetData() )[left[i]] == ( ( int32_t* )p_size->GetData() )[right[i]] );
            else
                ++count;
        }
        ASSERT_EQ( count, 795936 );
        printf( "result null count is: %d\n", count );
    }
}

TEST(UT_join, int_semi)
{
    standard_context_t context;
    int num_needles = 100000;
    int num_haystack = 100000;

    managed_mem_t< int > needles( num_needles, context );
    managed_mem_t< int > haystack( num_haystack, context );

    for( int i = 0; i < num_needles; i++ )
        needles.data()[i] = i; //d( get_mt19937() );

    for( int i = 0; i < num_haystack; i++ )
        haystack.data()[i] = i * 3; //d( get_mt19937() );

    AriesDataType dataType
    { AriesValueType::INT32, 1 };
    AriesColumnType colType
    { dataType, false, false };
    managed_mem_t< int > clone_needles = needles.clone();
    managed_mem_t< int > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_semi_join( ( int8_t* )clone_haystack.data(), num_haystack, ( int8_t* )clone_needles.data(), num_needles,
            colType, colType, &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    auto left = result->DataToHostMemory();
    size_t totalCount = result->GetItemCount();

    size_t join_count = 0;

    {
        for( int i = 0; i < totalCount; ++i )
        {
            if( left[i] == 1 )
            {
                ASSERT_TRUE( std::find( needles.data(), needles.data() + num_needles, haystack.data()[i] ) != needles.data() + num_needles );
                ++join_count;
            }
            else
            {
                ASSERT_TRUE( std::find( needles.data(), needles.data() + num_needles, haystack.data()[i] ) == needles.data() + num_needles );
            }
        }
    }
    printf( "result count is: %ld\n", join_count );
    ASSERT_EQ( join_count, 33334 );
}

TEST(UT_join, int_semi_null)
{
    standard_context_t context;

    int num_needles = 100000;
    int num_haystack = 100000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< nullable_type< int > > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i].value = i;        //d( get_mt19937() );
        if( i % 3 == 0 )
        {
            haystack.data()[i].flag = 0;
        }
        else
            haystack.data()[i].flag = 1;
    }
    AriesDataType dataType
    { AriesValueType::INT32, 1 };
    AriesColumnType colType
    { dataType, true, false };
    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< nullable_type< int > > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_semi_join( ( int8_t* )clone_haystack.data(), num_haystack, ( int8_t* )clone_needles.data(), num_needles,
            colType, colType, &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result->DataToHostMemory();
    size_t totalCount = result->GetItemCount();

    size_t join_count = 0;

    {
        for( int i = 0; i < totalCount; ++i )
        {
            auto data = haystack.data()[i];
            if( left[i] == 1 )
            {
                //printf("haystack[%d], flag=%d, value=%d\n", i,  haystack.data()[i].flag,  haystack.data()[i].value );
                ASSERT_TRUE( haystack.data()[i].flag );
                ASSERT_TRUE( std::find_if( needles.data(), needles.data() + num_needles, [=](auto param )
                {   return param.flag = data.flag && param.value == data.value;} ) != needles.data() + num_needles );
                ++join_count;
            }
            else
            {
                if( data.flag )
                {
                    ASSERT_TRUE( std::find_if( needles.data(), needles.data() + num_needles, [=](auto param )
                    {   return param.flag == data.flag && param.value == data.value;} ) == needles.data() + num_needles );
                }
            }
        }
    }
    printf( "result count is: %ld\n", join_count );
    ASSERT_EQ( join_count, 60000 );
}

TEST(UT_join, int_anti)
{
    standard_context_t context;
    int num_needles = 100000;
    int num_haystack = 100000;

    managed_mem_t< int > needles( num_needles, context );
    managed_mem_t< long > haystack( num_haystack, context );

    for( int i = 0; i < num_needles; i++ )
        needles.data()[i] = i; //d( get_mt19937() );

    for( int i = 0; i < num_haystack; i++ )
        haystack.data()[i] = i * 3; //d( get_mt19937() );

    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_anti_join( haystack.data(), num_haystack, needles.data(), num_needles, &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    auto left = result->DataToHostMemory();
    size_t totalCount = result->GetItemCount();
    size_t join_count = 0;

    {
        for( int i = 0; i < totalCount; ++i )
        {
            if( left[i] == 1 )
            {
                ASSERT_TRUE( std::find( needles.data(), needles.data() + num_needles, haystack.data()[i] ) == needles.data() + num_needles );

                ++join_count;
            }
            else
            {
                ASSERT_TRUE( std::find( needles.data(), needles.data() + num_needles, haystack.data()[i] ) != needles.data() + num_needles );
            }
        }
    }
    printf( "result count is: %ld\n", join_count );
    ASSERT_EQ( join_count, 66666 );
}

TEST(UT_join, int_anti_null)
{
    standard_context_t context;

    int num_needles = 100000;
    int num_haystack = 100000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< nullable_type< long > > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i].value = i;        //d( get_mt19937() );
        if( i % 3 == 0 )
        {
            haystack.data()[i].flag = 0;
        }
        else
            haystack.data()[i].flag = 1;
    }

    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< nullable_type< long > > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_anti_join_has_null( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, &params,
            context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result->DataToHostMemory();
    size_t totalCount = result->GetItemCount();
    size_t join_count = 0;

    {
        for( int i = 0; i < totalCount; ++i )
        {
            auto data = haystack.data()[i];
            if( left[i] == 1 )
            {
                if( haystack.data()[i].flag )
                {
                    ASSERT_TRUE( std::find_if( needles.data(), needles.data() + num_needles, [=](auto param )
                    {   return param.flag == data.flag && param.value == data.value;} ) == needles.data() + num_needles );
                }
                ++join_count;
            }
            else
            {
                ASSERT_TRUE( haystack.data()[i].flag );
                ASSERT_TRUE( std::find_if( needles.data(), needles.data() + num_needles, [=](auto param )
                {   return param.flag == data.flag && param.value == data.value;} ) != needles.data() + num_needles );
            }
        }
    }
    printf( "result count is: %ld\n", join_count );
    ASSERT_EQ( join_count, 40000 );
}

TEST(UT_join, string_semi)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_semi_join( ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(),
            ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(), colSize, less_t_str_null_smaller_join< false, false >(),
            less_t_str_null_smaller< false, false >(), &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    size_t join_count = 0;
    for( int i = 0; i < result->GetItemCount(); ++i )
    {
        join_count += result->GetValue( i );
    }

    ASSERT_EQ( join_count, 53696 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, string_anti)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_anti_join( ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(),
            ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(), colSize, &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    size_t join_count = 0;
    for( int i = 0; i < result->GetItemCount(); ++i )
    {
        join_count += result->GetValue( i );
    }

    ASSERT_EQ( join_count, 96304 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, nullstring_semi)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < supplierPhone->GetItemCount(); i++ )
    {
        if( i % 3 == 0 )
            *( supplierPhone->GetData() + i * colSize ) = 0;
        else
            *( supplierPhone->GetData() + i * colSize ) = 1;
    }
    for( int i = 0; i < customerPhone->GetItemCount(); i++ )
    {
        if( i % 10 == 0 )
            *( customerPhone->GetData() + i * colSize ) = 0;
        else
            *( customerPhone->GetData() + i * colSize ) = 1;
    }

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_semi_join( ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(),
            ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(), colSize, less_t_str_null_smaller_join< true, true >(),
            less_t_str_null_smaller< true, true >(), &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    size_t join_count = 0;
    for( int i = 0; i < result->GetItemCount(); ++i )
    {
        join_count += result->GetValue( i );
    }

    ASSERT_EQ( join_count, 71683 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, nullstring_anti)
{
    standard_context_t context;

    AriesDataBufferSPtr supplierPhone = ReadColumn( DB_NAME, "supplier", 5 );
    AriesDataBufferSPtr customerPhone = ReadColumn( DB_NAME, "customer", 5 );

    int colSize = supplierPhone->GetItemSizeInBytes();

    for( int i = 0; i < supplierPhone->GetItemCount(); ++i )
        memset( supplierPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < customerPhone->GetItemCount(); ++i )
        memset( customerPhone->GetData() + i * colSize + 6, 0, 9 );

    for( int i = 0; i < supplierPhone->GetItemCount(); i++ )
    {
        if( i % 3 == 0 )
            *( supplierPhone->GetData() + i * colSize ) = 0;
        else
            *( supplierPhone->GetData() + i * colSize ) = 1;
    }
    for( int i = 0; i < customerPhone->GetItemCount(); i++ )
    {
        if( i % 10 == 0 )
            *( customerPhone->GetData() + i * colSize ) = 0;
        else
            *( customerPhone->GetData() + i * colSize ) = 1;
    }

    AriesDataBufferSPtr clone_supplierPhone = supplierPhone->Clone();
    AriesDataBufferSPtr clone_customerPhone = customerPhone->Clone();

    context.timer_begin();
    JoinDynamicCodeParams params;
    AriesInt32ArraySPtr result = sort_based_anti_join_has_null( ( char* )clone_customerPhone->GetData(), clone_customerPhone->GetItemCount(),
            ( char* )clone_supplierPhone->GetData(), clone_supplierPhone->GetItemCount(), colSize, &params, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    size_t join_count = 0;
    for( int i = 0; i < result->GetItemCount(); ++i )
    {
        join_count += result->GetValue( i );
    }

    ASSERT_EQ( join_count, 78317 );
    printf( "result count is: %ld\n", join_count );
}

TEST(UT_join, nullint_long_inner)
{
    standard_context_t context;

    int num_needles = 100000000;
    int num_haystack = 100000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< long > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i] = i;        //d( get_mt19937() );
    }

    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< long > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinPair result = sort_based_inner_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 90000000 );
    printf( "result count is: %ld\n", join_count );
    {
        for( int i = 0; i < join_count; ++i )
        {
            //printf( "lIndex=%d, rIndex=%d,   haystack is:%10.10s, needles is:%10.10s\n", pair->x, pair->y, haystack + pair->x * colSize, needles + pair->y * colSize );
            ASSERT_TRUE( left[i] >= 0 && right[i] >= 0 );
            ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
        }
    }
}

TEST(UT_join, nullint_double_inner)
{
    standard_context_t context;

    int num_needles = 100000000;
    int num_haystack = 100000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< double > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i] = i;        //d( get_mt19937() );
    }
    needles.PrefetchToGpu();
    haystack.PrefetchToGpu();
    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< double > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinPair result = sort_based_inner_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 90000000 );
    printf( "result count is: %ld\n", join_count );
    {
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( left[i] >= 0 && right[i] >= 0 );
            ASSERT_TRUE( needles.data()[right[i]].flag == 1 );
            ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
        }
    }
}

TEST(UT_join, nullint_decimal_inner)
{
    standard_context_t context;

    int num_needles = 10000000;
    int num_haystack = 10000000;

    managed_mem_t< nullable_type< int > > needles( num_needles, context );
    managed_mem_t< aries_acc::Decimal > haystack( num_haystack, context );
    std::uniform_int_distribution< int > d( 0, 100000 );

    for( int i = 0; i < num_needles; i++ )
    {
        needles.data()[i].value = i; //d( get_mt19937() );
        if( i % 10 == 0 )
        {
            needles.data()[i].flag = 0;
        }
        else
            needles.data()[i].flag = 1;
    }

    for( int i = 0; i < num_haystack; i++ )
    {
        haystack.data()[i] = i;        //d( get_mt19937() );
    }
    needles.PrefetchToGpu();
    haystack.PrefetchToGpu();
    managed_mem_t< nullable_type< int > > clone_needles = needles.clone();
    managed_mem_t< aries_acc::Decimal > clone_haystack = haystack.clone();
    context.timer_begin();
    JoinPair result = sort_based_inner_join( clone_haystack.data(), num_haystack, clone_needles.data(), num_needles, context, nullptr, nullptr );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    auto left = result.LeftIndices->DataToHostMemory();
    auto right = result.RightIndices->DataToHostMemory();
    size_t join_count = result.JoinCount;

    ASSERT_EQ( join_count, 9000000 );
    printf( "result count is: %ld\n", join_count );
    {
        for( int i = 0; i < join_count; ++i )
        {
            ASSERT_TRUE( left[i] >= 0 && right[i] >= 0 );
            ASSERT_TRUE( needles.data()[right[i]].flag == 1 );
            ASSERT_TRUE( haystack.data()[left[i]] == needles.data()[right[i]] );
        }
    }
}
