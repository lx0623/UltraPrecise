#include "test_common.h"

TEST( reduce, sum_integer )
{
    standard_context_t context;
    int count = 100;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< long > reduction( 1, context );
    int* pData = data.data();
    for( int i = 0; i < count; i++ )
    {
        pData[i] = 1;//d( get_mt19937() );
    }
    context.timer_begin();
    reduce( pData, count, reduction.data(), agg_sum_t< long >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    long sum = 0;
    for( int i = 0; i < count; ++i )
    {
        sum += pData[i];
        //printf("sum=%d\n", sum );
    }
    printf("sum=%ld\n", sum );
    ASSERT_EQ( sum, reduction.data()[0] );
}

TEST( reduce, sum_integer_null )
{
    standard_context_t context;
    int count = 100;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t < nullable_type< int > > data( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i].value = 1;//d( get_mt19937() );
        if( i % 10 )
            data.data()[i].flag = 1;
        else
        {
            data.data()[i].flag = 0;
        }

        //data.data()[i].flag = 1;
    }
    managed_mem_t < nullable_type< int64_t > >reduction( 1, context );
    //managed_mem_t < nullable_type< std::common_type<int, int64_t>::type > > reduction( 1, context );
    context.timer_begin();
    reduce( data.data(), count, reduction.data(), agg_sum_t< nullable_type< int64_t > >(), context );
    //reduce( data.data(), count, reduction.data(), agg_sum_t< nullable_type< std::common_type<int, int64_t>::type > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    long sum = 0;
    for( int i = 0; i < count; ++i )
    {
        if( data.data()[i].flag )
            sum += data.data()[i].value;
    }
    printf("sum=%d\n", sum );
    ASSERT_EQ( sum, reduction.data()[0].value );
}

TEST( reduce, count_integer )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > reduction( 1, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
    }
    context.timer_begin();
    reduce_count( data.data(), count, reduction.data(), agg_sum_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    ASSERT_EQ( count, reduction.data()[0] );
}

TEST( reduce, count_integer_null )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t < nullable_type< int > > data( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[i].flag = 1;
        else
        {
            data.data()[i].flag = 0;
        }
    }
    managed_mem_t< int > reduction( 1, context );
    context.timer_begin();
    reduce_count( data.data(), count, reduction.data(), agg_sum_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    int sum = 0;
    for( int i = 0; i < count; ++i )
    {
        if( data.data()[i].flag )
            sum += 1;
    }
    ASSERT_EQ( sum, reduction.data()[0] );
}

TEST( reduce, max_integer )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > reduction( 1, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
    }
    context.timer_begin();
    reduce( data.data(), count, reduction.data(), agg_max_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    int max = INT32_MIN;
    for( int i = 0; i < count; ++i )
    {
        max = std::max( max, data.data()[i] );
    }
    ASSERT_EQ( max, reduction.data()[0] );
}

TEST( reduce, max_integer_null )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t < nullable_type< int > > data( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[i].flag = 1;
        else
        {
            data.data()[i].flag = 0;
        }
    }
    managed_mem_t < nullable_type< int > > reduction( 1, context );
    context.timer_begin();
    reduce( data.data(), count, reduction.data(), agg_max_t< nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    int max = INT32_MIN;
    for( int i = 0; i < count; ++i )
    {
        if( data.data()[i].flag )
            max = std::max( max, data.data()[i].value );
    }
    ASSERT_EQ( max, reduction.data()[0].value );
}

TEST( reduce, min_integer )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > reduction( 1, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
    }
    context.timer_begin();
    reduce( data.data(), count, reduction.data(), agg_min_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    int min = INT32_MAX;
    for( int i = 0; i < count; ++i )
    {
        min = std::min( min, data.data()[i] );
    }
    ASSERT_EQ( min, reduction.data()[0] );
}

TEST( reduce, min_integer_null )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t < nullable_type< int > > data( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[i].flag = 1;
        else
        {
            data.data()[i].flag = 0;
        }
    }
    managed_mem_t < nullable_type< int > > reduction( 1, context );
    context.timer_begin();
    reduce( data.data(), count, reduction.data(), agg_min_t< nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    int min = INT32_MAX;
    for( int i = 0; i < count; ++i )
    {
        if( data.data()[i].flag )
            min = std::min( min, data.data()[i].value );
    }
    ASSERT_EQ( min, reduction.data()[0].value );
}

TEST( reduce, count_str )
{
    standard_context_t context;
    int repeat = 5;
    const char* file = "/var/rateup/data/scale_1/lineitem/lineitem14"; //6 shipdate char(10)
    char *h_data;
    int colSize = 10;
    int len = loadColumn( file, colSize, &h_data );
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char * keys_input = keys.data();
    for( int i = 0; i < repeat; ++i )
    {
        memcpy( keys_input + i * len * colSize, h_data, len * colSize );
    }
    free( h_data );
    int count = len * repeat;
    managed_mem_t< int > reduction( 1, context );
    context.timer_begin();
    reduce_count( keys_input, count, reduction.data(), agg_sum_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    ASSERT_EQ( count, reduction.data()[0] );
}

TEST( reduce, count_str_null )
{
    standard_context_t context;
    int repeat = 5;
    const char* file = "/var/rateup/data/scale_1/lineitem/lineitem14"; //6 shipdate char(10)
    char *h_data;
    int colSize = 10;
    int len = loadColumn( file, colSize, &h_data );
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char * keys_input = keys.data();
    for( int i = 0; i < repeat; ++i )
    {
        memcpy( keys_input + i * len * colSize, h_data, len * colSize );
    }
    free( h_data );
    int count = len * repeat;

    int validCount = 0;
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            *( keys_input + i * colSize ) = 1;
            ++validCount;
        }
        else
        {
            *( keys_input + i * colSize ) = 0;
        }
    }

    managed_mem_t< int > reduction( 1, context );
    context.timer_begin();
    reduce_count( keys_input, colSize, count, reduction.data(), agg_sum_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    ASSERT_EQ( validCount, reduction.data()[0] );
}

//
//TEST( reduce, min_integer )
//{
//    standard_context_t context;
//    int repeat = 5;
//    const char* file = "/var/rateup/data/scale_1/lineitem14"; //6 shipdate char(10)
//    char *h_data;
//    int colSize = 10;
//    int len = loadColumn( file, colSize, &h_data );
//    managed_mem_t< char > keys( len * colSize * repeat, context );
//    char * keys_input = keys.data();
//    for( int i = 0; i < repeat; ++i )
//    {
//        memcpy( keys_input + i * len * colSize, h_data, len * colSize );
//    }
//    free( h_data );
//    managed_mem_t< int > vals( len * repeat, context );
//    int * vals_input = vals.data();
//    int nullCount = 0;
//    for( int i = 0; i < len * repeat; i++ )
//    {
//        if( i % 10 )
//            *( keys_input + i * colSize ) = 1;
//        else
//        {
//            *( keys_input + i * colSize ) = 0;
//            ++nullCount;
//        }
//        vals_input[i] = i;
//    }
//    managed_mem_t< char > old = keys.clone();
//    char * old_input = old.data();
//    context.timer_begin();
//    reduce< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, less_t_str_null_smaller< true, true >(),
//            context );
//    printf( "gpu time: %3.1f\n", context.timer_end() );
//    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );
//    for( int i = 0; i < nullCount; ++i )
//        ASSERT_EQ( *( keys_input + colSize * i ), 0 );
//    const char* str1;
//    const char* str2;
//    const char* str3;
//    for( int i = nullCount; i < len * repeat - 1; ++i )
//    {
//        str1 = keys_input + colSize * i;
//        str2 = str1 + colSize;
//        str3 = old_input + vals_input[i] * colSize;
//        ASSERT_TRUE( *str1 && *str2 );
//        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
//        ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
//    }
//}
//
//TEST( reduce, min_integer_null )
//{
//    standard_context_t context;
//    int repeat = 5;
//    const char* file = "/var/rateup/data/scale_1/lineitem14"; //6 shipdate char(10)
//    char *h_data;
//    int colSize = 10;
//    int len = loadColumn( file, colSize, &h_data );
//    managed_mem_t< char > keys( len * colSize * repeat, context );
//    char * keys_input = keys.data();
//    for( int i = 0; i < repeat; ++i )
//    {
//        memcpy( keys_input + i * len * colSize, h_data, len * colSize );
//    }
//    free( h_data );
//    managed_mem_t< int > vals( len * repeat, context );
//    int * vals_input = vals.data();
//    int nullCount = 0;
//    for( int i = 0; i < len * repeat; i++ )
//    {
//        if( i % 10 )
//            *( keys_input + i * colSize ) = 1;
//        else
//        {
//            *( keys_input + i * colSize ) = 0;
//            ++nullCount;
//        }
//        vals_input[i] = i;
//    }
//    managed_mem_t< char > old = keys.clone();
//    char * old_input = old.data();
//    context.timer_begin();
//    reduce< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, less_t_str_null_bigger< true, true >(),
//            context );
//    printf( "gpu time: %3.1f\n", context.timer_end() );
//    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );
//    for( int i = len * repeat - nullCount; i < len * repeat; ++i )
//        ASSERT_EQ( *( keys_input + colSize * i ), 0 );
//    const char* str1;
//    const char* str2;
//    const char* str3;
//    for( int i = nullCount; i < len * repeat - nullCount - 1; ++i )
//    {
//        str1 = keys_input + colSize * i;
//        str2 = str1 + colSize;
//        str3 = old_input + vals_input[i] * colSize;
//        ASSERT_TRUE( *str1 && *str2 );
//        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
//        ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
//    }
//}
