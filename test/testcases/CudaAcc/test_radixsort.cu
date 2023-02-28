/*
 * test_radixsort.cu
 *
 *  Created on: Jul 17, 2019
 *      Author: lichi
 */

#include "test_common.h"
static const char* DB_NAME = "scale_1";

template< typename type_t, typename val_t, typename aries_acc::enable_if< aries_acc::is_arithmetic< type_t >::value, type_t >::type* = nullptr >
void radix_sort2( type_t* keys, val_t* associated, int count, context_t& context, bool bAsc = true, bool bShuffle = false )
{
    //hard code for test
    if( sizeof(type_t) * count > 4 * 300000000 )
    {
        int part1_count = count / 2;
        int part2_count = count - part1_count;

        managed_mem_t< type_t > keys_part1_output( part1_count, context, false );
        managed_mem_t< val_t > associated_part1_output( part1_count, context, false );

        managed_mem_t< type_t > keys_part2_output( part2_count, context, false );
        managed_mem_t< val_t > associated_part2_output( part2_count, context, false );

        radix_sort( keys, associated, keys_part1_output.data(), associated_part1_output.data(), part1_count, context, bAsc );
        radix_sort( keys + part1_count, associated + part1_count, keys_part2_output.data(), associated_part2_output.data(), part2_count, context,
                bAsc );

        int pos;
        if( bAsc )
            pos = merge_path( keys_part1_output.data(), part1_count, keys_part2_output.data(), part2_count, part1_count,
                    less_t_null_smaller< type_t >() );
        else
            pos = merge_path( keys_part1_output.data(), part1_count, keys_part2_output.data(), part2_count, part1_count,
                    greater_t_null_smaller< type_t >() );

        int part2_half = part1_count - pos;

        if( bShuffle )
        {
            if( bAsc )
                merge( keys_part1_output.data(), associated_part1_output.data(), pos, keys_part2_output.data(), associated_part2_output.data(),
                        part2_half, keys, associated, less_t_null_smaller< type_t >(), context );
            else
                merge( keys_part1_output.data(), associated_part1_output.data(), pos, keys_part2_output.data(), associated_part2_output.data(),
                        part2_half, keys, associated, greater_t_null_smaller< type_t >(), context );
            cudaMemPrefetchAsync( keys_part1_output.data(), sizeof(type_t) * pos, cudaCpuDeviceId );
            cudaMemPrefetchAsync( keys_part2_output.data(), sizeof(type_t) * part2_half, cudaCpuDeviceId );
            cudaMemPrefetchAsync( associated_part1_output.data(), sizeof(val_t) * pos, cudaCpuDeviceId );
            cudaMemPrefetchAsync( associated_part2_output.data(), sizeof(val_t) * part2_half, cudaCpuDeviceId );
            if( bAsc )
                merge( keys_part1_output.data() + pos, associated_part1_output.data() + pos, part1_count - pos, keys_part2_output.data() + part2_half,
                        associated_part2_output.data() + part2_half, part2_count - part2_half, keys + pos + part2_half, associated + pos + part2_half,
                        less_t_null_smaller< type_t >(), context );
            else
                merge( keys_part1_output.data() + pos, associated_part1_output.data() + pos, part1_count - pos, keys_part2_output.data() + part2_half,
                        associated_part2_output.data() + part2_half, part2_count - part2_half, keys + pos + part2_half, associated + pos + part2_half,
                        greater_t_null_smaller< type_t >(), context );
        }
        else
        {
            if( bAsc )
                merge_vals_only( keys_part1_output.data(), associated_part1_output.data(), pos, keys_part2_output.data(),
                        associated_part2_output.data(), part2_half, associated, less_t_null_smaller< type_t >(), context );
            else
                merge_vals_only( keys_part1_output.data(), associated_part1_output.data(), pos, keys_part2_output.data(),
                        associated_part2_output.data(), part2_half, associated, greater_t_null_smaller< type_t >(), context );
            if( bAsc )
                merge_vals_only( keys_part1_output.data() + pos, associated_part1_output.data() + pos, part1_count - pos,
                        keys_part2_output.data() + part2_half, associated_part2_output.data() + part2_half, part2_count - part2_half,
                        associated + pos + part2_half, less_t_null_smaller< type_t >(), context );
            else
                merge_vals_only( keys_part1_output.data() + pos, associated_part1_output.data() + pos, part1_count - pos,
                        keys_part2_output.data() + part2_half, associated_part2_output.data() + part2_half, part2_count - part2_half,
                        associated + pos + part2_half, greater_t_null_smaller< type_t >(), context );
        }
    }
    else
    {
        managed_mem_t< type_t > keys_output( count, context, false );
        managed_mem_t< val_t > associated_output( count, context, false );
        radix_sort( keys, associated, keys_output.data(), associated_output.data(), count, context, bAsc );
        cudaMemcpy( associated, associated_output.data(), count * sizeof(val_t), cudaMemcpyDefault );
        if( bShuffle )
            cudaMemcpy( keys, keys_output.data(), count * sizeof(type_t), cudaMemcpyDefault );
    }
}

TEST( radixsort, int_asc )
{
    standard_context_t context;
    int count = 10000000;

    for( int i = 0; i < 10; ++i )
    {
        std::uniform_int_distribution< int > d( 0, 10000000 );
        managed_mem_t< int > data( count, context );
        managed_mem_t< int > vals_output( count, context );
        data.PrefetchToCpu();
        vals_output.PrefetchToCpu();
        for( int i = 0; i < count; i++ )
        {
            data.data()[i] = d( get_mt19937() );
            vals_output.data()[i] = i;
        }
        managed_mem_t< int > data_cloned = data.clone();
        data.PrefetchToGpu();
        vals_output.PrefetchToGpu();
        data_cloned.PrefetchToGpu();
        context.timer_begin();
        radix_sort( data.data(), vals_output.data(), count, context, true, true );
        printf( "gpu time: %3.1f\n", context.timer_end() );
        printf( "item count is: %d\n", count );
    }
//    for( int i = 0; i < count - 1; ++i )
//    {
//        ASSERT_TRUE( data_cloned.data()[vals_output.data()[i]] <= data_cloned.data()[vals_output.data()[i + 1]] );
//    }
}

TEST( radixsort, int_desc )
{
    standard_context_t context;
    int count = 400000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_output( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
        vals_output.data()[i] = i;
    }
    managed_mem_t< int > data_cloned = data.clone();

    context.timer_begin();
    radix_sort( data.data(), vals_output.data(), count, context, false, true );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( data_cloned.data()[vals_output.data()[i]] >= data_cloned.data()[vals_output.data()[i + 1]] );
    }
}

TEST( radixsort, int_nullsmaller_asc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
    managed_mem_t< int > vals_output( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[i].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[i].flag = 1;
        else
        {
            data.data()[i].flag = 0;
            ++nullCount;
        }
        vals_output.data()[i] = i;
    }

    context.timer_begin();
    radix_sort( data.data(), vals_output.data(), count, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[vals_output.data()[i]].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[vals_output.data()[i]].flag && data.data()[vals_output.data()[i + 1]].flag );
        ASSERT_TRUE( data.data()[vals_output.data()[i]] <= data.data()[vals_output.data()[i + 1]] );
    }
}

TEST( radixsort, long_asc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< long > d( 0, 10000000 );
    managed_mem_t< long > data( count, context );
    managed_mem_t< int > vals_output( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
        vals_output.data()[i] = i;
    }
    data.PrefetchToGpu();
    vals_output.PrefetchToGpu();
    context.timer_begin();
    radix_sort( data.data(), vals_output.data(), count, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[vals_output.data()[i]] <= data.data()[vals_output.data()[i + 1]] );
    }
}

TEST( radixsort, nulllong_smaller_asc )
{
    standard_context_t context;
    int count = 67238110;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< long > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[i].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[i].flag = 1;
        else
        {
            data.data()[i].flag = 0;
            ++nullCount;
        }
        vals_input.data()[i] = i;
    }
    managed_mem_t< nullable_type< long > > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    radix_sort( data.data(), vals_input.data(), count, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[vals_input.data()[i]].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[vals_input.data()[i]].flag && data.data()[vals_input.data()[i + 1]].flag );
        ASSERT_TRUE( data.data()[vals_input.data()[i]] <= data.data()[vals_input.data()[i + 1]] );
    }
}

TEST( radixsort, string_asc )
{
    string a = "aaaa";
    cout << a.length() << endl;
    a[0] = 0;
    cout << a.length() << endl;
    standard_context_t context;
//    cudaSetDevice( 0 );
//    cudaSetDevice( 1 );
    {

        AriesDataBufferSPtr orderPriority = ReadColumn( DB_NAME, "lineitem", 16 );
        size_t colSize = orderPriority->GetItemSizeInBytes();
        AriesDataBufferSPtr clone_orderPriority = orderPriority->Clone();
        int count = orderPriority->GetItemCount();
        managed_mem_t< int > vals( count, context );
        int * vals_input = vals.data();
        for( int i = 0; i < count; i++ )
        {
            vals_input[i] = i;
        }
        char* keys_input = ( char* )clone_orderPriority->GetData();
        context.timer_begin();
        radix_sort( keys_input, colSize, vals_input, count, context );
        printf( "gpu time: %3.1f\n", context.timer_end() );
        printf( "item count is: %d, colSize is: %d\n", count, colSize );

        const char* str1;
        const char* str2;
        for( int i = 0; i < count - 1; ++i )
        {
            str1 = keys_input + colSize * vals_input[i];
            str2 = keys_input + colSize * vals_input[i + 1];
            ASSERT_TRUE( strncmp( str1, str2, colSize ) <= 0 );
        }
    }
}

TEST( radixsort, string_nullsmaller_asc )
{
    standard_context_t context;
    int repeat = 1;
    AriesDataBufferSPtr orderPriority = ReadColumn( DB_NAME, "lineitem", 16 );
    size_t colSize = orderPriority->GetItemSizeInBytes();
    size_t len = orderPriority->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char * keys_input = keys.data();
    for( int i = 0; i < repeat; ++i )
    {
        memcpy( keys_input + i * len * colSize, orderPriority->GetData(), len * colSize );
    }
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();
    int nullCount = 0;
    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 10 )
            *( keys_input + i * colSize ) = 1;
        else
        {
            *( keys_input + i * colSize ) = 0;
            ++nullCount;
        }
        vals_input[i] = i;
    }

    context.timer_begin();
    radix_sort_has_null( keys_input, colSize, vals_input, len * repeat, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( *( keys_input + colSize * vals_input[i] ), 0 );

    const char* str1;
    const char* str2;
    for( int i = nullCount; i < len * repeat - 1; ++i )
    {
        str1 = keys_input + colSize * vals_input[i];
        str2 = keys_input + colSize * vals_input[i + 1];
        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
    }
}

