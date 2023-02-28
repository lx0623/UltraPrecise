#include "test_common.h"
#include "CudaAcc/AriesEngineAlgorithm.h"
#include "CudaAcc/AriesSqlOperator_sort.h"
static const char* DB_NAME = "scale_1";

// template< bounds_t bounds, typename a_keys_it, typename b_keys_it, typename comp_t >
// vector< int > merge_path_partitions( a_keys_it a, int64_t a_count, b_keys_it b, int64_t b_count, int64_t spacing, comp_t comp )
// {
//     vector< int > result;
//     int64_t num_partitions = div_up( a_count + b_count, spacing ) + 1;

//     for( int64_t index = 0; index < num_partitions; ++index )
//     {
//         int64_t diag = min(spacing * index, a_count + b_count);
//         result.push_back( merge_path< bounds >( a, a_count, b, b_count, diag, comp ) );
//     }
//     return result;
// }

// template< typename launch_arg_t = empty_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
// void mergesort( key_t* keys_input, val_t* vals_input, size_t count, comp_t< key_t, key_t > comp, context_t& context, size_t limit_in_mega_bytes )
// {
//     int device_id = context.active_device_id();
//     size_t mem_cost = ( sizeof( key_t ) + sizeof( val_t ) ) * count * 2.5;
//     size_t mem_available = limit_in_mega_bytes * 1024 * 1024;
//     size_t block_count = 1;
//     size_t round_count = 1;
//     while( mem_cost > mem_available )
//     {
//         mem_cost >>= 1;
//         block_count <<= 1;
//         ++round_count;
//     }

//     size_t block_size = count / block_count;
//     size_t tail_size = count % block_count;

//     //分块排序
//     size_t block_offset = 0;
//     for( size_t i = 0; i < block_count; ++i )
//     {
//         size_t item_count = ( i == 0 ? block_size + tail_size : block_size );

//         cudaMemPrefetchAsync( keys_input + block_offset, sizeof( key_t ) * item_count, device_id );
//         cudaMemPrefetchAsync( vals_input + block_offset, sizeof( val_t ) * item_count, device_id );
//         mergesort( keys_input + block_offset , vals_input + block_offset, item_count, comp, context );
//         block_offset += item_count;
//     }

//     if( block_count > 1 )
//     {
//         //merge
//         managed_mem_t< key_t > keys( count, context, false );
//         managed_mem_t< val_t > vals( count, context, false );
//         key_t* keys_output = keys.data();
//         val_t* vals_output = vals.data();
//         vector< merge_range_t > merge_ranges;
//         for( int round = 1; round < round_count; ++round )
//         {
//             size_t input_offset = 0;
//             size_t output_offset = 0;
//             for( int i = 0; i < ( block_count >> round ); ++i )
//             {
//                 size_t second_sorted_block_size = block_size * ( 1 << ( round - 1 ) );
//                 size_t first_sorted_block_size = second_sorted_block_size + ( i == 0 ? tail_size : 0 );
                
//                 vector< int > partitions = merge_path_partitions< bounds_lower >( keys_input + input_offset, first_sorted_block_size, 
//                                                                                  keys_input + input_offset + first_sorted_block_size, second_sorted_block_size, 
//                                                                                  block_size, comp );

//                 merge_ranges.clear();
//                 for( int p = 0; p < partitions.size() - 1; ++p )
//                 {
//                     merge_range_t r = compute_merge_range( first_sorted_block_size, second_sorted_block_size, p, block_size, 
//                                                         partitions[ p ], partitions[ p + 1 ] );
//                     merge_ranges.push_back( r );
//                 }

//                 for( const auto& r : merge_ranges )
//                 {
//                     cudaMemPrefetchAsync( keys_input + input_offset + r.a_begin, sizeof( key_t ) * r.a_count(), device_id );
//                     cudaMemPrefetchAsync( keys_input + input_offset + first_sorted_block_size + r.b_begin, sizeof( key_t ) * r.b_count(), device_id );
//                     cudaMemPrefetchAsync( vals_input + input_offset + r.a_begin, sizeof( val_t ) * r.a_count(), device_id );
//                     cudaMemPrefetchAsync( vals_input + input_offset + first_sorted_block_size + r.b_begin, sizeof( val_t ) * r.b_count(), device_id );
//                     cudaMemPrefetchAsync( keys_output + output_offset, sizeof( key_t ) * r.total(), device_id );
//                     cudaMemPrefetchAsync( vals_output + output_offset, sizeof( val_t ) * r.total(), device_id );
//                     merge( keys_input + input_offset + r.a_begin, 
//                            vals_input + input_offset + r.a_begin, r.a_count(), 
//                            keys_input + input_offset + first_sorted_block_size + r.b_begin, 
//                            vals_input + input_offset + first_sorted_block_size + r.b_begin, r.b_count(),
//                            keys_output + output_offset, vals_output + output_offset, comp, context );
//                     output_offset += r.total();
//                 }
                
//                 input_offset += first_sorted_block_size + second_sorted_block_size;
//             }
//             std::swap( keys_input, keys_output );
//             std::swap( vals_input, vals_output );
//         }
//         if( round_count % 2 == 0 )
//         {
//             cudaMemcpy( keys_output, keys_input, count * sizeof( key_t ), cudaMemcpyDefault );
//             keys.free();
//             cudaMemcpy( vals_output, vals_input, count * sizeof( val_t ), cudaMemcpyDefault );
//         }
//     }
// }



TEST( UT_mergesort, int_asc2 )
{
    standard_context_t context;
    int count = 60000000;

    std::uniform_int_distribution< int > d( 0, 1000000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< int > old = data.clone();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( data.data(), vals_input.data(), count, less_t_null_smaller< int >(), context, 100 );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( old.data()[ vals_input.data()[ i ] ] <= old.data()[ vals_input.data()[ i + 1 ] ] );
    }
}

TEST( UT_mergesort, int_asc )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< int > old = data.clone();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( data.data(), vals_input.data(), count, less_t_null_smaller< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( old.data()[ vals_input.data()[ i ] ] <= old.data()[ vals_input.data()[ i + 1 ] ] );
    }
}

struct less_t_int_as_string
{
    ARIES_HOST_DEVICE
    bool operator()( const char* a, const char* b, int len ) const
    {
        return *( int* )a < *( int* )b;
    }
};

TEST( UT_mergesort, int_asc_as_string )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< int > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( ( char* )data.data(), sizeof(int), vals_input.data(), count, less_t_int_as_string(),
            context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, decimal_asc )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< aries_acc::Decimal > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< aries_acc::Decimal > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort_large< launch_box_t< arch_52_cta< 256, 3 > > >( data.data(), vals_input.data(), count, less_t_null_smaller< aries_acc::Decimal >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

struct less_t_decimal_as_string
{
    ARIES_HOST_DEVICE
    bool operator()( const char* a, const char* b, int len ) const
    {
        return *( aries_acc::Decimal* )a < *( aries_acc::Decimal* )b;
    }
};

TEST( UT_mergesort, decimal_asc_as_string )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< aries_acc::Decimal > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< aries_acc::Decimal > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 3 > > >( ( char* )data.data(), sizeof(aries_acc::Decimal), vals_input.data(), count,
            less_t_decimal_as_string(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, long_asc )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< long > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< long > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort_large< launch_box_t< arch_52_cta< 256, 7 > > >( data.data(), vals_input.data(), count, less_t< long >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

struct less_t_long_as_string
{
    ARIES_HOST_DEVICE
    bool operator()( const char* a, const char* b, int len ) const
    {
        return *( long* )a < *( long* )b;
    }
};

TEST( UT_mergesort, long_asc_as_string )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< long > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< long > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 7 > > >( ( char* )data.data(), sizeof(long), vals_input.data(), count, less_t_long_as_string(),
            context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, int_desc )
{
    standard_context_t context;
    int count = 40000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ] = d( get_mt19937() );
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< int > old = data.clone();
    context.timer_begin();
    mergesort_large< launch_box_t< arch_52_cta< 256, 5 > > >( data.data(), vals_input.data(), count, greater_t_null_smaller< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        //ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( old.data()[ vals_input.data()[ i ] ] >= old.data()[ vals_input.data()[ i + 1 ] ] );
    }
}

TEST( UT_mergesort, nullint_smaller_asc )
{
    standard_context_t context;
    int count = 10000001;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    managed_mem_t< nullable_type< int > > old = data.clone();
    context.timer_begin();
    mergesort_large< launch_box_t< arch_52_cta< 256, 9 > > >( data.data(), vals_input.data(), count,
            less_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

struct less_t_nullable_int_as_string
{
    ARIES_DEVICE
    bool operator()( const char* a, const char* b, int len ) const
    {
        return less_t_null_smaller< nullable_type< int >, nullable_type< int > >()( *( nullable_type< int >* )a, *( nullable_type< int >* )b );
    }
};

TEST( UT_mergesort, nullint_smaller_asc_as_string )
{
    standard_context_t context;
    int count = 100000001;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    managed_mem_t< nullable_type< int > > old = data.clone();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 9 > > >( ( char* )data.data(), sizeof(nullable_type< int > ), vals_input.data(), count,
            less_t_nullable_int_as_string(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, nulllong_smaller_asc )
{
    standard_context_t context;
    int count = 67238110;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< long > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< nullable_type< long > > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort_large< launch_box_t< arch_52_cta< 256, 7 > > >( data.data(), vals_input.data(), count,
            less_t_null_smaller< nullable_type< long >, nullable_type< long > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

struct less_t_nullable_long_as_string
{
    ARIES_DEVICE
    bool operator()( const char* a, const char* b, int len ) const
    {
        return less_t_null_smaller< nullable_type< long >, nullable_type< long > >()( *( nullable_type< long >* )a, *( nullable_type< long >* )b );
    }
};

TEST( UT_mergesort, nulllong_smaller_asc_as_string )
{
    standard_context_t context;
    int count = 67238110;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< long > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< nullable_type< long > > old = data.clone();
    data.PrefetchToGpu();
    vals_input.PrefetchToGpu();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 3 > > >( ( char* )data.data(), sizeof(nullable_type< long > ), vals_input.data(), count,
            less_t_nullable_long_as_string(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, nullint_bigger_asc )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< nullable_type< int > > old = data.clone();
    context.timer_begin();
    mergesort_large( data.data(), vals_input.data(), count, less_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = count - nullCount; i < count; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = 0; i < count - nullCount - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] <= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, nullint_bigger_desc )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< nullable_type< int > > old = data.clone();
    context.timer_begin();
    mergesort_large( data.data(), vals_input.data(), count, greater_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = nullCount; i < count - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] >= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, nullint_smaller_desc )
{
    standard_context_t context;
    int count = 10000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        data.data()[ i ].value = d( get_mt19937() );
        if( i % 10 )
            data.data()[ i ].flag = 1;
        else
        {
            data.data()[ i ].flag = 0;
            ++nullCount;
        }
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< nullable_type< int > > old = data.clone();
    context.timer_begin();
    mergesort_large( data.data(), vals_input.data(), count, greater_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = count - nullCount; i < count; ++i )
        ASSERT_EQ( data.data()[ i ].flag, 0 );
    for( int i = 0; i < count - nullCount - 1; ++i )
    {
        ASSERT_TRUE( data.data()[ i ].flag && data.data()[ i + 1 ].flag );
        ASSERT_EQ( old.data()[ vals_input.data()[ i ] ], data.data()[ i ] );
        ASSERT_TRUE( data.data()[ i ] >= data.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, string_asc )
{
    standard_context_t context;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    int colSize = shipMode->GetItemSizeInBytes();
    AriesDataBufferSPtr clone_shipMode = shipMode->Clone();
    int count = shipMode->GetItemCount();
    managed_mem_t< int > vals( count, context );
    int * vals_input = vals.data();
    for( int i = 0; i < count; i++ )
    {
        vals_input[ i ] = i;
    }

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( ( char* )clone_shipMode->GetData(), colSize, vals_input, count, less_t_str< false, false >(),
            context, 100 );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", count, colSize );

    const char* str1;
    const char* str2;
    const char* str3;
    for( int i = 0; i < count - 1; ++i )
    {
        str1 = ( char* )clone_shipMode->GetData() + colSize * i;
        str2 = str1 + colSize;
        str3 = ( char* )shipMode->GetData() + vals_input[ i ] * colSize;
        ASSERT_TRUE( strncmp( str1, str2, colSize ) <= 0 );
        ASSERT_TRUE( strncmp( str1, str3, colSize ) == 0 );
    }
}

TEST( UT_mergesort, nullstring_smaller_asc )
{
    standard_context_t context;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    int colSize = shipMode->GetItemSizeInBytes();
    int count = shipMode->GetItemCount();
    managed_mem_t< int > vals( count, context );
    int * vals_input = vals.data();
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
            *( shipMode->GetData() + i * colSize ) = 1;
        else
        {
            *( shipMode->GetData() + i * colSize ) = 0;
            ++nullCount;
        }
        vals_input[ i ] = i;
    }

    AriesDataBufferSPtr clone_shipMode = shipMode->Clone();

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( ( char* )clone_shipMode->GetData(), colSize, vals_input, count,
            less_t_str_null_smaller< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", count, colSize );
    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( *( clone_shipMode->GetData() + colSize * i ), 0 );
    const char* str1;
    const char* str2;
    const char* str3;
    for( int i = nullCount; i < count - 1; ++i )
    {
        str1 = ( char* )clone_shipMode->GetData() + colSize * i;
        str2 = str1 + colSize;
        str3 = ( char* )shipMode->GetData() + vals_input[ i ] * colSize;
        ASSERT_TRUE( *str1 && *str2 );
        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
        ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
    }
}

TEST( UT_mergesort, nullstring_bigger_asc )
{
    standard_context_t context;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    int colSize = shipMode->GetItemSizeInBytes();
    int count = shipMode->GetItemCount();
    managed_mem_t< int > vals( count, context );
    int * vals_input = vals.data();
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
            *( shipMode->GetData() + i * colSize ) = 1;
        else
        {
            *( shipMode->GetData() + i * colSize ) = 0;
            ++nullCount;
        }
        vals_input[ i ] = i;
    }

    AriesDataBufferSPtr clone_shipMode = shipMode->Clone();

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( ( char* )clone_shipMode->GetData(), colSize, vals_input, count,
            less_t_str_null_bigger< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", count, colSize );

    for( int i = count - nullCount; i < count; ++i )
        ASSERT_EQ( *( ( char* )clone_shipMode->GetData() + colSize * i ), 0 );
    const char* str1;
    const char* str2;
    const char* str3;
    for( int i = nullCount; i < count - nullCount - 1; ++i )
    {
        str1 = ( char* )clone_shipMode->GetData() + colSize * i;
        str2 = str1 + colSize;
        str3 = ( char* )shipMode->GetData() + vals_input[ i ] * colSize;
        ASSERT_TRUE( *str1 && *str2 );
        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
        ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
    }
}

TEST( UT_mergesort, nullstring_bigger_desc )
{
    standard_context_t context;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    int colSize = shipMode->GetItemSizeInBytes();
    int count = shipMode->GetItemCount();
    managed_mem_t< int > vals( count, context );
    int * vals_input = vals.data();
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
            *( shipMode->GetData() + i * colSize ) = 1;
        else
        {
            *( shipMode->GetData() + i * colSize ) = 0;
            ++nullCount;
        }
        vals_input[ i ] = i;
    }

    AriesDataBufferSPtr clone_shipMode = shipMode->Clone();

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( ( char* )clone_shipMode->GetData(), colSize, vals_input, count,
            greater_t_str_null_bigger< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", count, colSize );

    for( int i = 0; i < nullCount; ++i )
        ASSERT_EQ( *( ( char* )clone_shipMode->GetData() + colSize * i ), 0 );
    const char* str1;
    const char* str2;
    const char* str3;
    for( int i = nullCount; i < count - 1; ++i )
    {
        str1 = ( char* )clone_shipMode->GetData() + colSize * i;
        str2 = str1 + colSize;
        str3 = ( char* )shipMode->GetData() + vals_input[ i ] * colSize;
        ASSERT_TRUE( *str1 && *str2 );
        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) >= 0 );
        ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
    }
}

TEST( UT_mergesort, nullstring_smaller_desc )
{
    standard_context_t context;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    int colSize = shipMode->GetItemSizeInBytes();
    int count = shipMode->GetItemCount();
    managed_mem_t< int > vals( count, context );
    int * vals_input = vals.data();
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
            *( shipMode->GetData() + i * colSize ) = 1;
        else
        {
            *( shipMode->GetData() + i * colSize ) = 0;
            ++nullCount;
        }
        vals_input[ i ] = i;
    }

    AriesDataBufferSPtr clone_shipMode = shipMode->Clone();

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( ( char* )clone_shipMode->GetData(), colSize, vals_input, count,
            greater_t_str_null_smaller< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", count, colSize );

    for( int i = count - nullCount; i < count; ++i )
        ASSERT_EQ( *( ( char* )clone_shipMode->GetData() + colSize * i ), 0 );
    const char* str1;
    const char* str2;
    const char* str3;
    for( int i = nullCount; i < count - nullCount - 1; ++i )
    {
        str1 = ( char* )clone_shipMode->GetData() + colSize * i;
        str2 = str1 + colSize;
        str3 = ( char* )shipMode->GetData() + vals_input[ i ] * colSize;
        ASSERT_TRUE( *str1 && *str2 );
        ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) >= 0 );
        ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
    }
}

TEST( UT_mergesort, string_desc )
{
    standard_context_t context;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    int colSize = shipMode->GetItemSizeInBytes();
    int count = shipMode->GetItemCount();
    managed_mem_t< int > vals( count, context );
    int * vals_input = vals.data();
    int nullCount = 0;
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
            *( shipMode->GetData() + i * colSize ) = 1;
        else
        {
            *( shipMode->GetData() + i * colSize ) = 0;
            ++nullCount;
        }
        vals_input[ i ] = i;
    }

    AriesDataBufferSPtr clone_shipMode = shipMode->Clone();

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( ( char* )clone_shipMode->GetData(), colSize, vals_input, count,
            greater_t_str< false, false >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", count, colSize );

    const char* str1;
    const char* str2;
    const char* str3;
    for( int i = 0; i < count - 1; ++i )
    {
        str1 = ( char* )clone_shipMode->GetData() + colSize * i;
        str2 = str1 + colSize;
        str3 = ( char* )shipMode->GetData() + vals_input[ i ] * colSize;
        ASSERT_TRUE( strncmp( str1, str2, colSize ) >= 0 );
        ASSERT_TRUE( strncmp( str1, str3, colSize ) == 0 );
    }
}

TEST( UT_mergesort, materialize_fewer_index )
{
    standard_context_t context;
    int div = 3;
    int count = 60000000;
    aries_engine::AriesColumnSPtr column = std::make_shared< aries_engine::AriesColumn >();
    AriesDataType dataType { AriesValueType::INT32, 1 };
    AriesDataBufferSPtr buffer = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false }, count );
    column->AddDataBuffer( buffer );
    int indices_count = count / div;
    aries_engine::AriesIndicesSPtr indices = std::make_shared< aries_engine::AriesIndices >();
    AriesIndicesArraySPtr indexArray = std::make_shared< AriesIndicesArray >( indices_count );
    indices->AddIndices( indexArray );

    aries_engine::AriesColumnReferenceSPtr columnRef = std::make_shared< aries_engine::AriesColumnReference >( column );
    columnRef->SetIndices( indices );

    managed_mem_t< int > vals( indices_count, context );
    managed_mem_t< int > temp( indices_count, context );
    vector< int > hostIndices;
    int * vals_input = vals.data();
    int* pIndex = ( int* )indexArray->GetData();
    int* pData = ( int* )buffer->GetData();
    for( int i = 0; i < indices_count; i++ )
    {
        hostIndices.push_back( i * div );
        vals_input[ i ] = i;
    }
    indexArray->CopyFromHostMem( hostIndices.data(), indexArray->GetTotalBytes() );
    //std::random_shuffle( pData, pData + count );
    //std::random_shuffle( pIndex, pIndex + indices_count );
    vals.PrefetchToGpu();

    context.timer_begin();
    auto input_buffer = columnRef->GetDataBuffer();
    int* input_data = ( int* )input_buffer->GetData();
    input_buffer->PrefetchToGpu();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( input_data, vals_input, indices_count, less_t_null_smaller< int >(), context );
    printf( "sort gpu time: %3.1f\n", context.timer_end() );

    for( int i = 0; i < indices_count; i++ )
    {
        vals_input[ i ] = i;
    }
    vals.PrefetchToGpu();

    context.timer_begin();
    temp.PrefetchToGpu();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( pData, pIndex, ( int* )temp.data(), vals_input, indices_count, less_t_null_smaller< int >(),
            context );
    printf( "sort using indices gpu time: %3.1f\n", context.timer_end() );

    printf( "item count is: %d\n", indices_count );
    for( int i = 0; i < indices_count - 1; ++i )
    {
        ASSERT_TRUE( temp.data()[ i ] <= temp.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, materialize )
{
    standard_context_t context;
    int count = 20000000;
    aries_engine::AriesColumnSPtr column = std::make_shared< aries_engine::AriesColumn >();
    AriesDataType dataType { AriesValueType::INT32, 1 };
    AriesDataBufferSPtr buffer = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false }, count );
    column->AddDataBuffer( buffer );

    aries_engine::AriesIndicesSPtr indices = std::make_shared< aries_engine::AriesIndices >();
    AriesIndicesArraySPtr indexArray = std::make_shared< AriesIndicesArray >( count );
    indices->AddIndices( indexArray );

    aries_engine::AriesColumnReferenceSPtr columnRef = std::make_shared< aries_engine::AriesColumnReference >( column );
    columnRef->SetIndices( indices );

    managed_mem_t< int > vals( count, context );
    managed_mem_t< int > temp( count, context );
    int * vals_input = vals.data();
    int* pIndex = ( int* )indexArray->GetData();
    vector< int > hostIndices;
    int* pData = ( int* )buffer->GetData();
    for( int i = 0; i < count; i++ )
    {
        pData[ i ] = i;
        hostIndices.push_back( i );
        vals_input[ i ] = i;
    }
    std::random_shuffle( pData, pData + count );
    std::random_shuffle( hostIndices.data(), hostIndices.data() + count );

    indexArray->CopyFromHostMem( hostIndices.data(), indexArray->GetTotalBytes() );
    context.timer_begin();
    auto input_buffer = columnRef->GetDataBuffer();
    int* input_data = ( int* )input_buffer->GetData();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( input_data, vals_input, count, less_t_null_smaller< int >(), context );
    printf( "sort gpu time: %3.1f\n", context.timer_end() );

    for( int i = 0; i < count; i++ )
    {
        vals_input[ i ] = i;
    }

    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( pData, pIndex, ( int* )temp.data(), vals_input, count, less_t_null_smaller< int >(),
            context );
    printf( "sort using indices gpu time: %3.1f\n", context.timer_end() );

    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( temp.data()[ i ] <= temp.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, multiblock_int_asc )
{
    standard_context_t context;
    int count = 20000000;
    int block_count = 10;
    int block_size = count / block_count;
    managed_mem_t< int* > all_blocks( block_count, context );
    managed_mem_t< int64_t > prefix_sum( block_count, context );

    int64_t sum = 0;
    vector< managed_mem_t< int > > blocks;
    std::uniform_int_distribution< int > d( 0, 10000000 );
    int index = 0;
    for( int i = 0; i < block_count; ++i )
    {
        managed_mem_t< int > data( block_size, context );
        for( int j = 0; j < block_size; ++j )
        {
            data.data()[ j ] = d( get_mt19937() );
        }
        all_blocks.data()[ i ] = data.data();
        blocks.push_back( std::move( data ) );
        prefix_sum.data()[ i ] = sum;
        sum += block_size;
    }

    managed_mem_t< int > vals_input( count, context );

    for( int i = 0; i < count; i++ )
    {
        vals_input.data()[ i ] = i;
    }
    managed_mem_t< int > output_buffer( count, context );
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( all_blocks.data(), prefix_sum.data(), block_count, output_buffer.data(), vals_input.data(),
            count, less_t_null_smaller< int >(), context );
    printf( "multi block gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    for( int i = 0; i < block_count; ++i )
    {
        blocks[ i ].PrefetchToCpu();
    }

    managed_mem_t< int > input_buffer( count, context );
    for( int i = 0; i < count; i++ )
    {
        vals_input.data()[ i ] = i;
    }
    context.timer_begin();
    for( int i = 0; i < block_count; ++i )
    {
        cudaMemcpy( input_buffer.data() + i * block_size, all_blocks.data()[ i ], sizeof(int) * block_size, cudaMemcpyDefault );
    }
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( input_buffer.data(), vals_input.data(), count, less_t_null_smaller< int >(), context );
    printf( "single block gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( output_buffer.data()[ i ] <= output_buffer.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, multiblock_materialize )
{
    standard_context_t context;
    int count = 20000000;
    int block_count = 10;
    int block_size = count / block_count;
    managed_mem_t< int* > all_blocks( block_count, context );
    managed_mem_t< int64_t > prefix_sum( block_count, context );
    managed_mem_t< int > indices( count, context );
    int* pIndex = indices.data();
    int64_t sum = 0;
    vector< managed_mem_t< int > > blocks;
    std::uniform_int_distribution< int > d( 0, 10000000 );

    for( int i = 0; i < block_count; ++i )
    {
        managed_mem_t< int > data( block_size, context );
        for( int j = 0; j < block_size; ++j )
        {
            data.data()[ j ] = d( get_mt19937() );
        }
        all_blocks.data()[ i ] = data.data();
        blocks.push_back( std::move( data ) );
        prefix_sum.data()[ i ] = sum;
        sum += block_size;
    }

    managed_mem_t< int > vals_input( count, context );

    for( int i = 0; i < count; i++ )
    {
        vals_input.data()[ i ] = i;
        pIndex[ i ] = i;
    }
    std::random_shuffle( pIndex, pIndex + count );
    managed_mem_t< int > output_buffer( count, context );
    output_buffer.PrefetchToCpu();
    context.timer_begin();
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( all_blocks.data(), prefix_sum.data(), block_count, pIndex, false, ( int* )nullptr,
            vals_input.data(), count, less_t_null_smaller< int >(), context );
    printf( "multi block gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( output_buffer.data()[ i ] <= output_buffer.data()[ i + 1 ] );
    }

    for( int i = 0; i < block_count; ++i )
    {
        blocks[ i ].PrefetchToCpu();
    }

    managed_mem_t< int > input_buffer( count, context );
    for( int i = 0; i < count; i++ )
    {
        vals_input.data()[ i ] = i;
    }
    indices.PrefetchToCpu();
    output_buffer.PrefetchToCpu();
    context.timer_begin();
    for( int i = 0; i < block_count; ++i )
    {
        cudaMemcpy( input_buffer.data() + i * block_size, all_blocks.data()[ i ], sizeof(int) * block_size, cudaMemcpyDefault );
    }
    mergesort< launch_box_t< arch_52_cta< 256, 5 > > >( input_buffer.data(), pIndex, output_buffer.data(), vals_input.data(), count,
            less_t_null_smaller< int >(), context );
    printf( "single block gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( output_buffer.data()[ i ] <= output_buffer.data()[ i + 1 ] );
    }
}

TEST( UT_mergesort, newapi_int )
{
    standard_context_t context;
    int count = 20000000;
    aries_engine::AriesColumnSPtr column = std::make_shared< aries_engine::AriesColumn >();
    AriesDataType dataType { AriesValueType::INT32, 1 };
    AriesDataBufferSPtr buffer = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false }, count );
    column->AddDataBuffer( buffer );

    aries_engine::AriesIndicesSPtr indices = std::make_shared< aries_engine::AriesIndices >();
    AriesIndicesArraySPtr indexArray = std::make_shared< AriesIndicesArray >( count );
    indices->AddIndices( indexArray );

    aries_engine::AriesColumnReferenceSPtr columnRef = std::make_shared< aries_engine::AriesColumnReference >( column );
    columnRef->SetIndices( indices );

    AriesInt32ArraySPtr associated = std::make_shared< AriesInt32Array >( count );

    int * vals_input = ( int* )associated->GetData();
    int* pIndex = ( int* )indexArray->GetData();
    vector< int > hostIndices;
    int* pData = ( int* )buffer->GetData();
    for( int i = 0; i < count; i++ )
    {
        pData[ i ] = i;
        hostIndices.push_back( i );
    }
    associated->CopyFromHostMem( hostIndices.data(), indexArray->GetTotalBytes() );
    std::random_shuffle( pData, pData + count );
    std::random_shuffle( hostIndices.data(), hostIndices.data() + count );

    indexArray->CopyFromHostMem( hostIndices.data(), indexArray->GetTotalBytes() );

    context.timer_begin();
    AriesDataBufferSPtr output = SortColumn( columnRef, AriesOrderByType::ASC, associated );
    printf( "multi block gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    int* pOutput = ( int* )output->GetData();
    for( int i = 0; i < count - 1; ++i )
    {
        ASSERT_TRUE( pOutput[ i ] <= pOutput[ i + 1 ] );
    }
}
