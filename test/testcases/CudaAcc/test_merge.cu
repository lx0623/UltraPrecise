#include "test_common.h"
using namespace aries_acc;
static const char* DB_NAME = "scale_1";
TEST( merge, int_asc )
{
    standard_context_t context;

    int count = 1000000000;

    int a_count = count / 2;
    int b_count = count - a_count;

    mem_t< int > a = fill_random( 0, count, a_count, true, context );
    mem_t< int > b = fill_random( 0, count, b_count, true, context );
    mem_t< int > c( count );

    context.timer_begin();
    merge( a.data(), a_count, b.data(), b_count, c.data(), less_t< int >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );

    // // Download the results.
    // std::vector< int > a_host = from_mem( a );
    // std::vector< int > b_host = from_mem( b );
    // std::vector< int > c_host = from_mem( c );

    // // Do merge on the host and compare.
    // std::vector< int > c2( count );
    // std::merge( a_host.begin(), a_host.end(), b_host.begin(), b_host.end(), c2.begin() );

    // bool success = c2 == c_host;
    // ASSERT_TRUE( success );
}

TEST( merge, int_asc2 )
{
    standard_context_t context;

    int count = 1000000000;
    int spacing = 100000000;
    int a_count = count / 2;
    int b_count = count - a_count;

    mem_t< int > a = fill_random( 0, count, a_count, true, context );
    mem_t< int > b = fill_random( 0, count, b_count, true, context );
    mem_t< int > c( count );

    context.timer_begin();
    mem_t< int > partitions = merge_path_partitions< bounds_lower >( a.data(), a_count, b.data(), b_count, spacing, less_t< int >(), context );

    // cout<<"a:";
    // for( int i = 0; i < a_count; ++i )
    //     cout<<a.data()[ i ]<<", ";
    // cout<<endl;

    // cout<<"b:";
    // for( int i = 0; i < b_count; ++i )
    //     cout<<b.data()[ i ]<<", ";
    // cout<<endl;
    cout<<"partitions:";
    for( int i = 0; i < partitions.size(); ++i )
        cout<<partitions.data()[ i ]<<", ";
    cout<<endl;

    vector< merge_range_t > merge_ranges;
    cout<<"merge_range_t:"<<endl;
    for( int i = 0; i < partitions.size() - 1; ++i )
    {
        merge_range_t r = compute_merge_range( a_count, b_count, i, spacing, partitions.data()[ i ], partitions.data()[ i + 1 ] );
        merge_ranges.push_back( r );
        cout<<"a_begin="<<r.a_begin<<", a_end="<<r.a_end<<", b_begin="<<r.b_begin<<", b_end="<<r.b_end<<endl;
    }
    // b_count = 0;
    // context.timer_begin();
    // merge( a.data(), a_count, b.data(), b_count, c.data(), less_t< int >(), context );
    // printf( "gpu time: %3.1f\n", context.timer_end() );
    // cout<<"c:";
    // for( int i = 0; i < a_count + b_count; ++i )
    //     cout<<c.data()[i]<<", ";
    // cout<<endl;

    
    size_t offset = 0;
    for( const auto& r : merge_ranges )
    {
        merge( a.data() + r.a_begin, r.a_count(), b.data() + r.b_begin, r.b_count(), c.data() + offset, less_t< int >(), context );
        offset += spacing;
    }
    printf( "gpu time: %3.1f\n", context.timer_end() );

    for( int i = 0; i < a_count + b_count - 1; ++i )
        ASSERT_TRUE( c.data()[i] <= c.data()[i+1]);
    // cout<<"c:";
    // for( int i = 0; i < a_count + b_count; ++i )
    //     cout<<c.data()[i]<<", ";
    // cout<<endl;
    // // Download the results.
    // std::vector< int > a_host = from_mem( a );
    // std::vector< int > b_host = from_mem( b );
    // std::vector< int > c_host = from_mem( c );

    // // Do merge on the host and compare.
    // std::vector< int > c2( count );
    // std::merge( a_host.begin(), a_host.end(), b_host.begin(), b_host.end(), c2.begin() );

    // bool success = c2 == c_host;
    // ASSERT_TRUE( success );
}

TEST( merge, string_asc )
{
    standard_context_t context;

    int repeat = 2;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    mem_t< char > keys( len * colSize * repeat );
    mem_t< char > keys_c( len * colSize * repeat );
    char * keys_input = keys.data();
    char * keys_output = keys_c.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );
    cudaMemcpy( keys_input + len * repeat / 2, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    mem_t< char > old = keys.clone();
    char * old_input = old.data();
    managed_mem_t< int > vals( len * repeat, context );
    managed_mem_t< int > vals_c( len * repeat, context );
    int * vals_input = vals.data();
    int * vals_output = vals_c.data();
    for( int i = 0; i < len * repeat; i++ )
    {
        vals_input[i] = i;
    }
    managed_mem_t< int > vals_old = vals.clone();

    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat / 2, less_t_str< false, false >(), context );
    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input + ( len * repeat / 2 ) * colSize, colSize, vals_input + len * repeat / 2,
            len * repeat / 2, less_t_str< false, false >(), context );
    context.timer_begin();
    merge( keys_input, vals_input, len * repeat / 2, keys_input + ( len * repeat / 2 ) * colSize, vals_input + len * repeat / 2, len * repeat / 2,
            keys_output, vals_output, colSize, less_t_str< false, false >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    mergesort< launch_box_t< arch_52_cta< 256, 6 > > >( old_input, colSize, vals_old.data(), len * repeat, less_t_str< false, false >(), context );

    for( int i = 0; i < len * repeat; ++i )
    {
        ASSERT_EQ( vals_output[i], vals_old.data()[i] );
    }
}
