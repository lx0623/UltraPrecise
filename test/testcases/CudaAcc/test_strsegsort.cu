#include "test_common.h"
static const char* DB_NAME = "scale_1";
TEST( segsort, string_asc )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    size_t len = shipMode->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char* keys_input = keys.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > old = keys.clone();
    char * old_input = old.data();
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();
    for( int i = 0; i < len * repeat; i++ )
    {
        vals_input[i] = i;
    }

    int num_segments = div_up( len * repeat - 1, 100ul );

    std::uniform_int_distribution< int > d( 0, len * repeat - 1 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );
    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, segments, num_segments,
            less_t_str< false, false >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );
    const char* str1;
    const char* str2;
    const char* str3;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            str1 = keys_input + colSize * i;
            str2 = str1 + colSize;
            str3 = old_input + vals_input[i] * colSize;
            ASSERT_TRUE( strncmp( str1, str2, colSize ) <= 0 );
            ASSERT_TRUE( strncmp( str1, str3, colSize ) == 0 );
        }
    }
    printf( "done\n" );
}

TEST( segsort, nullstring_smaller_asc )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char* keys_input = keys.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 10 )
            *( keys_input + i * colSize ) = 1;
        else
        {
            *( keys_input + i * colSize ) = 0;
        }
        vals_input[i] = i;
    }
    managed_mem_t< char > old = keys.clone();
    char * old_input = old.data();

    int num_segments = div_up( len * repeat - 1, 100 );
    std::uniform_int_distribution< int > d( 0, len * repeat - 1 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, segments, num_segments,
            less_t_str_null_smaller< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str1;
    const char* str2;
    const char* str3;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            str1 = keys_input + colSize * i;
            str2 = str1 + colSize;
            str3 = old_input + vals_input[i] * colSize;
            if( *str1 )
            {
                ASSERT_TRUE( *str2 );
                ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
                ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
            }
        }
    }
}

TEST( segsort, nullstring_bigger_asc )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char* keys_input = keys.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 10 )
            *( keys_input + i * colSize ) = 1;
        else
        {
            *( keys_input + i * colSize ) = 0;
        }
        vals_input[i] = i;
    }
    managed_mem_t< char > old = keys.clone();
    char * old_input = old.data();

    int num_segments = div_up( len * repeat - 1, 100 );
    std::uniform_int_distribution< int > d( 0, len * repeat - 1 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, segments, num_segments,
            less_t_str_null_bigger< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str1;
    const char* str2;
    const char* str3;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            str1 = keys_input + colSize * i;
            str2 = str1 + colSize;
            str3 = old_input + vals_input[i] * colSize;
            if( *str2 )
            {
                ASSERT_TRUE( *str1 );
                ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) <= 0 );
                ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
            }
            if( !*str1 )
                ASSERT_TRUE( !*str2 );
        }
    }
}

TEST( segsort, nullstring_bigger_desc )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char* keys_input = keys.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 10 )
            *( keys_input + i * colSize ) = 1;
        else
        {
            *( keys_input + i * colSize ) = 0;
        }
        vals_input[i] = i;
    }
    managed_mem_t< char > old = keys.clone();
    char * old_input = old.data();

    int num_segments = div_up( len * repeat - 1, 100 );
    std::uniform_int_distribution< int > d( 0, len * repeat - 1 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, segments, num_segments,
            greater_t_str_null_bigger< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str1;
    const char* str2;
    const char* str3;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            str1 = keys_input + colSize * i;
            str2 = str1 + colSize;
            str3 = old_input + vals_input[i] * colSize;
            if( *str1 )
            {
                ASSERT_TRUE( *str2 );
                ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) >= 0 );
                ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
            }
        }
    }
}

TEST( segsort, nullstring_smaller_desc )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char* keys_input = keys.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 10 )
            *( keys_input + i * colSize ) = 1;
        else
        {
            *( keys_input + i * colSize ) = 0;
        }
        vals_input[i] = i;
    }
    managed_mem_t< char > old = keys.clone();
    char * old_input = old.data();

    int num_segments = div_up( len * repeat - 1, 100 );
    std::uniform_int_distribution< int > d( 0, len * repeat - 1 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, segments, num_segments,
            greater_t_str_null_smaller< true, true >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str1;
    const char* str2;
    const char* str3;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            str1 = keys_input + colSize * i;
            str2 = str1 + colSize;
            str3 = old_input + vals_input[i] * colSize;
            if( *str2 )
            {
                ASSERT_TRUE( *str1 );
                ASSERT_TRUE( strncmp( str1 + 1, str2 + 1, colSize - 1 ) >= 0 );
                ASSERT_TRUE( strncmp( str1 + 1, str3 + 1, colSize - 1 ) == 0 );
            }
            if( !*str1 )
                ASSERT_TRUE( !*str2 );
        }
    }
}

TEST( segsort, string_desc )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > keys( len * colSize * repeat, context );
    char* keys_input = keys.data();
    cudaMemcpy( keys_input, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );
    managed_mem_t< char > old = keys.clone();
    char * old_input = old.data();
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();
    for( int i = 0; i < len * repeat; i++ )
    {
        vals_input[i] = i;
    }

    int num_segments = div_up( len * repeat - 1, 100 );
    std::uniform_int_distribution< int > d( 0, len * repeat - 1 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 6 > > >( keys_input, colSize, vals_input, len * repeat, segments, num_segments,
            greater_t_str< false, false >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );
    const char* str1;
    const char* str2;
    const char* str3;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            str1 = keys_input + colSize * i;
            str2 = str1 + colSize;
            str3 = old_input + vals_input[i] * colSize;
            ASSERT_TRUE( strncmp( str1, str2, colSize ) >= 0 );
            ASSERT_TRUE( strncmp( str1, str3, colSize ) == 0 );
        }
    }
    printf( "done\n" );
}

TEST( segsort, int_asc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, count - 1 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
        vals_input.data()[i] = i;
    }
    managed_mem_t< int > old = data.clone();

    int num_segments = div_up( count, 100 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 11 > > >( data.data(), vals_input.data(), count, segments, num_segments, less_t< int >(),
            context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    int a;
    int b;
    int c;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            a = data.data()[i];
            b = data.data()[i + 1];
            c = old.data()[vals_input.data()[i]];
            ASSERT_TRUE( a <= b );
            ASSERT_TRUE( a == c );
        }
    }
    printf( "done\n" );
}

TEST( segsort, int_desc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, count - 1 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > vals_input( count, context );
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = d( get_mt19937() );
        vals_input.data()[i] = i;
    }
    managed_mem_t< int > old = data.clone();

    int num_segments = div_up( count, 100 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 11 > > >( data.data(), vals_input.data(), count, segments, num_segments, greater_t< int >(),
            context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    int a;
    int b;
    int c;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            a = data.data()[i];
            b = data.data()[i + 1];
            c = old.data()[vals_input.data()[i]];
            ASSERT_TRUE( a >= b );
            ASSERT_TRUE( a == c );
        }
    }
    printf( "done\n" );
}

TEST( segsort, nullint_smaller_asc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
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
    managed_mem_t< nullable_type< int > > old = data.clone();

    int num_segments = div_up( count, 100 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 11 > > >( data.data(), vals_input.data(), count, segments, num_segments,
            less_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int > a;
    nullable_type< int > b;
    nullable_type< int > c;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            a = data.data()[i];
            b = data.data()[i + 1];
            c = old.data()[vals_input.data()[i]];
            if( !a.is_null() )
            {
                ASSERT_TRUE( !b.is_null() );
                ASSERT_TRUE( a <= b );
                ASSERT_TRUE( a == c );
            }
        }
    }
}

TEST( segsort, nullint_bigger_asc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
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
    managed_mem_t< nullable_type< int > > old = data.clone();

    int num_segments = div_up( count, 100 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 11 > > >( data.data(), vals_input.data(), count, segments, num_segments,
            less_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int > a;
    nullable_type< int > b;
    nullable_type< int > c;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            a = data.data()[i];
            b = data.data()[i + 1];
            c = old.data()[vals_input.data()[i]];
            if( !b.is_null() )
            {
                ASSERT_TRUE( !a.is_null() );
                ASSERT_TRUE( a <= b );
                ASSERT_TRUE( a == c );
            }
            if( a.is_null() )
                ASSERT_TRUE( b.is_null() );
        }
    }
}

TEST( segsort, nullint_bigger_desc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
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
    managed_mem_t< nullable_type< int > > old = data.clone();

    int num_segments = div_up( count, 100 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 11 > > >( data.data(), vals_input.data(), count, segments, num_segments,
            greater_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int > a;
    nullable_type< int > b;
    nullable_type< int > c;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            a = data.data()[i];
            b = data.data()[i + 1];
            c = old.data()[vals_input.data()[i]];
            if( !a.is_null() )
            {
                ASSERT_TRUE( !b.is_null() );
                ASSERT_TRUE( a >= b );
                ASSERT_TRUE( a == c );
            }
        }
    }
}

TEST( segsort, nullint_smaller_desc )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< nullable_type< int > > data( count, context );
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
    managed_mem_t< nullable_type< int > > old = data.clone();

    int num_segments = div_up( count, 100 );
    managed_mem_t< int > segs( num_segments, context );
    for( int i = 0; i < num_segments; i++ )
    {
        segs.data()[i] = d( get_mt19937() );
    }
    int* segments = segs.data();
    mergesort( segments, num_segments, less_t< int >(), context );

    context.timer_begin();
    segmented_sort< launch_box_t< arch_52_cta< 256, 11 > > >( data.data(), vals_input.data(), count, segments, num_segments,
            greater_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int > a;
    nullable_type< int > b;
    nullable_type< int > c;
    for( int j = 0; j < num_segments - 1; ++j )
    {
        for( int i = segments[j]; i < segments[j + 1] - 1; ++i )
        {
            a = data.data()[i];
            b = data.data()[i + 1];
            c = old.data()[vals_input.data()[i]];
            if( !b.is_null() )
            {
                ASSERT_TRUE( !a.is_null() );
                ASSERT_TRUE( a >= b );
                ASSERT_TRUE( a == c );
            }
            if( a.is_null() )
                ASSERT_TRUE( b.is_null() );
        }
    }
}
