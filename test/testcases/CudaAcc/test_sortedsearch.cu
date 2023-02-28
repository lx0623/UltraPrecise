
#include "test_common.h"
static const char* DB_NAME = "scale_1";

TEST( sortedsearch, int_asc_lower )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< int > needles( count, context );
    managed_mem_t< long > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        needles.data()[i] = d( get_mt19937() );
        haystack.data()[i] = d( get_mt19937() );
    }
    mergesort( needles.data(), count, less_t< int >(), context );
    mergesort( haystack.data(), count, less_t< long >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< less_t, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    int* needles_host = needles.data();
    long* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        int needle = needles_host[i];
        int index = indices_host[i];

        ASSERT_TRUE( index <= 0 || needle > haystack_host[index - 1] );
        ASSERT_TRUE( index >= count || needle <= haystack_host[index] );
    }
}

TEST( sortedsearch, int_asc_upper )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< int > needles( count, context );
    managed_mem_t< int > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        needles.data()[i] = d( get_mt19937() );
        haystack.data()[i] = d( get_mt19937() );
    }
    mergesort( needles.data(), count, less_t< int >(), context );
    mergesort( haystack.data(), count, less_t< int >(), context );
    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< less_t, bounds_upper, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    int* needles_host = needles.data();
    int* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        int needle = needles_host[i];
        int index = indices_host[i];

        ASSERT_TRUE( index <= 0 || needle >= haystack_host[index - 1] );
        ASSERT_TRUE( index >= count || needle < haystack_host[index] );
    }
}

TEST( sortedsearch, int_desc_lower )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< int > needles( count, context );
    managed_mem_t< int > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        needles.data()[i] = d( get_mt19937() );
        haystack.data()[i] = d( get_mt19937() );
    }
    mergesort( needles.data(), count, greater_t< int >(), context );
    mergesort( haystack.data(), count, greater_t< int >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< greater_t, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    int* needles_host = needles.data();
    int* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        int needle = needles_host[i];
        int index = indices_host[i];

        ASSERT_TRUE( index <= 0 || needle < haystack_host[index - 1] );
        ASSERT_TRUE( index >= count || needle >= haystack_host[index] );
    }
}

TEST( sortedsearch, int_desc_upper )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< int > needles( count, context );
    managed_mem_t< int > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        needles.data()[i] = d( get_mt19937() );
        haystack.data()[i] = d( get_mt19937() );
    }
    mergesort( needles.data(), count, greater_t< int >(), context );
    mergesort( haystack.data(), count, greater_t< int >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< greater_t, bounds_upper, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    int* needles_host = needles.data();
    int* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        int needle = needles_host[i];
        int index = indices_host[i];

        ASSERT_TRUE( index <= 0 || needle <= haystack_host[index - 1] );
        ASSERT_TRUE( index >= count || needle > haystack_host[index] );
    }
}

TEST( sortedsearch, nullint_asc_lower_smaller )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< long > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, less_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, less_t_null_smaller< nullable_type< long >, nullable_type< long > >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< less_t_null_smaller, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< long >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];

        if( !needle.is_null() )
        {
            if( !haystack_host[index - 1].is_null() )
                ASSERT_TRUE( index <= 0 || needle > haystack_host[index - 1] );
            ASSERT_TRUE( index >= count || needle <= haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_asc_upper_smaller )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, less_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, less_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< less_t_null_smaller, bounds_upper, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];
        if( !needle.is_null() )
        {
            if( !haystack_host[index - 1].is_null() )
                ASSERT_TRUE( index <= 0 || needle >= haystack_host[index - 1] );
            ASSERT_TRUE( index >= count || needle < haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_desc_lower_smaller )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, greater_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, greater_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< greater_t_null_smaller, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];
        if( !needle.is_null() )
        {
            ASSERT_TRUE( index <= 0 || needle < haystack_host[index - 1] );
            if( !haystack_host[index].is_null() )
                ASSERT_TRUE( index >= count || needle >= haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_desc_upper_smaller )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, greater_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, greater_t_null_smaller< nullable_type< int >, nullable_type< int > >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< greater_t_null_smaller, bounds_upper, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];
        if( !needle.is_null() )
        {
            ASSERT_TRUE( index <= 0 || needle <= haystack_host[index - 1] );
            if( !haystack_host[index].is_null() )
                ASSERT_TRUE( index >= count || needle > haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_asc_lower_bigger )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, less_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, less_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< less_t_null_bigger, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];

        if( !needle.is_null() )
        {
            ASSERT_TRUE( index <= 0 || needle > haystack_host[index - 1] );
            if( !haystack_host[index].is_null() )
                ASSERT_TRUE( index >= count || needle <= haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_asc_upper_bigger )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, less_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, less_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< less_t_null_bigger, bounds_upper, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];
        if( !needle.is_null() )
        {
            ASSERT_TRUE( index <= 0 || needle >= haystack_host[index - 1] );
            if( !haystack_host[index].is_null() )
                ASSERT_TRUE( index >= count || needle < haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_desc_lower_bigger )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, greater_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, greater_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< greater_t_null_bigger, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];
        if( !needle.is_null() )
        {
            if( !haystack_host[index - 1].is_null() )
                ASSERT_TRUE( index <= 0 || needle < haystack_host[index - 1] );
            ASSERT_TRUE( index >= count || needle >= haystack_host[index] );
        }
    }
}

TEST( sortedsearch, nullint_desc_upper_bigger )
{
    standard_context_t context;

    int count = 10000000;
    managed_mem_t< nullable_type< int > > needles( count, context );
    managed_mem_t< nullable_type< int > > haystack( count, context );

    std::uniform_int_distribution< int > d( 0, 100000 );
    for( int i = 0; i < count; i++ )
    {
        if( i % 10 )
        {
            needles.data()[i].flag = 1;
            haystack.data()[i].flag = 1;
        }
        else
        {
            needles.data()[i].flag = 0;
            haystack.data()[i].flag = 0;
        }
        needles.data()[i].value = d( get_mt19937() );
        haystack.data()[i].value = d( get_mt19937() );
    }
    mergesort( needles.data(), count, greater_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );
    mergesort( haystack.data(), count, greater_t_null_bigger< nullable_type< int >, nullable_type< int > >(), context );

    managed_mem_t< int > indices( count, context );
    context.timer_begin();
    sorted_search< greater_t_null_bigger, bounds_upper, launch_box_t< arch_52_cta< 256, 13 > > >( needles.data(), count, haystack.data(), count, indices.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    nullable_type< int >* needles_host = needles.data();
    nullable_type< int >* haystack_host = haystack.data();
    int* indices_host = indices.data();

    for( int i = 0; i < count; ++i )
    {
        nullable_type< int > needle = needles_host[i];
        int index = indices_host[i];
        if( !needle.is_null() )
        {
            if( !haystack_host[index - 1].is_null() )
                ASSERT_TRUE( index <= 0 || needle <= haystack_host[index - 1] );
            ASSERT_TRUE( index >= count || needle > haystack_host[index] );
        }
    }
}

TEST( sortedsearch, str_asc_lower )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, less_t_str< false, false >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, less_t_str< false, false >(), context );

    context.timer_begin();
    sorted_search< bounds_lower, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            less_t_str< false, false >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        ASSERT_TRUE( index <= 0 || ( strncmp( needle, haystack + ( index - 1 ) * colSize, colSize ) > 0 ) );
        ASSERT_TRUE( index >= len * repeat || ( strncmp( needle, haystack + index * colSize, colSize ) <= 0 ) );
    }
}

TEST( sortedsearch, str_asc_upper )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );


    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, less_t_str< false, false >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, less_t_str< false, false >(), context );

    context.timer_begin();
    sorted_search< bounds_upper, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            less_t_str< false, false >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        ASSERT_TRUE( index <= 0 || ( strncmp( needle, haystack + ( index - 1 ) * colSize, colSize ) >= 0 ) );
        ASSERT_TRUE( index >= len * repeat || ( strncmp( needle, haystack + index * colSize, colSize ) < 0 ) );
    }
}

TEST( sortedsearch, str_desc_lower )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str< false, false >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str< false, false >(), context );

    context.timer_begin();
    sorted_search< bounds_lower, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            greater_t_str< false, false >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        ASSERT_TRUE( index <= 0 || ( strncmp( needle, haystack + ( index - 1 ) * colSize, colSize ) < 0 ) );
        ASSERT_TRUE( index >= len * repeat || ( strncmp( needle, haystack + index * colSize, colSize ) >= 0 ) );
    }
}

TEST( sortedsearch, str_desc_upper )
{
    standard_context_t context;
    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str< false, false >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str< false, false >(), context );

    context.timer_begin();
    sorted_search< bounds_upper, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            greater_t_str< false, false >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        ASSERT_TRUE( index <= 0 || ( strncmp( needle, haystack + ( index - 1 ) * colSize, colSize ) <= 0 ) );
        ASSERT_TRUE( index >= len * repeat || ( strncmp( needle, haystack + index * colSize, colSize ) > 0 ) );
    }
}

TEST( sortedsearch, nullstr_asc_lower_smaller )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_smaller< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_smaller< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_lower, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            less_t_str_null_smaller< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index - 1 ) * colSize ) )
                ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) > 0 ) );
            ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) <= 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_asc_upper_smaller )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_smaller< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_smaller< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_upper, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            less_t_str_null_smaller< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index - 1 ) * colSize ) )
                ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) >= 0 ) );
            ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) < 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_desc_lower_smaller )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_smaller< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_smaller< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_lower, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            greater_t_str_null_smaller< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index ) * colSize ) )
                ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) >= 0 ) );
            ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) < 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_desc_upper_smaller )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_smaller< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_smaller< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_upper, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            greater_t_str_null_smaller< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index ) * colSize ) )
                ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) > 0 ) );
            ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) <= 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_asc_lower_bigger )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_bigger< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_bigger< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_lower, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            less_t_str_null_bigger< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index ) * colSize ) )
                ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) <= 0 ) );
            ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) > 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_asc_upper_bigger )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_bigger< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, less_t_str_null_bigger< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_upper, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            less_t_str_null_bigger< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index ) * colSize ) )
                ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) < 0 ) );
            ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) >= 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_desc_lower_bigger )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_bigger< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_bigger< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_lower, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            greater_t_str_null_bigger< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index - 1 ) * colSize ) )
                ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) < 0 ) );
            ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) >= 0 ) );
        }
    }
}

TEST( sortedsearch, nullstr_desc_upper_bigger )
{
    standard_context_t context;

    int repeat = 1;

    AriesDataBufferSPtr shipMode = ReadColumn( DB_NAME, "lineitem", 15 );
    size_t colSize = shipMode->GetItemSizeInBytes();
    int len = shipMode->GetItemCount();
    managed_mem_t< char > haystack_data( len * colSize * repeat, context );
    char* haystack = haystack_data.data();
    cudaMemcpy( haystack, shipMode->GetData(), shipMode->GetTotalBytes(), cudaMemcpyDefault );

    managed_mem_t< char > needles_data = haystack_data.clone();
    char * needles = needles_data.data();

    for( int i = 0; i < len * repeat; i++ )
    {
        if( i % 700 )
        {
            *( haystack + i * colSize ) = 1;
        }
        else
            *( haystack + i * colSize ) = 0;
        if( i % 300 )
        {
            *( needles + i * colSize ) = 1;
        }
        else
            *( needles + i * colSize ) = 0;
    }

    managed_mem_t< int > indices( len * repeat, context );

    mergesort( needles, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_bigger< true, true >(), context );
    mergesort( haystack, colSize, ( empty_t* )nullptr, len * repeat, greater_t_str_null_bigger< true, true >(), context );

    context.timer_begin();
    sorted_search< bounds_upper, launch_box_t< arch_52_cta< 256, 6 > > >( needles, len * repeat, haystack, len * repeat, colSize, indices.data(),
            greater_t_str_null_bigger< true, true >(), context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", len * repeat );
    for( int i = 0; i < len * repeat; ++i )
    {
        const char* needle = needles + i * colSize;
        int index = indices.data()[i];

        if( *needle )
        {
            if( *( haystack + ( index - 1 ) * colSize ) )
                ASSERT_TRUE( index <= 0 || ( strncmp( needle + 1, haystack + ( index - 1 ) * colSize + 1, colSize - 1 ) <= 0 ) );
            ASSERT_TRUE( index >= len * repeat || ( strncmp( needle + 1, haystack + index * colSize + 1, colSize - 1 ) > 0 ) );
        }
    }
}

