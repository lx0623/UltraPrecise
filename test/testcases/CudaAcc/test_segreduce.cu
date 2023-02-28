#include "test_common.h"

TEST( segreduce, sum_integer_null )
{
    standard_context_t context;
    int count = 10000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t < nullable_type< aries_acc::Decimal > > data( count, context );
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
    managed_mem_t < nullable_type< aries_acc::Decimal > >reduction( 1, context );
    managed_mem_t< int32_t > seg( 1, context );
    int32_t* pSeg = seg.data();
    *pSeg = 0;
    //managed_mem_t < nullable_type< std::common_type<int, int64_t>::type > > reduction( 1, context );
    context.timer_begin();
    segreduce( data.data(), count, pSeg, 1, reduction.data(), agg_sum_t< nullable_type< aries_acc::Decimal > >(), nullable_type< aries_acc::Decimal >(), context );
    //void segreduce( input_it input, int count, segments_it segments, int num_segments, output_it output, op_t op, type_t init, context_t& context )
    //reduce( data.data(), count, reduction.data(), agg_sum_t< nullable_type< std::common_type<int, int64_t>::type > >(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

//    aries_acc::Decimal sum = 0;
//    for( int i = 0; i < count; ++i )
//    {
//        if( data.data()[i].flag )
//            sum += data.data()[i].value;
//    }
//    printf("sum=%d\n", sum );
//    ASSERT_EQ( sum, reduction.data()[0].value );
}
