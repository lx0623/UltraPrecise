#include "test_common.h"
#include <thread>
#include "AriesEngine/cpu_algorithm.h"

template< typename launch_arg_t = empty_t, typename type_t, typename type_u = type_t, typename comp_t, typename output_t >
void filter_data_if( const type_t* data, int* indices, size_t count, comp_t cmp, type_u param, output_t* output, context_t& context )
{
    typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
    auto k = [=] ARIES_DEVICE(int index)
    {
        output[ index ] = (output_t)cmp( data[ indices[ index ] ], param );
    };

    transform< launch_t >( k, count, context );
    context.synchronize();
}

TEST( filter, int_lt_by_indices )
{
    standard_context_t context;
    int count = 100000000;
    aries_engine::AriesColumnSPtr column = std::make_shared< aries_engine::AriesColumn >();
    AriesDataType dataType
    { AriesValueType::INT32, 1 };
    AriesDataBufferSPtr buffer = make_shared< AriesDataBuffer >( AriesColumnType
    { dataType, false, false }, count );
    column->AddDataBuffer( buffer );

    aries_engine::AriesIndicesSPtr indices = std::make_shared< aries_engine::AriesIndices >();
    AriesIndicesArraySPtr indexArray = std::make_shared< AriesIndicesArray >( count );
    indices->AddIndices( indexArray );

    aries_engine::AriesColumnReferenceSPtr columnRef = std::make_shared< aries_engine::AriesColumnReference >( column );
    columnRef->SetIndices( indices );

    managed_mem_t< int > output( count, context );
    int* pOutput = output.data();
    int* pIndex = ( int* )indexArray->GetData();
    int* pData = ( int* )buffer->GetData();
    for( int i = 0; i < count; i++ )
    {
        pData[i] = i;
        //pIndex[i] = i;
    }
    indexArray->CopyFromHostMem( pData, buffer->GetTotalBytes() );
    std::random_shuffle( pData, pData + count );
    std::random_shuffle( pIndex, pIndex + count );
    context.timer_begin();
    auto buf = columnRef->GetDataBuffer();
    int* input_data = ( int* )( buf->GetData() );
    filter_data_if( input_data, count, less_t< int >(), 5, pOutput, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count; ++i )
    {
        if( pOutput[i] )
        {
            ASSERT_TRUE( input_data[i] < 5 );
        }
        else
        {
            ASSERT_TRUE( input_data[i] >= 5 );
        }
    }
    buffer->PrefetchToCpu();
    output.PrefetchToCpu();
    //indexArray->PrefetchToCpu();
    context.timer_begin();
    filter_data_if( pData, pIndex, count, less_t< int >(), 5, pOutput, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );

    auto hostindices = indexArray->DataToHostMemory();
    pIndex = hostindices.get();
    for( int i = 0; i < count; ++i )
    {
        if( pOutput[i] )
        {
            ASSERT_TRUE( pData[pIndex[i]] < 5 );
        }
        else
        {
            ASSERT_TRUE( pData[pIndex[i]] >= 5 );
        }
    }
    printf( "done!\n" );
}

TEST( filter, int_lt )
{
    standard_context_t context;
    int count = 100000000;

    std::uniform_int_distribution< int > d( 0, 10000000 );
    managed_mem_t< int > data( count, context );
    managed_mem_t< int > output( count, context );
    int* pData = data.data();
    int* pOutput = output.data();
    for( int i = 0; i < count; i++ )
    {
        data.data()[i] = i;
    }
    context.timer_begin();
    filter_data_if( pData, count, less_t< int >(), 5, pOutput, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count; ++i )
    {
        if( pOutput[i] )
        {
            ASSERT_TRUE( pData[i] < 5 );
        }
        else
        {
            ASSERT_TRUE( pData[i] >= 5 );
        }
    }
    printf( "done!\n" );
}

TEST( filter, str_eq_null )
{
    standard_context_t context;
    int repeat = 10;
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
    }

    context.timer_begin();
    filter_data_if( keys_input, colSize, len * repeat, equal_to_t_str< true, false >(), keys_input + 1, vals_input, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str;
    for( int i = 0; i < len * repeat; ++i )
    {
        str = keys_input + colSize * i;
        if( vals_input[i] )
        {
            ASSERT_TRUE( strncmp( str + 1, keys_input + 1, colSize - 1 ) == 0 );
        }
        else
        {
            if( strncmp( str + 1, keys_input + 1, colSize - 1 ) == 0 )
            {
                ASSERT_FALSE( *str );
            }
        }
    }
    printf( "done!\n" );
}

TEST( filter, str_lt )
{
    standard_context_t context;
    int repeat = 10;
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
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();
    managed_mem_t< char > param( colSize, context );
    const char *pStr = "1998-08-01";
    char* pParam = param.data();
    memcpy( pParam, pStr, colSize );
    context.timer_begin();
    filter_data_if( keys_input, colSize, len * repeat, less_equal_t_str< false, false >(), pParam, vals_input, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str;
    for( int i = 0; i < len * repeat; ++i )
    {
        str = keys_input + colSize * i;
        if( vals_input[i] )
        {
            ASSERT_TRUE( strncmp( str, pParam, colSize ) <= 0 );
        }
        else
        {
            ASSERT_TRUE( strncmp( str, pParam, colSize ) > 0 );
        }
    }
    printf( "done!\n" );
}

TEST( filter, str_in )
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
    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();
    managed_mem_t< char > param( colSize * 10, context );
    char* pParam = param.data();
    memcpy( pParam, keys_input, colSize * 10 );
    context.timer_begin();
    filter_data_if( keys_input, colSize, len * repeat, op_in_t_str< false >(), pParam, 10, vals_input, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    const char* str;
    for( int i = 0; i < len * repeat; ++i )
    {
        str = keys_input + colSize * i;
        bool b = false;
        if( vals_input[i] )
        {
            for( int j = 0; j < 10; ++j )
                if( strncmp( str, pParam + j * colSize, colSize ) == 0 )
                    b = true;
            ASSERT_TRUE( b );
        }
        else
        {
            for( int j = 0; j < 10; ++j )
                if( strncmp( str, pParam + j * colSize, colSize ) == 0 )
                    b = true;
            ASSERT_FALSE( b );
        }
    }
    printf( "done!\n" );
}

extern "C" __global__ void check_data( const char *input, int tupleNum, int len, int *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const char* data;
    for( size_t i = tid; i < tupleNum; i += stride )
    {
        output[i] = !op_like_t< false >()( input + i * len, "%special%packages%", len );
    }
}

extern "C" __global__ void check_data2( const char *input, const int* indices, int tupleNum, int len, int *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const char* data;
    for( size_t i = tid; i < tupleNum; i += stride )
    {
        output[i] = op_like_t< false >()( input + indices[i] * len, "%special%packages%", len );
    }
}

extern "C" __global__ void check_data3( const char *input, int tupleNum, int len, int *selected_count, int *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const char* data;
    for( size_t i = tid; i < tupleNum; i += stride )
    {
        if( !op_like_t< false >()( input + i * len, "%special%packages%", len ) )
            output[atomicAdd( selected_count, 1 )] = i;
    }
}

//__global__ void get_filtered_index_wa(void *source, int64_t *filtered_flag_with_pos, int *selected_count, AriesDate target, size_t count) {
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;
//    for (int i = index; i < count; i += stride) {
//        AriesDate* start = (AriesDate*) source;
//        auto date = start + i;
//        bool selected = false;
//        if (date->year < target.year) {
//            selected = true;
//        } else if (date->year == target.year) {
//            if (date->month < target.month) {
//                selected = true;
//            } else if (date->month == target.month) {
//                selected = date->day <= target.day;
//            }
//        }
//        if (selected) {
//            filtered_flag_with_pos[atomicAggInc(selected_count)] = i;
//        }
//    }
//}

TEST( filter, str_like )
{
    standard_context_t context;
    int repeat = 15;
    const char* file = "/var/rateup/data/scale_1/orders/orders8"; //6 shipdate char(10)
    char *h_data;
    int colSize = 78;
    size_t len = loadColumn( file, colSize, &h_data );
    //len = 30012150;
    size_t size = len * colSize * repeat;
    cout << "len=" << len * repeat << endl;
    cout << "size=" << size << endl;
    managed_mem_t< char > keys( size, context );
    char * keys_input = keys.data();
    for( int i = 0; i < repeat; ++i )
    {
        memcpy( keys_input + i * len * colSize, h_data, len * colSize );
    }
    free( h_data );
    managed_mem_t< int > vals( len * repeat, context );
    cout << "vals size=" << len << endl;
    int * vals_input = vals.data();
    managed_mem_t< char > param( 19, context );
    const char *pStr = "%special%packages%";
    char* pParam = param.data();
    memset( pParam, 0, 19 );
    strcpy( pParam, pStr );
    managed_mem_t< long > rdata( 1, context );
    long* match_count = rdata.data();
    *match_count = 0;
    size_t threadNum = 12;
    size_t blockSize = len * repeat / threadNum;
    size_t extraSize = ( len * repeat ) % threadNum;
    auto funcWrapper = []( const char* keys_input, size_t len, size_t colSize, const char* pParam, int* vals_input )
    {
        for( size_t i = 0; i < len; ++i )
        {
            vals_input[ i ] = op_like_t< false >()( keys_input + i * colSize, pParam, colSize );
        }
    };
    vals.PrefetchToCpu();
    keys.PrefetchToCpu();
    vector< thread > allThread;
    CPU_Timer t;
    t.begin();
    for( int i = 0; i < threadNum; ++i )
    {

        if( i == threadNum - 1 )
            allThread.emplace_back( funcWrapper, keys_input + ( blockSize * i ) * colSize, blockSize + extraSize, colSize, pParam,
                    vals_input + blockSize * i );
        else
            allThread.emplace_back( funcWrapper, keys_input + ( blockSize ) * i * colSize, blockSize, colSize, pParam, vals_input + blockSize * i );
    }
    for( auto & t : allThread )
        t.join();

    t.end();
    for( int i = 0; i < len * repeat; ++i )
    {
        if( vals_input[i] )
        {
            *match_count += 1;
        }
    }
    cout << "cpu match_count=" << *match_count << endl;

    managed_mem_t< int > indices( len * repeat, context );
    for( int i = 0; i < len * repeat; ++i )
    {
        indices.data()[i] = i;
    }
    cout << "indices.data()[ i ] = i OK" << endl;
    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( indices.data(), indices.data() + indices.size(), g );
    cout << "shuffle OK" << endl;
    ////////////////////////
    vector< thread > allThread3;
    managed_mem_t< char > data_output( size, context );
    data_output.PrefetchToCpu();
    auto shuffle = []( const char* keys_input, const int* indices, size_t len, int count, char* output )
    {
        for( int i = 0; i < count; ++i )
        {
            memcpy( output + i * len, keys_input + indices[i] * len, len );
        }
    };

    t.begin();
    for( int i = 0; i < threadNum; ++i )
    {
        char* pResult = data_output.data() + blockSize * i * colSize;
        int* pIndex = indices.data() + blockSize * i;
        if( i == threadNum - 1 )
            allThread3.emplace_back( shuffle, keys_input, pIndex, colSize, blockSize + extraSize, pResult );
        else
            allThread3.emplace_back( shuffle, keys_input, pIndex, colSize, blockSize, pResult );
    }
    for( auto & t : allThread3 )
        t.join();
    cout << "-----------------------shuffle:" << t.end() << endl;
    keys.PrefetchToGpu();
    indices.PrefetchToGpu();
    data_output.PrefetchToGpu();
    context.timer_begin();
    shuffle_by_index( keys.data(), colSize, len * repeat, indices.data(), data_output.data(), context );
    printf( "shuffle_by_index gpu time: %3.1f\n", context.timer_end() );
    ///////////////////////

    vals.PrefetchToCpu();
    keys.PrefetchToCpu();
    indices.PrefetchToCpu();
    auto funcWrapper2 = []( const char* keys_input, const int* indices, size_t len, size_t colSize, const char* pParam, int* vals_input )
    {
        for( size_t i = 0; i < len; ++i )
        {
            vals_input[ i ] = op_like_t< false >()( keys_input + indices[i] * colSize, pParam, colSize );
        }
    };

    vector< thread > allThread2;
    t.begin();
    for( int i = 0; i < threadNum; ++i )
    {

        if( i == threadNum - 1 )
            allThread2.emplace_back( funcWrapper2, keys_input, indices.data() + ( blockSize * i ), blockSize + extraSize, colSize, pParam,
                    vals_input + blockSize * i );
        else
            allThread2.emplace_back( funcWrapper2, keys_input, indices.data() + ( blockSize * i ), blockSize, colSize, pParam,
                    vals_input + blockSize * i );
    }
    for( auto & t : allThread2 )
        t.join();

    t.end();
    *match_count = 0;
    for( int i = 0; i < len * repeat; ++i )
    {
        if( vals_input[i] )
        {
            *match_count += 1;
        }
    }
    cout << "cpu2 match_count=" << *match_count << endl;
    managed_mem_t< int > selected_count( 1, context );
//    cudaMemPrefetchAsync( keys_input, len* repeat / 2 * colSize, 0 );
//    cudaMemPrefetchAsync( vals_input, len* repeat / 2 * sizeof(int), 0 );
    vals.PrefetchToGpu();
    keys.PrefetchToGpu();
    indices.PrefetchToGpu();
    *selected_count.data() = 0;
    context.timer_begin();

    check_data<<< div_up( len * repeat, 256ul ), 256ul >>>( keys_input, len * repeat, colSize, vals_input );

    get_filtered_index( vals_input, len * repeat, indices.data(), selected_count.data(), context );
//    check_data<<<div_up( len* repeat / 2, 256ul ),256ul>>>(keys_input, len* repeat / 2, colSize, vals_input );
//    cudaMemPrefetchAsync( keys_input, len* repeat / 2 * colSize, cudaCpuDeviceId );
//    cudaMemPrefetchAsync( vals_input, len* repeat / 2 * sizeof(int), cudaCpuDeviceId );
//    check_data<<<div_up( len* repeat / 2, 256ul ),256ul>>>(keys_input + len* repeat / 2 * colSize, len* repeat / 2, colSize, vals_input + len* repeat / 2 );
    printf( "check_data gpu time: %3.1f\n", context.timer_end() );

    //check_data3( const char *input, int tupleNum, int len, int *selected_count, int *output )
    int total = 0;

    for( int i = 0; i < len * repeat; ++i )
    {
        if( vals_input[i] )
        {
            ++total;
        }
    }
    cout << "check_data match_count=" << *selected_count.data() << endl;
    ASSERT_EQ( total, *selected_count.data() );
    *selected_count.data() = 0;
    vals.PrefetchToGpu();
    keys.PrefetchToGpu();
    indices.PrefetchToGpu();
    context.timer_begin();

    check_data3<<< div_up( len * repeat, 256ul ), 256ul >>>( keys_input, len * repeat, colSize, selected_count.data(), vals_input );
//    check_data<<<div_up( len* repeat / 2, 256ul ),256ul>>>(keys_input, len* repeat / 2, colSize, vals_input );
//    cudaMemPrefetchAsync( keys_input, len* repeat / 2 * colSize, cudaCpuDeviceId );
//    cudaMemPrefetchAsync( vals_input, len* repeat / 2 * sizeof(int), cudaCpuDeviceId );
//    check_data<<<div_up( len* repeat / 2, 256ul ),256ul>>>(keys_input + len* repeat / 2 * colSize, len* repeat / 2, colSize, vals_input + len* repeat / 2 );
    printf( "check_data3 gpu time: %3.1f\n", context.timer_end() );
    cout << "check_data3 match_count=" << *selected_count.data() << endl;

    vals.PrefetchToGpu();
    keys.PrefetchToGpu();
    indices.PrefetchToGpu();
    context.timer_begin();

    check_data2<<< div_up( len * repeat, 256ul ), 256ul >>>( keys_input, indices.data(), len * repeat, colSize, vals_input );
//    check_data<<<div_up( len* repeat / 2, 256ul ),256ul>>>(keys_input, len* repeat / 2, colSize, vals_input );
//    cudaMemPrefetchAsync( keys_input, len* repeat / 2 * colSize, cudaCpuDeviceId );
//    cudaMemPrefetchAsync( vals_input, len* repeat / 2 * sizeof(int), cudaCpuDeviceId );
//    check_data<<<div_up( len* repeat / 2, 256ul ),256ul>>>(keys_input + len* repeat / 2 * colSize, len* repeat / 2, colSize, vals_input + len* repeat / 2 );
    printf( "check_data2 gpu time: %3.1f\n", context.timer_end() );

    vals.PrefetchToGpu();
    keys.PrefetchToGpu();
    context.timer_begin();
    filter_data_if( keys_input, colSize, len * repeat, op_like_t< false >(), pParam, vals_input, context );

    printf( "gpu time: %3.1f\n", context.timer_end() );
    total = 0;
    for( int i = 0; i < len * repeat; ++i )
    {
        if( vals_input[i] )
        {
            ++total;
        }
    }
    cout << "filter_data_if match_count=" << total << endl;
    printf( "item count is: %d, colSize is: %d\n", len * repeat, colSize );

    //ASSERT_EQ( total, len * repeat );
    printf( "done!\n" );
    managed_mem_t< int > sum( 1, context );
    vals.PrefetchToGpu();
    indices.PrefetchToGpu();
    context.timer_begin();
    scan( vals_input, len * repeat, indices.data(), plus_t< int32_t >(), sum.data(), context );
    printf( "scan gpu time: %3.1f\n", context.timer_end() );
    managed_mem_t< int > indices2( len * repeat, context );
    vals.PrefetchToCpu();
    indices2.PrefetchToCpu();

    managed_mem_t< int > cloned = indices2.clone();
    cloned.PrefetchToCpu();
    int* pcloned = cloned.data();
    pcloned[0] = 0;
    int *pIndices = indices2.data();

    pIndices[0] = 0;
    int start = 0;
    for( int i = 1; i < len * repeat; ++i )
    {
        start += vals_input[i - 1];
        pIndices[i] = start;
    }

    t.begin();

    //SumInSingleThread( vals_input, len * repeat, 0, indices2.data() );
    PrefixSum( vals_input, len * repeat, indices2.data() );

//    pIndices[ 0 ] = 0;
//    start = 0;
//    for( int i = 1; i < len* repeat; ++i )
//    {
//        start += vals_input[i-1];
//        pIndices[i]=start;
//    }
//    int *pIndices = indices2.data();
//    int start = 0;
//    for( int i = 0; i < len* repeat; ++i )
//    {
//        pIndices[i]= start + i;
//    }
    t.end();
    for( int i = 0; i < len * repeat; ++i )
    {
        ASSERT_EQ( indices.data()[i], indices2.data()[i] );
    }
}
