#include "test_common.h"
#include <thread>
static const char* DB_NAME = "tpch218_100";
//static const char* DB_NAME = "scale_1";

extern "C" __global__ void str_like_func( const char *input, int tupleNum, int len, int *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( size_t i = tid; i < tupleNum; i += stride )
    {
        output[i] = op_like_t< false >()( input + i * len, "%special%packages%", len );
    }
}

TEST( strlike, like )
{
    standard_context_t context;

    AriesDataBufferSPtr o_comment = ReadColumn( DB_NAME, "orders", 9 );
    size_t colSize = o_comment->GetItemSizeInBytes();
    size_t len = o_comment->GetItemCount();
    int repeat = 1;
    size_t size = len * colSize * repeat;
    cout << "count =" << len * repeat << endl;
    cout << "size =" << size << endl;
    managed_mem_t< char > keys( size, context );
    char * keys_input = keys.data();
    for( int i = 0; i < repeat; ++i )
    {
        memcpy( keys_input + i * len * colSize, o_comment->GetData(), o_comment->GetTotalBytes() );
    }

    managed_mem_t< int > vals( len * repeat, context );
    int * vals_input = vals.data();
    cudaMemset( vals_input, 0, len * repeat * sizeof( int ) );

    size_t threadNum = thread::hardware_concurrency();
    size_t blockSize = len * repeat / threadNum;
    size_t extraSize = ( len * repeat ) % threadNum;
    auto funcWrapper = []( const char* keys_input, size_t len, size_t colSize, int* vals_input )
    {
        for( size_t i = 0; i < len; ++i )
        {
            vals_input[ i ] = op_like_t< false >()( keys_input + i * colSize, "%special%packages%", colSize );
        }
    };
    vals.PrefetchToCpu();
    keys.PrefetchToCpu();
    cudaDeviceSynchronize();
    vector< thread > allThread;
    
    //use CPU
    CPU_Timer t;
    t.begin();
    for( int i = 0; i < threadNum; ++i )
    {
        if( i == threadNum - 1 )
            allThread.emplace_back( funcWrapper, keys_input + ( blockSize * i ) * colSize, blockSize + extraSize, colSize, vals_input + blockSize * i );
        else
            allThread.emplace_back( funcWrapper, keys_input + ( blockSize ) * i * colSize, blockSize, colSize, vals_input + blockSize * i );
    }
    for( auto & t : allThread )
        t.join();

    cout<<"cpu time cost:"<<t.end()<<endl;
    int match_count = 0;
    for( int i = 0; i < len * repeat; ++i )
    {
        if( vals_input[i] )
        {
            match_count += 1;
        }
    }
    cout << "cpu match_count=" << match_count << endl;


    memset( vals_input, 0, len * repeat * sizeof( int ) );
    vals.PrefetchToCpu();
    keys.PrefetchToCpu();
    cudaDeviceSynchronize();

    // use GPU
    context.timer_begin();
    vals.PrefetchToGpu();
    keys.PrefetchToGpu();

    str_like_func<<< div_up( len * repeat, 256ul ), 256ul >>>( keys_input, len * repeat, colSize, vals_input );

    printf( "gpu time const: %3.1f\n", context.timer_end() );

    match_count = 0;
    for( int i = 0; i < len * repeat; ++i )
    {
        if( vals_input[i] )
        {
            match_count += 1;
        }
    }
    cout << "gpu match_count=" << match_count << endl;
}