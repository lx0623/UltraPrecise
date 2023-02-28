/*
 * test_atomic.cu
 *
 *  Created on: May 3, 2020
 *      Author: lichi
 */
#include "test_common.h"

struct ARIES_PACKED AriesUdt
{
    uint8_t intg;        //:7;
    uint8_t frac;        //:5;
    //最后 2 bits由mode使用,前面 6 bits由intg使用 2^6 = 64 > SUPPORTED_MAX_PRECISION
    uint8_t mode;    //:2;
    //最后 3 bits由error使用,前面 5 bits由frac使用 2^5 = 32 > SUPPORTED_MAX_SCALE
    uint8_t error;    //:2;
    int32_t value = 0;ARIES_DEVICE
    AriesUdt& operator+=( const AriesUdt& d )
    {
        this->value += d.value;
        return *this;
    }
};

ARIES_DEVICE void lock_udt( AriesUdt* val )
{
    uint32_t* p = ( uint32_t* )val;
    uint32_t old;
    do
    {
        old = atomicOr( p, 0x80000000 );
    } while( old & 0x80000000 );
}

ARIES_DEVICE void unlock_udt( AriesUdt* val )
{
    uint32_t* p = ( uint32_t* )val;
    atomicAnd( p, 0x7fffffff );
}

extern "C" __global__ void addudt( const AriesUdt *input, int tupleNum, AriesUdt *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int i = tid; i < tupleNum; i += stride )
    {
        lock_udt( output );
        *output += input[i];
        unlock_udt( output );
    }
}

TEST(atomic, udt)
{
    standard_context_t context;
    size_t count = 1000000;
    managed_mem_t< AriesUdt > data( count, context );
    managed_mem_t< AriesUdt > result( 1, context );
    AriesUdt* pdata = data.data();
    AriesUdt* pResult = result.data();
    pResult->intg = 0;
    pResult->frac = 0;
    pResult->mode = 0;
    pResult->error = 0;
    pResult->value = 0;

    int n = 0;
    for( int i = 0; i < count; ++i )
    {
        n += i;
        pdata[i].value = i;
    }
    cout << "n=" << n << endl;
    data.PrefetchToGpu();
    context.timer_begin();
    addudt<<< div_up( count, 256ul ), 256ul >>>( pdata, count, pResult );
    context.synchronize();
    printf( "addudt gpu time: %3.1f\n", context.timer_end() );
    ASSERT_EQ( n, pResult->value );
}

TEST(device, prop)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0 );
    cout << "directManagedMemAccessFromHost:" << prop.directManagedMemAccessFromHost << endl;
    cout << "asyncEngineCount:" << prop.asyncEngineCount << endl;
    cout << "concurrentManagedAccess :" << prop.concurrentManagedAccess << endl;
}

