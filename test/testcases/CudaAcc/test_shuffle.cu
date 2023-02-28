/*
 * test_shuffle.cu
 *
 *  Created on: Jun 6, 2019
 *      Author: lichi
 */
#include <thread>
#include "test_common.h"
#include "CudaAcc/AriesEngineDef.h"

#include "CpuTimer.h"

TEST( shuffle, integer )
{
    standard_context_t context;

    int count = 100000000;
    managed_mem_t< aries_acc::Decimal > data_input( count, context );
    managed_mem_t< int > indices( count, context );
    managed_mem_t< aries_acc::Decimal > data_output( count, context );

    for( int i = 0; i < count; ++i )
    {
        data_input.data()[ i ] = i;
        indices.data()[ i ] = i;
    }
    managed_mem_t< int > indices2( count, context );
    std::random_device rd;
    std::mt19937 g( rd() );
    int* pBegin = indices.data();
    int* pEnd = indices.data() + indices.size();
    std::shuffle( pBegin, pEnd, g );
    auto funcWrapper = [=]( const aries_acc::Decimal* data, const int* indice, int count, aries_acc::Decimal* output )
    {
        for( int i = 0; i < count; ++i )
        {
            int val = indice[i];
            auto it = std::upper_bound( indice, indice + 30, val );
            output[ i ] = data[ val ];
        }
    };
    int threadNum = 8;
    int blockSize = count / threadNum;
    int extraSize = count % threadNum;
    data_output.PrefetchToCpu();
    vector< thread > allThread;
    aries::CPU_Timer t;
    t.begin();
    int size = count/25;
//    for( int i = 0; i < 25; ++i )
//    {
//        memcpy(data_output.data() + size * i, data_input.data() + size * i , size );
//    }
    for( int i = 0; i < threadNum; ++i )
    {
        aries_acc::Decimal* pResult = data_output.data() + blockSize * i;
        aries_acc::Decimal* input = data_input.data();
        int* pIndex = indices.data() + blockSize * i;
        if( i == threadNum - 1 )
            allThread.emplace_back( funcWrapper, input, pIndex, blockSize + extraSize, pResult );
        else
            allThread.emplace_back( funcWrapper, input, pIndex, blockSize, pResult );
    }
    for( auto & t : allThread )
        t.join();
    cout << "-----------------------shuffle:" << t.end() << endl;

    data_input.PrefetchToGpu();
    indices.PrefetchToGpu();
    data_output.PrefetchToGpu();
    context.timer_begin();

    shuffle_by_index< launch_box_t< arch_52_cta< 256, 17 > > >( data_input.data(), count, indices.data(), data_output.data(), context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count; ++i )
    {
        ASSERT_TRUE( data_output.data()[ i ] == data_input.data()[ indices.data()[ i ] ] );
    }
}

TEST( shuffle, str )
{
    standard_context_t context;

    int repeat = 20;
    const char* file = "/var/rateup/data/scale_1/lineitem/lineitem14"; //6 shipdate char(10)
    char *h_data;
    int colSize = 10;
    int len = loadColumn( file, colSize, &h_data );
    managed_mem_t< int8_t > data_input( len * colSize * repeat, context );
    for( int i = 0; i < repeat; ++i )
    {
        memcpy( data_input.data() + i * len * colSize, h_data, len * colSize );
    }
    free( h_data );
    managed_mem_t< int8_t > data_output( len * colSize * repeat, context );
    int count = len * repeat;
    managed_mem_t< int > indices( count, context );

    for( int i = 0; i < count; ++i )
    {
        indices.data()[ i ] = count - 1 - i;
    }

    AriesArray< DataBlockInfo > blocks( 2 );
    DataBlockInfo* block = blocks.GetData();
    block[ 0 ].Data = data_input.data();
    block[ 0 ].ElementSize = colSize;

    AriesArray< int8_t* > outputs( 2 );
    int8_t** output = outputs.GetData();
    output[ 0 ] = data_output.data();

    managed_mem_t< int > data_input2( count, context );
    managed_mem_t< int > data_output2( count, context );

    for( int i = 0; i < count; ++i )
    {
        data_input2.data()[ i ] = i * 10;
    }

    block[ 1 ].Data = ( int8_t* )data_input2.data();
    block[ 1 ].ElementSize = 4;
    output[ 1 ] = ( int8_t* )data_output2.data();

//    void shuffle_by_index( const DataBlockInfo* data_input, int32_t block_count, size_t count, const index_t* indices, char** data_output,
//            context_t& context )

    context.timer_begin();
    shuffle_by_index< launch_box_t< arch_52_cta< 256, 17 > > >( block, 2, count, indices.data(), output, context );
    printf( "gpu time: %3.1f\n", context.timer_end() );
    printf( "item count is: %d\n", count );
    for( int i = 0; i < count; ++i )
    {
        ASSERT_TRUE(
                strncmp( ( const char* )data_output.data() + ( count - i - 1 ) * colSize, ( const char* )data_input.data() + i * colSize, colSize )
                        == 0 );
    }
}

TEST( memcpy, integer )
{
    standard_context_t context;

    int count = 1000000;
    managed_mem_t< int > input( count, context );
    managed_mem_t< int > output( count, context );
    int* data_input = input.data();
    int* data_output = output.data();
    input.PrefetchToCpu();
    output.PrefetchToCpu();

    aries::CPU_Timer t;
    t.begin();
    for( int i = 0; i < count; ++i )
    {
        memcpy( data_output + i, data_input + i, sizeof(int) );
    }
    cout << "-----------------------memcpy:" << t.end() << endl;
    t.begin();
    for( int i = 0; i < count; ++i )
    {
        cudaMemcpy( data_output + i, data_input + i, sizeof(int), cudaMemcpyHostToHost );
    }
    cout << "-----------------------cudaMemcpy:" << t.end() << endl;
    input.PrefetchToGpu();
    output.PrefetchToGpu();
    t.begin();
    for( int i = 0; i < count; ++i )
    {
        memcpy( data_output + i, data_input + i, sizeof(int) );
    }
    cout << "-----------------------memcpy:" << t.end() << endl;
    input.PrefetchToGpu();
    output.PrefetchToGpu();
    t.begin();
    for( int i = 0; i < count; ++i )
    {
        cudaMemcpy( data_output + i, data_input + i, sizeof(int), cudaMemcpyDeviceToDevice );
    }
    cout << "-----------------------cudaMemcpy:" << t.end() << endl;
}

TEST( meminfo, managed )
{
	size_t freeMem, totalMem;
	cudaError_t result = cudaMemGetInfo( &freeMem, &totalMem );
//	if( cudaSuccess != result )
//		throw aries_cuda_exception( result );
	cout<<"before create AriesDataBufferSPtr freeMem: "<<freeMem<<" bytes"<<endl;

	AriesDataType dataType { AriesValueType::DECIMAL, 1 };
	AriesDataBufferSPtr columnA = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
	columnA->AllocArray( 100000000, cudaMemAttachHost );
	AriesDataBufferSPtr columnB = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
	columnB->AllocArray( 110000000, cudaMemAttachHost );
	AriesDataBufferSPtr columnC = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
	columnC->AllocArray( 120000000, cudaMemAttachHost );
	result = cudaMemGetInfo( &freeMem, &totalMem );
//	if( cudaSuccess != result )
//		throw aries_cuda_exception( result );
	cout<<"after create AriesDataBufferSPtr freeMem: "<<freeMem<<" bytes"<<endl;
	//columnA->PrefetchToGpu();
	//columnB->PrefetchToGpu();
	//columnC->PrefetchToGpu();
	columnA->PrefetchToCpu();
	columnB->PrefetchToCpu();
	columnC->PrefetchToCpu();
	cudaDeviceSynchronize();
	result = cudaMemGetInfo( &freeMem, &totalMem );
//	if( cudaSuccess != result )
//		throw aries_cuda_exception( result );
	cout<<"after PrefetchToCpu() freeMem: "<<freeMem<<" bytes"<<endl;
	columnA->Reset();
	//columnB->Reset();
	//columnC->Reset();
	result = cudaMemGetInfo( &freeMem, &totalMem );
//	if( cudaSuccess != result )
//		throw aries_cuda_exception( result );
	cout<<"after Reset() freeMem: "<<freeMem<<" bytes"<<endl;
}

