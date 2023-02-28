///*
// * test_materialize.cu
// *
// *  Created on: Aug 18, 2020
// *      Author: lichi
// */
//#include <gtest/gtest.h>
//#include "CpuTimer.h"
//#include "AriesEngine/AriesDataDef.h"
//#include "AriesEngine/cpu_algorithm.h"
//#include "CudaAcc/AriesSqlOperator.h"
//
//using namespace aries_engine;
//
//class MaterializeTest: public testing::Test
//{
//protected:
//    void SetUp() override
//    {
//        static_assert( TOTAL_ROW_COUNT % DATA_BLOCK_COUNT == 0 && TOTAL_ROW_COUNT % SETP_COUNT == 0 );
//        std::random_device rd;
//        std::mt19937 gen( rd() );
//        m_shuffledArray.resize( TOTAL_ROW_COUNT );
//        for( int i = 0; i < TOTAL_ROW_COUNT; ++i )
//            m_shuffledArray[i] = i;
//        std::shuffle( m_shuffledArray.begin(), m_shuffledArray.end(), gen );
//    }
//    void TearDown() override
//    {
//    }
//
//    AriesColumnSPtr CreateDataSource( AriesValueType valueType, size_t tupleNum, size_t blockCount )
//    {
//        AriesColumnSPtr data = std::make_shared<AriesColumn>();
//        size_t perBlockRowCount = tupleNum / blockCount;
//        for( int i = 0; i < blockCount; ++i )
//            data->AddDataBuffer( std::make_shared< AriesDataBuffer >( AriesColumnType{ AriesDataType{ valueType, 1 }, false, false }, perBlockRowCount ) );
//        return data;
//    }
//
//    void PrefetchColumnToCpu( const AriesColumnSPtr& column )
//    {
//        for( auto& buffer : column->GetDataBuffers() )
//            buffer->PrefetchToCpu();
//    }
//
//    void PrefetchColumnToGpu( const AriesColumnSPtr& column )
//    {
//        for( auto& buffer : column->GetDataBuffers() )
//            buffer->PrefetchToGpu();
//    }
//
//    AriesIndicesArraySPtr CreateIndicesRandomDuplicate( size_t count )
//    {
//        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( count );
//        std::random_device rd;
//        std::mt19937 gen( rd() );
//        std::uniform_int_distribution<> distrib( 0, TOTAL_ROW_COUNT - 1 );
//        index_t* pData = result->GetData();
//        for( int i = 0; i < count; ++i )
//            *pData++ = distrib( gen );
//        return result;
//    }
//
//    AriesIndicesArraySPtr CreateIndicesAscDuplicate( size_t count )
//    {
//        AriesIndicesArraySPtr result = CreateIndicesRandomDuplicate( count );
//        std::sort( result->GetData(), result->GetData() + count );
//        return result;
//    }
//
//    AriesIndicesArraySPtr CreateIndicesRandomUnique( size_t count )
//    {
//        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( count );
//        memcpy( result->GetData(), m_shuffledArray.data(), count * sizeof( int ) );
//        return result;
//    }
//
//    AriesIndicesArraySPtr CreateIndicesAscUnique( size_t count )
//    {
//        AriesIndicesArraySPtr result = CreateIndicesRandomUnique( count );
//        std::sort( result->GetData(), result->GetData() + count );
//        return result;
//    }
//
//    void RunTest( AriesValueType srcValueType, bool bInCpu,bool bAsc, bool bUnique )
//    {
//        AriesColumnSPtr column = CreateDataSource( srcValueType, TOTAL_ROW_COUNT, DATA_BLOCK_COUNT );
//
//        for( int indices_count = STEP_ROW_COUNT; indices_count < TOTAL_ROW_COUNT / 3; indices_count += STEP_ROW_COUNT )
//        {
//            AriesIndicesArraySPtr indices;
//            if( bUnique )
//            {
//                if( bAsc )
//                    indices = CreateIndicesAscUnique( indices_count );
//                else
//                    indices = CreateIndicesRandomUnique( indices_count );
//            }
//            else
//            {
//                if( bAsc )
//                    indices = CreateIndicesAscDuplicate( indices_count );
//                else
//                    indices = CreateIndicesRandomDuplicate( indices_count );
//            }
//            indices->PrefetchToGpu();
//            if( bInCpu )
//                PrefetchColumnToCpu( column );
//            else
//                PrefetchColumnToGpu( column );
//            AriesColumnType colType = column->GetColumnType();
//            cudaDeviceSynchronize();
//
//            // cpu begin, source in cpu, indices in gpu
//            CPU_Timer t;
//            t.begin();
//            PrefetchColumnToCpu( column );
//            indices->PrefetchToCpu();
//            AriesDataBufferSPtr result = GetDataBufferByIndices( indices, column->GetDataBuffers(), column->GetBlockSizePsumArray(), colType.HasNull,
//                    colType );
//            result->PrefetchToGpu();
//            cudaDeviceSynchronize();
//            auto cpuTimeCost = t.end();
//
//            result.reset();
//            indices->PrefetchToGpu();
//            if( bInCpu )
//                PrefetchColumnToCpu( column );
//            else
//                PrefetchColumnToGpu( column );
//            cudaDeviceSynchronize();
//            // gpu begin, source in cpu, indices in gpu, prefetch first
//            t.begin();
//            PrefetchColumnToGpu( column );
//            AriesInt64ArraySPtr prefixSumArray = GetPrefixSumOfBlockSize( column->GetBlockSizePsumArray() );
//            result = MaterializeColumn( column->GetDataBuffers(), prefixSumArray, indices, colType );
//            cudaDeviceSynchronize();
//            auto gpuTimeCost = t.end();
//
//            result.reset();
//            prefixSumArray.reset();
//            indices->PrefetchToGpu();
//            if( bInCpu )
//                PrefetchColumnToCpu( column );
//            else
//                PrefetchColumnToGpu( column );
//            cudaDeviceSynchronize();
//            //gpu begin, source in cpu, indices in gpu, no prefetch
//            t.begin();
//            prefixSumArray = GetPrefixSumOfBlockSize( column->GetBlockSizePsumArray() );
//            result = MaterializeColumn( column->GetDataBuffers(), prefixSumArray, indices, colType );
//            cudaDeviceSynchronize();
//            auto gpuTimeCost2 = t.end();
//
//            cout << "total row count=" << column->GetRowCount() << ", indices count=" << indices_count << ", cpu time cost:" << cpuTimeCost
//                    << ", prefetch gpu time cost:" << gpuTimeCost << ", no prefetch gpu time cost:" << gpuTimeCost2 << endl;
//        }
//    }
//    vector< int > m_shuffledArray;
//    static const int TOTAL_ROW_COUNT = 600000000;
//    static const int DATA_BLOCK_COUNT = 30;
//    static const int SETP_COUNT = 100000;
//    static const int STEP_ROW_COUNT = TOTAL_ROW_COUNT / SETP_COUNT;
//};
//
//TEST( prefetch, gpus )
//{
//    CPU_Timer t;
//    size_t totalBytes = 2l * 500000000;
//    cout<<"totalBytes:"<<totalBytes<<endl;
//    long timeCostPrefetch = 0;
//    long timeCostMemcpy = 0;
//    void* src;
//    void* dst;
//
//    for( int i = 0; i < 10; ++i )
//    {
//        int deviceId = 0;
//        cudaMallocManaged( &src, totalBytes );
//        cudaMallocManaged( &dst, totalBytes );
//        cudaMemPrefetchAsync( src, totalBytes, cudaCpuDeviceId );
//        memset( src, 0, totalBytes );
//        cudaMemPrefetchAsync( dst, totalBytes, deviceId );
//        cudaDeviceSynchronize();
//
//        //cudaSetDevice(deviceId);
//        t.begin();
//        cudaMemPrefetchAsync( src, totalBytes, deviceId );
//        cudaDeviceSynchronize();
//        timeCostPrefetch = t.end();
//        t.begin();
//        cudaMemcpy( dst, src, totalBytes, cudaMemcpyDeviceToDevice );
//        cudaDeviceSynchronize();
//        timeCostMemcpy = t.end();
//        cout<<"gpu 0 prefetch="<<timeCostPrefetch<<", cudaMemcpy="<<timeCostMemcpy<<", total="<<timeCostPrefetch + timeCostMemcpy<<endl;
//
//        cudaFree( src );
//        cudaFree( dst );
//
//        deviceId = 0;
//        cudaMallocManaged( &src, totalBytes );
//        cudaMallocManaged( &dst, totalBytes );
//        cudaMemPrefetchAsync( src, totalBytes, cudaCpuDeviceId );
//        memset( src, 0, totalBytes );
//        cudaMemPrefetchAsync( dst, totalBytes, deviceId );
//        cudaDeviceSynchronize();
//
//        timeCostPrefetch = 0;
//        timeCostMemcpy = 0;
//        //cudaSetDevice(deviceId);
//        t.begin();
//        cudaMemPrefetchAsync( src, totalBytes, deviceId );
//        cudaDeviceSynchronize();
//        timeCostPrefetch = t.end();
//        t.begin();
//        cudaMemcpy( dst, src, totalBytes, cudaMemcpyDeviceToDevice );
//        cudaMemPrefetchAsync( dst, totalBytes, deviceId );
//        cudaDeviceSynchronize();
//        timeCostMemcpy = t.end();
//        cout<<"gpu 1 prefetch="<<timeCostPrefetch<<", cudaMemcpy="<<timeCostMemcpy<<", total="<<timeCostPrefetch + timeCostMemcpy<<endl;
//
//        cudaFree( src );
//        cudaFree( dst );
//    }
//}
//
//TEST( prefetch, gpus_threads )
//{
//    size_t totalBytes = 2l * 500000000;
//    cout<<"totalBytes:"<<totalBytes<<endl;
//    vector< future< void > > allThreads;
//
//    for( int j = 0; j < 2; ++j )
//    {
//        allThreads.push_back( std::async( std::launch::async, [&]( int id )
//        {
//            for( int i = 0; i < 10; ++i )
//            {
//                CPU_Timer t;
//                long timeCostPrefetch = 0;
//                long timeCostMemcpy = 0;
//                void* src;
//                void* dst;
//                int deviceId = id;
//                cudaMallocManaged( &src, totalBytes );
//                cudaMallocManaged( &dst, totalBytes );
//                cudaMemPrefetchAsync( src, totalBytes, cudaCpuDeviceId );
//                memset( src, 0, totalBytes );
//                cudaMemPrefetchAsync( dst, totalBytes, deviceId );
//                cudaDeviceSynchronize();
//
//                cudaSetDevice(deviceId);
//                t.begin();
//                cudaMemPrefetchAsync( src, totalBytes, deviceId );
//                cudaDeviceSynchronize();
//                timeCostPrefetch = t.end();
//                t.begin();
//                cudaMemcpy( dst, src, totalBytes, cudaMemcpyDeviceToDevice );
//                cudaDeviceSynchronize();
//                timeCostMemcpy = t.end();
//                cout<<"gpu "<<deviceId<<" prefetch="<<timeCostPrefetch<<", cudaMemcpy="<<timeCostMemcpy<<", total="<<timeCostPrefetch + timeCostMemcpy<<endl;
//
//                cudaFree( src );
//                cudaFree( dst );
//            }
//        }, j ) );
//    }
//    for( auto& t : allThreads )
//        t.wait();
//}
//
//TEST( prefetch, gpus_threads_many )
//{
//    size_t totalBytes = 2l * 500000000;
//    cout<<"totalBytes:"<<totalBytes<<endl;
//    vector< future< void > > allThreads;
//
//    for( int j = 0; j < 10; ++j )
//    {
//        allThreads.push_back( std::async( std::launch::async, [&]( int id )
//        {
//            for( int i = 0; i < 2; ++i )
//            {
//                CPU_Timer t;
//                long timeCostPrefetch = 0;
//                long timeCostMemcpy = 0;
//                void* src;
//                void* dst;
//                int deviceId = id % 2;
//                cudaMallocManaged( &src, totalBytes );
//                cudaMallocManaged( &dst, totalBytes );
//                cudaMemPrefetchAsync( src, totalBytes, cudaCpuDeviceId );
//                memset( src, 0, totalBytes );
//                cudaMemPrefetchAsync( dst, totalBytes, deviceId );
//                cudaDeviceSynchronize();
//
//                cudaSetDevice(deviceId);
//                t.begin();
//                cudaMemPrefetchAsync( src, totalBytes, deviceId );
//                cudaDeviceSynchronize();
//                timeCostPrefetch = t.end();
//                t.begin();
//                cudaMemcpy( dst, src, totalBytes, cudaMemcpyDeviceToDevice );
//                cudaDeviceSynchronize();
//                timeCostMemcpy = t.end();
//                cout<<"thread id:"<< std::this_thread::get_id()<<" gpu "<<deviceId<<" prefetch="<<timeCostPrefetch<<", cudaMemcpy="<<timeCostMemcpy<<", total="<<timeCostPrefetch + timeCostMemcpy<<endl;
//
//                cudaFree( src );
//                cudaFree( dst );
//            }
//        }, j ) );
//    }
//    for( auto& t : allThreads )
//        t.wait();
//}
//
//TEST_F( MaterializeTest, int32_in_cpu_indices_random_dupliate )
//{
//    RunTest( AriesValueType::INT32, true, false, false );
//}
//
//TEST_F( MaterializeTest, int32_in_cpu_indices_random_unique )
//{
//    RunTest( AriesValueType::INT32, true, false, true );
//}
//
//TEST_F( MaterializeTest, int32_in_cpu_indices_asc_dupliate )
//{
//    RunTest( AriesValueType::INT32, true, true, false );
//}
//
//TEST_F( MaterializeTest, int32_in_cpu_indices_asc_unique )
//{
//    RunTest( AriesValueType::INT32, true, true, true );
//}
//
//TEST_F( MaterializeTest, int32_in_gpu_indices_random_dupliate )
//{
//    RunTest( AriesValueType::INT32, false, false, false );
//}
//
//TEST_F( MaterializeTest, int32_in_gpu_indices_random_unique )
//{
//    RunTest( AriesValueType::INT32, false, false, true );
//}
//
//TEST_F( MaterializeTest, int32_in_gpu_indices_asc_dupliate )
//{
//    RunTest( AriesValueType::INT32, false, true, false );
//}
//
//TEST_F( MaterializeTest, int32_in_gpu_indices_asc_unique )
//{
//    RunTest( AriesValueType::INT32, false, true, true );
//}
//
//TEST_F( MaterializeTest, int64_in_cpu_indices_random_dupliate )
//{
//    RunTest( AriesValueType::INT64, true, false, false );
//}
//
//TEST_F( MaterializeTest, int64_in_cpu_indices_random_unique )
//{
//    RunTest( AriesValueType::INT64, true, false, true );
//}
//
//TEST_F( MaterializeTest, int64_in_cpu_indices_asc_dupliate )
//{
//    RunTest( AriesValueType::INT64, true, true, false );
//}
//
//TEST_F( MaterializeTest, int64_in_cpu_indices_asc_unique )
//{
//    RunTest( AriesValueType::INT64, true, true, true );
//}
//
//TEST_F( MaterializeTest, int64_in_gpu_indices_random_dupliate )
//{
//    RunTest( AriesValueType::INT64, false, false, false );
//}
//
//TEST_F( MaterializeTest, int64_in_gpu_indices_random_unique )
//{
//    RunTest( AriesValueType::INT64, false, false, true );
//}
//
//TEST_F( MaterializeTest, int64_in_gpu_indices_asc_dupliate )
//{
//    RunTest( AriesValueType::INT64, false, true, false );
//}
//
//TEST_F( MaterializeTest, int64_in_gpu_indices_asc_unique )
//{
//    RunTest( AriesValueType::INT64, false, true, true );
//}
