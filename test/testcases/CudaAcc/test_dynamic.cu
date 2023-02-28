///*
// * test_dynamic.cu
// *
// *  Created on: Jul 18, 2019
// *      Author: lichi
// */
//#include <regex>
//#include "test_common.h"
//#include "aries_char.hxx"
//#include "CudaAcc/AriesSqlOperator.h"
//#include "AriesSqlFunctions.hxx"
//static const char* DB_NAME = "scale_1";
//using namespace aries_acc;
//using namespace std;
//
//ARIES_HOST_DEVICE bool str_like2( const char* text, const char* regexp, int len )
//{
//    const char* posStar = 0;
//    const char* flagInS = 0;
//    const char* end = text + len;
//    while( *text && text < end )
//    {
//        if( *regexp == '%' )
//        {
//            flagInS = text;
//            posStar = regexp;
//            ++regexp;
//        }
//        else if( *regexp && ( *text == *regexp || *regexp == '_' ) )
//        {
//            regexp++;
//            text++;
//        }
//        else if( posStar )
//        {
//            text = ++flagInS;
//            regexp = posStar + 1;
//        }
//        else
//        {
//            return false;
//        }
//    }
//    while( *regexp && *regexp == '%' )
//    {
//        ++regexp;
//    }
//
//    return ( *text == 0 || text == end ) && *regexp == 0;
//}
//
///**
// * C++ version 0.4 char* style "itoa":
// * Written by Lukás Chmela
// * Released under GPLv3.
//
// */
//char* aries_itoa( int value, char* result, int base = 10 )
//{
//    // check that the base if valid
//    if( base < 2 || base > 36 )
//    {
//        *result = '\0';
//        return result;
//    }
//
//    char* ptr = result, *ptr1 = result, tmp_char;
//    int tmp_value;
//
//    do
//    {
//        tmp_value = value;
//        value /= base;
//        *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz"[35 + ( tmp_value - value * base )];
//    } while( value );
//
//    // Apply negative sign
//    if( tmp_value < 0 )
//        *ptr++ = '-';
//    *ptr-- = '\0';
//    while( ptr1 < ptr )
//    {
//        tmp_char = *ptr;
//        *ptr-- = *ptr1;
//        *ptr1++ = tmp_char;
//    }
//    return result;
//}
//
//TEST(engine, atoi)
//{
//    string s = "期金融";
//    int b = -123;
//    aries_char< 12 > r;
//    cout << "aries_itoa-->" << op_tostr_t()( b ) << endl;
//    nullable_type< aries_char< 10 > > a = "12期金融";
//
//    nullable_type< int > result = sql_function_wrapper< nullable_type< int > >( CastAsIntWrapper< aries_char< 10 > >(), a );
//    cout << "aries_itoa nullable-->" << ( int )op_tostr_t()( result ).flag << " val:" << op_tostr_t()( result ).value << endl;
//    cout << result.value << endl;
//}
//
//struct ColumnDataIterator
//{
//    int8_t** m_data;
//    int64_t* m_blockSizePrefixSum;
//    int* m_indices;
//    int8_t* m_nullData;
//    int m_blockCount;
//    int m_perItemSize;ARIES_HOST_DEVICE
//    int8_t* GetData( int index ) const
//    {
//        int pos = index;
//        if( m_indices != nullptr )
//            pos = m_indices[index];
//        if( pos != -1 )
//        {
//            int blockIndex = binary_search< bounds_upper >( m_blockSizePrefixSum, m_blockCount, pos ) - 1;
//            return m_data[blockIndex] + ( pos - m_blockSizePrefixSum[blockIndex] ) * m_perItemSize;
//        }
//        else
//            return m_nullData;
//    }
//};
//
//void TestCallKernel( const vector< CUmoduleSPtr >& modules, const char *functionName, const int8_t **input, int tupleNum, const int** indices,
//        const vector< AriesDynamicCodeComparator >& items, int8_t *output )
//{
//    CUdevice dev;
//    cuCtxGetDevice( &dev );
//    ARIES_ASSERT( dev < modules.size(), "CUDA module for device " + std::to_string( dev ) + " is empty, no dynamic code or no compiling for it?" );
//    CUmoduleSPtr module = modules[dev];
//    ARIES_ASSERT( module != nullptr, "CUDA module for " + string( functionName ) + " is empty, no dynamic code or no compiling for it?" );
//
//    CallableComparator** operators = nullptr;
//    CUfunction kernel;
//    CUDA_SAFE_CALL( cuModuleGetFunction( &kernel, *module, functionName ) );
//
//    void *args[] =
//    { &input, &tupleNum, &indices, &operators, &output };
//
//    int numThreads = 256;
//    int numBlocks = ( tupleNum + numThreads - 1 ) / numThreads;
//    CUresult res = cuLaunchKernel( kernel, numBlocks, 1, 1,    // grid dim
//            numThreads, 1, 1,   // block dim
//            0, NULL,             // shared mem and stream
//            args, 0 );           // arguments
//    CUDA_SAFE_CALL( res );
//    CUDA_SAFE_CALL( cuCtxSynchronize() );
//}
//
//void TestCallKernel4( const vector< CUmoduleSPtr >& modules, const char *functionName, const ColumnDataIterator *input, int tupleNum,
//        const vector< AriesDynamicCodeComparator >& items, int8_t *output )
//{
//    CUdevice dev;
//    cuCtxGetDevice( &dev );
//    ARIES_ASSERT( dev < modules.size(), "CUDA module for device " + std::to_string( dev ) + " is empty, no dynamic code or no compiling for it?" );
//    CUmoduleSPtr module = modules[dev];
//    ARIES_ASSERT( module != nullptr, "CUDA module for " + string( functionName ) + " is empty, no dynamic code or no compiling for it?" );
//    CallableComparator** operators = nullptr;
//    CUfunction kernel;
//    CUDA_SAFE_CALL( cuModuleGetFunction( &kernel, *module, functionName ) );
//
//    void *args[] =
//    { &input, &tupleNum, &operators, &output };
//
//    int numThreads = 256;
//    int numBlocks = ( tupleNum + numThreads - 1 ) / numThreads;
//    CUresult res = cuLaunchKernel( kernel, numBlocks, 1, 1,    // grid dim
//            numThreads, 1, 1,   // block dim
//            0, NULL,             // shared mem and stream
//            args, 0 );           // arguments
//    CUDA_SAFE_CALL( res );
//    CUDA_SAFE_CALL( cuCtxSynchronize() );
//}
//
//TEST(engine, dynamic)
//{
//    standard_context_t context;
//    AriesDataBufferSPtr extendedPrice = ReadColumn( DB_NAME, "lineitem", 6 );
//    int colSize = extendedPrice->GetItemSizeInBytes();
//    int count = extendedPrice->GetItemCount();
//    AriesDataBufferSPtr discount = ReadColumn( DB_NAME, "lineitem", 7 );
//    AriesDataBufferSPtr tax = ReadColumn( DB_NAME, "lineitem", 8 );
//    const char* code =
//            R"(
//#include "functions.hxx"
//#include "AriesDateFormat.hxx"
//#include "aries_char.hxx"
//#include "decimal.hxx"
//#include "AriesDate.hxx"
//#include "AriesDatetime.hxx"
//#include "AriesIntervalTime.hxx"
//#include "AriesTime.hxx"
//#include "AriesTimestamp.hxx"
//#include "AriesYear.hxx"
//#include "AriesTimeCalc.hxx"
//#include "AriesSqlFunctions.hxx"
//using namespace aries_acc;
//
//enum bounds_t
//{
//    bounds_lower, bounds_upper
//};
//
//template< bounds_t bounds, typename type_t, typename type_u, typename int_t >
//ARIES_HOST_DEVICE int_t binary_search( const type_u* keys, int_t count, type_t key )
//{
//    int_t begin = 0;
//    int_t end = count;
//    while( begin < end )
//    {
//        int_t mid = ( begin + end ) / 2;
//        bool pred = ( bounds_upper == bounds ) ? key >= keys[ mid ] : keys[ mid ] < key;
//        if( pred )
//            begin = mid + 1;
//        else
//            end = mid;
//    }
//    return begin;
//}
//
//struct ColumnDataIterator
//{
//    int8_t** m_data;
//    int64_t* m_blockSizePrefixSum;
//    int* m_indices;
//    int8_t* m_nullData;
//    int m_blockCount;
//    int m_perItemSize;
//    ARIES_HOST_DEVICE int8_t* GetData( int index ) const
//    {
//        int pos = index;
//        if( m_indices != nullptr )
//            pos = m_indices[index];
//        if( pos != -1 )
//        {
//            int blockIndex = binary_search< bounds_upper >( m_blockSizePrefixSum, m_blockCount, pos ) - 1;
//            return m_data[blockIndex] + ( pos - m_blockSizePrefixSum[blockIndex] ) * m_perItemSize;
//        }
//        else
//            return m_nullData;
//    }
//};
//
//extern "C"  __global__ void expression2( const char **input, int tupleNum, const CallableComparator** comparators, char *output )
//{
//    int stride = blockDim.x * gridDim.x;
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    for( int i = tid; i < tupleNum; i += stride )
//    {
//        Decimal columnId_left_4( (CompactDecimal*)( input[0] + i * 7 ), 15, 2 );
//        Decimal columnId_left_5( (CompactDecimal*)( input[1] + i * 7 ), 15, 2 );
//        Decimal columnId_left_6( (CompactDecimal*)( input[2] + i * 7 ), 15, 2 );
//        Decimal Cuda_Dyn_resultValueName = ((columnId_left_4*(1-columnId_left_5))*(1+columnId_left_6));
//        auto tmp = output + i * 17;
//        Decimal(36, 6).cast(Cuda_Dyn_resultValueName).ToCompactDecimal( tmp, 17);
//    }
//}
//
//extern "C"  __global__ void expression3( const char **input, int tupleNum, const int** indices, const CallableComparator** comparators, char *output )
//{
//    int stride = blockDim.x * gridDim.x;
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    for( int i = tid; i < tupleNum; i += stride )
//    {
//        Decimal columnId_left_4( (CompactDecimal*)( input[0] + indices[0][i] * 7 ), 15, 2 );
//        Decimal columnId_left_5( (CompactDecimal*)( input[1] + indices[1][i] * 7 ), 15, 2 );
//        Decimal columnId_left_6( (CompactDecimal*)( input[2] + indices[2][i] * 7 ), 15, 2 );
//        Decimal Cuda_Dyn_resultValueName = ((columnId_left_4*(1-columnId_left_5))*(1+columnId_left_6));
//        auto tmp = output + i * 17;
//        Decimal(36, 6).cast(Cuda_Dyn_resultValueName).ToCompactDecimal( tmp, 17);
//    }
//}
//
//
//extern "C"  __global__ void expression4( const ColumnDataIterator *input, int tupleNum, const CallableComparator** comparators, char *output )
//{
//    int stride = blockDim.x * gridDim.x;
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    for( int i = tid; i < tupleNum; i += stride )
//    {
//        Decimal columnId_left_4( (CompactDecimal*)( input[0].GetData( i ) ), 15, 2 );
//        Decimal columnId_left_5( (CompactDecimal*)( input[1].GetData( i ) ), 15, 2 );
//        Decimal columnId_left_6( (CompactDecimal*)( input[2].GetData( i ) ), 15, 2 );
//        Decimal Cuda_Dyn_resultValueName = ((columnId_left_4*(1-columnId_left_5))*(1+columnId_left_6));
//        auto tmp = output + i * 17;
//        Decimal(36, 6).cast(Cuda_Dyn_resultValueName).ToCompactDecimal( tmp, 17);
//    }
//}
//
//)";
//    AriesDataType dataType3
//    { AriesValueType::COMPACT_DECIMAL, 36, 6 };
//    AriesDataBufferSPtr output3 = make_shared< AriesDataBuffer >( AriesColumnType
//    { dataType3, false, false }, count );
//
//    auto module = CompileKernels( code );
//    vector< AriesDynamicCodeComparator > comps;
//    aries_engine::AriesIndicesSPtr indices1 = std::make_shared< aries_engine::AriesIndices >();
//    AriesIndicesArraySPtr indexArray1 = std::make_shared< AriesIndicesArray >( count );
//    indices1->AddIndices( indexArray1 );
//
//    aries_engine::AriesIndicesSPtr indices2 = std::make_shared< aries_engine::AriesIndices >();
//    AriesIndicesArraySPtr indexArray2 = std::make_shared< AriesIndicesArray >( count );
//    indices2->AddIndices( indexArray2 );
//
//    aries_engine::AriesIndicesSPtr indices3 = std::make_shared< aries_engine::AriesIndices >();
//    AriesIndicesArraySPtr indexArray3 = std::make_shared< AriesIndicesArray >( count );
//    indices3->AddIndices( indexArray3 );
//
//    aries_engine::AriesColumnSPtr column1 = std::make_shared< aries_engine::AriesColumn >();
//    column1->AddDataBuffer( extendedPrice );
//    aries_engine::AriesColumnReferenceSPtr columnRef1 = std::make_shared< aries_engine::AriesColumnReference >( column1 );
//    columnRef1->SetIndices( indices1 );
//
//    aries_engine::AriesColumnSPtr column2 = std::make_shared< aries_engine::AriesColumn >();
//    column2->AddDataBuffer( discount );
//    aries_engine::AriesColumnReferenceSPtr columnRef2 = std::make_shared< aries_engine::AriesColumnReference >( column2 );
//    columnRef2->SetIndices( indices2 );
//
//    aries_engine::AriesColumnSPtr column3 = std::make_shared< aries_engine::AriesColumn >();
//    column3->AddDataBuffer( tax );
//    aries_engine::AriesColumnReferenceSPtr columnRef3 = std::make_shared< aries_engine::AriesColumnReference >( column3 );
//    columnRef3->SetIndices( indices3 );
//
//    int* pIndex1 = indexArray1->GetData();
//    int* pIndex2 = indexArray2->GetData();
//    int* pIndex3 = indexArray3->GetData();
//    for( int i = 0; i < count; i++ )
//    {
//        pIndex1[i] = i;
//        pIndex2[i] = i;
//        pIndex3[i] = i;
//    }
//    std::random_shuffle( pIndex1, pIndex1 + count );
//    std::random_shuffle( pIndex2, pIndex2 + count );
//    std::random_shuffle( pIndex3, pIndex3 + count );
//    AriesArray< const int32_t* > index( 3 );
//    index.GetData()[0] = ( const int32_t* )pIndex1;
//    index.GetData()[1] = ( const int32_t* )pIndex2;
//    index.GetData()[2] = ( const int32_t* )pIndex3;
//    AriesArray< const int8_t* > columns( 3 );
//    columns.GetData()[0] = extendedPrice->GetData();
//    columns.GetData()[1] = discount->GetData();
//    columns.GetData()[2] = tax->GetData();
//
//    context.timer_begin();
//    TestCallKernel( module, "expression3", columns.GetData(), count, index.GetData(), comps, output3->GetData() );
//    printf( "expression3 gpu time: %3.1f\n", context.timer_end() );
//
//    extendedPrice->PrefetchToCpu();
//    discount->PrefetchToCpu();
//    tax->PrefetchToCpu();
//    output3->PrefetchToCpu();
//    indexArray1->PrefetchToCpu();
//    indexArray2->PrefetchToCpu();
//    indexArray3->PrefetchToCpu();
//
////    context.timer_begin();
////    CPU_Timer t;
////    t.begin();
////    auto buffer1 = columnRef1->GetDataBuffer();
////    auto buffer2 = columnRef2->GetDataBuffer();
////    auto buffer3 = columnRef3->GetDataBuffer();
////    columns.GetData()[0] = buffer1->GetData();
////    columns.GetData()[1] = buffer2->GetData();
////    columns.GetData()[2] = buffer3->GetData();
////    t.end();
////    CallKernel( module, "expression2", columns.GetData(), count, comps, output3->GetData() );
////    printf( "expression2 gpu time: %3.1f\n", context.timer_end() );
//
//    extendedPrice->PrefetchToCpu();
//    discount->PrefetchToCpu();
//    tax->PrefetchToCpu();
//    output3->PrefetchToCpu();
//    indexArray1->PrefetchToCpu();
//    indexArray2->PrefetchToCpu();
//    indexArray3->PrefetchToCpu();
//
//    AriesArray< ColumnDataIterator > iter( 3 );
//
//    vector< AriesDataBufferSPtr > dataBuffers1 = column1->GetDataBuffers();
//    AriesArray< int8_t* > dataBlocks1( dataBuffers1.size() );
//    int i = 0;
//    for( const auto& block : dataBuffers1 )
//        dataBlocks1[i++] = block->GetData();
//
//    ColumnDataIterator iter1;
//    iter1.m_data = dataBlocks1.GetData();
//    iter1.m_blockCount = dataBuffers1.size();
//    iter1.m_indices = pIndex1;
//    iter1.m_nullData = nullptr;
//    auto prefixSum1 = column1->GetPrefixSumOfBlockSize();
//    iter1.m_blockSizePrefixSum = prefixSum1->GetData();
//    iter1.m_perItemSize = column1->GetColumnType().GetDataTypeSize();
//    iter[0] = iter1;
//
//    vector< AriesDataBufferSPtr > dataBuffers2 = column2->GetDataBuffers();
//    AriesArray< int8_t* > dataBlocks2( dataBuffers2.size() );
//    i = 0;
//    for( const auto& block : dataBuffers2 )
//        dataBlocks2[i++] = block->GetData();
//
//    ColumnDataIterator iter2;
//    iter2.m_data = dataBlocks2.GetData();
//    iter2.m_blockCount = dataBuffers2.size();
//    iter2.m_indices = pIndex2;
//    iter2.m_nullData = nullptr;
//    auto prefixSum2 = column2->GetPrefixSumOfBlockSize();
//    iter2.m_blockSizePrefixSum = prefixSum2->GetData();
//    iter2.m_perItemSize = column2->GetColumnType().GetDataTypeSize();
//    iter[1] = iter2;
//
//    vector< AriesDataBufferSPtr > dataBuffers3 = column3->GetDataBuffers();
//    AriesArray< int8_t* > dataBlocks3( dataBuffers3.size() );
//    i = 0;
//    for( const auto& block : dataBuffers3 )
//        dataBlocks3[i++] = block->GetData();
//
//    ColumnDataIterator iter3;
//    iter3.m_data = dataBlocks3.GetData();
//    iter3.m_blockCount = dataBuffers3.size();
//    iter3.m_indices = pIndex3;
//    iter3.m_nullData = nullptr;
//    auto prefixSum3 = column3->GetPrefixSumOfBlockSize();
//    iter3.m_blockSizePrefixSum = prefixSum3->GetData();
//    iter3.m_perItemSize = column3->GetColumnType().GetDataTypeSize();
//    iter[2] = iter3;
//
//    context.timer_begin();
//    TestCallKernel4( module, "expression4", iter.GetData(), count, comps, output3->GetData() );
//    printf( "expression4 gpu time: %3.1f\n", context.timer_end() );
//}
//
