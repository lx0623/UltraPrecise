/*
 * test_cpu_algorithm.h
 *
 *  Created on: Oct 22, 2019
 *      Author: lichi
 */

#ifndef TEST_CPU_ALGORITHM_H_
#define TEST_CPU_ALGORITHM_H_
#include <vector>
#include <algorithm>
#include <numeric>
#include <future>
#include "AriesDataDef.h"
using namespace std;
using namespace aries_engine;

static constexpr size_t ITEMS_PER_THREAD = 16;

static inline size_t GetThreadNum( size_t count )
{
	return std::min( std::max( count / ITEMS_PER_THREAD, 1ul ), (size_t)thread::hardware_concurrency() );
}

template< typename output_t, typename input_t, typename seg_t, typename op_t >
vector< output_t > SegmentReduceSingleThread( const input_t* input, const int* associated, int64_t input_offset, const vector< seg_t >& segments,
        op_t op, output_t init_value )
{
    int64_t seg_count = segments.size();
    vector< output_t > output( seg_count );
    int64_t index = input_offset;
    for( int64_t i = 0; i < seg_count; ++i )
    {
        output_t reduced = init_value;
        seg_t currentEnd = segments[i];
        while( index < currentEnd )
            reduced = op( reduced, input[associated[index++]] );
        output[i] = reduced;
    }
    return output;
}

template< typename output_t, typename input_t, typename op_t >
output_t SegmentReduceSingleThread( const input_t* input, const int* associated, int64_t input_offset, int64_t count, op_t op, output_t init_value )
{
    output_t output = init_value;
    int64_t index = input_offset;
    for( int64_t i = 0; i < count; ++i )
    {
    	output = op( output, input[index++] );
    }
    return output;
}

template< typename output_t, typename input_t, typename op_t >
vector< output_t > SegmentReduce( const input_t* input, const int* associated, int64_t count, op_t op, output_t init_value )
{
    int64_t threadNum = GetThreadNum( count );
    int64_t blockSize = count / threadNum;
    int64_t extraSize = count % threadNum;
    int64_t blockBegin = 0;
    int64_t itemCount = 0;

    vector< output_t > allResult;
    vector< future< output_t > > allThreads;
    for( int i = 0; i < threadNum; ++i )
    {
        blockBegin = blockSize * i;
        itemCount = ( ( i == threadNum - 1 ) ? blockSize + extraSize : blockSize );
        allThreads.push_back( std::async( std::launch::async, [=]
        {   return SegmentReduceSingleThread( input, associated, blockBegin, itemCount, op, init_value );} ) );
    }

    for( auto& t : allThreads )
        allResult.push_back( t.get() );

    vector< output_t > result;
    output_t reduced = init_value;
    for( const auto& r : allResult )
    	reduced = op( reduced, r );

    result.push_back( reduced );
    return result;
}

template< typename output_t, typename input_t, typename seg_t, typename op_t >
vector< output_t > SegmentReduce( const input_t* input, const int* associated, int64_t count, const seg_t* segments, int64_t seg_count, op_t op,
        output_t init_value )
{
	if( seg_count == 1 )
		return SegmentReduce( input, associated, count, op, init_value );
    int64_t threadNum = GetThreadNum( count );
    int64_t blockSize = count / threadNum;
    int64_t extraSize = count % threadNum;
    seg_t blockBegin = 0;
    seg_t blockEnd = 0;
    const seg_t* pBegin = segments + 1; // start from the first no-zero element
    const seg_t* pEnd = segments + seg_count;
    const seg_t* pCurEnd;
    vector< bool > bNeedMerge( threadNum );
    vector< vector< output_t > > allResult;
    vector< future< vector< output_t > > > allThreads;
    for( int i = 0; i < threadNum; ++i )
    {
        blockBegin = blockSize * i;
        blockEnd = blockBegin + blockSize + ( ( i == threadNum - 1 ) ? extraSize : 0 );
        pBegin = std::upper_bound( pBegin, pEnd, blockBegin );
        pCurEnd = std::upper_bound( pBegin, pEnd, blockEnd );
        bNeedMerge[i] = ( blockBegin != pBegin[-1] );
        vector< seg_t > segs;
        while( pBegin != pCurEnd )
        {
            segs.push_back( *pBegin++ );
        }
        if( segs.empty() || segs.back() != blockEnd )
        {
            segs.push_back( blockEnd );
        }
        bNeedMerge[i] = bNeedMerge[i] && ( segs[0] != blockBegin );
        allThreads.push_back( std::async( std::launch::async, [=]
        {   return SegmentReduceSingleThread( input, associated, blockBegin, segs, op, init_value );} ) );
    }
    bNeedMerge[0] = false;

    for( auto& t : allThreads )
        allResult.push_back( t.get() );

    vector< output_t > result;
    int index = 0;
    for( const auto& r : allResult )
    {
        bool merge = bNeedMerge[index++];
        for( int i = 0; i < r.size(); ++i )
        {
            if( i == 0 )
            {
                if( merge )
                    result.back() = op( result.back(), r[0] );
                else
                    result.push_back( r[0] );
            }
            else
                result.push_back( r[i] );
        }
    }
    return result;
}

template< typename output_t, typename input_t >
void SumInSingleThread( const input_t* input, int64_t count, output_t init, output_t* output )
{
    output[0] = init;
    output_t tmp = init;
    for( int64_t i = 1; i < count; ++i )
    {
        tmp += input[i - 1];
        output[i] = tmp;
    }
}

template< typename output_t, typename input_t >
output_t PrefixSum( const input_t* input, int64_t count, output_t* output )
{
    int64_t threadNum = GetThreadNum( count );
    int64_t blockSize = count / threadNum;
    int64_t extraSize = count % threadNum;
    vector< output_t > partialSum;
    vector< future< output_t > > allThreads;
    const input_t* inputBlock;

    for( int i = 0; i < threadNum; ++i )
    {
        inputBlock = input + blockSize * i;
        if( i == threadNum - 1 )
            allThreads.push_back( std::async( std::launch::async, [=]
            {   return std::accumulate(inputBlock, inputBlock + blockSize + extraSize, output_t
                        {});} ) );
        else
            allThreads.push_back( std::async( std::launch::async, [=]
            {   return std::accumulate(inputBlock, inputBlock + blockSize, output_t
                        {});} ) );
    }
    for( auto& t : allThreads )
        partialSum.push_back( t.get() );

    vector< output_t > partialPrefixSum( threadNum );
    partialPrefixSum[0] = output_t
    { };
    for( int i = 1; i < threadNum; ++i )
    {
        partialPrefixSum[i] = partialPrefixSum[i - 1] + partialSum[i - 1];
    }
    output_t total = partialPrefixSum[threadNum - 1] + partialSum[threadNum - 1];

    vector< future< void > > workThreads;
    for( int i = 0; i < threadNum; ++i )
    {
        if( i == threadNum - 1 )
            workThreads.push_back( std::async( std::launch::async, [=]
            {   SumInSingleThread(input + blockSize * i, blockSize + extraSize, partialPrefixSum[i], output + blockSize * i);} ) );
        else
            workThreads.push_back( std::async( std::launch::async, [=]
            {   SumInSingleThread(input + blockSize * i, blockSize, partialPrefixSum[i], output + blockSize * i);} ) );
    }
    for( auto& t : workThreads )
        t.wait();

    return total;
}

template< typename type_t >
void ShuffleInSingleThread( const type_t* input, const type_t* new_indices, type_t* output, int64_t output_count )
{
    for( int64_t i = 0; i < output_count; ++i )
    {
        type_t newIdx = new_indices[i];
        if( NULL_INDEX == newIdx )
        {
            output[i] = NULL_INDEX;
        }
        else
        {
            output[i] = input[newIdx];
        }
    }
}

template< typename type_t >
void ShuffleInSingleThread( const vector< type_t* >& inputs,
                            const vector< int64_t >& inputsRowCountPrefixSum,
                            const type_t* new_indices,
                            type_t* output,
                            int64_t output_count );

template< typename type_t >
void AddOffsetInSingleThread( type_t* input, int64_t count, bool hasNull, int64_t offset )
{
    if( hasNull )
    {
        for( int64_t i = 0; i < count; ++i )
        {
            type_t inputIdx = input[i];
            if( NULL_INDEX != inputIdx )
                input[i] = inputIdx + offset;
        }
    }
    else
    {
        for( int64_t i = 0; i < count; ++i )
            input[i] += offset;
    }
}

AriesIndicesArraySPtr ShuffleIndices( const AriesIndicesArraySPtr& oldIndices, const AriesIndicesArraySPtr& indices );
AriesIndicesArraySPtr ShuffleIndices( const AriesIndicesArraySPtr& oldIndices, const vector< AriesIndicesArraySPtr >& indices );
vector< AriesIndicesArraySPtr > ShuffleIndices( const vector< AriesIndicesArraySPtr >& oldIndices, const AriesIndicesArraySPtr& indices );
AriesIndicesArraySPtr ShuffleIndices( const AriesIndicesSPtr& oldIndices, const AriesIndicesArraySPtr& indices );
vector< AriesIndicesSPtr > ShuffleIndices( const vector< AriesIndicesSPtr >& oldIndices, const AriesIndicesArraySPtr& indices );
void AddOffsetToIndices( AriesIndicesArraySPtr indices, bool hasNull, int64_t offset );

void GetDataBufferByCpuSingleThread( const index_t* pIndex, int64_t itemCount, const vector< AriesDataBufferSPtr >& dataBlocks,
        const vector< int64_t >& prefixSum, bool hasNull, AriesColumnType columnType, int8_t* output );

AriesDataBufferSPtr GetDataBufferByIndices( const AriesIndicesArraySPtr& indices, const vector< AriesDataBufferSPtr >& dataBlocks,
        const vector< int64_t >& prefixSum, bool hasNull, AriesColumnType columnType );

void GetDataBufferByCpuSingleThread( const index_t* pIndex, int64_t itemCount, const int8_t* data, bool hasNull, AriesColumnType columnType,
        int8_t* output );

AriesDataBufferSPtr GetDataBufferByIndices( const AriesIndicesArraySPtr& indices, const AriesDataBufferSPtr& data, bool hasNull,
        AriesColumnType columnType );

AriesIndicesArraySPtr CreateNullIndex( int64_t count );
AriesIndicesArraySPtr CreateSequenceIndex( int64_t count, index_t startVal = 0 );

struct PartitionedIndices
{
    shared_ptr< vector< index_t > > Indices;
    int Round = 1;
};

AriesIndicesArraySPtr ConvertToAriesIndices( const PartitionedIndices& partIndices );

vector< PartitionedIndices > PartitionColumnData(
    const vector< AriesDataBufferSPtr >& buffers,
    size_t bucketCount,
    uint32_t seed,
    bool *hasNullValues = nullptr,
    shared_ptr< vector< index_t > > indices = nullptr,
    int round = 1 );


class AriesHostDataBuffer;
using AriesHostDataBufferSPtr = std::shared_ptr< AriesHostDataBuffer >;
class AriesHostDataBuffer
{
public:
    AriesHostDataBuffer( AriesColumnType columnType, size_t reservedCount = 0 ) 
        : m_columnType( columnType ), m_itemSizeInBytes( columnType.GetDataTypeSize() ), m_reservedCount( 0 ), m_itemCount( 0 ), m_pDst( nullptr )
    {
        if( reservedCount > 0 )
            AddBlock( reservedCount );
    }

    ~AriesHostDataBuffer(){}

    AriesDataBufferSPtr ToAriesDataBuffer( bool onGPU = true ) const
    {
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( m_columnType, m_itemCount );
        if( m_itemCount > 0 )
        {
            if ( onGPU )
                result->PrefetchToGpu();
            else
                result->PrefetchToCpu();

            int8_t *pOutput = result->GetData();
            int index = 0;
            for( auto& block : m_dataBlocks )
            {
                cudaMemcpy( pOutput, block.get(), m_blockItemCount[ index ] * m_itemSizeInBytes, cudaMemcpyDefault );
                pOutput += m_blockItemCount[ index ] * m_itemSizeInBytes;
                ++index;
            }
        }
        return result;
    }

    size_t GetItemCount() const
    {
        return m_itemCount;
    }

    void AddItem( const int8_t* data )
    {
        assert( data && m_pDst );
        assert( m_reservedCount > 0 );
        if( ++m_itemCount > m_reservedCount )
        {
            size_t expandCount = m_reservedCount / 10;
            if( expandCount < 100 )
                expandCount = 100;
            AddBlock( expandCount );
        }
        memcpy( m_pDst, data, m_itemSizeInBytes );
        m_pDst += m_itemSizeInBytes;
        ++( m_blockItemCount.back() );
    }

    void Merge( AriesHostDataBufferSPtr buffer )
    {
        int index = 0;
        for( auto& block : buffer->m_dataBlocks )
        {
            m_dataBlocks.push_back( std::move( block ) );
            m_blockItemCount.push_back( buffer->m_blockItemCount[ index++ ] );
            m_itemCount += m_blockItemCount.back();
        }
        m_pDst = nullptr;
        m_reservedCount = 0;
    }

private:
    void AddBlock( size_t blockItemCount )
    {
        m_reservedCount += blockItemCount;
        m_blockItemCount.push_back( 0 );
        int8_t* ptr = new int8_t[ blockItemCount * m_itemSizeInBytes ];
        m_dataBlocks.push_back( unique_ptr< int8_t >{ ptr } );
        m_pDst = m_dataBlocks.back().get();
    }

    AriesColumnType m_columnType;
    size_t m_itemSizeInBytes;
    size_t m_reservedCount;
    size_t m_itemCount;
    int8_t* m_pDst;
    vector< unique_ptr< int8_t > > m_dataBlocks;
    vector< size_t > m_blockItemCount;
};

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const vector< int >& hashColumnIds,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues = nullptr );

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const AriesDataBufferSPtr &hashColumnBuffer,
                              vector< vector< AriesHostDataBufferSPtr > >& partBuffers,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues = nullptr );

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const AriesDataBufferSPtr &hashColumnBuffer,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues = nullptr );

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const AriesDataBufferSPtr &hashColumnBuffer,
                              vector< vector< AriesDataBufferSPtr > >& partBuffers,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues = nullptr );

struct PartitionResult
{
    vector< vector< AriesHostDataBufferSPtr > > AllBuffers;
    vector< PartitionedIndices > AllIndices;
};

PartitionResult PartitionColumnDataEx(
    const vector< AriesDataBufferSPtr >& buffers,
    size_t bucketCount,
    uint32_t seed,
    bool *hasNullValues = nullptr );


 void GatherGroupDataByCPU(
    const int8_t* input,
    const size_t item_length,
    const int32_t block_count,
    const index_t* associated,
    const index_t* indices,
    const size_t indices_count,
    const int8_t* output );

#endif /* TEST_CPU_ALGORITHM_H_ */
