/*
 * AriesDataDef.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: lichi
 */
#include <algorithm>
#include "AriesDataDef.h"
#include "AriesUtil.h"
#include "cpu_algorithm.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "CpuTimer.h"
#include "utils/utils.h"
#include "datatypes/functions.hxx"
#include "Compression/dict/AriesDict.h"

BEGIN_ARIES_ENGINE_NAMESPACE

   bool IndicesOk( const AriesIndicesArraySPtr& indices, int64_t rowCount, bool hasNul = false )
   {
       indices->PrefetchToCpu();
       bool result = true;
       index_t* pIndices = indices->GetData();
       int64_t count = indices->GetItemCount();
       for( int i = 0; i < count; ++i )
       {
           if( -1 == pIndices[i] )
           {
               if( !hasNul )
               {
                   result = false;
                   break;
               }
               continue;
           }
           if( pIndices[i] < 0 || pIndices[i] >= rowCount )
           {
               LOG( INFO )<< "rowCount=" << rowCount << " i=" << i << " pIndices[i]=" << pIndices[ i ];
               result = false;
               break;
           }
       }
       return result;
   }

    AriesInt64ArraySPtr GetPrefixSumOfBlockSize( const vector< int64_t >& blockSizePrefixSum )
    {
        AriesInt64ArraySPtr result = std::make_shared< AriesInt64Array >( blockSizePrefixSum.size() );
        AriesMemAllocator::MemCopy( result->GetData(), blockSizePrefixSum.data(), sizeof(int64_t) * blockSizePrefixSum.size() );
        return result;
    }

    AriesBaseColumnSPtr AriesBaseColumn::CloneWithNoContent() const
    {
        AriesColumnSPtr result = std::make_shared< AriesColumn >();
        result->AddDataBuffer( std::make_shared< AriesDataBuffer >( m_columnType ) );
        return result;
    }

    void AriesTableBlockStats::Print( const string& extraMsg ) const
    {
        string msg( extraMsg );
        msg.append( ", table block stats: \n" );
        msg.append( "materialization time: " ).append( std::to_string( m_materializeTime ) );
        msg.append( ", get sub table time: " ).append( std::to_string( m_getSubTableTime ) );
        msg.append( ", update indice time: " ).append( std::to_string( m_updateIndiceTime ) );
#ifdef ARIES_PROFILE
        LOG( INFO )<< msg;
#endif
    }

    string AriesTableBlockStats::ToJson( int inputRows ) const
    {
        string stats = "{\"inputRows\":" + std::to_string( inputRows ) + ", \"m\":" + std::to_string( m_materializeTime );
        stats += ", \"s\":" + std::to_string( m_getSubTableTime );
        stats += ", \"u\":" + std::to_string( m_updateIndiceTime ) + "}";
        return stats;
    }

    AriesColumn::AriesColumn()
            : m_totalCount( 0 )
    {
        m_blockSizePrefixSum.push_back( 0 );
    }

    int64_t AriesColumn::AddDataBuffer( const AriesDataBufferSPtr& dataBuffer )
    {
        ARIES_ASSERT( dataBuffer, "dataBlock is null!!!" );
        ARIES_ASSERT( m_totalCount > 0 ? m_columnType == dataBuffer->GetDataType() : true, "columnType is different!!!" );
        m_columnType = dataBuffer->GetDataType();
        m_dataBlocks.push_back( dataBuffer );
        int64_t count = dataBuffer->GetItemCount();
        m_blockSizePrefixSum.push_back( m_blockSizePrefixSum.back() + count );
        m_totalCount += count;
        ARIES_ASSERT( m_blockSizePrefixSum.back() == m_totalCount, "m_totalCount should be equal to m_blockSizes.back()!!!" );

        return m_totalCount;
    }

    const vector< int64_t >& AriesColumn::GetBlockSizePsumArray() const
    {
        return m_blockSizePrefixSum;
    }

    bool AriesColumn::IsGpuMaterilizeBetter( size_t resultCount )
    {
        assert( !m_dataBlocks.empty() );

        size_t currentSize = m_totalCount * m_columnType.GetDataTypeSize();
        size_t materializedSize = resultCount * m_columnType.GetDataTypeSize() + resultCount * sizeof( index_t );
        #ifndef NDEBUG
        if( materializedSize > currentSize )
            cout<<"IsGpuMaterilizeBetter: true ->";
        else
            cout<<"IsGpuMaterilizeBetter: false ->";
        cout<<", result row count:"<<resultCount<<", source row count:"<<m_totalCount<<", cpu cost transfer bytes:"<< materializedSize <<", gpu cost transfer bytes:"<<currentSize<<endl;
        #endif
        return materializedSize > currentSize;
    }

    AriesDataBufferSPtr AriesColumn::GetDataBuffer( const AriesIndicesSPtr& indices, bool bRunOnGpu )
    {
        AriesDataBufferSPtr result;

        if( IsGpuMaterilizeBetter( indices->GetRowCount() ) )
        {
            AriesColumnType columnType = m_columnType;
            columnType.HasNull |= indices->HasNull();

            vector< AriesDataBufferSPtr > dataBlocks;
            for( auto block : GetDataBuffers() )
            {
                block->PrefetchToGpu();
                dataBlocks.push_back( block );
            }
            auto dataBlocksPrefixSum = GetPrefixSumOfBlockSize( GetBlockSizePsumArray() );

            auto indicesPrefixSum = GetPrefixSumOfBlockSize( indices->GetBlockSizePsumArray() );
            for( auto& indice : indices->GetIndicesArray() )
                indice->PrefetchToGpu();
            result = MaterializeColumn( dataBlocks, dataBlocksPrefixSum, indices->GetIndicesArray(), indicesPrefixSum, columnType ); 
        }
        else 
        {
            for( auto& indice : indices->GetIndicesArray() )
                indice->PrefetchToCpu();
            result = GetDataBufferByCpu( indices->GetIndices(), indices->HasNull() );
        }

        return result;
    }

    AriesDataBufferSPtr AriesColumn::GetDataBuffer( const AriesIndicesArraySPtr& indices, bool hasNull, bool bRunOnGpu )
    {
        ARIES_ASSERT( indices && indices->GetItemCount() > 0, "indices is null or empty!!!" );
        
        assert( IndicesOk( indices, GetRowCount(), hasNull ) );
        AriesDataBufferSPtr result;

        if( IsGpuMaterilizeBetter( indices->GetItemCount() ) )
        {
            AriesColumnType columnType = m_columnType;
            columnType.HasNull |= hasNull;
            vector< AriesDataBufferSPtr > dataBlocks;
            for( auto block : GetDataBuffers() )
            {
                block->PrefetchToGpu();
                dataBlocks.push_back( block );
            }
            indices->PrefetchToGpu();
            result = MaterializeColumn( dataBlocks, GetPrefixSumOfBlockSize( GetBlockSizePsumArray() ), indices, columnType );
        }
        else
        {
            indices->PrefetchToCpu();
            result = GetDataBufferByCpu( indices, hasNull );
        }
           
        return result;
    }

    // AriesDataBufferSPtr AriesColumn::GetDataBuffer( int64_t offset, int64_t count ) const
    // {
    //     ARIES_ASSERT( m_totalCount > 0, "AriesColumn is empty()!!!" );
    //     ARIES_ASSERT( offset + count <= m_totalCount, "offset or count are out of range!!!" );
    //     ARIES_ASSERT( count > 0, "count must > 0!!! count=" + std::to_string( count ) );
    //     return GetDataBufferByCpu( offset, count );
    // }

    vector< AriesDataBufferSPtr > AriesColumn::GetDataBuffers() const
    {
        return m_dataBlocks;
    }

    size_t AriesColumn::GetDataBlockCount() const
    {
        return m_dataBlocks.size();
    }

    vector< AriesDataBufferSPtr > AriesColumn::GetDataBuffers( int64_t offset, int64_t& count, bool bStrict ) const
    {
        ARIES_ASSERT( m_totalCount > 0, "AriesColumn is empty()!!!" );
        ARIES_ASSERT( offset + count <= m_totalCount, "offset or count are out of range!!! offset + count=" + std::to_string( offset + count ) );
        ARIES_ASSERT( count > 0, "count must > 0!!! count=" + std::to_string( count ) );
        vector< AriesDataBufferSPtr > result;
        int64_t margin = bStrict ? 0 : count / 10; // 10%
        auto it = std::upper_bound( m_blockSizePrefixSum.cbegin(), m_blockSizePrefixSum.cend(), offset );
        int blockIndex = ( it - m_blockSizePrefixSum.cbegin() ) - 1;
        ARIES_ASSERT( blockIndex >= 0 && std::size_t( blockIndex ) < m_dataBlocks.size(), "blockIndex is out of range!!!" + std::to_string( blockIndex ) );
        ARIES_ASSERT( offset >= m_blockSizePrefixSum[blockIndex],
                "offset should >= m_blockSizePrefixSum[ blockIndex ], offset=" + std::to_string( offset ) );
        int64_t offsetInBlock = offset - m_blockSizePrefixSum[blockIndex];
        int blockCount = m_dataBlocks.size();
        int64_t copyCount;
        int64_t dataTypeSizeInBytes = m_columnType.GetDataTypeSize();
        int64_t totalCount = 0;
        int leftRows = count;
        for( int i = blockIndex; i < blockCount && leftRows > 0; ++i )
        {
            const AriesDataBufferSPtr& data = m_dataBlocks[i];
            ARIES_ASSERT( offsetInBlock >= 0 && std::size_t( offsetInBlock ) < data->GetItemCount(),
                    "offsetInBlock is out of range, offsetInBlock=" + std::to_string( offsetInBlock ) );
            if( offsetInBlock > 0 )
            {
                //当前分块剩下的数据行数
                int64_t leftCount = data->GetItemCount() - offsetInBlock;

                //干脆一起拷贝
                if( leftCount < leftRows + margin )
                    copyCount = leftCount;
                else
                    copyCount = leftRows;

                AriesDataBufferSPtr dataBuf = std::make_shared< AriesDataBuffer >( m_columnType, copyCount );
                dataBuf->PrefetchToCpu();
                data->PrefetchToCpu();
                memcpy( dataBuf->GetData(), data->GetData() + offsetInBlock * dataTypeSizeInBytes, dataBuf->GetTotalBytes() );
                totalCount += copyCount;
                leftRows -= copyCount;
                offsetInBlock = 0;
                result.push_back( dataBuf );
            }
            else
            {
                int64_t itemCount = data->GetItemCount();
                if( itemCount <= leftRows + margin )
                {
                    //直接添加
                    result.push_back( m_dataBlocks[i] );
                    totalCount += itemCount;
                    leftRows -= itemCount;
                }
                else if( leftRows > margin )
                {
                    //需要切分
                    copyCount = leftRows;
                    AriesDataBufferSPtr dataBuf = std::make_shared< AriesDataBuffer >( m_columnType, copyCount );
                    dataBuf->PrefetchToCpu();
                    data->PrefetchToCpu();
                    memcpy( dataBuf->GetData(), data->GetData(), dataBuf->GetTotalBytes() );
                    totalCount += copyCount;
                    leftRows -= copyCount;
                    result.push_back( dataBuf );
                }
                else
                    break; //数据差不多够了，不用凑满
            }
        }

        count = totalCount;
        return result;
    }

    AriesDataBufferSPtr AriesColumn::GetDataBuffer()
    {
        return GetDataBufferByGpu();
    }

    int64_t AriesColumn::GetRowCount() const
    {
        return m_totalCount;
    }

    AriesBaseColumnSPtr AriesColumn::Clone() const
    {
        AriesColumnSPtr result = std::make_shared< AriesColumn >( *this );
        return result;
    }

    //按照 indices 生成buffer给外界
    AriesDataBufferSPtr AriesColumn::GetDataBufferByCpu( const AriesIndicesArraySPtr& indices, bool hasNull ) const
    {
        PrefetchDataToCpu();
        indices->PrefetchToCpu();
        return GetDataBufferByIndices( indices, m_dataBlocks, m_blockSizePrefixSum, hasNull, m_columnType );
    }

    int8_t* AriesColumn::GetFieldContent( index_t index ) const
    {
        ARIES_ASSERT( index >= 0 && index < m_totalCount, "Invalid index!!!" );
        auto it = std::upper_bound( m_blockSizePrefixSum.cbegin(), m_blockSizePrefixSum.cend(), index );
        int blockIndex = ( it - m_blockSizePrefixSum.cbegin() ) - 1;
        int64_t offsetInBlock = index - m_blockSizePrefixSum[blockIndex];
        return m_dataBlocks[blockIndex]->GetItemDataAt( offsetInBlock );
    }

    void AriesColumn::UpdateFieldContent( index_t index, int8_t* newData )
    {
        int8_t* oldData = GetFieldContent( index );
        memcpy( oldData, newData, m_columnType.GetDataTypeSize() );
    }

    //生成一个buffer给外界，大小精确为count
    AriesDataBufferSPtr AriesColumn::GetDataBufferByCpu( int64_t offset, int64_t count ) const
    {
        PrefetchDataToCpu();
        int64_t dataTypeSize = m_columnType.GetDataTypeSize();
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( m_columnType, count );
        result->PrefetchToCpu();
        int8_t* pOutput = result->GetData();

        auto it = std::upper_bound( m_blockSizePrefixSum.cbegin(), m_blockSizePrefixSum.cend(), offset );
        int blockIndex = ( it - m_blockSizePrefixSum.cbegin() ) - 1;
        int64_t offsetInBlock = offset - m_blockSizePrefixSum[blockIndex];
        int blockCount = m_dataBlocks.size();
        int64_t copyCount;
        int64_t copySizeInBytes;
        int64_t leftRows = count;
        for( int i = blockIndex; i < blockCount && leftRows > 0; ++i )
        {
            const AriesDataBufferSPtr& data = m_dataBlocks[i];
            data->PrefetchToCpu();
            copyCount = std::min( ( int64_t )data->GetItemCount() - offsetInBlock, leftRows );
            copySizeInBytes = copyCount * dataTypeSize;
            memcpy( pOutput, data->GetData() + offsetInBlock * dataTypeSize, copySizeInBytes );
            leftRows -= copyCount;
            pOutput += copySizeInBytes;
            offsetInBlock = 0;
        }
        return result;
    }

    //将m_dataBlocks中的block合并，并返回合并后的block的databuffer
    AriesDataBufferSPtr AriesColumn::GetDataBufferByCpu()
    {
        if( m_dataBlocks.size() == 1 )
            return m_dataBlocks[0];

        PrefetchDataToCpu();
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( m_columnType, m_totalCount );
        result->PrefetchToCpu();
        int64_t copySizeInBytes;
        int8_t* pOutput = result->GetData();
        for( const auto it : m_dataBlocks )
        {
            it->PrefetchToCpu();
            copySizeInBytes = it->GetTotalBytes();
            memcpy( pOutput, it->GetData(), copySizeInBytes );
            pOutput += copySizeInBytes;
        }
        m_dataBlocks.clear();
        m_dataBlocks.push_back( result );
        m_blockSizePrefixSum.clear();
        m_blockSizePrefixSum.push_back( 0 );
        m_blockSizePrefixSum.push_back( m_totalCount );
        return result;
    }

    AriesDataBufferSPtr AriesColumn::GetDataBufferByGpu()
    {
        if( m_dataBlocks.size() == 1 )
            return m_dataBlocks[0];

        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( m_columnType, m_totalCount );
        result->PrefetchToGpu();
        int64_t copySizeInBytes;
        int8_t* pOutput = result->GetData();
        for( const auto it : m_dataBlocks )
        {
            it->PrefetchToGpu();
            copySizeInBytes = it->GetTotalBytes();
            cudaMemcpy( pOutput, it->GetData(), copySizeInBytes, cudaMemcpyDefault );
            // memcpy( pOutput, it->GetData(), copySizeInBytes );
            pOutput += copySizeInBytes;
        }
        m_dataBlocks.clear();
        m_dataBlocks.push_back( result );
        m_blockSizePrefixSum.clear();
        m_blockSizePrefixSum.push_back( 0 );
        m_blockSizePrefixSum.push_back( m_totalCount );
        return result;
    }

    AriesColumnType AriesColumn::GetColumnType() const
    {
        return m_columnType;
    }

    void AriesColumn::PrefetchDataToCpu() const
    {
        for( const auto dataBlock : m_dataBlocks )
            dataBlock->PrefetchToCpu();
    }

    void AriesColumn::PrefetchDataToGpu() const
    {
        for( const auto dataBlock : m_dataBlocks )
            dataBlock->PrefetchToGpu();
    }

    void AriesColumn::MaterilizeSelfByIndices( const AriesIndicesSPtr& indices )
    {
        auto buffer = MaterializeColumn( m_dataBlocks,
                                         GetPrefixSumOfBlockSize( m_blockSizePrefixSum ),
                                         indices->GetIndices(),
                                         m_columnType );

        m_dataBlocks.clear();
        m_blockSizePrefixSum.clear();
        m_totalCount = 0;
        m_blockSizePrefixSum.push_back( 0 );
        AddDataBuffer( buffer );
    }

    AriesIndices::AriesIndices()
            : m_bHasNull( false ), m_rowCount( 0 )
    {
        m_blockSizePrefixSum.push_back( 0 );
    }

    int64_t AriesIndices::GetRowCount() const
    {
        return m_rowCount;
    }

    void AriesIndices::SetHasNull( bool bHasNull )
    {
        m_bHasNull = bHasNull;
    }

    bool AriesIndices::HasNull() const
    {
        return m_bHasNull;
    }

    void AriesIndices::Clear()
    {
        m_indices.clear();
        m_rowCount = 0;
        m_blockSizePrefixSum.clear();
        m_blockSizePrefixSum.push_back( 0 );
    }

    AriesIndicesSPtr AriesIndices::Clone() const
    {
        AriesIndicesSPtr result = std::make_shared< AriesIndices >( *this );
        vector< AriesIndicesArraySPtr > tmp;
        for( const auto it : m_indices )
            tmp.push_back( it->Clone() );
        result->m_indices = tmp;
        return result;
    }

    void AriesIndices::AddIndices( const AriesIndicesArraySPtr& indices )
    {
        ARIES_ASSERT( indices->GetItemCount() > 0, "indices's size should > 0 !!!" );
        int64_t count = indices->GetItemCount();
        m_rowCount += count;
        m_blockSizePrefixSum.push_back( m_blockSizePrefixSum.back() + count );
        ARIES_ASSERT( m_blockSizePrefixSum.back() == m_rowCount,
                "m_rowCount should be equal to m_blockSizes.back()!!!" + std::to_string( m_blockSizePrefixSum.back() ) + " != "
                        + std::to_string( m_rowCount ) );
        m_indices.push_back( indices );
    }

    const vector< AriesIndicesArraySPtr >& AriesIndices::GetIndicesArray() const
    {
        return m_indices;
    }

    AriesIndicesArraySPtr AriesIndices::GetIndices()
    {
        ARIES_ASSERT( !m_indices.empty(), "m_indices is empty" );
        return GetIndicesByGpu();
    }

    AriesIndicesArraySPtr AriesIndices::GetIndices( int64_t offset, int64_t count ) const
    {
        ARIES_ASSERT( !m_indices.empty(), "m_indices is empty" );
        ARIES_ASSERT( offset + count <= m_rowCount, "offset or count are out of range!!!" );
        return GetIndicesByGpu( offset, count );
    }

    void AriesIndices::PrefetchDataToCpu() const
    {
        for( const auto indices : m_indices )
            indices->PrefetchToCpu();
    }

    void AriesIndices::PrefetchDataToGpu() const
    {
        for( const auto indices : m_indices )
            indices->PrefetchToGpu();
    }

    const vector< int64_t >& AriesIndices::GetBlockSizePsumArray() const
    {
        return m_blockSizePrefixSum;
    }

    void AriesIndices::MoveToDevice( int deviceId )
    {
        int deviceCount;
        cudaGetDeviceCount( &deviceCount );
        assert( deviceId >= 0 && deviceId < deviceCount && m_rowCount > 0 );

        int32_t oldDeviceId;
        cudaGetDevice( &oldDeviceId );

        cudaSetDevice( deviceId );
        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( m_rowCount );
        auto* pOutput = result->GetData();
        for( const auto it : m_indices )
        {
            cudaMemcpy( pOutput, it->GetData(), it->GetTotalBytes(), cudaMemcpyDeviceToDevice );
            pOutput += it->GetItemCount();
        }

        m_indices.clear();
        m_indices.push_back( result );
        m_blockSizePrefixSum.clear();
        m_blockSizePrefixSum.push_back( 0 );
        m_blockSizePrefixSum.push_back( m_rowCount );

        cudaSetDevice( oldDeviceId );
    }

    AriesIndicesArraySPtr AriesIndices::GetIndicesByCpu()
    {
        if( m_indices.size() == 1 )
            return m_indices[0];
        PrefetchDataToCpu();
        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( m_rowCount );
        result->PrefetchToCpu();
        auto* pOutput = result->GetData();
        for( const auto it : m_indices )
        {
            memcpy( pOutput, it->GetData(), it->GetTotalBytes() );
            pOutput += it->GetItemCount();
        }
        if( m_rowCount > 0 )
        {
            m_indices.clear();
            m_indices.push_back( result );
            m_blockSizePrefixSum.clear();
            m_blockSizePrefixSum.push_back( 0 );
            m_blockSizePrefixSum.push_back( m_rowCount );
        }
        return result;
    }

    AriesIndicesArraySPtr AriesIndices::GetIndicesByCpu( int64_t offset, int64_t count ) const
    {
        PrefetchDataToCpu();
        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( count );
        result->PrefetchToCpu();
        auto* pOutput = result->GetData();

        auto it = std::upper_bound( m_blockSizePrefixSum.cbegin(), m_blockSizePrefixSum.cend(), offset );
        int blockIndex = ( it - m_blockSizePrefixSum.cbegin() ) - 1;
        int64_t offsetInBlock = offset - m_blockSizePrefixSum[blockIndex];
        int blockCount = m_indices.size();
        int64_t copyCount;
        int64_t leftRows = count;
        for( int i = blockIndex; i < blockCount && leftRows > 0; ++i )
        {
            const AriesIndicesArraySPtr& data = m_indices[i];
            copyCount = std::min( ( int64_t )data->GetItemCount() - offsetInBlock, leftRows );
            memcpy( pOutput, data->GetData() + offsetInBlock, copyCount * sizeof(index_t) );
            leftRows -= copyCount;
            pOutput += copyCount;
            offsetInBlock = 0;
        }
        return result;
    }

    AriesIndicesArraySPtr AriesIndices::GetIndicesByGpu()
    {
        if( m_indices.size() == 1 )
            return m_indices[0];
        PrefetchDataToGpu();
        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( m_rowCount );
        result->PrefetchToGpu();
        auto* pOutput = result->GetData();
        for( const auto it : m_indices )
        {
            cudaMemcpy( pOutput, it->GetData(), it->GetTotalBytes(), cudaMemcpyDefault );
            pOutput += it->GetItemCount();
        }
        if( m_rowCount > 0 )
        {
            m_indices.clear();
            m_indices.push_back( result );
            m_blockSizePrefixSum.clear();
            m_blockSizePrefixSum.push_back( 0 );
            m_blockSizePrefixSum.push_back( m_rowCount );
        }
        return result;
    }

    AriesIndicesArraySPtr AriesIndices::GetIndicesByGpu( int64_t offset, int64_t count ) const
    {
        AriesIndicesArraySPtr result = std::make_shared< AriesIndicesArray >( count );
        auto* pOutput = result->GetData();

        auto it = std::upper_bound( m_blockSizePrefixSum.cbegin(), m_blockSizePrefixSum.cend(), offset );
        int blockIndex = ( it - m_blockSizePrefixSum.cbegin() ) - 1;
        int64_t offsetInBlock = offset - m_blockSizePrefixSum[blockIndex];
        int blockCount = m_indices.size();
        int64_t copyCount;
        int64_t leftRows = count;
        for( int i = blockIndex; i < blockCount && leftRows > 0; ++i )
        {
            const AriesIndicesArraySPtr& data = m_indices[i];
            copyCount = std::min( ( int64_t )data->GetItemCount() - offsetInBlock, leftRows );
            cudaMemcpy( pOutput, data->GetData() + offsetInBlock, copyCount * sizeof(index_t), cudaMemcpyDefault );
            leftRows -= copyCount;
            pOutput += copyCount;
            offsetInBlock = 0;
        }
        return result;
    }

    AriesColumnReference::AriesColumnReference( const AriesBaseColumnSPtr& refColumn )
            : m_refColumn( refColumn )
    {
        ARIES_ASSERT( refColumn, "refColum:77n is null" );
    }

    void AriesColumnReference::SetIndices( const AriesIndicesSPtr& indices )
    {
        ARIES_ASSERT( indices, "indices is null" );
        m_indices = indices;
        m_mtrlzedBuff = nullptr;
    }

    AriesIndicesSPtr AriesColumnReference::GetIndices() const
    {
        ARIES_ASSERT( m_indices, "m_indices is null" );
        return m_indices;
    }

    int64_t AriesColumnReference::GetRowCount() const
    {
        return m_indices->GetRowCount();
    }

    AriesBaseColumnSPtr AriesColumnReference::GetReferredColumn() const
    {
        return m_refColumn;
    }

    ColumnRefRowInfo AriesColumnReference::GetRowInfo() const
    {
        return
        {   m_refColumn->GetRowCount(), GetRowCount()};
    }

    bool AriesColumnReference::IsMaterializeNeeded() const
    {
        size_t currentSize = m_refColumn->GetRowCount() * m_refColumn->GetColumnType().GetDataTypeSize() + m_indices->GetRowCount() * sizeof(index_t);
        size_t materializedSize = m_indices->GetRowCount() * m_refColumn->GetColumnType().GetDataTypeSize();
        if( materializedSize >= currentSize )
            LOG( INFO )<<"IsMaterializeNeeded: false, "<< "materializedSize:"<<materializedSize<<", currentSize:"<<currentSize<<endl;
        return materializedSize < currentSize;
    }

    AriesColumnReferenceSPtr AriesColumnReference::CloneWithEmptyContent() const
    {
        return std::make_shared< AriesColumnReference >( m_refColumn );
    }

    AriesColumnReferenceSPtr AriesColumnReference::Clone() const
    {
        auto column = m_refColumn->Clone();
        auto columnRef = make_shared< AriesColumnReference >( column );
        columnRef->SetIndices( m_indices->Clone() );
        return columnRef;
    }

    AriesDataBufferSPtr AriesColumnReference::GetDataBuffer( bool bRunOnGpu )
    {
        return m_refColumn->GetDataBuffer( m_indices->GetIndices(), m_indices->HasNull(), bRunOnGpu );
        // if( !m_mtrlzedBuff )
        //     m_mtrlzedBuff = m_refColumn->GetDataBuffer( m_indices, bRunOnGpu );
        // return m_mtrlzedBuff;
    }

    AriesDataBufferSPtr AriesColumnReference::GetDataBufferUsingIndices( const AriesIndicesArraySPtr& indices, bool hasNull, bool bRunOnGpu ) const
    {
        return m_refColumn->GetDataBuffer( indices, hasNull, bRunOnGpu );
    }

    AriesDictEncodedColumn::AriesDictEncodedColumn( const aries::AriesDictSPtr& dict, const AriesVariantIndicesSPtr& indices )
            : m_dict( dict ), m_indices( indices )
    {
        m_columnType = AriesColumnType{ { AriesValueType::CHAR, ( int )dict->GetSchemaSize() }, dict->IsNullable(), false };
        m_encodeType = EncodeType::DICT;
    }

    bool AriesDictEncodedColumn::IsGpuMaterilizeBetter( size_t resultCount )
    {
        return true;
    }

    AriesBaseColumnSPtr AriesDictEncodedColumn::Clone() const
    {
        auto dictIndices = std::dynamic_pointer_cast< AriesColumn >( m_indices->Clone() );
        auto result = std::make_shared< AriesDictEncodedColumn >( m_dict, dictIndices );
        return result;
    }

    AriesBaseColumnSPtr AriesDictEncodedColumn::CloneWithNoContent() const
    {
        auto dictIndices = std::dynamic_pointer_cast< AriesColumn >( m_indices->CloneWithNoContent() );
        auto result = std::make_shared< AriesDictEncodedColumn >( m_dict, dictIndices );
        return result;
    }

    aries::AriesDictSPtr AriesDictEncodedColumn::GetDict() const
    {
        return m_dict;
    }
    AriesDataBufferSPtr AriesDictEncodedColumn::GetDictDataBuffer() const
    {
        return m_dict->getDictBuffer();
    }

    AriesVariantIndicesSPtr AriesDictEncodedColumn::GetIndices() const
    {
        return m_indices;
    }

    int8_t* AriesDictEncodedColumn::GetFieldContent( index_t index ) const
    {
        ARIES_ASSERT( index >= 0 && index < GetRowCount(), "Invalid index!!!" );
        index_t dataIdx = *( ( index_t* )m_indices->GetFieldContent( index ) );
        return m_dict->getDictBuffer()->GetItemDataAt( dataIdx );
    }

    AriesDataBufferSPtr AriesDictEncodedColumn::GetDataBuffer( const AriesIndicesSPtr& indices, bool bRunOnGpu )
    {
        vector< AriesVariantIndicesArraySPtr > input;
        auto oldIndices = m_indices->GetDataBuffer();
        input.push_back( oldIndices );

        // 1. 先对字典索引进行物化
        // 非nullable的字典列经过outer join之后，可能会变成nullable字典列
        AriesColumnType columnType = m_indices->GetColumnType();
        columnType.HasNull |= indices->HasNull();
        vector< int64_t > dataPsumArray;
        dataPsumArray.push_back( 0 );
        dataPsumArray.push_back( oldIndices->GetItemCount() );
        auto dataPrefixSum = GetPrefixSumOfBlockSize( dataPsumArray );
        auto indicesPrefixSum = GetPrefixSumOfBlockSize( indices->GetBlockSizePsumArray() );
        auto newIndices = MaterializeColumn( input,
                                             dataPrefixSum,
                                             indices->GetIndicesArray(),
                                             indicesPrefixSum,
                                             columnType );

        // 2. 再进行字典物化
        columnType = m_columnType;
        columnType.HasNull |= indices->HasNull();

        vector< AriesDataBufferSPtr > dataBlocks;
        m_dict->getDictBuffer()->PrefetchToGpu();
        dataBlocks.push_back( m_dict->getDictBuffer() );

        dataPsumArray.clear();
        dataPsumArray.push_back( 0 );
        dataPsumArray.push_back( m_dict->getDictBuffer()->GetItemCount() );
        dataPrefixSum = GetPrefixSumOfBlockSize( dataPsumArray );

        return MaterializeColumn( dataBlocks, dataPrefixSum, newIndices, columnType );
    }

    AriesDataBufferSPtr AriesDictEncodedColumn::GetDataBuffer( const AriesIndicesArraySPtr& indices,
                                                               bool hasNull,
                                                               bool bRunOnGpu )
    {
        ARIES_ASSERT( indices && indices->GetItemCount() > 0, "indices is null or empty!!!" );
        assert( IndicesOk( indices, GetRowCount(), hasNull ) );
        AriesDataBufferSPtr result;

        if( IsGpuMaterilizeBetter( indices->GetItemCount() ) )
        {
            vector< AriesVariantIndicesArraySPtr > input;
            auto oldIndices = m_indices->GetDataBuffer();
            input.push_back( oldIndices );

            // 1. 先对字典索引进行物化
            // 非nullable的字典列经过outer join之后，可能会变成nullable字典列
            AriesColumnType columnType = m_indices->GetColumnType();
            columnType.HasNull |= hasNull;
            vector< int64_t > dataPsumArray;
            dataPsumArray.push_back( 0 );
            dataPsumArray.push_back( oldIndices->GetItemCount() );
            auto dataPrefixSum = GetPrefixSumOfBlockSize( dataPsumArray );
            auto newIndices = MaterializeColumn( input,
                                                 dataPrefixSum,
                                                 indices,
                                                 columnType );

            // 2. 再进行字典物化
            columnType = m_columnType;
            columnType.HasNull |= hasNull;

            vector< AriesDataBufferSPtr > dataBlocks;
            m_dict->getDictBuffer()->PrefetchToGpu();
            dataBlocks.push_back( m_dict->getDictBuffer() );

            dataPsumArray.clear();
            dataPsumArray.push_back( 0 );
            dataPsumArray.push_back( m_dict->getDictBuffer()->GetItemCount() );
            dataPrefixSum = GetPrefixSumOfBlockSize( dataPsumArray );

            result = MaterializeColumn( dataBlocks, dataPrefixSum, newIndices, columnType );
        }
        else
        {
            // TODO: 实现CPU物化
            // result = GetDataBufferByCpu( indices, hasNull );
            assert( 0 );
        }
        return result;
    }

    AriesDataBufferSPtr AriesDictEncodedColumn::GetDataBuffer()
    {
        if( m_mtrlzedBuff )
            return m_mtrlzedBuff;

        AriesDataBufferSPtr result;
        auto indices = m_indices->GetDataBuffer();

        // if( IsGpuMaterilizeBetter( indices ) )
        // {
        AriesColumnType columnType = m_columnType;
        columnType.HasNull |= m_indices->GetColumnType().isNullable();
        vector< AriesDataBufferSPtr > dataBlocks;
        m_dict->getDictBuffer()->PrefetchToGpu();
        dataBlocks.push_back( m_dict->getDictBuffer() );

        vector< int64_t > dataPsumArray;
        dataPsumArray.push_back( 0 );
        auto dataBlocksPrefixSum = GetPrefixSumOfBlockSize( dataPsumArray );

        result = MaterializeColumn( dataBlocks, dataBlocksPrefixSum, indices, columnType );
        // }
        // else
        // {
        //     // TODO: 实现CPU物化
        //     // result = GetDataBufferByCpu( indices, hasNull );
        //     assert( 0 );
        // }
        m_mtrlzedBuff = result;
        return result;

    }

    void AriesDictEncodedColumn::MaterilizeSelfByIndices( const AriesIndicesSPtr& indices )
    {
        m_mtrlzedBuff = nullptr;
        m_indices->MaterilizeSelfByIndices( indices );
    }

    AriesTableBlock::AriesTableBlock()
            : m_rowCount( 0 ), m_deviceId( cudaCpuDeviceId ), m_partitionID( -1 ), m_partitionedColumnID( -1 )
    {
    }

    string AriesTableBlock::GetColumnName( int32_t columnId ) const
    {
        const auto it = m_columnNames.find( columnId );
        ARIES_ASSERT( it != m_columnNames.end(), "id doesn't exists" );
        return it->second;
    }

    void AriesTableBlock::AddColumnName( int32_t columnId, const string& name )
    {
        ARIES_ASSERT( m_columnNames.find( columnId ) == m_columnNames.end(), "id already exists" );
        m_columnNames[columnId] = name;
    }

    AriesLiteralValue AriesTableBlock::GetLiteralValue()
    {
        AriesDataBufferSPtr buf = GetColumnBuffer( 1 );
        ARIES_ASSERT( GetRowCount() == 1, "GetRowCount() should be 0 but it is: " + to_string( GetRowCount() ) );

        return ConvertRawDataToLiteral( buf->GetData(), buf->GetDataType() );
    }

    //添加属于这个table block的column。不能重复添加！
    //列如scan node会调用这个函数仅仅一次。后续block的添加由scan node通过自己保存的AriesColumnSPtr直接添加
    //该函数在进行数据输出时可能被调用，在收集数据时不应该被调用！
    void AriesTableBlock::AddColumn( int32_t columnId, AriesColumnSPtr column )
    {
        ARIES_ASSERT( !ColumnExists( columnId ), "id already exists" );
        m_columns[columnId] = column;
    }

    void AriesTableBlock::AddColumn( int32_t columnId, AriesColumnReferenceSPtr columnRef )
    {
        ARIES_ASSERT( !ColumnExists( columnId ), "id already exists" );
        m_columnReferences[columnId] = columnRef;
        m_indices.push_back( columnRef->GetIndices() );
    }

    void AriesTableBlock::AddColumn( int32_t columnId, const AriesDictEncodedColumnSPtr& dictColumn )
    {
        ARIES_ASSERT( !ColumnExists( columnId ), "id already exists" );
        m_dictEncodedColumns[columnId] = dictColumn;
    }

    void AriesTableBlock::AddColumn( const map< int32_t, AriesColumnSPtr >& columns, int64_t offset, int64_t count )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        ARIES_ASSERT( m_columns.empty(), "m_columns should be empty" );
        ARIES_ASSERT( m_columnReferences.empty(), "m_columnReferences should be empty" );
        ARIES_ASSERT( m_dictEncodedColumns.empty(), "m_dictEncodedColumns should be empty" );
        for( const auto it : columns )
        {
            AriesColumnSPtr col = std::make_shared< AriesColumn >();
            for( const auto buf : it.second->GetDataBuffers( offset, count, true ) )
                col->AddDataBuffer( buf );
            m_columns.insert(
            { it.first, col } );
        }
#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        m_stats.m_materializeTime += timeCost;
        LOG( INFO )<< "-----------------------AriesTableBlock::AddColumn time cost is:" << timeCost;
#endif
    }

    //用一个column替换表中对应的column(只能替换掉原表中对应物化的列，非物化的列不能替换），目前只有sort节点可能会调用该函数(建议调用带多个参数的UpdateIndices一步到位）。其他场景支持，根据具体需求再进行设计
    void AriesTableBlock::UpdateColumn( int32_t columnId, AriesColumnSPtr column )
    {
        ARIES_ASSERT( m_columns.find( columnId ) != m_columns.end(), "id doesn't exists" );
        m_columns[columnId] = column;
    }

    void AriesTableBlock::RemoveColumns( const set< int32_t >& toRemove )
    {
        for( auto id : toRemove )
        {
            ARIES_ASSERT( ColumnExists( id ), "column doesn't exist, columnId=" + std::to_string( id ) );
            const auto itCol = m_columns.find( id );
            if( itCol != m_columns.end() )
                m_columns.erase( itCol );
            else
            {
                auto itDictColum = m_dictEncodedColumns.find( id );
                if( m_dictEncodedColumns.end() != itDictColum )
                    m_dictEncodedColumns.erase( itDictColum );
                else
                {
                    const auto itColRef = m_columnReferences.find( id );
                    ARIES_ASSERT( itColRef != m_columnReferences.end(), "id doesn't exists" );
                    m_columnReferences.erase( itColRef );
                }
            }
        }

        CleanupIndices();
    }

    void AriesTableBlock::KeepColumns( const set< int32_t >& toKeep )
    {
        map< int32_t, AriesColumnSPtr > newCols;
        map< int32_t, AriesDictEncodedColumnSPtr > newDictCols;
        map< int32_t, AriesColumnReferenceSPtr > newColRefs;
        for( auto id : toKeep )
        {
            ARIES_ASSERT( ColumnExists( id ), "column doesn't exist, columnId=" + std::to_string( id ) );
            const auto itCol = m_columns.find( id );
            if( itCol != m_columns.end() )
                newCols[id] = itCol->second;
            else
            {
                auto itDictColum = m_dictEncodedColumns.find( id );
                if( m_dictEncodedColumns.end() != itDictColum )
                    newDictCols[id] = itDictColum->second;
                else
                {
                    const auto itColRef = m_columnReferences.find( id );
                    ARIES_ASSERT( itColRef != m_columnReferences.end(), "id doesn't exists" );
                    newColRefs[id] = itColRef->second;
                }
            }
        }
        m_columns = std::move( newCols );
        m_dictEncodedColumns = std::move( newDictCols );
        m_columnReferences = std::move( newColRefs );

        CleanupIndices();
    }

    void AriesTableBlock::UpdateColumnIds( const map< int32_t, int32_t >& idNewAndOld )
    {
        map< int32_t, AriesColumnSPtr > newCols;
        map< int32_t, AriesDictEncodedColumnSPtr > newDictCols;
        map< int32_t, AriesColumnReferenceSPtr > newColRefs;
        int32_t newId;
        int32_t oldId;
        vector< int32_t > colToRemove;
        vector< int32_t > dictColToRemove;
        vector< int32_t > colRefToRemove;
        //生成新数据
        for( const auto pair : idNewAndOld )
        {
            ARIES_ASSERT( ColumnExists( pair.second ), "column doesn't exist, columnId=" + std::to_string( pair.second ) );
            newId = pair.first;
            oldId = pair.second;
            const auto itCol = m_columns.find( oldId );
            if( itCol != m_columns.end() )
            {
                newCols[newId] = itCol->second;
                colToRemove.push_back( oldId );
            }
            else
            {
                auto itDictColum = m_dictEncodedColumns.find( oldId );
                if( m_dictEncodedColumns.end() != itDictColum )
                {
                    newDictCols[newId] = itDictColum->second;
                    dictColToRemove.push_back( oldId );
                }
                else
                {
                    const auto itColRef = m_columnReferences.find( oldId );
                    ARIES_ASSERT( itColRef != m_columnReferences.end(), "column id doesn't exists" + std::to_string( oldId ) );
                    newColRefs[newId] = itColRef->second;
                    colRefToRemove.push_back( oldId );
                }
            }
        }
        //删除老数据
        for( auto id : colToRemove )
            m_columns.erase( id );
        for( auto id : dictColToRemove )
            m_dictEncodedColumns.erase( id );
        for( auto id : colRefToRemove )
            m_columnReferences.erase( id );
        //合并新数据
        m_columns.insert( newCols.begin(), newCols.end() );
        m_dictEncodedColumns.insert( newDictCols.begin(), newDictCols.end() );
        m_columnReferences.insert( newColRefs.begin(), newColRefs.end() );
    }

    //获取该table block包含的总数据行数
    int64_t AriesTableBlock::GetRowCount() const
    {
        if( m_rowCount > 0 )
            return m_rowCount;

        int64_t count = 0;
        const auto it = m_columns.cbegin();
        if( it != m_columns.cend() )
            count = it->second->GetRowCount();
        else
        {
            auto itDictColum = m_dictEncodedColumns.cbegin();
            if( m_dictEncodedColumns.cend() != itDictColum )
                count = itDictColum->second->GetRowCount();
            else
            {
                const auto it2 = m_columnReferences.cbegin();
                if( it2 != m_columnReferences.cend() )
                    count = it2->second->GetRowCount();
            }
        }
        return count;
    }

    int64_t AriesTableBlock::GetColumnCount() const
    {
        return m_columns.size() + m_dictEncodedColumns.size() + m_columnReferences.size();
    }

    void AriesTableBlock::SetRowCount( int64_t count )
    {
        m_rowCount = count;
    }

    void AriesTableBlock::MoveIndicesToDevice( int deviceId )
    {
        if( deviceId != m_deviceId )
        {
            for( auto& ind : m_indices )
                ind->MoveToDevice( deviceId );
            m_deviceId = deviceId;
        }
    }
    //当数据接收端需要累积数据进行处理时，调用此函数收集收到的table block。对调用者而言，收集到的table block合并成为一个整体对外提供数据。
    //物化的拼接到一个AriesColumn中，通过AddDataBuffer合并数据（不会memcpy)，indices直接拼接，需要memcpy
    void AriesTableBlock::AddBlock( AriesTableBlockUPtr table )
    {
        if( table->GetRowCount() > 0 )
        {
            if( 0 == GetRowCount() )
            {
                m_columns = std::move( table->m_columns );
                m_dictEncodedColumns = std::move( table->m_dictEncodedColumns );
                m_columnReferences = std::move( table->m_columnReferences );
                m_indices = std::move( table->m_indices );
                m_rowCount = table->m_rowCount;
                return;
            }
#ifdef ARIES_PROFILE
            aries::CPU_Timer t;
            t.begin();
#endif
            //拼接物化的column
            for( const auto it : table->m_columns )
            {
                auto itThis = m_columns.find( it.first );
                ARIES_ASSERT( itThis != m_columns.end(), "can't find the column id!!!" );
                for( const auto buf : it.second->GetDataBuffers() )
                    itThis->second->AddDataBuffer( buf );
            }

            // 字典压缩的列，只需要拼接索引
            for( const auto it : table->m_dictEncodedColumns )
            {
                auto itThis = m_dictEncodedColumns.find( it.first );
                ARIES_ASSERT( itThis != m_dictEncodedColumns.end(), "can't find the column id!!!" );
                auto newIndices = it.second->GetIndices()->GetDataBuffers();
                for( auto& ind : newIndices )
                    itThis->second->GetIndices()->AddDataBuffer( ind );
            }

            //处理非物化的column，场景有二个：
            //1.两个表的非物化的column引用了同一个物化结果(AriesColumnSPtr相等),那简单的拼接indices即可
            //2.两个表的非物化的column引用了不同的物化结果，那需要将引用的物化列进行拼接，同时在拼接indices时需要加上对应的偏移
            map< AriesIndicesSPtr, int64_t > indicesToModify;
            int64_t offsetToAdd = 0;
            for( const auto it : table->m_columnReferences )
            {
                auto itThis = m_columnReferences.find( it.first );
                ARIES_ASSERT( itThis != m_columnReferences.end(), "can't find the column ref id!!!" );
                auto thisColRef = itThis->second->GetReferredColumn();
                offsetToAdd = thisColRef->GetRowCount();
                auto tableColRef = it.second->GetReferredColumn();
                if( thisColRef != tableColRef )
                {
                    auto colEncodeType = tableColRef->GetEncodeType();
                    switch( colEncodeType )
                    {
                        case EncodeType::NONE:
                        {
                            auto column = std::dynamic_pointer_cast< AriesColumn >( tableColRef );
                            auto thisColumn = std::dynamic_pointer_cast< AriesColumn >( thisColRef );
                            for( const auto buf : column->GetDataBuffers() )
                                thisColumn->AddDataBuffer( buf );
                            indicesToModify.insert(
                            { it.second->GetIndices(), offsetToAdd } );
                            break;
                        }

                        case EncodeType::DICT:
                        {
                            auto column = std::dynamic_pointer_cast< AriesDictEncodedColumn >( tableColRef );
                            auto thisColumn = std::dynamic_pointer_cast< AriesDictEncodedColumn >( thisColRef );
                            for( const auto ind : column->GetIndices()->GetDataBuffers() )
                                thisColumn->GetIndices()->AddDataBuffer( ind );
                            indicesToModify.insert(
                            { it.second->GetIndices(), offsetToAdd } );
                            break;
                        }
                    }
                }
            }

            //调整table中的indices加上对应的偏移
#ifdef ARIES_PROFILE
            aries::CPU_Timer t2;
            t2.begin();
#endif
            for( auto ind : indicesToModify )
            {
                AriesIndicesArraySPtr indices = ind.first->GetIndices();
                AddOffsetToIndices( indices, ind.second );
                //AddOffsetToIndices( ind.first->GetIndices(), ind.first->HasNull(), ind.second );
            }
            //合并indices
            auto count = m_indices.size();
            ARIES_ASSERT( count == table->m_indices.size(), "indices size is different!!!" );
            for( std::size_t i = 0; i < count; ++i )
                m_indices[i]->AddIndices( table->m_indices[i]->GetIndices() );
#ifdef ARIES_PROFILE
            auto timeCost = t2.end();
            m_stats.m_materializeTime += timeCost;            //拼接indices算作物化时间
            LOG( INFO )<< "-----------------------AriesTableBlock AddOffsetToIndices time cost is:" << timeCost;
#endif
            if( m_rowCount > 0 )
                m_rowCount += table->m_rowCount;
#ifdef ARIES_PROFILE
            LOG( INFO )<< "-----------------------AriesTableBlock::AddBlock time cost is:" << t.end() << endl;
#endif
        }
    }

    AriesTableBlockUPtr AriesTableBlock::GetOneBlock( int blockIndex ) const
    {
        assert( m_columnReferences.empty() );
        assert( m_indices.empty() );

        AriesTableBlockUPtr result = std::make_unique< AriesTableBlock >();
        for ( const auto& col : m_columns )
        {
            auto newCol = std::make_shared< AriesColumn >();
            const auto& buffers = col.second->GetDataBuffers();
            newCol->AddDataBuffer( buffers[ blockIndex ] );
            result->m_columns[ col.first ] = newCol;
        }

        for ( const auto& col : m_dictEncodedColumns )
        {
            auto newIndices = std::make_shared< AriesVariantIndices >();
            auto dictIndicesArray = col.second->GetIndices()->GetDataBuffers();
            newIndices->AddDataBuffer( dictIndicesArray[ blockIndex ] );

            auto newCol = std::make_shared< AriesDictEncodedColumn >( col.second->GetDict(), newIndices );
            result->m_dictEncodedColumns[ col.first ] = newCol;
        }

        if ( !m_blockPartitionID.empty() )
        {
            auto it = m_blockPartitionID.find( blockIndex );
            assert( it != m_blockPartitionID.cend() );
            // result->SetBlockPartitionID( blockIndex, it->second );
            result->m_partitionedColumnID = m_partitionedColumnID;
            result->m_partitionID = it->second;
        }

        return result;
    }

    size_t AriesTableBlock::GetBlockCount() const
    {
        assert( m_columnReferences.empty() );
        assert( m_indices.empty() );
        for ( const auto& col : m_columns )
            return col.second->GetDataBuffers().size();
        return 0;
    }

    void AriesTableBlock::SetBlockPartitionID( int blockIndex, int partition )
    {
        m_blockPartitionID[ blockIndex ] = partition;
    }

    // 供range scan使用。
    // 对于非物化的列，index需要是有序的。
    AriesTableBlockUPtr AriesTableBlock::GetSubTable2( int totalSliceCount, int sliceIdx ) const
    {
        ARIES_ASSERT( totalSliceCount > 0, "invalid slice count: " + std::to_string( totalSliceCount ) );
        ARIES_ASSERT( sliceIdx >= 0 && sliceIdx < totalSliceCount, "invalid slice index: " + std::to_string( sliceIdx ) );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesTableBlockUPtr result = std::make_unique< AriesTableBlock >();

        // 生成新的物化列
        for( const auto col : m_columns )
        {
            auto newCol = std::make_shared< AriesColumn >();
            auto buffers = col.second->GetDataBuffers();
            auto totalBuffCount = buffers.size();
            auto buffCountPerSlice = DIV_UP( totalBuffCount, totalSliceCount );
            std::size_t buffIdx = buffCountPerSlice * sliceIdx;
            unsigned long resultBuffCount = 0;
            for( ; resultBuffCount < buffCountPerSlice && buffIdx < totalBuffCount; ++buffIdx, ++resultBuffCount )
                newCol->AddDataBuffer( buffers[buffIdx] );

            result->m_columns[col.first] = newCol;
        }

        // 字典压缩的列，对字典索引进行切割
        for( const auto col : m_dictEncodedColumns )
        {
            auto newIndices = std::make_shared< AriesVariantIndices >();
            auto buffers = col.second->GetIndices()->GetDataBuffers();
            auto totalBuffCount = buffers.size();
            auto buffCountPerSlice = DIV_UP( totalBuffCount, totalSliceCount );
            std::size_t buffIdx = buffCountPerSlice * sliceIdx;
            unsigned long resultBuffCount = 0;
            for( ; resultBuffCount < buffCountPerSlice && buffIdx < totalBuffCount; ++buffIdx, ++resultBuffCount )
                newIndices->AddDataBuffer( buffers[buffIdx] );
            auto newCol = std::make_shared< AriesDictEncodedColumn >( col.second->GetDict(), newIndices );
            result->m_dictEncodedColumns[col.first] = newCol;
        }

        // 非物化的列，将原始数据块进行切分，并将切分后的原始数据块对应的index拷贝并调整偏移
        for( const auto colRef : m_columnReferences )
        {
            auto indices = colRef.second->GetIndices();
            int64_t indiceItemCount = indices->GetRowCount();
            auto indicesArray = colRef.second->GetIndices()->GetIndicesArray();
            AriesManagedArray< index_t* > indicesArrayData( indicesArray.size() );
            indicesArrayData.PrefetchToCpu();
            int i = 0;
            for( auto& indice : indicesArray )
                indicesArrayData[i++] = indice->GetData();
            auto indicesCountPsumData = GetPrefixSumOfBlockSize( indices->GetBlockSizePsumArray() );
            auto referedCol = colRef.second->GetReferredColumn();
            auto colEncodeType = referedCol->GetEncodeType();
            switch( colEncodeType )
            {
                case EncodeType::NONE:
                {
                    auto column = std::dynamic_pointer_cast< AriesColumn >( referedCol );
                    auto buffers = column->GetDataBuffers();
                    auto totalBuffCount = buffers.size();
                    auto buffCountPerSlice = DIV_UP( totalBuffCount, totalSliceCount );

                    auto newCol = make_shared< AriesColumn >();
                    auto newColRef = make_shared< AriesColumnReference >( newCol );
                    auto newIndices = make_shared< AriesIndices >();
                    newColRef->SetIndices( newIndices );
                    result->m_indices.push_back( newIndices );
                    result->m_columnReferences[colRef.first] = newColRef;

                    std::size_t buffIdx = buffCountPerSlice * sliceIdx;
                    int64_t startDataIdx = column->GetBlockSizePsumArray()[buffIdx];
                    int64_t startIndiceIdx = aries_acc::binary_search< bounds_lower >( ( const index_t** )indicesArrayData.GetData(),
                            indicesArrayData.GetItemCount(), indicesCountPsumData->GetData(), indiceItemCount, startDataIdx );

                    unsigned long resultBuffCount = 0;
                    for( ; resultBuffCount < buffCountPerSlice && buffIdx < totalBuffCount; ++buffIdx, ++resultBuffCount )
                    {
                        newCol->AddDataBuffer( buffers[buffIdx] );
                    }

                    auto endDataIdx = column->GetBlockSizePsumArray()[buffIdx];
                    auto endIndiceIdx = aries_acc::binary_search< bounds_lower >( ( const index_t** )indicesArrayData.GetData(),
                            indicesArrayData.GetItemCount(), indicesCountPsumData->GetData(), indiceItemCount, endDataIdx );
                    auto indiceSlice = indices->GetIndices( startIndiceIdx, endIndiceIdx - startIndiceIdx );

                    AddOffsetToIndices( indiceSlice, -startDataIdx );
                    newIndices->AddIndices( indiceSlice );

                    break;
                }
                case EncodeType::DICT:
                {
                    // 非物化的字典压缩的列，对index进行切分
                    auto column = std::dynamic_pointer_cast< AriesDictEncodedColumn >( referedCol );
                    auto buffers = column->GetIndices()->GetDataBuffers();
                    auto totalBuffCount = buffers.size();
                    auto buffCountPerSlice = DIV_UP( totalBuffCount, totalSliceCount );

                    auto newDictIndices = make_shared< AriesVariantIndices >();
                    auto newCol = make_shared< AriesDictEncodedColumn >( column->GetDict(), newDictIndices );
                    auto newColRef = make_shared< AriesColumnReference >( newCol );
                    auto newIndices = make_shared< AriesIndices >();
                    newColRef->SetIndices( newIndices );
                    result->m_indices.push_back( newIndices );
                    result->m_columnReferences[colRef.first] = newColRef;

                    std::size_t buffIdx = buffCountPerSlice * sliceIdx;
                    auto startDataIdx = column->GetIndices()->GetBlockSizePsumArray()[buffIdx];
                    auto startIndiceIdx = aries_acc::binary_search< bounds_lower >( ( const index_t** )indicesArrayData.GetData(),
                            indicesArrayData.GetItemCount(), indicesCountPsumData->GetData(), indiceItemCount, startDataIdx );

                    unsigned long resultBuffCount = 0;
                    for( ; resultBuffCount < buffCountPerSlice && buffIdx < totalBuffCount; ++buffIdx, ++resultBuffCount )
                    {
                        newDictIndices->AddDataBuffer( buffers[buffIdx] );
                    }

                    auto endDataIdx = column->GetIndices()->GetBlockSizePsumArray()[buffIdx];
                    auto endIndiceIdx = aries_acc::binary_search< bounds_lower >( ( const index_t** )indicesArrayData.GetData(),
                            indicesArrayData.GetItemCount(), indicesCountPsumData->GetData(), indiceItemCount, endDataIdx );

                    auto indiceSlice = indices->GetIndices( startIndiceIdx, endIndiceIdx - startIndiceIdx );

                    AddOffsetToIndices( indiceSlice, -startDataIdx );
                    newIndices->AddIndices( indiceSlice );

                    break;
                }
            }
        }
#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        LOG( INFO )<< "-----------------------AriesTableBlock::GetSubTable2 time: " << timeCost;
        m_stats.m_getSubTableTime += timeCost;
#endif
        return result;
    }

    //数据接收端收集完数据后，根据自己的处理能力，进行分块读取。新生成的table block内部包含物化和非物化的列其中
    //1.对于物化的列直接是数据本身
    //2.对于非物化的列，需要保存自己的indices和完整的原始数据块引用(AriesColumnSPtr）
    AriesTableBlockUPtr AriesTableBlock::GetSubTable( int64_t offset, int64_t count, bool bStrict ) const
    {
        ARIES_ASSERT( offset + count <= GetRowCount(), "offset or count are out of range!!! offset + count=" + std::to_string( offset + count ) );
        ARIES_ASSERT( count > 0, "count must > 0!!! count=" + std::to_string( count ) );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesTableBlockUPtr result = std::make_unique< AriesTableBlock >();
        int64_t realCount = count;

        // 生成新的物化列
        for( const auto col : m_columns )
        {
            auto newCol = std::make_shared< AriesColumn >();
            auto buffers = col.second->GetDataBuffers( offset, realCount, bStrict );
            for( const auto buf : buffers )
                newCol->AddDataBuffer( buf );
            result->m_columns[col.first] = newCol;
        }

        for( const auto col : m_dictEncodedColumns )
        {
            auto newIndices = std::make_shared< AriesVariantIndices >();
            auto newIndicesArray = col.second->GetIndices()->GetDataBuffers( offset, realCount, bStrict );
            for( const auto& indices : newIndicesArray )
                newIndices->AddDataBuffer( indices );
            auto newCol = std::make_shared< AriesDictEncodedColumn >( col.second->GetDict(), newIndices );
            result->m_dictEncodedColumns[col.first] = newCol;
        }

        // 生成新的indices信息,将涉及到的indices合并为一整块
        int64_t indiceCount = m_indices.size();
        for( int i = 0; i < indiceCount; ++i )
        {
            auto newIndices = std::make_shared< AriesIndices >();
            newIndices->AddIndices( m_indices[i]->GetIndices( offset, realCount ) );
            newIndices->SetHasNull( m_indices[i]->HasNull() );
            result->m_indices.push_back( newIndices );
        }

        // 生成新的非物化列
        for( const auto colRef : m_columnReferences )
            result->m_columnReferences[colRef.first] = colRef.second->CloneWithEmptyContent();

        //更新非物化的列的indices
        map< int32_t, int32_t > colRefIndiceMapping = GetColRefIndicePos();
        for( const auto & it : colRefIndiceMapping )
            result->m_columnReferences[it.first]->SetIndices( result->m_indices[it.second] );

        //空表
        if( result->GetRowCount() == 0 && result->GetColumnCount() == 0 )
        {
            result->SetRowCount( count );
        }

#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        LOG( INFO )<< "-----------------------AriesTableBlock::GetSubTable time: " << timeCost;
        m_stats.m_getSubTableTime += timeCost;
#endif
        return result;
    }

    //隐式物化，获取某column完整数据。
    //需要遍历m_tables保证数据被完整获取。(sort是典型使用此函数的场景)
    AriesDataBufferSPtr AriesTableBlock::GetColumnBuffer( int32_t columnId, bool bReturnCopy, bool bRunOnGpu )
    {
        bool needCopy = false;
        // LOG( INFO )<< "GetColumnBuffer columnId=" << columnId;
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesDataBufferSPtr result;
        const auto itCol = m_columns.find( columnId );
        if( itCol != m_columns.end() )
        {
            needCopy = ( itCol->second->GetDataBlockCount() == 1 );
            result = itCol->second->GetDataBuffer();
        }
        else
        {
            auto itDictColum = m_dictEncodedColumns.find( columnId );
            if( m_dictEncodedColumns.end() != itDictColum )
            {
                needCopy = true;
                result = itDictColum->second->GetDataBuffer();
            }
            else
            {
                auto colEncodeType = GetColumnEncodeType( columnId );
                const auto itColRef = m_columnReferences.find( columnId );
                ARIES_ASSERT( itColRef != m_columnReferences.end(), "id doesn't exists" );
                result = itColRef->second->GetDataBuffer( bRunOnGpu );

                switch( colEncodeType )
                {
                    case EncodeType::NONE:
                    {
                        auto newCol = std::make_shared< AriesColumn >();
                        newCol->AddDataBuffer( result );
                        m_columns.insert(
                        { columnId, newCol } );
                        m_columnReferences.erase( columnId );
                        break;
                    }

                        // 对于字典压缩的列，后面可能还会参与等于比较，需要保留(q12)。
                    case EncodeType::DICT:
                    {
                        break;
                    }
                }
            }
        }
        if( !m_columnReferences.empty() )
            CheckIfNeedMaterilize();

        CleanupIndices();

        if( bReturnCopy && needCopy )
            result = result->Clone();
#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        m_stats.m_materializeTime += timeCost;
        // LOG(INFO)<< "-----------------------AriesTableBlock::GetColumnBuffer time: " << timeCost;
#endif
        return result;
    }

    void AriesTableBlock::CleanupIndices()
    {
        vector< AriesIndicesSPtr > newIndices;
        for ( auto& indice : m_indices )
        {
            if ( indice.use_count() > 1 )
                newIndices.emplace_back( indice );
        }

        if ( newIndices.size() != m_indices.size() )
            std::swap( m_indices, newIndices );
    }

    AriesDataBufferSPtr AriesTableBlock::GetColumnBufferByIndices( int32_t columnId, const AriesIndicesArraySPtr& indices, bool bRunOnGpu ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        ARIES_ASSERT( indices->GetItemCount() > 0, "indices's size should > 0 !!!" );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesDataBufferSPtr result;
        auto itCol = m_columns.find( columnId );
        if( itCol != m_columns.end() )
            result = itCol->second->GetDataBuffer( indices, false, bRunOnGpu );
        else
        {
            auto itDictColum = m_dictEncodedColumns.find( columnId );
            if( m_dictEncodedColumns.end() != itDictColum )
            {
                result = itDictColum->second->GetDataBuffer( indices, false, bRunOnGpu );
            }
            else
            {
                auto itColRef = m_columnReferences.find( columnId );
                ARIES_ASSERT( itColRef != m_columnReferences.end(), "can't find column" );
                vector< AriesIndicesArraySPtr > input;
                input.push_back( itColRef->second->GetIndices()->GetIndices() );
                //AriesIndicesArraySPtr newIndices = aries_acc::ShuffleIndices( input, indices )[0];
                AriesIndicesArraySPtr newIndices = ::ShuffleIndices( input, indices )[0];
                result = itColRef->second->GetDataBufferUsingIndices( newIndices, itColRef->second->GetIndices()->HasNull(), bRunOnGpu );
            }
        }
#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        m_stats.m_materializeTime += timeCost;
        LOG( INFO )<< "-----------------------AriesTableBlock::GetColumnBufferByIndices time: " << timeCost;
#endif
        return result;
    }

    //返回columnId对应的column是否没有被物化
    bool AriesTableBlock::IsColumnUnMaterilized( int32_t columnId ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        return m_columnReferences.find( columnId ) != m_columnReferences.end();
    }

    //获取未物化的column
    AriesColumnReferenceSPtr AriesTableBlock::GetUnMaterilizedColumn( int32_t columnId ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        const auto itColRef = m_columnReferences.find( columnId );
        ARIES_ASSERT( itColRef != m_columnReferences.end(), "column should be unmaterilized" );
        return itColRef->second;
    }

    EncodeType AriesTableBlock::GetColumnEncodeType( int32_t columnId ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        const auto itCol = m_columns.find( columnId );
        if( itCol != m_columns.end() )
        {
            return itCol->second->GetEncodeType();
        }
        else
        {
            auto itDictCol = m_dictEncodedColumns.find( columnId );
            if( itDictCol != m_dictEncodedColumns.end() )
                return itDictCol->second->GetEncodeType();
            else
            {
                const auto itColRef = m_columnReferences.find( columnId );
                auto refferedCol = itColRef->second->GetReferredColumn();
                return refferedCol->GetEncodeType();
            }
        }
    }

    //获取物化的column,内部可能包含多个data buffer block!
    AriesColumnSPtr AriesTableBlock::GetMaterilizedColumn( int32_t columnId ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        const auto itCol = m_columns.find( columnId );
        ARIES_ASSERT( itCol != m_columns.end(), "column should be materilized" );
        return itCol->second;
    }

    void AriesTableBlock::ShuffleIndices( const AriesIndicesArraySPtr& indices )
    {
        vector< AriesIndicesSPtr > result;
        vector< AriesIndicesArraySPtr > oldIndices;
        auto size = m_indices.size();
        for( std::size_t i = 0; i < size; ++i )
        {
            // CloneWithNoContent产生的空表
            if ( m_indices[i]->GetIndicesArray().empty() )
            {
                auto ind = std::make_shared< AriesIndices >();
                ind->SetHasNull( m_indices[i]->HasNull() );
                ind->AddIndices( indices );
                result.push_back( ind );
            }
            else
            {
                oldIndices.push_back( m_indices[i]->GetIndices() );
            }
        }
        assert( result.empty() || oldIndices.empty() );
        if ( !oldIndices.empty() )
        {
            vector< AriesIndicesArraySPtr > newIndices = aries_acc::ShuffleIndices( oldIndices, indices );
            for( std::size_t i = 0; i < size; ++i )
            {
                auto ind = std::make_shared< AriesIndices >();
                ind->SetHasNull( m_indices[i]->HasNull() );
                ind->AddIndices( newIndices[i] );
                result.push_back( ind );
            }
        }
        std::swap( m_indices, result );
    }

    AriesDataBufferSPtr AriesTableBlock::GetDictEncodedColumnIndiceBuffer( const int32_t columnId )
    {
        AriesDataBufferSPtr indiceBuff;
        AriesDictEncodedColumnSPtr dictColumn;
        if ( IsColumnUnMaterilized( columnId ) )
        {
            auto columnReference = GetUnMaterilizedColumn( columnId );
            dictColumn = std::dynamic_pointer_cast< AriesDictEncodedColumn >( columnReference->GetReferredColumn() );
            // indiceBuff = dictColumn->GetIndices()->GetDataBuffer( columnReference->GetIndices(), false );
            dictColumn->GetIndices()->MaterilizeSelfByIndices( columnReference->GetIndices() );

            m_dictEncodedColumns.insert( { columnId, dictColumn } );
            m_columnReferences.erase( columnId );
            CleanupIndices();
        }
        else
        {
            dictColumn = GetDictEncodedColumn( columnId );
        }

        indiceBuff = dictColumn->GetIndices()->GetDataBuffer();
        return indiceBuff;
    }

    AriesDictEncodedColumnSPtr AriesTableBlock::GetDictEncodedColumn( int32_t columnId ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        const auto itCol = m_dictEncodedColumns.find( columnId );
        ARIES_ASSERT( itCol != m_dictEncodedColumns.end(), "column should be dict encoded" );
        return itCol->second;
    }

    //该函数在进行数据输出时调用，用以更新相关数据，内部操作流程如下：
    //1.遍历和更新m_indices。让所有未物化的列下标得到更新
    //2.将输入的indices添加到m_indices中
    //3.遍历m_columns，将除输入的columnId以外的物化列和输入的indices一起打包，生成新的m_columnReferences
    //4.清空m_columns
    //5.如果columnId != -1 且column != nullptr，则将输入的columnId，column插入到m_columns中，成为唯一的物化列
    void AriesTableBlock::UpdateIndices( const AriesIndicesArraySPtr& indexArray, bool bHasNull, int32_t columnId, AriesColumnSPtr column )
    {
        ARIES_ASSERT( indexArray->GetItemCount() > 0, "indices's size should > 0 !!!" );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        if( m_rowCount > 0 )
            m_rowCount = indexArray->GetItemCount();
        if( m_indices.empty() )
        {
            ARIES_ASSERT( m_columnReferences.empty(), "m_columnReferences should be empty" );
            auto indices = std::make_shared< AriesIndices >();
            indices->AddIndices( indexArray );
            m_indices.push_back( indices );
            for( const auto it : m_columns )
            {
                AriesColumnReferenceSPtr colRef = std::make_shared< AriesColumnReference >( it.second );
                colRef->SetIndices( indices );
                m_columnReferences.insert(
                { it.first, colRef } );
            }
            for( const auto it : m_dictEncodedColumns )
            {
                AriesColumnReferenceSPtr colRef = std::make_shared< AriesColumnReference >( it.second );
                colRef->SetIndices( indices );
                m_columnReferences.insert(
                { it.first, colRef } );
            }
        }
        else
        {
            //1.遍历和更新m_indices。让所有未物化的列下标得到更新
            map< int32_t, int32_t > colRefIndiceMapping = GetColRefIndicePos();

            //m_indices = ::ShuffleIndices( m_indices, indexArray );
            ShuffleIndices( indexArray );
            auto indices = std::make_shared< AriesIndices >();
            indices->AddIndices( indexArray );
            //更新非物化的列的indices
            for( const auto & it : colRefIndiceMapping )
                m_columnReferences[it.first]->SetIndices( m_indices[it.second] );

            //2.遍历m_columns，将除输入的columnId以外的物化列和输入的indices一起打包，生成新的m_columnReferences
            bool needAddNewIndices = false;
            for( const auto it : m_columns )
            {
                if( it.first != columnId )
                {
                    needAddNewIndices = true;
                    AriesColumnReferenceSPtr colRef = std::make_shared< AriesColumnReference >( it.second );
                    colRef->SetIndices( indices );
                    m_columnReferences.insert(
                    { it.first, colRef } );
                }
            }

            for( const auto it : m_dictEncodedColumns )
            {
                if( it.first != columnId )
                {
                    needAddNewIndices = true;
                    AriesColumnReferenceSPtr colRef = std::make_shared< AriesColumnReference >( it.second );
                    colRef->SetIndices( indices );
                    m_columnReferences.insert(
                    { it.first, colRef } );
                }
            }

            //3.将输入的indices添加到m_indices中
            if( needAddNewIndices )
                m_indices.push_back( indices );
        }

        //4.清空m_columns
        m_columns.clear();
        m_dictEncodedColumns.clear();
        //5.如果columnId != -1 且column != nullptr，则将输入的columnId，column插入到m_columns中，成为唯一的物化列
        if( columnId != -1 && column )
            m_columns[columnId] = column;

        if( bHasNull )
        {
            for( auto indice : m_indices )
                indice->SetHasNull( true );
        }
#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        m_stats.m_updateIndiceTime += timeCost;
        LOG( INFO )<< "-----------------------AriesTableBlock::UpdateIndices time: " << timeCost;
#endif
    }

    AriesTableBlockUPtr AriesTableBlock::MakeTableByColumns( const vector< int32_t >& columnIds, bool bReturnCopy )
    {
        for( auto id : columnIds )
            ARIES_ASSERT( ColumnExists( id ), "column doesn't exist, columnId=" + std::to_string( id ) );
        AriesTableBlockUPtr result = std::make_unique< AriesTableBlock >();
        if( columnIds.empty() )
        {
            result->SetRowCount( GetRowCount() );
            return result;
        }
        int index = 1;
        int newPartitionedColumnId = -1;
        for( auto id : columnIds )
        {
            if( id == m_partitionedColumnID )
                newPartitionedColumnId = index;
            const auto itCol = m_columns.find( id );
            if( itCol != m_columns.end() )
            {
                if( bReturnCopy )
                {
                    auto newColumn = itCol->second->Clone();
                    result->m_columns[itCol->first] = std::dynamic_pointer_cast< AriesColumn >( newColumn );
                }
                else
                    result->m_columns[itCol->first] = itCol->second;
            }
            else
            {
                auto itDictColum = m_dictEncodedColumns.find( id );
                if( m_dictEncodedColumns.end() != itDictColum )
                {
                    if( bReturnCopy )
                    {
                        auto newDictColumn = itDictColum->second->Clone();
                        result->m_dictEncodedColumns[itDictColum->first] = std::dynamic_pointer_cast< AriesDictEncodedColumn >( newDictColumn );
                    }
                    else
                        result->m_dictEncodedColumns[itDictColum->first] = itDictColum->second;
                }
                else
                {
                    const auto itColRef = m_columnReferences.find( id );
                    ARIES_ASSERT( itColRef != m_columnReferences.end(), "id doesn't exists" );
                    result->m_columnReferences[itColRef->first] = itColRef->second->CloneWithEmptyContent();
                }
            }
            ++index;
        }

        map< int32_t, int32_t > colRefIndiceMapping = GetColRefIndicePos( columnIds );
        map< int32_t, int32_t > indicesToClone;
        index = 0;
        for( const auto & it : colRefIndiceMapping )
        {
            auto ind = indicesToClone.insert(
            { it.second, index } );
            if( ind.second )
            {
                ++index;
                if( bReturnCopy )
                    result->m_indices.push_back( m_indices[it.second]->Clone() );
                else
                    result->m_indices.push_back( m_indices[it.second] );
            }
            result->m_columnReferences[it.first]->SetIndices( result->m_indices[ind.first->second] );
        }

        if( m_columnReferences.empty() && m_indices.empty() )
        {
            result->SetPartitionedColumnID( newPartitionedColumnId );
            if( newPartitionedColumnId != -1 )
                result->SetPartitionInfo( m_blockPartitionID );
        }
            
        return result;
    }

    AriesTableBlockUPtr AriesTableBlock::Clone( bool bDeepCopy )
    {
        return MakeTableByColumns( GetAllColumnsId(), bDeepCopy );
    }

    std::vector< int32_t > AriesTableBlock::GetAllColumnsId() const
    {
        vector< int32_t > columnsId;
        for( auto it = m_columns.begin(); it != m_columns.end(); ++it )
        {
            columnsId.push_back( it->first );
        }
        for( auto it = m_dictEncodedColumns.begin(); it != m_dictEncodedColumns.end(); ++it )
        {
            columnsId.push_back( it->first );
        }
        for( auto it = m_columnReferences.begin(); it != m_columnReferences.end(); ++it )
        {
            columnsId.push_back( it->first );
        }
        return columnsId;
    }

    void AriesTableBlock::MergeTable( AriesTableBlockUPtr src )
    {
        if( 0 != GetColumnCount() && 0 != src->GetColumnCount() )
        {
            auto ok = MergeOk( src );
            ARIES_ASSERT( ok, "duplicate column id or different row count in two tables!!!" );
        }
        m_columns.insert( src->m_columns.begin(), src->m_columns.end() );
        m_dictEncodedColumns.insert( src->m_dictEncodedColumns.begin(), src->m_dictEncodedColumns.end() );
        m_columnReferences.insert( src->m_columnReferences.begin(), src->m_columnReferences.end() );
        for( const auto it : src->m_indices )
            m_indices.push_back( it );
    }

    vector< int64_t > AriesTableBlock::GetMaterilizedColumnDataBlockSizePsumArray() const
    {
        // assert( IsMaterilizedColumnHasSameBlockSize() );
        vector< int64_t > blockSizePrefixSumArray;
        for( const auto col : m_columns )
        {
            blockSizePrefixSumArray = col.second->GetBlockSizePsumArray();
            break;
        }
        if( blockSizePrefixSumArray.empty() )
        {
            for( const auto col : m_dictEncodedColumns )
            {
                blockSizePrefixSumArray = col.second->GetIndices()->GetBlockSizePsumArray();
                break;
            }
        }
        return blockSizePrefixSumArray;
    }

    bool AriesTableBlock::IsMaterilizedColumnHasSameBlockSize() const
    {
        vector< int64_t > blockSizePrefixSumArray;
        for( const auto col : m_columns )
        {
            blockSizePrefixSumArray = col.second->GetBlockSizePsumArray();
            break;
        }
        if( blockSizePrefixSumArray.empty() )
        {
            for( const auto col : m_dictEncodedColumns )
            {
                blockSizePrefixSumArray = col.second->GetIndices()->GetBlockSizePsumArray();
                break;
            }
        }
        if( !blockSizePrefixSumArray.empty() )
        {
            for( const auto col : m_columns )
            {
                if( blockSizePrefixSumArray != col.second->GetBlockSizePsumArray() )
                    return false;
            }
            for( const auto col : m_dictEncodedColumns )
            {
                if( blockSizePrefixSumArray != col.second->GetIndices()->GetBlockSizePsumArray() )
                    return false;
            }
        }

        return true;
    }

    bool AriesTableBlock::MergeOk( const AriesTableBlockUPtr& src ) const
    {
        for( const auto it : src->m_columns )
        {
            if( m_columns.find( it.first ) != m_columns.end() )
            {
                LOG( ERROR )<< "duplicated column " << it.first;
                return false;
            }
        }
        for( const auto it : src->m_dictEncodedColumns )
        {
            if( m_dictEncodedColumns.find( it.first ) != m_dictEncodedColumns.end() )
            {
                LOG( ERROR ) << "duplicated dict column " << it.first;
                return false;
            }
        }
        for( const auto it : src->m_columnReferences )
        {
            if( m_columnReferences.find( it.first ) != m_columnReferences.end() )
            {
                LOG( ERROR ) << "duplicated column reference " << it.first;
                return false;
            }
        }
        auto thisRowCount = GetRowCount();
        auto srcRowCount = src->GetRowCount();
        bool ok = ( thisRowCount == srcRowCount );
        if ( !ok )
            LOG( ERROR ) << "row count differs: " << thisRowCount << ", " << srcRowCount;
        return ok;
    }

    map< int32_t, int32_t > AriesTableBlock::GetColRefIndicePos() const
    {
        map< int32_t, int32_t > result;
        for( const auto it : m_columnReferences )
        {
            for( std::size_t i = 0; i < m_indices.size(); ++i )
            {
                if( it.second->GetIndices() == m_indices[i] )
                {
                    result.insert(
                    { it.first, i } );
                    break;
                }
            }
        }
        return result;
    }

    map< int32_t, int32_t > AriesTableBlock::GetColRefIndicePos( const vector< int32_t >& columnIds ) const
    {
        for( auto id : columnIds )
            ARIES_ASSERT( ColumnExists( id ), "column doesn't exist, columnId=" + std::to_string( id ) );
        map< int32_t, int32_t > result;
        for( const auto id : columnIds )
        {
            auto it = m_columnReferences.find( id );
            if( it != m_columnReferences.end() )
            {
                for( std::size_t i = 0; i < m_indices.size(); ++i )
                {
                    if( it->second->GetIndices() == m_indices[i] )
                    {
                        result.insert(
                        { it->first, i } );
                        break;
                    }
                }
            }
        }
        return result;
    }

    set< AriesIndicesSPtr > AriesTableBlock::FindReferredIndices( const map< int32_t, AriesColumnReferenceSPtr >& colRefs ) const
    {
        set< AriesIndicesSPtr > result;
        for( const auto it : colRefs )
            result.insert( it.second->GetIndices() );
        return result;
    }

    bool AriesTableBlock::VerifyContent() const
    {
        return true;
    }

    void AriesTableBlock::MaterilizeAll()
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        for( auto it : m_columnReferences )
        {
            auto column = make_shared< AriesColumn >();
            column->AddDataBuffer( it.second->GetDataBuffer() );
            m_columns.insert(
            { it.first, column } );
        }
        for( auto it : m_dictEncodedColumns )
        {
            auto column = make_shared< AriesColumn >();
            column->AddDataBuffer( it.second->GetDataBuffer() );
            m_columns.insert(
            { it.first, column } );
        }
        m_columnReferences.clear();
        m_dictEncodedColumns.clear();
        m_indices.clear();
#ifdef ARIES_PROFILE
        m_stats.m_materializeTime += t.end();
#endif
    }

    vector< int > AriesTableBlock::GetAllMaterilizedColumnIds() const
    {
        vector< int > result;
        for( auto it : m_columns )
            result.push_back( it.first );
        return result;
    }

    void AriesTableBlock::MaterilizeColumns( const vector< int > columnIds, bool bRunOnGpu )
    {
        for( int id : columnIds )
            GetColumnBuffer( id, false, bRunOnGpu );    //use GetColumnBuffer to make sure the column will be materilized
    }

    map< int32_t, AriesColumnSPtr > AriesTableBlock::GetAllColumns()
    {
        MaterilizeAll();
        return m_columns;
    }

    AriesIndicesArraySPtr AriesTableBlock::GetTheSharedIndiceForColumns( const vector< int > columnIds ) const
    {
        /*
        AriesIndicesArraySPtr result;
        if( IsColumnsShareSameIndices( columnIds ) )
            result = GetUnMaterilizedColumn( columnIds[0] )->GetIndices()->GetIndices();
        return result;
        */
       return nullptr;
    }

    void AriesTableBlock::ReplaceTheOnlyOneIndices( const AriesIndicesArraySPtr& indices, bool bHasNull )
    {
        assert( IsAllColumnsShareSameIndices() );
        m_indices[0]->Clear();
        m_indices[0]->AddIndices( indices );
        if( bHasNull )
            m_indices[0]->SetHasNull( true );
    }

    bool AriesTableBlock::IsColumnsShareSameIndices( const vector< int > columnIds ) const
    {
        bool result = !columnIds.empty();
        if( result )
        {
            AriesIndicesSPtr indices;
            for( auto id : columnIds )
            {
                if( !IsColumnUnMaterilized( id ) )
                {
                    result = false;
                    break;
                }
                AriesColumnReferenceSPtr colRef = GetUnMaterilizedColumn( id );
                if( indices )
                {
                    if( indices != colRef->GetIndices() )
                    {
                        result = false;
                        break;
                    }
                }
                else
                    indices = colRef->GetIndices();
            }
        }
        return result;
    }

    bool AriesTableBlock::IsAllColumnsShareSameIndices() const
    {
        bool result = ( m_columns.empty() && m_dictEncodedColumns.empty() && m_indices.size() == 1 );
        assert( result ? !m_columnReferences.empty() : 1 );
        return result;
    }

    // void AriesTableBlock::TryShrinkData( int marginSize, int ratio )
    // {
    //     if( !m_columnReferences.empty() )
    //     {
    //         vector< int32_t > toRemove;
    //         for( const auto colRef : m_columnReferences )
    //         {
    //             auto rowInfo = colRef.second->GetRowInfo();
    //             if( rowInfo.TotalRowNum > marginSize && rowInfo.TotalRowNum > rowInfo.MyRowNum * ratio )
    //             {
    //                 auto column = make_shared< AriesColumn >();
    //                 column->AddDataBuffer( colRef.second->GetDataBuffer() );
    //                 m_columns.insert(
    //                 { colRef.first, column } );
    //                 toRemove.push_back( colRef.first );
    //             }
    //         }

    //         for( auto id : toRemove )
    //             m_columnReferences.erase( id );
    //         if( !toRemove.empty() )
    //             CheckIfNeedMaterilize();
    //     }
    // }

    AriesTableBlockUPtr AriesTableBlock::CloneWithNoContent() const
    {
        AriesTableBlockUPtr result = make_unique< AriesTableBlock >();
        for( const auto col : m_columns )
            result->AddColumn( col.first, dynamic_pointer_cast< AriesColumn >( col.second->CloneWithNoContent() ) );
        for( const auto col : m_dictEncodedColumns )
            result->AddColumn( col.first, dynamic_pointer_cast< AriesDictEncodedColumn >( col.second->CloneWithNoContent() ) );
        for( const auto colRef : m_columnReferences )
        {
            AriesColumnReferenceSPtr newColRef = colRef.second->CloneWithEmptyContent();
            auto indices = std::make_shared< AriesIndices >();
            newColRef->SetIndices( indices );
            result->AddColumn( colRef.first, newColRef );
        }
        return result;
    }

    AriesTableBlockUPtr AriesTableBlock::CreateTableWithNoRows( const vector< AriesColumnType >& types )
    {
        AriesTableBlockUPtr result = make_unique< AriesTableBlock >();
        int32_t id = 0;
        for( const auto type : types )
        {
            auto column = make_shared< AriesColumn >();
            column->AddDataBuffer( make_shared< AriesDataBuffer >( type ) );
            result->AddColumn( ++id, column );
        }
        return result;
    }

    AriesColumnType AriesTableBlock::GetColumnType( int32_t columnId ) const
    {
        ARIES_ASSERT( ColumnExists( columnId ), "column doesn't exist, columnId=" + std::to_string( columnId ) );
        auto it = m_columns.find( columnId );
        if( it != m_columns.end() )
            return it->second->GetColumnType();
        else
        {
            auto itDictCol = m_dictEncodedColumns.find( columnId );
            if( itDictCol != m_dictEncodedColumns.end() )
                return itDictCol->second->GetColumnType();
            else
                return m_columnReferences.find( columnId )->second->GetReferredColumn()->GetColumnType();
        }
    }

    vector< AriesColumnType > AriesTableBlock::GetColumnTypes( const vector< int32_t >& columnIds ) const
    {
        for( auto id : columnIds )
            ARIES_ASSERT( ColumnExists( id ), "column doesn't exist, columnId=" + std::to_string( id ) );
        vector< AriesColumnType > columnTypes;
        for( auto id : columnIds )
        {
            auto it = m_columns.find( id );
            if( it != m_columns.end() )
            {
                columnTypes.push_back( it->second->GetColumnType() );
            }
            else
            {
                auto itDictCol = m_dictEncodedColumns.find( id );
                if( itDictCol != m_dictEncodedColumns.end() )
                    columnTypes.push_back( itDictCol->second->GetColumnType() );
                else
                {
                    auto it2 = m_columnReferences.find( id );
                    if( it2 != m_columnReferences.end() )
                        columnTypes.push_back( it2->second->GetReferredColumn()->GetColumnType() );
                    else
                        ARIES_ASSERT( 0, "Invalid column id:" + std::to_string( id ) );
                }
            }
        }
        return columnTypes;
    }

    void AriesTableBlock::CheckIfNeedMaterilize()
    {
        vector< AriesIndicesSPtr > indicesToKeep;
        for( const auto ind : m_indices )
        {
            long refCount = ind.use_count();
            if( refCount > 2 )
                indicesToKeep.push_back( ind );
            else if( refCount == 2 )
            {
                int32_t toMarterilize = -1;
                for( const auto colRef : m_columnReferences )
                {
                    auto colEncodeType = colRef.second->GetReferredColumn()->GetEncodeType();
                    if( colRef.second->GetIndices() == ind && colRef.second->GetReferredColumn()->GetColumnType().GetDataTypeSize() <= sizeof(index_t)
                            && EncodeType::NONE == colEncodeType )
                    {
                        toMarterilize = colRef.first;
                        auto column = make_shared< AriesColumn >();
                        column->AddDataBuffer( colRef.second->GetDataBuffer() );
                        m_columns.insert(
                        { toMarterilize, column } );
                    }
                }
                if( toMarterilize != -1 )
                    m_columnReferences.erase( toMarterilize );
                else
                    indicesToKeep.push_back( ind );
            }
        }
        m_indices = indicesToKeep;
    }

    ColumnRefRowInfo AriesTableBlock::GetRowInfo() const
    {
        const auto it = m_columnReferences.cbegin();
        if( it != m_columnReferences.cend() )
            return it->second->GetRowInfo();
        else
        {
            int64_t rowCount = GetRowCount();
            return
            {   rowCount, rowCount};
        }
    }

    bool AriesTableBlock::ColumnExists( int32_t columnId ) const
    {
        bool bResult = false;
        const auto it = m_columns.find( columnId );
        if( it != m_columns.cend() )
            bResult = true;
        else
        {
            const auto it2 = m_columnReferences.find( columnId );
            if( it2 != m_columnReferences.cend() )
                bResult = true;
            else
            {
                const auto it3 = m_dictEncodedColumns.find( columnId );
                if( it3 != m_dictEncodedColumns.cend() )
                    bResult = true;
            }
        }
        return bResult;
    }

    void AriesTableBlock::ResetAllStats()
    {
        ResetTimeCostStats();
    }
    void AriesTableBlock::ResetTimeCostStats()
    {
        m_stats.m_materializeTime = 0;
        m_stats.m_updateIndiceTime = 0;
        m_stats.m_getSubTableTime = 0;
    }
    
    void AriesTableBlock::MaterilizeDictEncodedColumn( int columnId )
    {
        auto it = m_columnReferences.find( columnId );
        if( it != m_columnReferences.end() )
        {
            AriesBaseColumnSPtr column = it->second->GetReferredColumn();
            EncodeType encodeType = column->GetEncodeType();
            assert( encodeType == EncodeType::DICT );
            AriesIndicesSPtr Indices = it->second->GetIndices();
            column->MaterilizeSelfByIndices( Indices );

            m_dictEncodedColumns.insert( { it->first, dynamic_pointer_cast< AriesDictEncodedColumn >( column ) } );
            m_columnReferences.erase( it );
            CleanupIndices();
        }
        auto itDict = m_dictEncodedColumns.find( columnId );
        assert( itDict != m_dictEncodedColumns.end() );

        auto column = make_shared< AriesColumn >();
        column->AddDataBuffer( itDict->second->GetDataBuffer() );
        m_columns.insert( { itDict->first, column } );

        m_dictEncodedColumns.erase( itDict );
    }

    void AriesTableBlock::MaterilizeAllDataBlocks()
    {
        for( auto it : m_columnReferences )
        {
            AriesBaseColumnSPtr column = it.second->GetReferredColumn();
            EncodeType encodeType = column->GetEncodeType();
            AriesIndicesSPtr Indices = it.second->GetIndices();
            column->MaterilizeSelfByIndices( Indices );
            switch( encodeType )
            {
                case EncodeType::DICT:
                {
                    m_dictEncodedColumns.insert(
                    { it.first, dynamic_pointer_cast< AriesDictEncodedColumn >( column ) } );
                    break;
                }
                case EncodeType::NONE:
                {
                    m_columns.insert(
                    { it.first, dynamic_pointer_cast< AriesColumn >( column ) } );
                    break;
                }
                default:
                    break;
            }
        }
        m_columnReferences.clear();
        m_indices.clear();
    }

    int32_t AriesTableBlock::GetPartitionID() const
    {
        return m_partitionID;
    }

    int32_t AriesTableBlock::GetPartitionedColumnID() const
    {
        return m_partitionedColumnID;
    }

    void AriesTableBlock::SetPartitionedColumnID( int32_t partitionedColumnID )
    {
        m_partitionedColumnID = partitionedColumnID;
    }

    void AriesTableBlock::ClearPartitionInfo()
    {
        m_partitionID = -1;
        m_partitionedColumnID = -1;
    }

    void AriesTableBlock::SetPartitionID( int32_t partitionID )
    {
        m_partitionID = partitionID;
    }

    const map< int32_t, int32_t >& AriesTableBlock::GetPartitionInfo() const
    {
        return m_blockPartitionID;
    }

    void AriesTableBlock::SetPartitionInfo( const map< int32_t, int32_t >& info )
    {
        m_blockPartitionID = info;
    }


END_ARIES_ENGINE_NAMESPACE
