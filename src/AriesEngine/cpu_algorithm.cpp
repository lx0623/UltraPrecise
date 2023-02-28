#include <atomic>
#include "cpu_algorithm.h"

AriesIndicesArraySPtr ShuffleIndices(const AriesIndicesArraySPtr &oldIndices, const AriesIndicesArraySPtr &indices)
{
    indices->PrefetchToCpu();
    oldIndices->PrefetchToCpu();
    int64_t count = indices->GetItemCount();
    AriesIndicesArraySPtr result = std::make_shared<AriesIndicesArray>(count, false, false);
    result->PrefetchToCpu();
    int64_t blockCount = GetThreadNum(count);
    int64_t blockSize = count / blockCount;
    int64_t extraSize = count % blockCount;
    vector<future<void>> workThreads;
    auto pNew = indices->GetData();

    auto pOutput = result->GetData();
    for (int i = 0; i < blockCount; ++i)
    {
        auto pOutputBlock = pOutput + blockSize * i;
        auto pNewBlock = pNew + blockSize * i;
        if (i == blockCount - 1)
            workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndices->GetData(), pNewBlock, pOutputBlock, blockSize + extraSize); }));
        else
            workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndices->GetData(), pNewBlock, pOutputBlock, blockSize); }));
    }

    for (auto &t : workThreads)
        t.wait();
    return result;
}

AriesIndicesArraySPtr ShuffleIndices(const AriesIndicesArraySPtr &oldIndices, const vector<AriesIndicesArraySPtr> &indices)
{
    int64_t totalCount = 0;
    for (const auto &ind : indices)
    {
        totalCount += ind->GetItemCount();
        ind->PrefetchToCpu();
    }
    oldIndices->PrefetchToCpu();

    AriesIndicesArraySPtr result = std::make_shared<AriesIndicesArray>(totalCount, false, false);
    result->PrefetchToCpu();
    int64_t offset = 0;

    for (const auto &ind : indices)
    {
        int64_t count = ind->GetItemCount();
        int64_t blockCount = GetThreadNum(count);
        int64_t blockSize = count / blockCount;
        int64_t extraSize = count % blockCount;
        vector<future<void>> workThreads;
        auto pNew = ind->GetData();

        auto pOutput = result->GetData() + offset;
        offset += count;

        for (int i = 0; i < blockCount; ++i)
        {
            auto pOutputBlock = pOutput + blockSize * i;
            auto pNewBlock = pNew + blockSize * i;
            if (i == blockCount - 1)
                workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndices->GetData(), pNewBlock, pOutputBlock, blockSize + extraSize); }));
            else
                workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndices->GetData(), pNewBlock, pOutputBlock, blockSize); }));
        }

        for (auto &t : workThreads)
            t.wait();
    }
    return result;
}

AriesIndicesArraySPtr ShuffleIndices(const AriesIndicesSPtr &oldIndices, const AriesIndicesArraySPtr &indices)
{
    indices->PrefetchToCpu();
    oldIndices->PrefetchDataToCpu();

    vector<index_t *> oldIndicesArray;
    for (const auto &ind : oldIndices->GetIndicesArray())
    {
        oldIndicesArray.push_back(ind->GetData());
    }

    int64_t count = indices->GetItemCount();
    AriesIndicesArraySPtr result = std::make_shared<AriesIndicesArray>(count, false, false);
    result->PrefetchToCpu();
    int64_t blockCount = GetThreadNum(count);
    int64_t blockSize = count / blockCount;
    int64_t extraSize = count % blockCount;
    vector<future<void>> workThreads;
    auto pNew = indices->GetData();
    auto pOutput = result->GetData();

    for (int i = 0; i < blockCount; ++i)
    {
        auto pOutputBlock = pOutput + blockSize * i;
        auto pNewBlock = pNew + blockSize * i;
        if (i == blockCount - 1)
            workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndicesArray, oldIndices->GetBlockSizePsumArray(), pNewBlock, pOutputBlock, blockSize + extraSize); }));
        else
            workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndicesArray, oldIndices->GetBlockSizePsumArray(), pNewBlock, pOutputBlock, blockSize); }));
    }

    for (auto &t : workThreads)
        t.wait();

    return result;
}

vector<AriesIndicesArraySPtr> ShuffleIndices(const vector<AriesIndicesArraySPtr> &oldIndices, const AriesIndicesArraySPtr &indices)
{
    indices->PrefetchToCpu();
    for (const auto &ind : oldIndices)
        ind->PrefetchToCpu();
    vector<AriesIndicesArraySPtr> result;
    int64_t size = oldIndices.size();
    int64_t count = indices->GetItemCount();
    for (int64_t i = 0; i < size; ++i)
    {
        result.push_back(std::make_shared<AriesIndicesArray>(count, false, false));
        result.back()->PrefetchToCpu();
    }

    int64_t blockCount = GetThreadNum(count);
    int64_t blockSize = count / blockCount;
    int64_t extraSize = count % blockCount;
    vector<future<void>> workThreads;
    auto pNew = indices->GetData();
    int index = 0;
    for (auto &it : result)
    {
        auto pOutput = it->GetData();
        for (int i = 0; i < blockCount; ++i)
        {
            auto pOutputBlock = pOutput + blockSize * i;
            auto pNewBlock = pNew + blockSize * i;
            if (i == blockCount - 1)
                workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndices[index]->GetData(), pNewBlock, pOutputBlock, blockSize + extraSize); }));
            else
                workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(oldIndices[index]->GetData(), pNewBlock, pOutputBlock, blockSize); }));
        }
        ++index;
    }
    for (auto &t : workThreads)
        t.wait();
    return result;
}

vector<AriesIndicesSPtr> ShuffleIndices(const vector<AriesIndicesSPtr> &oldIndices, const AriesIndicesArraySPtr &indices)
{
    indices->PrefetchToCpu();
    for (const auto &ind : oldIndices)
        ind->PrefetchDataToCpu();
    vector<AriesIndicesSPtr> result;
    vector<AriesIndicesArraySPtr> newIndices;
    int64_t size = oldIndices.size();
    int64_t count = indices->GetItemCount();
    for (int64_t i = 0; i < size; ++i)
    {
        newIndices.push_back(std::make_shared<AriesIndicesArray>(count, false, false));
        newIndices.back()->PrefetchToCpu();
        auto ind = std::make_shared<AriesIndices>();
        ind->AddIndices(newIndices.back());
        ind->SetHasNull(oldIndices[i]->HasNull());
        result.push_back(ind);
    }

    vector<AriesIndicesArraySPtr> old;
    for (const auto it : oldIndices)
        old.push_back(it->GetIndices());

    int64_t blockCount = GetThreadNum(count);
    int64_t blockSize = count / blockCount;
    int64_t extraSize = count % blockCount;
    vector<future<void>> workThreads;
    auto pNew = indices->GetData();
    int index = 0;
    for (auto &it : newIndices)
    {
        auto pOutput = it->GetData();
        for (int i = 0; i < blockCount; ++i)
        {
            auto pOutputBlock = pOutput + blockSize * i;
            auto pNewBlock = pNew + blockSize * i;
            if (i == blockCount - 1)
                workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(old[index]->GetData(), pNewBlock, pOutputBlock, blockSize + extraSize); }));
            else
                workThreads.push_back(std::async(std::launch::async, [=] { ShuffleInSingleThread(old[index]->GetData(), pNewBlock, pOutputBlock, blockSize); }));
        }
        ++index;
    }
    for (auto &t : workThreads)
        t.wait();
    return result;
}

void AddOffsetToIndices(AriesIndicesArraySPtr indices, bool hasNull, int64_t offset)
{
    if (offset > 0)
    {
        indices->PrefetchToCpu();
        int64_t count = indices->GetItemCount();

        int64_t blockCount = GetThreadNum(count);
        int64_t blockSize = count / blockCount;
        int64_t extraSize = count % blockCount;
        vector<future<void>> workThreads;
        auto pIndice = indices->GetData();
        for (int i = 0; i < blockCount; ++i)
        {
            if (i == blockCount - 1)
                workThreads.push_back(std::async(std::launch::async, [=] { AddOffsetInSingleThread(pIndice + blockSize * i, blockSize + extraSize, hasNull, offset); }));
            else
                workThreads.push_back(std::async(std::launch::async, [=] { AddOffsetInSingleThread(pIndice + blockSize * i, blockSize, hasNull, offset); }));
        }

        for (auto &t : workThreads)
            t.wait();
    }
}

void GetDataBufferByCpuSingleThread(const index_t *pIndex, int64_t itemCount, const AriesDataBufferSPtr& dataBuffer, bool hasNull, AriesColumnType columnType,
                                    int8_t *output)
{
    int64_t srcDataTypeSizeInBytes = columnType.GetDataTypeSize();
    AriesColumnType outputType = columnType;
    outputType.HasNull = hasNull || columnType.HasNull;

    if (outputType.HasNull)
    {
        int64_t outputDataTypeSizeInBytes = outputType.GetDataTypeSize();
        int8_t *pOutput;
        int64_t offset = outputDataTypeSizeInBytes - srcDataTypeSizeInBytes;
        index_t index;
        for (int64_t i = 0; i < itemCount; ++i)
        {
            index = pIndex[i];
            if (index != -1)
            {
                pOutput = output + i * outputDataTypeSizeInBytes;
                *pOutput = 1;
                memcpy( pOutput + offset, dataBuffer->GetItemDataAt( index ), srcDataTypeSizeInBytes );
            }
            else
            {
                *(output + i * outputDataTypeSizeInBytes) = 0;
            }
        }
    }
    else
    {
        for (int64_t i = 0; i < itemCount; ++i)
            memcpy( output + i * srcDataTypeSizeInBytes, dataBuffer->GetItemDataAt( pIndex[i] ), srcDataTypeSizeInBytes );
    }
}

void GetDataBufferByCpuSingleThread(const index_t *pIndex, int64_t itemCount, const vector<AriesDataBufferSPtr> &dataBlocks,
                                    const vector<int64_t> &prefixSum, bool hasNull, AriesColumnType columnType, int8_t *output)
{
    int64_t srcDataTypeSizeInBytes = columnType.GetDataTypeSize();
    AriesColumnType outputType = columnType;
    outputType.HasNull = hasNull || columnType.HasNull;

    int blockIndex;
    int offsetInBlock;
    auto itBegin = prefixSum.cbegin();
    auto itEnd = prefixSum.cend();
    auto it = itEnd;
    auto index = *pIndex;
    if (outputType.HasNull)
    {
        int64_t outputDataTypeSizeInBytes = outputType.GetDataTypeSize();
        int8_t *pOutput;
        int64_t offset = outputDataTypeSizeInBytes - srcDataTypeSizeInBytes;
        for (int64_t i = 0; i < itemCount; ++i)
        {
            index = pIndex[i];
            if (index != -1)
            {
                it = std::upper_bound(itBegin, itEnd, index);
                blockIndex = (it - itBegin) - 1;
                offsetInBlock = index - prefixSum[blockIndex];
                pOutput = output + i * outputDataTypeSizeInBytes;
                *pOutput = 1;
                memcpy(pOutput + offset, dataBlocks[blockIndex]->GetItemDataAt( offsetInBlock ), srcDataTypeSizeInBytes);
            }
            else
            {
                *(output + i * outputDataTypeSizeInBytes) = 0;
            }
        }
    }
    else
    {
        int8_t *pOutput;
        int8_t *pInput;
        for (int64_t i = 0; i < itemCount; ++i)
        {
            index = pIndex[i];
            it = std::upper_bound(itBegin, itEnd, index);
            blockIndex = (it - itBegin) - 1;
            offsetInBlock = index - prefixSum[blockIndex];

            pOutput = output + i * srcDataTypeSizeInBytes;
            pInput = dataBlocks[blockIndex]->GetItemDataAt( offsetInBlock );
            memcpy(pOutput, pInput, srcDataTypeSizeInBytes);
        }
    }
}

AriesDataBufferSPtr GetDataBufferByIndices(const AriesIndicesArraySPtr &indices, const vector<AriesDataBufferSPtr> &dataBlocks,
                                           const vector<int64_t> &prefixSum, bool hasNull, AriesColumnType columnType)
{
    int64_t itemCount = indices->GetItemCount();

    AriesColumnType outputType = columnType;
    outputType.HasNull = hasNull || columnType.HasNull;
    AriesDataBufferSPtr result = std::make_shared<AriesDataBuffer>(outputType, itemCount);
    result->PrefetchToCpu();
    int64_t outputDataTypeSizeInBytes = outputType.GetDataTypeSize();

    const auto *pIndice = indices->GetData();
    int8_t *pOutput = result->GetData();

    int64_t blockCount = GetThreadNum(itemCount);
    int64_t blockSize = itemCount / blockCount;
    int64_t extraSize = itemCount % blockCount;
    vector<future<void>> workThreads;

    for (int i = 0; i < blockCount; ++i)
    {
        if (i == blockCount - 1)
            workThreads.push_back(
                std::async(std::launch::async,
                           [=] { GetDataBufferByCpuSingleThread(pIndice + blockSize * i, blockSize + extraSize, dataBlocks, prefixSum, hasNull, columnType, pOutput + blockSize * i * outputDataTypeSizeInBytes); }));
        else
            workThreads.push_back(
                std::async(std::launch::async,
                           [=] { GetDataBufferByCpuSingleThread(pIndice + blockSize * i, blockSize, dataBlocks, prefixSum, hasNull, columnType, pOutput + blockSize * i * outputDataTypeSizeInBytes); }));
    }

    for (auto &t : workThreads)
        t.wait();

    return result;
}

AriesDataBufferSPtr GetDataBufferByIndices(const AriesIndicesArraySPtr &indices, const AriesDataBufferSPtr &data, bool hasNull,
                                           AriesColumnType columnType)
{
    int64_t itemCount = indices->GetItemCount();

    AriesColumnType outputType = columnType;
    outputType.HasNull = hasNull || columnType.HasNull;
    AriesDataBufferSPtr result = std::make_shared<AriesDataBuffer>(outputType, itemCount);
    result->PrefetchToCpu();
    int64_t outputDataTypeSizeInBytes = outputType.GetDataTypeSize();

    const auto *pIndice = indices->GetData();
    int8_t *pOutput = result->GetData();

    int64_t blockCount = GetThreadNum(itemCount);
    int64_t blockSize = itemCount / blockCount;
    int64_t extraSize = itemCount % blockCount;
    vector<future<void>> workThreads;

    for (int i = 0; i < blockCount; ++i)
    {
        if (i == blockCount - 1)
            workThreads.push_back(
                std::async(std::launch::async,
                           [=] { GetDataBufferByCpuSingleThread(pIndice + blockSize * i, blockSize + extraSize, data, hasNull, columnType, pOutput + blockSize * i * outputDataTypeSizeInBytes); }));
        else
            workThreads.push_back(
                std::async(std::launch::async,
                           [=] { GetDataBufferByCpuSingleThread(pIndice + blockSize * i, blockSize, data, hasNull, columnType, pOutput + blockSize * i * outputDataTypeSizeInBytes); }));
    }

    for (auto &t : workThreads)
        t.wait();

    return result;
}

AriesIndicesArraySPtr CreateNullIndex(int64_t count)
{
    AriesIndicesArraySPtr result = std::make_shared<AriesIndicesArray>(count);
    cudaMemset(result->GetData(), 0xFF, result->GetTotalBytes());
    return result;
}

AriesIndicesArraySPtr CreateSequenceIndex(int64_t count, index_t startVal)
{
    AriesIndicesArraySPtr result = std::make_shared<AriesIndicesArray>(count, false, false);
    result->PrefetchToCpu();
    index_t *pData = result->GetData();
    for (int64_t i = 0; i < count; ++i)
        *pData++ = startVal++;
    return result;
}

template <typename type_t>
void ShuffleInSingleThread(const vector<type_t *> &inputs,
                           const vector<int64_t> &inputsRowCountPrefixSum,
                           const type_t *new_indices,
                           type_t *output,
                           int64_t output_count)
{
    auto itBegin = inputsRowCountPrefixSum.cbegin();
    auto itEnd = inputsRowCountPrefixSum.cend();
    auto it = itEnd;
    for (int64_t i = 0; i < output_count; ++i)
    {
        type_t newIdx = new_indices[i];
        if (NULL_INDEX == newIdx)
        {
            output[i] = NULL_INDEX;
        }
        else
        {
            it = std::upper_bound(itBegin, itEnd, newIdx);
            int indicesBlockIndex = it - itBegin - 1;
            int offset = newIdx - inputsRowCountPrefixSum[indicesBlockIndex];
            output[i] = inputs[indicesBlockIndex][offset];
        }
    }
}

vector< PartitionedIndices > PartitionColumnData(
    const vector< AriesDataBufferSPtr >& buffers,
    size_t bucketCount,
    uint32_t seed,
    bool *hasNullValues,
    shared_ptr< vector< index_t > > indices,
    int round )
{
    assert( !buffers.empty() );

    vector< PartitionedIndices > result;
    for( size_t i = 0; i < bucketCount; ++i )
        result.push_back( PartitionedIndices{ std::make_shared< vector< index_t > >(), round } );

    size_t totalItemCount = indices ? indices->size() : buffers[0]->GetItemCount();
    assert( totalItemCount > 0 );

    size_t threadNum = GetThreadNum( totalItemCount );
    size_t perThreadItemCount = totalItemCount / threadNum;
    size_t offset = 0;

    struct DataWrapper
    {
        int8_t* data;
        size_t perItemSize;
        bool hasNull;
    };

    auto BKDRHash = []( const int8_t *p, int size, uint32_t seed )
    {
        uint32_t hash = seed;
    
        while( size-- )
            hash = ( hash * 131 ) + *p++;
    
        return hash;
    };

    vector< DataWrapper > dataWrapper;
    for( const auto& buf : buffers )
        dataWrapper.emplace_back( DataWrapper{ buf->GetData( 0 ), buf->GetItemSizeInBytes(), buf->GetDataType().HasNull } );

    atomic_bool bHasNull( false );
    vector< future< vector< PartitionedIndices > > > allThreads;
    for( size_t i = 0; i < threadNum; ++i )
    {
        size_t itemCount = perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
        allThreads.push_back( std::async( std::launch::async, [&]( size_t pos, size_t count )
        {   
            vector< PartitionedIndices > output;
            for( size_t n = 0; n < bucketCount; ++n )
                output.push_back( PartitionedIndices{ std::make_shared< vector< index_t > >(), round } );

            for( size_t index = 0; index < count; ++index )
            {
                uint32_t hashVal = 0;
                size_t itemPos = indices ? ( *indices )[ pos + index ] : pos + index;
                for( const auto& wrapper : dataWrapper )
                {
                    auto itemStart = wrapper.data + ( pos + index ) * wrapper.perItemSize;
                    hashVal ^= BKDRHash( itemStart + wrapper.hasNull, wrapper.perItemSize - wrapper.hasNull, seed );
                    if ( hasNullValues && wrapper.hasNull && !( *itemStart ) )
                    {
                        bHasNull = true;
                    }
                }
                output[ hashVal % bucketCount ].Indices->push_back( itemPos );
            }
            return output;
        }, offset, itemCount ) );
        offset += perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
    }

    vector< vector< PartitionedIndices > > allResult;
    for( auto& t : allThreads )
        allResult.push_back( t.get() );

    vector< size_t > totalSizes;
    totalSizes.resize( bucketCount, 0 );
    for( const auto& r : allResult )
    {
        for( size_t i = 0; i < r.size(); ++i )
            totalSizes[ i ] += r[ i ].Indices->size();
    }

    for( size_t i = 0; i < bucketCount; ++i )
    {
        result[ i ].Indices->reserve( totalSizes[ i ] );
        for( const auto& r : allResult )
            result[ i ].Indices->insert( result[ i ].Indices->end(), r[ i ].Indices->begin(), r[ i ].Indices->end() );
    }

    if ( hasNullValues )
        *hasNullValues = bHasNull; 

    return result;    
}

AriesIndicesArraySPtr ConvertToAriesIndices( const PartitionedIndices& partIndices )
{
    AriesIndicesArraySPtr indices = std::make_shared< AriesIndicesArray >( partIndices.Indices->size(), false, true );
    indices->CopyFromHostMem( partIndices.Indices->data(), partIndices.Indices->size() * sizeof( index_t ) );
    return indices;
}

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const vector< AriesDataBufferSPtr > &hashColumnBuffers,
                              vector<  vector< AriesDataBufferSPtr > >& partBuffers,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues )
{
    auto partitionResult = PartitionColumnDataEx( hashColumnBuffers, partitionCount, seed, hasNullValues );
    AriesTableBlock *notEmptyTable = nullptr;
    for( const auto& part : partitionResult.AllIndices )
    {
        AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
        if ( indices->GetItemCount() == 0 )
        {
            outputSubTables.emplace_back( nullptr );
            continue;
        }
        auto tableClone = table->Clone( false );
        tableClone->UpdateIndices( indices );
        if ( !notEmptyTable )
        {
            notEmptyTable = tableClone.get();
        }
        outputSubTables.push_back( std::move( tableClone ) );
    }

    partBuffers.resize( partitionResult.AllBuffers.size() );
    for ( size_t i = 0; i < partitionResult.AllBuffers.size(); i++ )
    {
        for ( size_t j = 0; j < partitionResult.AllBuffers[ i ].size(); j++ )
        {
            partBuffers[ i ].emplace_back( partitionResult.AllBuffers[ i ][ j ]->ToAriesDataBuffer( false ) );
        }
    }

    vector< AriesTableBlockUPtr > newOutputSubTables;
    for ( auto& table : outputSubTables )
    {
        if ( table )
        {
            newOutputSubTables.push_back( std::move( table ) );
        }
        else
        {
            newOutputSubTables.push_back( std::move( notEmptyTable->CloneWithNoContent() ) );
        }
    }
    std::swap( outputSubTables, newOutputSubTables );
}

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const vector< AriesDataBufferSPtr > &hashColumnBuffers,
                              const vector< int >& hashColumnIds,
                              const vector< int32_t >& columnsIdToMaterialize,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues )
{
    auto partitionResult = PartitionColumnDataEx( hashColumnBuffers, partitionCount, seed, hasNullValues );
    AriesTableBlock *notEmptyTable = nullptr;

    for ( size_t i = 0; i < partitionResult.AllIndices.size(); i++ )
    {
        const auto& part = partitionResult.AllIndices[ i ];
        AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
        if ( indices->GetItemCount() == 0 )
        {
            outputSubTables.emplace_back( nullptr );
            continue;
        }

        auto tableClone = table->MakeTableByColumns( columnsIdToMaterialize );
        tableClone->UpdateIndices( indices );

        for ( size_t j = 0; j < hashColumnIds.size(); j++ )
        {
            auto column = std::make_shared< AriesColumn >();
            column->AddDataBuffer( partitionResult.AllBuffers[ j ][ i ]->ToAriesDataBuffer( false ) );
            tableClone->AddColumn( hashColumnIds[ j ], column );
        }

        if ( !notEmptyTable )
        {
            notEmptyTable = tableClone.get();
        }
        outputSubTables.push_back( std::move( tableClone ) );
    }

    vector< AriesTableBlockUPtr > newOutputSubTables;
    for ( auto& table : outputSubTables )
    {
        if ( table )
        {
            newOutputSubTables.push_back( std::move( table ) );
        }
        else
        {
            newOutputSubTables.push_back( std::move( notEmptyTable->CloneWithNoContent() ) );
        }
    }
    std::swap( outputSubTables, newOutputSubTables );
}

// void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
//                               const vector< int >& hashColumnIds,
//                               const size_t partitionCount,
//                               uint32_t seed,
//                               vector< AriesTableBlockUPtr >& outputSubTables,
//                               bool *hasNullValues )
// {
//     if ( 1 == partitionCount )
//     {
//         outputSubTables.emplace_back( table->Clone( false ) );
//     }
//     else
//     {
//         vector< AriesDataBufferSPtr > buffers;

//         std::vector< int32_t > columnsToMaterialize;
//         for( int id : hashColumnIds)
//         {
//             buffers.push_back( table->GetColumnBuffer( id ) );
//         }

//         for ( const auto id : table->GetAllColumnsId() )
//         {
//             if ( std::find( hashColumnIds.cbegin(), hashColumnIds.cend(), id ) == hashColumnIds.cend() )
//             {
//                 columnsToMaterialize.emplace_back( id );
//             }
//         }

//         GraceHashPartitionTable( table, buffers, hashColumnIds, columnsToMaterialize, partitionCount, seed, outputSubTables, hasNullValues );
//     }
// }

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const vector< int >& hashColumnIds,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues )
{
    if ( 1 == partitionCount )
    {
        outputSubTables.push_back( table->Clone( false ) );
    }
    else
    {
        vector< AriesDataBufferSPtr > buffers;
        std::vector< int32_t > columnsToMaterialize;
        for( int id : hashColumnIds)
        {
            buffers.push_back( table->GetColumnBuffer( id ) );
        }

        for ( const auto id : table->GetAllColumnsId() )
        {
            if ( std::find( hashColumnIds.cbegin(), hashColumnIds.cend(), id ) == hashColumnIds.cend() )
            {
                columnsToMaterialize.emplace_back( id );
            }
        }
        GraceHashPartitionTable( table, buffers, hashColumnIds, columnsToMaterialize, partitionCount, seed, outputSubTables, hasNullValues );
    }
}

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const AriesDataBufferSPtr &hashColumnBuffer,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues )
{
    if ( 1 == partitionCount )
    {
        outputSubTables.push_back( table->Clone( false ) );
    }
    else
    {
        GraceHashPartitionTable( table, { hashColumnBuffer }, partitionCount, seed, outputSubTables, hasNullValues );
    }
}

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const AriesDataBufferSPtr &hashColumnBuffer,
                              vector< vector< AriesDataBufferSPtr > >& partBuffers,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues )
{
    if ( 1 == partitionCount )
    {
        outputSubTables.push_back( table->Clone( false ) );
        partBuffers.resize( 1 );
        partBuffers[ 0 ].emplace_back( hashColumnBuffer );
    }
    else
    {
        vector< AriesDataBufferSPtr > buffers;
        buffers.push_back( hashColumnBuffer );
        GraceHashPartitionTable( table, buffers, partBuffers, partitionCount, seed, outputSubTables, hasNullValues );
    }
}

void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
                              const vector< int >& hashColumnIds,
                              vector< vector< AriesDataBufferSPtr > >& partBuffers,
                              const size_t partitionCount,
                              uint32_t seed,
                              vector< AriesTableBlockUPtr >& outputSubTables,
                              bool *hasNullValues )
{
    if ( 1 == partitionCount )
    {
        outputSubTables.push_back( table->Clone( false ) );
        partBuffers.resize( hashColumnIds.size() );
        for ( const auto id : hashColumnIds )
        {
            partBuffers[ id ].emplace_back( table->GetColumnBuffer( id ) );
        }
    }
    else
    {
        vector< AriesDataBufferSPtr > buffers;

        for( int id : hashColumnIds)
            buffers.push_back( table->GetColumnBuffer( id ) );
        GraceHashPartitionTable( table, buffers, partBuffers, partitionCount, seed, outputSubTables, hasNullValues );
    }
}

// void GraceHashPartitionTable( const AriesTableBlockUPtr& table,
//                               const AriesDataBufferSPtr &hashColumnBuffer,
//                               vector< vector< AriesDataBufferSPtr > >& partBuffers,
//                               const size_t partitionCount,
//                               uint32_t seed,
//                               vector< AriesTableBlockUPtr >& outputSubTables,
//                               bool *hasNullValues )
// {
//     if ( 1 == partitionCount )
//     {
//         outputSubTables.push_back( table->Clone( false ) );
//         partBuffers.resize( 1 );
//         partBuffers[ 0 ].emplace_back( hashColumnBuffer );
//     }
//     else
//     {
//         vector< AriesDataBufferSPtr > buffers;
//         buffers.push_back( hashColumnBuffer );
//         GraceHashPartitionTable( table, buffers, partBuffers, partitionCount, seed, outputSubTables, hasNullValues );
//     }
// }

PartitionResult PartitionColumnDataEx(
    const vector< AriesDataBufferSPtr >& buffers,
    size_t bucketCount,
    uint32_t seed,
    bool *hasNullValues )
{
    assert( !buffers.empty() );
    size_t totalItemCount = buffers[0]->GetItemCount();
    assert( totalItemCount > 0 );
    size_t bufferCount = buffers.size();

    PartitionResult result;
    result.AllBuffers.resize( bufferCount );
    for( size_t i = 0; i < bucketCount; ++i )
    {
        result.AllIndices.push_back( PartitionedIndices{ std::make_shared< vector< index_t > >() } );
        for( size_t j = 0; j < bufferCount; ++j )
        {
            result.AllBuffers[ j ].push_back( std::make_shared< AriesHostDataBuffer >( buffers[ j ]->GetDataType() ) );
        }
    }

    size_t threadNum = GetThreadNum( totalItemCount );
    size_t perThreadItemCount = totalItemCount / threadNum;
    size_t offset = 0;

    struct DataWrapper
    {
        int8_t* data;
        size_t perItemSize;
        bool hasNull;
    };
    
    static const size_t BUCKET_NUM = 16381;
    size_t perBucketSize = BUCKET_NUM / bucketCount + BUCKET_NUM % bucketCount;

    auto BKDRHash = []( const int8_t *p, int size, uint32_t seed )
    {
        uint32_t hash = seed;
    
        while( size-- )
            hash = ( hash * 131 ) + *p++;
    
        return hash;
    };

    vector< DataWrapper > dataWrapper;
    for( const auto& buf : buffers )
        dataWrapper.emplace_back( DataWrapper{ buf->GetData( 0 ), buf->GetItemSizeInBytes(), buf->GetDataType().HasNull } );

    atomic_bool bHasNull = false;

    vector< future< PartitionResult > > allThreads;
    
    for( size_t i = 0; i < threadNum; ++i )
    {
        size_t itemCount = perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
        allThreads.push_back( std::async( std::launch::async, [&]( size_t pos, size_t count )
        {   
            PartitionResult output;
            output.AllBuffers.resize( bufferCount );
            for( size_t n = 0; n < bucketCount; ++n )
            {
                output.AllIndices.push_back( PartitionedIndices{ std::make_shared< vector< index_t > >() } );
                for( size_t j = 0; j < bufferCount; ++j )
                {
                    output.AllBuffers[ j ].push_back( std::make_shared< AriesHostDataBuffer >( buffers[ j ]->GetDataType(), count * 1.1 ) );
                }
            }
            
            for( size_t index = 0; index < count; ++index )
            {
                uint32_t hashVal = 0;
                size_t itemPos = pos + index;
                for( const auto& wrapper : dataWrapper )
                {
                    auto itemStart = wrapper.data + ( pos + index ) * wrapper.perItemSize;
                    hashVal ^= BKDRHash( itemStart + wrapper.hasNull, wrapper.perItemSize - wrapper.hasNull, seed );
                    if ( hasNullValues && wrapper.hasNull && !( *itemStart ) )
                    {
                        bHasNull = true;
                    }
                }
                output.AllIndices[ ( hashVal % BUCKET_NUM ) / perBucketSize ].Indices->push_back( itemPos );
                for( size_t j = 0; j < bufferCount; ++j )
                {
                    output.AllBuffers[ j ][ ( hashVal % BUCKET_NUM ) / perBucketSize ]->AddItem( buffers[ j ]->GetData( itemPos ) );
                }
            }
            return output;
        }, offset, itemCount ) );
        offset += perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
    }

    vector< PartitionResult > allResult;
    for( auto& t : allThreads )
        allResult.push_back( t.get() );

    vector< size_t > totalSizes;
    totalSizes.resize( bucketCount, 0 );
    for( const auto& r : allResult )
    {
        for( size_t i = 0; i < r.AllIndices.size(); ++i )
            totalSizes[ i ] += r.AllIndices[ i ].Indices->size();
    }

    for( size_t i = 0; i < bucketCount; ++i )
    {
        result.AllIndices[ i ].Indices->reserve( totalSizes[ i ] );
        for( const auto& r : allResult )
        {
            result.AllIndices[ i ].Indices->insert( result.AllIndices[ i ].Indices->end(), r.AllIndices[ i ].Indices->begin(), r.AllIndices[ i ].Indices->end() );
            for( size_t j = 0; j < bufferCount; ++j )
            {
                result.AllBuffers[ j ][ i ]->Merge( r.AllBuffers[ j ][ i ] );
            }
        }
    }

    if ( hasNullValues )
        *hasNullValues = bHasNull; 

    return result;    
}

 void GatherGroupDataByCPU(
    const int8_t* input,
    const size_t item_length,
    const int32_t block_count,
    const index_t* associated,
    const index_t* indices,
    const size_t indices_count,
    const int8_t* output )
{
    int64_t blockCount = GetThreadNum( indices_count );
    int64_t blockSize = indices_count / blockCount;
    int64_t extraSize = indices_count % blockCount;

    std::vector< std::future< void > > workThreads;

    for ( int i = 0; i < blockCount; ++i )
    {
        auto pOutputBlock = output + item_length * blockSize * i;
        auto start = blockSize * i;
        auto count = i == blockCount - 1 ? blockSize + extraSize : blockSize;
        workThreads.push_back( std::async( std::launch::async, [=] {
            for ( int64_t j = 0; j < count; j++ )
            {
                auto idx = associated[ indices[ j + start ] ];
                memcpy( ( void* )( pOutputBlock + j * item_length ), input + idx * item_length, item_length );
            }
        } ) );
    }

    for (auto &t : workThreads)
        t.wait();
}