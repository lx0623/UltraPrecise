#include "AriesSqlOperator_hashjoin.h"
#include "AriesSqlOperator_helper.h"
#include "AriesSqlOperator_materialize.h"
#include "kernel_util.hxx"
#include "kernel_scan.hxx"
#include "AriesEngine/AriesDataDef.h"
#include "CudaAcc/hash_join.hxx"
#include "DynamicKernel.h"
#include "CpuTimer.h"
#include <future>
BEGIN_ARIES_ACC_NAMESPACE

    static void get_column_info( const aries_engine::AriesTableBlockUPtr& table, int column_id, bool can_use_dict, std::vector< AriesDataBufferSPtr >& data_buffers, AriesManagedArray< int8_t* >& data_blocks,
            int8_t*& indices, AriesValueType& indicesValueType, AriesColumnType& column_type, AriesInt64ArraySPtr& prefix_sum_of_data_block_size,
            size_t& count )
    {
        auto col_encode_type = table->GetColumnEncodeType( column_id );
        indices = nullptr;
        if( col_encode_type == EncodeType::DICT && !can_use_dict )
        {
            //将字典列还原成字符串
            table->MaterilizeDictEncodedColumn( column_id );
            col_encode_type = EncodeType::NONE;
            assert( table->GetColumnEncodeType( column_id ) == EncodeType::NONE );
        }
            
        aries_engine::AriesColumnSPtr column;
        vector<int64_t> PrefixSum;
        if( table->IsColumnUnMaterilized( column_id ) )
        {
#ifdef ARIES_PROFILE
            CPU_Timer t;
            t.begin();
#endif
            aries_engine::AriesTableBlockStats oldStats = table->GetStats();

            auto column_reference = table->GetUnMaterilizedColumn( column_id );
            aries_engine::AriesBaseColumnSPtr reffered_column = column_reference->GetReferredColumn();
            switch( col_encode_type )
            {
                case EncodeType::NONE:
                {
                    indices = ( int8_t* )column_reference->GetIndices()->GetIndices()->GetData();
#ifdef ARIES_PROFILE
                    oldStats.m_materializeTime += t.end();
#endif
                    table->SetStats( oldStats );

                    indicesValueType = AriesValueType::INT32;
                    column = std::dynamic_pointer_cast< aries_engine::AriesColumn >( reffered_column );
                    count = column_reference->GetRowCount();
                    data_buffers = column->GetDataBuffers();
                    PrefixSum = column->GetBlockSizePsumArray();
                    column_type = column->GetColumnType();
                    break;
                }
                case EncodeType::DICT:
                {
                    assert( can_use_dict );
                    auto dictColumn = std::dynamic_pointer_cast< aries_engine::AriesDictEncodedColumn >( reffered_column );
                    indices = ( int8_t* )column_reference->GetIndices()->GetIndices()->GetData();
                    indicesValueType = AriesValueType::INT32;
                    count = column_reference->GetRowCount();

                    data_buffers = dictColumn->GetIndices()->GetDataBuffers();
                    PrefixSum = dictColumn->GetIndices()->GetBlockSizePsumArray();
                    assert( !data_buffers.empty() );
                    column_type = data_buffers[ 0 ]->GetDataType();
                    break;
                }
            }
        }
        else
        {
            switch( col_encode_type )
            {
                case EncodeType::NONE:
                {
                    column = table->GetMaterilizedColumn( column_id );
                    count = column->GetRowCount();
                    data_buffers = column->GetDataBuffers();
                    PrefixSum = column->GetBlockSizePsumArray();
                    column_type = column->GetColumnType();
                    break;
                }
                case EncodeType::DICT:
                {
                    assert( can_use_dict );
                    auto dictColumn = table->GetDictEncodedColumn( column_id );
                    data_buffers = dictColumn->GetIndices()->GetDataBuffers();
                    count = dictColumn->GetRowCount();
                    PrefixSum = dictColumn->GetIndices()->GetBlockSizePsumArray();
                    assert( !data_buffers.empty() );
                    column_type = data_buffers[ 0 ]->GetDataType();
                    break;
                }
            }
        }
        
        prefix_sum_of_data_block_size = aries_engine::GetPrefixSumOfBlockSize( PrefixSum );
        
        if( column_type.DataType.ValueType == AriesValueType::INT8 || column_type.DataType.ValueType == AriesValueType::INT16 )
        {
            auto int32_column_type = column_type;
            int32_column_type.DataType.ValueType = AriesValueType::INT32;
            size_t tupleNum = count;
            auto int32_buffer = std::make_shared< AriesDataBuffer >( int32_column_type, tupleNum );
            int32_buffer->PrefetchToCpu();
            int32_t* p = (int32_t*)int32_buffer->GetData();

            int64_t blockCount = std::min( std::max( tupleNum / 16, 1ul ), (size_t)thread::hardware_concurrency() );
            int64_t blockSize = tupleNum / blockCount;
            int64_t extraSize = tupleNum % blockCount;
            vector<future<void>> workThreads;
            for (int i = 0; i < blockCount; ++i)
            {
                int64_t offset = blockSize * i;
                int64_t itemCount = blockSize;
                if( i == blockCount - 1 )
                    itemCount += extraSize;
                if( !int32_column_type.HasNull )
                {
                    workThreads.push_back( std::async(std::launch::async,
                        [ & ]( int64_t start, int64_t count, bool is_int16 ) 
                        { 
                            int32_t* pOutput = p + start;
                            int64_t end = start + count;
                            for( int64_t i = start; i < end; ++i )
                            {
                                auto it = std::upper_bound( PrefixSum.cbegin(), PrefixSum.cend(), i );
                                int blockIndex = ( it - PrefixSum.cbegin() ) - 1;
                                int offsetInBlock = i - PrefixSum[ blockIndex ];
                                int8_t* item = data_buffers[ blockIndex ]->GetItemDataAt( offsetInBlock );
                                *pOutput++ = is_int16 ? *( int16_t* )item : *item;
                            }
                        }, offset, itemCount, column_type.DataType.ValueType == AriesValueType::INT16 ) );
                }
                else 
                {
                    nullable_type< int32_t >* pNullable = ( nullable_type< int32_t >* )p;
                    workThreads.push_back( std::async(std::launch::async,
                    [ & ]( int64_t start, int64_t count, bool is_int16 ) 
                    { 
                        nullable_type< int32_t >* pOutput = pNullable + start;
                        int64_t end = start + count;
                        for( int64_t i = start; i < end; ++i )
                        {
                            auto it = std::upper_bound( PrefixSum.cbegin(), PrefixSum.cend(), i );
                            int blockIndex = ( it - PrefixSum.cbegin() ) - 1;
                            int offsetInBlock = i - PrefixSum[ blockIndex ];
                            int8_t* item = data_buffers[ blockIndex ]->GetItemDataAt( offsetInBlock );
                            if( is_int16 )
                                *pOutput = *( nullable_type<int16_t>* )item;
                            else 
                                *pOutput = *( nullable_type<int8_t>* )item;
                            ++pOutput;
                        }
                    }, offset, itemCount, column_type.DataType.ValueType == AriesValueType::INT16 ) );
                }
            }
            for (auto &t : workThreads)
                t.wait();

            data_buffers.clear();
            data_buffers.push_back( int32_buffer );
            column_type = int32_column_type;
        }

        data_blocks.AllocArray( data_buffers.size() );

        int i = 0;
        for( const auto& buffer : data_buffers )
        {
            buffer->PrefetchToGpu();
            data_blocks[i++] = buffer->GetData();
        }
        data_blocks.PrefetchToGpu();
    }

    AriesHashTableMultiKeysUPtr BuildHashTable( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id, const vector< bool >& can_use_dict )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( columns_id.size() );

        std::vector< std::vector< AriesDataBufferSPtr > > data_buffers_array( columns_id.size() );
        for( int i = 0; i < columns_id.size(); i++ )
        {
            const auto& column_id = columns_id[i];
            auto& column_data = columns_data[i];

            AriesColumnType column_type;

            AriesInt64ArraySPtr prefix_sum_of_block_size;

            size_t count = 0;

            get_column_info( table_block, column_id, can_use_dict[ i ], data_buffers_array[ i ], data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                    prefix_sum_of_block_size, count );
            column_data.m_blockCount = data_blocks.GetItemCount();
            column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->ReleaseData();
            column_data.m_data = data_blocks.ReleaseData();
            column_data.m_hasNull = column_type.HasNull;
            column_data.m_nullData = nullptr;
            column_data.m_perItemSize = column_type.GetDataTypeSize();
            auto value_type = column_type.DataType.ValueType;
            switch( value_type )
            {
                case AriesValueType::INT8:
                case AriesValueType::INT16:
                case AriesValueType::INT32:
                case AriesValueType::UINT8:
                case AriesValueType::UINT16:
                case AriesValueType::UINT32:
                {
                    //column_data.m_perItemSize = 4;
                    break;
                }

                case AriesValueType::INT64:
                case AriesValueType::UINT64:
                {
                    //column_data.m_perItemSize = 8;
                    break;
                }
                case AriesValueType::CHAR:
                {
                    //column_data.m_perItemSize = column_type.DataType.Length;
                    break;
                }

                default:
                    return nullptr;
            }
        }

        auto result = build_hash_table( columns_data.GetData(), columns_id.size(), table_block->GetRowCount(), *ctx );
        result->hash_row_count = table_block->GetRowCount();
        for( int i = 0; i < columns_id.size(); i++ )
        {
            const auto& column_id = columns_id[i];
            auto& column_data = columns_data[i];
            ctx->free( column_data.m_blockSizePrefixSum );
            ctx->free( column_data.m_data );

            for ( const auto& buffer : data_buffers_array[ i ] )
            {
                buffer->PrefetchToCpu();
            }
        }
        return result;
    }

    AriesHashTableUPtr BuildHashTable( const aries_engine::AriesTableBlockUPtr& table_block, int column_id, bool can_use_dict )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( 1 );

        AriesColumnType column_type;
        AriesInt64ArraySPtr prefix_sum_of_block_size;
        size_t count = 0;

        auto& column_data = columns_data[0];
        std::vector< AriesDataBufferSPtr > data_buffers;
        get_column_info( table_block, column_id, can_use_dict, data_buffers, data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                prefix_sum_of_block_size, count );
        column_data.m_blockCount = data_blocks.GetItemCount();
        column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->GetData();
        column_data.m_data = data_blocks.GetData();
        column_data.m_hasNull = column_type.HasNull;
        column_data.m_nullData = nullptr;

        auto value_type = column_type.DataType.ValueType;
        auto result = build_hash_table( columns_data.GetData(), table_block->GetRowCount(), value_type, *ctx );
        for ( const auto& buffer : data_buffers )
        {
            buffer->PrefetchToCpu();
        }
        return result;
    }

    void ReleaseHashTable( AriesHashTableMultiKeysUPtr& table )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        if( !table )
        {
            return;
        }

        if( table->keys_array )
        {
            for( int i = 0; i < table->count; i++ )
            {
                ctx->free( table->keys_array[i] );
            }

            ctx->free( table->keys_array );
        }

        if( table->ids )
        {
            ctx->free( table->ids );
        }

        if( table->bad_values_array )
        {
            for( int i = 0; i < table->count; i++ )
            {
                ctx->free( table->bad_values_array[i] );
            }
            ctx->free( table->bad_values_array );
        }

        if( table->bad_ids )
        {
            ctx->free( table->bad_ids );
        }

        if( table->flags_table )
        {
            ctx->free( table->flags_table );
        }

        if( table->keys_length )
        {
            ctx->free( table->keys_length );
        }
    }

    void ReleaseHashTable( AriesHashTableUPtr& table )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        if( !table )
        {
            return;
        }

        if( table->Keys )
        {
            ctx->free( table->Keys );
            table->Keys = nullptr;
        }

        if( table->Ids )
        {
            ctx->free( table->Ids );
            table->Ids = nullptr;
        }

        if( table->BadValues )
        {
            ctx->free( table->BadValues );
            table->BadValues = nullptr;
        }

        if( table->BadIds )
        {
            ctx->free( table->BadIds );
            table->BadIds = nullptr;
        }
    }

    template< typename type_t >
    static void convert_hash_table( const AriesHashTableUPtr& origin, HashTable< type_t >& to )
    {
        to.keys = ( type_t* )origin->Keys;
        to.ids = origin->Ids;
        to.table_size = origin->TableSize;
        to.bad_values = ( type_t* )origin->BadValues;
        to.bad_ids = origin->BadIds;
        to.bad_count = origin->BadCount;
        to.null_value_index = origin->NullValueIndex;
    }

    JoinPair InnerJoinWithHash( const AriesHashTableUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
            const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, int column_id, bool can_use_dict )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        if( table_block->GetRowCount() == 0 )
        {
            return
            {   nullptr, nullptr, 0};
        }

        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( 1 );

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        size_t count = 0;

        auto& column_data = columns_data[0];
        std::vector< AriesDataBufferSPtr > data_buffers;
        get_column_info( table_block, column_id, can_use_dict, data_buffers, data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                prefix_sum_of_block_size, count );
        column_data.m_blockCount = data_blocks.GetItemCount();
        column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->GetData();
        column_data.m_data = data_blocks.GetData();
        column_data.m_hasNull = column_type.HasNull;
        column_data.m_nullData = nullptr;

        switch( hash_table->ValueType )
        {
            case AriesValueType::INT8:
            case AriesValueType::INT16:
            case AriesValueType::INT32:
            case AriesValueType::UINT8:
            case AriesValueType::UINT16:
            case AriesValueType::UINT32:
            {
                HashTable< int32_t > hash_table_impl;
                convert_hash_table( hash_table, hash_table_impl );
                auto result = hash_inner_join( hash_table_impl, columns_data.GetData(), count, column_type, hash_table_indices, table_indices, *ctx );
                for ( const auto& buffer : data_buffers )
                {
                    buffer->PrefetchToCpu();
                }
                return result;
            }

            case AriesValueType::INT64:
            case AriesValueType::UINT64:
            {
                HashTable< unsigned long long > hash_table_impl;
                convert_hash_table( hash_table, hash_table_impl );
                auto result =  hash_inner_join( hash_table_impl, columns_data.GetData(), count, column_type, hash_table_indices, table_indices, *ctx );
                for ( const auto& buffer : data_buffers )
                {
                    buffer->PrefetchToCpu();
                }
                return result;
            }
        }

        throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );

        // return { nullptr, nullptr, 0 };
    }

    AriesJoinResult InnerJoinWithHash( const AriesHashTableMultiKeysUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
            const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, const std::vector< int >& columns_id, const vector< bool >& can_use_dict )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesManagedArray< int8_t* > data_blocks;

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        AriesManagedArray< ColumnDataIterator > columns_data( hash_table->count );

        size_t count = 0;

        std::vector< std::vector< AriesDataBufferSPtr > > data_buffers_array( columns_id.size() );
        for( int i = 0; i < hash_table->count; i++ )
        {
            const auto& column_id = columns_id[i];
            auto& column_data = columns_data[i];

            get_column_info( table_block, column_id, can_use_dict[ i ], data_buffers_array[ i ], data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                    prefix_sum_of_block_size, count );
            column_data.m_blockCount = data_blocks.GetItemCount();
            column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->ReleaseData();
            column_data.m_data = data_blocks.ReleaseData();
            column_data.m_hasNull = column_type.HasNull;
            column_data.m_nullData = nullptr;
            column_data.m_perItemSize = column_type.GetDataTypeSize();
        }

        auto result = hash_inner_join( *hash_table, columns_data.GetData(), hash_table->count, table_block->GetRowCount(), hash_table_indices,
                table_indices, *ctx );
        for( int i = 0; i < hash_table->count; i++ )
        {
            const auto& column_id = columns_id[i];
            auto& column_data = columns_data[i];
            ctx->free( column_data.m_blockSizePrefixSum );
            ctx->free( column_data.m_data );
            for ( const auto& buffer : data_buffers_array[ i ] )
            {
                buffer->PrefetchToCpu();
            }
        }
        return result;
    }

    AriesStarJoinResult StarInnerJoinWithHash( const std::vector< AriesHashTableWrapper >& tables,
            const std::vector< AriesIndicesArraySPtr >& dimension_tables_indices, const std::vector< AriesHashJoinDataWrapper >& datas,
            const AriesIndicesArraySPtr& fact_table_indices, size_t row_count )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        return hash_inner_join( tables, datas, row_count, dimension_tables_indices, fact_table_indices, *ctx );
    }

    JoinPair FullJoinWithHash( const AriesHashTableUPtr& hash_table,
                               const AriesIndicesArraySPtr& hash_table_indices,
                               const aries_engine::AriesTableBlockUPtr& table_block,
                               const AriesIndicesArraySPtr& table_indices,
                               int column_id,
                               const JoinDynamicCodeParams* joinDynamicCodeParams,
                               bool can_use_dict,
                               const bool needToSwap )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        if ( hash_table->HashRowCount == 0 && table_block->GetRowCount() == 0 )
        {
            return { nullptr, nullptr, 0 };
        }

        if ( hash_table->HashRowCount == 0 )
        {
            JoinPair result;
            result.JoinCount = table_block->GetRowCount();

            result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
            init_value( result.LeftIndices->GetData(), result.JoinCount, -1, *ctx );

            if( table_indices )
                result.RightIndices = table_indices;
            else
            {
                result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                init_sequence( result.RightIndices->GetData(), result.JoinCount, *ctx );
            }
            return result;
        }
        else if ( table_block->GetRowCount() == 0 )
        {
            JoinPair result;
            result.JoinCount = hash_table->HashRowCount;
            if( hash_table_indices )
                result.LeftIndices = hash_table_indices;
            else
            {
                result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
            }

            result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
            init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );

            return result;
        }

        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( 1 );

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        size_t count = 0;

        auto& column_data = columns_data[0];
        std::vector< AriesDataBufferSPtr > data_buffers;
        get_column_info( table_block, column_id, can_use_dict, data_buffers, data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                prefix_sum_of_block_size, count );
        column_data.m_blockCount = data_blocks.GetItemCount();
        column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->GetData();
        column_data.m_data = data_blocks.GetData();
        column_data.m_hasNull = column_type.HasNull;
        column_data.m_nullData = nullptr;

        JoinPair result;
        switch( hash_table->ValueType )
        {
            case AriesValueType::INT8:
            case AriesValueType::INT16:
            case AriesValueType::INT32:
            case AriesValueType::UINT8:
            case AriesValueType::UINT16:
            case AriesValueType::UINT32:
            {
                HashTable< int32_t > hash_table_impl;
                convert_hash_table( hash_table, hash_table_impl );
                result = hash_inner_join( hash_table_impl, columns_data.GetData(), count, column_type, hash_table_indices, table_indices, *ctx );
                for ( const auto& buffer : data_buffers )
                {
                    buffer->PrefetchToCpu();
                }
                break;
            }

            case AriesValueType::INT64:
            case AriesValueType::UINT64:
            {
                HashTable< unsigned long long > hash_table_impl;
                convert_hash_table( hash_table, hash_table_impl );
                result =  hash_inner_join( hash_table_impl, columns_data.GetData(), count, column_type, hash_table_indices, table_indices, *ctx );
                for ( const auto& buffer : data_buffers )
                {
                    buffer->PrefetchToCpu();
                }
                break;
            }
            default:
                throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );
        }

        if ( result.JoinCount > 0 )
        {
            auto associated = make_shared< AriesBoolArray >();
            associated->AllocArray( result.JoinCount );
            auto pAssociated = associated->GetData();
            auto left_indices = result.LeftIndices->GetData();
            auto right_indices = result.RightIndices->GetData();

            if( joinDynamicCodeParams && !joinDynamicCodeParams->functionName.empty() )
            {
                shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
                int8_t** constValues = nullptr;

                auto constValueSize = joinDynamicCodeParams->constValues.size();
                if ( constValueSize  > 0 )
                {
                    constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                    for ( int i = 0; i < constValueSize; ++i )
                        ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                    constValueArr->PrefetchToGpu();
                    constValues = constValueArr->GetData();
                }

                if ( needToSwap )
                    AriesDynamicKernelManager::GetInstance().CallKernel( joinDynamicCodeParams->cuModules,
                                                                    joinDynamicCodeParams->functionName.c_str(),
                                                                    joinDynamicCodeParams->input,
                                                                    right_indices,
                                                                    left_indices,
                                                                    result.JoinCount,
                                                                    ( const int8_t** )constValues,
                                                                    joinDynamicCodeParams->comparators,
                                                                    ( int8_t* )( associated->GetData() ) );
                else
                    AriesDynamicKernelManager::GetInstance().CallKernel( joinDynamicCodeParams->cuModules,
                                                                    joinDynamicCodeParams->functionName.c_str(),
                                                                    joinDynamicCodeParams->input,
                                                                    left_indices,
                                                                    right_indices,
                                                                    result.JoinCount,
                                                                    ( const int8_t** )constValues,
                                                                    joinDynamicCodeParams->comparators,
                                                                    ( int8_t* )( associated->GetData() ) );
                ARIES_CALL_CUDA_API( cudaDeviceSynchronize() );
            }
            else
            {
                FillWithValue( associated, true );
            }

            AriesInt32ArraySPtr prefixSum;
            auto matchedCount = ExclusiveScan( associated, prefixSum );

            AriesInt32Array matchedLeftIndices( matchedCount );
            AriesInt32Array matchedRightIndices( matchedCount );

            auto leftUnmatchedFlags = std::make_shared< AriesInt32Array >( hash_table->HashRowCount );
            auto rightUnmatchedFlags = std::make_shared< AriesInt32Array >( table_block->GetRowCount() );

            FillWithValue( leftUnmatchedFlags, 1 );
            FillWithValue( rightUnmatchedFlags, 1 );

            auto pMatchedLeftIndices = matchedLeftIndices.GetData();
            auto pMatchedRightIndices = matchedRightIndices.GetData();
            auto pPrefixSum = prefixSum->GetData();
            auto pLeftUnmatchedFlags = leftUnmatchedFlags->GetData();
            auto pRightUnmatchedFlags = rightUnmatchedFlags->GetData();

            transform( [=]ARIES_DEVICE( int index )
                {
                    if ( pAssociated[ index ].is_true() )
                    {
                        auto offset = pPrefixSum[ index ];
                        auto left_index = left_indices[ index ];
                        auto right_index = right_indices[ index ];
                        pMatchedLeftIndices[ offset ] = left_index;
                        pMatchedRightIndices[ offset ] = right_index;

                        atomicCAS( pLeftUnmatchedFlags + left_index, 1, 0 );
                        atomicCAS( pRightUnmatchedFlags + right_index, 1, 0 );
                    }
                }, result.JoinCount, *ctx );
            ctx->synchronize();

            auto leftUnmatchedCount = ExclusiveScan( leftUnmatchedFlags, prefixSum );
            AriesInt32Array unmatchedLeftIndicesForLeft( leftUnmatchedCount );
            AriesInt32Array unmatchedRightIndicesForLeft( leftUnmatchedCount );

            auto pUnmatchedLeftIndices = unmatchedLeftIndicesForLeft.GetData();
            auto pUnmatchedRightIndices = unmatchedRightIndicesForLeft.GetData();
            pPrefixSum = prefixSum->GetData();

            transform( [=]ARIES_DEVICE( int index )
                {
                    if( pLeftUnmatchedFlags[ index ] == 1 )
                    {
                        auto offset = pPrefixSum[ index ];
                        pUnmatchedLeftIndices[ offset ] = index;
                        pUnmatchedRightIndices[ offset ] = -1;
                    }
                }, leftUnmatchedFlags->GetItemCount(), *ctx );
            ctx->synchronize();

            auto rightUnmatchedCount = ExclusiveScan( rightUnmatchedFlags, prefixSum );
            AriesInt32Array unmatchedLeftIndicesForRight( rightUnmatchedCount );
            AriesInt32Array unmatchedRightIndicesForRight( rightUnmatchedCount );

            pUnmatchedLeftIndices = unmatchedLeftIndicesForRight.GetData();
            pUnmatchedRightIndices = unmatchedRightIndicesForRight.GetData();
            pPrefixSum = prefixSum->GetData();

            transform( [=]ARIES_DEVICE( int index )
                {
                    if ( pRightUnmatchedFlags[ index ] == 1 )
                    {
                        auto offset = pPrefixSum[ index ];
                        pUnmatchedLeftIndices[ offset ] = -1;
                        pUnmatchedRightIndices[ offset ] = index;
                    }
                }, rightUnmatchedFlags->GetItemCount(), *ctx );
            ctx->synchronize();

            auto totalCount = matchedCount + leftUnmatchedCount + rightUnmatchedCount;
            auto mergedLeftIndices = std::make_shared< AriesInt32Array >( totalCount );
            auto mergedRightIndices = std::make_shared< AriesInt32Array >( totalCount );

            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData(),
                                 matchedLeftIndices.GetData(),
                                 sizeof( int32_t ) * matchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData(),
                                 matchedRightIndices.GetData(),
                                 sizeof( int32_t ) * matchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            auto offset = matchedCount;
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData() + offset,
                                 unmatchedLeftIndicesForLeft.GetData(),
                                 sizeof( int32_t ) * leftUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData() + offset,
                                 unmatchedRightIndicesForLeft.GetData(),
                                 sizeof( int32_t ) * leftUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            offset += leftUnmatchedCount;
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData() + offset,
                                 unmatchedLeftIndicesForRight.GetData(),
                                 sizeof( int32_t ) * rightUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData() + offset,
                                 unmatchedRightIndicesForRight.GetData(),
                                 sizeof( int32_t ) * rightUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            JoinPair result;
            result.JoinCount = totalCount;
            result.LeftIndices = mergedLeftIndices;
            result.RightIndices = mergedRightIndices;
            return result;
        }
        else 
        {
            JoinPair result;
            result.JoinCount = hash_table->HashRowCount + table_block->GetRowCount();
            result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
            result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );

            if( hash_table_indices )
            {
                ARIES_CALL_CUDA_API( cudaMemcpy( result.LeftIndices->GetData(),
                                    hash_table_indices->GetData(),
                                    sizeof(int32_t) * hash_table->HashRowCount,
                                    cudaMemcpyKind::cudaMemcpyDefault ) );
            }
            else
                init_sequence( result.LeftIndices->GetData(), hash_table->HashRowCount, *ctx );

            init_value( result.RightIndices->GetData(), hash_table->HashRowCount, -1, *ctx );

            if( table_indices )
            {
                ARIES_CALL_CUDA_API( cudaMemcpy( result.RightIndices->GetData() + hash_table->HashRowCount,
                                                table_indices->GetData(),
                                                sizeof(int32_t) * table_block->GetRowCount(),
                                                cudaMemcpyKind::cudaMemcpyDefault ) );
            }
            else
                init_sequence( result.RightIndices->GetData() + hash_table->HashRowCount, table_block->GetRowCount(), *ctx );

            init_value( result.LeftIndices->GetData() + hash_table->HashRowCount, table_block->GetRowCount(), -1, *ctx );
            
            return result;
        }
    }

    JoinPair LeftJoinWithHash( const AriesHashTableMultiKeysUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
        const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, const std::vector< int >& column_ids,
        const JoinDynamicCodeParams* joinDynamicCodeParams, const vector< bool >& can_use_dict, bool left_as_hash )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        if( left_as_hash )
        {
            if( hash_table->hash_row_count == 0 )
            {
                return
                {   nullptr, nullptr, 0};
            }
    
            if( table_block->GetRowCount() == 0 )
            {
                JoinPair result;
                result.JoinCount = hash_table->hash_row_count;
                if( hash_table_indices )
                    result.LeftIndices = hash_table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
    
                result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );
    
                return result;
            }
        }
        else 
        {
            if( table_block->GetRowCount() == 0 )
            {
                return
                {   nullptr, nullptr, 0};
            }

            if( hash_table->hash_row_count == 0 )
            {
                JoinPair result;
                result.JoinCount = table_block->GetRowCount();
                if( table_indices )
                    result.LeftIndices = table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
    
                result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );
    
                return result;
            }
        }

        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( hash_table->count );

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        size_t count = 0;
        vector< vector< AriesDataBufferSPtr > > data_buffers_array( column_ids.size() );
        for( int i = 0; i < hash_table->count; i++ )
        {
            auto& column_data = columns_data[ i ];
            
            get_column_info( table_block, column_ids[ i ], can_use_dict[ i ], data_buffers_array[ i ], data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                    prefix_sum_of_block_size, count );
            column_data.m_blockCount = data_blocks.GetItemCount();
            column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->ReleaseData();
            column_data.m_data = data_blocks.ReleaseData();
            column_data.m_hasNull = column_type.HasNull;
            column_data.m_nullData = nullptr;
            column_data.m_perItemSize = column_type.GetDataTypeSize();
        }
        
        AriesJoinResult join_result = hash_inner_join( 
            *hash_table, 
            columns_data.GetData(), 
            hash_table->count, 
            table_block->GetRowCount(),
            hash_table_indices, 
            table_indices, 
            *ctx );

        for( int i = 0; i < hash_table->count; i++ )
        {
            const auto& column_id = column_ids[i];
            auto& column_data = columns_data[i];
            ctx->free( column_data.m_blockSizePrefixSum );
            ctx->free( column_data.m_data );
            for ( const auto& buffer : data_buffers_array[ i ] )
            {
                buffer->PrefetchToCpu();
            }
        }

        JoinPair result = boost::get< JoinPair >( join_result );
        if ( result.JoinCount > 0 )
        {
            cout<<"result.JoinCount="<<result.JoinCount<<endl;
            auto associated = make_shared< AriesBoolArray >();
            associated->AllocArray( result.JoinCount );
            auto pAssociated = associated->GetData();
            auto left_indices = result.LeftIndices->GetData();
            auto right_indices = result.RightIndices->GetData();
            if( !left_as_hash )
                std::swap( left_indices, right_indices );
            if( joinDynamicCodeParams && !joinDynamicCodeParams->functionName.empty() )
            {
                shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
                int8_t** constValues = nullptr;

                auto constValueSize = joinDynamicCodeParams->constValues.size();
                if ( constValueSize  > 0 )
                {
                    constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                    for ( int i = 0; i < constValueSize; ++i )
                        ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                    constValueArr->PrefetchToGpu();
                    constValues = constValueArr->GetData();
                }

                AriesDynamicKernelManager::GetInstance().CallKernel( joinDynamicCodeParams->cuModules,
                                                                joinDynamicCodeParams->functionName.c_str(),
                                                                joinDynamicCodeParams->input,
                                                                left_indices,
                                                                right_indices,
                                                                result.JoinCount,
                                                                ( const int8_t** )constValues,
                                                                joinDynamicCodeParams->comparators,
                                                                ( int8_t* )( associated->GetData() ) );
                ARIES_CALL_CUDA_API( cudaDeviceSynchronize() );
            }
            else
            {
                FillWithValue( associated, true );
            }

            AriesInt32ArraySPtr prefixSum;
            auto matchedCount = ExclusiveScan( associated, prefixSum );
            auto pPrefixSum = prefixSum->GetData();

            AriesInt32Array matchedLeftIndices( matchedCount );
            AriesInt32Array matchedRightIndices( matchedCount );
            size_t leftTableRowCount = left_as_hash ? hash_table->hash_row_count : table_block->GetRowCount();
            auto leftUnmatchedFlags = std::make_shared< AriesInt32Array >( leftTableRowCount );
            FillWithValue( leftUnmatchedFlags, 1 );

            auto pMatchedLeftIndices = matchedLeftIndices.GetData();
            auto pMatchedRightIndices = matchedRightIndices.GetData();
            auto pLeftUnmatchedFlags = leftUnmatchedFlags->GetData();
            
            transform( [=]ARIES_DEVICE( int index )
                {
                    if ( pAssociated[ index ].is_true() )
                    {
                        auto offset = pPrefixSum[ index ];
                        auto left_index = left_indices[ index ];
                        pMatchedLeftIndices[ offset ] = left_index;
                        pMatchedRightIndices[ offset ] = right_indices[ index ];
                        atomicCAS( pLeftUnmatchedFlags + left_index, 1, 0 );
                    }
                }, result.JoinCount, *ctx );
            ctx->synchronize();

            auto leftUnmatchedCount = ExclusiveScan( leftUnmatchedFlags, prefixSum );
            AriesInt32Array unmatchedLeftIndicesForLeft( leftUnmatchedCount );
            AriesInt32Array unmatchedRightIndicesForLeft( leftUnmatchedCount );

            auto pUnmatchedLeftIndices = unmatchedLeftIndicesForLeft.GetData();
            auto pUnmatchedRightIndices = unmatchedRightIndicesForLeft.GetData();
            pPrefixSum = prefixSum->GetData();

            transform( [=]ARIES_DEVICE( int index )
                {
                    if( pLeftUnmatchedFlags[ index ] == 1 )
                    {
                        auto offset = pPrefixSum[ index ];
                        pUnmatchedLeftIndices[ offset ] = index;
                        pUnmatchedRightIndices[ offset ] = -1;
                    }
                }, leftUnmatchedFlags->GetItemCount(), *ctx );
            ctx->synchronize();

            auto totalCount = matchedCount + leftUnmatchedCount;
            auto mergedLeftIndices = std::make_shared< AriesInt32Array >( totalCount );
            auto mergedRightIndices = std::make_shared< AriesInt32Array >( totalCount );

            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData(),
                                 matchedLeftIndices.GetData(),
                                 sizeof( int32_t ) * matchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData(),
                                 matchedRightIndices.GetData(),
                                 sizeof( int32_t ) * matchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            auto offset = matchedCount;
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData() + offset,
                                 unmatchedLeftIndicesForLeft.GetData(),
                                 sizeof( int32_t ) * leftUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData() + offset,
                                 unmatchedRightIndicesForLeft.GetData(),
                                 sizeof( int32_t ) * leftUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            ARIES_CALL_CUDA_API( cudaDeviceSynchronize() );

            JoinPair result;
            result.JoinCount = totalCount;
            result.LeftIndices = mergedLeftIndices;
            result.RightIndices = mergedRightIndices;
            return result;
        }
        else 
        {
            JoinPair result;
            if( left_as_hash )
            {
                result.JoinCount = hash_table->hash_row_count;
                if( hash_table_indices )
                    result.LeftIndices = hash_table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
            }
            else 
            {
                result.JoinCount = table_block->GetRowCount();
                if( table_indices )
                    result.LeftIndices = table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
            }
            result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
            init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );

            return result;
        }
    }

    JoinPair LeftJoinWithHash( const AriesHashTableUPtr& hash_table,
                               const AriesIndicesArraySPtr& hash_table_indices,
                               const aries_engine::AriesTableBlockUPtr& table_block,
                               const AriesIndicesArraySPtr& table_indices,
                               int column_id,
                               const JoinDynamicCodeParams* joinDynamicCodeParams,
                               bool can_use_dict,
                               bool left_as_hash )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        if( left_as_hash )
        {
            if( hash_table->HashRowCount == 0 )
            {
                return
                {   nullptr, nullptr, 0};
            }
    
            if( table_block->GetRowCount() == 0 )
            {
                JoinPair result;
                result.JoinCount = hash_table->HashRowCount;
                if( hash_table_indices )
                    result.LeftIndices = hash_table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
    
                result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );
    
                return result;
            }
        }
        else 
        {
            if( table_block->GetRowCount() == 0 )
            {
                return
                {   nullptr, nullptr, 0};
            }

            if( hash_table->HashRowCount == 0 )
            {
                JoinPair result;
                result.JoinCount = table_block->GetRowCount();
                if( table_indices )
                    result.LeftIndices = table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
    
                result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );
    
                return result;
            }
        }

        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( 1 );

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        size_t count = 0;

        auto& column_data = columns_data[0];
        std::vector< AriesDataBufferSPtr > data_buffers;
        get_column_info( table_block, column_id, can_use_dict, data_buffers, data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                prefix_sum_of_block_size, count );
        column_data.m_blockCount = data_blocks.GetItemCount();
        column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->GetData();
        column_data.m_data = data_blocks.GetData();
        column_data.m_hasNull = column_type.HasNull;
        column_data.m_nullData = nullptr;

        JoinPair result;
        switch( hash_table->ValueType )
        {
            case AriesValueType::INT8:
            case AriesValueType::INT16:
            case AriesValueType::INT32:
            case AriesValueType::UINT8:
            case AriesValueType::UINT16:
            case AriesValueType::UINT32:
            {
                HashTable< int32_t > hash_table_impl;
                convert_hash_table( hash_table, hash_table_impl );
                result = hash_inner_join( hash_table_impl, columns_data.GetData(), count, column_type, hash_table_indices, table_indices, *ctx );
                for ( const auto& buffer : data_buffers )
                {
                    buffer->PrefetchToCpu();
                }
                break;
            }

            case AriesValueType::INT64:
            case AriesValueType::UINT64:
            {
                HashTable< unsigned long long > hash_table_impl;
                convert_hash_table( hash_table, hash_table_impl );
                result = hash_inner_join( hash_table_impl, columns_data.GetData(), count, column_type, hash_table_indices, table_indices, *ctx );
                for ( const auto& buffer : data_buffers )
                {
                    buffer->PrefetchToCpu();
                }
                break;
            }
            default:
                throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );
        }

        if ( result.JoinCount > 0 )
        {
            auto associated = make_shared< AriesBoolArray >();
            associated->AllocArray( result.JoinCount );
            auto pAssociated = associated->GetData();
            auto left_indices = result.LeftIndices->GetData();
            auto right_indices = result.RightIndices->GetData();
            if( !left_as_hash )
                std::swap( left_indices, right_indices );
            if( joinDynamicCodeParams && !joinDynamicCodeParams->functionName.empty() )
            {
                shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
                int8_t** constValues = nullptr;

                auto constValueSize = joinDynamicCodeParams->constValues.size();
                if ( constValueSize  > 0 )
                {
                    constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                    for ( int i = 0; i < constValueSize; ++i )
                        ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                    constValueArr->PrefetchToGpu();
                    constValues = constValueArr->GetData();
                }

                AriesDynamicKernelManager::GetInstance().CallKernel( joinDynamicCodeParams->cuModules,
                                                                joinDynamicCodeParams->functionName.c_str(),
                                                                joinDynamicCodeParams->input,
                                                                left_indices,
                                                                right_indices,
                                                                result.JoinCount,
                                                                ( const int8_t** )constValues,
                                                                joinDynamicCodeParams->comparators,
                                                                ( int8_t* )( associated->GetData() ) );
                ARIES_CALL_CUDA_API( cudaDeviceSynchronize() );
            }
            else
            {
                FillWithValue( associated, true );
            }

            AriesInt32ArraySPtr prefixSum;
            auto matchedCount = ExclusiveScan( associated, prefixSum );
            auto pPrefixSum = prefixSum->GetData();

            AriesInt32Array matchedLeftIndices( matchedCount );
            AriesInt32Array matchedRightIndices( matchedCount );
            size_t leftTableRowCount = left_as_hash ? hash_table->HashRowCount : table_block->GetRowCount();
            auto leftUnmatchedFlags = std::make_shared< AriesInt32Array >( leftTableRowCount );
            FillWithValue( leftUnmatchedFlags, 1 );

            auto pMatchedLeftIndices = matchedLeftIndices.GetData();
            auto pMatchedRightIndices = matchedRightIndices.GetData();
            auto pLeftUnmatchedFlags = leftUnmatchedFlags->GetData();
            
            transform( [=]ARIES_DEVICE( int index )
                {
                    if ( pAssociated[ index ].is_true() )
                    {
                        auto offset = pPrefixSum[ index ];
                        auto left_index = left_indices[ index ];
                        pMatchedLeftIndices[ offset ] = left_index;
                        pMatchedRightIndices[ offset ] = right_indices[ index ];
                        atomicCAS( pLeftUnmatchedFlags + left_index, 1, 0 );
                    }
                }, result.JoinCount, *ctx );
            ctx->synchronize();

            auto leftUnmatchedCount = ExclusiveScan( leftUnmatchedFlags, prefixSum );
            AriesInt32Array unmatchedLeftIndicesForLeft( leftUnmatchedCount );
            AriesInt32Array unmatchedRightIndicesForLeft( leftUnmatchedCount );

            auto pUnmatchedLeftIndices = unmatchedLeftIndicesForLeft.GetData();
            auto pUnmatchedRightIndices = unmatchedRightIndicesForLeft.GetData();
            pPrefixSum = prefixSum->GetData();

            transform( [=]ARIES_DEVICE( int index )
                {
                    if( pLeftUnmatchedFlags[ index ] == 1 )
                    {
                        auto offset = pPrefixSum[ index ];
                        pUnmatchedLeftIndices[ offset ] = index;
                        pUnmatchedRightIndices[ offset ] = -1;
                    }
                }, leftUnmatchedFlags->GetItemCount(), *ctx );
            ctx->synchronize();

            auto totalCount = matchedCount + leftUnmatchedCount;
            auto mergedLeftIndices = std::make_shared< AriesInt32Array >( totalCount );
            auto mergedRightIndices = std::make_shared< AriesInt32Array >( totalCount );

            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData(),
                                 matchedLeftIndices.GetData(),
                                 sizeof( int32_t ) * matchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData(),
                                 matchedRightIndices.GetData(),
                                 sizeof( int32_t ) * matchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            auto offset = matchedCount;
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices->GetData() + offset,
                                 unmatchedLeftIndicesForLeft.GetData(),
                                 sizeof( int32_t ) * leftUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices->GetData() + offset,
                                 unmatchedRightIndicesForLeft.GetData(),
                                 sizeof( int32_t ) * leftUnmatchedCount,
                                 cudaMemcpyKind::cudaMemcpyDefault ) );

            ARIES_CALL_CUDA_API( cudaDeviceSynchronize() );

            JoinPair result;
            result.JoinCount = totalCount;
            result.LeftIndices = mergedLeftIndices;
            result.RightIndices = mergedRightIndices;
            return result;
        }
        else 
        {
            JoinPair result;
            if( left_as_hash )
            {
                result.JoinCount = hash_table->HashRowCount;
                if( hash_table_indices )
                    result.LeftIndices = hash_table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
            }
            else
            {
                result.JoinCount = table_block->GetRowCount();
                if( table_indices )
                    result.LeftIndices = table_indices;
                else
                {
                    result.LeftIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
                    init_sequence( result.LeftIndices->GetData(), result.JoinCount, *ctx );
                }
            }

            result.RightIndices = std::make_shared< AriesInt32Array >( result.JoinCount );
            init_value( result.RightIndices->GetData(), result.JoinCount, -1, *ctx );

            return result;
        }
    }

    AriesInt32ArraySPtr HalfJoinWithLeftHashInternal( const AriesHashTableUPtr& left_hash_table, const aries_engine::AriesTableBlockUPtr& table,
            int column_id, const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool isSemiJoin, bool isNotIn )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesInt32ArraySPtr result = std::make_shared< AriesInt32Array >();

        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( 1 );

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        size_t count = 0;

        auto& column_data = columns_data[0];
        std::vector< AriesDataBufferSPtr > data_buffers;
        get_column_info( table, column_id, can_use_dict, data_buffers, data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type, prefix_sum_of_block_size,
                count );
        column_data.m_blockCount = data_blocks.GetItemCount();
        column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->GetData();
        column_data.m_data = data_blocks.GetData();
        column_data.m_hasNull = column_type.HasNull;
        column_data.m_nullData = nullptr;
        if( !joinDynamicCodeParams || joinDynamicCodeParams->functionName.empty() )
        {
            switch( left_hash_table->ValueType )
            {
                case AriesValueType::INT8:
                case AriesValueType::INT16:
                case AriesValueType::INT32:
                case AriesValueType::UINT8:
                case AriesValueType::UINT16:
                case AriesValueType::UINT32:
                {
                    HashTable< int32_t > hash_table_impl;
                    convert_hash_table( left_hash_table, hash_table_impl );
                    result = simple_half_join_left_as_hash( hash_table_impl, columns_data.GetData(), column_type, table->GetRowCount(),
                            left_hash_table->HashRowCount, isSemiJoin, isNotIn, *ctx );
                    break;
                }

                case AriesValueType::INT64:
                case AriesValueType::UINT64:
                {
                    HashTable< unsigned long long > hash_table_impl;
                    convert_hash_table( left_hash_table, hash_table_impl );
                    result = simple_half_join_left_as_hash( hash_table_impl, columns_data.GetData(), column_type, table->GetRowCount(),
                            left_hash_table->HashRowCount, isSemiJoin, isNotIn, *ctx );
                    break;
                }
                default:
                    ARIES_ASSERT( 0, "unsupported value type with hash half join" );
            }
        }
        else
        {
            ARIES_ASSERT( 0, "dynamic kernel is not supported in hash half join" );
        }
        return result;
    }

    AriesInt32ArraySPtr HalfJoinWithLeftHash( AriesJoinType joinType, const AriesHashTableUPtr& left_hash_table, const AriesIndicesArraySPtr& indices,
            const aries_engine::AriesTableBlockUPtr& right_table, int column_id, const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool isNotIn )
    {
        assert( joinType == AriesJoinType::SEMI_JOIN || joinType == AriesJoinType::ANTI_JOIN );
        assert( left_hash_table->HashRowCount > 0 && right_table->GetRowCount() > 0 );
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();

        AriesInt32ArraySPtr result = std::make_shared< AriesInt32Array >();

        AriesInt32ArraySPtr associated = HalfJoinWithLeftHashInternal( left_hash_table, right_table, column_id, joinDynamicCodeParams, can_use_dict,
                joinType == AriesJoinType::SEMI_JOIN, isNotIn );
        AriesInt32ArraySPtr psum;
        int32_t resultTupleNum = ExclusiveScan( associated, psum );
        if( resultTupleNum > 0 )
        {
            result->AllocArray( resultTupleNum );

            AriesInt32ArraySPtr indicesArray = indices;
            if( !indicesArray )
            {
                indicesArray = associated->CloneWithNoContent();
                InitSequenceValue( indicesArray );
            }

            gather_filtered_data( indicesArray->GetData(), indicesArray->GetItemCount(), associated->GetData(), psum->GetData(), result->GetData(),
                    *ctx );
        }

        return result;
    }

    AriesInt32ArraySPtr HalfJoinWithRightHash( AriesJoinType joinType, const AriesHashTableUPtr& right_hash_table,
            const AriesIndicesArraySPtr& indices, const aries_engine::AriesTableBlockUPtr& left_table, int column_id,
            const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool isNotIn )
    {
        assert( joinType == AriesJoinType::SEMI_JOIN || joinType == AriesJoinType::ANTI_JOIN );
        assert( right_hash_table->HashRowCount > 0 && left_table->GetRowCount() > 0 );
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();

        AriesInt32ArraySPtr associated;
        AriesManagedArray< int8_t* > data_blocks;
        AriesManagedArray< ColumnDataIterator > columns_data( 1 );

        AriesColumnType column_type;

        AriesInt64ArraySPtr prefix_sum_of_block_size;

        size_t count = 0;

        auto& column_data = columns_data[0];
        std::vector< AriesDataBufferSPtr > data_buffers;
        get_column_info( left_table, column_id, can_use_dict, data_buffers, data_blocks, column_data.m_indices, column_data.m_indiceValueType, column_type,
                prefix_sum_of_block_size, count );
        column_data.m_blockCount = data_blocks.GetItemCount();
        column_data.m_blockSizePrefixSum = prefix_sum_of_block_size->GetData();
        column_data.m_data = data_blocks.GetData();
        column_data.m_hasNull = column_type.HasNull;
        column_data.m_nullData = nullptr;
        if( !joinDynamicCodeParams || joinDynamicCodeParams->functionName.empty() )
        {
            switch( right_hash_table->ValueType )
            {
                case AriesValueType::INT8:
                case AriesValueType::INT16:
                case AriesValueType::INT32:
                case AriesValueType::UINT8:
                case AriesValueType::UINT16:
                case AriesValueType::UINT32:
                {
                    HashTable< int32_t > hash_table_impl;
                    convert_hash_table( right_hash_table, hash_table_impl );
                    associated = simple_half_join_right_as_hash( hash_table_impl, columns_data.GetData(), column_type, left_table->GetRowCount(),
                            joinType == AriesJoinType::SEMI_JOIN, isNotIn, ( bool )indices, *ctx );
                    break;
                }

                case AriesValueType::INT64:
                case AriesValueType::UINT64:
                {
                    HashTable< unsigned long long > hash_table_impl;
                    convert_hash_table( right_hash_table, hash_table_impl );
                    associated = simple_half_join_right_as_hash( hash_table_impl, columns_data.GetData(), column_type, left_table->GetRowCount(),
                            joinType == AriesJoinType::SEMI_JOIN, isNotIn, ( bool )indices, *ctx );
                    break;
                }
                default:
                    ARIES_ASSERT( 0, "unsupported value type with hash half join" );
            }
        }
        else
        {
            ARIES_ASSERT( 0, "dynamic kernel is not supported in hash half join" );
        }

        AriesInt32ArraySPtr result;
        if( !indices )
        {
            result = std::make_shared< AriesInt32Array >();
            AriesInt32ArraySPtr psum;
            int32_t resultTupleNum = ExclusiveScan( associated, psum );
            if( resultTupleNum > 0 )
            {
                result->AllocArray( resultTupleNum );

                AriesInt32ArraySPtr indicesArray = associated->CloneWithNoContent();
                InitSequenceValue( indicesArray );

                gather_filtered_data( indicesArray->GetData(), indicesArray->GetItemCount(), associated->GetData(), psum->GetData(),
                        result->GetData(), *ctx );
            }
        }
        else
            result = associated;
        return result;
    }

    size_t GetHashTableSizePerRow( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id )
    {
        size_t usage = 0;
        usage += sizeof( HashIdType ) * ( HASH_SCALE_FACTOR + HASH_BAD_SCALE_FACTOR );
        if ( columns_id.size() > 1 )
        {
            usage += sizeof( flag_type_t ) * ( HASH_SCALE_FACTOR + HASH_BAD_SCALE_FACTOR );
        }
        return usage;
    }

    size_t EstimateBuildHashTableMemOccupancyPerRow( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id )
    {
        size_t usage = 0;
        for ( const auto id : columns_id )
        {
            auto column_type = table_block->GetColumnType( id );
            usage += column_type.GetDataTypeSize();
        }

        usage += GetHashTableSizePerRow( table_block, columns_id );
        return usage;
    }

    size_t GetLeftHashJoinUsage(
        const size_t hash_tabel_row_count,
        const size_t value_table_row_count,
        const aries_engine::AriesTableBlockUPtr& value_table,
        const std::vector< int >& columns_id,
        bool has_indices )
    {
        // 对于左表的每一行数据，right_row_count 代表与之对应的右表数据的行数
        auto right_row_count = ( value_table_row_count - 1 + hash_tabel_row_count ) / hash_tabel_row_count;

        // left & right output usage
        auto usage = sizeof( int32_t ) * 2 * ( 1 + right_row_count );

        usage += sizeof( int32_t ); // left_matched_count

        for ( const auto& id : columns_id )
        {
            auto type = value_table->GetColumnType( id );
            usage += type.GetDataTypeSize() * right_row_count;
        }

        if ( has_indices )
        {
            usage += sizeof( index_t ) * right_row_count;
        }
        return usage;
    }

    size_t EstimateBuildHashTableMemOccupancy( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id )
    {
        size_t usage = 0;
        auto totalRowCount = table_block->GetRowCount();

        for( auto id : columns_id )
        {
            auto block = table_block->GetColumnBuffer( id );
            if( block->GetItemCount() == 0 )
                return 0;
            usage += block->GetTotalBytes();

            // add mem occupancy for hash key
            usage += ( size_t )( block->GetItemSizeInBytes() * totalRowCount * ( HASH_SCALE_FACTOR + HASH_BAD_SCALE_FACTOR ) );
        }

        // add mem occupancy for hash id
        usage += ( size_t )( sizeof( HashIdType ) * totalRowCount * ( HASH_SCALE_FACTOR + HASH_BAD_SCALE_FACTOR ) ); 

        // add mem occupancy for flags if needed
        if( columns_id.size() > 1 )
            usage += ( size_t )( sizeof( flag_type_t ) * totalRowCount * ( HASH_SCALE_FACTOR + HASH_BAD_SCALE_FACTOR ) );

        return usage;
    }

    size_t EstimateHashInnerJoinPerRowMemOccupancy( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id )
    {
        size_t usage = 0;

        for( const auto& id : columns_id )
        {
            auto block = table_block->GetColumnBuffer( id );
            if( block->GetItemCount() == 0 )
                return 0;
            usage += block->GetItemSizeInBytes();
        }
        // add mem occupancy for output
        usage += sizeof( index_t ) * 2;
        return usage;
    }

    size_t GetBuildHashTableUsage(
        const aries_engine::AriesTableBlockUPtr& originTable,
        const std::vector< int32_t >& columnIds,
        const size_t partRowCount )
    {
        auto totalRowCount = originTable->GetRowCount();
        if ( totalRowCount == 0 || partRowCount == 0 )
        {
            return 0;
        }

        auto hashTableSize = size_t( partRowCount * HASH_SCALE_FACTOR );
        auto badZoneSize = size_t( partRowCount * HASH_BAD_SCALE_FACTOR );

        size_t usage = sizeof( index_t ) * partRowCount;
        usage += sizeof( HashIdType ) * hashTableSize;
        for ( const auto id : columnIds )
        {
            const auto type = originTable->GetColumnType( id );
            usage += type.GetDataTypeSize() * totalRowCount;
            usage += type.GetDataTypeSize() * ( badZoneSize + hashTableSize );
        }

        if ( columnIds.size() > 1 )
        {
            usage += sizeof( flag_type_t ) * hashTableSize;
        }

        return usage;
    }

    size_t GetHashTableSize(
        const aries_engine::AriesTableBlockUPtr& originTable,
        const std::vector< int32_t >& columnIds,
        const size_t partRowCount
    )
    {
        auto hashTableSize = size_t( partRowCount * HASH_SCALE_FACTOR );
        auto badZoneSize = size_t( partRowCount * HASH_BAD_SCALE_FACTOR );

        size_t usage = sizeof( HashIdType ) * hashTableSize;
        for ( const auto id : columnIds )
        {
            const auto type = originTable->GetColumnType( id );
            usage += type.GetDataTypeSize() * ( badZoneSize + hashTableSize );
        }

        if ( columnIds.size() > 1 )
        {
            usage += sizeof( flag_type_t ) * hashTableSize;
        }

        return usage;
    }

END_ARIES_ACC_NAMESPACE
