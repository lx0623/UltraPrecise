#include "AriesSqlOperator_helper.h"
#include "sort_column.h"
#include "AriesEngine/AriesCommonExpr.h"
#include "AriesEngineAlgorithm.h"
#include "AriesSqlOperator_sort.h"
#include <future>
using namespace aries_engine;

BEGIN_ARIES_ACC_NAMESPACE

    aries_engine::AriesColumnSPtr CreateRowIdColumnMaterialized( int64_t initialTableRowCount, int64_t deltaTableRowCount, int64_t blockSize )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesColumnSPtr result = std::make_shared< AriesColumn >();

        //create initial table rowids, [-1, -2, ...]
        int initialBufferCount = ( initialTableRowCount + blockSize - 1 ) / blockSize;
        vector< int8_t* > hostDataBlocks;
        for( int i = 0; i < initialBufferCount; ++i )
        {
            AriesDataBufferSPtr buffer = std::make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::INT32, 1 ), false ),
                    std::min( blockSize, initialTableRowCount - i * blockSize ) );
            buffer->PrefetchToGpu();
            hostDataBlocks.push_back( buffer->GetData() );
            result->AddDataBuffer( buffer );
        }

        AriesArray< int8_t* > dataBlocks( initialBufferCount );
        dataBlocks.CopyFromHostMem( hostDataBlocks.data(), dataBlocks.GetTotalBytes() );
        int32_t** pDataBlocks = ( int32_t** )dataBlocks.GetData();
        transform( [ = ]ARIES_DEVICE( int64_t index )
                {
                    pDataBlocks[ index / blockSize ][ index % blockSize ] = -index - 1;
                }, initialTableRowCount, *ctx );

        //create delta table rowids, [1, 2, ...]
        AriesDataBufferSPtr buffer = std::make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::INT32, 1 ), false ),
                deltaTableRowCount );
        buffer->PrefetchToGpu();
        result->AddDataBuffer( buffer );
        init_sequence_begin_with( ( int32_t* )buffer->GetData(), buffer->GetItemCount(), 1, *ctx );

        return result;
    }

    AriesColumnSPtr CreateRowIdColumn( int64_t initialTableRowCount, int64_t deltaTableRowCount, int64_t blockSize )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesColumnSPtr result = std::make_shared< AriesColumn >();

        //create initial table rowids, [-1, -2, ...]
        int initialBufferCount = ( initialTableRowCount + blockSize - 1 ) / blockSize;
        vector< int8_t* > hostDataBlocks;
        for( int i = 0; i < initialBufferCount; ++i )
        {
            auto count = std::min( blockSize, initialTableRowCount - i * blockSize );
            int start = -1 - i * blockSize;
            AriesRowPosRangeBufferSPtr buffer = std::make_shared< AriesRowPosRangeBuffer >(
                AriesColumnType( AriesDataType( AriesValueType::INT32, 1 ), false ),
                start,
                count );
            result->AddDataBuffer( buffer );
        }

        result->AddDataBuffer( std::make_shared< AriesRowPosRangeBuffer >(
            AriesColumnType( AriesDataType( AriesValueType::INT32, 1 ), false ),
            1,
            deltaTableRowCount
            ) );

        return result;
    }

    AriesIndicesArraySPtr CreateIndicesForMvccTable( const std::vector< int >& inivisibleIdsInInitialTable,
                                                     const std::vector< int >& visibleIdsInDeltaTable,
                                                     size_t tupleNumInInitialTable,
                                                     size_t tupleNumInDeltaTable )
    {
        auto context = AriesSqlOperatorContext::GetInstance().GetContext();

        auto total_row_count = tupleNumInInitialTable - inivisibleIdsInInitialTable.size() + visibleIdsInDeltaTable.size();

        auto indices = std::make_shared< AriesIndicesArray >( total_row_count );

        InitSequenceValue( indices );

        auto size_of_invisible = inivisibleIdsInInitialTable.size() * sizeof( int );
        auto size_of_visible = visibleIdsInDeltaTable.size() * sizeof( int );
        auto tmp_buffer_size = max( size_of_invisible, size_of_visible );
        auto orig_size_of_associated = tupleNumInDeltaTable + tupleNumInInitialTable; // bool
        auto size_of_associated = orig_size_of_associated; // bool
        size_of_associated = div_up( size_of_associated, size_t( 4 ) ) * 4; // avoid "misaligned address"
        auto size_of_prefix_sum = ( tupleNumInDeltaTable + tupleNumInInitialTable ) * sizeof( index_t );

        if ( size_of_invisible == 0 && size_of_visible == 0 )
        {
            return indices;
        }
        AriesArray< int8_t > buffer( tmp_buffer_size + size_of_associated + size_of_prefix_sum );
        int* ptr = ( int* )buffer.GetData();

        auto associated_ptr = ( bool* )( (int8_t* )ptr + tmp_buffer_size );
        auto prefix_sum_ptr = ( int* )( (int8_t* )ptr + tmp_buffer_size + size_of_associated );
        init_value( associated_ptr, tupleNumInInitialTable, true, *context );
        init_value( associated_ptr + tupleNumInInitialTable, tupleNumInDeltaTable, false, *context );

        if ( size_of_invisible > 0 )
        {
            cudaMemcpy( ptr, inivisibleIdsInInitialTable.data(), size_of_invisible, cudaMemcpyHostToDevice );
            auto k = [ = ]ARIES_DEVICE( int index )
            {
                int i = -1 - ptr[ index ];
                associated_ptr[ i ] = false;
            };

            transform( k, inivisibleIdsInInitialTable.size(), *context );
        }

        if ( size_of_visible > 0 )
        {
            cudaMemcpy( ptr , visibleIdsInDeltaTable.data(), size_of_visible, cudaMemcpyHostToDevice );
            auto offset = tupleNumInInitialTable;
            auto k = [ = ]ARIES_DEVICE( int index )
            {
                int i = ptr[ index ] - 1 + offset;
                associated_ptr[ i ] = true;
            };

            transform( k, visibleIdsInDeltaTable.size(), *context );
        }

        managed_mem_t< index_t > psum_count( 1, *context );

        transform_scan< index_t >( [ = ]ARIES_DEVICE( int index ) 
        {
            return associated_ptr[ index ];
        }, orig_size_of_associated, prefix_sum_ptr, plus_t< index_t >(), psum_count.data(), *context );

        assert( ( context->synchronize(), *psum_count.data() == total_row_count ) );

        auto indices_ptr = indices->GetData();
        transform( [ = ]ARIES_DEVICE( int index )
        {
            if ( associated_ptr[ index ] )
            {
                auto idx = prefix_sum_ptr[ index ];
                indices_ptr[ idx ] = index;
            }
        }, orig_size_of_associated, *context );

        return indices;
    }

    AriesDataBufferSPtr CreateVisibleRowIds( int64_t tupleNum, const AriesInt32ArraySPtr& invisibleIds )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        mem_t< int > flags( tupleNum );
        init_value( flags.data(), tupleNum, 1, *ctx );

        mem_t< int > invisible_ids( invisibleIds->GetItemCount() );
        cudaMemcpy( invisible_ids.data(), invisibleIds->GetData(), invisibleIds->GetTotalBytes(), cudaMemcpyDefault );

        auto p_flags = flags.data();
        auto p_invisible_ids = invisible_ids.data();

        auto k = [ = ]ARIES_DEVICE( int index )
        {
            int i = -1 - p_invisible_ids[ index ];
            p_flags[ i ] = 0;
        };

        transform( k, invisibleIds->GetItemCount(), *ctx );
        ctx->synchronize();

        invisible_ids.free();

        mem_t< int > psum( tupleNum );
        managed_mem_t< int > psum_total( 1, *ctx );
        auto p_psum = psum.data();

        scan( p_flags, tupleNum, p_psum, plus_t< int32_t >(), psum_total.data(), *ctx );
        ctx->synchronize();

        auto visible_count = psum_total.data()[0];

        mem_t< int > visible_ids( visible_count );
        auto p_visible_ids = visible_ids.data();

        transform( [ = ]ARIES_DEVICE( int index )
                {
                    auto i = p_psum[ index ];
                    if ( p_flags[ index ] == 1 )
                    {
                        p_visible_ids[ i ] = 0 - index - 1;
                    }
                }, tupleNum, *ctx );
        ctx->synchronize();
        flags.free();

        auto visibleIds = std::make_shared< AriesDataBuffer >( AriesColumnType(
        { AriesValueType::INT32, 1 }, false, false ), visible_count );
        visibleIds->PrefetchToGpu();
        cudaMemcpy( visibleIds->GetData(), p_visible_ids, visibleIds->GetTotalBytes(), cudaMemcpyDefault );
        return visibleIds;
    }

    AriesIndicesArraySPtr ConvertRowIdToIndices( const AriesDataBufferSPtr& rowIds )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        mem_t< index_t > row_ids( rowIds->GetItemCount() );
        auto p_row_ids = row_ids.data();
        cudaMemcpy( p_row_ids, rowIds->GetData(), rowIds->GetTotalBytes(), cudaMemcpyDefault );

        transform( [ = ]ARIES_DEVICE( int index )
                {
                    p_row_ids[ index ] = 0 - p_row_ids[ index ] - 1;
                }, rowIds->GetItemCount(), *ctx );
        ctx->synchronize();

        auto indices = std::make_shared< AriesIndicesArray >( rowIds->GetItemCount() );
        cudaMemcpy( indices->GetData(), p_row_ids, sizeof(index_t) * rowIds->GetItemCount(), cudaMemcpyDefault );
        return indices;
    }

    AriesDataBufferSPtr CreateDataBufferWithLiteralExpr( const aries_engine::AriesCommonExprUPtr& expr, size_t count )
    {
        ARIES_ASSERT( expr->IsLiteralValue(), "expr should be literal" );
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();

        auto buffer = std::make_shared< AriesDataBuffer >( expr->GetValueType(), count, false );
        buffer->PrefetchToGpu();

        if ( count == 0 )
        {
            return buffer;
        }

        switch( expr->GetType() )
        {
            case AriesExprType::INTEGER:
            {
                if( expr->GetContent().type() == typeid(int8_t) )
                {
                    init_value( buffer->GetData(), count, boost::get< int8_t >( expr->GetContent() ), *ctx );
                }
                else if( expr->GetContent().type() == typeid(int16_t) )
                {
                    init_value( ( int16_t* )buffer->GetData(), count, boost::get< int16_t >( expr->GetContent() ), *ctx );
                }
                else if( expr->GetContent().type() == typeid(int32_t) )
                {
                    init_value( ( int32_t* )buffer->GetData(), count, boost::get< int32_t >( expr->GetContent() ), *ctx );
                }
                else if( expr->GetContent().type() == typeid(int64_t) )
                {
                    init_value( ( int64_t* )buffer->GetData(), count, boost::get< int64_t >( expr->GetContent() ), *ctx );
                }
                break;
            }
            case AriesExprType::STRING:
            {
                auto value = boost::get< std::string >( expr->GetContent() );
                return CreateDataBufferWithValue( value, count );
            }
            case AriesExprType::FLOATING:
            {
                if( expr->GetContent().type() == typeid(float) )
                {
                    init_value( ( float* )buffer->GetData(), count, boost::get< float >( expr->GetContent() ), *ctx );
                }
                else if( expr->GetContent().type() == typeid(double) )
                {
                    init_value( ( double* )buffer->GetData(), count, boost::get< double >( expr->GetContent() ), *ctx );
                }
                else
                {
                    ARIES_ASSERT( 0, "invalid value type for literal expression" );
                }
                break;
            }
            case AriesExprType::DECIMAL:
            {
                init_value( ( aries_acc::Decimal* )buffer->GetData(), count, boost::get< aries_acc::Decimal >( expr->GetContent() ), *ctx );
                break;
            }
            case AriesExprType::DATE:
            {
                init_value( ( aries_acc::AriesDate* )buffer->GetData(), count, boost::get< aries_acc::AriesDate >( expr->GetContent() ), *ctx );
                break;
            }
            case AriesExprType::DATE_TIME:
            {
                init_value( ( aries_acc::AriesDatetime* )buffer->GetData(), count, boost::get< aries_acc::AriesDatetime >( expr->GetContent() ),
                        *ctx );
                break;
            }
            case AriesExprType::TIME:
            {
                init_value( ( aries_acc::AriesTime* )buffer->GetData(), count, boost::get< aries_acc::AriesTime >( expr->GetContent() ), *ctx );
                break;
            }
            case AriesExprType::TIMESTAMP:
            {
                init_value( ( aries_acc::AriesTimestamp* )buffer->GetData(), count, boost::get< aries_acc::AriesTimestamp >( expr->GetContent() ),
                        *ctx );
                break;
            }
            case AriesExprType::NULL_VALUE:
            {
                // ARIES_ASSERT( 0, "not supported yet" );
                auto data_type = buffer->GetDataType();
                data_type.HasNull = true;
                buffer = std::make_shared< AriesDataBuffer >( data_type, count, false );
                for ( int i = 0; i < count; i++ )
                {
                    *( buffer->GetItemDataAt(i) ) = 0;
                }

                break;
            }
            case AriesExprType::TRUE_FALSE:
            {
                init_value( ( bool* )buffer->GetData(), count, boost::get< bool >( expr->GetContent() ), *ctx );
                break;
            }
            case AriesExprType::YEAR:
            {
                init_value( ( aries_acc::AriesYear* )buffer->GetData(), count, boost::get< aries_acc::AriesYear >( expr->GetContent() ), *ctx );
                break;
            }
            default:
                break;
        }
        LOG(INFO)<< "FillWithValue gpu time: " << ctx->timer_end();
        return buffer;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( const std::string& value, size_t count, AriesColumnType type )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int len = value.size();
        len = len > 0 ? len : 1;
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( type, count );
        char* data = ( char* )result->GetData();

        AriesDataBuffer buffer( type, 1 );
        char *p_buffer = ( char* )buffer.GetData();

        char *p = p_buffer;
        if( type.HasNull )
        {
            *p = 1;
            p++;
        }

        if( value.empty() )
        {
            memset( p, 0, len );
        }
        else
        {
            memcpy( p, value.data(), len );
        }

        auto size = buffer.GetItemSizeInBytes();
        result->PrefetchToGpu();
        init_value( data, size, count, p_buffer, *ctx );
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithNull( size_t count, AriesColumnType type )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        auto buffer = std::make_shared< AriesDataBuffer >( type, count );

        auto value_buffer = std::make_shared< AriesDataBuffer >( type, 1 );
        *( value_buffer->GetData() ) = 0;
        buffer->PrefetchToGpu();
        init_value( ( char* )buffer->GetData(), type.GetDataTypeSize(), count, ( char* )value_buffer->GetData(), *ctx );
        return buffer;
    }

    std::pair< AriesDataBufferSPtr, AriesDataBufferSPtr > MakeStringColumnsSameLength( const AriesDataBufferSPtr& leftColumn,
            const AriesDataBufferSPtr& rightColumn )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();
        if( leftType == rightType )
            return
            {   leftColumn, rightColumn};
        ARIES_ASSERT( leftType.HasNull == rightType.HasNull,
                "leftType.HasNull: " + to_string( leftType.HasNull ) + ", rightType.HasNull: " + to_string( rightType.HasNull ) );
        ARIES_ASSERT( leftType.DataType.ValueType == AriesValueType::CHAR && leftType.DataType.ValueType == rightType.DataType.ValueType,
                "leftType.DataType.ValueType: " + GetValueTypeAsString( leftType ) + ", rightType.DataType.ValueType: "
                        + GetValueTypeAsString( rightType ) );
        size_t len = std::max( leftType.GetDataTypeSize(), rightType.GetDataTypeSize() );
        bool bLeftNeedConvert = ( len != leftType.GetDataTypeSize() );
        AriesDataBufferSPtr dst =
                ( bLeftNeedConvert ?
                        rightColumn->CloneWithNoContent( leftColumn->GetItemCount() ) : leftColumn->CloneWithNoContent( rightColumn->GetItemCount() ) );
        AriesDataBufferSPtr src = ( bLeftNeedConvert ? leftColumn : rightColumn );
        cudaMemset( dst->GetData(), 0, dst->GetTotalBytes() );
        extend_data_size( src->GetData(), src->GetItemSizeInBytes(), src->GetItemCount(), dst->GetData(), dst->GetItemSizeInBytes(), 0, *ctx );
        if( bLeftNeedConvert )
            return
            {   dst, rightColumn};
        else
            return
            {   leftColumn, dst};
    }

    void FlipAssociated( AriesInt32ArraySPtr& associated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        flip_flags( associated->GetData(), associated->GetItemCount(), *ctx );
        LOG(INFO)<< "FlipAssociated gpu time: " << ctx->timer_end();
    }

    void FlipFlags( AriesInt8ArraySPtr &flags )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        flip_flags( flags->GetData(), flags->GetItemCount(), *ctx );
    }

    void MergeAssociates( AriesInt32ArraySPtr& dst, const AriesInt32ArraySPtr& src, AriesLogicOpType opType )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ARIES_ASSERT( dst->GetItemCount() == src->GetItemCount(),
                "dst->GetItemCount(): " + to_string( dst->GetItemCount() ) + ", src->GetItemCount(): " + to_string( src->GetItemCount() ) );
        if( dst->GetItemCount() > 0 )
        {
            switch( opType )
            {
                case AriesLogicOpType::AND:
                {
                    merge_flags_and( dst->GetData(), src->GetData(), dst->GetItemCount(), *ctx );
                    break;
                }
                case AriesLogicOpType::OR:
                {
                    merge_flags_or( dst->GetData(), src->GetData(), dst->GetItemCount(), *ctx );
                    break;
                }
                default:
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "logic Op type " + GetAriesLogicOpTypeName( opType ) );
            }
        }
    }

    void FlipAssociated( AriesBoolArraySPtr& associated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        flip_flags( associated->GetData(), associated->GetItemCount(), *ctx );
        LOG(INFO)<< "FlipAssociated gpu time: " << ctx->timer_end();
    }

    void MergeAssociates( AriesBoolArraySPtr& dst, const AriesBoolArraySPtr& src, AriesLogicOpType opType )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        ARIES_ASSERT( dst->GetItemCount() == src->GetItemCount(),
                "dst->GetItemCount(): " + to_string( dst->GetItemCount() ) + ", src->GetItemCount(): " + to_string( src->GetItemCount() ) );
        if( dst->GetItemCount() > 0 )
        {
            switch( opType )
            {
                case AriesLogicOpType::AND:
                {
                    merge_flags_and( dst->GetData(), src->GetData(), dst->GetItemCount(), *ctx );
                    break;
                }
                case AriesLogicOpType::OR:
                {
                    merge_flags_or( dst->GetData(), src->GetData(), dst->GetItemCount(), *ctx );
                    break;
                }
                default:
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "logic Op type " + GetAriesLogicOpTypeName( opType ) );
            }
        }
        LOG(INFO)<< "MergeAssociates gpu time: " << ctx->timer_end();
    }

    AriesInt32ArraySPtr ConvertToInt32Array( const AriesBoolArraySPtr& array )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesInt32ArraySPtr result;
        size_t tupleNum = array->GetItemCount();
        if( tupleNum > 0 )
        {
            result = std::make_shared< AriesInt32Array >( tupleNum );
            convert_ariesbool_to_numeric( array->GetData(), tupleNum, result->GetData(), *ctx );
        }
        LOG(INFO)<< "ConvertToInt32Array gpu time: " << ctx->timer_end();
        return result;
    }

    AriesDataBufferSPtr ConvertToDataBuffer( const AriesBoolArraySPtr& array )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();

        size_t tupleNum = array->GetItemCount();

        AriesColumnType columnType( AriesValueType::INT8, true, false );
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( columnType, tupleNum );
        if( tupleNum > 0 )
        {
            result->PrefetchToGpu();
            convert_ariesbool_to_buf( array->GetData(), tupleNum, ( nullable_type< int8_t >* )result->GetData(), *ctx );
        }
        LOG(INFO)<< "ConvertToDataBuffer gpu time: " << ctx->timer_end();
        return result;
    }

    AriesUInt8ArraySPtr ConvertToUInt8Array( const AriesBoolArraySPtr& array )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesUInt8ArraySPtr result;
        size_t tupleNum = array->GetItemCount();
        if( tupleNum > 0 )
        {
            result = std::make_shared< AriesUInt8Array >( tupleNum );
            convert_ariesbool_to_numeric( array->GetData(), tupleNum, result->GetData(), *ctx );
        }
        printf( "ConvertToUInt8Array gpu time: %3.1f\n", ctx->timer_end() );
        return result;
    }

    int32_t ExclusiveScan( const AriesInt32ArraySPtr& associated, AriesInt32ArraySPtr& outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int32_t total;
        size_t tupleNum = associated->GetItemCount();
        AriesManagedArray< int32_t > sum( 1 );
        outPrefixSum = make_shared< AriesInt32Array >( tupleNum );
        scan( associated->GetData(), tupleNum, outPrefixSum->GetData(), plus_t< int32_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetData()[0];
        return total;
    }

    int64_t ExclusiveScan( const AriesInt32ArraySPtr& associated, AriesInt64ArraySPtr& outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int64_t total;
        size_t tupleNum = associated->GetItemCount();
        AriesInt64Array sum( 1 );
        outPrefixSum = make_shared< AriesInt64Array >( tupleNum );
        int32_t* pAssociated = associated->GetData();
        transform_scan< int64_t >( [=]ARIES_LAMBDA(int index)
                {
                    return pAssociated[ index ];
                }, tupleNum, outPrefixSum->GetData(), plus_t< int64_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetValue( 0 );
        return total;
    }

    int32_t InclusiveScan( const AriesInt32ArraySPtr& associated, AriesInt32ArraySPtr& outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int32_t total;
        size_t tupleNum = associated->GetItemCount();
        AriesManagedArray< int > sum( 1 );
        outPrefixSum = make_shared< AriesInt32Array >( tupleNum );
        scan< scan_type_inc >( associated->GetData(), tupleNum, outPrefixSum->GetData(), plus_t< int32_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetData()[0];
        return total;
    }

    int32_t InclusiveScan( const AriesInt8ArraySPtr& flags, AriesInt32ArraySPtr& outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int32_t total;
        size_t tupleNum = flags->GetItemCount();
        AriesManagedArray< int > sum( 1 );
        outPrefixSum = make_shared< AriesInt32Array >( tupleNum );
        int8_t* pFlags = flags->GetData();

        transform_scan< int32_t, scan_type_inc >( [=]ARIES_LAMBDA(int index)
                {
                    return pFlags[ index ];
                }, tupleNum, outPrefixSum->GetData(), plus_t< int32_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetData()[0];
        return total;
    }

    int32_t ExclusiveScan( const AriesBoolArraySPtr& associated, AriesInt32ArraySPtr& outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int32_t total;
        size_t tupleNum = associated->GetItemCount();
        AriesManagedArray< int > sum( 1 );
        outPrefixSum = make_shared< AriesInt32Array >( tupleNum );
        AriesBool* pAssociated = associated->GetData();
        transform_scan< int32_t >( [=]ARIES_LAMBDA(int index)
                {
                    return pAssociated[ index ].is_true();
                }, tupleNum, outPrefixSum->GetData(), plus_t< int32_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetData()[0];
        return total;
    }

    int32_t ExclusiveScan( const AriesInt8ArraySPtr &flags, AriesInt32ArraySPtr &outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int32_t total;
        size_t tupleNum = flags->GetItemCount();
        AriesManagedArray< int > sum( 1 );
        outPrefixSum = make_shared< AriesInt32Array >( tupleNum );
        int8_t* pFlags = flags->GetData();
        transform_scan< int32_t >( [=]ARIES_LAMBDA(int index)
                {
                    return pFlags[ index ];
                }, tupleNum, outPrefixSum->GetData(), plus_t< int32_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetData()[0];
        return total;
    }

    int32_t ExclusiveScan( const AriesManagedInt8ArraySPtr &flags, AriesInt32ArraySPtr &outPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int32_t total;
        size_t tupleNum = flags->GetItemCount();
        AriesManagedArray< int > sum( 1 );
        outPrefixSum = make_shared< AriesInt32Array >( tupleNum );
        int8_t* pFlags = flags->GetData();
        flags->PrefetchToGpu();
        transform_scan< int32_t >( [=]ARIES_LAMBDA(int index)
                {
                    return pFlags[ index ];
                }, tupleNum, outPrefixSum->GetData(), plus_t< int32_t >(), sum.GetData(), *ctx );
        ctx->synchronize();
        total = sum.GetData()[0];
        return total;
    }

    AriesDataBufferSPtr ConvertStringColumn( const AriesDataBufferSPtr &column, AriesColumnType columnType )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesColumnType myType = column->GetDataType();
        if( myType == columnType && myType.GetDataTypeSize() == columnType.GetDataTypeSize() )
            return column;
        else
        {
            size_t tupleNum = column->GetItemCount();
            AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( columnType, tupleNum );
            if( tupleNum > 0 )
            {
                result->PrefetchToGpu();
                extend_data_size( column->GetData(), column->GetItemSizeInBytes(), tupleNum, result->GetData(), result->GetItemSizeInBytes(),
                        !myType.HasNull && columnType.HasNull, *ctx );
            }
            return result;
        }
    }

    AriesDataBufferSPtr ConvertToNullableType( const AriesDataBufferSPtr& column )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesColumnType columnType = column->GetDataType();
        if( columnType.HasNull )
            return column;

        DataBlockInfo block;
        block.Data = column->GetData();
        block.ElementSize = columnType.GetDataTypeSize();
        size_t tupleNum = column->GetItemCount();

        columnType.HasNull = true;
        AriesDataBufferSPtr newColumn = make_shared< AriesDataBuffer >( columnType, tupleNum );
        newColumn->PrefetchToGpu();

        make_column_nullable( block, tupleNum, newColumn->GetData(), *ctx );
        return newColumn;
    }

    void FillWithValue( AriesInt32ArraySPtr& buffer, int value )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        init_value( buffer->GetData(), buffer->GetItemCount(), value, *ctx );
        LOG( INFO )<< "FillWithValue gpu time: " << ctx->timer_end();
    }

    void FillWithValue( AriesBoolArraySPtr& buffer, AriesBool value )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        init_value( buffer->GetData(), buffer->GetItemCount(), value, *ctx );
        LOG( INFO )<< "FillWithValue gpu time: " << ctx->timer_end();
    }

    void InitSequenceValue( AriesInt32ArraySPtr& buffer, int beginValue )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        init_sequence_begin_with( buffer->GetData(), buffer->GetItemCount(), beginValue, *ctx );
#ifdef ARIES_PROFILE
        LOG( INFO )<< "InitSequenceValue gpu time: " << ctx->timer_end();
#endif
    }

    AriesDataBufferSPtr DivisionInt64( const AriesDataBufferSPtr& dividend, const AriesDataBufferSPtr& divisor )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        ARIES_ASSERT( divisor->GetDataType().DataType.ValueType == AriesValueType::INT64,
                "divisor->GetDataType().DataType.ValueType: " + GetValueTypeAsString( divisor->GetDataType() ) );
        AriesDataBufferSPtr result;
        size_t tupleNum = dividend->GetItemCount();
        if( tupleNum > 0 )
        {
            int64_t* divisor_data = ( int64_t* )divisor->GetData();
            AriesColumnType type = dividend->GetDataType();
            switch( type.DataType.ValueType )
            {
                case AriesValueType::FLOAT:
                case AriesValueType::DOUBLE:
                {
                    result = make_shared< AriesDataBuffer >( type, tupleNum );
                    break;
                }
                case AriesValueType::CHAR:
                case AriesValueType::BOOL:
                case AriesValueType::INT16:
                case AriesValueType::INT8:
                case AriesValueType::UINT16:
                case AriesValueType::UINT8:
                case AriesValueType::COMPACT_DECIMAL:
                {
                    ARIES_ASSERT( 0, "data type: " + GetValueTypeAsString( type ) + " can't be used to division" );
                }
                default:
                {
                    //convert dividend to decimal
                    result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType
                    { AriesValueType::DECIMAL, 1 }, type.HasNull, false ), tupleNum );

                    break;
                }
            }
            if( result )
            {
                result->PrefetchToGpu();
                division_data_int64( dividend->GetData(), type, divisor_data, result->GetData(), tupleNum, *ctx );
            }
        }
        LOG( INFO )<< "DivisionInt64 gpu time: " << ctx->timer_end();
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( int8_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::INT8, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< int8_t >* data = ( nullable_type< int8_t >* )result->GetData();
            init_value( data, count, nullable_type< int8_t >( 1, value ), *ctx );
        }
        else
        {
            int8_t* data = ( int8_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( uint8_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::UINT8, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< uint8_t >* data = ( nullable_type< uint8_t >* )result->GetData();
            init_value( data, count, nullable_type< uint8_t >( 1, value ), *ctx );
        }
        else
        {
            uint8_t* data = ( uint8_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( int16_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::INT16, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< int16_t >* data = ( nullable_type< int16_t >* )result->GetData();
            init_value( data, count, nullable_type< int16_t >( 1, value ), *ctx );
        }
        else
        {
            int16_t* data = ( int16_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( uint16_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::UINT16, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< uint16_t >* data = ( nullable_type< uint16_t >* )result->GetData();
            init_value( data, count, nullable_type< uint16_t >( 1, value ), *ctx );
        }
        else
        {
            uint16_t* data = ( uint16_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( int32_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::INT32, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< int32_t >* data = ( nullable_type< int32_t >* )result->GetData();
            init_value( data, count, nullable_type< int32_t >( 1, value ), *ctx );
        }
        else
        {
            int32_t* data = ( int32_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( uint32_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::UINT32, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< uint32_t >* data = ( nullable_type< uint32_t >* )result->GetData();
            init_value( data, count, nullable_type< uint32_t >( 1, value ), *ctx );
        }
        else
        {
            uint32_t* data = ( uint32_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( int64_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::INT64, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< int64_t >* data = ( nullable_type< int64_t >* )result->GetData();
            init_value( data, count, nullable_type< int64_t >( 1, value ), *ctx );
        }
        else
        {
            int64_t* data = ( int64_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( uint64_t value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::UINT64, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< uint64_t >* data = ( nullable_type< uint64_t >* )result->GetData();
            init_value( data, count, nullable_type< uint64_t >( 1, value ), *ctx );
        }
        else
        {
            uint64_t* data = ( uint64_t* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( float value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::FLOAT, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< float >* data = ( nullable_type< float >* )result->GetData();
            init_value( data, count, nullable_type< float >( 1, value ), *ctx );
        }
        else
        {
            float* data = ( float* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( double value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::DOUBLE, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< double >* data = ( nullable_type< double >* )result->GetData();
            init_value( data, count, nullable_type< double >( 1, value ), *ctx );
        }
        else
        {
            double* data = ( double* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( const Decimal& value, size_t count, bool nullable, bool unique )
    {
        // value 存储的就是 Decimal常量
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        // 申请内存空间
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::DECIMAL, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< Decimal >* data = ( nullable_type< Decimal >* )result->GetData();
            init_value( data, count, nullable_type< Decimal >( 1, value ), *ctx );
        }
        else
        {
            Decimal* data = ( Decimal* )result->GetData();
            // 向内存空间拷贝数据
            init_value( data, count, value, *ctx );
        }
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue(const Decimal& value, size_t count, size_t ariesDecSize, bool nullable, bool unique )
    {
        // value 存储的就是 Decimal常量
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        // 申请内存空间
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::ARIES_DECIMAL, ariesDecSize ), nullable, unique ),
                count );
        result->PrefetchToGpu();

        char* data = (char *)result->GetData();
        init_value( data, ariesDecSize*4+4, count, value, *ctx);

        // // 两种方法比较以下哪种方法更快
        // Decimal* data = ( Decimal* )result->GetData();
        // init_value( data, count, value, *ctx );

        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesDate& value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::DATE, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< AriesDate >* data = ( nullable_type< AriesDate >* )result->GetData();
            init_value( data, count, nullable_type< AriesDate >( 1, value ), *ctx );
        }
        else
        {
            AriesDate* data = ( AriesDate* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesTime& value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::TIME, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< AriesTime >* data = ( nullable_type< AriesTime >* )result->GetData();
            init_value( data, count, nullable_type< AriesTime >( 1, value ), *ctx );
        }
        else
        {
            AriesTime* data = ( AriesTime* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesDatetime& value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >(
                AriesColumnType( AriesDataType( AriesValueType::DATETIME, 1 ), nullable, unique ), count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< AriesDatetime >* data = ( nullable_type< AriesDatetime >* )result->GetData();
            init_value( data, count, nullable_type< AriesDatetime >( 1, value ), *ctx );
        }
        else
        {
            AriesDatetime* data = ( AriesDatetime* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesTimestamp& value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >(
                AriesColumnType( AriesDataType( AriesValueType::TIMESTAMP, 1 ), nullable, unique ), count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< AriesTimestamp >* data = ( nullable_type< AriesTimestamp >* )result->GetData();
            init_value( data, count, nullable_type< AriesTimestamp >( 1, value ), *ctx );
        }
        else
        {
            AriesTimestamp* data = ( AriesTimestamp* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }
    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesYear& value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::YEAR, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< AriesYear >* data = ( nullable_type< AriesYear >* )result->GetData();
            init_value( data, count, nullable_type< AriesYear >( 1, value ), *ctx );
        }
        else
        {
            AriesYear* data = ( AriesYear* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( char value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::CHAR, 1 ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        if( nullable )
        {
            nullable_type< char >* data = ( nullable_type< char >* )result->GetData();
            init_value( data, count, nullable_type< char >( 1, value ), *ctx );
        }
        else
        {
            char* data = ( char* )result->GetData();
            init_value( data, count, value, *ctx );
        }
        return result;
    }

    AriesDataBufferSPtr CreateDataBufferWithValue( const string& value, size_t count, bool nullable, bool unique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        int len = value.size();
        len = len > 0 ? len : 1;
        int realLen = nullable ? len + 1 : len;
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::CHAR, len ), nullable, unique ),
                count );
        result->PrefetchToGpu();
        char* data = ( char* )result->GetData();
        AriesDataBuffer buffer( AriesColumnType( AriesDataType( AriesValueType::CHAR, len ), nullable, unique ), 1 );
        int offset = 0;
        if( nullable )
        {
            *( buffer.GetData() ) = 1;
            offset = 1;
        }
        if( value.empty() )
        {
            *( buffer.GetData() + offset ) = 0;
        }
        else
        {
            memcpy( buffer.GetData() + offset, value.data(), len );
        }

        init_value( data, realLen, count, ( char* )buffer.GetData(), *ctx );
        return result;
    }

    AriesDataBufferSPtr CreateNullValueDataBuffer( size_t count )
    {
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( AriesDataType( AriesValueType::UNKNOWN, 1 ), false, false ),
                count );
        return result;
    }

    static AriesDataBufferSPtr ZipColumnDataByCpu( const std::vector< aries_engine::AriesColumnSPtr >& columns )    
    {
        struct ColumnIter
        {
            int64_t TotalCount;
            size_t PerItemSizeInBytes;
            vector<AriesDataBufferSPtr> Blocks;
            vector<int64_t> PrefixSum;
            int8_t* operator[]( int index ) const
            {
                auto it = std::upper_bound( PrefixSum.cbegin(), PrefixSum.cend(), index );
                int blockIndex = ( it - PrefixSum.cbegin() ) - 1;
                int offsetInBlock = index - PrefixSum[ blockIndex ];
                return Blocks[ blockIndex ]->GetItemDataAt( offsetInBlock );
            }
        };

        assert( !columns.empty() );
        size_t tupleNum = columns[ 0 ]->GetRowCount();
        size_t totalBytes = 0;
        vector< ColumnIter > columnIters;
        for( const auto& col : columns )
        {
            col->PrefetchDataToCpu();
            ColumnIter iter;
            assert( tupleNum == col->GetRowCount() );
            iter.TotalCount = tupleNum;
            iter.Blocks = col->GetDataBuffers();
            iter.PrefixSum = col->GetBlockSizePsumArray();
            iter.PerItemSizeInBytes = col->GetColumnType().GetDataTypeSize();
            columnIters.push_back( iter );
            totalBytes += iter.PerItemSizeInBytes;
        }

        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >(
            AriesColumnType( AriesDataType( AriesValueType::CHAR, totalBytes ), false, false ), tupleNum );
        result->PrefetchToCpu();
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

            workThreads.push_back( std::async(std::launch::async,
            [ & ]( int64_t start, int64_t count ) 
            { 
                int8_t* pOutput = result->GetItemDataAt( start );
                int64_t end = start + count;
                for( int64_t i = start; i < end; ++i )
                {
                    for( const auto& it : columnIters )
                    {
                        memcpy( pOutput, it[ i ], it.PerItemSizeInBytes );
                        pOutput += it.PerItemSizeInBytes;
                    }
                }
            }, offset, itemCount ) );
        }

        for (auto &t : workThreads)
            t.wait();
        return result;
    }

    static AriesDataBufferSPtr ZipColumnDataByGpu( const std::vector< aries_engine::AriesColumnSPtr >& columns )
    {
        assert( !columns.empty() );
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        size_t tupleNum = columns[0]->GetRowCount();
        size_t totalBytes = 0;
        vector< AriesManagedArraySPtr< int8_t* > > columnBuffers;

        AriesManagedArray< int8_t** > allBuffers( columns.size() );
        AriesManagedArray< size_t > dataTypeSizes( columns.size() );
        int index = 0;
        for( const auto& col : columns )
        {
            assert( tupleNum == col->GetRowCount() );
            vector< AriesDataBufferSPtr > dataBuffers = col->GetDataBuffers();
            AriesManagedArraySPtr< int8_t* > dataBlocks = std::make_shared< AriesManagedArray< int8_t* > >( dataBuffers.size() );
            int i = 0;
            for( const auto& block : dataBuffers )
            {
                block->PrefetchToGpu();
                ( *dataBlocks )[i++] = block->GetData();
            }
            dataBlocks->PrefetchToGpu();
            columnBuffers.push_back( dataBlocks );
            dataTypeSizes[index] = col->GetColumnType().GetDataTypeSize();
            allBuffers[index] = dataBlocks->GetData();
            totalBytes += col->GetColumnType().GetDataTypeSize();
            ++index;
        }
        AriesInt64ArraySPtr prefixSumArray = aries_engine::GetPrefixSumOfBlockSize( columns[0]->GetBlockSizePsumArray() );

        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >(
                AriesColumnType( AriesDataType( AriesValueType::CHAR, totalBytes ), false, false ), tupleNum );

        int8_t*** inputs = allBuffers.GetData();
        size_t* dataItemSizes = dataTypeSizes.GetData();
        int64_t* dataBlockSizePrefixSum = prefixSumArray->GetData();
        int32_t dataBlockCount = columns[0]->GetDataBuffers().size();
        int32_t inputCount = columns.size();
        int8_t* output = result->GetData();
        allBuffers.PrefetchToGpu();
        dataTypeSizes.PrefetchToGpu();
        result->PrefetchToGpu();
        transform( [ = ]ARIES_DEVICE( int index )
                {
                    int dataBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>( dataBlockSizePrefixSum, dataBlockCount, index ) - 1;
                    int itemIndexInBlock = index - dataBlockSizePrefixSum[ dataBlockIndex ];
                    int8_t* outputItem = output + index * totalBytes;
                    size_t itemSize;
                    for( int i = 0; i < inputCount; ++i )
                    {
                        itemSize = dataItemSizes[ i ];
                        memcpy( outputItem, inputs[ i ][ dataBlockIndex ] + itemIndexInBlock * itemSize, itemSize );
                        outputItem += itemSize;
                    }
                }, tupleNum, *ctx );

        ctx->synchronize();
        return result;
    }

    AriesDataBufferSPtr ZipColumnData( const std::vector< aries_engine::AriesColumnSPtr >& columns )
    {
        return ZipColumnDataByGpu( columns );
    }

    AriesTableKeysDataSPtr GenerateTableKeys( const std::vector< aries_engine::AriesColumnSPtr >& columns, bool checkDuplicate )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesDataBufferSPtr data = ZipColumnData( columns );

        string keyBuffer;
        vector< aries::AriesSimpleItemContainer< RowPos > > tupleLocations;

        size_t tupleNum = data->GetItemCount();
        if( tupleNum > 0 )
        {
            AriesInt32ArraySPtr associated = SortDataForTableKeys( data );
            AriesInt8ArraySPtr groupBoundFlags = make_shared< AriesInt8Array >( tupleNum );
            groupBoundFlags->SetValue( 0, 0 );
            int8_t* flags = groupBoundFlags->GetData();
            size_t len = data->GetItemSizeInBytes();
            const char* pData = ( const char* )data->GetData();
            if( tupleNum > 1 )
            {
                transform( [ = ]ARIES_DEVICE( int index )
                        {
                            size_t offset = index * len;
                            flags[ index + 1 ] = str_not_equal_to_t( pData + offset, pData + offset + len, len );
                        }, tupleNum - 1, *ctx );
                ctx->synchronize();
            }
            AriesInt32ArraySPtr psum;
            int32_t groupCount = InclusiveScan( groupBoundFlags, psum ) + 1;

            std::unique_ptr< int32_t[] > pAssociated( new int32_t[tupleNum] );
            ARIES_CALL_CUDA_API(
                    cudaMemcpy( pAssociated.get(), associated->GetData(), associated->GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
            int MAX_COUNT_SINGLE_THREAD = 100;
            if( checkDuplicate )
            {
                if( groupCount == tupleNum )
                {
                    // for primary key
                    keyBuffer.resize( data->GetTotalBytes() );
                    cudaMemcpy( ( char* )keyBuffer.data(), data->GetData(), data->GetTotalBytes(), cudaMemcpyDeviceToHost );
                    tupleLocations.resize( groupCount );
                    if( groupCount > MAX_COUNT_SINGLE_THREAD )
                    {
                        int64_t threadNum = thread::hardware_concurrency();
                        int64_t blockSize = groupCount / threadNum;
                        int64_t extraSize = groupCount % threadNum;
                        vector< future< void > > allThreads;
                        int startIndex;

                        for( int i = 0; i < threadNum; ++i )
                        {
                            startIndex = blockSize * i;
                            if( i == threadNum - 1 )
                                allThreads.push_back( std::async( std::launch::async, [&]( int index )
                                {
                                    for( int n = index; n < index + blockSize + extraSize; ++n )
                                    {
                                        tupleLocations[n].push_back( pAssociated[n] );
                                    }
                                }, startIndex ) );
                            else
                                allThreads.push_back( std::async( std::launch::async, [&]( int index )
                                {
                                    for( int n = index; n < index + blockSize; ++n )
                                    {
                                        tupleLocations[n].push_back( pAssociated[n] );
                                    }
                                }, startIndex ) );
                        }
                        for( auto& t : allThreads )
                            t.wait();
                    }
                    else
                    {
                        for( int i = 0; i < groupCount; ++i )
                            tupleLocations[i].push_back( pAssociated[i] );
                    }
                }
            }
            else
            {
                AriesDataBufferSPtr resultBuffer = data->CloneWithNoContent( groupCount, ACTIVE_DEVICE_ID );
                gather_filtered_data( ( const char* )data->GetData(), len, tupleNum, flags, psum->GetData(), ( char* )resultBuffer->GetData(), *ctx );
                keyBuffer.resize( resultBuffer->GetTotalBytes() );
                cudaMemcpy( ( char* )keyBuffer.data(), resultBuffer->GetData(), resultBuffer->GetTotalBytes(), cudaMemcpyDeviceToHost );

                std::unique_ptr< int32_t[] > psumData( new int32_t[tupleNum] );
                ARIES_CALL_CUDA_API( cudaMemcpy( psumData.get(), psum->GetData(), psum->GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
                tupleLocations.resize( groupCount );
                if( groupCount > MAX_COUNT_SINGLE_THREAD )
                {
                    int64_t threadNum = thread::hardware_concurrency();
                    int64_t blockSize = groupCount / threadNum;
                    int64_t extraSize = groupCount % threadNum;
                    vector< future< void > > allThreads;
                    int startIndex;

                    for( int i = 0; i < threadNum; ++i )
                    {
                        startIndex = blockSize * i;
                        if( i == threadNum - 1 )
                            allThreads.push_back( std::async( std::launch::async, [&]( int index )
                            {
                                for( int n = index; n < index + blockSize + extraSize; ++n )
                                {
                                    tupleLocations[psumData[n]].push_back( pAssociated[n] );
                                }
                            }, startIndex ) );
                        else
                            allThreads.push_back( std::async( std::launch::async, [&]( int index )
                            {
                                for( int n = index; n < index + blockSize; ++n )
                                {
                                    tupleLocations[psumData[n]].push_back( pAssociated[n] );
                                }
                            }, startIndex ) );
                    }
                    for( auto& t : allThreads )
                        t.wait();
                }
                else
                {
                    for( int i = 0; i < tupleNum; ++i )
                        tupleLocations[psumData[i]].push_back( pAssociated[i] );
                }
            }
        }

        return std::make_shared< AriesTableKeysData >( std::move( keyBuffer ), std::move( tupleLocations ) );
    }

    std::pair< bool, AriesInt32ArraySPtr > SortAndVerifyUniqueTableKeys( AriesDataBufferSPtr& data )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();

        size_t tupleNum = data->GetItemCount();
        assert( tupleNum > 0 );
        AriesInt32ArraySPtr result = make_shared< AriesInt32Array >();

        result->AllocArray( tupleNum );
        InitSequenceValue( result );
        sort_column_data( data->GetData(), data->GetDataType(), tupleNum, AriesOrderByType::ASC, result->GetData(), *ctx, true, true );

        AriesInt8ArraySPtr groupBoundFlags = make_shared< AriesInt8Array >( tupleNum );
        groupBoundFlags->SetValue( 0, 0 );
        int8_t* flags = groupBoundFlags->GetData();
        size_t len = data->GetItemSizeInBytes();
        const char* pData = ( const char* )data->GetData();
        if( tupleNum > 1 )
        {
            transform( [ = ]ARIES_DEVICE( int index )
                    {
                        size_t offset = index * len;
                        flags[ index + 1 ] = str_not_equal_to_t( pData + offset, pData + offset + len, len );
                    }, tupleNum - 1, *ctx );
            ctx->synchronize();
        }
        AriesInt32ArraySPtr psum;
        int32_t groupCount = InclusiveScan( groupBoundFlags, psum ) + 1;
        if( groupCount == tupleNum )
            return { true, result };
        else 
            return { false, nullptr };
    }

END_ARIES_ACC_NAMESPACE
