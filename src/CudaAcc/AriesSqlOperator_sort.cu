#include "AriesSqlOperator_sort.h"
#include "sort_column.h"
#include "AriesEngineAlgorithm.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

    AriesDataBufferSPtr SortData( const AriesDataBufferSPtr& data, AriesOrderByType order )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        size_t tupleNum = data->GetItemCount();
        AriesDataBufferSPtr result = data->Clone();
        if( tupleNum > 0 )
        {
            AriesInt32ArraySPtr associated = make_shared< AriesInt32Array >( tupleNum );
            init_sequence( associated->GetData(), tupleNum, *ctx );
            sort_column_data( result->GetData(), data->GetDataType(), tupleNum, order, associated->GetData(), *ctx, true, true );
        }
        LOG( INFO )<< "SortData gpu time: " << ctx->timer_end();
        return result;
    }

    std::pair< AriesDataBufferSPtr, AriesInt32ArraySPtr > SortOneColumn( const AriesDataBufferSPtr& data, AriesOrderByType order )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        size_t tupleNum = data->GetItemCount();
        AriesInt32ArraySPtr associated;
        AriesDataBufferSPtr buffer = data->Clone();
        if( tupleNum > 0 )
        {
            associated = make_shared< AriesInt32Array >( tupleNum );
            init_sequence( associated->GetData(), tupleNum, *ctx );
            sort_column_data( buffer->GetData(), buffer->GetDataType(), tupleNum, order, associated->GetData(), *ctx, true, true );
        }
        LOG( INFO )<< "SortOneColumn gpu time: " << ctx->timer_end();
        return
        {   buffer, associated};
    }

    AriesInt32ArraySPtr SortColumns( const std::vector< AriesDataBufferSPtr >& orderByColumns, const std::vector< AriesOrderByType >& orders,
            bool nullSmaller )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        ARIES_ASSERT( !orderByColumns.empty() && !orders.empty(),
                "orderByColumns.empty(): " + to_string( orderByColumns.empty() ) + ", orders.empty(): " + to_string( orders.empty() ) );

        int32_t tupleNum = orderByColumns[0]->GetItemCount();
        AriesInt32ArraySPtr associated = make_shared< AriesInt32Array >( tupleNum );
        init_sequence( associated->GetData(), tupleNum, *ctx );

        size_t columnCount = orderByColumns.size();
        int8_t *inputData;
        AriesColumnType dataType;
        for( int i = columnCount - 1; i >= 0; --i )
        {
            auto column = orderByColumns[i];
            inputData = column->GetData();
            dataType = column->GetDataType();
            column->PrefetchToGpu();
            if( i == columnCount - 1 )
            { // 第一顺序被排序的列
                sort_column_data( inputData, dataType, tupleNum, orders[i], associated->GetData(), *ctx, nullSmaller, false );
            }
            else
            {
                AriesDataBuffer tmp( dataType, tupleNum );
                tmp.PrefetchToGpu();
                shuffle_column_data( inputData, dataType, tupleNum, associated->GetData(), tmp.GetData(), *ctx );
                sort_column_data( tmp.GetData(), dataType, tupleNum, orders[i], associated->GetData(), *ctx, nullSmaller, false );
            }
        }
#ifdef ARIES_PROFILE
        LOG(INFO)<< "SortColumns gpu time: " << ctx->timer_end();
#endif
        return associated;
    }

    void SortColumn( AriesDataBufferSPtr &column, const AriesOrderByType &order, AriesInt32ArraySPtr& associated, bool nullSmaller )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        column->PrefetchToGpu();
        associated->PrefetchToGpu();
        sort_column_data( column->GetData(), column->GetDataType(), column->GetItemCount(), order, associated->GetData(), *ctx, nullSmaller, true );
#ifdef ARIES_PROFILE
        LOG( INFO )<< "SortColumn gpu time: " << ctx->timer_end();
#endif
    }

    AriesDataBufferSPtr SortColumn( const aries_engine::AriesColumnReferenceSPtr &columnRef, const AriesOrderByType &order,
            AriesInt32ArraySPtr& associated, bool nullSmaller )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesDataBufferSPtr result;
        auto refferedColumn = columnRef->GetReferredColumn();
        auto colEncodeType = refferedColumn->GetEncodeType();
        ARIES_ASSERT( colEncodeType == EncodeType::NONE,
                "SortColumn unmaterialized, column encode type not supported: " + std::to_string( ( int )colEncodeType ) );

        AriesColumnSPtr column = std::dynamic_pointer_cast< AriesColumn >( refferedColumn );
        vector< AriesDataBufferSPtr > dataBuffers = column->GetDataBuffers();
        AriesManagedArray< int8_t* > dataBlocks( dataBuffers.size() );
        int i = 0;
        for( const auto& block : dataBuffers )
            dataBlocks[i++] = block->GetData();

        AriesColumnType columnType = column->GetColumnType();
        AriesIndicesSPtr indices = columnRef->GetIndices();
        AriesIndicesArraySPtr indicesArray = indices->GetIndices();
        AriesInt64ArraySPtr prefixSumArray = aries_engine::GetPrefixSumOfBlockSize( column->GetBlockSizePsumArray() );
        dataBlocks.PrefetchToGpu();
        result = sort_column_data( ( int8_t** )dataBlocks.GetData(), prefixSumArray->GetData(), dataBuffers.size(), columnType,
                indicesArray->GetData(), indicesArray->GetItemCount(), indices->HasNull(), order, associated->GetData(), *ctx, nullSmaller );

        LOG( INFO )<<"SortColumn unmaterialized gpu time: " << ctx->timer_end();
        return result;
    }

    AriesDataBufferSPtr SortColumn( const aries_engine::AriesColumnSPtr &column, const AriesOrderByType &order, AriesInt32ArraySPtr& associated,
            bool nullSmaller )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesDataBufferSPtr result;
        vector< AriesDataBufferSPtr > dataBuffers = column->GetDataBuffers();
        AriesManagedArray< int8_t* > dataBlocks( dataBuffers.size() );
        int i = 0;
        for( const auto& block : dataBuffers )
            dataBlocks[i++] = block->GetData();

        AriesColumnType columnType = column->GetColumnType();
        AriesInt64ArraySPtr prefixSumArray = aries_engine::GetPrefixSumOfBlockSize( column->GetBlockSizePsumArray() );
        dataBlocks.PrefetchToGpu();
        result = sort_column_data( ( int8_t** )dataBlocks.GetData(), prefixSumArray->GetData(), dataBuffers.size(), columnType, column->GetRowCount(),
                order, associated->GetData(), *ctx, nullSmaller );

        LOG( INFO )<<"SortColumn materialized gpu time: " << ctx->timer_end();
        return result;
    }

    AriesInt32ArraySPtr SortDataForTableKeys( AriesDataBufferSPtr& data )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        size_t tupleNum = data->GetItemCount();
        AriesInt32ArraySPtr result = make_shared< AriesInt32Array >();
        if( tupleNum > 0 )
        {
            result->AllocArray( tupleNum );
            int32_t* pAssociated = result->GetData();
            transform( [ = ]ARIES_DEVICE( int index )
                    {
                        pAssociated[ index ] = -index - 1;
                    }, tupleNum, *ctx );
            ctx->synchronize();
            sort_column_data( data->GetData(), data->GetDataType(), tupleNum, AriesOrderByType::ASC, pAssociated, *ctx, true, true );
        }
        LOG( INFO )<< "SortDataForTableKeys gpu time: " << ctx->timer_end();
        return result;
    }
END_ARIES_ACC_NAMESPACE
