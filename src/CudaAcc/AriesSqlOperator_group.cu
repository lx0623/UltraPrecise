#include "AriesSqlOperator_group.h"
#include "AriesSqlOperator_sort.h"
#include "AriesSqlOperator_helper.h"
#include "AriesEngineAlgorithm.h"
// #include "AriesDecimalAlgorithm.h"
#include "sort_column.h"
#include "AriesEngine/cpu_algorithm.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

// the outGroups is like segment index for AggregateColumnData to segreduce. which is different with old version!!!
    int32_t GroupColumns( const std::vector< AriesDataBufferSPtr >& groupByColumns, AriesInt32ArraySPtr& outAssociated,
            AriesInt32ArraySPtr& outGroups, bool isUnique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();

        ARIES_ASSERT( !groupByColumns.empty(), "groupByColumns is empty" );
        int32_t groupCount = 0;
        if( isUnique )
        {
            groupCount = groupByColumns[0]->GetItemCount();
            outAssociated = make_shared< AriesInt32Array >( groupCount );
            outGroups = make_shared< AriesInt32Array >( groupCount );
            init_sequence( outAssociated->GetData(), groupCount, *ctx );
            init_sequence( outGroups->GetData(), groupCount, *ctx );
        }
        else
        {
            size_t columnCount = groupByColumns.size();
            vector< AriesOrderByType > orders;
            for( int i = 0; i < columnCount; ++i )
                orders.push_back( AriesOrderByType::ASC );

            outAssociated = SortColumns( groupByColumns, orders );
            size_t tupleNum = outAssociated->GetItemCount();
            AriesInt32ArraySPtr groupBoundFlags = make_shared< AriesInt32Array >( tupleNum );
            AriesInt32ArraySPtr psum;
            groupBoundFlags->SetValue(0,0);
            find_group_bounds( groupByColumns, outAssociated->GetData(), groupBoundFlags->GetData(), *ctx );
            groupCount = InclusiveScan( groupBoundFlags, psum ) + 1;
            outGroups = make_shared< AriesInt32Array >( groupCount );
            outGroups->SetValue( 0, 0 );
            AriesInt32ArraySPtr indices = make_shared< AriesInt32Array >( tupleNum );
            init_sequence( indices->GetData(), tupleNum, *ctx );
            gather_filtered_data( indices->GetData(), tupleNum, groupBoundFlags->GetData(), psum->GetData(), outGroups->GetData(), *ctx );
        }
#ifdef ARIES_PROFILE
        LOG( INFO )<< "GroupColumns gpu time: " << ctx->timer_end();
#endif
        return groupCount;
    }

    int32_t GroupColumnForSelfJoin( const AriesDataBufferSPtr &groupByColumn, AriesInt32ArraySPtr &outAssociated, AriesInt32ArraySPtr &outGroups,
            bool isUnique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();

        ARIES_ASSERT( groupByColumn, "groupByColumns is empty" );
        int32_t groupCount = 0;
        if( isUnique )
        {
            groupCount = groupByColumn->GetItemCount();
            outAssociated = make_shared< AriesInt32Array >( groupCount );
            outGroups = make_shared< AriesInt32Array >( groupCount );
            init_sequence( outAssociated->GetData(), groupCount, *ctx );
            init_sequence( outGroups->GetData(), groupCount, *ctx );
        }
        else
        {
            size_t tupleNum = groupByColumn->GetItemCount();
            outAssociated = std::make_shared< AriesInt32Array >( tupleNum );
            init_sequence( outAssociated->GetData(), tupleNum, *ctx );
            //ctx->timer_begin();
            sort_column_data( groupByColumn->GetData(), groupByColumn->GetDataType(), tupleNum, AriesOrderByType::ASC, outAssociated->GetData(), *ctx,
                    false );
            //cout << "sort_column_data gpu time: " << ctx->timer_end()<<endl;
            AriesInt32ArraySPtr groupBoundFlags = make_shared< AriesInt32Array >( tupleNum );
            AriesInt32ArraySPtr psum;
            vector< AriesDataBufferSPtr > groupByColumns;
            groupByColumns.push_back( groupByColumn );
            //ctx->timer_begin();
            groupBoundFlags->SetValue(0,0);
            find_group_bounds( groupByColumns, outAssociated->GetData(), groupBoundFlags->GetData(), *ctx );
            //cout << "find_group_bounds gpu time: " << ctx->timer_end()<<endl;

            //ctx->timer_begin();
            groupCount = InclusiveScan( groupBoundFlags, psum ) + 1;
            //cout << "InclusiveScan gpu time: " << ctx->timer_end()<<endl;
            outGroups = make_shared< AriesInt32Array >( groupCount );
            outGroups->SetValue( 0, 0 );
            AriesInt32ArraySPtr indices = make_shared< AriesInt32Array >( tupleNum );
            init_sequence( indices->GetData(), tupleNum, *ctx );
            //ctx->timer_begin();
            gather_filtered_data( indices->GetData(), tupleNum, groupBoundFlags->GetData(), psum->GetData(), outGroups->GetData(), *ctx );
            //cout << "gather_filtered_data gpu time: " << ctx->timer_end()<<endl;
        }
        LOG( INFO )<< "GroupColumnForSelfJoin gpu time: " << ctx->timer_end();
        return groupCount;
    }

    /*
    int64_t GetSelfJoinPairCount( const AriesInt32ArraySPtr& groups, size_t tupleNum, AriesInt64ArraySPtr& expandedGroupsPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        managed_mem_t< int64_t > joinCount( 1, *ctx );
        int64_t* pResult = joinCount.data();

        size_t groupCount = groups->GetItemCount();
        int32_t* pGroups = groups->GetData();
        expandedGroupsPrefixSum = std::make_shared< AriesInt64Array >( groupCount + 1 );
        int64_t* pData = expandedGroupsPrefixSum->GetData();

        transform_scan< int64_t >( [=]ARIES_LAMBDA(int index)
                {
                    int32_t groupSize = pGroups[ index + 1 ] - pGroups[ index ];
                    return groupSize * groupSize;
                }, groupCount - 1, pData, plus_t< int64_t >(), pResult, *ctx );

        ctx->synchronize();

        int32_t tmpGroupValue = groups->GetValue( groupCount - 1 );
        int32_t groupSize = tmpGroupValue - groups->GetValue( groupCount - 2 );
        int32_t expandedSize = groupSize * groupSize;

        int64_t tmpPrefixSumValue = expandedSize + expandedGroupsPrefixSum->GetValue( groupCount - 2 );
        expandedGroupsPrefixSum->SetValue( tmpPrefixSumValue, groupCount - 1 );

        groupSize = tupleNum - tmpGroupValue;
        expandedSize = groupSize * groupSize;
        int64_t result = expandedSize + tmpPrefixSumValue;
        expandedGroupsPrefixSum->SetValue( result, groupCount );
        return result;
    }
    */

    int32_t GetSelfJoinGroupSizePrefixSum( const AriesInt32ArraySPtr& groups, size_t tupleNum, AriesInt32ArraySPtr& groupsPrefixSum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        managed_mem_t< int32_t > joinCount( 1, *ctx );
        int32_t* pResult = joinCount.data();

        size_t groupCount = groups->GetItemCount();
        int32_t* pGroups = groups->GetData();
        groupsPrefixSum = std::make_shared< AriesInt32Array >( groupCount + 1 );
        if ( 1 == groupCount )
        {
            groupsPrefixSum->SetValue( 0, 0 );
            groupsPrefixSum->SetValue( tupleNum, 1 );
            return tupleNum;
        }
        int32_t* pData = groupsPrefixSum->GetData();

        transform_scan< int32_t >( [=]ARIES_LAMBDA(int index)
                {
                    return pGroups[ index + 1 ] - pGroups[ index ];
                }, groupCount - 1, pData, plus_t< int32_t >(), pResult, *ctx );

        ctx->synchronize();
        int32_t tmpGroupValue = groups->GetValue( groupCount - 1 );
        int32_t groupSize = tmpGroupValue - groups->GetValue( groupCount - 2 );

        int64_t tmpPrefixSumValue = groupSize + groupsPrefixSum->GetValue( groupCount - 2 );
        groupsPrefixSum->SetValue( tmpPrefixSumValue, groupCount - 1 );
        groupSize = tupleNum - tmpGroupValue;

        int64_t result = groupSize + tmpPrefixSumValue;
        groupsPrefixSum->SetValue( result, groupCount );

        return result;
    }

    int32_t GroupColumns( const std::vector< AriesDataBufferSPtr > &groupByColumns, AriesInt32ArraySPtr &outAssociated,
            AriesInt32ArraySPtr &outGroups, AriesInt32ArraySPtr &outGroupFlags, bool isUnique )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();

        ARIES_ASSERT( !groupByColumns.empty(), "groupByColumns is empty" );
        int32_t groupCount = 0;
        if( isUnique )
        {
            groupCount = groupByColumns[0]->GetItemCount();
            outAssociated = make_shared< AriesInt32Array >( groupCount );
            outGroups = make_shared< AriesInt32Array >( groupCount );
            init_sequence( outAssociated->GetData(), groupCount, *ctx );
            init_sequence( outGroups->GetData(), groupCount, *ctx );
        }
        else
        {
            size_t columnCount = groupByColumns.size();
            vector< AriesOrderByType > orders;
            for( int i = 0; i < columnCount; ++i )
                orders.push_back( AriesOrderByType::ASC );

            outAssociated = SortColumns( groupByColumns, orders );
            size_t tupleNum = outAssociated->GetItemCount();
            AriesInt32ArraySPtr groupBoundFlags = make_shared< AriesInt32Array >( tupleNum );
            groupBoundFlags->SetValue( 0, 0 );
            AriesInt32ArraySPtr psum;
            find_group_bounds( groupByColumns, outAssociated->GetData(), groupBoundFlags->GetData(), *ctx );
            groupCount = InclusiveScan( groupBoundFlags, psum ) + 1;
            outGroupFlags = psum;
            outGroups = make_shared< AriesInt32Array >( groupCount );
            outGroups->SetValue( 0, 0 );
            AriesInt32ArraySPtr indices = make_shared< AriesInt32Array >( tupleNum );
            init_sequence( indices->GetData(), tupleNum, *ctx );
            gather_filtered_data( indices->GetData(), tupleNum, groupBoundFlags->GetData(), psum->GetData(), outGroups->GetData(), *ctx );
        }
#ifdef ARIES_PROFILE
        LOG( INFO )<< "GroupColumns gpu time: " << ctx->timer_end();
#endif
        return groupCount;
    }

    AriesDataBufferSPtr GetItemCountInGroups( const AriesInt32ArraySPtr &groups, size_t tupleNum )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();

        size_t groupCount = groups->GetItemCount();
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( AriesColumnType
        {
        { AriesValueType::INT64, 1 }, false, false }, groupCount );
        result->PrefetchToGpu();
        int64_t* pResult = ( int64_t* )result->GetData();
        int32_t* pGroups = groups->GetData();
        transform( [=]ARIES_LAMBDA( int index )
                {
                    pResult[index] = pGroups[ index + 1 ] - pGroups[ index ];
                }, groupCount - 1, *ctx );
        ctx->synchronize();
        pResult[groupCount - 1] = tupleNum - groups->GetValue( groupCount - 1 );

#ifdef ARIES_PROFILE
        LOG( INFO )<< "GetItemCountInGroups gpu time: " << ctx->timer_end();
#endif
        return result;
    }

    std::vector< AriesDataBufferSPtr > GatherGroupedColumnData( const std::vector< AriesDataBufferSPtr >& columns,
            const AriesInt32ArraySPtr& associated, const AriesInt32ArraySPtr& groups )
    {
        ARIES_ASSERT( !columns.empty() && columns[0]->GetItemCount() == associated->GetItemCount(),
                "columns.empty(): " + to_string( columns.empty() )
                        + ( columns.empty() ?
                                "" :
                                "columns[0]->GetItemCount(): " + to_string( columns[0]->GetItemCount() ) + "associated->GetItemCount(): "
                                        + to_string( associated->GetItemCount() ) ) );
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        std::vector< AriesDataBufferSPtr > result;
        int count = columns.size();
        if( count > 0 )
        {
            AriesManagedArray< DataBlockInfo > blocks( count );
            AriesManagedArray< int8_t* > outputs( count );
            DataBlockInfo *block = blocks.GetData();
            int8_t** output = outputs.GetData();
            size_t tupleNum = groups->GetItemCount();

            size_t usage = associated->GetTotalBytes() + groups->GetTotalBytes();
            for( const auto& col : columns )
            {
                usage += col->GetItemSizeInBytes() * tupleNum + col->GetTotalBytes();
            }

            if ( usage < AriesDeviceProperty::GetInstance().GetMemoryCapacity() * 0.8 )
            {
                for( const auto& col : columns )
                {
                    AriesColumnType type = col->GetDataType();
                    block->Data = col->GetData();
                    block->ElementSize = type.GetDataTypeSize();
                    block->Offset = 0;

                    AriesDataBufferSPtr newColumn = make_shared< AriesDataBuffer >( type, tupleNum );
                    newColumn->PrefetchToGpu();
                    result.push_back( newColumn );
                    *output = newColumn->GetData();
                    ++output;
                    ++block;
                }
                blocks.PrefetchToGpu();
                outputs.PrefetchToGpu();
                gather_group_data( blocks.GetData(), count, associated->GetData(), groups->GetData(), tupleNum, outputs.GetData(), *ctx );
            }
            else
            {
                for( const auto& col : columns )
                {
                    AriesDataBufferSPtr newColumn = make_shared< AriesDataBuffer >( col->GetDataType(), tupleNum );
                    result.push_back( newColumn );
                    newColumn->PrefetchToCpu();

                    associated->PrefetchToCpu();
                    groups->PrefetchToCpu();
                    col->PrefetchToCpu();

                    ::GatherGroupDataByCPU(
                        col->GetData(),
                        col->GetItemSizeInBytes(),
                        col->GetItemCount(),
                        associated->GetData(),
                        groups->GetData(),
                        tupleNum,
                        newColumn->GetData() );
                }
            }
        }
#ifdef ARIES_PROFILE
        LOG( INFO )<< "GatherGroupedColumnData gpu time: " << ctx->timer_end();
#endif
        return result;
    }

    AriesDataBufferSPtr CombineGroupbyKeys( const std::vector< AriesDataBufferSPtr >& groupByColumns, const AriesInt32ArraySPtr& associated,
            const AriesInt32ArraySPtr& groups )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        ARIES_ASSERT( !groupByColumns.empty() && groupByColumns[0]->GetItemCount() == associated->GetItemCount(),
                "groupByColumns.empty(): " + to_string( groupByColumns.empty() )
                        + ( groupByColumns.empty() ?
                                "" :
                                "groupByColumns[0]->GetItemCount(): " + to_string( groupByColumns[0]->GetItemCount() )
                                        + "associated->GetItemCount(): " + to_string( associated->GetItemCount() ) ) );

        AriesDataBufferSPtr result;
        int32_t combinedSize = 0;
        size_t keyCount = groups->GetItemCount();
        size_t count = groupByColumns.size();

        AriesManagedArray< DataBlockInfo > blocks( count );
        DataBlockInfo *block = blocks.GetData();

        for( const auto& col : groupByColumns )
        {
            block->Data = col->GetData();
            block->ElementSize = col->GetDataType().GetDataTypeSize();
            block->Offset = col->GetDataType().HasNull;
            combinedSize += block->ElementSize;

            ++block;
        }

        result = make_shared< AriesDataBuffer >( AriesColumnType
        {
        { AriesValueType::CHAR, combinedSize }, false, false }, keyCount );
        result->PrefetchToGpu();
        blocks.PrefetchToGpu();
        combine_keys( blocks.GetData(), count, associated->GetData(), groups->GetData(), keyCount, result->GetData(), combinedSize, *ctx );
        LOG( INFO )<< "CombineGroupbyKeys gpu time: " << ctx->timer_end();
        return result;
    }

    AriesDataBufferSPtr AggregateColumnData( const AriesDataBufferSPtr &srcColumn, AriesAggFunctionType aggType,
            const AriesInt32ArraySPtr &associated, const AriesInt32ArraySPtr &groups, const AriesInt32ArraySPtr &groupFlags, bool bDistinct,
            bool bSumForCount, const SumStrategy &strategy )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        const int8_t* data = srcColumn->GetData();
        AriesColumnType type = srcColumn->GetDataType();
        size_t tupleNum = associated->GetItemCount();

        AriesDataBufferSPtr result;
        if( tupleNum > 0 )
        {
            if ( aggType == AriesAggFunctionType::ANY_VALUE )
            {
                auto groupCount = groups->GetItemCount();
                result = std::make_shared< AriesDataBuffer >( srcColumn->GetDataType(), groupCount );
                auto data = result->GetData();
                auto srcData = srcColumn->GetData();
                auto itemSize = result->GetDataType().GetDataTypeSize();
                auto pAssociated = associated->GetData();
                auto groupData = groups->GetData();
                transform( [=]ARIES_LAMBDA( int index )
                {
                    auto offset = pAssociated[ groupData[ index ] ] * itemSize;
                    memcpy( data + index * itemSize, srcData + offset, itemSize );
                }, groups->GetItemCount(), *ctx );
                ctx->synchronize();
                return result;
            }
            else if( tupleNum == srcColumn->GetItemCount() )
            {
                auto dataType = type.DataType.ValueType;
                // if( type.HasNull || bDistinct || ( aggType == AriesAggFunctionType::MAX || aggType == AriesAggFunctionType::MIN )
                //         || ( dataType != AriesValueType::COMPACT_DECIMAL && dataType != AriesValueType::DECIMAL ) )
                {
                    AriesDataBuffer tmp( type );
                    if( bDistinct )
                    {
                        tmp.AllocArray( tupleNum );
                        tmp.PrefetchToGpu();
                        shuffle_column_data( data, type, tupleNum, associated->GetData(), tmp.GetData(), *ctx );
                        data = tmp.GetData();
                    }
                    result = aggregate_column( data, type, tupleNum, aggType, associated->GetData(), groups->GetData(), groups->GetItemCount(), *ctx,
                            bDistinct, bSumForCount );
                }
                // else
                // {
                //     result = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcColumn, aggType, associated, groupFlags, groups, strategy );
                // }
            }
            else
            {
                ARIES_ASSERT( aggType == AriesAggFunctionType::COUNT, "aggType: " + GetAriesAggFunctionTypeName( aggType ) );
                // for count(*), there is a dummy column with only 1 item.
                result = aggregate_column( data, type, tupleNum, aggType, associated->GetData(), groups->GetData(), groups->GetItemCount(), *ctx,
                        bDistinct, bSumForCount );
            }
        }
#ifdef ARIES_PROFILE
        LOG(INFO)<< "AggregateColumnData gpu time: " << ctx->timer_end();
#endif
        return result;
    }

END_ARIES_ACC_NAMESPACE
