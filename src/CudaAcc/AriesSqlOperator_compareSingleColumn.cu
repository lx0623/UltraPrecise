#include "AriesSqlOperator_compareSingleColumn.h"
#include "AriesSqlOperator_helper.h"
#include "filter_column.h"
#include "utils/utils.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

    size_t GetCompareColumnPartitionCount( const AriesDataBufferSPtr& columnData )
    {
        size_t partitionCount = 1;
        size_t totalMemNeed = columnData->GetTotalBytes();
        totalMemNeed += sizeof( AriesBool ) * columnData->GetItemCount();

        size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        double currentRatio = totalMemNeed / available;
        if( currentRatio > MAX_RATIO )
            partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

        return partitionCount;
    }

    AriesBoolArraySPtr CompareColumn( const AriesDataBufferSPtr& columnData, AriesComparisonOpType opType, const AriesDataBufferSPtr& params )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesBoolArraySPtr associated = make_shared< AriesBoolArray >();
        size_t paramCount = params->GetItemCount();
        size_t tupleNum = columnData->GetItemCount();
        if( paramCount > 0 )
        {
            const int8_t* data = columnData->GetData();
            AriesColumnType type = columnData->GetDataType();

            const int8_t* param = params->GetData();
            AriesColumnType paramType = params->GetDataType();
            if ( tupleNum > 0 )
            {
                params->PrefetchToGpu();
                associated->AllocArray( tupleNum );

                if( opType == AriesComparisonOpType::IN && paramCount == 1 )
                {
                    opType = AriesComparisonOpType::EQ;
                }
                else if( opType == AriesComparisonOpType::NOTIN && paramCount == 1 )
                {
                    opType = AriesComparisonOpType::NE;
                }

                size_t partitionCount = GetCompareColumnPartitionCount( columnData );

                vector<size_t> partitionRowCounts;
                partitionItems( tupleNum, partitionCount, partitionRowCounts );

                size_t handledRowCount = 0;
                auto pAssociated = associated->GetData();
                for ( auto partItemCount : partitionRowCounts )
                {
                    if ( 0 == partItemCount )
                        break;

                    columnData->PrefetchToGpu( handledRowCount, partItemCount );
                    associated->PrefetchToGpu( handledRowCount, partItemCount );
                    filter_column_data( columnData->GetData( handledRowCount ),
                                        type,
                                        partItemCount,
                                        opType,
                                        param,
                                        paramType,
                                        paramCount,
                                        pAssociated,
                                        *ctx );
                    pAssociated += sizeof( AriesBool ) * partItemCount;
                    handledRowCount += partItemCount;
                }
                // filter_column_data( data, type, tupleNum, opType, param, paramType, paramCount, associated->GetData(), *ctx );
            }
        }
        else
        {
            ARIES_ASSERT( opType == AriesComparisonOpType::IN || opType == AriesComparisonOpType::NOTIN,
                    "opType: " + GetAriesComparisonOpTypeName( opType ) );
            // no row match the condition
            if( tupleNum > 0 )
            {
                associated->AllocArray( tupleNum );
                FillWithValue( associated, AriesBool::ValueType::False );
            }

        }
        LOG(INFO)<< "CompareColumn gpu time: " << ctx->timer_end();
        return associated;
    }

END_ARIES_ACC_NAMESPACE
