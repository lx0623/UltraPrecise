#include "AriesSqlOperator_columnCompare.h"
#include "compare_two_column.h"
#include "utils/utils.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE
    size_t GetCompareColumnsPartitionCount( const AriesDataBufferSPtr& leftColumn,
                                            const AriesDataBufferSPtr& rightColumn )
    {
        size_t partitionCount = 1;
        size_t totalMemNeed = leftColumn->GetTotalBytes();
        totalMemNeed += rightColumn->GetTotalBytes();
        totalMemNeed += sizeof( AriesBool ) * leftColumn->GetItemCount();

        size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        double currentRatio = totalMemNeed / available;
        if( currentRatio > MAX_RATIO )
            partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

        return partitionCount;
    }

    AriesBoolArraySPtr CompareTowColumns( const AriesDataBufferSPtr& leftColumn, AriesComparisonOpType opType,
            const AriesDataBufferSPtr& rightColumn )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        ARIES_ASSERT( leftColumn->GetItemCount() == rightColumn->GetItemCount(),
                "leftColumn->GetItemCount(): " + to_string( leftColumn->GetItemCount() ) + ", rightColumn->GetItemCount(): "
                        + to_string( rightColumn->GetItemCount() ) );
        AriesBoolArraySPtr associated = make_shared< AriesBoolArray >();
        size_t tupleNum = leftColumn->GetItemCount();

        if( tupleNum > 0 )
        {
            associated->AllocArray( tupleNum );

            size_t partitionCount = GetCompareColumnsPartitionCount( leftColumn, rightColumn );

            vector<size_t> partitionRowCounts;
            partitionItems( tupleNum, partitionCount, partitionRowCounts );

            size_t handledRowCount = 0;
            auto pAssociated = associated->GetData();
            for ( auto partItemCount : partitionRowCounts )
            {
                if ( 0 == partItemCount )
                    break;
                leftColumn->PrefetchToGpu( handledRowCount, partItemCount );
                rightColumn->PrefetchToGpu( handledRowCount, partItemCount );
                associated->PrefetchToGpu( handledRowCount, partItemCount );
                compare_two_column_data( leftColumn->GetData( handledRowCount ),
                                         rightColumn->GetData( handledRowCount ),
                                         leftColumn->GetDataType(),
                                         rightColumn->GetDataType(),
                                         partItemCount,
                                         opType,
                                         pAssociated,
                                         *ctx );
                pAssociated += sizeof( AriesBool ) * partItemCount;
                handledRowCount += partItemCount;
            }
            
            // compare_two_column_data( leftColumn->GetData(), rightColumn->GetData(), leftColumn->GetDataType(), rightColumn->GetDataType(), tupleNum,
            //         opType, associated->GetData(), *ctx );
        }
        LOG(INFO)<< "CompareTowColumns gpu time: " << ctx->timer_end();
        return associated;
    }

END_ARIES_ACC_NAMESPACE
