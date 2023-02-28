/*
 * AriesSqlOperator_compareSingleColumnIter.cu
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#include "AriesSqlOperator_compareSingleColumnIter.h"
#include "AriesSqlOperator_helper.h"
#include "filter_column.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

    /*
    AriesBoolArraySPtr CompareColumn( const AriesColumnDataIterator* input, AriesComparisonOpType opType, const AriesDataBufferSPtr& params )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesBoolArraySPtr associated = make_shared< AriesBoolArray >();
        size_t paramCount = params->GetItemCount();
        size_t tupleNum = input->m_itemCount;
        if( paramCount > 0 )
        {
            const int8_t* param = params->GetData();
            AriesColumnType paramType = params->GetDataType();
            if( tupleNum > 0 )
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
                filter_column_data( input, opType, param, paramType, paramCount, associated->GetData(), *ctx );
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
        LOG(INFO)<< "CompareColumn( column iterator ) gpu time: " << ctx->timer_end();
        return associated;
    }
    */


END_ARIES_ACC_NAMESPACE


