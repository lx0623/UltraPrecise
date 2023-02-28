/*
 * AriesSqlOperator_filter.cu
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#include "AriesSqlOperator_filter.h"
#include "AriesSqlOperator_helper.h"
#include "kernel_util.hxx"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

    AriesIndicesArraySPtr FilterAssociated( const AriesBoolArraySPtr& associated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesInt32ArraySPtr psum;
        int32_t resultTupleNum = ExclusiveScan( associated, psum );
        AriesIndicesArraySPtr filteredIdx = std::make_shared< AriesIndicesArray >( resultTupleNum );
        if( resultTupleNum > 0 )
        {
            gather_filtered_index( associated->GetData(), psum->GetData(), associated->GetItemCount(), filteredIdx->GetData(), *ctx );
        }
        return filteredIdx;
    }

    AriesIndicesArraySPtr FilterFlags( const AriesInt8ArraySPtr &flags )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesInt32ArraySPtr psum;
        int32_t resultTupleNum = ExclusiveScan( flags, psum );
        AriesIndicesArraySPtr filteredIdx = std::make_shared< AriesIndicesArray >( resultTupleNum );
        if( resultTupleNum > 0 )
        {
            gather_filtered_index( flags->GetData(), psum->GetData(), flags->GetItemCount(), filteredIdx->GetData(), *ctx );
        }
        return filteredIdx;
    }

    AriesManagedIndicesArraySPtr FilterFlags( const AriesManagedInt8ArraySPtr &flags )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        AriesInt32ArraySPtr psum;
        flags->PrefetchToGpu();
        int32_t resultTupleNum = ExclusiveScan( flags, psum );
        AriesManagedIndicesArraySPtr filteredIdx = std::make_shared< AriesManagedIndicesArray >( resultTupleNum );
        if( resultTupleNum > 0 )
        {
            filteredIdx->PrefetchToGpu();
            gather_filtered_index( flags->GetData(), psum->GetData(), flags->GetItemCount(), filteredIdx->GetData(), *ctx );
        }
        return filteredIdx;
    }

    std::vector< AriesDataBufferSPtr > FilterColumnData( const std::vector< AriesDataBufferSPtr >& columns, const AriesInt32ArraySPtr& associated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        std::vector< AriesDataBufferSPtr > result;
        int count = columns.size();
        if( count > 0 )
        {
            AriesInt32ArraySPtr psum;
            int32_t resultTupleNum = ExclusiveScan( associated, psum );
            if( resultTupleNum > 0 )
            {
                ctx->timer_begin();
                AriesManagedArray< DataBlockInfo > blocks( count );
                AriesManagedArray< int8_t* > outputs( count );
                DataBlockInfo *block = blocks.GetData();
                int8_t** output = outputs.GetData();
                for( const auto& col : columns )
                {
                    AriesDataBufferSPtr newColumn = make_shared< AriesDataBuffer >( col->GetDataType(), resultTupleNum );
                    newColumn->PrefetchToGpu();
                    result.push_back( newColumn );
                    block->Data = col->GetData();
                    block->ElementSize = col->GetDataType().GetDataTypeSize();
                    *output = newColumn->GetData();
                    ++output;
                    ++block;
                }
                blocks.PrefetchToGpu();
                outputs.PrefetchToGpu();
                gather_filtered_data( blocks.GetData(), count, columns[0]->GetItemCount(), associated->GetData(), psum->GetData(), outputs.GetData(),
                        *ctx );
                LOG(INFO)<< "FilterColumnData gpu time: " << ctx->timer_end();
            }
        }
        return result;
    }

    std::vector< AriesDataBufferSPtr > FilterColumnDataEx( const std::vector< AriesDataBufferSPtr >& columns, const AriesInt32ArraySPtr& associated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        std::vector< AriesDataBufferSPtr > result;
        int count = columns.size();
        if( count > 0 )
        {
            AriesInt32ArraySPtr psum;
            int32_t resultTupleNum = ExclusiveScan( associated, psum );

            if( resultTupleNum > 0 )
            {
                AriesInt32ArraySPtr sequences = associated->CloneWithNoContent();
                InitSequenceValue( sequences );
                AriesInt32Array indices( resultTupleNum );
                gather_filtered_data( sequences->GetData(), sequences->GetItemCount(), associated->GetData(), psum->GetData(), indices.GetData(),
                        *ctx );

                ctx->timer_begin();
                AriesManagedArray< SimpleDataBlockInfo > blocks( count );
                AriesManagedArray< int8_t* > outputs( count );
                SimpleDataBlockInfo *block = blocks.GetData();
                int8_t** output = outputs.GetData();
                for( const auto& col : columns )
                {
                    AriesDataBufferSPtr newColumn = make_shared< AriesDataBuffer >( col->GetDataType(), resultTupleNum );
                    newColumn->PrefetchToGpu();
                    result.push_back( newColumn );
                    block->Data = col->GetData();
                    block->ElementSize = col->GetDataType().GetDataTypeSize();
                    *output = newColumn->GetData();
                    ++output;
                    ++block;
                }
                blocks.PrefetchToGpu();
                outputs.PrefetchToGpu();
                gather_filtered_data_ex( blocks.GetData(), count, indices.GetItemCount(), indices.GetData(), outputs.GetData(), *ctx );
                LOG(INFO)<< "FilterColumnDataEx gpu time: " << ctx->timer_end();
            }
        }
        return result;
    }

END_ARIES_ACC_NAMESPACE

