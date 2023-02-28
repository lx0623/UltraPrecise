#ifndef FILTER_COLUMN_H_
#define FILTER_COLUMN_H_
#pragma once
#include "AriesEngineAlgorithm.h"
#include "algorithm/context.hxx"
#include "datatypes/decimal.hxx"
#include "compare_function.h"

using namespace aries_engine;
BEGIN_ARIES_ACC_NAMESPACE


template< typename output_t >
void filter_column_data( const AriesColumnDataIterator* data,
                         AriesComparisonOpType comp,
                         const int8_t* param,
                         AriesColumnType paramType,
                         size_t paramCount,
                         output_t* data_output,
                         context_t& context )
{
    AriesColumnType type = data->m_valueType;

    int block = ( data->m_itemCount + BLOCK_DIM - 1 ) / BLOCK_DIM;

    int left_type_size = type.GetDataTypeSize();
    switch(type.DataType.ValueType){
        case AriesValueType::COMPACT_DECIMAL:
            left_type_size = ((int)type.DataType.Precision << COMPACT_DECIMAL_PRECISION_OFFSET) | 
                                ((int)type.DataType.Scale << COMPACT_DECIMAL_SCALE_OFFSET) | 
                                left_type_size;
            break;       
    }
    int right_type_size = paramType.GetDataTypeSize();
    switch(paramType.DataType.ValueType){
        case AriesValueType::COMPACT_DECIMAL:
            right_type_size = ((int)paramType.DataType.Precision << COMPACT_DECIMAL_PRECISION_OFFSET) | 
                                ((int)paramType.DataType.Scale << COMPACT_DECIMAL_SCALE_OFFSET) | 
                                right_type_size;
            break;
    }

    switch( comp )
    {
        case AriesComparisonOpType::EQ:
        case AriesComparisonOpType::NE:
        case AriesComparisonOpType::GT:
        case AriesComparisonOpType::LT:
        case AriesComparisonOpType::GE:
        case AriesComparisonOpType::LE:
        case AriesComparisonOpType::LIKE:
        {
            CompareFunctionPointer cmp = GetCompareFunction(type, paramType, comp);
            if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(type.DataType.ValueType)+" and "+GetDataTypeStringName(paramType.DataType.ValueType));
            filter_column_data_iter<<<block, BLOCK_DIM>>>( data,
                                                    (int)type.GetDataTypeSize(),
                                                    param,
                                                    right_type_size,
                                                    data_output,
                                                    cmp,
                                                    data->m_itemCount);
            CUDA_LAST_ERROR
            return;
        }
        case AriesComparisonOpType::IN:
        {
            if (paramCount > 1)
            {
                CompareFunctionPointer cmp_gt = GetCompareFunction(type, paramType, AriesComparisonOpType::GT);
                assert(cmp_gt);
                CompareFunctionPointer cmp_eq = GetCompareFunction(type, paramType, AriesComparisonOpType::EQ);
                assert(cmp_eq);
                filter_column_data_iter_in_sorted<<<block, BLOCK_DIM>>>( data,
                                                        (int)type.GetDataTypeSize(),
                                                        param,
                                                        right_type_size,
                                                        data_output,
                                                        cmp_gt,
                                                        cmp_eq,
                                                        data->m_itemCount,
                                                        paramCount );;
                CUDA_LAST_ERROR
                return;
            }
            CompareFunctionPointer cmp = GetCompareFunction(type, paramType, comp);
            if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(type.DataType.ValueType)+" and "+GetDataTypeStringName(paramType.DataType.ValueType));
            filter_column_data_iter_in<<<block, BLOCK_DIM>>>( data,
                                                        (int)type.GetDataTypeSize(),
                                                        param,
                                                        right_type_size,
                                                        data_output,
                                                        cmp,
                                                        data->m_itemCount,
                                                        paramCount );
            CUDA_LAST_ERROR
            return;
        }
        case AriesComparisonOpType::NOTIN:
        {
            if(paramCount > 1)
            {
                CompareFunctionPointer cmp_gt = GetCompareFunction(type, paramType, AriesComparisonOpType::GT);
                assert(cmp_gt);
                CompareFunctionPointer cmp_eq = GetCompareFunction(type, paramType, AriesComparisonOpType::EQ);
                assert(cmp_eq);
                filter_column_data_iter_not_in_sorted<<<block, BLOCK_DIM>>>( data,
                                                        (int)type.GetDataTypeSize(),
                                                        param,
                                                        right_type_size,
                                                        data_output,
                                                        cmp_gt,
                                                        cmp_eq,
                                                        data->m_itemCount,
                                                        paramCount );;
                CUDA_LAST_ERROR
                return;
            }
            CompareFunctionPointer cmp = GetCompareFunction(type, paramType, comp);
            if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(type.DataType.ValueType)+" and "+GetDataTypeStringName(paramType.DataType.ValueType));
            filter_column_data_iter_not_in<<<block, BLOCK_DIM>>>( data,
                                                            (int)type.GetDataTypeSize(),
                                                            param,
                                                            right_type_size,
                                                            data_output,
                                                            cmp,
                                                            data->m_itemCount,
                                                            paramCount );
            CUDA_LAST_ERROR
            return;
        }
    }
}

template< typename output_t >
void filter_column_data( const int8_t* data, //
                         AriesColumnType type,
                         size_t tupleNum,
                         AriesComparisonOpType comp,
                         const int8_t* param,
                         AriesColumnType paramType,
                         size_t paramCount,
                         output_t* data_output,
                         context_t& context )
{
    int block = ( tupleNum + BLOCK_DIM - 1 ) / BLOCK_DIM;
    int left_type_size = type.GetDataTypeSize();
    switch(type.DataType.ValueType){
        case AriesValueType::COMPACT_DECIMAL:
            left_type_size = ((int)type.DataType.Precision << COMPACT_DECIMAL_PRECISION_OFFSET) | 
                                ((int)type.DataType.Scale << COMPACT_DECIMAL_SCALE_OFFSET) | 
                                left_type_size;
            break;       
    }
    int right_type_size = paramType.GetDataTypeSize();
    switch(paramType.DataType.ValueType){
        case AriesValueType::COMPACT_DECIMAL:
            right_type_size = ((int)paramType.DataType.Precision << COMPACT_DECIMAL_PRECISION_OFFSET) | 
                                ((int)paramType.DataType.Scale << COMPACT_DECIMAL_SCALE_OFFSET) | 
                                right_type_size;
            break;
    }

    switch( comp )
    {
        case AriesComparisonOpType::EQ:
        case AriesComparisonOpType::NE:
        case AriesComparisonOpType::GT:
        case AriesComparisonOpType::LT:
        case AriesComparisonOpType::GE:
        case AriesComparisonOpType::LE:
        case AriesComparisonOpType::LIKE:
        {
            CompareFunctionPointer cmp = GetCompareFunction(type, paramType, comp);
            if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(type.DataType.ValueType)+" and "+GetDataTypeStringName(paramType.DataType.ValueType));
            filter_column_data<<<block, BLOCK_DIM>>>( data,
                                                left_type_size,
                                                param,
                                                right_type_size,
                                                data_output,
                                                cmp,
                                                tupleNum );
            CUDA_LAST_ERROR
            return;
        }
        case AriesComparisonOpType::IN:
        {
            if (paramCount > 1)
            {
                CompareFunctionPointer cmp_gt = GetCompareFunction(type, paramType, AriesComparisonOpType::GT);
                assert(cmp_gt);
                CompareFunctionPointer cmp_eq = GetCompareFunction(type, paramType, AriesComparisonOpType::EQ);
                assert(cmp_eq);
                filter_column_data_in_sorted<<<block, BLOCK_DIM>>>( data,
                                                left_type_size,
                                                param,
                                                right_type_size,
                                                data_output,
                                                cmp_gt,
                                                cmp_eq,
                                                tupleNum,
                                                paramCount );
                CUDA_LAST_ERROR
                return;
            }
            CompareFunctionPointer cmp = GetCompareFunction(type, paramType, comp);
            if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(type.DataType.ValueType)+" and "+GetDataTypeStringName(paramType.DataType.ValueType));
            filter_column_data_in<<<block, BLOCK_DIM>>>( data,
                                                left_type_size,
                                                param,
                                                right_type_size,
                                                data_output,
                                                cmp,
                                                tupleNum,
                                                paramCount );
            CUDA_LAST_ERROR
            return;
        }
        case AriesComparisonOpType::NOTIN:
        {
            if (paramCount > 1)
            {
                CompareFunctionPointer cmp_gt = GetCompareFunction(type, paramType, AriesComparisonOpType::GT);
                assert(cmp_gt);
                CompareFunctionPointer cmp_eq = GetCompareFunction(type, paramType, AriesComparisonOpType::EQ);
                assert(cmp_eq);
                filter_column_data_not_in_sorted<<<block, BLOCK_DIM>>>( data,
                                                left_type_size,
                                                param,
                                                right_type_size,
                                                data_output,
                                                cmp_gt,
                                                cmp_eq,
                                                tupleNum,
                                                paramCount );
                CUDA_LAST_ERROR
                return;
            }
            CompareFunctionPointer cmp = GetCompareFunction(type, paramType, comp);
            if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(type.DataType.ValueType)+" and "+GetDataTypeStringName(paramType.DataType.ValueType));
            filter_column_data_not_in<<<block, BLOCK_DIM>>>( data,
                                                left_type_size,
                                                param,
                                                right_type_size,
                                                data_output,
                                                cmp,
                                                tupleNum,
                                                paramCount );
            CUDA_LAST_ERROR
            return;
        }
    }
}


END_ARIES_ACC_NAMESPACE
#endif // FILTER_COLUMN_H_