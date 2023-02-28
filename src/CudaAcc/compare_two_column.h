#pragma once

#include "algorithm/context.hxx"
#include "datatypes/decimal.hxx"
#include "compare_function.h"

using namespace aries_engine;
BEGIN_ARIES_ACC_NAMESPACE

template< typename output_t >
void compare_two_column_data(const int8_t* left,
                             const int8_t* right,
                             AriesColumnType leftType,
                             AriesColumnType rightType,
                             size_t tupleNum,
                             AriesComparisonOpType opType,
                             output_t* data_output,
                             context_t& context )
{
    int block = ( tupleNum + BLOCK_DIM - 1 ) / BLOCK_DIM;
    CompareFunctionPointer cmp = GetCompareFunction( leftType, rightType, opType );
    if(!cmp) ThrowNotSupportedException("comparison between "+GetDataTypeStringName(leftType.DataType.ValueType)+" and "+GetDataTypeStringName(rightType.DataType.ValueType));

    int left_type_size = leftType.GetDataTypeSize();
    switch(leftType.DataType.ValueType){
        case AriesValueType::COMPACT_DECIMAL:
            left_type_size = ((int)leftType.DataType.Precision << COMPACT_DECIMAL_PRECISION_OFFSET) | 
                                ((int)leftType.DataType.Scale << COMPACT_DECIMAL_SCALE_OFFSET) | 
                                left_type_size;
            break;       
    }
    int right_type_size = rightType.GetDataTypeSize();
    switch(rightType.DataType.ValueType){
        case AriesValueType::COMPACT_DECIMAL:
            right_type_size = ((int)rightType.DataType.Precision << COMPACT_DECIMAL_PRECISION_OFFSET) | 
                                ((int)rightType.DataType.Scale << COMPACT_DECIMAL_SCALE_OFFSET) | 
                                right_type_size;
            break;
    }
    compare_two_column_data<<< block, BLOCK_DIM >>>(left,
                                                    left_type_size,
                                                    right,
                                                    right_type_size,
                                                    data_output,
                                                    cmp, tupleNum );
}

END_ARIES_ACC_NAMESPACE
