#pragma once

#include "algorithm/context.hxx"
#include "algorithm/operators.hxx"
#include "AriesEngine/AriesUtil.h"
BEGIN_ARIES_ACC_NAMESPACE


template< typename type_t >
void sort_column_data( int8_t* data, AriesColumnType type, size_t tupleNum, AriesOrderByType order, type_t* associated, context_t& context,
        bool nullSmaller = true, bool bShuffle = true )
{
    size_t memLimitInBytes = AriesDeviceProperty::GetInstance().GetMemoryCapacity() * 0.9;
    size_t memCost = ( type.GetDataTypeSize() + sizeof( type_t ) ) * tupleNum * 2.5;
    bool usePartitionedMergeSort = memCost > memLimitInBytes;
    size_t memLimitInMb = memLimitInBytes / ( 1 << 20 );

    AriesValueType valueType = type.DataType.ValueType;
    switch( valueType )
    {
        case AriesValueType::CHAR:
        {
            size_t len = type.GetDataTypeSize();
            if( type.DataType.Length > 1 )
            {
                // if( usePartitionedMergeSort )
                // {
                //     mem_t< int8_t > buffer;
                //     if( !bShuffle )
                //     {
                //         size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                //         buffer.alloc( totalBytes );
                //         AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                //         data = buffer.data();
                //     }
                //     if( !type.HasNull )
                //     {
                //         if( order == AriesOrderByType::ASC )
                //             mergesort( ( char* )data, len, associated, tupleNum, less_t_str< false, false >(), context, memLimitInMb );
                //         else
                //             mergesort( ( char* )data, len, associated, tupleNum, greater_t_str< false, false >(), context, memLimitInMb );
                //     }
                //     else
                //     {
                //         if( order == AriesOrderByType::ASC )
                //         {
                //             if( nullSmaller )
                //                 mergesort( ( char* )data, len, associated, tupleNum, less_t_str_null_smaller< true, true >(), context, memLimitInMb );
                //             else
                //                 mergesort( ( char* )data, len, associated, tupleNum, less_t_str_null_bigger< true, true >(), context, memLimitInMb );
                //         }
                //         else
                //         {
                //             if( nullSmaller )
                //                 mergesort( ( char* )data, len, associated, tupleNum, greater_t_str_null_smaller< true, true >(), context, memLimitInMb );
                //             else
                //                 mergesort( ( char* )data, len, associated, tupleNum, greater_t_str_null_bigger< true, true >(), context, memLimitInMb );
                //         }
                //     }
                // }
                // else 
                {
                    if( !type.HasNull )
                        radix_sort( ( char* )data, len, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                    else
                        radix_sort_has_null( ( char* )data, len, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller, bShuffle );
                }
            }
            else
            {
                if( !type.HasNull )
                    radix_sort( ( int8_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                else
                    radix_sort( ( nullable_type< int8_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller, bShuffle );
                break;
            }
            break;
        }
        case AriesValueType::BOOL:
        case AriesValueType::INT8:
        {
            if( !type.HasNull )
                radix_sort( ( int8_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
            else
                radix_sort( ( nullable_type< int8_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                        bShuffle );
            break;
        }
        case AriesValueType::INT16:
        {
            if( usePartitionedMergeSort )
            {
                mem_t< int8_t > buffer;
                if( !bShuffle )
                {
                    size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                    buffer.alloc( totalBytes );
                    AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                    data = buffer.data();
                }
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int16_t* )data, associated, tupleNum, less_t< int16_t >(), context, memLimitInMb );
                    else
                        mergesort( ( int16_t* )data, associated, tupleNum, greater_t< int16_t >(), context, memLimitInMb );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int16_t >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< int16_t > >(),
                                    context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< int16_t >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< int16_t > >(),
                                    context, memLimitInMb );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int16_t >* )data, associated, tupleNum,
                                    greater_t_null_smaller< nullable_type< int16_t > >(), context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< int16_t >* )data, associated, tupleNum,
                                    greater_t_null_bigger< nullable_type< int16_t > >(), context, memLimitInMb );
                    }
                }
            }
            else 
            {
                if( !type.HasNull )
                    radix_sort( ( int16_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                else
                    radix_sort( ( nullable_type< int16_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                            bShuffle );
            }
            break;
        }
        case AriesValueType::INT32:
        {
            if( usePartitionedMergeSort )
            {
                mem_t< int8_t > buffer;
                if( !bShuffle )
                {
                    size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                    buffer.alloc( totalBytes );
                    AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                    data = buffer.data();
                }
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int32_t* )data, associated, tupleNum, less_t< int32_t >(), context, memLimitInMb );
                    else
                        mergesort( ( int32_t* )data, associated, tupleNum, greater_t< int32_t >(), context, memLimitInMb );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int32_t >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< int32_t > >(),
                                    context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< int32_t >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< int32_t > >(),
                                    context, memLimitInMb );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int32_t >* )data, associated, tupleNum,
                                    greater_t_null_smaller< nullable_type< int32_t > >(), context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< int32_t >* )data, associated, tupleNum,
                                    greater_t_null_bigger< nullable_type< int32_t > >(), context, memLimitInMb );
                    }
                }
            }
            else 
            {
                if( !type.HasNull )
                    radix_sort( ( int32_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                else
                    radix_sort( ( nullable_type< int32_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                            bShuffle );
            }
            break;
        }
        case AriesValueType::INT64:
        {
            if( usePartitionedMergeSort )
            {
                mem_t< int8_t > buffer;
                if( !bShuffle )
                {
                    size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                    buffer.alloc( totalBytes );
                    AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                    data = buffer.data();
                }
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int64_t* )data, associated, tupleNum, less_t< int64_t >(), context, memLimitInMb );
                    else
                        mergesort( ( int64_t* )data, associated, tupleNum, greater_t< int64_t >(), context, memLimitInMb );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int64_t >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< int64_t > >(),
                                    context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< int64_t >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< int64_t > >(),
                                    context, memLimitInMb );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int64_t >* )data, associated, tupleNum,
                                    greater_t_null_smaller< nullable_type< int64_t > >(), context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< int64_t >* )data, associated, tupleNum,
                                    greater_t_null_bigger< nullable_type< int64_t > >(), context, memLimitInMb );
                    }
                }
            }
            else 
            {
                if( !type.HasNull )
                    radix_sort( ( int64_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                else
                    radix_sort( ( nullable_type< int64_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                            bShuffle );
            }
            break;
        }
        case AriesValueType::UINT8:
        {
            if( !type.HasNull )
                radix_sort( ( uint8_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
            else
                radix_sort( ( nullable_type< uint8_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                        bShuffle );
            break;
        }
        case AriesValueType::UINT16:
        {
            if( !type.HasNull )
                radix_sort( ( uint16_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
            else
                radix_sort( ( nullable_type< uint16_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                        bShuffle );
            break;
        }
        case AriesValueType::UINT32:
        {
            if( !type.HasNull )
                radix_sort( ( uint32_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
            else
                radix_sort( ( nullable_type< uint32_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                        bShuffle );
            break;
        }
        case AriesValueType::UINT64:
        {
            if( !type.HasNull )
                radix_sort( ( uint64_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
            else
                radix_sort( ( nullable_type< uint64_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                        bShuffle );
            break;
        }
        case AriesValueType::FLOAT:
        {
            if( usePartitionedMergeSort )
            {
                mem_t< int8_t > buffer;
                if( !bShuffle )
                {
                    size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                    buffer.alloc( totalBytes );
                    AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                    data = buffer.data();
                }
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( float* )data, associated, tupleNum, less_t< float >(), context, memLimitInMb );
                    else
                        mergesort( ( float* )data, associated, tupleNum, greater_t< float >(), context, memLimitInMb );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< float >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< float > >(),
                                    context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< float >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< float > >(),
                                    context, memLimitInMb );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< float >* )data, associated, tupleNum,
                                    greater_t_null_smaller< nullable_type< float > >(), context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< float >* )data, associated, tupleNum,
                                    greater_t_null_bigger< nullable_type< float > >(), context, memLimitInMb );
                    }
                }
            }
            else 
            {
                if( !type.HasNull )
                    radix_sort( ( float* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                else
                    radix_sort( ( nullable_type< float >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                            bShuffle );
            }
            break;
        }
        case AriesValueType::DOUBLE:
        {
            if( usePartitionedMergeSort )
            {
                mem_t< int8_t > buffer;
                if( !bShuffle )
                {
                    size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                    buffer.alloc( totalBytes );
                    AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                    data = buffer.data();
                }
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( double* )data, associated, tupleNum, less_t< double >(), context, memLimitInMb );
                    else
                        mergesort( ( double* )data, associated, tupleNum, greater_t< double >(), context, memLimitInMb );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< double >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< double > >(),
                                    context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< double >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< double > >(),
                                    context, memLimitInMb );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< double >* )data, associated, tupleNum,
                                    greater_t_null_smaller< nullable_type< double > >(), context, memLimitInMb );
                        else
                            mergesort( ( nullable_type< double >* )data, associated, tupleNum,
                                    greater_t_null_bigger< nullable_type< double > >(), context, memLimitInMb );
                    }
                }
            }
            else 
            {
                if( !type.HasNull )
                    radix_sort( ( double* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
                else
                    radix_sort( ( nullable_type< double >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
                            bShuffle );
            }
            break;
        }
        case AriesValueType::DECIMAL:
        {
            mem_t< int8_t > buffer;
            if( !bShuffle )
            {
                size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                buffer.alloc( totalBytes );
                AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                data = buffer.data();
            }
            if( !type.HasNull )
            {
                if( order == AriesOrderByType::ASC )
                    mergesort( ( Decimal* )data, associated, tupleNum, less_t< Decimal >(), context, memLimitInMb );
                else
                    mergesort( ( Decimal* )data, associated, tupleNum, greater_t< Decimal >(), context, memLimitInMb );
            }
            else
            {
                if( order == AriesOrderByType::ASC )
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< Decimal >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< Decimal > >(),
                                context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< Decimal >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< Decimal > >(),
                                context, memLimitInMb );
                }
                else
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< Decimal >* )data, associated, tupleNum, greater_t_null_smaller< nullable_type< Decimal > >(),
                                context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< Decimal >* )data, associated, tupleNum, greater_t_null_bigger< nullable_type< Decimal > >(),
                                context, memLimitInMb );
                }
            }
            break;
        }
        case AriesValueType::COMPACT_DECIMAL:
        {
            mem_t< int8_t > buffer;
            if( !bShuffle )
            {
                size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                buffer.alloc( totalBytes );
                AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                data = buffer.data();
            }
            int itemLen = type.GetDataTypeSize();
            if( !type.HasNull )
            {
                if( order == AriesOrderByType::ASC )
                    mergesort( ( char * )data, itemLen, associated, tupleNum,
                            less_t_CompactDecimal_null_smaller< false >( type.DataType.Precision, type.DataType.Scale ), context, memLimitInMb );
                else
                    mergesort( ( char * )data, itemLen, associated, tupleNum,
                            greater_t_CompactDecimal_null_smaller< false >( type.DataType.Precision, type.DataType.Scale ), context, memLimitInMb );
            }
            else
            {
                if( order == AriesOrderByType::ASC )
                {
                    if( nullSmaller )
                        mergesort( ( char * )data, itemLen, associated, tupleNum,
                                less_t_CompactDecimal_null_smaller< true >( type.DataType.Precision, type.DataType.Scale ), context, memLimitInMb );
                    else
                        mergesort( ( char * )data, itemLen, associated, tupleNum,
                                less_t_CompactDecimal_null_bigger< true >( type.DataType.Precision, type.DataType.Scale ), context, memLimitInMb );
                }
                else
                {
                    if( nullSmaller )
                        mergesort( ( char * )data, itemLen, associated, tupleNum,
                                greater_t_CompactDecimal_null_smaller< true >( type.DataType.Precision, type.DataType.Scale ), context, memLimitInMb );
                    else
                        mergesort( ( char * )data, itemLen, associated, tupleNum,
                                greater_t_CompactDecimal_null_bigger< true >( type.DataType.Precision, type.DataType.Scale ), context, memLimitInMb );
                }
            }
            break;
        }
        case AriesValueType::DATE:
        {
            mem_t< int8_t > buffer;
            if( !bShuffle )
            {
                size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                buffer.alloc( totalBytes );
                AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                data = buffer.data();
            }
            if( !type.HasNull )
            {
                if( order == AriesOrderByType::ASC )
                    mergesort( ( AriesDate* )data, associated, tupleNum, less_t< AriesDate >(), context, memLimitInMb );
                else
                    mergesort( ( AriesDate* )data, associated, tupleNum, greater_t< AriesDate >(), context, memLimitInMb );
            }
            else
            {
                if( order == AriesOrderByType::ASC )
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesDate >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< AriesDate > >(),
                                context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesDate >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< AriesDate > >(),
                                context, memLimitInMb );
                }
                else
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesDate >* )data, associated, tupleNum,
                                greater_t_null_smaller< nullable_type< AriesDate > >(), context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesDate >* )data, associated, tupleNum,
                                greater_t_null_bigger< nullable_type< AriesDate > >(), context, memLimitInMb );
                }
            }
            break;
        }
        case AriesValueType::DATETIME:
        {
            mem_t< int8_t > buffer;
            if( !bShuffle )
            {
                size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                buffer.alloc( totalBytes );
                AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                data = buffer.data();
            }
            if( !type.HasNull )
            {
                if( order == AriesOrderByType::ASC )
                    mergesort( ( AriesDatetime* )data, associated, tupleNum, less_t< AriesDatetime >(), context, memLimitInMb );
                else
                    mergesort( ( AriesDatetime* )data, associated, tupleNum, greater_t< AriesDatetime >(), context, memLimitInMb );
            }
            else
            {
                if( order == AriesOrderByType::ASC )
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesDatetime >* )data, associated, tupleNum,
                                less_t_null_smaller< nullable_type< AriesDatetime > >(), context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesDatetime >* )data, associated, tupleNum,
                                less_t_null_bigger< nullable_type< AriesDatetime > >(), context, memLimitInMb );
                }
                else
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesDatetime >* )data, associated, tupleNum,
                                greater_t_null_smaller< nullable_type< AriesDatetime > >(), context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesDatetime >* )data, associated, tupleNum,
                                greater_t_null_bigger< nullable_type< AriesDatetime > >(), context, memLimitInMb );
                }
            }
            break;
        }
        case AriesValueType::TIMESTAMP:
        {
            mem_t< int8_t > buffer;
            if( !bShuffle )
            {
                size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                buffer.alloc( totalBytes );
                AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                data = buffer.data();
            }
            if( !type.HasNull )
            {
                if( order == AriesOrderByType::ASC )
                    mergesort( ( AriesTimestamp* )data, associated, tupleNum, less_t< AriesTimestamp >(), context, memLimitInMb );
                else
                    mergesort( ( AriesTimestamp* )data, associated, tupleNum, greater_t< AriesTimestamp >(), context, memLimitInMb );
            }
            else
            {
                if( order == AriesOrderByType::ASC )
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesTimestamp >* )data, associated, tupleNum,
                                less_t_null_smaller< nullable_type< AriesTimestamp > >(), context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesTimestamp >* )data, associated, tupleNum,
                                less_t_null_bigger< nullable_type< AriesTimestamp > >(), context, memLimitInMb );
                }
                else
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesTimestamp >* )data, associated, tupleNum,
                                greater_t_null_smaller< nullable_type< AriesTimestamp > >(), context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesTimestamp >* )data, associated, tupleNum,
                                greater_t_null_bigger< nullable_type< AriesTimestamp > >(), context, memLimitInMb );
                }
            }
            break;
        }
        case AriesValueType::YEAR:
        {
            mem_t< int8_t > buffer;
            if( !bShuffle )
            {
                size_t totalBytes = type.GetDataTypeSize() * tupleNum;
                buffer.alloc( totalBytes );
                AriesMemAllocator::MemCopy( buffer.data(), data, totalBytes );
                data = buffer.data();
            }
            if( !type.HasNull )
            {
                if( order == AriesOrderByType::ASC )
                    mergesort( ( AriesYear* )data, associated, tupleNum, less_t< AriesYear >(), context, memLimitInMb );
                else
                    mergesort( ( AriesYear* )data, associated, tupleNum, greater_t< AriesYear >(), context, memLimitInMb );
            }
            else
            {
                if( order == AriesOrderByType::ASC )
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesYear >* )data, associated, tupleNum, less_t_null_smaller< nullable_type< AriesYear > >(),
                                context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesYear >* )data, associated, tupleNum, less_t_null_bigger< nullable_type< AriesYear > >(),
                                context, memLimitInMb );
                }
                else
                {
                    if( nullSmaller )
                        mergesort( ( nullable_type< AriesYear >* )data, associated, tupleNum,
                                greater_t_null_smaller< nullable_type< AriesYear > >(), context, memLimitInMb );
                    else
                        mergesort( ( nullable_type< AriesYear >* )data, associated, tupleNum,
                                greater_t_null_bigger< nullable_type< AriesYear > >(), context, memLimitInMb );
                }
            }
            break;
        }
        default:
            //FIXME need support all data types.
            assert( 0 );
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
    }
}

template< typename type_t >
    AriesDataBufferSPtr sort_column_data( int8_t** data, int64_t* block_size_prefix_sum, int block_count, AriesColumnType type, int* indices,
            size_t indices_count, bool indices_has_null, AriesOrderByType order, type_t* associated, context_t& context, bool nullSmaller = true )
    {
        AriesDataBufferSPtr result;
        AriesColumnType outputType = type;
        outputType.HasNull = indices_has_null || type.HasNull;
        result = std::make_shared< AriesDataBuffer >( outputType, indices_count );
        AriesValueType valueType = type.DataType.ValueType;
        int8_t* output = result->GetData();
        switch( valueType )
        {
            case AriesValueType::CHAR:
            {
                int len = type.GetDataTypeSize();
                if( type.DataType.Length > 1 )
                {
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
                }
                else
                {
                    if( !type.HasNull )
                    {
                        if( order == AriesOrderByType::ASC )
                        {
                            if( indices_has_null )
                            {
                                if( nullSmaller )
                                    mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                            ( nullable_type< int8_t >* )output, associated, indices_count,
                                            less_t_null_smaller< nullable_type< int8_t > >(), context );
                                else
                                    mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                            ( nullable_type< int8_t >* )output, associated, indices_count,
                                            less_t_null_bigger< nullable_type< int8_t > >(), context );
                            }
                            else
                                mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int8_t * )output,
                                        associated, indices_count, less_t< int8_t >(), context );
                        }
                        else
                        {
                            if( indices_has_null )
                            {
                                if( nullSmaller )
                                    mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                            ( nullable_type< int8_t >* )output, associated, indices_count,
                                            greater_t_null_smaller< nullable_type< int8_t > >(), context );
                                else
                                    mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                            ( nullable_type< int8_t >* )output, associated, indices_count,
                                            greater_t_null_bigger< nullable_type< int8_t > >(), context );
                            }
                            else
                                mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int8_t * )output,
                                        associated, indices_count, greater_t< int8_t >(), context );
                        }
                    }
                    else
                    {
                        if( order == AriesOrderByType::ASC )
                        {
                            if( nullSmaller )
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< int8_t > >(), context );
                            else
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< int8_t > >(), context );
                        }
                        else
                        {
                            if( nullSmaller )
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< int8_t > >(), context );
                            else
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< int8_t > >(), context );
                        }
                    }
                    break;
                }
                // break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< int8_t > >(), context );
                            else
                                mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< int8_t > >(), context );
                        }
                        else
                            mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int8_t * )output,
                                    associated, indices_count, less_t< int8_t >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< int8_t > >(), context );
                            else
                                mergesort( ( int8_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int8_t >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< int8_t > >(), context );
                        }
                        else
                            mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int8_t * )output,
                                    associated, indices_count, greater_t< int8_t >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int8_t >* )output, associated, indices_count, less_t_null_smaller< nullable_type< int8_t > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int8_t >* )output, associated, indices_count, less_t_null_bigger< nullable_type< int8_t > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int8_t >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< int8_t > >(), context );
                        else
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int8_t >* )output, associated, indices_count, greater_t_null_bigger< nullable_type< int8_t > >(),
                                    context );
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int16_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int16_t >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< int16_t > >(), context );
                            else
                                mergesort( ( int16_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int16_t >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< int16_t > >(), context );
                        }
                        else
                            mergesort( ( int16_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int16_t * )output,
                                    associated, indices_count, less_t< int16_t >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int16_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int16_t >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< int16_t > >(), context );
                            else
                                mergesort( ( int16_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int16_t >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< int16_t > >(), context );
                        }
                        else
                            mergesort( ( int16_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int16_t * )output,
                                    associated, indices_count, greater_t< int16_t >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int16_t >* )output, associated, indices_count, less_t_null_smaller< nullable_type< int16_t > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int16_t >* )output, associated, indices_count, less_t_null_bigger< nullable_type< int16_t > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int16_t >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< int16_t > >(), context );
                        else
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int16_t >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< int16_t > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int32_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int32_t >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< int32_t > >(), context );
                            else
                                mergesort( ( int32_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int32_t >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< int32_t > >(), context );
                        }
                        else
                            mergesort( ( int32_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int32_t * )output,
                                    associated, indices_count, less_t< int32_t >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int32_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int32_t >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< int32_t > >(), context );
                            else
                                mergesort( ( int32_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int32_t >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< int32_t > >(), context );
                        }
                        else
                            mergesort( ( int32_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int32_t * )output,
                                    associated, indices_count, greater_t< int32_t >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int32_t >* )output, associated, indices_count, less_t_null_smaller< nullable_type< int32_t > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int32_t >* )output, associated, indices_count, less_t_null_bigger< nullable_type< int32_t > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int32_t >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< int32_t > >(), context );
                        else
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int32_t >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< int32_t > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int64_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int64_t >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< int64_t > >(), context );
                            else
                                mergesort( ( int64_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int64_t >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< int64_t > >(), context );
                        }
                        else
                            mergesort( ( int64_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int64_t * )output,
                                    associated, indices_count, less_t< int64_t >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( int64_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int64_t >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< int64_t > >(), context );
                            else
                                mergesort( ( int64_t ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< int64_t >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< int64_t > >(), context );
                        }
                        else
                            mergesort( ( int64_t** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( int64_t * )output,
                                    associated, indices_count, greater_t< int64_t >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int64_t >* )output, associated, indices_count, less_t_null_smaller< nullable_type< int64_t > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int64_t >* )output, associated, indices_count, less_t_null_bigger< nullable_type< int64_t > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int64_t >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< int64_t > >(), context );
                        else
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< int64_t >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< int64_t > >(), context );
                    }
                }
                break;
            }
//            case AriesValueType::UINT8:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint8_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint8_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
//            case AriesValueType::UINT16:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint16_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint16_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
//            case AriesValueType::UINT32:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint32_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint32_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
//            case AriesValueType::UINT64:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint64_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint64_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( float ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< float >* )output, associated, indices_count, less_t_null_smaller< nullable_type< float > >(),
                                        context );
                            else
                                mergesort( ( float ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< float >* )output, associated, indices_count, less_t_null_bigger< nullable_type< float > >(),
                                        context );
                        }
                        else
                            mergesort( ( float** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( float * )output, associated,
                                    indices_count, less_t< float >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( float ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< float >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< float > >(), context );
                            else
                                mergesort( ( float ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< float >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< float > >(), context );
                        }
                        else
                            mergesort( ( float** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( float * )output, associated,
                                    indices_count, greater_t< float >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< float >* )output, associated, indices_count, less_t_null_smaller< nullable_type< float > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< float >* )output, associated, indices_count, less_t_null_bigger< nullable_type< float > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< float >* )output, associated, indices_count, greater_t_null_smaller< nullable_type< float > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< float >* )output, associated, indices_count, greater_t_null_bigger< nullable_type< float > >(),
                                    context );
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( double ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< double >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< double > >(), context );
                            else
                                mergesort( ( double ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< double >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< double > >(), context );
                        }
                        else
                            mergesort( ( double** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( double * )output,
                                    associated, indices_count, less_t< double >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( double ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< double >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< double > >(), context );
                            else
                                mergesort( ( double ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< double >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< double > >(), context );
                        }
                        else
                            mergesort( ( double** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( double * )output,
                                    associated, indices_count, greater_t< double >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< double >* )output, associated, indices_count, less_t_null_smaller< nullable_type< double > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< double >* )output, associated, indices_count, less_t_null_bigger< nullable_type< double > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< double >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< double > >(), context );
                        else
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< double >* )output, associated, indices_count, greater_t_null_bigger< nullable_type< double > >(),
                                    context );
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( Decimal ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< Decimal >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< Decimal > >(), context );
                            else
                                mergesort( ( Decimal ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< Decimal >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< Decimal > >(), context );
                        }
                        else
                            mergesort( ( Decimal** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( Decimal * )output,
                                    associated, indices_count, less_t< Decimal >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( Decimal ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< Decimal >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< Decimal > >(), context );
                            else
                                mergesort( ( Decimal ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< Decimal >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< Decimal > >(), context );
                        }
                        else
                            mergesort( ( Decimal** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( Decimal * )output,
                                    associated, indices_count, greater_t< Decimal >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< Decimal >* )output, associated, indices_count, less_t_null_smaller< nullable_type< Decimal > >(),
                                    context );
                        else
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< Decimal >* )output, associated, indices_count, less_t_null_bigger< nullable_type< Decimal > >(),
                                    context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< Decimal >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< Decimal > >(), context );
                        else
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< Decimal >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< Decimal > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
                // break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesDate ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDate >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< AriesDate > >(), context );
                            else
                                mergesort( ( AriesDate ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDate >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< AriesDate > >(), context );
                        }
                        else
                            mergesort( ( AriesDate** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( AriesDate * )output,
                                    associated, indices_count, less_t< AriesDate >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesDate ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDate >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< AriesDate > >(), context );
                            else
                                mergesort( ( AriesDate ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDate >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< AriesDate > >(), context );
                        }
                        else
                            mergesort( ( AriesDate** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( AriesDate * )output,
                                    associated, indices_count, greater_t< AriesDate >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDate >* )output, associated, indices_count,
                                    less_t_null_smaller< nullable_type< AriesDate > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDate >* )output, associated, indices_count,
                                    less_t_null_bigger< nullable_type< AriesDate > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDate >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< AriesDate > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDate >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< AriesDate > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesDatetime ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< AriesDatetime > >(), context );
                            else
                                mergesort( ( AriesDatetime ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< AriesDatetime > >(), context );
                        }
                        else
                            mergesort( ( AriesDatetime** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( AriesDatetime * )output, associated, indices_count, less_t< AriesDatetime >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesDatetime ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< AriesDatetime > >(), context );
                            else
                                mergesort( ( AriesDatetime ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< AriesDatetime > >(), context );
                        }
                        else
                            mergesort( ( AriesDatetime** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( AriesDatetime * )output, associated, indices_count, greater_t< AriesDatetime >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                    less_t_null_smaller< nullable_type< AriesDatetime > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                    less_t_null_bigger< nullable_type< AriesDatetime > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< AriesDatetime > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesDatetime >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< AriesDatetime > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesTimestamp ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< AriesTimestamp > >(), context );
                            else
                                mergesort( ( AriesTimestamp ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< AriesTimestamp > >(), context );
                        }
                        else
                            mergesort( ( AriesTimestamp** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( AriesTimestamp * )output, associated, indices_count, less_t< AriesTimestamp >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesTimestamp ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< AriesTimestamp > >(), context );
                            else
                                mergesort( ( AriesTimestamp ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< AriesTimestamp > >(), context );
                        }
                        else
                            mergesort( ( AriesTimestamp** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( AriesTimestamp * )output, associated, indices_count, greater_t< AriesTimestamp >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                    less_t_null_smaller< nullable_type< AriesTimestamp > >(), context );
                        else
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                    less_t_null_bigger< nullable_type< AriesTimestamp > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< AriesTimestamp > >(), context );
                        else
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesTimestamp >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< AriesTimestamp > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesYear ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesYear >* )output, associated, indices_count,
                                        less_t_null_smaller< nullable_type< AriesYear > >(), context );
                            else
                                mergesort( ( AriesYear ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesYear >* )output, associated, indices_count,
                                        less_t_null_bigger< nullable_type< AriesYear > >(), context );
                        }
                        else
                            mergesort( ( AriesYear** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( AriesYear * )output,
                                    associated, indices_count, less_t< AriesYear >(), context );
                    }
                    else
                    {
                        if( indices_has_null )
                        {
                            if( nullSmaller )
                                mergesort( ( AriesYear ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesYear >* )output, associated, indices_count,
                                        greater_t_null_smaller< nullable_type< AriesYear > >(), context );
                            else
                                mergesort( ( AriesYear ** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                        ( nullable_type< AriesYear >* )output, associated, indices_count,
                                        greater_t_null_bigger< nullable_type< AriesYear > >(), context );
                        }
                        else
                            mergesort( ( AriesYear** )data, block_size_prefix_sum, block_count, indices, indices_has_null, ( AriesYear * )output,
                                    associated, indices_count, greater_t< AriesYear >(), context );
                    }
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesYear >* )output, associated, indices_count,
                                    less_t_null_smaller< nullable_type< AriesYear > >(), context );
                        else
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesYear >* )output, associated, indices_count,
                                    less_t_null_bigger< nullable_type< AriesYear > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesYear >* )output, associated, indices_count,
                                    greater_t_null_smaller< nullable_type< AriesYear > >(), context );
                        else
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count, indices, indices_has_null,
                                    ( nullable_type< AriesYear >* )output, associated, indices_count,
                                    greater_t_null_bigger< nullable_type< AriesYear > >(), context );
                    }
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
        }
        return result;
    }

template< typename type_t >
    AriesDataBufferSPtr sort_column_data( int8_t** data, int64_t* block_size_prefix_sum, int block_count, AriesColumnType type, size_t total_count,
            AriesOrderByType order, type_t* associated, context_t& context, bool nullSmaller = true )
    {
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( type, total_count );
        AriesValueType valueType = type.DataType.ValueType;
        int8_t* output = result->GetData();
        switch( valueType )
        {
            case AriesValueType::CHAR:
            {
                int len = type.GetDataTypeSize();
                if( type.DataType.Length > 1 )
                {
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
                }
                else
                {
                    if( !type.HasNull )
                    {
                        if( order == AriesOrderByType::ASC )
                            mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, ( int8_t * )output, associated, total_count,
                                    less_t< int8_t >(), context );
                        else
                            mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, ( int8_t * )output, associated, total_count,
                                    greater_t< int8_t >(), context );
                    }
                    else
                    {
                        if( order == AriesOrderByType::ASC )
                        {
                            if( nullSmaller )
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                        associated, total_count, less_t_null_smaller< nullable_type< int8_t > >(), context );
                            else
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                        associated, total_count, less_t_null_bigger< nullable_type< int8_t > >(), context );
                        }
                        else
                        {
                            if( nullSmaller )
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                        associated, total_count, greater_t_null_smaller< nullable_type< int8_t > >(), context );
                            else
                                mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                        associated, total_count, greater_t_null_bigger< nullable_type< int8_t > >(), context );
                        }
                    }
                    break;
                }
                // break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, ( int8_t * )output, associated, total_count,
                                less_t< int8_t >(), context );
                    else
                        mergesort( ( int8_t** )data, block_size_prefix_sum, block_count, ( int8_t * )output, associated, total_count,
                                greater_t< int8_t >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< int8_t > >(), context );
                        else
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< int8_t > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< int8_t > >(), context );
                        else
                            mergesort( ( nullable_type< int8_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int8_t >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< int8_t > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int16_t** )data, block_size_prefix_sum, block_count, ( int16_t * )output, associated, total_count,
                                less_t< int16_t >(), context );
                    else
                        mergesort( ( int16_t** )data, block_size_prefix_sum, block_count, ( int16_t * )output, associated, total_count,
                                greater_t< int16_t >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int16_t >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< int16_t > >(), context );
                        else
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int16_t >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< int16_t > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int16_t >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< int16_t > >(), context );
                        else
                            mergesort( ( nullable_type< int16_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int16_t >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< int16_t > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int32_t** )data, block_size_prefix_sum, block_count, ( int32_t * )output, associated, total_count,
                                less_t< int32_t >(), context );
                    else
                        mergesort( ( int32_t** )data, block_size_prefix_sum, block_count, ( int32_t * )output, associated, total_count,
                                greater_t< int32_t >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int32_t >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< int32_t > >(), context );
                        else
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int32_t >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< int32_t > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int32_t >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< int32_t > >(), context );
                        else
                            mergesort( ( nullable_type< int32_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int32_t >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< int32_t > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( int64_t** )data, block_size_prefix_sum, block_count, ( int64_t * )output, associated, total_count,
                                less_t< int64_t >(), context );
                    else
                        mergesort( ( int64_t** )data, block_size_prefix_sum, block_count, ( int64_t * )output, associated, total_count,
                                greater_t< int64_t >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int64_t >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< int64_t > >(), context );
                        else
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int64_t >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< int64_t > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int64_t >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< int64_t > >(), context );
                        else
                            mergesort( ( nullable_type< int64_t >** )data, block_size_prefix_sum, block_count, ( nullable_type< int64_t >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< int64_t > >(), context );
                    }
                }
                break;
            }
//            case AriesValueType::UINT8:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint8_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint8_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
//            case AriesValueType::UINT16:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint16_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint16_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
//            case AriesValueType::UINT32:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint32_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint32_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
//            case AriesValueType::UINT64:
//            {
//                if( !type.HasNull )
//                    radix_sort( ( uint64_t* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, bShuffle );
//                else
//                    radix_sort( ( nullable_type< uint64_t >* )data, associated, tupleNum, context, order == AriesOrderByType::ASC, nullSmaller,
//                            bShuffle );
//                break;
//            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( float** )data, block_size_prefix_sum, block_count, ( float * )output, associated, total_count, less_t< float >(),
                                context );
                    else
                        mergesort( ( float** )data, block_size_prefix_sum, block_count, ( float * )output, associated, total_count,
                                greater_t< float >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, ( nullable_type< float >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< float > >(), context );
                        else
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, ( nullable_type< float >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< float > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, ( nullable_type< float >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< float > >(), context );
                        else
                            mergesort( ( nullable_type< float >** )data, block_size_prefix_sum, block_count, ( nullable_type< float >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< float > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( double** )data, block_size_prefix_sum, block_count, ( double * )output, associated, total_count,
                                less_t< double >(), context );
                    else
                        mergesort( ( double** )data, block_size_prefix_sum, block_count, ( double * )output, associated, total_count,
                                greater_t< double >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, ( nullable_type< double >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< double > >(), context );
                        else
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, ( nullable_type< double >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< double > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, ( nullable_type< double >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< double > >(), context );
                        else
                            mergesort( ( nullable_type< double >** )data, block_size_prefix_sum, block_count, ( nullable_type< double >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< double > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( Decimal** )data, block_size_prefix_sum, block_count, ( Decimal * )output, associated, total_count,
                                less_t< Decimal >(), context );
                    else
                        mergesort( ( Decimal** )data, block_size_prefix_sum, block_count, ( Decimal * )output, associated, total_count,
                                greater_t< Decimal >(), context );
                }

                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, ( nullable_type< Decimal >* )output,
                                    associated, total_count, less_t_null_smaller< nullable_type< Decimal > >(), context );
                        else
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, ( nullable_type< Decimal >* )output,
                                    associated, total_count, less_t_null_bigger< nullable_type< Decimal > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, ( nullable_type< Decimal >* )output,
                                    associated, total_count, greater_t_null_smaller< nullable_type< Decimal > >(), context );
                        else
                            mergesort( ( nullable_type< Decimal >** )data, block_size_prefix_sum, block_count, ( nullable_type< Decimal >* )output,
                                    associated, total_count, greater_t_null_bigger< nullable_type< Decimal > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
                // break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( AriesDate** )data, block_size_prefix_sum, block_count, ( AriesDate * )output, associated, total_count,
                                less_t< AriesDate >(), context );
                    else
                        mergesort( ( AriesDate** )data, block_size_prefix_sum, block_count, ( AriesDate * )output, associated, total_count,
                                greater_t< AriesDate >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDate >* )output, associated, total_count,
                                    less_t_null_smaller< nullable_type< AriesDate > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDate >* )output, associated, total_count,
                                    less_t_null_bigger< nullable_type< AriesDate > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDate >* )output, associated, total_count,
                                    greater_t_null_smaller< nullable_type< AriesDate > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDate >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDate >* )output, associated, total_count,
                                    greater_t_null_bigger< nullable_type< AriesDate > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( AriesDatetime** )data, block_size_prefix_sum, block_count, ( AriesDatetime * )output, associated, total_count,
                                less_t< AriesDatetime >(), context );
                    else
                        mergesort( ( AriesDatetime** )data, block_size_prefix_sum, block_count, ( AriesDatetime * )output, associated, total_count,
                                greater_t< AriesDatetime >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDatetime >* )output, associated, total_count,
                                    less_t_null_smaller< nullable_type< AriesDatetime > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDatetime >* )output, associated, total_count,
                                    less_t_null_bigger< nullable_type< AriesDatetime > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDatetime >* )output, associated, total_count,
                                    greater_t_null_smaller< nullable_type< AriesDatetime > >(), context );
                        else
                            mergesort( ( nullable_type< AriesDatetime >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesDatetime >* )output, associated, total_count,
                                    greater_t_null_bigger< nullable_type< AriesDatetime > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( AriesTimestamp** )data, block_size_prefix_sum, block_count, ( AriesTimestamp * )output, associated, total_count,
                                less_t< AriesTimestamp >(), context );
                    else
                        mergesort( ( AriesTimestamp** )data, block_size_prefix_sum, block_count, ( AriesTimestamp * )output, associated, total_count,
                                greater_t< AriesTimestamp >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesTimestamp >* )output, associated, total_count,
                                    less_t_null_smaller< nullable_type< AriesTimestamp > >(), context );
                        else
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesTimestamp >* )output, associated, total_count,
                                    less_t_null_bigger< nullable_type< AriesTimestamp > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesTimestamp >* )output, associated, total_count,
                                    greater_t_null_smaller< nullable_type< AriesTimestamp > >(), context );
                        else
                            mergesort( ( nullable_type< AriesTimestamp >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesTimestamp >* )output, associated, total_count,
                                    greater_t_null_bigger< nullable_type< AriesTimestamp > >(), context );
                    }
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                {
                    if( order == AriesOrderByType::ASC )
                        mergesort( ( AriesYear** )data, block_size_prefix_sum, block_count, ( AriesYear * )output, associated, total_count,
                                less_t< AriesYear >(), context );
                    else
                        mergesort( ( AriesYear** )data, block_size_prefix_sum, block_count, ( AriesYear * )output, associated, total_count,
                                greater_t< AriesYear >(), context );
                }
                else
                {
                    if( order == AriesOrderByType::ASC )
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesYear >* )output, associated, total_count,
                                    less_t_null_smaller< nullable_type< AriesYear > >(), context );
                        else
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesYear >* )output, associated, total_count,
                                    less_t_null_bigger< nullable_type< AriesYear > >(), context );
                    }
                    else
                    {
                        if( nullSmaller )
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesYear >* )output, associated, total_count,
                                    greater_t_null_smaller< nullable_type< AriesYear > >(), context );
                        else
                            mergesort( ( nullable_type< AriesYear >** )data, block_size_prefix_sum, block_count,
                                    ( nullable_type< AriesYear >* )output, associated, total_count,
                                    greater_t_null_bigger< nullable_type< AriesYear > >(), context );
                    }
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + aries_engine::GetValueTypeAsString( type ) + " in column" );
        }
        return result;
    }

END_ARIES_ACC_NAMESPACE
