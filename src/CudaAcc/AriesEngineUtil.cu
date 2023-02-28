/*
 * AriesEngineUtil.cu
 *
 *  Created on: Jun 20, 2019
 *      Author: lichi
 */

#include "AriesEngineUtil.h"
#include <unordered_map>

BEGIN_ARIES_ACC_NAMESPACE

    struct ColumnComp
    {
        int8_t* Data;
        AriesColumnType ColumnType;
    };

    __device__ void create_eq_comparator_to_find_group_boundary( const ColumnComp* columns, int columnCount, IComparableColumn** output )
    {
        for( int i = 0; i < columnCount; ++i )
        {
            auto type = columns[i].ColumnType;
            switch( type.DataType.ValueType )
            {
                case AriesValueType::CHAR:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< char, equal_to_t_str_null_eq< false, false >, false >( ( const char* )columns[i].Data,
                                columns[i].ColumnType.GetDataTypeSize(), equal_to_t_str_null_eq< false, false >() );
                    else
                        output[i] = new ComparableColumn< char, equal_to_t_str_null_eq< true, true >, false >( ( const char* )columns[i].Data,
                                columns[i].ColumnType.GetDataTypeSize(), equal_to_t_str_null_eq< true, true >() );
                    break;
                }
                case AriesValueType::COMPACT_DECIMAL:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< char, equal_to_t_CompactDecimal_null_eq< false >, false >( ( const char* )columns[i].Data,
                                columns[i].ColumnType.GetDataTypeSize(),
                                equal_to_t_CompactDecimal_null_eq< false >( type.DataType.Precision, type.DataType.Scale ) );
                    else
                        output[i] = new ComparableColumn< char, equal_to_t_CompactDecimal_null_eq< true >, false >( ( const char* )columns[i].Data,
                                columns[i].ColumnType.GetDataTypeSize(),
                                equal_to_t_CompactDecimal_null_eq< true >( type.DataType.Precision, type.DataType.Scale ) );
                    break;
                }
                case AriesValueType::BOOL:
                case AriesValueType::INT8:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< int8_t, equal_to_t_null_eq< int8_t > >( ( const int8_t* )columns[i].Data,
                                equal_to_t_null_eq< int8_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< int8_t >, equal_to_t_null_eq< nullable_type< int8_t > > >(
                                ( const nullable_type< int8_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< int8_t > >() );
                    break;
                }
                case AriesValueType::INT16:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< int16_t, equal_to_t_null_eq< int16_t > >( ( const int16_t* )columns[i].Data,
                                equal_to_t_null_eq< int16_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< int16_t >, equal_to_t_null_eq< nullable_type< int16_t > > >(
                                ( const nullable_type< int16_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< int16_t > >() );
                    break;
                }
                case AriesValueType::INT32:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< int32_t, equal_to_t_null_eq< int32_t > >( ( const int32_t* )columns[i].Data,
                                equal_to_t_null_eq< int32_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< int32_t >, equal_to_t_null_eq< nullable_type< int32_t > > >(
                                ( const nullable_type< int32_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< int32_t > >() );
                    break;
                }
                case AriesValueType::INT64:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< int64_t, equal_to_t_null_eq< int64_t > >( ( const int64_t* )columns[i].Data,
                                equal_to_t_null_eq< int64_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< int64_t >, equal_to_t_null_eq< nullable_type< int64_t > > >(
                                ( const nullable_type< int64_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< int64_t > >() );
                    break;
                }
                case AriesValueType::UINT8:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< uint8_t, equal_to_t_null_eq< uint8_t > >( ( const uint8_t* )columns[i].Data,
                                equal_to_t_null_eq< uint8_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< uint8_t >, equal_to_t_null_eq< nullable_type< uint8_t > > >(
                                ( const nullable_type< uint8_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< uint8_t > >() );
                    break;
                }
                case AriesValueType::UINT16:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< uint16_t, equal_to_t_null_eq< uint16_t > >( ( const uint16_t* )columns[i].Data,
                                equal_to_t_null_eq< uint16_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< uint16_t >, equal_to_t_null_eq< nullable_type< uint16_t > > >(
                                ( const nullable_type< uint16_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< uint16_t > >() );
                    break;
                }
                case AriesValueType::UINT32:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< uint32_t, equal_to_t_null_eq< uint32_t > >( ( const uint32_t* )columns[i].Data,
                                equal_to_t_null_eq< uint32_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< uint32_t >, equal_to_t_null_eq< nullable_type< uint32_t > > >(
                                ( const nullable_type< uint32_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< uint32_t > >() );
                    break;
                }
                case AriesValueType::UINT64:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< uint64_t, equal_to_t_null_eq< uint64_t > >( ( const uint64_t* )columns[i].Data,
                                equal_to_t_null_eq< uint64_t >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< uint64_t >, equal_to_t_null_eq< nullable_type< uint64_t > > >(
                                ( const nullable_type< uint64_t >* )columns[i].Data, equal_to_t_null_eq< nullable_type< uint64_t > >() );
                    break;
                }
                case AriesValueType::FLOAT:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< float, equal_to_t_null_eq< float > >( ( const float* )columns[i].Data,
                                equal_to_t_null_eq< float >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< float >, equal_to_t_null_eq< nullable_type< float > > >(
                                ( const nullable_type< float >* )columns[i].Data, equal_to_t_null_eq< nullable_type< float > >() );
                    break;
                }
                case AriesValueType::DOUBLE:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< double, equal_to_t_null_eq< double > >( ( const double* )columns[i].Data,
                                equal_to_t_null_eq< double >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< double >, equal_to_t_null_eq< nullable_type< double > > >(
                                ( const nullable_type< double >* )columns[i].Data, equal_to_t_null_eq< nullable_type< double > >() );
                    break;
                }
                case AriesValueType::DECIMAL:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< Decimal, equal_to_t_null_eq< Decimal > >( ( const Decimal* )columns[i].Data,
                                equal_to_t_null_eq< Decimal >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< Decimal >, equal_to_t_null_eq< nullable_type< Decimal > > >(
                                ( const nullable_type< Decimal >* )columns[i].Data, equal_to_t_null_eq< nullable_type< Decimal > >() );
                    break;
                }
                case AriesValueType::DATE:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< AriesDate, equal_to_t_null_eq< AriesDate > >( ( const AriesDate* )columns[i].Data,
                                equal_to_t_null_eq< AriesDate >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< AriesDate >, equal_to_t_null_eq< nullable_type< AriesDate > > >(
                                ( const nullable_type< AriesDate >* )columns[i].Data, equal_to_t_null_eq< nullable_type< AriesDate > >() );
                    break;
                }
                case AriesValueType::DATETIME:
                {
                    if( !type.HasNull )
                        output[i] = new ComparableColumn< AriesDatetime, equal_to_t_null_eq< AriesDatetime > >(
                                ( const AriesDatetime* )columns[i].Data, equal_to_t_null_eq< AriesDatetime >() );
                    else
                        output[i] = new ComparableColumn< nullable_type< AriesDatetime >, equal_to_t_null_eq< nullable_type< AriesDatetime > > >(
                                ( const nullable_type< AriesDatetime >* )columns[i].Data, equal_to_t_null_eq< nullable_type< AriesDatetime > >() );
                    break;
                }
                case AriesValueType::TIMESTAMP:
                {
                    if( !type.HasNull )
                        output[ i ] = new ComparableColumn< AriesTimestamp, equal_to_t_null_eq< AriesTimestamp > >(
                                ( const AriesTimestamp* )columns[ i ].Data, equal_to_t_null_eq< AriesTimestamp >() );
                    else
                        output[ i ] = new ComparableColumn< nullable_type< AriesTimestamp >, equal_to_t_null_eq< nullable_type< AriesTimestamp > > >(
                                ( const nullable_type< AriesTimestamp >* )columns[ i ].Data, equal_to_t_null_eq< nullable_type< AriesTimestamp > >() );
                    break;
                }
                case AriesValueType::YEAR:
                {
                    if( !type.HasNull )
                        output[ i ] = new ComparableColumn< AriesYear, equal_to_t_null_eq< AriesYear > >(
                                ( const AriesYear* )columns[ i ].Data, equal_to_t_null_eq< AriesYear >() );
                    else
                        output[ i ] = new ComparableColumn< nullable_type< AriesYear >, equal_to_t_null_eq< nullable_type< AriesYear > > >(
                                ( const nullable_type< AriesYear >* )columns[ i ].Data, equal_to_t_null_eq< nullable_type< AriesYear > >() );
                    break;
                }
                default:
                    assert( 0 ); //FIXME need support all data types.
                    break;
            }
        }
    }

    void find_group_bounds( const std::vector< AriesDataBufferSPtr >& columns, const int *associated, int *groups, context_t& context )
    {
        ARIES_ASSERT( !columns.empty() && associated && groups,
                "columns.empty(): " + to_string(columns.empty()) + "associated is nullptr: " + to_string(!!associated) + "groups is nullptr: " + to_string(!!groups) );
        size_t comp_count = columns.size();

        AriesManagedArray< ColumnComp > pairs( comp_count );
        ColumnComp* newPair = nullptr;
        if( comp_count > 0 )
        {
            newPair = pairs.GetData();
            for( int i = 0; i < comp_count; ++i )
            {
                auto& dst = newPair[i];
                const auto& col = columns[i];
                dst.Data = col->GetData();
                dst.ColumnType = col->GetDataType();
            }
        }

        AriesArray< IComparableColumn* > compArray( comp_count );
        pairs.PrefetchToGpu();
        IComparableColumn** comparators = compArray.GetData();
        auto k1 = [=] ARIES_DEVICE(int index)
        {
            create_eq_comparator_to_find_group_boundary( newPair, comp_count, comparators );
        };
        transform< launch_box_t< arch_52_cta< 32, 1 > > >( k1, 1, context );

        size_t tupleNum = columns[0]->GetItemCount();
        //groups[0] = 0;
        auto k2 = [=] ARIES_DEVICE(int index)
        {
            int flag = 0;
            for( int i = 0; i < comp_count; ++i )
            {
                if( !comparators[i]->compare( associated[index], associated[index + 1] ).is_true() )
                {
                    flag = 1;
                    break;
                }
            }
            groups[ index + 1 ] = flag;
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k2, tupleNum - 1, context );

        auto k3 = [=] ARIES_DEVICE(int index)
        {
            delete comparators[index];
        };
        transform< launch_box_t< arch_52_cta< 32, 1 > > >( k3, comp_count, context );

        context.synchronize();
    }
    AriesInt32ArraySPtr ConvertToOriginalFlag( const AriesInt32ArraySPtr& flags, const AriesInt32ArraySPtr& oldOriginal,
            const AriesInt32ArraySPtr& oldPsum, context_t& context )
    {
        size_t count = oldOriginal->GetItemCount();
        ARIES_ASSERT( count > 0, "count: " + to_string(count) );
        AriesInt32ArraySPtr result = std::make_shared< AriesInt32Array >();
        result->AllocArray( count, true );
        const int32_t* flagData = flags->GetData();
        const int32_t* originalData = oldOriginal->GetData();
        const int32_t* psumData = oldPsum->GetData();
        int32_t* pOutput = result->GetData();

        auto k = [=] ARIES_DEVICE(int index)
        {
            if( originalData[ index ] )
            pOutput[ index ] = flagData[ psumData[ index ] ];
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        return result;
    }
    AriesInt32ArraySPtr FindBoundArray( const AriesUInt32ArraySPtr& data, context_t& context )
    {
        size_t count = data->GetItemCount();
        ARIES_ASSERT( count > 0, "count: " + to_string(count) );
        AriesInt32ArraySPtr result = std::make_shared< AriesInt32Array >();
        result->AllocArray( count, true );
        const uint32_t* inputData = data->GetData();
        int32_t* pOutput = result->GetData();
        pOutput[0] = 0;
        auto k = [=] ARIES_DEVICE(int index)
        {
            pOutput[ index + 1 ] = ( inputData[ index ] != inputData[ index + 1 ] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count - 1, context );
        context.synchronize();
        return result;
    }

    AriesUInt32ArraySPtr LoadDataAsUInt32( const AriesDataBufferSPtr& column, const AriesInt32ArraySPtr& flags, const AriesInt32ArraySPtr& psum,
            int offset, size_t count, context_t& context )
    {
        ARIES_ASSERT( count > 0, "count: " + to_string(count) );
        AriesUInt32ArraySPtr result = std::make_shared< AriesUInt32Array >( count );
        const int8_t* columnData = column->GetData();
        size_t dataTypeSize = column->GetDataType().GetDataTypeSize();
        size_t tupleNum = column->GetItemCount();
        const int32_t* flagData = flags->GetData();
        const int32_t* psumData = psum->GetData();
        size_t copySize = std::min( dataTypeSize - offset, sizeof(uint32_t) );
        uint32_t* pOutput = result->GetData();
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( flagData[ index ] )
            {
                uint32_t temp = 0;
                memcpy( &temp, columnData + dataTypeSize * index + offset, copySize );
                pOutput[ psumData[ index ] ] = __byte_perm (temp, 0, 0x0123);
            }
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
        context.synchronize();
        return result;
    }
    AriesUInt32ArraySPtr LoadDataAsUInt32Ex( const AriesDataBufferSPtr& column, const AriesInt32ArraySPtr& associated, int offset,
            context_t& context )
    {
        size_t tupleNum = associated->GetItemCount();
        ARIES_ASSERT( tupleNum > 0, "tupleNum: " + to_string(tupleNum) );
        AriesUInt32ArraySPtr result = std::make_shared< AriesUInt32Array >( tupleNum );
        const int8_t* columnData = column->GetData();
        size_t dataTypeSize = column->GetDataType().GetDataTypeSize();
        const int32_t* associatedData = associated->GetData();
        size_t copySize = std::min( dataTypeSize - offset, sizeof(uint32_t) );
        uint32_t* pOutput = result->GetData();
        auto k = [=] ARIES_DEVICE(int index)
        {
            uint32_t temp = 0;
            memcpy( &temp, columnData + dataTypeSize * associatedData[ index ] + offset, copySize );
            pOutput[ index ] = __byte_perm (temp, 0, 0x0123);
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
        context.synchronize();
        return result;
    }

END_ARIES_ACC_NAMESPACE
