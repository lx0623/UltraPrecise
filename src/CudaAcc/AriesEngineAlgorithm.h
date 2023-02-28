/*
 * AriesEngineAlgorithm.h
 *
 *  Created on: Jun 19, 2019
 *      Author: lichi
 */

#ifndef ARIESENGINEALGORITHM_H_
#define ARIESENGINEALGORITHM_H_
#include <vector>
#include "AriesEngineUtil.h"
#include "AriesEngineException.h"
#include "AriesEngine/AriesUtil.h"
#include "AriesSqlOperator_common.h"

using namespace aries_engine;
BEGIN_ARIES_ACC_NAMESPACE

template< typename index_t >
void shuffle_column_data( const int8_t* data, AriesColumnType type, size_t tupleNum, const index_t* indices, int8_t* data_output,
        context_t& context )
{
    AriesValueType valueType = type.DataType.ValueType;

    switch( valueType )
    {
        case AriesValueType::CHAR:
        {
            if( type.DataType.Length > 1 )
                shuffle_by_index( ( char* )data, type.GetDataTypeSize(), tupleNum, indices, ( char* )data_output, context );
            else
            {
                if( !type.HasNull )
                    shuffle_by_index( ( int8_t* )data, tupleNum, indices, ( int8_t* )data_output, context );
                else
                    shuffle_by_index( ( nullable_type< int8_t >* )data, tupleNum, indices, ( nullable_type< int8_t >* )data_output, context );
            }
            break;
        }
        case AriesValueType::COMPACT_DECIMAL:
        {
            shuffle_by_index( ( char* )data, type.GetDataTypeSize(), tupleNum, indices, ( char* )data_output, context );
            break;
        }
        case AriesValueType::BOOL:
        case AriesValueType::INT8:
        {
            if( !type.HasNull )
                shuffle_by_index( ( int8_t* )data, tupleNum, indices, ( int8_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< int8_t >* )data, tupleNum, indices, ( nullable_type< int8_t >* )data_output, context );
            break;
        }
        case AriesValueType::INT16:
        {
            if( !type.HasNull )
                shuffle_by_index( ( int16_t* )data, tupleNum, indices, ( int16_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< int16_t >* )data, tupleNum, indices, ( nullable_type< int16_t >* )data_output, context );
            break;
        }
        case AriesValueType::INT32:
        {
            if( !type.HasNull )
                shuffle_by_index( ( int32_t* )data, tupleNum, indices, ( int32_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< int32_t >* )data, tupleNum, indices, ( nullable_type< int32_t >* )data_output, context );
            break;
        }
        case AriesValueType::INT64:
        {
            if( !type.HasNull )
                shuffle_by_index( ( int64_t* )data, tupleNum, indices, ( int64_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< int64_t >* )data, tupleNum, indices, ( nullable_type< int64_t >* )data_output, context );
            break;
        }
        case AriesValueType::UINT8:
        {
            if( !type.HasNull )
                shuffle_by_index( ( uint8_t* )data, tupleNum, indices, ( uint8_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< uint8_t >* )data, tupleNum, indices, ( nullable_type< uint8_t >* )data_output, context );
            break;
        }
        case AriesValueType::UINT16:
        {
            if( !type.HasNull )
                shuffle_by_index( ( uint16_t* )data, tupleNum, indices, ( uint16_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< uint16_t >* )data, tupleNum, indices, ( nullable_type< uint16_t >* )data_output, context );
            break;
        }
        case AriesValueType::UINT32:
        {
            if( !type.HasNull )
                shuffle_by_index( ( uint32_t* )data, tupleNum, indices, ( uint32_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< uint32_t >* )data, tupleNum, indices, ( nullable_type< uint32_t >* )data_output, context );
            break;
        }
        case AriesValueType::UINT64:
        {
            if( !type.HasNull )
                shuffle_by_index( ( uint64_t* )data, tupleNum, indices, ( uint64_t* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< uint64_t >* )data, tupleNum, indices, ( nullable_type< uint64_t >* )data_output, context );
            break;
        }
        case AriesValueType::FLOAT:
        {
            if( !type.HasNull )
                shuffle_by_index( ( float* )data, tupleNum, indices, ( float* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< float >* )data, tupleNum, indices, ( nullable_type< float >* )data_output, context );
            break;
        }
        case AriesValueType::DOUBLE:
        {
            if( !type.HasNull )
                shuffle_by_index( ( double* )data, tupleNum, indices, ( double* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< double >* )data, tupleNum, indices, ( nullable_type< double >* )data_output, context );
            break;
        }
        case AriesValueType::DECIMAL:
        {
            if( !type.HasNull )
                shuffle_by_index( ( Decimal* )data, tupleNum, indices, ( Decimal* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< Decimal >* )data, tupleNum, indices, ( nullable_type< Decimal >* )data_output, context );
            break;
        }
        case AriesValueType::DATE:
        {
            if( !type.HasNull )
                shuffle_by_index( ( AriesDate* )data, tupleNum, indices, ( AriesDate* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< AriesDate >* )data, tupleNum, indices, ( nullable_type< AriesDate >* )data_output, context );
            break;
        }
        case AriesValueType::DATETIME:
        {
            if( !type.HasNull )
                shuffle_by_index( ( AriesDatetime* )data, tupleNum, indices, ( AriesDatetime* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< AriesDatetime >* )data, tupleNum, indices, ( nullable_type< AriesDatetime >* )data_output,
                        context );
            break;
        }
        case AriesValueType::TIMESTAMP:
        {
            if( !type.HasNull )
                shuffle_by_index( ( AriesTimestamp* )data, tupleNum, indices, ( AriesTimestamp* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< AriesTimestamp >* )data, tupleNum, indices, ( nullable_type< AriesTimestamp >* )data_output,
                        context );
            break;
        }
        case AriesValueType::YEAR:
        {
            if( !type.HasNull )
                shuffle_by_index( ( AriesYear* )data, tupleNum, indices, ( AriesYear* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< AriesYear >* )data, tupleNum, indices, ( nullable_type< AriesYear >* )data_output, context );
            break;
        }
        case AriesValueType::TIME:
        {
            if( !type.HasNull )
                shuffle_by_index( ( AriesTime* )data, tupleNum, indices, ( AriesTime* )data_output, context );
            else
                shuffle_by_index( ( nullable_type< AriesTime >* )data, tupleNum, indices, ( nullable_type< AriesTime >* )data_output, context );
            break;
        }
        default:
            //FIXME need support all data types.
            assert( 0 );
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "shuffling data type " + GetValueTypeAsString( type ) + " in column" );
    }
}    template< typename type_t, typename type_u >
    AriesDataBufferSPtr aggregate_sum( const type_t* data, size_t tupleNum, const type_u* associated, const type_u* groups, size_t groupCount,
            context_t& context, bool bDistinct )
    {
        using output_t = nullable_type<typename std::common_type<type_t, int64_t>::type >;
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >(
                ColumnTypeConverter< typename std::common_type< type_t, int64_t >::type >( true ).ColumnType, groupCount );
        result->PrefetchToGpu();
        if( bDistinct )
        {
            // create a temp column for segsort
            mem_t< type_t > tmp( tupleNum );
            type_t* pData = tmp.data();
            //memcpy( pData, data, tupleNum * sizeof( type_t ) );
            cudaMemcpy( pData, data, tupleNum * sizeof(type_t), cudaMemcpyDefault );
            segmented_sort< launch_box_t< arch_52_cta< 32, 3 > > >( pData, tupleNum, groups, groupCount, less_t_null_smaller< type_t >(), context );

            not_equal_to_t< type_t > func;
            lbs_segreduce< launch_box_t< arch_52_cta< 32, 5 > > >( [=]ARIES_LAMBDA( int index, int seg, int rank )
            {
                return index == groups[ seg ] || func( pData[ index - 1 ], pData[ index ] ) ? pData[ index ] : type_t();
            }, tupleNum, groups, groupCount, ( output_t* )result->GetData(), agg_sum_t< output_t >(), output_t(), context );
        }
        else
        {
            transform_segreduce( [=]ARIES_LAMBDA(int index)
            {
                return data[ associated[ index ] ];
            }, tupleNum, groups, groupCount, ( output_t* )( result->GetData() ), agg_sum_t< output_t >(), output_t(), context );
        }
        return result;
    }

    template< typename type_t, typename type_u >
    AriesDataBufferSPtr aggregate_sum_for_count( const type_t* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
        using output_t = typename std::common_type<type_t, int64_t>::type;
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( ColumnTypeConverter< output_t >( false ).ColumnType, groupCount );
        result->PrefetchToGpu();
        if( bDistinct )
        {
            // create a temp column for segsort
            mem_t< type_t > tmp( tupleNum );
            type_t* pData = tmp.data();
            //memcpy( pData, data, tupleNum * sizeof( type_t ) );
            cudaMemcpy( pData, data, tupleNum * sizeof(type_t), cudaMemcpyDefault );
            segmented_sort< launch_box_t< arch_52_cta< 32, 3 > > >( pData, tupleNum, groups, groupCount, less_t_null_smaller< type_t >(), context );

            not_equal_to_t< type_t > func;
            lbs_segreduce< launch_box_t< arch_52_cta< 32, 5 > > >( [=]ARIES_LAMBDA( int index, int seg, int rank )
            {
                return index == groups[ seg ] || func( pData[ index - 1 ], pData[ index ] ) ? pData[ index ] : type_t();
            }, tupleNum, groups, groupCount, ( output_t* )result->GetData(), agg_sum_t< output_t >(), output_t(), context );
        }
        else
        {
            transform_segreduce( [=]ARIES_LAMBDA(int index)
            {
                return data[ associated[ index ] ];
            }, tupleNum, groups, groupCount, ( output_t* )( result->GetData() ), agg_sum_t< output_t >(), output_t(), context );
        }
        return result;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum_for_count( const AriesDate* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDate
        return nullptr;
    }
    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum_for_count( const AriesDatetime* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDatetime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum_for_count( const AriesTimestamp* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesTimestamp
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum_for_count( const AriesTime* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesTime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum_for_count( const AriesYear* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesTime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const AriesDate* data, size_t tupleNum, const type_u* associated, const type_u* groups, size_t groupCount,
            context_t& context, bool bDistinct )
    {
// no sum for AriesDate
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const AriesDatetime* data, size_t tupleNum, const type_u* associated, const type_u* groups, size_t groupCount,
            context_t& context, bool bDistinct )
    {
// no sum for AriesDatetime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const AriesTimestamp* data, size_t tupleNum, const type_u* associated, const type_u* groups, size_t groupCount,
            context_t& context, bool bDistinct )
    {
// no sum for AriesTimestamp
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const AriesTime* data, size_t tupleNum, const type_u* associated, const type_u* groups, size_t groupCount,
            context_t& context, bool bDistinct )
    {
// no sum for AriesTime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const AriesYear* data, size_t tupleNum, const type_u* associated, const type_u* groups, size_t groupCount,
            context_t& context, bool bDistinct )
    {
// no sum for AriesYear
        return nullptr;
    }

    template< typename type_t, typename type_u >
    AriesDataBufferSPtr aggregate_column( const type_t* data, AriesColumnType columnType, size_t tupleNum, AriesAggFunctionType aggType,
            const type_u* associated, const type_u* groups, size_t groupCount, context_t& context, bool bDistinct, bool bSumForCount = false )
    {
        AriesDataBufferSPtr result;
        switch( aggType )
        {
            case AriesAggFunctionType::COUNT:
            {
                result = std::make_shared< AriesDataBuffer >( AriesColumnType
                {
                { AriesValueType::INT64, 1 }, false, false }, groupCount );
                result->PrefetchToGpu();
                if( bDistinct )
                {
                    // create a temp column for segsort
                    mem_t< type_t > tmp( tupleNum );
                    type_t* pData = tmp.data();
                    //memcpy( pData, data, tupleNum * sizeof( type_t ) );
                    cudaMemcpy( pData, data, tupleNum * sizeof(type_t), cudaMemcpyDefault );
                    segmented_sort< launch_box_t< arch_52_cta< 32, 3 > > >( pData, tupleNum, groups, groupCount, less_t_null_smaller< type_t >(),
                            context );

                    not_equal_to_t< type_t > func;
                    lbs_segreduce< launch_box_t< arch_52_cta< 64, 5 > > >( [=]ARIES_LAMBDA( int index, int seg, int rank )
                    {
                        return index == groups[ seg ] || func( pData[ index - 1 ], pData[ index ] );
                    }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                }
                else
                {
                    transform_segreduce( [=]ARIES_LAMBDA(int index)
                    {
                        return 1;
                    }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                }
                break;
            }
            case AriesAggFunctionType::MAX:
            {
                columnType.HasNull = true;
                result = std::make_shared< AriesDataBuffer >( columnType, groupCount );
                result->PrefetchToGpu();

                transform_segreduce( [=]ARIES_LAMBDA(int index)
                {
                    return data[ associated[ index ] ];
                }, tupleNum, groups, groupCount, ( nullable_type< type_t >* )( result->GetData() ), agg_max_t< nullable_type< type_t > >(),
                        nullable_type< type_t >( std::numeric_limits< nullable_type< type_t > >::min() ), context );
                break;
            }
            case AriesAggFunctionType::MIN:
            {
                columnType.HasNull = true;
                result = std::make_shared< AriesDataBuffer >( columnType, groupCount );
                result->PrefetchToGpu();
                transform_segreduce( [=]ARIES_LAMBDA(int index)
                {
                    return data[ associated[ index ] ];
                }, tupleNum, groups, groupCount, ( nullable_type< type_t >* )( result->GetData() ), agg_min_t< nullable_type< type_t > >(),
                        nullable_type< type_t >( std::numeric_limits< nullable_type< type_t > >::max() ), context );
                break;
            }
            case AriesAggFunctionType::SUM:
            {
                if( bSumForCount )
                    return aggregate_sum_for_count( data, tupleNum, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_sum( data, tupleNum, associated, groups, groupCount, context, bDistinct );
            }
            case AriesAggFunctionType::AVG:
            {
                // AVG will be replaced by sum and count for data partition process
                ARIES_ASSERT( 0, "AVG will be replaced by sum and count for data partition process" );
            }
            case AriesAggFunctionType::ANY_VALUE:
            {
                ARIES_ASSERT( 0, "ANY_VALUE should not be here" );
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "agg function " + GetAriesAggFunctionTypeName( aggType ) );
        }
        return result;
    }

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    AriesDataBufferSPtr aggregate_sum( const type_nullable< type_t >* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
        using output_t = typename std::common_type<type_t, int64_t>::type;
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( ColumnTypeConverter< output_t >( true ).ColumnType, groupCount );
        result->PrefetchToGpu();
        if( bDistinct )
        {
            // create a temp column for segsort
            mem_t< type_nullable< type_t > > tmp( tupleNum );
            type_nullable< type_t >* pData = tmp.data();
            //memcpy( pData, data, tupleNum * sizeof( type_nullable< type_t > ) );
            cudaMemcpy( pData, data, tupleNum * sizeof(type_nullable< type_t > ), cudaMemcpyDefault );
            segmented_sort< launch_box_t< arch_52_cta< 32, 5 > > >( pData, tupleNum, groups, groupCount,
                    less_t_null_smaller< type_nullable< type_t > >(), context );

            not_equal_to_t< type_nullable< type_t > > func;
            lbs_segreduce< launch_box_t< arch_52_cta< 32, 5 > > >( [=]ARIES_LAMBDA( int index, int seg, int rank )
            {
                return index == groups[ seg ] || func( pData[ index - 1 ], pData[ index ] ) ? pData[ index ] : type_nullable< type_t >();
            }, tupleNum, groups, groupCount, ( type_nullable< output_t >* )result->GetData(), agg_sum_t< type_nullable< output_t > >(),
                    type_nullable< output_t >(), context );
        }
        else
        {
            transform_segreduce( [=]ARIES_LAMBDA(int index)
            {
                return data[ associated[ index ] ];
            }, tupleNum, groups, groupCount, ( type_nullable< output_t >* )( result->GetData() ), agg_sum_t< type_nullable< output_t > >(),
                    type_nullable< output_t >(), context );
        }
        return result;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const nullable_type< AriesDate >* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDate
        return nullptr;
    }
    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const nullable_type< AriesDatetime >* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDatetime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const nullable_type< AriesTimestamp >* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDatetime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const nullable_type< AriesTime >* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDatetime
        return nullptr;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_sum( const nullable_type< AriesYear >* data, size_t tupleNum, const type_u* associated, const type_u* groups,
            size_t groupCount, context_t& context, bool bDistinct )
    {
// no sum for AriesDatetime
        return nullptr;
    }

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    AriesDataBufferSPtr aggregate_column( const type_nullable< type_t >* data, AriesColumnType columnType, size_t tupleNum,
            AriesAggFunctionType aggType, const type_u* associated, const type_u* groups, size_t groupCount, context_t& context, bool bDistinct,
            bool bSumForCount = false )
    {
        AriesDataBufferSPtr result;
        switch( aggType )
        {
            case AriesAggFunctionType::COUNT:
            {
                result = std::make_shared< AriesDataBuffer >( AriesColumnType
                {
                { AriesValueType::INT64, 1 }, false, false }, groupCount );
                result->PrefetchToGpu();
                if( bDistinct )
                {
                    // create a temp column for segsort
                    mem_t< type_nullable< type_t > > tmp( tupleNum );
                    type_nullable< type_t >* pData = tmp.data();
                    //memcpy( pData, data, tupleNum * sizeof(type_nullable< type_t >) );
                    cudaMemcpy( pData, data, tupleNum * sizeof(type_nullable< type_t > ), cudaMemcpyDefault );
                    segmented_sort< launch_box_t< arch_52_cta< 32, 5 > > >( pData, tupleNum, groups, groupCount,
                            less_t_null_smaller< type_nullable< type_t > >(), context );

                    equal_to_t_null_eq< type_nullable< type_t > > func;
                    lbs_segreduce< launch_box_t< arch_52_cta< 128, 5 > > >( [=]ARIES_LAMBDA( int index, int seg, int rank )
                    {
                        auto cur = pData[ index ];
                        if( cur.flag )
                        return (int)( index == groups[ seg ] || !func( pData[ index - 1 ], pData[ index ] ) );
                        else
                        return 0;
                    }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                }
                else
                {
                    transform_segreduce( [=]ARIES_LAMBDA(int index)
                    {
                        return data[ associated[ index ] ].flag ? 1 : 0;
                    }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                }

                break;
            }
            case AriesAggFunctionType::MAX:
            {
                result = std::make_shared< AriesDataBuffer >( columnType, groupCount );
                result->PrefetchToGpu();
                transform_segreduce( [=]ARIES_LAMBDA(int index)
                {
                    return data[ associated[ index ] ];
                }, tupleNum, groups, groupCount, ( type_nullable< type_t >* )( result->GetData() ), agg_max_t< type_nullable< type_t > >(),
                        type_nullable< type_t >( std::numeric_limits< type_nullable< type_t > >::min() ), context );
                break;
            }
            case AriesAggFunctionType::MIN:
            {
                result = std::make_shared< AriesDataBuffer >( columnType, groupCount );
                result->PrefetchToGpu();
                transform_segreduce( [=]ARIES_LAMBDA(int index)
                {
                    return data[ associated[ index ] ];
                }, tupleNum, groups, groupCount, ( type_nullable< type_t >* )( result->GetData() ), agg_min_t< type_nullable< type_t > >(),
                        type_nullable< type_t >( std::numeric_limits< type_nullable< type_t > >::max() ), context );
                break;
            }
            case AriesAggFunctionType::SUM:
            {
                return aggregate_sum( data, tupleNum, associated, groups, groupCount, context, bDistinct );
            }
            case AriesAggFunctionType::AVG:
            {
                // AVG will be replaced by sum and count for data partition process
                ARIES_ASSERT( 0, "AVG will be replaced by sum and count for data partition process" );
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "agg function " + GetAriesAggFunctionTypeName( aggType ) );
        }
        return result;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_column( const char* data, size_t len, bool hasNull, size_t tupleNum, AriesAggFunctionType aggType,
            const type_u* associated, const type_u* groups, size_t groupCount, context_t& context, bool bDistinct, bool bSumForCount = false )
    {
        AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( AriesColumnType
        {
        { AriesValueType::INT64, 1 }, false, false }, groupCount );
        result->PrefetchToGpu();

        switch( aggType )
        {
            case AriesAggFunctionType::COUNT:
            {
                if( bDistinct )
                {
                    int totalBytes = 30000;
                    int vt = 3;
                    int thread_num = ( totalBytes - len - sizeof( int ) ) / ( vt * len + sizeof( int ) );
                    // create a temp column for segsort
                    mem_t< char > tmp( len * tupleNum );
                    char* pData = tmp.data();
                    //memcpy( pData, data, len * tupleNum );
                    cudaMemcpy( pData, data, len * tupleNum, cudaMemcpyDefault );
                    if( !hasNull )
                    {
                        if( thread_num >= 256 )
                            segmented_sort< launch_box_t< arch_52_cta< 256, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< false, false >(),
                                    context );
                        else if( thread_num >= 128 )
                            segmented_sort< launch_box_t< arch_52_cta< 128, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< false, false >(),
                                    context );
                        else if( thread_num >= 64 )
                            segmented_sort< launch_box_t< arch_52_cta< 64, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< false, false >(),
                                    context );
                        else if( thread_num >= 32 )
                            segmented_sort< launch_box_t< arch_52_cta< 32, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< false, false >(),
                                    context );
                        else
                            ARIES_ENGINE_EXCEPTION( ER_TOO_LONG_STRING, "agg function " + GetAriesAggFunctionTypeName( aggType ) + " for string len:" + std::to_string( len ) );
                        not_equal_to_t_str< false, false > func;
                        lbs_segreduce( [=]ARIES_LAMBDA( int index, int seg, int rank )
                        {
                            char* cur = pData + index * len;
                            return (int)( index == groups[ seg ] || func( cur - len, cur, len ) );
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                    else
                    {
                        if( thread_num >= 256 )
                            segmented_sort< launch_box_t< arch_52_cta< 256, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< true, true >(),
                                    context );
                        else if( thread_num >= 128 )
                            segmented_sort< launch_box_t< arch_52_cta< 128, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< true, true >(),
                                    context );
                        else if( thread_num >= 64 )
                            segmented_sort< launch_box_t< arch_52_cta< 64, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< true, true >(),
                                    context );
                        else if( thread_num >= 32 )
                            segmented_sort< launch_box_t< arch_52_cta< 32, 3 > > >( pData, len, tupleNum, groups, groupCount, less_t_str_null_smaller< true, true >(),
                                    context );
                        else
                            ARIES_ENGINE_EXCEPTION( ER_TOO_LONG_STRING, "agg function " + GetAriesAggFunctionTypeName( aggType ) + " for string len:" + std::to_string( len ) );

                        equal_to_t_str_null_eq< true, true > func;
                        lbs_segreduce( [=]ARIES_LAMBDA( int index, int seg, int rank )
                        {
                            char* cur = pData + index * len;
                            if( * cur )
                            return (int)( index == groups[ seg ] || !func( cur - len, cur, len ) );
                            else
                            return 0;
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                }
                else
                {
                    if( !hasNull )
                    {
                        transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            return 1;
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                    else
                    {
                        transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            return *( data + associated[ index ] * len ) ? 1 : 0;
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                }

                break;
            }
            default:
                assert( 0 ); // we only support count for string type
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "agg function " + GetAriesAggFunctionTypeName( aggType ) + " for string type" );
        }
        return result;
    }

    template< typename type_u >
    AriesDataBufferSPtr aggregate_column( const CompactDecimal* data, AriesColumnType columnType, size_t tupleNum, AriesAggFunctionType aggType,
            const type_u* associated, const type_u* groups, size_t groupCount, context_t& context, bool bDistinct )
    {
        AriesDataBufferSPtr result;
        size_t len = columnType.GetDataTypeSize();
        uint16_t prec = columnType.DataType.Precision;
        uint16_t sca = columnType.DataType.Scale;
        bool hasNull = columnType.HasNull;
        AriesColumnType resultType
        {
        { AriesValueType::DECIMAL, 1 }, true };
        switch( aggType )
        {
            case AriesAggFunctionType::COUNT:
            {
                result = std::make_shared< AriesDataBuffer >( AriesColumnType
                {
                { AriesValueType::INT64, 1 }, false, false }, groupCount );
                result->PrefetchToGpu();
                if( bDistinct )
                {
                    // create a temp column for segsort
                    mem_t< char > tmp( len * tupleNum );
                    char* pData = tmp.data();
                    //memcpy( pData, data, len * tupleNum );
                    cudaMemcpy( pData, data, len * tupleNum, cudaMemcpyDefault );
                    if( !hasNull )
                    {
                        segmented_sort< launch_box_t< arch_52_cta< 256, 3 > > >( pData, len, tupleNum, groups, groupCount,
                                less_t_CompactDecimal_null_smaller< false, false >( prec, sca ), context );

                        not_equal_to_t_CompactDecimal< false, false > func( prec, sca );
                        lbs_segreduce( [=]ARIES_LAMBDA( int index, int seg, int rank )
                        {
                            char* cur = pData + index * len;
                            return (int)( index == groups[ seg ] || func( cur - len, cur, len ) );
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                    else
                    {
                        segmented_sort< launch_box_t< arch_52_cta< 256, 3 > > >( pData, len, tupleNum, groups, groupCount,
                                less_t_CompactDecimal_null_smaller< true, true >( prec, sca ), context );

                        equal_to_t_CompactDecimal_null_eq< true > func( prec, sca );
                        lbs_segreduce( [=]ARIES_LAMBDA( int index, int seg, int rank )
                        {
                            char* cur = pData + index * len;
                            if( * cur )
                            return (int)( index == groups[ seg ] || !func( cur - len, cur, len ) );
                            else
                            return 0;
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                }
                else
                {
                    if( !hasNull )
                    {
                        transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            return 1;
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                    else
                    {
                        transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            return *(char*)( data + associated[ index ] * len ) ? 1 : 0;
                        }, tupleNum, groups, groupCount, ( int64_t* )result->GetData(), agg_sum_t< int64_t >(), int64_t(), context );
                    }
                }

                break;
            }
            case AriesAggFunctionType::MAX:
            {
                result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
                result->PrefetchToGpu();
                if( !hasNull )
                {
                    transform_segreduce( [=]ARIES_LAMBDA(int index)
                    {
                        return Decimal( data + associated[ index ] * len, prec, sca );
                    }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )( result->GetData() ), agg_max_t< nullable_type< Decimal > >(),
                            nullable_type< Decimal >( std::numeric_limits< nullable_type< Decimal > >::min() ), context );
                }
                else
                {

                    transform_segreduce( [=]ARIES_LAMBDA(int index)
                    {
                        size_t offset = associated[ index ] * len;
                        return nullable_type< Decimal >( *(int8_t*)(data + offset), Decimal( data + offset + 1, prec, sca ) );
                    }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )( result->GetData() ), agg_max_t< nullable_type< Decimal > >(),
                            nullable_type< Decimal >( std::numeric_limits< nullable_type< Decimal > >::min() ), context );
                }
                break;
            }
            case AriesAggFunctionType::MIN:
            {
                result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
                result->PrefetchToGpu();
                if( !hasNull )
                {
                    transform_segreduce( [=]ARIES_LAMBDA(int index)
                    {
                        return Decimal( data + associated[ index ] * len, prec, sca );
                    }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )( result->GetData() ), agg_min_t< nullable_type< Decimal > >(),
                            nullable_type< Decimal >( std::numeric_limits< nullable_type< Decimal > >::max() ), context );
                }
                else
                {

                    transform_segreduce( [=]ARIES_LAMBDA(int index)
                    {
                        size_t offset = associated[ index ] * len;
                        return nullable_type< Decimal >( *(int8_t*)(data + offset), Decimal( data + offset + 1, prec, sca ) );
                    }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )( result->GetData() ), agg_min_t< nullable_type< Decimal > >(),
                            nullable_type< Decimal >( std::numeric_limits< nullable_type< Decimal > >::max() ), context );
                }
                break;
            }
            case AriesAggFunctionType::SUM:
            {
                result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
                result->PrefetchToGpu();
                if( bDistinct )
                {
                    // create a temp column for segsort
                    mem_t< char > tmp( len * tupleNum );
                    char* pData = tmp.data();
                    //memcpy( pData, data, len * tupleNum );
                    cudaMemcpy( pData, data, len * tupleNum, cudaMemcpyDefault );
                    if( !hasNull )
                    {
                        segmented_sort< launch_box_t< arch_52_cta< 256, 3 > > >( pData, len, tupleNum, groups, groupCount,
                                less_t_CompactDecimal_null_smaller< false, false >( prec, sca ), context );

                        not_equal_to_t_CompactDecimal< false, false > func( prec, sca );
                        lbs_segreduce< launch_box_t< arch_52_cta< 32, 3 > > >(
                                [=]ARIES_LAMBDA( int index, int seg, int rank )
                                {
                                    char* cur = pData + index * len;
                                    return ( index == groups[ seg ] || func( cur - len, cur, len ) ) ? Decimal( ( CompactDecimal* )cur, prec, sca ) : Decimal();
                                }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )result->GetData(),
                                agg_sum_t< nullable_type< Decimal > >(), nullable_type< Decimal >(), context );
                    }
                    else
                    {
                        segmented_sort< launch_box_t< arch_52_cta< 256, 3 > > >( pData, len, tupleNum, groups, groupCount,
                                less_t_CompactDecimal_null_smaller< true, true >( prec, sca ), context );

                        equal_to_t_CompactDecimal_null_eq< true > func( prec, sca );
                        lbs_segreduce< launch_box_t< arch_52_cta< 32, 3 > > >(
                                [=]ARIES_LAMBDA( int index, int seg, int rank )
                                {
                                    char* cur = pData + index * len;
                                    if( * cur )
                                    return ( index == groups[ seg ] || !func( cur - len, cur, len ) ) ? Decimal( ( CompactDecimal* )cur + 1, prec, sca ) : Decimal();
                                    else
                                    return Decimal();
                                }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )result->GetData(),
                                agg_sum_t< nullable_type< Decimal > >(), nullable_type< Decimal >(), context );
                    }
                }
                else
                {
                    if( !hasNull )
                    {
                        // 针对 not null 的 Decimal 列
                        transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            return Decimal( data + associated[ index ] * len, prec, sca );
                        }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )result->GetData(), agg_sum_t< nullable_type< Decimal > >(),
                                nullable_type< Decimal >(), context, sca);
                    }
                    else
                    {
                        // 针对 has Null 会变为 nullable_type 类型的
                        transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            size_t offset = associated[ index ] * len;
                            return nullable_type< Decimal >( *(int8_t*)(data + offset), Decimal( data + offset + 1, prec, sca ) );
                        }, tupleNum, groups, groupCount, ( nullable_type< Decimal >* )result->GetData(), agg_sum_t< nullable_type< Decimal > >(),
                                nullable_type< Decimal >(), context, sca );
                    }
                }
                break;
            }
            case AriesAggFunctionType::AVG:
            {
                // AVG will be replaced by sum and count for data partition process
                ARIES_ASSERT( 0, "AVG will be replaced by sum and count for data partition process" );
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "agg function " + GetAriesAggFunctionTypeName( aggType ) );
        }
        return result;
    }

    template< typename type_t >
    AriesDataBufferSPtr aggregate_column( const int8_t* data, AriesColumnType type, size_t tupleNum, AriesAggFunctionType aggType,
            const type_t* associated, const type_t* groups, size_t groupCount, context_t& context, bool bDistinct = false, bool bSumForCount = false )
    {
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( type.DataType.Length > 1 )
                    return aggregate_column( ( char* )data, type.GetDataTypeSize(), type.HasNull, tupleNum, aggType, associated, groups, groupCount,
                            context, bDistinct );
                else
                {
                    if( !type.HasNull )
                        return aggregate_column< int8_t, type_t >( ( int8_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                                bDistinct );
                    else
                        return aggregate_column( ( nullable_type< int8_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                                bDistinct );
                }
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    return aggregate_column< int8_t, type_t >( ( int8_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
                else
                    return aggregate_column( ( nullable_type< int8_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    return aggregate_column( ( int16_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< int16_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    return aggregate_column( ( int32_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< int32_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    return aggregate_column( ( int64_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct,
                            bSumForCount );
                else
                    return aggregate_column( ( nullable_type< int64_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    return aggregate_column( ( uint8_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< uint8_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    return aggregate_column( ( uint16_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< uint16_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    return aggregate_column( ( uint32_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< uint32_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    return aggregate_column( ( uint64_t* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< uint64_t >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    return aggregate_column( ( float* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< float >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    return aggregate_column( ( double* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< double >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    return aggregate_column( ( Decimal* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< Decimal >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                // Decimal做agg_CompactDecimal 作为输入
                return aggregate_column( ( CompactDecimal* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    return aggregate_column( ( AriesDate* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< AriesDate >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    return aggregate_column( ( AriesDatetime* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< AriesDatetime >* )data, type, tupleNum, aggType, associated, groups, groupCount,
                            context, bDistinct );
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    return aggregate_column( ( AriesTimestamp* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< AriesTimestamp >* )data, type, tupleNum, aggType, associated, groups, groupCount,
                            context, bDistinct );
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    return aggregate_column( ( AriesYear* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< AriesYear >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            case AriesValueType::TIME:
            {
                if( !type.HasNull )
                    return aggregate_column( ( AriesTime* )data, type, tupleNum, aggType, associated, groups, groupCount, context, bDistinct );
                else
                    return aggregate_column( ( nullable_type< AriesTime >* )data, type, tupleNum, aggType, associated, groups, groupCount, context,
                            bDistinct );
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "data type " + GetValueTypeAsString( type ) + " in aggregation expression" );
        }
//    return std::make_shared< AriesDataBuffer >( type );
    }

    template< typename type_t, typename index_t >
    void reorder_by_index( const type_t* data_input, size_t count, const index_t* indices, type_t* data_output, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            data_output[ index ] = indices[ data_input[ index ] ];
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename index_t >
    void reorder_flags( const type_t* data_input, size_t count, const index_t* indices, type_t* data_output, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            data_output[ indices[ index ] ] = data_input[ index ];
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename a_it, typename b_it >
    AriesBoolArraySPtr sort_based_semi_join( a_it a, int a_count, b_it b, int b_count, const JoinDynamicCodeParams* joinDynamicCodeParams,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr )
    {
        typedef typename std::iterator_traits< a_it >::value_type type_a;
        typedef typename std::iterator_traits< b_it >::value_type type_b;
        AriesBoolArraySPtr indicesArray = std::make_shared< AriesBoolArray >();
        if( a_count > 0 )
        {
            indicesArray->AllocArray( a_count );
            auto indices = indicesArray->GetData();
            if( left_associated && right_associated )
            {
                auto lower_upper = find_lower_and_upper_bound< less_t_null_smaller_join >( a, a_count, b, b_count, context );
                sort_based_semi_join_helper_dyn_code( lower_upper.first.data(),
                                                      lower_upper.second.data(),
                                                      left_associated,
                                                      right_associated,
                                                      a_count,
                                                      joinDynamicCodeParams->comparators,
                                                      joinDynamicCodeParams->cuModules,
                                                      joinDynamicCodeParams->functionName.c_str(),
                                                      joinDynamicCodeParams->input,
                                                      joinDynamicCodeParams->constValues,
                                                      indices,
                                                      context );
            }
            else
            {
                mem_t< int > vals_a( a_count );
                mem_t< int > vals_b( b_count );
                init_sequence( vals_a.data(), a_count, context );
                init_sequence( vals_b.data(), b_count, context );

                mergesort( a, vals_a.data(), a_count, less_t_null_smaller< type_a, type_a >(), context );
                mergesort( b, vals_b.data(), b_count, less_t_null_smaller< type_b, type_b >(), context );

                auto lower_upper = find_lower_and_upper_bound< less_t_null_smaller_join >( a, a_count, b, b_count, context );
                sort_based_semi_join_helper_dyn_code( lower_upper.first.data(),
                                                      lower_upper.second.data(),
                                                      vals_a.data(),
                                                      vals_b.data(),
                                                      a_count,
                                                      joinDynamicCodeParams->comparators,
                                                      joinDynamicCodeParams->cuModules,
                                                      joinDynamicCodeParams->functionName.c_str(),
                                                      joinDynamicCodeParams->input,
                                                      joinDynamicCodeParams->constValues,
                                                      indices,
                                                      context );
            }
        }
        return indicesArray;
    }

    template< typename type_t >
    bool has_null_value( type_t* data, int count, context_t& context )
    {
        return false;
    }

    template< typename type_t, template< typename > class type_nullable >
    bool has_null_value( type_nullable< type_t >* data, int count, context_t& context )
    {
        managed_mem_t< int > ret( 1, context );
        int* pValue = ret.data();
        *pValue = 0;
        transform< 256, 5 >( [=] ARIES_DEVICE(int index)
        {
            if( !data[index].flag )
                atomicExch( pValue, 1 );
        }, count, context );
        context.synchronize();
        return *pValue == 1;
    }

    template< typename a_it, typename b_it >
    AriesBoolArraySPtr sort_based_anti_join( a_it a, int a_count, b_it b, int b_count, const JoinDynamicCodeParams* joinDynamicCodeParams,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr, bool isNotIn = false )
    {
        AriesBoolArraySPtr result;
        if( isNotIn )
        {
            if( has_null_value( b, b_count, context ) )
                result = std::make_shared< AriesBoolArray >( a_count, true );
            else
            {
                mem_t< int8_t > nullFlags( a_count );
                int8_t* pFlags = nullFlags.data();
                transform< 256, 5 >( [=] ARIES_DEVICE(int index)
                {
                    if( is_null( a[index] ) )
                        pFlags[ index ] = 1;
                    else
                        pFlags[ index ] = 0;
                }, a_count, context );

                result = sort_based_semi_join( a, a_count, b, b_count, joinDynamicCodeParams, context, left_associated,
                        right_associated );

                auto data = result->GetData();
                auto k = [=] ARIES_DEVICE(int index)
                {
                    if( pFlags[ index ] )
                        data[ index ] = false;
                    else
                        data[index] = !data[index].is_true();
                };
                transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
            }
        }
        else
        {
            result = sort_based_semi_join( a, a_count, b, b_count, joinDynamicCodeParams, context, left_associated,
                    right_associated );
            auto data = result->GetData();
            auto k = [=] ARIES_DEVICE(int index)
            {
                data[index] = !data[index].is_true();
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
        }
        return result;
    }

    template< typename a_it, typename b_it >
    AriesBoolArraySPtr sort_based_anti_join_has_null( a_it a, int a_count, b_it b, int b_count, const JoinDynamicCodeParams* joinDynamicCodeParams,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr )
    {
        return sort_based_anti_join( a, a_count, b, b_count, joinDynamicCodeParams, context, left_associated, right_associated );
    }

    void sort_based_semi_join_helper_dyn_code( const index_t* lower_data,
                                               const index_t* upper_data,
                                               const index_t* vals_a, // indices of sorted left data
                                               const index_t* vals_b, // indices of sorted right data
                                               int a_count,
                                               const vector< AriesDynamicCodeComparator >& comparators,
                                               const vector< CUmoduleSPtr >& modules,
                                               const char *functionName,
                                               const AriesColumnDataIterator *input,
                                               const std::vector< AriesDataBufferSPtr >& constValues,
                                               AriesBool* output,
                                               context_t& context );

    template< typename cmp, typename cmp_sort >
    AriesBoolArraySPtr sort_based_semi_join( char* a, int a_count, char* b, int b_count, int len, cmp comp, cmp_sort comp_sort,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated = nullptr,
            const int32_t* right_associated = nullptr )
    {
        auto indicesArray = std::make_shared< AriesBoolArray >();
        if( a_count > 0 )
        {
            indicesArray->AllocArray( a_count );
            auto indices = indicesArray->GetData();
            if( left_associated && right_associated )
            {
                auto lower_upper = find_lower_and_upper_bound< less_t_null_smaller_join >( a, a_count, b, b_count, context );
                sort_based_semi_join_helper_dyn_code( lower_upper.first.data(),
                                                      lower_upper.second.data(),
                                                      left_associated,
                                                      right_associated,
                                                      a_count,
                                                      joinDynamicCodeParams->comparators,
                                                      joinDynamicCodeParams->cuModules,
                                                      joinDynamicCodeParams->functionName.c_str(),
                                                      joinDynamicCodeParams->input,
                                                      joinDynamicCodeParams->constValues,
                                                      indices,
                                                      context );
            }
            else
            {
                mem_t< int > vals_a( a_count );
                mem_t< int > vals_b( b_count );
                init_sequence( vals_a.data(), a_count, context );
                init_sequence( vals_b.data(), b_count, context );

                mergesort( a, len, vals_a.data(), a_count, comp_sort, context );
                mergesort( b, len, vals_b.data(), b_count, comp_sort, context );

                auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );

                sort_based_semi_join_helper_dyn_code( lower_upper.first.data(),
                                                      lower_upper.second.data(),
                                                      vals_a.data(),
                                                      vals_b.data(),
                                                      a_count,
                                                      joinDynamicCodeParams->comparators,
                                                      joinDynamicCodeParams->cuModules,
                                                      joinDynamicCodeParams->functionName.c_str(),
                                                      joinDynamicCodeParams->input,
                                                      joinDynamicCodeParams->constValues,
                                                      indices,
                                                      context );
            }
        }
        return indicesArray;
    }

    AriesBoolArraySPtr sort_based_anti_join( char* a, int a_count, char* b, int b_count, int len, const JoinDynamicCodeParams* joinDynamicCodeParams,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr );

    AriesBoolArraySPtr sort_based_anti_join_has_null( char* a, int a_count, char* b, int b_count, int len,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated = nullptr,
            const int32_t* right_associated = nullptr, bool isNotIn = false );

    AriesBoolArraySPtr sort_based_anti_join( CompactDecimal* a, int a_count, CompactDecimal* b, int b_count, int len, AriesColumnType leftType,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated = nullptr,
            const int32_t* right_associated = nullptr );

    AriesBoolArraySPtr sort_based_anti_join_has_null( CompactDecimal* a, int a_count, CompactDecimal* b, int b_count, int len,
            AriesColumnType leftType, const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context,
            const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr, bool isNotIn = false );

    template< typename a_it, typename b_it, join_pair_t< int32_t > (*join)( a_it, int32_t, b_it, int32_t, context_t&, const int*, const int*,
            const JoinDynamicCodeParams* ) >
    JoinPair sort_based_join( a_it a, int32_t a_count, b_it b, int32_t b_count, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr )
    {
        JoinPair result;

        typedef typename std::iterator_traits< a_it >::value_type type_a;
        typedef typename std::iterator_traits< b_it >::value_type type_b;
        if( a_count > 0 && b_count > 0 )
        {
            if( left_associated && right_associated )
            {
                join_pair_t< int32_t > pair = join( a, a_count, b, b_count, context, left_associated, right_associated, joinDynamicCodeParams );
                size_t count = pair.count;
                result.LeftIndices = std::make_shared< AriesInt32Array >();
                result.RightIndices = std::make_shared< AriesInt32Array >();
                result.JoinCount = count;
                if( count > 0 )
                {
                    result.LeftIndices->AttachBuffer( pair.left_indices.release_data(), count );
                    result.RightIndices->AttachBuffer( pair.right_indices.release_data(), count );
//                result.LeftIndices->PrefetchToGpu();
//                reorder_by_index( pair.left_indices.data(), count, left_associated, result.LeftIndices->GetData(), context );
//                result.RightIndices->PrefetchToGpu();
//                reorder_by_index( pair.right_indices.data(), count, right_associated, result.RightIndices->GetData(), context );
                }
            }
            else
            {
                //context.timer_begin();
                mem_t< int32_t > vals_a( a_count + 1 );
                cudaMemset( vals_a.data(), 0xff, sizeof( int32_t ) );
                init_sequence( vals_a.data() + 1, a_count, context );
                mem_t< int32_t > vals_b( b_count + 1 );
                cudaMemset( vals_b.data(), 0xff, sizeof( int32_t ) );
                init_sequence( vals_b.data() + 1, b_count, context );

                mergesort( a, vals_a.data() + 1, a_count, less_t_null_smaller< type_a, type_a >(), context );
                mergesort( b, vals_b.data() + 1, b_count, less_t_null_smaller< type_b, type_b >(), context );
                //printf( "mergesort gpu time: %3.1f\n", context.timer_end() );

                //context.timer_begin();
                join_pair_t< int32_t > pair = join( a, a_count, b, b_count, context, vals_a.data() + 1, vals_b.data() + 1, joinDynamicCodeParams );
                size_t count = pair.count;
                result.LeftIndices = std::make_shared< AriesInt32Array >();
                result.RightIndices = std::make_shared< AriesInt32Array >();
                result.JoinCount = count;
                if( count > 0 )
                {
                    result.LeftIndices->AttachBuffer( pair.left_indices.release_data(), count );
                    result.RightIndices->AttachBuffer( pair.right_indices.release_data(), count );
//                result.LeftIndices->PrefetchToGpu();
//                reorder_by_index( pair.left_indices.data(), count, vals_a.data() + 1, result.LeftIndices->GetData(), context );
//                result.RightIndices->PrefetchToGpu();
//                reorder_by_index( pair.right_indices.data(), count, vals_b.data() + 1, result.RightIndices->GetData(), context );
                }
                //printf( "other gpu time: %3.1f\n", context.timer_end() );
            }
        }
        return result;
    }

    template< typename cmp, typename cmp_sort, join_pair_t< int32_t > (*join)( const char*, int32_t, const char*, int32_t, int32_t, cmp, context_t&,
            const int*, const int*, const JoinDynamicCodeParams* ) >
    JoinPair sort_based_join( char* a, int a_count, char* b, int b_count, int len, cmp comp, cmp_sort comp_sort, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr )
    {
        JoinPair result;
        if( a_count > 0 && b_count > 0 )
        {
            if( left_associated && right_associated )
            {
                join_pair_t< int32_t > pair = join( a, a_count, b, b_count, len, comp, context, left_associated, right_associated,
                        joinDynamicCodeParams );
                size_t count = pair.count;
                result.LeftIndices = std::make_shared< AriesInt32Array >();
                result.RightIndices = std::make_shared< AriesInt32Array >();
                result.JoinCount = count;
                if( count > 0 )
                {
                    result.LeftIndices->AttachBuffer( pair.left_indices.release_data(), count );
                    result.RightIndices->AttachBuffer( pair.right_indices.release_data(), count );
//                result.LeftIndices->PrefetchToGpu();
//                reorder_by_index( pair.left_indices.data(), count, left_associated, result.LeftIndices->GetData(), context );
//                result.RightIndices->PrefetchToGpu();
//                reorder_by_index( pair.right_indices.data(), count, right_associated, result.RightIndices->GetData(), context );
                }
            }
            else
            {
                mem_t< int32_t > vals_a( a_count + 1 );
                cudaMemset( vals_a.data(), 0xff, sizeof( int32_t ) );
                init_sequence( vals_a.data() + 1, a_count, context );
                mem_t< int32_t > vals_b( b_count + 1 );
                cudaMemset( vals_b.data(), 0xff, sizeof( int32_t ) );
                init_sequence( vals_b.data() + 1, b_count, context );

                mergesort( a, len, vals_a.data() + 1, a_count, comp_sort, context );
                mergesort( b, len, vals_b.data() + 1, b_count, comp_sort, context );

                join_pair_t< int32_t > pair = join( a, a_count, b, b_count, len, comp, context, vals_a.data() + 1, vals_b.data() + 1,
                        joinDynamicCodeParams );
                size_t count = pair.count;
                result.LeftIndices = std::make_shared< AriesInt32Array >();
                result.RightIndices = std::make_shared< AriesInt32Array >();
                result.JoinCount = count;
                if( count > 0 )
                {
                    result.LeftIndices->AttachBuffer( pair.left_indices.release_data(), count );
                    result.RightIndices->AttachBuffer( pair.right_indices.release_data(), count );
//                result.LeftIndices->PrefetchToGpu();
//                reorder_by_index( pair.left_indices.data(), count, vals_a.data() + 1, result.LeftIndices->GetData(), context );
//                result.RightIndices->PrefetchToGpu();
//                reorder_by_index( pair.right_indices.data(), count, vals_b.data() + 1, result.RightIndices->GetData(), context );
                }
            }
        }
        return result;
    }

    template< typename a_it, typename b_it >
    JoinPair sort_based_inner_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int32_t* left_associated = nullptr,
            const int32_t* right_associated = nullptr )
    {
        return sort_based_join< a_it, b_it, &inner_join< less_t_null_smaller_join, empty_t, a_it, b_it > >( a, a_count, b, b_count, context, nullptr,
                left_associated, right_associated );
    }

    template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    join_pair_t< int > inner_join_wrapper( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a,
            const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams );

    template< typename launch_arg_t = empty_t, typename comp_t >
    join_pair_t< int > inner_join_wrapper( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );


    template< typename a_it, typename b_it >
    JoinPair sort_based_inner_join( a_it a, int a_count, b_it b, int b_count, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr  )
    {
        return sort_based_join< a_it, b_it, &inner_join_wrapper< less_t_null_smaller_join, empty_t, a_it, b_it > >( a, a_count, b, b_count, context, joinDynamicCodeParams );
    }

    template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    join_pair_t< int > left_join_wrapper( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a, const int* vals_b,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    template< typename launch_arg_t = empty_t, typename comp_t >
    join_pair_t< int > left_join_wrapper( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    template< template< typename, typename > class comp_t, typename launch_arg_t, typename a_it, typename b_it >
    join_pair_t< int > right_join_wrapper( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a, const int* vals_b,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        join_pair_t< int > result = left_join_wrapper< comp_t >( b, b_count, a, a_count, context, vals_b, vals_a, joinDynamicCodeParams );
        result.left_indices.swap( result.right_indices );
        return result;
    }

    template< typename launch_arg_t, typename comp_t >
    join_pair_t< int > right_join_wrapper( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        join_pair_t< int > result = left_join_wrapper( b, b_count, a, a_count, len, comp, context, vals_b, vals_a, joinDynamicCodeParams );
        result.left_indices.swap( result.right_indices );
        return result;
    }

    join_pair_t< int > cartesian_join_wrapper( size_t left_count, size_t right_count, bool need_keep_left, bool need_keep_right, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    template< typename a_it, typename b_it >
    JoinPair sort_based_left_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< a_it, b_it, &left_join_wrapper< less_t_null_smaller_join, empty_t, a_it, b_it > >( a, a_count, b, b_count, context,
                joinDynamicCodeParams );
    }

    template< typename a_it, typename b_it >
    JoinPair sort_based_right_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< a_it, b_it, &right_join_wrapper< less_t_null_smaller_join, empty_t, a_it, b_it > >( a, a_count, b, b_count, context,
                joinDynamicCodeParams );
    }

    join_pair_t< int > full_join_helper_dyn_code( const int *lower_data, const int *upper_data, int a_count, int b_count, context_t &context,
            const int *vals_a = nullptr, const int *vals_b = nullptr, const JoinDynamicCodeParams *joinDynamicCodeParams = nullptr );

    template< template< typename, typename > class comp_t, typename a_it, typename b_it >
    join_pair_t< int > full_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
            nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );
        return full_join_helper_dyn_code( lower_upper.first.data(), lower_upper.second.data(), a_count, b_count, context, vals_a, vals_b,
                joinDynamicCodeParams );
    }

    template< typename comp_t >
    join_pair_t< int > full_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a = nullptr, const int* vals_b = nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );
        return full_join_helper_dyn_code( lower_upper.first.data(), lower_upper.second.data(), a_count, b_count, context, vals_a, vals_b,
                joinDynamicCodeParams );
    }

    template< typename a_it, typename b_it >
    JoinPair sort_based_full_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< a_it, b_it, &full_join< less_t_null_smaller_join, a_it, b_it > >( a, a_count, b, b_count, context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_inner_join( char* a, int a_count, char* b, int b_count, int len, context_t& context, const int32_t* left_associated = nullptr,
            const int32_t* right_associated = nullptr );
    JoinPair sort_based_inner_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context, const int32_t* left_associated =
            nullptr, const int32_t* right_associated = nullptr );

    JoinPair sort_based_left_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams );
    JoinPair sort_based_left_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair sort_based_right_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams );
    JoinPair sort_based_right_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair sort_based_full_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams );
    JoinPair sort_based_full_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair sort_based_inner_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr );
    JoinPair sort_based_inner_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr );

    JoinPair sort_based_left_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams );
    JoinPair sort_based_left_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams );

    JoinPair sort_based_right_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams );
    JoinPair sort_based_right_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams );

    JoinPair sort_based_full_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams );
    JoinPair sort_based_full_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams );

    AriesBoolArraySPtr sort_based_semi_join( int8_t* leftData, size_t leftTupleNum, int8_t* rightData, size_t rightTupleNum,
            AriesColumnType leftType, AriesColumnType rightType, const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context,
            const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr );

    AriesBoolArraySPtr sort_based_anti_join( int8_t* leftData, size_t leftTupleNum, int8_t* rightData, size_t rightTupleNum,
            AriesColumnType leftType, AriesColumnType rightType, const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context,
            const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr, bool isNotIn = false );

    AriesBoolArraySPtr
    sort_based_semi_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams,
            context_t& context, const int32_t* left_associated = nullptr, const int32_t* right_associated = nullptr );

    AriesBoolArraySPtr sort_based_anti_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated = nullptr,
            const int32_t* right_associated = nullptr, bool isNotIn = false );

    JoinPair sort_based_inner_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const AriesInt32ArraySPtr leftAssociated = nullptr, const AriesInt32ArraySPtr rightAssociated = nullptr );
    JoinPair sort_based_inner_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );
    JoinPair sort_based_left_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );
    JoinPair sort_based_right_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );
    JoinPair sort_based_full_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    void division_data_int64( const int8_t* data, AriesColumnType type, const int64_t* divisor, int8_t* output, size_t tupleNum, context_t& context );

    JoinPair create_cartesian_product( size_t left_count, size_t right_count, context_t& context );

    AriesBoolArraySPtr is_null( int8_t* data, size_t tupleNum, AriesColumnType type, context_t& context );
    AriesBoolArraySPtr is_not_null( int8_t* data, size_t tupleNum, AriesColumnType type, context_t& context );
    AriesDataBufferSPtr datetime_to_date( const AriesDataBufferSPtr& column, context_t& context );
    AriesDataBufferSPtr sql_abs_func( const AriesDataBufferSPtr& column, context_t& context );
    AriesDataBufferSPtr month( const AriesDataBufferSPtr& column, context_t& context );
    AriesDataBufferSPtr date_format( const AriesDataBufferSPtr& column, const AriesDataBufferSPtr& format, const LOCALE_LANGUAGE &locale,
            context_t& context );

    JoinPair hash_join( AriesDataBufferSPtr leftColumn,
                        AriesDataBufferSPtr rightColumn,
                        context_t& context, 
                        const AriesInt32ArraySPtr leftAssociated = nullptr,
                        const AriesInt32ArraySPtr rightAssociated = nullptr );
END_ARIES_ACC_NAMESPACE

#endif /* ARIESENGINEALGORITHM_H_ */
