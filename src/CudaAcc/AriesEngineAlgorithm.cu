/*
 * AriesEngineAlgorithm.cu
 *
 *  Created on: Jun 19, 2019
 *      Author: lichi
 */

#include "AriesEngineAlgorithm.h"
#include "AriesTimeCalc.hxx"
#include "AriesDatetimeTrans.h"
#include "AriesSqlOperator.h"
#include "DynamicKernel.h"

BEGIN_ARIES_ACC_NAMESPACE

    __device__ IComparableColumnPair* create_eq_comparator( const ColumnPairs& column )
    {
        IComparableColumnPair* columnPair = nullptr;
        AriesColumnType type = column.ColumnType;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, equal_to_t_str< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), equal_to_t_str< false, false >() );
                else
                    columnPair = new ComparableColumnPair< char, char, equal_to_t_str< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), equal_to_t_str< true, true >() );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, equal_to_t_CompactDecimal< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            equal_to_t_CompactDecimal< false, false >( type.DataType.Precision, type.DataType.Scale ) );
                else
                    columnPair = new ComparableColumnPair< char, char, equal_to_t_CompactDecimal< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            equal_to_t_CompactDecimal< true, true >( type.DataType.Precision, type.DataType.Scale ) );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int8_t, int8_t, equal_to_t< int8_t > >( ( const int8_t* )column.LeftColumn,
                            ( const int8_t* )column.RightColumn, equal_to_t< int8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int8_t >, nullable_type< int8_t >, equal_to_t< nullable_type< int8_t > > >(
                            ( const nullable_type< int8_t >* )column.LeftColumn, ( const nullable_type< int8_t >* )column.RightColumn,
                            equal_to_t< nullable_type< int8_t > >() );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int16_t, int16_t, equal_to_t< int16_t > >( ( const int16_t* )column.LeftColumn,
                            ( const int16_t* )column.RightColumn, equal_to_t< int16_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int16_t >, nullable_type< int16_t >, equal_to_t< nullable_type< int16_t > > >(
                                    ( const nullable_type< int16_t >* )column.LeftColumn, ( const nullable_type< int16_t >* )column.RightColumn,
                                    equal_to_t< nullable_type< int16_t > >() );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int32_t, int32_t, equal_to_t< int32_t > >( ( const int32_t* )column.LeftColumn,
                            ( const int32_t* )column.RightColumn, equal_to_t< int32_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int32_t >, nullable_type< int32_t >, equal_to_t< nullable_type< int32_t > > >(
                                    ( const nullable_type< int32_t >* )column.LeftColumn, ( const nullable_type< int32_t >* )column.RightColumn,
                                    equal_to_t< nullable_type< int32_t > >() );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int64_t, int64_t, equal_to_t< int64_t > >( ( const int64_t* )column.LeftColumn,
                            ( const int64_t* )column.RightColumn, equal_to_t< int64_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int64_t >, nullable_type< int64_t >, equal_to_t< nullable_type< int64_t > > >(
                                    ( const nullable_type< int64_t >* )column.LeftColumn, ( const nullable_type< int64_t >* )column.RightColumn,
                                    equal_to_t< nullable_type< int64_t > >() );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint8_t, uint8_t, equal_to_t< uint8_t > >( ( const uint8_t* )column.LeftColumn,
                            ( const uint8_t* )column.RightColumn, equal_to_t< uint8_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< uint8_t >, nullable_type< uint8_t >, equal_to_t< nullable_type< uint8_t > > >(
                                    ( const nullable_type< uint8_t >* )column.LeftColumn, ( const nullable_type< uint8_t >* )column.RightColumn,
                                    equal_to_t< nullable_type< uint8_t > >() );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint16_t, uint16_t, equal_to_t< uint16_t > >( ( const uint16_t* )column.LeftColumn,
                            ( const uint16_t* )column.RightColumn, equal_to_t< uint16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint16_t >, nullable_type< uint16_t >,
                            equal_to_t< nullable_type< uint16_t > > >( ( const nullable_type< uint16_t >* )column.LeftColumn,
                            ( const nullable_type< uint16_t >* )column.RightColumn, equal_to_t< nullable_type< uint16_t > >() );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint32_t, uint32_t, equal_to_t< uint32_t > >( ( const uint32_t* )column.LeftColumn,
                            ( const uint32_t* )column.RightColumn, equal_to_t< uint32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint32_t >, nullable_type< uint32_t >,
                            equal_to_t< nullable_type< uint32_t > > >( ( const nullable_type< uint32_t >* )column.LeftColumn,
                            ( const nullable_type< uint32_t >* )column.RightColumn, equal_to_t< nullable_type< uint32_t > >() );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint64_t, uint64_t, equal_to_t< uint64_t > >( ( const uint64_t* )column.LeftColumn,
                            ( const uint64_t* )column.RightColumn, equal_to_t< uint64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint64_t >, nullable_type< uint64_t >,
                            equal_to_t< nullable_type< uint64_t > > >( ( const nullable_type< uint64_t >* )column.LeftColumn,
                            ( const nullable_type< uint64_t >* )column.RightColumn, equal_to_t< nullable_type< uint64_t > >() );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< float, float, equal_to_t< float > >( ( const float* )column.LeftColumn,
                            ( const float* )column.RightColumn, equal_to_t< float >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< float >, nullable_type< float >, equal_to_t< nullable_type< float > > >(
                            ( const nullable_type< float >* )column.LeftColumn, ( const nullable_type< float >* )column.RightColumn,
                            equal_to_t< nullable_type< float > >() );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< double, double, equal_to_t< double > >( ( const double* )column.LeftColumn,
                            ( const double* )column.RightColumn, equal_to_t< double >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< double >, nullable_type< double >, equal_to_t< nullable_type< double > > >(
                            ( const nullable_type< double >* )column.LeftColumn, ( const nullable_type< double >* )column.RightColumn,
                            equal_to_t< nullable_type< double > >() );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< Decimal, Decimal, equal_to_t< Decimal > >( ( const Decimal* )column.LeftColumn,
                            ( const Decimal* )column.RightColumn, equal_to_t< Decimal >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< Decimal >, nullable_type< Decimal >, equal_to_t< nullable_type< Decimal > > >(
                                    ( const nullable_type< Decimal >* )column.LeftColumn, ( const nullable_type< Decimal >* )column.RightColumn,
                                    equal_to_t< nullable_type< Decimal > >() );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDate, AriesDate, equal_to_t< AriesDate > >( ( const AriesDate* )column.LeftColumn,
                            ( const AriesDate* )column.RightColumn, equal_to_t< AriesDate >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDate >, nullable_type< AriesDate >,
                            equal_to_t< nullable_type< AriesDate > > >( ( const nullable_type< AriesDate >* )column.LeftColumn,
                            ( const nullable_type< AriesDate >* )column.RightColumn, equal_to_t< nullable_type< AriesDate > >() );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDatetime, AriesDatetime, equal_to_t< AriesDatetime > >(
                            ( const AriesDatetime* )column.LeftColumn, ( const AriesDatetime* )column.RightColumn, equal_to_t< AriesDatetime >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDatetime >, nullable_type< AriesDatetime >,
                            equal_to_t< nullable_type< AriesDatetime > > >( ( const nullable_type< AriesDatetime >* )column.LeftColumn,
                            ( const nullable_type< AriesDatetime >* )column.RightColumn, equal_to_t< nullable_type< AriesDatetime > >() );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesTimestamp, AriesTimestamp, equal_to_t< AriesTimestamp > >(
                            ( const AriesTimestamp* )column.LeftColumn, ( const AriesTimestamp* )column.RightColumn, equal_to_t< AriesTimestamp >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesTimestamp >, nullable_type< AriesTimestamp >,
                            equal_to_t< nullable_type< AriesTimestamp > > >( ( const nullable_type< AriesTimestamp >* )column.LeftColumn,
                            ( const nullable_type< AriesTimestamp >* )column.RightColumn, equal_to_t< nullable_type< AriesTimestamp > >() );
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesYear, AriesYear, equal_to_t< AriesYear > >( ( const AriesYear* )column.LeftColumn,
                            ( const AriesYear* )column.RightColumn, equal_to_t< AriesYear >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesYear >, nullable_type< AriesYear >,
                            equal_to_t< nullable_type< AriesYear > > >( ( const nullable_type< AriesYear >* )column.LeftColumn,
                            ( const nullable_type< AriesYear >* )column.RightColumn, equal_to_t< nullable_type< AriesYear > >() );
                break;
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                break;
        }
        return columnPair;
    }

    __device__ IComparableColumnPair* create_ne_comparator( const ColumnPairs& column )
    {
        IComparableColumnPair* columnPair = nullptr;
        AriesColumnType type = column.ColumnType;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, not_equal_to_t_str< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), not_equal_to_t_str< false, false >() );
                else
                    columnPair = new ComparableColumnPair< char, char, not_equal_to_t_str< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), not_equal_to_t_str< true, true >() );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, not_equal_to_t_CompactDecimal< false, false > >(
                            ( const char* )column.LeftColumn, ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            not_equal_to_t_CompactDecimal< false, false >( type.DataType.Precision, type.DataType.Scale ) );
                else
                    columnPair = new ComparableColumnPair< char, char, not_equal_to_t_CompactDecimal< true, true > >(
                            ( const char* )column.LeftColumn, ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            not_equal_to_t_CompactDecimal< true, true >( type.DataType.Precision, type.DataType.Scale ) );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int8_t, int8_t, not_equal_to_t< int8_t > >( ( const int8_t* )column.LeftColumn,
                            ( const int8_t* )column.RightColumn, not_equal_to_t< int8_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int8_t >, nullable_type< int8_t >, not_equal_to_t< nullable_type< int8_t > > >(
                                    ( const nullable_type< int8_t >* )column.LeftColumn, ( const nullable_type< int8_t >* )column.RightColumn,
                                    not_equal_to_t< nullable_type< int8_t > >() );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int16_t, int16_t, not_equal_to_t< int16_t > >( ( const int16_t* )column.LeftColumn,
                            ( const int16_t* )column.RightColumn, not_equal_to_t< int16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int16_t >, nullable_type< int16_t >,
                            not_equal_to_t< nullable_type< int16_t > > >( ( const nullable_type< int16_t >* )column.LeftColumn,
                            ( const nullable_type< int16_t >* )column.RightColumn, not_equal_to_t< nullable_type< int16_t > >() );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int32_t, int32_t, not_equal_to_t< int32_t > >( ( const int32_t* )column.LeftColumn,
                            ( const int32_t* )column.RightColumn, not_equal_to_t< int32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int32_t >, nullable_type< int32_t >,
                            not_equal_to_t< nullable_type< int32_t > > >( ( const nullable_type< int32_t >* )column.LeftColumn,
                            ( const nullable_type< int32_t >* )column.RightColumn, not_equal_to_t< nullable_type< int32_t > >() );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int64_t, int64_t, not_equal_to_t< int64_t > >( ( const int64_t* )column.LeftColumn,
                            ( const int64_t* )column.RightColumn, not_equal_to_t< int64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int64_t >, nullable_type< int64_t >,
                            not_equal_to_t< nullable_type< int64_t > > >( ( const nullable_type< int64_t >* )column.LeftColumn,
                            ( const nullable_type< int64_t >* )column.RightColumn, not_equal_to_t< nullable_type< int64_t > >() );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint8_t, uint8_t, not_equal_to_t< uint8_t > >( ( const uint8_t* )column.LeftColumn,
                            ( const uint8_t* )column.RightColumn, not_equal_to_t< uint8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint8_t >, nullable_type< uint8_t >,
                            not_equal_to_t< nullable_type< uint8_t > > >( ( const nullable_type< uint8_t >* )column.LeftColumn,
                            ( const nullable_type< uint8_t >* )column.RightColumn, not_equal_to_t< nullable_type< uint8_t > >() );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint16_t, uint16_t, not_equal_to_t< uint16_t > >( ( const uint16_t* )column.LeftColumn,
                            ( const uint16_t* )column.RightColumn, not_equal_to_t< uint16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint16_t >, nullable_type< uint16_t >,
                            not_equal_to_t< nullable_type< uint16_t > > >( ( const nullable_type< uint16_t >* )column.LeftColumn,
                            ( const nullable_type< uint16_t >* )column.RightColumn, not_equal_to_t< nullable_type< uint16_t > >() );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint32_t, uint32_t, not_equal_to_t< uint32_t > >( ( const uint32_t* )column.LeftColumn,
                            ( const uint32_t* )column.RightColumn, not_equal_to_t< uint32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint32_t >, nullable_type< uint32_t >,
                            not_equal_to_t< nullable_type< uint32_t > > >( ( const nullable_type< uint32_t >* )column.LeftColumn,
                            ( const nullable_type< uint32_t >* )column.RightColumn, not_equal_to_t< nullable_type< uint32_t > >() );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint64_t, uint64_t, not_equal_to_t< uint64_t > >( ( const uint64_t* )column.LeftColumn,
                            ( const uint64_t* )column.RightColumn, not_equal_to_t< uint64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint64_t >, nullable_type< uint64_t >,
                            not_equal_to_t< nullable_type< uint64_t > > >( ( const nullable_type< uint64_t >* )column.LeftColumn,
                            ( const nullable_type< uint64_t >* )column.RightColumn, not_equal_to_t< nullable_type< uint64_t > >() );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< float, float, not_equal_to_t< float > >( ( const float* )column.LeftColumn,
                            ( const float* )column.RightColumn, not_equal_to_t< float >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< float >, nullable_type< float >, not_equal_to_t< nullable_type< float > > >(
                            ( const nullable_type< float >* )column.LeftColumn, ( const nullable_type< float >* )column.RightColumn,
                            not_equal_to_t< nullable_type< float > >() );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< double, double, not_equal_to_t< double > >( ( const double* )column.LeftColumn,
                            ( const double* )column.RightColumn, not_equal_to_t< double >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< double >, nullable_type< double >, not_equal_to_t< nullable_type< double > > >(
                                    ( const nullable_type< double >* )column.LeftColumn, ( const nullable_type< double >* )column.RightColumn,
                                    not_equal_to_t< nullable_type< double > >() );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< Decimal, Decimal, not_equal_to_t< Decimal > >( ( const Decimal* )column.LeftColumn,
                            ( const Decimal* )column.RightColumn, not_equal_to_t< Decimal >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< Decimal >, nullable_type< Decimal >,
                            not_equal_to_t< nullable_type< Decimal > > >( ( const nullable_type< Decimal >* )column.LeftColumn,
                            ( const nullable_type< Decimal >* )column.RightColumn, not_equal_to_t< nullable_type< Decimal > >() );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDate, AriesDate, not_equal_to_t< AriesDate > >( ( const AriesDate* )column.LeftColumn,
                            ( const AriesDate* )column.RightColumn, not_equal_to_t< AriesDate >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDate >, nullable_type< AriesDate >,
                            not_equal_to_t< nullable_type< AriesDate > > >( ( const nullable_type< AriesDate >* )column.LeftColumn,
                            ( const nullable_type< AriesDate >* )column.RightColumn, not_equal_to_t< nullable_type< AriesDate > >() );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDatetime, AriesDatetime, not_equal_to_t< AriesDatetime > >(
                            ( const AriesDatetime* )column.LeftColumn, ( const AriesDatetime* )column.RightColumn,
                            not_equal_to_t< AriesDatetime >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDatetime >, nullable_type< AriesDatetime >,
                            not_equal_to_t< nullable_type< AriesDatetime > > >( ( const nullable_type< AriesDatetime >* )column.LeftColumn,
                            ( const nullable_type< AriesDatetime >* )column.RightColumn, not_equal_to_t< nullable_type< AriesDatetime > >() );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesTimestamp, AriesTimestamp, not_equal_to_t< AriesTimestamp > >(
                            ( const AriesTimestamp* )column.LeftColumn, ( const AriesTimestamp* )column.RightColumn,
                            not_equal_to_t< AriesTimestamp >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesTimestamp >, nullable_type< AriesTimestamp >,
                            not_equal_to_t< nullable_type< AriesTimestamp > > >( ( const nullable_type< AriesTimestamp >* )column.LeftColumn,
                            ( const nullable_type< AriesTimestamp >* )column.RightColumn, not_equal_to_t< nullable_type< AriesTimestamp > >() );
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesYear, AriesYear, not_equal_to_t< AriesYear > >( ( const AriesYear* )column.LeftColumn,
                            ( const AriesYear* )column.RightColumn, not_equal_to_t< AriesYear >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesYear >, nullable_type< AriesYear >,
                            not_equal_to_t< nullable_type< AriesYear > > >( ( const nullable_type< AriesYear >* )column.LeftColumn,
                            ( const nullable_type< AriesYear >* )column.RightColumn, not_equal_to_t< nullable_type< AriesYear > >() );
                break;
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                break;
        }
        return columnPair;
    }

    __device__ IComparableColumnPair* create_ge_comparator( const ColumnPairs& column )
    {
        IComparableColumnPair* columnPair = nullptr;
        AriesColumnType type = column.ColumnType;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, greater_equal_t_str< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), greater_equal_t_str< false, false >() );
                else
                    columnPair = new ComparableColumnPair< char, char, greater_equal_t_str< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), greater_equal_t_str< true, true >() );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, greater_equal_t_CompactDecimal< false, false > >(
                            ( const char* )column.LeftColumn, ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            greater_equal_t_CompactDecimal< false, false >( type.DataType.Precision, type.DataType.Scale ) );
                else
                    columnPair = new ComparableColumnPair< char, char, greater_equal_t_CompactDecimal< true, true > >(
                            ( const char* )column.LeftColumn, ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            greater_equal_t_CompactDecimal< true, true >( type.DataType.Precision, type.DataType.Scale ) );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int8_t, int8_t, greater_equal_t< int8_t > >( ( const int8_t* )column.LeftColumn,
                            ( const int8_t* )column.RightColumn, greater_equal_t< int8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int8_t >, nullable_type< int8_t >,
                            greater_equal_t< nullable_type< int8_t > > >( ( const nullable_type< int8_t >* )column.LeftColumn,
                            ( const nullable_type< int8_t >* )column.RightColumn, greater_equal_t< nullable_type< int8_t > >() );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int16_t, int16_t, greater_equal_t< int16_t > >( ( const int16_t* )column.LeftColumn,
                            ( const int16_t* )column.RightColumn, greater_equal_t< int16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int16_t >, nullable_type< int16_t >,
                            greater_equal_t< nullable_type< int16_t > > >( ( const nullable_type< int16_t >* )column.LeftColumn,
                            ( const nullable_type< int16_t >* )column.RightColumn, greater_equal_t< nullable_type< int16_t > >() );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int32_t, int32_t, greater_equal_t< int32_t > >( ( const int32_t* )column.LeftColumn,
                            ( const int32_t* )column.RightColumn, greater_equal_t< int32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int32_t >, nullable_type< int32_t >,
                            greater_equal_t< nullable_type< int32_t > > >( ( const nullable_type< int32_t >* )column.LeftColumn,
                            ( const nullable_type< int32_t >* )column.RightColumn, greater_equal_t< nullable_type< int32_t > >() );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int64_t, int64_t, greater_equal_t< int64_t > >( ( const int64_t* )column.LeftColumn,
                            ( const int64_t* )column.RightColumn, greater_equal_t< int64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int64_t >, nullable_type< int64_t >,
                            greater_equal_t< nullable_type< int64_t > > >( ( const nullable_type< int64_t >* )column.LeftColumn,
                            ( const nullable_type< int64_t >* )column.RightColumn, greater_equal_t< nullable_type< int64_t > >() );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint8_t, uint8_t, greater_equal_t< uint8_t > >( ( const uint8_t* )column.LeftColumn,
                            ( const uint8_t* )column.RightColumn, greater_equal_t< uint8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint8_t >, nullable_type< uint8_t >,
                            greater_equal_t< nullable_type< uint8_t > > >( ( const nullable_type< uint8_t >* )column.LeftColumn,
                            ( const nullable_type< uint8_t >* )column.RightColumn, greater_equal_t< nullable_type< uint8_t > >() );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint16_t, uint16_t, greater_equal_t< uint16_t > >( ( const uint16_t* )column.LeftColumn,
                            ( const uint16_t* )column.RightColumn, greater_equal_t< uint16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint16_t >, nullable_type< uint16_t >,
                            greater_equal_t< nullable_type< uint16_t > > >( ( const nullable_type< uint16_t >* )column.LeftColumn,
                            ( const nullable_type< uint16_t >* )column.RightColumn, greater_equal_t< nullable_type< uint16_t > >() );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint32_t, uint32_t, greater_equal_t< uint32_t > >( ( const uint32_t* )column.LeftColumn,
                            ( const uint32_t* )column.RightColumn, greater_equal_t< uint32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint32_t >, nullable_type< uint32_t >,
                            greater_equal_t< nullable_type< uint32_t > > >( ( const nullable_type< uint32_t >* )column.LeftColumn,
                            ( const nullable_type< uint32_t >* )column.RightColumn, greater_equal_t< nullable_type< uint32_t > >() );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint64_t, uint64_t, greater_equal_t< uint64_t > >( ( const uint64_t* )column.LeftColumn,
                            ( const uint64_t* )column.RightColumn, greater_equal_t< uint64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint64_t >, nullable_type< uint64_t >,
                            greater_equal_t< nullable_type< uint64_t > > >( ( const nullable_type< uint64_t >* )column.LeftColumn,
                            ( const nullable_type< uint64_t >* )column.RightColumn, greater_equal_t< nullable_type< uint64_t > >() );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< float, float, greater_equal_t< float > >( ( const float* )column.LeftColumn,
                            ( const float* )column.RightColumn, greater_equal_t< float >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< float >, nullable_type< float >, greater_equal_t< nullable_type< float > > >(
                                    ( const nullable_type< float >* )column.LeftColumn, ( const nullable_type< float >* )column.RightColumn,
                                    greater_equal_t< nullable_type< float > >() );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< double, double, greater_equal_t< double > >( ( const double* )column.LeftColumn,
                            ( const double* )column.RightColumn, greater_equal_t< double >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< double >, nullable_type< double >,
                            greater_equal_t< nullable_type< double > > >( ( const nullable_type< double >* )column.LeftColumn,
                            ( const nullable_type< double >* )column.RightColumn, greater_equal_t< nullable_type< double > >() );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< Decimal, Decimal, greater_equal_t< Decimal > >( ( const Decimal* )column.LeftColumn,
                            ( const Decimal* )column.RightColumn, greater_equal_t< Decimal >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< Decimal >, nullable_type< Decimal >,
                            greater_equal_t< nullable_type< Decimal > > >( ( const nullable_type< Decimal >* )column.LeftColumn,
                            ( const nullable_type< Decimal >* )column.RightColumn, greater_equal_t< nullable_type< Decimal > >() );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDate, AriesDate, greater_equal_t< AriesDate > >(
                            ( const AriesDate* )column.LeftColumn, ( const AriesDate* )column.RightColumn, greater_equal_t< AriesDate >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDate >, nullable_type< AriesDate >,
                            greater_equal_t< nullable_type< AriesDate > > >( ( const nullable_type< AriesDate >* )column.LeftColumn,
                            ( const nullable_type< AriesDate >* )column.RightColumn, greater_equal_t< nullable_type< AriesDate > >() );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDatetime, AriesDatetime, greater_equal_t< AriesDatetime > >(
                            ( const AriesDatetime* )column.LeftColumn, ( const AriesDatetime* )column.RightColumn,
                            greater_equal_t< AriesDatetime >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDatetime >, nullable_type< AriesDatetime >,
                            greater_equal_t< nullable_type< AriesDatetime > > >( ( const nullable_type< AriesDatetime >* )column.LeftColumn,
                            ( const nullable_type< AriesDatetime >* )column.RightColumn, greater_equal_t< nullable_type< AriesDatetime > >() );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesTimestamp, AriesTimestamp, greater_equal_t< AriesTimestamp > >(
                            ( const AriesTimestamp* )column.LeftColumn, ( const AriesTimestamp* )column.RightColumn,
                            greater_equal_t< AriesTimestamp >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesTimestamp >, nullable_type< AriesTimestamp >,
                            greater_equal_t< nullable_type< AriesTimestamp > > >( ( const nullable_type< AriesTimestamp >* )column.LeftColumn,
                            ( const nullable_type< AriesTimestamp >* )column.RightColumn, greater_equal_t< nullable_type< AriesTimestamp > >() );
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesYear, AriesYear, greater_equal_t< AriesYear > >(
                            ( const AriesYear* )column.LeftColumn, ( const AriesYear* )column.RightColumn, greater_equal_t< AriesYear >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesYear >, nullable_type< AriesYear >,
                            greater_equal_t< nullable_type< AriesYear > > >( ( const nullable_type< AriesYear >* )column.LeftColumn,
                            ( const nullable_type< AriesYear >* )column.RightColumn, greater_equal_t< nullable_type< AriesYear > >() );
                break;
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                break;
        }
        return columnPair;
    }

    __device__ IComparableColumnPair* create_le_comparator( const ColumnPairs& column )
    {
        IComparableColumnPair* columnPair = nullptr;
        AriesColumnType type = column.ColumnType;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, less_equal_t_str< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), less_equal_t_str< false, false >() );
                else
                    columnPair = new ComparableColumnPair< char, char, less_equal_t_str< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), less_equal_t_str< true, true >() );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, less_equal_t_CompactDecimal< false, false > >(
                            ( const char* )column.LeftColumn, ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            less_equal_t_CompactDecimal< false, false >( type.DataType.Precision, type.DataType.Scale ) );
                else
                    columnPair = new ComparableColumnPair< char, char, less_equal_t_CompactDecimal< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            less_equal_t_CompactDecimal< true, true >( type.DataType.Precision, type.DataType.Scale ) );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int8_t, int8_t, less_equal_t< int8_t > >( ( const int8_t* )column.LeftColumn,
                            ( const int8_t* )column.RightColumn, less_equal_t< int8_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int8_t >, nullable_type< int8_t >, less_equal_t< nullable_type< int8_t > > >(
                                    ( const nullable_type< int8_t >* )column.LeftColumn, ( const nullable_type< int8_t >* )column.RightColumn,
                                    less_equal_t< nullable_type< int8_t > >() );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int16_t, int16_t, less_equal_t< int16_t > >( ( const int16_t* )column.LeftColumn,
                            ( const int16_t* )column.RightColumn, less_equal_t< int16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int16_t >, nullable_type< int16_t >,
                            less_equal_t< nullable_type< int16_t > > >( ( const nullable_type< int16_t >* )column.LeftColumn,
                            ( const nullable_type< int16_t >* )column.RightColumn, less_equal_t< nullable_type< int16_t > >() );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int32_t, int32_t, less_equal_t< int32_t > >( ( const int32_t* )column.LeftColumn,
                            ( const int32_t* )column.RightColumn, less_equal_t< int32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int32_t >, nullable_type< int32_t >,
                            less_equal_t< nullable_type< int32_t > > >( ( const nullable_type< int32_t >* )column.LeftColumn,
                            ( const nullable_type< int32_t >* )column.RightColumn, less_equal_t< nullable_type< int32_t > >() );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int64_t, int64_t, less_equal_t< int64_t > >( ( const int64_t* )column.LeftColumn,
                            ( const int64_t* )column.RightColumn, less_equal_t< int64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int64_t >, nullable_type< int64_t >,
                            less_equal_t< nullable_type< int64_t > > >( ( const nullable_type< int64_t >* )column.LeftColumn,
                            ( const nullable_type< int64_t >* )column.RightColumn, less_equal_t< nullable_type< int64_t > >() );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint8_t, uint8_t, less_equal_t< uint8_t > >( ( const uint8_t* )column.LeftColumn,
                            ( const uint8_t* )column.RightColumn, less_equal_t< uint8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint8_t >, nullable_type< uint8_t >,
                            less_equal_t< nullable_type< uint8_t > > >( ( const nullable_type< uint8_t >* )column.LeftColumn,
                            ( const nullable_type< uint8_t >* )column.RightColumn, less_equal_t< nullable_type< uint8_t > >() );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint16_t, uint16_t, less_equal_t< uint16_t > >( ( const uint16_t* )column.LeftColumn,
                            ( const uint16_t* )column.RightColumn, less_equal_t< uint16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint16_t >, nullable_type< uint16_t >,
                            less_equal_t< nullable_type< uint16_t > > >( ( const nullable_type< uint16_t >* )column.LeftColumn,
                            ( const nullable_type< uint16_t >* )column.RightColumn, less_equal_t< nullable_type< uint16_t > >() );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint32_t, uint32_t, less_equal_t< uint32_t > >( ( const uint32_t* )column.LeftColumn,
                            ( const uint32_t* )column.RightColumn, less_equal_t< uint32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint32_t >, nullable_type< uint32_t >,
                            less_equal_t< nullable_type< uint32_t > > >( ( const nullable_type< uint32_t >* )column.LeftColumn,
                            ( const nullable_type< uint32_t >* )column.RightColumn, less_equal_t< nullable_type< uint32_t > >() );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint64_t, uint64_t, less_equal_t< uint64_t > >( ( const uint64_t* )column.LeftColumn,
                            ( const uint64_t* )column.RightColumn, less_equal_t< uint64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint64_t >, nullable_type< uint64_t >,
                            less_equal_t< nullable_type< uint64_t > > >( ( const nullable_type< uint64_t >* )column.LeftColumn,
                            ( const nullable_type< uint64_t >* )column.RightColumn, less_equal_t< nullable_type< uint64_t > >() );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< float, float, less_equal_t< float > >( ( const float* )column.LeftColumn,
                            ( const float* )column.RightColumn, less_equal_t< float >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< float >, nullable_type< float >, less_equal_t< nullable_type< float > > >(
                            ( const nullable_type< float >* )column.LeftColumn, ( const nullable_type< float >* )column.RightColumn,
                            less_equal_t< nullable_type< float > >() );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< double, double, less_equal_t< double > >( ( const double* )column.LeftColumn,
                            ( const double* )column.RightColumn, less_equal_t< double >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< double >, nullable_type< double >, less_equal_t< nullable_type< double > > >(
                                    ( const nullable_type< double >* )column.LeftColumn, ( const nullable_type< double >* )column.RightColumn,
                                    less_equal_t< nullable_type< double > >() );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< Decimal, Decimal, less_equal_t< Decimal > >( ( const Decimal* )column.LeftColumn,
                            ( const Decimal* )column.RightColumn, less_equal_t< Decimal >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< Decimal >, nullable_type< Decimal >,
                            less_equal_t< nullable_type< Decimal > > >( ( const nullable_type< Decimal >* )column.LeftColumn,
                            ( const nullable_type< Decimal >* )column.RightColumn, less_equal_t< nullable_type< Decimal > >() );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDate, AriesDate, less_equal_t< AriesDate > >( ( const AriesDate* )column.LeftColumn,
                            ( const AriesDate* )column.RightColumn, less_equal_t< AriesDate >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDate >, nullable_type< AriesDate >,
                            less_equal_t< nullable_type< AriesDate > > >( ( const nullable_type< AriesDate >* )column.LeftColumn,
                            ( const nullable_type< AriesDate >* )column.RightColumn, less_equal_t< nullable_type< AriesDate > >() );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDatetime, AriesDatetime, less_equal_t< AriesDatetime > >(
                            ( const AriesDatetime* )column.LeftColumn, ( const AriesDatetime* )column.RightColumn, less_equal_t< AriesDatetime >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDatetime >, nullable_type< AriesDatetime >,
                            less_equal_t< nullable_type< AriesDatetime > > >( ( const nullable_type< AriesDatetime >* )column.LeftColumn,
                            ( const nullable_type< AriesDatetime >* )column.RightColumn, less_equal_t< nullable_type< AriesDatetime > >() );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesTimestamp, AriesTimestamp, less_equal_t< AriesTimestamp > >(
                            ( const AriesTimestamp* )column.LeftColumn, ( const AriesTimestamp* )column.RightColumn,
                            less_equal_t< AriesTimestamp >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesTimestamp >, nullable_type< AriesTimestamp >,
                            less_equal_t< nullable_type< AriesTimestamp > > >( ( const nullable_type< AriesTimestamp >* )column.LeftColumn,
                            ( const nullable_type< AriesTimestamp >* )column.RightColumn, less_equal_t< nullable_type< AriesTimestamp > >() );
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesYear, AriesYear, less_equal_t< AriesYear > >( ( const AriesYear* )column.LeftColumn,
                            ( const AriesYear* )column.RightColumn, less_equal_t< AriesYear >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesYear >, nullable_type< AriesYear >,
                            less_equal_t< nullable_type< AriesYear > > >( ( const nullable_type< AriesYear >* )column.LeftColumn,
                            ( const nullable_type< AriesYear >* )column.RightColumn, less_equal_t< nullable_type< AriesYear > >() );
                break;
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                break;
        }
        return columnPair;
    }

    __device__ IComparableColumnPair* create_lt_comparator( const ColumnPairs& column )
    {
        IComparableColumnPair* columnPair = nullptr;
        AriesColumnType type = column.ColumnType;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, less_t_str< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), less_t_str< false, false >() );
                else
                    columnPair = new ComparableColumnPair< char, char, less_t_str< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), less_t_str< true, true >() );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, less_t_CompactDecimal< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            less_t_CompactDecimal< false, false >( type.DataType.Precision, type.DataType.Scale ) );
                else
                    columnPair = new ComparableColumnPair< char, char, less_t_CompactDecimal< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            less_t_CompactDecimal< true, true >( type.DataType.Precision, type.DataType.Scale ) );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int8_t, int8_t, less_t< int8_t > >( ( const int8_t* )column.LeftColumn,
                            ( const int8_t* )column.RightColumn, less_t< int8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int8_t >, nullable_type< int8_t >, less_t< nullable_type< int8_t > > >(
                            ( const nullable_type< int8_t >* )column.LeftColumn, ( const nullable_type< int8_t >* )column.RightColumn,
                            less_t< nullable_type< int8_t > >() );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int16_t, int16_t, less_t< int16_t > >( ( const int16_t* )column.LeftColumn,
                            ( const int16_t* )column.RightColumn, less_t< int16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int16_t >, nullable_type< int16_t >, less_t< nullable_type< int16_t > > >(
                            ( const nullable_type< int16_t >* )column.LeftColumn, ( const nullable_type< int16_t >* )column.RightColumn,
                            less_t< nullable_type< int16_t > >() );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int32_t, int32_t, less_t< int32_t > >( ( const int32_t* )column.LeftColumn,
                            ( const int32_t* )column.RightColumn, less_t< int32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int32_t >, nullable_type< int32_t >, less_t< nullable_type< int32_t > > >(
                            ( const nullable_type< int32_t >* )column.LeftColumn, ( const nullable_type< int32_t >* )column.RightColumn,
                            less_t< nullable_type< int32_t > >() );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int64_t, int64_t, less_t< int64_t > >( ( const int64_t* )column.LeftColumn,
                            ( const int64_t* )column.RightColumn, less_t< int64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int64_t >, nullable_type< int64_t >, less_t< nullable_type< int64_t > > >(
                            ( const nullable_type< int64_t >* )column.LeftColumn, ( const nullable_type< int64_t >* )column.RightColumn,
                            less_t< nullable_type< int64_t > >() );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint8_t, uint8_t, less_t< uint8_t > >( ( const uint8_t* )column.LeftColumn,
                            ( const uint8_t* )column.RightColumn, less_t< uint8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint8_t >, nullable_type< uint8_t >, less_t< nullable_type< uint8_t > > >(
                            ( const nullable_type< uint8_t >* )column.LeftColumn, ( const nullable_type< uint8_t >* )column.RightColumn,
                            less_t< nullable_type< uint8_t > >() );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint16_t, uint16_t, less_t< uint16_t > >( ( const uint16_t* )column.LeftColumn,
                            ( const uint16_t* )column.RightColumn, less_t< uint16_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< uint16_t >, nullable_type< uint16_t >, less_t< nullable_type< uint16_t > > >(
                                    ( const nullable_type< uint16_t >* )column.LeftColumn, ( const nullable_type< uint16_t >* )column.RightColumn,
                                    less_t< nullable_type< uint16_t > >() );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint32_t, uint32_t, less_t< uint32_t > >( ( const uint32_t* )column.LeftColumn,
                            ( const uint32_t* )column.RightColumn, less_t< uint32_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< uint32_t >, nullable_type< uint32_t >, less_t< nullable_type< uint32_t > > >(
                                    ( const nullable_type< uint32_t >* )column.LeftColumn, ( const nullable_type< uint32_t >* )column.RightColumn,
                                    less_t< nullable_type< uint32_t > >() );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint64_t, uint64_t, less_t< uint64_t > >( ( const uint64_t* )column.LeftColumn,
                            ( const uint64_t* )column.RightColumn, less_t< uint64_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< uint64_t >, nullable_type< uint64_t >, less_t< nullable_type< uint64_t > > >(
                                    ( const nullable_type< uint64_t >* )column.LeftColumn, ( const nullable_type< uint64_t >* )column.RightColumn,
                                    less_t< nullable_type< uint64_t > >() );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< float, float, less_t< float > >( ( const float* )column.LeftColumn,
                            ( const float* )column.RightColumn, less_t< float >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< float >, nullable_type< float >, less_t< nullable_type< float > > >(
                            ( const nullable_type< float >* )column.LeftColumn, ( const nullable_type< float >* )column.RightColumn,
                            less_t< nullable_type< float > >() );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< double, double, less_t< double > >( ( const double* )column.LeftColumn,
                            ( const double* )column.RightColumn, less_t< double >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< double >, nullable_type< double >, less_t< nullable_type< double > > >(
                            ( const nullable_type< double >* )column.LeftColumn, ( const nullable_type< double >* )column.RightColumn,
                            less_t< nullable_type< double > >() );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< Decimal, Decimal, less_t< Decimal > >( ( const Decimal* )column.LeftColumn,
                            ( const Decimal* )column.RightColumn, less_t< Decimal >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< Decimal >, nullable_type< Decimal >, less_t< nullable_type< Decimal > > >(
                            ( const nullable_type< Decimal >* )column.LeftColumn, ( const nullable_type< Decimal >* )column.RightColumn,
                            less_t< nullable_type< Decimal > >() );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDate, AriesDate, less_t< AriesDate > >( ( const AriesDate* )column.LeftColumn,
                            ( const AriesDate* )column.RightColumn, less_t< AriesDate >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDate >, nullable_type< AriesDate >,
                            less_t< nullable_type< AriesDate > > >( ( const nullable_type< AriesDate >* )column.LeftColumn,
                            ( const nullable_type< AriesDate >* )column.RightColumn, less_t< nullable_type< AriesDate > >() );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDatetime, AriesDatetime, less_t< AriesDatetime > >(
                            ( const AriesDatetime* )column.LeftColumn, ( const AriesDatetime* )column.RightColumn, less_t< AriesDatetime >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDatetime >, nullable_type< AriesDatetime >,
                            less_t< nullable_type< AriesDatetime > > >( ( const nullable_type< AriesDatetime >* )column.LeftColumn,
                            ( const nullable_type< AriesDatetime >* )column.RightColumn, less_t< nullable_type< AriesDatetime > >() );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesTimestamp, AriesTimestamp, less_t< AriesTimestamp > >(
                            ( const AriesTimestamp* )column.LeftColumn, ( const AriesTimestamp* )column.RightColumn, less_t< AriesTimestamp >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesTimestamp >, nullable_type< AriesTimestamp >,
                            less_t< nullable_type< AriesTimestamp > > >( ( const nullable_type< AriesTimestamp >* )column.LeftColumn,
                            ( const nullable_type< AriesTimestamp >* )column.RightColumn, less_t< nullable_type< AriesTimestamp > >() );
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesYear, AriesYear, less_t< AriesYear > >( ( const AriesYear* )column.LeftColumn,
                            ( const AriesYear* )column.RightColumn, less_t< AriesYear >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesYear >, nullable_type< AriesYear >,
                            less_t< nullable_type< AriesYear > > >( ( const nullable_type< AriesYear >* )column.LeftColumn,
                            ( const nullable_type< AriesYear >* )column.RightColumn, less_t< nullable_type< AriesYear > >() );
                break;
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                break;
        }
        return columnPair;
    }

    __device__ IComparableColumnPair* create_gt_comparator( const ColumnPairs& column )
    {
        IComparableColumnPair* columnPair = nullptr;
        AriesColumnType type = column.ColumnType;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, greater_t_str< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), greater_t_str< false, false >() );
                else
                    columnPair = new ComparableColumnPair< char, char, greater_t_str< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(), greater_t_str< true, true >() );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< char, char, greater_t_CompactDecimal< false, false > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            greater_t_CompactDecimal< false, false >( type.DataType.Precision, type.DataType.Scale ) );
                else
                    columnPair = new ComparableColumnPair< char, char, greater_t_CompactDecimal< true, true > >( ( const char* )column.LeftColumn,
                            ( const char* )column.RightColumn, type.GetDataTypeSize(),
                            greater_t_CompactDecimal< true, true >( type.DataType.Precision, type.DataType.Scale ) );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int8_t, int8_t, greater_t< int8_t > >( ( const int8_t* )column.LeftColumn,
                            ( const int8_t* )column.RightColumn, greater_t< int8_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< int8_t >, nullable_type< int8_t >, greater_t< nullable_type< int8_t > > >(
                            ( const nullable_type< int8_t >* )column.LeftColumn, ( const nullable_type< int8_t >* )column.RightColumn,
                            greater_t< nullable_type< int8_t > >() );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int16_t, int16_t, greater_t< int16_t > >( ( const int16_t* )column.LeftColumn,
                            ( const int16_t* )column.RightColumn, greater_t< int16_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int16_t >, nullable_type< int16_t >, greater_t< nullable_type< int16_t > > >(
                                    ( const nullable_type< int16_t >* )column.LeftColumn, ( const nullable_type< int16_t >* )column.RightColumn,
                                    greater_t< nullable_type< int16_t > >() );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int32_t, int32_t, greater_t< int32_t > >( ( const int32_t* )column.LeftColumn,
                            ( const int32_t* )column.RightColumn, greater_t< int32_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int32_t >, nullable_type< int32_t >, greater_t< nullable_type< int32_t > > >(
                                    ( const nullable_type< int32_t >* )column.LeftColumn, ( const nullable_type< int32_t >* )column.RightColumn,
                                    greater_t< nullable_type< int32_t > >() );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< int64_t, int64_t, greater_t< int64_t > >( ( const int64_t* )column.LeftColumn,
                            ( const int64_t* )column.RightColumn, greater_t< int64_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< int64_t >, nullable_type< int64_t >, greater_t< nullable_type< int64_t > > >(
                                    ( const nullable_type< int64_t >* )column.LeftColumn, ( const nullable_type< int64_t >* )column.RightColumn,
                                    greater_t< nullable_type< int64_t > >() );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint8_t, uint8_t, greater_t< uint8_t > >( ( const uint8_t* )column.LeftColumn,
                            ( const uint8_t* )column.RightColumn, greater_t< uint8_t >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< uint8_t >, nullable_type< uint8_t >, greater_t< nullable_type< uint8_t > > >(
                                    ( const nullable_type< uint8_t >* )column.LeftColumn, ( const nullable_type< uint8_t >* )column.RightColumn,
                                    greater_t< nullable_type< uint8_t > >() );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint16_t, uint16_t, greater_t< uint16_t > >( ( const uint16_t* )column.LeftColumn,
                            ( const uint16_t* )column.RightColumn, greater_t< uint16_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint16_t >, nullable_type< uint16_t >,
                            greater_t< nullable_type< uint16_t > > >( ( const nullable_type< uint16_t >* )column.LeftColumn,
                            ( const nullable_type< uint16_t >* )column.RightColumn, greater_t< nullable_type< uint16_t > >() );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint32_t, uint32_t, greater_t< uint32_t > >( ( const uint32_t* )column.LeftColumn,
                            ( const uint32_t* )column.RightColumn, greater_t< uint32_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint32_t >, nullable_type< uint32_t >,
                            greater_t< nullable_type< uint32_t > > >( ( const nullable_type< uint32_t >* )column.LeftColumn,
                            ( const nullable_type< uint32_t >* )column.RightColumn, greater_t< nullable_type< uint32_t > >() );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< uint64_t, uint64_t, greater_t< uint64_t > >( ( const uint64_t* )column.LeftColumn,
                            ( const uint64_t* )column.RightColumn, greater_t< uint64_t >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< uint64_t >, nullable_type< uint64_t >,
                            greater_t< nullable_type< uint64_t > > >( ( const nullable_type< uint64_t >* )column.LeftColumn,
                            ( const nullable_type< uint64_t >* )column.RightColumn, greater_t< nullable_type< uint64_t > >() );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< float, float, greater_t< float > >( ( const float* )column.LeftColumn,
                            ( const float* )column.RightColumn, greater_t< float >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< float >, nullable_type< float >, greater_t< nullable_type< float > > >(
                            ( const nullable_type< float >* )column.LeftColumn, ( const nullable_type< float >* )column.RightColumn,
                            greater_t< nullable_type< float > >() );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< double, double, greater_t< double > >( ( const double* )column.LeftColumn,
                            ( const double* )column.RightColumn, greater_t< double >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< double >, nullable_type< double >, greater_t< nullable_type< double > > >(
                            ( const nullable_type< double >* )column.LeftColumn, ( const nullable_type< double >* )column.RightColumn,
                            greater_t< nullable_type< double > >() );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< Decimal, Decimal, greater_t< Decimal > >( ( const Decimal* )column.LeftColumn,
                            ( const Decimal* )column.RightColumn, greater_t< Decimal >() );
                else
                    columnPair =
                            new ComparableColumnPair< nullable_type< Decimal >, nullable_type< Decimal >, greater_t< nullable_type< Decimal > > >(
                                    ( const nullable_type< Decimal >* )column.LeftColumn, ( const nullable_type< Decimal >* )column.RightColumn,
                                    greater_t< nullable_type< Decimal > >() );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDate, AriesDate, greater_t< AriesDate > >( ( const AriesDate* )column.LeftColumn,
                            ( const AriesDate* )column.RightColumn, greater_t< AriesDate >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDate >, nullable_type< AriesDate >,
                            greater_t< nullable_type< AriesDate > > >( ( const nullable_type< AriesDate >* )column.LeftColumn,
                            ( const nullable_type< AriesDate >* )column.RightColumn, greater_t< nullable_type< AriesDate > >() );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesDatetime, AriesDatetime, greater_t< AriesDatetime > >(
                            ( const AriesDatetime* )column.LeftColumn, ( const AriesDatetime* )column.RightColumn, greater_t< AriesDatetime >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesDatetime >, nullable_type< AriesDatetime >,
                            greater_t< nullable_type< AriesDatetime > > >( ( const nullable_type< AriesDatetime >* )column.LeftColumn,
                            ( const nullable_type< AriesDatetime >* )column.RightColumn, greater_t< nullable_type< AriesDatetime > >() );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesTimestamp, AriesTimestamp, greater_t< AriesTimestamp > >(
                            ( const AriesTimestamp* )column.LeftColumn, ( const AriesTimestamp* )column.RightColumn, greater_t< AriesTimestamp >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesTimestamp >, nullable_type< AriesTimestamp >,
                            greater_t< nullable_type< AriesTimestamp > > >( ( const nullable_type< AriesTimestamp >* )column.LeftColumn,
                            ( const nullable_type< AriesTimestamp >* )column.RightColumn, greater_t< nullable_type< AriesTimestamp > >() );
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !type.HasNull )
                    columnPair = new ComparableColumnPair< AriesYear, AriesYear, greater_t< AriesYear > >( ( const AriesYear* )column.LeftColumn,
                            ( const AriesYear* )column.RightColumn, greater_t< AriesYear >() );
                else
                    columnPair = new ComparableColumnPair< nullable_type< AriesYear >, nullable_type< AriesYear >,
                            greater_t< nullable_type< AriesYear > > >( ( const nullable_type< AriesYear >* )column.LeftColumn,
                            ( const nullable_type< AriesYear >* )column.RightColumn, greater_t< nullable_type< AriesYear > >() );
                break;
            }
            default:
                assert( 0 ); //FIXME need support all data types.
                break;
        }
        return columnPair;
    }

    __device__ void create_comparators( const ColumnPairs* columns, int columnCount, IComparableColumnPair** output )
    {
        for( int i = 0; i < columnCount; ++i )
        {
            const auto& column = columns[i];
            switch( column.OpType )
            {
                case AriesComparisonOpType::EQ:
                    output[i] = create_eq_comparator( column );
                    break;
                case AriesComparisonOpType::NE:
                    output[i] = create_ne_comparator( column );
                    break;
                case AriesComparisonOpType::GT:
                    output[i] = create_gt_comparator( column );
                    break;
                case AriesComparisonOpType::LT:
                    output[i] = create_lt_comparator( column );
                    break;
                case AriesComparisonOpType::GE:
                    output[i] = create_ge_comparator( column );
                    break;
                case AriesComparisonOpType::LE:
                    output[i] = create_le_comparator( column );
                    break;
                default:
                    assert( 0 );
                    break;
            }
        }
    }

    void sort_based_semi_join_helper_dyn_code( const index_t* lower_data, const index_t* upper_data,
            const index_t* vals_a, // indices of sorted left data
            const index_t* vals_b, // indices of sorted right data
            int a_count,
            const vector< AriesDynamicCodeComparator >& comparators,
            const vector< CUmoduleSPtr >& modules,
            const char *functionName,
            const AriesColumnDataIterator *input,
            const std::vector< AriesDataBufferSPtr >& constValues,
            AriesBool* output,
            context_t& context )
    {
        if( !functionName || functionName[0] == '\0' )
        {
            auto k = [=] ARIES_DEVICE(int index)
            {
                output[ vals_a[ index ] ] = bool( upper_data[index] - lower_data[index] );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
        }
        else
        {
            cudaMemset( output, 0, a_count * sizeof( AriesBool ) );
            mem_t< int > scanned_sizes( a_count );
            managed_mem_t< int64_t > count( 1, context );

            transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                    {
                        return upper_data[index] - lower_data[index];
                    }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );
            context.synchronize();
            // Allocate an int2 output array and use load-balancing search to compute
            // the join.
            size_t join_count = count.data()[0];
            if( join_count > INT_MAX )
            {
                printf( "join count=%lu\n", join_count );
                ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "semi join's result too big" );
            }
            if( join_count > 0 )
            {
                mem_t< index_t > left_indices( join_count );
                mem_t< index_t > right_indices( join_count );
                index_t* left_indices_data = left_indices.data();
                index_t* right_indices_data = right_indices.data();
                // Use load-balancing search on the segmens. The output is a pair with
                // a_index = seg and b_index = lower_data[seg] + rank.
                auto k = [=]ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
                {
                    left_indices_data[ index ] = vals_a[ seg ];
                    right_indices_data[ index ] = vals_b[ get<0>(lower) + rank ];
                };
                transform_lbs< empty_t >( k, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );

                AriesBoolArraySPtr outputFlags = make_shared< AriesBoolArray >();
                outputFlags->AllocArray( join_count );
                AriesBool* outputFlagsData = outputFlags->GetData();

                auto constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValues.size() );
                for ( int i = 0; i < constValues.size(); ++i )
                    ( *constValueArr )[ i ] = constValues[ i ]->GetData();
                constValueArr->PrefetchToGpu();

                AriesDynamicKernelManager::GetInstance().CallKernel( modules,
                                                                     functionName,
                                                                     input,
                                                                     left_indices_data,
                                                                     right_indices_data,
                                                                     join_count,
                                                                     ( const int8_t** )constValueArr->GetData(),
                                                                     comparators,
                                                                     ( int8_t* )outputFlagsData );

                auto k2 = [=] ARIES_DEVICE(int index)
                {
                    if ( outputFlagsData[ index ] )
                    // left_indices_data[ index ] is the index into the original left data,
                    output[ left_indices_data[ index ] ] = true;
                };
                transform< launch_box_t< arch_52_cta< 256, 15 > > >( k2, join_count, context );
            }
        }
        context.synchronize();
    }

    bool has_null_value( char* data, size_t len, int count, context_t& context )
    {
        managed_mem_t< int > ret( 1, context );
        int* pValue = ret.data();
        *pValue = 0;
        transform< 256, 5 >( [=] ARIES_DEVICE(int index)
        {
            if( !data[ index * len ] )
                atomicExch( pValue, 1 );
        }, count, context );
        context.synchronize();
        return *pValue == 1;
    }

    AriesBoolArraySPtr sort_based_anti_join( char* a, int a_count, char* b, int b_count, int len, const JoinDynamicCodeParams* joinDynamicCodeParams,
            context_t& context, const int32_t* left_associated, const int32_t* right_associated )
    {
        auto result = sort_based_semi_join( a, a_count, b, b_count, len, less_t_str_null_smaller_join< false, false >(),
                less_t_str_null_smaller< false, false >(), joinDynamicCodeParams, context, left_associated, right_associated );

        auto data = result->GetData();
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[ index ] = !data[ index ].is_true();
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
        return result;
    }

    AriesBoolArraySPtr sort_based_anti_join_has_null( char* a, int a_count, char* b, int b_count, int len,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated, const int32_t* right_associated, bool isNotIn )
    {
        AriesBoolArraySPtr result;
        if( isNotIn )
        {
            if( has_null_value( b, len, b_count, context ) )
                result = std::make_shared< AriesBoolArray >( a_count, true );
            else
            {
                mem_t< int8_t > nullFlags( a_count );
                int8_t* pFlags = nullFlags.data();
                transform< 256, 5 >( [=] ARIES_DEVICE(int index)
                {
                    if( !a[ index * len ] )
                        pFlags[ index ] = 1;
                    else
                        pFlags[ index ] = 0;
                }, a_count, context );

                result = sort_based_semi_join( a, a_count, b, b_count, len, less_t_str_null_smaller_join< true, true >(),
                        less_t_str_null_smaller< true, true >(), joinDynamicCodeParams, context, left_associated, right_associated );

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
            result = sort_based_semi_join( a, a_count, b, b_count, len, less_t_str_null_smaller_join< true, true >(),
                    less_t_str_null_smaller< true, true >(), joinDynamicCodeParams, context, left_associated, right_associated );

            auto data = result->GetData();
            auto k = [=] ARIES_DEVICE(int index)
            {
                data[index] = !data[index].is_true();
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
        }
        return result;
    }

    AriesBoolArraySPtr sort_based_anti_join( CompactDecimal* a, int a_count, CompactDecimal* b, int b_count, int len, AriesColumnType leftType,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated, const int32_t* right_associated )
    {
        auto result = sort_based_semi_join( ( char* )a, a_count, ( char* )b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< false >( leftType.DataType.Precision, leftType.DataType.Scale ),
                less_t_CompactDecimal_null_smaller< false >( leftType.DataType.Precision, leftType.DataType.Scale ), joinDynamicCodeParams, context,
                left_associated, right_associated );

        auto data = result->GetData();
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[index] = !data[index].is_true();
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
        return result;
    }

    AriesBoolArraySPtr sort_based_anti_join_has_null( CompactDecimal* a, int a_count, CompactDecimal* b, int b_count, int len,
            AriesColumnType leftType, const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated,
            const int32_t* right_associated, bool isNotIn )
    {
        AriesBoolArraySPtr result;
        if( isNotIn )
        {
            if( has_null_value( (char*)b, len, b_count, context ) )
                result = std::make_shared< AriesBoolArray >( a_count, true );
            else
            {
                mem_t< int8_t > nullFlags( a_count );
                int8_t* pFlags = nullFlags.data();
                transform< 256, 5 >( [=] ARIES_DEVICE(int index)
                {
                    if( !( ( char* )a )[ index * len ] )
                        pFlags[ index ] = 1;
                    else
                        pFlags[ index ] = 0;
                }, a_count, context );

                result = sort_based_semi_join( ( char* )a, a_count, ( char* )b, b_count, len,
                        less_t_CompactDecimal_null_smaller_join< true >( leftType.DataType.Precision, leftType.DataType.Scale ),
                        less_t_CompactDecimal_null_smaller< true >( leftType.DataType.Precision, leftType.DataType.Scale ), joinDynamicCodeParams, context,
                        left_associated, right_associated );

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
            result = sort_based_semi_join( ( char* )a, a_count, ( char* )b, b_count, len,
                    less_t_CompactDecimal_null_smaller_join< true >( leftType.DataType.Precision, leftType.DataType.Scale ),
                    less_t_CompactDecimal_null_smaller< true >( leftType.DataType.Precision, leftType.DataType.Scale ), joinDynamicCodeParams, context,
                    left_associated, right_associated );

            auto data = result->GetData();
            auto k = [=] ARIES_DEVICE(int index)
            {
                data[index] = !data[index].is_true();
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, a_count, context );
        }
        return result;
    }

    AriesBoolArraySPtr sort_based_semi_join( int8_t* leftData, size_t leftTupleNum, int8_t* rightData, size_t rightTupleNum,
            AriesColumnType leftType, AriesColumnType rightType, const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context,
            const int32_t* left_associated, const int32_t* right_associated )
    {
        AriesBoolArraySPtr result;
        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_semi_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            less_t_str_null_smaller_join< false, false >(), less_t_str_null_smaller< false, false >(), joinDynamicCodeParams, context,
                            left_associated, right_associated );
                else
                    result = sort_based_semi_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            less_t_str_null_smaller_join< true, true >(), less_t_str_null_smaller< true, true >(), joinDynamicCodeParams, context,
                            left_associated, right_associated );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_semi_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            less_t_CompactDecimal_null_smaller_join< false >( leftType.DataType.Precision, leftType.DataType.Scale ),
                            less_t_CompactDecimal_null_smaller< false >( leftType.DataType.Precision, leftType.DataType.Scale ),
                            joinDynamicCodeParams, context, left_associated, right_associated );
                else
                    result = sort_based_semi_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            less_t_CompactDecimal_null_smaller_join< true >( leftType.DataType.Precision, leftType.DataType.Scale ),
                            less_t_CompactDecimal_null_smaller< true >( leftType.DataType.Precision, leftType.DataType.Scale ), joinDynamicCodeParams,
                            context, left_associated, right_associated );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_semi_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated );
            //         else
            //             result = sort_based_semi_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                            else
                                result = sort_based_semi_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_semi_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated );
                    else
                        result = sort_based_semi_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data leftType " + GetValueTypeAsString( leftType ) + " for semi-jion" );
        }

        return result;
    }

    AriesBoolArraySPtr sort_based_anti_join( int8_t* leftData, size_t leftTupleNum, int8_t* rightData, size_t rightTupleNum,
            AriesColumnType leftType, AriesColumnType rightType, const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context,
            const int32_t* left_associated, const int32_t* right_associated, bool isNotIn )
    {
        AriesBoolArraySPtr result;
        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_anti_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            joinDynamicCodeParams, context, left_associated, right_associated );
                else
                    result = sort_based_anti_join_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_anti_join( ( CompactDecimal* )leftData, leftTupleNum, ( CompactDecimal* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType, joinDynamicCodeParams, context, left_associated, right_associated );
                else
                    result = sort_based_anti_join_has_null( ( CompactDecimal* )leftData, leftTupleNum, ( CompactDecimal* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_anti_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //         else
            //             result = sort_based_anti_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                            else
                                result = sort_based_anti_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                        right_associated, isNotIn );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated, isNotIn );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated, isNotIn );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated, isNotIn );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_anti_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
                    else
                        result = sort_based_anti_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, joinDynamicCodeParams, context, left_associated,
                                right_associated, isNotIn );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data leftType " + GetValueTypeAsString( leftType ) + " for semi-jion" );
        }

        return result;
    }

    JoinPair sort_based_inner_join( char* a, int a_count, char* b, int b_count, int len, context_t& context, const int32_t* left_associated,
            const int32_t* right_associated )
    {
        return sort_based_join< less_t_str_null_smaller_join< false, false >, less_t_str_null_smaller< false, false >,
                &inner_join< empty_t, less_t_str_null_smaller_join< false, false > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< false, false >(), less_t_str_null_smaller< false, false >(), context, nullptr, left_associated,
                right_associated );
    }

    JoinPair sort_based_inner_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context, const int32_t* left_associated,
            const int32_t* right_associated )
    {
        return sort_based_join< less_t_str_null_smaller_join< true, true >, less_t_str_null_smaller< true, true >,
                &inner_join< empty_t, less_t_str_null_smaller_join< true, true > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< true, true >(), less_t_str_null_smaller< true, true >(), context, nullptr, left_associated,
                right_associated );
    }

    JoinPair sort_based_inner_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        return sort_based_join< less_t_str_null_smaller_join< false, false >, less_t_str_null_smaller< false, false >,
                &inner_join_wrapper< empty_t, less_t_str_null_smaller_join< false, false > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< false, false >(), less_t_str_null_smaller< false, false >(), context, joinDynamicCodeParams );
    }

    JoinPair sort_based_inner_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        return sort_based_join< less_t_str_null_smaller_join< true, true >, less_t_str_null_smaller< true, true >,
                &inner_join_wrapper< empty_t, less_t_str_null_smaller_join< true, true > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< true, true >(), less_t_str_null_smaller< true, true >(), context, joinDynamicCodeParams );
    }
    JoinPair sort_based_left_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_str_null_smaller_join< false, false >, less_t_str_null_smaller< false, false >,
                &left_join_wrapper< empty_t, less_t_str_null_smaller_join< false, false > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< false, false >(), less_t_str_null_smaller< false, false >(), context, joinDynamicCodeParams );
    }

    JoinPair sort_based_left_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_str_null_smaller_join< true, true >, less_t_str_null_smaller< true, true >,
                &left_join_wrapper< empty_t, less_t_str_null_smaller_join< true, true > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< true, true >(), less_t_str_null_smaller< true, true >(), context, joinDynamicCodeParams );
    }

    JoinPair sort_based_right_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_str_null_smaller_join< false, false >, less_t_str_null_smaller< false, false >,
                &right_join_wrapper< empty_t, less_t_str_null_smaller_join< false, false > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< false, false >(), less_t_str_null_smaller< false, false >(), context, joinDynamicCodeParams );
    }

    JoinPair sort_based_right_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_str_null_smaller_join< true, true >, less_t_str_null_smaller< true, true >,
                &right_join_wrapper< empty_t, less_t_str_null_smaller_join< true, true > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< true, true >(), less_t_str_null_smaller< true, true >(), context, joinDynamicCodeParams );
    }

    // for full join
    join_pair_t< int > full_join_helper_dyn_code( const int* lower_data, const int* upper_data, int a_count, int b_count, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        //先做inner join
        mem_t< int > scanned_sizes( a_count );
        managed_mem_t< int64_t > count( 1, context );
        transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                {
                    return upper_data[index] - lower_data[index];
                }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );
        context.synchronize();
        // Allocate an int2 output array and use load-balancing search to compute
        // the join.
        size_t join_count = count.data()[0];
        if( join_count > INT_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "inner join's result too big" );
        }

        mem_t< int > inner_join_left_output( join_count );
        mem_t< int > inner_join_right_output( join_count );
        int* left_data = inner_join_left_output.data();
        int* right_data = inner_join_right_output.data();
        if( join_count > 0 )
        {
            // Use load-balancing search on the segmens. The output is a pair with
            // a_index = seg and b_index = lower_data[seg] + rank.
            auto k = [=] ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
            {
                left_data[index] = vals_a[seg];
                right_data[index] = vals_b[get<0>(lower) + rank];
            };
            transform_lbs( k, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );
            AriesBoolArraySPtr outFlags = make_shared< AriesBoolArray >();
            outFlags->AllocArray( join_count );
            AriesBool* outFlagsData = outFlags->GetData();

            if( joinDynamicCodeParams != nullptr && !joinDynamicCodeParams->functionName.empty() )
            {
                
                shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
                int8_t** constValues = nullptr;

                auto constValueSize = joinDynamicCodeParams->constValues.size();
                if ( constValueSize  > 0 )
                {
                    constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                    for ( int i = 0; i < constValueSize; ++i )
                        ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                    constValueArr->PrefetchToGpu();
                    constValues = constValueArr->GetData();
                }

                AriesDynamicKernelManager::GetInstance().CallKernel( joinDynamicCodeParams->cuModules,
                                                                     joinDynamicCodeParams->functionName.c_str(),
                                                                     joinDynamicCodeParams->input,
                                                                     left_data,
                                                                     right_data,
                                                                     join_count,
                                                                     ( const int8_t** )constValues,
                                                                     joinDynamicCodeParams->comparators,
                                                                     ( int8_t* )outFlagsData );
            }
            else 
            {
                init_value( outFlagsData, join_count, true, context );
            }
            mem_t< int32_t > outPrefixSum( join_count + 1 );
            auto pPrefixSum = outPrefixSum.data();
            transform_scan< int32_t >( [=]ARIES_LAMBDA(int index)
            {
                return outFlagsData[ index ].is_true();
            }, join_count, pPrefixSum, plus_t< int32_t >(), pPrefixSum + join_count, context );
            context.synchronize();

            int32_t matchedCount = *( pPrefixSum + join_count );

            mem_t< int32_t > matchedLeftIndices( matchedCount );
            mem_t< int32_t > matchedRightIndices( matchedCount );

            mem_t< int32_t > leftUnmatchedFlags( a_count );
            mem_t< int32_t > rightUnmatchedFlags( b_count );
            
            auto pMatchedLeftIndices = matchedLeftIndices.data();
            auto pMatchedRightIndices = matchedRightIndices.data();
            
            auto pLeftUnmatchedFlags = leftUnmatchedFlags.data();
            auto pRightUnmatchedFlags = rightUnmatchedFlags.data();
            init_value( pLeftUnmatchedFlags, a_count, 1, context );
            init_value( pRightUnmatchedFlags, b_count, 1, context );

            transform( [=]ARIES_DEVICE( int index )
            {
                if ( outFlagsData[ index ].is_true() )
                {
                    auto offset = pPrefixSum[ index ];
                    auto left_index = left_data[ index ];
                    auto right_index = right_data[ index ];
                    pMatchedLeftIndices[ offset ] = left_index;
                    pMatchedRightIndices[ offset ] = right_index;

                    atomicCAS( pLeftUnmatchedFlags + left_index, 1, 0 );
                    atomicCAS( pRightUnmatchedFlags + right_index, 1, 0 );
                }
            }, join_count, context );
            context.synchronize();

            mem_t< int32_t > prefixSum( std::max( a_count, b_count ) + 1 );
            int32_t* pSum = prefixSum.data();
            scan( pLeftUnmatchedFlags, a_count, pSum, plus_t< int32_t >(), pSum + a_count, context );
            context.synchronize();

            int32_t leftUnmatchedCount = *( pSum + a_count );
            mem_t< int32_t > unmatchedLeftIndicesForLeft( leftUnmatchedCount );
            mem_t< int32_t > unmatchedRightIndicesForLeft( leftUnmatchedCount );

            auto pUnmatchedLeftIndices = unmatchedLeftIndicesForLeft.data();
            auto pUnmatchedRightIndices = unmatchedRightIndicesForLeft.data();

            transform( [=]ARIES_DEVICE( int index )
            {
                if( pLeftUnmatchedFlags[ index ] == 1 )
                {
                    auto offset = pSum[ index ];
                    pUnmatchedLeftIndices[ offset ] = index;
                    pUnmatchedRightIndices[ offset ] = -1;
                }
            }, a_count, context );
            context.synchronize();

            scan( pRightUnmatchedFlags, b_count, pSum, plus_t< int32_t >(), pSum + b_count, context );
            context.synchronize();

            auto rightUnmatchedCount = *( pSum + b_count );
            mem_t< int32_t > unmatchedLeftIndicesForRight( rightUnmatchedCount );
            mem_t< int32_t > unmatchedRightIndicesForRight( rightUnmatchedCount );

            pUnmatchedLeftIndices = unmatchedLeftIndicesForRight.data();
            pUnmatchedRightIndices = unmatchedRightIndicesForRight.data();

            transform( [=]ARIES_DEVICE( int index )
            {
                if ( pRightUnmatchedFlags[ index ] == 1 )
                {
                    auto offset = pSum[ index ];
                    pUnmatchedLeftIndices[ offset ] = -1;
                    pUnmatchedRightIndices[ offset ] = index;
                }
            }, b_count, context );
            context.synchronize();

            auto totalCount = ( int64_t )matchedCount + leftUnmatchedCount + rightUnmatchedCount;
            if( totalCount > INT_MAX )
            {
                printf( "join count=%lu\n", totalCount );
                ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "inner join's result too big" );
            }
            mem_t< int32_t > mergedLeftIndices( totalCount );
            mem_t< int32_t > mergedRightIndices( totalCount );

            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices.data(),
                                pMatchedLeftIndices,
                                sizeof( int32_t ) * matchedCount,
                                cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices.data(),
                                pMatchedRightIndices,
                                sizeof( int32_t ) * matchedCount,
                                cudaMemcpyKind::cudaMemcpyDefault ) );

            auto offset = matchedCount;
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices.data() + offset,
                                unmatchedLeftIndicesForLeft.data(),
                                sizeof( int32_t ) * leftUnmatchedCount,
                                cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices.data() + offset,
                                unmatchedRightIndicesForLeft.data(),
                                sizeof( int32_t ) * leftUnmatchedCount,
                                cudaMemcpyKind::cudaMemcpyDefault ) );

            offset += leftUnmatchedCount;
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedLeftIndices.data() + offset,
                                unmatchedLeftIndicesForRight.data(),
                                sizeof( int32_t ) * rightUnmatchedCount,
                                cudaMemcpyKind::cudaMemcpyDefault ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( mergedRightIndices.data() + offset,
                                unmatchedRightIndicesForRight.data(),
                                sizeof( int32_t ) * rightUnmatchedCount,
                                cudaMemcpyKind::cudaMemcpyDefault ) );

            return {   std::move(mergedLeftIndices), std::move(mergedRightIndices), totalCount};
        }
        return
        {   mem_t<int32_t>(), mem_t<int32_t>(), 0};
    }

    JoinPair sort_based_full_join( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_str_null_smaller_join< false, false >, less_t_str_null_smaller< false, false >,
                &full_join< less_t_str_null_smaller_join< false, false > > >( a, a_count, b, b_count, len,
                less_t_str_null_smaller_join< false, false >(), less_t_str_null_smaller< false, false >(), context, joinDynamicCodeParams );
    }

    JoinPair sort_based_full_join_has_null( char* a, int a_count, char* b, int b_count, int len, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_str_null_smaller_join< true, true >, less_t_str_null_smaller< true, true >,
                &full_join< less_t_str_null_smaller_join< true, true > > >( a, a_count, b, b_count, len, less_t_str_null_smaller_join< true, true >(),
                less_t_str_null_smaller< true, true >(), context, joinDynamicCodeParams );
    }

    JoinPair sort_based_inner_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const int32_t* left_associated, const int32_t* right_associated )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< false >, less_t_CompactDecimal_null_smaller< false >,
                &inner_join< empty_t, less_t_CompactDecimal_null_smaller_join< false > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< false >( prec, sca ), less_t_CompactDecimal_null_smaller< false >( prec, sca ), context,
                nullptr, left_associated, right_associated );
    }

    JoinPair sort_based_inner_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const int32_t* left_associated, const int32_t* right_associated )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< true >, less_t_CompactDecimal_null_smaller< true >,
                &inner_join< empty_t, less_t_CompactDecimal_null_smaller_join< true > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< true >( prec, sca ), less_t_CompactDecimal_null_smaller< true >( prec, sca ), context,
                nullptr, left_associated, right_associated );
    }

    JoinPair sort_based_inner_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< false >, less_t_CompactDecimal_null_smaller< false >,
                &inner_join_wrapper< empty_t, less_t_CompactDecimal_null_smaller_join< false > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< false >( prec, sca ), less_t_CompactDecimal_null_smaller< false >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_inner_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< true >, less_t_CompactDecimal_null_smaller< true >,
                &inner_join_wrapper< empty_t, less_t_CompactDecimal_null_smaller_join< true > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< true >( prec, sca ), less_t_CompactDecimal_null_smaller< true >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_left_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< false >, less_t_CompactDecimal_null_smaller< false >,
                &left_join< empty_t, less_t_CompactDecimal_null_smaller_join< false > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< false >( prec, sca ), less_t_CompactDecimal_null_smaller< false >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_left_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< true >, less_t_CompactDecimal_null_smaller< true >,
                &left_join< empty_t, less_t_CompactDecimal_null_smaller_join< true > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< true >( prec, sca ), less_t_CompactDecimal_null_smaller< true >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_right_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< false >, less_t_CompactDecimal_null_smaller< false >,
                &right_join_wrapper< empty_t, less_t_CompactDecimal_null_smaller_join< false > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< false >( prec, sca ), less_t_CompactDecimal_null_smaller< false >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_right_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< true >, less_t_CompactDecimal_null_smaller< true >,
                &right_join_wrapper< empty_t, less_t_CompactDecimal_null_smaller_join< true > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< true >( prec, sca ), less_t_CompactDecimal_null_smaller< true >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_full_join_compact_decimal( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< false >, less_t_CompactDecimal_null_smaller< false >,
                &full_join< less_t_CompactDecimal_null_smaller_join< false > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< false >( prec, sca ), less_t_CompactDecimal_null_smaller< false >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    JoinPair sort_based_full_join_compact_decimal_has_null( char* a, int a_count, char* b, int b_count, int len, uint16_t prec, uint16_t sca,
            context_t& context, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        return sort_based_join< less_t_CompactDecimal_null_smaller_join< true >, less_t_CompactDecimal_null_smaller< true >,
                &full_join< less_t_CompactDecimal_null_smaller_join< true > > >( a, a_count, b, b_count, len,
                less_t_CompactDecimal_null_smaller_join< true >( prec, sca ), less_t_CompactDecimal_null_smaller< true >( prec, sca ), context,
                joinDynamicCodeParams );
    }

    AriesBoolArraySPtr sort_based_semi_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated, const int32_t* right_associated )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        return sort_based_semi_join( leftColumn->GetData(), leftColumn->GetItemCount(), rightColumn->GetData(), rightColumn->GetItemCount(),
                leftColumn->GetDataType(), rightColumn->GetDataType(), joinDynamicCodeParams, context, left_associated, right_associated );
    }

    AriesBoolArraySPtr sort_based_anti_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn,
            const JoinDynamicCodeParams* joinDynamicCodeParams, context_t& context, const int32_t* left_associated, const int32_t* right_associated, bool isNotIn )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        return sort_based_anti_join( leftColumn->GetData(), leftColumn->GetItemCount(), rightColumn->GetData(), rightColumn->GetItemCount(),
                leftColumn->GetDataType(), rightColumn->GetDataType(), joinDynamicCodeParams, context, left_associated, right_associated, isNotIn );
    }

    JoinPair sort_based_inner_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const AriesInt32ArraySPtr leftAssociated, const AriesInt32ArraySPtr rightAssociated )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        JoinPair result;
        int8_t* leftData = leftColumn->GetData();
        size_t leftTupleNum = leftColumn->GetItemCount();
        int8_t* rightData = rightColumn->GetData();
        size_t rightTupleNum = rightColumn->GetItemCount();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();
        int32_t* left_associated = leftAssociated ? leftAssociated->GetData() : nullptr;
        int32_t* right_associated = rightAssociated ? rightAssociated->GetData() : nullptr;

        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_inner_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            context, left_associated, right_associated );
                else
                    result = sort_based_inner_join_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), context, left_associated, right_associated );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_inner_join_compact_decimal( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, left_associated,
                            right_associated );
                else
                    result = sort_based_inner_join_compact_decimal_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, left_associated,
                            right_associated );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum, context,
            //                     left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum, context,
            //                     left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum, context,
            //                     left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum, context,
            //                     left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     context, left_associated, right_associated );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, context, left_associated, right_associated );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, left_associated, right_associated );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, left_associated, right_associated );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum, context,
                                left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, context, left_associated, right_associated );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum, context,
                                left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, context, left_associated, right_associated );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum, context,
                                left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, context, left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, context, left_associated, right_associated );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, context, left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, context, left_associated, right_associated );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum, context,
                                left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                context, left_associated, right_associated );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, context, left_associated, right_associated );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + GetValueTypeAsString( leftType ) + " for inner-jion" );
        }
        return result;
    }

    JoinPair sort_based_inner_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        JoinPair result;
        int8_t* leftData = leftColumn->GetData();
        size_t leftTupleNum = leftColumn->GetItemCount();
        int8_t* rightData = rightColumn->GetData();
        size_t rightTupleNum = rightColumn->GetItemCount();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();

        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_inner_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            context, joinDynamicCodeParams );
                else
                    result = sort_based_inner_join_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_inner_join_compact_decimal( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                else
                    result = sort_based_inner_join_compact_decimal_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_inner_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_inner_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_inner_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_inner_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_inner_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + GetValueTypeAsString( leftType ) + " for inner-jion" );
        }
        return result;
    }
    JoinPair sort_based_left_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        JoinPair result;
        int8_t* leftData = leftColumn->GetData();
        size_t leftTupleNum = leftColumn->GetItemCount();
        int8_t* rightData = rightColumn->GetData();
        size_t rightTupleNum = rightColumn->GetItemCount();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();

        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_left_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            context, joinDynamicCodeParams );
                else
                    result = sort_based_left_join_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_left_join_compact_decimal( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                else
                    result = sort_based_left_join_compact_decimal_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_left_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_left_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_left_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_left_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_left_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + GetValueTypeAsString( leftType ) + " for inner-jion" );
        }
        return result;
    }

    JoinPair sort_based_right_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        JoinPair result;
        int8_t* leftData = leftColumn->GetData();
        size_t leftTupleNum = leftColumn->GetItemCount();
        int8_t* rightData = rightColumn->GetData();
        size_t rightTupleNum = rightColumn->GetItemCount();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();

        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_right_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            context, joinDynamicCodeParams );
                else
                    result = sort_based_right_join_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_right_join_compact_decimal( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                else
                    result = sort_based_right_join_compact_decimal_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_right_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_right_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_right_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_right_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_right_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + GetValueTypeAsString( leftType ) + " for inner-jion" );
        }
        return result;
    }

    JoinPair sort_based_full_join( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        ARIES_ASSERT( leftColumn && rightColumn,
                "leftColumn is nullptr: " + to_string( !!leftColumn ) + ", rightColumn is nullptr: " + to_string( !!rightColumn )
                        + ", leftColumn->GetDataType(): " + ( leftColumn ? GetValueTypeAsString( leftColumn->GetDataType() ) : "" )
                        + ", rightColumn->GetDataType(): " + ( rightColumn ? GetValueTypeAsString( rightColumn->GetDataType() ) : "" ) );
        JoinPair result;
        int8_t* leftData = leftColumn->GetData();
        size_t leftTupleNum = leftColumn->GetItemCount();
        int8_t* rightData = rightColumn->GetData();
        size_t rightTupleNum = rightColumn->GetItemCount();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();

        switch( leftType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
            {
                if( !leftType.HasNull )
                    result = sort_based_full_join( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum, leftType.GetDataTypeSize(),
                            context, joinDynamicCodeParams );
                else
                    result = sort_based_full_join_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {
                if( !leftType.HasNull )
                    result = sort_based_full_join_compact_decimal( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                else
                    result = sort_based_full_join_compact_decimal_has_null( ( char* )leftData, leftTupleNum, ( char* )rightData, rightTupleNum,
                            leftType.GetDataTypeSize(), leftType.DataType.Precision, leftType.DataType.Scale, context, joinDynamicCodeParams );
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int8_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int8_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT16:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int16_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int16_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT32:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int32_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int32_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::INT64:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< float >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( int64_t* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( float* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< int64_t >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            // case AriesValueType::UINT8:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( uint8_t* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( uint8_t* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( uint8_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( nullable_type< uint8_t >* )leftData, leftTupleNum, ( nullable_type< uint8_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT16:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( uint16_t* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( uint16_t* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( uint16_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( nullable_type< uint16_t >* )leftData, leftTupleNum, ( nullable_type< uint16_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT32:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( uint32_t* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( uint32_t* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( uint32_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( nullable_type< uint32_t >* )leftData, leftTupleNum, ( nullable_type< uint32_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            // case AriesValueType::UINT64:
            // {
            //     if( !leftType.HasNull )
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( uint64_t* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum, context,
            //                     joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( uint64_t* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //     }
            //     else
            //     {
            //         if( !rightType.HasNull )
            //             result = sort_based_full_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( uint64_t* )rightData, rightTupleNum,
            //                     context, joinDynamicCodeParams );
            //         else
            //             result = sort_based_full_join( ( nullable_type< uint64_t >* )leftData, leftTupleNum, ( nullable_type< uint64_t >* )rightData,
            //                     rightTupleNum, context, joinDynamicCodeParams );
            //     }
            //     break;
            // }
            case AriesValueType::FLOAT:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( float* )leftData, leftTupleNum, ( nullable_type< double >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< float >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !leftType.HasNull )
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( int8_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( nullable_type< int8_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( int16_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( nullable_type< int16_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( int32_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( nullable_type< int32_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( int64_t* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( nullable_type< int64_t >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( nullable_type< float >* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( double* )rightData, rightTupleNum, context,
                                        joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( double* )leftData, leftTupleNum, ( nullable_type< double >* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                else
                {
                    switch( rightType.DataType.ValueType )
                    {
                        case AriesValueType::BOOL:
                        case AriesValueType::INT8:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int8_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int8_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT16:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int16_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int16_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT32:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int32_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int32_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::INT64:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum, ( int64_t* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< int64_t >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::FLOAT:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum, ( float* )rightData, rightTupleNum,
                                        context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< float >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                        case AriesValueType::DOUBLE:
                        {
                            if( !rightType.HasNull )
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum, ( double* )rightData,
                                        rightTupleNum, context, joinDynamicCodeParams );
                            else
                                result = sort_based_full_join( ( nullable_type< double >* )leftData, leftTupleNum,
                                        ( nullable_type< double >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                            break;
                        }
                    }
                }
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( Decimal* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( Decimal* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( Decimal* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( nullable_type< Decimal >* )leftData, leftTupleNum, ( nullable_type< Decimal >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATE:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( AriesDate* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( AriesDate* )leftData, leftTupleNum, ( nullable_type< AriesDate >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum, ( AriesDate* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( nullable_type< AriesDate >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDate >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( AriesDatetime* )leftData, leftTupleNum, ( AriesDatetime* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( AriesDatetime* )leftData, leftTupleNum, ( nullable_type< AriesDatetime >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum, ( AriesDatetime* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( nullable_type< AriesDatetime >* )leftData, leftTupleNum,
                                ( nullable_type< AriesDatetime >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( AriesTimestamp* )leftData, leftTupleNum, ( AriesTimestamp* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( AriesTimestamp* )leftData, leftTupleNum, ( nullable_type< AriesTimestamp >* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum, ( AriesTimestamp* )rightData,
                                rightTupleNum, context, joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( nullable_type< AriesTimestamp >* )leftData, leftTupleNum,
                                ( nullable_type< AriesTimestamp >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            case AriesValueType::YEAR:
            {
                if( !leftType.HasNull )
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( AriesYear* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum, context,
                                joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( AriesYear* )leftData, leftTupleNum, ( nullable_type< AriesYear >* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                }
                else
                {
                    if( !rightType.HasNull )
                        result = sort_based_full_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum, ( AriesYear* )rightData, rightTupleNum,
                                context, joinDynamicCodeParams );
                    else
                        result = sort_based_full_join( ( nullable_type< AriesYear >* )leftData, leftTupleNum,
                                ( nullable_type< AriesYear >* )rightData, rightTupleNum, context, joinDynamicCodeParams );
                }
                break;
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "sorting data type " + GetValueTypeAsString( leftType ) + " for inner-jion" );
        }
        return result;
    }

    void division_data_int64( const int8_t* data, AriesColumnType type, const int64_t* divisor, int8_t* output, size_t tupleNum, context_t& context )
    {
        switch( type.DataType.ValueType )
        {
            case AriesValueType::INT32:
            {
                if( !type.HasNull )
                    do_div( ( int32_t* )data, divisor, ( Decimal* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< int32_t >* )data, divisor, ( nullable_type< Decimal >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !type.HasNull )
                    do_div( ( int64_t* )data, divisor, ( Decimal* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< int64_t >* )data, divisor, ( nullable_type< Decimal >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !type.HasNull )
                    do_div( ( uint32_t* )data, divisor, ( Decimal* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< uint32_t >* )data, divisor, ( nullable_type< Decimal >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !type.HasNull )
                    do_div( ( uint64_t* )data, divisor, ( Decimal* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< uint64_t >* )data, divisor, ( nullable_type< Decimal >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !type.HasNull )
                    do_div( ( float* )data, divisor, ( float* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< float >* )data, divisor, ( nullable_type< float >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !type.HasNull )
                    do_div( ( double* )data, divisor, ( double* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< double >* )data, divisor, ( nullable_type< double >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !type.HasNull )
                    do_div( ( Decimal* )data, divisor, ( Decimal* )output, tupleNum, context );
                else
                    do_div( ( nullable_type< Decimal >* )data, divisor, ( nullable_type< Decimal >* )output, tupleNum, context );
                break;
            }
            case AriesValueType::DATE:
            case AriesValueType::DATETIME:
            case AriesValueType::TIMESTAMP:
            case AriesValueType::TIME:
            case AriesValueType::YEAR:
            {
                ARIES_ASSERT( 0, "division type is DATE, TIMESTAMP, DATETIME, TIME or YEAR" );
            }
            default:
                //FIXME need support all data types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "division data type " + GetValueTypeAsString( type ) );
        }
    }

    JoinPair create_cartesian_product( size_t left_count, size_t right_count, context_t& context )
    {
        size_t total = left_count * right_count;
        if( total > INT_MAX )
        {
            printf( "join count=%lu\n", total );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "carstsion product join's result too big" );
        }

        JoinPair pair;
        pair.JoinCount = total;
        if( total > 0 )
        {
            pair.LeftIndices = std::make_shared< AriesInt32Array >( total );
            pair.RightIndices = std::make_shared< AriesInt32Array >( total );

            int32_t *left = pair.LeftIndices->GetData();
            int32_t *right = pair.RightIndices->GetData();
            auto k = [=] ARIES_DEVICE(int index)
            {
                left[ index ] = index / right_count;
                right[ index ] = index % right_count;
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, total, context );
            context.synchronize();
        }

        return pair;
    }

    AriesBoolArraySPtr is_null( int8_t* data, size_t tupleNum, AriesColumnType type, context_t& context )
    {
        AriesBoolArraySPtr result;
        if( tupleNum > 0 )
        {
            result = std::make_shared< AriesBoolArray >( tupleNum );
            size_t dataTypeSize = type.GetDataTypeSize();
            AriesBool* output = result->GetData();
            if( type.HasNull )
            {
                auto k = [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = !( *( data + index * dataTypeSize ) );
                };

                transform< launch_box_t< arch_52_cta< 256, 16 > > >( k, tupleNum, context );
                context.synchronize();
            }
            else
            {
                init_value( output, tupleNum, AriesBool( AriesBool::ValueType::False ), context );
            }
        }
        return result;
    }

    AriesBoolArraySPtr is_not_null( int8_t* data, size_t tupleNum, AriesColumnType type, context_t& context )
    {
        AriesBoolArraySPtr result;
        if( tupleNum > 0 )
        {
            result = std::make_shared< AriesBoolArray >( tupleNum );
            size_t dataTypeSize = type.GetDataTypeSize();
            AriesBool* output = result->GetData();
            if( type.HasNull )
            {
                auto k = [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = *( data + index * dataTypeSize );
                };

                transform< launch_box_t< arch_52_cta< 256, 16 > > >( k, tupleNum, context );
                context.synchronize();
            }
            else
            {
                init_value( output, tupleNum, AriesBool( AriesBool::ValueType::True ), context );
            }
        }
        return result;
    }

    AriesDataBufferSPtr datetime_to_date( const AriesDataBufferSPtr& column, context_t& context )
    {
        AriesColumnType type = column->GetDataType();
        ARIES_ASSERT( type.DataType.ValueType == AriesValueType::DATETIME, "type.DataType.ValueType: " + GetValueTypeAsString( type ) );
        AriesDataBufferSPtr result;
        size_t tupleNum = column->GetItemCount();
        if( tupleNum > 0 )
        {
            result = std::make_shared< AriesDataBuffer >( AriesColumnType(
            { AriesValueType::DATE, 1 }, type.HasNull, type.IsUnique ), tupleNum );
            result->PrefetchToGpu();
            if( type.HasNull )
            {
                nullable_type< AriesDate >* output = ( nullable_type< AriesDate >* )result->GetData();
                nullable_type< AriesDatetime >* input = ( nullable_type< AriesDatetime >* )column->GetData();
//                auto k = [=] ARIES_DEVICE(int index)
//                {
//                    output[ index ].flag = input[ index ].flag;
//                    output[ index ].value = aries_acc::DATE( input[ index ].value );
//                };
//
//                transform< launch_box_t< arch_52_cta< 256, 16 > > >( k, tupleNum, context );
//                context.synchronize();
                sql_date( input, tupleNum, output, context );
            }
            else
            {
                AriesDate * output = ( AriesDate * )result->GetData();
                AriesDatetime * input = ( AriesDatetime * )column->GetData();
//                auto k = [=] ARIES_DEVICE(int index)
//                {
//                    output[ index ] = aries_acc::DATE( input[ index ] );
//                };
//
//                transform< launch_box_t< arch_52_cta< 256, 16 > > >( k, tupleNum, context );
//                context.synchronize();
                sql_date( input, tupleNum, output, context );
            }
        }
        return result;
    }

    AriesDataBufferSPtr sql_abs_func( const AriesDataBufferSPtr& column, context_t& context )
    {
        AriesColumnType type = column->GetDataType();
        size_t tupleNum = column->GetItemCount();
        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            const CompactDecimal *data = ( CompactDecimal * )column->GetData();
            AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( AriesColumnType(
            { AriesValueType::DECIMAL, 1 }, type.HasNull ), tupleNum );
            if( tupleNum > 0 )
            {
                result->PrefetchToGpu();
                if( !type.HasNull )
                {
                    sql_abs_compact_decimal( data, type.GetDataTypeSize(), type.DataType.Precision, type.DataType.Scale, tupleNum,
                            ( Decimal * )result->GetData(), context );
                }
                else
                {
                    sql_abs_compact_decimal( data, type.GetDataTypeSize(), type.DataType.Precision, type.DataType.Scale, tupleNum,
                            ( nullable_type< Decimal > * )result->GetData(), context );
                }
            }
            return result;
        }
        else
        {
            AriesDataBufferSPtr result = column->CloneWithNoContent();
            if( tupleNum > 0 )
            {
                int8_t* leftData = column->GetData();
                int8_t* rightData = result->GetData();
                switch( type.DataType.ValueType )
                {
                    case AriesValueType::INT8:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( int8_t* )leftData, tupleNum, ( int8_t* )rightData, context );
                        else
                            sql_abs( ( nullable_type< int8_t >* )leftData, tupleNum, ( nullable_type< int8_t >* )rightData, context );
                        break;
                    }
                    case AriesValueType::INT16:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( int16_t* )leftData, tupleNum, ( int16_t* )rightData, context );
                        else
                            sql_abs( ( nullable_type< int16_t >* )leftData, tupleNum, ( nullable_type< int16_t >* )rightData, context );
                        break;
                    }
                    case AriesValueType::INT32:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( int32_t* )leftData, tupleNum, ( int32_t* )rightData, context );
                        else
                            sql_abs( ( nullable_type< int32_t >* )leftData, tupleNum, ( nullable_type< int32_t >* )rightData, context );
                        break;
                    }
                    case AriesValueType::INT64:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( int64_t* )leftData, tupleNum, ( int64_t* )rightData, context );
                        else
                            sql_abs( ( nullable_type< int64_t >* )leftData, tupleNum, ( nullable_type< int64_t >* )rightData, context );
                        break;
                    }
                    case AriesValueType::UINT8:
                    case AriesValueType::UINT16:
                    case AriesValueType::UINT32:
                    case AriesValueType::UINT64:
                    {
                        result = column;
                        break;
                    }
                    case AriesValueType::FLOAT:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( float* )leftData, tupleNum, ( float* )rightData, context );
                        else
                            sql_abs( ( nullable_type< float >* )leftData, tupleNum, ( nullable_type< float >* )rightData, context );
                        break;
                    }
                    case AriesValueType::DOUBLE:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( double* )leftData, tupleNum, ( double* )rightData, context );
                        else
                            sql_abs( ( nullable_type< double >* )leftData, tupleNum, ( nullable_type< double >* )rightData, context );
                        break;
                    }
                    case AriesValueType::DECIMAL:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( Decimal* )leftData, tupleNum, ( Decimal* )rightData, context );
                        else
                            sql_abs( ( nullable_type< Decimal >* )leftData, tupleNum, ( nullable_type< Decimal >* )rightData, context );
                        break;
                    }
                    case AriesValueType::TIME:
                    {
                        result->PrefetchToGpu();
                        if( !type.HasNull )
                            sql_abs( ( AriesTime* )leftData, tupleNum, ( AriesTime* )rightData, context );
                        else
                            sql_abs( ( nullable_type< AriesTime >* )leftData, tupleNum, ( nullable_type< AriesTime >* )rightData, context );
                        break;
                    }
                    default:
                        assert( 0 );
                        ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "data type " + GetValueTypeAsString( type ) + " for abs function" );
                }
            }
            return result;
        }
    }

    AriesDataBufferSPtr month( const AriesDataBufferSPtr& column, context_t& context )
    {
        AriesColumnType type = column->GetDataType();
        AriesDataBufferSPtr result;
        size_t tupleNum = column->GetItemCount();
        if( tupleNum > 0 )
        {
            result = std::make_shared< AriesDataBuffer >( AriesColumnType(
            { AriesValueType::UINT8, 1 }, type.HasNull, type.IsUnique ), tupleNum );
            result->PrefetchToGpu();
            switch( type.DataType.ValueType )
            {
                case AriesValueType::DATE:
                    if( type.HasNull )
                    {
                        nullable_type< uint8_t >* output = ( nullable_type< uint8_t >* )result->GetData();
                        nullable_type< AriesDate >* input = ( nullable_type< AriesDate >* )column->GetData();
                        sql_month( input, tupleNum, output, context );
                    }
                    else
                    {
                        uint8_t * output = ( uint8_t * )result->GetData();
                        AriesDate * input = ( AriesDate * )column->GetData();
                        sql_month( input, tupleNum, output, context );
                    }
                    break;
                case AriesValueType::DATETIME:
                    if( type.HasNull )
                    {
                        nullable_type< uint8_t >* output = ( nullable_type< uint8_t >* )result->GetData();
                        nullable_type< AriesDatetime >* input = ( nullable_type< AriesDatetime >* )column->GetData();
                        sql_month( input, tupleNum, output, context );
                    }
                    else
                    {
                        uint8_t * output = ( uint8_t * )result->GetData();
                        AriesDatetime * input = ( AriesDatetime * )column->GetData();
                        sql_month( input, tupleNum, output, context );
                    }
                    break;
                case AriesValueType::TIMESTAMP:
                    if( type.HasNull )
                    {
                        nullable_type< uint8_t >* output = ( nullable_type< uint8_t >* )result->GetData();
                        nullable_type< AriesTimestamp >* input = ( nullable_type< AriesTimestamp >* )column->GetData();
                        sql_month( input, tupleNum, output, context );
                    }
                    else
                    {
                        uint8_t * output = ( uint8_t * )result->GetData();
                        AriesTimestamp * input = ( AriesTimestamp * )column->GetData();
                        sql_month( input, tupleNum, output, context );
                    }
                    break;
                default:
                    ARIES_ASSERT( 0, "NOT support DataType: " + GetValueTypeAsString( type ) );
            }
        }
        return result;
    }

    AriesDataBufferSPtr date_format( const AriesDataBufferSPtr& column, const AriesDataBufferSPtr& format, const LOCALE_LANGUAGE &locale,
            context_t& context )
    {
        AriesColumnType type = column->GetDataType();
        AriesDataBufferSPtr result;
        size_t tupleNum = column->GetItemCount();
        if( tupleNum > 0 )
        {
            const char *pfstr = ( char * )format->GetData();
            int32_t targetLen = get_format_length( pfstr, locale );
            ARIES_ASSERT( targetLen > 1, "format target length too short: " + std::to_string( targetLen ) );
            result = std::make_shared< AriesDataBuffer >( AriesColumnType(
            { AriesValueType::CHAR, targetLen }, type.HasNull, type.IsUnique ), tupleNum );
            size_t itemLen = result->GetDataType().GetDataTypeSize();
            result->PrefetchToGpu();
            switch( type.DataType.ValueType )
            {
                case AriesValueType::DATETIME:
                    if( type.HasNull )
                    {
                        char *output = ( char * )result->GetData();
                        nullable_type< AriesDatetime >* input = ( nullable_type< AriesDatetime >* )column->GetData();
                        sql_dateformat( input, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    else
                    {
                        char *output = ( char * )result->GetData();
                        AriesDatetime * input = ( AriesDatetime * )column->GetData();
                        sql_dateformat( input, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    break;
                case AriesValueType::DATE:
                    if( type.HasNull )
                    {
                        char *output = ( char * )result->GetData();
                        nullable_type< AriesDate >* input = ( nullable_type< AriesDate >* )column->GetData();
                        sql_dateformat( input, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    else
                    {
                        char *output = ( char * )result->GetData();
                        AriesDate * input = ( AriesDate * )column->GetData();
                        sql_dateformat( input, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    break;
                case AriesValueType::TIMESTAMP:
                    if( type.HasNull )
                    {
                        char *output = ( char * )result->GetData();
                        nullable_type< AriesTimestamp >* input = ( nullable_type< AriesTimestamp >* )column->GetData();
                        sql_dateformat( input, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    else
                    {
                        char *output = ( char * )result->GetData();
                        AriesTimestamp * input = ( AriesTimestamp * )column->GetData();
                        sql_dateformat( input, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    break;
                case AriesValueType::TIME:
                {
                    auto currentDate = AriesDatetimeTrans::Now();
                    //use date of today only
                    currentDate.hour = 0;
                    currentDate.minute = 0;
                    currentDate.second = 0;
                    currentDate.second_part = 0;
                    if( type.HasNull )
                    {
                        char *output = ( char * )result->GetData();
                        nullable_type< AriesTime >* input = ( nullable_type< AriesTime >* )column->GetData();
                        sql_dateformat( input, currentDate, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                    else
                    {
                        char *output = ( char * )result->GetData();
                        AriesTime * input = ( AriesTime * )column->GetData();
                        sql_dateformat( input, currentDate, tupleNum, pfstr, locale, output, itemLen, context );
                    }
                }
                    break;
                default:
                    ARIES_ASSERT( 0, "NOT support DataType: " + GetValueTypeAsString( type ) );
            }
        }
        return result;
    }

    static join_pair_t< int > left_join_by_dynamic_code( const int* lower_data,
                                                         const int* upper_data,
                                                         int a_count,
                                                         context_t& context,
                                                         const int* vals_a,
                                                         const int* vals_b,
                                                         const char* function_name,
                                                         const std::vector< CUmoduleSPtr >& modules,
                                                         std::vector< AriesDynamicCodeComparator > comparators,
                                                         const AriesColumnDataIterator *input,
                                                         const int8_t** constValues )
    {
        mem_t< int > scanned_sizes( a_count );
        mem_t< char > match_flag( a_count );
        char* flag_data = match_flag.data();

        managed_mem_t< int64_t > count( 1, context );
        transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                {
                    int ret = upper_data[index] - lower_data[index];
                    flag_data[ index ] = ret > 0;
                    return ret > 0 ? ret : 1;
                }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );
        context.synchronize();
        // Allocate an int2 output array and use load-balancing search to compute
        // the join.
        size_t join_count = count.data()[0];
        if( join_count > INT_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "left join's result too big" );
        }
        mem_t< int > left_output( join_count );
        mem_t< int > right_output( join_count );
        int* left_data = left_output.data();
        int* right_data = right_output.data();
        if( join_count > 0 )
        {
            // Use load-balancing search on the segmens. The output is a pair with
            // a_index = seg and b_index = lower_data[seg] + rank.
            auto k = [=]ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
            {
                left_data[ index ] = vals_a[ seg ];
                right_data[ index ] = flag_data[ seg ] ? vals_b[ get<0>(lower) + rank ] : -1;
            };
            transform_lbs< empty_t >( k, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );

            if( function_name )
            {
                auto associated = make_shared< AriesBoolArray >();
                associated->AllocArray( join_count );
                auto pAssociated = associated->GetData();

                AriesDynamicKernelManager::GetInstance().CallKernel( modules,
                                                                     function_name,
                                                                     input,
                                                                     left_data,
                                                                     right_data,
                                                                     join_count,
                                                                     constValues,
                                                                     comparators,
                                                                     ( int8_t* )pAssociated );

                mem_t< int > lUnMatchedIndexFlag( a_count );
                auto pUnMatchedFlag = lUnMatchedIndexFlag.data();
                init_value( pUnMatchedFlag, a_count, 1, context );

                mem_t< int > matchedIndexFlag( join_count );
                auto pMatchedFlag = matchedIndexFlag.data();
                cudaMemset( pMatchedFlag, 0, join_count * sizeof(int) );

                transform( [=]ARIES_DEVICE(int index)
                        {
                            if( pAssociated[ index ].is_true() )
                            {
                                pUnMatchedFlag[ left_data[ index] ] = 0;
                                pMatchedFlag[ index ] = 1;
                            }
                        }, join_count, context );

                mem_t< int > psumFlag( a_count );
                managed_mem_t< int > psumFlagTotal( 1, context );
                auto sum = psumFlag.data();
                scan( pUnMatchedFlag, a_count, sum, plus_t< int32_t >(), psumFlagTotal.data(), context );
                context.synchronize();
                size_t unmatched_total = psumFlagTotal.data()[0];

                mem_t< int > matchedSumFlag( join_count );
                managed_mem_t< int > matchedSumFlagTotal( 1, context );
                auto pMatchedSum = matchedSumFlag.data();
                scan( pMatchedFlag, join_count, pMatchedSum, plus_t< int32_t >(), matchedSumFlagTotal.data(), context );
                context.synchronize();
                size_t matched_total = matchedSumFlagTotal.data()[0];

                mem_t< int > left_indices( matched_total );
                mem_t< int > right_indices( matched_total );
                int* pLeftIndices = left_indices.data();
                int* pRightIndices = right_indices.data();

                mem_t< int > left_indices2( unmatched_total );
                mem_t< int > right_indices2( unmatched_total );
                int* pLeftIndices2 = left_indices2.data();
                int* pRightIndices2 = right_indices2.data();

                transform( [=]ARIES_DEVICE( int index )
                        {
                            auto leftIndex = left_data[ index ];
                            if( pUnMatchedFlag[ leftIndex ] )
                            {
                                int pos = sum[ leftIndex ];
                                pLeftIndices2[ pos ] = leftIndex;
                                pRightIndices2[ pos ] = -1;
                            }
                            else if ( pMatchedFlag[ index ] )
                            {
                                int pos = pMatchedSum[ index ];
                                pLeftIndices[ pos ] = leftIndex;
                                pRightIndices[ pos ] = right_data[ index ];
                            }
                        }, join_count, context );

                mem_t< int > leftData( matched_total + unmatched_total );
                mem_t< int > rightData( matched_total + unmatched_total );
                cudaMemcpy( leftData.data(), pLeftIndices, matched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( leftData.data() + matched_total, pLeftIndices2, unmatched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data(), pRightIndices, matched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data() + matched_total, pRightIndices2, unmatched_total * sizeof(int), cudaMemcpyDefault );
                return
                {   std::move( leftData ), std::move( rightData ), matched_total + unmatched_total};
            }
            else
            {
                return
                {   std::move( left_output ), std::move( right_output ), join_count};
            }

        }
        return
        {   std::move(left_output), std::move(right_output), join_count};
    }

    template< template< typename, typename > class comp_t, typename launch_arg_t, typename a_it, typename b_it >
    join_pair_t< int > left_join_wrapper( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a, const int* vals_b,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );

        const char* function_name = nullptr;
        const AriesColumnDataIterator *input = nullptr;
        std::vector< std::shared_ptr< CUmodule > > modules;
        std::vector< AriesDynamicCodeComparator > comparators;
        shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
        int8_t** constValues = nullptr;

        if( joinDynamicCodeParams )
        {
            function_name = joinDynamicCodeParams->functionName.empty() ? nullptr : joinDynamicCodeParams->functionName.c_str();
            input = joinDynamicCodeParams->input;
            modules = joinDynamicCodeParams->cuModules;
            comparators.assign( joinDynamicCodeParams->comparators.cbegin(), joinDynamicCodeParams->comparators.cend() );

            auto constValueSize = joinDynamicCodeParams->constValues.size();
            if ( constValueSize  > 0 )
            {
                constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                for ( int i = 0; i < constValueSize; ++i )
                    ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                constValueArr->PrefetchToGpu();
                constValues = constValueArr->GetData();
            }
        }

        return left_join_by_dynamic_code( lower_upper.first.data(),
                                          lower_upper.second.data(),
                                          a_count,
                                          context,
                                          vals_a,
                                          vals_b,
                                          function_name,
                                          modules,
                                          comparators,
                                          input,
                                          ( const int8_t** )constValues );
    }

    static join_pair_t< int > inner_join_by_dynamic_code( const int* lower_data,
                                                          const int* upper_data,
                                                          int a_count,
                                                          context_t& context,
                                                          const int* vals_a, const int* vals_b,
                                                          const char* function_name,
                                                          const std::vector< CUmoduleSPtr >& modules,
                                                          std::vector< AriesDynamicCodeComparator > comparators,
                                                          const AriesColumnDataIterator *input,
                                                          const int8_t** constValues )
    {

        mem_t< int > scanned_sizes( a_count );
        managed_mem_t< int64_t > count( 1, context );
        transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                {
                    return upper_data[index] - lower_data[index];
                }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );
        context.synchronize();
        // Allocate an int2 output array and use load-balancing search to compute
        // the join.
        size_t join_count = count.data()[0];
        if( join_count > INT_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "inner join's result too big" );
        }

        mem_t< int > left_output( join_count );
        mem_t< int > right_output( join_count );
        int* left_data = left_output.data();
        int* right_data = right_output.data();

        if( join_count > 0 )
        {
            // Use load-balancing search on the segmens. The output is a pair with
            // a_index = seg and b_index = lower_data[seg] + rank.
            auto k = [=] ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
            {
                left_data[index] = vals_a[seg];
                right_data[index] = vals_b[get<0>(lower) + rank];
            };
            transform_lbs< empty_t >( k, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );

            if( function_name )
            {
                auto associated = make_shared< AriesBoolArray >();
                associated->AllocArray( join_count );
                auto pAssociated = associated->GetData();

                AriesDynamicKernelManager::GetInstance().CallKernel( modules,
                                                                     function_name,
                                                                     input,
                                                                     left_data,
                                                                     right_data,
                                                                     join_count,
                                                                     constValues,
                                                                     comparators,
                                                                     ( int8_t* )pAssociated );

                mem_t< int > matchedIndexFlag( join_count );
                auto pMatchedFlag = matchedIndexFlag.data();
                cudaMemset( pMatchedFlag, 0, join_count * sizeof(int) );

                transform( [=]ARIES_DEVICE(int index)
                        {
                            if( pAssociated[ index ].is_true() )
                            {
                                pMatchedFlag[ index ] = 1;
                            }
                        }, join_count, context );

                mem_t< int > matchedSumFlag( join_count );
                managed_mem_t< int > matchedSumFlagTotal( 1, context );
                auto pMatchedSum = matchedSumFlag.data();
                scan( pMatchedFlag, join_count, pMatchedSum, plus_t< int32_t >(), matchedSumFlagTotal.data(), context );
                context.synchronize();
                size_t matched_total = matchedSumFlagTotal.data()[0];

                mem_t< int > left_indices( matched_total );
                mem_t< int > right_indices( matched_total );
                int* pLeftIndices = left_indices.data();
                int* pRightIndices = right_indices.data();

                transform( [=]ARIES_DEVICE( int index )
                        {
                            auto leftIndex = left_data[ index ];
                            if ( pMatchedFlag[ index ] )
                            {
                                int pos = pMatchedSum[ index ];
                                pLeftIndices[ pos ] = leftIndex;
                                pRightIndices[ pos ] = right_data[ index ];
                            }
                        }, join_count, context );

                mem_t< int > leftData( matched_total );
                mem_t< int > rightData( matched_total );
                cudaMemcpy( leftData.data(), pLeftIndices, matched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data(), pRightIndices, matched_total * sizeof(int), cudaMemcpyDefault );
                return
                {   std::move( leftData ), std::move( rightData ), matched_total};
            }
            else
            {
                return
                {   std::move( left_output ), std::move( right_output ), join_count};
            }

        }
        return
        {   std::move(left_output), std::move(right_output), join_count};
    }

    template< template< typename, typename > class comp_t, typename launch_arg_t, typename a_it, typename b_it >
    join_pair_t< int > inner_join_wrapper( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a, const int* vals_b,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );

        const char* function_name = nullptr;
        const AriesColumnDataIterator* input = nullptr;
        std::vector< CUmoduleSPtr > modules;
        std::vector< AriesDynamicCodeComparator > comparators;
        shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
        int8_t** constValues = nullptr;

        if( joinDynamicCodeParams )
        {
            function_name = joinDynamicCodeParams->functionName.empty() ? nullptr : joinDynamicCodeParams->functionName.c_str();
            input = joinDynamicCodeParams->input;
            modules = joinDynamicCodeParams->cuModules;
            comparators.assign( joinDynamicCodeParams->comparators.cbegin(), joinDynamicCodeParams->comparators.cend() );

            auto constValueSize = joinDynamicCodeParams->constValues.size();
            if ( constValueSize  > 0 )
            {
                constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                for ( int i = 0; i < constValueSize; ++i )
                    ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                constValueArr->PrefetchToGpu();
                constValues = constValueArr->GetData();
            }
        }

        return inner_join_by_dynamic_code( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, function_name,
                modules, comparators, input, ( const int8_t** )constValues );
    }

    template< typename launch_arg_t, typename comp_t >
    join_pair_t< int > inner_join_wrapper( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );

        const char* function_name = nullptr;
        const AriesColumnDataIterator* input = nullptr;
        std::vector< CUmoduleSPtr > modules;
        std::vector< AriesDynamicCodeComparator > comparators;
        shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
        int8_t** constValues = nullptr;

        if( joinDynamicCodeParams )
        {
            function_name = joinDynamicCodeParams->functionName.empty() ? nullptr : joinDynamicCodeParams->functionName.c_str();
            input = joinDynamicCodeParams->input;
            modules = joinDynamicCodeParams->cuModules;
            comparators.assign( joinDynamicCodeParams->comparators.cbegin(), joinDynamicCodeParams->comparators.cend() );

            auto constValueSize = joinDynamicCodeParams->constValues.size();
            if ( constValueSize  > 0 )
            {
                constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                for ( int i = 0; i < constValueSize; ++i )
                    ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                constValueArr->PrefetchToGpu();
                constValues = constValueArr->GetData();
            }
        }

        return inner_join_by_dynamic_code( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, function_name,
                modules, comparators, input, ( const int8_t** )constValues );
    }

    template< typename launch_arg_t, typename comp_t >
    join_pair_t< int > inner_join_wrapper( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams );
    template< typename launch_arg_t, typename comp_t >
    join_pair_t< int > left_join_wrapper( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a, const int* vals_b, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );

        const char* function_name = nullptr;
        const AriesColumnDataIterator *input = nullptr;
        std::vector< std::shared_ptr< CUmodule > > modules;
        std::vector< AriesDynamicCodeComparator > comparators;
        shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
        int8_t** constValues = nullptr;

        if( joinDynamicCodeParams )
        {
            function_name = joinDynamicCodeParams->functionName.empty() ? nullptr : joinDynamicCodeParams->functionName.c_str();
            input = joinDynamicCodeParams->input;
            modules = joinDynamicCodeParams->cuModules;
            comparators.assign( joinDynamicCodeParams->comparators.cbegin(), joinDynamicCodeParams->comparators.cend() );

            auto constValueSize = joinDynamicCodeParams->constValues.size();
            if ( constValueSize  > 0 )
            {
                constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                for ( int i = 0; i < constValueSize; ++i )
                    ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                constValueArr->PrefetchToGpu();
                constValues = constValueArr->GetData();
            }
        }

        return left_join_by_dynamic_code( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, function_name,
                modules, comparators, input, ( const int8_t** )constValues );
    }

    join_pair_t< int > cartesian_join_wrapper( size_t left_count, size_t right_count, bool need_keep_left, bool need_keep_right, context_t& context,
            const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto join_count = left_count * right_count;

        if( join_count > INT32_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT );
        }

        if( joinDynamicCodeParams && !joinDynamicCodeParams->functionName.empty() )
        {
            managed_mem_t< int > left_indices( join_count, context, false );
            managed_mem_t< int > right_indices( join_count, context, false );

            managed_mem_t< int > left_unmatched_flag( left_count, context, false );
            managed_mem_t< int > right_unmatched_flag( right_count, context, false );

            auto left_data = left_indices.data();
            auto right_data = right_indices.data();

            int* p_left_unmatched_flag = nullptr;
            int* p_right_unmatched_flag = nullptr;

            if( need_keep_left )
            {
                p_left_unmatched_flag = left_unmatched_flag.data();
                init_value( p_left_unmatched_flag, left_count, 1, context );
            }

            if( need_keep_right )
            {
                p_right_unmatched_flag = right_unmatched_flag.data();
                init_value( p_right_unmatched_flag, right_count, 1, context );
            }

            managed_mem_t< unsigned long long > count( 1, context );
            auto pCount = count.data();
            cudaMemset( pCount, 0, sizeof(unsigned long long) );

            shared_ptr< AriesManagedArray< int8_t* > > constValueArr;
            int8_t** constValues = nullptr;
            auto constValueSize = joinDynamicCodeParams->constValues.size();
            if ( constValueSize  > 0 )
            {
                constValueArr = make_shared< AriesManagedArray< int8_t* > >( constValueSize );
                for ( int i = 0; i < constValueSize; ++i )
                    ( *constValueArr )[ i ] = joinDynamicCodeParams->constValues[ i ]->GetData();
                constValueArr->PrefetchToGpu();
                constValues = constValueArr->GetData();
            }

            AriesDynamicKernelManager::GetInstance().CallKernel( joinDynamicCodeParams->cuModules,
                                                                 joinDynamicCodeParams->functionName.c_str(),
                                                                 joinDynamicCodeParams->input,
                                                                 left_count,
                                                                 right_count,
                                                                 join_count,
                                                                 p_left_unmatched_flag,
                                                                 p_right_unmatched_flag,
                                                                 ( const int8_t** )constValues,
                                                                 joinDynamicCodeParams->comparators,
                                                                 left_data,
                                                                 right_data,
                                                                 pCount );
            if( pCount[0] > MAX_JOIN_RESULT_COUNT )
            {
                string msg( "join result rows( " );
                msg.append( to_string( pCount[0] ) ).append( " ) is too big( max = " );
                msg.append( to_string( MAX_JOIN_RESULT_COUNT ) ).append( " )" );
                ARIES_EXCEPTION_SIMPLE( ER_TOO_BIG_SELECT, msg );
            }

            int left_unmatched_count = 0;
            managed_mem_t< int > left_unmatched_left;
            managed_mem_t< int > left_unmatched_right;
            if( need_keep_left )
            {
                managed_mem_t< int > sum( left_count, context );
                managed_mem_t< int > psum_total( 1, context );
                auto p_sum = sum.data();
                scan( p_left_unmatched_flag, left_count, p_sum, plus_t< int32_t >(), psum_total.data(), context );
                context.synchronize();
                left_unmatched_count = psum_total.data()[0];

                if( left_unmatched_count > 0 )
                {
                    managed_mem_t< int > unmatched_left( left_unmatched_count, context );
                    auto p_left = unmatched_left.data();

                    managed_mem_t< int > unmatched_right( left_unmatched_count, context );
                    auto p_right = unmatched_right.data();

                    transform( [ = ]ARIES_DEVICE( int index )
                            {
                                if ( p_left_unmatched_flag[ index ] )
                                {
                                    auto pos = p_sum[ index ];
                                    p_left[ pos ] = index;
                                    p_right[ pos ] = -1;
                                }
                            }, left_count, context );

                    left_unmatched_left = std::move( unmatched_left );
                    left_unmatched_right = std::move( unmatched_right );
                }
            }

            int right_unmatched_count = 0;
            managed_mem_t< int > right_unmatched_left;
            managed_mem_t< int > right_unmatched_right;
            if( need_keep_right )
            {
                managed_mem_t< int > sum( right_count, context );
                managed_mem_t< int > psum_total( 1, context );
                auto p_sum = sum.data();
                scan( p_right_unmatched_flag, right_count, p_sum, plus_t< int32_t >(), psum_total.data(), context );
                context.synchronize();
                right_unmatched_count = psum_total.data()[0];

                if( right_unmatched_count > 0 )
                {
                    managed_mem_t< int > unmatched_left( right_unmatched_count, context );
                    auto p_left = unmatched_left.data();

                    managed_mem_t< int > unmatched_right( right_unmatched_count, context );
                    auto p_right = unmatched_right.data();

                    transform( [ = ]ARIES_DEVICE( int index )
                            {
                                if ( p_right_unmatched_flag[ index ] )
                                {
                                    auto pos = p_sum[ index ];
                                    p_left[ pos ] = -1;
                                    p_right[ pos ] = index;
                                }
                            }, right_count, context );

                    right_unmatched_left = std::move( unmatched_left );
                    right_unmatched_right = std::move( unmatched_right );
                }
            }

            auto total_count = pCount[0] + left_unmatched_count + right_unmatched_count;

            if( total_count > INT32_MAX )
            {
                printf( "join total_count = %llu\n", pCount[0] );
                ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "join's result too big" );
            }

            mem_t< int > actual_left_indices( total_count );
            mem_t< int > actual_right_indices( total_count );

            cudaMemcpy( actual_left_indices.data(), left_indices.data(), actual_left_indices.size() * sizeof(int),
                    cudaMemcpyKind::cudaMemcpyDefault );
            left_indices.free();
            cudaMemcpy( actual_right_indices.data(), right_indices.data(), actual_right_indices.size() * sizeof(int),
                    cudaMemcpyKind::cudaMemcpyDefault );
            right_indices.free();

            if( left_unmatched_count > 0 )
            {
                cudaMemcpy( actual_left_indices.data() + pCount[0], left_unmatched_left.data(), left_unmatched_count * sizeof(int),
                        cudaMemcpyKind::cudaMemcpyDefault );
                cudaMemcpy( actual_right_indices.data() + pCount[0], left_unmatched_right.data(), left_unmatched_count * sizeof(int),
                        cudaMemcpyKind::cudaMemcpyDefault );
            }

            if( right_unmatched_count > 0 )
            {
                cudaMemcpy( actual_left_indices.data() + pCount[0] + left_unmatched_count, right_unmatched_left.data(),
                        right_unmatched_count * sizeof(int), cudaMemcpyKind::cudaMemcpyDefault );
                cudaMemcpy( actual_right_indices.data() + pCount[0] + left_unmatched_count, right_unmatched_right.data(),
                        right_unmatched_count * sizeof(int), cudaMemcpyKind::cudaMemcpyDefault );
            }

            return
            {   std::move( actual_left_indices ), std::move( actual_right_indices ),total_count};
        }
        else if( join_count > 0 )
        {
            mem_t< int > left_indices( join_count );
            mem_t< int > right_indices( join_count );

            auto left_data = left_indices.data();
            auto right_data = right_indices.data();

            transform( [ = ]ARIES_DEVICE( size_t index)
                    {
                        left_data[ index ] = index / right_count;
                        right_data[ index ] = index % right_count;
                    }, join_count, context );

            return
            {   std::move( left_indices ), std::move( right_indices ), join_count};
        }
        else
        {
            return
            {   mem_t< int >(), mem_t< int >(), 0};
        }
    }

END_ARIES_ACC_NAMESPACE
