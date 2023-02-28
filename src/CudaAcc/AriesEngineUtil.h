/*
 * AriesEngineUtil.h
 *
 *  Created on: Jun 16, 2019
 *      Author: lichi
 */

#ifndef ARIESENGINEUTIL_H_
#define ARIESENGINEUTIL_H_
#include <type_traits>
#include <vector>
#include "algorithm/context.hxx"
#include "algorithm/types.hxx"
#include "algorithm/kernel_join.hxx"
#include "algorithm/kernel_mergesort.hxx"
#include "algorithm/kernel_radixsort.hxx"
#include "algorithm/kernel_shuffle.hxx"
#include "algorithm/kernel_sortedsearch.hxx"
#include "algorithm/kernel_segsort.hxx"
#include "algorithm/kernel_load_balance.hxx"
#include "algorithm/kernel_segreduce.hxx"
#include "algorithm/kernel_reduce.hxx"
#include "algorithm/kernel_filter.hxx"
#include "algorithm/kernel_util.hxx"
#include "algorithm/kernel_intervalmove.hxx"
#include "algorithm/kernel_functions.hxx"
#include "AriesEngineDef.h"

template<>
struct std::numeric_limits< aries_acc::Decimal >: public std::numeric_limits< long >
{
    //TODO add specilize for decimal;
};

BEGIN_ARIES_ACC_NAMESPACE

    void find_group_bounds( const std::vector< AriesDataBufferSPtr >& columns, const int *associated, int *groups, context_t& context );
    AriesUInt32ArraySPtr LoadDataAsUInt32( const AriesDataBufferSPtr& column, const AriesInt32ArraySPtr& flags, const AriesInt32ArraySPtr& psum,
            int offset, size_t count, context_t& context );
    AriesUInt32ArraySPtr LoadDataAsUInt32Ex( const AriesDataBufferSPtr& column, const AriesInt32ArraySPtr& associated, int offset, context_t& context );
    AriesInt32ArraySPtr ConvertToOriginalFlag( const AriesInt32ArraySPtr& flags, const AriesInt32ArraySPtr& oldOriginal, const AriesInt32ArraySPtr& oldPsum, context_t& context );
    AriesInt32ArraySPtr FindBoundArray(const AriesUInt32ArraySPtr& data, context_t& context );
    template< typename type_t >
    struct ColumnTypeConverter
    {
        ColumnTypeConverter( bool hasNull )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< int8_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::INT8, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< uint8_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::UINT8, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< int16_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::INT16, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< uint16_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::UINT16, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< int32_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::INT32, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< uint32_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::UINT32, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< int64_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::INT64, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< uint64_t >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::UINT64, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< float >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::FLOAT, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< double >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::DOUBLE, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< Decimal >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::DECIMAL, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< AriesDate >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::DATE, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< AriesDatetime >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::DATETIME, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< AriesTimestamp >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::TIMESTAMP, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< AriesTime >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::TIME, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

    template< >
    struct ColumnTypeConverter< AriesYear >
    {
        ColumnTypeConverter( bool hasNull )
                : ColumnType( { AriesValueType::YEAR, 1 }, hasNull, false )
        {
        }
        AriesColumnType ColumnType;
    };

END_ARIES_ACC_NAMESPACE

template< typename type_t, template< typename > class type_nullable >
struct std::numeric_limits< type_nullable< type_t > >: public std::numeric_limits< type_t >
{

};

#endif /* ARIESENGINEUTIL_H_ */
