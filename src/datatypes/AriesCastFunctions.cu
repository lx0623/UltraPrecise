/*
 * AriesCastFunctions.cu
 *
 *  Created on: Sep 25, 2019
 *      Author: lichi
 */

#include "AriesCastFunctions.hxx"
BEGIN_ARIES_ACC_NAMESPACE

ARIES_HOST_DEVICE_NO_INLINE double cast_as_double( const aries_acc::Decimal& d )
{
    return d.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE float cast_as_float( const aries_acc::Decimal& d )
{
    return d.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE Decimal cast_as_decimal( const char* data, int len, uint32_t precision, uint32_t scale )
{
    return Decimal( precision, scale ).cast( Decimal( data, len ) );
}

ARIES_HOST_DEVICE_NO_INLINE Decimal cast_as_decimal( const char* compact_decimal, uint32_t src_precision, uint32_t src_scale, uint32_t dst_precision, uint32_t dst_scale )
{
    return Decimal( dst_precision, dst_scale ).cast( Decimal( (CompactDecimal*)compact_decimal, src_precision, src_scale ) );
}

ARIES_HOST_DEVICE_NO_INLINE Decimal cast_as_decimal( const Decimal &decimal, uint32_t dst_precision, uint32_t dst_scale )
{
    return Decimal( dst_precision, dst_scale ).cast( decimal );
}

ARIES_HOST_DEVICE_NO_INLINE nullable_type< Decimal > cast_as_nullable_decimal( const nullable_type< Decimal > &data, uint32_t dst_precision, uint32_t dst_scale )
{
    return data.flag ? nullable_type< Decimal >( 1, cast_as_decimal( data.value, dst_precision, dst_scale ) ) : nullable_type< Decimal >( 0, Decimal( dst_precision, dst_scale ) );
}

ARIES_HOST_DEVICE_NO_INLINE nullable_type< Decimal > cast_as_nullable_decimal( const char* data, int len, uint32_t precision, uint32_t scale )
{
    return *data ? nullable_type< Decimal >( 1, cast_as_decimal( data + 1, len -1, precision, scale ) ) : nullable_type< Decimal >( 0, Decimal( precision, scale ) );
}

ARIES_HOST_DEVICE_NO_INLINE nullable_type< Decimal > cast_as_nullable_decimal( const char* compact_decimal, uint32_t src_precision, uint32_t src_scale, uint32_t dst_precision, uint32_t dst_scale )
{
    return *compact_decimal ? nullable_type< Decimal >( 1, cast_as_decimal( compact_decimal + 1, src_precision, src_scale, dst_precision, dst_scale ) ) : nullable_type< Decimal >( 0, Decimal( dst_precision, dst_scale ) );
}

ARIES_HOST_DEVICE_NO_INLINE AriesDatetime cast_as_datetime( const char* data, int len )
{
    AriesDatetime result;
    STRING_TO_DATE( data, len, result );
    return result;
}

ARIES_HOST_DEVICE_NO_INLINE nullable_type< AriesDatetime > cast_as_nullable_datetime( const char* data, int len )
{
    return *data ? nullable_type< AriesDatetime >( 1, cast_as_datetime( data + 1, len -1 ) ) : nullable_type< AriesDatetime >( 0, AriesDatetime() );
}

ARIES_HOST_DEVICE_NO_INLINE AriesDate cast_as_date( const char* data, int len )
{
    AriesDate result;
    STRING_TO_DATE( data, len, result );
    return result;
}

ARIES_HOST_DEVICE_NO_INLINE nullable_type< AriesDate > cast_as_nullable_date( const char* data, int len )
{
    return *data ? nullable_type< AriesDate >( 1, cast_as_date( data + 1, len -1 ) ) : nullable_type< AriesDate >( 0, AriesDate() );
}

END_ARIES_ACC_NAMESPACE