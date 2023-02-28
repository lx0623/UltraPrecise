/*
 * AriesCastFunctions.hxx
 *
 *  Created on: Sep 25, 2019
 *      Author: lichi
 */

#ifndef ARIESCASTFUNCTIONS_HXX_
#define ARIESCASTFUNCTIONS_HXX_

#include "AriesDefinition.h"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesTime.hxx"
#include "decimal.hxx"
#include "AriesDataTypeUtil.hxx"
#include "aries_types.hxx"
#include "aries_char.hxx"
#include "AriesTimeCalc.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename type_t >
    ARIES_HOST_DEVICE Decimal cast_as_decimal( type_t data, uint32_t precision, uint32_t scale )
    {
        return Decimal( precision, scale ).cast( data );
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal cast_as_decimal( const char* data, int len, uint32_t precision, uint32_t scale );

    ARIES_HOST_DEVICE_NO_INLINE Decimal cast_as_decimal( const char* compact_decimal, uint32_t src_precision, uint32_t src_scale, uint32_t dst_precision, uint32_t dst_scale );

    ARIES_HOST_DEVICE_NO_INLINE Decimal cast_as_decimal( const Decimal &decimal, uint32_t dst_precision, uint32_t dst_scale );

    ARIES_HOST_DEVICE_NO_INLINE nullable_type< Decimal > cast_as_nullable_decimal( const nullable_type< Decimal > &data, uint32_t dst_precision, uint32_t dst_scale );

    template< typename type_t >
    ARIES_HOST_DEVICE nullable_type< Decimal > cast_as_nullable_decimal( nullable_type< type_t > data, uint32_t precision, uint32_t scale )
    {
        return data.flag ? nullable_type< Decimal >( 1, cast_as_decimal( data.value, precision, scale ) ) : nullable_type< Decimal >( 0, Decimal( precision, scale ) );
    }

    ARIES_HOST_DEVICE_NO_INLINE nullable_type< Decimal > cast_as_nullable_decimal( const char* data, int len, uint32_t precision, uint32_t scale );

    ARIES_HOST_DEVICE_NO_INLINE nullable_type< Decimal > cast_as_nullable_decimal( const char* compact_decimal, uint32_t src_precision, uint32_t src_scale, uint32_t dst_precision, uint32_t dst_scale );

    template< typename type_t >
    ARIES_HOST_DEVICE AriesDatetime cast_as_datetime( const type_t &data )
    {
        return AriesDatetime( data );
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime cast_as_datetime( const char* data, int len );

    template< typename type_t >
    ARIES_HOST_DEVICE nullable_type< AriesDatetime > cast_as_nullable_datetime( const nullable_type< type_t > &data )
    {
        return data.flag ? nullable_type< AriesDatetime >( 1, AriesDatetime( data.value ) ) :  nullable_type< AriesDatetime >( 0, AriesDatetime() );
    }

    ARIES_HOST_DEVICE_NO_INLINE nullable_type< AriesDatetime > cast_as_nullable_datetime( const char* data, int len );

    template< typename type_t >
    ARIES_HOST_DEVICE AriesDate cast_as_date( const type_t &data )
    {
        return AriesDate( data );
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate cast_as_date( const char* data, int len );

    template< typename type_t >
    ARIES_HOST_DEVICE nullable_type< AriesDate > cast_as_nullable_date( const nullable_type< type_t > &data )
    {
        return data.flag ? nullable_type< AriesDate >( 1, AriesDate( data.value ) ) :  nullable_type< AriesDate >( 0, AriesDate() );
    }

    ARIES_HOST_DEVICE_NO_INLINE nullable_type< AriesDate > cast_as_nullable_date( const char* data, int len );

    template< typename type_t >
    ARIES_HOST_DEVICE int32_t cast_as_signed( const type_t &data )
    {
        int32_t result = data;
        return result;
    }

    template< int LEN >
    ARIES_HOST_DEVICE int32_t cast_as_signed( const aries_char< LEN >& data )
    {
        return aries_atoi( data.value, data.value + LEN );
    }

    template< typename type_t >
    ARIES_HOST_DEVICE int64_t cast_as_long( const type_t &data )
    {
        int64_t result = data;
        return result;
    }

    template< int LEN >
    ARIES_HOST_DEVICE int64_t cast_as_long( const aries_char< LEN >& data )
    {
        return aries_atol( data.value, data.value + LEN );
    }

    ARIES_HOST_DEVICE_NO_INLINE double cast_as_double( const aries_acc::Decimal& d );

    ARIES_HOST_DEVICE_NO_INLINE float cast_as_float( const aries_acc::Decimal& d );

    template< typename type_t, int LEN, bool has_null >
    struct convert_to_string
    {
    };

    template< int LEN >
    struct convert_to_string< aries_acc::AriesDate, LEN, false >
    {
        ARIES_HOST_DEVICE aries_char< LEN > operator()( const aries_acc::AriesDate &date ) const
        {
            aries_char< LEN > result;
            char* pOut = result;
            if( !DATE_FORMAT( pOut, "%Y-%m-%d", date, aries_acc::LOCALE_LANGUAGE::en_US ) )
            {
                //error happened
                pOut[ 0 ] = 'E';
                pOut[ 1 ] = '\0';
            }
            return result;
        }
    };

    template< int LEN >
    struct convert_to_string< aries_acc::AriesDate, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_char< LEN > > operator()( const nullable_type< aries_acc::AriesDate > &date ) const
        {
            nullable_type< aries_char< LEN > > result;
            char* pOut = result;
            if( date.flag )
            {
                pOut[ 0 ] = 1;
                if( !DATE_FORMAT( pOut + 1, "%Y-%m-%d", date.value, aries_acc::LOCALE_LANGUAGE::en_US ) )
                {
                    //error happened
                    pOut[ 1 ] = 'E';
                    pOut[ 2 ] = '\0';
                }
            }
            else
            {
                pOut[ 0 ] = 0;
            }
            return result;
        }
    };

    template< int LEN >
    struct convert_to_string< aries_acc::AriesDatetime, LEN, false >
    {
        ARIES_HOST_DEVICE aries_char< LEN > operator()( const aries_acc::AriesDatetime &date ) const
        {
            aries_char< LEN > result;
            char* pOut = result;
            if( !DATE_FORMAT( pOut, "%Y-%m-%d %H:%i:%s", date, aries_acc::LOCALE_LANGUAGE::en_US ) )
            {
                //error happened
                pOut[ 0 ] = 'E';
                pOut[ 1 ] = '\0';
            }
            return result;
        }
    };

    template< int LEN >
    struct convert_to_string< aries_acc::AriesDatetime, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_char< LEN > > operator()( const nullable_type< aries_acc::AriesDatetime > &date ) const
        {
            nullable_type< aries_char< LEN > > result;
            char* pOut = result;
            if( date.flag )
            {
                pOut[ 0 ] = 1;
                if( !DATE_FORMAT( pOut + 1, "%Y-%m-%d %H:%i:%s", date.value, aries_acc::LOCALE_LANGUAGE::en_US ) )
                {
                    //error happened
                    pOut[ 1 ] = 'E';
                    pOut[ 2 ] = '\0';
                }
            }
            else
            {
                pOut[ 0 ] = 0;
            }
            return result;
        }
    };

    template< int LEN >
    struct convert_to_string< aries_acc::AriesTimestamp, LEN, false >
    {
        ARIES_HOST_DEVICE aries_char< LEN > operator()( const aries_acc::AriesTimestamp &date ) const
        {
            aries_char< LEN > result;
            char* pOut = result;
            if( !DATE_FORMAT( pOut, "%Y-%m-%d %H:%i:%s", date, aries_acc::LOCALE_LANGUAGE::en_US ) )
            {
                //error happened
                pOut[ 0 ] = 'E';
                pOut[ 1 ] = '\0';
            }
            return result;
        }
    };

    template< int LEN >
    struct convert_to_string< aries_acc::AriesTimestamp, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_char< LEN > > operator()( const nullable_type< aries_acc::AriesTimestamp > &date ) const
        {
            nullable_type< aries_char< LEN > > result;
            char* pOut = result;
            if( date.flag )
            {
                pOut[ 0 ] = 1;
                if( !DATE_FORMAT( pOut + 1, "%Y-%m-%d %H:%i:%s", date.value, aries_acc::LOCALE_LANGUAGE::en_US ) )
                {
                    //error happened
                    pOut[ 1 ] = 'E';
                    pOut[ 2 ] = '\0';
                }
            }
            else
            {
                pOut[ 0 ] = 0;
            }
            return result;
        }
    };

    template< typename type_t, int LEN, bool has_null >
    struct convert_string_to_type
    {
    };

    template< int LEN >
    struct convert_string_to_type< nullable_type< aries_acc::AriesDate >, LEN, false >
    {
        ARIES_HOST_DEVICE nullable_type< aries_acc::AriesDate > operator()( const aries_char< LEN >& value ) const
        {
            nullable_type< aries_acc::AriesDate > result;
            const char* pData = value;
            result.flag = STRING_TO_DATE( pData, LEN, result.value );
            return result;
        }
    };

    template< int LEN >
    struct convert_string_to_type< aries_acc::AriesDate, LEN, false >
    {
        ARIES_HOST_DEVICE aries_acc::AriesDate operator()( const aries_char< LEN >& value ) const
        {
            aries_acc::AriesDate result;
            const char* pData = value;
            STRING_TO_DATE( pData, LEN, result );
            return result;
        }
    };

    template< int LEN >
    struct convert_string_to_type< nullable_type< aries_acc::AriesDate >, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_acc::AriesDate > operator()( const nullable_type< aries_char< LEN > >& value ) const
        {
            nullable_type< aries_acc::AriesDate > result;
            const char* pData = value;
            if( value.flag )
                result.flag = STRING_TO_DATE( pData + 1, LEN, result.value );
            else
                result.flag = 0;
            return result;
        }
    };

    template< int LEN >
    struct convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, LEN, false >
    {
        ARIES_HOST_DEVICE nullable_type< aries_acc::AriesDatetime > operator()( const aries_char< LEN >& value ) const
        {
            nullable_type< aries_acc::AriesDatetime > result;
            const char* pData = value;
            result.flag = STRING_TO_DATE( pData, LEN, result.value );
            return result;
        }
    };

    template< int LEN >
    struct convert_string_to_type< aries_acc::AriesDatetime, LEN, false >
    {
        ARIES_HOST_DEVICE aries_acc::AriesDatetime operator()( const aries_char< LEN >& value ) const
        {
            aries_acc::AriesDatetime result;
            const char* pData = value;
            STRING_TO_DATE( pData, LEN, result );
            return result;
        }
    };

    template< int LEN >
    struct convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_acc::AriesDatetime > operator()( const nullable_type< aries_char< LEN > >& value ) const
        {
            nullable_type< aries_acc::AriesDatetime > result;
            const char* pData = value;
            if( value.flag )
                result.flag = STRING_TO_DATE( pData + 1, LEN, result.value );
            else
                result.flag = 0;
            return result;
        }
    };

END_ARIES_ACC_NAMESPACE

#endif /* ARIESCASTFUNCTIONS_HXX_ */
