/*
 * kernel_functions.hxx
 *
 *  Created on: Jul 24, 2019
 *      Author: lichi
 */

#ifndef KERNEL_FUNCTIONS_HXX_
#define KERNEL_FUNCTIONS_HXX_

#include "functions.hxx"
#include "transform.hxx"
#include "AriesTimeCalc.hxx"
#include "AriesSqlFunctions.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< bool has_null >
    void sql_substring( const char *data, size_t len, size_t count, int start, int size, char *outData, size_t outTypeSize, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            op_substring_t< has_null >()( data + index * len, start, size, outData + index * outTypeSize );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename output_t >
    void sql_extract( const type_t *data, size_t count, const interval_type &type, output_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< output_t >( ExtractWrapper< type_t, interval_type >(), data[index], type );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename output_t, template< typename > class type_nullable >
    void sql_extract( const type_nullable< type_t > *data, size_t count, const interval_type &type, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( ExtractWrapper< type_t, interval_type >(), data[index], type );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t >
    void sql_date( const type_t *data, size_t count, AriesDate *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< AriesDate >( DateWrapper< type_t >(), data[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t >
    void sql_date( const nullable_type< type_t > *data, size_t count, nullable_type< AriesDate > *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< nullable_type< AriesDate > >( DateWrapper< type_t >(), data[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t >
    void sql_date_sub( const type_t *data, size_t count, const type_u* interval, output_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< output_t >( DateSubWrapper< type_t, type_u >(), data[index], interval[0] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_date_sub( const type_nullable< type_t > *data, size_t count, const type_u* interval, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( DateSubWrapper< type_t, type_u >(), data[index], interval[0] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename output_t >
    void sql_unix_timestamp( const type_t *data, size_t count, int offset, output_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< output_t >( UnixTimestampWrapper< type_t, int >(), data[index], offset );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename output_t, template< typename > class type_nullable >
    void sql_unix_timestamp( const type_nullable< type_t > *data, size_t count, int offset, type_nullable< output_t > *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( UnixTimestampWrapper< type_t, int >(), data[index], offset );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t >
    void sql_date_diff( const type_t *left, size_t count, const type_u* right, output_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< output_t >( DateDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_date_diff( const type_nullable< type_t > *left, size_t count, const type_u* right, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( DateDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t >
    void sql_date_diff_literal( const type_t *left, size_t count, const type_u* right, output_t *outData, bool bNegative, context_t& context )
    {
        if( bNegative )
        {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = -( sql_function_wrapper< output_t >( DateDiffWrapper< type_t, type_u >(), left[index], right[0] ) );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        }
        else
        {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = sql_function_wrapper< output_t >( DateDiffWrapper< type_t, type_u >(), left[index], right[0] );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        }

        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_date_diff_literal( const type_nullable< type_t > *left, size_t count, const type_u* right, type_nullable< output_t > *outData,
            bool bNegative, context_t& context )
    {
        if( bNegative )
        {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( DateDiffWrapper< type_t, type_u >(), left[index], right[0] );
                outData[index].value = -outData[index].value;
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        }
        else
        {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( DateDiffWrapper< type_t, type_u >(), left[index], right[0] );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        }

        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_date_diff( const type_t *left, size_t count, const type_nullable< type_u >* right, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( DateDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_date_diff( const type_nullable< type_t > *left, size_t count, const type_nullable< type_u >* right, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( DateDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t >
    void sql_time_diff( const type_t *left, size_t count, const type_u* right, output_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< output_t >( TimeDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_time_diff( const type_nullable< type_t > *left, size_t count, const type_u* right, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( TimeDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_time_diff( const type_t *left, size_t count, const type_nullable< type_u >* right, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( TimeDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_time_diff( const type_nullable< type_t > *left, size_t count, const type_nullable< type_u >* right, type_nullable< output_t > *outData,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( TimeDiffWrapper< type_t, type_u >(), left[index], right[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t >
    void sql_time_diff_literal( const type_t *left, size_t count, const type_u* right, output_t *outData, bool bNegative, context_t& context )
    {
        if (bNegative) {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = -sql_function_wrapper< output_t >( TimeDiffWrapper< type_t, type_u >(), left[index], right[0] );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        } else {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = sql_function_wrapper< output_t >( TimeDiffWrapper< type_t, type_u >(), left[index], right[0] );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        }
        context.synchronize();
    }

    template< typename type_t, typename type_u, typename output_t, template< typename > class type_nullable >
    void sql_time_diff_literal( const type_nullable< type_t > *left, size_t count, const type_u* right, type_nullable< output_t > *outData,
                        bool bNegative, context_t& context )
    {
        if (bNegative) {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( TimeDiffWrapper< type_t, type_u >(), left[index], right[0] );
                outData[index].value = -outData[index].value;
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        } else {
            auto k = [=] ARIES_DEVICE(int index)
            {
                outData[ index ] = sql_function_wrapper< type_nullable< output_t > >( TimeDiffWrapper< type_t, type_u >(), left[index], right[0] );
            };
            transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        }
        context.synchronize();
    }

    template< typename type_t >
    void sql_abs( const type_t *data, size_t count, type_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_t >( AbsWrapper< type_t >(), data[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t, template< typename > class type_nullable >
    void sql_abs( const type_nullable< type_t > *data, size_t count, type_nullable< type_t > *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< type_nullable< type_t > >( AbsWrapper< type_t >(), data[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t >
    void sql_month( const type_t *data, size_t count, uint8_t *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< uint8_t >( MonthWrapper< type_t >(), data[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t >
    void sql_month( const nullable_type< type_t > *data, size_t count, nullable_type< uint8_t > *outData, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< nullable_type< uint8_t > >( MonthWrapper< type_t >(), data[index] );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t>
    void sql_dateformat( const type_t *data, size_t count, const char* format, const LOCALE_LANGUAGE &locale, char *outData, size_t outItemLen, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            char *pOut = outData + index * outItemLen;
            if (!DATE_FORMAT( pOut, format, data[index], locale ))
            {
                //error happened
                pOut[0] = 'E';
                pOut[1] = '\0';
            }
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t>
    void sql_dateformat( const nullable_type< type_t > *data, size_t count, const char* format, const LOCALE_LANGUAGE &locale, char *outData, size_t outItemLen, context_t& context )
    {
        auto k = [=] ARIES_DEVICE( int index )
        {
            char *pOut = outData + index * outItemLen;
            if (data[index].flag)
            {
                pOut[0] = 1;
                if (!DATE_FORMAT( pOut + 1, format, data[index].value, locale ))
                {
                    //error happened
                    pOut[1] = 'E';
                    pOut[2] = '\0';
                }
            }
            else
            {
                pOut[0] = 0;
            }
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t>
    void sql_dateformat( const type_t *data, const AriesDatetime &currentDate, size_t count, const char* format, const LOCALE_LANGUAGE &locale, char *outData, size_t outItemLen, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            char *pOut = outData + index * outItemLen;
            if (!DATE_FORMAT( pOut, format, currentDate + data[index], locale ))
            {
                // error happened
                pOut[0] = 'E';
                pOut[1] = '\0';
            }
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename type_t>
    void sql_dateformat( const nullable_type< type_t > *data, const AriesDatetime &currentDate, size_t count, const char* format, const LOCALE_LANGUAGE &locale, char *outData, size_t outItemLen, context_t& context )
    {
        auto k = [=] ARIES_DEVICE( int index )
        {
            char *pOut = outData + index * outItemLen;
            if (data[index].flag)
            {
                pOut[0] = 1;
                if (!DATE_FORMAT( pOut + 1, format, currentDate + data[index].value, locale ))
                {
                    //error happened
                    pOut[1] = 'E';
                    pOut[2] = '\0';
                }
            }
            else
            {
                pOut[0] = 0;
            }
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t >
    void sql_abs_compact_decimal( const CompactDecimal *data, size_t len, uint16_t precision, uint16_t scale, size_t count, Decimal *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< Decimal >( AbsWrapper< Decimal >(), Decimal( data + index * len, precision, scale ) );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t >
    void sql_abs_compact_decimal( const CompactDecimal *data, size_t len, uint16_t precision, uint16_t scale, size_t count, nullable_type< Decimal > *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ index ] = sql_function_wrapper< nullable_type< Decimal > >( AbsWrapper< Decimal >(), nullable_type< Decimal >( *(int8_t*)(data + index * len), Decimal( data + index * len + 1, precision, scale ) ) );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_FUNCTIONS_HXX_ */
