/*
 * kernel_filter.hxx
 *
 *  Created on: Jun 18, 2019
 *      Author: lichi
 */

#ifndef KERNEL_FILTER_HXX_
#define KERNEL_FILTER_HXX_

#include "transform.hxx"
#include "AriesColumnDataIterator.hxx"

BEGIN_ARIES_ACC_NAMESPACE
    template< typename input_type_t, typename comp_t, typename launch_arg_t = empty_t >
    void filter_data_if( const AriesColumnDataIterator *input, size_t count, comp_t cmp, int32_t param, AriesBool* output, context_t& context )
    {
         typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
         auto k = [=] ARIES_DEVICE(int index)
         {
            auto item = *( ( input_type_t* )( input[ 0 ][ index ] ) );
            output[ index ] = ( AriesBool )cmp( item, param );
         };
 
        transform< launch_t >( k, count, context );
         context.synchronize();
     }

    template< typename type_t, typename launch_arg_t = empty_t, typename type_u = type_t, typename comp_t, typename output_t >
    void filter_data_if( const AriesColumnDataIterator* input, comp_t cmp, type_u param, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            auto item = *( ( type_t * )( input[ 0 ][ index ] ) );
            output[ index ] = ( output_t )cmp( item, param );
        };

        transform< launch_t >( k, input->m_itemCount, context );
        context.synchronize();
    }

    template< typename type_t, typename launch_arg_t = empty_t, typename type_u = type_t, typename comp_t, typename output_t >
    void filter_data_if( const AriesColumnDataIterator* data, comp_t cmp, const type_u* params, size_t param_count, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( * ( ( type_t* )data[ 0 ][ index ] ), params, param_count );
        };

        transform< launch_t >( k, data->m_itemCount, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u = type_t, typename comp_t, typename output_t >
    void filter_data_if( const type_t* data, size_t count, comp_t cmp, type_u param, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( data[ index ], param );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void filter_data_if( const AriesColumnDataIterator* data, size_t len, comp_t cmp, const char* param, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( ( const char* )( data[ 0 ][ index ] ), param, len );
        };

        transform< launch_t >( k, data->m_itemCount, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void filter_data_if( const char* data, size_t len, size_t count, comp_t cmp, const char* param, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( data + index * len, param, len );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u = type_t, typename comp_t, typename output_t >
    void filter_data_if( const type_t* data, size_t count, comp_t cmp, const type_u* params, size_t param_count, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( data[ index ], params, param_count );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void filter_data_if( const AriesColumnDataIterator* data, size_t len, comp_t cmp, const char* param, size_t param_count, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( ( const char* )( data[ 0 ][ index ] ), len, param, param_count );
        };

        transform< launch_t >( k, data->m_itemCount, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void filter_data_if( const char* data, size_t len, size_t count, comp_t cmp, const char* param, size_t param_count, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( data + index * len, len, param, param_count );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void compare_two_array( const char* left, size_t len, size_t count, comp_t cmp, const char* right, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            size_t offset = index * len;
            output[ index ] = (output_t)cmp( left + offset, right + offset, len );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u = type_t, typename comp_t, typename output_t >
    void compare_two_array( const type_t* left, size_t count, comp_t cmp, type_u* right, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( left[ index ], right[ index ] );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void filter_data_if_has_null( const CompactDecimal* data, size_t len, uint32_t precision, uint32_t scale, size_t count, comp_t cmp, type_t param,
            output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( nullable_type< Decimal >( *(int8_t*)(data + index * len), Decimal( data + index * len + 1, precision, scale ) ), param );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void filter_data_if( const CompactDecimal* data, size_t len, uint32_t precision, uint32_t scale, size_t count, comp_t cmp, type_t param,
            output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( Decimal( data + index * len, precision, scale ), param );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void filter_data_if_has_null( const CompactDecimal* data, size_t len, uint32_t precision, uint32_t scale, size_t count, comp_t cmp,
            const type_t* params, size_t param_count, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( nullable_type< Decimal >( *(int8_t*)(data + index * len), Decimal( data + index * len + 1, precision, scale ) ), params, param_count );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void filter_data_if( const CompactDecimal* data, size_t len, uint32_t precision, uint32_t scale, size_t count, comp_t cmp, const type_t* params,
            size_t param_count, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( Decimal( data + index * len, precision, scale ), params, param_count );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void compare_two_array( const CompactDecimal* left, size_t len, uint32_t precision, uint32_t scale, size_t count, comp_t cmp, const type_t* right,
            output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( Decimal( left + index * len, precision, scale ), right[ index ] );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void compare_two_array_has_null( const CompactDecimal* left, size_t len, uint32_t precision, uint32_t scale, size_t count, comp_t cmp,
            const type_t* right, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( nullable_type< Decimal >( *(int8_t*)(left + index * len), Decimal( left + index * len + 1, precision, scale ) ), right[ index ] );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void compare_two_array( const type_t* left, size_t count, comp_t cmp, const CompactDecimal* right, size_t len, uint32_t precision, uint32_t scale,
            output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = (output_t)cmp( left[ index ], Decimal( right + index * len, precision, scale ) );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename comp_t, typename output_t >
    void compare_two_array_has_null( const type_t* left, size_t count, comp_t cmp, const CompactDecimal* right, size_t len, uint32_t precision,
            uint32_t scale, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( left[ index ], nullable_type< Decimal >( *(int8_t*)(right + index * len), Decimal( right + index * len + 1, precision, scale ) ) );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void compare_two_array( const CompactDecimal* left, size_t left_len, uint32_t left_precision, uint32_t left_scale, size_t count, comp_t cmp,
            const CompactDecimal* right, size_t right_len, uint32_t right_precision, uint32_t right_scale, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( Decimal( left + index * left_len, left_precision, left_scale ), Decimal( right + index * right_len, right_precision, right_scale ) );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void compare_two_array_left_has_null( const CompactDecimal* left, size_t left_len, uint32_t left_precision, uint32_t left_scale, size_t count,
            comp_t cmp, const CompactDecimal* right, size_t right_len, uint32_t right_precision, uint32_t right_scale, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( nullable_type< Decimal >( *(int8_t*)(left + index * left_len), Decimal( left + index * left_len + 1, left_precision, left_scale ) ), Decimal( right + index * right_len, right_precision, right_scale ) );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void compare_two_array_right_has_null( const CompactDecimal* left, size_t left_len, uint32_t left_precision, uint32_t left_scale, size_t count,
            comp_t cmp, const CompactDecimal* right, size_t right_len, uint32_t right_precision, uint32_t right_scale, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( Decimal( left + index * left_len, left_precision, left_scale ), nullable_type< Decimal >( *(int8_t*)(right + index * right_len), Decimal( right + index * right_len + 1, right_precision, right_scale ) ) );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename comp_t, typename output_t >
    void compare_two_array_both_has_null( const CompactDecimal* left, size_t left_len, uint32_t left_precision, uint32_t left_scale, size_t count,
            comp_t cmp, const CompactDecimal* right, size_t right_len, uint32_t right_precision, uint32_t right_scale, output_t* output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k =
                [=] ARIES_DEVICE(int index)
                {
                    output[ index ] = (output_t)cmp( nullable_type< Decimal >( *(int8_t*)(left + index * left_len), Decimal( left + index * left_len + 1, left_precision, left_scale ) ), nullable_type< Decimal >( *(int8_t*)(right + index * right_len), Decimal( right + index * right_len + 1, right_precision, right_scale ) ) );
                };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_FILTER_HXX_ */
