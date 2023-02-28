/*
 * kernel_shuffle.hxx
 *
 *  Created on: Jun 6, 2019
 *      Author: lichi
 */

#ifndef KERNEL_SHUFFLE_HXX_
#define KERNEL_SHUFFLE_HXX_
#include "transform.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename launch_arg_t = empty_t, typename type_t, typename index_t >
    void shuffle_by_index( const type_t* data_input, size_t count, const index_t* indices, type_t* data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 16 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            data_output[ index ] = data_input[ indices[ index ] ];
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename index_t >
    void shuffle_indices( const type_t** data_input, int input_count, size_t tuple_num, const index_t* indices, type_t** data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 16 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            int ind = indices[ index ];
            if( ind == -1 )
            {
                for( int i = 0; i < input_count; ++i )
                {
                    data_output[ i ][ index ] = -1;
                }
            }
            else
            {
                for( int i = 0; i < input_count; ++i )
                {
                    data_output[ i ][ index ] = data_input[ i ][ ind ];
                }
            }
        };

        transform< launch_t >( k, tuple_num, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename index_t >
    void shuffle_by_index( const char* data_input, size_t len, size_t count, const index_t* indices, char* data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 16 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            memcpy( data_output + index * len, data_input + indices[ index ] * len, len );
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename index_t >
    void shuffle_by_index( const DataBlockInfo* data_input, int32_t block_count, size_t count, const index_t* indices, int8_t** data_output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 16 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            index_t idx = indices[ index ];
            for( int32_t i = 0; i < block_count; ++i )
            {
                DataBlockInfo info = data_input[i];
                memcpy( data_output[i] + index * (size_t)info.ElementSize, info.Data + idx * (size_t)info.ElementSize, (size_t)info.ElementSize );
            }
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

//    template< typename launch_arg_t = empty_t, typename type_t, typename index_t >
//    void shuffle_by_index( const char* data_input_as_char, index_t count, const index_t* indices, char* data_output_as_char, context_t& context )
//    {
//        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 6 > > >::type_t launch_t;
//        int nv = launch_t::nv( context );
//        const type_t* data_input = ( const type_t* )data_input_as_char;
//        type_t* data_output = ( type_t* )data_output_as_char;
//        auto k = [=] ARIES_DEVICE(int tid, int cta)
//        {
//            typedef typename launch_t::sm_ptx params_t;
//            enum
//            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
//            range_t tile = get_tile(cta, nv, count);
//            const index_t* ind = indices + tile.begin;
//            strided_iterate< nt, vt >( [&](int i, int j)
//                    {
//                        data_output[ j + nv * cta ] = data_input[ ind[ j ] ];
//                    }, tid, tile.count() );
//        };
//
//        cta_transform< launch_arg_t >( k, count, context );
//        context.synchronize();
//    }

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_SHUFFLE_HXX_ */
