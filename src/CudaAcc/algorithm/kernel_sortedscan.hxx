/*
 * kernel_sortedscan.hxx
 *
 *  Created on: Jun 10, 2019
 *      Author: lichi
 */

#ifndef KERNEL_SORTEDSCAN_HXX_
#define KERNEL_SORTEDSCAN_HXX_

#include "cta_merge.hxx"
#include "search.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< bounds_t bounds, typename launch_arg_t = empty_t, typename needles_it, typename haystack_it, typename indices_it, typename comp_it >
    void sorted_scan( needles_it needles, int num_needles, haystack_it haystack, int num_haystack, indices_it indices, comp_it comp,
            context_t& context )
    {

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 11 > > >::type_t launch_t;
        typedef typename std::iterator_traits< indices_it >::value_type type_t;
        transform< launch_t >([=]ARIES_DEVICE(int index)
                {
                    indices[index] = binary_search<bounds>(haystack, num_haystack, needles[ index ], comp);
                }, num_needles, context);

        context.synchronize();

//        auto k = [=]ARIES_DEVICE(int tid, int cta)
//        {
//            typedef typename launch_t::sm_ptx params_t;
//            enum
//            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
//
//            __shared__ type_t shared[ nt * vt ];
//            range_t tile = get_tile(cta, nv, num_needles);
//            int count = tile.count();
//            // Load the range for this CTA and merge the values into register.
//            const type_t* p = needles + tile.begin;
//            strided_iterate< nt, vt >( [&](int i, int j)
//                    {
//                        shared[j]=p[j];
//                    }, tid, count );
//            __syncthreads();
//
//            array_t< type_t, vt > x;
//            thread_iterate< vt >( [&](int i, int j)
//                    {
//                        x[i] = binary_search<bounds>(haystack, num_haystack, shared[j], comp);
//                    }, tid, num_needles );
//
//            reg_to_shared_thread< nt, vt >( x, tid, shared, count );
//            shared_to_mem< nt, vt >( shared, tid, count, indices + tile.begin, false );
//
//        };
//
//        cta_transform< launch_t >( k, num_needles, 0, context );
//        context.synchronize();
    }

    template< bounds_t bounds, typename launch_arg_t = empty_t, typename indices_it, typename comp_it >
    void sorted_scan( const char* needles, int num_needles, const char* haystack, int num_haystack, size_t len, indices_it indices, comp_it comp,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 3 > > >::type_t launch_t;
        typedef typename std::iterator_traits< indices_it >::value_type type_t;
//        transform< launch_t >([=]ARIES_DEVICE(int index)
//                {
//                    indices[index] = binary_search<bounds>(haystack, len, num_haystack, needles + index * len, comp);
//                }, num_needles, context);
//
//        context.synchronize();

        size_t shared_mem_size = ( launch_t::nv( context ) ) * len;
        auto k = [=]ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

            extern __shared__ char shared[];
            range_t tile = get_tile(cta, nv, num_needles);
            int count = tile.count();
            // Load the range for this CTA and merge the values into register.
            const char* p = needles + tile.begin * len;
            strided_iterate< nt, vt >( [&, len](int i, int j)
                    {
                        memcpy( shared + j * len, p + j * len, len );
                    }, tid, count );
            __syncthreads();

            array_t< type_t, vt > x;
            strided_iterate< nt, vt >( [&](int i, int j)
                    {
                        x[i] = binary_search<bounds>(haystack, len, num_haystack, shared + j * len, comp);
                    }, tid, count );

            reg_to_mem_strided< nt, vt >( x, tid, count, indices + tile.begin );
        };

        cta_transform< launch_t >( k, num_needles, shared_mem_size, context );
        context.synchronize();
    }

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_SORTEDSCAN_HXX_ */
