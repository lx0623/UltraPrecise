#pragma once
#include "cta_merge.hxx"
#include "search.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< template< typename, typename > class comp_t, bounds_t bounds, typename launch_arg_t = empty_t, typename needles_it,
            typename haystack_it, typename indices_it >
    void sorted_search( needles_it needles, int num_needles, haystack_it haystack, int num_haystack, indices_it indices, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< MODERN_GPU_PARA, 3 > > >::type_t launch_t;
        typedef typename std::common_type< typename std::iterator_traits< needles_it >::value_type,
                typename std::iterator_traits< haystack_it >::value_type >::type type_t;

        // Partition the needles and haystacks into tiles.
        mem_t< int > partitions = merge_path_partitions< comp_t, bounds >( needles, num_needles, haystack, num_haystack, launch_t::nv( context ),
                context );
        const int* mp_data = partitions.data();

        auto k = [=]ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

            __shared__ union named_union
            {
                type_t keys[nv + 1];
                int indices[nv];
                named_union()
                {}
            }shared;

            // Load the range for this CTA and merge the values into register.
            int mp0 = mp_data[cta + 0];
            int mp1 = mp_data[cta + 1];
            merge_range_t range = compute_merge_range(num_needles, num_haystack, cta,
                    nv, mp0, mp1);

            // Merge the values needles and haystack.
            array_t<int, vt> merge = cta_merge_indices_from_mem<comp_t, bounds, nt, vt>(
                    needles, haystack, range, tid, shared.keys);

            // Store the needle indices to shared memory.
            iterate<vt>([&](int i)
                    {
                        if(merge[i] < range.a_count())
                        {
                            int needle = merge[i];
                            int haystack = range.b_begin + vt * tid + i - needle;
                            shared.indices[needle] = haystack;
                        }
                    });
            __syncthreads();

            shared_to_mem<nt, vt>(shared.indices, tid, range.a_count(),
                    indices + range.a_begin);
        };

        cta_transform< launch_t >( k, num_needles + num_haystack, context );
        context.synchronize();
    }

    template< bounds_t bounds, typename launch_arg_t = empty_t, typename indices_it, typename comp_it >
    void sorted_search( const char* needles, int num_needles, const char* haystack, int num_haystack, size_t len, indices_it indices, comp_it comp,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 3 > > >::type_t launch_t;
        typedef typename std::iterator_traits< indices_it >::value_type type_t;
        // Partition the needles and haystacks into tiles.
        mem_t< int > partitions = merge_path_partitions< bounds >( needles, num_needles, haystack, num_haystack, len, launch_t::nv( context ), comp,
                context );
        const int* mp_data = partitions.data();
        size_t shared_mem_half_size = ( launch_t::nv( context ) + 1 ) * max( len, sizeof(type_t) );
        auto k = [=]ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

            extern __shared__ char shared[];

            // Load the range for this CTA and merge the values into register.
            int mp0 = mp_data[cta + 0];
            int mp1 = mp_data[cta + 1];
            merge_range_t range = compute_merge_range(num_needles, num_haystack, cta,
                    nv, mp0, mp1);

            // Merge the values needles and haystack.
            array_t< int, vt > merge = cta_merge_from_mem<bounds, nt, vt>(
                    needles, haystack, len, range, tid, comp, shared, shared + shared_mem_half_size );

            // Store the needle indices to shared memory.
            size_t offset = ( unsigned long long )shared % sizeof(type_t);
            if( offset > 0 )
            offset = sizeof( type_t ) - offset;

            indices_it pIndices = ( indices_it )( shared + offset );
            iterate<vt>([&](int i)
                    {
                        if(merge[i] < range.a_count() )
                        {
                            int needle = merge[i];
                            type_t haystack = range.b_begin + vt * tid + i - needle;
                            pIndices[needle] = haystack;
                        }
                    });
            __syncthreads();

            shared_to_mem<nt, vt>(pIndices, tid, range.a_count(),
                    indices + range.a_begin);
        };

        cta_transform< launch_t >( k, num_needles + num_haystack, shared_mem_half_size * 2, context );
        context.synchronize();
    }

    template< bounds_t bounds, typename launch_arg_t = empty_t, typename needles_it, typename haystack_it, typename indices_it, typename comp_it >
    void sorted_search_with_match_tag( needles_it needles, int num_needles, haystack_it haystack, int num_haystack, indices_it indices, comp_it comp,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        typedef typename std::iterator_traits< needles_it >::value_type type_t;
        typedef typename std::iterator_traits< indices_it >::value_type indices_type_t;

        // Partition the needles and haystacks into tiles.
        mem_t< int > partitions = merge_path_partitions< bounds >( needles, num_needles, haystack, num_haystack, launch_t::nv( context ), comp,
                context );
        const int* mp_data = partitions.data();

        auto k = [=]ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

            __shared__ union named_union
            {
                type_t keys[nv + 1];
                indices_type_t indices[nv];
                named_union()
                {}
            }shared;

            // Load the range for this CTA and merge the values into register.
            int mp0 = mp_data[cta + 0];
            int mp1 = mp_data[cta + 1];
            merge_range_t range = compute_merge_range(num_needles, num_haystack, cta,
                    nv, mp0, mp1);

            // Merge the values needles and haystack.
            array_t< indices_type_t, vt > merge = cta_merge_from_mem_with_match_tag<bounds, nt, vt, indices_type_t>(
                    needles, haystack, range, tid, comp, shared.keys);

            // Store the needle indices to shared memory.
            iterate<vt>([&](int i)
                    {
                        auto needle = merge[i];
                        if(needle.value < range.a_count())
                        shared.indices[needle.value] =
                        {   needle.flag, range.b_begin + vt * tid + i - needle.value};
                    });
            __syncthreads();

            shared_to_mem<nt, vt>(shared.indices, tid, range.a_count(),
                    indices + range.a_begin);
        };

        cta_transform< launch_t >( k, num_needles + num_haystack, context );
        context.synchronize();
    }

    template< bounds_t bounds, typename launch_arg_t = empty_t, typename indices_it, typename comp_it >
    void sorted_search_with_match_tag( const char* needles, int num_needles, const char* haystack, int num_haystack, size_t len, indices_it indices,
            comp_it comp, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 6 > > >::type_t launch_t;
        typedef typename std::iterator_traits< indices_it >::value_type type_t;
        // Partition the needles and haystacks into tiles.
        mem_t< int > partitions = merge_path_partitions< bounds >( needles, num_needles, haystack, num_haystack, len, launch_t::nv( context ), comp,
                context );
        const int* mp_data = partitions.data();
        printf( "type_t size is:%d\n", sizeof(type_t) );
        size_t shared_mem_size = ( launch_t::nv( context ) + 1 ) * max( len, sizeof(type_t) );

        auto k = [=]ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

            extern __shared__ char shared[];

            // Load the range for this CTA and merge the values into register.
            int mp0 = mp_data[cta + 0];
            int mp1 = mp_data[cta + 1];
            merge_range_t range = compute_merge_range(num_needles, num_haystack, cta,
                    nv, mp0, mp1);

            // Merge the values needles and haystack.
            array_t< type_t, vt > merge = cta_merge_from_mem_with_match_tag<bounds, nt, vt,type_t>(
                    needles, haystack, len, range, tid, comp, shared );

            // Store the needle indices to shared memory.
            size_t offset = ( unsigned long long )shared % sizeof(type_t);
            if( offset > 0 )
            offset = sizeof(type_t) - offset;
            indices_it pIndices = ( indices_it )( shared + offset );
            iterate<vt>([&](int i)
                    {
                        auto needle = merge[i];
                        if(needle.value < range.a_count())
                        pIndices[needle.value] =
                        {   needle.flag, range.b_begin + vt * tid + i - needle.value};
                    });
            __syncthreads();

            shared_to_mem<nt, vt>(pIndices, tid, range.a_count(),
                    indices + range.a_begin);
        };

        cta_transform< launch_t >( k, num_needles + num_haystack, shared_mem_size, context );
        context.synchronize();
    }

END_ARIES_ACC_NAMESPACE
