// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "cta_merge.hxx"
#include "search.hxx"

BEGIN_ARIES_ACC_NAMESPACE

// Key-value merge.
    template< typename launch_arg_t = empty_t, typename a_keys_it, typename a_vals_it, typename b_keys_it, typename b_vals_it, typename c_keys_it,
            typename c_vals_it, typename comp_t >
    void merge( a_keys_it a_keys, a_vals_it a_vals, int a_count, b_keys_it b_keys, b_vals_it b_vals, int b_count, c_keys_it c_keys, c_vals_it c_vals,
            comp_t comp, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< MODERN_GPU_PARA, 3 > > >::type_t launch_t;

        typedef typename std::iterator_traits< a_keys_it >::value_type type_t;
        typedef typename std::iterator_traits< a_vals_it >::value_type val_t;
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };

        mem_t< int > partitions = merge_path_partitions< bounds_lower >( a_keys, a_count, b_keys, b_count, launch_t::nv( context ), comp, context );
        int* mp_data = partitions.data();

        auto k = [=] ARIES_DEVICE (int tid, int cta)
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
            merge_range_t range = compute_merge_range(a_count, b_count, cta, nv,
                    mp0, mp1);

            merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                    a_keys, b_keys, range, tid, comp, shared.keys);

            int dest_offset = nv * cta;
            reg_to_mem_thread<nt>(merge.keys, tid, range.total(), c_keys + dest_offset,
                    shared.keys);

            if(has_values)
            {
                // Transpose the indices from thread order to strided order.
                array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices, tid,
                        shared.indices);

                // Gather the input values and merge into the output values.
                transfer_two_streams_strided<nt>(a_vals + range.a_begin, range.a_count(),
                        b_vals + range.b_begin, range.b_count(), indices, tid,
                        c_vals + dest_offset);
            }
        };
        cta_transform< launch_t >( k, a_count + b_count, context );
    }

    template< typename launch_arg_t = empty_t, typename a_keys_it, typename a_vals_it, typename b_keys_it, typename b_vals_it, typename c_vals_it,
            typename comp_t >
    void merge_vals_only( a_keys_it a_keys, a_vals_it a_vals, int a_count, b_keys_it b_keys, b_vals_it b_vals, int b_count, c_vals_it c_vals,
            comp_t comp, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 3 > > >::type_t launch_t;

        typedef typename std::iterator_traits< a_keys_it >::value_type type_t;
        typedef typename std::iterator_traits< a_vals_it >::value_type val_t;

        mem_t< int > partitions = merge_path_partitions< bounds_lower >( a_keys, a_count, b_keys, b_count, launch_t::nv( context ), comp, context );
        int* mp_data = partitions.data();

        auto k = [=] ARIES_DEVICE (int tid, int cta)
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
            merge_range_t range = compute_merge_range(a_count, b_count, cta, nv,
                    mp0, mp1);

            merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                    a_keys, b_keys, range, tid, comp, shared.keys);

            int dest_offset = nv * cta;

            // Transpose the indices from thread order to strided order.
            array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices, tid,
                    shared.indices);

            // Gather the input values and merge into the output values.
            transfer_two_streams_strided<nt>(a_vals + range.a_begin, range.a_count(),
                    b_vals + range.b_begin, range.b_count(), indices, tid,
                    c_vals + dest_offset);
        };
        cta_transform< launch_t >( k, a_count + b_count, context );
    }

    template< typename launch_arg_t = empty_t, typename vals_it, typename comp_t >
    void merge( const char* a_keys, vals_it a_vals, int a_count, const char* b_keys, vals_it b_vals, int b_count, char* c_keys, vals_it c_vals,
            size_t len, comp_t comp, context_t& context )
    {

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 15 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 6 > > >::type_t launch_t;

        typedef typename std::iterator_traits< vals_it >::value_type val_t;
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };
        int nv = launch_t::nv( context );
        mem_t< int > partitions = merge_path_partitions< bounds_lower >( a_keys, a_count, b_keys, b_count, len, launch_t::nv( context ), comp,
                context );
        int* mp_data = partitions.data();
        size_t shared_mem_half_size = ( nv + 1 ) * max( len, sizeof(val_t) ); // merge时会对共享内存越界访问１个元素，所以多分配一个元素
        auto k = [=] ARIES_DEVICE (int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

            extern __shared__ char keys[];
            range_t tile = get_tile(cta, nv, a_count + b_count);
            // Load the range for this CTA and merge the values into register.
            int mp0 = mp_data[cta + 0];
            int mp1 = mp_data[cta + 1];
            merge_range_t range = compute_merge_range(a_count, b_count, cta, nv,
                    mp0, mp1);

            //int keysOffset;
            array_t< int, vt > merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                    a_keys, b_keys, len, range, tid, comp, keys, keys + shared_mem_half_size );

            // Store merged values back out.
            shared_to_mem<nt,vt>( keys + shared_mem_half_size, len, tid, tile.count(), c_keys + tile.begin * len );

            if(has_values)
            {
                char* p = keys;
                //　必须对int对齐处理
                size_t offset = ( unsigned long long )keys % sizeof(int);
                if( offset > 0 )
                offset = sizeof(int) - offset;
                p += offset;

                array_t<int, vt> indices = reg_thread_to_strided<nt>(merge,
                        tid, (int*)p, nv);

                // Gather the input values and merge into the output values.
                transfer_two_streams_strided<nt>(a_vals + range.a_begin,
                        range.a_count(), b_vals + range.b_begin, range.b_count(),
                        indices, tid, c_vals + tile.begin);
            }
        };
        cta_transform< launch_t >( k, a_count + b_count, shared_mem_half_size * 2, context );
    }

// Key-only merge.
    template< typename launch_t = empty_t, typename a_keys_it, typename b_keys_it, typename c_keys_it, typename comp_t >
    void merge( a_keys_it a_keys, int a_count, b_keys_it b_keys, int b_count, c_keys_it c_keys, comp_t comp, context_t& context )
    {
        merge< launch_t >( a_keys, ( const empty_t* )nullptr, a_count, b_keys, ( const empty_t* )nullptr, b_count, c_keys, ( empty_t* )nullptr, comp,
                context );
    }

    template< typename launch_t = empty_t, typename comp_t >
    void merge( const char* a_keys, int a_count, const char* b_keys, int b_count, char* c_keys, int len, comp_t comp, context_t& context )
    {
        merge< launch_t >( a_keys, ( const empty_t* )nullptr, a_count, b_keys, ( const empty_t* )nullptr, b_count, c_keys, ( empty_t* )nullptr, len,
                comp, context );
    }

END_ARIES_ACC_NAMESPACE
