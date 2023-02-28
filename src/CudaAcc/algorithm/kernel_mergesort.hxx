// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "transform.hxx"
#include "kernel_merge.hxx"
#include "search.hxx"
#include "cta_mergesort.hxx"
#include "intrinsics.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename keys_it, typename comp_t >
    mem_t< int > merge_sort_partitions( keys_it keys, int count, int coop, int spacing, comp_t comp, context_t& context )
    {

        int num_partitions = div_up( count, spacing ) + 1;
        auto k = [=]ARIES_DEVICE(int index)
        {
            merge_range_t range = compute_mergesort_range(count, index, coop, spacing);
            int diag = min(spacing * index, count) - range.a_begin;
            return merge_path<bounds_lower>(keys + range.a_begin, range.a_count(),
                    keys + range.b_begin, range.b_count(), diag, comp);
        };

        return fill_function< int >( k, num_partitions, context );
    }

    template< typename comp_t >
    mem_t< int > merge_sort_partitions( const char* keys, size_t len, int count, int coop, int spacing, comp_t comp, context_t& context )
    {

        int num_partitions = div_up( count, spacing ) + 1;
        auto k = [=]ARIES_DEVICE(int index)
        {
            merge_range_t range = compute_mergesort_range(count, index, coop, spacing);
            int diag = min(spacing * index, count) - range.a_begin;
            return merge_path<bounds_lower>(keys + range.a_begin * len, range.a_count(),
                    keys + range.b_begin * len, range.b_count(), len, diag, comp);
        };

        return fill_function< int >( k, num_partitions, context );
    }

// Key-value mergesort.
    template< typename launch_arg_t = empty_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
    void mergesort( key_t* keys_input, val_t* vals_input, int count, comp_t< key_t, key_t > comp, context_t& context )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 17 >, arch_35_cta< 128, 11 >, arch_52_cta< MODERN_GPU_PARA, 3 > > >::type_t launch_t;
        int nv = launch_t::nv( context );
        int num_ctas = div_up( count, nv );
        int num_passes = find_log2( num_ctas, true );

        mem_t< key_t > keys_temp( num_passes ? count : 0 );
        key_t* keys_output = keys_temp.data();

        mem_t< val_t > vals_temp( has_values && num_passes ? count : 0 );
        val_t* vals_output = vals_temp.data();

        key_t* keys_blocksort = ( 1 & num_passes ) ? keys_output : keys_input;
        val_t* vals_blocksort = ( 1 & num_passes ) ? vals_output : vals_input;

        auto k = [=] ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
            typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;

            __shared__ union named_union
            {
                typename sort_t::storage_t sort;
                key_t keys[nv];
                val_t vals[nv];
                named_union()
                {}
            }shared;

            range_t tile = get_tile(cta, nv, count);

            // Load the keys and values.
                kv_array_t<key_t, val_t, vt> unsorted;
                unsorted.keys = mem_to_reg_thread<nt, vt>((key_t*)keys_input + tile.begin, tid,
                        tile.count(), shared.keys);
                if(has_values)
                unsorted.vals = mem_to_reg_thread<nt, vt>(vals_input + tile.begin, tid,
                        tile.count(), shared.vals);

                // Blocksort.
                kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
                        tid, tile.count(), comp, shared.sort);

                // Store the keys and values.
                reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(),
                        keys_blocksort + tile.begin, shared.keys);
                if(has_values)
                reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(),
                        vals_blocksort + tile.begin, shared.vals);
            };

        cta_transform< launch_t >( k, count, context );

        if( 1 & num_passes )
        {
            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }

        for( int pass = 0; pass < num_passes; ++pass )
        {
            int coop = 2 << pass;
            mem_t< int > partitions = merge_sort_partitions( keys_input, count, coop, nv, comp, context );
            int* mp_data = partitions.data();

            auto k = [=] ARIES_DEVICE(int tid, int cta)
            {
                typedef typename launch_t::sm_ptx params_t;
                enum
                {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

                __shared__ union named_union
                {
                    key_t keys[nv + 1];
                    int indices[nv];
                    named_union()
                    {}
                }shared;

                range_t tile = get_tile(cta, nv, count);

                // Load the range for this CTA and merge the values into register.
                    merge_range_t range = compute_mergesort_range(count, cta, coop, nv,
                            mp_data[cta + 0], mp_data[cta + 1]);

                    merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                            keys_input, keys_input, range, tid, comp, shared.keys);

                    // Store merged values back out.
                    reg_to_mem_thread<nt>(merge.keys, tid, tile.count(),
                            keys_output + tile.begin, shared.keys);

                    if(has_values)
                    {
                        // Transpose the indices from thread order to strided order.
                        array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
                                tid, shared.indices);

                        // Gather the input values and merge into the output values.
                        transfer_two_streams_strided<nt>(vals_input + range.a_begin,
                                range.a_count(), vals_input + range.b_begin, range.b_count(),
                                indices, tid, vals_output + tile.begin);
                    }
                };
            cta_transform< launch_t >( k, count, context );

            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
    void mergesort( key_t** key_blocks, int64_t* block_size_prefix_sum, int block_count, key_t* keys_input, val_t* vals_input, int count,
            comp_t< key_t, key_t > comp, context_t& context )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 17 >, arch_35_cta< 128, 11 >, arch_52_cta< MODERN_GPU_PARA, 3 > > >::type_t launch_t;
        int nv = launch_t::nv( context );
        int num_ctas = div_up( count, nv );
        int num_passes = find_log2( num_ctas, true );

        mem_t< key_t > keys_temp( num_passes ? count : 0 );
        key_t* keys_output = keys_temp.data();

        mem_t< val_t > vals_temp( has_values && num_passes ? count : 0 );
        val_t* vals_output = vals_temp.data();

        key_t* keys_blocksort = ( 1 & num_passes ) ? keys_output : keys_input;
        val_t* vals_blocksort = ( 1 & num_passes ) ? vals_output : vals_input;

        auto k = [=] ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
            typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;

            __shared__ union named_union
            {
                typename sort_t::storage_t sort;
                key_t keys[nv];
                val_t vals[nv];
                named_union()
                {}
            }shared;

            range_t tile = get_tile(cta, nv, count);

            // Load the keys and values.
                kv_array_t<key_t, val_t, vt> unsorted;
                unsorted.keys = mem_to_reg_thread_multi_block<nt, vt>(key_blocks, block_size_prefix_sum, block_count, tile.begin, tid,
                        tile.count(), shared.keys);
                if(has_values)
                unsorted.vals = mem_to_reg_thread<nt, vt>(vals_input + tile.begin, tid,
                        tile.count(), shared.vals);

                // Blocksort.
                kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
                        tid, tile.count(), comp, shared.sort);

                // Store the keys and values.
                reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(),
                        keys_blocksort + tile.begin, shared.keys);
                if(has_values)
                reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(),
                        vals_blocksort + tile.begin, shared.vals);
            };

        cta_transform< launch_t >( k, count, context );

        if( 1 & num_passes )
        {
            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }

        for( int pass = 0; pass < num_passes; ++pass )
        {
            int coop = 2 << pass;
            mem_t< int > partitions = merge_sort_partitions( keys_input, count, coop, nv, comp, context );
            int* mp_data = partitions.data();

            auto k = [=] ARIES_DEVICE(int tid, int cta)
            {
                typedef typename launch_t::sm_ptx params_t;
                enum
                {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

                __shared__ union named_union
                {
                    key_t keys[nv + 1];
                    int indices[nv];
                    named_union()
                    {}
                }shared;

                range_t tile = get_tile(cta, nv, count);

                // Load the range for this CTA and merge the values into register.
                    merge_range_t range = compute_mergesort_range(count, cta, coop, nv,
                            mp_data[cta + 0], mp_data[cta + 1]);

                    merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                            keys_input, keys_input, range, tid, comp, shared.keys);

                    // Store merged values back out.
                    reg_to_mem_thread<nt>(merge.keys, tid, tile.count(),
                            keys_output + tile.begin, shared.keys);

                    if(has_values)
                    {
                        // Transpose the indices from thread order to strided order.
                        array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
                                tid, shared.indices);

                        // Gather the input values and merge into the output values.
                        transfer_two_streams_strided<nt>(vals_input + range.a_begin,
                                range.a_count(), vals_input + range.b_begin, range.b_count(),
                                indices, tid, vals_output + tile.begin);
                    }
                };
            cta_transform< launch_t >( k, count, context );

            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
    void mergesort( key_t* data, int* indices, key_t* keys_input, val_t* vals_input, int count, comp_t< key_t, key_t > comp, context_t& context )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 17 >, arch_35_cta< 128, 11 >, arch_52_cta< 256, 3 > > >::type_t launch_t;
        int nv = launch_t::nv( context );
        int num_ctas = div_up( count, nv );
        int num_passes = find_log2( num_ctas, true );

        mem_t< key_t > keys_temp( num_passes ? count : 0 );
        key_t* keys_output = keys_temp.data();

        mem_t< val_t > vals_temp( has_values && num_passes ? count : 0 );
        val_t* vals_output = vals_temp.data();

        key_t* keys_blocksort = ( 1 & num_passes ) ? keys_output : keys_input;
        val_t* vals_blocksort = ( 1 & num_passes ) ? vals_output : vals_input;

        auto k = [=] ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
            typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;

            __shared__ union named_union
            {
                typename sort_t::storage_t sort;
                key_t keys[nv];
                val_t vals[nv];
                named_union()
                {}
            }shared;

            range_t tile = get_tile(cta, nv, count);

            // Load the keys and values.
                kv_array_t<key_t, val_t, vt> unsorted;
                unsorted.keys = mem_to_reg_thread_by_indices<nt, vt>(data, indices + tile.begin, tid,
                        tile.count(), shared.keys);
                if(has_values)
                unsorted.vals = mem_to_reg_thread<nt, vt>(vals_input + tile.begin, tid,
                        tile.count(), shared.vals);

                // Blocksort.
                kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
                        tid, tile.count(), comp, shared.sort);

                // Store the keys and values.
                reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(),
                        keys_blocksort + tile.begin, shared.keys);
                if(has_values)
                reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(),
                        vals_blocksort + tile.begin, shared.vals);
            };

        cta_transform< launch_t >( k, count, context );

        if( 1 & num_passes )
        {
            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }

        for( int pass = 0; pass < num_passes; ++pass )
        {
            int coop = 2 << pass;
            mem_t< int > partitions = merge_sort_partitions( keys_input, count, coop, nv, comp, context );
            int* mp_data = partitions.data();

            auto k = [=] ARIES_DEVICE(int tid, int cta)
            {
                typedef typename launch_t::sm_ptx params_t;
                enum
                {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

                __shared__ union named_union
                {
                    key_t keys[nv + 1];
                    int indices[nv];
                    named_union()
                    {}
                }shared;

                range_t tile = get_tile(cta, nv, count);

                // Load the range for this CTA and merge the values into register.
                    merge_range_t range = compute_mergesort_range(count, cta, coop, nv,
                            mp_data[cta + 0], mp_data[cta + 1]);

                    merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                            keys_input, keys_input, range, tid, comp, shared.keys);

                    // Store merged values back out.
                    reg_to_mem_thread<nt>(merge.keys, tid, tile.count(),
                            keys_output + tile.begin, shared.keys);

                    if(has_values)
                    {
                        // Transpose the indices from thread order to strided order.
                        array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
                                tid, shared.indices);

                        // Gather the input values and merge into the output values.
                        transfer_two_streams_strided<nt>(vals_input + range.a_begin,
                                range.a_count(), vals_input + range.b_begin, range.b_count(),
                                indices, tid, vals_output + tile.begin);
                    }
                };
            cta_transform< launch_t >( k, count, context );

            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename input_key_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
    void mergesort( input_key_t** key_blocks, int64_t* block_size_prefix_sum, int block_count, int* indices, bool indices_has_null, key_t* keys_input,
            val_t* vals_input, int count, comp_t< key_t, key_t > comp, context_t& context )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 17 >, arch_35_cta< 128, 11 >, arch_52_cta< MODERN_GPU_PARA, 3 > > >::type_t launch_t;
        int nv = launch_t::nv( context );
        int num_ctas = div_up( count, nv );
        int num_passes = find_log2( num_ctas, true );
        mem_t< key_t > input_buffer;
        if( !keys_input )
            keys_input = input_buffer.alloc( count );
        mem_t< key_t > keys_temp( num_passes ? count : 0 );
        key_t* keys_output = keys_temp.data();

        mem_t< val_t > vals_temp( has_values && num_passes ? count : 0 );
        val_t* vals_output = vals_temp.data();

        key_t* keys_blocksort = ( 1 & num_passes ) ? keys_output : keys_input;
        val_t* vals_blocksort = ( 1 & num_passes ) ? vals_output : vals_input;

        auto k = [=] ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
            typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;

            __shared__ union named_union
            {
                typename sort_t::storage_t sort;
                key_t keys[nv];
                val_t vals[nv];
                named_union()
                {}
            }shared;

            range_t tile = get_tile(cta, nv, count);

            // Load the keys and values.
                kv_array_t<key_t, val_t, vt> unsorted;
                unsorted.keys = mem_to_reg_thread_multi_block_by_indices<nt, vt>(key_blocks, block_size_prefix_sum, block_count, indices + tile.begin, indices_has_null, tid,
                        tile.count(), shared.keys);
                if(has_values)
                unsorted.vals = mem_to_reg_thread<nt, vt>(vals_input + tile.begin, tid,
                        tile.count(), shared.vals);

                // Blocksort.
                kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
                        tid, tile.count(), comp, shared.sort);

                // Store the keys and values.
                reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(),
                        keys_blocksort + tile.begin, shared.keys);
                if(has_values)
                reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(),
                        vals_blocksort + tile.begin, shared.vals);
            };

        cta_transform< launch_t >( k, count, context );

        if( 1 & num_passes )
        {
            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }

        for( int pass = 0; pass < num_passes; ++pass )
        {
            int coop = 2 << pass;
            mem_t< int > partitions = merge_sort_partitions( keys_input, count, coop, nv, comp, context );
            int* mp_data = partitions.data();

            auto k = [=] ARIES_DEVICE(int tid, int cta)
            {
                typedef typename launch_t::sm_ptx params_t;
                enum
                {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

                __shared__ union named_union
                {
                    key_t keys[nv + 1];
                    int indices[nv];
                    named_union()
                    {}
                }shared;

                range_t tile = get_tile(cta, nv, count);

                // Load the range for this CTA and merge the values into register.
                    merge_range_t range = compute_mergesort_range(count, cta, coop, nv,
                            mp_data[cta + 0], mp_data[cta + 1]);

                    merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                            keys_input, keys_input, range, tid, comp, shared.keys);

                    // Store merged values back out.
                    reg_to_mem_thread<nt>(merge.keys, tid, tile.count(),
                            keys_output + tile.begin, shared.keys);

                    if(has_values)
                    {
                        // Transpose the indices from thread order to strided order.
                        array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
                                tid, shared.indices);

                        // Gather the input values and merge into the output values.
                        transfer_two_streams_strided<nt>(vals_input + range.a_begin,
                                range.a_count(), vals_input + range.b_begin, range.b_count(),
                                indices, tid, vals_output + tile.begin);
                    }
                };
            cta_transform< launch_t >( k, count, context );

            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename val_t, typename comp_t >
    void mergesort( char* keys_input, size_t len, val_t* vals_input, int count, comp_t comp, context_t& context )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };

        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 3 > > >::type_t launch_t;

        int nv = launch_t::nv( context );
        int num_ctas = div_up( count, nv );
        int num_passes = find_log2( num_ctas, true );

        mem_t< char > keys_temp( num_passes ? count * len : 0 );
        char* keys_output = keys_temp.data();

        mem_t< val_t > vals_temp( has_values && num_passes ? count : 0 );
        val_t* vals_output = vals_temp.data();

        char* keys_blocksort = ( 1 & num_passes ) ? keys_output : keys_input;
        val_t* vals_blocksort = ( 1 & num_passes ) ? vals_output : vals_input;

        size_t shared_mem_half_size = ( nv + 1 ) * max( len, sizeof(val_t) ); // merge时会对共享内存越界访问１个元素，所以多分配一个元素

        auto k = [=] ARIES_DEVICE(int tid, int cta)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum
            {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};
            typedef cta_sort_t<nt, vt, char, val_t> sort_t;

            extern __shared__ char keys[];

            range_t tile = get_tile(cta, nv, count);

            array_t<val_t, vt> vals;
            if(has_values)
            vals = mem_to_reg<vt>(vals_input + tile.begin, tid,
                    tile.count());

            mem_to_shared<nt, vt>(keys_input + tile.begin * len, len, tid,
                    tile.count(), keys);

            // Blocksort.
                int keysOffset;
                array_t< val_t, vt > sorted = sort_t().block_sort(vals,
                        tid, tile.count(), comp, keys, len, shared_mem_half_size, keysOffset);

                // Store the keys and values.
                shared_to_mem<nt, vt>(keys + keysOffset, len, tid, tile.count(),
                        keys_blocksort + tile.begin * len);

                if(has_values)
                reg_to_mem< vt>(sorted, tid, tile.count(),
                        vals_blocksort + tile.begin);
            };
        cta_transform< launch_t >( k, count, shared_mem_half_size * 2, context );

        if( 1 & num_passes )
        {
            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }

        for( int pass = 0; pass < num_passes; ++pass )
        {
            int coop = 2 << pass;
            mem_t< int > partitions = merge_sort_partitions( keys_input, len, count, coop, nv, comp, context );
            int* mp_data = partitions.data();

            auto k = [=] ARIES_DEVICE(int tid, int cta)
            {
                typedef typename launch_t::sm_ptx params_t;
                enum
                {   nt = params_t::nt, vt = params_t::vt, nv = nt * vt};

                extern __shared__ char keys[];
                range_t tile = get_tile(cta, nv, count);

                // Load the range for this CTA and merge the values into register.
                    merge_range_t range = compute_mergesort_range(count, cta, coop, nv,
                            mp_data[cta + 0], mp_data[cta + 1]);

                    //int keysOffset;
                    array_t< int, vt > merge = cta_merge_from_mem<bounds_lower, nt, vt>(
                            keys_input, keys_input, len, range, tid, comp, keys, keys + shared_mem_half_size );

                    // Store merged values back out.
                    shared_to_mem<nt,vt>( keys + shared_mem_half_size, len, tid, tile.count(), keys_output + tile.begin * len );

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
                        transfer_two_streams_strided<nt>(vals_input + range.a_begin,
                                range.a_count(), vals_input + range.b_begin, range.b_count(),
                                indices, tid, vals_output + tile.begin);
                    }
                    __syncthreads();
                };
            cta_transform< launch_t >( k, count, shared_mem_half_size * 2, context );

            std::swap( keys_input, keys_output );
            std::swap( vals_input, vals_output );
        }
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
    void mergesort_large( key_t* keys_input, val_t* vals_input, int count, comp_t< key_t, key_t > comp, context_t& context )
    {
        //hard code for test
        if( sizeof(key_t) * count > 4 * 300000000 )
        {
            int part1_count = count / 2;
            int part2_count = count - part1_count;
            mem_t< val_t > vals_output( count );

            mergesort( keys_input, vals_input, part1_count, comp, context );
            mergesort( keys_input + part1_count, vals_input + part1_count, part2_count, comp, context );

            int pos = merge_path( keys_input, part1_count, keys_input + part1_count, part2_count, part1_count, comp );
            int part2_half = part1_count - pos;
            mem_t< key_t > keys_output( count );
            merge( keys_input, vals_input, pos, keys_input + part1_count, vals_input + part1_count, part2_half, keys_output.data(),
                    vals_output.data(), comp, context );

            merge( keys_input + pos, vals_input + pos, part1_count - pos, keys_input + part1_count + part2_half,
                    vals_input + part1_count + part2_half, part2_count - part2_half, keys_output.data() + pos + part2_half,
                    vals_output.data() + pos + part2_half, comp, context );
            cudaMemcpy( keys_input, keys_output.data(), count * sizeof(key_t), cudaMemcpyDefault );
            keys_output.free();

            //        merge_vals_only( keys_input, vals_input, pos, keys_input + part1_count, vals_input + part1_count, part2_half,
            //                vals_output.data(), comp, context );
            //
            //        merge_vals_only( keys_input + pos, vals_input + pos, part1_count - pos, keys_input + part1_count + part2_half,
            //                vals_input + part1_count + part2_half, part2_count - part2_half,
            //                vals_output.data() + pos + part2_half, comp, context );

            cudaMemcpy( vals_input, vals_output.data(), count * sizeof(val_t), cudaMemcpyDefault );
        }
        else
        {
            mergesort( keys_input, vals_input, count, comp, context );
        }
    }

    template< typename launch_arg_t = empty_t, typename val_t, typename comp_t >
    void mergesort( char* keys_input, size_t len, val_t* vals_input, int count, comp_t comp, context_t& context, size_t limit_in_mega_bytes )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };
        int device_id = context.active_device_id();
        size_t mem_cost = ( len + sizeof( val_t ) ) * count * 2.5;
        size_t mem_available = limit_in_mega_bytes * 1024 * 1024;
        size_t block_count = 1;
        size_t round_count = 1;
        while( mem_cost > mem_available )
        {
            mem_cost >>= 1;
            block_count <<= 1;
            ++round_count;
        }

        size_t block_size = count / block_count;
        size_t tail_size = count % block_count;

        //分块排序
        size_t block_offset = 0;
        for( size_t i = 0; i < block_count; ++i )
        {
            size_t item_count = ( i == 0 ? block_size + tail_size : block_size );

            cudaMemPrefetchAsync( keys_input + len * block_offset, len * item_count, device_id );
            if( has_values )
                cudaMemPrefetchAsync( vals_input + block_offset, sizeof( val_t ) * item_count, device_id );
            mergesort< launch_arg_t >( keys_input + len * block_offset, len, vals_input + block_offset, item_count, comp, context );
            block_offset += item_count;
        }

        if( block_count > 1 )
        {
            //merge
            managed_mem_t< char > keys( count * len, context, false );
            char* keys_output = keys.data();
            managed_mem_t< val_t > vals( context );
            if( has_values )
                vals.alloc( count, false );
            val_t* vals_output = vals.data();

            for( int round = 1; round < round_count; ++round )
            {
                size_t input_offset = 0;
                size_t output_offset = 0;
                for( int i = 0; i < ( block_count >> round ); ++i )
                {
                    size_t second_sorted_block_size = block_size * ( 1 << ( round - 1 ) );
                    size_t first_sorted_block_size = second_sorted_block_size + ( i == 0 ? tail_size : 0 );
                    
                    mem_t< int > partitions = merge_path_partitions< bounds_lower >( keys_input + len * input_offset, first_sorted_block_size, 
                                                                                    keys_input + len * ( input_offset + first_sorted_block_size ), second_sorted_block_size, 
                                                                                    len, block_size, comp, context );

                    managed_mem_t< merge_range_t > merge_ranges( partitions.size() - 1, context, false ); 
                    merge_range_t *range = merge_ranges.data();
                    for( int p = 0; p < partitions.size() - 1; ++p )
                    {
                        merge_range_t r = compute_merge_range( first_sorted_block_size, second_sorted_block_size, p, block_size, 
                                                            partitions.get_value( p ), partitions.get_value( p + 1 ) );
                        range[ p ] = r;
                    }

                    for( int index = 0; index < merge_ranges.size(); ++index )
                    {
                        merge_range_t r = range[ index ];
                        if( r.a_count() > 0 )
                            cudaMemPrefetchAsync( keys_input + len * ( input_offset + r.a_begin ), len * r.a_count(), device_id );
                        if( r.b_count() > 0 )
                            cudaMemPrefetchAsync( keys_input + len * ( input_offset + first_sorted_block_size + r.b_begin ), len * r.b_count(), device_id );
                        if( has_values )
                        {
                            if( r.a_count() > 0 )
                                cudaMemPrefetchAsync( vals_input + input_offset + r.a_begin, sizeof( val_t ) * r.a_count(), device_id );
                            if( r.b_count() > 0 )
                                cudaMemPrefetchAsync( vals_input + input_offset + first_sorted_block_size + r.b_begin, sizeof( val_t ) * r.b_count(), device_id );
                        }
                        
                        cudaMemPrefetchAsync( keys_output + len* output_offset, len * r.total(), device_id );
                        cudaMemPrefetchAsync( vals_output + output_offset, sizeof( val_t ) * r.total(), device_id );
                        merge< launch_arg_t >( keys_input + len * ( input_offset + r.a_begin ), 
                            vals_input + input_offset + r.a_begin, r.a_count(), 
                            keys_input + len * ( input_offset + first_sorted_block_size + r.b_begin ), 
                            vals_input + input_offset + first_sorted_block_size + r.b_begin, r.b_count(),
                            keys_output + len * output_offset, vals_output + output_offset, len, comp, context );
                        output_offset += r.total();
                    }
                    
                    input_offset += first_sorted_block_size + second_sorted_block_size;
                }
                std::swap( keys_input, keys_output );
                std::swap( vals_input, vals_output );
            }
            if( round_count % 2 == 0 )
            {
                cudaMemcpy( keys_output, keys_input, count * len, cudaMemcpyDefault );
                keys.free();
                if( has_values )
                    cudaMemcpy( vals_output, vals_input, count * sizeof( val_t ), cudaMemcpyDefault );
            }
        }
    }

    template< typename launch_arg_t = empty_t, typename key_t, typename val_t, template< typename, typename > class comp_t >
    void mergesort( key_t* keys_input, val_t* vals_input, size_t count, comp_t< key_t, key_t > comp, context_t& context, size_t limit_in_mega_bytes )
    {
        enum
        {
            has_values = !std::is_same< val_t, empty_t >::value
        };
        int device_id = context.active_device_id();
        size_t mem_cost = ( sizeof( key_t ) + sizeof( val_t ) ) * count * 2.5;
        size_t mem_available = limit_in_mega_bytes * 1024 * 1024;
        size_t block_count = 1;
        size_t round_count = 1;
        while( mem_cost > mem_available )
        {
            mem_cost >>= 1;
            block_count <<= 1;
            ++round_count;
        }

        size_t block_size = count / block_count;
        size_t tail_size = count % block_count;

        //分块排序
        size_t block_offset = 0;
        for( size_t i = 0; i < block_count; ++i )
        {
            size_t item_count = ( i == 0 ? block_size + tail_size : block_size );

            cudaMemPrefetchAsync( keys_input + block_offset, sizeof( key_t ) * item_count, device_id );
            if( has_values )
                cudaMemPrefetchAsync( vals_input + block_offset, sizeof( val_t ) * item_count, device_id );

            mergesort< launch_arg_t >( keys_input + block_offset, vals_input + block_offset, item_count, comp, context );
            block_offset += item_count;
        }

        if( block_count > 1 )
        {
            //merge
            managed_mem_t< key_t > keys( count, context, false );
            key_t* keys_output = keys.data();
            managed_mem_t< val_t > vals( context );
            if( has_values )
                vals.alloc( count, false );
            val_t* vals_output = vals.data();

            for( int round = 1; round < round_count; ++round )
            {
                size_t input_offset = 0;
                size_t output_offset = 0;
                for( int i = 0; i < ( block_count >> round ); ++i )
                {
                    size_t second_sorted_block_size = block_size * ( 1 << ( round - 1 ) );
                    size_t first_sorted_block_size = second_sorted_block_size + ( i == 0 ? tail_size : 0 );
                    
                    mem_t< int > partitions = merge_path_partitions< bounds_lower >( keys_input + input_offset, first_sorted_block_size, 
                                                                                    keys_input + input_offset + first_sorted_block_size, second_sorted_block_size, 
                                                                                    block_size, comp, context );

                    managed_mem_t< merge_range_t > merge_ranges( partitions.size() - 1, context, false ); 
                    merge_range_t *range = merge_ranges.data();
                    for( int p = 0; p < partitions.size() - 1; ++p )
                    {
                        merge_range_t r = compute_merge_range( first_sorted_block_size, second_sorted_block_size, p, block_size, 
                                                            partitions.get_value( p ), partitions.get_value( p + 1 ) );
                        range[ p ] = r;
                    }

                    for( int index = 0; index < merge_ranges.size(); ++index )
                    {
                        merge_range_t r = range[ index ];
                        if( r.a_count() > 0 )
                            cudaMemPrefetchAsync( keys_input + input_offset + r.a_begin, sizeof( key_t ) * r.a_count(), device_id );
                        if( r.b_count() > 0 )
                            cudaMemPrefetchAsync( keys_input + input_offset + first_sorted_block_size + r.b_begin, sizeof( key_t ) * r.b_count(), device_id );
                        if( has_values )
                        {
                            if( r.a_count() > 0 )
                                cudaMemPrefetchAsync( vals_input + input_offset + r.a_begin, sizeof( val_t ) * r.a_count(), device_id );
                            if( r.b_count() > 0 )
                                cudaMemPrefetchAsync( vals_input + input_offset + first_sorted_block_size + r.b_begin, sizeof( val_t ) * r.b_count(), device_id );
                        }
                        
                        cudaMemPrefetchAsync( keys_output + output_offset, sizeof( key_t ) * r.total(), device_id );
                        cudaMemPrefetchAsync( vals_output + output_offset, sizeof( val_t ) * r.total(), device_id );
                        merge< launch_arg_t >( keys_input + input_offset + r.a_begin, 
                            vals_input + input_offset + r.a_begin, r.a_count(), 
                            keys_input + input_offset + first_sorted_block_size + r.b_begin, 
                            vals_input + input_offset + first_sorted_block_size + r.b_begin, r.b_count(),
                            keys_output + output_offset, vals_output + output_offset, comp, context );
                        output_offset += r.total();
                    }
                    
                    input_offset += first_sorted_block_size + second_sorted_block_size;
                }
                std::swap( keys_input, keys_output );
                std::swap( vals_input, vals_output );
            }
            if( round_count % 2 == 0 )
            {
                cudaMemcpy( keys_output, keys_input, count * sizeof( key_t ), cudaMemcpyDefault );
                keys.free();
                if( has_values )
                    cudaMemcpy( vals_output, vals_input, count * sizeof( val_t ), cudaMemcpyDefault );
            }
        }
    }

    // Key-only mergesort
    template< typename launch_arg_t = empty_t, typename key_t, template< typename, typename > class comp_t >
    void mergesort( key_t* keys_input, int count, comp_t< key_t, key_t > comp, context_t& context )
    {
        mergesort< launch_arg_t >( keys_input, ( empty_t* )nullptr, count, comp, context );
    }

    template< typename launch_arg_t = empty_t, typename comp_t >
    void mergesort( char* keys_input, int len, int count, comp_t comp, context_t& context )
    {
        mergesort< launch_arg_t >( keys_input, len, ( empty_t* )nullptr, count, comp, context );
    }

    // Key-only mergesort
    template< typename launch_arg_t = empty_t, typename key_t, template< typename, typename > class comp_t >
    void mergesort( key_t* keys_input, int count, comp_t< key_t, key_t > comp, context_t& context, size_t limit_in_mega_bytes )
    {
        mergesort< launch_arg_t >( keys_input, ( empty_t* )nullptr, count, comp, context, limit_in_mega_bytes );
    }

    template< typename launch_arg_t = empty_t, typename comp_t >
    void mergesort( char* keys_input, int len, int count, comp_t comp, context_t& context, size_t limit_in_mega_bytes )
    {
        mergesort< launch_arg_t >( keys_input, len, ( empty_t* )nullptr, count, comp, context, limit_in_mega_bytes );
    }

END_ARIES_ACC_NAMESPACE
