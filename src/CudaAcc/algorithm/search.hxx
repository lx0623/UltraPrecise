// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "loadstore.hxx"
#include "operators.hxx"
#include "cta_search.hxx"
#include "memory.hxx"
#include "launch_params.hxx"
#include "context.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< template< typename, typename > class comp_t, bounds_t bounds, typename a_keys_it, typename b_keys_it >
    mem_t< int > merge_path_partitions( a_keys_it a, int64_t a_count, b_keys_it b, int64_t b_count, int64_t spacing, context_t& context )
    {
        typedef int int_t;
        int num_partitions = ( int )div_up( a_count + b_count, spacing ) + 1;
        mem_t< int_t > mem( num_partitions );
        int_t* p = mem.data();
        context.synchronize();
        transform([=]ARIES_DEVICE(int index)
                {
                    int64_t diag = min(spacing * index, a_count + b_count);
                    p[index] = merge_path<comp_t, bounds>(a, a_count, b, b_count,
                            diag);
                }, num_partitions, context);
        context.synchronize();
        return mem;
    }

    template< bounds_t bounds, typename a_keys_it, typename b_keys_it, typename comp_t >
    mem_t< int > merge_path_partitions( a_keys_it a, int64_t a_count, b_keys_it b, int64_t b_count, int64_t spacing, comp_t comp, context_t& context )
    {
        typedef int int_t;
        int num_partitions = ( int )div_up( a_count + b_count, spacing ) + 1;
        mem_t< int_t > mem( num_partitions );
        int_t* p = mem.data();
        context.synchronize();
        transform([=]ARIES_DEVICE(int index)
                {
                    int64_t diag = min(spacing * index, a_count + b_count);
                    p[index] = merge_path<bounds>(a, a_count, b, b_count,
                            diag, comp);
                }, num_partitions, context);
        context.synchronize();
        return mem;
    }

    template< bounds_t bounds, typename a_keys_it, typename b_keys_it, typename comp_t >
    mem_t< int64_t > merge_path_partitions_64( a_keys_it a, int64_t a_count, b_keys_it b, int64_t b_count, int64_t spacing, comp_t comp, context_t& context )
    {
        typedef int64_t int_t;
        int_t num_partitions = ( int_t )div_up( a_count + b_count, spacing ) + 1;
        mem_t< int_t > mem( num_partitions );
        int_t* p = mem.data();
        context.synchronize();
        transform([=]ARIES_DEVICE(int_t index)
                {
                    int64_t diag = min(spacing * index, a_count + b_count);
                    p[index] = merge_path<bounds>(a, a_count, b, b_count,
                            diag, comp);
                }, num_partitions, context);
        context.synchronize();
        return mem;
    }

    template< bounds_t bounds, typename comp_t >
    mem_t< int > merge_path_partitions( const char* a, int64_t a_count, const char* b, int64_t b_count, size_t len, int64_t spacing, comp_t comp,
            context_t& context )
    {
        typedef int int_t;
        int num_partitions = ( int )div_up( a_count + b_count, spacing ) + 1;
        mem_t< int_t > mem( num_partitions );
        int_t* p = mem.data();
        transform([=]ARIES_DEVICE(int index)
                {
                    int64_t diag = min(spacing * index, a_count + b_count);
                    p[index] = merge_path<bounds>(a, a_count, b, b_count, len,
                            diag, comp);
                }, num_partitions, context);
        context.synchronize();
        return mem;
    }

    template< typename segments_it >
    auto load_balance_partitions( int64_t dest_count, segments_it segments, int num_segments, int spacing, context_t& context ) ->
    mem_t<typename std::iterator_traits<segments_it>::value_type>
    {
        typedef typename std::iterator_traits< segments_it >::value_type int_t;
        return merge_path_partitions< bounds_upper >( counting_iterator_t< int_t >( 0 ), dest_count, segments, num_segments, spacing,
                less_t< int_t >(), context );
    }

    template< typename segments_it >
    auto load_balance_partitions_64( int64_t dest_count, segments_it segments, int64_t num_segments, int64_t spacing, context_t& context ) ->
    mem_t<typename std::iterator_traits<segments_it>::value_type>
    {
        typedef typename std::iterator_traits< segments_it >::value_type int_t;
        return merge_path_partitions_64< bounds_upper >( counting_iterator_t< int_t, int_t >( 0 ), dest_count, segments, num_segments, spacing,
                less_t< int_t >(), context );
    }

    template< bounds_t bounds, typename keys_it >
    mem_t< int > binary_search_partitions( keys_it keys, int count, int num_items, int spacing, context_t& context )
    {

        int num_partitions = div_up( count, spacing ) + 1;
        mem_t< int > mem( num_partitions );
        int* p = mem.data();
        transform([=]ARIES_DEVICE(int index)
                {
                    int key = min(spacing * index, count);
                    p[index] = binary_search<bounds>(keys, num_items, key, less_t<int>());
                }, num_partitions, context);
        return mem;
    }

END_ARIES_ACC_NAMESPACE
