// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "loadstore.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< template< typename, typename > class comp_t, bounds_t bounds = bounds_lower, typename a_keys_it, typename b_keys_it, typename int_t >
    ARIES_HOST_DEVICE int_t merge_path( a_keys_it a_keys, int_t a_count, b_keys_it b_keys, int_t b_count, int_t diag )
    {
        typedef typename std::iterator_traits< a_keys_it >::value_type type_a;
        typedef typename std::iterator_traits< b_keys_it >::value_type type_b;
        int_t begin = max( ( int_t )0, diag - b_count );
        int_t end = min( diag, a_count );

        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            type_a a_key = a_keys[mid];
            type_b b_key = b_keys[diag - 1 - mid];
            bool pred = ( bounds_upper == bounds ) ? comp_t< type_a, type_b >()( a_key, b_key ) : !comp_t< type_b, type_a >()( b_key, a_key );

            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< bounds_t bounds = bounds_lower, typename a_keys_it, typename b_keys_it, typename int_t, typename comp_t >
    ARIES_HOST_DEVICE int_t merge_path( a_keys_it a_keys, int_t a_count, b_keys_it b_keys, int_t b_count, int_t diag, comp_t comp )
    {
        typedef typename std::iterator_traits< a_keys_it >::value_type type_a;
        typedef typename std::iterator_traits< b_keys_it >::value_type type_b;
        int_t begin = max( ( int_t )0, diag - b_count );
        int_t end = min( diag, a_count );

        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            type_a a_key = a_keys[mid];
            type_b b_key = b_keys[diag - 1 - mid];
            bool pred = ( bounds_upper == bounds ) ? comp( a_key, b_key ) : !comp( b_key, a_key );

            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< bounds_t bounds = bounds_lower, typename int_t, typename comp_t >
    ARIES_HOST_DEVICE int_t merge_path( const char* a_keys, int_t a_count, const char* b_keys, int_t b_count, size_t len, int_t diag, comp_t comp )
    {

        int_t begin = max( ( int_t )0, diag - b_count );
        int_t end = min( diag, a_count );

        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            const char* a_key = &a_keys[mid * len];
            const char* b_key = &b_keys[( diag - 1 - mid ) * len];
            bool pred = ( bounds_upper == bounds ) ? comp( a_key, b_key, len ) : !comp( b_key, a_key, len );

            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< template< typename, typename > class comp_t, bounds_t bounds, typename keys_it >
    ARIES_HOST_DEVICE int merge_path( keys_it keys, merge_range_t range, int diag )
    {
        return merge_path< comp_t, bounds >( keys + range.a_begin, range.a_count(), keys + range.b_begin, range.b_count(), diag );
    }

    template< bounds_t bounds, typename keys_it, typename comp_t >
    ARIES_HOST_DEVICE int merge_path( keys_it keys, merge_range_t range, int diag, comp_t comp )
    {
        return merge_path< bounds >( keys + range.a_begin, range.a_count(), keys + range.b_begin, range.b_count(), diag, comp );
    }

    template< bounds_t bounds, typename comp_t >
    ARIES_HOST_DEVICE int merge_path( const char* keys, size_t len, merge_range_t range, int diag, comp_t comp )
    {
        return merge_path< bounds >( keys + range.a_begin * len, range.a_count(), keys + range.b_begin * len, range.b_count(), len, diag, comp );
    }

    template< bounds_t bounds, bool range_check, typename type_t, typename type_u, typename comp_t >
    ARIES_HOST_DEVICE bool merge_predicate( type_t a_key, type_u b_key, merge_range_t range, comp_t comp )
    {
        bool p;
        if( range_check && !range.a_valid() )
            p = false;
        else if( range_check && !range.b_valid() )
            p = true;
        else
            p = ( bounds_upper == bounds ) ? comp( a_key, b_key ) : !comp( b_key, a_key );
        return p;
    }

    template< template< typename, typename > class comp_t, bounds_t bounds, bool range_check, typename type_t, typename type_u >
    ARIES_HOST_DEVICE bool merge_predicate( type_t a_key, type_u b_key, merge_range_t range )
    {
        bool p;
        if( range_check && !range.a_valid() )
            p = false;
        else if( range_check && !range.b_valid() )
            p = true;
        else
            p = ( bounds_upper == bounds ) ? comp_t< type_t, type_u >()( a_key, b_key ) : !comp_t< type_u, type_t >()( b_key, a_key );
        return p;
    }

    template< bounds_t bounds, bool range_check, typename comp_t >
    ARIES_HOST_DEVICE bool merge_predicate( const char* a_key, const char* b_key, size_t len, merge_range_t range, comp_t comp )
    {

        bool p;
        if( range_check && !range.a_valid() )
            p = false;
        else if( range_check && !range.b_valid() )
            p = true;
        else
            p = ( bounds_upper == bounds ) ? comp( a_key, b_key, len ) : !comp( b_key, a_key, len );
        return p;
    }

    ARIES_HOST_DEVICE merge_range_t compute_merge_range( int a_count, int b_count, int partition, int spacing, int mp0, int mp1 )
    {

        int diag0 = spacing * partition;
        int diag1 = min( a_count + b_count, diag0 + spacing );

        return merge_range_t
        { mp0, mp1, diag0 - mp0, diag1 - mp1 };
    }

    ARIES_HOST_DEVICE merge_range_t_64 compute_merge_range_64( int64_t a_count, int64_t b_count, int64_t partition, int64_t spacing, int64_t mp0,
            int64_t mp1 )
    {

        int64_t diag0 = spacing * partition;
        int64_t diag1 = min( a_count + b_count, diag0 + spacing );

        return merge_range_t_64
        { mp0, mp1, diag0 - mp0, diag1 - mp1 };
    }

// Specialization that emits just one LD instruction. Can only reliably used
// with raw pointer types. Fixed not to use pointer arithmetic so that 
// we don't get undefined behaviors with unaligned types.

    template< int nt, int vt, typename type_t >
    ARIES_DEVICE array_t< type_t, vt > load_two_streams_reg( const type_t* a, int a_count, const type_t* b, int b_count, int tid )
    {

        b -= a_count;
        array_t< type_t, vt > x;
        strided_iterate< nt, vt >( [&](int i, int index)
        {
            const type_t* p = (index >= a_count) ? b : a;
            x[i] = p[index];
        }, tid, a_count + b_count );

        return x;
    }

    template< int nt, int vt, typename type_t, typename a_it, typename b_it >
    ARIES_DEVICE array_t< type_t, vt > load_two_streams_reg( a_it a, int a_count, b_it b, int b_count, int tid )
    {
        b -= a_count;
        array_t< type_t, vt > x;
        strided_iterate< nt, vt >( [&](int i, int index)
        {
            x[i] = (index < a_count) ? (type_t)a[index] : (type_t)b[index];
        }, tid, a_count + b_count );
        return x;
    }

    template< int nt, int vt, typename a_it, typename b_it, typename type_t, int shared_size >
    ARIES_DEVICE void load_two_streams_shared( a_it a, int a_count, b_it b, int b_count, int tid, type_t (&shared)[shared_size], bool sync = true )
    {

        // Load into register then make an unconditional strided store into memory.
        array_t< type_t, vt > x = load_two_streams_reg< nt, vt, type_t >( a, a_count, b, b_count, tid );
        reg_to_shared_strided< nt >( x, tid, shared, sync );
    }

    template< int nt, int vt >
    ARIES_DEVICE void load_two_streams_shared( const char* a, int a_count, const char* b, int b_count, size_t len, int tid, char* shared, bool sync =
            true )
    {
        b -= a_count * len;
        const char* p;
        strided_iterate< nt, vt >( [&, len](int i, int j)
        {
            p = (j >= a_count) ? b : a;
            memcpy( shared + j * len, p + j * len, len );
        }, tid, a_count + b_count );
        __syncthreads();
    }

    template< int nt, int vt, typename type_t >
    ARIES_DEVICE array_t< type_t, vt > gather_two_streams_strided( const type_t* a, int a_count, const type_t* b, int b_count,
            array_t< int, vt > indices, int tid )
    {

        ptrdiff_t b_offset = b - a - a_count;
        int count = a_count + b_count;

        array_t< type_t, vt > x;
        strided_iterate< nt, vt >( [&](int i, int j)
        {
            ptrdiff_t gather = indices[i];
            if(gather >= a_count) gather += b_offset;
            x[i] = a[gather];
        }, tid, count );

        return x;
    }
    template< int nt, int vt, typename type_t, typename a_it, typename b_it >
    ARIES_DEVICE
    enable_if_t< !( std::is_pointer< a_it >::value && std::is_pointer< b_it >::value ), array_t< type_t, vt > > gather_two_streams_strided( a_it a,
            int a_count, b_it b, int b_count, array_t< int, vt > indices, int tid )
    {

        b -= a_count;
        array_t< type_t, vt > x;
        strided_iterate< nt, vt >( [&](int i, int j)
        {
            x[i] = (indices[i] < a_count) ? a[indices[i]] : b[indices[i]];
        }, tid, a_count + b_count );

        return x;
    }

    template< int nt, int vt, typename a_it, typename b_it, typename c_it >
    ARIES_DEVICE void transfer_two_streams_strided( a_it a, int a_count, b_it b, int b_count, array_t< int, vt > indices, int tid, c_it c )
    {

        typedef typename std::iterator_traits< a_it >::value_type type_t;
        array_t< type_t, vt > x = gather_two_streams_strided< nt, vt, type_t >( a, a_count, b, b_count, indices, tid );

        reg_to_mem_strided< nt >( x, tid, a_count + b_count, c );
    }

// This function must be able to dereference keys[a_begin] and keys[b_begin],
// no matter the indices for each. The caller should allocate at least 
// nt * vt + 1 elements for 
    template< bounds_t bounds, int vt, typename type_t, typename comp_t >
    ARIES_DEVICE merge_pair_t< type_t, vt > serial_merge( const type_t* keys_shared, merge_range_t range, comp_t comp, bool sync = true )
    {
        type_t a_key = keys_shared[range.a_begin];
        type_t b_key = keys_shared[range.b_begin];

        merge_pair_t< type_t, vt > merge_pair;
        iterate< vt >( [&](int i)
        {
            bool p = merge_predicate<bounds, true>(a_key, b_key,range, comp);
            int index = p ? range.a_begin : range.b_begin;

            merge_pair.keys[i] = p ? a_key : b_key;
            merge_pair.indices[i] = index;

            type_t c_key = keys_shared[++index];
            if(p) a_key = c_key, range.a_begin = index;
            else b_key = c_key, range.b_begin = index;
        } );

        if( sync )
            __syncthreads();
        return merge_pair;
    }

    template< template< typename, typename > class comp_t, bounds_t bounds, int vt, typename type_t >
    ARIES_DEVICE array_t< int, vt > serial_merge_indices( const type_t* keys_shared, merge_range_t range, bool sync = true )
    {
        type_t a_key = keys_shared[range.a_begin];
        type_t b_key = keys_shared[range.b_begin];

        array_t< int, vt > merge_indices;
        iterate< vt >( [&](int i)
        {
            bool p = merge_predicate<comp_t, bounds, true>(a_key, b_key,range);
            int index = p ? range.a_begin : range.b_begin;
            merge_indices[i] = index;

            type_t c_key = keys_shared[++index];
            if(p) a_key = c_key, range.a_begin = index;
            else b_key = c_key, range.b_begin = index;
        } );

        if( sync )
            __syncthreads();
        return merge_indices;
    }

    template< bounds_t bounds, int vt, typename indices_t, typename type_t, typename comp_t >
    ARIES_DEVICE array_t< indices_t, vt > serial_merge_with_match_tag( const type_t* keys_shared, merge_range_t range, comp_t comp, bool sync = true )
    {
        type_t a_key = keys_shared[range.a_begin];
        type_t b_key = keys_shared[range.b_begin];

        array_t< indices_t, vt > merge_indices;
        iterate< vt >( [&](int i)
        {
            bool p = merge_predicate<bounds, true>(a_key, b_key,range, comp);
            int index = p ? range.a_begin : range.b_begin;
            merge_indices[i].value = index;
            merge_indices[i].flag = !comp(a_key,b_key) && !comp( b_key, a_key );

            type_t c_key = keys_shared[++index];
            if(p) a_key = c_key, range.a_begin = index;
            else b_key = c_key, range.b_begin = index;
        } );

        if( sync )
            __syncthreads();
        return merge_indices;
    }

    template< bounds_t bounds, int vt, typename comp_t >
    ARIES_DEVICE array_t< int, vt > serial_merge( const char* keys_source, size_t len, char* keys_dest, int tid, merge_range_t range, comp_t comp,
            bool sync = true )
    {
        const char* a_key = &keys_source[range.a_begin * len];
        const char* b_key = &keys_source[range.b_begin * len];
        const char* c_key;
        array_t< int, vt > merge_indices;
        iterate< vt >( [&](int i)
        {
            bool p = merge_predicate<bounds, true>(a_key, b_key, len, range, comp);
            int index = p ? range.a_begin : range.b_begin;
            c_key = p ? a_key : b_key;
            memcpy( keys_dest + ( vt * tid + i) * len, c_key, len );
            merge_indices[i] = index;

            c_key = &keys_source[(++index) * len];
            if(p) a_key = c_key, range.a_begin = index;
            else b_key = c_key, range.b_begin = index;
        } );

        if( sync )
            __syncthreads();
        return merge_indices;
    }

    template< bounds_t bounds, int vt, typename indices_t, typename comp_t >
    ARIES_DEVICE array_t< indices_t, vt > serial_merge_with_match_tag( const char* keys_source, size_t len, int tid, merge_range_t range, comp_t comp,
            bool sync = true )
    {
        const char* a_key = &keys_source[range.a_begin * len];
        const char* b_key = &keys_source[range.b_begin * len];
        const char* c_key;
        array_t< indices_t, vt > merge_indices;
        iterate< vt >( [&](int i)
        {
            bool p = merge_predicate<bounds, true>(a_key, b_key, len, range, comp);
            int index = p ? range.a_begin : range.b_begin;
            merge_indices[i].value = index;
            merge_indices[i].flag = !comp(a_key,b_key, len) && !comp(b_key,a_key, len);

            c_key = &keys_source[(++index) * len];
            if(p) a_key = c_key, range.a_begin = index;
            else b_key = c_key, range.b_begin = index;
        } );

        if( sync )
            __syncthreads();
        return merge_indices;
    }

    // Load arrays a and b from global memory and merge into register.
    template< bounds_t bounds, int nt, int vt, typename a_it, typename b_it, typename type_t, typename comp_t, int shared_size >
    ARIES_DEVICE merge_pair_t< type_t, vt > cta_merge_from_mem( a_it a, b_it b, merge_range_t range_mem, int tid, comp_t comp,
            type_t (&keys_shared)[shared_size] )
    {

        static_assert(shared_size >= nt * vt + 1,
                "cta_merge_from_mem requires temporary storage of at "
                "least nt * vt + 1 items");

        // Load the data into shared memory.
        load_two_streams_shared< nt, vt >( a + range_mem.a_begin, range_mem.a_count(), b + range_mem.b_begin, range_mem.b_count(), tid, keys_shared,
                true );

        // Run a merge path to find the start of the serial merge for each thread.
        merge_range_t range_local = range_mem.to_local();
        int diag = vt * tid;
        int mp = merge_path< bounds >( keys_shared, range_local, diag, comp );

        // Compute the ranges of the sources in shared memory. The end iterators
        // of the range are inaccurate, but still facilitate exact merging, because
        // only vt elements will be merged.
        merge_pair_t< type_t, vt > merged = serial_merge< bounds, vt >( keys_shared, range_local.partition( mp, diag ), comp );

        return merged;
    }

    template< template< typename, typename > class comp_t, bounds_t bounds, int nt, int vt, typename a_it, typename b_it, typename type_t, int shared_size >
    ARIES_DEVICE array_t< int, vt > cta_merge_indices_from_mem( a_it a, b_it b, merge_range_t range_mem, int tid, type_t (&keys_shared)[shared_size] )
    {

        static_assert(shared_size >= nt * vt + 1,
                "cta_merge_from_mem requires temporary storage of at "
                "least nt * vt + 1 items");

        // Load the data into shared memory.
        load_two_streams_shared< nt, vt >( a + range_mem.a_begin, range_mem.a_count(), b + range_mem.b_begin, range_mem.b_count(), tid, keys_shared,
                true );

        // Run a merge path to find the start of the serial merge for each thread.
        merge_range_t range_local = range_mem.to_local();
        int diag = vt * tid;
        int mp = merge_path< comp_t, bounds >( keys_shared, range_local, diag );

        // Compute the ranges of the sources in shared memory. The end iterators
        // of the range are inaccurate, but still facilitate exact merging, because
        // only vt elements will be merged.
        array_t< int, vt > merged = serial_merge_indices< comp_t, bounds, vt >( keys_shared, range_local.partition( mp, diag ) );

        return merged;
    }

    template< bounds_t bounds, int nt, int vt, typename indices_t, typename a_it, typename b_it, typename type_t, typename comp_t, int shared_size >
    ARIES_DEVICE array_t< indices_t, vt > cta_merge_from_mem_with_match_tag( a_it a, b_it b, merge_range_t range_mem, int tid, comp_t comp,
            type_t (&keys_shared)[shared_size] )
    {

        static_assert(shared_size >= nt * vt + 1,
                "cta_merge_from_mem requires temporary storage of at "
                "least nt * vt + 1 items");

        // Load the data into shared memory.
        load_two_streams_shared< nt, vt >( a + range_mem.a_begin, range_mem.a_count(), b + range_mem.b_begin, range_mem.b_count(), tid, keys_shared,
                true );

        // Run a merge path to find the start of the serial merge for each thread.
        merge_range_t range_local = range_mem.to_local();
        int diag = vt * tid;
        int mp = merge_path< bounds >( keys_shared, range_local, diag, comp );

        // Compute the ranges of the sources in shared memory. The end iterators
        // of the range are inaccurate, but still facilitate exact merging, because
        // only vt elements will be merged.
        array_t< indices_t, vt > merged = serial_merge_with_match_tag< bounds, vt, indices_t >( keys_shared, range_local.partition( mp, diag ),
                comp );

        return merged;
    }

    template< bounds_t bounds, int nt, int vt, typename comp_t >
    ARIES_DEVICE array_t< int, vt > cta_merge_from_mem( const char* a, const char* b, size_t len, merge_range_t range_mem, int tid, comp_t comp,
            char* keys_source, char* keys_dest )
    {

        // Load the data into shared memory.
        load_two_streams_shared< nt, vt >( a + range_mem.a_begin * len, range_mem.a_count(), b + range_mem.b_begin * len, range_mem.b_count(), len,
                tid, keys_source, true );

        // Run a merge path to find the start of the serial merge for each thread.
        merge_range_t range_local = range_mem.to_local();
        int diag = vt * tid;
        int mp = merge_path< bounds >( keys_source, len, range_local, diag, comp );

        // Compute the ranges of the sources in shared memory. The end iterators
        // of the range are inaccurate, but still facilitate exact merging, because
        // only vt elements will be merged.
        array_t< int, vt > merged = serial_merge< bounds, vt >( keys_source, len, keys_dest, tid, range_local.partition( mp, diag ), comp );

        return merged;
    }

    template< bounds_t bounds, int nt, int vt, typename indices_t, typename comp_t >
    ARIES_DEVICE array_t< indices_t, vt > cta_merge_from_mem_with_match_tag( const char* a, const char* b, size_t len, merge_range_t range_mem, int tid,
            comp_t comp, char* keys_source )
    {

        // Load the data into shared memory.
        load_two_streams_shared< nt, vt >( a + range_mem.a_begin * len, range_mem.a_count(), b + range_mem.b_begin * len, range_mem.b_count(), len,
                tid, keys_source, true );

        // Run a merge path to find the start of the serial merge for each thread.
        merge_range_t range_local = range_mem.to_local();
        int diag = vt * tid;
        int mp = merge_path< bounds >( keys_source, len, range_local, diag, comp );

        // Compute the ranges of the sources in shared memory. The end iterators
        // of the range are inaccurate, but still facilitate exact merging, because
        // only vt elements will be merged.
        array_t< indices_t, vt > merged = serial_merge_with_match_tag< bounds, vt, indices_t >( keys_source, len, tid,
                range_local.partition( mp, diag ), comp );

        return merged;
    }

END_ARIES_ACC_NAMESPACE
