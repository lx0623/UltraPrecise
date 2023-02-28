// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include <limits>
#include "launch_params.hxx"
#include "context.hxx"
#include "meta.hxx"
#include "operators.hxx"
#include "decimal.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename type_t, int size >
    struct array_t
    {
        type_t data[size];

        ARIES_HOST_DEVICE
        type_t operator[]( int i ) const
        {
            return data[i];
        }
        ARIES_HOST_DEVICE
        type_t& operator[]( int i )
        {
            return data[i];
        }

        ARIES_HOST_DEVICE
        array_t()
        {

        }

        ARIES_HOST_DEVICE
        array_t( const array_t& src )
        {
            //memcpy( &data, &src.data, sizeof( data ) );
            iterate< size >( [&](int i)
            {   data[i] = src.data[i];} );
        }

        ARIES_HOST_DEVICE
        array_t& operator=( const array_t& src )
        {
            //memcpy( &data, &src.data, sizeof( data ) );
            iterate< size >( [&](int i)
            {   data[i] = src.data[i];} );
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE array_t( const array_t< type_u, size >& src )
        {
            //memcpy( &data, &src.data, sizeof( data ) );
            iterate< size >( [&](int i)
            {   data[i] = src.data[i];} );
        }

        template< typename type_u >
        ARIES_HOST_DEVICE array_t& operator=( const array_t< type_u, size >& src )
        {
            //memcpy( &data, &src.data, sizeof( data ) );
            iterate< size >( [&](int i)
            {   data[i] = src.data[i];} );
            return *this;
        }

        // Fill the array with x.
        ARIES_HOST_DEVICE
        array_t( type_t x )
        {
            iterate< size >( [&](int i)
            {   data[i] = x;} );
        }
    };

    template< typename type_t >
    struct array_t< type_t, 0 >
    {
        ARIES_HOST_DEVICE
        type_t operator[]( int i ) const
        {
            return type_t();
        }
        ARIES_HOST_DEVICE
        type_t& operator[]( int i )
        {
            return *( type_t* )nullptr;
        }
    };

// Reduce on components of array_t.
    template< typename type_t, int size, typename op_t = plus_t< type_t > >
    ARIES_HOST_DEVICE type_t reduce( array_t< type_t, size > x, op_t op = op_t() )
    {
        type_t a;
        iterate< size >( [&](int i)
        {
            a = i ? op(a, x[i]) : x[i];
        } );
        return a;
    }

// Call the operator component-wise on all components.
    template< typename type_t, int size, typename op_t >
    ARIES_HOST_DEVICE array_t< type_t, size > combine( array_t< type_t, size > x, array_t< type_t, size > y, op_t op )
    {

        array_t< type_t, size > z;
        iterate< size >( [&](int i)
        {   z[i] = op(x[i], y[i]);} );
        return z;
    }

    template< typename type_t, int size >
    ARIES_HOST_DEVICE array_t< type_t, size > operator+( array_t< type_t, size > a, array_t< type_t, size > b )
    {
        return combine( a, b, plus_t< type_t >() );
    }

    template< typename type_t, int size >
    ARIES_HOST_DEVICE array_t< type_t, size > operator-( array_t< type_t, size > a, array_t< type_t, size > b )
    {
        return combine( a, b, minus_t< type_t >() );
    }

    template< typename key_t, typename val_t, int size >
    struct kv_array_t
    {
        array_t< key_t, size > keys;
        array_t< val_t, size > vals;
    };

    struct ARIES_ALIGN(8) range_t
    {
        int begin;
        int end;

        ARIES_HOST_DEVICE
        int size() const
        {
            return end - begin;
        }
        ARIES_HOST_DEVICE
        int count() const
        {
            return size();
        }
        ARIES_HOST_DEVICE
        bool valid() const
        {
            return end > begin;
        }
    };

    struct ARIES_ALIGN(16) range_t_64
    {
        int64_t begin;
        int64_t end;

        ARIES_HOST_DEVICE
        int64_t size() const
        {
            return end - begin;
        }
        ARIES_HOST_DEVICE
        int64_t count() const
        {
            return size();
        }
        ARIES_HOST_DEVICE
        bool valid() const
        {
            return end > begin;
        }
    };

    ARIES_HOST_DEVICE range_t get_tile( int cta, int nv, int count )
    {
        return range_t
        { nv * cta, min( count, nv * ( cta + 1 ) ) };
    }

    struct ARIES_ALIGN(16) merge_range_t
    {
        int a_begin, a_end, b_begin, b_end;

        ARIES_HOST_DEVICE
        int a_count() const
        {
            return a_end - a_begin;
        }
        ARIES_HOST_DEVICE
        int b_count() const
        {
            return b_end - b_begin;
        }
        ARIES_HOST_DEVICE
        int total() const
        {
            return a_count() + b_count();
        }

        ARIES_HOST_DEVICE
        range_t a_range() const
        {
            return range_t
            { a_begin, a_end };
        }
        ARIES_HOST_DEVICE
        range_t b_range() const
        {
            return range_t
            { b_begin, b_end };
        }

        ARIES_HOST_DEVICE
        merge_range_t to_local() const
        {
            return merge_range_t
            { 0, a_count(), a_count(), total() };
        }

        // Partition from mp to the end.
        ARIES_HOST_DEVICE
        merge_range_t partition( int mp0, int diag ) const
        {
            return merge_range_t
            { a_begin + mp0, a_end, b_begin + diag - mp0, b_end };
        }

        // Partition from mp0 to mp1.
        ARIES_HOST_DEVICE
        merge_range_t partition( int mp0, int diag0, int mp1, int diag1 ) const
        {
            return merge_range_t
            { a_begin + mp0, a_begin + mp1, b_begin + diag0 - mp0, b_begin + diag1 - mp1 };
        }

        ARIES_HOST_DEVICE
        bool a_valid() const
        {
            return a_begin < a_end;
        }
        ARIES_HOST_DEVICE
        bool b_valid() const
        {
            return b_begin < b_end;
        }
    };

    struct ARIES_ALIGN(32) merge_range_t_64
    {
        int64_t a_begin, a_end, b_begin, b_end;

        ARIES_HOST_DEVICE
        int64_t a_count() const
        {
            return a_end - a_begin;
        }
        ARIES_HOST_DEVICE
        int64_t b_count() const
        {
            return b_end - b_begin;
        }
        ARIES_HOST_DEVICE
        int64_t total() const
        {
            return a_count() + b_count();
        }

        ARIES_HOST_DEVICE
        range_t_64 a_range() const
        {
            return range_t_64
            { a_begin, a_end };
        }
        ARIES_HOST_DEVICE
        range_t_64 b_range() const
        {
            return range_t_64
            { b_begin, b_end };
        }

        ARIES_HOST_DEVICE
        merge_range_t_64 to_local() const
        {
            return merge_range_t_64
            { 0, a_count(), a_count(), total() };
        }

        // Partition from mp to the end.
        ARIES_HOST_DEVICE
        merge_range_t_64 partition( int64_t mp0, int64_t diag ) const
        {
            return merge_range_t_64
            { a_begin + mp0, a_end, b_begin + diag - mp0, b_end };
        }

        // Partition from mp0 to mp1.
        ARIES_HOST_DEVICE
        merge_range_t_64 partition( int64_t mp0, int64_t diag0, int64_t mp1, int64_t diag1 ) const
        {
            return merge_range_t_64
            { a_begin + mp0, a_begin + mp1, b_begin + diag0 - mp0, b_begin + diag1 - mp1 };
        }

        ARIES_HOST_DEVICE
        bool a_valid() const
        {
            return a_begin < a_end;
        }
        ARIES_HOST_DEVICE
        bool b_valid() const
        {
            return b_begin < b_end;
        }
    };

    template< typename type_t, int size >
    struct merge_pair_t
    {
        array_t< type_t, size > keys;
        array_t< int, size > indices;
    };

    template< typename type_t, typename indices_t, int size >
    struct merge_pair_t_with_match_tag
    {
        array_t< type_t, size > keys;
        array_t< indices_t, size > indices;
    };

    template< typename type_t >
    struct join_pair_t
    {
        mem_t< type_t > left_indices;
        mem_t< type_t > right_indices;
        size_t count;
    };

    struct DataBlockInfo
    {
        int8_t* Data;
        int32_t ElementSize;
        int32_t Offset; // 1 for not null -> nullable, otherwise 0
    };

    struct SimpleDataBlockInfo
    {
        int8_t* Data;
        int32_t ElementSize;
    };

END_ARIES_ACC_NAMESPACE
