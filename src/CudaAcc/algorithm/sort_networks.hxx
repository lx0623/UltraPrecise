// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "operators.hxx"

BEGIN_ARIES_ACC_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// Odd-even transposition sorting network. Sorts keys and values in-place in
// register.
// http://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

    template< typename type_t, int vt, typename comp_t >
    ARIES_HOST_DEVICE array_t< type_t, vt > odd_even_sort( array_t< type_t, vt > x, comp_t comp, int flags = 0 )
    {
        iterate< vt >( [&](int I)
        {
            PRAGMA_UNROLL
            for(int i = 1 & I; i < vt - 1; i += 2)
            {
                if((0 == ((2<< i) & flags)) && comp(x[i + 1], x[i]))
                    aries_swap(x[i], x[i + 1]);
            }
        } );
        return x;
    }

//排序后的字符串直接存放于key_shared,函数返回调整顺序后对应的vals
    template< typename val_t, int vt, typename comp_t >
    ARIES_HOST_DEVICE array_t< val_t, vt > odd_even_sort( char* key_shared, int len, array_t< val_t, vt > x, comp_t comp, int flags = 0 )
    {
        iterate< vt >( [&](int I)
        {
            PRAGMA_UNROLL
            for(int i = 1 & I; i < vt - 1; i += 2)
            {
                int offset = i * len;
                if((0 == ((2<< i) & flags)) && comp(&key_shared[offset+len], &key_shared[offset ], len ))
                {
                    aries_swap(&key_shared[offset], &key_shared[offset + len], len );
                    aries_swap(x.data[i], x.data[i + 1]);
                }
            }
        } );
        return x;
    }

    template< typename key_t, typename val_t, int vt, typename comp_t >
    ARIES_HOST_DEVICE kv_array_t< key_t, val_t, vt > odd_even_sort( kv_array_t< key_t, val_t, vt > x, comp_t comp, int flags = 0 )
    {
        iterate< vt >( [&](int I)
        {
            PRAGMA_UNROLL
            for(int i = 1 & I; i < vt - 1; i += 2)
            {
                if((0 == ((2<< i) & flags)) && comp(x.keys[i + 1], x.keys[i]))
                {
                    aries_swap(x.keys[i], x.keys[i + 1]);
                    aries_swap(x.vals[i], x.vals[i + 1]);
                }
            }
        } );
        return x;
    }

END_ARIES_ACC_NAMESPACE
