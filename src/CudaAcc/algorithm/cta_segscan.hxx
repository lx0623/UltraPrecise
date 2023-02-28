// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_scan.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename type_t >
    struct segscan_result_t
    {
        type_t scan;
        type_t reduction;
        bool has_carry_in;
        int left_lane;
    };

    template< int nt, typename type_t, int tpi = 1 >
    struct cta_segscan_t
    {
        enum
        {
            num_warps = nt / warp_size
        };

        union storage_t
        {
            storage_t(){}
            int delta[ num_warps + (nt/tpi) ];
            struct
            {
                type_t values[ 2 * (nt/tpi) ];
                int packed[ (nt/tpi) ];
            };
        };

        ARIES_DEVICE int find_left_lane( int tid, bool has_head_flag, storage_t& storage ) const
        {

            int warp = tid / warp_size;
            int lane = ( warp_size - 1 ) & tid;
            int warp_mask = 0xffffffff >> ( 31 - lane );   // inclusive search.
            int cta_mask = 0x7fffffff >> ( 31 - lane );    // exclusive search.

            // Build a head flag bitfield and store it into shared memory.
            int warp_bits = __ballot_sync( __activemask(), has_head_flag );
            storage.delta[ warp ] = warp_bits;
            __syncthreads();

            if( tid < num_warps )
            {
                int cta_bits = __ballot_sync( __activemask(), 0 != storage.delta[ tid ] );
                int warp_segment = 31 - clz( cta_mask & cta_bits );
                int start = ( -1 != warp_segment ) ? ( 31 - clz( storage.delta[ warp_segment ] ) + 32 * warp_segment ) : 0;
                storage.delta[ num_warps + tid ] = start;
            }
            __syncthreads();

            // Find the closest flag to the left of this thread within the warp.
            // Include the flag for this thread.
            int start = 31 - clz( warp_mask & warp_bits );
            if( -1 != start )
                start += ~31 & tid;
            else
                start = storage.delta[ num_warps + warp ];
            __syncthreads();

            return start;
        }

        template< typename op_t = plus_t< type_t > >
        ARIES_DEVICE segscan_result_t< type_t > segscan( int tid, bool has_head_flag, bool has_carry_out, type_t x, storage_t& storage, type_t init =
                type_t(), op_t op = op_t() ) const
        {

            if( !has_carry_out )
                x = init;

            int left_lane = find_left_lane( tid, has_head_flag, storage );
            int tid_delta = tid - left_lane;

            // Store the has_carry_out flag.
            storage.packed[ tid ] = ( int )has_carry_out | ( left_lane << 1 );

            // Run an inclusive scan.
            int first = 0;
            storage.values[ first + tid ] = x;
            __syncthreads();

            int packed = storage.packed[ left_lane ];
            left_lane = packed >> 1;
            tid_delta = tid - left_lane;
            if( 0 == ( 1 & packed ) )
                --tid_delta;

            iterate< s_log2( nt ) >( [&](int pass)
            {
                int offset = 1<< pass;
                if(tid_delta >= offset)
                x = op(x, storage.values[first + tid - offset]);
                first = nt - first;
                storage.values[first + tid] = x;
                __syncthreads();
            } );

            // Get the exclusive scan by fetching the preceding element. Also return
            // the carry-out value as the total.
            bool has_carry_in = tid ? ( 0 != ( 1 & storage.packed[ tid - 1 ] ) ) : false;

            segscan_result_t< type_t > result { ( has_carry_in && tid ) ? storage.values[ first + tid - 1 ] : init, storage.values[ first + nt - 1 ],
                    has_carry_in, left_lane };
            __syncthreads();

            return result;
        }

         template< typename op_t = aries_acc::agg_sum_t< aries_acc::nullable_type<aries_acc::Decimal>> >
        ARIES_DEVICE segscan_result_t< type_t > segscan( int tid, bool has_head_flag, bool has_carry_out, uint32_t* x, storage_t& storage, type_t init =
                type_t(), aries_acc::agg_sum_t< aries_acc::nullable_type<aries_acc::Decimal>> op = aries_acc::agg_sum_t< aries_acc::nullable_type<aries_acc::Decimal>>(), uint8_t x_sign = 0) const
        {

            if( !has_carry_out ){      // 如果 merge_bits[vt-1] == 0 表示这个值不需要向后传递值 这个 segment 到这个 x 就结束了
                x_sign = 0;
                #pragma unroll
                for(int i_limbs = 0; i_limbs<LIMBS; i_limbs++){
                    x[i_limbs] = 0;
                }
            }

            // storage 
            int left_lane = find_left_lane( tid, has_head_flag, storage );  // 猜测是 查找 block 中 上一个 segment 的界限出现的线程组
            left_lane = left_lane /TPI;
            int tid_delta = (tid / TPI) - left_lane;   // 当前线程 - segment界限出现的线程

            // Store the has_carry_out flag.
            storage.packed[ tid/TPI ] = ( int )has_carry_out | ( left_lane << 1 );  // left_lane * 2 + has_carry_out    TODO 多个线程向同一地址写数据

            // // Run an inclusive scan.
            int first = 0;
            if(tid % TPI == 0)
                storage.values[ first + (tid/TPI) ].value.sign = x_sign;
            #pragma unroll
            for(int i_limbs = 0; i_limbs<LIMBS; i_limbs++){
                storage.values[ first + (tid/TPI) ].value.v[ tid % TPI * LIMBS + i_limbs] = x[i_limbs] ;  // values[tid] = x[vt-1]
            }
            __syncthreads();

            int packed = storage.packed[ left_lane ];   // packed = storage.packed[ left_lane ] 也就是 left_lane 那个线程中 left_lane * 2 + has_carry_out
            left_lane = packed >> 1;    // left_lane 依然等于 left_lane
            tid_delta = (tid/TPI) - left_lane;    // tid - left_lane
            if( 0 == ( 1 & packed ) )   // has_carry_out == 0 那么 --tid_delta  目前没有这种情况
                --tid_delta;

            iterate< s_log2( (nt/TPI) ) >( [&](int pass)
            {
                int offset = 1<< pass;

                if(tid_delta >= offset){
                    uint32_t t[LIMBS];
                    uint8_t t_sign =  storage.values[first + (tid/TPI) - offset].value.sign;
                    #pragma unroll
                    for(int i_limbs = 0; i_limbs<LIMBS; i_limbs++){
                        t[i_limbs] = storage.values[first + (tid/TPI) - offset].value.v[tid%TPI * LIMBS + i_limbs];
                    }

                    x_sign = aries_acc::operator_add( x, x, t, 0, x_sign, t_sign);
                }
                    
                first = (nt/TPI) - first;

                if(tid % TPI == 0)
                    storage.values[ first + (tid/TPI) ].value.sign = x_sign;
                #pragma unroll
                for(int i_limbs = 0; i_limbs<LIMBS; i_limbs++){
                    storage.values[ first + (tid/TPI) ].value.v[ tid % TPI * LIMBS + i_limbs] = x[i_limbs] ;
                }
                __syncthreads();
            } );

            // // Get the exclusive scan by fetching the preceding element. Also return
            // // the carry-out value as the total.
            bool has_carry_in = (tid/TPI) ? ( 0 != ( 1 & storage.packed[ (tid/TPI) - 1 ] ) ) : false;

            segscan_result_t< type_t > result { ( has_carry_in && (tid/TPI) ) ? storage.values[ first + (tid/TPI) - 1 ] : init, storage.values[ first + (nt/TPI) - 1 ],
                    has_carry_in, left_lane };

            __syncthreads();

            return result;
        }
    };

END_ARIES_ACC_NAMESPACE
