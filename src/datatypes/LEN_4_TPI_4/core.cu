#include "static_divide.cu"
#include "mp.cu"
#include "math.h"

namespace cgbn_LEN_4_TPI_4{

  class core {

      static const uint32_t        TPI_ONES=(1ull<<TPI_ONE)-1;

      public:

          __device__ __forceinline__ static uint32_t instance_sync_mask() {
              uint32_t group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
              
              return TPI_ONES<<(group_thread ^ warp_thread);
          }
          __device__ __forceinline__ static uint32_t sync_mask() {
              return instance_sync_mask();
          }

          __device__ __forceinline__ static uint32_t clz(const uint32_t a[LIMBS_ONE]) {
            uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
            uint32_t clz, topclz;
            
            clz=cgbn_LEN_4_TPI_4::mpclz(a);
            topclz=__ballot_sync(sync, clz!=32*LIMBS_ONE);
            if(TPI_ONE<warpSize)
              topclz=topclz<<(warpSize-TPI_ONE)-(warp_thread-group_thread);
            topclz=__clz(topclz);
            if(topclz>=TPI_ONE)
              return LIMBS_ONE*TPI_ONE*32;
            return __shfl_sync(sync, (TPI_ONE-1-group_thread)*32*LIMBS_ONE + clz, 31-topclz, TPI_ONE)-LIMBS_ONE*TPI_ONE*32+LIMBS_ONE*TPI_ONE*32;
        }

        __device__ __forceinline__ static void rotate_right(const uint32_t sync, uint32_t r[], const uint32_t x[], const uint32_t numbits) {
          uint32_t rotate_bits=numbits & 0x1F, numlimbs=numbits>>5, threads=static_divide_small(numlimbs);

          numlimbs=numlimbs-threads*LIMBS_ONE;
          if(numlimbs==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
              r[index]=__shfl_sync(sync, x[index], threadIdx.x+threads, TPI_ONE);
          }
          else {
            mprotate_right(r, x, numlimbs);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
              r[index]=__shfl_sync(sync, r[index], threadIdx.x+threads+(LIMBS_ONE-index<=numlimbs), TPI_ONE);
          }

          if(rotate_bits>0) {
            uint32_t fill=__shfl_sync(sync, r[0], threadIdx.x+1, TPI_ONE);

            mpright(r, r, rotate_bits, fill);
          }
        }

        __device__ __forceinline__ static void rotate_left(const uint32_t sync, uint32_t r[], const uint32_t x[], const uint32_t numbits) {
          // printf("drotate_left _in\n");
          uint32_t rotate_bits=numbits & 0x1F, numlimbs=numbits>>5, threads=static_divide_small(numlimbs);

          numlimbs=numlimbs-threads*LIMBS_ONE;
          if(numlimbs==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
              r[index]=__shfl_sync(sync, x[index], threadIdx.x-threads, TPI_ONE);
          }
          else {
            mprotate_left(r, x, numlimbs);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
              r[index]=__shfl_sync(sync, r[index], threadIdx.x-threads-(index<numlimbs), TPI_ONE);
          }

          if(rotate_bits>0) {
            uint32_t fill=__shfl_sync(sync, r[LIMBS_ONE-1], threadIdx.x-1, TPI_ONE);

            mpleft(r, r, rotate_bits, fill);
          }
        }

        __device__ __forceinline__ static void bitwise_xor(uint32_t r[LIMBS_ONE], const uint32_t a[LIMBS_ONE], const uint32_t b[LIMBS_ONE]) {
          #pragma unroll
          for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=a[index] ^ b[index];
        }
        
        __device__ __forceinline__ static void bitwise_mask_and(uint32_t r[], const uint32_t a[], const int32_t numbits) {
          // printf("dmask_and \n");
          int32_t group_thread=threadIdx.x & TPI_ONE-1, group_base=group_thread*LIMBS_ONE;
          int32_t bits=TPI_ONE*LIMBS_ONE*32;
          
          if(numbits>=bits || numbits<=-bits) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
              r[index]=a[index];
          }
          else if(numbits>=0) {
            int32_t limb=(numbits>>5)-group_base;
            int32_t straddle=uleft_wrap(0xFFFFFFFF, 0, numbits);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
              if(limb<index)
                r[index]=0;
              else if(limb>index)
                r[index]=a[index];
              else
                r[index]=a[index] & straddle;
            }
          }
          else {
            int32_t limb=(numbits+bits>>5)-group_base;
            int32_t straddle=uleft_wrap(0, 0xFFFFFFFF, numbits);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
              if(limb<index)
                r[index]=a[index];
              else if(limb>index)
                r[index]=0;
              else
                r[index]=a[index] & straddle;
            }
          }
        }

        __device__ __forceinline__ static uint32_t clzt(const uint32_t a[LIMBS_ONE]) {
          // printf("clzt 1\n");
          uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
          uint32_t lor, topclz;

          lor=mplor(a);
          topclz=__ballot_sync(sync, lor!=0);
          if(TPI_ONE<warpSize)
            topclz=topclz<<(warpSize-TPI_ONE)-(warp_thread-group_thread);
          topclz= __clz(topclz);
          return umin(topclz, TPI_ONE);
        }

        __device__ __forceinline__ static int32_t resolve_add_a(const int32_t carry, uint32_t &x) {
          // printf("resolve_add in 3\n");
          uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
          uint32_t lane=(group_thread==0) ? 0 : 1<<(threadIdx.x & warpSize-1);
          uint32_t g, p, c;
          uint64_t sum;
        
          c=__shfl_up_sync(sync, carry, 1, TPI_ONE);
          c=(group_thread==0) ? 0 : c;
          x=add_cc(x, c);
          c=addc(0, 0);

          g=__ballot_sync(sync, c==1);
          p=__ballot_sync(sync, x==0xFFFFFFFF && group_thread!=0);
        
          // wrap the carry around  
          sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
          c=lane&(p^sum);

          x=x+(c!=0);
          c=uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_ONE) & 0x01;
          return __shfl_sync(sync, carry+c, TPI_ONE-1, TPI_ONE);
        }

        __device__ __forceinline__ static int32_t resolve_add_b(const int32_t carry, uint32_t x[LIMBS_ONE]) {
          // printf("resolve_add in 4\n");
          uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
          uint32_t lane=(group_thread==0) ? 0 : 1<<(threadIdx.x & warpSize-1);
          uint32_t g, p, c, land;
          uint64_t sum;
          
          c=__shfl_up_sync(sync, carry, 1, TPI_ONE);
          c=(group_thread==0) ? 0 : c;
          x[0]=add_cc(x[0], c);
          #pragma unroll
          for(int32_t index=1;index<LIMBS_ONE;index++) 
            x[index]=addc_cc(x[index], 0);
          c=addc(0, 0);
        
          land=mpland(x);
          g=__ballot_sync(sync, c==1);
          p=__ballot_sync(sync, land==0xFFFFFFFF && group_thread!=0);
        
          sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
          c=lane&(p^sum);
      
          x[0]=add_cc(x[0], c!=0);
          #pragma unroll
          for(int32_t index=1;index<LIMBS_ONE;index++)
            x[index]=addc_cc(x[index], 0);
          
          c=uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_ONE) & 0x01;
          return __shfl_sync(sync, carry+c, TPI_ONE-1, TPI_ONE);
        }
  };

}
