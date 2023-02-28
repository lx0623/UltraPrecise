#include "asm.cu"
#include "shifter_t.cu"
namespace cgbn_LEN_8_TPI_8 {

    /* asm routines */
    __device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b);
    __device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b);
    __device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b);
    __device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b);
    __device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b);
    __device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b);

    #define CGBN_INF_CHAIN 0xFFFFFFFF

    /* classes */
    template<uint32_t length=CGBN_INF_CHAIN, bool carry_in=false, bool carry_out=false>
    class chain_t {
        public:
            uint32_t _position;

            __device__ __forceinline__ chain_t();
            __device__ __forceinline__ ~chain_t();
            __device__ __forceinline__ uint32_t add(uint32_t a, uint32_t b);
            __device__ __forceinline__ uint32_t sub(uint32_t a, uint32_t b);
            __device__ __forceinline__ uint32_t madlo(uint32_t a, uint32_t b, uint32_t c);
            __device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c);
    };

    __device__ __forceinline__ uint32_t mpclz(const uint32_t a[]) {
        uint32_t word=0, count=0;
        
        #pragma unroll
        for(int32_t index=LIMBS_ONE-1;index>=0;index--) {
            word=(word!=0) ? word : a[index];
            count=(word!=0) ? count : (LIMBS_ONE-index)*32;
        }
        if(word!=0)
            count=count+__clz(word);
        return count;
    }

    __device__ __forceinline__ void mpleft(uint32_t r[], const uint32_t a[], const uint32_t numbits, const uint32_t fill=0) {
        #pragma unroll
        for(int32_t index=LIMBS_ONE-1;index>=1;index--)
            r[index]=uleft_clamp(a[index-1], a[index], numbits);
        r[0]=uleft_clamp(fill, a[0], numbits);
    }

    __device__ __forceinline__ void mpright(uint32_t r[], const uint32_t a[], const uint32_t numbits, const uint32_t fill=0) {
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE-1;index++) 
            r[index]=uright_clamp(a[index], a[index+1], numbits);
        r[LIMBS_ONE-1]=uright_clamp(a[LIMBS_ONE-1], fill, numbits);
    }

    __device__ __forceinline__ uint32_t mpland(const uint32_t a[]) {
        uint32_t r=a[0];
        
        #pragma unroll
        for(int32_t index=1;index<LIMBS_ONE;index++)
        r=r & a[index];
        return r;
    }

    __device__ __forceinline__ uint32_t mplor(const uint32_t a[]) {
        uint32_t r=a[0];
        
        #pragma unroll
        for(int32_t index=1;index<LIMBS_ONE;index++)
        r=r | a[index];
        return r;
    }

    __device__ __forceinline__ void mpmul(uint32_t lo[], uint32_t hi[], const uint32_t a[], const uint32_t b[]) {
        uint32_t c;
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++) {
            lo[index]=0;
            hi[index]=0;
        }
        
        #pragma unroll
        for(int32_t i=0;i<LIMBS_ONE;i++) {
            chain_t<LIMBS_ONE,false,true> chain1;
            #pragma unroll
            for(int32_t j=0;j<LIMBS_ONE;j++) {
            if(i+j<LIMBS_ONE)
                lo[i+j]=chain1.madlo(a[i], b[j], lo[i+j]);
            else
                hi[i+j-LIMBS_ONE]=chain1.madlo(a[i], b[j], hi[i+j-LIMBS_ONE]);
            }
            if(i==0)
            c=0;
            else
            c=addc(0, 0);
            
            chain_t<LIMBS_ONE> chain2;
            #pragma unroll
            for(int32_t j=0;j<LIMBS_ONE-1;j++) {
            if(i+j+1<LIMBS_ONE)
                lo[i+j+1]=chain2.madhi(a[i], b[j], lo[i+j+1]);
            else
                hi[i+j+1-LIMBS_ONE]=chain2.madhi(a[i], b[j], hi[i+j+1-LIMBS_ONE]);
            }
            hi[i]=chain2.madhi(a[i], b[LIMBS_ONE-1], c);
        }
    }

    __device__ __forceinline__ void mpsub_cc(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        chain_t<LIMBS_ONE,false,true> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=chain.sub(a[index], b[index]);
    }

    __device__ __forceinline__ uint32_t mpsub(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        mpsub_cc(r, a, b);
        return subc(0, 0);
    }

    __device__ __forceinline__ uint32_t mpmul32(uint32_t r[], const uint32_t a[], const uint32_t b) {
        uint32_t carry=0;
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++) {
            uint32_t temp=a[index];
            
            r[index]=madlo_cc(temp, b, carry);
            carry=madhic(temp, b, 0);
        }
        return carry;
    }

    __device__ __forceinline__ void mpadd_cc(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        chain_t<LIMBS_ONE,false,true> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=chain.add(a[index], b[index]);
    }

    __device__ __forceinline__ uint32_t mpadd(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        mpadd_cc(r, a, b);
        return addc(0, 0);
    }

    __device__ __forceinline__ void mpsub32_cc(uint32_t r[], const uint32_t a[], const uint32_t b) {
        chain_t<LIMBS_ONE,false,true> chain;
        r[0]=chain.sub(a[0], b);
        #pragma unroll
        for(int32_t index=1;index<LIMBS_ONE;index++)
            r[index]=chain.sub(a[index], 0);
    }

    __device__ __forceinline__ uint32_t mpsub32(uint32_t r[], const uint32_t a[], const uint32_t b) {
        mpsub32_cc(r, a, b);
        return subc(0, 0);
    }

    __device__ __forceinline__ void mprotate_left(uint32_t r[], const uint32_t a[], const uint32_t numlimbs) {
        // printf("mprotate_left tid =%d max_roration = %d\n",threadIdx.x, MAX_ROTATION);
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++) 
            r[index]=a[index];
        
        if(LIMBS_ONE>bit_set<MAX_ROTATION>::high_bit*2)
            shifter_t<LIMBS_ONE, bit_set<MAX_ROTATION>::high_bit, true>::mprotate_left(r, numlimbs);
        else if((LIMBS_ONE-1&LIMBS_ONE)==0)
            shifter_t<LIMBS_ONE, LIMBS_ONE/2, false>::mprotate_left(r, numlimbs);
        else
            shifter_t<LIMBS_ONE, bit_set<LIMBS_ONE>::high_bit, false>::mprotate_left(r, numlimbs);
    }

    __device__ __forceinline__ void mprotate_right(uint32_t r[], const uint32_t a[], const uint32_t numlimbs) {
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++) 
            r[index]=a[index];

        if(LIMBS_ONE>bit_set<MAX_ROTATION>::high_bit*2)
            shifter_t<LIMBS_ONE, bit_set<MAX_ROTATION>::high_bit, true>::mprotate_right(r, numlimbs);
        else if((LIMBS_ONE-1&LIMBS_ONE)==0)
            shifter_t<LIMBS_ONE, LIMBS_ONE/2, false>::mprotate_right(r, numlimbs);
        else
            shifter_t<LIMBS_ONE, bit_set<LIMBS_ONE>::high_bit, false>::mprotate_right(r, numlimbs);
    }

}