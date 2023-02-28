namespace cgbn_LEN_8_TPI_4 {

    __device__ __forceinline__ uint32_t umin(uint32_t a, uint32_t b) {
        uint32_t r;
        
        asm volatile ("min.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
        uint32_t r;

        asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
        uint32_t r;

        asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
        uint32_t r;

        asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
        uint32_t r;

        asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
        uint32_t r;

        asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
        uint32_t r;

        asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;

        asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t uleft_clamp(uint32_t lo, uint32_t hi, uint32_t amt) {
        uint32_t r;
        
        #if __CUDA_ARCH__>=320   
            asm volatile ("shf.l.clamp.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
        #else
            amt=umin(amt, 32);
            r=hi<<amt;
            r=r | (lo>>32-amt);
        #endif
        return r;
    }

    __device__ __forceinline__ uint32_t uright_clamp(uint32_t lo, uint32_t hi, uint32_t amt) {
        uint32_t r;
        
        #if __CUDA_ARCH__>=320   
            asm volatile ("shf.r.clamp.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
        #else
            amt=umin(amt, 32);
            r=lo>>amt;
            r=r | (hi<<32-amt);
        #endif
        return r;
    }

    __device__ __forceinline__ uint32_t uleft_wrap(uint32_t lo, uint32_t hi, uint32_t amt) {
        uint32_t r;
        
        #if __CUDA_ARCH__>=320   
            asm volatile ("shf.l.wrap.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
        #else
            amt=amt & 0x1F;
            r=hi<<amt;
            r=r | (lo>>32-amt);
        #endif
        return r;
    }

    __device__ __forceinline__ uint64_t mad_wide(uint32_t a, uint32_t b, uint64_t c) {
        uint64_t r;
        
        asm volatile ("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t ulow(uint64_t wide) {
        uint32_t r;

        asm volatile ("{\n\t"
                        ".reg .u32 %ignore;\n\t"
                        "mov.b64 {%0,%ignore},%1;\n\t"
                        "}" : "=r"(r) : "l"(wide));
        return r;
    }

    __device__ __forceinline__ uint32_t uhigh(uint64_t wide) {
        uint32_t r;

        asm volatile ("{\n\t"
                        ".reg .u32 %ignore;\n\t"
                        "mov.b64 {%ignore,%0},%1;\n\t"
                        "}" : "=r"(r) : "l"(wide));
        return r;
    }

    __device__ __forceinline__ uint64_t make_wide(uint32_t lo, uint32_t hi) {
        uint64_t r;
        
        asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(r) : "r"(lo), "r"(hi));
        return r;
    }

    __device__ __forceinline__ uint32_t uright_wrap(uint32_t lo, uint32_t hi, uint32_t amt) {
        uint32_t r;
        
        #if __CUDA_ARCH__>=320   
            asm volatile ("shf.r.wrap.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
        #else
            amt=amt & 0x1F;
            r=lo>>amt;
            r=r | (hi<<32-amt);
        #endif
        return r;
    }
    // __device__ __forceinline__ uint64_t make_wide(uint32_t lo, uint32_t hi) {
    //     uint64_t r;
        
    //     asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(r) : "r"(lo), "r"(hi));
    //     return r;
    // }

}