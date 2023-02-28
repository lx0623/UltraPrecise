namespace cgbn_LEN_32_TPI_8 {

    template<uint32_t length, bool carry_in, bool carry_out>
    __device__ __forceinline__ chain_t<length, carry_in, carry_out>::chain_t() : _position(0) {
    }

    template<uint32_t length, bool carry_in, bool carry_out>
    __device__ __forceinline__ chain_t<length, carry_in, carry_out>::~chain_t() {
    }

    template<uint32_t length, bool carry_in, bool carry_out>
    __device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::add(uint32_t a, uint32_t b) {
        uint32_t r;
        
        _position++;
        if(length==1 && _position==1 && !carry_in && !carry_out)
            r=a+b;
        else if(_position==1 && !carry_in)
            r=add_cc(a, b);
        else if(_position<length || carry_out)
            r=addc_cc(a, b);
        else
            r=addc(a, b);
        
        return r;
    }

    template<uint32_t length, bool carry_in, bool carry_out>
    __device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::sub(uint32_t a, uint32_t b) {
        uint32_t r;
        
        _position++;
        if(length==1 && _position==1 && !carry_in && !carry_out)
            r=a-b;
        else if(_position==1 && !carry_in)
            r=sub_cc(a, b);
        else if(_position<length || carry_out)
            r=subc_cc(a, b);
        else
            r=subc(a, b);
        return r;
    }

    template<uint32_t length, bool carry_in, bool carry_out>
    __device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::madlo(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;
        
        _position++;
        if(length==1 && _position==1 && !carry_in && !carry_out)
            r=cgbn_LEN_32_TPI_8::madlo(a, b, c);
        else if(_position==1 && !carry_in)
            r=cgbn_LEN_32_TPI_8::madlo_cc(a, b, c);
        else if(_position<length || carry_out)
            r=cgbn_LEN_32_TPI_8::madloc_cc(a, b, c);
        else
            r=cgbn_LEN_32_TPI_8::madloc(a, b, c);
        return r;
    }

    template<uint32_t length, bool carry_in, bool carry_out>
    __device__ __forceinline__ uint32_t chain_t<length, carry_in, carry_out>::madhi(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;
        
        _position++;
        if(length==1 && _position==1 && !carry_in && !carry_out)
            r=cgbn_LEN_32_TPI_8::madhi(a, b, c);
        else if(_position==1 && !carry_in)
            r=cgbn_LEN_32_TPI_8::madhi_cc(a, b, c);
        else if(_position<length || carry_out)
            r=cgbn_LEN_32_TPI_8::madhic_cc(a, b, c);
        else
            r=cgbn_LEN_32_TPI_8::madhic(a, b, c);
        return r;
    }

    __device__ __forceinline__ static int32_t fast_propagate_add_a(const uint32_t carry, uint32_t &x) {
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t lane=(group_thread==0) ? 0 : 1<<warp_thread;
        uint32_t g, p, c; 
        uint64_t sum;

        g=__ballot_sync(sync, carry==1);
        p=__ballot_sync(sync, x==0xFFFFFFFF && group_thread!=0);

        // wrap the carry around  
        sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
        c=lane&(p^sum);

        x=x+(c!=0);

        return uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_TWO) & 0x01;
    }

    __device__ __forceinline__ int32_t fast_propagate_add(const uint32_t carry, uint32_t x[]) {
        // printf("fast_propagate_add 4\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1, warp_thread=threadIdx.x & warpSize-1;
        
        uint32_t lane=(group_thread==0) ? 0 : 1<<warp_thread;
        uint32_t land, g, p, c; 
        uint64_t sum;
        
        land=mpland(x);
        g=__ballot_sync(sync, carry==1);
        p=__ballot_sync(sync, land==0xFFFFFFFF && group_thread!=0);

        // wrap the carry around  
        sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
        c=lane & (p ^ sum);
        
        // printf("fast_propagate_add:: tid = %d\n sync = %d lane = %d land = %d g = %d p = %d c = %d sum = %lld\n",blockIdx.x*blockDim.x + threadIdx.x, sync, lane, land, g, p, c, sum);
        x[0]=add_cc(x[0], c!=0);
        #pragma unroll
        for(int32_t index=1;index<LIMBS_THR;index++)
            x[index]=addc_cc(x[index], 0);

        return uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_TWO) & 0x01;
    }

    __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t x[]) {
        // printf("fast_propagate_sub 4\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1, warp_thread=threadIdx.x & warpSize-1;
        
        uint32_t lane=(group_thread==0) ? 0 : 1<<warp_thread;
        uint32_t lor, g, p, c; 
        uint64_t sum;
        
        lor=mplor(x);
        g=__ballot_sync(sync, carry==0xFFFFFFFF);
        p=__ballot_sync(sync, lor==0 && group_thread!=0);

        // wrap the carry around  
        sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
        c=lane & (p ^ sum);
        c=(c==0) ? 0 : 0xFFFFFFFF;
        
        // printf("fast_propagate_sub:: tid = %d\n sync = %d lane = %d lor = %d g = %d p = %d c = %d sum = %lld\n",blockIdx.x*blockDim.x + threadIdx.x, sync, lane, lor, g, p, c, sum);
        x[0]=add_cc(x[0], c);
        #pragma unroll
        for(int32_t index=1;index<LIMBS_THR;index++)
            x[index]=addc_cc(x[index], c);

        return uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_TWO) & 0x01;
    }

    __device__ __forceinline__ int32_t resolve_sub(const int32_t carry, uint32_t x[LIMBS_THR]) {
        // printf("resolve_sub in_ 4\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t lane=(group_thread==0) ? 0 : 1<<(threadIdx.x & warpSize-1);
        uint32_t g, p, lor;
        int32_t  c;
        uint64_t sum;

        c=__shfl_up_sync(sync, carry, 1, TPI_TWO);
        c=(group_thread==0) ? 0 : c;
        x[0]=add_cc(x[0], c);
        c=c>>31;
        #pragma unroll
        for(int32_t index=1;index<LIMBS_THR;index++) 
            x[index]=addc_cc(x[index], c);
        c=addc(0, c);

        lor=mplor(x);
        g=__ballot_sync(sync, c==0xFFFFFFFF);
        p=__ballot_sync(sync, lor==0 && group_thread!=0);

        // wrap the carry around  
        sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
        c=lane&(p^sum);
        c=(c==0) ? 0 : 0xFFFFFFFF;
        x[0]=add_cc(x[0], c);
        #pragma unroll
        for(int32_t index=1;index<LIMBS_THR;index++) 
            x[index]=addc_cc(x[index], c);

        c=uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_TWO) & 0x01;
        return __shfl_sync(sync, carry-c, TPI_TWO-1, TPI_TWO);
    }

    __device__ __forceinline__ int32_t resolve_sub_a(const int32_t carry, uint32_t &x) {
        // printf("resolve_sub in_ 3\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t lane=(group_thread==0) ? 0 : 1<<(threadIdx.x & warpSize-1);
        uint32_t g, p;
        int32_t  c;
        uint64_t sum;
    
        c=__shfl_up_sync(sync, carry, 1, TPI_TWO);
        c=(group_thread==0) ? 0 : c;
        x=add_cc(x, c);
        c=addc(0, c>>31);

        g=__ballot_sync(sync, c==0xFFFFFFFF);
        p=__ballot_sync(sync, x==0 && group_thread!=0);
    
        // wrap the carry around  
        sum=make_wide(g, g) + make_wide(g, g) + make_wide(p, p);
        c=lane&(p^sum);

        x=x-(c!=0);
        c=uright_wrap(sum>>32, 0, threadIdx.x - group_thread + TPI_TWO) & 0x01;
        return __shfl_sync(sync, carry-c, TPI_TWO-1, TPI_TWO);
    }

    __device__ __forceinline__ uint32_t uapprox(uint32_t d) {
        float    f;
        uint32_t a, t0, t1;
        int32_t  s;

        // special case d=0x80000000
        if(d==0x80000000)
            return 0xFFFFFFFF;
        
        // get a first estimate using float 1/x
        f=__uint_as_float((d>>8) + 0x3F000000);
        asm volatile("rcp.approx.f32 %0,%1;" : "=f"(f) : "f"(f));
        a=__float_as_uint(f);
        a=madlo(a, 512, 0xFFFFFE00);
        
        // use Newton-Raphson to improve estimate
        s=madhi(d, a, d);
        t0=abs(s);
        t0=madhi(t0, a, t0);
        a=(s>=0) ? a-t0 : a+t0;

        // two corrections steps give exact result
        a=a-madhi(d, a, d);       // first correction

        t0=madlo_cc(d, a, 0);     // second correction
        t1=madhic(d, a, d);
        t1=(t1!=0) ? t1 : (t0>=d);
        a=a-t1;
        return a;
    }

    __device__ __forceinline__ uint32_t ucorrect(const uint32_t x0, const uint32_t x1, const int32_t x2, const uint32_t d0, const uint32_t d1) {
        uint32_t q=0, add, y0, y1, y2;
        
        add=x2>>31;

        // first correction
        y0=add_cc(x0, d0);
        y1=addc_cc(x1, d1);
        y2=addc_cc(x2, 0);
        add=addc(add, 0);
        q=q-add;
        
        // second correction
        y0=add_cc(y0, d0);
        y1=addc_cc(y1, d1);
        y2=addc_cc(y2, 0);
        add=addc(add, 0);
        q=q-add;
        
        // third correction
        y0=add_cc(y0, d0);
        y1=addc_cc(y1, d1);
        y2=addc_cc(y2, 0);
        add=addc(add, 0);
        q=q-add;

        // fourth correction
        y0=add_cc(y0, d0);
        y1=addc_cc(y1, d1);
        y2=addc_cc(y2, 0);
        add=addc(add, 0);
        q=q-add;

        return q;
    }


    __device__ __forceinline__ uint32_t udiv(const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t d0, const uint32_t d1, const uint32_t approx) {
        uint32_t q, add, y0, y1, y2;

        // q=MIN(0xFFFFFFFF, HI(approx * hi) + hi + ((lo<d) ? 1 : 2));
        sub_cc(x1, d1);
        add=subc(x2, 0xFFFFFFFE);
        q=madhi(x2, approx, add);
        q=(q<x2) ? 0xFFFFFFFF : q;           // the only case where this can carry out is if hi and approx are both 0xFFFFFFFF and add=2
                                            // but in this case, q will end up being 0xFFFFFFFF, which is what we want
                                            // if q+hi carried out, set q to 0xFFFFFFFF

        y0=madlo(q, d0, 0);
        y1=madhi(q, d0, 0);
        y1=madlo_cc(q, d1, y1);
        y2=madhic(q, d1, 0);
        
        y0=sub_cc(x0, y0);    // first correction
        y1=subc_cc(x1, y1);
        y2=subc_cc(x2, y2);
        add=subc(0, 0);
        q=q+add;

        y0=add_cc(y0, d0);    // second correction
        y1=addc_cc(y1, d1);
        y2=addc_cc(y2, 0);
        add=addc(add, 0);
        q=q+add;
        
        y0=add_cc(y0, d0);    // third correction
        y1=addc_cc(y1, d1);
        y2=addc_cc(y2, 0);
        add=addc(add, 0);
        q=q+add;
        
        y0=add_cc(y0, d0);    // fourth correction
        y1=addc_cc(y1, d1);
        y2=addc_cc(y2, 0);
        q=addc(q, add);
        
        return q;
    }

    __device__ __forceinline__ static void dlimbs_scatter(uint32_t r[DLIMBS_ONE], const uint32_t x[LIMBS_THR], const uint32_t source_thread) {
        // printf("dlimbs_scatter in 1\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t t;

        #pragma unroll
        for(int32_t index=0;index<DLIMBS_ONE;index++)
            r[index] = 0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++) {
        t=__shfl_sync(sync, x[index], source_thread, TPI_TWO);
        r[(index+LIMB_OFFSET_TWO)%DLIMBS_ONE]=(group_thread==(index+LIMB_OFFSET_TWO)/DLIMBS_ONE) ? t : r[(index+LIMB_OFFSET_TWO)%DLIMBS_ONE];
        }
    }

    __device__ __forceinline__  void dlimbs_approximate(uint32_t approx[DLIMBS_ONE], const uint32_t denom[DLIMBS_ONE]) {
        // printf("dlimbs_approximate in 1\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t x, d0, d1, x0, x1, x2, est, a, h, l;
        int32_t  c, top;
        
        // computes (beta^2 - 1) / denom - beta, where beta=1<<32*LIMBS_THR
        
        x=0xFFFFFFFF-denom[0];
        
        d1=__shfl_sync(sync, denom[0], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[0], TPI_TWO-2, TPI_TWO);
        
        approx[0]=0;
        a=uapprox(d1);
    
        #pragma unroll(1)
        for(int32_t thread=LIMBS_THR-1;thread>=0;thread--) {
        x0=__shfl_sync(sync, x, TPI_TWO-3, TPI_TWO);
        x1=__shfl_sync(sync, x, TPI_TWO-2, TPI_TWO);
        x2=__shfl_sync(sync, x, TPI_TWO-1, TPI_TWO);
        est=udiv(x0, x1, x2, d0, d1, a);

        l=madlo_cc(est, denom[0], 0);
        h=madhic(est, denom[0], 0);

        x=sub_cc(x, h);
        c=subc(0, 0);  // thread TPI_TWO-1 is zero
        
        top=__shfl_sync(sync, x, TPI_TWO-1, TPI_TWO);
        x=__shfl_sync(sync, x, threadIdx.x-1, TPI_TWO);
        c=__shfl_sync(sync, c, threadIdx.x-1, TPI_TWO);

        x=sub_cc(x, l);
        c=subc(c, 0);

        if(top+resolve_sub_a(c, x)<0) {
            // means a correction is required, should be very rare
            x=add_cc(x, denom[0]);
            c=addc(0, 0);
            fast_propagate_add_a(c, x);
            est--;
        }
        approx[0]=(group_thread==thread+TPI_TWO-LIMBS_THR) ? est : approx[0];
        }
    }

    __device__ __forceinline__ static void dlimbs_div_estimate(uint32_t q[DLIMBS_ONE], const uint32_t x[DLIMBS_ONE], const uint32_t approx[DLIMBS_ONE]) {
        // printf("dlimbs_div_estimate in 1\n");
        uint32_t sync=core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t t, c;
        uint64_t w;
    
        // computes q=(x*approx>>32*LIMBS_THR) + x + 3
        //          q=min(q, (1<<32*LIMBS_THR)-1);
        // 
        // Notes:   leaves junk in lower words of q 
        
        w=0;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++) {
            t=__shfl_sync(sync, x[0], TPI_TWO-LIMBS_THR+index, TPI_TWO);
            w=mad_wide(t, approx[0], w);
            t=__shfl_sync(sync, ulow(w), threadIdx.x+1, TPI_TWO);   // half size: take advantage of zero wrapping
            w=(w>>32)+t;
        }
        
        // increase the estimate by 3
        t=(group_thread==TPI_TWO-LIMBS_THR) ? 3 : 0;
        w=w + t + x[0];
        
        q[0]=ulow(w);
        c=uhigh(w);
        if(core::resolve_add_a(c, q[0])!=0)
        q[0]=0xFFFFFFFF;
    }

    __device__ __forceinline__ void dlimbs_all_gather(uint32_t r[LIMBS_THR], const uint32_t x[DLIMBS_ONE]) {
        // printf("dlimbs_all_gather in 1\n");
        uint32_t sync=core::sync_mask();
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++) 
        r[index]=__shfl_sync(sync, x[(index+LIMB_OFFSET_TWO)%DLIMBS_ONE], (index+LIMB_OFFSET_TWO)/DLIMBS_ONE, TPI_TWO);
    }

    __device__ __forceinline__ int32_t ucmp(uint32_t a, uint32_t b) {
        int32_t compare;
        
        compare=(a>b) ? 1 : 0;
        compare=(a<b) ? -1 : compare;
        return compare;
    }
}

// #include "chain.cu"