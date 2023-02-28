namespace cgbn_LEN_8_TPI_8 {
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
}