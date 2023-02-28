namespace cgbn_LEN_16_TPI_16 {

    __device__ __forceinline__ uint32_t static_divide_small(uint32_t numerator) {
    uint32_t est=0xFFFFFFFF/LIMBS_ONE;

    // not exact, but ok for den<2^10 and num<2^20
    return __umulhi((uint32_t)est, numerator+1);
    }

    // template<uint32_t denominator>
    // __device__ __forceinline__ uint32_t static_remainder_small(uint32_t numerator) {

    //   // not exact, but ok for den<1024 and num<2^20
    //   return numerator-static_divide_small<denominator>(numerator)*denominator;
    // }
}