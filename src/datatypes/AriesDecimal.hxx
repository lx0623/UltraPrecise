/*
 * decimal.hxx
 *
 *  Created on: 2019年6月21日
 *      Author: david
 */
#include "AriesDefinition.h"
#include "AriesDataTypeUtil.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    __device__  static uint32_t ARIES_ARRAY_SCALE[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    
    __device__  static uint32_t ARIES_POW10_ARRAY[][NUM_TOTAL_DIG] = {
        {0x0000000a},
        {0x00000064},
        {0x000003e8},
        {0x00002710},
        {0x000186a0},
        {0x000f4240},
        {0x00989680},
        {0x05f5e100},
        {0x3b9aca00},
        {0x540be400, 0x00000002},
        {0x4876e800, 0x00000017},
        {0xd4a51000, 0x000000e8},
        {0x4e72a000, 0x00000918},
        {0x107a4000, 0x00005af3},
        {0xa4c68000, 0x00038d7e},
        {0x6fc10000, 0x002386f2},
        {0x5d8a0000, 0x01634578},
        {0xa7640000, 0x0de0b6b3},
        {0x89e80000, 0x8ac72304},
        {0x63100000, 0x6bc75e2d, 0x00000005},
        {0xdea00000, 0x35c9adc5, 0x00000036},
        {0xb2400000, 0x19e0c9ba, 0x0000021e},
        {0xf6800000, 0x02c7e14a, 0x0000152d},
        {0xa1000000, 0x1bcecced, 0x0000d3c2},
        {0x4a000000, 0x16140148, 0x00084595},
        {0xe4000000, 0xdcc80cd2, 0x0052b7d2},
        {0xe8000000, 0x9fd0803c, 0x033b2e3c},
        {0x10000000, 0x3e250261, 0x204fce5e},
        {0xa0000000, 0x6d7217ca, 0x431e0fae, 0x00000001},
        {0x40000000, 0x4674edea, 0x9f2c9cd0, 0x0000000c},
        {0x80000000, 0xc0914b26, 0x37be2022, 0x0000007e},
        {0x00000000, 0x85acef81, 0x2d6d415b, 0x000004ee},
        {0x00000000, 0x38c15b0a, 0xc6448d93, 0x0000314d},
        {0x00000000, 0x378d8e64, 0xbead87c0, 0x0001ed09},
        {0x00000000, 0x2b878fe8, 0x72c74d82, 0x00134261},
        {0x00000000, 0xb34b9f10, 0x7bc90715, 0x00c097ce},
        {0x00000000, 0x00f436a0, 0xd5da46d9, 0x0785ee10},
        {0x00000000, 0x098a2240, 0x5a86c47a, 0x4b3b4ca8},
        // {0x00000000,0x5f655680,0x8943acc4,0xf050fe93,0x00000002},
        // {0x00000000,0xb9f56100,0x5ca4bfab,0x6329f1c3,0x0000001d},
        // {0x00000000,0x4395ca00,0x9e6f7cb5,0xdfa371a1,0x00000125},
        // {0x00000000,0xa3d9e400,0x305adf14,0xbc627050,0x00000b7a},
        // {0x00000000,0x6682e800,0xe38cb6ce,0x5bd86321,0x000072cb},
        // {0x00000000,0x011d1000,0xe37f2410,0x9673df52,0x00047bf1},
        // {0x00000000,0x0b22a000,0xe2f768a0,0xe086b93c,0x002cd76f},
        // {0x00000000,0x6f5a4000,0xddaa1640,0xc5433c60,0x01c06a5e},
        // {0x00000000,0x59868000,0xa8a4de84,0xb4a05bc8,0x118427b3},
        // {0x00000000,0x7f410000,0x9670b12b,0x0e4395d6,0xaf298d05}
    };


template <int DEC_LEN>
struct AriesDecimal
{
    //最高位存放 sign , 末尾 6 位表示 prec 2^6 = 64 > 5*9;
    uint8_t sign;
    //最高位存放 over , 末尾 6 位表示 frac 2^6 = 64 > 5*9;
    uint8_t prec;

    //为了字节对齐
    uint8_t frac;
    uint8_t error;

    uint32_t v[DEC_LEN];

public:

    ARIES_DEVICE AriesDecimal()
    {
        sign = 0;
        prec = 0;
        frac = 0;
        error = 0;
        aries_memset(v, 0x00, sizeof(v));
    }

    ARIES_DEVICE void initialize(uint32_t pr, uint32_t fc, uint32_t m)
    {
        //符号 + prec
        prec = pr;
        //小数点位数
        frac = fc;
        //数组存储位置
        aries_memset(v, 0x00, sizeof(v));

        sign = 0;
        error = 0;
    }

    // CompactDecimal to AriesDecimal，根据 compact 和 精度信息完成构造，m无用
    ARIES_DEVICE AriesDecimal(const CompactDecimal *compact, uint32_t precision, uint32_t scale, uint32_t m = ARIES_MODE_EMPTY)
    {
        initialize(precision, scale, 0);
        int len = GetDecimalRealBytes(precision, scale);
        aries_memcpy((char *)(v), compact->data, len);
        char *temp = ((char *)(v));
        temp += len - 1;
        sign = GET_SIGN_FROM_BIT(*temp);
        *temp = *temp & 0x7f;
    }

    ARIES_DEVICE AriesDecimal &operator<<(int n)
    {
        if(n<=0){
            return *this;
        }
        uint32_t res[DEC_LEN] = {0};
        uint32_t carry = 0;
        uint64_t temp;
        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++)
        {
            carry = 0;
            #pragma unroll
            for (uint32_t j = 0; j < ARIES_ARRAY_SCALE[n - 1]; j++)
            {
                if (i + j > DEC_LEN)
                {
                    break;
                }
                temp = (uint64_t)v[i] * (uint64_t)ARIES_POW10_ARRAY[n - 1][j] + res[i + j] + carry;
                carry = (temp & 0xffffffff00000000) >> 32;
                res[i + j] = temp & 0x00000000ffffffff;
            }
            if (i + ARIES_ARRAY_SCALE[n - 1] < DEC_LEN)
            {
                res[i + ARIES_ARRAY_SCALE[n - 1]] = carry;
            }
        }
        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++)
        {
            v[i] = res[i];
        }
        return *this;
    }

    ARIES_DEVICE void AlignAddSubData(AriesDecimal &d)
    {
        if (frac == d.frac)
        {
            return;
        }
        if (frac < d.frac)
        {
            *this << d.frac - frac;
            frac = d.frac;
        }
        else
        {
            d << frac - d.frac;
            d.frac = frac;
        }
    }

    // for add
    ARIES_DEVICE AriesDecimal &operator+=(const AriesDecimal &d)
    {
        AriesDecimal added(d);
        added.AlignAddSubData(*this);
        if (sign == added.sign)
        {
            asm volatile("add.cc.u32 %0, %1, %2;"
                         : "=r"(v[0])
                         : "r"(added.v[0]), "r"(v[0]));
            #pragma unroll
            for (int32_t i = 1; i < DEC_LEN; i++)
                asm volatile("addc.cc.u32 %0, %1, %2;"
                             : "=r"(v[i])
                             : "r"(added.v[i]), "r"(v[i]));
        }
        else
        {
            int64_t r = 0;
            #pragma unroll
            for (int i = DEC_LEN - 1; i >= 0; i--)
            {
                r = (int64_t)v[i] - added.v[i];
                if (r != 0)
                {
                    break;
                }
            }
            if (r >= 0)
            {
                asm volatile("sub.cc.u32 %0, %1, %2;"
                             : "=r"(v[0])
                             : "r"(v[0]), "r"(added.v[0]));
                #pragma unroll
                for (int32_t i = 1; i < DEC_LEN; i++)
                    asm volatile("subc.cc.u32 %0, %1, %2;"
                                 : "=r"(v[i])
                                 : "r"(v[i]), "r"(added.v[i]));
            }
            else
            {
                asm volatile("sub.cc.u32 %0, %1, %2;"
                             : "=r"(v[0])
                             : "r"(added.v[0]), "r"(v[0]));
                #pragma unroll
                for (int32_t i = 1; i < DEC_LEN; i++)
                    asm volatile("subc.cc.u32 %0, %1, %2;"
                                 : "=r"(v[i])
                                 : "r"(added.v[i]), "r"(v[i]));
            }
            sign = (r > 0 && !d.sign) || (r < 0 && d.sign);
        }
        return *this;
    }

    // two operators
    friend ARIES_DEVICE AriesDecimal operator+(const AriesDecimal &left, const AriesDecimal &right)
    {
        //将 const left 赋值到temp进行操作
        AriesDecimal tmp(left);
        return tmp += right;
    }

    // for sub
    ARIES_DEVICE AriesDecimal &operator-=(const AriesDecimal &d)
    {
        AriesDecimal added(d);
        added.AlignAddSubData(*this);

        if (added.sign != sign)
        {
            asm volatile("add.cc.u32 %0, %1, %2;"
                         : "=r"(v[0])
                         : "r"(added.v[0]), "r"(v[0]));

            #pragma unroll
            for (int32_t i = 1; i < DEC_LEN; i++)
                asm volatile("addc.cc.u32 %0, %1, %2;"
                             : "=r"(v[i])
                             : "r"(added.v[i]), "r"(v[i]));
        }
        else
        {
            int64_t r = 0;
            #pragma unroll
            for (int i = DEC_LEN - 1; i >= 0; i--)
            {
                r = (int64_t)v[i] - added.v[i];
                if (r != 0)
                {
                    break;
                }
            }
            if (r >= 0)
            {
                asm volatile("sub.cc.u32 %0, %1, %2;"
                             : "=r"(v[0])
                             : "r"(v[0]), "r"(added.v[0]));
                #pragma unroll
                for (int32_t i = 1; i < DEC_LEN; i++)
                    asm volatile("subc.cc.u32 %0, %1, %2;"
                                 : "=r"(v[i])
                                 : "r"(v[i]), "r"(added.v[i]));
            }
            else
            {
                asm volatile("sub.cc.u32 %0, %1, %2;"
                             : "=r"(v[0])
                             : "r"(added.v[0]), "r"(v[0]));
                #pragma unroll
                for (int32_t i = 1; i < DEC_LEN; i++)
                    asm volatile("subc.cc.u32 %0, %1, %2;"
                                 : "=r"(v[i])
                                 : "r"(added.v[i]), "r"(v[i]));
            }   
            sign = (r > 0 && added.sign) || (r < 0 && !added.sign);
        }

        return *this;
    }

    friend ARIES_DEVICE AriesDecimal operator-(const AriesDecimal &left, const AriesDecimal &right)
    {
        AriesDecimal tmp(left);
        return tmp -= right;
    }
    
    ARIES_DEVICE AriesDecimal operator-() {
        AriesDecimal tmp(*this);
        if(tmp.sign == 0){
            tmp.sign = 1;
        }
        else{
            tmp.sign = 0;
        }
        return tmp;
    }
    
    ARIES_DEVICE bool ToCompactDecimal(char *buf, int len)
    {
        aries_memcpy(buf, (char *)(v), len);
        SET_SIGN_BIT(buf[len - 1], sign);
        return true;
    }

    // for multiple
    ARIES_DEVICE AriesDecimal &operator*=(const AriesDecimal &d)
    {
        sign = d.sign ^ sign;
        frac = frac + d.frac;

        uint32_t inner_res[DEC_LEN * 2] = {0};

        uint64_t temp;
        uint32_t carry;

        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++)
        {
            carry = 0;
            #pragma unroll
            for (int j = 0; j < DEC_LEN; j++)
            {
                // temp 表示范围最大值 2^64-1 右侧表达式 表示范围最大值 (2^32-1) * (2^32-1) + (2^32-1) + (2^32-1) = 2^64-1
                temp = (uint64_t)v[i] * d.v[j] + inner_res[i + j] + carry;
                carry = temp / PER_DEC_MAX_SCALE;
                inner_res[i + j] = temp % PER_DEC_MAX_SCALE;
            }
            inner_res[i + DEC_LEN] = carry;
        }

        // uint32_t carry=0;
        // uint32_t inner_carry=0;
        // #pragma unroll
        // for (int i = 0; i < DEC_LEN; i++)
        // {
        //     carry = 0;
        //     #pragma unroll
        //     for (int j = 0; j < DEC_LEN; j++)
        //     {
        //         inner_carry = 0;
        //         asm  volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j]));
        //         asm  volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j+1]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j+1]));
        //         // 将 CC.CF 的值提取出来并置为零
        //         asm  volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(inner_carry) : "r"(0), "r"(0));
        //         // inner_res 加 carry 并将进位写入 CC.CF
        //         asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(inner_res[i+j+1]) : "r"(inner_res[i+j+1]), "r"(carry));
        //         // 将 CC.CF 的值提取到 carry 上 并加上 inner_carry 同时令 CC.CF 为 0
        //         asm  volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(carry) : "r"(0), "r"(inner_carry));
        //     }
        //     inner_res[i + DEC_LEN] = carry;
        // }

        // uint32_t tmp_carry = 0;
        // uint32_t grp_carry = 0;
        // asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(tmp_carry) : "r"(0), "r"(0));
        // #pragma unroll
        // for (int i=0; i < DEC_LEN; i++){
        //     int j = 0;
        //     for(j=0; j< DEC_LEN; j+=2){
        //         asm  volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j]));
        //         asm  volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j+1]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j+1]));
        //     }

        //     // 把当前的进位暂存
        //     asm  volatile ("addc.u32 %0, %1, %2;" : "=r"(tmp_carry) : "r"(0), "r"(0));
        //     // 添加上一组留下的进位
        //     asm  volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(inner_res[i+j-1]) : "r"(inner_res[i+j-1]), "r"(grp_carry));
        //     // 将两个进位相加就是要传递给下一个组的进位 且 由于 grp_carry 和 tmp_carry <= 1 所以执行 CC.CF = 0
        //     asm  volatile ("addc.u32 %0, %1, %2;" : "=r"(grp_carry) : "r"(tmp_carry), "r"(0));
        // }

        // #pragma unroll
        // for (int i=0; i < DEC_LEN; i++){
        //     int j = 0;
        //     for(j=1; j< DEC_LEN; j+=2){
        //         asm  volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j]));
        //         asm  volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j+1]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j+1]));
        //     }
        //     // 把当前的进位暂存
        //     asm  volatile ("addc.u32 %0, %1, %2;" : "=r"(tmp_carry) : "r"(0), "r"(0));
        //     // 添加上一组留下的进位
        //     asm  volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(inner_res[i+j-1]) : "r"(inner_res[i+j-1]), "r"(grp_carry));
        //     // 将两个进位相加就是要传递给下一个组的进位 且 由于 grp_carry 和 tmp_carry <= 1 所以执行 CC.CF = 0
        //     asm  volatile ("addc.u32 %0, %1, %2;" : "=r"(grp_carry) : "r"(tmp_carry), "r"(0));
        // }

        #pragma unroll
        for (int i = DEC_LEN; i >= 0; i--)
        {
            v[i] = inner_res[i];
        }

        return *this;
    }

    friend ARIES_DEVICE AriesDecimal operator*(const AriesDecimal &left, int32_t right)
    {
        AriesDecimal tmp(right);
        return tmp *= left;
    }

    friend ARIES_DEVICE AriesDecimal operator*(const AriesDecimal &left, const AriesDecimal &right)
    {
        AriesDecimal tmp(left);
        return tmp *= right;
    }

    // for div
    ARIES_HOST_DEVICE_NO_INLINE int32_t GetPowers10(int i) const{
        int32_t res = 1;
        switch (i) {
            case 0:
                res = 1;
                break;
            case 1:
                res = 10;
                break;
            case 2:
                res = 100;
                break;
            case 3:
                res = 1000;
                break;
            case 4:
                res = 10000;
                break;
            case 5:
                res = 100000;
                break;
            case 6:
                res = 1000000;
                break;
            case 7:
                res = 10000000;
                break;
            case 8:
                res = 100000000;
                break;
            case 9:
                res = 1000000000;
                break;
            default:
                break;
        }
        return res;
    }

    friend ARIES_DEVICE bool operator<(const AriesDecimal &left, int32_t right){
        AriesDecimal tmp(right);
        return left < tmp;
    }

    friend ARIES_DEVICE bool operator>(const AriesDecimal &left, int32_t right){
        AriesDecimal tmp(right);
        return left > tmp;
    }

    friend ARIES_DEVICE bool operator<=(const AriesDecimal &left, int32_t right){
        AriesDecimal tmp(right);
        return !(left > tmp);
    }

    friend ARIES_DEVICE bool operator>=(const AriesDecimal &left, int32_t right){
        AriesDecimal tmp(right);
        return !(left < tmp);
    }

    friend ARIES_DEVICE bool operator<( int32_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return left < tmp;
    }

    friend ARIES_DEVICE bool operator>( int32_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return left > tmp;
    }

    friend ARIES_DEVICE bool operator<=( int32_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return !(left > tmp);
    }

    friend ARIES_DEVICE bool operator>=( int32_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return !(left < tmp);
    }

    friend ARIES_DEVICE bool operator<( int8_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return left < tmp;
    }

    friend ARIES_DEVICE bool operator>( int8_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return left > tmp;
    }

    friend ARIES_DEVICE bool operator<=( int8_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return !(left > tmp);
    }

    friend ARIES_DEVICE bool operator>=( int8_t left, const AriesDecimal & right){
        AriesDecimal tmp(left);
        return !(left < tmp);
    }

    friend ARIES_DEVICE bool operator<(const AriesDecimal &left, const AriesDecimal &right) {
        long long temp;
        if(left.sign != right.sign){
            //符号不同
            if(left.sign == 0){
                return false;
            }
            return true;
        }
        else{
            //符号相同
            AriesDecimal l(left);
            AriesDecimal r(right);
            if( l.frac != r.frac){
                l.AlignAddSubData(r);
            }
            if( left.sign == 0){
                #pragma unroll
                for (int i = DEC_LEN - 1; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp < 0;
                    }
                }
            }
            else{
                #pragma unroll
                for (int i = DEC_LEN - 1; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp > 0;
                    }
                }
            }
        }
        return false;
    }

    friend ARIES_DEVICE bool operator==( const AriesDecimal& left, const AriesDecimal& right ){
        if( left.sign != right.sign ){
            return false;
        }
        AriesDecimal l(left);
        AriesDecimal r(right);
        if( l.frac != r.frac ){
            l.AlignAddSubData(r);
        }
        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++) {
            if (l.v[i] != r.v[i]) {
                return false;
            }
        }
        return true;
    }

    friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const AriesDecimal& left, const AriesDecimal& right ){
        return !(left < right);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint64_t ToUint64() const{
        uint64_t res = v[0] + (uint64_t)v[1] * PER_DEC_MAX_SCALE;
        return res;
    }

    ARIES_DEVICE AriesDecimal& DivByInt64(const AriesDecimal &divisor, int shift, bool isMod = false){
        //被除数 在 int64 范围内
        uint64_t dvs = ToUint64();
        while (shift > DIG_PER_INT32) {
            dvs *= GetPowers10(DIG_PER_INT32);
            shift -= DIG_PER_INT32;
        }
        dvs *= GetPowers10(shift);
        //被除数
        uint64_t dvt = divisor.ToUint64();
        uint64_t res = isMod ? (dvs % dvt) : (dvs / dvt + (((dvs % dvt) << 1) >= dvt ? 1 : 0));

        v[1] = res / PER_DEC_MAX_SCALE;
        v[0] = res % PER_DEC_MAX_SCALE;
        return *this;
    }

    ARIES_DEVICE AriesDecimal& DivByInt(const AriesDecimal &d, int shift, bool isMod = false){
        //存放除数
        uint32_t dvs = d.v[0];
        //存放余数
        uint32_t remainder = 0;
        //左移 shift 位数
        *this << shift;
        for (int i = DEC_LEN-1; i >=0; i--) 
        {
            if (v[i] || remainder) 
            {
                uint64_t tmp = (uint64_t) v[i] + (uint64_t) remainder * PER_DEC_MAX_SCALE;
                v[i] = tmp / dvs;
                remainder = tmp % dvs;
            }
        }
        //四舍五入
        if (isMod) {
            // *this = remainder;
        } else {
            if(remainder*2>=dvs){
                asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]), "r"(1));
                #pragma unroll
                for(int32_t i = 1; i < DEC_LEN; i++)
                    asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(v[i]), "r"(0));
            }
        }
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool isZero() const{
        for (int i = 0; i < DEC_LEN; ++i) {
            if (v[i] != 0) {
                return false;
            }
        }
        return true;
    }

    ARIES_DEVICE AriesDecimal HalfIntDecimal(const AriesDecimal d1, const AriesDecimal d2){
        AriesDecimal tmp(d1);
        tmp += d2;
        uint32_t rds = 0;
        uint64_t t[DEC_LEN];
        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++) {
            t[i] = tmp.v[i];
        }
        //此时t[i]中存放qmax+qmin的值

        //修正代码
        #pragma unroll
        for (int i = DEC_LEN-1; i>=0 ; i--) {
            if (rds) {
                t[i] += rds * PER_DEC_MAX_SCALE;
            }
            if (t[i]) {
                rds = t[i] % 2;
                t[i] /= 2;
            }
        }
        //上述过程将qmax+qmin的值除以2，并将值存放于t[i]之中
        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++) {
            tmp.v[i] = t[i];
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE void CopyValue(AriesDecimal &d){
        #pragma unroll
        for (int i = 0; i < DEC_LEN; ++i) {
            v[i] = d.v[i];
        }
    }

    ARIES_DEVICE AriesDecimal DivInt(const AriesDecimal ds, const AriesDecimal dt, AriesDecimal &residuel, uint32_t dtHitBit) {

        //被除数为0
        if (ds.isZero()) {
            aries_memset(residuel.v, 0x00, sizeof(residuel.v));
            return ds;
        }
        //被除数小于除数
        if(ds < dt){
            residuel = ds;
            AriesDecimal res(0);
            return res;
        }
        //通过二进制的方法得出数据上下限
        uint32_t dsz = DEC_LEN-1;
        while(ds.v[dsz]==0)    
            dsz--;
        // 被除数 的最高位的 bit 所在的位置
        uint32_t dsHitBit = 0;
        asm volatile ("bfind.u32 %0, %1;" : "=r"(dsHitBit) : "r"(ds.v[dsz]) );
        dsHitBit++;
        // 被除数的最高位 是 dsHitBit
        dsHitBit += dsz * PER_INT_MAX_BIT;

        // 被除数 最高位的 bit 所在位置是 dsHitBit 所以被除数表示的范围是 [ 2^(dsHitBit-1) , 2^dsHitBit )
        // 除数的 最高位的 bit 所在位置是 dtHitBit 所以除数表示的范围是 [ 2^(dtHitBit-1) , 2^dtHitBit )
        // 上限 2^dsHitBit / 2^(dtHitBit-1) -> 2^(dsHitBit-dtHitBit+1)
        // 下限 2^(dsHitBit-1) / 2^dtHitBit -> 2^(dsHitBit-dtHitBit-1)
        // 所以 上限是 pmax 下限是 pmin
        int32_t pmax = dsHitBit - dtHitBit + 1;
        int32_t pmin = dsHitBit - dtHitBit - 1;
        if(pmin < 0)
            pmin = 0;

        // 根据上下限构造数字
        AriesDecimal qmax, qmin, qmid, restmp;
        // 为 qmax 赋值
        int pmax_index = pmax/PER_INT_MAX_BIT;
        qmax.v[pmax_index] = 1;
        qmax.v[pmax_index] <<= (pmax%PER_INT_MAX_BIT);
        // 为 qmin 赋值
        int pmin_index = pmin/PER_INT_MAX_BIT;
        qmin.v[pmin_index] = 1;
        qmin.v[pmin_index] <<= (pmin%PER_INT_MAX_BIT);
        
        // 采用二分法求值
        while (qmin < qmax) 
        {
            //取中值
            qmid=HalfIntDecimal(qmax, qmin);
            //比较 qmid 和 qmin 的大小。若相等说明结果找到 就是qmid
            if (qmid == qmin) {
                break;
            }
            //计算 中值 * 除数 与 被除数进行比较
            restmp = qmid * dt;
            //如果 rsdtmp == ds说明被整除，直接返回
            if (restmp == ds) {
                qmin.CopyValue(qmid);
                break;
            }else if (restmp < ds) {
               //如果为小于被除数，说明 商 在qmid ~ qmax 区间
               qmin = qmid;
            }
            else {
               //如果为大于被除数，说明 商 在qmin ~ qmid 区间
               qmax = qmid;
            }
        }
        residuel = ds - qmin * dt;
        return qmin;
    }

    //DivOrMod函数 zzh
    ARIES_DEVICE AriesDecimal& DivOrMod( const AriesDecimal &d, bool isMod = false ) {
        // 被除数 和 除数
        AriesDecimal divitend(*this);
        AriesDecimal divisor(d);

        // 对符号位进行判断
        sign = d.sign ^ sign;

        // 判断是否为mod
        if (isMod){

        }
        else{
            //除法操作 计算精度 遵循old.cu的精度 被除数的精度 + DIV_FIX_INNER_FRAC
            frac += DIV_FIX_EX_FRAC;
        }
         
        // 被除数为0，返回被除数，return *this返回的是对象本身
        if (isZero()){
            sign = 0;
            prec = frac + 1;
            return *this;
        }

        // 除数为零时，error 标志位
        if(d.isZero()){
            error = ERR_DIV_BY_ZERO;
            return *this;
        }

        // 因为保留了 被除数的精度 + DIV_FIX_INNER_FRAC 个 10 进制位， 所以右移动 除数的精度 + DIV_FIX_INNER_FRAC 个 10 进制位
        uint32_t shift = divisor.frac + DIV_FIX_EX_FRAC;

        divitend.frac = 0;
        divisor.frac = 0;
        divitend.sign = 0;
        divisor.sign = 0;

        //用res来存储计算结果
        AriesDecimal res;

        uint32_t dtz = DEC_LEN-1;
        uint32_t dsz = DEC_LEN-1;
        while(divitend.v[dtz]==0)   
            dtz--;
        while(divisor.v[dsz]==0)    
            dsz--;

        // 被除数 的最高位的 bit 所在的位置 第一个位置是 0
        uint32_t hitDtBit = 0;
        asm volatile ("bfind.u32 %0, %1;" : "=r"(hitDtBit) : "r"(divitend.v[dtz]) );
        hitDtBit++;
        hitDtBit += dtz * PER_INT_MAX_BIT;

        // 左移 shift 位 相当于 被除数 乘以 shift 的 10 次方
        // 需要向左偏移的 shiftVal 的最高位
        // 此处 十进制的 10 100 1000 分别在 二进制下的位数是 4 3 3
        uint32_t hitShiftBit = (shift / 3)*10;
        if(shift % 3 == 1)
            hitShiftBit += 4;
        else if(shift % 3 == 2)
            hitShiftBit += 7;

        // 除数 的最高位的 bit 所在的位置
        uint32_t hitDsBit = 0;
        asm volatile ("bfind.u32 %0, %1;" : "=r"(hitDsBit) : "r"(divisor.v[dsz]) );
        hitDsBit++;
        hitDsBit += dsz * PER_INT_MAX_BIT;

        // 对于unsign long类型数据的处理,这里的被除数都应该是unsign long类型数据
        // 被除数占用了hitDtBit位 最大值为 2^hitDtBit-1 左移的十进制数占用了 hitShiftBit 最大值位数为 2^hitShiftBit-1 可以表示为 2^(hitShiftBit-1)*1 + 2^(hitShiftBit-2)*1 or 0 + ……
        // 所以被除数左移后最高位 在 hitDtBit + hitShiftBit - 1  这个位置上
        if( hitDtBit + hitShiftBit - 1 <= PER_INT_MAX_BIT*2 && hitDsBit <= PER_INT_MAX_BIT*2){
            res = divitend.DivByInt64(divisor, shift, isMod);
        }  
        //对于unsign int类型数据的处理,这里的除数都应该是unsign int类型数据
        else if(dsz==0){
            res=divitend.DivByInt(divisor,shift,isMod);
        }
        else {
            //二分计算;
            //待左移的量
            int tmpEx = shift;
            //左移量 因为不能一下子完成左移 可能需要分多次
            int nDigits = 0;
            //tmpRes保存中间结果
            AriesDecimal tmpRes;
            for (; tmpEx > 0;) {
                //这样能算出一次性最多左移的大小-1此处遵循old
                // 最大可左移的 10 的倍数
                // 这里缩小范围  一个 Uint32 最大可以左移 9
                nDigits = (DEC_LEN - 1 - dtz + 1) * DIG_PER_INT32;
                uint32_t tmp = divitend.v[dtz];
                // 那么最大可 左移（十进制）的数量是 没用到的uint32 * 9 + 9 -用到的第一个 uint32 已用的十进制空间
                while( tmp > 0){
                    tmp /= 10;
                    nDigits--;
                }
                //可左移的量 比 待左移的量大
                if (nDigits > tmpEx ) {
                    nDigits = tmpEx;
                }
                //此次左移 nDigits 
                tmpEx -= nDigits;
                //左移
                divitend << nDigits;
                //除法
                tmpRes = DivInt(divitend, divisor, divitend, hitDsBit);
                if (!res.isZero()) {
                    //res左移nDigits位
                    res << nDigits;
                }
                res += tmpRes;
            }   
            // 四舍五入
            if (isMod) {
                res = divitend;
            } else {
                //进行四舍五入
                AriesDecimal doubleDivitend = divitend + divitend;
                if ( doubleDivitend >= divisor ) {
                    asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(res.v[0]) : "r"(res.v[0]), "r"(1));
                    #pragma unroll
                    for(int32_t i = 1; i < DEC_LEN; i++)
                        asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(res.v[i]) : "r"(res.v[i]), "r"(0));
                }
            }
        }
        CopyValue(res);
        return *this;
    }

    ARIES_DEVICE AriesDecimal &operator/=(const AriesDecimal &d) {
        return DivOrMod(d);
    }

    friend ARIES_DEVICE AriesDecimal operator/(const AriesDecimal &left, const AriesDecimal &right) {
        AriesDecimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE AriesDecimal(int8_t t) {
        // 默认精度
        initialize(10, 0, 0);
        v[0] = t;
        if(t<0){
            sign = 1;
            v[0] = -v[0];
        }
    }

    ARIES_DEVICE AriesDecimal(int32_t t) {
        // 默认精度
        initialize(10, 0, 0);
        v[0] = t;
        if(t<0){
            sign = 1;
            v[0] = -v[0];
        }
    }

    ARIES_DEVICE AriesDecimal& ModByInt(const AriesDecimal &d){
        //存放除数
        uint32_t dvs = d.v[0];
        //存放余数
        uint32_t remainder = 0;

        #pragma unroll
        for (int i = DEC_LEN-1; i >=0; i--) 
        {
            if (v[i] || remainder) 
            {
                uint64_t tmp = (uint64_t) v[i] + (uint64_t) remainder * PER_DEC_MAX_SCALE;
                v[i] = tmp / dvs;
                remainder = tmp % dvs;
            }
        }
 
        v[0] = remainder;
        #pragma unroll
        for(int32_t i=1; i<DEC_LEN; i++){
            v[i] = 0;
        }

        return *this;
    }

    ARIES_DEVICE AriesDecimal &operator+=(int8_t i) {
        AriesDecimal d(i);
        return *this += d;
    }

    ARIES_DEVICE AriesDecimal& ModByInt64(const AriesDecimal &divisor){
        //被除数 在 int64 范围内
        uint64_t dvs = ToUint64();

        //被除数
        uint64_t dvt = divisor.ToUint64();
        uint64_t res = dvs % dvt;
        
        v[1] = res / PER_DEC_MAX_SCALE;
        v[0] = res % PER_DEC_MAX_SCALE;
        return *this;
    }

    ARIES_DEVICE AriesDecimal& ModCalc( const AriesDecimal &d ) {
        // 被除数 和 除数
        AriesDecimal divitend(*this);
        AriesDecimal divisor(d);

        // 对符号位进行判断
        sign = d.sign ^ sign;
        prec = d.prec;
        frac = 0;

        // 被除数为0，返回被除数，return *this返回的是对象本身
        if (isZero()){
            sign = 0;
            prec = frac + 1;
            return *this;
        }

        divitend.frac = 0;
        divisor.frac = 0;
        divitend.sign = 0;
        divisor.sign = 0;

        //用res来存储计算结果
        AriesDecimal res(0);

        uint32_t dtz = DEC_LEN - 1;
        uint32_t dsz = DEC_LEN - 1;
        while(divitend.v[dtz]==0)   
            dtz--;
        while(divisor.v[dsz]==0)
            dsz--;

        // 被除数 的最高位的 bit 所在的位置 第一个位置是 0
        uint32_t hitDtBit = 0;
        asm volatile ("bfind.u32 %0, %1;" : "=r"(hitDtBit) : "r"(divitend.v[dtz]) );
        hitDtBit++;
        hitDtBit += dtz * PER_INT_MAX_BIT;

        // 除数 的最高位的 bit 所在的位置
        uint32_t hitDsBit = 0;
        asm volatile ("bfind.u32 %0, %1;" : "=r"(hitDsBit) : "r"(divisor.v[dsz]) );
        hitDsBit++;
        hitDsBit += dsz * PER_INT_MAX_BIT;

        // 对于unsign long类型数据的处理,这里的被除数都应该是unsign long类型数据
        if( hitDtBit <= PER_INT_MAX_BIT*2 && hitDsBit <= PER_INT_MAX_BIT*2){
            res = divitend.ModByInt64(divisor);
        }  
        //对于unsign int类型数据的处理,这里的除数都应该是unsign int类型数据
        else if(dsz==0){
            res=divitend.ModByInt(divisor);
        }
        else {
            //除法
            DivInt(divitend, divisor, divitend, hitDsBit);
            res = divitend;
        }
        CopyValue(res);
        return *this;
    }

    ARIES_DEVICE AriesDecimal &operator%=(const AriesDecimal &d) {
        return ModCalc(d);
    }

    friend ARIES_DEVICE AriesDecimal operator%(const AriesDecimal &left, const AriesDecimal &right) {
        AriesDecimal tmp(left);
        return tmp %= right;
    }

    friend ARIES_DEVICE bool operator>(const AriesDecimal &left, const AriesDecimal &right) {
        long long temp = 0;
        if(left.sign != right.sign){
            //符号不同
            if(left.sign == 0){
                return true;
            }
            return false;
        }
        else{
            //符号相同
            AriesDecimal l(left);
            AriesDecimal r(right);
            if( l.frac != r.frac){
                l.AlignAddSubData(r);
            }
            if( left.sign == 0){
                #pragma unroll
                for (int i = DEC_LEN - 1; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp > 0;
                    }
                }
            }
            else{
                #pragma unroll
                for (int i = DEC_LEN - 1; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp < 0;
                    }
                }
            }
        }
        return false;
    }
    
    friend ARIES_DEVICE bool operator<=(const AriesDecimal &left, const AriesDecimal &right) {
        return !(left > right);
    }

    ARIES_DEVICE operator bool() const {
        for (int i = 0; i < DEC_LEN; ++i) {
           if (v[i] != 0) {
               return true;
           }
       }
       return false;
    }
};

END_ARIES_ACC_NAMESPACE
