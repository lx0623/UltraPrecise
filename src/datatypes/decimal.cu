/*
 * decimal.cxx
 *
 *  Created on: 2019年6月26日
 *      Author: david
 */
#include "decimal.hxx"
#include <cassert>
#include <cmath>
#include "AriesDataTypeUtil.hxx"
#include "./xmp_utils/core.cu"
#include "./xmp_utils/arith.h"
#include "./LEN_4_TPI_4/core.cu"
#include "./LEN_4_TPI_4/arith.h"
#include "./LEN_8_TPI_4/core.cu"
#include "./LEN_8_TPI_4/arith.h"
#include "./LEN_16_TPI_4/core.cu"
#include "./LEN_16_TPI_4/arith.h"
#include "./LEN_32_TPI_4/core.cu"
#include "./LEN_32_TPI_4/arith.h"

#include "./LEN_8_TPI_8/core.cu"
#include "./LEN_8_TPI_8/arith.h"
#include "./LEN_16_TPI_8/core.cu"
#include "./LEN_16_TPI_8/arith.h"
#include "./LEN_32_TPI_8/core.cu"
#include "./LEN_32_TPI_8/arith.h"

#include "./LEN_16_TPI_16/core.cu"
#include "./LEN_16_TPI_16/arith.h"
#include "./LEN_32_TPI_16/core.cu"
#include "./LEN_32_TPI_16/arith.h"

#include "./LEN_32_TPI_32/core.cu"
#include "./LEN_32_TPI_32/arith.h"

BEGIN_ARIES_ACC_NAMESPACE

__device__ __managed__  uint32_t __ARRAY_SCALE[] = {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5};

__device__ __managed__  uint32_t __POW10_ARRAY[][NUM_TOTAL_DIG]={
    {0x0000000a},
    {0x00000064},
    {0x000003e8},
    {0x00002710},
    {0x000186a0},
    {0x000f4240},
    {0x00989680},
    {0x05f5e100},
    {0x3b9aca00},
    {0x540be400,0x00000002},
    {0x4876e800,0x00000017},
    {0xd4a51000,0x000000e8},
    {0x4e72a000,0x00000918},
    {0x107a4000,0x00005af3},
    {0xa4c68000,0x00038d7e},
    {0x6fc10000,0x002386f2},
    {0x5d8a0000,0x01634578},
    {0xa7640000,0x0de0b6b3},
    {0x89e80000,0x8ac72304},
    {0x63100000,0x6bc75e2d,0x00000005},
    {0xdea00000,0x35c9adc5,0x00000036},
    {0xb2400000,0x19e0c9ba,0x0000021e},
    {0xf6800000,0x02c7e14a,0x0000152d},
    {0xa1000000,0x1bcecced,0x0000d3c2},
    {0x4a000000,0x16140148,0x00084595},
    {0xe4000000,0xdcc80cd2,0x0052b7d2},
    {0xe8000000,0x9fd0803c,0x033b2e3c},
    {0x10000000,0x3e250261,0x204fce5e},
    {0xa0000000,0x6d7217ca,0x431e0fae,0x00000001},
    {0x40000000,0x4674edea,0x9f2c9cd0,0x0000000c},
    {0x80000000,0xc0914b26,0x37be2022,0x0000007e},
    {0x00000000,0x85acef81,0x2d6d415b,0x000004ee},
    {0x00000000,0x38c15b0a,0xc6448d93,0x0000314d},
    {0x00000000,0x378d8e64,0xbead87c0,0x0001ed09},
    {0x00000000,0x2b878fe8,0x72c74d82,0x00134261},
    {0x00000000,0xb34b9f10,0x7bc90715,0x00c097ce},
    {0x00000000,0x00f436a0,0xd5da46d9,0x0785ee10},
    {0x00000000,0x098a2240,0x5a86c47a,0x4b3b4ca8},
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

#define FIX_INTG_FRAC_ERROR(len, intg1, frac1, error)       \
    do                                                      \
    {                                                       \
        if (intg1+frac1 > (len))                            \
        {                                                   \
            if (intg1 > (len))                              \
            {                                               \
                intg1=(len);                                \
                frac1=0;                                    \
                error=ERR_OVER_FLOW;                        \
            }                                               \
            else                                            \
            {                                               \
                frac1=(len)-intg1;                          \
                error=ERR_TRUNCATED;                        \
            }                                               \
        }                                                   \
        else                                                \
        {                                                   \
            error=ERR_OK;                                   \
        }                                                   \
    } while(0)

#define FIX_TAGET_INTG_FRAC_ERROR(len, intg1, frac1, error) \
    do                                                      \
    {                                                       \
        if (intg1+frac1 > (len))                            \
        {                                                   \
            if (frac1 > (len))                              \
            {                                               \
                intg1=(len);                                \
                frac1=0;                                    \
                error=ERR_OVER_FLOW;                        \
            }                                               \
            else                                            \
            {                                               \
                intg1=(len)-frac1;                          \
                error=ERR_TRUNCATED;                        \
            }                                               \
        }                                                   \
        else                                                \
        {                                                   \
            error=ERR_OK;                                   \
        }                                                   \
    } while(0)

#define SET_PREC_SCALE_VALUE(t, d0, d1, d2) (t = (d1 != d2 ? d1 * DIG_PER_INT32 : d0))

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal() : Decimal(DEFAULT_PRECISION, DEFAULT_SCALE) {}

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale) : Decimal(precision, scale, (uint32_t) ARIES_MODE_EMPTY) {
    }

    //构造函数,传入精度信息,m无用
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, uint32_t m) {
        //初始化精度信息
        initialize(precision, scale, 0);
    }

    //构造函数，传入精度和字符串，转入另一个构造函数，加入m，m无用
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, const char s[]) : Decimal( precision, scale, ARIES_MODE_EMPTY, s) {
    }

    //根据精度和字符串构造相应decimal
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, uint32_t m, const char s[] ) {
        //对 this 初始化精度信息
        initialize(precision, scale, 0);
        //根据字符串 s 所提供的信息，构造 Decimal d，此时 d 的精度信息为初始精度信息
        Decimal d(s);
        //将 d 的内容根据对 this 精度信息要求的精度信息做出 截断操作，并返回给 this
        cast(d);
    }

    //CompactDecimal to Decimal，根据 compact 和 精度信息完成构造，m无用 
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const CompactDecimal *compact, uint32_t precision, uint32_t scale, uint32_t m) {
        initialize(precision, scale,0);
        int len = GetDecimalRealBytes(precision,scale);
        aries_memcpy((char *)(v), compact->data , len);
        char *temp = ((char *)(v));
        temp += len - 1;
        sign = GET_SIGN_FROM_BIT(*temp);
        *temp = *temp & 0x7f;
    }

    //构造函数，传入字符串s
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char s[]) {
        //初始化精度信息
        initialize(0, 0, 0);
        //将字符串转化为Decimal
        bool success = StringToDecimal((char *) s);
    }

    //构造函数，根据传入字符串s和字符串长度构造decimal
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal( const char* s, int len )
    {
        //初始化精度信息
        initialize(0, 0, 0);
        //将字符串转化为Decimal
        bool success = StringToDecimal((char *) s);
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char s[], int len, bool a) {
        //初始化精度信息
        initialize(0, 0, 0);
        //第一个字符表示符号
        if(s[0]=='+')
            sign = 0;
        else
            sign = 1;
        //后五个字符表示frac (xxx)
        int t = len - 2;
        for(int i=t-2; i<=t ; i++){
            frac = frac * 10 + (s[i]-'0');
        }
        t = len - 5;
        //之后每8个char代表一个数 len中1个char表示sign 5个char表示frac
        for(int i=0; i<(len-6)/8; i++){
            t -= 8;
            for(int j=0; j<8; j++){
                v[i] = v[i]*16;
                if( s[t+j] <= '9' ){
                    v[i]+=s[t+j]-'0';
                }
                else if( s[t+j] >= 'A' ){
                    v[i]+=s[t+j]-'A'+10;
                }
                else{
                   error = 1;
                }
            }
        }
        //这里向上估算 一个uint32十进制下位数为10
        prec = ((len-6)/8) * 10;
    }

    //构造函数， int32_t t 并让其 frac = scale
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int32_t t, int32_t scale, bool a, bool b, bool c) {
        int prection = INT_PRECISION;
        if( scale>prection )
            prection += scale;
        initialize(prection, scale, ARIES_MODE_EMPTY);
        if(t<0){
            sign = 1;
            t = - t;
        }
        v[0] = t;
        while( scale>DIG_PER_INT32 ){
            scale -= DIG_PER_INT32;
            int64_t tmp = 0;
            int32_t carry = 0;
            for(int i=0 ; i<NUM_TOTAL_DIG ; i++){
                tmp = (int64_t)v[i] * MAX_BASE10_PER_INT + carry ;
                carry = tmp / PER_DEC_MAX_SCALE;
                v[i] = tmp % PER_DEC_MAX_SCALE; 
            }
        }
        if( scale!=0 ){
            int32_t pow = GetPowers10(scale);
            int64_t tmp = 0;
            int32_t carry = 0;
            for(int i=0 ; i<NUM_TOTAL_DIG ; i++){
                tmp = (int64_t)pow * v[i] + carry ;
                carry = tmp / PER_DEC_MAX_SCALE;
                v[i] = tmp % PER_DEC_MAX_SCALE; 
            }
        }
    }

    //构造函数， Decimal t 并让其 frac = scale
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(Decimal t, int32_t scale, bool a, bool b, bool c, bool d) {
        int32_t gap = scale - t.frac;
        *this = t<<gap;
        frac = scale;
        prec = t.prec+gap;
    }

    //构造函数， int8_t
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int8_t t) {
        initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        v[0] = t;
        if(t<0){
            sign = 1;
            v[0] = -v[0];
        }
    }

    //构造函数， int16_t
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int16_t t) {
        initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        v[0] = t;
        if(t<0){
            sign = 1;
            v[0] = -v[0];
        }
    }

    //构造函数， int32_t
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int32_t t) {
        initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        if(t<0){
            sign = 1;
            t = - t;
        }
		v[0] = t % PER_DEC_MAX_SCALE;
    }

    //构造函数， int64_t
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int64_t t) {
        initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        if(t<0){
            sign = 1;
            t = - t;
        }
        v[1] = t / PER_DEC_MAX_SCALE;
        v[0] = t % PER_DEC_MAX_SCALE;
    }

    //构造函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint8_t t) {
        initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        v[0] = t;
    }

    //构造函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint16_t t) {
        initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        v[0] = t;
    }

    //构造函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t t) {
        initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        v[0] = t % PER_DEC_MAX_SCALE;
    }

    //构造函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint64_t t) {
        initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        v[1] = t / PER_DEC_MAX_SCALE;
        v[0] = t % PER_DEC_MAX_SCALE;
    }

    //Decimal To CompactDecimal
    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::ToCompactDecimal(char * buf, int len) {
        aries_memcpy(buf, (char *)(v),len);
        SET_SIGN_BIT(buf[len - 1], sign);
        return true;
    }

    //获取精度信息，将prec和frac打包成字符串返回
	ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetPrecisionScale(char result[]) {
        char temp[8];
		int prec0 = prec;
		int frac0 = frac;
        aries_sprintf(temp, "%d", prec0);
        aries_strcpy(result, temp);
        aries_strcat(result, ",");
        aries_sprintf((char *) temp, "%d", frac0);
        aries_strcat(result, temp);
        return result;
    }

    //返回错误代码
    ARIES_HOST_DEVICE_NO_INLINE uint16_t Decimal::GetError() {
        return error;
    }

    //将 decimal 放入字符串 result 中输出
    ARIES_HOST_DEVICE_NO_INLINE char * Decimal::GetDecimal(char result[]) {
        char numberDict[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
        int index = 0;
        if( sign==0 ){
            result[index]='+';
        }
        else{
            result[index]='-';
        }
        int flag = 0;
        uint32_t tmp = 0;
        #pragma unroll
        for(int i=INDEX_LAST_DIG ; i>=0; i--){
            if( v[i] != 0 ){
                flag = 1;
            }
            if( flag ==1 ){
                tmp = v[i];
                index += 8;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    result[index-j] = numberDict[tmp&0xF];
                    tmp >>= 4;
                }
            }
        }
        index++;
        result[index++] = '(';
        result[index++] = frac / 100 + '0';
        result[index++] = frac / 10 % 10 + '0';
        result[index++] = frac % 10 +'0';
        result[index++] = ')';
        result[index] = '\0';
        return result;
    }

    //检查溢出  此处用 double 实现的 后续查看是否需要优化
	ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckOverFlow() {
        int i = INDEX_LAST_DIG;
        for ( ; i >= 0; i--){
            if ( v[i]!= 0){
                break;
            }
        }
        int prec0 = i * DIG_PER_INT32;
        double maxUint32 = 4.294967295;
        double tmpMul = 1; 
        for(int j=0; j<i ;j++){
            tmpMul *= maxUint32;
        }
        tmpMul *= v[i];
        int tt = (int) tmpMul;
        while( tt>1 ){
            tt /= 10;
            prec0++;
        }
        // 声明位数小于实际位数
        if( prec<prec0){
            error = ERR_OVER_FLOW;
        }
	}

    /*
     * integer/frac part by pos index
     *   0: value of 0 int
     *   1: value of 1 int
     *   2: value of 2 int
     *   3: value of 3 int
     * */

    //此函数原本功能是 例如在 old decimal 中 (13,2) 的 12 345678912 560000000 调用 setIntPart(123456789,1) 变为 12 123456789 12000000
    //但是此函数只有在 AriesDatetime 中使用 , decimal 的 格式是 (17,6) v = 0 0 0 0 0, 且调用时 pos 始终为 0 , value 的值在 62167219200 ~ 64314691200
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::setIntPart(int value, int pos) {
        v[0] += value * GetPowers10(frac) % PER_DEC_MAX_SCALE;
        v[1] += value * GetPowers10(frac) / PER_DEC_MAX_SCALE;
    }

    //此函数原本功能是 例如在 old decimal 中 (13,11) 的 12 345678912 560000000 调用 setFracPart(123456789,2) 变为 12 123456789 12000000
    //但是此函数只有在 AriesDatetime 中使用 , decimal 的 格式是 (17,6) 在 setIntPart 之后执行 , 且调用时 pos 始终为 0 , value 的值在 0 ~ 999999
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::setFracPart(int value, int pos) {
        v[pos] += value;
    }

    //此函数原本功能是 例如在 old decimal 中 (13,2) 的 12 345678912 560000000 调用 getIntPart(2) 返回 小数点前面 第 pos + 1 个 int 位上的数 12
    ARIES_HOST_DEVICE_NO_INLINE int Decimal::getIntPart(int pos) const {
        return v[1]*PER_DEC_MAX_SCALE + v[0] / GetPowers10(frac);
    }

    //此函数原本功能是 例如在 old decimal 中 (13,11) 的 12 345678912 560000000 调用 getFracPart(2) 返回 倒数 第 pos + 1 个 int 位上的数 12
    //但是此函数只有在 AriesDatetime 中使用 , pos 始终为 0
    ARIES_HOST_DEVICE_NO_INLINE int Decimal::getFracPart(int pos) const {
        return v[0] % GetPowers10(frac);
    }

    //global method
    ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal) {
        decimal.sign = 0;
        return decimal;
    }

    //获取 Decimal To CompactDecimal 需要的 Byte
    ARIES_HOST_DEVICE_NO_INLINE int GetDecimalRealBytes(uint16_t precision, uint16_t scale) {
        int needBytes = precision / DIG_PER_INT32 * 4;
         switch (precision % DIG_PER_INT32) {
            case 0:
                needBytes += 0;
                break;
            case 1:
                needBytes += 1;   //4个bit < 1 个字节
                break;
            case 2:
                needBytes += 1;  //7个bit < 1 个字节
                break;
            case 3:
                needBytes += 2;  //10个bit < 2 个字节
                break;
            case 4:
                needBytes += 2;   //14个bit < 2 个字节
                break;
            case 5:
                needBytes += 3;    //17个bit <  3个字节
                break;
            case 6:
                needBytes += 3;    //20个bit < 3 个字节
                break;
            case 7:
                needBytes += 4;    //24个bit < 4 个字节
                break;
            case 8:
                needBytes += 4;    //27个bit < 4 个字节
                break;
        }
        return needBytes;
    }


   //截断函数，根据 精度信息 prec，frac 对 t.v 进行截断
	ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::cast(const Decimal &t) {
		sign = t.sign;
        // 小数部分可能需要扩充
        if(frac >= t.frac){
            // 查看小数部分
            int shift = frac - t.frac;
            // 需要扩充
            if(shift !=0 ){
                aries_memcpy(v,t.v,sizeof(t.v));
                *this << shift;
            }
            // 不需要扩充
            else{
                aries_memcpy(v,t.v,sizeof(t.v));
            }
            // 查看整数部分
            // 如果规定的规格下数值的整数部分大小 小于 其原本应该占据的部分大小,那么需要查看是否溢出
            if( (prec - frac) < (t.prec-t.frac)){
                // 需要检查正数部分是否溢出 CheckOverFlow()
                CheckOverFlow();
            }
        }
        // 
        else{
            //小数部分需要缩减
			//小数需要右移,断尾
			//向右要缩进了几个 int 单位
            int shift = t.frac - frac;
            aries_memcpy(v,t.v,sizeof(t.v));
            // 是否需要 四舍五入  如果不需要 这里可使用 *this >> shift;
            *this >> shift;
            // 如需要四舍五入
            // uint64_t temp = 0;
            // uint32_t remainder = 0;
            // while ( shift>DIG_PER_INT32){
            //     for (int i = INDEX_LAST_DIG; i>=0 ; i--){
            //         temp = remainder * PER_DEC_MAX_SCALE + v[i];
            //         v[i] = temp % PER_DEC_MAX_SCALE;
            //         remainder = temp / PER_DEC_MAX_SCALE;
            //     }
            //     shift -= DIG_PER_INT32;
            // }
            // uint32_t pow10n = GetPowers10(shift);
            // int jw = 0;
            // if( v[0] % pow10n / (pow10n/10) >= 5){
			// 	jw = 1;
			// }
            // remainder = 0;
            // for (int i = INDEX_LAST_DIG; i>=0 ; i--){
            //     temp = remainder * PER_DEC_MAX_SCALE + v[i];
            //     v[i] = temp % PER_DEC_MAX_SCALE;
            //     remainder = temp / PER_DEC_MAX_SCALE;
            // }
            // if( jw == 1 ){
            //     asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]), "r"(1));
            //     #pragma unroll
            //     for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
            //         asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(v[i]), "r"(0));
            // }
            CheckOverFlow();
        }
        return *this;
	}

    /* CalcTruncTargetPrecision
        * int p: > 0 try to truncate frac part to p scale
        *        = 0 try to truncate to integer
        *        < 0 try to truncate to integer, and intg part will be truncated
    * */
    //直接截断函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::truncate( int p ) {
        if( p>0 ){
            if( frac>=p ){
                *this>>frac-p;
                prec -= frac -p;
            }
            else{
                *this<<p-frac;
                prec += p - frac;
            }
            frac = p;
        }
        else{
            p = -p;
            // 让 frac = 0 的同时，该数字末尾是 abs(p) 个 0
            int intg = prec - frac;
            if( intg>p ){
                *this >> frac+p;
                *this << p;
                prec = intg;
            }
            else{
                aries_memset(v, 0x00, sizeof(v));
                prec = 1;
            }
            frac = 0;
        }

        return *this;
    }


    ARIES_HOST_DEVICE_NO_INLINE Decimal::operator bool() const {
        return !isZero();
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator-() {
        if(sign == 0){
            sign = 1;
        }
        else{
            sign = 0;
        }
        return *this;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int8_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int16_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int32_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int64_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint8_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint16_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint32_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint64_t t) {
        Decimal tmp(t);
        *this = tmp;
        return *this;
    }

    //for decimal
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, const Decimal &right) {
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
            Decimal l(left);
            Decimal r(right);
            if( l.frac != r.frac){
                l.AlignAddSubData(r);
            }
            if( left.sign == 0){
                #pragma unroll
                for (int i = INDEX_LAST_DIG; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp > 0;
                    }
                    // if( l.v[i] > r.v[i] ){
                    //     return true;
                    // }
                    // if( l.v[i] < r.v[i] ){
                    //     return false;
                    // }
                }
            }
            else{
                #pragma unroll
                for (int i = INDEX_LAST_DIG; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp < 0;
                    }
                }
            }
        }
        return false;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, const Decimal &right) {
        return !(left < right);
    }
    
    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, const Decimal &right) {
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
            Decimal l(left);
            Decimal r(right);
            if( l.frac != r.frac){
                l.AlignAddSubData(r);
            }
            if( left.sign == 0){
                #pragma unroll
                for (int i = INDEX_LAST_DIG; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp < 0;
                    }
                }
            }
            else{
                #pragma unroll
                for (int i = INDEX_LAST_DIG; i >= 0; i--) {
                    if( temp = (long long)l.v[i] - r.v[i] ){
                        return temp > 0;
                    }
                }
            }
        }
        return false;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, const Decimal &right) {
        return !(left > right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, const Decimal &right) {
        if( left.sign != right.sign ){
            return false;
        }
        Decimal l(left);
        Decimal r(right);
        if( l.frac != r.frac ){
            l.AlignAddSubData(r);
        }
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            if (l.v[i] != r.v[i]) {
                return false;
            }
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, const Decimal &right) {
        return !(left == right);
    }

    // for int8_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int8_t left, const Decimal &right) {
        return (int32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int8_t left, const Decimal &right) {
        return (int32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int8_t left, const Decimal &right) {
        return (int32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int8_t left, const Decimal &right) {
        return (int32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int8_t left, const Decimal &right) {
        return (int32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int8_t left, const Decimal &right) {
        return !(left == right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int8_t right) {
        return left > (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int8_t right) {
        return left >= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int8_t right) {
        return left < (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int8_t right) {
        return left <= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int8_t right) {
       return left == (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int8_t right) {
        return left != (int32_t) right;
    }

    // for uint8_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint8_t left, const Decimal &right) {
        return (uint32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint8_t left, const Decimal &right) {
        return (uint32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint8_t left, const Decimal &right) {
       return (uint32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint8_t left, const Decimal &right) {
        return (uint32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint8_t left, const Decimal &right) {
        return (uint32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint8_t left, const Decimal &right) {
        return !(left == right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint8_t right) {
        return left > (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint8_t right) {
       return left >= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint8_t right) {
        return left < (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint8_t right) {
        return left <= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint8_t right) {
        return left == (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint8_t right) {
        return left != (uint32_t) right;
    }

    //for int16_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int16_t left, const Decimal &right) {
        return (int32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int16_t left, const Decimal &right) {
        return (int32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int16_t left, const Decimal &right) {
        return (int32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int16_t left, const Decimal &right) {
        return (int32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int16_t left, const Decimal &right) {
        return (int32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int16_t left, const Decimal &right) {
        return (int32_t) left != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int16_t right) {
        return left > (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int16_t right) {
        return left >= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int16_t right) {
        return left < (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int16_t right) {
        return left <= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int16_t right) {
        return left == (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int16_t right) {
        return left != (int32_t) right;
    }

    //for uint16_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint16_t left, const Decimal &right) {
        return (uint32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint16_t left, const Decimal &right) {
        return (uint32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint16_t left, const Decimal &right) {
        return (uint32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint16_t left, const Decimal &right) {
        return (uint32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint16_t left, const Decimal &right) {
        return (uint32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint16_t left, const Decimal &right) {
        return (uint32_t) left != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint16_t right) {
        return left > (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint16_t right) {
        return left >= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint16_t right) {
        return left < (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint16_t right) {
        return left <= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint16_t right) {
        return left == (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint16_t right) {
        return left != (uint32_t) right;
    }

    //for int32_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left != d;
    }

    //for uint32_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left != d;
    }

    //for int64_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left != d;
    }

    //for uint64_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left != d;
    }

    //for float
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(float left, const Decimal &right) {
        return (double) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(float left, const Decimal &right) {
        return (double) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(float left, const Decimal &right) {
        return (double) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(float left, const Decimal &right) {
        return (double) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(float left, const Decimal &right) {
        return (double) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(float left, const Decimal &right) {
        return (double) left != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, float right) {
        return left > (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, float right) {
        return left >= (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, float right) {
        return left < (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, float right) {
        return left <= (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, float right) {
        return left == (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, float right) {
        return left != (double) right;
    }

    //for double
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(double left, const Decimal &right) {
        return left > right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(double left, const Decimal &right) {
        return left >= right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(double left, const Decimal &right) {
        return left < right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(double left, const Decimal &right) {
        return left <= right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(double left, const Decimal &right) {
        return left == right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(double left, const Decimal &right) {
        return left != right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, double right) {
        return left.GetDouble() > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, double right) {
        return left.GetDouble() >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, double right) {
        return left.GetDouble() < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, double right) {
        return left.GetDouble() <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, double right) {
        return left.GetDouble() == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, double right) {
        return left.GetDouble() != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::AddDecimalOnCpu( const Decimal &d ){
        Decimal added(d);
        added.AlignAddSubData(*this);
        uint64_t temp;
        uint32_t carry = 0;
        if(sign == added.sign){
            #pragma unroll
            for (int32_t i = 0; i < NUM_TOTAL_DIG; i++){
                temp = (uint64_t)v[i] + added.v[i] + carry;
                v[i] = temp & 0x00000000ffffffff;
                carry = (temp & 0xffffffff00000000) >> 32;
            }
        }else{
            int64_t r = 0;
            #pragma unroll
            for (int32_t i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
                r = (int64_t)v[i] - added.v[i];
                if(r != 0){
                    break;
                }
            }
            if(r >= 0 ){
                #pragma unroll
                for (int32_t i = 0; i < NUM_TOTAL_DIG; i++){
                    temp = (uint64_t)v[i] + PER_DEC_MAX_SCALE - added.v[i] - carry;
                    carry = ( temp < PER_DEC_MAX_SCALE);    // 比 PER_DEC_MAX_SCALE 小表示借位了
                    v[i] = temp & 0x00000000ffffffff;   // 对 temp 取模
                }
            }
            else{
                #pragma unroll
                for (int32_t i = 0; i < NUM_TOTAL_DIG; i++){
                    temp = (uint64_t)added.v[i] + PER_DEC_MAX_SCALE - v[i] - carry;
                    carry = ( temp < PER_DEC_MAX_SCALE);    // 比 PER_DEC_MAX_SCALE 小表示借位了
                    v[i] = temp & 0x00000000ffffffff;
                }
            }
            sign = (r > 0 && !d.sign) || (r < 0 && d.sign);
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::MulDecimalOnCpu( const Decimal &d ){
        sign = d.sign ^ sign;
        frac = frac + d.frac;

        uint32_t inner_res[NUM_TOTAL_DIG*2]={0};

        uint64_t temp;
        uint32_t carry;

        #pragma unroll
        for(int i = 0; i < NUM_TOTAL_DIG; i++){
            carry = 0;
            #pragma unroll
            for(int j = 0; j < NUM_TOTAL_DIG; j++){
                temp = (uint64_t)v[i] * d.v[j] + inner_res[i+j] + carry;
                carry = temp / PER_DEC_MAX_SCALE;
                inner_res[i+j] = temp % PER_DEC_MAX_SCALE;
            }
            inner_res[i+NUM_TOTAL_DIG] = carry;
        }

        #pragma unroll
        for(int i = INDEX_LAST_DIG; i >=0 ; i--){
            v[i] = inner_res[i];
        }
    }

    //计算加法目标精度，生成的动态代码需要此数
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcAddTargetPrecision( const Decimal& d ) {
        prec = aries_max(prec-frac,d.prec-d.frac)+1;
        frac = aries_max(frac,d.frac);
        prec += frac;
    }



    //加法函数：decimalx += decimaly 函数
    ARIES_DEVICE Decimal &Decimal::operator+=(const Decimal &d) {
		Decimal added(d);
        added.AlignAddSubData(*this);

        if(sign == added.sign){
            asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(added.v[0]), "r"(v[0]));
            #pragma unroll
            for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(added.v[i]), "r"(v[i]));
        }else{
            int64_t r = 0;
            #pragma unroll
            for (int i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
                r = (int64_t)v[i] - added.v[i];
                if(r != 0){
                    break;
                }
            }
            if(r >= 0 ){
                asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]),"r"(added.v[0]));
                #pragma unroll
                for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(v[i]), "r"(added.v[i]));
            }
            else{
                asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(added.v[0]), "r"(v[0]));
                #pragma unroll
                for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(added.v[i]), "r"(v[i]));
            }
            sign = (r > 0 && !d.sign) || (r < 0 && d.sign);
        }
        return *this;
    }

    //signed
    ARIES_DEVICE Decimal &Decimal::operator+=(int8_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_DEVICE Decimal &Decimal::operator+=(int16_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_DEVICE Decimal &Decimal::operator+=(int32_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_DEVICE Decimal &Decimal::operator+=(int64_t i) {
        Decimal d(i);
        return *this += d;
    }

    //unsigned
    ARIES_DEVICE Decimal &Decimal::operator+=(uint8_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_DEVICE Decimal &Decimal::operator+=(uint16_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_DEVICE Decimal &Decimal::operator+=(uint32_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_DEVICE Decimal &Decimal::operator+=(uint64_t i) {
        Decimal d(i);
        return *this += d;
    }

    //double / float
    ARIES_DEVICE double Decimal::operator+=(const float &f) {
        return *this += (double) f;
    }

    ARIES_DEVICE double Decimal::operator+=(const double &l) {
        return GetDouble() + l;
    }

    //self operator
    ARIES_DEVICE Decimal &Decimal::operator++() {
        Decimal d((int8_t) 1);
        *this += d;
        return *this;
    }

    ARIES_DEVICE Decimal Decimal::operator++(int32_t) {
        Decimal d((int8_t) 1);
        *this += d;
        return *this;
    }

   //加法函数，decimalx + decimaly
    ARIES_DEVICE Decimal operator+(const Decimal &left, const Decimal &right) {
        //将 const left 赋值到temp进行操作
		Decimal tmp(left);
		return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    //unsigned
    ARIES_DEVICE Decimal operator+(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_DEVICE Decimal operator+(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    //double / float
    ARIES_DEVICE double operator+(const Decimal &left, float right) {
        return left.GetDouble() + right;
    }

    ARIES_DEVICE double operator+(const Decimal &left, double right) {
        return left.GetDouble() + right;
    }

    ARIES_DEVICE double operator+(float left, const Decimal &right) {
        return left + right.GetDouble();
    }

    ARIES_DEVICE double operator+(double left, const Decimal &right) {
        return left + right.GetDouble();
    }


    //计算目标结果精度，在SQL转化为动态代码时用到
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcSubTargetPrecision(const Decimal &d) {
        prec = aries_max(prec-frac,d.prec-d.frac)+1;
        frac = aries_max(frac,d.frac);
        prec += frac;
    }

   //减法函数 decimalx -= decimaly
   ARIES_DEVICE Decimal &Decimal::operator-=(const Decimal &d) {

       Decimal added(d);
        added.AlignAddSubData(*this);

        if(added.sign != sign){
            asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(added.v[0]), "r"(v[0]));

            #pragma unroll
            for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(added.v[i]), "r"(v[i]));

        }else{
            int64_t r = 0;
            #pragma unroll
            for (int i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
                r = (int64_t)v[i] - added.v[i];
                if(r != 0){
                    break;
                }
            }
            if(r >= 0 ){
                asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]), "r"(added.v[0]));
                #pragma unroll
                for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(v[i]), "r"(added.v[i]));
            }
            else{
                asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(added.v[0]), "r"(v[0]));
                #pragma unroll
                for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(added.v[i]), "r"(v[i]));
            }
            sign = (r > 0 && added.sign) || (r < 0 && !added.sign);
        }

        return *this;
   }

    //signed
    ARIES_DEVICE Decimal &Decimal::operator-=(int8_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator-=(int16_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator-=(int32_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator-=(int64_t i) {
        Decimal d(i);
        return *this -= d;
    }

    //unsigned
    ARIES_DEVICE Decimal &Decimal::operator-=(uint8_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator-=(uint16_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator-=(uint32_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator-=(uint64_t i) {
        Decimal d(i);
        return *this -= d;
    }

    //double / float
    ARIES_DEVICE double Decimal::operator-=(const float &f) {
        return GetDouble() - f;
    }

    ARIES_DEVICE double Decimal::operator-=(const double &l) {
        return GetDouble() - l;
    }

    //self operator
    ARIES_DEVICE Decimal &Decimal::operator--() {
        Decimal d((int8_t) 1);
        return *this -= d;
    }

    ARIES_DEVICE Decimal Decimal::operator--(int32_t) {
        Decimal tmp(*this);
        Decimal d((int8_t) 1);
        return tmp -= d;
    }

    //减法函数，decimalx - decimaly
   ARIES_DEVICE Decimal operator-(const Decimal &left, const Decimal &right) {
       Decimal tmp(left);
       return tmp -= right;
   }

    ARIES_DEVICE Decimal operator-(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    //减法函数，decimalx -= decimaly
    ARIES_DEVICE Decimal operator-(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    //unsigned
    ARIES_DEVICE Decimal operator-(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_DEVICE Decimal operator-(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    //double / float
   ARIES_DEVICE double operator-(const Decimal &left, const float right) {
        return left.GetDouble() - right;
    }

    ARIES_DEVICE double operator-(const Decimal &left, const double right) {
        return left.GetDouble() - right;
    }

    ARIES_DEVICE double operator-(const float left, const Decimal &right) {
        return left - right.GetDouble();
    }

    ARIES_DEVICE double operator-(const double left, const Decimal &right) {
        return left - right.GetDouble();
    }

    //计算乘法目标精度
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcMulTargetPrecision(const Decimal &d) {
        frac = frac + d.frac;
        prec = prec + d.prec;
    }

   //乘法函数 decimalx *= decimaly
   ARIES_DEVICE Decimal &Decimal::operator*=(const Decimal &d) {
	    sign = d.sign ^ sign;
        frac = frac + d.frac;

        uint32_t inner_res[NUM_TOTAL_DIG*2]={0};

        // uint64_t temp;
        // uint32_t carry;

        // #pragma unroll
        // for(int i = 0; i < NUM_TOTAL_DIG; i++){
        //     carry = 0;
        //     #pragma unroll
        //     for(int j = 0; j < NUM_TOTAL_DIG; j++){
        //         // temp 表示范围最大值 2^64-1 右侧表达式 表示范围最大值 (2^32-1) * (2^32-1) + (2^32-1) + (2^32-1) = 2^64-1
        //         temp = (uint64_t)v[i] * d.v[j] + inner_res[i+j] + carry;
        //         carry = temp / PER_DEC_MAX_SCALE;
        //         inner_res[i+j] = temp % PER_DEC_MAX_SCALE;
        //     }
        //     inner_res[i+NUM_TOTAL_DIG] = carry;
        // }

        uint32_t carry=0;
        uint32_t inner_carry=0;
        #pragma unroll
        for (int i = 0; i < DEC_LEN; i++)
        {
            carry = 0;
            #pragma unroll
            for (int j = 0; j < DEC_LEN; j++)
            {
                inner_carry = 0;
                asm  volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j]));
                asm  volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(inner_res[i+j+1]) : "r"(v[i]), "r"(d.v[j]), "r"(inner_res[i+j+1]));
                asm  volatile ("addc.u32 %0, %1, %2;" : "=r"(inner_carry) : "r"(0), "r"(0));
                asm  volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(inner_res[i+j+1]) : "r"(inner_res[i+j+1]), "r"(carry));
                asm  volatile ("addc.u32 %0, %1, %2;" : "=r"(carry) : "r"(0), "r"(0));
                carry = carry + inner_carry;
            }
            inner_res[i + DEC_LEN] = carry;
        }

        #pragma unroll
        for(int i = INDEX_LAST_DIG; i >=0 ; i--){
            v[i] = inner_res[i];
        }

        return *this;
   }

    //signed
    ARIES_DEVICE Decimal &Decimal::operator*=(int8_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_DEVICE Decimal &Decimal::operator*=(int16_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_DEVICE Decimal &Decimal::operator*=(int32_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_DEVICE Decimal &Decimal::operator*=(int64_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    //unsigned
    ARIES_DEVICE Decimal &Decimal::operator*=(uint8_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_DEVICE Decimal &Decimal::operator*=(uint16_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_DEVICE Decimal &Decimal::operator*=(uint32_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_DEVICE Decimal &Decimal::operator*=(uint64_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    //double / float
    ARIES_DEVICE double Decimal::operator*=(const float &f) {
        return GetDouble() * f;
    }

    ARIES_DEVICE double Decimal::operator*=(const double &d) {
        return GetDouble() * d;
    }

    //乘法运算 decimalx * decimaly
   ARIES_DEVICE Decimal operator*(const Decimal &left, const Decimal &right) {
       Decimal tmp(left);
       return tmp *= right;
   }

    //signed
    ARIES_DEVICE Decimal operator*(const Decimal &left, int8_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(const Decimal &left, int16_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(const Decimal &left, int32_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(const Decimal &left, int64_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_DEVICE Decimal operator*(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_DEVICE Decimal operator*(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_DEVICE Decimal operator*(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    //unsigned
    ARIES_DEVICE Decimal operator*(const Decimal &left, uint8_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(const Decimal &left, uint16_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(const Decimal &left, uint32_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(const Decimal &left, uint64_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_DEVICE Decimal operator*(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_DEVICE Decimal operator*(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_DEVICE Decimal operator*(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_DEVICE Decimal operator*(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    //double / float
    ARIES_DEVICE double operator*(const Decimal &left, const float right) {
        return left.GetDouble() * right;
    }

    ARIES_DEVICE double operator*(const Decimal &left, const double right) {
        return left.GetDouble() * right;
    }

    ARIES_DEVICE double operator*(const float left, const Decimal &right) {
        return left * right.GetDouble();
    }

    ARIES_DEVICE double operator*(const double left, const Decimal &right) {
        return left * right.GetDouble();
    }


    //除法目标精度
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcDivTargetPrecision( const Decimal &d ) {
        prec = (prec-frac) + d.frac;
        frac = frac + DIV_FIX_EX_FRAC;
        prec += frac;
    }

    //右移 n个 10 进制位 zzh
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator>>(int n) {
        //转化为循环除法
        uint64_t temp = 0;
        uint32_t remainder = 0;
        while ( n>DIG_PER_INT32){
            for (int i = INDEX_LAST_DIG; i>=0 ; i--){
                temp = remainder * PER_DEC_MAX_SCALE + v[i];
                v[i] = temp / MAX_BASE10_PER_INT;
                remainder = temp % MAX_BASE10_PER_INT;
            }
            n -= DIG_PER_INT32;
        }
        
        uint32_t pow10n = GetPowers10(n);
        remainder = 0;
        for (int i = INDEX_LAST_DIG; i>=0 ; i--){
            temp = remainder * PER_DEC_MAX_SCALE + v[i];
            v[i] = temp / pow10n;
            remainder = temp % pow10n;
        }

        return *this;
    }

    //左移 n个 intBit
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator<<(int n) {
        if(n<=0){
            return *this;
        }
        uint32_t res[NUM_TOTAL_DIG] = {0};
		uint32_t carry = 0;
        uint64_t temp;
        #pragma unroll
        for(int i = 0; i < NUM_TOTAL_DIG; i++){
            carry = 0;
            #pragma unroll
            for(int j = 0; j < __ARRAY_SCALE[n-1]; j++){
                if( i+j > NUM_TOTAL_DIG){
                    break;
                }
                temp = (uint64_t)v[i] *  (uint64_t)__POW10_ARRAY[n-1][j] + res[i+j] + carry;
                carry = (temp & 0xffffffff00000000) >> 32;
                res[i+j] = temp & 0x00000000ffffffff;
            }
            if(i+__ARRAY_SCALE[n-1]<NUM_TOTAL_DIG){
                res[i+__ARRAY_SCALE[n-1]] = carry;
            }
        }
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++)
        {
            v[i] = res[i];
        }
        return *this;
    }

    //检查并设置位真实精度，更新精度
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckAndSetRealPrecision() {
        int i = INDEX_LAST_DIG;
        for ( ; i >= 0; i--){
            if ( v[i]!= 0){
                break;
            }
        }
        if(i==-1){
            prec = 1;
            frac = 0;
        }
        else{
            prec = i * DIG_PER_INT32;
            double maxUint32 = 4.294967295;
            double tmpMul = 1; 
            for(int j=0; j<i ;j++){
                tmpMul *= maxUint32;
            }
            tmpMul *= v[i];  
            int tt = (int) tmpMul;
            while( tt>0 ){
                tt /= 10;
                prec++;
            }
        }
    }

    //无用函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::GenIntDecimal(int shift) {
        return *this;
    }

    //取中值函数，计算 (decimalx + decimaly) / 2  zzh
    ARIES_DEVICE Decimal Decimal::HalfIntDecimal(const Decimal d1, const Decimal d2) {
        Decimal tmp(d1);
        tmp += d2;
        uint32_t rds = 0;
        uint64_t t[NUM_TOTAL_DIG];
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            t[i] = tmp.v[i];
        }
        //此时t[i]中存放qmax+qmin的值

        //修正代码
        #pragma unroll
        for (int i = INDEX_LAST_DIG; i>=0 ; i--) {
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
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            tmp.v[i] = t[i];
        }
        return tmp;
    }

    // DivInt函数，计算两个数相除 decimalx / decimal y 时使用  zzh
    // residuel 是余数
    ARIES_DEVICE Decimal Decimal::DivInt(const Decimal ds, const Decimal dt, Decimal &residuel, uint32_t dtHitBit) {

        //被除数为0
        if (ds.isZero()) {
            aries_memset(residuel.v, 0x00, sizeof(residuel.v));
            return ds;
        }
        //通过二进制的方法得出数据上下限
        uint32_t dsz = INDEX_LAST_DIG;
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
        Decimal qmax(0), qmin(0), qmid(0), restmp(0);
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

   //将 this v值转给 decimal d
   ARIES_HOST_DEVICE_NO_INLINE void Decimal::CopyValue(Decimal &d) {
       #pragma unroll
       for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
           v[i] = d.v[i];
       }
   }

    //DivOrMod函数  zzh
    ARIES_DEVICE Decimal& Decimal::DivOrMod( const Decimal &d, bool isMod ) {
        // 被除数 和 除数
        Decimal divitend(*this);
        Decimal divisor(d);

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
        Decimal res(0);

        uint32_t dtz = INDEX_LAST_DIG;
        uint32_t dsz = INDEX_LAST_DIG;
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
            Decimal tmpRes(0);
            for (; tmpEx > 0;) {
                //这样能算出一次性最多左移的大小-1此处遵循old
                // 最大可左移的 10 的倍数
                // 这里缩小范围  一个 Uint32 最大可以左移 9
                nDigits = (INDEX_LAST_DIG - dtz + 1) * DIG_PER_INT32;
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
                Decimal doubleDivitend = divitend + divitend;
                if ( doubleDivitend >= divisor ) {
                    asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(res.v[0]) : "r"(res.v[0]), "r"(1));
                    #pragma unroll
                    for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                        asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(res.v[i]) : "r"(res.v[i]), "r"(0));
                }
            }
        }
        CopyValue(res);
        return *this;
    }


    //lixin 改写按int位相除法 zzh
    ARIES_DEVICE Decimal& Decimal::DivByInt(const Decimal &d, int shift, bool isMod) {
        //存放除数
        uint32_t dvs = d.v[0];
        //存放余数
        uint32_t remainder = 0;
        //先判断是否能一次左移成功
        //待左移的量
        int tmpEx = shift;
        //因为有可能不能一下子完成左移 可能需要分多次
        int nDigits = 0;
        // 计算可左移的量
        uint32_t dtz = INDEX_LAST_DIG;
        while(v[dtz]==0)
            dtz--;
        //最大可左移的 10 的倍数
        //这里缩小范围  一个 Uint32 最大可以左移 9
        nDigits = (INDEX_LAST_DIG - dtz + 1) * DIG_PER_INT32;
        uint32_t tmp = v[dtz];
        // 那么最大可 左移（十进制）的数量是 没用到的uint32 * 9 + 9 -用到的第一个 uint32 已用的十进制空间
        while( tmp > 0){
            tmp /= 10;
            nDigits--;
        }
        if(nDigits<0){
            nDigits=0;
        }
        if (nDigits > tmpEx ) {
            nDigits = tmpEx;
        }
        tmpEx -= nDigits;
        //左移 nDigits 位数
        *this << nDigits;
        for (int i = NUM_TOTAL_DIG-1; i >=0; i--) 
        {
            if (v[i] || remainder) 
            {
                uint64_t tmp = (uint64_t) v[i] + (uint64_t) remainder * PER_DEC_MAX_SCALE;
                v[i] = tmp / dvs;
                remainder = tmp % dvs;
            }
        }
        //tmpRes保存中间结果
        Decimal tmpRes;
        while(tmpEx>0){
            tmpRes = remainder;
            // 计算下次左移位数
            dtz = INDEX_LAST_DIG;
            while(tmpRes.v[dtz]==0)
                dtz--;
            //这样能算出一次性最多左移的大小-1此处遵循old
            //最大可左移的 10 的倍数
            //这里缩小范围  一个 Uint32 最大可以左移 9
            nDigits = (INDEX_LAST_DIG - dtz + 1) * DIG_PER_INT32;
            tmp = tmpRes.v[dtz];
            // 那么最大可 左移（十进制）的数量是 没用到的uint32 * 9 + 9 -用到的第一个 uint32 已用的十进制空间
            while( tmp > 0){
                tmp /= 10;
                nDigits--;
            }
            if (nDigits > tmpEx ) {
                nDigits = tmpEx;
            }
            //此次左移 nDigits
            tmpEx -= nDigits;
            // 左移
            tmpRes << nDigits;
            remainder = 0;
            for (int i = NUM_TOTAL_DIG-1; i >=0; i--) 
            {
                if (tmpRes.v[i] || remainder) 
                {
                    uint64_t tmp = (uint64_t) tmpRes.v[i] + (uint64_t) remainder * PER_DEC_MAX_SCALE;
                    tmpRes.v[i] = tmp / dvs;
                    remainder = tmp % dvs;
                }
            }
            // 结果相加
            *this << nDigits;
            *this += tmpRes;
        }
        //四舍五入
        if (isMod) {
            // *this = remainder;
        } else {
            if(remainder*2>=dvs){
                asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]), "r"(1));
                #pragma unroll
                for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
                    asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(v[i]), "r"(0));
            }
        }
        return *this;
    }


    //int64相除
    ARIES_DEVICE Decimal& Decimal::DivByInt64(const Decimal &divisor, int shift, bool isMod) {
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

    //除法函数，decimalx /= decimaly zzh
    ARIES_DEVICE Decimal &Decimal::operator/=(const Decimal &d) {
        return DivOrMod(d);
    }

    //signed
    ARIES_DEVICE Decimal &Decimal::operator/=(int8_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator/=(int16_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator/=(int32_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator/=(int64_t i) {
        Decimal d(i);
        return *this /= d;
    }

    //unsigned
    ARIES_DEVICE Decimal &Decimal::operator/=(uint8_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator/=(uint16_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator/=(uint32_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_DEVICE Decimal &Decimal::operator/=(uint64_t i) {
        Decimal d(i);
        return *this /= d;
    }

    //double / float
    ARIES_DEVICE double Decimal::operator/=(const float &f) {
        return GetDouble() / f;
    }

    ARIES_DEVICE double Decimal::operator/=(const double &d) {
        return GetDouble() / d;
    }

    //除法函数，decimalx / decimaly
   ARIES_DEVICE Decimal operator/(const Decimal &left, const Decimal &right) {
       Decimal tmp(left);
       return tmp /= right;
   }

    //signed
     ARIES_DEVICE Decimal operator/(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    //unsigned
    ARIES_DEVICE Decimal operator/(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_DEVICE Decimal operator/(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    //double / float
    ARIES_DEVICE double operator/(const Decimal &left, const float right) {
        return left.GetDouble() / right;
    }

    ARIES_DEVICE double operator/(const Decimal &left, const double right) {
        return left.GetDouble() / right;
    }

    ARIES_DEVICE double operator/(const float left, const Decimal &right) {
        return left / right.GetDouble();
    }

    ARIES_DEVICE double operator/(const double left, const Decimal &right) {
        return left / right.GetDouble();
    }

    //计算取余的目标精度
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcModTargetPrecision( const Decimal &d ) {
        prec = d.prec;
        frac = 0;
    }

    //取余操作
    ARIES_DEVICE Decimal &Decimal::operator%=(const Decimal& d) {
        return ModCalc(d);
    }


    ARIES_DEVICE Decimal& Decimal::ModCalc( const Decimal &d ) {
        // 被除数 和 除数
        Decimal divitend(*this);
        Decimal divisor(d);

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
        Decimal res(0);

        uint32_t dtz = INDEX_LAST_DIG;
        uint32_t dsz = INDEX_LAST_DIG;
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

    ARIES_DEVICE Decimal& Decimal::ModByInt64(const Decimal &divisor) {
        //被除数 在 int64 范围内
        uint64_t dvs = ToUint64();

        //被除数
        uint64_t dvt = divisor.ToUint64();
        uint64_t res = dvs % dvt;
        
        v[1] = res / PER_DEC_MAX_SCALE;
        v[0] = res % PER_DEC_MAX_SCALE;
        return *this;
    }

    ARIES_DEVICE Decimal& Decimal::ModByInt(const Decimal &d) {
        //存放除数
        uint32_t dvs = d.v[0];
        //存放余数
        uint32_t remainder = 0;

        for (int i = NUM_TOTAL_DIG-1; i >=0; i--) 
        {
            if (v[i] || remainder) 
            {
                uint64_t tmp = (uint64_t) v[i] + (uint64_t) remainder * PER_DEC_MAX_SCALE;
                v[i] = tmp / dvs;
                remainder = tmp % dvs;
            }
        }
        
        *this = remainder;

        return *this;
    }

    //signed
    ARIES_DEVICE Decimal &Decimal::operator%=(int8_t i) {
        return *this;
    }

    ARIES_DEVICE Decimal &Decimal::operator%=(int16_t i) {
        return *this;
    }

    ARIES_DEVICE Decimal &Decimal::operator%=(int32_t i) {
       return *this;
    }

    ARIES_DEVICE Decimal &Decimal::operator%=(int64_t i) {
        return *this;
    }

    //unsigned
    ARIES_DEVICE Decimal &Decimal::operator%=(uint8_t i) {
         return *this;
    }

    ARIES_DEVICE Decimal &Decimal::operator%=(uint16_t i) {
         return *this;
    }

    ARIES_DEVICE Decimal &Decimal::operator%=(uint32_t i) {
        return *this;
    }

    ARIES_DEVICE Decimal &Decimal::operator%=(uint64_t i) {
         return *this;
    }

    //double % float
    ARIES_DEVICE double Decimal::operator%=(const float &f) {
       return 0.0;
    }

    ARIES_DEVICE double Decimal::operator%=(const double &d) {
       return 0.0;
    }

    //two operators
    ARIES_DEVICE Decimal operator%(const Decimal &left, const Decimal &right) {
        return left;
    }

    //signed
    ARIES_DEVICE Decimal operator%(const Decimal &left, int8_t right) {
        return left;
    }

    ARIES_DEVICE Decimal operator%(const Decimal &left, int16_t right) {
         return left;
    }

    ARIES_DEVICE Decimal operator%(const Decimal &left, int32_t right) {
         return left;
    }

    ARIES_DEVICE Decimal operator%(const Decimal &left, int64_t right) {
        return left;
    }

    ARIES_DEVICE Decimal operator%(int8_t left, const Decimal &right) {
        return right;
    }

    ARIES_DEVICE Decimal operator%(int16_t left, const Decimal &right) {
       return right;
    }

    ARIES_DEVICE Decimal operator%(int32_t left, const Decimal &right) {
         return right;
    }

    ARIES_DEVICE Decimal operator%(int64_t left, const Decimal &right) {
        return right;
    }

    //unsigned
    ARIES_DEVICE Decimal operator%(const Decimal &left, uint8_t right) {
         return left;
    }

    ARIES_DEVICE Decimal operator%(const Decimal &left, uint16_t right) {
         return left;
    }

    ARIES_DEVICE Decimal operator%(const Decimal &left, uint32_t right) {
        return left;
    }

    ARIES_DEVICE Decimal operator%(const Decimal &left, uint64_t right) {
         return left;
    }

    ARIES_DEVICE Decimal operator%(uint8_t left, const Decimal &right) {
        return right;
    }

    ARIES_DEVICE Decimal operator%(uint16_t left, const Decimal &right) {
         return right;
    }

    ARIES_DEVICE Decimal operator%(uint32_t left, const Decimal &right) {
        return right;
    }

    ARIES_DEVICE Decimal operator%(uint64_t left, const Decimal &right) {
         return right;
    }

    //double % float
    ARIES_DEVICE double operator%(const Decimal &left, const float right) {
         return 0.0;
    }

    ARIES_DEVICE double operator%(const Decimal &left, const double right) {
        return 0.0;
    }

    ARIES_DEVICE double operator%(const float left, const Decimal &right) {
       return 0.0;
    }

    ARIES_DEVICE double operator%(const double left, const Decimal &right) {
        return 0.0;
    }

    //lixin 判断是否为0
   ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isZero() const {
       for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
           if (v[i] != 0) {
               return false;
           }
       }
       return true;
   }

    //10的i次方
   ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::GetPowers10(int i) const {
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


    //根据精度获取其最大值
	ARIES_HOST_DEVICE_NO_INLINE void Decimal::GenMaxDecByPrecision() {

	}


    //加减对齐函数
   ARIES_HOST_DEVICE_NO_INLINE void Decimal::AlignAddSubData(Decimal &d) {
        if (frac == d.frac) {
            //do nothing
            return;
        }
        // 例如 a = 2.4 (frac = 1) b = 1.23 (frac = 2) ：frac < d.frac : this(a) 左移一位
        // 例如 a = 1.23 (frac = 2) b = 2.4 (frac = 1) ：frac > d.frac : b 左移 一位
        if (frac < d.frac) {
            *this<< d.frac-frac;
            frac = d.frac;
        } else {
            d << frac-d.frac;
            d.frac = frac;
        }
   }

    //初始化精度函数
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::initialize(uint32_t pr, uint32_t fc, uint32_t m) {
		//符号 + prec
        prec = pr;
		//小数点位数
        frac = fc;
        //数组存储位置
        aries_memset(v, 0x00, sizeof(v));

        sign = 0;
        error = 0;
    }

    //将 Decimal 转化为数字
    ARIES_HOST_DEVICE_NO_INLINE double Decimal::GetDouble() const {
        double z = 0;
        for(int i = INDEX_LAST_DIG; i>=0 ; i--){
            if(v[i]){
                z += v[i];
            }
            if(z){
                z *= PER_DEC_MAX_SCALE;
            }
        }
        z = z / GetPowers10(frac);
        if(sign == 1){
            z = -z;
        }
        return z;
    }

    //转换为int64
   ARIES_HOST_DEVICE_NO_INLINE uint64_t Decimal::ToUint64() const {
        uint64_t res = v[0] + (uint64_t)v[1] * PER_DEC_MAX_SCALE;
        return res;
   }

    //无用函数
    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::StringToDecimal( char * str, int len )
    {
        return true;
    }

    /* mysql> select 999999999999999999999999999999999999999999999999999999999999999999999999999999999999;
       +--------------------------------------------------------------------------------------+
       | 999999999999999999999999999999999999999999999999999999999999999999999999999999999999 |
       +--------------------------------------------------------------------------------------+
       |                    99999999999999999999999999999999999999999999999999999999999999999 |
       +--------------------------------------------------------------------------------------+
       1 row in set, 1 warning (0.00 sec)

       mysql> show warnings;
       +---------+------+------------------------------------------------------------------------------------------------------------------------+
       | Level   | Code | Message                                                                                                                |
       +---------+------+------------------------------------------------------------------------------------------------------------------------+
       | Warning | 1292 | Truncated incorrect DECIMAL value: '999999999999999999999999999999999999999999999999999999999999999999999999999999999' |
       +---------+------+------------------------------------------------------------------------------------------------------------------------+
    */
	//string转decimal函数
    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::StringToDecimal( char * str )
    {
         memset(v,0x00,sizeof(uint32_t)*NUM_TOTAL_DIG);
        int flag=0;
        uint64_t temp = 0;
        uint32_t carry = 0;
        int dotflag = 0;
        int dot = 0;
        sign = 0;
        int t = 0;
        if(str[0]=='-'){
            flag=1;
            t = 1;
        }
        int i=0;
        for(i=t; str[i]!='\0' ;i++){
            if( str[i]=='.'){
                dotflag = 1;
                i++;
            }
            if(dotflag != 0){
                dot++;
            }
            #pragma unroll
            for(int j=0 ; j<NUM_TOTAL_DIG ; j++){
                temp = (uint64_t)v[j] * 10 + carry;
                v[j] = temp % MAX_INT32;
                carry = temp / MAX_INT32;
            }
            temp = (uint64_t)v[0] + str[i] -'0';
            v[0] = temp % MAX_INT32;
            carry = temp / MAX_INT32;
            #pragma unroll
            for(int j=1 ; j<NUM_TOTAL_DIG && carry > 0; j++){
                temp = (uint64_t)v[j] + carry;
                v[j] = temp % MAX_INT32;
                carry = temp / MAX_INT32;
            }
        }
        if( flag==1 ){
            sign = 1;
        }
        prec = i - flag - dotflag;
        frac = dot;
        return true;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        
        uint32_t carry;
    
        cgbn::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
        r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);

        int32_t sr =  -cgbn::fast_propagate_sub(carry, r);
        
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_sig(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add){
        uint32_t sync=cgbn::core::sync_mask(), group_thread=threadIdx.x & TPI-1;
        uint32_t rl, p0=add, p1=0, t;
        int32_t  threads = TPI;

        #pragma unroll
        for(int32_t index=0;index<threads;index++) {
            t=__shfl_sync(sync, b, index, TPI);

            p0=cgbn::madlo_cc(a, t, p0);
            p1=cgbn::addc(p1, 0);
            
            if(group_thread<threads-index) 
            rl=p0;

            rl=__shfl_sync(sync, rl, threadIdx.x+1, TPI);

            p0=cgbn::madhi_cc(a, t, p1);
            p1=cgbn::addc(0, 0);
            
            p0=cgbn::add_cc(p0, rl);
            p1=cgbn::addc(p1, 0);

            // printf("tid =%d p0 = %08x p1 = %08x a = %08x b = %08x\n",threadIdx.x, p0, p1, a, b);
        }
        r=rl;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        // printf("mont_mul 6 tid = %d LIMBS = %d a = %08x %08x b = %08x %08x add = %08x %08x r = %08x %08x\n",threadIdx.x, LIMBS, a[0],a[1], b[0], b[1], add[0], add[1], r[0], r[1]);
        uint32_t sync=cgbn::core::sync_mask(), group_thread=threadIdx.x & TPI-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS], ra[LIMBS+2], ru[LIMBS+1];
        int32_t  threads = TPI;

        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            ra[index]=add[index];

        ra[LIMBS]=0;
        ra[LIMBS+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            ru[index]=add[index];

        ru[LIMBS]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS*2;l+=2) {
            if(l<LIMBS) 
                term0=__shfl_sync(sync, b[l], row, TPI);
            else
                term0=__shfl_sync(sync, b[l-LIMBS], row+1, TPI);
            if(l+1<LIMBS)
                term1=__shfl_sync(sync, b[l+1], row, TPI);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS], row+1, TPI);

            cgbn::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS%2==0)
                ra[LIMBS]=chain1.add(ra[LIMBS], 0);      
            
            cgbn::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS%2==1)
                ru[LIMBS-1]=chain2.add(0, 0);

            cgbn::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS%2==1)
                ru[LIMBS-1]=0;
            else 
                ru[LIMBS-2]=chain3.add(0, 0);
            ru[LIMBS-1+LIMBS%2]=0;

            cgbn::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS-2-LIMBS%2]=chain4.madlo(a[LIMBS-1-LIMBS%2], term1, ra[LIMBS-LIMBS%2]);
            ra[LIMBS-1-LIMBS%2]=chain4.madhi(a[LIMBS-1-LIMBS%2], term1, ra[LIMBS+1-LIMBS%2]);
            if(LIMBS%2==1)
                ra[LIMBS-1]=chain4.add(0, 0);

            if(l<LIMBS) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS]=t0;
                rl[l-LIMBS]=__shfl_sync(sync, rl[l-LIMBS], threadIdx.x+1, TPI);
                t0=rl[l-LIMBS];
            }
            if(l+1<LIMBS) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS]=t1;
                rl[l-LIMBS+1]=__shfl_sync(sync, rl[l+1-LIMBS], threadIdx.x+1, TPI);
                t1=rl[l+1-LIMBS];
            }
                    
            ra[LIMBS-2]=cgbn::add_cc(ra[LIMBS-2], t0);
            ra[LIMBS-1]=cgbn::addc_cc(ra[LIMBS-1], t1);
            ra[LIMBS]=cgbn::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE_FORCE int32_t  cgbn_compare(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI)-1;
        
        uint32_t group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        if(LIMBS==1) {
        a_ballot=__ballot_sync(sync, a[0]>=b[0]);
        b_ballot=__ballot_sync(sync, a[0]<=b[0]);
        }
        else {
        cgbn::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        }
        
        if(TPI<warpSize) {
        uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);
        
        a_ballot=a_ballot & mask;
        b_ballot=b_ballot & mask;
        }
        
        return cgbn::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE void div_wide(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        // printf("div_wide_ in 1\n");
        uint32_t sync=cgbn::core::sync_mask(), group_thread=threadIdx.x & TPI-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS], approx[DLIMBS], estimate[DLIMBS], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS], y[LIMBS], plo[LIMBS], phi[LIMBS], quotient[LIMBS];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            quotient[index] = 0;

        if(numthreads<TPI) {
            cgbn::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS-1], TPI-1, TPI);
        d0=__shfl_sync(sync, denom[LIMBS-2], TPI-1, TPI);

        cgbn::dlimbs_scatter(dtemp, denom, TPI-1);  
        cgbn::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn::dlimbs_scatter(dtemp, x, TPI-1);
            cgbn::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn::dlimbs_all_gather(y, estimate);
            
            cgbn::mpmul(plo, phi, y, denom);
            c=cgbn::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI-1, TPI);

            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn::mpsub(x, x, plo);
            
            x2=x2+cgbn::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS-1], TPI-1, TPI);
            x0=__shfl_sync(sync, x[LIMBS-2], TPI-1, TPI);
            
            correction=cgbn::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn::mpmul32(plo, denom, correction);
            t=cgbn::core::resolve_add_b(c, plo);
            c=cgbn::mpadd(x, x, plo);
            x2=x2+t+cgbn::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn::mpadd(x, x, denom);
            cgbn::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  uint8_t operator_add(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;
        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // printf("******************b 数组需要向右移动 %d 个单位******************\n",shift);
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS];

            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++)
                add[index]=0;

            if(LIMBS == 1){ // 这里后面要改成宏
                cgbn_mul_sig(b[0], b[0], __POW10_ARRAY[shift-1][group_thread*LIMBS], add[0]);
            }
            else{
                // printf("tid = %d group_thread = %d LIMBS = %d group_thread*LIMBS = %d sizeof(LIMBS) = %lu\n", threadIdx.x, group_thread, LIMBS, group_thread*LIMBS, sizeof(LIMBS));
                cgbn_mul_mlt(b, b, __POW10_ARRAY[shift-1]+group_thread*LIMBS, add);
            }
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS];

            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++)
                add[index]=0;

            if(LIMBS == 1){ // 这里后面要改成宏
                cgbn_mul_sig(a[0], a[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS], add[0]);
            }
            else{
                cgbn_mul_mlt(a, a, __POW10_ARRAY[-shift-1]+group_thread*LIMBS, add);
            }
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare(cgbn::core::sync_mask(), a, b);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub(r, a, b);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub(r, b, a);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add(r, a, b);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;
        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS];

            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++)
                add[index]=0;

            if(LIMBS == 1){ // 这里后面要改成宏
                cgbn_mul_sig(b[0], b[0], __POW10_ARRAY[shift-1][group_thread*LIMBS], add[0]);
            }
            else{
                cgbn_mul_mlt(b, b, __POW10_ARRAY[shift-1]+group_thread*LIMBS, add);
            }
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS;index++)
                add[index]=0;

            if(LIMBS == 1){ // 这里后面要改成宏
                cgbn_mul_sig(a[0], a[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS], add[0]);
            }
            else{
                cgbn_mul_mlt(a, a, __POW10_ARRAY[-shift-1]+group_thread*LIMBS, add);
            }
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add(r, a, b);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare(cgbn::core::sync_mask(), a, b);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub(r, a, b);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub(r, b, a);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul(uint32_t r[], uint32_t a[], uint32_t b[]){
        // printf("****************** operator_mul******************\n");
        uint32_t add[LIMBS];

        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            add[index]=0;
        if(LIMBS == 1){ // 这里后面要改成宏
            cgbn_mul_sig(r[0], a[0], b[0], add[0]);
        }
        else{
            cgbn_mul_mlt(r, a, b, add);
        }
    }

    ARIES_DEVICE  void operator_div(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI-1;

        uint32_t add[LIMBS];
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++)
            add[index]=0;

        if(LIMBS == 1){ // 这里后面要改成宏
            cgbn_mul_sig(a[0], a[0], __POW10_ARRAY[divitend_shift-1][group_thread*LIMBS],  add[0]);
        }
        else{
            cgbn_mul_mlt(a, a, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS, add);
        }

        uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
        uint32_t shift, numthreads;

        shift=cgbn::core::clz(b);

        cgbn::core::rotate_left(cgbn::core::sync_mask(), denom_local, b, shift);
        cgbn::core::rotate_left(cgbn::core::sync_mask(), num_low, a, shift);
        cgbn::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI-cgbn::core::clzt(num_high);
        div_wide(r, num_low, num_high, denom_local, numthreads);
    }

// LEN = 4 TPI = 4 LIMBS = 1 DLIMBS = 1

    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_4_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_ONE)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        a_ballot=__ballot_sync(sync, a[0]>=b[0]);
        b_ballot=__ballot_sync(sync, a[0]<=b[0]);

        if(TPI_ONE<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_4_TPI_4::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_4_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        
        uint32_t carry;
    
        cgbn_LEN_4_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
        r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);

        int32_t sr =  -cgbn_LEN_4_TPI_4::fast_propagate_sub(carry, r);
        
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_4_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_4_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_4_TPI_4::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_sig_LEN_4_TPI_4(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add){
        uint32_t sync=cgbn_LEN_4_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        uint32_t rl, p0=add, p1=0, t;
        int32_t  threads = TPI_ONE;

        #pragma unroll
        for(int32_t index=0;index<threads;index++) {
            t=__shfl_sync(sync, b, index, TPI_ONE);

            p0=cgbn_LEN_4_TPI_4::madlo_cc(a, t, p0);
            p1=cgbn_LEN_4_TPI_4::addc(p1, 0);
            
            if(group_thread<threads-index) 
            rl=p0;

            rl=__shfl_sync(sync, rl, threadIdx.x+1, TPI_ONE);

            p0=cgbn_LEN_4_TPI_4::madhi_cc(a, t, p1);
            p1=cgbn_LEN_4_TPI_4::addc(0, 0);
            
            p0=cgbn_LEN_4_TPI_4::add_cc(p0, rl);
            p1=cgbn_LEN_4_TPI_4::addc(p1, 0);

        }
        r=rl;
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

       // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_4_TPI_4(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_4_TPI_4(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_4_TPI_4(cgbn_LEN_4_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_4_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_4_TPI_4(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_4_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_4_TPI_4(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_ONE] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_4_TPI_4(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_4_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_4_TPI_4(cgbn_LEN_4_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_4_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_4_TPI_4(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_ONE];
        uint32_t a_tmp[LIMBS_ONE];
        uint32_t b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_sig_LEN_4_TPI_4(r[0], a_tmp[0], b_tmp[0], add[0]);
    }

    ARIES_DEVICE  void operator_div_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;

        uint32_t add[LIMBS_ONE], a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_sig_LEN_4_TPI_4(a_tmp[0], a_tmp[0], __POW10_ARRAY[divitend_shift-1][group_thread*LIMBS_ONE],  add[0]);

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_4_TPI_4::core::clz(b_tmp);

        cgbn_LEN_4_TPI_4::core::rotate_left(cgbn_LEN_4_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_4_TPI_4::core::rotate_left(cgbn_LEN_4_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_4_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_ONE-cgbn_LEN_4_TPI_4::core::clzt(num_high);
        cgbn_div_LEN_4_TPI_4(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_4_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_4_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE], quotient[LIMBS_ONE];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            quotient[index] = 0;

        if(numthreads<TPI_ONE) {
            cgbn_LEN_4_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
                x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
                x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
                y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_4_TPI_4::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=y[index];
                quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_4_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_4_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_4_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_4_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_4_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_4_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_4_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_4_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_4_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_ONE-1, TPI_ONE);
            
            correction=cgbn_LEN_4_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_4_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_4_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_4_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_4_TPI_4::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_4_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_4_TPI_4::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_4_TPI_4::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_4_TPI_4::core::clz(b_tmp);
        cgbn_LEN_4_TPI_4::core::rotate_left(cgbn_LEN_4_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_4_TPI_4::core::rotate_left(cgbn_LEN_4_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_4_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_4_TPI_4::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_ONE-cgbn_LEN_4_TPI_4::core::clzt(num_high);
        cgbn_mod_LEN_4_TPI_4(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_4_TPI_4::core::rotate_right(cgbn_LEN_4_TPI_4::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_4_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_4_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE];

        if(numthreads<TPI_ONE) {
            cgbn_LEN_4_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_4_TPI_4::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_ONE;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_4_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_4_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_4_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_4_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_4_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_4_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_4_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_4_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_4_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_ONE-1, TPI_ONE);

            correction=cgbn_LEN_4_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_4_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_4_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_4_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_4_TPI_4::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_4_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_4_TPI_4::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=x[index];
    }

// LEN = 8 TPI = 4 LIMBS = 2 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_8_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_ONE)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        cgbn_LEN_8_TPI_4::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn_LEN_8_TPI_4::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        
        if(TPI_ONE<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);   
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_8_TPI_4::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_8_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {  
        uint32_t carry;
        cgbn_LEN_8_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);
        int32_t sr =  -cgbn_LEN_8_TPI_4::fast_propagate_sub(carry, r);    
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_8_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_8_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_8_TPI_4::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt_LEN_8_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        uint32_t sync=cgbn_LEN_8_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS_TWO], ra[LIMBS_TWO+2], ru[LIMBS_TWO+1];
        int32_t  threads = TPI_ONE;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            ra[index]=add[index];

        ra[LIMBS_TWO]=0;
        ra[LIMBS_TWO+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            ru[index]=add[index];

        ru[LIMBS_TWO]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS_TWO*2;l+=2) {
            if(l<LIMBS_TWO) 
                term0=__shfl_sync(sync, b[l], row, TPI_ONE);
            else
                term0=__shfl_sync(sync, b[l-LIMBS_TWO], row+1, TPI_ONE);
            if(l+1<LIMBS_TWO)
                term1=__shfl_sync(sync, b[l+1], row, TPI_ONE);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS_TWO], row+1, TPI_ONE);

            cgbn_LEN_8_TPI_4::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS_TWO%2==0)
                ra[LIMBS_TWO]=chain1.add(ra[LIMBS_TWO], 0);      
            
            cgbn_LEN_8_TPI_4::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS_TWO%2==1)
                ru[LIMBS_TWO-1]=chain2.add(0, 0);

            cgbn_LEN_8_TPI_4::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS_TWO%2==1)
                ru[LIMBS_TWO-1]=0;
            else 
                ru[LIMBS_TWO-2]=chain3.add(0, 0);
            ru[LIMBS_TWO-1+LIMBS_TWO%2]=0;

            cgbn_LEN_8_TPI_4::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS_TWO-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS_TWO-2-LIMBS_TWO%2]=chain4.madlo(a[LIMBS_TWO-1-LIMBS_TWO%2], term1, ra[LIMBS_TWO-LIMBS_TWO%2]);
            ra[LIMBS_TWO-1-LIMBS_TWO%2]=chain4.madhi(a[LIMBS_TWO-1-LIMBS_TWO%2], term1, ra[LIMBS_TWO+1-LIMBS_TWO%2]);
            if(LIMBS_TWO%2==1)
                ra[LIMBS_TWO-1]=chain4.add(0, 0);

            if(l<LIMBS_TWO) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI_ONE);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS_TWO]=t0;
                rl[l-LIMBS_TWO]=__shfl_sync(sync, rl[l-LIMBS_TWO], threadIdx.x+1, TPI_ONE);
                t0=rl[l-LIMBS_TWO];
            }
            if(l+1<LIMBS_TWO) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI_ONE);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS_TWO]=t1;
                rl[l-LIMBS_TWO+1]=__shfl_sync(sync, rl[l+1-LIMBS_TWO], threadIdx.x+1, TPI_ONE);
                t1=rl[l+1-LIMBS_TWO];
            }
                    
            ra[LIMBS_TWO-2]=cgbn_LEN_8_TPI_4::add_cc(ra[LIMBS_TWO-2], t0);
            ra[LIMBS_TWO-1]=cgbn_LEN_8_TPI_4::addc_cc(ra[LIMBS_TWO-1], t1);
            ra[LIMBS_TWO]=cgbn_LEN_8_TPI_4::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_8_TPI_4(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_TWO, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_8_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_TWO, add);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_8_TPI_4(cgbn_LEN_8_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_8_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_8_TPI_4(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_8_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_8_TPI_4(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_TWO, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_TWO] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_8_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_TWO, add);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_8_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_8_TPI_4(cgbn_LEN_8_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_8_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_8_TPI_4(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_TWO];
        uint32_t a_tmp[LIMBS_TWO];
        uint32_t b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_8_TPI_4(r, a_tmp, b_tmp, add);
    }

    ARIES_DEVICE  void operator_div_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;

        uint32_t add[LIMBS_TWO], a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_mlt_LEN_8_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS_TWO, add);

        uint32_t num_low[LIMBS_TWO], num_high[LIMBS_TWO], denom_local[LIMBS_TWO];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_8_TPI_4::core::clz(b_tmp);

        cgbn_LEN_8_TPI_4::core::rotate_left(cgbn_LEN_8_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_8_TPI_4::core::rotate_left(cgbn_LEN_8_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_8_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_ONE-cgbn_LEN_8_TPI_4::core::clzt(num_high);
        cgbn_div_LEN_8_TPI_4(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE  void operator_mod_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_TWO], num_high[LIMBS_TWO], denom_local[LIMBS_TWO];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_8_TPI_4::core::clz(b_tmp);
        cgbn_LEN_8_TPI_4::core::rotate_left(cgbn_LEN_8_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_8_TPI_4::core::rotate_left(cgbn_LEN_8_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_8_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_8_TPI_4::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_ONE-cgbn_LEN_8_TPI_4::core::clzt(num_high);
        cgbn_mod_LEN_8_TPI_4(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_8_TPI_4::core::rotate_right(cgbn_LEN_8_TPI_4::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_8_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_8_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_TWO], y[LIMBS_TWO], plo[LIMBS_TWO], phi[LIMBS_TWO];

        if(numthreads<TPI_ONE) {
            cgbn_LEN_8_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_8_TPI_4::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_TWO;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_TWO-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_TWO-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_8_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_8_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_8_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_8_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_8_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_8_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_8_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_8_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_8_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_TWO-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_TWO-2], TPI_ONE-1, TPI_ONE);

            correction=cgbn_LEN_8_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_8_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_8_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_8_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_8_TPI_4::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_8_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_8_TPI_4::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            q[index]=x[index];
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_8_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_8_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_TWO], y[LIMBS_TWO], plo[LIMBS_TWO], phi[LIMBS_TWO], quotient[LIMBS_TWO];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            quotient[index] = 0;

        if(numthreads<TPI_ONE) {
            cgbn_LEN_8_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_8_TPI_4::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_TWO-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_TWO-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_8_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_8_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_8_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_8_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_8_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_8_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_8_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_8_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_8_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_TWO-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_TWO-2], TPI_ONE-1, TPI_ONE);
            
            correction=cgbn_LEN_8_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_8_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_8_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_8_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_8_TPI_4::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_8_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_8_TPI_4::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_8_TPI_4::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            q[index]=quotient[index];
    }

// LEN = 16 TPI = 4 LIMBS = 4 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_16_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_ONE)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        cgbn_LEN_16_TPI_4::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn_LEN_16_TPI_4::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        
        if(TPI_ONE<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);   
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_16_TPI_4::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_16_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {  
        uint32_t carry;
        cgbn_LEN_16_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);
        int32_t sr =  -cgbn_LEN_16_TPI_4::fast_propagate_sub(carry, r);    
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_16_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_16_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_16_TPI_4::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt_LEN_16_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        uint32_t sync=cgbn_LEN_16_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS_THR], ra[LIMBS_THR+2], ru[LIMBS_THR+1];
        int32_t  threads = TPI_ONE;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            ra[index]=add[index];

        ra[LIMBS_THR]=0;
        ra[LIMBS_THR+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            ru[index]=add[index];

        ru[LIMBS_THR]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS_THR*2;l+=2) {
            if(l<LIMBS_THR) 
                term0=__shfl_sync(sync, b[l], row, TPI_ONE);
            else
                term0=__shfl_sync(sync, b[l-LIMBS_THR], row+1, TPI_ONE);
            if(l+1<LIMBS_THR)
                term1=__shfl_sync(sync, b[l+1], row, TPI_ONE);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS_THR], row+1, TPI_ONE);

            cgbn_LEN_16_TPI_4::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS_THR%2==0)
                ra[LIMBS_THR]=chain1.add(ra[LIMBS_THR], 0);      
            
            cgbn_LEN_16_TPI_4::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS_THR%2==1)
                ru[LIMBS_THR-1]=chain2.add(0, 0);

            cgbn_LEN_16_TPI_4::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS_THR%2==1)
                ru[LIMBS_THR-1]=0;
            else 
                ru[LIMBS_THR-2]=chain3.add(0, 0);
            ru[LIMBS_THR-1+LIMBS_THR%2]=0;

            cgbn_LEN_16_TPI_4::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS_THR-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS_THR-2-LIMBS_THR%2]=chain4.madlo(a[LIMBS_THR-1-LIMBS_THR%2], term1, ra[LIMBS_THR-LIMBS_THR%2]);
            ra[LIMBS_THR-1-LIMBS_THR%2]=chain4.madhi(a[LIMBS_THR-1-LIMBS_THR%2], term1, ra[LIMBS_THR+1-LIMBS_THR%2]);
            if(LIMBS_THR%2==1)
                ra[LIMBS_THR-1]=chain4.add(0, 0);

            if(l<LIMBS_THR) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI_ONE);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS_THR]=t0;
                rl[l-LIMBS_THR]=__shfl_sync(sync, rl[l-LIMBS_THR], threadIdx.x+1, TPI_ONE);
                t0=rl[l-LIMBS_THR];
            }
            if(l+1<LIMBS_THR) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI_ONE);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS_THR]=t1;
                rl[l-LIMBS_THR+1]=__shfl_sync(sync, rl[l+1-LIMBS_THR], threadIdx.x+1, TPI_ONE);
                t1=rl[l+1-LIMBS_THR];
            }
                    
            ra[LIMBS_THR-2]=cgbn_LEN_16_TPI_4::add_cc(ra[LIMBS_THR-2], t0);
            ra[LIMBS_THR-1]=cgbn_LEN_16_TPI_4::addc_cc(ra[LIMBS_THR-1], t1);
            ra[LIMBS_THR]=cgbn_LEN_16_TPI_4::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;
        
        uint32_t a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_THR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_4(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_THR, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_THR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_THR, add);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_16_TPI_4(cgbn_LEN_16_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_16_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_16_TPI_4(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_16_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_THR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_4(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_THR, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_THR] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_THR, add);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_16_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_16_TPI_4(cgbn_LEN_16_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_16_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_16_TPI_4(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_THR];
        uint32_t a_tmp[LIMBS_THR];
        uint32_t b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_16_TPI_4(r, a_tmp, b_tmp, add);
    }

    ARIES_DEVICE  void operator_div_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;

        uint32_t add[LIMBS_THR], a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_mlt_LEN_16_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS_THR, add);

        uint32_t num_low[LIMBS_THR], num_high[LIMBS_THR], denom_local[LIMBS_THR];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_16_TPI_4::core::clz(b_tmp);

        cgbn_LEN_16_TPI_4::core::rotate_left(cgbn_LEN_16_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_16_TPI_4::core::rotate_left(cgbn_LEN_16_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_16_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_ONE-cgbn_LEN_16_TPI_4::core::clzt(num_high);
        cgbn_div_LEN_16_TPI_4(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_16_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_16_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_THR], y[LIMBS_THR], plo[LIMBS_THR], phi[LIMBS_THR], quotient[LIMBS_THR];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            quotient[index] = 0;

        if(numthreads<TPI_ONE) {
            cgbn_LEN_16_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_16_TPI_4::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_THR-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_THR-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_16_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_16_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_16_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_16_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_16_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_16_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_16_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_16_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_16_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_THR-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_THR-2], TPI_ONE-1, TPI_ONE);
            
            correction=cgbn_LEN_16_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_16_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_16_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_16_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_16_TPI_4::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_16_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_16_TPI_4::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_16_TPI_4::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_THR], num_high[LIMBS_THR], denom_local[LIMBS_THR];
        uint32_t shift, numthreads;
       
        uint32_t a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        shift=cgbn_LEN_16_TPI_4::core::clz(b_tmp);
        cgbn_LEN_16_TPI_4::core::rotate_left(cgbn_LEN_16_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_16_TPI_4::core::rotate_left(cgbn_LEN_16_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_16_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_16_TPI_4::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_ONE-cgbn_LEN_16_TPI_4::core::clzt(num_high);
        cgbn_mod_LEN_16_TPI_4(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_16_TPI_4::core::rotate_right(cgbn_LEN_16_TPI_4::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_16_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_16_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_THR], y[LIMBS_THR], plo[LIMBS_THR], phi[LIMBS_THR];

        if(numthreads<TPI_ONE) {
            cgbn_LEN_16_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_16_TPI_4::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_THR;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_THR-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_THR-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_16_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_16_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_16_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_16_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_16_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_16_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_16_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_16_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_16_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_THR-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_THR-2], TPI_ONE-1, TPI_ONE);

            correction=cgbn_LEN_16_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_16_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_16_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_16_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_16_TPI_4::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_16_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_16_TPI_4::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            q[index]=x[index];
    }

// LEN = 32 TPI = 4 LIMBS = 8 DLIMBS = 2
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_32_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_ONE)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_ONE-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        cgbn_LEN_32_TPI_4::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn_LEN_32_TPI_4::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        
        if(TPI_ONE<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);   
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_32_TPI_4::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_32_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {  
        uint32_t carry;
        cgbn_LEN_32_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);
        int32_t sr =  -cgbn_LEN_32_TPI_4::fast_propagate_sub(carry, r);    
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_32_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_32_TPI_4::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_32_TPI_4::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt_LEN_32_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        uint32_t sync=cgbn_LEN_32_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS_FOR], ra[LIMBS_FOR+2], ru[LIMBS_FOR+1];
        int32_t  threads = TPI_ONE;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            ra[index]=add[index];

        ra[LIMBS_FOR]=0;
        ra[LIMBS_FOR+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            ru[index]=add[index];

        ru[LIMBS_FOR]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS_FOR*2;l+=2) {
            if(l<LIMBS_FOR) 
                term0=__shfl_sync(sync, b[l], row, TPI_ONE);
            else
                term0=__shfl_sync(sync, b[l-LIMBS_FOR], row+1, TPI_ONE);
            if(l+1<LIMBS_FOR)
                term1=__shfl_sync(sync, b[l+1], row, TPI_ONE);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS_FOR], row+1, TPI_ONE);

            cgbn_LEN_32_TPI_4::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS_FOR%2==0)
                ra[LIMBS_FOR]=chain1.add(ra[LIMBS_FOR], 0);      
            
            cgbn_LEN_32_TPI_4::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS_FOR%2==1)
                ru[LIMBS_FOR-1]=chain2.add(0, 0);

            cgbn_LEN_32_TPI_4::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS_FOR%2==1)
                ru[LIMBS_FOR-1]=0;
            else 
                ru[LIMBS_FOR-2]=chain3.add(0, 0);
            ru[LIMBS_FOR-1+LIMBS_FOR%2]=0;

            cgbn_LEN_32_TPI_4::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS_FOR-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS_FOR-2-LIMBS_FOR%2]=chain4.madlo(a[LIMBS_FOR-1-LIMBS_FOR%2], term1, ra[LIMBS_FOR-LIMBS_FOR%2]);
            ra[LIMBS_FOR-1-LIMBS_FOR%2]=chain4.madhi(a[LIMBS_FOR-1-LIMBS_FOR%2], term1, ra[LIMBS_FOR+1-LIMBS_FOR%2]);
            if(LIMBS_FOR%2==1)
                ra[LIMBS_FOR-1]=chain4.add(0, 0);

            if(l<LIMBS_FOR) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI_ONE);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS_FOR]=t0;
                rl[l-LIMBS_FOR]=__shfl_sync(sync, rl[l-LIMBS_FOR], threadIdx.x+1, TPI_ONE);
                t0=rl[l-LIMBS_FOR];
            }
            if(l+1<LIMBS_FOR) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI_ONE);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS_FOR]=t1;
                rl[l-LIMBS_FOR+1]=__shfl_sync(sync, rl[l+1-LIMBS_FOR], threadIdx.x+1, TPI_ONE);
                t1=rl[l+1-LIMBS_FOR];
            }
                    
            ra[LIMBS_FOR-2]=cgbn_LEN_32_TPI_4::add_cc(ra[LIMBS_FOR-2], t0);
            ra[LIMBS_FOR-1]=cgbn_LEN_32_TPI_4::addc_cc(ra[LIMBS_FOR-1], t1);
            ra[LIMBS_FOR]=cgbn_LEN_32_TPI_4::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;
                
        uint32_t a_tmp[LIMBS_FOR], b_tmp[LIMBS_FOR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_FOR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_4(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_FOR, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_FOR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_FOR, add);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_32_TPI_4(cgbn_LEN_32_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_32_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_32_TPI_4(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_32_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_FOR], b_tmp[LIMBS_FOR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_FOR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_4(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_FOR, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_FOR] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_FOR, add);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_32_TPI_4(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_32_TPI_4(cgbn_LEN_32_TPI_4::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_32_TPI_4(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_32_TPI_4(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_FOR];
        uint32_t a_tmp[LIMBS_FOR];
        uint32_t b_tmp[LIMBS_FOR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_32_TPI_4(r, a_tmp, b_tmp, add);
    }

    ARIES_DEVICE  void operator_div_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_ONE-1;

        uint32_t add[LIMBS_FOR], a_tmp[LIMBS_FOR], b_tmp[LIMBS_FOR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_mlt_LEN_32_TPI_4(a_tmp, a_tmp, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS_FOR, add);

        uint32_t num_low[LIMBS_FOR], num_high[LIMBS_FOR], denom_local[LIMBS_FOR];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_32_TPI_4::core::clz(b_tmp);

        cgbn_LEN_32_TPI_4::core::rotate_left(cgbn_LEN_32_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_4::core::rotate_left(cgbn_LEN_32_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_ONE-cgbn_LEN_32_TPI_4::core::clzt(num_high);
        cgbn_div_LEN_32_TPI_4(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_32_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_TWO], approx[DLIMBS_TWO], estimate[DLIMBS_TWO], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_FOR], y[LIMBS_FOR], plo[LIMBS_FOR], phi[LIMBS_FOR], quotient[LIMBS_FOR];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            quotient[index] = 0;

        if(numthreads<TPI_ONE) {
            cgbn_LEN_32_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_4::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_FOR-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_FOR-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_32_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_32_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_32_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_FOR-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_FOR-2], TPI_ONE-1, TPI_ONE);
            
            correction=cgbn_LEN_32_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_4::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_4::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_32_TPI_4::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_FOR], num_high[LIMBS_FOR], denom_local[LIMBS_FOR];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_FOR], b_tmp[LIMBS_FOR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        shift=cgbn_LEN_32_TPI_4::core::clz(b_tmp);
        cgbn_LEN_32_TPI_4::core::rotate_left(cgbn_LEN_32_TPI_4::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_4::core::rotate_left(cgbn_LEN_32_TPI_4::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_4::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_32_TPI_4::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_ONE-cgbn_LEN_32_TPI_4::core::clzt(num_high);
        cgbn_mod_LEN_32_TPI_4(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_32_TPI_4::core::rotate_right(cgbn_LEN_32_TPI_4::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_32_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_4::core::sync_mask(), group_thread=threadIdx.x & TPI_ONE-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_TWO], approx[DLIMBS_TWO], estimate[DLIMBS_TWO], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_FOR], y[LIMBS_FOR], plo[LIMBS_FOR], phi[LIMBS_FOR];

        if(numthreads<TPI_ONE) {
            cgbn_LEN_32_TPI_4::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_ONE);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_4::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_FOR;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_FOR-1], TPI_ONE-1, TPI_ONE);
        d0=__shfl_sync(sync, denom[LIMBS_FOR-2], TPI_ONE-1, TPI_ONE);

        cgbn_LEN_32_TPI_4::dlimbs_scatter(dtemp, denom, TPI_ONE-1);  
        cgbn_LEN_32_TPI_4::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_4::dlimbs_scatter(dtemp, x, TPI_ONE-1);
            cgbn_LEN_32_TPI_4::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_4::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_4::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_4::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_ONE-1, TPI_ONE);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_FOR;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_ONE);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_ONE);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_ONE);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_4::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_4::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_FOR-1], TPI_ONE-1, TPI_ONE);
            x0=__shfl_sync(sync, x[LIMBS_FOR-2], TPI_ONE-1, TPI_ONE);

            correction=cgbn_LEN_32_TPI_4::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_4::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_4::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_4::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_4::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_4::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_4::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_FOR;index++)
            q[index]=x[index];
    }

// LEN = 8 TPI = 8 LIMBS = 1 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_8_TPI_8(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_TWO)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_TWO-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        a_ballot=__ballot_sync(sync, a[0]>=b[0]);
        b_ballot=__ballot_sync(sync, a[0]<=b[0]);

        if(TPI_TWO<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_8_TPI_8::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_8_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        
        uint32_t carry;
    
        cgbn_LEN_8_TPI_8::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
        r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);

        int32_t sr =  -cgbn_LEN_8_TPI_8::fast_propagate_sub(carry, r);
        
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_8_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_8_TPI_8::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_8_TPI_8::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_sig_LEN_8_TPI_8(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add){
        uint32_t sync=cgbn_LEN_8_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t rl, p0=add, p1=0, t;
        int32_t  threads = TPI_TWO;

        #pragma unroll
        for(int32_t index=0;index<threads;index++) {
            t=__shfl_sync(sync, b, index, TPI_TWO);

            p0=cgbn_LEN_8_TPI_8::madlo_cc(a, t, p0);
            p1=cgbn_LEN_8_TPI_8::addc(p1, 0);
            
            if(group_thread<threads-index) 
            rl=p0;

            rl=__shfl_sync(sync, rl, threadIdx.x+1, TPI_TWO);

            p0=cgbn_LEN_8_TPI_8::madhi_cc(a, t, p1);
            p1=cgbn_LEN_8_TPI_8::addc(0, 0);
            
            p0=cgbn_LEN_8_TPI_8::add_cc(p0, rl);
            p1=cgbn_LEN_8_TPI_8::addc(p1, 0);

        }
        r=rl;
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

       // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_8_TPI_8(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_8_TPI_8(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_8_TPI_8(cgbn_LEN_8_TPI_8::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_8_TPI_8(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_8_TPI_8(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_8_TPI_8(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_8_TPI_8(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_ONE] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_8_TPI_8(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_8_TPI_8(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_8_TPI_8(cgbn_LEN_8_TPI_8::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_8_TPI_8(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_8_TPI_8(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_ONE];
        uint32_t a_tmp[LIMBS_ONE];
        uint32_t b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_sig_LEN_8_TPI_8(r[0], a_tmp[0], b_tmp[0], add[0]);
    }

    ARIES_DEVICE  void operator_div_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;

        uint32_t add[LIMBS_ONE], a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_sig_LEN_8_TPI_8(a_tmp[0], a_tmp[0], __POW10_ARRAY[divitend_shift-1][group_thread*LIMBS_ONE],  add[0]);

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_8_TPI_8::core::clz(b_tmp);

        cgbn_LEN_8_TPI_8::core::rotate_left(cgbn_LEN_8_TPI_8::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_8_TPI_8::core::rotate_left(cgbn_LEN_8_TPI_8::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_8_TPI_8::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_TWO-cgbn_LEN_8_TPI_8::core::clzt(num_high);
        cgbn_div_LEN_8_TPI_8(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_8_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_8_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE], quotient[LIMBS_ONE];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            quotient[index] = 0;

        if(numthreads<TPI_TWO) {
            cgbn_LEN_8_TPI_8::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
                x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
                x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_TWO);
                y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_8_TPI_8::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=y[index];
                quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_TWO-1, TPI_TWO);

        cgbn_LEN_8_TPI_8::dlimbs_scatter(dtemp, denom, TPI_TWO-1);  
        cgbn_LEN_8_TPI_8::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_8_TPI_8::dlimbs_scatter(dtemp, x, TPI_TWO-1);
            cgbn_LEN_8_TPI_8::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_8_TPI_8::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_8_TPI_8::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_8_TPI_8::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_TWO-1, TPI_TWO);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_TWO);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_TWO);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_TWO);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_8_TPI_8::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_8_TPI_8::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_TWO-1, TPI_TWO);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_TWO-1, TPI_TWO);
            
            correction=cgbn_LEN_8_TPI_8::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_8_TPI_8::mpmul32(plo, denom, correction);
            t=cgbn_LEN_8_TPI_8::core::resolve_add_b(c, plo);
            c=cgbn_LEN_8_TPI_8::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_8_TPI_8::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_8_TPI_8::mpadd(x, x, denom);
            cgbn_LEN_8_TPI_8::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_8_TPI_8::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_8_TPI_8::core::clz(b_tmp);
        cgbn_LEN_8_TPI_8::core::rotate_left(cgbn_LEN_8_TPI_8::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_8_TPI_8::core::rotate_left(cgbn_LEN_8_TPI_8::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_8_TPI_8::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_8_TPI_8::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_TWO-cgbn_LEN_8_TPI_8::core::clzt(num_high);
        cgbn_mod_LEN_8_TPI_8(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_8_TPI_8::core::rotate_right(cgbn_LEN_8_TPI_8::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_8_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_8_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE];

        if(numthreads<TPI_TWO) {
            cgbn_LEN_8_TPI_8::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_TWO);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_8_TPI_8::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_ONE;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_TWO-1, TPI_TWO);

        cgbn_LEN_8_TPI_8::dlimbs_scatter(dtemp, denom, TPI_TWO-1);  
        cgbn_LEN_8_TPI_8::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_8_TPI_8::dlimbs_scatter(dtemp, x, TPI_TWO-1);
            cgbn_LEN_8_TPI_8::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_8_TPI_8::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_8_TPI_8::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_8_TPI_8::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_TWO-1, TPI_TWO);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_TWO);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_TWO);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_TWO);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_8_TPI_8::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_8_TPI_8::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_TWO-1, TPI_TWO);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_TWO-1, TPI_TWO);

            correction=cgbn_LEN_8_TPI_8::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_8_TPI_8::mpmul32(plo, denom, correction);
            t=cgbn_LEN_8_TPI_8::core::resolve_add_b(c, plo);
            c=cgbn_LEN_8_TPI_8::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_8_TPI_8::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_8_TPI_8::mpadd(x, x, denom);
            cgbn_LEN_8_TPI_8::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=x[index];
    }

// LEN = 16 TPI = 8 LIMBS = 2 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_16_TPI_8(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_TWO)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_TWO-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        cgbn_LEN_16_TPI_8::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn_LEN_16_TPI_8::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        
        if(TPI_TWO<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);   
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_16_TPI_8::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_16_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]) {  
        uint32_t carry;
        cgbn_LEN_16_TPI_8::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);
        int32_t sr =  -cgbn_LEN_16_TPI_8::fast_propagate_sub(carry, r);    
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_16_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_16_TPI_8::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_16_TPI_8::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt_LEN_16_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        uint32_t sync=cgbn_LEN_16_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS_TWO], ra[LIMBS_TWO+2], ru[LIMBS_TWO+1];
        int32_t  threads = TPI_TWO;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            ra[index]=add[index];

        ra[LIMBS_TWO]=0;
        ra[LIMBS_TWO+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            ru[index]=add[index];

        ru[LIMBS_TWO]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS_TWO*2;l+=2) {
            if(l<LIMBS_TWO) 
                term0=__shfl_sync(sync, b[l], row, TPI_TWO);
            else
                term0=__shfl_sync(sync, b[l-LIMBS_TWO], row+1, TPI_TWO);
            if(l+1<LIMBS_TWO)
                term1=__shfl_sync(sync, b[l+1], row, TPI_TWO);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS_TWO], row+1, TPI_TWO);

            cgbn_LEN_16_TPI_8::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS_TWO%2==0)
                ra[LIMBS_TWO]=chain1.add(ra[LIMBS_TWO], 0);      
            
            cgbn_LEN_16_TPI_8::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS_TWO%2==1)
                ru[LIMBS_TWO-1]=chain2.add(0, 0);

            cgbn_LEN_16_TPI_8::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS_TWO%2==1)
                ru[LIMBS_TWO-1]=0;
            else 
                ru[LIMBS_TWO-2]=chain3.add(0, 0);
            ru[LIMBS_TWO-1+LIMBS_TWO%2]=0;

            cgbn_LEN_16_TPI_8::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS_TWO-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS_TWO-2-LIMBS_TWO%2]=chain4.madlo(a[LIMBS_TWO-1-LIMBS_TWO%2], term1, ra[LIMBS_TWO-LIMBS_TWO%2]);
            ra[LIMBS_TWO-1-LIMBS_TWO%2]=chain4.madhi(a[LIMBS_TWO-1-LIMBS_TWO%2], term1, ra[LIMBS_TWO+1-LIMBS_TWO%2]);
            if(LIMBS_TWO%2==1)
                ra[LIMBS_TWO-1]=chain4.add(0, 0);

            if(l<LIMBS_TWO) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI_TWO);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS_TWO]=t0;
                rl[l-LIMBS_TWO]=__shfl_sync(sync, rl[l-LIMBS_TWO], threadIdx.x+1, TPI_TWO);
                t0=rl[l-LIMBS_TWO];
            }
            if(l+1<LIMBS_TWO) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI_TWO);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS_TWO]=t1;
                rl[l-LIMBS_TWO+1]=__shfl_sync(sync, rl[l+1-LIMBS_TWO], threadIdx.x+1, TPI_TWO);
                t1=rl[l+1-LIMBS_TWO];
            }
                    
            ra[LIMBS_TWO-2]=cgbn_LEN_16_TPI_8::add_cc(ra[LIMBS_TWO-2], t0);
            ra[LIMBS_TWO-1]=cgbn_LEN_16_TPI_8::addc_cc(ra[LIMBS_TWO-1], t1);
            ra[LIMBS_TWO]=cgbn_LEN_16_TPI_8::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_8(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_TWO, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_8(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_TWO, add);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_16_TPI_8(cgbn_LEN_16_TPI_8::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_16_TPI_8(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_16_TPI_8(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_16_TPI_8(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_8(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_TWO, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_TWO] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_16_TPI_8(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_TWO, add);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_16_TPI_8(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_16_TPI_8(cgbn_LEN_16_TPI_8::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_16_TPI_8(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_16_TPI_8(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_TWO];
        uint32_t a_tmp[LIMBS_TWO];
        uint32_t b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_16_TPI_8(r, a_tmp, b_tmp, add);
    }

    ARIES_DEVICE  void operator_div_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;

        uint32_t add[LIMBS_TWO], a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_mlt_LEN_16_TPI_8(a_tmp, a_tmp, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS_TWO, add);

        uint32_t num_low[LIMBS_TWO], num_high[LIMBS_TWO], denom_local[LIMBS_TWO];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_16_TPI_8::core::clz(b_tmp);

        cgbn_LEN_16_TPI_8::core::rotate_left(cgbn_LEN_16_TPI_8::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_16_TPI_8::core::rotate_left(cgbn_LEN_16_TPI_8::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_16_TPI_8::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_TWO-cgbn_LEN_16_TPI_8::core::clzt(num_high);
        cgbn_div_LEN_16_TPI_8(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_16_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_16_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_TWO], y[LIMBS_TWO], plo[LIMBS_TWO], phi[LIMBS_TWO], quotient[LIMBS_TWO];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            quotient[index] = 0;

        if(numthreads<TPI_TWO) {
            cgbn_LEN_16_TPI_8::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_TWO);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_16_TPI_8::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_TWO-1], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[LIMBS_TWO-2], TPI_TWO-1, TPI_TWO);

        cgbn_LEN_16_TPI_8::dlimbs_scatter(dtemp, denom, TPI_TWO-1);  
        cgbn_LEN_16_TPI_8::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_16_TPI_8::dlimbs_scatter(dtemp, x, TPI_TWO-1);
            cgbn_LEN_16_TPI_8::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_16_TPI_8::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_16_TPI_8::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_16_TPI_8::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_TWO-1, TPI_TWO);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_TWO);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_TWO);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_TWO);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_16_TPI_8::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_16_TPI_8::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_TWO-1], TPI_TWO-1, TPI_TWO);
            x0=__shfl_sync(sync, x[LIMBS_TWO-2], TPI_TWO-1, TPI_TWO);
            
            correction=cgbn_LEN_16_TPI_8::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_16_TPI_8::mpmul32(plo, denom, correction);
            t=cgbn_LEN_16_TPI_8::core::resolve_add_b(c, plo);
            c=cgbn_LEN_16_TPI_8::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_16_TPI_8::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_16_TPI_8::mpadd(x, x, denom);
            cgbn_LEN_16_TPI_8::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_16_TPI_8::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_TWO], num_high[LIMBS_TWO], denom_local[LIMBS_TWO];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_16_TPI_8::core::clz(b_tmp);
        cgbn_LEN_16_TPI_8::core::rotate_left(cgbn_LEN_16_TPI_8::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_16_TPI_8::core::rotate_left(cgbn_LEN_16_TPI_8::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_16_TPI_8::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_16_TPI_8::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_TWO-cgbn_LEN_16_TPI_8::core::clzt(num_high);
        cgbn_mod_LEN_16_TPI_8(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_16_TPI_8::core::rotate_right(cgbn_LEN_16_TPI_8::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_16_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_16_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_TWO], y[LIMBS_TWO], plo[LIMBS_TWO], phi[LIMBS_TWO];

        if(numthreads<TPI_TWO) {
            cgbn_LEN_16_TPI_8::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_TWO);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_16_TPI_8::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_TWO;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_TWO-1], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[LIMBS_TWO-2], TPI_TWO-1, TPI_TWO);

        cgbn_LEN_16_TPI_8::dlimbs_scatter(dtemp, denom, TPI_TWO-1);  
        cgbn_LEN_16_TPI_8::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_16_TPI_8::dlimbs_scatter(dtemp, x, TPI_TWO-1);
            cgbn_LEN_16_TPI_8::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_16_TPI_8::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_16_TPI_8::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_16_TPI_8::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_TWO-1, TPI_TWO);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_TWO);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_TWO);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_TWO);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_16_TPI_8::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_16_TPI_8::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_TWO-1], TPI_TWO-1, TPI_TWO);
            x0=__shfl_sync(sync, x[LIMBS_TWO-2], TPI_TWO-1, TPI_TWO);

            correction=cgbn_LEN_16_TPI_8::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_16_TPI_8::mpmul32(plo, denom, correction);
            t=cgbn_LEN_16_TPI_8::core::resolve_add_b(c, plo);
            c=cgbn_LEN_16_TPI_8::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_16_TPI_8::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_16_TPI_8::mpadd(x, x, denom);
            cgbn_LEN_16_TPI_8::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            q[index]=x[index];
    }

// LEN = 32 TPI = 8 LIMBS = 4 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_32_TPI_8(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_TWO)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_TWO-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        cgbn_LEN_32_TPI_8::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn_LEN_32_TPI_8::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        
        if(TPI_TWO<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);   
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_32_TPI_8::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_32_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]) {  
        uint32_t carry;
        cgbn_LEN_32_TPI_8::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);
        int32_t sr =  -cgbn_LEN_32_TPI_8::fast_propagate_sub(carry, r);    
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_32_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_32_TPI_8::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_32_TPI_8::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt_LEN_32_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        uint32_t sync=cgbn_LEN_32_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS_THR], ra[LIMBS_THR+2], ru[LIMBS_THR+1];
        int32_t  threads = TPI_TWO;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            ra[index]=add[index];

        ra[LIMBS_THR]=0;
        ra[LIMBS_THR+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            ru[index]=add[index];

        ru[LIMBS_THR]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS_THR*2;l+=2) {
            if(l<LIMBS_THR) 
                term0=__shfl_sync(sync, b[l], row, TPI_TWO);
            else
                term0=__shfl_sync(sync, b[l-LIMBS_THR], row+1, TPI_TWO);
            if(l+1<LIMBS_THR)
                term1=__shfl_sync(sync, b[l+1], row, TPI_TWO);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS_THR], row+1, TPI_TWO);

            cgbn_LEN_32_TPI_8::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS_THR%2==0)
                ra[LIMBS_THR]=chain1.add(ra[LIMBS_THR], 0);      
            
            cgbn_LEN_32_TPI_8::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS_THR%2==1)
                ru[LIMBS_THR-1]=chain2.add(0, 0);

            cgbn_LEN_32_TPI_8::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS_THR%2==1)
                ru[LIMBS_THR-1]=0;
            else 
                ru[LIMBS_THR-2]=chain3.add(0, 0);
            ru[LIMBS_THR-1+LIMBS_THR%2]=0;

            cgbn_LEN_32_TPI_8::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS_THR-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS_THR-2-LIMBS_THR%2]=chain4.madlo(a[LIMBS_THR-1-LIMBS_THR%2], term1, ra[LIMBS_THR-LIMBS_THR%2]);
            ra[LIMBS_THR-1-LIMBS_THR%2]=chain4.madhi(a[LIMBS_THR-1-LIMBS_THR%2], term1, ra[LIMBS_THR+1-LIMBS_THR%2]);
            if(LIMBS_THR%2==1)
                ra[LIMBS_THR-1]=chain4.add(0, 0);

            if(l<LIMBS_THR) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI_TWO);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS_THR]=t0;
                rl[l-LIMBS_THR]=__shfl_sync(sync, rl[l-LIMBS_THR], threadIdx.x+1, TPI_TWO);
                t0=rl[l-LIMBS_THR];
            }
            if(l+1<LIMBS_THR) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI_TWO);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS_THR]=t1;
                rl[l-LIMBS_THR+1]=__shfl_sync(sync, rl[l+1-LIMBS_THR], threadIdx.x+1, TPI_TWO);
                t1=rl[l+1-LIMBS_THR];
            }
                    
            ra[LIMBS_THR-2]=cgbn_LEN_32_TPI_8::add_cc(ra[LIMBS_THR-2], t0);
            ra[LIMBS_THR-1]=cgbn_LEN_32_TPI_8::addc_cc(ra[LIMBS_THR-1], t1);
            ra[LIMBS_THR]=cgbn_LEN_32_TPI_8::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_THR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_8(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_THR, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_THR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_8(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_THR, add);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_32_TPI_8(cgbn_LEN_32_TPI_8::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_32_TPI_8(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_32_TPI_8(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_32_TPI_8(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;
        
        uint32_t a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        
        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_THR];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_8(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_THR, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_THR] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_8(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_THR, add);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_32_TPI_8(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_32_TPI_8(cgbn_LEN_32_TPI_8::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_32_TPI_8(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_32_TPI_8(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_THR];
        uint32_t a_tmp[LIMBS_THR];
        uint32_t b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_32_TPI_8(r, a_tmp, b_tmp, add);
    }

    ARIES_DEVICE  void operator_div_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_TWO-1;

        uint32_t add[LIMBS_THR], a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_32_TPI_8(a_tmp, a_tmp, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS_THR, add);

        uint32_t num_low[LIMBS_THR], num_high[LIMBS_THR], denom_local[LIMBS_THR];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_32_TPI_8::core::clz(b_tmp);

        cgbn_LEN_32_TPI_8::core::rotate_left(cgbn_LEN_32_TPI_8::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_8::core::rotate_left(cgbn_LEN_32_TPI_8::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_8::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_TWO-cgbn_LEN_32_TPI_8::core::clzt(num_high);
        cgbn_div_LEN_32_TPI_8(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_32_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_THR], y[LIMBS_THR], plo[LIMBS_THR], phi[LIMBS_THR], quotient[LIMBS_THR];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            quotient[index] = 0;

        if(numthreads<TPI_TWO) {
            cgbn_LEN_32_TPI_8::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_TWO);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_8::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_THR-1], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[LIMBS_THR-2], TPI_TWO-1, TPI_TWO);

        cgbn_LEN_32_TPI_8::dlimbs_scatter(dtemp, denom, TPI_TWO-1);  
        cgbn_LEN_32_TPI_8::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_8::dlimbs_scatter(dtemp, x, TPI_TWO-1);
            cgbn_LEN_32_TPI_8::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_8::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_8::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_8::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_TWO-1, TPI_TWO);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_TWO);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_TWO);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_TWO);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_8::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_8::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_THR-1], TPI_TWO-1, TPI_TWO);
            x0=__shfl_sync(sync, x[LIMBS_THR-2], TPI_TWO-1, TPI_TWO);
            
            correction=cgbn_LEN_32_TPI_8::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_8::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_8::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_8::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_8::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_8::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_8::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_32_TPI_8::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_THR], num_high[LIMBS_THR], denom_local[LIMBS_THR];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_THR], b_tmp[LIMBS_THR];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_32_TPI_8::core::clz(b_tmp);
        cgbn_LEN_32_TPI_8::core::rotate_left(cgbn_LEN_32_TPI_8::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_8::core::rotate_left(cgbn_LEN_32_TPI_8::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_8::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_32_TPI_8::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_TWO-cgbn_LEN_32_TPI_8::core::clzt(num_high);
        cgbn_mod_LEN_32_TPI_8(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_32_TPI_8::core::rotate_right(cgbn_LEN_32_TPI_8::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_32_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_8::core::sync_mask(), group_thread=threadIdx.x & TPI_TWO-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_THR], y[LIMBS_THR], plo[LIMBS_THR], phi[LIMBS_THR];

        if(numthreads<TPI_TWO) {
            cgbn_LEN_32_TPI_8::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_TWO);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_8::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_THR;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_THR-1], TPI_TWO-1, TPI_TWO);
        d0=__shfl_sync(sync, denom[LIMBS_THR-2], TPI_TWO-1, TPI_TWO);

        cgbn_LEN_32_TPI_8::dlimbs_scatter(dtemp, denom, TPI_TWO-1);  
        cgbn_LEN_32_TPI_8::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_8::dlimbs_scatter(dtemp, x, TPI_TWO-1);
            cgbn_LEN_32_TPI_8::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_8::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_8::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_8::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_TWO-1, TPI_TWO);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_THR;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_TWO);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_TWO);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_TWO);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_8::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_8::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_THR-1], TPI_TWO-1, TPI_TWO);
            x0=__shfl_sync(sync, x[LIMBS_THR-2], TPI_TWO-1, TPI_TWO);

            correction=cgbn_LEN_32_TPI_8::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_8::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_8::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_8::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_8::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_8::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_8::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_THR;index++)
            q[index]=x[index];
    }

// LEN = 16 TPI = 16 LIMBS = 1 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_16_TPI_16(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_THR)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_THR-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        a_ballot=__ballot_sync(sync, a[0]>=b[0]);
        b_ballot=__ballot_sync(sync, a[0]<=b[0]);

        if(TPI_THR<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_16_TPI_16::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_16_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        
        uint32_t carry;
    
        cgbn_LEN_16_TPI_16::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
        r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);

        int32_t sr =  -cgbn_LEN_16_TPI_16::fast_propagate_sub(carry, r);
        
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_16_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_16_TPI_16::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_16_TPI_16::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_sig_LEN_16_TPI_16(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add){
        uint32_t sync=cgbn_LEN_16_TPI_16::core::sync_mask(), group_thread=threadIdx.x & TPI_THR-1;
        uint32_t rl, p0=add, p1=0, t;
        int32_t  threads = TPI_THR;

        #pragma unroll
        for(int32_t index=0;index<threads;index++) {
            t=__shfl_sync(sync, b, index, TPI_THR);

            p0=cgbn_LEN_16_TPI_16::madlo_cc(a, t, p0);
            p1=cgbn_LEN_16_TPI_16::addc(p1, 0);
            
            if(group_thread<threads-index) 
            rl=p0;

            rl=__shfl_sync(sync, rl, threadIdx.x+1, TPI_THR);

            p0=cgbn_LEN_16_TPI_16::madhi_cc(a, t, p1);
            p1=cgbn_LEN_16_TPI_16::addc(0, 0);
            
            p0=cgbn_LEN_16_TPI_16::add_cc(p0, rl);
            p1=cgbn_LEN_16_TPI_16::addc(p1, 0);

        }
        r=rl;
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_THR-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

       // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_16_TPI_16(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_16_TPI_16(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_16_TPI_16(cgbn_LEN_16_TPI_16::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_16_TPI_16(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_16_TPI_16(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_16_TPI_16(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_THR-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_16_TPI_16(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_ONE] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_16_TPI_16(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_16_TPI_16(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_16_TPI_16(cgbn_LEN_16_TPI_16::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_16_TPI_16(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_16_TPI_16(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_ONE];
        uint32_t a_tmp[LIMBS_ONE];
        uint32_t b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        cgbn_mul_sig_LEN_16_TPI_16(r[0], a_tmp[0], b_tmp[0], add[0]);
    }

    ARIES_DEVICE  void operator_div_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_THR-1;

        uint32_t add[LIMBS_ONE], a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_sig_LEN_16_TPI_16(a_tmp[0], a_tmp[0], __POW10_ARRAY[divitend_shift-1][group_thread*LIMBS_ONE],  add[0]);

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_16_TPI_16::core::clz(b_tmp);

        cgbn_LEN_16_TPI_16::core::rotate_left(cgbn_LEN_16_TPI_16::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_16_TPI_16::core::rotate_left(cgbn_LEN_16_TPI_16::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_16_TPI_16::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_THR-cgbn_LEN_16_TPI_16::core::clzt(num_high);
        cgbn_div_LEN_16_TPI_16(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_16_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_16_TPI_16::core::sync_mask(), group_thread=threadIdx.x & TPI_THR-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE], quotient[LIMBS_ONE];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            quotient[index] = 0;

        if(numthreads<TPI_THR) {
            cgbn_LEN_16_TPI_16::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
                x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
                x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_THR);
                y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_16_TPI_16::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=y[index];
                quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_THR-1, TPI_THR);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_THR-1, TPI_THR);

        cgbn_LEN_16_TPI_16::dlimbs_scatter(dtemp, denom, TPI_THR-1);  
        cgbn_LEN_16_TPI_16::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_16_TPI_16::dlimbs_scatter(dtemp, x, TPI_THR-1);
            cgbn_LEN_16_TPI_16::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_16_TPI_16::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_16_TPI_16::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_16_TPI_16::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_THR-1, TPI_THR);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_THR);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_THR);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_THR);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_16_TPI_16::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_16_TPI_16::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_THR-1, TPI_THR);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_THR-1, TPI_THR);
            
            correction=cgbn_LEN_16_TPI_16::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_16_TPI_16::mpmul32(plo, denom, correction);
            t=cgbn_LEN_16_TPI_16::core::resolve_add_b(c, plo);
            c=cgbn_LEN_16_TPI_16::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_16_TPI_16::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_16_TPI_16::mpadd(x, x, denom);
            cgbn_LEN_16_TPI_16::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_16_TPI_16::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_16_TPI_16::core::clz(b_tmp);
        cgbn_LEN_16_TPI_16::core::rotate_left(cgbn_LEN_16_TPI_16::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_16_TPI_16::core::rotate_left(cgbn_LEN_16_TPI_16::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_16_TPI_16::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_16_TPI_16::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_THR-cgbn_LEN_16_TPI_16::core::clzt(num_high);
        cgbn_mod_LEN_16_TPI_16(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_16_TPI_16::core::rotate_right(cgbn_LEN_16_TPI_16::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_16_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_16_TPI_16::core::sync_mask(), group_thread=threadIdx.x & TPI_THR-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE];

        if(numthreads<TPI_THR) {
            cgbn_LEN_16_TPI_16::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_THR);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_16_TPI_16::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_ONE;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_THR-1, TPI_THR);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_THR-1, TPI_THR);

        cgbn_LEN_16_TPI_16::dlimbs_scatter(dtemp, denom, TPI_THR-1);  
        cgbn_LEN_16_TPI_16::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_16_TPI_16::dlimbs_scatter(dtemp, x, TPI_THR-1);
            cgbn_LEN_16_TPI_16::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_16_TPI_16::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_16_TPI_16::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_16_TPI_16::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_THR-1, TPI_THR);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_THR);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_THR);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_THR);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_16_TPI_16::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_16_TPI_16::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_THR-1, TPI_THR);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_THR-1, TPI_THR);

            correction=cgbn_LEN_16_TPI_16::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_16_TPI_16::mpmul32(plo, denom, correction);
            t=cgbn_LEN_16_TPI_16::core::resolve_add_b(c, plo);
            c=cgbn_LEN_16_TPI_16::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_16_TPI_16::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_16_TPI_16::mpadd(x, x, denom);
            cgbn_LEN_16_TPI_16::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=x[index];
    }

// LEN = 32 TPI = 16 LIMBS = 2 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_32_TPI_16(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_THR)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_THR-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        cgbn_LEN_32_TPI_16::chain_t<> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            chain1.sub(a[index], b[index]);
        a_ballot=chain1.sub(0, 0);
        a_ballot=__ballot_sync(sync, a_ballot==0);
        
        cgbn_LEN_32_TPI_16::chain_t<> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            chain2.sub(b[index], a[index]);
        b_ballot=chain2.sub(0, 0);
        b_ballot=__ballot_sync(sync, b_ballot==0);
        
        if(TPI_THR<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);   
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_32_TPI_16::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_32_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]) {  
        uint32_t carry;
        cgbn_LEN_32_TPI_16::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);
        int32_t sr =  -cgbn_LEN_32_TPI_16::fast_propagate_sub(carry, r);    
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_32_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_32_TPI_16::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_32_TPI_16::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_mlt_LEN_32_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]) {
        uint32_t sync=cgbn_LEN_32_TPI_16::core::sync_mask(), group_thread=threadIdx.x & TPI_THR-1;
        uint32_t t0, t1, term0, term1, carry, rl[LIMBS_TWO], ra[LIMBS_TWO+2], ru[LIMBS_TWO+1];
        int32_t  threads = TPI_THR;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            ra[index]=add[index];

        ra[LIMBS_TWO]=0;
        ra[LIMBS_TWO+1]=0;

        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            ru[index]=add[index];

        ru[LIMBS_TWO]=0;
        
        carry=0;
        #pragma unroll(1) //此处不能识别 #pragma nounroll ???
        for(int32_t row=0;row<threads;row+=2) {
            #pragma unroll
            for(int32_t l=0;l<LIMBS_TWO*2;l+=2) {
            if(l<LIMBS_TWO) 
                term0=__shfl_sync(sync, b[l], row, TPI_THR);
            else
                term0=__shfl_sync(sync, b[l-LIMBS_TWO], row+1, TPI_THR);
            if(l+1<LIMBS_TWO)
                term1=__shfl_sync(sync, b[l+1], row, TPI_THR);
            else
                term1=__shfl_sync(sync, b[l+1-LIMBS_TWO], row+1, TPI_THR);

            cgbn_LEN_32_TPI_16::chain_t<> chain1;                               // aligned:   T0 * A_even
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index+=2) {
                ra[index]=chain1.madlo(a[index], term0, ra[index]);
                ra[index+1]=chain1.madhi(a[index], term0, ra[index+1]);
            }
            if(LIMBS_TWO%2==0)
                ra[LIMBS_TWO]=chain1.add(ra[LIMBS_TWO], 0);      
            
            cgbn_LEN_32_TPI_16::chain_t<> chain2;                               // unaligned: T0 * A_odd
            t0=chain2.add(ra[0], carry);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO-1;index+=2) {
                ru[index]=chain2.madlo(a[index+1], term0, ru[index]);
                ru[index+1]=chain2.madhi(a[index+1], term0, ru[index+1]);
            }
            if(LIMBS_TWO%2==1)
                ru[LIMBS_TWO-1]=chain2.add(0, 0);

            cgbn_LEN_32_TPI_16::chain_t<> chain3;                               // unaligned: T1 * A_even
            t1=chain3.madlo(a[0], term1, ru[0]);
            carry=chain3.madhi(a[0], term1, ru[1]);
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO-2;index+=2) {
                ru[index]=chain3.madlo(a[index+2], term1, ru[index+2]);
                ru[index+1]=chain3.madhi(a[index+2], term1, ru[index+3]);
            }
            if(LIMBS_TWO%2==1)
                ru[LIMBS_TWO-1]=0;
            else 
                ru[LIMBS_TWO-2]=chain3.add(0, 0);
            ru[LIMBS_TWO-1+LIMBS_TWO%2]=0;

            cgbn_LEN_32_TPI_16::chain_t<> chain4;                               // aligned:   T1 * A_odd
            t1=chain4.add(t1, ra[1]);
            #pragma unroll
            for(int32_t index=0;index<(int32_t)(LIMBS_TWO-3);index+=2) {
                ra[index]=chain4.madlo(a[index+1], term1, ra[index+2]);
                ra[index+1]=chain4.madhi(a[index+1], term1, ra[index+3]);
            }
            ra[LIMBS_TWO-2-LIMBS_TWO%2]=chain4.madlo(a[LIMBS_TWO-1-LIMBS_TWO%2], term1, ra[LIMBS_TWO-LIMBS_TWO%2]);
            ra[LIMBS_TWO-1-LIMBS_TWO%2]=chain4.madhi(a[LIMBS_TWO-1-LIMBS_TWO%2], term1, ra[LIMBS_TWO+1-LIMBS_TWO%2]);
            if(LIMBS_TWO%2==1)
                ra[LIMBS_TWO-1]=chain4.add(0, 0);

            if(l<LIMBS_TWO) {
                if(group_thread<threads-row)
                rl[l]=t0;
                rl[l]=__shfl_sync(sync, rl[l], threadIdx.x+1, TPI_THR);
                t0=rl[l];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l-LIMBS_TWO]=t0;
                rl[l-LIMBS_TWO]=__shfl_sync(sync, rl[l-LIMBS_TWO], threadIdx.x+1, TPI_THR);
                t0=rl[l-LIMBS_TWO];
            }
            if(l+1<LIMBS_TWO) {
                if(group_thread<threads-row)
                rl[l+1]=t1;
                rl[l+1]=__shfl_sync(sync, rl[l+1], threadIdx.x+1, TPI_THR);
                t1=rl[l+1];
            }
            else {
                if(group_thread<threads-1-row)
                rl[l+1-LIMBS_TWO]=t1;
                rl[l-LIMBS_TWO+1]=__shfl_sync(sync, rl[l+1-LIMBS_TWO], threadIdx.x+1, TPI_THR);
                t1=rl[l+1-LIMBS_TWO];
            }
                    
            ra[LIMBS_TWO-2]=cgbn_LEN_32_TPI_16::add_cc(ra[LIMBS_TWO-2], t0);
            ra[LIMBS_TWO-1]=cgbn_LEN_32_TPI_16::addc_cc(ra[LIMBS_TWO-1], t1);
            ra[LIMBS_TWO]=cgbn_LEN_32_TPI_16::addc(0, 0);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            r[index]=rl[index];
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_THR-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_16(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_TWO, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_16(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_TWO, add);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_32_TPI_16(cgbn_LEN_32_TPI_16::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_32_TPI_16(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_32_TPI_16(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_32_TPI_16(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_THR-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_TWO];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_16(b_tmp, b_tmp, __POW10_ARRAY[shift-1]+group_thread*LIMBS_TWO, add);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_TWO] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                add[index]=0;

            cgbn_mul_mlt_LEN_32_TPI_16(a_tmp, a_tmp, __POW10_ARRAY[-shift-1]+group_thread*LIMBS_TWO, add);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_32_TPI_16(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_32_TPI_16(cgbn_LEN_32_TPI_16::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_32_TPI_16(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_32_TPI_16(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_TWO];
        uint32_t a_tmp[LIMBS_TWO];
        uint32_t b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_32_TPI_16(r, a_tmp, b_tmp, add);
    }

    ARIES_DEVICE  void operator_div_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_THR-1;

        uint32_t add[LIMBS_TWO], a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_mlt_LEN_32_TPI_16(a_tmp, a_tmp, __POW10_ARRAY[divitend_shift-1]+group_thread*LIMBS_TWO, add);

        uint32_t num_low[LIMBS_TWO], num_high[LIMBS_TWO], denom_local[LIMBS_TWO];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_32_TPI_16::core::clz(b_tmp);

        cgbn_LEN_32_TPI_16::core::rotate_left(cgbn_LEN_32_TPI_16::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_16::core::rotate_left(cgbn_LEN_32_TPI_16::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_16::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_THR-cgbn_LEN_32_TPI_16::core::clzt(num_high);
        cgbn_div_LEN_32_TPI_16(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_32_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_16::core::sync_mask(), group_thread=threadIdx.x & TPI_THR-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_TWO], y[LIMBS_TWO], plo[LIMBS_TWO], phi[LIMBS_TWO], quotient[LIMBS_TWO];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            quotient[index] = 0;

        if(numthreads<TPI_THR) {
            cgbn_LEN_32_TPI_16::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_THR);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_16::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                x[index]=y[index];
            quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
            x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_TWO-1], TPI_THR-1, TPI_THR);
        d0=__shfl_sync(sync, denom[LIMBS_TWO-2], TPI_THR-1, TPI_THR);

        cgbn_LEN_32_TPI_16::dlimbs_scatter(dtemp, denom, TPI_THR-1);  
        cgbn_LEN_32_TPI_16::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_16::dlimbs_scatter(dtemp, x, TPI_THR-1);
            cgbn_LEN_32_TPI_16::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_16::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_16::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_16::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_THR-1, TPI_THR);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_THR);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_THR);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_THR);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_16::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_16::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_TWO-1], TPI_THR-1, TPI_THR);
            x0=__shfl_sync(sync, x[LIMBS_TWO-2], TPI_THR-1, TPI_THR);
            
            correction=cgbn_LEN_32_TPI_16::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_16::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_16::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_16::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_16::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_16::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_16::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_32_TPI_16::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_TWO], num_high[LIMBS_TWO], denom_local[LIMBS_TWO];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_TWO], b_tmp[LIMBS_TWO];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_32_TPI_16::core::clz(b_tmp);
        cgbn_LEN_32_TPI_16::core::rotate_left(cgbn_LEN_32_TPI_16::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_16::core::rotate_left(cgbn_LEN_32_TPI_16::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_16::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_32_TPI_16::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_THR-cgbn_LEN_32_TPI_16::core::clzt(num_high);
        cgbn_mod_LEN_32_TPI_16(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_32_TPI_16::core::rotate_right(cgbn_LEN_32_TPI_16::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_32_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_16::core::sync_mask(), group_thread=threadIdx.x & TPI_THR-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_TWO], y[LIMBS_TWO], plo[LIMBS_TWO], phi[LIMBS_TWO];

        if(numthreads<TPI_THR) {
            cgbn_LEN_32_TPI_16::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_THR);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_16::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_TWO;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_TWO-1], TPI_THR-1, TPI_THR);
        d0=__shfl_sync(sync, denom[LIMBS_TWO-2], TPI_THR-1, TPI_THR);

        cgbn_LEN_32_TPI_16::dlimbs_scatter(dtemp, denom, TPI_THR-1);  
        cgbn_LEN_32_TPI_16::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_16::dlimbs_scatter(dtemp, x, TPI_THR-1);
            cgbn_LEN_32_TPI_16::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_16::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_16::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_16::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_THR-1, TPI_THR);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_TWO;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_THR);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_THR);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_THR);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_16::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_16::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_TWO-1], TPI_THR-1, TPI_THR);
            x0=__shfl_sync(sync, x[LIMBS_TWO-2], TPI_THR-1, TPI_THR);

            correction=cgbn_LEN_32_TPI_16::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_16::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_16::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_16::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_16::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_16::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_16::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_TWO;index++)
            q[index]=x[index];
    }

// LEN = 32 TPI = 32 LIMBS = 1 DLIMBS = 1
    ARIES_DEVICE_FORCE int32_t  cgbn_compare_LEN_32_TPI_32(const uint32_t sync, const uint32_t a[], const uint32_t b[]){
        static const uint32_t TPI_ONES=(1ull<<TPI_FOR)-1;
        
        uint32_t group_thread=threadIdx.x & TPI_FOR-1, warp_thread=threadIdx.x & warpSize-1;
        uint32_t a_ballot, b_ballot;

        a_ballot=__ballot_sync(sync, a[0]>=b[0]);
        b_ballot=__ballot_sync(sync, a[0]<=b[0]);

        if(TPI_FOR<warpSize) {
            uint32_t mask=TPI_ONES<<(warp_thread ^ group_thread);
            a_ballot=a_ballot & mask;
            b_ballot=b_ballot & mask;
        }
        
        return cgbn_LEN_32_TPI_32::ucmp(a_ballot, b_ballot);
    }

    ARIES_DEVICE_FORCE int32_t cgbn_sub_LEN_32_TPI_32(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        
        uint32_t carry;
    
        cgbn_LEN_32_TPI_32::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
        r[index]=chain.sub(a[index], b[index]);
        carry=chain.sub(0, 0);

        int32_t sr =  -cgbn_LEN_32_TPI_32::fast_propagate_sub(carry, r);
        
        return sr;
    }

    ARIES_DEVICE_FORCE int32_t cgbn_add_LEN_32_TPI_32(uint32_t r[], const uint32_t a[], const uint32_t b[]) {
        uint32_t carry;
        cgbn_LEN_32_TPI_32::chain_t<> chain;
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            r[index]=chain.add(a[index], b[index]);
        carry=chain.add(0, 0);

        int32_t sr =  cgbn_LEN_32_TPI_32::fast_propagate_add(carry, r);
        return sr;
    }

    ARIES_DEVICE_FORCE void  cgbn_mul_sig_LEN_32_TPI_32(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add){
        uint32_t sync=cgbn_LEN_32_TPI_32::core::sync_mask(), group_thread=threadIdx.x & TPI_FOR-1;
        uint32_t rl, p0=add, p1=0, t;
        int32_t  threads = TPI_FOR;

        #pragma unroll
        for(int32_t index=0;index<threads;index++) {
            t=__shfl_sync(sync, b, index, TPI_FOR);

            p0=cgbn_LEN_32_TPI_32::madlo_cc(a, t, p0);
            p1=cgbn_LEN_32_TPI_32::addc(p1, 0);
            
            if(group_thread<threads-index) 
            rl=p0;

            rl=__shfl_sync(sync, rl, threadIdx.x+1, TPI_FOR);

            p0=cgbn_LEN_32_TPI_32::madhi_cc(a, t, p1);
            p1=cgbn_LEN_32_TPI_32::addc(0, 0);
            
            p0=cgbn_LEN_32_TPI_32::add_cc(p0, rl);
            p1=cgbn_LEN_32_TPI_32::addc(p1, 0);

        }
        r=rl;
    }

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_FOR-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

       // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向右移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_32_TPI_32(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向右移动 -shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_32_TPI_32(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号需要比较大小
            int t = cgbn_compare_LEN_32_TPI_32(cgbn_LEN_32_TPI_32::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b r符号为 a 的符号
                cgbn_sub_LEN_32_TPI_32(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 b 的符号
                cgbn_sub_LEN_32_TPI_32(r, b_tmp, a_tmp);
                ans_sign = b_sign;
            }
        }
        else{
            cgbn_add_LEN_32_TPI_32(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        return ans_sign;
    }

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign){
        int32_t group_thread=threadIdx.x & TPI_FOR-1;
        uint8_t ans_sign = 0;
        bool sign_flag = a_sign ^ b_sign;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        // 根据 shift 判断是否需要移位
        if(shift > 0){
            // 说明 b 数组需要向左移动 shift 个单位
            uint32_t add[LIMBS_ONE];

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_32_TPI_32(b_tmp[0], b_tmp[0], __POW10_ARRAY[shift-1][group_thread*LIMBS_ONE], add[0]);
        }
        if(shift < 0){
            // 说明 a 数组需要向左移动 -shift 个单位
            uint32_t add[LIMBS_ONE] = {0};

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                add[index]=0;

            cgbn_mul_sig_LEN_32_TPI_32(a_tmp[0], a_tmp[0], __POW10_ARRAY[-shift-1][group_thread*LIMBS_ONE], add[0]);
        }

        // 根据 sign_flag 判断是否同号
        if( sign_flag == 1){
            // 如果不同号 则两个相加 符号为 被减数的符号
            cgbn_add_LEN_32_TPI_32(r, a_tmp, b_tmp);
            ans_sign = a_sign;
        }
        else{
            // 如果同号那么需要比较大小
            int t = cgbn_compare_LEN_32_TPI_32(cgbn_LEN_32_TPI_32::core::sync_mask(), a_tmp, b_tmp);
            if(t>0){
                // a 比 b 大 那么 a - b 符号为 a 的符号
                cgbn_sub_LEN_32_TPI_32(r, a_tmp, b_tmp);
                ans_sign = a_sign;
            }
            else{
                // a 比 b 小 那么 a - b 符号为 a 的符号 取反
                cgbn_sub_LEN_32_TPI_32(r, b_tmp, a_tmp);
                ans_sign = !a_sign;
            }
        }
        return ans_sign;
    }

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[]){
        uint32_t add[LIMBS_ONE];
        uint32_t a_tmp[LIMBS_ONE];
        uint32_t b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_sig_LEN_32_TPI_32(r[0], a_tmp[0], b_tmp[0], add[0]);
    }

    ARIES_DEVICE  void operator_div_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift){
        int32_t group_thread=threadIdx.x & TPI_FOR-1;

        uint32_t add[LIMBS_ONE], a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            add[index]=0;
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }
        cgbn_mul_sig_LEN_32_TPI_32(a_tmp[0], a_tmp[0], __POW10_ARRAY[divitend_shift-1][group_thread*LIMBS_ONE],  add[0]);

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        shift=cgbn_LEN_32_TPI_32::core::clz(b_tmp);

        cgbn_LEN_32_TPI_32::core::rotate_left(cgbn_LEN_32_TPI_32::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_32::core::rotate_left(cgbn_LEN_32_TPI_32::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_32::core::bitwise_mask_and(num_high, num_low, shift);
        numthreads=TPI_FOR-cgbn_LEN_32_TPI_32::core::clzt(num_high);
        cgbn_div_LEN_32_TPI_32(r, num_low, num_high, denom_local, numthreads);
    }

    ARIES_DEVICE_FORCE void cgbn_div_LEN_32_TPI_32(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_32::core::sync_mask(), group_thread=threadIdx.x & TPI_FOR-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE], quotient[LIMBS_ONE];
        
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            quotient[index] = 0;

        if(numthreads<TPI_FOR) {
            cgbn_LEN_32_TPI_32::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
                x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
                x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_FOR);
                y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_32::resolve_sub(c, y)==0) {
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=y[index];
                quotient[0]=(group_thread==numthreads) ? 1 : quotient[0];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_FOR-1, TPI_FOR);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_FOR-1, TPI_FOR);

        cgbn_LEN_32_TPI_32::dlimbs_scatter(dtemp, denom, TPI_FOR-1);  
        cgbn_LEN_32_TPI_32::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_32::dlimbs_scatter(dtemp, x, TPI_FOR-1);
            cgbn_LEN_32_TPI_32::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_32::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_32::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_32::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_FOR-1, TPI_FOR);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_FOR);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_FOR);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_FOR);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_32::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_32::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_FOR-1, TPI_FOR);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_FOR-1, TPI_FOR);
            
            correction=cgbn_LEN_32_TPI_32::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_32::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_32::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_32::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_32::fast_propagate_add(c, x);
            }
            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_32::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_32::fast_propagate_add(c, x);
            correction++;
            }
            if(group_thread==thread)
            cgbn_LEN_32_TPI_32::mpsub32(quotient, y, correction);
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=quotient[index];
    }

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[]){

        uint32_t num_low[LIMBS_ONE], num_high[LIMBS_ONE], denom_local[LIMBS_ONE];
        uint32_t shift, numthreads;

        uint32_t a_tmp[LIMBS_ONE], b_tmp[LIMBS_ONE];
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++){
            a_tmp[index] = a[index];
            b_tmp[index] = b[index];
        }

        shift=cgbn_LEN_32_TPI_32::core::clz(b_tmp);
        cgbn_LEN_32_TPI_32::core::rotate_left(cgbn_LEN_32_TPI_32::core::sync_mask(), denom_local, b_tmp, shift);
        cgbn_LEN_32_TPI_32::core::rotate_left(cgbn_LEN_32_TPI_32::core::sync_mask(), num_low, a_tmp, shift);
        cgbn_LEN_32_TPI_32::core::bitwise_mask_and(num_high, num_low, shift);
        cgbn_LEN_32_TPI_32::core::bitwise_xor(num_low, num_low, num_high);
        numthreads=TPI_FOR-cgbn_LEN_32_TPI_32::core::clzt(num_high);
        cgbn_mod_LEN_32_TPI_32(r, num_low, num_high, denom_local, numthreads);
        // padding == 0 的 rotate_right 也就是 drotate_right<TPI, LIMBS, MAX_ROTATION>(core::sync_mask(), r, a, numbits);
        cgbn_LEN_32_TPI_32::core::rotate_right(cgbn_LEN_32_TPI_32::core::sync_mask(), r, r, shift);
    }

    ARIES_DEVICE_FORCE void cgbn_mod_LEN_32_TPI_32(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads) {
        uint32_t sync=cgbn_LEN_32_TPI_32::core::sync_mask(), group_thread=threadIdx.x & TPI_FOR-1;
        int32_t  x2;
        uint32_t dtemp[DLIMBS_ONE], approx[DLIMBS_ONE], estimate[DLIMBS_ONE], t, c, x0, x1, d0, d1, correction;
        uint32_t x[LIMBS_ONE], y[LIMBS_ONE], plo[LIMBS_ONE], phi[LIMBS_ONE];

        if(numthreads<TPI_FOR) {
            cgbn_LEN_32_TPI_32::chain_t<> chain;
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {
            x[index]=(group_thread<numthreads) ? hi[index] : lo[index];
            x[index]=__shfl_sync(sync, x[index], threadIdx.x+numthreads, TPI_FOR);
            y[index]=chain.sub(x[index], denom[index]);
            }   
            c=chain.sub(0, 0);
            
            if(cgbn_LEN_32_TPI_32::resolve_sub(c, y)==0) {
                #pragma unroll
                for(int32_t index=0;index<LIMBS_ONE;index++)
                    x[index]=y[index];
            }
        }
        else{
            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++)
                x[index]=hi[index];
        }

            
        d1=__shfl_sync(sync, denom[LIMBS_ONE-1], TPI_FOR-1, TPI_FOR);
        d0=__shfl_sync(sync, denom[LIMBS_ONE-2], TPI_FOR-1, TPI_FOR);

        cgbn_LEN_32_TPI_32::dlimbs_scatter(dtemp, denom, TPI_FOR-1);  
        cgbn_LEN_32_TPI_32::dlimbs_approximate(approx, dtemp);
            
        // main loop that discovers the quotient
        #pragma unroll(1)
        for(int32_t thread=numthreads-1;thread>=0;thread--) {
            cgbn_LEN_32_TPI_32::dlimbs_scatter(dtemp, x, TPI_FOR-1);
            cgbn_LEN_32_TPI_32::dlimbs_div_estimate(estimate, dtemp, approx);
            cgbn_LEN_32_TPI_32::dlimbs_all_gather(y, estimate);
            
            cgbn_LEN_32_TPI_32::mpmul(plo, phi, y, denom);
            c=cgbn_LEN_32_TPI_32::mpsub(x, x, phi);
            x2=__shfl_sync(sync, x[0], TPI_FOR-1, TPI_FOR);

            #pragma unroll
            for(int32_t index=0;index<LIMBS_ONE;index++) {       // shuffle x up by 1
            t=__shfl_sync(sync, lo[index], thread, TPI_FOR);
            x[index]=__shfl_up_sync(sync, x[index], 1, TPI_FOR);
            x[index]=(group_thread==0) ? t : x[index];
            }
        
            c=__shfl_up_sync(sync, c, 1, TPI_FOR);                // shuffle carry up by 1
            c=(group_thread==0) ? 0 : c;
            c=c+cgbn_LEN_32_TPI_32::mpsub(x, x, plo);
            
            x2=x2+cgbn_LEN_32_TPI_32::resolve_sub(c, x);
            x1=__shfl_sync(sync, x[LIMBS_ONE-1], TPI_FOR-1, TPI_FOR);
            x0=__shfl_sync(sync, x[LIMBS_ONE-2], TPI_FOR-1, TPI_FOR);

            correction=cgbn_LEN_32_TPI_32::ucorrect(x0, x1, x2, d0, d1);
            if(correction!=0) {
            c=cgbn_LEN_32_TPI_32::mpmul32(plo, denom, correction);
            t=cgbn_LEN_32_TPI_32::core::resolve_add_b(c, plo);
            c=cgbn_LEN_32_TPI_32::mpadd(x, x, plo);
            x2=x2+t+cgbn_LEN_32_TPI_32::fast_propagate_add(c, x);
            }

            if(x2<0) {
            // usually the case
            c=cgbn_LEN_32_TPI_32::mpadd(x, x, denom);
            cgbn_LEN_32_TPI_32::fast_propagate_add(c, x);
            }
        }
        #pragma unroll
        for(int32_t index=0;index<LIMBS_ONE;index++)
            q[index]=x[index];
    }
    //below methods are for computing long 10 based integer by char string
#ifdef COMPUTE_BY_STRING
    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::GetDivDecimalStr( char *to)
    {
        int start = -1;
        for( int i = 0; i < NUM_TOTAL_DIG; i++ )
        {
            if (values[i] == 0)
            continue;
            start = i;
            break;
        }
        if( start == -1 )
        {
            aries_strcpy( to, "0");
        }
        else
        {
            aries_sprintf( to, "%d", values[start++] );
            char temp[16];
            for( int i = start; i < NUM_TOTAL_DIG - 1; i++ )
            {
                aries_sprintf( temp, values[i] < 0 ? "%010d" : "%09d", values[i] );
                aries_strcat( to, values[i] < 0 ? temp + 1 : temp );
            }
            //handle last one
            int remainLen = frac % DIG_PER_INT32;
            int end = NUM_TOTAL_DIG - 1;
            aries_sprintf( temp, values[end] < 0 ? "%010d" : "%09d", values[end] );
            aries_strncat( to, values[end] < 0 ? temp + 1 : temp, remainLen );
        }
        return to;
    }

    ARIES_HOST_DEVICE_NO_INLINE int Decimal::Compare( char *cmp1, char *cmp2)
    {
        size_t len1 = aries_strlen(cmp1), len2 = aries_strlen(cmp2);
        if (len1 > len2)
        {
            return 1;
        }
        else if (len1 < len2)
        {
            return -1;
        }
        else
        {
            return aries_strcmp(cmp1, cmp2);
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int Decimal::FindFirstNotOf( char *s, char ch)
    {
        char *p = s;
        if (ch)
        {
            while (*p && *p == ch) ++p;
        }
        return p - s;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::Erase( char *s, int startPos, int n)
    {
        int l = aries_strlen(s);
        if (l <= startPos || n <= 0)
        {
            return s;
        }
        int endPos = startPos + n;
        if (l <= endPos)
        {
            s[startPos] = 0;
        }
        else
        {
            aries_strcpy(s + startPos, s + endPos);
        }
        return s;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::DivInt(char *str1, char *str2, int mode, char * result)
    {
        char quotient[128] =
        {   0}, residue[128] =
        {   0};   //定义商和余数
        int signds = 1, signdt = 1;
        if (*str2 == '0')//判断除数是否为0
        {
            error = ERR_DIV_BY_ZERO;
            aries_strcpy(result, "ERROR!");
            return result;
        }
        if (*str1 == '0')     //判断被除数是否为0
        {
            aries_strcpy(quotient, "0");
            aries_strcpy(residue, "0");
        }
        if (str1[0] == '-')
        {
            ++str1;
            signds *= -1;
            signdt = -1;
        }
        if (str2[0] == '-')
        {
            ++str2;
            signds *= -1;
        }
        int res = Compare(str1, str2);
        if (res < 0)
        {
            aries_strcpy(quotient, "0");
            aries_strcpy(residue, str1);
        }
        else if (res == 0)
        {
            aries_strcpy(quotient, "1");
            aries_strcpy(residue, "0");
        }
        else
        {
            int divitendLen = aries_strlen(str1), divisorLen = aries_strlen(str2);
            char tempstr[128] =
            {   0};
            aries_strncpy(tempstr, str1, divisorLen - 1);
            tempstr[divisorLen] = 0;
            int len = 0;
            //模拟手工除法竖式
            for (int i = divisorLen - 1; i < divitendLen; i++)
            {
                len = aries_strlen(tempstr);
                tempstr[len] = str1[i];
                tempstr[len + 1] = 0;
                Erase(tempstr, 0, FindFirstNotOf(tempstr, '0'));
                if (aries_strlen(tempstr) == 0)
                {
                    aries_strcpy(tempstr, "0");
                }
                for (char ch = '9'; ch >= '0'; ch--) //试商
                {
                    char temp[16];
                    temp[0] = ch;
                    temp[1] = 0;
                    char r[128] =
                    {   0};
                    if( Compare( MulInt( (char *)str2, (char *)temp, r), tempstr ) <= 0 )
                    {
                        len = aries_strlen(quotient);
                        quotient[len] = ch;
                        quotient[len + 1] = 0;
                        SubInt( tempstr, MulInt( str2, temp, r ) , tempstr);
                        break;
                    }
                }
            }
            aries_strcpy(residue, tempstr);
        }
        //去除结果中的前导0
        Erase(quotient, 0, FindFirstNotOf(quotient, '0'));
        if (aries_strlen(quotient) == 0)
        {
            aries_strcpy(quotient, "0");
        }
        if ((signds == -1) && (quotient[0] != '0'))
        {
            InsertCh(quotient, 0, '-');
        }
        if ((signdt == -1) && (residue[0] != '0'))
        {
            InsertCh(residue, 0, '-');
        }
        if (mode == 1)
        {
            aries_strcpy(result, quotient);
        }
        else
        {
            aries_strcpy(result, residue);
        }
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::MulInt(char *str1, char *str2, char * result)
    {
        int sign = 1;
        char str[128] =
        {   0};  //记录当前值
        str[0] = '0';
        if (str1[0] == '-')
        {
            sign *= -1;
            str1++;
        }
        if (str2[0] == '-')
        {
            sign *= -1;
            str2++;
        }
        int i, j;
        size_t L1 = aries_strlen(str1), L2 = aries_strlen(str2);
        for (i = L2 - 1; i >= 0; i--)              //模拟手工乘法竖式
        {
            char tempstr[128] =
            {   0};
            int int1 = 0, int2 = 0, int3 = int(str2[i]) - '0';
            if (int3 != 0)
            {
                for (j = 1; j <= (int)(L2 - 1 - i); j++)
                {
                    tempstr[j - 1] = 0;
                }
                for (j = L1 - 1; j >= 0; j--)
                {
                    int1 = (int3*(int(str1[j]) - '0') + int2) % 10;
                    int2 = (int3*(int(str1[j]) - '0') + int2) / 10;
                    InsertCh(tempstr, 0, char(int1 + '0'));
                }
                if (int2 != 0)
                {
                    InsertCh(tempstr, 0, char(int2 + '0'));
                }
            }
            AddInt(str, tempstr, str);
        }
        //去除结果中的前导0
        Erase(str, 0, FindFirstNotOf(str, '0'));
        if (aries_strlen(str) == 0)
        {
            aries_strcpy(str, "0");
        }
        if ((sign == -1) && (str[0] != '0'))
        {
            InsertCh(str, 0, '-');
        }

        aries_strcpy(result, str);
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::SubInt(char *str1, char *str2, char *result)
    {
        int sign = 1; //sign为符号位
        int i, j;
        if (str2[0] == '-')
        {
            result = AddInt(str1, str2 + 1, result);
        }
        else
        {
            int res = Compare(str1, str2);
            if (res == 0)
            {
                aries_strcpy(result, "0");
                return result;
            }
            if (res < 0)
            {
                sign = -1;
                char *temp = str1;
                str1 = str2;
                str2 = temp;
            }
            int len1 = aries_strlen(str1), len2 = aries_strlen(str2);
            int tmplen = len1 - len2;
            for (i = len2 - 1; i >= 0; i--)
            {
                if (str1[i + tmplen] < str2[i])          //借位
                {
                    j = 1;
                    while (1)
                    {
                        if (str1[tmplen - j + i] == '0')
                        {
                            str1[i + tmplen - j] = '9';
                            j++;
                        }
                        else
                        {
                            str1[i + tmplen - j] = char(int(str1[i + tmplen - j]) - 1);
                            break;
                        }
                    }
                    result[i + tmplen] = char(str1[i + tmplen] - str2[i] + ':');
                }
                else
                {
                    result[i + tmplen] = char(str1[i + tmplen] - str2[i] + '0');
                }
            }
            for (i = tmplen - 1; i >= 0; i--)
            result[i] = str1[i];
        }
        //去出结果中多余的前导0
        Erase(result, 0, FindFirstNotOf(result, '0'));
        if (aries_strlen(result) == 0)
        {
            aries_strcpy(result, "0");
        }
        if ((sign == -1) && (result[0] != '0'))
        {
            InsertCh(result, 0, '-');
        }
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::AddInt(char *str1, char *str2, char *result)
    {
        int sign = 1;          //sign为符号为
        char str[128] =
        {   0};
        if (str1[0] == '-')
        {
            if (str2[0] == '-')       //负负
            {
                sign = -1;
                AddInt(str1 + 1, str2 + 1, str);       //去掉正负号
            }
            else             //负正
            {
                SubInt(str2, str1 + 1, str);
            }
        }
        else
        {
            if (str2[0] == '-')        //正负
            {
                SubInt(str1, str2 + 1, str);
            }
            else                    //正正，把两个整数对齐，短整数前面加0补齐
            {
                int L1 = aries_strlen(str1), L2 = aries_strlen(str2);
                int i, l;
                char tmp[128];
                if (L1 < L2)
                {
                    l = L2 - L1;
                    for (i = 0; i < l; i++)
                    {
                        tmp[i] = '0';
                    }
                    tmp[l] = 0;
                    InsertStr(str1, 0, tmp);
                }
                else
                {
                    l = L1 - L2;
                    for (i = 0; i < L1 - L2; i++)
                    {
                        tmp[i] = '0';
                    }
                    tmp[l] = 0;
                    InsertStr(str2, 0, tmp);
                }
                int int1 = 0, int2 = 0; //int2记录进位
                l = aries_strlen(str1);
                for (i = l - 1; i >= 0; i--)
                {
                    int1 = (int(str1[i]) - '0' + int(str2[i]) - '0' + int2) % 10;
                    int2 = (int(str1[i]) - '0' + int(str2[i]) - '0' + int2) / 10;
                    str[i + 1] = char(int1 + '0');
                }
                str[l + 1] = 0;
                if (int2 != 0)
                {
                    result[0] = char(int2 + '0');
                }
                else
                {
                    aries_strcpy(str, str + 1);
                }
            }
        }
        //运算符处理符号
        if ((sign == -1) && (str[0] != '0'))
        {
            InsertCh(str, 0, '-');
        }
        aries_strcpy(result, str);
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::InsertStr(char *str, int pos, char *in)
    {
        int len = aries_strlen(str);
        int inLen = aries_strlen(in);
        assert(len + inLen < 128);
        int insertPos = len < pos ? len : pos;
        if (len == insertPos)
        {
            aries_strcat(str, in);
        }
        else
        {
            char tmp[128];
            aries_strcpy(tmp, str + insertPos);
            aries_strcpy(str + insertPos, in);
            aries_strcpy(str + insertPos + inLen, tmp);
        }
        return str;
    }

    ARIES_HOST_DEVICE_NO_INLINE char* Decimal::InsertCh(char *str, int pos, char in)
    {
        char temp[8];
        temp[0] = in;
        temp[1] = 0;
        return InsertStr(str, pos, temp);
    }
#endif

END_ARIES_ACC_NAMESPACE