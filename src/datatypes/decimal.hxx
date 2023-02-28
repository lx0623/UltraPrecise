/*
 * decimal.hxx
 *
 *  Created on: 2019年6月21日
 *      Author: david
 */
#ifndef DECIMAL_HXX_
#define DECIMAL_HXX_
#include "AriesDefinition.h"

//#define COMPUTE_BY_STRING

//关于MODE定义是按照bit位定义的，如果需要联合，请使用或运算符 '|'
//放在mode的后2 bits
#define ARIES_MODE_EMPTY 0 //0b00

#define ARIES_MODE_STRICT_ALL_TABLES 1 //0b01
#define ARIES_MODE_ERROR_FOR_DIVISION_BY_ZERO 2 //0b10

#define MAX_INT32 4294967296
#define MAX_NUM32 (0xFFFFFFFF)

#define MAX_INT32_base10 1000000000

//错误code, 放在error的后 3 bits
#define ERR_OK 0
#define ERR_OVER_FLOW 1
#define ERR_DIV_BY_ZERO 2
#define ERR_STR_2_DEC 3
#define ERR_TRUNCATED 4

//#define COMPUTE_BY_STRING

#define TYPE_INT 0x1
#define TYPE_DEC 0x0


// precision/scale精度的定义
#define DEFAULT_PRECISION 10 //默认的precision精度
#define TINYINT_PRECISION 3
#define SMALLINT_PRECISION 5
#define MEDIUMINT_PRECISION 8
#define INT_PRECISION 10
#define BIGINT_PRECISION 19
#define DEFAULT_SCALE 0  //默认的scale精度

#define DIG_PER_INT32 9 //每个int能够表示的十进制的位数: 9

#define NUM_TOTAL_DIG 32 //decimal结构中的数值需要用5个int来表示,目前最大支持的精度是(36,30),由于小数和正数是分开存储的，所以需要的最大int的个数为5个
#define MODERN_GPU_PARA 64 // 当 LEN = 32 TPI = 4/8/16/32 参数为 64 （16 -> 128） (8 -> 256) (LEN<8 -> 256)
#define MODERN_GPU_SEG_PARA 32 // 当 LEN = 32 TPI = 4/8/16 参数为 64 (LEN<=16 -> 128) | 当 LEN = 32 TPI = 32 时 为 32
#define MODERN_GPU_NT_PARA 64 // 当 LEN = 32 TPI = 4 参数为 128 （LEN<=16 256）/ 当 LEN = 32 TPI = 8/16 参数为 64 / 当 LEN = 32 TPI = 32 参数为 32

#define INDEX_LAST_DIG (NUM_TOTAL_DIG - 1)
#define DIG_PER_INT64 18 //每个int64_t能表示的十进制位数为18位
#define BYTES_PER_INT32 4 //每个int需要4个bytes来存放其值
#define NUM_EX_DIG 2 //中间计算除法时，额外多保留了2位小数
#define DIV_FIX_INNER_FRAC 6 //内部计算除法时需要多加了6位小数
#define DIV_FIX_EX_FRAC 4 //计算除法时，正常结果只需要多保留4位小数

#define TPI 4

#define TPI_ONE 4
#define TPI_TWO 8
#define TPI_THR 16
#define TPI_FOR 32

#define LIMBS_ONE 1
#define LIMBS_TWO 2
#define LIMBS_THR 4
#define LIMBS_FOR 8

#define DLIMBS_ONE 1
#define DLIMBS_TWO 2    // 当 LEN=32 TPI=4 时

#define LIMB_OFFSET_ONE 0
#define LIMB_OFFSET_TWO 4
#define LIMB_OFFSET_THR 14
#define LIMB_OFFSET_FOR 31
#define LIMB_OFFSET_FIV 6
#define LIMB_OFFSET_SIX 15
#define LIMB_OFFSET_SEV 2
#define LIMB_OFFSET_EIG 7
#define LIMB_OFFSET_NIN 3

#define LIMBS int32_t((NUM_TOTAL_DIG+TPI-1)/TPI)
#define DLIMBS int32_t((LIMBS+TPI-1)/TPI)
#define LIMB_OFFSET (DLIMBS*TPI-LIMBS)
#define MAX_ROTATION 4

#define SET_SIGN_BIT(x, sign) (x = (x | sign << 7)) //对compact decimal，将符号位设置到某一个byte的第一个bit上
#define GET_SIGN_FROM_BIT(x) ((x >> 7) & 0x1) //对compact decimal，获取符号位

#define ALIGNED(a, b) ((a == b) || (NEEDELEMENTS(a) == NEEDELEMENTS(b))) //判断是否需要将小数对齐
#define SUPPORTED_MAX_PRECISION (NUM_TOTAL_DIG*DIG_PER_INT32)  //支持的最大precision
#define SUPPORTED_MAX_SCALE (NUM_TOTAL_DIG*DIG_PER_INT32)  //支持的最大scale
#define INNER_MAX_PRECISION (SUPPORTED_MAX_PRECISION + DIG_PER_INT32) //内部计算时，有时可以支持更大精度的数，这里表示可以支持的最大precision
#define INNER_MAX_PRECISION_INT32_NUM NUM_TOTAL_DIG  //内部计算时，有时可以支持更大精度的数，这里表示可以支持的最大scale

#define PER_DEC_MAX_SCALE 4294967296  //每个int的值不能大于此值
#define PER_INT_MAX_BIT 32  //每个uint32 的最大 bit 数量
#define MAX_BIT_PRECISION PER_INT_MAX_BIT*NUM_TOTAL_DIG;
#define MAX_BASE10_PER_INT 1000000000

BEGIN_ARIES_ACC_NAMESPACE
    // 表示符号位所在的位置: 整数部分，小数部分或额外加一个byte表示
    enum SignPos {
        INTG_PART,
        FRAC_PART,
        ADDITIONAL_PART
    };
    /**
     * Ubuntu is litter endian system
     * CompactDecimal format: data + sign
     * data format:
     * |----------intg part----------------|--------------frac part-------------|
     * intg byte1|intg byte2|...|intg byten|frac byte1|frac byte2|...|frac byten|
     * |-----------------------------------|------------------------------------|
     * sign format: 1 bit表示正负值 0: 表示正数；1: 表示负数
     * 按下面优先级进行计算:
     * 1, 如果intg部分有空闲的bit，若至少有一个Integer大小，将放到最后一个Integer的最后一个byte的第一个bit，否则将放在compact部分的最后一个byte的第一个bit (按此规则，符号位总在intg的最后一个byte的第一个bit)
     * 2, 如果frac部分有空闲的bit，若至少有一个Integer大小，将放到最后一个Integer的最后一个byte的第一个bit，否则将放在compact部分的最后一个byte的第一个bit
     * 3, buffer将多留最后一个byte用来存放sign信息
     * 注：compact部分为不满一个Integer大小的部分。
     * */
    struct ARIES_PACKED CompactDecimal {
        char data[1];
    };

    struct ARIES_PACKED Decimal
    {
		//最高位存放 sign , 末尾 6 位表示 prec 2^6 = 64 > 5*9;
		uint8_t sign;
        //最高位存放 over , 末尾 6 位表示 frac 2^6 = 64 > 5*9;
		uint8_t prec;

        //为了字节对齐
        uint8_t frac;
        uint8_t error;

		uint32_t v[NUM_TOTAL_DIG];	//#define NUM_TOTAL_DIG 5

    public:
//        ARIES_HOST_DEVICE_NO_INLINE Decimal(const Decimal& d);
        ARIES_HOST_DEVICE_NO_INLINE Decimal();
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale);
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale, uint32_t m);
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale, const char s[] );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale, uint32_t m, const char s[] );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(const CompactDecimal *compact, uint32_t precision, uint32_t scale, uint32_t m = ARIES_MODE_EMPTY);
        ARIES_HOST_DEVICE_NO_INLINE Decimal(const char s[] );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(const char s[], int len, bool a);
        ARIES_HOST_DEVICE_NO_INLINE Decimal( const char* s, int len );

        ARIES_HOST_DEVICE_NO_INLINE Decimal(int32_t t, int32_t scale, bool a, bool b, bool c);
        ARIES_HOST_DEVICE_NO_INLINE Decimal(Decimal t, int32_t scale, bool a, bool b, bool c, bool d);

        ARIES_HOST_DEVICE_NO_INLINE Decimal(int8_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(int16_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(int32_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(int64_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint8_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint16_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal(uint64_t t );

        ARIES_HOST_DEVICE_NO_INLINE bool ToCompactDecimal(char *buf, int len);
        ARIES_HOST_DEVICE_NO_INLINE char * GetPrecisionScale(char result[]);
        ARIES_HOST_DEVICE_NO_INLINE uint16_t GetError();
        ARIES_HOST_DEVICE_NO_INLINE char * GetDecimal(char result[]);
        ARIES_HOST_DEVICE_NO_INLINE void CheckOverFlow();
        ARIES_HOST_DEVICE_NO_INLINE double GetDouble() const;
        ARIES_HOST_DEVICE_NO_INLINE uint64_t ToUint64() const;
        /*
         * integer/frac part by pos index
         *   0: value of 0 int
         *   1: value of 1 int
         *   2: value of 2 int
         *   3: value of 3 int
         * */
        ARIES_HOST_DEVICE_NO_INLINE void setIntPart(int value, int pos);
        ARIES_HOST_DEVICE_NO_INLINE void setFracPart(int value, int pos);
        ARIES_HOST_DEVICE_NO_INLINE int getIntPart(int pos) const;
        ARIES_HOST_DEVICE_NO_INLINE int getFracPart(int pos) const;
        ARIES_HOST_DEVICE_NO_INLINE Decimal& cast( const Decimal& t );
        /* CalcTruncTargetPrecision
         * int p: > 0 try to truncate frac part to p scale
         *        = 0 try to truncate to integer
         *        < 0 try to truncate to integer, and intg part will be truncated
         * */
        ARIES_HOST_DEVICE_NO_INLINE Decimal& truncate( int p );
        ARIES_HOST_DEVICE_NO_INLINE explicit operator bool() const;
        ARIES_HOST_DEVICE_NO_INLINE Decimal operator-();
        //signed
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int8_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int16_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int32_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int64_t t );
        //unsigned
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint8_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint16_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint32_t t );
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint64_t t );
        //for decimal
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, const Decimal& right );

        // for int8_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int8_t right );
        // for uint8_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint8_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint8_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint8_t right );

        //for int16_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int16_t right );

        //for uint16_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint16_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint16_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint16_t right );

        //for int32_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int32_t right );

        //for uint32_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint32_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint32_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint32_t right );

        //for int64_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int64_t right );

        //for uint64_t
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint64_t left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint64_t right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint64_t right );

        //for float
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( float left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( float left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( float left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( float left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( float left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( float left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, float right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, float right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, float right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, float right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, float right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, float right );

        //for double
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( double left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( double left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( double left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( double left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( double left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( double left, const Decimal& right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, double right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, double right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, double right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, double right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, double right );
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, double right );

        // for add
        ARIES_DEVICE Decimal& operator+=(const Decimal& d);
        //signed
        ARIES_DEVICE Decimal& operator+=(int8_t i);
        ARIES_DEVICE Decimal& operator+=( int16_t i );
        ARIES_DEVICE Decimal& operator+=( int32_t i );
        ARIES_DEVICE Decimal& operator+=( int64_t i );
        //unsigned
        ARIES_DEVICE Decimal& operator+=(uint8_t i);
        ARIES_DEVICE Decimal& operator+=( uint16_t i );
        ARIES_DEVICE Decimal& operator+=( uint32_t i );
        ARIES_DEVICE Decimal& operator+=( uint64_t i );
        //double / float
        ARIES_DEVICE double operator+=(const float& f);
        ARIES_DEVICE double operator+=(const double& l);
        // self operator
        ARIES_DEVICE Decimal& operator++( );
        ARIES_DEVICE Decimal operator++( int32_t );
        //two operators
        friend ARIES_DEVICE Decimal operator+( const Decimal& left, const Decimal& right );
        //signed
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, int8_t right);
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, int16_t right);
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, int32_t right);
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, int64_t right);
        friend ARIES_DEVICE Decimal operator+( int8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator+( int16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator+( int32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator+( int64_t left, const Decimal& right );
        //unsigned
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint8_t right);
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint16_t right);
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint32_t right);
        friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint64_t right);
        friend ARIES_DEVICE Decimal operator+( uint8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator+( uint16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator+( uint32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator+( uint64_t left, const Decimal& right );
        //double / float
        friend ARIES_DEVICE double operator+( const Decimal& left, float right );
        friend ARIES_DEVICE double operator+( const Decimal& left, double right );
        friend ARIES_DEVICE double operator+( float left, const Decimal& right );
        friend ARIES_DEVICE double operator+( double left, const Decimal& right );

        // for sub
        ARIES_DEVICE Decimal& operator-=(const Decimal& d);
        //signed
        ARIES_DEVICE Decimal& operator-=( int8_t i );
        ARIES_DEVICE Decimal& operator-=( int16_t i );
        ARIES_DEVICE Decimal& operator-=( int32_t i );
        ARIES_DEVICE Decimal& operator-=( int64_t i );
        //unsigned
        ARIES_DEVICE Decimal& operator-=( uint8_t i );
        ARIES_DEVICE Decimal& operator-=( uint16_t i );
        ARIES_DEVICE Decimal& operator-=( uint32_t i );
        ARIES_DEVICE Decimal& operator-=( uint64_t i );
        //double / float
        ARIES_DEVICE double operator-=(const float& f);
        ARIES_DEVICE double operator-=(const double& l);
        //self operator
        ARIES_DEVICE Decimal& operator--( );
        ARIES_DEVICE Decimal operator--( int32_t );
        //two operators
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, const Decimal& right );
        //signed
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, int8_t right );
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, int16_t right );
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, int32_t right );
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, int64_t right );
        friend ARIES_DEVICE Decimal operator-( int8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator-( int16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator-( int32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator-( int64_t left, const Decimal& right );
        //unsigned
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, uint8_t right );
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, uint16_t right );
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, uint32_t right );
        friend ARIES_DEVICE Decimal operator-( const Decimal& left, uint64_t right );
        friend ARIES_DEVICE Decimal operator-( uint8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator-( uint16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator-( uint32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator-( uint64_t left, const Decimal& right );
        //double / float
        friend ARIES_DEVICE double operator-( const Decimal& left, const float right );
        friend ARIES_DEVICE double operator-( const Decimal& left, const double right );
        friend ARIES_DEVICE double operator-( const float left, const Decimal& right );
        friend ARIES_DEVICE double operator-( const double left, const Decimal& right );

        // for multiple
        ARIES_DEVICE Decimal& operator*=(const Decimal& d);
        //signed
        ARIES_DEVICE Decimal& operator*=( int8_t i );
        ARIES_DEVICE Decimal& operator*=( int16_t i );
        ARIES_DEVICE Decimal& operator*=( int32_t i );
        ARIES_DEVICE Decimal& operator*=( int64_t i );
        //unsigned
        ARIES_DEVICE Decimal& operator*=( uint8_t i );
        ARIES_DEVICE Decimal& operator*=( uint16_t i );
        ARIES_DEVICE Decimal& operator*=( uint32_t i );
        ARIES_DEVICE Decimal& operator*=( uint64_t i );
        //double / float
        ARIES_DEVICE double operator*=(const float& f);
        ARIES_DEVICE double operator*=(const double& d);
        //two operators
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, const Decimal& right);
        //signed
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int8_t right );
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int16_t right );
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int32_t right );
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int64_t right );
        friend ARIES_DEVICE Decimal operator*( int8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator*( int16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator*( int32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator*( int64_t left, const Decimal& right );
        //unsigned
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int8_t right );
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int16_t right );
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int32_t right );
        friend ARIES_DEVICE Decimal operator*( const Decimal& left, int64_t right );
        friend ARIES_DEVICE Decimal operator*( int8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator*( int16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator*( int32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator*( int64_t left, const Decimal& right );
        //double / float
        friend ARIES_DEVICE double operator*( const Decimal& left, const float right );
        friend ARIES_DEVICE double operator*( const Decimal& left, const double right );
        friend ARIES_DEVICE double operator*( const float left, const Decimal& right );
        friend ARIES_DEVICE double operator*( const double left, const Decimal& right );

        // for division
        ARIES_DEVICE Decimal& DivOrMod( const Decimal &d, bool isMod = false );
        ARIES_DEVICE Decimal& ModCalc( const Decimal &d);

        ARIES_DEVICE Decimal& operator/=(const Decimal& d);
        //signed
        ARIES_DEVICE Decimal& operator/=( int8_t i );
        ARIES_DEVICE Decimal& operator/=( int16_t i );
        ARIES_DEVICE Decimal& operator/=( int32_t i );
        ARIES_DEVICE Decimal& operator/=( int64_t i );
        //unsigned
        ARIES_DEVICE Decimal& operator/=( uint8_t i );
        ARIES_DEVICE Decimal& operator/=( uint16_t i );
        ARIES_DEVICE Decimal& operator/=( uint32_t i );
        ARIES_DEVICE Decimal& operator/=( uint64_t i );
        //double / float
        ARIES_DEVICE double operator/=(const float& f);
        ARIES_DEVICE double operator/=(const double& d);
        //two operators
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, const Decimal& right );
        //signed
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, int8_t right );
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, int16_t right );
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, int32_t right );
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, int64_t right );
        friend ARIES_DEVICE Decimal operator/( int8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator/( int16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator/( int32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator/( int64_t left, const Decimal& right );
        //unsigned
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, uint8_t right );
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, uint16_t right );
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, uint32_t right );
        friend ARIES_DEVICE Decimal operator/( const Decimal& left, uint64_t right );
        friend ARIES_DEVICE Decimal operator/( uint8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator/( uint16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator/( uint32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator/( uint64_t left, const Decimal& right );
        //float
        friend ARIES_DEVICE double operator/( const Decimal& left, const float right );
        friend ARIES_DEVICE double operator/( const Decimal& left, const double right );
        friend ARIES_DEVICE double operator/( const float left, const Decimal& right );
        friend ARIES_DEVICE double operator/( const double left, const Decimal& right );

        // for mod
        ARIES_DEVICE Decimal& operator%=(const Decimal& d);
        //signed
        ARIES_DEVICE Decimal& operator%=( int8_t i );
        ARIES_DEVICE Decimal& operator%=( int16_t i );
        ARIES_DEVICE Decimal& operator%=( int32_t i );
        ARIES_DEVICE Decimal& operator%=( int64_t i );
        //unsigned
        ARIES_DEVICE Decimal& operator%=( uint8_t i );
        ARIES_DEVICE Decimal& operator%=( uint16_t i );
        ARIES_DEVICE Decimal& operator%=( uint32_t i );
        ARIES_DEVICE Decimal& operator%=( uint64_t i );
        //double % float
        ARIES_DEVICE double operator%=(const float& f);
        ARIES_DEVICE double operator%=(const double& d);
        //two operators
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, const Decimal& right );
        //signed
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, int8_t right );
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, int16_t right );
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, int32_t right );
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, int64_t right );
        friend ARIES_DEVICE Decimal operator%( int8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator%( int16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator%( int32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator%( int64_t left, const Decimal& right );
        //unsigned
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, uint8_t right );
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, uint16_t right );
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, uint32_t right );
        friend ARIES_DEVICE Decimal operator%( const Decimal& left, uint64_t right );
        friend ARIES_DEVICE Decimal operator%( uint8_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator%( uint16_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator%( uint32_t left, const Decimal& right );
        friend ARIES_DEVICE Decimal operator%( uint64_t left, const Decimal& right );
        //float
        friend ARIES_DEVICE double operator%( const Decimal& left, const float right );
        friend ARIES_DEVICE double operator%( const Decimal& left, const double right );
        friend ARIES_DEVICE double operator%( const float left, const Decimal& right );
        friend ARIES_DEVICE double operator%( const double left, const Decimal& right );

        ARIES_HOST_DEVICE_NO_INLINE void AddDecimalOnCpu( const Decimal &d );
        ARIES_HOST_DEVICE_NO_INLINE void MulDecimalOnCpu( const Decimal &d );
    public:
        // 仅计算结果精度,可针对各精度数据四则混合运算后的结果精度进行计算
        ARIES_HOST_DEVICE_NO_INLINE void CalcAddTargetPrecision( const Decimal& d );
        ARIES_HOST_DEVICE_NO_INLINE void CalcSubTargetPrecision( const Decimal& d );
        ARIES_HOST_DEVICE_NO_INLINE void CalcMulTargetPrecision( const Decimal& d );
        ARIES_HOST_DEVICE_NO_INLINE void CalcDivTargetPrecision( const Decimal &d );
        ARIES_HOST_DEVICE_NO_INLINE void CalcModTargetPrecision( const Decimal &d );
        /* CalcTruncTargetPrecision
         * int p: > 0 try to truncate frac part to p scale
         *        = 0 try to truncate to integer
         *        < 0 try to truncate to integer, and intg part will be truncated
         * */
        ARIES_HOST_DEVICE_NO_INLINE bool isZero() const;

    public:
        // 检查和设置实际精度
        ARIES_HOST_DEVICE_NO_INLINE void CheckAndSetRealPrecision();

    protected:
        // decimal按10进制进行右移n位，相当除以 10的n次方
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator>>(int n);
        // decimal按10进制进行左移n位，相当乘以 10的n次方
        ARIES_HOST_DEVICE_NO_INLINE Decimal& operator<<(int n);
        // 将decimal变成一个整数decimal
        ARIES_HOST_DEVICE_NO_INLINE Decimal& GenIntDecimal(int shift = 0);
        // 将两个整数decimal折半取值
        ARIES_DEVICE Decimal HalfIntDecimal(const Decimal d1, const Decimal d2);
        ARIES_DEVICE Decimal DivInt(const Decimal ds, const Decimal dt, Decimal& residuel, uint32_t dtHitBit);

        ARIES_HOST_DEVICE_NO_INLINE int32_t GetPowers10(int i) const;

        // 根据精度获取其最大值
        ARIES_HOST_DEVICE_NO_INLINE void GenMaxDecByPrecision();
        // 对齐加减法的两个操作数的小数位
        ARIES_HOST_DEVICE_NO_INLINE void AlignAddSubData(Decimal &d);
        ARIES_HOST_DEVICE_NO_INLINE void initialize(uint32_t pr, uint32_t fc, uint32_t m);
        ARIES_HOST_DEVICE_NO_INLINE void CopyValue(Decimal &d);
        ARIES_HOST_DEVICE_NO_INLINE bool StringToDecimal(char * str);
        ARIES_HOST_DEVICE_NO_INLINE bool StringToDecimal(char * str, int len);
        // 以int进行除法处理
        ARIES_DEVICE Decimal& DivByInt(const Decimal &d, int shift, bool isMod = false);
        ARIES_DEVICE Decimal& ModByInt(const Decimal &d);
        // 以long进行除法处理
        ARIES_DEVICE Decimal& DivByInt64(const Decimal &d, int shift, bool isMod = false);
        ARIES_DEVICE Decimal& ModByInt64(const Decimal &d);

        //below methods are for computing long 10 based integer by char string
#ifdef COMPUTE_BY_STRING
        ARIES_HOST_DEVICE_NO_INLINE char* GetDivDecimalStr( char *to);
        ARIES_HOST_DEVICE_NO_INLINE int Compare( char *cmp1, char *cmp2);
        ARIES_HOST_DEVICE_NO_INLINE int FindFirstNotOf( char *s, char ch);
        ARIES_HOST_DEVICE_NO_INLINE char* Erase( char *s, int startPos, int n);
        ARIES_HOST_DEVICE_NO_INLINE char* DivInt(char *str1, char *str2, int mode, char * result);
        ARIES_HOST_DEVICE_NO_INLINE char* MulInt(char *str1, char *str2, char * result);
        ARIES_HOST_DEVICE_NO_INLINE char* SubInt(char *str1, char *str2, char *result);
        ARIES_HOST_DEVICE_NO_INLINE char* AddInt(char *str1, char *str2, char *result);
        ARIES_HOST_DEVICE_NO_INLINE char* InsertStr(char *str, int pos, char *in);
        ARIES_HOST_DEVICE_NO_INLINE char* InsertCh(char *str, int pos, char in);
#endif
    };

    ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal);
    // 根据精度信息获得compact需要存储的 bytes
    ARIES_HOST_DEVICE_NO_INLINE int GetDecimalRealBytes(uint16_t precision, uint16_t scale);

    // XMP 加法
    ARIES_DEVICE int32_t cgbn_add(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_sig(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE void  cgbn_mul_mlt(uint32_t r[], const uint32_t a[], const uint32_t b[], const uint32_t add[]);

    ARIES_DEVICE int32_t  cgbn_compare(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_div(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    // LEN_4_TPI_4
    ARIES_DEVICE int32_t cgbn_add_LEN_4_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_sig_LEN_4_TPI_4(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_4_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_4_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_4_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_4_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_4_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_8_TPI_4
    ARIES_DEVICE int32_t cgbn_add_LEN_8_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_mul_mlt_LEN_8_TPI_4(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_8_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_8_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_8_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE void cgbn_mod_LEN_8_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE  void operator_mod_LEN_8_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_16_TPI_4
    ARIES_DEVICE int32_t cgbn_add_LEN_16_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_mlt_LEN_16_TPI_4(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_16_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_16_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_16_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_16_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_16_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_32_TPI_4
    ARIES_DEVICE int32_t cgbn_add_LEN_32_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_mlt_LEN_32_TPI_4(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_32_TPI_4(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_32_TPI_4(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_32_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_32_TPI_4(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_4(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_8_TPI_8
    ARIES_DEVICE int32_t cgbn_add_LEN_8_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_sig_LEN_8_TPI_8(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_8_TPI_8(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_8_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_8_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_8_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_8_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_16_TPI_8
    ARIES_DEVICE int32_t cgbn_add_LEN_16_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_mlt_LEN_16_TPI_8(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_16_TPI_8(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_16_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_16_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_16_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_16_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_32_TPI_8
    ARIES_DEVICE int32_t cgbn_add_LEN_32_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_mlt_LEN_32_TPI_8(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_32_TPI_8(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_32_TPI_8(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_32_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_32_TPI_8(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_8(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_16_TPI_16
    ARIES_DEVICE int32_t cgbn_add_LEN_16_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_sig_LEN_16_TPI_16(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_16_TPI_16(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_16_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_16_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_16_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_16_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_32_TPI_16
    ARIES_DEVICE int32_t cgbn_add_LEN_32_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_mlt_LEN_32_TPI_16(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_32_TPI_16(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_32_TPI_16(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_32_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_32_TPI_16(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_16(uint32_t r[], uint32_t a[], uint32_t b[]);

    // LEN_32_TPI_32
    ARIES_DEVICE int32_t cgbn_add_LEN_32_TPI_32(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void  cgbn_mul_sig_LEN_32_TPI_32(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t add);

    ARIES_DEVICE int32_t  cgbn_compare_LEN_32_TPI_32(const uint32_t sync, const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE int32_t cgbn_sub_LEN_32_TPI_32(uint32_t r[], const uint32_t a[], const uint32_t b[]);

    ARIES_DEVICE void cgbn_div_LEN_32_TPI_32(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  uint8_t operator_add_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  uint8_t operator_sub_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[], int shift, uint8_t a_sign, uint8_t b_sign);

    ARIES_DEVICE  void operator_mul_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[]);

    ARIES_DEVICE  void operator_div_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[], uint32_t divitend_shift);

    ARIES_DEVICE void cgbn_mod_LEN_32_TPI_32(uint32_t q[], const uint32_t lo[], const uint32_t hi[], const uint32_t denom[], const uint32_t numthreads);

    ARIES_DEVICE  void operator_mod_LEN_32_TPI_32(uint32_t r[], uint32_t a[], uint32_t b[]);

END_ARIES_ACC_NAMESPACE

#endif



// /*
//  * decimal.hxx
//  *
//  *  Created on: 2019年6月21日
//  *      Author: david
//  */
// #ifndef DECIMAL_HXX_
// #define DECIMAL_HXX_
// #include "AriesDefinition.h"

// //#define COMPUTE_BY_STRING

// //关于MODE定义是按照bit位定义的，如果需要联合，请使用或运算符 '|'
// //放在mode的后2 bits
// #define ARIES_MODE_EMPTY 0 //0b00

// #define ARIES_MODE_STRICT_ALL_TABLES 1 //0b01
// #define ARIES_MODE_ERROR_FOR_DIVISION_BY_ZERO 2 //0b10

// //错误code, 放在error的后 3 bits
// #define ERR_OK 0
// #define ERR_OVER_FLOW 1
// #define ERR_DIV_BY_ZERO 2
// #define ERR_STR_2_DEC 3
// #define ERR_TRUNCATED 4

// //mode最后 2 bits由mode使用,前面 6 bits由intg使用 2^6 = 64 > SUPPORTED_MAX_PRECISION
// #define SET_MODE(m, v) (m = ((m & ~0x03) | (v & 0x03)))
// #define GET_MODE(m) (m & 0x03)
// #define SET_CALC_INTG(m, v) (m = ((m & 0x03) | (v << 2)))
// #define GET_CALC_INTG(m) ((m & ~0x03) >> 2)
// //error最后 3 bits由error使用,前面 5 bits由frac使用 2^5 = 32 > SUPPORTED_MAX_SCALE
// #define SET_ERR(e, v) (e = ((e & ~0x07) | (v & 0x07)))
// #define GET_ERR(e) (e & 0x07)
// #define SET_CALC_FRAC(e, v) (e = ((e & 0x07) | (v << 3)))
// #define GET_CALC_FRAC(e) ((e & ~0x07) >> 3)

// #define TYPE_INT 0x1
// #define TYPE_DEC 0x0

// // precision/scale精度的定义
// #define DEFAULT_PRECISION 10 //默认的precision精度
// #define TINYINT_PRECISION 3
// #define SMALLINT_PRECISION 5
// #define MEDIUMINT_PRECISION 8
// #define INT_PRECISION 10
// #define BIGINT_PRECISION 19
// #define DEFAULT_SCALE 0  //默认的scale精度

// #define DIG_PER_INT32 9 //每个int能够表示的十进制的位数: 9
// #define NUM_TOTAL_DIG 5 //decimal结构中的数值需要用5个int来表示,目前最大支持的精度是(36,30),由于小数和正数是分开存储的，所以需要的最大int的个数为5个
// #define INDEX_LAST_DIG (NUM_TOTAL_DIG - 1)
// #define DIG_PER_INT64 18 //每个int64_t能表示的十进制位数为18位
// #define BYTES_PER_INT32 4 //每个int需要4个bytes来存放其值
// #define NUM_EX_DIG 2 //中间计算除法时，额外多保留了2位小数
// #define DIV_FIX_INNER_FRAC 6 //内部计算除法时需要多加了6位小数
// #define DIV_FIX_EX_FRAC 4 //计算除法时，正常结果只需要多保留4位小数

// #define NEEDELEMENTS(x) ((x + (DIG_PER_INT32 - 1)) / DIG_PER_INT32) //计算整数部分intg和小数部分frac分别需要多少个int来存储
// #define NEEDBYTES(x) (((int) x + 7) >> 3) //对compact decimal，计算intg/scale需要多少个bytes来存储
// #define SET_SIGN_BIT(x, sign) (x = (x | sign << 7)) //对compact decimal，将符号位设置到某一个byte的第一个bit上
// #define GET_SIGN_FROM_BIT(x) ((x >> 7) & 0x1) //对compact decimal，获取符号位
// #define GET_WHOLE_INTS(x) (x >> 2) //对compact decimal，根据bytes的个数可以计算出需要int的个数
// #define GET_COMPACT_BYTES(x) (x & 0x3) //对compact decimal，获取对应值在compact decimal里的byte偏移
// //有空闲bit的总bits条件: <=7bits 或 >=32 bits
// #define HAS_FREE_BIT(x) ((x & 0x7) || (x >> 5))

// #define ALIGNED(a, b) ((a == b) || (NEEDELEMENTS(a) == NEEDELEMENTS(b))) //判断是否需要将小数对齐
// #define SUPPORTED_MAX_PRECISION 36  //支持的最大precision
// #define SUPPORTED_MAX_SCALE 30  //支持的最大scale
// #define INNER_MAX_PRECISION (SUPPORTED_MAX_PRECISION + DIG_PER_INT32) //内部计算时，有时可以支持更大精度的数，这里表示可以支持的最大precision
// #define INNER_MAX_PRECISION_INT32_NUM NUM_TOTAL_DIG  //内部计算时，有时可以支持更大精度的数，这里表示可以支持的最大scale

// #define PER_DEC_MAX_SCALE 1000000000  //每个int的值不能大于此值

// BEGIN_ARIES_ACC_NAMESPACE
//     // 表示符号位所在的位置: 整数部分，小数部分或额外加一个byte表示
//     enum SignPos {
//         INTG_PART,
//         FRAC_PART,
//         ADDITIONAL_PART
//     };
//     /**
//      * Ubuntu is litter endian system
//      * CompactDecimal format: data + sign
//      * data format:
//      * |----------intg part----------------|--------------frac part-------------|
//      * intg byte1|intg byte2|...|intg byten|frac byte1|frac byte2|...|frac byten|
//      * |-----------------------------------|------------------------------------|
//      * sign format: 1 bit表示正负值 0: 表示正数；1: 表示负数
//      * 按下面优先级进行计算:
//      * 1, 如果intg部分有空闲的bit，若至少有一个Integer大小，将放到最后一个Integer的最后一个byte的第一个bit，否则将放在compact部分的最后一个byte的第一个bit (按此规则，符号位总在intg的最后一个byte的第一个bit)
//      * 2, 如果frac部分有空闲的bit，若至少有一个Integer大小，将放到最后一个Integer的最后一个byte的第一个bit，否则将放在compact部分的最后一个byte的第一个bit
//      * 3, buffer将多留最后一个byte用来存放sign信息
//      * 注：compact部分为不满一个Integer大小的部分。
//      * */
//     struct ARIES_PACKED CompactDecimal {
//         char data[1];
//     };

//     struct ARIES_PACKED Decimal
//     {
//         uint8_t intg;//:7; //表示中间计算时的精度，当mode/error都未设值时用此值
//         uint8_t frac;//:5; //表示中间计算时的精度，当mode/error都未设值时用此值
//         //最后 2 bits由mode使用,前面 6 bits由intg使用 2^6 = 64 > SUPPORTED_MAX_PRECISION,表示最后结果的精度
//         uint8_t mode;//:2;
//         //最后 3 bits由error使用,前面 5 bits由frac使用 2^5 = 32 > SUPPORTED_MAX_SCALE,表示最后结果的精度
//         uint8_t error;//:2;
//         int32_t values[NUM_TOTAL_DIG];

//     public:
// //        ARIES_HOST_DEVICE_NO_INLINE Decimal(const Decimal& d);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal();
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale, uint32_t m);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale, const char s[] );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale, uint32_t m, const char s[] );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(const CompactDecimal *compact, uint32_t precision, uint32_t scale, uint32_t m = ARIES_MODE_EMPTY);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(const char s[] );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal( const char* s, int len );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(int8_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(int16_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(int32_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(int64_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint8_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint16_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal(uint64_t v );
//         ARIES_HOST_DEVICE_NO_INLINE bool ToCompactDecimal(char *buf, int len);
//         ARIES_HOST_DEVICE_NO_INLINE char * GetInnerPrecisionScale(char result[]);
//         ARIES_HOST_DEVICE_NO_INLINE char * GetPrecisionScale(char result[]);
//         ARIES_HOST_DEVICE_NO_INLINE char * GetTargetPrecisionScale(char result[]);
//         ARIES_HOST_DEVICE_NO_INLINE uint16_t GetSqlMode();
//         ARIES_HOST_DEVICE_NO_INLINE uint16_t GetError();
//         ARIES_HOST_DEVICE_NO_INLINE char * GetInnerDecimal(char result[]);
//         ARIES_HOST_DEVICE_NO_INLINE char * GetDecimal(char result[]);
//         ARIES_HOST_DEVICE_NO_INLINE void CheckOverFlow();
//         ARIES_HOST_DEVICE_NO_INLINE double GetDouble() const;
//         ARIES_HOST_DEVICE_NO_INLINE int64_t ToInt64() const;
//         /*
//          * integer/frac part by pos index
//          *   0: value of 0 int
//          *   1: value of 1 int
//          *   2: value of 2 int
//          *   3: value of 3 int
//          * */
//         ARIES_HOST_DEVICE_NO_INLINE void setIntPart(int value, int pos);
//         ARIES_HOST_DEVICE_NO_INLINE void setFracPart(int value, int pos);
//         ARIES_HOST_DEVICE_NO_INLINE int getIntPart(int pos) const;
//         ARIES_HOST_DEVICE_NO_INLINE int getFracPart(int pos) const;
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& cast( const Decimal& v );
//         /* CalcTruncTargetPrecision
//          * int p: > 0 try to truncate frac part to p scale
//          *        = 0 try to truncate to integer
//          *        < 0 try to truncate to integer, and intg part will be truncated
//          * */
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& truncate( int p );
//         ARIES_HOST_DEVICE_NO_INLINE explicit operator bool() const;
//         ARIES_HOST_DEVICE_NO_INLINE Decimal operator-();
//         //signed
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int8_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int16_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int32_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( int64_t v );
//         //unsigned
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint8_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint16_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint32_t v );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=( uint64_t v );
//         //for decimal
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, const Decimal& right );

//         // for int8_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int8_t right );
//         // for uint8_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint8_t right );

//         //for int16_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int16_t right );

//         //for uint16_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint16_t right );

//         //for int32_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int32_t right );

//         //for uint32_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint32_t right );

//         //for int64_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( int64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( int64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( int64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( int64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( int64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( int64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, int64_t right );

//         //for uint64_t
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( uint64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( uint64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( uint64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( uint64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( uint64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( uint64_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, uint64_t right );

//         //for float
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, float right );

//         //for double
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( double left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( double left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( double left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( double left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( double left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( double left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>( const Decimal& left, double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=( const Decimal& left, double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<( const Decimal& left, double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=( const Decimal& left, double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator==( const Decimal& left, double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=( const Decimal& left, double right );

//         // for add
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=(const Decimal& d);
//         //signed
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=(int8_t i);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=( int16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=( int32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=( int64_t i );
//         //unsigned
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=(uint8_t i);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=( uint16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=( uint32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator+=( uint64_t i );
//         //double / float
//         ARIES_HOST_DEVICE_NO_INLINE double operator+=(const float& f);
//         ARIES_HOST_DEVICE_NO_INLINE double operator+=(const double& l);
//         // self operator
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator++( );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal operator++( int32_t );
//         //two operators
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( const Decimal& left, const Decimal& right );
//         //signed
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, int8_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, int16_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, int32_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, int64_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( int64_t left, const Decimal& right );
//         //unsigned
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, uint8_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, uint16_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, uint32_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal& left, uint64_t right);
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator+( uint64_t left, const Decimal& right );
//         //double / float
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator+( const Decimal& left, float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator+( const Decimal& left, double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator+( float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator+( double left, const Decimal& right );

//         // for sub
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=(const Decimal& d);
//         //signed
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( int8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( int16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( int32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( int64_t i );
//         //unsigned
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( uint8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( uint16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( uint32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator-=( uint64_t i );
//         //double / float
//         ARIES_HOST_DEVICE_NO_INLINE double operator-=(const float& f);
//         ARIES_HOST_DEVICE_NO_INLINE double operator-=(const double& l);
//         //self operator
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator--( );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal operator--( int32_t );
//         //two operators
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, const Decimal& right );
//         //signed
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( int64_t left, const Decimal& right );
//         //unsigned
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator-( uint64_t left, const Decimal& right );
//         //double / float
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator-( const Decimal& left, const float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator-( const Decimal& left, const double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator-( const float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator-( const double left, const Decimal& right );

//         // for multiple
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=(const Decimal& d);
//         //signed
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( int8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( int16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( int32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( int64_t i );
//         //unsigned
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( uint8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( uint16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( uint32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator*=( uint64_t i );
//         //double / float
//         ARIES_HOST_DEVICE_NO_INLINE double operator*=(const float& f);
//         ARIES_HOST_DEVICE_NO_INLINE double operator*=(const double& d);
//         //two operators
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, const Decimal& right);
//         //signed
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int64_t left, const Decimal& right );
//         //unsigned
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator*( int64_t left, const Decimal& right );
//         //double / float
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator*( const Decimal& left, const float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator*( const Decimal& left, const double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator*( const float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator*( const double left, const Decimal& right );

//         // for division
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& DivOrMod( const Decimal &d, bool isMod = false );

//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=(const Decimal& d);
//         //signed
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( int8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( int16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( int32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( int64_t i );
//         //unsigned
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( uint8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( uint16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( uint32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator/=( uint64_t i );
//         //double / float
//         ARIES_HOST_DEVICE_NO_INLINE double operator/=(const float& f);
//         ARIES_HOST_DEVICE_NO_INLINE double operator/=(const double& d);
//         //two operators
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, const Decimal& right );
//         //signed
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( int64_t left, const Decimal& right );
//         //unsigned
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator/( uint64_t left, const Decimal& right );
//         //float
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator/( const Decimal& left, const float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator/( const Decimal& left, const double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator/( const float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator/( const double left, const Decimal& right );

//         // for mod
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=(const Decimal& d);
//         //signed
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( int8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( int16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( int32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( int64_t i );
//         //unsigned
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( uint8_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( uint16_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( uint32_t i );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator%=( uint64_t i );
//         //double % float
//         ARIES_HOST_DEVICE_NO_INLINE double operator%=(const float& f);
//         ARIES_HOST_DEVICE_NO_INLINE double operator%=(const double& d);
//         //two operators
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, const Decimal& right );
//         //signed
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, int8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, int16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, int32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, int64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( int8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( int16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( int32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( int64_t left, const Decimal& right );
//         //unsigned
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, uint8_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, uint16_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, uint32_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( const Decimal& left, uint64_t right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( uint8_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( uint16_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( uint32_t left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE Decimal operator%( uint64_t left, const Decimal& right );
//         //float
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator%( const Decimal& left, const float right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator%( const Decimal& left, const double right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator%( const float left, const Decimal& right );
//         friend ARIES_HOST_DEVICE_NO_INLINE double operator%( const double left, const Decimal& right );

//     public:
//         // 仅计算结果精度,可针对各精度数据四则混合运算后的结果精度进行计算
//         ARIES_HOST_DEVICE_NO_INLINE void CalcAddTargetPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcSubTargetPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcMulTargetPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcDivTargetPrecision( const Decimal &d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcModTargetPrecision( const Decimal &d );
//         /* CalcTruncTargetPrecision
//          * int p: > 0 try to truncate frac part to p scale
//          *        = 0 try to truncate to integer
//          *        < 0 try to truncate to integer, and intg part will be truncated
//          * */
//         ARIES_HOST_DEVICE_NO_INLINE void CalcTruncTargetPrecision( int p );

//     public:
//         // 检查和设置实际精度
//         ARIES_HOST_DEVICE_NO_INLINE void CheckAndSetRealPrecision();

//     protected:
//         // 最终结果精度计算，其值写到mode和error里
//         ARIES_HOST_DEVICE_NO_INLINE void CalcAddPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcSubPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcMulPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcDivPrecision( const Decimal &d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcModPrecision( const Decimal &d );
//         // 当结果精度超出最大支持的精度时，截取合适的精度
//         ARIES_HOST_DEVICE_NO_INLINE void CalcTruncatePrecision( int p );
//         // 内部运算精度计算，其值写到intg和frac里
//         ARIES_HOST_DEVICE_NO_INLINE void CalcInnerAddPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcInnerSubPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcInnerMulPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcInnerDivPrecision( const Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE void CalcInnerModPrecision( const Decimal &d );
//         // 当内部运算精度超出最大支持的精度时，截取合适的精度
//         ARIES_HOST_DEVICE_NO_INLINE void CalcInnerTruncatePrecision( int p );
//         // decimal按10进制进行右移n位，相当除以 10的n次方
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator>>(int n);
//         // decimal按10进制进行左移n位，相当乘以 10的n次方
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& operator<<(int n);
//         // 更新整数部分(intg)精度
//         ARIES_HOST_DEVICE_NO_INLINE void UpdateIntgDigits();
//         // 获取数据的实际精度
//         ARIES_HOST_DEVICE_NO_INLINE int GetRealPrecision(int &highestValue) const;
//         // 检查和设置结果精度
//         ARIES_HOST_DEVICE_NO_INLINE void CheckAndSetCalcPrecision();
//         ARIES_HOST_DEVICE_NO_INLINE int GetRealIntgSize(int &highestValue) const;
//         // 将decimal变成一个整数decimal
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& GenIntDecimal(int shift = 0);
//         // 将两个整数decimal折半取值
//         ARIES_HOST_DEVICE_NO_INLINE Decimal HalfIntDecimal(const Decimal d1, const Decimal d2);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal DivInt(const Decimal ds, const Decimal dt, Decimal& residuel);
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& Negate();
//         // 将一个整数decimal转成带有小数的decimal
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& IntToFrac(int fracDigits);
//         ARIES_HOST_DEVICE_NO_INLINE bool isFracZero() const;
//         ARIES_HOST_DEVICE_NO_INLINE bool isZero() const;
//         ARIES_HOST_DEVICE_NO_INLINE bool isLessZero() const;
//         ARIES_HOST_DEVICE_NO_INLINE bool isLessEqualZero() const;
//         ARIES_HOST_DEVICE_NO_INLINE bool isGreaterZero() const;
//         ARIES_HOST_DEVICE_NO_INLINE bool isGreaterEqualZero() const;
//         ARIES_HOST_DEVICE_NO_INLINE int32_t GetPowers10(int i) const;
//         ARIES_HOST_DEVICE_NO_INLINE int32_t GetFracMaxTable(int i) const;
//         // 根据精度获取其最大值
//         ARIES_HOST_DEVICE_NO_INLINE void GenMaxDecByPrecision();
//         // 根据精度获取其最小值
//         ARIES_HOST_DEVICE_NO_INLINE void GenMinDecByPrecision();
//         ARIES_HOST_DEVICE_NO_INLINE void TransferData( const Decimal* v );
//         // 对齐加减法的两个操作数的小数位
//         ARIES_HOST_DEVICE_NO_INLINE void AlignAddSubData(Decimal &d);
//         ARIES_HOST_DEVICE_NO_INLINE void initialize(uint32_t ig, uint32_t fc, uint32_t m);
//         ARIES_HOST_DEVICE_NO_INLINE void CopyValue(Decimal &d);
//         ARIES_HOST_DEVICE_NO_INLINE bool StringToDecimal(char * str);
//         ARIES_HOST_DEVICE_NO_INLINE bool StringToDecimal(char * str, int len);
//         // 以int进行除法处理
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& DivByInt(const Decimal &d, int shift, bool isMod = false);
//         // 以long进行除法处理
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& DivByInt64(const Decimal &d, int shift, bool isMod = false);
//         ARIES_HOST_DEVICE_NO_INLINE bool CheckIfValidStr2Dec(char * str);
//         ARIES_HOST_DEVICE_NO_INLINE bool CheckIfValidStr2Dec(char * str, int len);

//         ARIES_HOST_DEVICE_NO_INLINE Decimal& AddBothPositiveNums( Decimal& d );
//         ARIES_HOST_DEVICE_NO_INLINE int32_t CompareInt( int32_t *op1, int32_t *op2 );
//         ARIES_HOST_DEVICE_NO_INLINE Decimal& SubBothPositiveNums( Decimal& d );

//         //below methods are for computing long 10 based integer by char string
// #ifdef COMPUTE_BY_STRING
//         ARIES_HOST_DEVICE_NO_INLINE char* GetDivDecimalStr( char *to);
//         ARIES_HOST_DEVICE_NO_INLINE int Compare( char *cmp1, char *cmp2);
//         ARIES_HOST_DEVICE_NO_INLINE int FindFirstNotOf( char *s, char ch);
//         ARIES_HOST_DEVICE_NO_INLINE char* Erase( char *s, int startPos, int n);
//         ARIES_HOST_DEVICE_NO_INLINE char* DivInt(char *str1, char *str2, int mode, char * result);
//         ARIES_HOST_DEVICE_NO_INLINE char* MulInt(char *str1, char *str2, char * result);
//         ARIES_HOST_DEVICE_NO_INLINE char* SubInt(char *str1, char *str2, char *result);
//         ARIES_HOST_DEVICE_NO_INLINE char* AddInt(char *str1, char *str2, char *result);
//         ARIES_HOST_DEVICE_NO_INLINE char* InsertStr(char *str, int pos, char *in);
//         ARIES_HOST_DEVICE_NO_INLINE char* InsertCh(char *str, int pos, char in);
// #endif
//     };

//     ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal);
//     // 根据精度信息获得compact需要存储的 bytes
//     ARIES_HOST_DEVICE_NO_INLINE int GetDecimalRealBytes(uint16_t precision, uint16_t scale);
//     // 根据精度信息获得compact需要存储的 bits
//     ARIES_HOST_DEVICE_NO_INLINE int GetDecimalNeedBits(int base10Precision);
//     // 根据精度信息获取decimal中values值有效的个数
//     ARIES_HOST_DEVICE_NO_INLINE int GetDecimalValidElementsCount( uint16_t precision, uint16_t scale );

// END_ARIES_ACC_NAMESPACE

// #endif
