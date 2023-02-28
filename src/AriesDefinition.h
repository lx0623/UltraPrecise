/*
 * AriesDefinition.h
 *
 *  Created on: Jul 9, 2019
 *      Author: lichi
 */

#ifndef ARIESDEFINITION_H_
#define ARIESDEFINITION_H_

#ifdef __CUDACC__

#ifndef ARIES_HOST_DEVICE
#define ARIES_HOST_DEVICE __forceinline__ __device__ __host__
#endif

#ifndef ARIES_HOST_DEVICE_NO_INLINE
#define ARIES_HOST_DEVICE_NO_INLINE  __device__ __host__
#endif

#ifndef ARIES_DEVICE
#define ARIES_DEVICE __device__
#endif

#ifndef ARIES_DEVICE_FORCE
#define ARIES_DEVICE_FORCE __forceinline__ __device__ 
#endif

#ifndef ARIES_LAMBDA
#define ARIES_LAMBDA __device__ __host__
#endif

#else // #ifndef __CUDACC__

#define ARIES_HOST_DEVICE
#define ARIES_HOST_DEVICE_NO_INLINE
#define ARIES_DEVICE
#define ARIES_DEVICE_FORCE
#define ARIES_LAMBDA

#endif // #ifdef __CUDACC__

#ifdef _MSC_VER
#define ARIES_ALIGN(x) __declspec(align(x))
#define ARIES_PACKED __declspec((packed))
#else
#define ARIES_ALIGN(x) __attribute__((aligned(x)))
#define ARIES_PACKED __attribute__((packed))
#endif

#define NAMESPACE_ARIES_START namespace aries {
#define NAMESPACE_ARIES_END }

#define BEGIN_ARIES_ACC_NAMESPACE namespace aries_acc {
#define END_ARIES_ACC_NAMESPACE }

#define BEGIN_ARIES_ENGINE_NAMESPACE namespace aries_engine {
#define END_ARIES_ENGINE_NAMESPACE }

#ifdef __CUDACC_RTC__
/* Signed.  */
typedef signed char int8_t;
typedef short int int16_t;
typedef int int32_t;
# if __WORDSIZE == 64
typedef long int int64_t;
# else
typedef long long int int64_t;
# endif

/* Unsigned.  */
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
#if __WORDSIZE == 64
typedef unsigned long int uint64_t;
#else
typedef unsigned long long int uint64_t;
#endif
#else
#include <cstdint>
#endif // #ifdef __CUDACC__

NAMESPACE_ARIES_START
    static const int ARIES_MAX_CHAR_WIDTH = 1024; /* Max length for a CHAR colum */

    enum class AriesExprType
        : int32_t
        {
            INTEGER,
        FLOATING,
        DECIMAL,
        STRING,
        DATE,
        DATE_TIME,
        TIME,
        TIMESTAMP,
        YEAR,
        COLUMN_ID,
        STAR,
        AGG_FUNCTION,
        SQL_FUNCTION,
        ARRAY,
        CALC,
        NOT,
        COMPARISON,
        LIKE,
        IN,
        NOT_IN,
        BETWEEN,
        EXISTS,
        CASE,
        AND_OR,
        BRACKETS,
        TRUE_FALSE,
        IF_CONDITION,
        IS_NOT_NULL,
        IS_NULL,
        DISTINCT,
        NULL_VALUE,
        INTERVAL,
        BUFFER,
        COALESCE
    };

    enum class AriesValueType
        : int32_t
        {
            UNKNOWN = 0, 
            INT8, 
            INT16, 
            INT32, 
            INT64,
            UINT8, 
            UINT16,
            UINT32, 
            UINT64, 
            DECIMAL, 
            FLOAT, 
            DOUBLE, // DO keep numeric types' order
            CHAR,
            BOOL,
            DATE,
            TIME,
            DATETIME,
            TIMESTAMP,
            YEAR,
            LIST,
            COMPACT_DECIMAL,
            ARIES_DECIMAL,

            COUNT  //只用于计数, 并不是一种ValueType
    };

    enum class AriesLogicOpType
        : int32_t
        {
            AND, OR
    };

    enum class AriesCalculatorOpType
        : int32_t
        {
            ADD, SUB, MUL, DIV, MOD
    };

    enum class AriesComparisonOpType
        : int32_t
        {
            EQ,     //equal =
            NE,     //not equal !=
            GT,     //greater than >
            LT,     //less than <
            GE,     //greater or equal >=
            LE,      //less or equal <=
            IN,      // in
            NOTIN,   // not in
            LIKE,     // like

            COUNT  //只用于计数, 并不是一种OPType
    };

    enum class AriesOrderByType
        : int32_t
        {
            NONE, ASC, DESC
    };

    enum class AriesJoinType
        : int32_t
        {
            INNER_JOIN, LEFT_JOIN, RIGHT_JOIN, FULL_JOIN, SEMI_JOIN, ANTI_JOIN
    };

    enum class AriesAggFunctionType
        : int32_t
        {
            NONE, COUNT, SUM, AVG, MAX, MIN, ANY_VALUE
    };

    enum class AriesSqlFunctionType
        : int32_t
        {
            SUBSTRING,
        EXTRACT,
        ST_VOLUMN,
        DATE,
        NOW,
        DATE_SUB,
        DATE_ADD,
        DATE_DIFF,
        DATE_FORMAT,
        TIME_DIFF,
        ABS,
        COUNT,
        SUM,
        AVG,
        MAX,
        MIN,
        UNIX_TIMESTAMP,
        CAST,
        CONVERT,
        MONTH,
        COALESCE,
        CONCAT,
        TRUNCATE,
        DICT_INDEX,
        ANY_VALUE
    };

    enum class AriesSetOpType
        : int32_t
        {
            UNION, UNION_ALL, INTERSECT, INTERSECT_ALL, EXCEPT, EXCEPT_ALL
    };

    enum class AriesBtreeSearchOpType : int32_t
    {
        EQ,  // equal =
        NE,  // not equal !=
        GT,  // greater than >
        LT,  // less than <
        GE,  // greater or equal >=
        LE,  // less or equal <=
        GTAndLT,  // greater than and less < x <
        GEAndLT,  // greater or equal and less <= x <
        GTAndLE,  // greater and less or equal < x <=
        GEAndLE,  // greater or equal and less or equaal <= x <=
        LIKE  // like
    };

    enum class EncodeType : int8_t
    {
        NONE, DICT
    };

NAMESPACE_ARIES_END

#endif /* ARIESDEFINITION_H_ */
