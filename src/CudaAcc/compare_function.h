#ifndef COMPARE_FUNCTION_H_
#define COMPARE_FUNCTION_H_
#pragma once
#include "cuda.h"
#include "AriesEngineAlgorithm.h"

using namespace aries_engine;
BEGIN_ARIES_ACC_NAMESPACE

#define BLOCK_DIM 256
#define COMPACT_DECIMAL_PRECISION_OFFSET 24
#define COMPACT_DECIMAL_SCALE_OFFSET 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
# define CUDA_LAST_ERROR    { \
    cudaDeviceSynchronize(); \
    gpuErrchk( cudaGetLastError() );\
    }


void InitCompareFunctionMatrix();

typedef aries_acc::AriesBool (*CompareFunctionPointer)(const int8_t* left, const int l_size, const int8_t* right, const int r_size);


// 1. left, right
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool equal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *)left == *(const right_t *)right;
}
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool notEqual(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *)left != *(const right_t *)right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool greater(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *)left > *(const right_t *)right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool less(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *)left < *(const right_t *)right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *)left >= *(const right_t *)right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool lessOrEqual(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left <= *(const right_t *) right;
}
// 2. leftHasNull, right
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool equal_leftHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t >*) left == *(const right_t *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool notEqual_leftHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *)left != *(const right_t *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool greater_leftHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *)left > *(const right_t *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool less_leftHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *)left < *(const right_t *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual_leftHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *)left >= *(const right_t *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool lessOrEqual_leftHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *)left <= *(const right_t *) right;
}
// 3. left, rightHasNull
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool equal_left_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left == *(const nullable_type< right_t > *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool notEqual_left_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left != *(const nullable_type< right_t > *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool greater_left_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left > *(const nullable_type< right_t > *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool less_left_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left < *(const nullable_type< right_t > *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual_left_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left >= *(const nullable_type< right_t > *) right;
}
template<typename left_t, typename right_t>
__device__ aries_acc::AriesBool lessOrEqual_left_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left <= *(const nullable_type< right_t > *) right;
}
// 4. leftHasNull, rightHasNull
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool equal_leftHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left == *(const nullable_type< right_t > *) right;
}
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool notEqual_leftHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left != *(const nullable_type< right_t > *) right;
}
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool greater_leftHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left > *(const nullable_type< right_t > *) right;
}
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool less_leftHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left < *(const nullable_type< right_t > *) right;
}
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool greaterOrEqual_leftHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left >= *(const nullable_type< right_t > *) right;
}
template< typename left_t, typename right_t >
__device__ aries_acc::AriesBool lessOrEqual_leftHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left <= *(const nullable_type< right_t > *) right;
}

// compactDecimal
// 1. left compactDecimal, right
template<typename right_t>
__device__ aries_acc::AriesBool equal_compactDecimal_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) == *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool notEqual_compactDecimal_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) != *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greater_compactDecimal_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) > *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool less_compactDecimal_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) < *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimal_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) >= *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool lessOrEqual_compactDecimal_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) <= *(const right_t *) right;
}

// 2. left, right compactDecimal
template<typename left_t>
__device__ aries_acc::AriesBool equal_left_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left == Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000ul ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000ul ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool notEqual_left_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left != Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000ul ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000ul ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greater_left_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left > Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000ul ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000ul ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool less_left_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left < Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000ul ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000ul ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greaterOrEqual_left_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left >= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000ul ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000ul ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool lessOrEqual_left_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left <= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000ul ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000ul ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}

// 3. left compactDecimal HasNull, right
template<typename right_t>
__device__ aries_acc::AriesBool equal_compactDecimalHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) == *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool notEqual_compactDecimalHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) != *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greater_compactDecimalHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) > *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool less_compactDecimalHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) < *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimalHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) >= *(const right_t *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool lessOrEqual_compactDecimalHasNull_right(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) <= *(const right_t *) right;
}

// 4. leftHasNull, right compactDecimal
template<typename left_t>
__device__ aries_acc::AriesBool equal_leftHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left == Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool notEqual_leftHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left != Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greater_leftHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left > Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool less_leftHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left < Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greaterOrEqual_leftHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left >= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool lessOrEqual_leftHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left <= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}

// 5. left compactDecimal, rightHasNull
template<typename right_t>
__device__ aries_acc::AriesBool equal_compactDecimal_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) == *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool notEqual_compactDecimal_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) != *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greater_compactDecimal_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) > *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool less_compactDecimal_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) < *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimal_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) >= *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool lessOrEqual_compactDecimal_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) <= *(const nullable_type< right_t > *) right;
}

// 6. left, right compactDecimal HasNull
template<typename left_t>
__device__ aries_acc::AriesBool equal_left_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left == nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool notEqual_left_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left != nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greater_left_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left > nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool less_left_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left < nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greaterOrEqual_left_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left >= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool lessOrEqual_left_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const left_t *) left <= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}

// 7. left compactDecimal HasNull, rightHasNull
template<typename right_t>
__device__ aries_acc::AriesBool equal_compactDecimalHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) == *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool notEqual_compactDecimalHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) != *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greater_compactDecimalHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) > *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool less_compactDecimalHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) < *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimalHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) >= *(const nullable_type< right_t > *) right;
}
template<typename right_t>
__device__ aries_acc::AriesBool lessOrEqual_compactDecimalHasNull_rightHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) <= *(const nullable_type< right_t > *) right;
}

// 8. leftHasNull, right compactDecimal HasNull
template<typename left_t>
__device__ aries_acc::AriesBool equal_leftHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left == nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool notEqual_leftHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left != nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greater_leftHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left > nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool less_leftHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left < nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool greaterOrEqual_leftHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left >= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
template<typename left_t>
__device__ aries_acc::AriesBool lessOrEqual_leftHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *(const nullable_type< left_t > *) left <= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}

// compact deccimal, compact decimal
__device__ aries_acc::AriesBool equal_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool notEqual_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greater_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool less_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool lessOrEqual_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);

__device__ aries_acc::AriesBool equal_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool notEqual_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greater_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool less_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool lessOrEqual_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);

__device__ aries_acc::AriesBool equal_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool notEqual_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greater_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool less_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool lessOrEqual_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size);

__device__ aries_acc::AriesBool equal_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool notEqual_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greater_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool less_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool lessOrEqual_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size);

// char
__device__ aries_acc::AriesBool equal_char_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool notEqual_char_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greater_char_char( const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool less_char_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greaterOrEqual_char_char( const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool lessOrEqual_char_char( const int8_t* left, int l_size, const int8_t* right, int r_size);
__device__ aries_acc::AriesBool like_char_char( const int8_t *left, int l_size, const int8_t *right, int r_size );

__device__ aries_acc::AriesBool equal_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool notEqual_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greater_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool less_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greaterOrEqual_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool lessOrEqual_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool like_charHasNull_char( const int8_t *left, int l_size, const int8_t *right, int r_size );

__device__ aries_acc::AriesBool equal_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool notEqual_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greater_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool less_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greaterOrEqual_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool lessOrEqual_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool like_char_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );

__device__ aries_acc::AriesBool equal_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool notEqual_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greater_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool less_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool greaterOrEqual_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool lessOrEqual_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );
__device__ aries_acc::AriesBool like_charHasNull_charHasNull( const int8_t *left, int l_size, const int8_t *right, int r_size );



CompareFunctionPointer GetCompareFunction( AriesColumnType leftType,
                                           AriesColumnType rightType,
                                           AriesComparisonOpType opType);


template < typename output_t>
__global__ void compare_two_column_data(const int8_t* left,
                                        int left_type_size,
                                        const int8_t* right,
                                        int right_type_size,
                                        output_t * output_data,
                                        CompareFunctionPointer cmp,
                                        int count )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count){
        output_data[tid] = (*cmp)(left + tid*(left_type_size & 0x0000fffful),
                                  left_type_size,
                                  right + tid*(right_type_size & 0x0000fffful),
                                  right_type_size);
    } 
}

template< typename output_t >
__global__ void filter_column_data(const int8_t* left,
                                  int left_type_size,
                                  const int8_t* right,
                                  int right_type_size,
                                  output_t* output_data,
                                  CompareFunctionPointer cmp,
                                  size_t tupleNum )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < tupleNum)
    {
        output_data[tid] = (*cmp)(left + tid*(left_type_size & 0x0000fffful),
                                    left_type_size,
                                    right,
                                    right_type_size);

    }
}

template< typename output_t >
__global__ void filter_column_data_in( const int8_t* left,
                                        int left_type_size,
                                        const int8_t* right,
                                        int right_type_size,
                                        output_t* output_data,
                                        CompareFunctionPointer cmp,
                                        size_t tuple_num,
                                        size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        output_data[tid] = false;
        for ( int i = 0; i < right_count; ++i )
        {
            aries_acc::AriesBool ret = (*cmp)( left + tid*(left_type_size & 0x0000fffful),
                                                left_type_size,
                                                right + i*(right_type_size & 0x0000fffful),
                                                right_type_size);
            if ( ret )
            {
                output_data[tid] = true;
                break;
            }
        }
    }
}

template< typename output_t >
__global__ void filter_column_data_in_sorted( const int8_t* left,
                                        int left_type_size,
                                        const int8_t* right,
                                        int right_type_size,
                                        output_t* output_data,
                                        CompareFunctionPointer cmp_gt,
                                        CompareFunctionPointer cmp_eq,
                                        size_t tuple_num,
                                        size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        int begin = 0;
        int end = right_count;
        while( begin < end )
        {
            int mid = (begin + end) / 2;
            aries_acc::AriesBool left_gt_right = (*cmp_gt)(left + tid*(left_type_size & 0x0000fffful),
                                                            left_type_size,
                                                            right + mid*(right_type_size & 0x0000fffful),
                                                            right_type_size);
            if(left_gt_right)
                begin = mid + 1;
            else
                end = mid;
        }
        aries_acc::AriesBool ret = ( begin < right_count && 
                                     (*cmp_eq)( left + tid*(left_type_size & 0x0000fffful),
                                                left_type_size,
                                                right + begin*(right_type_size & 0x0000fffful),
                                                right_type_size ) );
        output_data[tid] = ret;
    }
}

template< typename output_t >
__global__ void filter_column_data_not_in( const int8_t* left,
                                            int left_type_size,
                                            const int8_t* right,
                                            int right_type_size,
                                            output_t* output_data,
                                            CompareFunctionPointer cmp,
                                            size_t tuple_num,
                                            size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        output_data[tid] = true;
        for ( int i = 0; i < right_count; ++i)
        {
            aries_acc::AriesBool ret = (*cmp)( left + tid*(left_type_size & 0x0000fffful),
                                                left_type_size,
                                                right + i*(right_type_size & 0x0000fffful),
                                                right_type_size);
            if ( ret )
            {
                output_data[tid] = false;
                break;
            }
        }
    }
}

template< typename output_t >
__global__ void filter_column_data_not_in_sorted( const int8_t* left,
                                        int left_type_size,
                                        const int8_t* right,
                                        int right_type_size,
                                        output_t* output_data,
                                        CompareFunctionPointer cmp_gt,
                                        CompareFunctionPointer cmp_eq,
                                        size_t tuple_num,
                                        size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        int begin = 0;
        int end = right_count;
        while( begin < end )
        {
            int mid = (begin + end) / 2;
            aries_acc::AriesBool left_gt_right = (*cmp_gt)(left + tid*(left_type_size & 0x0000fffful),
                                                            left_type_size,
                                                            right + mid*(right_type_size & 0x0000fffful),
                                                            right_type_size);
            if(left_gt_right)
                begin = mid + 1;
            else
                end = mid;
        }
        aries_acc::AriesBool ret = !( begin < right_count && 
                                      (*cmp_eq)( left + tid*(left_type_size & 0x0000fffful),
                                                left_type_size,
                                                right + begin*(right_type_size & 0x0000fffful),
                                                right_type_size ) );
        output_data[tid] = ret;
    }
}

template< typename output_t >
__global__ void filter_column_data_iter(const aries_acc::AriesColumnDataIterator* left,
                                        int left_type_size,
                                        const int8_t* right,
                                        int right_type_size,
                                        output_t* output_data,
                                        CompareFunctionPointer cmp,
                                        size_t tupleNum)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < tupleNum){
        output_data[tid] = (*cmp)(  left[0][tid],
                                    left_type_size,
                                    right,
                                    right_type_size);
    }
}

template< typename output_t >
__global__ void filter_column_data_iter_in(const aries_acc::AriesColumnDataIterator* left,
                                            int left_type_size,
                                            const int8_t* right,
                                            int right_type_size,
                                            output_t* output_data,
                                            CompareFunctionPointer cmp,
                                            size_t tuple_num,
                                            size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        output_data[tid] = false;
        for ( int i = 0; i < right_count; ++i )
        {
            aries_acc::AriesBool ret = (*cmp)( left[0][tid],
                                                left_type_size,
                                                right + i*(right_type_size & 0x0000fffful),
                                                right_type_size);
            if ( ret )
            {
                output_data[tid] = true;
                break;
            }
        }
    }
}

template< typename output_t >
__global__ void filter_column_data_iter_in_sorted(const aries_acc::AriesColumnDataIterator* left,
                                            int left_type_size,
                                            const int8_t* right,
                                            int right_type_size,
                                            output_t* output_data,
                                            CompareFunctionPointer cmp_gt,
                                            CompareFunctionPointer cmp_eq,
                                            size_t tuple_num,
                                            size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        int begin = 0;
        int end = right_count;
        while( begin < end )
        {
            int mid = ( begin + end ) / 2;
            aries_acc::AriesBool left_gt_right = (*cmp_gt)( left[0][tid],
                                                            left_type_size,
                                                            right + mid*(right_type_size & 0x0000fffful),
                                                            right_type_size );
            if( left_gt_right )
                begin = mid + 1;
            else
                end = mid;
        }
        output_data[tid] = ( begin < right_count && 
                            (*cmp_eq)( left[0][tid],
                                        left_type_size,
                                        right + begin*(right_type_size & 0x0000fffful),
                                        right_type_size ) );
    }
}

template< typename output_t >
__global__ void filter_column_data_iter_not_in( const aries_acc::AriesColumnDataIterator* left,
                                                int left_type_size,
                                                const int8_t* right,
                                                int right_type_size,
                                                output_t* output_data,
                                                CompareFunctionPointer cmp,
                                                size_t tuple_num,
                                                size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        output_data[tid] = true;
        for ( int i = 0; i < right_count; ++i)
        {
            aries_acc::AriesBool ret = (*cmp)( left[0][tid],
                                                left_type_size,
                                                right + i*(right_type_size & 0x0000fffful),
                                                right_type_size);
            if ( ret )
            {
                output_data[tid] = false;
                break;
            }
        }
    }
}

template< typename output_t >
__global__ void filter_column_data_iter_not_in_sorted(const aries_acc::AriesColumnDataIterator* left,
                                            int left_type_size,
                                            const int8_t* right,
                                            int right_type_size,
                                            output_t* output_data,
                                            CompareFunctionPointer cmp_gt,
                                            CompareFunctionPointer cmp_eq,
                                            size_t tuple_num,
                                            size_t right_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < tuple_num )
    {
        int begin = 0;
        int end = right_count;
        while( begin < end )
        {
            int mid = ( begin + end ) / 2;
            aries_acc::AriesBool left_gt_right = (*cmp_gt)( left[0][tid],
                                                            left_type_size,
                                                            right + mid*(right_type_size & 0x0000fffful),
                                                            right_type_size );
            if( left_gt_right )
                begin = mid + 1;
            else
                end = mid;
        }
        output_data[tid] = !( begin < right_count && 
                             (*cmp_eq)( left[0][tid],
                                        left_type_size,
                                        right + begin*(right_type_size & 0x0000fffful),
                                        right_type_size ) );
    }
}

END_ARIES_ACC_NAMESPACE
#endif // COMPARE_FUNCTION_H_