#include "compare_function.h"
BEGIN_ARIES_ACC_NAMESPACE

// 1. char, char
__device__ aries_acc::AriesBool equal_char_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    // l_size、r_size是值数据类型的长度，并不一定是真实数据长度，比如char(10)的列可能存的是'hello'
    int min_size = l_size < r_size ? l_size : r_size;
    for( int i = 0; i < min_size; ++i )
    {
        if( *( (const char*)left + i ) != *( (const char*)right + i ) )
            return false;
    }
    if( l_size>r_size && *((const char*)(left+min_size)) )
        return false;
    else if ( r_size>l_size && *((const char*)(right+min_size)) )
        return false;
    return true;
}
__device__ aries_acc::AriesBool notEqual_char_char(const int8_t *left, int l_size, const int8_t *right, int r_size)
{
    return !equal_char_char(left, l_size, right, r_size);
}
__device__ aries_acc::AriesBool greater_char_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    int min_size = l_size < r_size ? l_size : r_size;
    for ( int i = 0; i < min_size; ++i)
    {
        if ( *((const char*)left + i) > *((const char*)right + i) )
            return true;
        else if ( *((const char*)left + i) < *((const char*)right + i) )
            return false;
    }
    if( l_size>r_size && *((const char*)(left+min_size)) )
        return true;
    return false;
}
__device__ aries_acc::AriesBool less_char_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    int min_size = l_size < r_size ? l_size : r_size;
    for ( int i = 0; i < min_size; ++i)
    {
        if ( *((const char*)left + i) < *((const char*)right + i) )
            return true;
        else if ( *((const char*)left + i) > *((const char*)right + i) )
            return false;
    }
    if( l_size<r_size && *(const char*)(right+min_size) )
        return true;
    return false;
}
__device__ aries_acc::AriesBool greaterOrEqual_char_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    int min_size = l_size < r_size ? l_size : r_size;
    for ( int i=0; i<min_size; ++i)
    {
        if ( *((const char*)left + i) < *((const char*)right + i))
            return false;
    }
    if( l_size<r_size and *(const char*)(right+min_size) )
        return false;
    return true;
}
__device__ aries_acc::AriesBool lessOrEqual_char_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    int min_size = l_size < r_size ? l_size : r_size;
    for ( int i=0; i<min_size; ++i)
    {
        if ( *((const char*)left + i) > *((const char*)right + i))
            return false;
    }
    if( l_size>r_size && *(const char*)(left+min_size) )
        return false;
    return true;
}
__device__ aries_acc::AriesBool like_char_char( const int8_t* left , int l_size, const int8_t* right, int r_size )
    {
        const char* text = (const char*)left;
        const char* regexp = (const char*)right;

        const char* star = nullptr;
        const char* ss = text;
        const char* end = text + l_size;
        while( text < end && *text )
        {
            if( *regexp == '%' )
            {
                star = regexp++;
                ss = text;
            }
            else if( *regexp == '_' || *regexp == *text )
            {
                ++text;
                ++regexp;
            }
            else if( star )
            {
                regexp = star + 1;
                text = ++ss;
                continue;
            }
            else
                return false;
        }
        while( *regexp == '%' )
            ++regexp;
        return !*regexp;
    }

// 2. char HasNull, char
__device__ aries_acc::AriesBool equal_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    if( *left )
        return equal_char_char( ++left, --l_size, right, r_size);
    else
        return AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool notEqual_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *left ? notEqual_char_char( ++left, --l_size, right, r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool greater_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *left ? greater_char_char( ++left, --l_size, right, r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool less_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *left ? less_char_char( ++left, --l_size, right, r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool greaterOrEqual_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *left ? greaterOrEqual_char_char( ++left, --l_size, right, r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool lessOrEqual_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *left ? lessOrEqual_char_char( ++left, --l_size, right, r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool like_charHasNull_char(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *left ? like_char_char( ++left, --l_size, right, r_size ) : AriesBool::ValueType::Unknown;
}

// 3. char, char HasNull
__device__ aries_acc::AriesBool equal_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? equal_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool notEqual_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? notEqual_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool greater_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? greater_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool less_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? less_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool greaterOrEqual_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? greaterOrEqual_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool lessOrEqual_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? lessOrEqual_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool like_char_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return *right ? like_char_char( left, l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}

// 4. char HasNull, char HasNull
__device__ aries_acc::AriesBool equal_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? equal_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool notEqual_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? notEqual_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool greater_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? greater_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool less_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? less_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool greaterOrEqual_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? greaterOrEqual_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool lessOrEqual_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? lessOrEqual_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}
__device__ aries_acc::AriesBool like_charHasNull_charHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return ( *left && *right ) ? like_char_char( ++left, --l_size, ++right, --r_size ) : AriesBool::ValueType::Unknown;
}

__device__ aries_acc::AriesBool equal_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) == Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool notEqual_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) != Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool greater_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) > Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool less_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) < Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) >= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool lessOrEqual_compactDecimal_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) <= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}

__device__ aries_acc::AriesBool equal_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) == nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool notEqual_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) != nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool greater_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) > nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool less_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) < nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) >= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool lessOrEqual_compactDecimal_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return Decimal( (CompactDecimal *) left, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) <= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}

__device__ aries_acc::AriesBool equal_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) == Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool notEqual_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) != Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool greater_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) > Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool less_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) < Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) >= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}
__device__ aries_acc::AriesBool lessOrEqual_compactDecimalHasNull_compactDecimal(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) <= Decimal( (CompactDecimal *) right, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) );
}

__device__ aries_acc::AriesBool equal_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) == nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool notEqual_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) != nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool greater_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) > nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool less_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) < nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool greaterOrEqual_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) >= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}
__device__ aries_acc::AriesBool lessOrEqual_compactDecimalHasNull_compactDecimalHasNull(const int8_t* left, int l_size, const int8_t* right, int r_size)
{
    return nullable_type< Decimal > ( *left, Decimal( (CompactDecimal *) left + 1, ( ( l_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( l_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) ) <= nullable_type< Decimal > ( *right, Decimal( (CompactDecimal *) right + 1, ( ( r_size & 0xff000000 ) >> COMPACT_DECIMAL_PRECISION_OFFSET ), ( ( r_size & 0x00ff0000 ) >> COMPACT_DECIMAL_SCALE_OFFSET ) ) );
}

////////////////////
#include "device_statement.h"
////////////////////

const int VALUE_TYPE_COUNT = (int)AriesValueType::COUNT;
const int OP_TYPE_COUNT = (int)AriesComparisonOpType::COUNT;
struct host_matrix_struct
{
    // mactix[ leftType ][ l_hasNull ][ rightType ][ r_hasNull ][ opType ]
    CompareFunctionPointer host_matrix[VALUE_TYPE_COUNT][2][VALUE_TYPE_COUNT][2][OP_TYPE_COUNT] = {};
};

vector< host_matrix_struct > host_matrixes;

void InitCompareFunctionMatrix()
{
    int32_t oldDeviceId;
    cudaGetDevice( &oldDeviceId );
    int deviceCount;
    cudaGetDeviceCount( &deviceCount );
    host_matrixes.reserve( deviceCount );
    for ( int i = 0; i < deviceCount; ++i )
    {
        host_matrix_struct hm;
        auto host_matrix = hm.host_matrix;
        cudaSetDevice(i);
////////////////////
#include "host_matrix_statement.h"
////////////////////
        CUDA_LAST_ERROR
        
        host_matrixes.push_back(hm);
    }
    cudaSetDevice( oldDeviceId );
}


CompareFunctionPointer GetCompareFunction(AriesColumnType leftType,
                                            AriesColumnType rightType,
                                            AriesComparisonOpType opType)
{
    const int leftTypeIndex = (int)leftType.DataType.ValueType;
    assert(leftTypeIndex<VALUE_TYPE_COUNT);
    const int l_hasNull = (int)leftType.HasNull;

    const int rightTypeIndex = (int)rightType.DataType.ValueType;
    assert(rightTypeIndex<VALUE_TYPE_COUNT);
    const int r_hasNull = (int)rightType.HasNull;

    const int opTypeIndex = (int)opType;
    assert(opTypeIndex<OP_TYPE_COUNT);

    int deviceId;
    cudaError_t ret = cudaGetDevice( &deviceId );
    if( cudaSuccess != ret )
        throw cuda_exception_t( ret );
    // printf("filter_column_data current devive id: %d\n", deviceId);
    auto host_matrix = host_matrixes[deviceId].host_matrix;
    return host_matrix[leftTypeIndex][l_hasNull][rightTypeIndex][r_hasNull][opTypeIndex];
}


END_ARIES_ACC_NAMESPACE
