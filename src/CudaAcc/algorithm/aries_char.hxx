#ifndef ARIESCHAR_HXX_
#define ARIESCHAR_HXX_
#include "cpptraits.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    using PSTR = char*;
    using PCSTR = const char*;

    template< int LEN >
    struct ARIES_PACKED aries_char
    {
        char value[LEN];

    public:
        ARIES_HOST_DEVICE aries_char()
        {
            memset( value, 0, LEN );
        }

        template< int SRC_LEN, typename enable_if< SRC_LEN <= LEN, int >::type* = nullptr >
        ARIES_HOST_DEVICE aries_char( const aries_char< SRC_LEN >& src )
        {
            if( SRC_LEN < LEN )
                memset( value, 0, LEN );
            if( SRC_LEN <= LEN )
                memcpy( value, src.value, SRC_LEN );
            else
                memcpy( value, src.value, LEN ); //cut off
        }

        ARIES_HOST_DEVICE aries_char( char data )
        {
            int i = 0;
            value[i++] = data;

            while( i < LEN )
                value[i++] = 0;
        }
        ARIES_HOST_DEVICE aries_char( const char* data )
        {
            int i = 0;

            while( *data && i < LEN )
                value[i++] = *data++;

            while( i < LEN )
                value[i++] = 0;
        }

        ARIES_HOST_DEVICE operator PSTR()
        {
            return value;
        }

        ARIES_HOST_DEVICE operator PCSTR() const
        {
            return value;
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE aries_char& operator+=( const aries_char< SRC_LEN >& src )
        {
            int offset = 0;
            while( value[ offset ] && offset < LEN )
                ++offset;
            int leftSpace = LEN - offset;
            int copyLen = 0;
            while( src.value[ copyLen ] && copyLen < SRC_LEN && leftSpace-- )
                ++copyLen;
            memcpy( value + offset, src.value, copyLen );
            return *this;
        }

        ARIES_HOST_DEVICE aries_char& operator+=( const char& src )
        {
            int offset = 0;
            while( value[ offset ] && offset < LEN )
                ++offset;
            int leftSpace = LEN - offset;
            if( leftSpace > 0 )
                value[ offset ] = src;
            return *this;
        }

        template< int SRC_LEN, typename enable_if< SRC_LEN <= LEN, int >::type* = nullptr >
        ARIES_HOST_DEVICE aries_char& operator=( const aries_char< SRC_LEN >& src )
        {
            if( SRC_LEN < LEN )
                memset( value, 0, LEN );
            if( SRC_LEN <= LEN )
                memcpy( value, src.value, SRC_LEN );
            else
                memcpy( value, src.value, LEN ); //cut off
            return *this;
        }

        ARIES_HOST_DEVICE aries_char& operator=( const char* data )
        {
            int i = 0;

            while( *data && i < LEN )
                value[i++] = *data++;

            while( i < LEN )
                value[i++] = 0;

            return *this;
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE bool operator==( const aries_char< SRC_LEN >& src ) const
        {
            for( int i = 0; i < ( LEN < SRC_LEN ? LEN : SRC_LEN ); ++i )
            {
                if( value[i] != src.value[i] )
                    return false;
            }
            return LEN < SRC_LEN ? !src.value[LEN] : !value[SRC_LEN];
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE bool operator!=( const aries_char< SRC_LEN >& src ) const
        {
            return !( *this == src );
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE bool operator<( const aries_char< SRC_LEN >& src ) const
        {
            uint8_t val;
            uint8_t srcVal;
            for( int i = 0; i < ( LEN < SRC_LEN ? LEN : SRC_LEN ); ++i )
            {
                val = ( ( uint8_t* )value )[i];
                srcVal = ( ( uint8_t* )src.value )[i];
                if( val < srcVal )
                    return true;
                else if( val > srcVal )
                    return false;
            }
            return LEN < SRC_LEN ? src.value[LEN] : false;
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE bool operator>=( const aries_char< SRC_LEN >& src ) const
        {
            return !( *this < src );
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE bool operator>( const aries_char< SRC_LEN >& src ) const
        {
            uint8_t val;
            uint8_t srcVal;
            for( int i = 0; i < ( LEN < SRC_LEN ? LEN : SRC_LEN ); ++i )
            {
                val = ( ( uint8_t* )value )[i];
                srcVal = ( ( uint8_t* )src.value )[i];
                if( val > srcVal )
                    return true;
                else if( val < srcVal )
                    return false;
            }
            return LEN < SRC_LEN ? false : value[SRC_LEN];
        }

        template< int SRC_LEN >
        ARIES_HOST_DEVICE bool operator<=( const aries_char< SRC_LEN >& src ) const
        {
            return !( *this > src );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE int get_utf8_string_len( const char* buf, int len );
    //index is 1 based.
    ARIES_HOST_DEVICE_NO_INLINE const char* get_utf8_char_pos( const char* buf, int index );
    ARIES_HOST_DEVICE_NO_INLINE void copy_utf8_char( char* output, const char* input, int count );

#define calc_start_pos( start, strLen )                 \
    do                                                  \
    {                                                   \
        if ( start > 0 )                                \
        {                                               \
            start = start > strLen ? strLen : start;    \
        }                                               \
        else                                            \
        {                                               \
            /*find start pos*/                          \
            start = strLen + start + 1;                 \
            start = start < 1 ? 1 : start;              \
        }                                               \
    } while ( 0 );

    template< int OUTPUT_SIZE_IN_BYTES, int OUTPUT_CHAR_COUNT, bool has_null = false >
    struct op_substr_utf8_t
    {
        ARIES_HOST_DEVICE
        aries_char< OUTPUT_SIZE_IN_BYTES > operator()( const char* text, int start, int len ) const
        {
            aries_char< OUTPUT_SIZE_IN_BYTES > result;
            if( start != 0 )
            {
                int strLen = get_utf8_string_len( text, len );
                if( strLen > 0 )
                {
                    calc_start_pos( start, strLen );
                    int count = 0;
                    if( OUTPUT_CHAR_COUNT == -1 )
                        count = strLen - start + 1;
                    else
                        count = ( start + OUTPUT_CHAR_COUNT - 1 > strLen ? strLen - start + 1 : OUTPUT_CHAR_COUNT );

                    copy_utf8_char( result, get_utf8_char_pos( text, start ), count );
                }
            }
            return result;
        }
    };

    template< int OUTPUT_SIZE_IN_BYTES, int OUTPUT_CHAR_COUNT >
    struct op_substr_utf8_t< OUTPUT_SIZE_IN_BYTES, OUTPUT_CHAR_COUNT, true >
    {
        ARIES_HOST_DEVICE
        nullable_type< aries_char< OUTPUT_SIZE_IN_BYTES > > operator()( const char* text, int start, int len ) const
        {
            nullable_type< aries_char< OUTPUT_SIZE_IN_BYTES > > result;
            result.flag = *text;
            if( start != 0 && *text )
            {
                ++text;
                --len;
                int strLen = get_utf8_string_len( text, len );
                if( strLen > 0 )
                {
                    calc_start_pos( start, strLen );
                    int count = 0;
                    if( OUTPUT_CHAR_COUNT == -1 )
                        count = strLen - start + 1;
                    else
                        count = ( start + OUTPUT_CHAR_COUNT - 1 > strLen ? strLen - start + 1 : OUTPUT_CHAR_COUNT );

                    copy_utf8_char( result.value, get_utf8_char_pos( text, start ), count );
                }
            }
            return result;
        }
    };

    template< int OUTPUT_CHAR_COUNT >
    struct op_substr_utf8_t< 1, OUTPUT_CHAR_COUNT, false >
    {
        ARIES_HOST_DEVICE
        char operator()( const char* text, int start, int len ) const
        {
            char result = 0;
            if( start != 0 )
            {
                int strLen = get_utf8_string_len( text, len );
                if( strLen > 0 )
                {
                    calc_start_pos( start, strLen );
                    result = *get_utf8_char_pos( text, start );
                }
            }
            return result;
        }
    };

    template< int OUTPUT_CHAR_COUNT >
    struct op_substr_utf8_t< 1, OUTPUT_CHAR_COUNT, true >
    {
        ARIES_HOST_DEVICE
        nullable_type< char > operator()( const char* text, int start, int len ) const
        {
            nullable_type< char > result( *text, 0 );
            if( start != 0 && *text )
            {
                ++text;
                --len;
                int strLen = get_utf8_string_len( text, len );
                if( strLen > 0 )
                {
                    calc_start_pos( start, strLen );
                    result = *get_utf8_char_pos( text, start );
                }
            }
            return result;
        }
    };

    template< int LEN, bool has_null = false >
    struct op_substr_t
    {
        ARIES_HOST_DEVICE
        aries_char< LEN > operator()( const char* text, int start, int len ) const
        {
            aries_char< LEN > result;
            char* output = result;
            if( start > 0 )
            {
                memcpy( output, text + start - 1, LEN );
            }
            else
            {
                // find start pos
                int index = len - 1;
                while( index >= 0 && text[ index ] == 0 )
                    --index;
                int end = index;
                index += start + 1;
                index = index < 0 ? 0 : index;
                memcpy( output, text + index, end - index + 1 > LEN ? LEN : end - index + 1 );
            }
            return result;
        }
    };

    template< int LEN >
    struct op_substr_t< LEN, true >
    {
        ARIES_HOST_DEVICE
        nullable_type< aries_char< LEN > > operator()( const char* text, int start, int len ) const
        {
            nullable_type< aries_char< LEN > > result;
            char* output = result;
            *output = *text++;
            if( start > 0 )
            {
                memcpy( output + 1, text + start - 1, LEN );
            }
            else
            {
                // find start pos
                int index = len - 2;
                while( index >= 0 && text[ index ] == 0 )
                    --index;
                int end = index;
                index += start + 1;
                index = index < 0 ? 0 : index;
                memcpy( output + 1, text + index, end - index + 1 > LEN ? LEN : end - index + 1 );
            }
            return result;
        }
    };

    template< >
    struct op_substr_t< 1, false >
    {
        ARIES_HOST_DEVICE
        char operator()( const char* text, int start, int len ) const
        {
            char result;
            if( start > 0 )
            {
                result = text[ start - 1 ];
            }
            else
            {
                // find start pos
                int index = len - 1;
                while( index >= 0 && text[ index ] == 0 )
                    --index;
                index += start + 1;
                index = index < 0 ? 0 : index;
                result = text[ index ];
            }
            return result;
        }
    };

    template< >
    struct op_substr_t< 1, true >
    {
        ARIES_HOST_DEVICE
        nullable_type< char > operator()( const char* text, int start, int len ) const
        {
            nullable_type< char > result;
            char* output = result;
            *output = *text++;
            if( start > 0 )
            {
                output[ 1 ] = text[ start - 1 ];
            }
            else
            {
                // find start pos
                int index = len - 2;
                while( index >= 0 && text[ index ] == 0 )
                    --index;
                index += ( start + 1 );
                index = index < 0 ? 0 : index;
                output[ 1 ] = text[ index ];
            }
            return result;
        }
    };

    template< typename type_t >
    struct TypeSizeInString
    {
        static constexpr int LEN = 0;
    };
    template<>
    struct TypeSizeInString< int8_t >
    {
        static constexpr int LEN = 4;
    };
    template<>
    struct TypeSizeInString< int16_t >
    {
        static constexpr int LEN = 6;
    };
    template<>
    struct TypeSizeInString< int32_t >
    {
        static constexpr int LEN = 11;
    };
    template<>
    struct TypeSizeInString< int64_t >
    {
        static constexpr int LEN = 20;
    };

    struct op_tostr_t
    {
        template< typename type_t, int LEN = TypeSizeInString< type_t >::LEN >
        ARIES_HOST_DEVICE aries_char< LEN > operator()( type_t val, int base = 10 ) const
        {
            return to_string( val, base );
        }
        template< typename type_t, int LEN = TypeSizeInString< type_t >::LEN >
        ARIES_HOST_DEVICE nullable_type< aries_char< LEN > > operator()( nullable_type< type_t > val, int base = 10 ) const
        {
            nullable_type< aries_char< LEN > > result;
            result.flag = val.flag;
            if( result.flag )
                result.value = to_string( val.value, base );
            return result;
        }

    private:
        template< typename type_t, int LEN = TypeSizeInString< type_t >::LEN >
        ARIES_HOST_DEVICE aries_char< LEN > to_string( type_t value, int base ) const
        {
            aries_char< LEN > result;
            if( base < 2 || base > 36 )
            {
                *result = '\0';
                return result;
            }

            char* ptr = result, *ptr1 = result, tmp_char;
            int tmp_value;

            do
            {
                tmp_value = value;
                value /= base;
                *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz"[ 35 + ( tmp_value - value * base ) ];
            } while( value );

            // Apply negative sign
            if( tmp_value < 0 )
                *ptr++ = '-';
            *ptr-- = '\0';
            while( ptr1 < ptr )
            {
                tmp_char = *ptr;
                *ptr-- = *ptr1;
                *ptr1++ = tmp_char;
            }
            return result;
        }
    };

    template< typename output_t, typename type_t >
    ARIES_HOST_DEVICE type_t aries_concat( type_t val )
    {
        return val;
    }

    template< typename output_t >
    ARIES_HOST_DEVICE output_t aries_concat( const char* val )
    {
        return output_t( val );
    }

    template< typename output_t, typename type_t >
    ARIES_HOST_DEVICE type_t make_value( type_t val )
    {
        return val;
    }

    template< typename output_t >
    ARIES_HOST_DEVICE output_t make_value( const char* val )
    {
        return output_t( val );
    }

    template< typename output_t, typename type_t, typename ... args_t >
    ARIES_HOST_DEVICE output_t aries_concat( type_t val, args_t ... args )
    {
        output_t result;
        result += make_value< output_t >( val );
        result += aries_concat< output_t >( args... );
        return result;
    }

END_ARIES_ACC_NAMESPACE

#endif
