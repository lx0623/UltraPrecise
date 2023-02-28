#include "functions.hxx"

BEGIN_ARIES_ACC_NAMESPACE

ARIES_HOST_DEVICE_NO_INLINE bool str_less_t( const char* a, const char* b, int len )
    {
        for( int i = 0; i < len; i++ )
        {
            if( *( ( const uint8_t* )a + i ) < *( ( const uint8_t* )b + i ) )
                return true;
            else if( *( ( const uint8_t* )a + i ) > *( ( const uint8_t* )b + i ) )
                return false;
        }
        return false;
    }
ARIES_HOST_DEVICE_NO_INLINE bool str_less_equal_t( const char* a, const char* b, int len )
    {
        for( int i = 0; i < len; i++ )
        {
            if( *( ( const uint8_t* )a + i ) < *( ( const uint8_t* )b + i ) )
                return true;
            else if( *( ( const uint8_t* )a + i ) > *( ( const uint8_t* )b + i ) )
                return false;
        }
        return true;
    }


ARIES_HOST_DEVICE_NO_INLINE bool str_greater_t( const char* a, const char* b, int len )
    {
        for( int i = 0; i < len; i++ )
        {
            if( *( ( const uint8_t* )a + i ) > *( ( const uint8_t* )b + i ) )
                return true;
            else if( *( ( const uint8_t* )a + i ) < *( ( const uint8_t* )b + i ) )
                return false;
        }
        return false;
    }


ARIES_HOST_DEVICE_NO_INLINE bool str_greater_equal_t( const char* a, const char* b, int len )
    {
        for( int i = 0; i < len; i++ )
        {
            if( *( ( const uint8_t* )a + i ) > *( ( const uint8_t* )b + i ) )
                return true;
            else if( *( ( const uint8_t* )a + i ) < *( ( const uint8_t* )b + i ) )
                return false;
        }
        return true;
    }

ARIES_HOST_DEVICE_NO_INLINE bool str_equal_to_t( const char* a, const char* b, int len )
    {
        for( int i = 0; i < len; i++ )
        {
            if( *( a + i ) != *( b + i ) )
                return false;
        }
        return true;
    }


ARIES_HOST_DEVICE_NO_INLINE bool str_not_equal_to_t( const char* a, const char* b, int len )
    {
        for( int i = 0; i < len; i++ )
        {
            if( *( a + i ) != *( b + i ) )
            {
                return true;
            }
        }

        return false;
    }


ARIES_HOST_DEVICE_NO_INLINE bool str_like( const char* text, const char* regexp, int len )
    {
        const char* star = nullptr;
        const char* ss = text;
        const char* end = text + len;
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

// start is 1 based
    ARIES_HOST_DEVICE_NO_INLINE void sub_string( const char* text, int start, int len, char* output )
    {
        memcpy( output, text + start - 1, len );
    }

//add for CompactDecimal
    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_less_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 )
    {
        return Decimal( s1, precision1, scale1 ) < Decimal( s2, precision2, scale2 );
    }


ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_less_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 )
    {
        return Decimal( s1, precision1, scale1 ) <= Decimal( s2, precision2, scale2 );
    }

ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_greater_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 )
    {
        return Decimal( s1, precision1, scale1 ) > Decimal( s2, precision2, scale2 );
    }

ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_greater_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 )
    {
        return Decimal( s1, precision1, scale1 ) >= Decimal( s2, precision2, scale2 );
    }


    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 )
    {
        return Decimal( s1, precision1, scale1 ) == Decimal( s2, precision2, scale2 );
    }


ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_not_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 )
    {
        return Decimal( s1, precision1, scale1 ) != Decimal( s2, precision2, scale2 );
    }



    // extern ARIES_HOST_DEVICE_NO_INLINE bool is_null< true >( const char* data )
    // {
    //     return !*data;
    // }

END_ARIES_ACC_NAMESPACE