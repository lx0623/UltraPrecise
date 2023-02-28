/*
 * functions.hxx
 *
 *  Created on: Jul 17, 2019
 *      Author: lichi
 */

#ifndef FUNCTIONS_HXX_
#define FUNCTIONS_HXX_

#include "aries_types.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE bool str_less_t( const char* a, const char* b, int len );

    template< bool left_has_null, bool right_has_null >
    struct less_t_str
    {
    };

    template< >
    struct less_t_str< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, str_less_t( a, b, len - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_t_str< true, false >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, str_less_t( a, b, aLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_t_str< false, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, str_less_t( a, b, bLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_t_str< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_less_t( a, b, len );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool str_less_equal_t( const char* a, const char* b, int len );

    template< bool left_has_null, bool right_has_null >
    struct less_equal_t_str
    {
    };

    template< >
    struct less_equal_t_str< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, str_less_equal_t( a, b, len - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_equal_t_str< true, false >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, str_less_equal_t( a, b, aLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_equal_t_str< false, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, str_less_equal_t( a, b, bLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_equal_t_str< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_less_equal_t( a, b, len );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool str_greater_t( const char* a, const char* b, int len );

    template< bool left_has_null, bool right_has_null >
    struct greater_t_str
    {
    };

    template< >
    struct greater_t_str< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, str_greater_t( a, b, len - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_t_str< true, false >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, str_greater_t( a, b, aLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_t_str< false, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, str_greater_t( a, b, bLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_t_str< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_greater_t( a, b, len );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool str_greater_equal_t( const char* a, const char* b, int len );

    template< bool left_has_null, bool right_has_null >
    struct greater_equal_t_str
    {
    };

    template< >
    struct greater_equal_t_str< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, str_greater_equal_t( a, b, len - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_equal_t_str< true, false >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, str_greater_equal_t( a, b, aLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_equal_t_str< false, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, str_greater_equal_t( a, b, bLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_equal_t_str< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_greater_equal_t( a, b, len );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool str_equal_to_t( const char* a, const char* b, int len );

    template< bool left_has_null, bool right_has_null >
    struct equal_to_t_str
    {
    };

    template< >
    struct equal_to_t_str< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, str_equal_to_t( a, b, len - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct equal_to_t_str< true, false >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, str_equal_to_t( a, b, aLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct equal_to_t_str< false, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, str_equal_to_t( a, b, bLen - 1 ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct equal_to_t_str< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_equal_to_t( a, b, len );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool str_not_equal_to_t( const char* a, const char* b, int len );

    template< bool left_has_null, bool right_has_null >
    struct not_equal_to_t_str
    {
    };

    template< >
    struct not_equal_to_t_str< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, str_not_equal_to_t( a, b, len - 1 ) );
            else
                return AriesBool
                { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct not_equal_to_t_str< true, false >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, str_not_equal_to_t( a, b, aLen - 1 ) );
            else
                return AriesBool
                { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct not_equal_to_t_str< false, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, str_not_equal_to_t( a, b, bLen - 1 ) );
            else
                return AriesBool
                { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct not_equal_to_t_str< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_not_equal_to_t( a, b, len );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool str_like( const char* text, const char* regexp, int len );
    template< bool has_null = false >
    struct op_like_t
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* text, const char* regexp, int len ) const
        {
            return str_like( text, regexp, len );
        }
    };

    template< >
    struct op_like_t< true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* text, const char* regexp, int len ) const
        {
            return *text ? AriesBool( str_like( text + 1, regexp, len - 1 ) ) : AriesBool::ValueType::Unknown;
        }
    };

    // start is 1 based
    ARIES_HOST_DEVICE_NO_INLINE void sub_string( const char* text, int start, int len, char* output );

    template< bool has_null = false >
    struct op_substring_t
    {
        ARIES_HOST_DEVICE
        void operator()( const char* text, int start, int len, char* output ) const
        {
            sub_string( text, start, len, output );
        }
    };

    template< >
    struct op_substring_t< true >
    {
        ARIES_HOST_DEVICE
        void operator()( const char* text, int start, int len, char* output ) const
        {
            *output = *text;
            sub_string( text + 1, start, len, output + 1 );
        }
    };

    template< typename type_t >
    ARIES_HOST_DEVICE bool is_null( type_t data )
    {
        return false;
    }

    template< typename type_t, template< typename > class type_nullable >
    ARIES_HOST_DEVICE bool is_null( type_nullable< type_t > data )
    {
        return !data.flag;
    }

    template< bool has_null = false >
    ARIES_HOST_DEVICE bool is_null( const char* data )
    {
        return false;
    }

    template< >
    __inline__ __device__ __host__ bool is_null< true >( const char* data )
    {
        return !*data;
    }
    

    //add for CompactDecimal
    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_less_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 );

    template< bool left_has_null, bool right_has_null >
    struct less_t_CompactDecimal
    {
    };

    struct CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        CompactDecimal_cmp_base( uint16_t precision, uint16_t scale )
                : prec( precision ), sca( scale )
        {
        }
        uint16_t prec;
        uint16_t sca;
    };

    template< >
    struct less_t_CompactDecimal< true, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_t_CompactDecimal< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a )
                return ( AriesBool )( ++a, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_t_CompactDecimal< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *b )
                return ( AriesBool )( ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_t_CompactDecimal< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_less_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 );

    template< bool left_has_null, bool right_has_null >
    struct less_equal_t_CompactDecimal
    {
    };

    template< >
    struct less_equal_t_CompactDecimal< true, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, CompactDecimal_less_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_equal_t_CompactDecimal< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, CompactDecimal_less_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_equal_t_CompactDecimal< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, CompactDecimal_less_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct less_equal_t_CompactDecimal< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        less_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_less_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_greater_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 );
    template< bool left_has_null, bool right_has_null >
    struct greater_t_CompactDecimal
    {
    };

    template< >
    struct greater_t_CompactDecimal< true, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_t_CompactDecimal< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_t_CompactDecimal< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_t_CompactDecimal< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_greater_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 );

    template< bool left_has_null, bool right_has_null >
    struct greater_equal_t_CompactDecimal
    {
    };

    template< >
    struct greater_equal_t_CompactDecimal< true, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, CompactDecimal_greater_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_equal_t_CompactDecimal< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, CompactDecimal_greater_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_equal_t_CompactDecimal< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, CompactDecimal_greater_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct greater_equal_t_CompactDecimal< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        greater_equal_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_greater_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 );

    template< bool left_has_null, bool right_has_null >
    struct equal_to_t_CompactDecimal
    {
    };

    template< >
    struct equal_to_t_CompactDecimal< true, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, CompactDecimal_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct equal_to_t_CompactDecimal< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, CompactDecimal_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct equal_to_t_CompactDecimal< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, CompactDecimal_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool::ValueType::Unknown;
        }
    };

    template< >
    struct equal_to_t_CompactDecimal< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    ARIES_HOST_DEVICE_NO_INLINE bool CompactDecimal_not_equal_t( const CompactDecimal *s1, const uint16_t precision1, const uint16_t scale1,
            const CompactDecimal *s2, const uint16_t precision2, const uint16_t scale2 );

    template< bool left_has_null, bool right_has_null >
    struct not_equal_to_t_CompactDecimal
    {
    };

    template< >
    struct not_equal_to_t_CompactDecimal< true, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        not_equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( AriesBool )( ++a, ++b, CompactDecimal_not_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool
                { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct not_equal_to_t_CompactDecimal< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        not_equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }

        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( AriesBool )( ++a, CompactDecimal_not_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool
                { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct not_equal_to_t_CompactDecimal< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        not_equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( AriesBool )( ++b, CompactDecimal_not_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return AriesBool
                { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct not_equal_to_t_CompactDecimal< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE
        not_equal_to_t_CompactDecimal( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {
        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_not_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< bool left_has_null, bool right_has_null >
    struct less_t_str_null_smaller
    {
    };

    template< >
    struct less_t_str_null_smaller< true, true >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *b && ( !*a || ( ++a, ++b, str_less_t( a, b, len - 1 ) ) );
            else
                return false;
        }
    };

    template< >
    struct less_t_str_null_smaller< true, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return !*a || ( ++a, str_less_t( a, b, aLen - 1 ) );
        }
    };

    template< >
    struct less_t_str_null_smaller< false, true >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return *b && ( ++b, str_less_t( a, b, bLen - 1 ) );
        }
    };

    template< >
    struct less_t_str_null_smaller< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_less_t( a, b, len );
        }
    };

    template< typename type_t, typename type_u >
    ARIES_HOST_DEVICE bool binary_find( const type_u* keys, int count, type_t key )
    {
        int begin = 0;
        int end = count;
        while( begin < end )
        {
            int mid = ( begin + end ) / 2;
            if( keys[mid] < key )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin < count && key == keys[begin];
    }

    template< bool has_null >
    ARIES_HOST_DEVICE bool binary_find( const char* keys, size_t len, int count, const char* key )
    {
        int begin = 0;
        int end = count;
        while( begin < end )
        {
            int mid = ( begin + end ) / 2;
            const char* key2 = keys + mid * ( len - has_null );
            if( less_t_str_null_smaller< false, has_null >()( key2, key, len ) )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin < count && equal_to_t_str< has_null, false >()( key, keys + begin * ( len - has_null ), len );
    }

    struct CallableComparator
    {
        ARIES_HOST_DEVICE
        CallableComparator( const void* data, int count )
                : Data( data ), Count( count )
        {
        }
        ARIES_HOST_DEVICE
        virtual bool Compare( const void* val ) const = 0;

        ARIES_HOST_DEVICE
        virtual ~CallableComparator()
        {
        }

    protected:
        const void* Data;
        int Count;
    };

    enum bounds_t
    {
        bounds_lower, bounds_upper
    };

    template< bounds_t bounds, typename type_t, typename type_u, typename int_t >
    ARIES_HOST_DEVICE int_t binary_search( const type_u* keys, int_t count, type_t key )
    {
        int_t begin = 0;
        int_t end = count;
        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            bool pred = ( bounds_upper == bounds ) ? key >= keys[mid] : keys[mid] < key;
            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< bounds_t bounds, typename type_u, typename type_t, typename int_t >
    ARIES_HOST_DEVICE int_t binary_search( const type_u** keys_blocks, size_t block_count, int64_t* keys_count_psum, int_t count, type_t key )
    {
        int_t begin = 0;
        int_t end = count;
        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            int_t blockIdx;
            int_t pos = binary_search< bounds_upper >( keys_count_psum, block_count + 1, mid );
            blockIdx =  pos - 1;
            int_t offset = mid - keys_count_psum[ blockIdx ];
            auto keyTmp = keys_blocks[ blockIdx ][ offset ];
            bool pred = ( bounds_upper == bounds ) ? key >= keyTmp : keyTmp < key;
            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< bounds_t bounds, bool has_null, typename int_t >
    ARIES_HOST_DEVICE int_t binary_search( const char* keys, size_t len, int_t count, const char* key )
    {
        int_t begin = 0;
        int_t end = count;
        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            const char* key2 = keys + mid * len;
            bool pred =
                    ( bounds_upper == bounds ) ?
                            !less_t_str_null_smaller< false, has_null >()( key, key2, len ) :
                            less_t_str_null_smaller< has_null, false >()( key2, key, len );
            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< typename type_t >
    ARIES_HOST_DEVICE type_t dict_index( type_t data )
    {
        return data;
    }

END_ARIES_ACC_NAMESPACE

#endif /* FUNCTIONS_HXX_ */
