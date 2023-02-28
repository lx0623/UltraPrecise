// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#ifndef ARIESOPERATORS_HXX_
#define ARIESOPERATORS_HXX_
#include "meta.hxx"
#include "datatypes/functions.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    namespace detail
    {

        template< typename it_t, typename type_t = typename std::iterator_traits< it_t >::value_type, bool use_ldg = std::is_pointer< it_t >::value
                && std::is_arithmetic< type_t >::value >
        struct ldg_load_t
        {
            ARIES_HOST_DEVICE
            static type_t load( it_t it )
            {
                return *it;
            }
        };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350

        template< typename it_t, typename type_t >
        struct ldg_load_t< it_t, type_t, true >
        {
            ARIES_HOST_DEVICE
            static type_t load( it_t it )
            {
                return __ldg( it );
            }
        };

#endif

    } // namespace detail

    template< typename it_t >
    ARIES_HOST_DEVICE typename std::iterator_traits< it_t >::value_type ldg( it_t it )
    {
        return detail::ldg_load_t< it_t >::load( it );
    }

    template< typename real_t >
    ARIES_HOST_DEVICE real_t sq( real_t x )
    {
        return x * x;
    }

    template< typename type_t >
    ARIES_HOST_DEVICE void aries_swap( type_t& a, type_t& b )
    {
        type_t c = a;
        a = b;
        b = c;
    }

    ARIES_HOST_DEVICE void aries_swap( char* a, char* b, int len )
    {
        while( len-- )
            aries_swap( *a++, *b++ );
    }

////////////////////////////////////////////////////////////////////////////////
// Device-side comparison operators.

    template< typename type_t, typename type_u = type_t >
    struct less_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? ( AriesBool )( a.value < b ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( AriesBool )( a < b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct less_t< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( AriesBool )( a.value < b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct less_t_null_smaller: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t_null_smaller< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return !a.flag || a.value < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t_null_smaller< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag && a < b.value;
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct less_t_null_smaller< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            if( a.flag || b.flag )
                return ( b.flag && ( !a.flag || a.value < b.value ) );
            else
                return false;
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct less_t_null_smaller_join: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t_null_smaller_join< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return !a.flag || a.value < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t_null_smaller_join< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag && a < b.value;
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct less_t_null_smaller_join< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            if( a.flag || b.flag )
                return ( AriesBool )( b.flag && ( !a.flag || a.value < b.value ) );
            else
                return AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< bool left_has_null, bool right_has_null >
    struct less_t_str_null_smaller_join: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct less_t_str_null_smaller_join< true, true > : public std::binary_function< char*, char*, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return ( AriesBool )( *b && ( !*a || ( ++a, ++b, str_less_t( a, b, len - 1 ) ) ) );
            else
                return AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct less_t_str_null_smaller_join< true, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return !*a || ( ++a, str_less_t( a, b, aLen - 1 ) );
        }
    };

    template< >
    struct less_t_str_null_smaller_join< false, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return *b && ( ++b, str_less_t( a, b, bLen - 1 ) );
        }
    };

    template< >
    struct less_t_str_null_smaller_join< false, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_less_t( a, b, len );
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct less_t_null_bigger: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t_null_bigger< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag && a.value < b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_t_null_bigger< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return !b.flag || a < b.value;
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct less_t_null_bigger< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            if( a.flag || b.flag )
                return a.flag && ( !b.flag || a.value < b.value );
            else
                return false;
        }
    };

    template< bool left_has_null, bool right_has_null = left_has_null >
    struct less_t_CompactDecimal_null_smaller
    {
    };

    template< >
    struct less_t_CompactDecimal_null_smaller< true, true > : public CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }

        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *b && ( !*a || ( ++a, ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) ) );
            else
                return false;
        }
    };

    template< >
    struct less_t_CompactDecimal_null_smaller< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return !*a || ( ++a, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
        }
    };

    template< >
    struct less_t_CompactDecimal_null_smaller< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return *b && ( ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
        }
    };

    template< >
    struct less_t_CompactDecimal_null_smaller< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< bool left_has_null, bool right_has_null = left_has_null >
    struct less_t_CompactDecimal_null_smaller_join
    {
    };

    template< >
    struct less_t_CompactDecimal_null_smaller_join< true, true > : public CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller_join( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }

        ARIES_HOST_DEVICE
        AriesBool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *b && ( !*a || ( ++a, ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) ) );
            else
                return AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< >
    struct less_t_CompactDecimal_null_smaller_join< true, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller_join( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return !*a || ( ++a, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
        }
    };

    template< >
    struct less_t_CompactDecimal_null_smaller_join< false, true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller_join( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return *b && ( ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
        }
    };

    template< >
    struct less_t_CompactDecimal_null_smaller_join< false, false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_smaller_join( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< bool left_has_null, bool right_has_null >
    struct less_t_str_null_bigger: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct less_t_str_null_bigger< true, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *a && ( !*b || ( ++a, ++b, str_less_t( a, b, len - 1 ) ) );
            else
                return false;
        }
    };

    template< >
    struct less_t_str_null_bigger< true, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return *a && ( ++a, str_less_t( a, b, aLen - 1 ) );
        }
    };

    template< >
    struct less_t_str_null_bigger< false, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return !*b || ( ++b, str_less_t( a, b, bLen - 1 ) );
        }
    };

    template< >
    struct less_t_str_null_bigger< false, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_less_t( a, b, len );
        }
    };

    template< bool hasNull >
    struct less_t_CompactDecimal_null_bigger: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct less_t_CompactDecimal_null_bigger< true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_bigger( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *a && ( !*b || ( ++a, ++b, CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) ) );
            else
                return false;
        }
    };

    template< >
    struct less_t_CompactDecimal_null_bigger< false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE less_t_CompactDecimal_null_bigger( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_less_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct less_equal_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a <= b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_equal_t< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? ( AriesBool )( a.value <= b ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct less_equal_t< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( AriesBool )( a <= b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct less_equal_t< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( AriesBool )( a.value <= b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct greater_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a > b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_t< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? ( AriesBool )( a.value > b ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_t< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( AriesBool )( a > b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct greater_t< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( AriesBool )( a.value > b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct greater_t_null_smaller: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a > b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_t_null_smaller< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag && a.value > b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_t_null_smaller< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return !b.flag || a > b.value;
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct greater_t_null_smaller< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            if( a.flag || b.flag )
                return ( a.flag && ( !b.flag || a.value > b.value ) );
            else
                return false;
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct greater_t_null_bigger: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a > b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_t_null_bigger< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return !a.flag || a.value > b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_t_null_bigger< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return !b.flag && a > b.value;
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct greater_t_null_bigger< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            if( a.flag || b.flag )
                return b.flag && ( !a.flag || a.value > b.value );
            else
                return false;
        }
    };

    template< bool left_has_null, bool right_has_null >
    struct greater_t_str_null_smaller: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct greater_t_str_null_smaller< true, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *a && ( !*b || ( ++a, ++b, str_greater_t( a, b, len - 1 ) ) );
            else
                return false;

        }
    };

    template< >
    struct greater_t_str_null_smaller< true, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return *a && ( ++a, str_greater_t( a, b, aLen - 1 ) );
        }
    };

    template< >
    struct greater_t_str_null_smaller< false, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return !*b || ( ++b, str_greater_t( a, b, bLen - 1 ) );
        }
    };

    template< >
    struct greater_t_str_null_smaller< false, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_greater_t( a, b, len );
        }
    };

    template< bool hasNull >
    struct greater_t_CompactDecimal_null_smaller: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct greater_t_CompactDecimal_null_smaller< true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE greater_t_CompactDecimal_null_smaller( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a || *b )
                return *a && ( !*b || ( ++a, ++b, CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) ) );
            else
                return false;
        }
    };

    template< >
    struct greater_t_CompactDecimal_null_smaller< false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE greater_t_CompactDecimal_null_smaller( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< bool left_has_null, bool right_has_null >
    struct greater_t_str_null_bigger: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct greater_t_str_null_bigger< true, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *b || *a )
                return *b && ( !*a || ( ++a, ++b, str_greater_t( a, b, len - 1 ) ) );
            else
                return false;
        }
    };

    template< >
    struct greater_t_str_null_bigger< true, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            return !*a || ( ++a, str_greater_t( a, b, aLen - 1 ) );
        }
    };

    template< >
    struct greater_t_str_null_bigger< false, true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            return !*b && ( ++b, str_greater_t( a, b, bLen - 1 ) );
        }
    };

    template< >
    struct greater_t_str_null_bigger< false, false > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_greater_t( a, b, len );
        }
    };

    template< bool hasNull >
    struct greater_t_CompactDecimal_null_bigger: public std::binary_function< char*, char*, bool >
    {
    };

    template< >
    struct greater_t_CompactDecimal_null_bigger< true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE greater_t_CompactDecimal_null_bigger( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *b || *a )
                return *b && ( !*a || ( ++a, ++b, CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) ) );
            else
                return false;
        }
    };

    template< >
    struct greater_t_CompactDecimal_null_bigger< false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE greater_t_CompactDecimal_null_bigger( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_greater_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct greater_equal_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a >= b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_equal_t< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? ( AriesBool )( a.value >= b ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct greater_equal_t< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( AriesBool )( a >= b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct greater_equal_t< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( AriesBool )( a.value >= b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct equal_to_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a == b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct equal_to_t< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? AriesBool( a.value == b ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct equal_to_t< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( AriesBool )( a == b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct equal_to_t< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( AriesBool )( a.value == b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< bool left_has_null, bool right_has_null >
    struct equal_to_t_str_null_eq
    {
    };

    template< >
    struct equal_to_t_str_null_eq< true, true >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( bool )( ++a, ++b, str_equal_to_t( a, b, len - 1 ) );
            else
                return !( *a || *b );
        }
    };

    template< >
    struct equal_to_t_str_null_eq< true, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int aLen ) const
        {
            if( *a )
                return ( bool )( ++a, str_equal_to_t( a, b, aLen - 1 ) );
            else
                return false;
        }
    };

    template< >
    struct equal_to_t_str_null_eq< false, true >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int bLen ) const
        {
            if( *b )
                return ( bool )( ++b, str_equal_to_t( a, b, bLen - 1 ) );
            else
                return false;
        }
    };

    template< >
    struct equal_to_t_str_null_eq< false, false >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return str_equal_to_t( a, b, len );
        }
    };

    template< bool hasNull >
    struct equal_to_t_CompactDecimal_null_eq
    {
    };

    template< >
    struct equal_to_t_CompactDecimal_null_eq< true > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE equal_to_t_CompactDecimal_null_eq( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            if( *a && *b )
                return ( bool )( ++a, ++b, CompactDecimal_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca ) );
            else
                return !( *a || *b );
        }
    };

    template< >
    struct equal_to_t_CompactDecimal_null_eq< false > : CompactDecimal_cmp_base
    {
        ARIES_HOST_DEVICE equal_to_t_CompactDecimal_null_eq( uint16_t precision, uint16_t scale )
                : CompactDecimal_cmp_base( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* a, const char* b, int len ) const
        {
            return CompactDecimal_equal_t( ( CompactDecimal* )a, prec, sca, ( CompactDecimal* )b, prec, sca );
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct equal_to_t_null_eq: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a == b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct equal_to_t_null_eq< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? ( a.value == b ) : false;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct equal_to_t_null_eq< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( a == b.value ) : false;
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct equal_to_t_null_eq< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( a.value == b.value ) : !( a.flag || b.flag );
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct not_equal_to_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t a, type_u b ) const
        {
            return a != b;
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct not_equal_to_t< type_nullable< type_t >, type_u > : public std::binary_function< type_nullable< type_t >, type_u, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_nullable< type_t > a, type_u b ) const
        {
            return a.flag ? ( AriesBool )( a.value != b ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_nullable, typename type_u >
    struct not_equal_to_t< type_u, type_nullable< type_t > > : public std::binary_function< type_u, type_nullable< type_t >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_u a, type_nullable< type_t > b ) const
        {
            return b.flag ? ( AriesBool )( a != b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

    template< typename type_t, template< typename > class type_t_nullable, typename type_u, template< typename > class type_u_nullable >
    struct not_equal_to_t< type_t_nullable< type_t >, type_u_nullable< type_u > > : public std::binary_function< type_t_nullable< type_t >,
            type_u_nullable< type_u >, AriesBool >
    {
        ARIES_HOST_DEVICE
        AriesBool operator()( type_t_nullable< type_t > a, type_u_nullable< type_u > b ) const
        {
            return a.flag && b.flag ? ( AriesBool )( a.value != b.value ) : AriesBool { AriesBool::ValueType::Unknown };
        }
    };

////////////////////////////////////////////////////////////////////////////////
// Device-side arithmetic operators.

    template< typename type_t >
    struct plus_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return a + b;
        }
    };

    template< typename type_t >
    struct minus_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return a - b;
        }
    };

    template< typename type_t >
    struct multiplies_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return a * b;
        }
    };

    template< typename type_t >
    struct maximum_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return max( a, b );
        }
    };

    template< typename type_t >
    struct minimum_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return min( a, b );
        }
    };

    // for kernel_reduce
    template< typename type_t >
    struct agg_sum_t: public std::binary_function< type_t, type_t, typename std::common_type< type_t, int64_t >::type >
    {
        ARIES_HOST_DEVICE
        typename std::common_type< type_t, int64_t >::type operator()( type_t a, type_t b ) const
        {
            return a + b;
        }
    };

    template< typename type_t, template< typename > class type_nullable >
    struct agg_sum_t< type_nullable< type_t > > : public std::binary_function< type_nullable< type_t >, type_nullable< type_t >,
            type_nullable< typename std::common_type< type_t, int64_t >::type > >
    {
        ARIES_HOST_DEVICE
        type_nullable< typename std::common_type< type_t, int64_t >::type > operator()( type_nullable< type_t > a, type_nullable< type_t > b ) const
        {
            if( a.flag && b.flag )
                return
                {   1, a.value + b.value};
            else if( a.flag )
                return
                {   1, a.value};
            else if( b.flag )
                return
                {   1, b.value};
            else
                return
                {   0, typename std::common_type<type_t, int64_t>::type()};
        }
    };

    template< typename type_t >
    struct agg_max_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return max( a, b );
        }
    };

    template< typename type_t, template< typename > class type_nullable >
    struct agg_max_t< type_nullable< type_t > > : public std::binary_function< type_nullable< type_t >, type_nullable< type_t >,
            type_nullable< type_t > >
    {
        ARIES_HOST_DEVICE
        type_nullable< type_t > operator()( type_nullable< type_t > a, type_nullable< type_t > b ) const
        {
            if( a.flag && b.flag )
                return max( a, b );
            else if( a.flag )
                return a;
            else
                return b;
        }
    };

    template< typename type_t >
    struct agg_min_t: public std::binary_function< type_t, type_t, type_t >
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t a, type_t b ) const
        {
            return min( a, b );
        }
    };

    template< typename type_t, template< typename > class type_nullable >
    struct agg_min_t< type_nullable< type_t > > : public std::binary_function< type_nullable< type_t >, type_nullable< type_t >,
            type_nullable< type_t > >
    {
        ARIES_HOST_DEVICE
        type_nullable< type_t > operator()( type_nullable< type_t > a, type_nullable< type_t > b ) const
        {
            if( a.flag && b.flag )
                return min( a, b );
            else if( a.flag )
                return a;
            else
                return b;
        }
    };

    template< typename type_t, typename type_u = type_t >
    struct op_in_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t value, const type_u* items, int count ) const
        {
            while( count-- )
            {
                if( value == *items++ )
                    return true;
            }
            return false;
        }
    };

    template< bool has_null = false >
    struct op_in_t_str: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            equal_to_t_str< false, false > eq;
            while( count-- )
            {
                if( eq( value, items, len ) )
                    return true;
                items += len;
            }
            return false;
        }
    };

    template< >
    struct op_in_t_str< true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            equal_to_t_str< true, false > eq;
            if( *value )
            {
                while( count-- )
                {
                    if( eq( value, items, len ) )
                        return true;
                    items += len - 1;
                }
                return false;
            }
            else
                return false;
        }
    };

    template< bool hasNull = false >
    struct op_in_t_decimal: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE op_in_t_decimal( int precision, int scale )
                : eq( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            while( count-- )
            {
                if( eq( value, items, len ) )
                    return true;
                items += len;
            }
            return false;
        }
        equal_to_t_CompactDecimal< false, false > eq;
    };

    template< >
    struct op_in_t_decimal< true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE op_in_t_decimal( int precision, int scale )
                : eq( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            if( *value )
            {
                while( count-- )
                {
                    if( eq( value, items, len ) )
                        return true;
                    items += len - 1;
                }
                return false;
            }
            else
                return false;
        }
        equal_to_t_CompactDecimal< true, false > eq;
    };

    template< typename type_t, typename type_u = type_t >
    ARIES_HOST_DEVICE_NO_INLINE bool binary_find_for_in( const void* vkeys, size_t len, int count, const void* vkey )
    {
        const type_t* keys = ( const type_t* )vkeys;
        type_u key = *( type_u* )vkey;
        int begin = 0;
        int end = count;
        while( begin < end )
        {
            int mid = ( begin + end ) / 2;
            if( keys[ mid ] < key )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin < count && key == keys[ begin ];
    }

    template< bool has_null >
    ARIES_HOST_DEVICE_NO_INLINE bool binary_find_for_in( const void* vkeys, size_t len, int count, const void* vkey )
    {
        const char* keys = ( const char* )vkeys;
        const char* key = ( const char* )vkey;
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

    template< bool hasNull >
    struct binary_find_decimal
    {
        ARIES_HOST_DEVICE binary_find_decimal( int precision, int scale )
                : lt( precision, scale ), eq( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* keys, size_t len, int count, const char* key ) const
        {
            int begin = 0;
            int end = count;
            while( begin < end )
            {
                int mid = ( begin + end ) / 2;
                const char* key2 = keys + mid * ( len - hasNull );
                if( lt( key2, key, len ) )
                    begin = mid + 1;
                else
                    end = mid;
            }
            return begin < count && eq( key, keys + begin * ( len - hasNull ), len );
        }
        less_t_CompactDecimal_null_smaller< false, hasNull > lt;
        equal_to_t_CompactDecimal< hasNull, false > eq;
    };

    template< bounds_t bounds, typename type_t, typename type_u, typename int_t >
    ARIES_HOST_DEVICE int_t binary_search_by_associated( const type_u* keys, int_t count, int_t* associated, type_t key )
    {
        int_t begin = 0;
        int_t end = count;
        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            bool pred = ( bounds_upper == bounds ) ? key >= keys[ associated[ mid ] ] : keys[ associated[ mid ] ] < key;
            if( pred )
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template< bounds_t bounds, bool has_null, typename int_t >
    ARIES_HOST_DEVICE int_t binary_search_by_associated( const char* keys, size_t len, int_t count, int_t* associated, const char* key )
    {
        int_t begin = 0;
        int_t end = count;
        while( begin < end )
        {
            int_t mid = ( begin + end ) / 2;
            const char* key2 = keys + associated[ mid ] * len;
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

    template< typename type_t, typename type_u = type_t >
    struct op_in_t_sorted: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t value, const type_u* items, int count ) const
        {
            return binary_find( items, count, value );
        }
    };

    template< bool has_null >
    struct op_in_t_str_sorted: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            return binary_find< has_null >( items, len, count, value );
        }
    };

    template< bool hasNull >
    struct op_in_t_decimal_sorted: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE op_in_t_decimal_sorted( int precision, int scale )
                : find_decimal( precision, scale )
        {

        }

        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            return find_decimal( items, len, count, value );
        }
        binary_find_decimal< hasNull > find_decimal;
    };

    template< typename type_t, typename type_u = type_t >
    struct op_notin_t_sorted: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t value, const type_u* items, int count ) const
        {
            return !binary_find( items, count, value );
        }
    };

    template< bool has_null >
    struct op_notin_t_str_sorted: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            return !binary_find< has_null >( items, len, count, value );
        }
    };

    template< bool hasNull >
    struct op_notin_t_decimal_sorted: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE op_notin_t_decimal_sorted( int precision, int scale )
                : find_decimal( precision, scale )
        {

        }

        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            return !find_decimal( items, len, count, value );
        }
        binary_find_decimal< hasNull > find_decimal;
    };

    template< typename type_t, typename type_u = type_t >
    struct op_notin_t: public std::binary_function< type_t, type_u, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( type_t value, const type_u* items, int count ) const
        {
            bool bRet = true;
            while( count-- )
            {
                bRet = bRet && ( value != *items++ );
                if( !bRet )
                    break;
            }
            return bRet;
        }
    };

    template< bool has_null = false >
    struct op_notin_t_str: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            bool bRet = true;

            not_equal_to_t_str< false, false > ne;
            while( count-- )
            {
                bRet = bRet && ne( value, items, len );
                if( !bRet )
                    break;
                items += len;
            }
            return bRet;
        }
    };

    template< >
    struct op_notin_t_str< true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            if( *value )
            {
                bool bRet = true;

                not_equal_to_t_str< true, false > ne;
                while( count-- )
                {
                    bRet = bRet && ne( value, items, len );
                    if( !bRet )
                        break;
                    items += len - 1;
                }
                return bRet;
            }
            else
                return false;
        }
    };

    template< bool has_null = false >
    struct op_notin_t_decimal: public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE op_notin_t_decimal( int precision, int scale )
                : ne( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            bool bRet = true;
            while( count-- )
            {
                bRet = bRet && ne( value, items, len );
                if( !bRet )
                    break;
                items += len;
            }
            return bRet;
        }
        not_equal_to_t_CompactDecimal< false, false > ne;
    };

    template< >
    struct op_notin_t_decimal< true > : public std::binary_function< char*, char*, bool >
    {
        ARIES_HOST_DEVICE op_notin_t_decimal( int precision, int scale )
                : ne( precision, scale )
        {

        }
        ARIES_HOST_DEVICE
        bool operator()( const char* value, int len, const char* items, int count ) const
        {
            if( *value )
            {
                bool bRet = true;
                while( count-- )
                {
                    bRet = bRet && ne( value, items, len );
                    if( !bRet )
                        break;
                    items += len - 1;
                }
                return bRet;
            }
            else
                return false;
        }
        not_equal_to_t_CompactDecimal< true, false > ne;
    };

////////////////////////////////////////////////////////////////////////////////
// iterator_t and const_iterator_t are base classes for customized iterators.

    template< typename outer_t, typename int_t, typename value_type >
    struct iterator_t: public std::iterator_traits< const value_type* >
    {

        iterator_t() = default;
        ARIES_HOST_DEVICE iterator_t( int_t i )
                : index( i )
        {
        }

        ARIES_HOST_DEVICE
        outer_t operator+( int_t diff ) const
        {
            outer_t next = *static_cast< const outer_t* >( this );
            next += diff;
            return next;
        }
        ARIES_HOST_DEVICE
        outer_t operator-( int_t diff ) const
        {
            outer_t next = *static_cast< const outer_t* >( this );
            next -= diff;
            return next;
        }
        ARIES_HOST_DEVICE
        outer_t& operator+=( int_t diff )
        {
            index += diff;
            return *static_cast< outer_t* >( this );
        }
        ARIES_HOST_DEVICE
        outer_t& operator-=( int_t diff )
        {
            index -= diff;
            return *static_cast< outer_t* >( this );
        }

        int_t index;
    };

    template< typename outer_t, typename int_t, typename value_type >
    struct const_iterator_t: public iterator_t< outer_t, int_t, value_type >
    {
        typedef iterator_t< outer_t, int_t, value_type > base_t;

        const_iterator_t() = default;
        ARIES_HOST_DEVICE const_iterator_t( int_t i )
                : base_t( i )
        {
        }

        // operator[] and operator* are tagged as DEVICE-ONLY.  This is to ensure
        // compatibility with lambda capture in CUDA 7.5, which does not support
        // marking a lambda as __host__ __device__.
        // We hope to relax this when a future CUDA fixes this problem.
        ARIES_HOST_DEVICE
        value_type operator[]( int_t diff ) const
        {
            return static_cast< const outer_t& >( *this )( base_t::index + diff );
        }
        ARIES_HOST_DEVICE
        value_type operator*() const
        {
            return ( *this )[ 0 ];
        }
    };

////////////////////////////////////////////////////////////////////////////////
// discard_iterator_t is a store iterator that discards its input.

    template< typename value_type >
    struct discard_iterator_t: iterator_t< discard_iterator_t< value_type >, int, value_type >
    {

        struct assign_t
        {
            ARIES_HOST_DEVICE
            value_type operator=( value_type v )
            {
                return value_type();
            }
        };

        ARIES_HOST_DEVICE
        assign_t operator[]( int index ) const
        {
            return assign_t();
        }
        ARIES_HOST_DEVICE
        assign_t operator*() const
        {
            return assign_t();
        }
    };

////////////////////////////////////////////////////////////////////////////////
// counting_iterator_t returns index.

    template< typename type_t, typename int_t = int >
    struct counting_iterator_t: const_iterator_t< counting_iterator_t< type_t, int_t >, int_t, type_t >
    {

        counting_iterator_t() = default;
        ARIES_HOST_DEVICE counting_iterator_t( type_t i )
                : const_iterator_t< counting_iterator_t, int_t, type_t >( i )
        {
        }

        ARIES_HOST_DEVICE
        type_t operator()( int_t index ) const
        {
            return ( type_t )index;
        }
    };

////////////////////////////////////////////////////////////////////////////////
// strided_iterator_t returns offset + index * stride.

    template< typename type_t, typename int_t = int >
    struct strided_iterator_t: const_iterator_t< strided_iterator_t< type_t >, int_t, int >
    {

        strided_iterator_t() = default;
        ARIES_HOST_DEVICE strided_iterator_t( type_t offset_, type_t stride_ )
                : const_iterator_t< strided_iterator_t, int_t, type_t >( 0 ), offset( offset_ ), stride( stride_ )
        {
        }

        ARIES_HOST_DEVICE
        type_t operator()( int_t index ) const
        {
            return offset + index * stride;
        }

        type_t offset, stride;
    };

////////////////////////////////////////////////////////////////////////////////
// constant_iterator_t returns the value it was initialized with.

    template< typename type_t >
    struct constant_iterator_t: const_iterator_t< constant_iterator_t< type_t >, int, type_t >
    {

        type_t value;

        ARIES_HOST_DEVICE constant_iterator_t( type_t value_ )
                : value( value_ )
        {
        }

        ARIES_HOST_DEVICE
        type_t operator()( int index ) const
        {
            return value;
        }
    };

// These types only supported with nvcc until CUDA 8.0 allows host-device
// lambdas and ARIES_LAMBDA is redefined to ARIES_HOST_DEVICE

#ifdef __CUDACC__

////////////////////////////////////////////////////////////////////////////////
// lambda_iterator_t

    template< typename load_t, typename store_t, typename value_type, typename int_t >
    struct lambda_iterator_t: std::iterator_traits< const value_type* >
    {

        load_t load;
        store_t store;
        int_t base;

        lambda_iterator_t( load_t load_, store_t store_, int_t base_ )
                : load( load_ ), store( store_ ), base( base_ )
        {
        }

        struct assign_t
        {
            load_t load;
            store_t store;
            int_t index;

            ARIES_LAMBDA
            assign_t& operator=( value_type rhs )
            {
                static_assert(!std::is_same<store_t, empty_t>::value,
                        "load_iterator is being stored to.");
                store( rhs, index );
                return *this;
            }
            ARIES_LAMBDA operator value_type() const
            {
                static_assert(!std::is_same<load_t, empty_t>::value,
                        "store_iterator is being loaded from.");
                return load( index );
            }
        };

        ARIES_LAMBDA
        assign_t operator[]( int_t index ) const
        {
            return assign_t { load, store, base + index };
        }
        ARIES_LAMBDA
        assign_t operator*() const
        {
            return assign_t { load, store, base };
        }

        ARIES_HOST_DEVICE
        lambda_iterator_t operator+( int_t offset ) const
        {
            lambda_iterator_t cp = *this;
            cp += offset;
            return cp;
        }

        ARIES_HOST_DEVICE
        lambda_iterator_t& operator+=( int_t offset )
        {
            base += offset;
            return *this;
        }

        ARIES_HOST_DEVICE
        lambda_iterator_t operator-( int_t offset ) const
        {
            lambda_iterator_t cp = *this;
            cp -= offset;
            return cp;
        }

        ARIES_HOST_DEVICE
        lambda_iterator_t& operator-=( int_t offset )
        {
            base -= offset;
            return *this;
        }
    };

    template< typename value_type >
    struct trivial_load_functor
    {
        template< typename int_t >
        ARIES_HOST_DEVICE value_type operator()( int_t index ) const
        {
            return value_type();
        }
    };

    template< typename value_type >
    struct trivial_store_functor
    {
        template< typename int_t >
        ARIES_HOST_DEVICE void operator()( value_type v, int_t index ) const
        {
        }
    };

    template< typename value_type, typename int_t = int, typename load_t, typename store_t >
    lambda_iterator_t< load_t, store_t, value_type, int_t > make_load_store_iterator( load_t load, store_t store, int_t base = 0 )
    {
        return lambda_iterator_t< load_t, store_t, value_type, int_t >( load, store, base );
    }

    template< typename value_type, typename int_t = int, typename load_t >
    lambda_iterator_t< load_t, empty_t, value_type, int_t > make_load_iterator( load_t load, int_t base = 0 )
    {
        return make_load_store_iterator< value_type >( load, empty_t(), base );
    }

    template< typename value_type, typename int_t = int, typename store_t >
    lambda_iterator_t< empty_t, store_t, value_type, int_t > make_store_iterator( store_t store, int_t base = 0 )
    {
        return make_load_store_iterator< value_type >( empty_t(), store, base );
    }

#endif // #ifdef __CUDACC__

END_ARIES_ACC_NAMESPACE


template< typename type_t, typename type_u >
struct std::common_type< aries_acc::nullable_type< type_t >, aries_acc::nullable_type< type_u > >
{
    typedef aries_acc::nullable_type< typename std::common_type< type_t, type_u >::type > type;
};

template< typename type_t, typename type_u >
struct std::common_type< aries_acc::nullable_type< type_t >, type_u >
{
    typedef aries_acc::nullable_type< typename std::common_type< type_t, type_u >::type > type;
};

template< typename type_t, typename type_u >
struct std::common_type< type_t, aries_acc::nullable_type< type_u > >
{
    typedef aries_acc::nullable_type< typename std::common_type< type_t, type_u >::type > type;
};
#endif
