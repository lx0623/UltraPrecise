/*
 * aries_types.hxx
 *
 *  Created on: Aug 13, 2019
 *      Author: lichi
 */

#ifndef ARIESNULLABLETYPE_HXX_
#define ARIESNULLABLETYPE_HXX_
#include "cpptraits.hxx"
#include "decimal.hxx"
#include "AriesTimestamp.hxx"
#include "AriesYear.hxx"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesTime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< int LEN >
    struct aries_char;

    template< int LEN >
    struct AriesDecimal;

    using PCSTR = const char*;

    template< typename >
    struct is_decimal: public false_type
    {
    };

    template< >
    struct is_decimal< Decimal > : public true_type
    {
    };

    template< int LEN>
    struct is_decimal< AriesDecimal< LEN > > : public true_type
    {
    };

    template< typename _Tp >
    struct is_simple_type: public __or_< is_arithmetic< _Tp >, is_decimal< _Tp > >::type
    {
    };

    template< typename >
    struct is_aries_char: public false_type
    {
    };

    template< int LEN >
    struct is_aries_char< aries_char< LEN > > : public true_type
    {
    };

    template< typename _Tp >
    struct is_can_add_type: public __or_< is_simple_type< _Tp >, is_aries_char< _Tp > >::type
    {
    };


    template< typename >
    struct is_date_type: public false_type
    {

    };

    template< >
    struct is_date_type< AriesDate > : public true_type
    {

    };
    template< >
    struct is_date_type< AriesDatetime > : public true_type
    {

    };

    template< >
    struct is_date_type< AriesTime > : public true_type
    {

    };

    template< >
    struct is_date_type< AriesTimestamp > : public true_type
    {

    };

    template< >
    struct is_date_type< AriesYear > : public true_type
    {

    };

    template< typename _Tp >
    struct is_comparable_type: public __or_< is_simple_type< _Tp >, is_date_type< _Tp > >::type
    {

    };

    class AriesBool
    {
    public:
        enum class ValueType
            : int8_t
            {
                False = 0, True, Unknown
        };
    private:
        ValueType m_value;

    public:
        template< typename type_t >
        ARIES_HOST_DEVICE AriesBool( type_t value )
                : AriesBool( ( bool )value )
        {
        }

        ARIES_HOST_DEVICE AriesBool(): m_value( ValueType::False )
        {
        }

        template< typename type_t, template< typename > class type_nullable >
        ARIES_HOST_DEVICE AriesBool( const type_nullable< type_t >& src )
        {
            m_value = src.flag ? src.value ? AriesBool::ValueType::True : AriesBool::ValueType::False : AriesBool::ValueType::Unknown;
        }

        ARIES_HOST_DEVICE AriesBool( bool value )
                : m_value( value ? ValueType::True : ValueType::False )
        {
        }

        ARIES_HOST_DEVICE AriesBool( AriesBool::ValueType value )
                : m_value( value )
        {
        }

        ARIES_HOST_DEVICE bool is_unknown() const
        {
            return m_value == ValueType::Unknown;
        }

        ARIES_HOST_DEVICE bool is_true() const
        {
            return m_value == ValueType::True;
        }

        ARIES_HOST_DEVICE AriesBool operator !() const
        {
            switch( m_value )
            {
                case ValueType::True:
                    return AriesBool
                    { ValueType::False };
                case ValueType::False:
                    return AriesBool
                    { ValueType::True };
                case ValueType::Unknown:
                    return AriesBool
                    { ValueType::Unknown };
                default:
                    return AriesBool
                    { ValueType::Unknown };
            }
        }

        template< typename type_t >
        ARIES_HOST_DEVICE explicit operator type_t() const
        {
            if( m_value == ValueType::True )
                return type_t
                { 1 };
            else
                return type_t
                { 0 };
        }

        ARIES_HOST_DEVICE operator bool() const
        {
            return m_value == ValueType::True;
        }


//        ARIES_HOST_DEVICE explicit operator int32_t() const
//        {
//            return m_value == ValueType::True;
//        }

        template< typename type_t >
        ARIES_HOST_DEVICE AriesBool operator &&( const type_t& src ) const
        {
            return *this && AriesBool( src );
        }

        ARIES_HOST_DEVICE AriesBool operator &&( const AriesBool& src ) const
        {
            switch( m_value )
            {
                case ValueType::True:
                    switch( src.m_value )
                    {
                        case ValueType::True:
                            return AriesBool
                            { ValueType::True };
                        case ValueType::False:
                            return AriesBool
                            { ValueType::False };
                        case ValueType::Unknown:
                            return AriesBool
                            { ValueType::Unknown };
                        default:
                            return AriesBool
                            { ValueType::Unknown };
                    }
                case ValueType::False:
                    return AriesBool
                    { ValueType::False };
                case ValueType::Unknown:
                    switch( src.m_value )
                    {
                        case ValueType::True:
                            return AriesBool
                            { ValueType::Unknown };
                        case ValueType::False:
                            return AriesBool
                            { ValueType::False };
                        case ValueType::Unknown:
                            return AriesBool
                            { ValueType::Unknown };
                        default:
                            return AriesBool
                            { ValueType::Unknown };
                    }
                default:
                    return AriesBool
                    { ValueType::Unknown };
            }
        }

        template< typename type_t >
        ARIES_HOST_DEVICE AriesBool operator ||( const type_t& src ) const
        {
            return *this || AriesBool( src );
        }

        ARIES_HOST_DEVICE AriesBool operator ||( const AriesBool& src ) const
        {
            switch( m_value )
            {
                case ValueType::True:
                    return AriesBool
                    { ValueType::True };
                case ValueType::False:
                    switch( src.m_value )
                    {
                        case ValueType::True:
                            return AriesBool
                            { ValueType::True };
                        case ValueType::False:
                            return AriesBool
                            { ValueType::False };
                        case ValueType::Unknown:
                            return AriesBool
                            { ValueType::Unknown };
                        default:
                            return AriesBool
                            { ValueType::Unknown };
                    }
                case ValueType::Unknown:
                    switch( src.m_value )
                    {
                        case ValueType::True:
                            return AriesBool
                            { ValueType::True };
                        case ValueType::False:
                            return AriesBool
                            { ValueType::Unknown };
                        case ValueType::Unknown:
                            return AriesBool
                            { ValueType::Unknown };
                        default:
                            return AriesBool
                            { ValueType::Unknown };
                    }
                default:
                    return AriesBool
                    { ValueType::Unknown };
            }
        }
    };

    template< typename type_t >
    struct ARIES_PACKED nullable_type
    {
        int8_t flag;
        type_t value;

    public:
        ARIES_HOST_DEVICE nullable_type()
                : nullable_type( 1, type_t
                { } )
        {

        }

//        ARIES_HOST_DEVICE nullable_type( type_t val )
//                : nullable_type( 1, val )
//        {
//
//        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type( type_u val )
                : nullable_type( 1, val )
        {

        }

        ARIES_HOST_DEVICE nullable_type( int8_t f, type_t val )
                : flag
                { f }, value
                { val }
        {
            //static_assert( is_simple_type< type_t >::value );
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type( const nullable_type< type_u >& src )
        {
            flag = src.flag;
            value = src.value;
        }

        ARIES_HOST_DEVICE bool is_null() const
        {
            return !flag;
        }

        ARIES_HOST_DEVICE explicit operator bool() const
        {
            return value && flag;
        }

        ARIES_HOST_DEVICE operator char*()
        {
            return (char*)this;
        }

        ARIES_HOST_DEVICE operator const char*() const
        {
            return (const char*)this;
        }

        ARIES_HOST_DEVICE operator AriesBool() const
        {
            return flag ? AriesBool
            { value } :
                          AriesBool::ValueType::Unknown;
        }

        ARIES_HOST_DEVICE nullable_type& operator++()
        {
            ++value;
            return *this;
        }

        ARIES_HOST_DEVICE nullable_type operator++( int )
        {
            nullable_type tmp( *this );
            operator++();
            return tmp;
        }

        ARIES_HOST_DEVICE nullable_type& operator--()
        {
            --value;
            return *this;
        }

        ARIES_HOST_DEVICE nullable_type operator--( int )
        {
            nullable_type tmp( *this );
            operator--();
            return tmp;
        }

        ARIES_HOST_DEVICE nullable_type operator-() const
        {
            nullable_type tmp( *this );
            tmp.value = -tmp.value;
            return tmp;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator&&( const nullable_type& l, const type_u& r )
        {
            if( l.flag )
                return l.value && r;
            else if( r )
                return AriesBool::ValueType::Unknown;
            else
                return AriesBool::ValueType::False;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator&&( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return r.value && l;
            else if( l )
                return AriesBool::ValueType::Unknown;
            else
                return AriesBool::ValueType::False;
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE AriesBool operator&&( const nullable_type& l, const nullable_type< type_u >& r )
        {
            if( l.flag && r.flag )
                return l.value && r.value;
            else if( l.flag )
                return l.value ? AriesBool::ValueType::Unknown : AriesBool::ValueType::False;
            else if( r.flag )
                return r.value ? AriesBool::ValueType::Unknown : AriesBool::ValueType::False;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator||( const nullable_type& l, const type_u& r )
        {
            if( l.flag )
                return l.value || r;
            else if( r )
                return AriesBool::ValueType::True;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator||( const type_u& l, const nullable_type& r)
        {
            if( r.flag )
                return r.value || l;
            else if( l )
                return AriesBool::ValueType::True;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE AriesBool operator||( const nullable_type& l, const nullable_type< type_u >& r )
        {
            if( l.flag && r.flag )
                return l.value || r.value;
            else if( l.flag )
                return l.value ? AriesBool::ValueType::True : AriesBool::ValueType::Unknown;
            else if( r.flag )
                return r.value ? AriesBool::ValueType::True : AriesBool::ValueType::Unknown;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_can_add_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator+=( const type_u& src )
        {
            value += src;
            return *this;
        }

        template< typename type_u, typename enable_if< is_can_add_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator+=( const nullable_type< type_u >& src )
        {
            value += src.value;
            flag = flag && src.flag;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator+( nullable_type l, type_u r )
        {
            auto value = l.value + r;
            return nullable_type< decltype(value) >
            { l.flag, value };
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator+( const type_u& l, const nullable_type& r )
        {
            auto value = r.value + l;
            return nullable_type< decltype(value) >
            { r.flag, value };
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator+( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value + r.value;
            char flag = l.flag && r.flag;
            return nullable_type< decltype(value) >
            { flag, value };
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator-=( const type_u& src )
        {
            value -= src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator-=( const nullable_type< type_u >& src )
        {
            value -= src.value;
            flag = flag && src.flag;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator-( const nullable_type& l, const type_u& r )
        {
            auto value = l.value - r;
            return nullable_type< decltype(value) >( l.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator-( const type_u& l, const nullable_type& r )
        {
            auto value = l - r.value;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator-( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value - r.value;
            char flag = l.flag && r.flag;
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator*=( const type_u& src )
        {
            value *= src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator*=( const nullable_type< type_u >& src )
        {
            value *= src.value;
            flag = flag && src.flag;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator*( const nullable_type& l, const type_u& r )
        {
            auto value = l.value * r;
            return nullable_type< decltype(value) >( l.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator*( const type_u& l, const nullable_type& r )
        {
            auto value = r.value * l;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator*( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value * r.value;
            char flag = l.flag && r.flag;
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator<<=( const type_u& src )
        {
            value <<= src;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator>>=( const type_u& src )
        {
            value >>= src;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator<<( const nullable_type& l, const type_u& r )
        {
            auto value = l.value << r;
            return nullable_type< decltype(value) >( l.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator<<( const type_u& l, const nullable_type& r )
        {
            auto value = l << r.value;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator>>( const nullable_type& l, const type_u& r )
        {
            auto value = l.value >> r;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator>>( const type_u& l, const nullable_type& r )
        {
            auto value = l >> r.value;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator/=( const type_u& src )
        {
            value /= src;
            flag = flag && src; // if src.value is 0, return null
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator/=( const nullable_type< type_u >& src )
        {
            value /= src.value;
            flag = flag && src.flag && src.value; // if src.value is 0, return null
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator/( const type_u& l, const nullable_type& r )
        {
            auto value = l / r.value;
            char flag = r.flag && r.value; // if r.value is 0, return null
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator/( const nullable_type& l, const type_u& r )
        {
            auto value = l.value / r;
            char flag = l.flag && r; // if r.value is 0, return null
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator/( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value / r.value;
            char flag = l.flag && r.flag && r.value; // if r.value is 0, return null
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator^=( const type_u& src )
        {
            value ^= src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator^=( const nullable_type< type_u >& src )
        {
            value ^= src.value;
            flag = flag && src.flag;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator^( const nullable_type& l, const type_u& r )
        {
            auto value = l.value ^ r;
            return nullable_type< decltype(value) >( l.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator^( const type_u& l, const nullable_type& r )
        {
            auto value = l ^ r.value;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator^( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value ^ r.value;
            char flag = l.flag && r.flag;
            return nullable_type< decltype(value) >( flag, value );
        }

        ARIES_HOST_DEVICE nullable_type operator~()
        {
            return nullable_type( flag, ~value );
        }

        ARIES_HOST_DEVICE nullable_type operator!()
        {
            return nullable_type( flag, !value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator%=( const type_u& src )
        {
            value %= src;
            flag = flag && src.flag && src; // if src is 0, return null
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator%=( const nullable_type< type_u >& src )
        {
            value %= src.value;
            flag = flag && src.flag && src.value; // if src.value is 0, return null
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator%( const type_u& l, const nullable_type& r )
        {
            auto value = l % r.value;
            char flag = r.flag && r.value; // if r.value is 0, return null
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator%( const nullable_type& l, const type_u& r )
        {
            auto value = l.value % r;
            char flag = l.flag && r; // if r is 0, return null
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator%( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value % r.value;
            char flag = l.flag && r.flag && r.value; // if r.value is 0, return null
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator&=( const type_u& src )
        {
            value &= src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator&=( const nullable_type< type_u >& src )
        {
            value &= src.value;
            flag = flag && src.flag;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator&( const nullable_type& l, const type_u& r )
        {
            auto value = l.value & r;
            return nullable_type< decltype(value) >( l.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator&( const type_u& l, const nullable_type& r )
        {
            auto value = l & r.value;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator&( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value & r.value;
            char flag = l.flag && r.flag;
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator|=( const type_u& src )
        {
            value |= src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator|=( const nullable_type< type_u >& src )
        {
            value |= src.value;
            flag = flag && src.flag;
            return *this;
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator|( const nullable_type& l, const type_u& r )
        {
            auto value = l.value | r;
            return nullable_type< decltype(value) >( l.flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE auto operator|( const type_u& l, const nullable_type& r )
        {
            auto value = l | r.value;
            return nullable_type< decltype(value) >( r.flag, value );
        }

        template< typename type_u >
        friend ARIES_HOST_DEVICE auto operator|( const nullable_type& l, const nullable_type< type_u >& r )
        {
            auto value = l.value | r.value;
            char flag = l.flag && r.flag;
            return nullable_type< decltype(value) >( flag, value );
        }

        template< typename type_u, typename enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE nullable_type& operator=( const type_u& src )
        {
            flag = 1;
            value = src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE nullable_type& operator=( const nullable_type< type_u >& src )
        {
            flag = src.flag;
            value = src.value;
            return *this;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE AriesBool operator==( const type_u& src ) const
        {
            if( flag )
                return value == src;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE AriesBool operator==( const nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value == src.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator==( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return r.value == l;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE AriesBool operator!=( const type_u& src ) const
        {
            if( flag )
                return value != src;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE AriesBool operator!=( const nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value != src.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator!=( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return r.value != l;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE AriesBool operator<( const type_u& src ) const
        {
            if( flag )
                return value < src;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE AriesBool operator<( const nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value < src.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator<( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return l < r.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE AriesBool operator<=( const type_u& src ) const
        {
            if( flag )
                return value <= src;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE AriesBool operator<=( const nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value <= src.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator<=( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return l <= r.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE AriesBool operator>( const type_u& src ) const
        {
            if( flag )
                return value > src;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE AriesBool operator>( const nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value > src.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator>( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return l > r.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE AriesBool operator>=( const type_u& src ) const
        {
            if( flag )
                return value >= src;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE AriesBool operator>=( const nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value >= src.value;
            else
                return AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename enable_if< is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE AriesBool operator>=( const type_u& l, const nullable_type& r )
        {
            if( r.flag )
                return l >= r.value;
            else
                return AriesBool::ValueType::Unknown;
        }
    };
END_ARIES_ACC_NAMESPACE

#endif /* ARIESNULLABLETYPE_HXX_ */
