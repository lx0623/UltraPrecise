/*
 * AriesIndex.h
 *
 *  Created on: Apr 21, 2020
 *      Author: lichi
 */

#ifndef ARIESINDEX_H_
#define ARIESINDEX_H_
#include <memory>
#include <string>
#include <vector>
#include "AriesColumnType.h"
#include "aries_types.hxx"

BEGIN_ARIES_ENGINE_NAMESPACE
// BEGIN_ARIES_ACC_NAMESPACE
using aries_acc::is_simple_type;

    using AriesTupleLocation = int;

    enum class key_hint_t
    {
        most_unique, most_duplicate
    };

    template< typename type_t >
    struct unpacked_nullable_type
    {
        int8_t flag;
        type_t value;

        public:
        ARIES_HOST_DEVICE unpacked_nullable_type()
                : unpacked_nullable_type( 1, type_t
                { } )
        {

        }

        ARIES_HOST_DEVICE unpacked_nullable_type( type_t val )
                : unpacked_nullable_type( 1, val )
        {

        }

        template< typename type_u >
        ARIES_HOST_DEVICE unpacked_nullable_type( const unpacked_nullable_type< type_u >& src )
        {
            flag = src.flag;
            value = src.value;
        }

        ARIES_HOST_DEVICE unpacked_nullable_type( int8_t f, type_t val )
                : flag
                { f }, value
                { val }
        {
            //static_assert( is_simple_type< type_t >::value );
        }

        template< typename type_u, typename aries_acc::enable_if< aries_acc::is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE aries_acc::AriesBool operator<( const type_u& src ) const
        {
            if( flag )
                return value < src;
            else
                return aries_acc::AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE aries_acc::AriesBool operator<( const unpacked_nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value < src.value;
            else
                return aries_acc::AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename aries_acc::enable_if< aries_acc::is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE aries_acc::AriesBool operator<( const type_u& l, const unpacked_nullable_type& r )
        {
            if( r.flag )
                return l < r.value;
            else
                return aries_acc::AriesBool::ValueType::Unknown;


            std::string s;
        }

        template< typename type_u, typename aries_acc::enable_if< aries_acc::is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE aries_acc::AriesBool operator==( const type_u& src ) const
        {
            if( flag )
                return value == src;
            else
                return aries_acc::AriesBool::ValueType::Unknown;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE aries_acc::AriesBool operator==( const unpacked_nullable_type< type_u >& src ) const
        {
            if( flag && src.flag )
                return value == src.value;
            else
                return aries_acc::AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename aries_acc::enable_if< aries_acc::is_comparable_type< type_u >::value, type_u >::type* = nullptr >
        friend ARIES_HOST_DEVICE aries_acc::AriesBool operator==( const type_u& l, const unpacked_nullable_type& r )
        {
            if( r.flag )
                return r.value == l;
            else
                return aries_acc::AriesBool::ValueType::Unknown;
        }

        template< typename type_u, typename aries_acc::enable_if< is_simple_type< type_u >::value, type_u >::type* = nullptr >
        ARIES_HOST_DEVICE unpacked_nullable_type& operator=( const type_u& src )
        {
            flag = 1;
            value = src;
            return *this;
        }

        template< typename type_u >
        ARIES_HOST_DEVICE unpacked_nullable_type& operator=( const unpacked_nullable_type< type_u >& src )
        {
            flag = src.flag;
            value = src.value;
            return *this;
        }
    };

    struct AriesCompositeKeyType
    {
        AriesCompositeKeyType()
        {
        }
        AriesCompositeKeyType( const std::string& key )
                : m_key( key )
        {
        }
        AriesCompositeKeyType( const AriesCompositeKeyType& a )
        {
            m_key = a.m_key;
        }
        AriesCompositeKeyType( AriesCompositeKeyType && a )
        {
            m_key = std::move( a.m_key );
        }

        AriesCompositeKeyType & operator=( const AriesCompositeKeyType & a )
        {
            m_key = a.m_key;
            return *this;
        }
        AriesCompositeKeyType & operator=( AriesCompositeKeyType && a )
        {
            m_key = std::move( a.m_key );
            return *this;
        }

        std::string m_key;
    };

    class IAriesIndex
    {
    public:
        virtual bool KeyExists( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindEqual( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindNotEqual( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindLess( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindLessOrEqual( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindGreater( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindGreaterOrEqual( const void* key ) const = 0;
        virtual std::vector< AriesTupleLocation > FindGTAndLT( const void* minKey, const void* maxKey ) const = 0;
        virtual std::vector< AriesTupleLocation > FindGEAndLT( const void* minKey, const void* maxKey ) const = 0;
        virtual std::vector< AriesTupleLocation > FindGTAndLE( const void* minKey, const void* maxKey ) const = 0;
        virtual std::vector< AriesTupleLocation > FindGEAndLE( const void* minKey, const void* maxKey ) const = 0;
        virtual bool Save( const std::string& filePath ) const = 0;
        virtual bool Load( const std::string& filePath ) = 0;
        virtual std::pair< void*, bool > Insert( const void* key, AriesTupleLocation value ) = 0;
        virtual bool Erase( const void* key, AriesTupleLocation value ) = 0;
        virtual void DumpStatus() const = 0;
        virtual void Lock() = 0;
        virtual void Unlock() = 0;
        virtual ~IAriesIndex()
        {
        }
    };
    using IAriesIndexSPtr = std::shared_ptr< IAriesIndex >;

    class AriesDictKeyComparator;
    class AriesIndexCreator
    {
    public:
        static IAriesIndexSPtr CreateAriesIndex( const std::vector< aries::AriesColumnType >& types, key_hint_t hint );
        static IAriesIndexSPtr CreateAriesIndex( const aries::AriesColumnType& type, key_hint_t hint );
        // static IAriesIndexSPtr CreateAriesIndexKeyPosition( const aries_acc::AriesDictKeyComparator& comp, key_hint_t hint );
    };

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESINDEX_H_ */
