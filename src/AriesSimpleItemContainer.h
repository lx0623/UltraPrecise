/*
 * AriesSimpleItemContainer.h
 *
 *  Created on: Sep 10, 2020
 *      Author: lichi
 */

#ifndef ARIESSIMPLEITEMCONTAINER_H_
#define ARIESSIMPLEITEMCONTAINER_H_

#include <string>
NAMESPACE_ARIES_START

template< typename type_t >
class AriesSimpleItemContainer
{
public:
    AriesSimpleItemContainer()
    {
    }

    AriesSimpleItemContainer( const string& buff ) : m_buff( buff )
    {
    }

    AriesSimpleItemContainer( const AriesSimpleItemContainer& src )
    {
        m_buff = src.m_buff;
    }

    AriesSimpleItemContainer( AriesSimpleItemContainer&& src )
    {
        m_buff = std::move( src.m_buff );
    }

    AriesSimpleItemContainer& operator=( const AriesSimpleItemContainer& src )
    {
        m_buff = src.m_buff;
        return *this;
    }

    AriesSimpleItemContainer& operator=( AriesSimpleItemContainer&& src )
    {
        m_buff = std::move( src.m_buff );
        return *this;
    }

    void push_back( const type_t& key )
    {
        m_buff.insert( m_buff.size(), ( const char* )&key, sizeof(type_t) );
    }

    size_t size() const
    {
        assert( m_buff.size() % sizeof(type_t) == 0 );
        return m_buff.size() / sizeof(type_t);
    }

    bool empty() const
    {
        return m_buff.empty();
    }

    type_t& operator[]( size_t n )
    {
        assert( n < size() );
        type_t *data = ( type_t* )m_buff.data();
        return data[n];
    }

    const type_t& operator[]( size_t n ) const
    {
        assert( n < size() );
        const type_t *data = ( const type_t* )m_buff.data();
        return data[n];
    }

    void add_offset( int64_t offset )
    {
        for( size_t i = 0; i < size(); ++i )
            (*this)[ i ] += offset;
    }

    AriesSimpleItemContainer clone() const
    {
        return AriesSimpleItemContainer( m_buff );
    }

private:
    std::string m_buff;
};

NAMESPACE_ARIES_END

#endif /* ARIESSIMPLEITEMCONTAINER_H_ */