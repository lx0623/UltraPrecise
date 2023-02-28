/*
 * AriesIndexBtree.h
 *
 *  Created on: Apr 21, 2020
 *      Author: lichi
 */

#ifndef ARIESINDEXBTREE_H_
#define ARIESINDEXBTREE_H_
#include "AriesSimpleKeyComparator.h"
#include "AriesCompositeKeyComparator.h"
#include "AriesDataType.h"
#include <vector>
#include <type_traits>
#include <mutex>

using namespace btree;

namespace boost
{
    namespace serialization
    {
        template< typename Archive >
        void serialize( Archive & ar, aries::AriesDataType & data, const unsigned int version )
        {
            ar & data.ValueType;
            ar & data.Length;
            ar & data.Precision;
            ar & data.Scale;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries::AriesColumnType & data, const unsigned int version )
        {
            ar & data.DataType;
            ar & data.HasNull;
            ar & data.IsUnique;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_engine::AriesCompositeKeyType & data, const unsigned int version )
        {
            ar & data.m_key;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_acc::AriesDate & data, const unsigned int version )
        {
            uint16_t value = data.year;
            ar & value;
            data.year = value;
            ar & data.month;
            ar & data.day;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_acc::AriesDatetime & data, const unsigned int version )
        {
            uint16_t value = data.year;
            ar & value;
            data.year = value;
            ar & data.month;
            ar & data.day;
            ar & data.hour;
            ar & data.minute;
            ar & data.second;
            uint32_t part = data.second_part;
            ar & part;
            data.second_part = part;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_acc::AriesTimestamp & data, const unsigned int version )
        {
            int64_t value = data.timestamp;
            ar & value;
            data.timestamp = value;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_acc::AriesYear & data, const unsigned int version )
        {
            uint16_t value = data.year;
            ar & value;
            data.year = value;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_acc::AriesTime & data, const unsigned int version )
        {
            uint16_t value = data.hour;
            ar & value;
            data.hour = value;
            ar & data.minute;
            ar & data.second;
            uint32_t part = data.second_part;
            ar & part;
            data.second_part = part;
            ar & data.sign;
        }

        template< typename Archive >
        void serialize( Archive & ar, aries_acc::Decimal & data, const unsigned int version )
        {
            ar & data.intg;
            ar & data.frac;
            ar & data.mode;
            ar & data.error;
            for( int i = 0; i < NUM_TOTAL_DIG; ++i )
            {
                int32_t value = data.values[i];
                ar & value;
                data.values[i] = value;
            }
        }

        template< typename Archive, typename type_t >
        void serialize( Archive & ar, aries_engine::unpacked_nullable_type< type_t > & data, const unsigned int version )
        {
            ar & data.flag;
            type_t value = data.value;
            ar & value;
            data.value = value;
        }

        template< typename Archive, typename type_t >
        void serialize( Archive & ar, aries_acc::nullable_type< type_t > & data, const unsigned int version )
        {
            ar & data.flag;
            type_t value = data.value;
            ar & value;
            data.value = value;
        }

        // template< typename Archive >
        // void serialize( Archive &ar, aries_acc::KeyPosition &data, const unsigned int version )
        // {
        //     uint32_t offset = data.offset;
        //     ar & offset;
        //     data.offset = offset;
        // }
    }
}

BEGIN_ARIES_ENGINE_NAMESPACE

    template< typename type_t, key_hint_t hint_t = key_hint_t::most_unique >
    class AriesIndexBtree
    {
    public:
        using AriesSimpleBtree = typename std::conditional< hint_t == key_hint_t::most_unique, btree_multimap< type_t, AriesTupleLocation, AriesSimpleKeyCompartor< type_t > >, btree_map< type_t, vector< AriesTupleLocation >, AriesSimpleKeyCompartor< type_t > > >::type;
        using AriesCompositeBtree = typename std::conditional< hint_t == key_hint_t::most_unique, btree_multimap< AriesCompositeKeyType, AriesTupleLocation, AriesCompositeKeyCompartor >, btree_map< AriesCompositeKeyType, vector< AriesTupleLocation >, AriesCompositeKeyCompartor > >::type;
        // using AriesKeyPositionBtree = typename std::conditional< hint_t == key_hint_t::most_unique, btree_multimap< KeyPosition, AriesTupleLocation, AriesDictKeyComparator >, btree_map< KeyPosition, vector< AriesTupleLocation >, AriesDictKeyComparator > >::type;

        // using AriesBreeMap = typename std::conditional<std::is_same< type_t, AriesCompositeKeyType >::value, AriesCompositeBtree, typename std::conditional<std::is_same< type_t, KeyPosition >::value, AriesKeyPositionBtree, AriesSimpleBtree >::type >::type;
        using AriesBreeMap = typename std::conditional<
                                        std::is_same< type_t, AriesCompositeKeyType >::value, 
                                        AriesCompositeBtree,
                                        AriesSimpleBtree  
                                    >::type;
        using iterator = typename AriesBreeMap::iterator;
        using const_iterator = typename AriesBreeMap::const_iterator;
        using reverse_iterator = typename AriesBreeMap::reverse_iterator;
        using const_reverse_iterator = typename AriesBreeMap::const_reverse_iterator;
        using size_type = typename AriesBreeMap::size_type;
        using key_type = typename AriesBreeMap::key_type;
        using value_type = typename AriesBreeMap::value_type;
        using key_compare = typename AriesBreeMap::key_compare;

    public:
        template< typename T = type_t, typename std::enable_if< std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        AriesIndexBtree( const std::vector< aries::AriesColumnType >& types )
        {
            assert( types.size() > 1 );
            m_columnTypes = types;
            m_comparators = CreateComparators( types );
            AriesCompositeKeyCompartor comp;
            comp.SetComparators( &m_comparators );
            m_btree = std::make_unique < AriesBreeMap > ( comp );

            AriesCompositeKeyCompartor search_comp;
            m_searchComparator.push_back( m_comparators[0] );
            search_comp.SetComparators( &m_searchComparator );
            m_btree->set_search_comp( search_comp );
        }

        // template< typename T = type_t, typename std::enable_if< std::is_same< T, KeyPosition >::value, T >::type* = nullptr >
        // AriesIndexBtree( const AriesDictKeyComparator& comp )
        // {
        //     m_btree = std::make_unique < AriesBreeMap > ( comp );
        // }

        AriesIndexBtree( const aries::AriesColumnType& type );

        ~AriesIndexBtree();

    public:

        bool Save( const std::string& filePath ) const;

        template< typename T = type_t, typename std::enable_if< std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        bool Load( const std::string& filePath )
        {
            bool bRet = false;
            Clear();
            std::ifstream in( filePath, std::ios::binary | std::ios::in );
            if( in.is_open() )
            {
                iarchive ia( in );
                assert( m_columnTypes.size() > 1 );

                m_comparators = CreateComparators( m_columnTypes );
                AriesCompositeKeyCompartor comp;
                comp.SetComparators( &m_comparators );
                m_btree = std::make_unique < AriesBreeMap > ( comp );
                m_btree->load( ia );
                bRet = true;
            }
            return bRet;
        }

        // template< typename T = type_t, typename std::enable_if< std::is_same< T, KeyPosition >::value, T >::type * = nullptr >
        // bool Load( const std::string &filePath )
        // {
        //     bool bRet = false;
        //     std::ifstream in( filePath, std::ios::binary | std::ios::in );
        //     if ( in.is_open() )
        //     {
        //         iarchive ia( in );
        //         m_btree->load( ia );
        //         bRet = true;
        //     }
        //     return bRet;
        // }

        // template< typename T = type_t, typename std::enable_if< !std::is_same< T, AriesCompositeKeyType >::value && !std::is_same< T, KeyPosition >::value, T >::type* = nullptr >
        template< typename T = type_t, typename std::enable_if< !std::is_same< T, AriesCompositeKeyType >::value , T >::type* = nullptr >
        bool Load( const std::string& filePath )
        {
            bool bRet = false;
            Clear();
            std::ifstream in( filePath, std::ios::binary | std::ios::in );
            if( in.is_open() )
            {
                iarchive ia( in );
                m_btree = std::make_unique< AriesBreeMap >();
                m_btree->load( ia );
                bRet = true;
            }
            return bRet;
        }

    public:
        // Iterator routines.
        iterator begin()
        {
            return m_btree->begin();
        }
        const_iterator begin() const
        {
            return m_btree->begin();
        }
        iterator end()
        {
            return m_btree->end();
        }
        const_iterator end() const
        {
            return m_btree->end();
        }
        reverse_iterator rbegin()
        {
            return m_btree->rbegin();
        }
        const_reverse_iterator rbegin() const
        {
            return m_btree->rbegin();
        }
        reverse_iterator rend()
        {
            return m_btree->rend();
        }
        const_reverse_iterator rend() const
        {
            return m_btree->rend();
        }

        // Lookup routines.
        iterator find( const key_type &key )
        {
            return m_btree->find( key );
        }
        const_iterator find( const key_type &key ) const
        {
            return m_btree->find( key );
        }

        iterator find_first_key( const key_type &key )
        {
            return m_btree->find_first_key( key );
        }
        const_iterator find_first_key( const key_type &key ) const
        {
            return m_btree->find_first_key( key );
        }

        size_type count( const key_type &key ) const
        {
            return m_btree->count( key );
        }

        iterator lower_bound( const key_type &key )
        {
            return m_btree->lower_bound( key );
        }
        const_iterator lower_bound( const key_type &key ) const
        {
            return m_btree->lower_bound( key );
        }

        iterator lower_bound_first_key( const key_type &key )
        {
            return m_btree->lower_bound_first_key( key );
        }
        const_iterator lower_bound_first_key( const key_type &key ) const
        {
            return m_btree->lower_bound_first_key( key );
        }

        iterator upper_bound( const key_type &key )
        {
            return m_btree->upper_bound( key );
        }
        const_iterator upper_bound( const key_type &key ) const
        {
            return m_btree->upper_bound( key );
        }

        iterator upper_bound_first_key( const key_type &key )
        {
            return m_btree->upper_bound_first_key( key );
        }
        const_iterator upper_bound_first_key( const key_type &key ) const
        {
            return m_btree->upper_bound_first_key( key );
        }

        std::pair< iterator, iterator > equal_range( const key_type &key )
        {
            return m_btree->equal_range( key );
        }
        std::pair< const_iterator, const_iterator > equal_range( const key_type &key ) const
        {
            return m_btree->equal_range( key );
        }

        std::pair< iterator, iterator > equal_range_first_key( const key_type &key )
        {
            return m_btree->equal_range_first_key( key );
        }
        std::pair< const_iterator, const_iterator > equal_range_first_key( const key_type &key ) const
        {
            return m_btree->equal_range_first_key( key );
        }

        // Insertion routines.
        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_duplicate, T >::type* = nullptr >
        std::pair< iterator, bool > insert( const value_type &x )
        {
            return m_btree->insert( x );
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_duplicate, T >::type * = nullptr >
        std::pair< iterator, bool > insert( const T& key, const value_type &x )
        {
            return m_btree->insert( key, x );
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique, T >::type* = nullptr >
        iterator insert( const value_type &x )
        {
            return m_btree->insert( x );
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique, T >::type * = nullptr >
        iterator insert_unique( const T &key, const value_type &x )
        {
            return m_btree->insert( key, x );
        }

        iterator insert( iterator position, const value_type &x )
        {
            return m_btree->insert( position, x );
        }

        template< typename InputIterator >
        void insert( InputIterator b, InputIterator e )
        {
            m_btree->insert( b, e );
        }

        // Deletion routines.
        int erase( const key_type &key )
        {
            return m_btree->erase( key );
        }
        // Erase the specified iterator from the btree. The iterator must be valid
        // (i.e. not equal to end()).  Return an iterator pointing to the node after
        // the one that was erased (or end() if none exists).
        iterator erase( const iterator &iter )
        {
            return m_btree->erase( iter );
        }
        void erase( const iterator &first, const iterator &last )
        {
            m_btree->erase( first, last );
        }

        // Utility routines.
        void clear()
        {
            m_btree->clear();
        }
        void verify() const
        {
            m_btree->verify();
        }

        // Size routines.
        size_type size() const
        {
            return m_btree->size();
        }
        size_type max_size() const
        {
            return m_btree->max_size();
        }
        bool empty() const
        {
            return m_btree->empty();
        }
        size_type height() const
        {
            return m_btree->height();
        }
        size_type internal_nodes() const
        {
            return m_btree->internal_nodes();
        }
        size_type leaf_nodes() const
        {
            return m_btree->leaf_nodes();
        }
        size_type nodes() const
        {
            return m_btree->nodes();
        }
        size_type bytes_used() const
        {
            return m_btree->bytes_used();
        }
        static double average_bytes_per_value()
        {
            return AriesBreeMap::average_bytes_per_value();
        }
        double fullness() const
        {
            return m_btree->fullness();
        }
        double overhead() const
        {
            return m_btree->overhead();
        }

    private:
        vector< shared_ptr< IAriesCompositeCompareTo > > CreateComparators( const std::vector< aries::AriesColumnType >& types );
        void Clear();

    private:
        std::vector< aries::AriesColumnType > m_columnTypes;
        vector< shared_ptr< IAriesCompositeCompareTo > > m_comparators;
        vector< shared_ptr< IAriesCompositeCompareTo > > m_searchComparator;
        unique_ptr< AriesBreeMap > m_btree;
    };

    template< typename T, key_hint_t K >
    AriesIndexBtree< T, K >::AriesIndexBtree( const aries::AriesColumnType& type )
    {
        m_columnTypes.push_back( type );
        m_btree = std::make_unique< AriesBreeMap >();
    }

    template< typename T, key_hint_t K >
    AriesIndexBtree< T, K >::~AriesIndexBtree()
    {
        // TODO Auto-generated destructor stub
    }

    template< typename T, key_hint_t K >
    void AriesIndexBtree< T, K >::Clear()
    {
        m_columnTypes.clear();
        m_comparators.clear();
        m_btree->clear();
    }

    template< typename T, key_hint_t K >
    bool AriesIndexBtree< T, K >::Save( const std::string& filePath ) const
    {
        assert( m_btree );
        bool bRet = false;
        std::ofstream out( filePath, std::ios::binary | std::ios::trunc | std::ios::out );
        if( out.is_open() )
        {
            oarchive oa( out );
            m_btree->save( oa );
            bRet = true;
        }

        return bRet;
    }

    template< typename T, key_hint_t K >
    vector< shared_ptr< IAriesCompositeCompareTo > > AriesIndexBtree< T, K >::CreateComparators(
            const std::vector< aries::AriesColumnType >& types )
    {
        assert( !types.empty() );
        vector< shared_ptr< IAriesCompositeCompareTo > > result;
        for( const auto& type : types )
        {
            switch( type.DataType.ValueType )
            {
                case aries::AriesValueType::INT8:
                    result.push_back( std::make_shared< AriesCompositeComparator< int8_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::UINT8:
                    result.push_back( std::make_shared< AriesCompositeComparator< uint8_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::BOOL:
                    result.push_back( std::make_shared< AriesCompositeComparator< bool > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::CHAR:
                    if( type.DataType.Length > 1 )
                        result.push_back( std::make_shared< AriesCompositeStringComparator >( type.HasNull, type.DataType.Length ) );
                    else
                        result.push_back( std::make_shared< AriesCompositeComparator< char > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::COMPACT_DECIMAL:
                    //TODO:
                    assert( 0 );
                    break;
                case aries::AriesValueType::INT16:
                    result.push_back( std::make_shared< AriesCompositeComparator< int16_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::UINT16:
                    result.push_back( std::make_shared< AriesCompositeComparator< uint16_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::INT32:
                    result.push_back( std::make_shared< AriesCompositeComparator< int32_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::UINT32:
                    result.push_back( std::make_shared< AriesCompositeComparator< uint32_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::INT64:
                    result.push_back( std::make_shared< AriesCompositeComparator< int64_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::UINT64:
                    result.push_back( std::make_shared< AriesCompositeComparator< uint64_t > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::FLOAT:
                    result.push_back( std::make_shared< AriesCompositeComparator< float > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::DOUBLE:
                    result.push_back( std::make_shared< AriesCompositeComparator< double > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::DECIMAL:
                    result.push_back( std::make_shared< AriesCompositeComparator< aries_acc::Decimal > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::DATE:
                    result.push_back( std::make_shared< AriesCompositeComparator< aries_acc::AriesDate > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::DATETIME:
                    result.push_back( std::make_shared< AriesCompositeComparator< aries_acc::AriesDatetime > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::TIMESTAMP:
                    result.push_back( std::make_shared< AriesCompositeComparator< aries_acc::AriesTimestamp > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::TIME:
                    result.push_back( std::make_shared< AriesCompositeComparator< aries_acc::AriesTime > >( type.HasNull ) );
                    break;
                case aries::AriesValueType::YEAR:
                    result.push_back( std::make_shared< AriesCompositeComparator< aries_acc::AriesYear > >( type.HasNull ) );
                    break;
                default:
                    assert( 0 );
                    break;
            }
        }
        return result;
    }

    template< typename type_t, key_hint_t hint_t = key_hint_t::most_unique >
    class AriesIndex: public IAriesIndex
    {
        using BTree = AriesIndexBtree< type_t, hint_t >;
        using value_type = typename BTree::value_type;

    public:
        AriesIndex( const std::vector< aries::AriesColumnType >& types )
                : m_btree( types )
        {
        }

        AriesIndex( const aries::AriesColumnType& type )
                : m_btree( type )
        {
        }

        // AriesIndex( const AriesDictKeyComparator& comp )
        //         : m_btree( comp )
        // {
        // }

        virtual ~AriesIndex()
        {
        }

    public:
        virtual void Lock()
        {
            m_mutex.lock();
        }

        virtual void Unlock()
        {
            m_mutex.unlock();
        }

        virtual bool KeyExists( const void* key ) const override final
        {
            return m_btree.find( *reinterpret_cast< const type_t* >( key ) ) != m_btree.end();
        }

        virtual std::vector< AriesTupleLocation > FindEqual( const void* key ) const override final
        {
            return FindEqualInternal( *reinterpret_cast< const type_t* >( key ) );
        }

        virtual std::vector< AriesTupleLocation > FindNotEqual( const void* key ) const override final
        {
            return FindNotEqualInternal( *reinterpret_cast< const type_t* >( key ) );
        }

        virtual std::vector< AriesTupleLocation > FindLess( const void* key ) const override final
        {
            return FindLessInternal( *reinterpret_cast< const type_t* >( key ) );
        }

        virtual std::vector< AriesTupleLocation > FindLessOrEqual( const void* key ) const override final
        {
            return FindLessOrEqualInternal( *reinterpret_cast< const type_t* >( key ) );
        }

        virtual std::vector< AriesTupleLocation > FindGreater( const void* key ) const override final
        {
            return FindGreaterInternal( *reinterpret_cast< const type_t* >( key ) );
        }

        virtual std::vector< AriesTupleLocation > FindGreaterOrEqual( const void* key ) const override final
        {
            return FindGreaterOrEqualInternal( *reinterpret_cast< const type_t* >( key ) );
        }

        virtual std::vector< AriesTupleLocation > FindGTAndLT( const void* minKey, const void* maxKey ) const override final
        {
            return FindGTAndLTInternal( *reinterpret_cast< const type_t* >( minKey ), *reinterpret_cast< const type_t* >( maxKey ) );
        }

        virtual std::vector< AriesTupleLocation > FindGEAndLT( const void* minKey, const void* maxKey ) const override final
        {
            return FindGEAndLTInternal( *reinterpret_cast< const type_t* >( minKey ), *reinterpret_cast< const type_t* >( maxKey ) );
        }

        virtual std::vector< AriesTupleLocation > FindGTAndLE( const void* minKey, const void* maxKey ) const override final
        {
            return FindGTAndLEInternal( *reinterpret_cast< const type_t* >( minKey ), *reinterpret_cast< const type_t* >( maxKey ) );
        }

        virtual std::vector< AriesTupleLocation > FindGEAndLE( const void* minKey, const void* maxKey ) const override final
        {
            return FindGEAndLEInternal( *reinterpret_cast< const type_t* >( minKey ), *reinterpret_cast< const type_t* >( maxKey ) );
        }

        virtual bool Save( const std::string& filePath ) const override final
        {
            return m_btree.Save( filePath );
        }

        virtual bool Load( const std::string& filePath )
        {
            return m_btree.Load( filePath );
        }

        virtual std::pair< void*, bool > Insert( const void* key, AriesTupleLocation value )
        {
            return InsertInternal( *reinterpret_cast< const type_t* >( key ), value );
        }

        virtual bool Erase( const void* key, AriesTupleLocation value )
        {
            return EraseInternal( *reinterpret_cast< const type_t* >( key ), value );
        }

        virtual void DumpStatus() const
        {
#ifndef NDEBUG
            cout<<"-------------------------AriesIndices status begin----------------------------"<<endl;
            cout<<"using vector:"<<( hint_t == key_hint_t::most_duplicate ? "yes" : "no" )<<endl;
            cout<<"key count:"<<m_btree.size()<<endl;
            cout<<"height:"<<m_btree.height()<<endl;
            cout<<"internal_nodes:"<<m_btree.internal_nodes()<<endl;
            cout<<"leaf_nodes:"<<m_btree.leaf_nodes()<<endl;
            cout<<"nodes:"<<m_btree.nodes()<<endl;
            cout<<"bytes_used:"<<m_btree.bytes_used() / ( 1023.0 * 1024.0 )<<"mb"<<endl;
            cout<<"average_bytes_per_value:"<<m_btree.average_bytes_per_value()<<endl;
            cout<<"--------------------------AriesIndices status end-----------------------------"<<endl;
#endif
        }

    private:
        template< typename T >
        bool IsNotNull( const T& data ) const
        {
            return true;
        }

        template< typename T >
        bool IsNotNull( const nullable_type< T >& data ) const
        {
            return data.flag;
        }

        template< typename T = type_t,
                typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value >::type* = nullptr >
        std::vector< AriesTupleLocation > FindEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto eqRange = m_btree.equal_range_first_key( key );
            for( auto it = eqRange.first; it != eqRange.second; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value >::type* = nullptr >
        std::vector< AriesTupleLocation > FindEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto eqRange = m_btree.equal_range_first_key( key );
            for( auto it = eqRange.first; it != eqRange.second; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindEqualInternal( const T& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto eqRange = m_btree.equal_range( key );
            for( auto it = eqRange.first; it != eqRange.second; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindEqualInternal( const T& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto it = m_btree.find( key );
            if( it != m_btree.end() )
                result.assign( it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindNotEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto eqRange = m_btree.equal_range_first_key( key );
            for( auto it = m_btree.begin(); it != eqRange.first; ++it )
                result.push_back( it->second );
            for( auto it = eqRange.second; it != m_btree.end(); ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindNotEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto eqRange = m_btree.equal_range_first_key( key );
            for( auto it = m_btree.begin(); it != eqRange.first; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );
            for( auto it = eqRange.second; it != m_btree.end(); ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindNotEqualInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto eqRange = m_btree.equal_range( key );
            for( auto it = m_btree.begin(); it != eqRange.first; ++it )
                result.push_back( it->second );
            for( auto it = eqRange.second; it != m_btree.end(); ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindNotEqualInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.find( key );
            for( auto it = m_btree.begin(); it != m_btree.end(); ++it )
            {
                if( it != itKey )
                    result.insert( result.end(), it->second.begin(), it->second.end() );
            }

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound_first_key( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
                if( IsNotNull( it->first ) )
                    result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound_first_key( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
            {
                if( IsNotNull( it->first ) )
                    result.insert( result.end(), it->second.begin(), it->second.end() );
            }

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
                if( IsNotNull( it->first ) )
                    result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
            {
                if( IsNotNull( it->first ) )
                    result.insert( result.end(), it->second.begin(), it->second.end() );
            }

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessOrEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound_first_key( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessOrEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound_first_key( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessOrEqualInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindLessOrEqualInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound( key );
            for( auto it = m_btree.begin(); it != itKey; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound_first_key( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.push_back( itKey->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound_first_key( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.insert( result.end(), itKey->second.begin(), itKey->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.push_back( itKey->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.upper_bound( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.insert( result.end(), itKey->second.begin(), itKey->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterOrEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound_first_key( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.push_back( itKey->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterOrEqualInternal( const AriesCompositeKeyType& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound_first_key( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.insert( result.end(), itKey->second.begin(), itKey->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterOrEqualInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.push_back( itKey->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGreaterOrEqualInternal( const type_t& key ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKey = m_btree.lower_bound( key );
            for( ; itKey != m_btree.end(); ++itKey )
                result.insert( result.end(), itKey->second.begin(), itKey->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLTInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound_first_key( keyMin );
            auto itKeyMax = m_btree.lower_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLTInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound_first_key( keyMin );
            auto itKeyMax = m_btree.lower_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLTInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound( keyMin );
            auto itKeyMax = m_btree.lower_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLTInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound( keyMin );
            auto itKeyMax = m_btree.lower_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLTInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound_first_key( keyMin );
            auto itKeyMax = m_btree.lower_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLTInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {

            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound_first_key( keyMin );
            auto itKeyMax = m_btree.lower_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLTInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound( keyMin );
            auto itKeyMax = m_btree.lower_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLTInternal( const type_t& keyMin, const type_t& keyMax ) const
        {

            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound( keyMin );
            auto itKeyMax = m_btree.lower_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLEInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound_first_key( keyMin );
            auto itKeyMax = m_btree.upper_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLEInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound_first_key( keyMin );
            auto itKeyMax = m_btree.upper_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLEInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound( keyMin );
            auto itKeyMax = m_btree.upper_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGTAndLEInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.upper_bound( keyMin );
            auto itKeyMax = m_btree.upper_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLEInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound_first_key( keyMin );
            auto itKeyMax = m_btree.upper_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLEInternal( const AriesCompositeKeyType& keyMin, const AriesCompositeKeyType& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound_first_key( keyMin );
            auto itKeyMax = m_btree.upper_bound_first_key( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique && !std::is_same< T, AriesCompositeKeyType >::value,
                T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLEInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound( keyMin );
            auto itKeyMax = m_btree.upper_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.push_back( it->second );

            return result;
        }

        template< typename T = type_t, typename std::enable_if<
                hint_t == key_hint_t::most_duplicate && !std::is_same< T, AriesCompositeKeyType >::value, T >::type* = nullptr >
        std::vector< AriesTupleLocation > FindGEAndLEInternal( const type_t& keyMin, const type_t& keyMax ) const
        {
            std::vector< AriesTupleLocation > result;
            auto itKeyMin = m_btree.lower_bound( keyMin );
            auto itKeyMax = m_btree.upper_bound( keyMax );
            for( auto it = itKeyMin; it != itKeyMax; ++it )
                result.insert( result.end(), it->second.begin(), it->second.end() );

            return result;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_duplicate, T >::type* = nullptr >
        std::pair< void*, bool > InsertInternal( const T& key, AriesTupleLocation value )
        {
            auto it = m_btree.insert( key, { key, vector< AriesTupleLocation >() } );
            it.first->second.push_back( value );
            return { (void*)&it.first->first, it.second };
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique, T >::type* = nullptr >
        std::pair< void*, bool > InsertInternal( const T& key, AriesTupleLocation value )
        {
            auto it = m_btree.insert_unique( key, { key, value } );
            return { (void*)(it.operator->()), true };
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_unique, T >::type* = nullptr >
        bool EraseInternal( const T& key, AriesTupleLocation value )
        {
            bool bFound = false;
            auto itKeyMin = m_btree.lower_bound( key );
            auto itKeyMax = m_btree.upper_bound( key );
            //just liner search for now
            for( auto it = itKeyMin; it != itKeyMax; ++it )
            {
                if( it->second == value )
                {
                    bFound = true;
                    m_btree.erase( it );
                    break;
                }
            }
            return bFound;
        }

        template< typename T = type_t, typename std::enable_if< hint_t == key_hint_t::most_duplicate, T >::type* = nullptr >
        bool EraseInternal( const T& key, AriesTupleLocation value )
        {
            bool bFound = false;
            auto itKey = m_btree.find( key );
            if( itKey != m_btree.end() )
            {
                for( auto it = itKey->second.begin(); it != itKey->second.end(); ++it )
                {
                    if( *it == value )
                    {
                        itKey->second.erase( it );
                        bFound = true;
                        break;
                    }
                }
            }
            return bFound;
        }

    private:
        BTree m_btree;
        std::mutex m_mutex;
    };

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESINDEXBTREE_H_ */
