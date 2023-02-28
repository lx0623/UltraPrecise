/*
 * AriesKeyComparator.h
 *
 *  Created on: Apr 21, 2020
 *      Author: lichi
 */

#ifndef ARIESCOMPOSITEKEYCOMPARATOR_H_
#define ARIESCOMPOSITEKEYCOMPARATOR_H_
#include "btree_map.h"
#include "../AriesIndex.h"
using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE

    struct IAriesCompositeCompareTo: public btree::btree_key_compare_to_tag
    {
        virtual int CompareTo( const char* key1, const char* key2 ) const = 0;
        virtual ~IAriesCompositeCompareTo()
        {
        }
        int CompareFlag( char flag1, char flag2 ) const
        {
            assert( !( flag1 && flag2 ) );
            return flag1 ? 1 : flag2 ? -1 : 0;
        }
        bool m_bHasNull;
        int m_itemSizeInBytes;
    };

    template< typename type_t >
    struct AriesCompositeComparator: public IAriesCompositeCompareTo
    {
        AriesCompositeComparator( bool hasNull )
        {
            m_bHasNull = hasNull;
            m_itemSizeInBytes = sizeof(type_t) + hasNull;
        }
        virtual int CompareTo( const char* key1, const char* key2 ) const override final
        {
            if( m_bHasNull )
            {
                if( !( *key1 && *key2 ) )
                    return CompareFlag( *key1, *key2 );
                ++key1;
                ++key2;
            }
            const type_t& k1 = *( const type_t* )key1;
            const type_t& k2 = *( const type_t* )key2;
            if( k1 < k2 )
                return -1;
            else if( k1 == k2 )
                return 0;
            else
                return 1;
        }
    };

    struct AriesCompositeStringComparator: public IAriesCompositeCompareTo
    {
        AriesCompositeStringComparator( bool hasNull, int len )
        {
            m_bHasNull = hasNull;
            m_itemSizeInBytes = len;
        }
        virtual int CompareTo( const char* key1, const char* key2 ) const override final
        {
            int len = m_itemSizeInBytes;
            if( m_bHasNull )
            {
                if( !( *key1 && *key2 ) )
                    return CompareFlag( *key1, *key2 );
                ++key1;
                ++key2;
                --len;
            }
            return std::strncmp( key1, key2, len );
        }
    };

    struct AriesCompositeKeyCompartor: public btree::btree_key_compare_to_tag
    {
        AriesCompositeKeyCompartor()
                : m_comps( nullptr )
        {
        }

        void SetComparators( vector< shared_ptr< IAriesCompositeCompareTo > >* comparators )
        {
            m_comps = comparators;
        }

        int operator()( const AriesCompositeKeyType &a, const AriesCompositeKeyType &b ) const
        {
            assert( m_comps );
            const char* key1 = a.m_key.data();
            const char* key2 = b.m_key.data();
            int res;
            for( const auto& comp : *m_comps )
            {
                res = comp->CompareTo( key1, key2 );
                if( res == 0 )
                {
                    int offset = comp->m_itemSizeInBytes;
                    key1 += offset;
                    key2 += offset;
                }
                else
                    break;
            }
            return res;
        }

        vector< shared_ptr< IAriesCompositeCompareTo > >* m_comps;
    };

END_ARIES_ENGINE_NAMESPACE

#endif
