/*
 * AriesSimpleKeyComparator.h
 *
 *  Created on: Apr 22, 2020
 *      Author: lichi
 */

#ifndef ARIESSIMPLEKEYCOMPARATOR_H_
#define ARIESSIMPLEKEYCOMPARATOR_H_

#include "btree_map.h"
#include "../AriesIndex.h"
#include "aries_types.hxx"
using namespace std;
using aries_acc::nullable_type;
BEGIN_ARIES_ENGINE_NAMESPACE

    template< typename type_t >
    struct AriesSimpleKeyCompartor: public btree::btree_key_compare_to_tag
    {
        AriesSimpleKeyCompartor()
        {
        }

        int operator()( const type_t &a, const type_t &b ) const
        {
            if( a < b )
                return -1;
            else if( a == b )
                return 0;
            else
                return 1;
        }
    };

    template< typename type_t >
    struct AriesSimpleKeyCompartor< nullable_type< type_t > > : public btree::btree_key_compare_to_tag
    {
        AriesSimpleKeyCompartor()
        {
        }

        int operator()( const nullable_type< type_t > &a, const nullable_type< type_t > &b ) const
        {
            if( a.flag && b.flag )
            {
                if( a.value < b.value )
                    return -1;
                else if( a.value == b.value )
                    return 0;
                else
                    return 1;
            }
            else
            {
                if( a.flag )
                    return 1;
                else if( b.flag )
                    return -1;
                else
                    return 0;
            }
        }
    };

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESSIMPLEKEYCOMPARATOR_H_ */
