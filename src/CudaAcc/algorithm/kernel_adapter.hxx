/*
 * kernel_adapter.hxx
 *
 *  Created on: Jun 19, 2019
 *      Author: lichi
 */

#ifndef KERNEL_ADAPTER_HXX_
#define KERNEL_ADAPTER_HXX_
#include "operators.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    struct IComparableColumnPair
    {
        ARIES_HOST_DEVICE virtual AriesBool compare( int leftIndex, int rightIndex ) const
        {
            return AriesBool::ValueType::Unknown;
        }
        ARIES_HOST_DEVICE virtual ~IComparableColumnPair()
        {
        }
    };

    template< typename type_t, typename type_u, typename comp_t >
    struct ComparableColumnPair: public IComparableColumnPair
    {
        ARIES_HOST_DEVICE ComparableColumnPair( const type_t* leftData, const type_u* rightData, comp_t op )
                : leftColumnData( leftData ), rightColumnData( rightData ), comp( op )
        {
        }

        ARIES_HOST_DEVICE virtual AriesBool compare( int leftIndex, int rightIndex ) const
        {
            return comp( leftColumnData[ leftIndex ], rightColumnData[ rightIndex ] );
        }

    private:
        const type_t* leftColumnData;
        const type_u* rightColumnData;
        comp_t comp;
    };

    // for char type
    template< typename comp_t >
    struct ComparableColumnPair< char, char, comp_t > : public IComparableColumnPair
    {
        ARIES_HOST_DEVICE ComparableColumnPair( const char* leftData, const char* rightData, int len, comp_t op )
                : leftColumnData( leftData ), rightColumnData( rightData ), length( len ), comp( op )
        {
        }

        ARIES_HOST_DEVICE virtual AriesBool compare( int leftIndex, int rightIndex ) const
        {
            return comp( leftColumnData + leftIndex * length, rightColumnData + rightIndex * length, length );
        }

    public:
        const char* leftColumnData;
        const char* rightColumnData;
        size_t length;
        comp_t comp;
    };

    struct IComparableColumn
    {
        ARIES_HOST_DEVICE virtual AriesBool compare( int aIndex, int bIndex ) const
        {
            return AriesBool::ValueType::Unknown;
        }
        ARIES_HOST_DEVICE virtual ~IComparableColumn()
        {
        }
    };

    template< typename type_t, typename comp_t, bool bNumeric = true >
    struct ComparableColumn: public IComparableColumn
    {
        ARIES_HOST_DEVICE ComparableColumn( const type_t* data, comp_t op )
                : columnData( data ), comp( op )
        {
        }

        ARIES_HOST_DEVICE virtual AriesBool compare( int aIndex, int bIndex ) const
        {
            return comp( columnData[ aIndex ], columnData[ bIndex ] );
        }

    private:
        const type_t* columnData;
        comp_t comp;
    };

    // for char type
    template< typename comp_t >
    struct ComparableColumn< char, comp_t, false > : public IComparableColumn
    {
        ARIES_HOST_DEVICE ComparableColumn( const char* data, int len, comp_t op )
                : columnData( data ), length( len ), comp( op )
        {
        }

        ARIES_HOST_DEVICE virtual AriesBool compare( int aIndex, int bIndex ) const
        {
            return comp( columnData + aIndex * length, columnData + bIndex * length, length );
        }

    private:
        const char* columnData;
        size_t length;
        comp_t comp;
    };

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_ADAPTER_HXX_ */
