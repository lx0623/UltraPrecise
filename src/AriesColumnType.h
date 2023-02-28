/*
 * AriesColumnType.h
 *
 *  Created on: Jun 15, 2019
 *      Author: lichi
 */

#ifndef ARIESCOLUMNTYPE_H_
#define ARIESCOLUMNTYPE_H_
#include "AriesDefinition.h"
#include "AriesDataType.h"

NAMESPACE_ARIES_START

    struct AriesColumnType
    {
        AriesDataType DataType;
        bool HasNull;
        bool IsUnique;
        ARIES_HOST_DEVICE AriesColumnType()
                : AriesColumnType( AriesDataType(), false, false )
        {

        }
        ARIES_HOST_DEVICE AriesColumnType( AriesDataType dataType, bool hasNull, bool isUnique = false )
                : DataType( dataType ), HasNull( hasNull ), IsUnique( isUnique )
        {

        }

        ARIES_HOST_DEVICE bool isNullable() const
        {
            return HasNull;
        }

        ARIES_HOST_DEVICE bool isUnique() const
        {
            return IsUnique;
        }

        ARIES_HOST_DEVICE bool operator==( const AriesColumnType& src )
        {
            return DataType == src.DataType && HasNull == src.HasNull;
        }

        ARIES_HOST_DEVICE_NO_INLINE uint64_t GetDataTypeSize() const;
        
    };


NAMESPACE_ARIES_END

#endif /* ARIESCOLUMNTYPE_H_ */
