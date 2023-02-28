#ifndef ARIESDATATYPE_H_
#define ARIESDATATYPE_H_
#include "AriesDefinition.h"

NAMESPACE_ARIES_START

    struct AriesDataType
    {
        AriesValueType ValueType = AriesValueType::UNKNOWN;
        int32_t Length = 1;

        // for decimal
        uint16_t Precision = 0;
        uint16_t Scale = 0;
        // for Adaptive Computing
        uint32_t AdaptiveLen = 0;

        ARIES_HOST_DEVICE AriesDataType()
        {
        }

        ARIES_HOST_DEVICE AriesDataType( AriesValueType valueType, int32_t length )
                : ValueType( valueType ), Length( length )
        {
        }

        ARIES_HOST_DEVICE_NO_INLINE AriesDataType( AriesValueType valueType, uint16_t precision, uint16_t scale );

        ARIES_HOST_DEVICE AriesDataType( AriesValueType valueType )
                : ValueType( valueType )
        {
        }

        ARIES_HOST_DEVICE bool operator==( const AriesDataType& src )
        {
            return ValueType == src.ValueType && Length == src.Length && Precision == src.Precision && Scale == src.Scale;
        }

        ARIES_HOST_DEVICE bool operator!=( const AriesDataType& src )
        {
            return !( *this == src );
        }
    };

NAMESPACE_ARIES_END

#endif //ARIESDATATYPE_H_