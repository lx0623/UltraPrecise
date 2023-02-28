#include "AriesDataType.h"
#include "AriesDefinition.h"
#include "decimal.hxx"

NAMESPACE_ARIES_START

ARIES_HOST_DEVICE_NO_INLINE AriesDataType::AriesDataType( AriesValueType valueType, uint16_t precision, uint16_t scale )
        : ValueType( valueType ), Precision( precision ), Scale( scale )
{
    if ( AriesValueType::COMPACT_DECIMAL == valueType )
        Length = aries_acc::GetDecimalRealBytes(precision, scale);
}

NAMESPACE_ARIES_END