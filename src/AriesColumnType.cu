#include "AriesColumnType.h"
#include "AriesDefinition.h"

#include "datatypes/decimal.hxx"
#include "datatypes/AriesDate.hxx"
#include "datatypes/AriesDatetime.hxx"
#include "datatypes/AriesTime.hxx"
#include "datatypes/AriesTimestamp.hxx"
#include "datatypes/AriesYear.hxx"

using namespace aries_acc;
NAMESPACE_ARIES_START

ARIES_HOST_DEVICE_NO_INLINE uint64_t AriesColumnType::GetDataTypeSize() const
{
    uint64_t size = 0;
    switch( DataType.ValueType )
    {
        case AriesValueType::INT8:
        case AriesValueType::UINT8:
        case AriesValueType::BOOL:
        case AriesValueType::CHAR:
        case AriesValueType::COMPACT_DECIMAL:
            size = sizeof(int8_t) * DataType.Length;
            break;
        case AriesValueType::INT16:
        case AriesValueType::UINT16:
            size = sizeof(int16_t) * DataType.Length;
            break;
        case AriesValueType::INT32:
        case AriesValueType::UINT32:
            size = sizeof(int32_t) * DataType.Length;
            break;
        case AriesValueType::INT64:
        case AriesValueType::UINT64:
            size = sizeof(int64_t) * DataType.Length;
            break;
        case AriesValueType::FLOAT:
            size = sizeof(float) * DataType.Length;
            break;
        case AriesValueType::DOUBLE:
            size = sizeof(double) * DataType.Length;
            break;
        case AriesValueType::DECIMAL:
            size = sizeof(aries_acc::Decimal) * DataType.Length;
            break;
        case AriesValueType::ARIES_DECIMAL:
            size = 4 * DataType.Length + 4; // AriesDecimal 是由 DataType.Length 个 32 数组 加上 4 个字节的属性位 组成
            break;
        case AriesValueType::DATE:
            size = sizeof(AriesDate) * DataType.Length;
            break;
        case AriesValueType::DATETIME:
            size = sizeof(AriesDatetime) * DataType.Length;
            break;
        case AriesValueType::TIME:
            size = sizeof(AriesTime) * DataType.Length;
            break;
        case AriesValueType::TIMESTAMP:
            size = sizeof(AriesTimestamp) * DataType.Length;
            break;
        case AriesValueType::YEAR:
            size = sizeof(AriesYear) * DataType.Length;
            break;
        case AriesValueType::UNKNOWN:
            size = 1;
            break;
        default:
            // assert( 0 );     //FIXME need support all data types
            break;
    }
    return HasNull ? size + 1 : size;     // use one extra byte for 'null' flag
}

NAMESPACE_ARIES_END