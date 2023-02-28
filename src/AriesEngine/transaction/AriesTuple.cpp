//
// Created by david.shen on 2020/3/16.
//

#include "AriesTuple.h"
#include "../AriesUtil.h"
#include "AriesInitialTable.h"
#include "datatypes/AriesDate.hxx"
#include "datatypes/AriesDatetimeTrans.h"
#include "utils/utils.h"

extern bool STRICT_MODE;

BEGIN_ARIES_ENGINE_NAMESPACE

    using Decimal = aries_acc::Decimal;
    TupleParser::TupleParser(TableEntrySPtr &tableEntry)
    {
        m_tupleSize = 0;
        ParsTableColumnInfo(tableEntry);
        if (m_tupleSize <= 0) {
            ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, "parse TableEntry error, table: " + tableEntry->GetName() );
        }
    }

    void TupleParser::FillData( const std::vector< int8_t* >& columnBuffers, TupleDataSPtr dataBuffer, int dataIndex )
    {
        assert( columnBuffers.size() == dataBuffer->data.size() );
        if (m_tupleSize <= 0)
            ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, "tupleSize error: " + to_string(m_tupleSize) );

        for (auto it = dataBuffer->data.begin(); it != dataBuffer->data.end(); ++it)
        {
            int columnIndex = it->first - 1;
            auto &targetType = m_columnTypes[columnIndex];
            auto sourceType = it->second->GetDataType();
            auto sourceData = it->second->GetItemDataAt(dataIndex);
            auto targetDataBuf = columnBuffers[ columnIndex ];
            if (targetType == sourceType)
                memcpy(targetDataBuf, sourceData, sourceType.GetDataTypeSize());
            else
            {
                AriesDataBuffer newSource( targetType, it->second->GetItemCount() );
                if( TransferColumnData( m_columnName[columnIndex], newSource, *it->second ) )
                    memcpy( targetDataBuf, newSource.GetItemDataAt( dataIndex ), targetType.GetDataTypeSize() );
                else
                    ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + m_columnName[columnIndex] + ") from " + GenerateParamType(sourceType) + " to " + GenerateParamType(targetType));
            }
        }
    }

    AriesColumnType & TupleParser::GetColumnType( int colId )
    {
        return m_columnTypes[colId - 1];
    }

    void TupleParser::ParsTableColumnInfo( TableEntrySPtr &tableEntry )
    {
        auto colCount = tableEntry->GetColumnsCount();
        // convert to byte
        int tupleDataLen = 0;
        for (size_t i = 0; i < colCount; ++i)
        {
            auto column = tableEntry->GetColumnById(i + 1);
            auto colType = column->GetType();
            //TODO 需要只按照column的schema处理列信息, 目前按照列文件里的实际信息设置column信息
            AriesColumnType dataType;
            if ( aries::EncodeType::DICT == column->encode_type )
            {
                dataType = CovertToAriesColumnType( column->GetDictIndexDataType(), 1, column->IsAllowNull() );
            }
            else
            {
                string filePath = tableEntry->GetColumnLocationString_ByIndex(i);
                int length = column->GetLength();
                shared_ptr<ifstream> colFile = make_shared< ifstream >( filePath + "_0");
                if( colFile->is_open() )
                {
                    // check if header info is valid
                    BlockFileHeader headerInfo;
                    int res = GetBlockFileHeaderInfo(colFile, headerInfo);
                    colFile->close();

                    ARIES_ASSERT(res == 0, "get header info failed, table: " + tableEntry->GetName() + ", column: " + to_string(i));

                    int8_t containNull = headerInfo.containNull;
                    uint16_t itemLen = headerInfo.itemLen;

                    dataType = CovertToAriesColumnType(colType, length, column->IsAllowNull(), true, column->numeric_precision, column->numeric_scale);
                    size_t dataTypeSize = dataType.GetDataTypeSize();
                    if (itemLen != dataTypeSize) {
                        switch( dataType.DataType.ValueType )
                        {
                        case AriesValueType::CHAR:
                            //set actual item length
                            dataType.DataType.Length = ( int )itemLen - ( int )containNull;
                            break;
                        default:
                            ARIES_ASSERT(false, "bad data type size, actual: " + to_string(itemLen) + "expect: " + to_string(dataTypeSize));
                            break;
                        }
                    }
                }
                else
                {
                    dataType = CovertToAriesColumnType(colType, length, column->IsAllowNull(), 1, column->numeric_precision, column->numeric_scale);
                }
            }
            m_columnTypes.emplace_back(dataType);
            m_columnName.emplace_back(column->GetName());
            size_t dataLen = dataType.GetDataTypeSize();
            tupleDataLen += dataLen;
        }
        m_tupleSize = tupleDataLen;
    }

    #define CONVERT_NUMERIC_NO_NULLABLE( left_type_t, right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( left_type_t* )( targetBuf + sizeOfTargetItem * i ) = *( right_type_t* )( sourceData + i * sizeOfSourceItem ); \
        } \
    }

    #define CONVERT_NUMERIC_TARGET_NULLABLE( left_type_t, right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = 1; \
            *( left_type_t* )( targetBuf + sizeOfTargetItem * i + 1 ) = *( right_type_t* )( sourceData + i * sizeOfSourceItem ); \
        } \
    }

    #define CONVERT_NUMERIC_SOURCE_NULLABLE( left_type_t, right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            if( *( sourceData + i * sizeOfSourceItem ) == 0 ) \
                return false; \
            *( left_type_t* )( targetBuf + sizeOfTargetItem * i ) = *( right_type_t* )( sourceData + i * sizeOfSourceItem + 1 );; \
        } \
    }

    #define CONVERT_NUMERIC_BOTH_NULLABLE( left_type_t, right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = *( sourceData + i * sizeOfSourceItem ); \
            if ( *( targetBuf + sizeOfTargetItem * i ) != 0 ) \
            { \
                *( left_type_t* )( targetBuf + sizeOfTargetItem * i + 1 ) = *( right_type_t* )( sourceData + i * sizeOfSourceItem + 1 ); \
            } \
        } \
    }

    #define CONVERT_NUMERIC( left_type_t, right_type_t ) \
    { \
        if ( sourceNullable && targetNullable ) \
        { \
            CONVERT_NUMERIC_BOTH_NULLABLE( left_type_t, right_type_t ); \
        } \
        else if ( sourceNullable && !targetNullable ) \
        { \
            CONVERT_NUMERIC_SOURCE_NULLABLE( left_type_t, right_type_t ); \
        } \
        else if ( !sourceNullable && targetNullable ) \
        { \
            CONVERT_NUMERIC_TARGET_NULLABLE( left_type_t, right_type_t ); \
        } \
        else \
        { \
            CONVERT_NUMERIC_NO_NULLABLE( left_type_t, right_type_t ); \
        } \
    }

    // decimal to number
    #define CONVERT_DECIMAL_TO_NUMERIC_NO_NULLABLE( left_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( left_type_t* )( targetBuf + sizeOfTargetItem * i ) = ( ( Decimal* )( sourceData + i * sizeOfSourceItem ) )->GetDouble(); \
        } \
    }

    #define CONVERT_DECIMAL_TO_NUMERIC_TARGET_NULLABLE( left_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = 1; \
            *( left_type_t* )( targetBuf + sizeOfTargetItem * i + 1 ) = ( ( Decimal* )( sourceData + i * sizeOfSourceItem ) )->GetDouble(); \
        } \
    }

    #define CONVERT_DECIMAL_TO_NUMERIC_SOURCE_NULLABLE( left_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            if( *( sourceData + i * sizeOfSourceItem ) == 0 ) \
                return false; \
            *( left_type_t* )( targetBuf + sizeOfTargetItem * i ) = ( ( Decimal* )( sourceData + i * sizeOfSourceItem + 1 ) )->GetDouble(); \
        } \
    }

    #define CONVERT_DECIMAL_TO_NUMERIC_BOTH_NULLABLE( left_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = *( sourceData + i * sizeOfSourceItem ); \
            if ( *( targetBuf + sizeOfTargetItem * i ) != 0 ) \
            { \
                *( left_type_t* )( targetBuf + sizeOfTargetItem * i + 1 ) = ( ( Decimal* )( sourceData + i * sizeOfSourceItem + 1 ) )->GetDouble(); \
            } \
        } \
    }

    #define CONVERT_DECIMAL_TO_NUMERIC( left_type_t ) \
    { \
        if ( sourceNullable && targetNullable ) \
        { \
            CONVERT_DECIMAL_TO_NUMERIC_BOTH_NULLABLE( left_type_t ); \
        } \
        else if ( sourceNullable && !targetNullable ) \
        { \
            CONVERT_DECIMAL_TO_NUMERIC_SOURCE_NULLABLE( left_type_t ); \
        } \
        else if ( !sourceNullable && targetNullable ) \
        { \
            CONVERT_DECIMAL_TO_NUMERIC_TARGET_NULLABLE( left_type_t ); \
        } \
        else \
        { \
            CONVERT_DECIMAL_TO_NUMERIC_NO_NULLABLE( left_type_t ); \
        } \
    }

    // number to decimal
    /*
    #define CONVERT_DECIAML_NO_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            ::memcpy( targetBuf + sizeOfTargetItem * i, \
                    &Decimal( targetType.Precision, targetType.Scale ).cast( Decimal( *( right_type_t * )( sourceData + i * sizeOfSourceItem ) ) ), sizeof( Decimal ) ); \
        } \
    }

    #define CONVERT_DECIAML_TARGET_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = 1; \
            ::memcpy( targetBuf + sizeOfTargetItem * i + 1, \
                    &Decimal( targetType.Precision, targetType.Scale ).cast( Decimal( *( right_type_t * )( sourceData + i * sizeOfSourceItem ) ) ), sizeof( Decimal ) ); \
        } \
    }

    #define CONVERT_DECIAML_SOURCE_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            if( *( sourceData + i * sizeOfSourceItem ) == 0 ) \
                return false; \
            ::memcpy( targetBuf + sizeOfTargetItem * i, \
                    &Decimal( targetType.Precision, targetType.Scale ).cast( Decimal( *( right_type_t * )( sourceData + i * sizeOfSourceItem + 1 ) ) ), sizeof( Decimal ) ); \
        } \
    }

    #define CONVERT_DECIAML_BOTH_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = *( sourceData + i * sizeOfSourceItem ); \
            if ( *( targetBuf + sizeOfTargetItem * i ) != 0 ) \
            { \
                ::memcpy( targetBuf + sizeOfTargetItem * i + 1, \
                    &Decimal( targetType.Precision, targetType.Scale ).cast( Decimal( *( right_type_t * )( sourceData + i * sizeOfSourceItem + 1 ) ) ), sizeof( Decimal ) ); \
            } \
        } \
    }

    #define CONVERT_DECIAML( right_type_t ) \
    { \
        if ( sourceNullable && targetNullable ) \
        { \
            CONVERT_DECIAML_BOTH_NULLABLE( right_type_t ); \
        } \
        else if ( sourceNullable && !targetNullable ) \
        { \
            CONVERT_DECIAML_SOURCE_NULLABLE( right_type_t ); \
        } \
        else if ( !sourceNullable && targetNullable ) \
        { \
            CONVERT_DECIAML_TARGET_NULLABLE( right_type_t ); \
        } \
        else \
        { \
            CONVERT_DECIAML_NO_NULLABLE( right_type_t ); \
        } \
    }
    */

    #define CONVERT_COMPACT_DECIAML_NO_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            auto sourceValue = *( right_type_t * )( sourceData + i * sizeOfSourceItem ); \
            auto value = Decimal( targetType.Precision, targetType.Scale, ARIES_MODE_STRICT_ALL_TABLES ) \
                         .cast( Decimal( sourceValue ) ); \
            CheckDecimalError( std::to_string( sourceValue ), value, colName, i ); \
            if ( !value.ToCompactDecimal( ( char* )( targetBuf + sizeOfTargetItem * i ), targetType.Length ) ) \
            { \
                errorCode = FormatTruncWrongValueError( colName, std::to_string( sourceValue ), i, "decimal", errorMsg ); \
                if ( STRICT_MODE ) \
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg ); \
                LOG(WARNING) << "Convert data warning: " << errorMsg; \
            } \
        } \
    }

    #define CONVERT_COMPACT_DECIAML_TARGET_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = 1; \
            auto sourceValue = *( right_type_t * )( sourceData + i * sizeOfSourceItem ); \
            auto value = Decimal( targetType.Precision, targetType.Scale, ARIES_MODE_STRICT_ALL_TABLES ) \
                         .cast( Decimal( sourceValue ) ); \
            CheckDecimalError( std::to_string( sourceValue ), value, colName, i ); \
            if ( !value.ToCompactDecimal( ( char* )( targetBuf + sizeOfTargetItem * i + 1 ), targetType.Length ) ) \
            { \
                errorCode = FormatTruncWrongValueError( colName, std::to_string( sourceValue ), i, "decimal", errorMsg ); \
                if ( STRICT_MODE ) \
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg ); \
                LOG(WARNING) << "Convert data warning: " << errorMsg; \
            } \
        } \
    }

    #define CONVERT_COMPACT_DECIAML_SOURCE_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            ARIES_ASSERT( *( sourceData + i * sizeOfSourceItem ) != 0, "Cannot insert null value into non-nullable column" ); \
            auto sourceValue = *( right_type_t * )( sourceData + i * sizeOfSourceItem + 1 ); \
            auto value = Decimal( targetType.Precision, targetType.Scale, ARIES_MODE_STRICT_ALL_TABLES ) \
                         .cast( Decimal( sourceValue ) ); \
            CheckDecimalError( std::to_string( sourceValue ), value, colName, i ); \
            if ( !value.ToCompactDecimal( ( char* )( targetBuf + sizeOfTargetItem * i ), targetType.Length ) ) \
            { \
                errorCode = FormatTruncWrongValueError( colName, std::to_string( sourceValue ), i, "decimal", errorMsg ); \
                if ( STRICT_MODE ) \
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg ); \
                LOG(WARNING) << "Convert data warning: " << errorMsg; \
            } \
        } \
    }

    #define CONVERT_COMPACT_DECIAML_BOTH_NULLABLE( right_type_t ) \
    { \
        for ( size_t i = 0; i < count; i++ ) \
        { \
            *( targetBuf + sizeOfTargetItem * i ) = *( sourceData + i * sizeOfSourceItem ); \
            auto sourceValue = *( right_type_t * )( sourceData + i * sizeOfSourceItem + 1 ); \
            auto value = Decimal( targetType.Precision, targetType.Scale, ARIES_MODE_STRICT_ALL_TABLES ) \
                        .cast( Decimal( sourceValue ) ); \
            CheckDecimalError( std::to_string( sourceValue ), value, colName, i ); \
            if ( !value.ToCompactDecimal( ( char* )( targetBuf + sizeOfTargetItem * i + 1 ), targetType.Length ) ) \
            { \
                errorCode = FormatTruncWrongValueError( colName, std::to_string( sourceValue ), i, "decimal", errorMsg ); \
                if ( STRICT_MODE ) \
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg ); \
                LOG(WARNING) << "Convert data warning: " << errorMsg; \
            } \
        } \
    }

    #define CONVERT_COMPACT_DECIAML( right_type_t ) \
    { \
        if ( sourceNullable && targetNullable ) \
        { \
            CONVERT_COMPACT_DECIAML_BOTH_NULLABLE( right_type_t ); \
        } \
        else if ( sourceNullable && !targetNullable ) \
        { \
            CONVERT_COMPACT_DECIAML_SOURCE_NULLABLE( right_type_t ); \
        } \
        else if ( !sourceNullable && targetNullable ) \
        { \
            CONVERT_COMPACT_DECIAML_TARGET_NULLABLE( right_type_t ); \
        } \
        else \
        { \
            CONVERT_COMPACT_DECIAML_NO_NULLABLE( right_type_t ); \
        } \
    }

    #define FOR_START() \
    for ( size_t i = 0; i < count; i++ ) \
    { \
        auto sourceDataPtr = ( sourceData + i * sizeOfSourceItem );\
        auto targetDataPtr = ( targetBuf + i * sizeOfTargetItem ); \
        if ( targetNullable ) \
        { \
            if ( !sourceNullable ) \
                *targetDataPtr = 1; \
            else \
            { \
                *targetDataPtr = *sourceDataPtr; \
                sourceDataPtr ++; \
                if ( *targetDataPtr == 0 ) \
                { \
                    continue; \
                } \
            } \
            targetDataPtr ++; \
        } \
        else \
        { \
            if ( sourceNullable ) \
            { \
                if ( *sourceDataPtr == 0 ) \
                { \
                    return false; \
                } \
                sourceDataPtr ++; \
            } \
        }

    #define FOR_END() \
    }

    bool TransferColumnData(const string& colName, const AriesDataBuffer& target, const AriesDataBuffer& source )
    {
        assert( target.GetItemCount() == source.GetItemCount() );

        string errorMsg;
        int errorCode = 0;

        auto targetColumnType = target.GetDataType();
        auto sourceColumnType = source.GetDataType();
        auto targetType = targetColumnType.DataType;
        auto sourceType = sourceColumnType.DataType;

        auto sizeOfTargetItem = targetColumnType.GetDataTypeSize();
        auto sizeOfSourceItem = sourceColumnType.GetDataTypeSize();

        auto targetBuf = target.GetData();
        auto sourceData = source.GetData();

        auto sourceNullable = sourceColumnType.HasNull;
        auto targetNullable = targetColumnType.HasNull;

        auto count = target.GetItemCount();

        // 如果 source 中的所有数据都为 NULL，那么 suorce 的数据类型就不重要了，直接将 target 设置为 NULL
        auto convertNullValue = [&] {
            for ( size_t i = 0; i < count; i++ )
            {
                auto sourceDataPtr = ( sourceData + i * sizeOfSourceItem );
                auto targetDataPtr = ( targetBuf + i * sizeOfTargetItem );
                if ( !targetNullable && sourceNullable && *sourceDataPtr == 0 )
                {
                    ARIES_EXCEPTION(ER_BAD_NULL_ERROR, colName.c_str());
                }
                if ( !targetNullable || !sourceNullable || *sourceDataPtr != 0 )
                {
                    return false;
                }
                *targetDataPtr = 0;
            }
            return true;
        };

        if ( convertNullValue() )
        {
            return true;
        }

        switch ( targetType.ValueType )
        {
        case AriesValueType::INT8:
            switch (sourceType.ValueType)
            {
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( int8_t, uint8_t );
                break;
            case AriesValueType::INT8:
                CONVERT_NUMERIC( int8_t, int8_t );
                break;
            case AriesValueType::CHAR:
            {
                std::string string_value( ( const char* )sourceData, sourceType.Length );
                try
                {
                    auto value = std::stoll( string_value );
                    if ( value > INT8_MAX )
                    {
                        return false;
                    }
                    *( int8_t * )targetBuf = static_cast< int8_t >( value );
                }
                catch ( ... )
                {
                    return false;
                }
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::UINT8:
            switch (sourceType.ValueType)
            {
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( uint8_t, uint8_t );
                break;
            case AriesValueType::INT8:
                CONVERT_NUMERIC( uint8_t, int8_t );
                break;
            case AriesValueType::CHAR:
            {
                std::string string_value( ( const char* )sourceData, sourceType.Length );
                try
                {
                    auto value = std::stoull( string_value );
                    if ( value > UINT8_MAX )
                    {
                        return false;
                    }
                    *( uint8_t * )targetBuf = static_cast< uint8_t >( value );
                }
                catch ( ... )
                {
                    return false;
                }
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::INT16:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( int16_t, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( int16_t, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( int16_t, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( int16_t, uint16_t );
                break;
            case AriesValueType::CHAR:
            {
                std::string string_value( ( const char* )sourceData, sourceType.Length );
                try
                {
                    auto value = std::stoll( string_value );
                    if ( value > INT16_MAX )
                    {
                        return false;
                    }
                    *( int16_t * )targetBuf = static_cast< int16_t >( value );
                }
                catch ( ... )
                {
                    return false;
                }
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::UINT16:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( uint16_t, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( uint16_t, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( uint16_t, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( uint16_t, uint16_t );
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                    try
                    {
                        auto value = std::stoull( string_value );
                        if ( value > UINT16_MAX )
                        {
                            return false;
                        }
                        *( uint16_t * )targetDataPtr = static_cast< uint16_t >( value );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::INT32:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( int32_t, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( int32_t, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( int32_t, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( int32_t, uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_NUMERIC( int32_t, int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_NUMERIC( int32_t, uint32_t );
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                    try
                    {
                        auto value = std::stoll( string_value );
                        if ( value > INT32_MAX )
                        {
                            return false;
                        }
                        *( int32_t * )targetDataPtr = static_cast< int32_t >( value );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::UINT32:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( uint32_t, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( uint32_t, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( uint32_t, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( uint32_t, uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_NUMERIC( uint32_t, int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_NUMERIC( uint32_t, uint32_t );
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                try
                {
                    auto value = std::stoull( string_value );
                    if ( value > UINT32_MAX )
                    {
                        return false;
                    }
                    *( uint32_t * )targetDataPtr = static_cast< uint32_t >( value );
                }
                catch ( ... )
                {
                    return false;
                }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::INT64:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( int64_t, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( int64_t, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( int64_t, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( int64_t, uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_NUMERIC( int64_t, int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_NUMERIC( int64_t, uint32_t );
                break;
            case AriesValueType::INT64:
                CONVERT_NUMERIC( int64_t, int64_t );
                break;
            case AriesValueType::UINT64:
                CONVERT_NUMERIC( int64_t, uint64_t );
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                    try
                    {
                        *( int64_t * )targetDataPtr = ( int64_t )std::stoll( string_value );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::UINT64:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( uint64_t, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( uint64_t, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( uint64_t, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( uint64_t, uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_NUMERIC( uint64_t, int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_NUMERIC( uint64_t, uint32_t );
                break;
            case AriesValueType::INT64:
                CONVERT_NUMERIC( uint64_t, int64_t );
                break;
            case AriesValueType::UINT64:
                CONVERT_NUMERIC( uint64_t, uint64_t );
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                    try
                    {
                        *( uint64_t * )targetDataPtr = ( uint64_t )std::stoull( string_value );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::FLOAT:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( float, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( float, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( float, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( float, uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_NUMERIC( float, int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_NUMERIC( float, uint32_t );
                break;
            case AriesValueType::INT64:
                CONVERT_NUMERIC( float, int64_t );
                break;
            case AriesValueType::UINT64:
                CONVERT_NUMERIC( float, uint64_t );
                break;
            case AriesValueType::DECIMAL:
                CONVERT_DECIMAL_TO_NUMERIC( float );
                break;
            case AriesValueType::COMPACT_DECIMAL:
                FOR_START()
                    auto d = Decimal( (CompactDecimal *) sourceDataPtr, sourceType.Precision, sourceType.Scale );
                    *( float* )targetDataPtr = d.GetDouble();
                FOR_END()
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                    try
                    {
                        *( float * )targetDataPtr = ( int64_t )std::stof( string_value );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::DOUBLE:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_NUMERIC( double, int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_NUMERIC( double, uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_NUMERIC( double, int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_NUMERIC( double, uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_NUMERIC( double, int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_NUMERIC( double, uint32_t );
                break;
            case AriesValueType::INT64:
                CONVERT_NUMERIC( double, int64_t );
                break;
            case AriesValueType::UINT64:
                CONVERT_NUMERIC( double, uint64_t );
                break;
            case AriesValueType::FLOAT:
                CONVERT_NUMERIC( double, float );
                break;
            case AriesValueType::DECIMAL:
                CONVERT_DECIMAL_TO_NUMERIC( double );
                break;
            case AriesValueType::COMPACT_DECIMAL:
                FOR_START()
                    auto d = Decimal( (CompactDecimal *) sourceDataPtr, sourceType.Precision, sourceType.Scale );
                    *( double* )targetDataPtr = d.GetDouble();
                FOR_END()
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                    try
                    {
                        *( double * )targetDataPtr = ( int64_t )std::stod( string_value );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::DECIMAL:
            ARIES_ASSERT( 0, "unexpected target column type: decimal" );
            // for decimal column, target type should be COMPACT_DECIMAL
            /*
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_DECIAML( int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_DECIAML( uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_DECIAML( int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_DECIAML( uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_DECIAML( int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_DECIAML( uint32_t );
                break;
            case AriesValueType::INT64:
                CONVERT_DECIAML( int64_t );
                break;
            case AriesValueType::UINT64:
                CONVERT_DECIAML( uint64_t );
                break;
            case AriesValueType::DECIMAL:
                CONVERT_DECIAML( Decimal );
                break;
            case AriesValueType::COMPACT_DECIMAL:
            {
                FOR_START()
                    auto d = Decimal( (CompactDecimal *) sourceDataPtr, sourceType.Precision, sourceType.Scale );
                    if ( sourceType.Precision != targetType.Precision || sourceType.Scale != targetType.Scale )
                    {
                        d = Decimal( targetType.Precision, targetType.Scale ).cast( d );
                    }
                    ::memcpy( targetDataPtr, &d, sizeof( Decimal ) );
                FOR_END()
                break;
            }
            case AriesValueType::CHAR:
            {
                FOR_START()
                    try
                    {
                        std::string string_value( ( const char* )sourceDataPtr, sizeOfSourceItem );
                        auto d = Decimal( string_value.data() );
                        if ( targetType.Scale != d.frac || targetType.Precision != d.frac + d.intg) {
                            d = Decimal( targetType.Precision, targetType.Scale ).cast( d );
                        }
                        memcpy( targetDataPtr, &d, sizeof( Decimal ) );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            */
            break;
        case AriesValueType::COMPACT_DECIMAL:
            switch (sourceType.ValueType)
            {
            case AriesValueType::INT8:
                CONVERT_COMPACT_DECIAML( int8_t );
                break;
            case AriesValueType::UINT8:
                CONVERT_COMPACT_DECIAML( uint8_t );
                break;
            case AriesValueType::INT16:
                CONVERT_COMPACT_DECIAML( int16_t );
                break;
            case AriesValueType::UINT16:
                CONVERT_COMPACT_DECIAML( uint16_t );
                break;
            case AriesValueType::INT32:
                CONVERT_COMPACT_DECIAML( int32_t );
                break;
            case AriesValueType::UINT32:
                CONVERT_COMPACT_DECIAML( uint32_t );
                break;
            case AriesValueType::INT64:
                CONVERT_COMPACT_DECIAML( int64_t );
                break;
            case AriesValueType::UINT64:
                CONVERT_COMPACT_DECIAML( uint64_t );
                break;
            case AriesValueType::DECIMAL:
            {
                FOR_START()
                    auto& d = *(Decimal *) sourceDataPtr;
                    if ( sourceType.Scale != targetType.Scale || sourceType.Precision != targetType.Precision )
                    {
                        d = Decimal( targetType.Precision, targetType.Scale ).cast(d);
                        CheckDecimalError( "", d, colName, i );
                    }
                    auto ret =  d.ToCompactDecimal( (char *) targetDataPtr, targetType.Length );

                    if ( !ret )
                    {
                        errorCode = FormatTruncWrongValueError( colName, "", i, "decimal", errorMsg );
                        if ( STRICT_MODE )
                            ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                FOR_END()
                break;
            }
            case AriesValueType::COMPACT_DECIMAL:
            {

                FOR_START()
                    auto value = Decimal( targetType.Precision, targetType.Scale, ARIES_MODE_STRICT_ALL_TABLES )
                    .cast( Decimal( (CompactDecimal *) sourceDataPtr, sourceType.Precision, sourceType.Scale ) );

                    CheckDecimalError( "", value, colName, i );

                    if ( !value.ToCompactDecimal( (char *) targetDataPtr, targetType.Length ) )
                    {
                        errorCode = FormatTruncWrongValueError( colName, "", i, "decimal", errorMsg );
                        if ( STRICT_MODE )
                            ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }

                FOR_END()
                break;
            }
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sourceType.Length );

                    CheckDecimalPrecision( string_value );

                    aries_acc::Decimal value( targetType.Precision,
                                              targetType.Scale,
                                              ARIES_MODE_STRICT_ALL_TABLES,
                                              string_value.data() );

                    CheckDecimalError( string_value, value, colName, i );

                    if ( !value.ToCompactDecimal( (char *)targetDataPtr, targetType.Length) )
                    {
                        errorCode = FormatTruncWrongValueError( colName, string_value, i, "decimal", errorMsg );
                        if ( STRICT_MODE )
                            ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        case AriesValueType::YEAR:
        {
            auto tmp = AriesYear();
            switch (sourceType.ValueType)
            {
            case AriesValueType::YEAR:
                ARIES_ASSERT(0, "can't hit here");
                break;
            case AriesValueType::DATE:
                memcpy(targetBuf, &(tmp=AriesYear(((AriesDate*)sourceData)->year)), sizeof(AriesYear));
                break;
            case AriesValueType::DATETIME:
                memcpy(targetBuf, &(tmp=AriesYear(((AriesDatetime*)sourceData)->year)), sizeof(AriesYear));
                break;
            case AriesValueType::TIMESTAMP:
                memcpy(targetBuf, &(tmp=AriesYear(AriesDate(*(AriesTimestamp *)sourceData).year)), sizeof(AriesYear));
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sourceType.Length );
                    try {
                        memcpy( targetDataPtr, &( tmp=AriesDatetimeTrans::GetInstance().ToAriesYear( string_value ) ), sizeof( AriesYear ));
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        }
        case AriesValueType::DATE:
        {
            auto tmp = AriesDate();
            switch (sourceType.ValueType)
            {
            case AriesValueType::DATE:
                FOR_START()
                *( AriesDate* )( targetDataPtr )  = *( AriesDate* )( sourceDataPtr );
                FOR_END()
                break;
            case AriesValueType::DATETIME:
            {
                FOR_START()
                    auto date = (AriesDatetime*)sourceDataPtr;
                    memcpy(targetDataPtr, &(tmp=AriesDate(date->year, date->month, date->day)), sizeof(AriesDate));
                FOR_END()
                break;
            }
            case AriesValueType::TIMESTAMP:
                FOR_START()
                    memcpy(targetDataPtr, &(tmp=AriesDate(*(AriesTimestamp *)sourceDataPtr)), sizeof(AriesDate));
                FOR_END()
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sourceType.Length );
                    try {
                        memcpy( targetDataPtr, &( tmp=AriesDatetimeTrans::GetInstance().ToAriesDate( string_value ) ), sizeof( AriesDate ) );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        }
        case AriesValueType::DATETIME:
        {
            auto tmp = AriesDatetime();
            switch (sourceType.ValueType)
            {
            case AriesValueType::DATE:
                FOR_START()
                    tmp = AriesDatetime( *( AriesDate *)sourceDataPtr );
                    *( AriesDatetime* )( targetDataPtr ) = tmp;
                FOR_END()
                break;
            case AriesValueType::DATETIME:
                FOR_START()
                    *( AriesDatetime* )( targetDataPtr )  = *( AriesDatetime* )( sourceDataPtr );
                FOR_END()
                break;
            case AriesValueType::TIMESTAMP:
                FOR_START()
                    memcpy(targetDataPtr, &(tmp=AriesDatetime(*(AriesTimestamp *)sourceDataPtr)), sizeof(AriesDatetime));
                FOR_END()
                break;
            case AriesValueType::CHAR:
            {

                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sourceType.Length );
                    try {
                        memcpy( targetDataPtr, &( tmp=AriesDatetimeTrans::GetInstance().ToAriesDatetime( string_value ) ), sizeof( AriesDatetime ) );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        }
        case AriesValueType::TIMESTAMP:
        {
            auto tmp = AriesTimestamp();
            switch (sourceType.ValueType)
            {
            case AriesValueType::DATE:
                FOR_START()
                    memcpy(targetDataPtr, &(tmp=AriesTimestamp(*(AriesDate*)sourceDataPtr)), sizeof(AriesTimestamp));
                FOR_END()
                break;
            case AriesValueType::DATETIME:
                FOR_START()
                    tmp = AriesTimestamp( ( ( AriesDatetime* )sourceDataPtr )->toTimestamp() );
                    // memcpy(targetDataPtr, &(tmp=AriesTimestamp(((AriesDatetime*)sourceDataPtr)->toTimestamp())), sizeof(AriesTimestamp));
                    *( AriesTimestamp* )( targetDataPtr )  = tmp;
                FOR_END()
                break;
            case AriesValueType::TIMESTAMP:
                FOR_START()
                    *( AriesTimestamp* )( targetDataPtr )  = *( AriesTimestamp* )( sourceDataPtr );
                FOR_END()
                break;
            case AriesValueType::CHAR:
            {
                FOR_START()
                    std::string string_value( ( const char* )sourceDataPtr, sourceType.Length );
                    try {
                        memcpy( targetDataPtr, &( tmp=AriesDatetimeTrans::GetInstance().ToAriesTimestamp( string_value ) ), sizeof( AriesTimestamp ) );
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                FOR_END()
                break;
            }
            default:
                return false;
            }
            break;
        }
        case AriesValueType::CHAR:
            switch (sourceType.ValueType)
            {
            case AriesValueType::CHAR:
                FOR_START()
                    auto actualDataSize = strlen( ( const char* )sourceDataPtr );
                    actualDataSize = std::min( actualDataSize, ( size_t )sourceType.Length );
                    if ( actualDataSize > ( size_t )targetType.Length )
                    {
                        if ( STRICT_MODE )
                        {
                            ARIES_EXCEPTION( ER_DATA_TOO_LONG, colName.data(), i + 1 );
                        }
                        else
                        {
                            string tmpErrorMsg;
                            FormatDataTruncError( colName, i, tmpErrorMsg );
                            LOG(WARNING) << "Convert data warning: " << tmpErrorMsg;
                            actualDataSize = targetType.Length;
                        }
                    }
                    memcpy( targetDataPtr, sourceDataPtr, actualDataSize );
                    if ( ( size_t )targetType.Length > actualDataSize )
                    {
                        memset( targetDataPtr + actualDataSize, 0x00, targetType.Length - actualDataSize );
                    }
                FOR_END()
                break;
            default:
                return false;
            }
            break;
        default:
            return false;
        }
        return true;

    }

END_ARIES_ENGINE_NAMESPACE
