#include <string.h>
#include "AriesConstantNode.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"
#include "datatypes/AriesDatetimeTrans.h"

extern bool STRICT_MODE;
BEGIN_ARIES_ENGINE_NAMESPACE

AriesConstantNode::AriesConstantNode(const std::string &dbName, const std::string &tableName)
    : m_dbName(dbName), m_tableName(tableName)
{
}

AriesConstantNode::~AriesConstantNode()
{
}

bool AriesConstantNode::Open()
{
    m_hasMoreData = true;
    return true;
}

void AriesConstantNode::Close()
{
}

#define SET_VALUE(dst_type, src_type)                       \
    do                                                      \
    {                                                       \
        ARIES_ASSERT(content.type() == typeid(src_type),    \
                     "invalid type: " + content.which());   \
        *((dst_type *)(p)) = boost::get<src_type>(content); \
    } while (0)

int ConvertInt( const AriesExpressionContent& content, 
                const string& colName,
                AriesColumnType dstType,
                size_t i, // row index
                int8_t *p,
                int mode,
                string& errorMsg )
{
    int errorCode = 0;
    // 整数常量在词法解析时，如果没有超过int32,都解析成了int32类型
    if (content.type() == typeid(int8_t))
    {
        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + colName + ") from int8_t");
    }
    else if (content.type() == typeid(int16_t))
    {
        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + colName + ") from int16_t");
    }
    else if (content.type() == typeid(int32_t))
    {
        int32_t c = boost::get< int32_t >( content );
        switch (dstType.DataType.ValueType)
        {
        case AriesValueType::INT8:
        case AriesValueType::BOOL:
        {
            if ( c > INT8_MAX || c < INT8_MIN )
            {
                errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
                if ( STRICT_MODE )
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }
            SET_VALUE(int8_t, int32_t);

            break;
        }
        case AriesValueType::INT16:
        {
            if ( c > INT16_MAX || c < INT16_MIN )
            {
                errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
                if ( STRICT_MODE )
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }
            SET_VALUE(int16_t, int32_t);
            break;
        }
        case AriesValueType::INT32:
        {
            SET_VALUE(int32_t, int32_t);
            break;
        }
        case AriesValueType::INT64:
        {
            SET_VALUE(int64_t, int32_t);
            break;
        }
        case AriesValueType::FLOAT:
        {
            SET_VALUE(float, int32_t);
            break;
        }
        case AriesValueType::DOUBLE:
        {
            SET_VALUE(double, int32_t);
            break;
        }
        case AriesValueType::DECIMAL:
        {
            SET_VALUE(aries_acc::Decimal, int32_t);
            break;
        }
        case AriesValueType::COMPACT_DECIMAL:
        {
            auto dec = aries_acc::Decimal(dstType.DataType.Precision, dstType.DataType.Scale, ARIES_MODE_STRICT_ALL_TABLES);
            dec.cast( aries_acc::Decimal(boost::get<int32_t>(content)) );
            if ( dec.GetError() == ERR_OVER_FLOW )
            {
                errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
                if ( STRICT_MODE  )
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }

            if ( !dec.ToCompactDecimal((char *)p, dstType.DataType.Length) )
            {
                errorCode = FormatTruncWrongValueError( colName, std::to_string( boost::get<int32_t>(content) ), i, "decimal", errorMsg );
                if ( STRICT_MODE  )
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }
            break;
        }
        case AriesValueType::YEAR:
        {
            aries_acc::AriesYear year(0);
            string contentStr = std::to_string( c );
            try
            {
                year = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesYear( contentStr, mode );
            }
            catch ( ... )
            {
                errorCode = FormatTruncWrongValueError( colName, contentStr, i, "year", errorMsg );
                if ( STRICT_MODE )
                    return errorCode;
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesYear * )p = year;
            break;
        }
        /*
        case AriesValueType::DATE:
        {
            aries_acc::AriesDate date;
            string contentStr = std::to_string( c );
            try
            {
                date = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, i, "date", errorMsg );
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesDate * )p = date;
            break;
        }
        case AriesValueType::TIME:
        {
            aries_acc::AriesTime time;
            string contentStr = std::to_string( c );
            try
            {
                time = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, c, "time", errorMsg);
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesTime * )p = time;
            break;
        }
        case AriesValueType::DATETIME:
        {
            aries_acc::AriesDatetime datetime;
            string contentStr = std::to_string( c );
            try
            {
                datetime = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, c, "datetime", errorMsg);
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesDatetime * )p = datetime;
            break;
        }
        case AriesValueType::TIMESTAMP:
        {
            aries_acc::AriesTimestamp timestamp;
            string contentStr = std::to_string( c );
            try
            {
                timestamp = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTimestamp( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, c, "timestamp", errorMsg);
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesTimestamp * )p = timestamp;
            break;
        }
        */
        default:
            ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + colName + ") from int32_t");
        }
    }

    else if (content.type() == typeid(int64_t))
    {
        switch (dstType.DataType.ValueType)
        {
        case AriesValueType::INT8:
        case AriesValueType::BOOL:
        case AriesValueType::INT16:
        {
            errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
            if ( STRICT_MODE )
                return errorCode;
            LOG( WARNING ) << "Convert data warning: " << errorMsg;
            break;
        }
        case AriesValueType::INT32:
        {
            // INT32_MIN, -2147483648， 被解析成了int64
            int64_t c = boost::get< int64_t >( content );
            if ( c > INT32_MAX || c < INT32_MIN )
            {
                errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
                if ( STRICT_MODE )
                    return errorCode;
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            SET_VALUE(int32_t, int64_t);
            break;
        }
        case AriesValueType::INT64:
        {
            SET_VALUE(int64_t, int64_t);
            break;
        }
        case AriesValueType::FLOAT:
        {
            SET_VALUE(float, int64_t);
            break;
        }
        case AriesValueType::DOUBLE:
        {
            SET_VALUE(double, int64_t);
            break;
        }
        case AriesValueType::DECIMAL:
        {
            SET_VALUE(aries_acc::Decimal, int64_t);
            break;
        }
        case AriesValueType::COMPACT_DECIMAL:
        {
            auto dec = aries_acc::Decimal(dstType.DataType.Precision, dstType.DataType.Scale, ARIES_MODE_STRICT_ALL_TABLES);
            dec.cast( aries_acc::Decimal(boost::get<int64_t>(content)) );
            if ( dec.GetError() == ERR_OVER_FLOW )
            {
                errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
                if ( STRICT_MODE  )
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }

            if ( !dec.ToCompactDecimal((char *)p, dstType.DataType.Length) )
            {
                errorCode = FormatTruncWrongValueError( colName, std::to_string( boost::get<int64_t>(content) ), i, "decimal", errorMsg );
                if ( STRICT_MODE  )
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }

            break;
        }
        case AriesValueType::YEAR:
        {
            aries_acc::AriesYear year(0);
            int64_t c = boost::get< int64_t >( content );
            string contentStr = std::to_string( c );
            try
            {
                year = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesYear( contentStr, mode );
            }
            catch ( ... )
            {
                errorCode = FormatTruncWrongValueError( colName, contentStr, i, "year", errorMsg );
                if ( STRICT_MODE )
                    return errorCode;
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesYear * )p = year;
            break;
        }
        /*
        case AriesValueType::DATE:
        {
            int64_t c = boost::get<int64_t>( content );
            aries_acc::AriesDate date;
            string contentStr = std::to_string( c );
            try
            {
                date = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, i, "date", errorMsg );
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesDate * )p = date;
            break;
        }
        case AriesValueType::TIME:
        {
            int64_t c = boost::get<int64_t>( content );
            aries_acc::AriesTime time;
            string contentStr = std::to_string( c );
            try
            {
                time = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, c, "time", errorMsg);
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesTime * )p = time;
            break;
        }
        case AriesValueType::DATETIME:
        {
            int64_t c = boost::get<int64_t>( content );
            aries_acc::AriesDatetime datetime;
            string contentStr = std::to_string( c );
            try
            {
                datetime = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, c, "datetime", errorMsg);
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesDatetime * )p = datetime;
            break;
        }
        case AriesValueType::TIMESTAMP:
        {
            int64_t c = boost::get<int64_t>( content );
            aries_acc::AriesTimestamp timestamp;
            string contentStr = std::to_string( c );
            try
            {
                timestamp = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTimestamp( contentStr, mode );
            }
            catch ( ... )
            {
                string errorMsg;
                int errorCode = FormatTruncWrongValueError( colEntry->GetName(), contentStr, c, "timestamp", errorMsg);
                if ( STRICT_MODE )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                LOG( WARNING ) << "Convert data warning: " << errorMsg;
            }
            *( AriesTimestamp * )p = timestamp;
            break;
        }
        */
        default:
            ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + colName + ") from int64_t");
        }
    }
    else
    {
        ARIES_ASSERT(0, "invalid content type: " + content.which());
    }
    return 0;
}

int ConvertString( const AriesExpressionContent& content,                     
                   ColumnEntryPtr& colEntry,
                   AriesColumnType dstType,
                   size_t i, // row index
                   int8_t *p,
                   int mode,
                   string& errorMsg )
{
    int errorCode = 0;
    auto string_value = boost::get<std::string>(content);
    char *tail;
    long longVal = 0;

    switch (dstType.DataType.ValueType)
    {
    case AriesValueType::INT8:
    case AriesValueType::BOOL:
    {
        int8_t value = 0;
        longVal = std::strtol( string_value.data(), &tail, 10 );
        CheckTruncError(colEntry->GetName(), "integer", string_value, i, errorMsg);

        if (longVal > INT8_MAX || longVal < INT8_MIN)
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            if (longVal > INT8_MAX)
                value = INT8_MAX;
            else
                value = INT8_MIN;
            LOG( WARNING ) << "Convert data warning: " << errorMsg;
        }
        else
            value = longVal;

        *(int8_t *)p = static_cast<int8_t>(value);
        break;
    }
    case AriesValueType::INT16:
    {
        int16_t value = 0;
        longVal = std::strtol(string_value.data(), &tail, 10);
        CheckTruncError(colEntry->GetName(), "integer", string_value, i, errorMsg);

        if (longVal > INT16_MAX || longVal < INT16_MIN)
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            if (longVal > INT16_MAX)
                value = INT16_MAX;
            else
                value = INT16_MIN;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        else
            value = longVal;

        *(int16_t *)p = static_cast<int16_t>(value);
        break;
    }
    case AriesValueType::INT32:
    {
        errno = 0;
        int32_t value = 0;
        longVal = std::strtol(string_value.data(), &tail, 10);
        CheckTruncError( colEntry->GetName(), "integer", string_value, i, errorMsg );

        if ( longVal > INT32_MAX || longVal < INT32_MIN )
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            if (longVal > INT32_MAX)
                value = INT32_MAX;
            else
                value = INT32_MIN;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        else
        {
            if (ERANGE == errno)
            {
                errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
                if (STRICT_MODE)
                    return errorCode;
                LOG(WARNING) << "Convert data warning: " << errorMsg;
            }
            value = longVal;
        }

        *(int32_t *)p = static_cast<int32_t>(value);
        break;
    }
    case AriesValueType::INT64:
    {
        errno = 0;
        int64_t value = std::strtoll(string_value.data(), &tail, 10);
        CheckTruncError( colEntry->GetName(), "integer", string_value, i, errorMsg );

        if (errno == ERANGE /*&& llValUnsigned == ULLONG_MAX*/)
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }

        *(int64_t *)p = value;
        break;
    }
    case AriesValueType::FLOAT:
    {
        errno = 0;
        float value = std::strtof( string_value.data(), &tail );
        CheckTruncError(colEntry->GetName(), "float", string_value, i, errorMsg);

        if ( ERANGE == errno )
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        if (colEntry->is_unsigned && value < 0)
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
            value = 0;
        }
        *( ( float *)p ) = value;
        break;
    }
    case AriesValueType::DOUBLE:
    {
        errno = 0;
        double value = std::strtod( string_value.data(), &tail );
        CheckTruncError(colEntry->GetName(), "double", string_value, i, errorMsg);

        if ( ERANGE == errno )
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        if (colEntry->is_unsigned && value < 0)
        {
            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), i, errorMsg);
            if (STRICT_MODE)
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
            value = 0;
        }
        *( ( double *)p ) = value;
        break;
    }
    case AriesValueType::DECIMAL:
    {
        /*
        try
        {
            auto d = aries_acc::Decimal(string_value.data());
            if (dstType.DataType.Scale != d.frac || dstType.DataType.Precision != d.frac + d.intg)
                d = aries_acc::Decimal(dstType.DataType.Precision, dstType.DataType.Scale).cast(d);
            memcpy(p, &d, sizeof(aries_acc::Decimal));
        }
        catch (...)
        {
            ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from string");
        }
        */
        CheckDecimalPrecision( string_value );
        aries_acc::Decimal value(colEntry->numeric_precision,
                                 colEntry->numeric_scale,
                                 ARIES_MODE_STRICT_ALL_TABLES,
                                 string_value.data());
        if ( value.GetError() == ERR_STR_2_DEC )
        {
            errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "decimal", errorMsg );
            if ( STRICT_MODE )
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        if ( value.GetError() == ERR_OVER_FLOW )
        {
            errorCode = FormatOutOfRangeValueError( colEntry->GetName(), i, errorMsg );
            if ( STRICT_MODE )
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }

        memcpy(p, &value, sizeof(aries_acc::Decimal));
        break;
    }
    case AriesValueType::COMPACT_DECIMAL:
    {
        /*
        try
        {
            auto d = aries_acc::Decimal(string_value.data());
            if (dstType.DataType.Scale != d.frac || dstType.DataType.Precision != d.frac + d.intg)
                d = aries_acc::Decimal(dstType.DataType.Precision, dstType.DataType.Scale).cast(d);
            d.ToCompactDecimal((char *)p, dstType.DataType.Length);
        }
        catch (...)
        {
            ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from string");
        }
        */

        CheckDecimalPrecision( string_value );
        aries_acc::Decimal value(colEntry->numeric_precision,
                                 colEntry->numeric_scale,
                                 ARIES_MODE_STRICT_ALL_TABLES,
                                 string_value.data());
        if ( value.GetError() == ERR_STR_2_DEC )
        {
            errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "decimal", errorMsg );
            if ( STRICT_MODE )
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        if ( value.GetError() == ERR_OVER_FLOW )
        {
            errorCode = FormatOutOfRangeValueError( colEntry->GetName(), i, errorMsg );
            if ( STRICT_MODE )
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }

        if ( !value.ToCompactDecimal((char *)p, dstType.DataType.Length) )
        {
            errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "decimal", errorMsg );
            if ( STRICT_MODE )
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        break;
    }
    case AriesValueType::CHAR:
    {
        CheckCharLen( colEntry->GetName(), string_value.size() );
        size_t maxSize = string_value.size();
        if ( dstType.DataType.Length > 0 && maxSize > std::size_t( dstType.DataType.Length ) )
        {
            errorCode = FormatDataTruncError(colEntry->GetName(), i, errorMsg);
            if ( STRICT_MODE )
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
            maxSize = dstType.DataType.Length;
        }
        memset(p, 0, dstType.DataType.Length);
        memcpy(p, string_value.data(), maxSize);
        break;
    }
    case AriesValueType::YEAR:
    {
        try
        {
            *(AriesYear *)p = AriesDatetimeTrans::GetInstance().ToAriesYear(string_value, mode);
        }
        catch (...)
        {
            string errorMsg;
            int errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "year", errorMsg);
            if ( STRICT_MODE )
                return errorCode;
            LOG( WARNING ) << "Convert data warning: " << errorMsg;
        }
        break;
    }
    case AriesValueType::DATE:
    {
        try
        {
            *(AriesDate *)p = AriesDatetimeTrans::GetInstance().ToAriesDate(string_value, mode);
        }
        catch (...)
        {
            string errorMsg;
            int errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "date", errorMsg);
            if ( STRICT_MODE )
                return errorCode;
            LOG( WARNING ) << "Convert data warning: " << errorMsg;
        }
        break;
    }
    case AriesValueType::TIME:
    {
        try
        {
            *(AriesTime *)p = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( string_value, mode );
        }
        catch ( ... )
        {
            errorCode = FormatTruncWrongValueError(colEntry->GetName(), string_value, i, "time", errorMsg);
            if (STRICT_MODE)
                return errorCode;
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        break;
    }
    case AriesValueType::DATETIME:
    {
        try
        {
            *(AriesDatetime *)p = AriesDatetimeTrans::GetInstance().ToAriesDatetime(string_value, mode);
        }
        catch (...)
        {
            string errorMsg;
            int errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "datetime", errorMsg);
            if ( STRICT_MODE )
                return errorCode;
            LOG( WARNING ) << "Convert data warning: " << errorMsg;
        }
        break;
    }
    case AriesValueType::TIMESTAMP:
    {
        try
        {
            *(AriesTimestamp *)p = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(string_value, mode);
        }
        catch (...)
        {
            string errorMsg;
            int errorCode = FormatTruncWrongValueError( colEntry->GetName(), string_value, i, "timestamp", errorMsg);
            if ( STRICT_MODE )
                return errorCode;
            LOG( WARNING ) << "Convert data warning: " << errorMsg;
        }
        break;
    }
    default:
        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + colEntry->GetName() + ") from string");
    }

    return 0;
}

int AriesConstantNode::SetColumnData( const std::vector<std::vector<AriesCommonExprUPtr>> &data,
                                      const std::vector<int> &columnIds,
                                      string& errorMsg )
{
    ARIES_ASSERT(!data.empty() && !data[0].empty(), "data should not be empty");
    int errorCode = 0;
    int mode = STRICT_MODE ? ARIES_DATE_STRICT_MODE : ARIES_DATE_NOT_STRICT_MODE;

    auto tableEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(m_dbName)->GetTableByName(m_tableName);

#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif
    m_buffers.clear();
    for (std::size_t i = 0; i < data[0].size(); i++)
    {
        m_buffers.emplace_back(nullptr);
        if (data[0][i]->GetValueType().DataType.ValueType == AriesValueType::CHAR)
        {
            int max_len = 0;
            int max_row_index = -1;
            for (std::size_t j = 0; j < data.size(); j++)
            {
                if (data[j][i]->GetValueType().DataType.Length > max_len)
                {
                    max_len = data[j][i]->GetValueType().DataType.Length;
                    max_row_index = j;
                }
            }

            auto colEntry = tableEntry->GetColumnById(columnIds[i]);
            AriesColumnType dstType = CovertToAriesColumnType(colEntry->GetType(), colEntry->GetLength(), colEntry->IsAllowNull(), true,
                                                              colEntry->numeric_precision, colEntry->numeric_scale);
            if ( dstType.DataType.ValueType == AriesValueType::CHAR )
            {
                CheckCharLen( colEntry->GetName(), max_len );
                if ( max_len > colEntry->GetLength() )
                {
                    if ( STRICT_MODE )
                    {
                        return FormatDataTooLongError( colEntry->GetName(), max_row_index, errorMsg );
                    }
                    errorCode = FormatDataTruncError(colEntry->GetName(), max_row_index, errorMsg);
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
            }

            m_buffers[i] = std::make_shared<AriesDataBuffer>(dstType);
            int8_t* buff = ( int8_t* ) malloc( data.size() * dstType.GetDataTypeSize() );
            m_buffers[i]->AttachBuffer( buff, data.size() );
        }
    }

    for (std::size_t i = 0; i < data.size(); i++)
    {
        const auto &row = data[i];
        for (std::size_t j = 0; j < row.size(); j++)
        {
            const auto &item = row[j];
            ARIES_ASSERT(item->IsLiteralValue(), "data should be literal");
            AriesColumnType dstType;
            auto colEntry = tableEntry->GetColumnById(columnIds[j]);
            if (!m_buffers[j])
            {
                dstType = CovertToAriesColumnType(colEntry->GetType(), colEntry->GetLength(), colEntry->IsAllowNull(), true,
                                                  colEntry->numeric_precision, colEntry->numeric_scale);

                m_buffers[j] = std::make_shared<AriesDataBuffer>(dstType);
                int8_t* buff = ( int8_t* ) malloc( data.size() * dstType.GetDataTypeSize() );
                m_buffers[j]->AttachBuffer( buff, data.size() );
            }
            else
                dstType = m_buffers[j]->GetDataType();

            const auto &content = item->GetContent();
            int8_t *p = m_buffers[j]->GetItemDataAt(i);

            if (item->GetType() == AriesExprType::NULL_VALUE)
            {
                if (!dstType.HasNull)
                    ARIES_EXCEPTION(ER_BAD_NULL_ERROR, tableEntry->GetColumnById(columnIds[j])->GetName().c_str());

                *p = 0;
            }

            if (dstType.HasNull && !item->GetValueType().HasNull)
            {
                *p = 1;
                p++;
            }
            switch (item->GetType())
            {
            case AriesExprType::INTEGER:
            {
                errorCode = ConvertInt( content,
                                        colEntry->GetName(),
                                        dstType,
                                        i,
                                        p,
                                        mode,
                                        errorMsg );
                if ( 0 != errorCode )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                break;
            }
            case AriesExprType::STRING:
            {
                errorCode = ConvertString( content,
                                           colEntry,
                                           dstType,
                                           i,
                                           p,
                                           mode,
                                           errorMsg );
                if ( 0 != errorCode )
                    ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );
                break;
            }
            case AriesExprType::FLOATING:
            {
                ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from float");
                /*
                if (content.type() == typeid(float))
                {
                    SET_VALUE(float, float);
                }
                else if (content.type() == typeid(double))
                {
                    SET_VALUE(double, double);
                }
                else
                {
                    ARIES_ASSERT(0, "invalid content type: " + content.which());
                }
                */
                break;
            }
            case AriesExprType::DECIMAL:
            {
                // TODO: scientific floating to decimal has bugs
                ARIES_ASSERT(content.type() == typeid(aries_acc::Decimal),
                             "invalid content type: " + static_cast<int>(content.which()));
                aries_acc::Decimal dec = boost::get< aries_acc::Decimal >( content );
                switch (dstType.DataType.ValueType)
                {
                    case AriesValueType::INT8:
                    case AriesValueType::BOOL:
                    {
                        if ( dec > INT8_MAX || dec < INT8_MIN )
                        {
                            string errMsg;
                            FormatOutOfRangeValueError( colEntry->GetName(), i, errMsg );
                            ARIES_EXCEPTION_SIMPLE( ER_WARN_DATA_OUT_OF_RANGE, errMsg.data() );
                        }
                        double d = dec.GetDouble();
                        int8_t i = ( int8_t ) d;
                        *( ( int8_t* )p ) = i;

                        break;
                    }
                    case AriesValueType::INT16:
                    {
                        if ( dec > INT16_MAX || dec < INT16_MIN )
                        {
                            string errMsg;
                            FormatOutOfRangeValueError( colEntry->GetName(), i, errMsg );
                            ARIES_EXCEPTION_SIMPLE( ER_WARN_DATA_OUT_OF_RANGE, errMsg.data() );
                        }
                        double d = dec.GetDouble();
                        int16_t i = ( int16_t ) d;
                        *( ( int16_t* )p ) = i;
                        break;
                    }
                    case AriesValueType::INT32:
                    {
                        if ( dec > INT32_MAX || dec < INT32_MIN )
                        {
                            string errMsg;
                            FormatOutOfRangeValueError( colEntry->GetName(), i, errMsg );
                            ARIES_EXCEPTION_SIMPLE( ER_WARN_DATA_OUT_OF_RANGE, errMsg.data() );
                        }
                        double d = dec.GetDouble();
                        int32_t i = ( int32_t ) d;
                        *( ( int32_t* )p ) = i;
                        break;
                    }
                    case AriesValueType::INT64:
                    {
                        if ( dec > INT64_MAX || dec < INT64_MIN )
                        {
                            string errMsg;
                            FormatOutOfRangeValueError( colEntry->GetName(), i, errMsg );
                            ARIES_EXCEPTION_SIMPLE( ER_WARN_DATA_OUT_OF_RANGE, errMsg.data() );
                        }
                        // int64_t i64 = dec.ToInt64();
                        double d = dec.GetDouble();
                        int64_t i64 = ( int64_t ) d;
                        *( ( int64_t* )p ) = i64;
                        break;
                    }
                    case AriesValueType::FLOAT:
                    {
                        *( ( float* )p ) = dec.GetDouble();
                        break;
                    }
                    case AriesValueType::DOUBLE:
                    {
                        *( ( double* )p ) = dec.GetDouble();
                        break;
                    }
                    case AriesValueType::DECIMAL:
                    {
                        SET_VALUE(aries_acc::Decimal, aries_acc::Decimal);
                        break;
                    }
                    case AriesValueType::COMPACT_DECIMAL:
                    {
                        auto dec = aries_acc::Decimal(dstType.DataType.Precision, dstType.DataType.Scale, ARIES_MODE_STRICT_ALL_TABLES);
                        dec.cast( aries_acc::Decimal(boost::get<aries_acc::Decimal>(content)) );
                        if ( dec.GetError() == ERR_OVER_FLOW )
                        {
                            errorCode = FormatOutOfRangeValueError( colEntry->GetName(), i, errorMsg );
                            if ( STRICT_MODE  )
                                ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
                            LOG(WARNING) << "Convert data warning: " << errorMsg;
                        }

                        if ( !dec.ToCompactDecimal((char *)p, dstType.DataType.Length) )
                        {
                            char tmpBuff[ 64 ];
                            errorCode = FormatTruncWrongValueError( colEntry->GetName(), dec.GetDecimal( tmpBuff ), i, "decimal", errorMsg );
                            if ( STRICT_MODE  )
                                ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
                            LOG(WARNING) << "Convert data warning: " << errorMsg;
                        }
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from decimal");
                }
                break;
            }
            case AriesExprType::DATE:
            {
                AriesDate srcData = boost::get<AriesDate>(content);
                switch (dstType.DataType.ValueType)
                {
                    case AriesValueType::DATE:
                    {
                        *(AriesDate *)p = srcData;
                        break;
                    }
                    case AriesValueType::TIMESTAMP:
                    {
                        *(AriesTimestamp *)p = AriesTimestamp( srcData );
                        break;
                    }
                    case AriesValueType::YEAR:
                    {
                        *(AriesYear *)p = AriesYear( srcData.year );
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from date");
                }
                break;
            }
            case AriesExprType::DATE_TIME:
            {
                AriesDatetime srcData = boost::get<AriesDatetime>(content);
                switch (dstType.DataType.ValueType)
                {
                    case AriesValueType::DATE:
                    {
                        *(AriesDate *)p = AriesDate( srcData.year, srcData.month, srcData.day );
                        break;
                    }
                    case AriesValueType::DATETIME:
                    {
                        *(AriesDatetime *)p = srcData;
                        break;
                    }
                    case AriesValueType::TIME:
                    {
                        *(AriesTime *)p = AriesTime((uint8_t)1, srcData.hour, srcData.minute, srcData.second, srcData.second_part);
                        break;
                    }
                    case AriesValueType::TIMESTAMP:
                    {
                        *(AriesTimestamp *)p = AriesTimestamp( srcData.toTimestamp() );
                        break;
                    }
                    case AriesValueType::YEAR:
                    {
                        *(AriesYear *)p = AriesYear( srcData.year );
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from datetime");
                }
                break;
            }
            case AriesExprType::TIME:
            {
                AriesTime srcData = boost::get<AriesTime>(content);
                switch (dstType.DataType.ValueType)
                {
                    case AriesValueType::TIME:
                    {
                        *(AriesTime *)p = srcData;
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from time");
                }
                break;
            }
            case AriesExprType::TIMESTAMP:
            {
                AriesTimestamp srcData = boost::get<AriesTimestamp>(content);
                switch (dstType.DataType.ValueType)
                {
                    case AriesValueType::DATE:
                    {
                        *(AriesDate *)p = AriesDate( srcData );
                        break;
                    }
                    case AriesValueType::DATETIME:
                    {
                        *(AriesDatetime *)p = AriesDatetime( srcData );
                        break;
                    }
                    case AriesValueType::TIMESTAMP:
                    {
                        *(AriesTimestamp *)p = srcData;
                        break;
                    }
                    case AriesValueType::YEAR:
                    {
                        *(AriesYear *)p = AriesDate( srcData ).year;
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from timestamp");
                }
                break;
            }
            case AriesExprType::NULL_VALUE:
            {
                break;
            }
            case AriesExprType::TRUE_FALSE:
            {
                switch (dstType.DataType.ValueType)
                {
                case AriesValueType::BOOL:
                {
                    SET_VALUE(bool, bool);
                    break;
                }
                case AriesValueType::INT8:
                {
                    SET_VALUE(int8_t, bool);
                    break;
                }
                case AriesValueType::INT16:
                {
                    SET_VALUE(int16_t, bool);
                    break;
                }
                case AriesValueType::INT32:
                {
                    SET_VALUE(int32_t, bool);
                    break;
                }
                case AriesValueType::INT64:
                {
                    SET_VALUE(int64_t, bool);
                    break;
                }
                /*
                case AriesValueType::DECIMAL:
                {
                    SET_VALUE(aries_acc::Decimal, bool);
                    break;
                }
                case AriesValueType::COMPACT_DECIMAL:
                {
                    bool c = boost::get< bool >( content );
                    aries_acc::Decimal(dstType.DataType.Precision, dstType.DataType.Scale)
                        .cast(aries_acc::Decimal( c ))
                        .ToCompactDecimal((char *)p, dstType.DataType.Length);
                    break;
                }
                */
                default:
                    ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from int8_t");
                }
                break;
            }
            case AriesExprType::YEAR:
            {
                AriesYear srcData = boost::get<AriesYear>(content);
                switch (dstType.DataType.ValueType)
                {
                    case AriesValueType::YEAR:
                    {
                        *(AriesYear *)p = srcData;
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from year");
                }
                break;
            }
            default:
                ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + tableEntry->GetColumnById(columnIds[j])->GetName() + ") from unknown type");
                break;
            }
        }
    }

#ifdef ARIES_PROFILE
    m_opTime += t.end();
#endif
    return 0;
}

AriesOpResult AriesConstantNode::GetNext()
{
    if (m_hasMoreData)
    {
        m_hasMoreData = false;
    }
    else
    {
        return {AriesOpNodeStatus::END, nullptr};
    }
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif
    auto table = std::make_unique<AriesTableBlock>();
    for (std::size_t i = 0; i < m_buffers.size(); i++)
    {
        auto column = std::make_shared<AriesColumn>();
        column->AddDataBuffer(m_buffers[i]);
        table->AddColumn(i + 1, column);
    }
#ifdef ARIES_PROFILE
    m_opTime += t.end();
#endif
    m_rowCount += table->GetRowCount();

    return {AriesOpNodeStatus::END, std::move(table)};
}

JSON AriesConstantNode::GetProfile() const
{
    JSON stat = this->AriesOpNode::GetProfile();
    stat["type"] = "AriesConstantNode";
    return stat;
}

END_ARIES_ENGINE_NAMESPACE
