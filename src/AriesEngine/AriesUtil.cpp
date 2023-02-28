/*
 * AriesUtil.cpp
 *
 *  Created on: Sep 28, 2018
 *      Author: lichi
 */
#include <algorithm>
#include <regex>
#include <regex>
#include "AriesUtil.h"
#include "AriesDatetimeTrans.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"
#include "AriesColumnDataIterator.hxx"
#include "CpuTimer.h"
#include "CudaAcc/AriesSqlOperator.h"

extern bool STRICT_MODE;

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesColumnType CovertToAriesColumnType( ColumnType type, int length, bool nullable, bool compact, int precision, int scale )
    {
        ARIES_ASSERT( length > 0 , "type: " + to_string( ( int )type ) + ", length: " + to_string( length ) );
        AriesDataType dataType;
        dataType.Length = length;
        AriesColumnType result;

        switch( type )
        {
            case ColumnType::TINY_INT:
                dataType.ValueType = AriesValueType::INT8;
                break;
            case ColumnType::SMALL_INT:
                dataType.ValueType = AriesValueType::INT16;
                break;
            case ColumnType::INT:
                dataType.ValueType = AriesValueType::INT32;
                break;
            case ColumnType::LONG_INT:
                dataType.ValueType = AriesValueType::INT64;
                break;
            case ColumnType::DECIMAL:
            {
                if( compact )
                {
                    dataType = AriesDataType( AriesValueType::COMPACT_DECIMAL, precision, scale );
                }
                else
                {
                    dataType.ValueType = AriesValueType::DECIMAL;
                }
                break;
            }
            case ColumnType::FLOAT:
                dataType.ValueType = AriesValueType::FLOAT;
                break;
            case ColumnType::DOUBLE:
                dataType.ValueType = AriesValueType::DOUBLE;
                break;
            case ColumnType::DATE:
                dataType.ValueType = AriesValueType::DATE;
                break;
            case ColumnType::BOOL:
                dataType.ValueType = AriesValueType::INT8;
                break;
            case ColumnType::TEXT:
                dataType.ValueType = AriesValueType::CHAR;
                break;
            case ColumnType::DATE_TIME:
                dataType.ValueType = AriesValueType::DATETIME;
                break;
            case ColumnType::TIME:
                dataType.ValueType = AriesValueType::TIME;
                break;
            case ColumnType::TIMESTAMP:
                dataType.ValueType = AriesValueType::TIMESTAMP;
                break;
            case ColumnType::YEAR:
                dataType.ValueType = AriesValueType::YEAR;
                break;
            case ColumnType::BINARY:
                dataType.ValueType = AriesValueType::CHAR;
                break;
            case ColumnType::VARBINARY:
                dataType.ValueType = AriesValueType::CHAR;
                break;
            default:
                //FIXME: need support all data types;
                LOG(INFO)<< "unhandled column type: " << static_cast< int >( type ) << std::endl;
                assert( 0 );
                ARIES_ENGINE_EXCEPTION(ER_NOT_SUPPORTED_YET, "converting column type: " + to_string(static_cast< int >( type )));
            }

        result.DataType = dataType;
        result.HasNull = nullable;
        return result;
    }

    string LogicOpToString( AriesLogicOpType opType )
    {
        static map< AriesLogicOpType, string > tmp_logictype_map =
        {
        { AriesLogicOpType::AND, "and" },
        { AriesLogicOpType::OR, "or" } };

        ARIES_ASSERT( tmp_logictype_map.find( opType ) != tmp_logictype_map.end(), "opType: " + GetAriesLogicOpTypeName( opType ) );
        return tmp_logictype_map[opType];
    }

    string ComparisonOpToString( AriesComparisonOpType opType )
    {
        static map< AriesComparisonOpType, string > tmp_comptype_map =
        {
        { AriesComparisonOpType::EQ, "==" },
        { AriesComparisonOpType::NE, "!=" },
        { AriesComparisonOpType::GT, ">" },
        { AriesComparisonOpType::GE, ">=" },
        { AriesComparisonOpType::LT, "<" },
        { AriesComparisonOpType::LE, "<=" } };

        ARIES_ASSERT( tmp_comptype_map.find( opType ) != tmp_comptype_map.end(), "opType: " + GetAriesComparisonOpTypeName( opType ) );
        return tmp_comptype_map[opType];
    }

    void StringToUpper( string &data )
    {
        transform( data.begin(), data.end(), data.begin(), ::toupper );
    }

    size_t GetAriesDataTypeSizeInBytes( const AriesDataType &dataType )
    {
        size_t size = 0;
        switch( dataType.ValueType )
        {
            case AriesValueType::INT8:
            case AriesValueType::UINT8:
            case AriesValueType::BOOL:
            case AriesValueType::CHAR:
            case AriesValueType::COMPACT_DECIMAL:
                size = sizeof(int8_t) * dataType.Length;
                break;
            case AriesValueType::INT16:
            case AriesValueType::UINT16:
                size = sizeof(int16_t) * dataType.Length;
                break;
            case AriesValueType::INT32:
            case AriesValueType::UINT32:
                size = sizeof(int32_t) * dataType.Length;
                break;
            case AriesValueType::INT64:
            case AriesValueType::UINT64:
                size = sizeof(int64_t) * dataType.Length;
                break;
            case AriesValueType::FLOAT:
                size = sizeof(float) * dataType.Length;
                break;
            case AriesValueType::DOUBLE:
                size = sizeof(double) * dataType.Length;
                break;
            case AriesValueType::DECIMAL:
                size = sizeof(aries_acc::Decimal) * dataType.Length;
                break;
            case AriesValueType::DATE:
                size = sizeof(AriesDate) * dataType.Length;
                break;
            case AriesValueType::DATETIME:
                size = sizeof(AriesDatetime) * dataType.Length;
                break;
            case AriesValueType::TIMESTAMP:
                size = sizeof(AriesTimestamp) * dataType.Length;
                break;
            case AriesValueType::TIME:
                size = sizeof(AriesTime) * dataType.Length;
                break;
            case AriesValueType::YEAR:
                size = sizeof(AriesYear) * dataType.Length;
                break;
            default:
                //FIXME need support all data types
                LOG(INFO)<< "unsupported type: " << static_cast< int >( dataType.ValueType ) << std::endl;
                assert( 0 );
                ARIES_ENGINE_EXCEPTION(ER_NOT_SUPPORTED_YET, "getting type size: " + to_string((int) dataType.ValueType));
                break;
            }
        return size;
    }

    bool IsIntegerType( const AriesColumnType& type )
    {
        bool bRet = false;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::INT8:
            case AriesValueType::UINT8:
            case AriesValueType::INT16:
            case AriesValueType::UINT16:
            case AriesValueType::INT32:
            case AriesValueType::UINT32:
            case AriesValueType::INT64:
            case AriesValueType::UINT64:
                bRet = true;
                break;
            default:
                break;
        }
        return bRet;
    }
    string GetDataTypeStringName( const AriesValueType& valueType )
    {
        switch( valueType )
        {
            case AriesValueType::BOOL:
                return "bool";
            case AriesValueType::CHAR:
                return "string";
            case AriesValueType::COMPACT_DECIMAL:
            case AriesValueType::DECIMAL:
                return "decimal";
            case AriesValueType::DATE:
                return "date";
            case AriesValueType::DATETIME:
                return "datetime";
            case AriesValueType::DOUBLE:
                return "double";
            case AriesValueType::FLOAT:
                return "float";
            case AriesValueType::INT8:
                return "tinyint";
            case AriesValueType::INT16:
                return "smallint";
            case AriesValueType::INT32:
                return "int";
            case AriesValueType::INT64:
                return "bigint";
            case AriesValueType::TIME:
                return "time";
            case AriesValueType::TIMESTAMP:
                return "timestamp";
            case AriesValueType::YEAR:
                return "year";
            default:
                return "data type";
        }
    }
    string GetValueTypeAsString( const AriesColumnType& columnType )
    {
        string type;
        switch( columnType.DataType.ValueType )
        {
            case AriesValueType::CHAR:
                if( columnType.DataType.Length == 1 )
                    type = "char";
                else
                {
                    type = "aries_char< ";
                    type += std::to_string( columnType.DataType.Length );
                    type += " >";
                }
                break;
            case AriesValueType::INT8:
            case AriesValueType::BOOL:
                type = "int8_t";
                break;
            case AriesValueType::UINT8:
                type = "uint8_t";
                break;
            case AriesValueType::INT16:
                type = "int16_t";
                break;
            case AriesValueType::UINT16:
                type = "uint16_t";
                break;
            case AriesValueType::INT32:
                type = "int32_t";
                break;
            case AriesValueType::UINT32:
                type = "uint32_t";
                break;
            case AriesValueType::INT64:
                type = "int64_t";
                break;
            case AriesValueType::UINT64:
                type = "uint64_t";
                break;
            case AriesValueType::FLOAT:
                type = "float";
                break;
            case AriesValueType::DOUBLE:
                type = "double";
                break;
            case AriesValueType::DECIMAL:
            case AriesValueType::COMPACT_DECIMAL: //in dynamic code, we always convert COMPACT_DECIMAL to DECIMAL at first
                type = "AriesDecimal<" + to_string( columnType.DataType.AdaptiveLen ) +">";
                break;
            case AriesValueType::DATE:
                type = "AriesDate";
                break;
            case AriesValueType::DATETIME:
                type = "AriesDatetime";
                break;
            case AriesValueType::TIME:
                type = "AriesTime";
                break;
            case AriesValueType::TIMESTAMP:
                type = "AriesTimestamp";
                break;
            case AriesValueType::YEAR:
                type = "AriesYear";
                break;
            case AriesValueType::LIST:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "getting data type: LIST" );
                break;
            default:
                //TODO need support other data types
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "getting data type: " + to_string( ( int ) columnType.DataType.ValueType ) );
                break;
        }
        return type;
    }

    AriesLiteralValue ConvertRawDataToLiteral( const int8_t* data, const AriesColumnType& type )
    {
        AriesLiteralValue result;
        bool hasNull = type.HasNull;
        bool isNull = !data;
        result.IsNull = isNull;
        switch( type.DataType.ValueType )
        {
            case AriesValueType::CHAR:
                if( type.DataType.Length == 1 )
                {
                    if( isNull )
                        result.Value = int8_t();
                    else
                    {
                        if( hasNull )
                        {
                            result.IsNull = !*data;
                            ++data;
                        }
                        result.Value = *data;
                    }
                }
                else
                {
                    if( isNull )
                        result.Value = string();
                    else
                    {
                        if( hasNull )
                        {
                            result.IsNull = !*data;
                            ++data;
                        }
                        result.Value = string( ( char* )data, type.DataType.Length );
                    }
                }
                break;
            case AriesValueType::INT8:
                if( isNull )
                    result.Value = int8_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *data;
                }
                break;
            case AriesValueType::BOOL:
                if( isNull )
                    result.Value = int8_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *data;
                }
                break;
            case AriesValueType::UINT8:
                if( isNull )
                    result.Value = uint8_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( uint8_t* )data;
                }
                break;
            case AriesValueType::INT16:
                if( isNull )
                    result.Value = int16_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( int16_t* )data;
                }
                break;
            case AriesValueType::UINT16:
                if( isNull )
                    result.Value = uint16_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( uint16_t* )data;
                }
                break;
            case AriesValueType::INT32:
                if( isNull )
                    result.Value = int32_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( int32_t* )data;
                }
                break;
            case AriesValueType::UINT32:
                if( isNull )
                    result.Value = uint32_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( uint32_t* )data;
                }
                break;
            case AriesValueType::INT64:
                if( isNull )
                    result.Value = int64_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( int64_t* )data;
                }
                break;
            case AriesValueType::UINT64:
                if( isNull )
                    result.Value = uint64_t();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( uint64_t* )data;
                }
                break;
            case AriesValueType::FLOAT:
                if( isNull )
                    result.Value = float();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( float* )data;
                }
                break;
            case AriesValueType::DOUBLE:
                if( isNull )
                    result.Value = double();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( double* )data;
                }
                break;
            case AriesValueType::DECIMAL:
                if( isNull )
                    result.Value = aries_acc::Decimal();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( aries_acc::Decimal* )data;
                }
                break;
            case AriesValueType::DATE:
                if( isNull )
                    result.Value = AriesDate();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( AriesDate* )data;
                }
                break;
            case AriesValueType::DATETIME:
                if( isNull )
                    result.Value = AriesDatetime();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( AriesDatetime* )data;
                }
                break;
            case AriesValueType::TIMESTAMP:
                if( isNull )
                    result.Value = AriesTimestamp();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( AriesTimestamp* )data;
                }
                break;
            case AriesValueType::TIME:
                if( isNull )
                    result.Value = AriesTime();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( AriesTime* )data;
                }
                break;
            case AriesValueType::YEAR:
                if( isNull )
                    result.Value = AriesYear();
                else
                {
                    if( hasNull )
                    {
                        result.IsNull = !*data;
                        ++data;
                    }
                    result.Value = *( AriesYear* )data;
                }
                break;
            default:
                //TODO need support other data types
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "getting Literal data for type: " + to_string( ( int ) type.DataType.ValueType ) );
                break;
        }
        return result;
    }

    std::string GenerateParamType( const AriesColumnType &param )
    {
        string type = GetValueTypeAsString( param );
//        if( param.HasNull && param.DataType.Length == 1 )
//        {
//            // nullable type
//            type = " nullable_type< " + type + " > ";
//        }

        if( param.HasNull )
        {
            // nullable type
            type = " nullable_type< " + type + " > ";
        }
        return type;
    }

    string GetTypeNameFromConstructor( const string& str )
    {
        regex r( "\\s*(.+?)\\(" );
        smatch m;
        string result;
        while( regex_search( str, m, r ) )
        {
            result = m[1];
            break;
        }
        return result;
    }

    string IntervalToString( interval_type interval )
    {
        string result;
        switch( interval )
        {
            case INTERVAL_YEAR:
                result = "INTERVAL_YEAR";
                break;
            case INTERVAL_QUARTER:
                result = "INTERVAL_QUARTER";
                break;
            case INTERVAL_MONTH:
                result = "INTERVAL_MONTH";
                break;
            case INTERVAL_WEEK:
                result = "INTERVAL_WEEK";
                break;
            case INTERVAL_DAY:
                result = "INTERVAL_DAY";
                break;
            case INTERVAL_HOUR:
                result = "INTERVAL_HOUR";
                break;
            case INTERVAL_MINUTE:
                result = "INTERVAL_MINUTE";
                break;
            case INTERVAL_SECOND:
                result = "INTERVAL_SECOND";
                break;
            case INTERVAL_MICROSECOND:
                result = "INTERVAL_MICROSECOND";
                break;
            case INTERVAL_YEAR_MONTH:
                result = "INTERVAL_YEAR_MONTH";
                break;
            case INTERVAL_DAY_HOUR:
                result = "INTERVAL_DAY_HOUR";
                break;
            case INTERVAL_DAY_MINUTE:
                result = "INTERVAL_DAY_MINUTE";
                break;
            case INTERVAL_DAY_SECOND:
                result = "INTERVAL_DAY_SECOND";
                break;
            case INTERVAL_HOUR_MINUTE:
                result = "INTERVAL_HOUR_MINUTE";
                break;
            case INTERVAL_HOUR_SECOND:
                result = "INTERVAL_HOUR_SECOND";
                break;
            case INTERVAL_MINUTE_SECOND:
                result = "INTERVAL_MINUTE_SECOND";
                break;
            case INTERVAL_DAY_MICROSECOND:
                result = "INTERVAL_DAY_MICROSECOND";
                break;
            case INTERVAL_HOUR_MICROSECOND:
                result = "INTERVAL_HOUR_MICROSECOND";
                break;
            case INTERVAL_MINUTE_MICROSECOND:
                result = "INTERVAL_MINUTE_MICROSECOND";
                break;
            case INTERVAL_SECOND_MICROSECOND:
                result = "INTERVAL_SECOND_MICROSECOND";
                break;
            case INTERVAL_LAST:
                result = "INTERVAL_LAST";
                break;
            default:
                //FIXME need support other intervals
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "unknown interval type: " + to_string( ( int ) interval ) );
                break;
        }
        return result;
    }

    string& ReplaceString( string& str, const string& old_value, const string& new_value )
    {
        for( string::size_type pos( 0 ); pos != string::npos; pos += new_value.length() )
        {
            if( ( pos = str.find( old_value, pos ) ) != string::npos )
                str.replace( pos, old_value.length(), new_value );
            else
                break;
        }
        return str;
    }

    AriesComparisonOpType SwapComparisonType( AriesComparisonOpType type )
    {
        AriesComparisonOpType result = type;
        switch( type )
        {
            case AriesComparisonOpType::GE:
                result = AriesComparisonOpType::LE;
                break;
            case AriesComparisonOpType::GT:
                result = AriesComparisonOpType::LT;
                break;
            case AriesComparisonOpType::LE:
                result = AriesComparisonOpType::GE;
                break;
            case AriesComparisonOpType::LT:
                result = AriesComparisonOpType::GT;
                break;
            default:
                ARIES_ASSERT( false, "unhandled AriesComparisonOpType" );
        }
        return result;
    }

    string GetAriesExprTypeName( AriesExprType type )
    {
        string result;
        switch( type )
        {
            case AriesExprType::INTEGER:
                result = "INTEGER";
                break;
            case AriesExprType::FLOATING:
                result = "FLOATING";
                break;
            case AriesExprType::DECIMAL:
                result = "DECIMAL";
                break;
            case AriesExprType::STRING:
                result = "STRING";
                break;
            case AriesExprType::DATE:
                result = "DATE";
                break;
            case AriesExprType::DATE_TIME:
                result = "DATE_TIME";
                break;
            case AriesExprType::TIME:
                result = "TIME";
                break;
            case AriesExprType::TIMESTAMP:
                result = "TIMESTAMP";
                break;
            case AriesExprType::COLUMN_ID:
                result = "COLUMN_ID";
                break;
            case AriesExprType::STAR:
                result = "STAR";
                break;
            case AriesExprType::AGG_FUNCTION:
                result = "AGG_FUNCTION";
                break;
            case AriesExprType::SQL_FUNCTION:
                result = "SQL_FUNCTION";
                break;
            case AriesExprType::ARRAY:
                result = "ARRAY";
                break;
            case AriesExprType::CALC:
                result = "CALC";
                break;
            case AriesExprType::NOT:
                result = "NOT";
                break;
            case AriesExprType::COMPARISON:
                result = "COMPARISON";
                break;
            case AriesExprType::LIKE:
                result = "LIKE";
                break;
            case AriesExprType::IN:
                result = "IN";
                break;
            case AriesExprType::NOT_IN:
                result = "NOT_IN";
                break;
            case AriesExprType::BETWEEN:
                result = "BETWEEN";
                break;
            case AriesExprType::EXISTS:
                result = "EXISTS";
                break;
            case AriesExprType::CASE:
                result = "CASE";
                break;
            case AriesExprType::AND_OR:
                result = "AND_OR";
                break;
            case AriesExprType::BRACKETS:
                result = "BRACKETS";
                break;
            case AriesExprType::TRUE_FALSE:
                result = "TRUE_FALSE";
                break;
            case AriesExprType::IF_CONDITION:
                result = "IF_CONDITION";
                break;
            case AriesExprType::IS_NOT_NULL:
                result = "IS_NOT_NULL";
                break;
            case AriesExprType::IS_NULL:
                result = "IS_NULL";
                break;
            case AriesExprType::DISTINCT:
                result = "DISTINCT";
                break;
            case AriesExprType::NULL_VALUE:
                result = "NULL_VALUE";
                break;
            case AriesExprType::INTERVAL:
                result = "INTERVAL";
                break;
            case AriesExprType::BUFFER:
                result = "BUFFER";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN EXPR TYPE: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetAriesComparisonOpTypeName( AriesComparisonOpType type )
    {
        string result;
        switch( type )
        {
            case AriesComparisonOpType::EQ:
                result = "EQ";
                break;
            case AriesComparisonOpType::NE:
                result = "NE";
                break;
            case AriesComparisonOpType::GT:
                result = "GT";
                break;
            case AriesComparisonOpType::LT:
                result = "LT";
                break;
            case AriesComparisonOpType::GE:
                result = "GE";
                break;
            case AriesComparisonOpType::LE:
                result = "LE";
                break;
            case AriesComparisonOpType::IN:
                result = "IN";
                break;
            case AriesComparisonOpType::NOTIN:
                result = "NOT IN";
                break;
            case AriesComparisonOpType::LIKE:
                result = "LIKE";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN COMP OP TYPE " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetAriesSqlFunctionTypeName( AriesSqlFunctionType type )
    {
        string result;
        switch( type )
        {
            case AriesSqlFunctionType::SUBSTRING:
                result = "SUBSTRING";
                break;
            case AriesSqlFunctionType::EXTRACT:
                result = "EXTRACT";
                break;
            case AriesSqlFunctionType::ST_VOLUMN:
                result = "ST_VOLUMN";
                break;
            case AriesSqlFunctionType::DATE:
                result = "DATE";
                break;
            case AriesSqlFunctionType::NOW:
                result = "NOW";
                break;
            case AriesSqlFunctionType::DATE_SUB:
                result = "DATE_SUB";
                break;
            case AriesSqlFunctionType::DATE_ADD:
                result = "DATE_ADD";
                break;
            case AriesSqlFunctionType::DATE_FORMAT:
                result = "DATE_FORMAT";
                break;
            case AriesSqlFunctionType::DATE_DIFF:
                result = "DATE_DIFF";
                break;
            case AriesSqlFunctionType::TIME_DIFF:
                result = "TIME_DIFF";
                break;
            case AriesSqlFunctionType::ABS:
                result = "ABS";
                break;
            case AriesSqlFunctionType::COUNT:
                result = "COUNT";
                break;
            case AriesSqlFunctionType::SUM:
                result = "SUM";
                break;
            case AriesSqlFunctionType::AVG:
                result = "AVG";
                break;
            case AriesSqlFunctionType::MAX:
                result = "MAX";
                break;
            case AriesSqlFunctionType::MIN:
                result = "MIN";
                break;
            case AriesSqlFunctionType::UNIX_TIMESTAMP:
                result = "UNIX_TIMESTAMP";
                break;
            case AriesSqlFunctionType::CAST:
                result = "CAST";
                break;
            case AriesSqlFunctionType::CONVERT:
                // result = "CONVERT";
                ARIES_ASSERT( 0, "Not support function convert" );
                break;
            case AriesSqlFunctionType::MONTH:
                result = "MONTH";
                break;
            case AriesSqlFunctionType::COALESCE:
                result = "COALESCE";
                break;
            case AriesSqlFunctionType::TRUNCATE:
                result = "TRUNCATE";
                break;
            case AriesSqlFunctionType::CONCAT:
                result = "CONCAT";
                break;
            case AriesSqlFunctionType::ANY_VALUE:
                result = "ANY_VALUE";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN SQL FUNCTION TYPE: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetAriesJoinTypeName( AriesJoinType type )
    {
        string result;
        switch( type )
        {
            case AriesJoinType::INNER_JOIN:
                result = "INNER JION";
                break;
            case AriesJoinType::LEFT_JOIN:
                result = "LEFT JION";
                break;
            case AriesJoinType::RIGHT_JOIN:
                result = "RIGHT JION";
                break;
            case AriesJoinType::FULL_JOIN:
                result = "FULL JION";
                break;
            case AriesJoinType::SEMI_JOIN:
                result = "SEMI JION";
                break;
            case AriesJoinType::ANTI_JOIN:
                result = "ANTI JION";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN JION TYPE: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetAriesAggFunctionTypeName( AriesAggFunctionType type )
    {
        string result;
        switch( type )
        {
            case AriesAggFunctionType::NONE:
                result = "NONE";
                break;
            case AriesAggFunctionType::COUNT:
                result = "COUNT";
                break;
            case AriesAggFunctionType::SUM:
                result = "SUM";
                break;
            case AriesAggFunctionType::AVG:
                result = "AVG";
                break;
            case AriesAggFunctionType::MAX:
                result = "MAX";
                break;
            case AriesAggFunctionType::MIN:
                result = "MIN";
                break;
            case AriesAggFunctionType::ANY_VALUE:
                result = "ANY_VALUE";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN AGG FUNCTION: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetAriesSetOpTypeName( AriesSetOpType type )
    {
        string result;
        switch( type )
        {
            case AriesSetOpType::UNION:
                result = "UNION";
                break;
            case AriesSetOpType::UNION_ALL:
                result = "UNION_ALL";
                break;
            case AriesSetOpType::INTERSECT:
                result = "INTERSECT";
                break;
            case AriesSetOpType::INTERSECT_ALL:
                result = "INTERSECT_ALL";
                break;
            case AriesSetOpType::EXCEPT:
                result = "EXCEPT";
                break;
            case AriesSetOpType::EXCEPT_ALL:
                result = "EXCEPT_ALL";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN SET OP TYPE: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetColumnTypeName( ColumnType type )
    {
        string result;
        switch( type )
        {
            case ColumnType::TINY_INT:
                result = "TINY_INT";
                break;
            case ColumnType::SMALL_INT:
                result = "SMALL_INT";
                break;
            case ColumnType::INT:
                result = "INT";
                break;
            case ColumnType::LONG_INT:
                result = "LONG_INT";
                break;
            case ColumnType::DECIMAL:
                result = "DECIMAL";
                break;
            case ColumnType::FLOAT:
                result = "FLOAT";
                break;
            case ColumnType::DOUBLE:
                result = "DOUBLE";
                break;
            case ColumnType::TEXT:
                result = "TEXT";
                break;
            case ColumnType::BOOL:
                result = "BOOL";
                break;
            case ColumnType::DATE:
                result = "DATE";
                break;
            case ColumnType::TIME:
                result = "TIME";
                break;
            case ColumnType::DATE_TIME:
                result = "DATE_TIME";
                break;
            case ColumnType::TIMESTAMP:
                result = "TIMESTAMP";
                break;
            case ColumnType::UNKNOWN:
                result = "UNKNOWN";
                break;
            case ColumnType::YEAR:
                result = "YEAR";
                break;
            case ColumnType::LIST:
                result = "LIST";
                break;
            default:
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN ColumnType: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string GetAriesLogicOpTypeName( AriesLogicOpType type )
    {
        string result;
        switch( type )
        {
            case AriesLogicOpType::AND:
                result = "AND";
                break;
            case AriesLogicOpType::OR:
                result = "OR";
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN AriesLogicOpType: " + to_string( ( int ) type ) );
                break;
        }
        return result;
    }

    string AriesBreeSearchOpTypeToString( AriesBtreeSearchOpType &opType )
    {
        switch( opType )
        {
            case AriesBtreeSearchOpType::EQ:
                return "EQ";
            case AriesBtreeSearchOpType::NE:
                return "NE";
            case AriesBtreeSearchOpType::GT:
                return "GT";
            case AriesBtreeSearchOpType::GE:
                return "GE";
            case AriesBtreeSearchOpType::LE:
                return "LE";
            case AriesBtreeSearchOpType::GTAndLT:
                return "GTAndLT";
            case AriesBtreeSearchOpType::GEAndLT:
                return "GEAndLT";
            case AriesBtreeSearchOpType::GTAndLE:
                return "GTAndLE";
            case AriesBtreeSearchOpType::GEAndLE:
                return "GEAndLE";
            case AriesBtreeSearchOpType::LIKE:
                return "LIKE";
            default:
            {
                string msg( "unexpected op: " );
                msg.append( std::to_string( ( int )opType ) );
                ARIES_ASSERT( 0, msg );
            }
        }
    }

    bool IsSupportNewMergeSort( const AriesColumnType& columnType )
    {
        bool bRet = true;
        if( columnType.DataType.ValueType == AriesValueType::COMPACT_DECIMAL
                || ( columnType.DataType.ValueType == AriesValueType::CHAR && columnType.DataType.Length > 1 ) )
            bRet = false;
        return bRet;
    }

    void DispatchTablesToMultiGpu( const vector< AriesTableBlockUPtr >& tables )
    {
        ARIES_ASSERT( !tables.empty(), "tables is empty! " );
        int deviceCount;
        cudaGetDeviceCount( &deviceCount );
        ARIES_ASSERT( deviceCount >= 0 && tables.size() <= size_t( deviceCount ), "table count is more than device count!!!" );
        vector< char > occuppiedDeviceFlags( deviceCount, 0 );
        vector< AriesTableBlock* > tablesForDispatch;
        int deviceId;
        for( const auto& table : tables )
        {
            deviceId = table->GetDeviceId();
            if( deviceId == cudaCpuDeviceId || occuppiedDeviceFlags[ deviceId ] )
            {
                tablesForDispatch.push_back( table.get() );
            }
            else
                occuppiedDeviceFlags[ deviceId ] = 1;
        }

        deviceId = 0;
        for( const auto t : tablesForDispatch )
        {
            for(; deviceId < deviceCount; ++deviceId )
            {
                if( !occuppiedDeviceFlags[ deviceId ] )
                {
                    t->SetDeviceId( deviceId );
                    occuppiedDeviceFlags[ deviceId ] = 1;
                    break;
                }
            }
        }
    }

    AriesNullValueProvider::AriesNullValueProvider()
    {

    }

    AriesNullValueProvider::~AriesNullValueProvider()
    {

    }

    void AriesNullValueProvider::Clear()
    {
        m_nullValues.clear();
    }
    
    int8_t* AriesNullValueProvider::GetNullValue( const AriesColumnType& type )
    {
        lock_guard< mutex > lock( m_mutex );
        AriesColumnType tmp = type;
        tmp.HasNull = true;
        size_t perItemSize = tmp.GetDataTypeSize();
        auto it = m_nullValues.find( perItemSize );
        if( it == m_nullValues.end() )
            it = m_nullValues.insert( { perItemSize, std::make_shared< AriesInt8Array >( perItemSize, true ) } ).first;
        return it->second->GetData();
    }
    void GetAriesColumnDataIteratorInfo( AriesColumnDataIterator& iter,
                                         AriesColumnDataIteratorHelper& iterHelper,
                                         const AriesTableBlockUPtr& tableBlock,
                                         int32_t columnId,
                                         AriesColumnType columnType,
                                         bool useDictIndex )
    {
#ifdef ARIES_PROFILE
        CPU_Timer t;
        t.begin();
#endif
        iter = AriesColumnDataIterator();

        iter.m_nullData = AriesNullValueProvider::GetInstance().GetNullValue( columnType );
        iter.m_perItemSize = columnType.GetDataTypeSize();

        auto colEncodeType = tableBlock->GetColumnEncodeType( columnId );
        if( colEncodeType == EncodeType::NONE )
            tableBlock->GetColumnBuffer( columnId );
        aries_engine::AriesColumnSPtr column;
        if ( tableBlock->IsColumnUnMaterilized( columnId ) )
        {
            auto columnReference = tableBlock->GetUnMaterilizedColumn( columnId );
            auto refferedColumn = columnReference->GetReferredColumn();
            switch ( colEncodeType )
            {
                case EncodeType::NONE:
                {
                    column = std::dynamic_pointer_cast< AriesColumn >( refferedColumn );

                    auto indicesArray = columnReference->GetIndices()->GetIndicesArray();
                    int indicesBlockCount = indicesArray.size();
                    auto indicesBuffers = make_shared< AriesManagedArray< int8_t* > >( indicesBlockCount );
                    int i = 0;
                    for( const auto& indice : indicesArray )
                    {
                        iter.m_indiceItemCount += indice->GetItemCount();
                        ( *indicesBuffers )[ i++ ] = ( int8_t* )indice->GetData();
                    }
                    indicesBuffers->PrefetchToGpu();
                    iter.m_indices = indicesBuffers->GetData();
                    iter.m_indiceValueType = AriesColumnType{ {AriesValueType::INT32}, false };
                    iter.m_indiceBlockCount = indicesBlockCount;

                    auto indiceBlockSizePrefixSum = GetPrefixSumOfBlockSize( columnReference->GetIndices()->GetBlockSizePsumArray() );
                    iter.m_indiceBlockSizePrefixSum = indiceBlockSizePrefixSum->GetData();

                    iterHelper.m_indiceBlockSizePrefixSumArray = indiceBlockSizePrefixSum;
                    iterHelper.m_indiceBlocks = indicesArray;
                    iterHelper.m_indiceBlockPtrs = indicesBuffers;
                    break;
                }

                case EncodeType::DICT:
                {
                    auto dictColumn = std::dynamic_pointer_cast< AriesDictEncodedColumn >( refferedColumn );

                    vector< AriesVariantIndicesArraySPtr > input;
                    auto oldIndices = dictColumn->GetIndices()->GetDataBuffer();
                    input.push_back( oldIndices );
                    auto newIndices = aries_acc::ShuffleColumns( input, columnReference->GetIndices()->GetIndices() )[ 0 ];
                    newIndices->PrefetchToGpu();
                    iterHelper.m_variantIndiceBlocks.push_back( newIndices );

                    vector< int64_t > tmpPsumArray;
                    tmpPsumArray.push_back( 0 );
                    tmpPsumArray.push_back( newIndices->GetItemCount() );
                    auto blockSizePrefixSum = GetPrefixSumOfBlockSize( tmpPsumArray );

                    if ( useDictIndex )
                    {
                        auto dataBlocks = make_shared< AriesManagedArray< int8_t* > >( 1 );
                        ( *dataBlocks )[ 0 ] = ( int8_t* )newIndices->GetData();
                        dataBlocks->PrefetchToGpu();
                        iterHelper.m_dataBlockPtrs = dataBlocks;
                        iterHelper.m_dataBlockSizePrefixSumArray = blockSizePrefixSum;

                        iter.m_data = dataBlocks->GetData();
                        iter.m_itemCount = newIndices->GetItemCount();
                        iter.m_valueType = newIndices->GetDataType();
                        iter.m_perItemSize = newIndices->GetDataType().GetDataTypeSize();
                        iter.m_dataBlockSizePrefixSum = blockSizePrefixSum->GetData();
                        iter.m_dataBlockCount = 1;
                        iter.m_indices = nullptr;

                        return;
                    }
                    else
                    {
                        column = make_shared< AriesColumn >();
                        column->AddDataBuffer( dictColumn->GetDictDataBuffer() );

                        iterHelper.m_indiceBlockSizePrefixSumArray = blockSizePrefixSum;
                        iter.m_indiceBlockSizePrefixSum = blockSizePrefixSum->GetData();

                        auto indicesBuffers = make_shared< AriesManagedArray< int8_t* > >( 1 );
                        iterHelper.m_indiceBlockPtrs = indicesBuffers;
                        ( *indicesBuffers )[ 0 ] = newIndices->GetData();
                        indicesBuffers->PrefetchToGpu();
                        iter.m_indices = indicesBuffers->GetData();
                        iter.m_indiceItemCount = newIndices->GetItemCount();
                        iter.m_indiceValueType = newIndices->GetDataType();
                        iter.m_indiceBlockCount = 1;
                    }

                    break;
                }
            }
        }
        else
        {
            switch ( colEncodeType )
            {
                case EncodeType::NONE:
                {
                    column = tableBlock->GetMaterilizedColumn( columnId );
                    iter.m_indices = nullptr;
                    break;
                }

                case EncodeType::DICT:
                {
                    auto dictColumn = tableBlock->GetDictEncodedColumn( columnId );
                    column = make_shared< AriesColumn >();

                    if ( useDictIndex )
                    {
                        auto indicesArray = dictColumn->GetIndices()->GetDataBuffers();
                        auto dataBlocks = make_shared< AriesManagedArray< int8_t* > >( indicesArray.size() );

                        iterHelper.m_dataBlockPtrs = dataBlocks;
                        int i = 0;
                        for ( const auto& indices : indicesArray )
                        {
                            iter.m_itemCount += indices->GetItemCount();
                            indices->PrefetchToGpu();
                            ( *dataBlocks )[ i++ ] = indices->GetData();
                        }
                        dataBlocks->PrefetchToGpu();
                        iter.m_data = dataBlocks->GetData();
                        iter.m_valueType = dictColumn->GetIndices()->GetColumnType();
                        iter.m_perItemSize = dictColumn->GetIndices()->GetColumnType().GetDataTypeSize();
                        iter.m_dataBlockCount = indicesArray.size();

                        auto dataBlockSizePrefixSum = GetPrefixSumOfBlockSize( dictColumn->GetIndices()->GetBlockSizePsumArray() );
                        iterHelper.m_dataBlockSizePrefixSumArray = dataBlockSizePrefixSum;

                        iter.m_dataBlockSizePrefixSum = dataBlockSizePrefixSum->GetData();

                        iter.m_indices = nullptr;

                        return;
                    }
                    else
                    {
                        column->AddDataBuffer( dictColumn->GetDictDataBuffer() );

                        auto indicesArray = dictColumn->GetIndices()->GetDataBuffers();
                        iterHelper.m_variantIndiceBlocks = indicesArray;

                        int indicesBlockCount = indicesArray.size();
                        auto indicesBuffers = make_shared< AriesManagedArray< int8_t* > >( indicesBlockCount );
                        iterHelper.m_indiceBlockPtrs = indicesBuffers;

                        int i = 0;
                        for( const auto& indice : indicesArray )
                        {
                            iter.m_indiceItemCount += indice->GetItemCount();
                            indice->PrefetchToGpu();
                            ( *indicesBuffers )[ i++ ] = indice->GetData();
                        }
                        indicesBuffers->PrefetchToGpu();
                        iter.m_indices = indicesBuffers->GetData();
                        iter.m_indiceValueType = indicesArray[ 0 ]->GetDataType();
                        iter.m_indiceBlockCount = indicesBlockCount;

                        auto indiceBlockSizePrefixSum = GetPrefixSumOfBlockSize( dictColumn->GetIndices()->GetBlockSizePsumArray() );
                        iterHelper.m_indiceBlockSizePrefixSumArray = indiceBlockSizePrefixSum;
                        iter.m_indiceBlockSizePrefixSum = indiceBlockSizePrefixSum->GetData();
                    }
                    break;
                }
            }
        }

        auto dataBuffers = column->GetDataBuffers();
        iterHelper.m_dataBlocks = dataBuffers;

        auto dataBlocks = make_shared< AriesManagedArray< int8_t* > >( dataBuffers.size() );

        iterHelper.m_dataBlockPtrs = dataBlocks;
        int i = 0;
#ifdef ARIES_PROFILE
        t.end();
#endif
        for ( const auto& buffer : dataBuffers )
        {
            iter.m_itemCount += buffer->GetItemCount();
            buffer->PrefetchToGpu();
            ( *dataBlocks )[ i++ ] = buffer->GetData();
        }
        dataBlocks->PrefetchToGpu();
        iter.m_data = dataBlocks->GetData();
        iter.m_valueType = column->GetColumnType();
        iter.m_dataBlockCount = dataBuffers.size();

        auto dataBlockSizePrefixSum = GetPrefixSumOfBlockSize( column->GetBlockSizePsumArray() );
        iterHelper.m_dataBlockSizePrefixSumArray = dataBlockSizePrefixSum;

        iter.m_dataBlockSizePrefixSum = dataBlockSizePrefixSum->GetData();
#ifdef ARIES_PROFILE
        LOG( INFO )<< "GetAriesColumnDataIteratorInfo:" << t.end() << endl;
#endif
    }

    void CheckDecimalPrecision( const string& string_value )
    {
        // check precision
        const char* pStr = string_value.data();
        if ( '-' == *pStr )
            ++pStr;
        const char *intgend = strchr( pStr, '.');
        auto strLen = strlen( pStr );
        int intgLen = intgend ? intgend - pStr : strLen;
        int fracLen = intgend ? strLen - intgLen - 1 : 0;
        if ( fracLen > SUPPORTED_MAX_SCALE )
            ARIES_EXCEPTION( ER_TOO_BIG_SCALE, fracLen, string_value.data(),
                             static_cast<ulong>( SUPPORTED_MAX_SCALE ) );
        if ( intgLen + fracLen > SUPPORTED_MAX_PRECISION )
            ARIES_EXCEPTION( ER_TOO_BIG_PRECISION, intgLen + fracLen, string_value.data(),
                             static_cast<ulong>( SUPPORTED_MAX_PRECISION ) );
    }

    void CheckDecimalError( const string& strValue, aries_acc::Decimal& value, const string& colName, size_t i )
    {
        int errorCode;
        string errorMsg;
        auto decError = value.GetError();
        if ( decError == ERR_STR_2_DEC )
        {
            errorCode = FormatTruncWrongValueError( colName, strValue, i, "decimal", errorMsg );
            if ( STRICT_MODE )
                ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
        if ( decError == ERR_OVER_FLOW )
        {
            errorCode = FormatOutOfRangeValueError( colName, i, errorMsg );
            if ( STRICT_MODE )
                ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
            LOG(WARNING) << "Convert data warning: " << errorMsg;
        }
    }
END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
