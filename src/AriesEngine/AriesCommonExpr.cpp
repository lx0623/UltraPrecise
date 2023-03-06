//
// Created by 胡胜刚 on 2019-07-22.
//
#include <random>
#include "AriesCommonExpr.h"
#include "AriesAssert.h"
#include "AriesUtil.h"
#include "CudaAcc/AriesEngineException.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "datatypes/AriesTimeCalc.hxx"
#include "datatypes/AriesDatetimeTrans.h"
#include <queue>

static const char *COLUMN_NAME_PARAM_FORMAT_STR = "columnId_%s_";

BEGIN_ARIES_ENGINE_NAMESPACE

    std::map< AriesCalculatorOpType, string > AriesCommonExpr::calcOpToStr = { { AriesCalculatorOpType::ADD, "+" },
            { AriesCalculatorOpType::SUB, "-" }, { AriesCalculatorOpType::MUL, "*" }, { AriesCalculatorOpType::DIV, "/" }, {
                    AriesCalculatorOpType::MOD, "%" } };

    std::map< AriesAggFunctionType, string > AriesCommonExpr::aggFuncToStr = { { AriesAggFunctionType::COUNT, "count" }, { AriesAggFunctionType::SUM,
            "sum" }, { AriesAggFunctionType::AVG, "avg" }, { AriesAggFunctionType::MAX, "max" }, { AriesAggFunctionType::MIN, "min" } };

    AriesCommonExprUPtr AriesCommonExpr::Clone() const
    {
        AriesCommonExprUPtr result = make_unique < AriesCommonExpr > ( type, content, value_type );
        for( const auto& child : children )
            result->AddChild( child->Clone() );
        return result;
    }

    void AriesCommonExpr::SetId( int& id )
    {
        m_id = id;
        for( const auto& child : children )
            child->SetId( ++id );
    }

    int AriesCommonExpr::GetId() const
    {
        assert( m_id != -1 );
        return m_id;
    }

    AriesCommonExprUPtr AriesCommonExpr::Create( AriesExprType type, AriesExpressionContent content, AriesColumnType value_type )
    {
        return AriesCommonExprUPtr( new AriesCommonExpr( type, content, value_type ) );
    }

    AriesCommonExpr::AriesCommonExpr( AriesExprType _type, AriesExpressionContent _content, AriesColumnType _value_type )
            : type( _type ), useDictIndex( false ), content( _content ), value_type( _value_type ), m_id( -1 )
    {
    }

    AriesCommonExpr::~AriesCommonExpr()
    {
        children.clear();
    }

    void AriesCommonExpr::AddChild( AriesCommonExprUPtr child )
    {
        children.emplace_back( std::move( child ) );
    }

    int AriesCommonExpr::GetChildrenCount() const
    {
        return children.size();
    }

    const AriesCommonExprUPtr &AriesCommonExpr::GetChild( int index ) const
    {
        ARIES_ASSERT( index >= 0 && std::size_t( index ) < children.size(), "index: " + to_string( index ) + ", children.size(): " + to_string( children.size() ) );
        return children[ index ];
    }

    AriesExprType AriesCommonExpr::GetType() const
    {
        return type;
    }

    void AriesCommonExpr::SetType( AriesExprType type )
    {
        AriesCommonExpr::type = type;
    }

    const AriesExpressionContent &AriesCommonExpr::GetContent() const
    {
        return content;
    }

    void AriesCommonExpr::SetContent( const AriesExpressionContent &content )
    {
        AriesCommonExpr::content = content;
    }

    AriesColumnType AriesCommonExpr::GetValueType() const
    {
        return value_type;
    }

    void AriesCommonExpr::SetValueType( AriesColumnType value_type )
    {
        AriesCommonExpr::value_type = value_type;
    }

    bool AriesCommonExpr::IsDistinct() const
    {
        bool result = false;
        for( const auto &child : children )
        {
            if( child->type == AriesExprType::DISTINCT )
            {
                result = boost::get< bool >( child->content );
                break;
            }
        }
        return result;
    }

    void AriesCommonExpr::SetUseDictIndex( bool b )
    {
        useDictIndex = b;
    }

    bool AriesCommonExpr::IsUseDictIndex() const
    {
        return useDictIndex;
    }

    static bool contains_decimal( const AriesCommonExprUPtr& expr )
    {
        if ( expr->GetValueType().DataType.ValueType == AriesValueType::DECIMAL || expr->GetValueType().DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            return true;
        }

        for ( int i = 0; i < expr->GetChildrenCount(); i++ )
        {
            const auto& child = expr->GetChild( i );
            if ( contains_decimal( child ) )
            {
                return true;
            }
        }

        return false;
    }

    static int search_string( const string& str, const string& target )
    {
        int count = 0;
        for( string::size_type pos( 0 ); pos != string::npos; pos += target.length() )
        {
            if( ( pos = str.find( target, pos ) ) != string::npos )
                count++;
            else
                break;
        }
        return count;
    }

    std::string AriesCommonExpr::StringForDynamicCode( std::map< string, AriesCommonExprUPtr > &aggFunctions,
                                                       vector< AriesDynamicCodeParam >& ariesParams,
                                                       vector< AriesDataBufferSPtr >& constValues,
                                                       vector< AriesDynamicCodeComparator >& ariesComparators )
    {
        assert( m_id >= 0 );
        aggFunctions.clear();
        ariesParams.clear();
        constValues.clear();
        ariesComparators.clear();
        int seq = 0;
        set< AriesDynamicCodeParam > tmpParams;
        if ( type == AriesExprType::AND_OR )
        {
            auto left_contains_decimal = contains_decimal( children[ 0 ] );
            auto right_contains_decimal = contains_decimal( children[ 1 ] );

            auto logic_type = static_cast< AriesLogicOpType >( boost::get< int >( content ) );
            std::string calcExpr = "";
            auto left_dynamic_code = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, tmpParams, constValues, ariesComparators );
            auto right_dynamic_code = children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, tmpParams, constValues, ariesComparators );
            bool need_reverse = contains_decimal( children[ 0 ] ) && !contains_decimal( children[ 1 ] );

            std::vector< AriesDynamicCodeParam > left_params, right_params;
            for ( const auto& param : tmpParams )
            {
                auto left_count = search_string( left_dynamic_code, param.ParamName );
                auto right_count = search_string( right_dynamic_code, param.ParamName );

                if ( ( left_count + right_count ) < 2 )
                {
                    continue;
                }

                if ( need_reverse )
                {
                    if ( right_count > 0 )
                    {
                        right_params.emplace_back( param );
                    }
                    else
                    {
                        left_params.emplace_back( param );
                    }
                }
                else
                {
                    if ( left_count > 0 )
                    {
                        left_params.emplace_back( param );
                    }
                    else
                    {
                        right_params.emplace_back( param );
                    }
                }

            }

            if ( left_contains_decimal == right_contains_decimal )
            {
                if ( left_params.empty() && right_params.empty() )
                {
                    constValues.clear();
                    ariesComparators.clear();
                    aggFunctions.clear();
                    auto result = stringForDynamicCodeInternal( aggFunctions, seq, tmpParams, constValues, ariesComparators );
                    ariesParams.assign( tmpParams.cbegin(), tmpParams.cend() );
                    return result;
                }
            }

            if ( need_reverse )
            {
                for ( const auto& param : right_params )
                {
                    auto prefix = param.ColumnIndex > 0 ? "left_" : "right_";
                    auto tmp_param_name = prefix + std::to_string( abs( param.ColumnIndex ) );
                    calcExpr += "            auto " + tmp_param_name + " = "  + param.ParamName + ";\n";
                    ReplaceString( right_dynamic_code, param.ParamName, tmp_param_name );
                    ReplaceString( left_dynamic_code, param.ParamName, tmp_param_name );
                }
                calcExpr += "            AriesBool and_or_expr_result = " + right_dynamic_code + ";\n";
                if ( logic_type == AriesLogicOpType::OR )
                {
                    calcExpr += "            if ( !and_or_expr_result )\n";
                }
                else
                {
                    calcExpr += "            if ( and_or_expr_result )\n";
                }

                calcExpr += "            {\n";
                for ( const auto& param : left_params )
                {
                    auto prefix = param.ColumnIndex > 0 ? "left_" : "right_";
                    auto tmp_param_name = prefix + std::to_string( abs( param.ColumnIndex ) );
                    calcExpr += "                auto " + tmp_param_name + " = "  + param.ParamName + ";\n";
                    ReplaceString( left_dynamic_code, param.ParamName, tmp_param_name );
                }
                calcExpr += "                and_or_expr_result = " + left_dynamic_code + ";\n";
                calcExpr += "            }\n";
            }
            else
            {
                for ( const auto& param : left_params )
                {
                    auto prefix = param.ColumnIndex > 0 ? "left_" : "right_";
                    auto tmp_param_name = prefix + std::to_string( abs( param.ColumnIndex ) );
                    calcExpr += "            auto " + tmp_param_name + " = "  + param.ParamName + ";\n";
                    ReplaceString( left_dynamic_code, param.ParamName, tmp_param_name );
                    ReplaceString( right_dynamic_code, param.ParamName, tmp_param_name );
                }
                calcExpr += "            AriesBool and_or_expr_result = " + left_dynamic_code + ";\n";
                if ( logic_type == AriesLogicOpType::OR )
                {
                    calcExpr += "            if ( !and_or_expr_result )\n";
                }
                else
                {
                    calcExpr += "            if ( and_or_expr_result )\n";
                }
                calcExpr += "            {\n";
                for ( const auto& param : right_params )
                {
                    auto prefix = param.ColumnIndex > 0 ? "left_" : "right_";
                    auto tmp_param_name = prefix + std::to_string( abs( param.ColumnIndex ) );
                    calcExpr += "                auto " + tmp_param_name + " = "  + param.ParamName + ";\n";
                    ReplaceString( right_dynamic_code, param.ParamName, tmp_param_name );
                }
                calcExpr += "                and_or_expr_result = " + right_dynamic_code + ";\n";
                calcExpr += "            }\n";
            }
            ariesParams.insert( ariesParams.end(), tmpParams.begin(), tmpParams.end() );
            return "[&]() {\n" + calcExpr + "\n            return and_or_expr_result; }()";
        }
        else
        {
            int ansDecLength = value_type.DataType.AdaptiveLen;
            auto result = stringForDynamicCodeInternal( aggFunctions, seq, tmpParams, constValues, ariesComparators, 0, ansDecLength );
            ariesParams.assign( tmpParams.cbegin(), tmpParams.cend() );
            return result;
        }
    }

    std::string AriesCommonExpr::StringForXmpDynamicCode( std::map< string, AriesCommonExprUPtr > &aggFunctions,
                                                       vector< AriesDynamicCodeParam >& ariesParams,
                                                       vector< AriesDataBufferSPtr >& constValues,
                                                       vector< AriesDynamicCodeComparator >& ariesComparators,
                                                       int ansLEN,
                                                       int ansTPI )
    {
        assert( m_id >= 0 );
        aggFunctions.clear();
        ariesParams.clear();
        constValues.clear();
        ariesComparators.clear();
        int seq = 0;
        vector< AriesDynamicCodeParam > tmpParams;
        set< int > interVar;    // 避免重复的中间变量被多次定义
        string result = "";
        stringForXmpDynamicCodeInternal( aggFunctions, seq, tmpParams, constValues, interVar, ariesComparators, ansLEN, ansTPI, result);
        ariesParams.assign( tmpParams.cbegin(), tmpParams.cend() );
        return result;
    }

    std::string AriesCommonExpr::makeNullLiteral( AriesColumnType type ) const
    {
        ARIES_ASSERT( type.HasNull, "type.HasNull is False" );
        std::string value = GenerateParamType( type );
        value += "( 0, ";
        switch( type.DataType.ValueType )
        {
            case AriesValueType::DECIMAL:
                value += "Decimal() ) ";
                break;
            case AriesValueType::DATE:
                value += "AriesDate() ) ";
                break;
            case AriesValueType::DATETIME:
                value += "AriesDatetime() ) ";
                break;
            case AriesValueType::TIMESTAMP:
                value += "AriesTimestamp() ) ";
                break;
            case AriesValueType::YEAR:
                value += "AriesYear() ) ";
                break;
            case AriesValueType::CHAR:
                value += "'0' ) ";
                break;
            default:
                value += "0 ) ";
                break;
        }
        return value;
    }

    AriesDataBufferSPtr ConstExprContentToDataBuffer( const AriesExprContent& content, size_t tupleNum, int ansDecLength )
    {
        AriesDataBufferSPtr buffer;
        if( boost::get< int32_t >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< int32_t >( content ), tupleNum );
        }
        else if( boost::get< int64_t >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< int64_t >( content ), tupleNum );
        }
        else if( boost::get< double >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< double >( content ), tupleNum );
        }
        else if( boost::get< std::string >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< std::string >( content ), tupleNum );
        }
        else if( boost::get< aries_acc::Decimal >( &content ) )
        {
            // 常量传过来是 decimal 
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< aries_acc::Decimal >( content ), tupleNum, ansDecLength, false, false );
        }
        else if( boost::get< aries_acc::AriesDate >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< aries_acc::AriesDate >( content ), tupleNum );
        }
        else if( boost::get< aries_acc::AriesDatetime >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< aries_acc::AriesDatetime >( content ), tupleNum );
        }
        else if( boost::get< aries_acc::AriesTime >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< aries_acc::AriesTime >( content ), tupleNum );
        }
        else if( boost::get< aries_acc::AriesYear >( &content ) )
        {
            buffer = aries_acc::CreateDataBufferWithValue( boost::get< aries_acc::AriesYear >( content ), tupleNum );
        }
        else
        {
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "column type " + string( content.type().name() ) );
        }
        return buffer;

    }

    std::string AriesCommonExpr::contentToString( set< AriesDynamicCodeParam >& ariesParams,
                                                  vector< AriesDataBufferSPtr >& constValues,
                                                  bool printConstAsLiteral,
                                                  int ansDecLength ) const
    {
        string result;
        switch( type )
        {
            case AriesExprType::INTEGER:
            {
                if( value_type.DataType.ValueType == AriesValueType::INT64 )
                {
                    ARIES_ASSERT( boost::get< int64_t >( &content ), "content type: " + string( content.type().name() ) );
                    if ( printConstAsLiteral ) // for substring
                        result = std::to_string( boost::get< int64_t >( content ) );
                    else
                    {
                        result = "( *( int64_t* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                        constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                    }
                }
                else
                {
                    ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                    if ( printConstAsLiteral )
                        result = std::to_string( boost::get< int >( content ) );
                    else
                    {
                        result = "( *( int32_t* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                        constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                    }
                }
                break;
            }
            case AriesExprType::FLOATING:
            {
                ARIES_ASSERT( boost::get< double >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                    result = std::to_string( boost::get< double >( content ) );
                else
                {
                    result = "( *( double* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                }
                break;
            }
            case AriesExprType::STRING:
            {
                ARIES_ASSERT( boost::get< std::string >( &content ), "content type: " + string( content.type().name() ) );
                auto strContent = boost::get< string >( content );
                if ( 1 == strContent.size() )
                    result = "'" +  strContent + "'";
                else
                    result = "\"" +  strContent + "\"";
                break;
            }
            case AriesExprType::STAR:
            case AriesExprType::INTERVAL:
            {
                ARIES_ASSERT( boost::get< std::string >( &content ), "content type: " + string( content.type().name() ) );
                result = boost::get< string >( content );
                break;
            }
            case AriesExprType::COLUMN_ID:
            {
                //"columnId_%s"
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                int columnId = boost::get< int >( content );
                string columnParamName;
                if( columnId > 0 )
                    columnParamName = "left_";
                else
                {
                    columnId = -columnId;
                    columnParamName = "right_";
                }
                columnParamName += std::to_string( columnId );
                char buf[ 64 ];
                sprintf( buf, COLUMN_NAME_PARAM_FORMAT_STR, columnParamName.c_str() );
                ariesParams.insert( AriesDynamicCodeParam { boost::get< int >( content ), buf, value_type, useDictIndex } );
                result = buf;
                break;
            }
            case AriesExprType::CALC:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                result = AriesCommonExpr::calcOpToStr[ static_cast< AriesCalculatorOpType >( boost::get< int >( content ) ) ];
                break;
            }
            case AriesExprType::AGG_FUNCTION:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                result = AriesCommonExpr::aggFuncToStr[ static_cast< AriesAggFunctionType >( boost::get< int >( content ) ) ];
                break;
            }
            case AriesExprType::SQL_FUNCTION:
            {
                //we handle sql functions in stringForSqlFunctions function
                break;
            }
            case AriesExprType::AND_OR:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                result = " " + LogicOpToString( static_cast< AriesLogicOpType >( boost::get< int >( content ) ) ) + " ";
                break;
            }
            case AriesExprType::COMPARISON:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                result = " " + ComparisonOpToString( static_cast< AriesComparisonOpType >( boost::get< int >( content ) ) ) + " ";
                break;
            }
            case AriesExprType::DECIMAL:
            {
                ARIES_ASSERT( boost::get< aries_acc::Decimal >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    char value[ 64 ];
                    aries_acc::Decimal dec = boost::get< aries_acc::Decimal >( content );
                    result = " Decimal( \"";
                    result += dec.GetDecimal( value );
                    result += "\" ) ";
                }
                else
                {
                    // 此处我们将 Decimal 常量 重新 构造成 AriesDecimal 常量
                    result = "( *( AriesDecimal<" + to_string(ansDecLength)+ ">* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    // 此处将 常量 放置在了 constvalue 中
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1, ansDecLength) );
                }

                break;
            }
            case AriesExprType::DATE:
            {
                ARIES_ASSERT( boost::get< aries_acc::AriesDate >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    aries_acc::AriesDate date = boost::get< aries_acc::AriesDate >( content );
                    result = " AriesDate( ";
                    result += std::to_string( date.getYear() );
                    result += ", ";
                    result += std::to_string( date.getMonth() );
                    result += ", ";
                    result += std::to_string( date.getDay() );
                    result += " ) ";
                }
                else
                {
                    result = "( *( AriesDate* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                }

                break;
            }
            case AriesExprType::DATE_TIME:
            {
                ARIES_ASSERT( boost::get< aries_acc::AriesDatetime >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    aries_acc::AriesDatetime date = boost::get< aries_acc::AriesDatetime >( content );
                    result = " AriesDatetime( ";
                    result += std::to_string( date.getYear() );
                    result += ", ";
                    result += std::to_string( date.getMonth() );
                    result += ", ";
                    result += std::to_string( date.getDay() );
                    result += ", ";
                    result += std::to_string( date.getHour() );
                    result += ", ";
                    result += std::to_string( date.getMinute() );
                    result += ", ";
                    result += std::to_string( date.getSecond() );
                    result += ", ";
                    result += std::to_string( date.getMicroSec() );
                    result += " ) ";
                }
                else
                {
                    result = "( *( AriesDatetime* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                }

                break;
            }
            case AriesExprType::TIME:
            {
                ARIES_ASSERT( boost::get< aries_acc::AriesTime >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    aries_acc::AriesTime time = boost::get< aries_acc::AriesTime >( content );
                    result = " AriesTime( ";
                    result += std::to_string( time.sign );
                    result += ", ";
                    result += std::to_string( time.hour );
                    result += ", ";
                    result += std::to_string( time.minute );
                    result += ", ";
                    result += std::to_string( time.second );
                    result += ", ";
                    result += std::to_string( time.second_part );
                    result += " ) ";
                }
                else
                {
                    result = "( *( AriesTime* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                }

                break;
            }
            case AriesExprType::TIMESTAMP:
            {
                ARIES_ASSERT( boost::get< aries_acc::AriesTimestamp >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    aries_acc::AriesTimestamp timestamp = boost::get< aries_acc::AriesTimestamp >( content );
                    result = " AriesTimestamp( ";
                    result += std::to_string( timestamp.getTimeStamp() );
                    result += " ) ";
                }
                else
                {
                    result = "( *( AriesTimestamp* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                }
                break;
            }
            case AriesExprType::YEAR:
            {
                ARIES_ASSERT( boost::get< aries_acc::AriesYear >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    aries_acc::AriesYear year = boost::get< aries_acc::AriesYear >( content );
                    result = " AriesYear( ";
                    result += std::to_string( year.getYear() );
                    result += " ) ";
                }
                else
                {
                    result = "( *( AriesYear* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1 ) );
                }
                break;
            }
            case AriesExprType::NULL_VALUE:
            {
                result = makeNullLiteral( value_type );
                break;
            }
            case AriesExprType::TRUE_FALSE:
            {
                ARIES_ASSERT( boost::get< bool >( &content ), "content type: " + string( content.type().name() ) );
                result = boost::get< bool >( content ) ? "(1)" : "(0)";
                break;
            }
            default:
                //for other expr types, nothing to do
                break;
        }
        return result;
    }

    // 获取 该节点 生成的 string 代码
    std::string AriesCommonExpr::contentToXmpString( vector< AriesDynamicCodeParam >& ariesParams,
                                                  vector< AriesDataBufferSPtr >& constValues,
                                                  int ansLIMBS,
                                                  bool printConstAsLiteral ) const
    {
        string result = "";
        switch( type )
        {
            case AriesExprType::COLUMN_ID:
            {
                //"columnId_%s" 如果是 COLUMN 则需要生成从 compact 到 decimal 过程的代码
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                // int columnId = boost::get< int >( content );
                //通过 varIndexInKernel 将该变量的名称获取出来
                string str = to_string(varIndexInKernel);
                string str_val = "var_"+ str;
                string str_index = to_string(ariesParams.size());
                int inAriesParams = false;
                for(auto i=ariesParams.begin() ;i!=ariesParams.end(); i++){
                    // 其他类型的也可以使用此方法 暂时没有添加
                    if( (*i).ColumnIndex == boost::get< int >( content ) && value_type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL){
                        inAriesParams = true;
                        result += "       uint32_t " + str_val + "["+ to_string(ansLIMBS) +"] = {0};\n";
                        result += "       uint8_t " + str_val + "_sign = 0;\n";
                        if( value_reverse)
                            result += "       var_" + str + "_sign = !" + (*i).ParamName + "_sign;\n";
                        else
                            result += "       var_" + str + "_sign = "+ (*i).ParamName +"_sign;\n";
                        for(int j=0; j<ansLIMBS; j++){
                            result += "       " + str_val+"["+ to_string(j) +"] = "+ (*i).ParamName +"["+ to_string(j) +"];\n";
                        }
                        break;
                    }
                }
                if(inAriesParams == false){
                    ariesParams.push_back( AriesDynamicCodeParam { boost::get< int >( content ), str_val, value_type, useDictIndex } );
                    if( value_type.DataType.ValueType ==  AriesValueType::COMPACT_DECIMAL && inAriesParams == false){
                        // 如果类型为 compact_decimal 则需要展开成 decimal
                        int len_var = aries_acc::GetDecimalRealBytes(value_type.DataType.Precision, value_type.DataType.Scale);
                        // per_size 为 ansLIMBS *sizeof(uint32_t)
                        uint32_t per_size = ansLIMBS*sizeof(uint32_t);
                        uint32_t div_var = len_var / per_size;
                        uint32_t mod_var = len_var % per_size;
                        string str_div_var = to_string(div_var);
                        string str_per_size = to_string(per_size);
                        result ="       char *var_" + str + "_temp = (char *)(input["+ str_index +"][index]);\n" +
                                "       var_"+ str + "_temp += " + to_string(len_var-1) + ";\n" +
                                "       char c_" + str + "= *" + str_val + "_temp;\n" +
                                "       var_" + str + "_sign = GET_SIGN_FROM_BIT(c_" + str + ");\n";
                                if( value_reverse){
                                    result += "       " + str_val + "_sign = !" + str_val + "_sign;\n";
                                }
                                if(mod_var == 0){
                                    result +=   "       if(group_thread < " + str_div_var + "){\n"
                                                "              aries_memcpy("+ str_val +", ((CompactDecimal*)( input["+ str_index +"][index] )) + group_thread * " + str_per_size + ", " + str_per_size + ");\n"
                                                "       }\n"
                                                "       if(group_thread == " + str_div_var + "-1){\n"
                                                "              char *inner_temp = (char *)("+ str_val +");\n"
                                                "              inner_temp += " + str_per_size + " - 1;\n"
                                                "              *inner_temp = *inner_temp & 0x7f;\n"
                                                "       }\n";
                                }
                                else{
                                    string str_mod_var = to_string(mod_var);
                                    result +=   "       if(group_thread < " + str_div_var + "){\n"
                                                "              aries_memcpy("+ str_val +", ((CompactDecimal*)( input["+ str_index +"][index] )) + group_thread * " + str_per_size + ", " + str_per_size + ");\n"
                                                "       }\n"
                                                "       if(group_thread == " + str_div_var + "){\n"
                                                "              aries_memcpy("+ str_val +", ((CompactDecimal*)( input["+ str_index +"][index] )) + group_thread * " + str_per_size + ", " + str_mod_var + ");\n"
                                                "              char *inner_temp = (char *)("+ str_val +");\n"
                                                "              inner_temp += " + str_mod_var + " - 1;\n"
                                                "              *inner_temp = *inner_temp & 0x7f;\n"
                                                "       }\n";
                                }
                    }
                    else if( value_type.DataType.ValueType ==  AriesValueType::INT32 || value_type.DataType.ValueType ==  AriesValueType::INT16 || value_type.DataType.ValueType ==  AriesValueType::INT8 ){
                        // 如果类型为 uint32_t 范围内的有符号数
                        string str_type = GetValueTypeAsString(value_type);
                        result =    "       if(group_thread == 0){\n"
                                    "              " + str_type +" "+ str_val + "_data = *( ( "+ str_type +"* )( input["+ str_index +"][index] ) );\n"
                                    "              if(" + str_val + "_data < 0){\n"
                                    "                     " + str_val + "_data = -" + str_val + "_data;\n"
                                    "                     var_" + str + "_sign = 1;\n"
                                    "              }\n" +
                                    "              "+ str_val +"[0] = "+ str_val + "_data;\n"
                                    "       }\n";
                    }
                    else if( value_type.DataType.ValueType ==  AriesValueType::UINT32 || value_type.DataType.ValueType ==  AriesValueType::UINT16 || value_type.DataType.ValueType ==  AriesValueType::UINT8){
                        // 如果类型为 uint32_t 范围内的无符号数
                        string str_type = GetValueTypeAsString(value_type);
                        result =    "       if(group_thread == 0){\n"
                                    "              "+ str_val +"[0] = *( ( "+ str_type +"* )( input["+ str_index +"][index] ) );\n"
                                    "       }\n";
                    }
                    else if( value_type.DataType.ValueType == AriesValueType::INT64){
                        // 如果类型为 uint64_t 范围内的有符号数
                        // 如果类型为 uint32_t 范围内的有符号数
                        string str_type = GetValueTypeAsString(value_type);
                        result =    "       if(group_thread == 0){\n"
                                    "              " + str_type +" "+ str_val + "_data = *( ( "+ str_type +"* )( input["+ str_index +"][index] ) );\n"
                                    "              if(" + str_val + "_data < 0){\n"
                                    "                     " + str_val + "_data = -" + str_val + "_data;\n"
                                    "                     var_" + str + "_sign = 1;\n"
                                    "              }\n" +
                                    "              "+ str_val +"[0] = "+ str_val + "_data % PER_DEC_MAX_SCALE;\n"
                                    "              "+ str_val +"[1] = "+ str_val + "_data / PER_DEC_MAX_SCALE\n"
                                    "       }\n";
                    }
                    else if( value_type.DataType.ValueType == AriesValueType::UINT64){
                        // 如果类型为 uint64_t 范围内的无符号数
                        string str_type = GetValueTypeAsString(value_type);
                        result =    "       if(group_thread == 0){\n"
                                    "              "+ str_val +"[0] = "+ str_val + "_data % PER_DEC_MAX_SCALE;\n"
                                    "              "+ str_val +"[1] = "+ str_val + "_data / PER_DEC_MAX_SCALE\n"
                                    "       }\n";
                    }
                    else{
                        assert(0);
                    }
                }
                break;
            }
            case AriesExprType::CALC:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                result = AriesCommonExpr::calcOpToStr[ static_cast< AriesCalculatorOpType >( boost::get< int >( content ) ) ];
                break;
            }
            case AriesExprType::AGG_FUNCTION:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                result = AriesCommonExpr::aggFuncToStr[ static_cast< AriesAggFunctionType >( boost::get< int >( content ) ) ];
                break;
            }
            case AriesExprType::SQL_FUNCTION:
            {
                //we handle sql functions in stringForSqlFunctions function
                break;
            }
            case AriesExprType::DECIMAL:
            {
                ARIES_ASSERT( boost::get< aries_acc::Decimal >( &content ), "content type: " + string( content.type().name() ) );
                if ( printConstAsLiteral )
                {
                    char value[ 64 ];
                    aries_acc::Decimal dec = boost::get< aries_acc::Decimal >( content );
                    result = " Decimal( \"";
                    result += dec.GetDecimal( value );
                    result += "\" ) ";
                }
                else
                {
                    // 如果是 decimal 则直接获取即可

                    string str_var_limb = "";

                    if( ansLIMBS == 1)
                        str_var_limb = "LIMBS_ONE";
                    else if ( ansLIMBS == 2)
                        str_var_limb = "LIMBS_TWO";
                    else if ( ansLIMBS == 4)
                        str_var_limb = "LIMBS_THR";
                    else if ( ansLIMBS == 8)
                        str_var_limb = "LIMBS_FOR";

                    uint32_t per_size = ansLIMBS*sizeof(uint32_t);
                    string str_index = to_string( constValues.size() );
                    string str = to_string(varIndexInKernel);
                    string str_val = "var_"+ str;
                    string str_per_size = to_string(per_size);
                    result = "";
                    result += "       uint32_t " + str_val + "["+ str_var_limb +"] = {0};\n";
                    result += "       aries_memcpy("+ str_val +", ( *( Decimal* )( constValues[ "+ str_index +" ] )).v+group_thread*"+ str_var_limb +", "+ str_per_size +");\n";
                    result += "       uint32_t " + str_val + "_sign = ( *( Decimal* )( constValues[ "+ str_index +"  ] )).sign;\n";
                    result += "       uint32_t " + str_val + "_frac = ( *( Decimal* )( constValues[ "+ str_index +"  ] )).frac;\n";
                    int ansDecLength = NUM_TOTAL_DIG;
                    constValues.push_back( ConstExprContentToDataBuffer( content, 1, ansDecLength) );
                }

                break;
            }
            default:
                //for other expr types, nothing to do
                break;
        }
        return result;
    }

    string AriesCommonExpr::stringForSqlFunctions( std::map< string, AriesCommonExprUPtr > &aggFunctions,
                                                   int &seq,
                                                   set< AriesDynamicCodeParam >& ariesParams,
                                                   vector< AriesDataBufferSPtr >& constValues,
                                                   vector< AriesDynamicCodeComparator >& ariesComparators,
                                                   bool printConstAsLiteral )
    {
        string result = "sql_function_wrapper< " + GenerateParamType( value_type ) + " >( ";
        AriesSqlFunctionType type = GetSqlFunctionType();
        switch( type )
        {
            case AriesSqlFunctionType::UNIX_TIMESTAMP:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    result += "UnixTimestampWrapper< aries_acc::AriesDatetime, " + GetValueTypeAsString( children[ 1 ]->GetValueType() ) + " >(), ";
                    result += ( children[ 0 ]->value_type.HasNull ? "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " : "convert_string_to_type< aries_acc::AriesDatetime, " )  + std::to_string( children[ 0 ]->value_type.DataType.Length )
                            + ", " + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " )
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ), ";
                }
                else
                {
                    result += "UnixTimestampWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + ", "
                            + GetValueTypeAsString( children[ 1 ]->GetValueType() ) + " >(), ";
                    result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                }
                result += children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                break;
            }
            case AriesSqlFunctionType::EXTRACT:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );

                if( children[ 1 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 1 ]->value_type.DataType.Length > 1 )
                {
                    result += "ExtractWrapper< aries_acc::AriesDatetime, interval_type >(), convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, "
                            + std::to_string( children[ 1 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 1 ]->value_type.HasNull ? "true >()( " : "false >()( " )
                            + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ), ";
                }
                else
                {
                    result += "ExtractWrapper< " + GetValueTypeAsString( children[ 1 ]->GetValueType() ) + ", interval_type >(), ";
                    result += children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                }

                string intervalValue = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                // remove "" for intervalValue. ex: "YEAR"->YEAR
                result += IntervalToString( get_interval_type( intervalValue.substr( 1, intervalValue.size() - 2 ) ) ) + " ) ";
                break;
            }
            case AriesSqlFunctionType::DATE:
            {
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    result += "DateWrapper< aries_acc::AriesDate >(), convert_string_to_type< nullable_type< aries_acc::AriesDate >, "
                            + std::to_string( children[ 0 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " )
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ) ";
                }
                else
                {
                    result += "DateWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + " >(), ";
                    result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                }
                break;
            }
            case AriesSqlFunctionType::DATE_SUB:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                string interval = children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    result += "DateSubWrapper< aries_acc::AriesDatetime, " + children[ 1 ]->GetIntervalUnitTypeName() + " >(), ";
                    result += "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " + std::to_string( children[ 0 ]->value_type.DataType.Length )
                            + ", " + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " )
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ), ";
                }
                else
                {
                    result += "DateSubWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + ", "
                            + children[ 1 ]->GetIntervalUnitTypeName() + " >(), ";
                    result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                }
                result += interval + " ) ";
                break;
            }
            case AriesSqlFunctionType::DATE_ADD:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                string interval = children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    result += "DateAddWrapper< aries_acc::AriesDatetime, " + children[ 1 ]->GetIntervalUnitTypeName() + " >(), ";
                    result += "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " + std::to_string( children[ 0 ]->value_type.DataType.Length )
                            + ", " + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " )
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ), ";
                }
                else
                {
                    result += "DateAddWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + ", "
                            + children[ 1 ]->GetIntervalUnitTypeName() + " >(), ";
                    result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                }
                result += interval + " ) ";
                break;
            }
            case AriesSqlFunctionType::DATE_DIFF:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                result += "DateDiffWrapper< ";
                string child0 = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    child0 = "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " + std::to_string( children[ 0 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " ) + child0 + " )";
                    result += "aries_acc::AriesDatetime, ";
                }
                else
                    result += GetValueTypeAsString( children[ 0 ]->GetValueType() ) + ", ";

                string child1 = children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                if( children[ 1 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 1 ]->value_type.DataType.Length > 1 )
                {
                    child1 = "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " + std::to_string( children[ 1 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 1 ]->value_type.HasNull ? "true >()( " : "false >()( " ) + child1 + " )";
                    result += "aries_acc::AriesDatetime >(), ";
                }
                else
                    result += GetValueTypeAsString( children[ 1 ]->GetValueType() ) + " >(), ";
                result += child0 + ", " + child1 + " ) ";
                break;
            }
            case AriesSqlFunctionType::TIME_DIFF:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                result += "TimeDiffWrapper< ";
                string child0 = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    child0 = "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " + std::to_string( children[ 0 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " ) + child0 + " )";
                    result += "aries_acc::AriesDatetime, ";
                }
                else
                    result += GetValueTypeAsString( children[ 0 ]->GetValueType() ) + ", ";

                string child1 = children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                if( children[ 1 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 1 ]->value_type.DataType.Length > 1 )
                {
                    child1 = "convert_string_to_type< nullable_type< aries_acc::AriesDatetime >, " + std::to_string( children[ 1 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 1 ]->value_type.HasNull ? "true >()( " : "false >()( " ) + child1 + " )";
                    result += "aries_acc::AriesDatetime >(), ";
                }
                else
                    result += GetValueTypeAsString( children[ 1 ]->GetValueType() ) + " >(), ";
                result += child0 + ", " + child1 + " ) ";
                break;
            }
            case AriesSqlFunctionType::ABS:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                result += "AbsWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + " >(), ";
                result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                break;
            }
            case AriesSqlFunctionType::SUBSTRING:
            {
                ARIES_ASSERT( children.size() == 3, "children.size(): " + to_string( children.size() ) );
                string hasNull = children[ 0 ]->GetValueType().HasNull ? "true" : "false";
                string child0 = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, true );
                string child1 = children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, true );
                string child2 = children[ 2 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, true );
                auto type = children[ 0 ]->GetValueType();
                result = "op_substr_utf8_t< ";
                result += to_string( value_type.DataType.Length ) + ", ";
                result += child2 + ", ";
                result += hasNull + " >()( ";
                int hasNulFlag = children[ 0 ]->GetValueType().HasNull ? 1 : 0;
                switch( type.DataType.ValueType )
                {
                    case AriesValueType::INT8:
                        result += "op_tostr_t()(" + child0 + "), " + child1 + ", " + to_string( TypeSizeInString< int8_t >::LEN + hasNulFlag )
                                + " ) ";
                        break;
                    case AriesValueType::INT16:
                        result += "op_tostr_t()(" + child0 + "), " + child1 + ", " + to_string( TypeSizeInString< int16_t >::LEN + hasNulFlag )
                                + " ) ";
                        break;
                    case AriesValueType::INT32:
                        result += "op_tostr_t()(" + child0 + "), " + child1 + ", " + to_string( TypeSizeInString< int32_t >::LEN + hasNulFlag )
                                + " ) ";
                        break;
                    case AriesValueType::INT64:
                        result += "op_tostr_t()(" + child0 + "), " + child1 + ", " + to_string( TypeSizeInString< int64_t >::LEN + hasNulFlag )
                                + " ) ";
                        break;
                    case AriesValueType::CHAR:
                        result += child0 + ", " + child1 + ", " + to_string( type.GetDataTypeSize() ) + " ) ";
                        break;
                    default:
                        ARIES_ASSERT( 0, "SUBSTRING doesn't support type: " + GetValueTypeAsString( type ) );
                        break;
                }
                break;
            }
            case AriesSqlFunctionType::MONTH:
            {
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    result += "MonthWrapper< aries_acc::AriesDate >(), convert_string_to_type< nullable_type< aries_acc::AriesDate >, "
                            + std::to_string( children[ 0 ]->value_type.DataType.Length ) + ", "
                            + ( children[ 0 ]->value_type.HasNull ? "true >()( " : "false >()( " )
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ) ";
                }
                else
                {
                    result += "MonthWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + " >(), ";
                    result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                }
                break;
            }
            case AriesSqlFunctionType::DATE_FORMAT:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                string hasNull = children[ 0 ]->GetValueType().HasNull ? "true" : "false";
                auto format = boost::get< std::string >( children[ 1 ]->GetContent() );
                auto targetLen = aries_acc::get_format_length( format.data() );
                result = "op_dateformat_t< ";
                if( children[ 0 ]->GetValueType().DataType.ValueType == AriesValueType::TIME )
                {
                    result += "aries_acc::AriesDatetime, ";
                }
                else
                {
                    result += GetValueTypeAsString( children[ 0 ]->GetValueType() ) + ", ";
                }
                auto now = AriesDatetimeTrans::GetInstance().Now();
                result += std::to_string( targetLen ) + ", ";
                result += hasNull + " >() (";
                result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                result += "AriesDatetime( " + std::to_string( now.year ) + ", " + std::to_string( now.month ) + ", " + std::to_string( now.day )
                        + ", 0, 0, 0, 0) ,";
                result += children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                result += "aries_acc::LOCALE_LANGUAGE::en_US ) ";
                break;
            }
            case AriesSqlFunctionType::CAST:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                switch( value_type.DataType.ValueType )
                {
                    case AriesValueType::DOUBLE:
                        // result += "CastAsDoubleWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + " >(), ";
                        // result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                        ARIES_ASSERT( 0, "Not support target type " + GetValueTypeAsString( value_type ) + " for CAST function now" );
                        break;
                    case AriesValueType::INT32:
                        result += "CastAsIntWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + " >(), ";
                        result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                        break;
                    case AriesValueType::INT64:
                        result += "CastAsLongWrapper< " + GetValueTypeAsString( children[ 0 ]->GetValueType() ) + " >(), ";
                        result += children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ";
                        break;
                    case AriesValueType::DECIMAL:
                    {
                        AriesColumnType childType = children[ 0 ]->GetValueType();
                        assert( childType.HasNull == value_type.HasNull );
                        if( !childType.HasNull )
                            result = "cast_as_decimal( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                        else 
                            result = "cast_as_nullable_decimal( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";

                        switch( childType.DataType.ValueType )
                        {
                            case AriesValueType::CHAR:
                            {
                                result += std::to_string( childType.GetDataTypeSize() ) + ", " + std::to_string( value_type.DataType.Precision ) + ", " + std::to_string( value_type.DataType.Scale ) + " )";
                                break;
                            }
                            case AriesValueType::DECIMAL:
                            case AriesValueType::COMPACT_DECIMAL:
                            {
                                result += std::to_string( value_type.DataType.Precision ) + ", " + std::to_string( value_type.DataType.Scale ) + " )";
                                break;
                            }
                            /*
                            case AriesValueType::COMPACT_DECIMAL:
                            {
                                result += std::to_string( childType.DataType.Precision ) + ", " + std::to_string( childType.DataType.Scale ) + ", " + std::to_string( value_type.DataType.Precision ) + ", " + std::to_string( value_type.DataType.Scale ) + " )";
                                break;
                            }
                            */
                            case AriesValueType::BOOL:
                            case AriesValueType::INT8:
                            case AriesValueType::INT16:
                            case AriesValueType::INT32:
                            case AriesValueType::INT64:
                            {
                                result += std::to_string( value_type.DataType.Precision ) + ", " + std::to_string( value_type.DataType.Scale ) + " )";
                                break;
                            }
                            default:
                            {
                                string msg( "cast " +  GetValueTypeAsString( value_type ) + " as decimal" );
                                ThrowNotSupportedException( msg.data() );
                            }
                        }
                        break;
                    }
                    case AriesValueType::DATE:
                    {
                        AriesColumnType childType = children[ 0 ]->GetValueType();
                        assert( childType.HasNull == value_type.HasNull );
                        if( !childType.HasNull )
                            result = "cast_as_date( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                        else 
                            result = "cast_as_nullable_date( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );

                        switch( childType.DataType.ValueType )
                        {
                            case AriesValueType::CHAR:
                            {
                                result += ", " + std::to_string( childType.GetDataTypeSize() ) + " )";
                                break;
                            }
                            default:
                            {
                                result += " )";
                                break;
                            }
                        }
                        break;
                    }
                    case AriesValueType::DATETIME:
                    {
                        AriesColumnType childType = children[ 0 ]->GetValueType();
                        assert( childType.HasNull == value_type.HasNull );
                        if( !childType.HasNull )
                            result = "cast_as_datetime( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, ++seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                        else
                            result = "cast_as_nullable_datetime( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, ++seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );

                        switch( childType.DataType.ValueType )
                        {
                            case AriesValueType::CHAR:
                            {
                                result += ", " + std::to_string( childType.GetDataTypeSize() ) + " )";
                                break;
                            }
                            default:
                            {
                                result += " )";
                                break;
                            }
                        }
                        break;
                    }
                    default:
                        ARIES_ASSERT( 0, "Not support target type " + GetValueTypeAsString( value_type ) + " for CAST function now" );
                        break;
                }
                break;
            }
            case AriesSqlFunctionType::CONCAT:
            {
                ARIES_ASSERT( children.size() > 1, "children.size(): " + to_string( children.size() ) );
                ARIES_ASSERT( value_type.DataType.Length > 0, "Datatype length: " + to_string( value_type.DataType.Length ) );
                int len = value_type.DataType.Length;
                if( len == 1 )
                {
                    result = "aries_concat< nullable_type< char > >( ";
                }
                else
                {
                    result = "aries_concat< nullable_type< aries_char< ";
                    result += std::to_string( value_type.DataType.Length );
                    result += " > > >( ";
                }
                for( std::size_t i = 0; i < children.size(); ++i )
                {
                    result += children[ i ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                    result += ",";
                }
                result.pop_back();
                result += " )";
                break;
            }
            case AriesSqlFunctionType::TRUNCATE:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                result += "TruncateWrapper<" + GetValueTypeAsString( children[0]->GetValueType() ) + ">(), ";
                result += children[0]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ", ";
                result += children[1]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " )";
                break;
            }
            case AriesSqlFunctionType::DICT_INDEX:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                result = "dict_index( " + children[0]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " )";
                break;
            }
            case AriesSqlFunctionType::ANY_VALUE:
            {
                result = "";
                break;
            }
            default:
                LOG(ERROR)<< "unsupported function type: " << (int)type << std::endl;
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "function type " );
            }
        return result;
    }

    string AriesCommonExpr::ToString() const
    {
        string result;
        switch( type )
        {
            case AriesExprType::INTEGER:
            {
                if( value_type.DataType.ValueType == AriesValueType::INT64 )
                    result += std::to_string( boost::get< int64_t >( content ) );
                else
                    result += std::to_string( boost::get< int >( content ) );
                result += "_INTEGER";
                break;
            }
            case AriesExprType::FLOATING:
            {
                result += std::to_string( boost::get< double >( content ) );
                result += "_FLOATING";
                break;
            }
            case AriesExprType::DECIMAL:
            {
                char value[ 64 ];
                aries_acc::Decimal dec = boost::get< aries_acc::Decimal >( content );
                result += dec.GetDecimal( value );
                result += "_DECIMAL";
                break;
            }
            case AriesExprType::STRING:
            {
                result += boost::get< string >( content );
                result += "_STRING";
                break;
            }
            case AriesExprType::DATE:
            {
                aries_acc::AriesDate date = boost::get< aries_acc::AriesDate >( content );
                result += std::to_string( date.getYear() );
                result += "-";
                result += std::to_string( date.getMonth() );
                result += "-";
                result += std::to_string( date.getDay() );
                result += "_DATE";
                break;
            }
            case AriesExprType::DATE_TIME:
            {
                aries_acc::AriesDatetime date = boost::get< aries_acc::AriesDatetime >( content );
                result += std::to_string( date.getYear() );
                result += "-";
                result += std::to_string( date.getMonth() );
                result += "-";
                result += std::to_string( date.getDay() );
                result += "-";
                result += std::to_string( date.getHour() );
                result += "-";
                result += std::to_string( date.getMinute() );
                result += "-";
                result += std::to_string( date.getSecond() );
                result += "-";
                result += std::to_string( date.getMicroSec() );
                result += "_DATE_TIME";
                break;
            }
            case AriesExprType::TIME:
            {
                aries_acc::AriesTime time = boost::get< aries_acc::AriesTime >( content );
                result += std::to_string( time.sign );
                result += ",";
                result += std::to_string( time.hour );
                result += "-";
                result += std::to_string( time.minute );
                result += "-";
                result += std::to_string( time.second );
                result += "-";
                result += std::to_string( time.second_part );
                result += "_TIME";
                break;
            }
            case AriesExprType::TIMESTAMP:
            {
                aries_acc::AriesTimestamp timestamp = boost::get< aries_acc::AriesTimestamp >( content );
                result += std::to_string( timestamp.getTimeStamp() );
                result += "_TIMESTAMP";
                break;
            }
            case AriesExprType::YEAR:
            {
                aries_acc::AriesYear year = boost::get< aries_acc::AriesYear >( content );
                result += std::to_string( year.getYear() );
                result += "_YEAR";
                break;
            }
            case AriesExprType::COLUMN_ID:
            {
                int columnId = boost::get< int >( content );
                result += std::to_string( columnId );
                result += GenerateParamType( value_type );
                result += "_COLUMN_ID";
                break;
            }
            case AriesExprType::STAR:
            {
                result += boost::get< string >( content );
                result += "_STAR";
                break;
            }
            case AriesExprType::AGG_FUNCTION:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_AGG_FUNCTION";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::SQL_FUNCTION:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_SQL_FUNCTION";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::CALC:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_CALC";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::NOT:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_NOT";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::COMPARISON:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_COMPARISON";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::LIKE:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_LIKE";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::IN:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_IN";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::NOT_IN:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_NOT_IN";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::BETWEEN:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_BETWEEN";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::EXISTS:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_EXISTS";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::CASE:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_CASE";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::AND_OR:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_AND_OR";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::BRACKETS:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_BRACKETS";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::TRUE_FALSE:
            {
                result += std::to_string( boost::get< bool >( content ) );
                result += "_TRUE_FALSE";
                break;
            }
            case AriesExprType::IF_CONDITION:
            {
                result += std::to_string( boost::get< int >( content ) );
                result += "_IF_CONDITION";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::IS_NOT_NULL:
            {
                result += "_IS_NOT_NULL";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::IS_NULL:
            {
                result += "_IS_NULL";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::DISTINCT:
            {
                result += std::to_string( boost::get< bool >( content ) );
                result += "_DISTINCT";
                break;
            }
            case AriesExprType::NULL_VALUE:
            {
                result += "_NULL_VALUE";
                break;
            }
            case AriesExprType::INTERVAL:
            {
                result += boost::get< string >( content );
                result += "_INTERVAL";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            case AriesExprType::COALESCE:
            {
                result += "_COALESCE";
                for( const auto& child : children )
                    result += child->ToString();
                break;
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "expression type " + GetAriesExprTypeName( type ) );
                break;
        }
        return result;
    }

    string AriesIntervalToString( const AriesInterval& interval,
                                  vector< AriesDataBufferSPtr >& constValues )
    {
        string result;
        size_t size;
        AriesDataBufferSPtr constValue;
        switch( interval.type )
        {
            case INTERVAL_YEAR:
            {
                YearInterval value = AriesDatetimeTrans::GetInstance().ToYearInterval( interval.interval );
                // result = " YearInterval( " + std::to_string( value.year ) + ", " + std::to_string( value.sign ) + " ) ";
                result = "( *( ( YearInterval* )constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                size = sizeof( YearInterval );
                AriesColumnType colType( AriesDataType( AriesValueType::INT8, size ), false );
                constValue = make_shared< AriesDataBuffer >( colType, 1 );
                memcpy( constValue->GetData(), &value, size );
                break;
            }
            case INTERVAL_MONTH:
            {
                MonthInterval value = AriesDatetimeTrans::GetInstance().ToMonthInterval( interval.interval );
                // result = " MonthInterval( " + std::to_string( value.month ) + ", " + std::to_string( value.sign ) + " ) ";
                result = "( *( ( MonthInterval* )constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                size = sizeof( MonthInterval );
                AriesColumnType colType( AriesDataType( AriesValueType::INT8, size ), false );
                constValue = make_shared< AriesDataBuffer >( colType, 1 );
                memcpy( constValue->GetData(), &value, size );
                break;
            }
            case INTERVAL_DAY:
            {
                DayInterval value = AriesDatetimeTrans::GetInstance().ToDayInterval( interval.interval );
                // result = " DayInterval( " + std::to_string( value.day ) + ", " + std::to_string( value.sign ) + " ) ";
                result = "( *( ( DayInterval* )constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                size = sizeof( DayInterval );
                AriesColumnType colType( AriesDataType( AriesValueType::INT8, size ), false );
                constValue = make_shared< AriesDataBuffer >( colType, 1 );
                memcpy( constValue->GetData(), &value, size );
                break;
            }
            case INTERVAL_HOUR:
            {
                SecondInterval value = AriesDatetimeTrans::GetInstance().ToSecondInterval( interval.interval );
                // result = " SecondInterval( " + std::to_string( value.second ) + ", " + std::to_string( value.sign ) + " ) ";
                result = "( *( ( SecondInterval* )constValues[ " + std::to_string( constValues.size() ) + " ] ) )";
                size = sizeof( SecondInterval );
                AriesColumnType colType( AriesDataType( AriesValueType::INT8, size ), false );
                constValue = make_shared< AriesDataBuffer >( colType, 1 );
                memcpy( constValue->GetData(), &value, size );
                break;
            }
            default:
                //FIXME need support other intervals
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "transferring interval type: " + to_string( ( int ) interval.type ) );
                break;
        }
        constValues.push_back( constValue );
        return result;
    }

    string AriesCommonExpr::GetIntervalUnitTypeName() const
    {
        ARIES_ASSERT( AriesExprType::INTERVAL == type, "not a interval expression" );
        ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
        string typeName;
        auto content_string = boost::get< string >( content );
        interval_type intervalType = get_interval_type( content_string );
        switch ( intervalType )
        {
            case interval_type::INTERVAL_YEAR:
                typeName = "YearInterval";
                break;
            case interval_type::INTERVAL_MONTH:
                typeName = "MonthInterval";
                break;
            case interval_type::INTERVAL_DAY:
                typeName = "DayInterval";
                break;
            case interval_type::INTERVAL_HOUR:
                typeName = "SecondInterval";
                break;
            default:
                //FIXME need support other intervals
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "transferring interval type: " + content_string );
                break;
        }
        return typeName;
    }

    string AriesGetXmpCalcFunctionName( int ansLEN, int ansTPI){
        string fun_name = "";
        if( ansLEN == 4 )
            fun_name += "_LEN_4";
        else if( ansLEN == 8 )
            fun_name += "_LEN_8";
        else if( ansLEN == 16 )
            fun_name += "_LEN_16";
        else if( ansLEN == 32 )
            fun_name += "_LEN_32";
        else
            return fun_name;

        if( ansTPI == 4 )
            fun_name += "_TPI_4";
        else if( ansTPI == 8 )
            fun_name += "_TPI_8";
        else if( ansTPI == 16 )
            fun_name += "_TPI_16";
        else if( ansTPI == 32 )
            fun_name += "_TPI_32";

        return fun_name;
    }

    std::string AriesCommonExpr::stringForDynamicCodeInternal( std::map< string, AriesCommonExprUPtr > &aggFunctions,
                                                               int &seq,
                                                               set< AriesDynamicCodeParam >& ariesParams,
                                                               vector< AriesDataBufferSPtr >& constValues,
                                                               vector< AriesDynamicCodeComparator >& ariesComparators,
                                                               bool printConstAsLiteral,
                                                               int ansDecLength)
    {
        seq++;
        std::string result;
        std::string content_string = contentToString( ariesParams, constValues, printConstAsLiteral, ansDecLength);
        if( value_reverse){
            content_string = "-"+content_string;
        }
        switch( type )
        {
            case AriesExprType::INTEGER:
            case AriesExprType::FLOATING:
            case AriesExprType::COLUMN_ID:
            case AriesExprType::STAR:
            case AriesExprType::DECIMAL:
            case AriesExprType::DATE:
            case AriesExprType::TIME:
            case AriesExprType::TIMESTAMP:
            case AriesExprType::YEAR:
            case AriesExprType::DATE_TIME:
            case AriesExprType::NULL_VALUE:
            case AriesExprType::TRUE_FALSE:
            {
                result = std::move( content_string );
                break;
            }
            case AriesExprType::STRING:
            {
                result = std::move( content_string );
                break;
            }
            case AriesExprType::CALC:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                AriesCalculatorOpType opType = static_cast< AriesCalculatorOpType >( boost::get< int >( content ) );
                if( opType == AriesCalculatorOpType::DIV )
                {
                    AriesColumnType type_0 = children[ 0 ]->GetValueType();
                    AriesColumnType type_1 = children[ 1 ]->GetValueType();
                    if( IsIntegerType( type_0 ) && IsIntegerType( type_1 ) )
                    {
                        if( type_0.HasNull )
                        {
                            result = "( ( nullable_type< Decimal > )"
                                    + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral )
                                    + content_string
                                    + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")";
                        }
                        else
                        {
                            result = "( ( Decimal )"
                                    + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral )
                                    + content_string
                                    + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")";
                        }
                        break;
                    }
                }

                result = "(" + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral, ansDecLength )+" " + content_string+" "
                        + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral, ansDecLength ) + ")";  // 这里生成了表达式代码
                break;
            }
            case AriesExprType::AND_OR:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                if ( contains_decimal(  children[ 0 ] ) && !contains_decimal( children[ 1 ] ) )
                {
                    result = "(" + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + content_string
                        + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")";
                }
                else
                {
                    result = "(" + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + content_string
                        + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")";
                }
                break;
            }
            case AriesExprType::COMPARISON:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                AriesCommonExpr *leftChild = children[ 0 ].get();
                AriesCommonExpr *rightChild = children[ 1 ].get();
                AriesComparisonOpType cmpType = static_cast< AriesComparisonOpType >( boost::get< int >( content ) );

                if( leftChild->value_type.DataType.ValueType == AriesValueType::CHAR
                        || rightChild->value_type.DataType.ValueType == AriesValueType::CHAR )
                {
                    if( leftChild->type == AriesExprType::STRING )
                    {
                        // make sure left param is columnId
                        std::swap( leftChild, rightChild );
                        //after swap, we need change the cmpType.
                        switch( cmpType )
                        {
                            case AriesComparisonOpType::GT:
                                cmpType = AriesComparisonOpType::LT;
                                break;
                            case AriesComparisonOpType::GE:
                                cmpType = AriesComparisonOpType::LE;
                                break;
                            case AriesComparisonOpType::LT:
                                cmpType = AriesComparisonOpType::GT;
                                break;
                            case AriesComparisonOpType::LE:
                                cmpType = AriesComparisonOpType::GE;
                                break;
                            default:
                                break;
                        }
                    }

                    if( ( leftChild->type == AriesExprType::COLUMN_ID || leftChild->type == AriesExprType::SQL_FUNCTION )
                            && rightChild->type == AriesExprType::STRING )
                    {
                        string param = boost::get< string >( rightChild->content );
                        auto len = param.size();
                        auto dataLen = leftChild->GetValueType().DataType.Length;
                        bool bContinue = true;
                        if( dataLen >= 0 && len > std::size_t( dataLen ) )
                        {
                            //由于会做截断，将转换比较操作符
                            switch( cmpType )
                            {
                                case AriesComparisonOpType::EQ:
                                {
                                    // 对于字符串相等比较，如果常量的长度大于目标类型长度，则不可能有相等值，直接返回0
                                    result = "(0)";
                                    bContinue = false;
                                    break;
                                }
                                case AriesComparisonOpType::NE:
                                {
                                    // 对于字符串不等比较，如果常量的长度大于目标类型长度，则不等于全部成立，直接返回1
                                    result = "(1)";
                                    bContinue = false;
                                    break;
                                }
                                case AriesComparisonOpType::GE:
                                {
                                    //对于>=，由于对常量进行了截断，=条件不可能满足。所以转化为GT
                                    cmpType = AriesComparisonOpType::GT;
                                    break;
                                }
                                case AriesComparisonOpType::LT:
                                {
                                    //对于<，由于对常量进行了截断，=条件也满足最终结果。所以转化为LE
                                    cmpType = AriesComparisonOpType::LE;
                                    break;
                                }
                                default:
                                    break;
                            }
                        }
                        if( bContinue )
                        {
                            if( dataLen > 1 )
                            {
                                result = generateCompareFunctionHeader( cmpType, leftChild->GetValueType().HasNull, false );
                                result += leftChild->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                                result += ", ";
                                // result += "\"" + boost::get< string >( rightChild->content ) + "\", ";
                                result += "( ( char* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) ), ";
                                param.resize( leftChild->GetValueType().DataType.Length, 0 );
                                AriesColumnType colType( AriesDataType( AriesValueType::CHAR, leftChild->GetValueType().DataType.Length ), false );
                                auto constValue = aries_acc::CreateDataBufferWithValue( param, 1, colType );
                                constValues.push_back( constValue );
                                // if( len < dataLen )
                                //     result += to_string( len + 1 + leftChild->GetValueType().HasNull );
                                // else
                                    result += to_string( leftChild->GetValueType().GetDataTypeSize() );
                                result += " )";
                            }
                            else
                            {
                                ARIES_ASSERT( dataLen == 1, "=dataLen: " + to_string( dataLen ) );
                                result = "( " + leftChild->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " "
                                        + ComparisonOpToString( cmpType ) + " '" + boost::get< string >( rightChild->content ) + "'" + " )";
                            }
                        }
                    }
                    else if( leftChild->type == AriesExprType::COLUMN_ID && rightChild->type == AriesExprType::COLUMN_ID )
                    {
                        ARIES_ASSERT( leftChild->GetValueType().DataType.Length == rightChild->GetValueType().DataType.Length,
                                "leftChild->GetValueType().DataType.Length: " + to_string( leftChild->GetValueType().DataType.Length )
                                        + ", rightChild->GetValueType().DataType.Length: "
                                        + to_string( rightChild->GetValueType().DataType.Length ) );
                        if( leftChild->GetValueType().DataType.Length > 1 )
                        {
                            result = generateCompareFunctionHeader( cmpType, leftChild->GetValueType().HasNull, rightChild->GetValueType().HasNull );
                            result += leftChild->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                            result += ", ";
                            result += rightChild->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                            result += ", ";
                            result += to_string(
                                    std::max( leftChild->GetValueType().GetDataTypeSize(), rightChild->GetValueType().GetDataTypeSize() ) );
                            result += " )";
                        }
                        else
                        {
                            result = "( (" + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")"
                                    + content_string + "("
                                    + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ") )";
                        }
                    }
                    else
                    {
                        //FIXME need error code here
                        assert( 0 );
                        ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET,
                                "compare two different data types, left data type: " + GetValueTypeAsString( leftChild->GetValueType() )
                                        + ", right data type: " + GetValueTypeAsString( rightChild->GetValueType() ) );
                    }
                }
                else
                {
                    result = "( (" + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")"
                            + content_string + "(" + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral )
                            + ") )";
                }

                break;
            }
            case AriesExprType::BRACKETS:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                result = "(" + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ")";
                break;
            }
            case AriesExprType::SQL_FUNCTION:
            {
                result = stringForSqlFunctions( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                break;
            }
            case AriesExprType::AGG_FUNCTION:
            {
                ARIES_ASSERT( children.size() <= 2, "children.size(): " + to_string( children.size() ) );
                char buf[ 128 ];
                //use 0 as the column id for columns generated by agg functions.
                //the "%s%s%d_%d" will be used as param name for outer calc expr. play as other simple column from ref table
                //so when do outer expr calculation, we can know the column is from an agg function instead of the ref table
                content_string += std::to_string( m_id );
                sprintf( buf, "%s_%s_%d", "column", content_string.c_str(), seq );
                ariesParams.insert( AriesDynamicCodeParam { 0, buf, value_type } );
                result = buf;
                aggFunctions.insert( { result, AriesCommonExprUPtr( this ) } );
                break;
            }
            case AriesExprType::LIKE:
            {
                ARIES_ASSERT( boost::get< int >( &content ), "content type: " + string( content.type().name() ) );
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                const AriesCommonExprUPtr &leftChild = children[ 0 ];
                const AriesCommonExprUPtr &rightChild = children[ 1 ];
                //assert( leftChild->type == AriesExprType::COLUMN_ID );
                ARIES_ASSERT( rightChild->type == AriesExprType::STRING, "rightChild->type: " + GetAriesExprTypeName( rightChild->type ) );
                string param = boost::get< string >( rightChild->content );
                param.resize( param.size() + 1, 0 );

                if( leftChild->GetValueType().HasNull )
                    result = " op_like_t< true >()( ";
                else
                    result = " op_like_t< false >()( ";
                result += leftChild->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                result += ", ";
                // result += "\"" + boost::get< string >( rightChild->content ) + "\", ";
                result += "( ( char* )( constValues[ " + std::to_string( constValues.size() ) + " ] ) ), ";
                result += to_string( leftChild->GetValueType().GetDataTypeSize() );
                result += " )";

                AriesColumnType colType( AriesDataType( AriesValueType::CHAR, param.size() ), false );
                auto constValue = aries_acc::CreateDataBufferWithValue( param, 1, colType );
                constValues.push_back( constValue );
                break;
            }
            case AriesExprType::CASE:
            case AriesExprType::IF_CONDITION:
            {
                ARIES_ASSERT( children.size() % 2 == 1, "children.size(): " + to_string( children.size() ) ); // assume there is always a ELSE
                const char *op = "?:";
                int cur = 0;
                for( const auto &child : children )
                {
                    if( value_type.DataType.ValueType == AriesValueType::CHAR && value_type.DataType.Length > 1 )
                    {
                        if( child->value_type.DataType.ValueType == AriesValueType::CHAR )
                        {
                            result += GenerateParamType( value_type ) + "( "
                                    + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " )";
                        }
                        else
                        {
                            string val;
                            if( cur == 1 )
                                val = "convert_to_string< " + GetValueTypeAsString( child->value_type ) + ", "
                                        + std::to_string( value_type.DataType.Length ) + ", "
                                        + ( child->value_type.HasNull ? "true >()( " : "false >()( " );
                            result += val + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                            if( cur == 1 )
                                result += " ) ";
                        }
                    }
                    else
                    {
                        if( cur )
                        {
                            result += GenerateParamType( value_type ) + "( "
                                    + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " )";
                        }
                        else 
                        {
                            result += child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                        }
                    }
                        
                    result += op[ cur ];
                    cur ^= 1;
                }
                if( !result.empty() )
                {
                    result.erase( result.end() - 1 );
                }
                break;
            }
            case AriesExprType::COALESCE:
            {
                ARIES_ASSERT( children.size() > 0, "children.size(): " + to_string( children.size() ) );
                const char *op = "?:";
                int cur = 0;
                for( const auto &child : children )
                {
                    AriesColumnType columnType = child->GetValueType();
                    string codeForChild = child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                    if( columnType.HasNull )
                    {
                        if( ( columnType.DataType.ValueType == AriesValueType::CHAR ) && ( columnType.DataType.Length > 1 ) )
                            result += "( !is_null< true >( " + codeForChild + ") ) ";
                        else
                            result += "( !is_null( " + codeForChild + ") ) ";
                    }
                    else
                    {
                        result += "(1)";
                    }
                    result += op[ cur ];
                    result += codeForChild;
                    cur ^= 1;
                    result += op[ cur ];
                    cur ^= 1;
                }
                if( !result.empty() )
                    result += makeNullLiteral( value_type );
                break;
            }
            case AriesExprType::IS_NULL:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                const auto& child = children[ 0 ];
                AriesColumnType columnType = child->GetValueType();
                if( columnType.HasNull )
                {
                    if( ( columnType.DataType.ValueType == AriesValueType::CHAR ) && ( columnType.DataType.Length > 1 ) )
                    {
                        result = "( is_null< true >( " + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral )
                                + ") ) ";
                    }
                    else
                        result = "( is_null( " + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ") ) ";
                }
                else
                {
                    result = "(0)";
                }
                break;
            }
            case AriesExprType::IS_NOT_NULL:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                const auto& child = children[ 0 ];
                AriesColumnType columnType = child->GetValueType();
                if( columnType.HasNull )
                {
                    if( ( columnType.DataType.ValueType == AriesValueType::CHAR ) && ( columnType.DataType.Length > 1 ) )
                    {
                        result = "( !is_null< true >( " + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral )
                                + ") ) ";
                    }
                    else
                        result = "( !is_null( " + child->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + ") ) ";
                }
                else
                {
                    result = "(1)";
                }
                break;
            }
            case AriesExprType::INTERVAL:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                string value = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, true );
                interval_type intervalType = get_interval_type( content_string );
                INTERVAL interval = getIntervalValue( value, intervalType );
                result = AriesIntervalToString( AriesInterval { intervalType, interval }, constValues );
                break;
            }
            case AriesExprType::BETWEEN:
            {
                ARIES_ASSERT( children.size() == 3, "children.size(): " + to_string( children.size() ) );
                string value = children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral );
                result = "( ( " + value + " >= " + children[ 1 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral )
                        + " ) and ( ";
                result += value + " <= " + children[ 2 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) )";
                break;
            }
            case AriesExprType::IN:
            case AriesExprType::NOT_IN:
            {
                size_t childCount = children.size();
                ARIES_ASSERT( childCount > 0, "children.size(): " + to_string( childCount ) );
                size_t currentSize = ariesComparators.size();
                string tmpName;
                if( children[ 0 ]->value_type.DataType.ValueType == AriesValueType::CHAR && children[ 0 ]->value_type.DataType.Length > 1 )
                {
                    result = " comparators[ " + to_string( currentSize ) + " ]->Compare( ( "
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ) ";
                }
                else
                {
                    tmpName = "tempVar_";
                    tmpName += to_string( seq );
                    result = " comparators[ " + to_string( currentSize ) + " ]->Compare( &( " + tmpName + " = ( "
                            + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) ) ) ";
                }
                AriesDataBufferSPtr buffer = ConvertLiteralArrayToDataBuffer( children[ 0 ]->value_type, AriesExprType::NOT_IN == type );
                ariesComparators.emplace_back( children[ 0 ]->value_type, buffer,
                        AriesExprType::IN == type ? AriesComparisonOpType::IN : AriesComparisonOpType::NOTIN, tmpName );
                break;
            }
            case AriesExprType::NOT:
            {
                ARIES_ASSERT( children.size() == 1, "children.size(): " + to_string( children.size() ) );
                result = "( not ( " + children[ 0 ]->stringForDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, ariesComparators, printConstAsLiteral ) + " ) )";
                break;
            }
            default:
                //FIXME need handle other expr type
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "expression type " + GetAriesExprTypeName( type ) + " in dynamic code" );
                break;
        }
        return result;
    }

    // 根据 树节点 生成整个树的 string 代码
    std::string AriesCommonExpr::stringForXmpDynamicCodeInternal( std::map< string, AriesCommonExprUPtr > &aggFunctions,
                                                               int &seq,
                                                               vector< AriesDynamicCodeParam >& ariesParams,
                                                               vector< AriesDataBufferSPtr >& constValues,
                                                               set< int >& interVar,
                                                               vector< AriesDynamicCodeComparator >& ariesComparators,
                                                               int ansLEN,
                                                               int ansTPI,
                                                               string &result,
                                                               bool printConstAsLiteral)
    {
        seq++;
        // 获取 该节点 生成的 string 代码
        int ansLIMBS = ansLEN / ansTPI;
        std::string content_string = contentToXmpString( ariesParams, constValues, ansLIMBS, printConstAsLiteral);
        switch( type )
        {
            case AriesExprType::INTEGER:
            case AriesExprType::FLOATING:
            case AriesExprType::COLUMN_ID:
            case AriesExprType::STAR:
            case AriesExprType::DECIMAL:
            case AriesExprType::DATE:
            case AriesExprType::TIME:
            case AriesExprType::TIMESTAMP:
            case AriesExprType::YEAR:
            case AriesExprType::DATE_TIME:
            case AriesExprType::NULL_VALUE:
            case AriesExprType::TRUE_FALSE:
            {
                result += std::move( content_string );
                break;
            }
            case AriesExprType::CALC:
            {
                ARIES_ASSERT( children.size() == 2, "children.size(): " + to_string( children.size() ) );
                // AriesCalculatorOpType opType = static_cast< AriesCalculatorOpType >( boost::get< int >( content ) );

                // 利用递归的方法 先将子树的变量定义后 再生成根节点的代码
                children[ 0 ]->stringForXmpDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, interVar, ariesComparators, ansLEN, ansTPI, result, printConstAsLiteral);
                children[ 1 ]->stringForXmpDynamicCodeInternal( aggFunctions, seq, ariesParams, constValues, interVar, ariesComparators, ansLEN, ansTPI, result, printConstAsLiteral);

                AriesColumnType leftValueType = children[0]->GetValueType();
                AriesColumnType rightValueType = children[1]->GetValueType();

                string ans_var_index = to_string(varIndexInKernel);
                string left_var_index = to_string(children[0]->varIndexInKernel);
                string right_var_index = to_string(children[1]->varIndexInKernel);

                string str_var_limb = "";
                string str_var_fun_name = "";

                if( ansLIMBS == 1)
                    str_var_limb = "LIMBS_ONE";
                else if ( ansLIMBS == 2)
                    str_var_limb = "LIMBS_TWO";
                else if ( ansLIMBS == 4)
                    str_var_limb = "LIMBS_THR";
                else if ( ansLIMBS == 8)
                    str_var_limb = "LIMBS_FOR";

                str_var_fun_name = AriesGetXmpCalcFunctionName(ansLEN, ansTPI);

                if( interVar.count(varIndexInKernel) == 0){
                    result += "       uint32_t var_" + ans_var_index + "["+ str_var_limb +"] = {0};\n";
                    result += "       uint8_t var_" + ans_var_index + "_sign = 0;\n";
                    interVar.insert(varIndexInKernel);
                }

                if( content_string == "+"){
                    result += "       var_" + ans_var_index + "_sign = aries_acc::operator_add"+ str_var_fun_name +"(var_"+ ans_var_index +", var_"+ left_var_index +", var_"+ right_var_index +", "+to_string(leftValueType.DataType.Scale-rightValueType.DataType.Scale)+", var_"+ left_var_index +"_sign, var_"+ right_var_index +"_sign);\n";
                }
                else if( content_string == "-"){
                    result += "       var_" + ans_var_index + "_sign = aries_acc::operator_sub"+ str_var_fun_name +"(var_"+ ans_var_index +", var_"+  left_var_index +", var_"+ right_var_index +", "+to_string(leftValueType.DataType.Scale-rightValueType.DataType.Scale)+", var_"+ left_var_index +"_sign, var_"+ right_var_index +"_sign);\n";
                }
                else if( content_string == "*"){
                    result += "       aries_acc::operator_mul"+ str_var_fun_name +"(var_"+ ans_var_index +", var_"+ left_var_index +", var_"+ right_var_index +");\n";
                    result += "       var_" + ans_var_index + "_sign = var_"+ left_var_index +"_sign ^ var_"+ right_var_index +"_sign;\n";
                }
                else if( content_string == "/"){
                    result += "       aries_acc::operator_div"+ str_var_fun_name +"(var_"+ ans_var_index +", var_"+ left_var_index +", var_"+ right_var_index+ ", "+ to_string(rightValueType.DataType.Scale+DIV_FIX_EX_FRAC)+");\n";
                    result += "       var_" + ans_var_index + "_sign = var_"+ left_var_index +"_sign ^ var_"+ right_var_index +"_sign;\n";
                }
                else if( content_string == "%"){
                    result += "       aries_acc::operator_mod"+ str_var_fun_name +"(var_"+ ans_var_index +", var_"+ left_var_index +", var_"+ right_var_index + ");\n";
                    result += "       var_" + ans_var_index + "_sign = var_"+ left_var_index +"_sign ^ var_"+ right_var_index +"_sign;\n";
                }
                break;
            }
            default:
                //FIXME need handle other expr type
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "expression type " + GetAriesExprTypeName( type ) + " in dynamic code" );
                break;
        }
        return result;
    }

    AriesAggFunctionType AriesCommonExpr::GetAggFunctionType() const
    {
        ARIES_ASSERT( type == AriesExprType::AGG_FUNCTION, "type: " + GetAriesExprTypeName( type ) );
        return static_cast< AriesAggFunctionType >( boost::get< int >( content ) );
    }

    AriesSqlFunctionType AriesCommonExpr::GetSqlFunctionType() const
    {
        ARIES_ASSERT( type == AriesExprType::SQL_FUNCTION, "type: " + GetAriesExprTypeName( type ) );
        return static_cast< AriesSqlFunctionType >( boost::get< int >( content ) );
    }

    std::string AriesCommonExpr::generateCompareFunctionHeader( AriesComparisonOpType op, bool leftHasNull, bool rightHasNull ) const
    {
        std::string leftNullStr = leftHasNull ? "true" : "false";
        std::string rightNullStr = rightHasNull ? "true" : "false";
        std::string name;
        switch( op )
        {
            case AriesComparisonOpType::EQ:
                name = " equal_to_t_str< " + leftNullStr + ", " + rightNullStr + " >()( ";
                break;
            case AriesComparisonOpType::NE:
                name = " not_equal_to_t_str< " + leftNullStr + ", " + rightNullStr + " >()( ";
                break;
            case AriesComparisonOpType::LT:
                name = " less_t_str< " + leftNullStr + ", " + rightNullStr + " >()( ";
                break;
            case AriesComparisonOpType::LE:
                name = " less_equal_t_str< " + leftNullStr + ", " + rightNullStr + " >()( ";
                break;
            case AriesComparisonOpType::GT:
                name = " greater_t_str< " + leftNullStr + ", " + rightNullStr + " >()( ";
                break;
            case AriesComparisonOpType::GE:
                name = " greater_equal_t_str< " + leftNullStr + ", " + rightNullStr + " >()( ";
                break;
            default:
                //TODO need support other compare types.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "generating compare function by type " + GetAriesComparisonOpTypeName( op ) );
                break;
        }
        return name;
    }

    AriesDataBufferSPtr AriesCommonExpr::ConvertLiteralArrayToDataBuffer( AriesColumnType dataType, bool bHasNot ) const
    {
        size_t childCount = children.size();
        ARIES_ASSERT( childCount > 1, "children.size(): " + to_string( childCount ) );

        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( dataType.DataType, false, false ) );
        result->AllocArray( childCount - 1 );
        size_t validItemCount = 0;

        for( std::size_t i = 1; i < childCount; ++i )
        {
            AriesExprType exprType = children[ i ]->type;
            AriesExpressionContent otherChildData = children[ i ]->content;
            if( exprType == AriesExprType::NULL_VALUE )
            {
                if( bHasNot )
                {
                    //如果not in的集合中包含null，那没有任何记录会满足条件
                    validItemCount = 0;
                    break;
                }
                else
                {
                    //如果 in集合中包含null,则剔除null值，以简化opt_in的操作符的实现。
                    continue;
                }
            }
            const type_info& type = otherChildData.type();

            /**
             * In 表达式的第 1 - N 是 In 的候选项
             * 当独立子查询出现在 In 表达式中时，In 的第 1 个 child 应该是 AriesDataBufferSPtr，且只能有 0 和 1 两个 child
             */
            if ( type == typeid( AriesDataBufferSPtr ) )
            {
                ARIES_ASSERT( childCount == 2, "children count must be 2" );
                return boost::get< AriesDataBufferSPtr >( otherChildData );
            }

            switch( dataType.DataType.ValueType )
            {
                case AriesValueType::DECIMAL:
                {
                    if( type == typeid(aries_acc::Decimal) )
                    {
                        *reinterpret_cast< aries_acc::Decimal* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< aries_acc::Decimal >(
                                otherChildData );
                    }
                    break;
                }
                case AriesValueType::DATE:
                {
                    if( type == typeid(AriesDate) )
                    {
                        *reinterpret_cast< AriesDate* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesDate >( otherChildData );
                    }
                    break;
                }
                case AriesValueType::DATETIME:
                {
                    if( type == typeid(AriesDatetime) )
                    {
                        *reinterpret_cast< AriesDatetime* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesDatetime >(
                                otherChildData );
                    }
                    break;
                }
                case AriesValueType::TIMESTAMP:
                {
                    if( type == typeid(AriesTimestamp) )
                    {
                        *reinterpret_cast< AriesTimestamp* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesTimestamp >(
                                otherChildData );
                    }
                    break;
                }
                case AriesValueType::TIME:
                {
                    if( type == typeid(AriesTime) )
                    {
                        *reinterpret_cast< AriesTime* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesTime >( otherChildData );
                    }
                    break;
                }
                case AriesValueType::YEAR:
                {
                    if( type == typeid(AriesYear) )
                    {
                        *reinterpret_cast< AriesYear* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesYear >( otherChildData );
                    }
                    break;
                }
                case AriesValueType::DOUBLE:
                {
                    if( type == typeid(double) )
                    {
                        *reinterpret_cast< double* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< double >( otherChildData );
                    }
                    break;
                }
                case AriesValueType::FLOAT:
                {
                    if( type == typeid(double) )
                    {
                        *reinterpret_cast< float* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< float >( boost::get< double >(
                                otherChildData ) );
                    }
                    break;
                }
                case AriesValueType::INT8:
                {
                    if( type == typeid(int32_t) )
                    {
                        int32_t val = boost::get< int32_t >( otherChildData );
                        if( val >= numeric_limits< signed char >::min() && val <= numeric_limits< signed char >::max() )
                        {
                            *reinterpret_cast< signed char* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< signed char >( val );
                        }
                    }
                    else if( type == typeid(int64_t) )
                    {
                        int64_t val = boost::get< int64_t >( otherChildData );
                        if( val >= numeric_limits< signed char >::min() && val <= numeric_limits< signed char >::max() )
                        {
                            *reinterpret_cast< signed char* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< signed char >( val );
                        }
                    }
                    break;
                }
                case AriesValueType::INT16:
                {
                    if( type == typeid(int32_t) )
                    {
                        int32_t val = boost::get< int32_t >( otherChildData );
                        if( val >= numeric_limits< short >::min() && val <= numeric_limits< short >::max() )
                        {
                            *reinterpret_cast< short* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< short >( val );
                        }
                    }
                    else if( type == typeid(int64_t) )
                    {
                        int64_t val = boost::get< int64_t >( otherChildData );
                        if( val >= numeric_limits< short >::min() && val <= numeric_limits< short >::max() )
                        {
                            *reinterpret_cast< short* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< short >( val );
                        }
                    }
                    break;
                }
                case AriesValueType::INT32:
                {
                    if( type == typeid(int) )
                    {
                        int value = boost::get< int >( otherChildData );
                        *reinterpret_cast< int* >( result->GetItemDataAt( validItemCount++ ) ) = value;
                    }
                    else if( type == typeid(int64_t) )
                    {
                        int64_t val = boost::get< int64_t >( otherChildData );
                        if( val >= numeric_limits< int >::min() && val <= numeric_limits< int >::max() )
                        {
                            *reinterpret_cast< int* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< int >( val );
                        }
                    }

                    break;
                }
                case AriesValueType::INT64:
                {
                    int64_t value;
                    if( type == typeid(int64_t) )
                    {
                        value = boost::get< int64_t >( otherChildData );
                    }
                    else if( type == typeid(int) )
                    {
                        value = boost::get< int >( otherChildData );
                    }
                    else
                    {
                        break;
                    }

                    *reinterpret_cast< long long* >( result->GetItemDataAt( validItemCount++ ) ) = value;
                    break;
                }
                case AriesValueType::UINT8:
                {
                    if( type == typeid(int) )
                    {
                        int val = boost::get< int >( otherChildData );
                        if( val >= numeric_limits< unsigned char >::min() && val <= numeric_limits< unsigned char >::max() )
                        {
                            *reinterpret_cast< unsigned char* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< unsigned char >( val );
                        }
                    }
                    break;
                }
                case AriesValueType::UINT16:
                {
                    if( type == typeid(int) )
                    {
                        int val = boost::get< int >( otherChildData );
                        if( val >= numeric_limits< unsigned short >::min() && val <= numeric_limits< unsigned short >::max() )
                        {
                            *reinterpret_cast< unsigned short* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< unsigned short >( val );
                        }
                    }
                    break;
                }
                case AriesValueType::UINT32:
                {
                    if( type == typeid(int) )
                    {
                        int val = boost::get< int >( otherChildData );
                        if( val >= 0 )
                        {
                            *reinterpret_cast< unsigned int* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< unsigned int >( val );
                        }
                    }
                    break;
                }
                case AriesValueType::UINT64:
                {
                    if( type == typeid(int) )
                    {
                        *reinterpret_cast< unsigned long long* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< int >( otherChildData );
                    }
                    break;
                }
                case AriesValueType::CHAR:
                {
                    if( type == typeid(string) )
                    {
                        string val = boost::get< string >( otherChildData );
                        size_t len = dataType.DataType.Length;
                        if( val.size() <= len )
                        {
                            val.resize( len, 0 );
                            memcpy( result->GetItemDataAt( validItemCount++ ), val.c_str(), len );
                        }
                    }
                    break;
                }
                default:
                {
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "converting data type " + GetValueTypeAsString( dataType ) + " for IN expression" );
                    break;
                }
            }
        }
        result->SetItemCount( validItemCount );
        if( validItemCount > 0 )
        {
            result->PrefetchToGpu();
            result = SortData( result, AriesOrderByType::ASC );
        }
        return result;
    }

    bool AriesCommonExpr::IsLiteralValue() const
    {
        switch( type )
        {
            case AriesExprType::INTEGER:
            case AriesExprType::STRING:
            case AriesExprType::FLOATING:
            case AriesExprType::DECIMAL:
            case AriesExprType::DATE:
            case AriesExprType::DATE_TIME:
            case AriesExprType::TIME:
            case AriesExprType::TIMESTAMP:
            case AriesExprType::NULL_VALUE:
            case AriesExprType::TRUE_FALSE:
            case AriesExprType::YEAR:
                return true;
            default:
                return false;
        }
    }

    bool AriesCommonExpr::IsEqualExpression() const
    {
        if ( type != AriesExprType::COMPARISON )
        {
            return false;
        }

        return static_cast< int >( AriesComparisonOpType::EQ ) == boost::get< int >( content );
    }

    bool AriesCommonExpr::IsAddExpression() const
    {
        if ( type != AriesExprType::AND_OR )
        {
            return false;
        }

        return static_cast< int >( AriesLogicOpType::AND ) == boost::get< int >( content );
    }

END_ARIES_ENGINE_NAMESPACE
// namespace aries
