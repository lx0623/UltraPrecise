/*
 * File: ExpressionSimplifier.cpp

 * File Created: 2019-08-23
 * Author: Jerry Hu (shenggang.hu@rateup.com.cn)
 */

#include <memory>

#include "ExpressionSimplifier.h"
#include "datatypes/AriesDatetimeTrans.h"
#include "datatypes/timefunc.h"
#include "datatypes/AriesMysqlInnerTime.h"
#include "datatypes/AriesTimeCalc.hxx"
#include "datatypes/decimal.hxx"
#include "datatypes/functions.hxx"
#include "datatypes/AriesTruncateFunctions.hxx"
#include "utils/string_util.h"
#include <server/mysql/include/mysqld_error.h>
#include "AriesAssert.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"

using namespace aries_utils;

BEGIN_ARIES_ENGINE_NAMESPACE

#define GET_TYPE_NAME( value, type, name )  \
if ( CHECK_VARIANT_TYPE( value, type) ) {   \
    return name;                            \
}

#define GET_CHILD_BY_INDEX( parent, index ) ( ( CommonBiaodashi* )( ( parent )->GetChildByIndex( index ).get() ) )

static string SIMPLIFIED_RESULT_TYPE_NAMES[] =
{
    "int",
    "double",
    "decimal",
    "date",
    "datetime",
    "timestamp",
    "char",
    "time",
    "bool",
    "null",
    "year",
    "bigint",
    "float"
};

static std::string get_result_type_name( const SimplifiedResult& result )
{
    GET_TYPE_NAME( result, int, "int" );
    GET_TYPE_NAME( result, double, "double" );
    GET_TYPE_NAME( result, aries_acc::Decimal, "decimal" );
    GET_TYPE_NAME( result, aries_acc::AriesDate, "date" );
    GET_TYPE_NAME( result, aries_acc::AriesDatetime, "datetime" );
    GET_TYPE_NAME( result, aries_acc::AriesTimestamp, "timestamp" );
    GET_TYPE_NAME( result, std::string, "string" );
    GET_TYPE_NAME( result, aries_acc::AriesTime, "time" );
    GET_TYPE_NAME( result, bool, "bool" );
    GET_TYPE_NAME( result, aries_acc::AriesNull, "null" );
    GET_TYPE_NAME( result, aries_acc::AriesYear, "year" );
    GET_TYPE_NAME( result, int64_t, "bigint" );
    GET_TYPE_NAME( result, float, "float" );
    ARIES_ASSERT( 0, ( std::string("unhandle type: ") + result.type().name() ).c_str() );
}

/**
 * @brief 从 SimplifiedResult 得到 timestamp 值, 只支持 AriesDate/AriesDatetime/AriesTimestamp
 * @param[in] value
 * @param[out] success 成功转换为 true，否则为 false
 * @return timestamp 值
 */
static int64_t get_timestamp_value( SimplifiedResult value, bool& success );

/**!
 * @brief 将 value 转换成 timestamp，支持 string/time/integer 到 timestamp 的转换
 * @param[in] value
 * @param[out] success 成功转换为 true，否则为 false
 * @return timestamp 值
 */
static int64_t convert_to_timestamp( SimplifiedResult value, bool& success );

AriesValueType static convertValueType(BiaodashiValueType type)
{
    static std::map<BiaodashiValueType, AriesValueType> map = {
        {BiaodashiValueType::UNKNOWN, AriesValueType::UNKNOWN},
        {BiaodashiValueType::BOOL, AriesValueType::BOOL},
        {BiaodashiValueType::FLOAT, AriesValueType::FLOAT},
        {BiaodashiValueType::INT, AriesValueType::INT32},
        {BiaodashiValueType::TINY_INT, AriesValueType::INT8},
        {BiaodashiValueType::SMALL_INT, AriesValueType::INT16},
        {BiaodashiValueType::LONG_INT, AriesValueType::INT64},
        {BiaodashiValueType::DOUBLE, AriesValueType::DOUBLE},
        {BiaodashiValueType::DECIMAL, AriesValueType::DECIMAL},
        {BiaodashiValueType::TEXT, AriesValueType::CHAR},
        {BiaodashiValueType::DATE, AriesValueType::DATE},
        {BiaodashiValueType::DATE_TIME, AriesValueType::DATETIME},
        {BiaodashiValueType::TIMESTAMP, AriesValueType::TIMESTAMP},
        {BiaodashiValueType::TIME, AriesValueType::TIME},
        {BiaodashiValueType::VARBINARY, AriesValueType::CHAR},
        {BiaodashiValueType::BINARY, AriesValueType::CHAR},
        {BiaodashiValueType::LIST, AriesValueType::LIST},
        {BiaodashiValueType::YEAR, AriesValueType::YEAR}};
    if (map.find(type) == map.end()) {
#ifndef NDEBUG
        std::cout << "type = " << static_cast<int>(type) << std::endl;
#endif
    }
    assert(map.find(type) != map.end());
    return map[type];
}

BiaodashiPointer ExpressionSimplifier::SimplifyAsCommonBiaodashi(CommonBiaodashi *origin, THD* thd)
{
    bool success = false;
    auto value = doSimplify(origin, thd, success);

    if (!success)
    {
        return nullptr;
    }
    return resultToBiaodashi( value );
}

AriesCommonExprUPtr ExpressionSimplifier::Simplify(CommonBiaodashi *origin, THD* thd)
{
    bool success = false;
    auto value = doSimplify(origin, thd, success);

    if (!success)
    {
        return nullptr;
    }

    auto expr = resultToAriesCommonExpr(value);
    auto value_type = expr->GetType() == AriesExprType::NULL_VALUE ? AriesColumnType(
        aries::AriesDataType{convertValueType(origin->GetValueType()), 1}, false, false
    ) : expr->GetValueType();
    value_type.HasNull = origin->IsNullable();

    expr->SetValueType(value_type);
    return expr;
}

static SimplifiedResult stringToNumeric(const std::string &string_value)
{
    if (string_value.find('.') != std::string::npos)
    {
        return aries_acc::Decimal(string_value.c_str());
    }
    else
    {
        return int64_t(std::stoll(string_value));
    }
}

static std::string simplifiedResultToString( const SimplifiedResult& result )
{
    std::string string_value;
    if ( IS_STRING_VARIANT( result ) )
    {
        string_value = boost::get< std::string >( result );
    }
    else if ( CHECK_VARIANT_TYPE( result, int ) )
    {
        string_value = std::to_string( boost::get< int >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, int64_t ) )
    {
        string_value = std::to_string( boost::get< int64_t >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, double ) )
    {
        string_value = std::to_string( boost::get< double >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, float ) )
    {
        string_value = std::to_string( boost::get< float >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::Decimal ) )
    {
        auto decimal = boost::get< aries_acc::Decimal >( result );
        char buffer[ 64 ] = { 0 };
        decimal.GetDecimal( buffer );
        string_value = std::string( buffer );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesDate ) )
    {
        auto date = boost::get< aries_acc::AriesDate >( result );
        string_value = aries_acc::AriesDatetimeTrans::GetInstance().ToString( date );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesDatetime ) )
    {
        auto datetime = boost::get< aries_acc::AriesDatetime >( result );
        string_value = aries_acc::AriesDatetimeTrans::GetInstance().ToString( datetime );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesTimestamp ) )
    {
        auto timestamp = boost::get< aries_acc::AriesTimestamp >( result );
        string_value = aries_acc::AriesDatetimeTrans::GetInstance().ToString( timestamp );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesTime ) )
    {
        auto time = boost::get< aries_acc::AriesTime >( result );
        string_value = aries_acc::AriesDatetimeTrans::GetInstance().ToString( time );
    }
    else if ( CHECK_VARIANT_TYPE( result, bool ) )
    {
        string_value = std::to_string( boost::get< bool >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesYear ) )
    {
        auto year = boost::get< aries_acc::AriesYear >( result );
        string_value = aries_acc::AriesDatetimeTrans::GetInstance().ToString( year );
    }
    else
    {
        ARIES_ASSERT( 0 , "should not be here, result type: " + get_result_type_name( result ) );
    }

    return string_value;
}

void ExpressionSimplifier::ConvertString(const std::vector<AriesValueType> &expect_types,
                                         AriesCommonExpr *constant_child)
{
    assert(!expect_types.empty());

    std::string string_content;
    switch (constant_child->GetType())
    {
    case AriesExprType::INTEGER:
    {
        if (constant_child->GetValueType().DataType.ValueType == AriesValueType::INT64) {
            auto int_content = boost::get<int64_t>(constant_child->GetContent());
            string_content = std::to_string(int_content);
        }
        else
        {
            auto int_content = boost::get<int>(constant_child->GetContent());
            string_content = std::to_string(int_content);
        }
        break;
    }
    case AriesExprType::DECIMAL:
    {
        auto decimal_content = boost::get<aries_acc::Decimal>(constant_child->GetContent());
        char tmp_buffer[64] = {0};
        decimal_content.GetDecimal(tmp_buffer);
        string_content = tmp_buffer;
        break;
    }
    case AriesExprType::FLOATING:
    {
        auto double_content = boost::get<double>(constant_child->GetContent());
        string_content = std::to_string(double_content);
        break;
    }
    case AriesExprType::STRING:
    {
        string_content = boost::get<std::string>(constant_child->GetContent());
        break;
    }
    default:
        return;
    }

    SimplifiedResult result;
    bool success = false;
    for ( const auto &candidate_type : expect_types )
    {
        try
        {
            switch (candidate_type)
            {
            case AriesValueType::DATE:
                result = AriesDatetimeTrans::GetInstance().ToAriesDate(string_content);
                success = true;
                break;
            case AriesValueType::DATETIME:
                result = AriesDatetimeTrans::GetInstance().ToAriesDatetime(string_content);
                success = true;
                break;
            case AriesValueType::TIMESTAMP:
                result = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(string_content);
                success = true;
                break;
            case AriesValueType::TIME:
                result = AriesDatetimeTrans::GetInstance().ToAriesTime(string_content);
                success = true;
                break;
            /**
             * 目标类型为 CHAR（string）也是可能的
             */
            case AriesValueType::CHAR:
                result = string_content;
                success = true;
                break;
            case AriesValueType::INT8:
            case AriesValueType::INT16:
            case AriesValueType::INT32:
            case AriesValueType::INT64:
            case AriesValueType::DECIMAL:
            case AriesValueType::FLOAT:
            case AriesValueType::DOUBLE:
                result = stringToNumeric(string_content);
                success = true;
                break;
            case AriesValueType::YEAR:
                result = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesYear(string_content);
                success = true;
                break;
            default:
                return;
            }
            if (success)
                break;
        }
        catch (const std::exception &e)
        {
            LOG(INFO) << "cannot convert to " << static_cast<int>(candidate_type) << " from "
                      << static_cast<int>(constant_child->GetValueType().DataType.ValueType)
                      << ", exception: " << e.what() << std::endl;
            continue;
        }
    }

    if (!success)
    {
        return;
    }

    auto new_expr = resultToAriesCommonExpr(result);
    constant_child->SetValueType(new_expr->GetValueType());
    constant_child->SetType(new_expr->GetType());
    constant_child->SetContent(new_expr->GetContent());

    LOG(INFO) << "Here converted: " << string_content << " to: " << static_cast<int>(constant_child->GetType())
              << std::endl;
}

void ExpressionSimplifier::ConvertConstantChildrenIfNeed(const AriesCommonExprUPtr &expr)
{
    std::vector<AriesValueType> expect_types;
    std::vector<AriesCommonExpr *> constant_children;
    AriesValueType candidate_type = AriesValueType::UNKNOWN;

    for (int i = 0; i < expr->GetChildrenCount(); i++)
    {
        if ((expr->GetType() == AriesExprType::CASE) && ( i == 0 || (i % 2 != 0 && i != (expr->GetChildrenCount() - 1))))
        {
            continue;
        }
        else if (expr->GetType() == AriesExprType::IF_CONDITION && i == 0)
        {
            continue;
        }
        else if (expr->GetType() == AriesExprType::SQL_FUNCTION)
        {
            auto function_type = static_cast<AriesSqlFunctionType>(boost::get<int>(expr->GetContent()));
            /**
             * 目前 unix_timestamp 的第二个参数是整数，无需转换
             */
            if (function_type == AriesSqlFunctionType::UNIX_TIMESTAMP && i == 1) {
                continue;
            }
        }

        const auto &child = expr->GetChild(i);
        auto child_type = child->GetValueType().DataType.ValueType;
        switch (child->GetType())
        {
        case AriesExprType::INTEGER:
        case AriesExprType::DECIMAL:
        case AriesExprType::FLOATING:
            if (candidate_type == AriesValueType::UNKNOWN)
            {
                candidate_type = child_type;
            }
        case AriesExprType::STRING:
            constant_children.emplace_back(child.get());
            break;
        case AriesExprType::NULL_VALUE:
            constant_children.emplace_back(child.get());
            break;
        default:
            candidate_type = child_type;
            break;
        }
    }

    if (constant_children.empty())
    {
        return;
    }

    switch (expr->GetType())
    {
    case AriesExprType::AGG_FUNCTION:
        expect_types.emplace_back(AriesValueType::INT32);
        expect_types.emplace_back(AriesValueType::INT64);
        expect_types.emplace_back(AriesValueType::DECIMAL);
        break;
    case AriesExprType::SQL_FUNCTION:
    {
        auto function_type = static_cast<AriesSqlFunctionType>(boost::get<int>(expr->GetContent()));
        switch (function_type)
        {
        case AriesSqlFunctionType::DATE_DIFF:
            expect_types.emplace_back(AriesValueType::DATE);
            expect_types.emplace_back(AriesValueType::DATETIME);
            break;
        case AriesSqlFunctionType::TIME_DIFF:
        {
            switch ( candidate_type )
            {
                case AriesValueType::DATETIME:
                case AriesValueType::DATE:
                case AriesValueType::TIME:
                    expect_types.emplace_back( candidate_type );
                    break;
                default:
                    ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "timediff" );
            }
            break;
        }
        case AriesSqlFunctionType::UNIX_TIMESTAMP:
            expect_types.emplace_back(AriesValueType::TIME);
            expect_types.emplace_back(AriesValueType::DATETIME);
            expect_types.emplace_back(AriesValueType::DATE);
            break;
        case AriesSqlFunctionType::ABS:
        case AriesSqlFunctionType::COUNT:
        case AriesSqlFunctionType::SUM:
        case AriesSqlFunctionType::MAX:
        case AriesSqlFunctionType::MIN:
        case AriesSqlFunctionType::AVG:
            expect_types.emplace_back(AriesValueType::INT32);
            expect_types.emplace_back(AriesValueType::INT64);
            expect_types.emplace_back(AriesValueType::DECIMAL);
            break;
        default:
            break;
        }
        break;
    }
    case AriesExprType::COMPARISON:
    case AriesExprType::CALC:
    case AriesExprType::BETWEEN:
        expect_types.emplace_back(candidate_type);
        break;
    case AriesExprType::IN:
    case AriesExprType::NOT_IN: {
        const auto &child = expr->GetChild(0);
        auto child_type = child->GetValueType().DataType.ValueType;
        expect_types.emplace_back(child_type);
        break;
    }
    default:
        break;
    }

    if (expect_types.empty())
    {
        return;
    }

    auto old_value_type = expr->GetValueType();
    AriesValueType converted_type = old_value_type.DataType.ValueType;
    for (const auto &child : constant_children)
    {
        auto found = std::find(expect_types.begin(), expect_types.end(), child->GetValueType().DataType.ValueType);
        if (found != expect_types.end())
        {
            continue;
        }

        if (child->GetType() == AriesExprType::NULL_VALUE)
        {
            if (expect_types.size() == 1)
            {
                child->SetValueType({{expect_types[0], 1}, true, false});
            }
            continue;
        }

        // TODO 这是为了 比较大的 定点数可以顺利进行运算
        if(child->GetType() != AriesExprType::DECIMAL && expect_types.size() == 1 && expect_types[0] != AriesValueType::COMPACT_DECIMAL)
            ConvertString(expect_types, child);
        converted_type = child->GetValueType().DataType.ValueType;
    }

    if (converted_type != old_value_type.DataType.ValueType && expr->GetType() == AriesExprType::CALC)
    {
        int new_type = max(static_cast<int>(converted_type), static_cast<int>(old_value_type.DataType.ValueType));
        old_value_type.DataType.ValueType = static_cast<AriesValueType>(new_type);
        expr->SetValueType(old_value_type);
    }
}

BiaodashiPointer ExpressionSimplifier::resultToBiaodashi(SimplifiedResult& result)
{
    CommonBiaodashiPtr expr;
    switch (result.which())
    {
    case 0:
        expr = std::make_shared< CommonBiaodashi >( BiaodashiType::Zhengshu, boost::get<int>( result) );
        expr->SetValueType( BiaodashiValueType::INT );
        break;
    case 1:
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Fudianshu, boost::get<double>(result) );
        expr->SetValueType( BiaodashiValueType::DOUBLE );
        break;
    case 2: // decimal
    {
        auto dec = boost::get<aries_acc::Decimal>(result);
        char decStr[64] = { 0 };
        dec.GetDecimal( decStr );
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Decimal, std::string( decStr ) );
        expr->SetValueType( BiaodashiValueType::DECIMAL );
        break;
    }
    case 3: // AriesDate
    {
        auto str = aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get<aries_acc::AriesDate>(result) );
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, str );
        break;
    }
    case 4: // AriesDatetime
    {
        auto str = aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get<aries_acc::AriesDatetime>(result) );
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, str );
        break;
    }
    case 5: // AriesTimestamp
    {
        auto str = aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get<aries_acc::AriesTimestamp>(result) );
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, str );
        break;
    }
    case 6: // string
    {
        auto string_value = boost::get<std::string>(result);
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, string_value );
        break;
    }
    case 7: // AriesTime
    {
        auto str = aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get<aries_acc::AriesTime>(result) );
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, str );
        break;
    }
    case 8: // bool
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, boost::get<bool>(result) );
        expr->SetValueType( BiaodashiValueType::BOOL );
        break;
    case 9: // AriesNull
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Null, 0 );
        break;
    case 10: // AriesYear
    {
        auto str = aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get<aries_acc::AriesYear>(result) );
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, str );
        break;
    }
    case 11: // int64_t
        expr = std::make_shared< CommonBiaodashi >( BiaodashiType::Zhengshu, boost::get<int64_t>(result) );
        expr->SetValueType( BiaodashiValueType::LONG_INT );
        break;
    case 12:
        expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Fudianshu, ( double )boost::get<float>(result) );
        expr->SetValueType( BiaodashiValueType::DOUBLE );
        break;
    }
    return expr;
}

AriesCommonExprUPtr ExpressionSimplifier::resultToAriesCommonExpr(SimplifiedResult &result)
{
    AriesCommonExprUPtr expr = nullptr;
    switch (result.which())
    {
    case 0:
        expr =
            AriesCommonExpr::Create(AriesExprType::INTEGER, boost::get<int>(result),
                                    AriesColumnType(aries::AriesDataType{AriesValueType::INT32, 1}, false, false));
        break;
    case 1:
        expr =
            AriesCommonExpr::Create(AriesExprType::FLOATING, boost::get<double>(result),
                                    AriesColumnType(aries::AriesDataType{AriesValueType::DOUBLE, 1}, false, false));
        break;
    case 2: // decimal
        expr = AriesCommonExpr::Create(
            AriesExprType::DECIMAL, boost::get<aries_acc::Decimal>(result),
            AriesColumnType(aries::AriesDataType{AriesValueType::DECIMAL, 1}, false, false));
        break;
    case 3: // AriesDate
        expr =
            AriesCommonExpr::Create(AriesExprType::DATE, boost::get<aries_acc::AriesDate>(result),
                                    AriesColumnType(aries::AriesDataType{AriesValueType::DATE, 1}, false, false));
        break;
    case 4: // AriesDatetime
        expr = AriesCommonExpr::Create(
            AriesExprType::DATE_TIME, boost::get<aries_acc::AriesDatetime>(result),
            AriesColumnType(aries::AriesDataType{AriesValueType::DATETIME, 1}, false, false));
        break;
    case 5: // AriesTimestamp
        expr = AriesCommonExpr::Create(
            AriesExprType::TIMESTAMP, boost::get<aries_acc::AriesTimestamp>(result),
            AriesColumnType(aries::AriesDataType{AriesValueType::TIMESTAMP, 1}, false, false));
        break;
    case 6: // string
    {
        auto string_value = boost::get<std::string>(result);
        expr = AriesCommonExpr::Create(
            AriesExprType::STRING, string_value,
            AriesColumnType(aries::AriesDataType{AriesValueType::CHAR, static_cast<int>(string_value.size())},
                            false, false));
        break;
    }
    case 7: // AriesTime
    {
        expr =
            AriesCommonExpr::Create(AriesExprType::TIME, boost::get<aries_acc::AriesTime>(result),
                                    AriesColumnType(aries::AriesDataType{AriesValueType::TIME, 1}, false, false));
        break;
    }
    case 8: // bool
        expr =
                AriesCommonExpr::Create(AriesExprType::TRUE_FALSE, boost::get<bool>(result),
                                        AriesColumnType(aries::AriesDataType{AriesValueType::BOOL, 1}, false, false));
        break;
    case 9: // AriesNull
        expr =
                AriesCommonExpr::Create(AriesExprType::NULL_VALUE, 0,
                                        AriesColumnType(aries::AriesDataType{AriesValueType::UNKNOWN, 1}, false, false));
            break;
    case 10: // AriesYear
        expr =
                AriesCommonExpr::Create(AriesExprType::YEAR, boost::get<aries_acc::AriesYear>(result),
                                        AriesColumnType(aries::AriesDataType{AriesValueType::YEAR, 1}, false, false));
            break;
    case 11: // int64_t
        expr =
                AriesCommonExpr::Create(AriesExprType::INTEGER, boost::get<int64_t>(result),
                                        AriesColumnType(aries::AriesDataType{AriesValueType::INT64, 1}, false, false));
            break;
    case 12:
        expr =
            AriesCommonExpr::Create(AriesExprType::FLOATING, boost::get<float>(result),
                                    AriesColumnType(aries::AriesDataType{AriesValueType::FLOAT, 1}, false, false));
        break;
    }
    return expr;
}

aries_acc::AriesBool ExpressionSimplifier::conditionValue(const SimplifiedResult& condition, bool &success) {
    aries_acc::AriesBool result(aries_acc::AriesBool::ValueType::Unknown);

    if (CHECK_VARIANT_TYPE(condition, aries_acc::AriesNull)) {
        return result;
    } else if (CHECK_VARIANT_TYPE(condition, int)) {
        auto int_value = boost::get<int>(condition);
        return int_value != 0;
    } else if (IS_STRING_VARIANT(condition)) {
        auto string_value = boost::get< std::string >(condition);
        return std::atoi( string_value.c_str() ) != 0;
    } else if (CHECK_VARIANT_TYPE(condition, bool)) {
        return boost::get<bool>(condition);
    }
    else if ( CHECK_VARIANT_TYPE( condition, aries_acc::AriesDatetime ) )
    {
        auto datetime = boost::get< aries_acc::AriesDatetime >( condition );
        return AriesDatetimeTrans::GetInstance().ToBool( datetime );
    }
    else if ( CHECK_VARIANT_TYPE( condition, aries_acc::AriesDate ) )
    {
        auto date = boost::get< aries_acc::AriesDate >( condition );
        return AriesDatetimeTrans::GetInstance().ToBool( date );
    }
    else if ( CHECK_VARIANT_TYPE( condition, aries_acc::AriesTime ) )
    {
        auto time = boost::get< aries_acc::AriesTime >( condition );
        return AriesDatetimeTrans::GetInstance().ToBool( time );
    }
    else if ( CHECK_VARIANT_TYPE( condition, aries_acc::AriesTimestamp ) )
    {
        auto timestamp = boost::get< aries_acc::AriesTimestamp >( condition );
        return AriesDatetimeTrans::GetInstance().ToBool( timestamp );
    }
    else if ( CHECK_VARIANT_TYPE( condition, int64_t ) )
    {
        auto value = boost::get< int64_t >( condition );
        return value != 0;
    }
    else if ( CHECK_VARIANT_TYPE( condition, double ) )
    {
        auto value = boost::get< double >( condition );
        return value != 0;
    }
    else if ( CHECK_VARIANT_TYPE( condition, aries_acc::AriesYear ) )
    {
        auto value = boost::get< aries_acc::AriesYear >( condition );
        return AriesDatetimeTrans::GetInstance().ToBool( value );
    }
    else if ( CHECK_VARIANT_TYPE( condition, float ) )
    {
        auto value = boost::get< float >( condition );
        return value != 0;
    }
    else
    {
        ARIES_ASSERT( 0, "unhandled simplified result type" );
    }
    return result;
}

SimplifiedResult ExpressionSimplifier::doSimplify(CommonBiaodashi *origin, THD* thd, bool &success)
{
    switch (origin->GetType())
    {
    case BiaodashiType::SQLFunc:
        return simplifyFunction(origin, thd, success);
    case BiaodashiType::Zhengshu:
    case BiaodashiType::Zifuchuan:
    case BiaodashiType::Fudianshu:
    case BiaodashiType::Decimal:
    case BiaodashiType::Zhenjia:
    case BiaodashiType::Null:
        return simplifyConstant(origin, thd, success);
    case BiaodashiType::IsNull: {
        return simplifyIsNull(origin, thd, success);
    }
    case BiaodashiType::IsNotNull: {
        return simplifyIsNotNull(origin, thd, success);
    }
    case BiaodashiType::Andor: {
        return simplifyAndOr(origin, thd, success);
        break;
    }
    case BiaodashiType::Bijiao: {
        return simplifyCompare(origin, thd, success);
        break;
    }
    case BiaodashiType::IfCondition: {
        return handleIfCondition(origin, thd, success);
    }
    case BiaodashiType::Case: {
        return handleCase(origin, thd, success);
    }
    case BiaodashiType::Between:
    {
        return handleBetween( origin, thd, success );
    }
    case BiaodashiType::Qiufan:
    {
        return handleQiufan( origin, thd, success );
    }
    case BiaodashiType::Inop: {
        return handleIn( origin, thd, success );
    }
    case BiaodashiType::NotIn: {
        return handleNotIn( origin, thd, success );
    }
    case BiaodashiType::Cunzai:
        return handleExists( origin, thd, success );
    case BiaodashiType::Likeop:
        return simplifyLike( origin, thd, success );
    case BiaodashiType::Yunsuan:
        return simplifyCalc(origin, thd, success);
    default:
        success = false;
        return false;
    }
}

SimplifiedResult ExpressionSimplifier::simplifyConstant(CommonBiaodashi *origin, THD* thd, bool &success)
{
    success = true;
    switch (origin->GetType())
    {
    case BiaodashiType::Zhengshu:
        if (origin->GetValueType() == BiaodashiValueType::LONG_INT)
        {
            return boost::get<int64_t>(origin->GetContent());
        }
        else
        {
            return boost::get<int>(origin->GetContent());
        }
    case BiaodashiType::Zifuchuan:
        return boost::get<std::string>(origin->GetContent());
    case BiaodashiType::Fudianshu:
        return boost::get<double>(origin->GetContent());
    case BiaodashiType::Decimal:
    {
        // 解决 subquery 返回值为 Decimal 字符串
        auto value = boost::get<std::string>(origin->GetContent());
        const char* pStr = value.data();
        auto strLen = strlen( pStr );
        if( *(pStr+strLen-1) == ')'){
            return aries_acc::Decimal(value.c_str(), strLen, true);
        }
        else{
            CheckDecimalPrecision( value );
            return aries_acc::Decimal(value.c_str());
        }
    }
    case BiaodashiType::Zhenjia:
        return boost::get<bool>(origin->GetContent());
    case BiaodashiType::Null:
        return aries_acc::AriesNull();
    default:
    {
        string msg( "unexpected expr type: " );
        msg.append( get_name_of_expr_type( origin->GetType() ) );
        ARIES_ASSERT( 0, msg );
        return false;
    }
    }

    success = false;
    return false;
}

SimplifiedResult ExpressionSimplifier::handleDateSubOrAdd(CommonBiaodashi *origin, THD* thd, bool &success)
{
    auto first = GET_CHILD_BY_INDEX( origin, 0 );

    auto first_ret = doSimplify(first, thd, success);
    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( first_ret, aries_acc::AriesNull )  ) // null
    {
        return first_ret;
    }

    MYSQL_TIME my_time = {0};
    if (IS_ARIESDATE_VARIANT(first_ret))
    {
        auto aries_date = boost::get<AriesDate>(first_ret);
        my_time.day = aries_date.day;
        my_time.year = aries_date.year;
        my_time.month = aries_date.month;
        my_time.time_type = enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE;
    }
    else if (IS_ARIESDATETIME_VARIANT(first_ret))
    {
        auto aries_datetime = boost::get<AriesDatetime>(first_ret);
        my_time.year = aries_datetime.year;
        my_time.month = aries_datetime.month;
        my_time.day = aries_datetime.day;
        my_time.hour = aries_datetime.hour;
        my_time.minute = aries_datetime.minute;
        my_time.second = aries_datetime.second;
        my_time.time_type = enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME;
    }
    else if (IS_STRING_VARIANT(first_ret))
    {
        int was_cut = 0;
        auto string_value = boost::get<std::string>(first_ret);
        auto type_itme = str_to_datetime(string_value.c_str(), string_value.size(), &my_time, 0, &was_cut);

        switch (type_itme)
        {
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE:
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME:
            break;
        default:
            assert(0);
            break;
        }
    }

    auto second = GET_CHILD_BY_INDEX( origin, 1 );

    std::string interval_type;
    std::string interval_value;
    if (second->GetType() == BiaodashiType::IntervalExpression)
    {
        interval_type = boost::get<std::string>(second->GetContent());
        auto value_expression = std::dynamic_pointer_cast<CommonBiaodashi>(second->GetChildByIndex(0));
        auto content = value_expression->GetContent();
        if (value_expression->GetType() == BiaodashiType::Zhengshu)
        {
            interval_value = std::to_string(boost::get<int>(content));
        }
        else if (value_expression->GetType() == BiaodashiType::Zifuchuan)
        {
            interval_value = boost::get<std::string>(content);
        }
        else
        {
            auto ret = doSimplify(value_expression.get(), thd, success);
            if (!success)
            {
                return false;
            }

            if (CHECK_VARIANT_TYPE(ret, std::string))
            {
                interval_value = boost::get<std::string>(ret);
            }
            else if (CHECK_VARIANT_TYPE(ret, int))
            {
                interval_value = std::to_string(boost::get<int>(ret));
            }
            else
            {
                success = false;
                return false;
            }
        }
    }
    else if (second->GetType() == BiaodashiType::Zhengshu)
    {
        interval_type = "DAY";
        if (second->GetValueType() == BiaodashiValueType::LONG_INT)
        {
            interval_value = std::to_string(boost::get<int64_t>(second->GetContent()));
        }
        else
        {
            interval_value = std::to_string(boost::get<int>(second->GetContent()));
        }
    }
    else if (second->GetType() == BiaodashiType::Zifuchuan)
    {
        interval_type = "DAY";
        interval_value = boost::get<int>(second->GetContent());
    }
    else
    {
        LOG(INFO) << "invalid interval expression type: " << static_cast<int>(second->GetType()) << std::endl;
        success = false;
        return false;
    }

    auto my_interval = getIntervalValue(interval_value, interval_type);
    auto my_interval_type = get_interval_type(interval_type);

    if (boost::get<SQLFunctionPointer>(origin->GetContent())->GetType() == AriesSqlFunctionType::DATE_SUB)
    {
        my_interval.neg = !my_interval.neg;
    }

    date_add_interval(&my_time, my_interval_type, my_interval);

    switch (my_time.time_type)
    {
    case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE:
    {
        AriesDate aries_date(my_time.year, my_time.month, my_time.day);
        LOG(INFO) << "return one date: " << static_cast<int>(aries_date.year) << "-"
                  << static_cast<int>(aries_date.month) << "-" << static_cast<int>(aries_date.day) << std::endl;
        return aries_date;
    }
    case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME:
    {
        AriesDatetime aries_datetime(my_time.year, my_time.month, my_time.day, my_time.hour, my_time.minute,
                                     my_time.second, 0);
        LOG(INFO) << "return one datetime: " << static_cast<int>(aries_datetime.year) << "-"
                  << static_cast<int>(aries_datetime.month) << "-" << static_cast<int>(aries_datetime.day) << " "
                  << static_cast<int>(aries_datetime.hour) << ":" << static_cast<int>(aries_datetime.minute) << ":"
                  << static_cast<int>(aries_datetime.second) << std::endl;
        return aries_datetime;
    }
    default:
        LOG(INFO) << "unsupported mysql time type: " << (int)my_time.time_type << std::endl;
        ARIES_ASSERT(0, "unsupported mysql time type" );
    }
}

SimplifiedResult ExpressionSimplifier::handleDate(CommonBiaodashi *origin, THD* thd, bool &success)
{
    auto child = GET_CHILD_BY_INDEX( origin, 0 );
    auto child_ret = doSimplify(child, thd, success);

    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( child_ret, aries_acc::AriesNull )  ) // null
    {
        return child_ret;
    }

    if ( CHECK_VARIANT_TYPE( child_ret, int ) || CHECK_VARIANT_TYPE( child_ret, int64_t ) )
    {
        child_ret = std::to_string( CHECK_VARIANT_TYPE( child_ret, int ) ?
            boost::get< int >( child_ret ) : boost::get< int64_t >( child_ret ) );
    }
    else if ( CHECK_VARIANT_TYPE( child_ret, float ) || CHECK_VARIANT_TYPE( child_ret, double ) )
    {
        child_ret = std::to_string( CHECK_VARIANT_TYPE( child_ret, float ) ?
            boost::get< float >( child_ret ) : boost::get< double >( child_ret ) );
    }
    else if ( CHECK_VARIANT_TYPE( child_ret, aries_acc::Decimal ) )
    {
        auto decimal = boost::get< aries_acc::Decimal >( child_ret );
        char tmp[ 64 ] = { 0 };
        child_ret = std::string( decimal.GetDecimal( tmp ) );
    }

    if (IS_STRING_VARIANT(child_ret))
    {
        auto string_value = boost::get<std::string>(child_ret);

        aries_acc::MYSQL_TIME my_time;
        int was_cut = 0;
        auto time_type = str_to_datetime(string_value.c_str(), string_value.size(), &my_time, 0, &was_cut);
        switch (time_type)
        {
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_NONE:
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_ERROR:
            return AriesNull();
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE:
            return AriesDate(my_time.year, my_time.month, my_time.day);
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME:
            return AriesDatetime(my_time.year, my_time.month, my_time.day, my_time.hour, my_time.minute, my_time.second,
                                 0);
        default:
        {
            string msg( "unexpected timestamp type: " );
            msg.append( std::to_string( ( int )time_type ) );
            ARIES_ASSERT( 0, msg );
        }
        }
    }
    else if (IS_ARIESDATETIME_VARIANT(child_ret))
    {
        auto date_time = boost::get<AriesDatetime>(child_ret);
        AriesDate date(date_time.getYear(), date_time.getMonth(), date_time.getDay());
        return date;
    }
    else if (IS_ARIESDATE_VARIANT(child_ret))
    {
        return boost::get<AriesDate>(child_ret);
    }
    else
    {
        return aries_acc::AriesNull();
    }
}

SimplifiedResult ExpressionSimplifier::handleDateDiff(CommonBiaodashi *origin, THD* thd, bool &success)
{
    auto param1 = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(0));
    auto param2 = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(1));

    auto param1_ret = doSimplify(param1.get(), thd, success);
    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( param1_ret, aries_acc::AriesNull )  ) // null
    {
        return param1_ret;
    }

    auto param2_ret = doSimplify(param2.get(), thd, success);
    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( param2_ret, aries_acc::AriesNull )  ) // null
    {
        return param2_ret;
    }

    AriesDate date1;
    AriesDate date2;
    if (IS_STRING_VARIANT(param1_ret))
    {
        auto string_value = boost::get<std::string>(param1_ret);
        date1 = AriesDatetimeTrans::GetInstance().ToAriesDate(string_value);
    }
    else if (IS_ARIESDATETIME_VARIANT(param1_ret))
    {
        auto datetime = boost::get<AriesDatetime>(param1_ret);
        date1.day = datetime.day;
        date1.month = datetime.month;
        date1.year = datetime.year;
    }
    else if (IS_ARIESDATE_VARIANT(param1_ret))
    {
        date1 = boost::get<AriesDate>(param1_ret);
    }
    else
    {
        LOG(INFO) << "Invalid param1 type: " << param1_ret.which() << std::endl;
        assert(0);
    }

    if (IS_STRING_VARIANT(param2_ret))
    {
        auto string_value = boost::get<std::string>(param2_ret);
        date2 = AriesDatetimeTrans::GetInstance().ToAriesDate(string_value);
    }
    else if (IS_ARIESDATETIME_VARIANT(param2_ret))
    {
        auto datetime = boost::get<AriesDatetime>(param2_ret);
        date2.day = datetime.day;
        date2.month = datetime.month;
        date2.year = datetime.year;
    }
    else if (IS_ARIESDATE_VARIANT(param2_ret))
    {
        date2 = boost::get<AriesDate>(param2_ret);
    }
    else
    {
        LOG(INFO) << "Invalid param2 type: " << param2_ret.which() << std::endl;
        assert(0);
    }

    success = true;
    return DATEDIFF(date1, date2);
}

SimplifiedResult ExpressionSimplifier::handleTimeDiff(CommonBiaodashi *origin, THD* thd, bool &success)
{
    auto param1 = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(0));
    auto param2 = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(1));

    auto param1_ret = doSimplify(param1.get(), thd, success);
    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( param1_ret, aries_acc::AriesNull )  ) // null
    {
        return param1_ret;
    }

    auto param2_ret = doSimplify(param2.get(), thd, success);
    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( param2_ret, aries_acc::AriesNull )  ) // null
    {
        return param2_ret;
    }

    aries_acc::MYSQL_TIME my_time1 = {0};
    aries_acc::MYSQL_TIME my_time2 = {0};
    if (IS_STRING_VARIANT(param1_ret))
    {
        auto string_value = boost::get<std::string>(param1_ret);
        int was_cut = 0;
        aries_acc::str_to_datetime(string_value.c_str(), string_value.size(), &my_time1, 0, &was_cut);
    }
    else if (IS_ARIESDATETIME_VARIANT(param1_ret))
    {
        aries_acc::AriesDatetime time = boost::get<AriesDatetime>(param1_ret);
        my_time1.year = time.year;
        my_time1.month = time.month;
        my_time1.day = time.day;
        my_time1.hour = time.hour;
        my_time1.minute = time.minute;
        my_time1.second = time.second;
        my_time1.time_type = aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME;
    }
    else if (IS_ARIESDATE_VARIANT(param1_ret))
    {
        auto time_value = boost::get<AriesDate>(param1_ret);
        my_time1.year = time_value.year;
        my_time1.month = time_value.month;
        my_time1.day = time_value.day;
        my_time1.time_type = aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE;
    }
    else
    {
        LOG(INFO) << "Invalid param1 type: " << param1_ret.which() << std::endl;
        assert(0);
    }

    if (IS_STRING_VARIANT(param2_ret))
    {
        auto string_value = boost::get<std::string>(param2_ret);
        int was_cut = 0;
        aries_acc::str_to_datetime(string_value.c_str(), string_value.size(), &my_time2, 0, &was_cut);
    }
    else if (IS_ARIESDATETIME_VARIANT(param2_ret))
    {
        auto time = boost::get<AriesDatetime>(param2_ret);
        my_time2.year = time.year;
        my_time2.month = time.month;
        my_time2.day = time.day;
        my_time2.hour = time.hour;
        my_time2.minute = time.minute;
        my_time2.second = time.second;
        my_time2.time_type = aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME;
    }
    else if (IS_ARIESDATE_VARIANT(param2_ret))
    {
        auto time_value = boost::get<AriesDate>(param2_ret);
        my_time1.year = time_value.year;
        my_time1.month = time_value.month;
        my_time1.day = time_value.day;
        my_time1.time_type = aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE;
    }
    else
    {
        LOG(INFO) << "Invalid param2 type: " << param2_ret.which() << std::endl;
        assert(0);
    }

    success = true;

    if (my_time1.time_type == my_time2.time_type)
    {
        switch (my_time1.time_type)
        {
        case aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE:
            return aries_acc::TIMEDIFF(aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate(my_time1),
                                       aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate(my_time2));
        case aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME:
            return aries_acc::TIMEDIFF(aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime(my_time1),
                                       aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime(my_time2));
        default:
            break;
        }
    }

    LOG(INFO) << "invalid param for timediff: ( " << param1->ToString() << ", " << param2->ToString() << " )"
              << std::endl;
    ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "timediff" );
}

SimplifiedResult ExpressionSimplifier::handleNow(CommonBiaodashi *origin, THD* thd, bool &success)
{
    auto now = AriesDatetimeTrans::Now();
    success = true;
    return now;
}

SimplifiedResult ExpressionSimplifier::simplifyFunction(CommonBiaodashi *origin, THD* thd, bool &success)
{
    auto function = boost::get<SQLFunctionPointer>(origin->GetContent());
    switch (function->GetType())
    {
    case AriesSqlFunctionType::ABS:
        return handleABS(origin, thd, success);
    case AriesSqlFunctionType::DATE:
        return handleDate(origin, thd, success);
    case AriesSqlFunctionType::DATE_ADD:
    case AriesSqlFunctionType::DATE_SUB:
        return handleDateSubOrAdd(origin, thd, success);
    case AriesSqlFunctionType::DATE_DIFF:
        return handleDateDiff(origin, thd, success);
    case AriesSqlFunctionType::TIME_DIFF:
        return handleTimeDiff(origin, thd, success);
    case AriesSqlFunctionType::NOW:
        return handleNow(origin, thd, success);
    case AriesSqlFunctionType::UNIX_TIMESTAMP:
        return handleUnixTimestamp(origin, thd, success);
    case AriesSqlFunctionType::CAST:
        return handleCast(origin, thd, success);
    case AriesSqlFunctionType::DATE_FORMAT:
        return handleDateFormat(origin, thd, success);
    case AriesSqlFunctionType::COALESCE:
        return handleCoalesce(origin, thd, success);
    case AriesSqlFunctionType::SUBSTRING: {
        return handleSubstring(origin, thd, success);
    }
    case AriesSqlFunctionType::CONCAT: {
        return handleConcat(origin, thd, success);
    }
    case AriesSqlFunctionType::TRUNCATE: {
        return handleTruncate( origin, thd, success );
    }
    case AriesSqlFunctionType::MONTH:
    {
        return handleMonth( origin, thd, success );
    }
    case AriesSqlFunctionType::EXTRACT:
    {
        return handleExtract( origin, thd, success );
    }
    default:
        success = false;
        return false;
    }
}

SimplifiedResult ExpressionSimplifier::handleABS(CommonBiaodashi *origin, THD* thd, bool &success)
{
    if (origin->GetChildrenCount() != 1)
    {
        LOG(INFO) << "ABS's must have only one param" << std::endl;
        success = false;
        return false;
    }

    CommonBiaodashi *param = GET_CHILD_BY_INDEX( origin, 0 );

    auto result = doSimplify(param, thd, success);

    // int, double, aries_acc::Decimal, aries_acc::AriesDate, aries_acc::AriesDatetime, std::string,
    // aries_acc::AriesTime
    if (!success)
    {
        return false;
    }
    if (  CHECK_VARIANT_TYPE( result, aries_acc::AriesNull )  ) // null
    {
        return result;
    }

    switch (result.which())
    {
    case 0: // int
        return abs(boost::get<int>(result));
    case 1: // double
        return abs(boost::get<double>(result));
    case 2: // decimal
        return abs(boost::get<aries_acc::Decimal>(result));
    default:
        break;
    }

    success = false;
    return false;
}

SimplifiedResult ExpressionSimplifier::simplifyLike(CommonBiaodashi* origin, THD* thd, bool& success)
{
    assert(origin->GetChildrenCount() == 2);
    success = false;
    CommonBiaodashi *leftChild = GET_CHILD_BY_INDEX( origin, 0 );
    CommonBiaodashi *rightChild = GET_CHILD_BY_INDEX( origin, 1 );

    bool leftSuccess = false, rightSuccess = false;
    SimplifiedResult left  = doSimplify(leftChild, thd, leftSuccess);
    SimplifiedResult right = doSimplify(rightChild, thd, rightSuccess);
    if( leftSuccess && rightSuccess )
    {
        if( CHECK_VARIANT_TYPE( left, aries_acc::AriesNull ) ) // null
            return left;
        if( CHECK_VARIANT_TYPE( right, aries_acc::AriesNull ) ) // null
            return right;
        if( left.which() == 6 && right.which() == 6 ) // string
        {
            string leftVal = boost::get< string >( left );
            string rightVal = boost::get< string >( right );
            success = true;
            return aries_acc::str_like( leftVal.data(), rightVal.data(), leftVal.size() );
        }
    }

    return false;
}

SimplifiedResult ExpressionSimplifier::simplifyIsNull(CommonBiaodashi* origin, THD* thd, bool& success)
{
    assert(origin->GetChildrenCount() == 1);
    CommonBiaodashi *child = GET_CHILD_BY_INDEX( origin, 0 );
    auto result = doSimplify(child, thd, success);
    if ( !success )
    {
        return false;
    }

    return CHECK_VARIANT_TYPE(result, aries_acc::AriesNull);
}

SimplifiedResult ExpressionSimplifier::simplifyIsNotNull(CommonBiaodashi* origin, THD* thd, bool& success)
{
    assert(origin->GetChildrenCount() == 1);
    CommonBiaodashi *child = GET_CHILD_BY_INDEX( origin, 0 );
    auto result = doSimplify(child, thd, success);
    if ( !success )
    {
        return false;
    }
    return !CHECK_VARIANT_TYPE(result, aries_acc::AriesNull);
}

static SimplifiedResult ToBooleanValue(const SimplifiedResult& sr)
{
    SimplifiedResult ret;
    switch ( sr.which() )
    {
        case 0: // int
            ret = (bool) boost::get<int>( sr );
            break;
        case 1: // double
            ret = (bool) boost::get<double>( sr );
            break;
        case 2: // decimal
        {
            auto v = boost::get<aries_acc::Decimal>( sr );
            ret = (v != (int32_t)0);
            break;
        }
        /*
        MySQL:
        mysql> select cast("0000-00-00" as date);
        +----------------------------+
        | cast("0000-00-00" as date) |
        +----------------------------+
        | NULL                       |
        +----------------------------+
        1 row in set, 1 warning (0.00 sec)
        mysql> select not cast("0000-00-00" as date);
        +--------------------------------+
        | not cast("0000-00-00" as date) |
        +--------------------------------+
        |                           NULL |
        +--------------------------------+
        1 row in set, 1 warning (0.00 sec)
        */
        case 3: // AriesDate
        case 4: // AriesDatetime
        case 5: // AriesTimestamp
        case 7: // AriesTime
        case 10: // AriesYear
        {
            ret = true;
            break;
        }
        case 6: // string
        {
            string v = boost::get<string> ( sr );
            int32_t i = 1;
            try
            {
                i = std::stoi( v );
            }
            catch( std::invalid_argument &e )
            {
                i = 0;
            }
            catch( ... )
            {
                // ignore any other exceptions
            }
            ret = (bool) i;
            break;
        }
        case 8: // bool
            ret = sr;
            break;
        case 9: // null
            ret = sr;
            break;
        case 11: // int64
            ret = ( bool ) boost::get<int64_t>( sr );
            break;
        case 12: // float
            ret = ( bool ) boost::get<float>( sr );
            break;
        default:
            ARIES_ASSERT( 0, "unexpected value type: " + std::to_string( sr.which() ) );
            break;
    }
    ARIES_ASSERT( ( 8 == ret.which() ) || ( 9 == ret.which() ),
                  "bool value type: " + std::to_string(ret.which()) );
    return ret;
}

static SimplifiedResult NullAndBoolLogic(LogicType logicType, bool b)
{
    SimplifiedResult result;
    switch ( logicType ) {
        case LogicType::AND:
            if ( b ) // true and null
            {
                result = AriesNull();
            }
            else // false and null
            {
                result = false;
            }
            break;
        case LogicType::OR:
            if ( b ) // null or true
            {
                result = true;
            }
            else // null or false
            {
                result = AriesNull();
            }
            break;
    }
    return result;
}

static SimplifiedResult DoLogicWork(LogicType logicType, const SimplifiedResult& leftValue, const SimplifiedResult& rightValue)
{
    ARIES_ASSERT( ( CHECK_VARIANT_TYPE( leftValue, bool ) || CHECK_VARIANT_TYPE( leftValue, aries_acc::AriesNull ) ),
                  "left value type: " + std::to_string(leftValue.which()) );
    ARIES_ASSERT( ( CHECK_VARIANT_TYPE( rightValue, bool ) || CHECK_VARIANT_TYPE( rightValue, aries_acc::AriesNull ) ),
                  "right value type: " + std::to_string(rightValue.which()) );
    SimplifiedResult result;
    if ( CHECK_VARIANT_TYPE( leftValue, aries_acc::AriesNull ) ) // left is null
    {
        if ( CHECK_VARIANT_TYPE( rightValue, aries_acc::AriesNull ) ) // null and null, null or null
        {
            result = AriesNull();
        }
        else
        {
            result = NullAndBoolLogic( logicType, boost::get< bool >( rightValue ) );
        }
    }
    else if ( CHECK_VARIANT_TYPE( rightValue, aries_acc::AriesNull ) ) // right is null
    {
        result = NullAndBoolLogic( logicType, boost::get< bool >( leftValue ) );
    }
    else
    {
        bool leftBoolValue = boost::get< bool >( leftValue );
        bool rightBoolValue = boost::get< bool >( rightValue );
        switch ( logicType )
        {
            case LogicType::AND:
                result = leftBoolValue && rightBoolValue;
                break;
            case LogicType::OR:
                result = leftBoolValue || rightBoolValue;
                break;
        }
    }
    return result;
}

/**
mysql server 5.7.26:
select null and true: null
select null and false: 0
select null or true: 1
select null or false: null
select null and null: null
select null or null: null

select null and 0.0; 0
select null and 0.1; null
select null or 0.1; 1
select null or 0.0; null

select null and "abc"; 0
select null and "1abc"; null
select null or "1abc"; 1
select null or "abc"; null
*/
SimplifiedResult ExpressionSimplifier::simplifyAndOr(CommonBiaodashi* origin, THD* thd, bool &success)
{
    assert(origin->GetChildrenCount() == 2);
    auto logicType = static_cast<LogicType>(boost::get<int>(origin->GetContent()));

    CommonBiaodashi *leftChild = GET_CHILD_BY_INDEX( origin, 0 );
    CommonBiaodashi *rightChild = GET_CHILD_BY_INDEX( origin, 1 );

    bool leftSuccess = false, rightSuccess = false;
    SimplifiedResult left  = doSimplify(leftChild, thd, leftSuccess);
    SimplifiedResult right = doSimplify(rightChild, thd, rightSuccess);

    SimplifiedResult result;
    success = true;
    SimplifiedResult leftBoolResult;
    SimplifiedResult rightBoolResult;
    if ( leftSuccess )
    {
        leftBoolResult = ToBooleanValue( left );
        if ( rightSuccess )
        {
            rightBoolResult = ToBooleanValue( right );
            result = DoLogicWork( logicType, leftBoolResult, rightBoolResult );
        }
        else
        {
            if ( !CHECK_VARIANT_TYPE( left, aries_acc::AriesNull ) ) // not null
            {
                bool bLeft = boost::get<bool>( leftBoolResult );
                if( LogicType::AND == logicType && !bLeft ) // false and any
                    result = false;
                else if ( LogicType::OR == logicType && bLeft ) // true or any
                    result = true;
                else // {true and any} | {false or any}: cann't be simplified
                    success = false;
            }
            else // null {and | or} column
            {
                // leftChild->SetIsNullable( true );
                // origin->SetIsNullable( true );
                if ( LogicType::AND == logicType ) // null and any -> false
                    result = AriesNull();
                else
                    success = false;
            }
        }
    }
    else if ( rightSuccess )
    {
        rightBoolResult = ToBooleanValue( right );
        if ( !CHECK_VARIANT_TYPE( right, aries_acc::AriesNull ) ) // not null
        {
            bool bRight = boost::get<bool> ( rightBoolResult );
            if ( LogicType::AND == logicType && !bRight )
                result = false;
            else if ( LogicType::OR == logicType && bRight )
                result = true;
            else
                success = false;
        }
        else
        {
            // rightChild->SetIsNullable( true );
            // origin->SetIsNullable( true );
            if ( LogicType::AND == logicType ) // any and null -> false
                result = AriesNull();
            else
                success = false;
        }
    }
    else
    {
        // origin->SetIsNullable( origin->IsNullable() || leftChild->IsNullable() || rightChild->IsNullable() );
        success = false;
    }

    return result;
}

#define DO_COMPARE_WORK(compareType, leftValue, rightValue) \
    switch (compareType)                                    \
    {                                                       \
    case ComparisonType::DengYu:                            \
        return leftValue == rightValue;                     \
    case ComparisonType::BuDengYu:                          \
    case ComparisonType::SQLBuDengYu:                       \
        return leftValue != rightValue;                     \
    case ComparisonType::XiaoYuDengYu:                      \
        return leftValue <= rightValue;                     \
    case ComparisonType::DaYuDengYu:                        \
        return leftValue >= rightValue;                     \
    case ComparisonType::XiaoYu:                            \
        return leftValue < rightValue;                      \
    case ComparisonType::DaYu:                              \
        return leftValue > rightValue;                      \
    default:                                                \
        assert(0);                                          \
    }

SimplifiedResult
ExpressionSimplifier::SimplifyDictEncodedColumnComparison( CommonBiaodashi* origin, THD* thd, bool& success )
{
    success = false;

    auto compType = static_cast<ComparisonType>(boost::get<int>(origin->GetContent()));

    // 对于字典压缩的列，如果参与等于或者非等于比较，转换成列的字典索引与字符串常量对应的字典索引的比较
    auto leftChild = ( CommonBiaodashi * )origin->GetChildByIndex( 0 ).get();
    auto rightChild = ( CommonBiaodashi * )origin->GetChildByIndex( 1 ).get();
    if ( (   ( BiaodashiType::Lie == leftChild->GetType() && BiaodashiType::Zifuchuan == rightChild->GetType() )
          || ( BiaodashiType::Zifuchuan == leftChild->GetType() && BiaodashiType::Lie == rightChild->GetType() ) )
      && ( ComparisonType::DengYu == compType || ComparisonType::BuDengYu == compType ) )
    {
        ColumnShellPointer column;
        string strConst;
        if ( BiaodashiType::Lie == leftChild->GetType() )
        {
            column = boost::get<ColumnShellPointer>( leftChild->GetContent() );
            strConst = boost::get< string >( rightChild->GetContent() );
        }
        else
        {
            column = boost::get<ColumnShellPointer>( rightChild->GetContent() );
            strConst = boost::get< string >( leftChild->GetContent() );
        }

        if ( column->GetColumnStructure() &&
             aries::EncodeType::DICT == column->GetColumnStructure()->GetEncodeType() )
        {
            vector< int32_t > columnIds;
            auto columnId = column->GetLocationInTable() + 1;
            columnIds.push_back( columnId );
            auto table = column->GetTable();
            auto tableData = aries_engine::AriesMvccTableManager::GetInstance().getTable( thd->m_tx, table->GetDb(), table->GetID(), columnIds );

            AriesDataBufferSPtr dictBuff;
            AriesDictEncodedColumnSPtr dictColumn;
            if ( tableData->IsColumnUnMaterilized( 1 ) )
            {
                auto columnRef = tableData->GetUnMaterilizedColumn( 1 );
                dictColumn = std::dynamic_pointer_cast< AriesDictEncodedColumn >( columnRef->GetReferredColumn() );
            }
            else
            {
                dictColumn = tableData->GetDictEncodedColumn( 1 );
            }

            dictBuff = dictColumn->GetDictDataBuffer();
            auto nullable = dictBuff->isNullableColumn();
            if ( nullable )
                return false;

            auto dictItemCount = dictBuff->GetItemCount();
            index_t pos = -1;
            for ( size_t i = 0; i < dictItemCount; ++i )
            {
                string dictStr;
                if ( nullable )
                {
                    if ( !dictBuff->isStringDataNull( i ) )
                    {
                        dictStr = dictBuff->GetNullableString( i );
                    }
                }
                else
                {
                    dictStr = dictBuff->GetString( i );
                }
                if ( dictStr == strConst )
                {
                    pos = i;
                    break;
                }
            }

            if( pos != -1 )
            {
                return false;
            }
            else
            {
                success = true;
                bool bValue = false;
                switch( compType )
                {
                    case ComparisonType::DengYu:
                    {
                        bValue = false;
                        break;
                    }
                    case ComparisonType::BuDengYu:
                    {
                        bValue = true;
                        break;
                    }
                    default:
                        assert( 0 );
                        break;
                }
                return bValue;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

SimplifiedResult ExpressionSimplifier::simplifyCompare(CommonBiaodashi* origin, THD* thd, bool& success)
{
    assert(origin->GetChildrenCount() == 2);
    CommonBiaodashi *leftChild = GET_CHILD_BY_INDEX( origin, 0 );
    CommonBiaodashi *rightChild = GET_CHILD_BY_INDEX( origin, 1 );

    bool leftSuccess = false, rightSuccess = false;
    SimplifiedResult left = doSimplify(leftChild, thd, leftSuccess);
    SimplifiedResult right = doSimplify(rightChild, thd, rightSuccess);
    if ( ( leftSuccess && CHECK_VARIANT_TYPE( left, aries_acc::AriesNull ) ) || ( rightSuccess &&  CHECK_VARIANT_TYPE( right, aries_acc::AriesNull ) ) ) // null
    {
        success = true;
        return AriesNull();
    }

    success = leftSuccess && rightSuccess;
    if ( !success )
    {
        if ( !m_simplifyOnlyConst )
        {
            auto dictCompareResult = SimplifyDictEncodedColumnComparison( origin, thd, success );
            if ( success )
                return dictCompareResult;
            else
                return false;
        }
        else
            return false;
    }

    auto compareType = static_cast<ComparisonType>(boost::get<int>(origin->GetContent()));

    aries_acc::Decimal leftNumberValue = 0;
    aries_acc::Decimal rightNumberValue = 0;
    aries_acc::AriesDatetime leftDatetimeValue;
    aries_acc::AriesDatetime rightDatetimeValue;
    switch ( left.which() ) {
        case 0: // int
            leftNumberValue += boost::get<int>(left);
            break;
        case 1: // double
            leftNumberValue += boost::get<double>(left);
            break;
        case 2: // decimal
            leftNumberValue += boost::get<aries_acc::Decimal>(left);
            break;
        case 9: // bool
            leftNumberValue += boost::get<bool>(left);
            break;
        case 3: // AriesDate
            leftDatetimeValue = boost::get<aries_acc::AriesDate>(left);
            break;
        case 4: // AriesDatetime
            leftDatetimeValue = boost::get<aries_acc::AriesDatetime>(left);
            break;
        case 5: // AriesTimestamp
            leftDatetimeValue = boost::get<aries_acc::AriesTimestamp>(left);
            break;
        case 11: // int64_t
            leftNumberValue += boost::get<int64_t>(left);
            break;
    }
    switch ( right.which() ) {
        case 0: // int
            rightNumberValue += boost::get<int>(right);
            break;
        case 1: // double
            rightNumberValue += boost::get<double>(right);
            break;
        case 2: // decimal
            rightNumberValue += boost::get<aries_acc::Decimal>(right);
            break;
        case 8: // bool
            rightNumberValue += boost::get<bool>(right);
            break;
        case 3: // AriesDate
            rightDatetimeValue = boost::get<aries_acc::AriesDate>(right);
            break;
        case 4: // AriesDatetime
            rightDatetimeValue = boost::get<aries_acc::AriesDatetime>(right);
            break;
        case 5: // AriesTimestamp
            rightDatetimeValue = boost::get<aries_acc::AriesTimestamp>(right);
            break;
        case 11: // int64_t
            rightDatetimeValue += boost::get<int64_t>(right);
            break;
    }

    switch ( left.which() ) {
        case 0:
        case 1:
        case 2:
        case 8:
        case 11:
            switch ( right.which() )
            {
                case 0:
                case 1:
                case 2:
                case 8:
                case 11:
                {
                    DO_COMPARE_WORK( compareType, leftNumberValue, rightNumberValue );
                    break;
                }
                case 6: // string
                {
                    auto stringValue = boost::get< std::string >( right );
                    try {
                        auto value = aries_acc::Decimal( stringValue.data() );
                        if ( value.GetError() == ERR_OK )
                        {
                            rightNumberValue = value;
                            DO_COMPARE_WORK( compareType, leftNumberValue, rightNumberValue );
                        }
                        else
                        {
                            return false;
                        }
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                }
                default:
                    success = false;
                    break;
            }
            break;
        case 3: // AriesDate
        case 4: // AriesDatetime
            switch ( right.which() )
            {
                case 3:
                case 4:
                {
                    DO_COMPARE_WORK( compareType, leftDatetimeValue, rightDatetimeValue );
                    break;
                }
                default:
                    success = false;
                    break;
            }
            break;
        case 6: // string
            switch ( right.which() )
            {
                case 0:
                case 1:
                case 2:
                case 8:
                case 11:
                {
                    auto stringValue = boost::get< std::string >( left );
                    try {
                        auto value = aries_acc::Decimal( stringValue.data() );
                        if ( value.GetError() == ERR_OK )
                        {
                            leftNumberValue = value;
                            DO_COMPARE_WORK( compareType, leftNumberValue, rightNumberValue );
                        }
                        else
                        {
                            return false;
                        }
                    }
                    catch ( ... )
                    {
                        return false;
                    }
                }
                case 6:
                {
                    DO_COMPARE_WORK( compareType,
                                     boost::get<string>(left),
                                     boost::get<string>(right) );
                    break;
                }
                default:
                    success = false;
                    break;
            }
            break;
        case 7: // AriesTime
            switch ( right.which() )
            {
                case 7:
                {
                    DO_COMPARE_WORK( compareType,
                                     boost::get<aries_acc::AriesTime>(left),
                                     boost::get<aries_acc::AriesTime>(right) );
                    break;
                }
                default:
                    success = false;
                    break;
            }
            break;
        default:
            success = false;
            break;
    }

    return false;
}

#define DO_CALC_WORK(calc_type, leftValue, rightValue)                                                                 \
    switch (calc_type)                                                                                                 \
    {                                                                                                                  \
    case CalcType::ADD:                                                                                                \
        return leftValue + rightValue;                                                                                 \
    case CalcType::SUB:                                                                                                \
        return leftValue - rightValue;                                                                                 \
    case CalcType::MUL:                                                                                                \
        return leftValue * rightValue;                                                                                 \
    case CalcType::DIV:                                                                                                \
        return leftValue / rightValue;                                                                                 \
    case CalcType::MOD:                                                                                                \
        return leftValue % rightValue;                                                                                 \
    default:                                                                                                           \
        ARIES_ASSERT( 0, "should not be here" );                                                                                                     \
    }

#define DO_CALC_WORK2(calc_type, leftValue, rightValue)                                                                \
    switch (calc_type)                                                                                                 \
    {                                                                                                                  \
    case CalcType::ADD:                                                                                                \
        return leftValue + rightValue;                                                                                 \
    case CalcType::SUB:                                                                                                \
        return leftValue - rightValue;                                                                                 \
    case CalcType::MUL:                                                                                                \
        return leftValue * rightValue;                                                                                 \
    case CalcType::DIV:                                                                                                \
        return leftValue / rightValue;                                                                                 \
    default:                                                                                                           \
        assert(0);                                                                                                     \
    }

SimplifiedResult ExpressionSimplifier::simplifyCalc(CommonBiaodashi *origin, THD* thd, bool &success)
{
    //包含常量 int 与 int 的优化 double 与 int double decimal 的运算 
    assert(origin->GetChildrenCount() == 2);
    CommonBiaodashi *leftChild = GET_CHILD_BY_INDEX( origin, 0 );
    CommonBiaodashi *rightChild = GET_CHILD_BY_INDEX( origin, 1 );

    bool leftSuccess = false, rightSuccess = false;
    SimplifiedResult left = doSimplify(leftChild, thd, leftSuccess);
    SimplifiedResult right = doSimplify(rightChild, thd, rightSuccess);
    if ( ( leftSuccess &&  CHECK_VARIANT_TYPE( left, aries_acc::AriesNull )  ) || ( rightSuccess &&  CHECK_VARIANT_TYPE( right, aries_acc::AriesNull )  ) ) // null
    {
        success = true;
        return AriesNull();
    }
    else if ( rightSuccess && CHECK_VARIANT_TYPE( right, int ) && boost::get< int >( right ) == 0 )
    {
        success = true;
        return AriesNull();
    }

    success = leftSuccess && rightSuccess;
    if ( !success )
        return false;

    auto calc_type = static_cast<CalcType>(boost::get<int>(origin->GetContent()));

    if (CHECK_VARIANT_TYPE(left, int) && CHECK_VARIANT_TYPE(right, int))
    {
        // auto leftValue = boost::get<int>(left);
        // auto leftDecimalValue = aries_acc::Decimal(leftValue);
        // auto rightValue = boost::get<int>(right);

        // if (calc_type == CalcType::DIV) {
        //     DO_CALC_WORK(calc_type, leftDecimalValue, rightValue);
        // } else {
        //     DO_CALC_WORK(calc_type, leftValue, rightValue);
        // }

        // 此处将其放入的后期计算过程到 AlignOptimize 再进行优化
        // TODO 后期除法检测到左右为 int 类型时 需要将 左侧的 int 转换为 decimal 类型
        success = false;
        return false;
    }
    else if (CHECK_VARIANT_TYPE(left, double) || CHECK_VARIANT_TYPE(right, double))
    {
        double leftValue = 0;
        double rightValue = 0;

        switch (left.which())
        {
        case 0:
        {
            leftValue = boost::get<int>(left);
            break;
        }
        case 1:
        {
            leftValue = boost::get<double>(left);
            break;
        }
        case 2:
        {
            auto value = boost::get<aries_acc::Decimal>(left);
            leftValue = value + 0.0; // decimal to double
            break;
        }
        }

        switch (right.which())
        {
        case 0:
        {
            rightValue = boost::get<int>(right);
            break;
        }
        case 1:
        {
            rightValue = boost::get<double>(right);
            break;
        }
        case 2:
        {
            auto value = boost::get<aries_acc::Decimal>(right);
            rightValue = value + 0.0; // decimal to double
            break;
        }
        }

        if (CalcType::MOD != calc_type) {
            DO_CALC_WORK2(calc_type, leftValue, rightValue);
        } else {
            return ::fmod(leftValue, rightValue);
        }
    }
    else
    {
        // 此处只需要将 left 和 right 的属性转为 decimal 即可 计算过程到 AlignOptimize 再进行优化

        leftChild->SetType(BiaodashiType::Decimal);
        rightChild->SetType(BiaodashiType::Decimal);

        switch (left.which())
        {
        case 0:
        {
            string str = to_string( boost::get<int>(left));
            leftChild->SetContent( str);
            break;
        }
        case 1:
        {
            ARIES_ASSERT( 0, "should not be here" );
            break;
        }
        case 2:
        {
            // 保持原本的值
            // leftValue = boost::get<aries_acc::Decimal>(left);
            break;
        }
        case 8:
        {
            string str = to_string( boost::get<bool>(left));
            leftChild->SetContent( str);
            break;
        }
        case 11:
        {
            string str = to_string( boost::get<int64_t>(left));
            leftChild->SetContent( str);
            break;
        }
        }

        switch (right.which())
        {
        case 0:
        {
            string str = to_string( boost::get<int>(right));
            rightChild->SetContent( str);
            break;
        }
        case 1:
        {
            ARIES_ASSERT( 0, "should not be here" );
            break;
        }
        case 2:
        {
            // rightValue = boost::get<aries_acc::Decimal>(right);
            break;
        }
        case 8:
        {
            string str = to_string( boost::get<bool>(right));
            rightChild->SetContent( str);
            break;
        }
        case 11:
        {
            string str = to_string( boost::get<int64_t>(right));
            rightChild->SetContent( str);
            break;
        }
        }

        success = false;
        return false;
        // DO_CALC_WORK(calc_type, leftValue, rightValue);
    }
    ARIES_ASSERT( 0, "should not be here" );
}

SimplifiedResult ExpressionSimplifier::handleUnixTimestamp(CommonBiaodashi *origin, THD* thd, bool &success)
{

    if (origin->GetChildrenCount() == 1)
    {
        auto date_time = aries_acc::AriesDatetimeTrans::GetInstance().Now();
        auto timezone_offset = GET_CHILD_BY_INDEX( origin, 0 );
        success = true;
        return date_time.getUnixTimestamp(boost::get<int>(timezone_offset->GetContent()));
    }

    auto param = GET_CHILD_BY_INDEX( origin, 0 );
    auto param_simplified = doSimplify(param, thd, success);

    if (!success)
    {
        return false;
    }

    auto timezone_offset = GET_CHILD_BY_INDEX( origin, 1 );

    int time_zone_offset = boost::get<int>(timezone_offset->GetContent());

    if (IS_STRING_VARIANT(param_simplified))
    {
        /*
        mysql> select unix_timestamp("abc");
        +-----------------------+
        | unix_timestamp("abc") |
        +-----------------------+
        |              0.000000 |
        +-----------------------+
        1 row in set, 1 warning (0.00 sec)

        mysql> show warnings;
        +---------+------+---------------------------------+
        | Level   | Code | Message                         |
        +---------+------+---------------------------------+
        | Warning | 1292 | Incorrect datetime value: 'abc' |
        +---------+------+---------------------------------+
        */
        auto string_content = boost::get<std::string>(param_simplified);
        MYSQL_TIME my_time = {0};
        int was_cut = 0;
        auto time_type =
            aries_acc::str_to_datetime(string_content.c_str(), string_content.length(), &my_time, 0, &was_cut);
        switch (time_type)
        {
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE:
            return aries_acc::UNIX_TIMESTAMP(aries_acc::AriesDate(my_time.year, my_time.month, my_time.day),
                                             time_zone_offset);
        case enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME:
            return aries_acc::UNIX_TIMESTAMP(aries_acc::AriesDatetime(my_time.year, my_time.month, my_time.day,
                                                                      my_time.hour, my_time.minute, my_time.second, 0),
                                             time_zone_offset);
        default:
        {
            auto msg = format_err_msg( ER( ER_WRONG_VALUE),
                                       "datetime",
                                       string_content.data() );
            ARIES_EXCEPTION_SIMPLE( ER_TRUNCATED_WRONG_VALUE, msg.data() );
            return false;
            break;
        }

        }
    }
    else if (CHECK_VARIANT_TYPE(param_simplified, aries_acc::AriesDate))
    {
        return aries_acc::UNIX_TIMESTAMP(boost::get<aries_acc::AriesDate>(param_simplified), time_zone_offset);
    }
    else if (CHECK_VARIANT_TYPE(param_simplified, aries_acc::AriesDatetime))
    {
        return aries_acc::UNIX_TIMESTAMP(boost::get<aries_acc::AriesDatetime>(param_simplified), time_zone_offset);
    }
    else
    {
        string msg( "parameter type '" );
        msg.append( SIMPLIFIED_RESULT_TYPE_NAMES[ param_simplified.which() ] ).append( "' for function UNIX_TIMESTAMP");
        ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, msg );
    }
}

SimplifiedResult  ExpressionSimplifier::handleCast(CommonBiaodashi* origin, THD* thd, bool &success) {
    auto value = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(0));

    auto ret = doSimplify(value.get(), thd, success);

    if (!success) {
        return false;
    }
    if ( CHECK_VARIANT_TYPE(ret, aries_acc::AriesNull) ) // null
    {
        return ret;
    }

    std::string string_value;
    if (IS_STRING_VARIANT(ret)) {
        string_value = boost::get<std::string>(ret);
    } else if (CHECK_VARIANT_TYPE(ret, int)) {
        string_value = std::to_string(boost::get<int>(ret));
    } else if (CHECK_VARIANT_TYPE(ret, int64_t)) {
        string_value = std::to_string(boost::get<int64_t>(ret));
    } else {
        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "cannot cast: " + value->GetName());
    }


    success = true;
    switch (origin->GetValueType()) {
        case BiaodashiValueType::TEXT: {
            auto actual_length = string_value.size();
            if (actual_length > ( size_t )origin->GetLength()) {
                return std::string(string_value.data(), origin->GetLength());
            } else {
                origin->SetLength(actual_length);
                return string_value;
            }
        }
        case BiaodashiValueType::INT: {
            try {
                return std::stoi(string_value);
            } catch (const std::exception& e) {
                return 0;
            }
        }
        case BiaodashiValueType::LONG_INT: {
            try {
                return int64_t(std::stoll(string_value));
            } catch (const std::exception& e) {
                return int64_t(0);
            }
        }
        case BiaodashiValueType::DECIMAL: {
            if (origin->GetLength() != 0) {
                return aries_acc::Decimal(origin->GetLength(), origin->GetAssociatedLength(), string_value.c_str());
            }
            return aries_acc::Decimal(string_value.c_str());
        }
        case BiaodashiValueType::DOUBLE: {
            try {
                return std::stod(string_value);
            } catch (const std::exception& e) {
                return double(0);
            }
        }
        case BiaodashiValueType::FLOAT: {
            try {
                return std::stof(string_value);
            } catch (const std::exception& e) {
                return float(0);
            }
        }
        case BiaodashiValueType::DATE: {
            return aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate(string_value);
        }
        case BiaodashiValueType::DATE_TIME: {
            return aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime(string_value);
        }
        case BiaodashiValueType::TIME: {
            return aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( string_value );
        }

        default: break;
    }

    success = false;
    return false;
}


SimplifiedResult ExpressionSimplifier::handleDateFormat(CommonBiaodashi* origin, THD* thd, bool &success) {
    auto value = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(0));
    auto format = std::dynamic_pointer_cast<CommonBiaodashi>(origin->GetChildByIndex(1));

    auto ret = doSimplify(value.get(), thd, success);

    if (!success) {
        return false;
    }

    if ( CHECK_VARIANT_TYPE( ret, aries_acc::AriesNull ) )
    {
        return ret;
    }

    auto format_value = doSimplify(format.get(), thd, success);
    if (!success || !IS_STRING_VARIANT(format_value)) {
        LOG(ERROR) << "cannot handle 2nd parameter of date_format: " << format->ToString();
        return false;
    }

    auto format_string = boost::get<std::string>(format_value);

    auto result_len = aries_acc::get_format_length(format_string.c_str());

    if (IS_STRING_VARIANT(ret)) {
        try {
            ret = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime(boost::get<std::string>(ret));
        } catch (const std::exception &e) {
            try {
                ret = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( boost::get< std::string >( ret ));
            } catch (const std::exception &e) {
               ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
            }
        }
    }

    std::shared_ptr< char > buffer( new char[ result_len + 1 ] );
    memset( buffer.get(), 0, result_len + 1 );

    SimplifiedResult result = aries_acc::AriesNull();
    try {
        if (IS_ARIESDATE_VARIANT(ret)) {
            auto date = boost::get<aries_acc::AriesDate>(ret);
            if (aries_acc::DATE_FORMAT(buffer.get(), format_string.c_str(), date)) {
                result = std::string( buffer.get() );
            }
            else
            {
                ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
            }
        } else if (IS_ARIESDATETIME_VARIANT(ret)) {
            auto date_time = boost::get<aries_acc::AriesDatetime>(ret);
            if (aries_acc::DATE_FORMAT(buffer.get(), format_string.c_str(), date_time)) {
                result = std::string( buffer.get() );
            }
            else
            {
                ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
            }
        } else if (CHECK_VARIANT_TYPE(ret, aries_acc::AriesTime)) {
            //get today's date
            auto date = AriesDatetimeTrans::Now();
            date.hour = 0;
            date.minute = 0;
            date.second = 0;
            date.second_part = 0;
            auto time = boost::get< aries_acc::AriesTime >( ret );
            if (aries_acc::DATE_FORMAT(buffer.get(), format_string.c_str(), date + time)) {
                result = std::string( buffer.get() );
            }
            else
            {
                ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
            }
        } else if (CHECK_VARIANT_TYPE(ret, aries_acc::AriesTimestamp)) {
            auto timestamp = boost::get<aries_acc::AriesTimestamp>(ret);
            if (aries_acc::DATE_FORMAT(buffer.get(), format_string.c_str(), timestamp)) {
                result = std::string( buffer.get() );
            }
            else
            {
                ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
            }
        } else {
            ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
        }
    } catch (const std::exception &e) {
        ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "date_format" );
    }
    return result;
}

SimplifiedResult ExpressionSimplifier::handleCoalesce(CommonBiaodashi* origin, THD* thd, bool &success) {
    ARIES_ASSERT( origin->GetChildrenCount() > 0, "invalid coalesce expression" );
    for (size_t i = 0; i < origin->GetChildrenCount(); i++) {
        auto child = (CommonBiaodashi*)(origin->GetChildByIndex(i).get());

        auto child_result = doSimplify(child, thd, success);

        if (!success) {
            return false;
        }

        if (!CHECK_VARIANT_TYPE(child_result, aries_acc::AriesNull) || i == (origin->GetChildrenCount() - 1)) {
            convertTo(child_result, origin->GetValueType(), success);
            if (success) {
                return child_result;
            } else {
                return false;
            }
        }
    }
    ARIES_ASSERT( 0, "unexpected error" );
}


SimplifiedResult ExpressionSimplifier::handleSubstring(CommonBiaodashi* origin, THD* thd, bool &success) {
    auto value = GET_CHILD_BY_INDEX( origin, 0 );

    auto start = GET_CHILD_BY_INDEX( origin, 1 );
    auto length = GET_CHILD_BY_INDEX( origin, 2 );

    auto has_false = false;
    auto start_simplified = doSimplify( start, thd, success );
    if ( !success )
    {
        has_false = true;
    }
    else if ( CHECK_VARIANT_TYPE( start_simplified, aries_acc::AriesNull ) )
    {
        success = true;
        return aries_acc::AriesNull();
    }

    auto length_simplified = doSimplify( length, thd, success );
    if ( !success )
    {
        has_false = true;
    }
    else if ( CHECK_VARIANT_TYPE( length_simplified, aries_acc::AriesNull ) )
    {
        success = true;
        return aries_acc::AriesNull();
    }

    auto value_simplified = doSimplify( value, thd, success );
    if ( !success )
    {
        has_false = true;
    }
    else if ( CHECK_VARIANT_TYPE( value_simplified, aries_acc::AriesNull ) )
    {
        return aries_acc::AriesNull();
    }

    if ( has_false )
    {
        success = false;
        return false;
    }

    int start_value = 0;
    success = false;
    if (CHECK_VARIANT_TYPE(start_simplified, int)) {
        start_value = boost::get<int>(start_simplified);
    } else if(IS_STRING_VARIANT(start_simplified)) {
        start_value = std::atoi(boost::get<std::string>(start_simplified).c_str());
    } else {
        ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "substring's start type: " + SIMPLIFIED_RESULT_TYPE_NAMES[ start_simplified.which() ] );
    }

    if (!value->IsNullable() && start_value == 0) {
        success = true;
        return std::string();
    }

    success = false;
    std::string string_value = simplifiedResultToString( value_simplified );

    if (start_value == 0) {
        success = true;
        return std::string();
    }

    int length_value = 0;
    if (CHECK_VARIANT_TYPE(length_simplified, int)) {
        length_value = boost::get<int>(length_simplified);
    } else if(IS_STRING_VARIANT(length_simplified)) {
        length_value = std::stoi(boost::get<std::string>(length_simplified));
    } else {
        ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "substring's length type: " + SIMPLIFIED_RESULT_TYPE_NAMES[ start_simplified.which() ] );
        return false;
    }

    const auto buf = (unsigned char*)(string_value.c_str());
    int total_character_length = 0;
    for (size_t i = 0; i < string_value.size(); ) {
        auto c = buf[i];

        total_character_length += 1;
        if (c < 0x80) {
            i += 1;
        } else if (c < 0xe0) {
            i += 2;
        } else/* if (c < 0xf0)*/ {
            i += 3;
        }
    }

    if (start_value < 0) {
        start_value = total_character_length + start_value + 1;
    }

    if (length_value == -1) {
        length_value = total_character_length;
    }

    int actual_length = 0;
    start_value -= 1;
    int actual_start = -1;
    int cursor = 0;
    for (size_t i = 0; i < string_value.size() && length_value > 0;) {
        auto c = buf[i];
        int char_len = 0;
        if (c < 0x80) {
            char_len = 1;
        } else if (c < 0xe0) {
            char_len = 2;
        } else/* if (c < 0xf0)*/ {
            char_len = 3;
        }

        if (cursor == start_value) {
            actual_start = i;
        }

        if (actual_start != -1) {
            actual_length += char_len;
            length_value --;
        }

        i += char_len;
        cursor ++;
    }

    success = true;
    if ( actual_length == 0 )
    {
        return std::string();
    }
    return std::string((char *)(buf + actual_start), actual_length);
}

SimplifiedResult ExpressionSimplifier::handleConcat(CommonBiaodashi* origin, THD* thd, bool &success) {
    std::vector<SimplifiedResult> args;
    auto has_failed = false;
    for (size_t i = 0; i < origin->GetChildrenCount(); i++) {
        auto arg = (CommonBiaodashi*)(origin->GetChildByIndex(i).get());

        if (arg->GetType() == BiaodashiType::Null) {
            success = true;
            return aries_acc::AriesNull();
        }

        auto arg_simplified = doSimplify(arg, thd, success);
        if (!success) {
            has_failed = true;
        } else {
            convertTo(arg_simplified, BiaodashiValueType::TEXT, success);
            if (success) {
                args.emplace_back(arg_simplified);
            } else {
                has_failed = true;
            }
        }
    }

    if (has_failed) {
        success = false;
        return false;
    }

    std::string result;
    for (auto const& arg : args) {
        result += boost::get<std::string>(arg);
    }

    success = true;
    return result;
}

SimplifiedResult ExpressionSimplifier::handleCase(CommonBiaodashi* origin, THD* thd, bool &success) {
    auto condition_child = GET_CHILD_BY_INDEX( origin, 0 );
    SimplifiedResult condition = false;
    if (condition_child != nullptr) {
        condition = doSimplify(GET_CHILD_BY_INDEX( origin, 0 ), thd, success);
        if (!success) {
            return false;
        }
    }

    for (size_t i = 1; i < origin->GetChildrenCount() - 1; i += 2) {
        auto value = doSimplify((CommonBiaodashi*)(origin->GetChildByIndex(i).get()), thd, success);
        if (!success) {
            return false;
        }

        if ((condition_child != nullptr && isEqual(condition, value)) || conditionValue(value, success)) {
            auto result = doSimplify((CommonBiaodashi*)(origin->GetChildByIndex(i + 1).get()), thd, success);
            if (!success) {
                return false;
            }

            convertTo(result, origin->GetValueType(), success);
            if (!success) {
                return false;
            }
            return result;
        }
    }

    auto else_child = (CommonBiaodashi*)(origin->GetChildByIndex(origin->GetChildrenCount() - 1).get());
    if (else_child) {
        auto else_value = doSimplify(else_child, thd, success);
        if (!success) {
            return false;
        }

        convertTo(else_value, origin->GetValueType(), success);
        if (!success) {
            return false;
        }
        return else_value;
    }
    else
    {
        return aries_acc::AriesNull();
    }

    return false;
}

SimplifiedResult ExpressionSimplifier::handleIfCondition(CommonBiaodashi* origin, THD* thd, bool &success) {
    auto condition = doSimplify(GET_CHILD_BY_INDEX( origin, 0 ), thd, success);
    if (!success) {
        return false;
    }

    auto fulfilled = conditionValue(condition, success);

    auto value = doSimplify((CommonBiaodashi*)(origin->GetChildByIndex(fulfilled ? 1 : 2).get()), thd, success);
    if (!success) {
        return false;
    }

    convertTo(value, origin->GetValueType(), success);
    if (!success) {
        LOG(ERROR) << "Cannot convert [" << origin->ToString() << "] to " << static_cast<int>(origin->GetValueType());
        return false;
    }
    return value;
}

SimplifiedResult ExpressionSimplifier::handleTruncate( CommonBiaodashi *origin, THD* thd, bool &success ) {
    auto data = GET_CHILD_BY_INDEX( origin, 0 );
    auto precision_expr = GET_CHILD_BY_INDEX( origin, 1 );
    bool is_data_simplified = false;
    auto data_simplified = doSimplify( data, thd, is_data_simplified );
    bool is_precision_simplified = false;
    auto precision_simplified = doSimplify( precision_expr, thd, is_precision_simplified );

    if ( ( is_data_simplified && CHECK_VARIANT_TYPE( data_simplified, aries_acc::AriesNull ) )
        || ( is_precision_simplified && CHECK_VARIANT_TYPE( precision_simplified, aries_acc::AriesNull ) ) )
    {
        success = true;
        return aries_acc::AriesNull();
    }

    if ( !is_precision_simplified )
    {
        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "precision in truncate should be constant." );
    }

    success = is_precision_simplified && is_data_simplified;

    if ( !success ) {
        return false;
    }

    int precision = 0;
    if ( IS_STRING_VARIANT( precision_simplified ) )
    {
        auto string_value = boost::get< std::string >( precision_simplified );
        try
        {
            precision = std::stoi( string_value );
        }
        catch ( ... )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_STORED_FCT, "truncate" );
        }
    }
    else if ( CHECK_VARIANT_TYPE( precision_simplified, int ) )
    {
        precision = boost::get< int >( precision_simplified );
    }
    else if ( CHECK_VARIANT_TYPE( precision_simplified, int64_t ) )
    {
        auto value = boost::get< int64_t >( precision_simplified );
        if ( value > INT32_MAX )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_STORED_FCT, "truncate" );
        }
        precision = static_cast< int >( value );
    }
    else
    {
        ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_STORED_FCT, "truncate" );
    }


    if ( CHECK_VARIANT_TYPE( data_simplified, int ) )
    {
        return truncate( boost::get< int >( data_simplified ), precision );
    }
    else if ( CHECK_VARIANT_TYPE( data_simplified, int64_t ) )
    {
        return truncate( boost::get< int64_t >( data_simplified ), precision );
    }
    else if ( CHECK_VARIANT_TYPE( data_simplified, aries_acc::Decimal ) )
    {
        return truncate( boost::get< aries_acc::Decimal >( data_simplified ), precision );
    }
    else if ( CHECK_VARIANT_TYPE( data_simplified, std::string ) )
    {
        auto data_str = boost::get< std::string >( data_simplified );
        if ( data_str.find( '.' ) != std::string::npos )
        {
            auto d = aries_acc::Decimal( data_str.data() );
            if ( d.GetError() == ERR_OK )
            {
                return truncate( d, precision );
            }
        }
        try
        {
            return truncate( std::stoi( data_str ), precision );
        }
        catch ( const std::exception &e1 )
        {
            try
            {
                return truncate( int64_t( std::stoll( data_str ) ), precision );
            }
            catch ( const std::exception &e2 )
            {
                return int32_t( 0 );
            }
        }
    }
    else
    {
        string msg( "parameter type '" );
        msg.append( SIMPLIFIED_RESULT_TYPE_NAMES[ data_simplified.which() ] ).append( "' for function truncate");
        ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, msg );
    }
}

SimplifiedResult ExpressionSimplifier::handleMonth( CommonBiaodashi *origin, THD* thd, bool &success )
{
    auto child = GET_CHILD_BY_INDEX( origin, 0 );
    success = false;
    auto child_value = doSimplify( child, thd, success );
    if ( !success )
    {
        return false;
    }
    else if ( CHECK_VARIANT_TYPE( child_value, aries_acc::AriesNull ) )
    {
        return child_value;
    }

    auto timestamp = get_timestamp_value( child_value, success );
    if ( !success )
    {
        timestamp = convert_to_timestamp( child_value, success );
        if ( !success )
        {
            success = true;
            return aries_acc::AriesNull();
        }
    }

    return aries_acc::MONTH( aries_acc::AriesTimestamp{ timestamp } );
}

SimplifiedResult ExpressionSimplifier::handleExtract( CommonBiaodashi *origin, THD* thd, bool &success )
{
    auto interval_type_string = boost::get< std::string >( ( GET_CHILD_BY_INDEX( origin, 0 ) )->GetContent() );

    auto interval_type = get_interval_type( interval_type_string );

    ARIES_ASSERT( interval_type != INTERVAL_LAST, std::string( "invalid interval type: " + interval_type_string ).c_str() );

    auto value = doSimplify( GET_CHILD_BY_INDEX( origin, 1 ), thd, success );
    if ( !success )
    {
        return false;
    }

    if ( CHECK_VARIANT_TYPE( value, aries_acc::AriesNull ) )
    {
        return value;
    }

    auto timestamp = get_timestamp_value( value, success );
    if ( !success )
    {
        timestamp = convert_to_timestamp( value, success );
        if ( !success )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMETERS_TO_NATIVE_FCT, "extract" );
        }
    }

    aries_acc::AriesDatetime datetime( timestamp, 0 );

    return aries_acc::EXTRACT( interval_type, datetime );
}

static int64_t get_timestamp_value( SimplifiedResult value, bool& success )
{
    success = true;
    if ( CHECK_VARIANT_TYPE( value, aries_acc::AriesDate ) )
    {
        return boost::get< aries_acc::AriesDate >( value ).toTimestamp();
    }
    else if ( CHECK_VARIANT_TYPE( value, aries_acc::AriesDatetime ) )
    {
        return boost::get< aries_acc::AriesDatetime >( value ).toTimestamp();
    }
    else if ( CHECK_VARIANT_TYPE( value, aries_acc::AriesTimestamp ) )
    {
        return boost::get< aries_acc::AriesTimestamp >( value ).getTimeStamp();
    }

    success = false;
    return 0;
}

static int64_t convert_to_timestamp( SimplifiedResult value, bool& success )
{
    auto convert_string_to_timestamp = [ & ]( const std::string& str )
    {
        try
        {
            auto datetime = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( str );
            return datetime.toTimestamp();
        }
        catch ( ... )
        {
            success = false;
            return int64_t( 0 );
        }
    };

    success = true;
    if ( IS_STRING_VARIANT( value ) )
    {
        return convert_string_to_timestamp( boost::get< std::string >( value ) );
    }
    else if ( CHECK_VARIANT_TYPE( value, aries_acc::AriesTime ) )
    {
        return aries_acc::AriesDate().toTimestamp() + boost::get< aries_acc::AriesTime >( value ).toMicroseconds();
    }
    else if ( CHECK_VARIANT_TYPE( value, int64_t ) || CHECK_VARIANT_TYPE( value, int ) )
    {
        auto intValue = CHECK_VARIANT_TYPE( value, int64_t ) ? boost::get< int64_t >( value ) : int64_t( boost::get< int >( value ) );
        return convert_string_to_timestamp( std::to_string( intValue ) );
    }
    success = false;
    return 0;
}

static std::string get_string_from_simplify_result( const SimplifiedResult& result )
{
    if ( CHECK_VARIANT_TYPE( result, int ) )
    {
        return std::to_string( boost::get< int >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, double ) )
    {
        return std::to_string( boost::get< double >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::Decimal ) )
    {
        auto value = boost::get< aries_acc::Decimal >( result );
        char temp[ 64 ] = { 0 };
        value.GetDecimal( temp );
        return std::string( temp );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesDate ) )
    {
        auto value = boost::get< aries_acc::AriesDate >( result );
        return aries_acc::AriesDatetimeTrans::GetInstance().ToString( value );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesDatetime ) )
    {
        auto value = boost::get< aries_acc::AriesDatetime >( result );
        return aries_acc::AriesDatetimeTrans::GetInstance().ToString( value );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesTimestamp ) )
    {
        auto value = boost::get< aries_acc::AriesTimestamp >( result );
        return aries_acc::AriesDatetimeTrans::GetInstance().ToString( value );
    }
    else if ( IS_STRING_VARIANT( result ) )
    {
        return boost::get< std::string >( result );
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesTime ) )
    {
        auto value = boost::get< aries_acc::AriesTime >( result );
        return aries_acc::AriesDatetimeTrans::GetInstance().ToString( value );
    }
    else if ( CHECK_VARIANT_TYPE( result, bool ) )
    {
        auto value = boost::get< bool >( result );
        return value ? "true" : "false";
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesNull ) )
    {
        ARIES_ASSERT( 0, "should handle null before here" );
        return "";
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesYear ) )
    {
        auto value = boost::get< aries_acc::AriesYear >( result );
        return aries_acc::AriesDatetimeTrans::GetInstance().ToString( value );
    }
    else if ( CHECK_VARIANT_TYPE( result, int64_t ) )
    {
        return std::to_string( boost::get< int64_t >( result ) );
    }
    else if ( CHECK_VARIANT_TYPE( result, float ) )
    {
        return std::to_string( boost::get< float >( result ) );
    }
    else
    {
        string msg( "unexpected value type: ");
        msg.append( std::to_string( result.which() ) );
        ARIES_ASSERT( 0, msg );
    }
}

/**!
 * @brief 比较两个 SimplifiedResult
 * @param[in] right
 * @retval  true: 如果 left 大于 right
 * @retval  false: 如果 left 不大于 right
 * 数据类型转换参考 mysql 文档: [12.3 Type Conversion in Expression Evaluation](https://dev.mysql.com/doc/refman/8.0/en/type-conversion.html)
 */
static bool compare( const SimplifiedResult& left, const SimplifiedResult& right )
{
    // If one or both arguments are NULL, the result of the comparison is NULL
    if ( CHECK_VARIANT_TYPE( left, aries_acc::AriesNull ) || CHECK_VARIANT_TYPE( right, aries_acc::AriesNull ) )
    {
        ARIES_ASSERT( 0, "cannot compare with null value" );
    }

    // If both arguments in a comparison operation are strings, they are compared as strings.
    if ( IS_STRING_VARIANT( left ) && IS_STRING_VARIANT( right ) )
    {
        return boost::get< std::string >( left ) > boost::get< std::string >( right );
    }

    // If both arguments are integers, they are compared as integers.
    if ( ( CHECK_VARIANT_TYPE( left, int ) || CHECK_VARIANT_TYPE( left, int64_t ) )
        && ( CHECK_VARIANT_TYPE( right, int ) || CHECK_VARIANT_TYPE( right, int64_t ) )
    )
    {
        auto leftIntValue = CHECK_VARIANT_TYPE( left, int ) ? int64_t( boost::get< int >( left ) ) : boost::get< int64_t >( left );
        auto rightIntValue = CHECK_VARIANT_TYPE( right, int ) ? int64_t( boost::get< int >( right ) ) : boost::get< int64_t >( right );
        return leftIntValue > rightIntValue;
    }

    bool leftIsTime = false;
    bool rightIsTime = false;

    int64_t leftTimestamp = get_timestamp_value( left, leftIsTime );
    int64_t rightTimestamp = get_timestamp_value( right, rightIsTime );

    // If one of the arguments is a TIMESTAMP or DATETIME column and the other argument is a constant,
    // the constant is converted to a timestamp before the comparison is performed.
    if ( leftIsTime && rightIsTime )
    {
        return leftTimestamp > rightTimestamp;
    }
    else if ( leftIsTime && !rightIsTime )
    {
        rightTimestamp = convert_to_timestamp( right, rightIsTime );
        if ( rightIsTime )
        {
            return leftTimestamp > rightTimestamp;
        }
        ARIES_EXCEPTION( ER_WRONG_VALUE, "timestamp", get_string_from_simplify_result( right ).data() );
    }
    else if ( !leftIsTime && rightIsTime )
    {
        leftTimestamp = convert_to_timestamp( left, leftIsTime );
        if ( leftIsTime )
        {
            return leftTimestamp > rightTimestamp;
        }
        ARIES_EXCEPTION( ER_WRONG_VALUE, "timestamp", get_string_from_simplify_result( left ).data() );
    }

    // If one of the arguments is a decimal value, comparison depends on the other argument.
    // The arguments are compared as decimal values if the other argument is a decimal or integer value,
    // or as floating-point values if the other argument is a floating-point value.
    if ( CHECK_VARIANT_TYPE( left, aries_acc::Decimal ) && CHECK_VARIANT_TYPE( right, aries_acc::Decimal ) )
    {
        return boost::get< aries_acc::Decimal >( left ) > boost::get< aries_acc::Decimal >( right );
    }
    if ( CHECK_VARIANT_TYPE( left, aries_acc::Decimal ) )
    {
        auto leftDecimal = boost::get< aries_acc::Decimal >( left );
        if ( CHECK_VARIANT_TYPE( right, int ) )
        {
            return leftDecimal > boost::get< int >( right );
        }
        else if ( CHECK_VARIANT_TYPE( right, int64_t ) )
        {
            return leftDecimal > boost::get< int64_t >( right );
        }
    }
    else if ( CHECK_VARIANT_TYPE( right, aries_acc::Decimal ) )
    {
        auto rightDecimal = boost::get< aries_acc::Decimal >( right );
        if ( CHECK_VARIANT_TYPE( left, int ) )
        {
            return boost::get< int >( left ) > rightDecimal;
        }
        else if ( CHECK_VARIANT_TYPE( left, int64_t ) )
        {
            return boost::get< int64_t >( left ) > rightDecimal;
        }
    }

    // In all other cases, the arguments are compared as floating-point (real) numbers.
    // For example, a comparison of string and numeric operands takes place as a comparison of floating-point numbers.
    double leftValue = 0;
    double rightValue = 0;

    std::string errmsg( "comparison between " );
    errmsg += get_result_type_name( left );
    errmsg += " and " + get_result_type_name( right );

    if ( IS_STRING_VARIANT( left ) )
    {
        try
        {
            auto tmp = boost::get< std::string >( left );
            leftValue = std::atof( tmp.c_str() );
            leftValue = std::atof( boost::get< std::string >( left ).c_str() );
        }
        catch ( ... )
        {
            ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, errmsg.c_str() );
        }
    }
    else if ( CHECK_VARIANT_TYPE( left, int ) || CHECK_VARIANT_TYPE( left, int64_t ) )
    {
        leftValue = CHECK_VARIANT_TYPE( left, int ) ? double( boost::get< int >( left ) ) : double( boost::get< int64_t >( left ) );
    }
    else if ( CHECK_VARIANT_TYPE( left, float ) || CHECK_VARIANT_TYPE( left, double ) )
    {
        leftValue = CHECK_VARIANT_TYPE( left, float ) ? double( boost::get< float >( left ) ) : boost::get< double >( left );
    }
    else if ( CHECK_VARIANT_TYPE( left, aries_acc::Decimal ) )
    {
        auto decimal = boost::get< aries_acc::Decimal >( left );
        leftValue = decimal.GetDouble();
    }
    else
    {
        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, errmsg.c_str() );
    }

    if ( IS_STRING_VARIANT( right ) )
    {
        try
        {
            rightValue = std::atof( boost::get< std::string >( right ).c_str() );
        }
        catch ( ... )
        {
            ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, errmsg.c_str() );
        }
    }
    else if ( CHECK_VARIANT_TYPE( right, int ) || CHECK_VARIANT_TYPE( right, int64_t ) )
    {
        rightValue = CHECK_VARIANT_TYPE( right, int ) ? double( boost::get< int >( right ) ) : double( boost::get< int64_t >( right ) );
    }
    else if ( CHECK_VARIANT_TYPE( right, float ) || CHECK_VARIANT_TYPE( right, double ) )
    {
        rightValue = CHECK_VARIANT_TYPE( right, float ) ? double( boost::get< float >( right ) ) : boost::get< double >( right );
    }
    else if ( CHECK_VARIANT_TYPE( right, aries_acc::Decimal ) )
    {
        auto decimal = boost::get< aries_acc::Decimal >( right );
        rightValue = decimal.GetDouble();
    }
    else
    {
        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, errmsg.c_str() );
    }

    return leftValue > rightValue;
}

SimplifiedResult ExpressionSimplifier::handleBetween( CommonBiaodashi *origin, THD* thd, bool &success )
{
    success = false;
    auto target = doSimplify( GET_CHILD_BY_INDEX( origin, 0 ), thd, success );
    if ( !success )
    {
        return false;
    }

    if ( CHECK_VARIANT_TYPE( target, aries_acc::AriesNull ) )
    {
        return aries_acc::AriesNull();
    }

    auto lower = doSimplify( GET_CHILD_BY_INDEX( origin, 1 ), thd, success );
    if ( !success )
    {
        return false;
    }

    if ( CHECK_VARIANT_TYPE( lower, aries_acc::AriesNull ) )
    {
        return aries_acc::AriesNull();
    }

    auto upper = doSimplify( GET_CHILD_BY_INDEX( origin, 2 ), thd, success );
    if ( !success )
    {
        return false;
    }

    if ( CHECK_VARIANT_TYPE( upper, aries_acc::AriesNull ) )
    {
        return aries_acc::AriesNull();
    }

    return !compare( target, upper ) && !compare( lower, target );

}

SimplifiedResult ExpressionSimplifier::handleQiufan(CommonBiaodashi *origin, THD* thd, bool &success)
{
    CommonBiaodashi *child = GET_CHILD_BY_INDEX( origin, 0 );
    auto result = doSimplify(child, thd, success);
    if ( !success )
    {
        return false;
    }

    result = ToBooleanValue( result );

    if(CHECK_VARIANT_TYPE(result, bool))
    {
        return boost::get<bool>(result) == false;
    }
    else if(CHECK_VARIANT_TYPE(result, AriesNull))
    {
        return result;
    }
    else
    {
        string msg( "parameter type error for NOT expression" );
        ThrowNotSupportedException( msg );
        return false;
    }
}

SimplifiedResult ExpressionSimplifier::handleIn( CommonBiaodashi *origin, THD* thd, bool &success )
{
    CommonBiaodashi *target = GET_CHILD_BY_INDEX( origin, 0 );
    auto result = doSimplify(target, thd, success);

    auto target_simplified = true;
    if ( !success )
    {
        target_simplified = false;
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesNull ) )
    {
        return false;
    }

    bool has_false = false;

    /**
     * 表示候选列表中除了 null 之外的成员数量，如果为 0 则该表示可以直接化简为 false
     */
    auto valid_count = 0;
    for ( size_t i = 1; i < origin->GetChildrenCount(); i++ )
    {
        auto ok = false;
        auto child = doSimplify( GET_CHILD_BY_INDEX( origin, i ), thd, ok );
        if ( !ok )
        {
            has_false = true;
            continue;
        }

        if ( CHECK_VARIANT_TYPE( child, aries_acc::AriesNull )  )
        {
            continue;
        }

        valid_count++;

        if ( target_simplified && isEqual( result, child ) )
        {
            success = true;
            return true;
        }
    }

    if ( valid_count == 0 && !has_false )
    {
        success = true;
        return false;
    }

    if ( target_simplified )
    {
        if ( has_false )
        {
            ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "constant value in in-expression as left operand" );
        }
        success = true;
        return false;
    }

    success = false;
    return false;
}

SimplifiedResult ExpressionSimplifier::handleNotIn( CommonBiaodashi *origin, THD* thd, bool &success )
{
    CommonBiaodashi *target = GET_CHILD_BY_INDEX( origin, 0 );
    auto result = doSimplify(target, thd, success);
    auto target_simplified = true;
    if ( !success )
    {
        target_simplified = false;
    }
    else if ( CHECK_VARIANT_TYPE( result, aries_acc::AriesNull ) )
    {
        return false;
    }

    bool has_false = false;
    for ( size_t i = 1; i < origin->GetChildrenCount(); i++ )
    {
        auto ok = false;
        auto child = doSimplify( GET_CHILD_BY_INDEX( origin, i ), thd, ok );
        if ( !ok )
        {
            has_false = true;
            continue;
        }

        if ( CHECK_VARIANT_TYPE( child, aries_acc::AriesNull ) )
        {
            success = true;
            return false;
        }

        if ( target_simplified && isEqual( result, child ) )
        {
            success = true;
            return false;
        }
    }

    if ( target_simplified )
    {
        if ( has_false )
        {
            ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "constant value in not-in-expression as left operand" );
        }
        success = true;
        return true;
    }

    success = false;
    return false;
}

SimplifiedResult ExpressionSimplifier::handleExists(CommonBiaodashi *origin, THD* thd, bool &success)
{
    assert( origin->GetChildrenCount() > 0 );
    bool result = false;
    success = false;
    if( origin->GetChildrenCount() == 1 )
    {
        auto child = GET_CHILD_BY_INDEX( origin, 0 );
        if( child->GetType() == BiaodashiType::Buffer )
        {
            if( child->GetBuffer() )
            {
                success = true;
                result = child->GetBuffer()->GetItemCount() > 0;
            }
        }
    }
    return result;
}

void ExpressionSimplifier::convertTo(SimplifiedResult& value, BiaodashiValueType type, bool &success) {
    success = false;

    if (CHECK_VARIANT_TYPE(value, int)) {
        auto int_value = boost::get<int>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
                success = true;
                break;
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "int value cannot be converted to tiny/small int or bool";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = true;
                value = aries_acc::Decimal(int_value);
                break;
            case BiaodashiValueType::FLOAT:
                success = true;
                value = float(int_value);
                break;
            case BiaodashiValueType::DOUBLE:
                success = true;
                value = double(int_value);
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = std::to_string(int_value);
                break;
            case BiaodashiValueType::DATE:
            case BiaodashiValueType::DATE_TIME:
            case BiaodashiValueType::TIME:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "int value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "int value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "int value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, aries_acc::Decimal)) {
        auto decimal_value = boost::get<aries_acc::Decimal>(value);
        switch (type) {
            case BiaodashiValueType::TEXT: {
                char buffer[64] = {0};
                decimal_value.GetDecimal(buffer);
                value = std::string(buffer);
                success = true;
                break;
            }
            case BiaodashiValueType::DECIMAL: {
                value = decimal_value;
                success = true;
                break;
            }
            default:
            {
                string msg( "decimal value cannot be converted to " );
                msg.append( get_name_of_value_type( type ) );
                ARIES_ASSERT( 0, msg );
                break;
            }
        }
    } else if (CHECK_VARIANT_TYPE(value, double)) {
        auto double_value = boost::get<double>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "double value cannot be converted to integer";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = false;
                LOG(ERROR) << "double value cannot be converted to decimal";
                break;
            case BiaodashiValueType::FLOAT:
                success = false;
                LOG(ERROR) << "double value cannot be converted to float";
                break;
            case BiaodashiValueType::DOUBLE:
                success = true;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = std::to_string(double_value);
                break;
            case BiaodashiValueType::DATE:
            case BiaodashiValueType::DATE_TIME:
            case BiaodashiValueType::TIME:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "int value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "int value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "int value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, aries_acc::AriesDate)) {
        auto date_value = boost::get<aries_acc::AriesDate>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "AriesDate value cannot be converted to integer";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = false;
                LOG(ERROR) << "AriesDate value cannot be converted to decimal";
                break;
            case BiaodashiValueType::FLOAT:
                success = false;
                LOG(ERROR) << "AriesDate value cannot be converted to float";
                break;
            case BiaodashiValueType::DOUBLE:
                LOG(ERROR) << "AriesDate value cannot be converted to double";
                success = false;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = aries_acc::AriesDatetimeTrans::GetInstance().ToString(date_value);
                break;
            case BiaodashiValueType::DATE:
                success = true;
                break;
            case BiaodashiValueType::DATE_TIME:
                success = true;
                value = aries_acc::AriesDatetime(date_value);
                break;
            case BiaodashiValueType::TIME:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "AriesDate value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "AriesDate value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "AriesDate value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, aries_acc::AriesDatetime)) {
        auto datetime_value = boost::get<aries_acc::AriesDatetime>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "AriesDatetime value cannot be converted to integer";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = false;
                LOG(ERROR) << "AriesDatetime value cannot be converted to decimal";
                break;
            case BiaodashiValueType::FLOAT:
                success = false;
                LOG(ERROR) << "AriesDatetime value cannot be converted to float";
                break;
            case BiaodashiValueType::DOUBLE:
                LOG(ERROR) << "AriesDatetime value cannot be converted to double";
                success = false;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = aries_acc::AriesDatetimeTrans::GetInstance().ToString(datetime_value);
                break;
            case BiaodashiValueType::DATE:
                success = false;
                LOG(ERROR) << "AriesDatetime value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::DATE_TIME:
                success = true;
                break;
            case BiaodashiValueType::TIME:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "AriesDatetime value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "AriesDatetime value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "AriesDatetime value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, aries_acc::AriesTimestamp)) {
        auto timestamp_value = boost::get<aries_acc::AriesTimestamp>(value);
        auto datetime_value = aries_acc::AriesDatetime(timestamp_value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to integer";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = false;
                LOG(ERROR) << "AriesTimestamp value cannot be converted to decimal";
                break;
            case BiaodashiValueType::FLOAT:
                success = false;
                LOG(ERROR) << "AriesTimestamp value cannot be converted to float";
                break;
            case BiaodashiValueType::DOUBLE:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to double";
                success = false;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = aries_acc::AriesDatetimeTrans::GetInstance().ToString(datetime_value);
                break;
            case BiaodashiValueType::DATE:
                success = false;
                LOG(ERROR) << "AriesTimestamp value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::DATE_TIME:
                success = true;
                value = datetime_value;
                break;
            case BiaodashiValueType::TIMESTAMP:
                success = true;
                break;
            case BiaodashiValueType::TIME:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (IS_STRING_VARIANT(value)) {
        auto string_value = boost::get<std::string>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "string value cannot be converted to integer";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = false;
                LOG(ERROR) << "string value cannot be converted to decimal";
                break;
            case BiaodashiValueType::FLOAT:
                success = false;
                LOG(ERROR) << "string value cannot be converted to float";
                break;
            case BiaodashiValueType::DOUBLE:
                LOG(ERROR) << "string value cannot be converted to double";
                success = false;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                break;
            case BiaodashiValueType::DATE:
            case BiaodashiValueType::DATE_TIME:
            case BiaodashiValueType::TIME:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "AriesTimestamp value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "AriesTimestamp value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, aries_acc::AriesTime)) {
        auto time_value = boost::get<aries_acc::AriesTime>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::BOOL:
                LOG(ERROR) << "AriesTime value cannot be converted to integer";
                success = false;
                break;
            case BiaodashiValueType::DECIMAL:
                success = false;
                LOG(ERROR) << "AriesTime value cannot be converted to decimal";
                break;
            case BiaodashiValueType::FLOAT:
                success = false;
                LOG(ERROR) << "AriesTime value cannot be converted to float";
                break;
            case BiaodashiValueType::DOUBLE:
                LOG(ERROR) << "AriesTime value cannot be converted to double";
                success = false;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = aries_acc::AriesDatetimeTrans::GetInstance().ToString(time_value);
                break;
            case BiaodashiValueType::DATE_TIME:
                success = true;
                value = aries_acc::AriesDatetimeTrans::GetInstance().DatetimeOfToday() + time_value;
                break;
            case BiaodashiValueType::TIME:
                success = true;
                break;
            case BiaodashiValueType::DATE:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "AriesTime value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                LOG(ERROR) << "AriesTime value cannot be converted to year";
                success = false;
                break;
            default:
                LOG(ERROR) << "AriesTime value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, bool)) {
        auto bool_value = boost::get<bool>(value);
        switch (type) {
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
                success = true;
                value = (int) bool_value;
                break;
            case BiaodashiValueType::BOOL:
                success = true;
                break;
            case BiaodashiValueType::DECIMAL:
                success = true;
                value = aries_acc::Decimal((int)bool_value);
                break;
            case BiaodashiValueType::FLOAT:
            case BiaodashiValueType::DOUBLE:
                success = true;
                value = (double)bool_value;
                break;
            case BiaodashiValueType::TEXT:
                success = true;
                value = std::to_string(bool_value);
                break;
            case BiaodashiValueType::DATE_TIME:
            case BiaodashiValueType::TIME:
            case BiaodashiValueType::DATE:
            case BiaodashiValueType::TIMESTAMP:
                success = false;
                LOG(ERROR) << "bool value cannot be converted to date/time type";
                break;
            case BiaodashiValueType::YEAR:
                success = true;
                value = int(bool_value);
                break;
            default:
                LOG(ERROR) << "bool value cannot be converted to " << static_cast<int>(type);
                success = false;
                break;

        }
    } else if (CHECK_VARIANT_TYPE(value, aries_acc::AriesNull)) {
        success = true;
    } else {
        LOG(ERROR) << "cannot handle value type: " << value.which() << " convert to type";
    }
}

bool ExpressionSimplifier::ConvertExpression(AriesCommonExprUPtr& expr, const AriesColumnType& type) {
    auto origin_type = expr->GetValueType();
    if (origin_type.DataType.ValueType == type.DataType.ValueType) {
        return true;
    }


    if (expr->GetType() == AriesExprType::NULL_VALUE) {
        expr->SetContent(0);

        origin_type.DataType.ValueType = type.DataType.ValueType;
        origin_type.DataType.Length = type.DataType.Length;
        expr->SetValueType(origin_type);
        return true;
    }

    auto actual_type = type;

    switch (expr->GetType()) {
        case AriesExprType::STRING:
        case AriesExprType::INTEGER:
        case AriesExprType::DECIMAL:
        case AriesExprType::FLOATING:
        case AriesExprType::TRUE_FALSE:
        case AriesExprType::DATE:
        case AriesExprType::DATE_TIME:
        case AriesExprType::TIME:
        case AriesExprType::TIMESTAMP:
        case AriesExprType::YEAR:
            break;
        default: return true;
    }

    AriesExprType target_type;
    AriesExprContent content;

    switch (expr->GetValueType().DataType.ValueType) {
        case AriesValueType::BOOL: {
            bool value = boost::get<bool>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::BOOL: {
                    content = value;
                    target_type = AriesExprType::TRUE_FALSE;
                    break;
                }

                case AriesValueType::CHAR: {
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                }

                case AriesValueType::DECIMAL: {
                    content = aries_acc::Decimal(value);
                    target_type = AriesExprType::DECIMAL;
                    break;
                }

                case AriesValueType::DOUBLE: {
                    content = double(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                }

                case AriesValueType::FLOAT: {
                    content = float(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                }

                case AriesValueType::INT8: {
                    content = int8_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::INT16: {
                    content = int16_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::INT32: {
                    content = int32_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::INT64: {
                    content = int64_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::UINT8: {
                    content = uint8_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::UINT16: {
                    content = uint16_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::UINT32: {
                    content = uint32_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                case AriesValueType::UINT64: {
                    content = uint64_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }
                default: return false;
            }
            break;
        }

        /**
         * 字符串不向其他类型转换
         * 一般由其他类型转换成字符串
         */
        case AriesValueType::CHAR: {
            auto value = boost::get<std::string>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    content = value;
                    target_type = AriesExprType::STRING;
                    break;
                }
                case AriesValueType::BOOL: {
                    auto lower = to_lower(value);
                    content = !value.empty() && lower == "true";
                    target_type = AriesExprType::TRUE_FALSE;
                    break;
                }

                case AriesValueType::DECIMAL: {
                    content = aries_acc::Decimal(value.c_str());
                    target_type = AriesExprType::DECIMAL;
                    break;
                }

                case AriesValueType::DOUBLE: {
                    content = std::stod(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                }

                case AriesValueType::FLOAT: {
                    content = std::stof(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                }

                case AriesValueType::INT8:
                case AriesValueType::INT16:
                case AriesValueType::INT32:
                case AriesValueType::INT64: {
                    auto int_value = (int64_t)(std::stoll(value));
                    target_type = AriesExprType::INTEGER;

                    if (type.DataType.ValueType == AriesValueType::INT64) {
                        content = int_value;
                    } else if (type.DataType.ValueType == AriesValueType::INT32) {
                        if (int_value < INT32_MAX && int_value > INT32_MIN) {
                            content = int32_t(int_value);
                        } else {
                            content = int_value;
                            actual_type.DataType.ValueType = AriesValueType::INT64;
                        }
                    } else if (type.DataType.ValueType == AriesValueType::INT16) {
                        if (int_value < INT16_MAX && int_value > INT16_MIN) {
                            content = int16_t(int_value);
                        } else {
                            content = int_value;
                            actual_type.DataType.ValueType = AriesValueType::INT64;
                        }
                    } else if (type.DataType.ValueType == AriesValueType::INT8) {
                        if (int_value < INT16_MAX && int_value > INT16_MIN) {
                            content = int8_t(int_value);
                        } else {
                            content = int_value;
                            actual_type.DataType.ValueType = AriesValueType::INT64;
                        }
                    }
                    break;
                }

                case AriesValueType::UINT8:
                case AriesValueType::UINT16:
                case AriesValueType::UINT32:
                case AriesValueType::UINT64: {
                    content = (uint64_t)std::stoull(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }
                default: {
                    LOG(ERROR) << "unsupport to converty string to other types";
                    // ARIES_EXCEPTION_SIMPLE(ER_WRONG_TYPE_COLUMN_VALUE_ERROR, "unsupport to converty string to other types");
                    return false;
                }
            }
            break;
        }

        case AriesValueType::DATE: {

            auto value = boost::get<aries_acc::AriesDate>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    content = aries_acc::AriesDatetimeTrans::GetInstance().ToString(value);
                    target_type = AriesExprType::STRING;
                    break;
                }

                case AriesValueType::DATETIME: {
                    content = aries_acc::AriesDatetime(value);
                    target_type = AriesExprType::DATE_TIME;
                    break;
                }

                default: return false;
            }
            break;
        }
        case AriesValueType::DATETIME: {
            auto value = boost::get<aries_acc::AriesDatetime>(expr->GetContent());
            switch (type.DataType.ValueType) {

                case AriesValueType::CHAR: {
                    content = aries_acc::AriesDatetimeTrans::GetInstance().ToString(value);
                    target_type = AriesExprType::STRING;
                    break;
                }

                case AriesValueType::DATETIME: {
                    target_type = AriesExprType::DATE_TIME;
                    content = value;
                    break;
                }

                default: return false;
            }
            break;
        }

        case AriesValueType::TIMESTAMP: {
            auto value = boost::get<aries_acc::AriesTimestamp>(expr->GetContent());
            auto datetime_value = aries_acc::AriesDatetime(value);
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    content = aries_acc::AriesDatetimeTrans::GetInstance().ToString(datetime_value);
                    target_type = AriesExprType::STRING;
                    break;
                }
                case AriesValueType::DATETIME: {
                    content = datetime_value;
                    target_type = AriesExprType::DATE_TIME;
                    break;
                }

                default: return false;
            }
            break;
        }

        case AriesValueType::TIME: {
            auto value = boost::get<aries_acc::AriesTime>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    content = aries_acc::AriesDatetimeTrans::GetInstance().ToString(value);
                    target_type = AriesExprType::STRING;
                    break;
                }
                case AriesValueType::DATETIME: {
                    content = aries_acc::AriesDatetimeTrans::GetInstance().Now() + value;
                    target_type = AriesExprType::DATE_TIME;
                    break;
                }

                default: return false;
            }
            break;
        }

        case AriesValueType::DECIMAL: {
            auto value = boost::get<aries_acc::Decimal>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    char str[64] = {0};
                    value.GetDecimal(str);
                    content = std::string(str);
                    target_type = AriesExprType::STRING;
                    break;
                }

                case AriesValueType::DOUBLE: {
                    content = double(0) + value;
                    target_type = AriesExprType::FLOATING;
                    break;
                }
                case AriesValueType::COMPACT_DECIMAL:
                {
                    content = value;
                    target_type = AriesExprType::DECIMAL;
                    actual_type.DataType.ValueType = AriesValueType::DECIMAL;
                    break;
                }
                case AriesValueType::INT16:
                {
                    content = int16_t(double(0) + value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }
                case AriesValueType::INT8:
                {
                    content = int8_t(double(0) + value);
                    target_type = AriesExprType::INTEGER;
                    break;
                }

                default: return false;
            }
            break;
        }

        case AriesValueType::DOUBLE: {
            auto value = boost::get<double>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                }

                default: return false;
            }
            break;
        }

        case AriesValueType::FLOAT: {
            auto value = boost::get<float>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::CHAR: {
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                }

                default: return false;
            }
            break;
        }

        case AriesValueType::INT8: {
            auto value = boost::get<int8_t>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::INT16:
                    content = int16_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::INT32:
                    content = int32_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::INT64:
                    content = int64_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::DECIMAL:
                    content = aries_acc::Decimal(value);
                    target_type = AriesExprType::DECIMAL;
                    break;
                case AriesValueType::DOUBLE:
                    content = double(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::FLOAT:
                    content = float(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::CHAR:
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                default: return false;
            }
            break;
        }
        case AriesValueType::INT16: {
            auto value = boost::get<int16>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::INT32:
                    content = int32_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::INT64:
                    content = int64_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::DECIMAL:
                    content = aries_acc::Decimal(value);
                    target_type = AriesExprType::DECIMAL;
                    break;
                case AriesValueType::DOUBLE:
                    content = double(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::FLOAT:
                    content = float(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::CHAR:
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                default: return false;
            }
            break;
        }
        case AriesValueType::INT32: {
            auto value = boost::get<int32_t>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::INT8:
                    content = int8_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::INT16:
                    content = int16_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::INT64:
                    content = int64_t(value);
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::DECIMAL:
                    content = aries_acc::Decimal(value);
                    target_type = AriesExprType::DECIMAL;
                    break;
                case AriesValueType::COMPACT_DECIMAL:
                    content = aries_acc::Decimal(std::to_string(value).c_str());
                    target_type = AriesExprType::DECIMAL;
                    // actual_type.DataType.ValueType = AriesValueType::DECIMAL;
                    break;
                case AriesValueType::DOUBLE:
                    content = double(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::CHAR:
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                default: return false;
            }
            break;
        }
        case AriesValueType::INT64: {
            auto value = boost::get<int64_t>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::DECIMAL:
                    content = aries_acc::Decimal(value);
                    target_type = AriesExprType::DECIMAL;
                    break;
                case AriesValueType::DOUBLE:
                    content = double(value);
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::CHAR:
                    content = std::to_string(value);
                    target_type = AriesExprType::STRING;
                    break;
                default: return false;
            }
            break;
        }

        case AriesValueType::UINT8:
        case AriesValueType::UINT16:
        case AriesValueType::UINT32:
        case AriesValueType::UINT64: {
            ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "unsigned value" );
        }

        case AriesValueType::YEAR: {
            auto value = boost::get<aries_acc::AriesYear>(expr->GetContent());
            switch (type.DataType.ValueType) {
                case AriesValueType::INT32:
                    value = int32_t(value.getYear());
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::INT64:
                    value = int32_t(value.getYear());
                    target_type = AriesExprType::INTEGER;
                    break;
                case AriesValueType::DECIMAL:
                    content = aries_acc::Decimal(value.getYear());
                    target_type = AriesExprType::DECIMAL;
                    break;
                case AriesValueType::DOUBLE:
                    content = double(value.getYear());
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::FLOAT:
                    content = float(value.getYear());
                    target_type = AriesExprType::FLOATING;
                    break;
                case AriesValueType::CHAR:
                    content = std::to_string(value.getYear());
                    target_type = AriesExprType::STRING;
                    break;
                default: return false;
            }
            break;
        }
        default:
        {
            string msg( "unexpected value type: " );
            msg.append( std::to_string( ( int )expr->GetValueType().DataType.ValueType ) );
            ARIES_ASSERT( 0, msg );
            break;
        }
    }

    expr->SetContent(content);
    if (actual_type.DataType.ValueType == AriesValueType::CHAR) {
        auto string_value = boost::get<std::string>(content);
        actual_type.DataType.Length = string_value.size();
    }

    expr->SetValueType(actual_type);
    expr->SetType(target_type);
    return true;
}

bool ExpressionSimplifier::isEqual(SimplifiedResult& left, SimplifiedResult& right) {

    if (CHECK_VARIANT_TYPE(left, int)) {
        auto left_value = boost::get<int>(left);
        if (CHECK_VARIANT_TYPE(right, int)) {
            auto right_value = boost::get<int>(right);
            return left_value == right_value;
        } else if (CHECK_VARIANT_TYPE(right, double)) {
            auto right_value = boost::get<double>(right);
            return right_value == (double)(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::Decimal)) {
            auto right_value = boost::get<aries_acc::Decimal>(right);
            return right_value == aries_acc::Decimal(left_value);
        } else if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == std::to_string(left_value);
        } else if (CHECK_VARIANT_TYPE(right, bool)) {
            auto right_value = boost::get<bool>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesYear)) {
            auto right_value = boost::get<aries_acc::AriesYear>(right);
            return right_value.getYear() == left_value;
        }
    } else if (CHECK_VARIANT_TYPE(left, double)) {
        auto left_value = boost::get<double>(left);
        if (CHECK_VARIANT_TYPE(right, int)) {
            auto right_value = boost::get<int>(right);
            return left_value == (double)right_value;
        } else if (CHECK_VARIANT_TYPE(right, double)) {
            auto right_value = boost::get<double>(right);
            return left_value == right_value;
        } else if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == std::to_string(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::Decimal)) {
            auto right_value = boost::get<aries_acc::Decimal>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesYear)) {
            auto right_value = boost::get<aries_acc::AriesYear>(right);
            return right_value.getYear() == left_value;
        }
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::Decimal)) {
        auto left_value = boost::get<aries_acc::Decimal>(left);
        if (CHECK_VARIANT_TYPE(right, int)) {
            auto right_value = boost::get<int>(right);
            return left_value == right_value;
        } else if (CHECK_VARIANT_TYPE(right, double)) {
            auto right_value = boost::get<double>(right);
            return left_value == right_value;
        } else if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            char left_str[64] = {0};
            left_value.GetDecimal(left_str);
            return right_value == std::string(left_str);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::Decimal)) {
            auto right_value = boost::get<aries_acc::Decimal>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesYear)) {
            auto right_value = boost::get<aries_acc::AriesYear>(right);
            return right_value.getYear() == left_value;
        }
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::AriesDate)) {
        auto left_value = boost::get<aries_acc::AriesDate>(left);
        if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == aries_acc::AriesDatetimeTrans::GetInstance().ToString(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDate)) {
            auto right_value = boost::get<aries_acc::AriesDate>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDatetime)) {
            auto right_value = boost::get<aries_acc::AriesDatetime>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTimestamp)) {
            auto right_value = boost::get<aries_acc::AriesTimestamp>(right);
            return aries_acc::AriesDatetime(right_value) == aries_acc::AriesDatetime(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTime)) {
            auto right_value = boost::get<aries_acc::AriesTime>(right);
            return (aries_acc::AriesDatetimeTrans::GetInstance().Now() + right_value) == aries_acc::AriesDatetime(left_value);
        }
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::AriesDatetime)) {
        auto left_value = boost::get<aries_acc::AriesDatetime>(left);
        if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == aries_acc::AriesDatetimeTrans::GetInstance().ToString(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDate)) {
            auto right_value = boost::get<aries_acc::AriesDate>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDatetime)) {
            auto right_value = boost::get<aries_acc::AriesDatetime>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTimestamp)) {
            auto right_value = boost::get<aries_acc::AriesTimestamp>(right);
            return aries_acc::AriesDatetime(right_value) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTime)) {
            auto right_value = boost::get<aries_acc::AriesTime>(right);
            return (aries_acc::AriesDatetimeTrans::GetInstance().Now() + right_value) == left_value;
        }
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::AriesTimestamp)) {
        auto left_value = boost::get<aries_acc::AriesTimestamp>(left);
        if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            aries_acc::AriesDatetime left_datetime_value(left_value);
            return right_value == aries_acc::AriesDatetimeTrans::GetInstance().ToString(left_datetime_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDate)) {
            auto right_value = boost::get<aries_acc::AriesDate>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDatetime)) {
            auto right_value = boost::get<aries_acc::AriesDatetime>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTimestamp)) {
            auto right_value = boost::get<aries_acc::AriesTimestamp>(right);
            return aries_acc::AriesDatetime(right_value) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTime)) {
            auto right_value = boost::get<aries_acc::AriesTime>(right);
            return (aries_acc::AriesDatetimeTrans::GetInstance().Now() + right_value) == aries_acc::AriesDatetime(left_value);
        }
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::AriesTime)) {
        auto left_value = boost::get<aries_acc::AriesTime>(left);
        if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == aries_acc::AriesDatetimeTrans::GetInstance().ToString(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDate)) {
            auto right_value = boost::get<aries_acc::AriesDate>(right);
            return aries_acc::AriesDatetime(right_value) == (aries_acc::AriesDatetimeTrans::GetInstance().Now() + left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDatetime)) {
            auto right_value = boost::get<aries_acc::AriesDatetime>(right);
            return right_value == (aries_acc::AriesDatetimeTrans::GetInstance().Now() + left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTimestamp)) {
            auto right_value = boost::get<aries_acc::AriesTimestamp>(right);
            return aries_acc::AriesDatetime(right_value) == (aries_acc::AriesDatetimeTrans::GetInstance().Now() + left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTime)) {
            auto right_value = boost::get<aries_acc::AriesTime>(right);
            return right_value == left_value;
        }
    } else if (CHECK_VARIANT_TYPE(left, std::string)) {
        auto left_value = boost::get<std::string>(left);
        if (CHECK_VARIANT_TYPE(right, int)) {
            auto right_value = boost::get<int>(right);
            return left_value == std::to_string(right_value);
        } else if (CHECK_VARIANT_TYPE(right, double)) {
            auto right_value = boost::get<double>(right);
            return std::to_string(right_value) ==left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::Decimal)) {
            auto right_value = boost::get<aries_acc::Decimal>(right);
            char right_str[64] = {0};
            return std::string(right_value.GetDecimal(right_str)) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, bool)) {
            auto right_value = boost::get<bool>(right);
            return std::to_string(right_value) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesYear)) {
            auto right_value = boost::get<aries_acc::AriesYear>(right);
            return std::to_string(right_value.getYear()) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDate)) {
            auto right_value = boost::get<aries_acc::AriesDate>(right);
            return aries_acc::AriesDatetimeTrans::GetInstance().ToString(right_value) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesDatetime)) {
            auto right_value = boost::get<aries_acc::AriesDatetime>(right);
            return aries_acc::AriesDatetimeTrans::GetInstance().ToString(right_value) == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTimestamp)) {
            auto right_value = boost::get<aries_acc::AriesTimestamp>(right);
            aries_acc::AriesDatetime right_datetime_value(right_value);
            return left_value == aries_acc::AriesDatetimeTrans::GetInstance().ToString(right_datetime_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesTime)) {
            auto right_value = boost::get<aries_acc::AriesTime>(right);
            return left_value == aries_acc::AriesDatetimeTrans::GetInstance().ToString(right_value);
        }
    } else if (CHECK_VARIANT_TYPE(left, bool)) {
        auto left_value = boost::get<bool>(left);
        if (CHECK_VARIANT_TYPE(right, int)) {
            auto right_value = boost::get<int>(right);
            return left_value == right_value;
        } else if (CHECK_VARIANT_TYPE(right, double)) {
            auto right_value = boost::get<double>(right);
            return right_value == (double)(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::Decimal)) {
            auto right_value = boost::get<aries_acc::Decimal>(right);
            return right_value == aries_acc::Decimal(left_value);
        } else if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == std::to_string(left_value);
        } else if (CHECK_VARIANT_TYPE(right, bool)) {
            auto right_value = boost::get<bool>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesYear)) {
            auto right_value = boost::get<aries_acc::AriesYear>(right);
            return right_value.getYear() == left_value;
        }
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::AriesNull)) {
        return CHECK_VARIANT_TYPE(right, aries_acc::AriesNull);
    } else if (CHECK_VARIANT_TYPE(left, aries_acc::AriesYear)) {
        auto left_value = boost::get<aries_acc::AriesYear>(left).getYear();
        if (CHECK_VARIANT_TYPE(right, int)) {
            auto right_value = boost::get<int>(right);
            return left_value == right_value;
        } else if (CHECK_VARIANT_TYPE(right, double)) {
            auto right_value = boost::get<double>(right);
            return right_value == (double)(left_value);
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::Decimal)) {
            auto right_value = boost::get<aries_acc::Decimal>(right);
            return right_value == aries_acc::Decimal(left_value);
        } else if (CHECK_VARIANT_TYPE(right, std::string)) {
            auto right_value = boost::get<std::string>(right);
            return right_value == std::to_string(left_value);
        } else if (CHECK_VARIANT_TYPE(right, bool)) {
            auto right_value = boost::get<bool>(right);
            return right_value == left_value;
        } else if (CHECK_VARIANT_TYPE(right, aries_acc::AriesYear)) {
            auto right_value = boost::get<aries_acc::AriesYear>(right);
            return right_value.getYear() == left_value;
        }
    }

    return false;
}

END_ARIES_ENGINE_NAMESPACE
