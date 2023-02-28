//
// Created by 胡胜刚 on 2019-07-23.
//

#include "regex"

#include "AriesExprBridge.h"
#include "datatypes/AriesDatetimeTrans.h"
#include "datatypes/timefunc.h"
#include "datatypes/AriesTimeCalc.hxx"
#include "AriesAssert.h"
#include <schema/SchemaManager.h>
#include "frontend/SQLExecutor.h"
#include "server/mysql/include/sql_class.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "utils/string_util.h"
#include <queue>

BEGIN_ARIES_ENGINE_NAMESPACE

namespace
{
    void ValidateValueType( const string& exprName, const AriesCommonExprUPtr& exprToCheck, const initializer_list<AriesValueType>& validTypes )
    {
        auto valueType = exprToCheck->GetValueType().DataType.ValueType;
        for( auto type : validTypes )
        {
            if( valueType == type )
                return;
        }
        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, ( GetDataTypeStringName( valueType ) + " type in " + exprName ).c_str() );
    }
}

// static void convert_to_date(const AriesCommonExprUPtr& expression) {
//     if (expression->GetType() == AriesExprType::DATE) {
//         return;
//     }
//
//     assert(expression->GetType() == AriesExprType::STRING);
//     expression->SetType(AriesExprType::DATE);
//     auto string_content = boost::get<std::string>(expression->GetContent());
//     auto date_content = AriesDatetimeTrans::GetInstance().ToAriesDate(string_content);
//     expression->SetContent(date_content);
//
//     AriesColumnType value_type {AriesDataType{AriesValueType::DATE, 1}, false, false};
//     expression->SetValueType(value_type);
// }

AriesExprBridge::AriesExprBridge()
{
    m_thd = current_thd;
    m_defaultDb = m_thd->db();
    m_tx = m_thd->m_tx;
}

AriesCommonExprUPtr
AriesExprBridge::BridgeDictEncodedColumnInOp( CommonBiaodashi* origin, AriesExprType op ) const
{
    AriesCommonExprUPtr result;

    AriesValueType _type = convertValueType(origin->GetValueType());
    aries::AriesDataType data_type{_type, 1};
    AriesColumnType value_type{data_type, origin->IsNullable(), false};

    auto leftChild = ( CommonBiaodashi * )origin->GetChildByIndex( 0 ).get();
    if ( BiaodashiType::Lie == leftChild->GetType()
      && AriesValueType::CHAR == convertValueType( leftChild->GetValueType() ) )
    {
        int count = origin->GetChildrenCount();
        CommonBiaodashi *child;
        vector< string > inValues;

        for (int i = 1; i < count; i ++ )
        {
            child = ( CommonBiaodashi * )origin->GetChildByIndex(i).get();
            string strContent;
            if ( BiaodashiType::Zhengshu == child->GetType() )
            {
                AriesValueType childValueType = convertValueType( child->GetValueType() );
                switch ( childValueType )
                {
                    case AriesValueType::INT8:
                    case AriesValueType::INT16:
                    case AriesValueType::INT32:
                    {
                        auto value = boost::get< int >( child->GetContent() );
                        strContent = std::to_string(value);
                        break;
                    }
                    case AriesValueType::INT64:
                    {
                        auto value = boost::get< int64_t >( child->GetContent() );
                        strContent = std::to_string(value);
                        break;
                    }
                    default:
                    {
                        ARIES_ASSERT( 0, "unexpected value type: " + std::to_string( (int)childValueType ) );
                        break;
                    }
                }
            }
            else if ( BiaodashiType::Zifuchuan == child->GetType() )
                strContent = boost::get<std::string>( child->GetContent() );
            else
                return nullptr;

            inValues.push_back( strContent );
        }

        ColumnShellPointer column = boost::get<ColumnShellPointer>( leftChild->GetContent() );
        if ( aries::EncodeType::DICT == column->GetColumnStructure()->GetEncodeType() )
        {
            result = AriesCommonExpr::Create( op, 0, value_type );
            AriesCommonExprUPtr leftExpr = Bridge( origin->GetChildByIndex( 0 ) );
            AriesValueType leftValueType = AriesValueType::INT8;
            switch ( column->GetColumnStructure()->GetEncodedIndexType() )
            {
            case ColumnType::TINY_INT:
                leftValueType = AriesValueType::INT8;
                break;
            case ColumnType::SMALL_INT:
                leftValueType = AriesValueType::INT16;
                break;
            case ColumnType::INT:
                leftValueType = AriesValueType::INT32;
                break;
            default:
                aries::ThrowNotSupportedException("dict encoding type: " + get_name_of_value_type( column->GetColumnStructure()->GetEncodedIndexType() ) );
                break;
            }
            aries::AriesDataType leftDataType{ leftValueType, 1 };
            AriesColumnType leftColumnType{ leftDataType, column->GetColumnStructure()->IsNullable(), false};
            leftExpr->SetValueType( leftColumnType );
            leftExpr->SetUseDictIndex( true );
            result->AddChild( std::move( leftExpr ) );

            vector< int32_t > columnIds;
            auto columnId = column->GetLocationInTable() + 1;
            columnIds.push_back( columnId );
            auto table = column->GetTable();
            auto tableData = aries_engine::AriesMvccTableManager::GetInstance().getTable( m_tx, table->GetDb(), table->GetID(), columnIds );

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
            auto dictItemCount = dictBuff->GetItemCount();

            AriesValueType rightValueType = AriesValueType::INT32;
            aries::AriesDataType rightDataType{ rightValueType, 1 };
            AriesColumnType rightColumnType{ rightDataType, false, false};

            for ( auto const& inValue : inValues )
            {
                index_t pos = -1;
                for ( size_t i = 0; i < dictItemCount; ++i )
                {
                    string dictStr;
                    if ( nullable )
                    {
                        dictStr = dictBuff->GetNullableString( i );
                    }
                    else
                    {
                        dictStr = dictBuff->GetString( i );
                    }
                    if ( dictStr == inValue )
                    {
                        pos = i;
                        break;
                    }
                }

                auto rightExpr = AriesCommonExpr::Create( AriesExprType::INTEGER, pos, rightColumnType );
                result->AddChild( std::move( rightExpr ) );
            }
        }
    }
    return result;
}

AriesCommonExprUPtr
AriesExprBridge::BridgeDictEncodedColumnComparison( CommonBiaodashi* origin ) const
{
    AriesCommonExprUPtr result;

    AriesValueType _type = convertValueType(origin->GetValueType());
    aries::AriesDataType data_type{_type, 1};
    AriesColumnType value_type{data_type, origin->IsNullable(), false};

    auto compType = getCompType(
        static_cast<ComparisonType>(boost::get<int>(origin->GetContent())));

    // 对于字典压缩的列，如果参与等于或者非等于比较，转换成列的字典索引与字符串常量对应的字典索引的比较
    auto leftChild = ( CommonBiaodashi * )origin->GetChildByIndex( 0 ).get();
    auto rightChild = ( CommonBiaodashi * )origin->GetChildByIndex( 1 ).get();
    if ( ( ( BiaodashiType::Lie == leftChild->GetType()
           && BiaodashiType::Zifuchuan == rightChild->GetType() )
      || ( BiaodashiType::Zifuchuan == leftChild->GetType()
           && BiaodashiType::Lie == rightChild->GetType() ) )
      && ( AriesComparisonOpType::EQ == compType || AriesComparisonOpType::NE == compType ) )
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

        if ( column->GetColumnStructure() && aries::EncodeType::DICT == column->GetColumnStructure()->GetEncodeType() )
        {
            auto content = static_cast<int>( compType );
            result = AriesCommonExpr::Create( AriesExprType::COMPARISON, content, value_type );
            AriesCommonExprUPtr leftExpr;
            if ( BiaodashiType::Lie == leftChild->GetType() )
                leftExpr = Bridge( origin->GetChildByIndex( 0 ) );
            else
                leftExpr = Bridge( origin->GetChildByIndex( 1 ) );

            AriesValueType leftValueType = AriesValueType::INT8;
            switch ( column->GetColumnStructure()->GetEncodedIndexType() )
            {
            case ColumnType::TINY_INT:
                leftValueType = AriesValueType::INT8;
                break;
            case ColumnType::SMALL_INT:
                leftValueType = AriesValueType::INT16;
                break;
            case ColumnType::INT:
                leftValueType = AriesValueType::INT32;
                break;
            default:
                aries::ThrowNotSupportedException("dict encoding type: " + get_name_of_value_type( column->GetColumnStructure()->GetEncodedIndexType() ) );
                break;
            }
            aries::AriesDataType leftDataType{ leftValueType, 1 };
            AriesColumnType leftColumnType{ leftDataType, column->GetColumnStructure()->IsNullable(), false};
            leftExpr->SetValueType( leftColumnType );
            leftExpr->SetUseDictIndex( true );
            result->AddChild( std::move( leftExpr ) );

            vector< int32_t > columnIds;
            auto columnId = column->GetLocationInTable() + 1;
            columnIds.push_back( columnId );
            auto table = column->GetTable();
            auto tableData = aries_engine::AriesMvccTableManager::GetInstance().getTable( m_tx, table->GetDb(), table->GetID(), columnIds );

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
            auto initialTable = AriesInitialTableManager::GetInstance().getTable( table->GetDb(), table->GetID() );
            auto columnDict = initialTable->GetColumnDict( columnId - 1 );
            auto dictItemCount = columnDict->getDictItemCount();

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

            if( pos != -1 || nullable )
            {
                AriesValueType rightValueType = AriesValueType::INT32;
                aries::AriesDataType rightDataType{ rightValueType, 1 };
                AriesColumnType rightColumnType{ rightDataType, false, false};

                auto rightExpr = AriesCommonExpr::Create( AriesExprType::INTEGER, pos, rightColumnType );
                result->AddChild( std::move( rightExpr ) );
            }
            else
            {
                bool bValue = false;
                switch( compType )
                {
                    case AriesComparisonOpType::EQ:
                    {
                        bValue = false;
                        break;
                    }
                    case AriesComparisonOpType::NE:
                    {
                        bValue = true;
                        break;
                    }
                    default:
                        assert( 0 );
                        break;
                }
                result = AriesCommonExpr::Create( AriesExprType::TRUE_FALSE, bValue, AriesColumnType( AriesDataType(AriesValueType::BOOL, 1 ), false, false ) );
            }
        }
    }
    return result;
}

AriesCommonExprUPtr
AriesExprBridge::Bridge(const BiaodashiPointer &expression) const
{
    auto origin = (CommonBiaodashi *)expression.get();

    AriesExprType type = AriesExprType::INTEGER;
    AriesValueType _type = convertValueType(origin->GetValueType());
    aries::AriesDataType data_type{_type, 1};
    AriesColumnType value_type{data_type, origin->IsNullable(), false};
    AriesExpressionContent content;

    ExpressionSimplifier exprSimplifier;

    switch (origin->GetType())
    {
    case BiaodashiType::Zhengshu:
        type = AriesExprType::INTEGER;
        if (origin->GetValueType() == BiaodashiValueType::LONG_INT)
        {
            content = boost::get<int64_t>(origin->GetContent());
        }
        else
        {
            content = boost::get<int>(origin->GetContent());
        }
        break;
    case BiaodashiType::Fudianshu:
        type = AriesExprType::FLOATING;
        content = boost::get<double>(origin->GetContent());
        break;
    case BiaodashiType::Zifuchuan:
    {
        type = AriesExprType::STRING;
        auto string_content = boost::get<std::string>(origin->GetContent());
        string_content = aries_utils::unescape_string( string_content );
        value_type.DataType.Length = string_content.size();
        content = string_content;
        break;
    }
    case BiaodashiType::Decimal: {
        type = AriesExprType::DECIMAL;
        auto string_content = boost::get<std::string>(origin->GetContent());
        const char* pStr = string_content.data();
        auto strLen = strlen( pStr );
        // 解决 subquery 返回值为 Decimal 字符串
        if( *(pStr+strLen-1) == ')'){
            aries_acc::Decimal decimal(string_content.c_str(), strLen, true);
            value_type.DataType.Precision = decimal.prec;
            value_type.DataType.Scale = decimal.frac;
            content = decimal;
        }
        else{
            CheckDecimalPrecision( string_content );
            aries_acc::Decimal decimal(string_content.c_str());
            value_type.DataType.Precision = decimal.prec;
            value_type.DataType.Scale = decimal.frac;
            content = decimal;
        }
        break;
    }
    case BiaodashiType::Lie:
    {
        type = AriesExprType::COLUMN_ID;
        auto column = boost::get<ColumnShellPointer>(origin->GetContent());
        auto alias_expr = std::dynamic_pointer_cast<CommonBiaodashi>(column->GetExpr4Alias());
        int origin_position = column->GetPositionInChildTables();
        if (alias_expr) {
            value_type.DataType.ValueType = convertValueType(alias_expr->GetValueType());
            value_type.DataType.Length = alias_expr->GetLength();
            if (alias_expr->GetType() == BiaodashiType::Lie) {
                column = boost::get<ColumnShellPointer>(alias_expr->GetContent());
            } else {
                origin_position = column->GetAliasExprIndex() + 1;
            }
        } else if (column->GetPlaceholderMark()) {
            value_type.DataType.Length = origin->GetLength();
            value_type.DataType.ValueType = convertValueType(column->GetValueType());
        } else {
            value_type.DataType.Length = column->GetLength();
        }

        if (value_type.DataType.ValueType != AriesValueType::CHAR &&
            value_type.DataType.ValueType != AriesValueType::LIST) {
            value_type.DataType.Length = 1;
        }

        if (column->GetPositionInChildTables() > 0) {
            origin_position = column->GetPositionInChildTables();
        }
        content = origin_position;
        if ( AriesValueType::COMPACT_DECIMAL == value_type.DataType.ValueType ||
             AriesValueType::DECIMAL == value_type.DataType.ValueType )
        {
            int precision = origin->GetLength();
            int scale = origin->GetAssociatedLength();
            if( precision != -1 && scale != -1 )
                value_type.DataType = AriesDataType( AriesValueType::COMPACT_DECIMAL, precision, scale );
        }

        break;
    }
    case BiaodashiType::Star:
        type = AriesExprType::STAR;
        content = boost::get<std::string>(origin->GetContent());
        break;
    case BiaodashiType::SQLFunc:
    {
        auto function = boost::get<SQLFunctionPointer>(origin->GetContent());

        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        value_type.DataType.Length = origin->GetLength();
        if (function->GetIsAggFunc())
        {
            type = AriesExprType::AGG_FUNCTION;
            content = static_cast<int>(getAggFuncType(function->GetName()));

            // count should not be nullable
            if (function->GetType() == AriesSqlFunctionType::COUNT) {
                value_type.HasNull = false;
            }
        }
        else
        {
            type = AriesExprType::SQL_FUNCTION;
            content = static_cast<int>(function->GetType());
            LOG(INFO) << "get function name:" << function->GetName() << std::endl;
            ///////////////// lichi need find the correct length for CHAR type ///////////////////////
            switch( function->GetType() )
            {
                case aries_acc::AriesSqlFunctionType::SUBSTRING:
                {
                    int len = ( ( CommonBiaodashi* )( origin->GetChildByIndex( 0 ).get() ) )->GetLength();
                    // auto exprType = ((CommonBiaodashi *)(origin->GetChildByIndex(0).get()))->GetType();
                    // ARIES_ASSERT(exprType == BiaodashiType::Lie, "substr('xxxx', 0) should be optimized as empty string by frontend");
                    bool bMultiByteChar = true;
                    switch (((CommonBiaodashi *)(origin->GetChildByIndex(0).get()))->GetValueType())
                    {
                    case ColumnType::TINY_INT:
                        len = TypeSizeInString<int8_t>::LEN;
                        bMultiByteChar = false;
                        break;
                    case ColumnType::SMALL_INT:
                        len = TypeSizeInString<int16_t>::LEN;
                        bMultiByteChar = false;
                        break;
                    case ColumnType::INT:
                        len = TypeSizeInString<int32_t>::LEN;
                        bMultiByteChar = false;
                        break;
                    case ColumnType::LONG_INT:
                        len = TypeSizeInString<int64_t>::LEN;
                        bMultiByteChar = false;
                        break;
                    default:
                        break;
                    }
                    int start = boost::get<int>(((CommonBiaodashi *)(origin->GetChildByIndex(1).get()))->GetContent());
                    int count = boost::get< int >( ( ( CommonBiaodashi* )( origin->GetChildByIndex( 2 ).get() ) )->GetContent() );
                    if (start < 0)
                    {
                        start = len + start;
                        if (start < 0)
                            start = 0;
                        if (count == -1)
                            count = len - start;
                        else
                            count = (start + count > len ? len - start : count);
                    }
                    else
                    {
                        if (count == -1)
                            count = len - start + 1;
                        else
                            count = (start + count - 1 > len ? len - start + 1 : count);
                    }
                    if( bMultiByteChar )
                    {
                        // reserve enough space for utf8 characters
                        count = std::min( count * 3, len );
                    }
                    value_type.DataType.Length = count;
                    break;
                }
                case aries_acc::AriesSqlFunctionType::DATE_FORMAT:
                {
                    ARIES_ASSERT( origin->GetChildrenCount() == 2, "DATE_FORMAT should have two parameters" );
                    auto second_param = ( ( CommonBiaodashi* )( origin->GetChildByIndex( 1 ).get() ) );
                    ARIES_ASSERT( second_param->GetType() == BiaodashiType::Zifuchuan, "DATE_FORMAT's 2nd parameter should be string" );
                    auto format_string = boost::get< std::string >( second_param->GetContent() );
                    value_type.DataType.Length = aries_acc::get_format_length( format_string.c_str() );
                    break;
                }
                case aries_acc::AriesSqlFunctionType::COALESCE:
                {
                    type = AriesExprType::COALESCE;
                    value_type.DataType.Length = origin->GetLength();
                    break;
                }
                case aries_acc::AriesSqlFunctionType::CAST:
                {
                    if( value_type.DataType.ValueType == aries::AriesValueType::DECIMAL )
                    {
                        value_type.DataType.Length = 1;
                        value_type.DataType.Precision = origin->GetLength();
                        value_type.DataType.Scale = origin->GetAssociatedLength();
                    }
                    break;
                }
                case aries_acc::AriesSqlFunctionType::DICT_INDEX:
                {
                    assert( ( ( CommonBiaodashi* )( origin->GetChildByIndex(0).get() ) )->GetType() == BiaodashiType::Lie );
                    auto funcExpr = AriesCommonExpr::Create( type, content, value_type );
                    AriesCommonExprUPtr childExpr = Bridge( origin->GetChildByIndex( 0 ) );
                    ColumnShellPointer csp = boost::get<ColumnShellPointer>( ( ( CommonBiaodashi* )( origin->GetChildByIndex(0).get() ) )->GetContent() );
                    assert( csp->GetColumnStructure()->GetEncodeType() == EncodeType::DICT );
                    AriesValueType childValueType = AriesValueType::INT8;
                    switch( csp->GetColumnStructure()->GetEncodedIndexType() )
                    {
                        case ColumnType::TINY_INT:
                            childValueType = AriesValueType::INT8;
                            break;
                        case ColumnType::SMALL_INT:
                            childValueType = AriesValueType::INT16;
                            break;
                        case ColumnType::INT:
                            childValueType = AriesValueType::INT32;
                            break;
                        default:
                            aries::ThrowNotSupportedException("dict encoding type: " + get_name_of_value_type( csp->GetColumnStructure()->GetEncodedIndexType() ) );
                            break;
                    }
                    aries::AriesDataType childDataType
                    { childValueType, 1 };
                    bool isNullable = childExpr->GetValueType().isNullable();
                    AriesColumnType childColumnType
                    { childDataType, isNullable, false };
                    childExpr->SetValueType( childColumnType );
                    childExpr->SetUseDictIndex( true );

                    funcExpr->AddChild( std::move( childExpr ) );
                    return funcExpr;
                }
                default:
                    break;
            }
        }
        break;
    }
    case BiaodashiType::Shuzu:
        type = AriesExprType::ARRAY;
        content = boost::get<int>(origin->GetContent());
        break;
    case BiaodashiType::Yunsuan:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        type = AriesExprType::CALC;
        content = static_cast<int>(
            getCalcType(static_cast<CalcType>(boost::get<int>(origin->GetContent()))));

        value_type.DataType.ValueType = convertValueType(origin->GetPromotedValueType());
        break;
    }
    case BiaodashiType::Qiufan:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            return result;
        }
        type = AriesExprType::NOT;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::Bijiao:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        else
        {
            result = BridgeDictEncodedColumnComparison( origin );
            if ( result )
                return result;

            type = AriesExprType::COMPARISON;
            auto compType = getCompType(
                static_cast<ComparisonType>(boost::get<int>(origin->GetContent())));
            content = static_cast<int>( compType );
        }
        break;
    }
    case BiaodashiType::Likeop:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            return result;
        }
        type = AriesExprType::LIKE;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::Inop:
    {
        auto result = BridgeDictEncodedColumnInOp( origin, AriesExprType::IN );
        if ( result )
            return result;

        result = exprSimplifier.Simplify( origin, m_thd );
        if ( result )
        {
            return result;
        }

        type = AriesExprType::IN;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::NotIn:
    {
        auto result = BridgeDictEncodedColumnInOp( origin, AriesExprType::NOT_IN );
        if ( result )
            return result;

        result = exprSimplifier.Simplify( origin, m_thd );
        if ( result )
        {
            return result;
        }

        type = AriesExprType::NOT_IN;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::Between:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if ( result )
        {
            return result;
        }
        type = AriesExprType::BETWEEN;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::Cunzai:
    {
        type = AriesExprType::EXISTS;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::Case:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        type = AriesExprType::CASE;
        content = boost::get<int>(origin->GetContent());
        // assume frontend has trasformed CASE expr WHEN expr THEN expr ->
        // CASE WHEN condition THEN expr so the first child of CASE is null
        assert(origin->GetChildByIndex(0) == nullptr);
        assert(origin->GetChildrenCount() % 2 == 0);

        value_type.DataType.Length = origin->GetLength();

        break;
    }
    case BiaodashiType::Andor:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        else
        {
            type = AriesExprType::AND_OR;
            content = static_cast<int>(getLogicType(
                    static_cast<LogicType>(boost::get<int>(origin->GetContent()))));
        }
        break;
    }
    case BiaodashiType::Kuohao:
    {
        type = AriesExprType::BRACKETS;
        content = boost::get<int>(origin->GetContent());
        break;
    }
    case BiaodashiType::Zhenjia:
    {
        type = AriesExprType::TRUE_FALSE;
        content = boost::get<bool>(origin->GetContent());
        break;
    }
    case BiaodashiType::IfCondition:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }

        type = AriesExprType::IF_CONDITION;
        content = boost::get<int>(origin->GetContent());

        value_type.DataType.Length = origin->GetLength();
        break;
    }
    case BiaodashiType::Null:
    {
        type = AriesExprType::NULL_VALUE;
        break;
    }
    case BiaodashiType::IsNull:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        else
        {
            type = AriesExprType::IS_NULL;
        }
        break;
    }
    case BiaodashiType::IsNotNull:
    {
        auto result = exprSimplifier.Simplify(origin, m_thd);
        if (result)
        {
            //if value type is unknown, we use origin's value type
            if( result->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN )
            {
                assert( value_type.HasNull );
                result->SetValueType( value_type );
            }
            return result;
        }
        else
        {
            type = AriesExprType::IS_NOT_NULL;
        }
        break;
    }
    case BiaodashiType::Distinct:
        type = AriesExprType::DISTINCT;
        content = boost::get<bool>(origin->GetContent());
        break;
    case BiaodashiType::IntervalExpression:
        type = AriesExprType::INTERVAL;
        content = boost::get<std::string>(origin->GetContent());
        break;
    case BiaodashiType::Buffer:
        type = AriesExprType::BUFFER;
        content = origin->GetBuffer();
        break;
    case BiaodashiType::Query:
        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "subquery is not supported here");
        break;
    default:
        ARIES_ASSERT( 0, "Unsupported expression type: " + std::to_string( ( int )origin->GetType() ) );
        break;
    }

    auto result = AriesCommonExpr::Create(type, content, value_type);

    bridgeChildren(exprSimplifier, result, origin);

    switch ( result->GetType() )
    {
        /**
         * 检查参与计算的数据类型，只有数字类型才能参与计算
         */
        case AriesExprType::CALC:
        {
            assert( result->GetChildrenCount() == 2 );
            auto calcType = static_cast< AriesCalculatorOpType >( boost::get< int >( result->GetContent() ) );
            if ( calcType == AriesCalculatorOpType::MOD )
            {
                /**
                 * cuda 不支持浮点数（float/double）参与求余运算
                 */
                ValidateValueType( "MOD calculation", result->GetChild( 0 ), { AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::INT8,
                                                                            AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
                ValidateValueType( "MOD calculation", result->GetChild( 1 ), { AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL,  AriesValueType::INT8,
                                                                            AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
            }
            else
            {
                ValidateValueType( "calculation", result->GetChild( 0 ), { AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::DOUBLE,
                                                                AriesValueType::FLOAT, AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
                ValidateValueType( "calculation", result->GetChild( 1 ), { AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::DOUBLE,
                                                                AriesValueType::FLOAT, AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
            }
            break;
        }
        case AriesExprType::SQL_FUNCTION:
        {
            auto functionType = static_cast< AriesSqlFunctionType >( boost::get< int >( result->GetContent() ) );
            string functionName = GetAriesSqlFunctionTypeName( functionType );
            switch ( functionType )
            {
                case AriesSqlFunctionType::ABS:
                {
                    assert( result->GetChildrenCount() == 1 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::TIME, AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::DOUBLE,
                                                                AriesValueType::FLOAT, AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
                    break;
                }
                case AriesSqlFunctionType::DATE_ADD:
                case AriesSqlFunctionType::DATE_SUB:
                {
                    assert( result->GetChildrenCount() == 2 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::CHAR } );
                    result->GetChild( 1 )->GetIntervalUnitTypeName();//this function throw not support exception if interval type is not supported;
                    break;
                }
                case AriesSqlFunctionType::DATE_DIFF:
                {
                    assert( result->GetChildrenCount() == 2 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::CHAR } );
                    ValidateValueType( functionName, result->GetChild( 1 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::CHAR } );
                    break;
                }
                case AriesSqlFunctionType::TIME_DIFF:
                {
                    assert( result->GetChildrenCount() == 2 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::TIME, AriesValueType::CHAR } );
                    ValidateValueType( functionName, result->GetChild( 1 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::TIME, AriesValueType::CHAR } );
                    if( result->GetChild( 0 )->GetValueType().DataType.ValueType != result->GetChild( 1 )->GetValueType().DataType.ValueType )
                        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, ( "different data type in TIME_DIFF, one is " + GetDataTypeStringName( result->GetChild( 0 )->GetValueType().DataType.ValueType )
                        + ", the other is " + GetDataTypeStringName( result->GetChild( 1 )->GetValueType().DataType.ValueType ) ).c_str() );
                    break;
                }
                case AriesSqlFunctionType::UNIX_TIMESTAMP:
                {
                    assert( result->GetChildrenCount() == 2 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::CHAR } );
                    break;
                }
                case AriesSqlFunctionType::SUBSTRING:
                {
                    assert( result->GetChildrenCount() == 3 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::CHAR } );
                    ValidateValueType( functionName, result->GetChild( 1 ), { AriesValueType::INT32 } );
                    ValidateValueType( functionName, result->GetChild( 2 ), { AriesValueType::INT32 } );
                    break;
                }
                case AriesSqlFunctionType::CONCAT:
                {
                    for( int i = 0; i < result->GetChildrenCount(); ++i )
                        ValidateValueType( functionName, result->GetChild( i ), { AriesValueType::CHAR } );
                    break;
                }
                case AriesSqlFunctionType::DATE_FORMAT:
                {
                    assert( result->GetChildrenCount() == 2 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::TIMESTAMP } );
                    ValidateValueType( functionName, result->GetChild( 1 ), { AriesValueType::CHAR } );
                    break;
                }
                case AriesSqlFunctionType::DATE:
                {
                    assert( result->GetChildrenCount() == 1 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::TIMESTAMP, AriesValueType::CHAR } );
                    break;
                }
                case AriesSqlFunctionType::MONTH:
                {
                    assert( result->GetChildrenCount() == 1 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::DATE, AriesValueType::DATETIME, AriesValueType::TIMESTAMP, AriesValueType::CHAR } );
                    break;
                }
                case AriesSqlFunctionType::TRUNCATE:
                {
                    assert( result->GetChildrenCount() == 2 );
                    ValidateValueType( functionName, result->GetChild( 0 ), { AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::INT8,
                                                                                AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
                    break;
                }
                default:
                    break;
            }
            break;
        }
        case AriesExprType::AGG_FUNCTION:
        {
            assert( result->GetChildrenCount() > 0 );
            int dataChildIndex = result->GetChildrenCount() == 1 ? 0 : 1;
            auto aggFunctionType = static_cast< AriesAggFunctionType >( boost::get< int >( result->GetContent() ) );
            string functionName = GetAriesAggFunctionTypeName( aggFunctionType );
            const auto& child = result->GetChild( dataChildIndex );
            switch ( aggFunctionType )
            {
                case AriesAggFunctionType::MAX:
                {
                    ValidateValueType( functionName, child, { AriesValueType::CHAR, AriesValueType::BOOL, AriesValueType::COMPACT_DECIMAL, AriesValueType::DATE,
                                                                AriesValueType::DATETIME, AriesValueType::DECIMAL, AriesValueType::DOUBLE, AriesValueType::FLOAT,
                                                                AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64, AriesValueType::TIME,
                                                                AriesValueType::TIMESTAMP, AriesValueType::YEAR } );
                    if( child->GetValueType().DataType.ValueType == AriesValueType::CHAR && child->GetValueType().DataType.Length > 1 )
                        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "string type in MAX function" );
                    break;
                }
                case AriesAggFunctionType::MIN:
                {
                    ValidateValueType( functionName, child, { AriesValueType::CHAR, AriesValueType::BOOL, AriesValueType::COMPACT_DECIMAL, AriesValueType::DATE,
                                                                AriesValueType::DATETIME, AriesValueType::DECIMAL, AriesValueType::DOUBLE, AriesValueType::FLOAT,
                                                                AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64, AriesValueType::TIME,
                                                                AriesValueType::TIMESTAMP, AriesValueType::YEAR } );
                    if( child->GetValueType().DataType.ValueType == AriesValueType::CHAR && child->GetValueType().DataType.Length > 1 )
                        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "string type in MIN function" );
                    break;
                }
                case AriesAggFunctionType::SUM:
                {
                    ValidateValueType( functionName, child, { AriesValueType::BOOL, AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::DOUBLE,
                                                                AriesValueType::FLOAT, AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
                    break;
                }
                case AriesAggFunctionType::COUNT:
                {
                    ValidateValueType( functionName, child, { AriesValueType::CHAR, AriesValueType::BOOL, AriesValueType::COMPACT_DECIMAL, AriesValueType::DATE,
                                                                AriesValueType::DATETIME, AriesValueType::DECIMAL, AriesValueType::DOUBLE, AriesValueType::FLOAT,
                                                                AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64, AriesValueType::TIME,
                                                                AriesValueType::TIMESTAMP, AriesValueType::YEAR, AriesValueType::UNKNOWN } );
                    if( child->GetValueType().DataType.ValueType == AriesValueType::UNKNOWN && child->GetType() != AriesExprType::STAR )
                        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "UNKNOWN type in COUNT function" );
                    break;
                }
                case AriesAggFunctionType::AVG:
                {
                    ValidateValueType( functionName, child, { AriesValueType::BOOL, AriesValueType::COMPACT_DECIMAL, AriesValueType::DECIMAL, AriesValueType::DOUBLE,
                                                                AriesValueType::FLOAT, AriesValueType::INT8, AriesValueType::INT16, AriesValueType::INT32, AriesValueType::INT64 } );
                    break;
                }
                case AriesAggFunctionType::ANY_VALUE:
                {
                    break;
                }
                default:
                    ARIES_ASSERT( false, "unknown agg function" );
                    break;
            }
            break;
        }
        case AriesExprType::EXISTS:
        {
            const auto& child = result->GetChild( 0 );
            if ( child->GetType() == AriesExprType::BUFFER )
            {
                auto childBuffer = boost::get< AriesDataBufferSPtr >( child->GetContent() );
                result = AriesCommonExpr::Create( AriesExprType::TRUE_FALSE,
                    childBuffer->GetItemCount() > 0,
                    AriesColumnType{ AriesDataType( AriesValueType::BOOL, 1 ), false, false } );
            }
            else
            {
                ARIES_ASSERT( false, "unhandled exists expression" );
            }
            break;
        }
        default:
            break;
    }

    if ( type != AriesExprType::SQL_FUNCTION && value_type.DataType.ValueType == AriesValueType::DECIMAL && origin->GetAssociatedLength() != -1 )
    {
        value_type.DataType.ValueType = AriesValueType::COMPACT_DECIMAL;
        value_type.DataType.Precision = origin->GetLength();
        value_type.DataType.Scale = origin->GetAssociatedLength();
        value_type.DataType.Length = GetDecimalRealBytes( origin->GetLength(), origin->GetAssociatedLength() );
        result->SetValueType( value_type );
    }
    return result;
}

void AriesExprBridge::bridgeChildren( ExpressionSimplifier &exprSimplifier, AriesCommonExprUPtr& self, CommonBiaodashi* origin) const
{
    int count = origin->GetChildrenCount();
    BiaodashiPointer child;

    auto target_type = self->GetValueType();
    for (int i = 0; i < count; i ++ )
    {
        child = origin->GetChildByIndex(i);
        if ( self->GetType() == AriesExprType::CASE )
        {
            if ( i == 0 ) {
                // the first child of CASE should be null, ignore it.
                continue;
            }
            else if ( i == ( count - 1 ) )
            {
                if ( !child )
                {
                    self->AddChild( AriesCommonExpr::Create( AriesExprType::NULL_VALUE, 0, self->GetValueType() ) );
                    continue;
                }
            }
        }


        auto bridged = Bridge(child);

        bool need_to_convert = false;
        switch (self->GetType()) {
            case AriesExprType::CASE: {
                need_to_convert = (i % 2 == 0) || (i == count - 1);
                break;
            }
            case AriesExprType::IF_CONDITION:
                need_to_convert = i > 0;
                break;
            case AriesExprType::COALESCE:
                need_to_convert = true;
                break;
            case AriesExprType::SQL_FUNCTION: {
                auto function_type = static_cast<aries_acc::AriesSqlFunctionType>(boost::get<int>(self->GetContent()));
                switch (function_type) {
                    case aries_acc::AriesSqlFunctionType::CONCAT: {
                        need_to_convert = true;
                        break;
                    }
                    case aries_acc::AriesSqlFunctionType::TRUNCATE:
                    {
                        need_to_convert = i == 1;
                        target_type.DataType.ValueType = AriesValueType::INT32;
                        target_type.HasNull = false;
                        target_type.DataType.Length = 1;
                    }
                    default: break;
                }
                break;
            }
            case AriesExprType::IN:
            case AriesExprType::NOT_IN: {
                if (i > 0) {
                    need_to_convert = true;
                } else {
                    target_type = bridged->GetValueType();
                }
                break;
            }
            default: break;
        }

        if ( need_to_convert ) {
            auto success = false;
            try
            {
                success = exprSimplifier.ConvertExpression( bridged, target_type );
            }
            catch ( ... )
            {
            }

            if ( !success )
            {
                std::string error_message = GetValueTypeAsString( bridged->GetValueType() );
                error_message += " cannot be converted to ";
                error_message += GetValueTypeAsString( target_type );
                ARIES_EXCEPTION_SIMPLE( ER_WRONG_TYPE_FOR_VAR, error_message );
            }
        }

        self->AddChild(std::move(bridged));
    }

    bool need_to_convert_constant_children = false;
    switch (self->GetType()) {
        /**
         * 常量不需要转换
         */
        case AriesExprType::INTEGER:
        case AriesExprType::FLOATING:
        case AriesExprType::DECIMAL:
        case AriesExprType::STRING:
        case AriesExprType::DATE:
        case AriesExprType::DATE_TIME:
        case AriesExprType::TIME:
        case AriesExprType::TIMESTAMP:
        case AriesExprType::YEAR:
        case AriesExprType::TRUE_FALSE:
        case AriesExprType::NULL_VALUE:
        case AriesExprType::INTERVAL:
        case AriesExprType::COLUMN_ID:
        case AriesExprType::STAR:
        case AriesExprType::AGG_FUNCTION:
        case AriesExprType::IN:
        case AriesExprType::NOT_IN:
        case AriesExprType::AND_OR:
        case AriesExprType::EXISTS:
        case AriesExprType::IS_NOT_NULL:
        case AriesExprType::IS_NULL:
        case AriesExprType::ARRAY:
        case AriesExprType::NOT:
        case AriesExprType::DISTINCT:
        case AriesExprType::BRACKETS:
        case AriesExprType::BUFFER:
        case AriesExprType::LIKE:
            break;
        case AriesExprType::CALC:
        case AriesExprType::COMPARISON:
            need_to_convert_constant_children = true;
            break;
        case AriesExprType::SQL_FUNCTION: {
            auto function_type = static_cast<AriesSqlFunctionType>(boost::get<int>(self->GetContent()));
            switch (function_type) {
                case AriesSqlFunctionType::DATE_DIFF:
                case AriesSqlFunctionType::TIME_DIFF:
                case AriesSqlFunctionType::UNIX_TIMESTAMP:
                    need_to_convert_constant_children = true;
                    break;
                default: // TODO: 其它函数是否需要转换
                    break;
            }
            break;
        }
        case AriesExprType::BETWEEN:
            need_to_convert_constant_children = true;
            break;
        case AriesExprType::CASE:
        case AriesExprType::COALESCE:
        case AriesExprType::IF_CONDITION:
            break;
    }

    if (need_to_convert_constant_children) {
        exprSimplifier.ConvertConstantChildrenIfNeed(self); // 此处保存常量
    }
}

AriesJoinType AriesExprBridge::ConvertToAriesJoinType(JoinType type) const
{
    switch (type)
    {
    case JoinType::RightOuterJoin:
    case JoinType::RightJoin:
        return AriesJoinType::RIGHT_JOIN;
    case JoinType::LeftOuterJoin:
    case JoinType::LeftJoin:
        return AriesJoinType::LEFT_JOIN;
    case JoinType::InnerJoin:
        return AriesJoinType::INNER_JOIN;
    case JoinType::FullOuterJoin:
    case JoinType::FullJoin:
        return AriesJoinType::FULL_JOIN;
    case JoinType::AntiJoin:
        return AriesJoinType::ANTI_JOIN;
    case JoinType::SemiJoin:
        return AriesJoinType::SEMI_JOIN;
    default:
    {
        string msg( "unexpected join type: " );
        msg.append( std::to_string( ( int )type ) );
        ARIES_ASSERT( 0, msg );
    }
    }
}

AriesSetOpType AriesExprBridge::ConvertToAriesSetOpType( SetOperationType type ) const
{
    static std::map<SetOperationType, AriesSetOpType> tmp_type_map = {
        {SetOperationType::UNION, AriesSetOpType::UNION},
        {SetOperationType::UNION_ALL, AriesSetOpType::UNION_ALL},
        {SetOperationType::INTERSECT, AriesSetOpType::INTERSECT},
        {SetOperationType::INTERSECT_ALL, AriesSetOpType::INTERSECT_ALL},
        {SetOperationType::EXCEPT, AriesSetOpType::EXCEPT},
        {SetOperationType::EXCEPT_ALL, AriesSetOpType::EXCEPT_ALL}};

    assert(tmp_type_map.find(type) != tmp_type_map.end());
    return tmp_type_map[type];
}

vector<AriesOrderByType> AriesExprBridge::ConvertToAriesOrderType(
    const vector<OrderbyDirection> &directions) const
{
    vector<AriesOrderByType> result;
    for (const auto &dir : directions)
    {
        if (dir == OrderbyDirection::ASC)
        {
            result.push_back(AriesOrderByType::ASC);
        }
        else if (dir == OrderbyDirection::DESC)
        {
            result.push_back(AriesOrderByType::DESC);
        }
        else
        {
            assert(0);
        }
    }
    return result;
}

AriesValueType AriesExprBridge::convertValueType(BiaodashiValueType type) const
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

AriesAggFunctionType
AriesExprBridge::getAggFuncType(const string &arg_value) const
{
    static std::map<string, AriesAggFunctionType> tmp_aggfunctype_map = {
        {"COUNT", AriesAggFunctionType::COUNT},
        {"SUM", AriesAggFunctionType::SUM},
        {"MAX", AriesAggFunctionType::MAX},
        {"MIN", AriesAggFunctionType::MIN},
        {"AVG", AriesAggFunctionType::AVG},
        {"ANY_VALUE", AriesAggFunctionType::ANY_VALUE}};

    assert(tmp_aggfunctype_map.find(arg_value) != tmp_aggfunctype_map.end());
    return tmp_aggfunctype_map[arg_value];
}

AriesCalculatorOpType AriesExprBridge::getCalcType(CalcType arg_value) const
{
    static std::map<CalcType, AriesCalculatorOpType> tmp_calctype_map = {
        {CalcType::ADD, AriesCalculatorOpType::ADD},
        {CalcType::SUB, AriesCalculatorOpType::SUB},
        {CalcType::MUL, AriesCalculatorOpType::MUL},
        {CalcType::DIV, AriesCalculatorOpType::DIV},
        {CalcType::MOD, AriesCalculatorOpType::MOD}};

    assert(tmp_calctype_map.find(arg_value) != tmp_calctype_map.end());
    return tmp_calctype_map[arg_value];
}

AriesComparisonOpType AriesExprBridge::getCompType(ComparisonType arg_value) const
{
    static std::map<ComparisonType, AriesComparisonOpType> tmp_comptype_map = {
        {ComparisonType::DengYu, AriesComparisonOpType::EQ},
        {ComparisonType::BuDengYu, AriesComparisonOpType::NE},
        {ComparisonType::DaYu, AriesComparisonOpType::GT},
        {ComparisonType::DaYuDengYu, AriesComparisonOpType::GE},
        {ComparisonType::XiaoYu, AriesComparisonOpType::LT},
        {ComparisonType::XiaoYuDengYu, AriesComparisonOpType::LE},
        {ComparisonType::SQLBuDengYu, AriesComparisonOpType::NE}};

    assert(tmp_comptype_map.find(arg_value) != tmp_comptype_map.end());
    return tmp_comptype_map[arg_value];
}

AriesLogicOpType AriesExprBridge::getLogicType(LogicType arg_value) const
{
    static std::map<LogicType, AriesLogicOpType> tmp_logictype_map = {
        {LogicType::AND, AriesLogicOpType::AND}, {LogicType::OR, AriesLogicOpType::OR}};

    assert(tmp_logictype_map.find(arg_value) != tmp_logictype_map.end());
    return tmp_logictype_map[arg_value];
}

END_ARIES_ENGINE_NAMESPACE // namespace aries
