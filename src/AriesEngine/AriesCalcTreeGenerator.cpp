/*
 * AriesCalcNodeGenerator.cpp
 *
 *  Created on: Sep 15, 2018
 *      Author: lichi
 */
#include "AriesCalcTreeGenerator.h"
#include "AriesAssert.h"
#include "AriesUtil.h"
#include "CudaAcc/AriesEngineException.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesCalcTreeGenerator::AriesCalcTreeGenerator()
    {
        // TODO Auto-generated constructor stub
    }

    AriesCalcTreeGenerator::~AriesCalcTreeGenerator()
    {
        // TODO Auto-generated destructor stub
    }

    AEExprNodeUPtr AriesCalcTreeGenerator::ConvertToCalcTree( const AriesCommonExprUPtr& expr, int nodeId ) const
    {
        AEExprNodeUPtr result;
        AriesExprType exprType = expr->GetType();
        switch( exprType )
        {
            case AriesExprType::AND_OR:
            {
                result = MakeAndOrNode( expr, nodeId );
                break;
            }
            case AriesExprType::COMPARISON:
            {
                result = MakeComparisonNode( expr );
                break;
            }
            case AriesExprType::CALC:
            {
                AriesColumnType value_type = expr->GetValueType();
                // 集成的 xmp 算法仅适用于 Decimal 相关的计算
                if(value_type.DataType.ValueType == aries::AriesValueType::COMPACT_DECIMAL){
                    result = MakeCalcXmpNode( expr, nodeId);
                }
                else{
                    result = MakeCalcNode( expr, nodeId );
                }
                break;
            }
            case AriesExprType::INTEGER:
            {
                result = MakeIntegerNode( expr );
                break;
            }
            case AriesExprType::FLOATING:
            {
                result = MakeFloatingNode( expr );
                break;
            }
            case AriesExprType::DECIMAL:
            {
                result = MakeDecimalNode( expr );
                break;
            }
            case AriesExprType::DATE:
            {
                result = MakeDateNode( expr );
                break;
            }
            case AriesExprType::DATE_TIME:
            {
                result = MakeDatetimeNode( expr );
                break;
            }
            case AriesExprType::TIME:
            {
                result = MakeTimeNode( expr );
                break;
            }
            case AriesExprType::STRING:
            {
                result = MakeStringNode( expr );
                break;
            }
            case AriesExprType::BETWEEN:
            {
                result = MakeBetweenNode( expr );
                break;
            }
            case AriesExprType::COLUMN_ID:
            {
                if(expr->value_reverse == true)
                    result = MakeCalcNode( expr, nodeId );
                else
                    result = MakeColumnIdNode( expr );
                break;
            }
            case AriesExprType::AGG_FUNCTION:
            {
                result = MakeAggFunctionNode( expr );
                break;
            }
            case AriesExprType::BRACKETS:
            {
                result = ConvertToCalcTree( expr->GetChild( 0 ), nodeId );
                break;
            }
            case AriesExprType::IN:
            {
                result = MakeInNode( expr );
                break;
            }
            case AriesExprType::NOT_IN:
            {
                result = MakeNotInNode( expr );
                break;
            }
            case AriesExprType::NOT:
            {
                result = MakeNotNode( expr );
                break;
            }
            case AriesExprType::LIKE:
            {
                result = MakeLikeNode( expr );
                break;
            }
            case AriesExprType::SQL_FUNCTION:
            {
                result = MakeSqlFunctionNode( expr, nodeId );
                break;
            }
            case AriesExprType::CASE:
            case AriesExprType::IF_CONDITION:
            case AriesExprType::COALESCE:
            {
                result = MakeCaseNode( expr, nodeId );
                break;
            }
            case AriesExprType::DISTINCT:
            {
                //"distinct" keyword, do nothing here.
                break;
            }
            case AriesExprType::STAR:
            {
                result = MakeStarNode( expr );
                break;
            }
            case AriesExprType::IS_NULL:
            {
                result = MakeIsNullNode( expr );
                break;
            }
            case AriesExprType::IS_NOT_NULL:
            {
                result = MakeIsNotNullNode( expr );
                break;
            }
            case AriesExprType::TRUE_FALSE:
            {
                result = MakeTrueFalseNode( expr );
                break;
            }
            case AriesExprType::INTERVAL:
            {
                result = MakeIntervalNode( expr );
                break;
            }
            case AriesExprType::NULL_VALUE:
            {
                result = MakeNullNode( expr );
                break;
            }
            case AriesExprType::BUFFER:
            {
                result = MakeBufferNode( expr );
                break;
            }
            case AriesExprType::TIMESTAMP:
            {
                result = MakeTimestampNode( expr );
                break;                
            }
            case AriesExprType::YEAR:
            {
                result = MakeYearNode( expr );
                break;
            }
            default:
                LOG(ERROR) << "unsupported type: " << GetAriesExprTypeName(exprType) << std::endl;
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "calc expression type " + GetAriesExprTypeName( exprType ) );
                break;
        }

        if( exprType == AriesExprType::AND_OR || exprType == AriesExprType::CALC || exprType == AriesExprType::BRACKETS || exprType == AriesExprType::CASE
                || exprType == AriesExprType::IF_CONDITION || exprType == AriesExprType::COALESCE )
        {
            //do nothing. Because all children of these node has been processed in MakeXxxNode function
        }
        else
        {
            // process children
            int count = expr->GetChildrenCount();
            for( int i = 0; i < count; ++i )
            {
                AEExprNodeUPtr childNode = ConvertToCalcTree( expr->GetChild( i ), nodeId );
                if( childNode )
                    result->AddChild( std::move( childNode ) );
            }
        }
        return result;
    }

    unique_ptr< AEExprAndOrNode > AriesCalcTreeGenerator::MakeAndOrNode( const AriesCommonExprUPtr& expr, int nodeId ) const
    {
        map< string, AriesCommonExprUPtr > aggFunctions;
        vector< AriesDynamicCodeParam > ariesParams;
        vector< AriesDataBufferSPtr > constValues;
        vector< AriesDynamicCodeComparator > ariesComparators;
        string calcExpr = expr->StringForDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators );
        LOG(INFO) << "and_or_expr->" << calcExpr << endl;
        for ( auto& param : ariesParams )
        {
            param.ColumnIndex = abs( param.ColumnIndex );
        }
        for ( auto& param : ariesParams )
        {
            param.ColumnIndex = abs( param.ColumnIndex );
        }
        return AEExprAndOrNode::Create( nodeId,
                                        expr->GetId(),
                                        static_cast< AriesLogicOpType >( boost::get< int >( expr->GetContent() ) ),
                                        std::move( ariesParams ),
                                        std::move( constValues ),
                                        std::move( ariesComparators ),
                                        calcExpr );
    }

    unique_ptr< AEExprComparisonNode > AriesCalcTreeGenerator::MakeComparisonNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprComparisonNode::Create( static_cast< AriesComparisonOpType >( boost::get< int >( expr->GetContent() ) ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeIntegerNode( const AriesCommonExprUPtr& expr ) const
    {
        switch ( expr->GetValueType().DataType.ValueType )
        {
            case AriesValueType::INT8:
                return AEExprLiteralNode::Create( boost::get< int8_t >( expr->GetContent() ) );
            case AriesValueType::INT16:
                return AEExprLiteralNode::Create( boost::get< int16_t >( expr->GetContent() ) );
            case AriesValueType::INT32:
                return AEExprLiteralNode::Create( boost::get< int >( expr->GetContent() ) );
            case AriesValueType::INT64:
                return AEExprLiteralNode::Create( boost::get< int64_t >( expr->GetContent() ) );
            default:
                ARIES_EXCEPTION_SIMPLE( ER_WRONG_TYPE_COLUMN_VALUE_ERROR, "invalid type" );
        }
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeFloatingNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< double >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeDecimalNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< aries_acc::Decimal >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeDateNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< aries_acc::AriesDate >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeDatetimeNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< aries_acc::AriesDatetime >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeTimeNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< aries_acc::AriesTime >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeTimestampNode( const AriesCommonExprUPtr &expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< aries_acc::AriesTimestamp >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeYearNode( const AriesCommonExprUPtr &expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< aries_acc::AriesYear >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeStringNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( boost::get< string >( expr->GetContent() ) );
    }

    unique_ptr< AEExprLiteralNode > AriesCalcTreeGenerator::MakeNullNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLiteralNode::Create( AriesNull() );
    }

    unique_ptr< AEExprCalcNode > AriesCalcTreeGenerator::MakeCalcNode( const AriesCommonExprUPtr& expr, int nodeId ) const
    {
        map< string, AriesCommonExprUPtr > aggFunctions;
        vector< AriesDynamicCodeParam > ariesParams;
        vector< AriesDataBufferSPtr > constValues;
        vector< AriesDynamicCodeComparator > ariesComparators;
        string calcExpr = expr->StringForDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators );
        LOG(INFO) << "calcExpr->" << calcExpr << endl;  //这里是生成的表达式动态代码
        for ( auto& param : ariesParams )
        {
            param.ColumnIndex = abs( param.ColumnIndex );
        }
        map< string, unique_ptr< AEExprAggFunctionNode > > aggNodes;
        for( auto& func : aggFunctions )
        {
            AEExprNodeUPtr agg = ConvertToCalcTree( func.second, nodeId );
            func.second.release(); // we borrow from expr, there will be two unique_ptr point to the same agg node. so release ownership here.
            aggNodes[func.first] = ( unique_ptr< AEExprAggFunctionNode > )( ( AEExprAggFunctionNode* )agg.release() );
        }

        return AEExprCalcNode::Create( nodeId,
                                       expr->GetId(),
                                       std::move( aggNodes ),
                                       std::move( ariesParams ),
                                       std::move( constValues ),
                                       std::move( ariesComparators ),
                                       calcExpr,
                                       expr->GetValueType() );
    }

    unique_ptr< AEExprCalcNode > AriesCalcTreeGenerator::MakeCalcXmpNode( const AriesCommonExprUPtr& expr, int nodeId ) const
    {
        map< string, AriesCommonExprUPtr > aggFunctions;
        vector< AriesDynamicCodeParam > ariesParams;
        vector< AriesDataBufferSPtr > constValues;
        vector< AriesDynamicCodeComparator > ariesComparators;

        // 这里决定 长度
        int ansLEN = expr->GetValueType().DataType.AdaptiveLen;
        if(ansLEN <= 4)
            ansLEN = 4;
        else if(ansLEN <= 8)
            ansLEN = 8;
        else if(ansLEN <= 16)
            ansLEN = 16;
        else
            ansLEN = 32;
        // 这里赋值 TPI 默认 TPI 为 4
        int ansTPI = 32;

        string calcExpr = expr->StringForXmpDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators, ansLEN, ansTPI );
        LOG(INFO) << "calcExpr->" << calcExpr << endl;  //这里是生成的表达式动态代码
        for ( auto& param : ariesParams )
        {
            param.ColumnIndex = abs( param.ColumnIndex );
        }
        map< string, unique_ptr< AEExprAggFunctionNode > > aggNodes;
        for( auto& func : aggFunctions )
        {
            AEExprNodeUPtr agg = ConvertToCalcTree( func.second, nodeId );
            func.second.release(); // we borrow from expr, there will be two unique_ptr point to the same agg node. so release ownership here.
            aggNodes[func.first] = ( unique_ptr< AEExprAggFunctionNode > )( ( AEExprAggFunctionNode* )agg.release() );
        }

        return AEExprCalcNode::CreateForXmp( nodeId,
                                       expr->GetId(),
                                       std::move( aggNodes ),
                                       std::move( ariesParams ),
                                       std::move( constValues ),
                                       std::move( ariesComparators ),
                                       calcExpr,
                                       expr->GetValueType(),
                                       ansLEN,
                                       ansTPI );
    }

    unique_ptr< AEExprBetweenNode > AriesCalcTreeGenerator::MakeBetweenNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprBetweenNode::Create();
    }

    unique_ptr< AEExprColumnIdNode > AriesCalcTreeGenerator::MakeColumnIdNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprColumnIdNode::Create( boost::get< int >( expr->GetContent() ) );
    }

    unique_ptr< AEExprAggFunctionNode > AriesCalcTreeGenerator::MakeAggFunctionNode( const AriesCommonExprUPtr& expr ) const
    {
        return MakeAggFunctionNode( expr.get() );
    }

    unique_ptr< AEExprAggFunctionNode > AriesCalcTreeGenerator::MakeAggFunctionNode( const AriesCommonExpr* expr ) const
    {
        return AEExprAggFunctionNode::Create( expr->GetAggFunctionType(), expr->IsDistinct() );
    }

    unique_ptr< AEExprInNode > AriesCalcTreeGenerator::MakeInNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprInNode::Create( false );
    }

    unique_ptr< AEExprInNode > AriesCalcTreeGenerator::MakeNotInNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprInNode::Create( true );
    }

    unique_ptr< AEExprNotNode > AriesCalcTreeGenerator::MakeNotNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprNotNode::Create();
    }

    unique_ptr< AEExprLikeNode > AriesCalcTreeGenerator::MakeLikeNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprLikeNode::Create();
    }

    unique_ptr< AEExprSqlFunctionNode > AriesCalcTreeGenerator::MakeSqlFunctionNode( const AriesCommonExprUPtr& expr, int nodeId ) const
    {
        map< string, AriesCommonExprUPtr > aggFunctions;
        vector< AriesDynamicCodeParam > ariesParams;
        vector< AriesDataBufferSPtr > constValues;
        vector< AriesDynamicCodeComparator > ariesComparators;
        string calcExpr = expr->StringForDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators );
        LOG(INFO) << "SqlFunctionExpr->" << calcExpr << endl;
        for ( auto& param : ariesParams )
        {
            param.ColumnIndex = abs( param.ColumnIndex );
        }
        map< string, unique_ptr< AEExprAggFunctionNode > > aggNodes;
        for( auto& func : aggFunctions )
        {
            AEExprNodeUPtr agg = ConvertToCalcTree( func.second, nodeId );
            func.second.release(); // we borrow from expr, there will be two unique_ptr point to the same agg node. so release ownership here.
            aggNodes[func.first] = ( unique_ptr< AEExprAggFunctionNode > )( ( AEExprAggFunctionNode* )agg.release() );
        }

        return AEExprSqlFunctionNode::Create( nodeId,
                                              expr->GetId(),
                                              expr->GetSqlFunctionType(),
                                              std::move( aggNodes ),
                                              std::move( ariesParams ),
                                              std::move( constValues ),
                                              std::move( ariesComparators ),
                                              calcExpr,
                                              expr->GetValueType() );
    }

    unique_ptr< AEExprCaseNode > AriesCalcTreeGenerator::MakeCaseNode( const AriesCommonExprUPtr& expr, int nodeId ) const
    {
        map< string, AriesCommonExprUPtr > aggFunctions;
        vector< AriesDynamicCodeParam > ariesParams;
        vector< AriesDataBufferSPtr > constValues;
        vector< AriesDynamicCodeComparator > ariesComparators;
        string calcExpr = expr->StringForDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators );
        for ( auto& param : ariesParams )
        {
            param.ColumnIndex = abs( param.ColumnIndex );
        }
        LOG(INFO) << "caseExpr->" << calcExpr << endl;
        map< string, unique_ptr< AEExprAggFunctionNode > > aggNodes;
        for( auto& func : aggFunctions )
        {
            AEExprNodeUPtr agg = ConvertToCalcTree( func.second, nodeId );
            func.second.release(); // we borrow from expr, there will be two unique_ptr point to the same agg node. so release ownership here.
            aggNodes[func.first] = ( unique_ptr< AEExprAggFunctionNode > )( ( AEExprAggFunctionNode* )agg.release() );
        }

        return AEExprCaseNode::Create( nodeId,
                                       expr->GetId(),
                                       std::move( aggNodes ),
                                       std::move( ariesParams ),
                                       std::move( constValues ),
                                       std::move( ariesComparators ),
                                       calcExpr,
                                       expr->GetValueType() );
    }

    unique_ptr< AEExprStarNode > AriesCalcTreeGenerator::MakeStarNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprStarNode::Create();
    }

    unique_ptr< AEExprIsNullNode > AriesCalcTreeGenerator::MakeIsNullNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprIsNullNode::Create( false );
    }

    unique_ptr< AEExprIsNullNode > AriesCalcTreeGenerator::MakeIsNotNullNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprIsNullNode::Create( true );
    }

    unique_ptr< AEExprTrueFalseNode > AriesCalcTreeGenerator::MakeTrueFalseNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprTrueFalseNode::Create( boost::get<bool>(expr->GetContent()) );
    }

    unique_ptr< AEExprIntervalNode > AriesCalcTreeGenerator::MakeIntervalNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprIntervalNode::Create( boost::get< string >( expr->GetContent() ) );
    }

    unique_ptr< AEExprBufferNode > AriesCalcTreeGenerator::MakeBufferNode( const AriesCommonExprUPtr& expr ) const
    {
        return AEExprBufferNode::Create( boost::get< AriesDataBufferSPtr >( expr->GetContent() ) );
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
