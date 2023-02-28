//
// Created by david shen on 2019-07-23.
//

#include "AriesJoinTreeGenerator.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesJoinTreeGenerator::AriesJoinTreeGenerator()
    {
        // TODO Auto-generated constructor stub

    }

    AriesJoinTreeGenerator::~AriesJoinTreeGenerator()
    {
        // TODO Auto-generated destructor stub
    }

    AriesExprJoinNodeUPtr AriesJoinTreeGenerator::ConvertToCalcTree( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        AriesExprJoinNodeUPtr result;
        AriesExprType exprType = expr->GetType();
        LOG(INFO) << "AriesJoinTreeGenerator::ConvertToCalcTree: exprType = " << int( exprType ) << "\n";
        switch( exprType )
        {
            case AriesExprType::COMPARISON:
            {
                result = MakeComparisonNode( type, expr );
                break;
            }
            case AriesExprType::COLUMN_ID:
            {
                result = MakeColumnIdNode( type, expr );
                break;
            }
            case AriesExprType::AND_OR:
            {
                result = MakeAndOrNode( type, expr );
                break;
            }
            case AriesExprType::TRUE_FALSE:
            {
                result = MakeCartesianProductNode( type );
                break;
            }
            case AriesExprType::STRING:
            {
                result = MakeStringNode( type, expr );
                break;
            }
            case AriesExprType::INTEGER:
            {
                result = MakeIntegerNode( type, expr );
                break;
            }
            case AriesExprType::FLOATING:
            {
                result = MakeFloatingNode( type, expr );
                break;
            }
            case AriesExprType::DATE:
            {
                result = MakeDateNode( type, expr );
                break;
            }
                //added by Rubao Li -- November 17, 2018 ----- start
                //if we met a brackets, then we directly process it child
            case AriesExprType::BRACKETS:
            {
                return this->ConvertToCalcTree( type, expr->GetChild( 0 ) );
                break;
            }
                //added by Rubao Li -- November 17, 2018 ----- end

            default:
                //FIXME we need support other exprTypes.
                assert( 0 );
                ARIES_ENGINE_EXCEPTION(ER_NOT_SUPPORTED_YET, "converting expression type " + GetAriesExprTypeName(exprType) + " for JOIN expression");
                break;
        }
        int count = expr->GetChildrenCount();
        for( int i = 0; i < count; ++i )
        {
            result->AddChild( ConvertToCalcTree( type, expr->GetChild( i ) ) );
        }
        return result;
    }

    unique_ptr< AriesExprJoinComparisonNode > AriesJoinTreeGenerator::MakeComparisonNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinComparisonNode::Create( type, static_cast< AriesComparisonOpType >( boost::get< int >( expr->GetContent() ) ) );
    }

    unique_ptr< AriesExprJoinColumnIdNode > AriesJoinTreeGenerator::MakeColumnIdNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinColumnIdNode::Create( type, boost::get< int >( expr->GetContent() ) );
    }

    unique_ptr< AriesExprJoinAndOrNode > AriesJoinTreeGenerator::MakeAndOrNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinAndOrNode::Create( type, static_cast< AriesLogicOpType >( boost::get< int >( expr->GetContent() ) ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesJoinTreeGenerator::MakeStringNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinLiteralNode::Create( type, boost::get< string >( expr->GetContent() ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesJoinTreeGenerator::MakeIntegerNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        if ( expr->GetValueType().DataType.ValueType == AriesValueType::INT64 )
        {
            return AriesExprJoinLiteralNode::Create( type, boost::get< int64_t >( expr->GetContent() ) );
        }
        else
        {
            return AriesExprJoinLiteralNode::Create( type, boost::get< int >( expr->GetContent() ) );
        }
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesJoinTreeGenerator::MakeFloatingNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinLiteralNode::Create( type, boost::get< double >( expr->GetContent() ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesJoinTreeGenerator::MakeDateNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinLiteralNode::Create( type, boost::get< AriesDate >( expr->GetContent() ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesJoinTreeGenerator::MakeDatetimeNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const
    {
        return AriesExprJoinLiteralNode::Create( type, boost::get< AriesDatetime >( expr->GetContent() ) );
    }

    unique_ptr< AriesExprJoinCartesianProductNode > AriesJoinTreeGenerator::MakeCartesianProductNode( AriesJoinType type ) const
    {
        return AriesExprJoinCartesianProductNode::Create( type );
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
