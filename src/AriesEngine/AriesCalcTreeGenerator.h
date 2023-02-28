/*
 * AriesCalcNodeGenerator.h
 *
 *  Created on: Sep 15, 2018
 *      Author: lichi
 */

#pragma once
#include "AriesCommonExpr.h"
#include "AriesExprCalcNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesCalcTreeGenerator: protected DisableOtherConstructors
    {
    public:
        AriesCalcTreeGenerator();
        virtual ~AriesCalcTreeGenerator();
        AEExprNodeUPtr ConvertToCalcTree( const AriesCommonExprUPtr& rootExpr, int nodeId ) const;

    private:
        unique_ptr< AEExprAndOrNode > MakeAndOrNode( const AriesCommonExprUPtr& expr, int nodeId ) const;
        unique_ptr< AEExprComparisonNode > MakeComparisonNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeIntegerNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeFloatingNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeDecimalNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeDateNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeDatetimeNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeTimeNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeTimestampNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeYearNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeStringNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLiteralNode > MakeNullNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprCalcNode > MakeCalcNode( const AriesCommonExprUPtr& expr, int nodeId ) const;
        unique_ptr< AEExprCalcNode > MakeCalcXmpNode( const AriesCommonExprUPtr& expr, int nodeId ) const;
        unique_ptr< AEExprBetweenNode > MakeBetweenNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprColumnIdNode > MakeColumnIdNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprAggFunctionNode > MakeAggFunctionNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprAggFunctionNode > MakeAggFunctionNode( const AriesCommonExpr* expr ) const;
        unique_ptr< AEExprInNode > MakeInNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprInNode > MakeNotInNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprNotNode > MakeNotNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprLikeNode > MakeLikeNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprSqlFunctionNode > MakeSqlFunctionNode( const AriesCommonExprUPtr& expr, int nodeId ) const;
        unique_ptr< AEExprCaseNode > MakeCaseNode( const AriesCommonExprUPtr& expr, int nodeId ) const;
        unique_ptr< AEExprStarNode > MakeStarNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprIsNullNode > MakeIsNullNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprIsNullNode > MakeIsNotNullNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprTrueFalseNode > MakeTrueFalseNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprIntervalNode > MakeIntervalNode( const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AEExprBufferNode > MakeBufferNode( const AriesCommonExprUPtr& expr ) const;
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

