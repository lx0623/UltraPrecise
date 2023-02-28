//
// Created by david shen on 2019-07-23.
//

#pragma once

#include "AriesExprJoinNode.h"
#include "AriesCommonExpr.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesJoinTreeGenerator: protected DisableOtherConstructors
    {
    public:
        AriesJoinTreeGenerator();
        virtual ~AriesJoinTreeGenerator();
        AriesExprJoinNodeUPtr ConvertToCalcTree( AriesJoinType type, const AriesCommonExprUPtr& rootExpr ) const;

    private:
        unique_ptr< AriesExprJoinComparisonNode > MakeComparisonNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinColumnIdNode > MakeColumnIdNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinAndOrNode > MakeAndOrNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinLiteralNode > MakeStringNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinLiteralNode > MakeIntegerNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinLiteralNode > MakeFloatingNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinLiteralNode > MakeDateNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinLiteralNode > MakeDatetimeNode( AriesJoinType type, const AriesCommonExprUPtr& expr ) const;
        unique_ptr< AriesExprJoinCartesianProductNode > MakeCartesianProductNode( AriesJoinType type ) const;
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
