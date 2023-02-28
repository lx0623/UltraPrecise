#pragma once

#include "QueryOptimizationPolicy.h"
#include "frontend/SQLTreeNodeBuilder.h"

namespace aries
{

class ExchangeBuilder : public QueryOptimizationPolicy
{
public:
    virtual std::string ToString() override;
    virtual SQLTreeNodePointer OptimizeTree( SQLTreeNodePointer arg_input ) override;

private:
    void handleNode( SQLTreeNodePointer arg_input );
    void handleGroupNode( SQLTreeNodePointer arg_input );
    SQLTreeNodePointer buildExchangeNode( SQLTreeNodePointer node );
    SQLTreeNodePointer buildExchangeNodeForGroupNode( SQLTreeNodePointer group_node );
    SQLTreeNodePointer buildExchangeNodeForFilterNode( SQLTreeNodePointer filter_node );

    SQLTreeNodePointer createExchangeNode( SQLTreeNodeBuilder& builder );
};

}