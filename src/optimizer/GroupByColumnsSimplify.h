#pragma once

#include "QueryOptimizationPolicy.h"

namespace aries {
class GroupByColumnsSimplify :  public QueryOptimizationPolicy
{
public:
    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;

private:
    void handleNode(SQLTreeNodePointer arg_input);
    void handleGroupNode(SQLTreeNodePointer arg_input);
};

}