#pragma once

#include "QueryOptimizationPolicy.h"

namespace aries
{

class JoinReorganization : public QueryOptimizationPolicy
{

private:
void handleNode(SQLTreeNodePointer arg_input);
void handleFilterNode(SQLTreeNodePointer arg_input);
void handleJoinNode(SQLTreeNodePointer arg_input);

public:
    virtual std::string ToString() override;
    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;

};

} // namespace aries