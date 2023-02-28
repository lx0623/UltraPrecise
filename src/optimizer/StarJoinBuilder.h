#pragma once

#include "QueryOptimizationPolicy.h"

namespace aries
{

struct HashJoinPair
{
    SQLTreeNodePointer hash_node; // contains hash key
    SQLTreeNodePointer value_node;
    std::vector< CommonBiaodashiPtr > conditions;
    CommonBiaodashiPtr other_condition;

    HashJoinPair( const SQLTreeNodePointer& hash_node,
                  const SQLTreeNodePointer& value_node,
                  const std::vector< CommonBiaodashiPtr >& conditions,
                  const CommonBiaodashiPtr& other_condition = nullptr )
    {
        this->hash_node = hash_node;
        this->value_node = value_node;
        this->conditions.assign( conditions.cbegin(), conditions.cend() );
        this->other_condition = other_condition;
    }
};

struct NonHashJoinNode
{
    SQLTreeNodePointer node;
    CommonBiaodashiPtr condition;
    CommonBiaodashiPtr other_condition;

    NonHashJoinNode( const SQLTreeNodePointer& node,
                     const CommonBiaodashiPtr& condition,
                     const CommonBiaodashiPtr& other_condition = nullptr )
    {
        this->node = node;
        this->condition = condition;
        this->other_condition = other_condition;
    }
};

class StarJoinBuilder : public QueryOptimizationPolicy
{
private:
    void handleNode( SQLTreeNodePointer arg_input );
    void handleJoinNode( SQLTreeNodePointer arg_input );
    void handleInnerJoinNode( SQLTreeNodePointer arg_input, std::vector< HashJoinPair >& pairs, std::vector< NonHashJoinNode >& other_nodes );

public:
    virtual std::string ToString() override;
    virtual SQLTreeNodePointer OptimizeTree( SQLTreeNodePointer arg_input ) override;

};

} // namespace aries