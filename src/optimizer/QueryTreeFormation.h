#ifndef ARIES_QUERY_TREE_FORMATION
#define ARIES_QUERY_TREE_FORMATION

#include "QueryOptimizationPolicy.h"

namespace aries {

class QueryTreeFormation : public QueryOptimizationPolicy {
public:

    QueryTreeFormation();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;

    void query_formation_single_query(SQLTreeNodePointer arg_input);

    void query_formation_handling_node(SQLTreeNodePointer arg_input);

    void HandlingColumnNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingColumnNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingColumnNode(SQLTreeNodePointer arg_input);

    void HandlingLimitNode_ProcessingOwnColumns( SQLTreeNodePointer arg_input );

    void HandlingLimitNode_FormingOutputColumns( SQLTreeNodePointer arg_input );

    void HandlingLimitNode( SQLTreeNodePointer arg_input );

    void HandlingGroupNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingGroupNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingGroupNode(SQLTreeNodePointer arg_input);

    void HandlingSortNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingSortNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingSortNode_RepositionColumns(SQLTreeNodePointer arg_input);

    void HandlingSortNode(SQLTreeNodePointer arg_input);


    void HandlingFilterNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingFilterNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingFilterNode(SQLTreeNodePointer arg_input);


    void HandlingBinaryJoinNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingBinaryJoinNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingBinaryJoinNode(SQLTreeNodePointer arg_input);

    void HandlingTableNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingTableNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingTableNode(SQLTreeNodePointer arg_input);

    void HandlingSetOperationNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input);

    void HandlingSetOperationNode_FormingOutputColumns(SQLTreeNodePointer arg_input);

    void HandlingSetOperationNode(SQLTreeNodePointer arg_input);

private:
    void ResetColumnNullableInfo( BiaodashiPointer expr, NodeRelationStructurePointer nrsp, bool& changed_to_nullable );

};

}


#endif
