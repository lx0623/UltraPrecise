#ifndef ARIES_UNCORRELATED_SUBQUERY_HANDLING
#define ARIES_UNCORRELATED_SUBQUERY_HANDLING


#include "QueryOptimizationPolicy.h"

namespace aries {
class UncorrelatedSubqueryHandling : public QueryOptimizationPolicy {
private:

    /*this is the top query -- only it has init_query*/
    SelectStructure *the_query_ss = NULL;


public:

    UncorrelatedSubqueryHandling();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input);


    void uc_subquery_handling_single_query(SQLTreeNodePointer arg_input);

    void uc_subquery_handling_node(SQLTreeNodePointer arg_input);

    void HandlingColumnNode(SQLTreeNodePointer arg_input);

    void HandlingGroupNode(SQLTreeNodePointer arg_input);

    void HandlingSortNode(SQLTreeNodePointer arg_input);

    void HandlingFilterNode(SQLTreeNodePointer arg_input);

    void HandlingBinaryJoinNode(SQLTreeNodePointer arg_input);

    void HandlingTableNode(SQLTreeNodePointer arg_input);


    bool ProcessFilterBasicExpr(BiaodashiPointer arg_expr);

private:
    void handle_expression(BiaodashiPointer expression);
};
}


#endif
