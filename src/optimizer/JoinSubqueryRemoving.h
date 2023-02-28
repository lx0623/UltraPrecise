#ifndef ARIES_JOIN_SUBQUERY_REMOVING
#define ARIES_JOIN_SUBQUERY_REMOVING


#include "QueryOptimizationPolicy.h"

namespace aries {
class JoinSubqueryRemoving : public QueryOptimizationPolicy {
private:

    int sequence = 0;

    AbstractQueryPointer the_subquery_aqp = nullptr;
    SelectStructure *the_subquery_ss = NULL;

    BiaodashiPointer outer_used_expr = nullptr;
    BiaodashiPointer outer_embedded_expr = nullptr; // i am the biaodashi
    ColumnShellPointer outer_embedded_column = nullptr; //i am its content
    BiaodashiPointer needed_groupby_expr = nullptr;
    std::vector< BiaodashiPointer > needed_groupby_exprs;
    std::vector< BiaodashiPointer > outer_embedded_exprs;

    bool the_top_column_node = false;
    bool see_group_node = false;

    /*we now process only the case of one correlated expr*/
    BiaodashiPointer correlated_expr;
    std::vector<BiaodashiPointer> uncorrelated_exprs;
    std::vector< BiaodashiPointer > correlated_exprs;


    BiaodashiPointer the_removable_expr = nullptr;

    SQLTreeNodePointer the_column_node = nullptr;
    SQLTreeNodePointer the_group_node = nullptr;
    SQLTreeNodePointer the_filter_node = nullptr;

    SQLTreeNodePointer outer_query_node = nullptr;

public:

    JoinSubqueryRemoving();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input);


    void js_subquery_handling_single_query(SQLTreeNodePointer arg_input);

    void js_subquery_handling_node(SQLTreeNodePointer arg_input);

    void HandlingColumnNode(SQLTreeNodePointer arg_input);

    void HandlingGroupNode(SQLTreeNodePointer arg_input);

    void HandlingSortNode(SQLTreeNodePointer arg_input);

    void HandlingFilterNode(SQLTreeNodePointer arg_input);

    void HandlingBinaryJoinNode(SQLTreeNodePointer arg_input);
    void HandlingInnerJoinNode( SQLTreeNodePointer arg_input );

    void HandlingTableNode(SQLTreeNodePointer arg_input);


    bool CheckBasicExpr(BiaodashiPointer arg_expr);

    bool CheckSubquery(SelectStructure *arg_ss);

    bool ProcessSubqueryNode(SQLTreeNodePointer arg_input);


    bool CheckFilterItemInsideSubquery(BiaodashiPointer arg_expr);

    bool MakeSureExprNoOuterColumn(BiaodashiPointer arg_expr);

    void ResetMyself();


    void ApplyOptimization(SQLTreeNodePointer arg_input);
};
}


#endif
