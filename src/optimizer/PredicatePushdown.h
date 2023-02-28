#ifndef ARIES_QUERY_OPTIMIZATION_PUSHDOWN
#define ARIES_QUERY_OPTIMIZATION_PUSHDOWN

#include "QueryOptimizationPolicy.h"

namespace aries {

class PredicatePushdown : public QueryOptimizationPolicy {
public:
    PredicatePushdown();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;


    void predicate_pushdown_single_query(SQLTreeNodePointer arg_input);

    void predicate_pushdown_handling_node(SQLTreeNodePointer arg_input);

    void pushdown_from_filter_into_join(SQLTreeNodePointer arg_input);

    void pushdown_from_filter_into_inner_join(SQLTreeNodePointer arg_input);

    void pushdown_from_join_into_child(SQLTreeNodePointer arg_input);

    void pushdown_from_inner_join_into_child( SQLTreeNodePointer arg_input );

    /*it means that we need add a new filter node between the join node and the child node*/
    void join_child_pushdown(SQLTreeNodePointer arg_binary_join_node, SQLTreeNodePointer arg_child_node,
                             std::vector<BiaodashiPointer> arg_expr_list, bool arg_left_or_right);

    bool the_first_list_cover_the_second_one(
            std::vector<BasicRelPointer> arg_first_list,
            std::vector<BasicRelPointer> arg_second_list);

    /**
     * return 0, if cannot be pushed down
     * return -1, if can be pushed down into left
     * return 1, if can be pushed down into right
     */
    int can_be_pushed_down(BiaodashiPointer expr, SQLTreeNodePointer node, bool from_filter_node = false);

};


}
#endif

