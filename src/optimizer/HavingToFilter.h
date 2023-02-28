#ifndef ARIES_HAVING_TO_FILTER
#define ARIES_HAVING_TO_FILTER

#include "QueryOptimizationPolicy.h"

namespace aries {

class HavingToFilter : public QueryOptimizationPolicy {

private:
    bool see_error = false;

    std::vector<BiaodashiPointer> needed_agg_exprs;

    std::map<BiaodashiPointer, int> expr_to_location;

    int original_select_count = -1;

    std::vector<BiaodashiPointer> all_agg_exprs_in_having;
    std::vector<BiaodashiPointer> all_placeholder_exprs;
public:

    HavingToFilter();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;

    void htf_single_query(SQLTreeNodePointer arg_input);

    void htf_handling_node(SQLTreeNodePointer arg_input);

    void HandlingGroupNode(SQLTreeNodePointer arg_input);

    bool ProcessHavingExpr(SQLTreeNodePointer arg_input, BiaodashiPointer arg_expr);

    std::vector<BiaodashiPointer> GetAggExprs(BiaodashiPointer arg_expr);

    void CreatePlaceHolderExprs(BiaodashiPointer arg_expr);

    int FindExprInArray(BiaodashiPointer arg_expr);

    void ConvertTheHavingExpr(BiaodashiPointer arg_expr);

    void ExecuteReplacement(BiaodashiPointer arg_expr);

};


}//namespace


#endif
