#ifndef ARIES_SUBQUERY_UNNESTING
#define ARIES_SUBQUERY_UNNESTING

#include "QueryOptimizationPolicy.h"

namespace aries
{

    class SubqueryUnnesting : public QueryOptimizationPolicy
    {

    private:
        SelectStructure *the_subquery_ss;
        SQLTreeNodePointer subquery_remain_node;

        int exist_or_in = 0; //1 means exist, -1 mean in;

        /*a temp solution for duplicate table names after unnesting*/
        /*this works for the subquery that has no further subqueries*/
        /*after scanning the subquery, we obtain all columns & the table*/
        /*then we change the table name to a new one!*/
        /*TODO: for general case, we should create a ColumnNode to cover the subquery_remain_code!*/
        std::vector<ColumnShellPointer> all_columns_in_subquery;
        std::vector<SQLTreeNodePointer> all_table_nodes_in_subquery;
        std::vector<SQLTreeNodePointer> all_column_nodes_in_subquery;
        //SQLTreeNodePointer the_table_node_in_subquery;
        int renaming_seq = 0;

        /*for each exists subquery*/
        std::vector<BiaodashiPointer> correlated_items;
        std::vector<BiaodashiPointer> uncorrelated_items;
        std::vector<ColumnShellPointer> outer_columns;

        bool is_a_not_exist = false;

        /*for each in subquery*/
        BiaodashiPointer the_expr_outside;
        BiaodashiPointer the_expr_inside;
        bool is_a_not_in = false;

    public:
        SubqueryUnnesting();

        virtual std::string ToString() override;

        virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input);

    private:

        void subquery_unnesting_single_query(SQLTreeNodePointer arg_input);

        void subquery_unnesting_handling_node(SQLTreeNodePointer arg_input);

        void HandlingColumnNode(SQLTreeNodePointer arg_input);

        void HandlingGroupNode(SQLTreeNodePointer arg_input);

        void HandlingSortNode(SQLTreeNodePointer arg_input);

        void HandlingFilterNode(SQLTreeNodePointer arg_input);

        void HandlingBinaryJoinNode(SQLTreeNodePointer arg_input);

        void HandlingTableNode(SQLTreeNodePointer arg_input);

        bool ProcessExists(CommonBiaodashi *arg_expr_p, bool arg_not_exists);

        bool CheckExistsSubquery(SelectStructure *arg_ss, bool arg_not_exists);

        bool ProcessExistsTreeNode(SQLTreeNodePointer arg_input, bool arg_not_exists);

        bool AnalyzeFilterInsideExists(BiaodashiPointer arg_expr);

        bool CheckFilterItemInsideExists(BiaodashiPointer arg_expr);

        bool MakeSureExprNoOuterColumn(BiaodashiPointer arg_expr);

        bool ProcessIn(CommonBiaodashi *arg_expr_p, bool arg_not_in, std::vector<BiaodashiPointer> &join_other_conditions);

        bool CheckInSubquery(SelectStructure *arg_ss, bool arg_not_in, const std::vector<BiaodashiPointer> &expr_outside_list, std::vector<BiaodashiPointer> &join_other_conditions);

        bool ProcessInTreeNode(SQLTreeNodePointer arg_input);

        bool ProcessFilterBasicExpr(BiaodashiPointer arg_expr, std::vector<BiaodashiPointer> &join_other_conditions);

        bool ObtainAllColumns_ProcessingExpr(BiaodashiPointer arg_expr);

        bool ChangeTableNamesInsideSubquery();
    };

} // namespace aries

#endif
