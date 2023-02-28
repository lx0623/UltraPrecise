#include "SubqueryUnnesting.h"

#include "frontend/SelectStructure.h"

#include "frontend/BiaodashiAuxProcessor.h"

#include "frontend/SelectPartStructure.h"

namespace aries
{

    SubqueryUnnesting::SubqueryUnnesting()
    {
    }

    std::string SubqueryUnnesting::ToString()
    {
        return std::string("Subquery Unnesting");
    }

    SQLTreeNodePointer SubqueryUnnesting::OptimizeTree(SQLTreeNodePointer arg_input)
    {
        QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
                                                   arg_input->GetMyQuery())
                                                   ->GetQueryContext();

        /*do for my subqueries*/
        for (size_t si = 0; si < my_query_context->subquery_context_array.size(); si++)
        {
            QueryContextPointer a_subquery_context = my_query_context->subquery_context_array[si];

            //this->subquery_unnesting_single_query(std::dynamic_pointer_cast<SelectStructure>(a_subquery_context->GetSelectStructure())->GetQueryPlanTree());
            this->OptimizeTree(
                std::dynamic_pointer_cast<SelectStructure>(a_subquery_context->GetSelectStructure())->GetQueryPlanTree());
        }

        /*do for myself*/
        /*if I am a from clause actually, then done already!*/
        if (my_query_context->type != QueryContextType::FromSubQuery)
        {
            this->subquery_unnesting_single_query(arg_input);
        }

        return arg_input;
    }

    void SubqueryUnnesting::subquery_unnesting_single_query(SQLTreeNodePointer arg_input)
    {
        this->subquery_unnesting_handling_node(arg_input);
    }

    void SubqueryUnnesting::subquery_unnesting_handling_node(SQLTreeNodePointer arg_input)
    {
        if (arg_input == nullptr)
            return;

        switch (arg_input->GetType())
        {

        case SQLTreeNodeType::Limit_NODE:
            subquery_unnesting_handling_node(arg_input->GetTheChild());
            break;

        case SQLTreeNodeType::Column_NODE:
            if (1 > 0)
            {
                this->HandlingColumnNode(arg_input);
            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0)
            {
                this->HandlingGroupNode(arg_input);
            }
            break;

        case SQLTreeNodeType::Sort_NODE:
            if (1 > 0)
            {
                this->HandlingSortNode(arg_input);
            }
            break;

            //case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0)
            {
                this->HandlingFilterNode(arg_input);
            }
            break;

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0)
            {
                this->HandlingTableNode(arg_input);
            }
            break;

        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0)
            {
                this->HandlingBinaryJoinNode(arg_input);
            }
            break;

        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0)
            {
                this->subquery_unnesting_handling_node(arg_input->GetLeftChild());
                this->subquery_unnesting_handling_node(arg_input->GetRightChild());
            }
            break;
        case SQLTreeNodeType::InnerJoin_NODE:
        {
            for (size_t i = 0; i < arg_input->GetChildCount(); i++)
            {
                subquery_unnesting_handling_node(arg_input->GetChildByIndex(i));
            }
            break;
        }

        default:
        {
            ARIES_ASSERT(0, "UnSupported node type: " + std::to_string((int)arg_input->GetType()));
            break;
        }
        }
    }

    void SubqueryUnnesting::HandlingColumnNode(SQLTreeNodePointer arg_input)
    {
        //std::cout << "SubqueryUnnesting::HandlingColumnNode" << "---Da Jiang You!\n";
        this->subquery_unnesting_handling_node(arg_input->GetTheChild());
    }

    void SubqueryUnnesting::HandlingGroupNode(SQLTreeNodePointer arg_input)
    {
        //std::cout << "SubqueryUnnesting::HandlingGroupNode" << "---Da Jiang You!\n";
        this->subquery_unnesting_handling_node(arg_input->GetTheChild());
    }

    void SubqueryUnnesting::HandlingSortNode(SQLTreeNodePointer arg_input)
    {
        //std::cout << "SubqueryUnnesting::HandlingSortNode" << "---Da Jiang You!\n";
        this->subquery_unnesting_handling_node(arg_input->GetTheChild());
    }

    void SubqueryUnnesting::HandlingFilterNode(SQLTreeNodePointer arg_input)
    {
        //std::cout << "SubqueryUnnesting::HandlingFilterNode" << "---Da Jiang You!\n";

        BiaodashiPointer filter_condition = arg_input->GetFilterStructure();

        /*for debug*/
        //	//std::cout << "\t\t FilterCondition: is !!!" << ((CommonBiaodashi* )(filter_condition.get()))->ToString() << "\n";

        BiaodashiAuxProcessor expr_processor;
        std::vector<BiaodashiPointer> expr_and_list = expr_processor.generate_and_list(filter_condition);

        BiaodashiPointer the_removable_expr = nullptr;
        std::vector<BiaodashiPointer> other_expr_list;
        std::vector<BiaodashiPointer> join_other_conditions;
        bool just_copy = false;

        for (size_t i = 0; i < expr_and_list.size(); i++)
        {
            BiaodashiPointer aexpr = expr_and_list[i];

            if (just_copy == true)
            {
                other_expr_list.push_back(aexpr);
            }
            else
            {

                bool subquery_removable = false;
                subquery_removable = this->ProcessFilterBasicExpr(aexpr, join_other_conditions);

                if (subquery_removable)
                {
                    the_removable_expr = aexpr;
                    just_copy = true;
                    //std::cout << "\t\t A removable subquery:  FilterCondition: is !!!" << ((CommonBiaodashi* )(aexpr.get()))->ToString() << "\n";
                }
                else
                {
                    other_expr_list.push_back(aexpr);
                }
            }
        }

        //now we can make a dicision
        if (the_removable_expr == nullptr)
        {
            //do nothing
        }
        else
        {

            //Z(0): Yes, we can do unnesting! so we need renaming tables in the subquery

            this->ChangeTableNamesInsideSubquery();

            //A: Get the node for the outer query

            SQLTreeNodePointer outer_node_for_join = nullptr;

            SQLTreeNodePointer old_child = arg_input->GetTheChild();
            if (other_expr_list.size() > 0)
            {
                BiaodashiPointer new_filter_expr = expr_processor.make_biaodashi_from_and_list(other_expr_list);
                //		arg_input->SetFilterStructure(new_filter_expr);

                SQLTreeNodeBuilder tree_builder(arg_input->GetMyQuery());
                SQLTreeNodePointer new_filter_node = tree_builder.makeTreeNode_Filter(new_filter_expr);
                SQLTreeNode::SetTreeNodeChild(new_filter_node, old_child);

                arg_input->CompletelyResetAChild(old_child, new_filter_node);
                new_filter_node->SetParent(arg_input);

                outer_node_for_join = new_filter_node;
            }
            else
            {
                outer_node_for_join = old_child;
            }

            //B: get the node for the subquery part
            SQLTreeNodePointer inner_node_for_join = nullptr;
            inner_node_for_join = this->subquery_remain_node;

            //C: get the join type
            JoinType the_join_type;
            if (this->exist_or_in == 1)
            {
                the_join_type = this->is_a_not_exist ? JoinType::AntiJoin : JoinType::SemiJoin;
            }
            else if (this->exist_or_in == -1)
            {
                the_join_type = this->is_a_not_in ? JoinType::AntiJoin : JoinType::SemiJoin;
            }
            else
            {
                ARIES_ASSERT( 0, "unexpected value: " + std::to_string( this->exist_or_in ) );
            }

            //D: get the join condition
            BiaodashiPointer the_join_condition = nullptr;

            if (this->exist_or_in == 1)
            {
                /*
             *we need to process outercolumn's level!
             *An outercolumn's absolute_level is different than the level of the query where the column occurs,
             *so that it cannot be processed when handling an expr which is required to provide all the reference columns!
             *
             *But after subquery unnesting, an outercolumn should be processed. So we have to let the absolute_level be equal to the query level.
             *We can either lift up the whole subquery leve, or just downgrade the column level.
             *We choose the later solution because it is easier to do -- although sounds not reasonable.
             */

                for (size_t oci = 0; oci < this->outer_columns.size(); oci++)
                {
                    ColumnShellPointer a_csp = this->outer_columns[oci];
                    int old_level = a_csp->GetAbsoluteLevel();
                    a_csp->SetAbsoluteLevel(old_level + 1); //crazy!
                }

                the_join_condition = expr_processor.make_biaodashi_from_and_list(this->correlated_items);
            }
            else
            {
                the_join_condition = expr_processor.make_biaodashi_compare_equal(this->the_expr_outside,
                                                                                 this->the_expr_inside);
            }

            //E: generate the new join node
            auto the_parent = arg_input->GetParent();

            SQLTreeNodeBuilder tree_builder(arg_input->GetMyQuery());
            BiaodashiPointer other_join_condition;
            if( !join_other_conditions.empty() )
            {
                other_join_condition = expr_processor.make_biaodashi_from_and_list( join_other_conditions );
            }
            SQLTreeNodePointer new_join_node = tree_builder.makeTreeNode_BinaryJoin(the_join_type, the_join_condition, other_join_condition, this->is_a_not_in);

            SQLTreeNode::AddTreeNodeChild(new_join_node, outer_node_for_join);
            SQLTreeNode::AddTreeNodeChild(new_join_node, inner_node_for_join);

            the_parent->CompletelyResetAChild(arg_input, new_join_node);
            new_join_node->SetParent(the_parent);

            //G: finanlly set the mask for this subquery -- it is gone!
            this->the_subquery_ss->SetDoIStillExist(false);
        }

        this->subquery_unnesting_handling_node(arg_input->GetTheChild());
    }

    void SubqueryUnnesting::HandlingBinaryJoinNode(SQLTreeNodePointer arg_input)
    {
        //std::cout << "SubqueryUnnesting::HandlingBinaryJoinNode" << "---Da Jiang You!\n";

        ////std::cout << "left-->\n";
        this->subquery_unnesting_handling_node(arg_input->GetLeftChild());

        ////std::cout << "right->\n";
        this->subquery_unnesting_handling_node(arg_input->GetRightChild());
    }

    void SubqueryUnnesting::HandlingTableNode(SQLTreeNodePointer arg_input)
    {
        //std::cout << "SubqueryUnnesting::HandlingTableNode" << "---Da Jiang You!\n";
    }

    bool SubqueryUnnesting::ProcessIn(CommonBiaodashi *arg_expr_p, bool arg_not_in, std::vector<BiaodashiPointer> &join_other_conditions)
    {
        ////std::cout << "SubqueryUnnesting::ProcessIN" << "\n";
        this->exist_or_in = -1; //means IN!

        this->is_a_not_in = arg_not_in;
        this->the_expr_inside = nullptr;
        this->the_expr_outside = nullptr;
        this->subquery_remain_node = nullptr;
        this->the_subquery_ss = NULL;

        this->all_columns_in_subquery.clear();
        this->all_table_nodes_in_subquery.clear();
        this->all_column_nodes_in_subquery.clear();

        //start

        // out_column in (subquery)

        if (arg_expr_p->GetChildrenCount() != 2)
        {
            return false;
        }

        BiaodashiPointer child_0_bp = arg_expr_p->GetChildByIndex(0);
        CommonBiaodashi *child_0_p = (std::dynamic_pointer_cast<CommonBiaodashi>(child_0_bp)).get();

        BiaodashiPointer child_1_bp = arg_expr_p->GetChildByIndex(1);
        CommonBiaodashi *child_1_p = (std::dynamic_pointer_cast<CommonBiaodashi>(child_1_bp)).get();
        std::vector<BiaodashiPointer> the_expr_outside_list;
        //check the first one
        if (child_0_p->GetType() == BiaodashiType::Lie)
        {
            ColumnShellPointer csp = boost::get<ColumnShellPointer>(child_0_p->GetContent());

            if (csp->GetAbsoluteLevel() != child_0_p->GetExprContext()->GetQueryContext()->query_level)
            {
                /*outer column!*/
                return false;
            }
            else
            {
                //TODO: NULL DETECTION NEEDED!!!!!!!!!!!!!!
                the_expr_outside_list.push_back(child_0_bp);
            }
        }
        else if (child_0_p->GetType() == BiaodashiType::ExprList)
        {
            for (size_t i = 0; i < child_0_p->GetChildrenCount(); ++i)
            {
                auto child = std::dynamic_pointer_cast<CommonBiaodashi>(child_0_p->GetChildByIndex(i));
                if (child->GetType() == BiaodashiType::Lie)
                {
                    ColumnShellPointer csp = boost::get<ColumnShellPointer>(child->GetContent());

                    if (csp->GetAbsoluteLevel() != child->GetExprContext()->GetQueryContext()->query_level)
                    {
                        /*outer column!*/
                        return false;
                    }
                    else
                    {
                        //TODO: NULL DETECTION NEEDED!!!!!!!!!!!!!!
                        the_expr_outside_list.push_back(child);
                    }
                }
                else
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }

        //check the second one
        if (child_1_p->GetType() != BiaodashiType::Query)
        {
            return false;
        }

        AbstractQueryPointer aqp = boost::get<AbstractQueryPointer>(child_1_p->GetContent());

        SelectStructure *the_ss = (SelectStructure *)(aqp.get());
        this->the_subquery_ss = the_ss;

        std::vector<ColumnShellPointer> v_csp = the_ss->GetQueryContext()->outer_column_array;

        if (v_csp.size() != 0)
        {
            /*this is a correlated subquery*/
            return false;
        }

        return this->CheckInSubquery(the_ss, arg_not_in, the_expr_outside_list, join_other_conditions);
    }

    bool SubqueryUnnesting::CheckInSubquery(SelectStructure *arg_ss, bool arg_not_in, const std::vector<BiaodashiPointer> &expr_outside_list, std::vector<BiaodashiPointer> &join_other_conditions)
    {
        //std::cout << "SubqueryUnnesting::CheckInSubquery" << "\n";

        bool ret = true;

        SQLTreeNodePointer tree_node = arg_ss->GetQueryPlanTree();

        //std::cout << "\n\n" << tree_node->ToString(0) << "\n\n";

        if (tree_node->GetType() == SQLTreeNodeType::Column_NODE)
        {

            std::vector<BiaodashiPointer> vbp = arg_ss->GetAllSelectExprs();
            if(vbp.size() != expr_outside_list.size())
                ARIES_EXCEPTION( ER_OPERAND_COLUMNS, expr_outside_list.size() );
            std::vector<BiaodashiPointer> expr_inside_list;
            for (auto the_expr : vbp)
            {
                CommonBiaodashi *the_expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(the_expr)).get();

                if ( the_expr_p->IsLiteral() )
                {
                    return false;
                }

                if (the_expr_p->GetType() != BiaodashiType::Lie)
                {
                    if ( the_expr_p->IsAggFunction() || the_expr_p->ContainsAggFunction() )
                    {
                        ObtainAllColumns_ProcessingExpr( the_expr );
                        return false;
                    }
                    else
                    {
                        expr_inside_list.push_back(the_expr);
                        ObtainAllColumns_ProcessingExpr( the_expr );
                    }
                }
                else
                {
                    ColumnShellPointer csp = boost::get<ColumnShellPointer>(the_expr_p->GetContent());

                    if (csp->GetAbsoluteLevel() != the_expr_p->GetExprContext()->GetQueryContext()->query_level)
                    {
                        /*outer column!*/
                        ret = false;
                    }
                    else
                    {
                        //TODO: NULL DETECTION NEEDED!!!!!!!!!!!!!!
                        /*we got it!*/
                        expr_inside_list.push_back(the_expr);

                        //now we consider the issue of renaming
                        this->all_columns_in_subquery.push_back(csp);
                    }
                }
            }
            this->subquery_remain_node = tree_node->GetTheChild();
            ret = this->ProcessInTreeNode(tree_node->GetTheChild());
            assert(expr_inside_list.size() == expr_outside_list.size());
            assert(!expr_inside_list.empty());
            the_expr_inside = expr_inside_list[0];
            the_expr_outside = expr_outside_list[0];
            BiaodashiAuxProcessor expr_processor;
            for (size_t i = 1; i < expr_inside_list.size(); ++i)
            {
                join_other_conditions.push_back(
                    expr_processor.make_biaodashi_compare(expr_outside_list[i],
                                                          expr_inside_list[i],
                                                          ComparisonType::BuDengYu));
            }
        }
        else
        {
            // TODO: handle other exprs
            ARIES_ASSERT(0, "checkinsubquery, not column_node");
        }

        return ret;
    }

    bool SubqueryUnnesting::ProcessInTreeNode(SQLTreeNodePointer arg_input)
    {
        bool ret = true;

        switch (arg_input->GetType())
        {
        case SQLTreeNodeType::Limit_NODE:
            ret = ProcessInTreeNode(arg_input->GetTheChild());
            break;

        case SQLTreeNodeType::Column_NODE:
            if (1 > 0)
            {
                /*this is not the top column_node for the whole query because we begin from its child!*/

                //basically we treat it similar to a TableNode. We don't go deep into its tree.
                //std::cout << "\n\t\t\t\t we have a columnnode !\t\n";
                this->all_column_nodes_in_subquery.push_back(arg_input);
            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0)
            {

                //handling all exprs in the group node!
                GroupbyStructurePointer gsp = arg_input->GetMyGroupbyStructure();
                for (size_t i = 0; i < gsp->GetGroupbyExprCount(); i++)
                {
                    BiaodashiPointer bp = gsp->GetGroupbyExpr(i);
                    ret = this->ObtainAllColumns_ProcessingExpr(bp);
                    if (ret == false)
                    {
                        break;
                    }
                }

                if ( gsp->GetHavingExpr() )
                    ret = this->ObtainAllColumns_ProcessingExpr(gsp->GetHavingExpr());

                //if ok, we continue to do child!
                if (ret == true)
                {
                    this->ProcessInTreeNode(arg_input->GetTheChild());
                }
            }
            break;

        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0)
            {
                ret = false;
            }
            break;

        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0)
            {
                BiaodashiPointer the_filter_expr = arg_input->GetFilterStructure();
                ret = this->ObtainAllColumns_ProcessingExpr(the_filter_expr);

                if (ret == true)
                {
                    this->ProcessInTreeNode(arg_input->GetTheChild());
                }
            }
            break;

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0)
            {

                //		this->the_table_node_in_subquery = arg_input;
                this->all_table_nodes_in_subquery.push_back(arg_input);
            }
            break;

        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0)
            {
                BiaodashiPointer the_join_condition = arg_input->GetJoinCondition();
                ret = this->ObtainAllColumns_ProcessingExpr(the_join_condition);

                if (ret == true)
                {
                    this->ProcessInTreeNode(arg_input->GetLeftChild());
                    this->ProcessInTreeNode(arg_input->GetRightChild());
                }
            }
            break;
        case SQLTreeNodeType::InnerJoin_NODE:
        {
            BiaodashiAuxProcessor processor;
            if (!arg_input->GetInnerJoinConditions().empty())
            {
                auto condition = processor.make_biaodashi_from_and_list(arg_input->GetInnerJoinConditions());
                ObtainAllColumns_ProcessingExpr(condition);
            }

            for (size_t i = 0; i < arg_input->GetChildCount(); i++)
            {
                ProcessInTreeNode(arg_input->GetChildByIndex(i));
            }
            break;
        }

        default:
            ret = false;
            ARIES_ASSERT(0, "UnSupported node type: " + std::to_string((int)arg_input->GetType()));
        }

        return ret;
    }

    bool SubqueryUnnesting::ProcessExists(CommonBiaodashi *arg_expr_p, bool arg_not_exists)
    {
        ////std::cout << "SubqueryUnnesting::ProcessExists" << "\n";
        this->exist_or_in = 1; //means EXISTS!

        this->is_a_not_exist = arg_not_exists;
        this->correlated_items.clear();
        this->uncorrelated_items.clear();
        this->outer_columns.clear();
        this->subquery_remain_node = nullptr;
        this->the_subquery_ss = NULL;

        this->all_columns_in_subquery.clear();
        this->all_table_nodes_in_subquery.clear();
        this->all_column_nodes_in_subquery.clear();

        //start

        BiaodashiPointer bp = arg_expr_p->GetChildByIndex(0);

        CommonBiaodashi *cb_p = (std::dynamic_pointer_cast<CommonBiaodashi>(bp)).get();

        ////std::cout << "SubqueryUnnesting::ProcessExists: " << cb_p->ToString() << "\n";

        if (cb_p->GetType() != BiaodashiType::Query)
        {
            //it should be kuohao!
            ARIES_ASSERT(0, "exists should have a query child");
            return false;
        }

        AbstractQueryPointer aqp = boost::get<AbstractQueryPointer>(cb_p->GetContent());

        SelectStructure *the_ss = (SelectStructure *)(aqp.get());
        this->the_subquery_ss = the_ss;

        std::vector<ColumnShellPointer> v_csp = the_ss->GetQueryContext()->outer_column_array;

        if (v_csp.size() == 0)
        {
            /*this is an uncorrelated subquery*/
            return false;
        }

        return this->CheckExistsSubquery(the_ss, arg_not_exists);
    }

    bool SubqueryUnnesting::CheckExistsSubquery(SelectStructure *arg_ss, bool arg_not_exists)
    {
        //std::cout << "SubqueryUnnesting::CheckExistsSubquery" << "\n";

        SQLTreeNodePointer tree_node = arg_ss->GetQueryPlanTree();

        bool ret = this->ProcessExistsTreeNode(tree_node, arg_not_exists);

        return ret;
    }

    bool SubqueryUnnesting::ProcessExistsTreeNode(SQLTreeNodePointer arg_input, bool arg_not_exists)
    {
        bool ret = true;

        switch (arg_input->GetType())
        {
        case SQLTreeNodeType::Limit_NODE:
            ret = ProcessExistsTreeNode(arg_input->GetTheChild(), arg_not_exists);
            break;

        case SQLTreeNodeType::Column_NODE:
            if (1 > 0)
            {

                //std::cout << "SubqueryUnnesting::ProcessExistsTreeNode" << "\t" << "Column_NODE" << "\n";

                SQLTreeNodePointer cn = arg_input->GetTheChild();
                if (cn->GetType() != SQLTreeNodeType::Filter_NODE)
                {
                    ret = false;
                }
                else
                {
                    ret = this->ProcessExistsTreeNode(cn, arg_not_exists);
                }
            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0)
            {
                ret = false;
            }
            break;

        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0)
            {
                ret = false;
            }
            break;

        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0)
            {

                //std::cout << "SubqueryUnnesting::ProcessExistsTreeNode" << "\t" << "Filter_NODE" << "\n";

                SQLTreeNodePointer cn = arg_input->GetTheChild();

                if (cn->GetType() != SQLTreeNodeType::Table_NODE)
                {
                    ret = false;
                }
                else
                {

                    ret = this->ProcessExistsTreeNode(cn, arg_not_exists);

                    BiaodashiPointer the_filter_expr = arg_input->GetFilterStructure();

                    /*we have to collect all columns for renaming!*/
                    ret = this->ObtainAllColumns_ProcessingExpr(the_filter_expr);

                    ret = this->AnalyzeFilterInsideExists(the_filter_expr);

                    if (ret == false)
                    {
                        //we cannot unnest this subquery
                    }
                    else
                    {
                        //we can do it.
                        //but

                        if (this->correlated_items.size() == 0)
                        {
                            /*we should at least have one correlated condition!*/
                            ret = false;
                        }
                        else
                        {
                            /*now we can really do it! what we do now is to build a query tree that contains the table node and a possible filter node*/

                            if (this->uncorrelated_items.size() == 0)
                            {
                                /*nothing left, so we just give the TableNode*/
                                this->subquery_remain_node = cn;
                            }
                            else
                            {
                                this->subquery_remain_node = arg_input;

                                //todo: this should be delayed to the apply phase after the check phase that make sure we can do the unnesting!
                                // I have to reset the filtercondition
                                BiaodashiAuxProcessor expr_processor;
                                BiaodashiPointer new_expr = expr_processor.make_biaodashi_from_and_list(
                                    this->uncorrelated_items);
                                arg_input->SetFilterStructure(new_expr);
                            }
                        }
                    }
                }
            }
            break;

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0)
            {

                //std::cout << "SubqueryUnnesting::ProcessExistsTreeNode" << "\t" << "Table_NODE" << "\n";

                //this->the_table_node_in_subquery = arg_input;
                this->all_table_nodes_in_subquery.push_back(arg_input);
            }
            break;

        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0)
            {
                ret = false;
            }
            break;

        default:
            ret = false;
            ARIES_ASSERT(0, "UnSupported node type: " + std::to_string((int)arg_input->GetType()));
        }

        return ret;
    }

    bool SubqueryUnnesting::AnalyzeFilterInsideExists(BiaodashiPointer arg_expr)
    {

        bool ret = true;

        BiaodashiAuxProcessor expr_processor;
        std::vector<BiaodashiPointer> expr_and_list = expr_processor.generate_and_list(arg_expr);

        for (size_t i = 0; i < expr_and_list.size(); i++)
        {
            BiaodashiPointer aexpr = expr_and_list[i];

            ret = this->CheckFilterItemInsideExists(aexpr);

            if (ret == false)
            {
                break;
            }
            else
            {
                //we have processed one item succesfully!
            }
        }

        return ret;
    }

    static void get_columns_in_expression(const BiaodashiPointer &expression, std::vector<ColumnShellPointer> &columns)
    {
        CommonBiaodashi *expr = (CommonBiaodashi *)(expression.get());
        if (expr->GetType() == BiaodashiType::Lie)
        {
            columns.emplace_back(boost::get<ColumnShellPointer>(expr->GetContent()));
        }
        else
        {
            for (size_t i = 0; i < expr->GetChildrenCount(); i++)
            {
                auto child = expr->GetChildByIndex(i);
                if (!child)
                {
                    continue;
                }

                get_columns_in_expression(child, columns);
            }
        }
    }

    bool SubqueryUnnesting::CheckFilterItemInsideExists(BiaodashiPointer arg_expr)
    {
        /*a correlated condition must be: out.column >=< in.column or one without out.columns at all!*/

        bool ret = true;

        CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

        //std::cout << "SubqueryUnnesting::CheckFilterItemInsideExists" << "\t" << expr_p->ToString() << "\n";

        bool correlated_condition = false;

        /*now find and process correlated conditions*/
        if (expr_p->GetType() == BiaodashiType::Bijiao)
        {
            BiaodashiPointer lc = expr_p->GetChildByIndex(0);
            BiaodashiPointer rc = expr_p->GetChildByIndex(1);
            auto query_level = expr_p->GetExprContext()->GetQueryContext()->query_level;

            std::vector<ColumnShellPointer> left_columns;
            std::vector<ColumnShellPointer> right_columns;

            get_columns_in_expression(lc, left_columns);
            get_columns_in_expression(rc, right_columns);

            if (!left_columns.empty() && !right_columns.empty())
            {
                bool l_out_col = true;
                bool r_out_col = true;

                for (const auto &column : left_columns)
                {
                    auto level = column->GetAbsoluteLevel();

                    if (level == query_level)
                    {
                        l_out_col = false;
                    }
                    else if (level != query_level - 1)
                    {
                        return false;
                    }
                }

                for (const auto &column : right_columns)
                {
                    auto level = column->GetAbsoluteLevel();

                    if (level == query_level)
                    {
                        r_out_col = false;
                    }
                    else if (level != query_level - 1)
                    {
                        return false;
                    }
                }

                //judge
                if (l_out_col && r_out_col)
                {
                    return false;
                }
                else if (l_out_col != r_out_col)
                {
                    correlated_condition = true;
                    /*tough work!*/

                    /*Now we get a legcally correlated condition!*/
                    this->correlated_items.push_back(arg_expr);

                    if (l_out_col)
                    {
                        outer_columns.insert(outer_columns.cend(), left_columns.cbegin(), left_columns.cend());
                    }
                    else
                    {
                        outer_columns.insert(outer_columns.cend(), right_columns.cbegin(), right_columns.cend());
                    }
                }
            }
        }

        /*now process uncorrelated conditions*/
        if (correlated_condition == false)
        {

            /*uncorrelated condition must not include any outer columns!*/

            if (this->MakeSureExprNoOuterColumn(arg_expr) == false)
            {
                return false;
            }
            else
            {
                // a legally local condition
                // todo

                this->uncorrelated_items.push_back(arg_expr);
            }
        }

        return ret;
    }

    bool SubqueryUnnesting::MakeSureExprNoOuterColumn(BiaodashiPointer arg_expr)
    {
        bool ret = true;

        CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

        switch (expr_p->GetType())
        {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Zhenjia:
        case BiaodashiType::Zifuchuan:

        case BiaodashiType::Star:
        case BiaodashiType::Hanshu:
        case BiaodashiType::Shuzu:
        case BiaodashiType::Yunsuan:
        case BiaodashiType::Bijiao:
        case BiaodashiType::Andor:
        case BiaodashiType::Qiufan:
        case BiaodashiType::Kuohao:
        case BiaodashiType::Cunzai:
        case BiaodashiType::Likeop:
        case BiaodashiType::Inop:
        case BiaodashiType::NotIn:
        case BiaodashiType::Between:
        case BiaodashiType::Case:
        case BiaodashiType::SQLFunc:
        case BiaodashiType::ExprList:
        case BiaodashiType::IsNull:
        case BiaodashiType::IsNotNull:
        case BiaodashiType::Null:
        case BiaodashiType::IfCondition:
        case BiaodashiType::IntervalExpression:
            if (1 > 0)
            {

                for (size_t ci = 0; ci < expr_p->GetChildrenCount(); ci++)
                {
                    bool child_ok = this->MakeSureExprNoOuterColumn(expr_p->GetChildByIndex(ci));
                    if (child_ok == false)
                    {
                        return false;
                    }
                }
            }
            break;

        case BiaodashiType::Lie:
            if (1 > 0)
            {
                ColumnShellPointer csp = boost::get<ColumnShellPointer>(expr_p->GetContent());

                if (csp->GetAbsoluteLevel() != expr_p->GetExprContext()->GetQueryContext()->query_level)
                {
                    /*outer column!*/
                    return false;
                }
            }
            break;

        case BiaodashiType::Query:
            if (1 > 0)
            {
                return false;
            }

        default:
            ARIES_ASSERT(0, "SubqueryUnnesting::MakeSureExprNoOuterColumn: unknown type: " +
                            std::to_string(int(expr_p->GetType())));
            return false;
        }

        return ret;
    }

    bool SubqueryUnnesting::ProcessFilterBasicExpr(BiaodashiPointer arg_expr, std::vector<BiaodashiPointer> &join_other_conditions)
    {

        bool can_we_process = false;

        CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

        //std::cout << expr_p->ToString()  << "\n";

        switch (expr_p->GetType())
        {

        case BiaodashiType::Inop:
            if (1 > 0)
            {
                can_we_process = ProcessIn(expr_p, false, join_other_conditions);
            }
            break;

        case BiaodashiType::NotIn:
            can_we_process = ProcessIn(expr_p, true, join_other_conditions);
            break;

        case BiaodashiType::Cunzai:
            if (1 > 0)
            {

                can_we_process = this->ProcessExists(expr_p, false);
            }
            break;

        case BiaodashiType::Qiufan:
            if (1 > 0)
            {
                CommonBiaodashi *real_expr_p = std::dynamic_pointer_cast<CommonBiaodashi>(
                                                   expr_p->GetChildByIndex(0))
                                                   .get();

                //std::cout << "\n\nQIUFAN: " << real_expr_p->ToString() << "\n";

                if (real_expr_p->GetType() == BiaodashiType::Cunzai)
                {
                    can_we_process = this->ProcessExists(real_expr_p, true);
                }
                else if (real_expr_p->GetType() == BiaodashiType::Inop)
                {
                    can_we_process = this->ProcessIn(real_expr_p, true, join_other_conditions);
                }
                else
                {
                    can_we_process = false;
                }
            }
            break;
            //
            //	case BiaodashiType::Yunsuan:
            //	    break;

        default:

            can_we_process = false;
        }

        return can_we_process;
    }

    bool SubqueryUnnesting::ObtainAllColumns_ProcessingExpr(BiaodashiPointer arg_expr)
    {
        //otbain all columns (ignore outercolumns) and then append to the global array

        CommonBiaodashi *the_expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

        //todo: we should try/catch this call!
        std::vector<ColumnShellPointer> ac = the_expr_p->GetAllReferencedColumns_NoQuery();

        this->all_columns_in_subquery.insert(
            this->all_columns_in_subquery.end(),
            ac.begin(),
            ac.end());

        return true;
    }

    bool SubqueryUnnesting::ChangeTableNamesInsideSubquery()
    {

        //std::vector<ColumnShellPointer> all_columns_in_subquery;
        //SQLTreeNodePointer the_table_node_in_subquery;

        std::string new_name_append = "19790609" + std::to_string(this->renaming_seq++);

        //table!

        for (size_t i = 0; i < this->all_table_nodes_in_subquery.size(); i++)
        {

            SQLTreeNodePointer stnp = this->all_table_nodes_in_subquery[i];

            std::string old_name = stnp->GetBasicRel()->GetMyOutputName();

            stnp->GetBasicRel()->ResetAlias(old_name + "[" + new_name_append + "]");
        }

        //columnnode

        for (size_t i = 0; i < this->all_column_nodes_in_subquery.size(); i++)
        {

            SQLTreeNodePointer stnp = this->all_column_nodes_in_subquery[i];

            std::string old_name = stnp->GetBasicRel()->GetMyOutputName();

            stnp->GetBasicRel()->ResetAlias(old_name + "[" + new_name_append + "]");
            //why a table node does not need do this?
            stnp->GetBasicRel()->GetRelationStructure()->SetName(old_name + "[" + new_name_append + "]");
        }

        //columns!
        for (size_t i = 0; i < this->all_columns_in_subquery.size(); i++)
        {
            ColumnShellPointer csp = this->all_columns_in_subquery[i];
            std::string old_name = csp->GetTableName();
            csp->SetTableName(old_name + "[" + new_name_append + "]");
        }

        return true;
    }

} // namespace aries
