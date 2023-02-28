#include "JoinSubqueryRemoving.h"

#include "frontend/SelectStructure.h"

#include "frontend/BiaodashiAuxProcessor.h"

namespace aries {

JoinSubqueryRemoving::JoinSubqueryRemoving() {
}


std::string JoinSubqueryRemoving::ToString() {
    return std::string("JoinSubqueryRemoving: unnesting agg-jon subqueries!");
}


SQLTreeNodePointer JoinSubqueryRemoving::OptimizeTree(SQLTreeNodePointer arg_input) {

    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->js_subquery_handling_single_query(arg_input);
    }

    /*do for my subqueries*/
    for (size_t si = 0; si < my_query_context->subquery_context_array.size(); si++) {
        QueryContextPointer a_subquery_context = my_query_context->subquery_context_array[si];

        //this->query_formation_single_query(std::dynamic_pointer_cast<SelectStructure>(a_subquery_context->GetSelectStructure())->GetQueryPlanTree());
        SelectStructure *the_ss = (SelectStructure *) (a_subquery_context->GetSelectStructure().get());
        if ( the_ss && the_ss->DoIStillExist() ) {
            this->OptimizeTree(the_ss->GetQueryPlanTree());
        }
    }

    return arg_input;


}


void JoinSubqueryRemoving::js_subquery_handling_single_query(SQLTreeNodePointer arg_input) {
    this->js_subquery_handling_node(arg_input);
}


void JoinSubqueryRemoving::js_subquery_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    switch (arg_input->GetType()) {

        case SQLTreeNodeType::Limit_NODE:
            js_subquery_handling_node(arg_input->GetTheChild());
            break;

        case SQLTreeNodeType::Column_NODE:
            if (1 > 0) {
                this->HandlingColumnNode(arg_input);
            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0) {
                this->HandlingGroupNode(arg_input);
            }
            break;

        case SQLTreeNodeType::Sort_NODE:
            if (1 > 0) {
                this->HandlingSortNode(arg_input);
            }
            break;


            //case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0) {
                this->HandlingFilterNode(arg_input);
            }
            break;

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0) {
                this->HandlingTableNode(arg_input);
            }
            break;


        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0) {
                this->HandlingBinaryJoinNode(arg_input);
            }
            break;

        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0) {
                this->js_subquery_handling_node(arg_input->GetLeftChild());
                this->js_subquery_handling_node(arg_input->GetRightChild());
            }
            break;

        case SQLTreeNodeType::InnerJoin_NODE:
            HandlingInnerJoinNode( arg_input );
            for( size_t i = 0; i < arg_input->GetChildCount(); ++i )
                this->js_subquery_handling_node( arg_input->GetChildByIndex( i ) );
            break;

        default:
        {
            ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()) );
            break;
        }
    }

}


void JoinSubqueryRemoving::HandlingColumnNode(SQLTreeNodePointer arg_input) {
    //std::cout << "JoinSubqueryRemoving::HandlingColumnNode" << "---Da Jiang You!\n";
    this->js_subquery_handling_node(arg_input->GetTheChild());
}


void JoinSubqueryRemoving::HandlingGroupNode(SQLTreeNodePointer arg_input) {
    //std::cout << "JoinSubqueryRemoving::HandlingGroupNode" << "---Da Jiang You!\n";
    this->js_subquery_handling_node(arg_input->GetTheChild());
}

void JoinSubqueryRemoving::HandlingSortNode(SQLTreeNodePointer arg_input) {
    //std::cout << "JoinSubqueryRemoving::HandlingSortNode" << "---Da Jiang You!\n";
    this->js_subquery_handling_node(arg_input->GetTheChild());
}

void JoinSubqueryRemoving::HandlingFilterNode( SQLTreeNodePointer arg_input )
{
    //std::cout << "JoinSubqueryRemoving::HandlingFilterNode" << "---Da Jiang You!\n";
    auto the_filter_node_condition = arg_input->GetFilterStructure();
    BiaodashiAuxProcessor expr_processor;
    auto expr_and_list = expr_processor.generate_and_list( the_filter_node_condition );

    BiaodashiPointer the_removable_expr = nullptr;
    std::vector< BiaodashiPointer > other_expr_list;

    bool just_copy = false;

    for( size_t i = 0; i < expr_and_list.size(); i++ )
    {
        BiaodashiPointer aexpr = expr_and_list[i];
        if( just_copy )
        {
            other_expr_list.push_back( aexpr );
        }
        else
        {
            bool subquery_removable = false;

            ResetMyself();
            subquery_removable = this->CheckBasicExpr( aexpr );

            if( subquery_removable )
            {
                the_removable_expr = aexpr;
                just_copy = true;
            }
            else
            {
                other_expr_list.push_back( aexpr );
            }
        }
    }

    if( the_removable_expr )
    {
        //ok we can do unnesting!
        this->the_removable_expr = the_removable_expr;

        if( !other_expr_list.empty() )
        {
            this->outer_query_node = arg_input;
            arg_input->SetFilterStructure( expr_processor.make_biaodashi_from_and_list( other_expr_list ) );
        }
        else
        {
            //before call ApplyOptimization(), we need remove the old filter node
            auto parent_p = arg_input->GetParent();
            parent_p->CompletelyResetAChild( arg_input, arg_input->GetTheChild() );
            arg_input->GetTheChild()->SetParent( parent_p );

            this->outer_query_node = arg_input->GetTheChild();
        }

        this->ApplyOptimization( arg_input );
    }
}

void JoinSubqueryRemoving::HandlingBinaryJoinNode(SQLTreeNodePointer arg_input) {

    //std::cout << "left-->\n";
    this->js_subquery_handling_node(arg_input->GetLeftChild());

    //std::cout << "right->\n";
    this->js_subquery_handling_node(arg_input->GetRightChild());
}

void JoinSubqueryRemoving::HandlingInnerJoinNode( SQLTreeNodePointer arg_input )
{
    auto& expr_and_list = arg_input->GetInnerJoinConditions();

    BiaodashiPointer the_removable_expr = nullptr;
    std::vector< BiaodashiPointer > other_expr_list;

    bool just_copy = false;

    for( size_t i = 0; i < expr_and_list.size(); i++ )
    {
        BiaodashiPointer aexpr = expr_and_list[i];

        if( just_copy )
        {
            other_expr_list.push_back( aexpr );
        }
        else
        {
            bool subquery_removable = false;

            ResetMyself();
            subquery_removable = this->CheckBasicExpr( aexpr );

            if( subquery_removable )
            {
                the_removable_expr = aexpr;
                just_copy = true;
            }
            else
            {
                other_expr_list.push_back( aexpr );
            }
        }
    }

    if( the_removable_expr )
    {
        //ok we can do unnesting!
        this->the_removable_expr = the_removable_expr;
        this->outer_query_node = arg_input;
        arg_input->GetInnerJoinConditions().clear();

        BiaodashiAuxProcessor expr_processor;
        if( !other_expr_list.empty() )
        {
            arg_input->GetInnerJoinConditions().assign( other_expr_list.cbegin(), other_expr_list.cend() );
            arg_input->SetJoinCondition( expr_processor.make_biaodashi_from_and_list( other_expr_list ) );
        }
        else
            arg_input->SetJoinCondition( expr_processor.make_biaodashi_boolean( true ) );

        this->ApplyOptimization( arg_input );
    }
}

void JoinSubqueryRemoving::ApplyOptimization(SQLTreeNodePointer arg_input) {
//	return;
    BiaodashiAuxProcessor expr_processor;
    SQLTreeNodeBuilder tree_builder(arg_input->GetMyQuery());

    this->sequence += 1;
    std::string agg_column_name = "agg_value_" + std::to_string(this->sequence);
    std::string subquery_table_name =
            std::string("joinsubquery_") + std::string("19790609_") + std::to_string(this->sequence);

    BasicRelPointer brp;


//	std::cout << "Let's do work!\n";


    //1. Inner
    //1.1: processing the filter node that contains the correlated expr
    if (1 > 0) {
        SQLTreeNodePointer tfn = this->the_filter_node;
//	    std::cout << tfn->ToString(1);

        SQLTreeNodePointer old_child = tfn->GetTheChild();

        if (this->uncorrelated_exprs.size() == 0) {
//		std::cout << "nothing left\n";
            /*we don't keep the filter*/
            tfn->GetParent()->CompletelyResetAChild_WithoutReCalculateInvolvedTableList(tfn, old_child);
            old_child->SetParent( tfn->GetParent() );
        } else {
            BiaodashiPointer new_filter_expr = expr_processor.make_biaodashi_from_and_list(uncorrelated_exprs);

            SQLTreeNodePointer new_filter_node = tree_builder.makeTreeNode_Filter(new_filter_expr);

            SQLTreeNode::SetTreeNodeChild( new_filter_node, old_child );
            tfn->GetParent()->CompletelyResetAChild_WithoutReCalculateInvolvedTableList(tfn, new_filter_node);
            new_filter_node->SetParent( tfn->GetParent() );

        }

    }
//	std::cout << "1.1: ok\n";

    //1.2 add the correspoinding column into groupby node
    if (1 > 0) {
        SQLTreeNodePointer tgn = this->the_group_node;

        GroupbyStructurePointer gbsp = tgn->GetMyGroupbyStructure();
        //before we can add the expr, we have to set a mark -- since it will be moved to group where context is different
        ((CommonBiaodashi *) ((this->needed_groupby_expr).get()))->GetExprContext()->not_orginal_group_expr = true;
        for ( const auto& expr : needed_groupby_exprs )
        {
            ( ( CommonBiaodashi* )( expr.get() ) )->GetExprContext()->not_orginal_group_expr = true;
            gbsp->AddGroupbyExpr( expr );
        }

        // gbsp->AddGroupbyExpr(this->needed_groupby_expr);

        AbstractQueryPointer aqp = tgn->GetMyQuery();
        SelectStructure *the_ss = (SelectStructure *) (aqp.get());



        the_ss->AddReferencedColumnIntoGroupbyPartArray();
    }
//	std::cout << "1.2: ok\n";

    //1.3 add the correspoinding column into column node
    if (1 > 0) {

        /*the existing expr has a new alias*/
        /*the whole subquery has a new alias*/


        SQLTreeNodePointer tcn = this->the_column_node;

        SelectPartStructurePointer spsp = tcn->GetMySelectPartStructure();
        AbstractQueryPointer aqp = tcn->GetMyQuery();
        SelectStructure *the_ss = (SelectStructure *) (aqp.get());



        RelationStructurePointer rsp = the_ss->GetRelationStructure();

//	    std::cout << "\n" << rsp->ToString() << "\n";

        rsp->SetName(subquery_table_name);
        rsp->ResetNameForTheOnlyColumn(agg_column_name);
        spsp->SetTheOnlyExprAlias(agg_column_name);
        the_ss->AddAliasForTheOnlyExpr(agg_column_name);

        //we are going to add the expr into select. So set a mark!
        ((CommonBiaodashi *) ((this->needed_groupby_expr).get()))->GetExprContext()->not_orginal_select_expr = true;
        for ( const auto& expr : needed_groupby_exprs )
        {
            CommonBiaodashi *rawpointer = ( CommonBiaodashi* )( expr.get() );
            rawpointer->GetExprContext()->not_orginal_group_expr = true;
            spsp->AddSelectExpr( expr, nullptr );
            spsp->AddCheckedExpr( expr );
            spsp->AddCheckedAlias(nullptr);
            the_ss->AddExtraExpr( expr );

            std::string column_name = rawpointer->GetName();
            BiaodashiValueType column_type = rawpointer->GetValueType();
            int column_length = -1;//todo
            bool column_allow_null = rawpointer->IsNullable();
            bool column_is_primary = false;
            ColumnStructurePointer csp = std::make_shared<ColumnStructure>(
                    column_name,
                    column_type,
                    column_length,
                    column_allow_null,
                    column_is_primary);

            if( rawpointer->GetType() == BiaodashiType::Lie )
            {
                ColumnShellPointer tmpCsp = boost::get<ColumnShellPointer>( rawpointer->GetContent() );
                csp->SetEncodeType( tmpCsp->GetColumnStructure()->GetEncodeType() );
                csp->SetEncodedIndexType( tmpCsp->GetColumnStructure()->GetEncodedIndexType() );
            }

            rsp->AddColumn(csp);
        }


        the_ss->AddReferencedColumnIntoSelectPartArray();


//	    std::cout << "\n" << rsp->ToString() << "\n";



//	    std::cout << "\n" << rsp->ToString() << "\n";
        //set a BasicRel for columnnode
        std::string brp_id;
        std::shared_ptr<std::string> brp_palias = std::make_shared<std::string>(subquery_table_name);
        AbstractQueryPointer brp_arp = this->the_subquery_aqp;

        brp = std::make_shared<BasicRel>(true, brp_id, brp_palias, brp_arp);
        brp->SetRelationStructure(rsp);
        brp->SetMyRelNode(tcn);

        // the_ss->AddFromTable( brp );
        ( ( SelectStructure* )( arg_input->GetMyQuery().get() ) )->AddFromTable( brp );

        tcn->SetIsTopNodeOfAFromSubquery(true);
        tcn->SetBasicRel(brp);

    }


//	std::cout << "1.3: ok\n";

    //2.2 create a new join node for the correlated subquery
    if (1 > 0) {
        SQLTreeNodePointer left_node = this->outer_query_node;
        SQLTreeNodePointer right_node = this->the_column_node;

        auto parent_p = left_node->GetParent();


        JoinType the_join_type = JoinType::InnerJoin;


        ExprContextPointer the_expr_context = ((CommonBiaodashi *) ((this->outer_used_expr).get()))->GetExprContext();

        int cs_absolute_level = the_expr_context->GetQueryContext()->query_level;


        std::vector< BiaodashiPointer > join_exprs;
        for ( size_t i = 0; i < outer_embedded_exprs.size(); i++ )
        {
            ColumnShellPointer csp_column = expr_processor.make_column_shell(brp, i + 1, cs_absolute_level);
            BiaodashiPointer subquery_expr_column = expr_processor.make_biaodashi_lie(csp_column);
            ((CommonBiaodashi *) (subquery_expr_column.get()))->SetExprContext(the_expr_context);
            //outer_embedded_expr is an outer column for the subquery. Now it comes out so that we have to make it become a normal column!
            auto col = boost::get< ColumnShellPointer >( ( ( CommonBiaodashi* )( outer_embedded_exprs[ i ].get() ) )->GetContent() );
            col->SetAbsoluteLevel( col->GetAbsoluteLevel() + 1 );
            // this->outer_embedded_column->SetAbsoluteLevel(this->outer_embedded_column->GetAbsoluteLevel() + 1);
            BiaodashiPointer the_correlated_join_expr = expr_processor.make_biaodashi_compare(outer_embedded_exprs[ i ],
                                                                                            subquery_expr_column,
                                                                                            ComparisonType::DengYu);
            ((CommonBiaodashi *) (the_correlated_join_expr.get()))->SetExprContext(the_expr_context);
            join_exprs.emplace_back( the_correlated_join_expr );
        }
        // //the first equal join
        // ColumnShellPointer csp_column = expr_processor.make_column_shell(brp, 1, cs_absolute_level);
        // BiaodashiPointer subquery_expr_column = expr_processor.make_biaodashi_lie(csp_column);
        // ((CommonBiaodashi *) (subquery_expr_column.get()))->SetExprContext(the_expr_context);
        // //outer_embedded_expr is an outer column for the subquery. Now it comes out so that we have to make it become a normal column!
        // this->outer_embedded_column->SetAbsoluteLevel(this->outer_embedded_column->GetAbsoluteLevel() + 1);
        // BiaodashiPointer the_correlated_join_expr = expr_processor.make_biaodashi_compare(this->outer_embedded_expr,
        //                                                                                   subquery_expr_column,
        //                                                                                   ComparisonType::DengYu);
        // ((CommonBiaodashi *) (the_correlated_join_expr.get()))->SetExprContext(the_expr_context);


        //the second comparion join for agg
        ColumnShellPointer csp_agg = expr_processor.make_column_shell(brp, 0, cs_absolute_level);
        BiaodashiPointer subquery_expr_agg = expr_processor.make_biaodashi_lie(csp_agg);
        ((CommonBiaodashi *) (subquery_expr_agg.get()))->SetExprContext(the_expr_context);
        ( ( CommonBiaodashi* ) ( subquery_expr_agg.get() ) )->SetIsNullable( csp_agg->GetColumnStructure()->IsNullable() );
        int comp_value = boost::get<int>(((CommonBiaodashi *) ((this->the_removable_expr).get()))->GetContent());
        BiaodashiPointer the_agg_join_expr = expr_processor.make_biaodashi_compare(this->outer_used_expr,
                                                                                   subquery_expr_agg,
                                                                                   static_cast<ComparisonType>(comp_value));
        ((CommonBiaodashi *) (the_agg_join_expr.get()))->SetExprContext(the_expr_context);

        //the final join condition
        std::vector<BiaodashiPointer> vbp;
        // for ( cosnt auto& expr : the_correlated_join_expr)
        vbp.assign( join_exprs.cbegin(), join_exprs.cend() );
        vbp.push_back(the_agg_join_expr);
        BiaodashiPointer the_final_join_condition = expr_processor.make_biaodashi_from_and_list(vbp);


        //make the join node
        SQLTreeNodePointer new_join_node = tree_builder.makeTreeNode_BinaryJoin(the_join_type,
                                                                                the_final_join_condition);
        SQLTreeNode::AddTreeNodeChild( new_join_node, left_node );
        SQLTreeNode::AddTreeNodeChild( new_join_node, right_node );

        parent_p->CompletelyResetAChild(left_node, new_join_node);
        new_join_node->SetParent( parent_p );
    }

//	std::cout << "2.2: ok\n";

    //G: finanlly set the mask for this subquery -- it is gone!
    this->the_subquery_ss->SetDoIStillExist(false);

}

void JoinSubqueryRemoving::HandlingTableNode(SQLTreeNodePointer arg_input) {
    //std::cout << "JoinSubqueryRemoving::HandlingTableNode" << "---Da Jiang You!\n";
}


bool JoinSubqueryRemoving::CheckBasicExpr(BiaodashiPointer arg_expr) {

    bool ret = false;

    CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

    //std::cout << "\t" << expr_p->ToString() << "\n";


    if (expr_p->GetType() == BiaodashiType::Bijiao) {

        BiaodashiPointer lc = expr_p->GetChildByIndex(0);
        BiaodashiPointer rc = expr_p->GetChildByIndex(1);

        CommonBiaodashi *lc_p = (CommonBiaodashi *) (lc.get());
        CommonBiaodashi *rc_p = (CommonBiaodashi *) (rc.get());

        if (!(lc_p->GetType() == BiaodashiType::Lie && rc_p->GetType() == BiaodashiType::Query)) {
            return false;
        }

        //left
        ColumnShellPointer l_csp = boost::get<ColumnShellPointer>(lc_p->GetContent());
        if (l_csp->GetAbsoluteLevel() != lc_p->GetExprContext()->GetQueryContext()->query_level) {
            /*this is an outer column FOR ME!*/
            return false;
        } else {
            this->outer_used_expr = lc;
        }

        //right
        AbstractQueryPointer aqp = boost::get<AbstractQueryPointer>(rc_p->GetContent());
        this->the_subquery_aqp = aqp;

        SelectStructure *the_ss = (SelectStructure *) (aqp.get());

        std::vector<ColumnShellPointer> v_csp = the_ss->GetQueryContext()->outer_column_array;

        if (v_csp.size() == 0) {
            return false;
        }

        ret = this->CheckSubquery(the_ss);
    }

    return ret;
}


bool JoinSubqueryRemoving::CheckSubquery(SelectStructure *arg_ss) {
    this->the_subquery_ss = arg_ss;

    SQLTreeNodePointer tree_node = arg_ss->GetQueryPlanTree();

    bool ret = this->ProcessSubqueryNode(tree_node);

    return ret;
}

bool JoinSubqueryRemoving::ProcessSubqueryNode(SQLTreeNodePointer arg_input) {
    bool ret = true;


    switch (arg_input->GetType()) {

        case SQLTreeNodeType::Column_NODE:
            if (1 > 0) {

                if (this->the_top_column_node == false) {
                    this->the_top_column_node = true;
                    this->the_column_node = arg_input;

                    SQLTreeNodePointer my_child = arg_input->GetTheChild();
                    if (my_child->GetType() != SQLTreeNodeType::Group_NODE) {
                        return false;
                    }

                    if (arg_input->GetMySelectPartStructure()->GetSelectItemCount() != 1) {
                        return false;
                    }


                    /*todo: we need to check whether the node has outer columns!*/


                    ret = this->ProcessSubqueryNode(arg_input->GetTheChild());
                    if (ret == false) {
                        return false;
                    }
                } else {
                    /*we have reached a subqury inside the join subquery*/
                    /*do nothing*/
                    /*todo: we should check the subquery to make sure it won't contain any outer columns!*/
                    return false;
                }


            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0) {
                if (this->see_group_node == false) {
                    this->see_group_node = true;
                    this->the_group_node = arg_input;
                    /*to know whether this group node is created. If so, it is right! */

                    if (arg_input->GetACreatedGroupNode() == false) {
                        return false;
                    } else {

                        ret = this->ProcessSubqueryNode(arg_input->GetTheChild());
                        if (ret == false) {
                            return false;
                        }

                    }
                } else {
                    return false;
                }

            }
            break;

        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0) {
                ret = false;
            }
            break;


        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0) {
                BiaodashiPointer filter_condition = arg_input->GetFilterStructure();

                BiaodashiAuxProcessor expr_processor;
                std::vector<BiaodashiPointer> expr_and_list = expr_processor.generate_and_list(filter_condition);

                for (size_t i = 0; i < expr_and_list.size(); i++) {
                    BiaodashiPointer aexpr = expr_and_list[i];

                    ret = this->CheckFilterItemInsideSubquery(aexpr);
                    if (ret == false) {
                        return false;
                    }
                }

                // if (this->correlated_expr == nullptr) {
                if ( correlated_exprs.empty() )
                {
                    /*this node is irrelevant.*/
                    this->uncorrelated_exprs.clear();
                } else {
                    /*now this filter node is the one!*/
                    if (this->the_filter_node != nullptr && the_filter_node != arg_input ) {
                        // TODO: test more than one correlated exprs
                        LOG( INFO ) << "we now process only the case of one correlated expr";
                        return false;
                    } else {
                        this->the_filter_node = arg_input;
                    }
                }


                if (this->ProcessSubqueryNode(arg_input->GetTheChild()) == false) {
                    return false;
                }

            }
            break;


        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0) {
                //what we need to do is make sure the join condition has no outer columns
                BiaodashiPointer join_condition = arg_input->GetJoinCondition();
                bool noc = this->MakeSureExprNoOuterColumn(join_condition);
                if (noc == false) {
                    return false;
                }

                if (this->ProcessSubqueryNode(arg_input->GetLeftChild()) == false) {
                    return false;
                }
                if (this->ProcessSubqueryNode(arg_input->GetRightChild()) == false) {
                    return false;
                }

            }

            break;

        case SQLTreeNodeType::InnerJoin_NODE:
        {
            BiaodashiAuxProcessor processor;
            BiaodashiPointer join_condition = processor.make_biaodashi_from_and_list( arg_input->GetInnerJoinConditions() );
            bool noc = this->MakeSureExprNoOuterColumn( join_condition );
            if ( !noc ) {
                return false;
            }

            for ( size_t i = 0; i < arg_input->GetChildCount(); i++ )
            {
                if ( !this->ProcessSubqueryNode( arg_input->GetChildByIndex( i ) ) ) {
                    return false;
                }
            }

            break;
        }

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0) {
                //Da Jiang You!

            }
            break;
        case SQLTreeNodeType::Limit_NODE:
            ret = false;
            break;

        default:
        {
            ret = false;
            ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()) );
        }
    }


    return ret;
}


bool JoinSubqueryRemoving::CheckFilterItemInsideSubquery(BiaodashiPointer arg_expr) {
    CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

    // correlated_expr = nullptr;

    if (expr_p->GetType() == BiaodashiType::Bijiao) {

        BiaodashiPointer lc = expr_p->GetChildByIndex(0);
        BiaodashiPointer rc = expr_p->GetChildByIndex(1);

        CommonBiaodashi *lc_p = (CommonBiaodashi *) (lc.get());
        CommonBiaodashi *rc_p = (CommonBiaodashi *) (rc.get());


        if (lc_p->GetType() == BiaodashiType::Lie && rc_p->GetType() == BiaodashiType::Lie) {
            ColumnShellPointer l_csp = boost::get<ColumnShellPointer>(lc_p->GetContent());
            ColumnShellPointer r_csp = boost::get<ColumnShellPointer>(rc_p->GetContent());

            /*if only one is outer column, then we get a correlated item */

            bool left_outer = (l_csp->GetAbsoluteLevel() != expr_p->GetExprContext()->GetQueryContext()->query_level);
            if (left_outer) {
                if (abs(l_csp->GetAbsoluteLevel() - expr_p->GetExprContext()->GetQueryContext()->query_level) > 1) {
                    return false;
                }
            }

            bool right_outer = (r_csp->GetAbsoluteLevel() != expr_p->GetExprContext()->GetQueryContext()->query_level);
            if (right_outer) {
                if (abs(r_csp->GetAbsoluteLevel() - expr_p->GetExprContext()->GetQueryContext()->query_level) > 1) {
                    return false;
                }
            }


            if (left_outer && right_outer) {
                return false;
            } else if ((!left_outer) && (!right_outer)) {
                if (this->the_filter_node == nullptr) {
                    this->uncorrelated_exprs.push_back(arg_expr);
                }
            } else {
                /*one outer, one local*/
                // if (this->correlated_expr != nullptr) {
                //     TroubleHandler::showMessage("we now process only the case of one correlated expr");
                //     return false;
                // } else {
                //     this->correlated_expr = arg_expr;
                // }
                correlated_exprs.emplace_back( arg_expr );

                if (left_outer) {
                    this->outer_embedded_expr = lc;
                    outer_embedded_exprs.emplace_back( lc );
                    this->outer_embedded_column = l_csp;
                    this->needed_groupby_expr = rc;
                    needed_groupby_exprs.emplace_back( rc );
                } else {
                    this->outer_embedded_expr = rc;
                    outer_embedded_exprs.emplace_back( rc );
                    this->outer_embedded_column = r_csp;
                    this->needed_groupby_expr = lc;
                    needed_groupby_exprs.emplace_back( lc );
                }

            }

        } else {
            /*call make sure!*/
            bool noc = this->MakeSureExprNoOuterColumn(arg_expr);
            if (noc == true) {
                if (this->the_filter_node == nullptr) {
                    this->uncorrelated_exprs.push_back(arg_expr);
                }
            } else {
                return false;
            }
        }


    } else {

        bool noc = this->MakeSureExprNoOuterColumn(arg_expr);
        if (noc == true) {
            if (this->the_filter_node == nullptr) {
                this->uncorrelated_exprs.push_back(arg_expr);
            }

        } else {
            return false;
        }

    }


    return true;
}

bool JoinSubqueryRemoving::MakeSureExprNoOuterColumn(BiaodashiPointer arg_expr) {
    bool ret = true;

    CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

    switch (expr_p->GetType()) {
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
            if (1 > 0) {

                for (size_t ci = 0; ci < expr_p->GetChildrenCount(); ci++) {
                    bool child_ok = this->MakeSureExprNoOuterColumn(expr_p->GetChildByIndex(ci));
                    if (child_ok == false) {
                        return false;
                    }
                }

            }
            break;

        case BiaodashiType::Lie:
            if (1 > 0) {
                ColumnShellPointer csp = boost::get<ColumnShellPointer>(expr_p->GetContent());

                if (csp->GetAbsoluteLevel() != expr_p->GetExprContext()->GetQueryContext()->query_level) {
                    /*outer column!*/
                    return false;
                }

            }
            break;

        case BiaodashiType::Query:
            if (1 > 0) {
                return false;
            }

        default:
        {
            ARIES_ASSERT( 0, "SubqueryUnnesting::MakeSureExprNoOuterColumn: unknown type: " +
                                            std::to_string(int(expr_p->GetType())) );
            return false;
        }
    }


    return ret;
}

void JoinSubqueryRemoving::ResetMyself() {
    this->the_subquery_aqp = nullptr;
    this->the_subquery_ss = NULL;
    this->the_removable_expr = nullptr;
    this->outer_used_expr = nullptr;
    this->outer_embedded_expr = nullptr;
    this->outer_embedded_column = nullptr;
    this->needed_groupby_expr = nullptr;
    this->the_top_column_node = false;
    this->see_group_node = false;
    this->correlated_expr = nullptr;
    correlated_exprs.clear();
    needed_groupby_exprs.clear();
    this->uncorrelated_exprs.clear();
    this->the_column_node = nullptr;
    this->the_group_node = nullptr;
    this->the_filter_node = nullptr;
    this->outer_query_node = nullptr;
}

}
