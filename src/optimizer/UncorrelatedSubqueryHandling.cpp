#include "UncorrelatedSubqueryHandling.h"

#include "frontend/SelectStructure.h"

#include "frontend/BiaodashiAuxProcessor.h"

namespace aries {

UncorrelatedSubqueryHandling::UncorrelatedSubqueryHandling() {
}


std::string UncorrelatedSubqueryHandling::ToString() {
    return std::string("UncorrelatedSubqueryHandling: Create init plans!");
}


SQLTreeNodePointer UncorrelatedSubqueryHandling::OptimizeTree(SQLTreeNodePointer arg_input) {

    //set the top query so that all init sub_queries
    if (this->the_query_ss == NULL) {
        this->the_query_ss = std::dynamic_pointer_cast<SelectStructure>(arg_input->GetMyQuery()).get();
    }

    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->uc_subquery_handling_single_query(arg_input);
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


void UncorrelatedSubqueryHandling::uc_subquery_handling_single_query(SQLTreeNodePointer arg_input) {
    this->uc_subquery_handling_node(arg_input);
}


void UncorrelatedSubqueryHandling::uc_subquery_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    switch (arg_input->GetType()) {
        case SQLTreeNodeType::Limit_NODE:
            uc_subquery_handling_node(arg_input->GetTheChild());
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
                this->uc_subquery_handling_node(arg_input->GetLeftChild());
                this->uc_subquery_handling_node(arg_input->GetRightChild());
            }
            break;
        case SQLTreeNodeType::InnerJoin_NODE:
        {
            for ( size_t i = 0; i < arg_input->GetChildCount(); i++ )
            {
                uc_subquery_handling_node( arg_input->GetChildByIndex( i ) );
            }
            break;
        }

        default:
        {
            ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string( (int) arg_input->GetType()) );
            break;
        }
    }

}


void UncorrelatedSubqueryHandling::HandlingColumnNode(SQLTreeNodePointer arg_input) {
    //std::cout << "UncorrelatedSubqueryHandling::HandlingColumnNode" << "---Da Jiang You!\n";
    auto select_part = the_query_ss->GetSelectPart();
    auto select_exprs = select_part->GetAllExprs();
    auto all_alias = select_part->GetALlAliasPointers();
    for (size_t i = 0; i < select_exprs.size(); i ++) {
        auto select_item = select_exprs[i];
        auto alias = all_alias[i];

        handle_expression(select_item);
    }
    this->uc_subquery_handling_node(arg_input->GetTheChild());
}

void UncorrelatedSubqueryHandling::handle_expression(BiaodashiPointer expression) {
    auto expr = (CommonBiaodashi*) (expression.get());

    if (expr->GetType() == BiaodashiType::Query) {
        return;
    }

    for (size_t i = 0; i < expr->GetChildrenCount(); i ++) {
        auto child = (CommonBiaodashi*) (expr->GetChildByIndex(i).get());
        if (child == nullptr) {
            continue;
        }
        switch (child->GetType()) {
        case BiaodashiType::Query: {
            auto aqp = boost::get<AbstractQueryPointer>(child->GetContent());

            auto the_ss = (SelectStructure *) (aqp.get());
            if( !the_ss->GetQueryContext()->HasOuterColumn() )
            {
                BiaodashiAuxProcessor expr_processor;
                BiaodashiPointer replace_expr = expr_processor.make_biaodashi_zifuchuan(
                        std::string("[INIT_QUERY_RESULT]"));

                switch (expr->GetType())
                {
                    case BiaodashiType::Inop:
                    case BiaodashiType::NotIn:
                    case BiaodashiType::Cunzai:
                    {
                        ((CommonBiaodashi*) replace_expr.get())->SetExpectBuffer(true);
                        break;
                    }
                    default:
                        break;
                }

                expr->ResetChildByIndex(i, replace_expr);

                the_query_ss->AddInitQueryAndExpr( std::dynamic_pointer_cast< SelectStructure >( aqp ), replace_expr);
            }
            break;
        }
        default:
            handle_expression(expr->GetChildByIndex(i));
            break;
        }
    }
}


void UncorrelatedSubqueryHandling::HandlingGroupNode(SQLTreeNodePointer arg_input) {
    //std::cout << "UncorrelatedSubqueryHandling::HandlingGroupNode" << "---Da Jiang You!\n";
    BiaodashiPointer having_expr = arg_input->GetMyGroupbyStructure()->GetHavingExpr();
    if (having_expr != nullptr) {
        BiaodashiAuxProcessor expr_processor;
        std::vector<BiaodashiPointer> expr_and_list = expr_processor.generate_and_list(having_expr);


        for (size_t i = 0; i < expr_and_list.size(); i++) {
            BiaodashiPointer aexpr = expr_and_list[i];

            this->ProcessFilterBasicExpr(aexpr);

        }
    }


    this->uc_subquery_handling_node(arg_input->GetTheChild());
}

void UncorrelatedSubqueryHandling::HandlingSortNode(SQLTreeNodePointer arg_input) {
    //std::cout << "UncorrelatedSubqueryHandling::HandlingSortNode" << "---Da Jiang You!\n";
    this->uc_subquery_handling_node(arg_input->GetTheChild());
}

void UncorrelatedSubqueryHandling::HandlingFilterNode(SQLTreeNodePointer arg_input) {
    //std::cout << "UncorrelatedSubqueryHandling::HandlingFilterNode" << "---Da Jiang You!\n";

    BiaodashiPointer filter_condition = arg_input->GetFilterStructure();

    /*for debug*/
    //std::cout << "\n\t\t FilterCondition: is !!!" << ((CommonBiaodashi* )(filter_condition.get()))->ToString() << "\n\n";

    BiaodashiAuxProcessor expr_processor;
    std::vector<BiaodashiPointer> expr_or_list;
    expr_processor.generate_or_list(filter_condition, expr_or_list);
    for (size_t i=0; i<expr_or_list.size(); ++i)
    {
        BiaodashiPointer or_expr = expr_or_list[i];
        std::vector<BiaodashiPointer> expr_and_list = expr_processor.generate_and_list(or_expr);
        for (auto and_expr : expr_and_list)
        {
            this->ProcessFilterBasicExpr(and_expr);
        }
    }

    this->uc_subquery_handling_node(arg_input->GetTheChild());
}

void UncorrelatedSubqueryHandling::HandlingBinaryJoinNode(SQLTreeNodePointer arg_input) {
    //std::cout << "UncorrelatedSubqueryHandling::HandlingBinaryJoinNode" << "---Da Jiang You!\n";

    //std::cout << "left-->\n";
    this->uc_subquery_handling_node(arg_input->GetLeftChild());

    //std::cout << "right->\n";
    this->uc_subquery_handling_node(arg_input->GetRightChild());

    handle_expression(arg_input->GetJoinCondition());
}

void UncorrelatedSubqueryHandling::HandlingTableNode(SQLTreeNodePointer arg_input) {
    //std::cout << "UncorrelatedSubqueryHandling::HandlingTableNode" << "---Da Jiang You!\n";
}


bool UncorrelatedSubqueryHandling::ProcessFilterBasicExpr(BiaodashiPointer arg_expr) {
    handle_expression(arg_expr);
    return true;
}


}
