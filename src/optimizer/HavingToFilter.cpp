#include <glog/logging.h>

#include "HavingToFilter.h"
#include "frontend/BiaodashiAuxProcessor.h"
#include "frontend/SelectStructure.h"

namespace aries {

HavingToFilter::HavingToFilter() {
}

std::string HavingToFilter::ToString() {
    return std::string("HavingToFilter -- using a filter node to execute having in group");
}

SQLTreeNodePointer HavingToFilter::OptimizeTree(SQLTreeNodePointer arg_input) {

    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->htf_single_query(arg_input);
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


void HavingToFilter::htf_single_query(SQLTreeNodePointer arg_input) {

    this->htf_handling_node(arg_input);
}

void HavingToFilter::htf_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    switch (arg_input->GetType()) {

        case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Column_NODE:
            if (1 > 0) {
                this->htf_handling_node(arg_input->GetTheChild());
            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0) {
                this->HandlingGroupNode(arg_input);
                this->htf_handling_node(arg_input->GetTheChild());
            }
            break;

        case SQLTreeNodeType::Sort_NODE:
            if (1 > 0) {
                this->htf_handling_node(arg_input->GetTheChild());
            }
            break;


            //case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0) {
                this->htf_handling_node(arg_input->GetTheChild());
            }
            break;

        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0) {
                this->htf_handling_node(arg_input->GetLeftChild());
                this->htf_handling_node(arg_input->GetRightChild());
            }
            break;

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0) {
                //da jiang you!
            }
            break;


        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0) {
                this->htf_handling_node(arg_input->GetLeftChild());
                this->htf_handling_node(arg_input->GetRightChild());
            }
            break;

        default:
        {
            ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()) );
            break;
        }

    }
}

void HavingToFilter::HandlingGroupNode(SQLTreeNodePointer arg_input) {

    GroupbyStructurePointer gbsp = arg_input->GetMyGroupbyStructure();
    BiaodashiPointer the_having_expr = gbsp->GetHavingExpr();

    if (the_having_expr == nullptr) {
        return;
    }


    //now we have a having. We move it to a separate filternode.

    if (this->ProcessHavingExpr(arg_input, the_having_expr) == false) {
        return;
    }

    //now we can re-shape the tree

    this->ConvertTheHavingExpr(the_having_expr);

    //1. modify the group node.
    gbsp->DeleteHavingExpr();
    gbsp->SetAdditionalExprsForSelect(this->needed_agg_exprs);
    gbsp->SetAggExprsInHaving(this->all_agg_exprs_in_having);
    gbsp->SetAggExprsLocationMap(this->expr_to_location);
    gbsp->SetAllPlaceHolderExprs(this->all_placeholder_exprs);
    LOG(INFO) << "1.1 ok";

    //2. create a new filter node

    auto parent_node = arg_input->GetParent();
    assert(parent_node != NULL);

    BiaodashiAuxProcessor expr_processor;
    SQLTreeNodeBuilder tree_builder(arg_input->GetMyQuery());

    BiaodashiPointer new_filter_expr = the_having_expr;
    SQLTreeNodePointer new_filter_node = tree_builder.makeTreeNode_Filter(new_filter_expr);
    SQLTreeNode::SetTreeNodeChild( new_filter_node, arg_input );

    new_filter_node->SetGroupHavingMode(true);
    new_filter_node->SetGroupOutputCount(this->original_select_count);


    parent_node->CompletelyResetAChild(arg_input, new_filter_node);
    new_filter_node->SetParent( parent_node );

    LOG(INFO) << "1.2 ok";

}


bool HavingToFilter::ProcessHavingExpr(SQLTreeNodePointer arg_input, BiaodashiPointer arg_expr) {
    this->see_error = false;
    this->needed_agg_exprs.clear();
    this->expr_to_location.clear();
    this->original_select_count = -1;
    this->all_agg_exprs_in_having.clear();
    this->all_placeholder_exprs.clear();


    std::vector<BiaodashiPointer> vbp = this->GetAggExprs(arg_expr);
    this->all_agg_exprs_in_having = vbp;
    if (this->see_error) {
        return false;
    }

    //debug
    for (size_t di = 0; di < vbp.size(); di++) {
        LOG(INFO) << di << "\t" << ((CommonBiaodashi *) ((vbp[di]).get()))->ToString();
    }


    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());
    int existing_expr_count = the_ss->GetAllExprCount();
    this->original_select_count = existing_expr_count;

    for (size_t i = 0; i < vbp.size(); i++) {
        //it could be one of select_exprs or not.
        int agg_location;

        int tmp_location = the_ss->LocateExprInSelectList(vbp[i]);

        if (tmp_location == -1) {
            //no found
            //BiaodashiAuxProcessor expr_processor;
            //BiaodashiPointer expr2 = expr_processor.shallow_copy_biaodashi(vbp[i]);
            this->needed_agg_exprs.push_back(vbp[i]);
            existing_expr_count += 1;

            agg_location = (existing_expr_count - 1);
        } else {
            agg_location = tmp_location;
        }

        this->expr_to_location[vbp[i]] = agg_location;

        LOG(INFO) << i << "\t" << tmp_location << "\t" << agg_location;
    }
    LOG(INFO) << "done";

    return true;
}


std::vector<BiaodashiPointer> HavingToFilter::GetAggExprs(BiaodashiPointer arg_expr) {
    std::vector<BiaodashiPointer> ret_vbp;

    CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

    switch (expr_p->GetType()) {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Zhenjia:
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Star:
        case BiaodashiType::Distinct:
        case BiaodashiType::Decimal:
            break;

        case BiaodashiType::SQLFunc:
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
        case BiaodashiType::ExprList:
        case BiaodashiType::IsNull:
        case BiaodashiType::IsNotNull:
        case BiaodashiType::Null:
        case BiaodashiType::IfCondition:
        case BiaodashiType::IntervalExpression:
            if (1 > 0) {

                if (expr_p->GetType() == BiaodashiType::SQLFunc) {
                    SQLFunctionPointer sfp = boost::get<SQLFunctionPointer>(expr_p->GetContent());

                    if (sfp->GetIsAggFunc() == true) {
                        ret_vbp.push_back(arg_expr);
                        break;
                    }

                }


                for (size_t ci = 0; ci < expr_p->GetChildrenCount(); ci++) {
                    std::vector<BiaodashiPointer> ci_vbp = this->GetAggExprs(expr_p->GetChildByIndex(ci));
                    if (this->see_error) {
                        return ret_vbp;
                    } else {
                        ret_vbp.insert(
                                ret_vbp.end(),
                                ci_vbp.begin(),
                                ci_vbp.end()
                        );
                    }
                }

            }
            break;

        case BiaodashiType::Lie:
            if (1 > 0) {
                ColumnShellPointer csp = boost::get<ColumnShellPointer>(expr_p->GetContent());
                
                auto real = csp->GetExpr4Alias();
                if (real) {
                    return GetAggExprs(real);
                }
                
                ret_vbp.emplace_back(arg_expr);
                if (csp->GetAbsoluteLevel() != expr_p->GetExprContext()->GetQueryContext()->query_level) {
                    this->see_error = true;
                }
            }
            break;

        case BiaodashiType::Query:
            if (1 > 0) {
                AbstractQueryPointer aqp = boost::get<AbstractQueryPointer>(expr_p->GetContent());

                SelectStructure *the_ss = (SelectStructure *) (aqp.get());

                std::vector<ColumnShellPointer> v_csp = the_ss->GetQueryContext()->outer_column_array;

                if (v_csp.size() != 0) {
                    /*this is a correlated subquery*/
                    this->see_error = true;
                }

            }

        default:
        {
            this->see_error = true;
            ARIES_ASSERT( 0, "HavingToFilter::GetAggExprs: unknown type: " +
                              std::to_string(int(expr_p->GetType())) );
        }

    }


    return ret_vbp;

}


void HavingToFilter::CreatePlaceHolderExprs(BiaodashiPointer arg_expr) {

    CommonBiaodashi *expr_p = (CommonBiaodashi *) (arg_expr.get());

    BiaodashiAuxProcessor expr_processor;

    for (size_t i = 0; i < this->all_agg_exprs_in_having.size(); i++) {
        BiaodashiPointer loop_bp = this->all_agg_exprs_in_having[i];
        CommonBiaodashi *loop_expr_p = (CommonBiaodashi *) (loop_bp.get());

        LOG(INFO) << i << "\t" << loop_expr_p->ToString();

        int my_position = this->expr_to_location[loop_bp];

        std::string table_name = "PLACEHOLDER_TABLE_19790609";
        std::string column_name = std::to_string(my_position);
        BiaodashiValueType value_type = loop_expr_p->GetValueType();
        int level = expr_p->GetExprContext()->GetQueryContext()->query_level;
        ColumnShellPointer csp = expr_processor.make_column_shell_only_placeholder(
                table_name,
                column_name,
                value_type,
                level);

        BiaodashiPointer placeholder_bp = expr_processor.make_biaodashi_lie(csp);
        auto expression = (CommonBiaodashi *) (placeholder_bp.get());
        expression->SetExprContext(expr_p->GetExprContext());
        expression->SetLength(expr_p->GetLength());
        expression->SetIsNullable(loop_expr_p->IsNullable());

        this->all_placeholder_exprs.push_back(placeholder_bp);
    }

    LOG(INFO) << "we created " << this->all_agg_exprs_in_having.size() << " placeholders!";
}

int HavingToFilter::FindExprInArray(BiaodashiPointer arg_expr) {
    int ret = -1;
    for (size_t i = 0; i < this->all_agg_exprs_in_having.size(); i++) {
        if (arg_expr == this->all_agg_exprs_in_having[i]) {
            ret = i;
            break;
        }
    }
    return ret;
}

void HavingToFilter::ConvertTheHavingExpr(BiaodashiPointer arg_expr) {
    this->CreatePlaceHolderExprs(arg_expr);

    this->ExecuteReplacement(arg_expr);
}

void HavingToFilter::ExecuteReplacement(BiaodashiPointer arg_expr) {
    CommonBiaodashi *expr_p = (CommonBiaodashi *) (arg_expr.get());

    for (size_t i = 0; i < expr_p->GetChildrenCount(); i++) {
        BiaodashiPointer c_bp = expr_p->GetChildByIndex(i);
        int c_bp_loc = this->FindExprInArray(c_bp);
        if (c_bp_loc == -1) {
            this->ExecuteReplacement(c_bp);
        } else {
            BiaodashiPointer new_bp = this->all_placeholder_exprs[c_bp_loc];
            expr_p->ResetChildByIndex(i, new_bp);
        }
    }
}

}
