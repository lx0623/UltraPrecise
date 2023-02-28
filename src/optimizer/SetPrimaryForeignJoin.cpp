#include "SetPrimaryForeignJoin.h"
#include "frontend/SelectStructure.h"

namespace aries {
SetPrimaryForeignJoin::SetPrimaryForeignJoin() {
}


std::string SetPrimaryForeignJoin::ToString() {
    return std::string("SetPrimaryForeignJoin -- determine each sort whether a primary/foreign join");
}


SQLTreeNodePointer SetPrimaryForeignJoin::OptimizeTree(SQLTreeNodePointer arg_input) {

    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->set_primary_foreign_join_single_query(arg_input);
    }

    /*do for my subqueries*/
    for (size_t si = 0; si < my_query_context->subquery_context_array.size(); si++) {
        QueryContextPointer a_subquery_context = my_query_context->subquery_context_array[si];

        //this->predicate_pushdown_single_query(std::dynamic_pointer_cast<SelectStructure>(a_subquery_context->GetSelectStructure())->GetQueryPlanTree());
        SelectStructure *the_ss = (SelectStructure *) (a_subquery_context->GetSelectStructure().get());
        if ( the_ss && the_ss->DoIStillExist() ) {
            this->OptimizeTree(the_ss->GetQueryPlanTree());
        }

    }

    return arg_input;
}


void SetPrimaryForeignJoin::set_primary_foreign_join_single_query(SQLTreeNodePointer arg_input) {
    ////LOG(INFO) << "------------------------------------------------\n";
    this->see_unsafe = false;
    this->set_primary_foreign_join_handling_node(arg_input);
}

void SetPrimaryForeignJoin::set_primary_foreign_join_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    ////LOG(INFO) << "\nNow we handle node: " << arg_input->ToString(0) << "\n\n";

    switch (arg_input->GetType()) {
        case SQLTreeNodeType::Limit_NODE:
            if (1 > 0) {
                this->set_primary_foreign_join_handling_node(arg_input->GetTheChild());
                this->see_unsafe = true;

            }
            break;
        case SQLTreeNodeType::Column_NODE:
        case SQLTreeNodeType::Group_NODE:
        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::Filter_NODE:
            /*Da Jiang You!*/
            this->set_primary_foreign_join_handling_node(arg_input->GetTheChild());
            break;

        case SQLTreeNodeType::SetOp_NODE:
            this->set_primary_foreign_join_handling_node(arg_input->GetLeftChild());
            this->set_primary_foreign_join_handling_node(arg_input->GetRightChild());
            break;

        case SQLTreeNodeType::Table_NODE:
            /*ending here*/
            break;


        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0) {
                this->set_primary_foreign_join_handling_node(arg_input->GetLeftChild());
                this->set_primary_foreign_join_handling_node(arg_input->GetRightChild());

                bool ret = this->process_join_node(arg_input);
                if (ret == false) {
                    this->see_unsafe = true;
                }

            }
            break;
        case SQLTreeNodeType::InnerJoin_NODE:
        {
            for ( size_t i = 0; i < arg_input->GetChildCount(); i++ )
            {
                set_primary_foreign_join_handling_node( arg_input->GetChildByIndex( i ) );
            }
            break;
        }

        default:
        {
            ARIES_ASSERT(0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()));
        }
    }

}


bool SetPrimaryForeignJoin::process_join_node(SQLTreeNodePointer arg_input) {

    if (arg_input->GetJoinType() != JoinType::InnerJoin) {
        return false;
    }


    BiaodashiPointer join_condition = arg_input->GetJoinCondition();

    if (join_condition == nullptr) {
        return false;
    }

    CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(join_condition)).get();


    if (expr_p->GetType() != BiaodashiType::Bijiao) {
        return false;
    } else {

        /*we only care equal!*/
        int op = boost::get<int>(expr_p->GetContent());
        if (op != (int) (ComparisonType::DengYu)) {
            return false;
        }

        //ok let's check seriously!
        BiaodashiPointer lc = expr_p->GetChildByIndex(0);
        BiaodashiPointer rc = expr_p->GetChildByIndex(1);

        CommonBiaodashi *lc_p = (CommonBiaodashi *) (lc.get());
        CommonBiaodashi *rc_p = (CommonBiaodashi *) (rc.get());


        ColumnShellPointer l_csp = nullptr;
        ColumnShellPointer r_csp = nullptr;

        if (lc_p->GetType() == BiaodashiType::Lie) {
            l_csp = boost::get<ColumnShellPointer>(lc_p->GetContent());
//		LOG(INFO)<< "l_csp\t" << l_csp->ToString() << "\n";
        } else {
            return false;
        }


        if (rc_p->GetType() == BiaodashiType::Lie) {
            r_csp = boost::get<ColumnShellPointer>(rc_p->GetContent());
//		LOG(INFO)<< "r_csp\t" << r_csp->ToString() << "\n";		
        } else {
            return false;
        }

//	    LOG(INFO) << "we reach here 1" << "\n";
        //are they outer columns?

//	    LOG(INFO) << l_csp->GetAbsoluteLevel() << "\t" << r_csp->GetAbsoluteLevel() << "\t" << expr_p->GetExprContext()->GetQueryContext()->query_level << "\n"; 
        if (
                l_csp->GetAbsoluteLevel() != lc_p->GetExprContext()->GetQueryContext()->query_level
                ||
                r_csp->GetAbsoluteLevel() != rc_p->GetExprContext()->GetQueryContext()->query_level
                ) {
            // we don't care outer columns!
            return false;
        }


        ColumnStructurePointer l_column_structure_p = l_csp->GetColumnStructure();
        ColumnStructurePointer r_column_structure_p = r_csp->GetColumnStructure();

        if (l_column_structure_p == nullptr || r_column_structure_p == nullptr) {
            LOG( WARNING ) << "if(l_column_structure_p == nullptr || r_column_structure_p == nullptr)";
            return false;
        }

        int result_form = 0;
        std::string primary_table_name;

        if (
                l_column_structure_p->GetIsPrimary() == true
                && r_column_structure_p->GetIsFk() == true
                && r_column_structure_p->GetFkColumn() == l_column_structure_p
                ) {
            result_form = 1;
            primary_table_name = l_csp->GetTableName();
        } else if (
                r_column_structure_p->GetIsPrimary() == true
                && l_column_structure_p->GetIsFk() == true
                && l_column_structure_p->GetFkColumn() == r_column_structure_p
                ) {
            result_form = -1;
            primary_table_name = r_csp->GetTableName();
        } else {
            return false;
        }

        if (result_form != 0) {

            //for second time to use this rule
            if (arg_input->GetPrimaryForeignJoinForm() != 0) {
                //this must have been set due to first time usage
                //we do nothing
                return true;

            }

            if (this->see_unsafe == true) {
                return false;
            }

            //LOG(INFO) << "we got a primary/foreign join!\n";
            arg_input->SetPrimaryForeignJoinForm(result_form);

            //we need to get another hint to know if the primary table is intact!
            //begin
            std::vector<BasicRelPointer> left_involved_table_list = arg_input->GetLeftChild()->GetInvolvedTableList();
            std::vector<BasicRelPointer> right_involved_table_list = arg_input->GetRightChild()->GetInvolvedTableList();
            int primary_table_position = 0;

            if (left_involved_table_list.size() == 1) {
                BasicRelPointer brp = left_involved_table_list[0];
                std::string rel_name = brp->GetMyOutputName();
                LOG(INFO) << "brp->GetMyOutputName() : primary_table_name = [" << rel_name << " : "
                          << primary_table_name << "]\n";
                if (rel_name == primary_table_name) {
                    primary_table_position = -1; // on left
                }
            }

            if (right_involved_table_list.size() == 1) {
                BasicRelPointer brp = right_involved_table_list[0];
                std::string rel_name = brp->GetMyOutputName();
                LOG(INFO) << "brp->GetMyOutputName() : primary_table_name = [" << rel_name << " : "
                          << primary_table_name << "]\n";
                if (rel_name == primary_table_name) {
                    if (primary_table_position == -1) {
                        LOG( WARNING ) << "primary_table_position == -1";
                        return false;
                    } else {
                        primary_table_position = 1; // on right
                    }
                }
            }

            bool primary_table_intact = false;

            if (primary_table_position == -1) {
                if (arg_input->GetLeftChild()->GetType() == SQLTreeNodeType::Table_NODE) {
                    primary_table_intact = true;
                }

            } else if (primary_table_position == 1) {
                if (arg_input->GetRightChild()->GetType() == SQLTreeNodeType::Table_NODE) {
                    primary_table_intact = true;
                }
            }

            arg_input->SetPrimaryTableIntact(primary_table_intact);
            //end

            return true;
        }


    }

    return false;
}


}
