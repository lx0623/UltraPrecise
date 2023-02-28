#include "TwoPhaseJoin.h"
#include "frontend/BiaodashiAuxProcessor.h"
#include "frontend/SelectStructure.h"

namespace aries {

TwoPhaseJoin::TwoPhaseJoin() {
}


std::string TwoPhaseJoin::ToString() {
    return std::string(
            "TwoPhaseJoin -- Converting a complext join into a simple single-column eqaul-join and a following filter");
}


SQLTreeNodePointer TwoPhaseJoin::OptimizeTree(SQLTreeNodePointer arg_input) {

    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->twophasejoin_single_query(arg_input);
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


void TwoPhaseJoin::twophasejoin_single_query(SQLTreeNodePointer arg_input) {

    this->twophasejoin_handling_node(arg_input);
}

void TwoPhaseJoin::twophasejoin_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    switch (arg_input->GetType()) {

        case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Column_NODE:
            if (1 > 0) {
                this->twophasejoin_handling_node(arg_input->GetTheChild());
            }
            break;

        case SQLTreeNodeType::Group_NODE:
            if (1 > 0) {
                this->twophasejoin_handling_node(arg_input->GetTheChild());
            }
            break;

        case SQLTreeNodeType::Sort_NODE:
            if (1 > 0) {
                this->twophasejoin_handling_node(arg_input->GetTheChild());
            }
            break;


            //case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Filter_NODE:
            if (1 > 0) {
                this->twophasejoin_handling_node(arg_input->GetTheChild());
            }
            break;

        case SQLTreeNodeType::SetOp_NODE:
            if (1 > 0) {
                this->twophasejoin_handling_node(arg_input->GetLeftChild());
                this->twophasejoin_handling_node(arg_input->GetRightChild());
            }
            break;

        case SQLTreeNodeType::Table_NODE:
            if (1 > 0) {
                //da jiang you!
            }
            break;


        case SQLTreeNodeType::BinaryJoin_NODE:
            if (1 > 0) {
                this->HandlingBinaryJoinNode(arg_input);
            }
            break;
        case SQLTreeNodeType::InnerJoin_NODE:
        {
            handleInnerJoinNode( arg_input );
            break;
        }

        default:
            ARIES_ASSERT(0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()));

    }
}

static bool
vector_contains_all( const std::vector< BasicRelPointer >& all, const std::vector< BasicRelPointer >& sub )
{
    for ( const auto& s : sub )
    {
        bool found = false;
        for ( const auto& a : all )
        {
            if ( s == a && s->GetMyOutputName() == a->GetMyOutputName() )
            {
                found = true;
                break;
            }
        }

        if ( !found )
        {
            return false;
        }
    }

    return true;
}

void TwoPhaseJoin::handleInnerJoinNode( SQLTreeNodePointer node )
{
    auto parent = node->GetParent();


    SQLTreeNodeBuilder builder( node->GetMyQuery() );
    BiaodashiAuxProcessor processor;

    std::map< int, bool > handled_map;

    auto& all_conditions = node->GetInnerJoinConditions();


    SQLTreeNodePointer new_join_node = node->GetChildByIndex( 0 );
    std::vector< BasicRelPointer > left_node_tables = new_join_node->GetInvolvedTableList();

    while ( true )
    {
        bool found_match = false;
        SQLTreeNodePointer unhandle_node;
        int unhandled_index = -1;
        for ( size_t i = 1; i < node->GetChildCount(); i++ )
        {
            if ( handled_map[ i ] )
            {
                continue;
            }

            auto child = node->GetChildByIndex( i );

            unhandle_node = child;
            unhandled_index = i;
            auto leaf_tables = child->GetInvolvedTableList();

            std::vector< BiaodashiPointer > matched_conditions;
            for ( const auto& condition : all_conditions )
            {
                auto expr = std::dynamic_pointer_cast< CommonBiaodashi >( condition );
                if ( expr->IsTrueConstant() )
                {
                    continue;
                }

                expr->ObtainReferenceTableInfo();
                auto left_condition = std::dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 0 ) );
                auto right_condition = std::dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 1 ) );

                auto left_tables = left_condition->GetInvolvedTableList();
                auto right_tables = right_condition->GetInvolvedTableList();

                if ( vector_contains_all( leaf_tables, left_tables ) && vector_contains_all( left_node_tables, right_tables ) )
                {
                    matched_conditions.emplace_back( condition );
                }
                else if ( vector_contains_all( leaf_tables, right_tables ) && vector_contains_all( left_node_tables, left_tables ) )
                {
                    matched_conditions.emplace_back( condition );
                }
            }

            if ( !matched_conditions.empty() )
            {
                auto join_node = builder.makeTreeNode_BinaryJoin( JoinType::InnerJoin, processor.make_biaodashi_from_and_list( matched_conditions ) );
                SQLTreeNode::AddTreeNodeChild( join_node, new_join_node );
                SQLTreeNode::AddTreeNodeChild( join_node, child );
                new_join_node = join_node;
                left_node_tables.insert( left_node_tables.end(), leaf_tables.cbegin(), leaf_tables.cend() );
                handled_map[ i ] = true;
                found_match = true;
                break;
            }
        }

        if ( unhandled_index == -1 )
        {
            break;
        }

        if ( !found_match )
        {
            auto true_condition = std::make_shared<CommonBiaodashi>( BiaodashiType::Zhenjia, true );
            true_condition->SetValueType( BiaodashiValueType::BOOL );
            auto join_node = builder.makeTreeNode_BinaryJoin( JoinType::InnerJoin, true_condition );
            SQLTreeNode::AddTreeNodeChild( join_node, new_join_node );
            SQLTreeNode::AddTreeNodeChild( join_node, unhandle_node );
            auto tables = unhandle_node->GetInvolvedTableList();
            left_node_tables.insert( left_node_tables.end(), tables.cbegin(), tables.cend() );
            new_join_node = join_node;
            handled_map[ unhandled_index ] = true;
        }
    }

    parent->CompletelyResetAChild( node, new_join_node );
    new_join_node->SetParent( parent );
    HandlingBinaryJoinNode( new_join_node );
}


void TwoPhaseJoin::HandlingBinaryJoinNode(SQLTreeNodePointer arg_input) {
    this->twophasejoin_handling_node(arg_input->GetLeftChild());
    this->twophasejoin_handling_node(arg_input->GetRightChild());

    //std::cout<< "\n\n TwoPhaseJoin::HandlingBinaryJoinNode -> Nw we will process a join node \n\n";

    BiaodashiPointer join_condition = arg_input->GetJoinCondition();

    BiaodashiAuxProcessor expr_processor;
    std::vector<BiaodashiPointer> expr_and_list = expr_processor.generate_and_list(join_condition);

    ARIES_ASSERT(!expr_and_list.empty(), "expression list should not be empty");

    if (expr_and_list.size() == 1) {
        //检查Join条件等式表达式左右两端关联的表是否分别在Join的左右表中,如不同则交换表达式左右两端
        if (!this->CheckEqualJoinCondition(expr_and_list[0], arg_input)) {
            arg_input->SetJoinCondition( nullptr );
            arg_input->SetJoinOtherCondition( expr_and_list[0] );
        }
        else
        {
            arg_input->ReCheckJoinConditionConstraintType();
        }
        return; // the common case!!!
    }

    bool we_found_the_one = false;
    BiaodashiPointer the_equal_join_expr = nullptr;
    std::vector<BiaodashiPointer> other_expr_list;

    for (size_t i = 0; i < expr_and_list.size(); i++) {
        BiaodashiPointer aexpr = expr_and_list[i];

        if (we_found_the_one == false && this->CheckEqualJoinCondition(aexpr, arg_input) == true) {
            the_equal_join_expr = aexpr;
//      std::cout << "\n" << ((CommonBiaodashi *)((the_equal_join_expr).get()))->ToString() << "\t--www\n";
            we_found_the_one = true;
        } else {
            other_expr_list.push_back(aexpr);
        }

    }

    if (we_found_the_one == false) {
        // if ( arg_input->GetJoinType() != JoinType::InnerJoin) {
        //     ARIES_EXCEPTION_SIMPLE(ER_TOO_BIG_ROWSIZE, "need equal-condition for join");
        // }
        arg_input->SetJoinCondition( nullptr );
        if (!other_expr_list.empty()) {
            arg_input->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_expr_list ) );
        }
        return; //do nothing
    }

    //else
    //we now have the_qual_join_expr, and other_expr_list!

    /**
     * Inner join 的其他条件也放入 join node 中处理（可能会使用 hash join）
     */
    //if (arg_input->GetJoinType() != JoinType::InnerJoin) 
    {
        /*
        * for others (except inner join) join:
        * if a join condition is "(T1.a = T2.a and (xxx))", then we keep
        * the first equal-join expr as the join condition, and we assign
        * all rest exprs as other condition!
        */
        arg_input->SetJoinCondition( the_equal_join_expr );
        if (!other_expr_list.empty()) {
            arg_input->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_expr_list ) );
        }
        return;
    }

}

bool TwoPhaseJoin::CheckEqualJoinCondition(BiaodashiPointer arg_expr, SQLTreeNodePointer arg_input) {

    CommonBiaodashi *expr_p = (std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr)).get();

//  std::cout << "TwoPhaseJoin::CheckEqualJoinCondition" << "\t" << expr_p->ToString() << "\n";

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

        /**
         * TODO: 需要统一调用 ObtainReferenceTableInfo()
         */
        lc_p->ObtainReferenceTableInfo();
        rc_p->ObtainReferenceTableInfo();

        std::vector<BasicRelPointer> refLeft = lc_p->GetInvolvedTableList();
        std::vector<BasicRelPointer> refRight = rc_p->GetInvolvedTableList();
        if ( refLeft.empty() || refRight.empty()) {
            return false;
        }
        //如果表达式左右两边包含了同一张表,则不能成为join的等值条件
        for ( size_t i = 0; i < refLeft.size(); ++i ) {
            for ( size_t j = 0; j < refRight.size(); ++j ) {
                if (refLeft[i]->GetDb() == refRight[j]->GetDb() &&
                    refLeft[i]->GetMyOutputName() == refRight[j]->GetMyOutputName()) {
                    return false;
                }
            }
        }

        std::vector< BasicRelPointer > leftTables = arg_input->GetLeftChild()->GetInvolvedTableList();
        std::vector<BasicRelPointer> rightTables = arg_input->GetRightChild()->GetInvolvedTableList();
        bool needSwap = false;
        //检查表达式左右refTable是否都在join的左右表中
        bool rightCondition = CheckTablesInTables( refLeft, leftTables ) && CheckTablesInTables( refRight, rightTables );
        if (!rightCondition) {
            rightCondition = CheckTablesInTables( refLeft, rightTables ) && CheckTablesInTables( refRight, leftTables );
            if (rightCondition) {
                needSwap = true;
            }
        }
        if (!rightCondition) {
            return false;
        }
        //如果join条件等式左右两端的列不是分别在左表和右表，则调换左右等式两边的表达式
        if (needSwap) {
            expr_p->ResetChildByIndex( 0, rc );
            expr_p->ResetChildByIndex( 1, lc );
        }
    }
    return true;
}

bool TwoPhaseJoin::CheckTablesInTables( std::vector<BasicRelPointer> ref, std::vector<BasicRelPointer> tables ) {
    for (size_t i = 0; i < ref.size(); i++) {
        bool found = false;
        for ( size_t t = 0; t < tables.size(); t++ ) {
            if (ref[i]->GetDb() == tables[t]->GetDb() && ref[i]->GetMyOutputName() == tables[t]->GetMyOutputName()) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}


}
