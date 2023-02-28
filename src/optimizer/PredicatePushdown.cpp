#include "frontend/SelectStructure.h"

#include "PredicatePushdown.h"

#include "frontend/BiaodashiAuxProcessor.h"

namespace aries {
PredicatePushdown::PredicatePushdown() {
}

std::string PredicatePushdown::ToString() {
    return std::string("Predicate Pushdown");
}

SQLTreeNodePointer PredicatePushdown::OptimizeTree(SQLTreeNodePointer arg_input) {

    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->predicate_pushdown_single_query(arg_input);
    }

    /*do for my subqueries*/
    for (size_t si = 0; si < my_query_context->subquery_context_array.size(); si++) {
        QueryContextPointer a_subquery_context = my_query_context->subquery_context_array[si];

        //this->predicate_pushdown_single_query(std::dynamic_pointer_cast<SelectStructure>(a_subquery_context->GetSelectStructure())->GetQueryPlanTree());
        this->OptimizeTree(
                std::dynamic_pointer_cast<SelectStructure>(a_subquery_context->GetSelectStructure())->GetQueryPlanTree());

    }

    return arg_input;

}


void PredicatePushdown::predicate_pushdown_single_query(SQLTreeNodePointer arg_input) {
    ////LOG(INFO) << "------------------------------------------------\n";
    this->predicate_pushdown_handling_node(arg_input);
}

void PredicatePushdown::predicate_pushdown_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    ////LOG(INFO) << "\nNow we handle node: " << arg_input->ToString(0) << "\n\n";

    switch (arg_input->GetType()) {
        case SQLTreeNodeType::Column_NODE:
        case SQLTreeNodeType::Group_NODE:
        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::Limit_NODE:
            /*Da Jiang You!*/
            this->predicate_pushdown_handling_node(arg_input->GetTheChild());
            break;

        case SQLTreeNodeType::SetOp_NODE:
            this->predicate_pushdown_handling_node(arg_input->GetLeftChild());
            this->predicate_pushdown_handling_node(arg_input->GetRightChild());
            break;

        case SQLTreeNodeType::Table_NODE:
            /*ending here*/
            break;

        case SQLTreeNodeType::Filter_NODE:
            if (arg_input->GetTheChild()->GetType() == SQLTreeNodeType::BinaryJoin_NODE) {
                /*pushdown*/
                //Here, I lost context information, this way is ugly!!

                this->pushdown_from_filter_into_join(arg_input);

            }
            else if ( arg_input->GetTheChild()->GetType() == SQLTreeNodeType::InnerJoin_NODE )
            {
                pushdown_from_filter_into_inner_join( arg_input );
            } else {
                this->predicate_pushdown_handling_node(arg_input->GetTheChild());
            }

            break;
        case SQLTreeNodeType::InnerJoin_NODE:
        {
            pushdown_from_inner_join_into_child( arg_input );
            break;
        }

        case SQLTreeNodeType::BinaryJoin_NODE:
            switch (arg_input->GetJoinType()) {
                // case JoinType::InnerJoin:
                case JoinType::RightJoin:
                case JoinType::RightOuterJoin:
                case JoinType::LeftJoin:
                case JoinType::LeftOuterJoin:
                    this->pushdown_from_join_into_child(arg_input);
                    break;
                default:
                    predicate_pushdown_handling_node(arg_input->GetLeftChild());
                    predicate_pushdown_handling_node(arg_input->GetRightChild());
                    break;
            }
            break;

        default:
        {
            ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()));
        }
    }

}

void PredicatePushdown::pushdown_from_filter_into_inner_join( SQLTreeNodePointer arg_input )
{
    BiaodashiPointer the_filter_node = arg_input->GetFilterStructure();
    if (the_filter_node == nullptr)
        return;
    BiaodashiAuxProcessor expr_processor;
    auto initial_and_list = expr_processor.generate_and_list(the_filter_node);
    std::vector<BiaodashiPointer> pushed_and_list;
    std::vector<BiaodashiPointer> unpushed_and_list;
    bool should_reset = false;

    for (size_t i = 0; i < initial_and_list.size(); i++) {
        BiaodashiPointer aexpr = initial_and_list[i];
        /*todo*/
        //obtain_for_pushdown();
        (std::dynamic_pointer_cast<CommonBiaodashi>(aexpr))->ObtainReferenceTableInfo();

//if(aexpr->contain_subquery == false)
        if (true) {
            pushed_and_list.push_back(aexpr);
            should_reset = true;
        } else {
            unpushed_and_list.push_back(aexpr);
        }
    }

    SQLTreeNodePointer the_join_node = arg_input->GetTheChild();
    if ( pushed_and_list.size() > 0 ) {
        the_join_node->receive_pushed_expr_list(pushed_and_list);
    }

    /*reset myself*/
    if (unpushed_and_list.size() == 0) {
        arg_input->GetParent()->ResetTheChild( arg_input->GetTheChild() );
        arg_input->GetTheChild()->SetParent(arg_input->GetParent()  );
    } else {
        if ( should_reset ) {
            /*first we make a biaodashi from unpushed_and_list, and then we use it*/
            BiaodashiAuxProcessor expr_processor;
            BiaodashiPointer new_expr = expr_processor.make_biaodashi_from_and_list( unpushed_and_list );

            arg_input->SetFilterStructure( new_expr );
        }
    }

    /*recursive call*/
    this->predicate_pushdown_handling_node(the_join_node);
}

void PredicatePushdown::pushdown_from_filter_into_join(SQLTreeNodePointer arg_input) {

    ////LOG(INFO) << "pushdown_from_filter_into_join\n";

    assert(arg_input != nullptr);
    BiaodashiPointer the_filter_node = arg_input->GetFilterStructure();
    if (the_filter_node == nullptr)
        return;

    /*pickup the exprs we should push down*/
    BiaodashiAuxProcessor expr_processor;
    ////LOG(INFO) << "\n\nthe filter node: " << the_filter_node->ToString() << "\n\n";

    std::vector<BiaodashiPointer> initial_and_list = expr_processor.generate_and_list(the_filter_node);

    ////LOG(INFO) << "initial_and_list length = " << initial_and_list.size() << "\n\n";

    std::vector<BiaodashiPointer> pushed_and_list;
    std::vector<BiaodashiPointer> unpushed_and_list;
    bool should_reset = false;

    for (size_t i = 0; i < initial_and_list.size(); i++) {
        BiaodashiPointer aexpr = initial_and_list[i];
        /*todo*/
        //obtain_for_pushdown();
        (std::dynamic_pointer_cast<CommonBiaodashi>(aexpr))->ObtainReferenceTableInfo();

//if(aexpr->contain_subquery == false)
        if (true) {
            pushed_and_list.push_back(aexpr);
            should_reset = true;
        } else {
            unpushed_and_list.push_back(aexpr);
        }
    }


/******************debug***********************/
    for (size_t pi = 0; pi < pushed_and_list.size(); pi++) {
        //LOG(INFO) << "pushed_and_list[" << pi << "]:  " + pushed_and_list[pi]->ToString() << "\n";
    }


    for (size_t ui = 0; ui < unpushed_and_list.size(); ui++) {
        //LOG(INFO) << "unpushed_and_list[" << ui << "]:  " + unpushed_and_list[ui]->ToString() << "\n";
    }

/******************debug***********************/


    SQLTreeNodePointer the_join_node = arg_input->GetTheChild();
    if (pushed_and_list.size() > 0) {
        the_join_node->receive_pushed_expr_list(pushed_and_list);
    }

    /*reset myself*/
    if (unpushed_and_list.size() == 0) {
        arg_input->GetParent()->ResetTheChild(arg_input->GetTheChild());
        arg_input->GetTheChild()->SetParent( arg_input->GetParent() );
    } else {
        if (should_reset) {
            /*first we make a biaodashi from unpushed_and_list, and then we use it*/
            BiaodashiAuxProcessor expr_processor;
            BiaodashiPointer new_expr = expr_processor.make_biaodashi_from_and_list(unpushed_and_list);

            arg_input->SetFilterStructure(new_expr);
        }
    }

    /*recursive call*/
    this->predicate_pushdown_handling_node(the_join_node);

}

int PredicatePushdown::can_be_pushed_down(BiaodashiPointer expr, SQLTreeNodePointer node, bool from_filter_node) {
    if (!expr) {
        return 0;
    }

    if (node->GetType() != SQLTreeNodeType::BinaryJoin_NODE) {
        return 0;
    }

    bool can_push_down_to_left = false;
    bool can_push_down_to_right = false;

    switch (node->GetJoinType()) {
        case JoinType::InnerJoin:
            can_push_down_to_left = true;
            can_push_down_to_right = true;
            break;
        case JoinType::LeftJoin:
        case JoinType::LeftOuterJoin:
            can_push_down_to_right = !from_filter_node;
            can_push_down_to_left = from_filter_node;
            break;
        case JoinType::RightJoin:
        case JoinType::RightOuterJoin:
            can_push_down_to_left = !from_filter_node;
            can_push_down_to_right = from_filter_node;
            break;
        default: break;
    }

    auto expr_involved_table_list = std::dynamic_pointer_cast<CommonBiaodashi>(expr)->GetInvolvedTableList();
    auto left_child_table_list = node->GetLeftChild()->GetInvolvedTableList();
    auto right_child_table_list = node->GetRightChild()->GetInvolvedTableList();

    auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( expr );

    auto columns = condition->GetAllReferencedColumns();
    bool has_decimal = false;
    for ( const auto& column : columns )
    {
        if ( column->GetValueType() == BiaodashiValueType::DECIMAL )
        {
            has_decimal = true;
            break;
        }
    }

    if ( has_decimal )
    {
        return 0;
    }

    if (can_push_down_to_left && the_first_list_cover_the_second_one(left_child_table_list, expr_involved_table_list)) {
        return -1;
    }

    if (can_push_down_to_right && the_first_list_cover_the_second_one(right_child_table_list, expr_involved_table_list)) {
        return 1;
    }

    return 0;

    // if (node->GetLeftChild()->GetType() != SQLTreeNodeType::BinaryJoin_NODE 
    //     && the_first_list_cover_the_second_one(left_child_table_list, expr_involved_table_list) 
    //     && can_push_down_to_left) {
    //     return -1;
    // }
    // if (node->GetRightChild()->GetType() != SQLTreeNodeType::BinaryJoin_NODE 
    //             && the_first_list_cover_the_second_one(right_child_table_list, expr_involved_table_list) 
    //             && can_push_down_to_right) {
    //     return 1;
    // }

    // return can_be_pushed_down(expr, node->GetLeftChild(), only_accept_inner_join) || can_be_pushed_down(expr, node->GetRightChild(), only_accept_inner_join);
}

void PredicatePushdown::pushdown_from_inner_join_into_child( SQLTreeNodePointer arg_input )
{
    auto pushed_from_parent = arg_input->GetPushedList();

    for ( auto& condition : arg_input->GetInnerJoinConditions() )
    {
        auto expr = std::dynamic_pointer_cast< CommonBiaodashi >( condition );
        expr->ObtainReferenceTableInfo();
    }

    pushed_from_parent.insert( pushed_from_parent.end(), arg_input->GetInnerJoinConditions().cbegin(), arg_input->GetInnerJoinConditions().cend() );
    arg_input->GetInnerJoinConditions().clear();

    BiaodashiAuxProcessor expr_processor;
    auto builder = std::make_shared<SQLTreeNodeBuilder>( arg_input->GetMyQuery() );

    /**
     * 首先将所有可能的 filter 条件下推到各个子节点，比如：
     * lineitem.o_orderky = 100， 可以下推到 lineitem 的 table node 节点
     */
    std::vector< BiaodashiPointer > unpushed;
    for ( auto& condition : pushed_from_parent )
    {
        std::vector< BiaodashiPointer > need_to_push_down;

        auto expr = std::dynamic_pointer_cast< CommonBiaodashi >( condition );
        auto involved_tables = expr->GetInvolvedTableList();
        auto columns = expr->GetAllReferencedColumns();

        bool has_decimal = false;

        for ( const auto& column : columns )
        {
            if ( column->GetValueType() == BiaodashiValueType::DECIMAL )
            {
                has_decimal = true;
                break;
            }
        }

        CommonBiaodashiPtr left_condition, right_condition;
        std::vector< BasicRelPointer > left_tables, right_tables;
        if ( expr->GetType() == BiaodashiType::Bijiao )
        {
            left_condition = std::dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 0 ) );
            right_condition = std::dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 1 ) );

            left_tables = left_condition->GetInvolvedTableList();
            right_tables = right_condition->GetInvolvedTableList();
        }
        else if ( expr->IsTrueConstant() || involved_tables.empty() )
        {
            continue;
        }

        bool left_matched = false;
        bool right_matched = false;

        SQLTreeNodePointer target_chaild;

        for ( size_t i = 0; i < arg_input->GetChildCount(); i++ )
        {
            auto child = arg_input->GetChildByIndex( i );
            auto child_tables = child->GetInvolvedTableList();

            if ( !has_decimal && the_first_list_cover_the_second_one( child_tables, involved_tables ) )
            {
                need_to_push_down.emplace_back( condition );
                target_chaild = child;
                break;
            }
            else if ( expr->GetType() == BiaodashiType::Bijiao )
            {

                
                if ( !left_tables.empty() && the_first_list_cover_the_second_one( child_tables, left_tables ) )
                {
                    left_matched = true;
                }
                else if ( !right_tables.empty() && the_first_list_cover_the_second_one( child_tables, right_tables ) )
                {
                    right_matched = true;
                }
            }

            if ( left_matched && right_matched )
            {
                break;
            }
        }

        if ( left_matched && right_matched )
        {
            arg_input->GetInnerJoinConditions().emplace_back( condition );
        }
        else if ( need_to_push_down.empty() )
        {
            unpushed.emplace_back( condition );
        }

        if ( need_to_push_down.empty() )
        {
            continue;
        }

        if ( target_chaild->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
        {
            target_chaild->receive_pushed_expr_list( need_to_push_down );
        }
        else if ( target_chaild->GetType() == SQLTreeNodeType::Filter_NODE )
        {
            auto old_condition = target_chaild->GetFilterStructure();
            if ( old_condition )
            {
                need_to_push_down.emplace_back( old_condition );
            }
            target_chaild->SetFilterStructure( expr_processor.make_biaodashi_from_and_list( need_to_push_down ) );
        }
        else
        {
            auto new_condition = expr_processor.make_biaodashi_from_and_list( need_to_push_down );
            auto new_filter_node = builder->makeTreeNode_Filter( new_condition );

            SQLTreeNode::SetTreeNodeChild( new_filter_node, target_chaild );
            arg_input->CompletelyResetAChild( target_chaild, new_filter_node );
            new_filter_node->SetParent( arg_input );
        }
    }

    for ( size_t i = 0; i < arg_input->GetChildCount(); i++ )
    {
        auto child = arg_input->GetChildByIndex( i );
        predicate_pushdown_handling_node( child );
    }

    if ( unpushed.empty() )
    {
        return;
    }

    auto parent = arg_input->GetParent();
    while ( parent->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
    {
        arg_input = parent;
        parent = parent->GetParent();
    }

    if ( parent->GetType() == SQLTreeNodeType::Filter_NODE )
    {
        unpushed.emplace_back( parent->GetFilterStructure() );
        auto new_filter_expr = expr_processor.make_biaodashi_from_and_list( unpushed );
        arg_input->GetParent()->SetFilterStructure( new_filter_expr );
    }
    else
    {
        auto new_filter_expr = expr_processor.make_biaodashi_from_and_list( unpushed );
        auto new_filter_node = builder->makeTreeNode_Filter(new_filter_expr);

        auto parent = arg_input->GetParent();
        SQLTreeNode::SetTreeNodeChild( new_filter_node, arg_input );
        parent->CompletelyResetAChild( arg_input, new_filter_node );
        new_filter_node->SetParent( parent );
    }
}

/*todo: we did not consider join type! maybe need recheck!*/
void PredicatePushdown::pushdown_from_join_into_child(SQLTreeNodePointer arg_input) {

    ////LOG(INFO) << "pushdown_from_join_into_child\n";

    /*its join condition can contain two parts: one is its own specified in the origial SQL ,
    and the other one is received from top*/

    BiaodashiPointer first_condition = arg_input->GetJoinCondition();
    std::vector<BiaodashiPointer> second_list = arg_input->GetPushedList();
    ////LOG(INFO) << "pushdown_from_join_into_child: second_list.size() = " << second_list.size() << "\n";

    /*combine the two parts*/
    BiaodashiAuxProcessor expr_processor;
    ////LOG(INFO) << "\n\n" << "first_condition: " << first_condition->ToString() << "\n\n";
    std::vector<BiaodashiPointer> first_list = expr_processor.generate_and_list(first_condition);
    for (size_t fi = 0; fi < first_list.size(); fi++) {
        std::dynamic_pointer_cast<CommonBiaodashi>(first_list[fi])->ObtainReferenceTableInfo();


    }

    ////LOG(INFO) << "pushdown_from_join_into_child: first_list.size() = " << first_list.size() << "\n";

    /*dispatch*/
    std::vector<BiaodashiPointer> left_child_list;
    std::vector<BiaodashiPointer> right_child_list;
    std::vector<BiaodashiPointer> myself_list;
    std::vector<BiaodashiPointer> filter_list;


    LOG(INFO) << "my join type: " << static_cast<int>(arg_input->GetJoinType()) << ", my condition: " << arg_input->GetJoinCondition()->ToString() << std::endl;
    SQLTreeNodePointer left_child = arg_input->GetLeftChild();
    SQLTreeNodePointer right_child = arg_input->GetRightChild();

    if (first_list.size() == 1 && !second_list.empty()) {
        auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(first_list[0]);
        if (expr->GetType() == BiaodashiType::Zhenjia && expr->GetMyBoolValue()) {
            first_list.clear();
        }
    }

    for (size_t i = 0; i < first_list.size(); i++) {
        BiaodashiPointer a_expr = first_list[i];


        std::vector<BasicRelPointer> expr_involved_table_list = std::dynamic_pointer_cast<CommonBiaodashi>(
                a_expr)->GetInvolvedTableList();

        if (expr_involved_table_list.size() == 0) {
            /*we cannot pushdow a 1==1 or 3>2"*/
            LOG(INFO) << "1.expr_involved_table_list is empty" << std::endl;
            myself_list.push_back(a_expr);
            continue;
        }

        /*dispatch*/
        int push_down_flag = can_be_pushed_down(a_expr, arg_input);
        if (push_down_flag == -1) {
            left_child_list.push_back(a_expr);
            LOG(INFO) << "a_expr : " << a_expr->ToString() << " push down left" << std::endl;
        } else if (push_down_flag == 1) {
            right_child_list.push_back(a_expr);
            LOG(INFO) << "a_expr : " << a_expr->ToString() << " push down right" << std::endl;
        } else {
            LOG(INFO) << "a_expr : " << a_expr->ToString() << " cannot push down" << std::endl;
            myself_list.push_back(a_expr);
        }
    }

    for (const auto& expr : second_list) {
        std::vector<BasicRelPointer> expr_involved_table_list = std::dynamic_pointer_cast<CommonBiaodashi>(
                expr)->GetInvolvedTableList();
        
        if (expr_involved_table_list.size() == 0) {
            /*we cannot pushdow a 1==1 or 3>2"*/
            filter_list.push_back(expr);
            LOG(INFO) << "expr_involved_table_list is empty" << std::endl;
            continue;
        }

        int push_down_flag = can_be_pushed_down(expr, arg_input, true);
        if (push_down_flag == -1) {
            left_child_list.push_back(expr);
            LOG(INFO) << "expr : " << expr->ToString() << " push down left" << std::endl;
        } else if (push_down_flag == 1) {
            right_child_list.push_back(expr);
            LOG(INFO) << "expr : " << expr->ToString() << " push down right" << std::endl;
        } else {
            LOG(INFO) << "expr : " << expr->ToString() << "cannot push down" << std::endl;
            filter_list.push_back(expr);
        }
    }

    /*handling left child*/
    this->join_child_pushdown(arg_input, left_child, left_child_list, true);
    this->predicate_pushdown_handling_node(left_child);

    /*handling right child*/
    this->join_child_pushdown(arg_input, right_child, right_child_list, false);
    this->predicate_pushdown_handling_node(right_child);


    /**
     * Filter 条件放在 inner join 的 condition 中是等价的，
     * 这里将 filter 条件放在 condition 中，以便后面的 TwoPhaseJoin 拆分“等值条件”
     */
    if (arg_input->GetJoinType() == JoinType::InnerJoin) {
        myself_list.insert(myself_list.cend(), filter_list.cbegin(), filter_list.cend());
        filter_list.clear();
    }

    /*reset myself*/

    BiaodashiPointer new_filter_expr = nullptr;
    BiaodashiPointer new_condition_expr = nullptr;
    if (myself_list.size() == 0) {
        new_condition_expr = expr_processor.make_biaodashi_boolean(true);
        LOG(INFO) << "myself_list is empty" << std::endl;
    } else {
        while (myself_list.size() > 1) {
            auto expr = (CommonBiaodashi*)(myself_list[0].get());
            if (expr->GetType() == BiaodashiType::Zhenjia && expr->GetMyBoolValue()) {
                myself_list.erase(myself_list.begin());
            } else {
                break;
            }
        }
        new_condition_expr = expr_processor.make_biaodashi_from_and_list(myself_list);
        for (const auto& i : myself_list) {
            LOG(INFO) << "myself_list: " << i->ToString() << std::endl;
        }
    }

    arg_input->SetJoinCondition(new_condition_expr);

    if (!filter_list.empty()) {
        if (arg_input->GetParent()->GetType() == SQLTreeNodeType::Filter_NODE) {
            filter_list.emplace_back(arg_input->GetParent()->GetFilterStructure());
            new_filter_expr = expr_processor.make_biaodashi_from_and_list(filter_list);
            arg_input->GetParent()->SetFilterStructure(new_filter_expr);
        } else {
            new_filter_expr = expr_processor.make_biaodashi_from_and_list(filter_list);
            auto builder = std::make_shared<SQLTreeNodeBuilder>(arg_input->GetMyQuery());
            auto new_filter_node = builder->makeTreeNode_Filter(new_filter_expr);

            auto parent = arg_input->GetParent();
            SQLTreeNode::SetTreeNodeChild( new_filter_node, arg_input );
            parent->CompletelyResetAChild(arg_input, new_filter_node);
            new_filter_node->SetParent( parent );
        }
    }

}


void PredicatePushdown::join_child_pushdown(SQLTreeNodePointer arg_binary_join_node, SQLTreeNodePointer arg_child_node,
                                            std::vector<BiaodashiPointer> arg_expr_list, bool arg_left_or_right) {
    if (arg_expr_list.size() == 0)
        return;


    for (size_t ai = 0; ai < arg_expr_list.size(); ai++) {
        ////LOG(INFO) << "join_child_pushdown " << ai << "   " << arg_expr_list[ai]->ToString() << "\n";
    }

    if (arg_child_node->GetType() == SQLTreeNodeType::BinaryJoin_NODE ||
        arg_child_node->GetType() == SQLTreeNodeType::InnerJoin_NODE) {
        arg_child_node->receive_pushed_expr_list(arg_expr_list);
    } else {
        /*we insert a new filter node between this and left_child*/

        BiaodashiAuxProcessor expr_processor;
        BiaodashiPointer new_filter_expr = expr_processor.make_biaodashi_from_and_list(arg_expr_list);
        ////LOG(INFO) << "new_filter_expr:  " << new_filter_expr << "\n";

        /*todo: we need to setup context*/

        SQLTreeNodeBuilder tree_builder(arg_binary_join_node->GetMyQuery());
        SQLTreeNodePointer new_filter_node = tree_builder.makeTreeNode_Filter(new_filter_expr);

        /*These two is not necessary? It might be convenient?*/
        //new_filter_node.setExpNum(arg_expr_list.length);
        //new_filter_node.setExpList(arg_expr_list);

        SQLTreeNode::SetTreeNodeChild( new_filter_node, arg_child_node );
        /*unncessary to setparent since it would be automatically done when add child*/
        //new_filter_node->SetParent(arg_binary_join_node.get());

        if (arg_left_or_right == true) {
            arg_binary_join_node->ResetLeftChild(new_filter_node);
            new_filter_node->SetParent( arg_binary_join_node );
        } else {
            arg_binary_join_node->ResetRightChild(new_filter_node);
            new_filter_node->SetParent( arg_binary_join_node );
        }
    }
}


bool PredicatePushdown::the_first_list_cover_the_second_one(
        std::vector<BasicRelPointer> arg_first_list,
        std::vector<BasicRelPointer> arg_second_list) {

    /*debug*/
    ////LOG(INFO) << "PredicatePushdown::the_first_list_cover_the_second_one\n";
//	for(int fi = 0; fi < arg_first_list.size(); fi++)
//	{
//	    ////LOG(INFO) << "arg_first_list:" << fi << "-->" << arg_first_list[fi]->ToString() << "\n";
//	}
//
//	for(int si = 0; si < arg_second_list.size(); si++)
//	{
    ////LOG(INFO) << "arg_second_list:" << si << "-->" << arg_second_list[si]->ToString() << "\n";
//	}
    /*debug*/

    /*we check every item in arg_second_list to determine whether it is in arg_first_list*/
    for (size_t si = 0; si < arg_second_list.size(); si++) {
        bool t_ret = false;

        for (size_t fi = 0; fi < arg_first_list.size(); fi++) {
            if (BasicRel::__compareTwoBasicRels(arg_second_list[si].get(), arg_first_list[fi].get()) == true) {
                t_ret = true;
                break;
            }
        }

        if (t_ret == false)
            return false;
    }

    return true;

}

}
