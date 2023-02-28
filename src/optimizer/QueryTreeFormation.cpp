#include <algorithm>
#include "QueryTreeFormation.h"
#include "frontend/NodeColumnStructure.h"

#include "frontend/SelectStructure.h"
#include "frontend/BiaodashiAuxProcessor.h"


namespace aries {

QueryTreeFormation::QueryTreeFormation() {
}

std::string QueryTreeFormation::ToString() {
    return std::string("Query Tree Formation");
}

SQLTreeNodePointer QueryTreeFormation::OptimizeTree(SQLTreeNodePointer arg_input) {
    QueryContextPointer my_query_context = std::dynamic_pointer_cast<SelectStructure>(
            arg_input->GetMyQuery())->GetQueryContext();

    /*do for myself*/
    /*if I am a from clause actually, then done already!*/
    if (my_query_context->type != QueryContextType::FromSubQuery) {
        this->query_formation_single_query(arg_input);
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


void QueryTreeFormation::query_formation_single_query(SQLTreeNodePointer arg_input) {
    this->query_formation_handling_node(arg_input);
}


/********************************************* A Node: Column*********************************************/


void QueryTreeFormation::HandlingColumnNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {
    /*no need to consider required columns when processing own*/

    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());

    /* if this query has group, then the columnnode
       is actually executed in the groupby part.
       i.e., the underlying group node.
       But it is possible that we still need such a node
       to filter unnecessary columns that are needed by sort node,
       unless this node is merged into sort node
    */

    if (the_ss->GetNeedGroupbyOperation() == true || the_ss->IsSetQuery() == true) {

        arg_input->SetForwardMode4ColumnNode(true);

        /*to know the number of columns I need to forward!*/
        RelationStructurePointer the_relation_structure = the_ss->GetRelationStructure();
        arg_input->SetForwardedColumnCount(the_relation_structure->GetColumnCount());

        /*do nothing! cause I just forward!*/

    } else {
        std::vector<ColumnShellPointer> columns = the_ss->GetReferencedColumnInSelect();

        //for(int i = 0; i < columns.size(); i++)
        //{
        //	columns[i]->SetIndexColumnSource(i);
        //}

        for (size_t i = 0; i < columns.size(); i++) {
            /*NOTICE:Start From 1, not 0!*/
            arg_input->SetPositionForReferencedColumn(columns[i], (i + 1));
        }

        arg_input->SetReferencedColumnArray(columns);

        // arg_input->GetTheChild()->SetRequiredColumnArray(arg_input->GetReferencedColumnArray());
        if(arg_input->GetTheChild())
        {
            arg_input->GetTheChild()->AddRequiredColumnArray(arg_input->GetReferencedColumnArray());
            if(arg_input->GetTheChild()->GetSpoolId() > -1 && arg_input->GetTheChild()->GetSameNode())
                arg_input->GetTheChild()->GetSameNode()->AddRequiredColumnArray( arg_input->GetReferencedColumnArray() );
        }
    }

    //LOG(INFO) << "QueryTreeFormation::HandlingColumnNode_ProcessingOwnColumns" << "\n";

}

void QueryTreeFormation::HandlingColumnNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {


    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    std::string table_name = "UnnamedQueryResult";

    bool reset_table_name = false;

    /*our output name should be set according to the relation alias if I am the top node of a subquery plan tree.*/
    if (arg_input->GetIsTopNodeOfAFromSubquery() == true) {
        table_name = arg_input->GetBasicRel()->GetRelationStructure()->GetName(); /*the alias of the whole subquery*/
        /*for debug*/

        ////LOG(INFO) << "111\tQueryTreeFormation::HandlingColumnNode_FormingOutputColumns-->" << table_name << "\n";

        reset_table_name = true;
    }

    if (arg_input->GetForwardMode4ColumnNode() == true) {

        //LOG(INFO) << "arg_input->GetForwardMode4ColumnNode() == true" << "\n";

        /*forward child node's structure*/
        NodeRelationStructurePointer child_nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();

        if (arg_input->GetForwardedColumnCount() == child_nrsp->GetColumnCount()) {
            /*we actually do nothing!*/
            /*for now! wait check after final_nrsp*/
            arg_input->SetColumnNodeRemovable(true);
        }

        for (size_t i = 0; i < arg_input->GetForwardedColumnCount(); i++) {
            //LOG(INFO) << "i < arg_input->GetForwardedColumnCount() : " << i << ":" << arg_input->GetForwardedColumnCount() << "\n";
            my_nrsp->AddColumn(
                    child_nrsp->GetColumnbyIndex(i),
                    reset_table_name ? table_name : child_nrsp->GetColumnSourceTableNamebyIndex(i)
            );

        }

    } else {
        /*here we use the_relation_structure of the whole query! But we can also use the checked_expr_array in the_ss. */
        SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());
        RelationStructurePointer the_relation_structure = the_ss->GetRelationStructure();
        for (size_t i = 0; i < the_relation_structure->GetColumnCount(); i++) {
            /*the relation_structure is basically equal to checked_expr_array*/
            my_nrsp->AddColumn(
                    NodeColumnStructure::__createInstanceFromColumnStructure(the_relation_structure->GetColumn(i)),
                    table_name);

        }
    }

    /****************now my_nrsp is done!*****************/

    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();
    /*maybe I am just the column node of a subquery!!!*/
    //LOG(INFO) << my_required_columns.size() << "\n";
    if (my_required_columns.size() == 0) {
        /*NO*/
        // LOG(INFO) << "ok.1\n" << arg_input->GetMyQuery()->ToString() << "\n";
        // assert(arg_input->GetMyQuery() != nullptr);
        SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());
        // assert(the_ss != NULL);
        // LOG(INFO) << "ok.2\n";
        std::vector<BiaodashiPointer> all_exprs = the_ss->GetAllSelectExprs();
        // LOG(INFO) << "ok.3\n";
        arg_input->SetExprs4ColumnNode(all_exprs);

        arg_input->SetNodeRelationStructure(my_nrsp);

        if (!arg_input->IsColumnNodeRemovable()) {
            std::vector<int> output_expr_sequence;
            for (size_t i = 1; i <= arg_input->GetForwardedColumnCount(); i++) {
                auto expr = (CommonBiaodashi*)(all_exprs[i - 1].get());
                if (expr->IsVisibleInResult()) {
                    output_expr_sequence.emplace_back(i);
                }
            }

            arg_input->SetColumnOutputSequence(output_expr_sequence);
        }
    } else {
        /*YES*/
        /*todo: it is possible to still be removable if the required columns are the same as the input*/
        arg_input->SetColumnNodeRemovable(false);

        std::vector<int> output_expr_sequence;


        NodeRelationStructurePointer final_nrsp = std::make_shared<NodeRelationStructure>();
        for (size_t mrci = 0; mrci < my_required_columns.size(); mrci++) {
            int tmp_column_index = -1;

            for (size_t tl = 0; tl < my_nrsp->GetColumnCount(); tl++) {
                ////LOG(INFO) << my_required_columns[mrci]->GetColumnName() << "-----------" << my_nrsp->GetColumnbyIndex(tl)->GetName() << "\n";
                if (my_required_columns[mrci]->GetColumnName() == my_nrsp->GetColumnbyIndex(tl)->GetName()) {
                    tmp_column_index = tl;
                    break;
                }
            }

            assert(tmp_column_index >= 0);

            /*todo: consider whether duplication is possible?*/
            /*Set (+1) is actually for the forward mode. For the real projection mode, unneeded!*/
            output_expr_sequence.push_back(tmp_column_index + 1);

            final_nrsp->AddColumn(
                    my_nrsp->GetColumnbyIndex(tmp_column_index),
                    my_nrsp->GetColumnSourceTableNamebyIndex(tmp_column_index)
            );
        }

        if(arg_input->GetTheChild())
        {
            auto& child_unique_keys = arg_input->GetTheChild()->GetUniqueKeys();
            std::vector< int > my_unique_keys;

            for ( const auto& keys : child_unique_keys )
            {
                std::vector< int > my_unique_keys;
                for ( const auto& key : keys )
                {
                    for ( size_t i = 0; i < output_expr_sequence.size(); i++ )
                    {
                        if ( key == output_expr_sequence[ i ] )
                        {
                            my_unique_keys.emplace_back( i + 1 );
                            break;
                        }
                    }
                }

                if ( my_unique_keys.size() == child_unique_keys.size() )
                {
                    arg_input->AddUniqueKeys( my_unique_keys );
                }
            }
        }

        if (arg_input->GetForwardMode4ColumnNode() == true) {
            /*now, the sequence means how to forward the input columns!*/
            arg_input->SetColumnOutputSequence(output_expr_sequence);

            /*if the sequence is exactly the child input, then removeable!*/
            bool check_forwarded_result = false;
            if (output_expr_sequence.size() == arg_input->GetForwardedColumnCount()) {
                bool oesi_result = true;
                for (size_t oesi = 0; oesi < output_expr_sequence.size(); oesi++) {
                    if (( size_t )output_expr_sequence[oesi] != (oesi + 1)) {
                        oesi_result = false;
                        break;
                    }
                }

                if (oesi_result == true) {
                    check_forwarded_result = true;
                }
            }

            if (check_forwarded_result == true) {
                arg_input->SetColumnNodeRemovable(true);
            }

        } else {
            /*now, the sequence means the exprs this node should do!*/
            SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());
            std::vector<BiaodashiPointer> all_exprs = the_ss->GetAllSelectExprs();


            std::vector<BiaodashiPointer> vbp;

            for (size_t oesi = 0; oesi < output_expr_sequence.size(); oesi++) {
                int real_index = output_expr_sequence[oesi] - 1;
                vbp.push_back(all_exprs[real_index]);
            }

            arg_input->SetExprs4ColumnNode(vbp);

        }

        arg_input->SetNodeRelationStructure(final_nrsp);
    }
    if( arg_input->GetTheChild() )
    {
        auto all_exprs = arg_input->GetExprs4ColumnNode();
        for( auto expr : all_exprs )
        {
            bool changed_to_nullable = false;
            ResetColumnNullableInfo( expr, arg_input->GetTheChild()->GetNodeRelationStructure(), changed_to_nullable );
            if( changed_to_nullable )
                std::dynamic_pointer_cast< CommonBiaodashi >( expr )->SetIsNullable( true );
        }
    }
    
    //LOG(INFO) << "QueryTreeFormation::HandlingColumnNod_FormatingOwnColumns" << "\n";
}


void QueryTreeFormation::HandlingColumnNode(SQLTreeNodePointer arg_input) {
    //TroubleHandler::showMessage("HandlingColumnNode...-------------------------------------------------------------begin\n");

    this->HandlingColumnNode_ProcessingOwnColumns(arg_input);

    //LOG(INFO) << "QueryTreeFormation::HandlingColumnNode ---- 1\n";

    this->query_formation_handling_node(arg_input->GetTheChild());
    //LOG(INFO) << "QueryTreeFormation::HandlingColumnNode ---- 2\n";

    this->HandlingColumnNode_FormingOutputColumns(arg_input);
    //LOG(INFO) << "QueryTreeFormation::HandlingColumnNode ---- 3\n";

    //TroubleHandler::showMessage("HandlingColumnNode...-------------------------------------------------------------end\n");
}

void QueryTreeFormation::HandlingLimitNode_ProcessingOwnColumns( SQLTreeNodePointer arg_input )
{
    auto child = arg_input->GetTheChild();

    child->AddRequiredColumnArray( arg_input->GetRequiredColumnArray() );
}

void QueryTreeFormation::HandlingLimitNode_FormingOutputColumns( SQLTreeNodePointer arg_input )
{
    auto child = arg_input->GetTheChild();
    arg_input->SetNodeRelationStructure( child->GetNodeRelationStructure() );
}

void QueryTreeFormation::HandlingLimitNode( SQLTreeNodePointer arg_input )
{
    HandlingLimitNode_ProcessingOwnColumns( arg_input );

    query_formation_handling_node(arg_input->GetTheChild());

    HandlingLimitNode_FormingOutputColumns( arg_input );

}

/********************************************* A Node *********************************************/




/********************************************* A Node: Group *********************************************/


void QueryTreeFormation::HandlingGroupNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {

    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());
//	LOG(INFO) << "\n\n" << the_ss->ToString() << "\n\n";

    std::vector<ColumnShellPointer> select_columns = the_ss->GetReferencedColumnInSelect();
    // for(int sci = 0; sci < select_columns.size(); sci++)
    // {
    //     LOG(INFO) << "select_columns: " << sci << " " << select_columns[sci]->ToString() << "\n";
    // }


    std::vector<ColumnShellPointer> groupby_columns = the_ss->GetReferencedColumnInGroupby();
    auto order_by_columns = the_ss->GetReferencedColumnInOrderby();


    // for(int gci = 0; gci < groupby_columns.size(); gci++)
    // {
    //     LOG(INFO) << "groupby_columns: " << gci << " " << groupby_columns[gci]->ToString() << "\n";
    // }

    std::vector<ColumnShellPointer> final_columns = select_columns;

    /*you have to dedup first!*/

    for (size_t gi = 0; gi < groupby_columns.size(); gi++) {

        bool found = false;
        for (size_t fi = 0; fi < final_columns.size(); fi++) {
            //if(groupby_columns[gi] == final_columns[fi])
            if (groupby_columns[gi]->GetTableName() == final_columns[fi]->GetTableName()
                &&
                groupby_columns[gi]->GetColumnName() == final_columns[fi]->GetColumnName()) {
                found = true;
                break;
            }
        }

        if (!found) {
            final_columns.push_back(groupby_columns[gi]);
        }
    }

    for (const auto& column : order_by_columns)
    {
        if (column->GetTableName() == "!-!SELECT_ALIAS!-!")
            continue;

        bool found = false;
        for (const auto& column_in_final : final_columns)
        {
            if (column->GetTableName() == column_in_final->GetTableName() && column->GetColumnName() == column_in_final->GetColumnName())
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            final_columns.emplace_back(column);
        }
    }

    arg_input->SetReferencedColumnArray(final_columns);

    arg_input->GetTheChild()->AddRequiredColumnArray( arg_input->GetReferencedColumnArray() );


    if ( arg_input->GetTheChild()->GetSpoolId() > -1 && arg_input->GetTheChild()->GetSameNode() )
    {
        arg_input->GetTheChild()->GetSameNode()->AddRequiredColumnArray( arg_input->GetReferencedColumnArray() );
    }

    /*the underlying node must use this order to give columns!*/
    auto child_required_columns = arg_input->GetTheChild()->GetRequiredColumnArray();
    for ( size_t i = 0; i < child_required_columns.size(); i++ )
    {
        arg_input->SetPositionForReferencedColumn( child_required_columns[ i ], ( i + 1 ) );
    }
}

void QueryTreeFormation::HandlingGroupNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {

    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());
    /*a groupby return a relation defined to return all select exprs and  groupby exprs: [select expr1, ..., groupby expr1, ...]*/

    /*step 1: handling select part*/
    /*we have already created the relationstructure based on the select part!*/


    RelationStructurePointer the_relation_structure = the_ss->GetRelationStructure();

    for (size_t i = 0; i < the_relation_structure->GetColumnCount(); i++) {
        /*the relation_structure is basically equal to checked_expr_array*/

        std::string tmp_table_name = the_ss->GetTableNameOfSelectItemIfColumn(i);
        my_nrsp->AddColumn(
                NodeColumnStructure::__createInstanceFromColumnStructure(the_relation_structure->GetColumn(i)),
                tmp_table_name);

    }


    //work for HavingToFilter query optimization:
    /*if we have some exprs that needed by having or others, then we have to output them*/
    GroupbyStructurePointer gbsp = arg_input->GetMyGroupbyStructure();
    std::vector<BiaodashiPointer> additional_agg_expr_array = gbsp->GetAdditionalExprsForSelect();
    for (size_t aaea_i = 0; aaea_i < additional_agg_expr_array.size(); aaea_i++) {
        CommonBiaodashi *expr_p = (CommonBiaodashi *) (additional_agg_expr_array[aaea_i].get());

        //this is a agg expr: a SQLFunction

        std::string tmp_name;
        BiaodashiValueType tmp_type;
        int tmp_length;

        tmp_name = "ADDITIONAL_GROUPBY_EXPR_" + std::to_string(aaea_i);
        tmp_type = expr_p->GetValueType();
        tmp_length = -1; /*agg funcs -- no need for this*/

        NodeColumnStructurePointer a_ncsp;

        a_ncsp = std::make_shared<NodeColumnStructure>(
                tmp_name,
                tmp_type,
                tmp_length,
                expr_p->IsNullable()
        );

        if( expr_p->GetType() == BiaodashiType::Lie )
        {
            ColumnShellPointer tmp_column_shell = boost::get< ColumnShellPointer >( expr_p->GetContent() );
            a_ncsp->SetPossibleRoot( tmp_column_shell->GetColumnStructure() );
        }

        std::string table_name = "meaningless_table_name";

        my_nrsp->AddColumn(
                a_ncsp,
                table_name
        );

    }



    /*todo: we need a demand from Sort! If Sort needs a thing not in the select list, the group node should output the involved items in groupby. For simplicity, we just ask for the group node to output all!*/
    bool output_items_in_groupby = true;

    if (output_items_in_groupby == true) {

        /*step 2: handling groupby part*/
        GroupbyStructurePointer the_groupby_part = the_ss->GetGroupbyPart();
        for (size_t i = 0; i < the_groupby_part->GetGroupbyExprCount(); i++) {

            NodeColumnStructurePointer a_ncsp; /*I will turn the expr into this!*/
            std::string table_name;


            CommonBiaodashi *a_gb_expr = (CommonBiaodashi *) ((the_groupby_part->GetGroupbyExpr(i)).get());

            /*it is a ColumnShell, it is a ColumnShell-Alias-from-select, or it is a non-ColumnShell (a biaodashi)*/

            std::string tmp_name;
            BiaodashiValueType tmp_type;
            int tmp_length;

            if (a_gb_expr->GetType() == BiaodashiType::Lie) /*a ColumnShell*/
            {
                ColumnShellPointer tmp_column_shell = boost::get<ColumnShellPointer>(a_gb_expr->GetContent());

                table_name = tmp_column_shell->GetTableName();

                if (table_name != "!-!SELECT_ALIAS!-!") {
                    /*this is a normal column*/
                    /*we can setup it manually or directly create it from the underlying columnstructure*/
                    a_ncsp = NodeColumnStructure::__createInstanceFromColumnStructure(
                            tmp_column_shell->GetColumnStructure());


                    /*we have to figure out those duplicated columns!*/

                    bool tmp_duplicated = false;

                    /*if a column in groupby is in the select part, then it is dupliated*/

                    for (size_t mi = 0; mi < my_nrsp->GetColumnCount(); mi++) {
                        std::string tmp_column_name = my_nrsp->GetColumnbyIndex(mi)->GetName();
                        std::string tmp_table_name = my_nrsp->GetColumnSourceTableNamebyIndex(mi);


                        if (a_ncsp->GetName() == tmp_column_name
                            && table_name == tmp_table_name) {
                            tmp_duplicated = true;
                            break;
                        }


                    }

                    if (tmp_duplicated == true) {
                        continue;
                    }

                    my_nrsp->AddColumn(
                            a_ncsp,
                            table_name
                    );

                    additional_agg_expr_array.emplace_back(the_groupby_part->GetGroupbyExpr(i));
                    gbsp->SetAdditionalExprsForSelect(additional_agg_expr_array);
                } else {
                    /*this is an alias to an expr in the select part*/
                    /*DO NOTHING! Because it has been alreay inserted into the noderelationstructure when processing select exprs!*/
                }
            } else {

                tmp_name = "UNNAMED_GROUPBY_EXPR_" + std::to_string(i);
                tmp_type = a_gb_expr->GetValueType(); /*I do have type!*/
                tmp_length = -1; /*I don't know!*/

                a_ncsp = std::make_shared<NodeColumnStructure>(
                        tmp_name,
                        tmp_type,
                        tmp_length,
                        a_gb_expr->IsNullable()
                );
                table_name = "meaningless_table_name";

                my_nrsp->AddColumn(
                        a_ncsp,
                        table_name
                );

            }


        }//for groupby_expr

    }

    arg_input->SetNodeRelationStructure(my_nrsp);

    auto select_structure = arg_input->GetMySelectPartStructure();
    size_t matched = 0;
    std::vector< int > unique_keys;
    for ( size_t i = 0; i < gbsp->GetGroupbyExprCount(); i++ )
    {
        auto group_by_expr = std::dynamic_pointer_cast< CommonBiaodashi >( gbsp->GetGroupbyExpr( i ) )->Normalized();
        for ( int j = 0; j < select_structure->GetAllExprCount(); j++ )
        {
            auto select_expr = std::dynamic_pointer_cast< CommonBiaodashi >( select_structure->GetSelectExpr( j ) )->Normalized();
            if ( *group_by_expr == *select_expr )
            {
                unique_keys.emplace_back( j + 1 );
                matched++;
                break;
            }
        }

        if ( matched == gbsp->GetGroupbyExprCount() )
        {
            break;
        }
    }


    if ( matched == gbsp->GetGroupbyExprCount() )
    {
        arg_input->AddUniqueKeys( unique_keys );
    }

    if( arg_input->GetTheChild() )
    {
        auto nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();
        bool changed_to_nullable;
        for( size_t gi = 0; gi < gbsp->GetGroupbyExprCount(); ++gi )
        {
            changed_to_nullable = false;
            ResetColumnNullableInfo( gbsp->GetGroupbyExpr( gi ), nrsp, changed_to_nullable );
            if( changed_to_nullable )
                std::dynamic_pointer_cast< CommonBiaodashi >( gbsp->GetGroupbyExpr( gi ) )->SetIsNullable( true );
        }

        std::vector< BiaodashiPointer > additional_exprs = gbsp->GetAdditionalExprsForSelect();
        for( auto expr : additional_exprs )
        {
            changed_to_nullable = false;
            ResetColumnNullableInfo( expr, nrsp, changed_to_nullable );
            if( changed_to_nullable )
                std::dynamic_pointer_cast< CommonBiaodashi >( expr )->SetIsNullable( true );
        }

        SelectPartStructurePointer spsp = arg_input->GetMySelectPartStructure();
        std::vector< BiaodashiPointer > select_exprs = spsp->GetAllExprs();
        for( auto expr : select_exprs )
        {
            changed_to_nullable = false;
            ResetColumnNullableInfo( expr, nrsp, changed_to_nullable );
            if( changed_to_nullable )
                std::dynamic_pointer_cast< CommonBiaodashi >( expr )->SetIsNullable( true );
        }
    }
    //LOG(INFO) << "QueryTreeFormation::HandlingGroupNode_FormingOutputColumns(SQLTreeNodePointer arg_input) -- done!\n";

}


void QueryTreeFormation::HandlingGroupNode(SQLTreeNodePointer arg_input) {
    //TroubleHandler::showMessage("HandlingGroupNode...\n");


    this->HandlingGroupNode_ProcessingOwnColumns(arg_input);

    this->query_formation_handling_node(arg_input->GetTheChild());

    this->HandlingGroupNode_FormingOutputColumns(arg_input);
}


/********************************************* A Node *********************************************/



/********************************************* A Node: Sort *********************************************/

void QueryTreeFormation::HandlingSortNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {

    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());

    /*if there is a groupby node below me, then I just need to determine the position of my reference column. Otherwise, I need to pass required columns into the underlying child node!*/

    if (the_ss->GetNeedGroupbyOperation() == true) {
        /*we do nothing here. The real work is in ReposistionColumns...*/
        ////LOG(INFO) << "we do nothing here. The real work is in ReposistionColumns\n";
    } else {

        std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();

        std::vector<ColumnShellPointer> orderby_columns = the_ss->GetReferencedColumnInOrderby();

        std::vector<ColumnShellPointer> final_columns = my_required_columns;

        for (size_t oi = 0; oi < orderby_columns.size(); oi++) {
            if (orderby_columns[oi]->GetTableName() == "!-!SELECT_ALIAS!-!") {
                continue;
            }
            bool found = false;
            for (size_t fi = 0; fi < final_columns.size(); fi++) {
                //if(orderby_columns[oi] == final_columns[fi])
                if (orderby_columns[oi]->GetTableName() == final_columns[fi]->GetTableName()
                    &&
                    orderby_columns[oi]->GetColumnName() == final_columns[fi]->GetColumnName()) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                final_columns.push_back(orderby_columns[oi]);
            }
        }


        /*the underlying node must use this order to give columns!*/
        for (size_t i = 0; i < final_columns.size(); i++) {
            //final_columns[i]->SetIndexColumnSource(i);
            arg_input->SetPositionForReferencedColumn(final_columns[i], (i + 1));
        }

        arg_input->SetReferencedColumnArray(final_columns);

        // arg_input->GetTheChild()->SetRequiredColumnArray(arg_input->GetReferencedColumnArray());
        arg_input->GetTheChild()->AddRequiredColumnArray(arg_input->GetReferencedColumnArray());
        if(arg_input->GetTheChild()->GetSpoolId() > -1 && arg_input->GetTheChild()->GetSameNode())
            arg_input->GetTheChild()->GetSameNode()->AddRequiredColumnArray(arg_input->GetReferencedColumnArray());

    }
}


void QueryTreeFormation::HandlingSortNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {
    NodeRelationStructurePointer child_nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();

    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    for (size_t i = 0; i < child_nrsp->GetColumnCount(); i++) {
        my_nrsp->AddColumn(
                child_nrsp->GetColumnbyIndex(i),
                child_nrsp->GetColumnSourceTableNamebyIndex(i)
        );
    }

    arg_input->SetNodeRelationStructure(my_nrsp);

    if( arg_input->GetTheChild() )
    {
        auto nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();
        OrderbyStructurePointer osp = arg_input->GetMyOrderbyStructure();
        for( size_t oi = 0; oi < osp->GetOrderbyItemCount(); oi++ )
        {
            bool changed_to_nullable = false;
            ResetColumnNullableInfo( osp->GetOrderbyItem( oi ), nrsp, changed_to_nullable );
            if( changed_to_nullable )
                std::dynamic_pointer_cast< CommonBiaodashi >( osp->GetOrderbyItem( oi ) )->SetIsNullable( true );
        }
    }
}

void QueryTreeFormation::HandlingSortNode_RepositionColumns(SQLTreeNodePointer arg_input) {
    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());

    /*we only handle the case there is a groupby below this!*/
    if (the_ss->GetNeedGroupbyOperation() != true) {
        return;
    }

    ////LOG(INFO) << "\t\tQueryTreeFormation::HandlingSortNode_RepositionColumns\n";

    std::vector<ColumnShellPointer> my_referenced_columns;

    /*we determine each column based on the underlying group node*/
    /*if an orderby_expr is an alias, then its alias_expr_index is its position from the underlying groupby child node!*/

    for (size_t i = 0; i < the_ss->GetOrderbyPart()->GetOrderbyItemCount(); i++) {

        CommonBiaodashi *a_ob_expr = (CommonBiaodashi *) ((the_ss->GetOrderbyPart()->GetOrderbyItem(i)).get());

        ////LOG(INFO) << i << "\t\t" << a_ob_expr->ToString() << "\n";

        if (a_ob_expr->GetType() == BiaodashiType::Lie) {
            ColumnShellPointer tmp_column_shell = boost::get<ColumnShellPointer>(a_ob_expr->GetContent());
            if (tmp_column_shell->GetTableName() == "!-!SELECT_ALIAS!-!") {
                int select_expr_index = tmp_column_shell->GetAliasExprIndex();
                //tmp_column_shell->SetIndexColumnSource(select_expr_index);
                my_referenced_columns.push_back(tmp_column_shell);
                arg_input->SetPositionForReferencedColumn(tmp_column_shell, (select_expr_index + 1));
            }
        }

    }

    /*we now can determine the position of each column from the underlying node!*/

    NodeRelationStructurePointer child_nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();

    std::vector<ColumnShellPointer> orderby_columns = the_ss->GetReferencedColumnInOrderby();

    auto all_select_items = the_ss->GetSelectPart()->GetAllExprs();
    auto all_alias = the_ss->GetSelectPart()->GetALlAliasPointers();

    for (size_t oi = 0; oi < orderby_columns.size(); oi++) {
        ColumnShellPointer acs = orderby_columns[oi];

        // LOG(INFO) << "oi:" << oi << "\t\t" << acs->ToString() << "\n";

        // maybe need to find it's alias
        std::shared_ptr<std::string> alias = nullptr;
        for (size_t j = 0; j < all_select_items.size(); j++) {
            auto raw_expr = std::dynamic_pointer_cast<CommonBiaodashi>(all_select_items[j]);
            if (raw_expr->GetType() != BiaodashiType::Lie) {
                continue;
            }

            auto column = boost::get<ColumnShellPointer>(raw_expr->GetContent());
            if (column->GetTableName() == acs->GetTableName() && column->GetColumnName() == acs->GetColumnName()) {
                alias = all_alias[j];
                break;
            }
        }


        for (size_t cni = 0; cni < child_nrsp->GetColumnCount(); cni++) {
            /*now we have a columnshell, we need to compare it to every column from the underlying child node!*/

            /*we compare both table_name and column_name*/
            bool comp_ret = false;
            if (acs->GetTableName() == child_nrsp->GetColumnSourceTableNamebyIndex(cni)) {
                if (acs->GetColumnName() == child_nrsp->GetColumnbyIndex(cni)->GetName()) {
                    comp_ret = true;
                } else if (alias && *alias == child_nrsp->GetColumnbyIndex(cni)->GetName()) {
                    comp_ret = true;
                }
            }

            if (comp_ret == true) {
                //acs->SetIndexColumnSource(cni);
                my_referenced_columns.push_back(acs);
                arg_input->SetPositionForReferencedColumn(acs, (cni + 1));
                break;
            }
        }
    }

    std::vector<int> output_column_ids;

    auto& unique_keys = arg_input->GetTheChild()->GetUniqueKeys();

    for (size_t cni = 0; cni < child_nrsp->GetColumnCount(); cni++) {
        if (cni < all_select_items.size()) {
            auto select_item = (CommonBiaodashi*) (all_select_items[cni].get());
            if (!select_item->IsVisibleInResult()) {
                continue;
            }

            output_column_ids.emplace_back(cni + 1);
        }
    }

    for ( const auto& keys : unique_keys )
    {
        bool has_no_match = false;
        for ( const auto& key : keys )
        {
            if ( std::find( output_column_ids.begin(), output_column_ids.end(), key ) == output_column_ids.end() )
            {
                has_no_match = true;
                break;
            }
        }

        if ( !has_no_match )
        {
            arg_input->AddUniqueKeys( keys );
        }
    }

    arg_input->SetColumnOutputSequence(output_column_ids);
    /*for debug*/
//	for(int i = 0; i < my_referenced_columns.size(); i++)
//	{
//	    ////LOG(INFO) << "\tQueryTreeFormation::HandlingSortNode_RepositionColumns: " << i << " " << my_referenced_columns[i]->ToString() << "\n";
//	}

    //arg_input->SetReferencedColumnArray(my_referenced_columns);

}

void QueryTreeFormation::HandlingSortNode(SQLTreeNodePointer arg_input) {
    //TroubleHandler::showMessage("HandlingSortNode...\n");


    this->HandlingSortNode_ProcessingOwnColumns(arg_input);

    this->query_formation_handling_node(arg_input->GetTheChild());

    this->HandlingSortNode_FormingOutputColumns(arg_input);

    this->HandlingSortNode_RepositionColumns(arg_input);
}
/********************************************* A Node *********************************************/




/********************************************* A Node: BinaryJoin *********************************************/

void QueryTreeFormation::HandlingBinaryJoinNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {

    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();

    /*for debug*/
    for (size_t mri = 0; mri < my_required_columns.size(); mri++) {
        //LOG(INFO) << "HandlingBinaryJoinNode_ProcessingOwnColumns\t" << "my_required_columns\t" << mri << "\t" << my_required_columns[mri]->ToString() << "\n";
    }


    BiaodashiPointer join_condition = arg_input->GetJoinCondition();
    std::vector<ColumnShellPointer> join_condition_columns;
    if (join_condition) {
        join_condition_columns = ((CommonBiaodashi *)(join_condition.get()))->GetAllReferencedColumns();
    }
    BiaodashiPointer other_condition = arg_input->GetJoinOtherCondition();
    if (other_condition) {
        std::vector<ColumnShellPointer> other_condition_columns = ((CommonBiaodashi *) (other_condition.get()))->GetAllReferencedColumns();
        join_condition_columns.insert( join_condition_columns.end(), other_condition_columns.begin(), other_condition_columns.end() );
    }

    std::vector<ColumnShellPointer> final_columns = my_required_columns;


    for (size_t jci = 0; jci < join_condition_columns.size(); jci++) {
        bool found = false;
        for (size_t fi = 0; fi < final_columns.size(); fi++) {
            //if(join_condition_columns[jci] == final_columns[fi])
            if (join_condition_columns[jci]->GetTableName() == final_columns[fi]->GetTableName()
                &&
                join_condition_columns[jci]->GetColumnName() == final_columns[fi]->GetColumnName()) {
                found = true;
                break;
            }
        }

        if (!found) {
            final_columns.push_back(join_condition_columns[jci]);
        }
    }


    /*for debug*/
//	for(int fi = 0; fi < final_columns.size(); fi++)
//	{
//	    ////LOG(INFO) << "HandlingBinaryJoinNode_ProcessingOwnColumns\t" << "final_columns\t" << fi << "\t" << final_columns[fi]->ToString() << "\n";
//	}



    arg_input->SetReferencedColumnArray(final_columns);

    /*now we need to dispatch columns!  based on involved_table_list*/

    std::vector<ColumnShellPointer> left_required_column_array;
    std::vector<ColumnShellPointer> right_required_column_array;

    std::vector< ColumnShellPointer > left_columns_with_alias;
    std::vector< ColumnShellPointer > right_columns_with_alias;

    std::vector<BasicRelPointer> left_involved_table_list = arg_input->GetLeftChild()->GetInvolvedTableList();
    std::vector<BasicRelPointer> right_involved_table_list = arg_input->GetRightChild()->GetInvolvedTableList();

    auto query = std::dynamic_pointer_cast< SelectStructure >( arg_input->GetMyQuery() );
    const auto& alias_map = query->GetSpoolAlias();

    for (size_t fi = 0; fi < final_columns.size(); fi++) {
        ColumnShellPointer acsp = final_columns[fi];
        BasicRelPointer belong_to = nullptr;

        bool belong_left = false;
        bool belong_right = false;

        for (size_t li = 0; li < left_involved_table_list.size(); li++) {
            /**
             * 如果子节点是 sub-query，sub-query 里面的 table 对外是不可见的
             * FIXME: How to identify sub-query node
             */
            if (acsp->GetTableName() == left_involved_table_list[li]->GetMyOutputName()/* && arg_input->GetLeftChild()->GetType() != SQLTreeNodeType::Column_NODE*/) {

                left_required_column_array.push_back(acsp);
                belong_left = true;
                belong_to = left_involved_table_list[li];
                break;
            }
        }

        if (!belong_left) {
            for (size_t ri = 0; ri < right_involved_table_list.size(); ri++) {
                if (acsp->GetTableName() == right_involved_table_list[ri]->GetMyOutputName()) {
                    right_required_column_array.push_back(acsp);
                    belong_right = true;
                    belong_to = right_involved_table_list[ri];
                    break;
                }
            }
        }

        if (!belong_left && !belong_right) {
            auto it = alias_map.find( acsp->GetTableName() );
            if ( it != alias_map.cend() )
            {
                for ( const auto& table : left_involved_table_list )
                {
                    if ( it->second == table->GetMyOutputName() )
                    {
                        left_columns_with_alias.emplace_back(acsp);
                        belong_left = true;
                        belong_to = table;
                        break;
                    }
                }

                for ( const auto& table : right_involved_table_list )
                {
                    if ( it->second == table->GetMyOutputName() )
                    {
                        right_columns_with_alias.push_back(acsp);
                        belong_right = true;
                        belong_to = table;
                        break;
                    }
                }
            }
        }

        assert(belong_left || belong_right);

    }


    std::vector< ColumnShellPointer > required_columns;
    required_columns.assign( left_required_column_array.cbegin(), left_required_column_array.cend() );
    if ( !left_columns_with_alias.empty() )
    {
        required_columns.insert( required_columns.end(), left_columns_with_alias.cbegin(), left_columns_with_alias.cend() );
    }

    arg_input->GetLeftChild()->AddRequiredColumnArray( required_columns );

    if(arg_input->GetLeftChild()->GetSpoolId() > -1 && arg_input->GetLeftChild()->GetSameNode()){ //
        arg_input->GetLeftChild()->GetSameNode()->AddRequiredColumnArray(required_columns);
    }

    required_columns.clear();
    required_columns.assign( right_required_column_array.cbegin(), right_required_column_array.cend() );
    if ( !right_columns_with_alias.empty() )
    {
        required_columns.insert( required_columns.end(), right_columns_with_alias.cbegin(), right_columns_with_alias.cend() );
    }
    arg_input->GetRightChild()->AddRequiredColumnArray(required_columns);

    if(arg_input->GetRightChild()->GetSpoolId() > -1 && arg_input->GetRightChild()->GetSameNode()){

        arg_input->GetRightChild()->GetSameNode()->AddRequiredColumnArray( required_columns );
    }

    left_required_column_array = arg_input->GetLeftChild()->GetRequiredColumnArray();
    for (size_t lrcai = 0; lrcai < left_required_column_array.size(); lrcai++) {

        arg_input->SetPositionForReferencedColumn(left_required_column_array[lrcai], (lrcai + 1));
        if ( arg_input->GetSpoolId() != -1 )
        {
            // ColumnShellPointer to_remove = nullptr;
            for ( const auto& column : left_columns_with_alias )
            {
                auto it = alias_map.find( column->GetTable()->GetMyOutputName() );
                assert( it != alias_map.cend() );
                if ( it->second == left_required_column_array[lrcai]->GetTable()->GetMyOutputName() )
                {
                    arg_input->SetPositionForReferencedColumn( column, lrcai + 1 );
                    // to_remove = column;
                    break;
                }
            }
        }
    }

    right_required_column_array = arg_input->GetRightChild()->GetRequiredColumnArray();
    for (size_t rrcai = 0; rrcai < right_required_column_array.size(); rrcai++) {
        arg_input->SetPositionForReferencedColumn(right_required_column_array[rrcai], (0 - (rrcai + 1)));
        if ( arg_input->GetSpoolId() != -1 )
        {
            for ( const auto& column : right_columns_with_alias )
            {
                auto it = alias_map.find( column->GetTable()->GetMyOutputName() );
                assert( it != alias_map.cend() );
                if ( it->second == right_required_column_array[ rrcai ]->GetTable()->GetMyOutputName() )
                {
                    arg_input->SetPositionForReferencedColumn( column, - ( rrcai + 1 ) );
                }
            }
        }
    }

}

void QueryTreeFormation::HandlingBinaryJoinNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {

    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();

    /*for debug*/
    for (size_t di = 0; di < my_required_columns.size(); di++) {
        //LOG(INFO) << "di: HandlingBinaryJoinNode_FormingOutputColumns\t" << di << "\t" << my_required_columns[di]->ToString() <<  "\n";
    }

    NodeRelationStructurePointer left_child_nrsp = arg_input->GetLeftChild()->GetNodeRelationStructure();
    NodeRelationStructurePointer right_child_nrsp = arg_input->GetRightChild()->GetNodeRelationStructure();


    /*for debug*/
    ////LOG(INFO) << "\t\t\tleft_child_nrsp count = " << left_child_nrsp->GetColumnCount() << "\n";
    ////LOG(INFO) << "\t\t\tright_child_nrsp count = " << right_child_nrsp->GetColumnCount() << "\n";


    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    std::vector<int> my_column_output_sequence;

    // if ( arg_input->GetSpoolId() != -1 )
    // {
    //     auto query = std::dynamic_pointer_cast< SelectStructure >( arg_input->GetMyQuery() );
    //     if ( query->GetQueryContext()->type != QueryContextType::TheTopQuery && query->GetQueryContext()->GetParent() )
    //     {
    //         query = std::dynamic_pointer_cast< SelectStructure>( query->GetQueryContext()->GetParent()->select_structure );
    //     }
    //     const auto& alias_map = query->GetSpoolAlias();

    //     std::vector< ColumnShellPointer > new_required_columns;
    //     auto involved_tables = arg_input->GetInvolvedTableList();
    //     for ( const auto& column : my_required_columns )
    //     {
    //         auto it = alias_map.find( column->GetTable()->GetMyOutputName() );
    //         if ( it != alias_map.cend() )
    //         {
    //             bool found = false;
    //             bool alias_found = false;
    //             BasicRelPointer target_table;
    //             for ( const auto& table : involved_tables )
    //             {
    //                 if ( table->GetMyOutputName() == it->second )
    //                 {
    //                     alias_found = true;
    //                     target_table = table;
    //                     break;
    //                 }
    //                 else if ( table->GetMyOutputName() == it->first )
    //                 {
    //                     found = true;
    //                     break;
    //                 }
    //             }

    //             if ( !found )
    //             {
    //                 assert( alias_found );
    //                 auto new_column = std::make_shared< ColumnShell >( it->second, column->GetColumnName() );
    //                 new_column->SetTable( target_table );
    //                 new_required_columns.emplace_back( new_column );
    //                 arg_input->SetPositionForReferencedColumn( new_column, arg_input->GetPositionForReferencedColumn( column ) );
    //             }
    //             else
    //             {
    //                 new_required_columns.emplace_back( column );
    //             }
    //         }
    //         else
    //         {
    //             new_required_columns.emplace_back( column );
    //         }
    //     }

    //     my_required_columns.assign( new_required_columns.cbegin(), new_required_columns.cend() );
    //     arg_input->SetRequiredColumnArray( my_required_columns );
    // }
    bool convert_left_to_nullable = false;
    bool convert_right_to_nullable = false;
    switch( arg_input->GetJoinType() )
    {
        case JoinType::LeftJoin:
        case JoinType::LeftOuterJoin:
            convert_right_to_nullable = true;
            break;
        case JoinType::RightJoin:
        case JoinType::RightOuterJoin:
            convert_left_to_nullable = true;
            break;
        case JoinType::FullJoin:
        case JoinType::FullOuterJoin:
            convert_left_to_nullable = true;
            convert_right_to_nullable = true;
            break;
        default:
            break;
    }

    for (size_t mrci = 0; mrci < my_required_columns.size(); mrci++) {
        ColumnShellPointer x = my_required_columns[mrci];
        //LOG(INFO) << mrci << "\t\tHandlingBinaryJoinNode_FormingOutputColumns\t" << x->ToString() << "\tx\n";

        int tmp_location = arg_input->GetPositionForReferencedColumn(x);
        assert(tmp_location != 0);


        my_column_output_sequence.push_back(tmp_location);

        //LOG(INFO) << mrci << "\t\tHandlingBinaryJoinNode_FormingOutputColumns\t" << x->ToString() << "\t" << tmp_location << "\tx\n";
        NodeColumnStructurePointer ncsp;
        if (tmp_location > 0) {
            int left_index = tmp_location - 1;
            ncsp = left_child_nrsp->GetColumnbyIndex( left_index );
            my_nrsp->AddColumn(
                    convert_left_to_nullable ? ncsp->CloneAsNullable() : ncsp,
                    left_child_nrsp->GetColumnSourceTableNamebyIndex(left_index)
            );

        } else {
            int right_index = (0 - tmp_location) - 1;
            ncsp = right_child_nrsp->GetColumnbyIndex( right_index );
            my_nrsp->AddColumn(
                    convert_right_to_nullable ? ncsp->CloneAsNullable() : ncsp,
                    right_child_nrsp->GetColumnSourceTableNamebyIndex(right_index)
            );

        }

    }


    //LOG(INFO) << "*********************----------------------------\n";

    arg_input->SetColumnOutputSequence(my_column_output_sequence);
    arg_input->SetNodeRelationStructure(my_nrsp);

    arg_input->ClearUniqueKeys();

    bool keep_left_unique_keys = false;
    bool keep_right_unique_keys = false;

    bool left_as_hash = false, right_as_hash = false;
    if ( arg_input->CanUseHashJoin( left_as_hash, right_as_hash ) )
    {
        if ( left_as_hash )
        {
            keep_right_unique_keys = true;
        }
        else if ( right_as_hash )
        {
            keep_left_unique_keys = true;
        }
    }

    if ( arg_input->GetJoinType() == JoinType::SemiJoin || arg_input->GetJoinType() == JoinType::AntiJoin )
        keep_left_unique_keys = true;

    if ( keep_left_unique_keys )
    {
        for ( const auto& keys : arg_input->GetLeftChild()->GetUniqueKeys() )
        {
            bool has_no_match = false;
            std::vector< int > my_keys;
            for ( const auto& key : keys )
            {
                bool found = false;
                for ( size_t i = 0; i < my_column_output_sequence.size(); i++ )
                {
                    if ( key == my_column_output_sequence[ i ] )
                    {
                        found = true;
                        my_keys.emplace_back( i + 1 );
                    }
                }
                if ( !found )
                {
                    has_no_match = true;
                    break;
                }
            }

            if ( !has_no_match )
            {
                arg_input->AddUniqueKeys( my_keys );
            }
        }
    }

    if ( keep_right_unique_keys )
    {
        for ( const auto& keys : arg_input->GetRightChild()->GetUniqueKeys() )
        {
            bool has_no_match = false;
            std::vector< int > my_keys;
            for ( const auto& key : keys )
            {
                bool found = false;
                for ( size_t i = 0; i < my_column_output_sequence.size(); i++ )
                {
                    if ( key == -my_column_output_sequence[ i ] )
                    {
                        found = true;
                        my_keys.emplace_back( i + 1 );
                    }
                }
                if ( !found )
                {
                    has_no_match = true;
                    break;
                }
            }

            if ( !has_no_match )
            {
                arg_input->AddUniqueKeys( my_keys );
            }
        }
    }
    bool changed_to_nullable = false;
    BiaodashiPointer equal_condition = arg_input->GetJoinCondition();
    ResetColumnNullableInfo( equal_condition, left_child_nrsp, changed_to_nullable );
    ResetColumnNullableInfo( equal_condition, right_child_nrsp, changed_to_nullable );
    if( changed_to_nullable )
        std::dynamic_pointer_cast< CommonBiaodashi >( equal_condition )->SetIsNullable( true );

    changed_to_nullable = false;
    BiaodashiPointer other_condition = arg_input->GetJoinOtherCondition();
    ResetColumnNullableInfo( other_condition, left_child_nrsp, changed_to_nullable );
    ResetColumnNullableInfo( other_condition, right_child_nrsp, changed_to_nullable );
    if( changed_to_nullable )
        std::dynamic_pointer_cast< CommonBiaodashi >( other_condition )->SetIsNullable( true );

    auto left_hash_join_info = arg_input->GetLeftHashJoinInfo();
    for( auto expr : left_hash_join_info.EqualConditions )
    {
        changed_to_nullable = false;
        ResetColumnNullableInfo( expr, left_child_nrsp, changed_to_nullable );
        ResetColumnNullableInfo( expr, right_child_nrsp, changed_to_nullable );
        if( changed_to_nullable )
            std::dynamic_pointer_cast< CommonBiaodashi >( expr )->SetIsNullable( true );
    }
    changed_to_nullable = false;
    ResetColumnNullableInfo( left_hash_join_info.OtherCondition, left_child_nrsp, changed_to_nullable );
    ResetColumnNullableInfo( left_hash_join_info.OtherCondition, right_child_nrsp, changed_to_nullable );
    if( changed_to_nullable )
        std::dynamic_pointer_cast< CommonBiaodashi >( left_hash_join_info.OtherCondition )->SetIsNullable( true );

    auto right_hash_join_info = arg_input->GetRightHashJoinInfo();
    for( auto expr : right_hash_join_info.EqualConditions )
    {
        changed_to_nullable = false;
        ResetColumnNullableInfo( expr, left_child_nrsp, changed_to_nullable );
        ResetColumnNullableInfo( expr, right_child_nrsp, changed_to_nullable );
        if( changed_to_nullable )
            std::dynamic_pointer_cast< CommonBiaodashi >( expr )->SetIsNullable( true );
    }
    changed_to_nullable = false;
    ResetColumnNullableInfo( right_hash_join_info.OtherCondition, left_child_nrsp, changed_to_nullable );
    ResetColumnNullableInfo( right_hash_join_info.OtherCondition, right_child_nrsp, changed_to_nullable );
    if( changed_to_nullable )
        std::dynamic_pointer_cast< CommonBiaodashi >( right_hash_join_info.OtherCondition )->SetIsNullable( true );
}


//	    if(x->GetIndexTableSource() == 0)
//	    {
//		int left_index = x->GetIndexColumnSource();
//
//		my_nrsp->AddColumn(
//		    left_child_nrsp->GetColumnbyIndex(left_index),
//		    left_child_nrsp->GetColumnSourceTableNamebyIndex(left_index)
//		    );
//	    }
//	    else if(x->GetIndexTableSource() == 1)
//	    {
//		int right_index = x->GetIndexColumnSource();
//
//		my_nrsp->AddColumn(
//		    right_child_nrsp->GetColumnbyIndex(right_index),
//		    right_child_nrsp->GetColumnSourceTableNamebyIndex(right_index)
//		    );
//          }


//	for(int li = 0; li < left_child_nrsp->GetColumnCount(); li++)
//	{
//	    my_nrsp->AddColumn(
//		left_child_nrsp->GetColumnbyIndex(li),
//		left_child_nrsp->GetColumnSourceTableNamebyIndex(li)
//		);
//	}
//
//	for(int ri = 0; ri < right_child_nrsp->GetColumnCount(); ri++)
//	{
//	    my_nrsp->AddColumn(
//		right_child_nrsp->GetColumnbyIndex(ri),
//		right_child_nrsp->GetColumnSourceTableNamebyIndex(ri)
//		);
//	}



void QueryTreeFormation::HandlingBinaryJoinNode(SQLTreeNodePointer arg_input) {

    //TroubleHandler::showMessage("HandlingBinaryJoinNode...begin\n");


    this->HandlingBinaryJoinNode_ProcessingOwnColumns(arg_input);

    //LOG(INFO) << "left-->begin\n";
    this->query_formation_handling_node(arg_input->GetLeftChild());
    //LOG(INFO) << "left-->end\n";

    //LOG(INFO) << "right->begin\n";
    this->query_formation_handling_node(arg_input->GetRightChild());
    //LOG(INFO) << "right-->end\n";

    this->HandlingBinaryJoinNode_FormingOutputColumns(arg_input);

    //TroubleHandler::showMessage("HandlingBinaryJoinNode...done\n");


}
/********************************************* A Node *********************************************/



/********************************************* A Node: Filter *********************************************/


void QueryTreeFormation::HandlingFilterNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {

    if (arg_input->GetGroupHavingMode() == true) {
        //I am a having for group.

        assert(arg_input->GetTheChild()->GetType() == SQLTreeNodeType::Group_NODE);
        GroupbyStructurePointer gbsp = arg_input->GetTheChild()->GetMyGroupbyStructure();
        std::vector<BiaodashiPointer> abp = gbsp->GetAllPlaceHolderExprs();

        std::vector<ColumnShellPointer> final_columns;
        std::vector<int> positions;

        for (size_t abpi = 0; abpi < abp.size(); abpi++) {
            BiaodashiPointer bp_abpi = abp[abpi];
            CommonBiaodashi *expr_p = (CommonBiaodashi *) (bp_abpi.get());

            ColumnShellPointer csp = boost::get<ColumnShellPointer>(expr_p->GetContent());
            int my_position = std::stoi(csp->GetColumnName());

            final_columns.push_back(csp);
            positions.push_back(my_position);

        }

        for (size_t i = 0; i < final_columns.size(); i++) {
            arg_input->SetPositionForReferencedColumn(final_columns[i], (positions[i] + 1));
        }

        arg_input->SetReferencedColumnArray(final_columns);


        return;
    }

    //--------------------------------the normal processing----------------



    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();
    // std::map< ColumnShellPointer, std::string > columns_with_alias;
    // if ( arg_input->GetSpoolId() != -1 )
    // {
    //     const auto& alias_map = arg_input->GetSpoolAlias();
    //     std::vector< ColumnShellPointer > new_required_columns;
    //     for ( const auto& column : my_required_columns )
    //     {
    //         auto it = alias_map.find( column->GetTable()->GetMyOutputName() );
    //         if ( it != alias_map.cend() )
    //         {
    //             columns_with_alias[ column ] = it->second;
    //         }
    //         else
    //         {
    //             new_required_columns.emplace_back( column );
    //         }
    //     }

    //     if ( !columns_with_alias.empty() )
    //     {
    //         my_required_columns.assign( new_required_columns.cbegin(), new_required_columns.cend() );
    //         arg_input->SetRequiredColumnArray( my_required_columns );
    //     }
    // }

    /*for debug*/
    //LOG(INFO) << "\n\n\n";
    // for(int mri = 0; mri < my_required_columns.size(); mri++)
    // {
    //     //LOG(INFO) << "HandlingFilterNode_ProcessingOwnColumns\t" << "my_required_columns\t" << mri << "\t" << my_required_columns[mri]->ToString() << "\n";
    // }


    BiaodashiPointer filter_condition = arg_input->GetFilterStructure();

    /*for debug*/
    //LOG(INFO) << "\t\t FilterCondition: is !!!" << ((CommonBiaodashi* )(filter_condition.get()))->ToString() << "\n";

    std::vector<ColumnShellPointer> filter_condition_columns = ((CommonBiaodashi *) (filter_condition.get()))->GetAllReferencedColumns();

    std::vector<ColumnShellPointer> final_columns = my_required_columns;


    for (size_t fci = 0; fci < filter_condition_columns.size(); fci++) {
        bool found = false;
        for (size_t fi = 0; fi < final_columns.size(); fi++) {
            //if(filter_condition_columns[fci] == final_columns[fi])
            if (filter_condition_columns[fci]->GetTableName() == final_columns[fi]->GetTableName()
                &&
                filter_condition_columns[fci]->GetColumnName() == final_columns[fi]->GetColumnName()) {
                found = true;
                break;
            }
        }

        if (!found) {
            final_columns.push_back(filter_condition_columns[fci]);
        }
    }

    for (size_t i = 0; i < final_columns.size(); i++) {
        arg_input->SetPositionForReferencedColumn(final_columns[i], (i + 1));
        // for ( const auto& pair : columns_with_alias )
        // {
        //     if ( pair.second == final_columns[ i ]->GetTable()->GetMyOutputName() )
        //     {
        //         arg_input->SetPositionForReferencedColumn( pair.first, ( i + 1 ) );
        //     }
        // }
    }

    arg_input->SetReferencedColumnArray(final_columns);

    // arg_input->GetTheChild()->SetRequiredColumnArray(arg_input->GetReferencedColumnArray());
    arg_input->GetTheChild()->AddRequiredColumnArray(arg_input->GetReferencedColumnArray());
    if(arg_input->GetTheChild()->GetSpoolId() > -1 && arg_input->GetTheChild()->GetSameNode())
    {
        arg_input->GetTheChild()->GetSameNode()->AddRequiredColumnArray(arg_input->GetReferencedColumnArray());
    }
}

void QueryTreeFormation::HandlingFilterNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {

    if (arg_input->GetGroupHavingMode() == true) {
        //I am a having for group. I just forward the first n columns of my child node -- which must be a group

        NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

        NodeRelationStructurePointer child_nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();

        std::vector<int> my_column_output_sequence;

        for (int i = 0; i < arg_input->GetGroupOutputCount(); i++) {
            my_nrsp->AddColumn(
                    child_nrsp->GetColumnbyIndex(i),
                    child_nrsp->GetColumnSourceTableNamebyIndex(i)
            );

            my_column_output_sequence.push_back((i) + 1);

        }

        arg_input->SetColumnOutputSequence(my_column_output_sequence);
        arg_input->SetNodeRelationStructure(my_nrsp);

        GroupbyStructurePointer gbsp = arg_input->GetTheChild()->GetMyGroupbyStructure();
        std::vector< BiaodashiPointer > all_exprs = gbsp->GetAllPlaceHolderExprs();
        for( auto expr : all_exprs )
        {
            bool changed_to_nullable = false;
            ResetColumnNullableInfo( expr, child_nrsp, changed_to_nullable );
            if( changed_to_nullable )
                std::dynamic_pointer_cast< CommonBiaodashi >( expr )->SetIsNullable( true );
        }
        return;
    }

    //--------------------------------the normal processing----------------
    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();

    // if ( arg_input->GetSpoolId() != -1 )
    // {
    //     const auto& alias_map = arg_input->GetSpoolAlias();
    //     std::vector< ColumnShellPointer > new_required_columns;
    //     for ( const auto& column : my_required_columns )
    //     {
    //         auto table = column->GetTable();
    //         auto table_output_name = table->GetMyOutputName();
    //         auto it = alias_map.find( table_output_name );

    //         if ( it == alias_map.cend() )
    //         {
    //             new_required_columns.emplace_back( column );
    //             continue;
    //         }
    //     }
    // }


    /*for debug*/
//	for(int di = 0; di < my_required_columns.size(); di++)
//	{
//	    ////LOG(INFO) << "!!!HandlingFilterNode_FormingOutputColumns\t" << di << "\t" << my_required_columns[di]->ToString() << "\n";
//	}

    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    NodeRelationStructurePointer child_nrsp = arg_input->GetTheChild()->GetNodeRelationStructure();

    std::vector<int> my_column_output_sequence;


    for (size_t mrci = 0; mrci < my_required_columns.size(); mrci++) {

        my_nrsp->AddColumn(
                child_nrsp->GetColumnbyIndex(mrci),
                child_nrsp->GetColumnSourceTableNamebyIndex(mrci)
        );

        my_column_output_sequence.push_back((mrci) + 1);


    }

//	for(int i = 0; i < child_nrsp->GetColumnCount(); i++)
//	{
//	    my_nrsp->AddColumn(
//		child_nrsp->GetColumnbyIndex(i),
//		child_nrsp->GetColumnSourceTableNamebyIndex(i)
//		);
//	}

    arg_input->SetColumnOutputSequence(my_column_output_sequence);
    arg_input->SetNodeRelationStructure(my_nrsp);

    for ( const auto& keys : arg_input->GetTheChild()->GetUniqueKeys() )
    {
        bool has_no_match = false;
        for ( const auto& key : keys )
        {
            if ( std::find( my_column_output_sequence.begin(), my_column_output_sequence.end(), key ) == my_column_output_sequence.end() )
            {
                has_no_match = true;
                break;
            }
        }

        if ( !has_no_match )
        {
            arg_input->AddUniqueKeys( keys );
        }
    }

    BiaodashiPointer filter_condition = arg_input->GetFilterStructure();
    bool changed_to_nullable = false;
    ResetColumnNullableInfo( filter_condition, child_nrsp, changed_to_nullable );
    if( changed_to_nullable )
        std::dynamic_pointer_cast< CommonBiaodashi >( filter_condition )->SetIsNullable( true );
}

void QueryTreeFormation::HandlingFilterNode(SQLTreeNodePointer arg_input) {
    //TroubleHandler::showMessage("HandlingFilterNode...\n");


    this->HandlingFilterNode_ProcessingOwnColumns(arg_input);
    this->query_formation_handling_node(arg_input->GetTheChild());
    this->HandlingFilterNode_FormingOutputColumns(arg_input);
}
/********************************************* A Node *********************************************/


/********************************************* A Node: Table *********************************************/
void QueryTreeFormation::HandlingTableNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {

    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();
    arg_input->SetReferencedColumnArray(my_required_columns);
    /*we set position when forming outputcolumns!*/
}


void QueryTreeFormation::HandlingTableNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {

    /*for debug*/
    // for(int di = 0; di < my_required_columns.size(); di++)
    // {
    //     LOG(INFO) << "***HandlingTableNode_FormingOutputColumns\t" << di << "\t" << my_required_columns[di]->ToString() << "\n";
    // }

    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    BasicRelPointer the_basic_rel = arg_input->GetBasicRel();
    RelationStructurePointer the_relation_structure = the_basic_rel->GetRelationStructure();
    auto& constraints = the_relation_structure->GetConstraints();

    std::vector< std::string > primary_keys;
    vector< vector< string > > uniq_keys;
    for ( const auto& it : constraints )
    {
        if ( it.second->type == schema::TableConstraintType::PrimaryKey )
        {
            primary_keys.assign( it.second->keys.cbegin(), it.second->keys.cend() );
        }
        else if ( it.second->type == schema::TableConstraintType::UniqueKey )
        {
            vector< string > uniq_key;
            uniq_key.assign( it.second->keys.cbegin(), it.second->keys.cend() );
            uniq_keys.emplace_back( uniq_key );
        }
    }

    std::vector<int> my_column_output_sequence;

    size_t matched_primary_keys_count = 0;
    std::vector< int > primary_key_ids( primary_keys.size() );
    vector< size_t > matched_uniq_keys_count( uniq_keys.size(), 0 );
    vector< vector< int > > uniq_key_ids( uniq_keys.size() );

    std::vector<ColumnShellPointer> my_required_columns = arg_input->GetRequiredColumnArray();

    for (size_t mrci = 0; mrci < my_required_columns.size(); mrci++) {

        ColumnShellPointer acsp = my_required_columns[mrci];

        int column_index = -1;

        /*which column I should use?*/
        for (size_t i = 0; i < the_relation_structure->GetColumnCount(); i++) {

            if (acsp->GetColumnName() == the_relation_structure->GetColumn(i)->GetName()) {
                column_index = i;
                break;
            }
        }

        //debug
        // if(column_index == -1)
        // {
        // 	LOG(INFO) << arg_input->ToString(1) << "\n";
        // 	LOG(INFO) << acsp->GetColumnName() << "\n";
        // }
        assert(column_index != -1);

        my_column_output_sequence.push_back((column_index) + 1);

        for ( size_t i = 0; i < primary_keys.size(); i++ )
        {
            if ( acsp->GetColumnName() == primary_keys[ i ] )
            {
                primary_key_ids[ i ] = mrci + 1;
                matched_primary_keys_count++;
                break;
            }
        }

        for ( size_t i = 0; i < uniq_keys.size(); i++ )
        {
            const auto &uniq_key_cols = uniq_keys[ i ];
            for ( size_t j = 0; j < uniq_key_cols.size(); ++j )
            {
                if ( acsp->GetColumnName() == uniq_key_cols[ j ] )
                {
                    uniq_key_ids[ i ].push_back( mrci + 1 );
                    matched_uniq_keys_count[ i ]++;
                    break;
                }
            }
        }

        arg_input->SetPositionForReferencedColumn(acsp, (column_index + 1));

        my_nrsp->AddColumn(
                NodeColumnStructure::__createInstanceFromColumnStructure(
                        the_relation_structure->GetColumn(column_index)),
                the_basic_rel->GetMyOutputName()
        );

    }

    if ( matched_primary_keys_count > 0 && matched_primary_keys_count == primary_keys.size() )
    {
        arg_input->AddUniqueKeys( primary_key_ids );
    }

    for ( size_t i = 0; i < uniq_keys.size(); i++ )
    {
        if ( matched_uniq_keys_count[ i ] > 0 && matched_uniq_keys_count[ i ] == uniq_keys[ i ].size() )
        {
            arg_input->AddUniqueKeys( uniq_key_ids[ i ] );
        }
    }

    arg_input->SetColumnOutputSequence(my_column_output_sequence);
    arg_input->SetNodeRelationStructure(my_nrsp);

}

void QueryTreeFormation::HandlingTableNode(SQLTreeNodePointer arg_input) {
    HandlingTableNode_ProcessingOwnColumns(arg_input);
    HandlingTableNode_FormingOutputColumns(arg_input);
}


/********************************************* A Node *********************************************/


/********************************************* A Node: SetOperation *********************************************/

void QueryTreeFormation::HandlingSetOperationNode_ProcessingOwnColumns(SQLTreeNodePointer arg_input) {
    /**/
    // do nothing
}

void QueryTreeFormation::HandlingSetOperationNode_FormingOutputColumns(SQLTreeNodePointer arg_input) {
    /**/
    //my output is exactly my first child's output
    //
    NodeRelationStructurePointer my_nrsp = std::make_shared<NodeRelationStructure>();

    NodeRelationStructurePointer child_nrsp = arg_input->GetLeftChild()->GetNodeRelationStructure();
//	LOG(INFO) << "HandlingSetOperationNode_FormingOutputColumns: " << arg_input->GetForwardedColumnCount() << "\n";
//	for(int i = 0; i < arg_input->GetForwardedColumnCount(); i++)
    for (size_t i = 0; i < child_nrsp->GetColumnCount(); i++) {
        my_nrsp->AddColumn(
                child_nrsp->GetColumnbyIndex(i),
                child_nrsp->GetColumnSourceTableNamebyIndex(i)
        );

    }
    //LOG(INFO) << "HandlingSetOperationNode_FormingOutputColumns: [" << my_nrsp->ToString() << "]" << "\n";
    arg_input->SetNodeRelationStructure(my_nrsp);

}


void QueryTreeFormation::HandlingSetOperationNode(SQLTreeNodePointer arg_input) {

    //TroubleHandler::showMessage("HandlingSetOperationNode...begin\n");


    this->HandlingSetOperationNode_ProcessingOwnColumns(arg_input);

    //LOG(INFO) << "left-->begin\n";
    this->query_formation_handling_node(arg_input->GetLeftChild());
    //LOG(INFO) << "left-->end\n";

    //LOG(INFO) << "right->begin\n";
    this->query_formation_handling_node(arg_input->GetRightChild());
    //LOG(INFO) << "right-->end\n";

    this->HandlingSetOperationNode_FormingOutputColumns(arg_input);

    //TroubleHandler::showMessage("HandlingSetOperationNode...done\n");


}


/********************************************* A Node *********************************************/


void QueryTreeFormation::query_formation_handling_node(SQLTreeNodePointer arg_input) {
    if (arg_input == nullptr)
        return;

    switch (arg_input->GetType()) {

        case SQLTreeNodeType::Limit_NODE: {
            HandlingLimitNode( arg_input );
            break;
        }

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
                this->HandlingSetOperationNode(arg_input);
            }
            break;

        default:
        {
            ARIES_ASSERT(0, "UnSupported node type: " + std::to_string((int) arg_input->GetType()));
        }

    }


    arg_input->SetTreeFormedTag(true);

}

void QueryTreeFormation::ResetColumnNullableInfo( BiaodashiPointer expr, NodeRelationStructurePointer nrsp, bool& changed_to_nullable )
{
    if( expr )
    {
        CommonBiaodashiPtr expr_p = std::dynamic_pointer_cast< CommonBiaodashi >( expr );
        if( expr_p->GetType() == BiaodashiType::Lie )
        {
            ColumnShellPointer tmp_column_shell = boost::get< ColumnShellPointer >( expr_p->GetContent() );
            auto my_csp = tmp_column_shell->GetColumnStructure();
            for( size_t i = 0; i < nrsp->GetColumnCount(); ++i )
            {
                auto col = nrsp->GetColumnbyIndex( i );
                auto tmp_csp = col->GetPossibleRoot();
                if( tmp_csp == my_csp )
                {
                    expr_p->SetIsNullable( col->IsNullable() );
                    if( col->IsNullable() )
                        changed_to_nullable = true;
                    break;
                }
            }
        }
        else 
        {
            for( size_t i = 0; i < expr_p->GetChildrenCount(); i++ )
                ResetColumnNullableInfo( expr_p->GetChildByIndex( i ), nrsp, changed_to_nullable );
        }
    }
}


}//namespace
