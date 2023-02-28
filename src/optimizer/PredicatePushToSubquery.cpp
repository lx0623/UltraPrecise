/*
 * PredicatePushToSubquery.cpp
 *
 *  Created on: Jul 20, 2020
 *      Author: lichi
 */

#include "PredicatePushToSubquery.h"
#include "frontend/BiaodashiAuxProcessor.h"
#include <random>
#include <set>

namespace aries
{

    PredicatePushToSubquery::PredicatePushToSubquery()
    {
    }

    PredicatePushToSubquery::~PredicatePushToSubquery()
    {
    }

    string PredicatePushToSubquery::ToString()
    {
        return string( "PredicatePushToSubquery Pushdown" );
    }

    SQLTreeNodePointer PredicatePushToSubquery::OptimizeTree( SQLTreeNodePointer arg_input )
    {
        assert( arg_input );
        QueryContextPointer mainQueryContext = dynamic_pointer_cast< SelectStructure >( arg_input->GetMyQuery() )->GetQueryContext();
        if( mainQueryContext->subquery_context_array.size() != 0 )
        {
            // find main query's where condition
            vector< SQLTreeNodePointer > mainFilterNodes;
            FindFilterNodes( arg_input, mainFilterNodes );
            for( const auto& node : mainFilterNodes )
            {
                // there is a "where" condition in query.
                // we break the condition into exprs
                FilterNodeInfo mainFilterInfo = ProcessMainFilterNode( node );

                // create new filter condition for subquery
                for( auto& query : mainFilterInfo.Subqueries )
                {
                    auto& outerColumns = query->GetQueryContext()->outer_column_array;
                    if( !outerColumns.empty() )
                    {
                        // all outer columns must be from same table
                        string outTableName = outerColumns[0]->GetTableName();
                        bool bFromSameTable = true;
                        for( auto& col : outerColumns )
                        {
                            if( outTableName != col->GetTableName() )
                            {
                                bFromSameTable = false;
                                break;
                            }
                        }
                        if( !bFromSameTable )
                            continue;

                        // find subquery's where condition
                        vector< SQLTreeNodePointer > subFilterNodes;
                        FindFilterNodes( query->GetQueryPlanTree(), subFilterNodes );

                        // we can only handle if subquery have 1 filter node no more!
                        if( subFilterNodes.size() == 1 )
                        {
                            vector< CommonBiaodashiPtr > subExprs = ExtractExprOfFilterNode( subFilterNodes[0] );
                            set< ColumnShellPointer, ColumnShellComparator > cols;
                            cols.insert( outerColumns.begin(), outerColumns.end() );
                            if( outerColumns.size() > 1 )
                            {
                                for( const auto& col : cols )
                                    SimplifyOuterColumnExpr( subExprs, col );
                            }
                            PushConditionToFilterNode( subFilterNodes[0], outerColumns, mainFilterInfo, subExprs );
                        }
                    }
                }
            }
        }
        return arg_input;
    }

    void PredicatePushToSubquery::PushConditionToFilterNode( SQLTreeNodePointer& filterNode, const vector< ColumnShellPointer >& outerColumns,
            const FilterNodeInfo& filterInfo, const vector< CommonBiaodashiPtr >& subqueryExprs ) const
    {
        ExprToPush exprToPush = FindExprsToPushDown( outerColumns, filterInfo, subqueryExprs,
                dynamic_pointer_cast< CommonBiaodashi >( filterNode->GetFilterStructure() )->GetInvolvedTableList() );

        if( !exprToPush.FilterExpr.empty() )
        {
            assert( !outerColumns.empty() );
            QueryContextPointer subqueryContext = dynamic_pointer_cast< SelectStructure >( filterNode->GetMyQuery() )->GetQueryContext();

            // find all tables need to push down
            set< BasicRelPointer, BasicRelComparator > tablesToCreate;
            for( auto& expr : exprToPush.FilterExpr )
            {
                assert( expr->GetInvolvedTableList().size() == 1 );
                tablesToCreate.insert( expr->GetInvolvedTableList()[0] );
            }

            for( auto& expr : exprToPush.JoinExpr )
            {
                auto tableList = expr->GetInvolvedTableList();
                assert( tableList.size() == 2 );
                tablesToCreate.insert( tableList[0] );
                tablesToCreate.insert( tableList[1] );
            }

            //make new table node and join node
            auto outerColumnTable = outerColumns[0]->GetTable();
            string outerTableId = outerColumnTable->GetID();
            SQLTreeNodePointer oldChildNode = filterNode->GetTheChild();
            SQLTreeNodePointer newChildNode = oldChildNode;
            BasicRelPointer newOuterTableInSubquery;
            map< BasicRelPointer, BasicRelPointer > outerInnerTableMapping;
            for( auto& table : tablesToCreate )
            {
                SQLTreeNodePointer newTableNode;
                // we can share BasicRelPointer with the outer query for other tables.( same id, same alias, same table )
                // !!!except that we need to make a alias for the outer column's table in subquery
                // later we can use the alias to create a new equal commonbiaodashi for the subqury
                if( table->GetID() == outerTableId )
                {
                    // we need a BasicRelPointer clone here because of different alias name in sub query
                    newOuterTableInSubquery = CloneBasicRelUsingAlias( table, GenerateRandomTableAlias( table->GetID() ) );
                    newTableNode = SQLTreeNode::makeTreeNode_Table( filterNode->GetMyQuery(), newOuterTableInSubquery );
                    outerInnerTableMapping.insert(
                    { table, newOuterTableInSubquery } );
                }
                else
                {
                    // share BasicRelPointer with outer query
                    newTableNode = SQLTreeNode::makeTreeNode_Table( filterNode->GetMyQuery(), table );
                    outerInnerTableMapping.insert(
                    { table, table } );
                }

                if ( oldChildNode->GetType() == SQLTreeNodeType::InnerJoin_NODE )
                {
                    oldChildNode->AddChild( newTableNode );
                    newTableNode->SetParent( oldChildNode );
                }
                else
                {
                    CommonBiaodashiPtr joinCondition = std::make_shared< CommonBiaodashi >( BiaodashiType::Zhenjia, true );
                    joinCondition->SetValueType( BiaodashiValueType::BOOL );
                    ExprContextPointer exprContext = std::make_shared< ExprContext >( ExprContextType::JoinOnExpr, nullptr, subqueryContext,
                            subqueryContext->expr_context, 0 );
                    joinCondition->SetExprContext( exprContext );
                    auto newJoinNode = SQLTreeNode::makeTreeNode_InnerJoin( filterNode->GetMyQuery() );
                    // SQLTreeNodePointer newJoinNode = SQLTreeNode::makeTreeNode_BinaryJoin( filterNode->GetMyQuery(), JoinType::InnerJoin, joinCondition,
                    //         nullptr );
                    // newJoinNode->SetLeftChild( newChildNode );
                    // newJoinNode->SetRightChild( newTableNode );
                    newJoinNode->AddChild( newChildNode );
                    newJoinNode->AddChild( newTableNode );

                    newChildNode->SetParent( newJoinNode );
                    newTableNode->SetParent( newJoinNode );

                    newChildNode = newJoinNode;
                }
            }
            assert( newOuterTableInSubquery );
            vector< BiaodashiPointer > newFilterConditions;
            std::copy( subqueryExprs.begin(), subqueryExprs.end(), std::back_inserter( newFilterConditions ) );

            ExprContextPointer whereExprContext = std::make_shared< ExprContext >( ExprContextType::WhereExpr, nullptr, subqueryContext,
                    subqueryContext->expr_context, 0 );
            for( auto& expr : exprToPush.FilterExpr )
                newFilterConditions.push_back( expr->CloneUsingNewExprContext( whereExprContext ) );
            for( auto& expr : exprToPush.JoinExpr )
                newFilterConditions.push_back( expr->CloneUsingNewExprContext( whereExprContext ) );

            string oldOuterTableNameInSubquery = outerColumnTable->GetMyOutputName();
            for( auto& it : outerInnerTableMapping )
            {
                for( auto& expr : newFilterConditions )
                {
                    dynamic_pointer_cast< CommonBiaodashi >( expr )->ResetReferencedColumnsInfo( it.first->GetMyOutputName(), it.second,
                            subqueryContext->query_level );
                }
            }
            BiaodashiAuxProcessor processer;
            for( auto& col : outerColumns )
            {
                //create new expr
                ColumnShellPointer colNew = processer.make_column_shell( newOuterTableInSubquery, col->GetLocationInTable(),
                        subqueryContext->query_level );

                colNew->SetIsPrimaryKey( col->IsPrimaryKey() );
                colNew->SetIsUnique( col->IsUnique() );
                colNew->SetTableName( newOuterTableInSubquery->GetMyOutputName() );

                BiaodashiPointer exprLeft = processer.make_biaodashi_lie( colNew );
                dynamic_pointer_cast< CommonBiaodashi >( exprLeft )->SetExprContext( whereExprContext );

                BiaodashiPointer exprRight = processer.make_biaodashi_lie( col );
                dynamic_pointer_cast< CommonBiaodashi >( exprRight )->SetExprContext( whereExprContext );

                BiaodashiPointer exprEqual = processer.make_biaodashi_compare_equal( exprLeft, exprRight );
                dynamic_pointer_cast< CommonBiaodashi >( exprEqual )->SetExprContext( whereExprContext );

                newFilterConditions.push_back( exprEqual );
            }

            BiaodashiAuxProcessor exprProcesser;
            auto newFilterStructure = exprProcesser.make_biaodashi_from_and_list( newFilterConditions );
            dynamic_pointer_cast< CommonBiaodashi >( newFilterStructure )->ObtainReferenceTableInfo();
            filterNode->SetFilterStructure( newFilterStructure );
            if ( oldChildNode != newChildNode )
            {
                filterNode->CompletelyResetAChild( oldChildNode, newChildNode );
                newChildNode->SetParent( filterNode );
            }
        }
    }

    PredicatePushToSubquery::FilterNodeInfo PredicatePushToSubquery::ProcessMainFilterNode( const SQLTreeNodePointer& filterNode ) const
    {
        FilterNodeInfo result;
        vector< CommonBiaodashiPtr > exprs = ExtractExprOfFilterNode( filterNode );
        for( auto& expr : exprs )
        {
            vector< AbstractQueryPointer > subqueries;
            vector< ColumnShellPointer > columns;
            expr->ObtainReferenceTableAndOtherInfo( subqueries, columns );
            vector< BasicRelPointer > tables = expr->GetInvolvedTableList();
            if( tables.size() == 1 && subqueries.empty() )
            {
                //only 1 table involved, save it for filter condition check
                result.ExprUsingOneTable.push_back( expr );
                result.ExprReferencedColumns.insert(
                { expr, std::move( columns ) } );
            }
            else if( tables.size() == 2 && subqueries.empty() )
            {
                //2 tables involved, save it for PF key join check
                if( IsTwoColumnEqualExpr( expr ) )
                {
                    result.ExprUsingTwoTable.push_back( expr );
                    result.ExprReferencedColumns.insert(
                    { expr, std::move( columns ) } );
                }
            }

            //we can only handle just 1 subquery in an expr
            if( subqueries.size() == 1 )
            {
                // the expr must be a comparision
                if( expr->GetType() == BiaodashiType::Bijiao )
                    result.Subqueries.push_back( dynamic_pointer_cast< SelectStructure >( subqueries[0] ) );
            }
        }
        return result;
    }

    vector< CommonBiaodashiPtr > PredicatePushToSubquery::ExtractExprOfFilterNode( const SQLTreeNodePointer& filterNode ) const
    {
        vector< CommonBiaodashiPtr > result;
        BiaodashiAuxProcessor exprProcesser;
        BiaodashiPointer filterExpr = filterNode->GetFilterStructure();
        dynamic_pointer_cast< CommonBiaodashi >( filterExpr )->ObtainReferenceTableInfo();                // obtain table info for later use.
        vector< BiaodashiPointer > exprs = exprProcesser.generate_and_list( filterExpr );
        for( auto expr : exprs )
            result.push_back( dynamic_pointer_cast< CommonBiaodashi >( expr ) );
        return result;
    }

    void PredicatePushToSubquery::FindFilterNodes( const SQLTreeNodePointer& node, vector< SQLTreeNodePointer >& output ) const
    {
        if( node == nullptr )
            return;

        switch( node->GetType() )
        {
            case SQLTreeNodeType::Column_NODE:
            case SQLTreeNodeType::Group_NODE:
            case SQLTreeNodeType::Sort_NODE:
            case SQLTreeNodeType::Limit_NODE:
            {
                FindFilterNodes( node->GetTheChild(), output );
                break;
            }
            case SQLTreeNodeType::SetOp_NODE:
            {
                FindFilterNodes( node->GetLeftChild(), output );
                FindFilterNodes( node->GetRightChild(), output );
                break;
            }
            case SQLTreeNodeType::Filter_NODE:
            {
                output.push_back( node );
                FindFilterNodes( node->GetTheChild(), output );
                break;
            }
            case SQLTreeNodeType::BinaryJoin_NODE:
            {
                FindFilterNodes( node->GetLeftChild(), output );
                FindFilterNodes( node->GetRightChild(), output );
                break;
            }
            case SQLTreeNodeType::Table_NODE:
            case SQLTreeNodeType::InnerJoin_NODE:
                break;
            default:
            {
                ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string( ( int )node->GetType() ) );
            }
        }
    }

    //only A.a = B.b can be a candidate to push( hash join )
    bool PredicatePushToSubquery::IsTwoColumnEqualExpr( const CommonBiaodashiPtr& expr ) const
    {
        bool result = false;
        if( expr->GetType() == BiaodashiType::Bijiao
                && static_cast< ComparisonType >( boost::get< int >( expr->GetContent() ) ) == ComparisonType::DengYu )
        {
            if( expr->GetChildrenCount() == 2 )
                result = dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 0 ) )->GetType() == BiaodashiType::Lie
                        && dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 1 ) )->GetType() == BiaodashiType::Lie;
        }
        return result;
    }

    bool PredicatePushToSubquery::AreFromSameTable( ColumnShellPointer col1, ColumnShellPointer col2 ) const
    {
        assert( col1 && col2 );
        return col1->GetTable()->GetDb() == col2->GetTable()->GetDb() && col1->GetTable()->GetID() == col2->GetTable()->GetID();
    }

    pair< int, vector< bool > > PredicatePushToSubquery::FindFilterToPush( const ColumnShellPointer& col, const FilterNodeInfo& filterInfo,
            const vector< CommonBiaodashiPtr >& subqueryExprs, const vector< BasicRelPointer >& subqueryTables ) const
    {
        int total = 0;
        vector< bool > pushedFilterFlags;

        pushedFilterFlags.resize( filterInfo.ExprUsingOneTable.size(), false );
        int index = 0;
        for( auto& expr : filterInfo.ExprUsingOneTable )
        {
            auto it = filterInfo.ExprReferencedColumns.find( expr );
            assert( it != filterInfo.ExprReferencedColumns.end() );
            auto tableList = expr->GetInvolvedTableList();
            assert( tableList.size() == 1 );
            //the referenced column should be same as subquery's outer column ( in this case, col )
            if( AreFromSameTable( col, it->second[0] ) && !IsExprExistsInSubquery( expr, subqueryExprs )
                    && !IsTableExistsInSubquery( tableList[0], subqueryTables ) )
            {
                pushedFilterFlags[index] = true;
                ++total;
            }
            ++index;
        }

        return
        {   total, pushedFilterFlags};
    }

    vector< CommonBiaodashiPtr > PredicatePushToSubquery::FindOtherConditionToPush( const ColumnShellPointer& col, const FilterNodeInfo& filterInfo,
            vector< bool >& pushedFilterFlags, const vector< CommonBiaodashiPtr >& subqueryExprs,
            const vector< BasicRelPointer >& subqueryTables ) const
    {
        vector< CommonBiaodashiPtr > result;
        for( auto& expr : filterInfo.ExprUsingTwoTable )
        {
            auto it = filterInfo.ExprReferencedColumns.find( expr );
            assert( it != filterInfo.ExprReferencedColumns.end() && it->second.size() == 2 );
            auto& col1 = it->second[0];
            auto& col2 = it->second[1];
            ColumnShellPointer colToCheck;
            // find a unique key join condition which involves the outer column's table.
            // the join result will be no more than the outer column's table's row count.
            if( AreFromSameTable( col, col1 ) )
            {
                // the join is act like a filter
                if( col2->IsUnique() )
                    colToCheck = col2;
            }
            else if( AreFromSameTable( col, col2 ) )
            {
                if( col1->IsUnique() )
                    colToCheck = col1;
            }

            if( colToCheck )
            {
                // if we found a filter condition in unique key's table, we can push it with the join condition to sub query.
                auto filterExprsToPush = FindFilterToPush( colToCheck, filterInfo, subqueryExprs, subqueryTables );
                if( filterExprsToPush.first > 0 )
                {
                    result.push_back( expr );
                    MergeFlags( pushedFilterFlags, filterExprsToPush.second );
                }
            }
        }
        return result;
    }

    PredicatePushToSubquery::ExprToPush PredicatePushToSubquery::FindExprsToPushDown( const vector< ColumnShellPointer >& outerColumns,
            const FilterNodeInfo& filterInfo, const vector< CommonBiaodashiPtr >& subqueryExprs,
            const vector< BasicRelPointer >& subqueryTables ) const
    {
        vector< CommonBiaodashiPtr > filterCondition;
        vector< CommonBiaodashiPtr > joinCondition;
        vector< bool > pushedFilterFlags;
        pushedFilterFlags.resize( filterInfo.ExprUsingOneTable.size(), false );
        for( auto& col : outerColumns )
        {
            //process expr using one table
            auto filterExprsToPush = FindFilterToPush( col, filterInfo, subqueryExprs, subqueryTables );

            //process expr using two table
            vector< CommonBiaodashiPtr > otherExprs = FindOtherConditionToPush( col, filterInfo, filterExprsToPush.second, subqueryExprs,
                    subqueryTables );
            std::copy( otherExprs.begin(), otherExprs.end(), std::back_inserter( joinCondition ) );

            MergeFlags( pushedFilterFlags, filterExprsToPush.second );
        }
        for( size_t i = 0; i < filterInfo.ExprUsingOneTable.size(); ++i )
        {
            if( pushedFilterFlags[i] )
                filterCondition.push_back( filterInfo.ExprUsingOneTable[i] );
        }
        return
        {   filterCondition, joinCondition};
    }

    bool PredicatePushToSubquery::IsExprExistsInSubquery( const CommonBiaodashiPtr& expr, const vector< CommonBiaodashiPtr >& subqueryExprs ) const
    {
        bool result = false;
        auto exprNormal = expr->Normalized();
        for( const auto& subExpr : subqueryExprs )
        {
            if( *exprNormal == *subExpr->Normalized() )
            {
                result = true;
                break;
            }
        }
        return result;
    }

    bool PredicatePushToSubquery::IsTableExistsInSubquery( const BasicRelPointer& table, const vector< BasicRelPointer >& subqueryTables ) const
    {
        bool result = false;
        for( const auto& subTable : subqueryTables )
        {
            if( table->GetID() == subTable->GetID() && table->GetDb() == subTable->GetDb() )
            {
                result = true;
                break;
            }
        }
        return result;
    }

    void PredicatePushToSubquery::MergeFlags( vector< bool >& inout, const vector< bool >& input ) const
    {
        assert( input.size() == inout.size() );
        for( size_t i = 0; i < inout.size(); ++i )
            inout[i] = inout[i] || input[i];
    }

    BasicRelPointer PredicatePushToSubquery::CloneBasicRelUsingAlias( const BasicRelPointer& table, const string& alias ) const
    {
        BasicRelPointer result = std::make_shared< BasicRel >( false, table->GetID(), nullptr, nullptr );
        result->SetDb( table->GetDb() );
        result->SetRelationStructure( table->GetRelationStructure() );
        result->SetUnderlyingTable( table->GetPhysicalTable() );
        result->ResetAlias( alias );
        return result;
    }

    string PredicatePushToSubquery::GenerateRandomTableAlias( const string& tableId ) const
    {
        std::random_device rd;
        return tableId + "_" + std::to_string( rd() );
        // return tableId;
    }

    bool PredicatePushToSubquery::AreSameColumnShell( const ColumnShellPointer& col1, const ColumnShellPointer& col2 ) const
    {
        return col1->GetTableName() == col2->GetTableName() && col1->GetColumnName() == col2->GetColumnName();
    }

    void PredicatePushToSubquery::SimplifyOuterColumnExpr( vector< CommonBiaodashiPtr >& exprs, const ColumnShellPointer& outerColumn ) const
    {
        ColumnShellPointer colToUse;
        vector< CommonBiaodashiPtr > colToReplace;
        for( auto& expr : exprs )
        {
            if( !colToUse && IsTwoColumnEqualExpr( expr ) )
            {
                auto leftCol = boost::get< ColumnShellPointer >(
                        dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 0 ) )->GetContent() );
                auto rightCol = boost::get< ColumnShellPointer >(
                        dynamic_pointer_cast< CommonBiaodashi >( expr->GetChildByIndex( 1 ) )->GetContent() );
                if( AreSameColumnShell( leftCol, outerColumn ) )
                    colToUse = rightCol;
                else if( AreSameColumnShell( rightCol, outerColumn ) )
                    colToUse = leftCol;
            }
            else
                colToReplace.push_back( expr );
        }
        assert( colToUse );
        for( auto& expr : colToReplace )
            expr->ReplaceReferencedColumns( outerColumn, colToUse );
    }

} /* namespace aries */
