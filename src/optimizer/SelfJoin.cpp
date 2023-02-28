/*
 * SelfJoin.cpp
 *
 *  Created on: Jun 28, 2020
 *      Author: lichi
 */

#include "SelfJoin.h"
#include "frontend/BiaodashiAuxProcessor.h"
#include "frontend/SQLTreeNodeBuilder.h"
#include <queue>
#include <set>

namespace aries
{

    SelfJoin::SelfJoin()
    {

    }

    SelfJoin::~SelfJoin()
    {

    }

    string SelfJoin::ToString()
    {
        return "SelfJoin: self anti or semi join handling.";
    }

    SQLTreeNodePointer SelfJoin::OptimizeTree( SQLTreeNodePointer arg_input )
    {
        SelfJoinSearchResult searchResult = SearchSelfJoin( arg_input );
        if( searchResult.RootNode )
        {
            //we found a self join
            //extract the left most table's filter condition if it exists.
            CommonBiaodashiPtr filter = ExtractFilterCondition( searchResult.RootNode );

            //collect all join condition
            vector< HalfJoinInfo > joinInfo;
            CollectSelfJoinInfo( searchResult.RootNode, -1, joinInfo );

            // set up the required columns of table node
            // the outputOfTableNode is used to remove the duplicate column only, need reorder to get the correct output sequence.
            set< int > allRequiredColumns;

            //1. add myself's required columns
            for( const auto& col : searchResult.RootNode->GetRequiredColumnArray() )
                allRequiredColumns.insert( col->GetLocationInTable() );

            //2. add filter required column
            if( filter )
            {
                for( const auto& col : filter->GetAllReferencedColumns() )
                    allRequiredColumns.insert( col->GetLocationInTable() );
            }

            //3. add join info required column
            for( auto& info : joinInfo )
            {
                if( info.JoinConditionExpr )
                {
                    for( const auto& col : info.JoinConditionExpr->GetAllReferencedColumns() )
                        allRequiredColumns.insert( col->GetLocationInTable() );
                }
            }

            // create the output sequence of table node
            map< int, int > outputColumnsOfTableNode;// key: column location, value: column id
            vector< int > outputSequenceOfTableNode;
            int index = 0;
            for( int columnLocation : allRequiredColumns )
            {
                outputColumnsOfTableNode.insert(
                { columnLocation, ++index } );
                outputSequenceOfTableNode.push_back( columnLocation + 1 );
            }

            // create table node
            SQLTreeNodePointer newTableNode = SQLTreeNode::makeTreeNode_Table( arg_input->GetMyQuery(), searchResult.Context.SourceTable );
            assert( allRequiredColumns.find( searchResult.Context.ColumnLocation ) != allRequiredColumns.end() );

            // create self join node
            SQLTreeNodePointer selfJoinNode = SQLTreeNode::makeTreeNode_SelfJoin( arg_input->GetMyQuery(),
                    outputColumnsOfTableNode[searchResult.Context.ColumnLocation], filter, joinInfo );

            NodeRelationStructurePointer tableRelationStructure = std::make_shared< NodeRelationStructure >();
            RelationStructurePointer relStructure = searchResult.Context.SourceTable->GetRelationStructure();
            vector< ColumnShellPointer > referencedColumns;
            int referencedColumnIdInSelfJoinNode = 0;
            for( int id : outputSequenceOfTableNode )
            {
                ColumnStructurePointer csp = relStructure->GetColumn( id - 1 );
                tableRelationStructure->AddColumn( NodeColumnStructure::__createInstanceFromColumnStructure( csp ), relStructure->GetName() );
                ColumnShellPointer pColumnShell = std::make_shared< ColumnShell >( relStructure->GetName(), csp->GetName() );
                pColumnShell->SetTable( searchResult.Context.SourceTable );
                referencedColumns.push_back( pColumnShell );
                newTableNode->SetPositionForReferencedColumn( pColumnShell, id );
                selfJoinNode->SetPositionForReferencedColumn( pColumnShell, ++referencedColumnIdInSelfJoinNode );
            }
            newTableNode->SetReferencedColumnArray( referencedColumns );
            newTableNode->SetColumnOutputSequence( outputSequenceOfTableNode );
            newTableNode->SetNodeRelationStructure( tableRelationStructure );
            newTableNode->SetTreeFormedTag( true );

            // set output of self join node
            vector< int > outputSequenceOfSelfJoinNode;
            for( const auto& col : searchResult.RootNode->GetRequiredColumnArray() )
            {
                assert( allRequiredColumns.find( col->GetLocationInTable() ) != allRequiredColumns.end() );
                outputSequenceOfSelfJoinNode.push_back( outputColumnsOfTableNode[col->GetLocationInTable()] );
            }

            selfJoinNode->AddRequiredColumnArray( searchResult.RootNode->GetRequiredColumnArray() );
            selfJoinNode->SetColumnOutputSequence( outputSequenceOfSelfJoinNode );
            selfJoinNode->SetNodeRelationStructure( searchResult.RootNode->GetNodeRelationStructure() );
            selfJoinNode->AddChild( newTableNode );
            newTableNode->SetParent( selfJoinNode );
            selfJoinNode->SetTreeFormedTag( true );

            // adjust filter and join info related columns
            if( filter )
                AdjustColumnPositionInChildTable( outputColumnsOfTableNode, filter->GetAllReferencedColumns() );

            for( auto& info : joinInfo )
            {
                if( info.JoinConditionExpr )
                    AdjustColumnPositionInChildTable( outputColumnsOfTableNode, info.JoinConditionExpr->GetAllReferencedColumns() );
            }

            //replace the old node
            auto parent = searchResult.RootNode->GetParent();
            parent->CompletelyResetAChild( searchResult.RootNode, selfJoinNode );
            selfJoinNode->SetParent( parent );
        }
        return arg_input;
    }

    void SelfJoin::AdjustColumnPositionInChildTable( const map< int, int >& idMapping, const vector< ColumnShellPointer >& columns ) const
    {
        for( const auto& col : columns )
        {
            assert( idMapping.find( col->GetLocationInTable() ) != idMapping.end() );
            int pos = idMapping.find( col->GetLocationInTable() )->second;
            col->SetPositionInChildTables( col->GetPositionInChildTables() > 0 ? pos : -pos );
        }
    }

    CommonBiaodashiPtr SelfJoin::ExtractFilterCondition( SQLTreeNodePointer node ) const
    {
        assert( node );
        CommonBiaodashiPtr result;

        //locate the main table's filter node and remove it if it exists, the main table should be a left most node
        SQLTreeNodePointer leftMost = node;
        while( leftMost->GetChildCount() > 0 )
            leftMost = leftMost->GetChildByIndex( 0 );
        assert( leftMost->GetType() == SQLTreeNodeType::Table_NODE );
        auto parent = leftMost->GetParent();
        if( parent->GetType() == SQLTreeNodeType::Filter_NODE )
        {
            //this is the main table's filter conditon. save it to the output filter param
            result = dynamic_pointer_cast< CommonBiaodashi >( parent->GetFilterStructure() );
            // set the position info of column, the column id of filter of main table is aways positive.
            for( auto& col : result->GetAllReferencedColumns() )
                col->SetPositionInChildTables( parent->GetPositionForReferencedColumn( col ) );

            //remove the old filter node.
            auto grandParent = parent->GetParent();
            if( grandParent->GetChildCount() == 1 )
            {
                grandParent->ResetTheChild( leftMost );
                leftMost->SetParent( grandParent );
            }
            else
            {
                grandParent->ResetLeftChild( leftMost );
                leftMost->SetParent( grandParent );
            }
        }
        return result;
    }

    void SelfJoin::CollectSelfJoinInfo( SQLTreeNodePointer node, int activeJoinInfo, vector< HalfJoinInfo >& joinInfo ) const
    {
        if( IsHalfJoinNode( node ) )
        {
            HalfJoinInfo info;
            info.HalfJoinType = node->GetJoinType();
            info.JoinConditionExpr = dynamic_pointer_cast< CommonBiaodashi >( node->GetJoinOtherCondition() );

            // set the position info of column, we need to know if the column from main or sibling table.
            for( auto& col : info.JoinConditionExpr->GetAllReferencedColumns() )
                col->SetPositionInChildTables( node->GetPositionForReferencedColumn( col ) );

            joinInfo.push_back( info );
            activeJoinInfo = joinInfo.size() - 1;
        }
        else if( IsFilterNode( node ) )
        {
            assert( activeJoinInfo >= 0 && ( size_t )activeJoinInfo < joinInfo.size() );
            auto& info = joinInfo[activeJoinInfo];
            auto filterExpr = dynamic_pointer_cast< CommonBiaodashi >( node->GetFilterStructure() );
            // set the position info of column, the column id of filter of sibling table is aways negetive.
            for( auto& col : filterExpr->GetAllReferencedColumns() )
                col->SetPositionInChildTables( -( node->GetPositionForReferencedColumn( col ) ) );

            if( info.JoinConditionExpr )
            {
                BiaodashiAuxProcessor aux;
                info.JoinConditionExpr = dynamic_pointer_cast< CommonBiaodashi >( aux.make_biaodashi_from_and_list(
                std::vector< CommonBiaodashiPtr >{ info.JoinConditionExpr, filterExpr } ) );
            }
            else
                info.JoinConditionExpr = filterExpr;
        }
        for( size_t i = 0; i < node->GetChildCount(); ++i )
            CollectSelfJoinInfo( node->GetChildByIndex( i ), activeJoinInfo, joinInfo );
    }

    // TODO: we search only 1 self join for simplicity, it's enough for tpch. But there might be more, need improve in future
    SelfJoin::SelfJoinSearchResult SelfJoin::SearchSelfJoin( SQLTreeNodePointer root ) const
    {
        assert( root );
        queue< SQLTreeNodePointer > nodes;
        nodes.push( root );
        SelfJoinSearchContext context;
        while( !nodes.empty() )
        {
            SQLTreeNodePointer node = nodes.front();
            context.Reset();
            if( IsSelfJoin( node, context ) )
                return
                {   node, context};
            nodes.pop();
            for( size_t i = 0; i < node->GetChildCount(); ++i )
                nodes.push( node->GetChildByIndex( i ) );
        }
        context.Reset();
        return
        {   nullptr, context};
    }

    bool SelfJoin::IsSelfJoin( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const
    {
        bool result = false;
        switch( context.AllowedTypes )
        {
            case AllowedNodeTypes::HALFJOIN:
            {
                if( IsHalfJoinNode( arg_input ) && IsJoinConditionMatched( arg_input, context ) )
                {
                    context.AllowedTypes = AllowedNodeTypes::HALFJOIN_FILTER_TABLE;
                    bool leftOK = IsSelfJoin( arg_input->GetLeftChild(), context );
                    //TODO: we only allow filter or table node for right child. because we can only handle 1 main table case.
                    context.AllowedTypes = AllowedNodeTypes::FILTER_TABLE;
                    bool rightOK = IsSelfJoin( arg_input->GetRightChild(), context );
                    return leftOK && rightOK;
                }
                else
                    return false;
            }
            case AllowedNodeTypes::HALFJOIN_FILTER_TABLE:
            {
                if( IsHalfJoinNode( arg_input ) && IsJoinConditionMatched( arg_input, context ) )
                {
                    context.AllowedTypes = AllowedNodeTypes::HALFJOIN_FILTER_TABLE;
                    bool leftOK = IsSelfJoin( arg_input->GetLeftChild(), context );
                    //TODO: we only allow filter or table node for right child. because we can only handle 1 main table case.
                    context.AllowedTypes = AllowedNodeTypes::FILTER_TABLE;
                    bool rightOK = IsSelfJoin( arg_input->GetRightChild(), context );
                    return leftOK && rightOK;
                }
                else if( IsFilterNode( arg_input ) && IsFilterConditionMatched( arg_input, context ) )
                {
                    context.AllowedTypes = AllowedNodeTypes::FILTER_TABLE;
                    return IsSelfJoin( arg_input->GetChildByIndex( 0 ), context );
                }
                else if( IsTableNode( arg_input ) && IsTableMatched( arg_input, context ) )
                {
                    context.AllowedTypes = AllowedNodeTypes::NONE;
                    return true;
                }
                else
                    return false;
            }
            case AllowedNodeTypes::FILTER_TABLE:
            {
                if( IsFilterNode( arg_input ) && IsFilterConditionMatched( arg_input, context ) )
                {
                    context.AllowedTypes = AllowedNodeTypes::FILTER_TABLE;
                    return IsSelfJoin( arg_input->GetChildByIndex( 0 ), context );
                }
                else if( IsTableNode( arg_input ) && IsTableMatched( arg_input, context ) )
                {
                    context.AllowedTypes = AllowedNodeTypes::NONE;
                    return true;
                }
                else
                    return false;
            }
            case AllowedNodeTypes::NONE:
                assert( 0 );
                break;
            default:
                assert( 0 );
                break;
        }
        return result;
    }

    bool SelfJoin::IsJoinConditionMatched( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const
    {
        assert( IsHalfJoinNode( arg_input ) );
        bool result = false;
        auto joinCondition = dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinCondition() );
        // must have only 2 children, and be simple column id.
        if( joinCondition && joinCondition->GetChildrenCount() == 2 && arg_input->GetJoinOtherCondition() )
        {
            auto leftExpr = dynamic_pointer_cast< CommonBiaodashi >( joinCondition->GetChildByIndex( 0 ) );
            auto rightExpr = dynamic_pointer_cast< CommonBiaodashi >( joinCondition->GetChildByIndex( 1 ) );
            if( leftExpr->GetType() == BiaodashiType::Lie && rightExpr->GetType() == BiaodashiType::Lie )
            {
                auto leftColumn = boost::get< ColumnShellPointer >( leftExpr->GetContent() );
                auto rightColumn = boost::get< ColumnShellPointer >( rightExpr->GetContent() );
                auto sourceTable = leftColumn->GetTable();
                int location = leftColumn->GetLocationInTable();
                if( AreTheSamePhysicalTable( sourceTable, rightColumn->GetTable() ) && location == rightColumn->GetLocationInTable() )
                {
                    if( context.SourceTable )
                    {
                        assert( context.ColumnLocation != -1 );
                        if( AreTheSamePhysicalTable( context.SourceTable, sourceTable ) && location == context.ColumnLocation )
                            result = true;
                    }
                    else
                    {
                        assert( context.ColumnLocation == -1 );
                        context.SourceTable = sourceTable;
                        context.ColumnLocation = location;
                        result = true;
                    }
                }
            }
        }
        return result;
    }

    bool SelfJoin::IsFilterConditionMatched( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const
    {
        assert( IsFilterNode( arg_input ) );
        assert( arg_input->GetChildCount() == 1 );
        bool result = true;
        auto filterCondition = dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetFilterStructure() );
        // must from same table
        assert( context.SourceTable && context.ColumnLocation >= 0 );
        auto columns = filterCondition->GetAllReferencedColumns();
        for( const auto& col : columns )
        {
            if( !AreTheSamePhysicalTable( col->GetTable(), context.SourceTable ) )
            {
                result = false;
                break;
            }
        }
        return result;
    }

    bool SelfJoin::IsTableMatched( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const
    {
        assert( IsTableNode( arg_input ) );
        assert( arg_input->GetChildCount() == 0 );
        assert( context.SourceTable && context.ColumnLocation >= 0 );
        return AreTheSamePhysicalTable( arg_input->GetBasicRel(), context.SourceTable );
    }

    bool SelfJoin::IsHalfJoinNode( SQLTreeNodePointer arg_input ) const
    {
        bool result = false;
        if( arg_input->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
        {
            auto joinType = arg_input->GetJoinType();
            if( joinType == JoinType::SemiJoin || joinType == JoinType::AntiJoin )
                result = true;
        }
        return result;
    }

} /* namespace aries */
