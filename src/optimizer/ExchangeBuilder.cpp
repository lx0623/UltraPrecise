#include "ExchangeBuilder.h"

#include <vector>

#include "schema/SchemaManager.h"
#include "frontend/SelectStructure.h"
#include "Configuration.h"
#include "frontend/BiaodashiAuxProcessor.h"

#define CAST_EXPRESSION( expr ) std::dynamic_pointer_cast< CommonBiaodashi >( ( expr ) )

namespace aries
{

std::string ExchangeBuilder::ToString()
{
    return "ExchangeBuilder";
}

SQLTreeNodePointer ExchangeBuilder::OptimizeTree( SQLTreeNodePointer arg_input )
{
    if ( Configuartion::GetInstance().GetCudaDeviceCount() == 1 )
    {
        return arg_input;
    }

    handleNode( arg_input );
    return arg_input;
}

static void collect_leaf_nodes( SQLTreeNodePointer root, std::vector< SQLTreeNodePointer >& leaf_nodes )
{
    if ( root->GetChildCount() != 0 )
    {
        for ( size_t i = 0; i < root->GetChildCount(); i++ )
        {
            collect_leaf_nodes( root->GetChildByIndex( i ), leaf_nodes );
        }
    }
    else
    {
        leaf_nodes.emplace_back( root );
    }
}

size_t get_row_count( schema::Schema& schema, const BasicRelPointer& rel, std::string default_db )
{
    auto db_name = rel->GetDb();
    if ( db_name.empty() )
    {
        db_name = default_db;
    }

    if ( db_name.empty() )
    {
        ARIES_EXCEPTION( ER_NO_DB_ERROR, "no database selected" );
    }

    auto database = schema.GetDatabaseByName( db_name );
    auto table = database->GetTableByName( rel->GetID() );
    return table->GetRowCount();
}

static void get_exchange_deep( SQLTreeNodePointer node, int& deep, SQLTreeNodePointer& head )
{
    auto parent = node->GetParent();
    switch( parent->GetType() )
    {
        case SQLTreeNodeType::Filter_NODE:
        case SQLTreeNodeType::Group_NODE:
            head = parent;
            get_exchange_deep( parent, ++deep, head );
            break;
        default:
            break;
    }
}

void ExchangeBuilder::handleNode( SQLTreeNodePointer arg_input )
{
    std::vector< SQLTreeNodePointer > leaf_nodes;
    collect_leaf_nodes( arg_input, leaf_nodes );

    auto schema = schema::SchemaManager::GetInstance()->GetSchema();

    for ( auto& leaf : leaf_nodes )
    {
        auto select_structure = std::dynamic_pointer_cast< SelectStructure >( leaf->GetMyQuery() );
        auto default_schema = select_structure->GetDefaultSchemaAgent();
        std::string default_db = default_schema ? default_schema->schema->GetName() : "";
        auto row_count = get_row_count( *schema, leaf->GetBasicRel(), default_db );
        if ( row_count < Configuartion::GetInstance().GetExchangeRowCountThreshold() )
        {
            continue;
        }

        int deep = 1;
        SQLTreeNodePointer head;
        get_exchange_deep( leaf, deep, head );

        if ( deep < 3 )
        {
            continue;
        }

        auto parent = head->GetParent();
        auto new_node = buildExchangeNode( head );
        new_node->SetRequiredColumnArray( head->GetRequiredColumnArray() );
        parent->CompletelyResetAChild( head, new_node );
        new_node->SetParent( parent );
    }
}

static std::vector< BiaodashiPointer > get_agg_exprs( const BiaodashiPointer& arg_expression )
{
    std::vector< BiaodashiPointer > result;
    auto expression = CAST_EXPRESSION( arg_expression );
    switch ( expression->GetType() )
    {
        case BiaodashiType::Hanshu:
        {
            auto function = boost::get< SQLFunctionPointer >( expression->GetContent() );
            if ( function->GetIsAggFunc() )
            {
                result.emplace_back( expression );
            }
            break;
        }
        default:
        {
            for ( size_t i = 0; i < expression->GetChildrenCount(); i++ )
            {
                auto child = expression->GetChildByIndex( i );
                if ( !child )
                {
                    continue;
                }

                auto child_result = get_agg_exprs( child );
                if ( child_result.empty() )
                {
                    continue;
                }

                result.insert( result.end(), child_result.cbegin(), child_result.cend() );
            }
            break;
        }
    }
    return result;
}

static ColumnShellPointer create_virtual_column( BiaodashiAuxProcessor& processor, int id, BiaodashiValueType value_type, int level )
{
    auto column_name = std::string( "col_" ) + std::to_string( id );
    return processor.make_column_shell_only_placeholder( "virtual_table", column_name, value_type, level );
}

// static BiaodashiPointer create_count_star_expression()
// {
//     auto function = std::make_shared< SQLFunction >( "COUNT" );
//     function->SetValueType( BiaodashiValueType::LONG_INT );
//     auto expression = std::make_shared< CommonBiaodashi >( BiaodashiType::SQLFunc, function );
//     expression->SetValueType( BiaodashiValueType::LONG_INT );
//     expression->SetLength( 1 );
//     expression->SetIsNullable( false );
//     expression->AddChild( std::make_shared< CommonBiaodashi >( BiaodashiType::Star, 0 ) );
//     return expression;
// }

static BiaodashiPointer create_count_expression( BiaodashiPointer child )
{
    auto function = std::make_shared< SQLFunction >( "COUNT" );
    function->SetValueType( BiaodashiValueType::LONG_INT );
    auto expression = std::make_shared< CommonBiaodashi >( BiaodashiType::SQLFunc, function );
    expression->SetValueType( BiaodashiValueType::LONG_INT );
    expression->SetLength( 1 );
    expression->SetIsNullable( false );
    expression->AddChild( child );
    return expression;
}

static BiaodashiPointer create_sum_expression( BiaodashiPointer child )
{
    auto child_expression = ( CommonBiaodashi* )( child.get() );
    auto function = std::make_shared< SQLFunction >( "SUM" );
    function->SetValueType( child_expression->GetValueType() );
    auto expression = std::make_shared< CommonBiaodashi >( BiaodashiType::SQLFunc, function );
    expression->AddChild( child );
    expression->SetIsNullable( true );
    expression->SetLength( child_expression->GetLength() );
    expression->SetValueType( child_expression->GetValueType() );
    expression->SetAssociatedLength( -1 );
    // expression->SetExprContext()
    return expression;
}

static BiaodashiPointer create_div_expression( BiaodashiPointer column_expression, BiaodashiPointer count_star_expression )
{
    auto expression = std::make_shared< CommonBiaodashi >( BiaodashiType::Yunsuan, static_cast< int >( CalcType::DIV ) );
    expression->SetValueType( BiaodashiValueType::DECIMAL );
    expression->AddChild( column_expression );
    expression->AddChild( count_star_expression );
    return expression;
}

static bool reset_expression_child( BiaodashiPointer expression, BiaodashiPointer old_child, BiaodashiPointer new_child )
{
    auto expression_ptr = CAST_EXPRESSION( expression );
    auto old_expression = CAST_EXPRESSION( old_child );

    for ( size_t i = 0; i < expression_ptr->GetChildrenCount(); i++ )
    {
        auto child = CAST_EXPRESSION( expression_ptr->GetChildByIndex( i ) );
        if ( !child )
        {
            continue;
        }

        if ( *child == *old_expression )
        {
            expression_ptr->ResetChildByIndex( i, new_child );
            return true;
        }

        if ( reset_expression_child( child, old_child, new_child ) )
        {
            return true;
        }
    }

    return false;
}

static CommonBiaodashiPtr
handle_expression( CommonBiaodashiPtr expression,
                    BiaodashiAuxProcessor& processor,
                    std::vector< CommonBiaodashiPtr >& old_exprs,
                    std::map< ColumnShellPointer, int >& position_map,
                    const std::vector< BiaodashiPointer >& group_by_exprs,
                    int& seq )
{
    bool is_groupby_expr = false;

    auto query_level = expression->GetExprContext()->GetQueryContext()->query_level;
    
    for ( const auto& groupby_expr : group_by_exprs )
    {
        if ( expression->CompareMyselfToAnotherExpr( groupby_expr ) )
        {
            is_groupby_expr = true;
            break;
        }
    }

    if ( is_groupby_expr )
    {
        auto virtual_column = create_virtual_column( processor, ++seq, expression->GetValueType(), query_level );

        bool found = false;
        int position = old_exprs.size() + 1;
        for ( size_t i = 0; i < old_exprs.size(); i++ )
        {
            auto& old_expr = old_exprs[ i ];
            if ( old_expr->CompareMyselfToAnotherExpr( expression ) )
            {
                position = i + 1;
                found = true;
                break;
            }
        }

        if ( !found )
        {
            old_exprs.emplace_back( expression );
        }
        position_map[ virtual_column ] = position;

        auto new_expression = CAST_EXPRESSION( processor.make_biaodashi_lie( virtual_column ) );
        new_expression->SetValueType( expression->GetValueType() );
        new_expression->SetLength( expression->GetLength() );
        new_expression->SetExprContext( expression->GetExprContext() );
        new_expression->SetAssociatedLength( expression->GetAssociatedLength() );
        new_expression->SetIsNullable( expression->IsNullable() );
        return new_expression;
    }

    if ( expression->IsAggFunction() )
    {
        auto function = boost::get< SQLFunctionPointer >( expression->GetContent() );
        auto child = CAST_EXPRESSION( expression->GetChildByIndex( 0 ) );


        if ( function->GetType() == AriesSqlFunctionType::AVG )
        {
            BiaodashiPointer sum_expression;
            int position = -1;
            int count_position = -1;
            for ( size_t i = 0; i < old_exprs.size(); i++ )
            {
                auto& old_expr = old_exprs[ i ];
                if ( old_expr->IsAggFunction() )
                {
                    auto old_function = boost::get< SQLFunctionPointer >( old_expr->GetContent() );
                    if ( old_function->GetType() == AriesSqlFunctionType::SUM )
                    {
                        auto old_child = CAST_EXPRESSION( old_expr->GetChildByIndex( 0 ) );
                        if ( child->CompareMyselfToAnotherExpr( old_child ) )
                        {
                            position = i + 1;
                            sum_expression = old_expr;
                        }
                    }
                    else if ( old_function->GetType() == AriesSqlFunctionType::COUNT )
                    {
                        auto old_child = CAST_EXPRESSION( old_expr->GetChildByIndex( 0 ) );
                        if ( child->CompareMyselfToAnotherExpr( old_child ) )
                        {
                            count_position = i + 1;
                        }
                    }
                }
            }

            if ( position == -1 )
            {
                position = old_exprs.size() + 1;
                sum_expression = create_sum_expression( child );
                old_exprs.emplace_back( CAST_EXPRESSION( sum_expression ) );
            }

            if ( count_position == -1 )
            {
                auto count_expression = create_count_expression( child );
                count_position = old_exprs.size() + 1;
                old_exprs.emplace_back( CAST_EXPRESSION( count_expression ) );
            }

            auto virtual_column = create_virtual_column( processor, ++seq, child->GetValueType(), query_level );
            auto virtual_count_column = create_virtual_column( processor, ++seq, BiaodashiValueType::LONG_INT, query_level );

            position_map[ virtual_column ] = position;
            position_map[ virtual_count_column ] = count_position;

            auto sum_column_expression = CAST_EXPRESSION( create_sum_expression( processor.make_biaodashi_lie( virtual_column ) ) );
            auto sum_count_expression = CAST_EXPRESSION( create_sum_expression( processor.make_biaodashi_lie( virtual_count_column) ) );

            auto new_expression = CAST_EXPRESSION( 
                create_div_expression( sum_column_expression, sum_count_expression ) 
            );
            new_expression->SetValueType( expression->GetValueType() );
            new_expression->SetLength( expression->GetLength() );
            new_expression->SetExprContext( expression->GetExprContext() );
            new_expression->SetAssociatedLength( expression->GetAssociatedLength() );
            new_expression->SetIsNullable( expression->IsNullable() );
            return new_expression;
        }
        else if ( function->GetType() == AriesSqlFunctionType::COUNT )
        {
            int count_position = -1;
            for ( size_t i = 0; i < old_exprs.size(); i++ )
            {
                auto& old_expr = old_exprs[ i ];
                if ( old_expr->IsAggFunction() )
                {
                    auto old_function = boost::get< SQLFunctionPointer >( old_expr->GetContent() );
                    if ( old_function->GetType() == AriesSqlFunctionType::COUNT )
                    {
                        auto old_child = CAST_EXPRESSION( old_expr->GetChildByIndex( 0 ) );
                        if ( child->CompareMyselfToAnotherExpr( old_child ) )
                        {
                            count_position = i + 1;
                            break;
                        }
                    }
                }
            }

            BiaodashiPointer count_expression;
            if ( count_position == -1 )
            {
                count_position = old_exprs.size() + 1;
                old_exprs.emplace_back( expression );
            }

            auto virtual_count_column = create_virtual_column( processor, ++seq, BiaodashiValueType::LONG_INT, query_level );
            position_map[ virtual_count_column ] = count_position;
            auto new_expression = create_sum_expression( processor.make_biaodashi_lie( virtual_count_column) );
            return CAST_EXPRESSION( new_expression );
        }
        else
        {
            ColumnShellPointer virtual_column;
            if ( function->GetType() == AriesSqlFunctionType::SUM )
            {
                BiaodashiPointer sum_expression;
                int position = -1;
                for ( size_t i = 0; i < old_exprs.size(); i++ )
                {
                    auto& old_expr = old_exprs[ i ];
                    if ( old_expr->IsAggFunction() )
                    {
                        auto old_function = boost::get< SQLFunctionPointer >( old_expr->GetContent() );
                        if ( old_function->GetType() == AriesSqlFunctionType::SUM )
                        {
                            auto old_child = CAST_EXPRESSION( old_expr->GetChildByIndex( 0 ) );
                            if ( child->CompareMyselfToAnotherExpr( old_child ) )
                            {
                                position = i + 1;
                                sum_expression = old_expr;
                                break;
                            }
                        }
                    }
                }

                if ( position == -1 )
                {
                    position = old_exprs.size() + 1;
                    old_exprs.emplace_back( expression );
                }

                virtual_column = create_virtual_column( processor, ++seq, expression->GetValueType(), query_level );
                position_map[ virtual_column ] = position;
            }
            else
            {
                virtual_column = create_virtual_column( processor, ++seq, expression->GetValueType(), query_level );
                position_map[ virtual_column ] = old_exprs.size() + 1;
                old_exprs.emplace_back( expression );
            }
            
            auto new_expression = CAST_EXPRESSION( processor.make_biaodashi_lie( virtual_column ) );
            new_expression->SetValueType( expression->GetValueType() );
            new_expression->SetLength( expression->GetLength() );
            new_expression->SetExprContext( expression->GetExprContext() );
            new_expression->SetAssociatedLength( expression->GetAssociatedLength() );
            new_expression->SetIsNullable( expression->IsNullable() );
            auto new_agg_expression = std::make_shared< CommonBiaodashi >( BiaodashiType::SQLFunc, function );
            new_agg_expression->AddChild( new_expression );
            new_agg_expression->SetValueType( expression->GetValueType() );
            new_agg_expression->SetIsNullable( expression->IsNullable() );
            new_agg_expression->SetLength( expression->GetLength() );
            new_agg_expression->SetAssociatedLength( expression->GetAssociatedLength() );
            new_agg_expression->SetExprContext( new_expression->GetExprContext() );
            return new_agg_expression;
        }
    }

    for ( size_t i = 0; i < expression->GetChildrenCount(); i++ )
    {
        auto child = CAST_EXPRESSION( expression->GetChildByIndex( i ) );
        auto new_child = handle_expression( child, processor, old_exprs, position_map, group_by_exprs, seq );
        if ( new_child != child )
        {
            expression->ResetChildByIndex( i, new_child );
        }   
    }

    return expression;
}

static void set_slice_range( SQLTreeNodePointer node, int count, int index )
{
    if ( node->GetType() == SQLTreeNodeType::Table_NODE )
    {
        node->SetSliceCount( count );
        node->SetSliceIndex( index );
        node->GetParent()->ReCalculateInvolvedTableList();
    }
    else
    {
        if ( node->GetChildCount() > 1 )
        {
            ARIES_ASSERT( 0, "too many children" );
        }

        set_slice_range( node->GetTheChild(), count, index );
    }
}

SQLTreeNodePointer ExchangeBuilder::createExchangeNode( SQLTreeNodeBuilder& builder )
{
    auto exchange_node = builder.makeTreeNode_Exchange();

    auto cuda_device_count = Configuartion::GetInstance().GetCudaDeviceCount();

    std::vector< int > source_devices_id( cuda_device_count );
    std::iota( source_devices_id.begin(), source_devices_id.end(), 0 );
    exchange_node->SetTargetDeviceId( 0 );
    exchange_node->SetSourceDevicesId( source_devices_id );
    return exchange_node;
}

SQLTreeNodePointer ExchangeBuilder::buildExchangeNodeForFilterNode( SQLTreeNodePointer filter_node )
{
    auto query = std::dynamic_pointer_cast< SelectStructure >( filter_node->GetMyQuery() );
    SQLTreeNodeBuilder builder( query );
    auto exchange_node = createExchangeNode( builder );
    auto source_devices_count = static_cast< int >( exchange_node->GetSourceDevicesId().size() );

    for ( int i = 0; i < source_devices_count; i++ )
    {
        auto source_node = filter_node->Clone();
        set_slice_range( source_node, source_devices_count, i );
        SQLTreeNode::AddTreeNodeChild( exchange_node, source_node );
    }

    return exchange_node;
}

SQLTreeNodePointer ExchangeBuilder::buildExchangeNodeForGroupNode( SQLTreeNodePointer group_node )
{
    auto query = std::dynamic_pointer_cast< SelectStructure >( group_node->GetMyQuery() );
    auto select_exprs = query->GetSelectPart()->GetAllExprs();
    auto groupby_part = query->GetGroupbyPart();
    auto additional_exprs = groupby_part->GetAdditionalExprsForSelect();
    auto groupby_exprs = groupby_part->GetGroupbyExprs();

    BiaodashiAuxProcessor processor;

    std::vector< CommonBiaodashiPtr > select_exprs_before_exchange;
    std::vector< BiaodashiPointer > select_exprs_after_exchange;
    std::vector< CommonBiaodashiPtr > children_of_sum_expression;

    std::map< ColumnShellPointer, int > position_map_after_exchange;
    std::map< CommonBiaodashiPtr, int > sum_expression_position;

    int virtual_column_id = 0;
    for ( auto& expr : select_exprs )
    {
        auto expression = CAST_EXPRESSION( expr );
        auto new_expr = handle_expression( expression, processor, select_exprs_before_exchange, position_map_after_exchange, groupby_exprs, virtual_column_id );
        select_exprs_after_exchange.emplace_back( new_expr );
    }

    std::vector< BiaodashiPointer > groupby_exprs_after_exchange;

    size_t seq = position_map_after_exchange.size();
    for ( const auto& expr : groupby_exprs )
    {
        auto expression = CAST_EXPRESSION( expr );
        int position = -1;
        for ( size_t i = 0; i < select_exprs_before_exchange.size(); i++ )
        {
            const auto& item = select_exprs_before_exchange[ i ];
            if ( expression->CompareMyselfToAnotherExpr( item ) )
            {
                position = i + 1;
                break;
            }
        }

        if ( position == -1 )
        {
            position = select_exprs.size() + 1;
            select_exprs_before_exchange.emplace_back( expression );
        }

        auto virtual_column = create_virtual_column( processor, ++seq, expression->GetValueType(), query->GetQueryContext()->query_level );
        position_map_after_exchange[ virtual_column ] = position;
        groupby_exprs_after_exchange.emplace_back( processor.make_biaodashi_lie( virtual_column ) );
    }

    SQLTreeNodeBuilder builder( group_node->GetMyQuery() );
    auto exchange_node = createExchangeNode( builder );
    auto source_devices_count = static_cast< int >( exchange_node->GetSourceDevicesId().size() );

    for ( int i = 0; i < source_devices_count; i++ )
    {
        auto source_node = group_node->Clone();
        auto groupby_part = std::make_shared< GroupbyStructure >();
        groupby_part->SetGroupbyExprs( groupby_exprs );
        source_node->SetMyGroupbyStructure( groupby_part );

        auto select_part = std::make_shared< SelectPartStructure >();
        for ( const auto& expr : select_exprs_before_exchange )
        {
            select_part->AddCheckedExpr( expr );
            select_part->AddCheckedAlias( nullptr );
        }

        source_node->SetMySelectPartStructure( select_part );
        set_slice_range( source_node, source_devices_count, i );
        SQLTreeNode::AddTreeNodeChild( exchange_node, source_node );
    }

    auto group_after_exchange = builder.makeTreeNode_Group();
    auto groupby_structure = std::make_shared< GroupbyStructure >();
    groupby_structure->SetGroupbyExprs( groupby_exprs_after_exchange );
    group_after_exchange->SetMyGroupbyStructure( groupby_structure );
    auto select_part = std::make_shared< SelectPartStructure >();
    for ( const auto& expr : select_exprs_after_exchange )
    {
        select_part->AddCheckedExpr( expr );
        select_part->AddCheckedAlias( nullptr );
    }
    group_after_exchange->SetMySelectPartStructure( select_part );
    for ( const auto& item : position_map_after_exchange )
    {
        group_after_exchange->SetPositionForReferencedColumn( item.first, item.second );
    }

    SQLTreeNode::SetTreeNodeChild( group_after_exchange, exchange_node );
    return group_after_exchange;
}

SQLTreeNodePointer ExchangeBuilder::buildExchangeNode( SQLTreeNodePointer node )
{
    switch( node->GetType() )
    {
        case SQLTreeNodeType::Group_NODE:
            return buildExchangeNodeForGroupNode( node );
        case SQLTreeNodeType::Filter_NODE:
            return buildExchangeNodeForFilterNode( node );
        default:
            ARIES_ASSERT( 0, "Not supported" );
            return nullptr;
    }
}

void ExchangeBuilder::handleGroupNode( SQLTreeNodePointer arg_input )
{
    
}

}