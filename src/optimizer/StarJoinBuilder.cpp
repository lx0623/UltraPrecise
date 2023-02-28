#include "StarJoinBuilder.h"

#include  <algorithm>
#include  <set>
#include <unordered_set>
#include "frontend/SQLTreeNodeBuilder.h"
#include "frontend/BiaodashiAuxProcessor.h"

namespace aries
{

void StarJoinBuilder::handleNode( SQLTreeNodePointer arg_input )
{
    if ( !arg_input || arg_input->GetSpoolId() != -1 )
    {
        return;
    }

    switch ( arg_input->GetType() )
    {
        case SQLTreeNodeType::BinaryJoin_NODE:
            handleJoinNode( arg_input );
            break;
        default:
            handleNode( arg_input->GetTheChild() );
            break;
    }
}

static bool contain_node(SQLTreeNodePointer origin, SQLTreeNodePointer target )
{
    if( origin == target )
        return true;
    for ( size_t i = 0; i < origin->GetChildCount(); i++ )
    {
        auto child = origin->GetChildByIndex( i );
        if ( child == target )
            return true;
        else if( contain_node( child, target ) )
            return true;
    }
    return false;
}

static bool is_true_condition( BiaodashiPointer arg_condition )
{
    auto* condition = dynamic_cast< CommonBiaodashi* >( arg_condition.get() );
    return condition->GetType() == BiaodashiType::Zhenjia && boost::get< bool >( condition->GetContent() );
}

static void
collect_join_nodes( const SQLTreeNodePointer& node,
                   std::vector< SQLTreeNodePointer >& nodes,
                   std::vector< BiaodashiPointer >& conditions )
{
    if ( !node->IsInnerJoin() )
    {
        nodes.emplace_back( node );
        return;
    }

    auto condition = node->GetJoinCondition();

    if ( !is_true_condition( condition ) )
    {
        conditions.emplace_back( condition );
    }

    collect_join_nodes( node->GetLeftChild(), nodes, conditions );
    collect_join_nodes( node->GetRightChild(), nodes, conditions );
}

static bool
vector_contains_all( const std::vector< BasicRelPointer >& all, const std::vector< BasicRelPointer >& sub )
{
    for ( const auto& s : sub )
    {
        bool found = false;
        for ( const auto& a : all )
        {
            if ( *s == *a )
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

SQLTreeNodePointer find_hash_node( const std::vector< BasicRelPointer >& tables, SQLTreeNodePointer node )
{
    if ( node->GetType() != SQLTreeNodeType::BinaryJoin_NODE )
    {
        if ( vector_contains_all( tables, node->GetInvolvedTableList() ) ) 
        {
            return node;
        }
       
    }

    if ( node->IsInnerJoin() )
    {
        auto ret = find_hash_node( tables, node->GetLeftChild() );
        if ( ret )
        {
            return ret;
        }

        return find_hash_node( tables, node->GetRightChild() );
    }

    if ( node->GetType() == SQLTreeNodeType::BinaryJoin_NODE && node->GetJoinType() == JoinType::LeftJoin )
    {
        return find_hash_node( tables, node->GetLeftChild() );
    }
    return nullptr;
}

SQLTreeNodePointer find_target_node( const std::vector< BasicRelPointer >& tables, const SQLTreeNodePointer& node )
{
    if ( node->IsInnerJoin() )
    {
        auto ret = find_target_node( tables, node->GetLeftChild() );
        if ( ret )
        {
            return ret;
        }

        return find_target_node( tables, node->GetRightChild() );
    }

    if ( vector_contains_all( node->GetInvolvedTableList(), tables ) )
    {
        node->SetConditionAsStarJoin( node->GetParent()->GetJoinCondition() );
        return node;
    }

    return nullptr;
}

void StarJoinBuilder::handleInnerJoinNode( SQLTreeNodePointer arg_input, std::vector< HashJoinPair >& pairs, std::vector< NonHashJoinNode >& other_nodes )
{
    if ( arg_input->GetSpoolId() != -1 )
    {
        return;
    }

    bool left_as_hash( false ), right_as_hash( false );
    auto left = arg_input->GetLeftChild();
    auto right = arg_input->GetRightChild();

    if ( !arg_input->CanUseHashJoin( left_as_hash, right_as_hash ) )
    {
        if ( left->IsInnerJoin() )
        {
            handleInnerJoinNode( left, pairs, other_nodes );
        }
        else
        {
            handleNode( left );
            auto other_condition = arg_input->GetJoinOtherCondition() ? std::dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinOtherCondition() ) : nullptr;
            other_nodes.emplace_back( left, std::dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinCondition() ), other_condition  );
        }

        if ( right->IsInnerJoin() )
        {
            handleInnerJoinNode( right, pairs, other_nodes );
        }
        else
        {
            handleNode( right );
            auto other_condition = arg_input->GetJoinOtherCondition() ? std::dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinOtherCondition() ) : nullptr;
            other_nodes.emplace_back( right, std::dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinCondition() ), other_condition  );
        }
        return;
    }

    auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinCondition() );
    auto other_condition = std::dynamic_pointer_cast< CommonBiaodashi >( arg_input->GetJoinOtherCondition() );
    if ( left_as_hash )
    {
        const auto& join_info = arg_input->GetLeftHashJoinInfo();
        std::vector< BasicRelPointer > tables;
        std::vector< BasicRelPointer > hash_tables;
        for ( const auto& condition : join_info.EqualConditions )
        {
            auto right_tables = ( ( CommonBiaodashi* )( ( ( CommonBiaodashi* )( condition.get() ) )->GetChildByIndex( 1 ).get() ) )->GetInvolvedTableList();
            tables.insert( tables.end(), right_tables.cbegin(), right_tables.cend() );

            auto left_tables = ( ( CommonBiaodashi* )( ( ( CommonBiaodashi* )( condition.get() ) )->GetChildByIndex( 0 ).get() ) )->GetInvolvedTableList();
            hash_tables.insert( hash_tables.end(), left_tables.cbegin(), left_tables.cend() );
        }

        // auto tables = right_condition->GetInvolvedTableList();
        auto target = find_target_node( tables, right );
        
        if ( target )
        {
            auto hash_node = left->IsInnerJoin() ? find_hash_node( hash_tables, left ) : left;
            pairs.emplace_back( hash_node, target, join_info.EqualConditions, join_info.OtherCondition );
        }

        if ( left->IsInnerJoin() )
        {
            handleInnerJoinNode( left, pairs, other_nodes );
        }

        if ( right->IsInnerJoin() )
        {
            handleInnerJoinNode( right, pairs, other_nodes );
        }
        else
        {
            if( target != right )
                other_nodes.emplace_back( right, condition, other_condition );
        }
    }

    if ( right_as_hash )
    {
        const auto& join_info = arg_input->GetRightHashJoinInfo();
        std::vector< BasicRelPointer > tables;
        std::vector< BasicRelPointer > hash_tables;
        for ( const auto& condition : join_info.EqualConditions )
        {
            auto left_tables = ( ( CommonBiaodashi* )( ( ( CommonBiaodashi* )( condition.get() ) )->GetChildByIndex( 0 ).get() ) )->GetInvolvedTableList();
            tables.insert( tables.end(), left_tables.cbegin(), left_tables.cend() );
            auto right_tables = ( ( CommonBiaodashi* )( ( ( CommonBiaodashi* )( condition.get() ) )->GetChildByIndex( 1 ).get() ) )->GetInvolvedTableList();
            hash_tables.insert( hash_tables.end(), right_tables.cbegin(), right_tables.cend() );
        }

        auto target = find_target_node( tables, left );

        if ( target )
        {
            auto hash_node = right->GetType() == SQLTreeNodeType::BinaryJoin_NODE ? find_hash_node( hash_tables, right ) : right;
            pairs.emplace_back( hash_node, target, join_info.EqualConditions, join_info.OtherCondition );
        }

        if ( right->IsInnerJoin() )
        {
            handleInnerJoinNode( right, pairs, other_nodes );
        }

        if ( left->IsInnerJoin() )
        {
            handleInnerJoinNode( left, pairs, other_nodes );
        }
        else
        {
            if( target != left )
                other_nodes.emplace_back( left, condition, other_condition );
        }
    }
}

static void merge_two_columns_vector( std::vector< ColumnShellPointer >& dest, const std::vector< ColumnShellPointer > source )
{
    for ( const auto& column : source )
    {
        bool found = false;
        for ( size_t i = 0; i < dest.size(); i++ )
        {
            const auto& column_in_dest = dest[ i ];
            if ( *column_in_dest->GetTable() == *column->GetTable() && column_in_dest->GetColumnName() == column->GetColumnName() )
            {
                found = true;
                break;
            }
        }

        if ( !found )
        {
            dest.emplace_back( column );
        }
    }
}

static void setup_join_node_unique_keys( SQLTreeNode* node )
{
    bool left_as_hash = false;
    bool right_as_hash = false;
    if ( node->CanUseHashJoin( left_as_hash, right_as_hash ) )
    {
        auto output_ids = node->GetColumnOutputSequence();
        if ( left_as_hash )
        {
            auto unique_keys = node->GetRightChild()->GetUniqueKeys();
            for ( const auto& keys : unique_keys )
            {
                bool match = true;
                std::vector< int > my_keys;
                for ( const auto& key : keys )
                {
                    bool found = false;
                    for ( size_t i = 0; i < output_ids.size(); i++ )
                    {
                        if ( key == -output_ids[ i ] )
                        {
                            my_keys.emplace_back( i + 1 );
                            found = true;
                            break;
                        }
                    }

                    if ( !found )
                    {
                        match = false;
                    }
                }
                if ( match )
                {
                    node->AddUniqueKeys( my_keys );
                }
            }
        }
        else
        {
            auto unique_keys = node->GetLeftChild()->GetUniqueKeys();
            for ( const auto& keys : unique_keys )
            {
                bool match = true;
                std::vector< int > my_keys;
                for ( const auto& key : keys )
                {
                    bool found = false;
                    for ( size_t i = 0; i < output_ids.size(); i++ )
                    {
                        if ( key == output_ids[ i ] )
                        {
                            my_keys.emplace_back( i + 1 );
                            found = true;
                            break;
                        }
                    }

                    if ( !found )
                    {
                        match = false;
                    }
                }
                if ( match )
                {
                    node->AddUniqueKeys( my_keys );
                }
            }
        }
    }
}

static void resetup_star_join_node( SQLTreeNodePointer& node, const std::vector< ColumnShellPointer >& required_columns );

static void resetup_join_node( SQLTreeNodePointer node, const std::vector< ColumnShellPointer >& required_columns )
{
    if ( required_columns.empty() )
    {
        return;
    }

    if ( !node->GetRequiredColumnArray().empty() )
    {
        auto origin_required_columns = node->GetRequiredColumnArray();

        if ( origin_required_columns.size() == required_columns.size() )
        {
            bool matched = true;
            for ( size_t i = 0; i < required_columns.size(); i++ )
            {
                if ( *origin_required_columns[ i ]->GetTable() == *required_columns[ i ]->GetTable() &&
                     origin_required_columns[ i ]->GetColumnName() == required_columns[ i ]->GetColumnName() )
                {
                    continue;
                }
                matched = false;
            }

            if ( matched )
            {
                return;
            }
        }
    }

    auto left = node->GetLeftChild();
    auto right = node->GetRightChild();

    auto left_tables = left->GetInvolvedTableList();
    auto right_tables = right->GetInvolvedTableList();

    std::vector< ColumnShellPointer > left_required;
    std::vector< ColumnShellPointer > right_required;

    std::vector< int > output_columns_id;

    std::vector< int > left_output_columns_id;
    std::vector< int > right_output_columns_id;

    auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( node->GetJoinCondition() );
    auto referenced_columns = condition->GetAllReferencedColumns();

    if ( node->GetJoinOtherCondition() )
    {
        auto other_condition = std::dynamic_pointer_cast< CommonBiaodashi >( node->GetJoinOtherCondition() );
        auto referenced_columns_in_other_codnition = other_condition->GetAllReferencedColumns();

        merge_two_columns_vector( referenced_columns, referenced_columns_in_other_codnition );
    }

    merge_two_columns_vector( referenced_columns, required_columns );

    for ( const auto& column : referenced_columns )
    {
        auto table = column->GetTable();
        bool found = false;
        for ( const auto& left_table : left_tables )
        {
            if ( *table == *left_table )
            {

                found = true;
                left_required.emplace_back( column );
                node->SetPositionForReferencedColumn( column, left_required.size() );
            }
        }
        if ( !found )
        {
            right_required.emplace_back( column );
            node->SetPositionForReferencedColumn( column, - right_required.size() );
        }
    }

    if ( !left_required.empty() )
    {
        auto child_required_columns = left->GetRequiredColumnArray();
        if ( child_required_columns.size() > 0 )
        {
            for ( const auto& col : left_required )
            {
                bool found = false;
                int id = 0;
                for ( size_t i = 0; i < child_required_columns.size(); i++ )
                {
                    if ( *col->GetTable() == *child_required_columns[ i ]->GetTable() &&
                         col->GetColumnName() == child_required_columns[ i ]->GetColumnName() )
                    {
                        found = true;
                        id = i + 1;
                    }
                }

                ARIES_ASSERT( found, "cannot find column" + col->ToString() );
                node->SetPositionForReferencedColumn( col, id );

            }
        }
        else if ( left->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
        {
            resetup_join_node( left, left_required );
        }
        else if ( left->GetType() == SQLTreeNodeType::StarJoin_NODE )
        {
            resetup_star_join_node( left, left_required );
        }
    }

    if ( !right_required.empty() )
    {
        auto child_required_columns = right->GetRequiredColumnArray();
        if ( child_required_columns.size() > 0 )
        {
            for ( const auto& col : right_required )
            {
                bool found = false;
                int id = 0;
                for ( size_t i = 0; i < child_required_columns.size(); i++ )
                {
                    if ( *col->GetTable() == *child_required_columns[ i ]->GetTable() &&
                         col->GetColumnName() == child_required_columns[ i ]->GetColumnName() )
                    {
                        found = true;
                        id = i + 1;
                    }
                }

                ARIES_ASSERT( found, "cannot find column" + col->ToString() );
                node->SetPositionForReferencedColumn( col, -id );
            }
        }
        else if ( right->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
        {
            resetup_join_node( right, right_required );
        }
        else if ( right->GetType() == SQLTreeNodeType::StarJoin_NODE )
        {
            resetup_star_join_node( right, right_required );
        }
    }

    for ( const auto& col : required_columns )
    {
        output_columns_id.emplace_back( node->GetPositionForReferencedColumn( col ) );
    }

    // output_columns_id.assign( left_output_columns_id.cbegin(), left_output_columns_id.cend() );
    // output_columns_id.insert( output_columns_id.end(), right_output_columns_id.cbegin(), right_output_columns_id.cend() );

    // node->SetRequiredColumnArray( required_columns );
    node->AddRequiredColumnArray( required_columns );
    if( node->GetSpoolId() > -1 && node->GetSameNode() )
        node->GetSameNode()->AddRequiredColumnArray( required_columns );
    node->SetReferencedColumnArray( referenced_columns );
    node->SetColumnOutputSequence( output_columns_id );


    setup_join_node_unique_keys( node.get() );
}

static void resetup_filter_node( SQLTreeNodePointer node, const std::vector< ColumnShellPointer >& required_columns )
{
    auto condition = node->GetFilterStructure();

    std::vector< ColumnShellPointer > referenced_columns;
    referenced_columns.assign( required_columns.cbegin(), required_columns.cend() );

    merge_two_columns_vector( referenced_columns, ( ( CommonBiaodashi* )( condition.get() ) )->GetAllReferencedColumns() );

    auto child = node->GetTheChild();
    if ( child->GetType() == SQLTreeNodeType::StarJoin_NODE )
    {
        resetup_star_join_node( child, referenced_columns );
    }

    int id = 1;
    for ( const auto& column : referenced_columns )
    {
        node->SetPositionForReferencedColumn( column, id++ );
    }

    std::vector< int > columns_id;
    for ( const auto& column : required_columns )
    {
        auto id = node->GetPositionForReferencedColumn( column );
        columns_id.emplace_back( id );
    }

    node->SetColumnOutputSequence( columns_id );
}

static void resetup_star_join_node( SQLTreeNodePointer& node, const std::vector< ColumnShellPointer >& required_columns )
{
    auto other_condition = node->GetJoinOtherCondition();
    if ( other_condition )
    {
        node->SetJoinOtherCondition( nullptr );
        SQLTreeNodeBuilder builder( node->GetMyQuery() );
        auto filter_node = builder.makeTreeNode_Filter( other_condition );
        
        auto parent = node->GetParent();
        SQLTreeNode::SetTreeNodeChild( filter_node, node );
        if ( parent )
        {
            parent->CompletelyResetAChild( node, filter_node );
            filter_node->SetParent( parent );
        }

        node = filter_node;
        resetup_filter_node( node, required_columns );

        auto child_unique_keys = node->GetTheChild()->GetUniqueKeys();
        if ( !child_unique_keys.empty() )
        {
            auto output_ids = filter_node->GetColumnOutputSequence();
            for ( const auto& keys : child_unique_keys )
            {
                bool match = true;
                std::vector< int > my_keys;
                for ( const auto& key : keys )
                {
                    bool found = false;
                    for ( size_t i = 0; i < output_ids.size(); i++ )
                    {
                        if ( output_ids[ i ] == key )
                        {
                            found = true;
                            my_keys.emplace_back( i + 1 );
                            break;
                        }
                    }

                    if ( !found )
                    {
                        match = false;
                        break;
                    }
                }

                if ( match )
                {
                    filter_node->AddUniqueKeys( my_keys );
                }
                
            }
        }

        return;
    }

    std::vector< ColumnShellPointer > referenced_columns;

    for ( const auto& condition_array : node->GetStarJoinConditions() )
    {
        for ( const auto& condition : condition_array )
        {
            auto columns = condition->GetAllReferencedColumns();
            merge_two_columns_vector( referenced_columns, columns );
            // referenced_columns.insert( referenced_columns.end(), columns.cbegin(), columns.cend() );
        }
    }

    merge_two_columns_vector( referenced_columns, required_columns );

    int index = 1;
    for ( size_t i = 0; i < node->GetChildCount(); i++ )
    {
        auto child = node->GetChildByIndex( i );
        auto child_tables = child->GetInvolvedTableList();

        std::vector< ColumnShellPointer > child_required;
        for ( const auto& column : referenced_columns )
        {
            auto table = column->GetTable();

            for ( const auto& child_table : child_tables )
            {
                if ( *child_table == *table )
                {
                    child_required.emplace_back( column );
                    node->SetPositionForReferencedColumn( column, index++ );
                }
            }
            // if ( std::find( child_tables.cbegin(), child_tables.cend(), table ) != child_tables.cend() )
            // {
            //     child_required.emplace_back( column );
            //     node->SetPositionForReferencedColumn( column, index++ );
            // }
        }

        if ( !child_required.empty() )
        {
            auto child_required_columns = child->GetRequiredColumnArray();
            if ( child_required_columns.size() > 0 )
            {
                index -= child_required.size();
                for ( const auto& col : child_required )
                {
                    bool found = false;
                    int id = 0;
                    for ( size_t i = 0; i < child_required_columns.size(); i++ )
                    {
                        if ( *col->GetTable() == *child_required_columns[ i ]->GetTable() &&
                             col->GetColumnName() == child_required_columns[ i ]->GetColumnName() )
                        {
                            found = true;
                            id = i;
                        }
                    }

                    ARIES_ASSERT( found, "cannot find column" + col->ToString() );
                    node->SetPositionForReferencedColumn( col, index + id );
                }

                index += child_required_columns.size();
            }
            else if ( child->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
            {
                resetup_join_node( child, child_required );
            }
            else if ( child->GetType() == SQLTreeNodeType::StarJoin_NODE )
            {
                resetup_star_join_node( child, child_required );
            }
        }
    }

    // node->SetRequiredColumnArray( required_columns );
    node->AddRequiredColumnArray( required_columns );
    if (node->GetSpoolId() > -1 && node->GetSameNode())
        node->GetSameNode()->AddRequiredColumnArray( required_columns );
    node->SetReferencedColumnArray( referenced_columns );
}

static bool replace_child(SQLTreeNodePointer origin, SQLTreeNodePointer target, SQLTreeNodePointer replace )
{
    for ( size_t i = 0; i < origin->GetChildCount(); i++ )
    {
        auto child = origin->GetChildByIndex( i );
        if ( child == target )
        {
            origin->CompletelyResetAChild( child, replace );
            replace->SetParent( origin );
            return true;
        }

        if ( replace_child( child, target, replace ) )
        {
            return true;
        }
    }
    return false;
}


/**
 * Star join 的各个 hash table 中如果存在过滤之后的数据，则优先使用过滤后的数据的 hash table
 * 来做 hash join，对 q9 有明显提升。
 */
bool contain_filter_node( const SQLTreeNodePointer& node )
{
    if ( node->GetType() == SQLTreeNodeType::Filter_NODE )
    {
#ifndef NDEBUG
        std::cout << "contains filter true" << std::endl;
#endif
        return true;
    }
    else if ( node->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
    {
        auto left_has_filter = contain_filter_node( node->GetLeftChild() );
        auto right_hash_filter = contain_filter_node( node->GetRightChild() );

        if ( node->GetJoinType() == JoinType::InnerJoin )
        {
            return left_has_filter || right_hash_filter;
        }
        else if ( node->GetJoinType() == JoinType::LeftJoin )
        {
            return left_has_filter;
        }
        else if ( node->GetJoinType() == JoinType::RightJoin )
        {
            return right_hash_filter;
        }
    }
    else if ( node->GetChildCount() == 1 )
    {
        return contain_filter_node( node->GetTheChild() );
    }

    return false;
}

void StarJoinBuilder::handleJoinNode( SQLTreeNodePointer arg_input )
{
    auto left = arg_input->GetLeftChild();
    auto right = arg_input->GetRightChild();
    
    if ( !arg_input->IsInnerJoin() || arg_input->GetSpoolId() != -1 )
    {
        handleNode( left );
        handleNode( right );
        return;
    }

    /**
     * pairs 在一连串的 inner join 中收集到的 hash join 对
     * other_nodes 在上述 inner join 中的非 hash join 中的节点
     * 比如: A <-- hash join --> B  <-- inner join --> D
     * 其中 A, B 构成一个 pair，D 是一个 other node
     */
    std::vector< HashJoinPair > pairs;
    std::vector< NonHashJoinNode > other_nodes;

    handleInnerJoinNode( arg_input, pairs, other_nodes );

    std::map< SQLTreeNodePointer, std::vector< HashJoinPair > > hash_join_map;

    bool has_star_join_node = false;
    for ( const auto& pair : pairs )
    {
        hash_join_map[ pair.value_node ].emplace_back( pair );
        if ( hash_join_map[ pair.value_node ].size() > 1 )
        {
            has_star_join_node = true;
        }
    }

    if ( !has_star_join_node )
    {
        return;
    }

    // 该 node 从其子节点获取到的 columns
    auto referenced_columns = arg_input->GetReferencedColumnArray();

    // 该 node 输出的 columns
    auto required_columns = arg_input->GetRequiredColumnArray();
    
    // 重新调整hash_join_map，按照star join长度由大到小排序．
    std::vector<SQLTreeNodePointer> key_node_in_hash_join_map;
    for( auto & it : hash_join_map )
        key_node_in_hash_join_map.push_back( it.first );
    std::sort( key_node_in_hash_join_map.begin(), key_node_in_hash_join_map.end(), [ & ]( const SQLTreeNodePointer& left, const SQLTreeNodePointer& right )
    {
        return hash_join_map[ left ].size() > hash_join_map[ right ].size();
    } );

    struct HashJoinPairHashHasher
    {
        uint64_t operator()(const HashJoinPair &k) const noexcept
        {
            return std::hash<uint64_t>{}( (uint64_t)k.hash_node.get() + (uint64_t)k.value_node.get() );
        }
    };

    struct HashJoinPairComparator
    {
        bool operator()( const HashJoinPair &a, const HashJoinPair &b ) const noexcept
        {
            return std::minmax( a.hash_node.get(), a.value_node.get() ) == std::minmax( b.hash_node.get(), b.value_node.get() );
        }
    };

    std::unordered_set< HashJoinPair, HashJoinPairHashHasher, HashJoinPairComparator > join_pairs;

    int hash_join_map_size = key_node_in_hash_join_map.size();
    assert( hash_join_map_size > 0 );
    // 先将最大star join的所有join pair添加;
    for( auto& it : hash_join_map[ key_node_in_hash_join_map[ 0 ] ] )
        join_pairs.insert( it ).second;
        
    //去掉其他重复的join pair
    std::vector< HashJoinPair > joinPairToKeep;
    for( int i = 1; i < hash_join_map_size; ++i )
    {
        joinPairToKeep.clear();
        for( auto& it : hash_join_map[ key_node_in_hash_join_map[ i ] ] )
        {
            if( join_pairs.insert( it ).second )
                joinPairToKeep.push_back( it );
        }
        if( joinPairToKeep.size() > 0 )
            hash_join_map[ key_node_in_hash_join_map[ i ] ] = joinPairToKeep;
        else
            hash_join_map.erase( key_node_in_hash_join_map[ i ] );
    }

    std::vector< SQLTreeNodePointer > new_hash_join_nodes;

    SQLTreeNodePointer new_join_node = nullptr;

    SQLTreeNodeBuilder builder( arg_input->GetMyQuery() );

    std::map< SQLTreeNodePointer, SQLTreeNodePointer > replace_map;
    //生成star join和hash join节点
    BiaodashiAuxProcessor expr_processor;
    for ( auto pair : hash_join_map )
    {
        if ( pair.second.size() > 1 )
        {
            std::vector< BiaodashiPointer > other_conditions;
            auto star_join_node = builder.makeTreeNode_StarJoin();
            star_join_node->AddChild( pair.first );
            pair.first->SetParent( star_join_node );

            std::sort( pair.second.begin(), pair.second.end(), []( const HashJoinPair& left, const HashJoinPair& right )
            {
                return ( contain_filter_node( left.hash_node ) || !contain_filter_node( right.hash_node ) );
            } );

            for ( auto& node : pair.second )
            {
                star_join_node->AddChild( node.hash_node );
                node.hash_node->SetParent( star_join_node );
                star_join_node->AddStarJoinCondition( node.conditions );

                if ( node.other_condition )
                {
                    other_conditions.emplace_back( node.other_condition );
                }
            }

            new_hash_join_nodes.emplace_back( star_join_node );
            replace_map[ pair.first ] = star_join_node;

            if ( !other_conditions.empty() )
            {
                star_join_node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_conditions ) );
            }
        }
        else
        {
            auto join_pair = pair.second[ 0 ];
            auto node = builder.makeTreeNode_BinaryJoin( JoinType::InnerJoin, join_pair.conditions[ 0 ] );


            std::vector< CommonBiaodashiPtr > other_conditions;
            if ( join_pair.conditions.size() > 1 )
            {
                other_conditions.assign( join_pair.conditions.cbegin() + 1, join_pair.conditions.cend() );
            }

            if ( join_pair.other_condition )
            {
                other_conditions.emplace_back( join_pair.other_condition );
            }

            if ( !other_conditions.empty() )
            {
                std::vector< BiaodashiPointer > conditions;
                for ( const auto& condition : other_conditions )
                {
                    conditions.emplace_back( condition );
                }
                node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( conditions ) );
            }

            SQLTreeNode::AddTreeNodeChild( node, join_pair.value_node );
            SQLTreeNode::AddTreeNodeChild( node, join_pair.hash_node );

            new_hash_join_nodes.emplace_back( node );
            replace_map[ pair.first ] = node;
        }  
    }

    // //遍历每个join的hash节点．查找是否可以被其他join整体替代（用来处理同一个节点在不同的join中分别是hash node和value node的场景,replace_map保存了相关对应关系)
    // std::set< SQLTreeNodePointer > toRemove;
    // for ( auto& node : new_hash_join_nodes )
    // {
    //     //避免循环join A->B->A
    //     if( toRemove.find( node ) != toRemove.end() )
    //         continue;
    //     for ( int i = 1; i < node->GetChildCount(); i++ )
    //     {
    //         auto child = node->GetChildByIndex( i );
    //         if ( replace_map.find( child ) != replace_map.cend() )
    //         {
    //             node->CompletelyResetAChild( child, replace_map[ child ] );
    //             replace_map[ child ]->SetParent( node );

    //             //该join被接入到外层的join，先记录一下，之后需要将其从new_join_nodes中删除
    //             toRemove.insert( replace_map[ child ] );

    //             node->AddRequiredColumnArray( std::vector< ColumnShellPointer >() );
    //             if(node->GetSpoolId() > -1 && node->GetSameNode())
    //                 node->GetSameNode()->AddRequiredColumnArray( std::vector< ColumnShellPointer >() );

    //             //处理完该child，可以从replace_map中删除
    //             replace_map.erase( child );
    //             break;
    //         }
    //     }
    // }

    // //去掉已经接入到其他star join的子join
    // std::vector< SQLTreeNodePointer > toKeep;
    // for ( auto& node : new_hash_join_nodes )
    // {
    //     if( toRemove.find( node ) == toRemove.end() )
    //         toKeep.emplace_back( node );
    // }
    // std::swap( toKeep, new_hash_join_nodes );

    // std::vector< bool > mask( new_hash_join_nodes.size() );
    // for ( int i = 0; i < new_hash_join_nodes.size(); i++ )
    // {
    //     auto& node = new_hash_join_nodes[ i ];
    //     auto center = node->GetChildByIndex( 0 );
    //     mask[ i ] = true;
    //     for ( int j = i + 1; j < new_hash_join_nodes.size(); j++ )
    //     {
    //         bool handled = false;
    //         auto& another_node = new_hash_join_nodes[ j ];
    //         handled = replace_child( another_node, center, node );
    //         if ( handled )
    //         {
    //             auto other_condition = node->GetJoinOtherCondition();
    //             if ( other_condition )
    //             {
    //                 std::vector< BiaodashiPointer > other_conditions;
    //                 other_conditions.emplace_back( other_condition );
    //                 if ( another_node->GetJoinOtherCondition() )
    //                 {
    //                     other_conditions.emplace_back( another_node->GetJoinOtherCondition() );
    //                 }

    //                 another_node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_conditions ) );
    //                 node->SetJoinOtherCondition( nullptr );
    //             }

    //             mask[ i ] = false;
    //             break;
    //         }


    //         auto center_of_another_node = another_node->GetChildByIndex( 0 );
    //         handled = replace_child( node, center_of_another_node, another_node );

    //         if ( handled )
    //         {
    //             auto another_other_condition = another_node->GetJoinOtherCondition();
    //             if ( another_other_condition )
    //             {
    //                 std::vector< BiaodashiPointer > other_conditions;
    //                 other_conditions.emplace_back( another_other_condition );
    //                 if ( node->GetJoinOtherCondition() )
    //                 {
    //                     other_conditions.emplace_back( node->GetJoinOtherCondition() );
    //                 }
    //                 node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_conditions ) );
    //                 another_node->SetJoinOtherCondition( nullptr );
    //             }

    //             new_hash_join_nodes[ j ] = node;
    //             mask[ i ] = 0;
    //             break;
    //         }
    //         if ( another_node->IsInnerJoin() )
    //         {
    //             auto center_of_another_node = another_node->GetChildByIndex( 1 );
    //             handled = replace_child( node, center_of_another_node, another_node );

    //             if ( handled )
    //             {
    //                 auto another_other_condition = another_node->GetJoinOtherCondition();
    //                 if ( another_other_condition )
    //                 {
    //                     std::vector< BiaodashiPointer > other_conditions;
    //                     other_conditions.emplace_back( another_other_condition );
    //                     if ( node->GetJoinOtherCondition() )
    //                     {
    //                         other_conditions.emplace_back( node->GetJoinOtherCondition() );
    //                     }
    //                     node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_conditions ) );
    //                     another_node->SetJoinOtherCondition( nullptr );
    //                 }

    //                 new_hash_join_nodes[ j ] = node;
    //                 mask[ i ] = 0;
    //                 break;
    //             }
    //         }
    //     }
    // }

    // std::vector< SQLTreeNodePointer > filtered_nodes;
    // for ( int i = 0; i < new_hash_join_nodes.size(); i++ )
    // {
    //     if ( mask[ i ] )
    //     {
    //         filtered_nodes.emplace_back( new_hash_join_nodes[ i ] );
    //     }
    // }

    // std::vector< SQLTreeNodePointer > nodes_without_condtion;
    // new_hash_join_nodes.clear();
    // nodes_without_condtion.assign( filtered_nodes.cbegin(), filtered_nodes.cend() );
    // filtered_nodes.clear();


    //遍历每个join的child节点．查找是否可以被其他join整体替代（replace_map保存了相关对应关系)
    //避免循环join A->B->A
    //对于已经连接到其他节点的join节点，用toRemove保存，方便之后删除
    std::set< SQLTreeNodePointer > toRemove;
    struct JoinNodePair
    {
        SQLTreeNodePointer left;
        SQLTreeNodePointer right;
    };

    struct HashJoinNodePairHashHasher
    {
        uint64_t operator()(const JoinNodePair &k) const noexcept
        {
            return std::hash<uint64_t>{}( (uint64_t)k.left.get() + (uint64_t)k.right.get() );
        }
    };

    struct HashJoinNodePairComparator
    {
        bool operator()( const JoinNodePair &a, const JoinNodePair &b ) const noexcept
        {
            return std::minmax( a.left.get(), a.right.get() ) == std::minmax( b.left.get(), b.right.get() );
        }
    };

    std::unordered_set< JoinNodePair, HashJoinNodePairHashHasher, HashJoinNodePairComparator > join_node_pairs;

    for( auto& node : replace_map )
    {
        for( auto & toReplace : replace_map )
        {
            if( node.first != toReplace.first && node.second != toReplace.second )
            {
                //这俩节点已经连接过了，跳过．避免A->B->A
                if( join_node_pairs.find( { node.second, toReplace.second } ) != join_node_pairs.end() )
                    continue;
                
                //节点未连接到其他节点，尝试连接 toReplace->node
                if( toRemove.find( toReplace.second ) == toRemove.end() && replace_child( node.second, toReplace.first, toReplace.second ) )
                {
                    auto other_condition = toReplace.second->GetJoinOtherCondition();
                    if ( other_condition )
                    {
                        std::vector< BiaodashiPointer > other_conditions;
                        other_conditions.emplace_back( other_condition );
                        if ( node.second->GetJoinOtherCondition() )
                        {
                            other_conditions.emplace_back( node.second->GetJoinOtherCondition() );
                        }
                        node.second->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_conditions ) );
                        toReplace.second->SetJoinOtherCondition( nullptr );
                    }
                    node.second->AddRequiredColumnArray( std::vector< ColumnShellPointer >() );
                    if( node.second->GetSpoolId() > -1 && node.second->GetSameNode() )
                        node.second->GetSameNode()->AddRequiredColumnArray( std::vector< ColumnShellPointer >() );
                    //成功连接，此节点未来可以删除
                    toRemove.insert( toReplace.second );
                    //保存连接关系，避免A->B->A
                    join_node_pairs.insert( { node.second, toReplace.second } );
                    replace_map[ node.first ] = node.second;
                    replace_map[ toReplace.first ] = node.second;
                    break;
                }

                //节点未连接到其他节点，尝试连接 node->toReplace
                if( toRemove.find( node.second ) == toRemove.end() && replace_child( toReplace.second, node.first, node.second ) )
                {
                    auto other_condition = node.second->GetJoinOtherCondition();
                    if ( other_condition )
                    {
                        std::vector< BiaodashiPointer > other_conditions;
                        other_conditions.emplace_back( other_condition );
                        if ( toReplace.second->GetJoinOtherCondition() )
                        {
                            other_conditions.emplace_back( toReplace.second->GetJoinOtherCondition() );
                        }
                        toReplace.second->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( other_conditions ) );
                        node.second->SetJoinOtherCondition( nullptr );
                    }
                    toReplace.second->AddRequiredColumnArray( std::vector< ColumnShellPointer >() );
                    if( toReplace.second->GetSpoolId() > -1 && toReplace.second->GetSameNode() )
                        toReplace.second->GetSameNode()->AddRequiredColumnArray( std::vector< ColumnShellPointer >() );
                    //成功连接，此节点未来可以删除
                    toRemove.insert( node.second );
                    //保存连接关系，避免A->B->A
                    join_node_pairs.insert( { toReplace.second, node.second } );
                    replace_map[ toReplace.first ] = toReplace.second;
                    replace_map[ node.first ] = toReplace.second;
                    break;
                }
            }
        }
    }

    //去掉已经接入到其他star join的子join
    std::vector< SQLTreeNodePointer > toKeep;
    for( auto& node : new_hash_join_nodes )
    {
        if( toRemove.find( node ) == toRemove.end() )
            toKeep.emplace_back( node );
    }
    std::swap( toKeep, new_hash_join_nodes );

    std::vector< SQLTreeNodePointer > nodes_without_condtion( std::move( new_hash_join_nodes ) );

    //处理other_nodes.如果other_nodes.node已经在某一个join中，则把node设为null，仅仅保留条件，用来连接nodes_without_condtion中的join
    std::vector< NonHashJoinNode > noHashToKeep;
    std::vector< NonHashJoinNode > noHashToCheck;
    for ( auto& node : other_nodes )
    {
        for( auto& join_node : nodes_without_condtion )
        {
            if( contain_node( join_node, node.node ) )
            {
                node.node = nullptr;
                //需要额外检查condition,再看是否能加入到noHashToKeep列表
                noHashToCheck.push_back( node );
                break;
            }
        }
        //一个没有在hash join中出现的节点，需要保留
        if( node.node )
            noHashToKeep.push_back( node );
    }

    for ( auto& node : noHashToCheck )
    {
        //如果noHashToKeep中包含了和hash join完全相同的condition．则不能加入到toKeep列表中(该condition可以直接连接两个hash join)
        bool bFound = false;
        for( auto& n : noHashToKeep )
        {
            if( *( node.condition->Normalized() ) == *( n.condition->Normalized() ) )
            {
                if( node.other_condition )
                {
                    if( n.other_condition )
                    {
                        bFound = ( *( node.other_condition->Normalized() ) == *( n.other_condition->Normalized() ) );
                        break;
                    }
                }
                else if( !n.other_condition )
                {
                    bFound = true;
                    break;
                }
            }
        }
        if( !bFound )
            noHashToKeep.push_back( node );
    }

    std::swap( other_nodes, noHashToKeep );
    for ( const auto& node : other_nodes )
    {
        auto condition = node.condition;
        condition->ObtainReferenceTableInfo();
        auto left_condition = std::dynamic_pointer_cast< CommonBiaodashi >( condition->GetChildByIndex( 0 ) );
        auto right_condition = std::dynamic_pointer_cast< CommonBiaodashi >( condition->GetChildByIndex( 1 ) );
        auto left_tables = left_condition->GetInvolvedTableList();
        auto right_tables = right_condition->GetInvolvedTableList();

        auto new_node = builder.makeTreeNode_BinaryJoin( JoinType::InnerJoin, condition );

        std::vector< BiaodashiPointer > new_node_other_conditions;
        if( node.other_condition )
            new_node_other_conditions.push_back( node.other_condition );
        int index;
        if( node.node )
        {
            //找到条件对应的hash join，与之连接
            index = -1;
            for ( size_t i = 0; i < nodes_without_condtion.size(); i++ )
            {
                if( !nodes_without_condtion[ i ] )
                    continue;
                auto& n = nodes_without_condtion[ i ];
                auto tables = n->GetInvolvedTableList();
                auto n_other_condition = n->GetJoinOtherCondition();
                if( vector_contains_all( tables, left_tables ) )
                {
                    if( n_other_condition )
                    {
                        new_node_other_conditions.push_back( n_other_condition );
                        n->SetJoinOtherCondition( nullptr );
                    }
                    if( !new_node_other_conditions.empty() )
                        new_node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( new_node_other_conditions ) );
                    SQLTreeNode::AddTreeNodeChild( new_node, n );
                    SQLTreeNode::AddTreeNodeChild( new_node, node.node );
                    index = i;
                    break;
                }
                else if( vector_contains_all( tables, right_tables ) )
                {
                    if( n_other_condition )
                    {
                        new_node_other_conditions.push_back( n_other_condition );
                        n->SetJoinOtherCondition( nullptr );
                    }
                    if( !new_node_other_conditions.empty() )
                        new_node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( new_node_other_conditions ) );
                    SQLTreeNode::AddTreeNodeChild( new_node, node.node );
                    SQLTreeNode::AddTreeNodeChild( new_node, n );
                    index = i;
                    break;
                }
            }
        }
        else
        {
            //直接连接两个hash join
            index = -1;
            SQLTreeNodePointer left = nullptr;
            SQLTreeNodePointer right = nullptr;
            for( size_t i = 0; i < nodes_without_condtion.size(); i++ )
            {
                if( !nodes_without_condtion[ i ] )
                    continue;
                auto& n = nodes_without_condtion[ i ];
                auto tables = n->GetInvolvedTableList();
                if( vector_contains_all( tables, left_tables ) )
                {
                    if( !left )
                        left = n;
                    nodes_without_condtion[ i ] = nullptr;
                    index = i;
                }
                else if( vector_contains_all( tables, right_tables ) )
                {
                    if( !right )
                        right = n;
                    nodes_without_condtion[ i ] = nullptr;
                    index = i;
                }
            }
            if( left && right )
            {
                auto left_other_condition = left->GetJoinOtherCondition();
                if( left_other_condition )
                {
                    new_node_other_conditions.push_back( left_other_condition );
                    left->SetJoinOtherCondition( nullptr );
                }
                    
                auto right_other_condition = right->GetJoinOtherCondition();
                if( right_other_condition )
                {
                    new_node_other_conditions.push_back( right_other_condition );
                    right->SetJoinOtherCondition( nullptr );
                }
                    
                if( !new_node_other_conditions.empty() )
                     new_node->SetJoinOtherCondition( expr_processor.make_biaodashi_from_and_list( new_node_other_conditions ) );
                SQLTreeNode::AddTreeNodeChild( new_node, left );
                SQLTreeNode::AddTreeNodeChild( new_node, right );
            }
            else
            {
                assert(0);
                return;
            }
        }
        if( index == -1 )
        {
            assert(0);
            return;
        }
        assert( index >= 0 && ( size_t )index < nodes_without_condtion.size() );    
        nodes_without_condtion[ index ] = new_node;
    }

    SQLTreeNodePointer new_node = nullptr;
    for ( size_t i = 0; i < nodes_without_condtion.size(); i++ )
    {
        if( nodes_without_condtion[ i ] )
        {
            if( new_node )
            {
                LOG( WARNING ) << "cann't handle this";
                return;
            }  
            new_node = nodes_without_condtion[ i ];
        }
    }

    if ( new_node->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
    {
        resetup_join_node( new_node, required_columns );
    }
    else if ( new_node->GetType() == SQLTreeNodeType::StarJoin_NODE )
    {
        resetup_star_join_node( new_node, required_columns );
    }
    
    arg_input->GetParent()->CompletelyResetAChild( arg_input, new_node );
    new_node->SetParent( arg_input->GetParent() );
}

std::string StarJoinBuilder::ToString()
{
    return "StarJoinBuilder";
}

SQLTreeNodePointer StarJoinBuilder::OptimizeTree( SQLTreeNodePointer arg_input )
{
    handleNode( arg_input );

    return arg_input;
}

};
