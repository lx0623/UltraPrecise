#include <algorithm>
#include "SpoolBuilder.h"

#include "frontend/BiaodashiAuxProcessor.h"
#include <algorithm>
#include <set>
#include <unordered_set>

namespace aries
{

SpoolBuilder::SpoolBuilder() : next_spool_id( 1 )
{
}

std::string SpoolBuilder::ToString()
{
    return "SpoolBuilder";
}

int SpoolBuilder::getNextSpoolId()
{
    return next_spool_id++;
}

namespace
{
    bool vector_contains_all( const std::vector< BasicRelPointer >& all, const std::vector< BasicRelPointer >& sub )
    {
        for ( const auto& s : sub )
        {
            bool found = false;
            for ( const auto& a : all )
            {
                if ( s == a )
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

    void collect_leaf_nodes( SQLTreeNodePointer root, std::vector< SQLTreeNodePointer >& leaf_nodes )
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

    bool is_same( const SQLTreeNodePointer& left, const SQLTreeNodePointer& right )
    {
        if ( left->GetType() != right->GetType() )
        {
            return false;
        }

        switch ( left->GetType() )
        {
            case SQLTreeNodeType::Table_NODE:
            {
                auto left_rel = left->GetBasicRel();
                auto right_rel = right->GetBasicRel();
                return left_rel->GetDb() == right_rel->GetDb() && left_rel->GetID() == right_rel->GetID();
            }
            case SQLTreeNodeType::Filter_NODE:
            {
                auto left_filter = std::dynamic_pointer_cast< CommonBiaodashi >( left->GetFilterStructure() );
                auto right_filter = std::dynamic_pointer_cast< CommonBiaodashi >( right->GetFilterStructure() );

                return *( left_filter->Normalized() ) == *( right_filter->Normalized() );
            }
            case SQLTreeNodeType::Group_NODE:
            {
                auto left_group = left->GetMyGroupbyStructure();
                auto right_group = right->GetMyGroupbyStructure();

                if ( left_group->GetGroupbyExprCount() != right_group->GetGroupbyExprCount() )
                {
                    return false;
                }

                std::map< int, bool > left_matched;
                std::map< int, bool > right_mathced;
                for ( size_t i = 0; i < left_group->GetGroupbyExprCount(); i++ )
                {
                    auto left_item = std::dynamic_pointer_cast< CommonBiaodashi >( left_group->GetGroupbyExpr( i ) );
                    for ( size_t j = 0; j < right_group->GetGroupbyExprCount(); j++ )
                    {
                        auto right_item = std::dynamic_pointer_cast< CommonBiaodashi >( right_group->GetGroupbyExpr( j ) );
                        if ( *( left_item->Normalized() ) == *( right_item->Normalized() ) )
                        {
                            left_matched[ i ] = true;
                            right_mathced[ j ] = true;
                            break;
                        }
                    }

                    if ( !left_matched[ i ] )
                    {
                        return false;
                    }
                }

                for ( size_t i = 0; i < left_group->GetGroupbyExprCount(); i++ )
                {
                    if ( !left_matched[ i ] || !right_mathced[ i  ] )
                    {
                        return false;
                    }
                }

                return true;
            }
            case SQLTreeNodeType::Sort_NODE:
            {
                auto left_sort = left->GetMyOrderbyStructure();
                auto right_sort = left->GetMyOrderbyStructure();

                if ( left_sort->GetOrderbyItemCount() != right_sort->GetOrderbyItemCount() )
                {
                    return false;
                }

                std::map< int, bool > left_matched;
                std::map< int, bool > right_mathced;
                for ( size_t i = 0; i < left_sort->GetOrderbyItemCount(); i++ )
                {
                    auto left_item = std::dynamic_pointer_cast< CommonBiaodashi >( left_sort->GetOrderbyItem( i ) );
                    auto left_diretion = left_sort->GetOrderbyDirection( i );
                    for ( size_t j = 0; j < right_sort->GetOrderbyItemCount(); j++ )
                    {
                        auto right_item = std::dynamic_pointer_cast< CommonBiaodashi >( right_sort->GetOrderbyItem( j ) );
                        auto right_diretion = left_sort->GetOrderbyDirection( i );
                        if ( *( left_item->Normalized() ) == *( right_item->Normalized() ) && left_diretion == right_diretion )
                        {
                            left_matched[ i ] = true;
                            right_mathced[ j ] = true;
                            break;
                        }
                    }

                    if ( !left_matched[ i ] )
                    {
                        return false;
                    }
                }

                for ( size_t i = 0; i < left_sort->GetOrderbyItemCount(); i++ )
                {
                    if ( !left_matched[ i ] || !right_mathced[ i  ] )
                    {
                        return false;
                    }
                }

                return true;
            }
            case SQLTreeNodeType::BinaryJoin_NODE:
            {
                return false;
                // TODO: 未来需要考虑如何处理binaryjoin节点．binaryjoin应该当成一个整体对待
                // if( left != right )
                // {
                //     if( left->GetJoinType() == right->GetJoinType() )
                //     {
                //         auto leftJoinCondition = std::dynamic_pointer_cast< CommonBiaodashi >( left->GetJoinCondition() );
                //         auto rightJoinCondition = std::dynamic_pointer_cast< CommonBiaodashi >( right->GetJoinCondition() );
                //         auto leftOtherCondition = std::dynamic_pointer_cast< CommonBiaodashi >( left->GetJoinOtherCondition() );
                //         auto rightOtherCondition = std::dynamic_pointer_cast< CommonBiaodashi >( right->GetJoinOtherCondition() );
                //         if( leftJoinCondition || rightJoinCondition )
                //         {
                //             if( leftJoinCondition && rightJoinCondition )
                //             {
                //                 if( *( leftJoinCondition->Normalized() ) != *( rightJoinCondition->Normalized() ) )
                //                     return false;
                //             }
                //             else
                //                 return false;
                //         }
                //         if( leftOtherCondition || rightOtherCondition )
                //         {
                //             if( leftOtherCondition && rightOtherCondition )
                //             {
                //                 if( *( leftOtherCondition->Normalized() ) != *( rightOtherCondition->Normalized() ) )
                //                     return false;
                //             }
                //             else
                //                 return false;
                //         }
                //         return is_same( left->GetLeftChild(), right->GetLeftChild() ) && is_same( left->GetRightChild(), right->GetRightChild() );
                //     }
                //     else 
                //         return false;
                // }
                // else
                //     return false;
            }
            default:
                return false;
        }
    }

    bool search_same_nodes( const SQLTreeNodePointer& left,
                               const SQLTreeNodePointer& right,
                               SQLTreeNodePointer& left_start,
                               SQLTreeNodePointer& right_start )
    {
        auto l = left;
        auto r = right;

        if ( left == right )
        {
            return false;
        }

        bool same = false;
        while ( is_same( l, r ) )
        {
            same = true;
            left_start = l;
            right_start = r;

            if ( !l->GetParent() || !r->GetParent() )
            {
                break;
            }

            l = l->GetParent();
            r = r->GetParent();
        }

        return same;
    }

    bool is_condition_matched( CommonBiaodashi& condition, SQLTreeNode& left, SQLTreeNode& right )
    {
        if ( !condition.IsEqualCondition() )
        {
            return false;
        }

        auto left_condition = std::dynamic_pointer_cast< CommonBiaodashi >( condition.GetChildByIndex( 0 ) );
        auto right_condition = std::dynamic_pointer_cast< CommonBiaodashi >( condition.GetChildByIndex( 1 ) );

        auto left_condition_tables = left_condition->GetInvolvedTableList();
        auto right_condition_tables = right_condition->GetInvolvedTableList();

        if ( left_condition_tables.empty() || right_condition_tables.empty() )
        {
            return false;
        }

        auto left_tables = left.GetInvolvedTableList();
        auto right_tables = right.GetInvolvedTableList();

        if ( ( vector_contains_all( left_tables, left_condition_tables) &&
            vector_contains_all( right_tables, right_condition_tables ) ) ||
            ( vector_contains_all( left_tables, right_condition_tables ) &&
            vector_contains_all( right_tables, left_condition_tables ) ) )
        {
            return true;
        }

        return false;
    }

    bool inner_join_node_contains( SQLTreeNodePointer join_node, SQLTreeNodePointer node )
    {
        auto left = join_node->GetLeftChild();
        auto right = join_node->GetRightChild();

        if ( left == node || right == node )
        {
            return true;
        }

        if ( left->GetType() == SQLTreeNodeType::BinaryJoin_NODE && left->GetJoinType() == JoinType::InnerJoin )
        {
            if ( inner_join_node_contains( left, node ) )
            {
                return true;
            }
        }
        else if ( right->GetType() == SQLTreeNodeType::BinaryJoin_NODE && right->GetJoinType() == JoinType::InnerJoin )
        {
            if ( inner_join_node_contains( right, node) )
            {
                return true;
            }
        }
        return false;
    }

    bool IsSameConditions( const std::vector< BiaodashiPointer >& left, const std::vector< BiaodashiPointer >& right )
    {
        bool bRet = false;
        if( left.size() == right.size() && !left.empty() )
        {   
            size_t matchCount = 0;
            set< int > rightMatched;
            for ( const auto& expr: left )
            {
                auto left_child = std::dynamic_pointer_cast< CommonBiaodashi >( expr );
                int rightIndex = 0;
                for ( const auto& right_expr : right )
                {
                    auto right_child = std::dynamic_pointer_cast< CommonBiaodashi >( right_expr );
                    if ( *left_child->Normalized() == *right_child->Normalized() )
                    {
                        ++matchCount;
                        rightMatched.insert( rightIndex );
                        break;
                    }
                    ++rightIndex;
                }
            }
            bRet = ( matchCount == left.size() && matchCount == rightMatched.size() );
        }
        return bRet;
    }

    // 表示两个子节点join的相关信息
    struct ChildNodeJoinInfo
    {
        SQLTreeNodePointer child1;
        SQLTreeNodePointer child2;
        std::vector< BiaodashiPointer > conditions;
        bool operator==( const ChildNodeJoinInfo& src ) const
        {
            return IsSameConditions( conditions, src.conditions );
        }
    };

    // 存储join点下所有子节点的join条件信息
    struct JoinInfo
    {
        SQLTreeNodePointer join_node;//在处理过程中，会逐渐生成join_node二叉树
        std::vector< ChildNodeJoinInfo > child_join_info;//在生成join_node二叉树过程中，会把用到的条件一个个删除．
        bool RemoveChildJoinInfo( const ChildNodeJoinInfo& src )
        {
            for( auto it = child_join_info.begin(); it != child_join_info.end(); ++it ) 
            {
                if( *it  == src ) 
                {
                    child_join_info.erase( it );
                    return true;
                }
            }
            return false;
        }
        std::set< SQLTreeNodePointer > used_nodes;//使用过的节点，未来会从parent中删除
        std::set< BiaodashiPointer > used_conditions;//使用过的条件，未来会从parent的条件中删除
    };

    //从origin为根节点的树中，查找target节点．找到了就返回true，否则返回false
    bool IsTargetNodeExists( SQLTreeNodePointer origin, SQLTreeNodePointer target )
    {
        if( origin == target )
            return true;
        for ( size_t i = 0; i < origin->GetChildCount(); ++i )
        {
            auto child = origin->GetChildByIndex( i );
            if ( child == target )
                return true;
            else if( IsTargetNodeExists( child, target ) )
                return true;
        }
        return false;
    }

    //从其他节点查找相同的子树及join条件
    std::map< SQLTreeNodePointer, ChildNodeJoinInfo > FindOtherSameChildJoinInfo( const SQLTreeNodePointer& srcJoinNode, 
                                                                                    const ChildNodeJoinInfo& srcChildJoinInfo, 
                                                                                    const std::vector< SQLTreeNodePointer >& allParents, 
                                                                                    size_t skipIndex, 
                                                                                    const std::map< SQLTreeNodePointer, JoinInfo >& joinInfoMap )
    {
        std::map< SQLTreeNodePointer, ChildNodeJoinInfo > result;
        for( size_t i = 0; i < allParents.size(); ++i )
        {
            if( i == skipIndex )
                continue;
            auto it = joinInfoMap.find( allParents[ i ] );
            assert( it != joinInfoMap.end() );

            int spoolIdSrc = srcJoinNode ? srcJoinNode->GetSpoolId() : -1;
            int spoolIdOther = it->second.join_node ? it->second.join_node->GetSpoolId() : -1;
            //相同的二叉树
            if( spoolIdSrc == spoolIdOther )
            {
                for( const auto& info : it->second.child_join_info )
                {
                    //相同的条件
                    if( info == srcChildJoinInfo )
                    {
                        result.insert( { it->first, info } );
                        break;
                    }
                }
            }
        }
        return result;
    }
}

SQLTreeNodePointer SpoolBuilder::OptimizeTree( SQLTreeNodePointer node )
{
    /**
     * 先收集所有的叶子节点
     */
    std::vector< SQLTreeNodePointer > leaf_nodes;
    collect_leaf_nodes( node, leaf_nodes );

    /**
     * (Inner) Join 节点下的相同子树
     */
    std::map< SQLTreeNodePointer, std::vector< SQLTreeNodePointer > > join_same_nodes;

    std::map< SQLTreeNodePointer, std::set< SQLTreeNodePointer > > tmp_join_same_nodes;

    /**
     * 相同子树的映射，比如子树 a 和 子树 b 相同，那么这里会产生两个元素：
     * { a: b } 和 { b: a }，方便快速找到与其匹配的子树
     */
    std::map< SQLTreeNodePointer, std::vector< SQLTreeNodePointer > > same_nodes_map;

    auto query = std::dynamic_pointer_cast< SelectStructure >( node->GetMyQuery() );

    /**
     * 这里开始遍历每棵子树，从叶子节点依次往下判断
     */
    for ( size_t i = 0; i < leaf_nodes.size(); i++ )
    {
        auto& left = leaf_nodes[ i ];
        for ( size_t j = i + 1; j < leaf_nodes.size(); j++ )
        {
            auto& right = leaf_nodes[ j ];
            SQLTreeNodePointer left_start = nullptr;
            SQLTreeNodePointer right_start = nullptr;
            if ( search_same_nodes( left, right, left_start, right_start ) )
            {
                same_nodes_map[ left_start ].push_back( right_start );
                same_nodes_map[ right_start ].push_back( left_start );

                if ( left_start->GetType() != SQLTreeNodeType::Table_NODE )
                {
                    auto spool_id = getNextSpoolId();
                    left_start->SetSpoolId( spool_id );
                    right_start->SetSpoolId( spool_id );
                    left_start->SetSameNode( right_start );
                    right_start->SetSameNode( left_start );
                }

                auto left_alias = left->GetBasicRel()->GetAliasNamePointer();
                auto right_alias = right->GetBasicRel()->GetAliasNamePointer();
                if ( left_alias || right_alias )
                {
                    left_start->SetSpoolAlias( right_alias ? *right_alias : right->GetBasicRel()->GetID(),
                                               left_alias ? *left_alias : left->GetBasicRel()->GetID() );
                    query->SetSpoolAlias( right_alias ? *right_alias : right->GetBasicRel()->GetID(),
                                         left_alias ? *left_alias : left->GetBasicRel()->GetID() );

                    right_start->SetSpoolAlias( left_alias ? *left_alias : left->GetBasicRel()->GetID(),
                                               right_alias ? *right_alias : right->GetBasicRel()->GetID() );
                    query->SetSpoolAlias( left_alias ? *left_alias : left->GetBasicRel()->GetID(),
                                          right_alias ? *right_alias : right->GetBasicRel()->GetID() );
                }
                auto left_parent = left_start->GetParent();
                auto right_parent = right_start->GetParent();
                if( left_parent && right_parent && left_parent != right_parent && left_parent->GetType() == right_parent->GetType() )
                {
                    if ( left_parent->GetType() == SQLTreeNodeType::InnerJoin_NODE )
                    {
                        tmp_join_same_nodes[ left_parent ].insert( left_start );
                        tmp_join_same_nodes[ right_parent ].insert( right_start );
                    }
                    else if ( left_parent->GetType() == SQLTreeNodeType::BinaryJoin_NODE )
                    {
                        //目前处理parent为BinaryJoin_NODE的场景是错的．暂时跳过对此情况的处理
                        //TODO:实际上应该把BinaryJoin_NODE当成一个原子节点进行判断．

                        // auto left_join_type = left_start->GetParent()->GetJoinType();
                        // auto right_join_type = right_start->GetParent()->GetJoinType();
                        // if ( left_join_type == JoinType::RightJoin )
                        // {
                        //     left_join_type = JoinType::LeftJoin;
                        //     left_start->GetParent()->SetJoinType( left_join_type );
                        //     auto left = left_start->GetParent()->GetLeftChild();
                        //     auto right = left_start->GetParent()->GetRightChild();

                        //     left_start->GetParent()->ResetLeftChild( right );
                        //     right->SetParent( left_start->GetParent() );
                        //     left_start->GetParent()->ResetRightChild( left );
                        //     left->SetParent( left_start->GetParent() );
                        // }

                        // if ( right_join_type == JoinType::RightJoin )
                        // {
                        //     right_join_type = JoinType::LeftJoin;
                        //     right_start->GetParent()->SetJoinType( right_join_type );
                        //     auto left = right_start->GetParent()->GetLeftChild();
                        //     auto right = right_start->GetParent()->GetRightChild();

                        //     right_start->GetParent()->ResetLeftChild( right );
                        //     right->SetParent( right_start->GetParent() );
                        //     right_start->GetParent()->ResetRightChild( left );
                        //     left->SetParent( right_start->GetParent() );
                        // }

                        // if ( left_join_type == right_join_type )
                        // {
                        //     SQLTreeNodePointer child_node;
                        //     if ( left_start == left_start->GetParent()->GetLeftChild() )
                        //     {
                        //         child_node = right_start->GetParent()->GetLeftChild();
                        //     }
                        //     else
                        //     {
                        //         child_node = right_start->GetParent()->GetRightChild();
                        //     }

                        //     if ( child_node.get() == right_start.get() )
                        //     {
                        //         join_same_nodes[ left_start->GetParent() ].emplace_back( left_start );
                        //         tmp_join_same_nodes[ left_start->GetParent() ].insert( left_start );

                        //         join_same_nodes[ right_start->GetParent() ].emplace_back( right_start );
                        //         tmp_join_same_nodes[ right_start->GetParent() ].insert( right_start );
                        //     }
                        // }
                    }
                }
            }
        }
    }

    for( auto& it : tmp_join_same_nodes )
        join_same_nodes[ it.first ] = { it.second.begin(), it.second.end() };

    //没有更多工作要做，可以退出了
    if( join_same_nodes.empty() )
        return node;

    // 遍历所有join节点，分别找出join节点内部的join关系
    std::map< SQLTreeNodePointer, JoinInfo > join_info_map;
    BiaodashiAuxProcessor processor;
    for( auto& it : join_same_nodes )
    {
        if( it.second.size() < 2 )
            continue;

        std::vector< std::pair< SQLTreeNodePointer, SQLTreeNodePointer > > matched_pairs;
        std::vector< std::vector< BiaodashiPointer > > all_matched_conditions;
        for ( size_t i = 0; i < it.second.size(); i++ )
        {
            auto& first = it.second[ i ];
            for ( size_t j = i + 1; j < it.second.size(); j++ )
            {
                auto& second = it.second[ j ];
                std::vector< BiaodashiPointer > matched_conditions;
                std::vector< BiaodashiPointer > conditions;

                if ( it.first->GetType() == SQLTreeNodeType::InnerJoin_NODE )
                {
                    conditions = it.first->GetInnerJoinConditions();
                }
                else
                {
                    auto condition = it.first->GetJoinCondition();
                    conditions = processor.generate_and_list( condition );
                }

                for ( const auto& condition : conditions )
                {
                    if ( is_condition_matched( *( ( CommonBiaodashi* )( condition.get() ) ), *first, *second ) )
                    {
                        matched_conditions.emplace_back( condition );
                    }
                }

                if ( !matched_conditions.empty() )
                {
                    JoinInfo& joinInfo = join_info_map[ it.first ];
                    joinInfo.child_join_info.push_back( { first, second, matched_conditions } );
                }
            }
        }
    }

    //把所有join类型的父节点保存在vector里，方便以后遍历，两两比较
    std::vector< SQLTreeNodePointer > parent_join_nodes;

    for( auto& it : join_info_map )
        parent_join_nodes.push_back( it.first );

    SQLTreeNodeBuilder builder( node->GetMyQuery() );
    size_t parent_count = parent_join_nodes.size();
    if( parent_count > 2 )
        return node;//目前无法处理多个table spool
    for( size_t i = 0; i < parent_count; ++i )
    {
        SQLTreeNodePointer srcParent = parent_join_nodes[ i ];
        JoinInfo& srcJoinInfo = join_info_map[ srcParent ];
        SQLTreeNodePointer& srcJoinNode = srcJoinInfo.join_node;
        size_t joinInfoCount = srcJoinInfo.child_join_info.size();
        if( joinInfoCount > 0 )
        {
            vector< bool > usedInfo;
            usedInfo.resize( joinInfoCount, false );//被使用的child_join_info标志位，true表示已经连接到join_node,此条数据在未来可以忽略

            set< size_t > checkedIndex;
            //遍历child_join_info，找到第一个可以形成table spool的条件
            while( checkedIndex.size() < joinInfoCount )
            {
                size_t n = 0;
                //找到一个可以连接到现有的join_node的条件．
                if( srcJoinNode )
                {
                    size_t searchIndex = 0;
                    for( ; searchIndex < joinInfoCount; ++searchIndex )
                    {
                        if( checkedIndex.find( searchIndex ) == checkedIndex.end() )
                        {
                            if( IsTargetNodeExists( srcJoinNode, srcJoinInfo.child_join_info[ searchIndex ].child1 ) || IsTargetNodeExists( srcJoinNode, srcJoinInfo.child_join_info[ searchIndex ].child2 ) )
                            {
                                break;
                            }
                        }
                    }
                    n = searchIndex;
                }
                if( n == joinInfoCount )
                    break;
                checkedIndex.insert( n );
                const auto& srcChildJoinInfo = srcJoinInfo.child_join_info[ n ];
                std::map< SQLTreeNodePointer, ChildNodeJoinInfo > otherSameChildren = FindOtherSameChildJoinInfo( srcJoinNode, srcChildJoinInfo, parent_join_nodes, i, join_info_map );
                if( !otherSameChildren.empty() && otherSameChildren.size() == 1 )
                {
                    //有相同的条件，可以形成table spool
                    usedInfo[ n ] = true;
                    auto new_node = builder.makeTreeNode_BinaryJoin( JoinType::InnerJoin, processor.make_biaodashi_from_and_list( srcChildJoinInfo.conditions ) );
                    SQLTreeNode::AddTreeNodeChild( new_node, srcChildJoinInfo.child1 );
                    SQLTreeNode::AddTreeNodeChild( new_node, srcChildJoinInfo.child2 );

                    srcJoinInfo.used_nodes.insert( srcChildJoinInfo.child1 );
                    srcJoinInfo.used_nodes.insert( srcChildJoinInfo.child2 );

                    srcJoinInfo.used_conditions.insert( srcChildJoinInfo.conditions.begin(), srcChildJoinInfo.conditions.end() );

                    //先处理自己的joininfo
                    if( !srcJoinNode )
                    {
                        //join_node不存在，生成一个新的
                        srcJoinNode = new_node;
                    }
                    else
                    {
                        //需要和原来的join_node进行merge
                        //把新节点设置为parent节点，确保生成倾斜的join二叉树
                        if ( inner_join_node_contains( srcJoinNode, new_node->GetLeftChild() ) )
                        {
                            new_node->ResetLeftChild( srcJoinNode );
                        }
                        else if ( inner_join_node_contains( srcJoinNode, new_node->GetRightChild() ) )
                        {
                            new_node->ResetRightChild( srcJoinNode );
                        }
                        else
                        {
                            assert(0);
                        }
                        srcJoinNode->SetParent( new_node );                               
                        srcJoinNode = new_node;
                        srcJoinNode->ReCalculateInvolvedTableList();
                    }
                    auto spool_id = getNextSpoolId();
                    srcJoinNode->SetSpoolId( spool_id );

                    //处理其他相同节点,逻辑类似
                    for( const auto& other : otherSameChildren )
                    {
                        //有相同的条件，可以形成table spool
                        auto new_node = builder.makeTreeNode_BinaryJoin( JoinType::InnerJoin, processor.make_biaodashi_from_and_list( other.second.conditions ) );
                        SQLTreeNode::AddTreeNodeChild( new_node,  other.second.child1  );
                        SQLTreeNode::AddTreeNodeChild( new_node,  other.second.child2  );

                        auto itOther = join_info_map.find( other.first );
                        assert( itOther != join_info_map.end() );

                        itOther->second.used_nodes.insert( other.second.child1 );
                        itOther->second.used_nodes.insert( other.second.child2 );

                        itOther->second.used_conditions.insert( other.second.conditions.begin(), other.second.conditions.end() );
                        //先处理自己的joininfo
                        if( !itOther->second.join_node )
                        {
                            //join_node不存在，生成一个新的
                            itOther->second.join_node = new_node;
                        }
                        else
                        {
                            //需要和原来的join_node进行merge
                            //把新节点设置为parent节点，确保生成倾斜的join二叉树
                            if ( inner_join_node_contains( itOther->second.join_node, new_node->GetLeftChild() ) )
                            {
                                new_node->ResetLeftChild( itOther->second.join_node );
                            }
                            else if ( inner_join_node_contains( itOther->second.join_node, new_node->GetRightChild() ) )
                            {
                                new_node->ResetRightChild( itOther->second.join_node );
                            }
                            else
                            {
                                assert(0);
                            }
                            itOther->second.join_node->SetParent( new_node );
                            itOther->second.join_node = new_node;
                            itOther->second.join_node->ReCalculateInvolvedTableList();
                        }
                        itOther->second.join_node->SetSpoolId( spool_id );
                        //连接到join节点后，去掉使用过的join info
                        itOther->second.RemoveChildJoinInfo( other.second );
                        // TODO　对于多个spool具有相同ＩＤ的情况还无法处理．目前只能把其中一对当成table spool
                        srcJoinNode->SetSameNode( itOther->second.join_node );
                        itOther->second.join_node->SetSameNode( srcJoinNode );
                    }
                }
            }

            //去掉连接到join node的join info
            std::vector< ChildNodeJoinInfo > toKeep;
            for( size_t i = 0; i < joinInfoCount; ++i )
            {
                if( !usedInfo[ i ] )
                    toKeep.push_back( srcJoinInfo.child_join_info[ i ] );
            }
            std::swap( srcJoinInfo.child_join_info, toKeep );
        }
    }
    
    //处理剩下未连接的子节点和join条件
    for( auto& joinInfo : join_info_map )
    {
        //从parent节点中找到所有没有处理过的子节点
        std::vector< SQLTreeNodePointer > new_children;
        for ( size_t i = 0; i < joinInfo.first->GetChildCount(); i++ )
        {
            auto child = joinInfo.first->GetChildByIndex( i );
            
            if ( joinInfo.second.used_nodes.find( child ) == joinInfo.second.used_nodes.end() )
            {
                new_children.emplace_back( child );
            }
        }
        //去掉所有子节点
        joinInfo.first->ClearChildren();
        //添加未处理的子节点
        for( auto& child : new_children )
            SQLTreeNode::AddTreeNodeChild( joinInfo.first, child );

        //从parent节点中删除已经处理过的join条件
        auto& inner_join_conditions = joinInfo.first->GetInnerJoinConditions();
        std::vector< BiaodashiPointer > new_conditions;
        for( auto& old_condition : inner_join_conditions )
        {
            if( joinInfo.second.used_conditions.find( old_condition ) == joinInfo.second.used_conditions.end() )
                new_conditions.push_back( old_condition );
        }
        std::swap( inner_join_conditions, new_conditions );

        //添加处理过的二叉树根节点
        if( joinInfo.second.join_node )
            SQLTreeNode::AddTreeNodeChild( joinInfo.first, joinInfo.second.join_node );

        //添加没有连接到二叉树的字节点
        for( auto& childJoinInfo : joinInfo.second.child_join_info )
        {
            if( !inner_join_node_contains( joinInfo.first, childJoinInfo.child1 ) )
                SQLTreeNode::AddTreeNodeChild( joinInfo.first, childJoinInfo.child1 );
            if( !inner_join_node_contains( joinInfo.first, childJoinInfo.child2 ) )
                SQLTreeNode::AddTreeNodeChild( joinInfo.first, childJoinInfo.child2 );
        }
    }
    return node;
}

void SpoolBuilder::handleSubQueries( const std::vector< AbstractQueryPointer >& sub_queries )
{
    for ( const auto& query : sub_queries )
    {
        auto select_query = std::dynamic_pointer_cast< SelectStructure >( query );
    }
}

} //namespace aries