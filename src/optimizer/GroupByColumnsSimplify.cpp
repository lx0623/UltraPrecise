#include <glog/logging.h>
#include "GroupByColumnsSimplify.h"
#include "schema/SchemaManager.h"
#include "frontend/SelectStructure.h"

using namespace aries::schema;

namespace aries {
std::string GroupByColumnsSimplify::ToString() {
    return std::string("GroupByColumnsSimplify -- remove unnecessary group columns");
}

SQLTreeNodePointer GroupByColumnsSimplify::OptimizeTree(SQLTreeNodePointer arg_input)
{
    handleNode( arg_input );
    return arg_input;
}
void GroupByColumnsSimplify::handleNode(SQLTreeNodePointer arg_input)
{
    if (arg_input == nullptr)
        return;

    switch (arg_input->GetType())
    {
        case SQLTreeNodeType::Column_NODE:
        case SQLTreeNodeType::BinaryJoin_NODE:
        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Filter_NODE:
            handleNode( arg_input->GetTheChild() );
            break;
        case SQLTreeNodeType::Group_NODE:
            handleGroupNode( arg_input );
            break;
        case SQLTreeNodeType::Table_NODE:
            // do nothing
            break;
        default:
            LOG(ERROR) << "unhandled node type: " << static_cast<int>(arg_input->GetType());
            break;
    }
}

void GroupByColumnsSimplify::handleGroupNode( SQLTreeNodePointer arg_input )
{
    SelectStructure *the_ss = (SelectStructure *) ((arg_input->GetMyQuery()).get());

    GroupbyStructurePointer the_groupby_part = the_ss->GetGroupbyPart();
    vector< BiaodashiPointer > old_groupby_exprs = the_groupby_part->GetGroupbyExprs();
    if ( old_groupby_exprs.size() < 2 )
        return;

    vector< BiaodashiPointer > new_groupby_exprs;

    set< BiaodashiPointer, BiaodashiComparator > groupby_exprs_set;
    groupby_exprs_set.insert( old_groupby_exprs.begin(), old_groupby_exprs.end() );
    vector< BiaodashiPointer > unresolved_groupby_exprs;
    unresolved_groupby_exprs.insert( unresolved_groupby_exprs.end(), groupby_exprs_set.begin(), groupby_exprs_set.end() );

    auto schema = SchemaManager::GetInstance()->GetSchema();

    auto node_tables = arg_input->GetInvolvedTableList();
    auto defaultSA = the_ss->GetDefaultSchemaAgent();

    for ( auto table : node_tables )
    {
        if ( table->IsSubquery() )
            continue;
        if ( 0 == unresolved_groupby_exprs.size() )
            break;

        auto cur_db_name = table->GetDb();
        if ( cur_db_name.empty() )
        {
            if ( !defaultSA )
                ARIES_EXCEPTION(ER_NO_DB_ERROR);
            cur_db_name = defaultSA->schema->GetName();
        }

        auto cur_table_name = table->GetID();
        auto db_entry = schema->GetDatabaseByName( cur_db_name );
        auto table_entry = db_entry->GetTableByName( cur_table_name );

        const auto uniq_keys = table_entry->GetAllUniqueKeys();
        for ( const auto& key : uniq_keys )
        {
            // 对table的每个unique key, 检查组成Key的所有字段是否都参与group by
            size_t in_groupby_keys_count = 0;

            vector< BiaodashiPointer > groupby_key_exprs;
            vector< BiaodashiPointer > pending_exprs; // 待定的列
            vector< BiaodashiPointer > added_exprs; // 确定保留的列
            for ( size_t i = 0; i < unresolved_groupby_exprs.size(); i++ )
            {
                CommonBiaodashi *a_gb_expr = (CommonBiaodashi *) ( unresolved_groupby_exprs[ i ].get() );
                if ( a_gb_expr->GetType() == BiaodashiType::Lie )
                {
                    ColumnShellPointer gb_column_shell = boost::get<ColumnShellPointer>(a_gb_expr->GetContent());
                    auto gb_rel =  gb_column_shell->GetTable();
                    if( !gb_rel )
                    {
                        //看看是否自己是别名
                        if( gb_column_shell->GetExpr4Alias() )
                        {
                            auto realExpr = std::dynamic_pointer_cast<CommonBiaodashi>( gb_column_shell->GetExpr4Alias() );
                            if( realExpr->GetType() == BiaodashiType::Lie )
                            {
                                gb_column_shell = boost::get<ColumnShellPointer>(realExpr->GetContent());
                                gb_rel = gb_column_shell->GetTable();
                            }
                        }
                    }
                    if ( !gb_rel || gb_rel->IsSubquery() )
                    {
                        new_groupby_exprs.emplace_back( unresolved_groupby_exprs[ i ] );
                        added_exprs.emplace_back( unresolved_groupby_exprs[ i ] );
                        continue;
                    }

                    bool is_key_col = false;
                    auto gb_db_name = gb_rel->GetDb();
                    if ( gb_db_name.empty() )
                    {
                        if ( !defaultSA )
                            ARIES_EXCEPTION(ER_NO_DB_ERROR);
                        gb_db_name = defaultSA->schema->GetName();
                    }
                    // 检查此group by列是否是此表此unikey的成员
                    for ( const auto& key_col_name : key->keys )
                    {
                        if ( gb_column_shell->GetColumnName() == key_col_name
                             && gb_db_name == cur_db_name
                             && gb_rel->GetID() == cur_table_name )
                        {
                            is_key_col = true;
                            ++in_groupby_keys_count;
                            groupby_key_exprs.emplace_back( unresolved_groupby_exprs[ i ] );
                            break;
                        }
                    }
                    if ( !is_key_col )
                    {
                        // 当前表的非key字段
                        if ( gb_db_name == cur_db_name && gb_rel->GetID() == cur_table_name )
                        {
                            // 有可能是冗余字段
                        }
                        else
                        {
                            pending_exprs.emplace_back( unresolved_groupby_exprs[ i ] );
                        }
                    }
                }
                else
                {
                    new_groupby_exprs.emplace_back( unresolved_groupby_exprs[ i ] );
                    added_exprs.emplace_back( unresolved_groupby_exprs[ i ] );
                }
            } // END of for ( int i = 0; i < unresolved_groupby_exprs.size(); i++ )

            // unique key的所有字段都参与了group by
            if ( in_groupby_keys_count == key->keys.size() )
            {
                new_groupby_exprs.insert(
                    new_groupby_exprs.end(),
                    groupby_key_exprs.begin(),
                    groupby_key_exprs.end() );

                std::swap( unresolved_groupby_exprs, pending_exprs );

                // 如果一个table有多个unique key，只需要一个unique key参与group 就足够了,
                // 即使还有有其他unique key字段也参与了group by，也可以忽略
                break;
            }
            else
            {
                if ( added_exprs.size() > 0 )
                {
                    vector< BiaodashiPointer > tmp_exprs;
                    for ( auto& unresolved_expr : unresolved_groupby_exprs )
                    {
                        bool added = false;
                        for ( auto& added_expr : added_exprs )
                        {
                            if ( unresolved_expr == added_expr )
                            {
                                added = true;
                                break;
                            }
                        }
                        if ( !added )
                            tmp_exprs.emplace_back( unresolved_expr );
                    }
                    std::swap( unresolved_groupby_exprs, tmp_exprs );
                }
            }
        } // END of for ( const auto& key : uniq_keys )
    } // END of for ( auto table : node_tables )

    if ( unresolved_groupby_exprs.size() > 0 )
    {
        new_groupby_exprs.insert(
            new_groupby_exprs.end(),
            unresolved_groupby_exprs.begin(),
            unresolved_groupby_exprs.end() );
    }
    the_groupby_part->SetGroupbyExprs( new_groupby_exprs );
}

}