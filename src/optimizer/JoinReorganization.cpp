#include <algorithm>

#include "JoinReorganization.h"

#include "frontend/BiaodashiAuxProcessor.h"
#include "frontend/SelectStructure.h"
#include "schema/SchemaManager.h"

namespace aries
{

std::string JoinReorganization::ToString()
{
    return std::string("JoinReorganization");
}

SQLTreeNodePointer JoinReorganization::OptimizeTree(SQLTreeNodePointer arg_input)
{
    handleNode(arg_input);
    return arg_input;
}

void JoinReorganization::handleNode(SQLTreeNodePointer arg_input)
{
    switch (arg_input->GetType())
    {
        case SQLTreeNodeType::Column_NODE:
        case SQLTreeNodeType::Group_NODE:
        case SQLTreeNodeType::Sort_NODE:
        case SQLTreeNodeType::Limit_NODE:
        case SQLTreeNodeType::Filter_NODE:
            handleNode(arg_input->GetTheChild());
            break;
        case SQLTreeNodeType::BinaryJoin_NODE:
            handleJoinNode(arg_input);
            break;
        case SQLTreeNodeType::Table_NODE:
            // do nothing
            break;
        default:
            LOG(ERROR) << "unhandled node type: " << static_cast<int>(arg_input->GetType());
            break;
    }
}

void JoinReorganization::handleFilterNode(SQLTreeNodePointer arg_input)
{
    auto the_child = arg_input->GetTheChild();
    if (the_child->GetType() == SQLTreeNodeType::BinaryJoin_NODE)
    {
        arg_input->GetFilterStructure();
    }
    else
    {
        return handleNode(the_child);
    }
}

static bool isTrueCondition(BiaodashiPointer arg_condition)
{
    auto condition = std::dynamic_pointer_cast<CommonBiaodashi>(arg_condition);
    return condition->GetType() == BiaodashiType::Zhenjia && boost::get<bool>(condition->GetContent());
}

static void collectInnerNodes(SQLTreeNodePointer arg_node, std::vector<SQLTreeNodePointer>& nodes, std::vector<BiaodashiPointer>& conditions)
{
    if (arg_node->GetType() != SQLTreeNodeType::BinaryJoin_NODE || arg_node->GetJoinType() != JoinType::InnerJoin)
    {
        return;
    }

    if (!isTrueCondition(arg_node->GetJoinCondition()))
    {
        conditions.emplace_back(arg_node->GetJoinCondition());
    }

    auto left = arg_node->GetLeftChild();
    auto right = arg_node->GetRightChild();

    if (left->GetType() == SQLTreeNodeType::BinaryJoin_NODE && left->GetJoinType() == JoinType::InnerJoin)
    {
        collectInnerNodes(left, nodes, conditions);
    }
    else
    {
        nodes.emplace_back(left);
        // collectInnerNodes(left, nodes, conditions);
    }

    if (right->GetType() == SQLTreeNodeType::BinaryJoin_NODE && right->GetJoinType() == JoinType::InnerJoin)
    {
        collectInnerNodes(left, nodes, conditions);
    }
    else
    {
        nodes.emplace_back(right);
    }
}

static int64_t 
getDifferRowCount(const BasicRelPointer& table1,
                  const BasicRelPointer& table2, 
                  schema::Schema* schema, 
                  const std::string& default_database_name)
{
    if (table1->IsSubquery() || table2->IsSubquery()) {
        return 0;
    }
    auto left_table_name = table1->GetID();
    auto right_table_name = table2->GetID();

    auto left_database_name = table1->GetDb().empty() ? default_database_name : table1->GetDb();
    auto right_database_name = table2->GetDb().empty() ? default_database_name : table2->GetDb();

    auto left_table = schema->GetDatabaseByName(left_database_name)->GetTableByName(left_table_name);
    auto right_table = schema->GetDatabaseByName(right_database_name)->GetTableByName(right_table_name);
    int64_t differ = left_table->GetRowCount() - right_table->GetRowCount();

    return differ < 0 ? - differ : differ;
}

void JoinReorganization::handleJoinNode(SQLTreeNodePointer arg_input)
{
    if (arg_input->GetJoinType() != JoinType::InnerJoin)
    {
        handleNode(arg_input->GetLeftChild());
        handleNode(arg_input->GetRightChild());
        return;
    }

    std::vector<SQLTreeNodePointer> join_sub_nodes;
    std::vector<BiaodashiPointer> conditions;
    collectInnerNodes(arg_input, join_sub_nodes, conditions);

    if (join_sub_nodes.size() <= 2)
    {
        return;
    }

    auto parent = arg_input->GetParent();

    BiaodashiAuxProcessor expr_processor;
    if (parent && parent->GetType() == SQLTreeNodeType::Filter_NODE)
    {
        auto where_condition = parent->GetFilterStructure();
        auto where_conditions = expr_processor.generate_and_list(where_condition);

        conditions.insert(conditions.cend(), where_conditions.cbegin(), where_conditions.cend());
    }

    for (const auto& condition : conditions)
    {
         std::dynamic_pointer_cast<CommonBiaodashi>(condition)->ObtainReferenceTableInfo();
    }

    std::string defaultDatabaseName;
    auto query = std::dynamic_pointer_cast<SelectStructure>(arg_input->GetMyQuery());
    if (query)
    {
        auto defaultSchema = query->GetDefaultSchemaAgent();
        if (defaultSchema)
        {
            defaultDatabaseName.assign(defaultSchema->schema->GetName());
        }
    }

    auto schema = schema::SchemaManager::GetInstance()->GetSchema();

    std::sort(conditions.begin(), conditions.end(), [&](const BiaodashiPointer& left, const BiaodashiPointer& right){
        auto left_expression = std::dynamic_pointer_cast<CommonBiaodashi>(left);
        auto right_expression = std::dynamic_pointer_cast<CommonBiaodashi>(right);

        auto left_involved_tables = left_expression->GetInvolvedTableList();
        auto right_involved_tables = right_expression->GetInvolvedTableList();

        int64_t left_differ = 0;
        if (left_involved_tables.size() == 2)
        {
            left_differ = getDifferRowCount(left_involved_tables[0], left_involved_tables[1], schema, defaultDatabaseName);
        }

        int64_t right_differ = 0;
        if (right_involved_tables.size() == 2)
        {
            right_differ = getDifferRowCount(right_involved_tables[0], right_involved_tables[1], schema, defaultDatabaseName);
        }
        return left_differ > right_differ;
    });

    std::vector<BiaodashiPointer> other_conditions;

    std::vector<std::pair<std::vector<std::string>, SQLTreeNodePointer>> node_groups;

    std::vector<SQLTreeNodePointer> join_nodes_unmached;
    join_nodes_unmached.assign(join_sub_nodes.cbegin(), join_sub_nodes.cend());

    for (const auto& condition : conditions)
    {
        auto con = std::dynamic_pointer_cast<CommonBiaodashi>(condition);
        if (con->GetType() != BiaodashiType::Bijiao)
        {
            other_conditions.emplace_back(condition);
            continue;
        }

        auto left = std::dynamic_pointer_cast<CommonBiaodashi>(con->GetChildByIndex(0));
        auto right = std::dynamic_pointer_cast<CommonBiaodashi>(con->GetChildByIndex(1));
        // left->ObtainReferenceTableInfo();
        // right->ObtainReferenceTableInfo();
        auto left_involved_tables = left->GetInvolvedTableList();
        auto right_involved_tables = right->GetInvolvedTableList();

        if (left_involved_tables.size() != 1 || right_involved_tables.size() != 1)
        {
            other_conditions.emplace_back(condition);
            continue;
        }

        auto left_table = left_involved_tables[0];
        auto right_table = right_involved_tables[0];

        if (*left_table == *right_table)
        {
            other_conditions.emplace_back(condition);
            continue;
        }

        SQLTreeNodePointer left_node = nullptr, right_node = nullptr;

        std::string table_name_found;

        join_sub_nodes.clear();
        join_sub_nodes.assign(join_nodes_unmached.cbegin(), join_nodes_unmached.cend());
        join_nodes_unmached.clear();

        for (const auto& node : join_sub_nodes)
        {
            auto node_tables = node->GetInvolvedTableList();
            if (node_tables.empty())
            {
                continue;
            }

            if (*(node_tables[0]) == *left_table)
            {
                table_name_found.assign(left_table->IsSubquery() ? left_table->GetMyOutputName() : left_table->GetID());
                left_node = node;
            }
            else if (*(node_tables[0]) == *right_table)
            {
                table_name_found.assign(right_table->IsSubquery() ? right_table->GetMyOutputName() : right_table->GetID());
                right_node = node;
            }
            else
            {
                join_nodes_unmached.emplace_back(node);
            }
        }

        if (!left_node && !right_node)
        {
            other_conditions.emplace_back(condition);
            continue;
        }

        std::string table_name_need_to_search;
        if (!left_node)
        {
            table_name_need_to_search.assign(left_table->IsSubquery() ? left_table->GetMyOutputName() : left_table->GetID());
        }
        else if (!right_node)
        {
            table_name_need_to_search.assign(right_table->IsSubquery() ? right_table->GetMyOutputName() : right_table->GetID());
        }

        int exist_index = -1;
        if (!table_name_need_to_search.empty())
        {
            for (size_t i = 0; i < node_groups.size(); i++)
            {
                auto& exist_node = node_groups[i];
                if (std::find(exist_node.first.cbegin(), exist_node.first.cend(), table_name_need_to_search) != exist_node.first.cend())
                {
                    if (!left_node)
                    {
                        left_node = exist_node.second;
                    }
                    else
                    {
                        right_node = exist_node.second;
                    }

                    exist_index = i;
                    break;
                }
            }
        }

        if (left_node && right_node)
        {
            auto new_node = SQLTreeNode::makeTreeNode_BinaryJoin(arg_input->GetMyQuery(), JoinType::InnerJoin, condition);
            SQLTreeNode::AddTreeNodeChild( new_node, left_node );
            SQLTreeNode::AddTreeNodeChild( new_node, right_node );

            if (exist_index != -1)
            {
                auto& exist_node = node_groups[exist_index];
                exist_node.first.emplace_back(table_name_found);
                exist_node.second = new_node;
            }
            else
            {
                std::vector<std::string> table_names;
                table_names.emplace_back(left_table->IsSubquery() ? left_table->GetMyOutputName() : left_table->GetID());
                table_names.emplace_back(right_table->IsSubquery() ? right_table->GetMyOutputName() : right_table->GetID());
                node_groups.emplace_back(std::make_pair(table_names, new_node));
            }
        }
        else
        {
            other_conditions.emplace_back(condition);
        }
    }

    for (const auto& condition : conditions)
    {
         std::dynamic_pointer_cast<CommonBiaodashi>(condition)->ClearInvolvedTableList();
    }

    auto& new_root_node = node_groups[0].second;
    if (node_groups.size() > 1)
    {
        for (size_t i = 1; i < node_groups.size(); i++)
        {
            auto new_join_node = SQLTreeNode::makeTreeNode_BinaryJoin(arg_input->GetMyQuery(), JoinType::InnerJoin,
                                                                      std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, true));
            SQLTreeNode::AddTreeNodeChild( new_join_node, new_root_node );
            SQLTreeNode::AddTreeNodeChild( new_join_node, node_groups[ i ].second );

            new_root_node = new_join_node;
        }
    }

    for (const auto& node : join_nodes_unmached)
    {
        auto new_join_node = SQLTreeNode::makeTreeNode_BinaryJoin(arg_input->GetMyQuery(), JoinType::InnerJoin,
                                                                    std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, true));
        SQLTreeNode::AddTreeNodeChild( new_join_node, new_root_node );
        SQLTreeNode::AddTreeNodeChild( new_join_node, node );

        new_root_node = new_join_node;
    }

    if (!other_conditions.empty())
    {
        auto filter_condition = expr_processor.make_biaodashi_from_and_list(other_conditions);
        if (parent && parent->GetType() == SQLTreeNodeType::Filter_NODE)
        {
            parent->CompletelyResetAChild(arg_input, new_root_node);
            new_root_node->SetParent( parent );
            parent->SetFilterStructure(filter_condition);
            // parent->SetTheChild(new_root_node);
            new_root_node->SetParent(parent);
            return;
        }
        else
        {
            auto new_filter_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Filter_NODE);
            new_filter_node->SetFilterStructure(filter_condition);
            SQLTreeNode::SetTreeNodeChild( new_filter_node, new_root_node );
            new_root_node = new_filter_node;
        }
    }

    parent->CompletelyResetAChild(arg_input, new_root_node);
    new_root_node->SetParent( parent );
}

} // namespace aries
