#include <glog/logging.h>

#include "ViewManager.h"
#include "SQLParserPortal.h"
#include "schema/SchemaManager.h"
#include "SchemaBuilder.h"
#include "QueryBuilder.h"
#include "SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

namespace aries
{

ViewManager ViewManager::instance;

ViewManager::ViewManager()
{

}

ViewManager::~ViewManager()
{

}

ViewManager& ViewManager::GetInstance()
{
    return instance;
}

void ViewManager::Init()
{
    auto result = SQLExecutor::GetInstance()->ExecuteSQL("select table_schema, table_name, view_definition from information_schema.views", "information_schema");
    if (!result->IsSuccess())
    {
        LOG(ERROR) << "cannot load views from information_schema";
        return;
    }

    if (result->GetResults().size() != 1)
    {
        LOG(ERROR) << "unexpected result in information_schema.views, result size: " << result->GetResults().size();
        return;
    }

    auto result_table = (aries_engine::AriesMemTable*)(result->GetResults()[0].get());
    auto result_data = result_table->GetContent();
    if (!result_data)
    {
        LOG(INFO) << "views table is empty";
        return;
    }

    LOG(INFO) << "views count is: " << result_data->GetRowCount();

    aries::SQLParserPortal sql_parser;

    auto database_name_column = result_data->GetColumnBuffer(1);
    auto view_name_column = result_data->GetColumnBuffer(2);
    auto view_definition_column = result_data->GetColumnBuffer(3);
    for (int i = 0; i < result_data->GetRowCount(); i++)
    {
        auto database_name = database_name_column->GetString(i);
        auto view_name = view_name_column->GetString(i);
        auto view_definition = view_definition_column->GetString(i);

        std::vector< AriesSQLStatementPointer > statements;
        try
        {
            statements = sql_parser.ParseSQLString4Statements(view_definition);
        }
        catch ( ... )
        {
            LOG(ERROR) << "invalid view definition string: " << view_definition;
            continue;
        }
        
        if (statements.size() != 1)
        {
            LOG(ERROR) << "invalid view definition string: " << view_definition;
            continue;
        }

        if (!statements[0]->IsCommand())
        {
            LOG(ERROR) << "invalid view definition string: " << view_definition << "(not a command)";
            continue;
        }

        auto command = (aries::CommandStructure*)(statements[0]->GetCommand().get());

        try {
            auto view_node = BuildFromCommand(command, database_name);
            if (view_node)
            {
                RegisterViewNode(view_node, view_name, database_name);
                DLOG(INFO) << "HERE LOAD view: " << database_name << "." << view_name << "(" << view_definition << ")";
            }
            else
            {
                LOG(ERROR) << "HERE LOAD view: " << database_name << "." << view_name << "(" << view_definition << ") failed";
            }
        } catch (...) {
            LOG(ERROR) << "cannot load view by: " << view_definition;
        }
    }
}


SQLTreeNodePointer ViewManager::BuildFromCommand(CommandStructure* command, const std::string& database_name)
{
    if (command->GetCommandType() != aries::CommandType::CreateView)
    {
        return nullptr;
    }
    
    auto view_node = ViewManager::BuildFromQuery(command->GetQuery(), database_name);

    if (!view_node)
    {
        return nullptr;
    }

    auto query = std::dynamic_pointer_cast<SelectStructure>(view_node->GetMyQuery());
    auto relation = query->GetRelationStructure();

    auto column_defs = command->GetColumns();
    if (column_defs) {
        if (relation->GetColumnCount() != column_defs->size()) {
            ARIES_EXCEPTION(ER_VIEW_WRONG_LIST, "View's SELECT and view's field list have different column counts");
        }

        auto new_relation = std::make_shared<RelationStructure>();

        for (size_t i = 0; i < relation->GetColumnCount(); i++) {
            auto column = relation->GetColumn(i);
            auto column_def = (ColumnDescription*)((*column_defs)[i].get());
            auto new_column = std::make_shared<ColumnStructure>(column_def->column_name, column->GetValueType(), column->GetLength(), column->IsNullable(), column->GetIsPrimary());
            new_column->SetEncodeType( column->GetEncodeType() );
            new_column->SetEncodedIndexType( column->GetEncodedIndexType() );
            new_relation->AddColumn(new_column);
        }

        query->SetRelationStructure(new_relation);
    }

    return view_node;
}

bool ViewManager::RegisterViewNode(SQLTreeNodePointer& node, const std::string& view_name, const std::string& database_name)
{
    Mutex_lock lock(&mutex);
    std::string view_node_name = database_name + "." + view_name;

    auto rel = std::make_shared<BasicRel>(false, view_name, nullptr, nullptr);
    rel->SetDb(database_name);
    node->SetBasicRel(rel);
    if (nodes.find(view_node_name) == nodes.end())
    {
        nodes[view_node_name] = node;
        queries[ view_node_name ] = node->GetMyQuery();
        rels[ view_node_name ] = rel;
        return true;
    }
    else
    {
        LOG(ERROR) << "cannot register view node because node with this name: " << view_name << " exists";
        return false;
    }
    
}

bool ViewManager::UnregisterViewNode(const std::string& view_name, const std::string& database_name)
{
    Mutex_lock lock(&mutex);
    std::string view_node_name = database_name + "." + view_name;
    auto it = nodes.find(view_node_name);
    if (it != nodes.end())
    {
        nodes.erase(it);
        queries.erase( view_node_name );
        rels.erase( view_node_name );
        return true;
    }
    else
    {
        LOG(ERROR) << "cannot unregister view node because node with this name: " << view_name << " does not exist";
        return false;
    }
}

SQLTreeNodePointer ViewManager::GetViewNode(const std::string& view_name, const std::string& database_name)
{
    Mutex_lock lock(&mutex);
    std::string view_node_name = database_name + "." + view_name;
    auto it = nodes.find(view_node_name);
    if (it != nodes.end())
    {
        return it->second;
    }
    else
    {
        LOG(ERROR) << "cannot find view node with this name: " << view_name;
        return nullptr;
    }
}

std::vector<SQLTreeNodePointer> ViewManager::GetViewNodes(const std::string& database_name) {
    Mutex_lock lock(&mutex);
    
    const char* target = database_name.data();
    auto target_size = database_name.size();

    std::vector<SQLTreeNodePointer> filtered_nodes;
    for (const auto& it : nodes)
    {
        if (database_name.size() > (it.first.size() - 2)) {
            continue;
        }

        if (memcmp(target, it.first.data(), target_size)) {
            continue;
        }

        if (it.first[target_size] != '.') {
            continue;
        }

        filtered_nodes.emplace_back(it.second);
    }

    return filtered_nodes;
}

SQLTreeNodePointer ViewManager::BuildFromQueryString(const std::string& query_string, const std::string& database_name)
{
    SQLParserPortal parser;
    auto statements = parser.ParseSQLString4Statements(query_string);

    ARIES_ASSERT(statements.size() == 1, "Create View should have one query statement");
    auto query = statements[0]->GetQuery();
    statements[0]->SetQuery(nullptr);

    ARIES_ASSERT(query, "Create View should have one query");

    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(database_name);
    ARIES_ASSERT(database, "database with name: " + database_name + " does not exist");

    auto database_schema = SchemaBuilder::BuildFromDatabase(database.get());

    auto query_builder = std::make_shared<QueryBuilder>();
    query_builder->InitDatabaseSchema(database_schema);
    query_builder->BuildQuery(query);

    auto select_query = (SelectStructure*)(query.get());
    auto node = select_query->GetQueryPlanTree();
    node->SetMyQuery(query);
    return node;
}

SQLTreeNodePointer ViewManager::BuildFromQuery(const AbstractQueryPointer& query, const std::string& database_name) 
{
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(database_name);
    ARIES_ASSERT(database, "database with name: " + database_name + " does not exist");

    auto database_schema = SchemaBuilder::BuildFromDatabase(database.get());

    auto query_builder = std::make_shared<QueryBuilder>();
    query_builder->InitDatabaseSchema(database_schema);
    query_builder->BuildQuery(query);

    auto select_query = (SelectStructure*)(query.get());
    auto node = select_query->GetQueryPlanTree();
    node->SetMyQuery(query);
    return node;
}

} // namespace aries
