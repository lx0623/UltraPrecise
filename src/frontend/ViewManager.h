#pragma once

#include <string>
#include <map>
#include <vector>

#include "SQLTreeNode.h"
#include "utils/mutex_lock.h"
#include "frontend/CommandStructure.h"
#include "AbstractQuery.h"

namespace aries
{

class ViewManager
{

private:
    static ViewManager instance;

    std::map<std::string, SQLTreeNodePointer> nodes;
    std::map< std::string, AbstractQueryPointer > queries;
    std::map< std::string, BasicRelPointer > rels;

    pthread_mutex_t mutex;

public:
    ViewManager();
    ~ViewManager();

    static ViewManager& GetInstance();

    static SQLTreeNodePointer BuildFromQueryString(const std::string& query_string, const std::string& database_name);

    static SQLTreeNodePointer BuildFromQuery(const AbstractQueryPointer& query, const std::string& database_name);

    static SQLTreeNodePointer BuildFromCommand(CommandStructure* command, const std::string& database_name);

    void Init();

    bool RegisterViewNode(SQLTreeNodePointer& node, const std::string& view_name, const std::string& database_name);
    bool UnregisterViewNode(const std::string& view_name, const std::string& database_name);

    SQLTreeNodePointer GetViewNode(const std::string& view_name, const std::string& database_name);

    std::vector<SQLTreeNodePointer> GetViewNodes(const std::string& database_name);
};

} // namespace aries