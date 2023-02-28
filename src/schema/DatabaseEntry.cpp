#include <algorithm>
#include <frontend/SQLExecutor.h>
#include "AriesAssert.h"
#include "DatabaseEntry.h"
#include "utils/string_util.h"

using namespace std;

namespace aries
{
namespace schema
{

DatabaseEntry::DatabaseEntry(string name, string argDefaultCharset, string argDefaultCollation)
        :DBEntry(name),
         defaultCharsetName(argDefaultCharset),
         defaultCollationName(argDefaultCollation) {
}

DatabaseEntry::~DatabaseEntry() {
    tables.clear();
}

std::string DatabaseEntry::GetColumnLocationString(std::string arg_table_name, std::string arg_column_name) {
    auto tableEntry = GetTableByName(arg_table_name);
    ARIES_ASSERT(nullptr != tableEntry, "table not found");
    return tableEntry->GetColumnLocationString(arg_column_name);
}

std::string DatabaseEntry::GetColumnLocationString_ByIndex(std::string arg_table_name, int arg_column_index) {
    auto tableEntry = GetTableByName(arg_table_name);
    ARIES_ASSERT(nullptr != tableEntry, "table not found");
    return tableEntry->GetColumnLocationString_ByIndex(arg_column_index);
}
void DatabaseEntry::PutTable(shared_ptr<TableEntry> table) {
    tables[table->GetName()] = table;
}

shared_ptr<TableEntry> DatabaseEntry::GetTableByName(string name) {
    string lowerName = name;
    auto it = tables.find(aries_utils::to_lower(lowerName));
    if (tables.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

int DatabaseEntry::GetTablesCount() {
    return tables.size();
}

std::vector<string> DatabaseEntry::GetNameListOfTables() {
    vector<string> keys;
    for (auto it = tables.begin(); it != tables.end(); it++)
    {
        keys.push_back(it->first);
    }

    return keys;
}

void DatabaseEntry::RemoveTable(std::shared_ptr<TableEntry> table) {
    tables.erase( table->GetName() );
}

int DatabaseEntry::GetTableIDByName(std::string name) {
    int id = 0;
    for (auto &item : tables) {
        if (item.first == name)
            return id;
        id ++;
    }

    return 0;
}


} // namespace schema
} // namespace aries
