#pragma once

#include <memory>
#include <map>
#include <vector>

#include "DBEntry.h"
#include "TableEntry.h"

namespace aries
{
namespace schema
{

class DatabaseEntry: public DBEntry {

private:
    string defaultCharsetName;
    string defaultCollationName;
    std::map<std::string, std::shared_ptr<TableEntry>> tables;
    string create_string; // used by show create database command

public:
    DatabaseEntry(string name, string argDefaultCharset = "utf8", string argDefaultCollation = "utf8_general_ci");
    ~DatabaseEntry();

    void SetCreateString(const string& s) {
        create_string = s;
    }
    string GetCreateString() {
        return create_string;
    }



    void PutTable(std::shared_ptr<TableEntry> table);
    std::string GetColumnLocationString(std::string arg_table_name,
                                        std::string arg_column_name);
    std::string GetColumnLocationString_ByIndex(std::string arg_table_name,
                                                int arg_column_index);

    std::shared_ptr<TableEntry> GetTableByName(std::string name);
    std::vector<string> GetNameListOfTables();
    int GetTablesCount();
    const std::map<std::string, std::shared_ptr<TableEntry>>& GetTables() {
        return tables;
    }
    void ClearTable()
    {
        tables.clear();
    }

    int GetTableIDByName(std::string name);

    void RemoveTable(std::shared_ptr<TableEntry> table);
};
using DatabaseEntrySPtr = std::shared_ptr<DatabaseEntry>;

} // namespace schema
} // namespace aries
