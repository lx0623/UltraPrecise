#pragma once

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <mutex>

#include "ColumnEntry.h"
#include "TableEntry.h"
#include "DatabaseEntry.h"

#ifndef DISABLE_NAMESPACE
#define NAMESPACE_OPEN(value) namespace value {
#define NAMESPACE_CLOSE(value) }
#else
#define NAMESPACE_OPEN(value)
#define NAMESPACE_CLOSE(value)
#endif

NAMESPACE_OPEN(aries)

using TableId = int64_t;

NAMESPACE_OPEN(schema)
extern const string RATEUP_EMPTY_STRING;
extern const string RATEUP_NULL_VALUE_STRING;
extern const string INFORMATION_SCHEMA;
extern const string PERFORMANCE_SCHEMA;
extern const string SCHEMATA;
extern const char* DEFAULT_CHARSET_NAME;
extern const char* DEFAULT_UTF8_COLLATION;
bool IsSysDb(const string& name);
ColumnType ToColumnType(const string& col_type_str);
int64_t get_auto_increment_table_id();
bool fix_column_length(const std::string& db_root_path);
class Schema {
private:
    bool isInit = false;
    std::mutex schema_lock;
    std::map<std::string, std::shared_ptr<DatabaseEntry>> databases;
    DatabaseEntrySPtr informationDbPtr;
    std::vector<std::string> keys_to_remove;

public:

    const std::string static DefaultDatabaseName;
    const static std::string KeyPrefix;
    static const std::string kSchemaColumnFamilyName;
    enum WriteMode {
        Override,
        Merge,
        Abort
    };

public:
    Schema();
    ~Schema();

    static std::string GetKeyForDatabase(std::string name);
    static std::string GetKeyForTable(std::string name, std::string database_name);
    static std::string GetKeyForColumn(std::string name, std::string database_name, std::string table_name);

    bool LoadBaseSchema();
    bool LoadSchema();

    void SetInit( bool b) { isInit = b; }
    bool IsInit() { return isInit; }
    bool Init();

    void InsertDbSchema(const string& dbName);
    void DeleteDbSchema(const string& dbName);
    void InsertTableSchema(const string& dbName, const TableEntrySPtr& tableEntry);

    void
    InsertViewSchema(const string& dbName,
                     const string& tableName,
                     const std::string& create_string,
                     const std::vector<ColumnEntryPtr>& columns);

    void InsertCharsetsRows();
    void InsertCollationRows();
    void InsertCollationCharsetApplicabilityRows();
    void InsertEngineRows();

    void InsertProcess(uint64_t pid,
                       const string& user,
                       const string& host,
                       const string& db,
                       const string& cmd,
                       int time,
                       const string& state,
                       const string& info);
    void DeleteProcess(uint64_t pid);

    void AddDatabase(std::shared_ptr<DatabaseEntry> database);
    void AddDatabase(std::string name, std::shared_ptr<DatabaseEntry> database);
    std::shared_ptr<DatabaseEntry> GetDatabaseByName(std::string name);
    const std::map<std::string, std::shared_ptr<DatabaseEntry>>& GetDatabases();

    TableId GetTableId( const string& dbName, const string& tableName );

    void Dump(const string& dbName = "");
    void Dump(const std::shared_ptr<DatabaseEntry>& dbEntry);

    void RemoveDatabase(std::shared_ptr<DatabaseEntry> database);
    void RemoveTable( const std::string& dbName, const string& tableName );
    void RemoveView( const std::string& dbName, const string& viewName );

    std::pair< DatabaseEntrySPtr, TableEntrySPtr > GetDatabaseAndTableById( TableId tableId );

private:
    bool InitInformationSchema();
    void InitDatabaseMysql();
    bool InitDefaultUser();
    bool LoadSchemata();
    bool LoadTables();
    bool LoadColumns();

    bool LoadConstraints();
    bool LoadColumnDictUsage();
    bool LoadKeys();
    bool LoadTablePartitions();
};

NAMESPACE_CLOSE(schema) // namespace schema
NAMESPACE_CLOSE(aries) // namespace aries
