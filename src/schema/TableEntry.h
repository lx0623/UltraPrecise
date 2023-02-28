#pragma once

#include <vector>
#include <memory>
#include <unordered_map>

#include "DBEntry.h"
#include "ColumnEntry.h"
#include "DatabaseEntry.h"
#include "TablePartition.h"

namespace aries
{

namespace schema
{
    extern const char* DEFAULT_CHARSET_NAME;
    extern const char* DEFAULT_UTF8_COLLATION;
    extern const string ENGINE_RATEUP;

enum class TableConstraintType : int16_t
{
    PrimaryKey,
    ForeignKey,
    UniqueKey
};

struct TableConstraint
{
    std::string name;
    TableConstraintType type;
    std::vector< std::string > keys;
    std::vector< std::string > referencedKeys;
    std::string referencedTable;
    std::string referencedSchema; // database
};

struct ReferencingContraint
{
    std::string name;
    std::vector< std::string > keys; // 主键
    std::vector< std::string > referencingKeys; // 外键
    std::string referencingTable;
    std::string referencingSchema; // database
};

using ForeignKeySPtr = std::shared_ptr< TableConstraint >;
using TableConstraintSPtr = std::shared_ptr< TableConstraint >;
using ReferencingContraintSPtr = std::shared_ptr< ReferencingContraint >;

class TableEntry: public DBEntry {

private:
    DatabaseEntrySPtr db_entry_ptr;
    int64_t table_id;
    string engine;
    string collation;
    string data_file_path;
    string create_table_string; // used by show create table command
    vector<std::shared_ptr<ColumnEntry>> columns;
    std::unordered_map<std::string, std::string> column_name_path_map;
    std::unordered_map<std::string, std::string> column_index_path_map;

    // TODO: remove primary_column
    shared_ptr<ColumnEntry> primary_column;

    std::string primary_key_name;
    std::vector< std::string > primary_key_columns;
    std::vector< TableConstraintSPtr > all_unique_keys; // including primary key and UNIQUE [KEY]
    std::vector< TableConstraintSPtr > unique_keys; // UNIQUE [KEY] ONLY
    std::vector< ForeignKeySPtr > foreign_keys;
    std::map< std::string, TableConstraintSPtr > constraints;

    // { { TableId : constraint name } : ReferencingContraintSPtr }
    std::map< std::pair< int64_t, std::string >, ReferencingContraintSPtr > referencings;
    std::vector< ReferencingContraintSPtr > referencings_array;

    uint64_t row_count = 0;

    //  RANGE, LIST, HASH, LINEAR HASH, KEY, or LINEAR KEY
    std::string m_partitionMethod;
    std::string m_partitionExprStr;
    int m_partitionColumnIndex;
    uint32_t m_partitionCount = 0;
    std::vector< TablePartitionSPtr > m_partitions;

public:
    TableEntry() = delete;
    TableEntry(const DatabaseEntrySPtr& db_entry_ptr_arg,
               const int64_t& id,
               const string& name,
               const string& engine,
               const string& collation = DEFAULT_UTF8_COLLATION);

    TableEntry(const DatabaseEntrySPtr& db_entry_ptr_arg,
               const string& name,
               const string& engine,
               const string& collation = DEFAULT_UTF8_COLLATION);
    ~TableEntry();

    void SetPartitionTypeDefInfo( const string &method,
                                  int partitionColIndex,
                                  const string &partitionExprStr )
    {
        m_partitionMethod = method;
        m_partitionColumnIndex = partitionColIndex;
        m_partitionExprStr = partitionExprStr;
    }
    void AddPartition( TablePartitionSPtr & part )
    {
        m_partitions.emplace_back( part );
        ++m_partitionCount;
    }
    const std::vector< TablePartitionSPtr > &GetPartitions()
    {
        return m_partitions;
    }
    string GetPartitionExprStr() const
    {
        return m_partitionExprStr;
    }
    int GetPartitionColumnIndex() const
    {
        return m_partitionColumnIndex;
    }
    bool IsPartitioned() const
    {
        return !m_partitionMethod.empty();
    }
    string GetPartitionMethod() const
    {
        return m_partitionMethod;
    }
    uint32_t GetPartitionCount() const
    {
        return m_partitionCount;
    }

    void SetCreateString(const string& s) {
        create_table_string = s;
    }
    string GetCreateString() {
        return create_table_string;
    }

    string GetDataDir() const
    {
        return data_file_path;
    }
    void AddColumn(const shared_ptr<ColumnEntry>& column);

    const shared_ptr<ColumnEntry> GetPrimaryKeyColumn() const { return primary_column; }

    string GetEngine() { return engine; }
    string GetCollation() { return collation; }

    std::shared_ptr<ColumnEntry> GetColumnById(size_t id);
    std::shared_ptr<ColumnEntry> GetColumnByName(std::string name);
    vector<std::string>& GetNameOfColumns();
    const vector<std::shared_ptr<ColumnEntry>>& GetColumns();
    size_t GetColumnsCount();

    void SetColumnLocationString(std::string arg_column_name,
                                 std::string arg_path);
    std::string GetColumnLocationString(std::string arg_column_name);
    void SetColumnLocationString_ByIndex(int arg_column_index, std::string arg_path);
    std::string GetColumnLocationString_ByIndex(int arg_column_index);

    uint64_t GetRowCount() const;
    void SetRowCount(uint64_t count);

    void SetId( int64_t id );
    int64_t GetId() const;

    size_t GetRowStoreSize();

    const std::vector< std::string >& GetPrimaryKey() const;
    const std::vector< ForeignKeySPtr >& GetUniqueKeys() const;
    const std::vector< ForeignKeySPtr >& GetAllUniqueKeys() const;
    const std::vector< ForeignKeySPtr >& GetForeignKeys() const;
    const std::vector< ReferencingContraintSPtr >& GetReferencings();

    std::string GetPrimaryKeyName() const;

    void AddConstraint( const TableConstraintSPtr& constraint );
    const std::map< std::string, TableConstraintSPtr >& GetConstraints() const;
    TableConstraintSPtr GetConstraint( const string& name ) const;
    void AddConstraintKey( const std::string& keyName, const std::string& constraintName );
    void AddConstraintReferencedKey( const std::string& keyName, const std::string& constraintName );
    void SetConstraintReferencedTableName( const std::string& schemaName, const std::string& tableName, const std::string& constraintName );

    void AddReferencing( const std::string& keyName,
                         const std::string& referencingKeyName,
                         const std::string& referencingTableName,
                         const std::string& referencingTableSchema,
                         const std::string& constraintName,
                         const int64_t& referencingTableId );
    void OnDrop( const string& dbName );

private:
    void SetupDataFilePath();

};
using TableEntrySPtr = shared_ptr<TableEntry>;

} // namespace schema
} // namespace schema
