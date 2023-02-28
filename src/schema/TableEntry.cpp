#include <iostream>
#include <boost/filesystem.hpp>

#include <frontend/SQLExecutor.h>
#include "utils/string_util.h"

#include "TableEntry.h"
#include "server/Configuration.h"
#include "schema/SchemaManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "Compression/dict/AriesDict.h"
#include "Compression/dict/AriesDictManager.h"
#include "AriesEngine/transaction/AriesXLogManager.h"

namespace aries
{
namespace schema
{
const string ENGINE_RATEUP = "rateup";
TableEntry::TableEntry(const DatabaseEntrySPtr& db_entry_ptr_arg,
                       const int64_t& id,
                       const string& argName,
                       const string& argEngine,
                       const string& argCollation)
:DBEntry(argName),
 db_entry_ptr(db_entry_ptr_arg),
 table_id(id),
 primary_column(nullptr),
 row_count(0)
{
    collation = argCollation;
    if (!argName.empty()) {
        SetupDataFilePath();
    }
}

TableEntry::TableEntry(const DatabaseEntrySPtr& db_entry_ptr_arg,
                       const string& argName,
                       const string& argEngine,
                       const string& argCollation): TableEntry( db_entry_ptr_arg, -1, argName, argEngine, argCollation ) {
}

TableEntry::~TableEntry() {
    #ifdef DEBUG_MEM
    LOG(INFO) << "TableEntry::~TableEntry()" << std::endl;
    #endif
    columns.clear();
}
void TableEntry::SetupDataFilePath() {
    string dbName = db_entry_ptr->GetName();
    aries_utils::to_lower(dbName);
    string dataDir = Configuartion::GetInstance().GetDataDirectory( dbName );

    string lowerTableName = GetName();
    aries_utils::to_lower(lowerTableName);
    data_file_path = dataDir + "/" + lowerTableName;
}

void TableEntry::AddColumn(const ColumnEntryPtr& column) {
    string lowerTableName = GetName();
    aries_utils::to_lower(lowerTableName);
    std::string path_str = data_file_path + "/" + lowerTableName + std::to_string(column->GetColumnIndex());

    SetColumnLocationString(column->GetName(), path_str);
    SetColumnLocationString_ByIndex(column->GetColumnIndex(), path_str);

    if (column->IsPrimary()) {
        primary_column = column;
    }
    // Schema::LoadColumns() 保证column 按index顺序添加
    columns.push_back(column);
}

std::shared_ptr<ColumnEntry> TableEntry::GetColumnById(size_t id) {
    assert( id > 0 );
    auto index = id - 1;
    return columns[index];
}

std::shared_ptr<ColumnEntry> TableEntry::GetColumnByName(std::string name) {
    for (size_t i = 0; i < columns.size(); i++)
    {
        if (name == columns[i]->GetName()) {
            return columns[i];
        }
    }

    return nullptr;
}

size_t TableEntry::GetColumnsCount() {
    return columns.size();
}
const vector<std::shared_ptr<ColumnEntry>> &TableEntry::GetColumns() {
    return columns;
}

void TableEntry::SetColumnLocationString(std::string arg_column_name, std::string arg_path) {
    column_name_path_map[GetName() + std::string(".") + arg_column_name] = arg_path;
}

std::string TableEntry::GetColumnLocationString(std::string arg_column_name) {
    return column_name_path_map[GetName() + std::string(".") + arg_column_name];
}

void TableEntry::SetColumnLocationString_ByIndex(int arg_column_index,
                                                     std::string arg_path) {
    column_index_path_map[GetName() + std::to_string(arg_column_index)] = arg_path;
}
std::string TableEntry::GetColumnLocationString_ByIndex(int arg_column_index) {
    return column_index_path_map[GetName() + std::to_string(arg_column_index)];
}

uint64_t TableEntry::GetRowCount() const {
    return row_count;
}

void TableEntry::SetRowCount(uint64_t count) {
    row_count = count;
}

void TableEntry::SetId( int64_t id ) {
    table_id = id;
}

int64_t TableEntry::GetId() const {
    return table_id;
}
size_t TableEntry::GetRowStoreSize()
{
    size_t size = 0;
    for ( auto& col : columns )
    {
        size += col->GetItemStoreSize();
    }
    return size;
}

const std::vector< std::string >& TableEntry::GetPrimaryKey() const
{
    return primary_key_columns;
}

const std::vector< ForeignKeySPtr >& TableEntry::GetUniqueKeys() const
{
    return unique_keys;
}
const std::vector< ForeignKeySPtr >& TableEntry::GetAllUniqueKeys() const
{
    return all_unique_keys;
}
const std::vector< ForeignKeySPtr >& TableEntry::GetForeignKeys() const
{
    return foreign_keys;
}

std::string TableEntry::GetPrimaryKeyName() const
{
    return primary_key_name;
}

void TableEntry::AddConstraint( const TableConstraintSPtr& constraint )
{
    constraints[ constraint->name ] = constraint;

    if ( constraint->type == TableConstraintType::PrimaryKey )
    {
        all_unique_keys.emplace_back( constraint );

        primary_key_name = constraint->name;

        for ( const auto& key : constraint->keys )
        {
            primary_key_columns.emplace_back( key );
        }
    }
    else if ( constraint->type == TableConstraintType::UniqueKey )
    {
        all_unique_keys.emplace_back( constraint );
        unique_keys.emplace_back( constraint );
    }
    else if ( constraint->type == TableConstraintType::ForeignKey )
    {
        foreign_keys.emplace_back( constraint );
    }
}

const std::map< std::string, TableConstraintSPtr >& TableEntry::GetConstraints() const
{
    return constraints;
}

TableConstraintSPtr TableEntry::GetConstraint( const string& name ) const
{
    const auto& it = constraints.find( name );
    if ( constraints.end() != it )
        return it->second;
    else
        return nullptr;
}

void TableEntry::AddConstraintKey( const std::string& keyName, const std::string& constraintName )
{
    auto column = GetColumnByName( keyName );
    if ( !column )
    {
        return;
    }

    constraints[ constraintName ]->keys.emplace_back( keyName );
    if ( constraints[ constraintName ]->type == TableConstraintType::PrimaryKey )
    {
        primary_key_columns.emplace_back( keyName );
        column->is_primary = true;
    }
    else if ( constraints[ constraintName ]->type == TableConstraintType::UniqueKey )
    {
        column->is_unique = true;
    }
    else if ( constraints[ constraintName ]->type == TableConstraintType::ForeignKey )
    {
        column->is_foreign_key = true;
    }
}

void TableEntry::AddConstraintReferencedKey( const std::string& keyName, const std::string& constraintName )
{
    constraints[ constraintName ]->referencedKeys.emplace_back( keyName );
}

void TableEntry::SetConstraintReferencedTableName( const std::string& schemaName, const std::string& tableName, const std::string& constraintName )
{
    constraints[ constraintName ]->referencedSchema = schemaName;
    constraints[ constraintName ]->referencedTable = tableName;
}

void TableEntry::AddReferencing( const std::string& keyName,
                                 const std::string& referencingKeyName,
                                 const std::string& referencingTableName,
                                 const std::string& referencingTableSchema,
                                 const std::string& constraintName,
                                 const int64_t& referencingTableId )
{
    auto key_pair = std::make_pair( referencingTableId, constraintName );
    if ( ! referencings[ key_pair ] )
    {
         referencings[ key_pair ] = std::make_shared< ReferencingContraint >();
    }

    auto& referencing = referencings[ key_pair ];
    referencing->keys.emplace_back( keyName );
    referencing->referencingKeys.emplace_back( referencingKeyName );
    referencing->referencingTable = referencingTableName;
    referencing->referencingSchema = referencingTableSchema;
    referencing->name = constraintName;

    referencings_array.clear();
}

const std::vector< ReferencingContraintSPtr >& TableEntry::GetReferencings()
{
    if ( referencings_array.empty() && !referencings.empty() )
    {
        for ( const auto& it : referencings )
        {
            referencings_array.emplace_back( it.second );
        }
    }

    return referencings_array;
}
void TableEntry::OnDrop( const string& dbName )
{
    bool hasDict = false;
    for ( auto& colEntry : columns )
    {
        auto dict = colEntry->GetDict();
        if ( dict )
        {
            AriesDictManager::GetInstance().DecDictRefCount( dict->GetId() );
            hasDict = true;
        }
    }

    if ( hasDict )
    {
        string sql = "DELETE FROM dict_column_usage WHERE TABLE_SCHEMA = '" + dbName +
                     "' AND TABLE_NAME = '" + GetName() + "';";
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA );
        if ( !result->IsSuccess() )
        {
            ARIES_EXCEPTION_SIMPLE( result->GetErrorCode(), result->GetErrorMessage() );
        }
    }

    auto myName = GetName();
    AriesXLogManager::GetInstance().AddTruncateEvent( GetId() );
    aries_engine::AriesMvccTableManager::GetInstance().removeMvccTable( dbName, myName );
    aries_engine::AriesMvccTableManager::GetInstance().deleteCache( dbName, myName );
    aries_engine::AriesInitialTableManager::GetInstance().removeTable( dbName, myName );

    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    schema->RemoveTable( dbName, myName );
    auto tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbName, myName );
    boost::filesystem::remove_all( tableDataDir );
}

} // namespace schema
} // namespace aries
