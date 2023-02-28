//
// Created by 胡胜刚 on 2019-06-13.
//


#include <sys/stat.h>
#include <glog/logging.h>
#include <AriesEngineWrapper/AriesExprBridge.h>
#include <server/mysql/include/mysys_err.h>
#include <sys/wait.h>
#include "server/mysql/include/mysqld_error.h"
#include "server/mysql/include/derror.h"
#include "server/mysql/include/mysqld.h"
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/set_var.h"
#include "server/mysql/include/sys_vars.h"
#include "utils/string_util.h"
#include "SQLExecutor.h"
#include "ExecutorPortal.h"
#include "SchemaBuilder.h"
#include "DatabaseSchema.h"
#include "SelectStructure.h"
#include "SQLParserPortal.h"
#include "QueryBuilder.h"
#include "AriesEngineByMaterialization.h"
#include "CommandStructure.h"
#include "CommandExecutor.h"
#include "ViewManager.h"

#include "schema/SchemaManager.h"
#include "schema/DatabaseEntry.h"
#include "schema/TableEntry.h"
#include "optimizer/QueryOptimizer.h"

#include "AriesException.h"
#include "AriesAssert.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngine/AriesDataCache.h"
#include "ShowStatusVariableStructure.h"
#include "CudaAcc/AriesSqlOperator.h"

#include "AriesEngine/transaction/AriesTransManager.h"
#include "AriesEngine/transaction/AriesVacuum.h"
#include "utils/datatypes.h"
#include "CpuTimer.h"

#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"

using namespace aries::schema;

using aries_engine::AriesMvccTableManager;
using aries_engine::AriesInitialTableManager;

extern bool g_isInitialize;
extern bool IsServerInitDone();
void sql_kill(THD *thd, my_thread_id id, bool only_kill_query);
bool shutdown(THD *thd, enum mysql_enum_shutdown_level level, enum enum_server_command command);
int importCsvFile( aries_engine::AriesTransactionPtr& tx,
                   const DatabaseEntrySPtr& dbEntry, const TableEntrySPtr& tableEntry,
                   const std::string& csvFilePath,
                   uint64_t& skipLines,
                   const std::string& fieldSeperator,
                   const bool escapeGiven,
                   const std::string& escapeChar,
                   bool optEnclosed,
                   const std::string& encloseChar,
                   const std::string& lineSeperator,
                   const std::string& lineStart );

void BuildQuery(SelectStructurePointer& query, const string& dbName, bool needRowIdColumn = false);
void BuildQuery( SelectStructurePointer& query, const string& dbName, bool needRowIdColumn ) {
    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);

    /*init*/
    DatabaseSchemaPointer schema_p = nullptr;
    if (database) {
        schema_p = SchemaBuilder::BuildFromDatabase(database.get(), needRowIdColumn);
    }

    /*build query*/
    QueryBuilderPointer query_builder = std::make_shared<aries::QueryBuilder>();
    query_builder->InitDatabaseSchema(schema_p);
    query_builder->BuildQuery(query);
}

bool NeedEndTransaction(AriesSQLStatementPointer statement, THD* thd)
{
    bool ret;
    ret = ( ( !statement->IsTxStmt() && !thd->m_explicitTx && (bool)get_sys_var_value<my_bool>(find_sys_var((char*)"autocommit"), OPT_SESSION) ) 
                 || statement->IsCommand() || current_thd->peer_port == USHRT_MAX );
    return ret;
}
namespace aries {

SQLResultPtr showCharSet(const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr);
SQLResultPtr showCollation(const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr);
SQLResultPtr showColumns(const ShowColumnsStructurePtr& showColumnsStructurePointer, const string& dbName);
SQLResultPtr showCreateDb(const string& dbName);
SQLResultPtr showCreateTable(const string& dbName, const string& tableName);
SQLResultPtr showDatabases();
SQLResultPtr showEngines();
SQLResultPtr showEngineStatus(const string& name);
SQLResultPtr showEngineMutex(const string& name);
SQLResultPtr showErrors(const LimitStructurePointer& limit);
SQLResultPtr showEvents(const std::string& dbName, const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr);
SQLResultPtr showFunctionStatus(const string& like, const BiaodashiPointer& whereExpr);
SQLResultPtr showIndex(bool extended, const string& db, const string& table, const BiaodashiPointer& whereExpr);
SQLResultPtr showMasterStatus();
SQLResultPtr showOpenTables(const string& dbName, const string& wild, const BiaodashiPointer& whereExpr);
SQLResultPtr showPrivileges();
SQLResultPtr showProcessList(bool full);
SQLResultPtr showSlaveHosts();
SQLResultPtr showStatus(bool global);
SQLResultPtr showPlugins();
SQLResultPtr showTables(const std::string& dbName, bool full);
SQLResultPtr showTableStatus(const string& dbName, const string& wild, const BiaodashiPointer& whereExpr);
SQLResultPtr showTriggers(const std::string& dbName, const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr);
SQLResultPtr showWarnings(const LimitStructurePointer& limit);
SQLResultPtr showSysVar(bool global, const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr);

SQLExecutor *SQLExecutor::instance = nullptr;

SQLExecutor::SQLExecutor()
{
    session = std::make_shared<AriesServerSession>();
}

void SQLExecutor::Init() {
    instance = new SQLExecutor();
}

SQLExecutor *SQLExecutor::GetInstance() {
    return instance;
}

// SelectStructurePointer SQLExecutor::parseSQL(std::string sql) {
//     return (std::make_unique<SQLParserPortal>())->ParseSQLString(sql);
// }

// SelectStructurePointer SQLExecutor::parseSQLFromFile(std::string file_path) {
//     return (std::make_unique<SQLParserPortal>())->ParseSQLFile(file_path);
// }

#define SET_CONSTANT_VALUE( addr, value, nullable )         \
do {                                                        \
    if ( nullable ) {                                       \
        *( ( int8_t* )( addr ) ) = 1;                       \
        memcpy( ( int8_t* )( addr ) + 1, &( value ), sizeof( value ) );\
    } else {                                                \
        memcpy( ( int8_t* )( addr ), &( value ), sizeof( value ) );    \
    }                                                       \
} while ( 0 )

#define SET_CONSTANT_VALUE_BY_ADDRESS( addr, ptr, size, nullable )              \
do {                                                                            \
    if ( nullable ) {                                                           \
        *( ( int8_t* )( addr ) ) = 1;                                           \
    }                                                                           \
    if ( size > 0 )                                                             \
    {                                                                           \
        memcpy( ( int8_t* )( addr + ( nullable ? 1 : 0 ) ), ( ptr ), size );  \
    }                                                                           \
} while ( 0 )

AbstractMemTablePointer executeQueryWithoutFrom(SelectStructurePointer query) {
    AriesMemTableSPtr table = std::make_shared< AriesMemTable >();
    auto dataBlockUPtr = std::make_unique<AriesTableBlock>();

    auto selectPart = query->GetSelectPart() ;
    int itemCount = selectPart->GetSelectItemCount();
    for (int i = 0; i < itemCount; i++) {
        string columnName = selectPart->GetName(i);
        dataBlockUPtr->AddColumnName(i + 1, columnName);

        BiaodashiPointer abp = selectPart->GetSelectExpr(i);
        AriesExprBridge bridge;
        AriesCommonExprUPtr ariesCommonExprUPtr = bridge.Bridge(abp);
        CommonBiaodashi *rawpointer = (CommonBiaodashi *) abp.get();

        AriesDataBufferSPtr bufferSPtr;
        if (BiaodashiType::Null == rawpointer->GetType()) {
            AriesColumnType columnType;
            AriesDataType dataType;
            dataType.Length = 1;
            dataType.ValueType = AriesValueType::UNKNOWN;

            columnType.DataType = dataType;
            columnType.HasNull = true;
            bufferSPtr = make_shared<AriesDataBuffer>(columnType, 1);
        } else {
            bufferSPtr = make_shared<AriesDataBuffer>(ariesCommonExprUPtr->GetValueType(), 1, true );
            auto valueType = ariesCommonExprUPtr->GetValueType().DataType.ValueType;
            auto isNullable = ariesCommonExprUPtr->GetValueType().isNullable();

            if ( ariesCommonExprUPtr->GetType() == AriesExprType::NULL_VALUE )
            {
                if ( !bufferSPtr->GetDataType().isNullable() )
                {
                    auto newType = ariesCommonExprUPtr->GetValueType();
                    newType.HasNull = true;
                    bufferSPtr = make_shared<AriesDataBuffer>( newType, 1 );
                }
                *( bufferSPtr->GetItemDataAt( 0 ) ) = 0;
            }
            else
            {
                switch (valueType) {
                    case aries::AriesValueType::UNKNOWN:
                        break;
                    case aries::AriesValueType::INT8: {
                        int8_t value = boost::get<int8_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::INT16: {
                        int16_t value = boost::get<int16_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::INT32: {
                        int32_t value = boost::get<int32_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::INT64: {
                        int64_t value = boost::get<int64_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::UINT8: {
                        uint8_t value = boost::get<uint8_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::UINT16: {
                        uint16_t value = boost::get<uint16_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::UINT32: {
                        uint32_t value = boost::get<uint32_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::UINT64: {
                        uint64_t value = boost::get<uint64_t>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::DECIMAL: {
                        auto decimal_content = boost::get<aries_acc::Decimal>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE( bufferSPtr->GetItemDataAt(0), decimal_content, isNullable );
                        break;
                    }
                    case aries::AriesValueType::FLOAT: {
                        float value = boost::get<float>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::DOUBLE: {
                        double value = boost::get<double>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::CHAR: {
                        string value = boost::get<string>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE_BY_ADDRESS( bufferSPtr->GetItemDataAt( 0 ), value.data(), value.size(), isNullable );
                        break;
                    }
                    case aries::AriesValueType::BOOL: {
                        bool value = boost::get<bool>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::DATE: {
                        auto value = boost::get<aries_acc::AriesDate>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::TIME: {
                        auto value = boost::get<aries_acc::AriesTime>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                    case aries::AriesValueType::DATETIME: {
                        auto value = boost::get<aries_acc::AriesDatetime>(ariesCommonExprUPtr->GetContent());
                        SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        break;
                    }
                        // TOTO: support timestamp and year in AriesExprContent
                        // case aries::AriesValueType::TIMESTAMP: {
                        //     auto value = boost::get<aries_acc::AriesTimestamp>(ariesCommonExprUPtr->GetContent());
                        //     SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        //     break;
                        // }
                        // case aries::AriesValueType::YEAR: {
                        //     auto value = boost::get<aries_acc::AriesYear>(ariesCommonExprUPtr->GetContent());
                        //     SET_CONSTANT_VALUE(bufferSPtr->GetItemDataAt(0), value, isNullable);
                        //     break;
                        // }
                    default: {
                        // TODO: handle other datatypes
                        string msg = "expression value type ";
                        msg.append(std::to_string((int32_t) valueType));
                        ARIES_EXCEPTION(ER_NOT_SUPPORTED_YET, msg.data());
                        break;
                    }
                }
            }
        }

        auto column = std::make_shared<AriesColumn>();
        column->AddDataBuffer(bufferSPtr);
        dataBlockUPtr->AddColumn(i + 1, column);
    }

    table->SetContent(std::move(dataBlockUPtr));
    return table;
}
AbstractMemTablePointer SQLExecutor::executeQuery(const AriesTransactionPtr& tx, SelectStructurePointer& query, const string& defaultDbName, bool buildQuery) {

    if (buildQuery) {
        BuildQuery(query, defaultDbName);
    }

    if (!query->IsSetQuery() && nullptr == query->GetFromPart()) {
        if ( query->GetWhereCondition() )
        {
            auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( query->GetWhereCondition() );
            if ( !condition->IsTrueConstant() )
            {
                AriesMemTableSPtr table = std::make_shared< AriesMemTable >();
                auto dataBlockUPtr = std::make_unique<AriesTableBlock>();
                table->SetContent( std::move( dataBlockUPtr ) );
                return table;
            }
        }
        return executeQueryWithoutFrom(query);
    }

    SQLTreeNodePointer query_plan_tree = query->GetQueryPlanTree();
    if ( !query_plan_tree->IsOptimized() )
    {
        LOG(INFO) << "\nBuilding Query Plan---------OK!\n";

        /*query optimization*/
        QueryOptimizerPointer query_optimizer = QueryOptimizer::GetQueryOptimizer();
        query_optimizer->OptimizeTree(query_plan_tree);
        LOG(INFO) << "Optimizing Query Plan---------OK!\n";
    }

    LOG(INFO) << query_plan_tree->ToString(0) << "\n";
    auto executorPortal = make_shared< ExecutorPortal >( std::make_shared<AriesEngineByMaterialization>() );
    AbstractMemTablePointer tablePointer = executorPortal->ExecuteQuery(tx, query_plan_tree, defaultDbName);
    AriesMemTableSPtr table = std::dynamic_pointer_cast<AriesMemTable>(tablePointer);

    if (table) {
        auto selectPart = query->GetSelectPart() ;
        int itemCount = selectPart->GetSelectItemCount();

        auto tableBlock = table->GetContent();

        if (tableBlock) {
            // if (tableBlock->GetRowCount() > 0) {
            for (int i = 0; i < itemCount; i++) {
                auto expr = (CommonBiaodashi*) (selectPart->GetSelectExpr(i).get());
                if (!expr->IsVisibleInResult()) {
                    continue;
                }

                string columnName = selectPart->GetName(i);
                tableBlock->AddColumnName(i + 1, columnName);
                // dataBlocks->GetBlocks()[0]->AddColumnName(columnName);
            }
            // }
            table->SetContent(std::move(tableBlock));
        }
    }
    return table;
}

void BiaodashiToColumnIds( const TableEntrySPtr& tableEntry,
                           const vector< BiaodashiPointer >& columnExprs,
                           vector< int >& columnIds )
{
    map< string, int > columnMap;
    for ( auto& expr : columnExprs )
    {
        auto ident = boost::get< SQLIdentPtr >( ( ( CommonBiaodashi* )expr.get() )->GetContent() );
        if ( columnMap.end() != columnMap.find( ident->id ) )
            ARIES_EXCEPTION( ER_FIELD_SPECIFIED_TWICE, ident->id.data() );
        if( tableEntry->GetColumnByName( ident->id ) == nullptr )
            ARIES_EXCEPTION( ER_WRONG_FIELD_SPEC, ident->id.data() );
        columnIds.push_back( tableEntry->GetColumnByName( ident->id )->GetColumnIndex() + 1 );
    }
}
SQLTreeNodePointer BuildAndOptimizeQueryForDML( SelectStructurePointer query,
                                                const string& defaultDbName )
{
    BuildQuery(query, defaultDbName, true);

    LOG(INFO) << "\nBuilding Query Plan---------OK!\n";
    SQLTreeNodePointer query_plan_tree = query->GetQueryPlanTree();

    /*query optimization*/
    QueryOptimizerPointer query_optimizer = QueryOptimizer::GetQueryOptimizer();
    query_optimizer->OptimizeTree(query_plan_tree);
    LOG(INFO) << "\nOptimizing Query Plan---------OK!\n";

    LOG(INFO) << query_plan_tree->ToString(0) << "\n";
    return query_plan_tree;

}
AbstractMemTablePointer SQLExecutor::executeInsert( const aries_engine::AriesTransactionPtr& tx,
                                                    InsertStructurePtr& insertStructure,
                                                    const string& defaultDbName )
{
    auto insertDbName = insertStructure->GetDbName();
    if ( insertDbName.empty() )
        insertDbName = defaultDbName;
    auto insertTableName = insertStructure->GetTableName();

    auto insertColumnExprs = insertStructure->GetInsertColumns();
    auto insertValueExprs = insertStructure->GetInsertColumnValues();
    auto optUpdateColumnExprs = insertStructure->GetOptUpdateColumns();
    auto optUpdateValueExprs = insertStructure->GetOptUpdateValues();
    auto selectStructure = insertStructure->GetSelectStructure();

    auto dbEntry =
       SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( insertDbName );
    auto tableEntry = dbEntry->GetTableByName( insertTableName );
    vector< int > insertColumnIds;
    if ( insertColumnExprs.empty() )
    {
        // insert into t values( ... )
        insertColumnIds.resize( tableEntry->GetColumnsCount() );
        std::iota( insertColumnIds.begin(), insertColumnIds.end(), 1 );
    }
    else
    {
        BiaodashiToColumnIds( tableEntry, insertColumnExprs, insertColumnIds );
    }
    SQLTreeNodePointer planTree = nullptr;
    if ( selectStructure )
    {
        auto query = std::dynamic_pointer_cast<SelectStructure>( selectStructure );
        planTree = BuildAndOptimizeQueryForDML( query, defaultDbName );

        auto selectItemCount = query->GetSelectPart()->GetSelectItemCount();

        if ( selectItemCount != insertColumnIds.size() )
        {
            ARIES_EXCEPTION( ER_WRONG_VALUE_COUNT_ON_ROW, 1 );
        }
        for ( size_t i = 0; i < insertColumnIds.size(); ++i )
        {
            auto colEntry = tableEntry->GetColumnById( insertColumnIds[ i ] );
            if ( EncodeType::DICT == colEntry->encode_type )
            {
                auto selectExpr = query->GetSelectPart()->GetSelectExpr( i );
                CommonBiaodashi *selectCommonExpr= (CommonBiaodashi *) selectExpr.get();
                auto exprType = selectCommonExpr->GetType();
                if ( BiaodashiType::SQLFunc == exprType )
                {
                    auto function = boost::get<SQLFunctionPointer>( selectCommonExpr->GetContent() );
                    if ( aries_acc::AriesSqlFunctionType::DICT_INDEX == function->GetType() )
                    {
                        auto selectColumnExpr = ( CommonBiaodashi * )( selectCommonExpr->GetChildByIndex( 0 ).get() );
                        auto selectColumnShell = boost::get<ColumnShellPointer>( selectColumnExpr->GetContent() );
                        auto selectColumnId = selectColumnShell->GetLocationInTable() + 1;
                        auto selectTable = selectColumnShell->GetTable();
                        auto selectExprDictDbName = selectTable->GetDb();
                        auto selectExprDictTableName = selectTable->GetID();
                        auto selectExprDictColumnName = selectColumnShell->GetColumnName();
                        auto selectExprDbEntry =
                            SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( selectExprDictDbName );
                        auto selectExprTableEntry = selectExprDbEntry->GetTableByName( selectExprDictTableName );
                        auto selectExprColumnEntry = selectExprTableEntry->GetColumnById( selectColumnId );

                        if ( colEntry->GetDictId() != selectExprColumnEntry->GetDictId() )
                        {
                            ThrowNotSupportedException( "insert DICT_INDEX values into column that does not share the same dict" );
                        }
                    }
                }
            }
        }
    }
    else
    {
        DatabaseSchemaPointer schema_p = SchemaBuilder::BuildFromDatabase(dbEntry.get());
        SchemaAgentPointer schema_agent = std::make_shared<SchemaAgent>();
        schema_agent->SetDatabaseSchema(schema_p);

        auto dummy_query = std::make_shared< SelectStructure >();
        dummy_query->SetDefaultSchemaAgent( schema_agent );

        QueryContextPointer query_context = std::make_shared<QueryContext>(QueryContextType::TheTopQuery,
                                                                        0,
                                                                        dummy_query, //query
                                                                        nullptr, //parent
                                                                        nullptr //possible expr
                                                                        );

        ExprContextPointer exprContext = std::make_shared< ExprContext >( ExprContextType::VirtualExpr, nullptr, query_context, nullptr, 0 );
        int rowIdx = 1;
        for( auto& rowValues : insertValueExprs )
        {
            if( rowValues->size() != insertColumnIds.size() )
                ARIES_EXCEPTION( ER_WRONG_VALUE_COUNT_ON_ROW, rowIdx );
            ++rowIdx;
            //lichi: seems we only need check expr to set correct value_type etc...
            for( auto& expr : *rowValues )
                dynamic_pointer_cast< CommonBiaodashi >( expr )->CheckExpr( exprContext, true );
        }
    }
    auto executorPortal = make_shared< ExecutorPortal >( std::make_shared<AriesEngineByMaterialization>() );
    AbstractMemTablePointer tablePointer =
        executorPortal->ExecuteInsert( tx,
                                       insertDbName,
                                       insertTableName,
                                       insertColumnIds,
                                       insertValueExprs,
                                       optUpdateColumnExprs,
                                       optUpdateValueExprs,
                                       planTree );
    return tablePointer;
}
AbstractMemTablePointer SQLExecutor::executeUpdate( const aries_engine::AriesTransactionPtr& tx,
                                                    UpdateStructurePtr& updateStructure,
                                                    const string& defaultDbName )
{
    auto query = std::dynamic_pointer_cast<SelectStructure>( updateStructure->GetSelectStructure() );
    auto executorPortal = make_shared< ExecutorPortal >( std::make_shared<AriesEngineByMaterialization>( true ) );
    query->SetSchemaAgentNeedRowIdColumn( updateStructure->GetTargetDbName(), true );
    auto planTree = BuildAndOptimizeQueryForDML( query, defaultDbName );
    AbstractMemTablePointer tablePointer = executorPortal->ExecuteUpdate( tx,
                                                                          planTree,
                                                                          updateStructure->GetTargetDbName(),
                                                                          defaultDbName );
    return tablePointer;

}
void CheckDeleteTables( const vector< BasicRelPointer >& tables,
                        const string& defaultDbName,
                        string& deleteDbName, // out param
                        string& deleteTableName ) // out param
{
    map< TableMapKey, int > tableMap;
    for ( auto& rel : tables )
    {
        deleteDbName = rel->GetDb();
        if ( deleteDbName.empty() )
            deleteDbName = defaultDbName;
        deleteTableName = rel->GetID();
        auto key = TableMapKey( deleteDbName, deleteTableName );
        auto it = tableMap.find( key );
        if ( tableMap.end() == it )
            tableMap[ key ] = 1;

        if ( tableMap.size() > 1 )
            ThrowNotSupportedException( "delete from multi tables" );
    }
}
AbstractMemTablePointer SQLExecutor::executeDelete( const aries_engine::AriesTransactionPtr& tx,
                                                    DeleteStructurePtr& deleteStructure,
                                                    const string& defaultDbName )
{
    string deleteDbName, deleteTableName;
    CheckDeleteTables( deleteStructure->GetTargetTables(),
                       defaultDbName,
                       deleteDbName,
                       deleteTableName );
    if ( deleteDbName.empty() )
       ARIES_EXCEPTION(ER_NO_DB_ERROR);

    auto query = std::dynamic_pointer_cast<SelectStructure>( deleteStructure->GetSelectStructure() );
    query->SetSchemaAgentNeedRowIdColumn( deleteDbName, true );
    auto planTree = BuildAndOptimizeQueryForDML( query, defaultDbName );
    auto executorPortal = make_shared< ExecutorPortal >( std::make_shared<AriesEngineByMaterialization>() );

    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( deleteDbName );
    if( !dbEntry )
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, deleteDbName.data() );
    auto tableEntry = dbEntry->GetTableByName( deleteTableName );
    if( !tableEntry )
        ARIES_EXCEPTION( ER_NO_SUCH_TABLE, deleteDbName.c_str(), deleteTableName.c_str() );
    if ( tableEntry->IsPartitioned() )
    {
        ThrowNotSupportedException( "delete from partitioned table" );
    }
    auto tableId = tableEntry->GetId();
    tx->SetUpdateTableId( tableId );

    AbstractMemTablePointer tablePointer =
        executorPortal->ExecuteDelete( tx,
                                       deleteDbName,
                                       deleteTableName,
                                       planTree,
                                       defaultDbName );

    if ( tablePointer )
    {
        tx->AddModifiedTable( tableId );
    }

    return tablePointer;
}

AbstractMemTablePointer SQLExecutor::executeCommand(AbstractCommandPointer command, const string& dbName) {
    CommandExecutor cmd_executor;
    cmd_executor.ExecuteCommand(command.get(), session.get(), dbName);

    // TODO: return a valid AbstractMemTablePointer
    return nullptr;
}

/**
 START TRANSACTION
[transaction_characteristic [, transaction_characteristic] ...]
transaction_characteristic: {
WITH CONSISTENT SNAPSHOT
| READ WRITE
| READ ONLY
}
BEGIN [WORK]
COMMIT [WORK] [AND [NO] CHAIN] [[NO] RELEASE]
ROLLBACK [WORK] [AND [NO] CHAIN] [[NO] RELEASE]
 */
void endTx( THD* thd, AriesSQLStatementPointer stmt, TransactionStatus txStatus )
{
    if ( thd->m_tx )
    {
        TxId txId = thd->m_tx->GetTxId();
        bool explicitTx = thd->m_explicitTx;
        AriesTransManager::GetInstance().EndTransaction( thd->m_tx, txStatus );
        thd->reset_tx();

        if ( stmt )
        {
#ifndef NDEBUG
            LOG( INFO ) << "end transaction explicitly";
#endif
            if ( stmt->IsTxStmt() )
            {
                auto txStructurePtr = stmt->GetTxStructurePtr();
                string msg( "end explicit transaction " );
                msg.append( to_string( txId ) ).append( ": " );
                if ( TransactionStatus::COMMITTED == txStatus )
                    msg.append( "commit" );
                else
                    msg.append( "abort" );
                if ( TX_START != txStructurePtr->txCmd )
                {
                    msg.append( ", do end" );
                    if ( TVL_YES == stmt->GetTxStructurePtr()->txChain )
                    {
                        try
                        {
                            thd->m_tx = AriesTransManager::GetInstance().NewTransaction();
                            msg.append(", chain");
                        }
                        catch (const AriesException& e)
                        {
                            LOG(INFO) << "chain transaction failed: " << e.errMsg;
                        }
                    }
                    if ( TVL_YES == stmt->GetTxStructurePtr()->txRelease )
                    {
                        msg.append(", release");
                        thd->killed = THD::KILL_CONNECTION;
                    }
                }
                LOG( INFO ) << msg;
            }
            else
            {
                if ( explicitTx )
                    DLOG( INFO ) << "end explicit transaction " << txId << ": "
                                 << ( TransactionStatus::COMMITTED == txStatus ? "commit" : "abort" );
                else
                    DLOG( INFO ) << "end implicit transaction " << txId << ": "
                                 << ( TransactionStatus::COMMITTED == txStatus ? "commit" : "abort" );
            }
            stmt->SetTransaction( nullptr );
        }
        else
        {
            LOG( INFO ) << "end transaction implicitly";
            if ( explicitTx )
                LOG( INFO ) << "end explicit transaction " << txId << ": "
                            << ( TransactionStatus::COMMITTED == txStatus ? "commit" : "abort" );
            else
                LOG( INFO ) << "end implicit transaction " << txId << ": "
                            << ( TransactionStatus::COMMITTED == txStatus ? "commit" : "abort" );
        }
    }
}

// based on rules in MySQL 5.7 Reference Manual 13.3.3 Statements That Cause an Implicit Commit
// minimal support
void checkTxImplicitCommit( THD* thd, AriesSQLStatementPointer stmt )
{
    bool commit = false;
    if ( current_thd->peer_port == USHRT_MAX )
    {
        commit = true;
    }
    if ( stmt->IsCommand() )
    {
        auto command = stmt->GetCommand();
        switch ( command->GetCommandType() )
        {
        case CommandType::CreateDatabase:
        case CommandType::DropDatabase:
        case CommandType::CreateTable:
        case CommandType::DropTable:
        case CommandType::CreateView:
        case CommandType::DropView:
        case CommandType::CreateUser:
        case CommandType::DropUser:
            commit = true;
            break;

        default:
            break;
        }
    }
    else if ( stmt->IsSetStatement() )
    {
        auto setStructurePtrs = stmt->GetSetStructurePtrs();
        auto setStructurePtr = (*setStructurePtrs)[0];
        if ( SET_CMD::SET_PASSWORD == setStructurePtr->m_setCmd )
        {
            commit = true;
        }
    }
    else if ( stmt->IsTxStmt() )
    {
        auto txStructurePtr = stmt->GetTxStructurePtr();
        // Beginning a transaction causes any pending transaction to be committed.
        if ( TX_START == txStructurePtr->txCmd )
            commit = true;
    }
    if ( commit )
    {
        endTx( thd, nullptr, TransactionStatus::COMMITTED );
    }
}

static void startNewTx( THD* thd, AriesSQLStatementPointer stmt )
{
    // if last statement end with CHAIN, a new transaction is already created
    if ( !thd->m_tx )
    {
        if ( stmt->IsTxStmt() )
        {
            auto txStructurePtr = stmt->GetTxStructurePtr();
            if ( TX_START == txStructurePtr->txCmd )
            {
                thd->m_tx = AriesTransManager::GetInstance().NewTransaction();
            }
        }
        else
        {
            // autocommit is default to true for us,
            // so each SQL statement forms a single transaction on its own
            if ( stmt->IsCommand() || current_thd->peer_port == USHRT_MAX )
            {
                thd->m_tx = AriesTransManager::GetInstance().NewTransaction( true );
                thd->m_explicitTx = true;
            }
            else
            {
                thd->m_tx = AriesTransManager::GetInstance().NewTransaction();
            }

        }
    }
    stmt->SetTransaction( thd->m_tx );
}

bool handleQueryResult( THD* thd, const AbstractMemTablePointer& amtp , SendDataType send_type, ulong fetch_row_count, ulong offset_row_count)
{
    auto table = ( AriesMemTable * )amtp.get();
    auto tableContent = table->GetContent();
    ARIES_ASSERT( tableContent, "null tableContent!!!" );
    ulong tupleNum = tableContent->GetRowCount();
    if(send_type == SendDataType::SEND_ROWDATA_ONLY && offset_row_count>=tupleNum) {
        thd->server_status|= SERVER_STATUS_LAST_ROW_SENT;
        my_eof(thd);
        return 0;
    }
    // tableContent->Dump( true );

    int columnCount = tableContent->GetColumnCount();

    thd->get_protocol_classic()->start_row();
    vector<Send_field*> fields;
    // fake column name info
    // TODO: need real column names
    if (0 == columnCount) {
        LOG(INFO) << "===============================================";
        LOG(INFO) << "Query result tuple count " << tupleNum << ", column count " << columnCount;
        Send_field field;
        field.db_name = "DB NAME";
        field.table_name = field.org_table_name = "TABLE NAME";
        field.col_name = "column" + std::to_string(1);
        field.org_col_name = field.col_name;

        field.length = 57;
        field.flags = 0;
        field.decimals = 31;
        field.type = MYSQL_TYPE_STRING;
        fields.insert(fields.end(), &field);
        thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF);
        table->SetContent( std::move( tableContent ) );
        return 0;
    }

    std::vector<AriesDataBufferSPtr> columns;
    for (int col = 1; col < columnCount + 1; col++)
    {
        AriesDataBufferSPtr column = tableContent->GetColumnBuffer(col);
        columns.push_back( column );

        Send_field *field = new Send_field();
        field->db_name = "DB NAME";
        field->table_name = field->org_table_name = "TABLE NAME";
        field->col_name = tableContent->GetColumnName(col);
        field->org_col_name = field->col_name;

        /**
         https://dev.mysql.com/doc/internals/en/com-query-response.html
         decimals (1) -- max shown decimal digits
             0x00 for integers and static strings
             0x1f for dynamic strings, double, float
             0x00 to 0x51 for decimals
        */
        field->decimals = 0;

        /**
         * flags, see MACRO definitions in mysql_com.h
         * .... .... .... ...0 = Field can't be NULL
         * .... .... .... ..0. = Field is part of a primary key
         * .... .... .... .0.. = Unique key
         * .... .... .... 0... = Multiple key
         * .... .... ...0 .... = Blob
         * .... .... ..0. .... = Field is unsigned
         * .... .... .0.. .... = Zero fill
         * .... .... 0... .... = Field is binary
         * .... ...0 .... .... = Enum
         * .... ..0. .... .... = Auto increment
         * .... .0.. .... .... = Field is a timestamp
         * .... 0... .... .... = Set
         */
        field->flags = 0x0000;

        if (!column->isNullableColumn()) {
            field->flags |= NOT_NULL_FLAG;
        }

        AriesValueType valueType = column->GetDataType().DataType.ValueType;
        switch (valueType) {
            case AriesValueType::UNKNOWN:
                field->type = MYSQL_TYPE_NULL;
                field->length = 4;
                break;
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            case AriesValueType::UINT8:
                field->type = MYSQL_TYPE_TINY;
                field->length = 4;
                if (AriesValueType::UINT8 == valueType) {
                    field->flags |= UNSIGNED_FLAG;
                }
                break;
            case AriesValueType::INT16:
            case AriesValueType::UINT16:
                field->type = MYSQL_TYPE_SHORT;
                field->length = 6;
                if (AriesValueType::UINT16 == valueType) {
                    field->flags |= UNSIGNED_FLAG;
                }
                break;
            case AriesValueType::INT32:
            case AriesValueType::UINT32:
                field->type = MYSQL_TYPE_LONG;
                field->length = 11;
                if (AriesValueType::UINT32 == valueType) {
                    field->flags |= UNSIGNED_FLAG;
                }
                break;
            case AriesValueType::INT64:
            case AriesValueType::UINT64:
                field->type = MYSQL_TYPE_LONGLONG;
                field->length = 20;
                if (AriesValueType::UINT64 == valueType) {
                    field->flags |= UNSIGNED_FLAG;
                }
                break;
            case AriesValueType::DECIMAL:
            case AriesValueType::COMPACT_DECIMAL:
                field->type = MYSQL_TYPE_NEWDECIMAL;
                field->length = 11; // for type: decimal
                field->decimals = 31;
                break;
            case AriesValueType::FLOAT:
                field->type = MYSQL_TYPE_FLOAT;
                field->length = 12;
                field->decimals = 0x1f;
                break;
            case AriesValueType::DOUBLE:
                field->type = MYSQL_TYPE_DOUBLE;
                field->length = 22;
                field->decimals = 0x1f;
                break;
            case AriesValueType::CHAR:
                field->type = MYSQL_TYPE_STRING;
                // field->length = column->GetDataType().GetDataTypeSize() * 3; // for char(256): 256 * 3
                field->length = column->GetDataType().GetDataTypeSize(); // for char(256): 256 * 3
                break;
            case AriesValueType::DATE:
                field->type = MYSQL_TYPE_DATE;
                field->length = 10;
                field->flags |= BINARY_FLAG;
                break;
            case AriesValueType::TIME:
                field->type = MYSQL_TYPE_TIME;
                field->length = 10;
                field->flags |= BINARY_FLAG;
                break;
            case AriesValueType::DATETIME:
                field->type = MYSQL_TYPE_DATETIME;
                field->length = 19;
                field->flags |= BINARY_FLAG;
                break;
            case AriesValueType::TIMESTAMP:
                field->type = MYSQL_TYPE_TIMESTAMP;
                field->length = 19;
                field->flags |= (NOT_NULL_FLAG | BINARY_FLAG | TIMESTAMP_FLAG);
                break;
            case AriesValueType::YEAR:
                field->type = MYSQL_TYPE_YEAR;
                field->length = 4;
                field->flags |= (UNSIGNED_FLAG | ZEROFILL_FLAG);
                break;
            default:
                LOG(WARNING) << "not supported data type: " << (int32_t) column->GetDataType().DataType.ValueType;
                field->type = MYSQL_TYPE_NULL;
                break;
        }
        // printf("Column type: %d, name: %s\n", (int)column->GetDataType().BaseType, tableContent->GetColumnName(col).c_str());
        fields.insert(fields.end(), field);
    }

    if(send_type == SendDataType::SEND_METADATA_AND_ROWDATA || send_type == SendDataType::SEND_METADATA_ONLY){
        bool error = thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF);
        for (Send_field *field : fields) {
            delete field;
        }
        if (error)
        {
            table->SetContent( std::move( tableContent ) );
            return error;
        }
    }

    if (send_type == SendDataType::SEND_METADATA_AND_ROWDATA || send_type == SendDataType::SEND_ROWDATA_ONLY){
        ulong send_count = tupleNum;
        if (send_type == SendDataType::SEND_ROWDATA_ONLY){
            send_count = std::min(offset_row_count+fetch_row_count, tupleNum);
        }
        for (auto tid = offset_row_count; tid < send_count; tid++)
        {
            thd->get_protocol_classic()->start_row();
            for ( auto &column : columns )
            {
                // LOG(INFO) << "[column type: " << (int)column->GetDataType().DataType.ValueType << "] ";
                auto dataType = column->GetDataType();
                switch ( dataType.DataType.ValueType )
                {
                    case AriesValueType::UNKNOWN:
                        // LOG(INFO) << "NULL";
                        thd->get_protocol_classic()->store_null();
                        break;
                    case AriesValueType::INT8:
                    case AriesValueType::BOOL: {
                        int8_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isInt8DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableInt8(tid).value;
                        } else {
                            columnData = column->GetInt8(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store_tiny(columnData);
                        break;
                    }
                    case AriesValueType::UINT8: {
                        uint8_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isUint8DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableUint8(tid).value;
                        } else {
                            columnData = column->GetUint8(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store_tiny(columnData);
                        break;
                    }
                    case AriesValueType::INT16: {
                        int16_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isInt16DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableInt16(tid).value;
                        } else {
                            columnData = column->GetInt16(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store_short(columnData);
                        break;
                    }
                    case AriesValueType::UINT16: {
                        uint16_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isUint16DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableUint16(tid).value;
                        } else {
                            columnData = column->GetUint16(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store_short(columnData);
                        break;
                    }
                    case AriesValueType::INT32: {
                        int32_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isInt32DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableInt32(tid).value;
                        } else {
                            columnData = column->GetInt32(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData);
                        break;
                    }
                    case AriesValueType::UINT32: {
                        uint32_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isUint32DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableUint32(tid).value;
                        } else {
                            columnData = column->GetUint32(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData);
                        break;
                    }
                    case AriesValueType::INT64: {
                        int64_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isInt64DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableInt64(tid).value;
                        } else {
                            columnData = column->GetInt64(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store((longlong) columnData);
                        break;
                    }
                    case AriesValueType::UINT64: {
                        uint64_t columnData;
                        if (column->isNullableColumn()) {
                            if (column->isUint64DataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableUint64(tid).value;
                        } else {
                            columnData = column->GetUint64(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store((ulonglong) columnData);
                        break;
                    }
                    case AriesValueType::FLOAT: {
                        float columnData;
                        if (column->isNullableColumn()) {
                            if (column->isFloatDataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableFloat(tid).value;
                        } else {
                            columnData = column->GetFloat(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData, 31);
                        break;
                    }
                    case AriesValueType::DOUBLE: {
                        double columnData;
                        if (column->isNullableColumn()) {
                            if (column->isDoubleDataNull(tid)) {
                                thd->get_protocol_classic()->store_null();
                                break;
                            }
                            columnData = column->GetNullableDouble(tid).value;
                        } else {
                            columnData = column->GetDouble(tid);
                        }
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData, 31);
                        break;
                    }
                    case AriesValueType::DECIMAL: {
                        if( column->isDecimalDataNull( tid ) ) {
                            thd->get_protocol_classic()->store_null();
                            break;
                        }
                        auto columnData = column->GetDecimalAsString( tid );
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData.c_str(),
                                                            columnData.length(), /* res->charset()*/
                                                            default_charset_info);
                        break;
                    }
                    case AriesValueType::COMPACT_DECIMAL: {
                        if( column->isCompactDecimalDataNull( tid ) ) {
                            thd->get_protocol_classic()->store_null();
                            break;
                        }
                        auto columnData = column->GetCompactDecimalAsString( tid, dataType.DataType.Precision, dataType.DataType.Scale );
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData.c_str(),
                                                            columnData.length(), /* res->charset()*/
                                                            default_charset_info);
                        break;
                    }
                    case AriesValueType::CHAR: {
                        if (column->isStringDataNull( tid )) {
                            thd->get_protocol_classic()->store_null();
                            break;
                        }
                        auto columnData = column->GetString( tid );
                        // LOG(INFO) << columnData;
                        thd->get_protocol_classic()->store(columnData.c_str(),
                                                            columnData.length(), /* res->charset()*/
                                                            default_charset_info);
                        break;
                    }
                    case AriesValueType::DATE: {
                        if (column->isNullableColumn()) {
                            auto tmp = column->GetNullableDate(tid);
                            if (tmp && tmp->flag) {
                                thd->get_protocol_classic()->store_date(&tmp->value);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        } else {
                            auto tmp = column->GetDate(tid);
                            if (tmp) {
                                thd->get_protocol_classic()->store_date(tmp);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        }
                    }
                    case AriesValueType::TIME: {
                        if (column->isNullableColumn()) {
                            auto tmp = column->GetNullableTime(tid);
                            if (tmp && tmp->flag) {
                                thd->get_protocol_classic()->store_time(&tmp->value, 0);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        } else {
                            auto tmp = column->GetTime(tid);
                            if (tmp) {
                                thd->get_protocol_classic()->store_time(tmp, 0);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        }
                        break;
                    }
                    case AriesValueType::DATETIME: {
                        if (column->isNullableColumn()) {
                            auto tmp = column->GetNullableDatetime(tid);
                            if (tmp && tmp->flag) {
                                thd->get_protocol_classic()->store(&tmp->value, 0);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        } else {
                            auto tmp = column->GetDatetime(tid);
                            if (tmp) {
                                thd->get_protocol_classic()->store(tmp, 0);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        }
                    }
                    case AriesValueType::TIMESTAMP: {
                        if (column->isNullableColumn()) {
                            auto tmp = column->GetNullableTimestamp(tid);
                            if (tmp && tmp->flag) {
                                AriesDatetime dt = AriesDatetime(tmp->value.getTimeStamp(), 0);
                                thd->get_protocol_classic()->store(&dt, 0);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        } else {
                            auto tmp = column->GetTimestamp(tid);
                            if (tmp) {
                                AriesDatetime dt = AriesDatetime(tmp->getTimeStamp(), 0);
                                thd->get_protocol_classic()->store(&dt, 0);
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        }
                    }
                    case AriesValueType::YEAR: {
                        if (column->isNullableColumn()) {
                            auto tmp = column->GetNullableYear(tid);
                            if (tmp && tmp->flag) {
                                thd->get_protocol_classic()->store_short(tmp->value.getYear());
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        } else {
                            auto tmp = column->GetYear(tid);
                            if (tmp) {
                                thd->get_protocol_classic()->store_short(tmp->getYear());
                            } else {
                                thd->get_protocol_classic()->store_null();
                            }
                            break;
                        }
                    }
                    default: {
                        LOG(WARNING) << "not supported data type: "
                                        << (int32_t) column->GetDataType().DataType.ValueType;
                        break;
                    }
                    // LOG(INFO) << "\t";
                    thd->get_protocol_classic()->store_null();
                }
            }
            // LOG(INFO) << endl;
            if ( thd->get_protocol_classic()->end_row() )
                return true;
        }
        if (send_type == SendDataType::SEND_ROWDATA_ONLY){
            thd->server_status |= SERVER_STATUS_CURSOR_EXISTS;
            // for bug22559575
            if (offset_row_count == 0 && offset_row_count+fetch_row_count >= tupleNum)
                thd->server_status |= SERVER_STATUS_LAST_ROW_SENT;
            my_eof(thd);
        }
    }
    table->SetContent( std::move( tableContent ) );
    LOG(INFO) << "===============================================\n";
    LOG(INFO) << "Query result tuple count " << tupleNum << ", column count " << columnCount;

    return 0;
}

SQLResultPtr SQLExecutor::executeStatements(std::vector<AriesSQLStatementPointer> statements, const string& argDefaultDbName, bool sendResp, bool buildQuery) {
    SQLResultPtr sqlResultPtr = std::make_shared<SQLResult>();
    vector<AbstractMemTablePointer> results;
    string defaultDbName = argDefaultDbName;
    aries_utils::to_lower( defaultDbName );

    THD* thd = current_thd;
    AriesSQLStatementPointer statement;
    auto stmtCnt = statements.size();
    try {
        for ( size_t i = 0; i < stmtCnt; ++i ) {
            bool lastOne = false;
            statement = statements[ i ];
            thd->reset_for_next_command();

            lastOne = ( i == stmtCnt - 1 );
            if ( lastOne )
                thd->server_status &= ~SERVER_MORE_RESULTS_EXISTS;
            else
                thd->server_status |= SERVER_MORE_RESULTS_EXISTS;

            checkTxImplicitCommit( thd, statement );
            startNewTx( thd, statement );

            if (statement->IsQuery()) {
                auto query = std::dynamic_pointer_cast<SelectStructure>(statement->GetQuery());
                auto table = executeQuery(statement->GetTransaction(), query, defaultDbName, buildQuery);
                if ( sendResp )
                {
                    if ( !handleQueryResult( thd, table ) )
                        my_eof( thd );
                }
                results.emplace_back( table );
            } else if (statement->IsInsertStmt()) {
                auto insertStructure = statement->GetInsertStructure();

                auto dbName = insertStructure->GetDbName();
                if( dbName.empty() )
                    dbName = defaultDbName;
                if ( dbName.empty() )
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);

                if ( IsSysDb( dbName ) && !g_isInitialize )
                {
                    auto tableName = insertStructure->GetTableName();
                    if ( !statement->GetTransaction()->isDDL() && tableName != "processlist" && current_thd->m_connection_id != 2 )
                    {
                        std::string msg( "Access denied to database information_schema" );
                        ARIES_EXCEPTION_SIMPLE( ER_DBACCESS_DENIED_ERROR, msg.c_str() );
                    }
                }
                auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
                if( !dbEntry )
                    ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
                auto tableEntry = dbEntry->GetTableByName( insertStructure->GetTableName() );
                if( !tableEntry )
                    ARIES_EXCEPTION( ER_NO_SUCH_TABLE, dbName.c_str(), insertStructure->GetTableName().c_str() );
                if ( tableEntry->IsPartitioned() )
                {
                    ThrowNotSupportedException( "insert into partitioned table" );
                }
                auto tableId = tableEntry->GetId();
                statement->GetTransaction()->SetUpdateTableId( tableId );

                auto table = executeInsert( statement->GetTransaction(), insertStructure, defaultDbName );
                auto tableBlock = ( ( AriesMemTable * )table.get() )->GetContent();
                if ( tableBlock )
                {
                    statement->GetTransaction()->AddModifiedTable( tableId );

                    auto dataBuff = tableBlock->GetColumnBuffer( 1 );
                    int64_t insertedRowCnt = dataBuff->GetInt64( 0 );
                    // ER_INSERT_INFO
                    if ( sendResp )
                        my_ok( thd, insertedRowCnt );
                }
                else
                {
                    ARIES_EXCEPTION( ER_LOCK_OR_ACTIVE_TRANSACTION );
                }
            } else if (statement->IsUpdateStmt()) {

                auto updateStructure = statement->GetUpdateStructure();

                string dbName = updateStructure->GetTargetDbName();
                if ( dbName.empty() )
                    dbName = defaultDbName;
                if ( dbName.empty() )
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);
                if ( IsSysDb( dbName ) && !g_isInitialize )
                {
                    if ( !statement->GetTransaction()->isDDL() && updateStructure->GetTargetTableName() != "processlist" && current_thd->m_connection_id != 2  )
                    {
                        std::string msg( "Access denied to database information_schema" );
                        ARIES_EXCEPTION_SIMPLE( ER_DBACCESS_DENIED_ERROR, msg.c_str() );
                    }
                }
                auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
                if( !dbEntry )
                    ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
                auto tableEntry = dbEntry->GetTableByName( updateStructure->GetTargetTableName() );
                if( !tableEntry )
                    ARIES_EXCEPTION( ER_NO_SUCH_TABLE, dbName.c_str(), updateStructure->GetTargetTableName().c_str() );
                if ( tableEntry->IsPartitioned() )
                {
                    ThrowNotSupportedException( "update partitioned table" );
                }
                auto tableId = tableEntry->GetId();
                statement->GetTransaction()->SetUpdateTableId( tableId );

                auto table = executeUpdate( statement->GetTransaction(), updateStructure, defaultDbName );
                auto tableBlock = ( ( AriesMemTable * )table.get() )->GetContent();
                if ( tableBlock )
                {
                    statement->GetTransaction()->AddModifiedTable( tableId );

                    auto dataBuff = tableBlock->GetColumnBuffer( 1, true );
                    int64_t updatedRowCnt = dataBuff->GetInt64( 1 );
                    char msgBuff[ 1024 ];
                    snprintf( msgBuff, sizeof( msgBuff ),
                             ER( ER_UPDATE_INFO ),
                             dataBuff->GetInt64( 0 ),   // matched
                             updatedRowCnt,             // changed
                             dataBuff->GetInt64( 2 ) ); // warnings
                    if ( sendResp )
                        my_ok( thd, updatedRowCnt, 0, msgBuff );
                }
                else // update failed: another tx has updated the same row, e.g.
                {
                    ARIES_EXCEPTION( ER_LOCK_OR_ACTIVE_TRANSACTION );
                }

            } else if (statement->IsDeleteStmt()) {
                auto deleteStructure = statement->GetDeleteStructure();

                if ( !statement->GetTransaction()->isDDL() )
                {
                    for ( const auto& table : deleteStructure->GetTargetTables() )
                    {
                        auto dbName = table->GetDb().empty() ? defaultDbName : table->GetDb();
                        if ( IsSysDb( dbName ) && !g_isInitialize && table->GetID() != "processlist" && current_thd->m_connection_id != 2 )
                        {
                            std::string msg( "Access denied to database information_schema" );
                            ARIES_EXCEPTION_SIMPLE( ER_DBACCESS_DENIED_ERROR, msg.c_str() );
                        }
                    }
                }
                auto table = executeDelete( statement->GetTransaction(), deleteStructure, defaultDbName );
                auto tableBlock = ( ( AriesMemTable * )table.get() )->GetContent();
                if ( tableBlock )
                {
                    auto dataBuff = tableBlock->GetColumnBuffer( 1, true );
                    if ( sendResp )
                        my_ok( thd, dataBuff->GetInt64( 0 ) );
                }
                else
                {
                    ARIES_EXCEPTION( ER_LOCK_OR_ACTIVE_TRANSACTION );
                }
            } else if (statement->IsShowStatement()) {
                auto tmpResult = executeShowStatement(statement->GetShowStructurePointer(), defaultDbName);
                if (tmpResult && tmpResult->GetResults().size() > 0) {
                    auto tmpTable = tmpResult->GetResults()[ 0 ];
                    if ( sendResp )
                    {
                        if ( !handleQueryResult( thd, tmpTable ) )
                            my_eof( thd );
                    }
                    results.emplace_back( tmpTable );
                }
            } else if (statement->IsSetStatement()) {
                executeSetStatements(statement->GetSetStructurePtrs(), defaultDbName);
            } else if (statement->IsCommand()) {
                auto result = executeCommand(statement->GetCommand(), defaultDbName);
                if (result != nullptr) {
                    if ( sendResp )
                    {
                        if ( !handleQueryResult( thd, result ) )
                            my_eof( thd );
                    }
                    results.emplace_back(result);
                }
            } else if (statement->IsPreparedStatement()) {
                executePreparedStatement(statement->GetPreparedStmtStructurePtr(), defaultDbName);
            } else if (statement->IsAdminStmt()) {
                executeAdminStatement(statement->GetAdminPtr(), defaultDbName);
            } else if (statement->IsLoadDataStmt()) {
                auto tx = statement->GetTransaction();
                executeLoadDataStatement( tx,
                                          statement->GetLoadDataStructurePtr(),
                                          defaultDbName );
            } else if ( statement->IsTxStmt() ) {
                executeTxStatement( thd, statement );
            } else if (statement->IsExplainStatement()) {
                auto table = std::make_shared<AriesMemTable>();
                auto table_block = std::make_unique<AriesTableBlock>();
                table_block->AddColumnName(1, "EXPLAIN");
                auto column = std::make_shared<AriesColumn>();
                auto query = std::dynamic_pointer_cast<SelectStructure>(statement->GetExplainQuery());
                BuildQuery(query, defaultDbName);
                if (!query->IsSetQuery() && nullptr != query->GetFromPart()) {
                    SQLTreeNodePointer query_plan_tree = query->GetQueryPlanTree();
                    QueryOptimizerPointer query_optimizer = QueryOptimizer::GetQueryOptimizer();
                    query_optimizer->OptimizeTree(query_plan_tree);
                }

                auto data_buffer = aries_acc::CreateDataBufferWithValue(query->GetQueryPlanTree()->ToString(0), 1);
                column->AddDataBuffer(data_buffer);
                table_block->AddColumn(1, column);
                table->SetContent(std::move(table_block));
                if ( sendResp )
                {
                    if ( !handleQueryResult( thd, table ) )
                        my_eof( thd );
                }
                results.emplace_back(table);
            } else {
                // endTx( thd, statement, TransactionStatus::ABORTED );
                sqlResultPtr->SetErrorCode(ER_SYNTAX_ERROR);
                if ( sendResp )
                    my_error(ER_SYNTAX_ERROR, MYF(0));
                LOG(ERROR) << "Syntax error.";
                break;
            }
            // for implicit transaction, each statement is a transaction
            if ( NeedEndTransaction( statement, thd ) )
            {
                endTx( thd, statement, TransactionStatus::COMMITTED );
            }
            if ( !lastOne && sendResp )
                thd->send_statement_status();
        }

        sqlResultPtr->SetSuccess(true);
        sqlResultPtr->SetResults(results);

        // if ( sendResp )
        // {
        //     AriesMvccTableManager::GetInstance().clearAll();
        //     AriesInitialTableManager::GetInstance().clearAll();
        // }
    } catch (const AriesException& e) {
        if( e.errCode == ER_ENGINE_OUT_OF_MEMORY )
        {
            AriesNullValueProvider::GetInstance().Clear();
            // AriesGPUMemoryManager::GetInstance().ResetTableZone();
            // AriesGPUMemoryManager::GetInstance().ResetWorkZone();
        }

        if ( NeedEndTransaction( statement, thd ) )
            endTx( thd, statement, TransactionStatus::ABORTED );
        sqlResultPtr->SetError(e.errCode, e.errMsg);
        if ( sendResp )
            my_message(e.errCode, e.errMsg.data(), MYF(0));
    } catch (const std::exception &e) {
        if ( NeedEndTransaction( statement, thd ) )
            endTx( thd, statement, TransactionStatus::ABORTED );
        sqlResultPtr->SetErrorCode(ER_UNKNOWN_ERROR);
        if ( sendResp )
            my_error(ER_UNKNOWN_ERROR, MYF(0));
        LOG(ERROR) << "std::exception: " << e.what();
    } catch (...) {
        if ( NeedEndTransaction( statement, thd ) )
            endTx( thd, statement, TransactionStatus::ABORTED );
        sqlResultPtr->SetErrorCode(ER_UNKNOWN_ERROR);
        if ( sendResp )
            my_error(ER_UNKNOWN_ERROR, MYF(0));
        LOG(ERROR) << "Unknown error.";
    }

    return sqlResultPtr;
}

ulong readAndStoreDataFile( string& path )
{
    NET *net = current_thd->get_protocol_classic()->get_net();
    char name_buff[FN_REFLEN] = {0};
    int tmpFd = create_temp_file( name_buff, nullptr, "aries_load_data_" );
    if ( tmpFd >= 0 )
    {
        ulong totalLen = 0;
        do
        {
            ulong readLen = my_net_read( net );
            if ( packet_error == readLen )
            {
                ARIES_EXCEPTION( ER_NET_READ_ERROR );
            }
            if ( 0 == readLen )
                break;

            size_t writeRet = my_write( tmpFd,
                                        net->read_pos,
                                        readLen,
                                        MYF( MY_FNABP ) );
            if ( 0 != writeRet )
            {
                char msgBuff[256] = {0};
                char* msg = strerror_r( my_errno(), msgBuff, 256 );
                ARIES_EXCEPTION_SIMPLE( ER_IO_WRITE_ERROR, "Failed to write data file, error: " + std::to_string( my_errno() ) + ", msg: " +  msg );
            }
            totalLen += readLen;

        } while ( true );
        path.assign( name_buff );
        return totalLen;
    }
    else
    {
        set_my_errno(errno);
        char errbuf[MYSYS_STRERROR_SIZE];
        ARIES_EXCEPTION( EE_CANTCREATEFILE, name_buff, my_errno(), strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
    }
}

void SQLExecutor::executeLoadDataStatement( aries_engine::AriesTransactionPtr& tx,
                                            const LoadDataStructurePtr& loadDataStructurePtr,
                                            const string& defaultDbName)
{
    string dbName = loadDataStructurePtr->tableIdent->GetDb();
    if ( dbName.empty() )
    {
        dbName = defaultDbName;
        if (dbName.empty()) {
            ARIES_EXCEPTION(ER_NO_DB_ERROR);
        }
    }
    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if ( !dbEntry )
    {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    string tableName = loadDataStructurePtr->tableIdent->GetID();
    auto tableEntry = dbEntry->GetTableByName(  tableName );
    if ( !tableEntry )
    {
        ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, tableName.data() );
    }

    const auto fieldTerm = loadDataStructurePtr->sqlExchange.field.field_term;
    const auto escaped = loadDataStructurePtr->sqlExchange.field.escaped;
    const auto enclosed = loadDataStructurePtr->sqlExchange.field.enclosed;
    string fileName = loadDataStructurePtr->sqlExchange.file_name;
    ulong skipLines = loadDataStructurePtr->sqlExchange.skip_lines;

    if (escaped->length() > 1 || enclosed->length() > 1) {
        ARIES_EXCEPTION( ER_WRONG_FIELD_TERMINATORS );
    }

    THD* thd = current_thd;
    char name[FN_REFLEN] = {0};
    string realDataFilePath;
    if ( loadDataStructurePtr->isLocalFile )
    {
        (void)net_request_file(thd->get_protocol_classic()->get_net(),
                               loadDataStructurePtr->sqlExchange.file_name);
        ulong fileLen = readAndStoreDataFile( realDataFilePath );
        if ( IsCurrentThdKilled() )
        {
            SendKillMessage();
        }
        if ( 0 == fileLen )
        {
            LOG(INFO) << "Load local data file, 0 bytes read";
            snprintf(name, sizeof(name),
                     ER(ER_LOAD_INFO),
                     0, 0,
                     0,
                    // (long) info.stats.records, (long) info.stats.deleted,
                    // (long) (info.stats.records - info.stats.copied),
                    /* (long) thd->get_stmt_da()->current_statement_cond_count()*/ 0);
            my_ok( thd, 0, 0, name );
            return;
        }
    }
    else
    {
        if ( !dirname_length( fileName.data() ) ) {
            string loadDir = defaultDbName;
            if ( loadDir.empty() )
                loadDir = loadDataStructurePtr->tableIdent->GetDb();
            strxnmov(name, FN_REFLEN - 1, mysql_real_data_home, loadDir.data(), NullS);
            (void)fn_format(name, fileName.data(), name, "",
                            MY_RELATIVE_PATH | MY_UNPACK_FILENAME);
        } else {
            (void)fn_format(
                    name, fileName.data(), mysql_real_data_home, "",
                    MY_RELATIVE_PATH | MY_UNPACK_FILENAME | MY_RETURN_REAL_PATH);
        }
        realDataFilePath.assign( name );
        struct stat stat_info;
        if (stat(name, &stat_info))
        {
            set_my_errno(errno);
            char errbuf[MYSYS_STRERROR_SIZE];
            ARIES_EXCEPTION( EE_STAT, name,
                             my_errno(), strerror_r(my_errno(), errbuf, sizeof(errbuf)));
        }
        if (!((stat_info.st_mode & S_IFLNK) != S_IFLNK &&   // symlink
              ((stat_info.st_mode & S_IFREG) == S_IFREG ||  // regular file
               (stat_info.st_mode & S_IFIFO) == S_IFIFO)))  // named pipe
        {
            ARIES_EXCEPTION( ER_TEXTFILE_NOT_READABLE, name );
        }
    }
    LOG(INFO) << "Loading " << ( loadDataStructurePtr->isLocalFile ? "local" : "server" ) << " data file " << realDataFilePath;
    int64_t ret =
    importCsvFile( tx, dbEntry, tableEntry, realDataFilePath,
                   skipLines,
                   *loadDataStructurePtr->sqlExchange.field.field_term,
                   loadDataStructurePtr->sqlExchange.escaped_given(),
                   *loadDataStructurePtr->sqlExchange.field.escaped,
                   loadDataStructurePtr->sqlExchange.field.opt_enclosed,
                   *loadDataStructurePtr->sqlExchange.field.enclosed,
                   *loadDataStructurePtr->sqlExchange.line.line_term,
                   *loadDataStructurePtr->sqlExchange.line.line_start );
    if ( loadDataStructurePtr->isLocalFile )
        unlink( realDataFilePath.data() );

    aries_engine::AriesDataCache::GetInstance().removeCache( dbEntry->GetName(), tableEntry->GetName() );

    // Records: 0  Deleted: 0  Skipped: 0  Warnings: 0
    snprintf(name, sizeof(name),
             ER(ER_LOAD_INFO),
             ret, 0,
             skipLines,
            // (long) info.stats.records, (long) info.stats.deleted,
            // (long) (info.stats.records - info.stats.copied),
            /* (long) thd->get_stmt_da()->current_statement_cond_count()*/ 0);
    if ( IsServerInitDone() )
    {
        my_ok( thd, ret, 0, name );
    }
}

SQLResultPtr SQLExecutor::executeShowStatement(const ShowStructurePtr& showStructurePointer, const std::string& defaultDbName) {
    switch (showStructurePointer->showCmd) {
        case SHOW_CMD::SHOW_DATABASES:
            return showDatabases();
            break;
        case SHOW_CMD::SHOW_TABLES: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->tableNameStructureSPtr->dbName;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);
                }
            }
            return showTables(db, showSchemaStructurePointer->full);
            break;
        }
        case SHOW_CMD::SHOW_TRIGGERS: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->id;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION( ER_NO_DB_ERROR );
                }
            }
            return showTriggers(db, showStructurePointer->wild, showStructurePointer->where, showStructurePointer->wildOrWhereStr);
            break;
        }
        case SHOW_CMD::SHOW_EVENTS: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->tableNameStructureSPtr->dbName;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION( ER_NO_DB_ERROR );
                }
            }
            return showEvents(db, showStructurePointer->wild, showStructurePointer->where, showStructurePointer->wildOrWhereStr);
            break;
        }
        case SHOW_CMD::SHOW_TABLE_STATUS: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->tableNameStructureSPtr->dbName;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION( ER_NO_DB_ERROR );
                }
            }
            return showTableStatus(db, showSchemaStructurePointer->wild, showSchemaStructurePointer->where);
            break;
        }
        case SHOW_CMD::SHOW_OPEN_TABLES: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->tableNameStructureSPtr->dbName;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);
                }
            }
            return showOpenTables(db, showStructurePointer->wild, showStructurePointer->where);
            break;
        }
        case SHOW_CMD::SHOW_PLUGINS:
            return showPlugins();
            break;
        case SHOW_CMD::SHOW_COLUMNS: {
            auto showColumnsStructurePointer = std::dynamic_pointer_cast<ShowColumnsStructure>(showStructurePointer);
            return showColumns(showColumnsStructurePointer, defaultDbName);
            break;
        }
        case SHOW_CMD::SHOW_INDEX: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->tableNameStructureSPtr->dbName;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);
                }
            }
            return showIndex(showSchemaStructurePointer->full, db,
                             showSchemaStructurePointer->tableNameStructureSPtr->tableName,
                             showSchemaStructurePointer->where);
            break;
        }
        case SHOW_CMD::SHOW_ENGINES:
            return showEngines();
            break;
        case SHOW_CMD::SHOW_ENGINE_STATUS: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string name = showSchemaStructurePointer->id;
            return showEngineStatus(name);
            break;
        }
        case SHOW_CMD::SHOW_ENGINE_MUTEX: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string name = showSchemaStructurePointer->id;
            return showEngineMutex(name);
            break;
        }
        case SHOW_CMD::SHOW_ERRORS: {
            return showErrors(showStructurePointer->limitExpr);
            break;
        }
        case SHOW_CMD::SHOW_WARNINGS: {
            return showWarnings(showStructurePointer->limitExpr);
            break;
        }
        case SHOW_CMD::SHOW_STATUS:
            return showStatus(true);
            break;
        case SHOW_CMD::SHOW_CHAR_SET: {
            return showCharSet(showStructurePointer->wild, showStructurePointer->where, showStructurePointer->wildOrWhereStr);
            break;
        }
        case SHOW_CMD::SHOW_COLLATION:
            return showCollation(showStructurePointer->wild, showStructurePointer->where, showStructurePointer->wildOrWhereStr);
            break;
        case SHOW_CMD::SHOW_CREATE_DB: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->id;
            return showCreateDb(db);
            break;
        }
        case SHOW_CMD::SHOW_CREATE_TABLE: {
            auto showSchemaStructurePointer = std::dynamic_pointer_cast<ShowSchemaInfoStructure>(showStructurePointer);
            string db = showSchemaStructurePointer->tableNameStructureSPtr->dbName;
            if (db.empty()) {
                db = defaultDbName;
                if (db.empty()) {
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);
                }
            }
            return showCreateTable(db, showSchemaStructurePointer->tableNameStructureSPtr->tableName);
            break;
        }
        case SHOW_CMD::SHOW_FUNC_STATUS:
        case SHOW_CMD::SHOW_PROCEDURE_STATUS: {
            return showFunctionStatus(showStructurePointer->wild, showStructurePointer->where);
            break;
        }
        case SHOW_CMD::SHOW_PROCESS_LIST: {
            return showProcessList(showStructurePointer->full);
            break;
        }
        case SHOW_CMD::SHOW_PRIVILEGES: {
            return showPrivileges();
            break;
        }
        case SHOW_CMD::SHOW_MASTER_STATUS: {
            return showMasterStatus();
            break;
        }
        case SHOW_CMD::SHOW_SLAVE_HOSTS: {
            return showSlaveHosts();
            break;
        }
        case SHOW_CMD::SHOW_VARIABLES: {
            auto showVarStructure = std::dynamic_pointer_cast<ShowStatusVariableStructure>(showStructurePointer);
            return showSysVar(showVarStructure->global,
                              showVarStructure->wild,
                              showVarStructure->where,
                              showVarStructure->wildOrWhereStr);
            break;
        }

        default:
            ARIES_EXCEPTION(ER_NOT_SUPPORTED_YET, aries::ToString(showStructurePointer->showCmd).data());
            break;
    }
}

static string ShowWhereExprToSelectExpr(const BiaodashiPointer& expr, const unordered_map<string, string>& aliasMap ) {

    auto commonBiaodashiPtr = std::dynamic_pointer_cast<CommonBiaodashi>(expr);
    std::string ret = "";

    std::string content_str = commonBiaodashiPtr->ContentToString();

    switch (commonBiaodashiPtr->GetType())
    {
        case BiaodashiType::Biaoshifu: {
            auto identPtr = boost::get<SQLIdentPtr>(commonBiaodashiPtr->GetContent());
            auto it = aliasMap.find(identPtr->id);
            if (it != aliasMap.end()) {
                content_str = it->second;
            } else {
                // ERROR 1054 (42S22): Unknown column 'Default collatioan' in 'where clause'
                ARIES_EXCEPTION(ER_BAD_FIELD_ERROR,
                                identPtr->id.c_str(),
                                "where clause");
            }
            ret = "(" + content_str + ")";
            break;
        }
        case BiaodashiType::Lie: {
            auto it = aliasMap.find(content_str);
            if (it != aliasMap.end()) {
                content_str = it->second;
            } else {
                ARIES_EXCEPTION(ER_BAD_FIELD_ERROR,
                                content_str.c_str(),
                                "where clause");
            }
            ret = "(" + content_str + ")";
            break;
        }
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Zhenjia:
        case BiaodashiType::Query:
            ret = "(" + content_str + ")";
            break;

        case BiaodashiType::Zifuchuan:
            ret = "(\"" + content_str + "\")";
            break;

        case BiaodashiType::Star:
            ret = "(";
            ret += content_str.empty() ? ("*") : (content_str + ".*");
            ret += ")";
            break;

        case BiaodashiType::Hanshu:
        case BiaodashiType::SQLFunc:
        case BiaodashiType::ExprList:
            ret = content_str;
            ret += "(";
            ret += commonBiaodashiPtr->ChildrenToString();
            ret += ")";
            break;

        case BiaodashiType::Yunsuan:
        case BiaodashiType::Bijiao:
        case BiaodashiType::Andor:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 2, "children count of this node should be 2");
            ret = "(";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " " + content_str + " ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(1), aliasMap);
            ret += ")";
            break;

        case BiaodashiType::Shuzu:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 2, "children count of this node should be 2");
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += "[";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(1), aliasMap);
            ret += "]";
            break;

        case BiaodashiType::Qiufan:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 1, "children count of this node should be 1");
            ret = "NOT (" + ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap) + ")";
            break;

        case BiaodashiType::Likeop:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 2, "children count of this node should be 2");
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " like ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(1), aliasMap);
            break;

        case BiaodashiType::Inop:
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " in (";
            ret += commonBiaodashiPtr->ChildrenToString_Skip0();
            ret += ")";
            break;

        case BiaodashiType::NotIn:
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " not in (";
            ret += commonBiaodashiPtr->ChildrenToString_Skip0();
            ret += ")";
            break;

        case BiaodashiType::Between:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 3, "children count of this node should be 3");
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " between ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(1), aliasMap);
            ret += " and ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(2), aliasMap);
            break;

        case BiaodashiType::Cunzai:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 1, "children count of this node should be 1");
            ret = "exists (";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += ")";
            break;

        case BiaodashiType::Kuohao:
            ARIES_ASSERT(commonBiaodashiPtr->GetChildrenCount() == 1, "children count of this node should be 1");
            ret = "(";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += ")";
            break;

        case BiaodashiType::Case:
            ret = commonBiaodashiPtr->CaseToString();
            break;
        case BiaodashiType::Decimal:
            ret = boost::get<std::string>(commonBiaodashiPtr->GetContent());
            break;
        case BiaodashiType::IfCondition:
            ret = "IF ( ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " ) ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(1), aliasMap);
            ret += " ELSE ";
            ret += ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(2), aliasMap);
            ret += " ENDIF";
            break;
        case BiaodashiType::IsNotNull:
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " Is Not Null";
            break;
        case BiaodashiType::IsNull:
            ret = ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap);
            ret += " Is Null";
            break;
        case BiaodashiType::Null:
            ret = "Null";
            break;
        case BiaodashiType::Distinct:
            ret = "Distinct";
            break;
        case BiaodashiType::IntervalExpression:
            ret = "Interval " + ShowWhereExprToSelectExpr(commonBiaodashiPtr->GetChildByIndex(0), aliasMap) + " " + boost::get<std::string>(commonBiaodashiPtr->GetContent());
            break;

        case BiaodashiType::QuestionMark:
            ret = "?";
            break;

        default:
            ARIES_ASSERT( 0, "unsupported expression type: " + std::to_string(static_cast<int>( commonBiaodashiPtr->GetType() )) );
    }

    return ret;
}
SQLResultPtr showSlaveHosts() {
    THD* thd = current_thd;
    vector<Send_field *> fields;

    Send_field field1;
    field1.col_name = "Server_id";
    field1.type = MYSQL_TYPE_LONG;
    field1.length = 10;
    field1.flags = 0x00a1;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "host";
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 60;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Port";
    field3.type = MYSQL_TYPE_LONG;
    field3.length = 7;
    field3.flags = 0x00a1;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.col_name = "Master_id";
    field4.type = MYSQL_TYPE_LONG;
    field4.length = 10;
    field4.flags = 0x00a1;
    field4.decimals = 0;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.col_name = "Slave_UUID";
    field5.type = MYSQL_TYPE_VAR_STRING;
    field5.length = 108;
    field5.flags = 0x0001;
    field5.decimals = 0;
    fields.insert(fields.end(), &field5);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;

}
SQLResultPtr showMasterStatus() {
    THD* thd = current_thd;
    vector<Send_field *> fields;

    Send_field field1;
    field1.col_name = "File";
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 1536;
    field1.flags = 0x0001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "Position";
    field2.type = MYSQL_TYPE_LONGLONG;
    field2.length = 20;
    field2.flags = 0x00a1;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Binlog_Do_DB";
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 765;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.col_name = "Binlog_Ignore_DB";
    field4.type = MYSQL_TYPE_VAR_STRING;
    field4.length = 765;
    field4.flags = 0x0001;
    field4.decimals = 0;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.col_name = "Executed_Gtid_Set";
    field5.type = MYSQL_TYPE_VAR_STRING;
    field5.length = 0;
    field5.flags = 0x0001;
    field5.decimals = 0;
    fields.insert(fields.end(), &field5);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF ) )
        my_eof(thd);
    return nullptr;

}
SQLResultPtr showPrivileges() {
    THD* thd = current_thd;
    vector<Send_field *> fields;

    Send_field field1;
    field1.col_name = "Privilege";
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 30;
    field1.flags = 0x0001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "Context";
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 45;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Comment";
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 192;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;
}
SQLResultPtr SQLExecutor::executeSelect( const string& sql, const string& defaultDbName ) {
    LOG(INFO) << "Executing select: " << sql << ", default database: " << defaultDbName;
    auto result = std::make_shared<SQLResult>();
    std::vector<AriesSQLStatementPointer> statements;
    std::pair<int, string> parseResult = ParseSQL( sql, false, statements );
    if ( 0 != parseResult.first )
    {
        if ( ER_FAKE_IMPL_OK != parseResult.first && ER_FAKE_IMPL_EOF != parseResult.first )
        {
            result->SetError( parseResult.first, parseResult.second);
            my_message( parseResult.first, parseResult.second.data(), MYF(0) );
        }
        else
        {
            result->SetSuccess( true );
        }
    }
    else
    {
        vector<AbstractMemTablePointer> results;
        auto statement = statements[ 0 ];
        auto thd = current_thd;
        if ( thd )
        {
            statement->SetTransaction( thd->m_tx );
        }
        auto query = std::dynamic_pointer_cast<SelectStructure>(statement->GetQuery());
        results.emplace_back( executeQuery(statement->GetTransaction(), query, defaultDbName ) );
        result->SetResults( results );
        result->SetSuccess( true );
    }
    return  result;
}
SQLResultPtr showProcessList(bool full) {
    static unordered_map<string, string> aliasMap {
            {"id", "ID"},
            {"user", "USER"},
            {"host", "HOST"},
            {"db", "DB"},
            {"command", "COMMAND"},
            {"time", "TIME"},
            {"state", "STATE"},
            {"info", "INFO"}
    };
    string sql = R"(SELECT ID as id, USER as user,
                    HOST as host, DB as db, COMMAND as command, TIME as time,
                    STATE as state, INFO as info
                    FROM information_schema.PROCESSLIST)";
    THD* thd = current_thd;
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());
}
SQLResultPtr showCharSet(const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr) {
    static unordered_map<string, string> aliasMap {
            {"charset", "CHARACTER_SET_NAME"},
            {"description", "DESCRIPTION"},
            {"default collation", "DEFAULT_COLLATE_NAME"},
            {"maxlen", "MAXLEN"}
    };
    string sql = R"(SELECT CHARACTER_SET_NAME as Charset,
                    DESCRIPTION as Description, DEFAULT_COLLATE_NAME as `Default collation`,
                    MAXLEN as Maxlen from `information_schema`.`CHARACTER_SETS`)";
    if (!wild.empty()) {
        sql.append(" WHERE character_set_name").append(" LIKE ").append(exprStr);
    } else if (whereExpr) {
        sql.append(" WHERE ").append(ShowWhereExprToSelectExpr(whereExpr, aliasMap));
    }
    THD* thd = current_thd;
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());

}

SQLResultPtr showSysVar(bool global,
                        const string& wild,
                        const BiaodashiPointer& whereExpr,
                        const string& exprStr) {
    static unordered_map<string, string> aliasMap {
            {"variable_name", "variable_name"},
            {"value", "variable_value"}
    };
    string sql = R"(SELECT VARIABLE_NAME AS Variable_name, VARIABLE_VALUE AS Value from information_schema.)";
    if (global) {
        sql.append("GLOBAL_VARIABLES");
    } else {
        sql.append("SESSION_VARIABLES");
    }
    if (!wild.empty()) {
        sql.append(" WHERE variable_name").append(" LIKE ").append(exprStr);
    } else if (whereExpr) {
        sql.append(" WHERE ").append(ShowWhereExprToSelectExpr(whereExpr, aliasMap));
    }
    THD* thd = current_thd;
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());

}
SQLResultPtr showFunctionStatus(const string& like, const BiaodashiPointer& whereExpr) {
    THD* thd = current_thd;
    vector<Send_field *> fields;

    Send_field field1;
    field1.db_name = "information_schema";
    field1.table_name = field1.org_table_name = "ROUTINES";
    field1.col_name = "Db";
    field1.org_col_name = "ROUTINE_SCHEMA";
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 256;
    field1.flags = 0x0001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.db_name = "information_schema";
    field2.table_name = field2.org_table_name = "ROUTINES";
    field2.col_name = "Name";
    field2.org_col_name = "ROUTINE_NAME";
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 256;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.db_name = "information_schema";
    field3.table_name = field3.org_table_name = "ROUTINES";
    field3.col_name = "Type";
    field3.org_col_name = "ROUTINE_TYPE";
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 36;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.db_name = "information_schema";
    field4.table_name = field4.org_table_name = "ROUTINES";
    field4.col_name = "Definer";
    field4.org_col_name = "DEFINER";
    field4.type = MYSQL_TYPE_VAR_STRING;
    field4.length = 372;
    field4.flags = 0x0001;
    field4.decimals = 0;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.db_name = "information_schema";
    field5.table_name = field5.org_table_name = "ROUTINES";
    field5.col_name = "Modified";
    field5.org_col_name = "LAST_ALTERED";
    field5.type = MYSQL_TYPE_DATETIME;
    field5.length = 19;
    field5.flags = 0x0081;
    field5.decimals = 0;
    fields.insert(fields.end(), &field5);

    Send_field field6;
    field6.db_name = "information_schema";
    field6.table_name = field6.org_table_name = "ROUTINES";
    field6.col_name = "Created";
    field6.org_col_name = "CREATED";
    field6.type = MYSQL_TYPE_DATETIME;
    field6.length = 19;
    field6.flags = 0x0081;
    field6.decimals = 0;
    fields.insert(fields.end(), &field6);

    Send_field field7;
    field7.db_name = "information_schema";
    field7.table_name = field7.org_table_name = "ROUTINES";
    field7.col_name = "Security_type";
    field7.org_col_name = "SECURITY_TYPE";
    field7.type = MYSQL_TYPE_VAR_STRING;
    field7.length = 28;
    field7.flags = 0x0001;
    field7.decimals = 0;
    fields.insert(fields.end(), &field7);

    Send_field field8;
    field8.db_name = "information_schema";
    field8.table_name = field8.org_table_name = "ROUTINES";
    field8.col_name = "Comment";
    field8.org_col_name = "ROUTINE_COMMENT";
    field8.type = MYSQL_TYPE_BLOB;
    field8.length = 786420;
    field8.flags = 0x0011;
    field8.decimals = 0;
    fields.insert(fields.end(), &field8);

    Send_field field9;
    field9.db_name = "information_schema";
    field9.table_name = field9.org_table_name = "ROUTINES";
    field9.col_name = "character_set_client";
    field9.org_col_name = "CHARACTER_SET_CLIENT";
    field9.type = MYSQL_TYPE_BLOB;
    field9.length = 128;
    field9.flags = 0x0001;
    field9.decimals = 0;
    fields.insert(fields.end(), &field9);

    Send_field field10;
    field10.db_name = "information_schema";
    field10.table_name = field10.org_table_name = "ROUTINES";
    field10.col_name = "collation_connection";
    field10.org_col_name = "COLLATION_CONNECTION";
    field10.type = MYSQL_TYPE_BLOB;
    field10.length = 128;
    field10.flags = 0x0001;
    field10.decimals = 0;
    fields.insert(fields.end(), &field10);

    Send_field field11;
    field11.db_name = "information_schema";
    field11.table_name = field11.org_table_name = "ROUTINES";
    field11.col_name = "Database Collation";
    field11.org_col_name = "DATABASE_COLLATION";
    field11.type = MYSQL_TYPE_BLOB;
    field11.length = 128;
    field11.flags = 0x0001;
    field11.decimals = 0;
    fields.insert(fields.end(), &field11);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;
}
SQLResultPtr showIndex(bool extended, const string& dbName, const string& tableName, const BiaodashiPointer& whereExpr) {
    THD* thd = current_thd;
    auto dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry )
    {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    auto tableEntry = dbEntry->GetTableByName(tableName);
    if (!tableEntry) {
        ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, tableName.data() );
    }
    vector<Send_field *> fields;

    Send_field field1;
    field1.db_name = "information_schema";
    field1.table_name = field1.org_table_name = "STATISTICS";
    field1.col_name = "Table";
    field1.org_col_name = "TABLE_NAME";
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 192;
    field1.flags = 0x0001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.db_name = "information_schema";
    field2.table_name = field2.org_table_name = "STATISTICS";
    field2.col_name = "Non_unique";
    field2.org_col_name = "NON_UNIQUE";
    field2.type = MYSQL_TYPE_LONGLONG;
    field2.length = 1;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.db_name = "information_schema";
    field3.table_name = field3.org_table_name = "STATISTICS";
    field3.col_name = "Key_name";
    field3.org_col_name = "INDEX_NAME";
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 192;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.db_name = "information_schema";
    field4.table_name = field4.org_table_name = "STATISTICS";
    field4.col_name = "Seq_in_index";
    field4.org_col_name = "SEQ_IN_INDEX";
    field4.type = MYSQL_TYPE_LONGLONG;
    field4.length = 2;
    field4.flags = 0x0001;
    field4.decimals = 0;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.db_name = "information_schema";
    field5.table_name = field5.org_table_name = "STATISTICS";
    field5.col_name = "Column_name";
    field5.org_col_name = "COLUMN_NAME";
    field5.type = MYSQL_TYPE_VAR_STRING;
    field5.length = 192;
    field5.flags = 0x0001;
    field5.decimals = 0;
    fields.insert(fields.end(), &field5);

    Send_field field6;
    field6.db_name = "information_schema";
    field6.table_name = field6.org_table_name = "STATISTICS";
    field6.col_name = "Collation";
    field6.org_col_name = "COLLATION";
    field6.type = MYSQL_TYPE_VAR_STRING;
    field6.length = 192;
    field6.flags = 0x0001;
    field6.decimals = 0;
    fields.insert(fields.end(), &field6);

    Send_field field7;
    field7.db_name = "information_schema";
    field7.table_name = field7.org_table_name = "STATISTICS";
    field7.col_name = "Cardinality";
    field7.org_col_name = "CARDINALITY";
    field7.type = MYSQL_TYPE_LONGLONG;
    field7.length = 21;
    field7.flags = 0x0000;
    field7.decimals = 0;
    fields.insert(fields.end(), &field7);

    Send_field field8;
    field8.db_name = "information_schema";
    field8.table_name = field8.org_table_name = "STATISTICS";
    field8.col_name = "Sub_part";
    field8.org_col_name = "SUB_PART";
    field8.type = MYSQL_TYPE_LONGLONG;
    field8.length = 21;
    field8.flags = 0x0000;
    field8.decimals = 0;
    fields.insert(fields.end(), &field8);

    Send_field field9;
    field9.db_name = "information_schema";
    field9.table_name = field9.org_table_name = "STATISTICS";
    field9.col_name = "Packed";
    field9.org_col_name = "PACKED";
    field9.type = MYSQL_TYPE_VAR_STRING;
    field9.length = 30;
    field9.flags = 0x0000;
    field9.decimals = 0;
    fields.insert(fields.end(), &field9);

    Send_field field10;
    field10.db_name = "information_schema";
    field10.table_name = field10.org_table_name = "STATISTICS";
    field10.col_name = "Null";
    field10.org_col_name = "NULLABLE";
    field10.type = MYSQL_TYPE_VAR_STRING;
    field10.length = 9;
    field10.flags = 0x0001;
    field10.decimals = 0;
    fields.insert(fields.end(), &field10);

    Send_field field11;
    field11.db_name = "information_schema";
    field11.table_name = field11.org_table_name = "STATISTICS";
    field11.col_name = "Index_type";
    field11.org_col_name = "INDEX_TYPE";
    field11.type = MYSQL_TYPE_VAR_STRING;
    field11.length = 48;
    field11.flags = 0x0001;
    field11.decimals = 0;
    fields.insert(fields.end(), &field11);

    Send_field field12;
    field12.db_name = "information_schema";
    field12.table_name = field12.org_table_name = "STATISTICS";
    field12.col_name = "Comment";
    field12.org_col_name = "COMMENT";
    field12.type = MYSQL_TYPE_VAR_STRING;
    field12.length = 48;
    field12.flags = 0x0000;
    field12.decimals = 0;
    fields.insert(fields.end(), &field12);

    Send_field field13;
    field13.db_name = "information_schema";
    field13.table_name = field13.org_table_name = "STATISTICS";
    field13.col_name = "Index_comment";
    field13.org_col_name = "INDEX_COMMENT";
    field13.type = MYSQL_TYPE_VAR_STRING;
    field13.length = 3072;
    field13.flags = 0x0001;
    field13.decimals = 0;
    fields.insert(fields.end(), &field13);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;
}
SQLResultPtr showStatus(bool global) {
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field1;
    field1.db_name = "";
    field1.table_name = field1.org_table_name = "session_status";
    field1.col_name = "Variable_name";
    field1.org_col_name = field1.col_name;
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 256;
    field1.flags = 0x1001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.db_name = "";
    field2.table_name = field2.org_table_name = "session_status";
    field2.col_name = "Value";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 4096;
    field2.flags = 0;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);

    /*
    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store("Aborted_clients", default_charset_info);
    thd->get_protocol_classic()->store("0", default_charset_info);
    thd->get_protocol_classic()->end_row();

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store("Bytes_received", default_charset_info);
    thd->get_protocol_classic()->store("0", default_charset_info);
    thd->get_protocol_classic()->end_row();

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store("Bytes_sent", default_charset_info);
    thd->get_protocol_classic()->store("0", default_charset_info);
    thd->get_protocol_classic()->end_row();

    */

    return nullptr;
}
// start transaction, commit, rollback
void SQLExecutor::executeTxStatement( THD* thd, const AriesSQLStatementPointer& statement )
{
    auto txStructurePtr = statement->GetTxStructurePtr();
    switch ( txStructurePtr->txCmd )
    {
    case TX_START:
    {
        // handled in checkTxImplicitCommit
        LOG( INFO ) << "start explicit transaction";
        thd->m_explicitTx = true;
        break;
    }
    case TX_COMMIT:
        endTx( thd, statement, TransactionStatus::COMMITTED );
        break;
    case TX_ROLLBACK:
        endTx( thd, statement, TransactionStatus::ABORTED );
        break;
    }
    my_ok( thd );
}

/**
 * mysql 5.7.26:
 * mysql> set @a = a;
ERROR 1054 (42S22): Unknown column 'a' in 'field list'
 */
void CheckExprForSetStatements( CommonBiaodashiPtr& expr, const string& defaultDb ) {
    static unordered_map<BiaodashiType, bool> supportedTypes = {
            {BiaodashiType::Zhenjia, true},
            {BiaodashiType::Zhengshu, true},
            {BiaodashiType::Fudianshu, true},
            {BiaodashiType::Decimal, true},
            {BiaodashiType::Zifuchuan, true},
            // {BiaodashiType::Qiufan, true},
            // {BiaodashiType::Inop, true},
            // {BiaodashiType::NotIn, true},
            // {BiaodashiType::Between, true},
            // {BiaodashiType::Case, true},
            // {BiaodashiType::Bijiao, true},
            {BiaodashiType::Yunsuan, true},
            // {BiaodashiType::Andor, true},
            // {BiaodashiType::IfCondition, true},
            // {BiaodashiType::IsNotNull, true},
            {BiaodashiType::Null, true},
    };
    if (supportedTypes.end() == supportedTypes.find(expr->GetType())) {
        string msg = "set variable value type ";
        msg.append(get_name_of_expr_type(expr->GetType()));
        ARIES_EXCEPTION(ER_NOT_SUPPORTED_YET, msg.data());
    }

    auto sps = std::make_shared<SelectPartStructure>();
    sps->AddSelectExpr(expr, nullptr);
    auto select_structure = std::make_shared<SelectStructure>();
    select_structure->init_simple_query(sps, nullptr, nullptr, nullptr, nullptr);
    BuildQuery(select_structure, defaultDb);
}

void SQLExecutor::executeKillStatement(const KillStructurePtr& killStructure, const string& defaultDbName)
{
    static unordered_map<BiaodashiType, bool> supportedTypes = {
            {BiaodashiType::Zhengshu, true},
            {BiaodashiType::Yunsuan, true},
    };
    auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(killStructure->procIdExpr);
    if (supportedTypes.end() == supportedTypes.find(expr->GetType())) {
        string msg = "processlist_id value type ";
        msg.append(get_name_of_expr_type(expr->GetType()));
        ARIES_EXCEPTION(ER_NOT_SUPPORTED_YET, msg.data());
    }

    auto sps = std::make_shared<SelectPartStructure>();
    sps->AddSelectExpr(expr, nullptr);
    auto select_structure = std::make_shared<SelectStructure>();
    select_structure->init_simple_query(sps, nullptr, nullptr, nullptr, nullptr);
    BuildQuery(select_structure, defaultDbName);

    my_thread_id thread_id= -1;
    AriesExprBridge bridge;
    AriesCommonExprUPtr ariesCommonExprUPtr = bridge.Bridge(killStructure->procIdExpr);
    auto valueType = ariesCommonExprUPtr->GetValueType().DataType.ValueType;
    switch (valueType) {
        case aries::AriesValueType::INT8: {
            int8_t value = boost::get<int8_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::INT16: {
            int16_t value = boost::get<int16_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::INT32: {
            int32_t value = boost::get<int32_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::INT64: {
            int64_t value = boost::get<int64_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::UINT8: {
            uint8_t value = boost::get<uint8_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::UINT16: {
            uint16_t value = boost::get<uint16_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::UINT32: {
            uint32_t value = boost::get<uint32_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        case aries::AriesValueType::UINT64: {
            uint64_t value = boost::get<uint64_t>(ariesCommonExprUPtr->GetContent());
            thread_id = value;
            break;
        }
        default: {
            /**
             *
             mysql> kill "123";
             ERROR 1094 (HY000): Unknown thread id: 123
             mysql> kill "abc";
             ERROR 1094 (HY000): Unknown thread id: 0
             mysql> set @id = "dfgdfg";
             Query OK, 0 rows affected (0.01 sec)
             mysql> kill @id;
             ERROR 1094 (HY000): Unknown thread id: 0
             */
            string msg = "Unknown thread id: '";
            msg.append(killStructure->procIdExpr->ToString()).append("'");
            my_message(ER_NO_SUCH_THREAD, msg.data(), 0);
            return;
        }
    }
    // if (thd->is_error())
    //     goto error;

    sql_kill(current_thd, thread_id, killStructure->killOpt & ONLY_KILL_QUERY);

}
void SQLExecutor::executeAdminStatement(const AdminStmtStructurePtr &adminStructure, const string &defaultDbName) {
    switch ( adminStructure->adminStmt ) {
        case ADMIN_STMT::KILL: {
            auto killStructure = std::dynamic_pointer_cast<KillStructure>(adminStructure);
            executeKillStatement( killStructure, defaultDbName );
            break;
        }
        case ADMIN_STMT::SHUTDOWN: {
            shutdown(current_thd, SHUTDOWN_DEFAULT, COM_QUERY );
            break;
        }
        default:
        {
            ThrowNotSupportedException( "admin statement: " + std::to_string( (int)adminStructure->adminStmt ) );
        }
    }
}
#define BOOL_VAR_VALUE_CHECK( name, value, b ) \
do \
{ \
    if ( 0 == value ) \
    { \
        b = 0; \
    } \
    else if ( 1 == value ) \
    { \
        b = 1; \
    } \
    else \
    { \
        ARIES_EXCEPTION( ER_WRONG_VALUE_FOR_VAR, name.data(), std::to_string( value ).data() ); \
    } \
} while ( 0 )

void SQLExecutor::executeSetStatements(const std::shared_ptr<std::vector<SetStructurePtr>>& setStructurePtrs, const std::string& dbName) {
    auto setStructurePtr = (*setStructurePtrs)[0];
    THD* thd = current_thd;
    for (SetStructurePtr& tmpPtr : *setStructurePtrs)
    {
        if ( SET_CMD::SET_SYS_VAR == tmpPtr->m_setCmd )
        {
            auto setVarStructurePtr = std::dynamic_pointer_cast<SetSysVarStructure>(tmpPtr);
            sys_var* sysVar = find_sys_var( setVarStructurePtr->m_sysVarStructurePtr->varName.data() );
            if (!sysVar) {
                ARIES_EXCEPTION( ER_UNKNOWN_SYSTEM_VARIABLE, setVarStructurePtr->m_sysVarStructurePtr->varName.data() );
            }
            if ( SHOW_TYPE::SHOW_MY_BOOL == sysVar->show_type() )
            {
                tmpPtr->Check( thd );
            }
        }
        else
            tmpPtr->Check( thd );
    }

    switch (setStructurePtr->m_setCmd) {
        case SET_CMD::SET_USER_VAR:
        case SET_CMD::SET_SYS_VAR: {
            for (SetStructurePtr tmpPtr : *setStructurePtrs) {
                if (SET_CMD::SET_SYS_VAR != tmpPtr->m_setCmd &&
                    SET_CMD::SET_USER_VAR != tmpPtr->m_setCmd) {
                    ThrowNotSupportedException("set statement");
                }

                AriesExprBridge bridge;
                switch (tmpPtr->m_setCmd) {
                    case SET_CMD::SET_USER_VAR: {
                        auto setVarStructurePtr = std::dynamic_pointer_cast<SetUserVarStructure>(tmpPtr);
                        CheckExprForSetStatements( setVarStructurePtr->m_valueExpr, dbName );
                        AriesCommonExprUPtr ariesCommonExprUPtr = bridge.Bridge(setVarStructurePtr->m_valueExpr);

                        user_var_entry_ptr userVarEntryPtr = user_var_entry::create(thd, setVarStructurePtr->m_userVarStructurePtr->varName);
                        userVarEntryPtr->setExpr(ariesCommonExprUPtr);
                        thd->store_user_var(userVarEntryPtr);
                        break;
                    }
                    case SET_CMD::SET_SYS_VAR: {
                        auto setVarStructurePtr = std::dynamic_pointer_cast<SetSysVarStructure>(tmpPtr);
                        string sysVarName = setVarStructurePtr->m_sysVarStructurePtr->varName;
                        aries_utils::to_lower( sysVarName );
                        sys_var* sysVar = find_sys_var( setVarStructurePtr->m_sysVarStructurePtr->varName.data() );
                        if (!sysVar) {
                            ARIES_EXCEPTION( ER_UNKNOWN_SYSTEM_VARIABLE, setVarStructurePtr->m_sysVarStructurePtr->varName.data() );
                        }
                        enum_var_type varScope = setVarStructurePtr->m_sysVarStructurePtr->varScope;
                        if( "autocommit" == sysVarName && OPT_SESSION == varScope )
                        {
                            sys_var* sysVar = find_sys_var(sysVarName.data());
                            bool old_autocommit_value = (bool)get_sys_var_value<my_bool>(sysVar, OPT_SESSION);

                            setVarStructurePtr->Update( thd );

                            bool new_autocommit_value = (bool)get_sys_var_value<my_bool>(sysVar, OPT_SESSION);

                            if ( false==old_autocommit_value && true==new_autocommit_value )
                                endTx( thd, nullptr, TransactionStatus::COMMITTED );
                        }
                        else
                        {
                            if ( SHOW_TYPE::SHOW_MY_BOOL == sysVar->show_type() )
                            {
                                setVarStructurePtr->Update( thd );
                            }
                        }
                        // string sysVarName = setVarStructurePtr->m_sysVarStructurePtr->varName;
                        // aries_utils::to_lower( sysVarName );
                        /*
                        if ( "foreign_key_checks" == sysVarName )
                        {
                            my_bool b;
                            switch ( valueType )
                            {
                                case aries::AriesValueType::BOOL: {
                                    bool value = boost::get<bool>(ariesCommonExprUPtr->GetContent());
                                    b = value;
                                    break;
                                }
                                case aries::AriesValueType::INT8: {
                                    int8_t value = boost::get<int8_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::INT16: {
                                    int16_t value = boost::get<int16_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::INT32: {
                                    int32_t value = boost::get<int32_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::INT64: {
                                    int64_t value = boost::get<int64_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::UINT8: {
                                    uint8_t value = boost::get<uint8_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::UINT16: {
                                    uint16_t value = boost::get<uint16_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::UINT32: {
                                    uint32_t value = boost::get<uint32_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                case aries::AriesValueType::UINT64: {
                                    uint64_t value = boost::get<uint64_t>(ariesCommonExprUPtr->GetContent());
                                    BOOL_VAR_VALUE_CHECK( sysVarName, value, b );
                                    break;
                                }
                                default:
                                    ARIES_EXCEPTION( ER_WRONG_VALUE_FOR_VAR,
                                                     sysVarName.data(),
                                                     commonExpr->ContentToString().data() );
                            }
                            thd->variables.foreign_key_checks = b;
                        }
                        */
                        break;
                    }
                    default:
                        break;
                }
            }
            break;
        }
        case SET_CMD::SET_NAMES: {
            LOG(INFO) << "Ignore set names command";
            break;
        }
        case SET_CMD::SET_CHAR_SET: {
            LOG(INFO) << "Ignore set charset command";
            break;
        }
        case SET_CMD::SET_TX: {
            // ThrowNotSupportedException("set transaction characteristics");
            // mysql workbench need ok response for this command
            LOG(INFO) << "Ignore set session command";
            break;
        }
        case SET_CMD::SET_PASSWORD:
        {
            auto setPasswordStructurePtr = std::dynamic_pointer_cast< SetPasswordStructure >( setStructurePtr );
            CommandExecutor cmd_executor;
            cmd_executor.ExecuteSetPassword( setPasswordStructurePtr->user,
                                             setPasswordStructurePtr->host,
                                             setPasswordStructurePtr->password );
            break;
        }
        default:
            break;
    }
    my_ok(thd);
}

/*
 * mysql> set @a=1;
Query OK, 0 rows affected (0.00 sec)

mysql> prepare s1 from @a;
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '1' at line 1

 * mysql> set @a="";
Query OK, 0 rows affected (0.00 sec)

mysql> prepare s1 from @a;
ERROR 1065 (42000): Query was empty

 mysql> set @a="a";
Query OK, 0 rows affected (0.00 sec)

mysql> prepare s1 from @a;
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'a' at line 1

 mysql> prepare s1 from @nonexist;
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use
 near 'NULL' at line 1

 mysql> select @aaa;
+------+
| @aaa |
+------+
| NULL |
+------+

 * */
void SQLExecutor::executePreparedStatement(const PreparedStmtStructurePtr preparedStmtStructurePtr, const std::string& dbName) {
    THD* thd = current_thd;
    switch (preparedStmtStructurePtr->stmtCmd) {
        case PREPARED_STMT_CMD::PREPARE: {
            mysql_sql_stmt_prepare(thd, preparedStmtStructurePtr);
            break;
        }
        case PREPARED_STMT_CMD::EXECUTE: {
            mysql_sql_stmt_execute(thd, preparedStmtStructurePtr);
            break;
        }
        case PREPARED_STMT_CMD::DEALLOCATE: {
            mysql_sql_stmt_close(thd, preparedStmtStructurePtr);
            break;
        }
    }
}
SQLResultPtr showCreateDb(const string& dbName) {
    THD *thd = current_thd;
    auto dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    vector<Send_field *> fields;
    Send_field field;
    field.col_name = "Database";
    field.org_col_name = field.col_name;
    field.type = MYSQL_TYPE_VAR_STRING;
    field.length = 192;
    field.flags = 0x0001;
    field.decimals = 0;
    fields.insert(fields.end(), &field);

    Send_field field2;
    field2.col_name = "Create Database";
    field2.org_col_name = field.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 3072;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    thd->get_protocol_classic()->start_row();
    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store(dbName.data(), default_charset_info);
    thd->get_protocol_classic()->store(dbEntry->GetCreateString().data(),
                                       dbEntry->GetCreateString().size(),
                                       default_charset_info);
    if ( !thd->get_protocol_classic()->end_row() )
        my_eof(thd);
end:
    return nullptr;
}
SQLResultPtr showCreateTable(const string& dbName, const string& tableName) {
    THD* thd = current_thd;
    auto dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry )
    {
        ARIES_EXCEPTION(ER_BAD_DB_ERROR, dbName.data());
    }
    auto tableEntry = dbEntry->GetTableByName(tableName);
    if (!tableEntry) {
        ARIES_EXCEPTION(ER_BAD_TABLE_ERROR, tableName.data());
    }
    vector<Send_field *> fields;
    Send_field field;
    field.col_name = "Table";
    field.org_col_name = field.col_name;
    field.type = MYSQL_TYPE_VAR_STRING;
    field.length = 256;
    field.flags = 0x0001;
    field.decimals = 0;
    fields.insert(fields.end(), &field);

    Send_field field2;
    field2.col_name = "Create Table";
    field2.org_col_name = field.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 256;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    thd->get_protocol_classic()->start_row();
    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store(tableEntry->GetName().data(),
                                       tableEntry->GetName().size(),
                                       default_charset_info);
    thd->get_protocol_classic()->store(tableEntry->GetCreateString().data(),
                                       tableEntry->GetCreateString().size(),
                                       default_charset_info);
    if ( !thd->get_protocol_classic()->end_row() )
        my_eof(thd);

end:
    return nullptr;
}
SQLResultPtr showOpenTables(const string& dbName, const string& wild, const BiaodashiPointer& whereExpr) {
    THD *thd = current_thd;
    shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    vector<Send_field *> fields;
    Send_field field;
    field.db_name = "information_schema";
    field.table_name = field.org_table_name = "OPEN_TABLES";
    field.col_name = "Database";
    field.org_col_name = "Database";
    field.type = MYSQL_TYPE_VAR_STRING;
    field.length = 192;
    field.flags = 0x0001;
    field.decimals = 0;
    fields.insert(fields.end(), &field);

    Send_field field2;
    field2.db_name = "information_schema";
    field2.table_name = field2.org_table_name = "OPEN_TABLES";
    field2.col_name = "Table";
    field2.org_col_name = "Table";
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 192;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.db_name = "information_schema";
    field3.table_name = field3.org_table_name = "OPEN_TABLES";
    field3.col_name = "In_use";
    field3.org_col_name = "In_use";
    field3.type = MYSQL_TYPE_LONGLONG;
    field3.length = 1;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    Send_field field5;
    field5.db_name = "information_schema";
    field5.table_name = field5.org_table_name = "OPEN_TABLES";
    field5.col_name = "Name_locked";
    field5.org_col_name = "Name_locked";
    field5.type = MYSQL_TYPE_LONGLONG;
    field5.length = 4;
    field5.flags = 0x0001;
    field5.decimals = 0;
    fields.insert(fields.end(), &field5);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;
}
SQLResultPtr showTableStatus(const string& dbName, const string& wild, const BiaodashiPointer& whereExpr) {
    THD *thd = current_thd;
    shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }

    static const char* sqlFormat = "select * from information_schema.tables where table_schema = ";
    string sql = sqlFormat;
    sql.append("'").append(dbName).append("'");

    SQLResultPtr sqlResultPtr = SQLExecutor::GetInstance()->executeSelect(sql, "information_schema");
    if (!sqlResultPtr->IsSuccess())
    {
        ARIES_EXCEPTION(ER_BAD_DB_ERROR, dbName.data());
    }

    thd->get_protocol_classic()->start_row();
    vector<Send_field *> fields;
    Send_field field;
    field.db_name = "information_schema";
    field.table_name = field.org_table_name = "TABLES";
    field.col_name = "Name";
    field.org_col_name = "TABLE_NAME";
    field.type = MYSQL_TYPE_VAR_STRING;
    field.length = 256;
    field.flags = 0x0001;
    field.decimals = 0;
    fields.insert(fields.end(), &field);

    Send_field field2;
    field2.db_name = "information_schema";
    field2.table_name = field.org_table_name = "TABLES";
    field2.col_name = "Engine";
    field2.org_col_name = "ENGINE";
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 57;
    field2.flags = 0;
    field2.decimals = 0x1f;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.db_name = "information_schema";
    field3.table_name = field3.org_table_name = "TABLES";
    field3.col_name = "Version";
    field3.org_col_name = "VERSION";
    field3.type = MYSQL_TYPE_LONGLONG;
    field3.length = 21;
    field3.flags = 0x20;
    field3.decimals = 0x0;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.db_name = "information_schema";
    field4.table_name = field4.org_table_name = "TABLES";
    field4.col_name = "Row_format";
    field4.org_col_name = "ROW_FORMAT";
    field4.type = MYSQL_TYPE_VAR_STRING;
    field4.length = 57;
    field4.flags = 0;
    field4.decimals = 0x0;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.db_name = "information_schema";
    field5.table_name = field5.org_table_name = "TABLES";
    field5.col_name = "Rows";
    field5.org_col_name = "TABLE_ROWS";
    field5.type = MYSQL_TYPE_LONGLONG;
    field5.length = 21;
    field5.flags = 0x20;
    field5.decimals = 0x0;
    fields.insert(fields.end(), &field5);

    Send_field field6;
    field6.db_name = "information_schema";
    field6.table_name = field6.org_table_name = "TABLES";
    field6.col_name = "Avg_row_length";
    field6.org_col_name = "AVG_ROW_LENGTH";
    field6.type = MYSQL_TYPE_LONGLONG;
    field6.length = 21;
    field6.flags = 0x20;
    field6.decimals = 0x0;
    fields.insert(fields.end(), &field6);

    Send_field field7;
    field7.db_name = "information_schema";
    field7.table_name = field7.org_table_name = "TABLES";
    field7.col_name = "Data_length";
    field7.org_col_name = "DATA_LENGTH";
    field7.type = MYSQL_TYPE_LONGLONG;
    field7.length = 21;
    field7.flags = 0x20;
    field7.decimals = 0x0;
    fields.insert(fields.end(), &field7);

    Send_field field8;
    field8.db_name = "information_schema";
    field8.table_name = field8.org_table_name = "TABLES";
    field8.col_name = "Max_data_length";
    field8.org_col_name = "MAX_DATA_LENGTH";
    field8.type = MYSQL_TYPE_LONGLONG;
    field8.length = 21;
    field8.flags = 0x20;
    field8.decimals = 0x0;
    fields.insert(fields.end(), &field8);

    Send_field field9;
    field9.db_name = "information_schema";
    field9.table_name = field9.org_table_name = "TABLES";
    field9.col_name = "Index_length";
    field9.org_col_name = "INDEX_LENGTH";
    field9.type = MYSQL_TYPE_LONGLONG;
    field9.length = 21;
    field9.flags = 0x20;
    field9.decimals = 0x0;
    fields.insert(fields.end(), &field9);

    Send_field field10;
    field10.db_name = "information_schema";
    field10.table_name = field10.org_table_name = "TABLES";
    field10.col_name = "Data_free";
    field10.org_col_name = "DATA_FREE";
    field10.type = MYSQL_TYPE_LONGLONG;
    field10.length = 21;
    field10.flags = 0x20;
    field10.decimals = 0x0;
    fields.insert(fields.end(), &field10);

    Send_field field11;
    field11.db_name = "information_schema";
    field11.table_name = field11.org_table_name = "TABLES";
    field11.col_name = "Auto_increment";
    field11.org_col_name = "AUTO_INCREMENT";
    field11.type = MYSQL_TYPE_LONGLONG;
    field11.length = 21;
    field11.flags = 0x20;
    field11.decimals = 0x0;
    fields.insert(fields.end(), &field11);

    Send_field field12;
    field12.db_name = "information_schema";
    field12.table_name = field12.org_table_name = "TABLES";
    field12.col_name = "Create_time";
    field12.org_col_name = "CREATE_TIME";
    field12.type = MYSQL_TYPE_DATETIME;
    field12.length = 19;
    field12.flags = 0x80;
    field12.decimals = 0x0;
    fields.insert(fields.end(), &field12);

    Send_field field13;
    field13.db_name = "information_schema";
    field13.table_name = field13.org_table_name = "TABLES";
    field13.col_name = "Update_time";
    field13.org_col_name = "UPDATE_TIME";
    field13.type = MYSQL_TYPE_DATETIME;
    field13.length = 19;
    field13.flags = 0x80;
    field13.decimals = 0x0;
    fields.insert(fields.end(), &field13);

    Send_field field14;
    field14.db_name = "information_schema";
    field14.table_name = field14.org_table_name = "TABLES";
    field14.col_name = "Check_time";
    field14.org_col_name = "Check_time";
    field14.type = MYSQL_TYPE_DATETIME;
    field14.length = 19;
    field14.flags = 0x80;
    field14.decimals = 0x0;
    fields.insert(fields.end(), &field14);

    Send_field field15;
    field15.db_name = "information_schema";
    field15.table_name = field15.org_table_name = "TABLES";
    field15.col_name = "Collation";
    field15.org_col_name = "COLLATION";
    field15.type = MYSQL_TYPE_VAR_STRING;
    field15.length = 128;
    field15.flags = 0x00;
    field15.decimals = 0x0;
    fields.insert(fields.end(), &field15);

    Send_field field16;
    field16.db_name = "information_schema";
    field16.table_name = field16.org_table_name = "TABLES";
    field16.col_name = "Checksum";
    field16.org_col_name = "CHECKSUM";
    field16.type = MYSQL_TYPE_LONGLONG;
    field16.length = 21;
    field16.flags = 0x20;
    field16.decimals = 0x0;
    fields.insert(fields.end(), &field16);

    Send_field field17;
    field17.db_name = "information_schema";
    field17.table_name = field17.org_table_name = "TABLES";
    field17.col_name = "Create_options";
    field17.org_col_name = "CREATE_OPTIONS";
    field17.type = MYSQL_TYPE_VAR_STRING;
    field17.length = 1020;
    field17.flags = 0x00;
    field17.decimals = 0x0;
    fields.insert(fields.end(), &field17);

    Send_field field18;
    field18.db_name = "information_schema";
    field18.table_name = field18.org_table_name = "TABLES";
    field18.col_name = "Comment";
    field18.org_col_name = "COMMENT";
    field18.type = MYSQL_TYPE_VAR_STRING;
    field18.length = 8192;
    field18.flags = 0x00;
    field18.decimals = 0x1;
    fields.insert(fields.end(), &field17);

    auto results = sqlResultPtr->GetResults();
    auto amtp = results[0];
    auto block = ( ( AriesMemTable * )amtp.get() )->GetContent();

    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;
    /**
mysql> desc tables;
+-----------------+--------------+------+-----+---------+-------+
| Field           | Type         | Null | Key | Default | Extra |
+-----------------+--------------+------+-----+---------+-------+
| table_catalog   | varchar(256) | NO   |     | NULL    |       | 1
| table_schema    | varchar(256) | NO   |     | NULL    |       | 2
| table_name      | varchar(256) | NO   |     | NULL    |       | 3
| table_type      | varchar(256) | NO   |     | NULL    |       | 4
| engine          | varchar(256) | YES  |     | NULL    |       | 5
| version         | int(11)      | YES  |     | NULL    |       | 6
| row_format      | varchar(256) | YES  |     | NULL    |       | 7
| table_rows      | int(11)      | YES  |     | NULL    |       | 8
| avg_row_length  | int(11)      | YES  |     | NULL    |       | 9
| data_length     | int(11)      | YES  |     | NULL    |       | 10
| max_data_length | int(11)      | YES  |     | NULL    |       | 11
| index_length    | int(11)      | YES  |     | NULL    |       | 12
| data_free       | int(11)      | YES  |     | NULL    |       | 13
| auto_increment  | int(11)      | YES  |     | NULL    |       | 14
| create_time     | date_time    | YES  |     | NULL    |       | 15
| update_time     | date_time    | YES  |     | NULL    |       | 16
| check_time      | date_time    | YES  |     | NULL    |       | 17
| table_collation | varchar(256) | YES  |     | NULL    |       | 18
| checksum        | int(11)      | YES  |     | NULL    |       | 19
| create_options  | varchar(256) | YES  |     | NULL    |       | 20
| table_comment   | varchar(256) | NO   |     | NULL    |       | 21
+-----------------+--------------+------+-----+---------+-------+
     */
    for (int tid = 0; tid < block->GetRowCount(); tid++) {
        thd->get_protocol_classic()->start_row();

        AriesDataBufferSPtr column = block->GetColumnBuffer(4);
        thd->get_protocol_classic()->store(column->GetString(tid).data(), default_charset_info); // table name

        column = block->GetColumnBuffer(6);
        thd->get_protocol_classic()->store(column->GetString(tid).data(), default_charset_info); // engine

        column = block->GetColumnBuffer(7);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // version

        column = block->GetColumnBuffer(8);
        thd->get_protocol_classic()->store(column->GetString(tid).data(), default_charset_info); // row_format

        column = block->GetColumnBuffer(9);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // rows

        column = block->GetColumnBuffer(10);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // avg_row_len

        column = block->GetColumnBuffer(11);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // data_len

        column = block->GetColumnBuffer(12);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // max_data_len

        column = block->GetColumnBuffer(13);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // index_len

        column = block->GetColumnBuffer(14);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // date_free

        column = block->GetColumnBuffer(15);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // auto_incr

        column = block->GetColumnBuffer(16);
        thd->get_protocol_classic()->store(column->GetDatetimeAsString( tid ).data(), default_charset_info); // create time

        column = block->GetColumnBuffer(17);
        thd->get_protocol_classic()->store(column->GetDatetimeAsString( tid ).data(), default_charset_info); // update time

        column = block->GetColumnBuffer(18);
        thd->get_protocol_classic()->store(column->GetDatetimeAsString( tid ).data(), default_charset_info); // check time

        column = block->GetColumnBuffer(19);
        thd->get_protocol_classic()->store(column->GetString(tid).data(), default_charset_info); // collation

        column = block->GetColumnBuffer(20);
        thd->get_protocol_classic()->store_longlong(column->GetUint64(tid), true); // checksum

        column = block->GetColumnBuffer(21);
        thd->get_protocol_classic()->store(column->GetString(tid).data(), default_charset_info); // create_options

        column = block->GetColumnBuffer(22);
        thd->get_protocol_classic()->store(column->GetString(tid).data(), default_charset_info); // comments
        if ( thd->get_protocol_classic()->end_row() )
            goto end;
    }
    my_eof(thd);

end:
    return nullptr;
}

SQLResultPtr showTriggers(const std::string& dbName, const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr) {
    static unordered_map<string, string> aliasMap {
            {"trigger", "TRIGGER_NAME"},
            {"event", "EVENT_MANIPULATION"},
            {"table", "EVENT_OBJECT_TABLE"},
            {"statement", "ACTION_STATEMENT"},
            {"timing", "ACTION_TIMING"},
            {"created", "CREATED"},
            {"sql_mode", "SQL_MODE"},
            {"definer", "DEFINER"},
            {"character_set_client", "CHARACTER_SET_CLIENT"},
            {"collation_connection", "COLLATION_CONNECTION"},
            {"database collation", "DATABASE_COLLATION"},
    };
    THD *thd = current_thd;
    shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    string sql = R"(SELECT TRIGGER_NAME as `Trigger`, EVENT_MANIPULATION as Event,
                    EVENT_OBJECT_TABLE as `Table`, ACTION_STATEMENT as Statement,
                    ACTION_TIMING as Timing, CREATED as Created,
                    SQL_MODE as sql_mode, DEFINER as Definer,
                    CHARACTER_SET_CLIENT as character_set_client,
                    COLLATION_CONNECTION as collation_connection,
                    DATABASE_COLLATION as `Database Collation`
                    FROM information_schema.TRIGGERS)";
    if (!wild.empty()) {
        sql.append(" WHERE TRIGGER_NAME").append(" LIKE ").append(exprStr);
    } else if (whereExpr) {
        sql.append(" WHERE ").append(ShowWhereExprToSelectExpr(whereExpr, aliasMap));
    }
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());

}
SQLResultPtr showEvents(const std::string& dbName, const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr) {
    THD *thd = current_thd;
    shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    string sql = R"(SELECT EVENT_SCHEMA as Db, EVENT_NAME as Name, DEFINER as Definer,
                    TIME_ZONE as `Time zone`, EVENT_TYPE as Type, EXECUTE_AT as `Execute at`,
                    INTERVAL_VALUE as `Interval value`, INTERVAL_FIELD as `Interval field`,
                    STARTS as Starts, ENDS as Ends, STATUS as Status, ORIGINATOR as Originator,
                    CHARACTER_SET_CLIENT as character_set_client, COLLATION_CONNECTION as collation_connection,
                    DATABASE_COLLATION as `Database Collation` from information_schema.EVENTS)";
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());

}

SQLResultPtr showTables(const std::string& dbName, bool full)
{
    THD* thd = current_thd;
    shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if (nullptr == dbEntry )
    {
        ARIES_EXCEPTION(ER_BAD_DB_ERROR, dbName.data());
    }

    vector<Send_field *> fields;
    Send_field field;
    field.db_name = "information_schema";
    field.table_name = field.org_table_name = "TABLE_NAMES";
    std::string colName("Tables_in_");
    colName.append(dbName);
    field.col_name = colName;

    field.org_col_name = field.col_name;
    field.type = MYSQL_TYPE_VAR_STRING;
    field.length = 256;
    field.flags = 0x0001;
    field.decimals = 0;
    fields.insert(fields.end(), &field);

    Send_field field2;
    if (full)
    {
        field2.db_name = "information_schema";
        field2.table_name = field2.org_table_name = "TABLE_NAMES";
        std::string colName("Table_type");
        colName.append(dbName);
        field2.col_name = colName;

        field2.org_col_name = field2.col_name;
        field2.type = MYSQL_TYPE_VAR_STRING;
        field2.length = 256;
        field2.flags = 0x0001;
        field2.decimals = 0;
        fields.insert(fields.end(), &field2);
    }

    std::vector<string> tables = dbEntry->GetNameListOfTables();
    auto view_nodes = ViewManager::GetInstance().GetViewNodes(dbName);

    thd->get_protocol_classic()->start_row();
    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    for (auto it = tables.begin(); it != tables.end(); it++)
    {
        thd->get_protocol_classic()->start_row();
        thd->get_protocol_classic()->store(it->c_str(), it->size(), default_charset_info);
        if (full)
        {
            thd->get_protocol_classic()->store("BASE TABLE", default_charset_info);
        }
        if ( thd->get_protocol_classic()->end_row() )
            goto end;
    }

    for (const auto& node : view_nodes) {
        thd->get_protocol_classic()->start_row();
        thd->get_protocol_classic()->store(node->GetBasicRel()->GetID().data(), node->GetBasicRel()->GetID().size(), default_charset_info);
        if (full)
        {
            thd->get_protocol_classic()->store("VIEW", default_charset_info);
        }
        if ( thd->get_protocol_classic()->end_row() )
            goto end;
    }

    my_eof(thd);

end:
    return nullptr;
}

SQLResultPtr showCollation(const string& wild, const BiaodashiPointer& whereExpr, const string& exprStr) {
    static unordered_map<string, string> aliasMap {
            {"collation", "COLLATION_NAME"},
            {"charset", "CHARACTER_SET_NAME"},
            {"id", "ID"},
            {"default", "IS_DEFAULT"},
            {"compiled", "IS_COMPILED"},
            {"Sortlen", "SORTLEN"}
    };
    string sql = R"(SELECT COLLATION_NAME as collation,
                    CHARACTER_SET_NAME as charset, ID as id,
                    IS_DEFAULT as `default`, IS_COMPILED as compiled, SORTLEN as sortlen
                    from information_schema.collations)";
    if (!wild.empty()) {
        sql.append(" WHERE collation_name").append(" LIKE ").append(exprStr);
    } else if (whereExpr) {
        sql.append(" WHERE ").append(ShowWhereExprToSelectExpr(whereExpr, aliasMap));
    }
    THD* thd = current_thd;
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());

}
SQLResultPtr showEngines() {
    THD* thd = current_thd;
    thd->get_protocol_classic()->start_row();
    vector<Send_field *> fields;
    Send_field field1;
    field1.db_name = "information_schema";
    field1.table_name = field1.org_table_name = "ENGINES";
    field1.col_name = "Engine";
    field1.org_col_name = field1.col_name;
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 57;
    field1.flags = 0;
    field1.decimals = 0x1f;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.db_name = "information_schema";
    field2.table_name = field2.org_table_name = "ENGINES";
    field2.col_name = "Support";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 57;
    field2.flags = 0;
    field2.decimals = 0x1f;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.db_name = "information_schema";
    field3.table_name = field3.org_table_name = "ENGINES";
    field3.col_name = "Comment";
    field3.org_col_name = field3.col_name;
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 57;
    field3.flags = 0;
    field3.decimals = 0x1f;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.db_name = "information_schema";
    field4.table_name = field4.org_table_name = "ENGINES";
    field4.col_name = "Transactions";
    field4.org_col_name = field4.col_name;
    field4.type = MYSQL_TYPE_VAR_STRING;
    field4.length = 57;
    field4.flags = 0;
    field4.decimals = 0x1f;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.db_name = "information_schema";
    field5.table_name = field5.org_table_name = "ENGINES";
    field5.col_name = "XA";
    field5.org_col_name = field5.col_name;
    field5.type = MYSQL_TYPE_VAR_STRING;
    field5.length = 57;
    field5.flags = 0;
    field5.decimals = 0x1f;
    fields.insert(fields.end(), &field5);

    Send_field field6;
    field6.db_name = "information_schema";
    field6.table_name = field6.org_table_name = "ENGINES";
    field6.col_name = "Savepoints";
    field6.org_col_name = field6.col_name;
    field6.type = MYSQL_TYPE_VAR_STRING;
    field6.length = 57;
    field6.flags = 0;
    field6.decimals = 0x1f;
    fields.insert(fields.end(), &field6);
    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store("Aries", default_charset_info);
    thd->get_protocol_classic()->store("YES", default_charset_info);
    thd->get_protocol_classic()->store("aries_storage_engine", default_charset_info);
    thd->get_protocol_classic()->store("NO", default_charset_info);
    thd->get_protocol_classic()->store("NO", default_charset_info);
    thd->get_protocol_classic()->store("NO", default_charset_info);
    if ( thd->get_protocol_classic()->end_row() )
        goto end;

    my_eof(thd);

end:
    return nullptr;
}

SQLResultPtr showEngineStatus(const string& name) {
    if ("aries" != name) {
        ARIES_EXCEPTION( ER_UNKNOWN_STORAGE_ENGINE, name.data() );
    }
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field1;
    field1.col_name = "Type";
    field1.org_col_name = field1.col_name;
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 30;
    field1.flags = 0x0001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "Name";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 1536;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Status";
    field3.org_col_name = field3.col_name;
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 30;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    thd->get_protocol_classic()->start_row();
    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store(name.data(), default_charset_info);
    thd->get_protocol_classic()->store("", default_charset_info);
    thd->get_protocol_classic()->store("", default_charset_info);
    if ( !thd->get_protocol_classic()->end_row() )
        my_eof(thd);

end:
    return nullptr;
}
SQLResultPtr showEngineMutex(const string& name) {
    if ("aries" != name) {
        ARIES_EXCEPTION( ER_UNKNOWN_STORAGE_ENGINE, name.data() );
    }
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field1;
    field1.col_name = "Type";
    field1.org_col_name = field1.col_name;
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 30;
    field1.flags = 0x0001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "Name";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 1536;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Status";
    field3.org_col_name = field3.col_name;
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 30;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    thd->get_protocol_classic()->start_row();
    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    thd->get_protocol_classic()->start_row();
    thd->get_protocol_classic()->store(name.data(), default_charset_info);
    thd->get_protocol_classic()->store("", default_charset_info);
    thd->get_protocol_classic()->store("", default_charset_info);
    if ( !thd->get_protocol_classic()->end_row() )
        my_eof(thd);

end:
    return nullptr;
}
SQLResultPtr showErrors(const LimitStructurePointer& limit) {
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field1;
    field1.col_name = "Level";
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 21;
    field1.flags = 0x1001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "Code";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_LONG;
    field2.length = 4;
    field2.flags = 0x00a1;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Message";
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 1536;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;
}
SQLResultPtr showWarnings(const LimitStructurePointer& limit) {
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field1;
    field1.col_name = "Level";
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 21;
    field1.flags = 0x1001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.col_name = "Code";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_LONG;
    field2.length = 4;
    field2.flags = 0x00a1;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.col_name = "Message";
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 1536;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);
    return nullptr;
}

SQLResultPtr showPlugins() {
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field1;
    field1.db_name = "information_schema";
    field1.table_name = field1.org_table_name = "PLUGINS";
    field1.col_name = "Name";
    field1.org_col_name = field1.col_name;
    field1.type = MYSQL_TYPE_VAR_STRING;
    field1.length = 256;
    field1.flags = 0x1001;
    field1.decimals = 0;
    fields.insert(fields.end(), &field1);

    Send_field field2;
    field2.db_name = "information_schema";
    field2.table_name = field2.org_table_name = "PLUGINS";
    field2.col_name = "Status";
    field2.org_col_name = field2.col_name;
    field2.type = MYSQL_TYPE_VAR_STRING;
    field2.length = 40;
    field2.flags = 0x0001;
    field2.decimals = 0;
    fields.insert(fields.end(), &field2);

    Send_field field3;
    field3.db_name = "information_schema";
    field3.table_name = field3.org_table_name = "PLUGINS";
    field3.col_name = "Type";
    field3.org_col_name = field3.col_name;
    field3.type = MYSQL_TYPE_VAR_STRING;
    field3.length = 40;
    field3.flags = 0x0001;
    field3.decimals = 0;
    fields.insert(fields.end(), &field3);

    Send_field field4;
    field4.db_name = "information_schema";
    field4.table_name = field4.org_table_name = "PLUGINS";
    field4.col_name = "Library";
    field4.org_col_name = field4.col_name;
    field4.type = MYSQL_TYPE_VAR_STRING;
    field4.length = 256;
    field4.flags = 0x0000;
    field4.decimals = 0;
    fields.insert(fields.end(), &field4);

    Send_field field5;
    field5.db_name = "information_schema";
    field5.table_name = field5.org_table_name = "PLUGINS";
    field5.col_name = "License";
    field5.org_col_name = field5.col_name;
    field5.type = MYSQL_TYPE_VAR_STRING;
    field5.length = 320;
    field5.flags = 0x0000;
    field5.decimals = 0;
    fields.insert(fields.end(), &field5);

    thd->get_protocol_classic()->start_row();
    if ( !thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        my_eof(thd);

    // thd->get_protocol_classic()->start_row();
    // thd->get_protocol_classic()->store("mysql_native_password", default_charset_info);
    // thd->get_protocol_classic()->store("ACTIVE", default_charset_info);
    // thd->get_protocol_classic()->store("AUTHENTICATION", default_charset_info);
    // thd->get_protocol_classic()->store("NULL", default_charset_info);
    // thd->get_protocol_classic()->store("GPL", default_charset_info);
    // thd->get_protocol_classic()->end_row();

    // thd->get_protocol_classic()->start_row();
    // thd->get_protocol_classic()->store("CSV", default_charset_info);
    // thd->get_protocol_classic()->store("ACTIVE", default_charset_info);
    // thd->get_protocol_classic()->store("STORAGE_ENGINE", default_charset_info);
    // thd->get_protocol_classic()->store("NULL", default_charset_info);
    // thd->get_protocol_classic()->store("GPL", default_charset_info);
    // thd->get_protocol_classic()->end_row();

    return nullptr;
}

SQLResultPtr showColumns(const ShowColumnsStructurePtr& showColumnsStructurePointer, const string& dbName)
{
    THD* thd = current_thd;
    string db = showColumnsStructurePointer->tableNameStructurePtr->dbName;
    string table = showColumnsStructurePointer->tableNameStructurePtr->tableName;
    if (db.empty()) {
        db = dbName;
        if (db.empty()) {
            ARIES_EXCEPTION(ER_NO_DB_ERROR);
        }
    }
    shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(db);
    if (nullptr == dbEntry )
    {
        ARIES_EXCEPTION(ER_BAD_DB_ERROR, db.data());
    }
     aries_utils::to_lower(table);
    shared_ptr<TableEntry> tableEntry = dbEntry->GetTableByName(table);
    if ( !tableEntry )
    {
        auto view = aries::ViewManager::GetInstance().GetViewNode( table, db );
        if ( !view )
            ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, table.data() );
    }
//
//     /**
// mysql> create table t_int2 (i1 tinyint, i2 tinyint unsigned, i3 smallint, i4 smallint unsigned, i5 mediumint, i6 mediumint unsigned, i7 int, i8 int unsigned, i9 bigint, i10 bigint unsigned);
// Query OK, 0 rows affected (0.02 sec)
//
// mysql> desc t_int2;
// +-------+-----------------------+------+-----+---------+-------+
// | Field | Type                  | Null | Key | Default | Extra |
// +-------+-----------------------+------+-----+---------+-------+
// | i1    | tinyint(4)            | YES  |     | NULL    |       |
// | i2    | tinyint(3) unsigned   | YES  |     | NULL    |       |
// | i3    | smallint(6)           | YES  |     | NULL    |       |
// | i4    | smallint(5) unsigned  | YES  |     | NULL    |       |
// | i5    | mediumint(9)          | YES  |     | NULL    |       |
// | i6    | mediumint(8) unsigned | YES  |     | NULL    |       |
// | i7    | int(11)               | YES  |     | NULL    |       |
// | i8    | int(10) unsigned      | YES  |     | NULL    |       |
// | i9    | bigint(20)            | YES  |     | NULL    |       |
// | i10   | bigint(20) unsigned   | YES  |     | NULL    |       |
// +-------+-----------------------+------+-----+---------+-------+
// 10 rows in set (0.00 sec)
//
// mysql> create table t_decimal(a float, b double, c decimal, d decimal(20, 6));
// Query OK, 0 rows affected (0.07 sec)
//
// mysql> desc t_decimal;
// +-------+---------------+------+-----+---------+-------+
// | Field | Type          | Null | Key | Default | Extra |
// +-------+---------------+------+-----+---------+-------+
// | a     | float         | YES  |     | NULL    |       |
// | b     | double        | YES  |     | NULL    |       |
// | c     | decimal(10,0) | YES  |     | NULL    |       |
// | d     | decimal(20,6) | YES  |     | NULL    |       |
// +-------+---------------+------+-----+---------+-------+
// 4 rows in set (0.00 sec)
    static unordered_map<string, string> aliasMap {
            {"Field", "COLUMN_NAME"},
            {"Type", "COLUMN_TYPE"},
            // if full, collation
            {"Collation", "COLLATION_NAME"},
            {"Nullable", "IS_NULLABLE"},
            {"Keytype", "COLUMN_KEY"},
            {"DefaultValue", "COLUMN_DEFAULT"},
            {"extra", "EXTRA"},
            // if full, privileges and comments
            {"Privileges", "PRIVILEGES"},
            {"comment", "COLUMN_COMMENT"},
    };
    string sql = R"(SELECT COLUMN_NAME as Field,
                    COLUMN_TYPE as Type,)";
    if ( showColumnsStructurePointer->full )
    {
        sql.append("COLLATION_NAME as Collation,");
    }
    sql.append(R"(IS_NULLABLE as Nullable,
                  COLUMN_KEY as Keytype,
                  COLUMN_DEFAULT as DefaultValue,
                  EXTRA as extra)");
    if ( showColumnsStructurePointer->full )
    {
        sql.append(",PRIVILEGES as Privileges,");
        sql.append("COLUMN_COMMENT as comment");
    }

    sql.append( " from information_schema.columns" );
    string condition = " WHERE TABLE_SCHEMA = '";
    condition.append(db).append("'");
    condition.append( " AND" ).append(" TABLE_NAME = '").append(table).append("'");

    if (!showColumnsStructurePointer->wild.empty()) {
        condition.append(" AND column_name").append(" LIKE ").append(showColumnsStructurePointer->wildOrWhereStr);
    } else if (showColumnsStructurePointer->where) {
        condition.append(" AND ").append(ShowWhereExprToSelectExpr(showColumnsStructurePointer->where, aliasMap));
    }
    sql.append(condition);
    sql.append( " ORDER BY ORDINAL_POSITION ASC" );
    return aries::SQLExecutor::GetInstance()->executeSelect(sql, thd->db());
}

SQLResultPtr showDatabases()
{
    THD* thd = current_thd;
    vector<Send_field *> fields;
    Send_field field;
    field.db_name = "information_schema";
    field.table_name = field.org_table_name = "SCHEMATA";
    field.col_name = "Database";
    field.org_col_name = field.col_name;
    field.type = MYSQL_TYPE_VAR_STRING;
    field.length = 57;
    field.flags = 0;
    field.decimals = 0x1f;
    fields.insert(fields.end(), &field);
    thd->get_protocol_classic()->start_row();

    const std::map<std::string, std::shared_ptr<DatabaseEntry>> &databases = SchemaManager::GetInstance()->GetSchema()->GetDatabases();

    if ( thd->send_result_metadata(&fields, Protocol_classic::SEND_NUM_ROWS | Protocol_classic::SEND_EOF) )
        goto end;

    for (auto it = databases.begin(); it != databases.end(); it++)
    {
        thd->get_protocol_classic()->start_row();
        string dbName = it->second->GetName().c_str();
        thd->get_protocol_classic()->store(dbName.c_str(), dbName.size(), default_charset_info);
        if ( thd->get_protocol_classic()->end_row() )
            goto end;
    }
    my_eof(thd);

end:
    return nullptr;
}

std::pair<int, string> SQLExecutor::ParseSQL(const std::string sql,
                                             bool file,
                                             std::vector<AriesSQLStatementPointer>& statements)
{
    auto result = std::make_pair<int, string>(0, "");
    try {
        THD* thd = current_thd;
        if( thd->m_tx && thd->m_tx->GetMyDbVersion() != AriesVacuum::GetInstance().GetCurrentDbVersion() )
        {
            //if user begin transaction explictly, the user may send a command after a vacuum process.
            //the transaction need abort.
            endTx( thd, nullptr, TransactionStatus::ABORTED );
            ARIES_EXCEPTION( ER_XA_RBROLLBACK );
        }

        SQLParserPortal parser;
        if ( file )
        {
            statements = parser.ParseSQLFile4Statements( sql );
        }
        else
        {
            statements = parser.ParseSQLString4Statements( sql );
        }
    } catch (const AriesFakeImplException& e) {
        result.first = e.errCode;
        switch ( e.errCode )
        {
            case ER_FAKE_IMPL_OK:
                my_ok( current_thd );
                break;
            case ER_FAKE_IMPL_EOF:
                my_eof( current_thd );
                break;
            default:
                result.second = e.errMsg;
                break;
        }
        LOG(INFO) << "Fake implementation: " << e.errMsg;
    } catch (const AriesException& e) {
        result.first = e.errCode;
        result.second = e.errMsg;
    } catch (const std::exception& e) {
        result.first = ER_PARSE_ERROR;
        result.second.append("Parsing SQL failed, ").append( e.what() );
        LOG(ERROR) << result.second;
    } catch (...) {
        result.first = ER_PARSE_ERROR;
        result.second = "Parsing SQL failed, unknown exception";
        LOG(ERROR) << result.second;
    }
    return result;
}

SQLResultPtr SQLExecutor::ExecuteSQL(const string& sql, const string& defaultDbName, bool sendResp ) {
    LOG(INFO) << "Executing sql: " << sql << ", default database: " << defaultDbName;
    auto result = std::make_shared<SQLResult>();
    std::vector<AriesSQLStatementPointer> statements;
#ifdef ARIES_PROFILE
    long parseTime;
    long execStmtTime;
    aries::CPU_Timer t;
    t.begin();
#endif

    std::pair<int, string> parseResult = ParseSQL( sql, false, statements );
#ifdef ARIES_PROFILE
    parseTime = t.end();
#endif

    if ( 0 != parseResult.first )
    {
        if ( ER_FAKE_IMPL_OK != parseResult.first && ER_FAKE_IMPL_EOF != parseResult.first )
        {
            result->SetError( parseResult.first, parseResult.second);
            my_message( parseResult.first, parseResult.second.data(), MYF(0) );
        }
        else
        {
            result->SetSuccess( true );
        }
    }
    else
    {
    #ifdef ARIES_PROFILE
        t.begin();
    #endif
        result = executeStatements( statements, defaultDbName, sendResp );
    #ifdef ARIES_PROFILE
        execStmtTime = t.end();
    #endif
    }

#ifdef ARIES_PROFILE
    LOG( INFO ) << "parse time: " << parseTime << ", exec statments: " << execStmtTime << "ms";
#endif
    return  result;
}

SQLResultPtr SQLExecutor::ExecuteSQLFromFile(std::string file_path, const std::string& dbName) {

    LOG(INFO) << "Executing sql file: " << file_path << ", default database: " << dbName;
    auto result = std::make_shared<SQLResult>();
    std::vector<AriesSQLStatementPointer> statements;
    std::pair<int, string> parseResult = ParseSQL( file_path, true, statements );
    if ( 0 != parseResult.first )
    {
        if ( ER_FAKE_IMPL_OK != parseResult.first && ER_FAKE_IMPL_EOF != parseResult.first )
        {
            result->SetError( parseResult.first, parseResult.second);
            my_message( parseResult.first, parseResult.second.data(), MYF(0) );
        }
        else
        {
            result->SetSuccess( true );
        }
    }
    else
    {
        result =  executeStatements(statements, dbName, false);
    }
    return result;
}


// AbstractMemTablePointer SQLExecutor::ExecuteQuery(std::string sql, const std::string& dbName) {
//
//     auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
//
//     if (database == nullptr) {
//         LOG(ERROR) << "database not picked yet";
//     }
//
//     /*parse SQL*/
//     SelectStructurePointer parsed_query = parseSQL(sql);
//     LOG(INFO) << "sql: " << sql << " parsed";
//
//     return executeQuery(parsed_query, dbName);
// }

// AbstractMemTablePointer SQLExecutor::ExecuteQueryFromFile(std::string file, std::string dbName) {
//     auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
//
//     if (database == nullptr) {
//         LOG(ERROR) << "database not picked yet";
//     }
//
//     /*parse SQL*/
//     SelectStructurePointer parsed_query = parseSQLFromFile(file);
//     LOG(INFO) << "sql: " << file << " parsed";
//
//     return executeQuery(parsed_query, dbName);
// }

} // namespace aries
