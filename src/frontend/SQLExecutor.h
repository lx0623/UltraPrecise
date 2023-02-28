//
// Created by 胡胜刚 on 2019-06-13.
//

#pragma once

#include <string>
#include <memory>
#include <vector>

#include "SelectStructure.h"
#include "CommandStructure.h"
#include "SQLTreeNode.h"
#include "AriesServerSession.h"
#include "AriesSQLStatement.h"
#include "SQLResult.h"
#include "ShowColumnsStructure.h"

#include "schema/Schema.h"
#include "AriesEngineWrapper/AbstractMemTable.h"

namespace aries_engine {
  class AriesTransaction;
  using AriesTransactionPtr = shared_ptr<AriesTransaction>;
}

namespace aries {

enum SendDataType {
  SEND_METADATA_AND_ROWDATA = 1,
  SEND_METADATA_ONLY,
  SEND_ROWDATA_ONLY,
};

bool handleQueryResult( THD* thd, const AbstractMemTablePointer& amtp,
                        SendDataType send_type=SendDataType::SEND_METADATA_AND_ROWDATA,
                        ulong fetch_row_count=0, ulong offset_row_count=0 );

class SQLExecutor {
private:
    static SQLExecutor *instance;

    std::shared_ptr<AriesServerSession> session;

    explicit SQLExecutor();

    ~SQLExecutor() = default;

    // SelectStructurePointer parseSQL(std::string sql);

    // SelectStructurePointer parseSQLFromFile(std::string file_path);

    AbstractMemTablePointer executeQuery(const aries_engine::AriesTransactionPtr& tx,
                                         SelectStructurePointer& query,
                                         const string& defaultDbName, 
                                         bool buildQuery = true);
    AbstractMemTablePointer executeUpdate(const aries_engine::AriesTransactionPtr& tx,
                                          UpdateStructurePtr& updateStructure,
                                          const string& defaultDbName );
    AbstractMemTablePointer executeDelete(const aries_engine::AriesTransactionPtr& tx,
                                          DeleteStructurePtr& deleteStructure,
                                          const string& defaultDbName );
    AbstractMemTablePointer executeInsert(const aries_engine::AriesTransactionPtr& tx,
                                          InsertStructurePtr& insertStructure,
                                          const string& defaultDbName );
    AbstractMemTablePointer executeCommand(AbstractCommandPointer command, const string& dbName);

    SQLResultPtr executeShowStatement(const ShowStructurePtr& showStructurePointer, const std::string& defaultDbName);

    void executeSetStatements(const std::shared_ptr<std::vector<SetStructurePtr>>& setStructurePtrs, const std::string& dbName);

    void executePreparedStatement(const PreparedStmtStructurePtr preparedStmtStructurePtr, const std::string& dbName);

    void executeAdminStatement(const AdminStmtStructurePtr &adminStructure, const string &defaultDbName);

    void executeKillStatement(const KillStructurePtr& killStructure, const string& defaultDbName);

    void executeLoadDataStatement( aries_engine::AriesTransactionPtr& tx,
                                   const LoadDataStructurePtr& loadDataStructurePtr,
                                   const string& defaultDbName );

    void executeTxStatement( THD* thd, const AriesSQLStatementPointer& statement );

public:
    static void Init();

    static SQLExecutor *GetInstance();

    SQLResultPtr executeStatements(std::vector<AriesSQLStatementPointer> statements, const std::string& defaultDbName, bool sendResp, bool buildQuery = true );

    std::pair<int, string> ParseSQL(const std::string sql, bool file, std::vector<AriesSQLStatementPointer>& statements);
    SQLResultPtr ExecuteSQL(const std::string& sql, const std::string& defaultDbName, bool sendResp = false );

    SQLResultPtr ExecuteSQLFromFile(std::string file_path, const std::string& dbName);

    SQLResultPtr executeSelect( const string& sql, const string& defaultDbName );

    // AbstractMemTablePointer ExecuteQuery(std::string sql, const std::string& dbName);

    // AbstractMemTablePointer ExecuteQueryFromFile(std::string file, std::string dbName);

};
} // namespace aries
