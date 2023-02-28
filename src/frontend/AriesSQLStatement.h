#ifndef ARIES_SQL_STATEMENT
#define ARIES_SQL_STATEMENT

#include "AbstractQuery.h"
#include "AbstractCommand.h"
#include "ShowStructure.h"
#include "SetStructure.h"
#include "PreparedStmtStructure.h"
#include "AdminStmtStructure.h"
#include "LoadDataStructure.h"
#include "TransactionStructure.h"
#include "UpdateStructure.h"
#include "InsertStructure.h"
#include "DeleteStructure.h"

BEGIN_ARIES_ENGINE_NAMESPACE
class AriesTransaction;
using AriesTransactionPtr = shared_ptr<AriesTransaction>;
END_ARIES_ENGINE_NAMESPACE

namespace aries {

class AriesSQLStatement {
private:
    AriesSQLStatement(const AriesSQLStatement &arg);

    AriesSQLStatement &operator=(const AriesSQLStatement &arg);

    AbstractQueryPointer query_ptr = nullptr;
    UpdateStructurePtr update_ptr;
    InsertStructurePtr insert_ptr;
    DeleteStructurePtr delete_ptr;
    AbstractCommandPointer command_ptr = nullptr;
    ShowStructurePtr show_structure_ptr = nullptr;
    std::shared_ptr<std::vector<SetStructurePtr>> set_structure_ptrs = nullptr;
    PreparedStmtStructurePtr prepared_stmt_ptr = nullptr;
    AdminStmtStructurePtr admin_stmt_ptr;
    LoadDataStructurePtr load_data_structure_ptr;
    AbstractQueryPointer explain_query_ptr = nullptr;
    TransactionStructurePtr tx_structure_ptr;

    aries_engine::AriesTransactionPtr m_tx;

public:
    AriesSQLStatement();

    ~AriesSQLStatement();

    bool IsQuery();
    bool IsCommand();
    bool IsShowStatement();
    bool IsSetStatement();
    bool IsPreparedStatement();
    bool IsAdminStmt()
    {
        return nullptr != admin_stmt_ptr;
    }
    bool IsLoadDataStmt()
    {
        return nullptr != load_data_structure_ptr;
    }

    bool IsExplainStatement() const;

    bool IsTxStmt() const { return nullptr != tx_structure_ptr; }

    bool IsUpdateStmt() const { return nullptr != update_ptr; }

    bool IsInsertStmt() const { return nullptr != insert_ptr; }

    bool IsDeleteStmt() const { return nullptr != delete_ptr; }

    void SetAdminStmt(const AdminStmtStructurePtr & arg_admin) {
        admin_stmt_ptr = arg_admin;
    }
    AdminStmtStructurePtr GetAdminPtr() {
        return admin_stmt_ptr;
    }

    AbstractQueryPointer& GetQuery();

    void SetQuery(AbstractQueryPointer arg_query);

    AbstractCommandPointer GetCommand();

    void SetCommand(AbstractCommandPointer arg_command);

    void SetShowStructure(const ShowStructurePtr& showStructurePointer) {
        show_structure_ptr = showStructurePointer;
    }
    const ShowStructurePtr& GetShowStructurePointer() const {
        return show_structure_ptr;
    }
    void SetSetStructures(const std::shared_ptr<std::vector<SetStructurePtr>>& setStructurePtrs) {
        set_structure_ptrs = setStructurePtrs;
    }
    const std::shared_ptr<std::vector<SetStructurePtr>>& GetSetStructurePtrs() const {
        return set_structure_ptrs;
    }

    void SetPreparedStmtStructurePtr(const PreparedStmtStructurePtr& arg_ptr) {
        prepared_stmt_ptr = arg_ptr;
    }
    const PreparedStmtStructurePtr& GetPreparedStmtStructurePtr() const {
        return prepared_stmt_ptr;
    }
    void SetLoadDataStructure(const LoadDataStructurePtr arg_load)
    {
        load_data_structure_ptr = arg_load;
    }
    const LoadDataStructurePtr& GetLoadDataStructurePtr() const {
        return load_data_structure_ptr;
    }

    void SetExplainQuery(const AbstractQueryPointer& query);

    AbstractQueryPointer& GetExplainQuery();

    void SetTxStructure( const TransactionStructurePtr txStructure )
    {
        tx_structure_ptr = txStructure;
    }
    const TransactionStructurePtr GetTxStructurePtr() const
    {
        return tx_structure_ptr;
    }

    void SetTransaction( const aries_engine::AriesTransactionPtr& tx )
    {
        m_tx = tx;
    }
    aries_engine::AriesTransactionPtr GetTransaction() const
    {
        return m_tx;
    }

    void SetUpdateStructure( const UpdateStructurePtr& updateStructure )
    {
        update_ptr = updateStructure;
    }
    UpdateStructurePtr GetUpdateStructure() const
    {
        return update_ptr;
    }

    void SetInsertStructure( const InsertStructurePtr& insertStructure )
    {
        insert_ptr = insertStructure;
    }
    InsertStructurePtr GetInsertStructure() const
    {
        return insert_ptr;
    }

    void SetDeleteStructure( const DeleteStructurePtr& deleteStructure )
    {
        delete_ptr = deleteStructure;
    }
    DeleteStructurePtr GetDeleteStructure() const
    {
        return delete_ptr;
    }
};

typedef std::shared_ptr<AriesSQLStatement> AriesSQLStatementPointer;

}//namespace


#endif
