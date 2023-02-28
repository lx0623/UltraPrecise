#include "AriesSQLStatement.h"
#include "SelectStructure.h"


namespace aries {


AriesSQLStatement::AriesSQLStatement() {
}

AriesSQLStatement::~AriesSQLStatement() {
    if (set_structure_ptrs) {
        set_structure_ptrs->clear();
        set_structure_ptrs = nullptr;
    }
    m_tx = nullptr;
}
bool AriesSQLStatement::IsQuery() {
    return (query_ptr != nullptr);
}

bool AriesSQLStatement::IsCommand() {
    return (command_ptr != nullptr);
}

bool AriesSQLStatement::IsShowStatement() {
    return (show_structure_ptr != nullptr);
}

bool AriesSQLStatement::IsSetStatement() {
    return (nullptr != set_structure_ptrs) && (set_structure_ptrs->size() > 0);
}

bool AriesSQLStatement::IsPreparedStatement() {
    return nullptr != prepared_stmt_ptr;
}

AbstractQueryPointer& AriesSQLStatement::GetQuery() {
    return query_ptr;
}

void AriesSQLStatement::SetQuery(AbstractQueryPointer arg_query) {
    query_ptr = arg_query;
}

AbstractCommandPointer AriesSQLStatement::GetCommand() {
    return command_ptr;
}

void AriesSQLStatement::SetCommand(AbstractCommandPointer arg_command) {
    command_ptr = arg_command;
}

bool AriesSQLStatement::IsExplainStatement() const {
    return explain_query_ptr != nullptr;
}

void AriesSQLStatement::SetExplainQuery(const AbstractQueryPointer& query) {
    explain_query_ptr = query;
}

AbstractQueryPointer& AriesSQLStatement::GetExplainQuery() {
    return explain_query_ptr;
}

} //namespace
