/* Copyright (c) 2002, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

/**
  @file

This file contains the implementation of prepared statements.

When one prepares a statement:

  - Server gets the query from client with command 'COM_STMT_PREPARE';
    in the following format:
    [COM_STMT_PREPARE:1] [query]
  - Parse the query and recognize any parameter markers '?' and
    store its information list in lex->param_list
  - Allocate a new statement for this prepare; and keep this in
    'thd->stmt_map'.
  - Without executing the query, return back to client the total
    number of parameters along with result-set metadata information
    (if any) in the following format:
    @verbatim
    [STMT_ID:4]
    [Column_count:2]
    [Param_count:2]
    [Params meta info (stubs only for now)]  (if Param_count > 0)
    [Columns meta info] (if Column_count > 0)
    @endverbatim

  During prepare the tables used in a statement are opened, but no
  locks are acquired.  Table opening will block any DDL during the
  operation, and we do not need any locks as we neither read nor
  modify any data during prepare.  Tables are closed after prepare
  finishes.

When one executes a statement:

  - Server gets the command 'COM_STMT_EXECUTE' to execute the
    previously prepared query. If there are any parameter markers, then the
    client will send the data in the following format:
    @verbatim
    [COM_STMT_EXECUTE:1]
    [STMT_ID:4]
    [NULL_BITS:(param_count+7)/8)]
    [TYPES_SUPPLIED_BY_CLIENT(0/1):1]
    [[length]data]
    [[length]data] .. [[length]data].
    @endverbatim
    (Note: Except for string/binary types; all other types will not be
    supplied with length field)
  - If it is a first execute or types of parameters were altered by client,
    then setup the conversion routines.
  - Assign parameter items from the supplied data.
  - Execute the query without re-parsing and send back the results
    to client

  During execution of prepared statement tables are opened and locked
  the same way they would for normal (non-prepared) statement
  execution.  Tables are unlocked and closed after the execution.

When one supplies long data for a placeholder:

  - Server gets the long data in pieces with command type
    'COM_STMT_SEND_LONG_DATA'.
  - The packet received will have the format as:
    [COM_STMT_SEND_LONG_DATA:1][STMT_ID:4][parameter_number:2][data]
  - data from the packet is appended to the long data value buffer for this
    placeholder.
  - It's up to the client to stop supplying data chunks at any point. The
    server doesn't care; also, the server doesn't notify the client whether
    it got the data or not; if there is any error, then it will be returned
    at statement execute.
*/

#include <glog/logging.h>
#include "./include/sql_class.h"
#include "./include/my_sqlcommand.h"
#include "./include/my_dbug.h"
#include "./include/sql_prepare.h"
#include "./include/m_string.h"
#include "./include/derror.h"
#include "./include/set_var.h"            // set_var_base
#include "./include/mysql_com.h"
#include "frontend/SQLResult.h"
#include "frontend/SQLParserPortal.h"
#include "frontend/AriesSQLStatement.h"

#include <vector>
#include <algorithm>
#include <limits>
#include <server/mysql/include/mysqld.h>
#include <schema/SchemaManager.h>
#include <frontend/SchemaBuilder.h>
#include <frontend/QueryBuilder.h>
#include <server/mysql/include/mysql_def.h>
#include "utils/string_util.h"

using std::max;
using std::min;
using std::vector;

using namespace aries;

bool mysql_change_db(THD *thd, const string &new_db_name, bool force_switch);
void BuildQuery(SelectStructurePointer& query, const string& dbName, bool needRowIdColumn = false);
void handleSQLResult(THD* thd, SQLResultPtr sqlResultPtr, SendDataType type=SendDataType::SEND_METADATA_AND_ROWDATA);

inline bool is_param_null(const uchar *pos, ulong param_no)
{
  return pos[param_no/8] & (1 << (param_no & 7));
}

 /**
   Send prepared statement id and metadata to the client after prepare.

   @todo
     Fix this nasty upcast from List<Item_param> to List<Item>

   @return
     0 in case of success, 1 otherwise
 */

static bool send_prep_stmt(Prepared_statement* stmt, uint columns)
{
  THD *thd= stmt->thd;
  DBUG_ENTER("send_prep_stmt");
  NET *net= thd->get_protocol_classic()->get_net();
  uchar buff[12];
  uint tmp;
  int error;

  buff[0]= 0;                                   /* OK packet indicator */
  int4store(buff+1, stmt->id);
  int2store(buff+5, columns);
  int2store(buff+7, stmt->param_count);
  buff[9]= 0;                                   // Guard against a 4.1 client
  tmp= 0;// min(stmt->thd->get_stmt_da()->current_statement_cond_count(), 65535UL);
  int2store(buff+10, tmp);

  /*
    Send types and names of placeholders to the client
    XXX: fix this nasty upcast from List<Item_param> to List<Item>
  */
  error= my_net_write(net, buff, sizeof(buff));
  if (stmt->param_count && ! error)
  {
      vector<Send_field*> fields;
      uint i = 0;
      while (i++ < stmt->param_count) {
          Send_field field;
          field.db_name = "";
          field.table_name = field.org_table_name = "";
          field.col_name = "?";
          field.org_col_name = field.col_name;
          field.type = MYSQL_TYPE_VAR_STRING;
          field.length = 0;
          field.flags = 0;
          field.decimals = 0;
          fields.insert(fields.end(), &field);
      }
      error = thd->send_result_metadata(&fields, Protocol_classic::SEND_EOF);
  }

  if (!error)
    /* Flag that a response has already been sent */
    thd->get_stmt_da()->disable_status();

  DBUG_RETURN(error);
}

/**
  Check whether this parameter data type is compatible with long data.
  Used to detect whether a long data stream has been supplied to a
  incompatible data type.
*/
inline bool is_param_long_data_type(enum enum_field_types param_type)
{
  return ((param_type >= MYSQL_TYPE_TINY_BLOB) &&
          (param_type <= MYSQL_TYPE_STRING));
}

/**
  COM_STMT_PREPARE handler.

    Given a query string with parameter markers, create a prepared
    statement from it and send PS info back to the client.

    If parameter markers are found in the query, then store the information
    using Item_param along with maintaining a list in lex->param_array, so
    that a fast and direct retrieval can be made without going through all
    field items.

  @param thd                thread handle
  @param query              query to be prepared
  @param length             query string length, including ignored
                            trailing NULL or quote char.

  @note
    This function parses the query and sends the total number of parameters
    and resultset metadata information back to client (if any), without
    executing the query i.e. without any log/disk writes. This allows the
    queries to be re-executed without re-parsing during execute.

  @return
    none: in case of success a new statement id and metadata is sent
    to the client, otherwise an error message is set in THD.
*/

void mysqld_stmt_prepare(THD *thd, const char *query, uint length)
{
  Protocol *save_protocol= thd->get_protocol();
  Prepared_statement_ptr stmt;
  DBUG_ENTER("mysqld_stmt_prepare");

  /* First of all clear possible warnings from the previous command */
  thd->reset_for_next_command();

  if (! (stmt= std::make_shared<Prepared_statement>(thd))) {
      my_error(ER_OUT_OF_RESOURCES, MYF(0));
      DBUG_VOID_RETURN;
  }

  if (thd->stmt_map.insert(thd, stmt))
  {
    DBUG_VOID_RETURN;
  }

  // set the current client capabilities before switching the protocol
  thd->protocol_binary.set_client_capabilities(
      thd->get_protocol()->get_client_capabilities());

  thd->set_protocol(&thd->protocol_binary);

  if (!stmt->prepare(query, length)->IsSuccess())
  {
    /* Delete this stmt stats from PS table. */
    thd->stmt_map.erase(stmt);
  }

  thd->set_protocol(save_protocol);

  // sp_cache_enforce_limit(thd->sp_proc_cache, stored_program_cache_size);
  // sp_cache_enforce_limit(thd->sp_func_cache, stored_program_cache_size);

  /* check_prepared_statement sends the metadata packet in case of success */
  DBUG_VOID_RETURN;
}

/**
  SQLCOM_PREPARE implementation.

    Prepare an SQL prepared statement. This is called from
    mysql_execute_command and should therefore behave like an
    ordinary query (e.g. should not reset any global THD data).

  @param thd     thread handle

  @return
    none: in case of success, OK packet is sent to the client,
    otherwise an error message is set in THD
*/

void mysql_sql_stmt_prepare(THD *thd, const PreparedStmtStructurePtr preparedStmtStructurePtr) {
    Prepared_statement_ptr stmt;

    if ((stmt = thd->stmt_map.find_by_name(preparedStmtStructurePtr->stmtName))) {
        /*
          If there is a statement with the same name, remove it. It is ok to
         remove old and fail to insert a new one at the same time.
        */
        if (stmt->is_in_use())
        {
            my_error(ER_PS_NO_RECURSION, MYF(0));
            DBUG_VOID_RETURN;
        }
        thd->stmt_map.erase(stmt);
    }
    string prepareStmt;
    if (preparedStmtStructurePtr->prepareSrcPtr->isVarRef) {
        auto userVarEntryPtr = thd->get_user_var(preparedStmtStructurePtr->prepareSrcPtr->stmtCode);
        if (!userVarEntryPtr) {
            string errMsg = format_err_msg("%s User defined variable %s does not exist",
                                           ERRMSG_SYNTAX_ERROR, preparedStmtStructurePtr->prepareSrcPtr->stmtCode.data());
            ARIES_EXCEPTION_SIMPLE(ER_PARSE_ERROR, errMsg);
        }
        if (AriesExprType::STRING != userVarEntryPtr->getExpr()->GetType()) {
            string errMsg = format_err_msg("%s Value of user defined variable %s is not string",
                                           ERRMSG_SYNTAX_ERROR, preparedStmtStructurePtr->prepareSrcPtr->stmtCode.data());
            ARIES_EXCEPTION_SIMPLE(ER_PARSE_ERROR, errMsg);
        }
        prepareStmt = boost::get<string>(userVarEntryPtr->getExpr()->GetContent());
    } else {
        prepareStmt = preparedStmtStructurePtr->prepareSrcPtr->stmtCode;
    }
    prepareStmt = aries_utils::trim(prepareStmt);
    if (prepareStmt.empty()) {
        ARIES_EXCEPTION( ER_EMPTY_QUERY );
    }
    if (! (stmt= std::make_shared<Prepared_statement>(thd))) {
        my_error(ER_OUT_OF_RESOURCES, MYF(0));
        DBUG_VOID_RETURN;
    }
    stmt->set_sql_prepare();
    stmt->set_name(preparedStmtStructurePtr->stmtName);
    if (thd->stmt_map.insert(thd, stmt))
    {
        DBUG_VOID_RETURN;
    }

    if (!stmt->prepare(prepareStmt.data(), prepareStmt.size())->IsSuccess())
    {
        /* Delete this stmt stats from PS table. */
        thd->stmt_map.erase(stmt);
    } else {
        my_ok(thd, 0, 0, "Statement prepared");
    }
}
 /**
   Clears parameters from data left from previous execution or long data.

   @param stmt               prepared statement for which parameters should
                             be reset
 */

static void reset_stmt_params(Prepared_statement *stmt)
{
    for (size_t i = 0; i < stmt->param_array.size(); i++) {
        auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(stmt->param_array[i]);
        expr->ClearPreparedStmtParam();
    }
}


/**
  COM_STMT_EXECUTE handler: execute a previously prepared statement.

    If there are any parameters, then replace parameter markers with the
    data supplied from the client, and then execute the statement.
    This function uses binary protocol to send a possible result set
    to the client.

  @param thd                current thread
  @param stmt_id            statement id
  @param flags              flags mask
  @param params             parameter types and data, if any
  @param params_length      packet length, including the terminator character.

  @return
    none: in case of success OK packet or a result set is sent to the
    client, otherwise an error message is set in THD.
*/

aries::SQLResultPtr mysqld_stmt_execute(THD *thd, ulong stmt_id, ulong flags, uchar *params,
                                        ulong params_length) {
    auto result = std::make_shared<SQLResult>();
    /* Query text for binary, general or slow log, if any of them is open */
    Prepared_statement_ptr stmt;
    Protocol *save_protocol = thd->get_protocol();
    bool open_cursor;
    DBUG_ENTER("mysqld_stmt_execute");

    /* First of all clear possible warnings from the previous command */
    thd->reset_for_next_command();

    if (!(stmt = thd->stmt_map.find(stmt_id))) {
        char llbuf[22];
        my_error(ER_UNKNOWN_STMT_HANDLER, MYF(0), static_cast<int>(sizeof(llbuf)),
                 aries_utils::llstr(stmt_id, llbuf), "mysqld_stmt_execute");
        result->SetErrorCode(ER_UNKNOWN_STMT_HANDLER);
        return result;
    }

    open_cursor= MY_TEST(flags & (ulong) CURSOR_TYPE_READ_ONLY);

    // set the current client capabilities before switching the protocol
    thd->protocol_binary.set_client_capabilities(
            thd->get_protocol()->get_client_capabilities());
    thd->set_protocol(&thd->protocol_binary);

    result = stmt->execute_loop(open_cursor, params, params + params_length, nullptr);
    SendDataType type = SendDataType::SEND_METADATA_AND_ROWDATA;
    if(open_cursor){
        //cache the result, so next can fetch this result row by row
        stmt->set_cache_result(result, 0);
        thd->server_status|= SERVER_STATUS_CURSOR_EXISTS;
        stmt->open_cursor();
        type = SendDataType::SEND_METADATA_ONLY;
    }
    handleSQLResult(thd, result, type);
    thd->set_protocol(save_protocol);

    return result;
}
//
//
/**
  SQLCOM_EXECUTE implementation.

    Execute prepared statement using parameter values from
    lex->prepared_stmt_params and send result to the client using
    text protocol. This is called from mysql_execute_command and
    therefore should behave like an ordinary query (e.g. not change
    global THD data, such as warning count, server status, etc).
    This function uses text protocol to send a possible result set.

  @param thd                thread handle

  @return
    none: in case of success, OK (or result set) packet is sent to the
    client, otherwise an error is set in THD
*/

void mysql_sql_stmt_execute(THD *thd, const PreparedStmtStructurePtr preparedStmtStructurePtr)
{
  Prepared_statement_ptr stmt;
  DBUG_ENTER("mysql_sql_stmt_execute");
  LOG(INFO) << "EXECUTE: " << preparedStmtStructurePtr->stmtName;

  if (!(stmt= thd->stmt_map.find_by_name(preparedStmtStructurePtr->stmtName)))
  {
    my_error(ER_UNKNOWN_STMT_HANDLER, MYF(0), preparedStmtStructurePtr->stmtName.size(), preparedStmtStructurePtr->stmtName.data(), "EXECUTE");
    DBUG_VOID_RETURN;
  }

  if (stmt->param_count != preparedStmtStructurePtr->executeVars.size())
  {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), "EXECUTE");
    DBUG_VOID_RETURN;
  }

  SQLResultPtr result = stmt->execute_loop(FALSE, NULL, NULL, &preparedStmtStructurePtr->executeVars);
  handleSQLResult(thd, result, SendDataType::SEND_METADATA_AND_ROWDATA);

  DBUG_VOID_RETURN;
}

/**
  COM_STMT_FETCH handler: fetches requested amount of rows from cursor.

  @param thd                Thread handle
  @param stmt_id            Packet from client (with stmt_id & num_rows)
  @param num_rows           number of fetch rows
*/

void mysqld_stmt_fetch(THD *thd, ulong stmt_id, ulong num_rows)
{
  Prepared_statement_ptr stmt;
  DBUG_ENTER("mysqld_stmt_fetch");
  Protocol *save_protocol = thd->get_protocol();

  /* First of all clear possible warnings from the previous command */
  thd->reset_for_next_command();
  if (!(stmt= thd->stmt_map.find(stmt_id)))
  {
    char llbuf[22];
    my_error(ER_UNKNOWN_STMT_HANDLER, MYF(0), static_cast<int>(sizeof(llbuf)),
             aries_utils::llstr(stmt_id, llbuf), "mysqld_stmt_fetch");
    DBUG_VOID_RETURN;
  }

  if (!stmt->is_cursor_open())
  {
    my_error(ER_STMT_HAS_NO_OPEN_CURSOR, MYF(0), stmt_id);
    DBUG_VOID_RETURN;
  }

  thd->protocol_binary.set_client_capabilities(
            thd->get_protocol()->get_client_capabilities());
    thd->set_protocol(&thd->protocol_binary);

  std::pair<aries::SQLResultPtr, size_t> result = stmt->get_cache_result();
  if (result.first && num_rows > 0)
  {
    //
    handleQueryResult( thd, result.first->GetResults()[ 0 ], SendDataType::SEND_ROWDATA_ONLY, num_rows, result.second );
  }
  if(thd->server_status & SERVER_STATUS_LAST_ROW_SENT){
    reset_stmt_params(stmt.get());
    stmt->close_cursor();
  }
    
  stmt->set_cache_result(result.first, result.second+num_rows);

  thd->set_protocol(save_protocol);

  DBUG_VOID_RETURN;
}

/**
  Reset a prepared statement in case there was a recoverable error.

    This function resets statement to the state it was right after prepare.
    It can be used to:
    - clear an error happened during mysqld_stmt_send_long_data
    - cancel long data stream for all placeholders without
      having to call mysqld_stmt_execute.
    - close an open cursor
    Sends 'OK' packet in case of success (statement was reset)
    or 'ERROR' packet (unrecoverable error/statement not found/etc).

  @param thd                Thread handle
  @param stmt_id            Stmt id
*/

void mysqld_stmt_reset(THD *thd, ulong stmt_id)
{
  Prepared_statement_ptr stmt;
  DBUG_ENTER("mysqld_stmt_reset");

  /* First of all clear possible warnings from the previous command */
  thd->reset_for_next_command();

  // thd->status_var.com_stmt_reset++;
  if (!(stmt= thd->stmt_map.find(stmt_id)))
  {
    char llbuf[22];
    my_error(ER_UNKNOWN_STMT_HANDLER, MYF(0), static_cast<int>(sizeof(llbuf)),
             aries_utils::llstr(stmt_id, llbuf), "mysqld_stmt_reset");
    DBUG_VOID_RETURN;
  }

  // stmt->close_cursor();

  /*
    Clear parameters from data which could be set by
    mysqld_stmt_send_long_data() call.
  */
  reset_stmt_params(stmt.get());

  // stmt->state= Query_arena::STMT_PREPARED;

  my_ok(thd);

  DBUG_VOID_RETURN;
}


/**
  Delete a prepared statement from memory.

  @note
    we don't send any reply to this command.
*/

void mysqld_stmt_close(THD *thd, ulong stmt_id)
{
  Prepared_statement_ptr stmt;
  DBUG_ENTER("mysqld_stmt_close");

  thd->get_stmt_da()->disable_status();

  if (!(stmt= thd->stmt_map.find(stmt_id)))
    DBUG_VOID_RETURN;

  /*
    The only way currently a statement can be deallocated when it's
    in use is from within Dynamic SQL.
  */
  DBUG_ASSERT(! stmt->is_in_use());
  thd->stmt_map.erase(stmt);

  DBUG_VOID_RETURN;
}


/**
  SQLCOM_DEALLOCATE implementation.

    Close an SQL prepared statement. As this can be called from Dynamic
    SQL, we should be careful to not close a statement that is currently
    being executed.

  @return
    none: OK packet is sent in case of success, otherwise an error
    message is set in THD
*/

void mysql_sql_stmt_close(THD *thd, const PreparedStmtStructurePtr preparedStmtStructurePtr)
{
  Prepared_statement_ptr stmt;
  LOG(INFO) << "DEALLOCATE PREPARE: " << preparedStmtStructurePtr->stmtName;

  if (! (stmt= thd->stmt_map.find_by_name(preparedStmtStructurePtr->stmtName)))
    my_error(ER_UNKNOWN_STMT_HANDLER, MYF(0), preparedStmtStructurePtr->stmtName.size(), preparedStmtStructurePtr->stmtName.data(), "DEALLOCATE PREPARE");
  else if (stmt->is_in_use())
    my_error(ER_PS_NO_RECURSION, MYF(0));
  else
  {
    // if (thd->session_tracker.get_tracker(SESSION_STATE_CHANGE_TRACKER)->is_enabled())
    //   thd->session_tracker.get_tracker(SESSION_STATE_CHANGE_TRACKER)->mark_as_changed(thd, NULL);
    my_ok(thd);
  }
  thd->stmt_map.erase(stmt);

}


/**
  Handle long data in pieces from client.

    Get a part of a long data. To make the protocol efficient, we are
    not sending any return packets here. If something goes wrong, then
    we will send the error on 'execute' We assume that the client takes
    care of checking that all parts are sent to the server. (No checking
    that we get a 'end of column' in the server is performed).

  @param thd                Thread handle
  @param stmt_id            Stmt id
  @param param_number       Number of parameters
  @param str                String to append
  @param length             Length of string (including end \\0)
*/

void mysql_stmt_get_longdata(THD *thd, ulong stmt_id, uint param_number,
                             uchar *str, ulong length) {
    Prepared_statement_ptr stmt;
    DBUG_ENTER("mysql_stmt_get_longdata");

    // thd->status_var.com_stmt_send_long_data++;

    thd->get_stmt_da()->disable_status();

    if (!(stmt = thd->stmt_map.find(stmt_id)))
        DBUG_VOID_RETURN;

    if (param_number >= stmt->param_count) {
        /* Error will be sent in execute call */
        // stmt->state= Query_arena::STMT_ERROR;
        stmt->last_errno = ER_WRONG_ARGUMENTS;
        sprintf(stmt->last_error, ER(ER_WRONG_ARGUMENTS),
                "mysqld_stmt_send_long_data");
        DBUG_VOID_RETURN;
    }
    auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(stmt->param_array[param_number]);
    expr->SetPreparedStmtLongData((char *) str, length);
}

/***************************************************************************
 Prepared_statement
****************************************************************************/

Prepared_statement::Prepared_statement(THD *thd_arg)
  :thd(thd_arg),
  // param_array(NULL),
  param_count(0),
  last_errno(0),
  id(++thd_arg->statement_id_counter),
  // result(thd_arg),
  flags((uint) IS_IN_USE),
  with_log(false)
{
  *last_error= '\0';
}

/**
  Parse statement text, validate the statement, and prepare it for execution.

    You should not change global THD state in this function, if at all
    possible: it may be called from any context, e.g. when executing
    a COM_* command, and SQLCOM_* command, or a stored procedure.

    mysql> prepare s1 from 'select * from ?';
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '?' at line 1

  @param query_str             statement text
  @param query_length

  @note
    Precondition:
  @note
    Postcondition:
*/

SQLResultPtr Prepared_statement::prepare(const char *query_str, size_t query_length) {
    auto result = std::make_shared<SQLResult>();
    std::string origQuery = query_str;
    origQuery.append(";");

    SQLParserPortal parser;
    AriesSQLStatementPointer statement = nullptr;
    string dbName = current_thd->db();

    current_thd->stmt_params.clear();

    std::pair<int, string> parseResult = aries::SQLExecutor::GetInstance()->ParseSQL( origQuery, false, m_statements );
    if ( 0 != parseResult.first )
    {
        auto result = std::make_shared<SQLResult>();
        if ( ER_FAKE_IMPL_OK != parseResult.first && ER_FAKE_IMPL_EOF != parseResult.first )
        {
            result->SetError( parseResult.first, parseResult.second );
            my_message( parseResult.first, parseResult.second.data(), MYF(0) );
        }
        else
        {
            result->SetSuccess( true );
        }

        flags&= ~ (uint) IS_IN_USE;
        return result;
    }

    if (m_statements.size() > 1) {
        flags&= ~ (uint) IS_IN_USE;
        char ebuff[ERRMSGSIZE];
        snprintf(ebuff, sizeof(ebuff), "%s %s", ERRMSG_SYNTAX_ERROR, "SQL syntax for prepared statements does not support multi-statements");
        my_message(ER_SYNTAX_ERROR, ebuff, MYF(0));
        result->SetError( ER_SYNTAX_ERROR, string( ebuff ) );
        return result;
    }
    param_array = current_thd->stmt_params;
    current_thd->stmt_params.clear();
    param_count = param_array.size();
    statement = m_statements[0];
    try {
        if (statement->IsQuery()) {
            auto query = std::dynamic_pointer_cast<SelectStructure>(statement->GetQuery());
            BuildQuery(query, dbName);

            auto selectPart = query->GetSelectPart() ;
            int itemCount = selectPart->GetSelectItemCount();

            if (!is_sql_prepare()) {
                send_prep_stmt(this, itemCount);

                vector<Send_field*> fields;
                for (int i = 0; i < itemCount; i++) {
                    string columnName = selectPart->GetName(i);
                    Send_field* field = new Send_field();
                    field->db_name = "";
                    field->table_name = field->org_table_name = "";
                    field->col_name = columnName;
                    field->org_col_name = field->col_name;
                    field->type = MYSQL_TYPE_VAR_STRING;
                    field->length = 0;
                    field->flags = 0;
                    field->decimals = 0;
                    fields.insert(fields.end(), field);

                    BiaodashiPointer abp = selectPart->GetSelectExpr(i);
                    CommonBiaodashi *rawpointer = (CommonBiaodashi *) abp.get();

                    switch (rawpointer->GetValueType()) {
                        case aries::ColumnValueType::BOOL: 
                        case aries::ColumnValueType::TINY_INT: 
                        {
                            field->type = MYSQL_TYPE_TINY;
                            field->length = 4;
                            break;
                        }
                        case aries::ColumnValueType::SMALL_INT: {
                            field->type = MYSQL_TYPE_SHORT;
                            field->length = 6;
                            break;
                        }
                        case aries::ColumnValueType::INT: {
                            field->type = MYSQL_TYPE_LONG;
                            field->length = 11;
                            break;
                        }
                        case aries::ColumnValueType::LONG_INT: {
                            field->type = MYSQL_TYPE_LONGLONG;
                            field->length = 20;
                            break;
                        }
                        case aries::ColumnValueType::FLOAT: {
                            field->type = MYSQL_TYPE_FLOAT;
                            field->length = 12;
                            field->decimals = 0x1f;
                            break;
                        }
                        case aries::ColumnValueType::DOUBLE: {
                            field->type = MYSQL_TYPE_DOUBLE;
                            field->length = 12;
                            field->decimals = 0x1f;
                            break;
                        }
                        case aries::ColumnValueType::DECIMAL: {
                            field->type = MYSQL_TYPE_NEWDECIMAL;
                            field->length = 11;
                            field->decimals = 31;
                            break;
                        }
                        case aries::ColumnValueType::TEXT: {
                            field->type = MYSQL_TYPE_VAR_STRING;
                            break;
                        }
                        case aries::ColumnValueType::DATE: {
                            field->type = MYSQL_TYPE_DATE;
                            field->length = 10;
                            field->flags |= BINARY_FLAG;
                            break;
                        }
                        case aries::ColumnValueType::TIME: {
                            field->type = MYSQL_TYPE_TIME;
                            field->length = 10;
                            field->flags |= BINARY_FLAG;
                            break;
                        }
                        case aries::ColumnValueType::DATE_TIME: {
                            field->type = MYSQL_TYPE_DATETIME;
                            field->length = 19;
                            field->flags |= BINARY_FLAG;
                            break;
                        }
                        case aries::ColumnValueType::TIMESTAMP: {
                            field->type = MYSQL_TYPE_TIMESTAMP;
                            field->length = 19;
                            field->flags |= BINARY_FLAG;
                            break;
                        }
                        case aries::ColumnValueType::YEAR: {
                            field->type = MYSQL_TYPE_YEAR;
                            field->length = 4;
                            field->flags |= (UNSIGNED_FLAG | ZEROFILL_FLAG);
                            break;
                        }
                        default:
                          // prepare 'select ?' 时，不知道？的类型时默认为MYSQL_TYPE_VAR_STRING
                          field->type = MYSQL_TYPE_VAR_STRING;
                          break;
                    }
                }
                thd->send_result_metadata(&fields, Protocol_classic::SEND_EOF);
                for (auto f : fields) {
                    delete f;
                }
                thd->get_protocol_classic()->flush();
            }
            result->SetSuccess(true);
        } else {
            string msg = "Non select prepared statements";
            ARIES_EXCEPTION(ER_NOT_SUPPORTED_YET, msg.data());
        }
    } catch (AriesException& ariesException) {
        result->SetError(ariesException.errCode, ariesException.errMsg);
        my_message(ariesException.errCode, ariesException.errMsg.data(), MYF(0));
        LOG(ERROR) << ariesException.errMsg;
    } catch (...) {
        result->SetError(ER_UNKNOWN_ERROR, "Unknown error");
        my_error(ER_UNKNOWN_ERROR, MYF(0));
        LOG(ERROR) << "Unknown error.";
    }

    flags&= ~ (uint) IS_IN_USE;
    return result;
}

/**
  Assign parameter values either from variables, in case of SQL PS
  or from the execute packet.

  @param expanded_query  a container with the original SQL statement.
                         '?' placeholders will be replaced with
                         their values in case of success.
                         The result is used for logging and replication
  @param packet          pointer to execute packet.
                         NULL in case of SQL PS
  @param packet_end      end of the packet. NULL in case of SQL PS

  @todo Use a paremeter source class family instead of 'if's, and
  support stored procedure variables.

  @retval TRUE an error occurred when assigning a parameter (likely
          a conversion error or out of memory, or malformed packet)
  @retval FALSE success
*/

bool
Prepared_statement::set_parameters(// string *expanded_query,
                                   uchar *packet, uchar *packet_end,
                                   std::vector<std::string>* executeVars)
{
  bool is_sql_ps= packet == NULL;
  bool res= FALSE;

  if (is_sql_ps)
  {
      /* SQL prepared statement */
      if (param_count != executeVars->size()) {
          return 1;
      }
      for (size_t i = 0; i < param_array.size(); i++)
      {
          std::shared_ptr<CommonBiaodashi> expr = std::dynamic_pointer_cast<CommonBiaodashi>(param_array[i]);
          user_var_entry_ptr userVarEntryPtr = current_thd->get_user_var((*executeVars)[i]);
          if (!userVarEntryPtr) {
              expr->SetPreparedStmtNullParam();
              continue;
          }
          expr->SetPreparedStmtParam(userVarEntryPtr);
      }
  }
  else if (param_count)
  {
      uchar *null_array= packet;
      uchar* data_end = packet_end;

      /* skip null bits */
      uchar *read_pos= packet + (param_count+7) / 8;
      if (read_pos >= data_end)
          DBUG_RETURN(1);

      DBUG_ENTER("setup_conversion_functions");

      uchar *data = read_pos + 1;
      uchar newParam = *read_pos++;
      if (newParam) //types supplied / first execute
      {
          data += param_array.size() * 2;
      }
      if (data > data_end)
          DBUG_RETURN(1);
      /*
        First execute or types altered by the client, setup the
        conversion routines for all parameters (one time)
      */
      const uint signed_bit= 1 << 15;
      for (size_t i = 0; i < param_array.size(); i++)
      {
          std::shared_ptr<CommonBiaodashi> expr = std::dynamic_pointer_cast<CommonBiaodashi>(param_array[i]);
          uchar param_type = 0xff;
          if (newParam) {
              if (read_pos >= data_end)
                  DBUG_RETURN(1);

              ushort typecode= sint2korr(read_pos);
              read_pos+= 2;
              // my_bool unsigned_flag = MY_TEST(typecode & signed_bit);
              param_type = (uchar)(typecode & ~signed_bit);
          } else {
              param_type = expr->GetParamType();
          }
          if (!expr->IsLongDataValue()) {
              if (is_param_null(null_array, i)) {
                  expr->SetPreparedStmtNullParam();
                  continue;
              }
              expr->SetPreparedStmtParam((enum enum_field_types) param_type, &data, data_end);
          } else {
              if (!is_param_long_data_type((enum enum_field_types)param_type))  {
                  DBUG_RETURN(1);
              }
              expr->SetPreparedStmtLongParam((enum enum_field_types) param_type);
          }
      }
  }
  return res;
}
//
//
/**
  Execute a prepared statement. Re-prepare it a limited number
  of times if necessary.

  Try to execute a prepared statement. If there is a metadata
  validation error, prepare a new copy of the prepared statement,
  swap the old and the new statements, and try again.
  If there is a validation error again, repeat the above, but
  perform no more than MAX_REPREPARE_ATTEMPTS.

  @note We have to try several times in a loop since we
  release metadata locks on tables after prepared statement
  prepare. Therefore, a DDL statement may sneak in between prepare
  and execute of a new statement. If this happens repeatedly
  more than MAX_REPREPARE_ATTEMPTS times, we give up.

  @return TRUE if an error, FALSE if success
  @retval  TRUE    either MAX_REPREPARE_ATTEMPTS has been reached,
                   or some general error
  @retval  FALSE   successfully executed the statement, perhaps
                   after having reprepared it a few times.
*/

aries::SQLResultPtr
Prepared_statement::execute_loop(// string *expanded_query,
                                 bool open_cursor,
                                 uchar *packet,
                                 uchar *packet_end,
                                 std::vector<std::string>* executeVars)
{
  auto result = std::make_shared<SQLResult>();
  bool error = 0;

  if (flags & (uint) IS_IN_USE)
  {
      result->SetErrorCode(ER_PS_NO_RECURSION);
      my_error(ER_PS_NO_RECURSION, MYF(0));
      return result;
  }

  flags|= IS_IN_USE;

  /* Check if we got an error when sending long data */
  // if (state == Query_arena::STMT_ERROR)
  // {
  //   my_message(last_errno, last_error, MYF(0));
  //   return TRUE;
  // }

  DBUG_ASSERT(!thd->get_stmt_da()->is_set());

  bool is_sql_ps= packet == NULL;
  error = set_parameters(packet, packet_end, executeVars);
  if (error)
  {
      flags&= ~ (uint) IS_IN_USE;
      my_error(ER_WRONG_ARGUMENTS, MYF(0),
               is_sql_ps ? "EXECUTE" : "mysqld_stmt_execute");
      reset_stmt_params(this);
      result->SetErrorCode(ER_WRONG_ARGUMENTS);
      return result;
  }
  result = SQLExecutor::GetInstance()->executeStatements(m_statements, current_thd->db(), false, false);

  flags&= ~ (uint) IS_IN_USE;
  reset_stmt_params(this);
  return result;
}
