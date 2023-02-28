#include <string.h>
#include <ctype.h>
#include <string>
#include <algorithm>
#include <glog/logging.h>
#include <server/mysql/include/mysqld_thd_manager.h>

#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/my_command.h"
#include "server/mysql/include/com_data.h"
#include "server/mysql/include/protocol_classic.h"
#include "server/mysql/include/binary_log_types.h"

#include "schema/SchemaManager.h"
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngineWrapper/AbstractMemTable.h"
#include "CudaAcc/AriesEngineDef.h"

#include "server/mysql/include/derror.h"
#include "utils/string_util.h"
#include "server/mysql/include/mysql_def.h"
#include "server/mysql/include/sql_prepare.h"
#include "server/mysql/include/sql_authentication.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "CpuTimer.h"
using namespace mysql;

using namespace std;
using namespace aries::schema;
using namespace aries_utils;
using namespace aries_acc;
using namespace aries_engine;
using aries::SQLResultPtr;
extern CHARSET_INFO *default_charset_info;

bool dispatch_command(THD *thd, const COM_DATA *com_data,
                      enum enum_server_command command);
bool net_send_error(THD *thd, uint sql_errno, const char *err);
bool mysql_change_db(THD *thd, const string &new_db_name, bool force_switch);
void store_lenenc_string(String &to, const uchar *from, size_t length);

/**
  Shutdown the mysqld server.

  @param  thd        Thread (session) context.
  @param  level      Shutdown level.
  @param command     type of command to perform

  @retval
    true                 success
  @retval
    false                When user has insufficient privilege or unsupported shutdown level

*/
bool shutdown(THD *thd, enum mysql_enum_shutdown_level level, enum enum_server_command command)
{
    DBUG_ENTER("shutdown");
    bool res= FALSE;

    // if (check_global_access(thd,SHUTDOWN_ACL))
    //     goto error; /* purecov: inspected */

    if (level == SHUTDOWN_DEFAULT)
        level= SHUTDOWN_WAIT_ALL_BUFFERS; // soon default will be configurable
    else if (level != SHUTDOWN_WAIT_ALL_BUFFERS)
    {
        my_error(ER_NOT_SUPPORTED_YET, MYF(0), "this shutdown level");
        goto error;
    }

    if(command == COM_SHUTDOWN)
        my_eof(thd);
    else if(command == COM_QUERY)
        my_ok(thd);
    else
    {
        my_error(ER_NOT_SUPPORTED_YET, MYF(0), "shutdown from this server command");
        goto error;
    }

    // DBUG_PRINT("quit",("Got shutdown command for level %u", level));
    // query_logger.general_log_print(thd, command, NullS);
    kill_mysql();
    res= TRUE;

    error:
    DBUG_RETURN(res);
}

/**
  Read one command from connection and execute it (query or simple command).
  This function is called in loop from thread function.

  For profiling to work, it must never be called recursively.

  @retval
    0  success
  @retval
    1  request of thread shutdown (see dispatch_command() description)
*/

bool do_command(THD *thd)
{
  bool return_value;
  int rc;
  const bool classic= true;

  NET *net= NULL;
  enum enum_server_command command;
  COM_DATA com_data;

    /*
      XXX: this code is here only to clear possible errors of init_connect.
      Consider moving to prepare_new_connection_state() instead.
      That requires making sure the DA is cleared before non-parsing statements
      such as COM_QUIT.
    */
    thd->clear_error();				// Clear error message
    thd->get_stmt_da()->reset_diagnostics_area();

  /*
      This thread will do a blocking read from the client which
      will be interrupted when the next command is received from
      the client, the connection is closed or "net_wait_timeout"
      number of seconds has passed.
    */
  net = thd->get_protocol_classic()->get_net();
  // my_net_set_read_timeout(net, thd->variables.net_wait_timeout);
  my_net_set_read_timeout(net, 28800);
  net_new_transaction(net);

  /*
    Synchronization point for testing of KILL_CONNECTION.
    This sync point can wait here, to simulate slow code execution
    between the last test of thd->killed and blocking in read().

    The goal of this test is to verify that a connection does not
    hang, if it is killed at this point of execution.
    (Bug#37780 - main.kill fails randomly)

    Note that the sync point wait itself will be terminated by a
    kill. In this case it consumes a condition broadcast, but does
    not change anything else. The consumed broadcast should not
    matter here, because the read/recv() below doesn't use it.
  */
  // DEBUG_SYNC(thd, "before_do_command_net_read");

  /*
    Because of networking layer callbacks in place,
    this call will maintain the following instrumentation:
    - IDLE events
    - SOCKET events
    - STATEMENT events
    - STAGE events
    when reading a new network packet.
    In particular, a new instrumented statement is started.
    See init_net_server_extension()
  */
  thd->m_server_idle= true;
  AriesTransManager::GetInstance().NotifyTxEndOrIdle();
  rc= thd->get_protocol_classic()->get_command(&com_data, &command);
  thd->m_server_idle= false;

  if (rc)
  {
    // /* The error must be set. */
    DBUG_ASSERT(thd->is_error());
    thd->send_statement_status();

    if (rc < 0)
    {
      return_value= TRUE;                       // We have to close it.
      goto out;
    }
    if (classic)
      net->error= 0;
    return_value= FALSE;
    goto out;
  }

  // DBUG_PRINT("info",("Command on %s = %d (%s)",
  //                    vio_description(net->vio), command,
  //                    command_name[command].str));

  // DBUG_PRINT("info", ("packet: '%*.s'; command: %d",
  //            thd->get_protocol_classic()->get_packet_length(),
  //            thd->get_protocol_classic()->get_raw_packet(), command));
  if (thd->get_protocol_classic()->bad_packet)
    DBUG_ASSERT(0);                // Should be caught earlier

  // Reclaim some memory
  // thd->get_protocol_classic()->get_packet()->shrink(
  //     thd->variables.net_buffer_length);
  /* Restore read timeout value */
  if (classic)
    my_net_set_read_timeout(net, 30);
  // my_net_set_read_timeout(net, thd->variables.net_read_timeout);

  return_value= dispatch_command(thd, &com_data, command);
  // thd->get_protocol_classic()->get_packet()->shrink(
  //     thd->variables.net_buffer_length); 16384

out:
  DBUG_RETURN(return_value);
}

#define SEND_BUFFER_SIZE 8192
#define RECV_BUFFER_SIZE 8192
// static char RecvBuffer[RECV_BUFFER_SIZE];
// static char SendBuffer[SEND_BUFFER_SIZE];

// /* mysql-connector-java-5.1.38 ( Revision: fe541c166cec739c74cc727c5da96c1028b4834a ) */
// SELECT
// @@session.auto_increment_increment AS auto_increment_increment,
// @@character_set_client AS character_set_client,
// @@character_set_connection AS character_set_connection,
// @@character_set_results AS character_set_results,
// @@character_set_server AS character_set_server,
// @@init_connect AS init_connect,
// @@interactive_timeout AS interactive_timeout,
// @@language AS language,
// @@license AS license,
// @@lower_case_table_names AS lower_case_table_names,
// @@max_allowed_packet AS max_allowed_packet,
// @@net_buffer_length AS net_buffer_length,
// @@net_write_timeout AS net_write_timeout,
// @@query_cache_size AS query_cache_size,
// @@query_cache_type AS query_cache_type,
// @@sql_mode AS sql_mode,
// @@system_time_zone AS system_time_zone,
// @@time_zone AS time_zone,
// @@tx_isolation AS tx_isolation,
// @@wait_timeout AS wait_timeout

// /* mysql-connector-java-5.1.47 ( Revision: fe1903b1ecb4a96a917f7ed3190d80c049b1de29 ) */
// SELECT
//@@session.auto_increment_increment AS auto_increment_increment,
//@@character_set_client AS character_set_client,
//@@character_set_connection AS character_set_connection,
//@@character_set_results AS character_set_results,
//@@character_set_server AS character_set_server,
//@@collation_server AS collation_server,
//@@collation_connection AS collation_connection,
//@@init_connect AS init_connect,
//@@interactive_timeout AS interactive_timeout,
//@@license AS license,
//@@lower_case_table_names AS lower_case_table_names,
//@@max_allowed_packet AS max_allowed_packet,  //////////
//@@net_buffer_length AS net_buffer_length,
//@@net_write_timeout AS net_write_timeout,
//@@query_cache_size AS query_cache_size,
//@@query_cache_type AS query_cache_type,
//@@sql_mode AS sql_mode,
//@@system_time_zone AS system_time_zone,
//@@time_zone AS time_zone,
//@@transaction_isolation AS transaction_isolation,
//@@wait_timeout AS wait_timeout

// mysql jdbc 8.0.16
// /* mysql-connector-java-8.0.16 (Revision: 34cbc6bc61f72836e26327537a432d6db7c77de6) */
// SELECT
// @@session.auto_increment_increment AS auto_increment_increment,
// @@character_set_client AS character_set_client,
// @@character_set_connection AS character_set_connection,
// @@character_set_results AS character_set_results,
// @@character_set_server AS character_set_server,
// @@collation_server AS collation_server,
// @@collation_connection AS collation_connection,
// @@init_connect AS init_connect,
// @@interactive_timeout AS interactive_timeout,
// @@license AS license,
// @@lower_case_table_names AS lower_case_table_names,
// @@max_allowed_packet AS max_allowed_packet,  ///
// @@net_write_timeout AS net_write_timeout,
// @@performance_schema AS performance_schema,
// @@query_cache_size AS query_cache_size,
// @@query_cache_type AS query_cache_type,
// @@sql_mode AS sql_mode,
// @@system_time_zone AS system_time_zone,
// @@time_zone AS time_zone,
// @@transaction_isolation AS transaction_isolation,
// @@wait_timeout AS wait_timeout;
// void connectorj_select_variables(THD* thd, std::string query)

void handle_command_query(THD *thd, const COM_DATA *com_data)
{
    /*
    if (strstr(query.c_str(), "set autocommit")) // set autocommit=1
    {
      printf("ignore command: %s\n", query.c_str());
      thd->get_protocol_classic()->start_row();
      const char stateName[] = "autocommit";
      const char stateValue[] = "ON";

      ulonglong nameLen, valueLen;
      nameLen = strlen(stateName);
      valueLen = strlen(stateValue);

      String session_state_info;
      session_state_info.append((const uchar*)"\0", 1);
      uchar stateInfoLen = nameLen + net_length_size(nameLen) + valueLen + net_length_size(valueLen);
      session_state_info.append((uchar*)&stateInfoLen, 1);
      store_lenenc_string(session_state_info, (const uchar*)stateName, strlen(stateName));
      store_lenenc_string(session_state_info, (const uchar*)stateValue, strlen(stateValue));

      thd->get_protocol_classic()->send_ok(thd->server_status, 0, 0, 0, "", (const char*)session_state_info.c_str(), stateInfoLen + 2);
      thd->get_stmt_da()->disable_status();
      return;
    }
    else if (strstr(query.c_str(), "set sql_mode"))
    {
        printf("ignore command: %s\n", query.c_str());
        thd->get_protocol_classic()->start_row();
        const char stateName[] = "sql_mode";
        const char stateValue[] = "ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION";

        ulonglong nameLen, valueLen;
        nameLen = strlen(stateName);
        valueLen = strlen(stateValue);

        String session_state_info;
        session_state_info.append((const uchar*)"\0", 1);
        uchar stateInfoLen = nameLen + net_length_size(nameLen) + valueLen + net_length_size(valueLen);
        session_state_info.append((uchar*)&stateInfoLen, 1);
        store_lenenc_string(session_state_info, (const uchar*)stateName, strlen(stateName));
        store_lenenc_string(session_state_info, (const uchar*)stateValue, strlen(stateValue));

        thd->get_protocol_classic()->send_ok(thd->server_status, 0, 0, 0, "", (const char*)session_state_info.c_str(), stateInfoLen + 2);
        thd->get_stmt_da()->disable_status();
        return;
    }
    else if (strstr(query.c_str(), "set character_set_results = ")) // SET character_set_results = NULL
    {
      printf("ignore command: %s\n", query.c_str());
      thd->get_protocol_classic()->start_row();
      my_ok(thd);
      return;
    }
    else if (strstr(query.c_str(), "set sql_safe_updates")) // SET SQL_SAFE_UPDATES=1
    {
      printf("ignore command: %s\n", query.c_str());
      thd->get_protocol_classic()->start_row();
      const char stateName[] = "sql_safe_updates";
      const char stateValue[] = "ON";

      ulonglong nameLen, valueLen;
      nameLen = strlen(stateName);
      valueLen = strlen(stateValue);

      String session_state_info;
      session_state_info.append((const uchar*)"\0", 1);
      uchar stateInfoLen = nameLen + net_length_size(nameLen) + valueLen + net_length_size(valueLen);
      session_state_info.append((uchar*)&stateInfoLen, 1);
      store_lenenc_string(session_state_info, (const uchar*)stateName, strlen(stateName));
      store_lenenc_string(session_state_info, (const uchar*)stateValue, strlen(stateValue));

      thd->get_protocol_classic()->send_ok(thd->server_status, 0, 0, 0, "", (const char*)session_state_info.c_str(), stateInfoLen + 2);
      thd->get_stmt_da()->disable_status();
      return;
    }
    else if (strstr(query.c_str(), "set session")) // SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ
    {
      printf("ignore command: %s\n", query.c_str());
      my_ok(thd);
      return;
    }
    else if (strstr(query.c_str(), "set character set")
    || strstr(query.c_str(), "set names")) // SET CHARACTER SET utf8 or SET NAMES utf8
    {
      printf("ignore command: %s\n", query.c_str());
      const char charSetConn[] = "character_set_connection";
      const char charSetClient[] = "character_set_client";
      const char charSetRes[] = "character_set_results";
      int namelen1 = strlen(charSetConn);
      int namelen2 = strlen(charSetClient);
      int namelen3 = strlen(charSetRes);
      std::string charSet;
      int spaceIdx = query.find_last_of(" ");
      if (spaceIdx >= 0)
      {
        charSet = query.substr(spaceIdx + 1);
      }
      if (charSet.empty())
      {
        my_error(ER_UNKNOWN_ERROR, MYF(0));
        return;
      }
      String session_state_info;
      session_state_info.append((const uchar*)"\0", 1);
      uchar stateInfoLen = namelen1 + net_length_size(namelen1) +
                           namelen2 + net_length_size(namelen2) +
                           namelen3 + net_length_size(namelen3) +
                           3 * (charSet.size() + net_length_size(charSet.size()));
      uchar nullChar = '\0';
      int charSetTotalLen = charSet.size() + net_length_size(charSet.size());

      int len = namelen1 + net_length_size(namelen1) + charSetTotalLen;
      session_state_info.append((uchar*)&len, 1);
      store_lenenc_string(session_state_info, (const uchar*)charSetConn, namelen1);
      store_lenenc_string(session_state_info, (const uchar*)charSet.c_str(), charSet.size());
      session_state_info.append((uchar*)&nullChar, 1);

      len = namelen2 + net_length_size(namelen2) + charSetTotalLen;
      session_state_info.append((uchar*)&len, 1);
      store_lenenc_string(session_state_info, (const uchar*)charSetClient, namelen2);
      store_lenenc_string(session_state_info, (const uchar*)charSet.c_str(), charSet.size());
      session_state_info.append((uchar*)&nullChar, 1);

      len = namelen3 + net_length_size(namelen3) + charSetTotalLen;
      session_state_info.append((uchar*)&len, 1);
      store_lenenc_string(session_state_info, (const uchar*)charSetRes, namelen3);
      store_lenenc_string(session_state_info, (const uchar*)charSet.c_str(), charSet.size());

      thd->get_protocol_classic()->send_ok(thd->server_status, 0, 0, 0, "", (const char*)session_state_info.c_str(), stateInfoLen + 1 + 5);
      thd->get_stmt_da()->disable_status();
      return;
    }
    */

    std::string query = com_data->com_query.query;

    aries::SQLExecutor::GetInstance()->ExecuteSQL(query, thd->db(), true);
}
void handleSQLResult(THD* thd, SQLResultPtr sqlResultPtr, SendDataType type) {
    if (sqlResultPtr->IsSuccess()) {
        if (sqlResultPtr->GetResults().size() > 0) {
            if ( !handleQueryResult( thd, sqlResultPtr->GetResults()[ 0 ], type) )
                my_eof( thd );
        } else {
            // my_ok(thd, sqlResultPtr->GetAffectedRowKeys().size(), 0);
        }
    } else {
        my_error2(sqlResultPtr->GetErrorCode(), MYF(0), sqlResultPtr->GetErrorMessage().data());
    }
}

void handle_command_field_list(THD *thd, const COM_DATA *com_data)
{
  shared_ptr<DatabaseEntry> dbEntry = SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(thd->db());
  shared_ptr<TableEntry> table = dbEntry->GetTableByName((char*)com_data->com_field_list.table_name);
  if (!table) {
      my_eof(thd);
      return;
  }
  const vector<shared_ptr<ColumnEntry>> columns = table->GetColumns();
  vector<Send_field *> fields;

  for ( const auto& col : columns )
  {
      Send_field *field = new Send_field();
      field->db_name = "DB NAME";
      field->table_name = field->org_table_name = "TABLE NAME";
      field->col_name = col->GetName();
      field->org_col_name = field->col_name;
      field->has_default = col->HasDefault();
      if ( field->has_default )
        field->default_val_str = col->GetConvertedDefaultValue();

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

      if (!col->IsAllowNull()) {
          field->flags |= NOT_NULL_FLAG;
      }

      auto valueType = col->GetType();
      switch ( valueType ) {
          case ColumnType::UNKNOWN:
              field->type = MYSQL_TYPE_NULL;
              field->length = 4;
              break;
          case ColumnType::BOOL:
          case ColumnType::TINY_INT:
              field->type = MYSQL_TYPE_TINY;
              field->length = 4;
              break;
          case ColumnType::SMALL_INT:
              field->type = MYSQL_TYPE_SHORT;
              field->length = 6;
              break;
          case ColumnType::INT:
              field->type = MYSQL_TYPE_LONG;
              field->length = 11;
              break;
          case ColumnType::LONG_INT:
              field->type = MYSQL_TYPE_LONGLONG;
              field->length = 20;
              break;
          case ColumnType::DECIMAL:
              field->type = MYSQL_TYPE_NEWDECIMAL;
              field->length = 11; // for type: decimal
              field->decimals = 31;
              break;
          case ColumnType::FLOAT:
              field->type = MYSQL_TYPE_FLOAT;
              field->length = 12;
              field->decimals = 0x1f;
              break;
          case ColumnType::DOUBLE:
              field->type = MYSQL_TYPE_DOUBLE;
              field->length = 22;
              field->decimals = 0x1f;
              break;
          case ColumnType::TEXT:
              field->type = MYSQL_TYPE_STRING;
              field->length = col->GetTypeSize() * 3; // for char(256): 256 * 3
              break;
          case ColumnType::DATE:
              field->type = MYSQL_TYPE_DATE;
              field->length = 10;
              field->flags |= BINARY_FLAG;
              break;
          case ColumnType::TIME:
              field->type = MYSQL_TYPE_TIME;
              field->length = 10;
              field->flags |= BINARY_FLAG;
              break;
          case ColumnType::DATE_TIME:
              field->type = MYSQL_TYPE_DATETIME;
              field->length = 19;
              field->flags |= BINARY_FLAG;
              break;
          case ColumnType::TIMESTAMP:
              field->type = MYSQL_TYPE_TIMESTAMP;
              field->length = 19;
              field->flags |= (NOT_NULL_FLAG | BINARY_FLAG | TIMESTAMP_FLAG);
              break;
          case ColumnType::YEAR:
              field->type = MYSQL_TYPE_YEAR;
              field->length = 4;
              field->flags |= (UNSIGNED_FLAG | ZEROFILL_FLAG);
              break;
          default:
              LOG(WARNING) << "not supported data type: " << (int32_t) valueType;
              field->type = MYSQL_TYPE_NULL;
              break;
      }
      fields.insert(fields.end(), field);
  }
  thd->get_protocol_classic()->start_row();
  bool res = thd->send_result_metadata(&fields, Protocol_classic::SEND_DEFAULTS);
  for (Send_field *field : fields)
  {
    delete field;
  }
  if (!res)
      my_eof(thd);
}

/**
  Perform one connection-level (COM_XXXX) command.

  @param thd             connection handle
  @param command         type of command to perform
  @com_data              com_data union to store the generated command

  @todo
    set thd->lex->sql_command to SQLCOM_END here.
  @todo
    The following has to be changed to an 8 byte integer

  @retval
    0   ok
  @retval
    1   request of thread shutdown, i. e. if command is
        COM_QUIT/COM_SHUTDOWN
*/
bool dispatch_command(THD *thd, const COM_DATA *com_data,
                      enum enum_server_command command)
{
  bool error= 0;
  Global_THD_manager *thd_manager= Global_THD_manager::get_instance();

  thd->set_command(command);

  thd->set_time();

  thd->set_query_id(next_query_id());
   thd_manager->inc_thread_running();

  /**
    Clear the set of flags that are expected to be cleared at the
    beginning of each command.
  */
  thd->server_status&= ~SERVER_STATUS_CLEAR_SET;

  switch (command)
  {
  case COM_QUIT:
    LOG(INFO) << "Connection " << thd->thread_id() << " got COM_QUIT";
    DBUG_RETURN(1);
    break;
  case COM_INIT_DB:
  {
    // LEX_STRING tmp;
    // thd->convert_string(&tmp, system_charset_info,
    //                     com_data->com_init_db.db_name,
    //                     com_data->com_init_db.length, thd->charset());
    LOG(INFO) << "Connection " << thd->thread_id() << " Got COM_INIT_DB: " << com_data->com_init_db.db_name;

    // com_init_db.db_name is not null terminated
    string newDb(com_data->com_init_db.db_name, com_data->com_init_db.length);
    char nullChar = '\0';
    newDb.append(&nullChar);
    if (!mysql_change_db(thd, newDb, false))
    {
        my_ok(thd);
    }
    break;
  }
  case COM_QUERY:
    DLOG(INFO) << "Connection " << thd->thread_id() << " got COM_QUERY.";
    // handle_command_query(thd, com_data);

    aries::SQLExecutor::GetInstance()->ExecuteSQL(com_data->com_query.query, thd->db(), true);
    break;
  case COM_STMT_PREPARE:
  {
      mysqld_stmt_prepare(thd, com_data->com_stmt_prepare.query,
                          com_data->com_stmt_prepare.length);
      break;
  }
  case COM_STMT_SEND_LONG_DATA: {
      mysql_stmt_get_longdata(thd, com_data->com_stmt_send_long_data.stmt_id,
                              com_data->com_stmt_send_long_data.param_number,
                              com_data->com_stmt_send_long_data.longdata,
                              com_data->com_stmt_send_long_data.length);
      break;
  }
  case COM_STMT_EXECUTE: {
      mysqld_stmt_execute(thd, com_data->com_stmt_execute.stmt_id,
                          com_data->com_stmt_execute.flags,
                          com_data->com_stmt_execute.params,
                          com_data->com_stmt_execute.params_length);
      break;
  }
  case COM_STMT_FETCH:
  {
    mysqld_stmt_fetch(thd, com_data->com_stmt_fetch.stmt_id,
                      com_data->com_stmt_fetch.num_rows);
    break;
  }
  case COM_STMT_CLOSE: {
      mysqld_stmt_close(thd, com_data->com_stmt_close.stmt_id);
      break;
  }
  case COM_STMT_RESET: {
      mysqld_stmt_reset(thd, com_data->com_stmt_reset.stmt_id);
      break;
  }
  case COM_FIELD_LIST:
    handle_command_field_list(thd, com_data);
    break;
  case COM_PROCESS_KILL: {
    std::string query = "kill " + std::to_string(com_data->com_kill.id);
    aries::SQLExecutor::GetInstance()->ExecuteSQL(query, thd->db(), true);
    break;
  }
  case COM_PROCESS_INFO: {
    std::string query = "show processlist";
    aries::SQLExecutor::GetInstance()->ExecuteSQL(query, thd->db(), true);
    break;
  }
  case COM_REFRESH: {
    DLOG(INFO) << "Connection " << thd->thread_id() << " got COM_REFRESH.";
    break;
  }
  case COM_PING:
      my_ok(thd);
    break;
  case  COM_CHANGE_USER: {
    DLOG(INFO) << "Connection " << thd->thread_id() << " got COM_CHANGE_USER.";
    // int error;
    thd->cleanup_connection();
    error= acl_authenticate(thd, COM_CHANGE_USER);

    // if(error){
    //   // thd->killed = THD::KILL_CONNECTION;
    //   DBUG_RETURN(error);
    // }
    break;
  }
  case COM_SET_OPTION:
  {
    // thd->status_var.com_stat[SQLCOM_SET_OPTION]++;

    switch (com_data->com_set_option.opt_command) {
    case (int) MYSQL_OPTION_MULTI_STATEMENTS_ON:
      //TODO: access of protocol_classic should be removed
      thd->get_protocol_classic()->add_client_capability(
          CLIENT_MULTI_STATEMENTS);
      my_eof(thd);
      break;
    case (int) MYSQL_OPTION_MULTI_STATEMENTS_OFF:
      thd->get_protocol_classic()->remove_client_capability(
        CLIENT_MULTI_STATEMENTS);
      my_eof(thd);
      break;
    default:
      my_message(ER_UNKNOWN_COM_ERROR, ER(ER_UNKNOWN_COM_ERROR), MYF(0));
      break;
    }
    break;
  }
  case COM_RESET_CONNECTION:
    thd->cleanup_connection();
    my_ok( thd );
    break;
  case COM_STATISTICS:
  {
      ulong uptime;
      size_t length MY_ATTRIBUTE((unused));
      ulonglong queries_per_second1000;
      char buff[250];
      size_t buff_len= sizeof(buff);

      if (!(uptime= (ulong) (thd->start_time.tv_sec - server_start_time)))
          queries_per_second1000= 0;
      else
          queries_per_second1000= thd->query_id * 1000LL / uptime;

      length = snprintf(buff, buff_len - 1,
                  "Uptime: %lu  Threads: %d  Questions: %lu  "
                  "Slow queries: %llu  Opens: %llu  Flush tables: %lu  "
                  "Open tables: %u  Queries per second avg: %u.%03u",
                  uptime,
                  (int) thd_manager->get_thd_count(), (ulong) thd->query_id,
                  (ulonglong) 0,// current_global_status_var.long_query_count,
                  (ulonglong) 0, //current_global_status_var.opened_tables,
                  (ulong) 0, // uint)refresh_version,
                  (uint) 0, // uint)table_cache_manager.cached_tables(),
                  (uint) (queries_per_second1000 / 1000),
                  (uint) (queries_per_second1000 % 1000));
      // TODO: access of protocol_classic should be removed.
      // should be rewritten using store functions
      thd->get_protocol_classic()->write((uchar*) buff, length);
      thd->get_protocol_classic()->flush_net();
      thd->get_stmt_da()->disable_status();
      break;
  }
  case COM_SHUTDOWN: {
      if(!shutdown(thd, SHUTDOWN_DEFAULT, command))
          break;
      error= TRUE;
      break;
  }
  default:
    LOG(ERROR) << "Connection " << thd->thread_id() << " got command not implemented: " <<  command;
    my_error(ER_UNKNOWN_COM_ERROR, MYF(0));
    break;
  }

  /* Finalize server status flags after executing a command. */
  // thd->update_server_status();
  // if (thd->killed)
  //   thd->send_kill_message();
  thd->send_statement_status();

    if (thd->killed == THD::KILL_QUERY ||
        thd->killed == THD::KILL_TIMEOUT ||
        thd->killed == THD::KILL_BAD_DATA)
    {
        thd->killed= THD::NOT_KILLED;
    }

    thd_manager->dec_thread_running();

    DBUG_RETURN(error);
}

/**
  kill on thread.

  @param thd			Thread class
  @param id			Thread id
  @param only_kill_query        Should it kill the query or the connection

  @note
    This is written such that we have a short lock on LOCK_thd_list
*/


static uint kill_one_thread(THD *thd, my_thread_id id, bool only_kill_query)
{
    THD *tmp= NULL;
    uint error=ER_NO_SUCH_THREAD;

    LOG(INFO) << "Killing thread " << id << ", only_kill: " << only_kill_query;

    DBUG_ENTER("kill_one_thread");
    // DBUG_PRINT("enter", ("id=%u only_kill=%d", id, only_kill_query));
    tmp= Global_THD_manager::get_instance()->find_thd(id);
    if (tmp)
    {
        /*
          If we're SUPER, we can KILL anything, including system-threads.
          No further checks.

          KILLer: thd->m_security_ctx->user could in theory be NULL while
          we're still in "unauthenticated" state. This is a theoretical
          case (the code suggests this could happen, so we play it safe).

          KILLee: tmp->m_security_ctx->user will be NULL for system threads.
          We need to check so Jane Random User doesn't crash the server
          when trying to kill a) system threads or b) unauthenticated users'
          threads (Bug#43748).

          If user of both killer and killee are non-NULL, proceed with
          slayage if both are string-equal.
        */

        // if ((thd->security_context()->check_access(SUPER_ACL)) ||
        //     thd->security_context()->user_matches(tmp->security_context()))
        // {
            /* process the kill only if thread is not already undergoing any kill
               connection.
            */
            if (tmp->killed != THD::KILL_CONNECTION)
            {
                tmp->awake(only_kill_query ? THD::KILL_QUERY : THD::KILL_CONNECTION);
            }
            error= 0;
        // }
        // else
        //     error=ER_KILL_DENIED_ERROR;
        // mysql_mutex_unlock(&tmp->LOCK_thd_data);
    }
    // DEBUG_SYNC(thd, "kill_thd_end");
    // DBUG_PRINT("exit", ("%d", error));
    DBUG_RETURN(error);
}


/*
  kills a thread and sends response

  SYNOPSIS
    sql_kill()
    thd			Thread class
    id			Thread id
    only_kill_query     Should it kill the query or the connection
*/

void sql_kill(THD *thd, my_thread_id id, bool only_kill_query)
{
    uint error;
    if (!(error= kill_one_thread(thd, id, only_kill_query)))
    {
        if (! thd->killed)
            my_ok(thd);
    }
    else
        my_error(error, MYF(0), id);
}