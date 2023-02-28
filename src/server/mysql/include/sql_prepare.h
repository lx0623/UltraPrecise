#ifndef SQL_PREPARE_H
#define SQL_PREPARE_H
/* Copyright (c) 2009, 2015, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   51 Franklin Street, Suite 500, Boston, MA 02110-1335 USA */

#include <string>
#include <vector>
#include <memory>
#include "frontend/SQLResult.h"
#include "frontend/AriesSQLStatement.h"
#include "frontend/PreparedStmtStructure.h"
#include "mysql_com.h"
#include "my_global.h"

using std::string;
using std::vector;
using aries::AriesSQLStatementPointer;
using aries::PreparedStmtStructurePtr;
struct LEX;
class THD;

/**
  An interface that is used to take an action when
  the locking module notices that a table version has changed
  since the last execution. "Table" here may refer to any kind of
  table -- a base table, a temporary table, a view or an
  information schema table.

  When we open and lock tables for execution of a prepared
  statement, we must verify that they did not change
  since statement prepare. If some table did change, the statement
  parse tree *may* be no longer valid, e.g. in case it contains
  optimizations that depend on table metadata.

  This class provides an interface (a method) that is
  invoked when such a situation takes place.
  The implementation of the method simply reports an error, but
  the exact details depend on the nature of the SQL statement.

  At most 1 instance of this class is active at a time, in which
  case THD::m_reprepare_observer is not NULL.

  @sa check_and_update_table_version() for details of the
  version tracking algorithm 

  @sa Open_tables_state::m_reprepare_observer for the life cycle
  of metadata observers.
*/

class Reprepare_observer
{
public:
  /**
    Check if a change of metadata is OK. In future
    the signature of this method may be extended to accept the old
    and the new versions, but since currently the check is very
    simple, we only need the THD to report an error.
  */
  bool report_error(THD *thd);
  bool is_invalidated() const { return m_invalidated; }
  void reset_reprepare_observer() { m_invalidated= 0; }
private:
  bool m_invalidated;
};


void mysqld_stmt_prepare(THD *thd, const char *query, uint length);
aries::SQLResultPtr mysqld_stmt_execute(THD *thd, ulong stmt_id, ulong flags, uchar *params,
                                        ulong params_length);
void mysqld_stmt_close(THD *thd, ulong stmt_id);
void mysql_sql_stmt_prepare(THD *thd, const PreparedStmtStructurePtr preparedStmtStructurePtr);
void mysql_sql_stmt_execute(THD *thd, const PreparedStmtStructurePtr preparedStmtStructurePtr);
void mysql_sql_stmt_close(THD *thd, const PreparedStmtStructurePtr preparedStmtStructurePtr);
void mysqld_stmt_fetch(THD *thd, ulong stmt_id, ulong num_rows);
void mysqld_stmt_reset(THD *thd, ulong stmt_id);
void mysql_stmt_get_longdata(THD *thd, ulong stmt_id, uint param_number,
                             uchar *longdata, ulong length);
bool reinit_stmt_before_use(THD *thd, LEX *lex);
bool select_like_stmt_cmd_test(THD *thd,
                               class Sql_cmd_dml *cmd,
                               ulong setup_tables_done_option);

/**
  Execute a fragment of server code in an isolated context, so that
  it doesn't leave any effect on THD. THD must have no open tables.
  The code must not leave any open tables around.
  The result of execution (if any) is stored in Ed_result.
*/

class Server_runnable
{
public:
  virtual bool execute_server_code(THD *thd)= 0;
  virtual ~Server_runnable();
};


/**
  Prepared_statement: a statement that can contain placeholders.
*/

class Prepared_statement
{
  enum flag_values
  {
    IS_IN_USE= 1,
    IS_SQL_PREPARE= 2
  };

public:
  THD *thd;
  vector< std::shared_ptr<aries::AbstractBiaodashi> > param_array;
  uint param_count;
  uint last_errno;
  char last_error[MYSQL_ERRMSG_SIZE];

  /*
    Uniquely identifies each statement object in thread scope; change during
    statement lifetime.
  */
  const ulong id;

  /**
    The query associated with this statement.
  */
  string m_query_string;

private:
  // Query_fetch_protocol_binary result;
  uint flags;
  bool with_log;
  string m_name; /* name for named prepared statements */
  /**
    Name of the current (default) database.

    If there is the current (default) database, "db" contains its name. If
    there is no current (default) database, "db" is NULL and "db_length" is
    0. In other words, "db", "db_length" must either be NULL, or contain a
    valid database name.

    @note this attribute is set and alloced by the slave SQL thread (for
    the THD of that thread); that thread is (and must remain, for now) the
    only responsible for freeing this member.
  */
  string m_db;
  std::vector<AriesSQLStatementPointer> m_statements;
  // 缓存mysqld_stmt_execute的结果供mysqld_stmt_fetch使用
  aries::SQLResultPtr cache_result;
  // 记录mysqld_stmt_fetch已经取了多少行
  ulong offset_row;
  bool cursor_is_open;

public:
  Prepared_statement(THD *thd_arg);
  void cleanup_stmt();
  void set_name(const string &name) {
      m_name = name;
  }
  const string &name() const
  { return m_name; }
  void close_cursor(){cursor_is_open=false;};
  void open_cursor(){cursor_is_open=true;};
  bool is_cursor_open(){ return cursor_is_open == true; }
  bool is_in_use() const { return flags & (uint) IS_IN_USE; }
  bool is_sql_prepare() const { return flags & (uint) IS_SQL_PREPARE; }
  void set_sql_prepare() { flags|= (uint) IS_SQL_PREPARE; }
  aries::SQLResultPtr prepare(const char *packet, size_t packet_length);
  void set_aries_statements(std::vector<AriesSQLStatementPointer> statements) {
      m_statements = statements;
  }
  const std::vector<AriesSQLStatementPointer>& get_aries_statements() const {
      return m_statements;
  }
  aries::SQLResultPtr
  execute_loop(// string *expanded_query,
               bool open_cursor,
               uchar *packet_arg, uchar *packet_end_arg,
               std::vector<std::string>* executeVars);
  void set_cache_result(aries::SQLResultPtr result, size_t offset){
    cache_result = result; offset_row=offset; 
  }
  std::pair<aries::SQLResultPtr, size_t > get_cache_result(){
    return {cache_result, offset_row};
  }
  
private:
    bool set_parameters(// string *expanded_query,
            uchar *packet, uchar *packet_end,
            std::vector<std::string>* executeVars);
  bool execute(string *expanded_query, bool open_cursor);
};

using Prepared_statement_ptr = std::shared_ptr<Prepared_statement>;

#endif // SQL_PREPARE_H
