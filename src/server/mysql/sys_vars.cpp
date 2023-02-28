//
// Created by tengjp on 19-8-13.
//
#include "server/mysql/include/my_sqlcommand.h"
#include "server/mysql/include/sql_const.h"
#include "server/mysql/include/sql_time.h"
#include "server/mysql/include/mysqld.h"
#include "server/mysql/include/mysql_version.h"
#include "server/mysql/include/my_config.h"
#include "server/mysql/include/sys_vars.h"
#include "server/mysql/include/pfs_server.h"
#include "server/mysql/include/derror.h"
#include "server/mysql/include/mysql_def.h"
#include "server/mysql/include/my_base.h"
#include "utils/string_util.h"

using namespace mysql;

template <typename T, typename R>
R get_sys_var_value(sys_var* var, enum_var_type type) {
    R value = *(R*) var->value_ptr(current_thd, type);
    return value;
}
template <>
uchar* get_sys_var_value<uchar*, uchar*>(sys_var* var, enum_var_type type) {
    uchar* value = var->value_ptr(current_thd, type);
    return value;
}
template <>
uchar* get_sys_var_value<uchar**, uchar*>(sys_var* var, enum_var_type type) {
    return *(uchar**)var->value_ptr(current_thd, type);
}

template bool get_sys_var_value<bool, bool>(sys_var*, enum_var_type);
template my_bool get_sys_var_value<my_bool, my_bool>(sys_var*, enum_var_type);
template int get_sys_var_value<int, int>(sys_var*, enum_var_type);
template long get_sys_var_value<long, long>(sys_var*, enum_var_type);
template ulong get_sys_var_value<ulong, ulong>(sys_var*, enum_var_type);
template ulonglong get_sys_var_value<ulonglong, ulonglong>(sys_var*, enum_var_type);
template double get_sys_var_value<double, double>(sys_var*, enum_var_type);

template <typename T, typename R>
R get_sys_var_value(const char* varName, enum_var_type scope )
{
    sys_var* sysVar = find_sys_var( varName );
    if (!sysVar)
    {
        ARIES_EXCEPTION(ER_UNKNOWN_SYSTEM_VARIABLE, varName );
    }
    R varValue = ( R )get_sys_var_value< T >( sysVar, scope );
    return varValue;
}

template my_bool get_sys_var_value<my_bool, my_bool>( const char*, enum_var_type );

#define PFS_TRAILING_PROPERTIES \
  NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL), ON_UPDATE(NULL), \
  NULL, sys_var::PARSE_EARLY

/**
mysql server 5.7.26 default charsets:
mysql> show global variables like "character_set%";
+--------------------------+-----------------------------------------+
| Variable_name            | Value                                   |
+--------------------------+-----------------------------------------+
| character_set_client     | latin1                                  |
| character_set_connection | latin1                                  |
| character_set_database   | latin1                                  |
| character_set_filesystem | binary                                  |
| character_set_results    | latin1                                  |
| character_set_server     | latin1                                  |
| character_set_system     | utf8                                    |
| character_sets_dir       | /usr/local/mysql-5.7.26/share/charsets/ |
+--------------------------+-----------------------------------------+
8 rows in set (0.01 sec)

mysql> show session variables like "character_set%";
+--------------------------+-----------------------------------------+
| Variable_name            | Value                                   |
+--------------------------+-----------------------------------------+
| character_set_client     | utf8                                    |
| character_set_connection | utf8                                    |
| character_set_database   | latin1                                  |
| character_set_filesystem | binary                                  |
| character_set_results    | utf8                                    |
| character_set_server     | latin1                                  |
| character_set_system     | utf8                                    |
| character_sets_dir       | /usr/local/mysql-5.7.26/share/charsets/ |
+--------------------------+-----------------------------------------+
8 rows in set (0.01 sec)
 */
static Sys_var_mybool Sys_pfs_enabled(
        "performance_schema",
        "Enable the performance schema.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_enabled),
        // /* CMD_LINE(OPT_ARG),*/
        (TRUE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_charptr Sys_pfs_instrument(
        "performance_schema_instrument",
        "Default startup value for a performance schema instrument.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_pfs_instrument),
        // CMD_LINE(OPT_ARG, OPT_PFS_INSTRUMENT),
        IN_FS_CHARSET,
        (""),
        PFS_TRAILING_PROPERTIES);
static Sys_var_mybool Sys_pfs_consumer_events_stages_current(
        "performance_schema_consumer_events_stages_current",
        "Default startup value for the events_stages_current consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_stages_current_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_stages_history(
        "performance_schema_consumer_events_stages_history",
        "Default startup value for the events_stages_history consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_stages_history_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_stages_history_long(
        "performance_schema_consumer_events_stages_history_long",
        "Default startup value for the events_stages_history_long consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_stages_history_long_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_statements_current(
        "performance_schema_consumer_events_statements_current",
        "Default startup value for the events_statements_current consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_statements_current_enabled),
        /* CMD_LINE(OPT_ARG),*/ (TRUE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_statements_history(
        "performance_schema_consumer_events_statements_history",
        "Default startup value for the events_statements_history consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_statements_history_enabled),
        /* CMD_LINE(OPT_ARG),*/ (TRUE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_statements_history_long(
        "performance_schema_consumer_events_statements_history_long",
        "Default startup value for the events_statements_history_long consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_statements_history_long_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_transactions_current(
        "performance_schema_consumer_events_transactions_current",
        "Default startup value for the events_transactions_current consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_transactions_current_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_transactions_history(
        "performance_schema_consumer_events_transactions_history",
        "Default startup value for the events_transactions_history consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_transactions_history_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_transactions_history_long(
        "performance_schema_consumer_events_transactions_history_long",
        "Default startup value for the events_transactions_history_long consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_transactions_history_long_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_waits_current(
        "performance_schema_consumer_events_waits_current",
        "Default startup value for the events_waits_current consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_waits_current_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_waits_history(
        "performance_schema_consumer_events_waits_history",
        "Default startup value for the events_waits_history consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_waits_history_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_events_waits_history_long(
        "performance_schema_consumer_events_waits_history_long",
        "Default startup value for the events_waits_history_long consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_events_waits_history_long_enabled),
        /* CMD_LINE(OPT_ARG),*/ (FALSE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_global_instrumentation(
        "performance_schema_consumer_global_instrumentation",
        "Default startup value for the global_instrumentation consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_global_instrumentation_enabled),
        /* CMD_LINE(OPT_ARG),*/ (TRUE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_thread_instrumentation(
        "performance_schema_consumer_thread_instrumentation",
        "Default startup value for the thread_instrumentation consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_thread_instrumentation_enabled),
        /* CMD_LINE(OPT_ARG),*/ (TRUE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_mybool Sys_pfs_consumer_statement_digest(
        "performance_schema_consumer_statements_digest",
        "Default startup value for the statements_digest consumer.",
        READ_ONLY NOT_VISIBLE GLOBAL_VAR(pfs_param.m_consumer_statement_digest_enabled),
        /* CMD_LINE(OPT_ARG),*/ (TRUE),
        PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_waits_history_long_size(
        "performance_schema_events_waits_history_long_size",
        "Number of rows in EVENTS_WAITS_HISTORY_LONG."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_waits_history_long_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_waits_history_size(
        "performance_schema_events_waits_history_size",
        "Number of rows per thread in EVENTS_WAITS_HISTORY."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_waits_history_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_cond_classes(
        "performance_schema_max_cond_classes",
        "Maximum number of condition instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_cond_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_COND_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_cond_instances(
        "performance_schema_max_cond_instances",
        "Maximum number of instrumented condition objects."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_cond_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_program_instances(
        "performance_schema_max_program_instances",
        "Maximum number of instrumented programs."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_program_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_prepared_stmt_instances(
        "performance_schema_max_prepared_statements_instances",
        "Maximum number of instrumented prepared statements."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_prepared_stmt_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_file_classes(
        "performance_schema_max_file_classes",
        "Maximum number of file instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_file_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_FILE_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_file_handles(
        "performance_schema_max_file_handles",
        "Maximum number of opened instrumented files.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_file_handle_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 1024*1024),
        (PFS_MAX_FILE_HANDLE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_file_instances(
        "performance_schema_max_file_instances",
        "Maximum number of instrumented files."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_file_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_sockets(
        "performance_schema_max_socket_instances",
        "Maximum number of opened instrumented sockets."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_socket_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_socket_classes(
        "performance_schema_max_socket_classes",
        "Maximum number of socket instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_socket_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_SOCKET_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_mutex_classes(
        "performance_schema_max_mutex_classes",
        "Maximum number of mutex instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_mutex_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_MUTEX_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_mutex_instances(
        "performance_schema_max_mutex_instances",
        "Maximum number of instrumented MUTEX objects."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_mutex_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 100*1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_rwlock_classes(
        "performance_schema_max_rwlock_classes",
        "Maximum number of rwlock instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_rwlock_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_RWLOCK_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_rwlock_instances(
        "performance_schema_max_rwlock_instances",
        "Maximum number of instrumented RWLOCK objects."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_rwlock_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 100*1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_table_handles(
        "performance_schema_max_table_handles",
        "Maximum number of opened instrumented tables."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_table_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_table_instances(
        "performance_schema_max_table_instances",
        "Maximum number of instrumented tables."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_table_share_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_table_lock_stat(
        "performance_schema_max_table_lock_stat",
        "Maximum number of lock statistics for instrumented tables."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_table_lock_stat_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_index_stat(
        "performance_schema_max_index_stat",
        "Maximum number of index statistics for instrumented tables."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_index_stat_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_thread_classes(
        "performance_schema_max_thread_classes",
        "Maximum number of thread instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_thread_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_THREAD_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_thread_instances(
        "performance_schema_max_thread_instances",
        "Maximum number of instrumented threads."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_thread_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_setup_actors_size(
        "performance_schema_setup_actors_size",
        "Maximum number of rows in SETUP_ACTORS."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_setup_actor_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_setup_objects_size(
        "performance_schema_setup_objects_size",
        "Maximum number of rows in SETUP_OBJECTS."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_setup_object_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_accounts_size(
        "performance_schema_accounts_size",
        "Maximum number of instrumented user@host accounts."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_account_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_hosts_size(
        "performance_schema_hosts_size",
        "Maximum number of instrumented hosts."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_host_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_users_size(
        "performance_schema_users_size",
        "Maximum number of instrumented users."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_user_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_stage_classes(
        "performance_schema_max_stage_classes",
        "Maximum number of stage instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_stage_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        (PFS_MAX_STAGE_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_stages_history_long_size(
        "performance_schema_events_stages_history_long_size",
        "Number of rows in EVENTS_STAGES_HISTORY_LONG."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_stages_history_long_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_stages_history_size(
        "performance_schema_events_stages_history_size",
        "Number of rows per thread in EVENTS_STAGES_HISTORY."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_stages_history_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);
/**
  Number of PSI_statement_info instruments
  for internal stored programs statements.
*/
#define SP_PSI_STATEMENT_INFO_COUNT 1
/**
  Variable performance_schema_max_statement_classes.
  The default number of statement classes is the sum of:
  - COM_END for all regular "statement/com/...",
  - 1 for "statement/com/new_packet", for unknown enum_server_command
  - 1 for "statement/com/Error", for invalid enum_server_command
  - SQLCOM_END for all regular "statement/sql/...",
  - 1 for "statement/sql/error", for invalid enum_sql_command.
  - SP_PSI_STATEMENT_INFO_COUNT for "statement/sp/...".
  - 1 for "statement/rpl/relay_log", for replicated statements.
  - 1 for "statement/scheduler/event", for scheduled events.
*/
static Sys_var_ulong Sys_pfs_max_statement_classes(
        "performance_schema_max_statement_classes",
        "Maximum number of statement instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_statement_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 256),
        ((ulong) SQLCOM_END + (ulong) COM_END + 5 + SP_PSI_STATEMENT_INFO_COUNT),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_statements_history_long_size(
        "performance_schema_events_statements_history_long_size",
        "Number of rows in EVENTS_STATEMENTS_HISTORY_LONG."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_statements_history_long_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_statements_history_size(
        "performance_schema_events_statements_history_size",
        "Number of rows per thread in EVENTS_STATEMENTS_HISTORY."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_statements_history_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_statement_stack_size(
        "performance_schema_max_statement_stack",
        "Number of rows per thread in EVENTS_STATEMENTS_CURRENT.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_statement_stack_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(1, 256),
        (PFS_STATEMENTS_STACK_SIZE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_ulong Sys_pfs_max_memory_classes(
        "performance_schema_max_memory_classes",
        "Maximum number of memory pool instruments.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_memory_class_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 1024),
        (PFS_MAX_MEMORY_CLASS),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_digest_size(
        "performance_schema_digests_size",
        "Size of the statement digest."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_digest_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024 * 1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_transactions_history_long_size(
        "performance_schema_events_transactions_history_long_size",
        "Number of rows in EVENTS_TRANSACTIONS_HISTORY_LONG."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_transactions_history_long_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024*1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_events_transactions_history_size(
        "performance_schema_events_transactions_history_size",
        "Number of rows per thread in EVENTS_TRANSACTIONS_HISTORY."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_events_transactions_history_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_digest_length(
        "performance_schema_max_digest_length",
        "Maximum length considered for digest text, when stored in performance_schema tables.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_max_digest_length),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 1024 * 1024),
        (1024),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_connect_attrs_size(
        "performance_schema_session_connect_attrs_size",
        "Size of session attribute string buffer per thread."
        " Use 0 to disable, -1 for automated sizing.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_session_connect_attrs_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 1024 * 1024),
        (PFS_AUTOSIZE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_metadata_locks(
        "performance_schema_max_metadata_locks",
        "Maximum number of metadata locks."
        " Use 0 to disable, -1 for automated scaling.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_metadata_lock_sizing),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(-1, 100*1024*1024),
        (PFS_AUTOSCALE_VALUE),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);

static Sys_var_long Sys_pfs_max_sql_text_length(
        "performance_schema_max_sql_text_length",
        "Maximum length of displayed sql text.",
        READ_ONLY GLOBAL_VAR(pfs_param.m_max_sql_text_length),
        /* CMD_LINE(REQUIRED_ARG),*/ VALID_RANGE(0, 1024 * 1024),
        (1024),
        BLOCK_SIZE(1), PFS_TRAILING_PROPERTIES);
/// end of pfs

static Sys_var_ulong Sys_auto_increment_increment(
        "auto_increment_increment",
        "Auto-increment columns are incremented by this",
        SESSION_VAR(auto_increment_increment),
        // /* CMD_LINE(OPT_ARG),*/
        VALID_RANGE(1, 65535), (1), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, IN_BINLOG);

static Sys_var_ulong Sys_auto_increment_offset(
        "auto_increment_offset",
        "Offset added to Auto-increment columns. Used when "
        "auto-increment-increment != 1",
        SESSION_VAR(auto_increment_offset),
        // CMD_LINE(OPT_ARG),
        VALID_RANGE(1, 65535), (1), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, IN_BINLOG);

static Sys_var_mybool Sys_automatic_sp_privileges(
       "automatic_sp_privileges",
       "Creating and dropping stored procedures alters ACLs",
       GLOBAL_VAR(sp_automatic_privileges),
       // CMD_LINE(OPT_ARG),
       (TRUE));

static Sys_var_ulong Sys_back_log(
        "back_log", "The number of outstanding connection requests "
                    "MySQL can have. This comes into play when the main MySQL thread "
                    "gets very many connection requests in a very short time",
        READ_ONLY GLOBAL_VAR(back_log), // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(0, 65535), (0), BLOCK_SIZE(1));

static Sys_var_charptr Sys_basedir(
        "basedir", "Path to installation directory. All paths are "
                   "usually resolved relative to this",
        READ_ONLY GLOBAL_VAR(mysql_home_ptr), // CMD_LINE(REQUIRED_ARG, 'b'),
        IN_FS_CHARSET, (0));
static Sys_var_charptr Sys_datadir(
        "datadir", "Path to the database root directory",
        READ_ONLY GLOBAL_VAR(mysql_real_data_home_ptr),
        /* CMD_LINE(REQUIRED_ARG, 'h'),*/ IN_FS_CHARSET, (0));

extern LEX_CSTRING default_auth_plugin_name;
static Sys_var_charptr Sys_default_authentication_plugin(
        "default_authentication_plugin", "The default authentication plugin "
                                         "used by the server to hash the password.",
        READ_ONLY GLOBAL_VAR(default_auth_plugin), // CMD_LINE(REQUIRED_ARG),
        IN_FS_CHARSET, default_auth_plugin_name.str);

// static PolyLock_mutex Plock_default_password_lifetime(
//         &LOCK_default_password_lifetime);
static Sys_var_uint Sys_default_password_lifetime(
        "default_password_lifetime", "The number of days after which the "
                                     "password will expire.",
        GLOBAL_VAR(default_password_lifetime), // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(0, UINT_MAX16), (0), BLOCK_SIZE(1),
        NO_MUTEX_GUARD);
/**
  MY_BIND_ALL_ADDRESSES defines a special value for the bind-address option,
  which means that the server should listen to all available network addresses,
  both IPv6 (if available) and IPv4.

  Basically, this value instructs the server to make an attempt to bind the
  server socket to '::' address, and rollback to '0.0.0.0' if the attempt fails.
*/
const char *MY_BIND_ALL_ADDRESSES= "*";
static Sys_var_charptr Sys_my_bind_addr(
        "bind_address", "IP address to bind to.",
        READ_ONLY GLOBAL_VAR(my_bind_addr_str), // CMD_LINE(REQUIRED_ARG),
        IN_FS_CHARSET, (MY_BIND_ALL_ADDRESSES));

static Sys_var_ulong Sys_binlog_cache_size(
        "binlog_cache_size", "The size of the transactional cache for "
                             "updates to transactional engines for the binary log. "
                             "If you often use transactions containing many statements, "
                             "you can increase this to get more performance",
        GLOBAL_VAR(binlog_cache_size),
        // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(IO_SIZE, ULONG_MAX), (32768), BLOCK_SIZE(IO_SIZE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0));

static Sys_var_ulong Sys_binlog_stmt_cache_size(
        "binlog_stmt_cache_size", "The size of the statement cache for "
                                  "updates to non-transactional engines for the binary log. "
                                  "If you often use statements updating a great number of rows, "
                                  "you can increase this to get more performance",
        GLOBAL_VAR(binlog_stmt_cache_size),
        // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(IO_SIZE, ULONG_MAX), (32768), BLOCK_SIZE(IO_SIZE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0));

static Sys_var_int32 Sys_binlog_max_flush_queue_time(
        "binlog_max_flush_queue_time",
        "The maximum time that the binary log group commit will keep reading"
        " transactions before it flush the transactions to the binary log (and"
        " optionally sync, depending on the value of sync_binlog).",
        GLOBAL_VAR(opt_binlog_max_flush_queue_time),
        // CMD_LINE(REQUIRED_ARG, OPT_BINLOG_MAX_FLUSH_QUEUE_TIME),
        VALID_RANGE(0, 100000), (0), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0),
        DEPRECATED(""));

static Sys_var_long Sys_binlog_group_commit_sync_delay(
        "binlog_group_commit_sync_delay",
        "The number of microseconds the server waits for the "
        "binary log group commit sync queue to fill before "
        "continuing. Default: 0. Min: 0. Max: 1000000.",
        GLOBAL_VAR(opt_binlog_group_commit_sync_delay),
        // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(0, 1000000 /* max 1 sec */), (0), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG);

static Sys_var_ulong Sys_binlog_group_commit_sync_no_delay_count(
        "binlog_group_commit_sync_no_delay_count",
        "If there are this many transactions in the commit sync "
        "queue and the server is waiting for more transactions "
        "to be enqueued (as set using --binlog-group-commit-sync-delay), "
        "the commit procedure resumes.",
        GLOBAL_VAR(opt_binlog_group_commit_sync_no_delay_count),
        // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(0, 100000 /* max connections */),
        (0), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG);

static Sys_var_test_flag Sys_core_file(
        "core_file", "write a core-file on crashes", TEST_CORE_ON_SIGNAL);

extern MYSQL_PLUGIN_IMPORT const char *binlog_format_names[];
static Sys_var_enum Sys_binlog_format(
        "binlog_format", "What form of binary logging the master will "
                         "use: either ROW for row-based binary logging, STATEMENT "
                         "for statement-based binary logging, or MIXED. MIXED is statement-"
                         "based binary logging except for those statements where only row-"
                         "based is correct: those which involve user-defined functions (i.e. "
                         "UDFs) or the UUID() function; for those, row-based binary logging is "
                         "automatically used. If NDBCLUSTER is enabled and binlog-format is "
                         "MIXED, the format switches to row-based and back implicitly per each "
                         "query accessing an NDBCLUSTER table",
        SESSION_VAR(binlog_format), // CMD_LINE(REQUIRED_ARG, OPT_BINLOG_FORMAT),
        binlog_format_names, (BINLOG_FORMAT_ROW),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0));

static const char *rbr_exec_mode_names[]=
        {"STRICT", "IDEMPOTENT", 0};
static Sys_var_enum rbr_exec_mode(
        "rbr_exec_mode",
        "Modes for how row events should be executed. Legal values "
        "are STRICT (default) and IDEMPOTENT. In IDEMPOTENT mode, "
        "the server will not throw errors for operations that are idempotent. "
        "In STRICT mode, server will throw errors for the operations that "
        "cause a conflict.",
        SESSION_VAR(rbr_exec_mode_options), // NO_CMD_LINE,
        rbr_exec_mode_names, (RBR_EXEC_MODE_STRICT),
        NO_MUTEX_GUARD, NOT_IN_BINLOG,
        ON_CHECK(NULL),
        ON_UPDATE(NULL));

static const char *binlog_row_image_names[]= {"MINIMAL", "NOBLOB", "FULL", NullS};
static Sys_var_enum Sys_binlog_row_image(
        "binlog_row_image",
        "Controls whether rows should be logged in 'FULL', 'NOBLOB' or "
        "'MINIMAL' formats. 'FULL', means that all columns in the before "
        "and after image are logged. 'NOBLOB', means that mysqld avoids logging "
        "blob columns whenever possible (eg, blob column was not changed or "
        "is not part of primary key). 'MINIMAL', means that a PK equivalent (PK "
        "columns or full row if there is no PK in the table) is logged in the "
        "before image, and only changed columns are logged in the after image. "
        "(Default: FULL).",
        SESSION_VAR(binlog_row_image), // CMD_LINE(REQUIRED_ARG),
        binlog_row_image_names, (BINLOG_ROW_IMAGE_FULL),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL),
        ON_UPDATE(NULL));

static const char *session_track_gtids_names[]=
        { "OFF", "OWN_GTID", "ALL_GTIDS", NullS };
static Sys_var_enum Sys_session_track_gtids(
        "session_track_gtids",
        "Controls the amount of global transaction ids to be "
        "included in the response packet sent by the server."
        "(Default: OFF).",
        SESSION_VAR(session_track_gtids), // CMD_LINE(REQUIRED_ARG),
        session_track_gtids_names, (OFF),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0));


static Sys_var_mybool Sys_binlog_direct(
        "binlog_direct_non_transactional_updates",
        "Causes updates to non-transactional engines using statement format to "
        "be written directly to binary log. Before using this option make sure "
        "that there are no dependencies between transactional and "
        "non-transactional tables such as in the statement INSERT INTO t_myisam "
        "SELECT * FROM t_innodb; otherwise, slaves may diverge from the master.",
        SESSION_VAR(binlog_direct_non_trans_update),
        // CMD_LINE(OPT_ARG),
        (FALSE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0));

/**
  This variable is read only to users. It can be enabled or disabled
  only at mysqld startup. This variable is used by User thread and
  as well as by replication slave applier thread to apply relay_log.
  Slave applier thread enables/disables this option based on
  relay_log's from replication master versions. There is possibility of
  slave applier thread and User thread to have different setting for
  explicit_defaults_for_timestamp, hence this options is defined as
  SESSION_VAR rather than GLOBAL_VAR.
*/
static Sys_var_mybool Sys_explicit_defaults_for_timestamp(
        "explicit_defaults_for_timestamp",
        "This option causes CREATE TABLE to create all TIMESTAMP columns "
        "as NULL with  NULL attribute, Without this option, "
        "TIMESTAMP columns are NOT NULL and have implicit  clauses. "
        "The old behavior is deprecated. "
        "The variable can only be set by users having the SUPER privilege.",
        SESSION_VAR(explicit_defaults_for_timestamp),
        // CMD_LINE(OPT_ARG),
        (FALSE), NO_MUTEX_GUARD, NOT_IN_BINLOG,
        ON_CHECK(0));

static const char *repository_names[]=
        {
                "FILE", "TABLE",
#ifndef DBUG_OFF
                "DUMMY",
#endif
                0
        };

enum enum_info_repository
{
    INFO_REPOSITORY_FILE= 0,
    INFO_REPOSITORY_TABLE,
    INFO_REPOSITORY_DUMMY,
    /*
      Add new types of repository before this
      entry.
    */
            INVALID_INFO_REPOSITORY
};

const char *log_bin_index= 0;
const char *log_bin_basename= 0;
/*
  Defines status on the repository.
*/
enum enum_return_check { REPOSITORY_DOES_NOT_EXIST= 1, REPOSITORY_EXISTS, ERROR_CHECKING_REPOSITORY };

ulong opt_mi_repository_id= INFO_REPOSITORY_FILE;
static Sys_var_enum Sys_mi_repository(
        "master_info_repository",
        "Defines the type of the repository for the master information."
        ,GLOBAL_VAR(opt_mi_repository_id), //CMD_LINE(REQUIRED_ARG),
        repository_names, (INFO_REPOSITORY_FILE), NO_MUTEX_GUARD,
        NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0));

ulong opt_rli_repository_id= INFO_REPOSITORY_FILE;
static Sys_var_enum Sys_rli_repository(
        "relay_log_info_repository",
        "Defines the type of the repository for the relay log information "
        "and associated workers."
        ,GLOBAL_VAR(opt_rli_repository_id), // CMD_LINE(REQUIRED_ARG),
        repository_names, (INFO_REPOSITORY_FILE), NO_MUTEX_GUARD,
        NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0));

static Sys_var_mybool Sys_binlog_rows_query(
        "binlog_rows_query_log_events",
        "Allow writing of Rows_query_log events into binary log.",
        SESSION_VAR(binlog_rows_query_log_events),
        // CMD_LINE(OPT_ARG),
        (FALSE));
bool opt_binlog_order_commits= true;
static Sys_var_mybool Sys_binlog_order_commits(
        "binlog_order_commits",
        "Issue internal commit calls in the same order as transactions are"
        " written to the binary log. Default is to order commits.",
        GLOBAL_VAR(opt_binlog_order_commits),
        // CMD_LINE(OPT_ARG),
        (TRUE));

static Sys_var_ulong Sys_bulk_insert_buff_size(
        "bulk_insert_buffer_size", "Size of tree cache used in bulk "
                                   "insert optimisation. Note that this is a limit per thread!",
        SESSION_VAR(bulk_insert_buff_size), // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(0, ULONG_MAX), (8192*1024), BLOCK_SIZE(1));
const char *charsets_dir= NULL;
static Sys_var_charptr Sys_character_sets_dir(
        "character_sets_dir", "Directory where character sets are",
        READ_ONLY GLOBAL_VAR(charsets_dir), // CMD_LINE(REQUIRED_ARG),
        IN_FS_CHARSET, (0));


static Sys_var_struct Sys_character_set_system(
        "character_set_system", "The character set used by the server "
                                "for storing identifiers",
        READ_ONLY GLOBAL_VAR(system_charset_info), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (0));

static Sys_var_struct Sys_character_set_server(
        "character_set_server", "The default character set",
        SESSION_VAR(collation_server), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG, ON_CHECK(0));
static Sys_var_struct Sys_character_set_database(
        "character_set_database",
        " The character set used by the default database",
        SESSION_VAR(collation_database), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG, ON_CHECK(0));

// end

static Sys_var_mybool Sys_tx_read_only(
       "tx_read_only", "Set default transaction access mode to read only."
       "This variable is deprecated and will be removed in a future release.",
        UNTRACKED_DEFAULT SESSION_VAR(tx_read_only), /*NO_CMD_LINE,*/ (TRUE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0),
        DEPRECATED("'@@transaction_read_only'"));

static Sys_var_mybool Sys_transaction_read_only(
       "transaction_read_only",
       "Set default transaction access mode to read only.",
       UNTRACKED_DEFAULT SESSION_VAR(transaction_read_only), // NO_CMD_LINE,
       (0), NO_MUTEX_GUARD, NOT_IN_BINLOG,
       ON_CHECK(0),
       ON_UPDATE(0));

static char *server_version_ptr;
static Sys_var_version Sys_version(
        "version", "Server version",
        READ_ONLY GLOBAL_VAR(server_version_ptr), //NO_CMD_LINE,
        IN_SYSTEM_CHARSET, (server_version));

static char *server_version_comment_ptr;
static Sys_var_charptr Sys_version_comment(
        "version_comment", "version_comment",
        READ_ONLY GLOBAL_VAR(server_version_comment_ptr), // NO_CMD_LINE,
        IN_SYSTEM_CHARSET, (MYSQL_COMPILATION_COMMENT));

static char *server_version_compile_machine_ptr;
static Sys_var_charptr Sys_version_compile_machine(
        "version_compile_machine", "version_compile_machine",
        READ_ONLY GLOBAL_VAR(server_version_compile_machine_ptr), // NO_CMD_LINE,
        IN_SYSTEM_CHARSET, (MACHINE_TYPE));

static char *server_version_compile_os_ptr;
static Sys_var_charptr Sys_version_compile_os(
        "version_compile_os", "version_compile_os",
        READ_ONLY GLOBAL_VAR(server_version_compile_os_ptr), // NO_CMD_LINE,
        IN_SYSTEM_CHARSET, (SYSTEM_TYPE));

static Sys_var_struct Sys_character_set_client(
        "character_set_client", "The character set for statements "
                                "that arrive from the client",
        SESSION_VAR(character_set_client), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG/*, ON_CHECK(check_cs_client),
        ON_UPDATE(fix_thd_charset)*/);

static Sys_var_struct Sys_character_set_connection(
        "character_set_connection", "The character set used for "
                                    "literals that do not have a character set introducer and for "
                                    "number-to-string conversion",
        SESSION_VAR(collation_connection), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG/*, ON_CHECK(check_charset_not_null),
        ON_UPDATE(fix_thd_charset)*/);

static Sys_var_struct Sys_character_set_results(
        "character_set_results", "The character set used for returning "
                                 "query results to the client",
        SESSION_VAR(character_set_results), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (&default_charset_info),
        NO_MUTEX_GUARD, NOT_IN_BINLOG/*, ON_CHECK(check_charset)*/);

static Sys_var_struct Sys_character_set_filesystem(
        "character_set_filesystem", "The filesystem character set",
        SESSION_VAR(character_set_filesystem), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, csname), (&character_set_filesystem),
        NO_MUTEX_GUARD, NOT_IN_BINLOG/*, ON_CHECK(check_charset_not_null),
        ON_UPDATE(fix_thd_charset)*/);
static Sys_var_struct Sys_collation_connection(
        "collation_connection", "The collation of the connection "
                                "character set",
        SESSION_VAR(collation_connection), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, name), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG/*, ON_CHECK(check_collation_not_null),
        ON_UPDATE(fix_thd_charset)*/);

static Sys_var_struct Sys_collation_database(
        "collation_database", "The collation of the database "
                              "character set",
        SESSION_VAR(collation_database), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, name), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG/*, ON_CHECK(check_collation_db),
        ON_UPDATE(update_deprecated)*/);

static Sys_var_struct Sys_collation_server(
        "collation_server", "The server default collation",
        SESSION_VAR(collation_server), // NO_CMD_LINE,
        offsetof(CHARSET_INFO, name), (&default_charset_info),
        NO_MUTEX_GUARD, IN_BINLOG/*, ON_CHECK(check_collation_not_null)*/);
static Sys_var_charptr Sys_init_connect(
        "init_connect", "Command(s) that are executed for each "
                        "new connection", GLOBAL_VAR(opt_init_connect),
         /* CMD_LINE(REQUIRED_ARG),*/ IN_SYSTEM_CHARSET,
        (""), NO_MUTEX_GUARD, NOT_IN_BINLOG/*,
        ON_CHECK(check_init_string)*/);

static Sys_var_ulong Sys_interactive_timeout(
        "interactive_timeout",
        "The number of seconds the server waits for activity on an interactive "
        "connection before closing it",
        SESSION_VAR(net_interactive_timeout),
        // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1, LONG_TIMEOUT), (NET_WAIT_TIMEOUT), BLOCK_SIZE(1));

static Sys_var_ulong Sys_join_buffer_size(
        "join_buffer_size",
        "The size of the buffer that is used for full joins",
        SESSION_VAR(join_buff_size), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(128, ULONG_MAX), (256 * 1024), BLOCK_SIZE(128));
static char *license;
static Sys_var_charptr Sys_license(
        "license", "The type of license the server has",
        READ_ONLY GLOBAL_VAR(license), // NO_CMD_LINE,
        IN_SYSTEM_CHARSET,
        (STRINGIFY_ARG(LICENSE)));
static Sys_var_uint Sys_lower_case_table_names(
        "lower_case_table_names",
        "If set to 1 table names are stored in lowercase on disk and table "
        "names will be case-insensitive.  Should be set to 2 if you are using "
        "a case insensitive file system",
        READ_ONLY GLOBAL_VAR(lower_case_table_names),
        // CMD_LINE(OPT_ARG, OPT_LOWER_CASE_TABLE_NAMES),
        VALID_RANGE(0, 2),
        (1),
        BLOCK_SIZE(1));
static Sys_var_ulong Sys_max_allowed_packet(
        "max_allowed_packet",
        "Max packet length to send to or receive from the server",
        SESSION_VAR(max_allowed_packet), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1024, 1024 * 1024 * 1024), (4096 * 1024),
        BLOCK_SIZE(1024), NO_MUTEX_GUARD, NOT_IN_BINLOG/* ,
        ON_CHECK(check_max_allowed_packet) */);

/*
  The new option is added to handle large packets that are sent from the master
  to the slave. It is used to increase the thd(max_allowed) for both the
  DUMP thread on the master and the SQL/IO thread on the slave.
*/
#define MAX_MAX_ALLOWED_PACKET 1024*1024*1024
static Sys_var_ulong Sys_slave_max_allowed_packet(
        "slave_max_allowed_packet",
        "The maximum packet length to sent successfully from the master to slave.",
        GLOBAL_VAR(slave_max_allowed_packet), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1024, MAX_MAX_ALLOWED_PACKET),
        (MAX_MAX_ALLOWED_PACKET),
        BLOCK_SIZE(1024));

static Sys_var_ulonglong Sys_max_binlog_cache_size(
        "max_binlog_cache_size",
        "Sets the total size of the transactional cache",
        GLOBAL_VAR(max_binlog_cache_size), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(IO_SIZE, ULLONG_MAX),
        ((ULLONG_MAX/IO_SIZE)*IO_SIZE),
        BLOCK_SIZE(IO_SIZE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0)/*,
        ON_UPDATE(fix_binlog_cache_size)*/);

static Sys_var_ulonglong Sys_max_binlog_stmt_cache_size(
        "max_binlog_stmt_cache_size",
        "Sets the total size of the statement cache",
        GLOBAL_VAR(max_binlog_stmt_cache_size), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(IO_SIZE, ULLONG_MAX),
        ((ULLONG_MAX/IO_SIZE)*IO_SIZE),
        BLOCK_SIZE(IO_SIZE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0)/*,
        ON_UPDATE(fix_binlog_stmt_cache_size)*/);
static Sys_var_int32 Sys_max_join_size(
       "max_join_size",
       "Joins that are probably going to read more than max_join_size "
       "records return an error",
       SESSION_VAR(max_join_size), // CMD_LINE(REQUIRED_ARG),
       VALID_RANGE(1, INT32_MAX), (INT32_MAX), BLOCK_SIZE(1),
       NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
       ON_UPDATE(0));
static Sys_var_ulong Sys_net_buffer_length(
        "net_buffer_length",
        "Buffer length for TCP/IP and socket communication",
        SESSION_VAR(net_buffer_length), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1024, 1024*1024), (16384), BLOCK_SIZE(1024),
        NO_MUTEX_GUARD, NOT_IN_BINLOG/*, ON_CHECK(check_net_buffer_length)*/);

static bool fix_net_read_timeout(sys_var *self, THD *thd, enum_var_type type)
{
    if (type != OPT_GLOBAL)
    {
        // net_buffer_length is a specific property for the classic protocols
        if (!thd->is_classic_protocol())
        {
            my_error(ER_PLUGGABLE_PROTOCOL_COMMAND_NOT_SUPPORTED, MYF(0));
            return true;
        }
        my_net_set_read_timeout(thd->get_protocol_classic()->get_net(),
                                thd->variables.net_read_timeout);
    }
    return false;
}
static Sys_var_ulong Sys_net_read_timeout(
        "net_read_timeout",
        "Number of seconds to wait for more data from a connection before "
        "aborting the read",
        SESSION_VAR(net_read_timeout), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1, LONG_TIMEOUT), (NET_READ_TIMEOUT), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(fix_net_read_timeout));

static bool fix_net_write_timeout(sys_var *self, THD *thd, enum_var_type type)
{
    if (type != OPT_GLOBAL)
    {
        // net_read_timeout is a specific property for the classic protocols
        if (!thd->is_classic_protocol())
        {
            my_error(ER_PLUGGABLE_PROTOCOL_COMMAND_NOT_SUPPORTED, MYF(0));
            return true;
        }
        my_net_set_write_timeout(thd->get_protocol_classic()->get_net(),
                                 thd->variables.net_write_timeout);
    }
    return false;
}
static Sys_var_ulong Sys_net_write_timeout(
        "net_write_timeout",
        "Number of seconds to wait for a block to be written to a connection "
        "before aborting the write",
        SESSION_VAR(net_write_timeout), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1, LONG_TIMEOUT), (NET_WRITE_TIMEOUT), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(fix_net_write_timeout));

static bool fix_net_retry_count(sys_var *self, THD *thd, enum_var_type type)
{
    if (type != OPT_GLOBAL)
    {
        // net_write_timeout is a specific property for the classic protocols
        if (!thd->is_classic_protocol())
        {
            my_error(ER_PLUGGABLE_PROTOCOL_COMMAND_NOT_SUPPORTED, MYF(0));
            return true;
        }
        thd->get_protocol_classic()->get_net()->retry_count=
                thd->variables.net_retry_count;
    }
    return false;
}
static Sys_var_ulong Sys_net_retry_count(
        "net_retry_count",
        "If a read on a communication port is interrupted, retry this "
        "many times before giving up",
        SESSION_VAR(net_retry_count), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1, ULONG_MAX), (MYSQLD_NET_RETRY_COUNT),
        BLOCK_SIZE(1), NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(fix_net_retry_count));

// static bool fix_query_cache_size(sys_var *self, THD *thd, enum_var_type type)
// {
//     ulong new_cache_size= query_cache.resize(query_cache_size);
//     /*
//        Note: query_cache_size is a global variable reflecting the
//        requested cache size. See also query_cache_size_arg
//     */
//     if (query_cache_size != new_cache_size)
//         push_warning_printf(current_thd, Sql_condition::SL_WARNING,
//                             ER_WARN_QC_RESIZE, ER(ER_WARN_QC_RESIZE),
//                             query_cache_size, new_cache_size);
//
//     query_cache_size= new_cache_size;
//     return false;
// }
static Sys_var_ulong Sys_query_cache_size(
        "query_cache_size",
        "The memory allocated to store results from old queries. "
        "This variable is deprecated and will be removed in a future release.",
        GLOBAL_VAR(query_cache_size), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(0, ULONG_MAX), (1024U*1024U), BLOCK_SIZE(1024),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0), DEPRECATED(""));

static Sys_var_ulong Sys_query_cache_limit(
        "query_cache_limit",
        "Don't cache results that are bigger than this. "
        "This variable is deprecated and will be removed in a future release.",
        GLOBAL_VAR(query_cache_limit), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(0, ULONG_MAX), (1024*1024), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL), ON_UPDATE(NULL),
        DEPRECATED(""));

// static bool fix_qcache_min_res_unit(sys_var *self, THD *thd, enum_var_type type)
// {
//     query_cache_min_res_unit=
//             query_cache.set_min_res_unit(query_cache_min_res_unit);
//     return false;
// }
static Sys_var_ulong Sys_query_cache_min_res_unit(
        "query_cache_min_res_unit",
        "The minimum size for blocks allocated by the query cache. "
        "This variable is deprecated and will be removed in a future release.",
        GLOBAL_VAR(query_cache_min_res_unit), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(0, ULONG_MAX), (QUERY_CACHE_MIN_RESULT_DATA_SIZE),
        BLOCK_SIZE(1), NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0),
        ON_UPDATE(0), DEPRECATED(""));

static const char *query_cache_type_names[]= { "OFF", "ON", "DEMAND", 0 };
static Sys_var_enum Sys_query_cache_type(
        "query_cache_type",
        "OFF = Don't cache or retrieve results. ON = Cache all results "
        "except SELECT SQL_NO_CACHE ... queries. DEMAND = Cache only "
        "SELECT SQL_CACHE ... queries. "
        "This variable is deprecated and will be removed in a future release.",
        SESSION_VAR(query_cache_type), // /* CMD_LINE(REQUIRED_ARG),*/
        query_cache_type_names, (0), NO_MUTEX_GUARD, NOT_IN_BINLOG,
        ON_CHECK(NULL), ON_UPDATE(NULL), DEPRECATED(""));

static Sys_var_mybool Sys_query_cache_wlock_invalidate(
        "query_cache_wlock_invalidate",
        "Invalidate queries in query cache on LOCK for write. "
        "This variable is deprecated and will be removed in a future release.",
        SESSION_VAR(query_cache_wlock_invalidate), // /* CMD_LINE(OPT_ARG),*/
        (FALSE), NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL),
        ON_UPDATE(NULL), DEPRECATED(""));

static Sys_var_mybool Sys_secure_auth(
        "secure_auth",
        "Disallow authentication for accounts that have old (pre-4.1) "
        "passwords. Deprecated. Always TRUE.",
        GLOBAL_VAR(opt_secure_auth), // CMD_LINE(OPT_ARG, OPT_SECURE_AUTH),
        (TRUE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG,
        ON_CHECK(0));

/*
  WARNING: When adding new SQL modes don't forget to update the
  tables definitions that stores it's value (ie: mysql.event, mysql.proc)
*/
static const char *sql_mode_names[]=
        {
                "REAL_AS_FLOAT", "PIPES_AS_CONCAT", "ANSI_QUOTES", "IGNORE_SPACE", ",",
                "ONLY_FULL_GROUP_BY", "NO_UNSIGNED_SUBTRACTION", "NO_DIR_IN_CREATE",
                "POSTGRESQL", "ORACLE", "MSSQL", "DB2", "MAXDB", "NO_KEY_OPTIONS",
                "NO_TABLE_OPTIONS", "NO_FIELD_OPTIONS", "MYSQL323", "MYSQL40", "ANSI",
                "NO_AUTO_VALUE_ON_ZERO", "NO_BACKSLASH_ESCAPES", "STRICT_TRANS_TABLES",
                "STRICT_ALL_TABLES", "NO_ZERO_IN_DATE", "NO_ZERO_DATE",
                "ALLOW_INVALID_DATES", "ERROR_FOR_DIVISION_BY_ZERO", "TRADITIONAL",
                "NO_AUTO_CREATE_USER", "HIGH_NOT_PRECEDENCE", "NO_ENGINE_SUBSTITUTION",
                "PAD_CHAR_TO_FULL_LENGTH",
                0
        };

/*
  sql_mode should *not* be IN_BINLOG: even though it is written to the binlog,
  the slave ignores the MODE_NO_DIR_IN_CREATE variable, so slave's value
  differs from master's (see log_event.cc: Query_log_event::do_apply_event()).
*/
static Sys_var_set Sys_sql_mode(
        "sql_mode",
        "Syntax: sql-mode=mode[,mode[,mode...]]. See the manual for the "
        "complete list of valid sql modes",
        SESSION_VAR(sql_mode), // /* CMD_LINE(REQUIRED_ARG),*/
        sql_mode_names,
        (MODE_NO_ENGINE_SUBSTITUTION |
                MODE_ONLY_FULL_GROUP_BY |
                MODE_STRICT_TRANS_TABLES |
                MODE_NO_ZERO_IN_DATE |
                MODE_NO_ZERO_DATE |
                MODE_ERROR_FOR_DIVISION_BY_ZERO |
                MODE_NO_AUTO_CREATE_USER),
        NO_MUTEX_GUARD,
        NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0));

static Sys_var_mybool Sys_strict_mode(
        "strict_mode", "strict mode",
        SESSION_VAR(strict_mode), FALSE);

static Sys_var_ulong Sys_max_execution_time(
        "max_execution_time",
        "Kill SELECT statement that takes over the specified number of milliseconds",
        SESSION_VAR(max_execution_time), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(0, ULONG_MAX), (0), BLOCK_SIZE(1));

#if defined(HAVE_OPENSSL) && !defined(EMBEDDED_LIBRARY)
#define SSL_OPT(X) CMD_LINE(REQUIRED_ARG,X)
#else
#define SSL_OPT(X) NO_CMD_LINE
#endif

/*
  If you are adding new system variable for SSL communication, please take a
  look at do_auto_cert_generation() function in sql_authentication.cc and
  add new system variable in checks if required.
*/

static Sys_var_charptr Sys_ssl_ca(
        "ssl_ca",
        "CA file in PEM format (check OpenSSL docs, implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_ca), // SSL_OPT(OPT_SSL_CA),
        IN_FS_CHARSET, (0));

static Sys_var_charptr Sys_ssl_capath(
        "ssl_capath",
        "CA directory (check OpenSSL docs, implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_capath), // SSL_OPT(OPT_SSL_CAPATH),
        IN_FS_CHARSET, (0));

static Sys_var_charptr Sys_tls_version(
        "tls_version",
        "TLS version, permitted values are TLSv1, TLSv1.1, TLSv1.2(Only for openssl)",
        READ_ONLY GLOBAL_VAR(opt_tls_version), // SSL_OPT(OPT_TLS_VERSION),
#ifdef HAVE_YASSL
        IN_FS_CHARSET, "TLSv1,TLSv1.1");
#else
        IN_FS_CHARSET, "TLSv1,TLSv1.1,TLSv1.2");
#endif

static Sys_var_charptr Sys_ssl_cert(
        "ssl_cert", "X509 cert in PEM format (implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_cert), // SSL_OPT(OPT_SSL_CERT),
        IN_FS_CHARSET, (0));

static Sys_var_charptr Sys_ssl_cipher(
        "ssl_cipher", "SSL cipher to use (implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_cipher), // SSL_OPT(OPT_SSL_CIPHER),
        IN_FS_CHARSET, (0));

static Sys_var_charptr Sys_ssl_key(
        "ssl_key", "X509 key in PEM format (implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_key), // SSL_OPT(OPT_SSL_KEY),
        IN_FS_CHARSET, (0));

static Sys_var_charptr Sys_ssl_crl(
        "ssl_crl",
        "CRL file in PEM format (check OpenSSL docs, implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_crl), // SSL_OPT(OPT_SSL_CRL),
        IN_FS_CHARSET, (0));

static Sys_var_charptr Sys_ssl_crlpath(
        "ssl_crlpath",
        "CRL directory (check OpenSSL docs, implies --ssl)",
        READ_ONLY GLOBAL_VAR(opt_ssl_crlpath), // SSL_OPT(OPT_SSL_CRLPATH),
        IN_FS_CHARSET, (0));

#if defined(HAVE_OPENSSL) && !defined(HAVE_YASSL)
static Sys_var_mybool Sys_auto_generate_certs(
       "auto_generate_certs",
       "Auto generate SSL certificates at server startup if --ssl is set to "
       "ON and none of the other SSL system variables are specified and "
       "certificate/key files are not present in data directory.",
       READ_ONLY GLOBAL_VAR(opt_auto_generate_certs),
       /* CMD_LINE(OPT_ARG),*/
       (TRUE),
       NO_MUTEX_GUARD,
       NOT_IN_BINLOG,
       ON_CHECK(NULL),
       ON_UPDATE(NULL),
       NULL);
#endif /* HAVE_OPENSSL && !HAVE_YASSL */

static char *system_time_zone_ptr;
static Sys_var_charptr Sys_system_time_zone(
        "system_time_zone", "The server system time zone",
        READ_ONLY GLOBAL_VAR(system_time_zone_ptr), // NO_CMD_LINE,
        IN_FS_CHARSET, (system_time_zone));

// static Sys_var_struct Sys_lc_messages(
//         "lc_messages", "Set the language used for the error messages",
//         SESSION_VAR(lc_messages), // NO_CMD_LINE,
//         my_offsetof(MY_LOCALE, name), (&my_default_lc_messages),
//         NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(check_locale));
//
// static Sys_var_struct Sys_lc_time_names(
//         "lc_time_names", "Set the language used for the month "
//                          "names and the days of the week",
//         SESSION_VAR(lc_time_names), NO_CMD_LINE,
//         my_offsetof(MY_LOCALE, name), (&my_default_lc_time_names),
//         NO_MUTEX_GUARD, IN_BINLOG, ON_CHECK(check_locale));

// https://dev.mysql.com/doc/refman/5.7/en/time-zone-support.html
// The session time zone setting affects display and storage of time values
// that are zone-sensitive. This includes the values displayed by
// functions such as NOW() or CURTIME(), and values stored in and
// retrieved from TIMESTAMP columns. Values for TIMESTAMP columns are
// converted from the session time zone to UTC for storage,
// and from UTC to the session time zone for retrieval.

// The session time zone setting does not affect values displayed by functions
// such as UTC_TIMESTAMP() or values in DATE, TIME, or DATETIME columns.
// Nor are values in those data types stored in UTC;
// the time zone applies for them only when converting from TIMESTAMP values.
// If you want locale-specific arithmetic for DATE, TIME, or DATETIME values,
// convert them to UTC, perform the arithmetic, and then convert back.
static Sys_var_tz Sys_time_zone(
        "time_zone", "time_zone",
        SESSION_VAR(time_zone), // NO_CMD_LINE,
        (&default_tz), NO_MUTEX_GUARD, IN_BINLOG);

/**
  This function updates the thd->variables.transaction_isolation
  to reflect the changes made to @@session.tx_isolation. 'tx_isolation' is
  deprecated and 'transaction_isolation' is its alternative.

  @param[in] self   A pointer to the sys_var.
  @param[in] thd    Thread handler.
  @param[in] type   The type SESSION, GLOBAL or .

  @retval   FALSE   Success.
  @retval   TRUE    Error.
*/
static bool update_transaction_isolation(sys_var *self, THD *thd,
                                         enum_var_type type)
{
    SV *sv= type == OPT_GLOBAL ? &global_system_variables : &thd->variables;
    sv->transaction_isolation= sv->tx_isolation;
    return false;
}
/**
  This function updates thd->variables.tx_isolation to reflect the
  changes to @@session.transaction_isolation. 'tx_isolation' is
  deprecated and 'transaction_isolation' is its alternative.

  @param[in] self   A pointer to the sys_var.
  @param[in] thd    Thread handler.
  @param[in] type   The type SESSION, GLOBAL or .

  @retval   FALSE   Success.
  @retval   TRUE    Error.
*/
static bool update_tx_isolation(sys_var *self, THD *thd,
                                enum_var_type type)
{
    SV *sv= type == OPT_GLOBAL ? &global_system_variables : &thd->variables;
    sv->tx_isolation= sv->transaction_isolation;
    return false;
}

// NO_CMD_LINE - different name of the option
static Sys_var_tx_isolation Sys_tx_isolation(
        "tx_isolation", "Default transaction isolation level."
                        "This variable is deprecated and will be removed in a future release.",
        UNTRACKED_DEFAULT SESSION_VAR(tx_isolation), // NO_CMD_LINE,
        tx_isolation_names, (ISO_REPEATABLE_READ),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL),
        ON_UPDATE(update_transaction_isolation),
        DEPRECATED("'@@transaction_isolation'"));


// NO_CMD_LINE
static Sys_var_tx_isolation Sys_transaction_isolation(
        "transaction_isolation", "Default transaction isolation level",
        UNTRACKED_DEFAULT SESSION_VAR(transaction_isolation), // NO_CMD_LINE,
        tx_isolation_names, (ISO_REPEATABLE_READ),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL),
        ON_UPDATE(update_tx_isolation));


static Sys_var_ulong Sys_net_wait_timeout(
        "wait_timeout",
        "The number of seconds the server waits for activity on a "
        "connection before closing it",
        SESSION_VAR(net_wait_timeout), // /* CMD_LINE(REQUIRED_ARG),*/
        VALID_RANGE(1, IF_WIN(INT_MAX32/1000, LONG_TIMEOUT)),
        (NET_WAIT_TIMEOUT), BLOCK_SIZE(1));
/**
 "time_format" "date_format" "datetime_format"

  the following three variables are unused, and the source of confusion
  (bug reports like "I've changed date_format, but date format hasn't changed.
  I've made them read-only, to alleviate the situation somewhat.

  @todo make them NO_CMD_LINE ?
*/
static Sys_var_charptr Sys_date_format(
        "date_format", "The DATE format (ignored)",
        READ_ONLY GLOBAL_VAR(global_date_format.format.str),
        // /* CMD_LINE(REQUIRED_ARG),*/
        IN_SYSTEM_CHARSET,
        (known_date_time_formats[ISO_FORMAT].date_format),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0),
        DEPRECATED(""));

static Sys_var_charptr Sys_datetime_format(
        "datetime_format", "The DATETIME format (ignored)",
        READ_ONLY GLOBAL_VAR(global_datetime_format.format.str),
        // /* CMD_LINE(REQUIRED_ARG),*/
        IN_SYSTEM_CHARSET,
        (known_date_time_formats[ISO_FORMAT].datetime_format),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0),
        DEPRECATED(""));

static Sys_var_charptr Sys_time_format(
        "time_format", "The TIME format (ignored)",
        READ_ONLY GLOBAL_VAR(global_time_format.format.str),
        // /* CMD_LINE(REQUIRED_ARG),*/
        IN_SYSTEM_CHARSET,
        (known_date_time_formats[ISO_FORMAT].time_format),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0),
        DEPRECATED(""));

static Sys_var_mybool Sys_autocommit(
        "autocommit", "autocommit",
        SESSION_VAR(autocommit), /*NO_CMD_LINE,*/ (TRUE),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(0), ON_UPDATE(0));

static Sys_var_ulong Sys_max_prepared_stmt_count(
        "max_prepared_stmt_count",
        "Maximum number of prepared statements in the server",
        GLOBAL_VAR(max_prepared_stmt_count), // CMD_LINE(REQUIRED_ARG),
        VALID_RANGE(0, 1024*1024), (16382), BLOCK_SIZE(1),
        NO_MUTEX_GUARD, NOT_IN_BINLOG, ON_CHECK(NULL),
        ON_UPDATE(NULL), NULL,
        /* max_prepared_stmt_count is used as a sizing hint by the performance schema. */
        sys_var::PARSE_EARLY);
static Sys_var_mybool Sys_transaction_allow_batching(
        "transaction_allow_batching", "transaction_allow_batching",
        SESSION_ONLY(transaction_allow_batching), FALSE);

static Sys_var_mybool Sys_big_selects(
        "sql_big_selects", "sql_big_selects",
        SESSION_VAR(sql_big_selects), FALSE);

static Sys_var_mybool Sys_log_off(
        "sql_log_off", "sql_log_off",
        SESSION_VAR(sql_log_off), FALSE);
static Sys_var_mybool Sys_sql_warnings(
        "sql_warnings", "sql_warnings",
        SESSION_VAR(sql_warnings), FALSE);

static Sys_var_mybool Sys_sql_notes(
        "sql_notes", "sql_notes",
        SESSION_VAR(sql_notes), TRUE);
static Sys_var_mybool Sys_auto_is_null(
        "sql_auto_is_null", "sql_auto_is_null",
        SESSION_VAR(sql_auto_is_null), FALSE);
static Sys_var_mybool Sys_safe_updates(
        "sql_safe_updates", "sql_safe_updates",
        SESSION_VAR(sql_safe_updates), FALSE);
static Sys_var_mybool Sys_buffer_results(
        "sql_buffer_result", "sql_buffer_result",
        SESSION_VAR(sql_buffer_result), FALSE);
static Sys_var_mybool Sys_quote_show_create(
        "sql_quote_show_create", "sql_quote_show_create",
        SESSION_VAR(sql_quote_show_create), (TRUE));
static Sys_var_mybool Sys_primary_key_checks(
        "primary_key_checks", "primary_key_checks",
        SESSION_VAR(primary_key_checks), FALSE);
static Sys_var_mybool Sys_foreign_key_checks(
        "foreign_key_checks", "foreign_key_checks",
        SESSION_VAR(foreign_key_checks), FALSE);
static Sys_var_mybool Sys_unique_checks(
        "unique_checks", "unique_checks",
        SESSION_VAR(unique_checks), FALSE);

static Sys_var_ulonglong Sys_select_limit(
       "sql_select_limit",
       "The maximum number of rows to return from SELECT statements",
       SESSION_VAR(select_limit), // NO_CMD_LINE,
       VALID_RANGE(0, HA_POS_ERROR), (HA_POS_ERROR), BLOCK_SIZE(1));

bool Sys_var_typelib::check_update_type( BiaodashiValueType value_type )
{
    return !schema::ColumnEntry::IsIntegerType( value_type ) &&
           BiaodashiValueType::BOOL != value_type &&
           !schema::ColumnEntry::IsStringType( value_type );
}
/*
  Function to find a string in a TYPELIB
  (similar to find_type() of mysys/typelib.c)

  SYNOPSIS
   find_type()
   lib			TYPELIB (struct of pointer to values + count)
   find			String to find

 RETURN
  0 error
  > 0 position in TYPELIB->type_names +1
*/

uint find_type(const TYPELIB *lib, const std::string& find)
{
    const char *j;
    std::string findUpper( find );
    aries_utils::to_upper( findUpper );

    for (uint pos=0 ; (j=lib->type_names[pos++]) ; )
    {
        std::string typeName( j );
        aries_utils::to_upper( typeName );
        if ( typeName == findUpper )
            return pos;
    }
    return 0;
}

void Sys_var_typelib::do_check(THD *thd, SetSysVarStructure* var)
{
    auto valueType = var->m_valueCommonExpr->GetValueType().DataType.ValueType;

    bool isSigned = false;
    bool isUnSigned = false;
    longlong tmp;
    switch ( valueType )
    {
        case aries::AriesValueType::BOOL:
        {
            bool value = boost::get<bool>( var->m_valueCommonExpr->GetContent() );
            var->m_ulonglongValue = value;
            break;
        }
        case aries::AriesValueType::INT8: {
            tmp = boost::get<int8_t>( var->m_valueCommonExpr->GetContent() );
            isSigned = true;
            break;
        }
        case aries::AriesValueType::INT16: {
            tmp = boost::get<int16_t>( var->m_valueCommonExpr->GetContent());
            isSigned = true;
            break;
        }
        case aries::AriesValueType::INT32: {
            tmp = boost::get<int32_t>(var->m_valueCommonExpr->GetContent());
            isSigned = true;
            break;
        }
        case aries::AriesValueType::INT64: {
            tmp = boost::get<int64_t>(var->m_valueCommonExpr->GetContent());
            isSigned = true;
            break;
        }
        case aries::AriesValueType::UINT8: {
            var->m_ulonglongValue = boost::get<uint8_t>(var->m_valueCommonExpr->GetContent());
            isUnSigned = true;
            break;
        }
        case aries::AriesValueType::UINT16: {
            var->m_ulonglongValue = boost::get<uint16_t>(var->m_valueCommonExpr->GetContent());
            isUnSigned = true;
            break;
        }
        case aries::AriesValueType::UINT32: {
            var->m_ulonglongValue = boost::get<uint32_t>(var->m_valueCommonExpr->GetContent());
            isUnSigned = true;
            break;
        }
        case aries::AriesValueType::UINT64: {
            var->m_ulonglongValue = boost::get<uint64_t>(var->m_valueCommonExpr->GetContent());
            isUnSigned = true;
            break;
        }
        case aries::AriesValueType::CHAR:
        {
            string value = boost::get<string>(var->m_valueCommonExpr->GetContent());
            var->m_ulonglongValue = find_type( &typelib, value );
            if ( !var->m_ulonglongValue )
            {
                 ARIES_EXCEPTION( ER_WRONG_VALUE_FOR_VAR, name.data(), value.data() );
            }
            var->m_ulonglongValue--;
        }
    
        default:
            break;
    }
    if ( isSigned )
    {
        if ( tmp < 0 || tmp >= typelib.count )
        {
            ARIES_EXCEPTION( ER_WRONG_VALUE_FOR_VAR, name.data(), std::to_string( tmp ).data() );
        }
        var->m_ulonglongValue = tmp;
    }
    else if ( isUnSigned )
    {
        if ( var->m_ulonglongValue >= typelib.count )
            ARIES_EXCEPTION( ER_WRONG_VALUE_FOR_VAR, name.data(), std::to_string( var->m_ulonglongValue >= typelib.count ).data() );
    }

}
void Sys_var_enum::session_save_default(THD *thd, SetSysVarStructure* var)
{

}
void Sys_var_enum::global_save_default(THD *thd, SetSysVarStructure* var)
{

}
bool Sys_var_enum::session_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
bool Sys_var_enum::global_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
void Sys_var_mybool::session_save_default(THD *thd, SetSysVarStructure* var)
{
    var->m_ulonglongValue = static_cast<ulonglong>(*(my_bool *)global_value_ptr(thd));
}
void Sys_var_mybool::global_save_default(THD *thd, SetSysVarStructure* var)
{
    var->m_ulonglongValue = option.def_value;
}
bool Sys_var_mybool::session_update(THD *thd, const SetSysVarStructure* var)
{
    session_var(thd, my_bool)=
      static_cast<my_bool>(var->m_ulonglongValue);
    return false;
}
bool Sys_var_mybool::global_update(THD *thd, const SetSysVarStructure* var)
{
    global_var(my_bool)=
       static_cast<my_bool>(var->m_ulonglongValue);
    return false;
}
bool Sys_var_struct::check_update_type( BiaodashiValueType value_type )
{
    return !schema::ColumnEntry::IsIntegerType( value_type ) && !schema::ColumnEntry::IsStringType( value_type );
}
void Sys_var_struct::session_save_default(THD *thd, SetSysVarStructure* var)
{

}
void Sys_var_struct::global_save_default(THD *thd, SetSysVarStructure* var)
{

}
bool Sys_var_struct::session_update(THD *thd, const SetSysVarStructure* var)
{
        return true;

}
bool Sys_var_struct::global_update(THD *thd, const SetSysVarStructure* var)
{
        return true;

}
bool Sys_var_charptr::check_update_type( BiaodashiValueType value_type )
{
    return !schema::ColumnEntry::IsStringType( value_type );
}
void Sys_var_charptr::do_check(THD *thd, SetSysVarStructure* var)
{
}
void Sys_var_charptr::session_save_default(THD *thd, SetSysVarStructure* var)
{

}
void Sys_var_charptr::global_save_default(THD *thd, SetSysVarStructure* var)
{

}
bool Sys_var_charptr::session_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
bool Sys_var_charptr::global_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
void Sys_var_set::do_check(THD *thd, SetSysVarStructure* var)
{
}
void Sys_var_set::session_save_default(THD *thd, SetSysVarStructure* var)
{

}
void Sys_var_set::global_save_default(THD *thd, SetSysVarStructure* var)
{

}
bool Sys_var_set::session_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
bool Sys_var_set::global_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
void Sys_var_tz::do_check(THD *thd, SetSysVarStructure* var)
{
}
void Sys_var_tz::session_save_default(THD *thd, SetSysVarStructure* var)
{

}
void Sys_var_tz::global_save_default(THD *thd, SetSysVarStructure* var)
{

}
bool Sys_var_tz::session_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
bool Sys_var_tz::global_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}
bool Sys_var_tx_isolation::session_update(THD *thd, const SetSysVarStructure* var)
{
        return true;
}