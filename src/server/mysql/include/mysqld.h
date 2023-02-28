/* Copyright (c) 2010, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA */

#ifndef MYSQLD_INCLUDED
#define MYSQLD_INCLUDED

#include "my_global.h" /* MYSQL_PLUGIN_IMPORT, FN_REFLEN, FN_EXTLEN */
#include "m_string.h"
// #include "my_bitmap.h"                     /* MY_BITMAP */
// #include "my_decimal.h"                         /* my_decimal */
#include "mysql_com.h"                     /* SERVER_VERSION_LENGTH */
// #include "my_atomic.h"                     /* my_atomic_add64 */
// #include "sql_cmd.h"                       /* SQLCOM_END */
#include "my_thread_local.h"               /* my_get_thread_local */
// #include "sql_class.h"
// #include "my_thread.h"                     /* my_thread_attr_t */
// #include "atomic_class.h"                  /* Atomic_int32 */
#include "derror.h"
#include "gcc_atomic.h"

class THD;
class Time_zone;

typedef struct st_mysql_const_lex_string LEX_CSTRING;
// typedef struct st_mysql_show_var SHOW_VAR;
/* minimal result data size when data allocated */
#define QUERY_CACHE_MIN_RESULT_DATA_SIZE	1024*4

extern pthread_mutex_t LOCK_thread_ids;
extern pthread_mutex_t LOCK_socket_listener_active;
extern pthread_cond_t COND_socket_listener_active;
extern pthread_mutex_t LOCK_start_signal_handler;
extern pthread_cond_t COND_start_signal_handler;
extern pthread_mutex_t LOCK_prepared_stmt_count;
extern pthread_attr_t connection_attr;

/*
  This forward declaration is used from C files where the real
  definition is included before.  Since C does not allow repeated
  typedef declarations, even when identical, the definition may not be
  repeated.
*/
#ifndef CHARSET_INFO_DEFINED
typedef struct charset_info_st CHARSET_INFO;
#endif  /* CHARSET_INFO_DEFINED */

#if MAX_INDEXES <= 64
#elif MAX_INDEXES > 255
#error "MAX_INDEXES values greater than 255 is not supported."
#else
#endif

	/* Bits from testflag */
#define TEST_CORE_ON_SIGNAL	256	/**< Give core if signal */

/* Function prototypes */
void kill_mysql(void);

/* query_id */
typedef int64 query_id_t;
extern query_id_t global_query_id;

/* increment query_id and return it.  */
inline MY_ATTRIBUTE((warn_unused_result)) query_id_t next_query_id()
{
    query_id_t id= my_atomic_add64(&global_query_id, 1);
    return (id+1);
}

// These are needed for unit testing.
// void set_remaining_args(int argc, char **argv);
int init_common_variables();

extern "C" MYSQL_PLUGIN_IMPORT CHARSET_INFO *system_charset_info;
// extern MYSQL_PLUGIN_IMPORT CHARSET_INFO *files_charset_info ;
// extern MYSQL_PLUGIN_IMPORT CHARSET_INFO *national_charset_info;
// extern MYSQL_PLUGIN_IMPORT CHARSET_INFO *table_alias_charset;
//
// enum enum_server_operational_state
// {
//   SERVER_BOOTING,      /* Server is not operational. It is starting */
//   SERVER_OPERATING,    /* Server is fully initialized and operating */
//   SERVER_SHUTTING_DOWN /* erver is shutting down */
// };
// enum_server_operational_state get_server_state();

/**
  Character set of the buildin error messages loaded from errmsg.sys.
*/
// extern CHARSET_INFO *error_message_charset_info;

extern time_t server_start_time;
extern CHARSET_INFO *character_set_filesystem;

extern MYSQL_PLUGIN_IMPORT bool volatile abort_loop;
extern my_bool opt_secure_auth;
extern char *aries_unix_port;
extern Time_zone *default_tz;
extern uint test_flags; // ,select_errors,ha_open_options;
extern char curr_dir[FN_REFLEN], home_dir_buff[FN_REFLEN];
extern char *home_dir;          /* Home directory for user */
extern char mysql_home[FN_REFLEN];
extern char* mysql_home_ptr; // , *pidfile_name_ptr;
extern char mysql_real_data_home[FN_REFLEN];
extern const char *mysql_real_data_home_ptr;
extern char *default_auth_plugin;
extern uint default_password_lifetime;
extern char *my_bind_addr_str;
extern char system_time_zone[30];
extern ulong query_cache_size, query_cache_limit, query_cache_min_res_unit;
extern ulong max_prepared_stmt_count, prepared_stmt_count;
extern ulong binlog_cache_size, binlog_stmt_cache_size;
extern ulonglong max_binlog_cache_size, max_binlog_stmt_cache_size;
extern int32 opt_binlog_max_flush_queue_time;
extern long opt_binlog_group_commit_sync_delay;
extern ulong opt_binlog_group_commit_sync_no_delay_count;
extern ulong slave_max_allowed_packet;
extern ulong back_log;
extern char language[FN_REFLEN];
extern struct system_variables max_system_variables;
extern char* opt_init_connect;
extern MYSQL_PLUGIN_IMPORT const char *tx_isolation_names[];
enum enum_tx_isolation { ISO_READ_UNCOMMITTED, ISO_READ_COMMITTED,
    ISO_REPEATABLE_READ, ISO_SERIALIZABLE};
extern my_bool sp_automatic_privileges;
/*
  THR_THD is a key which will be used to set/get THD* for a thread,
  using my_set_thread_local()/my_get_thread_local().
*/
extern MYSQL_PLUGIN_IMPORT thread_local_key_t THR_THD;
extern bool THR_THD_initialized;
extern CHARSET_INFO *default_charset_info;

static inline THD * my_thread_get_THR_THD()
{
  if (THR_THD_initialized) {
      return (THD*)my_get_thread_local(THR_THD);
  } else {
      return nullptr;
  }
}

static inline int my_thread_set_THR_THD(THD *thd)
{
  DBUG_ASSERT(THR_THD_initialized);
  return my_set_thread_local(THR_THD, thd);
}
#ifdef HAVE_OPENSSL
extern struct st_VioSSLFd * ssl_acceptor_fd;
#endif /* HAVE_OPENSSL */

extern MYSQL_PLUGIN_IMPORT uint lower_case_table_names;
extern "C" MYSQL_PLUGIN_IMPORT char server_version[SERVER_VERSION_LENGTH];
extern MYSQL_PLUGIN_IMPORT struct system_variables global_system_variables;

extern char *opt_ssl_ca, *opt_ssl_capath, *opt_ssl_cert, *opt_ssl_cipher,
            *opt_ssl_key, *opt_ssl_crl, *opt_ssl_crlpath, *opt_tls_version;

/*
  TODO: Replace this with an inline function.
 */
#ifndef EMBEDDED_LIBRARY
extern "C" void unireg_abort(int exit_code) /*MY_ATTRIBUTE((noreturn))*/;
#else
extern "C" void unireg_clear(int exit_code);
#define unireg_abort(exit_code) do { unireg_clear(exit_code); DBUG_RETURN(exit_code); } while(0)
#endif

static inline THD *_current_thd(void)
{
  return my_thread_get_THR_THD();
}
#define current_thd _current_thd()

#define ER(X)         ER_THD(current_thd,X)

#endif /* MYSQLD_INCLUDED */
