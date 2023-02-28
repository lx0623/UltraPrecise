#ifndef SQL_CLASS_INCLUDED
#define SQL_CLASS_INCLUDED

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <sys/time.h>
// #include "m_ctype.h"
#include "mysql_com.h"
#include "mysqld_error.h"
#include "protocol_classic.h"
// #include "sql_security_ctx.h"
#include "violite.h"
#include "binary_log_types.h"
#include "session_tracker.h"
#include "sql_error.h"
#include "my_sys.h"

#include "frontend/SQLExecutor.h"
#include "sql_prepare.h"
#include "derror.h"
#include "tztime.h"

namespace aries_engine {
  class AriesTransaction;
  using AriesTransactionPtr = shared_ptr<AriesTransaction>;
}

extern char empty_c_string[1];
extern LEX_STRING EMPTY_STR;
extern LEX_STRING NULL_STR;
extern LEX_CSTRING EMPTY_CSTR;
extern LEX_CSTRING NULL_CSTR;

using std::vector;
using std::unordered_map;
using std::shared_ptr;
using aries::SQLExecutor;

typedef ulonglong sql_mode_t;

class THD;
class Prepared_statement_map;
class AbstractBiaodashi;
class MY_LOCALE_ERRMSGS;

#ifdef __cplusplus
extern "C" {
#endif
void **thd_ha_data(const THD* thd/*, const struct handlerton *hton*/);
#ifdef __cplusplus
}
#endif

/* Bits for different SQL modes modes (including ANSI mode) */
#define MODE_REAL_AS_FLOAT              1
#define MODE_PIPES_AS_CONCAT            2
#define MODE_ANSI_QUOTES                4
#define MODE_IGNORE_SPACE               8
#define MODE_NOT_USED                   16
#define MODE_ONLY_FULL_GROUP_BY         32
#define MODE_NO_UNSIGNED_SUBTRACTION    64
#define MODE_NO_DIR_IN_CREATE           128
#define MODE_POSTGRESQL                 256
#define MODE_ORACLE                     512
#define MODE_MSSQL                      1024
#define MODE_DB2                        2048
#define MODE_MAXDB                      4096
#define MODE_NO_KEY_OPTIONS             8192
#define MODE_NO_TABLE_OPTIONS           16384
#define MODE_NO_FIELD_OPTIONS           32768
#define MODE_MYSQL323                   65536L
#define MODE_MYSQL40                    (MODE_MYSQL323*2)
#define MODE_ANSI                       (MODE_MYSQL40*2)
#define MODE_NO_AUTO_VALUE_ON_ZERO      (MODE_ANSI*2)
#define MODE_NO_BACKSLASH_ESCAPES       (MODE_NO_AUTO_VALUE_ON_ZERO*2)
#define MODE_STRICT_TRANS_TABLES        (MODE_NO_BACKSLASH_ESCAPES*2)
#define MODE_STRICT_ALL_TABLES          (MODE_STRICT_TRANS_TABLES*2)
/*
 * NO_ZERO_DATE, NO_ZERO_IN_DATE and ERROR_FOR_DIVISION_BY_ZERO modes are
 * removed in 5.7 and their functionality is merged with STRICT MODE.
 * However, For backward compatibility during upgrade, these modes are kept
 * but they are not used. Setting these modes in 5.7 will give warning and
 * have no effect.
 */
#define MODE_NO_ZERO_IN_DATE            (MODE_STRICT_ALL_TABLES*2)
#define MODE_NO_ZERO_DATE               (MODE_NO_ZERO_IN_DATE*2)
#define MODE_INVALID_DATES              (MODE_NO_ZERO_DATE*2)
#define MODE_ERROR_FOR_DIVISION_BY_ZERO (MODE_INVALID_DATES*2)
#define MODE_TRADITIONAL                (MODE_ERROR_FOR_DIVISION_BY_ZERO*2)
#define MODE_NO_AUTO_CREATE_USER        (MODE_TRADITIONAL*2)
#define MODE_HIGH_NOT_PRECEDENCE        (MODE_NO_AUTO_CREATE_USER*2)
#define MODE_NO_ENGINE_SUBSTITUTION     (MODE_HIGH_NOT_PRECEDENCE*2)
#define MODE_PAD_CHAR_TO_FULL_LENGTH    (1ULL << 31)

enum enum_delay_key_write { DELAY_KEY_WRITE_NONE, DELAY_KEY_WRITE_ON,
    DELAY_KEY_WRITE_ALL };
enum enum_rbr_exec_mode { RBR_EXEC_MODE_STRICT,
    RBR_EXEC_MODE_IDEMPOTENT,
    RBR_EXEC_MODE_LAST_BIT };
enum enum_transaction_write_set_hashing_algorithm { HASH_ALGORITHM_OFF= 0,
    HASH_ALGORITHM_MURMUR32= 1,
    HASH_ALGORITHM_XXHASH64= 2};
enum enum_slave_type_conversions { SLAVE_TYPE_CONVERSIONS_ALL_LOSSY,
    SLAVE_TYPE_CONVERSIONS_ALL_NON_LOSSY,
    SLAVE_TYPE_CONVERSIONS_ALL_UNSIGNED,
    SLAVE_TYPE_CONVERSIONS_ALL_SIGNED};
enum enum_slave_rows_search_algorithms { SLAVE_ROWS_TABLE_SCAN = (1U << 0),
    SLAVE_ROWS_INDEX_SCAN = (1U << 1),
    SLAVE_ROWS_HASH_SCAN  = (1U << 2)};
enum enum_binlog_row_image {
    /** PKE in the before image and changed columns in the after image */
            BINLOG_ROW_IMAGE_MINIMAL= 0,
    /** Whenever possible, before and after image contain all columns except blobs. */
            BINLOG_ROW_IMAGE_NOBLOB= 1,
    /** All columns in both before and after image. */
            BINLOG_ROW_IMAGE_FULL= 2
};

bool check_column_name(const char *name);
enum enum_session_track_gtids {
    OFF= 0,
    OWN_GTID= 1,
    ALL_GTIDS= 2
};

enum enum_binlog_format {
    BINLOG_FORMAT_MIXED= 0, ///< statement if safe, otherwise row - autodetected
    BINLOG_FORMAT_STMT=  1, ///< statement-based
    BINLOG_FORMAT_ROW=   2, ///< row-based
    BINLOG_FORMAT_UNSPEC=3  ///< thd_binlog_format() returns it when binlog is closed
};


class Protocol_classic;

typedef struct system_variables
{
    /*
      How dynamically allocated system variables are handled:

      The global_system_variables and max_system_variables are "authoritative"
      They both should have the same 'version' and 'size'.
      When attempting to access a dynamic variable, if the session version
      is out of date, then the session version is updated and realloced if
      neccessary and bytes copied from global to make up for missing data.
    */
    ulong dynamic_variables_version;
    char* dynamic_variables_ptr;
    uint dynamic_variables_head;    /* largest valid variable offset */
    uint dynamic_variables_size;    /* how many bytes are in use */
    // LIST *dynamic_variables_allocs; /* memory hunks for PLUGIN_VAR_MEMALLOC */

    ulonglong max_heap_table_size;
    ulonglong tmp_table_size;
    ulonglong long_query_time;
    my_bool end_markers_in_json;
    /* A bitmap for switching optimizations on/off */
    ulonglong optimizer_switch;
    ulonglong optimizer_trace; ///< bitmap to tune optimizer tracing
    ulonglong optimizer_trace_features; ///< bitmap to select features to trace
    long      optimizer_trace_offset;
    long      optimizer_trace_limit;
    ulong     optimizer_trace_max_mem_size;
    sql_mode_t sql_mode; ///< which non-standard SQL behaviour should be enabled
    my_bool strict_mode; // rateup specific
    // ulonglong option_bits; ///< OPTION_xxx constants, e.g. OPTION_PROFILING
    my_bool autocommit;
    my_bool sql_safe_updates;
    my_bool sql_quote_show_create;
    my_bool primary_key_checks;
    my_bool foreign_key_checks;
    my_bool unique_checks;
    my_bool sql_buffer_result;
    my_bool sql_auto_is_null;
    my_bool transaction_allow_batching;
    my_bool sql_warnings;
    my_bool sql_notes;
    my_bool sql_big_selects;
    my_bool sql_log_off;
    ulonglong select_limit;
    // ha_rows max_join_size;
    ulong auto_increment_increment, auto_increment_offset;
    ulong bulk_insert_buff_size;
    uint  eq_range_index_dive_limit;
    ulong join_buff_size;
    ulong lock_wait_timeout;
    ulong max_allowed_packet;
    ulong max_error_count;
    ulong max_length_for_sort_data;
    ulong max_points_in_geometry;
    ulong max_sort_length;
    ulong max_tmp_tables;
    ulong max_insert_delayed_threads;
    ulong min_examined_row_limit;
    ulong multi_range_count;
    ulong myisam_repair_threads;
    ulong myisam_sort_buff_size;
    ulong myisam_stats_method;
    ulong net_buffer_length;
    ulong net_interactive_timeout;
    ulong net_read_timeout;
    ulong net_retry_count;
    ulong net_wait_timeout;
    ulong net_write_timeout;
    ulong optimizer_prune_level;
    ulong optimizer_search_depth;
    ulonglong parser_max_mem_size;
    ulong range_optimizer_max_mem_size;
    ulong preload_buff_size;
    ulong profiling_history_size;
    ulong read_buff_size;
    ulong read_rnd_buff_size;
    ulong div_precincrement;
    ulong sortbuff_size;
    ulong max_sp_recursion_depth;
    ulong default_week_format;
    ulong max_seeks_for_key;
    ulong range_alloc_block_size;
    ulong query_alloc_block_size;
    ulong query_prealloc_size;
    ulong trans_alloc_block_size;
    ulong trans_prealloc_size;
    ulong group_concat_max_len;

    ulong binlog_format; ///< binlog format for this thd (see enum_binlog_format)
    ulong rbr_exec_mode_options;
    my_bool binlog_direct_non_trans_update;
    ulong binlog_row_image;
    my_bool sql_log_bin;
    ulong transaction_write_set_extraction;
    ulong completion_type;
    ulong query_cache_type;
    ulong tx_isolation;
    ulong transaction_isolation;
    ulong updatable_views_with_limit;
    uint max_user_connections;
    ulong my_aes_mode;

    /**
      In slave thread we need to know in behalf of which
      thread the query is being run to replicate temp tables properly
    */
    // my_thread_id pseudo_thread_id;
    /**
      Default transaction access mode. READ ONLY (true) or READ WRITE (false).
    */
    my_bool tx_read_only;
    my_bool transaction_read_only;
    my_bool low_priority_updates;
    my_bool new_mode;
    my_bool query_cache_wlock_invalidate;
    my_bool keep_files_on_create;

    my_bool old_alter_table;
    uint old_passwords;
    my_bool big_tables;
    int32_t max_join_size;

    // plugin_ref table_plugin;
    // plugin_ref temp_table_plugin;

    /* Only charset part of these variables is sensible */
    const CHARSET_INFO *character_set_filesystem;
    const CHARSET_INFO *character_set_client;
    const CHARSET_INFO *character_set_results;

    /* Both charset and collation parts of these variables are important */
    const CHARSET_INFO  *collation_server;
    const CHARSET_INFO  *collation_database;
    const CHARSET_INFO  *collation_connection;

    /* Error messages */
    // MY_LOCALE *lc_messages;
    MY_LOCALE_ERRMSGS *lc_messages;
    /* Locale Support */
    // MY_LOCALE *lc_time_names;

    Time_zone *time_zone;
    /*
      TIMESTAMP fields are by default created with DEFAULT clauses
      implicitly without users request. This flag when set, disables
      implicit default values and expect users to provide explicit
      default clause. i.e., when set columns are defined as NULL,
      instead of NOT NULL by default.
    */
    my_bool explicit_defaults_for_timestamp;

    my_bool sysdate_is_now;
    my_bool binlog_rows_query_log_events;

    double long_query_time_double;

    my_bool pseudo_slave_mode;

    // Gtid_specification gtid_next;
    // Gtid_set_or_null gtid_next_list;
    ulong session_track_gtids;

    ulong max_execution_time;

    char *track_sysvars_ptr;
    my_bool session_track_schema;
    my_bool session_track_state_change;
    ulong   session_track_transaction_info;
    /**
      Used for the verbosity of SHOW CREATE TABLE. Currently used for displaying
      the row format in the output even if the table uses default row format.
    */
    my_bool show_create_table_verbosity;
    /**
      Compatibility option to mark the pre MySQL-5.6.4 temporals columns using
      the old format using comments for SHOW CREATE TABLE and in I_S.COLUMNS
      'COLUMN_TYPE' field.
    */
    my_bool show_old_temporals;
} SV;

/**
  Storage engine specific thread local data.
*/

struct Ha_data
{
    /**
      Storage engine specific thread local data.
      Lifetime: one user connection.
    */
    void *ha_ptr;
    /**
      0: Life time: one statement within a transaction. If @@autocommit is
      on, also represents the entire transaction.
      @sa trans_register_ha()

      1: Life time: one transaction within a connection.
      If the storage engine does not participate in a transaction,
      this should not be used.
      @sa trans_register_ha()
    */
    // Ha_trx_info ha_info[2];
    /**
      NULL: engine is not bound to this thread
      non-NULL: engine is bound to this thread, engine shutdown forbidden
    */
    // plugin_ref lock;
    Ha_data() :ha_ptr(NULL) {}
};

/**
  Container for all prepared statements created/used in a connection.

  Prepared statements in Prepared_statement_map have unique id
  (guaranteed by id assignment in Prepared_statement::Prepared_statement).

  Non-empty statement names are unique too: attempt to insert a new statement
  with duplicate name causes older statement to be deleted.

  Prepared statements are auto-deleted when they are removed from the map
  and when the map is deleted.
*/

class Prepared_statement_map
{
public:
    Prepared_statement_map();

    /**
      Insert a new statement to the thread-local prepared statement map.

      If there was an old statement with the same name, replace it with the
      new one. Otherwise, check if max_prepared_stmt_count is not reached yet,
      increase prepared_stmt_count, and insert the new statement. It's okay
      to delete an old statement and fail to insert the new one.

      All named prepared statements are also present in names_hash.
      Prepared statement names in names_hash are unique.
      The statement is added only if prepared_stmt_count < max_prepard_stmt_count
      m_last_found_statement always points to a valid statement or is 0

      @retval 0  success
      @retval 1  error: out of resources or max_prepared_stmt_count limit has been
                        reached. An error is sent to the client, the statement
                        is deleted.
    */
    int insert(THD *thd, Prepared_statement_ptr statement);

    /** Find prepared statement by name. */
    Prepared_statement_ptr find_by_name(const string &name);

    /** Find prepared statement by ID. */
    Prepared_statement_ptr find(ulong id);

    /** Erase all prepared statements (calls Prepared_statement destructor). */
    void erase(Prepared_statement_ptr statement);

    void claim_memory_ownership();

    void reset();

    // ~Prepared_statement_map();
private:
    unordered_map<ulong, Prepared_statement_ptr> st_hash;
    unordered_map<string, Prepared_statement_ptr> names_hash;
    Prepared_statement_ptr m_last_found_statement;
};

class user_var_entry;
using user_var_entry_ptr = std::shared_ptr<user_var_entry>;

/**
  Convert microseconds since epoch to timeval.
  @param      micro_time  Microseconds.
  @param[out] tm          A timeval variable to write to.
*/
static inline void
my_micro_time_to_timeval(ulonglong micro_time, struct timeval *tm)
{
    tm->tv_sec=  (long) (micro_time / 1000000);
    tm->tv_usec= (long) (micro_time % 1000000);
}
ulonglong my_micro_time();
class THD {
public:
    THD();
    ~THD();
    void init(void);
    void release_resources();

    inline void set_time()
    {
        start_utime= my_micro_time();
        // if (user_time.tv_sec || user_time.tv_usec)
        // {
        //     start_time= user_time;
        // }
        // else
            my_micro_time_to_timeval(start_utime, &start_time);
    }

    /**
   Assign a new value to thd->query_id.
   Protected with the LOCK_thd_data mutex.
 */
    void set_query_id(query_id_t new_query_id)
    {
        mysql_mutex_lock(&LOCK_thd_data);
        query_id= new_query_id;
        mysql_mutex_unlock(&LOCK_thd_data);
    }

public:
    uint16 peer_port;
    string peer_host;
    struct timeval start_time;
    ulonglong  start_utime;
    /**
     Set to TRUE if execution of the current compound statement
     can not continue. In particular, disables activation of
     CONTINUE or EXIT handlers of stored routines.
     Reset in the end of processing of the current user request, in
     @see mysql_reset_thd_for_next_command().
    */
    bool is_fatal_error;
    struct  system_variables variables;	// Changeable local variables
    uint32 m_connection_id;
    NET net;
    // uint16 peer_port;
    std::atomic<bool> m_server_idle;

    /*
    Id of current query. Statement can be reused to execute several queries
    query_id is global in context of the whole MySQL server.
    ID is automatically generated from mutex-protected counter.
    It's used in handler code for various purposes: to check which columns
    from table are necessary for this select, to check if it's necessary to
    update auto-updatable fields (like auto_increment and timestamp).
  */
    query_id_t query_id;

    uint	     server_status;
    Vio *active_vio;
    Session_tracker session_tracker;

    /* container for handler's private per-connection data */
    Ha_data ha_data[1];
    bool       time_zone_used;
    /* Statement id is thread-wide. This counter is used to generate ids */
    ulong      statement_id_counter;
    Prepared_statement_map stmt_map;
    vector< std::shared_ptr<aries::AbstractBiaodashi> > stmt_params;

    /*
      If checking this in conjunction with a wait condition, please
      include a check after enter_cond() if you want to avoid a race
      condition. For details see the implementation of awake(),
      especially the "broadcast" part.
    */
    enum killed_state
    {
        NOT_KILLED=0,
        KILL_BAD_DATA=1,
        KILL_CONNECTION=ER_SERVER_SHUTDOWN,
        KILL_QUERY=ER_QUERY_INTERRUPTED,
        KILL_TIMEOUT=ER_QUERY_TIMEOUT,
        KILLED_NO_VALUE      /* means neither of the states */
    };
    killed_state volatile killed;

    pthread_t  real_id;                           /* For debugging */

    int is_killed() { return killed; }

    void awake(THD::killed_state state_to_set);

    bool is_killable;
    /**
      Mutex protecting access to current_mutex and current_cond.
    */
    mysql_mutex_t LOCK_current_cond;
    /**
      The mutex used with current_cond.
      @see current_cond
    */
    mysql_mutex_t * volatile current_mutex;

    /**
    Pointer to the condition variable the thread owning this THD
    is currently waiting for. If the thread is not waiting, the
    value is NULL. Set by THD::enter_cond().

    If this thread is killed (shutdown or KILL stmt), another
    thread will broadcast on this condition variable so that the
    thread can be unstuck.
  */
    mysql_cond_t * volatile current_cond;

    /* scramble - random string sent to client on handshake */
    char	     scramble[SCRAMBLE_LENGTH+1];

    Protocol_text protocol_text;     // Normal protocol
    Protocol_binary protocol_binary; // Binary protocol

    const CHARSET_INFO *charset() const
    { return variables.character_set_client; }

    const string &db() const
    { return m_db; }

    void set_user_name(char* user_name, int len)
    {
        if (m_user_name == user_name)
            return;
        if (NULL != m_user_name)
            free(m_user_name);
        m_user_name = (char*)malloc(len + 1);
        memset(m_user_name, 0, len + 1);
        strncpy(m_user_name, user_name, len);
    }
    const char* get_user_name() { return m_user_name; }

    bool set_db(const string &new_db)
    {
        m_db = new_db;

        return true;
    }

    Protocol *get_protocol()
    {
        return m_protocol;
    }
    inline void set_active_vio(Vio* vio)
    {
        mysql_mutex_lock(&LOCK_thd_data);
        active_vio = vio;
        mysql_mutex_unlock(&LOCK_thd_data);
    }
    inline void clear_active_vio()
    {
        mysql_mutex_lock(&LOCK_thd_data);
        active_vio = 0;
        // m_SSL = NULL;
        mysql_mutex_unlock(&LOCK_thd_data);
    }
    void disconnect(bool server_shutdown);
    void shutdown_active_vio();
    uint32 thread_id() const { return m_connection_id; }
    /**
      Asserts that the protocol is of type text or binary and then
      returns the m_protocol casted to Protocol_classic. This method
      is needed to prevent misuse of pluggable protocols by legacy code
    */
    Protocol_classic *get_protocol_classic() const
    {
        DBUG_ASSERT(m_protocol->type() == Protocol::PROTOCOL_TEXT ||
                    m_protocol->type() == Protocol::PROTOCOL_BINARY);

        return (Protocol_classic *)m_protocol;
    }
    void set_protocol(Protocol * protocol)
    {
        m_protocol= protocol;
    }
    inline bool is_classic_protocol()
    {
        // DBUG_ENTER("THD::is_classic_protocol");
        // DBUG_PRINT("info", ("type=%d", get_protocol()->type()));
        switch (get_protocol()->type())
        {
            case Protocol::PROTOCOL_BINARY:
            case Protocol::PROTOCOL_TEXT:
                DBUG_RETURN(true);
            default:
                break;
        }
        DBUG_RETURN(false);
    }
    void send_statement_status();
    void set_command(enum enum_server_command command);

    bool send_result_metadata(vector<Send_field*> *list, uint flags);

    /**
      Clear the current error, if any.
      We do not clear is_fatal_error or is_fatal_sub_stmt_error since we
      assume this is never called if the fatal error is set.
      @todo: To silence an error, one should use Internal_error_handler
      mechanism. In future this function will be removed.
    */
    inline void clear_error()
    {
        if (get_stmt_da()->is_error())
            get_stmt_da()->reset_diagnostics_area();
        // is_slave_error= false;
        DBUG_VOID_RETURN;
    }

    void reset_for_next_command();

    Sql_condition* raise_condition(uint sql_errno,
                                   const char* sqlstate,
                                   Sql_condition::enum_severity_level level,
                                   const char* msg,
                                   bool use_condition_handler);

    /**
      TRUE if there is an error in the error stack.

      Please use this method instead of direct access to
      net.report_error.

      If TRUE, the current (sub)-statement should be aborted.
      The main difference between this member and is_fatal_error
      is that a fatal error can not be handled by a stored
      procedure continue handler, whereas a normal error can.

      To raise this flag, use my_error().
    */
    inline bool is_error() const { return get_stmt_da()->is_error(); }

    /// Returns first Diagnostics Area for the current statement.
    Diagnostics_area *get_stmt_da()
    { return m_stmt_da; }

    /// Returns first Diagnostics Area for the current statement.
    const Diagnostics_area *get_stmt_da() const
    { return m_stmt_da; }

    bool store_globals();
    void restore_globals();

    // Security_context m_main_security_ctx;
    // Security_context *m_security_ctx;

    // Security_context *security_context() const { return m_security_ctx; }
    // void set_security_context(Security_context *sctx) { m_security_ctx = sctx; }

    bool store_user_var(const user_var_entry_ptr& userVarEntryPtr);
    user_var_entry_ptr get_user_var(const string& varName);

    inline int killed_errno() const
    {
        killed_state killed_val; /* to cache the volatile 'killed' */
        return (killed_val= killed) != KILL_BAD_DATA ? killed_val : 0;
    }
    inline void send_kill_message() const
    {
        int err= killed_errno();
        if (err && !get_stmt_da()->is_set())
        {
            if ((err == KILL_CONNECTION) && !abort_loop)
                err = KILL_QUERY;
            /*
              KILL is fatal because:
              - if a condition handler was allowed to trap and ignore a KILL, one
              could create routines which the DBA could not kill
              - INSERT/UPDATE IGNORE should fail: if KILL arrives during
              JOIN::optimize(), statement cannot possibly run as its caller expected
              => "OK" would be misleading the caller.
            */
            // my_message(err, ER(err), MYF(ME_FATALERROR));
            ARIES_EXCEPTION_SIMPLE( err, ER(err) );
        }
    }

    void cleanup_connection(void);

    void reset_tx()
    {
      m_tx = nullptr;
      m_explicitTx = false;
    }

public:
    aries_engine::AriesTransactionPtr m_tx;
    bool m_explicitTx = false;
    std::mutex m_lock_thd_sysvar;

private:
    bool m_release_resources_done;
    bool cleanup_done;
    void cleanup(void);


    Diagnostics_area main_da;
    Diagnostics_area *m_stmt_da;
    string m_db;
    char* m_user_name;
    Protocol *m_protocol;           // Current protocol
    /**
      SSL data attached to this connection.
      This is an opaque pointer,
      When building with SSL, this pointer is non NULL
      only if the connection is using SSL.
      When building without SSL, this pointer is always NULL.
      The SSL data can be inspected to read per thread
      status variables,
      and this can be inspected while the thread is running.
    */
    // SSL_handle m_SSL
    /**
      Type of current query: COM_STMT_PREPARE, COM_QUERY, etc.
      Set from first byte of the packet in do_command()
    */
    enum enum_server_command m_command;
    unordered_map<std::string, user_var_entry_ptr> m_user_vars;

    /**
    Protects THD data accessed from other threads.
    The attributes protected are:
    - thd->is_killable (used by KILL statement and shutdown).
    - thd->user_vars (user variables, inspected by monitoring)
    Is locked when THD is deleted.
  */
    mysql_mutex_t LOCK_thd_data;
};
extern "C" int thd_killed(const THD* thd);

#include "AriesEngine/AriesCommonExpr.h"
class user_var_entry {
    // char *m_ptr = nullptr;
    size_t m_length = 0;
    std::shared_ptr<aries_engine::AriesCommonExpr> ariesCommonExprUPtr;

    void reset_value() {
        m_length= 0;
    }

public:
    user_var_entry() {}                         /* Remove gcc warning */
    std::string entry_name;
    bool unsigned_flag;         // true if unsigned, false if signed

    /**
    Allocate and initialize a user variable instance.
    @param namec  Name of the variable.
    @param cs     Charset of the variable.
    @return
    @retval  Address of the allocated and initialized user_var_entry instance.
    @retval  NULL on allocation error.
  */
    static std::shared_ptr<user_var_entry> create(THD *thd, const std::string &name/*, const CHARSET_INFO *cs*/);
    /**
    Initialize all members
    @param name - Name of the user_var_entry instance.
    @cs         - charset information of the user_var_entry instance.
  */
    void init(THD *thd, const std::string &name/* , const CHARSET_INFO *cs*/)
    {
        DBUG_ASSERT(thd != NULL);
        // m_owner= thd;
        entry_name = name;
        reset_value();
        // update_query_id= 0;
        // collation.set(cs, DERIVATION_IMPLICIT, 0);
        unsigned_flag= 0;
        /*
          If we are here, we were called from a SET or a query which sets a
          variable. Imagine it is this:
          INSERT INTO t SELECT @a:=10, @a:=@a+1.
          Then when we have a Item_func_get_user_var (because of the @a+1) so we
          think we have to write the value of @a to the binlog. But before that,
          we have a Item_func_set_user_var to create @a (@a:=10), in this we mark
          the variable as "already logged" (line below) so that it won't be logged
          by Item_func_get_user_var (because that's not necessary).
        */
        // used_query_id= thd->query_id;
    }
    void setExpr(std::unique_ptr<aries_engine::AriesCommonExpr>& ariesCommonExprUPtr) {
        this->ariesCommonExprUPtr = std::move(ariesCommonExprUPtr);
    }
    const std::shared_ptr<aries_engine::AriesCommonExpr>& getExpr() const {
        return ariesCommonExprUPtr;
    }
    aries::BiaodashiPointer ToBiaodashi() const;
    size_t length() const { return m_length; }
};

/** A short cut for thd->get_stmt_da()->set_ok_status(). */

inline void
my_ok(THD *thd, ulonglong affected_rows= 0, ulonglong id= 0,
      const char *message= NULL)
{
    // thd->set_row_count_func(affected_rows);
    thd->get_stmt_da()->set_ok_status(affected_rows, id, message);
}


/** A short cut for thd->get_stmt_da()->set_eof_status(). */

inline void
my_eof(THD *thd)
{
    // thd->set_row_count_func(-1);
    thd->get_stmt_da()->set_eof_status(thd);
    // if (thd->variables.session_track_transaction_info > TX_TRACK_NONE)
    // {
    //     ((Transaction_state_tracker *)
    //             thd->session_tracker.get_tracker(TRANSACTION_INFO_TRACKER))
    //             ->add_trx_state(thd, TX_RESULT_SET);
    // }
}
bool IsCurrentThdKilled();
void SendKillMessage();

#endif
