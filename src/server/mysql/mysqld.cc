#include <glog/logging.h>
#include <server/mysql/include/sql_time.h>
#include <server/mysql/include/sql_const.h>
#include <server/mysql/include/mysql_thread.h>
#include <csignal>

#include "./include/mysqld.h"
#include "./include/my_sys.h"
#include "./include/sql_class.h"
#include "./include/derror.h"
#include "./include/my_thread_local.h"

#include "utils/mutex_lock.h"
#include "server/mysql/include/mysql_def.h"
#include "server/mysql/include/tztime.h"
using namespace mysql;

pthread_mutex_t LOCK_thread_ids;

pthread_mutex_t LOCK_socket_listener_active;
pthread_cond_t COND_socket_listener_active;

pthread_mutex_t LOCK_start_signal_handler;
pthread_cond_t COND_start_signal_handler;

/**
  The below lock protects access to two global server variables:
  max_prepared_stmt_count and prepared_stmt_count. These variables
  set the limit and hold the current total number of prepared statements
  in the server, respectively. As PREPARE/DEALLOCATE rate in a loaded
  server may be fairly high, we need a dedicated lock.
*/
mysql_mutex_t LOCK_prepared_stmt_count;

pthread_attr_t connection_attr;

struct system_variables global_system_variables;
struct system_variables max_system_variables;

char *opt_ssl_ca= NULL, *opt_ssl_capath= NULL, *opt_ssl_cert= NULL,
        *opt_ssl_cipher= NULL, *opt_ssl_key= NULL, *opt_ssl_crl= NULL,
        *opt_ssl_crlpath= NULL, *opt_tls_version= NULL;
char system_time_zone[30];


query_id_t global_query_id;

/* Thread specific variables */

thread_local_key_t THR_THD;
bool THR_THD_initialized= false;

CHARSET_INFO *default_charset_info = &my_charset_utf8_bin;
CHARSET_INFO *system_charset_info = &my_charset_utf8_bin;
CHARSET_INFO *character_set_filesystem = &my_charset_utf8_bin;
time_t server_start_time;
char* opt_init_connect;
ulong query_cache_limit;
ulong query_cache_min_res_unit= QUERY_CACHE_MIN_RESULT_DATA_SIZE;
my_bool opt_secure_auth= 0;
const char *tx_isolation_names[] =
        { "READ-UNCOMMITTED", "READ-COMMITTED", "REPEATABLE-READ", "SERIALIZABLE",
          NullS};
Date_time_format global_date_format, global_datetime_format, global_time_format;
Time_zone *default_tz;

char curr_dir[FN_REFLEN] = {0}, home_dir_buff[FN_REFLEN] = {0};
char *home_dir = 0;
char mysql_home[FN_REFLEN] = {0};
char* mysql_home_ptr = mysql_home;
char mysql_real_data_home[FN_REFLEN] = {0};
const char *mysql_real_data_home_ptr= mysql_real_data_home;

char *default_auth_plugin;
uint default_password_lifetime= 0;
char *my_bind_addr_str;

uint test_flags;
uint lower_case_table_names;
ulong back_log; // , connect_timeout, server_id;
ulong binlog_cache_size=0;
ulonglong  max_binlog_cache_size=0;
ulong slave_max_allowed_packet= 0;
ulong binlog_stmt_cache_size=0;
int32 opt_binlog_max_flush_queue_time= 0;
long opt_binlog_group_commit_sync_delay= 0;
ulong opt_binlog_group_commit_sync_no_delay_count= 0;
ulonglong  max_binlog_stmt_cache_size=0;
ulong query_cache_size=0;
my_bool sp_automatic_privileges= 1;
const char *binlog_format_names[]= {"MIXED", "STATEMENT", "ROW", NullS};
/**
  Limit of the total number of prepared statements in the server.
  Is necessary to protect the server against out-of-memory attacks.
*/
ulong max_prepared_stmt_count = 16382;
/**
  Current total number of prepared statements in the server. This number
  is exact, and therefore may not be equal to the difference between
  `com_stmt_prepare' and `com_stmt_close' (global status variables), as
  the latter ones account for all registered attempts to prepare
  a statement (including unsuccessful ones).  Prepared statements are
  currently connection-local: if the same SQL query text is prepared in
  two different connections, this counts as two distinct prepared
  statements.
*/
ulong prepared_stmt_count=0;

static int init_thread_environment();
static int mysql_init_variables(void);
char *strmake(char *dst, const char *src, size_t length);

extern pthread_t signal_thread_id;

void kill_mysql(void)
{
    DBUG_ENTER("kill_mysql");
    if (pthread_kill(signal_thread_id, SIGTERM))
    {
        LOG(INFO) << "Got error " << errno << " from pthread_kill"; /* purecov: inspected */
    }
    // DBUG_PRINT("quit",("After pthread_kill"));
    DBUG_VOID_RETURN;
}

int init_common_variables()
{
    {
        server_start_time = time(NULL);
        struct tm tm_tmp;
        localtime_r(&server_start_time,&tm_tmp);
        strmake(system_time_zone, tzname[tm_tmp.tm_isdst != 0 ? 1 : 0],
                sizeof(system_time_zone)-1);

        LOG(INFO) << "System time zone: " << system_time_zone;
    }
    if (init_thread_environment() ||
        mysql_init_variables())
        return 1;

    /*
    We set SYSTEM time zone as reasonable default and
    also for failure of my_tz_init() and bootstrap mode.
    If user explicitly set time zone with --default-time-zone
    option we will change this value in my_tz_init().
  */
    global_system_variables.time_zone= my_tz_OFFSET0;
    default_tz = my_tz_OFFSET0;

    /* Must be initialized early for comparison of options name */
    // global_system_variables.lc_messages= my_default_lc_messages;
    /* Set collactions that depends on the default collation */
    global_system_variables.collation_server = default_charset_info;
    global_system_variables.collation_database = default_charset_info;

    global_system_variables.collation_connection= default_charset_info;
    global_system_variables.character_set_filesystem= character_set_filesystem;
    global_system_variables.character_set_results= default_charset_info;
    global_system_variables.character_set_client= default_charset_info;

    global_system_variables.lc_messages = MY_LOCALE_ERRMSGS::getInstance();

    pthread_mutex_init(&LOCK_socket_listener_active, &fast_mutexattr);
    pthread_cond_init(&COND_socket_listener_active, nullptr);

    pthread_mutex_init(&LOCK_start_signal_handler, &fast_mutexattr);
    pthread_cond_init(&COND_start_signal_handler, nullptr);

    pthread_mutex_init(&LOCK_prepared_stmt_count, &fast_mutexattr);

    pthread_attr_init(&connection_attr);
    pthread_attr_setdetachstate(&connection_attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setscope(&connection_attr, PTHREAD_SCOPE_SYSTEM);

    // default_tz= default_tz_name ? global_system_variables.time_zone
    //                             : my_tz_SYSTEM;

    return 0;
}

/**
  Initialize MySQL global variables to default values.

  @note
    The reason to set a lot of global variables to zero is to allow one to
    restart the embedded server with a clean environment
    It's also needed on some exotic platforms where global variables are
    not set to 0 when a program starts.

    We don't need to set variables refered to in my_long_options
    as these are initialized by my_getopt.
*/

static int mysql_init_variables(void)
{
    return 0;
}

void clean_up(bool print_message)
{
    if (THR_THD_initialized)
    {
        THR_THD_initialized= false;
        (void) my_delete_thread_local_key(THR_THD);
    }
}

static int init_thread_environment()
{
    if (my_create_thread_local_key(&THR_THD,NULL))
    {
        LOG(ERROR) << ("Can't create thread-keys");
        return 1;
    }
    THR_THD_initialized= true;
    return 0;
}

void clean_up_mutexes()
{
    pthread_mutex_destroy(&LOCK_thread_ids);
}

