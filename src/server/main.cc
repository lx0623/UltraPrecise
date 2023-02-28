/*
 * main.c
 *
 *  Created on: May 31, 2019
 *      Author: tengjp
 */
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <sys/resource.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <server/mysql/include/mysqld_thd_manager.h>
#include "server/mysql/include/mysql_def.h"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

#include "version.h"

#include "mysql/include/m_string.h"
#include "mysql/include/mysqld.h"
#include "mysql/include/sql_const.h"
#include "mysql/include/sql_auth_cache.h"

#include "frontend/SQLExecutor.h"
#include "schema/SchemaManager.h"
#include "Compression/dict/AriesDictManager.h"

#include "utils/mutex_lock.h"
#include "utils/string_util.h"
#include "server/mysql/include/sys_vars.h"
#include "server/mysql/include/derror.h"
#include "AriesEngine/AriesDataCache.h"
#include "frontend/ViewManager.h"
#include "Configuration.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/transaction/AriesXLogRecoveryer.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "frontend/JsonExecutor.h"

#define MAXLISTEN 64
#define ARIES_SOMAXCONN 10000

static int ListenSocket[MAXLISTEN];

unsigned short SERVER_PORT = 3306;
#define DEFAULT_ARIES_UNIX_ADDR "/tmp/rateup_dev.sock"
char *aries_unix_port = nullptr;
int UNIX_SOCKET_INDEX = -1;

bool volatile abort_loop = false;

static volatile sig_atomic_t segfaulted = 0;
#define SIGNAL_FMT "signal %d"

pthread_t main_thread_id;
pthread_t signal_thread_id;

static bool socket_listener_active = false;

typedef void (*sigfunc)(int signo);
int init_common_variables();
extern void init_connection_handler();
static void init_signals();
static void start_signal_handler();
// static void signal_handler_child(int signo);
// static void signal_handler_quit(int signo);
// sigfunc
// signal_no_restart(int signo, sigfunc func);
void clean_up_mutexes(void);

static bool createLockFile(const string &path);
static int getUnixListenSocket();
static void startServerSocket(unsigned short port);
static int serverLoop();
void handleClientConnection(int cliFd, struct sockaddr *cliAddr, bool isUnix);
char *convert_dirname(char *to, const char *from, const char *from_end);
extern void InitDatabase(std::string db_name);
namespace aries_acc
{
    void InitCompareFunctionMatrix();
}

static const char *OPT_DATADIR_LONG = "--datadir=";
static const char *OPT_PORT_LONG = "--port=";
// static const char* OPT_DUMP_RDB = "--dump_rdb";
// static const char *OPT_DUMP_SCHEMA = "--dump-schema";
static const char *OPT_INITIALIZE = "--initialize";
static const char *OPT_INITIALIZE_INSECURE = "--initialize-insecure";
static const char *OPT_STRICT_MODE = "--strict-mode=";
// static const char* OPT_INIT_DB = "--init-db";
// static const char *OPT_CACHE_DB = "--cache-db";
static const char *OPT_EXECUTE_JSON = "--execute-json=";
static const char *OPT_LOAD_CONFIG = "--load-config=";

std::string DB_NAME;
static std::string PID_FILE = "/tmp/rateup_dev.pid";

// rateup specific variables
bool STRICT_MODE = true;

void print_help(const char *program_name)
{
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("OPTIONS:\n");
    printf("  -h, --help                 Print help messages and exit.\n");
    printf("  --initialize               Initialize Rateup server system databases and exit.\n");
    // printf("  --cache-db [db_name]        cache  the database.\n");
    printf("  -P, --port=#               Port number the server listens on.\n");
    printf("  --datadir=name             Path to the database root directory. Default is /var/rateup/data\n");
    // printf("  --dump-schema [db_name]    Dump database schema(s) and exit.\n");
    printf("  --strict-mode={true|false} Strict mode or not(default to true).\n");
    // printf("  --execute-json={json file path} Load & execute plan from json file\n");
    // printf("  --load-config={config file path} Load configuration from file\n");
    printf("  --version                  Show version.\n\n");

    printf("Examples:\n");
    printf("  1. Initialize and then start rateup server with the defatul data directory:\n");
    printf("    ./rateup --initialize\n");
    printf("    ./rateup\n");
    printf("  2. Initialize and then start rateup server with data directory /usr/local/rateup/data\n");
    printf("    ./rateup --initialize --datadir=/usr/local/rateup/data\n");
    printf("    ./rateup --datadir=/usr/local/rateup/data\n");

}

void cleanup(void)
{
    LOG(INFO) << "Cleanup...";
    if (aries_unix_port && strlen(aries_unix_port) > 1)
    {
        unlink(aries_unix_port);
    }

    unlink(PID_FILE.data());
}

static bool init_db()
{
    aries::SQLExecutor::Init();

    if (!aries::schema::SchemaManager::GetInstance()->LoadBaseSchama())
    {
        return false;
    }

    aries_engine::AriesInitialTableManager::GetInstance().DoPreCache();
    aries_engine::AriesInitialTableManager::GetInstance().CreatePrimaryKeyIndex();

    aries_engine::AriesXLogManager::GetInstance().SetUp();

    auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
    special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
    auto result = special_recoveryer->Recovery();
    ARIES_ASSERT(result, "cannot recovery(special) from xlog");

    if (!aries::schema::SchemaManager::GetInstance()->Load())
    {
        return false;
    }
    aries::ViewManager::GetInstance().Init();

    auto recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>();
    recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader());
    result = recoveryer->Recovery();
    ARIES_ASSERT(result, "cannot recovery from xlog");

    LoadAclUsers();

    return true;
}

bool InsertSysVarRows();
static int initialize()
{
#ifndef NDEBUG
    std::cout << "Initializing rateup...\n";
#endif
    int result = -1;

    aries::SQLExecutor::Init();

    aries_engine::AriesXLogManager::GetInstance().SetUp();

    if (!aries::schema::SchemaManager::GetInstance()->InitSchema()) {
        goto end;
    }

    if (!InsertSysVarRows()) {
        goto end;
    }
    /*
    else
    {
        auto special_recoveryer = std::make_shared<aries_engine::AriesXLogRecoveryer>(true);
        special_recoveryer->SetReader(aries_engine::AriesXLogManager::GetInstance().GetReader(true));
        auto ret = special_recoveryer->Recovery();
        ARIES_ASSERT(ret, "cannot recovery(special) from xlog");
    }
    */

    result = 0;
    std::cout << "\nInitialize OK.\n";
end:
    return result;
}

static std::string RESET_PERF_SCHEMA_SQL = R"(
DROP TABLE IF EXISTS `PROCESSLIST`;
CREATE TEMPORARY TABLE `PROCESSLIST` (
`ID` bigint(21) NOT NULL DEFAULT '0',
`USER` varchar(32) NOT NULL DEFAULT '',
`HOST` varchar(64) NOT NULL DEFAULT '',
`DB` varchar(64) DEFAULT NULL,
`COMMAND` varchar(16) NOT NULL DEFAULT '',
`TIME` int(7) NOT NULL DEFAULT '0',
`STATE` varchar(64) DEFAULT NULL,
`INFO` longtext
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;)";
static bool resetPerfSchema()
{
    // aries::schema::SchemaManager::GetInstance()->GetSchema()->SetInit( true );
    // auto sqlResultPtr = SQLExecutor::GetInstance()->ExecuteSQL(RESET_PERF_SCHEMA_SQL, "information_schema", true);
    // if (!sqlResultPtr->IsSuccess()) {
    //     LOG(ERROR) << "Reset performance schema failed.";
    // }
    // aries::schema::SchemaManager::GetInstance()->GetSchema()->SetInit( false );
    auto dbEntry = aries::schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName("information_schema");
    auto tableEntry = dbEntry->GetTableByName("processlist");
    // myrocks::AriesRdb::GetInstance( "information_schema" )->ClearTable( tableEntry );
    return true;
}

bool serverInitDone = false;
void InitGlog()
{
    string logDir = Configuartion::GetInstance().GetLogDirectory();
    boost::filesystem::path p(logDir);
    if (boost::filesystem::exists(p) && !boost::filesystem::is_directory(p))
    {
#ifndef NDEBUG
        cerr << "Error: " << logDir << " is not a directory" << endl;
#endif
        exit(EXIT_FAILURE);
    }
#ifndef NDEBUG
    cout << "Log directory: " << logDir << endl;
#endif
    if (!boost::filesystem::exists(p))
    {
#ifndef NDEBUG
        cout << "Creating log directory " << logDir << endl;
#endif
        boost::filesystem::create_directory(p);
    }

    //INFO, WARNING, ERROR, and FATAL correspond to 0, 1, 2, 3 respectively.
#ifdef NDEBUG
    //Log suppression level: messages logged at a lower level than this are suppressed.
    FLAGS_minloglevel = 0;
    //Log messages at a level <= this flag are buffered.
    FLAGS_logbuflevel = 0;
    //5 seconds,  which logs may be buffered for.
    FLAGS_logbufsecs = 5;
#else
    FLAGS_logbuflevel = -1;
#endif
    FLAGS_log_dir = logDir;
    google::InitGoogleLogging("rateup");
}

my_bool my_init_done = FALSE;
int my_umask = 0664, my_umask_dir = 0777;
my_bool my_thread_global_init();
void my_thread_global_end();
static ulong atoi_octal(const char *str)
{
    long int tmp;
    while (*str && my_isspace(&my_charset_latin1, *str))
        str++;
    str2int(str,
            (*str == '0' ? 8 : 10), /* Octalt or decimalt */
            0, INT_MAX, &tmp);
    return (ulong)tmp;
}

/**
  Initialize my_sys functions, resources and variables

  @return Initialization result
    @retval FALSE Success
    @retval TRUE  Error. Couldn't initialize environment
*/
my_bool my_init()
{
    char *str;

    if (my_init_done)
        return FALSE;

    my_init_done = TRUE;

    my_umask = 0640;     /* Default umask for new files */
    my_umask_dir = 0750; /* Default umask for new directories */

    /* Default creation of new files */
    if ((str = getenv("UMASK")) != 0)
        my_umask = (int)(atoi_octal(str) | 0600);
    /* Default creation of new dir's */
    if ((str = getenv("UMASK_DIR")) != 0)
        my_umask_dir = (int)(atoi_octal(str) | 0700);

    if (my_thread_global_init())
        return TRUE;

    if ((home_dir = getenv("HOME")) != 0)
        home_dir = intern_filename(home_dir_buff, home_dir);
    return FALSE;

} /* my_init */

/* End my_sys */

void my_end(int infoflag)
{
    if (!my_init_done)
        return;

    my_thread_global_end();

    my_init_done = FALSE;
} /* my_end */

extern THD *createPseudoThd();
bool g_isInitialize = false;
int main(int argc, char** argv) {
    // bool dumpSchema = false;
    bool isGtest = false;
    // bool isCacheDb = false;
    bool executeJson = false;
    string dbName, tableName;
    char *json_file_path = nullptr;
    char *config_file_path = nullptr;
    string dataDir;

    cout << "Version: " << VERSION_INFO_STRING << endl;

#ifdef ARIES_PROFILE
    cout << "run with profile\n";
#endif

    MY_LOCALE_ERRMSGS::Init();

    Global_THD_manager::create_instance();

    if (argc > 1)
    {
        if (strncmp("--version", argv[1], strlen("--version")) == 0)
        {
            SHOW_VERSION_SIMPLE();
            return 0;
        }
        for (int i = 1; i < argc; ++i)
        {
            if (!strncmp("--gtest_", argv[i], strlen("--gtest_")))
            {
#ifdef BUILD_TEST
                isGtest = true;
#else
                // cout << "gtest not built, try:\n\n\tcmake -DBUILD_TEST=ON ..\n\n"
                //      << endl;
                // return 0;
                LOG(ERROR) << "Unknown parameter:" << argv[i];
                print_help(argv[0]);
                exit(EXIT_FAILURE);
#endif
            }
            else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
            {
                print_help(argv[0]);
                exit(EXIT_SUCCESS);
            }
            else if (!strcmp(argv[i], OPT_INITIALIZE) || !strcmp(argv[i], OPT_INITIALIZE_INSECURE))
            {
                g_isInitialize = true;
            }
            /*
            else if (!strcmp(argv[i], OPT_CACHE_DB))
            {
                isCacheDb = true;
            }
            */
            else if (!strcmp(argv[i], "-P"))
            {
                if (i >= argc - 1)
                {
                    LOG(ERROR) << "Missing port number.";
                    print_help(argv[0]);
                    exit(EXIT_FAILURE);
                }
                char *endptr;
                long port = strtol(argv[i + 1], &endptr, 10);
                if ((errno == ERANGE && (port == LONG_MAX || port == LONG_MIN)) || (errno != 0 && port == 0))
                {
                    perror("strtol");
                    exit(EXIT_FAILURE);
                }
                if (endptr == argv[i + 1])
                {
                    LOG(ERROR) << "Wrong port number.";
                    exit(EXIT_FAILURE);
                }
                if (port > USHRT_MAX)
                {
                    LOG(ERROR) << "Port number is out of range.";
                    exit(EXIT_FAILURE);
                }
                SERVER_PORT = port;
                i += 1;
            }
            else if (!strncmp(argv[i], OPT_DATADIR_LONG, strlen(OPT_DATADIR_LONG)))
            {
                dataDir = argv[i] + strlen(OPT_DATADIR_LONG);
                if ('\0' == dataDir[0])
                {
                    LOG(ERROR) << "Wrong data dir.";
                    print_help(argv[0]);
                    exit(EXIT_FAILURE);
                }
            }
            else if (!strncmp(argv[i], OPT_PORT_LONG, strlen(OPT_PORT_LONG)))
            {
                char *endptr;
                long port = strtol(argv[i] + strlen(OPT_PORT_LONG), &endptr, 10);
                if ((errno == ERANGE && (port == LONG_MAX || port == LONG_MIN)) || (errno != 0 && port == 0))
                {
                    perror("strtol");
                    exit(EXIT_FAILURE);
                }
                if (endptr == argv[i] + strlen(OPT_PORT_LONG))
                {
                    LOG(ERROR) << "Wrong port number.";
                    exit(EXIT_FAILURE);
                }

                if (port <= 0 || port > USHRT_MAX)
                {
                    LOG(ERROR) << "Port number is out of range.";
                    exit(EXIT_FAILURE);
                }
                SERVER_PORT = port;
            }
            /*
            else if (!strncmp(argv[i], OPT_DUMP_SCHEMA, strlen(OPT_DUMP_SCHEMA)))
            {
                dumpSchema = true;
            }
            */
            else if (!strncmp(argv[i], OPT_STRICT_MODE, strlen(OPT_STRICT_MODE)))
            {
                const char *strict = argv[i] + strlen(OPT_STRICT_MODE);
                if (!strncasecmp(strict, "true", 4))
                    STRICT_MODE = true;
                else if (!strncasecmp(strict, "false", 5))
                    STRICT_MODE = false;
                else
                {
                    print_help(argv[0]);
                    exit(EXIT_FAILURE);
                }
            }
            else if (!strncmp(argv[i], OPT_EXECUTE_JSON, strlen(OPT_EXECUTE_JSON)))
            {
                json_file_path = strdup(argv[i] + strlen(OPT_EXECUTE_JSON));
                executeJson = true;
            }
            else if (!strncmp(argv[i], OPT_LOAD_CONFIG, strlen(OPT_LOAD_CONFIG)))
            {
                config_file_path = strdup(argv[i] + strlen(OPT_LOAD_CONFIG));
            }
            else if ('-' == argv[i][0])
            {
                LOG(ERROR) << "Unknown parameter:" << argv[i];
                print_help(argv[0]);
                exit(EXIT_FAILURE);
            }
            else
            {
                DB_NAME = argv[i];
            }
        }
    }

    if ( !dataDir.empty() )
        Configuartion::GetInstance().SetDataDirectory( dataDir );
    else
        dataDir = Configuartion::GetInstance().GetDataDirectory();

    InitGlog();

    LOG( INFO ) << "Version: " << VERSION_INFO_STRING;
    LOG( INFO ) << "Data dir: " << dataDir;

    if (config_file_path != nullptr)
    {
        if (!Configuartion::GetInstance().LoadFromFile(config_file_path))
        {
            LOG(ERROR) << std::string("cannot load config file: ") + config_file_path;
#ifndef NDEBUG
            std::cerr << "cannot load config file: " << config_file_path << std::endl;
#endif
        }
        else
        {
#ifndef NDEBUG
            std::cout << "loaded from config file: " << config_file_path << std::endl;
#endif
        }
    }

    if (isGtest || executeJson)
    {
      init_signals();
      sys_var_init();
      start_signal_handler();
      Configuartion::GetInstance().InitCudaEnv();

      aries_acc::InitCompareFunctionMatrix();

      if (init_common_variables()) {
        exit(MYSQLD_ABORT_EXIT);
      }
      THD *thd = createPseudoThd();
      thd->set_db( DB_NAME );
      if (!init_db()) {
        LOG(ERROR)
            << "Failed to start test; schema corrupted or not initialized.";
        exit(MYSQLD_ABORT_EXIT);
      }

        /*
        if (isCacheDb && !config_file_path)
        {
            if (DB_NAME.empty())
            {
                print_help(argv[0]);
                return 0;
            }
            try
            {
                // TODO: check if table exists
                DLOG(INFO) << "start caching db " + DB_NAME;
                aries_engine::AriesInitialTableManager::GetInstance().cacheTables(DB_NAME);
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "lineitem")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "orders")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "region")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "part")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "supplier")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "partsupp")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "customer")->CreatePrimaryKeyIndexIfNotExists();
                aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "nation")->CreatePrimaryKeyIndexIfNotExists();
            }
            catch (const AriesException &e)
            {
                printf("\nERROR %d: %s\n", e.errCode, e.errMsg.data());
                exit(EXIT_FAILURE);
            }
            catch (...)
            {
                printf("\nUnknown error when cache database %s.\n", DB_NAME.data());
                exit(EXIT_FAILURE);
            }
        }
        */

        if (executeJson)
        {
            aries::JsonExecutor executor;
            executor.Load(json_file_path);
            executor.Run();
            return 0;
        }

        testing::InitGoogleTest(&argc, argv);
        int ret = -1;
        try
        {
#ifndef NDEBUG
            cout << "RUN_ALL_TESTS" << endl;
#endif
            ret = RUN_ALL_TESTS();
            // aries::RocksDB::AriesRocksDB::CloseAllRocksDBs();
        }
        catch (...)
        {
            LOG(ERROR) << "exception caught";
        }

        aries_engine::AriesDataCache::GetInstance().removeAllCache();
        delete thd;
        aries_engine::AriesTransManager::GetInstance().Clear();

        return ret;
    }

    if (createLockFile(PID_FILE))
    {
        exit(MYSQLD_ABORT_EXIT);
    }

    if (0 != atexit(cleanup))
    {
        LOG(ERROR) << "cannot set exit function";
        exit(EXIT_FAILURE);
    }

    if (my_init())
    {
        LOG(ERROR) << "Failed to init rateup";
        exit(EXIT_FAILURE);
    }

    if (init_common_variables())
    {
        exit(MYSQLD_ABORT_EXIT);
    }

    (void)strmake(mysql_home, Configuartion::GetInstance().GetBaseDirectory().data(), sizeof(mysql_home) - 1);
    (void)strmake(mysql_real_data_home, dataDir.data(), sizeof(mysql_real_data_home) - 1);
    convert_dirname(mysql_home, mysql_home, NullS);
    convert_dirname(mysql_real_data_home, mysql_real_data_home, NullS);

    sys_var_init();

    THD *thd = createPseudoThd();

    if (g_isInitialize) {
        init_signals();
        Configuartion::GetInstance().InitCudaEnv();
        aries_acc::InitCompareFunctionMatrix();

        try
        {
            return initialize();
        }
        catch (const AriesException &e)
        {
            fprintf( stderr, "\nInitialize failed, %d: %s\n", e.errCode, e.errMsg.data() );
            exit(EXIT_FAILURE);
        }
        catch(const std::exception& e)
        {
            fprintf( stderr, "\nInitialize failed, %s\n", e.what() );
        }
        catch (...)
        {
            fprintf( stderr, "\nInitialize failed\n" );
            exit(EXIT_FAILURE);
        }
    }

    init_signals();
    start_signal_handler();
    Configuartion::GetInstance().InitCudaEnv();
    aries_acc::InitCompareFunctionMatrix();

    if (!init_db()) {
        LOG(ERROR) << "Failed to start rateup. Schema is not initialized or datadir is not correct.\n"
                   << "Specify the correct data directory with --datadir={path} or use `rateup --initialize [--datadir={path}]` to initialize rateup server.";
        exit(MYSQLD_ABORT_EXIT);
    }
    delete thd;
    thd = nullptr;

    /*
    if (isCacheDb && !config_file_path)
    {
        if (DB_NAME.empty())
        {
            print_help(argv[0]);
            return 0;
        }
        try
        {
            aries_engine::AriesInitialTableManager::GetInstance().cacheTables(DB_NAME);
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "lineitem")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "orders")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "region")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "part")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "supplier")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "partsupp")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "customer")->CreatePrimaryKeyIndexIfNotExists();
            aries_engine::AriesMvccTableManager::GetInstance().getMvccTable(DB_NAME, "nation")->CreatePrimaryKeyIndexIfNotExists();
        }
        catch (const AriesException &e)
        {
            printf("\nERROR %d: %s\n", e.errCode, e.errMsg.data());
            exit(EXIT_FAILURE);
        }
        catch (...)
        {
            printf("\nUnknown error when cache database %s.\n", DB_NAME.data());
            exit(EXIT_FAILURE);
        }
    }
    */

    /*
    if (dumpSchema)
    {
        aries::schema::SchemaManager::GetInstance()->GetSchema()->Dump(DB_NAME);
        aries_engine::AriesDataCache::GetInstance().removeAllCache();
        return 0;
    }
    */

    resetPerfSchema();

    init_connection_handler();

    startServerSocket(SERVER_PORT);

    cout << "rateup server started\n"
         << endl;

    serverInitDone = true;
    serverLoop();

    LOG(INFO) << "Exiting...";
    aries_engine::AriesTransManager::GetInstance().Clear();

    pthread_mutex_lock(&LOCK_socket_listener_active);
    // Notify the signal handler that we have stopped listening for connections.
    socket_listener_active = false;
    pthread_cond_broadcast(&COND_socket_listener_active);
    pthread_mutex_unlock(&LOCK_socket_listener_active);

    clean_up_mutexes();
    aries_engine::AriesDataCache::GetInstance().removeAllCache();

    int ret = 0;
    if (0 != signal_thread_id)
    {
        ret = pthread_join(signal_thread_id, NULL);
        signal_thread_id = 0;
    }
    if (0 != ret)
        LOG(WARNING) << "Could not join signal_thread. error: " << ret;

    my_end(0);

    return 0;
}

void startServerSocket(unsigned short port)
{
    struct addrinfo hints;
    struct addrinfo *result = NULL, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;     /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM; /* Stream socket */
    hints.ai_flags = AI_PASSIVE;     /* For wildcard IP address */
    hints.ai_protocol = IPPROTO_TCP;

    char portStr[32];
    snprintf(portStr, sizeof(portStr), "%d", port);

    int retval = getaddrinfo(NULL, portStr, &hints, &result);
    if (retval != 0)
    {
        LOG(ERROR) << "getaddrinfo failed: " << retval << ", " << gai_strerror(retval);
        exit(EXIT_FAILURE);
    }

    /* getaddrinfo() returns a list of address structures.
       Try each address until we successfully bind(2).
       If socket(2) (or bind(2)) fails, we (close the socket
       and) try the next address. */

    const socklen_t addrLen = INET6_ADDRSTRLEN;
    char addrBuff[addrLen];
    int listen_index = 0;
    for (int i = 0; i < MAXLISTEN; i++)
        ListenSocket[i] = -1;
    for (rp = result; rp != NULL; rp = rp->ai_next)
    {
        memset(addrBuff, 0, addrLen);
        if (AF_INET == rp->ai_addr->sa_family)
        {
            inet_ntop(rp->ai_addr->sa_family, &(((sockaddr_in *)rp->ai_addr)->sin_addr), addrBuff, addrLen);
        }
        else
        {
            inet_ntop(rp->ai_addr->sa_family, &(((sockaddr_in6 *)rp->ai_addr)->sin6_addr), addrBuff, addrLen);
        }
        int sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == -1)
        {
            LOG(ERROR) << "Cannot create socket.";
            continue;
        }

        int opt = 1;
        if ((setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                        (char *)&opt, sizeof(opt))) == -1)
        {
            close(sock);
            LOG(ERROR) << "failed to socket option SO_REUSEADDR.";
            continue;
        }
        if ((setsockopt(sock, SOL_SOCKET, SO_REUSEPORT,
                        (char *)&opt, sizeof(opt))) == -1)
        {
            close(sock);
            LOG(ERROR) << "failed to socket option SO_REUSEADDR.";
            continue;
        }
        if (bind(sock, rp->ai_addr, rp->ai_addrlen) == 0)
        {
            retval = listen(sock, ARIES_SOMAXCONN);
            if (retval < 0)
            {
                LOG(ERROR) << "failed to listen on " << addrBuff << ":" << port;
                continue;
            }
            ListenSocket[listen_index++] = sock;
            LOG(INFO) << "Listening on " << addrBuff << ":" << port;
        }
        else
        {
            LOG(ERROR) << "Cannot bind socket on " << addrBuff << ":" << port << ", error " << errno << ", " << strerror(errno);
            close(sock);
        }
    }
    if (0 == listen_index)
    {
        LOG(ERROR) << "Cannot start server.";
        exit(-1);
    }

    freeaddrinfo(result);
    if (listen_index < MAXLISTEN)
    {
        int unixSocket = getUnixListenSocket();
        if (-1 != unixSocket)
        {
            ListenSocket[listen_index] = unixSocket;
            UNIX_SOCKET_INDEX = listen_index;
        }
    }

    pthread_mutex_lock(&LOCK_socket_listener_active);
    // Make it possible for the signal handler to kill the listener.
    socket_listener_active = true;
    pthread_mutex_unlock(&LOCK_socket_listener_active);
}

bool createLockFile(const string &path)
{
    int fd;
    char buffer[8];
    pid_t cur_pid = getpid();

    int retries = 3;
    while (true)
    {
        if (!retries--)
        {
            LOG(ERROR) << "Unable to create lock file " << path << " after retries.";
            return true;
        }

        fd = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);

        if (fd >= 0)
            break;

        if (errno != EEXIST)
        {
            LOG(ERROR) << "Could not create lock file " << path;

            return true;
        }

        fd = open(path.c_str(), O_RDONLY, 0600);
        if (fd < 0)
        {
            LOG(ERROR) << "Could not open lock file " << path;
            return true;
        }

        ssize_t len;
        if ((len = read(fd, buffer, sizeof(buffer) - 1)) < 0)
        {
            LOG(ERROR) << "Could not read lock file " << path;
            close(fd);
            return true;
        }

        close(fd);

        if (len == 0)
        {
            LOG(ERROR) << "Pid file is empty " << path;
            return true;
        }
        buffer[len] = '\0';

        pid_t parent_pid = getppid();
        pid_t read_pid = atoi(buffer);

        if (read_pid <= 0)
        {
            LOG(ERROR) << "Invalid pid in lock file" << path;
            return true;
        }

        if (read_pid != cur_pid && read_pid != parent_pid)
        {
            if (kill(read_pid, 0) == 0)
            {
                LOG(ERROR) << "Another instance with pid " << static_cast<int>(read_pid) << " is already running.";
                return true;
            }
        }

        /*
          Unlink the lock file as it is not associated with any process and
          retry.
        */
        if (unlink(path.c_str()) < 0)
        {
            LOG(ERROR) << "Could not remove lock file " << path;
            return true;
        }
    }

    snprintf(buffer, sizeof(buffer), "%d\n", static_cast<int>(cur_pid));
    if (write(fd, buffer, strlen(buffer)) !=
        static_cast<signed>(strlen(buffer)))
    {
        close(fd);
        LOG(ERROR) << "Could not write lock file " << path << ", errno " << errno;
        return true;
    }

    if (fsync(fd) != 0)
    {
        close(fd);
        LOG(ERROR) << "Could not sync lock file " << path << ", errno " << errno;
        return true;
    }

    if (close(fd) != 0)
    {
        LOG(ERROR) << "Could not close lock file " << path << ", errno " << errno;
        return true;
    }

    return false;
}

int getUnixListenSocket()
{
    char *env;
    aries_unix_port = (char *)DEFAULT_ARIES_UNIX_ADDR;
    if ((env = getenv("RATEUP_UNIX_ADDR")))
        aries_unix_port = env;
    if (strlen(aries_unix_port) < 2)
    {
        return -1;
    }
    struct stat sts;
    if (0 == (stat(aries_unix_port, &sts))) // already exists
    {
        if (!S_ISDIR(sts.st_mode))
        {
            LOG(INFO) << "Rateup unix socket address already exists, UNIX Socket is not started: " << aries_unix_port;
        }
        else
        {
            LOG(ERROR) << "Rateup unix socket address is directory, UNIX Socket is not started: " << aries_unix_port;
        }
        // don't unlink the path
        aries_unix_port = nullptr;
        return -1;
    }

    struct sockaddr_un UNIXaddr;
    LOG(INFO) << "UNIX Socket is " << aries_unix_port;

    // Check path length, probably move to set unix port?
    if (strlen(aries_unix_port) > (sizeof(UNIXaddr.sun_path) - 1))
    {
        LOG(ERROR) << "The socket file path "
                   << (uint)sizeof(UNIXaddr.sun_path) - 1
                   << "is too long (> " << aries_unix_port << ")";
        return -1;
    }

    // std::string lock_filename= aries_unix_port;
    // lock_filename += ".lock";
    // if (createLockFile(lock_filename))
    // {
    //     fprintf(stderr, "Unable to setup unix socket lock file.\n");
    //     return -1;
    // }

    int listener_socket = socket(AF_UNIX, SOCK_STREAM, 0);

    memset(&UNIXaddr, 0, sizeof(UNIXaddr));
    UNIXaddr.sun_family = AF_UNIX;
    stpcpy(UNIXaddr.sun_path, aries_unix_port);
    (void)unlink(aries_unix_port);

    // Set socket option SO_REUSEADDR
    int option_enable = 1;
    (void)setsockopt(listener_socket, SOL_SOCKET, SO_REUSEADDR,
                     (char *)&option_enable, sizeof(option_enable));
    // bind
    umask(0);
    if (bind(listener_socket,
             reinterpret_cast<struct sockaddr *>(&UNIXaddr),
             sizeof(UNIXaddr)) < 0)
    {
        LOG(ERROR) << "Can't start server : Bind on unix socket: " << strerror(errno);
        LOG(ERROR) << "Do you already have another rateup server running on socket: " << aries_unix_port << "?";
        close(listener_socket);
        return -1;
    }
    umask(((~my_umask) & 0666));

    // listen
    if (listen(listener_socket, (int)900) < 0)
        LOG(ERROR) << "listen() on Unix socket failed with error " << socket_errno;

    return listener_socket;
}

/*
 * Initialise the masks for select() for the ports we are listening on.
 * Return the number of sockets to listen on.
 */
static int
initMasks(fd_set *rmask)
{
    int maxsock = -1;
    int i;

    FD_ZERO(rmask);

    for (i = 0; i < MAXLISTEN; i++)
    {
        int fd = ListenSocket[i];

        if (fd == -1)
            break;
        FD_SET(fd, rmask);

        if (fd > maxsock)
            maxsock = fd;
    }

    return maxsock + 1;
}

/* Produce a core for the thread */
void my_write_core(int sig)
{
    signal(sig, SIG_DFL);
    pthread_kill(pthread_self(), sig);
}

/**
 * Handler for fatal signals
 *
 * Fatal events (seg.fault, bus error etc.) will trigger
 * this signal handler.  The handler will try to dump relevant
 * debugging information to stderr and dump a core image.
 *
 * Signal handlers can only use a set of 'safe' system calls
 * and library functions.  A list of safe calls in POSIX systems
 * are available at:
 *  http://pubs.opengroup.org/onlinepubs/009695399/functions/xsh_chap02_04.html
 * For MS Windows, guidelines are available at:
 *  http://msdn.microsoft.com/en-us/library/xdkz3x12(v=vs.71).aspx
 *
 * @param sig Signal number
*/
extern "C"
{
    void handle_fatal_signal(int sig)
    {
        if (segfaulted)
        {
            printf("Fatal " SIGNAL_FMT " while backtracing\n", sig);
            _exit(-1); /* Quit without running destructors */
        }
        segfaulted = 1;
        printf("Fatal " SIGNAL_FMT "\n", sig);
        my_write_core(sig);
        _exit(-1);
    }
    static void empty_signal_handler(int sig MY_ATTRIBUTE((unused)))
    {
    }
}

void init_signals()
{
    struct sigaction sa;
    (void)sigemptyset(&sa.sa_mask);

    // Change limits so that we will get a core file.
    struct rlimit rl;
    rl.rlim_cur = rl.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_CORE, &rl))
    {
#ifndef NDEBUG
        cout << "setrlimit could not change the size of core files to"
                " 'infinity';  We may not be able to generate a"
                " core file on signals";
#endif
    }

    /*
      SA_RESETHAND resets handler action to default when entering handler.
      SA_NODEFER allows receiving the same signal during handler.
      E.g. SIGABRT during our signal handler will dump core (default action).
    */
    // sa.sa_flags= SA_RESETHAND | SA_NODEFER;
    // sa.sa_handler= handle_fatal_signal;
    // Treat all these as fatal and handle them.
    // (void) sigaction(SIGSEGV, &sa, NULL);
    // (void) sigaction(SIGABRT, &sa, NULL);
    // (void) sigaction(SIGBUS, &sa, NULL);
    // (void) sigaction(SIGILL, &sa, NULL);
    // (void) sigaction(SIGFPE, &sa, NULL);

    // Ignore SIGPIPE and SIGALRM
    sa.sa_flags = 0;
    sa.sa_handler = SIG_IGN;
    (void)sigaction(SIGPIPE, &sa, NULL);
    (void)sigaction(SIGALRM, &sa, NULL);

    // SIGUSR1 is used to interrupt the socket listener.
    sa.sa_handler = empty_signal_handler;
    (void)sigaction(SIGUSR1, &sa, NULL);

    // Fix signals if ignored by parents (can happen on Mac OS X).
    sa.sa_handler = SIG_DFL;
    (void)sigaction(SIGTERM, &sa, NULL);
    (void)sigaction(SIGHUP, &sa, NULL);

    sigset_t set;
    (void)sigemptyset(&set);
    /*
      Block SIGQUIT, SIGHUP and SIGTERM.
      The signal handler thread does sigwait() on these.
    */
    (void)sigaddset(&set, SIGQUIT);
    (void)sigaddset(&set, SIGHUP);
    (void)sigaddset(&set, SIGTERM);
    (void)sigaddset(&set, SIGTSTP);
    /*
      Block SIGINT unless debugging to prevent Ctrl+C from causing
      unclean shutdown of the server.
    */
    // if (!(test_flags & TEST_SIGINT))
    (void)sigaddset(&set, SIGINT);
    pthread_sigmask(SIG_SETMASK, &set, NULL);
}

/** This thread handles SIGTERM, SIGQUIT and SIGHUP signals. */
extern "C" void *signal_handler(void *arg)
{
    // my_thread_init();

    sigset_t set;
    (void)sigemptyset(&set);
    (void)sigaddset(&set, SIGTERM);
    (void)sigaddset(&set, SIGQUIT);
    // (void) sigaddset(&set, SIGINT);
    (void)sigaddset(&set, SIGHUP);

    /*
      Signal to start_signal_handler that we are ready.
      This works by waiting for start_signal_handler to free mutex,
      after which we signal it that we are ready.
    */
    pthread_mutex_lock(&LOCK_start_signal_handler);
    pthread_cond_broadcast(&COND_start_signal_handler);
    pthread_mutex_unlock(&LOCK_start_signal_handler);

#ifndef NDEBUG
    cout << "signal handler thread started" << endl;
#endif

    for (;;)
    {
        int sig;
        while (sigwait(&set, &sig) == EINTR)
        {
        }
#ifndef NDEBUG
        cout << "Got signal: " << sig << ", abort_loop: " << abort_loop;
#endif
        switch (sig)
        {
        case SIGTERM:
        case SIGQUIT:
        case SIGINT:
            if (!abort_loop)
            {
                abort_loop = true; // Mark abort for threads.
                /*
                      Kill the socket listener.
                      The main thread will then set socket_listener_active= false,
                      and wait for us to finish all the cleanup below.
                    */
                pthread_mutex_lock(&LOCK_socket_listener_active);
                while (socket_listener_active)
                {
#ifndef NDEBUG
                    cout << "Killing socket listener" << endl;
#endif
                    if (pthread_kill(main_thread_id, SIGUSR1))
                    {
                        DBUG_ASSERT(false);
                        break;
                    }
                    pthread_cond_wait(&COND_socket_listener_active,
                                      &LOCK_socket_listener_active);
                }
                pthread_mutex_unlock(&LOCK_socket_listener_active);

                // close_connections();
            }
            pthread_exit(0);
            return NULL; // Avoid compiler warnings
            break;
        case SIGHUP:
            if (!abort_loop)
            {
                // int not_used;
                // reload_acl_and_cache(NULL,
                //                      (REFRESH_LOG | REFRESH_TABLES | REFRESH_FAST |
                //                       REFRESH_GRANT | REFRESH_THREADS | REFRESH_HOSTS),
                //                      NULL, &not_used); // Flush logs
                // // Reenable query logs after the options were reloaded.
                // query_logger.set_handlers(log_output_options);
            }
            break;
        default:
            break; /* purecov: tested */
        }
    }
    return NULL; /* purecov: deadcode */
}

static void start_signal_handler()
{
    main_thread_id = pthread_self();
    pthread_setname_np(main_thread_id, "rateup-main");

    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    pthread_mutex_lock(&LOCK_start_signal_handler);
    pthread_create(&signal_thread_id, &thread_attr, signal_handler, 0);
    pthread_setname_np(signal_thread_id, "rateup-sig");
    pthread_cond_wait(&COND_start_signal_handler, &LOCK_start_signal_handler);
    pthread_mutex_unlock(&LOCK_start_signal_handler);
    pthread_attr_destroy(&thread_attr);
}

static int serverLoop()
{

    fd_set readmask, fdSetTmp;
    int nSockets = initMasks(&readmask);

    // signal_no_restart(SIGCHLD, signal_handler_child);
    // signal_no_restart(SIGINT, signal_handler_quit);
    // signal_no_restart(SIGQUIT, signal_handler_quit);
    // signal_no_restart(SIGTERM, signal_handler_quit);

    int nReady = 0;
    while (!abort_loop)
    {
        memcpy((char *)&fdSetTmp, (char *)&readmask, sizeof(fd_set));
        nReady = select(nSockets, &fdSetTmp, NULL, NULL, NULL);
        if (nReady < 0)
        {
            if (errno != EINTR && errno != EWOULDBLOCK)
            {
                LOG(ERROR) << "select failed: " << errno << strerror(errno);
                exit(-1);
            }
        }
        else if (nReady > 0)
        {
            for (int i = 0; i < MAXLISTEN; i++)
            {
                if (ListenSocket[i] == -1)
                    break;
                if (FD_ISSET(ListenSocket[i], &fdSetTmp))
                {
                    struct sockaddr cliAddr;
                    socklen_t cliAddrLen = sizeof(sockaddr);
                    int cliFd = accept(ListenSocket[i], &cliAddr, &cliAddrLen);
                    if (-1 == cliFd)
                    {
                        LOG(ERROR) << "Failed to accept client connection: " << errno << "(" << strerror(errno) << ")";
                        continue;
                    }
                    int v = 1;
                    if (i != UNIX_SOCKET_INDEX && setsockopt(cliFd, IPPROTO_TCP, TCP_NODELAY, (char *)&v, sizeof(v)) < 0)
                    {
                        LOG(ERROR) << "setsockopt TCP_NODELAY failed";
                        continue;
                    }
                    v = 1;
                    if (setsockopt(cliFd, SOL_SOCKET, SO_KEEPALIVE, (char *)&v, sizeof(v)) < 0)
                    {
                        LOG(ERROR) << "setsockopt SO_KEEPALIVE failed";
                        continue;
                    }
                    handleClientConnection(cliFd, &cliAddr, i == UNIX_SOCKET_INDEX);
                }
            }
        }
    }
    LOG(INFO) << "End server loop.";
    return 0;
}

int set_socket_nonblock(int fd)
{
    int ret = 0;
    int fdFlags = fcntl(fd, F_GETFL, 0);
    if (fdFlags < 0)
        return errno;
#if defined(O_NONBLOCK)
    fdFlags |= O_NONBLOCK;
#elif defined(O_NDELAY)
    fd_flags |= O_NDELAY;
#elif defined(O_FNDELAY)
    fd_flags |= O_FNDELAY;
#else
#error "No definition of non-blocking flag found."
#endif
    if (fcntl(fd, F_SETFL, fdFlags) == -1)
        ret = errno;
    return ret;
}

void *handle_connection(void *arg);
void handleClientConnection(int cliFd, struct sockaddr *cliAddr, bool isUnix)
{
    fflush(stdout);
    fflush(stderr);

    auto connArg = new CONN_ARG();
    int port = 0;
    if (AF_INET == cliAddr->sa_family)
    {
        inet_ntop(AF_INET, &(((sockaddr_in *)cliAddr)->sin_addr), connArg->client_addr, INET_ADDRSTRLEN);
        port = ntohs(((sockaddr_in *)cliAddr)->sin_port);
        LOG(INFO) << "Client connected, " << connArg->client_addr << ":" << port;
    }
    else if (AF_INET6 == cliAddr->sa_family)
    {
        inet_ntop(AF_INET6, &(((sockaddr_in6 *)cliAddr)->sin6_addr), connArg->client_addr, INET6_ADDRSTRLEN);
        port = ntohs(((sockaddr_in6 *)cliAddr)->sin6_port);
        LOG(INFO) << "Client connected, " << connArg->client_addr << ":" << port;
    }
    else
    {
        LOG(INFO) << "Client connected(unix socket)";
    }

    pthread_t tid;
    connArg->unix_sock = isUnix;
    connArg->client_port = port;
    connArg->client_fd = cliFd;
    pthread_create(&tid, &connection_attr, handle_connection, (void *)connArg);
    std::string tname = "rateup-conn";
    tname.append(std::to_string(port));
    pthread_setname_np(tid, tname.data());
}

void pr_exit(int status)
{
    if (WIFEXITED(status))
        printf("normal termination, exit status = %d\n",
               WEXITSTATUS(status));
    else if (WIFSIGNALED(status))
        printf("abnormal termination, signal number = %d%s\n",
               WTERMSIG(status),
#ifdef WCOREDUMP
               WCOREDUMP(status) ? " (core file generated)" : "");
#else
               "");
#endif
    else if (WIFSTOPPED(status))
        printf("child stopped, signal number = %d\n",
               WSTOPSIG(status));
}

// static void signal_handler_child(int signo)
// {
//     int pid, exitstatus;
//     while ((pid = waitpid(-1, &exitstatus, WNOHANG)) > 0)
// 	{
//         printf("Child terminated: %d, %d.\n", pid, exitstatus);
//         pr_exit(exitstatus);
//     }
// }

// static void signal_handler_quit(int signo)
// {
//     printf("Server exit.\n");
//     if ("" != aries_unix_port)
//     {
//         unlink(aries_unix_port);
//         std::string lock_filename = aries_unix_port;
//         lock_filename += ".lock";
//         unlink(lock_filename.c_str());
//     }
//     _exit(0);
// }

// sigfunc
// signal_no_restart(int signo, sigfunc func)
// {
//     struct sigaction act, oact;
//     act.sa_handler = func;
//     sigemptyset(&act.sa_mask);
//     act.sa_flags = 0;
// #ifdef SA_NOCLDSTOP
//     if (signo == SIGCHLD)
//         act.sa_flags |= SA_NOCLDSTOP;
// #endif
//     if (sigaction(signo, &act, &oact) < 0)
//         return SIG_ERR;
//     return oact.sa_handler;
// }
