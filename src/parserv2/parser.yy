%{
#include "common.h"
#include "driver.h"
#include "scanner.h"
#include "utils/string_util.h"


void yyerror(const char * msg);

/* this "connects" the bison parser in the driver to the flex scanner class
 * object. it defines the yylex() function call to pull the next token from the
 * current lexer object of the driver context. */
#undef yylex
#define yylex driver.lexer->lex

#define ADD_SELECT_ITEM(list, part) \
do { \
  std::shared_ptr<string> alias_ptr = nullptr; \
  auto expr = std::get<0>(part); \
  auto alias = std::get<1>(part); \
  if (!alias.empty()) { \
    alias_ptr = std::make_shared<string>(alias); \
  } \
  list->AddSelectExpr(expr, alias_ptr); \
} while(0)

using namespace aries_parser;

%}

%require "3.5"
%language "c++"
// %debug

%define api.prefix {aries_parser}

/* set the parser's class identifier */
%define api.parser.class {Parser}

%define api.value.type variant

%defines
%locations
%define api.location.type {aries_parser::location}
%code requires { #include "location.h" }
%initial-action
{
    // initialize the initial location object
    // lichi: don't track file name to save the memory
    /*@$.begin.filename = @$.end.filename = &driver.streamname;*/
};


/* The driver is passed by reference to the parser and to the scanner. This
 * provides a simple but effective pure interface, not relying on global
 * variables. */
%parse-param { class Driver& driver }

/*
   Tokens from MySQL 5.7, keep in alphabetical order.
*/

%token  ABORT_SYM                     /* INTERNAL (used in lex) */
%token  ACCESSIBLE_SYM
%token<string> ACCOUNT_SYM
%token<string> ACTION                /* SQL-2003-N */
%token  ADD                           /* SQL-2003-R */
%token<string> ADDDATE_SYM           /* MYSQL-FUNC */
%token<string> AFTER_SYM             /* SQL-2003-N */
%token<string> AGAINST
%token<string> AGGREGATE_SYM
%token<string> ALGORITHM_SYM
%token  ALL                           /* SQL-2003-R */
%token  ALTER                         /* SQL-2003-R */
%token<string> ALWAYS_SYM
%token  OBSOLETE_TOKEN_271            /* was: ANALYSE_SYM */
%token  ANALYZE_SYM
%token  AND_AND_SYM                   /* OPERATOR */
%token  AND_SYM                       /* SQL-2003-R */
%token<string> ANY_SYM               /* SQL-2003-R */
%token  AS                            /* SQL-2003-R */
%token<string> ASC                           /* SQL-2003-N */
%token<string> ASCII_SYM             /* MYSQL-FUNC */
%token  ASENSITIVE_SYM                /* FUTURE-USE */
%token<string> AT_SYM                /* SQL-2003-R */
%token<string> AUTOEXTEND_SIZE_SYM
%token<string> AUTO_INC
%token<string> AVG_ROW_LENGTH
%token<string> AVG_SYM               /* SQL-2003-N */
%token<string> BACKUP_SYM
%token  BEFORE_SYM                    /* SQL-2003-N */
%token<string> BEGIN_SYM             /* SQL-2003-R */
%token  BETWEEN_SYM                   /* SQL-2003-R */
%token  BIGINT_SYM                    /* SQL-2003-R */
%token  BINARY_SYM                    /* SQL-2003-R */
%token<string> BINLOG_SYM
%token  BIN_NUM
%token  BIT_AND                       /* MYSQL-FUNC */
%token  BIT_OR                        /* MYSQL-FUNC */
%token<string> BIT_SYM               /* MYSQL-FUNC */
%token  BIT_XOR                       /* MYSQL-FUNC */
%token  BLOB_SYM                      /* SQL-2003-R */
%token<string> BLOCK_SYM
%token<string> BOOLEAN_SYM           /* SQL-2003-R */
%token<string> BOOL_SYM
%token  BOTH                          /* SQL-2003-R */
%token<string> BTREE_SYM
%token  BY                            /* SQL-2003-R */
%token<string> BYTE_SYM
%token<string> CACHE_SYM
%token  CALL_SYM                      /* SQL-2003-R */
%token  CASCADE                       /* SQL-2003-N */
%token<string> CASCADED              /* SQL-2003-R */
%token  CASE_SYM                      /* SQL-2003-R */
%token  CAST_SYM                      /* SQL-2003-R */
%token<string> CATALOG_NAME_SYM      /* SQL-2003-N */
%token<string> CHAIN_SYM             /* SQL-2003-N */
%token  CHANGE
%token<string> CHANGED
%token<string> CHANNEL_SYM
%token<string> CHARSET
%token  CHAR_SYM                      /* SQL-2003-R */
%token<string> CHECKSUM_SYM
%token  CHECK_SYM                     /* SQL-2003-R */
%token<string> CIPHER_SYM
%token<string> CLASS_ORIGIN_SYM      /* SQL-2003-N */
%token<string> CLIENT_SYM
%token<string> CLOSE_SYM             /* SQL-2003-R */
%token<string> COALESCE              /* SQL-2003-N */
%token<string> CODE_SYM
%token  COLLATE_SYM                   /* SQL-2003-R */
%token<string> COLLATION_SYM         /* SQL-2003-N */
%token<string> COLUMNS
%token  COLUMN_SYM                    /* SQL-2003-R */
%token<string> COLUMN_FORMAT_SYM
%token<string> COLUMN_NAME_SYM       /* SQL-2003-N */
%token<string> COMMENT_SYM
%token<string> COMMITTED_SYM         /* SQL-2003-N */
%token<string> COMMIT_SYM            /* SQL-2003-R */
%token<string> COMPACT_SYM
%token<string> COMPLETION_SYM
%token<string> COMPRESSED_SYM
%token<string> COMPRESSION_SYM
%token<string> ENCRYPTION_SYM
%token<string> CONCURRENT
%token  CONDITION_SYM                 /* SQL-2003-R, SQL-2008-R */
%token CONNECTION_ID_SYM
%token<string> CONNECTION_SYM
%token<string> CONSISTENT_SYM
%token  CONSTRAINT                    /* SQL-2003-R */
%token<string> CONSTRAINT_CATALOG_SYM /* SQL-2003-N */
%token<string> CONSTRAINT_NAME_SYM   /* SQL-2003-N */
%token<string> CONSTRAINT_SCHEMA_SYM /* SQL-2003-N */
%token<string> CONTAINS_SYM          /* SQL-2003-N */
%token<string> CONTEXT_SYM
%token  CONTINUE_SYM                  /* SQL-2003-R */
%token  CONVERT_SYM                   /* SQL-2003-N */
%token  COUNT_SYM                     /* SQL-2003-N */
%token<string> CPU_SYM
%token<string>  CREATE                        /* SQL-2003-R */
%token  CROSS                         /* SQL-2003-R */
%token  CUBE_SYM                      /* SQL-2003-R */
%token  CURDATE                       /* MYSQL-FUNC */
%token<string> CURRENT_SYM           /* SQL-2003-R */
%token  CURRENT_USER                  /* SQL-2003-R */
%token  CURSOR_SYM                    /* SQL-2003-R */
%token<string> CURSOR_NAME_SYM       /* SQL-2003-N */
%token  CURTIME                       /* MYSQL-FUNC */
%token  DATABASE
%token  DATABASES
%token<string> DATAFILE_SYM
%token<string> DATA_SYM              /* SQL-2003-N */
%token<string> DATETIME_SYM          /* MYSQL */
%token  DATE_ADD_INTERVAL             /* MYSQL-FUNC */
%token  DATE_SUB_INTERVAL             /* MYSQL-FUNC */
%token<string> DATE_SYM              /* SQL-2003-R */
%token  DAY_HOUR_SYM
%token  DAY_MICROSECOND_SYM
%token  DAY_MINUTE_SYM
%token  DAY_SECOND_SYM
%token<string> DAY_SYM               /* SQL-2003-R */
%token<string> DEALLOCATE_SYM        /* SQL-2003-R */
%token  DECIMAL_NUM
%token  REAL_NUM
%token  DECIMAL_SYM                   /* SQL-2003-R */
%token  DECLARE_SYM                   /* SQL-2003-R */
%token  DEFAULT_SYM                   /* SQL-2003-R */
%token<string> DEFAULT_AUTH_SYM      /* INTERNAL */
%token<string> DEFINER_SYM
%token  DELAYED_SYM
%token<string> DELAY_KEY_WRITE_SYM
%token  DELETE_SYM                    /* SQL-2003-R */
%token<string> DESC                          /* SQL-2003-N */
%token  DESCRIBE                      /* SQL-2003-R */
%token  OBSOLETE_TOKEN_388            /* was: DES_KEY_FILE */
%token  DETERMINISTIC_SYM             /* SQL-2003-R */
%token<string> DIAGNOSTICS_SYM       /* SQL-2003-N */
%token<string> BYTEDICT_SYM
%token<string> SHORTDICT_SYM
%token<string> INTDICT_SYM
%token  DICT_INDEX_SYM
%token<string> DIRECTORY_SYM
%token<string> DISABLE_SYM
%token<string> DISCARD_SYM           /* MYSQL */
%token<string> DISK_SYM
%token  DISTINCT                      /* SQL-2003-R */
%token  DIV_SYM
%token  DOUBLE_SYM                    /* SQL-2003-R */
%token<string> DO_SYM
%token  DROP                          /* SQL-2003-R */
%token  DUAL_SYM
%token<string> DUMPFILE
%token<string> DUPLICATE_SYM
%token<string> DYNAMIC_SYM           /* SQL-2003-R */
%token  EACH_SYM                      /* SQL-2003-R */
%token  ELSE                          /* SQL-2003-R */
%token  ELSEIF_SYM
%token<string> ENABLE_SYM
%token  ENCLOSED
%token  ENCODING
%token<string> END                   /* SQL-2003-R */
%token<string> ENDS_SYM
%token  END_OF_INPUT  0               /* INTERNAL */
%token<string> ENGINES_SYM
%token<string> ENGINE_SYM
%token<string> ENUM_SYM              /* MYSQL */
%token  EQ                            /* OPERATOR */
%token  EQUAL_SYM                     /* OPERATOR */
%token<string> ERROR_SYM
%token<string> ERRORS
%token  ESCAPED
%token<string> ESCAPE_SYM            /* SQL-2003-R */
%token<string> EVENTS_SYM
%token<string> EVENT_SYM
%token<string> EVERY_SYM             /* SQL-2003-N */
%token<string> EXCHANGE_SYM
%token<string> EXECUTE_SYM           /* SQL-2003-R */
%token  EXISTS                        /* SQL-2003-R */
%token  EXIT_SYM
%token<string> EXPANSION_SYM
%token<string> EXPIRE_SYM
%token<string> EXPORT_SYM
%token<string> EXTENDED_SYM
%token<string> EXTENT_SIZE_SYM
%token  EXTRACT_SYM                   /* SQL-2003-N */
%token  FALSE_SYM                     /* SQL-2003-R */
%token<string> FAST_SYM
%token<string> FAULTS_SYM
%token  FETCH_SYM                     /* SQL-2003-R */
%token<string> FILE_SYM
%token<string> FILE_BLOCK_SIZE_SYM
%token<string> FILTER_SYM
%token<string> FIRST_SYM             /* SQL-2003-N */
%token<string> FIXED_SYM
%token  FLOAT_NUM
%token  FLOAT_SYM                     /* SQL-2003-R */
%token<string> FLUSH_SYM
%token<string> FOLLOWS_SYM           /* MYSQL */
%token  FORCE_SYM
%token  FOREIGN                       /* SQL-2003-R */
%token  FOR_SYM                       /* SQL-2003-R */
%token<string> FORMAT_SYM
%token<string> FOUND_SYM             /* SQL-2003-R */
%token  FROM
%token<string> FULL                  /* SQL-2003-R */
%token  FULLTEXT_SYM
%token  FUNCTION_SYM                  /* SQL-2003-R */
%token  GE
%token<string> GENERAL
%token  GENERATED
%token<string> GROUP_REPLICATION
%token<string> GEOMETRYCOLLECTION_SYM /* MYSQL */
%token<string> GEOMETRY_SYM
%token<string> GET_FORMAT            /* MYSQL-FUNC */
%token  GET_SYM                       /* SQL-2003-R */
%token<string> GLOBAL_SYM            /* SQL-2003-R */
%token  GRANT                         /* SQL-2003-R */
%token<string> GRANTS
%token  GROUP_SYM                     /* SQL-2003-R */
%token  GROUP_CONCAT_SYM
%token  GT_SYM                        /* OPERATOR */
%token<string> HANDLER_SYM
%token<string> HASH_SYM
%token  HAVING                        /* SQL-2003-R */
%token<string> HELP_SYM
%token  HEX_NUM
%token  HIGH_PRIORITY
%token<string> HOST_SYM
%token<string> HOSTS_SYM
%token  HOUR_MICROSECOND_SYM
%token  HOUR_MINUTE_SYM
%token  HOUR_SECOND_SYM
%token<string> HOUR_SYM              /* SQL-2003-R */
%token<string>  IDENT
%token<string> IDENTIFIED_SYM
%token  IDENT_QUOTED
%token  IF
%token  IGNORE_SYM
%token<string> IGNORE_SERVER_IDS_SYM
%token<string> IMPORT
%token<string> INDEXES
%token  INDEX_SYM
%token  INFILE
%token<string> INITIAL_SIZE_SYM
%token  INNER_SYM                     /* SQL-2003-R */
%token  INOUT_SYM                     /* SQL-2003-R */
%token  INSENSITIVE_SYM               /* SQL-2003-R */
%token  INSERT_SYM                    /* SQL-2003-R */
%token<string> INSERT_METHOD
%token<string> INSTANCE_SYM
%token<string> INSTALL_SYM
%token<string> INTERVAL_SYM                  /* SQL-2003-R */
%token  INTO                          /* SQL-2003-R */
%token  INT_SYM                       /* SQL-2003-R */
%token  INTEGER_SYM                       /* SQL-2003-R */
%token<string> INVOKER_SYM
%token  IN_SYM                        /* SQL-2003-R */
%token  IO_AFTER_GTIDS                /* MYSQL, FUTURE-USE */
%token  IO_BEFORE_GTIDS               /* MYSQL, FUTURE-USE */
%token<string> IO_SYM
%token<string> IPC_SYM
%token  IS                            /* SQL-2003-R */
%token<string> ISOLATION             /* SQL-2003-R */
%token<string> ISSUER_SYM
%token  ITERATE_SYM
%token  JOIN_SYM                      /* SQL-2003-R */
%token  JSON_SEPARATOR_SYM            /* MYSQL */
%token<string> JSON_SYM              /* MYSQL */
%token  KEYS
%token<string> KEY_BLOCK_SIZE
%token  KEY_SYM                       /* SQL-2003-N */
%token  KILL_SYM
%token<string> LANGUAGE_SYM          /* SQL-2003-R */
%token<string> LAST_SYM              /* SQL-2003-N */
%token  LE                            /* OPERATOR */
%token  LEADING                       /* SQL-2003-R */
%token<string> LEAVES
%token  LEAVE_SYM
%token  LEFT                          /* SQL-2003-R */
%token<string> LESS_SYM
%token<string> LEVEL_SYM
%token  LEX_HOSTNAME
%token  LIKE                          /* SQL-2003-R */
%token  LIMIT
%token  LINEAR_SYM
%token  LINES
%token<string> LINESTRING_SYM        /* MYSQL */
%token<string> LIST_SYM
%token  LOAD
%token<string> LOCAL_SYM             /* SQL-2003-R */
%token  OBSOLETE_TOKEN_538            /* was: LOCATOR_SYM */
%token<string> LOCKS_SYM
%token  LOCK_SYM
%token<string> LOGFILE_SYM
%token<string> LOGS_SYM
%token  LONGBLOB_SYM                  /* MYSQL */
%token  LONGTEXT_SYM                  /* MYSQL */
%token  LONG_NUM
%token  LONG_SYM
%token  LOOP_SYM
%token  LOW_PRIORITY
%token  LT                            /* OPERATOR */
%token<string> MASTER_AUTO_POSITION_SYM
%token  MASTER_BIND_SYM
%token<string> MASTER_CONNECT_RETRY_SYM
%token<string> MASTER_DELAY_SYM
%token<string> MASTER_HOST_SYM
%token<string> MASTER_LOG_FILE_SYM
%token<string> MASTER_LOG_POS_SYM
%token<string> MASTER_PASSWORD_SYM
%token<string> MASTER_PORT_SYM
%token<string> MASTER_RETRY_COUNT_SYM
%token<string> MASTER_SERVER_ID_SYM
%token<string> MASTER_SSL_CAPATH_SYM
%token<string> MASTER_TLS_VERSION_SYM
%token<string> MASTER_SSL_CA_SYM
%token<string> MASTER_SSL_CERT_SYM
%token<string> MASTER_SSL_CIPHER_SYM
%token<string> MASTER_SSL_CRL_SYM
%token<string> MASTER_SSL_CRLPATH_SYM
%token<string> MASTER_SSL_KEY_SYM
%token<string> MASTER_SSL_SYM
%token  MASTER_SSL_VERIFY_SERVER_CERT_SYM
%token<string> MASTER_SYM
%token<string> MASTER_USER_SYM
%token<string> MASTER_HEARTBEAT_PERIOD_SYM
%token  MATCH                         /* SQL-2003-R */
%token<string> MAX_CONNECTIONS_PER_HOUR
%token<string> MAX_QUERIES_PER_HOUR
%token<string> MAX_ROWS
%token<string> MAX_SIZE_SYM
%token  MAX_SYM                       /* SQL-2003-N */
%token<string> MAX_UPDATES_PER_HOUR
%token<string> MAX_USER_CONNECTIONS_SYM
%token  MAX_VALUE_SYM                 /* SQL-2003-N */
%token  MEDIUMBLOB_SYM                /* MYSQL */
%token  MEDIUMINT_SYM                 /* MYSQL */
%token  MEDIUMTEXT_SYM                /* MYSQL */
%token<string> MEDIUM_SYM
%token<string> MEMORY_SYM
%token<string> MERGE_SYM             /* SQL-2003-R */
%token<string> MESSAGE_TEXT_SYM      /* SQL-2003-N */
%token<string> MICROSECOND_SYM       /* MYSQL-FUNC */
%token<string> MIGRATE_SYM
%token  MINUTE_MICROSECOND_SYM
%token  MINUTE_SECOND_SYM
%token<string> MINUTE_SYM            /* SQL-2003-R */
%token<string> MIN_ROWS
%token  MIN_SYM                       /* SQL-2003-N */
%token<string> MODE_SYM
%token  MODIFIES_SYM                  /* SQL-2003-R */
%token<string> MODIFY_SYM
%token  MOD_SYM                       /* SQL-2003-N */
%token<string> MONTH_SYM             /* SQL-2003-R */
%token<string> MULTILINESTRING_SYM   /* MYSQL */
%token<string> MULTIPOINT_SYM        /* MYSQL */
%token<string> MULTIPOLYGON_SYM      /* MYSQL */
%token<string> MUTEX_SYM
%token<string> MYSQL_ERRNO_SYM
%token<string> NAMES_SYM             /* SQL-2003-N */
%token<string> NAME_SYM              /* SQL-2003-N */
%token<string> NATIONAL_SYM          /* SQL-2003-R */
%token  NATURAL                       /* SQL-2003-R */
%token  NCHAR_STRING
%token<string> NCHAR_SYM             /* SQL-2003-R */
%token<string> NDBCLUSTER_SYM
%token  NE                            /* OPERATOR */
%token  NEG
%token<string> NEVER_SYM
%token<string> NEW_SYM               /* SQL-2003-R */
%token<string> NEXT_SYM              /* SQL-2003-N */
%token<string> NODEGROUP_SYM
%token<string> NONE_SYM              /* SQL-2003-R */
%token  NOT2_SYM
%token  NOT_SYM                       /* SQL-2003-R */
%token  NOW_SYM
%token<string> NO_SYM                /* SQL-2003-R */
%token<string> NO_WAIT_SYM
%token  NO_WRITE_TO_BINLOG
%token  NULL_SYM                      /* SQL-2003-R */
%token  NUM
%token<string> NUMBER_SYM            /* SQL-2003-N */
%token  NUMERIC_SYM                   /* SQL-2003-R */
%token<string> NVARCHAR_SYM
%token<string> OFFSET_SYM
%token  ON_SYM                        /* SQL-2003-R */
%token<string> ONE_SYM
%token<string> ONLY_SYM              /* SQL-2003-R */
%token<string> OPEN_SYM              /* SQL-2003-R */
%token  OPTIMIZE
%token  OPTIMIZER_COSTS_SYM
%token<string> OPTIONS_SYM
%token  OPTION                        /* SQL-2003-N */
%token  OPTIONALLY
%token  OR2_SYM
%token  ORDER_SYM                     /* SQL-2003-R */
%token  OR_OR_SYM                     /* OPERATOR */
%token  OR_SYM                        /* SQL-2003-R */
%token  OUTER
%token  OUTFILE
%token  OUT_SYM                       /* SQL-2003-R */
%token<string> OWNER_SYM
%token<string> PACK_KEYS_SYM
%token<string> PAGE_SYM
%token  PARAM_MARKER
%token<string> PARSER_SYM
%token  OBSOLETE_TOKEN_654            /* was: PARSE_GCOL_EXPR_SYM */
%token<string> PARTIAL                       /* SQL-2003-N */
%token  PARTITION_SYM                 /* SQL-2003-R */
%token<string> PARTITIONS_SYM
%token<string> PARTITIONING_SYM
%token<string> PASSWORD
%token<string> PHASE_SYM
%token<string> PLUGIN_DIR_SYM        /* INTERNAL */
%token<string> PLUGIN_SYM
%token<string> PLUGINS_SYM
%token<string> POINT_SYM "."
%token<string> POLYGON_SYM           /* MYSQL */
%token<string> PORT_SYM
%token  POSITION_SYM                  /* SQL-2003-N */
%token<string> PRECEDES_SYM          /* MYSQL */
%token  PRECISION                     /* SQL-2003-R */
%token<string> PREPARE_SYM           /* SQL-2003-R */
%token<string> PRESERVE_SYM
%token<string> PREV_SYM
%token  PRIMARY_SYM                   /* SQL-2003-R */
%token<string> PRIVILEGES            /* SQL-2003-N */
%token  PROCEDURE_SYM                 /* SQL-2003-R */
%token<string> PROCESS
%token<string> PROCESSLIST_SYM
%token<string> PROFILE_SYM
%token<string> PROFILES_SYM
%token<string> PROXY_SYM
%token  PURGE
%token<string> QUARTER_SYM
%token<string> QUERY_SYM
%token<string> QUICK
%token  RANGE_SYM                     /* SQL-2003-R */
%token  READS_SYM                     /* SQL-2003-R */
%token<string> READ_ONLY_SYM
%token  READ_SYM                      /* SQL-2003-N */
%token  READ_WRITE_SYM
%token  REAL_SYM                      /* SQL-2003-R */
%token<string> REBUILD_SYM
%token<string> RECOVER_SYM
%token  OBSOLETE_TOKEN_693            /* was: REDOFILE_SYM */
%token<string> REDO_BUFFER_SIZE_SYM
%token<string> REDUNDANT_SYM
%token  REFERENCES                    /* SQL-2003-R */
%token  REGEXP
%token<string> RELAY
%token<string> RELAYLOG_SYM
%token<string> RELAY_LOG_FILE_SYM
%token<string> RELAY_LOG_POS_SYM
%token<string> RELAY_THREAD
%token  RELEASE_SYM                   /* SQL-2003-R */
%token<string> RELOAD
%token<string> REMOVE_SYM
%token  RENAME
%token<string> REORGANIZE_SYM
%token<string> REPAIR
%token<string> REPEATABLE_SYM        /* SQL-2003-N */
%token  REPEAT_SYM                    /* MYSQL-FUNC */
%token  REPLACE_SYM                   /* MYSQL-FUNC */
%token<string> REPLICATION
%token<string> REPLICATE_DO_DB
%token<string> REPLICATE_IGNORE_DB
%token<string> REPLICATE_DO_TABLE
%token<string> REPLICATE_IGNORE_TABLE
%token<string> REPLICATE_WILD_DO_TABLE
%token<string> REPLICATE_WILD_IGNORE_TABLE
%token<string> REPLICATE_REWRITE_DB
%token  REQUIRE_SYM
%token<string> RESET_SYM
%token  RESIGNAL_SYM                  /* SQL-2003-R */
%token<string> RESOURCES
%token<string> RESTORE_SYM
%token  RESTRICT
%token<string> RESUME_SYM
%token<string> RETURNED_SQLSTATE_SYM /* SQL-2003-N */
%token<string> RETURNS_SYM           /* SQL-2003-R */
%token  RETURN_SYM                    /* SQL-2003-R */
%token<string> REVERSE_SYM
%token  REVOKE                        /* SQL-2003-R */
%token  RIGHT                         /* SQL-2003-R */
%token<string> ROLLBACK_SYM          /* SQL-2003-R */
%token<string> ROLLUP_SYM            /* SQL-2003-R */
%token<string> ROTATE_SYM
%token<string> ROUTINE_SYM           /* SQL-2003-N */
%token  ROWS_SYM                      /* SQL-2003-R */
%token<string> ROW_FORMAT_SYM
%token  ROW_SYM                       /* SQL-2003-R */
%token<string> ROW_COUNT_SYM         /* SQL-2003-N */
%token<string> RTREE_SYM
%token<string> SAVEPOINT_SYM         /* SQL-2003-R */
%token<string> SCHEDULE_SYM
%token<string> SCHEMA_NAME_SYM       /* SQL-2003-N */
%token SCHEMA
%token  SECOND_MICROSECOND_SYM
%token<string> SECOND_SYM            /* SQL-2003-R */
%token<string> SECURITY_SYM          /* SQL-2003-N */
%token  SELECT_SYM                    /* SQL-2003-R */
%token  SENSITIVE_SYM                 /* FUTURE-USE */
%token  SEPARATOR_SYM
%token<string> SERIALIZABLE_SYM      /* SQL-2003-N */
%token<string> SERIAL_SYM
%token<string> SESSION_SYM           /* SQL-2003-N */
%token<string> SERVER_SYM
%token  OBSOLETE_TOKEN_755            /* was: SERVER_OPTIONS */
%token  SET                       /* SQL-2003-R */
%token  SET_VAR
%token<string> SHARE_SYM
%token<string> SHARES_SYM
%token  SHIFT_LEFT                    /* OPERATOR */
%token  SHIFT_RIGHT                   /* OPERATOR */
%token  SHOW
%token<string> SHUTDOWN
%token  SIGNAL_SYM                    /* SQL-2003-R */
%token<string> SIGNED_SYM
%token<string> SIMPLE_SYM            /* SQL-2003-N */
%token<string> SLAVE
%token<string> SLOW
%token  SMALLINT_SYM                  /* SQL-2003-R */
%token<string> SNAPSHOT_SYM
%token<string> SOCKET_SYM
%token<string> SONAME_SYM
%token<string> SOUNDS_SYM
%token<string> SOURCE_SYM
%token  SPATIAL_SYM
%token  SPECIFIC_SYM                  /* SQL-2003-R */
%token  SQLEXCEPTION_SYM              /* SQL-2003-R */
%token  SQLSTATE_SYM                  /* SQL-2003-R */
%token  SQLWARNING_SYM                /* SQL-2003-R */
%token<string> SQL_AFTER_GTIDS       /* MYSQL */
%token<string> SQL_AFTER_MTS_GAPS    /* MYSQL */
%token<string> SQL_BEFORE_GTIDS      /* MYSQL */
%token  SQL_BIG_RESULT
%token<string> SQL_BUFFER_RESULT
%token  OBSOLETE_TOKEN_784            /* was: SQL_CACHE_SYM */
%token  SQL_CALC_FOUND_ROWS
%token<string> SQL_NO_CACHE_SYM
%token  SQL_SMALL_RESULT
%token  SQL_SYM                       /* SQL-2003-R */
%token<string> SQL_THREAD
%token  SSL_SYM
%token<string> STACKED_SYM           /* SQL-2003-N */
%token  STARTING
%token<string> STARTS_SYM
%token<string> START_SYM             /* SQL-2003-R */
%token<string> STATS_AUTO_RECALC_SYM
%token<string> STATS_PERSISTENT_SYM
%token<string> STATS_SAMPLE_PAGES_SYM
%token<string> STATUS_SYM
%token  STDDEV_SAMP_SYM               /* SQL-2003-N */
%token  STD_SYM
%token<string> STOP_SYM
%token<string> STORAGE_SYM
%token  STORED_SYM
%token  STRAIGHT_JOIN
%token<string> STRING_SYM
%token<string> SUBCLASS_ORIGIN_SYM   /* SQL-2003-N */
%token<string> SUBDATE_SYM
%token<string> SUBJECT_SYM
%token<string> SUBPARTITIONS_SYM
%token<string> SUBPARTITION_SYM
%token  SUBSTRING                     /* SQL-2003-N */
%token  SUM_SYM                       /* SQL-2003-N */
%token<string> SUPER_SYM
%token<string> SUSPEND_SYM
%token<string> SWAPS_SYM
%token<string> SWITCHES_SYM
%token  SYSDATE
%token<string> TABLES
%token<string> VIEWS
%token<string> TABLESPACE_SYM
%token  OBSOLETE_TOKEN_820            /* was: TABLE_REF_PRIORITY */
%token  TABLE_SYM                     /* SQL-2003-R */
%token<string> TABLE_CHECKSUM_SYM
%token<string> TABLE_NAME_SYM        /* SQL-2003-N */
%token<string> TEMPORARY             /* SQL-2003-N */
%token<string> TEMPTABLE_SYM
%token  TERMINATED
%token  TEXT_STRING
%token<string> TEXT_SYM
%token<string> THAN_SYM
%token  THEN_SYM                      /* SQL-2003-R */
%token<string> TIMESTAMP_SYM         /* SQL-2003-R */
%token<string> TIMESTAMP_ADD
%token<string> TIMESTAMP_DIFF
%token<string> TIME_SYM              /* SQL-2003-R */
%token  TINYBLOB_SYM                  /* MYSQL */
%token  TINYINT_SYM                   /* MYSQL */
%token  TINYTEXT_SYN                  /* MYSQL */
%token  TO_SYM                        /* SQL-2003-R */
%token  TRAILING                      /* SQL-2003-R */
%token<string> TRANSACTION_SYM
%token<string> TRIGGERS_SYM
%token  TRIGGER_SYM                   /* SQL-2003-R */
%token  TRIM                          /* SQL-2003-N */
%token  TRUE_SYM                      /* SQL-2003-R */
%token<string> TRUNCATE_SYM
%token<string> TYPES_SYM
%token<string> TYPE_SYM              /* SQL-2003-N */
%token  OBSOLETE_TOKEN_848            /* was:  UDF_RETURNS_SYM */
%token  ULONGLONG_NUM
%token<string> UNCOMMITTED_SYM       /* SQL-2003-N */
%token<string> UNDEFINED_SYM
%token  UNDERSCORE_CHARSET
%token<string> UNDOFILE_SYM
%token<string> UNDO_BUFFER_SIZE_SYM
%token  UNDO_SYM                      /* FUTURE-USE */
%token<string> UNICODE_SYM
%token<string> UNINSTALL_SYM
%token  UNION_SYM                     /* SQL-2003-R */
%token  UNIQUE_SYM
%token<string> UNKNOWN_SYM           /* SQL-2003-R */
%token  UNLOCK_SYM
%token  UNSIGNED_SYM                  /* MYSQL */
%token<string> UNTIL_SYM
%token  UPDATE_SYM                    /* SQL-2003-R */
%token<string> UPGRADE_SYM
%token  USAGE                         /* SQL-2003-N */
%token<string> USER                  /* SQL-2003-R */
%token<string> USE_FRM
%token  USE_SYM
%token  USING                         /* SQL-2003-R */
%token  UTC_DATE_SYM
%token  UTC_TIMESTAMP_SYM
%token  UTC_TIME_SYM
%token<string> VALIDATION_SYM        /* MYSQL */
%token  VALUES                        /* SQL-2003-R */
%token<string> VALUE_SYM             /* SQL-2003-R */
%token<string> VARBINARY_SYM                 /* SQL-2008-R */
%token  VARCHAR_SYM                   /* SQL-2003-R */
%token<string> VARIABLES
%token  VARIANCE_SYM
%token  VARYING                       /* SQL-2003-R */
%token  VAR_SAMP_SYM
%token  VERSION_SYM
%token<string> VIEW_SYM              /* SQL-2003-N */
%token  VIRTUAL_SYM
%token<string> WAIT_SYM
%token<string> WARNINGS
%token<string> WEEK_SYM
%token<string> WEIGHT_STRING_SYM
%token  WHEN_SYM                      /* SQL-2003-R */
%token  WHERE                         /* SQL-2003-R */
%token  WHILE_SYM
%token  WITH                          /* SQL-2003-R */
%token  OBSOLETE_TOKEN_893            /* was: WITH_CUBE_SYM */
%token  WITH_ROLLUP_SYM               /* INTERNAL */
%token<string> WITHOUT_SYM           /* SQL-2003-R */
%token<string> WORK_SYM              /* SQL-2003-N */
%token<string> WRAPPER_SYM
%token  WRITE_SYM                     /* SQL-2003-N */
%token<string> X509_SYM
%token<string> XA_SYM
%token<string> XID_SYM               /* MYSQL */
%token<string> XML_SYM
%token  XOR
%token  YEAR_MONTH_SYM
%token<string> YEAR_SYM              /* SQL-2003-R */
%token  ZEROFILL_SYM                  /* MYSQL */
%token EXPLAIN_SYM
%token TREE_SYM
%token TRADITIONAL_SYM

/*
   Tokens from MySQL 8.0
*/
%token  JSON_UNQUOTED_SEPARATOR_SYM   /* MYSQL */
%token<string> PERSIST_SYM           /* MYSQL */
%token<string> ROLE_SYM              /* SQL-1999-R */
%token<string> ADMIN_SYM             /* SQL-2003-N */
%token<string> INVISIBLE_SYM
%token<string> VISIBLE_SYM
%token  EXCEPT_SYM                    /* SQL-1999-R */
%token<string> COMPONENT_SYM         /* MYSQL */
%token  RECURSIVE_SYM                 /* SQL-1999-R */
%token  GRAMMAR_SELECTOR_EXPR         /* synthetic token: starts single expr. */
%token  GRAMMAR_SELECTOR_GCOL       /* synthetic token: starts generated col. */
%token  GRAMMAR_SELECTOR_PART      /* synthetic token: starts partition expr. */
%token  GRAMMAR_SELECTOR_CTE             /* synthetic token: starts CTE expr. */
%token  JSON_OBJECTAGG                /* SQL-2015-R */
%token  JSON_ARRAYAGG                 /* SQL-2015-R */
%token  OF_SYM                        /* SQL-1999-R */
%token<string> SKIP_SYM              /* MYSQL */
%token<string> LOCKED_SYM            /* MYSQL */
%token<string> NOWAIT_SYM            /* MYSQL */
%token  GROUPING_SYM                  /* SQL-2011-R */
%token<string> PERSIST_ONLY_SYM      /* MYSQL */
%token<string> HISTOGRAM_SYM         /* MYSQL */
%token<string> BUCKETS_SYM           /* MYSQL */
%token<string> OBSOLETE_TOKEN_930    /* was: REMOTE_SYM */
%token<string> CLONE_SYM             /* MYSQL */
%token  CUME_DIST_SYM                 /* SQL-2003-R */
%token  DENSE_RANK_SYM                /* SQL-2003-R */
%token<string> EXCLUDE_SYM           /* SQL-2003-N */
%token  FIRST_VALUE_SYM               /* SQL-2011-R */
%token<string> FOLLOWING_SYM         /* SQL-2003-N */
%token  GROUPS_SYM                    /* SQL-2011-R */
%token  LAG_SYM                       /* SQL-2011-R */
%token  LAST_VALUE_SYM                /* SQL-2011-R */
%token  LEAD_SYM                      /* SQL-2011-R */
%token  NTH_VALUE_SYM                 /* SQL-2011-R */
%token  NTILE_SYM                     /* SQL-2011-R */
%token<string> NULLS_SYM             /* SQL-2003-N */
%token<string> OTHERS_SYM            /* SQL-2003-N */
%token  OVER_SYM                      /* SQL-2003-R */
%token  PERCENT_RANK_SYM              /* SQL-2003-R */
%token<string> PRECEDING_SYM         /* SQL-2003-N */
%token  RANK_SYM                      /* SQL-2003-R */
%token<string> RESPECT_SYM           /* SQL_2011-N */
%token  ROW_NUMBER_SYM                /* SQL-2003-R */
%token<string> TIES_SYM              /* SQL-2003-N */
%token<string> UNBOUNDED_SYM         /* SQL-2003-N */
%token  WINDOW_SYM                    /* SQL-2003-R */
%token  EMPTY_SYM                     /* SQL-2016-R */
%token  JSON_TABLE_SYM                /* SQL-2016-R */
%token<string> NESTED_SYM            /* SQL-2016-N */
%token<string> ORDINALITY_SYM        /* SQL-2003-N */
%token<string> PATH_SYM              /* SQL-2003-N */
%token<string> HISTORY_SYM           /* MYSQL */
%token<string> REUSE_SYM             /* MYSQL */
%token<string> SRID_SYM              /* MYSQL */
%token<string> THREAD_PRIORITY_SYM   /* MYSQL */
%token<string> RESOURCE_SYM          /* MYSQL */
%token  SYSTEM_SYM                    /* SQL-2003-R */
%token<string> VCPU_SYM              /* MYSQL */
%token<string> MASTER_PUBLIC_KEY_PATH_SYM    /* MYSQL */
%token<string> GET_MASTER_PUBLIC_KEY_SYM     /* MYSQL */
%token<string> RESTART_SYM                   /* SQL-2003-N */
%token<string> DEFINITION_SYM                /* MYSQL */
%token<string> DESCRIPTION_SYM               /* MYSQL */
%token<string> ORGANIZATION_SYM              /* MYSQL */
%token<string> REFERENCE_SYM                 /* MYSQL */
%token<string> ACTIVE_SYM                    /* MYSQL */
%token<string> INACTIVE_SYM                  /* MYSQL */
%token          LATERAL_SYM                   /* SQL-1999-R */
%token<string> OPTIONAL_SYM                  /* MYSQL */
%token<string> SECONDARY_SYM                 /* MYSQL */
%token<string> SECONDARY_ENGINE_SYM          /* MYSQL */
%token<string> SECONDARY_LOAD_SYM            /* MYSQL */
%token<string> SECONDARY_UNLOAD_SYM          /* MYSQL */
%token<string> RETAIN_SYM                    /* MYSQL */
%token<string> OLD_SYM                       /* SQL-2003-R */
%token<string> ENFORCED_SYM                  /* SQL-2015-N */
%token<string> OJ_SYM                        /* ODBC */
%token<string> NETWORK_NAMESPACE_SYM         /* MYSQL */

%token  ADD_SYM '+'
%token  MINUS_SYM '-'

%type <ulong> ulong_num opt_num_parts opt_num_subparts
%type <ulonglong>
        real_ulong_num ulonglong_num real_ulonglong_num size_number

%type <string>  select_alias text_literal limit_option
                      NUM LONG_NUM ULONGLONG_NUM DECIMAL_NUM FLOAT_NUM HEX_NUM
                      ident_or_text opt_component
                      LEX_HOSTNAME
                      TEXT_STRING TEXT_STRING_sys TEXT_STRING_literal TEXT_STRING_filesystem
                      TEXT_STRING_password TEXT_STRING_hash password
                      TEXT_STRING_sys_nonewline
                      opt_db execute_var_ident opt_field_length field_length
                      type_datetime_precision
                      text_string
                      key_part key_part_with_expression opt_constraint_name

%type <CastType> cast_type

%type <SQLIdentPtr> simple_ident simple_ident_q simple_ident_nospvar insert_ident 
%type <vector<Expression>> fields
%type <string> ident opt_describe_column opt_ident IDENT_sys opt_table_alias
                    interval interval_time_stamp
                    opt_ordering_direction ordering_direction
                    lvalue_keyword lvalue_ident internal_variable_name ident_keyword ident_keywords_unambiguous ident_keywords_ambiguous_2_labels
                    opt_index_name_and_type

%type <Field_option> field_option field_opt_list field_options
%type <string> nchar varchar nvarchar int_type real_type numeric_type opt_encode_index_type
%type <Precision_ptr> opt_precision precision float_options
%type <PT_ColumnType_ptr> type
%type <PT_column_attr_base_ptr> column_attribute
%type <ColAttrList> column_attribute_list opt_column_attribute_list
%type <Field_def_ptr> field_def
%type <keytype> constraint_key_type

%type <int> comp_op kill_option

//%type <numeric> NUM LONG_NUM ULONGLONG_NUM DECIMAL_NUM FLOAT_NUM

%type <Literal> literal NUM_literal signed_literal

%type <Expression>  bool_pri expr expr_or_default predicate case_expr simple_expr bit_expr opt_where_clause opt_where_clause_expr
                    function_call_keyword function_call_nonkeyword function_call_generic function_call_conflict set_function_specification now
                    in_sum_expr sum_expr
                    grouping_expr opt_having_clause
                    row_subquery
                    opt_else opt_expr func_datetime_precision
                    udf_expr
                    variable table_wild param_marker set_expr_or_default
                    now_or_signed_literal
                    ident_expr
                    expr_tmp

%type <enum_var_type> option_type opt_var_ident_type opt_var_type opt_set_var_ident_type
%type <VariableStructurePtr> variable_aux

%type <tuple<Expression, string>> select_item

%type <SelectStructurePointer> select_stmt query_expression query_expression_body query_primary query_specification table_subquery subquery query_expression_parens query_expression_or_parens view_select explain_stmt
%type <SelectPartStructurePointer> select_item_list
%type <UpdateStructurePtr> update_stmt
%type <InsertStructurePtr> insert_stmt
%type <DeleteStructurePtr> delete_stmt
%type <pair< BiaodashiPointer, BiaodashiPointer >> update_elem
%type <pair< vector< BiaodashiPointer >, vector< BiaodashiPointer > > > update_list opt_insert_update_list
%type <pair< EXPR_LIST, SelectStructurePointer >> insert_query_expression

%type <VALUES> opt_values values row_value 
%type <vector<Expression>> insert_from_constructor

%type <FromPartStructurePointer> opt_from_clause from_clause
%type <vector<JoinStructurePointer>> from_tables table_reference_list
%type <JoinStructurePointer> table_reference table_factor single_table single_table_parens joined_table derived_table joined_table_parens
%type <JoinType> inner_join_type outer_join_type natural_join_type
%type <SetOperationType> union_option
%type <shared_ptr<BasicRel>> table_ident sp_name table_ident_opt_wild
%type <TABLE_LIST> table_list opt_table_list table_alias_ref_list

%type <GroupbyStructurePointer> opt_group_clause

%type <OrderbyStructurePointer> opt_order_clause order_clause

%type <LimitStructurePointer> limit_clause opt_limit_clause limit_options opt_simple_limit

%type <vector<Expression>> group_list expr_list opt_expr_list
                  opt_udf_expr_list udf_expr_list
                  reference_list

%type <pair< shared_ptr<BasicRel>, vector< string > >> references opt_references

%type <OrderItem>  order_expr
%type <vector<OrderItem>> order_list

%type <vector<tuple<Expression, Expression>>> when_list

%type <bool> opt_full opt_extended opt_if_not_exists opt_temporary
    opt_linear visibility opt_not if_exists opt_ignore opt_distinct
%type <SHOW_CMD> show_engine_param
%type <Show_cmd_type> opt_show_cmd_type
%type <WildOrWhere_ptr> opt_wild_or_where_for_show opt_wild_or_where
%type <ShowStructurePtr> show show_param describe_stmt
%type <AdminStmtStructurePtr> kill shutdown_stmt

%type <AbstractCommandPointer> create
                               create_table_stmt
                               drop_database_stmt
                               drop_table_stmt
                               use
                               view_or_trigger_or_sp_or_event
                               no_definer_tail
                               view_tail
                               drop_view_stmt
                               drop_user_stmt
%type <AccountSPtr> user create_user
%type <shared_ptr< vector< AccountSPtr > >> create_user_list user_list

%type <TableElementDescriptionPtr> column_def table_constraint_def table_element
%type <shared_ptr<vector<TableElementDescriptionPtr>>> table_element_list

%type <PreparedStmtStructurePtr> prepare deallocate execute
%type <PrepareSrcPtr> prepare_src
%type <vector<string>> execute_using execute_var_list simple_ident_list opt_derived_column_list
                    key_list_with_expression key_list

%type <PT_option_value_following_option_type_ptr> option_value_following_option_type
%type <shared_ptr<vector<SetStructurePtr>>> set start_option_value_list option_value_list option_value_list_continued start_option_value_list_following_option_type
%type <SetStructurePtr> option_value option_value_no_option_type

%type <bool> opt_local
%type <enum_filetype> data_or_xml
%type <thr_lock_type> load_data_lock
%type <On_duplicate> duplicate opt_duplicate
%type <string> opt_load_data_charset charset_name opt_xml_rows_identified_by
%type <Line_separators> line_term line_term_list opt_line_term
%type <Field_separators> field_term field_term_list opt_field_term
%type <ulong> opt_ignore_lines
%type <LoadDataStructurePtr> load_stmt

%type <int> query_spec_option select_option select_options select_option_list
%type <TransactionStructurePtr> start commit rollback begin_stmt
%type <enum_yes_no_unknown> opt_chain opt_release
%type <pair< aries::EncodeType, string >> encode_type

// table partition
%type <PartTypeDef> part_type_def 
%type <PartValueItem> part_value_item
%type <PartValueItemsSPtr> part_func_max part_value_item_list_paren part_value_item_list
%type <PartValuesSPtr> opt_part_values;
%type <vector< PartValueItemsSPtr >> part_value_list part_values_in
%type <PartDef> part_definition
%type <PartDefList> opt_part_defs part_def_list
%type <PartitionStructureSPtr> opt_create_partitioning_etc
                               partition_clause

%type <CreateTableOptions> opt_create_table_options_etc

/*
  Resolve column attribute ambiguity -- force precedence of "UNIQUE KEY" against
  simple "UNIQUE" and "KEY" attributes:
*/
%right UNIQUE_SYM KEY_SYM

%left CONDITIONLESS_JOIN
%left   JOIN_SYM INNER_SYM CROSS STRAIGHT_JOIN NATURAL LEFT RIGHT ON_SYM USING
%left   SET_VAR
%left   OR_OR_SYM OR_SYM OR2_SYM
%left   XOR
%left   AND_SYM AND_AND_SYM
%left   BETWEEN_SYM CASE_SYM WHEN_SYM THEN_SYM ELSE
%left   EQ EQUAL_SYM GE GT_SYM LE LT NE IS LIKE REGEXP IN_SYM
%left   '|'
%left   '&'
%left   SHIFT_LEFT SHIFT_RIGHT
%left   '-' '+'
%left   '*' '/' '%' DIV_SYM MOD_SYM
%left   '^'
%left   NEG '~'
%right  NOT_SYM NOT2_SYM
%right  BINARY_SYM COLLATE_SYM
%left SUBQUERY_AS_EXPR
%left  INTERVAL_SYM
%left '(' ')'

%left EMPTY_FROM_CLAUSE
%right INTO

%start start_entry


//%lex-param { double a }

%%

start_entry: sql_statements
    ;

sql_statements:
      END_OF_INPUT
    | sql_statement sql_statements
    ;

sql_statement:
      /*empty*/ ';'
    | simple_statement END_OF_INPUT
    | simple_statement ';'
    ;

simple_statement:
        create {
            auto stmtStr = driver.get_string_at_location(@1);
            $1->SetCommandString(stmtStr);
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | create_table_stmt {
            auto stmtStr = driver.get_string_at_location(@1);
            $1->SetCommandString(stmtStr);
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | deallocate {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetPreparedStmtStructurePtr($1);
            driver.statements.emplace_back(statement);
        }
        | delete_stmt {
          auto statement = std::make_shared<AriesSQLStatement>();
          statement->SetDeleteStructure($1);
          driver.statements.emplace_back(statement);
        }
        | describe_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetShowStructure($1);
            driver.statements.emplace_back(statement);
        }
        | drop_database_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | drop_table_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | drop_user_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | drop_view_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | execute {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetPreparedStmtStructurePtr($1);
            driver.statements.emplace_back(statement);
        }
        | explain_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetExplainQuery($1);
            driver.statements.emplace_back(statement);
        }
        | insert_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetInsertStructure($1);
            driver.statements.emplace_back(statement);
        }
        | load_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetLoadDataStructure($1);
            driver.statements.emplace_back(statement);
        }
        | lock {
            ThrowFakeImplException( ER_FAKE_IMPL_OK, "lock tables");
        }
        | unlock {
            ThrowFakeImplException( ER_FAKE_IMPL_OK, "unlock tables");
        }
        | kill {
          auto statement = std::make_shared<AriesSQLStatement>();
          statement->SetAdminStmt($1);
          driver.statements.emplace_back(statement);
        }
        | prepare {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetPreparedStmtStructurePtr($1);
            driver.statements.emplace_back(statement);
        }
        | select_stmt  {
          auto statement = std::make_shared<AriesSQLStatement>();
          statement->SetQuery($1);
          driver.statements.emplace_back(statement);
        }
        | show {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetShowStructure($1);
            driver.statements.emplace_back(statement);
        }
        | shutdown_stmt {
          auto statement = std::make_shared<AriesSQLStatement>();
          statement->SetAdminStmt($1);
          driver.statements.emplace_back(statement);
        }
        | set {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetSetStructures($1);
            driver.statements.emplace_back(statement);
        }
        | start {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetTxStructure($1);
            driver.statements.emplace_back(statement);
        }
        | begin_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetTxStructure($1);
            driver.statements.emplace_back(statement);
        }
        | commit {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetTxStructure($1);
            driver.statements.emplace_back(statement);
        }
        | rollback {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetTxStructure($1);
            driver.statements.emplace_back(statement);
        }
        | use {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetCommand($1);
            driver.statements.emplace_back(statement);
        }
        | update_stmt {
            auto statement = std::make_shared<AriesSQLStatement>();
            statement->SetUpdateStructure($1);
            driver.statements.emplace_back(statement);
        }
    ;

/*
  Drop : delete tables or index or user or role
*/
table_or_tables:
          TABLE_SYM
        | TABLES
        ;
view_or_views:
          VIEW_SYM
        | TABLES

opt_restrict:
          /* empty */ { }
        | RESTRICT    { ThrowNotSupportedException("RESTRICT"); }
        | CASCADE     { ThrowNotSupportedException("CASCADE"); }
        ;
drop_table_stmt:
          DROP opt_temporary table_or_tables if_exists table_list opt_restrict
          {
            // Note: opt_restrict ($6) is ignored!
            $$ = CreateDropTablesStructure($2, $4, $5);
          }
        ;

drop_user_stmt:
          DROP USER if_exists user_list
          {
             $$ = CreateAccountMgmtStructure( CommandType::DropUser, $3, $4 );
          }
        ;

drop_view_stmt:
          DROP opt_temporary view_or_views if_exists table_list {
            $$ = CreateDropViewsStructure($2, $4, $5);
          }

drop_database_stmt:
          DROP DATABASE if_exists ident
          {
            $$ = CreateDropDatabaseStructure($3, $4);
          }
        ;
select_stmt:
        query_expression { $$ = $1; }
    ;

query_expression:
        query_expression_body
        opt_order_clause
        opt_limit_clause {
          auto select_structure = $1;
          select_structure->SetLimitStructure($3);
          if ($2) {
            select_structure->SetOrderbyPart($2);
          }
          $$ = select_structure;
        }
    |
        with_clause
        query_expression_body
        opt_order_clause
        opt_limit_clause {
          auto select_structure = $2;
          $$ = select_structure;
        }
    ;

query_expression_body:
        query_primary { $$ = $1; }
    |   query_expression_body UNION_SYM union_option query_primary {
      auto select_structure = std::make_shared<SelectStructure>();
      select_structure->init_set_query($3, $1, $4);
      $$ = select_structure;
    }
    ;

query_expression_parens:
          '(' query_expression_parens ')' { $$ = $2; }
        | '(' query_expression ')' { $$ = $2; }
        ;

query_primary:
        query_specification { $$ = $1; }
    ;

query_specification:
        SELECT_SYM
        select_options
        select_item_list
        opt_from_clause
        opt_where_clause
        opt_group_clause
        opt_having_clause
        opt_window_clause {
          auto select_structure = std::make_shared<SelectStructure>();
          auto group_by_expr = $6;
          if ($7) {
            if (!group_by_expr) {
              group_by_expr = std::make_shared<GroupbyStructure>();
            }
            group_by_expr->SetHavingExpr($7);
          }
          select_structure->init_simple_query($3, $4, $5, group_by_expr, nullptr);

          if ($2 & SelectOption::DISTINCT) {
            select_structure->SetDistinct(true);
          }
          $$ = select_structure;
        }
    ;

opt_from_clause:
        /* Empty. */ %prec EMPTY_FROM_CLAUSE  { $$ = nullptr; }
    |   from_clause { $$ = $1; }
    ;

from_clause:
        FROM from_tables {
          if (!$2.empty()) {
            auto from_part = std::make_shared<FromPartStructure>();
            for (const auto& join_part : $2) {
              from_part->AddFromItem(join_part);
            }
            $$ = from_part;
          } else {
            $$ = nullptr;
          }
        }
    ;
from_tables:
        DUAL_SYM { ThrowNotSupportedException("DUAL");}
    |   table_reference_list { $$ = $1; }
    ;

table_reference_list:
        table_reference {
          std::vector<JoinStructurePointer> list;
          list.emplace_back($1);
          $$ = list;
        }
    |   table_reference_list ',' table_reference {
      $1.emplace_back($3);
      $$ = $1;
    }
    ;

select_options:
        /* empty*/ { $$ = SelectOption::UNSET; }
    |   select_option_list  { $$ = $1; }
    ;

select_option_list:
        select_option_list select_option { $$ = $1 | $2;}
    |   select_option { $$ = $1; }
    ;

select_option:
        query_spec_option { $$ = $1; }
    |   SQL_NO_CACHE_SYM { $$ = SelectOption::SQL_NO_CACHE_SYM; }
    ;

select_item_list:
        select_item_list
        ',' 
        select_item {
          auto the_sps = $1;
          ADD_SELECT_ITEM(the_sps, ($3));
          $$ = $1;
        }
    |   select_item {
          auto the_sps = std::make_shared<SelectPartStructure>();
          ADD_SELECT_ITEM(the_sps, ($1));
          $$ = the_sps;
        }
    |   '*' {
      auto the_sps = std::make_shared<SelectPartStructure>();
      ADD_SELECT_ITEM(the_sps, std::make_tuple(std::make_shared<CommonBiaodashi>(BiaodashiType::Star, string()), string()));
      $$ = the_sps;
    }
    ;

select_item:
        table_wild  { $$ = std::make_tuple($1, string()); }
    |   expr select_alias {
        SetExprOrigName($1, driver.get_string_at_location(@1), $2);

        $$ = std::make_tuple($1, NormalizeIdent($2));
    }
    ;


select_alias:
          /* empty */ { $$ = string(); }
        | opt_as ident { $$ = $2; }
        | opt_as TEXT_STRING_sys { $$ = $2; }

        ;

optional_braces:
        /* empty */ {}
    |   '(' ')' {}
    ;

/* all possible expressions */
// TODO: support ' IS [NOT] {TRUE | FALSE | UNKNOWN} '
expr:
        expr or expr %prec OR_SYM {
      $$ = CreateLogicExpression(LogicType::OR, $1, $3);
    }
    // |   expr XOR expr %prec XOR
    |   expr and expr %prec AND_SYM {
      $$ = CreateLogicExpression(LogicType::AND, $1, $3);
    }
    |   NOT_SYM expr %prec NOT_SYM {
      $$ = GenerateNotExpression($2);
    }
    |   bool_pri IS TRUE_SYM %prec IS
    {
        ThrowNotSupportedException("IS TRUE");
    }
    |   bool_pri IS not TRUE_SYM %prec IS
    {
        ThrowNotSupportedException("IS NOT TRUE");
    }
    |   bool_pri IS FALSE_SYM %prec IS
    {
        ThrowNotSupportedException("IS FALSE");
    }
    |   bool_pri IS not FALSE_SYM %prec IS
    {
        ThrowNotSupportedException("IS NOT FALSE");
    }
    |   bool_pri IS UNKNOWN_SYM %prec IS
    {
        ThrowNotSupportedException("IS UNKNOWN");
    }
    |   bool_pri IS not UNKNOWN_SYM %prec IS
    {
        ThrowNotSupportedException("IS NOT UNKNOWN");
    }
    |   bool_pri {
      if ($1 == nullptr)
      {
        ThrowNotSupportedException("expression: " + driver.get_string_at_location(@1));
      }
       $$ = $1; 
    }
    ;

bool_pri:
        bool_pri IS NULL_SYM %prec IS {
          auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IsNull, 0);

          expression->AddChild($1);

          $$ = expression;
        }
    |   bool_pri IS not NULL_SYM %prec IS {

      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IsNotNull, 0);

      expression->AddChild($1);

      $$ = expression;
    }
    |   bool_pri comp_op predicate {
      $$ = CreateComparationExpression($1, $2, $3);
    }
    |   bool_pri comp_op all_or_any table_subquery %prec EQ
    |   predicate {
      $$ = $1;
    }
    ;

predicate:
        bit_expr IN_SYM table_subquery {
          CheckInExprFirstArg( $1 );
          auto in_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Inop, 0);
          in_expression->AddChild($1);

          auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Query, $3);
          in_expression->AddChild(expr);

          $$ = in_expression;
        }
    |   bit_expr not IN_SYM table_subquery {
          CheckInExprFirstArg( $1 );
          auto notin_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::NotIn, 0);
          notin_expression->AddChild($1);

          auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Query, $4);
          notin_expression->AddChild(expr);

          $$ = notin_expression;
        }
    |   bit_expr IN_SYM '(' expr ')' {
          CheckInExprFirstArg( $1 );
          auto in_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Inop, 0);
          in_expression->AddChild($1);
          in_expression->AddChild($4);
          $$ = in_expression;
      }
    |   bit_expr IN_SYM '(' expr_list_in ')' {
          CheckInExprFirstArg( $1 );
          auto in_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Inop, 0);
          in_expression->AddChild($1);

          for (auto const& expr : driver.global_expr_list) {
            in_expression->AddChild(expr);
          }
          driver.global_expr_list.clear();

          $$ = in_expression;
    }
    |   bit_expr not IN_SYM '(' expr ')' {
          CheckInExprFirstArg( $1 );
          auto in_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::NotIn, 0);
          in_expression->AddChild($1);
          in_expression->AddChild($5);
          $$ = in_expression;
      }
    |   bit_expr not IN_SYM '(' expr_list_in ')' {
          CheckInExprFirstArg( $1 );
          auto in_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::NotIn, 0);
          in_expression->AddChild($1);

          for (auto const& expr : driver.global_expr_list) {
            in_expression->AddChild(expr);
          }
          driver.global_expr_list.clear();

          $$ = in_expression;
    }
    |   bit_expr BETWEEN_SYM bit_expr AND_SYM predicate {
      $$ = CreateBetweenExpr( $1, $3, $5 );
    }
    |   bit_expr not BETWEEN_SYM bit_expr AND_SYM predicate {
      auto expression = CreateBetweenExpr( $1, $4, $6 );

      $$ = GenerateNotExpression(expression);
    }
    |   bit_expr SOUNDS_SYM LIKE bit_expr { ThrowNotSupportedException("SOUNDS"); }
    |   bit_expr LIKE simple_expr opt_escape {
      auto target = $1;
      auto object = $3;

      BiaodashiPointer expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Likeop, 0);
      expression->AddChild(target);
      expression->AddChild(object);
      $$ = expression;
    }
    |   bit_expr not LIKE simple_expr opt_escape {
      auto target = $1;
      auto object = $4;

      BiaodashiPointer expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Likeop, 0);
      expression->AddChild(target);
      expression->AddChild(object);
      $$ = GenerateNotExpression(expression);
    }
    |   bit_expr REGEXP bit_expr{ ThrowNotSupportedException("REGEXP"); }
    |   bit_expr not REGEXP bit_expr{ ThrowNotSupportedException("NOT REGEXP"); }
    |   bit_expr { $$ = $1; }
    ;

bit_expr:
        bit_expr '|' bit_expr %prec '|'{ ThrowNotSupportedException("operator bitor |"); }
    |   bit_expr '&' bit_expr %prec '&'{ ThrowNotSupportedException("operator bitand &"); }
    |   bit_expr SHIFT_LEFT bit_expr %prec SHIFT_LEFT{ ThrowNotSupportedException("operator SHIFT_LEFT"); }
    |   bit_expr SHIFT_RIGHT bit_expr %prec SHIFT_RIGHT{ ThrowNotSupportedException("operator SHIFT_RIGHT"); }
    |   bit_expr '+' bit_expr %prec '+' { 
      $$ = CreateCalcExpression($1, CalcType::ADD, $3);
     }
    |   bit_expr '-' bit_expr %prec '-' {
      $$ = CreateCalcExpression($1, CalcType::SUB, $3);
    }
    |   bit_expr '+' INTERVAL_SYM expr interval %prec '+'
    {
      auto interval_value = $4;
      auto interval_type = $5;

      auto interval_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IntervalExpression, interval_type);
      interval_expression->AddChild(interval_value);

      std::vector<Expression> args;
      args.emplace_back($1);
      args.emplace_back(interval_expression);
      $$ = CreateFunctionExpression("DATE_ADD", args);
    }
    |   bit_expr '-' INTERVAL_SYM expr interval %prec '-'
    {
      auto interval_value = $4;
      auto interval_type = $5;

      auto interval_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IntervalExpression, interval_type);
      interval_expression->AddChild(interval_value);

      std::vector<Expression> args;
      args.emplace_back($1);
      args.emplace_back(interval_expression);
      $$ = CreateFunctionExpression("DATE_SUB", args);
    }
    |   bit_expr '*' bit_expr %prec '*' {
      $$ = CreateCalcExpression($1, CalcType::MUL, $3);
    }
    |   bit_expr '/' bit_expr %prec '/' {
      $$ = CreateCalcExpression($1, CalcType::DIV, $3);
    }
    |   bit_expr '%' bit_expr %prec '%'{
      $$ = CreateCalcExpression($1, CalcType::MOD, $3);
    }
    |   bit_expr DIV_SYM bit_expr %prec DIV_SYM {
      $$ = CreateCalcExpression($1, CalcType::DIV, $3);
    }
    |   bit_expr MOD_SYM bit_expr %prec MOD_SYM {
      $$ = CreateCalcExpression($1, CalcType::MOD, $3);
    }
    |   bit_expr '^' bit_expr{ ThrowNotSupportedException("operator xor ^"); }
    |   simple_expr { $$ = $1; }
    ;

or:
        OR_SYM
    |   OR2_SYM
    ;

and:
        AND_SYM
    |   AND_AND_SYM
    ;

not:
        NOT_SYM
    |   NOT2_SYM
    ;

not2:
        '!'
    |   NOT2_SYM
    ;
/*
enum ComparisonType {
    DengYu,
    BuDengYu,
    SQLBuDengYu,
    XiaoYuDengYu,
    DaYuDengYu,
    XiaoYu,
    DaYu

};
*/
comp_op:
        EQ {
          $$ = static_cast<int>(ComparisonType::DengYu);
        }
    |   EQUAL_SYM  {
          ThrowNotSupportedException("<=>");
        }
    |   GE {
          $$ = static_cast<int>(ComparisonType::DaYuDengYu);
        }
    |   GT_SYM {
          $$ = static_cast<int>(ComparisonType::DaYu);
        }
    |   LE {
          $$ = static_cast<int>(ComparisonType::XiaoYuDengYu);
        }
    |   LT {
          $$ = static_cast<int>(ComparisonType::XiaoYu);
        }
    |   NE {
          $$ = static_cast<int>(ComparisonType::BuDengYu);
        }
    ;

all_or_any:
        ALL  { ThrowNotSupportedException("ALL"); }
    |   ANY_SYM { ThrowNotSupportedException("ANY"); }
    ;

expr_tmp:
    literal{
        $$ = LiteralToExpression( $1 );
    }

expr_tmp_list:
    expr_tmp {}
    |
    expr_tmp_list ',' expr_tmp {};

simple_expr:
        simple_ident {
          $$ = CreateIdentExpression($1);
        }
    |   function_call_keyword { $$ = $1; }
    |   function_call_nonkeyword { $$ = $1; }
    |   function_call_generic { $$ = $1; }
    |   function_call_conflict { $$ = $1; }
    |   set_function_specification { $$ = $1; }
    |   simple_expr COLLATE_SYM ident_or_text %prec NEG
    |   literal {
        $$ = LiteralToExpression( $1 );
    }
    |   param_marker
    |   variable
    |   window_func_call { ThrowNotSupportedException("WINDOW FUNCTION"); }
    |   simple_expr OR_OR_SYM simple_expr { ThrowNotSupportedException("OR_OR"); }

    |   '+' simple_expr %prec NEG {
      $$ = $2;
    }
    |   '-' simple_expr %prec NEG {
      auto child = (CommonBiaodashi*)($2.get());
      if (child->GetType() == BiaodashiType::Zhengshu) {
          if (child->GetValueType() == BiaodashiValueType::LONG_INT) {
              auto value = boost::get<int64_t>(child->GetContent());
              auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, 0 - value);
              expression->SetValueType(child->GetValueType());
              $$ = expression;
          } else {
              auto value = boost::get<int>(child->GetContent());
              auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, 0 - value);
              expression->SetValueType(child->GetValueType());
              $$ = expression;
          }
      } else if (child->GetType() == BiaodashiType::Fudianshu) {
          auto value = boost::get<double>(child->GetContent());
          $$ = std::make_shared<CommonBiaodashi>(BiaodashiType::Fudianshu, 0 - value);
      } else if (child->GetType() == BiaodashiType::Decimal) {
          $$ = NegateDecimalExpresstion( child->GetContent() );
      } else {
          auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Yunsuan, CalcType::SUB);
          expression->AddChild(CreateIntegerExpression("0"));
          expression->AddChild($2);
          $$ = expression;
      }
    }
    |   '~' simple_expr %prec NEG { ThrowNotSupportedException("operator ~"); }
    |   not2 simple_expr %prec NEG { ThrowNotSupportedException("NOT2"); }

    |   row_subquery { $$ = $1; }
    |   '(' expr ')' { $$ = $2; }
    |   '(' expr ',' expr_list ')' { 
       auto expression = CreateExprListExpression();
       expression->AddChild($2);
       for (auto expr : $4 ){
         expression->AddChild(expr);
       }
      $$ = expression;  
      }
    |   ROW_SYM '(' expr ',' expr_list ')' { ThrowNotSupportedException("ROW"); }
    |   EXISTS table_subquery {
          auto query_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Query, $2);
          auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Cunzai, 0);
          expression->AddChild(query_expression);
          $$ = expression;
    }
    |   MATCH ident_list_arg AGAINST { ThrowNotSupportedException("MATCH...AGAINST"); }
    |   BINARY_SYM simple_expr %prec NEG {
      $$ = $2;
    }
    |   CAST_SYM '(' expr AS cast_type ')' {
      $$ = CreateCastFunctionExpr( driver, @3, $3, $5 );
    }
    |   TRUNCATE_SYM '(' bit_expr ',' simple_expr ')' {
      std::vector<Expression> args;
      args.emplace_back($3);
      args.emplace_back($5);
      auto expression = CreateFunctionExpression("TRUNCATE", args);
      $$ = expression;
    }
    |   case_expr { $$ = $1; }
    |   CONVERT_SYM '(' expr ',' cast_type ')' {
      $$ = CreateCastFunctionExpr( driver, @3, $3, $5 );
    }
    |   DEFAULT_SYM '(' simple_ident ')' { ThrowNotSupportedException("DEFAULT"); }
    |   VALUES '(' simple_ident_nospvar ')' { ThrowNotSupportedException("VALUES"); }
    |   INTERVAL_SYM expr interval '+' expr %prec INTERVAL_SYM { ThrowNotSupportedException("INTERVAL"); }
    ;
case_expr:
    CASE_SYM opt_expr when_list opt_else END {
        $$ = CreateCaseWhenExpression($2, $3, $4);
    }
    ;
/*
Function call syntax using official SQL 2003 keywords.
Because the function name is an official token,
a dedicated grammar rule is needed in the parser.
There is no potential for conflicts
*/
function_call_keyword:
        CHAR_SYM '(' expr_list ')' { ThrowNotSupportedException("CHAR"); }
    | CURRENT_USER optional_braces {
        $$ = CreateCurrentUserExpression();
    }
    | CONNECTION_ID_SYM '(' ')'{
        $$ = CreateConnectionIdExpression();
    }
    |   DATE_SYM '(' expr ')' {
      $$ = CreateFunctionExpression("DATE", $3);
    }
    |   DATE_SYM TEXT_STRING {
      $$ = CreateFunctionExpression("DATE", CreateStringExpression($2));
    }
    |   DAY_SYM '(' expr ')' {
      $$ = CreateFunctionExpression("DAY", $3);
    }
    |   HOUR_SYM '(' expr ')' { ThrowNotSupportedException("HOUR"); }
    |   INSERT_SYM '(' expr ',' expr ',' expr ',' expr ')' { ThrowNotSupportedException("INSERT"); }
    |   INTERVAL_SYM '(' expr ',' expr ')' %prec INTERVAL_SYM { ThrowNotSupportedException("INTERVAL"); }
    |   INTERVAL_SYM '(' expr ',' expr ',' expr_list ')' %prec INTERVAL_SYM { ThrowNotSupportedException("INTERVAL"); }
    |   LEFT '(' expr ',' expr ')' { ThrowNotSupportedException("LEFT"); }
    |   MINUTE_SYM '(' expr ')' { ThrowNotSupportedException("MINUTE"); }
    |   MONTH_SYM '(' expr ')'  {
      $$ = CreateFunctionExpression("MONTH", $3);
    }
    |   RIGHT '(' expr ',' expr ')' { ThrowNotSupportedException("RIGHT"); }
    |   SECOND_SYM '(' expr ')' { ThrowNotSupportedException("SECOND"); }
    |   TIME_SYM '(' expr ')' { ThrowNotSupportedException("TIME"); }
    |   TIMESTAMP_SYM '(' expr ')' { ThrowNotSupportedException("TIMESTAMP"); }
    |   TIMESTAMP_SYM '(' expr ',' expr ')' { ThrowNotSupportedException("TIMESTAMP"); }
    |   TRIM '(' expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' LEADING expr FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' TRAILING expr FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' BOTH expr FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' LEADING FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' TRAILING FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' BOTH FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   TRIM '(' expr FROM expr ')' { ThrowNotSupportedException("TRIM"); }
    |   USER '(' ')' { ThrowNotSupportedException("USER"); }
    |   YEAR_SYM '(' expr ')' { ThrowNotSupportedException("YEAR"); }
    |   COALESCE  '(' expr_list ')' {
      $$ = CreateFunctionExpression("COALESCE", $3);
    }
    |   MOD_SYM '(' expr ',' expr  ')' {
      $$ = CreateCalcExpression($3, CalcType::MOD, $5);
    }
    ;

/*
Function calls using non reserved keywords, with special syntaxic forms.
Dedicated grammar rules are needed because of the syntax,
but also have the potential to cause incompatibilities with other
parts of the language.
MAINTAINER:
The only reasons a function should be added here are:
- for compatibility reasons with another SQL syntax (CURDATE),
- for typing reasons (GET_FORMAT)
Any other 'Syntaxic sugar' enhancements should be *STRONGLY*
discouraged.
*/
function_call_nonkeyword:
        ADDDATE_SYM '(' expr ',' expr ')' {
      std::vector<Expression> args;
      args.emplace_back($3);
      args.emplace_back($5);

      $$ = CreateFunctionExpression("DATE_ADD", args);
    }
    |   ADDDATE_SYM '(' expr ',' INTERVAL_SYM expr interval ')' {
      auto interval_value = $6;
      auto interval_type = $7;

      BiaodashiPointer interval_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IntervalExpression, interval_type);
      interval_expression->AddChild(interval_value);

      std::vector<Expression> args;
      args.emplace_back($3);
      args.emplace_back(interval_expression);

      $$ = CreateFunctionExpression("DATE_ADD", args);
    }
    |   CURDATE optional_braces { ThrowNotSupportedException("CURDATE"); }
    |   CURTIME func_datetime_precision { ThrowNotSupportedException("CURTIME"); }
    |   DATE_ADD_INTERVAL '(' expr ',' INTERVAL_SYM expr interval ')' {
      auto interval_value = $6;
      auto interval_type = $7;

      BiaodashiPointer interval_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IntervalExpression, interval_type);
      interval_expression->AddChild(interval_value);

      std::vector<Expression> args;
      args.emplace_back($3);
      args.emplace_back(interval_expression);

      $$ = CreateFunctionExpression("DATE_ADD", args);
    }
    |   DATE_SUB_INTERVAL '(' expr ',' INTERVAL_SYM expr interval ')' {
      auto interval_value = $6;
      auto interval_type = $7;

      BiaodashiPointer interval_expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IntervalExpression, interval_type);
      interval_expression->AddChild(interval_value);

      std::vector<Expression> args;
      args.emplace_back($3);
      args.emplace_back(interval_expression);

      $$ = CreateFunctionExpression("DATE_SUB", args);
    }
    |   EXTRACT_SYM '(' interval FROM expr ')' {
      string function_name = "EXTRACT";
      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, function_name);

      auto interval = CreateStringExpression($3);
      expression->AddChild(interval);
      expression->AddChild($5);
      $$ = expression;
    }
    |   EXTRACT_SYM '(' interval ',' expr ')' {
      string function_name = "EXTRACT";
      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, function_name);

      auto interval = CreateStringExpression($3);
      expression->AddChild(interval);
      expression->AddChild($5);
      $$ = expression;
    }
    |   GET_FORMAT '(' date_time_type  ',' expr ')' { ThrowNotSupportedException("GET_FORMAT"); }
    |   now 
    |   POSITION_SYM '(' bit_expr IN_SYM expr ')' { ThrowNotSupportedException("POSITION"); }
    |   SUBDATE_SYM '(' expr ',' expr ')' { ThrowNotSupportedException("SUBDATE"); }
    |   SUBDATE_SYM '(' expr ',' INTERVAL_SYM expr interval ')' { ThrowNotSupportedException("SUBDATE"); }
    |   SUBSTRING '(' expr ',' expr ',' expr ')' {
      string function_name = "SUBSTRING";
      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, function_name);

      expression->AddChild($3);
      expression->AddChild($5);
      expression->AddChild($7);

      $$ = expression;
    }
    |   SUBSTRING '(' expr ',' expr ')' {
      string function_name = "SUBSTRING";
      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, function_name);

      expression->AddChild($3);
      expression->AddChild($5);
      expression->AddChild(CreateIntegerExpression("-1"));

      $$ = expression;
    }
    |   SUBSTRING '(' expr FROM expr FOR_SYM expr ')' {
      string function_name = "SUBSTRING";
      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, function_name);

      expression->AddChild($3);
      expression->AddChild($5);
      expression->AddChild($7);

      $$ = expression;
    }
    |   SUBSTRING '(' expr FROM expr ')' {
      string function_name = "SUBSTRING";
      auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, function_name);

      expression->AddChild($3);
      expression->AddChild($5);
      expression->AddChild(CreateIntegerExpression("-1"));

      $$ = expression;
    }
    |   SYSDATE func_datetime_precision { ThrowNotSupportedException("SYSDATE"); }
    |   TIMESTAMP_ADD '(' interval_time_stamp ',' expr ',' expr ')' { ThrowNotSupportedException("TIMESTAMP_ADD"); }
    |   TIMESTAMP_DIFF '(' interval_time_stamp ',' expr ',' expr ')' { ThrowNotSupportedException("TIMESTAMP_DIFF"); }
    |   UTC_DATE_SYM optional_braces { ThrowNotSupportedException("UTC_DATE"); }
    |   UTC_TIME_SYM func_datetime_precision { ThrowNotSupportedException("UTC_TIME"); }
    |   UTC_TIMESTAMP_SYM func_datetime_precision { ThrowNotSupportedException("UTC_TIMESTAMP"); }
    |   DICT_INDEX_SYM '(' expr ')' {
        $$ = CreateFunctionExpression("DICT_INDEX", $3);
    }
    ;

/*
  Functions calls using a non reserved keyword, and using a regular syntax.
  Because the non reserved keyword is used in another part of the grammar,
  a dedicated rule is needed here.
*/
function_call_conflict:
        DATABASE '(' ')'
          {
            $$ = CreateCurrentDbExpression();
          }
        | IF '(' expr ',' expr ',' expr ')'
          {
              auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IfCondition, 0);

              expression->AddChild($3);
              expression->AddChild($5);
              expression->AddChild($7);

              // expression->SetName();
              $$ = expression;
          }
        ;

/*
  Regular function calls.
  The function name is *not* a token, and therefore is guaranteed to not
  introduce side effects to the language in general.
  MAINTAINER:
  All the new functions implemented for new features should fit into
  this category. The place to implement the function itself is
  in sql/item_create.cc
*/
function_call_generic:
        VERSION_SYM '(' ')' {
            $$ = CreateServerVersionExpression();
        }
        | IDENT_sys '(' opt_udf_expr_list ')'
          {
            $$ = CreateFunctionExpression($1, $3);
          }
        | ident '.' ident '(' opt_expr_list ')' { ThrowNotSupportedException( $1 + "." + $3 ); }
    ;

opt_udf_expr_list:
        /* empty */     { $$ = std::vector<Expression>(); }
        | udf_expr_list { $$ = $1; }
        ;

udf_expr_list:
          udf_expr
          {
            std::vector<Expression> list;
            list.emplace_back($1);
            $$ = list;
          }
        | udf_expr_list ',' udf_expr
          {
            $1.emplace_back($3);
            $$ = $1;
          }
        ;

udf_expr:
          expr /* select_alias*/
          {
            $$= $1;
          }
        ;

set_function_specification:
          sum_expr { $$ = $1; }
        | grouping_operation { ThrowNotSupportedException("GROUPING"); }
        ;
grouping_operation:
          GROUPING_SYM '(' expr_list ')'
        ;

sum_expr:
          AVG_SYM '(' in_sum_expr ')' opt_windowing_clause {
          $$ = CreateFunctionExpression("AVG", $3);
        }

        | AVG_SYM '(' DISTINCT in_sum_expr ')' opt_windowing_clause {
          $$ = CreateDistinctFunctionExpression("AVG", $4);
        }
        
        | COUNT_SYM '(' opt_all '*' ')' opt_windowing_clause {
          auto arg = std::make_shared<CommonBiaodashi>(BiaodashiType::Star, string());
          $$ = CreateFunctionExpression("COUNT", arg);
        }

        | COUNT_SYM '(' in_sum_expr ')' opt_windowing_clause {
          $$ = CreateFunctionExpression("COUNT", $3);
        }
        | COUNT_SYM '(' DISTINCT expr_list ')' opt_windowing_clause {
          if ( $4.size() > 1 )
          {
            ThrowNotSupportedException( "count distinct with more than 1 param" );
          }
          $$ = CreateDistinctFunctionExpression("COUNT", $4);
        }
        | MIN_SYM '(' in_sum_expr ')' opt_windowing_clause
        /*
          According to ANSI SQL, DISTINCT is allowed and has
          no sense inside MIN and MAX grouping functions; so MIN|MAX(DISTINCT ...)
          is processed like an ordinary MIN | MAX()
        */ {
          $$ = CreateFunctionExpression("MIN", $3);
        }
        | MIN_SYM '(' DISTINCT in_sum_expr ')' opt_windowing_clause {
          $$ = CreateDistinctFunctionExpression("MIN", $4);
        }

        | MAX_SYM '(' in_sum_expr ')' opt_windowing_clause {
          $$ = CreateFunctionExpression("MAX", $3);
        }

        | MAX_SYM '(' DISTINCT in_sum_expr ')' opt_windowing_clause {
          $$ = CreateDistinctFunctionExpression("MAX", $4);
        }

        | SUM_SYM '(' in_sum_expr ')' opt_windowing_clause {
          $$ = CreateFunctionExpression("SUM", $3);
        }
        | SUM_SYM '(' DISTINCT in_sum_expr ')' opt_windowing_clause {
          $$ = CreateDistinctFunctionExpression("SUM", $4);
        }
        | GROUP_CONCAT_SYM '(' opt_distinct
                  expr_list opt_gorder_clause
                  opt_gconcat_separator
                  ')' opt_windowing_clause  {
            $$ = CreateGroupConcatExpression();
        }
        ;
opt_distinct:
          /* empty */ { $$ = 0; }
        | DISTINCT    { $$ = 1; }
        ;
opt_gconcat_separator:
          /* empty */
        | SEPARATOR_SYM text_string { ThrowNotSupportedException("SEPARATOR"); }
        ;
opt_gorder_clause:
          /* empty */
        | ORDER_SYM BY gorder_list { ThrowNotSupportedException("ORDER BY"); }
        ;
gorder_list:
          gorder_list ',' order_expr { }
        | order_expr { }
        ;

/*
   UNIONS : glue selects together
*/


union_option:
          /* empty */ { $$ = SetOperationType::UNION; }
        | DISTINCT { $$ = SetOperationType::UNION; }
        | ALL { $$ = SetOperationType::UNION_ALL; }
        ;
    
row_subquery:
          subquery {
            debug_line();
            $$ = std::make_shared<CommonBiaodashi>(BiaodashiType::Query, $1);
          }
        ;

table_subquery:
          subquery { $$ = $1; }
        ;

subquery:
          query_expression_parens %prec SUBQUERY_AS_EXPR { $$ = $1; }
        ;

query_spec_option:
          STRAIGHT_JOIN { $$ = SelectOption::STRAIGHT_JOIN; }
        | HIGH_PRIORITY { $$ = SelectOption::HIGH_PRIORITY; }
        | DISTINCT { $$ = SelectOption::DISTINCT; }
        | SQL_SMALL_RESULT { $$ = SelectOption::SQL_SMALL_RESULT; }
        | SQL_BIG_RESULT { $$ = SelectOption::SQL_BIG_RESULT; }
        | SQL_BUFFER_RESULT  { $$ = SelectOption::SQL_BUFFER_RESULT; }
        | SQL_CALC_FOUND_ROWS { $$ = SelectOption::SQL_CALC_FOUND_ROWS; }
        | ALL { $$ = SelectOption::ALL; }
        ;

create:
          CREATE DATABASE opt_if_not_exists ident opt_create_database_options {
            $$ = CreateCreateDbStructure($3, $4);
          }
          | CREATE view_or_trigger_or_sp_or_event {
            if ($2 == nullptr) {
              ThrowNotSupportedException("create view or trigger");
            }
            $$ = $2;
          }
          | CREATE USER opt_if_not_exists create_user_list
                        require_clause connect_options
                        opt_account_lock_password_expire_options
            {
              $$ = CreateAccountMgmtStructure( CommandType::CreateUser, $3, $4 );
            }
          | CREATE LOGFILE_SYM GROUP_SYM ident ADD lg_undofile
            opt_logfile_group_options
            {
                ThrowNotSupportedException("create logfile group");
            }
          | CREATE TABLESPACE_SYM ident opt_ts_datafile_name
            opt_logfile_group_name opt_tablespace_options
            {
                ThrowNotSupportedException("create tablespace");
            }
          | CREATE UNDO_SYM TABLESPACE_SYM ident ADD ts_datafile
            opt_undo_tablespace_options
            {
                ThrowNotSupportedException("create undo tablespace");
            }
          | CREATE SERVER_SYM ident_or_text FOREIGN DATA_SYM WRAPPER_SYM
            ident_or_text OPTIONS_SYM '(' server_options_list ')'
            {
                ThrowNotSupportedException("create server");
            }
          ;

role_ident:
          IDENT_sys
        | role_keyword
          {
          }
        ;
role_ident_or_text:
          role_ident
        | TEXT_STRING_sys
        | LEX_HOSTNAME
        ;

role:
          role_ident_or_text
          {
          }
        | role_ident_or_text '@' ident_or_text
          {
          }
        ;
role_list:
          role
          {
          }
        | role_list ',' role
          {
          }
        ;
default_role_clause:
          /* empty */
          {
          }
        |
          DEFAULT_SYM ROLE_SYM role_list
          {
            ThrowNotSupportedException("ROLE");
          }
        ;
require_list:
          require_list_element opt_and require_list
        | require_list_element
        ;

require_list_element:
          SUBJECT_SYM TEXT_STRING
          {
              ThrowNotSupportedException("SSL/TLS option 'subject'");
          }
        | ISSUER_SYM TEXT_STRING
          {
              ThrowNotSupportedException("SSL/TLS option 'issuer'");
          }
        | CIPHER_SYM TEXT_STRING
          {
              ThrowNotSupportedException("SSL/TLS option 'cipher'");
          }
        ;
require_clause:
          /* empty */
        | REQUIRE_SYM require_list
          {
          }
        | REQUIRE_SYM SSL_SYM
          {
              ThrowNotSupportedException("SSL/TLS option 'ssl'");
          }
        | REQUIRE_SYM X509_SYM
          {
              ThrowNotSupportedException("SSL/TLS option 'x509'");
          }
        | REQUIRE_SYM NONE_SYM
          {
          }
        ;
connect_options:
          /* empty */ {}
        | WITH connect_option_list { ThrowNotSupportedException("WITH"); }
        ;

connect_option_list:
          connect_option_list connect_option {}
        | connect_option {}
        ;

connect_option:
          MAX_QUERIES_PER_HOUR ulong_num
          {
          }
        | MAX_UPDATES_PER_HOUR ulong_num
          {
          }
        | MAX_CONNECTIONS_PER_HOUR ulong_num
          {
          }
        | MAX_USER_CONNECTIONS_SYM ulong_num
          {
          }
        ;

opt_account_lock_password_expire_options:
          /* empty */ {}
        | opt_account_lock_password_expire_option_list{ ThrowNotSupportedException("account options"); }
        ;

opt_account_lock_password_expire_option_list:
          opt_account_lock_password_expire_option
        | opt_account_lock_password_expire_option_list opt_account_lock_password_expire_option
        ;

opt_account_lock_password_expire_option:
          ACCOUNT_SYM UNLOCK_SYM
          {
          }
        | ACCOUNT_SYM LOCK_SYM
          {
          }
        | PASSWORD EXPIRE_SYM
          {
          }
        | PASSWORD EXPIRE_SYM INTERVAL_SYM real_ulong_num DAY_SYM
          {
          }
        | PASSWORD EXPIRE_SYM NEVER_SYM
          {
          }
        | PASSWORD EXPIRE_SYM DEFAULT_SYM
          {
          }
        | PASSWORD HISTORY_SYM real_ulong_num
          {
          }
        | PASSWORD HISTORY_SYM DEFAULT_SYM
          {
          }
        | PASSWORD REUSE_SYM INTERVAL_SYM real_ulong_num DAY_SYM
          {
          }
        | PASSWORD REUSE_SYM INTERVAL_SYM DEFAULT_SYM
          {
          }
        | PASSWORD REQUIRE_SYM CURRENT_SYM
          {
          }
        | PASSWORD REQUIRE_SYM CURRENT_SYM DEFAULT_SYM
          {
          }
        | PASSWORD REQUIRE_SYM CURRENT_SYM OPTIONAL_SYM
          {
          }
        ;
user:
          ident_or_text
          {
            $$ = std::make_shared< Account >( $1, "", "", "" );
          }
        | ident_or_text '@' ident_or_text
          {
            $$ = std::make_shared< Account >( $1, $3, "", "" );
          }
        | CURRENT_USER optional_braces
          {
            $$ = std::make_shared< Account >();
          }
        ;

user_list:
          user
          {
            $$ = std::make_shared< vector< AccountSPtr > >();
            $$->emplace_back($1);
          }
        | user_list ',' user
          {
            $1->emplace_back($3);
            $$ = $1;
          }
        ;
create_user:
          user IDENTIFIED_SYM BY TEXT_STRING_password
          {
            $$ = $1;
            $$->m_authStr = $4;
          }
        | user IDENTIFIED_SYM WITH ident_or_text
          {
            $$ = $1;
            $$->SetAuthPlugin( $4 );
          }
        | user IDENTIFIED_SYM WITH ident_or_text AS TEXT_STRING_hash
          {
            $$ = $1;
            $$->SetAuthPlugin( $4 );
            $$->m_authStr = $6;
            $$->m_authStrHashed = true;
          }
        | user IDENTIFIED_SYM WITH ident_or_text BY TEXT_STRING_password
          {
            $$ = $1;
            $$->SetAuthPlugin( $4 );
            $$->m_authStr = $6;
          }
        | user
          {
            $$ = $1;
          }
        ;

create_user_list:
          create_user
          {
            $$ = std::make_shared< vector< AccountSPtr > >();
            $$->emplace_back($1);
          }
        | create_user_list ',' create_user
          {
            $1->emplace_back($3);
            $$ = $1;
          }
        ;
opt_create_database_options:
          /* empty */ {}
        | create_database_options {}
        ;

create_database_options:
          create_database_option {}
        | create_database_options create_database_option {}
        ;

create_database_option:
          default_collation
          {
          }
        | default_charset
          {
            //ThrowNotSupportedException("default_charset");
          }
        | default_encryption
          {
            ThrowNotSupportedException("default_encryption");
            // Validate if we have either 'y|Y' or 'n|N'
          }
        ;

duplicate:
          REPLACE_SYM { ThrowNotSupportedException("REPLACE"); }
        | IGNORE_SYM  { ThrowNotSupportedException("IGNORE"); }
        ;
opt_if_not_exists:
          /* empty */   { $$= false; }
        | IF not EXISTS { $$= true; }
        ;

create_table_options:
          create_table_option
          {
            
          }
        | create_table_options opt_comma create_table_option
          {
          }
        ;

opt_comma:
          /* empty */
        | ','
        ;

create_table_option:
          ENGINE_SYM opt_equal ident_or_text
          {
            //ThrowNotSupportedException("ENGINE");
          }
        | SECONDARY_ENGINE_SYM opt_equal NULL_SYM
          {
            ThrowNotSupportedException("SECONDARY_ENGINE");
          }
        | SECONDARY_ENGINE_SYM opt_equal ident_or_text
          {
            ThrowNotSupportedException("SECONDARY_ENGINE");
          }
        | MAX_ROWS opt_equal ulonglong_num
          {
            ThrowNotSupportedException("MAX_ROWS");
          }
        | MIN_ROWS opt_equal ulonglong_num
          {
            ThrowNotSupportedException("MIN_ROWS");
          }
        | AVG_ROW_LENGTH opt_equal ulong_num
          {
            ThrowNotSupportedException("AVG_ROW_LENGTH");
          }
        | PASSWORD opt_equal TEXT_STRING_sys
          {
            ThrowNotSupportedException("PASSWORD");
          }
        | COMMENT_SYM opt_equal TEXT_STRING_sys
          {
          }
        | COMPRESSION_SYM opt_equal TEXT_STRING_sys
      {
        ThrowNotSupportedException("COMPRESSION");
      }
        | ENCRYPTION_SYM opt_equal TEXT_STRING_sys
      {
        ThrowNotSupportedException("ENCRYPTION");
      }
        | AUTO_INC opt_equal ulonglong_num
          {
            ThrowNotSupportedException("AUTO_INC");
          }
        | PACK_KEYS_SYM opt_equal ternary_option
          {
            ThrowNotSupportedException("PACK_KEYS");
          }
        | STATS_AUTO_RECALC_SYM opt_equal ternary_option
          {
            ThrowNotSupportedException("STATS_AUTO_RECALC");
          }
        | STATS_PERSISTENT_SYM opt_equal ternary_option
          {
            ThrowNotSupportedException("STATS_PERSISTENT");
          }
        | STATS_SAMPLE_PAGES_SYM opt_equal ulong_num
          {
            /* From user point of view STATS_SAMPLE_PAGES can be specified as
            STATS_SAMPLE_PAGES=N (where 0<N<=65535, it does not make sense to
            scan 0 pages) or STATS_SAMPLE_PAGES=default. Internally we record
            =default as 0. See create_frm() in sql/table.cc, we use only two
            bytes for stats_sample_pages and this is why we do not allow
            larger values. 65535 pages, 16kb each means to sample 1GB, which
            is impractical. If at some point this needs to be extended, then
            we can store the higher bits from stats_sample_pages in .frm too. */
            if ($3 == 0 || $3 > 0xffff)
            {
              assert(0);
            }
          }
        | STATS_SAMPLE_PAGES_SYM opt_equal DEFAULT_SYM
          {
            ThrowNotSupportedException("STATS_SAMPLE_PAGES");
          }
        | CHECKSUM_SYM opt_equal ulong_num
          {
            ThrowNotSupportedException("CHECKSUM");
          }
        | TABLE_CHECKSUM_SYM opt_equal ulong_num
          {
            ThrowNotSupportedException("TABLE_CHECKSUM");
          }
        | DELAY_KEY_WRITE_SYM opt_equal ulong_num
          {
            ThrowNotSupportedException("DELAY_KEY_WRITE");
          }
        | ROW_FORMAT_SYM opt_equal row_types
          {
            ThrowNotSupportedException("ROW_FORMAT");
          }
        | UNION_SYM opt_equal '(' opt_table_list ')'
          {
            ThrowNotSupportedException("UNION");
          }
        | default_charset
          {
            //ThrowNotSupportedException("default_charset");
          }
        | default_collation
          {
          }
        | INSERT_METHOD opt_equal merge_insert_types
          {
            ThrowNotSupportedException("INSERT");
          }
        | DATA_SYM DIRECTORY_SYM opt_equal TEXT_STRING_sys
          {
            ThrowNotSupportedException("DATA DIRECTORY");
          }
        | INDEX_SYM DIRECTORY_SYM opt_equal TEXT_STRING_sys
          {
            ThrowNotSupportedException("INDEX DIRECTORY");
          }
        | TABLESPACE_SYM opt_equal ident
          {
            ThrowNotSupportedException("TABLESPACE");
          }
        | STORAGE_SYM DISK_SYM
          {
            ThrowNotSupportedException("STORAGE");
          }
        | STORAGE_SYM MEMORY_SYM
          {
            ThrowNotSupportedException("STORAGE");
          }
        | CONNECTION_SYM opt_equal TEXT_STRING_sys
          {
            ThrowNotSupportedException("CONNECTION");
          }
        | KEY_BLOCK_SIZE opt_equal ulong_num
          {
            ThrowNotSupportedException("KEY_BLOCK_SIZE");
          }
        ;

ternary_option:
          ulong_num
          {
            switch($1) {
            case 0:
                break;
            case 1:
                break;
            default:
                assert(0);
            }
          }
        | DEFAULT_SYM { ThrowNotSupportedException("DEFAULT"); }
        ;
default_charset:
          opt_default character_set opt_equal charset_name 
          { 
            //ThrowNotSupportedException("default_charset"); 
          }
        ;

default_collation:
          opt_default COLLATE_SYM opt_equal collation_name { }
        ;

default_encryption:
          opt_default ENCRYPTION_SYM opt_equal TEXT_STRING_sys { ThrowNotSupportedException("default_encryption"); }
        ;
row_types:
          DEFAULT_SYM    { ThrowNotSupportedException("row types DEFAULT"); }
        | FIXED_SYM      { ThrowNotSupportedException("row types FIXED"); }
        | DYNAMIC_SYM    { ThrowNotSupportedException("row types DYNAMIC"); }
        | COMPRESSED_SYM { ThrowNotSupportedException("row types COMPRESSED"); }
        | REDUNDANT_SYM  { ThrowNotSupportedException("row types REDUNDANT"); }
        | COMPACT_SYM    { ThrowNotSupportedException("row types CMPACT"); }
        ;

merge_insert_types:
         NO_SYM          { ThrowNotSupportedException("merge_insert_types NO"); }
       | FIRST_SYM       { ThrowNotSupportedException("merge_insert_types FIRST"); }
       | LAST_SYM        { ThrowNotSupportedException("merge_insert_types LAST"); }
       ;
udf_type:
          STRING_SYM { ThrowNotSupportedException("udf_type STRING"); }
        | REAL_SYM { ThrowNotSupportedException("udf_type REAL"); }
        | DECIMAL_SYM { ThrowNotSupportedException("udf_type DECIMAL"); }
        | INT_SYM { ThrowNotSupportedException("udf_type INT"); }
        | INTEGER_SYM { ThrowNotSupportedException("udf_type INTEGER"); }
        ;
opt_table_list:
          /* empty */  { $$ = nullptr; }
        | table_list
        ;
table_list:
          table_ident
          {
            $$ = std::make_shared<vector<std::shared_ptr<BasicRel>>>();
            $$->emplace_back($1);
          }
        | table_list ',' table_ident
          {
            $1->emplace_back($3);
            $$ = $1;
          }
        ;

table_element_list:
          table_element
          {
            auto table_element_list = std::make_shared<std::vector<TableElementDescriptionPtr>>();
            table_element_list->emplace_back($1);
            $$ = table_element_list;
          }
        | table_element_list ',' table_element
          {
            $1->emplace_back($3);
            $$ = $1;
          }
        ;

table_element:
          column_def            { $$ = $1; }
        | table_constraint_def  { $$ = $1; }
        ;

column_def:
          ident field_def opt_references
          {
            $$ = CreateColumnDef($1, $2 );
          }
        ;

opt_references:
          /* empty */      { }
        |  references
          {
            ThrowNotSupportedException("references");
             /* Currently we ignore FK references here: */
          }
        ;

table_constraint_def:
          key_or_index opt_index_name_and_type '(' key_list_with_expression ')'
          opt_index_options
          {
            ThrowNotSupportedException("multiple key or index");
            //$$ = std::make_shared<PT_inline_index_definition>(KEYTYPE_MULTIPLE, "");
          }
        | FULLTEXT_SYM opt_key_or_index opt_ident '(' key_list_with_expression ')'
          opt_fulltext_index_options
          {
              ThrowNotSupportedException("fulltext key");
          }
        | SPATIAL_SYM opt_key_or_index opt_ident '(' key_list_with_expression ')'
          opt_spatial_index_options
          {
              ThrowNotSupportedException("spatial key");
          }
        | opt_constraint_name constraint_key_type opt_index_name_and_type
          '(' key_list_with_expression ')' opt_index_options
          {
            /*
              Constraint-implementing indexes are named by the constraint type
              by default.
            */
            $$ = std::make_shared< PT_table_key_constraint_def >( $1, $3, $2, $5 );
          }
        | opt_constraint_name FOREIGN KEY_SYM opt_ident '(' key_list ')' references
          {
              ThrowNotSupportedException("foreign key");
            //$$ = std::make_shared< PT_table_key_constraint_def >( $1, KEYTYPE_FOREIGN, $6, $8.first, $8.second );
          }
        | opt_constraint_name check_constraint opt_constraint_enforcement
          {
              ThrowNotSupportedException("check constraint");
          }
        ;

check_constraint:
          CHECK_SYM '(' expr ')' { ThrowNotSupportedException("check constraint"); }
        ;

opt_constraint_name:
          /* empty */          { }
        | CONSTRAINT opt_ident { $$ = $2; }
        ;

opt_not:
          /* empty */  { $$= false; }
        | NOT_SYM      { $$= true; }
        ;

opt_constraint_enforcement:
          /* empty */            { }
        | constraint_enforcement {  }
        ;

constraint_enforcement:
          opt_not ENFORCED_SYM  { ThrowNotSupportedException("ENFORCED"); }
        ;

field_options:
          /* empty */ { $$= Field_option::NONE; }
        | field_opt_list
        ;

field_opt_list:
          field_opt_list field_option
          {
            $$= static_cast<Field_option>(static_cast<ulong>($1) |
                                          static_cast<ulong>($2));
          }
        | field_option
        ;
field_option:
          SIGNED_SYM   { $$= Field_option::NONE; } // TODO: remove undocumented ignored syntax
        | UNSIGNED_SYM {
            // $$= Field_option::UNSIGNED;
            ThrowNotSupportedException("unsigned data type");
        }
        | ZEROFILL_SYM {
            // $$= Field_option::ZEROFILL_UNSIGNED;
            ThrowNotSupportedException("unsigned data type");
        }
        ;

create_table_stmt:
          CREATE opt_temporary TABLE_SYM opt_if_not_exists table_ident // 5
          '(' table_element_list ')' opt_create_table_options_etc
          {
            $$ = CreateCreateTableStructure($2, $4, $5, $7, $9);
          }
        | CREATE opt_temporary TABLE_SYM opt_if_not_exists table_ident
          opt_create_table_options_etc
          {
            // $$ = CreateCreateTableStructure($2, $4, $5, $7);
            ThrowNotSupportedException("create table AS query expression");
          }
        | CREATE opt_temporary TABLE_SYM opt_if_not_exists table_ident
          LIKE table_ident
          {
            ThrowNotSupportedException("create table LIKE table");
          }
        | CREATE opt_temporary TABLE_SYM opt_if_not_exists table_ident
          '(' LIKE table_ident ')'
          {
            ThrowNotSupportedException("create table LIKE table");
          }
        ;

if_exists:
          /* empty */ { $$ = false; }
        | IF EXISTS { $$ = true; }
        ;

opt_temporary:
          /* empty */ { $$= false; }
        | TEMPORARY   { $$= true; }
        ;

/**************************************************************************

 CREATE VIEW | TRIGGER | PROCEDURE statements.

**************************************************************************/

init_lex_create_info:
          /* empty */
          {
          }
        ;

view_or_trigger_or_sp_or_event:
          definer init_lex_create_info definer_tail
          {
            $$ = nullptr;
          }
        | no_definer init_lex_create_info no_definer_tail
          {
            $$ = $3;
          }
        | view_replace_or_algorithm definer_opt init_lex_create_info view_tail
          {}
        ;

definer_tail:
          view_tail
        | trigger_tail
        | sp_tail
        | sf_tail
        | event_tail
        ;

no_definer_tail:
          view_tail {
            $$ = $1;
          }
        | trigger_tail { ThrowNotSupportedException("TRIGGER"); }
        | sp_tail { ThrowNotSupportedException("PROCEDURE"); }
        | sf_tail{ ThrowNotSupportedException("FUNCTION"); }
        | udf_tail{ ThrowNotSupportedException("FUNCTION"); }
        | event_tail{ ThrowNotSupportedException("EVENT"); }
        ;

sp_fetch_list:
          ident
          {
          }
        | sp_fetch_list ',' ident
          {
          }
        ;

sp_if:
          {                     /*$1*/
          }
          expr                  /*$2*/
          {                     /*$3*/
          }
          THEN_SYM              /*$4*/
          sp_proc_stmts1        /*$5*/
          {                     /*$6*/
          }
          sp_elseifs            /*$7*/
          {                     /*$8*/
          }
        ;

sp_elseifs:
          /* Empty */
        | ELSEIF_SYM sp_if
        | ELSE sp_proc_stmts1
        ;
case_stmt_specification:
          simple_case_stmt
        | searched_case_stmt
        ;

simple_case_stmt:
          CASE_SYM                      /*$1*/
          {                             /*$2*/
          }
          expr                          /*$3*/
          {                             /*$4*/
          }
          simple_when_clause_list       /*$5*/
          else_clause_opt               /*$6*/
          END                           /*$7*/
          CASE_SYM                      /*$8*/
          {                             /*$9*/
          }
        ;

searched_case_stmt:
          CASE_SYM
          {
          }
          searched_when_clause_list
          else_clause_opt
          END
          CASE_SYM
          {
          }
        ;

simple_when_clause_list:
          simple_when_clause
        | simple_when_clause_list simple_when_clause
        ;

searched_when_clause_list:
          searched_when_clause
        | searched_when_clause_list searched_when_clause
        ;

simple_when_clause:
          WHEN_SYM                      /*$1*/
          {                             /*$2*/
          }
          expr                          /*$3*/
          {                             /*$4*/
          }
          THEN_SYM                      /*$5*/
          sp_proc_stmts1                /*$6*/
          {                             /*$7*/
          }
        ;

searched_when_clause:
          WHEN_SYM                      /*$1*/
          {                             /*$2*/
          }
          expr                          /*$3*/
          {                             /*$4*/
          }
          THEN_SYM                      /*$6*/
          sp_proc_stmts1                /*$7*/
          {                             /*$8*/
          }
        ;

else_clause_opt:
          /* empty */
          {
          }
        | ELSE sp_proc_stmts1
        ;

sp_block_content:
          BEGIN_SYM
          { /* QQ This is just a dummy for grouping declarations and statements
              together. No [[NOT] ATOMIC] yet, and we need to figure out how
              make it coexist with the existing BEGIN COMMIT/ROLLBACK. */
          }
          sp_decls
          sp_proc_stmts
          END
          {
          }
        ;

sp_decls:
          /* Empty */
          {
          }
        | sp_decls sp_decl ';'
          {
            /* We check for declarations out of (standard) order this way
              because letting the grammar rules reflect it caused tricky
               shift/reduce conflicts with the wrong result. (And we get
               better error handling this way.) */
          }
        ;
sp_decl_idents:
          ident
          {

          }
        | sp_decl_idents ',' ident
          {
          }
        ;

sp_opt_default:
        /* Empty */
          {
          }
        | DEFAULT_SYM expr
          {
          }
        ;
sp_decl:
          DECLARE_SYM           /*$1*/
          sp_decl_idents        /*$2*/
          type                  /*$3*/
          opt_collate           /*$4*/
          sp_opt_default        /*$5*/
          {                     /*$6*/
          }
        | DECLARE_SYM ident CONDITION_SYM FOR_SYM sp_cond
          {
          }
        | DECLARE_SYM sp_handler_type HANDLER_SYM FOR_SYM
          {
          }
          sp_hcond_list sp_proc_stmt
          {
          }
        | DECLARE_SYM   /*$1*/
          ident         /*$2*/
          CURSOR_SYM    /*$3*/
          FOR_SYM       /*$4*/
          {             /*$5*/
          }
          select_stmt   /*$6*/
          {             /*$7*/
          }
        ;

sp_proc_stmts:
          /* Empty */ {}
        | sp_proc_stmts  sp_proc_stmt ';'
        ;
sp_proc_stmts1:
          sp_proc_stmt ';' {}
        | sp_proc_stmts1  sp_proc_stmt ';'
        ;

sp_proc_stmt:
          sp_proc_stmt_statement { ThrowNotSupportedException("proc statement"); }
        | sp_proc_stmt_return { ThrowNotSupportedException("RETURN"); }
        | sp_proc_stmt_if { ThrowNotSupportedException("IF"); }
        | case_stmt_specification { ThrowNotSupportedException("CASE"); }
        | sp_labeled_block{ ThrowNotSupportedException("block"); }
        | sp_unlabeled_block{ ThrowNotSupportedException("block"); }
        | sp_labeled_control{ ThrowNotSupportedException("label"); }
        | sp_proc_stmt_unlabeled{ ThrowNotSupportedException("statements"); }
        | sp_proc_stmt_leave{ ThrowNotSupportedException("LEAVE"); }
        | sp_proc_stmt_iterate{ ThrowNotSupportedException("ITERATE"); }
        | sp_proc_stmt_open{ ThrowNotSupportedException("OPEN"); }
        | sp_proc_stmt_fetch { ThrowNotSupportedException("FETCH"); }
        | sp_proc_stmt_close{ ThrowNotSupportedException("CLOSE"); }
        ;

sp_opt_fetch_noise:
          /* Empty */
        | NEXT_SYM FROM
        | FROM
        ;

sp_labeled_control:
          label_ident ':'
          {
          }
          sp_unlabeled_control sp_opt_label
          {
          }
        ;

sp_opt_label:
          /* Empty  */  { }
        | label_ident   { }
        ;
sp_unlabeled_block:
          { /* Unlabeled blocks get a secret label. */
          }
          sp_block_content
          {
          }
        ;
sp_labeled_block:
          label_ident ':'
          {
          }
          sp_block_content sp_opt_label
          {
          }
        ;
sp_unlabeled_control:
          LOOP_SYM
          sp_proc_stmts1 END LOOP_SYM
          {
          }
        | WHILE_SYM                     /*$1*/
          {                             /*$2*/
          }
          expr                          /*$3*/
          {                             /*$4*/
          }
          DO_SYM                        /*$10*/
          sp_proc_stmts1                /*$11*/
          END                           /*$12*/
          WHILE_SYM                     /*$13*/
          {                             /*$14*/
          }
        | REPEAT_SYM                    /*$1*/
          sp_proc_stmts1                /*$2*/
          UNTIL_SYM                     /*$3*/
          {                             /*$4*/
          }
          expr                          /*$5*/
          {                             /*$6*/
          }
          END                           /*$7*/
          REPEAT_SYM                    /*$8*/
        ;

trg_action_time:
            BEFORE_SYM
            { }
          | AFTER_SYM
            { }
          ;

trg_event:
            INSERT_SYM
            { }
          | UPDATE_SYM
            { }
          | DELETE_SYM
            { }
          ;
sp_proc_stmt_if:
          IF
          { }
          sp_if END IF
          {
          }
        ;

sp_proc_stmt_statement:
          {
          }
          simple_statement
          {
          }
        ;

sp_proc_stmt_return:
          RETURN_SYM    /*$1*/
          {
          }
          expr          /*$3*/
          {             /*$4*/
          }
        ;

sp_proc_stmt_unlabeled:
          { /* Unlabeled controls get a secret label. */
          }
          sp_unlabeled_control
          {
          }
        ;

sp_proc_stmt_leave:
          LEAVE_SYM label_ident
          {
          }
        ;

sp_proc_stmt_iterate:
          ITERATE_SYM label_ident
          {
          }
        ;

sp_proc_stmt_open:
          OPEN_SYM ident
          {
          }
        ;

sp_proc_stmt_fetch:
          FETCH_SYM sp_opt_fetch_noise ident INTO
          {
          }
          sp_fetch_list
          {
          }
        ;

sp_proc_stmt_close:
          CLOSE_SYM ident
          {
          }
        ;

/**************************************************************************

 DEFINER clause support.

**************************************************************************/

definer_opt:
          no_definer
        | definer
        ;

no_definer:
          /* empty */
          {
          }
        ;

definer:
          DEFINER_SYM EQ user
          {
            ThrowNotSupportedException("DEFINER");
          }
        ;

/**************************************************************************

 CREATE VIEW statement parts.

**************************************************************************/

view_replace_or_algorithm:
          view_replace
          {}
        | view_replace view_algorithm
          {}
        | view_algorithm
          {}
        ;

view_replace:
          OR_SYM REPLACE_SYM
          { ThrowNotSupportedException("OR REPLACE"); }
        ;

view_algorithm:
          ALGORITHM_SYM EQ UNDEFINED_SYM
          { ThrowNotSupportedException("ALGORITHM"); }
        | ALGORITHM_SYM EQ MERGE_SYM
          { ThrowNotSupportedException("ALGORITHM"); }
        | ALGORITHM_SYM EQ TEMPTABLE_SYM
          { ThrowNotSupportedException("ALGORITHM"); }
        ;

view_suid:
          /* empty */
          { }
        | SQL_SYM SECURITY_SYM DEFINER_SYM
          { ThrowNotSupportedException("SECURITY"); }
        | SQL_SYM SECURITY_SYM INVOKER_SYM
          { ThrowNotSupportedException("SECURITY"); }
        ;

view_tail:
          view_suid VIEW_SYM table_ident opt_derived_column_list
          AS view_select
          {
            std::shared_ptr<std::vector<TableElementDescriptionPtr>> columns = nullptr;
            for (const auto& column_name : $4) {
              if (!columns) {
                columns = std::make_shared<std::vector<TableElementDescriptionPtr>>();
              }

              auto desc = std::make_shared<ColumnDescription>();
              desc->column_name = NormalizeIdent(column_name);
              columns->emplace_back(desc);
            }
            $$ = CreateCreateViewStructure( $3, columns, $6 );
          }
        ;

view_select:
          query_expression_or_parens view_check_option
          {
            $$ = $1;
          }
        ;

view_check_option:
          /* empty */                     { }
        | WITH CHECK_SYM OPTION           { ThrowNotSupportedException("WITH"); }
        | WITH CASCADED CHECK_SYM OPTION  { ThrowNotSupportedException("WITH"); }
        | WITH LOCAL_SYM CHECK_SYM OPTION { ThrowNotSupportedException("WITH"); }
        ;

/**************************************************************************

 CREATE TRIGGER statement parts.

**************************************************************************/

trigger_action_order:
            FOLLOWS_SYM
            { }
          | PRECEDES_SYM
            { }
          ;

trigger_follows_precedes_clause:
            /* empty */
            {
            }
          |
            trigger_action_order ident_or_text
            {
            }
          ;

trigger_tail:
          TRIGGER_SYM       /* $1 */
          sp_name           /* $2 */
          trg_action_time   /* $3 */
          trg_event         /* $4 */
          ON_SYM            /* $5 */
          table_ident       /* $6 */
          FOR_SYM           /* $7 */
          EACH_SYM          /* $8 */
          ROW_SYM           /* $9 */
          trigger_follows_precedes_clause /* $10 */
          {                 /* $11 */
          }
          sp_proc_stmt /* $12 */
          { /* $13 */
          }
        ;

/**************************************************************************

 CREATE FUNCTION | PROCEDURE statements parts.

**************************************************************************/

udf_tail:
          AGGREGATE_SYM FUNCTION_SYM ident
          RETURNS_SYM udf_type SONAME_SYM TEXT_STRING_sys
          {
          }
        | FUNCTION_SYM ident
          RETURNS_SYM udf_type SONAME_SYM TEXT_STRING_sys
          {
          }
        ;

sf_tail:
          FUNCTION_SYM /* $1 */
          sp_name /* $2 */
          '(' /* $3 */
          { /* $4 */
          }
          sp_fdparam_list /* $5 */
          ')' /* $6 */
          { /* $7 */
          }
          RETURNS_SYM /* $8 */
          type        /* $9 */
          opt_collate /* $10 */
          { /* $11 */
          }
          sp_c_chistics /* $12 */
          { /* $13 */
          }
          sp_proc_stmt /* $14 */
          {
          }
        ;

sp_tail:
          PROCEDURE_SYM         /*$1*/
          sp_name               /*$2*/
          {                     /*$3*/
          }
          '('                   /*$4*/
          {                     /*$5*/
          }
          sp_pdparam_list       /*$6*/
          ')'                   /*$7*/
          {                     /*$8*/
          }
          sp_c_chistics         /*$9*/
          {                     /*$10*/
          }
          sp_proc_stmt          /*$11*/
          {                     /*$12*/
          }
        ;

/*
  This part of the parser contains common code for all TABLESPACE
  commands.
  CREATE TABLESPACE_SYM name ...
  ALTER TABLESPACE_SYM name ADD DATAFILE ...
  CREATE LOGFILE GROUP_SYM name ...
  ALTER LOGFILE GROUP_SYM name ADD UNDOFILE ..
  DROP TABLESPACE_SYM name
  DROP LOGFILE GROUP_SYM name
*/

opt_ts_datafile_name:
    /* empty */ { }
    | ADD ts_datafile
      {
      }
    ;
opt_undo_tablespace_options:
          /* empty */ { }
        | undo_tablespace_option_list
        ;

undo_tablespace_option_list:
          undo_tablespace_option
          {
          }
        | undo_tablespace_option_list opt_comma undo_tablespace_option
          {
          }
        ;

undo_tablespace_option:
          ts_option_engine
        ;
opt_logfile_group_name:
          /* empty */ { }
        | USE_SYM LOGFILE_SYM GROUP_SYM ident
          {
          }
        ;

opt_tablespace_options:
          /* empty */ { }
        | tablespace_option_list
        ;

tablespace_option_list:
          tablespace_option
          {
          }
        | tablespace_option_list opt_comma tablespace_option
          {
          }
        ;

tablespace_option:
          ts_option_initial_size
        | ts_option_autoextend_size
        | ts_option_max_size
        | ts_option_extent_size
        | ts_option_nodegroup
        | ts_option_engine
        | ts_option_wait
        | ts_option_comment
        | ts_option_file_block_size
    | ts_option_encryption
        ;

ts_datafile:
          DATAFILE_SYM TEXT_STRING_sys { }
        ;

opt_logfile_group_options:
          /* empty */ { }
        | logfile_group_option_list
        ;

logfile_group_option_list:
          logfile_group_option
          {
          }
        | logfile_group_option_list opt_comma logfile_group_option
          {
          }
        ;

logfile_group_option:
          ts_option_initial_size
        | ts_option_undo_buffer_size
        | ts_option_redo_buffer_size
        | ts_option_nodegroup
        | ts_option_engine
        | ts_option_wait
        | ts_option_comment

lg_undofile:
          UNDOFILE_SYM TEXT_STRING_sys { }

ts_option_initial_size:
          INITIAL_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_autoextend_size:
          AUTOEXTEND_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_max_size:
          MAX_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_extent_size:
          EXTENT_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_undo_buffer_size:
          UNDO_BUFFER_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_redo_buffer_size:
          REDO_BUFFER_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_nodegroup:
          NODEGROUP_SYM opt_equal real_ulong_num
          {
          }
        ;

ts_option_comment:
          COMMENT_SYM opt_equal TEXT_STRING_sys
          {
          }
        ;

ts_option_engine:
          opt_storage ENGINE_SYM opt_equal ident_or_text
          {
          }
        ;

ts_option_file_block_size:
          FILE_BLOCK_SIZE_SYM opt_equal size_number
          {
          }
        ;

ts_option_wait:
          WAIT_SYM
          {
          }
        | NO_WAIT_SYM
          {
          }
        ;

ts_option_encryption:
          ENCRYPTION_SYM opt_equal TEXT_STRING_sys
          {
          }
        ;

size_number:
          real_ulonglong_num { }
        | IDENT_sys
          {
          }
        ;

/*
  End tablespace part
*/

/*
  To avoid grammar conflicts, we introduce the next few rules in very details:
  we workaround empty rules for optional AS and DUPLICATE clauses by expanding
  them in place of the caller rule:

  opt_create_table_options_etc ::=
    create_table_options opt_create_partitioning_etc
  | opt_create_partitioning_etc

  opt_create_partitioning_etc ::=
    partitioin [opt_duplicate_as_qe] | [opt_duplicate_as_qe]

  opt_duplicate_as_qe ::=
    duplicate as_create_query_expression
  | as_create_query_expression

  as_create_query_expression ::=
    AS query_expression_or_parens
  | query_expression_or_parens

*/

opt_create_table_options_etc:
          create_table_options
          opt_create_partitioning_etc
          {
              $$ = CreateTableOptions();
              $$.m_partitionStructure = $2;
          }
        | opt_create_partitioning_etc
        {
            $$ = CreateTableOptions();
            $$.m_partitionStructure = $1;
        }
        ;

opt_create_partitioning_etc:
          partition_clause opt_duplicate_as_qe
          {
              $$ = $1;
          }
        | opt_duplicate_as_qe
        ;

opt_duplicate_as_qe:
          /* empty */
          {
          }
        | duplicate
          as_create_query_expression
          {
            ThrowNotSupportedException("duplicate");
          }
        | as_create_query_expression
          {
            ThrowNotSupportedException("create table AS query expression");
          }
        ;

as_create_query_expression:
          AS query_expression_or_parens { }
        | query_expression_or_parens
        ;

/*
 This part of the parser is about handling of the partition information.

 It's first version was written by Mikael Ronström with lots of answers to
 questions provided by Antony Curtis.

 The partition grammar can be called from two places.
 1) CREATE TABLE ... PARTITION ..
 2) ALTER TABLE table_name PARTITION ...
*/
partition_clause:
          PARTITION_SYM BY part_type_def opt_num_parts opt_sub_part
          opt_part_defs
          {
              $$ = std::make_shared< PartitionStructure >();
              $$->m_liner = $3.m_liner;
              $$->m_partMethod = $3.m_method;
              $$->m_partitionExprs = $3.m_partitionExprs;
              $$->m_partitionExprStr = $3.m_exprStr;

              $$->m_partitionCount = $4;
              $$->m_partitionDefList = $6;
          }
        ;

part_type_def:
          opt_linear KEY_SYM opt_key_algo '(' opt_name_list ')'
          {
              ThrowNotSupportedException( "partition method KEY" );
          }
        | opt_linear HASH_SYM '(' bit_expr ')'
          {
              $$ = PartTypeDef();
              $$.m_liner = $1;
              $$.m_method = "HASH";
              $$.m_partitionExprs.emplace_back( $4 );
              $$.m_exprStr = driver.get_string_at_location( @4 );
          }
        | RANGE_SYM '(' bit_expr ')'
          {
              $$ = PartTypeDef();
              $$.m_liner = false;
              $$.m_method = "RANGE";
              $$.m_partitionExprs.emplace_back( $3 );
              $$.m_exprStr = driver.get_string_at_location( @3 );
          }
        | RANGE_SYM COLUMNS '(' name_list ')'
          {
              ThrowNotSupportedException( "partition method 'RANGE COLUMNS'" );
          }
        | LIST_SYM '(' bit_expr ')'
          {
              $$ = PartTypeDef();
              $$.m_liner = false;
              $$.m_method = "LIST";
              $$.m_partitionExprs.emplace_back( $3 );
              $$.m_exprStr = driver.get_string_at_location( @3 );
          }
        | LIST_SYM COLUMNS '(' name_list ')'
          {
              ThrowNotSupportedException( "partition method 'LIST COLUMNS'" );
          }
        ;

opt_linear:
          /* empty */ { $$= false; }
        | LINEAR_SYM  { $$= true; }
        ;

opt_key_algo:
          /* empty */
          { }
        | ALGORITHM_SYM EQ real_ulong_num
          {
          }
        ;

opt_num_parts:
          /* empty */
          { $$= 1; }
        | PARTITIONS_SYM real_ulong_num
          {
            $$= $2;
          }
        ;

opt_sub_part:
          /* empty */ { }
        | SUBPARTITION_SYM BY opt_linear HASH_SYM '(' bit_expr ')'
          opt_num_subparts
          {
          }
        | SUBPARTITION_SYM BY opt_linear KEY_SYM opt_key_algo
          '(' name_list ')' opt_num_subparts
          {
          }
        ;


opt_name_list:
          /* empty */ { }
        | name_list
        ;


name_list:
          ident
          {
          }
        | name_list ',' ident
          {
          }
        ;

opt_num_subparts:
          /* empty */
          { $$= 1; }
        | SUBPARTITIONS_SYM real_ulong_num
          {
            if ($2 == 0)
            {
              my_error(ER_NO_PARTS_ERROR, MYF(0), "subpartitions");
              assert(0);
            }
            $$= $2;
          }
        ;

opt_part_defs:
          /* empty */           { $$ = nullptr; }
        | '(' part_def_list ')' { $$ = $2; }
        ;

part_def_list:
          part_definition
          {
              $$ = std::make_shared< vector< PartDef > >();
              $$->emplace_back( $1 );
          }
        | part_def_list ',' part_definition
          {
              $$ = $1;
              $$->emplace_back( $3 );
          }
        ;

part_definition:
          PARTITION_SYM ident opt_part_values opt_part_options opt_sub_partition
          {
              $$ = PartDef();
              $$.m_partitionName = $2;
              $$.m_partValues = $3;
          }
        ;

opt_part_values:
          /* empty */
          {
              $$ = nullptr;
          }
        | VALUES LESS_SYM THAN_SYM part_func_max
          {
              $$ = std::make_shared< PartValues >();
              $$->m_isRange = true;
              $$->m_valueItemsList.emplace_back( $4 );
          }
        | VALUES IN_SYM part_values_in
          {
              $$ = std::make_shared< PartValues >();
              $$->m_isRange = false;
              $$->m_valueItemsList = $3;
          }
        ;

part_func_max:
          MAX_VALUE_SYM
          {
              $$ = std::make_shared< PartValueItems >();
              $$->m_isMaxValue = true;
          }
        | part_value_item_list_paren { $$ = $1; }
        ;

part_values_in:
          part_value_item_list_paren
          {
              $$ = vector< PartValueItemsSPtr >();
              $$.emplace_back( $1 );
          }
        | '(' part_value_list ')'
          {
              $$ = $2;
          }
        ;

part_value_list:
          part_value_item_list_paren
          {
            $$ = vector< PartValueItemsSPtr >();
            $$.emplace_back( $1 );
          }
        | part_value_list ',' part_value_item_list_paren
          {
              $$ = $1;
              $$.emplace_back( $3 );
          }
        ;

part_value_item_list_paren:
          '(' part_value_item_list ')'
          {
              $$ = $2;
          }
        ;

part_value_item_list:
          part_value_item
          {
              $$ = std::make_shared< PartValueItems >();
              $$->m_valueItems.emplace_back( $1 );
          }
        | part_value_item_list ',' part_value_item
          {
              $$ = $1;
              $$->m_valueItems.emplace_back( $3 );
          }
        ;

part_value_item:
          MAX_VALUE_SYM
          {
              $$.m_isMaxValue = true;
          }
        | bit_expr
          {
             $$.m_expr = $1;
          }
        ;

opt_sub_partition:
          /* empty */           { }
        | '(' sub_part_list ')'
        {
            ThrowNotSupportedException("sub partition");
        }
        ;

sub_part_list:
          sub_part_definition
          {
          }
        | sub_part_list ',' sub_part_definition
          {
          }
        ;

sub_part_definition:
          SUBPARTITION_SYM ident_or_text opt_part_options
          {
          }
        ;

opt_part_options:
         /* empty */ { }
       | part_option_list
       ;

part_option_list:
          part_option_list part_option
          {
          }
        | part_option
          {
          }
        ;

part_option:
          TABLESPACE_SYM opt_equal ident
          { }
        | opt_storage ENGINE_SYM opt_equal ident_or_text
          { }
        | NODEGROUP_SYM opt_equal real_ulong_num
          { }
        | MAX_ROWS opt_equal real_ulonglong_num
          { }
        | MIN_ROWS opt_equal real_ulonglong_num
          { }
        | DATA_SYM DIRECTORY_SYM opt_equal TEXT_STRING_sys
          { }
        | INDEX_SYM DIRECTORY_SYM opt_equal TEXT_STRING_sys
          { }
        | COMMENT_SYM opt_equal TEXT_STRING_sys
          { }
        ;

/*
 End of partition parser part
*/

prepare:
          PREPARE_SYM ident FROM prepare_src
          {
            /*
              We don't know know at this time whether there's a password
              in prepare_src, so we err on the side of caution.  Setting
              the flag will force a rewrite which will obscure all of
              prepare_src in the "Query" log line.  We'll see the actual
              query (with just the passwords obscured, if any) immediately
              afterwards in the "Prepare" log lines anyway, and then again
              in the "Execute" log line if and when prepare_src is executed.
            */
            // lex->contains_plaintext_password= true;
            $$ = CreatePrepareStmt($2, $4);
          }
        ;

prepare_src:
          TEXT_STRING_sys
          {
            $$ = CreatePrepareSrc(false, $1);
          }
        | '@' ident_or_text
          {
            $$ = CreatePrepareSrc(true, $2);
          }
        ;

execute:
          EXECUTE_SYM ident execute_using {
            $$ = CreateExecuteStmt($2, $3);
          }
        ;

execute_using:
          /* nothing */ { }
        | USING execute_var_list {
            $$ = $2;
        }
        ;

execute_var_list:
        execute_var_ident {
            $$.emplace_back(aries_utils::strip_quotes($1));
        }
        |  execute_var_list ',' execute_var_ident {
            $1.emplace_back(aries_utils::strip_quotes($3));
            $$.assign($1.cbegin(), $1.cend());
        }
        ;

execute_var_ident:
          '@' ident_or_text
          {
            $$ = aries_utils::strip_quotes($2);
          }
        ;

deallocate:
          deallocate_or_drop PREPARE_SYM ident
          {
            $$ = CreateDeallocateStmt($3);
          }
        ;

deallocate_or_drop:
          DEALLOCATE_SYM
        | DROP
        ;


/* Common definitions */

text_literal:
        TEXT_STRING
    |   NCHAR_STRING 
    |   UNDERSCORE_CHARSET TEXT_STRING 
    |   text_literal TEXT_STRING_literal 
    ;

text_string:
          TEXT_STRING_literal
        /*
        | HEX_NUM
          {
            ThrowNotSupportedException("HEX_NUM");
          }
        | BIN_NUM
          {
            ThrowNotSupportedException("BIN_NUM");
          }
        */
        ;
param_marker:
        PARAM_MARKER {
            $$ = CreatePreparedStmtParamExpression();
        }
    ;

signed_literal:
          literal
        | '+' NUM_literal {
            $$ = $2;
        }
        | '-' NUM_literal
          {
            $2.str = "-" + $2.str;
            $$ = $2;
          }
        ;
literal:
        text_literal {
          $$ = Literal{LiteralType::STRING, $1}; 
        }
    |   NUM_literal {$$ = $1; }
    |   temporal_literal { ThrowNotSupportedException("literal value format"); }
    |   NULL_SYM {
          $$ = Literal{LiteralType::NULL_LITERAL, "NULL"}; 
        }
    |   FALSE_SYM {
          $$ = Literal{LiteralType::BOOL_FALSE, "FALSE"}; 
        }
    |   TRUE_SYM {
          $$ = Literal{LiteralType::BOOL_TRUE, "TRUE"}; 
        }
    |   HEX_NUM { $$ = Literal{LiteralType::LONG_INT, $1}; }
    |   BIN_NUM { ThrowNotSupportedException("BIN_NUM"); }
    |   UNDERSCORE_CHARSET HEX_NUM { ThrowNotSupportedException("HEX_NUM"); }
    |   UNDERSCORE_CHARSET BIN_NUM { ThrowNotSupportedException("BIN_NUM"); }
    ;

NUM_literal:
        NUM {
          $$ = Literal{LiteralType::INT, $1}; 
        }
    |   LONG_NUM  {
          $$ = Literal{LiteralType::LONG_INT, $1};
        }
    |   ULONGLONG_NUM {
          $$ = Literal{LiteralType::ULONGLONG_INT, $1};
        }
    |   DECIMAL_NUM {
          $$ = Literal{LiteralType::DECIMAL, $1}; 
        }
    |   FLOAT_NUM {
          $$ = Literal{LiteralType::DECIMAL, $1}; 
        }
    |   REAL_NUM {
          ThrowNotSupportedException( "scientific notation" );
        }
    ;


temporal_literal:
        DATE_SYM TEXT_STRING
    |   TIME_SYM TEXT_STRING
    |   TIMESTAMP_SYM TEXT_STRING
    ;


window_func_call:       // Window functions which do not exist as set functions
          ROW_NUMBER_SYM '(' ')' windowing_clause
          
        | RANK_SYM '(' ')' windowing_clause
          
        | DENSE_RANK_SYM '(' ')' windowing_clause
          
        | CUME_DIST_SYM '(' ')' windowing_clause
          
        | PERCENT_RANK_SYM '(' ')' windowing_clause
          
        | NTILE_SYM '(' simple_expr ')' windowing_clause
          
        | LEAD_SYM '(' expr opt_lead_lag_info ')' opt_null_treatment windowing_clause
        | LAG_SYM '(' expr opt_lead_lag_info ')' opt_null_treatment windowing_clause
        | FIRST_VALUE_SYM '(' expr ')' opt_null_treatment windowing_clause
          
        | LAST_VALUE_SYM  '(' expr ')' opt_null_treatment windowing_clause
          
        | NTH_VALUE_SYM '(' expr ',' simple_expr ')' opt_from_first_last opt_null_treatment windowing_clause
        ;

opt_lead_lag_info:
          /* Nothing */
        | ',' NUM_literal opt_ll_default

        | ',' param_marker opt_ll_default
        ;

opt_ll_default:
          /* Nothing */
          
        | ',' expr
          
        ;

opt_null_treatment:
          /* Nothing */
          
        | RESPECT_SYM NULLS_SYM
          
        | IGNORE_SYM NULLS_SYM
          
        ;


opt_from_first_last:
          /* Nothing */
          
        | FROM FIRST_SYM
          
        | FROM LAST_SYM
          
        ;

opt_windowing_clause:
          /* Nothing */
        | windowing_clause
        ;

windowing_clause:
          OVER_SYM window_name_or_spec { ThrowNotSupportedException( "OVER" ); }
          
        ;

window_name_or_spec:
          window_name
          
        | window_spec
          
        ;

window_name:
          ident
          
        ;

window_spec:
          '(' window_spec_details ')'
          
        ;

window_spec_details:
           opt_existing_window_name
           opt_partition_clause
           opt_window_order_by_clause
           opt_window_frame_clause
         ;

opt_existing_window_name:
          /* Nothing */
          
        | window_name
          
        ;

opt_partition_clause:
          /* Nothing */
          
        | PARTITION_SYM BY group_list
          
        ;

opt_window_order_by_clause:
          /* Nothing */
          
        | ORDER_SYM BY order_list
          
        ;

opt_window_frame_clause:
          /* Nothing*/
          
        | window_frame_units
          window_frame_extent
          opt_window_frame_exclusion
          
        ;

window_frame_extent:
          window_frame_start
        | window_frame_between
          
        ;

window_frame_start:
          UNBOUNDED_SYM PRECEDING_SYM
          
        | NUM_literal PRECEDING_SYM
          
        | param_marker PRECEDING_SYM
          
        | INTERVAL_SYM expr interval PRECEDING_SYM
          
        | CURRENT_SYM ROW_SYM
          
        ;

window_frame_between:
          BETWEEN_SYM window_frame_bound AND_SYM window_frame_bound
          
        ;

window_frame_bound:
          window_frame_start
          
        | UNBOUNDED_SYM FOLLOWING_SYM
          
        | NUM_literal FOLLOWING_SYM
          
        | param_marker FOLLOWING_SYM
          
        | INTERVAL_SYM expr interval FOLLOWING_SYM
          
        ;

opt_window_frame_exclusion:
          /* Nothing */
          
        | EXCLUDE_SYM CURRENT_SYM ROW_SYM
          
        | EXCLUDE_SYM GROUP_SYM
          
        | EXCLUDE_SYM TIES_SYM
          
        | EXCLUDE_SYM NO_SYM OTHERS_SYM
        ;

window_frame_units:
          ROWS_SYM    
        | RANGE_SYM   
        | GROUPS_SYM  
        ;

equal:
          EQ
        | SET_VAR
        ;

opt_equal:
          /* empty */
        | equal
        ;

variable:
        '@' variable_aux {
          auto variableStructurePtr = $2;
          $$ = CreateExpressionFromVariable(variableStructurePtr);
        }
        ;

variable_aux:
          ident_or_text SET_VAR expr { ThrowNotSupportedException("user defined variable"); }
        | ident_or_text {
            auto variableStructurePtr = CreateUserVariableStructure($1);
            $$ = variableStructurePtr;
        }
        | '@' opt_var_ident_type ident_or_text opt_component {
            auto variableStructurePtr = CreateSysVariableStructure($2, $3);
            $$ = variableStructurePtr;
        }
        ;

string_list:
          text_string
          {
          }
        | string_list ',' text_string
          {
          }
        ;

references:
          REFERENCES
          table_ident
          '(' key_list ')'
          opt_match_clause
          opt_on_update_delete
          {
            $$.first = $2;
            $$.second.assign( $4.cbegin(), $4.cend() );
          }
        ;

opt_ref_list:
          /* empty */      { }
        | '(' reference_list ')' { }
        ;

reference_list:
          reference_list ',' ident_expr
          {
            $$.emplace_back( $3 );
          }
        | ident_expr
          {
            $$.emplace_back( $1 );
          }
        ;

opt_match_clause:
          /* empty */      {  }
        | MATCH FULL       { }
        | MATCH PARTIAL    { }
        | MATCH SIMPLE_SYM { }
        ;

opt_on_update_delete:
          /* empty */
          {
          }
        | ON_SYM UPDATE_SYM delete_option
          {
          }
        | ON_SYM DELETE_SYM delete_option
          {
          }
        | ON_SYM UPDATE_SYM delete_option
          ON_SYM DELETE_SYM delete_option
          {
          }
        | ON_SYM DELETE_SYM delete_option
          ON_SYM UPDATE_SYM delete_option
          {
          }
        ;

delete_option:
          RESTRICT      {  }
        | CASCADE       { }
        | SET NULL_SYM  { }
        | NO_SYM ACTION { }
        | SET DEFAULT_SYM { }
        ;
constraint_key_type:
          PRIMARY_SYM KEY_SYM { $$= KEYTYPE_PRIMARY; }
        | UNIQUE_SYM opt_key_or_index 
          {
              $$= KEYTYPE_UNIQUE;
          };
key_or_index:
          KEY_SYM {}
        | INDEX_SYM { ThrowNotSupportedException( "INDEX" ); }
        ;

opt_key_or_index:
          /* empty */ {}
        | key_or_index
        ;

keys_or_index:
          KEYS {}
        | INDEX_SYM { ThrowNotSupportedException( "INDEX" ); }
        | INDEXES { ThrowNotSupportedException( "INDEXES" ); }
        ;

opt_fulltext_index_options:
          /* Empty. */ { }
        | fulltext_index_options
        ;

fulltext_index_options:
          fulltext_index_option
          {
          }
        | fulltext_index_options fulltext_index_option
          {
          }
        ;

fulltext_index_option:
          common_index_option
        | WITH PARSER_SYM IDENT_sys
          {
          }
        ;

opt_spatial_index_options:
          /* Empty. */ { }
        | spatial_index_options
        ;

spatial_index_options:
          spatial_index_option
          {
          }
        | spatial_index_options spatial_index_option
          {
          }
        ;

spatial_index_option:
          common_index_option
        ;

opt_index_options:
          /* Empty. */ { }
        | index_options
        ;

index_options:
          index_option
          {
          }
        | index_options index_option
          {
          }
        ;

index_option:
          common_index_option { ThrowNotSupportedException( "index options" ); }
        | index_type_clause { ThrowNotSupportedException( "index type" ); }
        ;

// These options are common for all index types.
common_index_option:
          KEY_BLOCK_SIZE opt_equal ulong_num { }
        | COMMENT_SYM TEXT_STRING_sys
          {
          }
        | visibility
          {
          }
        ;

/*
  The syntax for defining an index is:

    ... INDEX [index_name] [USING|TYPE] <index_type> ...

  The problem is that whereas USING is a reserved word, TYPE is not. We can
  still handle it if an index name is supplied, i.e.:

    ... INDEX type TYPE <index_type> ...

  here the index's name is unmbiguously 'type', but for this:

    ... INDEX TYPE <index_type> ...

  it's impossible to know what this actually mean - is 'type' the name or the
  type? For this reason we accept the TYPE syntax only if a name is supplied.
*/
opt_index_name_and_type:
          opt_ident                  { $$ = $1; }
        | opt_ident USING index_type { $$ = $1; }
        // | ident TYPE_SYM index_type  { }
        ;

index_type_clause:
          USING index_type    { }
        | TYPE_SYM index_type { }
        ;

visibility:
          VISIBLE_SYM { $$= true; }
        | INVISIBLE_SYM { $$= false; }
        ;

index_type:
          BTREE_SYM {  }
        | RTREE_SYM {  }
        | HASH_SYM  { }
        ;

key_list:
          key_list ',' key_part
          {
            $1.emplace_back( $3 );
            $$ = $1;
          }
        | key_part
          {
             $$.emplace_back( $1 );
          }
        ;

key_part:
          ident opt_ordering_direction
          {
            $$ = aries_utils::convert_to_lower( aries_utils::strip_quotes( $1 ) );
          }
        | ident '(' NUM ')' opt_ordering_direction
          {
            $$ = aries_utils::convert_to_lower( aries_utils::strip_quotes( $1 ) );
          }
        ;

key_list_with_expression:
          key_list_with_expression ',' key_part_with_expression
          {
            $1.emplace_back( $3 );
            $$ = $1;
          }
        | key_part_with_expression
          {
            $$.emplace_back( $1 );
            // The order is ignored.
          }
        ;

key_part_with_expression:
          key_part
        | '(' expr ')' opt_ordering_direction
          {
            ThrowNotSupportedException( "only support key with column" );
          }
        ;
opt_ident:
          /* empty */ { $$= ""; }
        | ident { $$ = $1; }
        ;
opt_component:
          /* empty */    { $$= ""; }
        | '.' ident      { $$= $2; }
        ;

charset_name:
          ident_or_text
          {
            $$ = $1;
          }
        | BINARY_SYM { $$ = "binary"; }
        ;

old_or_new_charset_name:
          ident_or_text
          {
          }
        | BINARY_SYM { }
        ;

old_or_new_charset_name_or_default:
          old_or_new_charset_name { }
        | DEFAULT_SYM    { }
        ;

collation_name:
          ident_or_text
          {
          }
        ;

opt_collate:
          /* empty */                { }
        | COLLATE_SYM collation_name { }
        ;

ascii:
          ASCII_SYM        { }
        | BINARY_SYM ASCII_SYM { }
        | ASCII_SYM BINARY_SYM { }
        ;
unicode:
          UNICODE_SYM
          {
          }
        | UNICODE_SYM BINARY_SYM
          {
          }
        | BINARY_SYM UNICODE_SYM
          {
          }
        ;
opt_charset_with_opt_binary:
          /* empty */
          {
          }
        | ascii
          {
          }
        | unicode
          {
          }
        | BYTE_SYM
          {
          }
        | character_set charset_name opt_bin_mod
          {
          }
        | BINARY_SYM
          {
          }
        | BINARY_SYM character_set charset_name
          {
          }
        ;

opt_bin_mod:
          /* empty */ { }
        | BINARY_SYM  { }
        ;
opt_default:
          /* empty */ {}
        | DEFAULT_SYM {}
        ;

// Remainder of the option value list after first option value.
option_value_list_continued:
          /* empty */           { $$= NULL; }
        | ',' option_value_list { $$= $2; }
        ;

// Start of option value list, option_type was given
start_option_value_list_following_option_type:
          option_value_following_option_type option_value_list_continued
          {
            auto tmpSetStructurePtr = CreateSetSysVarStructure(OPT_SESSION, $1);
            $$ = AppendOptionSetStructures(tmpSetStructurePtr, $2);
          }
        | TRANSACTION_SYM transaction_characteristics
          {
            auto setStructures = std::make_shared<std::vector<SetStructurePtr>>();
            auto setStructurePtr = std::make_shared<SetStructure>();
            setStructurePtr->m_setCmd = SET_CMD::SET_TX;
            setStructures->emplace_back(setStructurePtr);
            $$ = setStructures;
          }
        ;

// Repeating list of option values after first option value.
option_value_list:
          option_value
          {
            auto setStructures = std::make_shared<std::vector<SetStructurePtr>>();
            setStructures->emplace_back($1);
            $$ = setStructures;
          }
        | option_value_list ',' option_value
          {
            $1->emplace_back($3);
            $$ = $1;
          }
        ;

option_value:
          option_type option_value_following_option_type
          {
            $$ = CreateSetSysVarStructure($1, $2);
          }
        | option_value_no_option_type { $$= $1; }
        ;

option_type:
          GLOBAL_SYM  { $$=OPT_GLOBAL; }
        // | PERSIST_SYM { $$=OPT_PERSIST; }
        // | PERSIST_ONLY_SYM { $$=OPT_PERSIST_ONLY; }
        | LOCAL_SYM   { $$=OPT_SESSION; }
        | SESSION_SYM { $$=OPT_SESSION; }
        ;
opt_var_type:
          /* empty */ { $$=OPT_SESSION; }
        | GLOBAL_SYM  { $$=OPT_GLOBAL; }
        | LOCAL_SYM   { $$=OPT_SESSION; }
        | SESSION_SYM { $$=OPT_SESSION; }
        ;

opt_var_ident_type:
          /* empty */     { $$=OPT_DEFAULT; }
        | GLOBAL_SYM '.'  { $$=OPT_GLOBAL; }
        | LOCAL_SYM '.'   { $$=OPT_SESSION; }
        | SESSION_SYM '.' { $$=OPT_SESSION; }
        ;

opt_set_var_ident_type:
          /* empty */     { $$=OPT_DEFAULT; }
        // | PERSIST_SYM '.' { $$=OPT_PERSIST; }
        // | PERSIST_ONLY_SYM '.' {$$=OPT_PERSIST_ONLY; }
        | GLOBAL_SYM '.'  { $$=OPT_GLOBAL; }
        | LOCAL_SYM '.'   { $$=OPT_SESSION; }
        | SESSION_SYM '.' { $$=OPT_SESSION; }
         ;

// Option values with preceding option_type.
option_value_following_option_type:
          internal_variable_name equal set_expr_or_default
          {
            auto optionValuePtr = std::make_shared<PT_option_value_following_option_type>();
            optionValuePtr->varName = aries_utils::strip_quotes($1);
            optionValuePtr->expression = $3;
            $$ = optionValuePtr;
          }
        ;

// Option values without preceding option_type.
option_value_no_option_type:
          internal_variable_name        /*$1*/
          equal                         /*$2*/
          set_expr_or_default           /*$3*/
          {
            $$ = CreateSetSysVarStructure(false, $1, aries_utils::strip_quotes($1), $3);
          }
        | '@' ident_or_text equal expr
          {
            $$ = CreateSetUserVarStructure($2, $4);
          }
        | '@' '@' opt_set_var_ident_type internal_variable_name equal
          set_expr_or_default
          {
            $$ = CreateSetSysVarStructure($3, $4, aries_utils::strip_quotes($4), $6);
          }
        | character_set old_or_new_charset_name_or_default
          {
            auto setStructurePtr = std::make_shared<SetStructure>();
            setStructurePtr->m_setCmd = SET_CMD::SET_CHAR_SET;
            $$ = setStructurePtr;
          }
        | NAMES_SYM equal expr
          {
            /*
              Bad syntax, always fails with an error
            */
            ThrowNotSupportedException("set names");
          }
        | NAMES_SYM charset_name opt_collate
          {
            auto setStructurePtr = std::make_shared<SetStructure>();
            setStructurePtr->m_setCmd = SET_CMD::SET_NAMES;
            $$ = setStructurePtr;
          }
        | NAMES_SYM DEFAULT_SYM
          {
            auto setStructurePtr = std::make_shared<SetStructure>();
            setStructurePtr->m_setCmd = SET_CMD::SET_NAMES;
            $$ = setStructurePtr;
          }
        ;

internal_variable_name:
          lvalue_ident
        /*
        | lvalue_ident '.' ident
          {
            $$= NEW_PTN PT_internal_variable_name_2d(@$, $1, $3);
          }
        | DEFAULT_SYM '.' ident
          {
            $$= NEW_PTN PT_internal_variable_name_default($3);
          }
        */
        ;

set_expr_or_default:
          expr
        | DEFAULT_SYM { $$= nullptr; }
        | ON_SYM
          {
            $$= CreateStringExpression( "on" );
          }
        | ALL
          {
            $$= CreateStringExpression( "all" );
          }
        | BINARY_SYM
          {
            $$= CreateStringExpression( "binary" );
          }
        | ROW_SYM
          {
            $$= CreateStringExpression( "row" );
          }
        | SYSTEM_SYM
          {
            $$= CreateStringExpression( "system" );
          }
        ;

cast_type:
          BINARY_SYM opt_field_length { ThrowNotSupportedException("cast as binary"); }
        | CHAR_SYM opt_field_length {
          $$.value_type = BiaodashiValueType::TEXT;
          $$.length = $2;
        }
        | nchar opt_field_length { ThrowNotSupportedException("cast as nchar"); }
        | SIGNED_SYM {
          $$.value_type = BiaodashiValueType::INT;
        }
        | SIGNED_SYM INT_SYM {
          $$.value_type = BiaodashiValueType::INT;
        }
        | INT_SYM {
          $$.value_type = BiaodashiValueType::INT;
        }
        | BIGINT_SYM {
          $$.value_type = BiaodashiValueType::LONG_INT;
        }
        | UNSIGNED_SYM { ThrowNotSupportedException("cast as unsigned"); }
        | UNSIGNED_SYM INT_SYM { ThrowNotSupportedException("cast as unsigned int"); }
        | DATE_SYM {
          $$.value_type = BiaodashiValueType::DATE;
        }
        | TIME_SYM type_datetime_precision {
          $$.value_type = BiaodashiValueType::TIME;
        }
        | DATETIME_SYM type_datetime_precision {
          $$.value_type = BiaodashiValueType::DATE_TIME;
        }
        | DECIMAL_SYM float_options { 
          $$.value_type = BiaodashiValueType::DECIMAL;
          $$.length = $2->first;
          $$.associated_length = $2->second;

        }
        /*
        as of 5.7.26, MySQL does not support cast to float and double
        | FLOAT_SYM opt_field_length {
          $$.value_type = BiaodashiValueType::FLOAT;
        }
        | DOUBLE_SYM {
          $$.value_type = BiaodashiValueType::DOUBLE;
        }
        */
        | JSON_SYM { ThrowNotSupportedException("cast as json"); }
        ;

in_sum_expr:
          opt_all expr { $$ = $2; }
        ;

opt_expr_list:
          /* empty */ { $$ = std::vector<Expression>(); }
        | expr_list { $$ = $1; }
        ;


expr_list_in:
          expr {
            driver.global_expr_list.emplace_back($1);

          }
        | expr_list_in ',' expr
          {
            driver.global_expr_list.emplace_back($3);
          }
        ;

expr_list:
          expr {
            $$ = std::vector<Expression>();
            $$.emplace_back($1);
          }
        | expr_list ',' expr
          {
            $$ = $1;
            $$.emplace_back($3);
          }
        ;


ident_list_arg:
          ident_list          
        | '(' ident_list ')'  
        ;

ident_list:
          simple_ident
        | ident_list ',' simple_ident
        ;

opt_expr:
          /* empty */ { $$ = nullptr; }
        | expr           
        ;

opt_else:
          /* empty */ { $$ = nullptr; }
        | ELSE expr { $$ = $2; }  
        ;

when_list:
          WHEN_SYM expr THEN_SYM expr {
            auto tuple_value = std::make_tuple($2, $4);
            std::vector<std::tuple<Expression, Expression>> list;
            list.emplace_back(tuple_value);
            $$ = list;
          }
        | when_list WHEN_SYM expr THEN_SYM expr {
          auto tuple_value = std::make_tuple($3, $5);
          $1.emplace_back(tuple_value);
          $$ = $1;
        }
        ;

table_reference:
          table_factor  { $$ = $1; }
        | joined_table  { $$ = $1; }
        ;

/*
  Join operations are normally left-associative, as in

    t1 JOIN t2 ON t1.a = t2.a JOIN t3 ON t3.a = t2.a

  This is equivalent to

    (t1 JOIN t2 ON t1.a = t2.a) JOIN t3 ON t3.a = t2.a

  They can also be right-associative without parentheses, e.g.

    t1 JOIN t2 JOIN t3 ON t2.a = t3.a ON t1.a = t2.a

  Which is equivalent to

    t1 JOIN (t2 JOIN t3 ON t2.a = t3.a) ON t1.a = t2.a

  In MySQL, JOIN and CROSS JOIN mean the same thing, i.e.:

  - A join without a <join specification> is the same as a cross join.
  - A cross join with a <join specification> is the same as an inner join.

  For the join operation above, this means that the parser can't know until it
  has seen the last ON whether `t1 JOIN t2` was a cross join or not. The only
  way to solve the abiguity is to keep shifting the tokens on the stack, and
  not reduce until the last ON is seen. We tell Bison this by adding a fake
  token CONDITIONLESS_JOIN which has lower precedence than all tokens that
  would continue the join. These are JOIN_SYM, INNER_SYM, CROSS,
  STRAIGHT_JOIN, NATURAL, LEFT, RIGHT, ON and USING. This way the automaton
  only reduces to a cross join unless no other interpretation is
  possible. This gives a right-deep join tree for join *with* conditions,
  which is what is expected.

  The challenge here is that t1 JOIN t2 *could* have been a cross join, we
  just don't know it until afterwards. So if the query had been

    t1 JOIN t2 JOIN t3 ON t2.a = t3.a

  we will first reduce `t2 JOIN t3 ON t2.a = t3.a` to a <table_reference>,
  which is correct, but a problem arises when reducing t1 JOIN
  <table_reference>. If we were to do that, we'd get a right-deep tree. The
  solution is to build the tree downwards instead of upwards, as is normally
  done. This concept may seem outlandish at first, but it's really quite
  simple. When the semantic action for table_reference JOIN table_reference is
  executed, the parse tree is (please pardon the ASCII graphic):

                       JOIN ON t2.a = t3.a
                      /    \
                     t2    t3

  Now, normally we'd just add the cross join node on top of this tree, as:

                    JOIN
                   /    \
                 t1    JOIN ON t2.a = t3.a
                      /    \
                     t2    t3

  This is not the meaning of the query, however. The cross join should be
  addded at the bottom:


                       JOIN ON t2.a = t3.a
                      /    \
                    JOIN    t3
                   /    \
                  t1    t2

  There is only one rule to pay attention to: If the right-hand side of a
  cross join is a join tree, find its left-most leaf (which is a table
  name). Then replace this table name with a cross join of the left-hand side
  of the top cross join, and the right hand side with the original table.

  Natural joins are also syntactically conditionless, but we need to make sure
  that they are never right associative. We handle them in their own rule
  natural_join, which is left-associative only. In this case we know that
  there is no join condition to wait for, so we can reduce immediately.
*/
joined_table:
          table_reference inner_join_type table_reference ON_SYM expr {
            auto join_structure = $1;

            auto right = $3;
            join_structure->AddJoinRel($2, right, $5);

            // for (int i=0; i < right->GetJoinCount(); i++) {
            //   auto rel = right->GetJoinRel(i);
            //   auto type = right->GetJoinType(i);
            //   auto expr = right->GetJoinExpr(i);

            //   join_structure->AddJoinRel(type, rel, expr);
            // }

            $$ = join_structure;
          }
          
        | table_reference inner_join_type table_reference USING
          '(' using_list ')'
          {
            ThrowNotSupportedException("USING");
          }
          
        | table_reference outer_join_type table_reference ON_SYM expr {
            auto join_structure = $1;

            auto right = $3;
            join_structure->AddJoinRel($2, right, $5);

            // for (int i=0; i < right->GetJoinCount(); i++) {
            //   auto rel = right->GetJoinRel(i);
            //   auto type = right->GetJoinType(i);
            //   auto expr = right->GetJoinExpr(i);

            //   join_structure->AddJoinRel(type, rel, expr);
            // }

            $$ = join_structure;
          }
          
        | table_reference outer_join_type table_reference USING '(' using_list ')'
          {
            ThrowNotSupportedException("USING");
          }
          
        | table_reference inner_join_type table_reference
          %prec CONDITIONLESS_JOIN
          {
            auto join_structure = $1;
            auto right = $3;
            join_structure->AddJoinRel( $2, right, CreateBoolExpression( true ) );
            $$ = join_structure;
          }
        | table_reference natural_join_type table_factor
          {
            ThrowNotSupportedException("conditionless join");
          }
        ;

natural_join_type:
          NATURAL opt_inner JOIN_SYM   { $$ = JoinType::InnerJoin; }
        | NATURAL RIGHT opt_outer JOIN_SYM   { $$ = JoinType::RightJoin; }
        | NATURAL LEFT opt_outer JOIN_SYM   { $$ = JoinType::LeftJoin; }
        ;

inner_join_type:
          JOIN_SYM { $$ = JoinType::InnerJoin; }               
        | INNER_SYM JOIN_SYM  { $$ = JoinType::InnerJoin; }
        | CROSS JOIN_SYM  { $$ = JoinType::InnerJoin; }   
        | STRAIGHT_JOIN  { $$ = JoinType::InnerJoin; }
        ;

outer_join_type:

        FULL opt_outer JOIN_SYM { $$ = JoinType::FullJoin;  debug_line();  }
        | LEFT opt_outer JOIN_SYM  { $$ = JoinType::LeftJoin;  debug_line(); }   
        | RIGHT opt_outer JOIN_SYM   { $$ = JoinType::RightJoin;  debug_line();  }
        ;

opt_inner:
          /* empty */
        | INNER_SYM
        ;

opt_outer:
          /* empty */
        | OUTER
        ;

/*
  table PARTITION (list of partitions), reusing using_list instead of creating
  a new rule for partition_list.
*/
opt_use_partition:
          /* empty */ 
        | use_partition
        {
          ThrowNotSupportedException( "table partition" );
        }
        ;

use_partition:
          PARTITION_SYM '(' using_list ')'
        ;

/**
  MySQL has a syntax extension where a comma-separated list of table
  references is allowed as a table reference in itself, for instance

    SELECT * FROM (t1, t2) JOIN t3 ON 1

  which is not allowed in standard SQL. The syntax is equivalent to

    SELECT * FROM (t1 CROSS JOIN t2) JOIN t3 ON 1

  We call this rule table_reference_list_parens.

  A <table_factor> may be a <single_table>, a <subquery>, a <derived_table>, a
  <joined_table>, or the bespoke <table_reference_list_parens>, each of those
  enclosed in any number of parentheses. This makes for an ambiguous grammar
  since a <table_factor> may also be enclosed in parentheses. We get around
  this by designing the grammar so that a <table_factor> does not have
  parentheses, but all the sub-cases of it have their own parentheses-rules,
  i.e. <single_table_parens>, <joined_table_parens> and
  <table_reference_list_parens>. It's a bit tedious but the grammar is
  unambiguous and doesn't have shift/reduce conflicts.
*/
table_factor:
          single_table { $$ = $1; }
        | single_table_parens { $$ = $1; }
        | derived_table { $$ = $1; } 
        | joined_table_parens { $$ = $1; } 
          
        | table_reference_list_parens { ThrowNotSupportedException( "table list in parens" ); }
          
        ;

table_reference_list_parens:
          '(' table_reference_list_parens ')' 
        | '(' table_reference_list ',' table_reference ')'
          {
          }
        ;

single_table_parens:
          '(' single_table_parens ')'  { $$ = $2; }
        | '(' single_table ')'   { $$ = $2; }
        ;

single_table:
          table_ident opt_use_partition opt_table_alias opt_key_definition {
            auto join_structure = std::make_shared<JoinStructure>();
            if (!$3.empty()) {
              $1->ResetAlias($3);
            }
            join_structure->SetLeadingRel($1);
            $$ = join_structure;
          }
          
        ;

joined_table_parens:
          '(' joined_table_parens ')' { $$ = $2; }
        | '(' joined_table ')' { $$ = $2; }
        ;

derived_table:
          table_subquery opt_table_alias opt_derived_column_list
          {
            std::vector< ColumnDescriptionPointer > columns;
            for ( const auto& column_name : $3 ) {
              auto desc = std::make_shared< ColumnDescription >();
              desc->column_name = NormalizeIdent( column_name );
              columns.emplace_back( desc );
            }
            $$ = CreateDerivedTableJoinStructure( $1, $2, columns );
          }
        | LATERAL_SYM table_subquery opt_table_alias opt_derived_column_list
          {
            ThrowNotSupportedException("Lateral Derived Tables");
          }
        ;


opt_index_hints_list:
          /* empty */
        ;

opt_key_definition:
          opt_index_hints_list
        ;

using_list:
          ident_string_list
        ;

ident_string_list:
          ident
        | ident_string_list ',' ident
        ;

interval:
          interval_time_stamp    { $$ = $1; }
        | TEXT_STRING { $$ = $1; }
        | DAY_HOUR_SYM { ThrowNotSupportedException( "DAY_HOUR" ); }            
        | DAY_MICROSECOND_SYM { ThrowNotSupportedException( "DAY_MICROSECOND" ); }          
        | DAY_MINUTE_SYM { ThrowNotSupportedException( "DAY_MINUTE" ); }               
        | DAY_SECOND_SYM { ThrowNotSupportedException( "DAY_SECOND" ); }                
        | HOUR_MICROSECOND_SYM { ThrowNotSupportedException( "HOUR_MICROSECOND" ); }         
        | HOUR_MINUTE_SYM { ThrowNotSupportedException( "HOUR_MINUTE" ); }              
        | HOUR_SECOND_SYM { ThrowNotSupportedException( "HOUR_SECOND" ); }              
        | MINUTE_MICROSECOND_SYM { ThrowNotSupportedException( "MINUTE_MICROSECOND" ); }        
        | MINUTE_SECOND_SYM { ThrowNotSupportedException( "MINUTE_SECOND" ); }            
        | SECOND_MICROSECOND_SYM { ThrowNotSupportedException( "SECOND_MICROSECOND" ); }       
        | YEAR_MONTH_SYM { ThrowNotSupportedException( "YEAR_MONTH" ); }               
        ;

interval_time_stamp:
          DAY_SYM 
        | WEEK_SYM        
        | HOUR_SYM        
        | MINUTE_SYM       
        | MONTH_SYM        
        | QUARTER_SYM      
        | SECOND_SYM       
        | MICROSECOND_SYM  
        | YEAR_SYM         
        ;

date_time_type:
          DATE_SYM  
        | TIME_SYM  
        | TIMESTAMP_SYM 
        | DATETIME_SYM  
        ;

opt_as:
          /* empty */
        | AS
        ;

opt_table_alias:
          /* empty */  { $$ = string(); }
        | opt_as ident { $$ = $2; debug_line(); }
        ;

opt_all:
          /* empty */
        | ALL
        ;

opt_where_clause:
        opt_where_clause_expr { $$ = $1; }
        ;

opt_where_clause_expr: /* empty */  { $$ = nullptr; }
        | WHERE expr { $$ = $2; }
          
        ;

opt_having_clause:
          /* empty */ { $$ = nullptr; }
        | HAVING expr {
          $$ = $2;
        }
          
        ;

with_clause:
          WITH with_list { ThrowNotSupportedException( "WITH" ); }
          
        | WITH RECURSIVE_SYM with_list { ThrowNotSupportedException( "WITH" ); }
          
        ;

with_list:
          with_list ',' common_table_expr
        | common_table_expr
        ;

common_table_expr:
          ident opt_derived_column_list AS table_subquery
        ;

opt_derived_column_list:
          /* empty */ {
            $$ = std::vector<string>();
          }
        | '(' simple_ident_list ')'
        {
          $$ = $2;
        }
          
        ;

simple_ident_list:
          ident {
            $$ = std::vector<string>();
            $$.emplace_back(aries_parser::NormalizeIdent($1));
          }
        | simple_ident_list ',' ident
        {
          $$ = $1;
          $$.emplace_back(aries_parser::NormalizeIdent($3));
        }
        ;

opt_window_clause:
          /* Nothing */
          
        | WINDOW_SYM window_definition_list
          { 
            ThrowNotSupportedException( "WINDOW" ); 
          }
        ;

window_definition_list:
          window_definition
        | window_definition_list ',' window_definition
        ;

window_definition:
          window_name AS window_spec
        ;

opt_escape:
          ESCAPE_SYM simple_expr { ThrowNotSupportedException( "ESCAPE" ); }
        | /* empty */            
        ;

/*
   group by statement in select
*/

opt_group_clause:
          /* empty */  { }
        | GROUP_SYM BY group_list olap_opt {
          auto groupby_structure_pointer = std::make_shared<GroupbyStructure>();
          groupby_structure_pointer->SetGroupbyExprs($3);
          $$ = groupby_structure_pointer;
          debug_line();
        }
          
        ;

group_list:
          group_list ',' grouping_expr {
            $1.emplace_back($3);
            $$ = $1;
          }
        | grouping_expr {
          std::vector<Expression> group_by_items;
          group_by_items.emplace_back($1);
          $$ = group_by_items;
        }
        ;


olap_opt:
          /* empty */   
        | WITH_ROLLUP_SYM { ThrowNotSupportedException( "WITH_ROLLUP" ); }
            /*
              'WITH ROLLUP' is needed for backward compatibility,
              and cause LALR(2) conflicts.
              This syntax is not standard.
              MySQL syntax: GROUP BY col1, col2, col3 WITH ROLLUP
              SQL-2003: GROUP BY ... ROLLUP(col1, col2, col3)
            */
        ;

/*
  Order by statement in ALTER TABLE
*/

/*
   Order by statement in select
*/

opt_order_clause:
          /* empty */  { $$ = nullptr; }
        | order_clause { $$ = $1; }
        ;

order_clause:
          ORDER_SYM BY order_list {
            auto order_by_expression = std::make_shared<OrderbyStructure>();
            for (const auto& item : $3) {
              order_by_expression->AddOrderbyItem(item.order_expr, item.direction);
            }
            $$ = order_by_expression;
          }
          
        ;

order_list:
          order_list ',' order_expr {
            $1.emplace_back($3);
            $$ = $1;
          }
        | order_expr {
          std::vector<OrderItem> list;
          list.emplace_back($1);
          $$ = list;
        }
        ;

opt_ordering_direction:
          /* empty */  { $$ = "ASC"; }
        | ordering_direction { $$ = $1; }
        ;

ordering_direction:
          ASC
        | DESC
        ;

opt_limit_clause:
          /* empty */ { $$ = nullptr; }
        | limit_clause { $$ = $1; }
        ;

limit_clause:
          LIMIT limit_options { $$ = $2; }
          
        ;

limit_options:
          limit_option {
            int64_t offset = 0;
            int64_t size = std::stoll($1);
            $$ = std::make_shared<LimitStructure>(offset, size);
          }
        | limit_option ',' limit_option {
            int64_t offset = std::stoll($1);
            int64_t size = std::stoll($3);
            $$ = std::make_shared<LimitStructure>(offset, size);
          }
        | limit_option OFFSET_SYM limit_option {
            int64_t offset = std::stoll($1);
            int64_t size = std::stoll($3);
            $$ = std::make_shared<LimitStructure>(offset, size);
          }
        ;

limit_option:
          ident 
          
        | param_marker 
          
        | ULONGLONG_NUM 
          
        | LONG_NUM 
          
        | NUM 
          
        ;


/**********************************************************************
** Creating different items.
**********************************************************************/

insert_ident:
          simple_ident_nospvar
        | table_wild
        ;
table_wild:
          ident '.' '*' {
            $$ = std::make_shared<CommonBiaodashi>(BiaodashiType::Star, aries_utils::to_lower($1));
          }
          
        | ident '.' ident '.' '*' { ThrowNotSupportedException( $1 + "." + $3 + ".*" ); }
          
        ;

order_expr:
          expr opt_ordering_direction {
            OrderItem order_item;
            order_item.order_expr = $1;
            order_item.direction = ($2 == "desc") ? OrderbyDirection::DESC : OrderbyDirection::ASC;
            $$ = order_item;
          }
          
        ;

grouping_expr:
          expr { $$ = $1; }
          
        ;

ident_expr:
          simple_ident {
            $$ = CreateIdentExpression( $1 );
          };

simple_ident:
          ident { $$ = std::make_shared<SQLIdent>("", "", aries_utils::strip_quotes($1)); }
          
        | simple_ident_q
        ;

simple_ident_nospvar:
        ident {
            $$ = std::make_shared<SQLIdent>("",
                                            "",
                                            aries_utils::strip_quotes($1));
        }
        | simple_ident_q
        ;

simple_ident_q:
          ident '.' ident {
            $$ = std::make_shared<SQLIdent>("",
                                            aries_utils::strip_quotes($1),
                                            aries_utils::strip_quotes($3));
          }
          
        | ident '.' ident '.' ident {
          $$ = std::make_shared<SQLIdent>(aries_utils::strip_quotes($1),
                                          aries_utils::strip_quotes($3),
                                          aries_utils::strip_quotes($5));
        }
        ;

table_ident:
          ident {
            $$ = CreateTableIdent(false, "", $1, nullptr, nullptr);
          }
        | ident '.' ident {
            $$ = CreateTableIdent(false, $1, $3, nullptr, nullptr);
          }
        ;

IDENT_sys:
          IDENT
        ;

TEXT_STRING_sys:
          TEXT_STRING { $$= $1; }
        ;

TEXT_STRING_literal:
          TEXT_STRING
        ;

TEXT_STRING_filesystem:
          TEXT_STRING
          {
          /*
            THD *thd= YYTHD;

            if (thd->charset_is_character_set_filesystem)
              $$= $1;
            else
            {
              if (thd->convert_string(&$$,
                                      thd->variables.character_set_filesystem,
                                      $1.str, $1.length, thd->charset()))
                MYSQL_YYABORT;
            }
            */
            $$ = $1;
          }
        ;

TEXT_STRING_password:
          TEXT_STRING
        ;

TEXT_STRING_hash:
          TEXT_STRING_sys
        ;

ident:
          IDENT_sys
        | ident_keyword
        ;

ident_or_text:
          ident
        | TEXT_STRING_sys
        | LEX_HOSTNAME
        ;

field_def:
          type opt_column_attribute_list
          {
            $$ = CreateFieldDef($1, $2);
          }
        | type opt_collate opt_generated_always // $3
          AS '(' expr ')' // $7
          opt_stored_attribute opt_column_attribute_list
          {
            $$ = CreateFieldDef($1, $9);
          }
        ;

opt_generated_always:
          /* empty */
        | GENERATED ALWAYS_SYM { ThrowNotSupportedException( "GENERATED ALWAYS" ); }
        ;

opt_stored_attribute:
          /* empty */ { }
        | VIRTUAL_SYM { ThrowNotSupportedException( "VIRTUAL" ); }
        | STORED_SYM  { ThrowNotSupportedException( "STORED" ); }
        ;
type:
          int_type opt_field_length field_options
          {
            $$ = CreateColumnType($1, $2, $3);
          }
        | real_type opt_precision field_options
          {
            $$ = CreateColumnType($1, *($2), $3);
          }
        | numeric_type float_options field_options
          {
            $$ = CreateColumnType($1, *($2), $3);
          }
        | BIT_SYM
          {
            ThrowNotSupportedException("datatype BIT");
          }
        | BIT_SYM field_length
          {
            ThrowNotSupportedException("datatype BIT");
          }
        | BOOL_SYM
          {
            $$ = CreateColumnType("bool");
          }
        | BOOLEAN_SYM
          {
            $$ = CreateColumnType("bool");
          }
        | CHAR_SYM field_length opt_charset_with_opt_binary
          {
            $$ = CreateColumnType("char", $2);
          }
        | CHAR_SYM opt_charset_with_opt_binary
          {
            $$ = CreateColumnType("char");
          }
        | nchar field_length opt_bin_mod
          {
            ThrowNotSupportedException("datatype NCHAR");
          }
        | nchar opt_bin_mod
          {
            ThrowNotSupportedException("datatype NCHAR");
          }
        | BINARY_SYM field_length
          {
             $$ = CreateColumnType("binary", $2);
          }
        | BINARY_SYM
          {
            ThrowNotSupportedException("datatype BINARY");
          }
        | varchar field_length opt_charset_with_opt_binary
          {
            $$ = CreateColumnType($1, $2);
          }
        | nvarchar field_length opt_bin_mod
          {
            ThrowNotSupportedException("datatype NVARCHAR");
          }
        | VARBINARY_SYM field_length
          {
            $$ = CreateColumnType("varbinary", $2);
          }
        | YEAR_SYM opt_field_length field_options
          {
            $$ = CreateColumnType("year", $2, $3);
          }
        | DATE_SYM
          {
            $$ = CreateColumnType("date");
          }
        | TIME_SYM type_datetime_precision
          {
            $$ = CreateColumnType("time", $2);
          }
        | TIMESTAMP_SYM type_datetime_precision
          {
            $$ = CreateColumnType("timestamp", $2);
          }
        | DATETIME_SYM type_datetime_precision
          {
            $$ = CreateColumnType("datetime", $2);
          }
        | TINYBLOB_SYM
          {
            ThrowNotSupportedException("datatype TINYBLOB");
          }
        | BLOB_SYM opt_field_length
          {
            ThrowNotSupportedException("datatype BLOB");
          }
        | spatial_type { }
        | MEDIUMBLOB_SYM
          {
            ThrowNotSupportedException("datatype MEDIUMBLOB");
          }
        | LONGBLOB_SYM
          {
            ThrowNotSupportedException("datatype LONGBLOB");
          }
        | LONG_SYM VARBINARY_SYM
          {
            ThrowNotSupportedException("datatype LONG VARBINARY");
          }
        | LONG_SYM varchar opt_charset_with_opt_binary
          {
            ThrowNotSupportedException("datatype LONG VARCHAR");
          }
        | TINYTEXT_SYN opt_charset_with_opt_binary
          {
            // $$ = CreateColumnType("text", "256");
            $$ = CreateColumnType("text", "128");
          }
        | TEXT_SYM opt_field_length opt_charset_with_opt_binary
          {
            auto fieldLen = $2;
            if ( fieldLen.empty() )
            {
              // fieldLen = "65535";
              fieldLen = "256";
            }
            $$ = CreateColumnType("text", fieldLen);
          }
        | MEDIUMTEXT_SYM opt_charset_with_opt_binary
          {
            // fieldLen = "16777215";
            auto fieldLen = "1024";
            $$ = CreateColumnType("text", fieldLen);
          }
        | LONGTEXT_SYM opt_charset_with_opt_binary
          {
            // fieldLen = "4294967295";
            auto fieldLen = "1024";
            $$ = CreateColumnType("text", fieldLen);
          }
        | ENUM_SYM '(' string_list ')' opt_charset_with_opt_binary
          {
            ThrowNotSupportedException("datatype ENUM");
          }
        | SET '(' string_list ')' opt_charset_with_opt_binary
          {
            ThrowNotSupportedException("datatype SET");
          }
        | LONG_SYM opt_charset_with_opt_binary
          {
            ThrowNotSupportedException("datatype LONGBLOB");
          }
        | SERIAL_SYM
          {
            ThrowNotSupportedException("datatype SERIAL");
          }
        | JSON_SYM
          {
            ThrowNotSupportedException("datatype JSON");
          }
        ;

spatial_type:
          GEOMETRY_SYM
          { ThrowNotSupportedException("geometry");}
        | GEOMETRYCOLLECTION_SYM
          { ThrowNotSupportedException("geometrycollection");}
        | POINT_SYM
          { ThrowNotSupportedException("point");}
        | MULTIPOINT_SYM
          { ThrowNotSupportedException("multipoint");}
        | LINESTRING_SYM
          { ThrowNotSupportedException("linestring");}
        | MULTILINESTRING_SYM
          { ThrowNotSupportedException("multilinestring");}
        | POLYGON_SYM
          { ThrowNotSupportedException("polygon");}
        | MULTIPOLYGON_SYM
          { ThrowNotSupportedException("multipolygon");}
        ;

nchar:
     NCHAR_SYM {
        ThrowNotSupportedException("nchar");
     }
    | NATIONAL_SYM CHAR_SYM {
        ThrowNotSupportedException("national char");
    }
    ;
varchar:
          CHAR_SYM VARYING { $$ = "varchar"; }
        | VARCHAR_SYM { $$ = "varchar"; }
        ;
nvarchar:
          NATIONAL_SYM VARCHAR_SYM {ThrowNotSupportedException("nvarchar");}
        | NVARCHAR_SYM {ThrowNotSupportedException("nvarchar");}
        | NCHAR_SYM VARCHAR_SYM {ThrowNotSupportedException("nvarchar");}
        | NATIONAL_SYM CHAR_SYM VARYING {ThrowNotSupportedException("nvarchar");}
        | NCHAR_SYM VARYING {ThrowNotSupportedException("nvarchar");}
        ;

int_type:
          INT_SYM       { $$ = "int"; }
        | INTEGER_SYM   { $$ = "int"; }
        | TINYINT_SYM   { $$ = "tinyint"; }
        | SMALLINT_SYM  { $$ = "smallint"; }
        | MEDIUMINT_SYM { $$ = "mediumint"; }
        | BIGINT_SYM    { $$ = "bigint"; }
        ;

real_type:
          REAL_SYM
          {
            $$ = "real";
          }
        | DOUBLE_SYM opt_PRECISION
          { $$ = "double"; }
        ;
opt_PRECISION:
          /* empty */
        | PRECISION
        ;

numeric_type:
          FLOAT_SYM   { $$ = "float"; }
        | DECIMAL_SYM { $$ = "decimal"; }
        | NUMERIC_SYM { $$ = "numeric"; }
        | FIXED_SYM   { $$ = "fixed"; }
        ;

float_options:
        /* empty */ {
            $$ = std::make_shared< pair< string, string > >("", "");
        }
    | field_length {
            $$ = std::make_shared< pair< string, string > >($1, "");
    }
    | precision
    ;

precision:
        '(' NUM ',' NUM ')' {
            $$ = std::make_shared< pair< string, string > >($2, $4);
        }
    ;

type_datetime_precision:
        /* empty */ { $$ = ""; }
    | '(' NUM ')' { $$ = $2; }
    ;

func_datetime_precision:
        /* empty */ { $$ = nullptr; }
    | '(' ')'  { $$ = nullptr; }
    | '(' NUM ')'  { $$ = CreateIntegerExpression($2); }
    ;



field_length:
          '(' LONG_NUM ')'  { $$ = $2; }
        | '(' ULONGLONG_NUM ')' { $$ = $2; }
        | '(' DECIMAL_NUM ')' { $$ = $2; }
        | '(' NUM ')' { $$ = $2; }

opt_field_length:
          /* empty */ { $$ = ""; }
        | field_length
        ;
opt_precision:
          /* empty */
          {
            $$ = std::make_shared< pair< string, string > >("", "");
          }
        | precision
        ;
opt_column_attribute_list:
          /* empty */ { $$ = nullptr; }
        | column_attribute_list
        ;

column_attribute_list:
          column_attribute_list column_attribute
          {
            $$ = $1;
            if ($2 == nullptr) assert(0); // OOM
            /*
            if ($2->has_constraint_enforcement()) {
              // $2 is `[NOT] ENFORCED`
              if ($1->back()->set_constraint_enforcement(
                      $2->is_constraint_enforced())) {
                // $1 is not `CHECK(...)`
                YYTHD->syntax_error_at(@2);
                MYSQL_YYABORT;
              }
            } else {
            */
              $$->emplace_back($2);
            // }
          }
        | column_attribute
          {
            if ($1 == nullptr)
                assert(0); // OOM
            // if ($1->has_constraint_enforcement()) {
            //   // [NOT] ENFORCED doesn't follow the CHECK clause
            //   YYTHD->syntax_error_at(@1);
            //   assert(0);
            // }
            auto colAttrList = std::make_shared<std::vector<PT_column_attr_base_ptr>>();
            colAttrList->emplace_back($1);
            $$ = colAttrList;
          }
        ;

column_attribute:
          NULL_SYM
          {
            $$ = std::make_shared<PT_null_column_attr>();
          }
        | not NULL_SYM
          {
            $$ = std::make_shared<PT_not_null_column_attr>();
          }
        | not SECONDARY_SYM
          {
            ThrowNotSupportedException("SECONDARY");
            //$$ = std::make_shared<PT_secondary_column_attr >();
          }
        | DEFAULT_SYM now_or_signed_literal
          {
            $$ = CreateDefaultValueAttr( $2 );
          }
        | DEFAULT_SYM '(' expr ')'
          {
            ThrowNotSupportedException("expression as default value");
          }
        | ON_SYM UPDATE_SYM now
          {
            ThrowNotSupportedException("ON UPDATE");
            //$$ = std::make_shared<PT_on_update_column_attr >();
          }
        | AUTO_INC
          {
            ThrowNotSupportedException("AUTO_INC");
            //$$ = std::make_shared<PT_auto_increment_column_attr >();
          }
        | SERIAL_SYM DEFAULT_SYM VALUE_SYM
          {
            ThrowNotSupportedException("SERIAL");
            //$$ = std::make_shared<PT_serial_default_value_column_attr >();
          }
        | opt_primary KEY_SYM
          {
            $$ = std::make_shared<PT_primary_key_column_attr >();;
          }
        | UNIQUE_SYM
          {
            $$ = std::make_shared<PT_unique_key_column_attr >();
          }
        | UNIQUE_SYM KEY_SYM
          {
            $$ = std::make_shared<PT_unique_key_column_attr >();
          }
        | COMMENT_SYM TEXT_STRING_sys
          {
            $$ = std::make_shared<PT_comment_column_attr >($2);
          }
        | COLLATE_SYM collation_name
          {
            $$ = std::make_shared<PT_collate_column_attr >();
          }
        | COLUMN_FORMAT_SYM column_format
          {
            ThrowNotSupportedException("COLUMN_FORMAT");
            //$$ = std::make_shared<PT_column_format_column_attr >();
          }
        | STORAGE_SYM storage_media
          {
            ThrowNotSupportedException("STORAGE");
            //$$ = std::make_shared<PT_storage_media_column_attr >();
          }
        | SRID_SYM real_ulonglong_num
          {
            ThrowNotSupportedException("SRID");
            //$$ = std::make_shared<PT_srid_column_attr>();
          }
        | opt_constraint_name check_constraint
          /* See the next branch for [NOT] ENFORCED. */
          {
            //$$ = std::make_shared<PT_check_constraint_column_attr>();
          }
        | constraint_enforcement
          /*
            This branch is needed to workaround the need of a lookahead of 2 for
            the grammar:

             { [NOT] NULL | CHECK(...) [NOT] ENFORCED } ...

            Note: the column_attribute_list rule rejects all unexpected
                  [NOT] ENFORCED sequences.
          */
          {
            //$$ = std::make_shared<PT_constraint_enforcement_attr>();
          }
          | ENCODING encode_type AS ident
          {
            $$ = std::make_shared<PT_encode_type_column_attr>( $2.first, $2.second, $4 );
          }
        ;

encode_type:
        BYTEDICT_SYM
        { 
          $$.first = aries::EncodeType::DICT; 
          $$.second = "tinyint";
        }
        | SHORTDICT_SYM
        {
          $$.first = aries::EncodeType::DICT;
          $$.second = "smallint";
        }
        | INTDICT_SYM
        {
          $$.first = aries::EncodeType::DICT;
          $$.second = "int";
        }
        ;

column_format:
          DEFAULT_SYM { }
        | FIXED_SYM   { }
        | DYNAMIC_SYM { }
        ;

storage_media:
          DEFAULT_SYM { }
        | DISK_SYM    { }
        | MEMORY_SYM  { }
        ;
now_or_signed_literal:
          now
        | signed_literal {
            $$ = LiteralToExpression( $1 );
        }
        ;
now:
        NOW_SYM func_datetime_precision {
          $$ = CreateFunctionExpression("NOW", $2);
        }
    ;

/*
  Non-reserved keywords are allowed as unquoted identifiers in general.

  OTOH, in a few particular cases statement-specific rules are used
  instead of `ident_keyword` to avoid grammar ambiguities:

    * `label_keyword` for SP label names
    * `role_keyword` for role names
    * `lvalue_keyword` for variable prefixes and names in left sides of
                       assignments in SET statements

  Normally, new non-reserved words should be added to the
  the rule `ident_keywords_unambiguous`. If they cause grammar conflicts, try
  one of `ident_keywords_ambiguous_...` rules instead.
*/
ident_keyword:
          ident_keywords_unambiguous
        | ident_keywords_ambiguous_1_roles_and_labels
        | ident_keywords_ambiguous_2_labels
        | ident_keywords_ambiguous_3_roles
        | ident_keywords_ambiguous_4_system_variables
        ;

/*
  These non-reserved words cannot be used as role names and SP label names:
*/
ident_keywords_ambiguous_1_roles_and_labels:
          EXECUTE_SYM
        | RESTART_SYM
        | SHUTDOWN
        ;

/*
  These non-reserved keywords cannot be used as unquoted SP label names:
*/
ident_keywords_ambiguous_2_labels:
          ASCII_SYM
        | BEGIN_SYM
        | BYTE_SYM
        | CACHE_SYM
        | CHARSET
        | CHECKSUM_SYM
        | CLONE_SYM
        | COMMENT_SYM
        | COMMIT_SYM
        | CONTAINS_SYM
        | DEALLOCATE_SYM
        | DO_SYM
        | END
        | FLUSH_SYM
        | FOLLOWS_SYM
        | HANDLER_SYM
        | HELP_SYM
        | IMPORT
        | INSTALL_SYM
        | LANGUAGE_SYM
        | NO_SYM
        | PRECEDES_SYM
        | PREPARE_SYM
        | REPAIR
        | RESET_SYM
        | ROLLBACK_SYM
        | SAVEPOINT_SYM
        | SIGNED_SYM
        | SLAVE
        | START_SYM
        | STOP_SYM
        | TRUNCATE_SYM
        | UNICODE_SYM
        | UNINSTALL_SYM
        | XA_SYM
        ;

/*
  Keywords that we allow for labels in SPs in the unquoted form.
  Any keyword that is allowed to begin a statement or routine characteristics
  must be in `ident_keywords_ambiguous_2_labels` above, otherwise
  we get (harmful) shift/reduce conflicts.

  Not allowed:

    ident_keywords_ambiguous_1_roles_and_labels
    ident_keywords_ambiguous_2_labels
*/
label_keyword:
          ident_keywords_unambiguous
        | ident_keywords_ambiguous_3_roles
        | ident_keywords_ambiguous_4_system_variables
        ;

/*
  These non-reserved keywords cannot be used as unquoted role names:
*/
ident_keywords_ambiguous_3_roles:
          EVENT_SYM
        | FILE_SYM
        | NONE_SYM
        | PROCESS
        | PROXY_SYM
        | RELOAD
        | REPLICATION
        | RESOURCE_SYM
        | SUPER_SYM
        ;

/*
  These are the non-reserved keywords which may be used for unquoted
  identifiers everywhere without introducing grammar conflicts:
*/
ident_keywords_unambiguous:
          ACTION
        | ACCOUNT_SYM
        | ACTIVE_SYM
        | ADDDATE_SYM
        | ADMIN_SYM
        | AFTER_SYM
        | AGAINST
        | AGGREGATE_SYM
        | ALGORITHM_SYM
        | ALWAYS_SYM
        | ANY_SYM
        | AT_SYM
        | AUTOEXTEND_SIZE_SYM
        | AUTO_INC
        | AVG_ROW_LENGTH
        | AVG_SYM
        | BACKUP_SYM
        | BINLOG_SYM
        | BIT_SYM
        | BLOCK_SYM
        | BOOLEAN_SYM
        | BOOL_SYM
        | BTREE_SYM
        | BUCKETS_SYM
        | CASCADED
        | CATALOG_NAME_SYM
        | CHAIN_SYM
        | CHANGED
        | CHANNEL_SYM
        | CIPHER_SYM
        | CLASS_ORIGIN_SYM
        | CLIENT_SYM
        | CLOSE_SYM
        | COALESCE
        | CODE_SYM
        | COLLATION_SYM
        | COLUMNS
        | COLUMN_FORMAT_SYM
        | COLUMN_NAME_SYM
        | COMMITTED_SYM
        | COMPACT_SYM
        | COMPLETION_SYM
        | COMPONENT_SYM
        | COMPRESSED_SYM
        | COMPRESSION_SYM
        | CONCURRENT
        | CONNECTION_SYM
        | CONSISTENT_SYM
        | CONSTRAINT_CATALOG_SYM
        | CONSTRAINT_NAME_SYM
        | CONSTRAINT_SCHEMA_SYM
        | CONTEXT_SYM
        | CPU_SYM
        | CURRENT_SYM /* not reserved in MySQL per WL#2111 specification */
        | CURSOR_NAME_SYM
        | DATAFILE_SYM
        | DATA_SYM
        | DATETIME_SYM
        | DATE_SYM
        | DAY_SYM
        | DEFAULT_AUTH_SYM
        | DEFINER_SYM
        | DEFINITION_SYM
        | DELAY_KEY_WRITE_SYM
        | DESCRIPTION_SYM
        | DIAGNOSTICS_SYM
        | DIRECTORY_SYM
        | DISABLE_SYM
        | DISCARD_SYM
        | DISK_SYM
        | DUMPFILE
        | DUPLICATE_SYM
        | DYNAMIC_SYM
        | ENABLE_SYM
        | ENCRYPTION_SYM
        | ENDS_SYM
        | ENFORCED_SYM
        | ENGINES_SYM
        | ENGINE_SYM
        | ENUM_SYM
        | ERRORS
        | ERROR_SYM
        | ESCAPE_SYM
        | EVENTS_SYM
        | EVERY_SYM
        | EXCHANGE_SYM
        | EXCLUDE_SYM
        | EXPANSION_SYM
        | EXPIRE_SYM
        | EXPORT_SYM
        | EXTENDED_SYM
        | EXTENT_SIZE_SYM
        | FAST_SYM
        | FAULTS_SYM
        | FILE_BLOCK_SIZE_SYM
        | FILTER_SYM
        | FIRST_SYM
        | FIXED_SYM
        | FOLLOWING_SYM
        | FORMAT_SYM
        | FOUND_SYM
        // | FULL
        | GENERAL
        | GEOMETRYCOLLECTION_SYM
        | GEOMETRY_SYM
        | GET_FORMAT
        | GET_MASTER_PUBLIC_KEY_SYM
        | GRANTS
        | GROUP_REPLICATION
        | HASH_SYM
        | HISTOGRAM_SYM
        | HISTORY_SYM
        | HOSTS_SYM
        | HOST_SYM
        | HOUR_SYM
        | IDENTIFIED_SYM
        | IGNORE_SERVER_IDS_SYM
        | INACTIVE_SYM
        | INDEXES
        | INITIAL_SIZE_SYM
        | INSERT_METHOD
        | INSTANCE_SYM
        | INVISIBLE_SYM
        | INVOKER_SYM
        | IO_SYM
        | IPC_SYM
        | ISOLATION
        | ISSUER_SYM
        | JSON_SYM
        | KEY_BLOCK_SIZE
        | LAST_SYM
        | LEAVES
        | LESS_SYM
        | LEVEL_SYM
        | LINESTRING_SYM
        | LIST_SYM
        | LOCKED_SYM
        | LOCKS_SYM
        | LOGFILE_SYM
        | LOGS_SYM
        | MASTER_AUTO_POSITION_SYM
        | MASTER_CONNECT_RETRY_SYM
        | MASTER_DELAY_SYM
        | MASTER_HEARTBEAT_PERIOD_SYM
        | MASTER_HOST_SYM
        | NETWORK_NAMESPACE_SYM
        | MASTER_LOG_FILE_SYM
        | MASTER_LOG_POS_SYM
        | MASTER_PASSWORD_SYM
        | MASTER_PORT_SYM
        | MASTER_PUBLIC_KEY_PATH_SYM
        | MASTER_RETRY_COUNT_SYM
        | MASTER_SERVER_ID_SYM
        | MASTER_SSL_CAPATH_SYM
        | MASTER_SSL_CA_SYM
        | MASTER_SSL_CERT_SYM
        | MASTER_SSL_CIPHER_SYM
        | MASTER_SSL_CRLPATH_SYM
        | MASTER_SSL_CRL_SYM
        | MASTER_SSL_KEY_SYM
        | MASTER_SSL_SYM
        | MASTER_SYM
        | MASTER_TLS_VERSION_SYM
        | MASTER_USER_SYM
        | MAX_CONNECTIONS_PER_HOUR
        | MAX_QUERIES_PER_HOUR
        | MAX_ROWS
        | MAX_SIZE_SYM
        | MAX_UPDATES_PER_HOUR
        | MAX_USER_CONNECTIONS_SYM
        | MEDIUM_SYM
        | MEMORY_SYM
        | MERGE_SYM
        | MESSAGE_TEXT_SYM
        | MICROSECOND_SYM
        | MIGRATE_SYM
        | MINUTE_SYM
        | MIN_ROWS
        | MODE_SYM
        | MODIFY_SYM
        | MONTH_SYM
        | MULTILINESTRING_SYM
        | MULTIPOINT_SYM
        | MULTIPOLYGON_SYM
        | MUTEX_SYM
        | MYSQL_ERRNO_SYM
        | NAMES_SYM
        | NAME_SYM
        | NATIONAL_SYM
        | NCHAR_SYM
        | NDBCLUSTER_SYM
        | NESTED_SYM
        | NEVER_SYM
        | NEW_SYM
        | NEXT_SYM
        | NODEGROUP_SYM
        | NOWAIT_SYM
        | NO_WAIT_SYM
        | NULLS_SYM
        | NUMBER_SYM
        | NVARCHAR_SYM
        | OFFSET_SYM
        | OJ_SYM
        | OLD_SYM
        | ONE_SYM
        | ONLY_SYM
        | OPEN_SYM
        | OPTIONAL_SYM
        | OPTIONS_SYM
        | ORDINALITY_SYM
        | ORGANIZATION_SYM
        | OTHERS_SYM
        | OWNER_SYM
        | PACK_KEYS_SYM
        | PAGE_SYM
        | PARSER_SYM
        | PARTIAL
        | PARTITIONING_SYM
        | PARTITIONS_SYM
        | PASSWORD
        | PATH_SYM
        | PHASE_SYM
        | PLUGINS_SYM
        | PLUGIN_DIR_SYM
        | PLUGIN_SYM
        | POINT_SYM
        | POLYGON_SYM
        | PORT_SYM
        | PRECEDING_SYM
        | PRESERVE_SYM
        | PREV_SYM
        | PRIVILEGES
        | PROCESSLIST_SYM
        | PROFILES_SYM
        | PROFILE_SYM
        | QUARTER_SYM
        | QUERY_SYM
        | QUICK
        | READ_ONLY_SYM
        | REBUILD_SYM
        | RECOVER_SYM
        | REDO_BUFFER_SIZE_SYM
        | REDUNDANT_SYM
        | REFERENCE_SYM
        | RELAY
        | RELAYLOG_SYM
        | RELAY_LOG_FILE_SYM
        | RELAY_LOG_POS_SYM
        | RELAY_THREAD
        | REMOVE_SYM
        | REORGANIZE_SYM
        | REPEATABLE_SYM
        | REPLICATE_DO_DB
        | REPLICATE_DO_TABLE
        | REPLICATE_IGNORE_DB
        | REPLICATE_IGNORE_TABLE
        | REPLICATE_REWRITE_DB
        | REPLICATE_WILD_DO_TABLE
        | REPLICATE_WILD_IGNORE_TABLE
        | RESOURCES
        | RESPECT_SYM
        | RESTORE_SYM
        | RESUME_SYM
        | RETAIN_SYM
        | RETURNED_SQLSTATE_SYM
        | RETURNS_SYM
        | REUSE_SYM
        | REVERSE_SYM
        | ROLE_SYM
        | ROLLUP_SYM
        | ROTATE_SYM
        | ROUTINE_SYM
        | ROW_COUNT_SYM
        | ROW_FORMAT_SYM
        | RTREE_SYM
        | SCHEDULE_SYM
        | SCHEMA_NAME_SYM
        | SECONDARY_ENGINE_SYM
        | SECONDARY_LOAD_SYM
        | SECONDARY_SYM
        | SECONDARY_UNLOAD_SYM
        | SECOND_SYM
        | SECURITY_SYM
        | SERIALIZABLE_SYM
        | SERIAL_SYM
        | SERVER_SYM
        | SHARE_SYM
        | SIMPLE_SYM
        | SKIP_SYM
        | SLOW
        | SNAPSHOT_SYM
        | SOCKET_SYM
        | SONAME_SYM
        | SOUNDS_SYM
        | SOURCE_SYM
        | SQL_AFTER_GTIDS
        | SQL_AFTER_MTS_GAPS
        | SQL_BEFORE_GTIDS
        | SQL_BUFFER_RESULT
        | SQL_NO_CACHE_SYM
        | SQL_THREAD
        | SRID_SYM
        | STACKED_SYM
        | STARTS_SYM
        | STATS_AUTO_RECALC_SYM
        | STATS_PERSISTENT_SYM
        | STATS_SAMPLE_PAGES_SYM
        | STATUS_SYM
        | STORAGE_SYM
        | STRING_SYM
        | SUBCLASS_ORIGIN_SYM
        | SUBDATE_SYM
        | SUBJECT_SYM
        | SUBPARTITIONS_SYM
        | SUBPARTITION_SYM
        | SUSPEND_SYM
        | SWAPS_SYM
        | SWITCHES_SYM
        | TABLES
        | TABLESPACE_SYM
        | TABLE_CHECKSUM_SYM
        | TABLE_NAME_SYM
        | TEMPORARY
        | TEMPTABLE_SYM
        | TEXT_SYM
        | THAN_SYM
        | THREAD_PRIORITY_SYM
        | TIES_SYM
        | TIMESTAMP_ADD
        | TIMESTAMP_DIFF
        | TIMESTAMP_SYM
        | TIME_SYM
        | TRANSACTION_SYM
        | TRIGGERS_SYM
        | TYPES_SYM
        | TYPE_SYM
        | UNBOUNDED_SYM
        | UNCOMMITTED_SYM
        | UNDEFINED_SYM
        | UNDOFILE_SYM
        | UNDO_BUFFER_SIZE_SYM
        | UNKNOWN_SYM
        | UNTIL_SYM
        | UPGRADE_SYM
        | USER
        | USE_FRM
        | VALIDATION_SYM
        | VALUE_SYM
        | VARIABLES
        | VCPU_SYM
        | VIEW_SYM
        | VIEWS
        | VISIBLE_SYM
        | WAIT_SYM
        | WARNINGS
        | WEEK_SYM
        | WEIGHT_STRING_SYM
        | WITHOUT_SYM
        | WORK_SYM
        | WRAPPER_SYM
        | X509_SYM
        | XID_SYM
        | XML_SYM
        | YEAR_SYM
        | TREE_SYM
        | TRADITIONAL_SYM
        ;

/*
  Non-reserved keywords that we allow for unquoted role names:

  Not allowed:

    ident_keywords_ambiguous_1_roles_and_labels
    ident_keywords_ambiguous_3_roles
*/
role_keyword:
          ident_keywords_unambiguous
        | ident_keywords_ambiguous_2_labels
        | ident_keywords_ambiguous_4_system_variables
        ;

/*
  Non-reserved words allowed for unquoted unprefixed variable names and
  unquoted variable prefixes in the left side of assignments in SET statements:

  Not allowed:

    ident_keywords_ambiguous_4_system_variables
*/
lvalue_keyword:
          ident_keywords_unambiguous
        | ident_keywords_ambiguous_1_roles_and_labels
        | ident_keywords_ambiguous_2_labels
        | ident_keywords_ambiguous_3_roles
        ;

/*
  These non-reserved keywords cannot be used as unquoted unprefixed
  variable names and unquoted variable prefixes in the left side of
  assignments in SET statements:
*/
ident_keywords_ambiguous_4_system_variables:
          GLOBAL_SYM
        | LOCAL_SYM
        | PERSIST_SYM
        | PERSIST_ONLY_SYM
        | SESSION_SYM
        ;

TEXT_STRING_sys_nonewline:
          TEXT_STRING_sys
          {
          }
        ;

opt_channel:
         /*empty */
       {
       }
     | FOR_SYM CHANNEL_SYM TEXT_STRING_sys_nonewline
       {
         /*
           channel names are case insensitive. This means, even the results
           displayed to the user are converted to lower cases.
           system_charset_info is utf8_general_ci as required by channel name
           restrictions
         */
       }
    ;

opt_profile_defs:
  /* empty */
  | profile_defs;

profile_defs:
  profile_def
  | profile_defs ',' profile_def;

profile_def:
  CPU_SYM
    {
    }
  | MEMORY_SYM
    {
    }
  | BLOCK_SYM IO_SYM
    {
    }
  | CONTEXT_SYM SWITCHES_SYM
    {
    }
  | PAGE_SYM FAULTS_SYM
    {
    }
  | IPC_SYM
    {
    }
  | SWAPS_SYM
    {
    }
  | SOURCE_SYM
    {
    }
  | ALL
    {
    }
  ;

opt_profile_args:
  /* empty */
    {
    }
  | FOR_SYM QUERY_SYM NUM
    {
    }
  ;

character_set:
          CHAR_SYM SET
        | CHARSET
        ;

server_options_list:
          server_option
        | server_options_list ',' server_option
        ;

server_option:
          USER TEXT_STRING_sys
          {
          }
        | HOST_SYM TEXT_STRING_sys
          {
          }
        | DATABASE TEXT_STRING_sys
          {
          }
        | OWNER_SYM TEXT_STRING_sys
          {
          }
        | PASSWORD TEXT_STRING_sys
          {
          }
        | SOCKET_SYM TEXT_STRING_sys
          {
          }
        | PORT_SYM ulong_num
          {
          }
        ;

event_tail:
          EVENT_SYM opt_if_not_exists sp_name
          {
          }
          ON_SYM SCHEDULE_SYM ev_schedule_time
          opt_ev_on_completion
          opt_ev_status
          opt_ev_comment
          DO_SYM ev_sql_stmt
          {
          }
        ;

ev_schedule_time:
          EVERY_SYM expr interval
          {
          }
          ev_starts
          ev_ends
        | AT_SYM expr
          {
          }
        ;

opt_ev_status:
          /* empty */ { }
        | ENABLE_SYM
          {
          }
        | DISABLE_SYM ON_SYM SLAVE
          {
          }
        | DISABLE_SYM
          {
          }
        ;

ev_starts:
          /* empty */
          {
          }
        | STARTS_SYM expr
          {
          }
        ;

ev_ends:
          /* empty */
        | ENDS_SYM expr
          {
          }
        ;

opt_ev_on_completion:
          /* empty */ { }
        | ev_on_completion
        ;

ev_on_completion:
          ON_SYM COMPLETION_SYM PRESERVE_SYM
          {
          }
        | ON_SYM COMPLETION_SYM NOT_SYM PRESERVE_SYM
          {
          }
        ;

opt_ev_comment:
          /* empty */ { }
        | COMMENT_SYM TEXT_STRING_sys
          {
          }
        ;

ev_sql_stmt:
          {
          }
          ev_sql_stmt_inner
          {
          }
        ;

ev_sql_stmt_inner:
          sp_proc_stmt_statement{ ThrowNotSupportedException("proc statement"); }
        | sp_proc_stmt_return { ThrowNotSupportedException("RETURN"); }
        | sp_proc_stmt_if { ThrowNotSupportedException("IF"); }
        | case_stmt_specification { ThrowNotSupportedException("CASE"); }
        | sp_labeled_block{ ThrowNotSupportedException("labeled block"); }
        | sp_unlabeled_block{ ThrowNotSupportedException("unlabeled block"); }
        | sp_labeled_control{ ThrowNotSupportedException("label"); }
        | sp_proc_stmt_unlabeled{ ThrowNotSupportedException("unlabeled statements"); }
        | sp_proc_stmt_leave{ ThrowNotSupportedException("LEAVE"); }
        | sp_proc_stmt_iterate{ ThrowNotSupportedException("ITERATE"); }
        | sp_proc_stmt_open{ ThrowNotSupportedException("OPEN"); }
        | sp_proc_stmt_fetch { ThrowNotSupportedException("FETCH"); }
        | sp_proc_stmt_close{ ThrowNotSupportedException("CLOSE"); }
        ;

sp_name:
          ident '.' ident
          {
            $$ = CreateTableIdent(false, $1, $3, nullptr, nullptr);
          }
        | ident
          {
            $$ = CreateTableIdent(false, "", $1, nullptr, nullptr);
          }
        ;

sp_c_chistics:
          /* Empty */ {}
        | sp_c_chistics sp_c_chistic {}
        ;

/* Characteristics for both create and alter */
sp_chistic:
          COMMENT_SYM TEXT_STRING_sys
          { }
        | LANGUAGE_SYM SQL_SYM
          { /* Just parse it, we only have one language for now. */ }
        | NO_SYM SQL_SYM
          { }
        | CONTAINS_SYM SQL_SYM
          { }
        | READS_SYM SQL_SYM DATA_SYM
          { }
        | MODIFIES_SYM SQL_SYM DATA_SYM
          { }
        | sp_suid
          {}
        ;

/* Create characteristics */
sp_c_chistic:
          sp_chistic            { }
        | DETERMINISTIC_SYM     { }
        | not DETERMINISTIC_SYM { }
        ;

sp_suid:
          SQL_SYM SECURITY_SYM DEFINER_SYM
          {
          }
        | SQL_SYM SECURITY_SYM INVOKER_SYM
          {
          }
        ;

/* Stored FUNCTION parameter declaration list */
sp_fdparam_list:
          /* Empty */
        | sp_fdparams
        ;

sp_fdparams:
          sp_fdparams ',' sp_fdparam
        | sp_fdparam
        ;

sp_fdparam:
          ident type opt_collate
          {
          }
        ;

/* Stored PROCEDURE parameter declaration list */
sp_pdparam_list:
          /* Empty */
        | sp_pdparams
        ;

sp_pdparams:
          sp_pdparams ',' sp_pdparam
        | sp_pdparam
        ;

sp_pdparam:
          sp_opt_inout ident type opt_collate
          {
          }
        ;

sp_opt_inout:
          /* Empty */ { }
        | IN_SYM      { }
        | OUT_SYM     { }
        | INOUT_SYM   { }
        ;
describe_command:
          DESC
        | DESCRIBE
        ;
opt_describe_column:
          /* empty */ { $$= ""; }
        | text_string { ThrowNotSupportedException( "desc column" ); }
        | ident { ThrowNotSupportedException( "desc column" ); }
        ;
/* A Oracle compatible synonym for show */
/* desc information_schema.schemata;
   desc information_schema.schemata CATALOG_NAME;
*/
describe_stmt:
          describe_command table_ident opt_describe_column
          {
            $$ = CreateShowColumnsStructure(Show_cmd_type::STANDARD, $2, "", $3, nullptr);
          }
        ;
/* Show things */
show:
    SHOW show_param { $$ = $2; }
    ;

show_param:
    DATABASES opt_wild_or_where_for_show {
        $$ = CreateShowDatabasesStructure($2->wild, $2->where);
    }
    | opt_show_cmd_type TABLES opt_db opt_wild_or_where_for_show {
        $$ = CreateShowTablesStructure($1, $3, $4->wild, $4->where);
    }
    | opt_full TRIGGERS_SYM opt_db opt_wild_or_where_for_show {
        $$ = CreateShowTriggersStructure($1, $3, $4->wild, $4->where, $4->str);
    }
    | EVENTS_SYM opt_db opt_wild_or_where_for_show {
        $$ = CreateShowEventsStructure($2, $3->wild, $3->where);
    }
    | TABLE_SYM STATUS_SYM opt_db opt_wild_or_where_for_show {
        $$ = CreateShowTableStatusStructure($3, $4->wild, $4->where);
    }
    | OPEN_SYM TABLES opt_db opt_wild_or_where {
        $$ = CreateShowOpenTablesStructure($3, $4->wild, $4->where);
    }
    | PLUGINS_SYM {
        auto showGlobalInfoStructurePointer = std::make_shared<ShowGlobalInfoStructure>();
        showGlobalInfoStructurePointer->showCmd = SHOW_CMD::SHOW_PLUGINS;
        $$ = showGlobalInfoStructurePointer;
    }
    | ENGINE_SYM ident_or_text show_engine_param {
        $$ = CreateShowEngineStructure($2, $3);
    }
    | ENGINE_SYM ALL show_engine_param {
        ThrowNotSupportedException("show engine all");
    }
    | opt_show_cmd_type
      COLUMNS
      from_or_in
      table_ident
      opt_db
      opt_wild_or_where_for_show {
        $$ = CreateShowColumnsStructure($1, $4, $5, $6->wild, $6->where);
    }
    | master_or_binary LOGS_SYM {
        ThrowNotSupportedException("show binary logs");
    }
    | SLAVE HOSTS_SYM {
        auto showStructurePointer = std::make_shared<ShowStructure>();
        showStructurePointer->showCmd = SHOW_CMD::SHOW_SLAVE_HOSTS;
        $$ = showStructurePointer;
    }
    | BINLOG_SYM EVENTS_SYM binlog_in binlog_from opt_limit_clause {
        ThrowNotSupportedException("show binlog events");
    }
    | RELAYLOG_SYM EVENTS_SYM binlog_in binlog_from opt_limit_clause opt_channel {
        ThrowNotSupportedException("show relaylog events");
    }
    | opt_extended          /* #1 */
      keys_or_index         /* #2 */
      from_or_in            /* #3 */
      table_ident           /* #4 */
      opt_db                /* #5 */
      opt_where_clause_expr /* #6 */ {
        $$ = CreateShowIndexStructure($1, $5, $4, $6);
    }
    | opt_storage ENGINES_SYM {
        auto showGlobalInfoStructurePointer = std::make_shared<ShowGlobalInfoStructure>();
        showGlobalInfoStructurePointer->showCmd = SHOW_CMD::SHOW_ENGINES;
        $$ = showGlobalInfoStructurePointer;
    }
    /**
    mysql 5.7.26
    mysql> show count(*) warnings;
    +-------------------------+
    | @@session.warning_count |
    +-------------------------+
    |                       0 |
    +-------------------------+
    1 row in set (0.00 sec)

    mysql> show count(*) errors;
    +-----------------------+
    | @@session.error_count |
    +-----------------------+
    |                     0 |
    +-----------------------+
    */
    | COUNT_SYM '(' '*' ')' WARNINGS {
      ThrowNotSupportedException("show count(*) warnings");
        // TODO
        // create_select_for_variable(&pc, "warning_count")
    }
    | COUNT_SYM '(' '*' ')' ERRORS {
      ThrowNotSupportedException("show count(*) errors");
        // TODO
        // create_select_for_variable(&pc, "error_count")
    }
    | WARNINGS opt_limit_clause {
        $$ = CreateShowWarningsStructure($2);
    }
    | ERRORS opt_limit_clause {
        $$ = CreateShowErrorsStructure($2);
    }
    | PROFILES_SYM {
        ThrowNotSupportedException("show profiles");
    }
    | PROFILE_SYM opt_profile_defs opt_profile_args opt_limit_clause {
        ThrowNotSupportedException("show profile");
    }
    | opt_var_type STATUS_SYM opt_wild_or_where_for_show {
        auto statusVariableStructure = std::make_shared<ShowStatusVariableStructure>();
        statusVariableStructure->showCmd = SHOW_CMD::SHOW_STATUS;
        if (OPT_GLOBAL == $1) {
            statusVariableStructure->global = true;
        } else {
            statusVariableStructure->global = false;
        }
        statusVariableStructure->wild = $3->wild;
        statusVariableStructure->where = $3->where;
        $$ = statusVariableStructure;
    }
    | opt_full PROCESSLIST_SYM {
        $$ = CreateShowProcessListStructure($1);
    }
    | opt_var_type VARIABLES opt_wild_or_where_for_show {
        $$ = CreateShowVariableStructure($1, $3->wild, $3->where, $3->str);
    }
    | character_set opt_wild_or_where_for_show {
        $$ = CreateShowCharsetStructure($2->wild, $2->where, $2->str);
    }
    | COLLATION_SYM opt_wild_or_where_for_show {
        $$ = CreateShowCollationStructure($2->wild, $2->where, $2->str);
    }
    | PRIVILEGES {
        auto showStructurePointer = std::make_shared<ShowStructure>();
        showStructurePointer->showCmd = SHOW_CMD::SHOW_PRIVILEGES;
        $$ = showStructurePointer;
    }
    | GRANTS {
        ThrowNotSupportedException("show grants");
    }
    | GRANTS FOR_SYM user {
        ThrowNotSupportedException("show grants for user");
    }
    | GRANTS FOR_SYM user USING user_list {
        ThrowNotSupportedException("show grants for user");
    }
    | CREATE DATABASE opt_if_not_exists ident {
        $$ = CreateShowCreateDbStructure($4);
    }
    | CREATE SCHEMA opt_if_not_exists ident {
        $$ = CreateShowCreateDbStructure($4);
    }
    | CREATE TABLE_SYM table_ident {
        $$ = CreateShowCreateTableStructure($3->GetDb(), $3->GetID());
    }
    | CREATE VIEW_SYM table_ident {
        ThrowNotSupportedException("show create view");
    }
    | MASTER_SYM STATUS_SYM {
        auto showStructurePointer = std::make_shared<ShowStructure>();
        showStructurePointer->showCmd = SHOW_CMD::SHOW_MASTER_STATUS;
        $$ = showStructurePointer;
    }
    | SLAVE STATUS_SYM opt_channel {
        ThrowNotSupportedException("show slave status");
    }
    | CREATE PROCEDURE_SYM sp_name {
        ThrowNotSupportedException("show create procedure");
    }
    | CREATE FUNCTION_SYM sp_name {
        ThrowNotSupportedException("show create function");
    }
    | CREATE TRIGGER_SYM sp_name {
        ThrowNotSupportedException("show create trigger");
    }
    | PROCEDURE_SYM STATUS_SYM opt_wild_or_where_for_show {
        $$ = CreateShowProcedureStatusStructure($3->wild, $3->where);
        // ThrowNotSupportedException("show procedure status");
    }
    | FUNCTION_SYM STATUS_SYM opt_wild_or_where_for_show {
        $$ = CreateShowFunctionStatusStructure($3->wild, $3->where);
        // ThrowNotSupportedException("show function status");
    }
    | PROCEDURE_SYM CODE_SYM sp_name {
        // $$ = CreateShowProcedureCodeStructure($3);
        ThrowNotSupportedException("show procedure code");
    }
    | FUNCTION_SYM CODE_SYM sp_name {
        // $$ = CreateShowFunctionCodeStructure($3);
        ThrowNotSupportedException("show function code");
    }
    | CREATE EVENT_SYM sp_name {
        ThrowNotSupportedException("show create event");
    }
    | CREATE USER user {
        ThrowNotSupportedException("show create user");
    }
    ;

show_engine_param:
          STATUS_SYM
          { $$ = SHOW_CMD::SHOW_ENGINE_STATUS; }
        | MUTEX_SYM
          { $$ = SHOW_CMD::SHOW_ENGINE_MUTEX; }
        | LOGS_SYM
          { $$ = SHOW_CMD::SHOW_ENGINE_LOGS; }
        ;

master_or_binary:
          MASTER_SYM
        | BINARY_SYM
        ;

opt_wild_or_where:
          /* empty */           { $$= std::make_shared<WildOrWhere>( "", nullptr, ""); }
        | LIKE TEXT_STRING_sys
          {
            $$= std::make_shared<WildOrWhere>( $2, nullptr, driver.get_string_at_location(@2));
          }
        | WHERE expr            { $$= std::make_shared<WildOrWhere>( "", $2, driver.get_string_at_location(@2)); }
        ;

opt_wild_or_where_for_show:
          /* empty */                   { $$= std::make_shared<WildOrWhere>( "", nullptr, "" ); }
        | LIKE TEXT_STRING_literal      { $$= std::make_shared<WildOrWhere>( $2, nullptr, driver.get_string_at_location(@2)); }
        | WHERE expr                    { $$= std::make_shared<WildOrWhere>( "", $2, driver.get_string_at_location(@2)); }
        ;

opt_storage:
          /* empty */
        | STORAGE_SYM
        ;

opt_db:
          /* empty */  { $$= ""; }
        | from_or_in ident { $$= $2; }
        ;

opt_full:
          /* empty */ { $$= 0; }
        | FULL        { $$= 1; }
        ;

opt_extended:
          /* empty */   { $$= 0; }
        | EXTENDED_SYM  { $$= 1; }
        ;

opt_show_cmd_type:
          /* empty */          { $$= Show_cmd_type::STANDARD; }
        | FULL                 { $$= Show_cmd_type::FULL_SHOW; }
        | EXTENDED_SYM         { $$= Show_cmd_type::EXTENDED_SHOW; }
        | EXTENDED_SYM FULL    { $$= Show_cmd_type::EXTENDED_FULL_SHOW; }
        ;

from_or_in:
          FROM
        | IN_SYM
        ;

binlog_in:
          /* empty */            { }
        | IN_SYM TEXT_STRING_sys { }
        ;

binlog_from:
          /* empty */        { }
        | FROM ulonglong_num { }
        ;

ulong_num:
          NUM           { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        /* | HEX_NUM       { $$= (ulong) my_strtoll($1.data(), (char**) 0, 16); } */
        | LONG_NUM      { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        | ULONGLONG_NUM { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        | DECIMAL_NUM   { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        | FLOAT_NUM     { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        ;

real_ulong_num:
          NUM           { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        /* | HEX_NUM       { $$= (ulong) my_strtoll($1.data(), (char**) 0, 16); } */
        | LONG_NUM      { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        | ULONGLONG_NUM { int error; $$= (ulong) my_strtoll10($1.data(), nullptr, &error); }
        | dec_num_error { ThrowNotSupportedException("decimal number"); }
        ;

ulonglong_num:
          NUM           { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        | ULONGLONG_NUM { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        | LONG_NUM      { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        | DECIMAL_NUM   { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        | FLOAT_NUM     { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        ;

real_ulonglong_num:
          NUM           { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        /* | HEX_NUM       { $$= (ulonglong) my_strtoll($1.data(), (char**) 0, 16); } */
        | ULONGLONG_NUM { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        | LONG_NUM      { int error; $$= (ulonglong) my_strtoll10($1.data(), nullptr, &error); }
        | dec_num_error { ThrowNotSupportedException("decimal number"); }

dec_num_error:
          dec_num
          { }
        ;

dec_num:
          DECIMAL_NUM
        | FLOAT_NUM
        ;
sp_handler_type:
          EXIT_SYM      { }
        | CONTINUE_SYM  { }
        /*| UNDO_SYM      { QQ No yet } */
        ;

sp_hcond_list:
          sp_hcond_element
          { }
        | sp_hcond_list ',' sp_hcond_element
          { }
        ;

sp_hcond_element:
          sp_hcond
          {
          }
        ;
sp_cond:
          ulong_num
          { /* mysql errno */
          }
        | sqlstate
        ;

sqlstate:
          SQLSTATE_SYM opt_value TEXT_STRING_literal
          { /* SQLSTATE */

            /*
              An error is triggered:
                - if the specified string is not a valid SQLSTATE,
                - or if it represents the completion condition -- it is not
                  allowed to SIGNAL, or declare a handler for the completion
                  condition.
            */
          }
        ;
sp_hcond:
          sp_cond
          {
          }
        | ident /* CONDITION name */
          {
          }
        | SQLWARNING_SYM /* SQLSTATEs 01??? */
          {
          }
        | not FOUND_SYM /* SQLSTATEs 02??? */
          {
          }
        | SQLEXCEPTION_SYM /* All other SQLSTATEs */
          {
          }
        ;

opt_value:
          /* Empty */  {}
        | VALUE_SYM    {}
        ;

label_ident:
          IDENT_sys    { }
        | label_keyword
          {
          }
        ;
lvalue_ident:
          IDENT_sys
        | lvalue_keyword
        ;

/*
  SQLCOM_SET_OPTION statement.

  Note that to avoid shift/reduce conflicts, we have separate rules for the
  first option listed in the statement.
*/

set:
          SET start_option_value_list
          {
            $$ = $2;
          }
        ;

// Start of option value list
start_option_value_list:
          option_value_no_option_type option_value_list_continued
          {
            $$ = AppendOptionSetStructures($1, $2);
          }
        | TRANSACTION_SYM transaction_characteristics {
            auto setStructures = std::make_shared<std::vector<SetStructurePtr>>();
            auto setStructurePtr = std::make_shared<SetStructure>();
            setStructurePtr->m_setCmd = SET_CMD::SET_TX;
            setStructures->emplace_back(setStructurePtr);
            $$ = setStructures;
        }
        | option_type start_option_value_list_following_option_type
          {
            ReviseSetStructuresHead($1, $2);
            $$ = $2;
          }
         | PASSWORD equal password
          {
            auto setStructures = std::make_shared<std::vector<SetStructurePtr>>();
            std::string user( "" );
            auto setStructurePtr = CreateSetPasswordStructure( user, $3 );
            setStructures->emplace_back(setStructurePtr);
            $$ = setStructures;
          }
        | PASSWORD equal PASSWORD '(' password ')'
          {
            ThrowNotSupportedException( "SET PASSWORD = PASSWORD('<plaintext_password>'" );
          }
        | PASSWORD FOR_SYM user equal password
          {
            auto setStructures = std::make_shared<std::vector<SetStructurePtr>>();
            auto setStructurePtr = CreateSetPasswordStructure( $3->m_user, $5 );
            setStructures->emplace_back(setStructurePtr);
            $$ = setStructures;
          }
        | PASSWORD FOR_SYM user equal PASSWORD '(' password ')'
          {
            ThrowNotSupportedException( "SET PASSWORD FOR <user> = "
                                        "PASSWORD('<plaintext_password>'" );
          }
        ;

password:
  TEXT_STRING
  {
    $$=$1;
  }

transaction_characteristics:
          transaction_access_mode opt_isolation_level
          {
          }
        | isolation_level opt_transaction_access_mode
          {
          }
        ;

transaction_access_mode:
          transaction_access_mode_types
          {
            ThrowNotSupportedException("transaction_access_mode");
          }
        ;

opt_transaction_access_mode:
          /* empty */                 {  }
        | ',' transaction_access_mode { }
        ;

isolation_level:
          ISOLATION LEVEL_SYM isolation_types
          {
          }
        ;

opt_isolation_level:
          /* empty */         { }
        | ',' isolation_level { }
        ;

transaction_access_mode_types:
          READ_SYM ONLY_SYM { }
        | READ_SYM WRITE_SYM { }
        ;

isolation_types:
          READ_SYM UNCOMMITTED_SYM { ThrowNotSupportedException("READ UNCOMMITTED"); }
        | READ_SYM COMMITTED_SYM   { ThrowNotSupportedException("READ COMMITTED"); }
        | REPEATABLE_SYM READ_SYM  { }
        | SERIALIZABLE_SYM         { ThrowNotSupportedException("SERIALIZABLE"); }
        ;
opt_and:
          /* empty */
        | AND_SYM
        ;
opt_primary:
          /* empty */
        | PRIMARY_SYM
        ;

query_expression_or_parens:
          query_expression {
            $$ = $1;
          }
        | query_expression_parens {
          $$ = $1;
        }
        ;

/*
** Insert : add new data to table
*/

insert_stmt:
          INSERT_SYM                   /* #1 */
          insert_lock_option           /* #2 */
          opt_ignore                   /* #3 */
          opt_INTO                     /* #4 */
          table_ident                  /* #5 */
          opt_use_partition            /* #6 */
          insert_from_constructor      /* #7 */
          opt_insert_update_list       /* #8 */
          {
            $$ = CreateInsertStructure( $5,
                                        $7,
                                        std::move( driver.global_insert_values_list ),
                                        $8.first,
                                        $8.second );
            driver.global_insert_values_list.clear();
          }
        | INSERT_SYM                   /* #1 */
          insert_lock_option           /* #2 */
          opt_ignore                   /* #3 */
          opt_INTO                     /* #4 */
          table_ident                  /* #5 */
          opt_use_partition            /* #6 */
          SET                          /* #7 */
          update_list                  /* #8 */
          opt_insert_update_list       /* #9 */
          {
            $$ = CreateInsertStructure( $5,
                                        $8.first,
                                        $8.second,
                                        $9.first,
                                        $9.second );
          }
        | INSERT_SYM                   /* #1 */
          insert_lock_option           /* #2 */
          opt_ignore                   /* #3 */
          opt_INTO                     /* #4 */
          table_ident                  /* #5 */
          opt_use_partition            /* #6 */
          insert_query_expression      /* #7 */
          opt_insert_update_list       /* #8 */
          {
            $$ = CreateInsertStructure( $5,
                                        $7.first,
                                        $7.second,
                                        $8.first,
                                        $8.second );
          }
        ;
insert_lock_option:
          /* empty */   { }
        | LOW_PRIORITY  { ThrowNotSupportedException("LOW_PRIORITY"); }
        | DELAYED_SYM   { ThrowNotSupportedException("DELAYED"); }
        | HIGH_PRIORITY { ThrowNotSupportedException("HIGH_PRIORITY"); }
        ;
opt_ignore:
          /* empty */ { $$= false; }
        | IGNORE_SYM  { $$= true; }
        ;
opt_INTO:
          /* empty */
        | INTO
        ;
insert_query_expression:
          query_expression_or_parens
          {
            $$.second = $1;
          }
        | '(' ')' query_expression_or_parens
          {
            $$.second = $3;
          }
        | '(' fields ')' query_expression_or_parens
          {
            $$.first = $2;
            $$.second = $4;
          }
        ;
insert_from_constructor:
          insert_values
          {
            //$$ = std::vector< Expression >();
          }
        | '(' ')' insert_values
          {
            //$$ = std::vector< Expression >();
          }
        | '(' fields ')' insert_values
          {
            $$ = $2;
          }
        ;
insert_values:
          value_or_values values_list
          {
          }
        ;
value_or_values:
          VALUE_SYM
        | VALUES
        ;

values_list:
          values_list ','  row_value
          {

          }
        | row_value
          {

          }
        ;
row_value:
          '(' opt_values ')'
          {
              driver.global_insert_values_list.emplace_back( std::move( $2 ) );
          }
        ;

opt_values:
          /* empty */
          {
            $$ = make_shared< vector< BiaodashiPointer > >( );
          }
        | values
        ;

values:
          values ','  expr_or_default
          {
            $$ = std::move( $1 );
            $$->emplace_back( std::move( $3 ) );
          }
        | expr_or_default
          {
            $$ = make_shared< vector< BiaodashiPointer > >( );
            $$->emplace_back( std::move( $1 ) );
          }
        ;

expr_or_default:
         expr
        | DEFAULT_SYM
          {
            $$ = make_shared<CommonBiaodashi>( BiaodashiType::Default, 0 );
          }
        ;
opt_insert_update_list:
          /* empty */
          {
          }
        | ON_SYM DUPLICATE_SYM KEY_SYM UPDATE_SYM update_list
          {
            $$ = $5;
          }
        ;

/* Update rows in a table */
update_stmt:
          // opt_with_clause       /* #1 */
          UPDATE_SYM            /* #1 */
          opt_low_priority      /* #2 */
          opt_ignore            /* #3 */
          table_reference_list  /* #4 */
          SET                   /* #5 */
          update_list           /* #6 */
          opt_where_clause      /* #7 */
          opt_order_clause      /* #8 */
          opt_simple_limit      /* #9 */
          {
            $$ = CreateUpdateStructure( $4, $6.first, $6.second, $7, $8, $9 );
          }
        ;

opt_with_clause:
          /* empty */ { }
        | with_clause { }
        ;

update_list:
          update_list ',' update_elem
          {
            $$= $1;
            $$.first.push_back( $3.first );
            $$.second.push_back( $3.second );
          }
        | update_elem
          {
            $$.first.push_back( $1.first );
            $$.second.push_back( $1.second );
          }
        ;

update_elem:
          simple_ident_nospvar equal expr_or_default
          {
            $$.first = CreateIdentExpression( $1 );
            $$.second = $3;
          }
        ;

opt_low_priority:
          /* empty */ { }
        | LOW_PRIORITY { ThrowNotSupportedException("LOW_PRIORITY"); }
        ;

explain_stmt:
        explain_cmd explain_option select_stmt {
          $$ = $3;
        };

explain_cmd:
        DESCRIBE
        | EXPLAIN_SYM
        | DESC;

explain_option:
        /* empty */ { }
        | FORMAT_SYM EQ ident_or_text {
          if ($3 != "tree") {
            ThrowNotSupportedException("only support FORMAT=TREE");
          }
        };

/* Delete rows from a table */

delete_stmt:
          // opt_with_clause
          DELETE_SYM         // #1
          opt_delete_options // #2
          FROM               // #3
          table_ident        // #4
          opt_table_alias    // #5
          opt_use_partition  // #6
          opt_where_clause   // #7
          opt_order_clause   // #8
          opt_simple_limit   // #9
          {
            $$ = CreateDeleteStructure( $4, $5, $7, $8, $9 );
          }
        | // opt_with_clause
          DELETE_SYM
          opt_delete_options
          table_alias_ref_list
          FROM
          table_reference_list
          opt_where_clause
          {
            $$ = CreateDeleteStructure( *$3, $5, $6 );
          }
        | // opt_with_clause
          DELETE_SYM
          opt_delete_options
          FROM
          table_alias_ref_list
          USING
          table_reference_list
          opt_where_clause
          {
            $$ = CreateDeleteStructure( *$4, $6, $7 );
          }
        ;

opt_simple_limit:
          /* empty */        { $$= NULL; }
        | LIMIT limit_option
        {
            int64_t offset = 0;
            int64_t size = std::stoll($2);
            $$ = std::make_shared<LimitStructure>(offset, size);
        }
        ;

opt_wild:
          /* empty */
        | '.' '*'
        ;
opt_delete_options:
          /* empty */                          { }
        | opt_delete_option opt_delete_options { }
        ;

opt_delete_option:
          QUICK        { ThrowNotSupportedException("QUICK"); }
        | LOW_PRIORITY { ThrowNotSupportedException("LOW_PRIORITY"); }
        | IGNORE_SYM   { ThrowNotSupportedException("IGNORE"); }
        ;
table_alias_ref_list:
          table_ident_opt_wild
          {
            $$ = std::make_shared<vector<std::shared_ptr<BasicRel>>>();
            $$->emplace_back( $1 );
          }
        | table_alias_ref_list ',' table_ident_opt_wild
          {
            $$ = $1;
            $$->emplace_back( $3 );
          }
        ;
table_ident_opt_wild:
          ident opt_wild
          {
            $$ = CreateTableIdent( false, "", $1, nullptr, nullptr );
          }
        | ident '.' ident opt_wild
          {
            $$ = CreateTableIdent( false, $1, $3, nullptr, nullptr );
          }
        ;

fields:
          fields ',' insert_ident
          {
            $$ = $1;
            $$.push_back( CreateIdentExpression( $3 ) );
          }
        | insert_ident
          {
            $$.push_back( CreateIdentExpression( $1 ) );
          }
        ;

/* change database */

use:
          USE_SYM ident
          {
            $$ = CreateChangeDbStructure($2);
          }
        ;
/* kill threads */

kill:
          KILL_SYM kill_option expr
          {
            $$ = CreateKillStructure($2, $3);
          }
        ;

kill_option:
          /* empty */ { $$ = 0; }
        | CONNECTION_SYM { $$ = 0; }
        | QUERY_SYM      { $$ = 1; }
        ;

/* Lock function */

lock:
          LOCK_SYM table_or_tables
          table_lock_list
        ;

table_or_tables:
        TABLES
        | TABLE_SYM
        ;

table_lock_list:
          table_lock
        | table_lock_list ',' table_lock
        ;

table_lock:
          table_ident opt_table_alias lock_option
        ;

lock_option:
          READ_SYM               { }
        | WRITE_SYM              { }
        | LOW_PRIORITY WRITE_SYM
          {
          }
        | READ_SYM LOCAL_SYM     { }
        ;
unlock:
          UNLOCK_SYM
          table_or_tables
          {}
        ;
shutdown_stmt:
          SHUTDOWN
          {
            $$ = CreateShutdownStructure();
          }
        ;

/* import, export of files */

load_stmt:
          LOAD                          /*  1 */
          data_or_xml                   /*  2 */
          load_data_lock                /*  3 */
          opt_local                     /*  4 */
          INFILE                        /*  5 */
          TEXT_STRING_filesystem        /*  6 */
          opt_duplicate                 /*  7 */
          INTO                          /*  8 */
          TABLE_SYM                     /*  9 */
          table_ident                   /* 10 */
          opt_use_partition             /* 11 */
          opt_load_data_charset         /* 12 */
          opt_xml_rows_identified_by    /* 13 */
          opt_field_term                /* 14 */
          opt_line_term                 /* 15 */
          opt_ignore_lines              /* 16 */
          opt_field_or_var_spec         /* 17 */
          opt_load_data_set_spec        /* 18 */
          {
          /*
            $$= NEW_PTN PT_load_table($2,  // data_or_xml
                                      $3,  // load_data_lock
                                      $4,  // opt_local
                                      $6,  // TEXT_STRING_filesystem
                                      $7,  // opt_duplicate
                                      $10, // table_ident
                                      $11, // opt_use_partition
                                      $12, // opt_load_data_charset
                                      $13, // opt_xml_rows_identified_by
                                      $14, // opt_field_term
                                      $15, // opt_line_term
                                      $16, // opt_ignore_lines
                                      $17, // opt_field_or_var_spec
                                      $18.set_var_list,// opt_load_data_set_spec
                                      $18.set_expr_list,
                                      $18.set_expr_str_list);
                                      */
            $$ = CreateLoadDataStructure( $2,
                                          $3,
                                          $4,
                                          $6,
                                          $7,
                                          $10,
                                          $12,
                                          $14,
                                          $15,
                                          $16);
          }
        ;

data_or_xml:
          DATA_SYM{ $$= FILETYPE_CSV; }
        | XML_SYM { $$= FILETYPE_XML; }
        ;

opt_local:
          /* empty */ { $$= false; }
        | LOCAL_SYM   { $$= true; }
        ;

load_data_lock:
          /* empty */ { $$= TL_WRITE_DEFAULT; }
        | CONCURRENT  { $$= TL_WRITE_CONCURRENT_INSERT; }
        | LOW_PRIORITY { $$= TL_WRITE_LOW_PRIORITY; }
        ;

opt_duplicate:
          /* empty */ { $$= On_duplicate::ERROR; }
        | duplicate
        ;

duplicate:
          REPLACE_SYM { $$= On_duplicate::REPLACE_DUP; }
        | IGNORE_SYM  { $$= On_duplicate::IGNORE_DUP; }
        ;

opt_load_data_charset:
          /* Empty */ { $$= ""; }
        | character_set charset_name { $$ = $2; }
        ;
opt_field_term:
          /* empty */             { $$.cleanup(); }
        | COLUMNS field_term_list { $$= $2; }
        ;

field_term_list:
          field_term_list field_term
          {
            $$= $1;
            $$.merge_field_separators($2);
          }
        | field_term
        ;

field_term:
          TERMINATED BY text_string
          {
            $$.cleanup();
            $$.field_term= std::make_shared<string>($3);
          }
        | OPTIONALLY ENCLOSED BY text_string
          {
            $$.cleanup();
            $$.enclosed= std::make_shared<string>($4);
            $$.opt_enclosed= 1;
          }
        | ENCLOSED BY text_string
          {
            $$.cleanup();
            $$.enclosed= std::make_shared<string>($3);
          }
        | ESCAPED BY text_string
          {
            $$.cleanup();
            $$.escaped= std::make_shared<string>($3);
          }
        ;

opt_line_term:
          /* empty */          { $$.cleanup(); }
        | LINES line_term_list { $$= $2; }
        ;

line_term_list:
          line_term_list line_term
          {
            $$= $1;
            $$.merge_line_separators($2);
          }
        | line_term
        ;

line_term:
          TERMINATED BY text_string
          {
            $$.cleanup();
            $$.line_term= std::make_shared<string>($3);
          }
        | STARTING BY text_string
          {
            $$.cleanup();
            $$.line_start= std::make_shared<string>($3);
          }
        ;

opt_xml_rows_identified_by:
          /* empty */                            { $$= ""; }
        | ROWS_SYM IDENTIFIED_SYM BY text_string { $$= $4; }
        ;
opt_ignore_lines:
          /* empty */                   { $$= 0; }
        | IGNORE_SYM NUM lines_or_rows  { $$= std::atol($2.data()); }
        ;

lines_or_rows:
          LINES
        | ROWS_SYM
        ;
opt_field_or_var_spec:
          /* empty */            { }
        | '(' fields_or_vars ')' { ThrowNotSupportedException("fields_or_vars"); }
        | '(' ')'                { }
        ;

fields_or_vars:
          fields_or_vars ',' field_or_var
          {
          }
        | field_or_var
          {
          }
        ;

field_or_var:
          simple_ident_nospvar
        | '@' ident_or_text
          {
          }
        ;

opt_load_data_set_spec:
          /* empty */                { }
        | SET load_data_set_list { ThrowNotSupportedException("SET"); }
        ;

load_data_set_list:
          load_data_set_list ',' load_data_set_elem
          {
          }
        | load_data_set_elem
          {
          }
        ;

load_data_set_elem:
          simple_ident_nospvar equal expr_or_default
          {
          }
        ;
 start:
          START_SYM TRANSACTION_SYM opt_start_transaction_option_list
          {
            // lex->sql_command= SQLCOM_BEGIN;
            $$ = make_shared< TransactionStructure >();
            $$->txCmd = TX_CMD::TX_START;
          }
        ;

opt_start_transaction_option_list:
          /* empty */
          {
            // $$= 0;
          }
        | start_transaction_option_list
          {
            // $$= $1;
          }
        ;

start_transaction_option_list:
          start_transaction_option
          {
            ThrowNotSupportedException( "start transaction with characteristics" );
          }
        | start_transaction_option_list ',' start_transaction_option
          {
          }
        ;

start_transaction_option:
          WITH CONSISTENT_SYM SNAPSHOT_SYM
          {
            // $$= MYSQL_START_TRANS_OPT_WITH_CONS_SNAPSHOT;
          }
        | READ_SYM ONLY_SYM
          {
            // $$= MYSQL_START_TRANS_OPT_READ_ONLY;
          }
        | READ_SYM WRITE_SYM
          {
            // $$= MYSQL_START_TRANS_OPT_READ_WRITE;
          }
        ;       
begin_stmt:
          BEGIN_SYM opt_work 
          {
            $$ = make_shared< TransactionStructure >();
            $$->txCmd = TX_CMD::TX_START;
          }
        ;

opt_work:
          /* empty */ {}
        | WORK_SYM  { ThrowNotSupportedException( "WORK" ); }
        ;
opt_chain:
          /* empty */
          { $$ = TVL_UNKNOWN; }
        | AND_SYM NO_SYM CHAIN_SYM { $$ = TVL_NO; }
        | AND_SYM CHAIN_SYM        { $$ = TVL_YES;}
        ;
opt_release:
          /* empty */
          { $$ = TVL_UNKNOWN; }
        | RELEASE_SYM        { $$ = TVL_YES; }
        | NO_SYM RELEASE_SYM { $$ = TVL_NO; }
        ;

opt_savepoint:
          /* empty */ {}
        | SAVEPOINT_SYM {}
        
commit:
          COMMIT_SYM opt_work opt_chain opt_release
          {
            $$ = CreateEndTxStructure( TX_CMD::TX_COMMIT, $3, $4 );
          }
        ;

rollback:
          ROLLBACK_SYM opt_work opt_chain opt_release
          {
            $$ = CreateEndTxStructure( TX_CMD::TX_ROLLBACK, $3, $4 );
          }
        | ROLLBACK_SYM opt_work
          TO_SYM opt_savepoint ident
          {
            // LEX *lex=Lex;
            // lex->sql_command= SQLCOM_ROLLBACK_TO_SAVEPOINT;
            // lex->ident= $5;
            ThrowNotSupportedException("rollback to savepoint");
          }
        ;

%%/*** Additional Code ***/

void aries_parser::Parser::error(const aries_parser::Parser::location_type& l,
                const string& m)
{
    driver.error(l, m);
}
