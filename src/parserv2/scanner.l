%{
#include "scanner.h"
#include "utils/string_util.h"

using namespace aries_parser;

/* import the parser's token type into a local typedef */
typedef Parser::token token;
typedef Parser::token_type token_type;

#define FLEX_STD std::
#define yyterminate() return token::END_OF_INPUT;

#define GENERATE_TOKEN(token_name) \
    auto string_value = std::string( yytext, yyleng ); \
    aries_utils::to_lower( string_value ); \
    yylval->emplace<std::string>( std::move( string_value ) );\
    return Parser::token::token_name;

#define YY_USER_ACTION do {\
    yylloc->begin.line = yylineno; \
    yylloc->begin.column = yycolumn; \
    yylloc->end.line = yylineno; \
    yylloc->end.column = yycolumn + yyleng; \
    yycolumn += yyleng; \
} while(0);

%}

%option c++
%option prefix="Aries"
%option yylineno
%option 8bit
%option full
%option align

DEC_DIGIT [0-9]

EXPONENT_NUM_PART [Ee][-+]?{DEC_DIGIT}+

REAL_LITERAL  {DEC_DIGIT}*\.{DEC_DIGIT}+{EXPONENT_NUM_PART}|{DEC_DIGIT}+\.?{EXPONENT_NUM_PART}
DECIMAL_LITERAL  {DEC_DIGIT}*\.{DEC_DIGIT}+

%%
ABORT { GENERATE_TOKEN(ABORT_SYM) }
ACCESSIBLE { GENERATE_TOKEN(ACCESSIBLE_SYM) }
ACCOUNT { GENERATE_TOKEN(ACCOUNT_SYM) }
ACTION { GENERATE_TOKEN(ACTION) }
ADD { GENERATE_TOKEN(ADD) }
ADDDATE { GENERATE_TOKEN(ADDDATE_SYM) }
AFTER { GENERATE_TOKEN(AFTER_SYM) }
AGAINST { GENERATE_TOKEN(AGAINST) }
AGGREGATE { GENERATE_TOKEN(AGGREGATE_SYM) }
ALGORITHM { GENERATE_TOKEN(ALGORITHM_SYM) }
ALL { GENERATE_TOKEN(ALL) }
ALTER { GENERATE_TOKEN(ALTER) }
ALWAYS { GENERATE_TOKEN(ALWAYS_SYM) }
OBSOLETE_TOKEN_271 { GENERATE_TOKEN(OBSOLETE_TOKEN_271) }
ANALYZE { GENERATE_TOKEN(ANALYZE_SYM) }
AND_AND { GENERATE_TOKEN(AND_AND_SYM) }
AND { GENERATE_TOKEN(AND_SYM) }
ANY { GENERATE_TOKEN(ANY_SYM) }
AS { GENERATE_TOKEN(AS) }
ASC { GENERATE_TOKEN(ASC) }
ASCII { GENERATE_TOKEN(ASCII_SYM) }
ASENSITIVE { GENERATE_TOKEN(ASENSITIVE_SYM) }
AT { GENERATE_TOKEN(AT_SYM) }
AUTOEXTEND_SIZE { GENERATE_TOKEN(AUTOEXTEND_SIZE_SYM) }
AUTO_INC { GENERATE_TOKEN(AUTO_INC) }
AUTO_INCREMENT { GENERATE_TOKEN(AUTO_INC) }
AVG_ROW_LENGTH { GENERATE_TOKEN(AVG_ROW_LENGTH) }
AVG { GENERATE_TOKEN(AVG_SYM) }
BACKUP { GENERATE_TOKEN(BACKUP_SYM) }
BEFORE { GENERATE_TOKEN(BEFORE_SYM) }
BEGIN { GENERATE_TOKEN(BEGIN_SYM) }
BETWEEN { GENERATE_TOKEN(BETWEEN_SYM) }
BIGINT { GENERATE_TOKEN(BIGINT_SYM) }
BINARY { GENERATE_TOKEN(BINARY_SYM) }
BINLOG { GENERATE_TOKEN(BINLOG_SYM) }
BIN_NUM { GENERATE_TOKEN(BIN_NUM) }
BIT_AND { GENERATE_TOKEN(BIT_AND) }
BIT_OR { GENERATE_TOKEN(BIT_OR) }
BIT { GENERATE_TOKEN(BIT_SYM) }
BIT_XOR { GENERATE_TOKEN(BIT_XOR) }
BLOB { GENERATE_TOKEN(BLOB_SYM) }
BLOCK { GENERATE_TOKEN(BLOCK_SYM) }
BOOLEAN { GENERATE_TOKEN(BOOLEAN_SYM) }
BOOL { GENERATE_TOKEN(BOOL_SYM) }
BOTH { GENERATE_TOKEN(BOTH) }
BTREE { GENERATE_TOKEN(BTREE_SYM) }
BY { GENERATE_TOKEN(BY) }
BYTE { GENERATE_TOKEN(BYTE_SYM) }
CACHE { GENERATE_TOKEN(CACHE_SYM) }
CALL { GENERATE_TOKEN(CALL_SYM) }
CASCADE { GENERATE_TOKEN(CASCADE) }
CASCADED { GENERATE_TOKEN(CASCADED) }
CASE { GENERATE_TOKEN(CASE_SYM) }
CAST { GENERATE_TOKEN(CAST_SYM) }
CATALOG_NAME { GENERATE_TOKEN(CATALOG_NAME_SYM) }
CHAIN { GENERATE_TOKEN(CHAIN_SYM) }
CHANGE { GENERATE_TOKEN(CHANGE) }
CHANGED { GENERATE_TOKEN(CHANGED) }
CHANNEL { GENERATE_TOKEN(CHANNEL_SYM) }
CHARSET { GENERATE_TOKEN(CHARSET) }
CHAR { GENERATE_TOKEN(CHAR_SYM) }
CHARACTER { GENERATE_TOKEN(CHAR_SYM) }
CHECKSUM { GENERATE_TOKEN(CHECKSUM_SYM) }
CHECK { GENERATE_TOKEN(CHECK_SYM) }
CIPHER { GENERATE_TOKEN(CIPHER_SYM) }
CLASS_ORIGIN { GENERATE_TOKEN(CLASS_ORIGIN_SYM) }
CLIENT { GENERATE_TOKEN(CLIENT_SYM) }
CLOSE { GENERATE_TOKEN(CLOSE_SYM) }
COALESCE { GENERATE_TOKEN(COALESCE) }
CODE { GENERATE_TOKEN(CODE_SYM) }
COLLATE { GENERATE_TOKEN(COLLATE_SYM) }
COLLATION { GENERATE_TOKEN(COLLATION_SYM) }
COLUMNS { GENERATE_TOKEN(COLUMNS) }
COLUMN { GENERATE_TOKEN(COLUMN_SYM) }
COLUMN_FORMAT { GENERATE_TOKEN(COLUMN_FORMAT_SYM) }
COLUMN_NAME { GENERATE_TOKEN(COLUMN_NAME_SYM) }
COMMENT { GENERATE_TOKEN(COMMENT_SYM) }
COMMITTED { GENERATE_TOKEN(COMMITTED_SYM) }
COMMIT { GENERATE_TOKEN(COMMIT_SYM) }
COMPACT { GENERATE_TOKEN(COMPACT_SYM) }
COMPLETION { GENERATE_TOKEN(COMPLETION_SYM) }
COMPRESSED { GENERATE_TOKEN(COMPRESSED_SYM) }
COMPRESSION { GENERATE_TOKEN(COMPRESSION_SYM) }
ENCRYPTION { GENERATE_TOKEN(ENCRYPTION_SYM) }
CONCURRENT { GENERATE_TOKEN(CONCURRENT) }
CONDITION { GENERATE_TOKEN(CONDITION_SYM) }
CONNECTION { GENERATE_TOKEN(CONNECTION_SYM) }
CONNECTION_ID { GENERATE_TOKEN(CONNECTION_ID_SYM) }
CONSISTENT { GENERATE_TOKEN(CONSISTENT_SYM) }
CONSTRAINT { GENERATE_TOKEN(CONSTRAINT) }
CONSTRAINT_CATALOG { GENERATE_TOKEN(CONSTRAINT_CATALOG_SYM) }
CONSTRAINT_NAME { GENERATE_TOKEN(CONSTRAINT_NAME_SYM) }
CONSTRAINT_SCHEMA { GENERATE_TOKEN(CONSTRAINT_SCHEMA_SYM) }
CONTAINS { GENERATE_TOKEN(CONTAINS_SYM) }
CONTEXT { GENERATE_TOKEN(CONTEXT_SYM) }
CONTINUE { GENERATE_TOKEN(CONTINUE_SYM) }
CONVERT { GENERATE_TOKEN(CONVERT_SYM) }
COUNT { GENERATE_TOKEN(COUNT_SYM) }
CPU { GENERATE_TOKEN(CPU_SYM) }
CREATE { GENERATE_TOKEN(CREATE) }
CROSS { GENERATE_TOKEN(CROSS) }
CUBE { GENERATE_TOKEN(CUBE_SYM) }
CURRENT_DATE { GENERATE_TOKEN(CURDATE) }
CURRENT { GENERATE_TOKEN(CURRENT_SYM) }
CURRENT_USER { GENERATE_TOKEN(CURRENT_USER) }
CURSOR { GENERATE_TOKEN(CURSOR_SYM) }
CURSOR_NAME { GENERATE_TOKEN(CURSOR_NAME_SYM) }
CURRENT_TIME { GENERATE_TOKEN(CURTIME) }
CURRENT_TIMESTAMP { GENERATE_TOKEN(NOW_SYM) }
DATABASE { GENERATE_TOKEN(DATABASE) }
DATABASES { GENERATE_TOKEN(DATABASES) }
SCHEMAS { GENERATE_TOKEN(DATABASES) }
DATAFILE { GENERATE_TOKEN(DATAFILE_SYM) }
DATA { GENERATE_TOKEN(DATA_SYM) }
DATETIME { GENERATE_TOKEN(DATETIME_SYM) }
DATE_ADD { GENERATE_TOKEN(DATE_ADD_INTERVAL) }
DATE_SUB { GENERATE_TOKEN(DATE_SUB_INTERVAL) }
DATE { GENERATE_TOKEN(DATE_SYM) }
DAY_HOUR { GENERATE_TOKEN(DAY_HOUR_SYM) }
DAY_MICROSECOND { GENERATE_TOKEN(DAY_MICROSECOND_SYM) }
DAY_MINUTE { GENERATE_TOKEN(DAY_MINUTE_SYM) }
DAY_SECOND { GENERATE_TOKEN(DAY_SECOND_SYM) }
DAY { GENERATE_TOKEN(DAY_SYM) }
DEALLOCATE { GENERATE_TOKEN(DEALLOCATE_SYM) }
DECIMAL_NUM { GENERATE_TOKEN(DECIMAL_NUM) }
DECIMAL { GENERATE_TOKEN(DECIMAL_SYM) }
DECLARE { GENERATE_TOKEN(DECLARE_SYM) }
DEFAULT { GENERATE_TOKEN(DEFAULT_SYM) }
DEFAULT_AUTH { GENERATE_TOKEN(DEFAULT_AUTH_SYM) }
DEFINER { GENERATE_TOKEN(DEFINER_SYM) }
DELAYED { GENERATE_TOKEN(DELAYED_SYM) }
DELAY_KEY_WRITE { GENERATE_TOKEN(DELAY_KEY_WRITE_SYM) }
DELETE { GENERATE_TOKEN(DELETE_SYM) }
DESC { GENERATE_TOKEN(DESC) }
DESCRIBE { GENERATE_TOKEN(DESCRIBE) }
OBSOLETE_TOKEN_388 { GENERATE_TOKEN(OBSOLETE_TOKEN_388) }
DETERMINISTIC { GENERATE_TOKEN(DETERMINISTIC_SYM) }
DIAGNOSTICS { GENERATE_TOKEN(DIAGNOSTICS_SYM) }
BYTEDICT { GENERATE_TOKEN(BYTEDICT_SYM) }
SHORTDICT { GENERATE_TOKEN(SHORTDICT_SYM) }
INTDICT { GENERATE_TOKEN(INTDICT_SYM) }
DICT_INDEX { GENERATE_TOKEN(DICT_INDEX_SYM) }
DIRECTORY { GENERATE_TOKEN(DIRECTORY_SYM) }
DISABLE { GENERATE_TOKEN(DISABLE_SYM) }
DISCARD { GENERATE_TOKEN(DISCARD_SYM) }
DISK { GENERATE_TOKEN(DISK_SYM) }
DISTINCT { GENERATE_TOKEN(DISTINCT) }
DIV { GENERATE_TOKEN(DIV_SYM) }
DOUBLE { GENERATE_TOKEN(DOUBLE_SYM) }
DO { GENERATE_TOKEN(DO_SYM) }
DROP { GENERATE_TOKEN(DROP) }
DUAL { GENERATE_TOKEN(DUAL_SYM) }
DUMPFILE { GENERATE_TOKEN(DUMPFILE) }
DUPLICATE { GENERATE_TOKEN(DUPLICATE_SYM) }
DYNAMIC { GENERATE_TOKEN(DYNAMIC_SYM) }
EACH { GENERATE_TOKEN(EACH_SYM) }
ELSE { GENERATE_TOKEN(ELSE) }
ELSEIF { GENERATE_TOKEN(ELSEIF_SYM) }
ENABLE { GENERATE_TOKEN(ENABLE_SYM) }
ENCLOSED { GENERATE_TOKEN(ENCLOSED) }
ENCODING { GENERATE_TOKEN(ENCODING) }
END { GENERATE_TOKEN(END) }
ENDS { GENERATE_TOKEN(ENDS_SYM) }
END_OF_INPUT { GENERATE_TOKEN(END_OF_INPUT) }
ENGINES { GENERATE_TOKEN(ENGINES_SYM) }
ENGINE { GENERATE_TOKEN(ENGINE_SYM) }
ENUM { GENERATE_TOKEN(ENUM_SYM) }
ERROR { GENERATE_TOKEN(ERROR_SYM) }
ERRORS { GENERATE_TOKEN(ERRORS) }
ESCAPED { GENERATE_TOKEN(ESCAPED) }
ESCAPE { GENERATE_TOKEN(ESCAPE_SYM) }
EVENTS { GENERATE_TOKEN(EVENTS_SYM) }
EVENT { GENERATE_TOKEN(EVENT_SYM) }
EVERY { GENERATE_TOKEN(EVERY_SYM) }
EXCHANGE { GENERATE_TOKEN(EXCHANGE_SYM) }
EXECUTE { GENERATE_TOKEN(EXECUTE_SYM) }
EXISTS { GENERATE_TOKEN(EXISTS) }
EXIT { GENERATE_TOKEN(EXIT_SYM) }
EXPANSION { GENERATE_TOKEN(EXPANSION_SYM) }
EXPIRE { GENERATE_TOKEN(EXPIRE_SYM) }
EXPORT { GENERATE_TOKEN(EXPORT_SYM) }
EXTENDED { GENERATE_TOKEN(EXTENDED_SYM) }
EXTENT_SIZE { GENERATE_TOKEN(EXTENT_SIZE_SYM) }
EXTRACT { GENERATE_TOKEN(EXTRACT_SYM) }
FALSE { GENERATE_TOKEN(FALSE_SYM) }
FAST { GENERATE_TOKEN(FAST_SYM) }
FAULTS { GENERATE_TOKEN(FAULTS_SYM) }
FIELDS { GENERATE_TOKEN(COLUMNS) }
FETCH { GENERATE_TOKEN(FETCH_SYM) }
FILE { GENERATE_TOKEN(FILE_SYM) }
FILE_BLOCK_SIZE { GENERATE_TOKEN(FILE_BLOCK_SIZE_SYM) }
FILTER { GENERATE_TOKEN(FILTER_SYM) }
FIRST { GENERATE_TOKEN(FIRST_SYM) }
FIXED { GENERATE_TOKEN(FIXED_SYM) }
FLOAT_NUM { GENERATE_TOKEN(FLOAT_NUM) }
FLOAT { GENERATE_TOKEN(FLOAT_SYM) }
FLUSH { GENERATE_TOKEN(FLUSH_SYM) }
FOLLOWS { GENERATE_TOKEN(FOLLOWS_SYM) }
FORCE { GENERATE_TOKEN(FORCE_SYM) }
FOREIGN { GENERATE_TOKEN(FOREIGN) }
FOR { GENERATE_TOKEN(FOR_SYM) }
FORMAT { GENERATE_TOKEN(FORMAT_SYM) }
FOUND { GENERATE_TOKEN(FOUND_SYM) }
FROM { GENERATE_TOKEN(FROM) }
FULL { GENERATE_TOKEN(FULL) }
FULLTEXT { GENERATE_TOKEN(FULLTEXT_SYM) }
FUNCTION { GENERATE_TOKEN(FUNCTION_SYM) }
GENERAL { GENERATE_TOKEN(GENERAL) }
GENERATED { GENERATE_TOKEN(GENERATED) }
GROUP_REPLICATION { GENERATE_TOKEN(GROUP_REPLICATION) }
GEOMETRYCOLLECTION { GENERATE_TOKEN(GEOMETRYCOLLECTION_SYM) }
GEOMETRY { GENERATE_TOKEN(GEOMETRY_SYM) }
GET_FORMAT { GENERATE_TOKEN(GET_FORMAT) }
GET { GENERATE_TOKEN(GET_SYM) }
GLOBAL { GENERATE_TOKEN(GLOBAL_SYM) }
GRANT { GENERATE_TOKEN(GRANT) }
GRANTS { GENERATE_TOKEN(GRANTS) }
GROUP { GENERATE_TOKEN(GROUP_SYM) }
GROUP_CONCAT { GENERATE_TOKEN(GROUP_CONCAT_SYM) }
HANDLER { GENERATE_TOKEN(HANDLER_SYM) }
HASH { GENERATE_TOKEN(HASH_SYM) }
HAVING { GENERATE_TOKEN(HAVING) }
HELP { GENERATE_TOKEN(HELP_SYM) }
HEX_NUM { GENERATE_TOKEN(HEX_NUM) }
HIGH_PRIORITY { GENERATE_TOKEN(HIGH_PRIORITY) }
HOST { GENERATE_TOKEN(HOST_SYM) }
HOSTS { GENERATE_TOKEN(HOSTS_SYM) }
HOUR_MICROSECOND { GENERATE_TOKEN(HOUR_MICROSECOND_SYM) }
HOUR_MINUTE { GENERATE_TOKEN(HOUR_MINUTE_SYM) }
HOUR_SECOND { GENERATE_TOKEN(HOUR_SECOND_SYM) }
HOUR { GENERATE_TOKEN(HOUR_SYM) }
IDENTIFIED { GENERATE_TOKEN(IDENTIFIED_SYM) }
IDENT_QUOTED { GENERATE_TOKEN(IDENT_QUOTED) }
IF { GENERATE_TOKEN(IF) }
IGNORE { GENERATE_TOKEN(IGNORE_SYM) }
IGNORE_SERVER_IDS { GENERATE_TOKEN(IGNORE_SERVER_IDS_SYM) }
IMPORT { GENERATE_TOKEN(IMPORT) }
INDEXES { GENERATE_TOKEN(INDEXES) }
INDEX { GENERATE_TOKEN(INDEX_SYM) }
INFILE { GENERATE_TOKEN(INFILE) }
INITIAL_SIZE { GENERATE_TOKEN(INITIAL_SIZE_SYM) }
INNER { GENERATE_TOKEN(INNER_SYM) }
INOUT { GENERATE_TOKEN(INOUT_SYM) }
INSENSITIVE { GENERATE_TOKEN(INSENSITIVE_SYM) }
INSERT { GENERATE_TOKEN(INSERT_SYM) }
INSERT_METHOD { GENERATE_TOKEN(INSERT_METHOD) }
INSTANCE { GENERATE_TOKEN(INSTANCE_SYM) }
INSTALL { GENERATE_TOKEN(INSTALL_SYM) }
INTERVAL { GENERATE_TOKEN(INTERVAL_SYM) }
INTO { GENERATE_TOKEN(INTO) }
INT { GENERATE_TOKEN(INT_SYM) }
INTEGER { GENERATE_TOKEN(INTEGER_SYM) }
INVOKER { GENERATE_TOKEN(INVOKER_SYM) }
IN { GENERATE_TOKEN(IN_SYM) }
IO_AFTER_GTIDS { GENERATE_TOKEN(IO_AFTER_GTIDS) }
IO_BEFORE_GTIDS { GENERATE_TOKEN(IO_BEFORE_GTIDS) }
IO { GENERATE_TOKEN(IO_SYM) }
IPC { GENERATE_TOKEN(IPC_SYM) }
IS { GENERATE_TOKEN(IS) }
ISOLATION { GENERATE_TOKEN(ISOLATION) }
ISSUER { GENERATE_TOKEN(ISSUER_SYM) }
ITERATE { GENERATE_TOKEN(ITERATE_SYM) }
JOIN { 
    GENERATE_TOKEN(JOIN_SYM);
}
JSON_SEPARATOR { GENERATE_TOKEN(JSON_SEPARATOR_SYM) }
JSON { GENERATE_TOKEN(JSON_SYM) }
KEYS { GENERATE_TOKEN(KEYS) }
KEY_BLOCK_SIZE { GENERATE_TOKEN(KEY_BLOCK_SIZE) }
KEY { GENERATE_TOKEN(KEY_SYM) }
KILL { GENERATE_TOKEN(KILL_SYM) }
LANGUAGE { GENERATE_TOKEN(LANGUAGE_SYM) }
LAST { GENERATE_TOKEN(LAST_SYM) }
LEADING { GENERATE_TOKEN(LEADING) }
LEAVES { GENERATE_TOKEN(LEAVES) }
LEAVE { GENERATE_TOKEN(LEAVE_SYM) }
LEFT { GENERATE_TOKEN(LEFT) }
LESS { GENERATE_TOKEN(LESS_SYM) }
LEVEL { GENERATE_TOKEN(LEVEL_SYM) }
LIKE { GENERATE_TOKEN(LIKE) }
LIMIT { GENERATE_TOKEN(LIMIT) }
LINEAR { GENERATE_TOKEN(LINEAR_SYM) }
LINES { GENERATE_TOKEN(LINES) }
LINESTRING { GENERATE_TOKEN(LINESTRING_SYM) }
LIST { GENERATE_TOKEN(LIST_SYM) }
LOAD { GENERATE_TOKEN(LOAD) }
LOCAL { GENERATE_TOKEN(LOCAL_SYM) }
LOCALTIME { GENERATE_TOKEN(NOW_SYM) }
LOCALTIMESTAMP { GENERATE_TOKEN(NOW_SYM) }
OBSOLETE_TOKEN_538 { GENERATE_TOKEN(OBSOLETE_TOKEN_538) }
LOCKS { GENERATE_TOKEN(LOCKS_SYM) }
LOCK { GENERATE_TOKEN(LOCK_SYM) }
LOGFILE { GENERATE_TOKEN(LOGFILE_SYM) }
LOGS { GENERATE_TOKEN(LOGS_SYM) }
LONGBLOB { GENERATE_TOKEN(LONGBLOB_SYM) }
LONGTEXT { GENERATE_TOKEN(LONGTEXT_SYM) }
LONG_NUM { GENERATE_TOKEN(LONG_NUM) }
LONG { GENERATE_TOKEN(LONG_SYM) }
LOOP { GENERATE_TOKEN(LOOP_SYM) }
LOW_PRIORITY { GENERATE_TOKEN(LOW_PRIORITY) }
MASTER_AUTO_POSITION { GENERATE_TOKEN(MASTER_AUTO_POSITION_SYM) }
MASTER_BIND { GENERATE_TOKEN(MASTER_BIND_SYM) }
MASTER_CONNECT_RETRY { GENERATE_TOKEN(MASTER_CONNECT_RETRY_SYM) }
MASTER_DELAY { GENERATE_TOKEN(MASTER_DELAY_SYM) }
MASTER_HOST { GENERATE_TOKEN(MASTER_HOST_SYM) }
MASTER_LOG_FILE { GENERATE_TOKEN(MASTER_LOG_FILE_SYM) }
MASTER_LOG_POS { GENERATE_TOKEN(MASTER_LOG_POS_SYM) }
MASTER_PASSWORD { GENERATE_TOKEN(MASTER_PASSWORD_SYM) }
MASTER_PORT { GENERATE_TOKEN(MASTER_PORT_SYM) }
MASTER_RETRY_COUNT { GENERATE_TOKEN(MASTER_RETRY_COUNT_SYM) }
MASTER_SERVER_ID { GENERATE_TOKEN(MASTER_SERVER_ID_SYM) }
MASTER_SSL_CAPATH { GENERATE_TOKEN(MASTER_SSL_CAPATH_SYM) }
MASTER_TLS_VERSION { GENERATE_TOKEN(MASTER_TLS_VERSION_SYM) }
MASTER_SSL_CA { GENERATE_TOKEN(MASTER_SSL_CA_SYM) }
MASTER_SSL_CERT { GENERATE_TOKEN(MASTER_SSL_CERT_SYM) }
MASTER_SSL_CIPHER { GENERATE_TOKEN(MASTER_SSL_CIPHER_SYM) }
MASTER_SSL_CRL { GENERATE_TOKEN(MASTER_SSL_CRL_SYM) }
MASTER_SSL_CRLPATH { GENERATE_TOKEN(MASTER_SSL_CRLPATH_SYM) }
MASTER_SSL_KEY { GENERATE_TOKEN(MASTER_SSL_KEY_SYM) }
MASTER_SSL { GENERATE_TOKEN(MASTER_SSL_SYM) }
MASTER_SSL_VERIFY_SERVER_CERT { GENERATE_TOKEN(MASTER_SSL_VERIFY_SERVER_CERT_SYM) }
MASTER { GENERATE_TOKEN(MASTER_SYM) }
MASTER_USER { GENERATE_TOKEN(MASTER_USER_SYM) }
MASTER_HEARTBEAT_PERIOD { GENERATE_TOKEN(MASTER_HEARTBEAT_PERIOD_SYM) }
MATCH { GENERATE_TOKEN(MATCH) }
MAX_CONNECTIONS_PER_HOUR { GENERATE_TOKEN(MAX_CONNECTIONS_PER_HOUR) }
MAX_QUERIES_PER_HOUR { GENERATE_TOKEN(MAX_QUERIES_PER_HOUR) }
MAX_ROWS { GENERATE_TOKEN(MAX_ROWS) }
MAX_SIZE { GENERATE_TOKEN(MAX_SIZE_SYM) }
MAX { GENERATE_TOKEN(MAX_SYM) }
MAX_UPDATES_PER_HOUR { GENERATE_TOKEN(MAX_UPDATES_PER_HOUR) }
MAX_USER_CONNECTIONS { GENERATE_TOKEN(MAX_USER_CONNECTIONS_SYM) }
MAXVALUE { GENERATE_TOKEN(MAX_VALUE_SYM) }
MEDIUMBLOB { GENERATE_TOKEN(MEDIUMBLOB_SYM) }
MEDIUMINT { GENERATE_TOKEN(MEDIUMINT_SYM) }
MEDIUMTEXT { GENERATE_TOKEN(MEDIUMTEXT_SYM) }
MEDIUM { GENERATE_TOKEN(MEDIUM_SYM) }
MEMORY { GENERATE_TOKEN(MEMORY_SYM) }
MERGE { GENERATE_TOKEN(MERGE_SYM) }
MESSAGE_TEXT { GENERATE_TOKEN(MESSAGE_TEXT_SYM) }
MICROSECOND { GENERATE_TOKEN(MICROSECOND_SYM) }
MIGRATE { GENERATE_TOKEN(MIGRATE_SYM) }
MINUTE_MICROSECOND { GENERATE_TOKEN(MINUTE_MICROSECOND_SYM) }
MINUTE_SECOND { GENERATE_TOKEN(MINUTE_SECOND_SYM) }
MINUTE { GENERATE_TOKEN(MINUTE_SYM) }
MIN_ROWS { GENERATE_TOKEN(MIN_ROWS) }
MIN { GENERATE_TOKEN(MIN_SYM) }
MODE { GENERATE_TOKEN(MODE_SYM) }
MODIFIES { GENERATE_TOKEN(MODIFIES_SYM) }
MODIFY { GENERATE_TOKEN(MODIFY_SYM) }
MOD { GENERATE_TOKEN(MOD_SYM) }
MONTH { GENERATE_TOKEN(MONTH_SYM) }
MULTILINESTRING { GENERATE_TOKEN(MULTILINESTRING_SYM) }
MULTIPOINT { GENERATE_TOKEN(MULTIPOINT_SYM) }
MULTIPOLYGON { GENERATE_TOKEN(MULTIPOLYGON_SYM) }
MUTEX { GENERATE_TOKEN(MUTEX_SYM) }
MYSQL_ERRNO { GENERATE_TOKEN(MYSQL_ERRNO_SYM) }
NAMES { GENERATE_TOKEN(NAMES_SYM) }
NAME { GENERATE_TOKEN(NAME_SYM) }
NATIONAL { GENERATE_TOKEN(NATIONAL_SYM) }
NATURAL { GENERATE_TOKEN(NATURAL) }
NCHAR_STRING { GENERATE_TOKEN(NCHAR_STRING) }
NCHAR { GENERATE_TOKEN(NCHAR_SYM) }
NDBCLUSTER { GENERATE_TOKEN(NDBCLUSTER_SYM) }
NEG { GENERATE_TOKEN(NEG) }
NEVER { GENERATE_TOKEN(NEVER_SYM) }
NEW { GENERATE_TOKEN(NEW_SYM) }
NEXT { GENERATE_TOKEN(NEXT_SYM) }
NODEGROUP { GENERATE_TOKEN(NODEGROUP_SYM) }
NONE { GENERATE_TOKEN(NONE_SYM) }
NOT2 { GENERATE_TOKEN(NOT2_SYM) }
NOT { GENERATE_TOKEN(NOT_SYM) }
NOW { GENERATE_TOKEN(NOW_SYM) }
NO { GENERATE_TOKEN(NO_SYM) }
NO_WAIT { GENERATE_TOKEN(NO_WAIT_SYM) }
NO_WRITE_TO_BINLOG { GENERATE_TOKEN(NO_WRITE_TO_BINLOG) }
NULL { GENERATE_TOKEN(NULL_SYM) }
NUMBER { GENERATE_TOKEN(NUMBER_SYM) }
NUMERIC { GENERATE_TOKEN(NUMERIC_SYM) }
NVARCHAR { GENERATE_TOKEN(NVARCHAR_SYM) }
OFFSET { GENERATE_TOKEN(OFFSET_SYM) }
ON { GENERATE_TOKEN(ON_SYM) }
ONE { GENERATE_TOKEN(ONE_SYM) }
ONLY { GENERATE_TOKEN(ONLY_SYM) }
OPEN { GENERATE_TOKEN(OPEN_SYM) }
OPTIMIZE { GENERATE_TOKEN(OPTIMIZE) }
OPTIMIZER_COSTS { GENERATE_TOKEN(OPTIMIZER_COSTS_SYM) }
OPTIONS { GENERATE_TOKEN(OPTIONS_SYM) }
OPTION { GENERATE_TOKEN(OPTION) }
OPTIONALLY { GENERATE_TOKEN(OPTIONALLY) }
OR2 { GENERATE_TOKEN(OR2_SYM) }
ORDER { GENERATE_TOKEN(ORDER_SYM) }
OR_OR { GENERATE_TOKEN(OR_OR_SYM) }
OR { GENERATE_TOKEN(OR_SYM) }
OUTER { GENERATE_TOKEN(OUTER) }
OUTFILE { GENERATE_TOKEN(OUTFILE) }
OUT { GENERATE_TOKEN(OUT_SYM) }
OWNER { GENERATE_TOKEN(OWNER_SYM) }
PACK_KEYS { GENERATE_TOKEN(PACK_KEYS_SYM) }
PAGE { GENERATE_TOKEN(PAGE_SYM) }
PARSER { GENERATE_TOKEN(PARSER_SYM) }
OBSOLETE_TOKEN_654 { GENERATE_TOKEN(OBSOLETE_TOKEN_654) }
PARTIAL { GENERATE_TOKEN(PARTIAL) }
PARTITION { GENERATE_TOKEN(PARTITION_SYM) }
PARTITIONS { GENERATE_TOKEN(PARTITIONS_SYM) }
PARTITIONING { GENERATE_TOKEN(PARTITIONING_SYM) }
PASSWORD { GENERATE_TOKEN(PASSWORD) }
PHASE { GENERATE_TOKEN(PHASE_SYM) }
PLUGIN_DIR { GENERATE_TOKEN(PLUGIN_DIR_SYM) }
PLUGIN { GENERATE_TOKEN(PLUGIN_SYM) }
PLUGINS { GENERATE_TOKEN(PLUGINS_SYM) }
POINT { GENERATE_TOKEN(POINT_SYM) }
POLYGON { GENERATE_TOKEN(POLYGON_SYM) }
PORT { GENERATE_TOKEN(PORT_SYM) }
POSITION { GENERATE_TOKEN(POSITION_SYM) }
PRECEDES { GENERATE_TOKEN(PRECEDES_SYM) }
PRECISION { GENERATE_TOKEN(PRECISION) }
PREPARE { GENERATE_TOKEN(PREPARE_SYM) }
PRESERVE { GENERATE_TOKEN(PRESERVE_SYM) }
PREV { GENERATE_TOKEN(PREV_SYM) }
PRIMARY { GENERATE_TOKEN(PRIMARY_SYM) }
PRIVILEGES { GENERATE_TOKEN(PRIVILEGES) }
PROCEDURE { GENERATE_TOKEN(PROCEDURE_SYM) }
PROCESS { GENERATE_TOKEN(PROCESS) }
PROCESSLIST { GENERATE_TOKEN(PROCESSLIST_SYM) }
PROFILE { GENERATE_TOKEN(PROFILE_SYM) }
PROFILES { GENERATE_TOKEN(PROFILES_SYM) }
PROXY { GENERATE_TOKEN(PROXY_SYM) }
PURGE { GENERATE_TOKEN(PURGE) }
QUARTER { GENERATE_TOKEN(QUARTER_SYM) }
QUERY { GENERATE_TOKEN(QUERY_SYM) }
QUICK { GENERATE_TOKEN(QUICK) }
RANGE { GENERATE_TOKEN(RANGE_SYM) }
READS { GENERATE_TOKEN(READS_SYM) }
READ_ONLY { GENERATE_TOKEN(READ_ONLY_SYM) }
READ { GENERATE_TOKEN(READ_SYM) }
READ_WRITE { GENERATE_TOKEN(READ_WRITE_SYM) }
REAL { GENERATE_TOKEN(REAL_SYM) }
REBUILD { GENERATE_TOKEN(REBUILD_SYM) }
RECOVER { GENERATE_TOKEN(RECOVER_SYM) }
OBSOLETE_TOKEN_693 { GENERATE_TOKEN(OBSOLETE_TOKEN_693) }
REDO_BUFFER_SIZE { GENERATE_TOKEN(REDO_BUFFER_SIZE_SYM) }
REDUNDANT { GENERATE_TOKEN(REDUNDANT_SYM) }
REFERENCES { GENERATE_TOKEN(REFERENCES) }
REGEXP { GENERATE_TOKEN(REGEXP) }
RELAY { GENERATE_TOKEN(RELAY) }
RELAYLOG { GENERATE_TOKEN(RELAYLOG_SYM) }
RELAY_LOG_FILE { GENERATE_TOKEN(RELAY_LOG_FILE_SYM) }
RELAY_LOG_POS { GENERATE_TOKEN(RELAY_LOG_POS_SYM) }
RELAY_THREAD { GENERATE_TOKEN(RELAY_THREAD) }
RELEASE { GENERATE_TOKEN(RELEASE_SYM) }
RELOAD { GENERATE_TOKEN(RELOAD) }
REMOVE { GENERATE_TOKEN(REMOVE_SYM) }
RENAME { GENERATE_TOKEN(RENAME) }
REORGANIZE { GENERATE_TOKEN(REORGANIZE_SYM) }
REPAIR { GENERATE_TOKEN(REPAIR) }
REPEATABLE { GENERATE_TOKEN(REPEATABLE_SYM) }
REPEAT { GENERATE_TOKEN(REPEAT_SYM) }
REPLACE { GENERATE_TOKEN(REPLACE_SYM) }
REPLICATION { GENERATE_TOKEN(REPLICATION) }
REPLICATE_DO_DB { GENERATE_TOKEN(REPLICATE_DO_DB) }
REPLICATE_IGNORE_DB { GENERATE_TOKEN(REPLICATE_IGNORE_DB) }
REPLICATE_DO_TABLE { GENERATE_TOKEN(REPLICATE_DO_TABLE) }
REPLICATE_IGNORE_TABLE { GENERATE_TOKEN(REPLICATE_IGNORE_TABLE) }
REPLICATE_WILD_DO_TABLE { GENERATE_TOKEN(REPLICATE_WILD_DO_TABLE) }
REPLICATE_WILD_IGNORE_TABLE { GENERATE_TOKEN(REPLICATE_WILD_IGNORE_TABLE) }
REPLICATE_REWRITE_DB { GENERATE_TOKEN(REPLICATE_REWRITE_DB) }
REQUIRE { GENERATE_TOKEN(REQUIRE_SYM) }
RESET { GENERATE_TOKEN(RESET_SYM) }
RESIGNAL { GENERATE_TOKEN(RESIGNAL_SYM) }
RESOURCES { GENERATE_TOKEN(RESOURCES) }
RESTORE { GENERATE_TOKEN(RESTORE_SYM) }
RESTRICT { GENERATE_TOKEN(RESTRICT) }
RESUME { GENERATE_TOKEN(RESUME_SYM) }
RETURNED_SQLSTATE { GENERATE_TOKEN(RETURNED_SQLSTATE_SYM) }
RETURNS { GENERATE_TOKEN(RETURNS_SYM) }
RETURN { GENERATE_TOKEN(RETURN_SYM) }
REVERSE { GENERATE_TOKEN(REVERSE_SYM) }
REVOKE { GENERATE_TOKEN(REVOKE) }
RIGHT { GENERATE_TOKEN(RIGHT) }
ROLLBACK { GENERATE_TOKEN(ROLLBACK_SYM) }
ROLLUP { GENERATE_TOKEN(ROLLUP_SYM) }
ROTATE { GENERATE_TOKEN(ROTATE_SYM) }
ROUTINE { GENERATE_TOKEN(ROUTINE_SYM) }
ROWS { GENERATE_TOKEN(ROWS_SYM) }
ROW_FORMAT { GENERATE_TOKEN(ROW_FORMAT_SYM) }
ROW { GENERATE_TOKEN(ROW_SYM) }
ROW_COUNT { GENERATE_TOKEN(ROW_COUNT_SYM) }
RTREE { GENERATE_TOKEN(RTREE_SYM) }
SAVEPOINT { GENERATE_TOKEN(SAVEPOINT_SYM) }
SCHEDULE { GENERATE_TOKEN(SCHEDULE_SYM) }
SCHEMA { GENERATE_TOKEN(SCHEMA) }
SCHEMA_NAME { GENERATE_TOKEN(SCHEMA_NAME_SYM) }
SECOND_MICROSECOND { GENERATE_TOKEN(SECOND_MICROSECOND_SYM) }
SECOND { GENERATE_TOKEN(SECOND_SYM) }
SECURITY { GENERATE_TOKEN(SECURITY_SYM) }
SELECT { GENERATE_TOKEN(SELECT_SYM) }
SENSITIVE { GENERATE_TOKEN(SENSITIVE_SYM) }
SEPARATOR { GENERATE_TOKEN(SEPARATOR_SYM) }
SERIALIZABLE { GENERATE_TOKEN(SERIALIZABLE_SYM) }
SERIAL { GENERATE_TOKEN(SERIAL_SYM) }
SESSION { GENERATE_TOKEN(SESSION_SYM) }
SERVER { GENERATE_TOKEN(SERVER_SYM) }
OBSOLETE_TOKEN_755 { GENERATE_TOKEN(OBSOLETE_TOKEN_755) }
SET { GENERATE_TOKEN(SET) }
SET_VAR { GENERATE_TOKEN(SET_VAR) }
SHARE { GENERATE_TOKEN(SHARE_SYM) }
SHARES { GENERATE_TOKEN(SHARES_SYM) }
SHOW { GENERATE_TOKEN(SHOW) }
SHUTDOWN { GENERATE_TOKEN(SHUTDOWN) }
SIGNAL { GENERATE_TOKEN(SIGNAL_SYM) }
SIGNED { GENERATE_TOKEN(SIGNED_SYM) }
SIMPLE { GENERATE_TOKEN(SIMPLE_SYM) }
SLAVE { GENERATE_TOKEN(SLAVE) }
SLOW { GENERATE_TOKEN(SLOW) }
SMALLINT { GENERATE_TOKEN(SMALLINT_SYM) }
SNAPSHOT { GENERATE_TOKEN(SNAPSHOT_SYM) }
SOCKET { GENERATE_TOKEN(SOCKET_SYM) }
SONAME { GENERATE_TOKEN(SONAME_SYM) }
SOUNDS { GENERATE_TOKEN(SOUNDS_SYM) }
SOURCE { GENERATE_TOKEN(SOURCE_SYM) }
SPATIAL { GENERATE_TOKEN(SPATIAL_SYM) }
SPECIFIC { GENERATE_TOKEN(SPECIFIC_SYM) }
SQLEXCEPTION { GENERATE_TOKEN(SQLEXCEPTION_SYM) }
SQLSTATE { GENERATE_TOKEN(SQLSTATE_SYM) }
SQLWARNING { GENERATE_TOKEN(SQLWARNING_SYM) }
SQL_AFTER_GTIDS { GENERATE_TOKEN(SQL_AFTER_GTIDS) }
SQL_AFTER_MTS_GAPS { GENERATE_TOKEN(SQL_AFTER_MTS_GAPS) }
SQL_BEFORE_GTIDS { GENERATE_TOKEN(SQL_BEFORE_GTIDS) }
SQL_BIG_RESULT { GENERATE_TOKEN(SQL_BIG_RESULT) }
SQL_BUFFER_RESULT { GENERATE_TOKEN(SQL_BUFFER_RESULT) }
OBSOLETE_TOKEN_784 { GENERATE_TOKEN(OBSOLETE_TOKEN_784) }
SQL_CALC_FOUND_ROWS { GENERATE_TOKEN(SQL_CALC_FOUND_ROWS) }
SQL_NO_CACHE { GENERATE_TOKEN(SQL_NO_CACHE_SYM) }
SQL_SMALL_RESULT { GENERATE_TOKEN(SQL_SMALL_RESULT) }
SQL { GENERATE_TOKEN(SQL_SYM) }
SQL_THREAD { GENERATE_TOKEN(SQL_THREAD) }
SSL { GENERATE_TOKEN(SSL_SYM) }
STACKED { GENERATE_TOKEN(STACKED_SYM) }
STARTING { GENERATE_TOKEN(STARTING) }
STARTS { GENERATE_TOKEN(STARTS_SYM) }
START { GENERATE_TOKEN(START_SYM) }
STATS_AUTO_RECALC { GENERATE_TOKEN(STATS_AUTO_RECALC_SYM) }
STATS_PERSISTENT { GENERATE_TOKEN(STATS_PERSISTENT_SYM) }
STATS_SAMPLE_PAGES { GENERATE_TOKEN(STATS_SAMPLE_PAGES_SYM) }
STATUS { GENERATE_TOKEN(STATUS_SYM) }
STDDEV_SAMP { GENERATE_TOKEN(STDDEV_SAMP_SYM) }
STD { GENERATE_TOKEN(STD_SYM) }
STOP { GENERATE_TOKEN(STOP_SYM) }
STORAGE { GENERATE_TOKEN(STORAGE_SYM) }
STORED { GENERATE_TOKEN(STORED_SYM) }
STRAIGHT_JOIN { GENERATE_TOKEN(STRAIGHT_JOIN) }
STRING { GENERATE_TOKEN(STRING_SYM) }
SUBCLASS_ORIGIN { GENERATE_TOKEN(SUBCLASS_ORIGIN_SYM) }
SUBDATE { GENERATE_TOKEN(SUBDATE_SYM) }
SUBJECT { GENERATE_TOKEN(SUBJECT_SYM) }
SUBPARTITIONS { GENERATE_TOKEN(SUBPARTITIONS_SYM) }
SUBPARTITION { GENERATE_TOKEN(SUBPARTITION_SYM) }
SUBSTRING { GENERATE_TOKEN(SUBSTRING) }
SUM { GENERATE_TOKEN(SUM_SYM) }
SUPER { GENERATE_TOKEN(SUPER_SYM) }
SUSPEND { GENERATE_TOKEN(SUSPEND_SYM) }
SWAPS { GENERATE_TOKEN(SWAPS_SYM) }
SWITCHES { GENERATE_TOKEN(SWITCHES_SYM) }
SYSDATE { GENERATE_TOKEN(SYSDATE) }
TABLES { GENERATE_TOKEN(TABLES) }
VIEWS { GENERATE_TOKEN(VIEWS) }
TABLESPACE { GENERATE_TOKEN(TABLESPACE_SYM) }
OBSOLETE_TOKEN_820 { GENERATE_TOKEN(OBSOLETE_TOKEN_820) }
TABLE { GENERATE_TOKEN(TABLE_SYM) }
TABLE_CHECKSUM { GENERATE_TOKEN(TABLE_CHECKSUM_SYM) }
TABLE_NAME { GENERATE_TOKEN(TABLE_NAME_SYM) }
TEMPORARY { GENERATE_TOKEN(TEMPORARY) }
TEMPTABLE { GENERATE_TOKEN(TEMPTABLE_SYM) }
TERMINATED { GENERATE_TOKEN(TERMINATED) }
TEXT { GENERATE_TOKEN(TEXT_SYM) }
THAN { GENERATE_TOKEN(THAN_SYM) }
THEN { GENERATE_TOKEN(THEN_SYM) }
TIMESTAMP { GENERATE_TOKEN(TIMESTAMP_SYM) }
TIMESTAMP_ADD { GENERATE_TOKEN(TIMESTAMP_ADD) }
TIMESTAMP_DIFF { GENERATE_TOKEN(TIMESTAMP_DIFF) }
TIME { GENERATE_TOKEN(TIME_SYM) }
TINYBLOB { GENERATE_TOKEN(TINYBLOB_SYM) }
TINYINT { GENERATE_TOKEN(TINYINT_SYM) }
TINYTEXT_SYN { GENERATE_TOKEN(TINYTEXT_SYN) }
TO { GENERATE_TOKEN(TO_SYM) }
TRAILING { GENERATE_TOKEN(TRAILING) }
TRANSACTION { GENERATE_TOKEN(TRANSACTION_SYM) }
TRIGGERS { GENERATE_TOKEN(TRIGGERS_SYM) }
TRIGGER { GENERATE_TOKEN(TRIGGER_SYM) }
TRIM { GENERATE_TOKEN(TRIM) }
TRUE { GENERATE_TOKEN(TRUE_SYM) }
TRUNCATE { GENERATE_TOKEN(TRUNCATE_SYM) }
TYPES { GENERATE_TOKEN(TYPES_SYM) }
TYPE { GENERATE_TOKEN(TYPE_SYM) }
OBSOLETE_TOKEN_848 { GENERATE_TOKEN(OBSOLETE_TOKEN_848) }
ULONGLONG_NUM { GENERATE_TOKEN(ULONGLONG_NUM) }
UNCOMMITTED { GENERATE_TOKEN(UNCOMMITTED_SYM) }
UNDEFINED { GENERATE_TOKEN(UNDEFINED_SYM) }
UNDERSCORE_CHARSET { GENERATE_TOKEN(UNDERSCORE_CHARSET) }
UNDOFILE { GENERATE_TOKEN(UNDOFILE_SYM) }
UNDO_BUFFER_SIZE { GENERATE_TOKEN(UNDO_BUFFER_SIZE_SYM) }
UNDO { GENERATE_TOKEN(UNDO_SYM) }
UNICODE { GENERATE_TOKEN(UNICODE_SYM) }
UNINSTALL { GENERATE_TOKEN(UNINSTALL_SYM) }
UNION { GENERATE_TOKEN(UNION_SYM) }
UNIQUE { GENERATE_TOKEN(UNIQUE_SYM) }
UNKNOWN { GENERATE_TOKEN(UNKNOWN_SYM) }
UNLOCK { GENERATE_TOKEN(UNLOCK_SYM) }
UNSIGNED { GENERATE_TOKEN(UNSIGNED_SYM) }
UNTIL { GENERATE_TOKEN(UNTIL_SYM) }
UPDATE { GENERATE_TOKEN(UPDATE_SYM) }
UPGRADE { GENERATE_TOKEN(UPGRADE_SYM) }
USAGE { GENERATE_TOKEN(USAGE) }
USER { GENERATE_TOKEN(USER) }
USE_FRM { GENERATE_TOKEN(USE_FRM) }
USE { GENERATE_TOKEN(USE_SYM) }
USING { GENERATE_TOKEN(USING) }
UTC_DATE { GENERATE_TOKEN(UTC_DATE_SYM) }
UTC_TIMESTAMP { GENERATE_TOKEN(UTC_TIMESTAMP_SYM) }
UTC_TIME { GENERATE_TOKEN(UTC_TIME_SYM) }
VALIDATION { GENERATE_TOKEN(VALIDATION_SYM) }
VALUES { GENERATE_TOKEN(VALUES) }
VALUE { GENERATE_TOKEN(VALUE_SYM) }
VARBINARY { GENERATE_TOKEN(VARBINARY_SYM) }
VARCHAR { GENERATE_TOKEN(VARCHAR_SYM) }
VARIABLES { GENERATE_TOKEN(VARIABLES) }
VARIANCE { GENERATE_TOKEN(VARIANCE_SYM) }
VARYING { GENERATE_TOKEN(VARYING) }
VAR_SAMP { GENERATE_TOKEN(VAR_SAMP_SYM) }
VERSION { GENERATE_TOKEN(VERSION_SYM) }
VIEW { GENERATE_TOKEN(VIEW_SYM) }
VIRTUAL { GENERATE_TOKEN(VIRTUAL_SYM) }
WAIT { GENERATE_TOKEN(WAIT_SYM) }
WARNINGS { GENERATE_TOKEN(WARNINGS) }
WEEK { GENERATE_TOKEN(WEEK_SYM) }
WEIGHT_STRING { GENERATE_TOKEN(WEIGHT_STRING_SYM) }
WHEN { GENERATE_TOKEN(WHEN_SYM) }
WHERE { GENERATE_TOKEN(WHERE) }
WHILE { GENERATE_TOKEN(WHILE_SYM) }
WITH { GENERATE_TOKEN(WITH) }
OBSOLETE_TOKEN_893 { GENERATE_TOKEN(OBSOLETE_TOKEN_893) }
WITH_ROLLUP { GENERATE_TOKEN(WITH_ROLLUP_SYM) }
WITHOUT { GENERATE_TOKEN(WITHOUT_SYM) }
WORK { GENERATE_TOKEN(WORK_SYM) }
WRAPPER { GENERATE_TOKEN(WRAPPER_SYM) }
WRITE { GENERATE_TOKEN(WRITE_SYM) }
X509 { GENERATE_TOKEN(X509_SYM) }
XA { GENERATE_TOKEN(XA_SYM) }
XID { GENERATE_TOKEN(XID_SYM) }
XML { GENERATE_TOKEN(XML_SYM) }
XOR { GENERATE_TOKEN(XOR) }
YEAR_MONTH { GENERATE_TOKEN(YEAR_MONTH_SYM) }
YEAR { GENERATE_TOKEN(YEAR_SYM) }
ZEROFILL { GENERATE_TOKEN(ZEROFILL_SYM) }
JSON_UNQUOTED_SEPARATOR { GENERATE_TOKEN(JSON_UNQUOTED_SEPARATOR_SYM) }
PERSIST { GENERATE_TOKEN(PERSIST_SYM) }
ROLE { GENERATE_TOKEN(ROLE_SYM) }
ADMIN { GENERATE_TOKEN(ADMIN_SYM) }
INVISIBLE { GENERATE_TOKEN(INVISIBLE_SYM) }
VISIBLE { GENERATE_TOKEN(VISIBLE_SYM) }
EXCEPT { GENERATE_TOKEN(EXCEPT_SYM) }
COMPONENT { GENERATE_TOKEN(COMPONENT_SYM) }
RECURSIVE { GENERATE_TOKEN(RECURSIVE_SYM) }
GRAMMAR_SELECTOR_EXPR { GENERATE_TOKEN(GRAMMAR_SELECTOR_EXPR) }
GRAMMAR_SELECTOR_GCOL { GENERATE_TOKEN(GRAMMAR_SELECTOR_GCOL) }
GRAMMAR_SELECTOR_PART { GENERATE_TOKEN(GRAMMAR_SELECTOR_PART) }
GRAMMAR_SELECTOR_CTE { GENERATE_TOKEN(GRAMMAR_SELECTOR_CTE) }
JSON_OBJECTAGG { GENERATE_TOKEN(JSON_OBJECTAGG) }
JSON_ARRAYAGG { GENERATE_TOKEN(JSON_ARRAYAGG) }
OF { GENERATE_TOKEN(OF_SYM) }
SKIP { GENERATE_TOKEN(SKIP_SYM) }
LOCKED { GENERATE_TOKEN(LOCKED_SYM) }
NOWAIT { GENERATE_TOKEN(NOWAIT_SYM) }
GROUPING { GENERATE_TOKEN(GROUPING_SYM) }
PERSIST_ONLY { GENERATE_TOKEN(PERSIST_ONLY_SYM) }
HISTOGRAM { GENERATE_TOKEN(HISTOGRAM_SYM) }
BUCKETS { GENERATE_TOKEN(BUCKETS_SYM) }
OBSOLETE_TOKEN_930 { GENERATE_TOKEN(OBSOLETE_TOKEN_930) }
CLONE { GENERATE_TOKEN(CLONE_SYM) }
CUME_DIST { GENERATE_TOKEN(CUME_DIST_SYM) }
DENSE_RANK { GENERATE_TOKEN(DENSE_RANK_SYM) }
EXCLUDE { GENERATE_TOKEN(EXCLUDE_SYM) }
FIRST_VALUE { GENERATE_TOKEN(FIRST_VALUE_SYM) }
FOLLOWING { GENERATE_TOKEN(FOLLOWING_SYM) }
GROUPS { GENERATE_TOKEN(GROUPS_SYM) }
LAG { GENERATE_TOKEN(LAG_SYM) }
LAST_VALUE { GENERATE_TOKEN(LAST_VALUE_SYM) }
LEAD { GENERATE_TOKEN(LEAD_SYM) }
NTH_VALUE { GENERATE_TOKEN(NTH_VALUE_SYM) }
NTILE { GENERATE_TOKEN(NTILE_SYM) }
NULLS { GENERATE_TOKEN(NULLS_SYM) }
OTHERS { GENERATE_TOKEN(OTHERS_SYM) }
OVER { GENERATE_TOKEN(OVER_SYM) }
PERCENT_RANK { GENERATE_TOKEN(PERCENT_RANK_SYM) }
PRECEDING { GENERATE_TOKEN(PRECEDING_SYM) }
RANK { GENERATE_TOKEN(RANK_SYM) }
RESPECT { GENERATE_TOKEN(RESPECT_SYM) }
ROW_NUMBER { GENERATE_TOKEN(ROW_NUMBER_SYM) }
TIES { GENERATE_TOKEN(TIES_SYM) }
UNBOUNDED { GENERATE_TOKEN(UNBOUNDED_SYM) }
WINDOW { GENERATE_TOKEN(WINDOW_SYM) }
EMPTY { GENERATE_TOKEN(EMPTY_SYM) }
JSON_TABLE { GENERATE_TOKEN(JSON_TABLE_SYM) }
NESTED { GENERATE_TOKEN(NESTED_SYM) }
ORDINALITY { GENERATE_TOKEN(ORDINALITY_SYM) }
PATH { GENERATE_TOKEN(PATH_SYM) }
HISTORY { GENERATE_TOKEN(HISTORY_SYM) }
REUSE { GENERATE_TOKEN(REUSE_SYM) }
SRID { GENERATE_TOKEN(SRID_SYM) }
THREAD_PRIORITY { GENERATE_TOKEN(THREAD_PRIORITY_SYM) }
RESOURCE { GENERATE_TOKEN(RESOURCE_SYM) }
SYSTEM { GENERATE_TOKEN(SYSTEM_SYM) }
VCPU { GENERATE_TOKEN(VCPU_SYM) }
MASTER_PUBLIC_KEY_PATH { GENERATE_TOKEN(MASTER_PUBLIC_KEY_PATH_SYM) }
GET_MASTER_PUBLIC_KEY { GENERATE_TOKEN(GET_MASTER_PUBLIC_KEY_SYM) }
RESTART { GENERATE_TOKEN(RESTART_SYM) }
DEFINITION { GENERATE_TOKEN(DEFINITION_SYM) }
DESCRIPTION { GENERATE_TOKEN(DESCRIPTION_SYM) }
ORGANIZATION { GENERATE_TOKEN(ORGANIZATION_SYM) }
REFERENCE { GENERATE_TOKEN(REFERENCE_SYM) }
ACTIVE { GENERATE_TOKEN(ACTIVE_SYM) }
INACTIVE { GENERATE_TOKEN(INACTIVE_SYM) }
LATERAL { GENERATE_TOKEN(LATERAL_SYM) }
OPTIONAL { GENERATE_TOKEN(OPTIONAL_SYM) }
SECONDARY { GENERATE_TOKEN(SECONDARY_SYM) }
SECONDARY_ENGINE { GENERATE_TOKEN(SECONDARY_ENGINE_SYM) }
SECONDARY_LOAD { GENERATE_TOKEN(SECONDARY_LOAD_SYM) }
SECONDARY_UNLOAD { GENERATE_TOKEN(SECONDARY_UNLOAD_SYM) }
RETAIN { GENERATE_TOKEN(RETAIN_SYM) }
OLD { GENERATE_TOKEN(OLD_SYM) }
ENFORCED { GENERATE_TOKEN(ENFORCED_SYM) }
OJ { GENERATE_TOKEN(OJ_SYM) }
NETWORK_NAMESPACE { GENERATE_TOKEN(NETWORK_NAMESPACE_SYM) }
CONDITIONLESS_JOIN { GENERATE_TOKEN(CONDITIONLESS_JOIN) }
SUBQUERY_AS_EXPR { GENERATE_TOKEN(SUBQUERY_AS_EXPR) }
EMPTY_FROM_CLAUSE { GENERATE_TOKEN(EMPTY_FROM_CLAUSE) }
EXPLAIN { GENERATE_TOKEN(EXPLAIN_SYM) }
TREE { GENERATE_TOKEN(TREE_SYM) }
TRADITIONAL { GENERATE_TOKEN(TRADITIONAL_SYM) }

SUBSTR { GENERATE_TOKEN(SUBSTRING) }
:= {
    return Parser::token::SET_VAR;
}

= {
    return Parser::token::EQ;
}

\<\< {
    return Parser::token::SHIFT_LEFT;
}

\>\> {
    return Parser::token::SHIFT_RIGHT;
}
\<=\> {
    return Parser::token::EQUAL_SYM;
}

\<= {
    return Parser::token::LE;
}

\>= {
    return Parser::token::GE;
}

\<\>|!= {
    return Parser::token::NE;
}

\> {
    return Parser::token::GT_SYM;
}

\< {
    return Parser::token::LT;
}
[0-9]+ {
    static const char *long_str = "2147483647";
    static const uint long_len = 10;
    // static const char *signed_long_str = "-2147483648";
    static const char *longlong_str = "9223372036854775807";
    static const uint longlong_len = 19;
    // static const char *signed_longlong_str = "-9223372036854775808";
    // static const uint signed_longlong_len = 19;
    static const char *unsigned_longlong_str = "18446744073709551615";
    static const uint unsigned_longlong_len = 20;

    size_t length = yyleng;
    yylval->emplace<std::string>( yytext, yyleng );

    char* str = yytext;
    Parser::token_type smaller, bigger;
    const char *cmp;

    // bool neg = 0;

    if (*str == '+')  // Remove sign and pre-zeros
    {
        str++;
        length--;
    } else if (*str == '-') {
        str++;
        length--;
        // neg = 1;
    }
    while (*str == '0' && length) {
        str++;
        length--;
    }
    if (length < long_len) return Parser::token::NUM;

    if (length == long_len) {
        cmp = long_str;
        smaller = Parser::token::NUM;
        bigger = Parser::token::LONG_NUM;
    } else if (length < longlong_len)
        return Parser::token::LONG_NUM;
    else if (length > longlong_len) {
        if (length > unsigned_longlong_len) return Parser::token::DECIMAL_NUM;
        cmp = unsigned_longlong_str;
        smaller = Parser::token::ULONGLONG_NUM;
        bigger = Parser::token::DECIMAL_NUM;
    } else {
        cmp = longlong_str;
        smaller = Parser::token::LONG_NUM;
        bigger = Parser::token::ULONGLONG_NUM;
    }

    while (*cmp && *cmp++ == *str++)
        ;
    return ((unsigned char)(str[-1]) <= (unsigned char)(cmp[-1])) ? smaller : bigger;
}

0x[0-9a-fA-F]+ {
    yylval->emplace<std::string>( yytext, yyleng );
    return Parser::token::HEX_NUM;
}

{REAL_LITERAL} {
    yylval->emplace<std::string>( yytext, yyleng );
    return Parser::token::REAL_NUM;
}

{DECIMAL_LITERAL} {
    yylval->emplace<std::string>( yytext, yyleng );
    return Parser::token::DECIMAL_NUM;
}

[a-zA-Z0-9_\S]+ { 
    yylval->emplace<std::string>( yytext, yyleng );
    return Parser::token::IDENT;
}

\`([^\`]|\\\`)+\` {
    yylval->emplace<std::string>( yytext + 1, yyleng-2 );
    return Parser::token::IDENT;
}

\'([^\']|\\\')*\'  {
    yylval->emplace<std::string>( yytext + 1, yyleng-2 );
    return Parser::token::TEXT_STRING;
}

\"([^\"]|\\\")*\"  {
    yylval->emplace<std::string>( yytext + 1, yyleng-2 );
    return Parser::token::TEXT_STRING;
}

[^\0-/%\{-~:-@\[-\^\`]+ { 
    yylval->emplace<std::string>( yytext, yyleng );
    return Parser::token::IDENT;
}

\? {
    return Parser::token::PARAM_MARKER;
}


<<EOF>> {
    return Parser::token::END_OF_INPUT;
}

\-\-.* {}
\/\*.*?\*\/;? {}


[ \t\r]+ {}
[\n]  { yycolumn = 1; }

. {
    return static_cast<Parser::token_type>(*yytext);
}


%%
/*** Additional Code ***/

namespace aries_parser {

Scanner::Scanner(std::istream* in,
         std::ostream* out)
    : AriesFlexLexer(in, out)
{
    yycolumn = 1;
}

Scanner::~Scanner()
{
}

void Scanner::set_debug(bool b)
{
    yy_flex_debug = b;
}

}

/* This implementation of AriesFlexLexer::yylex() is required to fill the
 * vtable of the class AriesFlexLexer. We define the scanner's main yylex
 * function via YY_DECL to reside in the Scanner class instead. */

#ifdef yylex
#undef yylex
#endif

int AriesFlexLexer::yylex()
{
    std::cerr << "in AriesFlexLexer::yylex() !" << std::endl;
    return 0;
}

/* When the scanner receives an end-of-file indication from YY_INPUT, it then
 * checks the yywrap() function. If yywrap() returns false (zero), then it is
 * assumed that the function has gone ahead and set up `yyin' to point to
 * another input file, and scanning continues. If it returns true (non-zero),
 * then the scanner terminates, returning 0 to its caller. */

int AriesFlexLexer::yywrap()
{
    return 1;
}