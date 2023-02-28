#include <iostream>
#include <string>
#include <algorithm>
#include <time.h>
#include <termios.h>
#include <unistd.h>
#include <signal.h>
#include <unordered_map>
#include <glog/logging.h>

#include <utils/string_util.h>
#include <frontend/SQLExecutor.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <frontend/SchemaBuilder.h>
#include <frontend/CommandExecutor.h>
#include <frontend/SQLParserPortal.h>
#include <AriesEngine/transaction/AriesInitialTable.h>
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include <frontend/ViewManager.h>

#include "Schema.h"
#include "SchemaManager.h"
#include "Compression/dict/AriesDictManager.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngineWrapper/AbstractMemTable.h"
#include "server/Configuration.h"

#include "server/mysql/include/sql_const.h"
#include "server/mysql/include/sql_authentication.h"
#include "datatypes/AriesDatetimeTrans.h"

using namespace std;
using namespace aries_engine;

namespace aries
{

static const char* PRODUCT_NAME = "Rateup";

TableEntrySPtr CommandToTableEntry( CommandStructure *arg_command_p, const schema::DatabaseEntrySPtr& database );
namespace schema
{
// a string with a \0 byte, represent empty string when stored into rocksdb,
// to differentiate from NULL
const string RATEUP_EMPTY_STRING( "", 1 );
const string RATEUP_NULL_VALUE_STRING( "" );
const char* DEFAULT_CHARSET_NAME = "utf8" ;
const char* DEFAULT_UTF8_COLLATION = "utf8_bin" ;
const std::string Schema::DefaultDatabaseName("default");
const std::string Schema::KeyPrefix("aries_schema_");
const std::string Schema::kSchemaColumnFamilyName("schema_family");

const string INFORMATION_SCHEMA = "information_schema";
const string PERFORMANCE_SCHEMA = "performance_schema";
const string SCHEMATA = "schemata";
// mysql 5.7.26
const std::string SYS_DATABASES[] = {
        INFORMATION_SCHEMA,
        PERFORMANCE_SCHEMA,
        "mysql",
        "sys"
};
static size_t MAX_SCHEMA_ROW_COUNT = 1024 * 10;
static string BuildCreateTableString(const TableEntrySPtr& table);

static const int64_t TBALE_ID_INFORMATION_SCHEMA_TABLES = 1;
static const int64_t TBALE_ID_INFORMATION_SCHEMA_COLUMNS = 2;
static const int64_t TBALE_ID_INFORMATION_SCHEMA_SCHEMATA = 3;
static int64_t CURRENT_TABLE_ID = 0;

#define CHECK_SQL_RESULT( result ) \
if( !result->IsSuccess() ) \
    ARIES_EXCEPTION_SIMPLE( result->GetErrorCode(), result->GetErrorMessage() );

/**
 * mysql> show create table schemata;
+----------+---------------------------------------------------------------------
| Table    | Create Table
+----------+----------------------------------------------------------------------
| SCHEMATA | CREATE TEMPORARY TABLE `SCHEMATA` (
  `CATALOG_NAME` varchar(512) NOT NULL DEFAULT '',
  `SCHEMA_NAME` varchar(64) NOT NULL DEFAULT '',
  `DEFAULT_CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
  `DEFAULT_COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
  `SQL_PATH` varchar(512) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 |
+----------+--------------------------------------------------------------------------
1 row in set (0.00 sec)

mysql> desc information_schema.schemata;
+----------------------------+--------------+------+-----+---------+-------+
| Field                      | Type         | Null | Key | Default | Extra |
+----------------------------+--------------+------+-----+---------+-------+
| CATALOG_NAME               | varchar(512) | NO   |     |         |       |
| SCHEMA_NAME                | varchar(64)  | NO   |     |         |       |
| DEFAULT_CHARACTER_SET_NAME | varchar(32)  | NO   |     |         |       |
| DEFAULT_COLLATION_NAME     | varchar(32)  | NO   |     |         |       |
| SQL_PATH                   | varchar(512) | YES  |     | NULL    |       |
+----------------------------+--------------+------+-----+---------+-------+

mysql> select * from schemata;
+--------------+--------------------+----------------------------+------------------------+----------+
| CATALOG_NAME | SCHEMA_NAME        | DEFAULT_CHARACTER_SET_NAME | DEFAULT_COLLATION_NAME | SQL_PATH |
+--------------+--------------------+----------------------------+------------------------+----------+
| def          | information_schema | utf8                       | utf8_general_ci        | NULL     |
| def          | lhp                | latin1                     | latin1_swedish_ci      | NULL     |
| def          | mysql              | latin1                     | latin1_swedish_ci      | NULL     |
| def          | performance_schema | utf8                       | utf8_general_ci        | NULL     |
| def          | sys                | utf8                       | utf8_general_ci        | NULL     |
+--------------+--------------------+----------------------------+------------------------+----------+

 mysql> show create table tables;
+--------+--------------------------------------------------------
| Table  | Create Table
+--------+--------------------------------------------------------
| TABLES | CREATE TEMPORARY TABLE `TABLES` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `TABLE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `ENGINE` varchar(64) DEFAULT NULL,
  `VERSION` bigint(21) DEFAULT NULL,
  `ROW_FORMAT` varchar(10) DEFAULT NULL,
  `TABLE_ROWS` bigint(21) DEFAULT NULL,
  `AVG_ROW_LENGTH` bigint(21) DEFAULT NULL,
  `DATA_LENGTH` bigint(21) DEFAULT NULL,
  `MAX_DATA_LENGTH` bigint(21) DEFAULT NULL,
  `INDEX_LENGTH` bigint(21) DEFAULT NULL,
  `DATA_FREE` bigint(21) DEFAULT NULL,
  `AUTO_INCREMENT` bigint(21) DEFAULT NULL,
  `CREATE_TIME` datetime DEFAULT NULL,
  `UPDATE_TIME` datetime DEFAULT NULL,
  `CHECK_TIME` datetime DEFAULT NULL,
  `TABLE_COLLATION` varchar(32) DEFAULT NULL,
  `CHECKSUM` bigint(21) DEFAULT NULL,
  `CREATE_OPTIONS` varchar(255) DEFAULT NULL,
  `TABLE_COMMENT` varchar(2048) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 |
+--------+-----------------------------------------------------------

 mysql> select * from tables limit 5;
+---------------+--------------------+---------------------------------------+-------------+--------+---------+------------+------------+----------------+-------------+-----------------+--------------+-----------+----------------+---------------------+-------------+------------+-----------------+----------+----------------+---------------+
| TABLE_CATALOG | TABLE_SCHEMA       | TABLE_NAME                            | TABLE_TYPE  | ENGINE | VERSION | ROW_FORMAT | TABLE_ROWS | AVG_ROW_LENGTH | DATA_LENGTH | MAX_DATA_LENGTH | INDEX_LENGTH | DATA_FREE | AUTO_INCREMENT | CREATE_TIME         | UPDATE_TIME | CHECK_TIME | TABLE_COLLATION | CHECKSUM | CREATE_OPTIONS | TABLE_COMMENT |
+---------------+--------------------+---------------------------------------+-------------+--------+---------+------------+------------+----------------+-------------+-----------------+--------------+-----------+----------------+---------------------+-------------+------------+-----------------+----------+----------------+---------------+
| def           | information_schema | CHARACTER_SETS                        | SYSTEM VIEW | MEMORY |      10 | Fixed      |       NULL |            384 |           0 |        16434816 |            0 |         0 |           NULL | 2019-09-24 22:10:57 | NULL        | NULL       | utf8_general_ci |     NULL | max_rows=43690 |               |
| def           | information_schema | COLLATIONS                            | SYSTEM VIEW | MEMORY |      10 | Fixed      |       NULL |            231 |           0 |        16704765 |            0 |         0 |           NULL | 2019-09-24 22:10:57 | NULL        | NULL       | utf8_general_ci |     NULL | max_rows=72628 |               |
| def           | information_schema | COLLATION_CHARACTER_SET_APPLICABILITY | SYSTEM VIEW | MEMORY |      10 | Fixed      |       NULL |            195 |           0 |        16357770 |            0 |         0 |           NULL | 2019-09-24 22:10:57 | NULL        | NULL       | utf8_general_ci |     NULL | max_rows=86037 |               |
| def           | information_schema | COLUMNS                               | SYSTEM VIEW | InnoDB |      10 | Dynamic    |       NULL |              0 |       16384 |               0 |            0 |   8388608 |           NULL | NULL                | NULL        | NULL       | utf8_general_ci |     NULL | max_rows=2789  |               |
| def           | information_schema | COLUMN_PRIVILEGES                     | SYSTEM VIEW | MEMORY |      10 | Fixed      |       NULL |           2565 |           0 |        16757145 |            0 |         0 |           NULL | 2019-09-24 22:10:57 | NULL        | NULL       | utf8_general_ci |     NULL | max_rows=6540  |               |
| def           | lhp                | channel_type                          | BASE TABLE  | InnoDB |      10 | Dynamic    |          0 |              0 |       16384 |               0 |        16384 |         0 |              1 | 2019-09-10 16:00:51 | NULL        | NULL       | utf8mb4_general_ci |     NULL |                |               |
+---------------+--------------------+---------------------------------------+-------------+--------+---------+------------+------------+----------------+-------------+-----------------+--------------+-----------+----------------+---------------------+-------------+------------+-----------------+----------+----------------+---------------+

 mysql> show create table columns;
+---------+-----------------------------------------------------------
| Table   | Create Table
+---------+-----------------------------------------------------------
| COLUMNS | CREATE TEMPORARY TABLE `COLUMNS` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `ORDINAL_POSITION` bigint(21) NOT NULL DEFAULT '0',
  `COLUMN_DEFAULT` longtext,
  `IS_NULLABLE` varchar(3) NOT NULL DEFAULT '',
  `DATA_TYPE` varchar(64) NOT NULL DEFAULT '',
  `CHARACTER_MAXIMUM_LENGTH` bigint(21) DEFAULT NULL,
  `CHARACTER_OCTET_LENGTH` bigint(21) DEFAULT NULL,
  `NUMERIC_PRECISION` bigint(21) DEFAULT NULL,
  `NUMERIC_SCALE` bigint(21) DEFAULT NULL,
  `DATETIME_PRECISION` bigint(21) DEFAULT NULL,
  `CHARACTER_SET_NAME` varchar(32) DEFAULT NULL,
  `COLLATION_NAME` varchar(32) DEFAULT NULL,
  `COLUMN_TYPE` longtext NOT NULL,
  `COLUMN_KEY` varchar(3) NOT NULL DEFAULT '',
  `EXTRA` varchar(30) NOT NULL DEFAULT '',
  `PRIVILEGES` varchar(80) NOT NULL DEFAULT '',
  `COLUMN_COMMENT` varchar(1024) NOT NULL DEFAULT '',
  `GENERATION_EXPRESSION` longtext NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 |
+---------+--------------------------------------------------------------------------------

 mysql> select * from columns limit 5;
+---------------+--------------------+----------------+----------------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+-----------------+-------------+------------+-------+------------+----------------+-----------------------+
| TABLE_CATALOG | TABLE_SCHEMA       | TABLE_NAME     | COLUMN_NAME          | ORDINAL_POSITION | COLUMN_DEFAULT | IS_NULLABLE | DATA_TYPE | CHARACTER_MAXIMUM_LENGTH | CHARACTER_OCTET_LENGTH | NUMERIC_PRECISION | NUMERIC_SCALE | DATETIME_PRECISION | CHARACTER_SET_NAME | COLLATION_NAME  | COLUMN_TYPE | COLUMN_KEY | EXTRA | PRIVILEGES | COLUMN_COMMENT | GENERATION_EXPRESSION |
+---------------+--------------------+----------------+----------------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+-----------------+-------------+------------+-------+------------+----------------+-----------------------+
| def           | information_schema | CHARACTER_SETS | CHARACTER_SET_NAME   |                1 |                | NO          | varchar   |                       32 |                     96 |              NULL |          NULL |               NULL | utf8               | utf8_general_ci | varchar(32) |            |       | select     |                |                       |
| def           | information_schema | CHARACTER_SETS | DEFAULT_COLLATE_NAME |                2 |                | NO          | varchar   |                       32 |                     96 |              NULL |          NULL |               NULL | utf8               | utf8_general_ci | varchar(32) |            |       | select     |                |                       |
| def           | information_schema | CHARACTER_SETS | DESCRIPTION          |                3 |                | NO          | varchar   |                       60 |                    180 |              NULL |          NULL |               NULL | utf8               | utf8_general_ci | varchar(60) |            |       | select     |                |                       |
| def           | information_schema | CHARACTER_SETS | MAXLEN               |                4 | 0              | NO          | bigint    |                     NULL |                   NULL |                19 |             0 |               NULL | NULL               | NULL            | bigint(3)   |            |       | select     |                |                       |
| def           | information_schema | COLLATIONS     | COLLATION_NAME       |                1 |                | NO          | varchar   |                       32 |                     96 |              NULL |          NULL |               NULL | utf8               | utf8_general_ci | varchar(32) |            |       | select     |                |                       |
+---------------+--------------------+----------------+----------------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+-----------------+-------------+------------+-------+------------+----------------+-----------------------+
 */
static const std::string SCHEMA_SQL_TABLES = R"(
CREATE TEMPORARY TABLE `information_schema`.`TABLES` (
`TABLE_ID` bigint NOT NULL,
`TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
`TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
`TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
`TABLE_TYPE` varchar(64) NOT NULL DEFAULT '',
`ENGINE` varchar(64) DEFAULT NULL,
`VERSION` bigint(21) DEFAULT NULL,
`ROW_FORMAT` varchar(10) DEFAULT NULL,
`TABLE_ROWS` bigint(21) DEFAULT NULL,
`AVG_ROW_LENGTH` bigint(21) DEFAULT NULL,
`DATA_LENGTH` bigint(21) DEFAULT NULL,
`MAX_DATA_LENGTH` bigint(21) DEFAULT NULL,
`INDEX_LENGTH` bigint(21) DEFAULT NULL,
`DATA_FREE` bigint(21) DEFAULT NULL,
`AUTO_INCREMENT` bigint(21) DEFAULT NULL,
`CREATE_TIME` datetime DEFAULT NULL,
`UPDATE_TIME` datetime DEFAULT NULL,
`CHECK_TIME` datetime DEFAULT NULL,
`TABLE_COLLATION` varchar(32) DEFAULT NULL,
`CHECKSUM` bigint(21) DEFAULT NULL,
`CREATE_OPTIONS` varchar(255) DEFAULT NULL,
`TABLE_COMMENT` varchar(1024) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_COLUMNS = R"(
CREATE TEMPORARY TABLE `information_schema`.`COLUMNS` (
`TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
`TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
`TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
`COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
`ORDINAL_POSITION` bigint(21) NOT NULL DEFAULT '0',
`COLUMN_DEFAULT` longtext,
`IS_NULLABLE` varchar(3) NOT NULL DEFAULT '',
`DATA_TYPE` varchar(64) NOT NULL DEFAULT '',
`CHARACTER_MAXIMUM_LENGTH` bigint(21) DEFAULT NULL,
`CHARACTER_OCTET_LENGTH` bigint(21) DEFAULT NULL,
`NUMERIC_PRECISION` bigint(21) DEFAULT NULL,
`NUMERIC_SCALE` bigint(21) DEFAULT NULL,
`DATETIME_PRECISION` bigint(21) DEFAULT NULL,
`CHARACTER_SET_NAME` varchar(32) DEFAULT NULL,
`COLLATION_NAME` varchar(32) DEFAULT NULL,
`COLUMN_TYPE` longtext NOT NULL,
`COLUMN_KEY` varchar(3) NOT NULL DEFAULT '',
`EXTRA` varchar(30) NOT NULL DEFAULT '',
`PRIVILEGES` varchar(80) NOT NULL DEFAULT '',
`COLUMN_COMMENT` varchar(256) NOT NULL DEFAULT '',
`GENERATION_EXPRESSION` longtext NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_SCHEMATA = R"(
CREATE TEMPORARY TABLE `information_schema`.`SCHEMATA` (
`CATALOG_NAME` varchar(512) NOT NULL DEFAULT '',
`SCHEMA_NAME` varchar(64) NOT NULL DEFAULT '',
`DEFAULT_CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
`DEFAULT_COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
`SQL_PATH` varchar(512) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_VIEWS = R"(
CREATE TEMPORARY TABLE `VIEWS` (
`TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
`TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
`TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
`VIEW_DEFINITION` longtext NOT NULL,
`CHECK_OPTION` varchar(8) NOT NULL DEFAULT '',
`IS_UPDATABLE` varchar(3) NOT NULL DEFAULT '',
`DEFINER` varchar(93) NOT NULL DEFAULT '',
`SECURITY_TYPE` varchar(7) NOT NULL DEFAULT '',
`CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
`COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_TABLE_CONSTRAINTS = R"(
-- Table structure for table `TABLE_CONSTRAINTS`
-- DROP TABLE IF EXISTS `TABLE_CONSTRAINTS`;
CREATE TEMPORARY TABLE `TABLE_CONSTRAINTS` (
  `CONSTRAINT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `CONSTRAINT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_NAME` varchar(64) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_TYPE` varchar(64) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_KEY_COLUMN_USAGE = R"(
-- Table structure for table `KEY_COLUMN_USAGE`
-- DROP TABLE IF EXISTS `KEY_COLUMN_USAGE`;
CREATE TEMPORARY TABLE `KEY_COLUMN_USAGE` (
  `CONSTRAINT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `CONSTRAINT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_NAME` varchar(64) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `ORDINAL_POSITION` bigint(10) NOT NULL DEFAULT '0',
  `POSITION_IN_UNIQUE_CONSTRAINT` bigint(10) DEFAULT NULL,
  `REFERENCED_TABLE_SCHEMA` varchar(64) DEFAULT NULL,
  `REFERENCED_TABLE_NAME` varchar(64) DEFAULT NULL,
  `REFERENCED_COLUMN_NAME` varchar(64) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_PARTITIONS = R"(
-- Table structure for table `PARTITIONS`
-- DROP TABLE IF EXISTS `PARTITIONS`;
CREATE TEMPORARY TABLE `PARTITIONS` (
  `TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `PARTITION_NAME` varchar(64) DEFAULT NULL,
  `SUBPARTITION_NAME` varchar(64) DEFAULT NULL,
  `PARTITION_ORDINAL_POSITION` bigint(21) DEFAULT NULL,
  `SUBPARTITION_ORDINAL_POSITION` bigint(21) DEFAULT NULL,
  `PARTITION_METHOD` varchar(18) DEFAULT NULL,
  `SUBPARTITION_METHOD` varchar(12) DEFAULT NULL,
  `PARTITION_EXPRESSION` longtext,
  `SUBPARTITION_EXPRESSION` longtext,
  `PARTITION_DESCRIPTION` longtext,
  `TABLE_ROWS` bigint(21) NOT NULL DEFAULT '0',
  `AVG_ROW_LENGTH` bigint(21) NOT NULL DEFAULT '0',
  `DATA_LENGTH` bigint(21) NOT NULL DEFAULT '0',
  `MAX_DATA_LENGTH` bigint(21) DEFAULT NULL,
  `INDEX_LENGTH` bigint(21) NOT NULL DEFAULT '0',
  `DATA_FREE` bigint(21) NOT NULL DEFAULT '0',
  `CREATE_TIME` datetime DEFAULT NULL,
  `UPDATE_TIME` datetime DEFAULT NULL,
  `CHECK_TIME` datetime DEFAULT NULL,
  `CHECKSUM` bigint(21) DEFAULT NULL,
  `PARTITION_COMMENT` varchar(80) NOT NULL DEFAULT '',
  `NODEGROUP` varchar(12) NOT NULL DEFAULT '',
  `TABLESPACE_NAME` varchar(64) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_DICTS = R"(
CREATE TEMPORARY TABLE `DICTS` (
  `ID` bigint(10) NOT NULL DEFAULT '0',
  `NAME` varchar(64) NOT NULL DEFAULT '',
  `INDEX_TYPE` varchar(64) NOT NULL DEFAULT 'tinyint',
  `REF_COUNT` bigint(10) NOT NULL DEFAULT '0',
  `IS_NULLABLE` varchar(3) NOT NULL DEFAULT '',
  `CHARACTER_MAXIMUM_LENGTH` int(21) NOT NULL,
  `CHARACTER_OCTET_LENGTH` int(21) NOT NULL,
  `CHARACTER_SET_NAME` varchar(32) DEFAULT NULL,
  `COLLATION_NAME` varchar(32) DEFAULT NULL
)DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_DICT_COLUMN_USAGE = R"(
CREATE TEMPORARY TABLE `DICT_COLUMN_USAGE` (
  `TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `DICT_ID` bigint(10) NOT NULL
)DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_GLOBAL_VARIABLES = R"(
-- Table structure for table `GLOBAL_VARIABLES`
-- DROP TABLE IF EXISTS `GLOBAL_VARIABLES`;
CREATE TEMPORARY TABLE `GLOBAL_VARIABLES` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_SESSION_VARIABLES = R"(
-- Table structure for table `SESSION_VARIABLES`
-- DROP TABLE IF EXISTS `SESSION_VARIABLES`;
CREATE TEMPORARY TABLE `SESSION_VARIABLES` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_CHARACTER_SETS = R"(
CREATE TEMPORARY TABLE `CHARACTER_SETS` (
  `CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
  `DEFAULT_COLLATE_NAME` varchar(32) NOT NULL DEFAULT '',
  `DESCRIPTION` varchar(60) NOT NULL DEFAULT '',
  `MAXLEN` bigint(3) NOT NULL DEFAULT '0'
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_COLLATIONS = R"(
CREATE TEMPORARY TABLE `COLLATIONS` (
  `COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
  `CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
  `ID` bigint(11) NOT NULL DEFAULT '0',
  `IS_DEFAULT` varchar(3) NOT NULL DEFAULT '',
  `IS_COMPILED` varchar(3) NOT NULL DEFAULT '',
  `SORTLEN` bigint(3) NOT NULL DEFAULT '0'
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_COLLATION_CHARACTER_SET_APPLICABILITY = R"(
-- Table structure for table `COLLATION_CHARACTER_SET_APPLICABILITY`
-- DROP TABLE IF EXISTS `COLLATION_CHARACTER_SET_APPLICABILITY`;
CREATE TEMPORARY TABLE `COLLATION_CHARACTER_SET_APPLICABILITY` (
  `COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
  `CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;
-- INSERT INTO `COLLATION_CHARACTER_SET_APPLICABILITY` VALUES ('utf8_general_ci','utf8');)";

static const std::string SCHEMA_SQL_COLUMN_PRIVILEGES = R"(
-- Table structure for table `COLUMN_PRIVILEGES`
-- DROP TABLE IF EXISTS `COLUMN_PRIVILEGES`;
CREATE TEMPORARY TABLE `COLUMN_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(3) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_ENGINES = R"(
-- Table structure for table `ENGINES`
-- DROP TABLE IF EXISTS `ENGINES`;
CREATE TEMPORARY TABLE `ENGINES` (
  `ENGINE` varchar(64) NOT NULL DEFAULT '',
  `SUPPORT` varchar(8) NOT NULL DEFAULT '',
  `COMMENT` varchar(80) NOT NULL DEFAULT '',
  `TRANSACTIONS` varchar(3) DEFAULT NULL,
  `XA` varchar(3) DEFAULT NULL,
  `SAVEPOINTS` varchar(3) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;
-- INSERT INTO `ENGINES` VALUES ('Rateup','DEFAULT','Column store','YES','YES','YES');)";

static const std::string SCHEMA_SQL_EVENTS = R"(
-- Table structure for table `EVENTS`
-- DROP TABLE IF EXISTS `EVENTS`;
CREATE TEMPORARY TABLE `EVENTS` (
  `EVENT_CATALOG` varchar(64) NOT NULL DEFAULT '',
  `EVENT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `EVENT_NAME` varchar(64) NOT NULL DEFAULT '',
  `DEFINER` varchar(93) NOT NULL DEFAULT '',
  `TIME_ZONE` varchar(64) NOT NULL DEFAULT '',
  `EVENT_BODY` varchar(8) NOT NULL DEFAULT '',
  `EVENT_DEFINITION` longtext NOT NULL,
  `EVENT_TYPE` varchar(9) NOT NULL DEFAULT '',
  `EXECUTE_AT` datetime DEFAULT NULL,
  `INTERVAL_VALUE` varchar(256) DEFAULT NULL,
  `INTERVAL_FIELD` varchar(18) DEFAULT NULL,
  `SQL_MODE` varchar(1024) NOT NULL DEFAULT '',
  `STARTS` datetime DEFAULT NULL,
  `ENDS` datetime DEFAULT NULL,
  `STATUS` varchar(18) NOT NULL DEFAULT '',
  `ON_COMPLETION` varchar(12) NOT NULL DEFAULT '',
  `CREATED` datetime NOT NULL DEFAULT '0000-00-00 00:00:00',
  `LAST_ALTERED` datetime NOT NULL DEFAULT '0000-00-00 00:00:00',
  `LAST_EXECUTED` datetime DEFAULT NULL,
  `EVENT_COMMENT` varchar(64) NOT NULL DEFAULT '',
  `ORIGINATOR` bigint(10) NOT NULL DEFAULT '0',
  `CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
  `COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT '',
  `DATABASE_COLLATION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_FILES = R"(
-- Table structure for table `FILES`
-- DROP TABLE IF EXISTS `FILES`;
CREATE TEMPORARY TABLE `FILES` (
  `FILE_ID` bigint(4) NOT NULL DEFAULT '0',
  `FILE_NAME` varchar(1024) DEFAULT NULL,
  `FILE_TYPE` varchar(20) NOT NULL DEFAULT '',
  `TABLESPACE_NAME` varchar(64) DEFAULT NULL,
  `TABLE_CATALOG` varchar(64) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) DEFAULT NULL,
  `TABLE_NAME` varchar(64) DEFAULT NULL,
  `LOGFILE_GROUP_NAME` varchar(64) DEFAULT NULL,
  `LOGFILE_GROUP_NUMBER` bigint(4) DEFAULT NULL,
  `ENGINE` varchar(64) NOT NULL DEFAULT '',
  `FULLTEXT_KEYS` varchar(64) DEFAULT NULL,
  `DELETED_ROWS` bigint(4) DEFAULT NULL,
  `UPDATE_COUNT` bigint(4) DEFAULT NULL,
  `FREE_EXTENTS` bigint(4) DEFAULT NULL,
  `TOTAL_EXTENTS` bigint(4) DEFAULT NULL,
  `EXTENT_SIZE` bigint(4) NOT NULL DEFAULT '0',
  `INITIAL_SIZE` bigint(21) DEFAULT NULL,
  `MAXIMUM_SIZE` bigint(21) DEFAULT NULL,
  `AUTOEXTEND_SIZE` bigint(21) DEFAULT NULL,
  `CREATION_TIME` datetime DEFAULT NULL,
  `LAST_UPDATE_TIME` datetime DEFAULT NULL,
  `LAST_ACCESS_TIME` datetime DEFAULT NULL,
  `RECOVER_TIME` bigint(4) DEFAULT NULL,
  `TRANSACTION_COUNTER` bigint(4) DEFAULT NULL,
  `VERSION` bigint(21) DEFAULT NULL,
  `ROW_FORMAT` varchar(10) DEFAULT NULL,
  `TABLE_ROWS` bigint(21) DEFAULT NULL,
  `AVG_ROW_LENGTH` bigint(21) DEFAULT NULL,
  `DATA_LENGTH` bigint(21) DEFAULT NULL,
  `MAX_DATA_LENGTH` bigint(21) DEFAULT NULL,
  `INDEX_LENGTH` bigint(21) DEFAULT NULL,
  `DATA_FREE` bigint(21) DEFAULT NULL,
  `CREATE_TIME` datetime DEFAULT NULL,
  `UPDATE_TIME` datetime DEFAULT NULL,
  `CHECK_TIME` datetime DEFAULT NULL,
  `CHECKSUM` bigint(21) DEFAULT NULL,
  `STATUS` varchar(20) NOT NULL DEFAULT '',
  `EXTRA` varchar(255) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_GLOBAL_STATUS = R"(
-- Table structure for table `GLOBAL_STATUS`
-- DROP TABLE IF EXISTS `GLOBAL_STATUS`;
CREATE TEMPORARY TABLE `GLOBAL_STATUS` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_OPTIMIZER_TRACE = R"(
-- Table structure for table `OPTIMIZER_TRACE`
-- DROP TABLE IF EXISTS `OPTIMIZER_TRACE`;
CREATE TEMPORARY TABLE `OPTIMIZER_TRACE` (
  `QUERY` longtext NOT NULL,
  `TRACE` longtext NOT NULL,
  `MISSING_BYTES_BEYOND_MAX_MEM_SIZE` int(20) NOT NULL DEFAULT '0',
  `INSUFFICIENT_PRIVILEGES` tinyint(1) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_PARAMETERS = R"(
-- Table structure for table `PARAMETERS`
-- DROP TABLE IF EXISTS `PARAMETERS`;
CREATE TEMPORARY TABLE `PARAMETERS` (
  `SPECIFIC_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `SPECIFIC_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `SPECIFIC_NAME` varchar(64) NOT NULL DEFAULT '',
  `ORDINAL_POSITION` int(21) NOT NULL DEFAULT '0',
  `PARAMETER_MODE` varchar(5) DEFAULT NULL,
  `PARAMETER_NAME` varchar(64) DEFAULT NULL,
  `DATA_TYPE` varchar(64) NOT NULL DEFAULT '',
  `CHARACTER_MAXIMUM_LENGTH` int(21) DEFAULT NULL,
  `CHARACTER_OCTET_LENGTH` int(21) DEFAULT NULL,
  `NUMERIC_PRECISION` bigint(21) DEFAULT NULL,
  `NUMERIC_SCALE` int(21) DEFAULT NULL,
  `DATETIME_PRECISION` bigint(21) DEFAULT NULL,
  `CHARACTER_SET_NAME` varchar(64) DEFAULT NULL,
  `COLLATION_NAME` varchar(64) DEFAULT NULL,
  `DTD_IDENTIFIER` longtext NOT NULL,
  `ROUTINE_TYPE` varchar(9) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_PLUGINS = R"(
-- Table structure for table `PLUGINS`
-- DROP TABLE IF EXISTS `PLUGINS`;
CREATE TEMPORARY TABLE `PLUGINS` (
  `PLUGIN_NAME` varchar(64) NOT NULL DEFAULT '',
  `PLUGIN_VERSION` varchar(20) NOT NULL DEFAULT '',
  `PLUGIN_STATUS` varchar(10) NOT NULL DEFAULT '',
  `PLUGIN_TYPE` varchar(80) NOT NULL DEFAULT '',
  `PLUGIN_TYPE_VERSION` varchar(20) NOT NULL DEFAULT '',
  `PLUGIN_LIBRARY` varchar(64) DEFAULT NULL,
  `PLUGIN_LIBRARY_VERSION` varchar(20) DEFAULT NULL,
  `PLUGIN_AUTHOR` varchar(64) DEFAULT NULL,
  `PLUGIN_DESCRIPTION` longtext,
  `PLUGIN_LICENSE` varchar(80) DEFAULT NULL,
  `LOAD_OPTION` varchar(64) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
-- INSERT INTO `PLUGINS` VALUES ('mysql_native_password','1.1','ACTIVE','AUTHENTICATION','1.1',NULL,NULL,'R.J.Silk, Sergei Golubchik','Native MySQL authentication','GPL','FORCE');)";

static const std::string SCHEMA_SQL_PROCESSLIST = R"(
-- Table structure for table `PROCESSLIST`
-- DROP TABLE IF EXISTS `PROCESSLIST`;
CREATE TEMPORARY TABLE `PROCESSLIST` (
  `ID` bigint(21) NOT NULL DEFAULT '0',
  `USER` varchar(32) NOT NULL DEFAULT '',
  `HOST` varchar(64) NOT NULL DEFAULT '',
  `DB` varchar(64) DEFAULT NULL,
  `COMMAND` varchar(16) NOT NULL DEFAULT '',
  `TIME` int(7) NOT NULL DEFAULT '0',
  `STATE` varchar(64) DEFAULT NULL,
  `INFO` longtext
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_PROFILING = R"(
-- Table structure for table `PROFILING`
-- DROP TABLE IF EXISTS `PROFILING`;
CREATE TEMPORARY TABLE `PROFILING` (
  `QUERY_ID` int(20) NOT NULL DEFAULT '0',
  `SEQ` int(20) NOT NULL DEFAULT '0',
  `STATE` varchar(30) NOT NULL DEFAULT '',
  `DURATION` decimal(9,6) NOT NULL DEFAULT '0.000000',
  `CPU_USER` decimal(9,6) DEFAULT NULL,
  `CPU_SYSTEM` decimal(9,6) DEFAULT NULL,
  `CONTEXT_VOLUNTARY` int(20) DEFAULT NULL,
  `CONTEXT_INVOLUNTARY` int(20) DEFAULT NULL,
  `BLOCK_OPS_IN` int(20) DEFAULT NULL,
  `BLOCK_OPS_OUT` int(20) DEFAULT NULL,
  `MESSAGES_SENT` int(20) DEFAULT NULL,
  `MESSAGES_RECEIVED` int(20) DEFAULT NULL,
  `PAGE_FAULTS_MAJOR` int(20) DEFAULT NULL,
  `PAGE_FAULTS_MINOR` int(20) DEFAULT NULL,
  `SWAPS` int(20) DEFAULT NULL,
  `SOURCE_FUNCTION` varchar(30) DEFAULT NULL,
  `SOURCE_FILE` varchar(20) DEFAULT NULL,
  `SOURCE_LINE` int(20) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_REFERENTIAL_CONSTRAINTS = R"(
-- Table structure for table `REFERENTIAL_CONSTRAINTS`
-- DROP TABLE IF EXISTS `REFERENTIAL_CONSTRAINTS`;
CREATE TEMPORARY TABLE `REFERENTIAL_CONSTRAINTS` (
  `CONSTRAINT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `CONSTRAINT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_NAME` varchar(64) NOT NULL DEFAULT '',
  `UNIQUE_CONSTRAINT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `UNIQUE_CONSTRAINT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `UNIQUE_CONSTRAINT_NAME` varchar(64) DEFAULT NULL,
  `MATCH_OPTION` varchar(64) NOT NULL DEFAULT '',
  `UPDATE_RULE` varchar(64) NOT NULL DEFAULT '',
  `DELETE_RULE` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `REFERENCED_TABLE_NAME` varchar(64) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_ROUTINES = R"(
-- Table structure for table `ROUTINES`
-- DROP TABLE IF EXISTS `ROUTINES`;
CREATE TEMPORARY TABLE `ROUTINES` (
  `SPECIFIC_NAME` varchar(64) NOT NULL DEFAULT '',
  `ROUTINE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `ROUTINE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `ROUTINE_NAME` varchar(64) NOT NULL DEFAULT '',
  `ROUTINE_TYPE` varchar(9) NOT NULL DEFAULT '',
  `DATA_TYPE` varchar(64) NOT NULL DEFAULT '',
  `CHARACTER_MAXIMUM_LENGTH` int(21) DEFAULT NULL,
  `CHARACTER_OCTET_LENGTH` int(21) DEFAULT NULL,
  `NUMERIC_PRECISION` bigint(21) DEFAULT NULL,
  `NUMERIC_SCALE` int(21) DEFAULT NULL,
  `DATETIME_PRECISION` bigint(21) DEFAULT NULL,
  `CHARACTER_SET_NAME` varchar(64) DEFAULT NULL,
  `COLLATION_NAME` varchar(64) DEFAULT NULL,
  `DTD_IDENTIFIER` longtext,
  `ROUTINE_BODY` varchar(8) NOT NULL DEFAULT '',
  `ROUTINE_DEFINITION` longtext,
  `EXTERNAL_NAME` varchar(64) DEFAULT NULL,
  `EXTERNAL_LANGUAGE` varchar(64) DEFAULT NULL,
  `PARAMETER_STYLE` varchar(8) NOT NULL DEFAULT '',
  `IS_DETERMINISTIC` varchar(3) NOT NULL DEFAULT '',
  `SQL_DATA_ACCESS` varchar(64) NOT NULL DEFAULT '',
  `SQL_PATH` varchar(64) DEFAULT NULL,
  `SECURITY_TYPE` varchar(7) NOT NULL DEFAULT '',
  `CREATED` datetime NOT NULL DEFAULT '0000-00-00 00:00:00',
  `LAST_ALTERED` datetime NOT NULL DEFAULT '0000-00-00 00:00:00',
  `SQL_MODE` varchar(1024) NOT NULL DEFAULT '',
  `ROUTINE_COMMENT` longtext NOT NULL,
  `DEFINER` varchar(93) NOT NULL DEFAULT '',
  `CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
  `COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT '',
  `DATABASE_COLLATION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_SCHEMA_PRIVILEGES = R"(
-- Table structure for table `SCHEMA_PRIVILEGES`
-- DROP TABLE IF EXISTS `SCHEMA_PRIVILEGES`;
CREATE TEMPORARY TABLE `SCHEMA_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_SESSION_STATUS = R"(
-- Table structure for table `SESSION_STATUS`
--DROP TABLE IF EXISTS `SESSION_STATUS`;
CREATE TEMPORARY TABLE `SESSION_STATUS` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_STATISTICS = R"(
-- Table structure for table `STATISTICS`
-- DROP TABLE IF EXISTS `STATISTICS`;
CREATE TEMPORARY TABLE `STATISTICS` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `NON_UNIQUE` bigint(1) NOT NULL DEFAULT '0',
  `INDEX_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `INDEX_NAME` varchar(64) NOT NULL DEFAULT '',
  `SEQ_IN_INDEX` bigint(2) NOT NULL DEFAULT '0',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLLATION` varchar(1) DEFAULT NULL,
  `CARDINALITY` bigint(21) DEFAULT NULL,
  `SUB_PART` bigint(3) DEFAULT NULL,
  `PACKED` varchar(10) DEFAULT NULL,
  `NULLABLE` varchar(3) NOT NULL DEFAULT '',
  `INDEX_TYPE` varchar(16) NOT NULL DEFAULT '',
  `COMMENT` varchar(16) DEFAULT NULL,
  `INDEX_COMMENT` varchar(1024) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_TABLESPACES = R"(
-- Table structure for table `TABLESPACES`
-- DROP TABLE IF EXISTS `TABLESPACES`;
CREATE TEMPORARY TABLE `TABLESPACES` (
  `TABLESPACE_NAME` varchar(64) NOT NULL DEFAULT '',
  `ENGINE` varchar(64) NOT NULL DEFAULT '',
  `TABLESPACE_TYPE` varchar(64) DEFAULT NULL,
  `LOGFILE_GROUP_NAME` varchar(64) DEFAULT NULL,
  `EXTENT_SIZE` bigint(21) DEFAULT NULL,
  `AUTOEXTEND_SIZE` bigint(21) DEFAULT NULL,
  `MAXIMUM_SIZE` bigint(21) DEFAULT NULL,
  `NODEGROUP_ID` bigint(21) DEFAULT NULL,
  `TABLESPACE_COMMENT` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_TABLE_PRIVILEGES = R"(
-- Table structure for table `TABLE_PRIVILEGES`
-- DROP TABLE IF EXISTS `TABLE_PRIVILEGES`;
CREATE TEMPORARY TABLE `TABLE_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_TRIGGERS = R"(
-- Table structure for table `TRIGGERS`
-- DROP TABLE IF EXISTS `TRIGGERS`;
CREATE TEMPORARY TABLE `TRIGGERS` (
  `TRIGGER_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TRIGGER_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TRIGGER_NAME` varchar(64) NOT NULL DEFAULT '',
  `EVENT_MANIPULATION` varchar(6) NOT NULL DEFAULT '',
  `EVENT_OBJECT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `EVENT_OBJECT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `EVENT_OBJECT_TABLE` varchar(64) NOT NULL DEFAULT '',
  `ACTION_ORDER` bigint(4) NOT NULL DEFAULT '0',
  `ACTION_CONDITION` longtext,
  `ACTION_STATEMENT` longtext NOT NULL,
  `ACTION_ORIENTATION` varchar(9) NOT NULL DEFAULT '',
  `ACTION_TIMING` varchar(6) NOT NULL DEFAULT '',
  `ACTION_REFERENCE_OLD_TABLE` varchar(64) DEFAULT NULL,
  `ACTION_REFERENCE_NEW_TABLE` varchar(64) DEFAULT NULL,
  `ACTION_REFERENCE_OLD_ROW` varchar(3) NOT NULL DEFAULT '',
  `ACTION_REFERENCE_NEW_ROW` varchar(3) NOT NULL DEFAULT '',
  `CREATED` datetime(2) DEFAULT NULL,
  `SQL_MODE` varchar(1024) NOT NULL DEFAULT '',
  `DEFINER` varchar(93) NOT NULL DEFAULT '',
  `CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
  `COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT '',
  `DATABASE_COLLATION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";

static const std::string SCHEMA_SQL_USER_PRIVILEGES = R"(
-- Table structure for table `USER_PRIVILEGES`
-- DROP TABLE IF EXISTS `USER_PRIVILEGES`;
CREATE TEMPORARY TABLE `USER_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8;)";

/*
CREATE TABLE `user` (
  `Host` char(60) COLLATE utf8_bin NOT NULL DEFAULT '',
  `User` char(32) COLLATE utf8_bin NOT NULL DEFAULT '',
  `Select_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Insert_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Update_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Delete_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Create_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Drop_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Reload_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Shutdown_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Process_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `File_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Grant_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `References_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Index_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Alter_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Show_db_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Super_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Create_tmp_table_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Lock_tables_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Execute_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Repl_slave_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Repl_client_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Create_view_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Show_view_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Create_routine_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Alter_routine_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Create_user_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Event_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Trigger_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `Create_tablespace_priv` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `ssl_type` enum('','ANY','X509','SPECIFIED') CHARACTER SET utf8 NOT NULL DEFAULT '',
  `ssl_cipher` blob NOT NULL,
  `x509_issuer` blob NOT NULL,
  `x509_subject` blob NOT NULL,
  `max_questions` int(11) unsigned NOT NULL DEFAULT '0',
  `max_updates` int(11) unsigned NOT NULL DEFAULT '0',
  `max_connections` int(11) unsigned NOT NULL DEFAULT '0',
  `max_user_connections` int(11) unsigned NOT NULL DEFAULT '0',
  `plugin` char(64) COLLATE utf8_bin NOT NULL DEFAULT 'mysql_native_password',
  `authentication_string` text COLLATE utf8_bin,
  `password_expired` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  `password_last_changed` timestamp NULL DEFAULT NULL,
  `password_lifetime` smallint(5) unsigned DEFAULT NULL,
  `account_locked` enum('N','Y') CHARACTER SET utf8 NOT NULL DEFAULT 'N',
  PRIMARY KEY (`Host`,`User`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8 COLLATE=utf8_bin COMMENT='Users and global privileges';
*/
static const std::string SCHEMA_SQL_MYSQL_USER = R"(
CREATE TABLE `user` (
  `Host` char(60) COLLATE utf8_bin NOT NULL DEFAULT '',
  `User` char(32) COLLATE utf8_bin NOT NULL DEFAULT '',
  `Select_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Insert_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Update_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Delete_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Create_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Drop_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Reload_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Shutdown_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Process_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `File_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Grant_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `References_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Index_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Alter_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Show_db_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Super_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Create_tmp_table_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Lock_tables_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Execute_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Repl_slave_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Repl_client_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Create_view_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Show_view_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Create_routine_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Alter_routine_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Create_user_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Event_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Trigger_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `Create_tablespace_priv` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `ssl_type` char(16) COLLATE utf8_bin NOT NULL DEFAULT '',
  `ssl_cipher` char(128) COLLATE utf8_bin NOT NULL DEFAULT '',
  `x509_issuer` char(16) COLLATE utf8_bin NOT NULL DEFAULT '',
  `x509_subject` char(16) COLLATE utf8_bin NOT NULL DEFAULT '',
  `max_questions` int(11) NOT NULL DEFAULT '0',
  `max_updates` int(11) NOT NULL DEFAULT '0',
  `max_connections` int(11) NOT NULL DEFAULT '0',
  `max_user_connections` int(11) NOT NULL DEFAULT '0',
  `plugin` char(64) COLLATE utf8_bin NOT NULL DEFAULT 'mysql_native_password',
  `authentication_string` text COLLATE utf8_bin,
  `password_expired` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  `password_last_changed` timestamp NULL DEFAULT NULL,
  `password_lifetime` smallint(5) DEFAULT NULL,
  `account_locked` char(1) COLLATE utf8_bin NOT NULL DEFAULT 'N',
  PRIMARY KEY (`User`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8 COLLATE=utf8_bin COMMENT='Users and global privileges';
)";

static const vector< std::string > INFORMATION_SCHEMA_TABLE_SQLS =
{
    SCHEMA_SQL_TABLES, // 1
    SCHEMA_SQL_COLUMNS,
    SCHEMA_SQL_SCHEMATA,
    SCHEMA_SQL_VIEWS,
    SCHEMA_SQL_TABLE_CONSTRAINTS, // 5
    SCHEMA_SQL_KEY_COLUMN_USAGE,
    SCHEMA_SQL_DICTS,
    SCHEMA_SQL_DICT_COLUMN_USAGE,
    SCHEMA_SQL_GLOBAL_VARIABLES,
    SCHEMA_SQL_SESSION_VARIABLES, // 10
    SCHEMA_SQL_CHARACTER_SETS,
    SCHEMA_SQL_COLLATIONS,
    SCHEMA_SQL_COLLATION_CHARACTER_SET_APPLICABILITY,
    SCHEMA_SQL_COLUMN_PRIVILEGES,
    SCHEMA_SQL_ENGINES, // 15
    SCHEMA_SQL_EVENTS,
    SCHEMA_SQL_FILES,
    SCHEMA_SQL_GLOBAL_STATUS,
    SCHEMA_SQL_OPTIMIZER_TRACE,
    SCHEMA_SQL_PARAMETERS, // 20
    SCHEMA_SQL_PARTITIONS,
    SCHEMA_SQL_PLUGINS,
    SCHEMA_SQL_PROCESSLIST,
    SCHEMA_SQL_PROFILING,
    SCHEMA_SQL_REFERENTIAL_CONSTRAINTS, // 25
    SCHEMA_SQL_ROUTINES,
    SCHEMA_SQL_SCHEMA_PRIVILEGES,
    SCHEMA_SQL_SESSION_STATUS,
    SCHEMA_SQL_STATISTICS,
    SCHEMA_SQL_TABLESPACES, // 30
    SCHEMA_SQL_TABLE_PRIVILEGES,
    SCHEMA_SQL_TRIGGERS,
    SCHEMA_SQL_USER_PRIVILEGES
};

bool IsSysDb(const string& name) {
    auto lowerName = name;
    lowerName = aries_utils::to_lower(lowerName);
    for (auto& db : SYS_DATABASES) {
        if (lowerName == db) {
            return true;
        }
    }
    return false;
}
ColumnType ToColumnType(const string& arg_col_type_str)
{
    auto col_type_str = boost::algorithm::to_lower_copy( arg_col_type_str );
    ColumnType  column_type = schema::ColumnType::UNKNOWN;
    if (col_type_str == "int" || col_type_str == "integer" || col_type_str == "mediumint") {
        column_type = schema::ColumnType::INT;
    } else if (col_type_str == "tinyint") {
        column_type = schema::ColumnType::TINY_INT;
    } else if (col_type_str == "smallint") {
        column_type = schema::ColumnType::SMALL_INT;
    } else if (col_type_str == "bigint") {
        column_type = schema::ColumnType::LONG_INT;
    } else if (col_type_str == "float") {
        column_type = schema::ColumnType::FLOAT;
    } else if (col_type_str == "double") {
        column_type = schema::ColumnType::DOUBLE;
    } else if (col_type_str == "decimal" ||
               col_type_str == "numeric") {
        column_type = schema::ColumnType::DECIMAL;
    } else if (col_type_str == "text" || col_type_str == "char" || col_type_str == "varchar") {
        column_type = schema::ColumnType::TEXT;
    } else if (col_type_str == "date") {
        column_type = schema::ColumnType ::DATE;
    } else if (col_type_str == "datetime") {
        column_type = schema::ColumnType ::DATE_TIME;
    } else if (col_type_str == "time") {
        column_type = schema::ColumnType ::TIME;
    } else if (col_type_str == "timestamp") {
        column_type = schema::ColumnType ::TIMESTAMP;
    } else if (col_type_str == "year") {
        column_type = schema::ColumnType::YEAR;
    } else if (col_type_str == "bool") {
        column_type = schema::ColumnType::BOOL;
    } else if (col_type_str == "varbinary") {
        column_type = schema::ColumnType::VARBINARY;
    } else if (col_type_str == "binary") {
        column_type = schema::ColumnType::BINARY;
    } else {
        string msg("data type ");
        msg.append(col_type_str);
        ThrowNotSupportedException(msg);
    }
    return column_type;

}
Schema::Schema() {

}

Schema::~Schema() {
    databases.clear();
}

static std::vector<TableEntrySPtr> LoadTableSchemaFromSql(const string& sql, const DatabaseEntrySPtr& db) {
    std::vector<AriesSQLStatementPointer> statements;
    std::pair<int, string> parseResult = aries::SQLExecutor::GetInstance()->ParseSQL( sql, false, statements );
    ARIES_ASSERT ( 0 == parseResult.first, "Failed to load schema: " + std::to_string( parseResult.first ) );
    std::vector<TableEntrySPtr> tables;
    for (auto &statement : statements) {
        ARIES_ASSERT( statement->IsCommand(), "Failed to load schema: not command");
        auto command = std::dynamic_pointer_cast<CommandStructure>(statement->GetCommand());
        tables.emplace_back( aries::CommandToTableEntry( command.get(), db ) );
    }
    return tables;
}


bool fix_column_length(const std::string& db_root_path) {
    auto schema = aries::schema::SchemaManager::GetInstance()->GetSchema();

    for (const auto& db_item : schema->GetDatabases()) {
        auto db = db_item.second;
        auto db_name = db->GetName();
        aries_utils::to_lower(db_name);

        auto db_root = db_root_path + "/" + db_name;

        for (const auto& table : db->GetTables()) {
            auto table_name = table.first;
            auto table_path = db_root + "/" + table_name;
            auto metaFilePath = table_path + "/" + ARIES_INIT_TABLE_META_FILE_NAME;
            uint64_t totalRowCount;
            int ret = aries_engine::AriesInitialTable::GetTotalRowCount( metaFilePath, totalRowCount );
            if ( 0 != ret )
            {
                continue;
            }
            table.second->SetRowCount( totalRowCount );

            for (size_t i = 0; i < table.second->GetColumnsCount(); i ++) {
                auto column = table.second->GetColumnById(i + 1);

                switch (column->GetType()) {
                    case ColumnType::TEXT:
                    case ColumnType::VARBINARY:
                    case ColumnType::BINARY:
                        break;
                    default: {
                        continue;
                    }
                }

                if ( EncodeType::DICT == column->encode_type )
                    continue;

                auto column_path = table.second->GetColumnLocationString_ByIndex(i); //table_path + std::to_string(i);
                int8_t has_null;
                uint16_t item_len;
                BlockFileHeader headerInfo;
                ret = GetBlockFileHeaderInfo( column_path + "_0", headerInfo );

                if ( ret == 0 )
                {
                    item_len = headerInfo.itemLen;
                    has_null = headerInfo.containNull;
                    column->FixLength(item_len - has_null);
                }
                else
                {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool IsInforSchemaBaseTables( const string& schemaName, const string& tableName)
{
    return ( INFORMATION_SCHEMA == schemaName &&
             (SCHEMATA == tableName || "tables" == tableName ||
              "views" == tableName || "columns" == tableName) );

}
bool Schema::LoadSchema()
{
    databases.clear();

    bool ret = LoadBaseSchema();
    if ( !ret )
    {
        return ret;
    }

    ret = LoadSchemata();
    if ( !ret )
    {
        return ret;
    }

    ret = LoadTables();
    if ( !ret )
    {
        return ret;
    }

    ret = LoadColumns();
    if ( !ret )
    {
        return ret;
    }

    ret = LoadConstraints();
    if ( !ret )
    {
        return ret;
    }

    ret = LoadKeys();
    if ( !ret )
    {
        return ret;
    }

    ret = aries::AriesDictManager::GetInstance().Init();
    if ( !ret )
    {
        return ret;
    }

    ret = LoadColumnDictUsage();
    if ( !ret )
    {
        return ret;
    }

    if( !LoadTablePartitions() )
    {
        return false;
    }

    // ret = fix_column_length( aries::Configuartion::GetInstance().GetDataDirectory() );
    // if ( !ret )
    // {
    //     return ret;
    // }

    for ( auto& dbPair : databases )
    {
        for ( auto& tablePair : dbPair.second->GetTables() )
        {
            tablePair.second->SetCreateString( BuildCreateTableString( tablePair.second ) );
        }
    }
    return true;
}

bool Schema::LoadTablePartitions()
{
    auto sql = "select * from PARTITIONS order by table_schema, table_name, PARTITION_ORDINAL_POSITION;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA );
    if ( !result->IsSuccess() )
    {
        return false;
    }

    std::map< std::string, std::string > constraints;
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();

    auto schemaBuffer = table->GetColumnBuffer( 2 );
    auto tableBuffer = table->GetColumnBuffer( 3 );
    auto partitionNameBuffer = table->GetColumnBuffer( 4 );
    auto ordinalPosBuffer = table->GetColumnBuffer( 6 );
    auto partitionMethodBuffer = table->GetColumnBuffer( 8 );
    auto partitionExpr = table->GetColumnBuffer( 10 );
    auto partitionDesc = table->GetColumnBuffer( 12 );

    for ( int i = 0; i < table->GetRowCount(); i++ )
    {
        auto partition = std::make_shared< TablePartition >();

        auto dbName = schemaBuffer->GetString( i );
        auto tableName = tableBuffer->GetString( i );
        auto dbEntry = GetDatabaseByName( dbName );
        auto tableEntry = dbEntry->GetTableByName( tableName );

        auto partMethod = partitionMethodBuffer->GetString( i );
        auto partExpr = partitionExpr->GetString( i );
        aries_utils::to_lower( partExpr );
        auto columnEntry = tableEntry->GetColumnByName( partExpr );

        partition->m_partitionName = partitionNameBuffer->GetString( i );
        partition->m_partOrdPos = ordinalPosBuffer->GetNullableInt64( i ).value;
        partition->m_partDesc = partitionDesc->GetString( i );
        if ( "MAXVALUE" != partition->m_partDesc )
        {
            partition->m_value = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( partition->m_partDesc ).toTimestamp();
        }
        else
        {
            partition->m_value = INT64_MAX;
            partition->m_isMaxValue = true;
        }

        tableEntry->SetPartitionTypeDefInfo(
            partMethod,
            columnEntry->GetColumnIndex(),
            partExpr );
        tableEntry->AddPartition( partition );
    }
    return true;
}

bool Schema::LoadBaseSchema()
{
    CURRENT_TABLE_ID = 0;
    std::string inforschemaDir = Configuartion::GetInstance().GetDataDirectory( INFORMATION_SCHEMA );
    boost::filesystem::path p1(inforschemaDir);
    if (!boost::filesystem::exists(p1)) {
        LOG(ERROR) << "Failed to load schema: directory " << inforschemaDir << " not exists.\n";
        return false;
    }
    auto infoSchemaDb = std::make_shared<DatabaseEntry>( INFORMATION_SCHEMA );

    databases[ INFORMATION_SCHEMA ] = infoSchemaDb;

    for ( const auto& sql : INFORMATION_SCHEMA_TABLE_SQLS )
    {
        auto tableEntry = LoadTableSchemaFromSql( sql, infoSchemaDb )[0];
        tableEntry->SetId( ++CURRENT_TABLE_ID );
        infoSchemaDb->PutTable( tableEntry );
    }

    return true;
}

// load data from information_schema.schemata
bool Schema::LoadSchemata()
{
    string sql = "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM SCHEMATA";
    SQLResultPtr sqlResultPtr = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, INFORMATION_SCHEMA);
    if ( !sqlResultPtr->IsSuccess() )
        return false;
    const vector<aries::AbstractMemTablePointer>& results = sqlResultPtr->GetResults();
    auto amtp = results[0];
    auto table = ( ( AriesMemTable * )amtp.get() )->GetContent();
    int tupleNum;
    int columnCount = table->GetColumnCount();

    tupleNum = table->GetRowCount();
    std::vector<AriesDataBufferSPtr> columns;
    for (int col = 1; col < columnCount + 1; col++) {
        columns.push_back(table->GetColumnBuffer(col));
    }
    for (int tid = 0; tid < tupleNum; tid++) {
        auto schemaName = columns[0]->GetString( tid );
        if ( INFORMATION_SCHEMA == schemaName )
        {
            continue;
        }
        auto defaultCsName = columns[1]->GetString( tid );
        auto defaultCollationName = columns[2]->GetString( tid );
        databases[ schemaName ] = std::make_shared<DatabaseEntry>( schemaName, defaultCsName, defaultCollationName );
    }

    return true;
}

// load data from information_schema.tables
bool Schema::LoadTables()
{
    string sql = "SELECT TABLE_ID, TABLE_SCHEMA, TABLE_NAME, ENGINE, TABLE_COLLATION FROM TABLES";
    SQLResultPtr sqlResultPtr = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, INFORMATION_SCHEMA);
    if ( !sqlResultPtr->IsSuccess() )
        return false;
    const vector<aries::AbstractMemTablePointer>& results = sqlResultPtr->GetResults();
    auto amtp = results[0];
    auto table = ( ( AriesMemTable * )amtp.get() )->GetContent();
    int tupleNum;
    int columnCount = table->GetColumnCount();

    tupleNum = table->GetRowCount();
    std::vector<AriesDataBufferSPtr> columns;
    for (int col = 1; col < columnCount + 1; col++) {
        columns.push_back(table->GetColumnBuffer(col));
    }
    for (int tid = 0; tid < tupleNum; tid++) {
        auto tableId = columns[0]->GetInt64( tid );
        auto schemaName = columns[1]->GetString( tid );
        auto tableName = columns[2]->GetString( tid );
        if ( IsInforSchemaBaseTables( schemaName, tableName ) )
        {
            continue;
        }
        string tableEngine;
        if ( !columns[3]->isStringDataNull( tid ) )
            tableEngine = columns[3]->GetNullableString( tid );
        string tableCollation;
        if ( !columns[4]->isStringDataNull( tid ) )
            tableCollation = columns[4]->GetNullableString( tid );
        LOG(INFO) << "table " << schemaName << "." << tableName << ", engine " << tableEngine;
        auto db = GetDatabaseByName( schemaName );
        ARIES_ASSERT( db, "Failed to load information_schema.tables" );
        auto tableEntry = std::make_shared<TableEntry>( db, tableId, tableName, tableEngine, tableCollation );
        db->PutTable( tableEntry );
    }

    return true;
}

/*
 *
CREATE TEMPORARY TABLE `information_schema`.`COLUMNS` (
`TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '', // 0
`TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',   // 1
`TABLE_NAME` varchar(64) NOT NULL DEFAULT '',     // 2
`COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',    // 3
`ORDINAL_POSITION` bigint(21) NOT NULL DEFAULT '0', // 4
`COLUMN_DEFAULT` longtext, // 5
`IS_NULLABLE` varchar(3) NOT NULL DEFAULT '', // 6
`DATA_TYPE` varchar(64) NOT NULL DEFAULT '', // 7
`CHARACTER_MAXIMUM_LENGTH` bigint(21) DEFAULT NULL, // 8
`CHARACTER_OCTET_LENGTH` bigint(21) DEFAULT NULL, // 9
`NUMERIC_PRECISION` bigint(21) DEFAULT NULL, // 10
`NUMERIC_SCALE` bigint(21) DEFAULT NULL, // 11
`DATETIME_PRECISION` bigint(21) DEFAULT NULL, // 12
`CHARACTER_SET_NAME` varchar(32) DEFAULT NULL, // 13
`COLLATION_NAME` varchar(32) DEFAULT NULL, // 14
`COLUMN_TYPE` longtext NOT NULL, // 15
`COLUMN_KEY` varchar(3) NOT NULL DEFAULT '', // 16
`EXTRA` varchar(30) NOT NULL DEFAULT '', // 17
`PRIVILEGES` varchar(80) NOT NULL DEFAULT '', // 18
`COLUMN_COMMENT` varchar(1024) NOT NULL DEFAULT '', // 19
`GENERATION_EXPRESSION` longtext NOT NULL // 20
) ENGINE=InnoDB DEFAULT CHARSET=utf8;)";
 */
// load data from information_schema.columns
bool Schema::LoadColumns()
{
    string sql = "SELECT * FROM COLUMNS ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION ASC";
    SQLResultPtr sqlResultPtr = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, INFORMATION_SCHEMA);
    if ( !sqlResultPtr->IsSuccess() )
        return false;
    const vector<aries::AbstractMemTablePointer>& results = sqlResultPtr->GetResults();
    auto amtp = results[0];
    auto table = ( ( AriesMemTable * )amtp.get() )->GetContent();
    int tupleNum;
    int columnCount = table->GetColumnCount();

    tupleNum = table->GetRowCount();
    std::vector<AriesDataBufferSPtr> columns;
    for (int col = 1; col < columnCount + 1; col++) {
        columns.push_back(table->GetColumnBuffer(col));
    }
    for (int tid = 0; tid < tupleNum; tid++) {
        auto schemaName = columns[1]->GetString( tid );
        auto tableName = columns[2]->GetString( tid );
        if ( IsInforSchemaBaseTables( schemaName, tableName ) )
        {
            continue;
        }
        auto dbEntry = GetDatabaseByName( schemaName );
        ARIES_ASSERT( dbEntry, "Failed to load columns of table " + schemaName + "." + tableName );
        SQLTreeNodePointer viewNode;
        auto tableEntry = dbEntry->GetTableByName( tableName );
        // if ( !tableEntry )
        //     viewNode = ViewManager::GetInstance().GetViewNode( tableName, schemaName );
        // ARIES_ASSERT( tableEntry || viewNode, "Failed to load columns of table " + schemaName + "." + tableName );
        if ( !tableEntry ) // column of a view
            continue;

        auto columnName = columns[3]->GetString( tid );
        auto position = columns[4]->GetInt64( tid );
        ARIES_ASSERT( position >= 1, "Column ordinal position should start with 1");

        std::shared_ptr<string> defaultVal;
        if ( !columns[5]->isStringDataNull( tid ) ) {
            defaultVal = std::make_shared<string>( columns[5]->GetNullableString( tid ) );
        }

        auto nullable = columns[6]->GetString( tid );
        aries_utils::to_lower( nullable );
        bool bNullable = ("yes" == nullable);

        auto dataType = columns[7]->GetString( tid );

        int64_t charMaxLen = -1;
        if ( !columns[8]->isInt64DataNull( tid ) ) {
            charMaxLen = columns[8]->GetNullableInt64( tid ).value;
        }
        int64_t charOctLen = -1;
        if ( !columns[9]->isInt64DataNull( tid ) ) {
            charOctLen = columns[9]->GetNullableInt64(tid).value;
        }
        int64_t numericPrecision = -1;
        if ( !columns[10]->isInt64DataNull( tid ) ) {
            numericPrecision = columns[10]->GetNullableInt64(tid).value;
        }
        int64_t numericScale = -1;
        if ( !columns[11]->isInt64DataNull( tid ) ) {
            numericScale = columns[11]->GetNullableInt64(tid).value;
        }
        int64_t datetimePrecision = -1;
        if ( !columns[12]->isInt64DataNull( tid ) ) {
            datetimePrecision = columns[12]->GetNullableInt64(tid).value;
        }

        string csName = DEFAULT_CHARSET_NAME;
        if ( !columns[13]->isStringDataNull( tid ) )
        {
            csName = columns[13]->GetNullableString( tid );
        }
        string collationName = DEFAULT_UTF8_COLLATION;
        if ( !columns[14]->isStringDataNull( tid ) )
        {
            collationName = columns[14]->GetNullableString( tid );
        }
        auto columnType = columns[15]->GetString( tid );
        aries_utils::to_lower( columnType );
        bool bUnsigned = (string::npos != columnType.find( "unsigned" ));
        auto columnKey = columns[16]->GetString( tid ); // PRI, UNI, MUL
        aries_utils::to_lower( columnKey );
        bool bPrimaryKey = (string::npos != columnKey.find( "pri" ));
        bool bUniqueKey = (string::npos != columnKey.find( "uni" ));
        bool bMultiKey = (string::npos != columnKey.find( "mul" ));
        // foreigne key constraint: REFERENTIAL_CONSTRAINTS table;
        bool bForeignKey = false;

        string extra = columns[17]->GetString( tid );
        aries_utils::to_lower( extra );
        // extra.
        bool bHasDefault = false;
        auto extraAttrs = aries_utils::split( extra, ";" );
        for ( const auto& attr : extraAttrs )
        {
            auto attrKV = aries_utils::split( attr, ":" );
            if ( "d" == attrKV[ 0 ] )
            {
                bHasDefault = true;
            }
        }

        auto columnEntry = schema::ColumnEntry::MakeColumnEntry(
                columnName, ToColumnType( dataType ),  position - 1,
                bPrimaryKey, bUniqueKey, bMultiKey, bForeignKey,
                bUnsigned, bNullable, bHasDefault, defaultVal,
                charMaxLen, charOctLen,
                numericPrecision, numericScale, datetimePrecision,
                csName, collationName);

        tableEntry->AddColumn( columnEntry );
    }

    return true;
}

static TableConstraintType GetConstraintType( const std::string& typeName )
{
    static std::map< std::string, TableConstraintType > constraint_type_map =
    {
        { "PRIMARY KEY", TableConstraintType::PrimaryKey },
        { "PRIMARY", TableConstraintType::PrimaryKey },
        { "FOREIGN KEY", TableConstraintType::ForeignKey },
        { "FOREIGN", TableConstraintType::ForeignKey },
        { "UNIQUE", TableConstraintType::UniqueKey }
    };

    auto name = aries_utils::convert_to_upper( typeName );

    ARIES_ASSERT( constraint_type_map.find( name ) != constraint_type_map.cend(), "invalid constraint type: " + typeName );

    return constraint_type_map[ name ];
}

bool Schema::LoadConstraints()
{
    auto sql = "select table_schema, table_name, constraint_name, constraint_type from table_constraints order by table_schema, table_name;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA );
    if ( !result->IsSuccess() )
    {
        return false;
    }

    std::map< std::string, std::string > constraints;
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();

    auto schema_buffer = table->GetColumnBuffer( 1 );
    auto table_buffer = table->GetColumnBuffer( 2 );
    auto constraint_name_buffer = table->GetColumnBuffer( 3 );
    auto constraint_type_buffer = table->GetColumnBuffer( 4 );

    for ( int i = 0; i < table->GetRowCount(); i++ )
    {
        auto db_name = schema_buffer->GetString( i );
        auto table_name = table_buffer->GetString( i );

        auto database = GetDatabaseByName( db_name );
        auto table_entry = database->GetTableByName( table_name );

        auto constraint = std::make_shared< TableConstraint >();
        constraint->name = constraint_name_buffer->GetString( i );
        constraint->type = GetConstraintType( constraint_type_buffer->GetString( i ) );

        table_entry->AddConstraint( constraint );
    }
    return true;
}

bool Schema::LoadColumnDictUsage()
{
    auto sql = "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DICT_ID FROM DICT_COLUMN_USAGE ORDER BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA );
    if ( !result->IsSuccess() )
    {
        return false;
    }

    std::map< std::string, std::string > constraints;
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();

    auto schemaBuffer = table->GetColumnBuffer( 1 );
    auto tableBuffer  = table->GetColumnBuffer( 2 );
    auto columnBuffer = table->GetColumnBuffer( 3 );
    auto dictIdBuffer = table->GetColumnBuffer( 4 );

    for ( int i = 0; i < table->GetRowCount(); i++ )
    {
        auto dbName = schemaBuffer->GetString( i );
        auto tableName = tableBuffer->GetString( i );
        auto columnName = columnBuffer->GetString( i );
        auto dictId = dictIdBuffer->GetInt64( i );

        auto database = GetDatabaseByName( dbName );
        if ( !database )
        {
            LOG( WARNING ) << "Stray dict usage: " << dbName << "." << tableName << "." << columnName << ", dict: " << dictId;
            continue;
        }
        auto tableEntry = database->GetTableByName( tableName );
        if ( !tableEntry )
        {
            LOG( WARNING ) << "Stray dict usage: " << dbName << "." << tableName << "." << columnName << ", dict: " << dictId;
            continue;
        }
        auto colEntry = tableEntry->GetColumnByName( columnName );
        if ( !colEntry )
        {
            LOG( WARNING ) << "Stray dict usage: " << dbName << "." << tableName << "." << columnName << ", dict: " << dictId;
            continue;
        }

        auto dict = AriesDictManager::GetInstance().GetDict( dictId );
        colEntry->SetDict( dict );
    }

    return true;
}

bool Schema::LoadKeys()
{
    auto sql = R"(
select
    constraint_name,
    table_schema,
    table_name,
    column_name,
    referenced_table_schema,
    referenced_table_name,
    referenced_column_name
from
    key_column_usage
order by
    table_schema, table_name, ordinal_position ASC;
    )";

    auto sqlResultPtr = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA );
    if ( !sqlResultPtr->IsSuccess() )
        return false;

    const vector< aries::AbstractMemTablePointer >& results = sqlResultPtr->GetResults();
    auto amtp = results[ 0 ];
    auto table = ( ( AriesMemTable * )amtp.get() )->GetContent();

    auto tupleNum = table->GetRowCount();
    std::vector< AriesDataBufferSPtr > columns;

    std::string db_name;
    std::string table_name;
    std::string column_name;
    std::string referenced_table_name;
    std::string referenced_column_name;

    auto constraint_name_buffer         = table->GetColumnBuffer( 1 );
    auto table_schema_buffer            = table->GetColumnBuffer( 2 );
    auto table_name_buffer              = table->GetColumnBuffer( 3 );
    auto column_name_buffer             = table->GetColumnBuffer( 4 );
    auto referenced_table_schema_buffer = table->GetColumnBuffer( 5 );
    auto referenced_table_name_buffer   = table->GetColumnBuffer( 6 );
    auto referenced_column_name_buffer  = table->GetColumnBuffer( 7 );

    DatabaseEntrySPtr database;
    TableEntrySPtr table_entry;

    for ( int i = 0; i < tupleNum; i++ )
    {
        auto db_name = table_schema_buffer->GetString( i );
        auto table_name = table_name_buffer->GetString( i );
        auto column_name = column_name_buffer->GetString( i );

        auto constraint_name = constraint_name_buffer->GetString( i );
        if ( constraint_name.empty() )
            continue;

        database = GetDatabaseByName( db_name );
        table_entry = database->GetTableByName( table_name );

        table_entry->AddConstraintKey( column_name, constraint_name );

        if ( !referenced_table_name_buffer->isStringDataNull( i ) &&
             !referenced_table_schema_buffer->isStringDataNull( i ) &&
             !referenced_column_name_buffer->isStringDataNull( i ) )
        {
            auto referenced_table_name = referenced_table_name_buffer->GetString( i );
            auto referenced_table_schema = referenced_table_schema_buffer->GetString( i );
            auto referenced_column_name =  referenced_column_name_buffer->GetString( i );
            if ( referenced_table_name.empty() || referenced_table_schema.empty() || referenced_column_name.empty() )
                continue;

            table_entry->SetConstraintReferencedTableName( referenced_table_schema, referenced_table_name, constraint_name );
            table_entry->AddConstraintReferencedKey( referenced_column_name, constraint_name );

            auto referenced_table = GetDatabaseByName( referenced_table_schema )->GetTableByName( referenced_table_name );
            ARIES_ASSERT(referenced_table, "inivalid databse: " + referenced_table_schema + ", and table: " + referenced_table_name );

            referenced_table->AddReferencing( referenced_column_name, column_name, table_name, db_name, constraint_name, table_entry->GetId() );
        }
    }

    return true;
}


void Schema::InsertDbSchema(const string& dbName) {

    std::string sql = "insert into `" + SCHEMATA + "` values (";
    sql += " 'def'";
    sql += ", '" + dbName + "'";
    sql += ", '" + std::string( DEFAULT_CHARSET_NAME ) + "'";
    sql += ", '" + std::string( DEFAULT_UTF8_COLLATION ) + "'";
    sql += ", null";
    sql += ");";

    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

void Schema::DeleteDbSchema(const string& dbName) {
    auto name = dbName;
    aries_utils::to_lower(name);

    keys_to_remove.emplace_back(name);
    databases.erase(name);
}

void Schema::InsertEngineRows() {
    std::string sql = "insert into `engines` values('Rateup', 'DEFAULT', 'Rateup column store', 'NO', 'NO', 'NO')";
    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}
void Schema::InsertCollationCharsetApplicabilityRows() {

    std::string sql = "insert into `collation_character_set_applicability` values ";
    sql += "(";
    sql += "'" + std::string( DEFAULT_UTF8_COLLATION ) + "'";
    sql += ", '" + std::string( DEFAULT_CHARSET_NAME ) + "'";
    sql += ")";

    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}
void Schema::InsertCollationRows() {
    std::string sql = "insert into `collations` values ";
    sql += "(";
    sql += "'" + std::string( DEFAULT_UTF8_COLLATION ) + "'";
    sql += ", '" + std::string( DEFAULT_CHARSET_NAME ) + "'";
    sql += ", 33";
    sql += ", 'YES', 'YES'";
    sql += ", 1";
    sql += ")";

    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}
void Schema::InsertCharsetsRows() {
    std::string sql = "insert into `character_sets` values ";
    sql += "(";
    sql += "'" + std::string( DEFAULT_UTF8_COLLATION ) + "'";
    sql += ", '" + std::string( DEFAULT_CHARSET_NAME ) + "'";
    sql += ", 'UTF-8 Unicode'";
    sql += ", 'YES', 'YES'";
    sql += ", 3";
    sql += ")";

    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

static void InsertTableConstraints(const string& db, const TableEntrySPtr& table )
{

    std::string sql = "insert into `table_constraints` values ";
    bool has_data = false;
    if ( !table->GetPrimaryKey().empty() )
    {
        sql += "(";
        sql += "'def'";
        sql += ",\"" + db + "\"";
        sql += ", \"" + table->GetPrimaryKeyName() + "\"";
        sql += ", \"" + db + "\"";
        sql += ", \"" + table->GetName() + "\"";
        sql += ", 'PRIMARY KEY'";
        sql += ")";
        has_data = true;
    }

    for ( const auto& key : table->GetForeignKeys() )
    {
        if ( has_data )
        {
            sql += ", ";
        }
        has_data = true;
        sql += "(";
        sql += "'def'";
        sql += ",\"" + db + "\"";
        sql += ", \"" + key->name + "\"";
        sql += ", \"" + db + "\"";
        sql += ", \"" + table->GetName() + "\"";
        sql += ", 'FOREIGN KEY'";
        sql += ")";
    }

    for ( const auto& key : table->GetUniqueKeys() )
    {
        if ( has_data )
        {
            sql += ", ";
        }
        has_data = true;
        sql += "(";
        sql += "'def'";
        sql += ",\"" + db + "\"";
        sql += ", \"" + key->name + "\"";
        sql += ", \"" + db + "\"";
        sql += ", \"" + table->GetName() + "\"";
        sql += ", 'UNIQUE'";
        sql += ")";
    }

    if ( has_data )
    {
        sql += ";";
        CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
    }
}

static void InsertKeyColumnUsage(const string& db, const TableEntrySPtr& table )
{
    std::string sql = "insert into `key_column_usage` values";
    bool has_data = false;

    int64_t position = 1;
    for ( const auto& key : table->GetPrimaryKey() )
    {
        if ( has_data )
        {
            sql += ", ";
        }
        has_data = true;

        sql += "(";
        sql += "'def'";
        sql += ", '" + db + "'";
        sql += ", '" + table->GetPrimaryKeyName() + "'";
        sql += ", 'def'";
        sql += ", '" + db + "'";
        sql += ", '" + table->GetName() + "'";
        sql += ", '" + key + "'";
        sql += ", " + std::to_string( position++ );
        sql += ", null, null, null, null)";
    }

    for ( const auto& foreign_key : table->GetForeignKeys() )
    {
        position = 1;
        for ( size_t i = 0; i < foreign_key->keys.size(); i++ )
        {
            const auto& key = foreign_key->keys[ i ];
            if ( has_data )
            {
                sql += ", ";
            }
            has_data = true;

            sql += "(";
            sql += "'def'";
            sql += ", '" + db + "'";
            sql += ", '" + foreign_key->name + "'";
            sql += ", 'def'";
            sql += ", '" + db + "'";
            sql += ", '" + table->GetName() + "'";
            sql += ", '" + key + "'";
            sql += ", " + std::to_string( position );
            sql += ", " + std::to_string( position );
            position++;
            sql += ", '" + foreign_key->referencedSchema + "'";
            sql += ", '" + foreign_key->referencedTable + "'";
            sql += ", '" + foreign_key->referencedKeys[ i ] + "'";
            sql += ")";
        }
    }

    for ( const auto& uniq_key : table->GetUniqueKeys() )
    {
        position = 1;
        for ( size_t i = 0; i < uniq_key->keys.size(); i++ )
        {
            const auto& key = uniq_key->keys[ i ];
            if ( has_data )
            {
                sql += ", ";
            }
            has_data = true;

            sql += "(";
            sql += "'def'";
            sql += ", '" + db + "'";
            sql += ", '" + uniq_key->name + "'";
            sql += ", 'def'";
            sql += ", '" + db + "'";
            sql += ", '" + table->GetName() + "'";
            sql += ", '" + key + "'";
            sql += ", " + std::to_string( position++ );
            sql += ", null, null, null, null)";
        }
    }

    if ( has_data )
    {
        CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
    }
}

static void InsertPartitionInfo(const string& db, const TableEntrySPtr& table )
{
    auto partitionMethod = table->GetPartitionMethod();
    if ( partitionMethod.empty() )
    {
        return;
    }

    auto paritionExprStr = table->GetPartitionExprStr();

    std::string sql = R"(insert into `partitions` values)";
//         `TABLE_CATALOG`,
//         `TABLE_SCHEMA`,
//         `TABLE_NAME`,
//         `PARTITION_NAME`,
//         `SUBPARTITION_NAME`,
//         `PARTITION_ORDINAL_POSITION`,
//         `SUBPARTITION_ORDINAL_POSITION`,
//         `PARTITION_METHOD`,
//         `SUBPARTITION_METHOD`,
//         `PARTITION_EXPRESSION`,
//         `SUBPARTITION_EXPRESSION`,
//         `PARTITION_DESCRIPTION` ) values)";

    const std::vector< TablePartitionSPtr > & partitions = table->GetPartitions();
    bool has_data = false;
    for ( auto& part : partitions )
    {
        if ( has_data )
        {
            sql += ", ";
        }
        has_data = true;

        sql += "(";
        sql += "'def'";
        sql += ", '" + db + "'";
        sql += ", '" + table->GetName() + "'";
        sql += ", '" + part->m_partitionName + "', null";
        sql += ", " + std::to_string( part->m_partOrdPos ) + ", null";
        sql += ", '" + partitionMethod + "', null";
        sql += ", '" + paritionExprStr + "', null";
        sql += ", '" + part->m_partDesc + "'";
        sql += ", 0, 0, 0, 0, 0, 0";
        sql += ", now(), null, null, null, '', '', null";
        sql += " )";
    }
    if ( has_data )
    {
        CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
    }
}

static void InsertDict(const string& db, const TableEntrySPtr& table )
{
    for ( auto& colEntry : table->GetColumns() )
    {
        if ( EncodeType::DICT == colEntry->encode_type )
        {
            AriesDictManager::GetInstance().AddDict( colEntry->GetDict() );
        }
    }
}

static void InsertDictColumnUsage(const string& db, const TableEntrySPtr& table )
{
    std::string sql = "insert into `dict_column_usage` values";
    bool has_data = false;

    for ( auto& colEntry : table->GetColumns() )
    {
        if ( EncodeType::DICT == colEntry->encode_type )
        {
            if ( has_data )
            {
                sql += ", ";
            }
            has_data = true;

            sql += "(";
            sql += "'def'";
            sql += ", '" + db + "'";
            sql += ", '" + table->GetName() + "'";
            sql += ", '" + colEntry->GetName() + "'";
            sql += ", " + std::to_string( colEntry->GetDictId() );
            sql += " )";
        }
    }
    if ( has_data )
    {
        CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
    }
}

static void InsertColumnsSchema(const string& db, const string& table, const std::vector<ColumnEntryPtr>& columns)
{
    std::string sql = "insert into `columns` values ";

    bool has_data = false;
    int64_t columnIdx = 1;
    for (auto& column : columns) {
        if ( has_data )
        {
            sql += ", ";
        }
        has_data = true;
        sql += "(";
        sql += "'def'";
        sql += ", '" + db  + "'";
        sql += ", '" + table + "'";
        sql += ", '" + column->GetName() + "'";
        sql += ", " + std::to_string( columnIdx );
        columnIdx++;
        if ( column->has_default )
        {
            if ( column->default_val_ptr )
            {
                if ( column->default_val_ptr->empty() )
                {
                    sql += ", ''";
                }
                else
                {
                    sql += ", '" + *(column->default_val_ptr) + "'";
                }
            }
            else
            {
                sql += ", null";
            }
        }
        else
        {
            sql += ", ''";
        }

        std::string allow_null = column->IsAllowNull() ? "YES" : "NO";
        sql += ", '" + allow_null + "'";

        string dataTypeString = DataTypeString(column->GetType());
        sql += ", '" + dataTypeString + "'";
        int64_t int64Tmp = 0;

        if ( ColumnEntry::IsStringType( column->GetType() ) )
        {
            int64Tmp = column->GetLength();
            sql += ", " + std::to_string( int64Tmp );
            sql += ", " + std::to_string( int64Tmp );
        }
        else
        {
            sql += ", null";
            sql += ", null";
        }

        if ( ColumnEntry::IsNumberType( column->GetType() ) )
        {
            sql += ", " + std::to_string( column->numeric_precision );
        }
        else
        {
            sql += ", null";
        }

        if ( ColumnEntry::IsNumberType( column->GetType() ))
        {
            if ( -1 == column->numeric_scale )
                sql += ", null";
            else
                sql += ", " + std::to_string( column->numeric_scale );
        }
        else
        {
            sql += ", null";
        }
        sql += ", null";
        sql += ", '" + column->charset_name + "'";
        sql += ", '" + column->collation_name + "'";

        string columnTypeString = ColumnTypeString(column);
        sql += ", '" + columnTypeString + "'";

        string key;
        if (column->IsPrimary())
        {
            key.append("PRI");
        }
        if ( column->is_unique )
        {
            if ( !key.empty() )
            {
                key.append(",");
            }
            key.append("UNI");
        }
        if ( column->is_multi )
        {
            if ( !key.empty() )
            {
                key.append(",");
            }
            key.append("MUL");
        }
        if ( key.empty() )
            key = "";

        sql += ", '" + key + "'";

        std::string tmpValue;
        if ( column->has_default )
            tmpValue.append("d");

        if ( tmpValue.empty() )
            tmpValue = "";
        sql += ", '" + tmpValue + "'";
        sql += ", 'select'";
        sql += ", ''";
        sql += ", ''";
        sql +=")";
    }

    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

std::mutex mutex_for_table_id;
int64_t get_auto_increment_table_id()
{
    static int64_t current = CURRENT_TABLE_ID;
    std::lock_guard< std::mutex > lock( mutex_for_table_id );
    if ( current == CURRENT_TABLE_ID )
    {
        auto result = SQLExecutor::GetInstance()->ExecuteSQL( "select max(TABLE_ID) from tables", INFORMATION_SCHEMA );
        if ( result->IsSuccess() && result->GetResults().size() == 1 )
        {
            auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
            auto table_block = mem_table->GetContent();

            if ( table_block->GetRowCount() == 1 )
            {
                ARIES_ASSERT( table_block->GetColumnCount() == 1, "cannot get max table id" );
                current = max( table_block->GetColumnBuffer( 1 )->GetNullableInt64( 0 ).value, CURRENT_TABLE_ID );
            }
        }
    }

    current ++;
    return current;
}

// insert table schema info to information_schema.tables
void Schema::InsertTableSchema(const string& dbName, const TableEntrySPtr& tableEntry) {
    auto table_id = tableEntry->GetId();
    std::string sql = "insert into `tables` values(";
    sql += std::to_string( table_id );
    sql += ", \"def\"";
    sql += ", \"" + dbName + "\"";
    sql += ", \"" + tableEntry->GetName() + "\"";
    if (IsSysDb(dbName)) {
        sql += ", \"SYSTEM VIEW\"";
    }
    else
    {
        sql += ", \"BASE TABLE\"";
    }
    sql += ", \"" + tableEntry->GetEngine() + "\"";
    sql += ", 1";
    sql.append( ", \"" ).append( PRODUCT_NAME ).append( "\"" );
    sql += ", 0";
    sql += ", 0";
    sql += ", 0";
    sql += ", 0";
    sql += ", 0";
    sql += ", 0";
    sql += ", 0";
    sql += ", now()";
    sql += ", now()";
    sql += ", now()";
    sql += ", \"" + tableEntry->GetCollation() + "\"";
    sql += ", 0";
    sql += ", ''";
    sql += ", \"\"";
    sql += ");";

    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, "information_schema" );
    LOG( INFO ) << "insert table schema: " << sql;
    if ( !result->IsSuccess() )
    {
        ARIES_EXCEPTION_SIMPLE( result->GetErrorCode(), result->GetErrorMessage() );
    }

    InsertColumnsSchema( dbName, tableEntry->GetName(), tableEntry->GetColumns() );

    InsertTableConstraints( dbName, tableEntry );

    InsertKeyColumnUsage( dbName, tableEntry );

    InsertPartitionInfo( dbName, tableEntry );

    InsertDict( dbName, tableEntry );
    InsertDictColumnUsage( dbName, tableEntry );
}

// insert table schema info to information_schema.views
void
Schema::InsertViewSchema(const string& dbName,
                         const string& tableName,
                         const std::string& create_string,
                         const std::vector<ColumnEntryPtr>& columns) {
    std::string sql = "insert into `VIEWS` values(";
    sql += " 'def'";
    sql += ", '" + dbName + "'";
    sql += ", '" + tableName + "'";
    sql += ", '" + aries_utils::escape_string( create_string ) + "'";
    sql += ", 'NONE'";
    sql += ", 'NO'";
    sql += ", 'mysql.sys@localhost'";
    sql += ", 'INVOKER'";
    sql += ", '" + std::string( DEFAULT_CHARSET_NAME ) + "'";
    sql += ", '" + std::string( DEFAULT_UTF8_COLLATION ) + "'";
    sql += ")";
    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

void Schema::DeleteProcess(uint64_t pid)
{
    std::string sql = "delete from `processlist` where id = " + std::to_string( pid ) + ";";
    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}


void Schema::InsertProcess(uint64_t pid,
                           const string& user,
                           const string& host,
                           const string& db,
                           const string& cmd,
                           int time,
                           const string& state,
                           const string& info)
{
    std::string sql = "insert into `processlist` values(";
    sql += std::to_string( pid );
    sql += ", '" + user + "'";
    sql += ", '" + host + "'";
    sql += ", '" + db + "'";
    sql += ", '" + cmd + "'";
    sql += ", " + std::to_string( time );
    sql += ", '" + state + "'";
    sql += ", '" + info + "'";
    sql += ");";

    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

struct TempTableData
{
    std::vector< int8_t* > buffers;
    std::vector< int > columns_size;
    std::vector< bool > is_nullable;
    int current_row_count;
    int columns_count;

    int8_t* GetData( int index ) const {
        auto buffer = buffers[ index ];
        buffer = buffer + ( current_row_count * columns_size[ index ] );
        if ( is_nullable[ index ] )
        {
            *buffer = 1;
            buffer += 1;
        }

        return buffer;
    }

    void SetNull( int index ) const {
        if ( !is_nullable[ index ] )
        {
            ARIES_EXCEPTION( ER_CANT_CREATE_DB, "this column is not nullable" );
        }
        auto buffer = buffers[ index ];
        buffer = buffer + ( current_row_count * columns_size[ index ] );
        *buffer = 0;
    }

    void SetInt( int64_t value, int index ) const
    {
        *( int64_t* )( GetData( index ) ) = value;
    }

    void SetString( const std::string& value, int index ) const
    {
        memcpy( GetData( index ), value.data(), value.size() );
    }

    void Commit()
    {
        current_row_count++;
    }

    void Free()
    {
        for ( auto buffer : buffers )
        {
            delete[] buffer;
        }
    }
};

static TempTableData table_to_temp( const TableEntrySPtr& table )
{
    TempTableData temp;
    temp.columns_count = table->GetColumnsCount();
    temp.current_row_count = 0;

    const auto& columns = table->GetColumns();
    for ( int i = 0; i < temp.columns_count; i++ )
    {
        auto& column = columns[ i ];
        auto length = column->GetTypeSize();

        if ( column->IsAllowNull() )
        {
            temp.is_nullable.emplace_back( true );
            length ++;
        }
        else
        {
            temp.is_nullable.emplace_back( false );
        }

        temp.columns_size.emplace_back( length );
        auto buffer = new int8_t[ MAX_SCHEMA_ROW_COUNT * length ];
        memset( buffer, 0, MAX_SCHEMA_ROW_COUNT * length );
        temp.buffers.emplace_back( buffer );
    }
    return temp;
}

/**
 * 将 table 的 column 的信息写入到 information_schema.columns 对应的 buffer 中
 */
static void column_insert_into_temp_data( TempTableData& temp_data, const TableEntrySPtr& table )
{
    int64_t column_index = 1;
    for ( const auto& column : table->GetColumns() )
    {
        int index = 0;
        temp_data.SetString( "def", index++ ); // catelog
        temp_data.SetString( "information_schema", index++ ); // table_schema
        temp_data.SetString( table->GetName(), index++ ); // table_name
        temp_data.SetString( column->GetName(), index++ ); // column_name
        temp_data.SetInt( column_index++, index++ ); // ORDINAL_POSITION
        if ( column->has_default )
        {
            if ( column->default_val_ptr )
            {
                temp_data.SetString( *( column->default_val_ptr ), index++ );
            }
            else
            {
                temp_data.SetNull( index++ );
            }
        }
        else
        {
            temp_data.SetString( "", index++ );
        }

        temp_data.SetString( column->IsAllowNull() ? "YES" : "NO", index++ );  // IS_NULLABLE
        temp_data.SetString( DataTypeString( column->GetType() ), index++ );  // DATA_TYPE
        if ( ColumnEntry::IsStringType( column->GetType() ) )
        {
            temp_data.SetInt( column->GetLength(), index++ ); // CHARACTER_MAXIMUM_LENGTH
            temp_data.SetInt( column->GetLength(), index++ ); // CHARACTER_OCTET_LENGTH
        }
        else
        {
            temp_data.SetNull( index++ ); // CHARACTER_MAXIMUM_LENGTH
            temp_data.SetNull( index++ ); // CHARACTER_OCTET_LENGTH
        }

        if ( ColumnEntry::IsNumberType( column->GetType() ) && -1 != column->numeric_scale )
        {
            temp_data.SetInt( column->numeric_scale, index++ ); // NUMERIC_SCALE
        }
        else
        {
            temp_data.SetNull( index++ ); // NUMERIC_SCALE
        }

        temp_data.SetNull( index++ ); // DATETIME_PRECISION
        temp_data.SetString( column->charset_name, index++ ); // CHARACTER_SET_NAME
        temp_data.SetString( column->collation_name, index++ ); // COLLATION_NAME
        temp_data.SetString( ColumnTypeString( column ), index++ ); // COLUMN_TYPE

        string key;
        if (column->IsPrimary())
        {
            key.append("PRI");
        }
        if ( column->is_unique )
        {
            if ( !key.empty() )
            {
                key.append(",");
            }
            key.append("UNI");
        }
        if ( column->is_multi )
        {
            if ( !key.empty() )
            {
                key.append(",");
            }
            key.append("MUL");
        }

        temp_data.SetString( key, index++ );  // COLUMN_KEY
        std::string tmpValue;
        if ( column->has_default )
            tmpValue.append("d");

        temp_data.SetString( tmpValue, index++ ); // EXTRA
        temp_data.SetString( "select", index++ ); // PRIVILEGES
        temp_data.SetString( "", index++ ); // COLUMN_COMMENT
        temp_data.SetString( "", index++ ); // GENERATION_EXPRESSION

        temp_data.Commit();
    }
}

/**
 * 将 table 的信息写入到 information_schema.tables 对应的 buffer 中
 */
static void table_insert_into_temp_data( TempTableData& temp_data, const TableEntrySPtr& table )
{
    if ( table->GetId() == -1 ) {
        table->SetId( ++CURRENT_TABLE_ID );
    }
    std::string empty_string;
    auto table_id = table->GetId();
    int index = 0;
    temp_data.SetInt( table_id, index++ );

    temp_data.SetString( "def", index++ ); // catelog
    temp_data.SetString( "information_schema", index++ ); // schema
    temp_data.SetString( table->GetName(), index++ ); // table name

    std::string type;

    if ( IsSysDb( table->GetName() ) ) { // table type
        type = "SYSTEM VIEW";
    }
    else
    {
        type = "BASE TABLE";
    }
    temp_data.SetString( type, index++ ); // table type

    temp_data.SetString( table->GetEngine(), index++ ); // engine

    int64_t version = 1;
    temp_data.SetInt( version, index++ ); // version

    std::string row_format = PRODUCT_NAME;
    temp_data.SetString( row_format, index++ ); // row_format

    temp_data.SetInt( 0, index++ ); // rows
    temp_data.SetInt( 0, index++ ); // avg_row_length
    temp_data.SetInt( 0, index++ ); // data_length
    temp_data.SetInt( 0, index++ ); // max_data_length
    temp_data.SetInt( 0, index++ ); // index_length
    temp_data.SetInt( 0, index++ ); // data_free
    temp_data.SetInt( 0, index++ ); // auto_increment

    time_t currentTime = time(NULL);
    std::string create_time;
    create_time.assign( ( const char* )&currentTime, sizeof( currentTime ) );
    temp_data.SetString( create_time, index++ ); // create_time
    temp_data.SetString( create_time, index++ ); // update_time
    temp_data.SetString( create_time, index++ ); // check_time

    temp_data.SetString( table->GetCollation(), index++ ); // table_collation

    temp_data.SetInt( 0, index++ ); // checksum
    temp_data.SetNull( index++ ); // CREATE_OPTIONS
    temp_data.SetString( "", index++ ); // table_comment

    temp_data.Commit();
}

bool Schema::Init()
{
    auto ret = InitInformationSchema();
    if ( !ret )
        return ret;

    InitDatabaseMysql();
    return InitDefaultUser();
}
void SetStdinEcho(bool enable = true)
{
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    if( !enable )
        tty.c_lflag &= ~ECHO;
    else
        tty.c_lflag |= ECHO;

    (void) tcsetattr(STDIN_FILENO, TCSANOW, &tty);
}

void CleanUpSchema()
{
    auto dir = Configuartion::GetInstance().GetDataDirectory( INFORMATION_SCHEMA );
    boost::filesystem::remove_all( dir );
    dir = Configuartion::GetInstance().GetDataDirectory( "mysql" );
    boost::filesystem::remove_all( dir );
    dir = Configuartion::GetInstance().GetXLogDataDirectory();
    boost::filesystem::remove_all( dir );
}
extern "C"
{
    static void initialize_signal_handler( int sig )
    {
        cout << "\nCaught signal " << sig << endl;
        CleanUpSchema();
    }
}
void setup_signals()
{
    struct sigaction sa;
    (void) sigemptyset(&sa.sa_mask);

    // Ignore SIGPIPE and SIGALRM
    sa.sa_flags= 0;
    sa.sa_handler= SIG_IGN;
    (void) sigaction(SIGPIPE, &sa, NULL);
    (void) sigaction(SIGALRM, &sa, NULL);

    sa.sa_handler= initialize_signal_handler;
    (void) sigaction(SIGUSR1, &sa, NULL);
    (void) sigaction(SIGQUIT, &sa, NULL);
    (void) sigaction(SIGHUP, &sa, NULL);
    (void) sigaction(SIGTERM, &sa, NULL);
    (void) sigaction(SIGTSTP, &sa, NULL);
    auto ret = sigaction(SIGINT, &sa, NULL);
    if ( ret )
    {
        cout << "sigaction failed:" << strerror( errno ) << endl;
    }

    sigset_t set;
    (void) sigemptyset(&set);
    (void) sigaddset(&set, SIGQUIT);
    (void) sigaddset(&set, SIGHUP);
    (void) sigaddset(&set, SIGTERM);
    (void) sigaddset(&set, SIGTSTP);
    (void) sigaddset(&set, SIGINT);
    ret = pthread_sigmask(SIG_UNBLOCK, &set, NULL);
    if ( ret )
    {
        cout << "pthread_sigmask failed:" << strerror( ret ) << endl;
    }
}

bool Schema::InitDefaultUser()
{
    setup_signals();

    const short passwordLen = 64;
    char passwordBuff[passwordLen] = {0};
    char passwordBuff2[passwordLen] = {0};
    bool ok = true;
    // string password;
    // bool ok = get_tty_password( "Enter password for user rateup: ", password );

    SetStdinEcho(false);
    do
    {
        cout << "Enter password for user rateup: ";
        cin.getline(passwordBuff, sizeof(passwordBuff));
        if (cin.eof())
        {
            ok = false;
            cout << "\nEOF" << endl;
            break;
        }
        else if (!cin.good())
        {
            ok = false;
            cout << "\nFaild to read password" << endl;
            break;
        }
        if (!strlen(passwordBuff))
            cout << "\nInvalid password, try again" << endl;
    } while (!strlen(passwordBuff));

    if (ok)
    {
        do
        {
            cout << "\nEnter password again: ";
            cin.getline(passwordBuff2, sizeof(passwordBuff2));
            if (cin.eof())
            {
                ok = false;
                cout << "\nEOF" << endl;
                break;
            }
            else if (!cin.good())
            {
                ok = false;
                cout << "\nFaild to read password" << endl;
                break;
            }
            if (!strlen(passwordBuff2))
                cout << "\nInvalid password, try again" << endl;
        } while (!strlen(passwordBuff2));
    }

    SetStdinEcho(true);

    if (ok)
    {
        for (int i = 0; i < passwordLen; i++)
        {
            if (passwordBuff[i] != passwordBuff2[i])
            {
                cout << "\nTwo passwords not match" << endl;
                ok = false;
                break;
            }
        }
    }

    if (!ok)
    {
        CleanUpSchema();
        return false;
    }

    // std::string sql = "create user rateup identified by '";
    // sql.append( password ).append( "'" );

    std::string sql = "insert into mysql.user( Host, User, authentication_string ) values ";
    char outbuf[MAX_FIELD_WIDTH] = {0};
    unsigned int buflen = MAX_FIELD_WIDTH;

    if (generate_native_password(outbuf, &buflen, passwordBuff, strlen(passwordBuff)))
        ARIES_EXCEPTION(ER_CANNOT_USER, "CREATE USER");

    sql.append("( '', 'rateup', ");
    sql.append("'").append(outbuf).append("' )");

    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL(sql, "");
    if (!sqlResult->IsSuccess())
    {
        const auto &errMsg = sqlResult->GetErrorMessage();
        ARIES_EXCEPTION_SIMPLE(sqlResult->GetErrorCode(), errMsg.data());
    }
    return true;
}

void Schema::InitDatabaseMysql()
{
    const auto database_name = "mysql";
    string sql = "create database if not exists ";
    sql.append( database_name ).append( ";" );
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    if ( !result->IsSuccess() )
    {
        ARIES_EXCEPTION( ER_CANT_CREATE_DB, "cannot create database mysql" );
    }

    result = SQLExecutor::GetInstance()->ExecuteSQL( SCHEMA_SQL_MYSQL_USER, database_name );
    if ( !result->IsSuccess() )
    {
        auto errMsg = result->GetErrorMessage();
        ARIES_EXCEPTION_SIMPLE( result->GetErrorCode(), errMsg.data() );
    }
}
bool Schema::InitInformationSchema()
{
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( "create database if not exists information_schema;", "" );
    if ( !result->IsSuccess() )
    {
        ARIES_EXCEPTION( ER_CANT_CREATE_DB, "cannot create information_schema" );
    }

    const auto database_name = "information_schema";
    auto database = GetDatabaseByName( database_name );

    SQLParserPortal parser;

    // tables
    auto statements = parser.ParseSQLString4Statements( SCHEMA_SQL_TABLES );

    ARIES_ASSERT( statements.size() == 1, "cannot parse create table sql" );
    ARIES_ASSERT( statements[ 0 ]->IsCommand(), "should be create table sql" );
    auto command = std::dynamic_pointer_cast< CommandStructure >( statements[ 0 ]->GetCommand() );
    auto tables_to_create = CommandToTableEntry( command.get(), database );
    database->PutTable( tables_to_create );
    tables_to_create->SetId( TBALE_ID_INFORMATION_SCHEMA_TABLES );
    TempTableData tables_temp_data = table_to_temp( tables_to_create );
    auto tableDataDir = Configuartion::GetInstance().GetDataDirectory( database_name, tables_to_create->GetName() );
    boost::filesystem::create_directory( tableDataDir );
    aries_engine::AriesInitialTable::InitFiles( database_name, tables_to_create->GetName() );

    // columns
    statements = parser.ParseSQLString4Statements( SCHEMA_SQL_COLUMNS );

    ARIES_ASSERT( statements.size() == 1, "cannot parse create columns sql" );
    ARIES_ASSERT( statements[ 0 ]->IsCommand(), "should be create columns sql" );
    command = std::dynamic_pointer_cast< CommandStructure >( statements[ 0 ]->GetCommand() );
    auto columns_to_create = CommandToTableEntry( command.get(), database );
    database->PutTable( columns_to_create );
    columns_to_create->SetId( TBALE_ID_INFORMATION_SCHEMA_COLUMNS );
    TempTableData columns_temp_data = table_to_temp( columns_to_create );
    tableDataDir = Configuartion::GetInstance().GetDataDirectory( database_name, columns_to_create->GetName() );
    boost::filesystem::create_directory( tableDataDir );
    aries_engine::AriesInitialTable::InitFiles( database_name, columns_to_create->GetName() );

    table_insert_into_temp_data( tables_temp_data, tables_to_create );
    column_insert_into_temp_data( columns_temp_data, tables_to_create );

    table_insert_into_temp_data( tables_temp_data, columns_to_create );
    column_insert_into_temp_data( columns_temp_data, columns_to_create );

    // schemata
    statements = parser.ParseSQLString4Statements( SCHEMA_SQL_SCHEMATA );

    ARIES_ASSERT( statements.size() == 1, "cannot parse create schemata sql" );
    ARIES_ASSERT( statements[ 0 ]->IsCommand(), "should be create columns sql" );
    command = std::dynamic_pointer_cast< CommandStructure >( statements[ 0 ]->GetCommand() );
    auto schemata_to_create = CommandToTableEntry( command.get(), database );
    database->PutTable( schemata_to_create );
    schemata_to_create->SetId( TBALE_ID_INFORMATION_SCHEMA_SCHEMATA );
    TempTableData schemata_temp_data = table_to_temp( schemata_to_create );
    tableDataDir = Configuartion::GetInstance().GetDataDirectory( database_name, schemata_to_create->GetName() );
    boost::filesystem::create_directory( tableDataDir );
    aries_engine::AriesInitialTable::InitFiles( database_name, schemata_to_create->GetName() );

    table_insert_into_temp_data( tables_temp_data, schemata_to_create );
    column_insert_into_temp_data( columns_temp_data, schemata_to_create );

    CURRENT_TABLE_ID = TBALE_ID_INFORMATION_SCHEMA_SCHEMATA;
    for ( size_t i = 3; i < INFORMATION_SCHEMA_TABLE_SQLS.size(); ++i )
    {
        const auto& sql = INFORMATION_SCHEMA_TABLE_SQLS[ i ];
        statements = parser.ParseSQLString4Statements( sql );

        ARIES_ASSERT( statements.size() == 1, "cannot parse create table sql" );
        ARIES_ASSERT( statements[ 0 ]->IsCommand(), "should be command sql" );
        command = std::dynamic_pointer_cast< CommandStructure >( statements[ 0 ]->GetCommand() );
        auto tableEntry = CommandToTableEntry( command.get(), database );
        tableEntry->SetId( ++CURRENT_TABLE_ID );
        database->PutTable( tableEntry );
        tableDataDir = Configuartion::GetInstance().GetDataDirectory( database_name, tableEntry->GetName() );
        boost::filesystem::create_directory( tableDataDir );
        aries_engine::AriesInitialTable::InitFiles( database_name, tableEntry->GetName() );

        table_insert_into_temp_data( tables_temp_data, tableEntry );
        column_insert_into_temp_data( columns_temp_data, tableEntry );
    }

    aries_engine::AriesInitialTable tables( database_name, tables_to_create->GetName() );
    for ( int i = 0; i < tables_temp_data.columns_count; i++ )
    {
        if ( ! WriteColumnDataIntoBlocks( tables, i,
                                          tables_temp_data.is_nullable[ i ],
                                          tables_temp_data.columns_size[ i ],
                                          tables_temp_data.buffers[ i ],
                                          tables_temp_data.current_row_count ) )
        {
            return false;
        }
    }
    auto metaFilePath = aries_engine::AriesInitialTable::GetMetaFilePath( database_name, tables_to_create->GetName() );
    aries_engine::AriesInitialTable::WriteMetaFile( metaFilePath, tables_temp_data.current_row_count );

    aries_engine::AriesInitialTable columns( database_name, columns_to_create->GetName() );
    for ( int i = 0; i < columns_temp_data.columns_count; i++ )
    {
        if ( ! WriteColumnDataIntoBlocks( columns, i,
                                          columns_temp_data.is_nullable[ i ],
                                          columns_temp_data.columns_size[ i ],
                                          columns_temp_data.buffers[ i ],
                                          columns_temp_data.current_row_count ) )
        {
            return false;
        }
    }
    metaFilePath = aries_engine::AriesInitialTable::GetMetaFilePath( database_name, columns_to_create->GetName() );
    aries_engine::AriesInitialTable::WriteMetaFile( metaFilePath, columns_temp_data.current_row_count );

    int index = 0;
    schemata_temp_data.SetString( "def", index++ );
    schemata_temp_data.SetString( database_name, index++ );
    schemata_temp_data.SetString( "utf8", index++ );
    schemata_temp_data.SetString( "utf8_general_ci", index++ );
    schemata_temp_data.SetNull( index++ );
    schemata_temp_data.Commit();

    aries_engine::AriesInitialTable schematas( database_name, schemata_to_create->GetName() );
    for ( size_t i = 0; i < schemata_to_create->GetColumnsCount(); i++ )
    {
        if ( ! WriteColumnDataIntoBlocks( schematas, i,
                                          schemata_temp_data.is_nullable[ i ],
                                          schemata_temp_data.columns_size[ i ],
                                          schemata_temp_data.buffers[ i ],
                                          schemata_temp_data.current_row_count ) )
        {
            return false;
        }

    }
    metaFilePath = aries_engine::AriesInitialTable::GetMetaFilePath( database_name, schemata_to_create->GetName() );
    aries_engine::AriesInitialTable::WriteMetaFile( metaFilePath, schemata_temp_data.current_row_count );

    tables_temp_data.Free();
    columns_temp_data.Free();

    databases[ INFORMATION_SCHEMA ] = database;

    return true;
}

void Schema::AddDatabase(shared_ptr<DatabaseEntry> database) {
    string name = database->GetName();
    aries_utils::to_lower(name);
    for (auto it=keys_to_remove.begin(); it!=keys_to_remove.end(); it++) {
        if (*it == name) {
            keys_to_remove.erase(it);
            break;
        }
    }
    databases[name] = database;
}

void Schema::AddDatabase(string name, shared_ptr<DatabaseEntry> database) {
    aries_utils::to_lower(name);
    for (auto it=keys_to_remove.begin(); it!=keys_to_remove.end(); it++) {
        if (*it == name) {
            keys_to_remove.erase(it);
            break;
        }
    }
    databases[name] = database;
}

std::string Schema::GetKeyForDatabase(std::string name) {
    auto key = KeyPrefix + "db_" + name;
    aries_utils::to_lower(key);
    return key;
}

std::string Schema::GetKeyForTable(std::string name, std::string database_name) {
    auto key = KeyPrefix + database_name + "_" + "tb_" + name;
    aries_utils::to_lower(key);
    return key;
}

std::string Schema::GetKeyForColumn(std::string name, std::string database_name, std::string table_name) {
    auto key = KeyPrefix + database_name + "_" + table_name + "_" + "col_" + name;
    aries_utils::to_lower(key);
    return key;
}

shared_ptr<DatabaseEntry> Schema::GetDatabaseByName(string name) {
    aries_utils::to_lower(name);
    auto it = databases.find(name);
    if (it != databases.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

TableId Schema::GetTableId( const string& dbName, const string& tableName )
{
    auto dbEntry = GetDatabaseByName( dbName );
    if ( !dbEntry )
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    auto tableEntry = dbEntry->GetTableByName( tableName );
    if( !tableEntry )
        ARIES_EXCEPTION( ER_NO_SUCH_TABLE, dbName.c_str(), tableName.c_str() );
    return tableEntry->GetId();
}

const std::map<std::string, std::shared_ptr<DatabaseEntry>>& Schema::GetDatabases() {
    return databases;
}

std::pair< DatabaseEntrySPtr, TableEntrySPtr > Schema::GetDatabaseAndTableById( TableId tableId )
{
    for ( const auto& db : databases )
    {
        for ( const auto& table : db.second->GetTables() )
        {
            if ( table.second->GetId() == tableId )
            {
                return std::make_pair( db.second, table.second );
            }
        }
    }

    return std::make_pair( nullptr, nullptr );
}
void Schema::RemoveDatabase(std::shared_ptr<DatabaseEntry> database) {
    auto dbName = database->GetName();
    std::string sql = "delete from `key_column_usage` where `TABLE_SCHEMA` = '" + dbName + "';";

    auto executor = SQLExecutor::GetInstance();
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `table_constraints` where `TABLE_SCHEMA` = '" + dbName + "'";
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `columns` where `TABLE_SCHEMA` = '" + dbName + "'";
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `tables` where `TABLE_SCHEMA` = '" + dbName + "'";
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `views` where `TABLE_SCHEMA` = '" + dbName + "'";
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `schemata` where `SCHEMA_NAME` = '" + dbName + "'";
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

void Schema::RemoveView( const std::string& dbName, const string& viewName )
{
    std::string sql = "delete from `views` where `TABLE_SCHEMA` = '" + dbName + "' and `TABLE_NAME` = '" + viewName + "'";
    CHECK_SQL_RESULT( SQLExecutor::GetInstance()->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

void Schema::RemoveTable( const std::string& dbName, const string& tableName )
{
    std::string sql = "delete from `key_column_usage` where `TABLE_SCHEMA` = '" + dbName + "'";
    sql += " and `TABLE_NAME` = '" +  tableName + "';";
    auto executor = SQLExecutor::GetInstance();
    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `table_constraints` where `TABLE_SCHEMA` = '" + dbName + "'";
    sql += " and `TABLE_NAME` = '" +  tableName + "';";

    CHECK_SQL_RESULT ( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `partitions` where `TABLE_SCHEMA` = '" + dbName + "'";
    sql += " and `TABLE_NAME` = '" +  tableName + "';";

    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `columns` where `TABLE_SCHEMA` = '" + dbName + "'";
    sql += " and `TABLE_NAME` = '" +  tableName + "';";

    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );

    sql = "delete from `tables` where `TABLE_SCHEMA` = '" + dbName + "'";
    sql += " and `TABLE_NAME` = '" +  tableName + "';";

    CHECK_SQL_RESULT( executor->ExecuteSQL( sql, INFORMATION_SCHEMA ) );
}

string BuildCreateTableString(const TableEntrySPtr& table)
{
    string tableSchema = "CREATE TABLE `" + table->GetName() + "` (\n";

    for (size_t j=1; j<=table->GetColumnsCount(); j++) {
        auto column = table->GetColumnById(j);
        tableSchema.append("  `").append(column->GetName()).append("`");
        tableSchema.append(" ").append(ColumnTypeString( column ));
        if ( !column->IsAllowNull() )
        {
            tableSchema.append( " not null" );
        }
        if ( column->IsPrimary() )
        {
            tableSchema.append( " primary key" );
        }
        if ( column->HasDefault() )
        {
            auto defaultVal = column->GetDefault();
            if ( !defaultVal )
                tableSchema.append( " default null" );
            else
                tableSchema.append( " default '" ).append( *defaultVal ).append("'");
        }
        if ( j < table->GetColumnsCount() )
        {
            tableSchema.append(",");
        }
        tableSchema.append("\n");

    }
    tableSchema.append(");");
    return tableSchema;
}

void Schema::Dump(const std::shared_ptr<DatabaseEntry>& database)
{
    auto keys = database->GetNameListOfTables();
#ifndef NDEBUG
    cout << "--\n";
    cout << "-- Current Database: `" << database->GetName() << "`\n";
    cout << "--\n\n";
#endif
    for (size_t i = 0; i < keys.size(); i++)
    {
        auto key = keys[i];
        auto table = database->GetTableByName(key);
        LOG(INFO) << "\t\t-------------------------------------------------------------------" << std::endl;
        LOG(INFO) << "\t\tHere start to dump table: " << table->GetName() << std::endl;
        LOG(INFO) << "\t\tThere are " << table->GetColumnsCount() << " column(s) in this table:" << std::endl;
#ifndef NDEBUG
        cout << "--\n";
        cout << "-- Table structure for table `" << table->GetName() << "`\n";
        cout << "--\n\n";
#endif
        string tableSchema = BuildCreateTableString( table );
#ifndef NDEBUG
        cout << tableSchema << "\n\n";
        fflush(stdout);
#endif
    }
}

void Schema::Dump( const string& dbName ) {
    LOG(INFO) << "Here start to dump schema: " << this << std::endl;

    if ( dbName.empty() )
    {
        LOG(INFO) << "here are " << databases.size() << " database(s):" << std::endl;
        for (auto it = databases.begin(); it != databases.end(); it ++) {
            auto database = it->second;
            Dump( database );
        }
    }
    else
    {
        auto database = GetDatabaseByName( dbName );
        if ( !database )
        {
            LOG(ERROR) << "Database " << dbName << " does not exist.";
            return;
        }
        Dump( database );
    }
}

} // namespace schema
} // namespace aries
