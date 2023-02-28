-- MySQL dump 10.13  Distrib 5.7.26, for Linux (x86_64)
--
-- Host: localhost    Database: information_schema
-- ------------------------------------------------------
-- Server version	5.7.26

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `CHARACTER_SETS`
--

DROP TABLE IF EXISTS `CHARACTER_SETS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `CHARACTER_SETS` (
  `CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
  `DEFAULT_COLLATE_NAME` varchar(32) NOT NULL DEFAULT '',
  `DESCRIPTION` varchar(60) NOT NULL DEFAULT '',
  `MAXLEN` bigint(3) NOT NULL DEFAULT '0'
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `CHARACTER_SETS`
--

-- LOCK TABLES `CHARACTER_SETS` WRITE;
/*!40000 ALTER TABLE `CHARACTER_SETS` DISABLE KEYS */;

INSERT INTO `CHARACTER_SETS` VALUES ('utf8','utf8_general_ci','UTF-8 Unicode',3);
/*!40000 ALTER TABLE `CHARACTER_SETS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `COLLATIONS`
--

DROP TABLE IF EXISTS `COLLATIONS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `COLLATIONS` (
  `COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
  `CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
  `ID` bigint(11) NOT NULL DEFAULT '0',
  `IS_DEFAULT` varchar(3) NOT NULL DEFAULT '',
  `IS_COMPILED` varchar(3) NOT NULL DEFAULT '',
  `SORTLEN` bigint(3) NOT NULL DEFAULT '0'
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `COLLATIONS`
--

-- LOCK TABLES `COLLATIONS` WRITE;
/*!40000 ALTER TABLE `COLLATIONS` DISABLE KEYS */;

INSERT INTO `COLLATIONS` VALUES ('utf8_general_ci','utf8',33,'Yes','Yes',1);
/*!40000 ALTER TABLE `COLLATIONS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `COLLATION_CHARACTER_SET_APPLICABILITY`
--

DROP TABLE IF EXISTS `COLLATION_CHARACTER_SET_APPLICABILITY`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `COLLATION_CHARACTER_SET_APPLICABILITY` (
  `COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
  `CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `COLLATION_CHARACTER_SET_APPLICABILITY`
--
-- LOCK TABLES `COLLATION_CHARACTER_SET_APPLICABILITY` WRITE;
/*!40000 ALTER TABLE `COLLATION_CHARACTER_SET_APPLICABILITY` DISABLE KEYS */;
INSERT INTO `COLLATION_CHARACTER_SET_APPLICABILITY` VALUES ('utf8_general_ci','utf8');
/*!40000 ALTER TABLE `COLLATION_CHARACTER_SET_APPLICABILITY` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `COLUMNS`
--

DROP TABLE IF EXISTS `COLUMNS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `COLUMNS` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `ORDINAL_POSITION` bigint(21) unsigned NOT NULL DEFAULT '0',
  `COLUMN_DEFAULT` longtext,
  `IS_NULLABLE` varchar(3) NOT NULL DEFAULT '',
  `DATA_TYPE` varchar(64) NOT NULL DEFAULT '',
  `CHARACTER_MAXIMUM_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `CHARACTER_OCTET_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `NUMERIC_PRECISION` bigint(21) unsigned DEFAULT NULL,
  `NUMERIC_SCALE` bigint(21) unsigned DEFAULT NULL,
  `DATETIME_PRECISION` bigint(21) unsigned DEFAULT NULL,
  `CHARACTER_SET_NAME` varchar(32) DEFAULT NULL,
  `COLLATION_NAME` varchar(32) DEFAULT NULL,
  `COLUMN_TYPE` longtext NOT NULL,
  `COLUMN_KEY` varchar(3) NOT NULL DEFAULT '',
  `EXTRA` varchar(30) NOT NULL DEFAULT '',
  `PRIVILEGES` varchar(80) NOT NULL DEFAULT '',
  `COLUMN_COMMENT` varchar(1024) NOT NULL DEFAULT '',
  `GENERATION_EXPRESSION` longtext NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `COLUMNS`
--

-- LOCK TABLES `COLUMNS` WRITE;
/*!40000 ALTER TABLE `COLUMNS` DISABLE KEYS */;
/*!40000 ALTER TABLE `COLUMNS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `COLUMN_PRIVILEGES`
--

DROP TABLE IF EXISTS `COLUMN_PRIVILEGES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `COLUMN_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `COLUMN_PRIVILEGES`
--

-- LOCK TABLES `COLUMN_PRIVILEGES` WRITE;
/*!40000 ALTER TABLE `COLUMN_PRIVILEGES` DISABLE KEYS */;
/*!40000 ALTER TABLE `COLUMN_PRIVILEGES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `ENGINES`
--

DROP TABLE IF EXISTS `ENGINES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `ENGINES` (
  `ENGINE` varchar(64) NOT NULL DEFAULT '',
  `SUPPORT` varchar(8) NOT NULL DEFAULT '',
  `COMMENT` varchar(80) NOT NULL DEFAULT '',
  `TRANSACTIONS` varchar(3) DEFAULT NULL,
  `XA` varchar(3) DEFAULT NULL,
  `SAVEPOINTS` varchar(3) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ENGINES`
--

-- LOCK TABLES `ENGINES` WRITE;
/*!40000 ALTER TABLE `ENGINES` DISABLE KEYS */;
INSERT INTO `ENGINES` VALUES ('Aries','DEFAULT','Column store','YES','YES','YES');
/*!40000 ALTER TABLE `ENGINES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `EVENTS`
--

DROP TABLE IF EXISTS `EVENTS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
  `SQL_MODE` varchar(8192) NOT NULL DEFAULT '',
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `EVENTS`
--

-- LOCK TABLES `EVENTS` WRITE;
/*!40000 ALTER TABLE `EVENTS` DISABLE KEYS */;
/*!40000 ALTER TABLE `EVENTS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `FILES`
--

DROP TABLE IF EXISTS `FILES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `FILES` (
  `FILE_ID` bigint(4) NOT NULL DEFAULT '0',
  `FILE_NAME` varchar(4000) DEFAULT NULL,
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
  `INITIAL_SIZE` bigint(21) unsigned DEFAULT NULL,
  `MAXIMUM_SIZE` bigint(21) unsigned DEFAULT NULL,
  `AUTOEXTEND_SIZE` bigint(21) unsigned DEFAULT NULL,
  `CREATION_TIME` datetime DEFAULT NULL,
  `LAST_UPDATE_TIME` datetime DEFAULT NULL,
  `LAST_ACCESS_TIME` datetime DEFAULT NULL,
  `RECOVER_TIME` bigint(4) DEFAULT NULL,
  `TRANSACTION_COUNTER` bigint(4) DEFAULT NULL,
  `VERSION` bigint(21) unsigned DEFAULT NULL,
  `ROW_FORMAT` varchar(10) DEFAULT NULL,
  `TABLE_ROWS` bigint(21) unsigned DEFAULT NULL,
  `AVG_ROW_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `DATA_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `MAX_DATA_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `INDEX_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `DATA_FREE` bigint(21) unsigned DEFAULT NULL,
  `CREATE_TIME` datetime DEFAULT NULL,
  `UPDATE_TIME` datetime DEFAULT NULL,
  `CHECK_TIME` datetime DEFAULT NULL,
  `CHECKSUM` bigint(21) unsigned DEFAULT NULL,
  `STATUS` varchar(20) NOT NULL DEFAULT '',
  `EXTRA` varchar(255) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `FILES`
--

-- LOCK TABLES `FILES` WRITE;
/*!40000 ALTER TABLE `FILES` DISABLE KEYS */;
/*!40000 ALTER TABLE `FILES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `GLOBAL_STATUS`
--

DROP TABLE IF EXISTS `GLOBAL_STATUS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `GLOBAL_STATUS` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `GLOBAL_STATUS`
--

-- LOCK TABLES `GLOBAL_STATUS` WRITE;
/*!40000 ALTER TABLE `GLOBAL_STATUS` DISABLE KEYS */;
/*!40000 ALTER TABLE `GLOBAL_STATUS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `GLOBAL_VARIABLES`
--

DROP TABLE IF EXISTS `GLOBAL_VARIABLES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `GLOBAL_VARIABLES` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `GLOBAL_VARIABLES`
--

-- LOCK TABLES `GLOBAL_VARIABLES` WRITE;
/*!40000 ALTER TABLE `GLOBAL_VARIABLES` DISABLE KEYS */;
/*!40000 ALTER TABLE `GLOBAL_VARIABLES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `KEY_COLUMN_USAGE`
--

DROP TABLE IF EXISTS `KEY_COLUMN_USAGE`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `KEY_COLUMN_USAGE` (
  `CONSTRAINT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `CONSTRAINT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_NAME` varchar(64) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `COLUMN_NAME` varchar(64) NOT NULL DEFAULT '',
  `ORDINAL_POSITION` bigint(10) NOT NULL DEFAULT '0',
  `POSITION_IN_UNIQUE_CONSTRAINT` bigint(10) DEFAULT NULL,
  `REFERENCED_TABLE_SCHEMA` varchar(64) DEFAULT NULL,
  `REFERENCED_TABLE_NAME` varchar(64) DEFAULT NULL,
  `REFERENCED_COLUMN_NAME` varchar(64) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `KEY_COLUMN_USAGE`
--

-- LOCK TABLES `KEY_COLUMN_USAGE` WRITE;
/*!40000 ALTER TABLE `KEY_COLUMN_USAGE` DISABLE KEYS */;
/*!40000 ALTER TABLE `KEY_COLUMN_USAGE` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `OPTIMIZER_TRACE`
--

DROP TABLE IF EXISTS `OPTIMIZER_TRACE`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `OPTIMIZER_TRACE` (
  `QUERY` longtext NOT NULL,
  `TRACE` longtext NOT NULL,
  `MISSING_BYTES_BEYOND_MAX_MEM_SIZE` int(20) NOT NULL DEFAULT '0',
  `INSUFFICIENT_PRIVILEGES` tinyint(1) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `OPTIMIZER_TRACE`
--

-- LOCK TABLES `OPTIMIZER_TRACE` WRITE;
/*!40000 ALTER TABLE `OPTIMIZER_TRACE` DISABLE KEYS */;
/*!40000 ALTER TABLE `OPTIMIZER_TRACE` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `PARAMETERS`
--

DROP TABLE IF EXISTS `PARAMETERS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
  `NUMERIC_PRECISION` bigint(21) unsigned DEFAULT NULL,
  `NUMERIC_SCALE` int(21) DEFAULT NULL,
  `DATETIME_PRECISION` bigint(21) unsigned DEFAULT NULL,
  `CHARACTER_SET_NAME` varchar(64) DEFAULT NULL,
  `COLLATION_NAME` varchar(64) DEFAULT NULL,
  `DTD_IDENTIFIER` longtext NOT NULL,
  `ROUTINE_TYPE` varchar(9) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `PARAMETERS`
--

-- LOCK TABLES `PARAMETERS` WRITE;
/*!40000 ALTER TABLE `PARAMETERS` DISABLE KEYS */;
/*!40000 ALTER TABLE `PARAMETERS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `PARTITIONS`
--

DROP TABLE IF EXISTS `PARTITIONS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `PARTITIONS` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `PARTITION_NAME` varchar(64) DEFAULT NULL,
  `SUBPARTITION_NAME` varchar(64) DEFAULT NULL,
  `PARTITION_ORDINAL_POSITION` bigint(21) unsigned DEFAULT NULL,
  `SUBPARTITION_ORDINAL_POSITION` bigint(21) unsigned DEFAULT NULL,
  `PARTITION_METHOD` varchar(18) DEFAULT NULL,
  `SUBPARTITION_METHOD` varchar(12) DEFAULT NULL,
  `PARTITION_EXPRESSION` longtext,
  `SUBPARTITION_EXPRESSION` longtext,
  `PARTITION_DESCRIPTION` longtext,
  `TABLE_ROWS` bigint(21) unsigned NOT NULL DEFAULT '0',
  `AVG_ROW_LENGTH` bigint(21) unsigned NOT NULL DEFAULT '0',
  `DATA_LENGTH` bigint(21) unsigned NOT NULL DEFAULT '0',
  `MAX_DATA_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `INDEX_LENGTH` bigint(21) unsigned NOT NULL DEFAULT '0',
  `DATA_FREE` bigint(21) unsigned NOT NULL DEFAULT '0',
  `CREATE_TIME` datetime DEFAULT NULL,
  `UPDATE_TIME` datetime DEFAULT NULL,
  `CHECK_TIME` datetime DEFAULT NULL,
  `CHECKSUM` bigint(21) unsigned DEFAULT NULL,
  `PARTITION_COMMENT` varchar(80) NOT NULL DEFAULT '',
  `NODEGROUP` varchar(12) NOT NULL DEFAULT '',
  `TABLESPACE_NAME` varchar(64) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `PARTITIONS`
--

-- LOCK TABLES `PARTITIONS` WRITE;
/*!40000 ALTER TABLE `PARTITIONS` DISABLE KEYS */;
/*!40000 ALTER TABLE `PARTITIONS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `PLUGINS`
--

DROP TABLE IF EXISTS `PLUGINS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `PLUGINS`
--

-- LOCK TABLES `PLUGINS` WRITE;
/*!40000 ALTER TABLE `PLUGINS` DISABLE KEYS */;
INSERT INTO `PLUGINS` VALUES ('mysql_native_password','1.1','ACTIVE','AUTHENTICATION','1.1',NULL,NULL,'R.J.Silk, Sergei Golubchik','Native MySQL authentication','GPL','FORCE');
/*!40000 ALTER TABLE `PLUGINS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `PROCESSLIST`
--

DROP TABLE IF EXISTS `PROCESSLIST`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `PROCESSLIST` (
  `ID` bigint(21) unsigned NOT NULL DEFAULT '0',
  `USER` varchar(32) NOT NULL DEFAULT '',
  `HOST` varchar(64) NOT NULL DEFAULT '',
  `DB` varchar(64) DEFAULT NULL,
  `COMMAND` varchar(16) NOT NULL DEFAULT '',
  `TIME` int(7) NOT NULL DEFAULT '0',
  `STATE` varchar(64) DEFAULT NULL,
  `INFO` longtext
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `PROCESSLIST`
--

-- LOCK TABLES `PROCESSLIST` WRITE;
/*!40000 ALTER TABLE `PROCESSLIST` DISABLE KEYS */;
/*!40000 ALTER TABLE `PROCESSLIST` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `PROFILING`
--

DROP TABLE IF EXISTS `PROFILING`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `PROFILING`
--

-- LOCK TABLES `PROFILING` WRITE;
/*!40000 ALTER TABLE `PROFILING` DISABLE KEYS */;
/*!40000 ALTER TABLE `PROFILING` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `REFERENTIAL_CONSTRAINTS`
--

DROP TABLE IF EXISTS `REFERENTIAL_CONSTRAINTS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `REFERENTIAL_CONSTRAINTS`
--

-- LOCK TABLES `REFERENTIAL_CONSTRAINTS` WRITE;
/*!40000 ALTER TABLE `REFERENTIAL_CONSTRAINTS` DISABLE KEYS */;
/*!40000 ALTER TABLE `REFERENTIAL_CONSTRAINTS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `ROUTINES`
--

DROP TABLE IF EXISTS `ROUTINES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `ROUTINES` (
  `SPECIFIC_NAME` varchar(64) NOT NULL DEFAULT '',
  `ROUTINE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `ROUTINE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `ROUTINE_NAME` varchar(64) NOT NULL DEFAULT '',
  `ROUTINE_TYPE` varchar(9) NOT NULL DEFAULT '',
  `DATA_TYPE` varchar(64) NOT NULL DEFAULT '',
  `CHARACTER_MAXIMUM_LENGTH` int(21) DEFAULT NULL,
  `CHARACTER_OCTET_LENGTH` int(21) DEFAULT NULL,
  `NUMERIC_PRECISION` bigint(21) unsigned DEFAULT NULL,
  `NUMERIC_SCALE` int(21) DEFAULT NULL,
  `DATETIME_PRECISION` bigint(21) unsigned DEFAULT NULL,
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
  `SQL_MODE` varchar(8192) NOT NULL DEFAULT '',
  `ROUTINE_COMMENT` longtext NOT NULL,
  `DEFINER` varchar(93) NOT NULL DEFAULT '',
  `CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
  `COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT '',
  `DATABASE_COLLATION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ROUTINES`
--

-- LOCK TABLES `ROUTINES` WRITE;
/*!40000 ALTER TABLE `ROUTINES` DISABLE KEYS */;
/*!40000 ALTER TABLE `ROUTINES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `SCHEMATA`
--

DROP TABLE IF EXISTS `SCHEMATA`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `SCHEMATA` (
  `CATALOG_NAME` varchar(512) NOT NULL DEFAULT '',
  `SCHEMA_NAME` varchar(64) NOT NULL DEFAULT '',
  `DEFAULT_CHARACTER_SET_NAME` varchar(32) NOT NULL DEFAULT '',
  `DEFAULT_COLLATION_NAME` varchar(32) NOT NULL DEFAULT '',
  `SQL_PATH` varchar(512) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `SCHEMATA`
--

-- LOCK TABLES `SCHEMATA` WRITE;
/*!40000 ALTER TABLE `SCHEMATA` DISABLE KEYS */;
INSERT INTO `SCHEMATA` VALUES ('def','information_schema','utf8','utf8_general_ci',NULL),('def','mysql','latin1','latin1_swedish_ci',NULL),('def','performance_schema','utf8','utf8_general_ci',NULL),('def','sys','utf8','utf8_general_ci',NULL);
/*!40000 ALTER TABLE `SCHEMATA` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `SCHEMA_PRIVILEGES`
--

DROP TABLE IF EXISTS `SCHEMA_PRIVILEGES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `SCHEMA_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `SCHEMA_PRIVILEGES`
--

-- LOCK TABLES `SCHEMA_PRIVILEGES` WRITE;
/*!40000 ALTER TABLE `SCHEMA_PRIVILEGES` DISABLE KEYS */;
/*!40000 ALTER TABLE `SCHEMA_PRIVILEGES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `SESSION_STATUS`
--

DROP TABLE IF EXISTS `SESSION_STATUS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `SESSION_STATUS` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `SESSION_STATUS`
--

-- LOCK TABLES `SESSION_STATUS` WRITE;
/*!40000 ALTER TABLE `SESSION_STATUS` DISABLE KEYS */;
/*!40000 ALTER TABLE `SESSION_STATUS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `SESSION_VARIABLES`
--

DROP TABLE IF EXISTS `SESSION_VARIABLES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `SESSION_VARIABLES` (
  `VARIABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VARIABLE_VALUE` varchar(1024) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `SESSION_VARIABLES`
--

-- LOCK TABLES `SESSION_VARIABLES` WRITE;
/*!40000 ALTER TABLE `SESSION_VARIABLES` DISABLE KEYS */;
/*!40000 ALTER TABLE `SESSION_VARIABLES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `STATISTICS`
--

DROP TABLE IF EXISTS `STATISTICS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `STATISTICS`
--

-- LOCK TABLES `STATISTICS` WRITE;
/*!40000 ALTER TABLE `STATISTICS` DISABLE KEYS */;
/*!40000 ALTER TABLE `STATISTICS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `TABLES`
--

DROP TABLE IF EXISTS `TABLES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `TABLES` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `TABLE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `ENGINE` varchar(64) DEFAULT NULL,
  `VERSION` bigint(21) unsigned DEFAULT NULL,
  `ROW_FORMAT` varchar(10) DEFAULT NULL,
  `TABLE_ROWS` bigint(21) unsigned DEFAULT NULL,
  `AVG_ROW_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `DATA_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `MAX_DATA_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `INDEX_LENGTH` bigint(21) unsigned DEFAULT NULL,
  `DATA_FREE` bigint(21) unsigned DEFAULT NULL,
  `AUTO_INCREMENT` bigint(21) unsigned DEFAULT NULL,
  `CREATE_TIME` datetime DEFAULT NULL,
  `UPDATE_TIME` datetime DEFAULT NULL,
  `CHECK_TIME` datetime DEFAULT NULL,
  `TABLE_COLLATION` varchar(32) DEFAULT NULL,
  `CHECKSUM` bigint(21) unsigned DEFAULT NULL,
  `CREATE_OPTIONS` varchar(255) DEFAULT NULL,
  `TABLE_COMMENT` varchar(2048) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TABLES`
--

-- LOCK TABLES `TABLES` WRITE;
/*!40000 ALTER TABLE `TABLES` DISABLE KEYS */;
/*!40000 ALTER TABLE `TABLES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `TABLESPACES`
--

DROP TABLE IF EXISTS `TABLESPACES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `TABLESPACES` (
  `TABLESPACE_NAME` varchar(64) NOT NULL DEFAULT '',
  `ENGINE` varchar(64) NOT NULL DEFAULT '',
  `TABLESPACE_TYPE` varchar(64) DEFAULT NULL,
  `LOGFILE_GROUP_NAME` varchar(64) DEFAULT NULL,
  `EXTENT_SIZE` bigint(21) unsigned DEFAULT NULL,
  `AUTOEXTEND_SIZE` bigint(21) unsigned DEFAULT NULL,
  `MAXIMUM_SIZE` bigint(21) unsigned DEFAULT NULL,
  `NODEGROUP_ID` bigint(21) unsigned DEFAULT NULL,
  `TABLESPACE_COMMENT` varchar(2048) DEFAULT NULL
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TABLESPACES`
--

-- LOCK TABLES `TABLESPACES` WRITE;
/*!40000 ALTER TABLE `TABLESPACES` DISABLE KEYS */;
/*!40000 ALTER TABLE `TABLESPACES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `TABLE_CONSTRAINTS`
--

DROP TABLE IF EXISTS `TABLE_CONSTRAINTS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `TABLE_CONSTRAINTS` (
  `CONSTRAINT_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `CONSTRAINT_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_NAME` varchar(64) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `CONSTRAINT_TYPE` varchar(64) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TABLE_CONSTRAINTS`
--

-- LOCK TABLES `TABLE_CONSTRAINTS` WRITE;
/*!40000 ALTER TABLE `TABLE_CONSTRAINTS` DISABLE KEYS */;
INSERT INTO `TABLE_CONSTRAINTS` VALUES ('def','mysql','PRIMARY','mysql','columns_priv','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','db','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','engine_cost','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','event','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','func','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','gtid_executed','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','help_category','PRIMARY KEY'),('def','mysql','name','mysql','help_category','UNIQUE'),('def','mysql','PRIMARY','mysql','help_keyword','PRIMARY KEY'),('def','mysql','name','mysql','help_keyword','UNIQUE'),('def','mysql','PRIMARY','mysql','help_relation','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','help_topic','PRIMARY KEY'),('def','mysql','name','mysql','help_topic','UNIQUE'),('def','mysql','PRIMARY','mysql','innodb_index_stats','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','innodb_table_stats','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','ndb_binlog_index','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','plugin','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','proc','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','procs_priv','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','proxies_priv','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','server_cost','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','servers','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','slave_master_info','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','slave_relay_log_info','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','slave_worker_info','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','tables_priv','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','time_zone','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','time_zone_leap_second','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','time_zone_name','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','time_zone_transition','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','time_zone_transition_type','PRIMARY KEY'),('def','mysql','PRIMARY','mysql','user','PRIMARY KEY'),('def','sys','PRIMARY','sys','sys_config','PRIMARY KEY');
/*!40000 ALTER TABLE `TABLE_CONSTRAINTS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `TABLE_PRIVILEGES`
--

DROP TABLE IF EXISTS `TABLE_PRIVILEGES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `TABLE_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TABLE_PRIVILEGES`
--

-- LOCK TABLES `TABLE_PRIVILEGES` WRITE;
/*!40000 ALTER TABLE `TABLE_PRIVILEGES` DISABLE KEYS */;
/*!40000 ALTER TABLE `TABLE_PRIVILEGES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `TRIGGERS`
--

DROP TABLE IF EXISTS `TRIGGERS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
  `SQL_MODE` varchar(8192) NOT NULL DEFAULT '',
  `DEFINER` varchar(93) NOT NULL DEFAULT '',
  `CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
  `COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT '',
  `DATABASE_COLLATION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TRIGGERS`
--

-- LOCK TABLES `TRIGGERS` WRITE;
/*!40000 ALTER TABLE `TRIGGERS` DISABLE KEYS */;
/*!40000 ALTER TABLE `TRIGGERS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `USER_PRIVILEGES`
--

DROP TABLE IF EXISTS `USER_PRIVILEGES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `USER_PRIVILEGES` (
  `GRANTEE` varchar(81) NOT NULL DEFAULT '',
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `PRIVILEGE_TYPE` varchar(64) NOT NULL DEFAULT '',
  `IS_GRANTABLE` varchar(3) NOT NULL DEFAULT ''
) ENGINE=MEMORY DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `USER_PRIVILEGES`
--

-- LOCK TABLES `USER_PRIVILEGES` WRITE;
/*!40000 ALTER TABLE `USER_PRIVILEGES` DISABLE KEYS */;
/*!40000 ALTER TABLE `USER_PRIVILEGES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `VIEWS`
--

DROP TABLE IF EXISTS `VIEWS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TEMPORARY TABLE `VIEWS` (
  `TABLE_CATALOG` varchar(512) NOT NULL DEFAULT '',
  `TABLE_SCHEMA` varchar(64) NOT NULL DEFAULT '',
  `TABLE_NAME` varchar(64) NOT NULL DEFAULT '',
  `VIEW_DEFINITION` longtext NOT NULL,
  `CHECK_OPTION` varchar(8) NOT NULL DEFAULT '',
  `IS_UPDATABLE` varchar(3) NOT NULL DEFAULT '',
  `DEFINER` varchar(93) NOT NULL DEFAULT '',
  `SECURITY_TYPE` varchar(7) NOT NULL DEFAULT '',
  `CHARACTER_SET_CLIENT` varchar(32) NOT NULL DEFAULT '',
  `COLLATION_CONNECTION` varchar(32) NOT NULL DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8 stored as kv;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `VIEWS`
--

-- LOCK TABLES `VIEWS` WRITE;
/*!40000 ALTER TABLE `VIEWS` DISABLE KEYS */;
/*!40000 ALTER TABLE `VIEWS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_LOCKS`
--

DROP TABLE IF EXISTS `INNODB_LOCKS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_LOCKS`
--

-- LOCK TABLES `INNODB_LOCKS` WRITE;
/*!40000 ALTER TABLE `INNODB_LOCKS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_LOCKS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_TRX`
--

DROP TABLE IF EXISTS `INNODB_TRX`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_TRX`
--

-- LOCK TABLES `INNODB_TRX` WRITE;
/*!40000 ALTER TABLE `INNODB_TRX` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_TRX` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_DATAFILES`
--

DROP TABLE IF EXISTS `INNODB_SYS_DATAFILES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_DATAFILES`
--

-- LOCK TABLES `INNODB_SYS_DATAFILES` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_DATAFILES` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_DATAFILES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_FT_CONFIG`
--

DROP TABLE IF EXISTS `INNODB_FT_CONFIG`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_FT_CONFIG`
--

-- LOCK TABLES `INNODB_FT_CONFIG` WRITE;
/*!40000 ALTER TABLE `INNODB_FT_CONFIG` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_FT_CONFIG` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_VIRTUAL`
--

DROP TABLE IF EXISTS `INNODB_SYS_VIRTUAL`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_VIRTUAL`
--

-- LOCK TABLES `INNODB_SYS_VIRTUAL` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_VIRTUAL` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_VIRTUAL` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_CMP`
--

DROP TABLE IF EXISTS `INNODB_CMP`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_CMP`
--

-- LOCK TABLES `INNODB_CMP` WRITE;
/*!40000 ALTER TABLE `INNODB_CMP` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_CMP` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_FT_BEING_DELETED`
--

DROP TABLE IF EXISTS `INNODB_FT_BEING_DELETED`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_FT_BEING_DELETED`
--

-- LOCK TABLES `INNODB_FT_BEING_DELETED` WRITE;
/*!40000 ALTER TABLE `INNODB_FT_BEING_DELETED` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_FT_BEING_DELETED` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_CMP_RESET`
--

DROP TABLE IF EXISTS `INNODB_CMP_RESET`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_CMP_RESET`
--

-- LOCK TABLES `INNODB_CMP_RESET` WRITE;
/*!40000 ALTER TABLE `INNODB_CMP_RESET` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_CMP_RESET` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_CMP_PER_INDEX`
--

DROP TABLE IF EXISTS `INNODB_CMP_PER_INDEX`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_CMP_PER_INDEX`
--

-- LOCK TABLES `INNODB_CMP_PER_INDEX` WRITE;
/*!40000 ALTER TABLE `INNODB_CMP_PER_INDEX` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_CMP_PER_INDEX` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_CMPMEM_RESET`
--

DROP TABLE IF EXISTS `INNODB_CMPMEM_RESET`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_CMPMEM_RESET`
--

-- LOCK TABLES `INNODB_CMPMEM_RESET` WRITE;
/*!40000 ALTER TABLE `INNODB_CMPMEM_RESET` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_CMPMEM_RESET` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_FT_DELETED`
--

DROP TABLE IF EXISTS `INNODB_FT_DELETED`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_FT_DELETED`
--

-- LOCK TABLES `INNODB_FT_DELETED` WRITE;
/*!40000 ALTER TABLE `INNODB_FT_DELETED` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_FT_DELETED` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_BUFFER_PAGE_LRU`
--

DROP TABLE IF EXISTS `INNODB_BUFFER_PAGE_LRU`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_BUFFER_PAGE_LRU`
--

-- LOCK TABLES `INNODB_BUFFER_PAGE_LRU` WRITE;
/*!40000 ALTER TABLE `INNODB_BUFFER_PAGE_LRU` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_BUFFER_PAGE_LRU` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_LOCK_WAITS`
--

DROP TABLE IF EXISTS `INNODB_LOCK_WAITS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_LOCK_WAITS`
--

-- LOCK TABLES `INNODB_LOCK_WAITS` WRITE;
/*!40000 ALTER TABLE `INNODB_LOCK_WAITS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_LOCK_WAITS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_TEMP_TABLE_INFO`
--

DROP TABLE IF EXISTS `INNODB_TEMP_TABLE_INFO`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_TEMP_TABLE_INFO`
--

-- LOCK TABLES `INNODB_TEMP_TABLE_INFO` WRITE;
/*!40000 ALTER TABLE `INNODB_TEMP_TABLE_INFO` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_TEMP_TABLE_INFO` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_INDEXES`
--

DROP TABLE IF EXISTS `INNODB_SYS_INDEXES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_INDEXES`
--

-- LOCK TABLES `INNODB_SYS_INDEXES` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_INDEXES` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_INDEXES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_TABLES`
--

DROP TABLE IF EXISTS `INNODB_SYS_TABLES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_TABLES`
--

-- LOCK TABLES `INNODB_SYS_TABLES` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_TABLES` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_TABLES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_FIELDS`
--

DROP TABLE IF EXISTS `INNODB_SYS_FIELDS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_FIELDS`
--

-- LOCK TABLES `INNODB_SYS_FIELDS` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_FIELDS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_FIELDS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_CMP_PER_INDEX_RESET`
--

DROP TABLE IF EXISTS `INNODB_CMP_PER_INDEX_RESET`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_CMP_PER_INDEX_RESET`
--

-- LOCK TABLES `INNODB_CMP_PER_INDEX_RESET` WRITE;
/*!40000 ALTER TABLE `INNODB_CMP_PER_INDEX_RESET` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_CMP_PER_INDEX_RESET` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_BUFFER_PAGE`
--

DROP TABLE IF EXISTS `INNODB_BUFFER_PAGE`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_BUFFER_PAGE`
--

-- LOCK TABLES `INNODB_BUFFER_PAGE` WRITE;
/*!40000 ALTER TABLE `INNODB_BUFFER_PAGE` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_BUFFER_PAGE` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_FT_DEFAULT_STOPWORD`
--

DROP TABLE IF EXISTS `INNODB_FT_DEFAULT_STOPWORD`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_FT_DEFAULT_STOPWORD`
--

-- LOCK TABLES `INNODB_FT_DEFAULT_STOPWORD` WRITE;
/*!40000 ALTER TABLE `INNODB_FT_DEFAULT_STOPWORD` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_FT_DEFAULT_STOPWORD` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_FT_INDEX_TABLE`
--

DROP TABLE IF EXISTS `INNODB_FT_INDEX_TABLE`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_FT_INDEX_TABLE`
--

-- LOCK TABLES `INNODB_FT_INDEX_TABLE` WRITE;
/*!40000 ALTER TABLE `INNODB_FT_INDEX_TABLE` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_FT_INDEX_TABLE` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_FT_INDEX_CACHE`
--

DROP TABLE IF EXISTS `INNODB_FT_INDEX_CACHE`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_FT_INDEX_CACHE`
--

-- LOCK TABLES `INNODB_FT_INDEX_CACHE` WRITE;
/*!40000 ALTER TABLE `INNODB_FT_INDEX_CACHE` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_FT_INDEX_CACHE` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_TABLESPACES`
--

DROP TABLE IF EXISTS `INNODB_SYS_TABLESPACES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_TABLESPACES`
--

-- LOCK TABLES `INNODB_SYS_TABLESPACES` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_TABLESPACES` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_TABLESPACES` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_METRICS`
--

DROP TABLE IF EXISTS `INNODB_METRICS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_METRICS`
--

-- LOCK TABLES `INNODB_METRICS` WRITE;
/*!40000 ALTER TABLE `INNODB_METRICS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_METRICS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_FOREIGN_COLS`
--

DROP TABLE IF EXISTS `INNODB_SYS_FOREIGN_COLS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_FOREIGN_COLS`
--

-- LOCK TABLES `INNODB_SYS_FOREIGN_COLS` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_FOREIGN_COLS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_FOREIGN_COLS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_CMPMEM`
--

DROP TABLE IF EXISTS `INNODB_CMPMEM`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_CMPMEM`
--

-- LOCK TABLES `INNODB_CMPMEM` WRITE;
/*!40000 ALTER TABLE `INNODB_CMPMEM` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_CMPMEM` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_BUFFER_POOL_STATS`
--

DROP TABLE IF EXISTS `INNODB_BUFFER_POOL_STATS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_BUFFER_POOL_STATS`
--

-- LOCK TABLES `INNODB_BUFFER_POOL_STATS` WRITE;
/*!40000 ALTER TABLE `INNODB_BUFFER_POOL_STATS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_BUFFER_POOL_STATS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_COLUMNS`
--

DROP TABLE IF EXISTS `INNODB_SYS_COLUMNS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_COLUMNS`
--

-- LOCK TABLES `INNODB_SYS_COLUMNS` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_COLUMNS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_COLUMNS` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_FOREIGN`
--

DROP TABLE IF EXISTS `INNODB_SYS_FOREIGN`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_FOREIGN`
--

-- LOCK TABLES `INNODB_SYS_FOREIGN` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_FOREIGN` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_FOREIGN` ENABLE KEYS */;
-- UNLOCK TABLES;

--
-- Table structure for table `INNODB_SYS_TABLESTATS`
--

DROP TABLE IF EXISTS `INNODB_SYS_TABLESTATS`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `INNODB_SYS_TABLESTATS`
--

-- LOCK TABLES `INNODB_SYS_TABLESTATS` WRITE;
/*!40000 ALTER TABLE `INNODB_SYS_TABLESTATS` DISABLE KEYS */;
/*!40000 ALTER TABLE `INNODB_SYS_TABLESTATS` ENABLE KEYS */;
-- UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-09-25 11:26:52
