//
// Created by 胡胜刚 on 2019-06-12.
//

#include <vector>
#include <memory>

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "frontend/SQLExecutor.h"

using namespace aries;
extern std::string DATA_DIR;

static std::vector<std::shared_ptr<AbstractMemTable>> TestSidedoor_DBCommand(std::string arg_sql_content, const std::string& dbName)
{
    LOG(INFO) << "-----------------------------------------------" << "\n" << arg_sql_content;

    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQL(arg_sql_content, dbName);

    return results->GetResults();
}

static std::vector<std::shared_ptr<AbstractMemTable>> TestSidedoor_Normal(std::string arg_db_name, std::string arg_sql_content)
{
    LOG(INFO) << "-----------------------------------------------" << "\n" << arg_db_name << "\n" << arg_sql_content;

    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQL(arg_sql_content, arg_db_name);

    return results->GetResults();
}

void InitDatabase(std::string db_name) {
    TestSidedoor_DBCommand("create database " + db_name + ";", db_name);

    TestSidedoor_Normal(db_name, R"(CREATE TABLE PART (
    P_PARTKEY       INTEGER not null primary key,
    P_NAME          VARCHAR(55) NOT NULL,
    P_MFGR          CHAR(25) NOT NULL,
    P_BRAND         CHAR(10) NOT NULL,
    P_TYPE          VARCHAR(25) NOT NULL,
    P_SIZE          INTEGER NOT NULL,
    P_CONTAINER     CHAR(10) NOT NULL,
    P_RETAILPRICE   DECIMAL(15,2) NOT NULL,
    P_COMMENT       VARCHAR(23) NOT NULL
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE REGION (
        R_REGIONKEY INTEGER not null primary key,
        R_NAME      CHAR(25) NOT NULL,
        R_COMMENT   VARCHAR(152)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE NATION (
        N_NATIONKEY INTEGER not null primary key,
        N_NAME      CHAR(25) NOT NULL,
        N_REGIONKEY INTEGER NOT NULL references REGION(R_REGIONKEY),
        N_COMMENT   VARCHAR(152)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE SUPPLIER (
    S_SUPPKEY       INTEGER not null primary key,
    S_NAME          CHAR(25) NOT NULL,
    S_ADDRESS       VARCHAR(40) NOT NULL,
    S_NATIONKEY     INTEGER NOT NULL references NATION(N_NATIONKEY),
    S_PHONE         CHAR(15) NOT NULL,
    S_ACCTBAL       DECIMAL(15,2) NOT NULL,
    S_COMMENT       VARCHAR(101) NOT NULL
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE PARTSUPP (
    PS_PARTKEY      INTEGER NOT NULL references PART(P_PARTKEY),
    PS_SUPPKEY      INTEGER NOT NULL references SUPPLIER(S_SUPPKEY),
    PS_AVAILQTY     INTEGER NOT NULL,
    PS_SUPPLYCOST   DECIMAL(15,2) NOT NULL,
    PS_COMMENT      VARCHAR(199) NOT NULL
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE CUSTOMER (
    C_CUSTKEY       INTEGER not null primary key,
    C_NAME          VARCHAR(25) NOT NULL,
    C_ADDRESS       VARCHAR(40) NOT NULL,
    C_NATIONKEY     INTEGER NOT NULL references NATION(N_NATIONKEY),
    C_PHONE         CHAR(15) NOT NULL,
    C_ACCTBAL       DECIMAL(15,2) NOT NULL,
    C_MKTSEGMENT    CHAR(10) NOT NULL,
    C_COMMENT       VARCHAR(117) NOT NULL
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE ORDERS (
    O_ORDERKEY      INTEGER not null primary key,
    O_CUSTKEY       INTEGER NOT NULL references CUSTOMER(C_CUSTKEY),
    O_ORDERSTATUS   CHAR(1) NOT NULL,
    O_TOTALPRICE    DECIMAL(15,2) NOT NULL,
    O_ORDERDATE     DATE NOT NULL,
    O_ORDERPRIORITY CHAR(15) NOT NULL,
    O_CLERK         CHAR(15) NOT NULL,
    O_SHIPPRIORITY  INTEGER NOT NULL,
    O_COMMENT       VARCHAR(79) NOT NULL
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE LINEITEM (
    L_ORDERKEY      INTEGER NOT NULL references ORDERS(O_ORDERKEY),
    L_PARTKEY       INTEGER NOT NULL references PART(P_PARTKEY),
    L_SUPPKEY       INTEGER NOT NULL references SUPPLIER(S_SUPPKEY),
    L_LINENUMBER    INTEGER NOT NULL,
    L_QUANTITY      DECIMAL(15,2) NOT NULL,
    L_EXTENDEDPRICE DECIMAL(15,2) NOT NULL,
    L_DISCOUNT      DECIMAL(15,2) NOT NULL,
    L_TAX           DECIMAL(15,2) NOT NULL,
    L_RETURNFLAG    CHAR(1) NOT NULL,
    L_LINESTATUS    CHAR(1) NOT NULL,
    L_SHIPDATE      DATE NOT NULL,
    L_COMMITDATE    DATE NOT NULL,
    L_RECEIPTDATE   DATE NOT NULL,
    L_SHIPINSTRUCT  CHAR(25) NOT NULL,
    L_SHIPMODE      CHAR(10) NOT NULL,
    L_COMMENT       VARCHAR(44) NOT NULL
    );)");
}

void InitDatabaseWithNull(std::string db_name) {
    TestSidedoor_DBCommand("create database " + db_name + ";", db_name);

    TestSidedoor_Normal(db_name, R"(CREATE TABLE PART (
    P_PARTKEY       INTEGER not null primary key,
    P_NAME          VARCHAR(55),
    P_MFGR          CHAR(25),
    P_BRAND         CHAR(10),
    P_TYPE          VARCHAR(25),
    P_SIZE          INTEGER,
    P_CONTAINER     CHAR(10),
    P_RETAILPRICE   DECIMAL(15,2),
    P_COMMENT       VARCHAR(23)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE REGION (
        R_REGIONKEY INTEGER not null primary key,
        R_NAME      CHAR(25),
        R_COMMENT   VARCHAR(152)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE NATION (
        N_NATIONKEY INTEGER not null primary key,
        N_NAME      CHAR(25),
        N_REGIONKEY INTEGER NOT NULL references REGION(R_REGIONKEY),
        N_COMMENT   VARCHAR(152)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE SUPPLIER (
    S_SUPPKEY       INTEGER not null primary key,
    S_NAME          CHAR(25),
    S_ADDRESS       VARCHAR(40),
    S_NATIONKEY     INTEGER NOT NULL references NATION(N_NATIONKEY),
    S_PHONE         CHAR(15),
    S_ACCTBAL       DECIMAL(15,2),
    S_COMMENT       VARCHAR(101)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE PARTSUPP (
    PS_PARTKEY      INTEGER NOT NULL references PART(P_PARTKEY),
    PS_SUPPKEY      INTEGER NOT NULL references SUPPLIER(S_SUPPKEY),
    PS_AVAILQTY     INTEGER,
    PS_SUPPLYCOST   DECIMAL(15,2),
    PS_COMMENT      VARCHAR(199)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE CUSTOMER (
    C_CUSTKEY       INTEGER not null primary key,
    C_NAME          VARCHAR(25),
    C_ADDRESS       VARCHAR(40),
    C_NATIONKEY     INTEGER NOT NULL references NATION(N_NATIONKEY),
    C_PHONE         CHAR(15),
    C_ACCTBAL       DECIMAL(15,2),
    C_MKTSEGMENT    CHAR(10),
    C_COMMENT       VARCHAR(117)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE ORDERS (
    O_ORDERKEY      INTEGER not null primary key,
    O_CUSTKEY       INTEGER NOT NULL references CUSTOMER(C_CUSTKEY),
    O_ORDERSTATUS   CHAR(1),
    O_TOTALPRICE    DECIMAL(15,2),
    O_ORDERDATE     DATE,
    O_ORDERPRIORITY CHAR(15),
    O_CLERK         CHAR(15),
    O_SHIPPRIORITY  INTEGER,
    O_COMMENT       VARCHAR(79)
    );)");

    TestSidedoor_Normal(db_name, R"(CREATE TABLE LINEITEM (
    L_ORDERKEY      INTEGER NOT NULL references ORDERS(O_ORDERKEY),
    L_PARTKEY       INTEGER NOT NULL references PART(P_PARTKEY),
    L_SUPPKEY       INTEGER NOT NULL references SUPPLIER(S_SUPPKEY),
    L_LINENUMBER    INTEGER,
    L_QUANTITY      DECIMAL(15,2),
    L_EXTENDEDPRICE DECIMAL(15,2),
    L_DISCOUNT      DECIMAL(15,2),
    L_TAX           DECIMAL(15,2),
    L_RETURNFLAG    CHAR(1),
    L_LINESTATUS    CHAR(1),
    L_SHIPDATE      DATE,
    L_COMMITDATE    DATE,
    L_RECEIPTDATE   DATE,
    L_SHIPINSTRUCT  CHAR(25),
    L_SHIPMODE      CHAR(10),
    L_COMMENT       VARCHAR(44)
    );)");
}
