#include <gtest/gtest.h>
#include <string>

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"
#include "CudaAcc/AriesSqlOperator_group.h"

#include "utils/string_util.h"
#include "../../TestUtils.h"
#include "../../TestCommonBase.h"
using namespace std;
using namespace aries_test;

const string TEST_DB_NAME( "scale_1" );

class UT_self_join: public testing::Test
{
protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
        string sql = "drop table if exists lineitem_only_one_group;";
        aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, TEST_DB_NAME );
    }
};

TEST_F( UT_self_join, only_one_group )
{
    string tableName = "lineitem_only_one_group";
    InitTable( TEST_DB_NAME, tableName );

    string lineitemSchema = R"(
( l_orderkey    integer not null,
  l_partkey     integer not null,
  l_suppkey     integer not null,
  l_linenumber  integer not null,
  l_quantity    decimal(12,2) not null,
  l_extendedprice  decimal(12,2) not null,
  l_discount    decimal(12,2) not null,
  l_tax         decimal(12,2) not null,
  l_returnflag  char(1) not null,
  l_linestatus  char(1) not null,
  l_shipdate    date not null,
  l_commitdate  date not null,
  l_receiptdate date not null,
  l_shipinstruct char(25) not null encoding bytedict as l_shipinstruct,
  l_shipmode     char(10) not null encoding bytedict as l_shipmode,
  l_comment      varchar(44) not null,
  primary key ( l_orderkey, l_linenumber ) );
)";
    string testSql = R"(
select
	l_orderkey,
	count(*) as numwait
from
	lineitem_only_one_group l1
where
	exists (
		select
			*
		from
			lineitem_only_one_group l2
		where
			l2.l_orderkey = l1.l_orderkey
			and l2.l_suppkey <> l1.l_suppkey
	)
group by
	l_orderkey;
)";

    string sql = "create table " + tableName + lineitemSchema;
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    sql = "insert into " + tableName + " values ( 1, 155190, 7706, 1, 17, 21168.23, 0.04, 0.02, 'N', 'O', '1996-03-13', '1996-02-12', '1996-03-22', 'DELIVER IN PERSON', 'TRUCK', 'egular courts above the' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(testSql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());

    auto resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    //
    sql = "insert into " + tableName + " values ( 1, 155191, 7706, 11, 17, 21168.23, 0.04, 0.02, 'N', 'O', '1996-03-13', '1996-02-12', '1996-03-22', 'DELIVER IN PERSON', 'TRUCK', 'egular courts above the' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(testSql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 0 );

    // 
    sql = "insert into " + tableName + " values ( 1, 67310, 7311, 2, 36, 45983.16, 0.09, 0.06, 'N', 'O', '1996-04-12', '1996-02-28', '1996-04-20', 'TAKE BACK RETURN', 'MAIL', 'ly final dependencies: slyly bold' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(testSql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    auto columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), 3 );

    sql = "insert into " + tableName + " values ( 2, 106170, 1191, 1, 38, 44694.46, 0.00, 0.05, 'N', 'O', '1997-01-28', '1997-01-14', '1997-02-02', 'TAKE BACK RETURN', 'RAIL', 'ven requests. deposits breach a' )";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL(testSql, TEST_DB_NAME);
    ASSERT_TRUE(result->IsSuccess());
    resTable = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
    ASSERT_EQ( resTable->GetRowCount(), 1 );
    columnBuff = resTable->GetColumnBuffer( 1 );
    ASSERT_EQ( columnBuff->GetInt32( 0 ), 1 );
    columnBuff = resTable->GetColumnBuffer( 2 );
    ASSERT_EQ( columnBuff->GetInt64( 0 ), 3 );
}