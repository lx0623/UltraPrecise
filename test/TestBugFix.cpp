#include <gtest/gtest.h>
#include <regex>
#include <string>
#include <thread>

#include "AriesEngineWrapper/AriesMemTable.h"
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"
#include "AriesEngineWrapper/AriesExprBridge.h"
#include <server/mysql/include/mysqld_error.h>

#include <schema/SchemaManager.h>
#include "schema/DatabaseEntry.h"
#include "schema/TableEntry.h"

#include "AriesEngine/AriesJoinNodeHelper.h"
#include "optimizer/QueryOptimizer.h"
#include "utils/string_util.h"

#include "TestUtils.h"
#include "CudaAcc/DynamicKernel.h"

using namespace aries_engine;
using namespace std;
using std::string;

using aries::schema::SchemaManager;
using aries::schema::DatabaseEntry;
using aries::schema::TableEntry;

static const auto db_name = "scale_1";

using namespace aries_test;

TEST(bugfix, 639) {
    auto left_sql = "select n_nationkey % 3 from nation;";
    auto right_sql = "select mod(n_nationkey, 3) from nation;";

    AssertTwoSQLS( left_sql, right_sql, db_name );
}

TEST(bugfix, 485) {
    auto sql = R"(
select n_name, ( n_comment = "china" ) as comment from nation order by comment;
    )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_TRUE(result->GetResults().size() == 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 25);
    ASSERT_EQ(table->GetColumnCount(), 2);

    auto column_name = table->GetColumnBuffer(1);
    auto column_comment = table->GetColumnBuffer(2);

    for (int i = 0; i < table->GetRowCount(); i++) {
        auto value = column_comment->GetNullableInt8(i);
        ASSERT_TRUE(value.flag == 1);
        ASSERT_EQ(value.value, 0);
    }
}

TEST(bugfix, 486) {
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( "select n_name, n_nationkey + 1 from nation order by n_nationkey + 1;", db_name );
    ASSERT_TRUE(result->IsSuccess());
    ASSERT_TRUE(result->GetResults().size() == 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    auto result_raw = aries::SQLExecutor::GetInstance()->ExecuteSQL( "select n_name, n_nationkey from nation order by n_nationkey + 1;", db_name );
    ASSERT_TRUE(result_raw->IsSuccess());
    ASSERT_TRUE(result_raw->GetResults().size() == 1);

    auto buffer_raw = std::dynamic_pointer_cast<AriesMemTable>(result_raw->GetResults()[0]);
    auto table_raw = buffer_raw->GetContent();

    ASSERT_EQ(table_raw->GetRowCount(), 25);
    ASSERT_EQ(table->GetRowCount(), table_raw->GetRowCount());

    auto column_name_raw = table_raw->GetColumnBuffer(1);
    auto column_key_raw = table_raw->GetColumnBuffer(2);

    auto column_name = table->GetColumnBuffer(1);
    auto column_key = table->GetColumnBuffer(2);

    for (int i = 0; i < table_raw->GetRowCount(); i++) {
        ASSERT_EQ(column_name->GetString(i), column_name_raw->GetString(i));
        ASSERT_EQ(column_key_raw->GetInt32(i) + 1, column_key->GetInt32(i));
    }
}

TEST(bugfix, 655) {
    auto sql = R"(
select 
    case 
        when l_orderkey % 10 <= 2 then 'SMALL'
        when l_orderkey % 10 < 5 then 'MIDDLE' 
        else 'BIG' 
    end '大小',
    sum(l_orderkey) 
from
    lineitem
group by 1 
order by 1;
    )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
    ASSERT_TRUE(result->IsSuccess());
    ASSERT_TRUE(result->GetResults().size() == 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 3);
    ASSERT_EQ(table->GetColumnCount(), 2);

    auto size_column = table->GetColumnBuffer(1);
    auto sum_column = table->GetColumnBuffer(2);

    ASSERT_EQ(size_column->GetString(0), "BIG");
    ASSERT_EQ(size_column->GetString(1), "MIDDLE");
    ASSERT_EQ(size_column->GetString(2), "SMALL");

    ASSERT_EQ(sum_column->GetInt64AsString(0), "9012913327887");
    ASSERT_EQ(sum_column->GetInt64AsString(1), "3600629703761");
    ASSERT_EQ(sum_column->GetInt64AsString(2), "5391779933301");

}

TEST(bugfix, 656) {
    auto sql = R"(
select 
    distinct a.o_custkey
from
    (
        select 
            l.l_orderkey, o_custkey, l_shipdate
        from
            lineitem l
            join orders o on l.l_orderkey = o.o_orderkey
        where 
            l.l_shipdate > '1995-12-01' and l.l_shipdate < '1995-12-03'
    ) a
    join (
        select
            l.l_orderkey
        from
            lineitem l
        where
            l.l_shipdate >= '1994-12-01'
    ) b on a.l_orderkey = b.l_orderkey
    join (
        select
            l.l_orderkey
        from
            lineitem l
        where 
            l.l_shipdate > '1995-12-01' and l.l_shipdate < '1995-12-03'
    ) c on a.l_orderkey = c.l_orderkey
order by o_custkey
    )";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );
    ASSERT_TRUE(result->IsSuccess());
    ASSERT_TRUE(result->GetResults().size() == 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 2454);
    ASSERT_EQ(table->GetColumnCount(), 1);

    auto column_buffer = table->GetColumnBuffer(1);

    ASSERT_EQ(column_buffer->GetInt32AsString(2453), "149965");

    auto value = column_buffer->GetInt32(2453);
    for (int i = table->GetRowCount() - 2; i >= 0; i--) {
        ASSERT_LT(column_buffer->GetInt32(i), value);
        value = column_buffer->GetInt32(i);
    }
}

TEST(bugfix, 657) {
    auto sql = "select * from nation order by 'abc';";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_TRUE(result->GetResults().size() == 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 25);
    ASSERT_EQ(table->GetColumnCount(), 4);

    auto column_key = table->GetColumnBuffer(1);
    auto column_comment = table->GetColumnBuffer(2);

    for (int i = 0; i < table->GetRowCount(); i++) {
        ASSERT_EQ(column_key->GetInt32(i), i);
    }
}

TEST(bugfix, 663) {
    auto expression = std::make_shared<aries::CommonBiaodashi>(BiaodashiType::Inop, 0);
    auto function = std::make_shared<aries::SQLFunction>("SUBSTRING");
    auto value = std::make_shared<aries::CommonBiaodashi>(BiaodashiType::SQLFunc, function);
    value->SetValueType(BiaodashiValueType::TEXT);
    value->SetLength(23);

    auto arg1 = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, std::string("here is a test string content"));
    arg1->SetValueType(BiaodashiValueType::TEXT);
    arg1->SetLength(29);
    value->AddChild(arg1);

    auto arg2 = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, 1);
    arg2->SetValueType(BiaodashiValueType::INT);
    arg2->SetLength(1);
    value->AddChild(arg2);

    auto arg3 = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, 23);
    arg3->SetValueType(BiaodashiValueType::INT);
    arg3->SetLength(1);
    value->AddChild(arg3);

    expression->AddChild(value);

    auto in_arg1 = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, 1997);
    in_arg1->SetValueType(BiaodashiValueType::INT);
    in_arg1->SetLength(1);
    expression->AddChild(in_arg1);

    auto data_function = std::make_shared<aries::SQLFunction>("NOW");
    auto in_arg2 = std::make_shared<CommonBiaodashi>(BiaodashiType::SQLFunc, data_function);
    in_arg2->SetValueType(BiaodashiValueType::DATE_TIME);
    in_arg2->SetLength(1);

    expression->AddChild(in_arg2);
    expression->SetValueType(BiaodashiValueType::BOOL);

    aries_engine::AriesExprBridge bridge;

    auto bridged = bridge.Bridge(expression);

    ASSERT_EQ( bridged->GetType(), AriesExprType::TRUE_FALSE );
    ASSERT_EQ( boost::get< bool >( bridged->GetContent() ), false );
}

extern void BuildQuery(SelectStructurePointer& query, const string& dbName, bool needRowIdColumn = false);

TEST(bugfix, 664) {
    // default mode of loading csv does not optimize char columns
    // /**
    //  * 虽然在 schema 中 n_comment 的类型是 char(152)，但是我们将其优化成了 114 的长度，
    //  * 所以这里是按照114的长度来计算的。
    //  */
    auto sql = "select substring(n_comment, -2) from nation;";
    std::vector<AriesSQLStatementPointer> statements;

    auto r = aries::SQLExecutor::GetInstance()->ParseSQL(sql, false, statements);

    ASSERT_EQ(r.first, 0);
    ASSERT_EQ(statements.size(), 1);
    ASSERT_TRUE(statements[0]->IsQuery());

    auto query = std::dynamic_pointer_cast<SelectStructure>(statements[0]->GetQuery());
    BuildQuery(query, db_name);

    auto select_part = query->GetSelectPart();
    ASSERT_EQ(select_part->GetAllExprCount(), 1);

    auto select_item = std::dynamic_pointer_cast<aries::CommonBiaodashi>(select_part->GetAllExprs()[0]);

    ASSERT_EQ(select_item->GetType(), BiaodashiType::SQLFunc);
    ASSERT_EQ(select_item->GetValueType(), BiaodashiValueType::TEXT);
    ASSERT_EQ(select_item->GetLength(), 2);

    sql = "select substring(n_comment, 1) from nation;";
    statements.clear();

    r = aries::SQLExecutor::GetInstance()->ParseSQL(sql, false, statements);

    ASSERT_EQ(r.first, 0);

    query = std::dynamic_pointer_cast<SelectStructure>(statements[0]->GetQuery());
    BuildQuery(query, db_name);

    select_part = query->GetSelectPart();

    select_item = std::dynamic_pointer_cast<aries::CommonBiaodashi>(select_part->GetAllExprs()[0]);

    // default mode of loading csv does not optimize char columns
    ASSERT_EQ(select_item->GetLength(), 152);
    // ASSERT_EQ(select_item->GetLength(), 114);

    sql = "select substring(n_comment, 101, 100) from nation;";
    statements.clear();

    r = aries::SQLExecutor::GetInstance()->ParseSQL(sql, false, statements);

    ASSERT_EQ(r.first, 0);

    query = std::dynamic_pointer_cast<SelectStructure>(statements[0]->GetQuery());
    BuildQuery(query, db_name);

    select_part = query->GetSelectPart();

    select_item = std::dynamic_pointer_cast<aries::CommonBiaodashi>(select_part->GetAllExprs()[0]);
    ASSERT_EQ(select_item->GetLength(), 52);
    // ASSERT_EQ(select_item->GetLength(), 14);
}

TEST(bugfix, 683) {
    auto sql = "select case when l_returnflag = 0 then 'zero' when l_returnflag = '1' then 'one' else 'else' end tt from lineitem limit 17";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_TRUE(result->GetResults().size() == 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 17);

}

TEST(bugfix, 687) {
    auto sql = "select l_returnflag, l_linestatus, max(l_partkey) max_partkey from lineitem group by 1, 2 having max_partkey > 199999;";


    auto sql2 = "select l_returnflag, l_linestatus, max(l_partkey) max_partkey from lineitem group by 1, 2 having max(l_partkey) > 199999;";
    AssertTwoSQLS( sql, sql2, db_name );
}

TEST(bugfix, 690) {
    auto sql1 = R"(
select
    count(l_orderkey),
    sum(l_partkey), 
    sum(l_suppkey),
    sum(l_partkey) - sum(l_suppkey) alias
from
    lineitem 
group by 
    l_returnflag, l_linestatus
order by 1,2;
    )";

    auto sql2 = R"(
select
    count(l_orderkey),
    sum(l_partkey),
    sum(l_suppkey),
    sum(l_partkey) - sum(l_suppkey) alias
from
    lineitem 
group by
    l_returnflag, l_linestatus
order by
    1, sum(l_partkey) - sum(l_suppkey);
    )";

    AssertTwoSQLS( sql1, sql2, db_name );
}

TEST(bugfix, 691) {
    auto sql1 = R"(
select
    count(l_orderkey),
    sum(l_partkey), 
    l_linenumber + 1 l_linenumber
from
    lineitem 
group by 
    l_returnflag, l_linestatus, l_linenumber
order by 1,2;
    )";

    auto sql2 = R"(
select
    count(l_orderkey),
    sum(l_partkey), 
    l_linenumber + 1 alias
from
    lineitem 
group by 
    l_returnflag, l_linestatus, l_linenumber
order by
    1, 2;
    )";

    AssertTwoSQLS( sql1, sql2, db_name );
}

TEST(bugfix, 692) {
    auto sql = "select l_orderkey from lineitem l join orders o on l.l_orderkey = o.o_orderkey and o.o_custkey is null;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_EQ(result->GetResults().size(), 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 0);
}

TEST(bugfix, 693) {
    auto sql = "select l_orderkey from lineitem l join orders o on l.l_orderkey = o.o_orderkey and o.o_custkey is not null;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_EQ(result->GetResults().size(), 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 6001215);
}

TEST(bugfix, 694) {
    auto sql = "select sum(s_suppkey), max(s_suppkey), min(s_suppkey), count(s_suppkey) from supplier where s_suppkey > 10000;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_EQ(result->GetResults().size(), 1);

    auto buffer = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = buffer->GetContent();

    ASSERT_EQ(table->GetRowCount(), 1);
    // verify column 1
    AriesDataBufferSPtr column = table->GetColumnBuffer( 1 );
    ASSERT_EQ( column->GetString( 0 ), "NULL" );
    // verify column 2
    column = table->GetColumnBuffer( 2 );
    ASSERT_EQ( column->GetString( 0 ), "NULL" );
    // verify column 3
    column = table->GetColumnBuffer( 3 );
    ASSERT_EQ( column->GetString( 0 ), "NULL" );
    // verify column 4
    column = table->GetColumnBuffer( 4 );
    ASSERT_EQ( column->GetInt64( 0 ), 0 );
}

TEST(bugfix, 706)
{
    auto sql = " select count( * ) from lineitem where l_orderkey * (l_partkey /l_suppkey) - 100 = l_quantity * l_partkey;";

    std::vector<AriesSQLStatementPointer> statements;
    auto res = aries::SQLExecutor::GetInstance()->ParseSQL(sql, false, statements);

    ASSERT_EQ(res.first, 0);

    ASSERT_EQ(statements.size(), 1);

    ASSERT_TRUE(statements[0]->IsQuery());

    auto query = std::dynamic_pointer_cast<SelectStructure>(statements[0]->GetQuery());

    BuildQuery(query, db_name);

    auto plan_tree = query->GetQueryPlanTree();

    plan_tree = aries::QueryOptimizer::GetQueryOptimizer()->OptimizeTree(plan_tree);

    std::cout << "plan_tree\n" << plan_tree->ToString(0) << std::endl;
    
    SQLTreeNodePointer where_node;
    
    auto cursor = plan_tree;
    while (cursor) {
        if (cursor->GetType() == SQLTreeNodeType::Filter_NODE) {
            where_node = cursor;
            break;
        }

        cursor = cursor->GetTheChild();
    }

    ASSERT_TRUE(where_node);

    auto condition_expr = where_node->GetFilterStructure();

    AriesExprBridge bridge;
    auto condition = bridge.Bridge(condition_expr);
    int exprId = 0;
    condition->SetId( exprId );

    aries_engine::AriesJoinNodeHelper helper(nullptr, condition, condition, AriesJoinType::INNER_JOIN, 0 );

    std::cout << helper.GetDynamicCode() << std::endl;

    std::string code = R"(
#include "functions.hxx"
#include "decimal.hxx"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesIntervalTime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"
#include "AriesYear.hxx"
#include "aries_char.hxx"
#include "AriesSqlFunctions.hxx"
#include "AriesColumnDataIterator.hxx"
using namespace aries_acc;
    )";

    code += helper.GetDynamicCode();

    AriesDynamicCodeInfo codeInfo;
    codeInfo.KernelCode = code;
    auto modules = AriesDynamicKernelManager::GetInstance().CompileKernels( codeInfo );

    ASSERT_TRUE(!modules->Modules.empty());
}


TEST(bugfix, 733)
{
    auto sql1 = "select * from lineitem where l_partkey in ('155190', '67310') order by l_orderkey;";
    auto sql2 = "select * from lineitem where l_partkey in (155190, 67310) order by l_orderkey;";

    AssertTwoSQLS(sql1, sql2, db_name);
}

TEST(bugfix, 735)
{
    auto sql1 = R"(
select
	sum(l_extendedprice * l_discount) as revenue,
	l_shipdate
from
	lineitem
where
	l_shipdate >= '1995-01-01'
	and abs(datediff(l_shipdate, '1995-10-1')) < 3
group by
    l_shipdate
having 
    abs( sum(l_extendedprice) - sum(l_discount) ) > 10
order by
    l_shipdate
;
    )";

    auto sql2 = R"(
select
	sum(l_extendedprice * l_discount) as revenue,
	l_shipdate
from
	lineitem
where
	l_shipdate >= '1995-01-01'
	and abs(datediff(l_shipdate, date('1995-10-1'))) < 3
group by
    l_shipdate
having 
    abs( sum(l_extendedprice) - sum(l_discount) ) > 10
order by
    l_shipdate
;
    )";

    AssertTwoSQLS(sql1, sql2, db_name);
}

TEST(bugfix, 741)
{
    auto sql1 = R"(
select
    n_nationkey,
    n_name,
    r_regionkey,
    r_name
from
    nation 
    left join region on n_nationkey % 4 > r_regionkey % 2
order by
    n_nationkey, r_regionkey;
    )";

    auto sql2 = R"(
select
    n_nationkey,
    n_name,
    r_regionkey,
    r_name
from
    region 
    right join nation on n_nationkey % 4 > r_regionkey % 2
order by
    n_nationkey, r_regionkey;
    )";

    AssertTwoSQLS( sql1, sql2, db_name );
}

TEST(bugfix, 742)
{
    auto sql = R"(
select
    n_nationkey,
    n_name,
    r_regionkey,
    r_name
from
    nation 
    full join region on  n_nationkey % 4 > r_regionkey % 2
order by
    n_nationkey, r_regionkey;
    )";

    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );

    auto buffer = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] );
    auto table = buffer->GetContent();

    ASSERT_EQ( table->GetRowCount(), 85 );
    ASSERT_EQ( table->GetColumnCount(), 4 );

    auto column1 = table->GetColumnBuffer( 1 );
    auto column2 = table->GetColumnBuffer( 2 );
    auto column3 = table->GetColumnBuffer( 3 );
    auto column4 = table->GetColumnBuffer( 4 );

    ASSERT_EQ( column1->GetInt32AsString( 0 ), "0" );
    ASSERT_EQ( column3->GetInt32AsString( 0 ), "NULL" );

    ASSERT_EQ( column1->GetInt32AsString( 4 ), "2" );
    ASSERT_EQ( column2->GetString( 4 ), "BRAZIL" );
    ASSERT_EQ( column3->GetInt32AsString( 4 ), "0" );
    ASSERT_EQ( column4->GetString( 4 ), "AFRICA" );

    ASSERT_EQ( column1->GetInt32AsString( 14 ), "4" );
    ASSERT_EQ( column2->GetString( 14 ), "EGYPT" );
    ASSERT_EQ( column3->GetInt32AsString( 14 ), "NULL" );
    ASSERT_EQ( column4->GetString( 14 ), "NULL" );

    ASSERT_EQ( column1->GetInt32AsString( 84 ), "24" );
    ASSERT_EQ( column2->GetString( 84 ), "UNITED STATES" );
    ASSERT_EQ( column3->GetInt32AsString( 84 ), "NULL" );
    ASSERT_EQ( column4->GetString( 84 ), "NULL" );
}

TEST(bugfix, 743)
{
    auto sql1 = "select * from nation, region order by 1, 5 desc;";
    auto sql2 = "select nation.n_nationkey, nation.n_name, nation.n_regionkey, nation.n_comment, region.r_regionkey, "
                "region.r_name, region.r_comment from nation, region order by 1, 5 desc";

    AssertTwoSQLS(sql1, sql2, db_name);
}


TEST(bugfix, 747)
{
    auto sql1 = "select count(*) from lineitem;";
    auto sql2 = "select count(*) from (select l_orderkey l from lineitem) s;";
    // auto sql3 = "select count(*) from lineitem l left join orders o on l.l_orderkey = o.o_orderkey;";

    AssertTwoSQLS(sql1, sql2, db_name);
}

TEST(bugfix, 795)
{
    auto sql = "select sum(a.l_quantity) from (select l_quantity from lineitem order by l_orderkey, l_linenumber limit 2) a;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, db_name );

    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );

    auto buffer = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[0] );
    auto table = buffer->GetContent();

    ASSERT_EQ( table->GetRowCount(), 1 );
    ASSERT_EQ( table->GetColumnCount(), 1 );

    ASSERT_EQ( table->GetColumnBuffer( 1 )->GetDecimalAsString( 0 ), "53.00" );
}

TEST(bugfix, 801)
{
    auto sql1 = R"(
select
    datediff(if(l_orderkey % 2 = 0, l_shipdate, '2019-10-11'), '2019-10-11') 
from
    lineitem
order by
    l_orderkey
limit 20;
    )";

    auto sql2 = R"(
select
    datediff(if(l_orderkey % 2 = 0, l_shipdate, date('2019-10-11')), date('2019-10-11')) 
from
    lineitem
order by
    l_orderkey
limit 20;
    )";

    AssertTwoSQLS(sql1, sql2, db_name);
}

TEST(bugfix, 871) {
    auto sql = R"(
 insert into nation values ( 111, "111", 111, "111"), ( 222, "2222", 222, "2222");
    )";
    auto result = SQLExecutor::GetInstance()->ExecuteSQL(sql, db_name);
    EXPECT_TRUE(result->IsSuccess());

    result = SQLExecutor::GetInstance()->ExecuteSQL("select * from nation where n_nationkey=111 or n_nationkey=222;", db_name);

    ASSERT_TRUE(result->IsSuccess());
    ASSERT_EQ(result->GetResults().size(), 1);

    auto mem_table = std::dynamic_pointer_cast<AriesMemTable>(result->GetResults()[0]);
    auto table = mem_table->GetContent();

    ASSERT_EQ(table->GetRowCount(), 2);
    ASSERT_EQ(table->GetColumnCount(), 4);

    auto column1 = table->GetColumnBuffer(1);
    auto column4 = table->GetColumnBuffer(4);
    ASSERT_EQ(column1->GetInt32AsString(0), "111");
    ASSERT_EQ(column1->GetInt32AsString(1), "222");
    ASSERT_EQ(column4->GetString(0), "111");
    ASSERT_EQ(column4->GetString(1), "2222");


    SQLExecutor::GetInstance()->ExecuteSQL("delete from nation where n_nationkey=111 or n_nationkey=222;", db_name);
}

/**
 * std::string get_current_work_directory()
 */
TEST( bugfix, 876 )
{
    static int cnt = 1000;
    std::vector< std::shared_ptr< std::thread > > threads;

    char* dirs = new char[ PATH_MAX * cnt ];
    memset( dirs, 0, PATH_MAX * cnt );

    for ( int i = 0; i < cnt; i++ )
    {
        auto thread = std::make_shared< std::thread >( [ = ]()
        {
            auto dir = aries_utils::get_current_work_directory();
            ASSERT_FALSE( dir.empty() );

            strcpy( dirs + i * PATH_MAX, dir.c_str() );
        } );

        threads.emplace_back( thread );
    }


    for ( int i = 0; i < cnt; i++ )
    {
        threads[ i ]->join();
    }

    auto dir = aries_utils::get_current_work_directory();
    const auto dir_char = dir.c_str();

    for ( int i = 0; i < cnt; i++ )
    {
        ASSERT_EQ( strcmp( dir_char, dirs + i * PATH_MAX ), 0 );
    }
}

TEST( bugfix, 897 )
{
    std::string table_name( "lineitem" );
    auto db = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( db_name );
    auto table = db->GetTableByName( table_name );

    auto result = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseAndTableById( table->GetId() );

    ASSERT_TRUE( result.first && result.second );
    ASSERT_EQ( result.first->GetName(), db_name );
    ASSERT_EQ( result.second->GetName(), table_name );
    ASSERT_EQ( result.second->GetId(), table->GetId() );
}

TEST( bugfix, 900 )
{
    // test decimal In expr
    auto sql = "create database if not exists test;";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    sql = "drop table if exists t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    sql = "create table t1(a decimal(10,2));";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    sql = "SELECT a FROM t1 WHERE a IN(1, (SELECT 2));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "insert into t1 values (1),(2),(3),(4),(5),(6),(7);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    // sql = "SELECT a FROM t1 WHERE a IN(1, (SELECT 2));";
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    // ASSERT_TRUE( result->IsSuccess() );
    sql = "SELECT a FROM t1 WHERE a IN(1, 3);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "SELECT a FROM t1 WHERE a IN(1.0, 3.0);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );
    ASSERT_EQ( std::dynamic_pointer_cast< AriesMemTable >(result->GetResults()[0])->GetContent()->GetRowCount(), 2 );
    sql = "SELECT a FROM t1 WHERE a IN(2, 3.0);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );
    ASSERT_EQ( std::dynamic_pointer_cast< AriesMemTable >(result->GetResults()[0])->GetContent()->GetRowCount(), 2 );

    sql = "drop table if exists t1;";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "create table t1(a int); insert into t1 values(1),(2),(3),(4),(5),(6),(7),(8);";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    // sql = "select a from t1 where a in((select 1));";
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "SELECT a FROM t1 WHERE a IN((select 3), (select 4));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    ASSERT_EQ( result->GetResults().size(), 1 );
    ASSERT_EQ( std::dynamic_pointer_cast< AriesMemTable >(result->GetResults()[0])->GetContent()->GetRowCount(), 2 );
    sql = "select a from t1 where a in((select 1,2,3));";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_FALSE( result->IsSuccess() );
    ASSERT_EQ(result->GetErrorCode(), ER_OPERAND_COLUMNS);

    // sql = "select a from t1 where (a,a,a) in ((1,1,1),(1,2,1));";
    // result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    // ASSERT_TRUE( result->IsSuccess() );
}

TEST( bugfix, 901 )
{
    // 测试insert不会异常触发ER_OUT_OF_RESOURCES
    auto sql = "drop table if exists t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    sql = "CREATE TABLE t1 (a INT, b INT);";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 VALUES (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8);";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+8 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+16 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+32 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+64 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+128 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+256 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+512 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+1024 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+2048 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+4096 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+8192 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+16384 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
    sql = "INSERT INTO t1 SELECT a, b+32768 FROM t1;";
    aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, "test" );
    ASSERT_TRUE( result->IsSuccess() );
}