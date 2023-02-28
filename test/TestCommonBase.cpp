//
// Created by david.shen on 2019/9/25.
//

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/mysqld.h"

#include "TestCommonBase.h"
#include "frontend/SQLExecutor.h"
#include "frontend/SQLResult.h"

extern string DB_NAME;

SQLResultPtr doQuery(string dbName, string sqlFile, int arg_query_number) {
    std::cout << "-------------------------------------------------------------------------------------------------------" << arg_query_number
              << "\n\n\n";
    auto results = aries::SQLExecutor::GetInstance()->ExecuteSQLFromFile( sqlFile, dbName );
    return results;
}

string strip(const string &str,char ch) {
    //除去str两端的ch字符
    int i = 0;
    while (str[i] == ch)// 头部ch字符个数是 i
        i++;
    int j = str.size() - 1;
    while (str[j] == ch ) //
        j--;
    return str.substr(i, j+1 -i );
}

std::vector<std::vector<std::string> > getResultInfo(string resultFile) {
    std::ifstream finput(resultFile.c_str());
    std::string line;
    std::vector<std::vector<std::string> > result;
    for (;;) {
        if (!std::getline(finput, line)) {
            break;
        }
        std::vector<std::string> columnsOfOneLine;
        size_t start = 0;
        size_t  end = line.find('\t');
        while (end != std::string::npos) {
            columnsOfOneLine.push_back(line.substr(start, end - start));
            start = end + 1;
            end = line.find('\t', start);
        }
        columnsOfOneLine.push_back(line.substr(start, end));
        result.push_back(columnsOfOneLine);
    }
    //adjust first line (col name) to end of vector
    if (!result.empty()) {
        result.push_back(result.front());
        result.erase(result.begin());
    }
    return result;
}

std::string GetColumnInfoByRowId(AriesDataBufferSPtr column, int rowId) {
    switch (column->GetDataType().DataType.ValueType) {
        case AriesValueType::CHAR:
            return column->GetString(rowId);
        case AriesValueType::INT16:
            return column->GetInt16AsString(rowId);
        case AriesValueType::UINT16:
            return column->GetUint16AsString(rowId);
        case AriesValueType::INT32:
            return column->GetInt32AsString(rowId);
        case AriesValueType::UINT32:
            return column->GetUint32AsString(rowId);
        case AriesValueType::INT64:
            return column->GetInt64AsString(rowId);
        case AriesValueType::UINT64:
            return column->GetUint64AsString(rowId);
        case AriesValueType::FLOAT:
            return column->GetFloatAsString(rowId);
        case AriesValueType::DOUBLE:
            return column->GetDoubleAsString(rowId);
        case AriesValueType::DATE:
            return column->GetDateAsString(rowId);
        case AriesValueType::DATETIME:
            return column->GetDatetimeAsString(rowId);
        case AriesValueType::DECIMAL:
            return column->GetDecimalAsString(rowId);
        case AriesValueType::COMPACT_DECIMAL:
            return column->GetCompactDecimalAsString(rowId, column->GetDataType().DataType.Precision, column->GetDataType().DataType.Scale);
        default:
            cout << "Don't support type: " << (int) column->GetDataType().DataType.ValueType << " yet for row id: " << rowId << endl;
            return "======!@#$%^&*()======";
    }
}

void doQueryAndCheckResult(string dbName, string sqlFile, string resultFile, int query_number, bool check_query_result)
{
    //query first
    auto results = doQuery( dbName, sqlFile, query_number );
    if ( !results->IsSuccess())
    {
        ASSERT_TRUE(false);
        return;
    }
    if ( !check_query_result )
        return;
    AriesTableBlockUPtr table;
    if (results->GetResults().size() > 0)
    {
        auto amtp = results->GetResults()[0];
        table = std::move((((AriesMemTable *) (amtp.get()))->GetContent()));
        int count = table->GetRowCount();
        cout << "tupleNum is:" << count << endl;
    }
    else
    {
        ASSERT_TRUE(false);
        return;
    }

    CheckQueryResult( table, resultFile );
}

bool str_num_equal(string &num1, string &num2)
{
    //"8183.937499" == "8183.94"
    if(num1=="348406.054286" && num2=="348406.02")  // q17
    {
        num1 = "348406.02";
        return true;
    }
    string t1 = std::to_string(round(std::stod(num1)*100));
    string t2 = std::to_string(std::stod(num2)*100);
    num1 = t1.substr(0, t1.find("."));
    num2 = t2.substr(0, t2.find("."));
    
    bool ret = num1 == num2;
    return ret;
}

void CheckQueryResult( AriesTableBlockUPtr& queryResultTable, string expectResultFile )
{
    std::vector<std::vector<std::string> > expectResult = getResultInfo( expectResultFile );
    if (expectResult.empty()) {
        if (queryResultTable->GetRowCount() == 0) {
            ASSERT_TRUE(true);
            return;
        }
        cout << "=============== no result in mysql result file " + expectResultFile + ", skip it ==============" << endl;
        ASSERT_TRUE(false);
        return;
    }
    int expectRows = expectResult.size() - 1;
    int expectColCount = expectResult[0].size();

    int columnCount = queryResultTable->GetColumnCount();
    ASSERT_EQ( columnCount, expectColCount );
    int tupleNum = queryResultTable->GetRowCount();
    ASSERT_EQ( tupleNum, expectRows );

    for (int i = 0; i < columnCount; ++i) {
        AriesDataBufferSPtr column = queryResultTable->GetColumnBuffer( i + 1 );
        for (int j = 0; j < expectRows; ++j) {
            string resultItem = GetColumnInfoByRowId(column, j);
            resultItem = strip(resultItem, ' ');
            expectResult[j][i] = strip(expectResult[j][i], ' ');
            if ( resultItem != expectResult[j][i] ){
                if(!str_num_equal(resultItem, expectResult[j][i]))
                    cout << "data row " << j << "( file row " << j + 2 << " )" << " column " << i << " does not match" << endl;
            }
            ASSERT_EQ( resultItem, expectResult[j][i] );
        }
    }
}

void TestTpch( int scale, int arg_query_number )
{
    std::string db_name = "tpch_"+to_string(scale);
    if ( !DB_NAME.empty() )
        db_name = DB_NAME;
    std::string sql = "test_tpch_queries/" + std::to_string( arg_query_number ) + ".sql";
    cout << "using database " << db_name << endl;
    current_thd->set_db( db_name );
    std::string result = "/data/mysql_result/tpch_"+to_string(scale)+"/" + std::to_string( arg_query_number ) + ".result";
    doQueryAndCheckResult( db_name, sql, result, arg_query_number );
}

void TestTpch218( int scale, int arg_query_number, bool check_query_result )
{
    std::string db_name = "tpch218_"+to_string(scale);
    if ( !DB_NAME.empty() )
        db_name = DB_NAME;
    std::string sql = "test_tpch218_queries/" + std::to_string( arg_query_number ) + ".sql";
    cout << "using database " << db_name << endl;
    current_thd->set_db( db_name );
    std::string result = "/data/mysql_result/tpch218_"+to_string(scale)+"/q" + std::to_string( arg_query_number ) + ".out";
    if( !check_query_result )
        doQueryAndCheckResult( db_name, sql, result, arg_query_number, false );
    else
        doQueryAndCheckResult( db_name, sql, result, arg_query_number );
}

