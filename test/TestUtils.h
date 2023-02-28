#pragma once

#include <gtest/gtest.h>
#include "AriesEngine/AriesCommonExpr.h"
#include "AriesEngine/AriesDataDef.h"


#define ARIES_UNIT_TEST( test_suite_name, test_name ) TEST( UT_ ## test_suite_name, test_name )
#define ARIES_UNIT_TEST_F( test_suite_name, test_name ) TEST_F( UT_ ## test_suite_name, test_name )
#define ARIES_UNIT_TEST_CLASS( test_suite_name ) class UT_ ## test_suite_name : public ::testing::Test

namespace aries_test
{
// 用于快速生成 AriesCommonExpr
aries_engine::AriesCommonExprUPtr CreateConstantExpression( const AriesExprContent& value );

// 比较两个 sql 的结果，要求两个结果完全相同
void AssertTwoSQLS( const std::string& sql1, const std::string sql2, const std::string& dbName , bool requireSuccess = true);

aries_engine::AriesTableBlockUPtr ExecuteSQL( const std::string& sql, const std::string& dbName );
aries_engine::AriesTableBlockUPtr ExecuteSQLFromFile( const std::string& sqlFile, const std::string& dbName );
void InitTable( const string& dbName, const string& tableName );

}