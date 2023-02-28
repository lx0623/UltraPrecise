#include "TestUtils.h"

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

using namespace aries;
using namespace aries_engine;

namespace aries_test
{

aries_engine::AriesCommonExprUPtr CreateConstantExpression( const AriesExprContent& value )
{
    AriesExprType type = AriesExprType::INTEGER;
    AriesColumnType column_type;
    if ( value.type() == typeid( int8_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::INT8;
    }
    else if ( value.type() == typeid( int16_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::INT16;
    }
    else if ( value.type() == typeid( int32_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::INT32;
    }
    else if ( value.type() == typeid( int64_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::INT64;
    }
    else if ( value.type() == typeid( uint8_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::UINT8;
    }
    else if ( value.type() == typeid( uint16_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::UINT16;
    }
    else if ( value.type() == typeid( uint32_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::UINT32;
    }
    else if ( value.type() == typeid( uint64_t ) )
    {
        type = AriesExprType::INTEGER;
        column_type.DataType.ValueType = AriesValueType::UINT64;
    }
    else if ( value.type() == typeid( std::string ) )
    {
        type = AriesExprType::STRING;
        column_type.DataType.ValueType = AriesValueType::CHAR;
        column_type.DataType.Length = boost::get< std::string >( value ).size();
    }
    else if ( value.type() == typeid( aries_acc::Decimal ) )
    {
        type = AriesExprType::DECIMAL;
        column_type.DataType.ValueType = AriesValueType::DECIMAL;
    }
    else if ( value.type() == typeid( float ) )
    {
        type = AriesExprType::FLOATING;
        column_type.DataType.ValueType = AriesValueType::FLOAT;
    }
    else if ( value.type() == typeid( double ) )
    {
        type = AriesExprType::FLOATING;
        column_type.DataType.ValueType = AriesValueType::DOUBLE;
    }
    else if ( value.type() == typeid( bool ) )
    {
        type = AriesExprType::TRUE_FALSE;
        column_type.DataType.ValueType = AriesValueType::BOOL;
    }
    else if ( value.type() == typeid( aries_acc::AriesDate ) )
    {
        type = AriesExprType::DATE;
        column_type.DataType.ValueType = AriesValueType::DATE;
    }
    else if ( value.type() == typeid( aries_acc::AriesTime ) )
    {
        type = AriesExprType::TIME;
        column_type.DataType.ValueType = AriesValueType::TIME;
    }
    else if ( value.type() == typeid( aries_acc::AriesDatetime ) )
    {
        type = AriesExprType::DATE_TIME;
        column_type.DataType.ValueType = AriesValueType::DATETIME;
    }
    else if ( value.type() == typeid( aries_acc::AriesDatetime ) )
    {
        type = AriesExprType::DATE_TIME;
        column_type.DataType.ValueType = AriesValueType::DATETIME;
    }
    else if ( value.type() == typeid( aries_acc::AriesTimestamp ) )
    {
        type = AriesExprType::TIMESTAMP;
        column_type.DataType.ValueType = AriesValueType::TIMESTAMP;
    }
    else if ( value.type() == typeid( aries_acc::AriesYear ) )
    {
        type = AriesExprType::YEAR;
        column_type.DataType.ValueType = AriesValueType::YEAR;
    }
    else if ( value.type() == typeid( aries_acc::CompactDecimal ) )
    {
        type = AriesExprType::DECIMAL;
        column_type.DataType.ValueType = AriesValueType::COMPACT_DECIMAL;
    }
    else
    {
        assert( 0 );
    }

    return aries_engine::AriesCommonExpr::Create( type, value, column_type );
}

static void compare_two_results( const aries::SQLResultPtr& left, const aries::SQLResultPtr& right ) {
    ASSERT_EQ( left->IsSuccess(), right->IsSuccess() );

    const auto& left_results = left->GetResults();
    const auto& right_results = right->GetResults();

    ASSERT_EQ( left_results.size(), right_results.size() );

    for ( size_t i = 0; i < left_results.size(); i++ ) {
        const auto& left_result = left_results[ i ];
        const auto& right_result = right_results[ i ];

        auto left_buffer = ( AriesMemTable* )( left_result.get() );
        auto right_buffer = ( AriesMemTable* )( right_result.get() );

        auto left_table = left_buffer->GetContent();
        auto right_table = right_buffer->GetContent();

        ASSERT_EQ( left_table->GetRowCount(), right_table->GetRowCount() );
        ASSERT_EQ( left_table->GetColumnCount(), right_table->GetColumnCount() );

        std::vector< aries_acc::AriesDataBufferSPtr > left_columns;
        std::vector< aries_acc::AriesDataBufferSPtr > right_columns;

        for ( int i = 1; i <= left_table->GetColumnCount(); i++ ) {
            left_columns.emplace_back( left_table->GetColumnBuffer( i ) );
            right_columns.emplace_back( right_table->GetColumnBuffer( i ) );
        }

        for ( int i = 0; i < left_table->GetRowCount(); i++ ) {
            for ( size_t j = 0; j < left_columns.size(); j++ ) {
                auto l = left_columns[ j ]->GetString( i );
                auto r = right_columns[ j ]->GetString( i );
                ASSERT_EQ(l, r);
            }
        }
    }
}

void AssertTwoSQLS( const std::string& sql1, const std::string sql2, const std::string& db_name , bool requireSuccess )
{
    auto result1 = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql1, db_name );
    auto result2 = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql2, db_name );

    if ( requireSuccess )
    {
        ASSERT_TRUE( result1->IsSuccess() );
        ASSERT_TRUE( result2->IsSuccess() );
    }

    compare_two_results( result1, result2 );
}

AriesTableBlockUPtr ExecuteSQL( const std::string& sql, const std::string& dbName )
{
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    EXPECT_TRUE( result->IsSuccess() );

    if ( result->GetResults().empty() )
    {
        return nullptr;
    }
    return std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
}
AriesTableBlockUPtr ExecuteSQLFromFile( const std::string& sqlFile, const std::string& dbName )
{
    auto result = SQLExecutor::GetInstance()->ExecuteSQLFromFile( sqlFile, dbName );
    EXPECT_TRUE( result->IsSuccess() );

    if ( result->GetResults().empty() )
    {
        return nullptr;
    }
    return std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
}

void InitTable( const string& dbName, const string& tableName )
{
    // string sql = "drop database if exists " + dbName + ";";
    // auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    // ASSERT_TRUE( result->IsSuccess() );
    string sql = "create database if not exists " + dbName + ";";
    auto result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );

    sql = "drop table if exists " + tableName + ";";
    result = aries::SQLExecutor::GetInstance()->ExecuteSQL( sql, dbName );
    ASSERT_TRUE( result->IsSuccess() );
}

} // namespace aries_test