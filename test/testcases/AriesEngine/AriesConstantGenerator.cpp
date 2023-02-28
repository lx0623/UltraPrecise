#include "AriesConstantGenerator.h"
#include "../../TestUtils.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace aries_test;

std::shared_ptr< aries_engine::AriesConstantNode > GenerateConstNode(  const string& dbName, const string& tableName, const vector< int >& ids )
{
    auto node = std::make_shared< AriesConstantNode >( dbName, tableName );

    // auto expr1 = AriesCommonExpr::Create( AriesExprType::INTEGER, 1, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto expr1 = CreateConstantExpression( int32_t( 1 ) );
    auto expr2 = CreateConstantExpression( std::string( "abc" ) );
    auto expr3 = CreateConstantExpression( AriesDate( 2019, 10, 10 ) );
    auto expr4 = CreateConstantExpression( int32_t( 2 ) );
    auto expr5 = CreateConstantExpression( std::string( "efg" ) );
    auto expr6 = CreateConstantExpression( AriesDate( 2019, 10, 12 ) );
    std::vector< std::vector< AriesCommonExprUPtr > > data;

    std::vector< AriesCommonExprUPtr > exprs;
    exprs.emplace_back( std::move( expr1 ) );
    exprs.emplace_back( std::move( expr2 ) );
    exprs.emplace_back( std::move( expr3 ) );

    data.emplace_back( std::move( exprs ) );

    exprs.emplace_back( std::move( expr4 ) );
    exprs.emplace_back( std::move( expr5 ) );
    exprs.emplace_back( std::move( expr6 ) );
    data.emplace_back( std::move( exprs ) );

    string errorMsg;
    node->SetColumnData( data, ids, errorMsg );

    return node;
}
