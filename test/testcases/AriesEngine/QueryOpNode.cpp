#include "QueryOpNode.h"

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

using namespace aries;
using namespace aries_engine;

namespace aries_test
{

QueryOpNode::QueryOpNode( const std::string& sql, const std::string& dbName ) : m_sql( sql ), m_dbName( dbName )
{
}

bool QueryOpNode::Open()
{
    auto result = SQLExecutor::GetInstance()->ExecuteSQL( m_sql, m_dbName );

    if ( !result->IsSuccess() )
    {
        return false;
    }

    if ( result->GetResults().size() != 1 )
    {
        return false;
    }

    auto mem_table = std::dynamic_pointer_cast< AriesMemTable >( result->GetResults()[ 0 ] );
    m_table = std::move( mem_table->GetContent() );
    return true;
}

void QueryOpNode::Close()
{
}

AriesOpResult QueryOpNode::GetNext()
{
    if ( m_table )
    {
        return { AriesOpNodeStatus::END, std::move( m_table ) };
        m_table = nullptr;
    }
    else
    {
        return { AriesOpNodeStatus::END, nullptr };
    }
}

};