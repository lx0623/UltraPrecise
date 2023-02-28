#include "AriesUpdateCalcNode.h"
#include "AriesCalcTreeGenerator.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

AriesUpdateCalcNode::AriesUpdateCalcNode(): AriesOpNode()
{

}

AriesUpdateCalcNode::~AriesUpdateCalcNode()
{
}

void AriesUpdateCalcNode::SetCalcExprs( const std::vector< AriesCommonExprUPtr >& exprs )
{
    m_exprs.clear();
    AriesCalcTreeGenerator generator;
    for ( const auto& expr : exprs )
    {
        if ( !expr->IsLiteralValue() )
        {
            m_exprs.emplace_back( generator.ConvertToCalcTree( expr, m_nodeId ) );
        }
        else
        {
            m_exprs.emplace_back( nullptr );
        }
        m_originExprs.emplace_back( expr->Clone() );
    }
}

void AriesUpdateCalcNode::SetColumnIds( const std::vector< int >& columnIds )
{
    m_columnIds.assign( columnIds.cbegin(), columnIds.cend() );
}

bool AriesUpdateCalcNode::Open()
{
    return m_dataSource->Open();
}

AriesOpResult AriesUpdateCalcNode::GetNext()
{
    AriesOpResult cachedResult = GetCachedResult();
    if ( AriesOpNodeStatus::END == cachedResult.Status )
        return cachedResult;
    ARIES_ASSERT( m_columnIds.size() > 0, "column ids' count should not be zero" );
    ARIES_ASSERT( m_exprs.size() == m_columnIds.size(), "exprs' count must be equal with column ids' count" );
    auto data = m_dataSource->GetNext();
    if ( data.Status == AriesOpNodeStatus::ERROR )
    {
        return data;
    }

    if ( data.TableBlock == nullptr )
    {
        if ( data.Status != AriesOpNodeStatus::END )
        {
            return { AriesOpNodeStatus::ERROR, nullptr };
        }
        else
        {
            return data;
        }
    }
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif
    auto& table = data.TableBlock;

    if ( table )
        m_rowCount += table->GetRowCount();

    auto result_table = std::make_unique< AriesTableBlock >();
    for ( std::size_t i = 0; i < m_exprs.size(); i++ )
    {
        const auto& expr = m_exprs[ i ];
        const auto& id = m_columnIds[ i ];
        AriesColumnSPtr column = std::make_shared< AriesColumn >();
        if ( expr )
        {
            auto r = expr->Process( table );
            if ( r.type() == typeid( AriesDataBufferSPtr ) )
            {
                auto calced = boost::get< AriesDataBufferSPtr >( r );
                auto buffer = table->GetColumnBuffer( id );
                column->AddDataBuffer( calced );
                table->UpdateColumn( id, column );
            }
        }
        else
        {
            auto buffer = aries_acc::CreateDataBufferWithLiteralExpr( m_originExprs[ i ], table->GetRowCount() );
            column->AddDataBuffer( buffer );
        }

        result_table->AddColumn( i + 1, column );
    }
#ifdef ARIES_PROFILE
    m_opTime += t.end();
#endif
    CacheNodeData( result_table );

    return { data.Status, std::move( result_table ) };
}

void AriesUpdateCalcNode::Close()
{
    m_dataSource->Close();
}
JSON AriesUpdateCalcNode::GetProfile() const
{
    JSON stat = this->AriesOpNode::GetProfile();
    stat["type"] = "AriesUpdateCalcNode";
    return stat;
}

END_ARIES_ENGINE_NAMESPACE
