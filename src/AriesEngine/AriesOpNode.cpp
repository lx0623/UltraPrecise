#include "AriesOpNode.h"
BEGIN_ARIES_ENGINE_NAMESPACE

AriesTableBlockUPtr AriesOpNode::GetEmptyTable() const
{
    if ( !m_outputColumnTypes.empty() )
    {
        return AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
    }
    else
    {
        return std::make_unique< AriesTableBlock >();
    }
}

JSON AriesOpNode::GetProfile() const
{
    JSON stats = {
        {"type", m_opName},
        {"param", m_opParam},
        {"time", m_opTime},
        {"memory", {JSON::parse(m_tableStats.ToJson(m_rowCount))} },
    };
    if(m_spoolId > -1)
            stats["memory"].push_back({{"spool_id", m_spoolId}});
    if(m_dataSource)
        stats["children"] = {m_dataSource->GetProfile()};
    return stats;
}

void AriesOpNode::SetSpoolCache( const int spoolId, const AriesSpoolCacheManagerSPtr& manager )
{
    m_spoolId = spoolId;
    m_spool_cache_manager = manager;
}

void AriesOpNode::CacheNodeData( const AriesTableBlockUPtr& tableBlock )
{
    if ( m_spool_cache_manager )
    {
        assert( m_spoolId > -1 );
        m_spool_cache_manager->cacheNodeData(m_spoolId, tableBlock );
    }
}

AriesOpResult AriesOpNode::GetCachedResult() const
{
    if ( m_spool_cache_manager )
    {
        assert( m_spoolId > -1 );
        AriesTableBlockUPtr tableBlock = m_spool_cache_manager->getCachedNodeData(m_spoolId);
        if (tableBlock)
            return AriesOpResult( AriesOpNodeStatus::END, std::move(tableBlock) );
    }
    
    return AriesOpResult( AriesOpNodeStatus::CONTINUE, nullptr );
}

void AriesOpNode::ReleaseData()
{
    if ( m_leftSource )
    {
        m_leftSource->ReleaseData();
    }

    if ( m_rightSource )
    {
        m_rightSource->ReleaseData();
    }
}

END_ARIES_ENGINE_NAMESPACE