#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

AriesSpoolCacheManager::AriesSpoolCacheManager()
{

}

AriesSpoolCacheManager::~AriesSpoolCacheManager()
{
    cleanCacheNode();
}

AriesTableBlockUPtr AriesSpoolCacheManager::getCachedNodeData(int spoolId)
{
    if(spoolId == -1){
        return nullptr;
    }

    std::unique_lock< std::mutex > lock( m_mutex );
    auto &ret = m_spoolMap[spoolId];
    while ( ret.get() == 0 )
    {
        if ( !m_isCachingMap[ spoolId ] )
        {
            m_isCachingMap[ spoolId ] = true;
            return nullptr;
        }

        m_conditionVariable.wait( lock );
    }

    if ( m_spoolReferenceMap[spoolId] == 2 )
    {
        m_spoolReferenceMap[spoolId] = 0;
        return std::move( ret );
    }
    auto result = ret->Clone();
    decreaseSpoolReferenceCount(spoolId);
    return result;
}

void AriesSpoolCacheManager::cacheNodeData( int spoolId, const AriesTableBlockUPtr& tableBlock )
{
    if ( spoolId == -1 || !tableBlock )
    {
        return;
    }

    std::unique_lock< std::mutex > lock( m_mutex );
    if ( m_spoolReferenceMap[spoolId] > 1 )
    {
        auto table = tableBlock->Clone();
        if( m_spoolMap[ spoolId ] )
        {
            m_spoolMap[spoolId]->AddBlock( std::move( table ) );
        }
        else
        {
            m_spoolMap[ spoolId ] = std::move( table );
        }
        m_conditionVariable.notify_all();
    }
}

void AriesSpoolCacheManager::increaseSpoolReferenceCount(int spoolId)
{
    if (m_spoolReferenceMap[spoolId])
    {
        m_spoolReferenceMap[spoolId] = m_spoolReferenceMap[spoolId] + 1;
    }
    else
    {
        m_spoolReferenceMap[spoolId] = 1;
    }
}

void AriesSpoolCacheManager::decreaseSpoolReferenceCount(int spoolId)
{
    m_spoolReferenceMap[spoolId] = m_spoolReferenceMap[spoolId] - 1;
    if ( m_spoolChildrenMap[spoolId].size() > 0 )
    {
        for ( auto id : m_spoolChildrenMap[spoolId] ){
            m_spoolReferenceMap[id] = m_spoolReferenceMap[id] - 1;
        }
    }
    if (m_spoolReferenceMap[spoolId] <= 0)
    {
        m_spoolMap.erase(spoolId);
    }
}

void AriesSpoolCacheManager::cleanCacheNode()
{
    m_spoolMap.clear();
    m_spoolReferenceMap.clear();
    m_spoolChildrenMap.clear();
}

void AriesSpoolCacheManager::buildSpoolChildrenMap(SQLTreeNodePointer treeNode)
{
    if ( treeNode->GetSpoolId() != -1)
    {
        auto node = treeNode->GetParent();
        auto spoolId = treeNode->GetSpoolId();
        while ( node )
        {
            if ( node->GetSpoolId() != -1 ){
                m_spoolChildrenMap[node->GetSpoolId()].push_back(spoolId);
            }
            node = node->GetParent();
        }
    }
}

END_ARIES_ENGINE_NAMESPACE
