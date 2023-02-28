#pragma once

#include <mutex>
#include <condition_variable>


#include "frontend/SQLTreeNode.h"
#include "AriesDataDef.h"


BEGIN_ARIES_ENGINE_NAMESPACE
using SpoolId = int;
using SpoolReferenceCount = int;

class AriesSpoolCacheManager
{
public:
    AriesSpoolCacheManager();
    ~AriesSpoolCacheManager();

    AriesTableBlockUPtr getCachedNodeData( int spoolId );

    void cacheNodeData( int spoolId, const AriesTableBlockUPtr& tableBlock );

    void increaseSpoolReferenceCount( int spoolId );

    void decreaseSpoolReferenceCount( int spoolId );

    void cleanCacheNode();

    void buildSpoolChildrenMap( SQLTreeNodePointer treeNode );

private:
    map< SpoolId, AriesTableBlockUPtr > m_spoolMap;
    map< SpoolId, SpoolReferenceCount > m_spoolReferenceMap;
    map< SpoolId, vector< SpoolId > > m_spoolChildrenMap;

    std::mutex m_mutex;
    std::condition_variable m_conditionVariable;
    std::map< SpoolId, bool > m_isCachingMap;
};

using AriesSpoolCacheManagerSPtr = std::shared_ptr< AriesSpoolCacheManager >;

END_ARIES_ENGINE_NAMESPACE
