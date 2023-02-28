#include "CpuTimer.h"
#include "AriesMvccTableManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

TxId AriesMvccTableManager::getTableTxMax( const TableId tableId )
{
    TxId txId = INVALID_TX_ID;
    auto it = m_tableTxMaxMap.find( tableId );
    if ( m_tableTxMaxMap.end() != it )
        txId = it->second;
    return txId;
}

void AriesMvccTableManager::updateCaches( TxId txId, TxId txMax, const std::set< TableId >& modifiedTables )
{
    lock_guard< mutex > lock( m_mutex4CacheMap );
    for ( auto tableId : modifiedTables )
        m_tableTxMaxMap[ tableId ] = txMax;

    for ( auto tableId : modifiedTables )
    {
        auto it = m_cache.find( tableId );
        if ( m_cache.cend() != it )
        {
            LOG( INFO ) << "tx " << txId << ", cache info for table " << tableId << ": "
                        << it->second->m_cacheTxId << ", [" << it->second->m_xMin << ", " << it->second->m_xMax << ")";
            if ( txMax < it->second->m_xMax )
            {
                it->second->m_xMax = txMax;
                LOG( INFO ) << "tx " << txId << " updated cache info for table " << tableId
                            << " to [" << it->second->m_xMin << ", " << txMax << ")";
            }
        }
    }
}

AriesTableBlockUPtr AriesMvccTableManager::getTable(
    const AriesTransactionPtr &transaction,
    const string &dbName,
    const string &tableName,
    const vector< int > &colIds,
    const vector< AriesCommonExprUPtr > &partitionConditions )
{
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( dbName, tableName );
    auto thisTxId = transaction->GetTxId();

    // the transaction itself has modified this table,
    // ignore the cache, do mvcc scan directly
    if ( transaction->IsModifyingTable( tableId ) || transaction->IsTableModified( tableId ) )
    {
        AriesMvccTableSPtr tableSPtr = getMvccTable( dbName, tableName );
        auto resultTable = tableSPtr->GetTable( transaction, colIds, partitionConditions );
#ifdef ARIES_PROFILE
        LOG( INFO ) << "AriesMvccTableManager::getTable time ( ignore cache ): " << t.end();
#endif
        return resultTable;
    }

    AriesTableBlockUPtr resultTable;

    unique_lock< mutex > lock( m_mutex4CacheMap );

    LOG( INFO ) << "tx " << thisTxId << ", get table " << tableId << "(" << dbName << "." << tableName << ")";

    auto it = m_cache.find( tableId );
    if ( m_cache.cend() != it ) // 这里是查看 cache 中是否存在了 需要的数据
    {
        auto cacheTxMin = it->second->m_xMin;
        auto cacheTxMax = it->second->m_xMax;

        LOG( INFO ) << "tx " << thisTxId << ", cache info for table " << tableId << "(" << dbName + "." + tableName << "): "
                    << it->second->m_cacheTxId << ", [" << cacheTxMin << ", " << cacheTxMax << ")";

        auto& cachedTable = it->second->m_table;
        if ( thisTxId < cacheTxMin || thisTxId >= cacheTxMax )
        {
            // cache no usable for this tx
            bool createNewCache = true;
            if ( thisTxId < cacheTxMin )
                createNewCache = false;
            LOG( INFO ) << "tx " << thisTxId << ", cache not usable for table " << tableId << "(" << dbName + "." + tableName << "), "
                        << ( createNewCache ? "" : "not ") << "create new cache";
            resultTable = getAndCacheTableColumns( transaction, dbName, tableName, colIds, createNewCache );
        #ifdef ARIES_PROFILE
            LOG( INFO ) << "AriesMvccTableManager::getTable time: " << t.end();
        #endif
            return resultTable;
        }

        LOG( INFO ) << "tx " << thisTxId << ", cache hit for table " << tableId << "(" << dbName + "." + tableName << ")";
        std::map< int32_t, int32_t > cacheColIdsMap;
        std::map< int32_t, int32_t > outputColIdsMap;

        vector< int32_t > colIdsToGet;
        int32_t outputColId = 1;
        int32_t cacheColIdx = 1;

        for ( auto colId : colIds )
        {
            bool columnExists = cachedTable->ColumnExists( colId );
            if ( !columnExists )
            {
                LOG( INFO ) << "tx " << thisTxId << ", get mvcc column " << colId;
                colIdsToGet.push_back( colId );
                cacheColIdsMap[ colId ] = cacheColIdx++;
            }

            outputColIdsMap[ outputColId++ ] = colId;
        }

        if ( !colIdsToGet.empty() )
        {
            AriesMvccTableSPtr mvccTableSPtr = getMvccTable( dbName, tableName );
            auto tmpTable = mvccTableSPtr->GetTable( transaction, colIdsToGet );
            tmpTable->UpdateColumnIds( cacheColIdsMap );
            tmpTable->MaterilizeAllDataBlocks();
            cachedTable->MergeTable( std::move( tmpTable ) );
        }
        if ( colIdsToGet.size() < colIds.size() )
            it->second->m_hitCount++;

        resultTable = cachedTable->MakeTableByColumns( colIds, false );
        resultTable->UpdateColumnIds( outputColIdsMap );
    }
    else
    {
        LOG( INFO ) << "tx " << thisTxId << ", no cache for table " << tableId << "(" << dbName + "." + tableName << ")";
        resultTable = getAndCacheTableColumns( transaction, dbName, tableName, colIds, true, partitionConditions ); // 获取数据 并将它 存放到 cache table 中
    }

#ifdef ARIES_PROFILE
    LOG( INFO ) << "AriesMvccTableManager::getTable time: " << t.end();
#endif
    return resultTable;
}

AriesTableBlockUPtr
AriesMvccTableManager::getAndCacheTableColumns( const AriesTransactionPtr &transaction,
                                                const string& dbName,
                                                const string& tableName,
                                                const vector< int >& colIds,
                                                bool createNewCache,
                                                const vector< AriesCommonExprUPtr > &partitionConditions )
{
    std::map< int32_t, int32_t > cacheColIdsMap;
    std::map< int32_t, int32_t > outputColIdsMap;

    int32_t outputColId = 1;
    int32_t cacheColIdx = 1;

    auto thisTxId = transaction->GetTxId();
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( dbName, tableName );

    for ( auto colId : colIds )
    {
        LOG( INFO ) << "get mvcc column " << colId;
        cacheColIdsMap[ colId ] = cacheColIdx++;
        outputColIdsMap[ outputColId++ ] = colId;
    }
    AriesMvccTableSPtr mvccTableSPtr = getMvccTable( dbName, tableName );
    auto tmpTable = mvccTableSPtr->GetTable( transaction, colIds, partitionConditions );

    tmpTable->UpdateColumnIds( cacheColIdsMap );
    tmpTable->MaterilizeAllDataBlocks();

    auto resultTable = tmpTable->MakeTableByColumns( colIds, false );
    resultTable->UpdateColumnIds( outputColIdsMap );

    AriesMvccTableCacheSPtr cacheEntry;
    TxId txMin, txMax;
    auto tableTxMax = getTableTxMax( tableId );
    LOG( INFO ) << "table txMax: " << tableTxMax;
    // table没有被修改过
    if ( INVALID_TX_ID == tableTxMax )
    {
        txMin = START_TX_ID;
        txMax = INT32_MAX;
    }
    else
    {
        // 对此表的修改，对于此transaction不可见
        if ( thisTxId < tableTxMax )
        {
            LOG( INFO ) << "table txMax " << tableTxMax << " is INVISIBLE to this tx " << thisTxId;
            createNewCache = false;
            // txMin = START_TX_ID;
            // txMax = tableTxMax - 1;
        }
        else
        {
            LOG( INFO ) << "table txMax " << tableTxMax << " is VISIBLE to this tx " << thisTxId;
            txMin = tableTxMax;
            txMax = INT32_MAX;
        }
    }

    if ( createNewCache )   // 需要创建新的数据 createNewCache
    {
        cacheEntry = make_shared< AriesMvccTableCache >( thisTxId, txMin, txMax, tmpTable );
        m_cache[ tableId ] = cacheEntry;
        LOG( INFO ) << "tx " << thisTxId << " created cache for table " << tableId << "(" << dbName + "." + tableName << ")" << ": [ " << txMin << ", " << txMax << " )";
    }
    return resultTable;
}

void AriesMvccTableManager::deleteCache(const string &dbName, const string &tableName)
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetTableId( dbName, tableName );
    lock_guard< mutex > lock( m_mutex4CacheMap );
    m_cache.erase( tableId );
}

AriesMvccTableCacheSPtr AriesMvccTableManager::getCache( const TableId tableId ) const
{
    auto it = m_cache.find( tableId );
    if ( m_cache.cend() != it )
        return it->second;
    return nullptr;
}

void AriesMvccTableManager::resetInitTableOfMvccTable(const string &dbName, const string &tableName)
{
    auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName )->GetId();
    lock_guard< mutex > lock( m_mutex4TableMap );
    AriesMvccTableSPtr tableSPtr = m_mvccTableMap[tableId];
    if ( tableSPtr )
    {
        tableSPtr->ResetInitTable();
    }
}
END_ARIES_ENGINE_NAMESPACE