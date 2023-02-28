/*!
@file AriesMvccTable.h
@brief 管理 mvcc table对象，以及mvccTable对象对应的数据
*/
#pragma once

#include "AriesMvccTable.h"

#include "schema/SchemaManager.h"

using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE
/*!
@brief 缓存 TxId 可见的表数据. tx id 在[ m_xMin, m_xMax )范围内的transaction，可以使用此cache
*/
struct AriesMvccTableCache
{
    AriesMvccTableCache( TxId txId, TxId txMin, TxId txMax, AriesTableBlockUPtr& table )
        : m_cacheTxId( txId ),
          m_xMin( txMin ),
          m_xMax( txMax ),
          m_hitCount( 0 ),
          m_table( std::move( table ) )
    {
    }

    TxId m_cacheTxId; //!< 创建此cache的 tx ID
    TxId m_xMin; ///<mvcc 意义下可以看到这个cache 的最小trasactoin ID. 创建此cache时，取AriesMvccTableManager::m_tableTxMaxMap对应的值
    TxId m_xMax; ///<mvcc 意义下可以看到这个cache 的最大trasactoin ID.
    uint64_t m_hitCount; ///< 缓存命引用数，当引用数归零时可以清除这个cache
    AriesTableBlockUPtr m_table;
};

using AriesMvccTableCacheSPtr = std::shared_ptr< AriesMvccTableCache >;

class AriesMvccTableManager {

public:
    static AriesMvccTableManager & GetInstance() {
        static AriesMvccTableManager instance;
        return instance;
    }

    /*！
    @brief 根据db name和table name获取 MVCC table
    */
    AriesMvccTableSPtr getMvccTable(const string &dbName, const string &tableName) {
        auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName )->GetId();
        lock_guard< mutex > lock( m_mutex4TableMap );
        AriesMvccTableSPtr tableSPtr = m_mvccTableMap[tableId];
        if (tableSPtr == nullptr) {
            tableSPtr = m_mvccTableMap[tableId] = make_shared< AriesMvccTable >( dbName, tableName );
        }
        return tableSPtr;
    }

    void resetInitTableOfMvccTable(const string &dbName, const string &tableName);

    void removeMvccTable(const string &dbName, const string &tableName) {
        auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName )->GetId();
        lock_guard< mutex > lock( m_mutex4TableMap );
        m_mvccTableMap[tableId] = nullptr;
    }

    /*!
    @brief 
     - 清理 mvccTable对象
     - 清理 缓存的mvcc数据
    */
    void clearAll()
    {
        {
            lock_guard< mutex > lock( m_mutex4TableMap );
            m_mvccTableMap.clear();
        }

        {
            lock_guard< mutex > lock( m_mutex4CacheMap );
            m_cache.clear();
            m_tableTxMaxMap.clear();
        }
    }

    /*!
    @brief 获取table中colsId对应列的数据
    */
    AriesTableBlockUPtr getTable(
        const AriesTransactionPtr &transaction,
        const string &dbName,
        const string &tableName,
        const vector< int > &colsId,
        const vector< AriesCommonExprUPtr > &partitionConditions = {} );

    void deleteCache(const string &dbName, const string &tableName);

    //仅供测试使用
    AriesMvccTableCacheSPtr getCache( const TableId tableId ) const;

    TxId getTableTxMax( const TableId tableId );

    // 修改表的transaction commit时调用
    // txId: 调用者的tx ID
    // txMax: transaction commit时，系统中下一个待分配的tx ID
    void updateCaches( TxId txId, TxId txMax, const std::set< TableId >& modifiedTables );

private:
    AriesTableBlockUPtr
    getAndCacheTableColumns( const AriesTransactionPtr &transaction,
                             const string& dbName,
                             const string& tableName,
                             const vector< int >& colIds,
                             bool createNewCache,
                             const vector< AriesCommonExprUPtr > &partitionConditions = {} );

private:
    map<TableId, AriesMvccTableSPtr> m_mvccTableMap;
    mutex m_mutex4TableMap;

    /*!
    - key: table ID,
    - value: txId,
       +  txId: 修改此表的transaction commit时， 系统中下一个尚未分配的tx ID
    */
    unordered_map< TableId, TxId > m_tableTxMaxMap;
    unordered_map< TableId, AriesMvccTableCacheSPtr > m_cache;
    mutex m_mutex4CacheMap;///<保护 m_cache 和 m_tableTxMaxMap
};

END_ARIES_ENGINE_NAMESPACE