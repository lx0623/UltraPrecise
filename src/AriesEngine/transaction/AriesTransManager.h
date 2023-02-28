#pragma once

#include <map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <exception>

#include "AriesDefinition.h"
#include "AriesAssert.h"
#include "AriesTransaction.h"

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesTransManager
{
public:
    static bool Init( TxId nextTxId )
    {
        m_instance.m_nextTxId = nextTxId;
        m_instance.m_xMin = m_instance.m_xMax = nextTxId;
        m_instance.m_txInfoMap.clear();
        m_instance.m_ipTxsMap.clear();
        return true;
    }
    static AriesTransManager& GetInstance()
    {
        return m_instance;
    }

    // add status for a already commited or aborted transaction.
    // 仅供测试使用
    void AddTx( TxId txId, TransactionStatus status );

    /**
     * @brief create a new transaction and add to the in_progress_transaction map.
     * Transacton status is set to IN_PROGRESS.
     *
     * @return 
     *   a Transaction with valid txId for success, or with INVALID_TX_ID for faliure
     */
    AriesTransactionPtr NewTransaction( bool isDDL = false );

    TransactionStatus GetTxStatus( TxId txId );
    
    Snapshot GetTxSnapshot( TxId txId );

    /**
     * @brief End a transaction
     * call this function to commit or abort a transaction,
     * delete it from the in progress transaction map and keep its status in all transaction status map.
     *
     * @param txId the transaction to end
     * @param status transaction status, commit or abort
     */
    void EndTransaction( AriesTransactionPtr tx, TransactionStatus status );

    /** 
     * @brief wait all tx end except the ones waiting io. this function should only be called in vaccum thread.
     */
    void WaitForAllTxEndOrIdle();

    void NotifyTxEndOrIdle();

    TxId GetFreezeTxId();

    //删除所有 < 输入参数id的对应transacation状态
    //仅供测试用例使用
    void RemoveCommitLog( TxId txId );

    //仅供测试用例使用
    TxId GetNextTxId()
    {
        return m_nextTxId;
    }

    void Clear()
    {
        std::lock_guard< std::mutex > lock( m_mutexForTxMap );
        m_txInfoMap.clear();
        m_ipTxsMap.clear();
    }

private:
    AriesTransManager();
    Snapshot TakeSnapshot();
    TransactionInfoSPtr GetTxInfo( TxId txId );

private:
    static AriesTransManager m_instance;

    /// the next transaction ID to allocate
    atomic<int32_t> m_nextTxId;

    /// contain all transactions.
    std::map< TxId, TransactionInfoSPtr > m_txInfoMap;

    /// contain only currently in progress transactions.
    std::map<TxId, AriesTransactionPtr> m_ipTxsMap;

    TxId m_xMin = START_TX_ID;
    TxId m_xMax = START_TX_ID; //!< 当前时刻结束的最大transaction

    // 保护 m_txStatusMap 和 m_ipTxsMap
    // 在做缓存、扫描mvcc table和transaction结束的过程中,
    // 会同时用到此锁和 AriesMvccTableManager 中的 m_mutex4CacheMap 锁，
    // 为了防止死锁，加锁顺序为： [ m_mutex4CacheMap, m_mutexForTxMap ]
    std::mutex m_mutexForTxMap;

    condition_variable m_statusCond;
};

END_ARIES_ENGINE_NAMESPACE