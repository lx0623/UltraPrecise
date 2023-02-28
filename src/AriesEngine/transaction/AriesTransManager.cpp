#include "AriesTransManager.h"
#include "AriesMvccTableManager.h"
#include "AriesVacuum.h"
BEGIN_ARIES_ENGINE_NAMESPACE
    AriesTransManager AriesTransManager::m_instance;

    AriesTransManager::AriesTransManager()
    {
        m_nextTxId = START_TX_ID;
    }

    AriesTransactionPtr AriesTransManager::NewTransaction( bool isDDL )
    {
        // currently, we throw exception if vacuum is in progress, otherwisze return immediately with the current db version.
        int currentDbVersion = AriesVacuum::GetInstance().WaitVacuumDone();

        std::lock_guard< std::mutex > lock( m_mutexForTxMap );
        TxId txId = m_nextTxId++;
        AriesTransactionPtr tx = make_shared< AriesTransaction >( txId, currentDbVersion, isDDL );
        m_ipTxsMap[ txId ] = tx;
        m_txInfoMap.insert( std::make_pair( txId, tx->GetTxInfo() ) );
        tx->GetTxInfo()->SetSnapshot( TakeSnapshot() );

        LOG( INFO ) << "new transaction " << txId;

        return tx;
    }

    void AriesTransManager::AddTx( TxId txId, TransactionStatus status )
    {
        std::lock_guard< std::mutex > lock( m_mutexForTxMap );
        m_txInfoMap.insert( std::make_pair( txId, make_shared< TransactionInfo >( txId, status ) ) );
    }

    TransactionStatus AriesTransManager::GetTxStatus( TxId txId )
    {
        return GetTxInfo( txId )->GetStatus();
    }

    TransactionInfoSPtr AriesTransManager::GetTxInfo( TxId txId )
    {
        std::lock_guard< std::mutex > lock( m_mutexForTxMap );
        auto iter = m_txInfoMap.find( txId );
        assert( iter != m_txInfoMap.end() );

        return iter->second;
    }

    Snapshot AriesTransManager::GetTxSnapshot( TxId txId )
    {
        return GetTxInfo( txId )->GetSnapshot();
    }

    Snapshot AriesTransManager::TakeSnapshot()
    {
        // when taking snapshop, should at least exist one active transaction
        assert( !m_ipTxsMap.empty() );
        Snapshot snapshot;
        snapshot.xMin = m_ipTxsMap.cbegin()->first;
        snapshot.xMax = m_xMax;
        for ( auto it : m_ipTxsMap )
        {
            if ( it.second->GetTxId() >= m_xMax )
                break;
            snapshot.xInProgress.push_back( it.second->GetTxId() );
        }

        return snapshot;
    }

    void AriesTransManager::EndTransaction( AriesTransactionPtr tx, TransactionStatus status )
    {
        TxId txId = tx->GetTxId();
        int32_t nextTxId = -1;
        {
            std::lock_guard< std::mutex > lock( m_mutexForTxMap );
            m_ipTxsMap.erase( txId );

            if ( txId >= m_xMax )
                m_xMax = txId + 1;

            nextTxId = m_nextTxId;
        }

        if ( status == TransactionStatus::COMMITTED )
        {
            auto modifiedTables = tx->GetModifiedTables();
            if( !modifiedTables.empty() )
            {
                string msg;
                for ( auto tableId : modifiedTables )
                    msg.append( std::to_string( tableId ) ).append( ", ");

                LOG( INFO ) << "Transaction ( " << txId << ", " << nextTxId << " ) modified tables: " << msg;
                AriesMvccTableManager::GetInstance().updateCaches( txId, nextTxId, modifiedTables );
            }
        }
        tx->End( status );
        // will notify all waiters, currently only vacuum thread might wait for this
        NotifyTxEndOrIdle();
    }

    void AriesTransManager::NotifyTxEndOrIdle()
    {
        m_statusCond.notify_all();
    }

    void AriesTransManager::WaitForAllTxEndOrIdle()
    {
        assert( AriesVacuum::GetInstance().IsMySelfVacuumThread() );
        unique_lock<mutex> lock(m_mutexForTxMap);
        m_statusCond.wait(lock,[&]
        {
            bool bExitWait = true;
            for( auto it : m_ipTxsMap )
            {
                // only wait for the active transactions, ignore other ones which is waiting for io.
                // m_server_idle is true if the thread is waiting io.
                if( !it.second->GetMyTHD()->m_server_idle )
                {
                    bExitWait = false;
                    break;
                }
            }
            return bExitWait;
        });
    }

    TxId AriesTransManager::GetFreezeTxId()
    {
        std::lock_guard< std::mutex > lock( m_mutexForTxMap );
        if ( m_ipTxsMap.empty() )
        {
            //当前没有正在运行的tx,返回系统下一个即将分配出去的tx的id
            return m_nextTxId;
        }
        TxId minTxId = INT32_MAX;
        for ( auto it : m_ipTxsMap )
        {
            auto snapshot = it.second->GetSnapshot();
            if ( snapshot.xMin < minTxId )
                minTxId = snapshot.xMin;
        }
        return minTxId;
    }

    void AriesTransManager::RemoveCommitLog( TxId txId )
    {
        std::lock_guard< std::mutex > lock( m_mutexForTxMap );
        auto it = m_txInfoMap.lower_bound( txId );
        if ( m_txInfoMap.end() != it )
        {
            m_txInfoMap.erase( m_txInfoMap.begin(), it );
        }
    }

END_ARIES_ENGINE_NAMESPACE
