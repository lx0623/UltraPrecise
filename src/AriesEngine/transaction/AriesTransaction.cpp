#include "./AriesTransaction.h"
#include "AriesXLogManager.h"
#include "AriesMvccTableManager.h"
#include "utils/AriesDeviceManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    set< TxId > Snapshot::Diff( const Snapshot& snapshot ) const
    {
        set< TxId > result;
        auto txId = std::min( xMin, snapshot.xMin );
        auto diffCount = std::abs( xMin - snapshot.xMin );
        for ( int i = 0; i < diffCount; ++i )
        {
            result.insert( txId++ );
        }

        txId = std::min( xMax, snapshot.xMax );
        diffCount = std::abs( xMax - snapshot.xMax );
        for ( int i = 0; i < diffCount; ++i )
        {
            result.insert( txId++ );
        }

        vector< TxId > xIpDiff;
        std::set_symmetric_difference( xInProgress.cbegin(), xInProgress.cend(),
                                       snapshot.xInProgress.cbegin(), snapshot.xInProgress.cend(),
                                       std::back_inserter( xIpDiff ) );

        result.insert( xIpDiff.begin(), xIpDiff.end() );
        return result;
    }

    AriesTransaction::AriesTransaction( TxId argTxId, int currentDbVersion, bool isDDL )
    : m_currentDbVersion( currentDbVersion ), m_cmdId( 0 ), m_updateTableId( MAX_TABLE_ID ), mIsDDL(isDDL)
    {
        m_txInfo = make_shared< TransactionInfo >( argTxId, TransactionStatus::IN_PROGRESS );

        m_xLogWriter = AriesXLogManager::GetInstance().GetWriter( argTxId, isDDL );

        m_thd = current_thd;
    }

    AriesTransaction::~AriesTransaction()
    {
        AriesXLogManager::GetInstance().ReleaseWriter( GetTxId() );
    }

    bool AriesTransaction::IsTableModified( const TableId tableId ) const
    {
        bool modified = false;
        for ( auto tmpTableId : m_modifiedTables )
        {
            if ( tmpTableId == tableId )
            {
                modified = true;
                break;
            }
        }
        return modified;
    }

    void AriesTransaction::End( TransactionStatus status ) 
    {
        if ( status == TransactionStatus::COMMITTED && m_xLogWriter->Commit() )
        {
            m_txInfo->SetStatus( TransactionStatus::COMMITTED );
        }
        else
        {
            m_txInfo->SetStatus( TransactionStatus::ABORTED );
            m_xLogWriter->Abort();
        }
    }

END_ARIES_ENGINE_NAMESPACE
