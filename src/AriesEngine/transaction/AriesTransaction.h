#ifndef AIRES_TRANSACTION_H
#define AIRES_TRANSACTION_H

#include <vector>
#include <set>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "AriesDefinition.h"
#include "server/mysql/include/sql_class.h"

using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE

using TxId = int;
const TxId INVALID_TX_ID = 0;
const TxId BOOTSTRAP_TX_ID = 1;
const TxId FROZEN_TX_ID = 2;
const TxId START_TX_ID = 3;

using TableId = int64_t;
const TableId MAX_TABLE_ID = INT64_MAX;

enum class TransactionStatus: int8_t
{
    IN_PROGRESS = 0,
    COMMITTED,
    ABORTED,
};

enum ISOLATION_LEVEL
{
    REPEATABLE_READ
};

struct Snapshot
{
    TxId xMin; //!< 小于 xMin　的transaction均已结束
    TxId xMax; //!< 大于等于xMax 的transaction均在进行中
    std::vector< TxId > xInProgress;//!< 在trx启动时，[xMin,XMax)区间范围内且进行中的transaction列表

    /**
    *@brief 判断txId是否是活跃transaction
    *@details 
    *- used in visibility checks;
    *- 只要txId在本transaction启动时是active的，及时在调用IsTxActive 函数时 txId已经commit，对本transaction而言txId仍然是active的
    */
    bool IsTxActive( const TxId txId ) const
    {
        return txId >= xMax ||
               xInProgress.end() != std::find( xInProgress.begin(), xInProgress.end(), txId );
    }

    set< TxId > Diff( const Snapshot& snapshot ) const;
};

class AriesTransManager;
class AriesXLogWriter;

class TransactionInfo
{
public:
    TransactionInfo( TxId txId, TransactionStatus txStatus )
    : m_txId( txId ), m_txStatus( txStatus )
    {
    }

    TxId GetTxId() const 
    { 
        return m_txId; 
    }

    TransactionStatus GetStatus() 
    {
        return m_txStatus;
    }

    void SetStatus( TransactionStatus status )
    {
        m_txStatus = status;
    }

    void SetSnapshot( const Snapshot & snapshot )
    {
        m_snapshot = snapshot;
    }

    const Snapshot& GetSnapshot() const 
    { 
        return m_snapshot; 
    }

private:
    TxId m_txId;
    atomic<TransactionStatus> m_txStatus;
    Snapshot m_snapshot;
};

using TransactionInfoSPtr = shared_ptr< TransactionInfo >;

class AriesTransaction
{
public:
    TxId GetTxId() const 
    { 
        return m_txInfo->GetTxId(); 
    }

    const Snapshot& GetSnapshot() const 
    { 
        return m_txInfo->GetSnapshot(); 
    }

    TransactionInfoSPtr GetTxInfo() const 
    { 
        return m_txInfo; 
    }

    int GetCmdId() const 
    { 
        return m_cmdId; 
    }

    std::shared_ptr< AriesXLogWriter > GetXLogWriter()
    {
        return m_xLogWriter;
    }

    void AddModifiedTable( const TableId tableId )
    {
        m_modifiedTables.insert( tableId );
    }

    const std::set< TableId >& GetModifiedTables() const
    {
        return m_modifiedTables;
    }

    bool IsTableModified( const TableId tableId ) const;

    bool IsModifyingTable( TableId tableId ) const { return tableId == m_updateTableId; }
    void SetUpdateTableId( TableId tableId ) { m_updateTableId = tableId; }
    THD* GetMyTHD()
    {
        return m_thd;
    }

    int GetMyDbVersion()
    {
        return m_currentDbVersion;
    }

    bool isDDL() const
    {
        return mIsDDL;
    }

    void SetIsDDL( bool b )
    {
        mIsDDL = b;
    }

public:
    // create new transaction through transaction manager
    AriesTransaction( TxId argTxId, int currentDbVersion, bool isDDL = false );
    ~AriesTransaction();

private:

    /**
     * @brief Commit or abort this transaction
     */
    void End( TransactionStatus status );

private:
    THD* m_thd;
    int m_currentDbVersion;//!< the db version when this tx created. If vacuum happens, tx can read the latest db version and compare with this value.

    TransactionInfoSPtr m_txInfo;

    ISOLATION_LEVEL m_level = ISOLATION_LEVEL::REPEATABLE_READ;

    /// current command id, start from 0
    int m_cmdId;

    TableId m_updateTableId;///< 如果当前sql是insert, update或者delete语句，记录目标table ID

    std::shared_ptr< AriesXLogWriter > m_xLogWriter;

    friend class AriesTransManager;

    bool mIsDDL = false;

    std::set< TableId > m_modifiedTables;//!< 本transaction修改过的table
};

using AriesTransactionPtr = shared_ptr<AriesTransaction>;

END_ARIES_ENGINE_NAMESPACE

#endif
