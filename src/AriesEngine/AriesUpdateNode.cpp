#include "AriesUpdateNode.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "transaction/AriesMvccTableManager.h"
#include "transaction/AriesTransManager.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"
#include "Compression/dict/AriesDictManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE
AriesUpdateNode::AriesUpdateNode( const AriesTransactionPtr& transaction, const string& dbName, const string& tableName )
    : m_transaction(transaction),
      m_targetDbName( dbName ),
      m_targetTableName( tableName ),
      m_ColumnId4RowPos(0),
      m_aborted(false)
{
    m_dbEntry =
       schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    m_tableEntry = m_dbEntry->GetTableByName( tableName );
    m_mvccTable = AriesMvccTableManager::GetInstance().getMvccTable(dbName, tableName);
}

AriesUpdateNode::~AriesUpdateNode()
{
    m_updateColumnIds.clear();
}

bool AriesUpdateNode::Open()
{
    ARIES_ASSERT( m_updateColumnIds.size(), "no any update columnIds");
    ARIES_ASSERT( m_ColumnId4RowPos, "should set rowPos columnId");
    ARIES_ASSERT( m_dataSource , "m_dataSource is nullptr");
    return m_dataSource->Open();
}

void AriesUpdateNode::Close()
{
    ARIES_ASSERT( m_dataSource , "m_dataSource is nullptr");
    m_dataSource->Close();
}

void AriesUpdateNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
{
    ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
    m_dataSource->SetCuModule( modules );
}

string AriesUpdateNode::GetCudaKernelCode() const
{
    return m_dataSource->GetCudaKernelCode();
}

TupleDataSPtr AriesUpdateNode::Convert2TupleDataBuffer(AriesTableBlockUPtr &tableBlock,
                                                       unordered_map< int32_t, AriesManagedIndicesArraySPtr >& dictTupleData,
                                                       vector< AriesDataBufferSPtr >& origDictStringColumns )
{
    auto columsData = tableBlock->GetAllColumns();
    ARIES_ASSERT(columsData.size() == m_updateColumnIds.size() + 1, "error update columns count, TableBlock columns, count from source: "
                         + to_string(columsData.size() - 1) + ", target Columns Count: " + to_string(m_updateColumnIds.size()));
    TupleDataSPtr tupleDataBuffer = make_shared<TupleData>();
    int updateColumnsIdsIndex = 0;
    for (auto it = columsData.begin(); it != columsData.end(); ++it)
    {
        if (it->first == m_ColumnId4RowPos)
        {
            continue;
        }
        auto colId = m_updateColumnIds[ updateColumnsIdsIndex ];
        auto colEntry = m_tableEntry->GetColumnById( colId );
        auto colBuff = it->second->GetDataBuffer();
        if ( EncodeType::DICT == colEntry->encode_type )
        {
            auto initialTable = AriesInitialTableManager::GetInstance().getTable( m_targetDbName, m_targetTableName );
            auto columnDict = initialTable->GetColumnDict( colId - 1 );
            aries_acc::AriesManagedIndicesArraySPtr newDictDataRows;
            aries_acc::AriesManagedIndicesArraySPtr dictIndices;
            auto indiceColBuff = ConvertDictEncodedColumn( colEntry->GetName(),
                                                           colBuff,
                                                           columnDict,
                                                           colEntry->GetDictIndexColumnType(),
                                                           dictIndices );
            tupleDataBuffer->data[ colId ] = indiceColBuff;
            origDictStringColumns.push_back( colBuff );
            dictTupleData [ colId ] = dictIndices;
        }
        else
        {
            AriesColumnType targetType = CovertToAriesColumnType( colEntry->GetType(),
                                                colEntry->GetLength(),
                                                colEntry->IsAllowNull(),
                                                true,
                                                colEntry->numeric_precision,
                                                colEntry->numeric_scale );
            if ( colBuff->GetDataType() == targetType )
            {
                tupleDataBuffer->data[ colId ] = colBuff;
            }
            else
            {
                auto converted = std::make_shared< AriesDataBuffer >( targetType, colBuff->GetItemCount() );
                if ( !TransferColumnData( colEntry->GetName(), *converted, *colBuff ) )
                {
                    auto name = GenerateParamType( targetType );
                    auto sourceName = GenerateParamType( colBuff->GetDataType() );
                    ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, name.c_str(), sourceName.c_str(), colEntry->GetName().data(), 1 );
                }
                else
                {
                    tupleDataBuffer->data[ colId ] = converted;
                }
            }
        }

        updateColumnsIdsIndex++;
    }
    return tupleDataBuffer;
}

AriesOpResult AriesUpdateNode::GetNext()
{  
    int64_t totalCount = 0;
    int64_t changedCount = 0;
    int64_t warningsCount = 0;
    AriesOpResult ret;
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
#endif
    do
    {
        ret = m_dataSource->GetNext();
        if (ret.Status == AriesOpNodeStatus::ERROR)
        {
            return {AriesOpNodeStatus::ERROR, nullptr};
        }
#ifdef ARIES_PROFILE
        t.begin();
#endif
        totalCount += ret.TableBlock->GetRowCount();

        // pair: ( 插入数据行索引, 新增字典条目索引 )
        auto dictTupleData = unordered_map< int32_t, AriesManagedIndicesArraySPtr >();
        vector< AriesDataBufferSPtr > origDictStringColumns;

        auto tupleDataBuf = Convert2TupleDataBuffer( ret.TableBlock, dictTupleData, origDictStringColumns );
        auto rowPosBuf = ret.TableBlock->GetColumnBuffer(m_ColumnId4RowPos);
        LOG( INFO ) << "Updating datababse " << m_targetDbName << ", table " << m_targetTableName 
                    << ", " << rowPosBuf->GetItemCount() << " rows"; 

        for ( auto it = dictTupleData.begin(); it != dictTupleData.end(); ++ it )
        {
            auto colId = it->first;
            auto colEntry = m_tableEntry->GetColumnById( colId );
            auto dictIndices = it->second;
            if ( !AriesDictManager::GetInstance().AddDictTuple( m_transaction,
                                                                colEntry->GetDict(),
                                                                dictIndices ) )
                ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        }


        for (int64_t i = 0; i < ret.TableBlock->GetRowCount(); ++i)
        {
            auto pos = *reinterpret_cast< RowPos* >(rowPosBuf->GetItemDataAt(i));
            if (!UpdateTupleFirstWriterWin(pos, tupleDataBuf, i))
            {
                return {AriesOpNodeStatus::ERROR, nullptr};
            }
            ++changedCount;
        }
#ifdef ARIES_PROFILE
        long timeCost = t.end();
        m_opTime += timeCost;
        LOG( INFO ) << "Mvcc table update time " << timeCost << "ms";
#endif
    } while (ret.Status != AriesOpNodeStatus::END);

    //make result
    AriesDataBufferSPtr dataBuffer = make_shared<AriesDataBuffer>(AriesColumnType{{AriesValueType::INT64}, false}, 3);
    auto dataPtr = dataBuffer->GetData();
    auto itemSize = dataBuffer->GetDataType().GetDataTypeSize();
    memcpy( dataPtr, &totalCount, sizeof(int64_t));
    memcpy( dataPtr + itemSize, &changedCount, sizeof(int64_t));
    memcpy( dataPtr + 2 * itemSize, &warningsCount, sizeof(int64_t));
    AriesColumnSPtr column = make_shared<AriesColumn>();
    column->AddDataBuffer(dataBuffer);
    AriesTableBlockUPtr tableBlock = make_unique<AriesTableBlock>();
    tableBlock->AddColumn(1, column);

    LOG( INFO ) << "Updated database " << m_targetDbName << ", table " << m_targetTableName
                << ", " << changedCount << " rows, " << m_opTime << "ms";

    m_rowCount += totalCount;

    return AriesOpResult{AriesOpNodeStatus::END, move(tableBlock)};
}

// bool AriesUpdateNode::UpdateTuple(const RowPos &oldPos, const TupleDataSPtr dataBuffer, int dataIndex)
// {
//     auto newTxId = m_transaction->GetTxId();
//     auto &instance = AriesTransManager::GetInstance();
//     while ( true )
//     {
//         if (IsCurrentThdKilled())
//         {
//             return false;
//         }
//         /* oldTxMax is INVALID_TX_ID or oldTxMax is ABORTED, ready to try update */
//         auto oldTxMax = m_mvccTable->GetTxMax(oldPos);
//         if (oldTxMax != INVALID_TX_ID)
//         {
//             auto oldTxStatus = instance.GetTxStatus(oldTxMax);
//             if (oldTxStatus == TransactionStatus::IN_PROGRESS)
//             {
//                 instance.WaitForTxEnd(newTxId, oldTxMax);
//                 /* oldTxMax inprogress --> complete, should check again */
//                 continue;
//             }
//             /* oldTxMax is COMMITTED, should abort current tx */
//             else if (oldTxStatus == TransactionStatus::COMMITTED)
//             {
//                 return false;
//             }
//         }
//         if (m_mvccTable->TryLock(oldPos))
//         {
//             /* oldTxMax of oldPos is not changed, can be updated */
//             auto canBeUpdated = (oldTxMax == m_mvccTable->GetTxMax(oldPos));
//             if (canBeUpdated)
//             {
//                 m_mvccTable->SetTxMax(oldPos, newTxId);
//             }
//             m_mvccTable->Unlock(oldPos);
//             if (canBeUpdated)
//             {
//                 break;
//             }
//         }
//     }
// 
//     char* checkKeys = getenv("RATEUP_CHECK_KEYS");
//     bool bCheckKeys = false;
//     if ( checkKeys && '1' == checkKeys[ 0 ] )
//     {
//         bCheckKeys = true;
//     }
// 
//     if( !m_mvccTable->ModifyTuple(m_transaction, m_transaction->GetCmdId(), oldPos, dataBuffer, dataIndex, checkKeys) )
//         ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
//     return true;
// }

bool AriesUpdateNode::UpdateTupleFirstWriterWin( const RowPos &oldPos, const TupleDataSPtr dataBuffer, int dataIndex )
{
    if( IsCurrentThdKilled() )
        return false;

    bool bSuccess = false;
    if( m_mvccTable->TryLock( oldPos ) )
    {
        auto oldTxMax = m_mvccTable->GetTxMax( oldPos );
        if( oldTxMax == INVALID_TX_ID || AriesTransManager::GetInstance().GetTxStatus( oldTxMax ) == TransactionStatus::ABORTED )
        {
            m_mvccTable->SetTxMax( oldPos, m_transaction->GetTxId() );
            m_mvccTable->Unlock( oldPos );
            if( !m_mvccTable->ModifyTuple( m_transaction, oldPos, dataBuffer, dataIndex ) )
                ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
            bSuccess = true;
        }
        else
            m_mvccTable->Unlock( oldPos );
    }
    return bSuccess;
}

JSON AriesUpdateNode::GetProfile() const
{
    JSON stat = this->AriesOpNode::GetProfile();
    stat["type"] = "AriesUpdateNode";
    return stat;
}

END_ARIES_ENGINE_NAMESPACE
