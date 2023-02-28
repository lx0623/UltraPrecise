//
// Created by david.shen on 2020/3/12.
//

#include "AriesMvccTable.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesXLogWriter.h"
#include "../index/AriesIndex.h"
#include "AriesMvccTableManager.h"
#include "AriesInitialTableManager.h"
#include "CpuTimer.h"
#include "AriesEngine/AriesOpNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesMvccTable::AriesMvccTable( const string &dbName, const string &tableName )
            : m_dbName( dbName ),
              m_tableName( tableName ),
              m_allIndexKeysChecked( false )
    {
#ifdef ARIES_PROFILE
        CPU_Timer t;
        t.begin();
#endif
        m_createKeyIndexInprogress = false;
        auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
        m_tableEntry = database->GetTableByName( tableName );
        m_tableId = m_tableEntry->GetId();
        m_InitialTableSPtr = AriesInitialTableManager::GetInstance().getTable( dbName, tableName );
        m_tupleParseSPtr = make_shared< TupleParser >( m_tableEntry );
        // delta table row count range: [ DELTA_TABLE_TUPLE_COUNT_MIN, DELTA_TABLE_TUPLE_COUNT_MAX ]
        size_t percentCount = m_InitialTableSPtr->GetTotalRowCount() / 50;
        m_perDeltaTableBlockCapacity = std::max( percentCount, DELTA_TABLE_TUPLE_COUNT_MIN );
        m_perDeltaTableBlockCapacity = std::min( m_perDeltaTableBlockCapacity, ( size_t )DELTA_TABLE_TUPLE_COUNT_MAX );
        LOG( INFO ) << "delta table capacity for table " << dbName << "." << tableName << ": " << m_perDeltaTableBlockCapacity;
        m_deltaTableSPtr = make_shared< AriesDeltaTable >( m_perDeltaTableBlockCapacity,
                                                           m_tupleParseSPtr->GetColumnTypes() );

        ParseIndexKeys( m_tableEntry );
#ifdef ARIES_PROFILE
        LOG( INFO )<< "Create AriesMvccTable time: " << t.end();
#endif
    }

    // for load data
    AriesMvccTable::AriesMvccTable( const string &dbName,
                                    const string &tableName,
                                    const string& tableDataDir )
            : m_dbName( dbName ),
              m_tableName( tableName ),
              m_allIndexKeysChecked( false )
    {
#ifdef ARIES_PROFILE
        CPU_Timer t;
        t.begin();
#endif
        m_createKeyIndexInprogress = false;
        auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
        m_tableEntry = database->GetTableByName( tableName );
        m_tableId = m_tableEntry->GetId();
        m_InitialTableSPtr = make_shared< AriesInitialTable >( dbName, tableName, tableDataDir );

        // delta table row count range: [ DELTA_TABLE_TUPLE_COUNT_MIN, DELTA_TABLE_TUPLE_COUNT_MAX ]
        size_t percentCount = m_InitialTableSPtr->GetTotalRowCount() / 50;
        m_perDeltaTableBlockCapacity = std::max( percentCount, DELTA_TABLE_TUPLE_COUNT_MIN );
        m_perDeltaTableBlockCapacity = std::min( m_perDeltaTableBlockCapacity, ( size_t )DELTA_TABLE_TUPLE_COUNT_MAX );

        ParseIndexKeys( m_tableEntry );
#ifdef ARIES_PROFILE
        LOG( INFO )<< "Create AriesMvccTable time: " << t.end();
#endif
    }

    void AriesMvccTable::ResetInitTable()
    {
        m_InitialTableSPtr = AriesInitialTableManager::GetInstance().getTable( m_dbName, m_tableName );
    }

    bool AriesMvccTable::AddTuple( AriesTransactionPtr transaction, TupleDataSPtr dataBuffer, int dataIndex, bool checkKeys )
    {
        ARIES_ASSERT( dataBuffer->data.size() == m_tupleParseSPtr->GetColumnsCount(), "InsertNode should input all columns data" );
        string primaryKey;
        size_t existedPrimaryKeyLocationCount = 0;
        checkKeys = checkKeys && HasPrimaryKey();
        if( checkKeys )
        {
            CreatePrimaryKeyIndexIfNotExists();
            // check index keys
            primaryKey = MakePrimaryKey( dataBuffer, dataIndex );
            if( !CheckIfCanBeInserted( transaction, primaryKey, dataIndex, existedPrimaryKeyLocationCount ) )
            {
                ThrowException4PrimaryKeyConflict( dataBuffer, dataIndex );
                return false;
            }
        }
        bool unused;
        vector< RowPos > poses = m_deltaTableSPtr->ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
        if( poses.empty() )
            return false;
        RowPos newPos = poses[0];
        pTupleHeader newHeader = m_deltaTableSPtr->GetTupleHeader( newPos, AriesDeltaTableSlotType::AddedTuples );
        auto txId = transaction->GetTxId();
        newHeader->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );

        std::vector< int8_t* > columnBuffers( m_tupleParseSPtr->GetColumnsCount() );
        m_deltaTableSPtr->GetTupleFieldBuffer( newPos, columnBuffers );
        m_tupleParseSPtr->FillData( columnBuffers, dataBuffer, dataIndex );
        m_deltaTableSPtr->CompleteSlot(
        { newPos }, AriesDeltaTableSlotType::AddedTuples );

        if( checkKeys )
        {
            //add index keys
            if( !AddAllIndexKeys( primaryKey, newPos, existedPrimaryKeyLocationCount ) )
            {
                ThrowException4PrimaryKeyConflict( dataBuffer, newPos );
                return false;
            }
        }

        return transaction->GetXLogWriter()->WriteSingleInsertLog( m_tableId, newPos, columnBuffers, m_tupleParseSPtr );
    }

    bool AriesMvccTable::AddTuple( AriesTransactionPtr transaction, TupleDataSPtr dataBuffer, bool checkKeys )
    {
        ARIES_ASSERT( dataBuffer->data.size() == m_tupleParseSPtr->GetColumnsCount(), "InsertNode should input all columns data" );
        string primaryKey;
        auto rowCount = dataBuffer->data.begin()->second->GetItemCount();
        std::vector< size_t > existedPrimaryKeyLocationCounts( rowCount );
        std::vector< std::string > primaryKeys( rowCount );
        checkKeys = checkKeys && HasPrimaryKey();
        if( checkKeys )
        {
            CreatePrimaryKeyIndexIfNotExists();
            for ( size_t i = 0; i < rowCount; i++ )
            {
                primaryKey = MakePrimaryKey( dataBuffer, i );
                if( !CheckIfCanBeInserted( transaction, primaryKey, i, existedPrimaryKeyLocationCounts[ i ] ) )
                {
                    ThrowException4PrimaryKeyConflict( dataBuffer, i );
                    return false;
                }
                primaryKeys[ i ] = primaryKey;
            }
        }

        bool isContinuous = false;
        vector< RowPos > poses = m_deltaTableSPtr->ReserveSlot( rowCount, AriesDeltaTableSlotType::AddedTuples, isContinuous );
        if( poses.empty() )
            return false;

        auto txId = transaction->GetTxId();
        for ( size_t i = 0; i < rowCount; i++ )
        {
            RowPos newPos = poses[ i ];
            pTupleHeader newHeader = m_deltaTableSPtr->GetTupleHeader( newPos, AriesDeltaTableSlotType::AddedTuples );
            newHeader->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
        }

        std::vector< int8_t* > columnBuffers( m_tupleParseSPtr->GetColumnsCount() );
        std::vector< int8_t* > logBuffers( m_tupleParseSPtr->GetColumnsCount() );
        if( isContinuous )
        {
            m_deltaTableSPtr->GetTupleFieldBuffer( poses[ 0 ], columnBuffers );

            for ( const auto& it : dataBuffer->data )
            {
                const auto& columnId = it.first;
                const auto& buffer = it.second;

                auto& targetBuffer = columnBuffers[ columnId - 1 ];

                // TODO: buffer overflow
                memcpy( targetBuffer, buffer->GetData(), buffer->GetItemSizeInBytes() * rowCount );
            }
            logBuffers.clear();
            logBuffers.insert( logBuffers.end(), columnBuffers.begin(), columnBuffers.end() );
        }
        else
        {
            int total = poses.size();
            for( int start = 0; start < total; )
            {
                //当前slot pos
                RowPos startPos = poses[ start ];
                //下一个相邻slot pos
                RowPos endPos = startPos + 1;
                int end = start + 1;
                while( end < total && ( endPos % m_perDeltaTableBlockCapacity ) != 1 )
                {
                    //向后搜索连续相邻的slot
                    if( endPos == poses[ end ] )
                    {
                        ++endPos;
                        ++end;
                    }
                    else
                        break;
                }
                m_deltaTableSPtr->GetTupleFieldBuffer( startPos, columnBuffers );
                for( const auto& it : dataBuffer->data )
                {
                    const auto& columnId = it.first;
                    const auto& buffer = it.second;

                    auto& targetBuffer = columnBuffers[ columnId - 1 ];

                    memcpy( targetBuffer, buffer->GetItemDataAt( start ), buffer->GetItemSizeInBytes() * ( end - start ) );
                }
                start = end;
            }

            for ( const auto& it : dataBuffer->data )
                logBuffers[ it.first - 1 ] = it.second->GetData();
        }

        m_deltaTableSPtr->CompleteSlot( poses, AriesDeltaTableSlotType::AddedTuples );

        bool checkResult = true;
        for ( size_t i = 0; i < rowCount; i++ )
        {
            if( checkKeys && checkResult )
            {
                if( !AddAllIndexKeys( primaryKeys[ i ], poses[ i ], existedPrimaryKeyLocationCounts[ i ] ) )
                {
                    ThrowException4PrimaryKeyConflict( dataBuffer, i );
                    checkResult = false;
                }
            }
        }

        return checkResult && transaction->GetXLogWriter()->WriteBatchInsertLog( m_tableId, poses, logBuffers, m_tupleParseSPtr );
    }

    bool AriesMvccTable::DeleteTuple( AriesTransactionPtr transaction, const RowPos* tuplePos, size_t count )
    {
        bool bSuccess = true;
        size_t deletingInitialTupleCount = 0;
        const RowPos* tmpPos = tuplePos;
        for( size_t i = 0; i < count; ++i )
        {
            if( *tmpPos++ < 0 )
                ++deletingInitialTupleCount;
        }
        vector< RowPos > slotsForDeletingInitialTuple;
        bool isContinuous = false;
        if( deletingInitialTupleCount > 0 )
            slotsForDeletingInitialTuple = m_deltaTableSPtr->ReserveSlot( deletingInitialTupleCount, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
        assert( slotsForDeletingInitialTuple.size() == deletingInitialTupleCount );

        RowPos newPos;
        RowPos initialPos;
        int slotIndex = 0;
        auto txId = transaction->GetTxId();
        for( size_t i = 0; i < count; ++i )
        {
            assert( *tuplePos != 0 );
            if( TryLock( *tuplePos ) )
            {
                auto oldTxMax = GetTxMax( *tuplePos );
                if( oldTxMax == INVALID_TX_ID || AriesTransManager::GetInstance().GetTxStatus( oldTxMax ) == TransactionStatus::ABORTED )
                {
                    SetTxMax( *tuplePos, txId );
                    Unlock( *tuplePos );
                    initialPos = INVALID_ROWPOS;
                    newPos = INVALID_ROWPOS;
                    if( *tuplePos < 0 )
                    {
                        initialPos = *tuplePos;
                        newPos = slotsForDeletingInitialTuple[slotIndex++];
                        auto header = m_deltaTableSPtr->GetTupleHeader( newPos, AriesDeltaTableSlotType::DeletedInitialTableTuples );
                        header->Initial( INVALID_TX_ID, txId, *tuplePos );
                    }
                    else 
                    {
                        // 删除行存储数据, xmax已被修改, 这里啥都不做
                    }
                    
                    if( !transaction->GetXLogWriter()->WriteCommandLog( OperationType::Delete, m_tableId, *tuplePos, newPos, initialPos ) )
                    {
                        LOG( ERROR ) << "cannot write delete log";
                        bSuccess = false;
                        break;
                    }
                }
                else
                {
                    Unlock( *tuplePos );
                    bSuccess = false;
                    break;
                }
            }
            else
            {
                bSuccess = false;
                break;
            }
            
            ++tuplePos;
        }

        if( !slotsForDeletingInitialTuple.empty() )
        {
            if( bSuccess )
                m_deltaTableSPtr->CompleteSlot( slotsForDeletingInitialTuple, AriesDeltaTableSlotType::DeletedInitialTableTuples );
            else
                m_deltaTableSPtr->FreeSlot( slotsForDeletingInitialTuple, AriesDeltaTableSlotType::DeletedInitialTableTuples );
        }

        return bSuccess;
    }

    bool AriesMvccTable::ModifyTuple( AriesTransactionPtr transaction, RowPos oldPos, TupleDataSPtr dataBuffer, int dataIndex,
            bool checkKeys )
    {
        auto txId = transaction->GetTxId();
        {
            RowPos newPos = INVALID_ROWPOS;
            auto initialPos = oldPos;
            TupleHeader* header;
            //删除tuple
            if( oldPos < 0 )
            {
                //删除列存储数据, 新增tuple, 只修改TupleHeader即可
                bool isContinuous = false;
                vector< RowPos > slots = m_deltaTableSPtr->ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
                if( slots.empty() )
                {
                    return false;
                }
                newPos = slots[0];
                header = m_deltaTableSPtr->GetTupleHeader( slots[0], AriesDeltaTableSlotType::DeletedInitialTableTuples );
                header->Initial( INVALID_TX_ID, txId, oldPos );
                m_deltaTableSPtr->CompleteSlot(
                { slots[0] }, AriesDeltaTableSlotType::DeletedInitialTableTuples );
            }
            else
            {
                // 删除行存储数据, xmax已被修改, 这里只修改ctid, 防止错误ctid
                initialPos = INVALID_ROWPOS;
                header = m_deltaTableSPtr->GetTupleHeader( oldPos, AriesDeltaTableSlotType::AddedTuples );
            }

            if ( !transaction->GetXLogWriter()->WriteCommandLog( OperationType::Delete, m_tableId, oldPos, newPos, initialPos ) )
            {
                LOG( ERROR ) << "cannot write delete log";
                return false;
            }

            if ( dataBuffer == nullptr )
            {
                return true;
            }
        }

        auto new_data = std::make_shared< TupleData >();
        auto& data = new_data->data;
        std::vector< int8_t* > oldColumnBuffers( m_tupleParseSPtr->GetColumnsCount() );
        if ( oldPos > 0 )
        {
            m_deltaTableSPtr->GetTupleFieldBuffer(oldPos, oldColumnBuffers );
        }

        std::vector< int > all_column_ids( m_tupleParseSPtr->GetColumnsCount() );
        std::iota( all_column_ids.begin(), all_column_ids.end(), 1 );
        auto table = m_InitialTableSPtr->GetTable( all_column_ids );
        for ( size_t i = 0; i < m_tupleParseSPtr->GetColumnsCount(); i++ )
        {
            auto column_id = i + 1;
            auto column_type = m_tupleParseSPtr->GetColumnType( column_id );
            if ( column_type.GetDataTypeSize() == 1 )
            {
                data[ column_id ] = std::make_shared< AriesDataBuffer >( column_type, 4 );
            }
            else
            {
                data[ column_id ] = std::make_shared< AriesDataBuffer >( column_type, 1 );
            }

            if ( dataBuffer->data.find( column_id ) != dataBuffer->data.cend() )
            {
                memcpy( data[ column_id ]->GetData(), dataBuffer->data[ column_id ]->GetItemDataAt( dataIndex ), column_type.GetDataTypeSize() );
                continue;
            }

            if ( oldPos < 0 )
            {
                auto encode_type = table->GetColumnEncodeType( column_id );
                if ( encode_type == EncodeType::NONE )
                {
                    memcpy( data[ column_id ]->GetData(), m_InitialTableSPtr->GetTupleFieldContent( i + 1, oldPos ), column_type.GetDataTypeSize() );
                }
                else if ( encode_type == EncodeType::DICT )
                {
                    auto column = table->GetDictEncodedColumn( column_id );
                    memcpy( data[ column_id ]->GetData(), column->GetIndices()->GetFieldContent( -oldPos - 1 ), column_type.GetDataTypeSize() );
                }
            }
            else
            {
                memcpy( data[ column_id ]->GetData(), oldColumnBuffers[ i ], column_type.GetDataTypeSize() );
            }
        }

        return dataBuffer == nullptr || AddTuple( transaction, new_data, 0, checkKeys );
    }

    AriesTableBlockUPtr AriesMvccTable::MakeTable(
        const AriesTransactionPtr &transaction,
        const vector< int > &colsId,
        const std::vector< AriesCommonExprUPtr >& partitionConditions ) const
    {
#ifdef ARIES_PROFILE
        CPU_Timer t;
        t.begin();
#endif
        LOG( INFO )<< "AriesMvccTable::MakeTable";

        auto expectRowPosColumnId = m_tupleParseSPtr->GetColumnsCount() + 1;
        vector< int > physicalColumnIds;
        map< int32_t, int32_t > updateColIds;
        int rowPosColumnId = 0;
        for( size_t i = 0; i < colsId.size(); ++i )
        {
            int id = colsId[i];
            assert( id > 0 );
            if( ( size_t )id >= expectRowPosColumnId )
            {
                rowPosColumnId = i + 1;
                continue;
            }
            if( rowPosColumnId )
            {
                updateColIds[i + 1] = i;
            }
            physicalColumnIds.push_back( id );
        }
        // 1, acquire mvcc visible data indices
        vector< RowPos > visibleInDeltaTable, invisibleInInitialTable, visibleInInitialTable;

        m_deltaTableSPtr->GetVisibleRowIdsInDeltaTable( transaction->GetTxId(), transaction->GetSnapshot(), visibleInDeltaTable,
                invisibleInInitialTable );

        // 2, cache all required columns data from initial table    从 原始表中 缓存 所有必需的列数据
        auto resTable = m_InitialTableSPtr->GetTable( physicalColumnIds, partitionConditions );
        int64_t initialTableRowCount = resTable->GetRowCount();

        // 3, make data from deltaTable
        size_t deltaRowCount = visibleInDeltaTable.size();
        // 4, make totalVisibleRowPosesBuf and totalVisibleRowIndices
        int64_t totalVisibleRowCount = initialTableRowCount - invisibleInInitialTable.size() + deltaRowCount;
        if( invisibleInInitialTable.size() + deltaRowCount > 0 )
        {
            //clear partition info
            resTable->ClearPartitionInfo();
        }
        if( colsId.empty() )
        {
            resTable->SetRowCount( totalVisibleRowCount );
            return resTable;
        }

        if( totalVisibleRowCount > 0 )
        {
            AriesDataBufferSPtr totalVisibleRowPosesBuf = nullptr;
            auto deltaTableSize = m_deltaTableSPtr->GetDeltaTableSize();

            // 如果 initial table 中有被删除的数据，需要生成 indices
            if ( !invisibleInInitialTable.empty() )
            {
                if( visibleInDeltaTable.size() > 0 )
                {
                    auto deltaColumns = m_deltaTableSPtr->GetColumnBuffers();

                    std::vector< AriesDataBufferSPtr > columnDataBuffers( physicalColumnIds.size() );
                    for ( size_t i = 0; i < columnDataBuffers.size(); i++ )
                    {
                        auto columnId = physicalColumnIds[ i ];
                        auto &type = m_tupleParseSPtr->GetColumnType( columnId );
                        auto dataBuffer = std::make_shared< AriesDataBuffer >( type, deltaTableSize );
                        dataBuffer->PrefetchToGpu();
                        int8_t* pData = dataBuffer->GetData();
                        size_t copySize = type.GetDataTypeSize() * m_perDeltaTableBlockCapacity;
                        for ( const auto columns_buffer : deltaColumns )
                        {
                            cudaMemcpy( pData, columns_buffer[ columnId - 1 ], copySize, cudaMemcpyHostToDevice );
                            pData += copySize;
                        }

                        auto encodeType = resTable->GetColumnEncodeType( i + 1 );
                        if ( encodeType == EncodeType::DICT )
                        {
                            auto column = resTable->GetDictEncodedColumn( i + 1 );
                            column->GetIndices()->AddDataBuffer( dataBuffer );
                        }
                        else
                        {
                            auto column = resTable->GetMaterilizedColumn( i + 1 );
                            column->AddDataBuffer( dataBuffer );
                        }
                    }
                }

                if( rowPosColumnId )
                {
                    auto column = CreateRowIdColumn( initialTableRowCount, deltaTableSize, ARIES_DATA_BLOCK_ROW_SIZE );
                    resTable->UpdateColumnIds( updateColIds );
                    resTable->AddColumn( rowPosColumnId, column );
                }

                auto indices = CreateIndicesForMvccTable( invisibleInInitialTable,
                                                            visibleInDeltaTable,
                                                            initialTableRowCount,
                                                            deltaTableSize );
                resTable->UpdateIndices( indices );
            }
            // 因为 initial table 会全部被选中（没有不可见的行），这里将 delta table 中的数据物化出来，避免创建 0-N 的 indices
            else if( deltaRowCount )
            {
                std::vector< int8_t* > columnBuffers( physicalColumnIds.size() );

                std::vector< AriesDataBufferSPtr > columnDataBuffers( physicalColumnIds.size() );
                for ( size_t i = 0; i < columnDataBuffers.size(); i++ )
                {
                    auto &type = m_tupleParseSPtr->GetColumnType( physicalColumnIds[ i ] );
                    columnDataBuffers[ i ] = std::make_shared< AriesDataBuffer >( type, deltaRowCount );
                }

                AriesColumnSPtr rowPosColumn;
                AriesDataBufferSPtr rowPosDataBuffer;
                if( rowPosColumnId )
                {
                    rowPosColumn = CreateRowIdColumnMaterialized( initialTableRowCount, deltaRowCount, ARIES_DATA_BLOCK_ROW_SIZE );
                    auto buffers = rowPosColumn->GetDataBuffers();
                    rowPosDataBuffer = buffers[ buffers.size() - 1 ];
                }

                for( size_t i = 0; i < deltaRowCount; i++ )
                {
                    m_deltaTableSPtr->GetTupleFieldBuffer( visibleInDeltaTable[ i ], columnBuffers, physicalColumnIds );

                    for ( size_t j = 0; j < physicalColumnIds.size(); j++ )
                    {
                        int colId = physicalColumnIds[ j ];
                        auto &type = m_tupleParseSPtr->GetColumnType( colId );
                        const int8_t *pFieldBuf;

                        pFieldBuf = columnBuffers[ j ];

                        auto* p = columnDataBuffers[ j ]->GetItemDataAt( i );
                        memcpy( p, pFieldBuf, type.GetDataTypeSize() );
                    }

                    if ( rowPosColumnId )
                    {
                        *( RowPos* )( rowPosDataBuffer->GetItemDataAt( i ) ) = visibleInDeltaTable[ i ];
                    }
                }

                for( size_t i = 0; i < physicalColumnIds.size(); i++ )
                {
                    int colId = i + 1;//physicalColumnIds[i];
                    auto& deltaColumnDataBuffer = columnDataBuffers[ i ];
                    deltaColumnDataBuffer->MemAdvise( cudaMemAdviseSetReadMostly, 0 );

                    auto colEncodeType = resTable->GetColumnEncodeType( colId );
                    if( EncodeType::DICT == colEncodeType )
                    {
                        auto col = resTable->GetDictEncodedColumn( colId );
                        col->GetIndices()->AddDataBuffer( deltaColumnDataBuffer );
                    }
                    else
                    {
                        auto col = resTable->GetMaterilizedColumn( colId );
                        col->AddDataBuffer( deltaColumnDataBuffer );
                    }
                }

                if( rowPosColumnId )
                {
                    resTable->UpdateColumnIds( updateColIds );
                    resTable->AddColumn( rowPosColumnId, rowPosColumn );
                }
            }
            else
            {
                if( rowPosColumnId )
                {
                    auto rowPosColumn = CreateRowIdColumnMaterialized( initialTableRowCount, deltaRowCount, ARIES_DATA_BLOCK_ROW_SIZE );
                    resTable->UpdateColumnIds( updateColIds );
                    resTable->AddColumn( rowPosColumnId, rowPosColumn );
                }
            }

            resTable->SetRowCount( 0 ); // use row count in columns
#ifdef ARIES_PROFILE
            LOG( INFO )<< "AriesMvccTable::MakeTable time: " << t.end();
#endif
            return resTable;
        }
        else
        {
            if( rowPosColumnId )
            {
                resTable->UpdateColumnIds( updateColIds );
                AriesColumnSPtr visibleRowIdsColumn = make_shared< AriesColumn >();
                resTable->AddColumn( rowPosColumnId, visibleRowIdsColumn );
            }
            return resTable->CloneWithNoContent();
        }
    }

    AriesTableBlockUPtr AriesMvccTable::GetTable(
        const AriesTransactionPtr &transaction,
        const vector< int > &colsId,
        const std::vector< AriesCommonExprUPtr >& partitionConditions ) const
    {
        return MakeTable( transaction, colsId, partitionConditions );
    }

    //设置对应数据行的t_xmax值
    void AriesMvccTable::SetTxMax( RowPos pos, TxId value )
    {
        ARIES_ASSERT( pos != 0, "bad pos: 0" );
        if( pos < 0 )
        {
            m_InitialTableSPtr->SetTxMax( pos, value );
        }
        else
        {
            m_deltaTableSPtr->SetTxMax( pos, value );
        }
    }

    //获取对应数据行的t_xmax值
    TxId AriesMvccTable::GetTxMax( RowPos pos ) const
    {
        ARIES_ASSERT( pos != 0, "bad pos: 0" );
        if( pos < 0 )
        {
            return m_InitialTableSPtr->GetTxMax( pos );
        }
        else
        {
            return m_deltaTableSPtr->GetTxMax( pos );
        }
    }

    //对某行加锁
    void AriesMvccTable::Lock( RowPos pos )
    {
        assert( pos != 0 );
        if( pos < 0 )
            m_InitialTableSPtr->Lock( pos );
        else
            m_deltaTableSPtr->Lock( pos );
    }

    bool AriesMvccTable::TryLock( RowPos pos )
    {
        assert( pos != 0 );
        if( pos < 0 )
            return m_InitialTableSPtr->TryLock( pos );
        else
            return m_deltaTableSPtr->TryLock( pos );
    }

    //对某行解锁
    void AriesMvccTable::Unlock( RowPos pos )
    {
        assert( pos != 0 );
        if( pos < 0 )
            m_InitialTableSPtr->Unlock( pos );
        else
            m_deltaTableSPtr->Unlock( pos );
    }

    // pos是一个有效位置，并且对应key columns的值和参数key的值相同时返回true,否则返回false
    bool AriesMvccTable::IsKeyValid( const void *key, RowPos pos, const vector< int > &keyColumnIds ) const
    {
        bool ret = true;
        const int8_t* pKey = ( const int8_t* )key;
        for( int columnId : keyColumnIds )
        {
            size_t dataTypeSize = m_tupleParseSPtr->GetColumnType( columnId ).GetDataTypeSize();
            if( memcmp( pKey, GetFieldContent( pos, columnId ), dataTypeSize ) != 0 )
            {
                ret = false;
                break;
            }
            //让pKey指向下一个key
            pKey += dataTypeSize;
        }
        return ret;
    }

    const int8_t* AriesMvccTable::GetFieldContent( RowPos pos, int columnId ) const
    {
        const int8_t* pData;
        if( pos > 0 )
        {
            //delta table
            pData = m_deltaTableSPtr->GetTupleFieldBuffer( pos, columnId );
        }
        else
        {
            //initial table
            pData = m_InitialTableSPtr->GetTupleFieldContent( columnId, pos );
        }
        return pData;
    }

    //对于txId对应的transaction而言，主键可能存在（tuple的xmin或xmax处于inprogress状态时获取的信息不一定准确，随着事务的提交或者终止，对应tuple的主键可能变成有效
    bool AriesMvccTable::PrimaryKeyMightExists( TxId txId, const void *key, RowPos pos, const vector< int > &keyColumnIds ) const
    {
        bool bExists = false;
        AriesTransManager& transManager = AriesTransManager::GetInstance();
        if( pos > 0 )
        {
            //delta table
            if( m_deltaTableSPtr->IsTuplePublished( pos ) && IsKeyValid( key, pos, keyColumnIds ) )
            {
                TupleHeader *header = m_deltaTableSPtr->GetTupleHeader( pos, AriesDeltaTableSlotType::AddedTuples );
                TransactionStatus xminStatus = transManager.GetTxStatus( header->m_xmin );
                switch( xminStatus )
                {
                    case TransactionStatus::IN_PROGRESS:
                    {
                        bExists = ( header->m_xmax == INVALID_TX_ID );
                        break;
                    }
                    case TransactionStatus::COMMITTED:
                    {
                        bExists = ( header->m_xmax == INVALID_TX_ID ||
                                    ( header->m_xmax != txId && transManager.GetTxStatus( header->m_xmax ) != TransactionStatus::COMMITTED ) ) ||
                                    ( header->m_xmax == txId && transManager.GetTxStatus( header->m_xmax ) == TransactionStatus::ABORTED ) ;
                        break;
                    }
                    default:
                        break;
                }
            }
        }
        else
        {
            //initial table
            assert( IsKeyValid( key, pos, keyColumnIds ) );
            TxId xMax = m_InitialTableSPtr->GetTxMax( pos );
            bExists = ( ( xMax == INVALID_TX_ID || ( xMax != txId && transManager.GetTxStatus( xMax ) != TransactionStatus::COMMITTED ) ) ||
             ( xMax == txId && transManager.GetTxStatus( xMax ) == TransactionStatus::ABORTED ) );
        }
        return bExists;
    }

    //对于txId对应的transaction而言，主键肯定存在
    bool AriesMvccTable::PrimaryKeyExists( TxId txId, const void *key, RowPos pos, const vector< int > &keyColumnIds ) const
    {
        bool bExists = false;
        AriesTransManager& transManager = AriesTransManager::GetInstance();
        if( pos > 0 )
        {
            //delta table
            if( m_deltaTableSPtr->IsTuplePublished( pos ) && IsKeyValid( key, pos, keyColumnIds ) )
            {
                TupleHeader *header = m_deltaTableSPtr->GetTupleHeader( pos, AriesDeltaTableSlotType::AddedTuples );
                TransactionStatus xminStatus = transManager.GetTxStatus( header->m_xmin );
                switch( xminStatus )
                {
                    case TransactionStatus::IN_PROGRESS:
                    {
                        bExists = ( header->m_xmin == txId && header->m_xmax == INVALID_TX_ID );
                        break;
                    }
                    case TransactionStatus::COMMITTED:
                    {
                        bExists = ( header->m_xmax == INVALID_TX_ID || transManager.GetTxStatus( header->m_xmax ) == TransactionStatus::ABORTED );
                        break;
                    }
                    default:
                        break;
                }
            }
        }
        else
        {
            //initial table
            assert( IsKeyValid( key, pos, keyColumnIds ) );
            TxId xMax = m_InitialTableSPtr->GetTxMax( pos );
            bExists = ( xMax == INVALID_TX_ID || transManager.GetTxStatus( xMax ) == TransactionStatus::ABORTED );
        }
        return bExists;
    }

    void AriesMvccTable::ParseIndexKeys( TableEntrySPtr &tableEntry )
    {
        // for primaryKey
        auto pkColNames = tableEntry->GetPrimaryKey();
        if( !pkColNames.empty() )
        {
            for( auto colName : pkColNames )
            {
                auto colId = tableEntry->GetColumnByName( colName )->GetColumnIndex() + 1;
                m_primaryKey.columnIds.push_back( colId );
            }
        }
    }

    string AriesMvccTable::MakeIndexKey( const vector< int > &colIds, const TupleDataSPtr &dataBuffer, const int &rowIndex )
    {
        string key;
        for( auto const colId : colIds )
        {
            auto data = dataBuffer->data.find( colId );
            if( dataBuffer->data.end() == data )
            {
                ARIES_ASSERT( 0, "insert data error at column: " + to_string( colId ) );
            }
            key.insert( key.size(), ( const char* )data->second->GetItemDataAt( rowIndex ), data->second->GetItemSizeInBytes() );
        }
        return key;
    }

    string AriesMvccTable::MakePrimaryKey( const TupleDataSPtr &dataBuffer, int rowIndex )
    {
        string key;
        if( HasPrimaryKey() )
        {
            key = MakeIndexKey( m_primaryKey.columnIds, dataBuffer, rowIndex );
        }
        return key;
    }

    string AriesMvccTable::GetPrimaryKeyName()
    {
        string name;
        if( HasPrimaryKey() )
        {
            for( auto const colId : m_primaryKey.columnIds )
            {
                name += m_tableEntry->GetColumnById( colId )->GetName() + "_";
            }
            //remove last "_"
            name.erase( name.find_last_of( "_" ) );
        }
        return name;
    }

    string AriesMvccTable::GetPrimaryKeyValue( const int &rowIndex )
    {
        return string( "data in row " + to_string( rowIndex ) );
    }

    void AriesMvccTable::ThrowException4PrimaryKeyConflict( const TupleDataSPtr &dataBuffer, int rowIndex )
    {
        string conflictKeys;
        for( auto const colId : m_primaryKey.columnIds )
        {
            auto data = dataBuffer->data.find( colId );
            assert( dataBuffer->data.end() != data );
            conflictKeys += data->second->ItemToString( rowIndex ) + ",";
        }
        conflictKeys.erase( conflictKeys.size() -1, 1 );
        ARIES_EXCEPTION( ER_DUP_ENTRY_WITH_KEY_NAME, conflictKeys.c_str(), GetPrimaryKeyName().c_str() );
    }

    void AriesMvccTable::ThrowException4PrimaryKeyDelete()
    {
        ARIES_EXCEPTION( ER_ROW_IS_REFERENCED );
    }

    void AriesMvccTable::ThrowException4ForeignKeysNoReferenced( int errorKeyIndex )
    {
        auto tableInfo = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseAndTableById( m_tableId );
        auto dbEntry = tableInfo.first;
        const auto &fks = m_tableEntry->GetForeignKeys();
        auto &fk = fks[errorKeyIndex];
        string msg( "Cannot add or update a child row: a foreign key constraint fails (" );
        msg.append( "`" ).append( dbEntry->GetName() ).append( "`." ).append( "`" ).append( m_tableEntry->GetName() ).append( "`, " );
        msg.append( "CONSTRAINT `" ).append( fk->name ).append( "` FOREIGN KEY (" );
        for( auto &fkColName : fk->keys )
        {
            msg.append( "`" ).append( fkColName ).append( "`," );
        }
        msg.erase( msg.size() - 1 );
        msg.append( ") REFERENCES `" ).append( fk->referencedTable ).append( "` (" );
        for( auto &pkColName : fk->referencedKeys )
        {
            msg.append( "`" ).append( pkColName ).append( "`," );
        }
        msg.erase( msg.size() - 1 );
        msg.append( ")" );
        ARIES_EXCEPTION_SIMPLE( ER_NO_REFERENCED_ROW_2, msg.data() );
    }

    bool AriesMvccTable::CheckPrimaryKeyExist( AriesTransactionPtr &transaction, const string &key, size_t& existedPrimaryKeyLocationCount )
    {
        pair< bool, AriesRowPosContainer > result = m_primaryKey.index->FindKey( key );
        if( result.first )
        {
            size_t count = result.second.size();
            existedPrimaryKeyLocationCount = count;
            for( size_t i = 0; i < count; ++i )
            {
                if( PrimaryKeyMightExists( transaction->GetTxId(), key.data(), result.second[i], m_primaryKey.columnIds ) )
                {
                    return true;
                }
            }
        }
        return false;
    }

    bool AriesMvccTable::CheckIfCanBeInserted( AriesTransactionPtr &transaction, const string& pkKey, int rowIndex,
            size_t& existedPrimaryKeyLocationCount )
    {
        // check if pk exist
        return !CheckPrimaryKeyExist( transaction, pkKey, existedPrimaryKeyLocationCount );
    }

    bool AriesMvccTable::AddAllIndexKeys( const string &pkKey, RowPos rowPos, size_t existedPrimaryKeyLocationCount )
    {
        return m_primaryKey.index->InsertKey( pkKey, rowPos, true, existedPrimaryKeyLocationCount );
    }

    bool AriesMvccTable::CreateKeyIndex( const vector< int > &keyColIds, AriesTableKeysSPtr &index, bool checkDuplicate )
    {
        assert(index == nullptr);

        auto table = m_InitialTableSPtr->GetTable(keyColIds);

        vector< AriesColumnSPtr > columns;
        for( size_t i = 0; i < keyColIds.size(); ++i )
            columns.push_back( table->GetMaterilizedColumn( i + 1 ) );

        index = std::make_shared< AriesTableKeys >();
        return index->Build( columns, checkDuplicate, m_perDeltaTableBlockCapacity );
    }
    bool AriesMvccTable::CreateKeyIndex( const vector< int > &keyColIds,
                                         AriesTableKeysSPtr &index,
                                         const uint32_t startBlockIndex,
                                         const uint32_t startBlockLineIndex,
                                         bool checkDuplicate )
    {
        assert(index == nullptr);

        auto table = m_InitialTableSPtr->GetTable(keyColIds, startBlockIndex, startBlockLineIndex );

        vector< AriesColumnSPtr > columns;
        for( size_t i = 0; i < keyColIds.size(); ++i )
            columns.push_back( table->GetMaterilizedColumn( i + 1 ) );

        index = std::make_shared< AriesTableKeys >();
        return index->Build( columns, checkDuplicate, m_perDeltaTableBlockCapacity );
    }

    void AriesMvccTable::RebuildPrimaryKeyIndex()
    {
        bool bRet = true;
        if( HasPrimaryKey() )
        {
#ifdef ARIES_PROFILE
            aries::CPU_Timer t;
            t.begin();
#endif
            m_primaryKey.index = nullptr;
            bRet = CreateKeyIndex( m_primaryKey.columnIds, m_primaryKey.index, true );
#ifdef ARIES_PROFILE
            LOG( INFO )<< "RebuildPrimaryKeyIndex time: " << t.end();
#endif
            m_allIndexKeysChecked = true;
        }
        if( !bRet )
        {
            ARIES_EXCEPTION_SIMPLE( ER_DUP_ENTRY, "Duplicate entry for key" );
        }
    }

    void AriesMvccTable::CreatePrimaryKeyIndexIfNotExists()
    {
        bool bRet = true;
        if( HasPrimaryKey() && !m_allIndexKeysChecked )
        {
            bool expectedValue = false;
            if( m_createKeyIndexInprogress.compare_exchange_strong(expectedValue, true) )
            {
#ifdef ARIES_PROFILE
                aries::CPU_Timer t;
                t.begin();
#endif
                bRet = CreateKeyIndex( m_primaryKey.columnIds, m_primaryKey.index, true );
#ifdef ARIES_PROFILE
                LOG( INFO )<< "CreatePrimaryKeyIndexIfNotExists time: " << t.end();
#endif
                {
                    unique_lock<mutex> lock(m_lock4CreateKeyIndex);
                    m_allIndexKeysChecked = true;
                }
                m_statusCond.notify_all();
            }
            else
            {
                unique_lock<mutex> lock(m_lock4CreateKeyIndex);
                m_statusCond.wait(lock,
                                  [&] { return m_allIndexKeysChecked; });
            }

        }
        if( !bRet )
        {
            ARIES_EXCEPTION_SIMPLE( ER_DUP_ENTRY, "Duplicate entry for key" );
        }
    }

    void AriesMvccTable::CreateIncrementPrimaryKeyIndexIfNotExists(
             const uint32_t startBlockIndex,
             const uint32_t startBlockLineIndex )
    {
        bool bRet = true;
        if( HasPrimaryKey() && !m_allIndexKeysChecked )
        {
            bool expectedValue = false;
            if( m_createKeyIndexInprogress.compare_exchange_strong(expectedValue, true) )
            {
#ifdef ARIES_PROFILE
                aries::CPU_Timer t;
                t.begin();
#endif
                bRet = CreateKeyIndex( m_primaryKey.columnIds, m_primaryKey.index, startBlockIndex, startBlockLineIndex, true );
#ifdef ARIES_PROFILE
                LOG( INFO )<< "CreatePrimaryKeyIndexIfNotExists time: " << t.end();
#endif
                {
                    unique_lock<mutex> lock(m_lock4CreateKeyIndex);
                    m_allIndexKeysChecked = true;
                }
                m_statusCond.notify_all();
            }
            else
            {
                unique_lock<mutex> lock(m_lock4CreateKeyIndex);
                m_statusCond.wait(lock,
                                  [&] { return m_allIndexKeysChecked; });
            }

        }
        if( !bRet )
        {
            ARIES_EXCEPTION_SIMPLE( ER_DUP_ENTRY, "Duplicate entry for key" );
        }
    }

    AriesPrimaryKeyInfo AriesMvccTable::GetPrimaryKeyInfo( AriesTransactionPtr &transaction )
    {
        if( HasPrimaryKey() )
        {
            CreateKeyIndex( m_primaryKey.columnIds, m_primaryKey.index, true );
        }
        return m_primaryKey;
    }

    AriesTableKeysSPtr AriesMvccTable::GetPrimaryKey()
    {
        return m_primaryKey.index;
    }

    bool AriesMvccTable::HasPrimaryKey() const
    {
        return !m_primaryKey.columnIds.empty();
    }

END_ARIES_ENGINE_NAMESPACE
