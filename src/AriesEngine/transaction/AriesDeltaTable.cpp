/*! 
  @file AriesDeltaTable.cpp
*/

#include "AriesDeltaTable.h"
#include "AriesVacuum.h"
#include "utils/utils.h"
#include "AriesDeltaTableProperty.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesDeltaTableBlock::AriesDeltaTableBlock( int32_t totalSlotCount,
                                                const std::vector< AriesColumnType >& column_types,
                                                int32_t blockIndex ):
    m_total( totalSlotCount ), m_blockIndex( blockIndex ), m_availableCount( m_total ), m_availableStart( 0 )
    {
        assert( totalSlotCount > 0 && totalSlotCount < INT32_MAX );

        size_t total_column_size = sizeof(TupleHeader);

        for ( const auto& type : column_types )
        {
            total_column_size += type.GetDataTypeSize();
        }

        AriesDeltaTableProperty::GetInstance().TryToAddDeltaTableUsedMemory( totalSlotCount * total_column_size );

        auto buffer = new int8_t[ size_t( totalSlotCount * total_column_size ) ];
        m_buffer.reset( buffer );
        m_header = reinterpret_cast< TupleHeader* >( buffer );

        size_t offset = sizeof( TupleHeader ) * size_t( totalSlotCount );

        for ( size_t i = 0; i < column_types.size(); i++ )
        {
            m_columns.emplace_back( buffer + offset );
            offset += column_types[ i ].GetDataTypeSize() * size_t( totalSlotCount );
        }

        m_usedFlag.resize( totalSlotCount );
        m_publishedFlag.resize( totalSlotCount );
    }

    //获取count数量的Slot，返回的slot被标记为已占用，这批Slot存储的内容对其他线程不可见
    //返回empty的vector表示空间不足
    vector< RowPos > AriesDeltaTableBlock::ReserveSlot( int32_t count )
    {
        bool unused;
        return ReserveSlot( count, unused );
    }

    //获取count数量的Slot，返回的slot被标记为已占用，这批Slot存储的内容对其他线程不可见
    //返回empty的vector表示空间不足
    //isContinuous 表示返回的 slot 是否一整块连续空间
    vector< RowPos > AriesDeltaTableBlock::ReserveSlot( int32_t count, bool& isContinuous )
    {
        vector< RowPos > result;
        //! @TODO: 如果 count 大于 m_total,会调用一次sweep，导致性能下降
        //剩余空间不足，先进行一次清理
        if( count > m_availableCount )
        {
            m_recycledSlotIndex = Sweep();
            if(count > 1)
                isContinuous = false;
            else
                isContinuous = true;
        }
        else
        {
            isContinuous = true;
        }

        //可以为外界提供的free slot数量
        int32_t totalCount = std::min( count, m_availableCount );

        if( totalCount > 0 )
        {
            //从recycle区域获取slot
            int32_t slotCountFromRecycled = std::min( totalCount, (int32_t)m_recycledSlotIndex.size() );
            if( slotCountFromRecycled > 0 )
            {
                int32_t startPos = m_recycledSlotIndex.size() - slotCountFromRecycled;
                for( size_t i = startPos; i < m_recycledSlotIndex.size(); ++i )
                {
                    int32_t index = m_recycledSlotIndex[ i ];
                    assert( !m_usedFlag[ index ] );
                    m_usedFlag[ index ] = true; //表示占用
                    m_publishedFlag[ index ] = false; //设置未发布
                    result.push_back( index + 1 + m_blockIndex * m_total ); //RowPos > 1
                }
                totalCount -= slotCountFromRecycled;
                m_availableCount -= slotCountFromRecycled;
                m_recycledSlotIndex.erase( m_recycledSlotIndex.begin() + startPos, m_recycledSlotIndex.end() );
            }
            
            //获取剩下的slot
            while( totalCount > 0 )
            {
                if( !m_usedFlag[ m_availableStart ] )
                {
                    m_usedFlag[ m_availableStart ] = true; //表示占用
                    m_publishedFlag[ m_availableStart ] = false; //设置未发布
                    result.push_back( m_availableStart + 1 + m_blockIndex * m_total ); //RowPos > 1
                    --m_availableCount;
                    --totalCount;
                }
               ++m_availableStart;
                m_availableStart = m_availableStart % m_total;
            }
        }
        
        return result;
    }

    vector< int32_t > AriesDeltaTableBlock::Sweep()
    {
        vector< int32_t > deadIndex;
        AriesTransManager& transManager = AriesTransManager::GetInstance();
        TxId freezedId = transManager.GetFreezeTxId();
        assert( freezedId != INVALID_TX_ID );
        
        vector< int32_t > slots;
        for( int i = 0; i < m_total; ++i )
        {
            if( m_usedFlag[ i ] && m_publishedFlag[ i ] )
                slots.push_back( i );
        }
        
        TxId firstAlive = INT_MAX;
        for( auto slot : slots )
        {
            TupleHeader* header = m_header + slot;
            TxId xMax = header->m_xmax;
            //以下两种情况视为dead tuple
            //1.m_deadFlag == true
            //2.并非是delete列存区的数据,并且xMax有效且对应的事务在当前所有事务的快照里已经committed
            if( header->m_deadFlag
                    || ( header->m_xmin != INVALID_TX_ID && xMax != INVALID_TX_ID && xMax < freezedId && transManager.GetTxStatus( xMax ) == TransactionStatus::COMMITTED ) )
                deadIndex.push_back( slot );
            else
                firstAlive = std::min( firstAlive, header->m_xmin );                //在活着的tuple里，找到最小的xmin
        }

        for( auto index : deadIndex )
        {
            assert( m_usedFlag[ index ] );
            assert( m_publishedFlag[ index ] );
            //设置为未占用
            m_usedFlag[ index ] = false;
            ++m_availableCount;
        }
        
        return deadIndex;
    }

    void AriesDeltaTableBlock::FreeSlot( int32_t slot )
    {
        m_usedFlag[ slot ] = false;
        m_availableCount++;
    }

    void AriesDeltaTableBlock::CompleteSlot( int32_t slot )
    {
        assert( m_usedFlag[ slot ] );
        assert( !m_publishedFlag[ slot ] );
        m_publishedFlag[ slot ] = true;
    }

    AriesDeltaTableBlock::~AriesDeltaTableBlock()
    {
    }

    AriesDeltaTable::AriesDeltaTable( int32_t perBlockSlotCount, const std::vector< AriesColumnType >& types )
            : m_perBlockSlotCount( perBlockSlotCount ), m_columnTypes( types ), m_currentAddedStartPos( 0 ), m_currentDeletingStartPos( 0 )
    {
        assert( perBlockSlotCount > 0 );
        m_addedBlocks.emplace_back( std::make_unique< AriesDeltaTableBlock >( perBlockSlotCount, types, int32_t( m_addedBlocks.size() ) ) );
        m_deletingBlocks.emplace_back( std::make_unique< AriesDeltaTableBlock >( perBlockSlotCount, std::vector< AriesColumnType >{}, int32_t( m_deletingBlocks.size() ) ) );
    }

    vector< RowPos > AriesDeltaTable::ReserveSlot( int32_t count, AriesDeltaTableSlotType slotType, bool& isContinuous )
    {
        vector< RowPos > result;
        switch( slotType )
        {
            case AriesDeltaTableSlotType::AddedTuples:
            {
                unique_lock< mutex > lock( m_mutexForAddedBlocks );
                result = ReserveSlotInternal( count, m_columnTypes, m_addedBlocks, m_currentAddedStartPos, isContinuous );
                break;
            }
            case AriesDeltaTableSlotType::DeletedInitialTableTuples:
            {
                unique_lock< mutex > lock( m_mutexForDeletingBlocks );
                result = ReserveSlotInternal( count, {}, m_deletingBlocks, m_currentDeletingStartPos, isContinuous );
                break;
            }
            default:
                ARIES_ASSERT( 0, "unknown slot type" );
                break;
        }
        return result;
    }

    vector< RowPos > AriesDeltaTable::ReserveSlotInternal( int32_t count, const vector< AriesColumnType >& columnTypes, 
                                                            std::vector< std::unique_ptr< AriesDeltaTableBlock > >& blocks, uint64_t& startPos, bool& isContinuous )
    {
        assert( count > 0 );
        if( ( size_t )count > MAX_BLOCK_COUNT * m_perBlockSlotCount )
            ARIES_EXCEPTION( ER_ENGINE_OUT_OF_MEMORY, "rateup" );

        int32_t leftCount = count;

        //从当前block获取空闲的slots
        vector< RowPos > result = blocks[ startPos % MAX_BLOCK_COUNT ]->ReserveSlot( leftCount, isContinuous );
        leftCount -= result.size();
        size_t scannedBlockCount = 1;//表示扫描了１个block.每扫描一个不同的block,就+1.

        //增加新block
        while( leftCount > 0 && blocks.size() < MAX_BLOCK_COUNT )
        {
            blocks.emplace_back( std::make_unique< AriesDeltaTableBlock >( m_perBlockSlotCount, columnTypes, int32_t( blocks.size() ) ) );
            auto tmp = blocks[ ++startPos % MAX_BLOCK_COUNT ]->ReserveSlot( leftCount, isContinuous );
            ++scannedBlockCount;
            leftCount -= tmp.size();
            result.insert( result.end(), tmp.begin(), tmp.end() ); 
            isContinuous = false;
        }

        //无法再增加新block．尝试扫描剩下的block
        while( leftCount > 0 && scannedBlockCount < MAX_BLOCK_COUNT )
        {
            auto tmp = blocks[ ++startPos % MAX_BLOCK_COUNT ]->ReserveSlot( leftCount, isContinuous );
            ++scannedBlockCount;
            if( !tmp.empty() )
            {
                leftCount -= tmp.size();
                result.insert( result.end(), tmp.begin(), tmp.end() ); 
            }
            isContinuous = false;
        }

        if( leftCount > 0 )
        {
            Vacuum();
        }

        if( count == 1 )
            isContinuous = true;

        return result;
    }

    void AriesDeltaTable::FreeSlot( const vector< RowPos >& slots, AriesDeltaTableSlotType slotType )
    {
        switch( slotType )
        {
            case AriesDeltaTableSlotType::AddedTuples:
            {
                lock_guard< mutex > lock( m_mutexForAddedBlocks );
                FreeSlotInternal( slots, m_addedBlocks );
                break;
            }
            case AriesDeltaTableSlotType::DeletedInitialTableTuples:
            {
                lock_guard< mutex > lock( m_mutexForDeletingBlocks );
                FreeSlotInternal( slots, m_deletingBlocks );
                break;
            }
            default:
                ARIES_ASSERT( 0, "unknown slot type" );
                break;
        }
    }

    void AriesDeltaTable::FreeSlotInternal( const vector< RowPos >& slots, std::vector< std::unique_ptr< AriesDeltaTableBlock > >& blocks )
    {
        for( auto slot : slots )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * blocks.size() );

            slot -= 1;
            auto blockIndex = slot / m_perBlockSlotCount;
            auto slotInBlock = slot % m_perBlockSlotCount;
            
            blocks[ blockIndex ]->FreeSlot( slotInBlock );
        }
    }

    void AriesDeltaTable::CompleteSlot( const vector< RowPos >& slots, AriesDeltaTableSlotType slotType )
    {
        switch( slotType )
        {
            case AriesDeltaTableSlotType::AddedTuples:
            {
                lock_guard< mutex > lock( m_mutexForAddedBlocks );
                CompleteSlotInternal( slots, m_addedBlocks );
                break;
            }
            case AriesDeltaTableSlotType::DeletedInitialTableTuples:
            {
                lock_guard< mutex > lock( m_mutexForDeletingBlocks );
                CompleteSlotInternal( slots, m_deletingBlocks );
                break;
            }
            default:
                ARIES_ASSERT( 0, "unknown slot type" );
                break;
        }
    }

    void AriesDeltaTable::CompleteSlotInternal( const vector< RowPos >& slots, std::vector< std::unique_ptr< AriesDeltaTableBlock > >& blocks )
    {
        for( auto slot : slots )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * blocks.size() );
            slot -= 1;
            auto blockIndex = slot / m_perBlockSlotCount;
            auto slotInBlock = slot % m_perBlockSlotCount;
            
            blocks[ blockIndex ]->CompleteSlot( slotInBlock );
        }
    }

    void AriesDeltaTable::GetVisibleRowIdsInternal( TxId txId, const Snapshot &snapShot,
                                                    const vector< int32_t > &visibleSlots,
                                                    const vector< int32_t > &visibleSlotsForDeleting,
                                                    vector< RowPos > &visibleIds,
                                                    vector< RowPos > &initialIds )
    {
        AriesTransManager &transManager = AriesTransManager::GetInstance();
        for ( auto slot : visibleSlots )
        {
            auto blockIndex = slot / m_perBlockSlotCount;
            auto slotInBlock = slot % m_perBlockSlotCount;
            TupleHeader *header = m_addedBlocks[ blockIndex ]->m_header + slotInBlock;
            if ( !header->m_deadFlag )
            {
                TxId xMin = header->m_xmin;
                TxId xMax = header->m_xmax;
                bool bVisible = false;

                switch ( transManager.GetTxStatus( xMin ) )
                {
                    case TransactionStatus::ABORTED: {
                        header->m_deadFlag = true;
                        break;
                    }
                    case TransactionStatus::IN_PROGRESS: {
                        if ( xMin == txId )
                        {
                            if ( xMax == INVALID_TX_ID )
                                bVisible = true;
                        }
                        break;
                    }
                    case TransactionStatus::COMMITTED: {
                        if ( !snapShot.IsTxActive( xMin ) )
                        {
                            if ( xMax == INVALID_TX_ID )
                                bVisible = true;
                            else
                            {
                                switch ( transManager.GetTxStatus( xMax ) )
                                {
                                case TransactionStatus::ABORTED: {
                                    bVisible = true;
                                    break;
                                }
                                case TransactionStatus::IN_PROGRESS: {
                                    if ( xMax != txId )
                                        bVisible = true;
                                    break;
                                }
                                case TransactionStatus::COMMITTED: {
                                    if ( snapShot.IsTxActive( xMax ) )
                                        bVisible = true;
                                    break;
                                }
                                }
                            }
                        }
                        break;
                    }

                }
                if ( bVisible )
                {
                    visibleIds.push_back( slot + 1 );
                }
            }
        }

        for ( auto slot : visibleSlotsForDeleting )
        {
            auto blockIndex = slot / m_perBlockSlotCount;
            auto slotInBlock = slot % m_perBlockSlotCount;
            TupleHeader *header = m_deletingBlocks[ blockIndex ]->m_header + slotInBlock;
            if ( header->m_deadFlag )
            {
                continue;
            }

            TxId xMin = header->m_xmin;
            TxId xMax = header->m_xmax;

            ARIES_ASSERT( xMin == INVALID_TX_ID, "invalid status of deleting tuple" );

            ARIES_ASSERT( header->m_initialRowPos < 0, "invalid initialRowPos" );

            bool bDeleteOpValid = false;
            //对于delete产生的fake数据，xMin为INVALID_TX_ID，xMax为插入此条数据的transaction id
            switch ( transManager.GetTxStatus( xMax ) )
            {
                case TransactionStatus::ABORTED: {
                    header->m_deadFlag = true;
                    break;
                }
                case TransactionStatus::IN_PROGRESS: {
                    //是自己，删除有效
                    if ( xMax == txId )
                        bDeleteOpValid = true;
                    break;
                }
                case TransactionStatus::COMMITTED: {
                    //事务已提交，并且不是active，删除有效
                    bDeleteOpValid = !snapShot.IsTxActive( xMax );
                    break;
                }
            }
            if ( bDeleteOpValid )
                initialIds.push_back( header->m_initialRowPos );
        }
    }

    void AriesDeltaTable::GetVisibleRowIdsInDeltaTable( TxId txId, const Snapshot& snapShot, vector< RowPos >& visibleIds,
            vector< RowPos >& initialIds )
    {
        visibleIds.clear();
        initialIds.clear();

        vector< int32_t > publishedSlots;
        vector< int32_t >  publishedSlotsForDeleting;
        GetPublishedSlots( publishedSlots, publishedSlotsForDeleting );
        GetVisibleRowIdsInternal( txId, snapShot, publishedSlots, publishedSlotsForDeleting, visibleIds, initialIds );
    }

    void AriesDeltaTable::GetPublishedSlots( std::vector< int32_t >& publishedSlots, std::vector< int32_t >& publishedSlotsForDeleting )
    {
        {
            lock_guard< mutex > lock( m_mutexForAddedBlocks );
            for ( const auto& block : m_addedBlocks )
            {
                auto start = block->m_blockIndex * block->m_total;
                for( int i = 0; i < block->m_total; ++i )
                {
                    if( block->m_usedFlag[ i ] && block->m_publishedFlag[ i ] )
                        publishedSlots.push_back( i + start  );
                }
            }
        }
        
        {
            lock_guard< mutex > lock( m_mutexForDeletingBlocks );
            for ( const auto& block : m_deletingBlocks )
            {
                auto start = block->m_blockIndex * block->m_total;
                for( int i = 0; i < block->m_total; ++i )
                {
                    if( block->m_usedFlag[ i ] && block->m_publishedFlag[ i ] )
                        publishedSlotsForDeleting.push_back( i + start );
                }
            }
        }
    }

    void AriesDeltaTable::GetTupleFieldBuffer( RowPos pos, std::vector< int8_t* >& columnBuffers, std::vector< int > columnsId )
    {
        assert( pos > 0 && ( size_t )pos <= m_perBlockSlotCount * m_addedBlocks.size() );
        int32_t slot = pos - 1;
        assert( IsSlotUsed( slot, AriesDeltaTableSlotType::AddedTuples ) );

        if ( columnBuffers.empty() )
        {
            return;
        }

        if ( columnsId.empty() )
        {
            columnsId.resize( columnBuffers.size() );
            std::iota( columnsId.begin(), columnsId.end(), 1 );
        }

        
        auto blockIndex = slot / m_perBlockSlotCount;
        auto slotInBlock = slot % m_perBlockSlotCount;

        for ( size_t i = 0; i < columnsId.size(); i++ )
        {
            auto id = columnsId[ i ];
            const auto& type = m_columnTypes[ id - 1 ];
            const auto& buffer = m_addedBlocks[ blockIndex ]->m_columns[ id - 1 ];

            columnBuffers[ i ] = buffer + ( slotInBlock * type.GetDataTypeSize() );
        }
    }

    void AriesDeltaTable::Vacuum()
    {
        AriesVacuum::GetInstance().FlushAndReloadDb();
        ARIES_EXCEPTION( ER_XA_RBROLLBACK );
    }

END_ARIES_ENGINE_NAMESPACE
