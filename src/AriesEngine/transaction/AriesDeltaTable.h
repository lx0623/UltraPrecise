/*! 
  @file AriesDeltaTable.h
  @brief DB 运行期间表数据变化的部分
  @details 
 
 ## delta table
 数据库中表的数据分为两部分
    - initial table：数据库启动时的数据，这部分数据在运行期不变
    - delta table：运行期发生变化的tuple。


 ## delta table block
 delta table 由一组 delta_table_block 组成，block 分为两类
    - 新增型:新插入的tuple
    - 删除型:记录initial table中被删除的tuple rowPos

 初始状态会为每个delta table创建一个block。当block使用完了，会添加新的block。
 
 ## slot
 delta block 是 slot 数组。
 - slot for insertion：保存一条新增tuple，以及update后生成的新tuple
 - slot for deletion：保存initial table中被删除tuple的rowPos

 ### slot 的生命周期
```

    [A]没有被使用 --> [B]分配给某个trx正在写 ---> [C]当前sql执行成功--->[D]dead slot
                              |
                              |---> [A]trx写失败 used=0
    
```

A 和 C状态对其他trx可见，参与mvcc计算。B状态不能参与mvcc计算，因为B状态下slot包含随机数据，可能是已经被删除的数据。

 ### slot 状态
    used 和 published 表示一个slot在一个sql执行中的状态. 只有当used=1时 published 才有意义.
    dead 表示trx结束后slot的状态

    
    - A: used=0
    - B: used=1 published=0
    - C: used=1 published=1
    - D: dead=true 被sweep回收的slot，可重用
    
    可用slot：处于 状态A 和状态D 的 slot
    
### slot 状态转化：

    - A->B: ReserveSlot()
    - B->C: CompleteSlot()
    - B->A: FreeSlot()
    - C->D: Sweep()
    
## rowPos 和 rowId
为了区分tuple来自initial table 还是 delta table，定义rowPos如下
- 负数表示initial table中的slot，即initial table中第 n 个slot的 ```rowPos=-n-1```
- 正数 表示delta table中的slot， 即delta table 中第 n 个slot的```rowPos=n+1```
- 0 是无效的rowPos

rowId为 0 base的非负整数序列，用于定位经过mvcc筛选后的可见tuple
*/

#pragma once

#include "AriesTuple.h"
#include "AriesTransManager.h"
#include "AriesLockManager.h"

#define MAX_BLOCK_COUNT UINT32_MAX //!< delta table 可包含的最大block数量

BEGIN_ARIES_ENGINE_NAMESPACE
/*!
    @brief Delta table slot类型
*/
enum class AriesDeltaTableSlotType : int32_t
{
    AddedTuples,//!<存储新增的tuple数据
    DeletedInitialTableTuples//!<存储对initial table删除的RowPos
};

class AriesDeltaTableBlock
{
public:
    AriesDeltaTableBlock( int32_t totalSlotCount, const std::vector< AriesColumnType >& column_types, int32_t index );
    ~AriesDeltaTableBlock();
    /*!
     @brief 获取count个空闲的Slot，used=false 或 dead = true 的slot
     @param[in] count 需要申请的可用slot个数
     @retval 申请到的slot集合，如果返回空，表示没有足够的空闲slot
     @note slot的used 设置为1，published设置为0
    */
    vector< RowPos > ReserveSlot( int32_t count );
    /*!
     @brief 获取count个可用的Slot
     @param[in] count 需要申请的空闲slot个数
     @param[out] isContinuous 返回的slot是否是连续的数组  
     @return
         @retval 空vec，表示没有足够的空闲slot
         @retval 非空vec，申请到的slot集合
     @note 
       - slot的used 设置为1，published设置为0
       - 即使有部分slot是连续的，isContinuous也返回false
    */
    vector< RowPos > ReserveSlot( int32_t count, bool& isContinuous );
    /*!
     @brief sql执行失败时调用，将slot归还给delta table，允许其他trx使用
     @param[in] 归还的slot rowId
     @return
     @note slot 的 used 设置为0，但出于效率考虑没有修改published，因此 published仍然是 1.
    */
    void FreeSlot( int32_t slot );
    /*!
     @brief sql 成功执行后调用， slot的used设置为1，published 设置为1.
     @param[in] trx 提交的slot
     @return 
     @note 
    */
    void CompleteSlot( int32_t slot );

    friend class AriesDeltaTable;
private:
    /*!
     @brief 回收 abort 的 trx占用的slot
     @param 
     @return 
     @note 修改slot的header.dead flag，增加 block.m_availableCount
    */
    vector< int32_t > Sweep();

private:
    std::unique_ptr< int8_t[] > m_buffer; //!<总共分配的空间 = header + columns
    std::vector< int8_t* > m_columns;//!<column头指针列表
    TupleHeader* m_header;//!<header区头指针

    int32_t m_total; //!< block 中的slot数
    int32_t m_blockIndex; //!< block 在delta table中的id
    int32_t m_availableCount; //!< 可用的slot数 = dead为true的slot + used=false 的slot
    RowPos m_availableStart;  //!< 第一个unused的slot 位置

    vector< bool > m_usedFlag; //!< 表示slot使用情况的标志位，true:占用, false:空闲
    vector< bool > m_publishedFlag; //!< 表示slot的内容是否对外界发布标志位，true:已发布，false:未发布.未发布的不会参与mvcc扫描
    vector< int32_t > m_recycledSlotIndex;//!< sweep() 函数回收的slot index, 0 based.
};

/*!
@brief 管理一个表的多个delta table block
*/
    class AriesDeltaTable
    {
    public:
        AriesDeltaTable( int32_t perBlockSlotCount, const std::vector< AriesColumnType >& types = std::vector< AriesColumnType >() );

        /*！
        @brief 获取count个可用Slot
        @param[in] count 申请的slot 个数
        @return
            @reval vec - 申请到的slot，如果vec empty, 表示没有足够slot
            
        @note 
            返回的slot被标记为已占用，这批Slot存储的内容对其他线程不可见
        */
        //vector< RowPos > ReserveSlot( int32_t count );

        /*！
        @brief 获取count个可用Slot
        @param[in] count 申请的slot 个数
        @param[in] slotType 申请的slot所属区域
        @param[out] isContinuous 分配的slot是否连续
        @return
            @reval vec - 申请到的slot，如果vec empty, 表示没有足够slot
            
        @note 
            返回的slot被标记为已占用，这批Slot存储的内容对其他线程不可见
        */
        vector< RowPos > ReserveSlot( int32_t count, AriesDeltaTableSlotType slotType, bool& isContinuous );

        /*!
        @brief 放弃对slot的所有修改,并标记对应的slot为未占用状态，此时slot存储的内容是无效的
        @param[in] slots 回收的slot
        @param[in] slotType slot所属区域
        @note 因为与其它transaction 冲突，当前trx的当前sql修改的slot会调用free，但被当前trx之前sql修改的slot不会调用free (通过sweep设置dead标记来回收)。
        */
        void FreeSlot( const vector< RowPos >& slots, AriesDeltaTableSlotType slotType );

        /*!
        @brief sql 执行完成后调用，将published 标记set为true。
        @param[in] slots 完成修改的slot
        @param[in] slotType slot所属区域
         */
        void CompleteSlot( const vector< RowPos >& slots, AriesDeltaTableSlotType slotType );

        /// @brief 设置slot header的t_xmax值
        inline void SetTxMax( RowPos slot, TxId value )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * m_addedBlocks.size() );
            TupleHeader* header = GetTupleHeader( slot, AriesDeltaTableSlotType::AddedTuples );
            header->m_xmax = value;
        }

        /// @brief 获取对应数据行的t_xmax值
        inline TxId GetTxMax( RowPos slot )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * m_addedBlocks.size() );
            TupleHeader* header = GetTupleHeader( slot, AriesDeltaTableSlotType::AddedTuples );
            return header->m_xmax;
        }

        //对某行加锁
        void Lock( RowPos slot )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * m_addedBlocks.size() );
            TupleHeader* header = GetTupleHeader( slot, AriesDeltaTableSlotType::AddedTuples );
            bool expected = false;
            while( !header->m_lockFlag.compare_exchange_weak( expected, true, std::memory_order_release, std::memory_order_relaxed ) )
                expected = false;
        }

        bool TryLock( RowPos slot )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * m_addedBlocks.size() );
            TupleHeader* header = GetTupleHeader( slot, AriesDeltaTableSlotType::AddedTuples );
            bool expected = false;
            return header->m_lockFlag.compare_exchange_strong( expected, true );
        }

        //对某行解锁
        void Unlock( RowPos slot )
        {
            assert( slot > 0 && ( size_t )slot <= m_perBlockSlotCount * m_addedBlocks.size() );
            TupleHeader* header = GetTupleHeader( slot, AriesDeltaTableSlotType::AddedTuples );
            bool expected = true;
            header->m_lockFlag.compare_exchange_strong( expected, false );
        }

        /*!
        @brief 获取Slot对应的存储区指针，以写入数据
        @note 调用者保证调用该函数前，必须先调用ReserveSlot，标记此Slot被占用, 只获取一个column的某一行数据时,可以调用此函数,更为方便
         */
        inline int8_t* GetTupleFieldBuffer( RowPos pos, int32_t columnID = 1 )
        {
            assert( pos > 0 && ( size_t )pos <= m_perBlockSlotCount * m_addedBlocks.size() );
            int32_t slot = pos - 1;
            assert( IsSlotUsed( slot, AriesDeltaTableSlotType::AddedTuples ) );
            auto blockIndex = slot / m_perBlockSlotCount;
            auto slotPosInBlock = slot % m_perBlockSlotCount;
            return m_addedBlocks[ blockIndex ]->m_columns[ columnID - 1 ] + slotPosInBlock * m_columnTypes[ columnID - 1 ].GetDataTypeSize();
        }

        /*!
        @brief 获取Slot对应的存储区指针，以写入数据
        @note 调用者保证调用该函数前，必须先调用ReserveSlot，标记此Slot被占用, 需要同时获取多个column的某一行数据时,需要调用此函数
         */
        void GetTupleFieldBuffer( RowPos pos, std::vector< int8_t* >& columnBuffers, std::vector< int > columnsId = {} );

        /*!
         @brief 通过rowPos获取TupleHeader
         @param[in] pos slot的rowPos
         @param[in] slotType slot所属区域
         */

        TupleHeader* GetTupleHeader( RowPos pos, AriesDeltaTableSlotType slotType )
        {
            assert( pos > 0 );
            int32_t slot = pos - 1;
            assert( IsSlotUsed( slot, slotType ) );
            TupleHeader* result = nullptr;

            auto blockIndex = slot / m_perBlockSlotCount;
            auto slotPosInBlock = slot % m_perBlockSlotCount;
            switch( slotType )
            {
                case AriesDeltaTableSlotType::AddedTuples:
                {
                    assert( ( size_t )slot < m_perBlockSlotCount * m_addedBlocks.size() );
                    result = m_addedBlocks[ blockIndex ]->m_header + slotPosInBlock;
                    break;
                }
                case AriesDeltaTableSlotType::DeletedInitialTableTuples:
                {
                    assert( ( size_t )slot < m_perBlockSlotCount * m_deletingBlocks.size() );
                    result = m_deletingBlocks[ blockIndex ]->m_header + slotPosInBlock;
                    break;
                }
                default:
                    ARIES_ASSERT( 0, "unknown slot type" );
                    break;
            }
            return result;
        }

        inline void SetColumnTypes( const vector< AriesColumnType >& types )
        {
            m_columnTypes.assign( types.cbegin(), types.cend() );
        }

        inline size_t GetDeltaTableSize() const
        {
            return m_perBlockSlotCount * m_addedBlocks.size();
        }

        inline std::vector< vector< int8_t* > > GetColumnBuffers() const
        {
            std::vector< vector< int8_t* > > columns_buffers;
            for ( const auto& block : m_addedBlocks )
            {
                columns_buffers.emplace_back( block->m_columns );
            }
            return columns_buffers;
        }

        /*!
        @brief 获取可见slot rowPos
        @param[out] visibleIds: 表示行存区可见的RowPos
        @param[out] initialIds: 表示列存区<em>不可见</em>的RowPos
        */
        void GetVisibleRowIdsInDeltaTable( TxId txId, const Snapshot& snapShot, vector< RowPos >& visibleIds, vector< RowPos >& initialIds );

        bool IsTuplePublished( RowPos pos )
        {
            assert( pos > 0 );
            lock_guard< mutex > lock( m_mutexForAddedBlocks );
            int32_t slotIndex = pos - 1;
            auto blockIndex = slotIndex / m_perBlockSlotCount;
            auto slotPosInBlock = slotIndex % m_perBlockSlotCount;
            return m_addedBlocks[ blockIndex ]->m_usedFlag[ slotPosInBlock ] & m_addedBlocks[ blockIndex ]->m_publishedFlag[ slotPosInBlock ];
        }

    private:
        bool IsSlotUsed( int32_t slotIndex, AriesDeltaTableSlotType slotType )
        {
            assert( slotIndex >= 0 );
            auto blockIndex = slotIndex / m_perBlockSlotCount;
            auto slotPosInBlock = slotIndex % m_perBlockSlotCount;
            bool ret = false;
            switch( slotType )
            {
                case AriesDeltaTableSlotType::AddedTuples:
                {
                    lock_guard< mutex > lock( m_mutexForAddedBlocks );
                    ret = m_addedBlocks[ blockIndex ]->m_usedFlag[ slotPosInBlock ];
                    break;
                }
                case AriesDeltaTableSlotType::DeletedInitialTableTuples:
                {
                    lock_guard< mutex > lock( m_mutexForDeletingBlocks );
                    ret = m_deletingBlocks[ blockIndex ]->m_usedFlag[ slotPosInBlock ];
                    break;
                }
                default:
                    ARIES_ASSERT( 0, "unknown slot type" );
                    break;
            }
            return ret;
        }

        /*! 
        @brief 从insert block和 delete block中找出 published slot的 rowId
        @param[out] publishedSlots - insert block中published的slot rowId
        @param[out] publishedSlotsForDeleting - delete block中published 的slot rowId
        */

        void GetPublishedSlots( std::vector< int32_t >& publishedSlots, std::vector< int32_t >& publishedSlotsForDeleting );

        void GetVisibleRowIdsInternal( TxId txId,
                                        const Snapshot &snapShot,
                                        const vector< int32_t > &publishedSlots,
                                        const vector< int32_t > &publishedSlotsForDeleting,
                                        vector< RowPos > &visibleIds,
                                        vector< RowPos > &initialIds );
        //! @brief invoke xlog recovery
        void Vacuum();

        vector< RowPos > ReserveSlotInternal( int32_t count, const vector< AriesColumnType >& columnTypes, std::vector< std::unique_ptr< AriesDeltaTableBlock > >& blocks, uint64_t& startPos, bool& isContinuous );

        void FreeSlotInternal( const vector< RowPos >& slots, std::vector< std::unique_ptr< AriesDeltaTableBlock > >& blocks );

        void CompleteSlotInternal( const vector< RowPos >& slots, std::vector< std::unique_ptr< AriesDeltaTableBlock > >& blocks );

    private:
        int32_t m_perBlockSlotCount; //!<每个block包含的slot数
        vector< AriesColumnType > m_columnTypes;

        std::vector< std::unique_ptr< AriesDeltaTableBlock > > m_addedBlocks;
        uint64_t m_currentAddedStartPos;
        mutex m_mutexForAddedBlocks;

        std::vector< std::unique_ptr< AriesDeltaTableBlock > > m_deletingBlocks;
        uint64_t m_currentDeletingStartPos;
        mutex m_mutexForDeletingBlocks;
    };

    using AriesDeltaTableSPtr = shared_ptr<AriesDeltaTable>;

END_ARIES_ENGINE_NAMESPACE
