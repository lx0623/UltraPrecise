//
// Created by david.shen on 2020/3/13.
//

#pragma once

#include "AriesTuple.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    using mutexSPtr = shared_ptr<mutex>;

#define ADJUST_ROWID(x) (x-1)
    class AriesLockManager
    {
    public:
        AriesLockManager( uint64_t initialTableRowCount, uint64_t deltaTableRowCount );
        void Lock( RowPos rowPos );
        bool TryLock( RowPos rowPos );
        void UnLock( RowPos rowPos );
        /**
         * addLock: 为newPos加上和oldPos一样的锁
         */
        void AddLock( RowPos newPos, RowPos oldPos );
        /**
         * addLock: 删除id中的锁,因为id是dead tuple将被收回, 锁也不要同步移除
         */
        void RemoveLock( RowPos pos );
    private:
        mutexSPtr getLock( RowPos rowPos );

    private:
        vector< mutexSPtr > m_initialTableMuxtexes;
        vector< mutexSPtr >::iterator m_initialTableEndPtr;
        vector< mutexSPtr > m_deltaTableMutexes;
        mutex m_mutexForCreateMutex; //只在新生成锁的时候才使用
    };

    using AriesLockManagerSPtr = shared_ptr<AriesLockManager>;

    class AriesLightweightLockManager
    {
    public:
        AriesLightweightLockManager()
                : m_initialTableRowCount( 0 ), m_deltaTableRowCount( 0 )
        {
            m_blockCount = 0;
        }

        void Initialize( int64_t initialTableRowCount, int64_t deltaTableRowCount, size_t maxBlockCount )
        {
            assert( m_pLocks.empty() && m_locksBuffer.empty() );
            m_initialTableRowCount = initialTableRowCount;
            m_deltaTableRowCount = deltaTableRowCount;
            m_locksBuffer.reserve( maxBlockCount );//reserve 一下，避免以后添加新block导致内存重新分配，这样就不用线程同步了
            m_pLocks.reserve( maxBlockCount );//reserve 一下，避免以后添加新block导致内存重新分配，这样就不用线程同步了
            auto lockBuf = make_shared< vector< int8_t > >();
            lockBuf->resize( initialTableRowCount + deltaTableRowCount + 1 );
            m_locksBuffer.push_back( lockBuf );
            m_pLocks.push_back( lockBuf->data() + initialTableRowCount );
            m_blockCount = m_locksBuffer.size();
        }

        void Lock( RowPos pos )
        {
            atomic< int8_t >* rowLock = reinterpret_cast< atomic< int8_t >* >( GetRowLock( pos ) );
            int8_t expected = 0;
            while( !rowLock->compare_exchange_weak( expected, 1, std::memory_order_release, std::memory_order_relaxed ) )
                expected = 0;
        }

        bool TryLock( RowPos pos )
        {
            atomic< int8_t >* rowLock = reinterpret_cast< atomic< int8_t >* >( GetRowLock( pos ) );
            int8_t expected = 0;
            return rowLock->compare_exchange_strong( expected, 1 );
        }

        void UnLock( RowPos pos )
        {
            atomic< int8_t >* rowLock = reinterpret_cast< atomic< int8_t >* >( GetRowLock( pos ) );
            int8_t expected = 1;
            rowLock->compare_exchange_strong( expected, 0 );
        }

        void AddLocksForDeltaTableBlock( int count )
        {
            assert( count == m_deltaTableRowCount );
            auto lockBuf = make_shared< vector< int8_t > >();
            lockBuf->resize( count );
            m_locksBuffer.push_back( lockBuf );
            m_pLocks.push_back( lockBuf->data() );
            m_blockCount = m_locksBuffer.size();
        }

    private:
        bool IsValidPos( RowPos pos ) const
        {
            return pos != 0 ? pos > 0 ? pos <= m_deltaTableRowCount * m_blockCount: -pos <= m_initialTableRowCount : false;
        }

        int8_t* GetRowLock( RowPos pos ) const 
        {
            assert( IsValidPos( pos ) );
            if( pos <= m_deltaTableRowCount )
                return m_pLocks[ 0 ] + pos;
            else
                return m_pLocks[ ( pos - 1 ) / m_deltaTableRowCount ] + ( pos - 1 ) % m_deltaTableRowCount;
        }

    private:
        int64_t m_initialTableRowCount;
        int64_t m_deltaTableRowCount;
        atomic< int64_t > m_blockCount;
        vector< int8_t* > m_pLocks;
        vector< shared_ptr< vector< int8_t > > > m_locksBuffer;
    };

    using AriesLightweightLockManagerSPtr = shared_ptr<AriesLightweightLockManager>;

END_ARIES_ENGINE_NAMESPACE
