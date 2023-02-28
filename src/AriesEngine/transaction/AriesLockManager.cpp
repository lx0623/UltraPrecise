//
// Created by david.shen on 2020/3/16.
//

#include "AriesLockManager.h"
#include "AriesAssert.h"

BEGIN_ARIES_ENGINE_NAMESPACE
    AriesLockManager::AriesLockManager(uint64_t initialTableRowCount, uint64_t deltaTableRowCount):
    m_initialTableMuxtexes{initialTableRowCount}, m_deltaTableMutexes{deltaTableRowCount}
    {
        m_initialTableEndPtr = m_initialTableMuxtexes.end();
    }

    mutexSPtr AriesLockManager::getLock(RowPos rowPos)
    {
        mutexSPtr lockSPtr = nullptr;
        if (rowPos < 0)
        {
            ARIES_ASSERT(( size_t )( -rowPos ) <= m_initialTableMuxtexes.size(), "rowPos :" + to_string(rowPos) + " is out of m_initialTableMuxtexes range: " + to_string(m_initialTableMuxtexes.size()));
            lockSPtr = m_initialTableEndPtr[rowPos];
            if (lockSPtr == nullptr) {
                unique_lock<mutex> mutexCreatingLock{m_mutexForCreateMutex};
                lockSPtr = m_initialTableEndPtr[rowPos];
                if (lockSPtr == nullptr)
                {
                    lockSPtr = m_initialTableEndPtr[rowPos] = make_shared<mutex>();
                }
            }
        }
        else
        {
            ARIES_ASSERT(( size_t )rowPos <= m_deltaTableMutexes.size(), "rowPos :" + to_string(rowPos) + " is out of m_deltaTableMutexes range: " + to_string(m_deltaTableMutexes.size()));
            rowPos = ADJUST_ROWID(rowPos);
            lockSPtr = m_deltaTableMutexes[rowPos];
            if (lockSPtr == nullptr) {
                unique_lock<mutex> mutexCreatingLock{m_mutexForCreateMutex};
                lockSPtr = m_deltaTableMutexes[rowPos];
                if (lockSPtr == nullptr)
                {
                    lockSPtr = m_deltaTableMutexes[rowPos] = make_shared<mutex>();
                }
            }
        }
        return lockSPtr;
    }

    void AriesLockManager::Lock(RowPos rowPos)
    {
        ARIES_ASSERT(rowPos != 0, "bad rowPos: 0");
        mutexSPtr lockSPtr = getLock(rowPos);
        lockSPtr->lock();
    }

    bool AriesLockManager::TryLock(RowPos rowPos)
    {
        ARIES_ASSERT(rowPos != 0, "bad rowPos: 0");
        mutexSPtr lockSPtr = getLock(rowPos);
        return lockSPtr->try_lock();
    }

    void AriesLockManager::UnLock(RowPos rowPos)
    {
        ARIES_ASSERT(rowPos != 0, "bad rowPos: 0");
        mutexSPtr lockSPtr = nullptr;
        if (rowPos < 0)
        {
            ARIES_ASSERT(( size_t )( -rowPos ) <= m_initialTableMuxtexes.size(), "rowPos :" + to_string(rowPos) + " is out of m_initialTableMuxtexes range: " + to_string(m_initialTableMuxtexes.size()));
            lockSPtr = m_initialTableEndPtr[rowPos];
        }
        else
        {
            ARIES_ASSERT(( size_t )rowPos <= m_deltaTableMutexes.size(), "rowPos :" + to_string(rowPos) + " is out of m_deltaTableMutexes range: " + to_string(m_deltaTableMutexes.size()));
            rowPos = ADJUST_ROWID(rowPos);
            lockSPtr = m_deltaTableMutexes[rowPos];
        }
        if (lockSPtr)
        {
            lockSPtr->unlock();
        }
    }

    void AriesLockManager::AddLock(RowPos newPos, RowPos oldPos)
    {
        ARIES_ASSERT(oldPos && newPos, "bad oldPos or newPos, oldPos: " + to_string(oldPos) + ", newPos: " + to_string(newPos));
        mutexSPtr lockSPtr = nullptr;
        if (oldPos < 0)
        {
            ARIES_ASSERT(( size_t )( -oldPos ) <= m_initialTableMuxtexes.size(), "oldPos :" + to_string(oldPos) + " is out of m_initialTableMuxtexes range: " + to_string(m_initialTableMuxtexes.size()));
            lockSPtr = m_initialTableEndPtr[oldPos];
        }
        else
        {
            ARIES_ASSERT(( size_t )oldPos <= m_deltaTableMutexes.size(), "oldPos :" + to_string(oldPos) + " is out of m_deltaTableMutexes range: " + to_string(m_deltaTableMutexes.size()));
            oldPos = ADJUST_ROWID(oldPos);
            lockSPtr = m_deltaTableMutexes[oldPos];
        }
        ARIES_ASSERT(newPos > 0, "bad newPos :" + to_string(newPos));
        ARIES_ASSERT(( size_t )newPos <= m_deltaTableMutexes.size(), "newPos :" + to_string(newPos) + " is out of m_deltaTableMutexes range: " + to_string(m_deltaTableMutexes.size()));
        newPos = ADJUST_ROWID(newPos);
        m_deltaTableMutexes[newPos] = lockSPtr;
    }

    void AriesLockManager::RemoveLock(RowPos pos)
    {
        ARIES_ASSERT(pos != 0, "bad pos: 0");
        if (pos < 0)
        {
            ARIES_ASSERT(( size_t )( -pos ) <= m_initialTableMuxtexes.size(), "pos :" + to_string(pos) + " is out of m_initialTableMuxtexes range: " + to_string(m_initialTableMuxtexes.size()));
            m_initialTableEndPtr[pos] = nullptr;
        }
        else
        {
            ARIES_ASSERT(( size_t )pos <= m_deltaTableMutexes.size(), "pos :" + to_string(pos) + " is out of m_deltaTableMutexes range: " + to_string(m_deltaTableMutexes.size()));
            pos = ADJUST_ROWID(pos);
            m_deltaTableMutexes[pos] = nullptr;
        }
    }

END_ARIES_ENGINE_NAMESPACE
