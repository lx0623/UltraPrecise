#include <thread>
#include <signal.h>
#include "AriesVacuum.h"
#include "AriesTransManager.h"
#include "AriesMvccTableManager.h"
#include "AriesXLogManager.h"
#include "AriesXLogRecoveryer.h"
#include "AriesInitialTableManager.h"
#include "frontend/ViewManager.h"

extern THD *createPseudoThd();

BEGIN_ARIES_ENGINE_NAMESPACE

thread_local bool AriesVacuum::m_bIsVacuumThread = false;

AriesVacuum::AriesVacuum()
{
    m_bInProgress = false;
    m_currentDbVersion = 0;
}

AriesVacuum::~AriesVacuum()
{
}

void AriesVacuum::FlushAndReloadDb()
{
    bool expectedValue = false;
    if (m_bInProgress.compare_exchange_strong(expectedValue, true))
    {
        //increase db version first, so all running tx can check this value and abort accordingly.
        ++m_currentDbVersion;

        // do vacuum in another thread.
        std::thread t([&] 
        {
            try
            {
                m_bIsVacuumThread = true;

                //wait all transactions done except the ones wating io.
                //if the tx wakes up in future, the m_currentDbVersion will be different from the tx begin. the tx can abort properly later.
                AriesTransManager::GetInstance().WaitForAllTxEndOrIdle();

                //clear all mvcc tables
                AriesMvccTableManager::GetInstance().clearAll();

                //recover special( schema related )
                auto special_recoveryer = std::make_shared<AriesXLogRecoveryer>(true);
                special_recoveryer->SetReader(AriesXLogManager::GetInstance().GetReader(true));
                auto result = special_recoveryer->Recovery();
                ARIES_ASSERT(result, "cannot recovery(special) from xlog");

                //load schema and views
                THD *thd = createPseudoThd();
                aries::schema::SchemaManager::GetInstance()->Load();
                aries::ViewManager::GetInstance().Init();
                delete thd;

                //recover data
                auto recoveryer = std::make_shared<AriesXLogRecoveryer>();
                recoveryer->SetReader(AriesXLogManager::GetInstance().GetReader());
                result = recoveryer->Recovery();
                ARIES_ASSERT(result, "cannot recovery from xlog");

                //do other preparation stuff.
                AriesInitialTableManager::GetInstance().DoPreCache();
                AriesInitialTableManager::GetInstance().CreatePrimaryKeyIndex();

                bool expected = true;
                m_bInProgress.compare_exchange_strong(expected, false);
                m_statusCond.notify_all();
            }
            catch (...)
            {
                LOG(ERROR) << "Failed to vacuum! Need restart rateup server manually.";
                kill(getpid(), SIGQUIT);
            }
        });
        t.detach();
    }
}

int AriesVacuum::WaitVacuumDone()
{
    if (!m_bIsVacuumThread)
    {
        // if (m_bInProgress)
        //     ARIES_EXCEPTION(ER_XA_RBROLLBACK);
        unique_lock<mutex> lock(m_statusLock);
        m_statusCond.wait(lock,
                        [&] { return !m_bInProgress; });
    }
    return m_currentDbVersion;
}

bool AriesVacuum::IsMySelfVacuumThread()
{
    return m_bIsVacuumThread;
}

END_ARIES_ENGINE_NAMESPACE