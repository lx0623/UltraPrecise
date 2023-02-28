#include <atomic>
#include <mutex>
#include <condition_variable>
#include "AriesDefinition.h"
using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesVacuum
{
public:
    static AriesVacuum &GetInstance()
    {
        static AriesVacuum instance;
        return instance;
    }

    int GetCurrentDbVersion()
    {
        return m_currentDbVersion;
    }

    AriesVacuum();
    ~AriesVacuum();

    void FlushAndReloadDb();
    int WaitVacuumDone();
    bool IsMySelfVacuumThread();

private:
    condition_variable m_statusCond;
    mutex m_statusLock;
    atomic<bool> m_bInProgress; // true : the vacuum is in progress. false: vacuum is done.
    atomic<int> m_currentDbVersion; // 0 based, +1 for each recovery. if server restarts, this value is set to 0 again.
    static thread_local bool m_bIsVacuumThread;// vacuum must run in a seperate thread. true: means the current thread is the vacuum thread.
};

END_ARIES_ENGINE_NAMESPACE