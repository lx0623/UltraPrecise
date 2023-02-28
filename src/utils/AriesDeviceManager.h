#pragma once
#include <unordered_map>
#include <mutex>
#include "AriesDefinition.h"

using namespace std;

NAMESPACE_ARIES_START
class AriesDeviceManage
{
public:
    static AriesDeviceManage& GetInstance()
    {
        static AriesDeviceManage instance;
        return instance;
    }

    int32_t GetDeviceUseCount( int32_t deviceId );

    // 为调用者选择并设置一个device
    int32_t UseDevice();

    // 使用制定的device
    void UseDevice( int32_t deviceId );

    void UnuseDevice();
    void UnuseDevice( int32_t deviceId );

    void DumpDeviceUseCount();
private:
    AriesDeviceManage();

    // 获取一个use count 最小的device
    int32_t GetFreeDevice();

    void UseDeviceInternal( int32_t deviceId );

    int32_t m_deviceCount;
    unordered_map< int32_t, int32_t > m_deviceUseCount;
    mutex m_mutex4DeviceMap;
};
NAMESPACE_ARIES_END