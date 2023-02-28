#ifndef ARIESDEVICEPROPERTY_H_
#define ARIESDEVICEPROPERTY_H_

#include "AriesAssert.h"

#include <sys/sysinfo.h>

class AriesDeviceProperty
{
public:
    static AriesDeviceProperty GetInstance()
    {
        static AriesDeviceProperty instance;
        return instance;
    }
    bool IsHighMemoryDevice();

    size_t GetMemoryCapacity();
    void GetHostSysInfo(  struct sysinfo& hostSysInfo );
private:
    size_t GetTotalMem();
    AriesDeviceProperty(){}
    size_t m_totalMem = 0;
    const int HIGH_MEM_WARTER_MARK = 24000;// 24000MB

};

#endif