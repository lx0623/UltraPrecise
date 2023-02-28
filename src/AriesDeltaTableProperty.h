#ifndef ARIESDELTATABLEPROPERTY_H_
#define ARIESDELTATABLEPROPERTY_H_

#include "AriesAssert.h"
#include "AriesDeviceProperty.h"

#define MAX_PERCENT_OF_RAM_FOR_DELTA_TABLE (0.5)  //总内存中delta table可使用的内存的占比

class AriesDeltaTableProperty
{
public:
    AriesDeltaTableProperty() = default;

    static AriesDeltaTableProperty GetInstance()
    {
        static AriesDeltaTableProperty instance;
        return instance;
    }

    void TryToAddDeltaTableUsedMemory( const size_t mem_size )
    {
        unique_lock< mutex > lock( MutexForAddDeltaTableUsedMemory );
        if ( (m_deltaTableUsedMemory+mem_size) > GetMemoryCapacity()  )
        {
            ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        }
        m_deltaTableUsedMemory += mem_size;
    }

private:
    size_t GetMemoryCapacity()
    {
        if( m_memoryCapacity == 0 )
        {
            struct sysinfo hostSysInfo;
            AriesDeviceProperty::GetInstance().GetHostSysInfo( hostSysInfo );
            m_memoryCapacity = hostSysInfo.totalram * MAX_PERCENT_OF_RAM_FOR_DELTA_TABLE;
        }
        return m_memoryCapacity;
    }

    size_t m_memoryCapacity = 0;
    size_t m_deltaTableUsedMemory = 0;
    static mutex  MutexForAddDeltaTableUsedMemory;

};

mutex  AriesDeltaTableProperty::MutexForAddDeltaTableUsedMemory;

#endif