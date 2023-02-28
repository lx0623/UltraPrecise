#include <cuda.h>
#include "AriesDeviceManager.h"
#include "AriesAssert.h"

NAMESPACE_ARIES_START
    AriesDeviceManage::AriesDeviceManage()
    {
        cudaGetDeviceCount( &m_deviceCount );
        for( int deviceId = 0; deviceId < m_deviceCount; ++deviceId )
        {
            m_deviceUseCount[ deviceId ] = 0;
        }
    }

    int32_t AriesDeviceManage::GetDeviceUseCount( int32_t deviceId )
    {
        lock_guard< mutex > lock( m_mutex4DeviceMap );
        auto it = m_deviceUseCount.find( deviceId );

        ARIES_ASSERT( m_deviceUseCount.cend() != it, "Invalid device id: " + std::to_string( deviceId ) );
        return it->second;
    }

    int32_t AriesDeviceManage::UseDevice()
    {
        lock_guard< mutex > lock( m_mutex4DeviceMap );
        int32_t deviceId = GetFreeDevice();
        UseDeviceInternal( deviceId );

        return deviceId;
    }

    void AriesDeviceManage::UseDevice( int32_t deviceId )
    {
        lock_guard< mutex > lock( m_mutex4DeviceMap );
        UseDeviceInternal( deviceId );
    }

    void AriesDeviceManage::UnuseDevice()
    {
        int32_t deviceId;
        cudaGetDevice( &deviceId );
        UnuseDevice( deviceId );
    }
    void AriesDeviceManage::UnuseDevice( int32_t deviceId )
    {
        lock_guard< mutex > lock( m_mutex4DeviceMap );
        auto it = m_deviceUseCount.find( deviceId );
        if ( m_deviceUseCount.cend() != it && it->second > 0 )
            m_deviceUseCount[ deviceId ] -= 1;
    }

    void AriesDeviceManage::DumpDeviceUseCount()
    {
        lock_guard< mutex > lock( m_mutex4DeviceMap );
        string msg( "Device use count: " );
        for( int deviceId = 0; deviceId < m_deviceCount; ++deviceId )
        {
            msg.append( std::to_string( deviceId ) ).append( ":" )
               .append( std::to_string( m_deviceUseCount[ deviceId ] ) );
            msg.append( ", " );
        }
        LOG( INFO ) << msg;
    }

    int32_t AriesDeviceManage::GetFreeDevice()
    {
        int32_t minUseCount = INT32_MAX;
        int32_t foundDeviceId = 0;
        for( int deviceId = 0; deviceId < m_deviceCount; ++deviceId )
        {
            auto useCount = m_deviceUseCount[ deviceId ];
            if ( 0 == useCount )
            {
                return deviceId;
            }
            if ( useCount < minUseCount )
            {
                minUseCount = useCount;
                foundDeviceId = deviceId;
            }
        }
        return foundDeviceId;
    }

    void AriesDeviceManage::UseDeviceInternal( int32_t deviceId )
    {
        auto it = m_deviceUseCount.find( deviceId );
        if ( m_deviceUseCount.cend() != it  )
        {
            cudaSetDevice( deviceId );
            m_deviceUseCount[ deviceId ] += 1;
        }
    }

NAMESPACE_ARIES_END