#include "AriesDeviceProperty.h"

    bool AriesDeviceProperty::IsHighMemoryDevice()
    {
        return false;
        //return ( ( int )( GetTotalMem() / ( 1 << 20 ) ) ) > HIGH_MEM_WARTER_MARK;
    }

    size_t AriesDeviceProperty::GetMemoryCapacity()
    {
        return GetTotalMem() * 0.9;
    }

    void AriesDeviceProperty::GetHostSysInfo(  struct sysinfo& hostSysInfo )
    {
        if ( sysinfo( &hostSysInfo ) )
        {
            ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        }
    }

    size_t AriesDeviceProperty::GetTotalMem()
    {
        if( m_totalMem == 0 )
        {
            size_t freeMem;
            cudaError_t result = cudaMemGetInfo( &freeMem, &m_totalMem );
            if( cudaSuccess != result )
                throw aries::cuda_exception_t( result );
        }
        return m_totalMem;
    }