/*
 * AriesEngineDef.cu
 *
 *  Created on: Jul 22, 2019
 *      Author: lichi
 */

#include "AriesEngineDef.h"
// #ifdef ARIES_PROFILE
// #include "CpuTimer.h"
// #endif
BEGIN_ARIES_ACC_NAMESPACE

    void* AriesMemAllocator::Alloc( size_t size, unsigned int flags )
    {
        void* p = nullptr;
        if( size )
            ARIES_CALL_CUDA_API( cudaMallocManaged( &p, size, flags ) );
        return p;
    }

    void* AriesMemAllocator::CloneFromHostMem( void* p, size_t size )
    {
        void* d = nullptr;
        if( size > 0 && p )
        {
            ARIES_CALL_CUDA_API( cudaMallocManaged( &d, size ) );
            ARIES_CALL_CUDA_API( cudaMemcpy( d, p, size, cudaMemcpyKind::cudaMemcpyHostToDevice ) );
        }
        return d;
    }

    void AriesMemAllocator::Free( void* p )
    {
        if( p )
            ARIES_CALL_CUDA_API( cudaFree( p ) );
    }

    void AriesMemAllocator::FillZero( void* p, size_t size )
    {
        if( p )
            ARIES_CALL_CUDA_API( cudaMemset( p, 0, size ) );
    }

    void AriesMemAllocator::MemCopy( void *dst, const void *src, size_t count, cudaMemcpyKind kind )
    {
        if( dst && src )
            ARIES_CALL_CUDA_API( cudaMemcpy( dst, src, count, kind ) );
    }

    void AriesMemAllocator::PrefetchToDeviceAsync( const void *devPtr, size_t count, int dstDevice, cudaStream_t stream )
    {
        if ( !devPtr )
            return;
// #ifdef ARIES_PROFILE
//         aries::CPU_Timer timer;
//         timer.begin();
// #endif
        if( count > 0 )
            ARIES_CALL_CUDA_API( cudaMemPrefetchAsync( devPtr, count, dstDevice, stream ) );
// #ifdef ARIES_PROFILE
//         cudaDeviceSynchronize();
//         std::string log = std::string( "prefetch: " ) + std::to_string( count ) + " used: " + std::to_string( timer.end() );
//         std::cout << log << std::endl;
// #endif
    }

    void AriesMemAllocator::MemAdvise( const void* devPtr, size_t count, cudaMemoryAdvise advice, int device )
    {
        ARIES_CALL_CUDA_API( cudaMemAdvise( devPtr, count, advice, device ) );
    }

    AriesDataBuffer::AriesDataBuffer( AriesColumnType columnType, size_t capacity, bool bInitZero, unsigned int flags )
            : m_columnType( columnType ), m_capacity( capacity ), m_sortType( AriesOrderByType::NONE ), m_bFromCache( false ), m_attachedBuff( false )
    {
        std::random_device rd;
        m_id = rd();
        m_itemCount = m_capacity;
        if( capacity > 0 )
            AllocArray( capacity, bInitZero, flags );
    }

    AriesDataBuffer::~AriesDataBuffer()
    {
        if ( m_attachedBuff )
        {
            int8_t* p = ReleaseData();
            free( p );
        }
    }

    int8_t* AriesDataBuffer::AllocArray( size_t itemCount, bool bInitZero, unsigned int flags )
    {
        ARIES_ASSERT( itemCount > 0, "itemCount: " + to_string( itemCount ) );
        size_t totalBytes = itemCount * GetItemSizeInBytes();
        int8_t* data = ( int8_t* )AriesMemAllocator::Alloc( totalBytes, flags );
        m_array.reset( data );
        m_capacity = m_itemCount = itemCount;
        if( bInitZero )
        {
            PrefetchToGpu();
            AriesMemAllocator::FillZero( data, totalBytes );
        }

        return data;
    }

    void AriesDataBuffer::MemAdvise( cudaMemoryAdvise advice, int device )
    {
        AriesMemAllocator::MemAdvise( GetData(), GetTotalBytes(), advice, device );
    }

    void AriesDataBuffer::SetFromCache( bool bFromCache )
    {
        m_bFromCache = bFromCache;
    }

    bool AriesDataBuffer::isFromCache()
    {
        return m_bFromCache;
    }

    void AriesDataBuffer::PrefetchToCpu() const
    {
        if ( m_attachedBuff )
            return;
        if( m_itemCount > 0 )
            AriesMemAllocator::PrefetchToDeviceAsync( m_array.get(), GetTotalBytes(), cudaCpuDeviceId );
    }

    void AriesDataBuffer::PrefetchToCpu( size_t offset, size_t itemCount ) const
    {
        if ( m_attachedBuff )
            return;

        ARIES_ASSERT( offset + itemCount < m_itemCount, "offset + itemCount must < m_itemCount" );
        size_t perItemSize = GetItemSizeInBytes();
        if( itemCount > 0 )
            AriesMemAllocator::PrefetchToDeviceAsync( m_array.get() + offset * perItemSize, itemCount * perItemSize, cudaCpuDeviceId );
    }

    void AriesDataBuffer::PrefetchToGpu( int deviceId ) const
    {
        assert( !m_attachedBuff );

        if( deviceId == ACTIVE_DEVICE_ID )
        {
            //prefetch to active device.
            cudaError_t result = cudaGetDevice( &deviceId );
            if( cudaSuccess != result )
            {
                cudaGetLastError();
                throw cuda_exception_t( result );
            }
        }
        AriesMemAllocator::PrefetchToDeviceAsync( m_array.get(), GetTotalBytes(), deviceId );
    }

    void AriesDataBuffer::PrefetchToGpu( size_t offset, size_t itemCount, int deviceId ) const
    {
        assert( !m_attachedBuff );

        ARIES_ASSERT( offset + itemCount <= m_itemCount, "offset + itemCount must < m_itemCount" );
        size_t perItemSize = GetItemSizeInBytes();
        if( deviceId == ACTIVE_DEVICE_ID )
        {
            //prefetch to active device.
            cudaError_t result = cudaGetDevice( &deviceId );
            if( cudaSuccess != result )
            {
                cudaGetLastError();
                throw cuda_exception_t( result );
            }
        }
        AriesMemAllocator::PrefetchToDeviceAsync( m_array.get() + offset * perItemSize, itemCount * perItemSize, deviceId );
    }

    int8_t* AriesDataBuffer::CopyFromHostMem( void* p, size_t itemCount )
    {
        ARIES_ASSERT( itemCount > 0 && p, "itemCount: " + to_string( itemCount ) + "p is nullptr: " + to_string( !!p ) );
        int8_t* data = ( int8_t* )AriesMemAllocator::CloneFromHostMem( p, itemCount * GetItemSizeInBytes() );
        m_array.reset( data );
        m_itemCount = itemCount;
        return data;
    }

    int8_t* AriesDataBuffer::GetData() const
    {
        return m_array.get();
    }

    int8_t* AriesDataBuffer::GetData( size_t offset ) const
    {
        ARIES_ASSERT( offset < m_itemCount, "offset must < m_itemCount" );
        return m_array.get() + offset * GetItemSizeInBytes();
    }

    size_t AriesDataBuffer::GetItemCount() const
    {
        return m_itemCount;
    }

    void AriesDataBuffer::SetItemCount( size_t itemCount )
    {
        ARIES_ASSERT( itemCount <= m_capacity, "itemCount: " + to_string( itemCount ) + ", m_itemCount: " + to_string( m_itemCount ) );
        // can only reduce item count, we don't actually release memory
        m_itemCount = itemCount;
    }

    size_t AriesDataBuffer::GetItemSizeInBytes() const
    {
        return m_columnType.GetDataTypeSize();
    }

    size_t AriesDataBuffer::GetTotalBytes() const
    {
        return m_itemCount * GetItemSizeInBytes();
    }

    // buf's ownership is transferred to the callee
    void AriesDataBuffer::AttachBuffer( int8_t* buf, size_t elementCount )
    {
        m_array.reset( buf );
        m_attachedBuff = true;
        m_itemCount = elementCount;
    }

    void AriesDataBuffer::Reset()
    {
        if ( !m_attachedBuff )
            m_array.reset( nullptr );
        else
        {
            int8_t* p = ReleaseData();
            free( p );
        }
        m_itemCount = 0;
    }

    // data's ownership is transferred to the caller
    int8_t* AriesDataBuffer::ReleaseData()
    {
        ARIES_ASSERT( m_array, "m_array is nullptr" );
        m_itemCount = 0;
        m_attachedBuff = false;
        return m_array.release();
    }

    AriesColumnType AriesDataBuffer::GetDataType() const
    {
        return m_columnType;
    }

    std::shared_ptr< AriesDataBuffer > AriesDataBuffer::Clone( int dstDevice ) const
    {
        AriesDataBufferSPtr tmp;
        if ( m_attachedBuff )
        {
            tmp = std::make_shared< AriesDataBuffer >( m_columnType );
            int8_t* buff = ( int8_t* )malloc( m_itemCount * m_columnType.GetDataTypeSize() );
            tmp->AttachBuffer( buff, m_itemCount );

            if( m_itemCount > 0 )
                memcpy( tmp->GetData(), m_array.get(), tmp->GetTotalBytes() );
        }
        else
        {
            tmp = std::make_shared< AriesDataBuffer >( m_columnType, m_itemCount );
            if( m_itemCount > 0 )
            {
                if( dstDevice == cudaCpuDeviceId )
                    tmp->PrefetchToCpu();
                else
                    tmp->PrefetchToGpu( dstDevice );

                AriesMemAllocator::MemCopy( tmp->GetData(), m_array.get(), tmp->GetTotalBytes() );
            }
        }
        return tmp;
    }

    std::shared_ptr< AriesDataBuffer > AriesDataBuffer::CloneWithNoContent( int dstDevice ) const
    {
        return CloneWithNoContent( m_itemCount, dstDevice );
    }

    std::shared_ptr< AriesDataBuffer > AriesDataBuffer::CloneWithNoContent( size_t elementCount, int dstDevice ) const
    {
        AriesDataBufferSPtr tmp;
        if ( m_attachedBuff )
        {
            tmp = std::make_shared< AriesDataBuffer >( m_columnType );
            int8_t* buff = ( int8_t* )malloc( elementCount * m_columnType.GetDataTypeSize() );
            tmp->AttachBuffer( buff, elementCount );
        }
        else
        {
            tmp = std::make_shared< AriesDataBuffer >( m_columnType, elementCount );
            if( elementCount > 0 )
            {
                if( dstDevice == cudaCpuDeviceId )
                    tmp->PrefetchToCpu();
                else
                    tmp->PrefetchToGpu( dstDevice );
            }
        }
        return tmp;
    }

    AriesOrderByType AriesDataBuffer::GetSortType() const
    {
        return m_sortType;
    }

    void AriesDataBuffer::SetSortType( AriesOrderByType sortType )
    {
        m_sortType = sortType;
    }

    int8_t* AriesDataBuffer::GetItemDataAt( int index ) const
    {
        ARIES_ASSERT( m_array, "m_array is nullptr" );
        return m_array.get() + index * GetItemSizeInBytes();
    }

    std::string AriesDataBuffer::ItemToString( int index ) const
    {
        assert( m_array && index >= 0 && index < m_itemCount );
        std::string result;
        switch( m_columnType.DataType.ValueType )
        {
            case AriesValueType::INT32:
                result = Int32ToString( index );
                break;
            default:
                break;
        }
        return result;
    }
    
    std::string AriesDataBuffer::Int32ToString( int index ) const
    {
        std::string result;
        if( m_columnType.HasNull )
        {
            auto data = GetNullableInt32( index );
            if( data.flag )
                return std::to_string( data.value );
            else
                return "NULL";
        }
        else
            return std::to_string( GetInt32( index ) );
    }

    void AriesDataBuffer::Dump( size_t dumpCount ) const
    {
        switch( m_columnType.DataType.ValueType )
        {
            case AriesValueType::INT8:
                DumpInt8( dumpCount );
                break;
            case AriesValueType::INT16:
                DumpInt16( dumpCount );
                break;
            case AriesValueType::INT32:
                DumpInt32( dumpCount );
                break;
            case AriesValueType::INT64:
                DumpInt64( dumpCount );
                break;
            case AriesValueType::CHAR:
                DumpString( dumpCount );
                break;
            case AriesValueType::DECIMAL:
                DumpDecimal( dumpCount );
                break;
            case AriesValueType::COMPACT_DECIMAL:
                DumpCompactDecimal( dumpCount );
                break;
            case AriesValueType::DATE:
                DumpDate( dumpCount );
                break;
            case AriesValueType::DATETIME:
                DumpDatetime( dumpCount );
                break;
            default:
                break;
        }
    }

    bool AriesDataBuffer::isNullableColumn() const
    {
        return m_columnType.isNullable();
    }

    bool AriesDataBuffer::isUniqueColumn() const
    {
        return m_columnType.isUnique();
    }

    char AriesDataBuffer::GetInt8( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const char* >( m_array.get() ) + index );
    }

    nullable_type< int8_t > AriesDataBuffer::GetNullableInt8( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< int8_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isInt8DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableInt8( index ).flag == 0;
    }

    unsigned char AriesDataBuffer::GetUint8( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const unsigned char* >( m_array.get() ) + index );
    }

    nullable_type< uint8_t > AriesDataBuffer::GetNullableUint8( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< uint8_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isUint8DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableUint8( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetInt16AsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< int16_t > nr = *( reinterpret_cast< const nullable_type< int16_t >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        int16_t r = *( reinterpret_cast< const int16_t * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    int16_t AriesDataBuffer::GetInt16( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const int16_t * >( m_array.get() ) + index );
    }

    nullable_type< int16_t > AriesDataBuffer::GetNullableInt16( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< int16_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isInt16DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableInt16( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetUint16AsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< uint16_t > nr = *( reinterpret_cast< const nullable_type< uint16_t >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        uint16_t r = *( reinterpret_cast< const uint16_t * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    uint16_t AriesDataBuffer::GetUint16( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const uint16_t * >( m_array.get() ) + index );
    }

    nullable_type< uint16_t > AriesDataBuffer::GetNullableUint16( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< uint16_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isUint16DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableUint16( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetInt32AsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< int32_t > nr = *( reinterpret_cast< const nullable_type< int32_t >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        int32_t r = *( reinterpret_cast< const int32_t * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    int32_t AriesDataBuffer::GetInt32( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const int* >( m_array.get() ) + index );
    }

    nullable_type< int32_t > AriesDataBuffer::GetNullableInt32( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< int32_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isInt32DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableInt32( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetUint32AsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< uint32_t > nr = *( reinterpret_cast< const nullable_type< uint32_t >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        uint32_t r = *( reinterpret_cast< const uint32_t * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    uint32_t AriesDataBuffer::GetUint32( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const uint32_t* >( m_array.get() ) + index );
    }

    nullable_type< uint32_t > AriesDataBuffer::GetNullableUint32( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< uint32_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isUint32DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableUint32( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetInt64AsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< int64_t > nr = *( reinterpret_cast< const nullable_type< int64_t >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        int64_t r = *( reinterpret_cast< const int64_t * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    int64_t AriesDataBuffer::GetInt64( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const int64_t* >( m_array.get() ) + index );
    }

    nullable_type< int64_t > AriesDataBuffer::GetNullableInt64( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< int64_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isInt64DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableInt64( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetUint64AsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< uint64_t > nr = *( reinterpret_cast< const nullable_type< uint64_t >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        uint64_t r = *( reinterpret_cast< const uint64_t * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    uint64_t AriesDataBuffer::GetUint64( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const int64_t* >( m_array.get() ) + index );
    }

    nullable_type< uint64_t > AriesDataBuffer::GetNullableUint64( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< uint64_t >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isUint64DataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableUint64( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetFloatAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< float > nr = *( reinterpret_cast< const nullable_type< float >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        float r = *( reinterpret_cast< const float * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    float AriesDataBuffer::GetFloat( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const float* >( m_array.get() ) + index );
    }

    nullable_type< float > AriesDataBuffer::GetNullableFloat( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< float >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isFloatDataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableFloat( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetDoubleAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( isNullableColumn() )
        {
            nullable_type< double > nr = *( reinterpret_cast< const nullable_type< double >* >( m_array.get() ) + index );
            if( nr.flag == 0 )
            {
                return "NULL";
            }
            return std::to_string( nr.value );
        }
        double r = *( reinterpret_cast< const double * >( m_array.get() ) + index );
        return std::to_string( r );
    }

    double AriesDataBuffer::GetDouble( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const double* >( m_array.get() ) + index );
    }

    nullable_type< double > AriesDataBuffer::GetNullableDouble( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const nullable_type< double >* >( m_array.get() ) + index );
    }

    bool AriesDataBuffer::isDoubleDataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableDouble( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetDecimalAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( m_columnType.isNullable() )
        {
            if( m_columnType.DataType.ValueType == AriesValueType::DECIMAL )
                return GetNullableDecimalAsString( index );
            else
            {
                //compact decimal
                return GetNullableCompactDecimalAsString( index, m_columnType.DataType.Precision, m_columnType.DataType.Scale );
            }
        }
        else
        {
            if( m_columnType.DataType.ValueType == AriesValueType::DECIMAL )
            {
                char result[64];
                auto dec = reinterpret_cast< Decimal* >( m_array.get() ) + index;
                return dec->GetDecimal( result );
            }
            else
            {
                //compact decimal
                return GetCompactDecimalAsString( index, m_columnType.DataType.Precision, m_columnType.DataType.Scale );
            }
        }
    }

    Decimal AriesDataBuffer::GetDecimal( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( m_columnType.DataType.ValueType == AriesValueType::DECIMAL )
            return *( reinterpret_cast< const Decimal* >( m_array.get() ) + index );
        else
        {
            const CompactDecimal* compDec = ( reinterpret_cast< const CompactDecimal* >( m_array.get() ) + index * m_columnType.GetDataTypeSize() );
            return Decimal( compDec, m_columnType.DataType.Precision, m_columnType.DataType.Scale );
        }
    }

    nullable_type< Decimal > AriesDataBuffer::GetNullableDecimal( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( m_columnType.DataType.ValueType == AriesValueType::DECIMAL )
            return *( reinterpret_cast< const nullable_type< Decimal >* >( m_array.get() ) + index );
        else
            return GetNullableCompactDecimal( index );
    }

    std::string AriesDataBuffer::GetNullableDecimalAsString( size_t index ) const
    {
        nullable_type< Decimal > dec = GetNullableDecimal( index );
        if( dec.flag )
        {
            char result[NUM_TOTAL_DIG*DIG_PER_INT32];
            return dec.value.GetDecimal( result );
        }
        else
            return "NULL";
    }

    bool AriesDataBuffer::isDecimalDataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableDecimal( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetCompactDecimalAsString( size_t index, uint16_t precision, uint16_t scale ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( m_columnType.isNullable() )
        {
            return GetNullableCompactDecimalAsString( index, precision, scale );
        }
        else
        {
            char result[NUM_TOTAL_DIG*DIG_PER_INT32];
            auto compactDec = reinterpret_cast< CompactDecimal* >( m_array.get() ) + index * m_columnType.GetDataTypeSize();
            Decimal dec( compactDec, precision, scale );
            return dec.GetDecimal( result );
        }
    }

    CompactDecimal AriesDataBuffer::GetCompactDecimal( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        return *( reinterpret_cast< const CompactDecimal* >( m_array.get() ) + index * m_columnType.GetDataTypeSize() );
    }

    nullable_type< Decimal > AriesDataBuffer::GetNullableCompactDecimal( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        int8_t* pData = m_array.get() + index * m_columnType.GetDataTypeSize();
        return nullable_type< Decimal >( *pData,
                Decimal( ( CompactDecimal* )pData + 1, m_columnType.DataType.Precision, m_columnType.DataType.Scale ) );
    }

    std::string AriesDataBuffer::GetNullableCompactDecimalAsString( size_t index, uint16_t precision, uint16_t scale ) const
    {
        nullable_type< Decimal > decimal = GetNullableCompactDecimal( index );
        if( decimal.flag )
        {
            char result[64];
            return decimal.value.GetDecimal( result );
        }
        else
            return "NULL";
    }

    bool AriesDataBuffer::isCompactDecimalDataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return GetNullableCompactDecimal( index ).flag == 0;
    }

    std::string AriesDataBuffer::GetString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( m_columnType.isNullable() )
        {
            return GetNullableString( index );
        }
        else
        {
            std::string value;
            value.assign( ( char* )m_array.get() + index * GetItemSizeInBytes(), GetItemSizeInBytes() );
            return value.c_str();
        }
    }

    std::string AriesDataBuffer::GetNullableString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        std::string value;
        if( *( ( char* )m_array.get() + index * GetItemSizeInBytes() ) )
        {
            std::string tmp;
            tmp.assign( ( char* )m_array.get() + index * GetItemSizeInBytes() + 1, GetItemSizeInBytes() - 1 );
            value = tmp.c_str();
        }
        else
            value = "NULL";
        return value;
    }

    bool AriesDataBuffer::isStringDataNull( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        return *( ( char* )m_array.get() + index * GetItemSizeInBytes() ) == 0;
    }

    AriesTime* AriesDataBuffer::GetTime( size_t index ) const
    {
        ARIES_ASSERT( !m_columnType.isNullable(), "time is nullable" );
        return reinterpret_cast< AriesTime* >( m_array.get() ) + index;
    }
    nullable_type< AriesTime >* AriesDataBuffer::GetNullableTime( size_t index ) const
    {
        ARIES_ASSERT( m_columnType.isNullable(), "time is not nullable" );
        return reinterpret_cast< nullable_type< AriesTime >* >( m_array.get() ) + index;
    }
    std::string AriesDataBuffer::GetTimeAsString( size_t index ) const
    {
        if( m_columnType.isNullable() )
        {
            return GetNullableTimeAsString( index );
        }
        else
        {
            auto time = *( reinterpret_cast< AriesTime* >( m_array.get() ) + index );
            char tmp[16] =
            { 0 };
            sprintf( tmp, "%s%hu:%02hhu:%02hhu", time.sign > 0 ? "" : "-", time.hour, time.minute, time.second );
            return std::string( tmp );
        }
    }
    std::string AriesDataBuffer::GetNullableTimeAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        auto time = *( reinterpret_cast< nullable_type< AriesTime >* >( m_array.get() ) + index );
        if( time.flag )
        {
            char tmp[16] =
            { 0 };
            sprintf( tmp, "%s%hu:%02hhu:%02hhu", time.value.sign > 0 ? "" : "-", time.value.hour, time.value.minute, time.value.second );
            return std::string( tmp );
        }
        else
            return "NULL";
    }
    bool AriesDataBuffer::isTimeDataNull( size_t index )
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        auto time = *( reinterpret_cast< nullable_type< AriesTime >* >( m_array.get() ) + index );
        return time.flag == 0;
    }
    AriesYear* AriesDataBuffer::GetYear( size_t index ) const
    {
        ARIES_ASSERT( !m_columnType.isNullable(), "year is nullable" );
        return reinterpret_cast< AriesYear* >( m_array.get() ) + index;
    }
    nullable_type< AriesYear >* AriesDataBuffer::GetNullableYear( size_t index ) const
    {
        ARIES_ASSERT( m_columnType.isNullable(), "year is not nullable" );
        return reinterpret_cast< nullable_type< AriesYear >* >( m_array.get() ) + index;
    }
    std::string AriesDataBuffer::GetYearAsString( size_t index ) const
    {
        if( m_columnType.isNullable() )
        {
            return GetNullableYearAsString( index );
        }
        else
        {
            auto year = *( reinterpret_cast< AriesYear* >( m_array.get() ) + index );
            char tmp[8] =
            { 0 };
            sprintf( tmp, "%04hu", year.year );
            return std::string( tmp );
        }
    }
    std::string AriesDataBuffer::GetNullableYearAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        auto year = *( reinterpret_cast< nullable_type< AriesYear >* >( m_array.get() ) + index );
        if( year.flag )
        {
            char tmp[8] =
            { 0 };
            sprintf( tmp, "%04hu", year.value.year );
            return std::string( tmp );
        }
        else
            return "NULL";
    }
    bool AriesDataBuffer::isYearDataNull( size_t index )
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        auto year = *( reinterpret_cast< nullable_type< AriesYear >* >( m_array.get() ) + index );
        return year.flag == 0;
    }
    AriesTimestamp* AriesDataBuffer::GetTimestamp( size_t index ) const
    {
        ARIES_ASSERT( !m_columnType.isNullable(), "timestamp is nullable" );
        return reinterpret_cast< AriesTimestamp* >( m_array.get() ) + index;
    }
    nullable_type< AriesTimestamp >* AriesDataBuffer::GetNullableTimestamp( size_t index ) const
    {
        ARIES_ASSERT( m_columnType.isNullable(), "timestamp is not nullable" );
        return reinterpret_cast< nullable_type< AriesTimestamp >* >( m_array.get() ) + index;
    }
    std::string AriesDataBuffer::GetTimestampAsString( size_t index ) const
    {
        if( m_columnType.isNullable() )
        {
            return GetNullableTimestampAsString( index );
        }
        else
        {
            auto ts = *( reinterpret_cast< AriesTimestamp* >( m_array.get() ) + index );
            AriesDatetime datetime( ts.getTimeStamp(), 0 );
            char tmp[32] =
            { 0 };
            if ( datetime.getMicroSec() )
            {
                sprintf( tmp, "%04hu-%02hhu-%02hhu %02hhu:%02hhu:%02hhu.%06u", datetime.getYear(), datetime.getMonth(), datetime.getDay(), datetime.getHour(),
                        datetime.getMinute(), datetime.getSecond(), datetime.getMicroSec() );
            }
            else
            {
                sprintf( tmp, "%04hu-%02hhu-%02hhu %02hhu:%02hhu:%02hhu", datetime.getYear(), datetime.getMonth(), datetime.getDay(), datetime.getHour(),
                        datetime.getMinute(), datetime.getSecond() );
            }
            return std::string( tmp );
        }
    }
    std::string AriesDataBuffer::GetNullableTimestampAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        auto ts = *( reinterpret_cast< nullable_type< AriesTimestamp >* >( m_array.get() ) + index );
        if( ts.flag )
        {
            char tmp[32] =
            { 0 };
            AriesDatetime datetime( ts.value.getTimeStamp(), 0 );
            if ( datetime.getMicroSec() )
            {
                sprintf( tmp, "%04hu-%02hhu-%02hhu %02hhu:%02hhu:%02hhu.%06u", datetime.getYear(), datetime.getMonth(), datetime.getDay(), datetime.getHour(),
                        datetime.getMinute(), datetime.getSecond(), datetime.getMicroSec() );
            }
            else
            {
                sprintf( tmp, "%04hu-%02hhu-%02hhu %02hhu:%02hhu:%02hhu", datetime.getYear(), datetime.getMonth(), datetime.getDay(), datetime.getHour(),
                        datetime.getMinute(), datetime.getSecond() );
            }
            return std::string( tmp );
        }
        else
            return "NULL";
    }
    bool AriesDataBuffer::isTimestampDataNull( size_t index )
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        auto ts = *( reinterpret_cast< nullable_type< AriesTimestamp >* >( m_array.get() ) + index );
        return ts.flag == 0;
    }
    AriesDate* AriesDataBuffer::GetDate( size_t index ) const
    {
        ARIES_ASSERT( !m_columnType.isNullable(), "date is nullable" );
        return reinterpret_cast< AriesDate* >( m_array.get() ) + index;
    }
    nullable_type< AriesDate >* AriesDataBuffer::GetNullableDate( size_t index ) const
    {
        ARIES_ASSERT( m_columnType.isNullable(), "date is not nullable" );
        return reinterpret_cast< nullable_type< AriesDate >* >( m_array.get() ) + index;
    }
    std::string AriesDataBuffer::GetDateAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( m_columnType.isNullable() )
        {
            return GetNullableDateAsString( index );
        }
        else
        {
            auto date = *( reinterpret_cast< AriesDate* >( m_array.get() ) + index );
            char tmp[32] =
            { 0 };
            sprintf( tmp, "%04u-%02u-%02u", date.getYear(), date.getMonth(), date.getDay() );
            return std::string( tmp );
        }
    }

    std::string AriesDataBuffer::GetNullableDateAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        nullable_type< AriesDate > date = *( reinterpret_cast< nullable_type< AriesDate >* >( m_array.get() ) + index );
        if( date.flag )
        {
            char tmp[32] =
            { 0 };
            sprintf( tmp, "%04u-%02u-%02u", date.value.getYear(), date.value.getMonth(), date.value.getDay() );
            return std::string( tmp );
        }
        else
            return "NULL";
    }

    bool AriesDataBuffer::isDateDataNull( size_t index )
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        auto date = *( reinterpret_cast< nullable_type< AriesDate >* >( m_array.get() ) + index );
        return date.flag == 0;
    }

    AriesDatetime* AriesDataBuffer::GetDatetime( size_t index ) const
    {
        ARIES_ASSERT( !m_columnType.isNullable(), "datetime is nullable" );
        return reinterpret_cast< AriesDatetime* >( m_array.get() ) + index;
    }
    nullable_type< AriesDatetime >* AriesDataBuffer::GetNullableDatetime( size_t index ) const
    {
        ARIES_ASSERT( m_columnType.isNullable(), "datetime is not nullable" );
        return reinterpret_cast< nullable_type< AriesDatetime >* >( m_array.get() ) + index;
    }
    std::string AriesDataBuffer::GetDatetimeAsString( size_t index ) const
    {
        if( m_columnType.isNullable() )
        {
            return GetNullableDatetimeAsString( index );
        }
        else
        {
            auto date = *( reinterpret_cast< AriesDatetime* >( m_array.get() ) + index );
            char tmp[32] =
            { 0 };
            sprintf( tmp, "%04u-%02u-%02u %02u:%02u:%02u", date.getYear(), date.getMonth(), date.getDay(), date.getHour(), date.getMinute(),
                    date.getSecond() );
            return std::string( tmp );
        }
    }

    std::string AriesDataBuffer::GetNullableDatetimeAsString( size_t index ) const
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        auto date = *( reinterpret_cast< nullable_type< AriesDatetime >* >( m_array.get() ) + index );
        if( date.flag )
        {
            char tmp[32] =
            { 0 };
            sprintf( tmp, "%04u-%02u-%02u %02u:%02u:%02u", date.value.getYear(), date.value.getMonth(), date.value.getDay(), date.value.getHour(),
                    date.value.getMinute(), date.value.getSecond() );
            return std::string( tmp );
        }
        else
            return "NULL";
    }

    bool AriesDataBuffer::isDatetimeDataNull( size_t index )
    {
        ARIES_ASSERT( index < m_itemCount, "index: " + to_string( index ) + ", m_itemCount: " + to_string( m_itemCount ) );
        if( !isNullableColumn() )
        {
            return false;
        }
        auto date = *( reinterpret_cast< nullable_type< AriesDatetime >* >( m_array.get() ) + index );
        return date.flag == 0;
    }

    uint32_t AriesDataBuffer::GetId() const
    {
        return m_id;
    }

    void AriesDataBuffer::DumpDate( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetNullableDateAsString( i ) << std::endl;
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetDateAsString( i ) << std::endl;
        }
    }

    void AriesDataBuffer::DumpDatetime( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetNullableDatetimeAsString( i ) << std::endl;
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetDatetimeAsString( i ) << std::endl;
        }
    }

    void AriesDataBuffer::DumpDecimal( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetNullableDecimalAsString( i ) << std::endl;
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetDecimalAsString( i ) << std::endl;
        }
    }

    void AriesDataBuffer::DumpCompactDecimal( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        size_t itemSizeInBytes = GetItemSizeInBytes();
        int8_t* pData;
        uint16_t prec = m_columnType.DataType.Precision;
        uint16_t sca = m_columnType.DataType.Scale;
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
            {
                pData = m_array.get() + i * itemSizeInBytes;
                nullable_type< Decimal > dec( *pData, Decimal( ( CompactDecimal* )pData + 1, prec, sca ) );
                if( dec.flag )
                {
                    char result[64];
                    std::cout << dec.value.GetDecimal( result ) << std::endl;
                }
                else
                    std::cout << "NULL" << std::endl;
            }
        }
        else
        {
            for( int i = 0; i < count; ++i )
            {
                pData = m_array.get() + i * itemSizeInBytes;
                Decimal dec( ( CompactDecimal* )pData, prec, sca );
                char result[64];
                std::cout << dec.GetDecimal( result ) << std::endl;
            }
        }
    }

    void AriesDataBuffer::DumpInt8( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
            {
                auto data = GetNullableInt8( i );
                if( data.flag )
                    printf( "%d\n", data.value );
                else
                    std::cout << "NULL" << std::endl;
            }
        }
        else
        {
            for( int i = 0; i < count; ++i )
                printf( "%d\n", GetInt8( i ) );
        }
    }

    void AriesDataBuffer::DumpInt16( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
            {
                auto data = GetNullableInt16( i );
                if( data.flag )
                    std::cout << data.value << std::endl;
                else
                    std::cout << "NULL" << std::endl;
            }
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetInt16( i ) << std::endl;
        }
    }

    void AriesDataBuffer::DumpInt32( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
            {
                auto data = GetNullableInt32( i );
                if( data.flag )
                    std::cout << data.value << std::endl;
                else
                    std::cout << "NULL" << std::endl;
            }
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetInt32( i ) << std::endl;
        }
    }

    void AriesDataBuffer::DumpInt64( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
            {
                auto data = GetNullableInt64( i );
                if( data.flag )
                    std::cout << data.value << std::endl;
                else
                    std::cout << "NULL" << std::endl;
            }
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << GetInt64( i ) << "\n";
        }
    }

    void AriesDataBuffer::DumpString( size_t dumpCount ) const
    {
        size_t count = std::min( dumpCount, ( size_t )m_itemCount );
        if( m_columnType.HasNull )
        {
            for( int i = 0; i < count; ++i )
                std::cout << "[" << i << "]" << GetNullableString( i ) << std::endl;
        }
        else
        {
            for( int i = 0; i < count; ++i )
                std::cout << "[" << i << "]" << GetString( i ) << std::endl;
        }
    }

    AriesDynamicCodeInfo& AriesDynamicCodeInfo::operator +=( const AriesDynamicCodeInfo& code )
    {
        size_t total = FunctionKeyNameMapping.size() + code.FunctionKeyNameMapping.size();
        FunctionKeyNameMapping.insert( code.FunctionKeyNameMapping.begin(), code.FunctionKeyNameMapping.end() );
        assert( total == FunctionKeyNameMapping.size() );
        KernelCode += code.KernelCode;
        return *this;
    }

    template< typename type_t >
    AriesRangeBuffer< type_t >::AriesRangeBuffer( AriesColumnType columnType, type_t start, size_t count ): AriesDataBuffer( columnType, 1 )
    {
        m_start = start;
        m_count = count;
        SetItemCount( 1 );
        const auto ptr = ( type_t* ) GetData();
        ptr[ 0 ] = start;
    }

    template< typename type_t >
    size_t AriesRangeBuffer< type_t >::GetItemCount() const
    {
        return m_count;
    }

    template< typename type_t >
    int8_t* AriesRangeBuffer< type_t >::GetItemDataAt( int index ) const
    {
        type_t value;
        if ( m_start > 0 )
        {
            value = m_start + index;
        }
        else
        {
            value = m_start - index;
        }

        const auto ptr = ( type_t* ) GetData();
        ptr[ 0 ] = value;
        return GetData();
    }

    void AriesDataBuffer::SetPartitionInfo( const AriesDataBuffer::PartitionInfo& info )
    {
        m_partitionInfo = info;
    }

    AriesDataBuffer::PartitionInfo AriesDataBuffer::GetPartitionInfo() const
    {
        return m_partitionInfo;
    }

    template AriesRangeBuffer< int >::AriesRangeBuffer( AriesColumnType columnType, int start, size_t count );
    template size_t AriesRangeBuffer< int >::GetItemCount() const;
    template int8_t* AriesRangeBuffer< int >::GetItemDataAt( int index ) const;

END_ARIES_ACC_NAMESPACE

