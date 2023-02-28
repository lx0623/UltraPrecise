/*
 * AriesEngineDef.h
 *
 *  Created on: Jun 15, 2019
 *      Author: lichi
 */

#ifndef ARIESENGINEDEF_H_
#define ARIESENGINEDEF_H_

#include <boost/variant.hpp>
#include <exception>
#include <memory>
#include <atomic>
#include <random>
#include <iostream>
#include <cuda.h>
#include <driver_types.h>
#include <map>
#include "cstring"
#include "AriesColumnType.h"
#include "datatypes/aries_types.hxx"
#include "AriesEngineException.h"
#include "AriesDeviceProperty.h"
using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

#define ACTIVE_DEVICE_ID INT_MAX

#define ARIES_CALL_CUDA_API( call )                                               \
    do {\
        const cudaError_t error = call;\
        if( error != cudaSuccess )\
        {\
            cudaGetLastError();\
            printf( "Error: %s:%d, ", __FILE__, __LINE__ );\
            printf( "code:%d, reason: %s\n", error, cudaGetErrorString(error));\
            throw cuda_exception_t( error );\
        }\
    } while( 0 )

    using namespace aries;

    struct AriesMemAllocator
    {
        static void* Alloc( size_t size, unsigned int flags = cudaMemAttachGlobal );
        static void* CloneFromHostMem( void* p, size_t size );
        static void Free( void* p );
        static void FillZero( void* p, size_t size );
        static void MemCopy( void *dst, const void *src, size_t count, cudaMemcpyKind kind = cudaMemcpyDefault );
        static void PrefetchToDeviceAsync( const void *devPtr, size_t count, int dstDevice, cudaStream_t stream = 0 );
        static void MemAdvise( const void* devPtr, size_t count, cudaMemoryAdvise advice, int device );
    };

    template< typename type_t >
    class AriesManagedArray
    {
    public:
        AriesManagedArray( size_t itemCount = 0, bool bInitZero = false, unsigned int flags = cudaMemAttachGlobal )
                : m_itemCount( itemCount )
        {
            if( itemCount > 0 )
                AllocArray( itemCount, bInitZero, flags );
        }

        ~AriesManagedArray()
        {

        }

        type_t* AllocArray( size_t itemCount, bool bInitZero = false, unsigned int flags = cudaMemAttachGlobal )
        {
            ARIES_ASSERT( itemCount > 0, "itemCount: " + to_string( itemCount ) );
            size_t totalBytes = sizeof(type_t) * itemCount;
            type_t* data = ( type_t* )AriesMemAllocator::Alloc( totalBytes, flags );

            m_array.reset( data );
            m_itemCount = itemCount;
            if( bInitZero )
            {
                PrefetchToGpu();
                AriesMemAllocator::FillZero( m_array.get(), totalBytes );
            }

            return data;
        }

        void MemAdvise( cudaMemoryAdvise advice, int device )
        {
            AriesMemAllocator::MemAdvise( GetData(), GetTotalBytes(), advice, device );
        }

        void PrefetchToCpu() const
        {
            if( m_itemCount > 0 )
                AriesMemAllocator::PrefetchToDeviceAsync( m_array.get(), GetTotalBytes(), cudaCpuDeviceId );
        }

        void PrefetchToCpu( size_t offset, size_t itemCount ) const
        {
            ARIES_ASSERT( offset + itemCount < m_itemCount, "offset + itemCount must < m_itemCount" );
            if( itemCount > 0 )
                AriesMemAllocator::PrefetchToDeviceAsync( m_array.get() + offset, itemCount * sizeof(type_t), cudaCpuDeviceId );
        }

        void PrefetchToGpu( int deviceId = ACTIVE_DEVICE_ID ) const
        {
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

        void PrefetchToGpu( size_t offset, size_t itemCount, int deviceId = ACTIVE_DEVICE_ID ) const
        {
            ARIES_ASSERT( offset + itemCount <= m_itemCount, "offset + itemCount must < m_itemCount" );
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
            AriesMemAllocator::PrefetchToDeviceAsync( m_array.get() + offset, itemCount * sizeof(type_t), deviceId );
        }

        std::shared_ptr< AriesManagedArray > CloneWithNoContent( int dstDevice = ACTIVE_DEVICE_ID ) const
        {
            auto tmp = std::make_shared< AriesManagedArray >( m_itemCount, false );
            if( m_itemCount > 0 )
            {
                if( dstDevice == cudaCpuDeviceId )
                    tmp->PrefetchToCpu();
                else
                    tmp->PrefetchToGpu( dstDevice );
            }
            return tmp;
        }

        std::shared_ptr< AriesManagedArray > Clone( int dstDevice = ACTIVE_DEVICE_ID ) const
        {
            auto tmp = std::make_shared< AriesManagedArray >( m_itemCount, false );
            if( m_itemCount > 0 )
            {
                if( dstDevice == cudaCpuDeviceId )
                    tmp->PrefetchToCpu();
                else
                    tmp->PrefetchToGpu( dstDevice );

                AriesMemAllocator::MemCopy( tmp->GetData(), m_array.get(), tmp->GetTotalBytes() );
            }
            return tmp;
        }

        type_t* GetData() const
        {
            return m_array.get();
        }

        type_t* GetData( size_t offset ) const
        {
            ARIES_ASSERT( offset < m_itemCount, "offset must < m_itemCount" );
            return m_array.get() + offset;
        }

        type_t& operator[]( size_t index )
        {
            ARIES_ASSERT( m_array, "m_array is nullptr" );
            return m_array.get()[index];
        }

        const type_t& operator[]( size_t index ) const
        {
            ARIES_ASSERT( m_array, "m_array is nullptr" );
            return m_array.get()[index];
        }

        size_t GetItemCount() const
        {
            return m_itemCount;
        }

        void SetItemCount( size_t itemCount )
        {
            ARIES_ASSERT( itemCount <= m_itemCount, "itemCount: " + to_string( itemCount ) + ", m_itemCount: " + to_string( m_itemCount ) );
            // only reduce item count, we don't actually release memory
            m_itemCount = itemCount;
        }

        size_t GetTotalBytes() const
        {
            return m_itemCount * sizeof(type_t);
        }

        // buf's ownership is transferred to the callee
        void AttachBuffer( type_t* buf, size_t elementCount )
        {
            m_array.reset( buf );
            m_itemCount = elementCount;
        }

        void Reset()
        {
            m_array.reset( nullptr );
            m_itemCount = 0;
        }
        // data's ownership is transferred to the caller
        type_t* ReleaseData()
        {
            ARIES_ASSERT( m_array, "m_array is nullptr" );
            m_itemCount = 0;
            return m_array.release();
        }

        void Dump( size_t dumpCount = 100 ) const
        {
            size_t count = std::min( dumpCount, m_itemCount );
            for( int i = 0; i < count; ++i )
            {
                LOG( INFO )<< m_array.get()[i] << std::endl;
            }
        }

    private:
        struct _MemFreeFunc
        {
            void operator()( void* p )
            {
                AriesMemAllocator::Free( p );
            }
        };

    private:
        std::unique_ptr< type_t, _MemFreeFunc > m_array;
        size_t m_itemCount;
    };

    template< typename type_t >
    class AriesArray
    {
    public:
        AriesArray( size_t itemCount = 0, bool bInitZero = false, bool onGpu = true )
                : m_itemCount( itemCount )
        {
            if( itemCount > 0 )
                AllocArray( itemCount, bInitZero, onGpu );
        }

        ~AriesArray()
        {

        }

        type_t* AllocArray( size_t itemCount, bool bInitZero = false, bool onGpu = true )
        {
            ARIES_ASSERT( itemCount > 0, "itemCount: " + to_string( itemCount ) );
            size_t totalBytes = sizeof(type_t) * itemCount;
            type_t* data;
            bool useManagedMem = !AriesDeviceProperty::GetInstance().IsHighMemoryDevice();
            if( useManagedMem )
            {
                ARIES_CALL_CUDA_API( cudaMallocManaged( &data, sizeof(type_t) * itemCount ) );
                int deviceId;
                ARIES_CALL_CUDA_API( cudaGetDevice( &deviceId ) );
                if( onGpu )
                    ARIES_CALL_CUDA_API( cudaMemPrefetchAsync( data, sizeof(type_t) * itemCount, deviceId ) );
                else
                    ARIES_CALL_CUDA_API( cudaMemPrefetchAsync( data, sizeof(type_t) * itemCount, cudaCpuDeviceId ) );
            }
            else
                ARIES_CALL_CUDA_API( cudaMalloc( &data, sizeof(type_t) * itemCount ) );

            m_array.reset( data );
            m_itemCount = itemCount;
            if( bInitZero )
                ARIES_CALL_CUDA_API( cudaMemset( m_array.get(), 0, totalBytes ) );

            return data;
        }

        std::shared_ptr< AriesArray > CloneWithNoContent() const
        {
            return std::make_shared< AriesArray >( m_itemCount );
        }

        std::shared_ptr< AriesArray > Clone() const
        {
            auto tmp = std::make_shared< AriesArray >( m_itemCount );
            if( m_itemCount > 0 )
            {
                bool useManagedMem = !AriesDeviceProperty::GetInstance().IsHighMemoryDevice();
                if( useManagedMem )
                    ARIES_CALL_CUDA_API( cudaMemcpyAsync( tmp->GetData(), m_array.get(), tmp->GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDefault ) );
                else 
                    ARIES_CALL_CUDA_API( cudaMemcpyAsync( tmp->GetData(), m_array.get(), tmp->GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDeviceToDevice ) );
            }
            return tmp;
        }

        void CopyFromHostMem( void* p, size_t size )
        {
            ARIES_ASSERT( size == GetTotalBytes(), "host memory size is not same as device memory" );
            if( size > 0 )
                ARIES_CALL_CUDA_API( cudaMemcpyAsync( m_array.get(), p, size, cudaMemcpyKind::cudaMemcpyHostToDevice ) );
        }

        std::unique_ptr< type_t[] > DataToHostMemory() const
        {
            std::unique_ptr< type_t[] > tmp( new type_t[m_itemCount] );
            ARIES_CALL_CUDA_API( cudaMemcpy( tmp.get(), m_array.get(), GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
            return tmp;
        }

        type_t* GetData() const
        {
            return m_array.get();
        }

        type_t* GetData( size_t offset ) const
        {
            ARIES_ASSERT( offset < m_itemCount, "offset must < m_itemCount" );
            return m_array.get() + offset;
        }

        void PrefetchToCpu() const
        {
            assert( !AriesDeviceProperty::GetInstance().IsHighMemoryDevice() );
            if( m_itemCount > 0 )
                AriesMemAllocator::PrefetchToDeviceAsync( m_array.get(), GetTotalBytes(), cudaCpuDeviceId );
        }

        void PrefetchToGpu( int deviceId = ACTIVE_DEVICE_ID ) const
        {
            assert( !AriesDeviceProperty::GetInstance().IsHighMemoryDevice() );
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

        void PrefetchToGpu( size_t offset, size_t itemCount, int deviceId = ACTIVE_DEVICE_ID ) const
        {
            ARIES_ASSERT( offset + itemCount <= m_itemCount, "offset + itemCount must < m_itemCount" );
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
            AriesMemAllocator::PrefetchToDeviceAsync( m_array.get() + offset, itemCount * sizeof(type_t), deviceId );
        }

        type_t GetValue( size_t index )
        {
            assert( index < m_itemCount );
            ARIES_ASSERT( m_array, "m_array is nullptr" );
            if( AriesDeviceProperty::GetInstance().IsHighMemoryDevice() )
            {
                type_t val;
                ARIES_CALL_CUDA_API( cudaMemcpy( &val, m_array.get() + index, sizeof( type_t ), cudaMemcpyDeviceToHost ) );
                return val;
            }
            else
            {
                // make sure synchronize is called before get_value!!!
                return m_array.get()[ index ];
            }
        }

        void SetValue( type_t val, size_t index )
        {
            assert( index < m_itemCount );
            ARIES_ASSERT( m_array, "m_array is nullptr" );
            if( AriesDeviceProperty::GetInstance().IsHighMemoryDevice() )
            {
                ARIES_CALL_CUDA_API( cudaMemcpy( m_array.get() + index, &val, sizeof( type_t ), cudaMemcpyHostToDevice ) );
            } 
            else
                m_array.get()[ index ] = val;
        }

        size_t GetItemCount() const
        {
            return m_itemCount;
        }

        void SetItemCount( size_t itemCount )
        {
            ARIES_ASSERT( itemCount <= m_itemCount, "itemCount: " + to_string( itemCount ) + ", m_itemCount: " + to_string( m_itemCount ) );
            // only reduce item count, we don't actually release memory
            m_itemCount = itemCount;
        }

        size_t GetTotalBytes() const
        {
            return m_itemCount * sizeof(type_t);
        }

        // buf's ownership is transferred to the callee
        void AttachBuffer( type_t* buf, size_t elementCount )
        {
            m_array.reset( buf );
            m_itemCount = elementCount;
        }

        void Reset()
        {
            m_array.reset( nullptr );
            m_itemCount = 0;
        }
        // data's ownership is transferred to the caller
        type_t* ReleaseData()
        {
            ARIES_ASSERT( m_array, "m_array is nullptr" );
            m_itemCount = 0;
            return m_array.release();
        }

        void Dump( size_t dumpCount = 100 ) const
        {
            size_t count = std::min( dumpCount, m_itemCount );
            if( count > 0 )
            {
                std::unique_ptr< type_t[] > tmp( new type_t[count] );
                ARIES_CALL_CUDA_API( cudaMemcpy( tmp.get(), m_array.get(), count * sizeof( type_t ), cudaMemcpyKind::cudaMemcpyDeviceToHost ) );

                for( std::size_t i = 0; i < count; ++i )
                {
                    LOG( INFO )<< tmp.get()[i] << std::endl;
                }
            }
        }

    private:
        struct _MemFreeFunc
        {
            void operator()( void* p )
            {
                if( p )
                ARIES_CALL_CUDA_API( cudaFree( p ) );
            }
        };

    private:
        std::unique_ptr< type_t, _MemFreeFunc > m_array;
        size_t m_itemCount;
    };

    using PartitionValueContent = boost::variant< aries_acc::AriesDate, aries_acc::AriesDatetime, int32_t >;
    class AriesDataBuffer
    {
    public:
        struct PartitionInfo
        {
            bool IsValid = false;
            PartitionValueContent MaxValue;
            PartitionValueContent MinValue;
        };
    public:
        AriesDataBuffer( AriesColumnType columnType, size_t itemCount = 0, bool bInitZero = false, unsigned int flags = cudaMemAttachGlobal );

        ~AriesDataBuffer();

        int8_t* AllocArray( size_t itemCount, bool bInitZero = false, unsigned int flags = cudaMemAttachGlobal );

        void MemAdvise( cudaMemoryAdvise advice, int device );

        void SetFromCache( bool bFromCache );

        bool isFromCache();

        void PrefetchToCpu() const;

        void PrefetchToCpu( size_t offset, size_t itemCount ) const;

        void PrefetchToGpu( int deviceId = ACTIVE_DEVICE_ID ) const;

        void PrefetchToGpu( size_t offset, size_t itemCount, int deviceId = ACTIVE_DEVICE_ID ) const;

        int8_t* CopyFromHostMem( void* p, size_t itemCount );

        int8_t* GetData() const;

        int8_t* GetData( size_t offset ) const;

        virtual size_t GetItemCount() const;

        void SetItemCount( size_t itemCount );

        size_t GetItemSizeInBytes() const;

        size_t GetTotalBytes() const;

        // buf's ownership is transferred to the callee
        void AttachBuffer( int8_t* buf, size_t elementCount );

        void Reset();

        // data's ownership is transferred to the caller
        int8_t* ReleaseData();

        AriesColumnType GetDataType() const;

        std::shared_ptr< AriesDataBuffer > Clone( int deviceId = ACTIVE_DEVICE_ID ) const;

        std::shared_ptr< AriesDataBuffer > CloneWithNoContent( int deviceId = ACTIVE_DEVICE_ID ) const;

        std::shared_ptr< AriesDataBuffer > CloneWithNoContent( size_t elementCount, int deviceId = ACTIVE_DEVICE_ID ) const;

        AriesOrderByType GetSortType() const;

        void SetSortType( AriesOrderByType sortType );

        virtual int8_t* GetItemDataAt( int index ) const;

        void Dump( size_t dumpCount = 100 ) const;
        
        std::string ItemToString( int index ) const;

        bool isNullableColumn() const;

        bool isUniqueColumn() const;

        char GetInt8( size_t index ) const;

        nullable_type< int8_t > GetNullableInt8( size_t index ) const;

        bool isInt8DataNull( size_t index ) const;

        unsigned char GetUint8( size_t index ) const;

        nullable_type< uint8_t > GetNullableUint8( size_t index ) const;
        bool isUint8DataNull( size_t index ) const;

        std::string GetInt16AsString( size_t index ) const;

        int16_t GetInt16( size_t index ) const;

        nullable_type< int16_t > GetNullableInt16( size_t index ) const;

        bool isInt16DataNull( size_t index ) const;

        std::string GetUint16AsString( size_t index ) const;

        uint16_t GetUint16( size_t index ) const;

        nullable_type< uint16_t > GetNullableUint16( size_t index ) const;

        bool isUint16DataNull( size_t index ) const;

        std::string GetInt32AsString( size_t index ) const;

        int32_t GetInt32( size_t index ) const;

        nullable_type< int32_t > GetNullableInt32( size_t index ) const;

        bool isInt32DataNull( size_t index ) const;

        std::string GetUint32AsString( size_t index ) const;

        uint32_t GetUint32( size_t index ) const;

        nullable_type< uint32_t > GetNullableUint32( size_t index ) const;

        bool isUint32DataNull( size_t index ) const;

        std::string GetInt64AsString( size_t index ) const;

        int64_t GetInt64( size_t index ) const;

        nullable_type< int64_t > GetNullableInt64( size_t index ) const;

        bool isInt64DataNull( size_t index ) const;

        std::string GetUint64AsString( size_t index ) const;

        uint64_t GetUint64( size_t index ) const;

        nullable_type< uint64_t > GetNullableUint64( size_t index ) const;

        bool isUint64DataNull( size_t index ) const;

        std::string GetFloatAsString( size_t index ) const;

        float GetFloat( size_t index ) const;

        nullable_type< float > GetNullableFloat( size_t index ) const;

        bool isFloatDataNull( size_t index ) const;

        std::string GetDoubleAsString( size_t index ) const;

        double GetDouble( size_t index ) const;

        nullable_type< double > GetNullableDouble( size_t index ) const;

        bool isDoubleDataNull( size_t index ) const;

        std::string GetDecimalAsString( size_t index ) const;

        Decimal GetDecimal( size_t index ) const;

        nullable_type< Decimal > GetNullableDecimal( size_t index ) const;

        std::string GetNullableDecimalAsString( size_t index ) const;

        bool isDecimalDataNull( size_t index ) const;

        std::string GetCompactDecimalAsString( size_t index, uint16_t precision, uint16_t scale ) const;

        CompactDecimal GetCompactDecimal( size_t index ) const;

        std::string GetNullableCompactDecimalAsString( size_t index, uint16_t precision, uint16_t scale ) const;

        bool isCompactDecimalDataNull( size_t index ) const;

        std::string GetString( size_t index ) const;

        std::string GetNullableString( size_t index ) const;

        bool isStringDataNull( size_t index ) const;

        AriesTime* GetTime( size_t index ) const;

        nullable_type< AriesTime >* GetNullableTime( size_t index ) const;

        std::string GetTimeAsString( size_t index ) const;

        std::string GetNullableTimeAsString( size_t index ) const;

        bool isTimeDataNull( size_t index );

        AriesYear* GetYear( size_t index ) const;

        nullable_type< AriesYear >* GetNullableYear( size_t index ) const;

        std::string GetYearAsString( size_t index ) const;

        std::string GetNullableYearAsString( size_t index ) const;

        bool isYearDataNull( size_t index );

        AriesTimestamp* GetTimestamp( size_t index ) const;

        nullable_type< AriesTimestamp >* GetNullableTimestamp( size_t index ) const;

        std::string GetTimestampAsString( size_t index ) const;

        std::string GetNullableTimestampAsString( size_t index ) const;

        bool isTimestampDataNull( size_t index );

        AriesDate* GetDate( size_t index ) const;

        nullable_type< AriesDate >* GetNullableDate( size_t index ) const;

        std::string GetDateAsString( size_t index ) const;

        std::string GetNullableDateAsString( size_t index ) const;

        bool isDateDataNull( size_t index );

        AriesDatetime* GetDatetime( size_t index ) const;

        nullable_type< AriesDatetime >* GetNullableDatetime( size_t index ) const;

        std::string GetDatetimeAsString( size_t index ) const;

        std::string GetNullableDatetimeAsString( size_t index ) const;

        bool isDatetimeDataNull( size_t index );

        uint32_t GetId() const;

        PartitionInfo GetPartitionInfo() const;
        void SetPartitionInfo( const PartitionInfo& info );

    private:
        nullable_type< Decimal > GetNullableCompactDecimal( size_t index ) const;

    private:
        void DumpDate( size_t dumpCount ) const;

        void DumpDatetime( size_t dumpCount ) const;

        void DumpDecimal( size_t dumpCount ) const;

        void DumpCompactDecimal( size_t dumpCount ) const;

        void DumpInt8( size_t dumpCount ) const;

        void DumpInt16( size_t dumpCount ) const;

        void DumpInt32( size_t dumpCount ) const;

        void DumpInt64( size_t dumpCount ) const;

        void DumpString( size_t dumpCount ) const;

        std::string Int32ToString( int index ) const;

        struct _MemFreeFunc
        {
            void operator()( void* p )
            {
                AriesMemAllocator::Free( p );
            }
        };

    private:
        AriesColumnType m_columnType;
        std::unique_ptr< int8_t, _MemFreeFunc > m_array;
        bool m_attachedBuff; // CPU only memory
        size_t m_capacity;
        atomic_size_t m_itemCount;
        AriesOrderByType m_sortType;
        bool m_bFromCache;
        uint32_t m_id;
        PartitionInfo m_partitionInfo;
    };

    template< typename type_t >
    class AriesRangeBuffer : public AriesDataBuffer {
    public:
        AriesRangeBuffer( AriesColumnType columnType, type_t start, size_t count );
        virtual size_t GetItemCount() const override;

        virtual int8_t* GetItemDataAt( int index ) const override;

        inline type_t GetStart() const
        {
            return m_start;
        }
    private:
        type_t m_start;
        size_t m_count;
    };

    using AriesRowPosRangeBuffer = AriesRangeBuffer< int >;
    using AriesRowPosRangeBufferSPtr = std::shared_ptr< AriesRowPosRangeBuffer >;

    typedef int32_t index_t;
#define NULL_INDEX -1

    template< typename type_t >
    using AriesArraySPtr = std::shared_ptr< AriesArray< type_t > >;

    template< typename type_t >
    using AriesManagedArraySPtr = std::shared_ptr< AriesManagedArray< type_t > >;

    using AriesDataBufferSPtr = std::shared_ptr< AriesDataBuffer >;

    using AriesIndicesArray = AriesArray< index_t >;
    using AriesIndicesArraySPtr = AriesArraySPtr< index_t >;

    using AriesManagedIndicesArray = AriesManagedArray< index_t >;
    using AriesManagedIndicesArraySPtr = AriesManagedArraySPtr< index_t >;

    using AriesInt8Array = AriesArray< int8_t >;
    using AriesInt8ArraySPtr = AriesArraySPtr< int8_t >;

    using AriesManagedInt8Array = AriesManagedArray< int8_t >;
    using AriesManagedInt8ArraySPtr = AriesManagedArraySPtr< int8_t >;

    using AriesInt32Array = AriesArray< int32_t >;
    using AriesInt32ArraySPtr = AriesArraySPtr< int32_t >;

    using AriesManagedInt32Array = AriesManagedArray< int32_t >;
    using AriesManagedInt32ArraySPtr = AriesManagedArraySPtr< int32_t >;

    using AriesInt64Array = AriesArray< int64_t >;
    using AriesInt64ArraySPtr = AriesArraySPtr< int64_t >;

    using AriesBoolArray = AriesArray< AriesBool >;
    using AriesBoolArraySPtr = AriesArraySPtr< AriesBool >;

    using AriesUInt64Array = AriesArray< uint64_t >;
    using AriesUInt64ArraySPtr = AriesArraySPtr< uint64_t >;

    using AriesUInt32Array = AriesArray< uint32_t >;
    using AriesUInt32ArraySPtr = AriesArraySPtr< uint32_t >;

    using AriesUInt8Array = AriesArray< uint8_t >;
    using AriesUInt8ArraySPtr = AriesArraySPtr< uint8_t >;

    struct ColumnsToCompare
    {
        AriesDataBufferSPtr LeftColumn;
        AriesDataBufferSPtr RightColumn;
        AriesComparisonOpType OpType;
    };

    struct JoinPair
    {
        AriesInt32ArraySPtr LeftIndices;
        AriesInt32ArraySPtr RightIndices;
        size_t JoinCount = 0;

        inline void swap()
        {
            std::swap( LeftIndices, RightIndices );
        }
    };

    using AriesJoinResult = boost::variant< AriesBoolArraySPtr, AriesInt32ArraySPtr, JoinPair >;

    struct AriesDynamicCodeComparator
    {
        AriesDynamicCodeComparator( const AriesColumnType& type, const AriesDataBufferSPtr& buffer, const AriesComparisonOpType& opType,
                const string& tmpName )
                : Type( type ), LiteralBuffer( buffer ), OpType( opType ), TempName( tmpName )
        {
        }
        AriesColumnType Type;
        AriesDataBufferSPtr LiteralBuffer;
        AriesComparisonOpType OpType;
        string TempName;
    };

    struct AriesDynamicCodeParam
    {
        int ColumnIndex;
        string ParamName;
        AriesColumnType Type;
        AriesComparisonOpType OpType = ( AriesComparisonOpType )-1;
        bool UseDictIndex;
        AriesDynamicCodeParam( int index, const string& name, const AriesColumnType& type, bool useDictIndex = false )
                : ColumnIndex( index ), ParamName( name ), Type( type ), UseDictIndex( useDictIndex )
        {

        }
        bool operator <( const AriesDynamicCodeParam& src ) const
        {
            return ParamName < src.ParamName;
        }
    };

    // empty struct define the NULL literal.
    struct AriesNull
    {

    };

    // for dynamic code we need aries::AriesColumnType to find correct string functions. such as less_t_str< true, false >
    using AriesExprContent = boost::variant< bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double, std::string, Decimal, AriesDate, AriesDatetime, AriesTime, AriesTimestamp, aries_acc::AriesYear, aries::AriesColumnType, AriesDataBufferSPtr >;

    struct AriesLiteralValue
    {
        bool IsNull;
        AriesExprContent Value;
    };

    struct DynamicCodeParams
    {
        std::vector< AriesDynamicCodeParam > params;
        std::vector< AriesDataBufferSPtr > constValues;
        std::vector< AriesDynamicCodeComparator > items;
        std::string functionName;
        std::vector< std::shared_ptr< CUmodule > > CUModules;
        std::vector< AriesDataBufferSPtr > constantValues;
        std::string code;
    };

    struct AriesDynamicCodeInfo
    {
        std::map< std::string, std::string > FunctionKeyNameMapping;
        std::string KernelCode;
        AriesDynamicCodeInfo& operator +=( const AriesDynamicCodeInfo& code );
    };

    using HashIdType = int32_t;
    template< typename type_t >
    struct HashTable
    {
        type_t* keys;
        HashIdType* ids;
        int32_t table_size;
        type_t* bad_values;
        HashIdType* bad_ids;
        int32_t bad_count;
        HashIdType null_value_index;
    };

    struct AriesHashTableMultiKeys
    {
        int count;
        int* keys_length;
        int table_size;

        void* flags_table;

        int8_t** keys_array;
        HashIdType* ids;

        int8_t** bad_values_array;
        HashIdType* bad_ids;
        int32_t bad_count;

        size_t hash_row_count;
    };

    using AriesHashTableMultiKeysUPtr = std::unique_ptr< AriesHashTableMultiKeys >;

    struct AriesHashTable
    {
        void* Keys;
        HashIdType* Ids;
        size_t TableSize;
        size_t HashRowCount;
        void* BadValues;
        HashIdType* BadIds;
        size_t BadCount;
        AriesValueType ValueType;
        HashIdType NullValueIndex;
    };

    using AriesHashTableUPtr = std::unique_ptr< AriesHashTable >;

    enum class HashTableType
        : int32
        {
            SingleKey, MultipleKeys
    };

    struct AriesHashTableWrapper
    {
        HashTableType Type;
        void* Ptr;
    };

    struct AriesHashJoinDataWrapper
    {
        int Count;
        void* Inputs;
    };

    struct ColumnDataIterator
    {
        int8_t** m_data;
        int64_t* m_blockSizePrefixSum;
        int8_t* m_indices;
        AriesValueType m_indiceValueType;

        int8_t* m_nullData;
        int m_blockCount;
        int m_perItemSize;
        bool m_hasNull;

        ARIES_HOST_DEVICE int8_t* GetData( int index ) const;
    };

    struct AriesStarJoinResult
    {
        AriesArraySPtr< HashIdType > FactIds;
        std::vector< AriesArraySPtr< HashIdType > > DimensionIds;
        size_t JoinCount = 0;
    };

    using RowPos = int;

END_ARIES_ACC_NAMESPACE

#endif /* ARIESENGINEDEF_H_ */
