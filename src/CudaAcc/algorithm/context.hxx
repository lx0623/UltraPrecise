#pragma once

#include <cstdarg>
#include <memory>
#include <vector>
#include <cassert>
#include <exception>
#include <cuda.h>
#include "AriesDefinition.h"
#include "AriesException.h"
#include "AriesDeviceProperty.h"
using namespace aries;
BEGIN_ARIES_ACC_NAMESPACE

    inline std::string stringprintf( const char* format, ... )
    {
        va_list args;
        va_start( args, format );
        int len = vsnprintf( 0, 0, format, args );
        va_end( args );

        // allocate space.
        std::string text;
        text.resize( len );

        va_start( args, format );
        vsnprintf( &text[0], len + 1, format, args );
        va_end( args );

        return text;
    }

    inline std::string device_prop_string( cudaDeviceProp prop )
    {
        int ordinal;
        cudaGetDevice( &ordinal );

        size_t freeMem, totalMem;
        cudaError_t result = cudaMemGetInfo( &freeMem, &totalMem );
        if( cudaSuccess != result )
            throw cuda_exception_t( result );

        double memBandwidth = ( prop.memoryClockRate * 1000.0 ) * ( prop.memoryBusWidth / 8 * 2 ) / 1.0e9;

        std::string s = stringprintf( "%s : %8.3lf Mhz   (Ordinal %d)\n"
                "%d SMs enabled. Compute Capability sm_%d%d\n"
                "FreeMem: %6dMB   TotalMem: %6dMB   %2d-bit pointers.\n"
                "Mem Clock: %8.3lf Mhz x %d bits   (%5.1lf GB/s)\n"
                "ECC %s\n\n", prop.name, prop.clockRate / 1000.0, ordinal, prop.multiProcessorCount, prop.major, prop.minor,
                ( int )( freeMem / ( 1 << 20 ) ), ( int )( totalMem / ( 1 << 20 ) ), 8 * sizeof(int*), prop.memoryClockRate / 1000.0,
                prop.memoryBusWidth, memBandwidth, prop.ECCEnabled ? "Enabled" : "Disabled" );
        return s;
    }

    template< int dummy_arg >
    __global__ void dummy_k()
    {
    }

////////////////////////////////////////////////////////////////////////////////
// context_t
// Derive context_t to add support for streams and a custom allocator.

    struct context_t
    {
        context_t() = default;
        virtual ~context_t() = default;

        // Disable copy ctor and assignment operator. We don't want to let the
        // user copy only a slice.
        context_t( const context_t& rhs ) = delete;
        context_t& operator=( const context_t& rhs ) = delete;

        virtual const cudaDeviceProp& props() const = 0;
        virtual int ptx_version() const = 0;
        virtual cudaStream_t stream() = 0;
        virtual int active_device_id() const = 0;

        // Alloc GPU memory.
        virtual void* alloc( size_t size ) = 0;
        virtual void free( void* p ) = 0;

        // cudaStreamSynchronize or cudaDeviceSynchronize for stream 0.
        virtual void synchronize() = 0;

        virtual cudaEvent_t event() = 0;
        virtual void timer_begin() = 0;
        virtual double timer_end() = 0;
    };

////////////////////////////////////////////////////////////////////////////////
// standard_context_t is a trivial implementation of context_t. Users can
// derive this type to provide a custom allocator.

    class standard_context_t: public context_t
    {
    protected:
        cudaDeviceProp _props;
        int _ptx_version;

        struct device_context
        {
            cudaStream_t _stream = 0;
            cudaEvent_t _timer[2];
            cudaEvent_t _event;
        };
        std::vector< device_context > _device_contexts;

        // Making this a template argument means we won't generate an instance
        // of dummy_k for each translation unit.
        template< int dummy_arg = 0 >
        void init()
        {
            cudaFuncAttributes attr;
            cudaError_t result = cudaFuncGetAttributes( &attr, ( const void* )dummy_k< 0 > );
            if( cudaSuccess != result )
                throw cuda_exception_t( result );
            _ptx_version = attr.ptxVersion;

            int ord;
            cudaGetDevice( &ord );
            cudaGetDeviceProperties( &_props, ord );
            int active_id = active_device_id();
            int deviceCount;
            cudaGetDeviceCount( &deviceCount );
            for( int deviceId = 0; deviceId < deviceCount; ++deviceId )
            {
                cudaSetDevice( deviceId );
                device_context context;
                cudaEventCreate( &context._timer[0] );
                cudaEventCreate( &context._timer[1] );
                cudaEventCreate( &context._event );
                _device_contexts.push_back( context );
            }
            cudaSetDevice( active_id );
        }

        const device_context& active_device_context() const
        {
            int active_id = active_device_id();
            assert( active_id < _device_contexts.size() );
            return _device_contexts[active_id];
        }

    public:
        standard_context_t( bool print_prop = false )
        {
            init();
            if( print_prop )
            {
                printf( "%s\n", device_prop_string( _props ).c_str() );
            }
        }

        virtual ~standard_context_t()
        {
            for( auto& context : _device_contexts )
            {
                cudaEventDestroy( context._timer[0] );
                cudaEventDestroy( context._timer[1] );
                cudaEventDestroy( context._event );
            }
            _device_contexts.clear();
        }

        virtual int active_device_id() const
        {
            int deviceId;
            cudaError_t result = cudaGetDevice( &deviceId );
            if( cudaSuccess != result )
                throw cuda_exception_t( result );
            return deviceId;
        }

        virtual const cudaDeviceProp& props() const
        {
            return _props;
        }

        virtual int ptx_version() const
        {
            return _ptx_version;
        }

        virtual cudaStream_t stream()
        {
            return active_device_context()._stream;
        }

        // Alloc GPU memory.
        virtual void* alloc( size_t size )
        {
            void* p = nullptr;
            if( size )
            {
                cudaError_t result = cudaMallocManaged( &p, size );
                if( cudaSuccess != result )
                {
                    cudaGetLastError();
                    throw cuda_exception_t( result );
                }
            }
            return p;
        }

        virtual void free( void* p )
        {
            if( p )
            {
                cudaError_t result = cudaFree( p );
                if( cudaSuccess != result )
                {
                    cudaGetLastError();
                    throw cuda_exception_t( result );
                }
            }
        }

        virtual void synchronize()
        {
            const device_context& context = active_device_context();
            context._stream ? cudaStreamSynchronize( context._stream ) : cudaDeviceSynchronize();
            auto result = cudaGetLastError();
            if ( cudaSuccess != result )
            {
                throw cuda_exception_t( result );
            }
        }

        virtual cudaEvent_t event()
        {
            return active_device_context()._event;
        }

        virtual void timer_begin()
        {
            const device_context& context = active_device_context();
            cudaEventRecord( context._timer[0], context._stream );
        }

        virtual double timer_end()
        {
            const device_context& context = active_device_context();
            cudaEventRecord( context._timer[1], context._stream );
            cudaEventSynchronize( context._timer[1] );
            float ms;
            cudaEventElapsedTime( &ms, context._timer[0], context._timer[1] );
            return ms;
        }
    };

    template< typename type_t >
    class mem_t
    {
        type_t* _pointer;
        size_t _size;

    public:
        void swap( mem_t& rhs )
        {
            std::swap( _pointer, rhs._pointer );
            std::swap( _size, rhs._size );
        }

        mem_t()
                : _pointer( nullptr ), _size( 0 )
        {
        }

        mem_t& operator=( const mem_t& rhs ) = delete;
        mem_t( const mem_t& rhs ) = delete;

        mem_t( size_t size )
                : _pointer( nullptr ), _size( size )
        {
            if( size > 0 )
            {
                cudaError_t result;
                bool useManagedMem = !AriesDeviceProperty::GetInstance().IsHighMemoryDevice();
                if( useManagedMem )
                    result = cudaMallocManaged( &_pointer, sizeof( type_t ) * size );
                else
                    result = cudaMalloc( &_pointer, sizeof( type_t ) * size );
                
                if( cudaSuccess != result )
                    throw cuda_exception_t( result );
                if( useManagedMem )
                {
                    int deviceId;
                    cudaError_t result = cudaGetDevice( &deviceId );
                    if( cudaSuccess != result )
                        throw cuda_exception_t( result );
                    cudaMemPrefetchAsync( _pointer, sizeof(type_t) * _size, deviceId );
                }
            }
        }

        mem_t( mem_t&& rhs )
                : mem_t()
        {
            swap( rhs );
        }

        mem_t& operator=( mem_t&& rhs )
        {
            swap( rhs );
            return *this;
        }

        ~mem_t()
        {
            free();
        }

        type_t* alloc( size_t size )
        {
            assert( !_pointer );
            _size = size;
            if( size > 0 )
            {
                cudaError_t result;
                bool useManagedMem = !AriesDeviceProperty::GetInstance().IsHighMemoryDevice();
                if( useManagedMem )
                    result = cudaMallocManaged( &_pointer, sizeof( type_t ) * size );
                else
                    result = cudaMalloc( &_pointer, sizeof( type_t ) * size );
                if( cudaSuccess != result )
                    throw cuda_exception_t( result );
                if( useManagedMem )
                {
                    int deviceId;
                    cudaError_t result = cudaGetDevice( &deviceId );
                    if( cudaSuccess != result )
                        throw cuda_exception_t( result );
                    cudaMemPrefetchAsync( _pointer, sizeof(type_t) * _size, deviceId );
                }
            }
            return _pointer;
        }

        void free()
        {
            if( _pointer )
            {
                cudaError_t result = cudaFree( _pointer );
                if( cudaSuccess != result )
                {
                    cudaGetLastError();
                    throw cuda_exception_t( result );
                }
            }
            _pointer = nullptr;
            _size = 0;
        }

        type_t* release_data()
        {
            type_t* data = _pointer;
            _pointer = nullptr;
            _size = 0;
            return data;
        }

        size_t size() const
        {
            return _size;
        }

        type_t* data() const
        {
            return _pointer;
        }

        type_t get_value( size_t index )
        {
            assert( index < _size );
            if( AriesDeviceProperty::GetInstance().IsHighMemoryDevice() )
            {
                type_t val;
                cudaMemcpy( &val, _pointer + index, sizeof(type_t), cudaMemcpyDeviceToHost );
                return val;
            }
            else
            {
                // make sure synchronize is called before get_value!!!
                return _pointer[ index ];
            }
        }

        void set_value( type_t val, size_t index )
        {
            assert( index < _size );
            if( AriesDeviceProperty::GetInstance().IsHighMemoryDevice() )
                cudaMemcpy( _pointer + index, &val, sizeof(type_t), cudaMemcpyHostToDevice );
            else
                _pointer[ index ] = val;
        }

        // Return a deep copy of this container.
        mem_t clone()
        {
            mem_t cloned( _size );
            if( _size > 0 )
                cudaMemcpy( cloned.data(), _pointer, sizeof(type_t) * _size, cudaMemcpyDefault );
            return cloned;
        }
    };

    template< typename type_t >
    class managed_mem_t
    {
        context_t* _context;
        type_t* _pointer;
        size_t _size;

    public:
        void swap( managed_mem_t& rhs )
        {
            std::swap( _context, rhs._context );
            std::swap( _pointer, rhs._pointer );
            std::swap( _size, rhs._size );
        }

        managed_mem_t()
                : _context( nullptr ), _pointer( nullptr ), _size( 0 )
        {
        }

        managed_mem_t( context_t& context )
                : _context( &context ), _pointer( nullptr ), _size( 0 )
        {
        }

        managed_mem_t& operator=( const managed_mem_t& rhs ) = delete;
        managed_mem_t( const managed_mem_t& rhs ) = delete;

        managed_mem_t( size_t size, context_t& context, bool bPreFetchToDevice = true )
                : _context( &context ), _pointer( nullptr ), _size( size )
        {
            _pointer = ( type_t* )context.alloc( sizeof(type_t) * size );
            if( size > 1 && bPreFetchToDevice )
                PrefetchToGpu();
        }

        managed_mem_t( managed_mem_t&& rhs )
                : managed_mem_t()
        {
            swap( rhs );
        }

        managed_mem_t& operator=( managed_mem_t&& rhs )
        {
            swap( rhs );
            return *this;
        }

        ~managed_mem_t()
        {
            free();
        }

        type_t* alloc( size_t size, bool bPreFetchToDevice = true )
        {
            assert( _context );
            _pointer = ( type_t* )_context->alloc( sizeof(type_t) * size );
            _size = size;
            if( size > 1 && bPreFetchToDevice )
                PrefetchToGpu();
            return _pointer;
        }

        void free()
        {
            if( _context && _pointer )
                _context->free( _pointer );
            _pointer = nullptr;
            _size = 0;
        }

        type_t* release_data()
        {
            type_t* data = _pointer;
            _pointer = nullptr;
            _size = 0;
            return data;
        }

        context_t& context()
        {
            return *_context;
        }

        size_t size() const
        {
            return _size;
        }

        type_t* data() const
        {
            return _pointer;
        }

        // Return a deep copy of this container.
        managed_mem_t clone()
        {
            managed_mem_t cloned( _size, *_context );
            if( _size > 0 )
                cudaMemcpy( cloned.data(), _pointer, sizeof(type_t) * _size, cudaMemcpyDefault );
            return cloned;
        }

        void PrefetchToCpu() const
        {
            if( _size > 0 )
                cudaMemPrefetchAsync( _pointer, sizeof(type_t) * _size, cudaCpuDeviceId );
        }

        void PrefetchToGpu( int deviceId = -1 ) const
        {
            if( _size > 0 )
            {
                if( deviceId == -1 )
                    //prefetch to active device.
                    cudaMemPrefetchAsync( _pointer, sizeof(type_t) * _size, _context->active_device_id() );
                else
                    cudaMemPrefetchAsync( _pointer, sizeof(type_t) * _size, deviceId );
            }
        }
    };

END_ARIES_ACC_NAMESPACE
