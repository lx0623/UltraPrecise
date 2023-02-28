#pragma once

#include <mutex>
#include <thread>
#include <memory>

#include "AriesAssert.h"
#include "AriesTuple.h"


#define DEFAULT_PAGE_SIZE (16*1024) // 8K
#define MAX_XLOG_PAGE_COUNT 256

BEGIN_ARIES_ENGINE_NAMESPACE

struct AriesXLogPage
{
    int8_t* data;
    size_t  offset;
    size_t  size;

public:
    AriesXLogPage( size_t totalSize = DEFAULT_PAGE_SIZE )
    {
        size = totalSize;
        offset = 0;
        data = ( int8_t* )malloc( size );
        ARIES_ASSERT( data != NULL, "out of memory" );
    }

    ~AriesXLogPage()
    {
        free( data );
    }

    void* Alloc( size_t requireSize )
    {
        if ( requireSize > ( size - offset ) )
        {
            return nullptr;
        }

        void* ptr = ( ( uint8_t* )data + offset );
        offset += requireSize;
        return ptr;
    }

    void Release( size_t size )
    {
        offset -= size;
    }

    size_t Available()
    {
        return size - offset;
    }

    size_t GetDataSize()
    {
        return offset;
    }
};

using AriesXLogPageSPtr = std::shared_ptr< AriesXLogPage >;

END_ARIES_ENGINE_NAMESPACE
