#pragma once
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include "AriesAssert.h"

using namespace std;

#define ARIES_FUNC_LOG( extraMsg ) \
LOG(INFO) << "========================================" << typeid(this).name() << ":" << __func__ << extraMsg

#define ARIES_FUNC_LOG_BEGIN ARIES_FUNC_LOG( " BEGIN" )
#define ARIES_FUNC_LOG_END   ARIES_FUNC_LOG( " END" )

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define SET_BIT_FLAG(flags, index)                      \
do                                                      \
{                                                       \
    int8_t *target = ( flags ) + ( ( index ) / 8 );     \
    *target = *target | (1 << (7 - ( index ) % 8 ) );   \
} while (0)

#define CLEAR_BIT_FLAG(flags, index)                    \
do                                                      \
{                                                       \
    int8_t *target = ( flags ) + ( ( index ) / 8 );     \
    *target = *target & ~( 1 << ( 7 - ( index ) % 8 ) );\
} while (0)

#define GET_BIT_FLAG(flags, index, out)                                 \
do                                                                      \
{                                                                       \
    uint8_t target = *( ( uint8_t* )( ( flags ) + ( ( index ) / 8) ) ); \
    out = ( (uint8_t )( target << ( ( index ) % 8 ) ) ) >> 7;           \
} while (0)

const static int MAX_BATCH_WRITE_BUFF_SIZE = 32 * 4 * 1024 * 1024; // 128M
const static int MIN_BATCH_WRITE_BUFF_SIZE = 4096; // 4K
const static int DEFAULT_BATCH_WRITE_BUFF_SIZE = 4096; // 4K
size_t block_size( int fd );
int64_t filesize(int fd);

using IfstreamSPtr = shared_ptr<ifstream>;
using MutexSPtr = shared_ptr<mutex>;

class WRITE_BUFF_INFO
{
public:
    WRITE_BUFF_INFO(size_t argBuffSize)
        : buffSize(argBuffSize),
          dataSize(0)
    {
        if ( buffSize > 0 )
        {
            buff.reset( new uchar[ buffSize ] );
        }
    }
    bool isFull()
    {
        return dataSize == buffSize;
    }
    // append data to buffer
    // return appended size
    size_t append( uchar* data, size_t size )
    {
        ARIES_ASSERT( buffSize > 0, "Invalid buffer size " + std::to_string( buffSize ) );
        if ( !isFull() )
        {
            auto maxSize = std::min( buffSize - dataSize, size );
            memcpy( buff.get() + dataSize, data, maxSize );
            dataSize += maxSize;
            return maxSize;
        }
        else
        {
            return 0;
        }

    }
    void clear()
    {
        dataSize = 0;
    }
    bool empty()
    {
        return dataSize == 0;
    }
    uchar* get()
    {
        return buff.get();
    }
    size_t getDataSize()
    {
        return dataSize;
    }

private:
    const size_t buffSize;
    size_t dataSize;
    std::shared_ptr<uchar[]> buff;
};
bool flushWriteBuff( int fd,
                     std::shared_ptr<WRITE_BUFF_INFO>& writeBuff );
bool batchWrite( int fd,
                 std::shared_ptr<WRITE_BUFF_INFO>& writeBuff,
                 uchar* data,
                 size_t dataSize,
                 bool flush  = true );
std::vector< std::string > listFiles( const string& dir, bool file = true );
void partitionItems( const size_t itemCount,
                     const size_t partitionCount,
                     vector<size_t>& partitionItemCounts,
                     vector<size_t> *partitionItemStartIndex = nullptr );
class fd_helper
{
public:
    fd_helper( int fd ) : m_fd( fd ) {}
    int GetFd() const { return m_fd; };
    ~fd_helper()
    {
        if ( m_fd > 0 )
        {
            close( m_fd );
            m_fd = -1;
        }
    }

private:
    int m_fd;
};

using fd_helper_ptr = std::shared_ptr< fd_helper >;
