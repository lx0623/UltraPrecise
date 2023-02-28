#include <mutex>
#include <glog/logging.h>
#include <unistd.h>
#include <sys/stat.h>

#include "AriesXLogManager.h"
#include "Configuration.h"
#include "CpuTimer.h"

using AutoLock = std::lock_guard< std::mutex >;

#define META_FILE_NAME "xlog_meta"

#define META_MAGIC_NUMBER 0xFE00CC77
#define META_MAGIC_NUMBER_SIZE 4

using namespace aries;

const std::string ARIES_XLOG_SPECIAL_PRIFIX( "spceial_" );

BEGIN_ARIES_ENGINE_NAMESPACE

AriesXLogManager::AriesXLogManager()
{
    special_file_handler = std::make_shared< AriesXLogFileHandler >( ARIES_XLOG_SPECIAL_PRIFIX );
    file_handler = std::make_shared< AriesXLogFileHandler >();
}

AriesXLogManager::~AriesXLogManager()
{
}

void AriesXLogManager::SetUp()
{
    AutoLock lock( mutex_for_writing );
    ::mkdir( Configuartion::GetInstance().GetXLogDataDirectory().c_str(), S_IRWXU | S_IRWXG | S_IRWXO );
    file_handler->LoadMetaData();
    file_handler->OpenLogFile();

    special_file_handler->LoadMetaData();
    special_file_handler->OpenLogFile();
}

AriesXLogWriterSPtr AriesXLogManager::GetWriter( const TxId& txid, bool special )
{
    AutoLock lock( mutex_for_writers );
    if ( special )
    {
        if ( special_writers.find( txid ) == special_writers.cend() )
        {
            special_writers[ txid ] = std::make_shared< AriesSpecialXLogWriter >( txid );
        }
        return special_writers[ txid ];
    }
    if ( writers.find( txid ) == writers.cend() )
    {
        writers[ txid ] = std::make_shared< AriesXLogWriter >( txid );
    }
    return writers[ txid ];
}

void AriesXLogManager::ReleaseWriter( const TxId& txid )
{
    AutoLock lock( mutex_for_writers );
    writers.erase( txid );
}

bool AriesXLogManager::WriteBuffer( const std::vector< int8_t* >& buffers, const std::vector< size_t >& buffersSize, bool needFlush, bool special )
{
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif

    AutoLock lock( mutex_for_writing );
    auto& handler = special ? special_file_handler : file_handler;
    auto success = handler->SwitchFileIfNeed();
    ARIES_ASSERT( success, "cannot switch to next log file" );

    for ( size_t i = 0; i < buffers.size(); i++ )
    {
        auto ptr = buffers[ i ];
        auto size = buffersSize[ i ];
        if ( !handler->Write( ptr, size ) )
        {
            return false;
        }
    }

#ifdef ARIES_PROFILE
    LOG( INFO ) << "AriesXLogManager::WriteBuffer time: " << t.end() << "us";
#endif

    bool bRet = true;
    if ( needFlush )
        bRet = handler->Flush();
    return bRet;
}

bool AriesXLogManager::WritePages( const std::vector< AriesXLogPageSPtr >& pages, bool needFlush, bool special )
{
    AutoLock lock( mutex_for_writing );
    auto& handler = special ? special_file_handler : file_handler;
    auto success = handler->SwitchFileIfNeed();
    ARIES_ASSERT( success, "cannot switch to next log file" );

    for ( const auto& page : pages )
    {
        if ( !handler->Write( page->data, page->GetDataSize() ) )
        {
            return false;
        }
    }

    if ( needFlush )
    {
        return handler->Flush();
    }

    return true;
}

AriesXLogReaderSPtr AriesXLogManager::GetReader( bool special, bool reverse )
{
    auto& handler = special ? special_file_handler : file_handler;
    auto prefix = special ? ARIES_XLOG_SPECIAL_PRIFIX : std::string();
    auto log_file_no = handler->GetMetaData().currentLogFileNo;
    auto log_file_count = handler->GetMetaData().logFileCount;
    return std::make_shared< AriesXLogReader >( log_file_no, log_file_count, prefix, reverse );
}

bool AriesXLogManager::SaveCheckPoint( bool isSpecial )
{
    return GetWriter( INVALID_TX_ID, isSpecial )->WriteCommandLog( OperationType::CheckPoint,
                                                 -1,
                                                 INVALID_ROWPOS,
                                                 INVALID_ROWPOS,
                                                 INVALID_ROWPOS,
                                                 true
                                               );
}

bool AriesXLogManager::AddTruncateEvent( TableId tableId )
{
    return GetWriter( INVALID_TX_ID )->WriteCommandLog( OperationType::Truncate,
                                                 tableId,
                                                 INVALID_ROWPOS,
                                                 INVALID_ROWPOS,
                                                 INVALID_ROWPOS,
                                                 true
                                               );
}

AriesXLogFileHandler::AriesXLogFileHandler( const std::string& prefix ) : log_fd( -1 ), file_prefix( prefix )
{
    meta_file_path = Configuartion::GetInstance().GetXLogDataDirectory() + "/" + file_prefix + META_FILE_NAME;
}

bool AriesXLogFileHandler::LoadMetaData()
{
    int fd = ::open( meta_file_path.c_str(), O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR );

    if ( fd <= 0 )
    {
        LOG ( ERROR ) << "cannot open or create meta file";
        return false;
    }

    char buffer[ sizeof( AriesXLogMetaData ) + META_MAGIC_NUMBER_SIZE ];
    auto read_size = ::read( fd, buffer, sizeof( AriesXLogMetaData ) + META_MAGIC_NUMBER_SIZE );
    close( fd );

    if ( read_size == 0 )
    {
        meta_data.currentLogFileNo = 1;
        meta_data.logFileCount = 1;
        return true;
    }
    else if ( read_size == ( sizeof( AriesXLogMetaData ) + META_MAGIC_NUMBER_SIZE ) )
    {
        auto* magic = ( uint32_t* )( buffer );
        if ( *magic != META_MAGIC_NUMBER )
        {
            LOG( ERROR ) << " invalid magic number of meta data: " << *magic;
            return false;
        }

        auto* data = ( AriesXLogMetaData* )( buffer + META_MAGIC_NUMBER_SIZE );
        meta_data.currentLogFileNo = data->currentLogFileNo;
        meta_data.logFileCount = data->logFileCount;
        return true;
    }
    else
    {
        LOG( ERROR ) << "invalid read size: " << read_size;
        return false;
    }
}

bool AriesXLogFileHandler::SaveMetaData()
{
    int fd = ::open( meta_file_path.c_str(), O_WRONLY );
    if ( fd <= 0 )
    {
        LOG( ERROR ) << "cannot open meta file for writing";
        return false;
    }

    char buffer[ sizeof( AriesXLogMetaData ) + META_MAGIC_NUMBER_SIZE ];
    auto* magic = ( uint32_t* )( buffer );
    *magic = META_MAGIC_NUMBER;
    memcpy( buffer + META_MAGIC_NUMBER_SIZE, &meta_data, sizeof( AriesXLogMetaData ) );

    auto writen = ::write( fd, buffer, sizeof( AriesXLogMetaData ) + META_MAGIC_NUMBER_SIZE );
    close( fd );

    if ( writen != sizeof( AriesXLogMetaData ) + META_MAGIC_NUMBER_SIZE )
    {
        LOG( ERROR ) << "invalid writen size: " << writen;
        return false;
    }

    return true;
}

bool AriesXLogFileHandler::Write( void* buffer, size_t size )
{
    auto written = ::write( log_fd, buffer, size );
    if ( written < 0 )
        return false;
    written_size += written;
    return ( size_t )written == size;
}

bool AriesXLogFileHandler::SwitchFileIfNeed()
{
    if ( written_size < MAX_SIZE_PER_FILE )
    {
        return true;
    }

    if ( log_fd > 0 )
    {
        close( log_fd );
        log_fd = -1;
    }

    meta_data.currentLogFileNo ++;
    meta_data.logFileCount ++;

    return OpenLogFile();
}

bool AriesXLogFileHandler::OpenLogFile()
{
    ARIES_ASSERT( log_fd == -1, "log file was already opened" );
    std::string log_file_path = Configuartion::GetInstance().GetXLogDataDirectory() 
        + "/" + file_prefix + std::to_string( meta_data.currentLogFileNo );

    log_fd = ::open( log_file_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR );
    ARIES_ASSERT( log_fd > STDERR_FILENO, "cannot open log file: " + log_file_path );
    auto stderr_no = fileno( stderr );
    ARIES_ASSERT( log_fd > stderr_no, "cannot open log file: " + log_file_path );

    auto off = ::lseek( log_fd, 0, SEEK_END );
    written_size = off;

    auto result = SaveMetaData();
    ARIES_ASSERT( result, "cannot save meta data" );

    return true;
}

bool AriesXLogFileHandler::Flush()
{
    return fsync( log_fd ) == 0;
}

const AriesXLogMetaData& AriesXLogFileHandler::GetMetaData() const
{
    return meta_data;
}

END_ARIES_ENGINE_NAMESPACE
