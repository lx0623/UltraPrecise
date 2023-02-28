#include "AriesXLogReader.h"
#include "Configuration.h"

using namespace aries;

BEGIN_ARIES_ENGINE_NAMESPACE

AriesXLogReader::AriesXLogReader( const int64_t& logFileNumber, const int64_t& logFileCount, const std::string& prefix, bool reverse )
: log_file_number( logFileNumber ), log_file_count( logFileCount ), file_prefix( prefix ), reverse( reverse ), buffer( nullptr )
, log_content_size( 0 ), buffer_size( 0 )
{
    current_file_number = reverse ? logFileNumber : logFileCount - logFileNumber + 1;
    openLogFile();
}

AriesXLogReader::~AriesXLogReader()
{
    if ( buffer )
    {
        delete[] buffer;
    }
}

bool AriesXLogReader::openLogFile()
{
    auto log_file_path = Configuartion::GetInstance().GetXLogDataDirectory() + "/" + file_prefix + std::to_string( current_file_number );
    int fd = ::open( log_file_path.c_str(), O_RDONLY );

    ARIES_ASSERT( fd > 0, log_file_path + " dosen't exist" );

    auto end = ::lseek( fd, 0, SEEK_END );

    if ( buffer_size < end )
    {
        if ( buffer )
        {
            delete[] buffer;
        }
        buffer = new int8_t[ end ];
        ARIES_ASSERT( buffer != nullptr, "out of memory" );
        buffer_size = end;
    }

    ::lseek( fd, 0, SEEK_SET );

    auto read_size = ::read( fd, buffer, end );
    ARIES_ASSERT( read_size == end, "cannot read from log file" );

    log_content_size = end;

    close( fd );

    return parse();
}

bool AriesXLogReader::switchLogFile()
{
    if ( !switchToNextFileNumber() )
    {
        return false;
    }

    return openLogFile();
}

bool AriesXLogReader::parse()
{
    auto remain_size = log_content_size;
    int64_t off = 0;
    logs_cursor = 0;

    logs.clear();

    while ( remain_size >= ( int64_t )sizeof( AriesXLogHeader ) )
    {
        auto* header = ( AriesXLogHeader* )( buffer + off );
        auto* data = ( int8_t* )( header ) + sizeof( AriesXLogHeader );

        ARIES_ASSERT( header->magic == ARIES_XLOG_HEADER_MAGIC, "invalid log header magic" );

        logs.emplace_back( std::make_pair( header, data ) );

        off += sizeof( AriesXLogHeader ) + header->dataLength;
        remain_size = log_content_size - off;
    }

    if ( reverse )
    {
        logs_cursor = logs.size() - 1;
    }

    return true;
}

bool AriesXLogReader::switchToNextFileNumber()
{
    if ( reverse )
    {
        current_file_number --;
        if ( current_file_number < ( log_file_number - log_file_count + 1 ) )
        {
            current_file_number ++;
            return false;
        }
    }
    else
    {
        current_file_number ++;
        if ( current_file_number > log_file_number )
        {
            current_file_number --;
            return false;
        }
    }

    return true;
}

std::pair< AriesXLogHeader*, int8_t* > AriesXLogReader::Next()
{
    auto cursor = logs_cursor;

    if ( reverse )
    {
        logs_cursor --;
    } 
    else
    {
        logs_cursor ++;
    }

    if ( cursor < 0 || ( size_t )cursor == logs.size() )
    {
        if ( !switchLogFile() )
        {
            return std::make_pair( nullptr, nullptr );
        }

        return Next();
    }

    return logs[ cursor ];

}

END_ARIES_ENGINE_NAMESPACE
