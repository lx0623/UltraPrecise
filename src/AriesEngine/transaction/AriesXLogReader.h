#pragma once

#include "AriesXLog.h"
#include "AriesXLogPage.h"

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesXLogReader
{
public:
    AriesXLogReader( const int64_t& logFileNumber, const int64_t& logFileCount, const std::string& prefix = std::string(), bool reverse = true );
    ~AriesXLogReader();

    std::pair< AriesXLogHeader*, int8_t* > Next();

private:
    bool openLogFile();
    bool switchLogFile();

    bool switchToNextFileNumber();

    bool parse();

private:
    int64_t log_file_number;
    int64_t log_file_count;

    int64_t current_file_number;
    std::string file_prefix;

    bool reverse;

    int8_t* buffer;
    int64_t log_content_size;
    int64_t buffer_size;

    std::vector< std::pair< AriesXLogHeader*, int8_t* > > logs;
    int64_t logs_cursor;
};

using AriesXLogReaderSPtr = std::shared_ptr< AriesXLogReader >;

END_ARIES_ENGINE_NAMESPACE
