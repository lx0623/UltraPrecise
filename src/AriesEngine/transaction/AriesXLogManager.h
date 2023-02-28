#pragma once

#include <cstdint>
#include <atomic>
#include <map>
#include <vector>
#include <memory>
#include <mutex>

#include "AriesTuple.h"
#include "AriesXLog.h"
#include "AriesXLogWriter.h"
#include "AriesDefinition.h"
#include "AriesXLogReader.h"

BEGIN_ARIES_ENGINE_NAMESPACE

#define MAX_SIZE_PER_FILE ( 16 * 1024 * 1024 ) // 16M logs per file


struct AriesXLogMetaData
{
    int64_t logFileCount;
    int64_t currentLogFileNo;
} ARIES_PACKED;

class AriesXLogFileHandler
{
private:
    /**
     * 当前文件 fd
     */
    int log_fd;

    /**
     * 当前文件已写入多少字节数据
     */
    size_t written_size;

    std::mutex mutex_for_writers;
    std::map< TxId, AriesXLogWriterSPtr > writers;

    std::mutex mutex_for_writing;

    std::string meta_file_path;
    AriesXLogMetaData meta_data;
    std::string file_prefix;

public:
    AriesXLogFileHandler( const std::string& prefix = std::string() );

    bool LoadMetaData();
    bool SaveMetaData();
    bool Write( void* buffer, size_t size );
    bool SwitchFileIfNeed();
    bool OpenLogFile();

    bool Flush();

    const AriesXLogMetaData& GetMetaData() const;
};

using AriesXLogFileHandlerSPtr = std::shared_ptr< AriesXLogFileHandler >;

class AriesXLogManager
{
private:

    std::mutex mutex_for_writers;
    std::map< TxId, AriesXLogWriterSPtr > writers;
    std::map< TxId, AriesXLogWriterSPtr > special_writers;

    std::mutex mutex_for_writing;

    AriesXLogFileHandlerSPtr file_handler;
    AriesXLogFileHandlerSPtr special_file_handler;

    AriesXLogManager();

public:
    static AriesXLogManager& GetInstance()
    {
        static AriesXLogManager instance;
        return instance;
    }

    ~AriesXLogManager();

    void SetUp();

    AriesXLogWriterSPtr GetWriter( const TxId& txid, bool special = false );
    void ReleaseWriter( const TxId& txid );
    bool WritePages( const std::vector< AriesXLogPageSPtr >& pages, bool needFlush, bool special = false );
    bool WriteBuffer( const std::vector< int8_t* >& buffers, const std::vector< size_t >& buffersSize, bool needFlush, bool special = false );

    AriesXLogReaderSPtr GetReader( bool special = false, bool reverse = true );
    bool SaveCheckPoint( bool isSpecial = false );
    bool AddTruncateEvent( TableId tableId );
};

END_ARIES_ENGINE_NAMESPACE
