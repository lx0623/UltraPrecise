#pragma once

#include "AriesXLog.h"
#include "AriesXLogPage.h"

BEGIN_ARIES_ENGINE_NAMESPACE

typedef bool ( *AriesXLogWriteHandler )( const std::vector< AriesXLogPageSPtr >& pages, bool needFlush );
typedef bool ( *AriesXLogWriteBufferHandler )( const std::vector< int8_t* >& buffers, const std::vector< size_t >& buffersSize, bool needFlush );

class AriesXLogWriter
{
private:
    std::vector< AriesXLogPageSPtr > pages;
    TxId txid;
    bool hasDataWritten;
    AriesXLogWriteHandler writeHandler;
    AriesXLogWriteBufferHandler writeBufferHandler;

public:
    AriesXLogWriter( TxId txid );

    /**
     * @return bool 返回是否成功写入日志，如果失败 transaction 应该 abort
     */
    bool WriteCommandLog( OperationType operation,
                   TableId tableId,
                   RowPos sourcePos,
                   RowPos targetPos,
                   RowPos initialPos,
                   bool forceFlush = false
                 );

    bool WriteSingleInsertLog( TableId tableId,
                               RowPos targetPos,
                               const std::vector< int8_t* >& columnsData,
                               const TupleParserSPtr& parser,
                               bool forceFlush = false
                             );

    bool WriteBatchInsertLog( TableId tableId,
                              const std::vector< RowPos >& targetPoses,
                              const std::vector< int8_t* >& columnsData,
                              const TupleParserSPtr& parser,
                              bool forceFlush = false
                            );

    bool WriteDictLog( OperationType operation,
                       TableId tableId,
                       RowPos sourcePos,
                       RowPos targetPos,
                       RowPos initialPos,
                       int8_t* data,
                       size_t dataSize );

    /**
     * @return bool 返回是否成功写入日志，如果失败 transaction 应该 abort
     */
    bool Commit();

    /**
     * @return bool 返回是否成功写入日志，如果失败 transaction 应该 abort
     */
    bool Abort();

protected:
    void SetWriteHandler( const AriesXLogWriteHandler& handler, const AriesXLogWriteBufferHandler& writeBufferHandler );

private:
    bool assembleLogData( OperationType operation,
                          TableId tableId,
                          RowPos sourcePos,
                          RowPos targetPos,
                          RowPos initialPos,
                          void* buffer,
                          size_t& size,
                          bool needCalcSize = true
                        );
    bool assembleDictLogData( OperationType operation,
                              TableId tableId,
                              RowPos sourcePos,
                              RowPos targetPos,
                              RowPos initialPos,
                              int8_t* data,
                              size_t dataSize,
                              void* buffer );
    AriesXLogPageSPtr createNewPage( size_t size );
    int8_t* AllocXLogBuffer( const size_t size );
};

using AriesXLogWriterSPtr = std::shared_ptr< AriesXLogWriter >;

class AriesSpecialXLogWriter : public AriesXLogWriter {
public:
    AriesSpecialXLogWriter( TxId txid );
};

END_ARIES_ENGINE_NAMESPACE
