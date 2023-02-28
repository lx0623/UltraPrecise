#include "AriesXLogWriter.h"
#include "AriesXLogManager.h"
#include "utils/utils.h"

BEGIN_ARIES_ENGINE_NAMESPACE

bool DefaultWriteHandler( const std::vector< AriesXLogPageSPtr >& pages, bool needFlush )
{
    return AriesXLogManager::GetInstance().WritePages( pages, needFlush, false );
}

bool DefaultWriteBufferHandler( const std::vector< int8_t* >& buffers, const std::vector< size_t >& buffersSize, bool needFlush )
{
    return AriesXLogManager::GetInstance().WriteBuffer( buffers, buffersSize, needFlush, false );
}


AriesXLogWriter::AriesXLogWriter( TxId txid ) : txid( txid ), hasDataWritten( false )
{
    pages.emplace_back( std::make_shared< AriesXLogPage >() );
    SetWriteHandler( DefaultWriteHandler, DefaultWriteBufferHandler );
}

bool AriesXLogWriter::WriteCommandLog( OperationType operation,
                                TableId tableId,
                                RowPos sourcePos,
                                RowPos targetPos,
                                RowPos initialPos,
                                bool forceFlush
                              )
{
    size_t available = 0;
    void* buffer = nullptr;
    AriesXLogPageSPtr page = nullptr;
    if ( pages.size() > 0 )
    {
        page = pages[ pages.size() - 1 ];
        available = page->Available();
        buffer = page->Alloc( available );
    }

    auto size = available;

    /**
     * 尝试在 available 大小的内存中组装 xlog 数据，这里 size 这个参数会被设置为这条 log 实际需要多少内存。
     */
    assembleLogData( operation, tableId, sourcePos, targetPos, initialPos, buffer, size );

    if ( size > available )
    {
        if ( buffer != nullptr )
        {
            page->Release( available );
        }
        page = createNewPage( size );
        buffer = page->Alloc( size );
        assembleLogData( operation, tableId, sourcePos, targetPos, initialPos, buffer, size, false );
    }
    else
    {
        /**
         * 实际使用的内存大小是 size，有 available - size 个字节的内存未被使用
         */
        if ( buffer != nullptr )
        {
            page->Release( available - size );
        }
    }

    if ( pages.size() >= MAX_XLOG_PAGE_COUNT || forceFlush )
    {
        LOG_IF( INFO, pages.size() >= MAX_XLOG_PAGE_COUNT ) << "here need to flush pages since too many pages";
        if ( writeHandler( pages, forceFlush ) )
        {
            pages.clear();
            hasDataWritten = true;
            return true;
        }

        // TODO: when clear the logs?
        LOG( ERROR ) << "write log failed";
        return false;
    }

    return true;
}

int8_t* AriesXLogWriter::AllocXLogBuffer( const size_t size )
{

    size_t available = 0;
    int8_t* buffer = nullptr;
    AriesXLogPageSPtr page = nullptr;
    if ( pages.size() > 0 )
    {
        page = pages[ pages.size() - 1 ];
        available = page->Available();
        if ( size <= available )
        {
            buffer = ( int8_t* )page->Alloc( size );
        } 
    }

    if ( !buffer )
    {
        page = createNewPage( size );
        buffer = ( int8_t* )page->Alloc( size );
    }
    return buffer;
}

bool AriesXLogWriter::WriteSingleInsertLog( TableId tableId,
                                            RowPos targetPos,
                                            const std::vector< int8_t* >& columnsData,
                                            const TupleParserSPtr& parser,
                                            bool forceFlush
                                          )
{
    size_t row_count = 1;

    std::vector< size_t > columnsSize;

    size_t data_size = sizeof( AriesBatchInsertInfo );
    data_size += sizeof( size_t ) * parser->GetColumnsCount(); // Columns Size
    data_size += sizeof( RowPos ); // RowPoses
    for ( size_t i = 0; i < columnsData.size(); i++ )
    {
        const auto& type = parser->GetColumnType( i + 1 );
        auto column_size = type.GetDataTypeSize();
        columnsSize.emplace_back( column_size );
        data_size += column_size;
    }

    auto total_size = data_size + sizeof( AriesXLogHeader );

    int8_t* buffer = AllocXLogBuffer( total_size );

    AriesXLogHeader* header = ( AriesXLogHeader *)buffer;
    header->operation = OperationType::InsertBatch;
    header->initialPos = INVALID_ROWPOS;
    header->magic = ARIES_XLOG_HEADER_MAGIC;
    header->sourcePos = INVALID_ROWPOS;
    header->targetPos = INVALID_ROWPOS;
    header->tableId = tableId;
    header->txid = txid;
    header->dataLength = data_size;

    int off = sizeof( AriesXLogHeader );
    AriesBatchInsertInfo* info = ( AriesBatchInsertInfo* )( buffer + off );
    info->rowCount = row_count;
    info->columnCount = columnsData.size();
    off += sizeof( AriesBatchInsertInfo );

    memcpy( buffer + off, columnsSize.data(), sizeof( size_t ) * columnsSize.size() );
    off += sizeof( size_t ) * columnsSize.size();

    *( RowPos* )( buffer + off ) = targetPos;
    off += sizeof( RowPos );

    for ( size_t i = 0; i < columnsData.size(); i++ )
    {
        const auto& type = parser->GetColumnType( i + 1 );
        auto column_size = type.GetDataTypeSize();
        memcpy( buffer + off, columnsData[ i ], column_size );
        off += column_size;
    }

    if ( pages.size() >= MAX_XLOG_PAGE_COUNT || forceFlush )
    {
        LOG_IF( INFO, pages.size() >= MAX_XLOG_PAGE_COUNT ) << "here need to flush pages since too many pages";
        if ( writeHandler( pages, forceFlush ) )
        {
            pages.clear();
            hasDataWritten = true;
            return true;
        }

        // TODO: when clear the logs?
        LOG( ERROR ) << "write log failed";
        return false;
    }

    return true;
}


/**
 * 批量插入的数据在 xlog 中的存储格式
 * |-----------------|-----------|
 * | AriesXLogHeader |   DATA    |
 * |-----------------|-----------|
 * 其中 DATA 部分：
 * |-----------------------|---------------|-----------|---------|---------|----------|
 * | AriesBatchInsertInfo  |  Columns Size | RowPoses  | Column1 |   ...   | Column N |
 * |-----------------------|---------------|-----------|---------|---------|----------|
 */
/*
bool AriesXLogWriter::WriteBatchInsertLog( TableId tableId,
                                           const std::vector< RowPos >& targetPoses,
                                           const std::vector< int8_t* >& columnsData,
                                           const TupleParserSPtr& parser,
                                           bool forceFlush
                                         )
{
    auto row_count = targetPoses.size();

    std::vector< size_t > columnsSize;

    AriesXLogHeader header;
    header.tableId = tableId;
    header.operation = OperationType::InsertBatch;
    header.initialPos = INVALID_ROWPOS;
    header.magic = ARIES_XLOG_HEADER_MAGIC;
    header.sourcePos = INVALID_ROWPOS;
    header.targetPos = INVALID_ROWPOS;
    header.txid = txid;

    std::vector< int8_t* > buffers;
    std::vector< size_t > buffersSize;

    AriesBatchInsertInfo info;
    info.rowCount = row_count;
    info.columnCount = columnsData.size();

    buffers.emplace_back( ( int8_t* )&header );
    buffersSize.emplace_back( sizeof( AriesXLogHeader ) );

    buffers.emplace_back( ( int8_t* )&info );
    buffersSize.emplace_back( sizeof( AriesBatchInsertInfo ) );

    size_t row_size = 0;
    for ( size_t i = 0; i < columnsData.size(); i++ )
    {
        const auto& type = parser->GetColumnType( i + 1 );
        auto column_size = type.GetDataTypeSize();
        row_size += column_size;
        columnsSize.emplace_back( column_size );
    }

    buffers.emplace_back( ( int8_t* )( columnsSize.data() ) );
    buffersSize.emplace_back( columnsSize.size() * sizeof( uint64_t ) );

    buffers.emplace_back( ( int8_t* )( targetPoses.data() ) );
    buffersSize.emplace_back( sizeof( RowPos ) * row_count );

    for ( size_t i = 0; i < columnsData.size(); i++ )
    {
        const auto& type = parser->GetColumnType( i + 1 );
        auto column_size = type.GetDataTypeSize();
        buffers.emplace_back( columnsData[ i ] );
        buffersSize.emplace_back( column_size * row_count );
    }

    row_size += sizeof( RowPos );
    header.dataLength = int32_t( row_size * row_count + buffersSize[ 1 ] + buffersSize[ 2 ] );

    hasDataWritten = true;
    return writeBufferHandler( buffers, buffersSize, forceFlush );
}
*/
bool AriesXLogWriter::WriteBatchInsertLog( TableId tableId,
                                           const std::vector< RowPos >& targetPoses,
                                           const std::vector< int8_t* >& columnsData,
                                           const TupleParserSPtr& parser,
                                           bool forceFlush
                                         )
{
    auto row_count = targetPoses.size();

    size_t total_size = sizeof( AriesXLogHeader ) + sizeof( AriesBatchInsertInfo );
    total_size += sizeof( size_t ) * parser->GetColumnsCount();;
    total_size += sizeof( RowPos ) * row_count;

    std::vector< size_t > columnsSize;
    for ( size_t i = 0; i < columnsData.size(); i++ )
    {
        const auto& type = parser->GetColumnType( i + 1 );
        auto column_size = type.GetDataTypeSize();
        columnsSize.emplace_back( column_size );
        total_size += column_size * row_count;
    }

    int8_t* buffer = AllocXLogBuffer( total_size );

    AriesXLogHeader* header = ( AriesXLogHeader *)buffer;
    header->tableId = tableId;
    header->operation = OperationType::InsertBatch;
    header->initialPos = INVALID_ROWPOS;
    header->magic = ARIES_XLOG_HEADER_MAGIC;
    header->sourcePos = INVALID_ROWPOS;
    header->targetPos = INVALID_ROWPOS;
    header->txid = txid;
    header->dataLength = 0;

    int off = sizeof( AriesXLogHeader );

    size_t dataSize = sizeof( AriesBatchInsertInfo );
    AriesBatchInsertInfo* info = ( AriesBatchInsertInfo* )( buffer + off );
    info->rowCount = row_count;
    info->columnCount = columnsData.size();

    header->dataLength += dataSize;
    off += dataSize;

    dataSize = sizeof( size_t ) * columnsSize.size();
    memcpy( buffer + off, columnsSize.data(), dataSize );

    header->dataLength += dataSize;
    off += dataSize;

    dataSize = sizeof( RowPos ) * row_count;
    memcpy( buffer + off, ( int8_t* )( targetPoses.data() ), dataSize );

    header->dataLength += dataSize;
    off += dataSize;

    for ( size_t i = 0; i < columnsData.size(); i++ )
    {
        const auto& type = parser->GetColumnType( i + 1 );
        auto column_size = type.GetDataTypeSize();
        dataSize = column_size * row_count;
        memcpy( buffer + off, columnsData[ i ], dataSize );

        header->dataLength += dataSize;
        off += dataSize;
    }

    if ( pages.size() >= MAX_XLOG_PAGE_COUNT || forceFlush )
    {
        LOG_IF( INFO, pages.size() >= MAX_XLOG_PAGE_COUNT ) << "here need to flush pages since too many pages";
        LOG( INFO ) << "call writeHandler, force flush: " << forceFlush;
        if ( writeHandler( pages, forceFlush ) )
        {
            pages.clear();
            hasDataWritten = true;
            return true;
        }

        // TODO: when clear the logs?
        LOG( ERROR ) << "write log failed";
        return false;
    }
    return true;
}

bool AriesXLogWriter::WriteDictLog( OperationType operation,
                                    TableId tableId,
                                    RowPos sourcePos,
                                    RowPos targetPos,
                                    RowPos initialPos,
                                    int8_t* data,
                                    size_t dataSize )
{
    size_t available = 0;
    void* buffer = nullptr;
    AriesXLogPageSPtr page = nullptr;
    if ( pages.size() > 0 )
    {
        page = pages[ pages.size() - 1 ];
        available = page->Available();
    }

    size_t totalSize = dataSize + sizeof( AriesXLogHeader );
    if ( available < totalSize )
        page = createNewPage( totalSize );

    buffer = page->Alloc( totalSize );
    assembleDictLogData( operation, tableId, sourcePos, targetPos, initialPos, data, dataSize, buffer );

    LOG_IF( INFO, pages.size() >= MAX_XLOG_PAGE_COUNT ) << "here need to flush pages since too many pages";
    if ( writeHandler( pages, false ) )
    {
        pages.clear();
        hasDataWritten = true;
        return true;
    }

    // TODO: when clear the logs?
    LOG( ERROR ) << "write log failed";
    return false;
}

bool AriesXLogWriter::Commit()
{
    if ( !hasDataWritten && pages.size() == 1 && pages[ 0 ]->GetDataSize() == 0 )
    {
        pages.clear();
        return true;
    }

    return WriteCommandLog( OperationType::Commit,
                     -1,
                     INVALID_ROWPOS,
                     INVALID_ROWPOS,
                     INVALID_ROWPOS,
                     true
                   );
}

bool AriesXLogWriter::Abort()
{
    if ( !hasDataWritten && pages.size() == 1 && pages[ 0 ]->GetDataSize() == 0 )
    {
        pages.clear();
        return true;
    }

    if ( hasDataWritten )
    {
        return WriteCommandLog( OperationType::Abort,
                        -1,
                        INVALID_ROWPOS,
                        INVALID_ROWPOS,
                        INVALID_ROWPOS,
                        true
                      );
    }

    pages.clear();
    return true;
}

bool AriesXLogWriter::assembleLogData( OperationType operation,
                                       TableId tableId,
                                       RowPos sourcePos,
                                       RowPos targetPos,
                                       RowPos initialPos,
                                       void* buffer,
                                       size_t& size,
                                       bool needCalcSize
                                     )
{
    size_t data_size;
    if ( needCalcSize )
    {
        data_size = sizeof( AriesXLogHeader );

        if ( data_size > size )
        {
            size = data_size;
            return false;
        }

        size = data_size;
    }
    else
    {
        data_size = size;
    }

    auto* header = ( AriesXLogHeader* )( buffer );
    header->magic = ARIES_XLOG_HEADER_MAGIC;
    header->operation = operation;
    header->tableId = tableId;
    header->sourcePos = sourcePos;
    header->targetPos = targetPos;
    header->initialPos = initialPos;
    header->txid = txid;
    header->dataLength = static_cast< int32_t >( data_size - sizeof( AriesXLogHeader ) );

    return true;
}


bool AriesXLogWriter::assembleDictLogData( OperationType operation,
                                           TableId tableId,
                                           RowPos sourcePos,
                                           RowPos targetPos,
                                           RowPos initialPos,
                                           int8_t* data,
                                           size_t dataSize,
                                           void* buffer )
{
    auto* header = ( AriesXLogHeader* )( buffer );
    header->magic = ARIES_XLOG_HEADER_MAGIC;
    header->operation = operation;
    header->tableId = tableId;
    header->sourcePos = sourcePos;
    header->targetPos = targetPos;
    header->initialPos = initialPos;
    header->txid = txid;
    header->dataLength = static_cast< int32_t >( dataSize );

    auto* data_buffer = ( int8_t* )buffer + sizeof( AriesXLogHeader );
    memcpy( data_buffer, data, dataSize );
    return true;
}

AriesXLogPageSPtr AriesXLogWriter::createNewPage( size_t size )
{
    auto actual_count = ( size + DEFAULT_PAGE_SIZE - 1 ) / DEFAULT_PAGE_SIZE;
    auto page = std::make_shared< AriesXLogPage >( actual_count * DEFAULT_PAGE_SIZE );
    pages.emplace_back( page );
    return page;
}

void AriesXLogWriter::SetWriteHandler( const AriesXLogWriteHandler& handler, const AriesXLogWriteBufferHandler& writeBufferHandler )
{
    writeHandler = handler;
    this->writeBufferHandler = writeBufferHandler;
}

bool SpecialWriteHandler( const std::vector< AriesXLogPageSPtr >& pages, bool needFlush )
{
    return AriesXLogManager::GetInstance().WritePages( pages, needFlush, true );
}

bool SpecialWriteBufferHandler( const std::vector< int8_t* >& buffers, const std::vector< size_t >& buffersSize, bool needFlush )
{
    return AriesXLogManager::GetInstance().WriteBuffer( buffers, buffersSize, needFlush, true );
}

AriesSpecialXLogWriter::AriesSpecialXLogWriter( TxId txid ) : AriesXLogWriter( txid )
{
    SetWriteHandler( SpecialWriteHandler, SpecialWriteBufferHandler );
}

END_ARIES_ENGINE_NAMESPACE
