#pragma once

#include <unordered_set>

#include "AriesXLog.h"
#include "AriesXLogPage.h"
#include "AriesXLogReader.h"
#include "AriesInitialTable.h"
#include "Compression/dict/AriesDictManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

#define MAX_XLOG_RECORD_CACHE_SIZE ( 16 * 1024 * 1024 ) // 16 MB

class AriesXLogReader;

struct XLogRecordCache
{
    TableId tableId;
    size_t rowSize;
    int32_t totalCount;
    int32_t usedCount;
    int8_t* data;

    XLogRecordCache( const TableId& tableId, const size_t& rowSize ): tableId( tableId ), rowSize( rowSize ), usedCount( 0 )
    {
        totalCount = ( MAX_XLOG_RECORD_CACHE_SIZE + rowSize - 1 ) / rowSize;
        data = new int8_t[ totalCount * rowSize ];
        memset( data, 0, totalCount * rowSize );
    }

    ~XLogRecordCache()
    {
        delete[] data;
    }

    bool NeedToFlush()
    {
        return usedCount == totalCount;
    }

    int8_t* Alloc()
    {
        int8_t* ptr = data + usedCount * rowSize;
        usedCount ++;
        return ptr;
    }

    int8_t* Get( int32_t index )
    {
        return data + index * rowSize;
    }

    void Reset()
    {
        usedCount = 0;
        memset( data, 0, totalCount * rowSize );
    }
};

using XLogRecordCacheSPtr = std::shared_ptr< XLogRecordCache >;

enum class XLogTableStatus : int16_t
{
    OK,
    Truancted,
    Dropped
};

class AriesXLogRecoveryer
{
private:
    AriesXLogReaderSPtr reader;
    bool is_special;
    /**
     * true if committed
     * false if aborted
     */
    std::map< TxId, bool > status_of_transations;

    /**
     * 每个 table 对应一个 std::map< RowPos, bool >
     * 如果 RowPos 在 map 中不存在，说明这条记录之后没有被修改过，那么这条记录应该是有效的
     * 如果 RowPos 在 map 中存在，其 bool 值表示是否为删除操作（ true：delete，false：update）
     */
    std::map< TableId, std::map< RowPos, bool > > status_of_rowpos;

    /**
     * Table 的状态
     */
    std::map< TableId, XLogTableStatus > status_of_tables;

    std::map< TableId, AriesInitialTableSPtr > initial_tables;
    std::map< TableId, XLogRecordCacheSPtr > inserted_data_cache;
    std::map< TableId, XLogRecordCacheSPtr > inserted_dict_cache;
    std::map< TableId, std::vector< index_t > > deleted_rows;
    std::map< TableId, XLogRecordCacheSPtr > updated_data_cache;
    std::map< TableId, std::vector< UpdateRowDataPtr > > updated_data;

public:
    AriesXLogRecoveryer( const bool isSpecial = false );
    bool Recovery();
    void SetReader( const AriesXLogReaderSPtr& reader );

    void MarkXLogRecoverDone();
    void PostXLogRecoverDone();

private:
    bool ContinueWithLastRecover();
    void InitXLogRecover();
    void MoveXLogRecoverResults();
    bool handleInsert( AriesXLogHeader* header, int8_t* data );
    bool handleInsertDict( AriesXLogHeader* header, int8_t* data );
    bool handleDelete( AriesXLogHeader* header );
    bool handleUpdate( AriesXLogHeader* header, int8_t* data );
    bool handleInsertBatch( AriesXLogHeader* header, int8_t* data );

    bool writeInsertedRows( AriesInitialTableSPtr table, const XLogRecordCacheSPtr& cache );
    bool writeInsertedDictRows( AriesDictSPtr& dict, const XLogRecordCacheSPtr &cache );
    bool writeUpdatedRows( TableId tableId );

    bool isValid( AriesXLogHeader* header );
    bool isRowValid( TableId tableId, RowPos rowPos );
    bool isRowValid( TableId tableId, RowPos rowPos, bool& isDelete );
    void updateRowStatus( TableId tableId, RowPos source, RowPos target, bool isDelete = false );

    bool flushAll();

    AriesInitialTableSPtr getInitialTable( const TableId& tableId );
    XLogRecordCacheSPtr getInsertRecordCache( const TableId& tableId );
    XLogRecordCacheSPtr getInsertDictRecordCache( AriesDictSPtr& dict );
    XLogRecordCacheSPtr getUpdateRecordCache( const TableId& tableId );

    XLogTableStatus getTableStatus( const TableId& tableId );
};

using AriesXLogRecoveryerSPtr = std::shared_ptr< AriesXLogRecoveryer >;

END_ARIES_ENGINE_NAMESPACE
