#include <boost/filesystem.hpp>
#include "AriesXLogRecoveryer.h"

#include "schema/SchemaManager.h"
#include "AriesXLogManager.h"
#include "AriesMvccTableManager.h"
#include "AriesInitialTableManager.h"
#include "server/Configuration.h"
#include "utils/utils.h"
#include "CpuTimer.h"

BEGIN_ARIES_ENGINE_NAMESPACE

void AriesXLogRecoveryer::SetReader( const AriesXLogReaderSPtr& reader )
{
    this->reader = reader;
}

bool AriesXLogRecoveryer::ContinueWithLastRecover()
{
    string xlogRecoverDoneFilePath( Configuartion::GetInstance().GetXLogRecoverDoneFilePath() );
    return boost::filesystem::exists( xlogRecoverDoneFilePath );
}

void InitXLogDir( const std::string& dir )
{
    boost::filesystem::path p1( dir );
    if (  boost::filesystem::exists( p1 ) )
    {
        if ( boost::filesystem::is_directory( p1 ) )
        {
            boost::filesystem::directory_iterator end_it;
            boost::filesystem::directory_iterator it( p1 );
            while ( it != end_it )
            {
                boost::filesystem::remove_all( it->path() );
                ++it;
            }
        }
        else
        {
            string errMsg( "already exsits" );
            ARIES_EXCEPTION( EE_CANT_MKDIR, dir.data(), 0, errMsg.data() );
        }
    }
    else if ( !boost::filesystem::create_directories( p1 ) )
    {
        char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
        ARIES_EXCEPTION( EE_CANT_MKDIR, dir.data(), errno,
                         strerror_r( errno, errbuf, sizeof(errbuf) ) );
    }
}
void AriesXLogRecoveryer::InitXLogRecover()
{
    string dir = Configuartion::GetInstance().GetDataXLogRecoverDirectory();
    InitXLogDir( dir );

    dir = Configuartion::GetInstance().GetDictXLogRecoverDirectory();
    InitXLogDir( dir );
}

void AriesXLogRecoveryer::MarkXLogRecoverDone()
{
    LOG( INFO ) << "Mark xlog recover done";

    string xlogRecoverDoneFilePath( Configuartion::GetInstance().GetXLogRecoverDoneFilePath() );
    int fd = open( xlogRecoverDoneFilePath.data(),
                   O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
    if ( -1 == fd )
    {
        char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
        ARIES_EXCEPTION( EE_WRITE, xlogRecoverDoneFilePath.data(), errno,
                         strerror_r( errno, errbuf, sizeof(errbuf) ) );
    }
    close( fd );

    LOG( INFO ) << "Mark xlog recover done OK";
}

void MoveColumnDataXLogRecoverResults()
{
    const auto& databases = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabases();
    for ( auto db : databases )
    {
        auto& dbName = db.first;
        auto& tables = db.second->GetTables();
        for ( auto table : tables )
        {
            auto& tableName = table.first;
            string xlogRecoverDir = Configuartion::GetInstance().GetDataXLogRecoverDirectory( dbName, tableName );
            string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbName, tableName );
            vector< string > files = listFiles( xlogRecoverDir );
            for ( auto& file : files )
            {
                string srcPath( xlogRecoverDir + "/" + file );
                string dstPath( tableDataDir + "/" + file );
                LOG(INFO) << "Moving data file from " << srcPath << " to " << dstPath;
                if ( 0 != rename( srcPath.data(), dstPath.data() ) )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_WRITE, dstPath.data(), errno,
                                     strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }
            }
        }
    }
}

void MoveDictXLogRecoverResults()
{
    string xlogRecoverDir = Configuartion::GetInstance().GetDictXLogRecoverDirectory();
    string dictDataDir = Configuartion::GetInstance().GetDictDataDirectory();
    vector< string > files = listFiles( xlogRecoverDir );
    for ( auto& file : files )
    {
        string srcPath( xlogRecoverDir + "/" + file );
        string dstPath( dictDataDir + "/" + file );
        LOG(INFO) << "Moving data file from " << srcPath << " to " << dstPath;
        if ( 0 != rename( srcPath.data(), dstPath.data() ) )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, dstPath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
    }
}

void AriesXLogRecoveryer::MoveXLogRecoverResults()
{
    MoveColumnDataXLogRecoverResults();
    
    MoveDictXLogRecoverResults();
}

void AriesXLogRecoveryer::PostXLogRecoverDone()
{
    AriesXLogManager::GetInstance().SaveCheckPoint( is_special );
    
    // move data files
    MoveXLogRecoverResults();
    // delete ok mark
    string xlogRecoverDoneFilePath( Configuartion::GetInstance().GetXLogRecoverDoneFilePath() );
    unlink( xlogRecoverDoneFilePath.data() );
}

AriesXLogRecoveryer::AriesXLogRecoveryer( const bool isSpecial ) : is_special( isSpecial )
{
}

bool AriesXLogRecoveryer::Recovery()
{
    aries::CPU_Timer t;
    t.begin();

    if ( ContinueWithLastRecover() )
    {
        LOG( INFO ) << "Continue with last xlog recovery";
        PostXLogRecoverDone();
        LOG( INFO ) << "xlog recover time: " << t.end();
        return true;
    }

    LOG( INFO ) << "xlog recovery";

    InitXLogRecover();

    auto record = reader->Next();

    bool have_data_to_recovery = false;

    while ( record.first != NULL )
    {
        bool reach_checkpoint = false;
        switch ( record.first->operation )
        {
            case OperationType::Commit:
                status_of_transations[ record.first->txid ] = true;
                break;
            case OperationType::Abort:
                status_of_transations[ record.first->txid ] = false;
                break;
            case OperationType::Delete:
                handleDelete( record.first );
                break;
            case OperationType::Insert:
                handleInsert( record.first, record.second );
                break;
            case OperationType::Update:
                handleUpdate( record.first, record.second );
                break;
            case OperationType::CheckPoint:
                LOG( INFO ) << "here got checkpoint, break";
                reach_checkpoint = true;
                break;
            case OperationType::Truncate:
                LOG( INFO ) << "got truncate here, table id: " + std::to_string( record.first->tableId );
                status_of_tables[ record.first->tableId ] = XLogTableStatus::Truancted;
                break;
            case OperationType::InsertDict:
                handleInsertDict( record.first, record.second );
                break;
            case OperationType::InsertBatch:
                // printf("here have batch insert\n");
                handleInsertBatch( record.first, record.second );
                break;
            default: break;
        }

        if ( reach_checkpoint )
        {
            break;
        }

        have_data_to_recovery = true;
        record = reader->Next();
    }

    if ( flushAll() )
    {
        if ( have_data_to_recovery )
        {
            MarkXLogRecoverDone();
            PostXLogRecoverDone();
        }

        LOG( INFO ) << "xlog recover time: " << t.end();
        return true;
    }
    return false;
}

bool AriesXLogRecoveryer::flushAll()
{
    for ( auto& item : deleted_rows )
    {
        if ( item.second.empty() )
        {
            continue;
        }

        auto table = getInitialTable( item.first );
        if ( !table )
        {
            LOG( INFO ) << "cannot recover delete record, maybe table be deleted";
            item.second.clear();
            continue;
        }

        if ( !table->XLogRecoverDeleteRows( item.second ) )
        {
            return false;
        }

        item.second.clear();
    }

    for ( auto& it : inserted_data_cache )
    {
        auto table = getInitialTable( it.first );
        if ( !table )
        {
            LOG( INFO ) << "cannot recover insert record, maybe table be deleted";
            it.second->Reset();
            continue;
        }
        if ( !writeInsertedRows( table, it.second ) )
        {
            return false;
        }

        it.second->Reset();
    }

    for ( auto& it : inserted_dict_cache )
    {
        auto dict = AriesDictManager::GetInstance().GetDict( it.first );
        if ( !dict )
        {
            LOG( INFO ) << "cannot recovery insert dict record, maybe dict was deleted";
            it.second->Reset();
            continue;
        }
        if ( !writeInsertedDictRows( dict, it.second ) )
        {
            return false;
        }
        it.second->Reset();
    }

    for ( auto& it : updated_data )
    {
        if ( it.second.empty() )
        {
            continue;
        }
        if ( !writeUpdatedRows( it.first ) )
        {
            return false;
        }

        it.second.clear();
    }

    for ( const auto& it : initial_tables )
    {
        it.second->Sweep();
        it.second->XLogRecoverDone();

        /**
         * 这里导致后续重新创建 mvcctable
         */
        AriesMvccTableManager::GetInstance().removeMvccTable( it.second->GetDbName(), it.second->GetTableName() );
        AriesMvccTableManager::GetInstance().deleteCache( it.second->GetDbName(), it.second->GetTableName() );
        AriesInitialTableManager::GetInstance().removeTable( it.second->GetDbName(), it.second->GetTableName() );
    }

    AriesDictManager::GetInstance().XLogRecoverDone();

    return true;
}

bool AriesXLogRecoveryer::handleInsert( AriesXLogHeader* header, int8_t* data )
{
    if ( !isValid( header ) )
    {
        return true;
    }

    if ( !isRowValid( header->tableId, header->targetPos ) )
    {
        return true;
    }

    auto cache = getInsertRecordCache( header->tableId );
    if ( !cache )
    {
        LOG( INFO ) << "cannot get cache for insert record, maybe table was deleted";
        return true;
    }

    if ( cache->NeedToFlush() )
    {
        auto table = getInitialTable( header->tableId );
        if ( !table )
        {
            LOG( INFO ) << "cannot recover insert record, maybe table be deleted";
        }
        else
        {
            if ( !writeInsertedRows( table, cache ) )
            {
                LOG( ERROR ) << "writeInsertedRows return false";
                return false;
            }
        }

        cache->Reset();
    }

    auto buffer = cache->Alloc();
    memcpy( buffer, data, header->dataLength );
    return true;
}

bool AriesXLogRecoveryer::handleInsertDict( AriesXLogHeader* header, int8_t* data )
{
    // table xlog is truncated
    if ( getTableStatus( header->tableId ) != XLogTableStatus::OK )
    {
        return true;
    }

    auto dict = AriesDictManager::GetInstance().GetDict( header->tableId );
    if ( !dict )
    {
        LOG( INFO ) << "cannot recovery insert dict record, maybe dict was deleted";
        return true;
    }

    auto cache = getInsertDictRecordCache( dict );
    if ( cache->NeedToFlush() )
    {
        if ( !writeInsertedDictRows( dict, cache ) )
        {
            LOG( ERROR ) << "writeInsertedDictRows return false";
            return false;
        }

        cache->Reset();
    }

    auto buffer = cache->Alloc();
    memcpy( buffer, data, header->dataLength );
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
bool AriesXLogRecoveryer::handleInsertBatch( AriesXLogHeader* header, int8_t* data )
{
    if ( !isValid( header ) )
    {
        return true;
    }

    auto table_id = header->tableId;

    AriesBatchInsertInfo* info = ( AriesBatchInsertInfo* )data;

    auto row_count = info->rowCount;
    auto column_count = info->columnCount;

    size_t columnSizesOffset = sizeof( AriesBatchInsertInfo );
    size_t* columnsSize = ( size_t* )( data + columnSizesOffset );

    size_t rowPosesOffset = columnSizesOffset + column_count * sizeof( size_t );
    RowPos* rowposes = ( RowPos* )(  data + rowPosesOffset );

    for ( size_t i = 0; i < row_count; i++ )
    {
        if ( !isRowValid( header->tableId, rowposes[ i ] ) )
        {
            rowposes[ i ] = 0;
        }
    }

    size_t columnsDataOffset = rowPosesOffset + sizeof( RowPos ) * row_count;
    auto table = getInitialTable( table_id );
    if ( !table )
    {
        LOG( INFO ) << "cannot recover insert record, maybe table be deleted";
    }
    else
    {
        table->XLogRecoverInsertBatch( data + columnsDataOffset,
                                       column_count,
                                       columnsSize,
                                       row_count,
                                       rowposes );
    }
    return true;
}

bool AriesXLogRecoveryer::handleDelete( AriesXLogHeader* header )
{
    if ( !isValid( header ) )
    {
        return true;
    }

    updateRowStatus( header->tableId, header->sourcePos, INVALID_ROWPOS, true );

    // 删除的是 initial table 中的数据
    if ( header->sourcePos < 0 )
    {
        deleted_rows[ header->tableId ].emplace_back( - header->sourcePos - 1 );
    }
    else
    {
        // do nothing
    }

    if ( deleted_rows[ header->tableId ].size() > 1000 )
    {
        auto table = getInitialTable( header->tableId );
        if ( !table )
        {
            LOG( INFO ) << "cannot recover delete record, maybe table be deleted";
            deleted_rows[ header->tableId ].clear();
            return true;
        }

        if ( !table->XLogRecoverDeleteRows( deleted_rows[ header->tableId ] ) )
        {
            deleted_rows[ header->tableId ].clear();
            return false;
        }
        deleted_rows[ header->tableId ].clear();
    }

    return true;
}

bool AriesXLogRecoveryer::handleUpdate( AriesXLogHeader* header, int8_t* data )
{
    if ( !isValid( header ) )
    {
        return true;
    }

    bool isDelete;
    if ( !isRowValid( header->tableId, header->targetPos, isDelete ) )
    {
        updateRowStatus( header->tableId, header->sourcePos, header->targetPos, isDelete );
        return true;
    }

    /**
     * 更新的是 delta table 中的数据
     * 这里其实应该是 insert
     */
    if ( header->initialPos == INVALID_ROWPOS )
    {
        auto cache = getInsertRecordCache( header->tableId );
        if ( !cache )
        {
            LOG( INFO ) << "cannot get cache for insert record, maybe table be deleted";
        }
        else
        {
            if ( cache->NeedToFlush() )
            {
                auto table = getInitialTable( header->tableId );
                if ( !table )
                {
                    LOG( INFO ) << "cannot recover update record, maybe table be deleted";
                }
                else
                {
                    if ( !writeInsertedRows( table, cache ) )
                    {
                        LOG( ERROR ) << "writeInsertedRows return false";
                        return false;
                    }
                }

                cache->Reset();
            }

            auto buffer = cache->Alloc();
            memcpy( buffer, data, header->dataLength );
        }
    }
    else
    {
        auto cache = getUpdateRecordCache( header->tableId );
        if ( !cache )
        {
            LOG( INFO ) << "cannot get cache for update record, maybe table be deleted";
        }
        else
        {
            if ( cache->NeedToFlush() )
            {
                if ( !writeUpdatedRows( header->tableId ) )
                {
                    LOG( ERROR ) << "writeUpdatedRows return false";
                    return false;
                }

                updated_data[ header->tableId ].clear();
                cache->Reset();
            }

            auto buffer = cache->Alloc();
            memcpy( buffer, data, header->dataLength );

            index_t index = - header->initialPos - 1;

            auto data = std::make_shared< UpdateRowData >();
            data->m_rowIdx = index;
            data->m_colDataBuffs = buffer;
        
            updated_data[ header->tableId ].emplace_back( data );
        }
    }

    updateRowStatus( header->tableId, header->sourcePos, header->targetPos );

    return true;
}

bool AriesXLogRecoveryer::writeUpdatedRows( TableId tableId )
{
    const auto& datas = updated_data[ tableId ];
    if ( datas.empty() )
    {
        return true;
    }

    auto table = getInitialTable( tableId );
    if ( !table )
    {
        LOG( INFO ) << "cannot recover update record, maybe table be deleted";
        return true;
    }
    return table->UpdateFileRows( updated_data[ tableId ] ); 
}

bool AriesXLogRecoveryer::writeInsertedRows( AriesInitialTableSPtr table, const XLogRecordCacheSPtr& cache )
{
    if ( cache->usedCount == 0 )
    {
        return true;
    }

    std::vector< int8_t* > rows( cache->usedCount );
    for ( int i = 0; i < cache->usedCount; i ++ )
    {
        rows[ i ] = cache->Get( i );
    }

    return table->XLogRecoverInsertRows( rows ).size() == rows.size();
}

bool AriesXLogRecoveryer::writeInsertedDictRows( AriesDictSPtr& dict, const XLogRecordCacheSPtr &cache )
{
    if ( cache->usedCount == 0 )
    {
        return true;
    }

    std::vector< int8_t* > rows( cache->usedCount );
    for ( int i = 0; i < cache->usedCount; i ++ )
    {
        rows[ i ] = cache->Get( i );
    }

    AriesDictManager::GetInstance().XLogRecoverInsertDict( dict, rows );
    return true;
}

bool AriesXLogRecoveryer::isValid( AriesXLogHeader* header )
{
    if ( getTableStatus( header->tableId ) != XLogTableStatus::OK )
    {
        return false;
    }

    if ( header->operation == OperationType::Commit )
    {
        return true;
    }

    auto it = status_of_transations.find( header->txid );
    if ( it == status_of_transations.cend() )
    {
        LOG( ERROR ) << "ignore data for unknown status transaction";
        return false;
    }

    return it->second;
}

AriesInitialTableSPtr AriesXLogRecoveryer::getInitialTable( const TableId& tableId )
{
    auto it = initial_tables.find( tableId );
    if ( it == initial_tables.cend() )
    {
        auto pair = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseAndTableById( tableId );
        if ( pair.first && pair.second )
        {
            auto table = AriesMvccTableManager::GetInstance().getMvccTable( pair.first->GetName(), pair.second->GetName() );
            initial_tables[ tableId ] = table->GetInitialTable();
            return table->GetInitialTable();
        }
        else
        {
            return nullptr;
        }
    }

    return it->second;
}

XLogRecordCacheSPtr AriesXLogRecoveryer::getInsertRecordCache( const TableId& tableId )
{
    auto it = inserted_data_cache.find( tableId );
    if ( it == inserted_data_cache.cend() )
    {
        auto result = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseAndTableById( tableId );
        if ( !result.second )
        {
            return nullptr;
        }

        auto parser = std::make_shared< TupleParser >( result.second );
        auto row_size = parser->GetTupleSize();
        auto cache = std::make_shared< XLogRecordCache >( tableId, row_size );
        inserted_data_cache[ tableId ] = cache;
        return cache;
    }

    return it->second;
}

XLogRecordCacheSPtr AriesXLogRecoveryer::getInsertDictRecordCache( AriesDictSPtr& dict )
{
    auto dictId = dict->GetId();
    auto it = inserted_dict_cache.find( dictId );
    if ( it == inserted_dict_cache.cend() )
    {
        auto row_size = DICT_DELTA_ROW_DATA_PREFIX_SIZE + dict->getDictItemStoreSize();
        auto cache = std::make_shared< XLogRecordCache >( dictId, row_size );
        inserted_dict_cache[ dictId ] = cache;
        return cache;
    }

    return it->second;
}

XLogRecordCacheSPtr AriesXLogRecoveryer::getUpdateRecordCache( const TableId& tableId )
{
    auto it = updated_data_cache.find( tableId );
    if ( it == updated_data_cache.cend() )
    {
        auto result = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseAndTableById( tableId );
        if ( !result.second )
        {
            return nullptr;
        }

        auto parser = std::make_shared< TupleParser >( result.second );
        auto row_size = parser->GetTupleSize();
        auto cache = std::make_shared< XLogRecordCache >( tableId, row_size );
        updated_data_cache[ tableId ] = cache;
        return cache;
    }

    return it->second;
}

/**
 * 检查 rowpos 对应的记录是否有效
 * 如果 rowpos 对应的记录在后面被修改则无效，反之有效
 */
bool AriesXLogRecoveryer::isRowValid( TableId tableId, RowPos rowPos, bool& isDelete )
{
    auto& status = status_of_rowpos[ tableId ];
    auto it = status.find( rowPos );
    if ( it != status.cend() )
    {
        isDelete = it->second;
        return false;
    }

    return true;
}

bool AriesXLogRecoveryer::isRowValid( TableId tableId, RowPos rowPos )
{
    bool isDelete;
    return isRowValid( tableId, rowPos, isDelete );
}

/**
 * 更新行状态
 * 因为 source 对应的记录被 target 修改，source 的状态应该是无效的
 * 同时我们不必再记录 target 的状态
 */
void AriesXLogRecoveryer::updateRowStatus( TableId tableId, RowPos source, RowPos target, bool isDelete )
{
    auto& status = status_of_rowpos[ tableId ];
    status.erase( target );
    status[ source ] = isDelete;
}

XLogTableStatus AriesXLogRecoveryer::getTableStatus( const TableId& tableId )
{
    if ( status_of_tables.find( tableId ) == status_of_tables.cend() )
    {
        status_of_tables[ tableId ] = XLogTableStatus::OK;
    }

    return status_of_tables[ tableId ];
}

END_ARIES_ENGINE_NAMESPACE