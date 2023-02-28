#include <glog/logging.h>
#include <boost/filesystem.hpp>

#include "server/Configuration.h"
#include "utils/utils.h"
#include "AriesEngine/transaction/AriesInitialTable.h"
#include "AriesEngine/transaction/AriesTransaction.h"
#include "AriesEngine/transaction/AriesXLogWriter.h"
#include "AriesDictManager.h"
#include <frontend/SQLExecutor.h>
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngineWrapper/AbstractMemTable.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "CpuTimer.h"

using namespace std;
using namespace aries;
using namespace aries_engine;

namespace aries {

namespace schema
{
    extern const string INFORMATION_SCHEMA;
}

AriesDictManager::AriesDictManager()
: m_nextTableId( 0 )
{

}
bool AriesDictManager::Init()
{
    m_dictNameMap.clear();
    m_dictIdMap.clear();

    string sql = "SELECT ID, NAME, INDEX_TYPE, REF_COUNT, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM DICTS";
    SQLResultPtr sqlResultPtr = aries::SQLExecutor::GetInstance()->ExecuteSQL(sql, schema::INFORMATION_SCHEMA);
    if ( !sqlResultPtr->IsSuccess() )
    {
        LOG( ERROR ) << "Init dict manager failed, error code: " << sqlResultPtr->GetErrorCode()
                     << ", message: " << sqlResultPtr->GetErrorMessage();
        return false;
    }

    const vector<aries::AbstractMemTablePointer>& results = sqlResultPtr->GetResults();
    auto amtp = results[0];
    auto table = ( ( AriesMemTable * )amtp.get() )->GetContent();
    int tupleNum;
    int columnCount = table->GetColumnCount();

    tupleNum = table->GetRowCount();
    std::vector<AriesDataBufferSPtr> columns;
    for (int colId = 1; colId < columnCount + 1; colId++)
    {
        columns.push_back( table->GetColumnBuffer( colId ) );
    }

    int64_t maxId = -1;
    for ( int tid = 0; tid < tupleNum; tid++ )
    {
        auto id = columns[ 0 ]->GetInt64( tid );
        auto name = columns[ 1 ]->GetString( tid );
        auto indexType = columns[ 2 ]->GetString( tid );
        auto refCount = columns[ 3 ]->GetInt64( tid );
        auto nullable = columns[ 4 ]->GetString( tid );
        auto charMaxLen = columns[ 5 ]->GetInt32( tid );

        schema::ColumnType indexDataType = aries::schema::ToColumnType( indexType );
        bool bNullable = ( nullable == "YES" );

        auto dictPtr = new AriesDict( id, name, indexDataType,
                                      refCount, bNullable, charMaxLen );
        AriesDictSPtr dictSPtr;
        dictSPtr.reset( dictPtr );

        InitDictDeltaTable( dictSPtr );

        m_dictNameMap[ name ] = dictSPtr;
        m_dictIdMap[ id ] = dictSPtr;

        if ( id > maxId )
            maxId = id;
    }

    if ( maxId >= 0 )
        m_nextTableId = maxId + 1;
    return true;
}

void AriesDictManager::InitDictDeltaTable( AriesDictSPtr dict )
{
    auto it = m_dictDeltaTables.find( dict->GetId() );
    if ( m_dictDeltaTables.end() == it )
    {
        std::vector< AriesColumnType > types;
        // dictIndex + string
        auto slotSize = DICT_DELTA_ROW_DATA_PREFIX_SIZE + dict->getDictItemStoreSize();
        types.push_back( AriesColumnType( AriesDataType{ AriesValueType::CHAR, int32_t( slotSize ) }, false, false ) );
        int32_t capacity = min( dict->getDictCapacity(), size_t( UINT16_MAX ) );

        auto deltaTableSPtr = make_shared< AriesDeltaTable >( capacity, types );
        m_dictDeltaTables[ dict->GetId() ] = deltaTableSPtr;
    }
}

AriesDictSPtr AriesDictManager::GetDict( int64_t id )
{
    std::lock_guard< std::mutex > lock( m_dictLock );

    auto it = m_dictIdMap.find( id );
    if ( m_dictIdMap.end() != it )
        return it->second;
    else
        return nullptr;
}

AriesDictSPtr AriesDictManager::GetDict( const string& name )
{
    std::lock_guard< std::mutex > lock( m_dictLock );

    auto it = m_dictNameMap.find( name );
    if ( m_dictNameMap.end() != it )
        return it->second;
    else
        return nullptr;
}

int64_t AriesDictManager::GetDictId()
{
    return m_nextTableId.fetch_add( 1 );
}

static void DoUpdateDictRefCount( const AriesDictSPtr& dict )
{
    std::string sql = "UPDATE DICTS SET REF_COUNT = " + std::to_string( dict->GetRefCount() ) +
                      " WHERE NAME = '" + dict->GetName() + "'";
    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, schema::INFORMATION_SCHEMA );
    if ( !sqlResult->IsSuccess() )
    {
        ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), sqlResult->GetErrorMessage().data() );
    }
}

void InitDictDataFile( AriesDictSPtr& dict )
{
    string dictFilePath = AriesDictManager::GetInstance().GetDictFilePath( dict->GetId(), dict->GetName() );
    int dictFd = open( dictFilePath.data(),
                       O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
    if ( -1 == dictFd )
    {
        char errbuf[ MYSYS_STRERROR_SIZE ];
        ARIES_EXCEPTION( EE_CANTCREATEFILE, dictFilePath.data(), errno,
                         strerror_r( errno, errbuf, sizeof(errbuf) ) );
    }
    auto fileHelper = std::make_shared< fd_helper >( dictFd );
    AriesInitialTable::WriteBlockFileHeader( dictFd,
                                             dictFilePath,
                                             0,
                                             dict->IsNullable(),
                                             dict->getDictItemStoreSize(),
                                             false );
}

AriesDictSPtr AriesDictManager::GetOrCreateDict( const std::string& name,
                                                 schema::ColumnType indexDataType,
                                                 bool nullable,
                                                 int32_t charMaxLen )
{
    assert( charMaxLen > 0 );
    AriesDictSPtr dict;

    std::lock_guard< std::mutex > lock( m_dictLock );

    auto it = m_dictNameMap.find( name );
    if ( m_dictNameMap.end() != it )
    {
        dict = it->second;
        if ( dict->GetSchemaSize() != ( size_t )charMaxLen ||
             dict->IsNullable() != nullable ||
             dict->GetIndexDataType() != indexDataType )
        {
             string errMsg = format_mysql_err_msg( ER_CANNOT_ADD_DICT ) +
                             ", dict encode definitions differ";
            ARIES_EXCEPTION_SIMPLE( ER_CANNOT_ADD_DICT, errMsg );
        }
    }
    else
    {
        auto dictId = GetDictId();
        auto dictPtr = new AriesDict( dictId, name, indexDataType, 1, nullable, charMaxLen ); 

        dict.reset( dictPtr );
    }

    return dict;
}

void AriesDictManager::AddDict( AriesDictSPtr dict )
{
    std::lock_guard< std::mutex > lock( m_dictLock );

    auto it = m_dictNameMap.find( dict->GetName() );
    if ( m_dictNameMap.end() != it )
    {
        dict = it->second;
        dict->IncRefCount();
        DoUpdateDictRefCount( dict );
    }
    else
    {
        m_dictNameMap[ dict->GetName() ] = dict;
        m_dictIdMap[ dict->GetId() ] = dict;

        std::string nullableStr = dict->IsNullable() ? "YES" : "NO";
        std::string sql = "INSERT INTO DICTS VALUES ( " +
                          std::to_string( dict->GetId() ) + ", '" + dict->GetName() + "', '" + DataTypeString( dict->GetIndexDataType() ) + "', " +
                          "1, '" + nullableStr + "', " +
                          std::to_string( dict->GetSchemaSize() ) + ", " +
                          std::to_string( dict->GetSchemaSize() ) + ", " +
                          "'" + std::string( schema::DEFAULT_CHARSET_NAME ) + "', " +
                          "'" + std::string( schema::DEFAULT_UTF8_COLLATION ) + "' )";

        auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, schema::INFORMATION_SCHEMA );
        if ( !sqlResult->IsSuccess() )
        {
            ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), sqlResult->GetErrorMessage().data() );
        }

        InitDictDataFile( dict );

        InitDictDeltaTable( dict );
    }
}

void AriesDictManager::DecDictRefCount( int64_t id )
{
    std::lock_guard< std::mutex > lock( m_dictLock );

    auto it = m_dictIdMap.find( id );
    if ( m_dictIdMap.end() != it )
    {
        auto refCount = it->second->DecRefCount();
        if ( 0 == refCount )
            DeleteDict( it->second );
        else
            DoUpdateDictRefCount( it->second );
    }
}

void AriesDictManager::DeleteDict( AriesDictSPtr& dict )
{
    string sql = "DELETE FROM DICTS WHERE ID = " + std::to_string( dict->GetId() );
    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, schema::INFORMATION_SCHEMA );
    if ( !sqlResult->IsSuccess() )
    {
        ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), sqlResult->GetErrorMessage().data() );
    }
    AriesXLogManager::GetInstance().AddTruncateEvent( dict->GetId() );

    string dictFilePath = AriesDictManager::GetInstance().GetDictFilePath( dict->GetId(), dict->GetName() );
    unlink( dictFilePath.data() );

    m_dictNameMap.erase( dict->GetName() );
    m_dictIdMap.erase( dict->GetId() );
    m_dictDeltaTables.erase( dict->GetId() );
}

std::string AriesDictManager::GetDictFilePath( int64_t id, const std::string& name ) const
{
    std::string dictFilePath = Configuartion::GetInstance().GetDictDataDirectory();
    dictFilePath.append( "/" ).append( GetDictFileName( id, name ) );
    return dictFilePath;
}

std::string AriesDictManager::GetDictFileName( int64_t id, const std::string& name ) const
{
    // id_name
    std::string fileName( std::to_string( id ) );
    return fileName.append( "_" ).append( name );
}

AriesDictSPtr AriesDictManager::ReadDictData( int64_t id )
{
    std::lock_guard< std::mutex > lock( m_dictLock );

    auto it = m_dictIdMap.find( id );
    ARIES_ASSERT( m_dictIdMap.end() != it, "Cannot find dict " + std::to_string( id ) );

    auto dict = it->second;
    dict->ReadData();
    return dict;
}

AriesDictSPtr AriesDictManager::ReadDictData( AriesDictSPtr& dict )
{
    std::lock_guard< std::mutex > lock( m_dictLock );

    dict->ReadData();
    return dict;
}

bool AriesDictManager::AddDictTuple( AriesTransactionPtr transaction,
                                     AriesDictSPtr dict,
                                     aries_acc::AriesManagedIndicesArraySPtr newDictIndices )
{
    auto dictId = dict->GetId();
    auto txId = transaction->GetTxId();
    auto dictDeltaTable = m_dictDeltaTables[ dictId ];
    auto dictItemSize = dict->getDictItemStoreSize();
    auto indiceCount = newDictIndices->GetItemCount();
    bool unused;
    for( size_t i = 0; i < indiceCount; ++i )
    {
        vector< RowPos > poses = dictDeltaTable->ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
        if( poses.empty() )
        {
            return false;
        }
        RowPos newPos = poses[0];
        pTupleHeader newHeader = dictDeltaTable->GetTupleHeader( newPos, AriesDeltaTableSlotType::AddedTuples );
        newHeader->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );

        auto slotBuffer = dictDeltaTable->GetTupleFieldBuffer( newPos );
        // dictIndex + string
        auto pNewDictIndex = newDictIndices->GetData( i );
        memcpy( slotBuffer, pNewDictIndex, sizeof(int32_t) );

        auto pSlotBuffer = slotBuffer + DICT_DELTA_ROW_DATA_PREFIX_SIZE;
        memcpy( pSlotBuffer, dict->getDictItem( *pNewDictIndex ), dictItemSize );

        dictDeltaTable->CompleteSlot(
        { newPos }, AriesDeltaTableSlotType::AddedTuples );

        bool success = transaction->GetXLogWriter()->WriteDictLog( OperationType::InsertDict, dictId,
            INVALID_ROWPOS, newPos,
            INVALID_ROWPOS, slotBuffer, dictItemSize + DICT_DELTA_ROW_DATA_PREFIX_SIZE);
        if( !success )
            return success;
    }

    return true;
}

bool AriesDictManager::AddDictTuple( AriesTransactionPtr transaction,
                                     AriesDictSPtr dict,
                                     const vector< int32_t> &newDictIndices )
{
    auto dictId = dict->GetId();
    auto txId = transaction->GetTxId();
    auto dictDeltaTable = m_dictDeltaTables[ dictId ];
    auto dictItemSize = dict->getDictItemStoreSize();
    auto indiceCount = newDictIndices.size();
    bool unused;
    for( size_t i = 0; i < indiceCount; ++i )
    {
        vector< RowPos > poses = dictDeltaTable->ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
        if( poses.empty() )
        {
            return false;
        }
        RowPos newPos = poses[0];
        pTupleHeader newHeader = dictDeltaTable->GetTupleHeader( newPos, AriesDeltaTableSlotType::AddedTuples );
        newHeader->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );

        auto slotBuffer = dictDeltaTable->GetTupleFieldBuffer( newPos );
        // dictIndex + string
        auto newDictIndex = newDictIndices[ i ];
        memcpy( slotBuffer, &newDictIndex, sizeof(int32_t) );

        auto pSlotBuffer = slotBuffer + DICT_DELTA_ROW_DATA_PREFIX_SIZE;
        memcpy( pSlotBuffer, dict->getDictItem( newDictIndex ), dictItemSize );

        dictDeltaTable->CompleteSlot(
        { newPos }, AriesDeltaTableSlotType::AddedTuples );

        bool success = transaction->GetXLogWriter()->WriteDictLog( OperationType::InsertDict, dictId,
            INVALID_ROWPOS, newPos,
            INVALID_ROWPOS, slotBuffer, dictItemSize + DICT_DELTA_ROW_DATA_PREFIX_SIZE);
        if( !success )
            return success;
    }

    return true;
}

int8_t* AriesDictManager::GetDictBufferForXLogRecover( AriesDictSPtr& dict, bool createIfNotExists )
{
    int8_t* dataAddr = nullptr;
    auto it = m_xlogRecoverDictBuffMap.find( dict->GetId() );
    if ( m_xlogRecoverDictBuffMap.end() != it )
    {
        dataAddr = it->second.get();
    }
    else if ( createIfNotExists )
    {
        auto dictId = dict->GetId();
        string dictFilePath = AriesDictManager::GetInstance().GetDictFilePath( dictId, dict->GetName() );
        int fd = open( dictFilePath.data(), O_RDWR );
        if ( -1 == fd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, dictFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
        fd_helper_ptr fdPtr = make_shared< fd_helper >( fd );

        auto itemStoreSize = dict->getDictItemStoreSize();

        size_t buffSize = itemStoreSize * dict->getDictCapacity();
        dataAddr = new int8_t[ buffSize ];
        shared_ptr< int8_t[] > addrSPtr;
        addrSPtr.reset( dataAddr );

        auto columnType = AriesColumnType{ { AriesValueType::CHAR, ( int )dict->GetSchemaSize() }, dict->IsNullable(), false };
        BlockFileHeader headerInfo;
        IfstreamSPtr ifBlockFile = make_shared< ifstream >( dictFilePath );
        ValidateBlockFile( ifBlockFile, dictFilePath, headerInfo, columnType );

        size_t sizeToRead = headerInfo.itemLen * headerInfo.rows;
        lseek( fd, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, SEEK_SET );
        auto readSize = read( fd, dataAddr, sizeToRead );
        if ( ( -1 == readSize ) || ( size_t )readSize != sizeToRead )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_READ, dictFilePath.data(),
                             errno, strerror_r( errno, errbuf, sizeof(errbuf)) );

        }
        m_xlogRecoverDictBuffMap[ dictId ] = addrSPtr;
        m_xlogRecoverDictRowCount[ dictId ] = headerInfo.rows;
    }

    return dataAddr;
}

void AriesDictManager::XLogRecoverInsertDict( AriesDictSPtr& dict, const vector< int8_t* >& rowsData )
{
    auto itemStoreSize = dict->getDictItemStoreSize();
    int8_t* dictData = GetDictBufferForXLogRecover( dict, true );

    for ( auto rowData : rowsData )
    {
        int32_t dictIndex = *( ( int32_t* )rowData );
        size_t offset = dictIndex * itemStoreSize;
        memcpy( dictData + offset, ( const uchar* )( rowData + sizeof( int32_t ) ), itemStoreSize );
    }
    m_xlogRecoverDictRowCount[ dict->GetId() ] += rowsData.size();
}

void AriesDictManager::XLogRecoverDone()
{
    FlushXLogRecoverResult();
}

void AriesDictManager::FlushXLogRecoverResult()
{
    aries::CPU_Timer t;
    t.begin();

    LOG( INFO ) << "Flush xlog recover result";

    string dictXlogRecoverDir = Configuartion::GetInstance().GetDictXLogRecoverDirectory();
    if ( !boost::filesystem::exists( dictXlogRecoverDir ) )
    {
        if ( !boost::filesystem::create_directories( dictXlogRecoverDir ) )
        {
            char errbuf[ 1014 ] = {0};
            ARIES_EXCEPTION( EE_CANT_MKDIR, dictXlogRecoverDir.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
    }

    for ( auto it : m_xlogRecoverDictBuffMap )
    {
        auto dictId = it.first;
        auto dictIt = m_dictIdMap.find( dictId );
        auto dict = dictIt->second;
        int8_t* dictBuff = it.second.get();

        string dictFilePath = dictXlogRecoverDir + "/" + AriesDictManager::GetInstance().GetDictFileName( dictId, dict->GetName() );
        int fd = open( dictFilePath.data(),
                       O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
        if ( -1 == fd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, dictFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
        auto fdHelper = make_shared< fd_helper >( fd );

        auto itemStoreSize = dict->getDictItemStoreSize();
        AriesInitialTable::WriteBlockFileHeader( fd, dictFilePath,
                                                 m_xlogRecoverDictRowCount[ dictId ],
                                                 dict->IsNullable(),
                                                 itemStoreSize,
                                                 false );
        size_t writtenSize = my_write( fd, ( uchar* )dictBuff,
                                       itemStoreSize * m_xlogRecoverDictRowCount[ dictId ],
                                       MYF( MY_FNABP ) );
        fdHelper.reset();
        if ( 0 != writtenSize )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, dictFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
    }

    m_xlogRecoverDictBuffMap.clear();
    m_xlogRecoverDictRowCount.clear();
}
} // namespace aries
