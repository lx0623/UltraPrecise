/*
 * AriesInitialTable.cpp
 *
 *  Created on: Mar 16, 2020
 *      Author: lichi
 */

#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <future>
#include <boost/filesystem.hpp>
#include <set>
#include "server/mysql/include/mysys_err.h"
#include "server/mysql/include/my_sys.h"
#include "AriesInitialTable.h"
#include "schema/SchemaManager.h"
#include "Compression/dict/AriesDictManager.h"
#include "AriesEngine/AriesUtil.h"
#include "frontend/ColumnStructure.h"
#include "frontend/SQLExecutor.h"
#include "utils/thread.h"
#include "CpuTimer.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "server/Configuration.h"
#include "datatypes/AriesDatetimeTrans.h"
// #define MULTI_THREAD_READ

bool IsCurrentThdKilled();
void SendKillMessage();

BEGIN_ARIES_ENGINE_NAMESPACE

    int8_t* NewBlockBitmap()
    {
        int8_t* bitmapPtr = new int8_t[ ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE ];
        memset( bitmapPtr, 0, ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE );
        return bitmapPtr;
    }

    inline int64_t ColumnBuffKey( int32_t colIndex, int32_t blockIndex )
    {
        return ( ( ( int64_t )colIndex ) << 32 ) | blockIndex;
    }

    int GetBlockFileHeaderInfo( IfstreamSPtr blockFile, BlockFileHeader& headerInfo )
    {
        int ret = 0;
        blockFile->seekg( 0, ios::end );
        int64_t dataLen = blockFile->tellg();
        if (dataLen < ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE) {
            ret = -2;
        } else {
            blockFile->seekg( 0, ios::beg );
            blockFile->read( (char *) &headerInfo, sizeof(BlockFileHeader) );
        }
        return ret;
    }
    /*
     * @input parameter:
     *  filePath: full path of column file
     * @output parameters:
     *  rows: total rows in file
     *  containNull: rows contain Null or not
     *  itemLen: actual item length saved in column file
     *
     * @return:
     *   1 : success, but file format is older version, no output parameter saved in file, output is not changed
     *   0 : success, caller can use output parameters;
     *   -1: file does not exist
     *   -2: header size in file is not correct
     *   -3: header version in file is not correct
     * */
    int GetBlockFileHeaderInfo( string filePath, BlockFileHeader& headerInfo )
    {
        int ret = -1;
        IfstreamSPtr blockFile = make_shared< ifstream >( filePath );
        if( blockFile->is_open() )
        {
            ret = GetBlockFileHeaderInfo( blockFile, headerInfo );
        }

        if ( 0 != ret )
        {
            LOG(ERROR) << "block file not opened: " << filePath << ", error code: " << ret;
        }

        return ret;
    }

    void ValidateBlockFile(
        IfstreamSPtr blockFile,
        const string& filePath,
        BlockFileHeader& headerInfo,
        AriesColumnType& dataType )
    {
        size_t dataTypeSize = dataType.GetDataTypeSize();
        blockFile->seekg( 0, ios::end );
        uint64_t dataLen = ( uint64_t )blockFile->tellg() - ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE;

        // check if header info is valid
        int res = GetBlockFileHeaderInfo( blockFile, headerInfo );
        if (res == 0)
        {
            if ( headerInfo.rows > 0 && headerInfo.itemLen > 0 )
            {
                if ( dataType.HasNull != (bool) headerInfo.containNull ) {
                    ARIES_EXCEPTION( ER_FILE_CORRUPT, filePath.data() );
                } else if ( headerInfo.itemLen != dataTypeSize ) {
                    switch( dataType.DataType.ValueType )
                    {
                        case AriesValueType::CHAR:
                            //set actual item length
                            dataType.DataType.Length = ( int )headerInfo.itemLen - ( int ) headerInfo.containNull;
                            //update dataType size
                            dataTypeSize = dataType.GetDataTypeSize();
                            break;
                        default:
                            ARIES_EXCEPTION( ER_FILE_CORRUPT, filePath.data() );
                            break;
                    }
                }
                if ( dataLen / dataTypeSize != headerInfo.rows ) {
                    ARIES_EXCEPTION( ER_FILE_CORRUPT, filePath.data() );
                }
            }
        }
        else
        {
            string msg = format_mysql_err_msg( ER_FILE_CORRUPT, filePath.data() );
            msg.append(" error code: " ).append( std::to_string( res ) );
            ARIES_EXCEPTION_SIMPLE( ER_FILE_CORRUPT, msg.data() );
        }

        //简单验证文件的有效性
        if( dataLen % dataTypeSize != 0 )
        {
            ARIES_EXCEPTION( ER_FILE_CORRUPT, filePath.data() );
        }
    }

    AriesInitialTable::AriesInitialTable( const string& dbName,
                                          const string& tableName ) :
        m_dbName( dbName),
        m_tableName( tableName ),
        m_pLocks( nullptr )
    {
        m_dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( m_dbName );
        m_tableEntry = m_dbEntry->GetTableByName( m_tableName );
        m_columnCount = m_tableEntry->GetColumnsCount();
        m_dataDir = Configuartion::GetInstance().GetDataDirectory( m_dbName, m_tableName );
    #ifdef ARIES_INIT_TABLE_CACHE
        m_table = make_unique< AriesTableBlock >();
    #endif

        GetMetaInfo();

        if ( m_tableEntry->IsPartitioned() )
        {
            LoadPartitionMetaInfo();
        }

        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            auto colEntry = m_tableEntry->GetColumnById( colIndex + 1 );
            if ( EncodeType::DICT == colEntry->encode_type )
                m_itemStoreSizes.push_back( colEntry->GetDictIndexItemSize() );
            else
                m_itemStoreSizes.push_back( colEntry->GetItemStoreSize() );
        }
        m_pageSize = sysconf(_SC_PAGE_SIZE);
    }

    AriesInitialTable::AriesInitialTable( const string& dbName,
                                          const string& tableName,
                                          const string& tableDataDir ) :
        m_dbName( dbName),
        m_tableName( tableName ),
        m_dataDir( tableDataDir ),
        m_pLocks( nullptr )
    {
        m_dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( m_dbName );
        m_tableEntry = m_dbEntry->GetTableByName( m_tableName );
        m_columnCount = m_tableEntry->GetColumnsCount();
    #ifdef ARIES_INIT_TABLE_CACHE
        m_table = make_unique< AriesTableBlock >();
    #endif

        GetMetaInfo();

        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            auto colEntry = m_tableEntry->GetColumnById( colIndex + 1 );
            if ( EncodeType::DICT == colEntry->encode_type )
                m_itemStoreSizes.push_back( colEntry->GetDictIndexItemSize() );
            else
                m_itemStoreSizes.push_back( colEntry->GetItemStoreSize() );
        }
        m_pageSize = sysconf(_SC_PAGE_SIZE);
    }

    AriesInitialTable::~AriesInitialTable()
    {
        Clear();
    }
    void AriesInitialTable::Clear()
    {
        lock_guard< mutex > colLock( m_columnInfoLock );
        m_rowCountOfBlocks.clear();
        m_allColumnInfos.clear();
        m_initialTupleHeader.clear();
        m_lockFlags = nullptr;
        m_pLocks = nullptr;
        m_itemStoreSizes.clear();
        m_blockChanged.clear();
        m_columnBlockDataBuffMap.clear();
        for ( auto bitmap : m_slotBitmaps )
        {
            delete[] bitmap;
        }
        m_slotBitmaps.clear();
    }

    size_t AriesInitialTable::GetBlockRowCount( uint32_t blockIndex )
    {
        assert( blockIndex >= 0 && blockIndex < m_initTableMetaInfo.m_blockCount );
        int colIndex = 0;
        auto colId = colIndex + 1;
        auto columnEntry = m_tableEntry->GetColumnById( colId );
        AriesColumnType dataType;

        if ( aries::EncodeType::DICT == columnEntry->encode_type )
        {
            dataType = columnEntry->GetDictIndexColumnType();
        }
        else
        {
            auto valueType = columnEntry->GetType();
            auto length = columnEntry->GetLength();
            auto nullable = columnEntry->IsAllowNull();
            dataType =
                CovertToAriesColumnType( valueType,
                                         length,
                                         nullable,
                                         true,
                                         columnEntry->numeric_precision,
                                         columnEntry->numeric_scale );
        }

        string blockFilePath = GetBlockFilePath( 0, blockIndex );
        IfstreamSPtr blockFile = make_shared< ifstream >( blockFilePath );
        if ( blockFile->is_open() )
        {
            BlockFileHeader headerInfo;
            ValidateBlockFile( blockFile, blockFilePath, headerInfo, dataType );
            return headerInfo.rows;
        }
        else
        {
            string msg = format_mysql_err_msg( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT,msg.data() );
        }
    }

    int AriesInitialTable::GetTotalRowCount( const string& metaInfoFile,
                                             uint64_t& totalRowCount )
    {
        int ret = 0;
        if ( !boost::filesystem::is_regular_file( metaInfoFile ) )
        {
            LOG( INFO ) << "table meta file not exists: " << metaInfoFile;
            ret = -1;
            totalRowCount = 0;
            return ret;
        }

        IfstreamSPtr ifs = make_shared< ifstream >( metaInfoFile );
        if( ifs->is_open() )
        {
            ifs->seekg( 0, ios::end );
            auto fileLen = ifs->tellg();
            if ( fileLen < ( int )( sizeof( InitTableMetaInfo ) ) )
            {
                string msg = "table meta file " + metaInfoFile +
                             " len: " + std::to_string( fileLen );
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }
            else
            {
                ifs->seekg( 0, ios::beg );
                InitTableMetaInfo initTableMetaInfo;
                ifs->read( ( char* ) &initTableMetaInfo, sizeof( InitTableMetaInfo ) );
                totalRowCount = initTableMetaInfo.m_totalRowCount;
            }
            ifs->close();
        }
        else
        {
            string msg = "open table meta file " + metaInfoFile + " failed";
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
        }
        return ret;

    }

    std::vector< PartitionMetaInfo >& AriesInitialTable::GetPartitionMetaInfo()
    {
        return m_partitionMetaInfoArray;
    }

    void AriesInitialTable::LoadPartitionMetaInfo()
    {
        m_partitionMetaInfoArray.clear();
        m_partitionMetaInfoArray.resize( m_tableEntry->GetPartitionCount() );
        for ( size_t i = 0; i < m_tableEntry->GetPartitionCount(); i++ )
        {
            const auto& partitionItem = m_tableEntry->GetPartitions()[ i ];
            assert( i + 1 == size_t( partitionItem->m_partOrdPos ) );
            std::string metaInfoPath = GetPartitionMetaFilePath( i );
            if ( !boost::filesystem::is_regular_file( metaInfoPath ) )
            {
                string msg( "table meta file not found: " + metaInfoPath );
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }
            
            int fd = open( metaInfoPath.data(), O_RDWR );
            if ( -1 == fd )
            {
                string msg( "table meta file cannot be opened: " + metaInfoPath );
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }

            struct stat sb;
            if ( fstat( fd, &sb ) == -1 )
            {
                string msg = "read table meta file " + metaInfoPath + " failed";
                close( fd );
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }

            if ( ( size_t )sb.st_size < sizeof( PartitionMetaInfoHeader ) )
            {
                close( fd );
                string msg = "table partition meta file " + metaInfoPath +
                             " len: " + std::to_string( sb.st_size );
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }

            PartitionMetaInfoHeader metaInfoHeader;
            size_t readCount = read( fd, ( char* ) &metaInfoHeader, sizeof( PartitionMetaInfoHeader ) );
            if ( sizeof( PartitionMetaInfoHeader ) != readCount )
            {
                close( fd );
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION( EE_READ, metaInfoPath.data(),
                                    errno, strerror_r( errno, errbuf, sizeof(errbuf)) );
            }

            PartitionMetaInfo& metaInfo = m_partitionMetaInfoArray[ i ];
            metaInfo.Version = metaInfoHeader.Version;
            metaInfo.RowCount = metaInfoHeader.RowCount;
            metaInfo.BlockCount = metaInfoHeader.BlockCount;
            metaInfo.BlocksID.resize( metaInfo.BlockCount );

            if ( metaInfo.BlockCount > 0 )
            {
                readCount = read( fd, ( char* )( metaInfo.BlocksID.data() ), sizeof( int32_t ) * metaInfo.BlockCount );
                if ( readCount != sizeof( int32_t ) * metaInfo.BlockCount )
                {
                    close( fd );
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_READ, metaInfoPath.data(),
                                    errno, strerror_r( errno, errbuf, sizeof( errbuf ) ) );
                }
            }
            close( fd );

            for ( const auto index : metaInfo.BlocksID )
            {
                m_blockPartitionID[ index ] = partitionItem->m_partOrdPos;
            }
        }
    }

    void AriesInitialTable::GetMetaInfo()
    {
        m_metaInfoFilePath = GetMetaFilePath();
        m_bitmapLength = DIV_UP( m_columnCount, 8 );
        if ( !boost::filesystem::is_regular_file( m_metaInfoFilePath ) )
        {
            string msg( "table meta file not found: " + m_metaInfoFilePath );
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
        }

        int fd = open( m_metaInfoFilePath.data(), O_RDWR );
        if( -1 != fd )
        {
            m_metaInfoFd = make_shared< fd_helper >( fd );
            struct stat sb;
            if ( fstat( fd, &sb ) == -1 )
            {
                string msg = "read table meta file " + m_metaInfoFilePath + " failed";
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }

            if ( ( size_t )sb.st_size < sizeof( InitTableMetaInfo ) )
            {
                string msg = "table meta file " + m_metaInfoFilePath +
                             " len: " + std::to_string( sb.st_size );
                ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
            }
            else
            {
                size_t readCount = read( fd, ( char* ) &m_initTableMetaInfo, sizeof( InitTableMetaInfo ) );
                if ( sizeof( InitTableMetaInfo ) != readCount )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_READ, m_metaInfoFilePath.data(),
                                     errno, strerror_r( errno, errbuf, sizeof(errbuf)) );
                }
                if ( COLUMN_FILE_VERSION != m_initTableMetaInfo.m_version )
                {
                    string msg( "table meta file with wrong version: " +
                                std::to_string( m_initTableMetaInfo.m_version ) );
                    ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
                }

                for ( uint32_t i = 0; i < m_initTableMetaInfo.m_blockCount; ++ i )
                {
                    int8_t* bitmapPtr = new int8_t[ ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE ];
                    readCount = read( fd, ( char* ) bitmapPtr, ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE );
                    if (  ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE != readCount )
                    {
                        char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                        ARIES_EXCEPTION( EE_READ, m_metaInfoFilePath.data(),
                                         errno, strerror_r( errno, errbuf, sizeof(errbuf)) );
                    }
                    m_slotBitmaps.push_back( bitmapPtr );
                    m_blockChanged.push_back( false );
                }
            }

            SetCapacity();

            m_initialTupleHeader.resize( m_initTableMetaInfo.m_totalRowCount );
            m_lockFlags.reset( new int8_t[ m_initTableMetaInfo.m_totalRowCount ] );
            memset( m_lockFlags.get(), 0, m_initTableMetaInfo.m_totalRowCount );
            m_pLocks = reinterpret_cast< atomic< int8_t >* >( m_lockFlags.get() );
        }
        else
        {
            string msg = "open table meta file " + m_metaInfoFilePath + " failed";
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT, msg.data() );
        }
    }

    string AriesInitialTable::GetBlockFilePath( const int colIndex, const uint32_t blockIndex )
    {
        auto columnEntry = m_tableEntry->GetColumnById( colIndex + 1 );
        string blockFileName;
        blockFileName = GetBlockFileName( m_tableName,
                                          colIndex,
                                          blockIndex,
                                          aries::EncodeType::DICT == columnEntry->encode_type );
        return m_dataDir + "/" + blockFileName;
    }

    void AriesInitialTable::ValidateColumnFiles( const size_t colIndex )
    {
        auto colId = colIndex + 1;
        auto columnEntry = m_tableEntry->GetColumnById( colId );
        AriesColumnType dataType;

        if ( aries::EncodeType::DICT == columnEntry->encode_type )
        {
            dataType = columnEntry->GetDictIndexColumnType();
        }
        else
        {
            auto valueType = columnEntry->GetType();
            auto length = columnEntry->GetLength();
            auto nullable = columnEntry->IsAllowNull();
            dataType =
                CovertToAriesColumnType( valueType,
                                         length,
                                         nullable,
                                         true,
                                         columnEntry->numeric_precision,
                                         columnEntry->numeric_scale );
        }

        vector< string >       blockFilePaths;
        vector< IfstreamSPtr > blockFileStreams;

        string name = columnEntry->GetName();

        bool getBlockFileInfo = ( 0 == m_rowCountOfBlocks.size() );

        uint32_t blockIdx = m_readStartBlockIndex;
        uint32_t blockCount = 0;
        string blockFilePath = GetBlockFilePath( colIndex, blockIdx );
        IfstreamSPtr blockFile = make_shared< ifstream >( blockFilePath );
        while ( blockFile->is_open() )
        {
            LOG(INFO) << "id = " << colId << ", name = " << name << ", filePath = " << blockFilePath;
            BlockFileHeader headerInfo;
            ValidateBlockFile( blockFile, blockFilePath, headerInfo, dataType );

            //跳过4K文件头
            blockFile->seekg( ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, ios::beg );
            blockFilePaths.emplace_back( blockFilePath );
            blockFileStreams.emplace_back( blockFile );

            if ( getBlockFileInfo )
            {
                auto rows = headerInfo.rows;
                m_rowCountOfBlocks.emplace_back( rows );
            }

            ++blockCount;
            if ( blockCount == m_initTableMetaInfo.m_blockCount )
                break;

            ++blockIdx;
            blockFilePath = GetBlockFilePath( colIndex, blockIdx );
            blockFile = make_shared< ifstream >( blockFilePath );
        }
        if ( blockFilePaths.size() == m_initTableMetaInfo.m_blockCount )
        {
            auto colFileInfo = make_shared< ColumnFileInfo >( dataType,
                                                              blockFilePaths,
                                                              blockFileStreams );
            m_allColumnInfos[ colId ] = colFileInfo;
        }
        else
        {
            string msg = format_mysql_err_msg( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
            msg.append(" block file count: " ).append( std::to_string( blockFilePaths.size() ) );
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_CORRUPT,msg.data() );
        }

    }
    void AriesInitialTable::Open( const vector< int32_t >& columnIds )
    {
        lock_guard< mutex > colLock( m_columnInfoLock );

        for( int id : columnIds )
        {
            // id is 1 based and so on
            int index = id - 1;
            ARIES_ASSERT( index >= 0 && ( size_t )index <= m_columnCount, "index: " + to_string( index ) + ", count: " + to_string(m_columnCount));

            if ( m_allColumnInfos.end() != m_allColumnInfos.find( id ) ||
                 ( size_t )index == m_columnCount ) // rowId column
            {
                continue;
            }
            ValidateColumnFiles( ( size_t )index );
        }
    }
#ifdef ARIES_INIT_TABLE_CACHE
    int64_t AriesInitialTable::CacheColumnData( const vector< int32_t >& columnIds )
    {
        Open( columnIds );

        for( int id : columnIds )
        {
            if ( IsCurrentThdKilled() )
            {
                LOG(INFO) << "thread killed while scan file";
                SendKillMessage();
            }

            if ( ( size_t )id == m_columnCount + 1 ) // rowId column
            {
                continue;
            }

            std::lock_guard< mutex > lock( *( m_allColumnInfos[ id ]->m_lock ) );
            if ( m_table->ColumnExists( id ) )
                continue;

            try
            {
            // #ifdef MULTI_THREAD_READ
            //     ReadColumnMultiThreads( id - 1, m_blockRowCount );
            // #else
                ReadColumn( id - 1 );
            // #endif
            }
            catch ( AriesException& e )
            {
                throw;
            }
            catch (...)
            {
                //读取文件出错
                ARIES_EXCEPTION( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
            }
        }
        return m_initTableMetaInfo.m_totalRowCount;
    }
#endif

    AriesTableBlockUPtr AriesInitialTable::GetColumnData(
        const vector< int32_t >& columnIds,
        const std::vector< AriesCommonExprUPtr >& partitionConditions )
    {
        Open( columnIds );

        AriesTableBlockUPtr table = make_unique< AriesTableBlock >();
        std::vector< uint32_t > filterBlockIndex;
        if ( !partitionConditions.empty() )
        {
            int blockIdxInTable = 0;
            for ( uint32_t idx = 0; idx < m_initTableMetaInfo.m_blockCount; idx++ )
            {
                auto partitionID = GetPartitionIDForBlock( idx, partitionConditions );
                if ( -1 != partitionID )
                {
                    filterBlockIndex.emplace_back( idx );
                    // TODO aaa: set m_partitionedColumnID
                    table->SetPartitionedColumnID( m_tableEntry->GetPartitionColumnIndex() + 1 );
                    table->SetBlockPartitionID( blockIdxInTable++, partitionID );
                }
            }
        }

        for( int id : columnIds )
        {
            if ( IsCurrentThdKilled() )
            {
                LOG(INFO) << "thread killed while scan file";
                SendKillMessage();
            }

            if ( ( size_t )id == m_columnCount + 1 ) // rowId column
            {
                continue;
            }

            try
            {
            // #ifdef MULTI_THREAD_READ
            //     ReadColumnMultiThreads( id - 1, m_blockRowCount );
            // #else
                ReadColumn( id - 1, table, filterBlockIndex );
            // #endif
            }
            catch ( AriesException& e )
            {
                throw;
            }
            catch (...)
            {
                //读取文件出错
                ARIES_EXCEPTION( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
            }
        }
        return std::move( table );
    }

    AriesDictSPtr AriesInitialTable::GetColumnDict( const string& colName )
    {
        auto colEntry = m_tableEntry->GetColumnByName( colName );
        return GetColumnDict( colEntry->GetColumnIndex() );
    }

    AriesDictSPtr AriesInitialTable::GetColumnDict( int columnIndex )
    {
        auto colEntry = m_tableEntry->GetColumnById( columnIndex + 1 );
        return AriesDictManager::GetInstance().ReadDictData( colEntry->GetDictId() );
    }

#ifdef ARIES_INIT_TABLE_CACHE
    void AriesInitialTable::ReadDictColumn( int32_t colIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        auto columnDict = GetColumnDict( colIndex );
        auto indices = ReadColumnBlockFiles( colIndex );
#ifdef ARIES_PROFILE
        float s = ( t.end() + 0.0 ) / 1000;
        LOG( INFO ) << "Read dict column time " << std::to_string( s ) << "s";
#endif
        auto dictColumn = make_shared< AriesDictEncodedColumn >( columnDict, indices );
        m_table->AddColumn( colIndex + 1, dictColumn );

        return;
    }
#endif

    void AriesInitialTable::ReadDictColumn( int32_t colIndex, AriesTableBlockUPtr& table, const std::vector< uint32_t >& filterBlockIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        auto columnDict = GetColumnDict( colIndex );
        auto indices = ReadColumnBlockFiles( colIndex, filterBlockIndex );
#ifdef ARIES_PROFILE
        float s = ( t.end() + 0.0 ) / 1000;
        LOG( INFO ) << "Read dict column time " << std::to_string( s ) << "s";
#endif
        auto dictColumn = make_shared< AriesDictEncodedColumn >( columnDict, indices );
        table->AddColumn( colIndex + 1, dictColumn );

    }


#ifdef ARIES_INIT_TABLE_CACHE
    void AriesInitialTable::ReadColumn( int32_t colIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        auto colId = colIndex + 1;

        auto columnEntry = m_tableEntry->GetColumnById( colId );
        if ( aries::EncodeType::DICT == columnEntry->encode_type )
        {
            ReadDictColumn( colIndex );
            return;
        }

        int64_t readColumnTime = 0;
#ifdef ARIES_PROFILE
        t.begin();
#endif
        auto column = ReadColumnBlockFiles( colIndex );
        m_table->AddColumn( colId, column );
#ifdef ARIES_PROFILE
        readColumnTime += t.end();
        float s = ( readColumnTime + 0.0 ) / 1000;
        LOG( INFO ) << "Read column time: " << std::to_string( s ) << "s";
#endif
    }
#endif

    void AriesInitialTable::ReadColumn( int32_t colIndex,
                                        AriesTableBlockUPtr& table,
                                        const std::vector< uint32_t >& filterBlockIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        auto colId = colIndex + 1;

        auto columnEntry = m_tableEntry->GetColumnById( colId );
        if ( aries::EncodeType::DICT == columnEntry->encode_type )
        {
            ReadDictColumn( colIndex, table, filterBlockIndex );
            return;
        }

        int64_t readColumnTime = 0;
#ifdef ARIES_PROFILE
        t.begin();
#endif
        auto column = ReadColumnBlockFiles( colIndex, filterBlockIndex );
        table->AddColumn( colId, column );
#ifdef ARIES_PROFILE
        readColumnTime += t.end();
        float s = ( readColumnTime + 0.0 ) / 1000;
        LOG( INFO ) << "Read column time: " << std::to_string( s ) << "s";
#endif
    }

    int32_t AriesInitialTable::GetPartitionIDForBlock( const size_t index, const std::vector< AriesCommonExprUPtr >& partitionConditions )
    {
        auto& partitions = m_tableEntry->GetPartitions();
        auto it = m_blockPartitionID.find( index );
        if ( m_blockPartitionID.end() == it )
        {
            return -1;
        }
        auto partitionID = it->second;
        auto partition = partitions[ partitionID - 1 ];
        for ( const auto& condition : partitionConditions )
        {
            assert( condition->GetType() == AriesExprType::COMPARISON );

            auto comparisonType = static_cast< AriesComparisonOpType >( boost::get< int >( condition->GetContent() ) );
            switch ( comparisonType )
            {
                case AriesComparisonOpType::EQ:
                case AriesComparisonOpType::GE:
                case AriesComparisonOpType::LE:
                case AriesComparisonOpType::GT:
                case AriesComparisonOpType::LT:
                    break;
                default:
                    continue;
            }

            auto& columnExpr = condition->GetChild( 0 );
            auto& valueExpr = condition->GetChild( 1 );
            ARIES_ASSERT( valueExpr->IsLiteralValue(), "value for partition should be literal" );

            auto columnID = boost::get< int32_t >( columnExpr->GetContent() );
            assert( m_tableEntry->GetPartitionColumnIndex() + 1 == columnID );
            auto upperValueString = partition->m_partDesc;
            std::string lowerValueString;
            if ( partitionID > 1 )
            {
                lowerValueString = partitions[ partitionID - 2 ]->m_partDesc;
            }

            switch ( valueExpr->GetType() )
            {
                case AriesExprType::INTEGER:
                {
                    auto exprValue = boost::get< int32_t >( valueExpr->GetContent() );
                    auto maxValue = INT32_MAX;
                    if( upperValueString != "MAXVALUE" )
                        maxValue = std::stoi( upperValueString );
                    auto minValue = INT32_MIN;
                    if ( !lowerValueString.empty() )
                    {
                        minValue = std::stoi( lowerValueString );
                    }

                    switch ( comparisonType )
                    {
                        case AriesComparisonOpType::EQ:
                        {
                            if ( exprValue >= maxValue || exprValue < minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::GE:
                        case AriesComparisonOpType::GT:
                        {
                            if ( exprValue >= maxValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::LE:
                        {
                            if ( exprValue < minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::LT:
                        {
                            if ( exprValue <= minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        default:
                            continue;
                    }
                    break;
                }
                case AriesExprType::STRING:
                case AriesExprType::FLOATING:
                case AriesExprType::DECIMAL:
                {
                    assert( 0 );
                    return -1;
                }
                case AriesExprType::DATE:
                {
                    auto exprValue = boost::get< aries_acc::AriesDate >( valueExpr->GetContent() ).toTimestamp();
                    auto transfer = aries_acc::AriesDatetimeTrans::GetInstance();
                    auto maxValue = INT64_MAX;
                    if( upperValueString != "MAXVALUE" )
                        maxValue = transfer.ToAriesDate( upperValueString ).toTimestamp();
                    auto minValue = INT64_MIN;
                    if ( !lowerValueString.empty() )
                    {
                        minValue = transfer.ToAriesDate( lowerValueString ).toTimestamp();
                    }

                    switch ( comparisonType )
                    {
                        case AriesComparisonOpType::EQ:
                        {
                            if ( exprValue >= maxValue || exprValue < minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::GE:
                        case AriesComparisonOpType::GT:
                        {
                            if ( exprValue >= maxValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::LE:
                        {
                            if ( exprValue < minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::LT:
                        {
                            if ( exprValue <= minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        default:
                            continue;
                    }
                    break;
                }
                case AriesExprType::DATE_TIME:
                {
                    auto exprValue = boost::get< aries_acc::AriesDatetime >( valueExpr->GetContent() ).toTimestamp();
                    auto transfer = aries_acc::AriesDatetimeTrans::GetInstance();
                    auto maxValue = INT64_MAX;
                    if( upperValueString != "MAXVALUE" )
                        maxValue = transfer.ToAriesDatetime( upperValueString ).toTimestamp();
                    auto minValue = INT64_MIN;
                    if ( !lowerValueString.empty() )
                    {
                        minValue = transfer.ToAriesDatetime( lowerValueString ).toTimestamp();
                    }

                    switch ( comparisonType )
                    {
                        case AriesComparisonOpType::EQ:
                        {
                            if ( exprValue >= maxValue || exprValue < minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::GE:
                        case AriesComparisonOpType::GT:
                        {
                            if ( exprValue >= maxValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::LE:
                        {
                            if ( exprValue < minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        case AriesComparisonOpType::LT:
                        {
                            if ( exprValue <= minValue )
                            {
                                return -1;
                            }
                            break;
                        }
                        default:
                            continue;
                    }
                    break;
                }
                case AriesExprType::TIME:
                case AriesExprType::TIMESTAMP:
                case AriesExprType::NULL_VALUE:
                case AriesExprType::TRUE_FALSE:
                case AriesExprType::YEAR:
                {
                    assert( 0 );
                    return false;
                }
                default:
                    return false;
            }

        }
        return partitionID;
    }

    AriesColumnSPtr AriesInitialTable::ReadColumnBlockFiles( int32_t colIndex, const std::vector< uint32_t >& filterBlockIndex )
    {
        ColumnFileInfoPtr colFileInfo = m_allColumnInfos[ colIndex + 1 ];

        const auto& dataType = colFileInfo->m_dataType;
        auto column = make_shared< AriesColumn >();

        if ( 0 == m_initTableMetaInfo.m_totalRowCount )
        {
            column->SetColumnType( colFileInfo->m_dataType );
            return column;
        }

        uint64_t readRowCount = 0;

        for ( size_t i = 0; i < m_initTableMetaInfo.m_blockCount; ++i )
        {
            if ( !filterBlockIndex.empty() )
            {
                if ( std::find( filterBlockIndex.cbegin(), filterBlockIndex.cend(), i ) == filterBlockIndex.cend() )
                {
                    DLOG( INFO ) << "skip block; " + std::to_string( i );
                    continue;
                }
            }

            if ( !colFileInfo->m_blockFileStreams[ i ] )
            {
                colFileInfo->m_blockFileStreams[ i ] = make_shared< ifstream >( colFileInfo->m_blockFilePaths[ i ] );
            }

            size_t offset = ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE;
            auto rowCount = m_rowCountOfBlocks[ i ];
            if ( 0 == i && UINT32_MAX != m_readStartBlockLineIndex )
            {
                offset += dataType.GetDataTypeSize() * m_readStartBlockLineIndex;
                rowCount -= m_readStartBlockLineIndex;
            }

            // 跳过文件头
            colFileInfo->m_blockFileStreams[ i ]->seekg( offset, ios::beg );
            AriesDataBufferSPtr data_buffer = make_shared< AriesDataBuffer >( dataType, rowCount );
            data_buffer->PrefetchToCpu();
            if ( rowCount > 0 )
            {
                char *buf = reinterpret_cast< char * >( data_buffer->GetData() );
                colFileInfo->m_blockFileStreams[ i ]->read( buf, data_buffer->GetTotalBytes() );
                data_buffer->MemAdvise( cudaMemAdviseSetReadMostly, 0 );
            }
            colFileInfo->m_blockFileStreams[ i ] = nullptr;

            column->AddDataBuffer( data_buffer );

            readRowCount += m_rowCountOfBlocks[ i ];
        }

        if( readRowCount > m_initTableMetaInfo.m_totalRowCount )
        {
            ARIES_EXCEPTION( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
        }
        return column;
    }

    /*
    void AriesInitialTable::ReadColumnMultiThreads( int32_t colIndex,
                                                    uint64_t blockSize )
    {
        auto& blockFileInfo = m_allColumnInfos[ colId ];
        if ( !blockFileInfo )
        {
            return;
        }

        string filePath = blockFileInfo->m_rowCountOfBlocks[ 0 ]->m_filePath;
        auto dataType = blockFileInfo->m_dataType;

        vector<size_t> threadsJobCnt;
        vector<size_t> threadsJobStartIdx;
        size_t threadCnt = getConcurrency( m_blockCount, threadsJobCnt, threadsJobStartIdx );
        vector< future< vector< AriesDataBufferSPtr > > > workThreads;
        LOG( INFO ) << "Reading column file " << filePath << ", blockSize " << blockSize
                    << ", threadCnt " << threadCnt;

        int64_t readColumnTime = 0;
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        for( size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx )
        {
            workThreads.push_back(std::async(std::launch::async, [=] {
                return ReadBlocks( filePath,
                                   dataType,
                                   blockSize,
                                   threadsJobStartIdx[ threadIdx ],
                                   threadsJobCnt[ threadIdx ] );
            }));
        }

        for( auto& thrd : workThreads )
            thrd.wait();

        AriesColumnSPtr column = m_allColumns[ colId ];
        for( auto& thrd : workThreads )
        {
            auto dataBuffers = thrd.get();
            for ( auto& buffer : dataBuffers )
            {
                column->AddDataBuffer( buffer );
            }
        }
        workThreads.clear();
#ifdef ARIES_PROFILE
        readColumnTime += t.end();
        float s = ( readColumnTime + 0.0 ) / 1000;
        LOG( INFO ) << "Read column " << filePath << ", time(multi threads): " << column->GetRowCount() << " rows, "
                     << std::to_string( s ) << "s";
#endif
    }

    vector< AriesDataBufferSPtr > AriesInitialTable::ReadBlocks( const string& filePath,
                                                                 AriesColumnType dataType,
                                                                 uint64_t blockSize,
                                                                 size_t startBlockIdx,
                                                                 size_t blockCount )
    {
        vector< AriesDataBufferSPtr > buffers;
        IfstreamSPtr fileStream = make_shared< ifstream >( filePath );
        if( fileStream->is_open() )
        {
            size_t dataTypeSize = dataType.GetDataTypeSize();
            size_t startOffset = ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE +
                                 startBlockIdx * blockSize * dataTypeSize;
            fileStream->seekg( startOffset, ios::beg );

            uint64_t maxReadRowCnt;
            uint64_t readRowIdx;
            for ( int i = 0; i < blockCount; ++i )
            {
                readRowIdx = ( startBlockIdx + i ) * blockSize;
                maxReadRowCnt = std::min( blockSize, m_totalRowCount - readRowIdx );
                AriesDataBufferSPtr dataBuffer = make_shared< AriesDataBuffer >( dataType );
                char *buf = reinterpret_cast< char * >( dataBuffer->AllocArray( maxReadRowCnt ) );
                dataBuffer->PrefetchToCpu();
                fileStream->read( buf, dataBuffer->GetTotalBytes() );
                buffers.push_back( dataBuffer );
            }
            fileStream->close();
        }
        return buffers;
    }
    */

#ifdef ARIES_INIT_TABLE_CACHE
    AriesTableBlockUPtr AriesInitialTable::GetTable( const vector< int >& columnIds )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesTableBlockUPtr table = make_unique< AriesTableBlock >();
        CacheColumnData( columnIds );

        int outputColumnId = 0;
        for ( int colId : columnIds )
        {
            if ( ( size_t )colId == m_columnCount + 1 ) // rowId column
            {
                continue;
            }
            auto colEncodeType = m_table->GetColumnEncodeType( colId );
            switch ( colEncodeType )
            {
                case EncodeType::NONE:
                {
                    auto column = std::dynamic_pointer_cast< AriesColumn >( m_table->GetMaterilizedColumn( colId )->Clone() );
                    table->AddColumn( ++outputColumnId, column );
                    break;
                }

                case EncodeType::DICT:
                {
                    auto column = std::dynamic_pointer_cast< AriesDictEncodedColumn >( m_table->GetDictEncodedColumn( colId )->Clone() );
                    table->AddColumn( ++outputColumnId, column );
                    break;
                }
            }
        }
        if ( columnIds.empty() )
            table->SetRowCount( m_initTableMetaInfo.m_totalRowCount );
#ifdef ARIES_PROFILE
        LOG( INFO ) << "AriesInitialTable::GetTable time: " << t.end();
#endif
        return std::move( table );
    }

#else

    AriesTableBlockUPtr AriesInitialTable::GetTable( const vector< int >& columnIds, const std::vector< AriesCommonExprUPtr >& partitionCondition )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesTableBlockUPtr resultTable = make_unique< AriesTableBlock >();

        AriesTableBlockUPtr table = GetColumnData( columnIds, partitionCondition );

        int outputColumnId = 0;
        for ( int colId : columnIds )
        {
            if ( ( size_t )colId == m_columnCount + 1 ) // rowId column
            {
                continue;
            }
            auto colEncodeType = table->GetColumnEncodeType( colId );
            switch ( colEncodeType )
            {
                case EncodeType::NONE:
                {
                    auto column = std::dynamic_pointer_cast< AriesColumn >( table->GetMaterilizedColumn( colId ) );
                    resultTable->AddColumn( ++outputColumnId, column );
                    break;
                }

                case EncodeType::DICT:
                {
                    auto column = std::dynamic_pointer_cast< AriesDictEncodedColumn >( table->GetDictEncodedColumn( colId ) );
                    resultTable->AddColumn( ++outputColumnId, column );
                    break;
                }
            }
        }
        if ( columnIds.empty() )
            resultTable->SetRowCount( m_initTableMetaInfo.m_totalRowCount );
#ifdef ARIES_PROFILE
        LOG( INFO ) << "AriesInitialTable::GetTable time: " << t.end();
#endif
        resultTable->SetPartitionInfo( table->GetPartitionInfo() );
        resultTable->SetPartitionedColumnID( table->GetPartitionedColumnID() );
        return std::move( resultTable );
    }
#endif // ARIES_INIT_TABLE_CACHE

    AriesTableBlockUPtr AriesInitialTable::GetTable( const vector< int >& columnIds,
                                                     const int startBlockIndex,
                                                     const uint32_t startBlockLineIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        m_readStartBlockIndex  = startBlockIndex;
        m_readStartBlockLineIndex  = startBlockLineIndex;

        AriesTableBlockUPtr resultTable = make_unique< AriesTableBlock >();
        AriesTableBlockUPtr table = GetColumnData( columnIds );

        int outputColumnId = 0;
        for ( int colId : columnIds )
        {
            if ( ( size_t )colId == m_columnCount + 1 ) // rowId column
            {
                continue;
            }
            auto colEncodeType = table->GetColumnEncodeType( colId );
            switch ( colEncodeType )
            {
                case EncodeType::NONE:
                {
                    auto column = std::dynamic_pointer_cast< AriesColumn >( table->GetMaterilizedColumn( colId ) );
                    resultTable->AddColumn( ++outputColumnId, column );
                    break;
                }

                case EncodeType::DICT:
                {
                    auto column = std::dynamic_pointer_cast< AriesDictEncodedColumn >( table->GetDictEncodedColumn( colId ) );
                    resultTable->AddColumn( ++outputColumnId, column );
                    break;
                }
            }
        }
        if ( columnIds.empty() )
            resultTable->SetRowCount( m_initTableMetaInfo.m_totalRowCount );
#ifdef ARIES_PROFILE
        LOG( INFO ) << "AriesInitialTable::GetTable time: " << t.end();
#endif
        return std::move( resultTable );
    }

#ifdef ARIES_INIT_TABLE_CACHE
    int8_t* AriesInitialTable::GetTupleFieldContent( int32_t columnId, RowPos rowPos )
    {
        int64_t index = INITIAL_TABLE_ROWPOS_ADJUST( rowPos );
        CheckRowIndex( index );
        ARIES_ASSERT( columnId > 0 && ( size_t )columnId <= m_columnCount,
                      "column id: " +  to_string( columnId ) + ", column count: " + to_string( m_columnCount ) );
        auto colEncodeType = m_table->GetColumnEncodeType( columnId );
        switch ( colEncodeType )
        {
            case EncodeType::NONE:
            {
                auto col = m_table->GetMaterilizedColumn( columnId );
                return col->GetFieldContent( index );
            }

            case EncodeType::DICT:
            {
                auto col = m_table->GetDictEncodedColumn( columnId );
                return col->GetFieldContent( index );
            }

            default:
            {
                string msg( "unexpected encode type: " );
                msg.append( std::to_string( ( int )colEncodeType ) );
                ARIES_ASSERT( 0, msg );
            }
        }
    }

#else

    int8_t* AriesInitialTable::GetTupleFieldContent( int32_t columnId, RowPos rowPos )
    {
        int64_t index = INITIAL_TABLE_ROWPOS_ADJUST( rowPos );
        CheckRowIndex( index );
        ARIES_ASSERT( columnId > 0 && ( size_t )columnId <= m_columnCount,
                      "column id: " +  to_string( columnId ) + ", column count: " + to_string( m_columnCount ) );
        AriesTableBlockUPtr table = GetColumnData( { columnId } );
        auto colEncodeType = table->GetColumnEncodeType( columnId );
        switch ( colEncodeType )
        {
            case EncodeType::NONE:
            {
                auto col = table->GetMaterilizedColumn( columnId );
                return col->GetFieldContent( index );
            }

            case EncodeType::DICT:
            {
                auto col = table->GetDictEncodedColumn( columnId );
                return col->GetFieldContent( index );
            }

            default:
            {
                string msg( "unexpected encode type: " );
                msg.append( std::to_string( ( int )colEncodeType ) );
                ARIES_ASSERT( 0, msg );
            }
        }
    }
#endif

    void AriesInitialTable::WriteColumnBlockFileHeader( ColumnEntryPtr colEntry,
                                                        int fd,
                                                        const string& filePath,
                                                        const uint32_t rowCnt,
                                                        bool useMmap )
    {
        size_t itemStoreSize;
        int8_t nullable = colEntry->IsAllowNull();
        if ( EncodeType::DICT == colEntry->encode_type )
            itemStoreSize = colEntry->GetDictIndexItemSize();
        else
            itemStoreSize = colEntry->GetItemStoreSize();
        WriteBlockFileHeader( fd, filePath, rowCnt, nullable, itemStoreSize, useMmap );
    }

    void AriesInitialTable::WriteBlockFileHeader( int fd,
                                                  const string& filePath,
                                                  const uint32_t rowCnt,
                                                  int8_t nullable,
                                                  size_t itemStoreSize,
                                                  bool useMmap )
    {
        BlockFileHeader headerInfo;
        memset( &headerInfo, 0, sizeof( headerInfo ) );
        headerInfo.rows = rowCnt;
        headerInfo.containNull = nullable;
        headerInfo.itemLen = itemStoreSize;

        if ( useMmap )
        {
            void* mapAddr = mmap( NULL,
                                  ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE,
                                  PROT_READ | PROT_WRITE, MAP_SHARED,
                                  fd,
                                  0 );
            if ( MAP_FAILED == mapAddr )
            {
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION( EE_WRITE, filePath.data(), errno,
                                 strerror_r( errno, errbuf, sizeof(errbuf) ) );
            }
            memcpy( ( uchar* ) mapAddr, &headerInfo, sizeof( BlockFileHeader ) );

            msync( mapAddr, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, MS_ASYNC );

            int unmapRet = munmap( mapAddr, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE );
            if ( 0 != unmapRet )
            {
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION(  EE_WRITE, filePath.data(), errno,
                                  strerror_r( errno, errbuf, sizeof(errbuf) ) );
            }
        }
        else
        {
            uchar header[ ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE ] = { 0 };
            memcpy( header, &headerInfo, sizeof( BlockFileHeader ) );
            if ( ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE != write( fd, header, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE ) )
            {
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION(  EE_WRITE, filePath.data(), errno,
                                  strerror_r( errno, errbuf, sizeof(errbuf) ) );
            }
        }
    }

    // bool AriesInitialTable::DeleteRow( const RowPos rowPos )
    // {
    //     index_t rowIdx = INITIAL_TABLE_ROWPOS_ADJUST( rowPos );
    //     ARIES_ASSERT( rowIdx >= 0 && rowIdx < m_totalRowCount,
    //                   "invalid row pos: " + std::to_string( rowPos ) +
    //                   ", table " + m_dbName + "." + m_tableName +
    //                   ", total row count: " + std::to_string( m_totalRowCount ) );
    //     CLEAR_BIT_FLAG( m_initTableMetaInfo.m_slotBitmap, rowIdx );
    //     return true;
    // }

    // bool AriesInitialTable::UpdateRow( const RowPos rowPos,
    //                                    const map< int32_t, string >& updateData )
    // {
    //     index_t rowIdx = INITIAL_TABLE_ROWPOS_ADJUST( rowPos );
    //     ARIES_ASSERT( rowIdx >= 0 && rowIdx < m_totalRowCount,
    //                   "invalid row pos: " + std::to_string( rowPos ) +
    //                   ", table " + m_dbName + "." + m_tableName +
    //                   ", total row count: " + std::to_string( m_totalRowCount ) );

    //     for ( auto it : updateData )
    //     {
    //         auto colId = it.first;
    //         ARIES_ASSERT( colId > 0 && colId <= m_tableEntry->GetColumnsCount(),
    //                       "invalid column id: " + std::to_string( colId ) +
    //                       ", table " + m_dbName + "." + m_tableName );
    //         m_allColumns[ colId ]->UpdateFieldContent( rowIdx, ( int8_t* )it.second.data() );
    //     }
    //     return true;
    // }

    pair< string, fd_helper_ptr >
    AriesInitialTable::NewBlockFile( uint32_t colIndex, uint32_t blockIdx, uint32_t reserveRowCount )
    {
        auto colEntry = m_tableEntry->GetColumnById( colIndex + 1 );
        string blockFilePath = GetBlockFilePath( colIndex, blockIdx );
        int fd = open( blockFilePath.data(),
                       O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
        if ( -1 == fd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_CANTCREATEFILE, blockFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
        WriteColumnBlockFileHeader( colEntry,
                                    fd,
                                    blockFilePath,
                                    reserveRowCount,
                                    false );
        pair< string, fd_helper_ptr > blockFile( blockFilePath, make_shared< fd_helper >( fd ) );
        return blockFile;
    }

    vector< pair< string, fd_helper_ptr > >
    AriesInitialTable::NewBlockFiles( uint32_t reserveRowCount, uint32_t blockIndex )
    {
        vector< pair< string, fd_helper_ptr > > blockFiles;
        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            auto colEntry = m_tableEntry->GetColumnById( colIndex + 1 );
            blockFiles.emplace_back( NewBlockFile( colIndex, blockIndex, reserveRowCount ) );
        }
        m_slotBitmaps.push_back( NewBlockBitmap() );

        ++m_initTableMetaInfo.m_blockCount;
        SetCapacity();

        return blockFiles;
    }

    vector< BlockDataBuffSPtr > AriesInitialTable::XLogRecoverNewBlockBuffs( uint32_t reserveRowCount, uint32_t blockIndex )
    {
        vector< BlockDataBuffSPtr > dataBlockBuffs;
        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            int64_t mapKey = ColumnBuffKey( colIndex, blockIndex );
            size_t buffSize;
            // empty block
            if ( 0 == reserveRowCount )
                buffSize = 1;
            else
                buffSize = m_itemStoreSizes[ colIndex ] * m_blockDeltaRowCount;
            auto buffPtr = make_shared< BlockDataBuff >();
            buffPtr->reserve( buffSize );
            m_columnBlockDataBuffMap[ mapKey ] = buffPtr;

            dataBlockBuffs.push_back( buffPtr );
        }
        m_slotBitmaps.push_back( NewBlockBitmap() );
        m_blockChanged.push_back( true );

        ++m_initTableMetaInfo.m_blockCount;
        SetCapacity();

        return dataBlockBuffs;
    }

    #ifdef BUILD_TEST
    bool AriesInitialTable::IsSlotFree( const index_t slotIdx )
    {
        CheckRowIndex( slotIdx );
        bool occupied;
        int32_t blockIdx = BLOCK_IDX( slotIdx );
        index_t blockSlotIdx = BLOCK_SLOT_IDX( slotIdx );

        GET_BIT_FLAG( m_slotBitmaps[ blockIdx ], blockSlotIdx, occupied );
        return !occupied;
    }
    #endif

    InitTableSlotPos AriesInitialTable::FindSlotHole( index_t startIdx )
    {
        InitTableSlotPos pos;
        bool occupied;
        for ( uint64_t i = startIdx; i < m_initTableMetaInfo.m_totalRowCount; ++i )
        {
            int32_t blockIdx = BLOCK_IDX( i );
            index_t blockSlotIdx = BLOCK_SLOT_IDX( i );

            GET_BIT_FLAG( m_slotBitmaps[ blockIdx ], blockSlotIdx, occupied );
            if ( !occupied )
            {
                pos.m_slotIdx = i;
                pos.m_blockIdx = blockIdx;
                pos.m_blockSlotIdx = blockSlotIdx;
                return pos;
            }
        }
        pos.m_slotIdx = NULL_INDEX;
        return pos;
    }

    vector< InitTableSlotPos > AriesInitialTable::FindAllSlotHoles()
    {
        vector< InitTableSlotPos > result;

        // bool occupied;
        // for ( index_t i = 0; i < m_initTableMetaInfo.m_totalRowCount; ++i )
        // {
        //     int32_t blockIdx = BLOCK_IDX( i );
        //     index_t blockSlotIdx = BLOCK_SLOT_IDX( i );

        //     GET_BIT_FLAG( m_slotBitmaps[ blockIdx ], blockSlotIdx, occupied );
        //     if ( !occupied )
        //     {
        //         result.push_back( InitTableSlotPos{ i, blockIdx, blockSlotIdx } );
        //     }
        // }

        vector< future< vector< InitTableSlotPos > > > allThreads;
        int blockCount = m_slotBitmaps.size();
        for( int i = 0; i < blockCount - 1; ++i )
        {
            allThreads.push_back( std::async( std::launch::async, [&]( int blockIndex )
            {
                vector< InitTableSlotPos > holes;
                int8_t* pFlags = m_slotBitmaps[ blockIndex ];
                index_t slotIndexOffset = blockIndex * ARIES_BLOCK_FILE_ROW_COUNT;
                bool occupied;
                for( size_t j = 0; j < ARIES_BLOCK_FILE_ROW_COUNT; ++j )
                {
                    GET_BIT_FLAG( pFlags, j, occupied );
                    size_t slotIndex = ( size_t )slotIndexOffset + j;
                    assert( slotIndex <= ( size_t )INT32_MAX );
                    if ( !occupied )
                        holes.push_back( InitTableSlotPos{ ( index_t )slotIndex, blockIndex, ( index_t )j } );
                }
                return holes;
            }, i ) );
        }

        // handle last block
        vector< InitTableSlotPos > lastHoles;
        int lastBlockIndex = blockCount - 1;
        index_t lastSlotIndexOffset = lastBlockIndex * ARIES_BLOCK_FILE_ROW_COUNT;
        int lastBlockRowCount = m_initTableMetaInfo.m_totalRowCount - lastSlotIndexOffset;
        int8_t* pLastFlags = m_slotBitmaps[ lastBlockIndex ];
        bool occupied;
        for( int i = 0; i < lastBlockRowCount; ++i )
        {
            GET_BIT_FLAG( pLastFlags, i, occupied );
            if ( !occupied )
                lastHoles.push_back( InitTableSlotPos{ lastSlotIndexOffset + i, lastBlockIndex, i } );
        }

        for( auto& t : allThreads )
        {
            auto holes = t.get();
            result.insert( result.end(), holes.begin(), holes.end() );
        }
        result.insert( result.end(), lastHoles.begin(), lastHoles.end() );

        return result;
    }

    void AriesInitialTable::BatchWriteBlockDatas( const vector< BlockDataBuffSPtr >& blockDataBuffs,
                                                  uint64 startSlotIdx,
                                                  const vector< int8_t* >& rowsData,
                                                  size_t& rowIdx, // start row index
                                                  uint32_t rowCount, // count of rows to write
                                                  const int colCount,
                                                  vector< index_t >& slotIndice )
    {
        for ( size_t i = 0; i < rowCount; ++rowIdx, ++i )
        {
            auto& rowData = rowsData[ rowIdx ];
            uint32_t buffPos = m_bitmapLength;
            for ( int colIndex = 0; colIndex < colCount; ++colIndex )
            {
                auto first = rowData + buffPos;
                auto last = first + m_itemStoreSizes[ colIndex ];
                auto buff = blockDataBuffs[ colIndex ];
                auto pos = buff->end();
                buff->insert( pos, first, last );
                buffPos += m_itemStoreSizes[ colIndex ];
            }
            slotIndice.push_back( startSlotIdx + i );
        }
    }

    BlockDataBuffSPtr AriesInitialTable::GetColumnBlockDataBuffer( int32_t colIndex, int32_t blockIndex, bool createIfNotExists )
    {
        BlockDataBuffSPtr buff = nullptr;
        int64_t mapKey = ColumnBuffKey( colIndex, blockIndex );
        auto it = m_columnBlockDataBuffMap.find( mapKey );
        if ( m_columnBlockDataBuffMap.end() != it )
        {
            buff = it->second;
        }
        else if ( createIfNotExists )
        {
            string blockFilePath = GetBlockFilePath( colIndex, blockIndex );
            int fd = open( blockFilePath.data(), O_RDWR );
            if ( -1 == fd )
            {
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION( EE_WRITE, blockFilePath.data(), errno,
                                 strerror_r( errno, errbuf, sizeof(errbuf) ) );
            }
            fd_helper_ptr fdPtr = make_shared< fd_helper >( fd );

            BlockFileHeader headerInfo;
            size_t readSize = read( fd, (char *) &headerInfo, sizeof( BlockFileHeader ) );
            if ( sizeof( BlockFileHeader ) != readSize )
            {
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION( EE_READ, blockFilePath.data(),
                                 errno, strerror_r( errno, errbuf, sizeof(errbuf)) );

            }
            lseek( fd, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, SEEK_SET );

            size_t itemStoreSize = m_itemStoreSizes[ colIndex ];
            size_t buffSize = itemStoreSize * headerInfo.rows;
            buff = make_shared< BlockDataBuff >();
            buff->reserve( buffSize );

            size_t sizeToRead = headerInfo.itemLen * headerInfo.rows;
            if ( sizeToRead > 0 )
            {
                buff->resize( sizeToRead );
                readSize = read( fd, buff->data(), sizeToRead );
                if ( readSize != sizeToRead )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_READ, blockFilePath.data(),
                                     errno, strerror_r( errno, errbuf, sizeof(errbuf)) );

                }
            }

            m_columnBlockDataBuffMap[ mapKey ] = buff;
        }

        return buff;
    }

    void AriesInitialTable::DeleteColumnBlockDataBuffer( int32_t colIndex, int32_t blockIndex )
    {
        int64_t mapKey = ColumnBuffKey( colIndex, blockIndex );
        m_columnBlockDataBuffMap.erase( mapKey );
    }

    bool AriesInitialTable::FillSlotHole( InitTableSlotPos& pos,
                                          vector< int8_t* >& columnBuffers,
                                          size_t rowIndex )
    {
        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            size_t itemStoreSize = m_itemStoreSizes[ colIndex ];

            int8_t* blockDataAddr = GetColumnBlockDataBuffer( colIndex, pos.m_blockIdx, true )->data();
            size_t offset = itemStoreSize * pos.m_blockSlotIdx;
            memcpy( blockDataAddr + offset,
                    columnBuffers[ colIndex ] + itemStoreSize * rowIndex,
                    itemStoreSize );
        }
        UpdateBlockBitmap( pos, true );
        return true;
    }

    void AriesInitialTable::XLogRecoverInsertBatch( int8_t* columnsData,
                                                    size_t columnCount,
                                                    size_t* columnSizes,
                                                    size_t rowCount,
                                                    RowPos* rowposes )
    {
        ARIES_ASSERT( columnCount == m_columnCount, "column count not match" );

        aries::CPU_Timer t;
        t.begin();

        LOG( INFO ) << "Batch insert " << rowCount << " rows into " + m_dbName + "." + m_tableName;
        size_t columnDataOffset = 0;
        vector< int8_t* > columnBuffers( columnCount );
        for ( size_t i = 0; i < columnCount; ++i )
        {
            columnBuffers[ i ] = columnsData + columnDataOffset;
            columnDataOffset += columnSizes[ i ] * rowCount;
        }

        size_t rowIndex = 0;

        size_t remainRowCount = rowCount - rowIndex;
        if ( remainRowCount > 0 )
        {
            size_t availSlotCount = m_capacity - m_initTableMetaInfo.m_totalRowCount;
            // fill the free slots of the last block
            if ( availSlotCount > 0 )
            {
                m_blockChanged[ m_initTableMetaInfo.m_blockCount - 1 ] = true;
                size_t maxCount = std::min( availSlotCount, remainRowCount );
                int32_t blockSlotIndex = BLOCK_SLOT_IDX( m_initTableMetaInfo.m_totalRowCount );
                for ( size_t colIndex = 0; colIndex < columnCount; ++colIndex )
                {
                    auto blockDataBuff = GetColumnBlockDataBuffer( colIndex, m_initTableMetaInfo.m_blockCount - 1, true );
                    auto first = columnBuffers[ colIndex ] + m_itemStoreSizes[ colIndex ] * rowIndex;
                    auto last = first + m_itemStoreSizes[ colIndex ] * maxCount;
                    blockDataBuff->insert( blockDataBuff->end(), first, last );
                }

                int8_t* bitmapPtr = m_slotBitmaps[ m_initTableMetaInfo.m_blockCount - 1 ];
                for ( size_t count = 0; count < maxCount; ++count, ++rowIndex, ++blockSlotIndex )
                {
                    if ( 0 != rowposes[ rowIndex ] )
                        SET_BIT_FLAG( bitmapPtr, blockSlotIndex );
                }

                m_initTableMetaInfo.m_totalRowCount += maxCount;
                remainRowCount -= maxCount;
            }

            // need new blocks
            if ( remainRowCount > 0 )
            {
                int newBlockCount = DIV_UP( remainRowCount, ARIES_BLOCK_FILE_ROW_COUNT );
                uint32_t lastBlockRowCount = remainRowCount - ( newBlockCount - 1 ) * ARIES_BLOCK_FILE_ROW_COUNT;

                for ( int blockIdx = 0; blockIdx < newBlockCount; ++blockIdx )
                {
                    uint32_t blockRowCount = ( blockIdx == newBlockCount - 1 ) ?
                                             lastBlockRowCount : ARIES_BLOCK_FILE_ROW_COUNT;
                    auto colDataBlockBuffs = XLogRecoverNewBlockBuffs( blockRowCount, m_initTableMetaInfo.m_blockCount );
                    for ( size_t colIndex = 0; colIndex < columnCount; ++colIndex )
                    {
                        auto first = columnBuffers[ colIndex ] + m_itemStoreSizes[ colIndex ] * rowIndex;
                        auto last = first + m_itemStoreSizes[ colIndex ] * blockRowCount;
                        colDataBlockBuffs[ colIndex ]->insert( colDataBlockBuffs[ colIndex ]->end(), first, last );
                    }
                    int8_t* bitmapPtr = m_slotBitmaps[ m_initTableMetaInfo.m_blockCount - 1 ];
                    for ( size_t slotIndex = 0; slotIndex < blockRowCount; ++slotIndex, ++rowIndex )
                    {
                        if ( 0 != rowposes[ rowIndex ] )
                            SET_BIT_FLAG( bitmapPtr, slotIndex );
                    }
                    m_initTableMetaInfo.m_totalRowCount += blockRowCount;
                }
            }
        }
        LOG( INFO ) << "Batch insert " << rowCount << " rows into "
                    <<  m_dbName + "." + m_tableName + ", time: " << t.end() << "ms";
    }

    vector< index_t > AriesInitialTable::XLogRecoverInsertRows( const vector< int8_t* >& rowsData )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        LOG( INFO ) << "Add " << rowsData.size() << " rows into " + m_dbName + "." + m_tableName;

        vector< index_t > slotIndice;
        size_t rowCount = rowsData.size();
        size_t rowIdx = 0;

        size_t remainRowCount = rowCount - rowIdx;
        if ( remainRowCount > 0 )
        {
            size_t availSlotCount = m_capacity - m_initTableMetaInfo.m_totalRowCount;

            // fill the free slots of the last block
            if ( availSlotCount > 0 )
            {
                m_blockChanged[ m_initTableMetaInfo.m_blockCount - 1 ] = true;
                size_t maxCount = std::min( availSlotCount, remainRowCount );
                vector< BlockDataBuffSPtr > blockDataBuffs;
                for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
                {
                    blockDataBuffs.push_back( GetColumnBlockDataBuffer( colIndex, m_initTableMetaInfo.m_blockCount - 1, true ) );
                }
                // 写数据文件
                BatchWriteBlockDatas( blockDataBuffs,
                                      m_initTableMetaInfo.m_totalRowCount,
                                      rowsData,
                                      rowIdx,
                                      maxCount,
                                      m_columnCount,
                                      slotIndice );

                // 更新bitmap
                UpdateBlockBitmaps( m_initTableMetaInfo.m_blockCount - 1,
                                    BLOCK_SLOT_IDX( m_initTableMetaInfo.m_totalRowCount ),
                                    maxCount,
                                    true );

                m_initTableMetaInfo.m_totalRowCount += maxCount;
                remainRowCount -= maxCount;
            }

            // need new blocks
            if ( remainRowCount > 0 )
            {
                int newBlockCount = DIV_UP( remainRowCount, ARIES_BLOCK_FILE_ROW_COUNT );
                uint32_t lastBlockRowCount = remainRowCount - ( newBlockCount - 1 ) * ARIES_BLOCK_FILE_ROW_COUNT;
                for ( int blockIdx = 0; blockIdx < newBlockCount; ++blockIdx )
                {
                    uint32_t blockRowCount = ( blockIdx == newBlockCount - 1 ) ?
                                             lastBlockRowCount : ARIES_BLOCK_FILE_ROW_COUNT;
                    auto colDataBlockAddrs = XLogRecoverNewBlockBuffs( blockRowCount, m_initTableMetaInfo.m_blockCount );
                    BatchWriteBlockDatas( colDataBlockAddrs,
                                          m_initTableMetaInfo.m_totalRowCount,
                                          rowsData,
                                          rowIdx,
                                          blockRowCount,
                                          m_columnCount,
                                          slotIndice );
                    UpdateBlockBitmaps( m_initTableMetaInfo.m_blockCount - 1,
                                        0,
                                        blockRowCount,
                                        true );
                    m_initTableMetaInfo.m_totalRowCount += blockRowCount;
                }
            }
        }
#ifdef ARIES_PROFILE
        LOG( INFO ) << "Add " << rowsData.size() << " rows into " + m_dbName + "." + m_tableName + ", time: " << t.end() << "ms";
#endif
        return slotIndice;
    }

    bool AriesInitialTable::UpdateFileRows( const vector< UpdateRowDataPtr >& updateRowDatas )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        LOG( INFO ) << "Update " << updateRowDatas.size() << " rows of " + m_dbName + "." + m_tableName;

        unordered_map< string, fd_helper_ptr > blockFds;

        for ( auto rowData : updateRowDatas )
        {
            index_t rowIdx = rowData->m_rowIdx;
            CheckRowIndex( rowIdx );

            int32_t blockIdx = BLOCK_IDX( rowIdx );
            index_t slotIdx = BLOCK_SLOT_IDX( rowIdx );

            bool validFlag = false;
            uint32_t buffPos = m_bitmapLength;
            for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
            {
                GET_BIT_FLAG( rowData->m_colDataBuffs, colIndex, validFlag );
                if ( !validFlag )
                    continue;

                string blockFilePath = GetBlockFilePath( colIndex, blockIdx );
                int fd;
                auto fdIt = blockFds.find( blockFilePath );
                if ( blockFds.end() == fdIt )
                {
                    fd = open( blockFilePath.data(), O_RDWR );
                    if ( -1 == fd )
                    {
                        char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                        ARIES_EXCEPTION( EE_READ, blockFilePath.data(), errno,
                                         strerror_r( errno, errbuf, sizeof(errbuf) ) );
                    }
                    blockFds[ blockFilePath ] = make_shared< fd_helper >( fd );
                }
                else
                {
                    fd = fdIt->second->GetFd();
                }

                size_t itemStoreSize = m_itemStoreSizes[ colIndex ];
                off_t offset = ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE + slotIdx * itemStoreSize;
                /* offset for mmap() must be page aligned */
                off_t paOffset = offset & ~( m_pageSize - 1);
                void* mapAddr = mmap( NULL, itemStoreSize + offset - paOffset,
                                      PROT_READ | PROT_WRITE,
                                      MAP_SHARED,
                                      fd,
                                      paOffset );
                if ( MAP_FAILED == mapAddr )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_WRITE, m_metaInfoFilePath.data(), errno,
                                     strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }
                memcpy( ( char* )mapAddr + offset - paOffset,
                         rowData->m_colDataBuffs + buffPos,
                         itemStoreSize );
                buffPos += itemStoreSize;
                int unmapRet = munmap( mapAddr, itemStoreSize + offset - paOffset );
                if ( 0 != unmapRet )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION(  EE_WRITE, fdIt->first.data(), errno,
                                      strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }
            }
        }
#ifdef ARIES_PROFILE
        LOG( INFO ) << "Update " << updateRowDatas.size() << " rows of " + m_dbName + "." + m_tableName + ", time: " << t.end() << "ms";
#endif
        return true;
    }

    bool AriesInitialTable::XLogRecoverDeleteRows( const vector< index_t >& rowIndice )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        LOG( INFO ) << "Delete " << rowIndice.size() << " rows from " + m_dbName + "." + m_tableName;

        for ( auto rowIdx : rowIndice )
        {
            UpdateBitmap( rowIdx, false );
        }
#ifdef ARIES_PROFILE
        t.end();
        LOG( INFO ) << "Delete " << rowIndice.size() << " rows from " + m_dbName + "." + m_tableName + " time: " << t.end() << "ms";
#endif
        return true;
    }

    string AriesInitialTable::GetPartitionMetaFilePath( const string& dbName,
                                                        const string& tableName,
                                                        uint32_t partitionIndex )
    {
        string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbName, tableName );
        return tableDataDir + "/meta_p" + std::to_string( partitionIndex );
    }
    string AriesInitialTable::GetMetaFilePath( const string& dbName, const string& tableName )
    {
        string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbName, tableName );
        return tableDataDir + "/" + ARIES_INIT_TABLE_META_FILE_NAME;
    }

    void AriesInitialTable::WritePartitionMetaFile( uint32_t partitiontIndex, const PartitionMetaInfo &metaInfo )
    {
        auto metaFilePath = GetPartitionMetaFilePath( partitiontIndex );
        auto metaFileFd = open( metaFilePath.data(), O_RDWR );
        if ( -1 == metaFileFd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_CANTCREATEFILE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }

        PartitionMetaInfoHeader header;
        header.Version = metaInfo.Version;
        header.RowCount = metaInfo.RowCount;
        header.BlockCount = metaInfo.BlockCount;

        size_t writtenSize = my_write( metaFileFd,
                                       ( const uchar* ) &header,
                                       sizeof( PartitionMetaInfoHeader ),
                                       MYF( MY_FNABP ) );
        if ( 0 != writtenSize )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
        writtenSize = my_write( metaFileFd,
                                       ( const uchar* ) metaInfo.BlocksID.data(),
                                       sizeof( int ) * metaInfo.BlocksID.size(),
                                       MYF( MY_FNABP ) );
        if ( 0 != writtenSize )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }

        close( metaFileFd );
    }
    string AriesInitialTable::GetPartitionMetaFilePath( uint32_t partitionIndex )
    {
        return m_dataDir + "/meta_p" + std::to_string( partitionIndex );
    }
    string AriesInitialTable::GetMetaFilePath()
    {
        return m_dataDir + "/" + ARIES_INIT_TABLE_META_FILE_NAME;
    }

    void AriesInitialTable::WriteMetaFile(
        const string& filePath,
        size_t totalRowCount,
        size_t blockCount )
    {
        int metaFileFd = open( filePath.data(),
                               O_CREAT | O_RDWR, S_IRUSR | S_IWUSR );
        if ( -1 == metaFileFd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_CANTCREATEFILE, filePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof( errbuf ) ) );
        }

        if ( UINT64_MAX == blockCount  )
            blockCount = DIV_UP( totalRowCount, ARIES_BLOCK_FILE_ROW_COUNT );
        // 分配 ARIES_BLOCK_FILE_ROW_COUNT 整数倍bitmap空间
        uint64_t bitmapSize = ARIES_INIT_FILE_BITMAP_SIZE( blockCount );
        shared_ptr< int8_t[] > rawSPtr;
        size_t metaInfoSize = sizeof( InitTableMetaInfo ) + bitmapSize;
        rawSPtr.reset( new int8_t[ metaInfoSize ] );

        InitTableMetaInfo* metaInfoPtr = (InitTableMetaInfo* )rawSPtr.get();
        metaInfoPtr->m_version = AriesInitialTable::COLUMN_FILE_VERSION;
        metaInfoPtr->m_totalRowCount = totalRowCount;
        metaInfoPtr->m_blockMaxRowCount = ARIES_BLOCK_FILE_ROW_COUNT;
        metaInfoPtr->m_blockCount = blockCount;
        memset( metaInfoPtr->m_reserved, 0, sizeof( metaInfoPtr->m_reserved ) );

        // fill bitmap
        memset( metaInfoPtr->m_slotBitmap, 0, bitmapSize );
        size_t bitmapByteCnt = totalRowCount / 8;
        if ( bitmapByteCnt > 0 )
        {
            memset( metaInfoPtr->m_slotBitmap, 0xff, bitmapByteCnt );
        }
        size_t remainRowsStartIdx = bitmapByteCnt * 8;
        size_t rowIdx = remainRowsStartIdx;
        while ( rowIdx < totalRowCount )
        {
            SET_BIT_FLAG( metaInfoPtr->m_slotBitmap, rowIdx );
            ++rowIdx;
        }

        auto writtenSize = my_write( metaFileFd,
                                     ( const uchar* ) metaInfoPtr,
                                     metaInfoSize,
                                     MYF( MY_FNABP ) );
        close( metaFileFd );
        if ( 0 != writtenSize )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION(EE_WRITE, filePath.data(), errno,
                            strerror_r( errno, errbuf, sizeof(errbuf)));
        }

    }

    void AriesInitialTable::WritePartitionMetaFile(
        const string& filePath,
        size_t totalRowCount,
        const std::vector< int32_t > &BlocksID )
    {

    }

    void AriesInitialTable::InitColumnFiles()
    {
        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            NewBlockFile( colIndex, 0, 0 );
        }
        m_slotBitmaps.push_back( NewBlockBitmap() );

        ++m_initTableMetaInfo.m_blockCount;
        SetCapacity();
    }
    void AriesInitialTable::InitFiles( const string& dbName,
                                       const string& tableName )
    {
        string metaFilePath = GetMetaFilePath( dbName, tableName );
        int metaFileFd = open( metaFilePath.data(),
                               O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
        if ( -1 == metaFileFd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_CANTCREATEFILE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }

        uint32_t blockCount = 1;
        uint64_t bitmapSize = ARIES_INIT_FILE_BITMAP_SIZE( blockCount );
        shared_ptr< int8_t[] > rawSPtr;
        size_t metaInfoSize = sizeof( InitTableMetaInfo ) + bitmapSize;
        rawSPtr.reset( new int8_t[ metaInfoSize ] );

        InitTableMetaInfo* metaInfoPtr = (InitTableMetaInfo* )rawSPtr.get();
        metaInfoPtr->m_version = AriesInitialTable::COLUMN_FILE_VERSION;
        metaInfoPtr->m_totalRowCount = 0;
        metaInfoPtr->m_blockMaxRowCount = ARIES_BLOCK_FILE_ROW_COUNT;
        metaInfoPtr->m_blockCount = blockCount;
        memset( metaInfoPtr->m_reserved, 0, sizeof( metaInfoPtr->m_reserved ) );
        memset( metaInfoPtr->m_slotBitmap, 0, bitmapSize );

        size_t writtenSize = my_write( metaFileFd,
                                       ( const uchar* ) metaInfoPtr,
                                       metaInfoSize,
                                       MYF( MY_FNABP ) );
        close( metaFileFd );
        if ( 0 != writtenSize )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }

        auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
        auto tableEntry = dbEntry->GetTableByName( tableName );
        if ( tableEntry->IsPartitioned() )
        {
            auto partitionCount = tableEntry->GetPartitionCount();
            for ( size_t i = 0; i < partitionCount; ++i )
            {
                metaFilePath = GetPartitionMetaFilePath( dbName, tableName, i );
                metaFileFd = open( metaFilePath.data(),
                                       O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
                if ( -1 == metaFileFd )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_CANTCREATEFILE, metaFilePath.data(), errno,
                                     strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }

                metaInfoSize = sizeof( PartitionMetaInfoHeader );

                PartitionMetaInfoHeader partMetaInfor;
                partMetaInfor.Version = AriesInitialTable::COLUMN_FILE_VERSION;
                partMetaInfor.RowCount = 0;
                partMetaInfor.BlockCount = 0;

                size_t writtenSize = my_write( metaFileFd,
                                               ( const uchar* ) &partMetaInfor,
                                               metaInfoSize,
                                               MYF( MY_FNABP ) );
                close( metaFileFd );
                if ( 0 != writtenSize )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                                     strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }

            }
        }

        AriesInitialTable initTable( dbName, tableName );
        initTable.InitColumnFiles();
    }

    void AriesInitialTable::TruncateBlocks( const size_t emptyBlockCount )
    {
        ARIES_ASSERT( emptyBlockCount <= m_initTableMetaInfo.m_blockCount,
                      "total block count: " + std::to_string( m_initTableMetaInfo.m_blockCount ) +
                      ", truncate block count: " + std::to_string( emptyBlockCount ) );

        int32_t blockIdx = m_initTableMetaInfo.m_blockCount - 1;
        for ( size_t i = 0; i < emptyBlockCount; ++i, --blockIdx )
        {
            for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
            {
                LOG( INFO ) << "Deleting empty block file " << blockIdx;
                DeleteColumnBlockDataBuffer( colIndex, blockIdx );
            }
            auto blockBitmap = m_slotBitmaps.back();
            m_slotBitmaps.pop_back();
            delete[] blockBitmap;
        }

        m_initTableMetaInfo.m_blockCount = m_initTableMetaInfo.m_blockCount - emptyBlockCount;
        // keep at least one empty block
        if ( 0 == m_initTableMetaInfo.m_blockCount )
            XLogRecoverNewBlockBuffs( 0, 0 );
        else
            SetCapacity();
        m_blockChanged.resize( m_initTableMetaInfo.m_blockCount );
    }

    void AriesInitialTable::SetBlockBitmap( const int32_t blockIdx )
    {
        int8_t* bitmapPtr = m_slotBitmaps[ blockIdx ];
        memset( bitmapPtr, 0xff, ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE );
    }

    void AriesInitialTable::UpdateBlockBitmaps( const int32_t blockIdx,
                                                const index_t startSlotIdx,
                                                const size_t  count,
                                                const bool    set )
    {
        int8_t* bitmapPtr = m_slotBitmaps[ blockIdx ];

        index_t startByteOffset = startSlotIdx / 8;
        index_t endSlotIdx = startSlotIdx + count - 1; // inclusive
        index_t endByteOffset = endSlotIdx / 8;

        int8_t bitOffset = startSlotIdx % 8;
        index_t idx = startSlotIdx;
        uint8_t maxBitCount = 8;
        if ( count < maxBitCount )
            maxBitCount = count;
        // the first byte
        for ( uint8_t i = bitOffset; i < maxBitCount; ++i, ++idx )
        {
            if ( set )
                SET_BIT_FLAG( bitmapPtr, idx );
            else
                CLEAR_BIT_FLAG( bitmapPtr, idx );
        }
        // whole bytes in the middle
        int32_t byteCount = endByteOffset - startByteOffset + 1;
        if (  byteCount > 2 )
        {
            if ( set )
                memset( bitmapPtr + startByteOffset + 1, 0xff, byteCount - 2 );
            else
                memset( bitmapPtr + startByteOffset + 1, 0, byteCount - 2 );
            idx += ( byteCount - 2 ) * 8;
        }
        // the last byte
        for ( ; idx <= endSlotIdx; ++idx )
        {
            if ( set )
                SET_BIT_FLAG( bitmapPtr, idx );
            else
                CLEAR_BIT_FLAG( bitmapPtr, idx );
        }
    }

    void AriesInitialTable::UpdateBitmap( index_t& rowIdx, bool set )
    {
        CheckRowIndex( rowIdx );
        InitTableSlotPos pos;
        pos.m_slotIdx = rowIdx;
        pos.m_blockIdx = BLOCK_IDX( rowIdx );
        pos.m_blockSlotIdx = BLOCK_SLOT_IDX( rowIdx );
        UpdateBlockBitmap( pos, set );
    }

    void AriesInitialTable::UpdateBlockBitmap( InitTableSlotPos& pos, bool set )
    {
        m_blockChanged[ pos.m_blockIdx ] = true;
        if ( set )
            SET_BIT_FLAG( m_slotBitmaps[ pos.m_blockIdx ], pos.m_blockSlotIdx );
        else
            CLEAR_BIT_FLAG( m_slotBitmaps[ pos.m_blockIdx ], pos.m_blockSlotIdx );
    }

    bool AriesInitialTable::FillSlotHole( InitTableSlotPos& pos,
                                          const int8_t* rowData )
    {
        uint32_t buffPos = m_bitmapLength;
        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            size_t itemStoreSize = m_itemStoreSizes[ colIndex ];

            auto blockDataBuff = GetColumnBlockDataBuffer( colIndex, pos.m_blockIdx, true );
            size_t offset = itemStoreSize * pos.m_blockSlotIdx;
            memcpy( blockDataBuff->data() + offset,
                    rowData + buffPos,
                    itemStoreSize );
            buffPos += itemStoreSize;
        }
        UpdateBlockBitmap( pos, true );
        return true;
    }

    void AriesInitialTable::CheckRowIndex( index_t rowIdx )
    {
        ARIES_ASSERT( rowIdx >= 0 && ( uint64 )rowIdx < m_initTableMetaInfo.m_totalRowCount,
                      "invalid row idx: " + std::to_string( rowIdx ) +
                      ", table " + m_dbName + "." + m_tableName +
                      ", total row count: " + std::to_string( m_initTableMetaInfo.m_totalRowCount ) );
    }
    InitTableSlotPos AriesInitialTable::ReverseFindDataSlot( index_t endIdx )
    {
        InitTableSlotPos pos;
        bool occupied;
        for ( index_t i = endIdx; i >= 0 ; --i )
        {
            int32_t blockIdx = BLOCK_IDX( i );
            index_t blockSlotIdx = BLOCK_SLOT_IDX( i );

            GET_BIT_FLAG( m_slotBitmaps[ blockIdx ], blockSlotIdx, occupied );
            if ( occupied )
            {
                pos.m_slotIdx = i;
                pos.m_blockIdx = blockIdx;
                pos.m_blockSlotIdx = blockSlotIdx;
                return pos;
            }
        }
        pos.m_slotIdx = NULL_INDEX;
        return pos;

    }

    vector< InitTableSlotPos > AriesInitialTable::ReverseFindAllDataSlots( int64_t count )
    {
        assert( count > 0 );
        vector< InitTableSlotPos > result;
        result.reserve( count );
        bool occupied;
        for( index_t i = m_initTableMetaInfo.m_totalRowCount - 1; i >= 0 && count > 0; --i )
        {
            int32_t blockIdx = BLOCK_IDX( i );
            index_t blockSlotIdx = BLOCK_SLOT_IDX( i );

            GET_BIT_FLAG( m_slotBitmaps[ blockIdx ], blockSlotIdx, occupied );
            if ( occupied )
            {
                --count;
                result.push_back( InitTableSlotPos{ i, blockIdx, blockSlotIdx } );
            }
        }
        return result;
    }

    // static int GetFds( unordered_map< string, fd_helper_ptr >& fds,
    //                    const string& filePath )
    // {
    //     int fd;
    //     auto fdIt = fds.find( filePath );
    //     if ( fds.end() == fdIt )
    //     {
    //         fd = open( filePath.data(), O_RDWR );
    //         if ( -1 == fd )
    //         {
    //             char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
    //             ARIES_EXCEPTION( EE_READ, filePath.data(), errno,
    //                              strerror_r( errno, errbuf, sizeof(errbuf) ) );
    //         }
    //         fds[ filePath ] = make_shared< fd_helper >( fd );
    //     }
    //     else
    //     {
    //         fd = fdIt->second->GetFd();
    //     }
    //     return fd;
    // }
    void AriesInitialTable::Sweep()
    {
        aries::CPU_Timer t;
        t.begin();
        LOG( INFO ) << "Sweep " + m_dbName + "." + m_tableName;
        auto allHoles = FindAllSlotHoles();
        if( !allHoles.empty() )
        {
            auto allDatas = ReverseFindAllDataSlots( allHoles.size() );
            int holeIndex = 0;
            for( const auto& dataPos : allDatas )
            {
                const auto& holePos = allHoles[ holeIndex++ ];
                if( dataPos.m_slotIdx > holePos.m_slotIdx )
                {
                    for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
                    {
                        m_blockChanged[ dataPos.m_blockIdx ] = true;
                        m_blockChanged[ holePos.m_blockIdx ] = true;
                        auto srcDataBuff = GetColumnBlockDataBuffer( colIndex, dataPos.m_blockIdx, true );
                        auto dstDataBuff = GetColumnBlockDataBuffer( colIndex, holePos.m_blockIdx, true );

                        size_t itemStoreSize = m_itemStoreSizes[ colIndex ];
                        off_t srcOffset = dataPos.m_blockSlotIdx * itemStoreSize;
                        off_t dstOffset = holePos.m_blockSlotIdx * itemStoreSize;
                        memcpy( dstDataBuff->data() + dstOffset,
                                srcDataBuff->data() + srcOffset,
                                itemStoreSize );
                    }
                    SET_BIT_FLAG( m_slotBitmaps[ holePos.m_blockIdx ], holePos.m_blockSlotIdx );
                    CLEAR_BIT_FLAG( m_slotBitmaps[ dataPos.m_blockIdx ], dataPos.m_blockSlotIdx );
                }
            }
            m_initTableMetaInfo.m_totalRowCount -= allHoles.size();
        }

        // delete empty blocks if necessary
        size_t emptyBlockCount = ( m_capacity - m_initTableMetaInfo.m_totalRowCount ) / ARIES_BLOCK_FILE_ROW_COUNT;
        LOG( INFO ) << "remain row count of " + m_dbName + "." + m_tableName << ": " <<
                    m_initTableMetaInfo.m_totalRowCount << ", empty block count: " << emptyBlockCount;
        if ( emptyBlockCount > 0 )
            TruncateBlocks( emptyBlockCount );

        LOG( INFO ) << "Sweep " + m_dbName + "." + m_tableName + ", time: " << t.end() << "ms";
    }

    void AriesInitialTable::XLogRecoverDone()
    {
        FlushXLogRecoverResult();
    }

    void AriesInitialTable::FlushXLogRecoverResult()
    {
        aries::CPU_Timer t;
        t.begin();

        LOG( INFO ) << "Flush xlog recover result";

        string dataXlogRecoverDir = Configuartion::GetInstance().GetDataXLogRecoverDirectory( m_dbName, m_tableName );
        if ( !boost::filesystem::exists( dataXlogRecoverDir ) )
        {
            if ( !boost::filesystem::create_directories( dataXlogRecoverDir ) )
            {
                char errbuf[ 1014 ] = {0};
                ARIES_EXCEPTION( EE_CANT_MKDIR, dataXlogRecoverDir.data(), errno,
                                 strerror_r( errno, errbuf, sizeof(errbuf) ) );
            }
        }

        // write meta file header
        string metaFilePath(  dataXlogRecoverDir );
        metaFilePath.append( "/" ).append( ARIES_INIT_TABLE_META_FILE_NAME );
        int fd = open( metaFilePath.data(),
                       O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
        if ( -1 == fd )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }
        auto fdHelper = make_shared< fd_helper >( fd );

       auto writtenSize = my_write( fd,
                                    ( const uchar* )&m_initTableMetaInfo,
                                    sizeof( InitTableMetaInfo ),
                                    MYF( MY_FNABP ) );
        if ( 0 != writtenSize )
        {
            char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
            ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                             strerror_r( errno, errbuf, sizeof(errbuf) ) );
        }

        // no holes
        assert( NULL_INDEX == FindSlotHole(0).m_slotIdx );

        // write slot bitmap
        for ( auto* bitmap : m_slotBitmaps )
        {
            writtenSize = my_write( fd,
                                    ( const uchar* )bitmap,
                                    ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE,
                                    MYF( MY_FNABP ) );
            if ( 0 != writtenSize )
            {
                char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                ARIES_EXCEPTION( EE_WRITE, metaFilePath.data(), errno,
                                 strerror_r( errno, errbuf, sizeof(errbuf) ) );
            }
        }
        fdHelper.reset();

        // write block files
        for ( size_t colIndex = 0; colIndex < m_columnCount; ++colIndex )
        {
            auto columnEntry = m_tableEntry->GetColumnById( colIndex + 1 );

            uint64_t remainRowCount = m_initTableMetaInfo.m_totalRowCount;
            for ( size_t blockIndex = 0; blockIndex < m_initTableMetaInfo.m_blockCount; ++blockIndex )
            {
                uint64_t blockRowCount = std::min( remainRowCount, ( uint64_t)ARIES_BLOCK_FILE_ROW_COUNT );
                remainRowCount -= blockRowCount;
                auto blockDataBuff = GetColumnBlockDataBuffer( colIndex, blockIndex, m_blockChanged[ blockIndex ] );
                if ( !blockDataBuff )
                    continue;

                string colBlockFilePath( dataXlogRecoverDir );
                string fileName = GetBlockFileName( m_tableName,
                                                    colIndex,
                                                    blockIndex,
                                                    EncodeType::DICT == columnEntry->encode_type );
                colBlockFilePath.append( "/" ).append( fileName );
                fd = open( colBlockFilePath.data(),
                           O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
                if ( -1 == fd )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_WRITE, colBlockFilePath.data(), errno,
                                     strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }
                fdHelper = make_shared< fd_helper >( fd );

                // write block file header
                WriteColumnBlockFileHeader( columnEntry, fd, colBlockFilePath, blockRowCount, false );

                // empty block
                if ( 0 == blockRowCount )
                    continue;

                // write block data
                size_t sizeToWrite = blockRowCount * m_itemStoreSizes[ colIndex ];

                writtenSize = my_write( fd,
                                        ( const uchar* )blockDataBuff->data(),
                                        sizeToWrite,
                                        MYF( MY_FNABP ) );
                fdHelper.reset();
                if ( 0 != writtenSize )
                {
                    char errbuf[ MYSYS_STRERROR_SIZE ] = {0};
                    ARIES_EXCEPTION( EE_WRITE, colBlockFilePath.data(), errno,
                                     strerror_r( errno, errbuf, sizeof(errbuf) ) );
                }
            }
        }

        LOG( INFO ) << "Flush xlog recover result time " << t.end();
    }

    std::string AriesInitialTable::GetDbName() const
    {
        return m_dbName;
    }
    std::string AriesInitialTable::GetTableName() const
    {
        return m_tableName;
    }

    // static members functions
    string AriesInitialTable::GetColumnFileBaseName( const string& tableName, const int colIndex )
    {
        // customer0
        return tableName + std::to_string( colIndex );
    }
    string AriesInitialTable::GetBlockFileName( const string& tableName,
                                                const int colIndex,
                                                const uint32_t blockIndex,
                                                bool isDictEncoded )
    {
        if ( isDictEncoded )
            // customer6_dict_idx_0
            return GetDictIndiceFileBaseName( tableName, colIndex ) + "_" +  std::to_string( blockIndex );
        else
            // customer0_0
            return GetColumnFileBaseName( tableName, colIndex ) + "_" + std::to_string( blockIndex );
    }

    string AriesInitialTable::GetDictIndiceFileBaseName( const string& tableName, const int colIndex )
    {
        // customer6_dict_idx
        return GetColumnFileBaseName( tableName, colIndex ) + ARIES_DICT_COLUMN_FILE_NAME_SUFFIX;
    }

    void AriesInitialTable::CheckRowPos( const RowPos pos ) const
    {
        ARIES_ASSERT( pos < 0 && ( size_t )( -pos ) <= m_initialTupleHeader.size(), "id is invalid" );
    }

    // BEGIN read column data of a parition
    void AriesInitialTable::ReadColumn( int32_t colIndex,
                                        AriesTableBlockUPtr& table,
                                        uint32_t partitionIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        auto colId = colIndex + 1;

        auto columnEntry = m_tableEntry->GetColumnById( colId );
        if ( aries::EncodeType::DICT == columnEntry->encode_type )
        {
            ReadDictColumn( colIndex, table, partitionIndex );
            return;
        }

        int64_t readColumnTime = 0;
#ifdef ARIES_PROFILE
        t.begin();
#endif
        auto column = ReadColumnBlockFiles( colIndex, partitionIndex );
        table->AddColumn( colId, column );
#ifdef ARIES_PROFILE
        readColumnTime += t.end();
        float s = ( readColumnTime + 0.0 ) / 1000;
        LOG( INFO ) << "Read column time: " << std::to_string( s ) << "s";
#endif
    }

    void AriesInitialTable::ReadDictColumn(
        int32_t colIndex,
        AriesTableBlockUPtr& table,
        uint32_t partitionIndex )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        auto columnDict = GetColumnDict( colIndex );
        auto indices = ReadColumnBlockFiles( colIndex, partitionIndex );
#ifdef ARIES_PROFILE
        float s = ( t.end() + 0.0 ) / 1000;
        LOG( INFO ) << "Read dict column time " << std::to_string( s ) << "s";
#endif
        auto dictColumn = make_shared< AriesDictEncodedColumn >( columnDict, indices );
        table->AddColumn( colIndex + 1, dictColumn );
    }

    AriesColumnSPtr AriesInitialTable::ReadColumnBlockFiles(
        int32_t colIndex,
        uint32_t partitionIndex )
    {
        ColumnFileInfoPtr colFileInfo = m_allColumnInfos[ colIndex + 1 ];

        const auto& dataType = colFileInfo->m_dataType;
        auto column = make_shared< AriesColumn >();

        if ( 0 == m_initTableMetaInfo.m_totalRowCount )
        {
            column->SetColumnType( colFileInfo->m_dataType );
            return column;
        }

        uint64_t readRowCount = 0;

        auto &partitions = GetPartitionMetaInfo();
        auto partition = partitions[ partitionIndex ];
        auto &partionBlockIndices = partition.BlocksID;
        for ( auto blockIndex : partionBlockIndices )
        {
            size_t offset = ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE;
            auto rowCount = m_rowCountOfBlocks[ blockIndex ];
            if ( 0 == blockIndex && UINT32_MAX != m_readStartBlockLineIndex )
            {
                offset += dataType.GetDataTypeSize() * m_readStartBlockLineIndex;
                rowCount -= m_readStartBlockLineIndex;
            }

            // 跳过文件头
            colFileInfo->m_blockFileStreams[ blockIndex ]->seekg( offset, ios::beg );
            AriesDataBufferSPtr data_buffer = make_shared< AriesDataBuffer >( dataType, rowCount );
            data_buffer->PrefetchToCpu();
            char *buf = reinterpret_cast< char * >( data_buffer->GetData() );
            colFileInfo->m_blockFileStreams[ blockIndex ]->read( buf, data_buffer->GetTotalBytes() );
            colFileInfo->m_blockFileStreams[ blockIndex ] = nullptr;
            data_buffer->MemAdvise( cudaMemAdviseSetReadMostly, 0 );

            column->AddDataBuffer( data_buffer );

            readRowCount += m_rowCountOfBlocks[ blockIndex ];
        }
        if( readRowCount != partition.RowCount )
        {
            ARIES_EXCEPTION( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
        }
        return column;
    }

    AriesTableBlockUPtr AriesInitialTable::GetPartitionData(
        const vector< int >& columnIds,
        uint32_t partitionIndex )
    {
        AriesTableBlockUPtr resultTable = make_unique< AriesTableBlock >();
        Open( columnIds );

        for( int id : columnIds )
        {
            if ( IsCurrentThdKilled() )
            {
                LOG(INFO) << "thread killed while scan file";
                SendKillMessage();
            }

            if ( ( size_t )id == m_columnCount + 1 ) // rowId column
            {
                continue;
            }

            try
            {
                ReadColumn( id - 1, resultTable, partitionIndex );
            }
            catch ( AriesException& e )
            {
                throw;
            }
            catch (...)
            {
                //读取文件出错
                ARIES_EXCEPTION( ER_TABLE_CORRUPT, m_dbName.data(), m_tableName.data() );
            }
        }
        return std::move( resultTable );

    }
    // END read column data of a parition

    bool WriteColumnDataIntoBlocks( AriesInitialTable &initTable,
                                    int colIndex,
                                    bool nullable,
                                    size_t itemSize,
                                    int8_t* buff,
                                    size_t rowCount )
    {
        uint32_t blockIndex = 0;
        size_t rowCountToWrite = std::min( rowCount, ( size_t )ARIES_BLOCK_FILE_ROW_COUNT );
        size_t rowCountWritten = 0;
        do
        {
            auto path = initTable.GetBlockFilePath( colIndex, blockIndex++ );
            auto fd = open( path.data(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR );

            rowCountToWrite = std::min( rowCount - rowCountWritten, ( size_t )ARIES_BLOCK_FILE_ROW_COUNT );

            BlockFileHeader header;
            header.containNull = nullable;
            header.rows = rowCountToWrite;
            header.itemLen = itemSize;
            ::lseek( fd, 0, SEEK_SET );
            auto written = write( fd, &header, sizeof( header ) );
            if ( written == -1 )
            {
                close( fd );
                std::cerr << "cannot write file: " << path << std::endl;
                return false;
            }

            ::lseek( fd, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, SEEK_SET );
            written = write( fd, buff + itemSize * rowCountWritten, itemSize * rowCountToWrite );
            if ( written == -1 )
            {
                close( fd );
                std::cerr << "cannot write file: " << path << std::endl;
                return false;
            }

            rowCountWritten += rowCountToWrite;

            fsync( fd );
            close( fd );

        } while ( rowCountWritten < rowCount );
        
        return true;
    }

END_ARIES_ENGINE_NAMESPACE
