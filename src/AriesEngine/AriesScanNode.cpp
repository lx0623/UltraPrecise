/*
 * AriesScanNode.cpp
 *
 *  Created on: Aug 31, 2018
 *      Author: lichi
 */

#include <schema/SchemaManager.h>
#include "AriesScanNode.h"
#include "AriesUtil.h"
#include "AriesDataCache.h"
#include "transaction/AriesInitialTable.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesScanNode::AriesScanNode( const string& dbName )
            : m_dbName( dbName ), m_readRowCount( 0 ), m_totalRowNum( 0 )
    {
        // TODO Auto-generated constructor stub
    }

    AriesScanNode::~AriesScanNode()
    {
        m_outputColumnIds.clear();
        m_files.clear();
    }

    void AriesScanNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        // assert( !columnIds.empty() );
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void AriesScanNode::SetPhysicalTable( PhysicalTablePointer physicalTable )
    {
        ARIES_ASSERT( physicalTable, "physicalTable is nullptr");
        m_physicalTable = physicalTable;
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
     *   -1: file is not exist
     *   -2: header size in file is not correct
     *   -3: header version in file is not correct
     * */
    int AriesScanNode::getColumnFileHeaderInfo(string filePath, uint64_t &rows, int8_t &containNull, int16_t &itemLen) {
        int ret = 0;
        IfstreamSPtr colFile = make_shared< ifstream >( filePath );
        if( colFile->is_open() )
        {
            ret = getHeaderInfo(colFile, rows, containNull, itemLen);
            colFile->close();
        } else {
            LOG(ERROR) << "column data file not opened: " << filePath;
            ret = -1;
        }
        return ret;
    }

    int AriesScanNode::getHeaderInfo(IfstreamSPtr colFile, uint64_t &rows, int8_t &containNull, int16_t &itemLen) {
        int ret = 0;
        // colFile->seekg( 0, ios::end );
        // int64_t dataLen = colFile->tellg();
        // if (dataLen < ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE ) {
        //     ret = -2;
        // } else {
        //     colFile->seekg( 0, ios::beg );
        //     BlockFileHeader info;
        //     colFile->read((char *) &info, sizeof(BlockFileHeader));
        //     if (info.version != 1) {
        //         LOG(ERROR) << "column data file with wrong version";
        //         ret = -3;
        //     } else {
        //         rows = info.rows;
        //         containNull = info.containNull;
        //         itemLen = info.itemLen;
        //     }
        // }
        return ret;
    }

    bool AriesScanNode::Open()
    {
        ARIES_ASSERT( m_physicalTable, "m_physicalTable is nullptr");
        m_files.clear();
        m_columns.clear();
        m_outputColumnTypes.clear();
        m_readRowCount = 0;
        m_totalRowNum = 0;
        bool result = true;
        RelationStructurePointer relation = m_physicalTable->GetRelationStructure();
        ARIES_ASSERT( relation, "relation is nullptr");
        auto count = relation->GetColumnCount();
        string tableName = relation->GetName();
#ifdef USE_DATA_CACHE
        m_tableName = tableName;
#endif
        auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( m_dbName );

        std::vector<int> output_column_ids;
        if ( m_outputColumnIds.empty() )
        {
            output_column_ids.emplace_back( 1 );
        }
        else 
        {
            output_column_ids.insert( output_column_ids.begin(), m_outputColumnIds.begin(), m_outputColumnIds.end() );
        }

        int outputColumnId = 0;
        for( int id : output_column_ids )
        {
            // id is 1 based and so on
            --id;
            DLOG_IF( ERROR, id >= count || id < 0 ) << "id: " + to_string(id) + ", count: " + to_string(count);
            ++ outputColumnId;
            ARIES_ASSERT( id >= 0 && id < count, "id: " + to_string(id) + ", count: " + to_string(count));

            m_columns[outputColumnId] = std::make_shared<AriesColumn>();

            string filePath = database->GetColumnLocationString_ByIndex( tableName, id );
            auto columnEntry = database->GetTableByName( tableName )->GetColumnById(id + 1);
            ColumnStructurePointer column = relation->GetColumn( id );
            string name = column->GetName();
            LOG(INFO) << "id = " << id + 1 << ", name = " << name << ", filePath = " << filePath << endl;
            ColumnValueType valueType = column->GetValueType();
            int length = 1;
            if( valueType == ColumnValueType::TEXT || valueType == ColumnValueType::BINARY || valueType == ColumnValueType::VARBINARY )
            {
                 length = column->GetLength();
            }
            IfstreamSPtr colFile = make_shared< ifstream >( filePath );
            if( colFile->is_open() )
            {
                colFile->seekg( 0, ios::end );
                uint64_t dataLen = ( uint64_t )colFile->tellg() - ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE;

                uint64_t rows = 0;
                int8_t containNull = false;
                int16_t itemLen = 0;
                // check if header info is valid
                int res = getHeaderInfo(colFile, rows, containNull, itemLen);
                AriesColumnType dataType = CovertToAriesColumnType( valueType, length, column->IsNullable(), true,
                                                                    columnEntry->numeric_precision, columnEntry->numeric_scale );
                size_t dataTypeSize = dataType.GetDataTypeSize();
                if (res == 0) {
                    if (rows > 0 && itemLen > 0) {
                        if (dataType.HasNull != (bool) containNull) {
                            result = false;
                        } else if (std::size_t(itemLen) != dataTypeSize) {
                            switch( dataType.DataType.ValueType )
                            {
                                case AriesValueType::CHAR:
                                    //set actual item length
                                    dataType.DataType.Length = ( int )itemLen - ( int )containNull;
                                    //update dataType size
                                    dataTypeSize = dataType.GetDataTypeSize();
                                    break;
                                default:
                                    result = false;
                                    break;
                            }
                        }
                        if (result && (dataLen / dataTypeSize != rows)) {
                            result = false;
                        }
                    }
                } else if (res < 0) {
                    ARIES_ASSERT(0, "ERROR: column file: " + filePath + " is error, error code: " + std::to_string( res ) + ", please check it");
                    result = false;
                }

                //简单验证文件的有效性
                if( result && dataLen > 0 && dataLen % dataTypeSize == 0 )
                {
                    m_totalRowNum = dataLen / dataTypeSize;
                    //跳过4K文件头
                    colFile->seekg( ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, ios::beg );
#ifdef USE_DATA_CACHE
                    m_files.push_back(
                    { colFile, dataType, name, -1 } );
#else
                    m_files.push_back(
                    { colFile, dataType } );
#endif
                    m_outputColumnTypes.push_back(dataType);
                }
                if ( !result )
                {
                    colFile->close();
                }
            }
            else
            {
                string msg("Can't find file for table ");
                msg.append( database->GetName() ).append( "." ).append( tableName );
                ARIES_EXCEPTION_SIMPLE( ER_FILE_NOT_FOUND, msg.data() );
                result = false;
            }
            if (result == false)
            {
                break;
            }
        }
        if ( m_outputColumnIds.empty() )
        {
            m_outputColumnTypes.clear();
        }
        if( !result )
        {
            m_files.clear();
            m_outputColumnTypes.clear();
        }
        return result;
    }

    AriesOpResult AriesScanNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesOpResult result
        { AriesOpNodeStatus::END, make_unique< AriesTableBlock >() };
        int id = 0;
        const int64_t countOfBlock = ARIES_DATA_BLOCK_ROW_SIZE;
        int64_t maxReadCount = std::min(countOfBlock, m_totalRowNum - m_readRowCount);
        if ( maxReadCount == 0 )
        {
            // no more data need read
            result.TableBlock = AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
            return result;
        }
        int64_t offset = 0;

        for ( auto &fileInfo : m_files )
        {
            if ( IsCurrentThdKilled() )
            {
                LOG(INFO) << "thread killed while scan file";
                SendKillMessage();
            }

            ++id;
            if ( id == 1 )
            {
                offset = ( fileInfo.blockId + 1 ) * countOfBlock;
            }
            auto &column = m_columns[ id ];
            try
            {
                auto dataBlock = ReadNextBlock( fileInfo, countOfBlock, maxReadCount );
                if( dataBlock)
                {
                    column->AddDataBuffer(dataBlock);
                }
            } 
            catch (...)
            {
                //读取文件出错
                LOG(ERROR) << "cannot read data from file";
                result =
                { AriesOpNodeStatus::ERROR, nullptr };
                break;
            }
        }

        if( result.Status != AriesOpNodeStatus::ERROR ) 
        {
            m_readRowCount += maxReadCount;
            result.Status = m_totalRowNum  == m_readRowCount ? AriesOpNodeStatus::END : AriesOpNodeStatus::CONTINUE;
            if ( IsOutputColumnsEmpty() )
            {
                result.TableBlock = AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
                result.TableBlock->SetRowCount( maxReadCount );
            }
            else
            {
                result.TableBlock->AddColumn( m_columns, offset, maxReadCount );
            }
        }
#ifdef ARIES_PROFILE
        LOG(INFO) << "---------------------AriesScanNode::GetNext() time cost is:" << t.end() << endl;
#endif
        CacheNodeData( result.TableBlock );

        return result;
    }

    void AriesScanNode::Close()
    {
        m_files.clear();
        m_readRowCount = 0;
        m_totalRowNum = 0;
    }

    AriesDataBufferSPtr AriesScanNode::ReadNextBlock( FileInfo& fileInfo, int64_t blockSize, int64_t maxReadCount )
    {
        const auto& dataType = fileInfo.DataType;
        AriesDataBufferSPtr data_buffer = make_shared< AriesDataBuffer >( dataType );
        if( maxReadCount > 0 )
        {
#ifdef USE_DATA_CACHE
            //start id: 0
            ++fileInfo.blockId;
            //use blockSize to check if data is valid
            AriesDataCache::GetInstance().setBlockSize(m_dbName, m_tableName, fileInfo.colName, blockSize);
            AriesDataBufferSPtr buffer = AriesDataCache::GetInstance().getCacheData(m_dbName, m_tableName, fileInfo.colName, fileInfo.blockId);
            if (buffer != nullptr) {
                return buffer;
            } else {
                std::lock_guard<std::mutex> guard(*(AriesDataCache::GetInstance().getCacheMutex(m_dbName, m_tableName, fileInfo.colName, fileInfo.blockId)));
                //check if cached by other thread/client
                AriesDataCache::GetInstance().setBlockSize(m_dbName, m_tableName, fileInfo.colName, blockSize);
                buffer = AriesDataCache::GetInstance().getCacheData(m_dbName, m_tableName, fileInfo.colName, fileInfo.blockId);
                if (buffer != nullptr) {
                    return buffer;
                }
#endif
                char *buf = reinterpret_cast< char * >( data_buffer->AllocArray(maxReadCount));
#ifdef USE_DATA_CACHE
                //no buffer left, remove oldest used cache
                while (buf == nullptr && AriesDataCache::GetInstance().removeOldestUnusedOne()) {
                    buf = reinterpret_cast< char * >( data_buffer->AllocArray(maxReadCount));
                }
#endif
                if (buf) {
                    data_buffer->PrefetchToCpu();
#ifdef USE_DATA_CACHE
                    //跳过4K文件头 和 已读数据
                    fileInfo.FileStream->seekg(ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE + data_buffer->GetItemSizeInBytes() * blockSize * fileInfo.blockId, ios::beg);
#endif
                    fileInfo.FileStream->read(buf, data_buffer->GetTotalBytes());
#ifdef USE_DATA_CACHE
                    AriesDataCache::GetInstance().cacheData(m_dbName, m_tableName, fileInfo.colName, fileInfo.blockId, data_buffer);
#endif
                } else {
                    data_buffer.reset();
                }
#ifdef USE_DATA_CACHE
            }
#endif
        }
        data_buffer->SetFromCache( true );
        return data_buffer;
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace QueryEngine */
