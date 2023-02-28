#include <future>

#include "AriesDictManager.h"
#include "CudaAcc/AriesSqlOperator_filter.h"
#include "CudaAcc/AriesSqlOperator_materialize.h"
#include "AriesDict.h"
#include "AriesEngine/transaction/AriesInitialTable.h"
#include "AriesEngine/AriesUtil.h"
#include "utils/thread.h"
#include "utils/utils.h"

extern bool STRICT_MODE;

namespace aries {

AriesDict::AriesDict( int64_t id,
                      std::string name,
                      schema::ColumnType indexDataType,
                      int64_t refCount,
                      bool nullable,
                      int32_t char_max_len )
: m_id( id ),
  m_name( name ),
  m_indexDataType( indexDataType ),
  m_refCount( refCount ),
  m_nullable( nullable ),
  m_itemSchemaSize( char_max_len ),
  m_itemStoreSize( m_itemSchemaSize + nullable ),
  m_dictItemCnt( 0 ),
  m_dictData ( nullptr )
{
    m_tmpBuff = new char[ m_itemStoreSize ];
    switch ( m_indexDataType )
    {
        case schema::ColumnType::TINY_INT:
            m_dictCapacity = INT8_MAX;
            m_dictIndexItemSize = 1;
            break;

        case schema::ColumnType::SMALL_INT:
            m_dictCapacity = INT16_MAX;
            m_dictIndexItemSize = 2;
            break;

        case schema::ColumnType::INT:
            // m_dictCapacity = INT32_MAX;
            m_dictCapacity = 1000000;
            m_dictIndexItemSize = 4;
            break;

        default:
            aries::ThrowNotSupportedException("dict encoding type: " + get_name_of_value_type( m_indexDataType ) );
            break;
    }
    m_dictIndexItemSize += nullable;
}

aries_acc::AriesDataBufferSPtr AriesDict::ReadData()
{
    if ( m_dictDataBuffer )
        return m_dictDataBuffer;

    AriesColumnType dataType  = CovertToAriesColumnType( ColumnType::TEXT,
                                                         GetSchemaSize(),
                                                         IsNullable() );

    m_dictDataBuffer = make_shared< aries_acc::AriesDataBuffer >( dataType, getDictCapacity(), true );
    m_dictDataBuffer->PrefetchToCpu();

    string dictFilePath = AriesDictManager::GetInstance().GetDictFilePath( GetId(), GetName() );

    IfstreamSPtr dictIfs = make_shared< ifstream >( dictFilePath );
    if ( dictIfs->is_open() )
    {
        BlockFileHeader headerInfo;
        ValidateBlockFile( dictIfs, dictFilePath, headerInfo, dataType );

        dictIfs->seekg( ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, ios::beg );

        auto dictItemCount = headerInfo.rows;
        m_dictDataBuffer->SetItemCount( dictItemCount );
        if ( dictItemCount > 0 )
        {
            char *buf = reinterpret_cast< char * >( m_dictDataBuffer->GetData() );
            dictIfs->read( buf, m_dictDataBuffer->GetTotalBytes() );

            m_dictDataBuffer->MemAdvise( cudaMemAdviseSetReadMostly, 0 );
        }
    }
    else
    {
        string msg = "open dict file " + dictFilePath + " failed";
        ARIES_EXCEPTION_SIMPLE( ER_FILE_CORRUPT, msg.data() );
    }

    m_dictData = ( char* )m_dictDataBuffer->GetData();
    m_dictItemCnt = m_dictDataBuffer->GetItemCount();
    buildDictHash();

    return m_dictDataBuffer;
}

// item is raw char string, not including nullable byte
bool AriesDict::addDict( const char* item,
                         size_t size,
                         size_t itemIndex,
                         int32_t* index,
                         int& errorCode,
                         string& errorMsg )
{
    errorCode = 0;
    /*
    insert null value into not null column
    mysql> insert into tchar values(null);
    ERROR 1048 (23000): Column 'f' cannot be null

    mysql> select * from tchar_nullable;
    +------+
    | f    |
    +------+
    | a    |
    | NULL |
    +------+
    2 rows in set (0.00 sec)

    mysql> insert into tchar select * from tchar_nullable;
    ERROR 1048 (23000): Column 'f' cannot be null
    */
    if ( 0 == size && !m_nullable )
    {
        errorCode = ER_BAD_NULL_ERROR;
        errorMsg = format_mysql_err_msg( errorCode, m_name.data() );
        return false;
    }

    size_t actualDataSize = size;
    if ( actualDataSize > m_itemSchemaSize )
    {
        if ( STRICT_MODE )
        {
            errorCode = FormatDataTooLongError( m_name, itemIndex, errorMsg );
            return false;
        }
        else
        {
            actualDataSize = m_itemSchemaSize;
            string tmpErrorMsg;
            FormatDataTruncError( m_name, itemIndex, tmpErrorMsg );
            LOG(WARNING) << "Convert data warning: " << tmpErrorMsg;
        }
    }

    size_t buffPos = 0;
    lock_guard< mutex > lock( m_dictLock );
    if ( m_nullable )
    {
        if ( size > 0 )
            *m_tmpBuff = 1;
        else // null value
            memset( m_tmpBuff, 0, m_itemStoreSize );

        buffPos = 1;
    }
    if ( size > 0 )
    {
        memcpy( m_tmpBuff + buffPos, item, actualDataSize );
        if ( m_itemSchemaSize > actualDataSize )
            memset( m_tmpBuff + buffPos + actualDataSize, 0, m_itemSchemaSize - actualDataSize );
    }

    string dictItem( m_tmpBuff, m_itemStoreSize );

    auto it = m_dictHash.find( dictItem );

    if ( m_dictHash.end() != it )
    {
        if ( index )
            *index = it->second;
        return false;
    }
    else
    {
        auto dictItemCnt = m_dictItemCnt.load();
        if ( dictItemCnt >= m_dictCapacity )
        {
            errorCode = ER_TOO_MANY_DICT_ITEMS;
            errorMsg = format_mysql_err_msg( errorCode );
            return false;
        }
        char* dictItemBuffPtr = m_dictData + dictItemCnt * m_itemStoreSize;
        if ( size > 0 )
            memcpy( dictItemBuffPtr, dictItem.data(), m_itemStoreSize );

        m_dictHash[ dictItem ] = dictItemCnt;
        if ( m_dictDataBuffer )
            m_dictDataBuffer->SetItemCount( dictItemCnt + 1 );
        if ( index )
            *index = dictItemCnt;

        m_dictItemCnt++;
        return true;
    }
}

void AriesDict::buildDictHash()
{
    for ( size_t i = 0; i < m_dictItemCnt; ++i )
    {
        string dictItem( m_dictData + i * m_itemStoreSize, m_itemStoreSize );
        assert( m_dictHash.end() == m_dictHash.find( dictItem ) );
        m_dictHash[ dictItem ] = i;
    }
}

static bool AddDataItem( const string& colName,
                         const aries_acc::AriesDataBufferSPtr& column,
                         size_t itemIndex,
                         const AriesDictSPtr& dict,
                         int32_t& newDictIndex,
                         const aries_acc::AriesDataBufferSPtr& result,
                         int& errorCode,
                         string& errorMsg )
{
    bool newDictItem;
    auto itemSize = column->GetItemSizeInBytes();
    auto pData = ( const char* )column->GetItemDataAt( itemIndex );
    int8_t* pIndex = result->GetItemDataAt( itemIndex );
    if ( dict->IsNullable() )
    {
        if ( column->isNullableColumn() )
        {
            if ( 0 == pData[ 0 ] ) // NULL
            {
                newDictItem = dict->addDict( pData, 0,
                                             itemIndex, &newDictIndex,
                                             errorCode, errorMsg );
                *pIndex = 0;
            }
            else
            {
                auto actualDataSize = strlen( pData + 1 );
                actualDataSize = std::min( actualDataSize, itemSize - 1 );
                newDictItem = dict->addDict( pData + 1, actualDataSize,
                                             itemIndex, &newDictIndex,
                                             errorCode, errorMsg );
                *pIndex = 1;
            }
        }
        else
        {
            auto actualDataSize = strlen( pData );
            actualDataSize = std::min( actualDataSize, itemSize );
            newDictItem = dict->addDict( pData, actualDataSize,
                                         itemIndex, &newDictIndex,
                                         errorCode, errorMsg );
            *pIndex = 1;
        }

        ++pIndex;
    }
    else
    {
        if ( column->isNullableColumn() )
        {
            if ( 0 == pData[ 0 ] ) // NULL
            {
                errorCode = ER_BAD_NULL_ERROR;
                errorMsg = format_mysql_err_msg( errorCode, colName.data() );
                return false;
            }
            else
            {
                auto actualDataSize = strlen( pData + 1 );
                actualDataSize = std::min( actualDataSize, itemSize - 1 );
                newDictItem = dict->addDict( pData + 1, actualDataSize,
                                             itemIndex, &newDictIndex,
                                             errorCode, errorMsg );
            }
        }
        else
        {
            auto actualDataSize = strlen( pData );
            actualDataSize = std::min( actualDataSize, itemSize );
            newDictItem = dict->addDict( pData, actualDataSize,
                                         itemIndex, &newDictIndex,
                                         errorCode, errorMsg );
        }
    }

    if ( 0 != errorCode )
        return false;

    switch ( result->GetDataType().DataType.ValueType )
    {
        case AriesValueType::INT8:
        {
            int8_t tmpIndex = newDictIndex;
            memcpy( pIndex, &tmpIndex, sizeof( tmpIndex ) );
            break;
        }
        case AriesValueType::INT16:
        {
            int16_t tmpIndex = newDictIndex;
            memcpy( pIndex, &tmpIndex, sizeof( tmpIndex ) );
            break;
        }
        case AriesValueType::INT32:
        {
            int32_t tmpIndex = newDictIndex;
            memcpy( pIndex, &tmpIndex, sizeof( tmpIndex ) );
            break;
        }
        default:
        {
            aries::ThrowNotSupportedException("dict encoding type: " + std::to_string( (int)result->GetDataType().DataType.ValueType ) );
            break;
        }
    }
    return newDictItem;
}

ConvertDictColumnResultSPtr
ConvertDictEncodedColumnWorker( const string& colName,
                                const aries_acc::AriesDataBufferSPtr& column,
                                size_t startRowIndex,
                                size_t rowCount,
                                const AriesDictSPtr& dict,
                                const aries_acc::AriesDataBufferSPtr& result, // 转换为字典后，原始的每一行数据，对应的字典中的条目的索引
                                const aries_acc::AriesManagedInt8ArraySPtr newDictFlags, // 原始数据中的每一行是否新增了字典项
                                const aries_acc::AriesManagedIndicesArraySPtr& newDictIndices ) // 原始数据中的每一行，如果新增了字典项，对应的字典中的条目的索引
{
    auto convertResult = make_shared< ConvertDictColumnResult >();

    auto pNewDictItemFlags = newDictFlags->GetData();
    auto pNewDictIndices   = newDictIndices->GetData();

    for ( size_t i = 0; i < rowCount; ++i )
    {
        auto globalIndex = startRowIndex + i;
        int32_t newDictIndex;
        bool newDictItem = AddDataItem( colName, column,
                                        globalIndex, dict, newDictIndex, result,
                                        convertResult->errorCode,
                                        convertResult->errorMsg );
        if ( 0 != convertResult->errorCode )
            break;
        if ( newDictItem )
        {
            pNewDictItemFlags[ globalIndex ] = 1;
            pNewDictIndices[ globalIndex ] = newDictIndex;
        }
    }

    return convertResult;
}

aries_acc::AriesDataBufferSPtr
ConvertDictEncodedColumn( const string& colName,
                          const aries_acc::AriesDataBufferSPtr& column,
                          AriesDictSPtr& dict,
                          const AriesColumnType& indexDataType,
                          aries_acc::AriesManagedIndicesArraySPtr& newDictIndices ) // OUT, 新增字典条目的索引
{
    auto rowCount = column->GetItemCount();

    auto result = make_shared< aries_acc::AriesDataBuffer >( indexDataType, rowCount );
    auto tmpDictIndices = make_shared< aries_acc::AriesManagedIndicesArray >( rowCount );
    aries_acc::AriesManagedInt8ArraySPtr newDictFlags = make_shared< aries_acc::AriesManagedInt8Array >( rowCount, true );

    if ( rowCount <= 100 )
    {
        auto convertResult = ConvertDictEncodedColumnWorker(
                                colName,
                                column,
                                0,
                                rowCount,
                                dict,
                                result,
                                newDictFlags,
                                tmpDictIndices );
        if ( 0 != convertResult->errorCode )
        {
            ARIES_EXCEPTION_SIMPLE( convertResult->errorCode,
                                    convertResult->errorMsg );
            return nullptr;
        }

    }
    else
    {
        vector<size_t> threadsJobCnt;
        vector<size_t> threadsJobStartIdx;
        size_t threadCnt = getConcurrency( rowCount, threadsJobCnt, threadsJobStartIdx );
        vector< future< ConvertDictColumnResultSPtr > > workThreads;
        for( size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx )
        {
            workThreads.push_back( std::async( std::launch::async, [=] {
                return ConvertDictEncodedColumnWorker( colName,
                                                       column,
                                                       threadsJobStartIdx[ threadIdx ],
                                                       threadsJobCnt[ threadIdx ],
                                                       dict,
                                                       result,
                                                       newDictFlags,
                                                       tmpDictIndices );
            } ) );
        }
        for( auto& thrd : workThreads )
            thrd.wait();

        for( auto& thrd : workThreads )
        {
            auto convertResult = thrd.get();
            if ( 0 != convertResult->errorCode )
            {
                ARIES_EXCEPTION_SIMPLE( convertResult->errorCode,
                                        convertResult->errorMsg );
                return nullptr;
            }
        }
    }
    
    auto newDictDataRows = aries_acc::FilterFlags( newDictFlags );
    newDictFlags = nullptr;

    vector< AriesManagedIndicesArraySPtr > input;
    input.push_back( tmpDictIndices );
    newDictIndices = aries_acc::ShuffleIndices( input, newDictDataRows )[ 0 ];
    newDictDataRows = nullptr;

    newDictIndices->PrefetchToCpu();

    return result;
}

} // namespace aries