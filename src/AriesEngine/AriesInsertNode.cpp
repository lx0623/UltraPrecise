#include <glog/logging.h>
#include "AriesInsertNode.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "utils/thread.h"
#include <future>
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"
#include "Compression/dict/AriesDictManager.h"

using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

AriesInsertNode::AriesInsertNode( const AriesTransactionPtr& transaction,
                                  const std::string& dbName,
                                  const std::string& tableName )
: m_transaction( transaction ), m_dbName( dbName ), m_tableName( tableName )
{
    m_dbEntry =
       schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
    m_tableEntry = m_dbEntry->GetTableByName( tableName );
}

AriesInsertNode::~AriesInsertNode()
{
}

bool AriesInsertNode::Open()
{
    m_mvccTable = AriesMvccTableManager::GetInstance().getMvccTable( m_dbName, m_tableName );
    return m_dataSource->Open();
}

void AriesInsertNode::Close()
{
    m_dataSource->Close();
}

void AriesInsertNode::SetColumnIds( const std::vector< int >& ids )
{
    m_columnIds.assign( ids.cbegin(), ids.cend() );
}

AriesDataBufferSPtr AriesInsertNode::GetColumnDefaultValueBuffer( int colId, size_t itemCount ) const
{
    ARIES_ASSERT( colId > 0 && ( size_t )colId <= m_tableEntry->GetColumnsCount(), "column id out of range" );
    auto colEntry = m_tableEntry->GetColumnById( ( size_t )colId );
    AriesColumnType type = CovertToAriesColumnType( colEntry->GetType(), colEntry->GetLength(), colEntry->IsAllowNull(), true,
            colEntry->numeric_precision, colEntry->numeric_scale );
    if( colEntry->IsAllowNull() )
    {
        if( !colEntry->GetDefault() )
        {
            return aries_acc::CreateDataBufferWithNull( itemCount, type );
        }
        else
        {
            string defValue = colEntry->GetConvertedDefaultValue();
            return aries_acc::CreateDataBufferWithValue( defValue, itemCount, type );
        }
    }
    else
    {
        // not null column, no default value specified when create table
        if( !colEntry->HasDefault() )
            ARIES_EXCEPTION( ER_NO_DEFAULT_FOR_FIELD, colEntry->GetName().c_str() );

        if( !colEntry->GetDefault() )
        {
            ARIES_EXCEPTION( ER_NO_DEFAULT_FOR_FIELD, colEntry->GetName().c_str() );
        }
        else
        {
            string defValue = colEntry->GetConvertedDefaultValue();
            return aries_acc::CreateDataBufferWithValue( defValue, itemCount, type );
        }
    }
}

// ConcurrentInsertResultSPtr
// AriesInsertNode::AddTuples( const TupleDataSPtr tupleData,
//                             size_t startRowIdx,
//                             size_t jobCount )
// {
//     ConcurrentInsertResultSPtr result = make_shared< ConcurrentInsertResult >();
//     for ( size_t i = 0; i < jobCount; i++ )
//     {

//         auto rowPos = m_mvccTable->AddTupleDelayXLog( m_transaction, 0, tupleData, startRowIdx + i );
//         if ( INVALID_ROWPOS == rowPos )
//         {
//             result->m_success = false;
//             break;
//         }
//         result->m_rowPoses.emplace_back( rowPos );
//     }
//     return result;
// }

AriesOpResult AriesInsertNode::GetNext()
{
    AriesOpResult result;
    int64_t inserted_count = 0;

    ARIES_ASSERT( !m_columnIds.empty(), "target column ids should not be empty" );

#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    aries::CPU_Timer tTotal;
#endif
    do
    {
        result = m_dataSource->GetNext();
#ifdef ARIES_PROFILE
        tTotal.begin();
#endif
        if ( result.Status == AriesOpNodeStatus::ERROR )
        {
            break;
        }

        const auto& table = result.TableBlock;
        if ( !table )
        {
#ifdef ARIES_PROFILE
            m_opTime += tTotal.end();
#endif
            break;
        }

        if ( table->GetRowCount() == 0 )
        {
            continue;
        }

        ARIES_ASSERT(  table->GetColumnCount() >= 0 && m_columnIds.size() >= std::size_t( table->GetColumnCount() ), "column count not match" );

        auto tuple_data = std::make_shared< TupleData >();
        // pair: ( 插入数据行索引, 新增字典条目索引 )
        auto dict_tuple_data = unordered_map< int32_t, AriesManagedIndicesArraySPtr >();
        vector< AriesDataBufferSPtr > origDictStringColumns;

        std::vector< int > ids( m_tableEntry->GetColumnsCount() );
        std::iota( ids.begin(), ids.end(), 1 );

        int index = 1;
        for ( const auto& id : m_columnIds )
        {
            auto colEntry = m_tableEntry->GetColumnById( id );
            auto colBuff = table->GetColumnBuffer( index++ );
            AriesColumnType type = CovertToAriesColumnType( colEntry->GetType(),
                                                colEntry->GetLength(),
                                                colEntry->IsAllowNull(),
                                                true,
                                                colEntry->numeric_precision,
                                                colEntry->numeric_scale );
            if ( EncodeType::DICT == colEntry->encode_type )
            {
                if ( AriesValueType::CHAR == colBuff->GetDataType().DataType.ValueType ||
                     AriesValueType::BOOL == colBuff->GetDataType().DataType.ValueType ) // null constant
                {
                    auto initialTable = AriesInitialTableManager::GetInstance().getTable( m_dbName, m_tableName );
                    auto columnDict = initialTable->GetColumnDict( id - 1 );
                    aries_acc::AriesManagedIndicesArraySPtr newDictIndices;
                    auto indiceColBuff = ConvertDictEncodedColumn( colEntry->GetName(),
                                                                   colBuff,
                                                                   columnDict,
                                                                   colEntry->GetDictIndexColumnType(),
                                                                   newDictIndices );
                    tuple_data->data[ id ] = indiceColBuff;
                    dict_tuple_data[ id ] = newDictIndices;
                }
                else // insert into T1 select dict_index( a ) from T2
                {
                    // T1 and t2 should share a same dict, no new dict items will be added
                    tuple_data->data[ id ] = colBuff;
                }
            }
            else
            {
                tuple_data->data[ id ] = colBuff;
                if ( !( colBuff->GetDataType() == type ) )
                {
                    auto converted = std::make_shared< AriesDataBuffer >( type, colBuff->GetItemCount() );
                    if ( !TransferColumnData( colEntry->GetName(), *converted, *colBuff ) )
                    {
#ifndef NDEBUG
                        std::cout << "Not match: " << id << ", left: "
                                << static_cast< int >( colBuff->GetDataType().DataType.ValueType )
                                << ", right: " << static_cast< int >( type.DataType.ValueType )
                                << std::endl;
#endif
                        ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, "convert to Column (" + colEntry->GetName() + ") from " + GenerateParamType(colBuff->GetDataType()) + " to " + GenerateParamType(type));
                    }
                    else
                    {
                        tuple_data->data[ id ] = converted;
                    }
                }
            }

            for ( auto it = ids.begin(); it != ids.cend(); it++ )
            {
                if ( *it == id )
                {
                    ids.erase( it );
                    break;
                }
            }
        }

        for ( const auto& id : ids )
        {
            // TODO: dict encoded column
            tuple_data->data[ id ] = GetColumnDefaultValueBuffer( id, table->GetRowCount() );
        }

        DLOG( INFO ) << "insert into database " << m_dbName << ", table " << m_tableName
                     << ", " << table->GetRowCount() << " rows";
#ifdef ARIES_PROFILE
        t.begin();
#endif
        /*
        vector<size_t> threadsJobCnt;
        vector<size_t> threadsJobStartIdx;
        size_t rowCnt = table->GetRowCount();
        size_t threadCnt = getConcurrency( rowCnt, threadsJobCnt, threadsJobStartIdx );
        LOG(INFO) << "Insert concurrency count: " << threadCnt;

        vector< future< ConcurrentInsertResultSPtr > > workThreads;
        for( size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx )
        {
            workThreads.push_back(std::async(std::launch::async, [=] {
                return AddTuples( tuple_data,
                                  threadsJobStartIdx[ threadIdx ],
                                  threadsJobCnt[ threadIdx ] );
            }));
        }

        for( auto& thrd : workThreads )
            thrd.wait();
        for( auto& thrd : workThreads )
        {
            auto insertResult = thrd.get();
            if( !insertResult->m_success )
                ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
            if ( !m_mvccTable->BatchWriteXLogs( m_transaction, insertResult->m_rowPoses ) )
                ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        }
        */

        index = 0;
        for ( auto it = dict_tuple_data.begin(); it != dict_tuple_data.end(); ++ it )
        {
            auto colId = it->first;
            auto colEntry = m_tableEntry->GetColumnById( colId );
            auto newDictIndices = it->second;
            if ( !AriesDictManager::GetInstance().AddDictTuple( m_transaction,
                                                                colEntry->GetDict(),
                                                                newDictIndices ) )
                ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        }


        LOG( INFO ) << "Batch insert into table";
        if ( !m_mvccTable->AddTuple( m_transaction, tuple_data ) )
        {
            ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        }


#ifdef ARIES_PROFILE
        LOG( INFO ) << "Mvcc table insert time " << t.end() << "ms";
#endif
        inserted_count += table->GetRowCount();
#ifdef ARIES_PROFILE
        m_opTime += tTotal.end();
#endif
    } while ( result.Status != AriesOpNodeStatus::END );

    if ( result.Status == AriesOpNodeStatus::END )
    {
        auto table = std::make_unique< AriesTableBlock >();
        auto data = std::make_shared< AriesDataBuffer >( AriesColumnType{  { AriesValueType::INT64 }, false, false } );
        int8_t* buff = ( int8_t* ) malloc( data->GetItemSizeInBytes() );
        data->AttachBuffer( buff, 1 );
        auto column = std::make_shared< AriesColumn >();
        column->AddDataBuffer( data );
        table->AddColumn( 1, column );
        *( ( int64_t* )( data->GetData() ) ) = inserted_count;
        result.TableBlock = std::move( table );

        DLOG( INFO ) << "inserted into database " << m_dbName << ", table " << m_tableName << ", "
                     << inserted_count << " rows, " << m_opTime << "ms";

        m_rowCount += inserted_count;

        return result;
    }

    result.Status = AriesOpNodeStatus::ERROR;
    result.TableBlock = nullptr;
    return result;
}
JSON AriesInsertNode::GetProfile() const
{
    JSON stat = this->AriesOpNode::GetProfile();
    stat["type"] = "AriesInsertNode";
    return stat;
}

END_ARIES_ENGINE_NAMESPACE
