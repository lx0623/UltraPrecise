#include "AriesDeleteNode.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE
AriesDeleteNode::AriesDeleteNode( const AriesTransactionPtr& transaction,
                                  const string& dbName,
                                  const string& tableName )
    : m_transaction( transaction ),
      m_dbName( dbName ),
      m_tableName( tableName )
{
    m_mvccTable = AriesMvccTableManager::GetInstance().getMvccTable( dbName, tableName );
}

AriesDeleteNode::~AriesDeleteNode() {}
bool AriesDeleteNode::Open()
{
    ARIES_ASSERT( m_dataSource , "m_dataSource is nullptr");
    return m_dataSource->Open();
}

void AriesDeleteNode::Close()
{
    ARIES_ASSERT( m_dataSource , "m_dataSource is nullptr");
    m_dataSource->Close();
}
void AriesDeleteNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
{
    m_dataSource->SetCuModule( modules );
}
string AriesDeleteNode::GetCudaKernelCode() const
{
    return m_dataSource->GetCudaKernelCode();
}

void AriesDeleteNode::SetColumnId4RowPos( int columnId )
{
    m_ColumnId4RowPos = columnId;
}

AriesOpResult AriesDeleteNode::GetNext()
{
    AriesOpResult result;
    int64_t deleted = 0;

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
            LOG( ERROR ) << "got error from source node";
            return { AriesOpNodeStatus::ERROR, nullptr };
        }

        if ( !result.TableBlock )
        {
            continue;
        }

        if ( result.TableBlock->IsColumnUnMaterilized( m_ColumnId4RowPos ) )
        {
            auto buffer_of_rowpos = result.TableBlock->GetColumnBuffer( m_ColumnId4RowPos );
            LOG( INFO ) << "Deleting from datababse " << m_dbName << ", table " << m_tableName 
                        << ", " << buffer_of_rowpos->GetItemCount() << " rows"; 
            #ifdef ARIES_PROFILE
            t.begin();
            #endif
            for ( std::size_t i = 0; i < buffer_of_rowpos->GetItemCount(); i++ )
            {
                auto pos = *reinterpret_cast< RowPos* >( buffer_of_rowpos->GetItemDataAt( i ) );
                if ( !internalDeleteFirstWriterWin( pos ) )
                {
                    return { AriesOpNodeStatus::ERROR, nullptr };
                }
                deleted ++;
            }

            #ifdef ARIES_PROFILE
            m_opTime += tTotal.end();
            LOG( INFO ) << "Mvcc table delete time " << t.end() << "ms";
            #endif
        }
        else
        {
            auto column = result.TableBlock->GetMaterilizedColumn( m_ColumnId4RowPos );
            auto buffers = column->GetDataBuffers();
            for ( const auto& buffer : buffers )
            {
                if ( buffer->GetItemCount() > 0 && !m_mvccTable->DeleteTuple( m_transaction, reinterpret_cast< RowPos* >( buffer->GetData() ), buffer->GetItemCount() ) )
                {
                    return { AriesOpNodeStatus::ERROR, nullptr };
                }
                deleted +=  buffer->GetItemCount();
                // for ( size_t i = 0; i < buffer->GetItemCount(); i++ )
                // {
                //     auto pos = *reinterpret_cast< RowPos* >( buffer->GetItemDataAt( i ) );
                //     if ( !internalDeleteFirstWriterWin( pos ) )
                //     {
                //         return { AriesOpNodeStatus::ERROR, nullptr };
                //     }

                //     deleted ++;
                // }
            }
        }

    } while ( result.Status != AriesOpNodeStatus::END );

    LOG( INFO ) << "deleted from database " << m_dbName << ", table " << m_tableName
                << ", " << deleted << " rows, " << m_opTime << "ms";

    auto buffer = std::make_shared< AriesDataBuffer >( AriesColumnType{ { AriesValueType::INT64 }, false, false } , 1 );
    memcpy( buffer->GetData(), &deleted, sizeof( int64_t ) );
    auto column = std::make_shared< AriesColumn >();
    column->AddDataBuffer( buffer );
    auto table = std::make_unique< AriesTableBlock >();
    table->AddColumn( 1, column );

    m_rowCount += deleted;
    return { AriesOpNodeStatus::END, std::move( table ) };
}

bool AriesDeleteNode::internalDeleteFirstWriterWin( RowPos pos )
{
    bool bSuccess = false;
    if( m_mvccTable->TryLock( pos ) )
    {
        auto oldTxMax = m_mvccTable->GetTxMax( pos );
        if( oldTxMax == INVALID_TX_ID || AriesTransManager::GetInstance().GetTxStatus( oldTxMax ) == TransactionStatus::ABORTED )
        {
            m_mvccTable->SetTxMax( pos, m_transaction->GetTxId() );
            m_mvccTable->Unlock( pos );
            if( !m_mvccTable->ModifyTuple( m_transaction, pos, nullptr, 0 ) )
                ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
            bSuccess = true;
        }
        else
            m_mvccTable->Unlock( pos );
    }
    return bSuccess;
}

JSON AriesDeleteNode::GetProfile() const
{
    JSON stat = this->AriesOpNode::GetProfile();
    stat["type"] = "AriesDeleteNode";
    return stat;
}

END_ARIES_ENGINE_NAMESPACE



