/*
 * AriesMvccScanNode.cpp
 *
 *  Created on: Mar 24, 2020
 *      Author: tengjianping
 */

#include "AriesMvccScanNode.h"
#include "transaction/AriesMvccTableManager.h"
#include "AriesUtil.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesMvccScanNode::AriesMvccScanNode( const AriesTransactionPtr& tx,
                                          const string& dbName,
                                          const string& tableName )
            : AriesOpNode(),
              m_dbName( dbName ),
              m_tableName( tableName ),
              m_tx( tx ),
              m_readRowCount( 0 ),
              m_totalSliceCount( -1 ),
              m_sliceIdx( -1 ),
              m_blockIndex( 0 )
    {
        m_opName = "scan";
        m_opParam = tableName;
    }

    AriesMvccScanNode::~AriesMvccScanNode()
    {
        m_outputColumnIds.clear();
    }

    void AriesMvccScanNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        // assert( !columnIds.empty() );
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    bool AriesMvccScanNode::Open()
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        m_outputTable = AriesMvccTableManager::GetInstance().getTable( m_tx, m_dbName, m_tableName, m_outputColumnIds, m_partitionConditions );

        m_tableStats = m_outputTable->GetStats();

        vector< int > colIds( m_outputColumnIds.size() );
        std::iota( colIds.begin(), colIds.end(), 1 );
        m_outputColumnTypes = m_outputTable->GetColumnTypes( colIds );
        m_rowCount = m_outputTable->GetRowCount();
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        m_blockIndex = 0;
        m_dataBlockPrefixSumArray = m_outputTable->GetMaterilizedColumnDataBlockSizePsumArray();
        return true;
    }

    AriesOpResult AriesMvccScanNode::GetNextRange()
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesOpResult result{ AriesOpNodeStatus::END, nullptr };
        result.TableBlock = m_outputTable->GetSubTable2( m_totalSliceCount, m_sliceIdx );
#ifdef ARIES_PROFILE
        long timeCost = t.end();
        m_opTime += timeCost;
        LOG(INFO) << "---------------------AriesMvccScanNode::GetNextRange time cost is:" << timeCost << endl;
#endif
        return result;
    }

    AriesOpResult AriesMvccScanNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

        if ( m_totalSliceCount > 0 )
        {
            return GetNextRange();
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesOpResult result
        { AriesOpNodeStatus::END, make_unique< AriesTableBlock >() };

        bool isTablePartitioned = ( m_outputTable->GetPartitionedColumnID() != -1 );

        const int64_t rowCnt = ARIES_DATA_BLOCK_ROW_SIZE;
        int64_t maxRowCount = std::min( rowCnt, m_rowCount - m_readRowCount );
        if ( maxRowCount == 0 )
        {
            // no more data to read
            result.TableBlock = AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
        }
        else
        {
#ifdef ARIES_PROFILE
            aries::CPU_Timer t2;
            t2.begin();
#endif
            if( m_spoolId > -1 )
            {
                result.TableBlock = m_outputTable->GetSubTable( 0, m_outputTable->GetRowCount(), true );
#ifdef ARIES_PROFILE
                m_tableStats.m_getSubTableTime += t2.end();
#endif
                m_readRowCount +=  m_outputTable->GetRowCount();
                result.Status = AriesOpNodeStatus::END;
                if ( result.TableBlock )
                    CacheNodeData( result.TableBlock );
            }
            else
            {
                if( isTablePartitioned )
                {
                    assert( m_blockIndex < m_outputTable->GetBlockCount() );
                    result.TableBlock = m_outputTable->GetOneBlock( m_blockIndex );
                    ++m_blockIndex;
                }
                else 
                {
                    if( !m_dataBlockPrefixSumArray.empty() )
                    {
                        assert( m_dataBlockPrefixSumArray.size() > 1 );
                        assert( m_blockIndex + 1 < m_dataBlockPrefixSumArray.size() );
                        maxRowCount = std::min( m_dataBlockPrefixSumArray[ m_blockIndex + 1 ] - m_dataBlockPrefixSumArray[ m_blockIndex ], m_rowCount - m_readRowCount );
                        ++m_blockIndex;
                    }
                    if ( maxRowCount == 0 )
                    {
                        result.TableBlock = m_outputTable->CloneWithNoContent();
                    }
                    else
                    {
                        result.TableBlock = m_outputTable->GetSubTable( m_readRowCount, maxRowCount, true );
                    }
                }
                
                
#ifdef ARIES_PROFILE
                m_tableStats.m_getSubTableTime += t2.end();
#endif
                m_readRowCount += result.TableBlock->GetRowCount();
                result.Status = m_rowCount == m_readRowCount ? AriesOpNodeStatus::END : AriesOpNodeStatus::CONTINUE;
            }
        }
#ifdef ARIES_PROFILE
        long timeCost = t.end();
        m_opTime += timeCost;
        LOG(INFO) << "---------------------AriesMvccScanNode::GetNext time cost is:" << timeCost << endl;
#endif
        return result;
    }

    void AriesMvccScanNode::ReleaseData()
    {
        m_outputTable = nullptr;
    }

    void AriesMvccScanNode::Close()
    {
        m_readRowCount = 0;
        m_outputTable = nullptr;
    }

    void AriesMvccScanNode::AddPartitionCondition( AriesCommonExprUPtr condition )
    {
        m_partitionConditions.emplace_back( std::move( condition ) );
    }

END_ARIES_ENGINE_NAMESPACE
