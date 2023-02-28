//
// Created by tengjp on 19-11-1.
//

#include "AriesFilterNode.h"
#include <future>
#include "utils/utils.h"
#include "CpuTimer.h"

#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE
    AriesFilterNode::AriesFilterNode()
    {
        m_opName = "filter";
        m_opParam = "";
    }
    AriesFilterNode::~AriesFilterNode()
    {
        m_outputColumnIds.clear();
        m_outputTables.clear();
    }

    void AriesFilterNode::SetCondition( const AriesCommonExprUPtr& condition )
    {
        ARIES_ASSERT( condition, "condition is nullptr" );
        m_rootOp = m_calcTreeGen.ConvertToCalcTree( condition, m_nodeId );
        LOG( INFO )<< "filterNode Condition:" << m_rootOp->ToString() << endl;
    }

    void AriesFilterNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        // assert( !columnIds.empty() );
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void AriesFilterNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        ARIES_ASSERT( m_rootOp && m_leftSource,
                "m_rootOp is nullptr: " + to_string( m_rootOp == nullptr ) + ", m_leftSource is nullptr: " + to_string( m_leftSource == nullptr ) );
        m_rootOp->SetCuModule( modules );
        m_leftSource->SetCuModule( modules );
    }

    string AriesFilterNode::GetCudaKernelCode() const
    {
        ARIES_ASSERT( m_rootOp && m_leftSource,
                "m_rootOp is nullptr: " + to_string( m_rootOp == nullptr ) + ", m_leftSource is nullptr: " + to_string( m_leftSource == nullptr ) );
        return m_rootOp->GetCudaKernelCode() + m_leftSource->GetCudaKernelCode();
    }

    bool AriesFilterNode::Open()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr: " + to_string( m_leftSource == nullptr ) );
        m_outputTables.clear();
        m_inputTables.clear();
        m_bAllDataReceived = false;
        return m_leftSource->Open();
    }

    void AriesFilterNode::Close()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr: " + to_string( m_leftSource == nullptr ) );
        m_outputTables.clear();
        m_inputTables.clear();
        m_leftSource->Close();
    }

    AriesOpResult AriesFilterNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

        AriesOpResult result;
        if( NeedCacheSpool() )
            result = ReadAllData();
        else
            result = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        long timeCost = 0;
#endif
        size_t outTupleNum = 0;

        ARIES_FUNC_LOG_BEGIN;
        while( result.Status != AriesOpNodeStatus::ERROR && !IsCurrentThdKilled() )
        {
#ifdef ARIES_PROFILE
            t.begin();
#endif
            const AriesTableBlockUPtr &tableBlock = result.TableBlock;
            int32_t tablePartitionId = tableBlock->GetPartitionID();
            int32_t partitionedColumnID = tableBlock->GetPartitionedColumnID();
            
            int tupleNum = tableBlock->GetRowCount();
            if( tupleNum > 0 )
            {
                m_rowCount += tupleNum;

                tableBlock->ResetAllStats();

                AriesBoolArraySPtr associated = boost::get< AriesBoolArraySPtr >( m_rootOp->Process( tableBlock ) );

                const auto& tableStats = tableBlock->GetStats();
                tableStats.Print( "AriesFilterNode::GetNext, process expr" );
                m_tableStats += tableStats;

                auto outIndex = FilterAssociated( associated );
                outTupleNum = outIndex->GetItemCount();

                if( outTupleNum > 0 )
                {
                    if( !m_outputColumnIds.empty() )
                    {
                        result.TableBlock = result.TableBlock->MakeTableByColumns( m_outputColumnIds, false );
                        map< int, int > idToUpdate;
                        int outputId = 0;
                        int32_t outputPartitionedColumnId = -1;
                        for ( const auto& id : m_outputColumnIds )
                        {
                            idToUpdate[ ++outputId ] = id;
                            if ( id == partitionedColumnID )
                            {
                                DLOG( INFO ) << "got partitioned column ID: " << partitionedColumnID << " as " << outputId;
                                outputPartitionedColumnId = outputId;
                            }
                        }
                        result.TableBlock->UpdateColumnIds( idToUpdate );

                        result.TableBlock->ResetAllStats();
                        result.TableBlock->UpdateIndices( outIndex );
                        result.TableBlock->SetPartitionID( tablePartitionId );
                        result.TableBlock->SetPartitionedColumnID( outputPartitionedColumnId );

                        const auto& tableStats = result.TableBlock->GetStats();
                        tableStats.Print( "AriesFilterNode::GetNext, update indice" );
                        m_tableStats += tableStats;
                    }
                    else
                    {
                        // select const, only need to output row count
                        result.TableBlock = std::make_unique< AriesTableBlock >();
                        result.TableBlock->SetRowCount( outTupleNum );
                    }
#ifdef ARIES_PROFILE
                    long tmpCost = t.end();
                    timeCost += tmpCost;
                    m_opTime += tmpCost;
#endif
                    break;
                }
            }
#ifdef ARIES_PROFILE
            long tmpCost = t.end();
            timeCost += tmpCost;
            m_opTime += tmpCost;
#endif

            if( result.Status == AriesOpNodeStatus::END )
            {
                break;
            }
            result = m_leftSource->GetNext();
        }
#ifdef ARIES_PROFILE
        t.begin();
#endif

        if( IsCurrentThdKilled() )
        {
            LOG(INFO)<< "thread was killed in FilterNode::GetNext()";
            SendKillMessage();
        }

        // empty result
        if( AriesOpNodeStatus::ERROR != result.Status )
        {
            if( outTupleNum == 0 )
            {
                if( !m_outputColumnIds.empty() )
                {
                    auto outputColumnTypes = result.TableBlock->GetColumnTypes( m_outputColumnIds );
                    result.TableBlock = result.TableBlock->CreateTableWithNoRows( outputColumnTypes );
                }
                else
                {
                    result.TableBlock = std::make_unique< AriesTableBlock >();
                }
            }
        }
        else
        {
            LOG(INFO)<< "thread was killed in FilterNode::GetNext()";
            result.TableBlock = nullptr;
        }
#ifdef ARIES_PROFILE
        long tmpCost = t.end();
        timeCost += tmpCost;
        m_opTime += tmpCost;

        LOG(INFO)<< "-----------------------AriesFilterNode::GetNext time cost is:" << timeCost << endl;
#endif
        ARIES_FUNC_LOG_END;

        CacheNodeData( result.TableBlock );

        return result;
    }

    AriesTableBlockUPtr AriesFilterNode::GetEmptyTable() const
    {
        if( !m_outputColumnIds.empty() )
        {
            auto outputColumnTypes = m_dataSource->GetEmptyTable()->GetColumnTypes( m_outputColumnIds );
            return AriesTableBlock::CreateTableWithNoRows( outputColumnTypes );
        }
        else
        {
            return std::make_unique< AriesTableBlock >();
        }
    }

    AriesOpResult AriesFilterNode::ReadAllData()
    {
        auto result = m_leftSource->GetNext();
        if( result.TableBlock )
            result.TableBlock->ResetAllStats();
        auto status = result.Status;
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        while( status == AriesOpNodeStatus::CONTINUE )
        {
            auto tmp = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            status = tmp.Status;
            if( status == AriesOpNodeStatus::ERROR )
                break;
            result.TableBlock->AddBlock( std::move( tmp.TableBlock ) );
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }

        if( result.TableBlock )
            m_tableStats += result.TableBlock->GetStats();

        result.Status = status;
        return result;
    }


END_ARIES_ENGINE_NAMESPACE
