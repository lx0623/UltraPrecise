/*
 * AriesSortNode.cpp
 *
 *  Created on: Sep 26, 2018
 *      Author: lichi
 */

#include "AriesSortNode.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "utils/utils.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesSortNode::AriesSortNode()
    {
        m_opName = "sort";
    }

    AriesSortNode::~AriesSortNode()
    {
        m_exprs.clear();
        m_orders.clear();
    }

    void AriesSortNode::SetOrderbyExprs( const vector< AriesCommonExprUPtr >& exprs )
    {
        m_exprs.clear();
        ARIES_ASSERT( !exprs.empty(), "exprs is empty" );
        AriesCalcTreeGenerator calcTreeGen;
        for( const auto& expr : exprs )
        {
            if ( !expr->IsLiteralValue() )
            {
                m_exprs.push_back( calcTreeGen.ConvertToCalcTree( expr, m_nodeId ) );
                m_outputColumnTypes.emplace_back( expr->GetValueType() );
            }
            else
            {
                LOG(INFO) << "constant expression in group-by clause was filtered here.";
            }
        }
    }

    void AriesSortNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void AriesSortNode::SetOrderbyType( const vector< AriesOrderByType >& orders )
    {
        m_orders.assign( orders.begin(), orders.end() );
    }

    bool AriesSortNode::Open()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        return m_leftSource->Open();
    }

    void AriesSortNode::UpdateOutColumns(AriesTableBlockUPtr &table)
    {
        if( !m_outputColumnIds.empty() )
        {
            set< int32_t > idToKeep;
            int id = 0;
            std::map< int32_t, int32_t > ids_to_update;
            for( int outputId : m_outputColumnIds )
            {
                idToKeep.insert( outputId );
                ++id;
                ids_to_update[id] = outputId;
            }
            table->KeepColumns( idToKeep );
            table->UpdateColumnIds( ids_to_update );
        }
    }

    AriesOpResult AriesSortNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        AriesOpResult allData;
        AriesOpResult data = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        allData.TableBlock = move( data.TableBlock );
        if ( allData.TableBlock )
            allData.TableBlock->ResetAllStats();

        while( data.Status == AriesOpNodeStatus::CONTINUE && !IsCurrentThdKilled() )
        {
            data = m_leftSource->GetNext();
#ifdef ARIES_PROFILE            
            t.begin();
#endif
            if( data.Status == AriesOpNodeStatus::ERROR )
            {
                break;
            }
            allData.TableBlock->AddBlock( move( data.TableBlock ) );
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }
        allData.Status = data.Status;

        if ( allData.TableBlock )
            m_rowCount += allData.TableBlock->GetRowCount();
#ifdef ARIES_PROFILE
        t.begin();
#endif
        if ( IsCurrentThdKilled() )
        {
            LOG(INFO) << "thread was kill in AriesSortNode::GetNext";
            SendKillMessage();
        }

        if( allData.Status == AriesOpNodeStatus::END )
        {
            auto tupleNum = allData.TableBlock->GetRowCount();
            //TODO 当m_outputColumnIds为空时表示输入所有列，需要表示不需要输入任何列的情况
            if( m_orders.size() > 0 && allData.TableBlock->GetRowCount() > 0 )
            {
                ARIES_FUNC_LOG_BEGIN;

                AriesInt32ArraySPtr associatedArray = make_shared< AriesInt32Array >( tupleNum );
                aries_acc::InitSequenceValue( associatedArray, 0 );
                size_t startIndex = m_exprs.size() - 1;
                for( int32_t i = startIndex; i >= 0; --i )
                {
                    AriesDataBufferSPtr dataBuffer;
                    if( AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( m_exprs[i].get() ) )
                    {
                        //TODO 最后排序的一列可以直接使用TableBlock中的数据,但由于radix sort不能直接将key字段进行shuffle，因此目前无法优化掉最后一列的shuffle操作
                        dataBuffer = allData.TableBlock->GetColumnBufferByIndices( node->GetId(), associatedArray );
                    }
                    else
                    {
                        auto result = m_exprs[i]->Process( allData.TableBlock );
                        if ( boost::get< AriesBoolArraySPtr >( &result ) )
                        {
                            dataBuffer = aries_acc::ConvertToDataBuffer( boost::get< AriesBoolArraySPtr >( result ) );
                        }
                        else
                        {
                            ARIES_ASSERT( typeid( AriesDataBufferSPtr ) == result.type(), "invalid result data type" );
                            dataBuffer = boost::get< AriesDataBufferSPtr >( result );
                        }

                        //非第一排序列，需要shuffle data
                        if( static_cast< std::size_t >( i ) != startIndex )
                            dataBuffer = aries_acc::ShuffleColumns(
                            { dataBuffer }, associatedArray )[0];
                    }
                    aries_acc::SortColumn( dataBuffer, m_orders[i], associatedArray );
                }
                UpdateOutColumns(allData.TableBlock);
                allData.TableBlock->UpdateIndices( associatedArray );

                ARIES_FUNC_LOG_END;
            }
            else
            {
                UpdateOutColumns(allData.TableBlock);
            }
        }
        else if ( allData.Status == AriesOpNodeStatus::ERROR )
        {
            allData.TableBlock = nullptr;
        }

        if ( allData.TableBlock )
            m_tableStats += allData.TableBlock->GetStats();
#ifdef ARIES_PROFILE
        m_opTime += t.end();
        LOG(INFO) << "---------------------AriesSortNode::GetNext() time cost is:" << m_opTime << endl;
#endif
        CacheNodeData( allData.TableBlock );

        return allData;
    }

    void AriesSortNode::Close()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        m_leftSource->Close();
    }

    AriesTableBlockUPtr AriesSortNode::GetEmptyTable() const
    {
        if ( !m_outputColumnIds.empty() )
        {
            auto columns_type = m_dataSource->GetEmptyTable()->GetColumnTypes( m_outputColumnIds );
            return AriesTableBlock::CreateTableWithNoRows( columns_type );
        }
        else
        {
            return std::make_unique< AriesTableBlock >();
        }
    }


END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
