/*
 * AriesColumnNode.cpp
 *
 *  Created on: Oct 11, 2018
 *      Author: lichi
 */

#include "AriesColumnNode.h"
#include "AriesUtil.h"
#include "CudaAcc/AriesEngineException.h"
#include "utils/utils.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"
#include "utils/string_util.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesColumnNode::AriesColumnNode()
            : m_mode( -1 )
    {
        m_opName = "column";
    }

    AriesColumnNode::~AriesColumnNode()
    {
        m_rootOps.clear();
        m_outputColumnIds.clear();
    }

    void AriesColumnNode::SetColumnExprs( const vector< AriesCommonExprUPtr >& exprs )
    {
        m_rootOps.clear();
        m_outputColumnTypes.clear();
        for( const auto & expr : exprs )
        {
            m_outputColumnTypes.push_back( expr->GetValueType() );
            switch ( expr->GetType() )
            {
                case AriesExprType::INTEGER:
                case AriesExprType::FLOATING:
                case AriesExprType::DECIMAL:
                case AriesExprType::STRING:
                case AriesExprType::DATE:
                case AriesExprType::TIME:
                case AriesExprType::DATE_TIME:
                case AriesExprType::TIMESTAMP:
                case AriesExprType::YEAR:
                case AriesExprType::NULL_VALUE:
                    m_rootOps.emplace_back( nullptr );
                    m_LiteralContents.emplace_back( std::make_pair(expr->GetType(), expr->GetContent()) );
                    break;
                default:
                {
                    AEExprNodeUPtr rootOp = m_calcTreeGen.ConvertToCalcTree( expr, m_nodeId );
                    LOG(INFO) << "column Node expr:" << rootOp->ToString() << endl;
                    m_rootOps.push_back( std::move( rootOp ) );
                    AriesExpressionContent tmp = 0;
                    m_LiteralContents.emplace_back( std::make_pair( expr->GetType(), 0 ) );
                    break;
                }
            }
        }
    }

    void AriesColumnNode::SetExecutionMode( int mode )
    {
        ARIES_ASSERT( mode == 0 || mode == 1 , "mode: " + to_string(mode)); // 0 means use exprs, 1 means use columnIds
        m_mode = mode;
    }

    void AriesColumnNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void AriesColumnNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        if( m_mode == 0 )
        {
            ARIES_ASSERT( !m_rootOps.empty() , "m_rootOps is empty");
            for( const auto & expr : m_rootOps )
            {
                if (expr)
                    expr->SetCuModule( modules );
            }
        }
        if(m_dataSource)
            m_dataSource->SetCuModule( modules );
    }

    string AriesColumnNode::GetCudaKernelCode() const
    {
        string code;
        if( m_mode == 0 )
        {
            for( const auto & expr : m_rootOps )
            {
                if ( expr )
                    code += expr->GetCudaKernelCode();
            }
        }
        if(m_dataSource)
             code += m_dataSource->GetCudaKernelCode();

            return code;
    }

    bool AriesColumnNode::Open()
    {
         return !m_dataSource || m_dataSource->Open();
    }

    AriesOpResult AriesColumnNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        AriesOpResult opResult = {AriesOpNodeStatus::END, nullptr};
        if(m_leftSource)
            opResult = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
        t.begin();
#endif
        if( opResult.Status != AriesOpNodeStatus::ERROR )
        {
            ARIES_FUNC_LOG_BEGIN;

            AriesTableBlockUPtr output = make_unique< AriesTableBlock >();
            auto& dataBlock = opResult.TableBlock;

            size_t tupleNum = m_dataSource ? dataBlock->GetRowCount() : 1;
            if( tupleNum > 0 )
            {
                if( m_mode == 0 )
                {
                    int id = 0;
                    for( const auto& expr : m_rootOps )
                    {
                        ++id;

                        if ( !expr )
                        {
                            // output->AddColumn(id, nullptr);
                            continue;
                        }
                        assert(dataBlock);
                        AriesDataBufferSPtr dataBuffer;

                        opResult.TableBlock->ResetAllStats();

                        AEExprNodeResult result = expr->Process( opResult.TableBlock );

                        const auto& tableStats = opResult.TableBlock->GetStats();
                        tableStats.Print( "AriesColumnNode::GetNext, process expr" );
                        m_tableStats += tableStats;

                        if ( typeid(AriesBoolArraySPtr) == result.type() )
                        {
                            dataBuffer = aries_acc::ConvertToDataBuffer( boost::get< AriesBoolArraySPtr >(result) );
                        }
                        else
                        {
                            ARIES_ASSERT( result.type() == typeid(AriesDataBufferSPtr) , "result.type(): " + string(result.type().name()));
                            dataBuffer = boost::get< AriesDataBufferSPtr >( result );
                        }

                        auto column = std::make_shared< AriesColumn >();
                        column->AddDataBuffer( dataBuffer );
                        output->AddColumn( id, column );
                    }
                }
                else
                {
                    set< int32_t > idToKeep;

                    int id = 0;
                    std::map<int32_t, int32_t> ids_to_update;
                    for( int outputId : m_outputColumnIds )
                    {
                        idToKeep.insert( outputId );
                        ++id;
                        ids_to_update[id] = outputId;
                    }
                    dataBlock->KeepColumns(idToKeep);
                    dataBlock->UpdateColumnIds(ids_to_update);
                    output = std::move( dataBlock );
                }
                int id = 0;
                for( const auto& pair : m_LiteralContents )
                {
                    ++ id;
                    if( m_rootOps[id - 1] )
                    {
                        continue;
                    }

                    AriesDataBufferSPtr new_buffer;
                    auto exprType = pair.first;
                    auto &content = pair.second;
                    switch (exprType)
                    {
                    case AriesExprType::INTEGER:
                        new_buffer = CreateDataBufferWithValue(boost::get<int32_t>(content), tupleNum);
                        break;
                    case AriesExprType::FLOATING:
                        if ( CHECK_VARIANT_TYPE( content, float ) )
                            new_buffer = CreateDataBufferWithValue(boost::get<float>(content), tupleNum);
                        else
                            new_buffer = CreateDataBufferWithValue(boost::get<double>(content), tupleNum);
                        break;
                    case AriesExprType::DECIMAL:
                        new_buffer = CreateDataBufferWithValue(boost::get<aries_acc::Decimal>(content), tupleNum);
                        break;
                    case AriesExprType::STRING:
                        new_buffer = CreateDataBufferWithValue(boost::get<std::string>(content), tupleNum);
                        break;
                    case AriesExprType::DATE:
                        new_buffer = CreateDataBufferWithValue(boost::get<aries_acc::AriesDate>(content), tupleNum);
                        break;
                    case AriesExprType::TIME:
                        new_buffer = CreateDataBufferWithValue(boost::get<aries_acc::AriesTime>(content), tupleNum);
                        break;
                    case AriesExprType::DATE_TIME:
                        new_buffer = CreateDataBufferWithValue(boost::get<aries_acc::AriesDatetime>(content), tupleNum);
                        break;
                    case AriesExprType::TIMESTAMP:
                        new_buffer = CreateDataBufferWithValue(boost::get<aries_acc::AriesTimestamp>(content), tupleNum);
                        break;
                    case AriesExprType::YEAR:
                        new_buffer = CreateDataBufferWithValue(boost::get<aries_acc::AriesYear>(content), tupleNum);
                        break;
                    case AriesExprType::NULL_VALUE:
                        new_buffer = CreateNullValueDataBuffer(tupleNum);
                        break;
                    default:
                        ARIES_ENGINE_EXCEPTION(ER_NOT_SUPPORTED_YET, "column type " + string(content.type().name()));
                        break;
                    }

                    if ( new_buffer )
                    {
                        auto column = std::make_shared< AriesColumn >();
                        column->AddDataBuffer( new_buffer );
                        output->AddColumn( id, column );
                    }
                }
            }
            else
            {
                if ( m_mode == 1 && m_outputColumnTypes.empty() )
                {
                    m_outputColumnTypes = dataBlock->GetColumnTypes( m_outputColumnIds );
                }
                if ( IsOutputColumnsEmpty() )
                {
                    output->SetRowCount( tupleNum );
                }
                else
                {
                    output = AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
                }
            }

            opResult.TableBlock = std::move( output );

            ARIES_FUNC_LOG_END;
        }
        else
        {
            opResult.TableBlock = nullptr;
        }
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        if ( opResult.TableBlock )
            m_rowCount += opResult.TableBlock->GetRowCount();

        CacheNodeData( opResult.TableBlock );

        return opResult;
    }

    void AriesColumnNode::Close()
    {
        if(m_dataSource) m_dataSource->Close();
    }

    AriesTableBlockUPtr AriesColumnNode::GetEmptyTable() const
    {
        if ( !m_outputColumnIds.empty() )
        {
            auto types = m_dataSource->GetEmptyTable()->GetColumnTypes( m_outputColumnIds );
            return AriesTableBlock::CreateTableWithNoRows( types );
        }
        else if ( !m_outputColumnTypes.empty() )
            return AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
        else
            return std::make_unique< AriesTableBlock >();
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
