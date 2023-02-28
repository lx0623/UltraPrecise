/*
 * AriesGroupNode.cpp
 *
 *  Created on: Sep 25, 2018
 *      Author: lichi
 */
#include <random>
#include <future>
#include "AriesGroupNode.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"
#include "utils/utils.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"
// #include "CudaAcc/AriesDecimalAlgorithm.h"
#include "CudaAcc/AriesSqlOperator.h"
using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesGroupNode::AriesGroupNode()
            : m_bReadAllData( false ), m_bNeedCount( false ), m_rowCountInOneBlock( -1 ), m_bCanUsePartitionInfo( false )
    {
        m_opName = "group";
    }

    AriesGroupNode::~AriesGroupNode()
    {
        m_selExprs.clear();
        m_groupExprs.clear();
    }

    void AriesGroupNode::SetSelectionExprs( const vector< AriesCommonExprUPtr > &sels )
    {
        m_selExprs.clear();
        m_outputColumnTypes.clear();

        std::random_device rd;
        for( const auto &sel : sels )
        {
            m_outputColumnTypes.push_back( sel->GetValueType() );
            auto exprNode = m_calcTreeGen.ConvertToCalcTree( sel, m_nodeId );
            switch( sel->GetType() )
            {
                case AriesExprType::INTEGER:
                case AriesExprType::STRING:
                case AriesExprType::DECIMAL:
                case AriesExprType::DATE:
                case AriesExprType::DATE_TIME:
                {
                    //随机生成参数名
                    string tmpParamName = "tempConst";
                    tmpParamName += std::to_string( rd() );
                    m_constantParams[tmpParamName] = sel->GetContent();

                    //方便输出时，查找对应的column数据
                    m_nodeToLiteralName[exprNode.get()] = tmpParamName;
                    break;
                }

                default:
                    break;
            }
            m_selExprs.push_back( std::move( exprNode ) );
        }
    }

    void AriesGroupNode::SetGroupByExprs( const vector< AriesCommonExprUPtr > &groups )
    {
        m_groupExprs.clear();
        for( const auto &group : groups )
        {
            m_groupExprs.push_back( m_calcTreeGen.ConvertToCalcTree( group, m_nodeId ) );
            LOG( INFO )<< "group by expr:" << m_groupExprs.back()->ToString() << endl;
        }

        for( const auto& expr : m_groupExprs )
        {
            AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( expr.get() );
            if( node )
                m_groupbyColumnIds.insert( node->GetId() );
        }
    }

    void AriesGroupNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        ARIES_ASSERT( m_leftSource && !m_selExprs.empty(),
                "m_leftSource is nullptr: " + to_string( !!m_leftSource ) + ", m_selExprs empty: " + to_string( m_selExprs.empty() ) );
        for( auto &sel : m_selExprs )
        {
            sel->SetCuModule( modules );
        }
        for( auto &expr : m_groupExprs )
        {
            expr->SetCuModule( modules );
        }
        m_leftSource->SetCuModule( modules );
    }

    string AriesGroupNode::GetCudaKernelCode() const
    {
        ARIES_ASSERT( m_leftSource && !m_selExprs.empty(),
                "m_leftSource is nullptr: " + to_string( !!m_leftSource ) + ", m_selExprs empty: " + to_string( m_selExprs.empty() ) );
        string code;
        for( const auto &sel : m_selExprs )
        {
            code += sel->GetCudaKernelCode();
        }
        for( const auto &expr : m_groupExprs )
        {
            code += expr->GetCudaKernelCode();
        }
        return code + m_leftSource->GetCudaKernelCode();
    }

    bool AriesGroupNode::HasDistinctKeyword( const map< string, AEExprAggFunctionNode * >& params ) const
    {
        bool hasDistinct = false;
        for( const auto& param : params )
        {
            if( param.second )
            {
                hasDistinct = param.second->IsDistinct();
                if( hasDistinct )
                    break;
            }
        }
        LOG_IF( INFO, hasDistinct ) << "hasDistinct is true in GroupNode";
        return hasDistinct;
    }

    bool AriesGroupNode::HasCountOrAvg( const map< string, AEExprAggFunctionNode * >& params ) const
    {
        bool bFound = false;
        for( const auto& param : params )
        {
            if( param.second )
            {
                auto type = param.second->GetFunctionType();
                bFound = ( type == AriesAggFunctionType::AVG || type == AriesAggFunctionType::COUNT );
                if( bFound )
                    break;
            }
        }
        return bFound;
    }

    bool AriesGroupNode::Open()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        m_allParams.clear();
        m_aggResults.clear();
        m_groupByKeys.clear();
        m_allAggResults.clear();
        m_nodeToParamName.clear();

        std::random_device rd;
        int index = 0;
        for( auto &sel : m_selExprs )
        {
            ++index;
            if( AEExprAggFunctionNode *node = dynamic_cast< AEExprAggFunctionNode * >( sel.get() ) )
            {
                //表达式为单一聚合函数，聚合操作完成后，可作为直接输出使用

                //随机生成参数名，区别于复杂表达式中的聚合函数。
                string tmpParamName = "tempAgg";
                tmpParamName += std::to_string( rd() );
                m_allParams[tmpParamName] = node;

                //方便输出时，查找对应的column数据
                m_nodeToParamName[node] = tmpParamName;
                node->GetAllParams( m_allParams );
            }
            else if( AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( sel.get() ) )
            {
                //普通的column，本质上就是group by的 key
                m_allParams[std::to_string( node->GetId() )] = nullptr;
            }
            else if( AEExprSqlFunctionNode *node = dynamic_cast< AEExprSqlFunctionNode * >( sel.get() ) )
            {
                //sql函数节点
                node->GetAllParams( m_allParams );
            }
            else if( AEExprCalcNode *node = dynamic_cast< AEExprCalcNode * >( sel.get() ) )
            {
                //复杂的CALC表达式
                node->GetAllParams( m_allParams );
            }
            else if( AEExprCaseNode *node = dynamic_cast< AEExprCaseNode * >( sel.get() ) )
            {
                //复杂的CASE表达式
                node->GetAllParams( m_allParams );
            }
            else if( dynamic_cast< AEExprLiteralNode * >( sel.get() ) )
            {
                continue;
            }
            else
            {
                DLOG( ERROR ) << "GROUP data type " + GetValueTypeAsString( m_outputColumnTypes[index - 1] );
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "GROUP data type " + GetValueTypeAsString( m_outputColumnTypes[index - 1] ) );
            }
        }
        if( AriesDeviceProperty::GetInstance().IsHighMemoryDevice() )
            m_bReadAllData = true;
        else 
            m_bReadAllData = HasDistinctKeyword( m_allParams );
        m_bReadAllData = true;
        m_dataSourceWrapper.SetSourceNode( m_leftSource, m_bReadAllData );
        m_bNeedCount = HasCountOrAvg( m_allParams );
        bool bOK = m_leftSource->Open();
        if( bOK )
        {
            auto result = m_dataSourceWrapper.CanUsePartitionInfo( m_groupbyColumnIds );
            m_bCanUsePartitionInfo = result.first;
            m_dictColumns = std::move( result.second );
        }
            
        if( m_bReadAllData )
            m_bCanUsePartitionInfo = false;// use old way to handle distinct keyword
        return bOK;
    }

    void AriesGroupNode::MergeWithLastResultsPartitioned( int partitionId, const map< string, AriesPartitialAggResult > &blockResult )
    {
        shared_ptr< PartitionedAggResult > myPartitionedResult;
        auto it = m_allAggResultsPartitioned.find( partitionId );
        if( it != m_allAggResultsPartitioned.end() )
        {
            myPartitionedResult = it->second;
            myPartitionedResult->NeedReGroup = true;
        }
        else 
        {
            myPartitionedResult = std::make_shared< PartitionedAggResult >();
            m_allAggResultsPartitioned.insert( { partitionId, myPartitionedResult } );
        }

        shared_ptr< vector< AriesGroupNode::AriesPartitialAggResult > > aggResult;
        for( const auto &data : blockResult )
        {
            auto it = myPartitionedResult->AggResults.find( data.first );
            if( it != myPartitionedResult->AggResults.end() )
                aggResult = it->second;
            else
            {
                aggResult = make_shared< vector< AriesGroupNode::AriesPartitialAggResult > >();
                myPartitionedResult->AggResults.insert( { data.first, aggResult } );
            }
            aggResult->push_back( data.second );
        }
    }

    void AriesGroupNode::MergeWithLastResults( const map< string, AriesGroupNode::AriesPartitialAggResult > &blockResult )
    {
        //存储blockResult
        shared_ptr< vector< AriesGroupNode::AriesPartitialAggResult > > aggResult;
        for( const auto &data : blockResult )
        {
            auto it = m_allAggResults.find( data.first );
            if( it != m_allAggResults.end() )
                aggResult = it->second;
            else
            {
                aggResult = make_shared< vector< AriesGroupNode::AriesPartitialAggResult > >();
                m_allAggResults.insert(
                { data.first, aggResult } );
            }
            aggResult->push_back( data.second );
        }
    }

    void AriesGroupNode::GenerateAllResultsPartitioned()
    {
        m_aggResultPartitioned.clear();
        for( const auto& aggResult : m_allAggResultsPartitioned )
        {
            shared_ptr< map< string, AriesPartitialAggResult > > myAggResult = std::make_shared< map< string, AriesPartitialAggResult > >();
            m_aggResultPartitioned.insert( { aggResult.first, myAggResult } );
            if( aggResult.second->NeedReGroup )
            {
                //针对聚合结果再次做groupby和聚合
                map< string, AriesGroupNode::AriesPartitialAggResult > allResults = CombineAllAggResults( aggResult.second->AggResults );
                
                //1.计算分组信息
                AriesInt32ArraySPtr groupArray;
                AriesInt32ArraySPtr associatedArray;
                AriesInt32ArraySPtr groupFlags;
                ARIES_FUNC_LOG_BEGIN;
                auto groupByKeys = *( m_groupByKeysPartitioned[ aggResult.first ] );
                if( !groupByKeys.empty() )
                    aries_acc::GroupColumns( groupByKeys, associatedArray, groupArray, groupFlags );
                else
                {
                    size_t tupleNum = 0;
                    for( const auto &result : allResults )
                    {
                        tupleNum = result.second.Result1->GetItemCount();
                        break;
                    }
                    groupArray = make_shared< AriesInt32Array >();
                    groupArray->AllocArray( 1, true );
                    groupFlags = make_shared< AriesInt32Array >();
                    groupFlags->AllocArray( tupleNum, true );
                    associatedArray = make_shared< AriesInt32Array >( tupleNum );
                    aries_acc::InitSequenceValue( associatedArray );
                }

                //2.计算聚合函数
                vector< AriesDataBufferSPtr > simpleColumns;
                vector< string > simpleParams;
                for( const auto &result : allResults )
                {
                    AriesGroupNode::AriesPartitialAggResult partitial;
                    const auto &aggData = result.second;
                    partitial.Type = aggData.Type;
                    SumStrategy strategy = SumStrategy::NONE;
                    switch ( partitial.Type )
                    {
                        case AriesAggFunctionType::COUNT:
                        {
                            // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result1, AriesAggFunctionType::SUM,
                                                                                                // groupArray->GetItemCount() );
                            partitial.Result1 = aries_acc::AggregateColumnData(
                                aggData.Result1, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, true, SumStrategy::NONE );
                            break;
                        }
                        case AriesAggFunctionType::SUM:
                        {
                            // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result1, AriesAggFunctionType::SUM,
                                                                                                // groupArray->GetItemCount() );
                            partitial.Result1 = aries_acc::AggregateColumnData(
                                aggData.Result1, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE  );
                            break;
                        }
                        case AriesAggFunctionType::MAX:
                        {
                            partitial.Result1 = aries_acc::AggregateColumnData( aggData.Result1, AriesAggFunctionType::MAX,
                                    associatedArray, groupArray, groupFlags, false, false, strategy );
                            break;
                        }
                        case AriesAggFunctionType::MIN:
                        {
                            partitial.Result1 = aries_acc::AggregateColumnData( aggData.Result1, AriesAggFunctionType::MIN,
                                    associatedArray, groupArray, groupFlags, false, false, strategy );
                            break;
                        }
                        case AriesAggFunctionType::AVG:
                        {
                            // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result1, AriesAggFunctionType::SUM,
                                                                                                // groupArray->GetItemCount() );
                            partitial.Result1 = aries_acc::AggregateColumnData(
                                aggData.Result1, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE  );
                            // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result2, AriesAggFunctionType::SUM,
                            //                                                                     groupArray->GetItemCount() );
                            partitial.Result2 = aries_acc::AggregateColumnData(
                                aggData.Result2, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, true, SumStrategy::NONE  );
                            break;
                        }
                        case AriesAggFunctionType::ANY_VALUE:
                        {
                            partitial.Result1 = aries_acc::AggregateColumnData( aggData.Result1, AriesAggFunctionType::ANY_VALUE,
                                    associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE );
                            break;
                        }
                        case AriesAggFunctionType::NONE:
                        {
                            simpleColumns.push_back( aggData.Result1 );
                            simpleParams.push_back( result.first );
                            break;
                        }
                        default:
                            DLOG( ERROR ) << "aggregation function " + GetAriesAggFunctionTypeName( partitial.Type ) + " in GROUP expression";
                            assert( 0 );
                            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET,
                                    "aggregation function " + GetAriesAggFunctionTypeName( partitial.Type ) + " in GROUP expression" );
                            break;
                    }

                    myAggResult->insert( { result.first, partitial } );
                }
                if( !simpleColumns.empty() )
                {
                    auto columns = aries_acc::GatherGroupedColumnData( simpleColumns, associatedArray, groupArray );
                    size_t count = columns.size();
                    AriesGroupNode::AriesPartitialAggResult partitial;
                    for( std::size_t i = 0; i < count; ++i )
                    {
                        ( *myAggResult )[ simpleParams[ i ] ] = { AriesAggFunctionType::NONE, columns[ i ], nullptr };
                    }
                }
            }
            else 
            {
                for( const auto &result : aggResult.second->AggResults )
                {
                    ARIES_ASSERT( result.second->size() == 1, "result.second->size(): " + to_string( result.second->size() ) );
                    myAggResult->insert( { result.first, result.second->at( 0 ) } );
                }
            }
        }
    }

    void AriesGroupNode::GenerateAllResults( bool bNeedReGroup )
    {
        m_aggResults.clear();
        if( bNeedReGroup )
        {
            //合并所有PartitialAggResult
            map< string, AriesGroupNode::AriesPartitialAggResult > allResults = CombineAllAggResults( m_allAggResults );

            //针对聚合结果再次做groupby和聚合
            //1.计算分组信息
            AriesInt32ArraySPtr groupArray;
            AriesInt32ArraySPtr associatedArray;
            AriesInt32ArraySPtr groupFlags;
            ARIES_FUNC_LOG_BEGIN;
            if( !m_groupByKeys.empty() )
                aries_acc::GroupColumns( m_groupByKeys, associatedArray, groupArray, groupFlags );
            else
            {
                size_t tupleNum = 0;
                for( const auto &result : allResults )
                {
                    tupleNum = result.second.Result1->GetItemCount();
                    break;
                }
                groupArray = make_shared< AriesInt32Array >();
                groupArray->AllocArray( 1, true );
                groupFlags = make_shared< AriesInt32Array >();
                groupFlags->AllocArray( tupleNum, true );
                associatedArray = make_shared< AriesInt32Array >( tupleNum );
                aries_acc::InitSequenceValue( associatedArray );
            }

            //2.计算聚合函数
            vector< AriesDataBufferSPtr > simpleColumns;
            vector< string > simpleParams;
            for( const auto &result : allResults )
            {
                AriesGroupNode::AriesPartitialAggResult partitial;
                const auto &aggData = result.second;
                partitial.Type = aggData.Type;
                SumStrategy strategy = SumStrategy::NONE;
                switch ( partitial.Type )
                {
                    case AriesAggFunctionType::COUNT:
                    {
                        // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result1, AriesAggFunctionType::SUM,
                        //                                                                     groupArray->GetItemCount() );
                        partitial.Result1 = aries_acc::AggregateColumnData(
                            aggData.Result1, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, true, SumStrategy::NONE  );
                        break;
                    }
                    case AriesAggFunctionType::SUM:
                    {
                        // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result1, AriesAggFunctionType::SUM,
                        //                                                                     groupArray->GetItemCount() );
                        partitial.Result1 = aries_acc::AggregateColumnData(
                            aggData.Result1, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE  );
                        break;
                    }
                    case AriesAggFunctionType::MAX:
                    {
                        partitial.Result1 = aries_acc::AggregateColumnData( aggData.Result1, AriesAggFunctionType::MAX,
                                associatedArray, groupArray, groupFlags, false, false, strategy );
                        break;
                    }
                    case AriesAggFunctionType::MIN:
                    {
                        partitial.Result1 = aries_acc::AggregateColumnData( aggData.Result1, AriesAggFunctionType::MIN,
                                associatedArray, groupArray, groupFlags, false, false, strategy );
                        break;
                    }
                    case AriesAggFunctionType::AVG:
                    {
                        // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result1, AriesAggFunctionType::SUM,
                        //                                                                     groupArray->GetItemCount() );
                        partitial.Result1 = aries_acc::AggregateColumnData(
                            aggData.Result1, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE  );
                        // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( aggData.Result2, AriesAggFunctionType::SUM,
                        //                                                                     groupArray->GetItemCount() );
                        partitial.Result2 = aries_acc::AggregateColumnData(
                            aggData.Result2, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, true, SumStrategy::NONE  );
                        break;
                    }
                    case AriesAggFunctionType::ANY_VALUE:
                    {
                        partitial.Result1 = aries_acc::AggregateColumnData( aggData.Result1, AriesAggFunctionType::ANY_VALUE,
                                associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE );
                        break;
                    }
                    case AriesAggFunctionType::NONE:
                    {
                        simpleColumns.push_back( aggData.Result1 );
                        simpleParams.push_back( result.first );
                        break;
                    }
                    default:
                        DLOG( ERROR ) << "aggregation function " + GetAriesAggFunctionTypeName( partitial.Type ) + " in GROUP expression";
                        assert( 0 );
                        ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET,
                                "aggregation function " + GetAriesAggFunctionTypeName( partitial.Type ) + " in GROUP expression" );
                        break;
                }

                m_aggResults.insert(
                { result.first, partitial } );
            }
            if( !simpleColumns.empty() )
            {
                auto columns = aries_acc::GatherGroupedColumnData( simpleColumns, associatedArray, groupArray );
                size_t count = columns.size();
                AriesGroupNode::AriesPartitialAggResult partitial;
                for( std::size_t i = 0; i < count; ++i )
                {
                    m_aggResults[ simpleParams[ i ] ] =
                    {   AriesAggFunctionType::NONE, columns[ i ], nullptr};
                }
            }
            ARIES_FUNC_LOG_END;
        }
        else
        {
            for( const auto &result : m_allAggResults )
            {
                ARIES_ASSERT( result.second->size() == 1, "result.second->size(): " + to_string( result.second->size() ) );
                m_aggResults.insert(
                { result.first, result.second->at( 0 ) } );
            }
        }
    }

    AriesOpResult AriesGroupNode::ReadAllData()
    {
        auto result = m_dataSourceWrapper.GetNext();
        if( result.TableBlock )
            result.TableBlock->ResetAllStats();
        auto status = result.Status;
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        while( status == AriesOpNodeStatus::CONTINUE )
        {
            auto tmp = m_dataSourceWrapper.GetNext();
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

    void AriesGroupNode::CountRowCountInOneBlock( AriesTableBlockUPtr &tableBlock )
    {
        m_rowCountInOneBlock = MAX_ROW_COUNT_ONE_BLOCK;
        int oneRowSize = 0;
        for ( const auto &param : m_allParams )
        {
            if ( !param.second )
            {
                oneRowSize += tableBlock->GetColumnType( atoi( param.first.data() ) ).GetDataTypeSize();
            }
        }
        oneRowSize *= 5;  //预估：中间计算过程中间数据+原始数据的size是原始数据size的5倍
        // auto gpuMemoryLimitSize = AriesDecimalAlgorithm::GetInstance().GetGpuMemoryLimitSize();
        // if ( oneRowSize )
        // {
        //     int64_t count = ROUNDUP( gpuMemoryLimitSize / oneRowSize, ARIES_DATA_BLOCK_ROW_SIZE ) * ARIES_DATA_BLOCK_ROW_SIZE;
        //     m_rowCountInOneBlock = count > m_rowCountInOneBlock ? m_rowCountInOneBlock : count;
        // }
    }

    AriesOpResult AriesGroupNode::ReadMultiBlock()
    {
        // int readCount = 80000000 / blockSize;// hard code for test
        int i = 0;
        auto result = m_dataSourceWrapper.GetNext();
        if ( result.TableBlock )
        {
            i = result.TableBlock->GetRowCount();
            result.TableBlock->ResetAllStats();
            if ( m_rowCountInOneBlock == -1 )
            {
                CountRowCountInOneBlock( result.TableBlock );
            }
        }

        auto status = result.Status;
        while ( status == AriesOpNodeStatus::CONTINUE && i < m_rowCountInOneBlock )
        {
            auto tmp = m_dataSourceWrapper.GetNext();
            i += tmp.TableBlock->GetRowCount();
            status = tmp.Status;
            if( status == AriesOpNodeStatus::ERROR )
                break;
            ARIES_FUNC_LOG_BEGIN;
            result.TableBlock->AddBlock( std::move( tmp.TableBlock ) );
            ARIES_FUNC_LOG_END;
        }

        if( result.TableBlock )
            m_tableStats += result.TableBlock->GetStats();

        result.Status = status;
        return result;
    }

    map< string, AriesGroupNode::AriesPartitialAggResult > AriesGroupNode::PartialGroup( const AriesTableBlockUPtr& dataBlock, int partitionId )
    {
        AriesInt32ArraySPtr associatedArray;
        AriesInt32ArraySPtr groupArray;
        // 获取group by column的数据
        vector< AriesDataBufferSPtr > groupByColumns;
        ARIES_FUNC_LOG( " get groupby columns data BEGIN" );

        dataBlock->ResetAllStats();
        for( const auto &group : m_groupExprs )
        {
            AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( group.get() );
            if( node )
            {
                int column_id = node->GetId();
                auto encodeType = dataBlock->GetColumnEncodeType( column_id );
                if( encodeType == EncodeType::DICT )
                    groupByColumns.push_back( dataBlock->GetDictEncodedColumnIndiceBuffer( column_id ) );
                else
                    groupByColumns.push_back( boost::get< AriesDataBufferSPtr >( group->Process( dataBlock ) ) );
            }
            else
                groupByColumns.push_back( boost::get< AriesDataBufferSPtr >( group->Process( dataBlock ) ) );
        }

        const auto& tableStats = dataBlock->GetStats();
        tableStats.Print( "get groupby columns" );
        m_tableStats += tableStats;

        ARIES_FUNC_LOG( " get groupby columns data END" );

        AriesInt32ArraySPtr groupFlags;
        //1.计算分组信息
        int columnNum = groupByColumns.size();
        if( columnNum > 0 )
        {
            ARIES_FUNC_LOG( " group BEGIN" );
            aries_acc::GroupColumns( groupByColumns, associatedArray, groupArray, groupFlags, false );
            ARIES_FUNC_LOG( " group END" );
            //存储此次的groupby keys
            ARIES_FUNC_LOG( " gather groupby keys BEGIN" );
            AddGroupByKeys( aries_acc::GatherGroupedColumnData( groupByColumns, associatedArray, groupArray ), partitionId );
            ARIES_FUNC_LOG( " gather groupby keys END" );
        }
        else
        {
            groupArray = make_shared< AriesInt32Array >();
            groupArray->AllocArray( 1, true );
            groupFlags = make_shared< AriesInt32Array >();
            groupFlags->AllocArray( dataBlock->GetRowCount(), true );
            associatedArray = make_shared< AriesInt32Array >( dataBlock->GetRowCount() );
            aries_acc::InitSequenceValue( associatedArray );
        }

        //2.计算所有子表达式
        ARIES_FUNC_LOG( " calc sub exprs BEGIN" );
        map< string, AriesGroupNode::AriesPartitialAggResult > currentResult;
        vector< AriesDataBufferSPtr > simpleColumns;
        vector< string > simpleParams;
        AriesDataBufferSPtr itemCountInGroups;
        if( m_bNeedCount )
            itemCountInGroups = aries_acc::GetItemCountInGroups( groupArray, associatedArray->GetItemCount() );
        for( auto &param : m_allParams )
        {
            AEExprAggFunctionNode *node = param.second;
            if( node )
            {
                //聚合函数
                dataBlock->ResetAllStats();
                pair< AEExprNodeResult, AEExprNodeResult > result = node->RunKernelFunction( dataBlock, associatedArray, groupArray, groupFlags, itemCountInGroups );

                const auto& tableStats = dataBlock->GetStats();
                tableStats.Print( "PartialGroup agg" );
                m_tableStats += tableStats;

                currentResult[param.first] =
                {   node->GetFunctionType(), boost::get<AriesDataBufferSPtr>(result.first), boost::get<AriesDataBufferSPtr>(result.second)};
            }
            else
            {
                //简单column
                int columnId = std::stoi(param.first);
                dataBlock->ResetAllStats();
                if( dataBlock->GetColumnEncodeType( columnId ) == EncodeType::DICT )
                    simpleColumns.push_back( dataBlock->GetDictEncodedColumnIndiceBuffer( columnId ) );
                else 
                    simpleColumns.push_back( dataBlock->GetColumnBuffer( columnId ) );

                const auto& tableStats = dataBlock->GetStats();
                tableStats.Print( "PartialGroup get param column buffer" );
                m_tableStats += tableStats;

                simpleParams.push_back( param.first );
            }
        }
        for( auto& param : m_constantParams )
        {
            currentResult[ param.first ] =
            {   AriesAggFunctionType::NONE, ConstExprContentToDataBuffer( param.second, groupArray->GetItemCount() ), nullptr};
        }
        if( !simpleColumns.empty() )
        {
            // TODO: can be improved, simpleColumns should already exist in group keys, no need to gather again
            auto columns = aries_acc::GatherGroupedColumnData( simpleColumns, associatedArray, groupArray );
            size_t count = columns.size();
            for( std::size_t i = 0; i < count; ++i )
            {
                currentResult[ simpleParams[ i ] ] =
                {   AriesAggFunctionType::NONE, columns[ i ], nullptr};
            }
        }

        ARIES_FUNC_LOG( " calc sub exprs END" );

        return currentResult;
    }

    AriesTableBlockUPtr AriesGroupNode::GenerateOutputOnlyRowCount()
    {
        AriesTableBlockUPtr output = make_unique< AriesTableBlock >();
        if( !m_groupByKeys.empty() )
        {
            //再次做groupby
            aries_acc::AriesInt32ArraySPtr groupArray;
            aries_acc::AriesInt32ArraySPtr associatedArray;
            aries_acc::GroupColumns( m_groupByKeys, associatedArray, groupArray );
            output->SetRowCount( groupArray->GetItemCount() );
        }
        return output;
    }

    AriesTableBlockUPtr AriesGroupNode::GenerateEmptyOutputWithColumnInfo() const
    {
        AriesTableBlockUPtr output;
        //输出空值
        /**
         *
         mysql server 5.7.26
         mysql> select * from t1;
         +----------+---------------------+------+
         | time     | ts                  | year |
         +----------+---------------------+------+
         | 23:59:59 | 2019-10-16 11:11:11 | 2019 |
         | 23:59:59 | 2019-10-16 11:11:11 | 2155 |
         | 00:00:00 | 2019-10-25 08:00:00 | 2019 |
         +----------+---------------------+------+
         3 rows in set (0.00 sec)

         mysql> select sum(year) from t1 where time = "23:59:58";
         +-----------+
         | sum(year) |
         +-----------+
         |      NULL |
         +-----------+
         1 row in set (0.00 sec)

         mysql> select sum(year) from t1 where time = "23:59:58" group by time;
         Empty set (0.00 sec)

         */
        if( m_groupExprs.empty() )
        {
            size_t group_count = 1;
            output = make_unique< AriesTableBlock >();
            int id = 0;
            for( const auto &columnType : m_outputColumnTypes )
            {
                ++id;
                AriesDataBufferSPtr p = make_shared< AriesDataBuffer >( columnType );
                p->AllocArray( group_count, true );
                AriesColumnSPtr column = std::make_shared< AriesColumn >();
                column->AddDataBuffer( p );
                output->AddColumn( id, column );
            }
        }
        else
        {
            output = AriesTableBlock::CreateTableWithNoRows( m_outputColumnTypes );
        }
        return output;
    }

    AriesOpResult AriesGroupNode::GetAll()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        AriesOpResult result = ReadAllData();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        if( result.Status == AriesOpNodeStatus::END )
        {
            AriesTableBlockUPtr output = make_unique< AriesTableBlock >();
            size_t tupleNum = result.TableBlock->GetRowCount();
            if( tupleNum > 0 )
            {
                m_rowCount += tupleNum;
                auto groupResult = PartialGroup( result.TableBlock );
                output = GenerateResult( groupResult );
            }
            if( !IsOutputColumnsEmpty() )
            {
                if( output->GetRowCount() == 0 )
                {
                    output = GenerateEmptyOutputWithColumnInfo();
                }
            }
            else
            {
                output = GenerateOutputOnlyRowCount();
            }
            result.TableBlock = std::move( output );
            CacheNodeData( result.TableBlock );
        }
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        return result;
    }

    AriesOpResult AriesGroupNode::GetNextPartitioned()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

        AriesOpResult result
        { AriesOpNodeStatus::ERROR, nullptr };
        auto data = m_dataSourceWrapper.GetNext();

        while( data.Status != AriesOpNodeStatus::ERROR && !IsCurrentThdKilled() )
        {

            size_t tupleNum = data.TableBlock->GetRowCount();
            if( tupleNum > 0 )
            {
                m_rowCount += tupleNum;
                auto partialResult = PartialGroup( data.TableBlock, data.TableBlock->GetPartitionID() );
                //3.将子表达式和上一次结果合并
                MergeWithLastResultsPartitioned( data.TableBlock->GetPartitionID(), partialResult );
            }
            else
            {
                LOG( INFO )<< "read 0 rows from source in GroupNode";
            }

            if( data.Status == AriesOpNodeStatus::END )
                break;
            data = m_dataSourceWrapper.GetNext();
        }

        if( IsCurrentThdKilled() )
            SendKillMessage();

        if( data.Status == AriesOpNodeStatus::END )
        {
            AriesTableBlockUPtr output = make_unique< AriesTableBlock >();
            if( !IsOutputColumnsEmpty() )
            {
                if( !m_allAggResultsPartitioned.empty() )
                {
                    //再次聚合，获得结果
                    GenerateAllResultsPartitioned();

                    //正常结束
                    //假设ｇｒｏｕｐ后数量不多，强制输出所有结果。
                    //4.处理ＡＶＧ函数，生成聚合结果，作为表达式参数存储
                    for( const auto& aggResult : m_aggResultPartitioned )
                        output->AddBlock( GenerateResult( *( aggResult.second ) ) );
                }
                else
                {
                    output = GenerateEmptyOutputWithColumnInfo();
                }
            }
            else
            {
                output = GenerateOutputOnlyRowCount();
            }
            result = { AriesOpNodeStatus::END, std::move( output ) };
            CacheNodeData( result.TableBlock );
        }
        else if( data.Status == AriesOpNodeStatus::ERROR )
            data.TableBlock = nullptr;
        return result;
    }

    AriesOpResult AriesGroupNode::GetNext()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        if( m_bCanUsePartitionInfo )
            return GetNextPartitioned();

        if ( m_bReadAllData )
            return GetAll();
        else
        {
            AriesOpResult cachedResult = GetCachedResult();
            if ( AriesOpNodeStatus::END == cachedResult.Status )
                return cachedResult;

            AriesOpResult result
            { AriesOpNodeStatus::ERROR, nullptr };

            // 考虑如下SQL
            //        select
            //            n_regionkey,
            //            n_nationkey,
            //            case  when sum(n_regionkey) + n_nationkey < 9 then 0 when sum(n_regionkey) + n_nationkey > 9 then 1 else -1 end,
            //            (sum(n_regionkey)) / (sum(n_nationkey) + 1.3 * n_regionkey + 0.8 * max(n_nationkey) + 1)
            //        from
            //            nation
            //        group by
            //            n_regionkey,
            //            n_nationkey

            //开始循环获取block：
            int partitialResultCount = 0;
            auto data = ReadMultiBlock();
#ifdef ARIES_PROFILE
            aries::CPU_Timer t;
#endif
            while( data.Status != AriesOpNodeStatus::ERROR && !IsCurrentThdKilled() )
            {
#ifdef ARIES_PROFILE
                t.begin();
#endif
                size_t tupleNum = data.TableBlock->GetRowCount();
                if( tupleNum > 0 )
                {
                    m_rowCount += tupleNum;
                    auto partialResult = PartialGroup( data.TableBlock );
                    //3.将子表达式和上一次结果合并
                    MergeWithLastResults( partialResult );
                    ++partitialResultCount;
                }
                else
                {
                    LOG( INFO )<< "read 0 rows from source in GroupNode";
                }
#ifdef ARIES_PROFILE
                m_opTime += t.end();
#endif

                if( data.Status == AriesOpNodeStatus::END )
                    break;
                data = ReadMultiBlock();
            }

            if( IsCurrentThdKilled() )
            {
                SendKillMessage();
            }
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if( data.Status == AriesOpNodeStatus::END )
            {
                AriesTableBlockUPtr output = make_unique< AriesTableBlock >();
                if( !IsOutputColumnsEmpty() )
                {
                    if( !m_allAggResults.empty() )
                    {
                        //再次聚合，获得结果
                        GenerateAllResults( partitialResultCount > 1 );

                        //正常结束
                        //假设ｇｒｏｕｐ后数量不多，强制输出所有结果。
                        //4.处理ＡＶＧ函数，生成聚合结果，作为表达式参数存储
                        output = GenerateResult( m_aggResults );
                    }
                    else
                    {
                        output = GenerateEmptyOutputWithColumnInfo();
                    }
                }
                else
                {
                    output = GenerateOutputOnlyRowCount();
                }
                result =
                {   AriesOpNodeStatus::END, std::move(output)};
                CacheNodeData( result.TableBlock );
            }
            else if( data.Status == AriesOpNodeStatus::ERROR )
            {
                data.TableBlock = nullptr;
            }
            // data.second->PrefetchDataToCpu();
#ifdef ARIES_PROFILE
            m_opTime += t.end();
            LOG( INFO )<< "--------------AriesGroupNode::GetNext() time cost is:" << m_opTime << endl;
#endif
            return result;
        }
    }

    AriesDataBufferSPtr AriesGroupNode::CombineDataBuffers( const vector< AriesDataBufferSPtr > &allKeys )
    {
        AriesDataBufferSPtr result;
        if( !allKeys.empty() )
        {
            result = make_shared< AriesDataBuffer >( allKeys[0]->GetDataType() );
            size_t allItemCount = 0;
            for( const auto& key : allKeys )
                allItemCount += key->GetItemCount();
            result->AllocArray( allItemCount );
            result->PrefetchToGpu();
            size_t offset = 0;
            size_t totalBytes = 0;
            int8_t* data = result->GetData();
            for( const auto& key : allKeys )
            {
                totalBytes = key->GetTotalBytes();
                AriesMemAllocator::MemCopy( data + offset, key->GetData(), totalBytes );
                offset += totalBytes;
            }
        }

        return result;
    }

    void AriesGroupNode::AddGroupByKeys( const vector< AriesDataBufferSPtr > &allKeys, int partitionId )
    {
        if( allKeys.empty() )
            return;
        if( partitionId == -1 )
        {
            if( m_groupByKeys.empty() )
                m_groupByKeys = allKeys;
            else
            {
                size_t count = m_groupByKeys.size();
                ARIES_ASSERT( allKeys.size() == count, "allKeys.size(): " + to_string( allKeys.size() ) + ", count: " + to_string( count ) );
                size_t totalElementCount = m_groupByKeys[0]->GetItemCount() + allKeys[0]->GetItemCount();
                for( std::size_t i = 0; i < count; ++i )
                {
                    ARIES_ASSERT( m_groupByKeys[i]->GetDataType() == allKeys[i]->GetDataType(),
                            "i: " + to_string( i ) + ", m_groupByKeys[i]->GetDataType(): " + GetValueTypeAsString( m_groupByKeys[i]->GetDataType() )
                                    + ", allKeys[i]->GetDataType(): " + GetValueTypeAsString( allKeys[i]->GetDataType() ) );
                    AriesDataBufferSPtr data = m_groupByKeys[i]->CloneWithNoContent( totalElementCount );
                    AriesMemAllocator::MemCopy( data->GetData(), m_groupByKeys[i]->GetData(), m_groupByKeys[i]->GetTotalBytes() );
                    AriesMemAllocator::MemCopy( data->GetData() + m_groupByKeys[i]->GetTotalBytes(), allKeys[i]->GetData(),
                            allKeys[i]->GetTotalBytes() );
                    m_groupByKeys[i] = data;
                }
            }
        }
        else 
        {
            shared_ptr< vector< AriesDataBufferSPtr > > groupByKeys;
            auto it = m_groupByKeysPartitioned.find( partitionId );
            if( it == m_groupByKeysPartitioned.end() )
            {
                groupByKeys = std::make_shared< vector< AriesDataBufferSPtr > >();
                m_groupByKeysPartitioned.insert( { partitionId, groupByKeys } );
            }
            else 
                groupByKeys = it->second;
            if( groupByKeys->empty() )
                *groupByKeys = allKeys;
            else
            {
                size_t count = (*groupByKeys).size();
                ARIES_ASSERT( allKeys.size() == count, "allKeys.size(): " + to_string( allKeys.size() ) + ", count: " + to_string( count ) );
                size_t totalElementCount = (*groupByKeys)[0]->GetItemCount() + allKeys[0]->GetItemCount();
                for( std::size_t i = 0; i < count; ++i )
                {
                    ARIES_ASSERT( (*groupByKeys)[i]->GetDataType() == allKeys[i]->GetDataType(),
                            "i: " + to_string( i ) + ", (*groupByKeys)[i]->GetDataType(): " + GetValueTypeAsString( (*groupByKeys)[i]->GetDataType() )
                                    + ", allKeys[i]->GetDataType(): " + GetValueTypeAsString( allKeys[i]->GetDataType() ) );
                    AriesDataBufferSPtr data = (*groupByKeys)[i]->CloneWithNoContent( totalElementCount );
                    AriesMemAllocator::MemCopy( data->GetData(), (*groupByKeys)[i]->GetData(), (*groupByKeys)[i]->GetTotalBytes() );
                    AriesMemAllocator::MemCopy( data->GetData() + (*groupByKeys)[i]->GetTotalBytes(), allKeys[i]->GetData(),
                            allKeys[i]->GetTotalBytes() );
                    (*groupByKeys)[i] = data;
                }
            }
        }
    }
    map< string, AriesGroupNode::AriesPartitialAggResult > AriesGroupNode::CombineAllAggResults(
            map< string, shared_ptr< vector< AriesGroupNode::AriesPartitialAggResult > > >&allAggResults )
    {
        map< string, AriesGroupNode::AriesPartitialAggResult > result;
        ARIES_FUNC_LOG_BEGIN;
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        for( auto &agg : allAggResults )
        {
            AriesGroupNode::AriesPartitialAggResult aggResult;
            const auto &allData = agg.second;
            ARIES_ASSERT( !allData->empty(), "allData is empty" );
            aggResult.Type = allData->at( 0 ).Type;
            vector< AriesDataBufferSPtr > result1;
            for( const auto &data : *allData )
                result1.push_back( data.Result1 );
            aggResult.Result1 = CombineDataBuffers( result1 );
            if( aggResult.Type == AriesAggFunctionType::AVG )
            {
                vector< AriesDataBufferSPtr > result2;
                for( const auto &data : *allData )
                    result2.push_back( data.Result2 );
                aggResult.Result2 = CombineDataBuffers( result2 );
            }
            agg.second = nullptr;
            result.insert(
            { agg.first, aggResult } );
        }
#ifdef ARIES_PROFILE
        auto timeCost = t.end();
        LOG( INFO )<< "AriesGroupNode::CombineAllAggResults time: " << timeCost;
#endif
        ARIES_FUNC_LOG_END;
        return result;
    }

    AriesTableBlockUPtr AriesGroupNode::GenerateResult( map< string, AriesGroupNode::AriesPartitialAggResult > &aggResults )
    {
        ARIES_FUNC_LOG_BEGIN;
        map< string, AriesDataBufferSPtr > params;
        for( auto &result : aggResults )
        {
            const auto &aggData = result.second;
            AriesDataBufferSPtr dataBuf;

            if( aggData.Type == AriesAggFunctionType::AVG )
                dataBuf = aries_acc::DivisionInt64( aggData.Result1, aggData.Result2 );
            else
                dataBuf = aggData.Result1;
            params.insert(
            { result.first, dataBuf } );
        }

        AriesTableBlockUPtr output = make_unique< AriesTableBlock >();
        int id = 0;
        if( !params.empty() )
        {
            //5.计算各输出表达式的值
            for( const auto &sel : m_selExprs )
            {
                AriesColumnSPtr column = std::make_shared< AriesColumn >();
                ++id;
                if( AEExprAggFunctionNode *node = dynamic_cast< AEExprAggFunctionNode * >( sel.get() ) )
                {
                    column->AddDataBuffer( params.at( m_nodeToParamName.at( node ) ) );
                    output->AddColumn( id, column );
                }
                else if( AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( sel.get() ) )
                {
                    int columnId = node->GetId();
                    auto it = m_dictColumns.find( columnId );
                    if( it != m_dictColumns.end() )
                    {
                        auto newDictIndices = std::make_shared< AriesVariantIndices >();
                        newDictIndices->AddDataBuffer( params.at( std::to_string( columnId ) ) );
                        auto newDictColumn = std::make_shared< AriesDictEncodedColumn >( it->second,  newDictIndices );
                        output->AddColumn( id, newDictColumn );
                    }
                    else 
                    {
                        column->AddDataBuffer( params.at( std::to_string( columnId ) ) );
                        output->AddColumn( id, column );
                    }
                }
                else if( AEExprCalcNode *node = dynamic_cast< AEExprCalcNode * >( sel.get() ) )
                {
                    column->AddDataBuffer( boost::get< AriesDataBufferSPtr >( node->RunKernelFunction( params ) ) );
                    output->AddColumn( id, column );
                }
                else if( AEExprCaseNode *node = dynamic_cast< AEExprCaseNode * >( sel.get() ) )
                {
                    column->AddDataBuffer( boost::get< AriesDataBufferSPtr >( node->RunKernelFunction( params ) ) );
                    output->AddColumn( id, column );
                }
                else if( AEExprSqlFunctionNode *node = dynamic_cast< AEExprSqlFunctionNode * >( sel.get() ) )
                {
                    column->AddDataBuffer( boost::get< AriesDataBufferSPtr >( node->RunKernelFunction( params ) ) );
                    output->AddColumn( id, column );
                }
                else if( AEExprLiteralNode *node = dynamic_cast< AEExprLiteralNode * >( sel.get() ) )
                {
                    column->AddDataBuffer( params.at( m_nodeToLiteralName.at( node ) ) );
                    output->AddColumn( id, column );
                }
                else
                {
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET,
                            "output type of aggregation function " + GetValueTypeAsString( m_outputColumnTypes[id - 1] ) + " in GROUP expression" );
                }
            }
        }
        ARIES_FUNC_LOG_END;
        return output;
    }

    void AriesGroupNode::Close()
    {
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        m_allParams.clear();
        m_aggResults.clear();
        m_nodeToParamName.clear();
        m_groupByKeys.clear();
        m_allAggResults.clear();
        m_leftSource->Close();
    }

    AriesTableBlockUPtr AriesGroupNode::GetEmptyTable() const
    {
        if ( !IsOutputColumnsEmpty() )
        {
            return GenerateEmptyOutputWithColumnInfo();
        }
        else
        {
            return std::make_unique< AriesTableBlock >();
        }
    }

END_ARIES_ENGINE_NAMESPACE
// namespace aries_engine
