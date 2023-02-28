/*
 * AriesGroupNode.h
 * Copied from AriesEngineV2
 */

#pragma once

#include <vector>
#include "AriesCalcTreeGenerator.h"
#include "AriesOpNode.h"

using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

    #define MAX_ROW_COUNT_ONE_BLOCK (ARIES_DATA_BLOCK_ROW_SIZE * 11)

    class AriesGroupNode: public AriesOpNode
    {
    public:
        AriesGroupNode();
        ~AriesGroupNode();

        void SetSelectionExprs( const std::vector< AriesCommonExprUPtr > &sels );
        void SetGroupByExprs( const std::vector< AriesCommonExprUPtr > &groups );
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
        virtual AriesTableBlockUPtr GetEmptyTable() const override final;

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;

    private:
        struct AriesPartitialAggResult
        {
            AriesAggFunctionType Type;
            AriesDataBufferSPtr Result1;
            AriesDataBufferSPtr Result2; //当聚合函数为AVG时，作为count结果（AVG拆成SUM 和 COUNT分别计算，最后合并）
        };

        struct PartialGroupResult
        {
            map< string, AriesGroupNode::AriesPartitialAggResult > PartialResult;
            vector< AriesDataBufferSPtr > GroupByKeys;
            AriesTableBlockStats BlockStats;
        };

        class DataSourceWrapper
        {
        public:
            DataSourceWrapper()
            : IsCachedResultValid( false )
            {
            }

            void SetSourceNode( AriesOpNodeSPtr sourceNode, bool bFetchWholePartition )
            {
                assert( sourceNode );
                SourceNode = sourceNode;
                NeedFetchWholePartition = bFetchWholePartition;
            }

            pair< bool, map< int, AriesDictSPtr > > CanUsePartitionInfo( const set< int >& columnIds )
            {
                assert( SourceNode );
                assert( !IsCachedResultValid );
                bool bResult = false;
                map< int, AriesDictSPtr > dictColumns;
                IsCachedResultValid = true;
                CachedResult = SourceNode->GetNext();
                while( CachedResult.Status == AriesOpNodeStatus::CONTINUE && CachedResult.TableBlock->GetRowCount() == 0 )
                    CachedResult = SourceNode->GetNext();
                if( CachedResult.Status != AriesOpNodeStatus::ERROR )
                {
                    auto &table = CachedResult.TableBlock;
                    bResult = columnIds.find( table->GetPartitionedColumnID() ) != columnIds.end(); 
                    if( table->GetRowCount() > 0 )
                    {
                        for( auto id : table->GetAllColumnsId() )
                        {
                            if( table->GetColumnEncodeType( id ) == EncodeType::DICT )
                            {
                                AriesDictEncodedColumnSPtr dictCol;
                                if( table->IsColumnUnMaterilized( id ) )
                                {
                                    auto columnReference = table->GetUnMaterilizedColumn( id );
                                    dictCol = std::dynamic_pointer_cast<AriesDictEncodedColumn>( columnReference->GetReferredColumn() );
                                }
                                else
                                    dictCol = table->GetDictEncodedColumn( id );

                                dictColumns.insert( { id, dictCol->GetDict() } );
                            }
                        }
                    }
                }
                    
                return { bResult, dictColumns };
            }

            AriesOpResult GetNext()
            {
                assert( SourceNode );
                if( NeedFetchWholePartition )
                {
                    //has distinct keyword, we need fetch the whole partition
                    assert( IsCachedResultValid );
                    if( CachedResult.Status == AriesOpNodeStatus::END )
                        return { CachedResult.Status, std::move( CachedResult.TableBlock ) };
                    else 
                    {
                        AriesOpResult curResult = { CachedResult.Status, std::move( CachedResult.TableBlock ) };
                        IsCachedResultValid = false;
                        auto nextResult = SourceNode->GetNext();
                        while( nextResult.Status != AriesOpNodeStatus::ERROR )
                        {
                            if( nextResult.TableBlock->GetPartitionID() == curResult.TableBlock->GetPartitionID() )
                            {
                                curResult.TableBlock->AddBlock( std::move( nextResult.TableBlock ) );
                                curResult.Status = nextResult.Status;
                            }
                            else
                            {
                                CachedResult.Status = nextResult.Status;
                                CachedResult.TableBlock = std::move( nextResult.TableBlock );
                                IsCachedResultValid = true;
                                break;
                            }
                            if( nextResult.Status == AriesOpNodeStatus::END )
                                break;
                            nextResult = SourceNode->GetNext();
                        }
                        if( nextResult.Status == AriesOpNodeStatus::ERROR )
                            return nextResult;
                        else
                            return curResult;
                    }
                }
                else 
                {
                    if( IsCachedResultValid )
                    {
                        IsCachedResultValid = false;
                        return { CachedResult.Status, std::move( CachedResult.TableBlock ) };
                    }
                    else
                        return SourceNode->GetNext();
                }
            }
        private:
            AriesOpResult CachedResult;
            AriesOpNodeSPtr SourceNode;
            bool IsCachedResultValid;
            bool NeedFetchWholePartition;
        };

        map< string, AriesGroupNode::AriesPartitialAggResult > PartialGroup( const AriesTableBlockUPtr& dataBlock, int partitionId = -1 );
        void MergeWithLastResults( const map< string, AriesPartitialAggResult > &blockResult );
        void GenerateAllResults( bool bNeedReGroup );
        AriesTableBlockUPtr GenerateEmptyOutputWithColumnInfo() const;
        AriesTableBlockUPtr GenerateOutputOnlyRowCount();
        void AddPartitialResults( const AriesDataBufferSPtr &groupKeys, const map< string, AriesPartitialAggResult > &blockResult );
        AriesTableBlockUPtr GenerateResult( map< string, AriesPartitialAggResult > &aggResults );
        AriesDataBufferSPtr CombineDataBuffers( const vector< AriesDataBufferSPtr > &allKeys );
        map< string, AriesPartitialAggResult > CombineAllAggResults(
                map< string, shared_ptr< vector< AriesPartitialAggResult > > >&allAggResults );
        bool HasDistinctKeyword( const map< string, AEExprAggFunctionNode * >& params ) const;
        bool HasCountOrAvg( const map< string, AEExprAggFunctionNode * >& params ) const;
        AriesOpResult GetAll();
        AriesOpResult ReadAllData();
        AriesOpResult ReadMultiBlock();
        void AddGroupByKeys( const vector< AriesDataBufferSPtr > &allKeys, int partitionId );
        void CountRowCountInOneBlock( AriesTableBlockUPtr &tableBlock );

        // for partitioned
        AriesOpResult GetNextPartitioned();
        void MergeWithLastResultsPartitioned( int partitionId, const map< string, AriesPartitialAggResult > &blockResult );
        void GenerateAllResultsPartitioned();

    private:
        AriesCalcTreeGenerator m_calcTreeGen;
        vector< AEExprNodeUPtr > m_selExprs;
        vector< AEExprNodeUPtr > m_groupExprs;
        map< int, AriesDictSPtr > m_dictColumns;
        map< string, AEExprAggFunctionNode * > m_allParams; //对于复杂表达式而言，聚合函数结果作为外包的表达式参数参与外包表达式运算。所以复杂表达式运算分为两步进行，先聚合在计算。string key对应的是外包表达式参数名。
        map< string, AriesExprContent > m_constantParams;
        map< string, AriesPartitialAggResult > m_aggResults; // PartitialAggResult的数据和m_groupByKeyBuffer的数据需要保持一一按序对应, string key和m_allParams的 string key保持一一对应
        map< AEExprAggFunctionNode*, string > m_nodeToParamName; //对于输出表达式为单一的聚合函数的情形，建立表达式和参数名的映射。最终输出时可以直接从合并结果中查找输出值
        map< AEExprNode*, string > m_nodeToLiteralName;
        map< string, shared_ptr< vector< AriesPartitialAggResult > > > m_allAggResults;
        bool m_bReadAllData;
        bool m_bNeedCount;
        vector< AriesDataBufferSPtr > m_groupByKeys;
        int64_t m_rowCountInOneBlock;

        struct PartitionedAggResult
        {
            map< string, shared_ptr< vector< AriesPartitialAggResult > > > AggResults;
            bool NeedReGroup = false;
        };

        bool m_bCanUsePartitionInfo;
        set< int > m_groupbyColumnIds; //将group by中，收集表达式是简单column的场景．用于之后partition判断
        map< int, shared_ptr< PartitionedAggResult > > m_allAggResultsPartitioned; //所有partition的中间结果
        map< int, shared_ptr< map< string, AriesPartitialAggResult > > > m_aggResultPartitioned;
        map< int, shared_ptr< vector< AriesDataBufferSPtr > > > m_groupByKeysPartitioned;
        DataSourceWrapper m_dataSourceWrapper;
    };
    using AriesGroupNodeSPtr = shared_ptr<AriesGroupNode>;

END_ARIES_ENGINE_NAMESPACE
// namespace aries_engine
