//
// Created by david shen on 2019-07-23.
//

#include "AriesJoinNode.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"
#include "utils/utils.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"
#include "AriesDeviceProperty.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesJoinNode::AriesJoinNode()
            : m_joinType( AriesJoinType::INNER_JOIN ),
            m_isLeftDataAllRead( false ),
            m_isRightDataAllRead( false ),
            m_filterOpTime( 0 ),
            m_usedHashJoin( false ),
            m_hashJoinType( HashJoinType::None )
    {
        m_opName = "join";
    }

    AriesJoinNode::~AriesJoinNode()
    {
        // TODO Auto-generated destructor stub
    }

    void AriesJoinNode::SetCondition( AriesCommonExprUPtr equalCondition, AriesCommonExprUPtr otherCondition, AriesJoinType type )
    {
        ARIES_ASSERT( equalCondition || otherCondition , "condition and other condition are nullptr" );
        m_joinType = type;
        m_equalCondition = std::move( equalCondition );
        m_otherCondition = std::move( otherCondition );
        m_joinHelper = std::make_shared<AriesJoinNodeHelper>( this, m_equalCondition, m_otherCondition, m_joinType, m_nodeId );
    }

    void AriesJoinNode::SetSourceNode( AriesOpNodeSPtr leftSource, AriesOpNodeSPtr rightSource )
    {
        m_dataSource = leftSource;
        m_leftSource = leftSource;
        m_rightSource = rightSource;

        ARIES_ASSERT( m_joinHelper, "should call SetCondition first" );

        const auto& left_unique_keys = m_leftSource->GetUniqueColumnsId();
        const auto& right_unique_keys = m_rightSource->GetUniqueColumnsId();

        if ( !left_unique_keys.empty() || !right_unique_keys.empty() )
        {
            m_joinHelper->InitColumnsIdInConditions( m_outputColumnIds );
            // m_joinHelper->HandleUniqueKeys( left_unique_keys, right_unique_keys );
            if ( m_joinType == AriesJoinType::INNER_JOIN && m_hashJoinType != HashJoinType::None )
            {
                m_joinHelper->PrepareForHashJoin( m_hashJoinInfo );
            }
        }
    }

    void AriesJoinNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        // ARIES_ASSERT( !columnIds.empty(), "columnIds is empty" );
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void AriesJoinNode::SetJoinHint( int joinHint, bool bIntact )
    {
    }

    bool AriesJoinNode::Open()
    {
        ARIES_ASSERT( m_leftSource && m_rightSource,
                      "m_leftSource is nullptr: " + to_string( !!m_leftSource ) + ", m_rightSource is nullptr: "
                      + to_string( !!m_rightSource ) );
        m_leftSubTablesCache.clear();
        m_rightSubTablesCache.clear();
        m_outputColumnTypes.clear();
        m_isLeftSubTableCached = false;
        m_isRightSubTableCached = false;
        m_rightHandledOffset = 0;
        m_rightOneBlockSize = SUBTABLE_COUNT;
        m_isRightDataAllRead = false;

        SplitOutputColumnIds();
        return m_leftSource->Open() && m_rightSource->Open();
    }

    vector< bool > AriesJoinNode::CheckHashJoinConditionForDict( const aries_engine::AriesTableBlockUPtr& hashTable, const std::vector< int >& unique_keys, 
                                            const aries_engine::AriesTableBlockUPtr& valueTable, const std::vector< int >& value_keys )
    {
        assert( unique_keys.size() == value_keys.size() );
        vector< bool > bCanUseDict;
        for( size_t i = 0; i < unique_keys.size(); ++i )
        {
            int hashColId = unique_keys[ i ];
            int valueColId = value_keys[ i ];
            auto hashColEncodeType = hashTable->GetColumnEncodeType( hashColId );
            auto valueColEncodeType = valueTable->GetColumnEncodeType( valueColId );

            if( hashColEncodeType != valueColEncodeType )
            {
                if( hashTable->GetColumnType( hashColId ).DataType != valueTable->GetColumnType( valueColId ).DataType )
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "joined columns must have the same type" );
                bCanUseDict.push_back( false );
            }
            else if( hashColEncodeType == EncodeType::DICT )
            {
                if( hashTable->GetColumnType( hashColId ).DataType != valueTable->GetColumnType( valueColId ).DataType )
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "joined columns must have the same type" );
                AriesDictEncodedColumnSPtr hashCol;
                AriesDictEncodedColumnSPtr valueCol;
                if( hashTable->IsColumnUnMaterilized( hashColId ) )
                {
                    auto columnReference = hashTable->GetUnMaterilizedColumn( hashColId );
                    hashCol = dynamic_pointer_cast< AriesDictEncodedColumn >( columnReference->GetReferredColumn() );
                }
                else 
                    hashCol = hashTable->GetDictEncodedColumn( hashColId );

                if( valueTable->IsColumnUnMaterilized( valueColId ) )
                {
                    auto columnReference = valueTable->GetUnMaterilizedColumn( valueColId );
                    valueCol = dynamic_pointer_cast< AriesDictEncodedColumn >( columnReference->GetReferredColumn() );
                }
                else 
                    valueCol = valueTable->GetDictEncodedColumn( valueColId );
                if( hashCol->GetDict() == valueCol->GetDict() )
                    bCanUseDict.push_back( true );
                else 
                    bCanUseDict.push_back( false );
            }
            else 
                bCanUseDict.push_back( false );
        }
        return bCanUseDict;
    }

    void AriesJoinNode::SwapRightJoinToLeft()
    {
        std::swap( m_leftSource, m_rightSource );
        vector< int > newOutputColumnIds;
        for ( int id : m_outputColumnIds )
        {
            newOutputColumnIds.push_back( -id );
        }
        std::swap( m_outputColumnIds, newOutputColumnIds );
        std::swap( m_leftIds, m_rightIds );
        std::swap( m_leftOutColumnIdMap, m_rightOutColumnIdMap );
        m_joinHelper->SwapRightJoinToLeft();
        m_joinType = AriesJoinType::LEFT_JOIN;
    }

    AriesOpResult AriesJoinNode::GetNext()
    {
        AriesOpResult result{ AriesOpNodeStatus::ERROR, nullptr };
        switch( m_joinType )
        {
            case AriesJoinType::LEFT_JOIN:
            {
                if ( m_hashJoinType != HashJoinType::None )
                {
                    return LeftJoinWithHashGracePartitioned();
                }
                else
                {
                    if ( m_joinHelper->IsCartesianProduct() )
                        return LeftJoinGetNext();
                    else
                        return LeftJoinGracePartitioned();
                }
            }
            case AriesJoinType::RIGHT_JOIN:
            {
                if ( m_hashJoinType != HashJoinType::None )
                {
                    SwapRightJoinToLeft();
                    m_hashJoinType = ( m_hashJoinType == HashJoinType::RightAsHash ? HashJoinType::LeftAsHash : HashJoinType::RightAsHash );
                    return LeftJoinWithHashGracePartitioned();
                }

                if ( m_joinHelper->IsCartesianProduct() )
                    return RightJoinGetNext();
                else
                {
                    SwapRightJoinToLeft();
                    return LeftJoinGracePartitioned();
                }
            }
            case AriesJoinType::INNER_JOIN:
            {
                if ( m_hashJoinType != HashJoinType::None )
                {
                    return InnerJoinWithHashSimplePartitioned();
                }
                else
                {
                    return SortBasedInnerJoinWithGracePartitioned();
                }
            }
            case AriesJoinType::SEMI_JOIN:
            case AriesJoinType::ANTI_JOIN:
                {
                    m_opParam="half";
                    if( m_hashJoinType != HashJoinType::None && m_uniqueKeys.size() == 1 && m_hashValueKeys.size() == 1 )
                        // return SemiOrAntiHashJoinGetNext();
                        return HashSemiOrAntiJoinGracePartitioned();
                    else
                        // return SemiOrAntiJoinGetNext();
                        return SemiOrAntiJoinGracePartitioned();
                }
            case AriesJoinType::FULL_JOIN:
                {
                    m_opParam = "full";
                    if ( m_hashJoinType != HashJoinType::None && m_hashValueKeys.size() == 1 )
                    {
                        return FullJoinWithHashGracePartitioned();
                    }
                    //return FullJoinGetNext();
                    return FullJoinGracePartitioned();
                }
            default:
                //FIXME need support other join types;
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "join type: " + GetAriesJoinTypeName( m_joinType ) );
                break;
        }
        return result;
    }

    void AriesJoinNode::Close()
    {
        ARIES_ASSERT( m_leftSource && m_rightSource,
                "m_leftSource is nullptr: " + to_string( !!m_leftSource ) + ", m_rightSource is nullptr: "
                        + to_string( !!m_rightSource ) );
        m_leftSource->Close();
        m_rightSource->Close();
        m_leftSubTablesCache.clear();
        m_rightSubTablesCache.clear();

        m_leftDataTable = nullptr;
        m_rightDataTable = nullptr;
    }

    AriesOpResult AriesJoinNode::InnerJoinWithHashSimplePartitioned()
    {
        assert( m_hashJoinType != HashJoinType::None );
        AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };

        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        
        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

        ReleaseData();

        if ( 0 == m_leftDataTable->GetRowCount() || 0 == m_rightDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif

        std::vector< int > left_ids;
        std::vector< int > right_ids;

        const auto& filter_node = m_joinHelper->GetOtherCondtionAsFilterNode();
        const auto& required_columns_in_left = m_joinHelper->GetLeftRequiredColumnsId();
        const auto& required_columns_in_right = m_joinHelper->GetRightRequiredColumnsId();
        std::vector< int > output_ids;

        std::map< int, int > ids_map;
        std::map< int, int > output_ids_map;

        if( filter_node )
        {
            int output_id = 0;
            m_leftOutColumnIdMap.clear();
            for( const auto& id : required_columns_in_left )
            {
                output_id ++;
                if( std::find( m_leftIds.cbegin(), m_leftIds.cend(), id ) != m_leftIds.cend() )
                {
                    output_ids.emplace_back( output_id );
                    ids_map[ id ] = output_id;
                }
                left_ids.emplace_back( id );
                m_leftOutColumnIdMap[ output_id ] = id;
            }

            m_rightOutColumnIdMap.clear();
            for( const auto& id : required_columns_in_right )
            {
                output_id ++;
                if( std::find( m_rightIds.cbegin(), m_rightIds.cend(), -id ) != m_rightIds.cend() )
                {
                    output_ids.emplace_back( output_id );
                    ids_map[ id ] = output_id;
                }
                right_ids.emplace_back( -id );
                m_rightOutColumnIdMap[ output_id ] = -id;
            }

            int idx = 0;
            for( int id : m_outputColumnIds )
            {
                ARIES_ASSERT( id != 0, "id: 0" );
                output_ids_map[ ++idx ] = ids_map[ id ];
            }
        }
        else
        {
            left_ids.assign( m_leftIds.cbegin(), m_leftIds.cend() );
            right_ids.assign( m_rightIds.cbegin(), m_rightIds.cend() );
        }

        if( m_hashJoinType == HashJoinType::RightAsHash )
        {
            //交换对应关系，让left始终为hash端，right为扫描端
            std::swap( m_leftDataTable, m_rightDataTable );
            std::swap( left_ids, right_ids );
            std::swap( m_leftOutColumnIdMap, m_rightOutColumnIdMap );
        }
            
        size_t hashTablePartitionCount = GetHashTablePartitionCount();

        //根据是否需要partition，将partition结果存入leftSubTables
        vector< AriesTableBlockUPtr > leftSubTables;
        if( hashTablePartitionCount == 1 )
            leftSubTables.push_back( m_leftDataTable->Clone( false ) );
        else
        {
            size_t partitionRowCount = m_leftDataTable->GetRowCount() / hashTablePartitionCount;
            int64_t offset = 0;
            while( offset < m_leftDataTable->GetRowCount() )
            {
                size_t readRowCount = std::min( partitionRowCount, ( size_t )( m_leftDataTable->GetRowCount() - offset ) );
                leftSubTables.push_back( m_leftDataTable->GetSubTable( offset, readRowCount, true ) );
                offset += readRowCount;
            }
        }

        const int64_t MIN_TABLE_ROW_COUNT = 100;
        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        bool is_char_type_key = false;
        for ( const auto key : m_uniqueKeys )
        {
            if ( m_leftDataTable->GetColumnType( key ).DataType.ValueType == AriesValueType::CHAR )
            {
                is_char_type_key = true;
                break;
            }
        }

        while( !leftSubTables.empty() )
        {
            auto leftSubTable = std::move( leftSubTables.back() );
            leftSubTables.pop_back();
            AriesHashTableUPtr tmpHashTable;
            AriesHashTableMultiKeysUPtr tmpTableMultiKeys;
            try
            {
                size_t hashTableSize, badCount;

                auto canUseDict = CheckHashJoinConditionForDict( m_leftDataTable, m_uniqueKeys, m_rightDataTable, m_hashValueKeys );

                if( m_uniqueKeys.size() == 1 && !is_char_type_key )
                {
                    tmpHashTable = aries_acc::BuildHashTable( leftSubTable, m_uniqueKeys[ 0 ], canUseDict[ 0 ] );
                    hashTableSize = tmpHashTable->TableSize;
                    badCount = tmpHashTable->BadCount;
                }
                else
                {
                    tmpTableMultiKeys = aries_acc::BuildHashTable( leftSubTable, m_uniqueKeys, canUseDict );
                    hashTableSize = tmpTableMultiKeys->table_size;
                    badCount = tmpTableMultiKeys->bad_count;
                }
                
                size_t hashTableUsage = hashTableSize * sizeof( HashIdType );
                hashTableUsage += badCount * sizeof( HashIdType );
                for ( const auto key : m_uniqueKeys )
                {
                    hashTableUsage += ( hashTableSize + badCount ) * leftSubTable->GetColumnType( key ).GetDataTypeSize();
                }

                //计算可以放多少value table的行
                auto valueTableMemOccupancy = EstimateHashInnerJoinPerRowMemOccupancy( m_rightDataTable, m_hashValueKeys );
                size_t totalRowCount = m_rightDataTable->GetRowCount();
                auto memoryCapacity = AriesDeviceProperty::GetInstance().GetMemoryCapacity() - hashTableUsage;

                size_t valueTableRowCount = memoryCapacity / valueTableMemOccupancy;
                if( valueTableRowCount > totalRowCount )
                    valueTableRowCount = totalRowCount;

                int64_t offset = 0;
                while( offset < m_rightDataTable->GetRowCount() )
                {
                    try
                    {
                        AriesTableBlockUPtr resultTableBlock;
                        size_t readRowCount = std::min( valueTableRowCount, ( size_t )( m_rightDataTable->GetRowCount() - offset ) );
                        auto rightSubTable = m_rightDataTable->GetSubTable( offset, readRowCount, true );

                        JoinPair keyPairs;
                        if( m_hashValueKeys.size() == 1 && !is_char_type_key )
                            keyPairs = aries_acc::InnerJoinWithHash( tmpHashTable, nullptr, rightSubTable, nullptr, m_hashValueKeys[ 0 ], canUseDict[ 0 ] );
                        else
                            keyPairs = boost::get< JoinPair>( aries_acc::InnerJoinWithHash( tmpTableMultiKeys, nullptr, rightSubTable, nullptr, m_hashValueKeys, canUseDict ) );

                        if( keyPairs.JoinCount > 0 )
                        {          
                            // keyPairs.RightIndices->PrefetchToGpu();
                            // keyPairs.LeftIndices->PrefetchToGpu();

                            LOG(INFO) << " join tupleNum:" << keyPairs.JoinCount << endl;
                            AriesTableBlockUPtr leftJoined;
                            if( !left_ids.empty() )
                            {
                                leftJoined = move( leftSubTable->MakeTableByColumns( left_ids, false ) );
                                leftJoined->ResetAllStats();
                                leftJoined->UpdateIndices( keyPairs.LeftIndices );
                                leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );

                                const auto& tmpTableStats = leftJoined->GetStats();
                                tmpTableStats.Print( "AriesJoinNode::InnerJoinWithHash, leftJoined UpdateIndices" );
                                m_leftTableStats += tmpTableStats;
                            }
                            AriesTableBlockUPtr rightJoined;
                            if( !right_ids.empty() )
                            {
                                rightJoined = move( rightSubTable->MakeTableByColumns( right_ids, false ) );
                                rightJoined->ResetAllStats();
                                rightJoined->UpdateIndices( keyPairs.RightIndices );
                                rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );

                                const auto& tmpTableStats = rightJoined->GetStats();
                                tmpTableStats.Print( "AriesJoinNode::InnerJoinWithHash, rightJoined UpdateIndices" );
                                m_rightTableStats += tmpTableStats;
                            }

                            if( leftJoined )
                            {
                                if( rightJoined )
                                {
                                    leftJoined->MergeTable( move( rightJoined ) );
                                }
                                resultTableBlock = move( leftJoined );
                            }
                            else if( rightJoined )
                            {
                                resultTableBlock = move( rightJoined );
                            }
                            else
                            {
                                //select 1 from ...
                                resultTableBlock = GenerateTableWithRowCountOnly( keyPairs.JoinCount );
                            }

                            if ( filter_node && resultTableBlock->GetRowCount() > 0 )
                            {
                    #ifdef ARIES_PROFILE
                                m_opTime += t.end();
                                t.begin();
                    #endif
                                resultTableBlock->ResetAllStats();
                                auto associated = boost::get< AriesBoolArraySPtr >( filter_node->Process( resultTableBlock ) );
                                auto outIndex = aries_acc::FilterAssociated( associated );

                                const auto& tableStats = resultTableBlock->GetStats();
                                tableStats.Print( "AriesFilterNode::GetNext, process expr" );
                                m_tableStats += tableStats;

                                auto outTupleNum = outIndex->GetItemCount();
                                if ( outTupleNum > 0 )
                                {
                                    if ( !output_ids.empty() )
                                    {
                                        resultTableBlock = resultTableBlock->MakeTableByColumns( output_ids, false );
                                        resultTableBlock->ResetAllStats();
                                        resultTableBlock->UpdateIndices( outIndex );
                                        resultTableBlock->UpdateColumnIds( output_ids_map );

                                        const auto& tableStats = resultTableBlock->GetStats();
                                        tableStats.Print( "AriesFilterNode::GetNext, update indice" );
                                        m_tableStats += tableStats;
                                    }
                                    else
                                    {
                                        // select const, only need to output row count
                                        resultTableBlock = std::make_unique< AriesTableBlock >();
                                        resultTableBlock->SetRowCount( outTupleNum );
                                    }
                                }
                                else
                                {
                                    resultTableBlock = GenerateEmptyTable();
                                }
                    #ifdef ARIES_PROFILE
                                auto end = t.end();
                                m_filterOpTime += end;
                                t.begin();
                    #endif
                            }
                        }
                        else
                        {
                            if( left_ids.empty() && right_ids.empty() )
                                resultTableBlock = GenerateTableWithRowCountOnly( keyPairs.JoinCount );
                            else
                                resultTableBlock = GenerateEmptyTable();
                        }
                        if( !result.TableBlock )
                            result.TableBlock = std::move( resultTableBlock );
                        else
                            result.TableBlock->AddBlock( std::move( resultTableBlock ) );
                        
                        offset += readRowCount;
                    }
                    catch( AriesException& e )
                    {
                        if ( e.errCode == ER_ENGINE_OUT_OF_MEMORY && valueTableRowCount > MIN_TABLE_ROW_COUNT )
                            valueTableRowCount = ( size_t )( valueTableRowCount * MAX_RATIO );
                        else
                            throw e;
                    }
                }

                if( m_uniqueKeys.size() == 1 && !is_char_type_key )
                    aries_acc::ReleaseHashTable( tmpHashTable );
                else
                    aries_acc::ReleaseHashTable( tmpTableMultiKeys );
            }
            catch ( AriesException& e )
            {
                if( m_uniqueKeys.size() == 1 && !is_char_type_key )
                    aries_acc::ReleaseHashTable( tmpHashTable );
                else
                    aries_acc::ReleaseHashTable( tmpTableMultiKeys );
                if( e.errCode == ER_ENGINE_OUT_OF_MEMORY && leftSubTable->GetRowCount() > MIN_TABLE_ROW_COUNT )
                {
                    //尝试将hash table再次partition(一分为二)
                    size_t partitionRowCount = leftSubTable->GetRowCount() / 2;
                    int64_t offset = 0;
                    while( offset < leftSubTable->GetRowCount() )
                    {
                        size_t readRowCount = std::min( partitionRowCount, ( size_t )( leftSubTable->GetRowCount() - offset ) );
                        leftSubTables.push_back( leftSubTable->GetSubTable( offset, readRowCount, true ) );
                        offset += readRowCount;
                    }
                }
                else
                    throw e;
            }
        }

        result.Status = AriesOpNodeStatus::END;
        return result;
    }

    AriesOpResult AriesJoinNode::SortBasedInnerJoinWithGracePartitioned()
    {
        AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };

        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        if ( 0 == m_leftDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };
        
        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

        ReleaseData();

        if ( 0 == m_rightDataTable->GetRowCount() || 0 == m_leftDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };

        return m_joinHelper->ProcessGracePartitioned( m_leftDataTable, m_rightDataTable );
    }

/*
    AriesOpResult AriesJoinNode::InnerJoinWithHashGracePartitioned()
    {
        assert( m_hashJoinType != HashJoinType::None );
        AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };

        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        if ( 0 == m_leftDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };
        
        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

        if ( 0 == m_rightDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif

        std::vector< int > left_ids;
        std::vector< int > right_ids;

        const auto& filter_node = m_joinHelper->GetOtherCondtionAsFilterNode();
        const auto& required_columns_in_left = m_joinHelper->GetLeftRequiredColumnsId();
        const auto& required_columns_in_right = m_joinHelper->GetRightRequiredColumnsId();
        std::vector< int > output_ids;

        std::map< int, int > ids_map;
        std::map< int, int > output_ids_map;

        if( filter_node )
        {
            int output_id = 0;
            m_leftOutColumnIdMap.clear();
            for( const auto& id : required_columns_in_left )
            {
                output_id ++;
                if( std::find( m_leftIds.cbegin(), m_leftIds.cend(), id ) != m_leftIds.cend() )
                {
                    output_ids.emplace_back( output_id );
                    ids_map[ id ] = output_id;
                }
                left_ids.emplace_back( id );
                m_leftOutColumnIdMap[ output_id ] = id;
            }

            m_rightOutColumnIdMap.clear();
            for( const auto& id : required_columns_in_right )
            {
                output_id ++;
                if( std::find( m_rightIds.cbegin(), m_rightIds.cend(), -id ) != m_rightIds.cend() )
                {
                    output_ids.emplace_back( output_id );
                    ids_map[ id ] = output_id;
                }
                right_ids.emplace_back( -id );
                m_rightOutColumnIdMap[ output_id ] = -id;
            }

            int idx = 0;
            for( int id : m_outputColumnIds )
            {
                ARIES_ASSERT( id != 0, "id: 0" );
                output_ids_map[ ++idx ] = ids_map[ id ];
            }
        }
        else
        {
            left_ids.assign( m_leftIds.cbegin(), m_leftIds.cend() );
            right_ids.assign( m_rightIds.cbegin(), m_rightIds.cend() );
        }

        if( m_hashJoinType == HashJoinType::RightAsHash )
        {
            //交换对应关系，让left始终为hash端，right为扫描端
            std::swap( m_leftDataTable, m_rightDataTable );
            std::swap( left_ids, right_ids );
            std::swap( m_leftOutColumnIdMap, m_rightOutColumnIdMap );
        }

        size_t hashTablePartitionCount = GetHashTablePartitionCount();

        //根据是否需要partition，将partition结果存入leftSubTables和rightSubTables
        uint32_t seed = 0;
        vector< AriesTableBlockUPtr > leftSubTables;
        GraceHashPartitionTable( m_leftDataTable,
                                 m_uniqueKeys,
                                 hashTablePartitionCount,
                                 seed,
                                 leftSubTables );
        vector< AriesTableBlockUPtr > rightSubTables;
        GraceHashPartitionTable( m_rightDataTable,
                                 m_hashValueKeys,
                                 hashTablePartitionCount,
                                 seed,
                                 rightSubTables );
        if( 1 != hashTablePartitionCount )
            ++seed;

        const int64_t MIN_TABLE_ROW_COUNT = 100;
        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        while( !leftSubTables.empty() )
        {
            auto leftPartTable = std::move( leftSubTables.back() );
            leftSubTables.pop_back();

            auto rightPartTable = std::move( rightSubTables.back() );
            rightSubTables.pop_back();

            if ( 0 == leftPartTable->GetRowCount() || 0 == rightPartTable->GetRowCount() )
            {
                continue;
            }

            AriesHashTableUPtr tmpHashTable;
            AriesHashTableMultiKeysUPtr tmpTableMultiKeys;
            try
            {
                auto canUseDict = CheckHashJoinConditionForDict( m_leftDataTable, m_uniqueKeys, m_rightDataTable, m_hashValueKeys );
                if( m_uniqueKeys.size() == 1 )
                    tmpHashTable = aries_acc::BuildHashTable( leftPartTable, m_uniqueKeys[ 0 ], canUseDict[ 0 ] );
                else
                    tmpTableMultiKeys = aries_acc::BuildHashTable( leftPartTable, m_uniqueKeys, canUseDict );
                
                //计算可以放多少value table的行
                auto valueTableMemOccupancy = EstimateHashInnerJoinPerRowMemOccupancy( rightPartTable, m_hashValueKeys );
                size_t rowCount = valueTableMemOccupancy == 0 ? UINT64_MAX : AriesDeviceProperty::GetInstance().GetMemoryCapacity() / valueTableMemOccupancy;

                size_t valueTableRowCount = std::min( ( size_t )( rowCount * MAX_RATIO ), ( size_t )rightPartTable->GetRowCount() );

                int64_t offset = 0;
                while( offset < rightPartTable->GetRowCount() )
                {
                    try
                    {
                        AriesTableBlockUPtr resultTableBlock;
                        size_t readRowCount = std::min( valueTableRowCount, ( size_t )( rightPartTable->GetRowCount() - offset ) );
                        auto rightSubTable = rightPartTable->GetSubTable( offset, readRowCount, true );
                        rightSubTable->MaterilizeColumns( m_hashValueKeys );

                        JoinPair keyPairs;
                        if( m_hashValueKeys.size() == 1 )
                            keyPairs = aries_acc::InnerJoinWithHash( tmpHashTable, nullptr, rightSubTable, nullptr, m_hashValueKeys[ 0 ], canUseDict[ 0 ] );
                        else
                            keyPairs = boost::get< JoinPair>( aries_acc::InnerJoinWithHash( tmpTableMultiKeys, nullptr, rightSubTable, nullptr, m_hashValueKeys, canUseDict ) );

                        if( keyPairs.JoinCount > 0 )
                        {          
                            keyPairs.RightIndices->PrefetchToGpu();
                            keyPairs.LeftIndices->PrefetchToGpu();

                            LOG(INFO) << " join tupleNum:" << keyPairs.JoinCount << endl;
                            AriesTableBlockUPtr leftJoined;
                            if( !left_ids.empty() )
                            {
                                leftJoined = move( leftPartTable->MakeTableByColumns( left_ids, false ) );
                                leftJoined->ResetAllStats();
                                leftJoined->UpdateIndices( keyPairs.LeftIndices );
                                leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );

                                const auto& tmpTableStats = leftJoined->GetStats();
                                tmpTableStats.Print( "AriesJoinNode::InnerJoinWithHash, leftJoined UpdateIndices" );
                                m_leftTableStats += tmpTableStats;
                            }
                            AriesTableBlockUPtr rightJoined;
                            if( !right_ids.empty() )
                            {
                                rightJoined = move( rightSubTable->MakeTableByColumns( right_ids, false ) );
                                rightJoined->ResetAllStats();
                                rightJoined->UpdateIndices( keyPairs.RightIndices );
                                rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );

                                const auto& tmpTableStats = rightJoined->GetStats();
                                tmpTableStats.Print( "AriesJoinNode::InnerJoinWithHash, rightJoined UpdateIndices" );
                                m_rightTableStats += tmpTableStats;
                            }

                            if( leftJoined )
                            {
                                if( rightJoined )
                                {
                                    leftJoined->MergeTable( move( rightJoined ) );
                                }
                                resultTableBlock = move( leftJoined );
                            }
                            else if( rightJoined )
                            {
                                resultTableBlock = move( rightJoined );
                            }
                            else
                            {
                                //select 1 from ...
                                resultTableBlock = GenerateTableWithRowCountOnly( keyPairs.JoinCount );
                            }

                            if ( filter_node && resultTableBlock->GetRowCount() > 0 )
                            {
                    #ifdef ARIES_PROFILE
                                m_opTime += t.end();
                                t.begin();
                    #endif
                                resultTableBlock->ResetAllStats();
                                auto associated = boost::get< AriesBoolArraySPtr >( filter_node->Process( resultTableBlock ) );
                                auto outIndex = aries_acc::FilterAssociated( associated );

                                const auto& tableStats = resultTableBlock->GetStats();
                                tableStats.Print( "AriesFilterNode::GetNext, process expr" );
                                m_tableStats += tableStats;

                                auto outTupleNum = outIndex->GetItemCount();
                                if ( outTupleNum > 0 )
                                {
                                    if ( !output_ids.empty() )
                                    {
                                        resultTableBlock = resultTableBlock->MakeTableByColumns( output_ids, false );
                                        resultTableBlock->ResetAllStats();
                                        resultTableBlock->UpdateIndices( outIndex );
                                        resultTableBlock->UpdateColumnIds( output_ids_map );

                                        const auto& tableStats = resultTableBlock->GetStats();
                                        tableStats.Print( "AriesFilterNode::GetNext, update indice" );
                                        m_tableStats += tableStats;
                                    }
                                    else
                                    {
                                        // select const, only need to output row count
                                        resultTableBlock = std::make_unique< AriesTableBlock >();
                                        resultTableBlock->SetRowCount( outTupleNum );
                                    }
                                }
                                else
                                {
                                    resultTableBlock = GenerateEmptyTable();
                                }
                    #ifdef ARIES_PROFILE
                                auto end = t.end();
                                m_filterOpTime += end;
                                t.begin();
                    #endif
                            }
                        }
                        else
                        {
                            if( left_ids.empty() && right_ids.empty() )
                                resultTableBlock = GenerateTableWithRowCountOnly( keyPairs.JoinCount );
                            else
                                resultTableBlock = GenerateEmptyTable();
                        }
                        if( !result.TableBlock )
                            result.TableBlock = std::move( resultTableBlock );
                        else
                            result.TableBlock->AddBlock( std::move( resultTableBlock ) );
                        
                        offset += readRowCount;
                    }
                    catch( AriesException& e )
                    {
                        if ( e.errCode == ER_ENGINE_OUT_OF_MEMORY && valueTableRowCount > MIN_TABLE_ROW_COUNT )
                            valueTableRowCount = ( size_t )( valueTableRowCount * MAX_RATIO );
                        else
                            throw e;
                    }
                }

                if( m_uniqueKeys.size() == 1 )
                    aries_acc::ReleaseHashTable( tmpHashTable );
                else
                    aries_acc::ReleaseHashTable( tmpTableMultiKeys );
            }
            catch ( AriesException& e )
            {
                if( m_uniqueKeys.size() == 1 )
                    aries_acc::ReleaseHashTable( tmpHashTable );
                else
                    aries_acc::ReleaseHashTable( tmpTableMultiKeys );
                if( e.errCode == ER_ENGINE_OUT_OF_MEMORY && leftPartTable->GetRowCount() > MIN_TABLE_ROW_COUNT )
                {
                    //尝试将 left table再次partition(一分为二)
                    vector< AriesDataBufferSPtr > buffers;
                    for( int id : m_uniqueKeys )
                        buffers.push_back( leftPartTable->GetColumnBuffer( id ) );
                    vector< PartitionedIndices > partitions = PartitionColumnData( buffers, 2, seed );
                    for( const auto& part : partitions )
                    {
                        AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
                        auto partTable = leftPartTable->Clone( false );
                        partTable->UpdateIndices( indices );
                        leftSubTables.push_back( std::move( partTable ) );
                    }

                    // right table必须同步partition
                    buffers.clear();
                    for( int id : m_hashValueKeys )
                        buffers.push_back( rightPartTable->GetColumnBuffer( id ) );
                    partitions = PartitionColumnData( buffers, 2, seed );
                    for( const auto& part : partitions )
                    {
                        AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
                        auto table = rightPartTable->Clone( false );
                        table->UpdateIndices( indices );
                        rightSubTables.push_back( std::move( table ) );
                    }
                    ++seed;
                }
                else
                    throw e;
            }
        }

        result.Status = AriesOpNodeStatus::END;
        return result;
    }
*/

    size_t AriesJoinNode::GetPartitionCountForLeftHashJoin() const
    {
        auto leftRowCount = m_leftDataTable->GetRowCount();
        auto rightRowCount = m_rightDataTable->GetRowCount();
        const aries_engine::AriesTableBlockUPtr& hashTable = ( m_hashJoinType == HashJoinType::LeftAsHash ? m_leftDataTable : m_rightDataTable );
        const aries_engine::AriesTableBlockUPtr& valueTable = ( m_hashJoinType == HashJoinType::LeftAsHash ? m_rightDataTable : m_leftDataTable );
        auto buildHashTableUsage = EstimateBuildHashTableMemOccupancyPerRow( hashTable, m_uniqueKeys );
        auto hashTableSizePerRow = GetHashTableSizePerRow( hashTable, m_uniqueKeys );

        auto hashJoinUsage = hashTableSizePerRow + GetLeftHashJoinUsage( leftRowCount, rightRowCount, valueTable, m_hashValueKeys, false );

        auto kernelUsage = hashTableSizePerRow + m_joinHelper->GetLeftHashJoinKernelUsage( leftRowCount, rightRowCount );

        // maxUsage 代表左表每一行数据将会占用多大显存
        auto maxUsage = max( max( kernelUsage, hashJoinUsage ) , buildHashTableUsage );

        // 这里乘以 0.7 是因为实际上似乎分块越多，效果越好
        auto available = size_t( AriesDeviceProperty::GetInstance().GetMemoryCapacity() * 0.7 );
        auto rowCount = ( available - 1 + maxUsage ) / maxUsage;
        return ( leftRowCount + rowCount - 1 ) / rowCount;
    }

    size_t AriesJoinNode::GetHashTablePartitionCount() const
    {
        size_t partitionCount = 1;

        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity() * MAX_RATIO;

        //计算hash table是否能整体放入显存
        auto hashTableMemOccupancy = EstimateBuildHashTableMemOccupancy( m_leftDataTable, m_uniqueKeys );

        double currentRatio = hashTableMemOccupancy / available;
        if( currentRatio > MAX_RATIO )
            partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

        return partitionCount;
    }

    AriesOpResult AriesJoinNode::InnerJoinGetNext()
    {
        if ( !CacheAllLeftToSubTables( SUBTABLE_COUNT ))
        {
            return { AriesOpNodeStatus::ERROR, nullptr };
        }

        CacheAllRightTable();
        if ( !m_rightDataTable )
        {
            m_leftSubTablesCache.clear();
            return { AriesOpNodeStatus::ERROR, nullptr };
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        while ( !IsCurrentThdKilled() )
        {
            if ( m_leftSubTablesCache.empty() )
            {
                break;
            }
            AriesOpResult result = InnerJoinOneBlock();
            if ( result.TableBlock != nullptr )
            {
#ifdef ARIES_PROFILE
                m_opTime += t.end();
#endif
                return result;
            }
        }
        if (IsCurrentThdKilled())
        {
            LOG(INFO) << "thread was killed in AriesJoinNode::InnerJoinGetNext";
            SendKillMessage();
        }
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        return { AriesOpNodeStatus::END, GenerateEmptyTable() };
    }

    AriesOpResult AriesJoinNode::InnerJoinGetNextV2()
    {
        if( !CacheAllLeftToSubTables() )
        {
            return
            {   AriesOpNodeStatus::ERROR, nullptr};
        }

        if( !CacheAllRightToSubTables() )
        {
            return
            {   AriesOpNodeStatus::ERROR, nullptr};
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        while( !IsCurrentThdKilled() )
        {
            if( m_leftSubTablesCache.empty() || m_rightSubTablesCache.empty() )
            {
                break;
            }
            AriesOpResult result = InnerJoinOneBlockV2();
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
            return result;
        }
        if( IsCurrentThdKilled() )
        {
            LOG(INFO)<< "thread was killed in AriesJoinNode::InnerJoinGetNext";
            SendKillMessage();
        }
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        return
        {   AriesOpNodeStatus::END, GenerateEmptyTable()};
    }

    bool AriesJoinNode::IsConstFalseCondition()
    {
        if ( !m_equalCondition && m_otherCondition->GetType() == AriesExprType::TRUE_FALSE )
        {
            auto b = boost::get< bool >( m_otherCondition->GetContent() );
            if ( !b )
                return true;
        }
        return false;
    }

    AriesOpResult AriesJoinNode::InnerJoinOneSubTable(AriesTableBlockUPtr &leftTable, AriesTableBlockUPtr &rightTable)
    {
        leftTable->ResetAllStats();
        rightTable->ResetAllStats();

        AriesOpResult result{ AriesOpNodeStatus::CONTINUE, nullptr };

        if ( IsConstFalseCondition() )
        {
            result.TableBlock = GenerateEmptyTable();
            result.Status = AriesOpNodeStatus::END;
            return result;
        }

        AriesJoinResult joinResult = m_joinHelper->Process( leftTable, rightTable );

        auto tmpTableStats = leftTable->GetStats();
        tmpTableStats.Print( "AriesJoinNode::InnerJoinOneSubTable, process leftTable" );
        m_leftTableStats += tmpTableStats;

        tmpTableStats = rightTable->GetStats();
        tmpTableStats.Print( "AriesJoinNode::InnerJoinOneSubTable, process rightTable" );
        m_rightTableStats += tmpTableStats;

        JoinPair keyPairs = boost::get<JoinPair>( joinResult );
        if ( keyPairs.JoinCount > 0 )
        {
            LOG(INFO) << " join tupleNum:" << keyPairs.JoinCount << endl;
            AriesTableBlockUPtr leftJoined;
            if ( !m_leftIds.empty() )
            {
                leftJoined = move( leftTable->MakeTableByColumns( m_leftIds, false ) );
                leftJoined->ResetAllStats();
                leftJoined->UpdateIndices( keyPairs.LeftIndices );
                leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );

                const auto& tmpTableStats = leftJoined->GetStats();
                tmpTableStats.Print( "AriesJoinNode::InnerJoinOneSubTable, leftJoined UpdateIndices" );
                m_leftTableStats += tmpTableStats;
            }
            AriesTableBlockUPtr rightJoined;
            if ( !m_rightIds.empty() )
            {
                rightJoined = move( rightTable->MakeTableByColumns( m_rightIds, false ) );
                rightJoined->ResetAllStats();
                rightJoined->UpdateIndices( keyPairs.RightIndices );
                rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );

                const auto& tmpTableStats = rightJoined->GetStats();
                tmpTableStats.Print( "AriesJoinNode::InnerJoinOneSubTable, rightJoined UpdateIndices" );
                m_rightTableStats += tmpTableStats;
            }
            if ( leftJoined )
            {
                if ( rightJoined )
                {
                    leftJoined->MergeTable( move( rightJoined ) );
                }
                result.TableBlock = move( leftJoined );
            }
            else if ( rightJoined )
            {
                result.TableBlock = move( rightJoined );
            }
            else
            {
                //select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( keyPairs.JoinCount );
            }
        }
        else
            result.TableBlock = GenerateEmptyTable();
        return result;
    }

    AriesOpResult AriesJoinNode::InnerJoinOneBlockV2()
    {
        AriesOpResult result
        { AriesOpNodeStatus::CONTINUE, nullptr };

        int leftTableCount = m_leftSubTablesCache.size();
        while( !m_rightSubTablesCache.empty() )
        {
            auto &rightTable = m_rightSubTablesCache.front();
            for( int i = 0; i < leftTableCount; ++i )
            {
                auto &leftTable = m_leftSubTablesCache[i];

                ARIES_FUNC_LOG_BEGIN;

                auto res = InnerJoinOneSubTable( leftTable, rightTable );

                ARIES_FUNC_LOG_END;

                if( res.TableBlock != nullptr )
                {
                    if( !result.TableBlock )
                        result.TableBlock = move( res.TableBlock );
                    else
                        result.TableBlock->AddBlock( move( res.TableBlock ) );
                }
            }
            m_rightSubTablesCache.pop_front();
        }

        if( m_rightSubTablesCache.empty() ) // all end
        {
            m_leftSubTablesCache.clear();
            result.Status = AriesOpNodeStatus::END;
        }

        return result;
    }

    AriesOpResult AriesJoinNode::InnerJoinOneBlock()
    {
        auto &leftTable = m_leftSubTablesCache.front();
        auto rightRowCount = m_rightDataTable->GetRowCount();
        AriesOpResult result{AriesOpNodeStatus::CONTINUE, nullptr};
        if ( m_rightHandledOffset < rightRowCount )
        {
            ARIES_FUNC_LOG_BEGIN;

            auto rightTryRowCount = std::min(rightRowCount - m_rightHandledOffset, m_rightOneBlockSize) ;

            m_rightDataTable->ResetAllStats();

            auto rightSubTable = m_rightDataTable->GetSubTable(m_rightHandledOffset, rightTryRowCount);

            const auto& tmpTableStats = m_rightDataTable->GetStats();
            tmpTableStats.Print( "AriesJoinNode::InnerJoinOneBlock, m_rightDataTable GetSubTable" );
            m_rightTableStats += tmpTableStats;

            auto res = InnerJoinOneSubTable(leftTable, rightSubTable);

            ARIES_FUNC_LOG_END;

            m_rightHandledOffset += rightSubTable->GetRowCount();
            if (res.TableBlock != nullptr)
            {
                result.TableBlock = move( res.TableBlock );
            }
        }

        if ( m_rightHandledOffset >= rightRowCount ) // one loop end
        {
            m_rightHandledOffset = 0;
            if ( m_leftSubTablesCache.size() == 1 ) // all end
            {
                m_leftSubTablesCache.clear();
                result.Status = AriesOpNodeStatus::END;
            }
            else
            {
                m_leftSubTablesCache.pop_front();
            }
        }

        return result;
    }

    bool AriesJoinNode::CacheAllToSubTables( deque< AriesTableBlockUPtr > &cache, int64_t& totalRowCount, AriesOpNodeSPtr &source,
            AriesTableBlockUPtr &emptyTable )
    {
        auto data = source->GetNext();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        if( data.TableBlock )
        {
            if( !emptyTable )
                emptyTable = data.TableBlock->CloneWithNoContent();

            if( data.TableBlock->GetRowCount() > 0 )
            {
                totalRowCount += data.TableBlock->GetRowCount();
                data.TableBlock->ResetAllStats();
                cache.push_back( move( data.TableBlock ) );
            }
        }

        while( data.Status == AriesOpNodeStatus::CONTINUE && !IsCurrentThdKilled() )
        {
            data = source->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if( data.TableBlock && data.TableBlock->GetRowCount() > 0 )
            {
                totalRowCount += data.TableBlock->GetRowCount();
                data.TableBlock->ResetAllStats();
                cache.push_back( move( data.TableBlock ) );
            }
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }

        if( IsCurrentThdKilled() )
        {
            LOG(INFO)<< "thread was killed in AriesJoinNode::CacheAllToSubTables";
            SendKillMessage();
        }

        if( data.Status == AriesOpNodeStatus::END )
        {
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if( data.TableBlock && data.TableBlock->GetRowCount() > 0 )
            {
                totalRowCount += data.TableBlock->GetRowCount();
                data.TableBlock->ResetAllStats();
                cache.push_back( move( data.TableBlock ) );
            }
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
            return true;
        }
        else
        {
            cache.clear();
            return false;
        }
    }

    bool AriesJoinNode::CacheAllToSubTables( deque<AriesTableBlockUPtr> &cache,
                                             int64_t& totalRowCount,
                                             AriesOpNodeSPtr &source,
                                             AriesTableBlockUPtr &emptyTable,
                                             std::size_t rowCount,
                                             AriesTableBlockStats& tableStats )
    {
        AriesTableBlockUPtr tableBlocks;
        auto data = source->GetNext();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        tableBlocks = move( data.TableBlock );
        if ( tableBlocks )
        {
            totalRowCount += tableBlocks->GetRowCount();
            tableBlocks->ResetAllStats();
        }
        while ( data.Status == AriesOpNodeStatus::CONTINUE  && !IsCurrentThdKilled() )
        {
            data = source->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if ( data.TableBlock && data.TableBlock->GetRowCount() > 0 )
            {
                totalRowCount += data.TableBlock->GetRowCount();
                tableBlocks->AddBlock( move( data.TableBlock ) );
            }
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }
#ifdef ARIES_PROFILE
        t.begin();
#endif
        if (IsCurrentThdKilled())
        {
            LOG(INFO) << "thread was killed in AriesJoinNode::CacheAllToSubTables";
            SendKillMessage();
        }

        if (data.Status == AriesOpNodeStatus::END)
        {
            size_t allCount = tableBlocks ? tableBlocks->GetRowCount() : 0;
            size_t offset = 0;
            size_t realCount = 0;
            ARIES_FUNC_LOG_BEGIN;
            while ( offset < allCount )
            {
                realCount = allCount - offset;
                realCount = realCount < rowCount ? realCount : rowCount;
                AriesTableBlockUPtr subTable = tableBlocks->GetSubTable( offset, realCount );
                offset += subTable->GetRowCount();
                cache.push_back( std::move( subTable ) );
            }
            ARIES_FUNC_LOG_END;
            if (emptyTable == nullptr)
            {
                emptyTable = tableBlocks->CloneWithNoContent();
            }
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
            const auto& tmpTableStats = tableBlocks->GetStats();
            tmpTableStats.Print( "AriesJoinNode::CacheAllToSubTables" );
            tableStats += tmpTableStats;

            return true;
        }
        cache.clear();
        return false;
    }

    bool AriesJoinNode::CacheAllLeftToSubTables(std::size_t rowCount)
    {
        if ( !m_isLeftSubTableCached )
        {
            m_isLeftSubTableCached = true;
            return CacheAllToSubTables( m_leftSubTablesCache,
                                        m_leftRowCount,
                                        m_leftSource,
                                        m_leftEmptyTable,
                                        rowCount,
                                        m_leftTableStats );
        }
        return true;
    }

    bool AriesJoinNode::CacheAllLeftToSubTables()
    {
        if( !m_isLeftSubTableCached )
        {
            m_isLeftSubTableCached = true;
            return CacheAllToSubTables( m_leftSubTablesCache, m_leftRowCount, m_leftSource, m_leftEmptyTable );
        }
        return true;
    }

    bool AriesJoinNode::CacheAllRightToSubTables(std::size_t rowCount)
    {
        if ( !m_isRightSubTableCached )
        {
            m_isRightSubTableCached = true;
            return CacheAllToSubTables( m_rightSubTablesCache,
                                        m_rightRowCount,
                                        m_rightSource,
                                        m_rightEmptyTable,
                                        rowCount,
                                        m_rightTableStats );
        }
        return true;
    }

    bool AriesJoinNode::CacheAllRightToSubTables()
    {
        if( !m_isRightSubTableCached )
        {
            m_isRightSubTableCached = true;
            return CacheAllToSubTables( m_rightSubTablesCache, m_rightRowCount, m_rightSource, m_rightEmptyTable );
        }
        return true;
    }

    AriesTableBlockUPtr AriesJoinNode::ReadAllData( AriesOpNodeSPtr dataSource,
                                                    AriesTableBlockUPtr &emptyTable,
                                                    AriesTableBlockStats& tableStats )
    {
        auto dataBlock = dataSource->GetNext();
        AriesTableBlockUPtr tableBlock = std::move( dataBlock.TableBlock );
        if ( tableBlock )
            tableBlock->ResetAllStats();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        while( dataBlock.Status == AriesOpNodeStatus::CONTINUE && !IsCurrentThdKilled() )
        {
            dataBlock = dataSource->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if ( dataBlock.TableBlock && dataBlock.TableBlock->GetRowCount() > 0 )
            {
                tableBlock->AddBlock( std::move( dataBlock.TableBlock ) );
            }
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }
        if (IsCurrentThdKilled())
        {
            LOG(INFO) << "thread was killed in AriesJoinNode::ReadAllData";
            SendKillMessage();
        }

        if ( AriesOpNodeStatus::ERROR == dataBlock.Status )
        {
            return nullptr;
        }
        if (emptyTable == nullptr) {
            emptyTable = tableBlock->CloneWithNoContent();
        }

        const auto& tmpTableStats = tableBlock->GetStats();
        tmpTableStats.Print( "AriesJoinNode::ReadAllData" );
        tableStats += tmpTableStats;

        return tableBlock;
    }

    void AriesJoinNode::CacheAllLeftTable()
    {
        if ( !m_isLeftDataAllRead )
        {
            m_isLeftDataAllRead = true;
            m_leftDataTable = ReadAllData( m_leftSource, m_leftEmptyTable, m_leftTableStats );
            if ( m_leftDataTable )
                m_leftRowCount = m_leftDataTable->GetRowCount();
        }
    }

    void AriesJoinNode::CacheAllRightTable()
    {
        if ( !m_isRightDataAllRead )
        {
            m_isRightDataAllRead = true;
            m_rightDataTable = ReadAllData( m_rightSource, m_rightEmptyTable, m_rightTableStats );
            if ( m_rightDataTable )
                m_rightRowCount = m_rightDataTable->GetRowCount();
        }
    }

    AriesTableBlockUPtr AriesJoinNode::GetNextCachedSubTable(deque<AriesTableBlockUPtr> &cache)
    {
        if ( cache.empty() ) {
            return nullptr;
        }
        auto table = std::move( cache.front() );
        cache.pop_front();
        return std::move(table);
    }

    AriesTableBlockUPtr AriesJoinNode::GenerateEmptyTable()
    {
        if ( !m_outputColumnIds.empty() )
        {
            if ( m_outputColumnTypes.empty())
            {
                if ( !m_leftEmptyTable )
                {
                    m_leftEmptyTable = m_leftSource->GetEmptyTable();
                }

                if ( !m_rightEmptyTable )
                {
                    m_rightEmptyTable = m_rightSource->GetEmptyTable();
                }

                auto left = m_leftEmptyTable->GetColumnTypes( m_leftIds );
                auto right = m_rightEmptyTable->GetColumnTypes( m_rightIds );
                index_t leftColIndex = 0;
                index_t rightColIndex = 0;
                for ( int id : m_outputColumnIds )
                {
                    m_outputColumnTypes.push_back( id > 0 ? left[ leftColIndex++ ] : right[ rightColIndex++ ] );
                }
            }
            return AriesTableBlock::CreateTableWithNoRows(m_outputColumnTypes);
        }
        else
        {
            return make_unique< AriesTableBlock >();
        }
    }

    AriesTableBlockUPtr AriesJoinNode::GenerateTableWithRowCountOnly(size_t count)
    {
        AriesTableBlockUPtr table = make_unique< AriesTableBlock >();
        table->SetRowCount(count);
        return table;
    }

    void AriesJoinNode::SplitOutputColumnIds()
    {
        bool bLeft;
        index_t outColumnId = 0;
        for( int id : m_outputColumnIds )
        {
            ++outColumnId;
            ARIES_ASSERT( id != 0, "id: 0" );
            bLeft = id > 0;
            id = bLeft ? id : -id;
            if( bLeft )
            {
                m_leftOutColumnIdMap[outColumnId] = id;
                m_leftIds.push_back( id );
            }
            else
            {
                m_rightOutColumnIdMap[outColumnId] = id;
                m_rightIds.push_back( id );
            }
        }
    }

    AriesOpResult AriesJoinNode::FullJoinWithHashGracePartitioned()
    {
        assert( m_hashJoinType != HashJoinType::None );
        assert( m_joinType == AriesJoinType::FULL_JOIN );
        assert( m_uniqueKeys.size() == 1 );

        bool needToSwap = false;
        if ( m_hashJoinType == HashJoinType::RightAsHash )
        {
            std::swap( m_leftSource, m_rightSource );
            std::swap( m_leftIds, m_rightIds );
            std::swap( m_leftOutColumnIdMap, m_rightOutColumnIdMap );
            needToSwap = true;
            // m_joinHelper->SwapRightJoinToLeft();
            m_hashJoinType = HashJoinType::LeftAsHash;
        }

        AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };
        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        if ( 0 == m_leftDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };
        
        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

        // if ( 0 == m_rightDataTable->GetRowCount() )
        //     return { AriesOpNodeStatus::END, GenerateEmptyTable() };

        //暂时使用left join类似的分块判断逻辑
        size_t partitionCount = GetPartitionCountForLeftHashJoin();

        //根据是否需要 partition，将 partition 结果存入 leftSubTables 和 rightSubTables
        std::vector< AriesTableBlockUPtr > leftSubTables;
        std::vector< AriesTableBlockUPtr > rightSubTables;
        uint32_t seed = 0;
        if( partitionCount == 1 )
        {
            leftSubTables.push_back( m_leftDataTable->Clone( false ) );
            rightSubTables.push_back( m_rightDataTable->Clone( false ) );
        }
        else
        {
            std::vector< AriesDataBufferSPtr > buffers;

            //partition left table
            for( int id : m_uniqueKeys )
                buffers.push_back( m_leftDataTable->GetColumnBuffer( id ) );
            std::vector< PartitionedIndices > partitions = PartitionColumnData( buffers, partitionCount, seed );
            for( const auto& part : partitions )
            {
                AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
                if ( indices->GetItemCount() == 0 )
                {
                    leftSubTables.emplace_back( m_leftDataTable->CloneWithNoContent() );
                    continue;
                }
                auto table = m_leftDataTable->Clone( false );
                table->UpdateIndices( indices );
                leftSubTables.push_back( std::move( table ) );
            }

            //partition right table
            buffers.clear();
            for( int id : m_hashValueKeys )
                buffers.push_back( m_rightDataTable->GetColumnBuffer( id ) );
            partitions = PartitionColumnData( buffers, partitionCount, seed );
            for( const auto& part : partitions )
            {
                AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
                if ( indices->GetItemCount() == 0 )
                {
                    rightSubTables.emplace_back( m_rightDataTable->CloneWithNoContent() );
                    continue;
                }
                auto table = m_rightDataTable->Clone( false );
                table->UpdateIndices( indices );
                rightSubTables.push_back( std::move( table ) );
            }
            ++seed;
        }

        AriesTableBlockUPtr resultTable = nullptr;
        size_t totalRowCount = 0;
        while ( !leftSubTables.empty() )
        {
            auto leftPartTable = std::move( leftSubTables.back() );
            auto rightPartTable = std::move( rightSubTables.back() );

            leftSubTables.pop_back();
            rightSubTables.pop_back();

            if ( leftPartTable->GetRowCount() == 0 )
            {
                continue;
            }
            auto canUseDict = CheckHashJoinConditionForDict( leftPartTable, m_uniqueKeys, rightPartTable, m_hashValueKeys );
            auto hashTable = aries_acc::BuildHashTable( leftPartTable, m_uniqueKeys[ 0 ], canUseDict[ 0 ] );
            auto joinResult = m_joinHelper->ProcessHashFullJoin( hashTable, leftPartTable, nullptr, rightPartTable, nullptr, m_hashValueKeys[ 0 ], canUseDict[ 0 ], needToSwap );
            aries_acc::ReleaseHashTable( hashTable );
            auto joinKeyPairs = boost::get< JoinPair >( joinResult );

            AriesTableBlockUPtr leftJoined = nullptr;
            if ( !m_leftIds.empty() )
            {
                leftJoined = leftPartTable->MakeTableByColumns( m_leftIds, false );
                leftJoined->UpdateIndices( joinKeyPairs.LeftIndices, true );
                leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );
            }

            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !m_rightIds.empty() )
            {
                rightJoined = rightPartTable->MakeTableByColumns( m_rightIds, false );
                rightJoined->UpdateIndices( joinKeyPairs.RightIndices, true );
                rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );
            }

            AriesTableBlockUPtr partResult;
            if ( leftJoined )
            {
                if ( rightJoined )
                {
                    leftJoined->MergeTable( move( rightJoined ) );
                }
                partResult = move( leftJoined );
            }
            else if ( rightJoined )
            {
                partResult = move( rightJoined );
            }
            else
            {
                //select 1 from ...
                totalRowCount += joinKeyPairs.JoinCount;
                continue;
            }

            if ( resultTable )
            {
                resultTable->AddBlock( std::move( partResult ) );
            }
            else
            {
                resultTable = std::move( partResult );
            }
        }

        if ( m_leftIds.empty() && m_rightIds.empty() )
        {
            result.TableBlock = GenerateTableWithRowCountOnly( totalRowCount );
        }
        else
        {
            result.TableBlock = std::move( resultTable );
        }

        result.Status = AriesOpNodeStatus::END;
        return result;
    }

    AriesOpResult AriesJoinNode::LeftJoinWithHashGracePartitioned()
    {
        assert( m_joinType == AriesJoinType::LEFT_JOIN );

        AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };

        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        if ( 0 == m_leftDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };
        
        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

        ReleaseData();

        // if ( 0 == m_rightDataTable->GetRowCount() )
        //     return { AriesOpNodeStatus::END, GenerateEmptyTable() };

        size_t partitionCount = GetPartitionCountForLeftHashJoin();
        std::cout << "partitionCount will be: " << partitionCount << std::endl;

        //根据是否需要 partition，将 partition 结果存入 leftSubTables 和 rightSubTables
        std::vector< AriesTableBlockUPtr > leftSubTables;
        std::vector< AriesTableBlockUPtr > rightSubTables;
        uint32_t seed = 0;
        if( partitionCount == 1 )
        {
            leftSubTables.push_back( m_leftDataTable->Clone( false ) );
            rightSubTables.push_back( m_rightDataTable->Clone( false ) );
        }
        else
        {
            std::vector< AriesDataBufferSPtr > buffers;
            auto leftKeys = m_uniqueKeys;
            auto rightKeys = m_hashValueKeys;
            if( m_hashJoinType == HashJoinType::RightAsHash )
                std::swap( leftKeys, rightKeys );
            //partition left table
            for( int id : leftKeys )
            {
                if ( EncodeType::DICT == m_leftDataTable->GetColumnEncodeType( id ) )
                {
                    buffers.push_back( m_leftDataTable->GetDictEncodedColumnIndiceBuffer( id ) );
                }
                else
                {
                    buffers.push_back( m_leftDataTable->GetColumnBuffer( id ) );
                }
            }
            bool hashNullValue = false;

            auto partitionResult = PartitionColumnDataEx( { buffers }, partitionCount, seed, &hashNullValue );
            buffers.clear();

            size_t partitionIndex = 0;

            AriesTableBlock *notEmptyTable = nullptr;
            for( const auto& part : partitionResult.AllIndices )
            {
                AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
                if ( indices->GetItemCount() == 0 )
                {
                    leftSubTables.emplace_back( nullptr );
                    partitionIndex ++;
                    continue;
                }

                std::vector< int32_t > columnsId;
                for ( const auto id : m_leftDataTable->GetAllColumnsId() )
                {
                    if ( std::find( leftKeys.cbegin(), leftKeys.cend(), id ) == leftKeys.cend() )
                    {
                        columnsId.emplace_back( id );
                    }
                }

                AriesTableBlockUPtr table;
                if ( !columnsId.empty() )
                {
                    table = m_leftDataTable->MakeTableByColumns( columnsId, false );
                    table->UpdateIndices( indices );
                }
                else
                {
                    table = std::make_unique< AriesTableBlock >();
                }

                for ( size_t i = 0; i < leftKeys.size(); i++ )
                {
                    if ( EncodeType::DICT == m_leftDataTable->GetColumnEncodeType( leftKeys[ i ] ) )
                    {
                        auto dictColumn = m_leftDataTable->GetDictEncodedColumn( leftKeys[ i ] );
                        auto newDictIndices = std::make_shared< AriesVariantIndices >();
                        newDictIndices->AddDataBuffer( partitionResult.AllBuffers[ i ][ partitionIndex ]->ToAriesDataBuffer() );
                        auto newDictColumn =
                            std::make_shared< AriesDictEncodedColumn >(
                                dictColumn->GetDict(),
                                newDictIndices );
                        table->AddColumn( leftKeys[ i ], newDictColumn );
                    }
                    else
                    {
                        auto column = std::make_shared< AriesColumn >();
                        column->AddDataBuffer( partitionResult.AllBuffers[ i ][ partitionIndex ]->ToAriesDataBuffer() );
                        table->AddColumn( leftKeys[ i ], column );
                    }
                }

                if ( !notEmptyTable )
                {
                    notEmptyTable = table.get();
                }
                partitionIndex ++;

                leftSubTables.push_back( std::move( table ) );
            }
            vector< AriesTableBlockUPtr > newOutputSubTables;
            for ( auto& table : leftSubTables)
            {
                if ( table )
                {
                    newOutputSubTables.push_back( std::move( table ) );
                }
                else
                {
                    newOutputSubTables.push_back( std::move( notEmptyTable->CloneWithNoContent() ) );
                }
            }
            std::swap( leftSubTables, newOutputSubTables );

            m_leftDataTable = nullptr;

            //partition right table
            for( int id : rightKeys )
            {
                if ( EncodeType::DICT == m_rightDataTable->GetColumnEncodeType( id ) )
                {
                    buffers.push_back( m_rightDataTable->GetDictEncodedColumnIndiceBuffer( id ) );
                }
                else
                {
                    buffers.push_back( m_rightDataTable->GetColumnBuffer( id ) );
                }
            }

            partitionResult = PartitionColumnDataEx( { buffers }, partitionCount, seed, &hashNullValue );
            buffers.clear();
            partitionIndex = 0;
            notEmptyTable = nullptr;
            for( const auto& part : partitionResult.AllIndices )
            {
                AriesIndicesArraySPtr indices = ConvertToAriesIndices( part );
                if ( indices->GetItemCount() == 0 )
                {
                    rightSubTables.emplace_back( nullptr );
                    partitionIndex ++;
                    continue;
                }
                AriesTableBlockUPtr table;
                std::vector< int32_t > columnsId;
                for ( const auto id : m_rightDataTable->GetAllColumnsId() )
                {
                    if ( std::find( rightKeys.cbegin(), rightKeys.cend(), id ) == rightKeys.cend() )
                    {
                        columnsId.emplace_back( id );
                    }
                }
                if ( !columnsId.empty() )
                {
                    table = m_rightDataTable->MakeTableByColumns( columnsId, false );
                    table->UpdateIndices( indices );
                }
                else
                {
                    table = std::make_unique< AriesTableBlock >();
                }

                for ( size_t i = 0; i < rightKeys.size(); i++ )
                {
                    if ( EncodeType::DICT == m_rightDataTable->GetColumnEncodeType( rightKeys[ i ] ) )
                    {
                        auto dictColumn = m_rightDataTable->GetDictEncodedColumn( rightKeys[ i ] );
                        auto newDictIndices = std::make_shared< AriesVariantIndices >();
                        newDictIndices->AddDataBuffer( partitionResult.AllBuffers[ i ][ partitionIndex ]->ToAriesDataBuffer() );
                        auto newDictColumn =
                            std::make_shared< AriesDictEncodedColumn >(
                                dictColumn->GetDict(),
                                newDictIndices );
                        table->AddColumn( rightKeys[ i ], newDictColumn );
                    }
                    else
                    {
                        auto column = std::make_shared< AriesColumn >();
                        column->AddDataBuffer( partitionResult.AllBuffers[ i ][ partitionIndex ]->ToAriesDataBuffer() );
                        table->AddColumn( rightKeys[ i ], column );
                    }
                }
                if ( !notEmptyTable )
                {
                    notEmptyTable = table.get();
                }
                partitionIndex ++;

                rightSubTables.push_back( std::move( table ) );
            }
            m_rightDataTable = nullptr;

            newOutputSubTables.clear();
            for ( auto& table : rightSubTables )
            {
                if ( table )
                {
                    newOutputSubTables.push_back( std::move( table ) );
                }
                else
                {
                    newOutputSubTables.push_back( std::move( notEmptyTable->CloneWithNoContent() ) );
                }
            }
            std::swap( rightSubTables, newOutputSubTables );
            ++seed;
        }
        assert( leftSubTables.size() == rightSubTables.size() );
        AriesTableBlockUPtr resultTable = nullptr;
        size_t totalRowCount = 0;
        while ( !leftSubTables.empty() )
        {
            assert( !rightSubTables.empty() );
            auto leftPartTable = std::move( leftSubTables.back() );
            auto rightPartTable = std::move( rightSubTables.back() );

            leftSubTables.pop_back();
            rightSubTables.pop_back();

            if ( leftPartTable->GetRowCount() == 0 )
            {
                continue;
            }
            if ( 0 == rightPartTable->GetRowCount() || IsConstFalseCondition() )
            {
                AriesTableBlockUPtr leftJoined = nullptr;
                AriesTableBlockUPtr rightJoined = nullptr;
                if ( !m_leftIds.empty() )
                {
                    leftJoined = leftPartTable->MakeTableByColumns( m_leftIds, false );
                    auto associatedArray = make_shared< AriesInt32Array >( leftJoined->GetRowCount() );
                    aries_acc::InitSequenceValue( associatedArray );
                    leftJoined->UpdateIndices( associatedArray );
                    leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );
                }

                if ( !m_rightIds.empty() )
                {
                    auto nullIndex = CreateNullIndex( leftPartTable->GetRowCount() );
                    rightJoined = rightPartTable->MakeTableByColumns( m_rightIds, false );
                    rightJoined->UpdateIndices( nullIndex, true );
                    rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );
                }

                AriesTableBlockUPtr partResult;
                if ( leftJoined )
                {
                    if ( rightJoined )
                    {
                        leftJoined->MergeTable( move( rightJoined ) );
                    }
                    partResult = move( leftJoined );
                }
                else if ( rightJoined )
                {
                    partResult = move( rightJoined );
                }
                else
                {
                    //select 1 from ...
                    totalRowCount += leftPartTable->GetRowCount();
                    continue;
                }
                if ( resultTable )
                {
                    resultTable->AddBlock( std::move( partResult ) );
                }
                else
                {
                    resultTable = std::move( partResult );
                }
                continue;
            }

            const aries_engine::AriesTableBlockUPtr& tableToBuildHash = ( m_hashJoinType == HashJoinType::LeftAsHash ? leftPartTable : rightPartTable );

            bool is_char_type_key = false;
            for ( const auto key : m_uniqueKeys )
            {
                if ( tableToBuildHash->GetColumnType( key ).DataType.ValueType == AriesValueType::CHAR )
                {
                    is_char_type_key = true;
                    break;
                }
            }

            m_joinHelper->MaterializeColumns( leftPartTable, rightPartTable );

            auto canUseDict = CheckHashJoinConditionForDict( tableToBuildHash, m_uniqueKeys, m_hashJoinType == HashJoinType::LeftAsHash ? rightPartTable : leftPartTable, m_hashValueKeys );

            AriesJoinResult joinResult;
            if ( !is_char_type_key && m_uniqueKeys.size() == 1 )
            {
                auto hashTable = aries_acc::BuildHashTable( tableToBuildHash, m_uniqueKeys[ 0 ], canUseDict[ 0 ] );
                joinResult = m_joinHelper->ProcessHashLeftJoin( hashTable, leftPartTable, nullptr, rightPartTable, nullptr, m_hashValueKeys[ 0 ], canUseDict[ 0 ], m_hashJoinType == HashJoinType::LeftAsHash );
                aries_acc::ReleaseHashTable( hashTable );
            }
            else
            {
                auto hashTable = aries_acc::BuildHashTable( tableToBuildHash, m_uniqueKeys, canUseDict );
                joinResult = m_joinHelper->ProcessHashLeftJoin( hashTable, leftPartTable, nullptr, rightPartTable, nullptr, m_hashValueKeys, canUseDict, m_hashJoinType == HashJoinType::LeftAsHash );
                aries_acc::ReleaseHashTable( hashTable );
            }

            auto joinKeyPairs = boost::get< JoinPair >( joinResult );

            AriesTableBlockUPtr leftJoined = nullptr;
            if ( !m_leftIds.empty() )
            {
                leftJoined = leftPartTable->MakeTableByColumns( m_leftIds, false );
                leftJoined->UpdateIndices( joinKeyPairs.LeftIndices );
                leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );
            }

            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !m_rightIds.empty() )
            {
                rightJoined = rightPartTable->MakeTableByColumns( m_rightIds, false );
                rightJoined->UpdateIndices( joinKeyPairs.RightIndices, true );
                rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );
            }

            AriesTableBlockUPtr partResult;
            if ( leftJoined )
            {
                if ( rightJoined )
                {
                    leftJoined->MergeTable( move( rightJoined ) );
                }
                partResult = move( leftJoined );
            }
            else if ( rightJoined )
            {
                partResult = move( rightJoined );
            }
            else
            {
                //select 1 from ...
                totalRowCount += joinKeyPairs.JoinCount;
                continue;
            }

            if ( resultTable )
            {
                resultTable->AddBlock( std::move( partResult ) );
            }
            else
            {
                resultTable = std::move( partResult );
            }
        }

        if ( m_leftIds.empty() && m_rightIds.empty() )
        {
            result.TableBlock = GenerateTableWithRowCountOnly( totalRowCount );
        }
        else
        {
            result.TableBlock = std::move( resultTable );
        }

        result.Status = AriesOpNodeStatus::END;
        return result;
    }

    AriesOpResult AriesJoinNode::FullJoinGracePartitioned()
    {
        AriesOpResult result{ AriesOpNodeStatus::ERROR, nullptr };

        CacheAllLeftTable();
        if ( !m_leftDataTable )
        {
            return result;
        }
        CacheAllRightTable();
        if ( !m_rightDataTable )
        {
            m_leftDataTable = nullptr;
            return result;
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesTableBlockUPtr leftOutTable;
        AriesTableBlockUPtr rightOutTable;
        int tupleNum = 0;

        ARIES_FUNC_LOG_BEGIN;

        if ( 0 == m_leftDataTable->GetRowCount() )
        {
            tupleNum = m_rightDataTable->GetRowCount();
            if ( tupleNum > 0 )
            {
                if ( !m_leftIds.empty() )
                {
                    leftOutTable = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                    leftOutTable->ResetAllStats();

                    leftOutTable->UpdateIndices( CreateNullIndex( tupleNum ), true );

                    const auto& tmpTableStats = leftOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, empty leftOutTable UpdateIndices" );
                    m_leftTableStats += tmpTableStats;
                }
                if ( !m_rightIds.empty() )
                {
                    rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
                }
            }
        }
        else if ( 0 == m_rightDataTable->GetRowCount() )
        {
            tupleNum = m_leftDataTable->GetRowCount();
            if ( tupleNum > 0 )
            {
                if ( !m_leftIds.empty() )
                {
                    leftOutTable = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                }
                if ( !m_rightIds.empty() )
                {
                    rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
                    rightOutTable->ResetAllStats();

                    rightOutTable->UpdateIndices( CreateNullIndex( tupleNum ), true );

                    const auto& tmpTableStats = rightOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, empty rightOutTable UpdateIndices" );
                    m_rightTableStats += tmpTableStats;
                }
            }
        }
        else
        {
            m_leftDataTable->ResetAllStats();
            m_rightDataTable->ResetAllStats();

            if ( IsConstFalseCondition() )
            {
                result = GetFullJoinResultOfFalseCondition();

                ARIES_FUNC_LOG_END;
        #ifdef ARIES_PROFILE
                m_opTime += t.end();
        #endif
                return result;
            }

            auto result = m_joinHelper->ProcessGracePartitioned( m_leftDataTable, m_rightDataTable );

            auto tmpTableStats = m_leftDataTable->GetStats();
            tmpTableStats.Print( "SemiOrAntiJoinGracePartitioned, process left_table_block" );
            m_leftTableStats += tmpTableStats;

            tmpTableStats = m_rightDataTable->GetStats();
            tmpTableStats.Print( "SemiOrAntiJoinGracePartitioned, process right_table_block" );
            m_rightTableStats += tmpTableStats;

            m_leftDataTable  = nullptr;
            m_rightDataTable = nullptr;

    #ifdef ARIES_PROFILE
            m_opTime += t.end();
    #endif
            return result;
        }
        if( tupleNum > 0 )
        {
            if ( leftOutTable )
            {
                leftOutTable->UpdateColumnIds( m_leftOutColumnIdMap );
                if ( rightOutTable )
                {
                    rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                    leftOutTable->MergeTable( std::move( rightOutTable ) );
                }
                result.TableBlock = std::move( leftOutTable );
            }
            else if ( rightOutTable )
            {
                rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                result.TableBlock = std::move( rightOutTable );
            }
            else
            {
                //sql: select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( tupleNum );
            }
        }
        else
        {
            result.TableBlock = GenerateEmptyTable();
        }

        result.Status = AriesOpNodeStatus::END;

        ARIES_FUNC_LOG_END;
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        return result;
    }

    AriesOpResult AriesJoinNode::LeftJoinGracePartitioned()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        if ( 0 == m_leftDataTable->GetRowCount() )
            return { AriesOpNodeStatus::END, GenerateEmptyTable() };
        
        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif

        if ( 0 == m_rightDataTable->GetRowCount() || IsConstFalseCondition() )
        {
            AriesTableBlockUPtr leftJoined = nullptr;
            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !m_leftIds.empty() )
            {
                leftJoined = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                leftJoined->UpdateColumnIds( m_leftOutColumnIdMap );
            }

            if ( !m_rightIds.empty() )
            {
                auto rightEmptyTable = m_rightDataTable->CloneWithNoContent();
                auto nullIndex = CreateNullIndex( m_leftDataTable->GetRowCount() );
                rightJoined = rightEmptyTable->MakeTableByColumns( m_rightIds, false );
                rightJoined->UpdateIndices( nullIndex, true );
                rightJoined->UpdateColumnIds( m_rightOutColumnIdMap );
            }

            AriesOpResult result { AriesOpNodeStatus::END, nullptr };
            if ( leftJoined )
            {
                if ( rightJoined )
                {
                    leftJoined->MergeTable( move( rightJoined ) );
                }
                result.TableBlock = move( leftJoined );
            }
            else if ( rightJoined )
            {
                result.TableBlock = move( rightJoined );
            }
            else
            {
                //sql: select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( m_leftDataTable->GetRowCount() );
            }
            return result;
        }

        auto result = m_joinHelper->ProcessGracePartitioned( m_leftDataTable, m_rightDataTable );

        auto tmpTableStats = m_leftDataTable->GetStats();
        tmpTableStats.Print( "SemiOrAntiJoinGracePartitioned, process left_table_block" );
        m_leftTableStats += tmpTableStats;

        tmpTableStats = m_rightDataTable->GetStats();
        tmpTableStats.Print( "SemiOrAntiJoinGracePartitioned, process right_table_block" );
        m_rightTableStats += tmpTableStats;

        m_leftDataTable  = nullptr;
        m_rightDataTable = nullptr;

#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif

        return result;
    }

    AriesOpResult AriesJoinNode::LeftJoinGetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;
        AriesOpResult result
                { AriesOpNodeStatus::ERROR, nullptr };

        if ( !CacheAllLeftToSubTables(SUBTABLE_COUNT) )
        {
            return result;
        }
        CacheAllRightTable();
        if( !m_rightDataTable )
        {
            return result;
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif

        auto leftDataBlock = GetNextCachedSubTable(m_leftSubTablesCache);
        if (leftDataBlock == nullptr)
        {
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
            return {AriesOpNodeStatus::END, GenerateEmptyTable()};
        }

        AriesIndices rightOutIndices;
        auto leftOutTable  = std::make_unique< AriesTableBlock >();
        // rightDataTable contains complete data of right table,
        auto rightOutTable = std::make_unique< AriesTableBlock >();
        if ( !m_rightIds.empty() )
        {
            rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
            rightOutTable->ResetAllStats();
        }

        int64_t totalTupleNum = 0;

        ARIES_FUNC_LOG_BEGIN;
        while( leftDataBlock && !IsCurrentThdKilled() )
        {
            int64_t leftTupleNum = leftDataBlock->GetRowCount();
            if( leftTupleNum > 0 )
            {
                if ( 0 == m_rightDataTable->GetRowCount() || IsConstFalseCondition() )
                {
                    totalTupleNum += leftTupleNum;
                    if ( !m_leftIds.empty() )
                    {
                        auto tmpTable = leftDataBlock->MakeTableByColumns( m_leftIds, false );
                        leftOutTable->AddBlock( std::move( tmpTable ) );
                    }
                    if ( !m_rightIds.empty() )
                    {
                        rightOutIndices.AddIndices( CreateNullIndex( leftTupleNum ) );
                    }

                    leftDataBlock = GetNextCachedSubTable(m_leftSubTablesCache);
                    continue;
                }
                // join a sub block of left table with the whole right table
                leftDataBlock->ResetAllStats();
                m_rightDataTable->ResetAllStats();

                AriesJoinResult joinResult = m_joinHelper->Process( leftDataBlock, m_rightDataTable );

                auto tmpTableStats = leftDataBlock->GetStats();
                tmpTableStats.Print( "AriesJoinNode::LeftJoinGetNext, process leftDataBlock" );
                m_leftTableStats += tmpTableStats;

                tmpTableStats = m_rightDataTable->GetStats();
                tmpTableStats.Print( "AriesJoinNode::LeftJoinGetNext, process m_rightDataTable" );
                m_rightTableStats += tmpTableStats;

                JoinPair keyPairs = boost::get< JoinPair >( joinResult );
                int tupleNum = keyPairs.JoinCount;
                if( tupleNum > 0 )
                {
                    totalTupleNum += tupleNum;
                    if ( !m_leftIds.empty() )
                    {
                        auto tmpTable = leftDataBlock->MakeTableByColumns( m_leftIds, false );
                        tmpTable->UpdateIndices( keyPairs.LeftIndices );

                        const auto& tmpTableStats = tmpTable->GetStats();
                        tmpTableStats.Print( "AriesJoinNode::LeftJoinGetNext, leftDataBlock UpdateIndices" );
                        m_leftTableStats += tmpTable->GetStats();

                        leftOutTable->AddBlock( std::move( tmpTable ) );
                    }

                    if ( !m_rightIds.empty() )
                    {
                        rightOutIndices.AddIndices( keyPairs.RightIndices );
                    }
                }
            }
            leftDataBlock = GetNextCachedSubTable(m_leftSubTablesCache);
        }

        if (IsCurrentThdKilled())
        {
            LOG(INFO) << "thread was killed in AriesJoinNode::LeftJoinGetNext";
            SendKillMessage();
        }

        result.Status = AriesOpNodeStatus::END;
        result.TableBlock = std::make_unique<AriesTableBlock>();
        if ( totalTupleNum > 0 )
        {
            if ( !m_outputColumnIds.empty() )
            {
                // adjust output column ids and indices
                if ( !m_leftIds.empty() )
                {
                    leftOutTable->UpdateColumnIds( m_leftOutColumnIdMap );
                }

                if ( !m_rightIds.empty() )
                {
                    rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                    rightOutTable->UpdateIndices( rightOutIndices.GetIndices(), true );

                    const auto& tmpTableStats = rightOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::LeftJoinGetNext, rightOutTable" );
                    m_rightTableStats += tmpTableStats;
                }

                if ( !m_leftIds.empty() )
                {
                    if ( !m_rightIds.empty() )
                    {
                        leftOutTable->MergeTable( std::move( rightOutTable ) );
                    }
                    result.TableBlock = std::move( leftOutTable );
                }
                else if ( !m_rightIds.empty() )
                {
                    result.TableBlock = std::move( rightOutTable );
                }
            }
            else
            {
                //sql: select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( totalTupleNum );
            }
        }
        else
        {
            // empty table, still need column infos
            result.TableBlock = GenerateEmptyTable();
        }

        ARIES_FUNC_LOG_END;
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif

        CacheNodeData( result.TableBlock );

        return result;
    }

    AriesOpResult AriesJoinNode::RightJoinGetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;

        AriesOpResult result{ AriesOpNodeStatus::ERROR, nullptr };

        if ( !CacheAllRightToSubTables(SUBTABLE_COUNT) )
        {
            return result;
        }

        CacheAllLeftTable();
        if( !m_leftDataTable )
        {
            return result;
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        auto rightDataBlock = GetNextCachedSubTable(m_rightSubTablesCache);
        if ( rightDataBlock == nullptr )
        {
            return {AriesOpNodeStatus::END, GenerateEmptyTable()};
        }

        AriesIndices leftOutIndices;
        // leftDataTable contains complete data of left table,
        auto leftOutTable = std::make_unique< AriesTableBlock >();
        if ( !m_leftIds.empty() )
        {
            leftOutTable  = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
            leftOutTable->ResetAllStats();
        }
        auto rightOutTable = std::make_unique< AriesTableBlock >();

        int64_t totalTupleNum = 0;

        ARIES_FUNC_LOG_BEGIN;
        // join a sub block of right table with the whole left table
        while( rightDataBlock && !IsCurrentThdKilled() )
        {
            int64_t rightTupleNum = rightDataBlock->GetRowCount();
            if( rightTupleNum > 0 )
            {
                if ( 0 == m_leftDataTable->GetRowCount() || IsConstFalseCondition() )
                {
                    totalTupleNum += rightTupleNum;
                    if ( !m_rightIds.empty() )
                    {
                        auto tmpTable = rightDataBlock->MakeTableByColumns( m_rightIds, false );
                        rightOutTable->AddBlock( std::move( tmpTable ) );
                    }
                    if ( !m_leftIds.empty() )
                    {
                        leftOutIndices.AddIndices( CreateNullIndex( rightTupleNum ) );
                    }

                    rightDataBlock = GetNextCachedSubTable(m_rightSubTablesCache);
                    continue;
                }

                m_leftDataTable->ResetAllStats();
                rightDataBlock->ResetAllStats();

                AriesJoinResult joinResult = m_joinHelper->Process( m_leftDataTable, rightDataBlock );

                auto tmpTableStats = m_leftDataTable->GetStats();
                tmpTableStats.Print( "AriesJoinNode::RightJoinGetNext, process m_leftDataTable" );
                m_leftTableStats += tmpTableStats;

                tmpTableStats = rightDataBlock->GetStats();
                tmpTableStats.Print( "AriesJoinNode::RightJoinGetNext, process rightDataBlock" );
                m_rightTableStats += tmpTableStats;

                JoinPair keyPairs = boost::get< JoinPair >( joinResult );
                int tupleNum = keyPairs.JoinCount;
                if( tupleNum > 0 )
                {
                    totalTupleNum += tupleNum;
                    if ( !m_rightIds.empty() )
                    {
                        auto tmpTable = rightDataBlock->MakeTableByColumns( m_rightIds, false );
                        tmpTable->ResetAllStats();
                        tmpTable->UpdateIndices( keyPairs.RightIndices );

                        const auto& tmpTableStats = tmpTable->GetStats();
                        tmpTableStats.Print( "AriesJoinNode::RightJoinGetNext, rightDataBlock UpdateIndices" );
                        m_rightTableStats += tmpTableStats;

                        rightOutTable->AddBlock( std::move( tmpTable ) );
                    }

                    if ( !m_leftIds.empty() )
                    {
                        leftOutIndices.AddIndices( keyPairs.LeftIndices );
                    }
                }
            }
            rightDataBlock = GetNextCachedSubTable(m_rightSubTablesCache);
        }

        if (IsCurrentThdKilled())
        {
            LOG(INFO) << "thread was killed in AriesJoinNode::RightJoinGetNext";
            SendKillMessage();
        }

        result.Status = AriesOpNodeStatus::END;
        result.TableBlock = std::make_unique<AriesTableBlock>();
        if ( totalTupleNum > 0 )
        {
            if ( !m_outputColumnIds.empty() )
            {
                // adjust output column ids and indices
                if ( !m_leftIds.empty() )
                {
                    leftOutTable->UpdateColumnIds( m_leftOutColumnIdMap );
                    leftOutTable->UpdateIndices( leftOutIndices.GetIndices(), true );
                }

                if ( !m_rightIds.empty() )
                {
                    rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                }

                if ( !m_leftIds.empty() )
                {
                    if ( !m_rightIds.empty() )
                    {
                        leftOutTable->MergeTable( std::move( rightOutTable ) );
                    }

                    auto tmpTableStats = leftOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::RightJoinGetNext, leftOutTable" );
                    m_leftTableStats += tmpTableStats;

                    result.TableBlock = std::move( leftOutTable );
                }
                else if ( !m_rightIds.empty() )
                {
                    result.TableBlock = std::move( rightOutTable );
                }
            }
            else
            {
                //sql: select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( totalTupleNum );
            }
        }
        else
        {
            result.TableBlock = GenerateEmptyTable();
        }

        ARIES_FUNC_LOG_BEGIN;
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        CacheNodeData( result.TableBlock );

        return result;
    }

    AriesOpResult AriesJoinNode::GetFullJoinResultOfFalseCondition()
    {
        AriesOpResult result{ AriesOpNodeStatus::ERROR, nullptr };

        AriesTableBlockUPtr leftOutTable;
        AriesTableBlockUPtr rightOutTable;

        auto leftTupleNum = m_leftDataTable->GetRowCount();
        auto rightTupleNum = m_rightDataTable->GetRowCount();
        int tupleNum = leftTupleNum + rightTupleNum;
        if ( !m_leftIds.empty() )
        {
            leftOutTable = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
            auto tmpLeftIndices = std::make_shared< AriesInt32Array >( leftTupleNum );
            aries_acc::InitSequenceValue( tmpLeftIndices );
            leftOutTable->UpdateIndices( tmpLeftIndices, true );

            auto leftOutTable2 = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
            leftOutTable2->ResetAllStats();
            leftOutTable2->UpdateIndices( CreateNullIndex( rightTupleNum ), true );
            const auto& tmpTableStats = leftOutTable2->GetStats();
            m_leftTableStats += tmpTableStats;

            leftOutTable->AddBlock( std::move( leftOutTable2 ) );
        }
        if ( !m_rightIds.empty() )
        {
            rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
            rightOutTable->ResetAllStats();
            rightOutTable->UpdateIndices( CreateNullIndex( leftTupleNum ), true );
            const auto& tmpTableStats = rightOutTable->GetStats();
            m_rightTableStats += tmpTableStats;

            auto rightOutTable2 = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
            auto tmpRightIndices = std::make_shared< AriesInt32Array >( rightTupleNum );
            aries_acc::InitSequenceValue( tmpRightIndices );
            rightOutTable2->UpdateIndices( tmpRightIndices, true );

            rightOutTable->AddBlock( std::move( rightOutTable2 ) );
        }

        if ( leftOutTable )
        {
            leftOutTable->UpdateColumnIds( m_leftOutColumnIdMap );
            if ( rightOutTable )
            {
                rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                leftOutTable->MergeTable( std::move( rightOutTable ) );
            }
            result.TableBlock = std::move( leftOutTable );
        }
        else if ( rightOutTable )
        {
            rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
            result.TableBlock = std::move( rightOutTable );
        }
        else
        {
            //sql: select 1 from ...
            result.TableBlock = GenerateTableWithRowCountOnly( tupleNum );
        }
        result.Status = AriesOpNodeStatus::END;

        return result;
    }

    AriesOpResult AriesJoinNode::FullJoinGetNext()
    {
        AriesOpResult result{ AriesOpNodeStatus::ERROR, nullptr };

        CacheAllLeftTable();
        if ( !m_leftDataTable )
        {
            return result;
        }
        CacheAllRightTable();
        if ( !m_rightDataTable )
        {
            m_leftDataTable = nullptr;
            return result;
        }
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesTableBlockUPtr leftOutTable;
        AriesTableBlockUPtr rightOutTable;
        int tupleNum = 0;

        ARIES_FUNC_LOG_BEGIN;

        if ( 0 == m_leftDataTable->GetRowCount() )
        {
            tupleNum = m_rightDataTable->GetRowCount();
            if ( tupleNum > 0 )
            {
                if ( !m_leftIds.empty() )
                {
                    leftOutTable = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                    leftOutTable->ResetAllStats();

                    leftOutTable->UpdateIndices( CreateNullIndex( tupleNum ), true );

                    const auto& tmpTableStats = leftOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, empty leftOutTable UpdateIndices" );
                    m_leftTableStats += tmpTableStats;
                }
                if ( !m_rightIds.empty() )
                {
                    rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
                }
            }
        }
        else if ( 0 == m_rightDataTable->GetRowCount() )
        {
            tupleNum = m_leftDataTable->GetRowCount();
            if ( tupleNum > 0 )
            {
                if ( !m_leftIds.empty() )
                {
                    leftOutTable = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                }
                if ( !m_rightIds.empty() )
                {
                    rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
                    rightOutTable->ResetAllStats();

                    rightOutTable->UpdateIndices( CreateNullIndex( tupleNum ), true );

                    const auto& tmpTableStats = rightOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, empty rightOutTable UpdateIndices" );
                    m_rightTableStats += tmpTableStats;
                }
            }
        }
        else
        {
            m_leftDataTable->ResetAllStats();
            m_rightDataTable->ResetAllStats();

            if ( IsConstFalseCondition() )
            {
                result = GetFullJoinResultOfFalseCondition();

                ARIES_FUNC_LOG_END;
        #ifdef ARIES_PROFILE
                m_opTime += t.end();
        #endif
                return result;
            }

            AriesJoinResult joinResult = m_joinHelper->Process( m_leftDataTable, m_rightDataTable );

            auto tmpTableStats = m_leftDataTable->GetStats();
            tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, process m_leftDataTable" );
            m_leftTableStats += tmpTableStats;

            tmpTableStats = m_rightDataTable->GetStats();
            tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, process m_rightDataTable" );
            m_rightTableStats += tmpTableStats;

            JoinPair keyPairs = boost::get< JoinPair >( joinResult );
            tupleNum = keyPairs.JoinCount;

            if( tupleNum > 0 )
            {
                // adjust output column ids and indices
                if ( !m_leftIds.empty() )
                {
                    leftOutTable  = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                    leftOutTable->ResetAllStats();

                    leftOutTable->UpdateIndices( keyPairs.LeftIndices, true );

                    const auto& tmpTableStats = leftOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, leftOutTable UpdateIndices" );
                    m_leftTableStats += tmpTableStats;
                }

                if ( !m_rightIds.empty() )
                {
                    rightOutTable = m_rightDataTable->MakeTableByColumns( m_rightIds, false );
                    rightOutTable->ResetAllStats();

                    rightOutTable->UpdateIndices( keyPairs.RightIndices, true );

                    const auto& tmpTableStats = rightOutTable->GetStats();
                    tmpTableStats.Print( "AriesJoinNode::FullJoinGetNext, rightOutTable UpdateIndices" );
                    m_rightTableStats += tmpTableStats;
                }

            }
        }
        if( tupleNum > 0 )
        {
            if ( leftOutTable )
            {
                leftOutTable->UpdateColumnIds( m_leftOutColumnIdMap );
                if ( rightOutTable )
                {
                    rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                    leftOutTable->MergeTable( std::move( rightOutTable ) );
                }
                result.TableBlock = std::move( leftOutTable );
            }
            else if ( rightOutTable )
            {
                rightOutTable->UpdateColumnIds( m_rightOutColumnIdMap );
                result.TableBlock = std::move( rightOutTable );
            }
            else
            {
                //sql: select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( tupleNum );
            }
        }
        else
        {
            result.TableBlock = GenerateEmptyTable();
        }

        result.Status = AriesOpNodeStatus::END;

        ARIES_FUNC_LOG_END;
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        return result;
    }

    size_t EstimateHashSemiAntiJoinPerRowMemOccupancy(
        const aries_engine::AriesTableBlockUPtr& table_block,
        const std::vector< int >& columns_id )
    {
        size_t usage = 0;

        for( const auto& id : columns_id )
        {
            auto block = table_block->GetColumnBuffer( id );
            if( block->GetItemCount() == 0 )
                return 0;
            usage += block->GetItemSizeInBytes();
        }
        // add mem occupancy for output
        usage += sizeof( index_t );
        return usage;
    }

    size_t GetHashSemiAntiJoinPartitionCount(
        AriesTableBlockUPtr& hashDataTable,
        const vector< int >& hashColumnIds,
        AriesTableBlockUPtr& valueDataTable,
        const vector< int >& valueColumnIds )
    {
        size_t partitionCount = 1;

        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        auto valueTableMemOccupancy = EstimateHashSemiAntiJoinPerRowMemOccupancy( valueDataTable, valueColumnIds );

        //至少预留value table一个分块，ARIES_DATA_BLOCK_ROW_SIZE
        size_t minValueTableRowCount = std::min( ARIES_DATA_BLOCK_ROW_SIZE, valueDataTable->GetRowCount() );
        valueTableMemOccupancy *= minValueTableRowCount;
        size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity() * MAX_RATIO;

        //计算hash table是否能整体放入显存
        auto hashTableMemOccupancy = EstimateBuildHashTableMemOccupancy( hashDataTable, hashColumnIds );

        double currentRatio = hashTableMemOccupancy / available;
        if( currentRatio > MAX_RATIO )
            partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

        return partitionCount;
    }

    AriesOpResult AriesJoinNode::SemiOrAntiJoinCheckShortcut()
    {
        if( m_rightRowCount == 0 || IsConstFalseCondition() )
        {
            if( m_joinType == AriesJoinType::SEMI_JOIN )
                return { AriesOpNodeStatus::END, GetEmptyTable() };
            else
            {
                if( !m_leftIds.empty() )
                {
                    auto leftOutTable = m_leftDataTable->MakeTableByColumns( m_leftIds, false );
                    leftOutTable->UpdateColumnIds( m_leftOutColumnIdMap );
                    return { AriesOpNodeStatus::END, std::move( leftOutTable ) };
                }
                else
                {
                    // select 1 from ...
                    return { AriesOpNodeStatus::END, GenerateTableWithRowCountOnly( m_leftDataTable->GetRowCount() ) };
                }
            }
        }
        return { AriesOpNodeStatus::CONTINUE, nullptr };
    }

    AriesOpResult AriesJoinNode::SemiOrAntiJoinReadData()
    {
        // 分别获取左右表所有数据
        CacheAllLeftTable();
        if( !m_leftDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };
        if( 0 == m_leftRowCount )
            return { AriesOpNodeStatus::END, GetEmptyTable() };

        CacheAllRightTable();
        if( !m_rightDataTable )
            return { AriesOpNodeStatus::ERROR, nullptr };

        auto shortcutResult = SemiOrAntiJoinCheckShortcut();
        if ( AriesOpNodeStatus::END == shortcutResult.Status )
            return shortcutResult;

        m_leftDataTable->ResetAllStats();
        m_rightDataTable->ResetAllStats();

        return { AriesOpNodeStatus::CONTINUE, nullptr };
    }

    AriesOpResult AriesJoinNode::HashSemiOrAntiJoinGracePartitionTable(
        uint32_t seed,
        size_t &hashTablePartitionCount,
        vector< AriesTableBlockUPtr > &leftSubTables,
        vector< AriesTableBlockUPtr > &rightSubTables )
    {
        if( m_hashJoinType == HashJoinType::LeftAsHash )
            hashTablePartitionCount =
                ::GetHashSemiAntiJoinPartitionCount( m_leftDataTable, m_uniqueKeys,
                                                     m_rightDataTable, m_hashValueKeys );
        else
            hashTablePartitionCount =
                ::GetHashSemiAntiJoinPartitionCount( m_rightDataTable, m_uniqueKeys,
                                                     m_leftDataTable, m_hashValueKeys );

        //根据是否需要partition，将partition结果存入leftSubTables和rightSubTables
        if( m_hashJoinType == HashJoinType::LeftAsHash )
        {
            bool rightHasNullValues = false;
            GraceHashPartitionTable( m_rightDataTable,
                                     m_hashValueKeys,
                                     hashTablePartitionCount,
                                     seed,
                                     rightSubTables,
                                     &rightHasNullValues );

            if ( m_joinType == AriesJoinType::ANTI_JOIN && m_joinHelper->IsNotIn() && rightHasNullValues )
            {
                AriesOpResult result;
                if ( !m_leftIds.empty() )
                    result.TableBlock = GenerateEmptyTable();
                else
                    result.TableBlock = GenerateTableWithRowCountOnly( 0 );
                result.Status = AriesOpNodeStatus::END;
                return result;
            }

            GraceHashPartitionTable( m_leftDataTable,
                                     m_uniqueKeys,
                                     hashTablePartitionCount,
                                     seed,
                                     leftSubTables );
        }
        else
        {
            GraceHashPartitionTable( m_leftDataTable,
                                     m_hashValueKeys,
                                     hashTablePartitionCount,
                                     seed,
                                     leftSubTables );

            GraceHashPartitionTable( m_rightDataTable,
                                     m_uniqueKeys,
                                     hashTablePartitionCount,
                                     seed,
                                     rightSubTables );
        }
        return { AriesOpNodeStatus::CONTINUE, nullptr };
    }

    /*
    To comply with the SQL standard,
    IN returns NULL not only if the expression on the left hand side is NULL,
    but also if no match is found in the list and one of the expressions in the list is NULL.

    * expr NOT IN (value,...)
        This is the same as NOT (expr IN (value,...))

    mysql result:
    mysql>  select 10 NOT IN (1,NULL);
    +--------------------+
    | 10 NOT IN (1,NULL) |
    +--------------------+
    |               NULL |
    +--------------------+

    mysql>  select 10 NOT IN (1,NULL,10);
    +-----------------------+
    | 10 NOT IN (1,NULL,10) |
    +-----------------------+
    |                     0 |
    +-----------------------+
    */
    AriesOpResult AriesJoinNode::HashSemiOrAntiJoinGracePartitioned()
    {
        assert( m_hashJoinType != HashJoinType::None );
        assert( m_uniqueKeys.size() == 1 );
        assert( m_hashValueKeys.size() == 1 );
        assert( m_rightIds.empty() );
        m_usedHashJoin = true;

        auto result = SemiOrAntiJoinReadData();
        if ( AriesOpNodeStatus::CONTINUE != result.Status )
            return result;

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif

        uint32_t seed = 0;
        size_t hashTablePartitionCount;
        //根据是否需要partition，将partition结果存入leftSubTables和rightSubTables
        vector< AriesTableBlockUPtr > leftSubTables;
        vector< AriesTableBlockUPtr > rightSubTables;
        result = HashSemiOrAntiJoinGracePartitionTable(
                     seed,
                     hashTablePartitionCount,
                     leftSubTables,
                     rightSubTables );
        if ( AriesOpNodeStatus::CONTINUE != result.Status )
            return result;

        result.Status = AriesOpNodeStatus::ERROR;

        if( 1 != hashTablePartitionCount )
            ++seed;

        const int64_t MIN_TABLE_ROW_COUNT = 100;
        size_t resultRowCount = 0;

        while( !leftSubTables.empty() )
        {
            auto leftPartTable = std::move( leftSubTables.back() );
            leftSubTables.pop_back();

            auto rightPartTable = std::move( rightSubTables.back() );
            rightSubTables.pop_back();

            if ( 0 == leftPartTable->GetRowCount() )
            {
                continue;
            }
            if( 0 == rightPartTable->GetRowCount() || IsConstFalseCondition() )
            {
                if( m_joinType == AriesJoinType::ANTI_JOIN )
                {
                    resultRowCount += leftPartTable->GetRowCount();
                    if( !m_leftIds.empty() )
                    {
                        auto partOutputTable = leftPartTable->MakeTableByColumns( m_leftIds, false );
                        partOutputTable->UpdateColumnIds( m_leftOutColumnIdMap );
                        if( !result.TableBlock )
                            result.TableBlock = std::move( partOutputTable );
                        else
                            result.TableBlock->AddBlock( std::move( partOutputTable ) );
                    }
                }
                continue;
            }

            AriesHashTableUPtr tmpHashTable;
            AriesJoinResult tmpJoinResult;
            try
            {
                AriesIndicesArraySPtr left_table_indices = leftPartTable->GetTheSharedIndiceForColumns( m_leftIds );

                AriesTableBlockUPtr resultTableBlock;
                AriesJoinResult joinResult;

                if( m_hashJoinType == HashJoinType::LeftAsHash )
                {
                    auto canUseDict = CheckHashJoinConditionForDict( leftPartTable, m_uniqueKeys, rightPartTable, m_hashValueKeys );
                    tmpHashTable = aries_acc::BuildHashTable( leftPartTable, m_uniqueKeys[ 0 ], canUseDict[ 0 ] );
                    joinResult = m_joinHelper->ProcessHalfJoinLeftAsHash( tmpHashTable,
                                                                          leftPartTable,
                                                                          left_table_indices,
                                                                          rightPartTable,
                                                                          m_hashValueKeys[ 0 ], canUseDict[ 0 ] );
                }
                else
                {
                    auto canUseDict = CheckHashJoinConditionForDict( rightPartTable, m_uniqueKeys, leftPartTable, m_hashValueKeys );
                    tmpHashTable = aries_acc::BuildHashTable( rightPartTable, m_uniqueKeys[ 0 ], canUseDict[ 0 ] );
                    joinResult = m_joinHelper->ProcessHalfJoinRightAsHash( tmpHashTable,
                                                                           leftPartTable,
                                                                           left_table_indices,
                                                                           rightPartTable,
                                                                           m_hashValueKeys[ 0 ], canUseDict[ 0 ] );
                }

                aries_acc::ReleaseHashTable( tmpHashTable );

                AriesInt32ArraySPtr indices = boost::get< AriesInt32ArraySPtr >( joinResult );
                size_t joinCount = indices->GetItemCount();
                if ( joinCount > 0 )
                {
                    resultRowCount += joinCount;
                    if ( !m_leftIds.empty() )
                    {
                        AriesTableBlockUPtr partOutputTable = leftPartTable->MakeTableByColumns( m_leftIds, false );
                        partOutputTable->ResetAllStats();
                        if( left_table_indices )
                            partOutputTable->ReplaceTheOnlyOneIndices( indices );
                        else
                            partOutputTable->UpdateIndices( indices );
                        partOutputTable->UpdateColumnIds( m_leftOutColumnIdMap );

                        const auto& tmpTableStats = partOutputTable->GetStats();
                        tmpTableStats.Print( "AriesJoinNode::SemiOrAntiHashJoinGetNext, leftJoined UpdateIndices" );
                        m_leftTableStats += tmpTableStats;
                        if( !result.TableBlock )
                            result.TableBlock = std::move( partOutputTable );
                        else
                            result.TableBlock->AddBlock( std::move( partOutputTable ) );
                    }
                }
            }
            catch ( AriesException& e )
            {
                aries_acc::ReleaseHashTable( tmpHashTable );
                if( e.errCode == ER_ENGINE_OUT_OF_MEMORY && leftPartTable->GetRowCount() > MIN_TABLE_ROW_COUNT )
                {
                    if( m_hashJoinType == HashJoinType::LeftAsHash )
                    {
                        GraceHashPartitionTable( leftPartTable, m_uniqueKeys, 2, seed, leftSubTables );
                        GraceHashPartitionTable( rightPartTable, m_hashValueKeys, 2, seed, rightSubTables );
                    }
                    else
                    {
                        GraceHashPartitionTable( rightPartTable, m_uniqueKeys, 2, seed, rightSubTables );
                        GraceHashPartitionTable( leftPartTable, m_hashValueKeys, 2, seed, leftSubTables );
                    }

                    ++seed;
                }
                else
                    throw e;
            }
        }

        if ( resultRowCount > 0 )
        {
            if ( m_leftIds.empty() )
            {
                //select 1 from ...
                result.TableBlock = GenerateTableWithRowCountOnly( resultRowCount );
            }
        }
        else
        {
            if ( !m_leftIds.empty() )
                result.TableBlock = GenerateEmptyTable();
            else
                result.TableBlock = GenerateTableWithRowCountOnly( resultRowCount );
        }

#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        m_leftTableStats += m_leftDataTable->GetStats();
        m_rightTableStats += m_rightDataTable->GetStats();

        m_leftDataTable  = nullptr;
        m_rightDataTable = nullptr;

        result.Status = AriesOpNodeStatus::END;
        return result;
    }

    AriesOpResult AriesJoinNode::SemiOrAntiJoinGracePartitioned()
    {
        auto result = SemiOrAntiJoinReadData();
        if ( AriesOpNodeStatus::CONTINUE != result.Status )
            return result;

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        result = m_joinHelper->ProcessGracePartitioned( m_leftDataTable, m_rightDataTable );

        auto tmpTableStats = m_leftDataTable->GetStats();
        tmpTableStats.Print( "SemiOrAntiJoinGracePartitioned, process left_table_block" );
        m_leftTableStats += tmpTableStats;

        tmpTableStats = m_rightDataTable->GetStats();
        tmpTableStats.Print( "SemiOrAntiJoinGracePartitioned, process right_table_block" );
        m_rightTableStats += tmpTableStats;

        m_leftDataTable  = nullptr;
        m_rightDataTable = nullptr;

#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif

        return result;
    }

    AriesOpResult AriesJoinNode::SemiOrAntiJoinGetNext()
    {
        auto result = SemiOrAntiJoinReadData();
        if ( AriesOpNodeStatus::CONTINUE != result.Status )
            return result;

        result.Status = AriesOpNodeStatus::ERROR;

#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif

        AriesJoinResult joinResult = m_joinHelper->Process( m_leftDataTable, m_rightDataTable );

        auto tmpTableStats = m_leftDataTable->GetStats();
        tmpTableStats.Print( "AriesJoinNode::SemiOrAntiJoinGetNext, process left_table_block" );
        m_leftTableStats += tmpTableStats;

        tmpTableStats = m_rightDataTable->GetStats();
        tmpTableStats.Print( "AriesJoinNode::SemiOrAntiJoinGetNext, process right_table_block" );
        m_rightTableStats += tmpTableStats;

        m_rightDataTable = nullptr;

        auto associated = boost::get< AriesBoolArraySPtr >( joinResult );

        if( IsCurrentThdKilled() )
        {
            LOG(INFO)<< "thread was killed in AriesJoinNode::SemiOrAntiJoinGetNext";
            SendKillMessage();
        }

        AriesInt32ArraySPtr joinAssociated = aries_acc::FilterAssociated( associated );
        if( joinAssociated->GetItemCount() > 0 )
        {
            if( !m_outputColumnIds.empty() )
            {
                m_leftDataTable->ResetAllStats();

                m_leftDataTable->UpdateIndices( joinAssociated );

                const auto& tmpTableStats = m_leftDataTable->GetStats();
                tmpTableStats.Print( "AriesJoinNode::SemiOrAntiJoinGetNext, left_table_block UpdateIndices" );
                m_leftTableStats += tmpTableStats;
            }
            else
            {
                // select 1 from ...
                m_leftDataTable = GenerateTableWithRowCountOnly( joinAssociated->GetItemCount() );
            }
        }
        else
        {
            m_leftDataTable= GenerateEmptyTable();
        }

#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif

        return
        {   AriesOpNodeStatus::END, move( m_leftDataTable )};
    }

    JSON AriesJoinNode::GetProfile() const
    {
        JSON filterStats;
        if (m_usedHashJoin && m_filterOpTime>0)
        {
            filterStats["type"] = "filter";
            filterStats["param"] = "hj";
            filterStats["time"] = m_filterOpTime;
            filterStats["memory"] = {JSON::parse(m_tableStats.ToJson(m_rowCount)), {"spool_id", m_spoolId}};
            if(m_leftSource)
                filterStats["children"] = {m_leftSource->GetProfile()};
        }

        JSON stats;
        stats["type"] = m_opName;
        stats["param"] = (m_usedHashJoin ? "hash" : "") + m_opParam;
        stats["time"] = m_opTime;
        stats["memory"] = {JSON::parse(m_leftTableStats.ToJson(m_leftRowCount)), 
                           JSON::parse(m_rightTableStats.ToJson(m_rightRowCount))};
        if(m_spoolId > -1)
            stats["memory"].push_back({{"spool_id", m_spoolId}});
        stats["children"] = {filterStats.empty() ? m_leftSource->GetProfile() : filterStats,
                             m_rightSource->GetProfile()};
        return stats;
    }

    void AriesJoinNode::SetHashJoinType( const HashJoinType& type )
    {
        m_hashJoinType = type;
    }

    void AriesJoinNode::SetHashJoinInfo( const HashJoinInfo& info )
    {
        m_hashJoinInfo = info;
    }

    void AriesJoinNode::SetUniqueKeys( const std::vector< int >& keys )
    {
        ARIES_ASSERT( !keys.empty(), "empty unique keys" );
        m_uniqueKeys.assign( keys.cbegin(), keys.cend() );
    }

    void AriesJoinNode::SetHashValueKeys( const std::vector< int >& keys )
    {
        m_hashValueKeys.assign( keys.cbegin(), keys.cend() );
    }

    void AriesJoinNode::SetIsNotIn( bool isNotIn )
    {
        assert( m_joinHelper );
        m_joinHelper->SetIsNotInFlag( isNotIn );
    }

    AriesTableBlockUPtr AriesJoinNode::GetEmptyTable() const
    {
        if ( m_outputColumnIds.empty() )
        {
            return std::make_unique< AriesTableBlock >();
        }
        else
        {
            auto left_table = m_leftSource->GetEmptyTable();
            auto right_table = m_rightSource->GetEmptyTable();

            auto left = left_table->GetColumnTypes( m_leftIds );
            auto right = right_table->GetColumnTypes( m_rightIds );
            index_t leftColIndex = 0;
            index_t rightColIndex = 0;

            std::vector< AriesColumnType > types;
            for ( int id : m_outputColumnIds )
            {
                types.push_back( id > 0 ? left[ leftColIndex++ ] : right[ rightColIndex++ ] );
            }

            return AriesTableBlock::CreateTableWithNoRows( types );
        }
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

