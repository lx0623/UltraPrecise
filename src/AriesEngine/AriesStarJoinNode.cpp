#include "AriesStarJoinNode.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE
AriesStarJoinNode::AriesStarJoinNode()
{
    m_opName = "star";
}
bool AriesStarJoinNode::Open()
{
    if ( !factSourceNode->Open() )
    {
        return false;
    }

    for ( const auto& source : dimensionSourceNodes )
    {
        if ( !source->Open() )
        {
            return false;
        }
    }

    rows_count.resize( 1 + dimensionSourceNodes.size() );
    for ( auto& row : rows_count )
    {
        row = 0;
    }

    return true;
}

void AriesStarJoinNode::Close()
{
    factSourceNode->Close();
    for ( const auto& source : dimensionSourceNodes )
    {
        source->Close();
    }
}


void AriesStarJoinNode::SetFactSourceNode( const AriesOpNodeSPtr& node )
{
    factSourceNode = node;
}

void AriesStarJoinNode::AddDimensionSourceNode( const AriesOpNodeSPtr& node )
{
    dimensionSourceNodes.emplace_back( node );
}


AriesOpResult AriesStarJoinNode::GetNext()
{
    AriesOpResult cachedResult = GetCachedResult();
    if ( AriesOpNodeStatus::END == cachedResult.Status )
        return cachedResult;
#ifdef ARIES_PROFILE
    aries::CPU_Timer timer;
#endif

    AriesOpResult result;
    bool hasEmptyDimensionTable = false;
    std::vector< AriesHashTableWrapper > hash_table_wrappers;
    std::vector< AriesIndicesArraySPtr > dimension_tables_indices;
    if ( dimension_tables.empty() )
    {
        for ( std::size_t i = 0; i < dimensionSourceNodes.size(); i++ )
        {
            auto& source = dimensionSourceNodes[ i ];
            auto table = ReadAllData( source );
            if( table->GetRowCount() == 0 )
            {
                result = { AriesOpNodeStatus::END, GetEmptyTable() };
                hasEmptyDimensionTable = true;
                break;
            }    
            rows_count[ i + 1 ] = table->GetRowCount();
#ifdef ARIES_PROFILE
            timer.begin();
#endif
            dimension_tables_indices.emplace_back( table->GetTheSharedIndiceForColumns( output_columns_ids[ i ] ) );
            AriesHashTableWrapper wrapper;
            vector< bool > can_use_dict;
            can_use_dict.resize( dimension_key_ids[ i ].size(), false );
            if ( dimension_key_ids[ i ].size() == 1 )
            {
                wrapper.Type = HashTableType::SingleKey;
                auto hash_table = aries_acc::BuildHashTable( table, dimension_key_ids[ i ][ 0 ], can_use_dict[ 0 ] );
                dimension_hash_tables.emplace_back( std::move( hash_table ) );
                wrapper.Ptr = dimension_hash_tables[ dimension_hash_tables.size() - 1 ].get();
            }
            else
            {
                wrapper.Type = HashTableType::MultipleKeys;
                auto hash_table = aries_acc::BuildHashTable( table, dimension_key_ids[ i ], can_use_dict );
                multi_key_hash_tables.emplace_back( std::move( hash_table ) );
                wrapper.Ptr = multi_key_hash_tables[ multi_key_hash_tables.size() - 1 ].get();
            }
            
            hash_table_wrappers.emplace_back( wrapper );
#ifdef ARIES_PROFILE
            m_opTime += timer.end();
#endif
            dimensionStats.emplace_back( table->GetStats() );
            dimension_tables.emplace_back( std::move( table ) );
        }
    }

    if( !hasEmptyDimensionTable )
    {
        AriesTableBlockUPtr fact_table;
        do
        {
            result = factSourceNode->GetNext();
            result.TableBlock->ResetAllStats();
            if ( result.Status == AriesOpNodeStatus::ERROR )
                break;

            if ( !fact_table )
            {
                fact_table = std::move( result.TableBlock );
                continue;
            }
                
            if ( result.TableBlock->GetRowCount() == 0 )
                continue;

            fact_table->AddBlock( std::move( result.TableBlock ) );
        } while ( result.Status == AriesOpNodeStatus::CONTINUE );


        if ( result.Status == AriesOpNodeStatus::END && fact_table->GetRowCount() > 0 )
        {
            rows_count[ 0 ] += fact_table->GetRowCount();

            std::vector< AriesHashJoinDataWrapper > data_wrappers;
            for ( const auto& ids : fact_key_ids )
            {
                AriesHashJoinDataWrapper wrapper;
                wrapper.Count = ids.size();

                AriesManagedArray< ColumnDataIterator > data_ptr( wrapper.Count );

                int index = 0;
                for ( const int& id : ids )
                {
                    auto col_encode_type = fact_table->GetColumnEncodeType( id );

                    auto& data_ptr_item = data_ptr[ index++ ];
                    ColumnDataIterator column_data;
                    column_data.m_indices = nullptr;
                    column_data.m_nullData = nullptr;
                    column_data.m_indiceValueType = AriesValueType::INT8;
                    column_data.m_perItemSize = 0;
                    AriesColumnSPtr column;
                    AriesColumnReferenceSPtr column_ref;

                    if ( fact_table->IsColumnUnMaterilized( id ) )
                    {
                        column_ref = fact_table->GetUnMaterilizedColumn( id );
                        auto reffered_column = column_ref->GetReferredColumn();
                        switch ( col_encode_type )
                        {
                            case EncodeType::NONE:
                            {
                                column = std::dynamic_pointer_cast< AriesColumn >( reffered_column );
                                column_data.m_indices = ( int8_t* )column_ref->GetIndices()->GetIndices()->GetData();
                                column_data.m_indiceValueType = AriesValueType::INT32;
                                break;
                            }
                            case EncodeType::DICT:
                            {
                                auto dictColumn = std::dynamic_pointer_cast< AriesDictEncodedColumn >( reffered_column );
                                column = make_shared< AriesColumn >();
                                column->AddDataBuffer( dictColumn->GetDataBuffer() );
                                column_data.m_indices = dictColumn->GetIndices()->GetDataBuffer()->GetData();
                                column_data.m_indiceValueType = dictColumn->GetIndices()->GetDataBuffer()->GetDataType().DataType.ValueType;
                                break;
                            }
                        }
                    }
                    else
                    {
                        switch ( col_encode_type )
                        {
                            case EncodeType::NONE:
                            {
                                column = fact_table->GetMaterilizedColumn( id );
                                break;
                            }
                            case EncodeType::DICT:
                            {
                                auto dictColumn = fact_table->GetDictEncodedColumn( id );
                                column = make_shared< AriesColumn >();
                                column->AddDataBuffer( dictColumn->GetDataBuffer() );
                                column_data.m_indices = dictColumn->GetIndices()->GetDataBuffer()->GetData();
                                column_data.m_indiceValueType = dictColumn->GetIndices()->GetDataBuffer()->GetDataType().DataType.ValueType;
                                break;
                            }
                        }
                    }

                    auto column_type = column->GetColumnType();

                    column_data.m_blockCount = column->GetDataBlockCount();
                    column_data.m_blockSizePrefixSum = GetPrefixSumOfBlockSize( column->GetBlockSizePsumArray() )->ReleaseData();

                    AriesManagedArray< int8_t* > datas( column_data.m_blockCount );

                    int idx = 0;
                    for ( const auto& data: column->GetDataBuffers() )
                    {
                        datas.GetData()[ idx++ ] = data->GetData();
                    }
                    datas.PrefetchToGpu();
                    column_data.m_data = datas.ReleaseData();
                    column_data.m_hasNull = column->GetColumnType().HasNull;

                    switch ( column_type.DataType.ValueType )
                    {
                        case AriesValueType::INT8:
                        case AriesValueType::INT16:
                        case AriesValueType::INT32:
                        case AriesValueType::UINT8:
                        case AriesValueType::UINT16:
                        case AriesValueType::UINT32:
                            column_data.m_perItemSize = 4;
                            break;
                        case AriesValueType::INT64:
                        case AriesValueType::UINT64:
                            column_data.m_perItemSize = 8;
                            break;
                        default:
                            assert(0);
                            break;
                    }

                    if ( column_type.HasNull )
                    {
                        column_data.m_perItemSize += 1;
                    }

                    data_ptr_item.m_blockCount = column_data.m_blockCount;
                    data_ptr_item.m_perItemSize = column_data.m_perItemSize;
                    data_ptr_item.m_blockSizePrefixSum = column_data.m_blockSizePrefixSum;
                    data_ptr_item.m_data = column_data.m_data;
                    data_ptr_item.m_nullData = column_data.m_nullData;
                    data_ptr_item.m_indices = column_data.m_indices;
                    data_ptr_item.m_indiceValueType = column_data.m_indiceValueType;
                    data_ptr_item.m_hasNull = column_data.m_hasNull;
                }
                data_ptr.PrefetchToGpu();
                wrapper.Inputs = data_ptr.ReleaseData();
                data_wrappers.emplace_back( wrapper );
            }
    #ifdef ARIES_PROFILE
            timer.begin();
    #endif
            AriesIndicesArraySPtr fact_table_indices = fact_table->GetTheSharedIndiceForColumns( fact_output_ids );
            auto join_result = aries_acc::StarInnerJoinWithHash( hash_table_wrappers, dimension_tables_indices, data_wrappers,
                    fact_table_indices, fact_table->GetRowCount() );
    #ifdef ARIES_PROFILE
            m_opTime += timer.end();
    #endif
            for ( const auto& wrapper : data_wrappers )
            {
                auto* inputs = ( ColumnDataIterator* )wrapper.Inputs;
                for ( int i = 0; i < wrapper.Count; i++ )
                {
                    AriesMemAllocator::Free( inputs[ i ].m_blockSizePrefixSum );
                    AriesMemAllocator::Free( inputs[ i ].m_data );
                }
                AriesMemAllocator::Free( wrapper.Inputs );
            }

            if ( join_result.JoinCount > 0 )
            {
                AriesTableBlockUPtr fact_table_joined;
                if ( !fact_output_ids.empty() )
                {
                    fact_table_joined = fact_table->MakeTableByColumns( fact_output_ids, false );
                    if( fact_table_indices )
                        fact_table_joined->ReplaceTheOnlyOneIndices( join_result.FactIds );
                    else
                        fact_table_joined->UpdateIndices( join_result.FactIds );
                    fact_table_joined->UpdateColumnIds( output_ids_map );
                    factStats += fact_table_joined->GetStats();
                }

                for ( std::size_t i = 0; i < dimension_tables.size(); i++ )
                {
                    const auto& table = dimension_tables[ i ];
                    if ( output_columns_ids[ i ].empty() )
                    {
                        continue;
                    }
                    auto dimension_table = table->MakeTableByColumns( output_columns_ids[ i ], false );
                    dimensionStats[ i ] += table->GetStats();
                    if( dimension_tables_indices[ i ] )
                        dimension_table->ReplaceTheOnlyOneIndices( join_result.DimensionIds[ i ] );
                    else
                        dimension_table->UpdateIndices( join_result.DimensionIds[ i ] );
                    dimension_table->UpdateColumnIds( dimension_ids_map[ i ] );
                    dimensionStats[ i ] += dimension_table->GetStats();

                    if ( fact_table_joined )
                    {
                        fact_table_joined->MergeTable( std::move( dimension_table ) );
                    }
                    else
                    {
                        fact_table_joined = std::move( dimension_table );
                    }
                }

                if ( !fact_table_joined )
                {
                    result.TableBlock = std::make_unique< AriesTableBlock >();
                }
                else
                {
                    result.TableBlock = std::move( fact_table_joined );
                }
            }
            else
            {
                result.TableBlock = GetEmptyTable();
            }
            
            factStats += fact_table->GetStats();
        }
        else
        {
            result.TableBlock = GetEmptyTable();
        }
    }

    if ( result.Status != AriesOpNodeStatus::CONTINUE )
    {
#ifdef ARIES_PROFILE
        timer.begin();
#endif
        dimension_tables.clear();
        for ( auto& table: dimension_hash_tables )
        {
            aries_acc::ReleaseHashTable( table );
        }
        dimension_hash_tables.clear();

        for ( auto& table: multi_key_hash_tables )
        {
            aries_acc::ReleaseHashTable( table );
        }
        multi_key_hash_tables.clear();
#ifdef ARIES_PROFILE
        m_opTime += timer.end();
#endif
    }

    CacheNodeData( result.TableBlock );

    return result;
}

AriesTableBlockUPtr AriesStarJoinNode::ReadAllData( AriesOpNodeSPtr dataSource )
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

    return tableBlock;
}

void AriesStarJoinNode::SetFactOutputColumnsId( const std::vector< int >& ids )
{
    fact_output_ids.assign( ids.cbegin(), ids.cend() );
}

void AriesStarJoinNode::SetDimensionOutputColumnsId( const std::vector< std::vector< int > >& ids )
{
    output_columns_ids.assign( ids.cbegin(), ids.cend() );
}

void AriesStarJoinNode::SetFactKeyIds( const std::vector< std::vector< int > >& ids )
{
    fact_key_ids.assign( ids.cbegin(), ids.cend() );
}

void AriesStarJoinNode::SetDimensionKeyIds( const std::vector< std::vector< int > >& ids )
{
    dimension_key_ids.assign( ids.cbegin(), ids.cend() );
}

JSON AriesStarJoinNode::GetProfile() const
{
    JSON stats = {
        {"type", m_opName},
        {"param", ""},
        {"time", m_opTime}
    };
    stats["memory"] = { JSON::parse( factStats.ToJson( rows_count[0] ) ) };
    for( std::size_t i=0; i<dimensionStats.size(); ++i)
        stats["memory"].push_back( JSON::parse(dimensionStats[ i ].ToJson(rows_count[i+1])) );
    stats["children"] = { factSourceNode->GetProfile() };
    for ( const auto &source : dimensionSourceNodes )
        stats["children"].push_back(source->GetProfile());
    return stats;
}

void AriesStarJoinNode::SetOuputIdsMap( const std::map< int, int >& map )
{
    output_ids_map = map;
}

void AriesStarJoinNode::SetDimensionIdsMaps( const std::vector< std::map< int, int > >& maps )
{
    dimension_ids_map.assign( maps.cbegin(), maps.cend() );
}


AriesTableBlockUPtr AriesStarJoinNode::GetEmptyTable() const
{
    auto table = std::make_unique< AriesTableBlock >();

    auto fact_table = factSourceNode->GetEmptyTable();
    for ( const auto& item : output_ids_map )
    {
        auto column_type = fact_table->GetColumnType( item.second );
        auto column = std::make_shared< AriesColumn >();
        column->AddDataBuffer( std::make_shared< AriesDataBuffer >( column_type ) );
        table->AddColumn( item.first, column );
    }

    for ( std::size_t i = 0; i < dimensionSourceNodes.size(); i++ )
    {
        auto dimension_table = dimensionSourceNodes[ i ]->GetEmptyTable();
        for ( const auto& item : dimension_ids_map[ i ] )
        {
            auto column_type = dimension_table->GetColumnType( item.second );
            auto column = std::make_shared< AriesColumn >();
            column->AddDataBuffer( std::make_shared< AriesDataBuffer >( column_type ) );
            table->AddColumn( item.first, column );
        }
    }

    return table;
}

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */
