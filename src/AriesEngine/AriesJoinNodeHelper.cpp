#include <stdio.h>
#include <random>

#include "AriesJoinNodeHelper.h"
#include "AriesJoinNode.h"

#include "AriesUtil.h"
#include "utils/string_util.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesEngineWrapper/AriesExprBridge.h"
#include "AriesColumnDataIterator.hxx"
#include "cpu_algorithm.h"

BEGIN_ARIES_ENGINE_NAMESPACE

static void collect_column_ids( const AriesCommonExprUPtr& expr, std::vector< int >& ids )
{
    if ( expr->IsLiteralValue() )
    {
        return;
    }

    switch ( expr->GetType() )
    {
        case AriesExprType::COLUMN_ID:
        {
            ids.emplace_back( boost::get< int >( expr->GetContent() ) );
            break;
        }
        default:
        {
            for ( int i = 0; i < expr->GetChildrenCount(); i++ )
            {
                collect_column_ids( expr->GetChild( i ), ids );
            }
            break;
        }
    }
}

static void remap_column_ids( const AriesCommonExprUPtr& expr, std::map< int, int >& ids )
{
    if ( expr->IsLiteralValue() )
    {
        return;
    }

    switch ( expr->GetType() )
    {
        case AriesExprType::COLUMN_ID:
        {
            auto id = boost::get< int >( expr->GetContent() );
            auto it = ids.find( id );
            ARIES_ASSERT( it != ids.cend(), "column id not found: " + std::to_string( id ) );

            expr->SetContent( ids[ id ] );
            DLOG( INFO ) << "remap: " << id << " to " << ids[ id ];
            break;
        }
        default:
        {
            for ( int i = 0; i < expr->GetChildrenCount(); i++ )
            {
                remap_column_ids( expr->GetChild( i ), ids );
            }
            break;
        }
    }
}

void AriesJoinNodeHelper::SwapRightJoinToLeft()
{
    std::swap( left_node_of_equal_condition, right_node_of_equal_condition );
    join_type = AriesJoinType::LEFT_JOIN;
    for ( auto& param : dynamic_code_params.params )
    {
        param.ColumnIndex = - param.ColumnIndex;
    }
}

AriesJoinNodeHelper::AriesJoinNodeHelper( AriesJoinNode *joinNode,
                                          const AriesCommonExprUPtr& equal_condition,
                                          const AriesCommonExprUPtr& other_condition,
                                          AriesJoinType type, int nodeId )
: m_joinNode( joinNode ), equal_condtion_constraint_type( JoinConditionConstraintType::None ), m_nodeId( nodeId ), m_isNotIn( false )
{
    join_type = type;

    if ( equal_condition == nullptr || equal_condition->GetType() == AriesExprType::TRUE_FALSE )
    {
        assert( equal_condition ? boost::get< bool >( equal_condition->GetContent() ) : 1 );
        is_cartesian_product = true;
    }
    else
    {
        is_cartesian_product = false;
        AriesCalcTreeGenerator generator;
        left_node_of_equal_condition = generator.ConvertToCalcTree( equal_condition->GetChild( 0 ), m_nodeId );
        right_node_of_equal_condition = generator.ConvertToCalcTree( equal_condition->GetChild( 1 ), m_nodeId );
        auto left_node_code = left_node_of_equal_condition->GetCudaKernelCode();
        auto right_node_code = right_node_of_equal_condition->GetCudaKernelCode();
        dynamic_code_params.code = left_node_code + right_node_code;

    }

    setOtherCondition ( other_condition );
}

std::string AriesJoinNodeHelper::GetDynamicCode() const
{
    return dynamic_code_params.code;
}

void AriesJoinNodeHelper::SetCUModule( const vector< CUmoduleSPtr >& modules )
{
    dynamic_code_params.CUModules = modules;
    if ( left_node_of_equal_condition )
    {
        left_node_of_equal_condition->SetCuModule( modules );
    }

    if ( right_node_of_equal_condition )
    {
        right_node_of_equal_condition->SetCuModule( modules );
    }

    if ( other_condition_as_filter_node )
    {
        other_condition_as_filter_node->SetCuModule( modules );
    }
}

void AriesJoinNodeHelper::getJoinDynamicInputs( AriesManagedArray< AriesColumnDataIterator >& columns,
                                                vector< AriesColumnDataIteratorHelper >& columnHelpers,
                                                const AriesTableBlockUPtr& left_table,
                                                const AriesTableBlockUPtr& right_table ) const
{
    for( std::size_t i = 0; i < dynamic_code_params.params.size(); i++ )
    {
        const auto& param = dynamic_code_params.params[i];
        auto& iter = columns[i];
        auto& iterHelper = columnHelpers[ i ];

        const auto columnId = abs( param.ColumnIndex );
        const auto& table = param.ColumnIndex > 0 ? left_table : right_table;
        GetAriesColumnDataIteratorInfo( iter, iterHelper, table, columnId, param.Type, param.UseDictIndex );
    }
}

AriesJoinResult
AriesJoinNodeHelper::ProcessHashLeftJoin( const AriesHashTableUPtr& hash_table,
                                          const AriesTableBlockUPtr& left_table,
                                          const AriesIndicesArraySPtr& left_table_indices,
                                          const AriesTableBlockUPtr& right_table,
                                          const AriesIndicesArraySPtr& right_table_indices,
                                          int column_id,
                                          bool can_use_dict,
                                          bool left_as_hash )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
    getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );

    JoinDynamicCodeParams params{ dynamic_code_params.CUModules, dynamic_code_params.functionName, columns.GetData(), dynamic_code_params.constValues, dynamic_code_params.items };
    columns.PrefetchToGpu();
    const AriesTableBlockUPtr& value_table = left_as_hash ? right_table : left_table;
    const AriesIndicesArraySPtr& hash_table_indices = left_as_hash ? left_table_indices : right_table_indices;
    const AriesIndicesArraySPtr& value_table_indices = left_as_hash ? right_table_indices : left_table_indices;
    return aries_acc::LeftJoinWithHash( hash_table, hash_table_indices, value_table, value_table_indices, column_id, &params, can_use_dict, left_as_hash );
}

AriesJoinResult
AriesJoinNodeHelper::ProcessHashLeftJoin( const AriesHashTableMultiKeysUPtr& hash_table,
                                          const AriesTableBlockUPtr& left_table,
                                          const AriesIndicesArraySPtr& left_table_indices,
                                          const AriesTableBlockUPtr& right_table,
                                          const AriesIndicesArraySPtr& right_table_indices,
                                          const std::vector< int > column_ids,
                                          const std::vector< bool >& can_use_dict,
                                          bool left_as_hash )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
    getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );

    JoinDynamicCodeParams params{ dynamic_code_params.CUModules, dynamic_code_params.functionName, columns.GetData(), dynamic_code_params.constValues, dynamic_code_params.items };
    columns.PrefetchToGpu();
    const AriesTableBlockUPtr& value_table = left_as_hash ? right_table : left_table;
    const AriesIndicesArraySPtr& hash_table_indices = left_as_hash ? left_table_indices : right_table_indices;
    const AriesIndicesArraySPtr& value_table_indices = left_as_hash ? right_table_indices : left_table_indices;
    return aries_acc::LeftJoinWithHash( hash_table, hash_table_indices, value_table, value_table_indices, column_ids, &params, can_use_dict, left_as_hash );
}

void
AriesJoinNodeHelper::MaterializeColumns( const AriesTableBlockUPtr& left, const AriesTableBlockUPtr& right )
{
    std::vector< int32_t > leftColumnIds, rightColumnIds;
    for( std::size_t i = 0; i < dynamic_code_params.params.size(); i++ )
    {
        const auto& param = dynamic_code_params.params[i];
        if ( param.ColumnIndex > 0 )
        {
            leftColumnIds.emplace_back( param.ColumnIndex );
        }
        else
        {
            rightColumnIds.emplace_back( -param.ColumnIndex );
        }
    }

    if ( !leftColumnIds.empty() )
        left->MaterilizeColumns( leftColumnIds );
    if ( !rightColumnIds.empty() )
        right->MaterilizeColumns( rightColumnIds );
}

AriesJoinResult
AriesJoinNodeHelper::ProcessHashFullJoin( const AriesHashTableUPtr& hash_table,
                                          const AriesTableBlockUPtr& left_table,
                                          const AriesIndicesArraySPtr& left_table_indices,
                                          const AriesTableBlockUPtr& right_table,
                                          const AriesIndicesArraySPtr& right_table_indices,
                                          const int column_id,
                                          bool can_use_dict,
                                          const bool needToSwap )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );

    if ( needToSwap )
        getJoinDynamicInputs( columns, columnHelpers, right_table, left_table );
    else
        getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );

    JoinDynamicCodeParams params{ dynamic_code_params.CUModules, dynamic_code_params.functionName, columns.GetData(), dynamic_code_params.constValues, dynamic_code_params.items };
    columns.PrefetchToGpu();
    return aries_acc::FullJoinWithHash( hash_table, left_table_indices, right_table, right_table_indices, column_id, &params, can_use_dict, needToSwap );
}

AriesJoinResult
AriesJoinNodeHelper::ProcessHalfJoinLeftAsHash( const AriesHashTableUPtr& hash_table,
                                          const AriesTableBlockUPtr& left_table,
                                          const AriesIndicesArraySPtr& left_table_indices,
                                          const AriesTableBlockUPtr& right_table,
                                          int column_id,
                                          bool can_use_dict )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
    getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );

    JoinDynamicCodeParams params{ dynamic_code_params.CUModules, dynamic_code_params.functionName, columns.GetData(), dynamic_code_params.constValues, dynamic_code_params.items };
    columns.PrefetchToGpu();
    return aries_acc::HalfJoinWithLeftHash( join_type, hash_table, left_table_indices, right_table, column_id, &params, can_use_dict, m_isNotIn );
}

AriesJoinResult
AriesJoinNodeHelper::ProcessHalfJoinRightAsHash( const AriesHashTableUPtr& hash_table,
                                          const AriesTableBlockUPtr& left_table,
                                          const AriesIndicesArraySPtr& left_table_indices,
                                          const AriesTableBlockUPtr& right_table,
                                          int column_id,
                                          bool can_use_dict )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
    getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );

    JoinDynamicCodeParams params{ dynamic_code_params.CUModules, dynamic_code_params.functionName, columns.GetData(), dynamic_code_params.constValues, dynamic_code_params.items };

    return aries_acc::HalfJoinWithRightHash( join_type, hash_table, left_table_indices, left_table, column_id, &params, can_use_dict, m_isNotIn );
}

AriesJoinResult AriesJoinNodeHelper::Process( const AriesTableBlockUPtr& left_table, const AriesTableBlockUPtr& right_table )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
    getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );
    columns.PrefetchToGpu();
    if( is_cartesian_product )
    {
        return aries_acc::CartesianJoin( join_type, left_table->GetRowCount(), right_table->GetRowCount(),
                dynamic_code_params, columns.GetData() );
    }

    auto left = left_node_of_equal_condition->Process( left_table );
    auto right = right_node_of_equal_condition->Process( right_table );

    ARIES_ASSERT( CHECK_VARIANT_TYPE( left, AriesDataBufferSPtr ), "left result is invalid" );
    ARIES_ASSERT( CHECK_VARIANT_TYPE( right, AriesDataBufferSPtr ), "right result is invalid" );

    auto left_buffer = boost::get< AriesDataBufferSPtr >( left );
    auto right_buffer = boost::get< AriesDataBufferSPtr >( right );

    return aries_acc::Join( join_type, left_buffer, right_buffer, &dynamic_code_params, columns.GetData(), m_isNotIn );
}

AriesJoinResult AriesJoinNodeHelper::ProcessWithMaterializedBuffer(
    const AriesTableBlockUPtr& left_table,
    const AriesTableBlockUPtr& right_table,
    const AriesDataBufferSPtr& left_buffer,
    const AriesDataBufferSPtr& right_buffer )
{
    AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
    vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
    getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );
    columns.PrefetchToGpu();
    if( is_cartesian_product )
    {
        return aries_acc::CartesianJoin( join_type, left_table->GetRowCount(), right_table->GetRowCount(),
                dynamic_code_params, columns.GetData() );
    }

    return aries_acc::Join( join_type, left_buffer, right_buffer, &dynamic_code_params, columns.GetData(), m_isNotIn );
}

size_t AriesJoinNodeHelper::GetSortFullJoinPartitionCount(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = 1;
    size_t leftRowCount  = leftColumnBuff->GetItemCount();
    size_t rightRowCount = rightColumnBuff->GetItemCount();

    size_t totalMemNeed = leftColumnBuff->GetTotalBytes();
    totalMemNeed += rightColumnBuff->GetTotalBytes();
    for ( const auto& param: dynamic_code_params.params )
    {
        size_t rowCount = param.ColumnIndex > 0 ? leftRowCount : rightRowCount;
        totalMemNeed += param.Type.GetDataTypeSize() * rowCount;
    }

    totalMemNeed += sizeof( index_t ) * leftRowCount * 4; // for left tmp
    totalMemNeed += sizeof( index_t ) * rightRowCount * 3; // for right tmp
    totalMemNeed += sizeof( index_t ) * ( leftRowCount + rightRowCount ); // for result estimated value
    totalMemNeed += ( sizeof( index_t ) + 1 ) * ( leftRowCount ); // left_join_by_dynamic_code: tmp vars
    if ( !dynamic_code_params.functionName.empty() )
    {
        totalMemNeed += sizeof( index_t ) * leftRowCount; // tmp vars
    }

    size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

    const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

    double currentRatio = totalMemNeed / available;
    if( currentRatio > MAX_RATIO )
        partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

    return partitionCount;
}

size_t AriesJoinNodeHelper::GetSortLeftJoinPartitionCount(
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const AriesDataBufferSPtr& leftColumnBuff,
    const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = 1;
    size_t leftRowCount  = leftColumnBuff->GetItemCount();
    size_t rightRowCount = rightColumnBuff->GetItemCount();

    size_t totalMemNeed = leftColumnBuff->GetTotalBytes();
    totalMemNeed += rightColumnBuff->GetTotalBytes();
    for ( const auto& param: dynamic_code_params.params )
    {
        size_t rowCount = param.ColumnIndex > 0 ? leftRowCount : rightRowCount;
        totalMemNeed += param.Type.GetDataTypeSize() * rowCount;
    }

    totalMemNeed += sizeof( index_t ) * 2 * leftRowCount; // for result
    totalMemNeed += sizeof( index_t ) * ( leftRowCount + rightRowCount ); // tmp vars
    totalMemNeed += ( sizeof( index_t ) + 1 ) * ( leftRowCount ); // left_join_by_dynamic_code: tmp vars
    if ( !dynamic_code_params.functionName.empty() )
    {
        totalMemNeed += sizeof( index_t ) * leftRowCount; // tmp vars
    }

    size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

    const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

    double currentRatio = totalMemNeed / available;
    if( currentRatio > MAX_RATIO )
        partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

    return partitionCount;
}

AriesOpResult AriesJoinNodeHelper::SortFullJoinGracePartitioned(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = 1;
    uint32_t seed = 0;

    partitionCount = GetSortFullJoinPartitionCount(
                         leftTable, rightTable,
                         leftColumnBuff, rightColumnBuff );

    vector< AriesTableBlockUPtr > leftSubTables;
    vector< AriesTableBlockUPtr > rightSubTables;

    vector< vector< AriesDataBufferSPtr > > leftPartBuffers, rightPartBuffers;
    GraceHashPartitionTable( leftTable,
                             leftColumnBuff,
                             leftPartBuffers,
                             partitionCount,
                             seed,
                             leftSubTables );
    GraceHashPartitionTable( rightTable,
                             rightColumnBuff,
                             rightPartBuffers,
                             partitionCount,
                             seed,
                             rightSubTables );

    AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };
    size_t resultRowCount = 0;

    auto leftOutputColumnIds   = m_joinNode->GetLeftOutputColumnIds();
    auto rightOutputColumnIds  = m_joinNode->GetRightOutputColumnIds();
    auto leftOutColumnIdMap    = m_joinNode->GetLeftOutColumnIdMap();
    auto rightOutColumnIdMap   = m_joinNode->GetRightOutColumnIdMap();

    while( !leftSubTables.empty() )
    {
        auto leftPartTable = std::move( leftSubTables.back() );
        leftSubTables.pop_back();

        auto rightPartTable = std::move( rightSubTables.back() );
        rightSubTables.pop_back();

        auto leftPartBuffer = leftPartBuffers[ 0 ].back();
        leftPartBuffers[ 0 ].pop_back();
        auto rightPartBuffer = rightPartBuffers[ 0 ].back();
        rightPartBuffers[ 0 ].pop_back();
        if( 0 == leftPartTable->GetRowCount() && 0 == rightPartTable->GetRowCount() )
            continue;
        if ( 0 == leftPartTable->GetRowCount() )
        {
            AriesTableBlockUPtr leftJoined = nullptr;
            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !leftOutputColumnIds.empty() )
            {
                auto nullIndex = CreateNullIndex( rightPartTable->GetRowCount() );
                leftJoined = leftPartTable->MakeTableByColumns( leftOutputColumnIds, false );
                leftJoined->UpdateIndices( nullIndex, true );
                leftJoined->UpdateColumnIds( leftOutColumnIdMap );
            }

            if ( !rightOutputColumnIds.empty() )
            {
                rightJoined = rightPartTable->MakeTableByColumns( rightOutputColumnIds, false );
                auto associatedArray = make_shared< AriesInt32Array >( rightJoined->GetRowCount() );
                aries_acc::InitSequenceValue( associatedArray );
                rightJoined->UpdateIndices( associatedArray, true );
                rightJoined->UpdateColumnIds( rightOutColumnIdMap );
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
                resultRowCount += leftPartTable->GetRowCount();
                continue;
            }

            if ( result.TableBlock )
            {
                result.TableBlock->AddBlock( std::move( partResult ) );
            }
            else
            {
                result.TableBlock = std::move( partResult );
            }
            continue;
        }
        if( 0 == rightPartTable->GetRowCount() )
        {
            AriesTableBlockUPtr leftJoined = nullptr;
            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !leftOutputColumnIds.empty() )
            {
                leftJoined = leftPartTable->MakeTableByColumns( leftOutputColumnIds, false );
                auto associatedArray = make_shared< AriesInt32Array >( leftJoined->GetRowCount() );
                aries_acc::InitSequenceValue( associatedArray );
                leftJoined->UpdateIndices( associatedArray, true );
                leftJoined->UpdateColumnIds( leftOutColumnIdMap );
            }

            if ( !rightOutputColumnIds.empty() )
            {
                auto nullIndex = CreateNullIndex( leftPartTable->GetRowCount() );
                rightJoined = rightPartTable->MakeTableByColumns( rightOutputColumnIds, false );
                rightJoined->UpdateIndices( nullIndex, true );
                rightJoined->UpdateColumnIds( rightOutColumnIdMap );
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
                resultRowCount += leftPartTable->GetRowCount();
                continue;
            }

            if ( result.TableBlock )
            {
                result.TableBlock->AddBlock( std::move( partResult ) );
            }
            else
            {
                result.TableBlock = std::move( partResult );
            }
            continue;
        }

        auto joinResult = ProcessWithMaterializedBuffer( leftPartTable, rightPartTable, leftPartBuffer, rightPartBuffer );
        JoinPair keyPairs = boost::get< JoinPair >( joinResult );
        int tupleNum = keyPairs.JoinCount;
        if( tupleNum > 0 )
        {
            resultRowCount += tupleNum;
            if ( leftOutputColumnIds.empty() && rightOutputColumnIds.empty() )
                continue;

            AriesTableBlockUPtr leftJoined = nullptr;
            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !leftOutputColumnIds.empty() )
            {
                leftJoined = leftPartTable->MakeTableByColumns( leftOutputColumnIds, false );
                leftJoined->UpdateIndices( keyPairs.LeftIndices, true );
                leftJoined->UpdateColumnIds( leftOutColumnIdMap );
            }

            if ( !rightOutputColumnIds.empty() )
            {
                rightJoined = rightPartTable->MakeTableByColumns( rightOutputColumnIds, false );
                rightJoined->UpdateIndices( keyPairs.RightIndices, true );
                rightJoined->UpdateColumnIds( rightOutColumnIdMap );
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

            if ( result.TableBlock )
            {
                result.TableBlock->AddBlock( std::move( partResult ) );
            }
            else
            {
                result.TableBlock = std::move( partResult );
            }
        }
    }

    if ( resultRowCount > 0 )
    {
        if ( leftOutputColumnIds.empty() && rightOutputColumnIds.empty() )
        {
            //sql: select 1 from ...
            result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( resultRowCount );
        }
    }
    else
    {
        // empty table, still need column infos
        result.TableBlock = m_joinNode->GenerateEmptyTable();
    }

    result.Status = AriesOpNodeStatus::END;
    return result;
}

AriesOpResult AriesJoinNodeHelper::SortLeftJoinGracePartitioned(
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const AriesDataBufferSPtr& leftColumnBuff,
    const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = 1;
    uint32_t seed = 0;

    partitionCount = GetSortLeftJoinPartitionCount(
                         leftTable, rightTable,
                         leftColumnBuff, rightColumnBuff );

    vector< AriesTableBlockUPtr > leftSubTables;
    vector< AriesTableBlockUPtr > rightSubTables;

    vector< vector< AriesDataBufferSPtr > > leftPartBuffers, rightPartBuffers;
    GraceHashPartitionTable( leftTable,
                             leftColumnBuff,
                             leftPartBuffers,
                             partitionCount,
                             seed,
                             leftSubTables );
    GraceHashPartitionTable( rightTable,
                             rightColumnBuff,
                             rightPartBuffers,
                             partitionCount,
                             seed,
                             rightSubTables );

    AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };
    size_t resultRowCount = 0;

    auto leftOutputColumnIds   = m_joinNode->GetLeftOutputColumnIds();
    auto rightOutputColumnIds  = m_joinNode->GetRightOutputColumnIds();
    auto leftOutColumnIdMap    = m_joinNode->GetLeftOutColumnIdMap();
    auto rightOutColumnIdMap   = m_joinNode->GetRightOutColumnIdMap();

    while( !leftSubTables.empty() )
    {
        auto leftPartTable = std::move( leftSubTables.back() );
        leftSubTables.pop_back();

        auto rightPartTable = std::move( rightSubTables.back() );
        rightSubTables.pop_back();

        auto leftPartBuffer = leftPartBuffers[ 0 ].back();
        leftPartBuffers[ 0 ].pop_back();
        auto rightPartBuffer = rightPartBuffers[ 0 ].back();
        rightPartBuffers[ 0 ].pop_back();

        if ( 0 == leftPartTable->GetRowCount() )
        {
            continue;
        }
        if( 0 == rightPartTable->GetRowCount() || m_joinNode->IsConstFalseCondition() )
        {
            AriesTableBlockUPtr leftJoined = nullptr;
            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !leftOutputColumnIds.empty() )
            {
                leftJoined = leftPartTable->MakeTableByColumns( leftOutputColumnIds, false );
                auto associatedArray = make_shared< AriesInt32Array >( leftJoined->GetRowCount() );
                aries_acc::InitSequenceValue( associatedArray );
                leftJoined->UpdateIndices( associatedArray );
                leftJoined->UpdateColumnIds( leftOutColumnIdMap );
            }

            if ( !rightOutputColumnIds.empty() )
            {
                auto nullIndex = CreateNullIndex( leftPartTable->GetRowCount() );
                rightJoined = rightPartTable->MakeTableByColumns( rightOutputColumnIds, false );
                rightJoined->UpdateIndices( nullIndex, true );
                rightJoined->UpdateColumnIds( rightOutColumnIdMap );
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
                resultRowCount += leftPartTable->GetRowCount();
                continue;
            }

            if ( result.TableBlock )
            {
                result.TableBlock->AddBlock( std::move( partResult ) );
            }
            else
            {
                result.TableBlock = std::move( partResult );
            }
            continue;
        }

        auto joinResult = ProcessWithMaterializedBuffer( leftPartTable, rightPartTable, leftPartBuffer, rightPartBuffer );
        JoinPair keyPairs = boost::get< JoinPair >( joinResult );
        int tupleNum = keyPairs.JoinCount;
        if( tupleNum > 0 )
        {
            resultRowCount += tupleNum;
            if ( leftOutputColumnIds.empty() && rightOutputColumnIds.empty() )
                continue;

            AriesTableBlockUPtr leftJoined = nullptr;
            AriesTableBlockUPtr rightJoined = nullptr;
            if ( !leftOutputColumnIds.empty() )
            {
                leftJoined = leftPartTable->MakeTableByColumns( leftOutputColumnIds, false );
                leftJoined->UpdateIndices( keyPairs.LeftIndices );
                leftJoined->UpdateColumnIds( leftOutColumnIdMap );
            }

            if ( !rightOutputColumnIds.empty() )
            {
                rightJoined = rightPartTable->MakeTableByColumns( rightOutputColumnIds, false );
                rightJoined->UpdateIndices( keyPairs.RightIndices, true );
                rightJoined->UpdateColumnIds( rightOutColumnIdMap );
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

            if ( result.TableBlock )
            {
                result.TableBlock->AddBlock( std::move( partResult ) );
            }
            else
            {
                result.TableBlock = std::move( partResult );
            }
        }
    }

    if ( resultRowCount > 0 )
    {
        if ( leftOutputColumnIds.empty() && rightOutputColumnIds.empty() )
        {
            //sql: select 1 from ...
            result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( resultRowCount );
        }
    }
    else
    {
        // empty table, still need column infos
        result.TableBlock = m_joinNode->GenerateEmptyTable();
    }

    result.Status = AriesOpNodeStatus::END;
    return result;
}

size_t AriesJoinNodeHelper::GetSortSemiAntiJoinPartitionCount(
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const AriesDataBufferSPtr& leftColumnBuff,
    const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = 1;
    size_t leftRowCount  = leftColumnBuff->GetItemCount();
    size_t rightRowCount = rightColumnBuff->GetItemCount();

    size_t totalMemNeed = leftColumnBuff->GetTotalBytes();
    totalMemNeed += rightColumnBuff->GetTotalBytes();
    for ( const auto& param: dynamic_code_params.params )
    {
        size_t rowCount = param.ColumnIndex > 0 ? leftRowCount : rightRowCount;
        totalMemNeed += param.Type.GetDataTypeSize() * rowCount;
    }

    totalMemNeed += sizeof( AriesBool ) * leftRowCount; // for result
    totalMemNeed += sizeof( index_t ) * ( leftRowCount + rightRowCount ); // tmp vars
    if ( !dynamic_code_params.functionName.empty() )
    {
        totalMemNeed += sizeof( index_t ) * leftRowCount; // tmp vars
    }

    size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

    const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

    double currentRatio = totalMemNeed / available;
    if( currentRatio > MAX_RATIO )
        partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

    return partitionCount;
}

size_t AriesJoinNodeHelper::GetSortInnerJoinPartitionCount(
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const AriesDataBufferSPtr& leftColumnBuff,
    const AriesDataBufferSPtr& rightColumnBuff ) const
{
    size_t partitionCount = 1;

    size_t leftRowCount = leftTable->GetRowCount();
    size_t rightRowCount = rightTable->GetRowCount();

    size_t leftUsagePerRow = leftColumnBuff->GetItemSizeInBytes();
    size_t rightUsagePerRow = rightColumnBuff->GetItemSizeInBytes();

    for ( const auto& param: dynamic_code_params.params )
    {
        if ( param.ColumnIndex > 0 )
        {
            leftUsagePerRow += param.Type.GetDataTypeSize();
        }
        else
        {
            rightUsagePerRow += param.Type.GetDataTypeSize();
        }
    }

    size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

    do {
        auto leftSubRowCount = leftRowCount / partitionCount;
        auto rightSubRowCount = rightRowCount / partitionCount;

        auto usage = leftSubRowCount * leftUsagePerRow + rightSubRowCount * rightUsagePerRow;
        auto estimatedResultCount = max( leftSubRowCount, rightSubRowCount );

        usage += sizeof( int ) * leftSubRowCount * 2; // lower and upper
        usage += sizeof( int ) * leftSubRowCount; // mem_t< int > scanned_sizes( a_count );

        usage += estimatedResultCount * sizeof( index_t ) * 2; // left_output and right_output
        usage += estimatedResultCount * sizeof( int ); // mem_t< int > matchedIndexFlag( join_count );
        usage += estimatedResultCount * sizeof( int ); // mem_t< int > matchedSumFlag( join_count );
        usage += estimatedResultCount * sizeof( int ); // mem_t< int > left_indices( matched_total );
        usage += estimatedResultCount * sizeof( int ); // mem_t< int > right_indices( matched_total );

        if ( usage < available )
        {
            break;
        }
        partitionCount++;
    } while ( true );

    #ifndef NDEBUG
    std::cout << " partitionCount = " << partitionCount << std::endl;
    #endif
    return partitionCount;
}

static AriesTableBlockUPtr generateResultForJoinKeyPairs(
    const JoinPair& keyPairs,
    const AriesJoinType joinType,
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const std::vector< int32_t >& leftIds,
    const std::vector< int32_t >& rightIds,
    const std::map< int, int >& leftOutputIdMap,
    const std::map< int, int >& rigthOutputIdMap
)
{
    bool needFillLeft = joinType == AriesJoinType::FULL_JOIN || joinType == AriesJoinType::RIGHT_JOIN;
    bool needFillRight = joinType == AriesJoinType::FULL_JOIN || joinType == AriesJoinType::LEFT_JOIN;
    AriesTableBlockUPtr leftJoined = nullptr;
    AriesTableBlockUPtr resultTable = nullptr;
    if ( !leftIds.empty() )
    {
        leftJoined = leftTable->MakeTableByColumns( leftIds, false );
        leftJoined->UpdateIndices( keyPairs.LeftIndices, needFillLeft );
        leftJoined->UpdateColumnIds( leftOutputIdMap );
    }

    AriesTableBlockUPtr rightJoined = nullptr;
    if ( !rightIds.empty() )
    {
        rightJoined = rightTable->MakeTableByColumns( rightIds, false );
        rightJoined->UpdateIndices( keyPairs.RightIndices, needFillRight );
        rightJoined->UpdateColumnIds( rigthOutputIdMap );
    }

    if ( leftJoined )
    {
        if ( rightJoined )
        {
            leftJoined->MergeTable( std::move( rightJoined ) );
        }

        return leftJoined;
    }
    else if ( rightJoined )
    {
        return rightJoined;
    }
    
    auto result = make_unique< AriesTableBlock >();
    result->SetRowCount( keyPairs.JoinCount );
    return result;
}

AriesOpResult AriesJoinNodeHelper::SortInnerJoinGracePartitioned(
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const AriesDataBufferSPtr& leftColumnBuff,
    const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = GetSortInnerJoinPartitionCount( leftTable, rightTable, leftColumnBuff, rightColumnBuff );
    vector< AriesTableBlockUPtr > leftSubTables;
    vector< AriesTableBlockUPtr > rightSubTables;
    bool rightHasNullValues;

    vector< vector< AriesDataBufferSPtr > > leftPartBuffers, rightPartBuffers;

    uint32_t seed = 0;
    GraceHashPartitionTable( leftTable,
                             leftColumnBuff,
                             leftPartBuffers,
                             partitionCount,
                             seed,
                             leftSubTables );
    GraceHashPartitionTable( rightTable,
                             rightColumnBuff,
                             rightPartBuffers,
                             partitionCount,
                             seed,
                             rightSubTables,
                             &rightHasNullValues );

    AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };

    size_t resultRowCount = 0;

    auto leftIds = m_joinNode->GetLeftOutputColumnIds();
    auto rightIds = m_joinNode->GetRightOutputColumnIds();

    while( !leftSubTables.empty() )
    {
        auto leftPartTable = std::move( leftSubTables.back() );
        leftSubTables.pop_back();

        auto rightPartTable = std::move( rightSubTables.back() );
        rightSubTables.pop_back();

        auto leftPartBuffer = leftPartBuffers[ 0 ].back();
        leftPartBuffers[ 0 ].pop_back();
        auto rightPartBuffer = rightPartBuffers[ 0 ].back();
        rightPartBuffers[ 0 ].pop_back();

        if ( !leftPartTable || !rightPartTable )
        {
            continue;
        }

        AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
        vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
        getJoinDynamicInputs( columns, columnHelpers, leftPartTable, rightPartTable );
        columns.PrefetchToGpu();

        auto joinResult = aries_acc::Join( join_type, leftPartBuffer, rightPartBuffer, &dynamic_code_params, columns.GetData(), m_isNotIn );
        assert( CHECK_VARIANT_TYPE( joinResult, JoinPair ) );
        auto keyPairs = boost::get< JoinPair >( joinResult );

        if ( keyPairs.JoinCount > 0 )
        {
            resultRowCount += keyPairs.JoinCount;

            auto resultTable = generateResultForJoinKeyPairs( keyPairs,
                                                              join_type,
                                                              leftPartTable,
                                                              rightPartTable,
                                                              leftIds,
                                                              rightIds,
                                                              m_joinNode->GetLeftOutColumnIdMap(),
                                                              m_joinNode->GetRightOutColumnIdMap() );
            if ( resultTable )
            {
                if( !result.TableBlock )
                    result.TableBlock = std::move( resultTable );
                else
                    result.TableBlock->AddBlock( std::move( resultTable ) );
            }
        }
    }

    if ( leftIds.empty() && rightIds.empty() )
    {
        //select 1 from ...
        result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( resultRowCount );
    }
    else if ( resultRowCount == 0 )
    {
        if ( leftIds.empty() && rightIds.empty() )
        {
            result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( 0 );
        }
        else
        {
            result.TableBlock = m_joinNode->GenerateEmptyTable();
        }
    }
    result.Status = AriesOpNodeStatus::END;
    return result;
}

AriesOpResult AriesJoinNodeHelper::SortSemiAntiJoinGracePartitioned(
    const AriesTableBlockUPtr& leftTable,
    const AriesTableBlockUPtr& rightTable,
    const AriesDataBufferSPtr& leftColumnBuff,
    const AriesDataBufferSPtr& rightColumnBuff )
{
    size_t partitionCount = 1;
    uint32_t seed = 0;
    // const int64_t MIN_TABLE_ROW_COUNT = 100;

    partitionCount = GetSortSemiAntiJoinPartitionCount(
                         leftTable, rightTable,
                         leftColumnBuff, rightColumnBuff );

    vector< AriesTableBlockUPtr > leftSubTables;
    vector< AriesTableBlockUPtr > rightSubTables;
    vector< vector< AriesDataBufferSPtr > > leftPartBuffers, rightPartBuffers;
    bool rightHasNullValues;
    GraceHashPartitionTable( leftTable,
                             leftColumnBuff,
                             leftPartBuffers,
                             partitionCount,
                             seed,
                             leftSubTables );
    GraceHashPartitionTable( rightTable,
                             rightColumnBuff,
                             rightPartBuffers,
                             partitionCount,
                             seed,
                             rightSubTables,
                             &rightHasNullValues );

    AriesOpResult result { AriesOpNodeStatus::ERROR, nullptr };

    auto outputColumnIds = m_joinNode->GetOutputColumnIds();

    if ( join_type == AriesJoinType::ANTI_JOIN && IsNotIn() && rightHasNullValues )
    {
        if ( !outputColumnIds.empty() )
            result.TableBlock = m_joinNode->GenerateEmptyTable();
        else
            result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( 0 );
        result.Status = AriesOpNodeStatus::END;
        return result;
    }

    size_t resultRowCount = 0;

    while( !leftSubTables.empty() )
    {
        auto leftPartTable = std::move( leftSubTables.back() );
        leftSubTables.pop_back();

        auto rightPartTable = std::move( rightSubTables.back() );
        rightSubTables.pop_back();


        auto leftPartBuffer = leftPartBuffers[ 0 ].back();
        leftPartBuffers[ 0 ].pop_back();
        auto rightPartBuffer = rightPartBuffers[ 0 ].back();
        rightPartBuffers[ 0 ].pop_back();     

        if ( 0 == leftPartTable->GetRowCount() )
        {
            continue;
        }
        if( 0 == rightPartTable->GetRowCount() || m_joinNode->IsConstFalseCondition() )
        {
            if( join_type == AriesJoinType::ANTI_JOIN )
            {
                resultRowCount += leftPartTable->GetRowCount();
                if( !outputColumnIds.empty() )
                {
                    if( !result.TableBlock )
                        result.TableBlock = std::move( leftPartTable );
                    else
                        result.TableBlock->AddBlock( std::move( leftPartTable ) );
                }
            }
            continue;
        }

        AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
        vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
        getJoinDynamicInputs( columns, columnHelpers, leftPartTable, rightPartTable );
        columns.PrefetchToGpu();

        auto joinResult = aries_acc::Join( join_type, leftPartBuffer, rightPartBuffer, &dynamic_code_params, columns.GetData(), m_isNotIn );
        auto associated = boost::get< AriesBoolArraySPtr >( joinResult );

        AriesInt32ArraySPtr indices = aries_acc::FilterAssociated( associated );
        size_t joinCount = indices->GetItemCount();
        if( joinCount > 0 )
        {
            resultRowCount += joinCount;
            if ( !outputColumnIds.empty() )
            {
                leftPartTable->ResetAllStats();
                leftPartTable->UpdateIndices( indices );

                const auto& tmpTableStats = leftPartTable->GetStats();
                tmpTableStats.Print( "SortSemiAntiJoinGracePartitioned UpdateIndices" );
                // m_leftTableStats += tmpTableStats;
                if( !result.TableBlock )
                    result.TableBlock = std::move( leftPartTable );
                else
                    result.TableBlock->AddBlock( std::move( leftPartTable ) );
            }
        }
    }

    if ( resultRowCount > 0 )
    {
        if ( outputColumnIds.empty() )
        {
            //select 1 from ...
            result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( resultRowCount );
        }
    }
    else
    {
        if ( !outputColumnIds.empty() )
            result.TableBlock = m_joinNode->GenerateEmptyTable();
        else
            result.TableBlock = m_joinNode->GenerateTableWithRowCountOnly( 0 );
    }

    result.Status = AriesOpNodeStatus::END;
    return result;
}

AriesOpResult AriesJoinNodeHelper::ProcessGracePartitioned(
    const AriesTableBlockUPtr& left_table,
    const AriesTableBlockUPtr& right_table )
{
    if ( is_cartesian_product )
    {
        AriesManagedArray< AriesColumnDataIterator > columns( dynamic_code_params.params.size() );
        vector< AriesColumnDataIteratorHelper > columnHelpers( dynamic_code_params.params.size() );
        getJoinDynamicInputs( columns, columnHelpers, left_table, right_table );
        columns.PrefetchToGpu();
        auto keyPairs = aries_acc::CartesianJoin( join_type, left_table->GetRowCount(), right_table->GetRowCount(),
                dynamic_code_params, columns.GetData() );

        auto resultTable = generateResultForJoinKeyPairs(
            keyPairs,
            join_type,
            left_table,
            right_table,
            m_joinNode->GetLeftOutputColumnIds(),
            m_joinNode->GetRightOutputColumnIds(),
            m_joinNode->GetLeftOutColumnIdMap(),
            m_joinNode->GetRightOutColumnIdMap() );
        return { AriesOpNodeStatus::END, std::move( resultTable ) };
    }

    AriesDataBufferSPtr left_buffer;
    AriesDataBufferSPtr right_buffer;
    AEExprColumnIdNode *left_node = dynamic_cast< AEExprColumnIdNode * >( left_node_of_equal_condition.get() );
    AEExprColumnIdNode *right_node = dynamic_cast< AEExprColumnIdNode * >( right_node_of_equal_condition.get() );
    if( left_node && right_node )
    {
        int left_column_id = left_node->GetId();
        int right_column_id = right_node->GetId();

        auto leftEncodeType = left_table->GetColumnEncodeType( left_column_id );
        auto rightEncodeType = right_table->GetColumnEncodeType( right_column_id );
        if( leftEncodeType == rightEncodeType && leftEncodeType == EncodeType::DICT )
        {
            AriesDictEncodedColumnSPtr leftCol;
            AriesDictEncodedColumnSPtr rightCol;
            if( left_table->IsColumnUnMaterilized( left_column_id ) )
            {
                auto columnReference = left_table->GetUnMaterilizedColumn( left_column_id );
                leftCol = std::dynamic_pointer_cast< AriesDictEncodedColumn >( columnReference->GetReferredColumn() );
            }
            else 
                leftCol = left_table->GetDictEncodedColumn( left_column_id );
            if( right_table->IsColumnUnMaterilized( right_column_id ) )
            {
                auto columnReference = right_table->GetUnMaterilizedColumn( right_column_id );
                leftCol = std::dynamic_pointer_cast< AriesDictEncodedColumn >( columnReference->GetReferredColumn() );
            }
            else
                rightCol = right_table->GetDictEncodedColumn( right_column_id );
            assert( leftCol && rightCol );
            if( leftCol->GetDict() == rightCol->GetDict() )
            {
                left_buffer = left_table->GetDictEncodedColumnIndiceBuffer( left_column_id );
                right_buffer = right_table->GetDictEncodedColumnIndiceBuffer( right_column_id );
            }
        }
    }
    if( !left_buffer && !right_buffer ) 
    {
        auto left = left_node_of_equal_condition->Process( left_table );
        auto right = right_node_of_equal_condition->Process( right_table );

        ARIES_ASSERT( CHECK_VARIANT_TYPE( left, AriesDataBufferSPtr ), "left result is invalid" );
        ARIES_ASSERT( CHECK_VARIANT_TYPE( right, AriesDataBufferSPtr ), "right result is invalid" );

        left_buffer = boost::get< AriesDataBufferSPtr >( left );
        right_buffer = boost::get< AriesDataBufferSPtr >( right );

        auto left_type = left_buffer->GetDataType();
        if( left_type.DataType.ValueType == AriesValueType::CHAR && left_type.GetDataTypeSize() > 16 )
        {
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "join column data type size is longer than 16 bytes" );
        }
    }
    
    switch( join_type )
    {
        case AriesJoinType::SEMI_JOIN:
        case AriesJoinType::ANTI_JOIN:
        {
            return SortSemiAntiJoinGracePartitioned( left_table, right_table, left_buffer, right_buffer );
        }
        case AriesJoinType::LEFT_JOIN:
        {
            return SortLeftJoinGracePartitioned( left_table, right_table, left_buffer, right_buffer );
        }
        case AriesJoinType::INNER_JOIN:
        {
            return SortInnerJoinGracePartitioned( left_table, right_table, left_buffer, right_buffer );
        }
        case AriesJoinType::RIGHT_JOIN:
        {
            // right join should be converted to left join in AriesJoinNode
            assert( 0 );
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "type " + GetAriesJoinTypeName( join_type ) + "for AND or OR expression" );
        }
        case AriesJoinType::FULL_JOIN:
        {
            return SortFullJoinGracePartitioned( left_table, right_table, left_buffer, right_buffer );
        }
        default:
            assert( 0 );
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "type " + GetAriesJoinTypeName( join_type ) + "for AND or OR expression" );
    }
}

void AriesJoinNodeHelper::SortColumnsForJoin( const AriesTableBlockUPtr& left_table, const AriesTableBlockUPtr& right_table, AriesDataBufferSPtr& left_buffer,
            AriesInt32ArraySPtr& left_associated, AriesDataBufferSPtr& right_buffer, AriesInt32ArraySPtr& right_associated )
{
    AriesColumnType columnType;
    if( AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( left_node_of_equal_condition.get() ) )
    {
        int id = node->GetId();
        if( IsSupportNewMergeSort( left_table->GetColumnType( id ) ) )
        {
            auto colEncodeType = left_table->GetColumnEncodeType( id );
            ARIES_ASSERT( colEncodeType == EncodeType::NONE,
                          "column encode type not supported: " + std::to_string( ( int )colEncodeType ) );

            if( left_table->IsColumnUnMaterilized( id ) )
            {
                auto columnRef = left_table->GetUnMaterilizedColumn( id );
                left_associated = std::make_shared< AriesInt32Array >( columnRef->GetRowCount() );
                aries_acc::InitSequenceValue( left_associated );
                left_buffer = aries_acc::SortColumn( columnRef, AriesOrderByType::ASC,
                        left_associated );
            }
            else
            {
                auto column = left_table->GetMaterilizedColumn( id );
                left_associated = std::make_shared< AriesInt32Array >( column->GetRowCount() );
                aries_acc::InitSequenceValue( left_associated );
                left_buffer = aries_acc::SortColumn( column, AriesOrderByType::ASC, left_associated );
            }
        }
        else
        {
            // old way
            left_buffer = left_table->GetColumnBuffer( id, true );
            left_associated = std::make_shared< AriesInt32Array >( left_buffer->GetItemCount() );
            aries_acc::InitSequenceValue( left_associated );
            aries_acc::SortColumn( left_buffer, AriesOrderByType::ASC, left_associated );
        }
    }
    else
    {
        auto left = left_node_of_equal_condition->Process( left_table );
        ARIES_ASSERT( CHECK_VARIANT_TYPE( left, AriesDataBufferSPtr ), "left result is invalid" );
        left_buffer = boost::get< AriesDataBufferSPtr >( left );
        left_associated = std::make_shared< AriesInt32Array >( left_buffer->GetItemCount() );
        aries_acc::InitSequenceValue( left_associated );
        aries_acc::SortColumn( left_buffer, AriesOrderByType::ASC, left_associated );
    }

    if( AEExprColumnIdNode *node = dynamic_cast< AEExprColumnIdNode * >( right_node_of_equal_condition.get() ) )
    {
        int id = node->GetId();

        if( IsSupportNewMergeSort( right_table->GetColumnType( id ) ) )
        {
            auto colEncodeType = right_table->GetColumnEncodeType( id );
            ARIES_ASSERT( colEncodeType == EncodeType::NONE,
                          "column encode type not supported: " + std::to_string( ( int )colEncodeType ) );

            if( right_table->IsColumnUnMaterilized( id ) )
            {
                auto columnRef = right_table->GetUnMaterilizedColumn( id );
                right_associated = std::make_shared< AriesInt32Array >( columnRef->GetRowCount() );
                aries_acc::InitSequenceValue( right_associated );
                right_buffer = aries_acc::SortColumn( columnRef, AriesOrderByType::ASC,
                        right_associated );
            }
            else
            {
                auto column = right_table->GetMaterilizedColumn( id );
                right_associated = std::make_shared< AriesInt32Array >( column->GetRowCount() );
                aries_acc::InitSequenceValue( right_associated );
                right_buffer = aries_acc::SortColumn( column, AriesOrderByType::ASC, right_associated );
            }
        }
        else
        {
            // old way
            right_buffer = right_table->GetColumnBuffer( id, true );
            right_associated = std::make_shared< AriesInt32Array >( right_buffer->GetItemCount() );
            aries_acc::InitSequenceValue( right_associated );
            aries_acc::SortColumn( right_buffer, AriesOrderByType::ASC, right_associated );
        }
    }
    else
    {
        auto right = right_node_of_equal_condition->Process( right_table );
        ARIES_ASSERT( CHECK_VARIANT_TYPE( right, AriesDataBufferSPtr ), "right result is invalid" );
        right_buffer = boost::get< AriesDataBufferSPtr >( right );
        right_associated = std::make_shared< AriesInt32Array >( right_buffer->GetItemCount() );
        aries_acc::InitSequenceValue( right_associated );
        aries_acc::SortColumn( right_buffer, AriesOrderByType::ASC, right_associated );
    }
}

void
AriesJoinNodeHelper::generateDynamicCode( int nodeId,
                                          const AriesCommonExprUPtr& other_condition,
                                          std::string& function_name,
                                          std::string& code,
                                          std::map< string, AriesCommonExprUPtr >& agg_functions,
                                          std::vector< AriesDynamicCodeParam >& params,
                                          vector< AriesDataBufferSPtr >& constantValues,
                                          std::vector< AriesDynamicCodeComparator >& comparators,
                                          bool need_swap,
                                          bool is_cartesian )
{
    auto body_code = other_condition->StringForDynamicCode( agg_functions, params, constantValues, comparators );

    if (body_code.empty()) {
        code.assign(body_code);
        return;
    }

    int64_t exprId = ( ( int64_t )nodeId << 32 ) | ( int64_t )other_condition->GetId();
    function_name.assign( "other_condition_fun_" + std::to_string( exprId ) );
    std::string function_string_tpl = R"(
extern "C"  __global__ void
#function_name#( const AriesColumnDataIterator *input,
                 const int *left_indices,
                 const int *right_indices,
                 int tupleNum,
                 const int8_t** constValues,
                 const CallableComparator** comparators,
                 char *output)
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int i = tid; i < tupleNum; i += stride )
    {
        if ( left_indices[i] == -1 || right_indices[ i ] == -1 )
        {
            *( ( AriesBool* )( output + i * sizeof( AriesBool ) ) ) = 0;
            continue;
        }
        auto Cuda_Dyn_resultValueName = #code#;
        *( ( AriesBool* )( output + i * sizeof( AriesBool ) ) ) = Cuda_Dyn_resultValueName;
    }
}
    )";

    std::string cartesian_function_string_tpl = R"(
extern "C"  __global__ void
#function_name#( const AriesColumnDataIterator *input,
                 size_t leftCount,
                 size_t rightCount,
                 size_t tupleNum,
                 int* left_unmatched_flag,
                 int* right_unmatched_flag,
                 const int8_t** constValues,
                 const CallableComparator** comparators,
                 int* left_output,
                 int* right_output,
                 unsigned long long int* output_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int i = tid; i < tupleNum; i += stride )
    {
        auto Cuda_Dyn_resultValueName = #code#;
        if ( Cuda_Dyn_resultValueName )
        {
            auto index = atomicAdd( output_count, 1 );
            if ( index <= #max_join_count# /* 2147483647 INT32_MAX */ )
            {
                auto lIndex = i / rightCount;
                auto rIndex = i % rightCount;
                left_output[ index ] = lIndex;
                right_output[ index ] = rIndex;
                if ( left_unmatched_flag != nullptr )
                {
                    left_unmatched_flag[ lIndex ] = 0;
                }

                if ( right_unmatched_flag != nullptr )
                {
                    right_unmatched_flag[ rIndex ] = 0;
                }
            }
        }
    }
}
    )";

    auto function_string = ReplaceString(is_cartesian ? cartesian_function_string_tpl : function_string_tpl, "#function_name#", function_name);
    function_string = ReplaceString(function_string, "#code#", body_code);
    function_string = ReplaceString(function_string, "#max_join_count#", std::to_string( aries_acc::MAX_JOIN_RESULT_COUNT ) );

    for (std::size_t index = 0; index < params.size(); index++)
    {
        const auto& param = params[index];
        char buf[ 1024 ];
        AriesColumnType type = param.Type;

        const char* indices;
        if ( is_cartesian )
        {
            indices = param.ColumnIndex > 0 ? "( i / rightCount )" : "( i % rightCount )";
        }
        else
        {
            if ( need_swap )
            {
                indices = param.ColumnIndex < 0 ? "left_indices[i]" : "right_indices[i]";
            }
            else
            {
                indices = param.ColumnIndex > 0 ? "left_indices[i]" : "right_indices[i]";
            }
        }

        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            if( type.HasNull )
            {
                ::sprintf( buf,
                        "( nullable_type< Decimal >( *(int8_t*)( input[%lu][%s] ), Decimal( (CompactDecimal*)( input[%lu][%s] + 1 ), %u, %u ) ) )",
                        index, indices, index, indices, type.DataType.Precision, type.DataType.Scale );
            }
            else
            {
                ::sprintf( buf, "( Decimal( ( CompactDecimal* )( input[%lu][%s] ), %u, %u ) )", index, indices,
                        type.DataType.Precision, type.DataType.Scale );
            }
        }
        else
        {
            ::sprintf( buf, "(*( ( %s* )( input[%lu][%s] ) ) )", GenerateParamType( type ).c_str(), index, indices );
        }

        function_string = ReplaceString( function_string, param.ParamName, std::string( buf ) );
    }

    code.assign(function_string);
}

void AriesJoinNodeHelper::PrepareForHashJoin( const HashJoinInfo& hash_join_info )
{
    AriesExprBridge bridge;
    int exprId = 0;

    AriesCommonExprUPtr other_condition = nullptr;
    if ( hash_join_info.OtherCondition )
    {
        other_condition = bridge.Bridge( hash_join_info.OtherCondition );
        other_condition->SetId( ++exprId );
    }

    resetOtherCondition( other_condition );
}

void AriesJoinNodeHelper::setOtherCondition( const AriesCommonExprUPtr& condition )
{
    if ( condition && ( condition->GetType() != AriesExprType::TRUE_FALSE || !boost::get< bool >( condition->GetContent() ) ) )
    {
        std::string other_condition_code, other_condition_function_name;
        std::map< string, AriesCommonExprUPtr > agg_functions;
        std::vector< AriesDynamicCodeParam > params;
        vector< AriesDataBufferSPtr > constantValues;
        std::vector< AriesDynamicCodeComparator > comparators;

        generateDynamicCode( m_nodeId,
                             condition,
                             other_condition_function_name,
                             other_condition_code,
                             agg_functions,
                             params,
                             constantValues,
                             comparators,
                             join_type == AriesJoinType::RIGHT_JOIN,
                             is_cartesian_product );

        dynamic_code_params.code += other_condition_code;
        dynamic_code_params.functionName = other_condition_function_name;
        dynamic_code_params.params.assign( params.cbegin(), params.cend() );
        dynamic_code_params.constValues = std::move( constantValues );
        dynamic_code_params.items.assign( comparators.cbegin(), comparators.cend() );
    }
}

void AriesJoinNodeHelper::resetOtherCondition( const AriesCommonExprUPtr& condition )
{
    if ( condition )
    {
        AriesCalcTreeGenerator generator;
        std::vector< int > columns_ids;
        collect_column_ids( condition, columns_ids );
        std::vector< int > columns_id_left, columns_id_right;
        for ( const auto& id : columns_ids )
        {
            if ( id > 0 )
            {
                if ( std::find( required_columns_id_in_left.cbegin(), required_columns_id_in_left.cend(), id ) ==
                     required_columns_id_in_left.cend() )
                {
                    required_columns_id_in_left.emplace_back( id );
                }

            }
            else if ( id < 0 )
            {
                if ( std::find( required_columns_id_in_right.cbegin(), required_columns_id_in_right.cend(), id ) ==
                     required_columns_id_in_right.cend() )
                {
                    required_columns_id_in_right.emplace_back( id );
                }
            }
            else
            {
                ARIES_ASSERT( 0, "unhandle column id == 0" );
            }
        }

        std::map< int, int > columns_id_map;
        int i = 0;
        for ( const auto& id : required_columns_id_in_left )
        {
            columns_id_map[ id ] = ++i;
        }

        for ( const auto& id : required_columns_id_in_right )
        {
            columns_id_map[ id ] = ++i;
        }

        remap_column_ids( condition, columns_id_map );

        other_condition_as_filter_node = generator.ConvertToCalcTree( condition, m_nodeId );
        dynamic_code_params.code = other_condition_as_filter_node->GetCudaKernelCode();
    }
    else
    {
        dynamic_code_params.code = "";
    }

    dynamic_code_params.functionName.resize(0);
    dynamic_code_params.params.clear();
    dynamic_code_params.items.clear();
}

void AriesJoinNodeHelper::InitColumnsIdInConditions( const std::vector< int >& columns_id )
{
    for ( const auto& id : columns_id )
    {
        if ( id > 0 )
        {
            required_columns_id_in_left.emplace_back( id );
        }
        else if ( id < 0 )
        {
            required_columns_id_in_right.emplace_back( id );
        }
        else
        {
            ARIES_ASSERT( 0, "unhandle column id == 0" );
        }
    }
}

const std::vector< int >& AriesJoinNodeHelper::GetLeftRequiredColumnsId() const
{
    return required_columns_id_in_left;
}

const std::vector< int >& AriesJoinNodeHelper::GetRightRequiredColumnsId() const
{
    return required_columns_id_in_right;
}

size_t AriesJoinNodeHelper::GetLeftHashJoinKernelUsage( const size_t left_row_count, const size_t right_row_count ) const
{
    size_t usage = 0;
    size_t right_count = ( right_row_count - 1 + left_row_count ) / left_row_count;
    if ( !dynamic_code_params.params.empty() )
    {
        for ( const auto& param : dynamic_code_params.params )
        {
            if ( param.ColumnIndex > 0 )
            {
                usage += param.Type.GetDataTypeSize();
            }
            else
            {
                usage += param.Type.GetDataTypeSize() * right_count;
            }
        }

        // associated->AllocArray( result.JoinCount );
        usage += sizeof( AriesBool ) * ( 1 + right_count );
    }

    //managed_mem_t< int32_t > prefixSum( hash_table->HashRowCount + 1, *ctx );
    usage += sizeof( int32_t );

    // managed_mem_t< int32_t > flags( matched_count, *ctx );
    usage += sizeof( int32_t ) * ( 1 + right_count );

    // AriesInt32Array new_left_indices( matched_count );
    // AriesInt32Array new_right_indices( matched_count );
    usage += sizeof( int32_t ) * 2 * ( 1 + right_count );
    return usage;
}

size_t AriesJoinNodeHelper::GetFullHashJoinKernelUsage( const size_t left_row_count, const size_t right_row_count ) const
{
    size_t usage = 0;
    size_t right_count = ( right_row_count - 1 + left_row_count ) / left_row_count;
    if ( !dynamic_code_params.params.empty() )
    {
        for ( const auto& param : dynamic_code_params.params )
        {
            if ( param.ColumnIndex > 0 )
            {
                usage += param.Type.GetDataTypeSize();
            }
            else
            {
                usage += param.Type.GetDataTypeSize() * right_count;
            }
        }

        // associated->AllocArray( result.JoinCount );
        usage += sizeof( AriesBool ) * ( 1 + right_count );
    }

    //managed_mem_t< int32_t > prefixSum( hash_table->HashRowCount + 1, *ctx );
    usage += sizeof( int32_t );

    // managed_mem_t< int32_t > flags( matched_count, *ctx );
    usage += sizeof( int32_t ) * ( 1 + right_count );

    // AriesInt32Array new_left_indices( matched_count );
    // AriesInt32Array new_right_indices( matched_count );
    usage += sizeof( int32_t ) * 2 * ( 1 + right_count );
    return usage;
}

END_ARIES_ENGINE_NAMESPACE
