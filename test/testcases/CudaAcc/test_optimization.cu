/*
 * test_optimization.cu
 *
 *  Created on: Jun 11, 2020
 *      Author: lichi
 */

#include "test_common.h"
#include <set>
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesEngine/AriesUtil.h"
static const char* DB_NAME = "scale_1";
using namespace aries_engine;
using namespace aries_acc;
using namespace std;

int64_t GetSelfJoinPairCount( const AriesInt32ArraySPtr& groups, size_t tupleNum, AriesInt64ArraySPtr& expandedGroupsPrefixSum,
        standard_context_t& context )
{
    managed_mem_t< int64_t > joinCount( 1, context );
    int64_t* pResult = joinCount.data();

    size_t groupCount = groups->GetItemCount();
    int32_t* pGroups = groups->GetData();
    expandedGroupsPrefixSum = std::make_shared< AriesInt64Array >( groupCount + 1 );
    int64_t* pData = expandedGroupsPrefixSum->GetData();

    transform_scan< int64_t >( [=]ARIES_LAMBDA(int index)
            {
                int64_t groupSize = pGroups[ index + 1 ] - pGroups[ index ];
                return groupSize * groupSize;
            }, groupCount - 1, pData, plus_t< int64_t >(), pResult, context );

    context.synchronize();

    int32_t groupSize = pGroups[groupCount - 1] - pGroups[groupCount - 2];
    int32_t expandedSize = groupSize * groupSize;
    pData[groupCount - 1] = expandedSize + pData[groupCount - 2];

    groupSize = tupleNum - pGroups[groupCount - 1];
    expandedSize = groupSize * groupSize;
    pData[groupCount] = expandedSize + pData[groupCount - 1];

    return pData[groupCount];
}

int32_t GetGroupSizePrefixSum( const AriesInt32ArraySPtr& groups, size_t tupleNum, AriesInt32ArraySPtr& groupsPrefixSum, standard_context_t& context )
{
    managed_mem_t< int32_t > joinCount( 1, context );
    int32_t* pResult = joinCount.data();

    size_t groupCount = groups->GetItemCount();
    int32_t* pGroups = groups->GetData();
    groupsPrefixSum = std::make_shared< AriesInt32Array >( groupCount + 1 );
    int32_t* pData = groupsPrefixSum->GetData();

    transform_scan< int32_t >( [=]ARIES_LAMBDA(int index)
            {
                return pGroups[ index + 1 ] - pGroups[ index ];
            }, groupCount - 1, pData, plus_t< int32_t >(), pResult, context );

    context.synchronize();

    int32_t groupSize = pGroups[groupCount - 1] - pGroups[groupCount - 2];
    pData[groupCount - 1] = groupSize + pData[groupCount - 2];

    groupSize = tupleNum - pGroups[groupCount - 1];
    pData[groupCount] = groupSize + pData[groupCount - 1];

    return pData[groupCount];
}

extern "C" __global__ void other_condition( const char **input, const int32_t* associated, const int32_t* groups,
        const int64_t *expanded_group_size_prefix_sum, int32_t group_count, int64_t tupleNum, int64_t sourceRowCount,
        const CallableComparator** comparators, int8_t *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left_index;
    int right_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_offset;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( expanded_group_size_prefix_sum, group_count, i ) - 1;
        group_offset = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_offset;
        else
            group_size = sourceRowCount - groups[group_index];
        offset_in_group = i - expanded_group_size_prefix_sum[group_index];
        left_index = associated[group_offset + offset_in_group / group_size];
        right_index = associated[group_offset + offset_in_group % group_size];
//        int Cuda_Dyn_resultValueName = ( ( ( *( ( int32_t* )( input[0] ) + left_index ) ) ) != ( ( *( ( int32_t* )( input[0] ) + right_index ) ) ) );
//        atomicOr( output + left_index, Cuda_Dyn_resultValueName );
        if( left_index != right_index && ( ( *( ( int32_t* )( input[0] ) + left_index ) ) ) != ( ( *( ( int32_t* )( input[0] ) + right_index ) ) ) )
            output[left_index] = 1;
    }
}

extern "C" __global__ void other_condition_ex( const char **input, const int32_t* associated, const int32_t* groups,
        const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum, const CallableComparator** comparators, int8_t *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left_index;
    int right_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_start_pos;
    for( int32_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
        group_start_pos = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_start_pos;
        else
            group_size = tupleNum - groups[group_index];
        offset_in_group = i - group_size_prefix_sum[group_index];
        left_index = associated[i];
        for( int32_t pos = 0; pos < group_size; ++pos )
        {
            //if( pos != offset_in_group )
            {
                right_index = associated[group_start_pos + pos];
                if( ( ( *( ( int32_t* )( input[0] ) + left_index ) ) ) != ( ( *( ( int32_t* )( input[0] ) + right_index ) ) ) )
                {
                    output[left_index] = 1;
                    break;
                }
            }
        }
    }
}

//extern "C" __global__ void other_condition_composed( const char **input, const int32_t* associated, const int32_t* groups,
//        const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum,
//        const CallableComparator** comparators, int8_t *output )
//{
//    int stride = blockDim.x * gridDim.x;
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    for( int32_t i = tid; i < tupleNum; i += stride )
//    {
//        int group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
//        int group_start_pos = groups[group_index];
//        int group_size;
//        if( group_index < group_count - 1 )
//            group_size = groups[group_index + 1] - group_start_pos;
//        else
//            group_size = tupleNum - groups[group_index];
//        int self_index = associated[i];
//
//        for( int32_t pos = 0; pos < group_size; ++pos )
//        {
//            int sibling_index = associated[group_start_pos + pos];
//            //Q21 semi
//            if( ( ( *( ( int32_t* )( input[2] ) + self_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) )
//            {
//                // myself condition, normally it's a filter condition, or TRUE for no condition
//                if( *( ( AriesDate* )( input[0] ) + sibling_index ) > *( ( AriesDate* )( input[1] ) + sibling_index ) )
//                {
//
//                    int anti_matched = 0;
//                    // check my siblings
//                    for( int32_t pos2 = 0; pos2 < group_size; ++pos2 )
//                    {
//                        int sibling_index2 = associated[group_start_pos + pos2];
//                        //Q21 anti
//                        if( *( ( AriesDate* )( input[0] ) + sibling_index2 ) > *( ( AriesDate* )( input[1] ) + sibling_index2 )
//                                && ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index2 ) ) ) )
//                        {
//                            anti_matched = 1;
//                            break;
//                        }
//                    }
//                    output[self_index] = !anti_matched;
//                }
//            }
//        }
//    }
//}

extern "C" __global__ void other_condition_composed_anti_semi( const char **input, const int32_t* associated, const int32_t* groups,
        const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum, const CallableComparator** comparators, int8_t *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int self_index;
    int sibling_index;
    int group_index;
    int group_size;
    int group_start_pos;
    int semi_matched = 1;
    int anti_matched = 0;
    for( int32_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
        group_start_pos = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_start_pos;
        else
            group_size = tupleNum - groups[group_index];
        self_index = associated[i];

        // myself condition, normally it's a filter condition, or TRUE for no condition
        if( *( ( AriesDate* )( input[0] ) + self_index ) > *( ( AriesDate* )( input[1] ) + self_index ) )
        {
            // anti check my siblings
            if( semi_matched )
            {
                anti_matched = 0;
                for( int32_t pos = 0; pos < group_size; ++pos )
                {
                    sibling_index = associated[group_start_pos + pos];
                    if( *( ( AriesDate* )( input[0] ) + sibling_index ) > *( ( AriesDate* )( input[1] ) + sibling_index )
                            && ( ( *( ( int32_t* )( input[2] ) + self_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) )
                    {
                        anti_matched = 1;
                        break;
                    }
                }
                if( anti_matched )
                    break;

            }

            if( !anti_matched )
            {
                // semi
                semi_matched = 0;
                for( int32_t pos = 0; pos < group_size; ++pos )
                {
                    sibling_index = associated[group_start_pos + pos];
                    if( ( ( *( ( int32_t* )( input[2] ) + self_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) )
                    {
                        semi_matched = 1;
                        break;
                    }
                }
                if( !semi_matched )
                    break;
            }

            output[self_index] = semi_matched && !anti_matched;
        }
    }
}

extern "C" __global__ void other_condition_composed_semi_anti( const char **input, const int32_t* associated, const int32_t* groups,
        const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum, const CallableComparator** comparators, int8_t *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int self_index;
    int sibling_index;
    int group_index;
    int group_size;
    int group_start_pos;
    int semi_matched = 1;
    int anti_matched = 0;
    for( int32_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
        group_start_pos = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_start_pos;
        else
            group_size = tupleNum - groups[group_index];
        self_index = associated[i];

        // myself condition, normally it's a filter condition, or TRUE for no condition
        if( *( ( AriesDate* )( input[0] ) + self_index ) > *( ( AriesDate* )( input[1] ) + self_index ) )
        {
            // semi
            if( !anti_matched )
            {
                semi_matched = 0;
                for( int32_t pos = 0; pos < group_size; ++pos )
                {
                    sibling_index = associated[group_start_pos + pos];
                    if( ( ( *( ( int32_t* )( input[2] ) + self_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) )
                    {
                        semi_matched = 1;
                        break;
                    }
                }
                if( !semi_matched )
                    break;
            }

            if( semi_matched )
            {
                anti_matched = 0;
                // anti check my siblings
                for( int32_t pos = 0; pos < group_size; ++pos )
                {
                    sibling_index = associated[group_start_pos + pos];
                    if( *( ( AriesDate* )( input[0] ) + sibling_index ) > *( ( AriesDate* )( input[1] ) + sibling_index )
                            && ( ( *( ( int32_t* )( input[2] ) + self_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) )
                    {
                        anti_matched = 1;
                        break;
                    }
                }
                if( anti_matched )
                    break;
            }

            output[self_index] = semi_matched && !anti_matched;
        }
    }
}

extern "C" __global__ void other_condition_anti_ex( const char **input, const int32_t* associated, const int32_t* groups,
        const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum, const CallableComparator** comparators, int8_t *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int self_index;
    int sibling_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_start_pos;

    for( int32_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
        group_start_pos = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_start_pos;
        else
            group_size = tupleNum - groups[group_index];
        offset_in_group = i - group_size_prefix_sum[group_index];
        self_index = associated[i];

        // myself condition, normally it's a filter condition, or TRUE for no condition
        if( *( ( AriesDate* )( input[0] ) + self_index ) > *( ( AriesDate* )( input[1] ) + self_index ) )
        {
            int anti_matched = 0;
            // check my siblings
            for( int32_t pos = 0; pos < group_size; ++pos )
            {
                sibling_index = associated[group_start_pos + pos];

                //Q21 anti
                if( *( ( AriesDate* )( input[0] ) + sibling_index ) > *( ( AriesDate* )( input[1] ) + sibling_index )
                        && ( ( *( ( int32_t* )( input[2] ) + self_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + sibling_index ) ) ) )
                {
                    anti_matched = 1;
                    break;
                }
            }
            output[self_index] = !anti_matched;
        }
    }
}

extern "C" __global__ void other_condition_anti( const char **input, const int32_t* associated, const int32_t* groups,
        const int64_t *expanded_group_size_prefix_sum, int32_t group_count, int64_t tupleNum, int64_t sourceRowCount,
        const CallableComparator** comparators, int *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left_index;
    int right_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_offset;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( expanded_group_size_prefix_sum, group_count, i ) - 1;
        group_offset = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_offset;
        else
            group_size = sourceRowCount - groups[group_index];
        offset_in_group = i - expanded_group_size_prefix_sum[group_index];
        left_index = associated[group_offset + offset_in_group / group_size];
        right_index = associated[group_offset + offset_in_group % group_size];
        if( *( ( AriesDate* )( input[0] ) + left_index ) > *( ( AriesDate* )( input[1] ) + left_index ) )
        {
            if( left_index != right_index && *( ( AriesDate* )( input[0] ) + right_index ) > *( ( AriesDate* )( input[1] ) + right_index ) )
            {
                int Cuda_Dyn_resultValueName = ( ( ( *( ( int32_t* )( input[2] ) + left_index ) ) )
                        != ( ( *( ( int32_t* )( input[2] ) + right_index ) ) ) );
                atomicAnd( output + left_index, !Cuda_Dyn_resultValueName );
            }
        }
        else
            output[left_index] = 0;
    }
}

extern "C" __global__ void other_condition_semi( const char **input, const int32_t* associated, const int32_t* groups,
        const int64_t *expanded_group_size_prefix_sum, int32_t group_count, int64_t tupleNum, int64_t sourceRowCount,
        const CallableComparator** comparators, int *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left_index;
    int right_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_offset;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( expanded_group_size_prefix_sum, group_count, i ) - 1;
        group_offset = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_offset;
        else
            group_size = sourceRowCount - groups[group_index];
        offset_in_group = i - expanded_group_size_prefix_sum[group_index];
        left_index = associated[group_offset + offset_in_group / group_size];
        right_index = associated[group_offset + offset_in_group % group_size];
        if( *( ( AriesDate* )( input[0] ) + left_index ) > *( ( AriesDate* )( input[1] ) + left_index )
                && *( ( AriesDate* )( input[0] ) + right_index ) > *( ( AriesDate* )( input[1] ) + right_index ) )
        {
            int Cuda_Dyn_resultValueName =
                    ( ( ( *( ( int32_t* )( input[2] ) + left_index ) ) ) != ( ( *( ( int32_t* )( input[2] ) + right_index ) ) ) );
            atomicOr( output + left_index, Cuda_Dyn_resultValueName );
        }
    }
}

void cpu_other_condition( const char **input, const int32_t* associated, const int32_t* groups, const int64_t *expanded_group_size_prefix_sum,
        int32_t group_count, int64_t tupleNum, int64_t sourceRowCount, const CallableComparator** comparators, int *output )
{
    int left_index;
    int right_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_offset;
    for( int64_t i = 0; i < tupleNum; ++i )
    {
        group_index = binary_search< bounds_upper >( expanded_group_size_prefix_sum, group_count, i ) - 1;
        group_offset = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_offset;
        else
            group_size = sourceRowCount - groups[group_index];
        offset_in_group = i - expanded_group_size_prefix_sum[group_index];
        left_index = associated[group_offset + offset_in_group / group_size];
        right_index = associated[group_offset + offset_in_group % group_size];
        int Cuda_Dyn_resultValueName = ( ( ( *( ( int32_t* )( input[0] ) + left_index ) ) ) != ( ( *( ( int32_t* )( input[0] ) + right_index ) ) ) );
        output[left_index] |= Cuda_Dyn_resultValueName;
    }
}

void cpu_other_condition_ex( const char **input, const int32_t* associated, const int32_t* groups, const int32_t *group_size_prefix_sum,
        int32_t group_count, int32_t tupleNum, const CallableComparator** comparators, int8_t *output )
{
    int left_index;
    int right_index;
    int group_index;
    int offset_in_group;
    int group_size;
    int group_start_pos;
    for( int32_t i = 0; i < tupleNum; ++i )
    {
        group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
        group_start_pos = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_start_pos;
        else
            group_size = tupleNum - groups[group_index];
        offset_in_group = i - group_size_prefix_sum[group_index];
        left_index = associated[group_start_pos + offset_in_group];
        for( int32_t pos = 0; pos < group_size; ++pos )
        {
            if( pos != offset_in_group )
            {
                right_index = associated[group_start_pos + pos];
                if( ( ( *( ( int32_t* )( input[0] ) + left_index ) ) ) != ( ( *( ( int32_t* )( input[0] ) + right_index ) ) ) )
                {
                    output[left_index] = 1;
                    break;
                }
            }
        }
    }
}

TEST(optimize, self_join)
{
    standard_context_t context;
    aries_engine::AriesTableBlockUPtr table = ReadTable( DB_NAME, "lineitem",
    { 1, 3, 12, 13 } );

    size_t tupleNum = table->GetRowCount();

    context.timer_begin();
    AriesDataBufferSPtr l_receiptdate = table->GetColumnBuffer( 4 );
    AriesDataBufferSPtr l_commitdate = table->GetColumnBuffer( 3 );
    l_receiptdate->PrefetchToGpu();
    l_commitdate->PrefetchToGpu();
    AriesBoolArraySPtr flags = CompareTowColumns( l_receiptdate, AriesComparisonOpType::GT, l_commitdate );
    printf( "CompareTowColumns gpu time: %3.1f\n", context.timer_end() );

    context.timer_begin();
    auto outIndex = FilterAssociated( flags );
    printf( "FilterAssociated gpu time: %3.1f\n", context.timer_end() );
    size_t outTupleNum = outIndex->GetItemCount();
    cout << "outTupleNum:" << outTupleNum << endl;
    flags = nullptr;

    context.timer_begin();
    table->UpdateIndices( outIndex );
    printf( "UpdateIndices gpu time: %3.1f\n", context.timer_end() );

    AriesInt32ArraySPtr outAssociated;
    AriesInt32ArraySPtr outGroups;
    AriesInt32ArraySPtr outGroupFlags;
    context.timer_begin();
    AriesDataBufferSPtr l_orderkey = table->GetColumnBuffer( 1 );
    l_orderkey->PrefetchToGpu();
    int32_t groupCount = GroupColumnForSelfJoin( l_orderkey, outAssociated, outGroups );
    printf( "GroupColumns gpu time: %3.1f\n", context.timer_end() );
    cout << "groupCount:" << groupCount << endl;
    AriesInt64ArraySPtr expandedGroupsPrefixSum;
    AriesInt32ArraySPtr groupsPrefixSum;
    context.timer_begin();
    int64_t joinCount = GetSelfJoinPairCount( outGroups, outTupleNum, expandedGroupsPrefixSum, context );
    int32_t joinCountEx = GetGroupSizePrefixSum( outGroups, outTupleNum, groupsPrefixSum, context );
    printf( "GetSelfJoinPairCount gpu time: %3.1f\n", context.timer_end() );
    cout << "joinCount=" << joinCount << endl;
    cout << "joinCountEx=" << joinCountEx << endl;

    cout << "outAssociated:" << outAssociated->GetItemCount() << endl;
    cout << "outGroups:" << outGroups->GetItemCount() << endl;
    cout << "expandedGroupsPrefixSum:" << expandedGroupsPrefixSum->GetItemCount() << endl;

    context.timer_begin();
    AriesArray< const int8_t* > columns( 1 );
    AriesDataBufferSPtr l_suppkey = table->GetColumnBuffer( 2 );
    l_suppkey->PrefetchToGpu();
    columns.GetData()[0] = l_suppkey->GetData();
    AriesArray< int8_t > output( outTupleNum, true );
    int8_t* pOutput = output.GetData();

    other_condition<<< div_up( joinCount, 256l ), 256l >>>( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
            expandedGroupsPrefixSum->GetData(), groupCount, joinCount, outTupleNum, nullptr, pOutput );
    printf( "other_condition gpu time: %3.1f\n", context.timer_end() );
    context.synchronize();

//    cpu_other_condition( (const char**)columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
//            expandedGroupsPrefixSum->GetData(), groupCount, joinCount, outTupleNum, nullptr, pOutput );
    int total = 0;
    for( int i = 0; i < outTupleNum; ++i )
        total += pOutput[i] ? 0 : 1;
    cout << "total:" << total << endl;
}

TEST(optimize, self_join_ex)
{
    standard_context_t context;
    aries_engine::AriesTableBlockUPtr table = ReadTable( DB_NAME, "lineitem",
    { 1, 3, 12, 13 } );

    size_t tupleNum = table->GetRowCount();

    context.timer_begin();
    AriesDataBufferSPtr l_receiptdate = table->GetColumnBuffer( 4 );
    AriesDataBufferSPtr l_commitdate = table->GetColumnBuffer( 3 );
    l_receiptdate->PrefetchToGpu();
    l_commitdate->PrefetchToGpu();
    AriesBoolArraySPtr flags = CompareTowColumns( l_receiptdate, AriesComparisonOpType::GT, l_commitdate );
    printf( "CompareTowColumns gpu time: %3.1f\n", context.timer_end() );

    context.timer_begin();
    auto outIndex = FilterAssociated( flags );
    printf( "FilterAssociated gpu time: %3.1f\n", context.timer_end() );
    size_t outTupleNum = outIndex->GetItemCount();
    cout << "outTupleNum:" << outTupleNum << endl;
    flags = nullptr;

    context.timer_begin();
    table->UpdateIndices( outIndex );
    printf( "UpdateIndices gpu time: %3.1f\n", context.timer_end() );

    AriesInt32ArraySPtr outAssociated;
    AriesInt32ArraySPtr outGroups;
    AriesInt32ArraySPtr outGroupFlags;
    context.timer_begin();
    AriesDataBufferSPtr l_orderkey = table->GetColumnBuffer( 1 );
    l_orderkey->PrefetchToGpu();
    int32_t groupCount = GroupColumnForSelfJoin( l_orderkey, outAssociated, outGroups );
    printf( "GroupColumns gpu time: %3.1f\n", context.timer_end() );
    cout << "groupCount:" << groupCount << endl;
    AriesInt32ArraySPtr groupsPrefixSum;
    context.timer_begin();
    int32_t joinCount = GetGroupSizePrefixSum( outGroups, outTupleNum, groupsPrefixSum, context );
    printf( "GetSelfJoinPairCount gpu time: %3.1f\n", context.timer_end() );
    cout << "joinCount=" << joinCount << endl;
    cout << "outAssociated:" << outAssociated->GetItemCount() << endl;
    cout << "outGroups:" << outGroups->GetItemCount() << endl;

    context.timer_begin();
    AriesArray< const int8_t* > columns( 1 );
    AriesDataBufferSPtr l_suppkey = table->GetColumnBuffer( 2 );
    l_suppkey->PrefetchToGpu();
    columns.GetData()[0] = l_suppkey->GetData();
    AriesArray< int8_t > output( outTupleNum, true );
    int8_t* pOutput = output.GetData();

//    cpu_other_condition_ex( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
//            groupsPrefixSum->GetData(), groupCount, joinCount, nullptr, pOutput );

    other_condition_ex<<< div_up( joinCount, 256 ), 256 >>>( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
            groupsPrefixSum->GetData(), groupCount, joinCount, nullptr, pOutput );
    printf( "other_condition gpu time: %3.1f\n", context.timer_end() );
    context.synchronize();

    int total = 0;
    for( int i = 0; i < outTupleNum; ++i )
        total += pOutput[i] ? 0 : 1;
    cout << "total:" << total << endl;
}

TEST(optimize, self_join_anti)
{
    standard_context_t context;

    AriesDataBufferSPtr l_orderkey = ReadColumn( DB_NAME, "lineitem", 1 );
    AriesDataBufferSPtr l_suppkey = ReadColumn( DB_NAME, "lineitem", 3 );
    AriesDataBufferSPtr l_commitdate = ReadColumn( DB_NAME, "lineitem", 12 );
    AriesDataBufferSPtr l_receiptdate = ReadColumn( DB_NAME, "lineitem", 13 );

    size_t tupleNum = l_orderkey->GetItemCount();

    vector< AriesDataBufferSPtr > groupByColumns;
    AriesInt32ArraySPtr outAssociated;
    AriesInt32ArraySPtr outGroups;
    AriesInt32ArraySPtr outGroupFlags;
    groupByColumns.push_back( l_orderkey );

    context.timer_begin();
    int32_t groupCount = GroupColumns( groupByColumns, outAssociated, outGroups );
    printf( "GroupColumns gpu time: %3.1f\n", context.timer_end() );

    AriesInt64ArraySPtr expandedGroupsPrefixSum;
    context.timer_begin();
    int64_t joinCount = GetSelfJoinPairCount( outGroups, tupleNum, expandedGroupsPrefixSum, context );
    printf( "GetSelfJoinPairCount gpu time: %3.1f\n", context.timer_end() );
    cout << "joinCount=" << joinCount << endl;

    AriesArray< const int8_t* > columns( 3 );
    columns.GetData()[0] = l_receiptdate->GetData();
    columns.GetData()[1] = l_commitdate->GetData();
    columns.GetData()[2] = l_suppkey->GetData();

    AriesInt32ArraySPtr output = std::make_shared< AriesInt32Array >( tupleNum, true );
    FillWithValue( output, 1 );
    int32_t* pOutput = output->GetData();

    context.timer_begin();
//    other_condition_semi<<< div_up( joinCount, 256l ), 256l >>>( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
//            expandedGroupsPrefixSum->GetData(), groupCount, joinCount, tupleNum, nullptr, pOutput );
    other_condition_anti<<< div_up( joinCount, 256l ), 256l >>>( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
            expandedGroupsPrefixSum->GetData(), groupCount, joinCount, tupleNum, nullptr, pOutput );
    printf( "other_condition_anti gpu time: %3.1f\n", context.timer_end() );
    context.synchronize();

    int total = 0;
    for( int i = 0; i < tupleNum; ++i )
        total += pOutput[i];
    cout << "total:" << total << endl;
}

TEST(optimize, self_join_composed)
{
    standard_context_t context;

    AriesDataBufferSPtr l_orderkey = ReadColumn( DB_NAME, "lineitem", 1 );
    AriesDataBufferSPtr l_suppkey = ReadColumn( DB_NAME, "lineitem", 3 );
    AriesDataBufferSPtr l_commitdate = ReadColumn( DB_NAME, "lineitem", 12 );
    AriesDataBufferSPtr l_receiptdate = ReadColumn( DB_NAME, "lineitem", 13 );

    size_t tupleNum = l_orderkey->GetItemCount();

    vector< AriesDataBufferSPtr > groupByColumns;
    AriesInt32ArraySPtr outAssociated;
    AriesInt32ArraySPtr outGroups;
    AriesInt32ArraySPtr outGroupFlags;
    groupByColumns.push_back( l_orderkey );

    context.timer_begin();
    int32_t groupCount = GroupColumns( groupByColumns, outAssociated, outGroups );
    printf( "GroupColumns gpu time: %3.1f\n", context.timer_end() );

    AriesInt32ArraySPtr groupsPrefixSum;
    context.timer_begin();
    int64_t joinCount = GetGroupSizePrefixSum( outGroups, tupleNum, groupsPrefixSum, context );
    printf( "GetSelfJoinPairCount gpu time: %3.1f\n", context.timer_end() );
    cout << "joinCount=" << joinCount << endl;

    AriesArray< const int8_t* > columns( 3 );
    columns.GetData()[0] = l_receiptdate->GetData();
    columns.GetData()[1] = l_commitdate->GetData();
    columns.GetData()[2] = l_suppkey->GetData();

    AriesInt8ArraySPtr output = std::make_shared< AriesInt8Array >( tupleNum, true );
    int8_t* pOutput = output->GetData();

    context.timer_begin();
//    other_condition_anti_ex<<< div_up( joinCount, 256l ), 256l >>>( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
//            groupsPrefixSum->GetData(), groupCount, joinCount, nullptr, pOutput );
//    other_condition_semi<<< div_up( joinCount, 256l ), 256l >>>( ( const char** )columns.GetData(), outAssociated->GetData(), outGroups->GetData(),
//            expandedGroupsPrefixSum->GetData(), groupCount, joinCount, tupleNum, nullptr, pOutput );
    other_condition_composed_semi_anti<<< div_up( joinCount, 256l ), 256l >>>( ( const char** )columns.GetData(), outAssociated->GetData(),
            outGroups->GetData(), groupsPrefixSum->GetData(), groupCount, joinCount, nullptr, pOutput );
    printf( "other_condition_composed gpu time: %3.1f\n", context.timer_end() );
    context.synchronize();

    int total = 0;
    for( int i = 0; i < tupleNum; ++i )
        total += pOutput[i];
    cout << "total:" << total << endl;
}

TEST(rowid, CreateRowIdColumn)
{
    CPU_Timer t;
    t.begin();
    aries_engine::AriesColumnSPtr result = CreateRowIdColumn( 600000000, 10000000, 20971520 );
    cout<<"time cost:"<<t.end()<<endl;

    auto buffers = result->GetDataBuffers();
    for( const auto & buffer : buffers )
    {
        cout<<"------------"<<buffer->GetItemCount()<<endl;
        buffer->Dump();
    }
}

