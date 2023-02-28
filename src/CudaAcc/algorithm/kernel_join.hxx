// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "kernel_sortedsearch.hxx"
#include "kernel_sortedscan.hxx"
#include "kernel_scan.hxx"
#include "kernel_intervalmove.hxx"
#include "kernel_adapter.hxx"
#include "kernel_util.hxx"
#include "AriesColumnType.h"
#include "AriesAssert.h"

BEGIN_ARIES_ACC_NAMESPACE

    struct ColumnPairs
    {
        int8_t* LeftColumn;
        int8_t* RightColumn;
        AriesColumnType ColumnType;
        AriesComparisonOpType OpType;
    };
    struct JoinDynamicCodeParams;
    extern __device__ void create_comparators( const ColumnPairs* columns, int columnCount, IComparableColumnPair** output );

    template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    std::pair< mem_t< int >, mem_t< int > > find_lower_and_upper_bound( a_it a, int a_count, b_it b, int b_count, context_t& context )
    {
        mem_t< int > lower( a_count );
        mem_t< int > upper( a_count );
        sorted_search< comp_t, bounds_lower, launch_arg_t >( a, a_count, b, b_count, lower.data(), context );
        sorted_search< comp_t, bounds_upper, launch_arg_t >( a, a_count, b, b_count, upper.data(), context );
        return
        {   std::move( lower ), std::move( upper )};
    }

    template< typename launch_arg_t = empty_t, typename comp_t >
    std::pair< mem_t< int >, mem_t< int > > find_lower_and_upper_bound( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp,
            context_t& context )
    {
        mem_t< int > lower( a_count );
        mem_t< int > upper( a_count );

        //sorted_scan< bounds_lower, launch_arg_t >( a, a_count, b, b_count, len, lower.data(), comp, context );
        //sorted_scan< bounds_upper, launch_arg_t >( a, a_count, b, b_count, len, upper.data(), comp, context );

        sorted_search< bounds_lower, launch_arg_t >( a, a_count, b, b_count, len, lower.data(), comp, context );
        sorted_search< bounds_upper, launch_arg_t >( a, a_count, b, b_count, len, upper.data(), comp, context );

        return
        {   std::move( lower ), std::move( upper )};
    }

    template< typename launch_arg_t = empty_t >
    join_pair_t< int > inner_join_helper( const int* lower_data, const int* upper_data, int a_count, context_t& context, const int* vals_a = nullptr,
            const int* vals_b = nullptr )
    {
        mem_t< int > scanned_sizes( a_count );
        managed_mem_t< int64_t > count( 1, context );
        transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                {
                    return upper_data[index] - lower_data[index];
                }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );
        context.synchronize();
        // Allocate an int2 output array and use load-balancing search to compute
        // the join.
        size_t join_count = count.data()[0];
        if( join_count > INT_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "inner join's result too big" );
        }

        mem_t< int > left_output( join_count );
        mem_t< int > right_output( join_count );
        int* left_data = left_output.data();
        int* right_data = right_output.data();

        if( join_count > 0 )
        {
            // Use load-balancing search on the segmens. The output is a pair with
            // a_index = seg and b_index = lower_data[seg] + rank.
            auto k = [=] ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
            {
                left_data[index] = vals_a[seg];
                right_data[index] = vals_b[get<0>(lower) + rank];
            };
            transform_lbs< launch_arg_t >( k, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );

            return
            {   std::move(left_output), std::move(right_output), join_count};
        }
        return
        {   std::move(left_output), std::move(right_output), join_count};
    }

    template< typename launch_arg_t = empty_t >
    join_pair_t< int > left_join_helper( const int* lower_data, const int* upper_data, int a_count, context_t& context, const int* vals_a = nullptr,
            const int* vals_b = nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    {
        mem_t< int > scanned_sizes( a_count );
        mem_t< char > match_flag( a_count );
        char* flag_data = match_flag.data();

        managed_mem_t< int64_t > count( 1, context );
        transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                {
                    int ret = upper_data[index] - lower_data[index];
                    flag_data[ index ] = ret > 0;
                    return ret > 0 ? ret : 1;
                }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );
        context.synchronize();
        // Allocate an int2 output array and use load-balancing search to compute
        // the join.
        size_t join_count = count.data()[0];
        if( join_count > INT_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "left join's result too big" );
        }
        mem_t< int > left_output( join_count );
        mem_t< int > right_output( join_count );
        int* left_data = left_output.data();
        int* right_data = right_output.data();
        if( join_count > 0 )
        {
            if( columns == nullptr )
            {
                // Use load-balancing search on the segmens. The output is a pair with
                // a_index = seg and b_index = lower_data[seg] + rank.
                auto k = [=]ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
                {
                    left_data[ index ] = vals_a[ seg ];
                    right_data[ index ] = flag_data[ seg ] ? vals_b[ get<0>(lower) + rank ] : -1;
                };
                transform_lbs< launch_arg_t >( k, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );
            }
            else
            {
                assert( comp_count > 0 );

                mem_t< IComparableColumnPair* > compArray( comp_count );
                IComparableColumnPair** comparators = compArray.data();
                auto k1 = [=] ARIES_DEVICE(int index)
                {
                    create_comparators( columns, comp_count, comparators );
                };
                transform< launch_box_t< arch_52_cta< 32, 1 > > >( k1, 1, context );

                mem_t< int > lUnMatchedIndexFlag( a_count );
                int* pUnMatchedFlag = lUnMatchedIndexFlag.data();
                cudaMemset( pUnMatchedFlag, 0, a_count * sizeof(int) );

                mem_t< int > associated( join_count );
                int* pAssociated = associated.data();
                init_value( pAssociated, join_count, 1, context );

                // Use load-balancing search on the segmens. The output is a pair with
                // a_index = seg and b_index = lower_data[seg] + rank.
                auto k2 = [=]ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
                {
                    int rIdx = flag_data[ seg ] ? get<0>(lower) + rank : -1;

                    if( rIdx != -1 )
                    {
                        for( int i = 0; i < comp_count; ++i )
                        {
                            if( !comparators[i]->compare( vals_a[seg], vals_b[rIdx] ).is_true() )
                            {
                                pUnMatchedFlag[ vals_a[ seg ] ] = 1;
                                pAssociated[ index ] = 0;
                                rIdx = -1;
                                break;
                            }
                        }
                    }
                    left_data[ index ] = vals_a[ seg ];
                    right_data[ index ] = rIdx != -1 ? vals_b[ rIdx ] : -1;
                };
                transform_lbs< launch_arg_t >( k2, join_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );

                auto k3 = [=] ARIES_DEVICE(int index)
                {
                    delete comparators[index];
                };
                transform< launch_box_t< arch_52_cta< 32, 1 > > >( k3, comp_count, context );

                mem_t< int > psum( join_count );
                managed_mem_t< int32_t > matched_total_count( 1, context );
                int* sum = psum.data();
                scan( pAssociated, join_count, sum, plus_t< int32_t >(), matched_total_count.data(), context );
                context.synchronize();
                size_t matched_total = matched_total_count.data()[0];

                mem_t< int > left_indices( matched_total );
                mem_t< int > right_indices( matched_total );
                int* pLeftIndices = left_indices.data();
                int* pRightIndices = right_indices.data();
                transform( [=]ARIES_DEVICE(int index)
                        {
                            int lIndex = left_data[ index ];
                            int rIndex = right_data[ index ];
                            if( rIndex != -1 )
                            pUnMatchedFlag[ lIndex ] = 0;
                            if( pAssociated[ index ] )
                            {
                                int pos = sum[index];
                                pLeftIndices[ pos ] = lIndex;
                                pRightIndices[ pos ] = rIndex;
                            }
                        }, join_count, context );

                mem_t< int > psumFlag( a_count );
                managed_mem_t< int32_t > unmatched_total_count( 1, context );
                sum = psumFlag.data();
                scan( pUnMatchedFlag, a_count, sum, plus_t< int32_t >(), unmatched_total_count.data(), context );
                context.synchronize();
                size_t unmatched_total = unmatched_total_count.data()[0];

                mem_t< int > left_indices2( unmatched_total );
                mem_t< int > right_indices2( unmatched_total );
                int* pLeftIndices2 = left_indices2.data();
                int* pRightIndices2 = right_indices2.data();
                transform( [=]ARIES_DEVICE(int index)
                        {
                            if( pUnMatchedFlag[ index ] )
                            {
                                int pos = sum[index];
                                pLeftIndices2[ pos ] = index;
                                pRightIndices2[ pos ] = -1;
                            }
                        }, a_count, context );

                mem_t< int > leftData( matched_total + unmatched_total );
                mem_t< int > rightData( matched_total + unmatched_total );
                cudaMemcpy( leftData.data(), pLeftIndices, matched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( leftData.data() + matched_total, pLeftIndices2, unmatched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data(), pRightIndices, matched_total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data() + matched_total, pRightIndices2, unmatched_total * sizeof(int), cudaMemcpyDefault );
                return
                {   std::move(leftData), std::move(rightData), matched_total + unmatched_total};
            }
        }
        return
        {   std::move(left_output), std::move(right_output), join_count};
    }

    template< typename launch_arg_t = empty_t >
    join_pair_t< int > full_join_helper( const int* lower_data, const int* upper_data, int a_count, int b_count, context_t& context,
            const int* vals_a = nullptr, const int* vals_b = nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    {
        mem_t< int > scanned_sizes( a_count );
        mem_t< int > indices_prefix_sum( b_count );
        mem_t< char > match_flag( a_count );
        mem_t< int > unmatched_b_indices( b_count );
        init_value( unmatched_b_indices.data(), b_count, 1, context );

        int* unmatched_b_indices_data = unmatched_b_indices.data();
        int* unmatched_b_indices_sum = indices_prefix_sum.data();
        char* flag_data = match_flag.data();

        managed_mem_t< int64_t > count( 1, context );
        transform_scan< int64_t >( [=]ARIES_DEVICE(int index)
                {
                    int ret = upper_data[index] - lower_data[index];
                    flag_data[ index ] = ret > 0;
                    return ret > 0 ? memset( unmatched_b_indices_data + lower_data[index], 0, ret * sizeof( int ) ), ret : 1;
                }, a_count, scanned_sizes.data(), plus_t< int64_t >(), count.data(), context );

        scan( unmatched_b_indices_data, b_count, unmatched_b_indices_sum, context );
        context.synchronize();
        // Allocate an int2 output array and use load-balancing search to compute
        // the join.
        size_t matched_count = count.data()[0];
        int unmatched_b_count = unmatched_b_indices_sum[b_count - 1] + unmatched_b_indices_data[b_count - 1];
        size_t join_count = matched_count + unmatched_b_count;
        if( join_count > INT_MAX )
        {
            printf( "join count=%lu\n", join_count );
            ARIES_EXCEPTION( ER_TOO_BIG_SELECT, "full join's result too big" );
        }
        mem_t< int > left_output( join_count );
        mem_t< int > right_output( join_count );
        int* left_data = left_output.data();
        int* right_data = right_output.data();
        if( join_count > 0 )
        {
            if( columns == nullptr )
            {
                // Use load-balancing search on the segmens. The output is a pair with
                // a_index = seg and b_index = lower_data[seg] + rank.
                auto k = [=]ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
                {
                    left_data[ index ] = vals_a[seg];
                    right_data[ index ] = flag_data[ seg ] ? vals_b[ get<0>(lower) + rank ] : -1;
                };
                transform_lbs< launch_arg_t >( k, matched_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );
                left_data += matched_count;
                right_data += matched_count;
                transform( [=]ARIES_DEVICE(int index)
                        {
                            if( unmatched_b_indices_data[ index ] )
                            {
                                int pos = unmatched_b_indices_sum[index];
                                left_data[ pos ] = -1;
                                right_data[ pos ] = vals_b[ index ];
                            }
                        }, b_count, context );
                return
                {   std::move(left_output), std::move(right_output), join_count};
            }
            else
            {
                assert( comp_count > 0 );

                mem_t< IComparableColumnPair* > compArray( comp_count );
                IComparableColumnPair** comparators = compArray.data();
                auto k1 = [=] ARIES_DEVICE(int index)
                {
                    create_comparators( columns, comp_count, comparators );
                };
                transform< launch_box_t< arch_52_cta< 32, 1 > > >( k1, 1, context );

                mem_t< int > rIndexFlag( b_count + 1 );
                int* pFlag = rIndexFlag.data() + 1;
                cudaMemset( pFlag, 0, b_count * sizeof(int) );

                // Use load-balancing search on the segmens. The output is a pair with
                // a_index = seg and b_index = lower_data[seg] + rank.
                auto k2 = [=]ARIES_DEVICE(int index, int seg, int rank, tuple<int> lower)
                {
                    int rIdx = flag_data[ seg ] ? get<0>(lower) + rank : -1;

                    if( rIdx != -1 )
                    {
                        for( int i = 0; i < comp_count; ++i )
                        {
                            if( !comparators[i]->compare( vals_a[seg], vals_b[rIdx] ).is_true() )
                            {
                                pFlag[ vals_b[rIdx] ] = 1;
                                rIdx = -1;
                                break;
                            }
                        }
                    }
                    left_data[ index ] = vals_a[ seg ];
                    right_data[ index ] = rIdx != -1 ? vals_b[ rIdx ] : -1;
                };
                transform_lbs< launch_arg_t >( k2, matched_count, scanned_sizes.data(), a_count, make_tuple( lower_data ), context );

                auto k3 = [=] ARIES_DEVICE(int index)
                {
                    delete comparators[index];
                };
                transform< launch_box_t< arch_52_cta< 32, 1 > > >( k3, comp_count, context );

                transform( [=]ARIES_DEVICE(int index)
                        {
                            pFlag[ right_data[ index ] ] = 0;
                        }, matched_count, context );

                left_data += matched_count;
                right_data += matched_count;
                transform( [=]ARIES_DEVICE(int index)
                        {
                            if( unmatched_b_indices_data[ index ] )
                            {
                                int pos = unmatched_b_indices_sum[index];
                                left_data[ pos ] = -1;
                                right_data[ pos ] = vals_b[ index ];
                            }
                        }, b_count, context );

                mem_t< int > psum( b_count );
                managed_mem_t< int32_t > psum_count( 1, context );
                int* sum = psum.data();
                scan( pFlag, b_count, sum, plus_t< int32_t >(), psum_count.data(), context );
                context.synchronize();
                size_t total = psum_count.data()[0];

                mem_t< int > left_indices( total );
                mem_t< int > right_indices( total );
                int* pLeftIndices = left_indices.data();
                int* pRightIndices = right_indices.data();
                transform( [=]ARIES_DEVICE(int index)
                        {
                            if( pFlag[ index ] )
                            {
                                int pos = sum[index];
                                pLeftIndices[ pos ] = -1;
                                pRightIndices[ pos ] = index;
                            }
                        }, b_count, context );

                mem_t< int > leftData( join_count + total );
                mem_t< int > rightData( join_count + total );
                cudaMemcpy( leftData.data(), left_output.data(), join_count * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( leftData.data() + join_count, pLeftIndices, total * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data(), right_output.data(), join_count * sizeof(int), cudaMemcpyDefault );
                cudaMemcpy( rightData.data() + join_count, pRightIndices, total * sizeof(int), cudaMemcpyDefault );
                return
                {   std::move(leftData), std::move(rightData), join_count + total};
            }
        }
        return
        {   std::move(left_output), std::move(right_output), join_count};
    }

    template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    join_pair_t< int > inner_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
            nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        ARIES_ASSERT( !joinDynamicCodeParams, "inner join: dynamic code not supported" );
        auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );
        return inner_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b );
    }

    // template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    // join_pair_t< int > left_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
    //         nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    // {
    //     auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );
    //     return left_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, columns, comp_count );
    // }

    template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    join_pair_t< int > left_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
            nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );
        // return left_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, columns, comp_count );
        return join_pair_t< int >();
    }

    // template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    // join_pair_t< int > right_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
    //         nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    // {
    //     join_pair_t< int > result = left_join< comp_t >( b, b_count, a, a_count, context, vals_b, vals_a, columns, comp_count );
    //     result.left_indices.swap( result.right_indices );
    //     return result;
    // }

    template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    join_pair_t< int > right_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
            nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        join_pair_t< int > result = left_join< comp_t >( b, b_count, a, a_count, context, vals_b, vals_a, joinDynamicCodeParams );
        result.left_indices.swap( result.right_indices );
        return result;
    }

    // template< template< typename, typename > class comp_t, typename launch_arg_t = empty_t, typename a_it, typename b_it >
    // join_pair_t< int > full_join( a_it a, int a_count, b_it b, int b_count, context_t& context, const int* vals_a = nullptr, const int* vals_b =
    //         nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    // {
    //     auto lower_upper = find_lower_and_upper_bound< comp_t >( a, a_count, b, b_count, context );
    //     return full_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, b_count, context, vals_a, vals_b, columns, comp_count );
    // }

    // template< typename launch_arg_t = empty_t, typename comp_t >
    // join_pair_t< int > full_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
    //         const int* vals_a = nullptr, const int* vals_b = nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    // {
    //     auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );
    //     return full_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, b_count, context, vals_a, vals_b, columns,
    //             comp_count );
    // }

    template< typename launch_arg_t = empty_t, typename comp_t >
    join_pair_t< int > inner_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a = nullptr, const int* vals_b = nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        ARIES_ASSERT( !joinDynamicCodeParams, "inner join: dynamic code not supported" );
        auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );
        return inner_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b );
    }

    // template< typename launch_arg_t = empty_t, typename comp_t >
    // join_pair_t< int > left_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
    //         const int* vals_a = nullptr, const int* vals_b = nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    // {
    //     auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );
    //     return left_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, columns, comp_count );
    // }

    template< typename launch_arg_t = empty_t, typename comp_t >
    join_pair_t< int > left_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a = nullptr, const int* vals_b = nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        auto lower_upper = find_lower_and_upper_bound( a, a_count, b, b_count, len, comp, context );
        // return left_join_helper( lower_upper.first.data(), lower_upper.second.data(), a_count, context, vals_a, vals_b, columns, comp_count );
        return join_pair_t< int >();
    }

    // template< typename launch_arg_t = empty_t, typename comp_t >
    // join_pair_t< int > right_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
    //         const int* vals_a = nullptr, const int* vals_b = nullptr, const ColumnPairs* columns = nullptr, size_t comp_count = 0 )
    // {
    //     join_pair_t< int > result = left_join( b, b_count, a, a_count, len, comp, context, vals_b, vals_a, columns, comp_count );
    //     result.left_indices.swap( result.right_indices );
    //     return result;
    // }

    template< typename launch_arg_t = empty_t, typename comp_t >
    join_pair_t< int > right_join( const char* a, int a_count, const char* b, int b_count, int len, comp_t comp, context_t& context,
            const int* vals_a = nullptr, const int* vals_b = nullptr, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr )
    {
        join_pair_t< int > result = left_join( b, b_count, a, a_count, len, comp, context, vals_b, vals_a, joinDynamicCodeParams );
        result.left_indices.swap( result.right_indices );
        return result;
    }

END_ARIES_ACC_NAMESPACE
