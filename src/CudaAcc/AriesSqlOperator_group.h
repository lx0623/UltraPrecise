/*
 * AriesSqlOperator_group.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_GROUP_H_
#define ARIESSQLOPERATOR_GROUP_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    int32_t GroupColumns( const std::vector< AriesDataBufferSPtr >& groupByColumns, AriesInt32ArraySPtr& outAssociated,
            AriesInt32ArraySPtr& outGroups, bool isUnique = false );

    int32_t GroupColumns( const std::vector< AriesDataBufferSPtr > &groupByColumns, AriesInt32ArraySPtr &outAssociated,
            AriesInt32ArraySPtr &outGroups, AriesInt32ArraySPtr &outGroupFlags, bool isUnique = false );

    int32_t GroupColumnForSelfJoin( const AriesDataBufferSPtr &groupByColumn, AriesInt32ArraySPtr &outAssociated, AriesInt32ArraySPtr &outGroups,
            bool isUnique = false );

    int64_t GetSelfJoinPairCount( const AriesInt32ArraySPtr& groups, size_t tupleNum, AriesInt64ArraySPtr& expandedGroupsPrefixSum );

    int32_t GetSelfJoinGroupSizePrefixSum( const AriesInt32ArraySPtr& groups, size_t tupleNum, AriesInt32ArraySPtr& groupsPrefixSum );

    AriesDataBufferSPtr GetItemCountInGroups( const AriesInt32ArraySPtr &groups, size_t tupleNum );

    std::vector< AriesDataBufferSPtr > GatherGroupedColumnData( const std::vector< AriesDataBufferSPtr >& columns,
            const AriesInt32ArraySPtr& associated, const AriesInt32ArraySPtr& groups );

    AriesDataBufferSPtr CombineGroupbyKeys( const std::vector< AriesDataBufferSPtr >& groupByColumns, const AriesInt32ArraySPtr& associated,
            const AriesInt32ArraySPtr& groups );

    AriesDataBufferSPtr AggregateColumnData( const AriesDataBufferSPtr &srcColumn, AriesAggFunctionType aggType,
            const AriesInt32ArraySPtr &associated, const AriesInt32ArraySPtr &groups, const AriesInt32ArraySPtr &groupFlags, bool bDistinct,
            bool bSumForCount, const SumStrategy &strategy );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_GROUP_H_ */
