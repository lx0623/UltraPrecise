/*
 * AriesSqlOperator_sortjoin.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_SORTJOIN_H_
#define ARIESSQLOPERATOR_SORTJOIN_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE
    extern const int32_t MAX_JOIN_RESULT_COUNT;

    AriesJoinResult Join( AriesJoinType joinType, AriesDataBufferSPtr leftData, AriesDataBufferSPtr rightData,
            const DynamicCodeParams* dynamicCodeParams, const AriesColumnDataIterator *input, bool isNotIn );

    AriesBoolArraySPtr SemiJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams,
            const AriesInt32ArraySPtr leftAssociated = nullptr, const AriesInt32ArraySPtr rightAssociated = nullptr );

    AriesBoolArraySPtr AntiJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams,
            const AriesInt32ArraySPtr leftAssociated = nullptr, const AriesInt32ArraySPtr rightAssociated = nullptr, bool isNotIn = false );

    JoinPair InnerJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const AriesInt32ArraySPtr leftAssociated = nullptr,
            const AriesInt32ArraySPtr rightAssociated = nullptr );

    JoinPair InnerJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair LeftJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair RightJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair FullJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams = nullptr );

    JoinPair CartesianProductJoin( size_t leftCount, size_t rightCount );

    JoinPair CartesianJoin( AriesJoinType joinType, size_t leftCount, size_t rightCount, const DynamicCodeParams& dynamicCodeParams,
            const AriesColumnDataIterator *input );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_SORTJOIN_H_ */
