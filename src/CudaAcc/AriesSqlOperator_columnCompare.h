/*
 * AriesSqlOperator_columnCompare.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_COLUMNCOMPARE_H_
#define ARIESSQLOPERATOR_COLUMNCOMPARE_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesBoolArraySPtr CompareTowColumns( const AriesDataBufferSPtr& leftColumn, AriesComparisonOpType opType,
            const AriesDataBufferSPtr& rightColumn );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_COLUMNCOMPARE_H_ */
