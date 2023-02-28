/*
 * AriesSqlOperator_compareSingleColumn.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_COMPARESINGLECOLUMN_H_
#define ARIESSQLOPERATOR_COMPARESINGLECOLUMN_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesBoolArraySPtr CompareColumn( const AriesDataBufferSPtr& columnData, AriesComparisonOpType opType, const AriesDataBufferSPtr& params );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_COMPARESINGLECOLUMN_H_ */
