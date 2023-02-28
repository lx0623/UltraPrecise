/*
 * AriesSqlOperator_compareSingleColumnIter.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_COMPARESINGLECOLUMNITER_H_
#define ARIESSQLOPERATOR_COMPARESINGLECOLUMNITER_H_


#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesBoolArraySPtr CompareColumn( const AriesColumnDataIterator* input, AriesComparisonOpType opType, const AriesDataBufferSPtr& params );

END_ARIES_ACC_NAMESPACE


#endif /* ARIESSQLOPERATOR_COMPARESINGLECOLUMNITER_H_ */
