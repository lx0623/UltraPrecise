/*
 * AriesSqlOperator_sqlfunc.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_SQLFUNC_H_
#define ARIESSQLOPERATOR_SQLFUNC_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesBoolArraySPtr IsNull( const AriesDataBufferSPtr& column );

    AriesBoolArraySPtr IsNotNull( const AriesDataBufferSPtr& column );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_SQLFUNC_H_ */
