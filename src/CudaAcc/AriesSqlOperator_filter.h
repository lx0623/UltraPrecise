/*
 * AriesSqlOperator_filter.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_FILTER_H_
#define ARIESSQLOPERATOR_FILTER_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesIndicesArraySPtr FilterAssociated( const AriesBoolArraySPtr& associated );

    AriesIndicesArraySPtr FilterFlags( const AriesInt8ArraySPtr &flags );

    AriesManagedIndicesArraySPtr FilterFlags( const AriesManagedInt8ArraySPtr &flags );

    std::vector< AriesDataBufferSPtr > FilterColumnData( const std::vector< AriesDataBufferSPtr >& columns, const AriesInt32ArraySPtr& associated );

    std::vector< AriesDataBufferSPtr > FilterColumnDataEx( const std::vector< AriesDataBufferSPtr >& columns, const AriesInt32ArraySPtr& associated );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_FILTER_H_ */
