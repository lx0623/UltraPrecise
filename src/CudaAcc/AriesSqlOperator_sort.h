/*
 * AriesSqlOperator_sort.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_SORT_H_
#define ARIESSQLOPERATOR_SORT_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesDataBufferSPtr SortData( const AriesDataBufferSPtr& data, AriesOrderByType order );

    std::pair< AriesDataBufferSPtr, AriesInt32ArraySPtr > SortOneColumn( const AriesDataBufferSPtr& data, AriesOrderByType order );

    AriesInt32ArraySPtr SortColumns( const std::vector< AriesDataBufferSPtr >& orderByColumns, const std::vector< AriesOrderByType >& orders,
            bool nullSmaller = true );

    void SortColumn( AriesDataBufferSPtr &column, const AriesOrderByType &order, AriesInt32ArraySPtr& associated, bool nullSmaller = true );

    AriesDataBufferSPtr SortColumn( const aries_engine::AriesColumnReferenceSPtr &columnRef, const AriesOrderByType &order,
            AriesInt32ArraySPtr& associated, bool nullSmaller = true );

    AriesDataBufferSPtr SortColumn( const aries_engine::AriesColumnSPtr &column, const AriesOrderByType &order, AriesInt32ArraySPtr& associated,
            bool nullSmaller = true );

    AriesInt32ArraySPtr SortDataForTableKeys( AriesDataBufferSPtr& data );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_SORT_H_ */
