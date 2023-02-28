/*
 * AriesSqlOperator_materialize.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_MATERIALIZE_H_
#define ARIESSQLOPERATOR_MATERIALIZE_H_

#include "AriesSqlOperator_common.h"
using namespace aries_engine;

BEGIN_ARIES_ACC_NAMESPACE

    void AddOffsetToIndices( AriesInt32ArraySPtr &indices, int offset );

    AriesDataBufferSPtr MaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks, const AriesInt64ArraySPtr& blockSizePrefixSum,
            const AriesVariantIndicesArraySPtr &indices, AriesColumnType resultType );

    AriesDataBufferSPtr MaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks, const AriesInt64ArraySPtr& blockSizePrefixSum,
            const AriesInt32ArraySPtr &indices, AriesColumnType resultType );

    AriesDataBufferSPtr MaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks, const AriesInt64ArraySPtr& dataBlockSizePrefixSum,
            const vector< AriesInt32ArraySPtr >& indices, const AriesInt64ArraySPtr& indiceBlockSizePrefixSum, AriesColumnType resultType );

    vector< AriesIndicesArraySPtr > ShuffleIndices( const vector< AriesIndicesArraySPtr >& oldIndices, const AriesIndicesArraySPtr& indices );

    vector< AriesManagedIndicesArraySPtr > ShuffleIndices( const vector< AriesManagedIndicesArraySPtr >& oldIndices, const AriesManagedIndicesArraySPtr& indices );

    std::vector< AriesDataBufferSPtr > ShuffleColumns( const std::vector< AriesDataBufferSPtr >& arg_columns, const AriesInt32ArraySPtr& associated );

    std::vector< AriesDataBufferSPtr > MaterializeDataBlocks( const std::vector< AriesDataBufferSPtr > &dataBlocks, const AriesInt64ArraySPtr& blockSizePrefixSum,
                const AriesInt32ArraySPtr &indices, AriesColumnType resultType );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_MATERIALIZE_H_ */
