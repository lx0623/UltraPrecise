/*
 * AriesSqlOperator_helper.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_HELPER_H_
#define ARIESSQLOPERATOR_HELPER_H_

#include "AriesSqlOperator_common.h"
#include "AriesSimpleItemContainer.h"

struct AriesTableKeysData
{
    AriesTableKeysData( string&& keys, vector< aries::AriesSimpleItemContainer< RowPos > >&& locations )
    {
        KeysData = std::move( keys );
        TupleLocations = std::move( locations );
    }
    string KeysData;
    vector< aries::AriesSimpleItemContainer< RowPos > > TupleLocations;
};

using AriesTableKeysDataSPtr = std::shared_ptr< AriesTableKeysData >;

BEGIN_ARIES_ACC_NAMESPACE

    AriesIndicesArraySPtr CreateIndicesForMvccTable( const std::vector< int >& inivisibleIdsInInitialTable,
                                                     const std::vector< int >& visibleIdsInDeltaTable,
                                                     size_t tupleNumInInitialTable,
                                                     size_t tupleNumInDeltaTable );

    AriesDataBufferSPtr CreateDataBufferWithNull( size_t count, AriesColumnType type );

    AriesDataBufferSPtr CreateDataBufferWithValue( const std::string& value, size_t count, AriesColumnType type );

    AriesDataBufferSPtr CreateDataBufferWithLiteralExpr( const aries_engine::AriesCommonExprUPtr& expr, size_t count );

    AriesIndicesArraySPtr ConvertRowIdToIndices( const AriesDataBufferSPtr& rowIds );

    aries_engine::AriesColumnSPtr CreateRowIdColumn( int64_t initialTableRowCount, int64_t deltaTableRowCount, int64_t blockSize );

    aries_engine::AriesColumnSPtr CreateRowIdColumnMaterialized( int64_t initialTableRowCount, int64_t deltaTableRowCount, int64_t blockSize );

    AriesDataBufferSPtr CreateVisibleRowIds( int64_t tupleNum, const AriesInt32ArraySPtr& invisibleIds );

    std::pair< AriesDataBufferSPtr, AriesDataBufferSPtr > MakeStringColumnsSameLength( const AriesDataBufferSPtr& leftColumn,
            const AriesDataBufferSPtr& rightColumn );

    void FlipAssociated( AriesInt32ArraySPtr& associated );

    void FlipFlags( AriesInt8ArraySPtr &flags );

    void MergeAssociates( AriesInt32ArraySPtr& dst, const AriesInt32ArraySPtr& src, AriesLogicOpType opType );

    void FlipAssociated( AriesBoolArraySPtr& associated );

    void MergeAssociates( AriesBoolArraySPtr& dst, const AriesBoolArraySPtr& src, AriesLogicOpType opType );

    AriesInt32ArraySPtr ConvertToInt32Array( const AriesBoolArraySPtr& array );

    AriesDataBufferSPtr ConvertToDataBuffer( const AriesBoolArraySPtr& array );

    AriesUInt8ArraySPtr ConvertToUInt8Array( const AriesBoolArraySPtr& array );

    int32_t ExclusiveScan( const AriesInt32ArraySPtr& associated, AriesInt32ArraySPtr& outPrefixSum );

    int64_t ExclusiveScan( const AriesInt32ArraySPtr& associated, AriesInt64ArraySPtr& outPrefixSum );

    int32_t InclusiveScan( const AriesInt32ArraySPtr& associated, AriesInt32ArraySPtr& outPrefixSum );

    int32_t InclusiveScan( const AriesInt8ArraySPtr& flags, AriesInt32ArraySPtr& outPrefixSum );

    int32_t ExclusiveScan( const AriesBoolArraySPtr& associated, AriesInt32ArraySPtr& outPrefixSum );

    int32_t ExclusiveScan( const AriesInt8ArraySPtr &flags, AriesInt32ArraySPtr &outPrefixSum );

    int32_t ExclusiveScan( const AriesManagedInt8ArraySPtr &flags, AriesInt32ArraySPtr &outPrefixSum );

    AriesDataBufferSPtr ConvertStringColumn( const AriesDataBufferSPtr &column, AriesColumnType columnType );

    AriesDataBufferSPtr ConvertToNullableType( const AriesDataBufferSPtr& column );

    void FillWithValue( AriesInt32ArraySPtr& buffer, int value );

    void FillWithValue( AriesBoolArraySPtr& buffer, AriesBool value );

    void InitSequenceValue( AriesInt32ArraySPtr& buffer, int beginValue = 0 );

    AriesDataBufferSPtr DivisionInt64( const AriesDataBufferSPtr& dividend, const AriesDataBufferSPtr& divisor );

    AriesDataBufferSPtr CreateDataBufferWithValue( int8_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( uint8_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( int16_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( uint16_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( int32_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( uint32_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( int64_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( uint64_t value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( float value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( double value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const Decimal& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const Decimal& value, size_t count, size_t ariesDecSize, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesDate& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesTime& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesDatetime& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesTimestamp& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const AriesYear& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( char value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateDataBufferWithValue( const string& value, size_t count, bool nullable = false, bool unique = false );

    AriesDataBufferSPtr CreateNullValueDataBuffer( size_t count );

    AriesDataBufferSPtr ZipColumnData( const std::vector< aries_engine::AriesColumnSPtr >& columns );

    AriesTableKeysDataSPtr GenerateTableKeys( const std::vector< aries_engine::AriesColumnSPtr >& columns, bool checkDuplicate );

    std::pair< bool, AriesInt32ArraySPtr > SortAndVerifyUniqueTableKeys( AriesDataBufferSPtr& data );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_HELPER_H_ */
