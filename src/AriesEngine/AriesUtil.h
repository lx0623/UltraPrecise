/*
 * AriesUtil.h
 *
 *  Created on: Sep 28, 2018
 *      Author: lichi
 */

#pragma once
#include <sys/time.h>
#include <schema/ColumnEntry.h>
#include "timefunc.h"
#include "CudaAcc/AriesEngineDef.h"
#include "AriesDataDef.h"
#include <mutex>
#include "AriesColumnDataIterator.hxx"


using namespace aries_acc;
using namespace std;

using aries::schema::ColumnType;

BEGIN_ARIES_ENGINE_NAMESPACE

    // struct CPU_Timer
    // {
    //     long start;
    //     long stop;
    //     void begin()
    //     {
    //         struct timeval tv;
    //         gettimeofday( &tv, NULL );
    //         start = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    //     }
    //     long end()
    //     {
    //         struct timeval tv;
    //         gettimeofday( &tv, NULL );
    //         stop = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    //         long elapsed = stop - start;
    //         //printf( "cpu time: %ld\n", elapsed );
    //         return elapsed;
    //     }
    // };

    class DisableOtherConstructors
    {
    protected:
        DisableOtherConstructors() = default;
        DisableOtherConstructors( const DisableOtherConstructors& ) = delete;
        DisableOtherConstructors( DisableOtherConstructors&& ) = delete;
        DisableOtherConstructors& operator =( const DisableOtherConstructors& ) = delete;
        DisableOtherConstructors& operator =( DisableOtherConstructors&& ) = delete;
    };

    string GenerateParamType( const AriesColumnType& param );
    string GetValueTypeAsString( const AriesColumnType& columnType );
    string GetDataTypeStringName( const AriesValueType& valueType );
    AriesColumnType CovertToAriesColumnType( ColumnType type, int length, bool nullable, bool compact = false, int precision = 0, int scale = 0 );
    string LogicOpToString( AriesLogicOpType opType );
    string ComparisonOpToString( AriesComparisonOpType opType );
    void StringToUpper( string& data );
    size_t GetAriesDataTypeSizeInBytes( const aries::AriesDataType& dataType );
    bool IsIntegerType( const AriesColumnType& type );
    string IntervalToString( interval_type interval );
    string GetTypeNameFromConstructor( const string& str );
    string& ReplaceString( string& str, const string& old_value, const string& new_value );
    AriesComparisonOpType SwapComparisonType( AriesComparisonOpType type );
    AriesLiteralValue ConvertRawDataToLiteral( const int8_t* data, const AriesColumnType& type );
    string GetAriesExprTypeName( AriesExprType type );
    string GetAriesComparisonOpTypeName( AriesComparisonOpType type );
    string GetAriesSqlFunctionTypeName( AriesSqlFunctionType type );
    string GetAriesJoinTypeName( AriesJoinType type );
    string GetAriesAggFunctionTypeName( AriesAggFunctionType type );
    string GetAriesSetOpTypeName( AriesSetOpType type );
    string GetColumnTypeName( ColumnType type );
    string GetAriesLogicOpTypeName( AriesLogicOpType type );
    string AriesBreeSearchOpTypeToString( AriesBtreeSearchOpType &opType );
    bool IsSupportNewMergeSort( const AriesColumnType& columnType );

    void DispatchTablesToMultiGpu( const vector< AriesTableBlockUPtr >& tables );

    class AriesNullValueProvider
    {
    public:
        static AriesNullValueProvider &GetInstance()
        {
            static AriesNullValueProvider instance;
            return instance;
        }

    private:
        AriesNullValueProvider();
        ~AriesNullValueProvider();
        AriesNullValueProvider( const AriesNullValueProvider & );
        AriesNullValueProvider &operator=( const AriesNullValueProvider &src );

    public:
        int8_t* GetNullValue( const AriesColumnType& type );
        void Clear();

    private:
        map< size_t, AriesInt8ArraySPtr > m_nullValues;
        mutex m_mutex;
    };


    struct AriesColumnDataIteratorHelper
    {
        AriesInt64ArraySPtr m_dataBlockSizePrefixSumArray;
        vector< AriesDataBufferSPtr > m_dataBlocks;
        shared_ptr< AriesManagedArray< int8_t* > > m_dataBlockPtrs;

        AriesInt64ArraySPtr m_indiceBlockSizePrefixSumArray;
        vector< AriesIndicesArraySPtr > m_indiceBlocks;
        vector< AriesVariantIndicesArraySPtr > m_variantIndiceBlocks;
        shared_ptr< AriesManagedArray< int8_t* > > m_indiceBlockPtrs;
    };
    void GetAriesColumnDataIteratorInfo( AriesColumnDataIterator& iter,
                                         AriesColumnDataIteratorHelper& iterHelper,
                                         const AriesTableBlockUPtr& refTable,
                                         int32_t columnId,
                                         AriesColumnType columnType,
                                         bool useDictIndex );
    void CheckDecimalPrecision( const string& string_value );
    void CheckDecimalError( const string& strValue, aries_acc::Decimal& value, const string& colName, size_t i );

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
