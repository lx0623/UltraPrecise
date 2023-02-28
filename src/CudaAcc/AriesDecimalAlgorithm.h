// //
// // Created by david.shen on 2020/5/22.
// //

// #pragma once

// #include "AriesEngineDef.h"
// #include "../datatypes/decimal.hxx"


// BEGIN_ARIES_ACC_NAMESPACE

// #define BLOCK_THREAD_COUNT 256
// #define ROUNDUP( count, n ) ( ( count + n - 1 ) / n )

// #define THRESHOLD_4_AGGREGATED_BY_ONESTEP ( 100 )
// #define THRESHOLD_4_BLOCK_COUNT_TOO_LESS ( 4 )
// #define THRESHOLD_4_TEMP_COUNT_TOO_LESS ( 4 )

// #define DECIMAL_VALUE_COUNT 5

// #define GPU_MEMORY_LIMIT_PERCENT ( 0.7 )  // 最大可用值按照GPUmemory的70%算



// struct SumDecimal
// {
//     int64_t values[1];
// };

// enum SumStrategy : int32_t
// {
//     NONE = 0,
//     SOLE_TEMP_SUM,  //每个BLOCK都有自己的中间结果，完成后选thread 0加到最后结果中
//     SHARE_TEMP_SUM,  //某几个BLOCK共用一个中间结果,所有block完成后再加到中间结果中
//     NO_TEMP_SUM  //每个BLOCK都没有自己的中间结果，直接加到最后结果中
// };

// class standard_context_t;

// class AriesDecimalAlgorithm
// {
// public:
//     static AriesDecimalAlgorithm &GetInstance()
//     {
//         static AriesDecimalAlgorithm instance;
//         return instance;
//     }

//     size_t inline GetGpuMemoryLimitSize()
//     {
//         return m_memorySizeLimit;
//     }
//     SumStrategy GetRuntimeStrategy( const AriesDataBufferSPtr &srcColumn, const AriesAggFunctionType &aggType, const size_t &groupCount,
//                                     const size_t handleCount = 0 );
//     AriesDataBufferSPtr AggregateColumn( const AriesDataBufferSPtr &srcColumn, const AriesAggFunctionType &aggType,
//                                          const AriesInt32ArraySPtr &associatedSPtr, const AriesInt32ArraySPtr &groupFlagsSPtr,
//                                          const AriesInt32ArraySPtr &groups, const SumStrategy &strategy, const size_t startIndex = 0,
//                                          const size_t handleCount = 0 );

//     // for ut testcase
//     void inline SetGpuMemorySize(size_t size)
//     {
//         m_gpuMemorySize = size;
//         m_memorySizeLimit = m_gpuMemorySize * GPU_MEMORY_LIMIT_PERCENT;
//     }

// private:
//     AriesDecimalAlgorithm();
//     ~AriesDecimalAlgorithm();

//     AriesDataBufferSPtr AggregateColumnSum( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum, const int *associated,
//                                             const int *groupFlags, const size_t &groupCount, const SumStrategy &strategy );
//     AriesDataBufferSPtr AggregateColumnSumCpu( const AriesDataBufferSPtr &srcColumn, const AriesInt32ArraySPtr &associatedSPtr,
//                                                const AriesInt32ArraySPtr &groupFlagsSPtr, const AriesInt32ArraySPtr &groups );
//     size_t findNextGroupStartIndex( const int *groupFlags, const size_t count, const int *groups );
//     void CpuSumSingleThread( const int8_t *data, const AriesColumnType &columnType, const size_t tupleNum, const int *associated,
//                              const int *groupFlags, int8_t *resData, const AriesColumnType &targetType );
//     AriesDataBufferSPtr AggregateColumnGetMaxmin( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum,
//                                                   const int *associated, const int *groupFlags, const size_t &groupCount, const bool &bMax,
//                                                   const SumStrategy &strategy );
//     AriesDataBufferSPtr AggregateColumnCount( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum, const int *associated,
//                                               const int *groupFlags, const size_t &groupCount, const SumStrategy &strategy );

//     void checkPrecision( AriesColumnType &type, const AriesDataBufferSPtr &srcColumn );

// private:
//     size_t m_memorySizeLimit;
//     size_t m_gpuMemorySize;
//     std::unique_ptr< standard_context_t > m_ctx;
// };

// END_ARIES_ACC_NAMESPACE
