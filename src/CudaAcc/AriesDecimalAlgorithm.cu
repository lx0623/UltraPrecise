// //
// // Created by david.shen on 2020/5/22.
// //

// #include <cooperative_groups.h>

// #include "AriesDecimalAlgorithm.h"
// #include "AriesEngine/AriesUtil.h"
// #include "AriesEngine/cpu_algorithm.h"
// #include "CudaAcc/AriesEngineAlgorithm.h"
// #include "algorithm/context.hxx"
// #include "CpuTimer.h"


// using namespace aries_engine;
// using namespace cooperative_groups;

// BEGIN_ARIES_ACC_NAMESPACE

//     __device__ void DecimalSumInitBuf( int64_t *buf, const size_t &len, const int &stride )
//     {
//         for ( size_t i = threadIdx.x; i < len; i += stride )
//         {
//             *( buf + i ) = 0;
//         }
//     }

//     __device__ void DecimalSumBlock( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum, const int *associated,
//                                     const int *groupFlags, SumDecimal *out, const int &validElements )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         int len = columnType.GetDataTypeSize();
//         auto prec = columnType.DataType.Precision;
//         auto scale = columnType.DataType.Scale;
//         auto hasNull = columnType.HasNull;
//         bool isCompact = columnType.DataType.ValueType == AriesValueType::COMPACT_DECIMAL;
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             auto pItemData = data + (int64_t) associated[i] * len;
//             if ( hasNull && *pItemData++ == 0 )
//             {
//                 continue;
//             }
//             const auto d = isCompact ? Decimal( (CompactDecimal *) pItemData, prec, scale ) : *(Decimal *) pItemData;
//             auto tmpTarget = out + ( groupFlags[i] - groupFlags[0] ) * validElements;
//             int k = i % validElements;
//             int sourceStartPos = DECIMAL_VALUE_COUNT - validElements;
//             for ( int j = 0; j < validElements; ++j )
//             {
//                 atomicAdd( (unsigned long long int *) ( tmpTarget->values + k ), (unsigned long long int) d.values[sourceStartPos + k] );
//                 if ( ++k >= validElements )
//                 {
//                     k = 0;
//                 }
//             }
//         }
//     }

//     __device__ void DecimalSumAddToTarget( thread_block tb, const size_t &groupCount, SumDecimal *tmpSum, SumDecimal *out, const int &validElements )
//     {
//         if ( tb.thread_rank() == 0 )
//         {
//             for ( int i = 0; i < groupCount; ++i )
//             {
//                 int overStep = i * validElements;
//                 auto target = out + overStep;
//                 auto source = tmpSum + overStep;
//                 int k = blockIdx.x % validElements;
//                 for ( int j = 0; j < validElements; ++j )
//                 {
//                     atomicAdd( (unsigned long long int *) ( target->values + k ), *(unsigned long long int *) ( source->values + k ) );
//                     if ( ++k >= validElements )
//                     {
//                         k = 0;
//                     }
//                 }
//             }
//         }
//     }

//     __global__ void InitLongDataBuf( int64_t *data, const size_t tupleNum )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             *( data + i ) = 0;
//         }
//     }

//     __global__ void DecimalColumnSum( const int8_t *data, const AriesColumnType columnType, const size_t tupleNum, const int *associated,
//                                       const int *groupFlags, const size_t groupCount, SumDecimal *tmpSum, const size_t tmpSumCount, SumDecimal *out,
//                                       const int validElements )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         auto threadBlock = this_thread_block();
//         bool addToOne = gridDim.x == tmpSumCount;
//         if ( addToOne )
//         {
//             size_t oneBlockElementsCount = groupCount * validElements;
//             auto blockTmpSum = tmpSum + oneBlockElementsCount * blockIdx.x;
//             int stride = blockIdx.x == gridDim.x - 1 ? tupleNum % blockDim.x : blockDim.x;
//             stride = stride == 0 ? blockDim.x : stride;
//             DecimalSumInitBuf( (int64_t *) blockTmpSum, oneBlockElementsCount, stride );
//             threadBlock.sync();
//             DecimalSumBlock( data, columnType, tupleNum, associated, groupFlags, blockTmpSum, validElements );
//             threadBlock.sync();
//             DecimalSumAddToTarget( threadBlock, groupCount, blockTmpSum, out, validElements );
//         }
//         else
//         {
//             // if tmpSumCount is 0, indicate add to out directory
//             auto blockTmpSum = tmpSumCount == 0 ? out : tmpSum + groupCount * validElements * ( blockIdx.x % tmpSumCount );
//             DecimalSumBlock( data, columnType, tupleNum, associated, groupFlags, blockTmpSum, validElements );
//         }
//     }

//     __global__ void AddAllGroupsToOne4Sum( SumDecimal *data, const size_t tupleNum, const size_t groupCount, SumDecimal *out,
//                                            const int validElements )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             auto source = data + i;
//             auto target = out + ( ( (int64_t) i / validElements ) % groupCount ) * validElements + i % validElements;
//             atomicAdd( (unsigned long long int *) target->values, *(unsigned long long int *) source->values );
//         }
//     }

//     __global__ void SumDecimalToDecimal( SumDecimal *data, const int validElements, const AriesColumnType columnType, const size_t tupleNum,
//                                          int8_t *out )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int len = columnType.GetDataTypeSize();
//         auto hasNull = columnType.HasNull;
//         auto scale = columnType.DataType.Scale;
//         auto precision = columnType.DataType.Precision;
//         int stride = blockDim.x * gridDim.x;
//         int targetValueStopPos = DECIMAL_VALUE_COUNT - validElements;
//         int carry;
//         int64_t tmp;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             auto source = data + (int64_t) i * validElements;
//             auto target = out + (int64_t)i * len;
//             if ( hasNull )
//             {
//                 *target++ = 1;
//             }
//             carry = 0;
//             auto d = (Decimal *) target;
//             for ( int j = DECIMAL_VALUE_COUNT - 1; j >= 0; --j )
//             {
//                 tmp = ( j >= targetValueStopPos ? source->values[j - targetValueStopPos] : 0 ) + carry;
//                 if ( tmp >= PER_DEC_MAX_SCALE || tmp <= -PER_DEC_MAX_SCALE )
//                 {
//                     carry = tmp / PER_DEC_MAX_SCALE;
//                     tmp %= PER_DEC_MAX_SCALE;
//                 }
//                 else
//                 {
//                     carry = 0;
//                 }
//                 d->values[j] = tmp;
//             }
//             d->frac = scale;
//             d->intg = precision - scale;
//             d->mode = ARIES_MODE_EMPTY;
//             d->error = ERR_OK;
//         }
//     }

//     // max must be set as min: -PER_DEC_MAX_SCALE
//     // min must be set as max: PER_DEC_MAX_SCALE
//     __device__ void DecimalMaxminInitBuf( int *maxes, int *mins, const size_t &groupCount, const int &stride )
//     {
//         for ( int i = threadIdx.x; i < groupCount; i += stride )
//         {
//             // for max
//             *( maxes + i ) = -PER_DEC_MAX_SCALE;
//             // for min
//             *( mins + i ) = PER_DEC_MAX_SCALE;
//         }
//     }

//     __device__ void DecimalMaxminBlock( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum, const int *associated,
//                                         const int *groupFlags, const size_t &groupCount, const int &valIndex, int *maxes, int *mins, int32_t *pFlag )
//     {
//         int len = columnType.GetDataTypeSize();
//         auto prec = columnType.DataType.Precision;
//         auto scale = columnType.DataType.Scale;
//         auto hasNull = columnType.HasNull;
//         bool isCompact = columnType.DataType.ValueType == AriesValueType::COMPACT_DECIMAL;
//         int stride = blockDim.x * gridDim.x;
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             bool valid = false;
//             GET_INT_BIT_FLAG( pFlag, i, valid );
//             if ( !valid )
//             {
//                 continue;
//             }
//             auto pItemData = data + (int64_t) associated[i] * len;
//             if ( hasNull && *pItemData++ == 0 )
//             {
//                 continue;
//             }
//             auto d = isCompact ? Decimal( (CompactDecimal *) pItemData, prec, scale ) : *(Decimal *) pItemData;
//             int groupId = groupFlags[i] - groupFlags[0];
//             atomicMax( maxes + groupId, d.values[valIndex] );
//             atomicMin( mins + groupId, d.values[valIndex] );
//         }
//     }

//     __device__ void DecimalMaxMinToTarget( thread_block tb, const size_t &groupCount, int *tmpMaxes, int *tmpMins, int *maxes, int *mins )
//     {
//         if ( tb.thread_rank() == 0 )
//         {
//             for ( int i = 0; i < groupCount; ++i )
//             {
//                 atomicMax( maxes + i, tmpMaxes[i] );
//                 atomicMin( mins + i, tmpMins[i] );
//             }
//         }
//     }

//     __global__ void AddAllGroupsToOne4Maxmin( int *tmpMaxmins, const size_t groupCount, const size_t tupleNum, int *resMaxmins )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             auto source = tmpMaxmins + i;
//             auto range = i % ( groupCount * 2 );
//             auto target = resMaxmins + range;
//             range < groupCount ? atomicMax( target, *source ) : atomicMin( target, *source );
//         }
//     }

//     __global__ void InitIntDataBuf( int32_t *data, const size_t tupleNum, int32_t initValue = 0 )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             *( data + i ) = initValue;
//         }
//     }

//     __global__ void InitMaxminsDataBuf( int *pMaxmins, const size_t groupCount, const size_t tupleNum )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             auto target = pMaxmins + i;
//             auto range = i % ( groupCount << 1 );
//             *target = range < groupCount ? -PER_DEC_MAX_SCALE : PER_DEC_MAX_SCALE;
//         }
//     }

//     __global__ void CheckIfMaxminsValid( int *pMaxmins, const size_t groupCount, int *maxminValid )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= groupCount )
//         {
//             return;
//         }
//         auto pMins = pMaxmins + groupCount;
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < groupCount; i += stride )
//         {
//             if ( *maxminValid == 0 &&
//                  ( ( *( pMaxmins + i ) == -PER_DEC_MAX_SCALE && *( pMins + i ) != PER_DEC_MAX_SCALE ) || *( pMaxmins + i ) != *( pMins + i ) ) )
//             {
//                 atomicAdd( maxminValid, 1 );
//             }
//         }
//     }

//     // maxes must be set as min: -PER_DEC_MAX_SCALE
//     // mins must be set as max: PER_DEC_MAX_SCALE
//     __global__ void DecimalColumnFindMaxmin( const int8_t *data, const AriesColumnType columnType, const size_t tupleNum, const int *associated,
//                                              const int *groupFlags, const size_t groupCount, const int valIndex, int *tmpMaxmins, int tmpMaxminCount,
//                                              int *resMaxmins, int32_t *pFlag )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }

//         auto threadBlock = this_thread_block();
//         auto resMins = resMaxmins + groupCount;
//         bool addToOne = gridDim.x == tmpMaxminCount;
//         if ( addToOne )
//         {
//             auto blockMaxes = tmpMaxmins + blockIdx.x * ( groupCount << 1 );
//             auto blockMins = blockMaxes + groupCount;
//             int stride = blockIdx.x == gridDim.x - 1 ? tupleNum % blockDim.x : blockDim.x;
//             stride = stride == 0 ? blockDim.x : stride;
//             DecimalMaxminInitBuf( blockMaxes, blockMins, groupCount, stride );
//             threadBlock.sync();
//             DecimalMaxminBlock( data, columnType, tupleNum, associated, groupFlags, groupCount, valIndex, blockMaxes, blockMins, pFlag );
//             threadBlock.sync();
//             DecimalMaxMinToTarget( threadBlock, groupCount, blockMaxes, blockMins, resMaxmins, resMins );
//         }
//         else
//         {
//             // if tmpSumCount is 0, indicate add to out directory
//             auto blockMaxes = tmpMaxminCount == 0 ? resMaxmins : tmpMaxmins + ( groupCount << 1 ) * ( blockIdx.x % tmpMaxminCount );
//             auto blockMins = tmpMaxminCount == 0 ? resMins : blockMaxes + groupCount;
//             DecimalMaxminBlock( data, columnType, tupleNum, associated, groupFlags, groupCount, valIndex, blockMaxes, blockMins, pFlag );
//         }
//     }

//     __global__ void DecimalColumnSetFlag( const int8_t *data, const AriesColumnType columnType, const size_t tupleNum, const int *associated,
//                                           const int *groupFlags, const size_t groupCount, int valIndex, int *pMaxmins, int32_t *pFlags, int *pCount,
//                                           const bool bMax )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }

//         int len = columnType.GetDataTypeSize();
//         auto prec = columnType.DataType.Precision;
//         auto scale = columnType.DataType.Scale;
//         auto hasNull = columnType.HasNull;
//         bool isCompact = columnType.DataType.ValueType == AriesValueType::COMPACT_DECIMAL;
//         auto pMins = pMaxmins + groupCount;
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             bool valid = false;
//             GET_INT_BIT_FLAG( pFlags, i, valid );
//             if ( !valid )
//             {
//                 continue;
//             }
//             int groupId = groupFlags[i] - groupFlags[0];
//             if ( ( pMaxmins[groupId] == -PER_DEC_MAX_SCALE && pMins[groupId] == PER_DEC_MAX_SCALE ) || pMaxmins[groupId] == pMins[groupId] )
//             {
//                 continue;
//             }
//             auto pItemData = data + (int64_t) associated[i] * len;
//             if ( hasNull && *pItemData++ == 0 )
//             {
//                 continue;
//             }
//             auto d = isCompact ? Decimal( (CompactDecimal *) pItemData, prec, scale ) : *(Decimal *) pItemData;
//             int target = bMax ? pMaxmins[groupId] : pMins[groupId];
//             if ( target == d.values[valIndex] )
//             {
//                 atomicAdd( pCount + groupId, 1 );
//             }
//             else
//             {
//                 CLEAR_INT_BIT_FLAG( pFlags, i );
//             }
//         }
//     }

//     __global__ void CheckIfHasSameMaxmin( int *pCount, const size_t groupCount, int *hasSame )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= groupCount )
//         {
//             return;
//         }

//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < groupCount; i += stride )
//         {
//             if ( *hasSame == 0 && pCount[i] > 1 )
//             {
//                 atomicAdd( hasSame, 1 );
//             }
//         }
//     }

//     __global__ void DecimalColumnGetMaxmin( const int8_t *data, const AriesColumnType columnType, const size_t tupleNum, const int *associated,
//                                             const int *groupFlags, const size_t groupCount, int32_t *pFlags, int *pCount, Decimal *out )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int len = columnType.GetDataTypeSize();
//         auto prec = columnType.DataType.Precision;
//         auto scale = columnType.DataType.Scale;
//         auto hasNull = columnType.HasNull;
//         bool isCompact = columnType.DataType.ValueType == AriesValueType::COMPACT_DECIMAL;
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             bool valid = false;
//             GET_INT_BIT_FLAG( pFlags, i, valid );
//             if ( !valid )
//             {
//                 continue;
//             }
//             auto pItemData = data + (int64_t) associated[i] * len;
//             if ( hasNull && *pItemData++ == 0 )
//             {
//                 continue;
//             }
//             int groupId = groupFlags[i] - groupFlags[0];
//             if ( atomicAdd( pCount + groupId, 1 ) == 0 )
//             {
//                 auto d = isCompact ? Decimal( (CompactDecimal *) pItemData, prec, scale ) : *(Decimal *) pItemData;
//                 memcpy( out + groupId, &d, sizeof( Decimal ) );
//             }
//         }
//     }

//     __device__ void DecimalCountInitBuf( int64_t *count, const size_t &groupCount, const int &stride )
//     {
//         for ( int i = threadIdx.x; i < groupCount; i += stride )
//         {
//             *( count + i ) = 0;
//         }
//     }

//     __device__ void DecimalCountBlock( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum, const int *associated,
//                                        const int *groupFlags, const size_t &groupCount, int64_t *out )
//     {
//         int len = columnType.GetDataTypeSize();
//         auto hasNull = columnType.HasNull;
//         int stride = blockDim.x * gridDim.x;
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             if ( hasNull && *( data + (int64_t) associated[i] * len ) == 0 )
//             {
//                 continue;
//             }
//             atomicAdd( (unsigned long long int *) ( out + ( groupFlags[i] - groupFlags[0] ) ), 1 );
//         }
//     }

//     __device__ void DecimalCountAddToTarget( thread_block tb, const size_t &groupCount, int64_t *tmpCount, int64_t *out )
//     {
//         if ( tb.thread_rank() == 0 )
//         {
//             for ( int i = 0; i < groupCount; ++i )
//             {
//                 atomicAdd( (unsigned long long int *) ( out + i ), *(unsigned long long int *) ( tmpCount + i ) );
//             }
//         }
//     }

//     __global__ void DecimalColumnCount( const int8_t *data, const AriesColumnType columnType, const size_t tupleNum, const int *associated,
//                                         const int *groupFlags, const size_t groupCount, int64_t *pTmpCount, const int tmpCountBufCount, int64_t *out )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }

//         auto threadBlock = this_thread_block();
//         bool addToOne = gridDim.x == tmpCountBufCount;
//         if ( addToOne )
//         {
//             auto blockTmpCount = pTmpCount + blockIdx.x * groupCount;
//             int stride = blockIdx.x == gridDim.x - 1 ? tupleNum % blockDim.x : blockDim.x;
//             stride = stride == 0 ? blockDim.x : stride;
//             DecimalCountInitBuf( blockTmpCount, groupCount, stride );
//             threadBlock.sync();
//             DecimalCountBlock( data, columnType, tupleNum, associated, groupFlags, groupCount, blockTmpCount );
//             threadBlock.sync();
//             DecimalCountAddToTarget( threadBlock, groupCount, blockTmpCount, out );
//         }
//         else
//         {
//             // if tmpCountBufCount is 0, indicate add to out directory
//             auto blockTmpCount = tmpCountBufCount == 0 ? out : pTmpCount + ( blockIdx.x % tmpCountBufCount ) * groupCount;
//             DecimalCountBlock( data, columnType, tupleNum, associated, groupFlags, groupCount, blockTmpCount );
//         }
//     }

//     __global__ void AddAllGroupsToOne4Count( int64_t *data, size_t tupleNum, size_t groupCount, int64_t *out )
//     {
//         int tid = blockIdx.x * blockDim.x + threadIdx.x;
//         if ( tid >= tupleNum )
//         {
//             return;
//         }
//         int stride = blockDim.x * gridDim.x;
//         for ( int i = tid; i < tupleNum; i += stride )
//         {
//             auto source = data + i;
//             auto target = out + i % groupCount;
//             atomicAdd( (unsigned long long int *) target, *(unsigned long long int *) source );
//         }
//     }

//     AriesDecimalAlgorithm::AriesDecimalAlgorithm()
//     {
// #ifdef ARIES_PROFILE
//         aries::CPU_Timer t;
//         t.begin();
// #endif
//         m_ctx = make_unique< standard_context_t >();
//         size_t freeSize, total;
//         cudaMemGetInfo( &freeSize, &total );
//         m_memorySizeLimit = total * GPU_MEMORY_LIMIT_PERCENT;
//         m_gpuMemorySize = total;
// #ifdef ARIES_PROFILE
//         LOG( INFO ) << "!!!!!!!!!!!!!!!!!!!!standard_context_t init time cost is:" << t.end() << endl;
// #endif
//     }

//     AriesDecimalAlgorithm::~AriesDecimalAlgorithm()
//     {
//     }

//     AriesDataBufferSPtr AriesDecimalAlgorithm::AggregateColumnSum( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum,
//                                                                    const int *associated, const int *groupFlags, const size_t &groupCount,
//                                                                    const SumStrategy &strategy )
//     {
//         if ( tupleNum >= PER_DEC_MAX_SCALE )
//         {
//             ARIES_ASSERT( 0, "data block size is too large: " + to_string( tupleNum ) );
//         }
//         size_t blockCount = ROUNDUP( tupleNum, BLOCK_THREAD_COUNT );
//         int validElements = GetDecimalValidElementsCount( columnType.DataType.Precision, columnType.DataType.Scale );
//         if ( validElements > DECIMAL_VALUE_COUNT )
//         {
//             validElements = DECIMAL_VALUE_COUNT;
//         }
//         size_t oneTmpSumLongCont = groupCount * validElements;
//         size_t oneTmpSumSize = oneTmpSumLongCont * sizeof( SumDecimal );
//         mem_t< SumDecimal > sumResult;
//         SumDecimal *pSumResult = nullptr;
//         size_t tmpSumCount = 0;
//         if ( strategy == SumStrategy::SOLE_TEMP_SUM )
//         {
//             tmpSumCount = blockCount;
//         }
//         else if ( strategy == SumStrategy::SHARE_TEMP_SUM )
//         {
//             size_t sourceDataSize = ( columnType.GetDataTypeSize() + sizeof( int ) * 2 ) * tupleNum;
//             if ( m_memorySizeLimit > sourceDataSize + oneTmpSumSize )
//             {
//                 tmpSumCount = ROUNDUP( m_memorySizeLimit - sourceDataSize - oneTmpSumSize, oneTmpSumSize );
//                 if ( tmpSumCount > blockCount )
//                 {
//                     // change to SOLE_TEMP_SUM
//                     tmpSumCount = blockCount;
//                 }
//                 else if ( tmpSumCount < THRESHOLD_4_TEMP_COUNT_TOO_LESS )
//                 {
//                     // change to NO_TEMP_SUM
//                     tmpSumCount = 0;
//                 }
//             }
//             // else  change to NO_TEMP_SUM
//         }
//         if ( tmpSumCount )
//         {
//             size_t tmpTotalSumCount = oneTmpSumLongCont * tmpSumCount;
//             mem_t< SumDecimal > tmpSumRes( tmpTotalSumCount );
//             SumDecimal *pTmpSumRes = tmpSumRes.data();
//             if ( tmpSumCount != blockCount )  // SHARE_TEMP_SUM
//             {
//                 InitLongDataBuf<<< ROUNDUP( tmpTotalSumCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( (int64_t *) pTmpSumRes, tmpTotalSumCount );
//                 DecimalColumnSum<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, pTmpSumRes,
//                                                                         tmpSumCount, nullptr, validElements );
//                 sumResult = mem_t< SumDecimal >( oneTmpSumLongCont );
//                 pSumResult = sumResult.data();
//                 InitLongDataBuf<<< ROUNDUP( oneTmpSumLongCont, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( (int64_t *) pSumResult, oneTmpSumLongCont );
//                 AddAllGroupsToOne4Sum<<< ROUNDUP( tmpTotalSumCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pTmpSumRes, tmpTotalSumCount,
//                                                                                                                   groupCount, pSumResult, validElements );
//             }
//             else
//             {
//                 sumResult = mem_t< SumDecimal >( oneTmpSumLongCont );
//                 pSumResult = sumResult.data();
//                 InitLongDataBuf<<< ROUNDUP( oneTmpSumLongCont, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( (int64_t *) pSumResult, oneTmpSumLongCont );
//                 DecimalColumnSum<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, pTmpSumRes,
//                                                                         tmpSumCount, pSumResult, validElements );
//             }
//         }
//         else
//         {
//             sumResult = mem_t< SumDecimal >( oneTmpSumLongCont );
//             pSumResult = sumResult.data();
//             InitLongDataBuf<<< ROUNDUP( oneTmpSumLongCont, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( (int64_t *) pSumResult, oneTmpSumLongCont );
//             DecimalColumnSum<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, nullptr, 0, pSumResult,
//                                                                     validElements );
//         }
//         // SumDecimal --> decimal
//         // use INNER_MAX_PRECISION indicate max vailable data
//         uint16_t precision = columnType.DataType.Precision + DIG_PER_INT32;
//         if ( precision > INNER_MAX_PRECISION )
//         {
//             precision = INNER_MAX_PRECISION;
//         }
//         AriesColumnType resultType{ { AriesValueType::DECIMAL, precision, columnType.DataType.Scale }, true };
//         auto result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
//         result->PrefetchToGpu();
//         SumDecimalToDecimal<<< ROUNDUP( groupCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pSumResult, validElements, resultType, groupCount,
//                                                                                                   result->GetData() );
//         cudaDeviceSynchronize();
//         return result;
//     }

//     void AriesDecimalAlgorithm::CpuSumSingleThread( const int8_t *data, const AriesColumnType &columnType, const size_t tupleNum,
//                                                     const int *associated, const int *groupFlags, int8_t *resData, const AriesColumnType &targetType )
//     {
//         int len = columnType.GetDataTypeSize();
//         auto prec = columnType.DataType.Precision;
//         auto scale = columnType.DataType.Scale;
//         auto hasNull = columnType.HasNull;
//         bool isCompact = columnType.DataType.ValueType == AriesValueType::COMPACT_DECIMAL;
//         auto tItemLen = targetType.GetDataTypeSize();
//         auto tHasNull = targetType.HasNull;
//         for ( size_t i = 0; i < tupleNum; ++i )
//         {
//             auto pItemData = data + (int64_t) associated[i] * len;
//             if ( hasNull && *pItemData++ == 0 )
//             {
//                 continue;
//             }
//             const auto d = isCompact ? Decimal( (CompactDecimal *) pItemData, prec, scale ) : *(Decimal *) pItemData;
//             auto tmpTarget = resData + (int64_t) groupFlags[i] * tItemLen;
//             if ( tHasNull )
//             {
//                 *tmpTarget++ = 1;
//             }
//             *(Decimal *) tmpTarget += d;
//         }
//     }

//     size_t AriesDecimalAlgorithm::findNextGroupStartIndex( const int *groupFlags, const size_t count, const int *groups )
//     {
//         return groups[groupFlags[count - 1] + 1];
//     }

//     AriesDataBufferSPtr AriesDecimalAlgorithm::AggregateColumnSumCpu( const AriesDataBufferSPtr &srcColumn, const AriesInt32ArraySPtr &associatedSPtr,
//                                                                       const AriesInt32ArraySPtr &groupFlagsSPtr, const AriesInt32ArraySPtr &groups )
//     {
//         // TODO 测试结果显示: 同一SQL(tpch1_100_query.q18),执行PrefetchToCpu比不执行PrefetchToCpu时AriesGroupNode所花的时间更多,所以暂时屏蔽
//         // srcColumn->PrefetchToCpu();
//         // associatedSPtr->PrefetchToCpu();
//         // groupFlagsSPtr->PrefetchToCpu();
//         // groups->PrefetchToCpu();
//         int8_t *srcData = srcColumn->GetData();
//         auto tupleNum = srcColumn->GetItemCount();
//         auto columnType = srcColumn->GetDataType();
//         int sItemLen = columnType.GetDataTypeSize();
//         int *associated = associatedSPtr->GetData();
//         int *groupFlags = groupFlagsSPtr->GetData();
//         auto groupCount = groups->GetItemCount();

//         uint16_t precision = columnType.DataType.Precision + DIG_PER_INT32;
//         if ( precision > INNER_MAX_PRECISION )
//         {
//             precision = INNER_MAX_PRECISION;
//         }
//         AriesColumnType resultType{ { AriesValueType::DECIMAL, precision, columnType.DataType.Scale }, true };
//         int tItemLen = resultType.GetDataTypeSize();
//         auto result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
//         result->PrefetchToCpu();
//         int8_t *resData = result->GetData();
//         memset( resData, 0x00, result->GetTotalBytes() );

//         vector< future< void > > workThreads;
//         int blockCount = GetThreadNum( tupleNum );
//         size_t blockSize = ROUNDUP( tupleNum, blockCount );
//         size_t overItems = 0;
//         for ( int i = 0; i < blockCount; ++i )
//         {
//             if ( overItems >= tupleNum)
//             {
//                 break;
//             }
//             auto a = associated + overItems;
//             auto g = groupFlags + overItems;
//             size_t itemCount = tupleNum - overItems;
//             if ( itemCount > blockSize )
//             {
//                 itemCount = blockSize;
//                 auto startIndex = findNextGroupStartIndex( groupFlags, overItems + itemCount, groups->GetData() );
//                 itemCount = startIndex - overItems;
//                 overItems = startIndex;
//             }
//             workThreads.push_back(
//                 std::async( std::launch::async, [=] { CpuSumSingleThread( srcData, columnType, itemCount, a, g, resData, resultType ); } ) );
//         }
//         for ( auto &t : workThreads )
//             t.wait();

//         return result;
//     }

//     AriesDataBufferSPtr AriesDecimalAlgorithm::AggregateColumnGetMaxmin( const int8_t *data, const AriesColumnType &columnType,
//                                                                          const size_t &tupleNum, const int *associated, const int *groupFlags,
//                                                                          const size_t &groupCount, const bool &bMax, const SumStrategy &strategy )
//     {
//         size_t flagsCount = ROUNDUP( tupleNum, 32 );
//         mem_t< int32_t > selectedFlag( flagsCount );
//         int32_t *pFlags = (int32_t *) selectedFlag.data();
//         InitIntDataBuf<<< ROUNDUP( flagsCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pFlags, flagsCount, 0xFFFFFFFF );

//         size_t oneBlockResIntCount = groupCount << 1;
//         mem_t< int > resMaxmins( oneBlockResIntCount );
//         int *pResMaxmins = resMaxmins.data();
//         mem_t< int > selectedCount( groupCount );
//         int *pCount = selectedCount.data();

//         managed_mem_t< int > maxMinValid( 1, *m_ctx );
//         int *pMaxminValid = maxMinValid.data();
//         *pMaxminValid = 0;
//         managed_mem_t< int > hasSame( 1, *m_ctx );
//         int *pHasSame = hasSame.data();
//         *pHasSame = 0;

//         size_t tmpSumCount = 0;
//         int blockCount = ROUNDUP( tupleNum, BLOCK_THREAD_COUNT );
//         if ( strategy == SumStrategy::SOLE_TEMP_SUM )
//         {
//             tmpSumCount = blockCount;
//         }
//         else if ( strategy == SumStrategy::SHARE_TEMP_SUM )
//         {
//             size_t sourceDataSize = ( columnType.GetDataTypeSize() + sizeof( int ) * 2 ) * tupleNum + ( flagsCount + groupCount * 3 ) * sizeof( int );
//             size_t oneTmpSize = oneBlockResIntCount * sizeof( int );
//             if ( sourceDataSize + oneTmpSize > m_memorySizeLimit )
//             {
//                 tmpSumCount = ROUNDUP( m_memorySizeLimit - sourceDataSize - oneTmpSize, oneTmpSize );
//                 if ( tmpSumCount > blockCount )
//                 {
//                     // change to SOLE_TEMP_SUM
//                     tmpSumCount = blockCount;
//                 }
//                 else if ( tmpSumCount < THRESHOLD_4_TEMP_COUNT_TOO_LESS )
//                 {
//                     // change to NO_TEMP_SUM
//                     tmpSumCount = 0;
//                 }
//             }
//             // else  change to NO_TEMP_SUM
//         }

//         int validValCount = GetDecimalValidElementsCount( columnType.DataType.Precision, columnType.DataType.Scale );
//         for ( int i = DECIMAL_VALUE_COUNT - validValCount; i < DECIMAL_VALUE_COUNT; ++i )
//         { 
//             // set all pMaxes as min data, all pMaxes as max data
//             InitMaxminsDataBuf<<< ROUNDUP( oneBlockResIntCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pResMaxmins, groupCount,
//                                                                                                               oneBlockResIntCount );
//             if ( tmpSumCount )
//             {
//                 size_t totalIntCount = oneBlockResIntCount * tmpSumCount;
//                 mem_t< int > tmpMaxmins( totalIntCount );
//                 int *pTmpMaxmins = tmpMaxmins.data();
//                 if ( tmpSumCount != blockCount )
//                 {
//                     InitMaxminsDataBuf<<< ROUNDUP( totalIntCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pTmpMaxmins, groupCount,
//                                                                                                                 totalIntCount );
//                     DecimalColumnFindMaxmin<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, i,
//                                                                                    pTmpMaxmins, tmpSumCount, pResMaxmins, pFlags );
//                     AddAllGroupsToOne4Maxmin<<< ROUNDUP(totalIntCount, BLOCK_THREAD_COUNT), BLOCK_THREAD_COUNT >>>( pTmpMaxmins, groupCount,
//                                                                                                                     totalIntCount, pResMaxmins );
//                 }
//                 else
//                 {
//                     DecimalColumnFindMaxmin<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, i,
//                                                                                    pTmpMaxmins, tmpSumCount, pResMaxmins, pFlags );
//                 }
//             }
//             else
//             {
//                 DecimalColumnFindMaxmin<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, i,
//                                                                                nullptr, 0, pResMaxmins, pFlags );
//             }
            
//             CheckIfMaxminsValid<<< ROUNDUP( groupCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>(pResMaxmins, groupCount, pMaxminValid);
//             cudaDeviceSynchronize();
//             if ( *pMaxminValid == 0 )
//             {
//                 continue;
//             }
//             // initialize selected count
//             InitIntDataBuf<<< ROUNDUP( groupCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pCount, groupCount );
//             DecimalColumnSetFlag<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, i, pResMaxmins,
//                                                                        pFlags, pCount, bMax );
//             CheckIfHasSameMaxmin<<< blockCount, BLOCK_THREAD_COUNT >>>(pCount, groupCount, pHasSame );
//             cudaDeviceSynchronize();
//             if ( *pHasSame == 0 )
//             {
//                 break;
//             }
//         }
//         AriesColumnType resultType{ { AriesValueType::DECIMAL, columnType.DataType.Precision, columnType.DataType.Scale }, false };
//         auto result = std::make_shared< AriesDataBuffer >( resultType, groupCount, true );

//         // initialize selected count
//         InitIntDataBuf<<< ROUNDUP( groupCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pCount, groupCount );
//         DecimalColumnGetMaxmin<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, pFlags, pCount,
//                                                                      (Decimal *) result->GetData() );
//         cudaDeviceSynchronize();
//         return result;
//     }

//     AriesDataBufferSPtr AriesDecimalAlgorithm::AggregateColumnCount( const int8_t *data, const AriesColumnType &columnType, const size_t &tupleNum,
//                                                                      const int *associated, const int *groupFlags, const size_t &groupCount,
//                                                                      const SumStrategy &strategy )
//     {
//         AriesDataBufferSPtr result;
//         AriesColumnType resultType{ { AriesValueType::INT64, 1 }, false };

//         int blockCount = ROUNDUP( tupleNum, BLOCK_THREAD_COUNT );
//         size_t oneTmpResultSize = sizeof( int64_t ) * groupCount;
//         size_t tmpResultCount = 0;
//         if ( strategy == SumStrategy::SOLE_TEMP_SUM )
//         {
//             tmpResultCount = blockCount;
//         }
//         else if ( strategy == SumStrategy::SHARE_TEMP_SUM )
//         {
//             size_t sourceDataItemSize = sizeof( int ) * 2;
//             if ( columnType.isNullable() )
//             {
//                 sourceDataItemSize += columnType.GetDataTypeSize();
//             }
//             size_t sourceDataSize = sourceDataItemSize * tupleNum;
//             if ( m_memorySizeLimit > sourceDataSize - oneTmpResultSize )
//             {
//                 tmpResultCount = ROUNDUP( m_memorySizeLimit - sourceDataSize - oneTmpResultSize, oneTmpResultSize );
//                 if ( tmpResultCount > blockCount )
//                 {
//                     // change to SOLE_TEMP_SUM
//                     tmpResultCount = blockCount;
//                 }
//                 else if ( tmpResultCount < THRESHOLD_4_TEMP_COUNT_TOO_LESS )
//                 {
//                     // change to NO_TEMP_SUM
//                     tmpResultCount = 0;
//                 }
//             }
//             // else change to NO_TEMP_SUM
//         }
//         if ( tmpResultCount )
//         {
//             size_t tmpTotalCount = groupCount * tmpResultCount;
//             mem_t< int64_t > tmpCountResult( tmpTotalCount );
//             int64_t *pTmpResult = tmpCountResult.data();
//             if ( tmpResultCount != blockCount )  // SHARE_TEMP_SUM
//             {
//                 InitLongDataBuf<<< ROUNDUP( tmpTotalCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( (int64_t *) pTmpResult, tmpTotalCount );
//                 DecimalColumnCount<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, pTmpResult,
//                                                                           tmpResultCount, nullptr );
//                 result = std::make_shared< AriesDataBuffer >( resultType, groupCount, true );
//                 result->PrefetchToGpu();
//                 AddAllGroupsToOne4Count<<< ROUNDUP( tmpTotalCount, BLOCK_THREAD_COUNT ), BLOCK_THREAD_COUNT >>>( pTmpResult, tmpTotalCount,
//                                                                                                                  groupCount,
//                                                                                                                  (int64_t *) result->GetData() );
//             }
//             else
//             {
//                 result = std::make_shared< AriesDataBuffer >( resultType, groupCount, true );
//                 result->PrefetchToGpu();
//                 DecimalColumnCount<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, pTmpResult,
//                                                                           tmpResultCount, (int64_t *) result->GetData() );
//             }
//         }
//         else
//         {
//             result = std::make_shared< AriesDataBuffer >( resultType, groupCount, true );
//             result->PrefetchToGpu();
//             DecimalColumnCount<<< blockCount, BLOCK_THREAD_COUNT >>>( data, columnType, tupleNum, associated, groupFlags, groupCount, nullptr,
//                                                                        tmpResultCount, (int64_t *) result->GetData() );
//         }
//         cudaDeviceSynchronize();
//         return result;
//     }

//     SumStrategy AriesDecimalAlgorithm::GetRuntimeStrategy( const AriesDataBufferSPtr &srcColumn, const AriesAggFunctionType &aggType,
//                                                            const size_t &groupCount, const size_t handleCount )
//     {
//         size_t handleDataCount = handleCount;
//         if ( handleCount == 0 )
//         {
//             handleDataCount = srcColumn->GetItemCount();
//         }
//         auto columnType = srcColumn->GetDataType();
//         if ( columnType.DataType.ValueType != AriesValueType::DECIMAL && columnType.DataType.ValueType != AriesValueType::COMPACT_DECIMAL )
//         {
//             return SumStrategy::NONE;
//         }
//         checkPrecision( columnType, srcColumn );
//         size_t blockCount = ROUNDUP( handleDataCount, BLOCK_THREAD_COUNT );
//         if ( blockCount <= THRESHOLD_4_BLOCK_COUNT_TOO_LESS )
//         {
//             return SumStrategy::NO_TEMP_SUM;
//         }
//         size_t sourceItemSize = columnType.GetDataTypeSize() + sizeof( int ) * 2;
//         size_t oneGroupSize = 0;
//         switch ( aggType )
//         {
//         case AriesAggFunctionType::SUM: {
//             int validElements = NEEDELEMENTS( columnType.DataType.Precision - columnType.DataType.Scale ) + NEEDELEMENTS( columnType.DataType.Scale );
//             if ( validElements > DECIMAL_VALUE_COUNT )
//             {
//                 validElements = DECIMAL_VALUE_COUNT;
//             }
//             oneGroupSize = validElements * sizeof( SumDecimal );
//             break;
//         }
//         case AriesAggFunctionType::COUNT: {
//             oneGroupSize = sizeof( int64_t );
//             if ( columnType.isNullable() )
//             {
//                 sourceItemSize -= columnType.GetDataTypeSize();
//             }
//             break;
//         }
//         case AriesAggFunctionType::MAX:
//         case AriesAggFunctionType::MIN:
//             sourceItemSize += ROUNDUP( handleDataCount, 32 ) / handleDataCount;  // BIT flags
//             oneGroupSize = sizeof( int ) * 5;  // tmpMaxmin + resMaxmin + selectedCount
//             break;
//         default:
//             // AVG will be replaced by sum and count for data partition process
//             ARIES_ASSERT( 0, "AVG will be replaced by sum and count for data partition process" );
//             // break;
//         }
//         size_t oneTmpResultSize = groupCount * oneGroupSize;
//         size_t sourceDataSize = sourceItemSize * handleDataCount;
//         size_t tmpResultCount = 0;
//         if ( sourceDataSize + oneTmpResultSize < m_memorySizeLimit )
//         {
//             tmpResultCount = ROUNDUP( m_memorySizeLimit - sourceDataSize - oneTmpResultSize, oneTmpResultSize );
//         }
//         if ( tmpResultCount > blockCount )
//         {
//             tmpResultCount = blockCount;
//         }
//         if ( tmpResultCount <= THRESHOLD_4_TEMP_COUNT_TOO_LESS || handleDataCount / groupCount < THRESHOLD_4_AGGREGATED_BY_ONESTEP )
//         {
//             return SumStrategy::NO_TEMP_SUM;
//         }
//         return tmpResultCount == blockCount ? SumStrategy::SOLE_TEMP_SUM : SumStrategy::SHARE_TEMP_SUM;
//     }

//     AriesDataBufferSPtr AriesDecimalAlgorithm::AggregateColumn( const AriesDataBufferSPtr &srcColumn, const AriesAggFunctionType &aggType,
//                                                                 const AriesInt32ArraySPtr &associatedSPtr, const AriesInt32ArraySPtr &groupFlagsSPtr,
//                                                                 const AriesInt32ArraySPtr &groups, const SumStrategy &strategy, const size_t startIndex,
//                                                                 const size_t handleCount )
//     {
//         const int8_t *data = srcColumn->GetData();
//         AriesColumnType columnType = srcColumn->GetDataType();
//         size_t tupleNum = srcColumn->GetItemCount();
//         auto groupFlags = (int *) groupFlagsSPtr->GetData() + startIndex;
//         auto associated = (int *) associatedSPtr->GetData() + startIndex;
//         int groupCount = groups->GetItemCount();
//         size_t handleDataCount = handleCount;
//         if ( handleDataCount )
//         {
//             groupCount = groupFlags[handleDataCount - 1] - groupFlags[0] + 1;
//         }
//         else
//         {
//             handleDataCount = tupleNum;
//         }
//         checkPrecision( columnType, srcColumn );
//         SumStrategy currentStrategy = strategy;
//         if ( currentStrategy == SumStrategy::NONE )
//         {
//             currentStrategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( srcColumn, aggType, groupCount, handleDataCount );
//         }
//         srcColumn->PrefetchToGpu();
//         AriesDataBufferSPtr result;
//         switch ( aggType )
//         {
//         case AriesAggFunctionType::SUM:
//         {
//             bool handled = false;
// //            if ( currentStrategy == SumStrategy::NO_TEMP_SUM )
// //            {
// //                // source data + SumDecimal data + associated + groupFlags
// //                int validElementsCount = GetDecimalValidElementsCount( columnType.DataType.Precision, columnType.DataType.Scale );
// //                size_t totalSize = tupleNum * ( columnType.GetDataTypeSize() + 2 * sizeof( int ) ) + validElementsCount * sizeof( long ) * groupCount;
// //                // 爆显存
// //                if ( totalSize > m_gpuMemorySize && handleDataCount == tupleNum )
// //                {
// //                    int threads = GetThreadNum( tupleNum );
// //                    // 每组重复率较小(重复个数不超过groupCount 1/threads个, 确保每一个线程处理数据相差不大)时使用内部CPU sum方法
// //                    if ( tupleNum < groupCount + groupCount / threads )
// //                    {
// //                        result = AggregateColumnSumCpu( srcColumn, associatedSPtr, groupFlagsSPtr, groups );
// //                        handled = true;
// //                    }
// //                    // DECIMAL 使用外部提供的CPU sum的方案
// //                    else if ( columnType.DataType.ValueType == AriesValueType::DECIMAL )
// //                    {
// //                        // TODO 测试结果显示:
// //                        // 1, 同一SQL(tpch1_100_query.q18),执行PrefetchToCpu比不执行PrefetchToCpu时AriesGroupNode所花的时间更多,所以暂时屏蔽
// //                        // 2, SegmentReduce方法对tpch1_100_query.q18效果比较差,可以针对特殊情况进行优化
// //                        // srcColumn->PrefetchToCpu();
// //                        // associatedSPtr->PrefetchToCpu();
// //                        // groupFlagsSPtr->PrefetchToCpu();
// //
// //                        std::unique_ptr< int32_t[] > pAssociated( new int32_t[associatedSPtr->GetItemCount()] );
// //                        ARIES_CALL_CUDA_API( cudaMemcpy( pAssociated.get(), associated, associatedSPtr->GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
// //                        std::unique_ptr< int32_t[] > pGroups( new int32_t[groups->GetItemCount()] );
// //                        ARIES_CALL_CUDA_API( cudaMemcpy( pGroups.get(), groups->GetData(), groups->GetTotalBytes(), cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
// //
// //                        if ( columnType.HasNull )
// //                        {
// //                            typedef nullable_type< Decimal > nulDecimal;
// //                            nulDecimal init;
// //                            init.flag = 1;
// //                            init.value = Decimal();
// //                            vector< nulDecimal > ret = SegmentReduce( (nulDecimal *) data, pAssociated.get(), tupleNum, pGroups.get(), groupCount,
// //                                                                      agg_sum_t< nulDecimal >(), init );
// //                            AriesColumnType resultType{ { AriesValueType::DECIMAL, INNER_MAX_PRECISION, columnType.DataType.Scale }, true };
// //                            int tItemLen = resultType.GetDataTypeSize();
// //                            result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
// //                            result->PrefetchToCpu();
// //                            int8_t *resData = result->GetData();
// //                            memcpy( resData, ret.data(), result->GetTotalBytes() );
// //                        }
// //                        else
// //                        {
// //                            vector< Decimal > ret = SegmentReduce( (Decimal *) data, pAssociated.get(), tupleNum, pGroups.get(), groupCount,
// //                                                                   agg_sum_t< Decimal >(), Decimal() );
// //                            AriesColumnType resultType{ { AriesValueType::DECIMAL, INNER_MAX_PRECISION, columnType.DataType.Scale }, true };
// //                            int tItemLen = resultType.GetDataTypeSize();
// //                            result = std::make_shared< AriesDataBuffer >( resultType, groupCount );
// //                            int8_t *resData = result->GetData();
// //                            for ( size_t i = 0; i < ret.size(); ++i )
// //                            {
// //                                auto target = resData + i * tItemLen;
// //                                *target++ = 1;
// //                                memcpy( target, &ret[i], sizeof( Decimal ) );
// //                            }
// //                        }
// //                        handled = true;
// //                    }
// //                }
// //            }
//             if ( !handled )
//             {
//                 result = AggregateColumnSum( data, columnType, handleDataCount, associated, groupFlags, groupCount, currentStrategy );
//             }
//             break;
//         }
//         case AriesAggFunctionType::MAX:
//             result = AggregateColumnGetMaxmin( data, columnType, handleDataCount, associated, groupFlags, groupCount, true, currentStrategy );
//             break;
//         case AriesAggFunctionType::MIN:
//             result = AggregateColumnGetMaxmin( data, columnType, handleDataCount, associated, groupFlags, groupCount, false, currentStrategy );
//             break;
//         case AriesAggFunctionType::COUNT:
//             result = AggregateColumnCount( data, columnType, handleDataCount, associated, groupFlags, groupCount, currentStrategy );
//             break;
//         case AriesAggFunctionType::AVG:
//             // AVG will be replaced by sum and count for data partition process
//             ARIES_ASSERT( 0, "AVG will be replaced by sum and count for data partition process" );
//             // break;
//         default:
//             assert( 0 );
//             ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "agg function in decimal " + GetAriesAggFunctionTypeName( aggType ) );
//         }

//         return result;
//     }

//     void AriesDecimalAlgorithm::checkPrecision( AriesColumnType &type, const AriesDataBufferSPtr &srcColumn )
//     {
//         // set scale only for decimal AriesColumnType if not set precision
//         if ( type.DataType.ValueType == AriesValueType::DECIMAL && type.DataType.Scale == 0 && type.DataType.Precision == 0 )
//         {
//             int8_t *validDecimal = nullptr;
//             if ( type.HasNull )
//             {
//                 int8_t *p;
//                 for ( size_t i = 0; i < srcColumn->GetItemCount(); i++ )
//                 {
//                     p = srcColumn->GetItemDataAt( i );
//                     if ( *p++ == 1 )
//                     {
//                         validDecimal = p;
//                         break;
//                     }
//                 }
//             }
//             else
//             {
//                 validDecimal = srcColumn->GetItemDataAt( 0 );
//             }
//             type.DataType.Scale = GET_CALC_FRAC( ( (Decimal *) validDecimal )->error );
//             if ( type.DataType.Scale == 0 )
//             {
//                 type.DataType.Scale = ( (Decimal *) validDecimal )->frac;
//             }
//             int intg = GET_CALC_INTG( ( (Decimal *) validDecimal )->mode );
//             type.DataType.Precision = intg == 0 ? SUPPORTED_MAX_PRECISION : intg + type.DataType.Scale;
//         }
//     }

// END_ARIES_ACC_NAMESPACE
