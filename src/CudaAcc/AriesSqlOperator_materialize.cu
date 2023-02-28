/*
 * AriesSqlOperator_materialize.cu
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */
 #include "AriesSqlOperator_materialize.h"
 #include "AriesEngineAlgorithm.h"
 #include "AriesEngineException.h"
 #include "utils/string_util.h"
 
 BEGIN_ARIES_ACC_NAMESPACE
 
     template< typename index_type_t >
     void DoMaterializeRowPosRangeColumn(const std::vector< AriesDataBufferSPtr > &dataBlocks, int64_t* pBlockSizePrefixSum, index_type_t* pIndices,
             size_t tupleNum, AriesColumnType resultType, int8_t* pResult, context_t& context )
     {
         size_t inputLen = dataBlocks[0]->GetDataType().GetDataTypeSize();
         size_t len = resultType.GetDataTypeSize();
 
         int blockCount = dataBlocks.size();
         AriesManagedArray< int > rowposes( blockCount );
         auto pRowpos = rowposes.GetData();
         int index = 0;
         for ( const auto& block : dataBlocks )
         {
             auto rowposBlock = std::dynamic_pointer_cast< AriesRowPosRangeBuffer >( block );
             rowposes[ index ] = rowposBlock->GetStart();
         }
 
         if ( inputLen == len )
         {
             auto k = [ = ]ARIES_DEVICE( int index )
             {
                 auto pos = pIndices[ index ];
                 if ( pos != -1 )
                 {
                     int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                     auto start = pRowpos[ blockIndex ];
                     if ( start > 0 )
                     {
                         *( int* )( pResult + index * len ) = start + blockIndex;
                     }
                     else
                     {
                         *( int* )( pResult + index * len ) = start - blockIndex;
                     }
                 }
             };
             transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
         }
         else
         {
             ARIES_ASSERT( inputLen == len - 1, "inputLen should be len - 1" );
             auto k = [ = ]ARIES_DEVICE( int index )
             {
                 auto pos = pIndices[ index ];
                 if ( pos != -1 )
                 {
                     int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                     auto start = pRowpos[ blockIndex ];
                     *( pResult + index * len ) = 1;
                     if ( start > 0 )
                     {
                         *( int* )( pResult + index * len + 1 ) = start + blockIndex;
                     }
                     else
                     {
                         *( int* )( pResult + index * len + 1 ) = start - blockIndex;
                     }
                 }
                 else
                 {
                     *( pResult + index * len ) = 0;
                 }
             };
             transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
         }
     }
 
     template< typename index_type_t >
     void DoMaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks,
                               int64_t* pBlockSizePrefixSum,
                               index_type_t* pIndices,
                               size_t tupleNum,
                               AriesColumnType resultType,
                               int8_t* pResult,
                               context_t& context )
     {
         int blockCount = dataBlocks.size();
         AriesManagedArray< int8_t* > dataBuffers( blockCount );
         int i = 0;
         for( const auto& block : dataBlocks )
             dataBuffers[i++] = block->GetData();
         int8_t** pBuffers = dataBuffers.GetData();
         size_t len = resultType.GetDataTypeSize();
         size_t inputLen = dataBlocks[0]->GetDataType().GetDataTypeSize();
         dataBuffers.PrefetchToGpu();
         if( inputLen == len )
         {
             auto k = [=] ARIES_DEVICE(int index)
             {
                 index_type_t pos = pIndices[ index ];
                 int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                 memcpy( pResult + index * len, pBuffers[ blockIndex ] + ( pos - pBlockSizePrefixSum[ blockIndex ] ) * len, len );
             };
             transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
         }
         else
         {
             ARIES_ASSERT( inputLen == len - 1, "inputLen should be len - 1" );
             auto k = [=] ARIES_DEVICE(int index)
             {
                 index_type_t pos = pIndices[ index ];
                 *( pResult + index * len ) = 1;
                 int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                 memcpy( pResult + index * len + 1, pBuffers[ blockIndex ] + ( pos - pBlockSizePrefixSum[ blockIndex ] ) * inputLen, inputLen );
             };
             transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
         }
         context.synchronize();
     }
 
     template< typename index_type_t >
     void DoMaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks,
                               int64_t* pBlockSizePrefixSum,
                               nullable_type< index_type_t >* pIndices,
                               size_t tupleNum,
                               AriesColumnType resultType,
                               int8_t* pResult,
                               context_t& context )
     {
         int blockCount = dataBlocks.size();
         AriesManagedArray< int8_t* > dataBuffers( blockCount );
         int i = 0;
         for( const auto& block : dataBlocks )
             dataBuffers[i++] = block->GetData();
         int8_t** pBuffers = dataBuffers.GetData();
         size_t len = resultType.GetDataTypeSize();
         size_t inputLen = dataBlocks[0]->GetDataType().GetDataTypeSize();
         dataBuffers.PrefetchToGpu();
 
         if( inputLen == len )
         {
             auto k = [=] ARIES_DEVICE(int index)
             {
                 // TODO: use pos2 will cause 'misaligned address' error
                 // nullable_type< index_type_t > nullablePos = pIndices[ index ];
                 // index_type_t pos2 = nullablePos.value;
                 if ( 0 == pIndices[ index ].flag )
                 {
                     *( pResult + index * len ) = 0;
                 }
                 else
                 {
                     index_type_t pos = pIndices[ index ].value;
                     int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                     int blockOffset = pos - pBlockSizePrefixSum[ blockIndex ];
                     memcpy( pResult + index * len, pBuffers[ blockIndex ] +  blockOffset * len, len );
                 }
             };
             transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
         }
         else
         {
             ARIES_ASSERT( inputLen == len - 1, "inputLen should be len - 1" );
             auto k = [=] ARIES_DEVICE(int index)
             {
                 if ( 0 == pIndices[ index ].flag )
                 {
                     *( pResult + index * len ) = 0;
                 }
                 else
                 {
                     index_type_t pos = pIndices[ index ].value;
                     *( pResult + index * len ) = 1;
                     int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                     memcpy( pResult + index * len + 1, pBuffers[ blockIndex ] + ( pos - pBlockSizePrefixSum[ blockIndex ] ) * inputLen, inputLen );
                 }
             };
             transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
         }
         context.synchronize();
     }
 
     void AddOffsetToIndices( AriesInt32ArraySPtr &indices, int offset )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         int* data = indices->GetData();
         auto k = [=] ARIES_DEVICE(int index)
         {
             int oldVal = data[ index ];
             if( oldVal != -1 )
             data[ index ] = oldVal + offset;
         };
         transform( k, indices->GetItemCount(), *ctx );
         ctx->synchronize();
     }
 
     // 用于字典列的物化
     AriesDataBufferSPtr MaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks,
                                            const AriesInt64ArraySPtr& blockSizePrefixSum,
                                            const AriesVariantIndicesArraySPtr &indices,
                                            AriesColumnType resultType )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         size_t tupleNum = indices->GetItemCount();
         AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( resultType, tupleNum );
         if( tupleNum > 0 )
         {
             int8_t* pResult = result->GetData();
             int8_t* pIndices = indices->GetData();
             auto indiceDataType = indices->GetDataType().DataType.ValueType;
 
             int64_t* pBlockSizePrefixSum = blockSizePrefixSum->GetData();
             result->PrefetchToGpu();
 
             if ( !dataBlocks.empty() && std::dynamic_pointer_cast< AriesRowPosRangeBuffer >( dataBlocks[ 0 ] ) )
             {
                 DoMaterializeRowPosRangeColumn( dataBlocks, pBlockSizePrefixSum, ( int32_t* )pIndices, tupleNum, resultType, pResult, *ctx );
                 return result;
             }
 
             switch( indiceDataType )
             {
                 case AriesValueType::INT8:
                     if ( indices->isNullableColumn() )
                         DoMaterializeColumn< int8_t >( dataBlocks, pBlockSizePrefixSum,
                                                        ( nullable_type< int8_t >* )pIndices,
                                                        tupleNum, resultType,
                                                        pResult, *ctx );
                     else
                         DoMaterializeColumn< int8_t >( dataBlocks, pBlockSizePrefixSum,
                                                        ( int8_t* )pIndices,
                                                        tupleNum, resultType,
                                                        pResult, *ctx );
                     break;
                 case AriesValueType::INT16:
                     if ( indices->isNullableColumn() )
                         DoMaterializeColumn< int16_t >( dataBlocks, pBlockSizePrefixSum,
                                                         ( nullable_type< int16_t >* )pIndices,
                                                         tupleNum, resultType,
                                                         pResult, *ctx );
                     else
                         DoMaterializeColumn< int16_t >( dataBlocks, pBlockSizePrefixSum,
                                                         ( int16_t* )pIndices,
                                                         tupleNum, resultType,
                                                         pResult, *ctx );
                     break;
                 case AriesValueType::INT32:
                     if ( indices->isNullableColumn() )
                         DoMaterializeColumn< int32_t >( dataBlocks, pBlockSizePrefixSum,
                                                         ( nullable_type< int32_t >* )pIndices,
                                                         tupleNum, resultType,
                                                         pResult, *ctx );
                     else
                         DoMaterializeColumn< int32_t >( dataBlocks, pBlockSizePrefixSum,
                                                         ( int32_t* )pIndices,
                                                         tupleNum, resultType,
                                                         pResult, *ctx );
                     break;
 
                 default:
                     ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, "index type error: " + std::to_string( ( int ) indiceDataType ) );
             }
         }
         return result;
     }
 
     namespace
     {
 //        bool IsIndicesOrderedAsc( const AriesInt32ArraySPtr &indices )
 //        {
 //            bool bRet = true;
 //            indices->PrefetchToCpu();
 //            int32_t* pIndices = indices->GetData();
 //            size_t count = indices->GetItemCount();
 //            for( int i = 0; i < count - 1; ++i )
 //            {
 //                if( pIndices[i] > pIndices[i + 1] )
 //                {
 //                    bRet = false;
 //                    break;
 //                }
 //            }
 //            indices->PrefetchToGpu();
 //            return bRet;
 //        }
     }
 
     // std::vector< AriesDataBufferSPtr > MaterializeDataBlocks( const std::vector< AriesDataBufferSPtr > &dataBlocks,
     //         const AriesInt64ArraySPtr& blockSizePrefixSum, const AriesInt32ArraySPtr &indices, AriesColumnType resultType )
     // {
     //     //assert( IsIndicesOrderedAsc( indices ) );
     //     auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
     //     vector< AriesDataBufferSPtr > result;
     //     size_t prefixSumCount = blockSizePrefixSum->GetItemCount();
     //     int64_t* pBlockSizePrefixsum = blockSizePrefixSum->GetData();
     //     AriesManagedInt32Array splitArray( prefixSumCount );
     //     splitArray.PrefetchToGpu();
     //     int32_t* pSplitArray = splitArray.GetData();
     //     int32_t* pIndices = indices->GetData();
     //     sorted_search< less_t, bounds_lower, launch_box_t< arch_52_cta< 256, 13 > > >( pBlockSizePrefixsum, prefixSumCount, pIndices,
     //             indices->GetItemCount(), pSplitArray, *ctx );
     //     ctx->synchronize();
     //     splitArray.PrefetchToCpu();
     //     if ( std::dynamic_pointer_cast< AriesRowPosRangeBuffer >( dataBlocks[ 0 ] ) )
     //     {
     //         AriesManagedArray< int > rowposBuffers( dataBlocks.size() );
     //         for ( int i = 0; i < dataBlocks.size(); i++ )
     //         {
     //             const auto& block = dataBlocks[ i ];
     //             int32_t tupleNum = pSplitArray[i + 1] - pSplitArray[i];
     //             auto buffer = std::make_shared< AriesDataBuffer >( block->GetDataType(), tupleNum );
     //             auto pBuffer = ( int* )( buffer->GetData() );
     //             auto rowposBuffer = std::dynamic_pointer_cast< AriesRowPosRangeBuffer >( block );
     //             auto start = rowposBuffer->GetStart();
     //             if( tupleNum == dataBlocks[i]->GetItemCount() )
     //             {
     //                 if ( start > 0 )
     //                 {
     //                     auto k = [=] ARIES_DEVICE(int index)
     //                     {   
     //                         pBuffer[ index ] = start + index;
     //                     };
     //                     transform( k, tupleNum, *ctx );
     //                 }
     //                 else
     //                 {
     //                     auto k = [=] ARIES_DEVICE(int index)
     //                     {   
     //                         pBuffer[ index ] = start - index;
     //                     };
     //                     transform( k, tupleNum, *ctx );
     //                 }
     //             }
     //             else if ( tupleNum > 0 )
     //             {
     //                 assert( tupleNum != dataBlocks[i]->GetItemCount() );
     //                 int32_t startIndex = pSplitArray[i];
 
     //                 dataBlocks[i]->PrefetchToGpu();
     //                 int8_t* pInput = dataBlocks[i]->GetData();
     //                 AriesDataBufferSPtr output = std::make_shared< AriesDataBuffer >( resultType, tupleNum );
     //                 output->PrefetchToGpu();
     //                 int8_t* pOutput = output->GetData();
     //                 size_t len = resultType.GetDataTypeSize();
 
     //                 if ( start > 0 )
     //                 {
     //                     transform< 256, 3 >( [=] ARIES_DEVICE(int index)
     //                     {
     //                         pBuffer[ index ] = start + pIndices[ index + startIndex ] - pBlockSizePrefixsum[ i ];
     //                     }, tupleNum, *ctx );
     //                 }
     //                 else
     //                 {
     //                     transform< 256, 3 >( [=] ARIES_DEVICE(int index)
     //                     {
     //                         pBuffer[ index ] = start - ( pIndices[ index + startIndex ] - pBlockSizePrefixsum[ i ] );
     //                     }, tupleNum, *ctx );
     //                 }
     //             }
 
     //             if ( tupleNum > 0 )
     //                 result.emplace_back( buffer );
     //         }
     //         return result;
     //     }
 
     //     for( int i = 0; i < dataBlocks.size(); ++i )
     //     {
     //         int32_t tupleNum = pSplitArray[i + 1] - pSplitArray[i];
     //         if( tupleNum == dataBlocks[i]->GetItemCount() )
     //             result.push_back( dataBlocks[i] );
     //         else if( tupleNum > 0 )
     //         {
     //             assert( tupleNum != dataBlocks[i]->GetItemCount() );
     //             int32_t startIndex = pSplitArray[i];
 
     //             dataBlocks[i]->PrefetchToGpu();
     //             int8_t* pInput = dataBlocks[i]->GetData();
     //             AriesDataBufferSPtr output = std::make_shared< AriesDataBuffer >( resultType, tupleNum );
     //             output->PrefetchToGpu();
     //             int8_t* pOutput = output->GetData();
     //             size_t len = resultType.GetDataTypeSize();
 
     //             transform< 256, 3 >( [=] ARIES_DEVICE(int index)
     //                     {
     //                         memcpy( pOutput + index * len, pInput + ( pIndices[ index + startIndex ] - pBlockSizePrefixsum[ i ] ) * len, len );
     //                     }, tupleNum, *ctx );
     //             result.push_back( output );
     //         }
     //     }
     //     ctx->synchronize();
     //     return result;
     // }
 
     AriesDataBufferSPtr
     MaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks,
                        const AriesInt64ArraySPtr& blockSizePrefixSum,
                        const AriesInt32ArraySPtr &indices,
                        AriesColumnType resultType )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         size_t tupleNum = indices->GetItemCount();
         AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( resultType, tupleNum );
         if( tupleNum > 0 )
         {
             indices->PrefetchToGpu();
             int blockCount = dataBlocks.size();
             AriesManagedArray< int8_t* > dataBuffers( blockCount );
             int i = 0;
             for( const auto& block : dataBlocks )
                 dataBuffers[i++] = block->GetData();
             int8_t* pResult = result->GetData();
             int8_t** pBuffers = dataBuffers.GetData();
             int32_t* pIndices = indices->GetData();
             size_t len = resultType.GetDataTypeSize();
             int64_t* pBlockSizePrefixSum = blockSizePrefixSum->GetData();
 
             size_t inputLen = dataBlocks[0]->GetDataType().GetDataTypeSize();
 
             result->PrefetchToGpu();
             dataBuffers.PrefetchToGpu();
             if( inputLen == len )
             {
                 auto k = [=] ARIES_DEVICE(int index )
                 {
                     int pos = pIndices[ index ];
                     if( pos != -1 )
                     {
                         int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                         memcpy( pResult + index * len, pBuffers[ blockIndex ] + ( pos - pBlockSizePrefixSum[ blockIndex ] ) * len, len );
                     }
                     else
                     *( pResult + index * len ) = 0;
                 };
                 transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, *ctx );
             }
             else
             {
                 ARIES_ASSERT( inputLen == len - 1, "inputLen should be len - 1" );
                 auto k =
                 [=] ARIES_DEVICE(int index)
                 {
                     int pos = pIndices[ index ];
                     if( pos != -1 )
                     {
                         *( pResult + index * len ) = 1;
                         int blockIndex = aries_acc::binary_search<bounds_upper>( pBlockSizePrefixSum, blockCount, pos ) - 1;
                         memcpy( pResult + index * len + 1, pBuffers[ blockIndex ] + ( pos - pBlockSizePrefixSum[ blockIndex ] ) * inputLen, inputLen );
                     }
                     else
                     *( pResult + index * len ) = 0;
                 };
                 transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, *ctx );
             }
             ctx->synchronize();
         }
         return result;
     }
 
     AriesDataBufferSPtr
     MaterializeColumn( const std::vector< AriesDataBufferSPtr > &dataBlocks,
                        const AriesInt64ArraySPtr& dataBlockSizePrefixSum,
                        const vector< AriesInt32ArraySPtr >& indices,
                        const AriesInt64ArraySPtr& indiceBlockSizePrefixSum,
                        AriesColumnType resultType )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         int64_t* pIndicesBlockSizePrefixSum = indiceBlockSizePrefixSum->GetData();
         size_t tupleCount = indiceBlockSizePrefixSum->GetValue( indiceBlockSizePrefixSum->GetItemCount() - 1 );
         AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( resultType, tupleCount );
         if( tupleCount > 0 )
         {
             int dataBlockCount = dataBlocks.size();
             size_t len = resultType.GetDataTypeSize();
             int8_t* pResult = result->GetData();
             size_t inputLen = dataBlocks[0]->GetDataType().GetDataTypeSize();
             int indicesBlockCount = indices.size();
             AriesManagedArray< index_t* > indicesBuffers( indicesBlockCount );
             int i = 0;
             for( const auto& indice : indices )
             {
                 indicesBuffers[i++] = indice->GetData();
             }
             index_t** pIndicesBuffers = indicesBuffers.GetData();
             int64_t* pDataBlockSizePrefixSum = dataBlockSizePrefixSum->GetData();
 
             auto isRowPosBuffer = std::dynamic_pointer_cast< AriesRowPosRangeBuffer >( dataBlocks[ 0 ] ) != nullptr;
             if ( isRowPosBuffer )
             {
                 AriesManagedArray< int > rowposBuffers( dataBlocks.size() );
                 for ( int i = 0; i < dataBlocks.size(); i++ )
                 {
                     const auto& block = dataBlocks[ i ];
                     auto buffer = std::dynamic_pointer_cast< AriesRowPosRangeBuffer >( block );
                     rowposBuffers[ i ] = buffer->GetStart();
                 }
 
                 auto pRowPosBuffer = rowposBuffers.GetData();
 
                 if ( inputLen == len )
                 {
                     auto k = [ = ]ARIES_DEVICE( int index )
                     {
                         int indicesBlockIndex = aries_acc::binary_search<bounds_upper>(
                             pIndicesBlockSizePrefixSum,
                             indicesBlockCount, index ) - 1;
                         int offset = index - pIndicesBlockSizePrefixSum[ indicesBlockIndex ];
                         int pos = pIndicesBuffers[ indicesBlockIndex ][ offset ];
 
                         if ( pos != -1 )
                         {
                             int dataBlockIndex = aries_acc::binary_search<bounds_upper>( pDataBlockSizePrefixSum, dataBlockCount, pos ) - 1;
                             auto start = pRowPosBuffer[ dataBlockIndex ];
                             if ( start > 0 )
                             {
                                 *( int* )( pResult + index * len ) = start + pos - pDataBlockSizePrefixSum[ dataBlockIndex ] ;
                             }
                             else
                             {
                                 *( int* )( pResult + index * len ) = start - ( pos - pDataBlockSizePrefixSum[ dataBlockIndex ] );
                             }
                         }
                         else
                         {
                             *( pResult + index * len ) = 0;
                         }
                     };
                     transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleCount, *ctx );
                 }
                 else
                 {
                     auto k = [ = ]ARIES_DEVICE( int index )
                     {
                         int indicesBlockIndex = aries_acc::binary_search<bounds_upper>(
                             pIndicesBlockSizePrefixSum,
                             indicesBlockCount, index ) - 1;
                         int offset = index - pIndicesBlockSizePrefixSum[ indicesBlockIndex ];
                         int pos = pIndicesBuffers[ indicesBlockIndex ][ offset ];
        
                         if ( pos != -1 )
                         {
                             *( pResult + index * len ) = 1;
                             int dataBlockIndex = aries_acc::binary_search<bounds_upper>( pDataBlockSizePrefixSum, dataBlockCount, pos ) - 1;
                             auto start = pRowPosBuffer[ dataBlockIndex ];
                             if ( start > 0 )
                             {
                                 *( int* )( pResult + index * len + 1 ) = start + pos - pDataBlockSizePrefixSum[ dataBlockIndex ] ;
                             }
                             else
                             {
                                 *( int* )( pResult + index * len + 1 ) = start - ( pos - pDataBlockSizePrefixSum[ dataBlockIndex ] );
                             }
                         }
                         else
                         {
                             *( pResult + index * len ) = 0;
                         }
                     };
                     transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleCount, *ctx );
                 }
 
                 return result;
             }
             AriesManagedArray< int8_t* > dataBuffers( dataBlockCount );
             i = 0;
             for( const auto& block : dataBlocks )
                 dataBuffers[i++] = block->GetData();
             int8_t** pDataBuffers = dataBuffers.GetData();
 
             result->PrefetchToGpu();
             dataBuffers.PrefetchToGpu();
             indicesBuffers.PrefetchToGpu();
             if( inputLen == len )
             {
                 auto k = [=] ARIES_DEVICE(int i)
                 {
                     int indicesBlockIndex = aries_acc::binary_search<bounds_upper>(
                             pIndicesBlockSizePrefixSum,
                             indicesBlockCount, i ) - 1;
                     int offset = i - pIndicesBlockSizePrefixSum[ indicesBlockIndex ];
                     int pos = pIndicesBuffers[ indicesBlockIndex ][ offset ];
                     if( pos != -1 )
                     {
                         int dataBlockIndex = aries_acc::binary_search<bounds_upper>( pDataBlockSizePrefixSum, dataBlockCount, pos ) - 1;
                         memcpy( pResult + i * len, pDataBuffers[ dataBlockIndex ] + ( pos - pDataBlockSizePrefixSum[ dataBlockIndex ] ) * len, len );
                     }
                     else
                     *( pResult + i * len ) = 0;
                 };
                 transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleCount, *ctx );
             }
             else
             {
                 ARIES_ASSERT( inputLen == len - 1, "inputLen should be len - 1" );
                 auto k =
                 [=] ARIES_DEVICE(int i)
                 {
                     int indicesBlockIndex = aries_acc::binary_search<bounds_upper>(
                             pIndicesBlockSizePrefixSum,
                             indicesBlockCount, i ) - 1;
                     int offset = i - pIndicesBlockSizePrefixSum[ indicesBlockIndex ];
                     int pos = pIndicesBuffers[ indicesBlockIndex ][ offset ];
                     if( pos != -1 )
                     {
                         *( pResult + i * len ) = 1;
                         int dataBlockIndex = aries_acc::binary_search<bounds_upper>( pDataBlockSizePrefixSum, dataBlockCount, pos ) - 1;
                         memcpy( pResult + i * len + 1, pDataBuffers[ dataBlockIndex ] + ( pos - pDataBlockSizePrefixSum[ dataBlockIndex ] ) * inputLen, inputLen );
                     }
                     else
                     *( pResult + i * len ) = 0;
                 };
                 transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleCount, *ctx );
             }
         }
         return result;
     }
 
     vector< AriesIndicesArraySPtr >
     ShuffleIndices( const vector< AriesIndicesArraySPtr >& oldIndices,
                     const AriesIndicesArraySPtr& indices )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         vector< AriesIndicesArraySPtr > result;
         int count = oldIndices.size();
         AriesManagedArray< int* > input_data( count );
         AriesManagedArray< int* > output_data( count );
         int index = 0;
         for( const auto& ind : oldIndices )
         {
             ind->PrefetchToGpu();
             AriesIndicesArraySPtr newInd = indices->CloneWithNoContent();
             input_data[index] = ind->GetData();
             output_data[index] = newInd->GetData();
             result.push_back( newInd );
             ++index;
         }
         input_data.PrefetchToGpu();
         output_data.PrefetchToGpu();
         shuffle_indices( ( const int** )input_data.GetData(), count, indices->GetItemCount(), indices->GetData(), output_data.GetData(), *ctx );
         return result;
     }
 
     vector< AriesManagedIndicesArraySPtr >
     ShuffleIndices( const vector< AriesManagedIndicesArraySPtr >& oldIndices,
                     const AriesManagedIndicesArraySPtr& indices )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         vector< AriesManagedIndicesArraySPtr > result;
         int count = oldIndices.size();
         AriesManagedArray< int* > input_data( count );
         AriesManagedArray< int* > output_data( count );
         int index = 0;
         indices->PrefetchToGpu();
         for( const auto& ind : oldIndices )
         {
             ind->PrefetchToGpu();
             AriesManagedIndicesArraySPtr newInd = indices->CloneWithNoContent();
             input_data[index] = ind->GetData();
             output_data[index] = newInd->GetData();
             result.push_back( newInd );
             ++index;
         }
         input_data.PrefetchToGpu();
         output_data.PrefetchToGpu();
         shuffle_indices( ( const int** )input_data.GetData(), count, indices->GetItemCount(), indices->GetData(), output_data.GetData(), *ctx );
         return result;
     }
 
     std::vector< AriesDataBufferSPtr >
     ShuffleColumns( const std::vector< AriesDataBufferSPtr >& arg_columns,
                     const AriesInt32ArraySPtr& associated )
     {
         auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
         ctx->timer_begin();
         std::vector< AriesDataBufferSPtr > result;
         std::vector< AriesDataBufferSPtr > columns;
         std::vector< AriesDataBufferSPtr > cloned_columns;
 
         for( const auto& column : arg_columns )
         {
             if( column )
             {
                 AriesDataBufferSPtr newColumn = column->CloneWithNoContent( associated->GetItemCount() );
                 result.emplace_back( newColumn );
                 cloned_columns.emplace_back( newColumn );
                 columns.emplace_back( column );
             }
             else
             {
                 result.emplace_back( nullptr );
             }
         }
 
         int count = columns.size();
         if( count > 0 )
         {
             AriesManagedArray< DataBlockInfo > blocks( count );
             AriesManagedArray< int8_t* > outputs( count );
             DataBlockInfo *block = blocks.GetData();
             int8_t** output = outputs.GetData();
             for( int i = 0; i < columns.size(); i++ )
             {
                 AriesDataBufferSPtr newColumn = cloned_columns[i];
                 auto col = columns[i];
                 block->Data = col->GetData();
                 block->ElementSize = col->GetDataType().GetDataTypeSize();
                 *output = newColumn->GetData();
                 ++output;
                 ++block;
             }
             blocks.PrefetchToGpu();
             outputs.PrefetchToGpu();
             shuffle_by_index( blocks.GetData(),
                               count,
                               associated->GetItemCount(),
                               associated->GetData(),
                               outputs.GetData(),
                               *ctx );
         }
         LOG(INFO)<< "ShuffleColumns gpu time: " << ctx->timer_end();
         return result;
     }
 END_ARIES_ACC_NAMESPACE
 