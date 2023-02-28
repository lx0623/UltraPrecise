/*
* test_cu
*
*  Created on: Jul 2, 2019
*      Author: lichi
*/
#include "test_common.h"
#include "AriesEngine/cpu_algorithm.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "CudaAcc/AriesEngineAlgorithm.h"
using namespace aries_acc;
// static const char* DB_NAME = "scale_1";
static const char* DB_NAME = "tpch218_100";

// using PartitionedIndices = vector< shared_ptr< vector< index_t > > >;

// // unsigned int BKDRHash( const int8_t *p, int size, uint32_t seed, bool aligned )
// // {
// //     unsigned int hash = seed;

// //     while( size-- )
// //         hash = ( hash * 131 ) + *p++;

// //     return hash;
// // }

// uint32_t murmur_hash2(const void* key,
//     int len,
//     const uint32_t seed,
//     bool aligned)
// {
//     const unsigned int m = 0xc6a4a793;
//     const int r = 16;
//     uint32_t h = seed ^ (len * m);

//     const unsigned char* data = (const unsigned char*)key;
//     while (len >= 4) {
//         unsigned int k = 0;
//         if (aligned)
//             k = *(unsigned int*)data;
//         else
//             k = (data[3] << 24) + (data[2] << 16) + (data[1] << 8) + data[0];

//         h += k;
//         h *= m;
//         h ^= h >> 16;

//         data += 4;
//         len -= 4;
//     }

//     switch (len) {
//     case 3:
//         h += data[2] << 16;
//     case 2:
//         h += data[1] << 8;
//     case 1:
//         h += data[0];
//         h *= m;
//         h ^= h >> r;
//     };

//     h *= m;
//     h ^= h >> 10;
//     h *= m;
//     h ^= h >> 17;
//     return h;
// }


// // PartitionedIndices PartitionColumnData( const vector< AriesDataBufferSPtr >& buffers, size_t bucketCount, uint32_t seed, shared_ptr< vector< index_t > > indices = nullptr )
// // {
// //     assert( !buffers.empty() );

// //     PartitionedIndices result;
// //     for( int i = 0; i < bucketCount; ++i )
// //         result.push_back( std::make_shared< vector< index_t > >() );

// //     size_t totalItemCount = indices ? indices->size() : buffers[0]->GetItemCount();
// //     assert( totalItemCount > 0 );

// //     size_t threadNum = GetThreadNum( totalItemCount );
// //     size_t perThreadItemCount = totalItemCount / threadNum;
// //     size_t offset = 0;

// //     struct DataWrapper
// //     {
// //         int8_t* data;
// //         size_t perItemSize;
// //         bool hasNull;
// //     };

// //     auto BKDRHash = []( const int8_t *p, int size, uint32_t seed )
// //     {
// //         uint32_t hash = seed;
    
// //         while( size-- )
// //             hash = ( hash * 131 ) + *p++;
    
// //         return hash;
// //     }

// //     vector< DataWrapper > dataWrapper;
// //     for( const auto& buf : buffers )
// //         dataWrapper.emplace_back( DataWrapper{ buf->GetData( 0 ), buf->GetItemSizeInBytes(), buf->GetDataType().HasNull } );

// //     vector< future< PartitionedIndices > > allThreads;
// //     for( size_t i = 0; i < threadNum; ++i )
// //     {
// //         size_t itemCount = perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
// //         allThreads.push_back( std::async( std::launch::async, [=]( size_t pos, size_t count )
// //         {   
// //             PartitionedIndices output;
// //             for( int n = 0; n < bucketCount; ++n )
// //                 output.push_back( std::make_shared< vector< index_t > >() );

// //             for( size_t index = 0; index < count; ++index )
// //             {
// //                 uint32_t hashVal = 0;
// //                 size_t itemPos = indices ? ( *indices )[ pos + index ] : pos + index;
// //                 for( const auto& wrapper : dataWrapper )
// //                 {
// //                     //hashVal ^= murmur_hash2( wrapper.data + itemPos * wrapper.perItemSize + wrapper.hasNull, wrapper.perItemSize - wrapper.hasNull, seed, true );
// //                     hashVal ^= BKDRHash( wrapper.data + ( pos + index ) * wrapper.perItemSize + wrapper.hasNull, wrapper.perItemSize - wrapper.hasNull, seed );
// //                 }
// //                 output[ hashVal % bucketCount ]->push_back( itemPos );
// //             }
// //             return output;
// //         }, offset, itemCount ) );
// //         offset += perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
// //     }

// //     vector< PartitionedIndices > allResult;
// //     for( auto& t : allThreads )
// //         allResult.push_back( t.get() );

// //     vector< size_t > totalSizes;
// //     totalSizes.resize( bucketCount, 0 );
// //     for( const auto& r : allResult )
// //     {
// //         for( size_t i = 0; i < r.size(); ++i )
// //             totalSizes[ i ] += r[ i ]->size();
// //     }

// //     for( size_t i = 0; i < bucketCount; ++i )
// //     {
// //         result[ i ]->reserve( totalSizes[ i ] );
// //         for( const auto& r : allResult )
// //             result[ i ]->insert( result[ i ]->end(), r[ i ]->begin(), r[ i ]->end() );
// //     }

// //     return result;    
// // }
// // #include <set>

// // class AriesHostDataBuffer;
// // using AriesHostDataBufferSPtr = std::shared_ptr< AriesHostDataBuffer >;
// // class AriesHostDataBuffer
// // {
// // public:
// //     AriesHostDataBuffer( AriesColumnType columnType, size_t reservedCount = 0 ) 
// //         : m_columnType( columnType ), m_itemSizeInBytes( columnType.GetDataTypeSize() ), m_reservedCount( 0 ), m_itemCount( 0 ), m_pDst( nullptr )
// //     {
// //         if( reservedCount > 0 )
// //             AddBlock( reservedCount );
// //     }

// //     ~AriesHostDataBuffer(){}

// //     AriesDataBufferSPtr ToAriesDataBuffer() const
// //     {
// //         AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( m_columnType, m_itemCount );
// //         if( m_itemCount > 0 )
// //         {
// //             result->PrefetchToGpu();
// //             int8_t *pOutput = result->GetData();
// //             int index = 0;
// //             for( auto& block : m_dataBlocks )
// //             {
// //                 cudaMemcpy( pOutput, block.get(), m_blockItemCount[ index ] * m_itemSizeInBytes, cudaMemcpyHostToDevice );
// //                 pOutput += m_blockItemCount[ index ] * m_itemSizeInBytes;
// //                 ++index;
// //             }
// //         }
// //         return result;
// //     }

// //     size_t GetItemCount() const
// //     {
// //         return m_itemCount;
// //     }

// //     void AddItem( const int8_t* data )
// //     {
// //         assert( data && m_pDst );
// //         assert( m_reservedCount > 0 );
// //         if( ++m_itemCount > m_reservedCount )
// //         {
// //             size_t expandCount = m_reservedCount / 10;
// //             if( expandCount < 100 )
// //                 expandCount = 100;
// //             AddBlock( expandCount );
// //         }
// //         memcpy( m_pDst, data, m_itemSizeInBytes );
// //         m_pDst += m_itemSizeInBytes;
// //         ++( m_blockItemCount.back() );
// //     }

// //     void Merge( AriesHostDataBufferSPtr buffer )
// //     {
// //         int index = 0;
// //         for( auto& block : buffer->m_dataBlocks )
// //         {
// //             m_dataBlocks.push_back( std::move( block ) );
// //             m_blockItemCount.push_back( buffer->m_blockItemCount[ index++ ] );
// //             m_itemCount += m_blockItemCount.back();
// //         }
// //         m_pDst = nullptr;
// //         m_reservedCount = 0;
// //     }

// // private:
// //     void AddBlock( size_t blockItemCount )
// //     {
// //         m_reservedCount += blockItemCount;
// //         m_blockItemCount.push_back( 0 );
// //         int8_t* ptr = nullptr;
// //         ARIES_CALL_CUDA_API( cudaMallocHost ( &ptr, blockItemCount * m_itemSizeInBytes ) );
// //         m_dataBlocks.push_back( unique_ptr< int8_t, _MemFreeFunc >{ ptr } );
// //         m_pDst = m_dataBlocks.back().get();
// //     }

// //     struct _MemFreeFunc
// //     {
// //         void operator()( void* p )
// //         {
// //             if( p )
// //             ARIES_CALL_CUDA_API( cudaFreeHost( p ) );
// //         }
// //     };
    
// //     AriesColumnType m_columnType;
// //     size_t m_itemSizeInBytes;
// //     size_t m_reservedCount;
// //     size_t m_itemCount;
// //     int8_t* m_pDst;
// //     vector< unique_ptr< int8_t, _MemFreeFunc > > m_dataBlocks;
// //     vector< size_t > m_blockItemCount;
// // };

// // using AriesHostDataBufferSPtr = std::shared_ptr< AriesHostDataBuffer >;

// // struct PartitionResult
// // {
// //     vector< vector< AriesHostDataBufferSPtr > > AllBuffers;
// //     vector< PartitionedIndices > AllIndices;
// // };

// // PartitionResult PartitionColumnData2(
// //     const vector< AriesDataBufferSPtr >& buffers,
// //     size_t bucketCount,
// //     uint32_t seed,
// //     bool *hasNullValues )
// // {
// //     assert( !buffers.empty() );
// //     size_t totalItemCount = buffers[0]->GetItemCount();
// //     assert( totalItemCount > 0 );
// //     size_t bufferCount = buffers.size();

// //     PartitionResult result;
// //     result.AllBuffers.resize( bufferCount );
// //     for( size_t i = 0; i < bucketCount; ++i )
// //     {
// //         result.AllIndices.push_back( PartitionedIndices{ std::make_shared< vector< index_t > >() } );
// //         for( int j = 0; j < bufferCount; ++j )
// //         {
// //             result.AllBuffers[ j ].push_back( std::make_shared< AriesHostDataBuffer >( buffers[ j ]->GetDataType() ) );
// //         }
// //     }

// //     size_t threadNum = GetThreadNum( totalItemCount );
// //     size_t perThreadItemCount = totalItemCount / threadNum;
// //     size_t offset = 0;

// //     struct DataWrapper
// //     {
// //         int8_t* data;
// //         size_t perItemSize;
// //         bool hasNull;
// //     };
    
// //     static const size_t BUCKET_NUM = 16381;
// //     size_t perBucketSize = BUCKET_NUM / bucketCount + BUCKET_NUM % bucketCount;

// //     auto BKDRHash = []( const int8_t *p, int size, uint32_t seed )
// //     {
// //         uint32_t hash = seed;
    
// //         while( size-- )
// //             hash = ( hash * 131 ) + *p++;
    
// //         return hash;
// //     };

// //     vector< DataWrapper > dataWrapper;
// //     for( const auto& buf : buffers )
// //         dataWrapper.emplace_back( DataWrapper{ buf->GetData( 0 ), buf->GetItemSizeInBytes(), buf->GetDataType().HasNull } );

// //     atomic_bool bHasNull = false;
// //     vector< future< PartitionResult > > allThreads;
    
// //     for( size_t i = 0; i < threadNum; ++i )
// //     {
// //         size_t itemCount = perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
// //         allThreads.push_back( std::async( std::launch::async, [&]( size_t pos, size_t count )
// //         {   
// //             PartitionResult output;
// //             output.AllBuffers.resize( bufferCount );
// //             for( size_t n = 0; n < bucketCount; ++n )
// //             {
// //                 output.AllIndices.push_back( PartitionedIndices{ std::make_shared< vector< index_t > >() } );
// //                 for( int j = 0; j < bufferCount; ++j )
// //                 {
// //                     output.AllBuffers[ j ].push_back( std::make_shared< AriesHostDataBuffer >( buffers[ j ]->GetDataType(), count * 1.1 ) );
// //                 }
// //             }
            
// //             for( size_t index = 0; index < count; ++index )
// //             {
// //                 uint32_t hashVal = 0;
// //                 size_t itemPos = pos + index;
// //                 for( const auto& wrapper : dataWrapper )
// //                 {
// //                     auto itemStart = wrapper.data + ( pos + index ) * wrapper.perItemSize;
// //                     hashVal ^= BKDRHash( itemStart + wrapper.hasNull, wrapper.perItemSize - wrapper.hasNull, seed );
// //                     if ( hasNullValues && wrapper.hasNull && !( *itemStart ) )
// //                     {
// //                         bHasNull = true;
// //                     }
// //                 }
// //                 output.AllIndices[ ( hashVal % BUCKET_NUM ) / perBucketSize ].Indices->push_back( itemPos );
// //                 for( int j = 0; j < bufferCount; ++j )
// //                 {
// //                     output.AllBuffers[ j ][ ( hashVal % BUCKET_NUM ) / perBucketSize ]->AddItem( buffers[ j ]->GetData( itemPos ) );
// //                 }
// //             }
// //             return output;
// //         }, offset, itemCount ) );
// //         offset += perThreadItemCount + ( i == 0 ? totalItemCount % threadNum : 0 );
// //     }

// //     vector< PartitionResult > allResult;
// //     for( auto& t : allThreads )
// //         allResult.push_back( t.get() );

// //     vector< size_t > totalSizes;
// //     totalSizes.resize( bucketCount, 0 );
// //     for( const auto& r : allResult )
// //     {
// //         for( size_t i = 0; i < r.AllIndices.size(); ++i )
// //             totalSizes[ i ] += r.AllIndices[ i ].Indices->size();
// //     }

// //     for( size_t i = 0; i < bucketCount; ++i )
// //     {
// //         result.AllIndices[ i ].Indices->reserve( totalSizes[ i ] );
// //         for( const auto& r : allResult )
// //         {
// //             result.AllIndices[ i ].Indices->insert( result.AllIndices[ i ].Indices->end(), r.AllIndices[ i ].Indices->begin(), r.AllIndices[ i ].Indices->end() );
// //             for( int j = 0; j < bufferCount; ++j )
// //             {
// //                 result.AllBuffers[ j ][ i ]->Merge( r.AllBuffers[ j ][ i ] );
// //             }
// //         }
// //     }

// //     if ( hasNullValues )
// //         *hasNullValues = bHasNull; 

// //     return result;    
// // }

void testreduce( AriesDataBufferSPtr l_quantity )
{
    standard_context_t context;
    char temp[128];
    const CompactDecimal *data = (const CompactDecimal *)l_quantity->GetData();
    size_t len = l_quantity->GetItemSizeInBytes();
    auto prec = l_quantity->GetDataType().DataType.Precision;
    auto sca = l_quantity->GetDataType().DataType.Scale;
    managed_mem_t < aries_acc::nullable_type< aries_acc::Decimal > >reduction( 1, context );
    managed_mem_t< int32_t > seg( 1, context );
    int32_t* pSeg = seg.data();
    *pSeg = 0;

    CPU_Timer t;
    t.begin();

    transform_segreduce( [=]ARIES_LAMBDA(int index)
                        {
                            return aries_acc::Decimal( 1 );
                        }, l_quantity->GetItemCount(), pSeg, 1, reduction.data(), agg_sum_t< aries_acc::nullable_type< aries_acc::Decimal > >(),
                                aries_acc::nullable_type< aries_acc::Decimal >(), context );
    context.synchronize();
    cout<<"sum time cost:"<<t.end()<<endl;
    
    cout<<reduction.data()->value.GetDecimal(temp)<<endl;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void changeToDecimal( const int8_t* input, size_t tupleNum, size_t item_size, aries_acc::Decimal *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        aries_acc::Decimal columnId_1_( (CompactDecimal*)(input+i*item_size), 12, 2);
        output[i]=columnId_1_;
    }
}

__global__ void addDecimal(const aries_acc::Decimal *add1, size_t add1_len, const aries_acc::Decimal *add2, aries_acc::Decimal *output )
{
    extern __shared__ aries_acc::Decimal sdata[];
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < add1_len){
        memcpy(sdata , add1+tid , sizeof(aries_acc::Decimal));
        memcpy(sdata+1 , add2+tid , sizeof(aries_acc::Decimal));
        __syncthreads();
        
        sdata[0] = sdata[0] + sdata[0] + sdata[0] + sdata[1] + sdata[1] + sdata[1] + sdata[0]+ sdata[0] + sdata[0] + sdata[1] + sdata[1] + sdata[1];
        memcpy(output+tid, sdata, sizeof(aries_acc::Decimal));
    }

}

__global__ void addCompactDecimal(const int8_t*  add1, const int8_t*  add2 , size_t len , size_t item_size , aries_acc::Decimal *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < len; i += stride )
    {
        aries_acc::Decimal columnId1( (CompactDecimal*)(add1+i*item_size), 12, 2 );
        aries_acc::Decimal columnId2( (CompactDecimal*)(add2+i*item_size), 12, 2 );
        output[i] = columnId1 + columnId2;
    }

}

TEST(cpu, partitionEx)
{
    standard_context_t context;
    AriesDataBufferSPtr l_quantity = ReadColumn( DB_NAME, "lineitem", 5 );
    AriesDataBufferSPtr l_discount = ReadColumn( DB_NAME, "lineitem", 7 );
    cout<<l_quantity->GetItemCount()<<endl;

    int total = GetDecimalRealBytes( 12, 2);
    size_t sumByte= total * l_quantity->GetItemCount();
    cout<<"sumByte:"<<sumByte<<endl;

    // int8_t *q_compact_cpu_1 =  (int8_t *)malloc(sizeof(int8_t) *sumByte);
    int8_t *q_compact_cpu_2 =  (int8_t *)malloc(sizeof(int8_t) *sumByte);

    //读
    FILE *fw = fopen("/home/server/work/lc/aries/test/testcases/CudaAcc/1.bin", "rb+");
    fseek(fw, 0, SEEK_SET);
    size_t offset = 0;

    fseek(fw, offset, SEEK_SET);
    size_t buffSize = fread(q_compact_cpu_2 , 1 , sumByte - offset, fw);
    offset += buffSize;
    fflush(fw);
    
    printf("out for :: offset = %lu\n",offset);
    fclose(fw);

    // size_t i=0;
    // for( i=0 ; i< l_quantity->GetItemCount() ; i++){
    //     aries_acc::Decimal decimal( (CompactDecimal*)(l_quantity->GetData()+i*l_quantity->GetItemSizeInBytes()), 12, 2 ,0,1);
    //     char temp[128];
    //     decimal.ToCompactDecimal(temp, total);
    //     memcpy( q_compact_cpu_2 + i * total, temp , sizeof(int8_t) * total);
    // }

    // printf("mmcmp = %d\n",memcmp(q_compact_cpu_1,q_compact_cpu_2,sumByte));

    // //写
    // FILE *fp = fopen("/home/server/work/lc/aries/test/testcases/CudaAcc/1.bin", "ab+");
    // fseek(fp, 0, SEEK_SET);
    // fwrite(q_compact_cpu ,1 , sumByte, fp);
    // fclose(fp);


    aries_acc::Decimal *q_compact_gpu;
    gpuErrchk( cudaMalloc((void **)&q_compact_gpu, sizeof(int8_t) *sumByte ) );
    gpuErrchk( cudaMemcpy(q_compact_gpu, q_compact_cpu_2, sizeof(int8_t) * sumByte, cudaMemcpyHostToDevice) );
    cudaDeviceSynchronize();

    aries_acc::Decimal *q_decimal_ans;
    gpuErrchk( cudaMalloc((void **)&q_decimal_ans,  sizeof(aries_acc::Decimal)*l_quantity->GetItemCount()) );

    int threadN = 256;
    size_t blockN = (l_quantity->GetItemCount() -1) / threadN + 1;

    CPU_Timer t;
    t.begin();
    changeToDecimal<<<blockN, threadN>>>( (int8_t *)q_compact_gpu, l_quantity->GetItemCount() , total , (aries_acc::Decimal *)q_decimal_ans);
    cudaDeviceSynchronize();
    cout<<"transfer:"<<t.end()<<endl;

    aries_acc::Decimal *q_decimal_cpu = (aries_acc::Decimal *)malloc(sizeof(aries_acc::Decimal)*l_quantity->GetItemCount());
    gpuErrchk( cudaMemcpy(q_decimal_cpu, q_decimal_ans, sizeof(aries_acc::Decimal)*l_quantity->GetItemCount(), cudaMemcpyDeviceToHost) );
        
    aries_acc::Decimal ret = aries_acc::Decimal(12,2,"0");
    for(int i=0; i<l_quantity->GetItemCount();i++){
        ret += q_decimal_cpu[i];
    }
    char sum_decimal_ans[128];
    cout<<ret.GetDecimal(sum_decimal_ans)<<endl;

    // free(q_compact_cpu_1);
    free(q_compact_cpu_2);
    free(q_decimal_cpu);
    gpuErrchk( cudaFree(q_compact_gpu) );
    gpuErrchk( cudaFree(q_decimal_ans) );

    // aries_acc::Decimal ret = aries_acc::Decimal(12,2,"0");

    // for( i=0 ; i< l_quantity->GetItemCount() ; i++){
    //     char temp[128];
    //     // memcpy(temp ,  q_compact_cpu + i * total , sizeof(int8_t) * total);
    //     aries_acc::Decimal decimal1( (CompactDecimal*)( q_compact_cpu + i * total), 12, 2 ,0 , 1);
    //     ret += decimal1;
    // }

    // char ans[128];
    // printf("i = %u , sum = %s\n",i,ret.GetDecimal(ans));


    // aries_acc::Decimal ret = aries_acc::Decimal(12,2,"0");

    // int i=0;
    // for( i=0 ; i< l_quantity->GetItemCount() ; i++){
    //     aries_acc::Decimal decimal( (CompactDecimal*)(l_quantity->GetData()+i*l_quantity->GetItemSizeInBytes()), 12, 2 );
    //     char temp[128];
    //     // printf("%d : %s \n",i,decimal.GetDecimal(temp));
    //     int total = GetDecimalRealBytes( 12, 2);
    //     decimal.ToCompactDecimal(temp, total);
    //     aries_acc::Decimal decimal1( (CompactDecimal*)(temp), 12, 2 ,0,1);
    //     // printf("%d : %s \n",i,decimal1.GetDecimal(temp));

    //     if(decimal != decimal1){
    //         printf("%d : %s \n",i,decimal.GetDecimal(temp));
    //         printf("%d : %s \n",i,decimal1.GetDecimal(temp));
    //         break;
    //     }
    //     else{
    //         ret += decimal1;
    //     }
    // }
    // char ans[128];
    // printf("i = %d , sum = %s\n",i,ret.GetDecimal(ans));


    // char *q_cpu =  (char *)malloc(sizeof(char) *sumByte);
    // FILE *fw = fopen("1.bin", "ab+");

    // fseek(fw, 0, SEEK_SET);

    // for(size_t i = 0 ; i < l_quantity->GetItemCount() ; i++){
    //     size_t offset = i * total; //这里指定从文件头的第几个字节开始读取
    //     fseek(fw, offset, SEEK_SET);
    //     size_t buffSize = fread(q_cpu + i * total ,1 , total, fw);
    //     fflush(fw);
    // }
    // fclose(fw);

    // aries_acc::Decimal *q_gpu;
    // gpuErrchk( cudaMalloc((void **)&q_gpu, sizeof(char) *sumByte ) );
    // gpuErrchk( cudaMemcpy(q_gpu, q_cpu, sizeof(char) * sumByte, cudaMemcpyHostToDevice) );
    // cudaDeviceSynchronize();

    // aries_acc::Decimal *q_ans;
    // gpuErrchk( cudaMalloc((void **)&q_ans,  sizeof(aries_acc::Decimal)*l_quantity->GetItemCount()) );

    // int threadN = 256;
    // size_t blockN = (l_quantity->GetItemCount() -1) / threadN + 1;


    //  for(int i=0;i<10;i++){
    //     char buff[128];
    //     memcpy(buff ,q_cpu + i * total , total);
    //     char compactBuf[128];
    //     char ans[128];
    //     memcpy(compactBuf, buff, total);
    //     aries_acc::Decimal decimal1 =aries_acc:: Decimal((CompactDecimal *)compactBuf,12,2);
    //     printf("decimal1: sign = %d prec = %d frac = %d, v = %09d %09d %09d %09d %09d\n",GET_SIGN(decimal1.prec),GET_CALC_PREC(decimal1.prec),decimal1.frac,decimal1.v[4],decimal1.v[3],decimal1.v[2],decimal1.v[1],decimal1.v[0]);
    //     printf("%s\n",decimal1.GetDecimal(ans));
    // }
    
    // CPU_Timer t;
    // t.begin();
    // changeToDecimal<<<blockN, threadN>>>( (char *)q_gpu, l_quantity->GetItemCount() , total , q_ans);
    // cudaDeviceSynchronize();
    // cout<<"transfer:"<<t.end()<<endl;

    // aries_acc::Decimal *ans_cpu = (aries_acc::Decimal *)malloc(sizeof(aries_acc::Decimal)*l_quantity->GetItemCount());
    // gpuErrchk( cudaMemcpy(ans_cpu, q_ans, sizeof(aries_acc::Decimal)*l_quantity->GetItemCount(), cudaMemcpyDeviceToHost) );
        
    // aries_acc::Decimal ret = aries_acc::Decimal(12,2,"0");
    // for(int i=0; i<l_quantity->GetItemCount();i++){
    //     ret += ans_cpu[i];
    // }
    // char temp[128];
    // cout<<ret.GetDecimal(temp)<<endl;

    // testreduce(l_quantity);

    // char temp[128];
    // auto groups = make_shared< AriesInt32Array >();
    // groups->AllocArray( 1, true );

    // auto associated = make_shared< AriesInt32Array >( l_quantity->GetItemCount() );
    // aries_acc::InitSequenceValue( associated );
    // CPU_Timer t;
    // t.begin();
    // auto result = aggregate_column( l_quantity->GetData(), l_quantity->GetDataType(), l_quantity->GetItemCount(),
    //  AriesAggFunctionType::SUM, associated->GetData(), groups->GetData(), groups->GetItemCount(), context,
    //                         false, false );
    // cout<<"sum time cost:"<<t.end()<<endl;
    // aries_acc::nullable_type< aries_acc::Decimal > *pResult = (aries_acc::nullable_type< aries_acc::Decimal > *)result->GetData();
    // cout<<pResult->value.GetDecimal(temp)<<endl;


    // aries_acc::Decimal *q_gpu;
    // aries_acc::Decimal *d_gpu;
    // gpuErrchk( cudaMalloc((void **)&q_gpu, sizeof(aries_acc::Decimal)*l_quantity->GetItemCount() ) );
    // gpuErrchk( cudaMalloc((void **)&d_gpu, sizeof(aries_acc::Decimal)*l_quantity->GetItemCount() ) );

    // l_quantity->PrefetchToGpu();
    // l_discount->PrefetchToGpu();

    // cudaDeviceSynchronize();

    // int threadN = 256;
    // size_t blockN = (l_quantity->GetItemCount() -1) / threadN + 1;
    // changeToDecimal<<<blockN, threadN>>>(l_quantity->GetData(), l_quantity->GetItemCount(), l_quantity->GetItemSizeInBytes(), (aries_acc::Decimal *)q_gpu);
    // cudaDeviceSynchronize();
    // changeToDecimal<<<blockN, threadN>>>(l_discount->GetData(), l_discount->GetItemCount(), l_discount->GetItemSizeInBytes(), (aries_acc::Decimal *)d_gpu);
    // cudaDeviceSynchronize();

    // aries_acc::Decimal *ans_gpu;
    // gpuErrchk( cudaMalloc((void **)&ans_gpu, sizeof(aries_acc::Decimal)*l_quantity->GetItemCount() ) );

    // CPU_Timer t;
    // t.begin();
    // addDecimal<<<blockN, threadN, sizeof(aries_acc::Decimal)*2 >>>(q_gpu,l_quantity->GetItemCount(),d_gpu,ans_gpu); 
    // cudaDeviceSynchronize();
    // cout<<"Decimal add time cost:"<<t.end()<<endl;


    
    // CPU_Timer t;
    // t.begin();
    // addCompactDecimal<<<blockN, threadN>>>( l_quantity->GetData() , l_discount->GetData() , l_quantity->GetItemCount() , l_quantity->GetItemSizeInBytes() , ans_gpu); 
    // cudaDeviceSynchronize();
    // cout<<"Decimal add time cost:"<<t.end()<<endl;

    // aries_acc::Decimal *ans_cpu = (aries_acc::Decimal *)malloc(sizeof(aries_acc::Decimal)*l_quantity->GetItemCount());
    // gpuErrchk( cudaMemcpy(ans_cpu, ans_gpu, sizeof(aries_acc::Decimal)*l_quantity->GetItemCount(), cudaMemcpyDeviceToHost) );
        
    // aries_acc::Decimal ret = aries_acc::Decimal(12,2,"0");
    // for(int i=0; i<l_quantity->GetItemCount();i++){
    //     ret += ans_cpu[i];
    // }
    // char temp[128];
    // cout<<ret.GetDecimal(temp)<<endl;


    // CPU_Timer t;
    // t.begin();
    // for(int i=0; i<l_quantity->GetItemCount();i++){
    //     if(i==0){
    //         decimal = aries_acc::Decimal((const CompactDecimal *)l_quantity->GetData(i),l_quantity->GetDataType().DataType.Precision,l_quantity->GetDataType().DataType.Scale,0);
    //     }
    //     else{
    //         aries_acc::Decimal decimal1 = aries_acc::Decimal((const CompactDecimal *)l_quantity->GetData(i),l_quantity->GetDataType().DataType.Precision,l_quantity->GetDataType().DataType.Scale,0);
    //         // decimal += decimal1;
    //     }
    // }
    // cout<<"sum time cost:"<<t.end()<<endl;

    // char temp[128];
    // cout<<decimal.GetDecimal(temp)<<endl;
    
}
// extern "C" __global__ void addudt( const aries_acc::Decimal *input, aries_acc::Decimal *output )
// {
//     *output = 1 - *input;
//     printf(" %09d %09d %09d %09d %09d\n",(*output).v[4],(*output).v[3],(*output).v[2],(*output).v[1],(*output).v[0]);
// }

// TEST(cpu, partitionEx2)
// {
//     standard_context_t context;
//     mem_t< aries_acc::Decimal > input(1);
//     mem_t< aries_acc::Decimal > output(1);
//     aries_acc::Decimal decimal =  aries_acc::Decimal(12,2,"0.04");
//     *(input.data()) = decimal;
//     addudt<<<1,1>>>( input.data(), output.data());
//     cudaDeviceSynchronize();
//     char temp[128];
//     cout<< (*(input.data())).GetDecimal(temp)<<endl;
//     cout<< (*(output.data())).GetDecimal(temp)<<endl;
// }
// //     cout<<"yes"<<endl;
// //     AriesDataBufferSPtr l_quantity = ReadColumn( DB_NAME, "lineitem", 7 );
// //     cout<<"yes2"<<endl;
// //     cout<<l_quantity->GetItemCount()<<endl;
// //     cout<<"yse3"<<endl;
// //     aries_acc::Decimal decimal;
// //     for(int i=0; i<10;i++){
// //         aries_acc::Decimal decimal1 = aries_acc::Decimal((const CompactDecimal *)l_quantity->GetData(i),l_quantity->GetDataType().DataType.Precision,l_quantity->GetDataType().DataType.Scale,0);
// //         aries_acc::Decimal decimal = 1-decimal1;
// //         char temp[128];
// //         cout<<decimal.GetDecimal(temp)<<endl;
// //     }
// // }

// // TEST(cpu, partitionEx)
// // {
// //     AriesDataBufferSPtr l_orderkey = ReadColumn( DB_NAME, "lineitem", 1 );
// //     AriesDataBufferSPtr l_linenumber = ReadColumn( DB_NAME, "lineitem", 4 );
// //     AriesDataBufferSPtr o_orderkey = ReadColumn( DB_NAME, "orders", 1 );

// //     CPU_Timer t;
// //     t.begin();
// //     auto l_data = PartitionColumnDataEx( {l_orderkey, l_linenumber }, 10, 1, nullptr );
// //     cout<<"l_orderkey and l_linenumber time cost:"<<t.end()<<endl;
// //     t.begin();
// //     auto o_data = PartitionColumnDataEx( {o_orderkey}, 10, 0, nullptr );
// //     cout<<"o_orderkey time cost:"<<t.end()<<endl;

// //     cout<<"l_data:"<<endl;
// //     size_t total = 0;
// //     set<int> result;
// //     int index = 0;
// //     for( const auto& l : l_data.AllIndices )
// //     {
// //         auto buf0 = l_data.AllBuffers[ 0 ][ index ]->ToAriesDataBuffer();
// //         auto buf1 = l_data.AllBuffers[ 1 ][ index ]->ToAriesDataBuffer();
// //         ++index;
// //         buf0->PrefetchToCpu();
// //         buf1->PrefetchToCpu();
// //         int i = 0;
// //         for( auto ind : *l.Indices )
// //         {
// //             ASSERT_EQ( memcmp( l_orderkey->GetData( ind ), buf0->GetData( i ), l_orderkey->GetItemSizeInBytes() ), 0 );
// //             ASSERT_EQ( memcmp( l_linenumber->GetData( ind ), buf1->GetData( i ), l_linenumber->GetItemSizeInBytes() ), 0 );
// //             ++i;
// //         }
// //         result.insert( l.Indices->begin(), l.Indices->end() );
// //         total += l.Indices->size();
// //         cout<<l.Indices->size()<<endl;
// //     }
// //     cout<<"l_data total:"<< result.size() <<endl;

// //     cout<<"o_data:"<<endl;
// //     total = 0;
// //     for( const auto& o : o_data.AllIndices )
// //     {
// //         total += o.Indices->size();
// //         cout<<o.Indices->size()<<endl;
// //     }
// //     cout<<"o_data total:"<< total <<endl;
// // }

// // TEST(cpu, partition)
// // {
// //     AriesDataBufferSPtr l_orderkey = ReadColumn( DB_NAME, "lineitem", 1 );
// //     AriesDataBufferSPtr l_linenumber = ReadColumn( DB_NAME, "lineitem", 4 );
// //     AriesDataBufferSPtr o_orderkey = ReadColumn( DB_NAME, "orders", 1 );

// //     CPU_Timer t;
// //     t.begin();
// //     vector< PartitionedIndices > l_data = PartitionColumnData( {l_orderkey, l_linenumber }, 10, 1, nullptr );
// //     cout<<"l_orderkey and l_linenumber time cost:"<<t.end()<<endl;
// //     t.begin();
// //     vector< PartitionedIndices > o_data = PartitionColumnData( {o_orderkey}, 10, 0, nullptr );
// //     cout<<"o_orderkey time cost:"<<t.end()<<endl;

// //     cout<<"l_data:"<<endl;
// //     size_t total = 0;
// //     set<int> result;
// //     for( const auto& l : l_data )
// //     {
// //         result.insert( l.Indices->begin(), l.Indices->end() );
// //         total += l.Indices->size();
// //         cout<<l.Indices->size()<<endl;
// //     }
// //     cout<<"l_data total:"<< result.size() <<endl;

// //     cout<<"o_data:"<<endl;
// //     total = 0;
// //     for( const auto& o : o_data )
// //     {
// //         total += o.Indices->size();
// //         cout<<o.Indices->size()<<endl;
// //     }
// //     cout<<"o_data total:"<< total <<endl;

// //     vector< PartitionedIndices > l_data2 = PartitionColumnData( {l_orderkey, l_linenumber }, 10, 2, nullptr, l_data[ 0 ].Indices, 2 );
// //     cout<<"l_data2:"<<endl;
// //     total = 0;
// //     for( const auto& o : l_data2 )
// //     {
// //         total += o.Indices->size();
// //         cout<<o.Indices->size()<<endl;
// //     }
// //     cout<<"l_data2 total:"<< total <<endl;

// //     vector< PartitionedIndices > l_data3 = PartitionColumnData( {l_orderkey, l_linenumber }, 10, 2, nullptr, l_data2[ 0 ].Indices, 3 );
// //     cout<<"l_data3:"<<endl;
// //     total = 0;
// //     for( const auto& o : l_data3 )
// //     {
// //         total += o.Indices->size();
// //         cout<<o.Indices->size()<<endl;
// //     }
// //     cout<<"l_data3 total:"<< total <<endl;
// // }
// //TEST(engine, groups)
// //{
// //    standard_context_t context;
// //
// //    const char* file = "/var/rateup/data/scale_1/lineitem/lineitem8"; //L_RETURNFLAG char(1)
// //    const char* file2 = "/var/rateup/data/scale_1/lineitem/lineitem9"; //L_LINESTATUS char(1)
// //    int colSize = 1;
// //    int repeat = 1;
// //
// //    char *h_data;
// //    int alen = loadColumn( file, colSize, &h_data );
// //    managed_mem_t< char > needles( alen * repeat * colSize, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( needles.data() + i * alen * colSize, h_data, alen * colSize );
// //    }
// //
// //    free( h_data );
// //
// //    int blen = loadColumn( file2, colSize, &h_data );
// //    managed_mem_t< char > haystack( blen * repeat * colSize, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( haystack.data() + i * blen * colSize, h_data, blen * colSize );
// //    }
// //    free( h_data );
// //    managed_mem_t< char > clone_needles = needles.clone();
// //    managed_mem_t< char > clone_haystack = haystack.clone();
// //
// //    AriesDataType dataType { AriesValueType::CHAR, 1 };
// //    AriesDataBufferSPtr columnA = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
// //    AriesDataBufferSPtr columnB = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
// //
// //    columnA->AttachBuffer( ( int8_t* )clone_needles.release_data(), alen * repeat );
// //    columnB->AttachBuffer( ( int8_t* )clone_haystack.release_data(), blen * repeat );
// //
// //    std::vector< AriesDataBufferSPtr > columns;
// //    columns.push_back( columnA );
// //    columns.push_back( columnB );
// //    AriesInt32ArraySPtr outAssociated;
// //    AriesInt32ArraySPtr outGroups;
// //    context.timer_begin();
// //    int32_t groupCount = GroupColumns( columns, outAssociated, outGroups );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //    printf( "groupCount=%d\n", groupCount );
// //    ASSERT_TRUE( groupCount == 4 );
// //}
// //
// //extern "C" void sum_decimal( const char **input, int tupleNum, char *output )
// //{
// //    aries_acc::Decimal* pOut = ( aries_acc::Decimal* )output;
// //    aries_acc::Decimal* d1 = ( aries_acc::Decimal* )input[ 0 ];
// //    while( tupleNum-- )
// //        *pOut++ = *d1++;
// //}
// //
// //extern "C" void sum_decimal_ex( const char **input, int tupleNum, char *output )
// //{
// //    aries_acc::Decimal* pOut = ( aries_acc::Decimal* )output;
// //    aries_acc::Decimal* d1 = ( aries_acc::Decimal* )input[ 0 ];
// //    aries_acc::Decimal* d2 = ( aries_acc::Decimal* )input[ 1 ];
// //    aries_acc::Decimal* d3 = ( aries_acc::Decimal* )input[ 2 ];
// //    while( tupleNum-- )
// //        *pOut++ = *d1++ * ( 1 - *d2++ ) * ( 1 + *d3++ );
// //}
// //
// // TEST(engine, agg_lineitem)
// // {
// //    standard_context_t context;
// //  AriesDataBufferSPtr l_quantity = ReadColumn( DB_NAME, "lineitem", 8 );
// //    //read group by columns
// //     vector< AriesDataBufferSPtr > groupByColumns;
// //     groupByColumns.push_back(ReadColumn( DB_NAME, "lineitem", 9 ));
// //     groupByColumns.push_back(ReadColumn( DB_NAME, "lineitem", 10 ));
// //    AriesInt32ArraySPtr groupFlags;
// //    AriesInt32ArraySPtr associatedArray;
// //    AriesInt32ArraySPtr groupArray;

// //    //do group by
// //    aries_acc::GroupColumns( groupByColumns, associatedArray, groupArray, groupFlags, false );

   
// //    int32_t groupCount = aries_acc::GroupColumns( groupByColumns, associatedArray, groupArray, groupFlags, false );
// //    ASSERT_TRUE( groupCount == 4 );


// //    AriesDataBufferSPtr aggResult  = AggregateColumnData( l_quantity, AriesAggFunctionType::SUM, associatedArray, groupArray, groupFlags, false, false, SumStrategy::NONE );
// //    aggResult->Dump();

// // }

// //TEST(engine, agg_orders)
// //{
// //    standard_context_t context;
// //
// //    const char* file = "/var/rateup/data/scale_1/orders/orders4"; //O_ORDERDATE char(10)
// //    const char* file2 = "/var/rateup/data/scale_1/orders/orders7"; //O_SHIPPRIORITY int
// //    int colSize = 4;
// //    int repeat = 1;
// //
// //    char *h_data;
// //    int alen = loadColumn( file, colSize, &h_data );
// //    managed_mem_t< char > needles( alen * repeat * colSize, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( needles.data() + i * alen * colSize, h_data, alen * colSize );
// //    }
// //
// //    free( h_data );
// //
// //    int* int_data;
// //    colSize = sizeof(int);
// //    int blen = loadIntCol( file2, int_data );
// //    managed_mem_t< int > haystack( blen * repeat, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( haystack.data() + i * blen, int_data, blen * colSize );
// //    }
// //    //memcpy( haystack.data(), h_data, blen * repeat * colSize );
// //    cudaFree( int_data );
// //    managed_mem_t< char > clone_needles = needles.clone();
// //    managed_mem_t< int > clone_haystack = haystack.clone();
// //
// //    AriesDataType dataType { AriesValueType::DATE, 1 };
// //    AriesDataBufferSPtr columnA = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
// //    AriesDataType dataType3 { AriesValueType::INT32, 1 };
// //    AriesDataBufferSPtr columnB = make_shared< AriesDataBuffer >( AriesColumnType { dataType3, false, false } );
// //
// //    columnA->AttachBuffer( ( int8_t* )clone_needles.release_data(), alen * repeat );
// //    columnB->AttachBuffer( ( int8_t* )clone_haystack.release_data(), blen * repeat );
// //
// //    std::vector< AriesDataBufferSPtr > columns;
// //    columns.push_back( columnA );
// //    columns.push_back( columnB );
// //    AriesInt32ArraySPtr outAssociated;
// //    AriesInt32ArraySPtr outGroups;
// //    context.timer_begin();
// //    int32_t groupCount = GroupColumns( columns, outAssociated, outGroups );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //    printf( "groupCount=%d\n", groupCount );
// //    ASSERT_TRUE( groupCount == 2406 );
// //
// //    const char* file3 = "/var/rateup/data/scale_1/orders/orders3"; //O_TOTALPRICE float
// //    aries_acc::Decimal* decimal_data;
// //    int len = loadDecimalCol( file3, 15, 2, decimal_data );
// //    managed_mem_t< aries_acc::Decimal > quantity( len * repeat , context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( quantity.data() + i * len , decimal_data, len * sizeof( aries_acc::Decimal ) );
// //    }
// //    cudaFree( decimal_data );
// //    AriesDataType dataType2 { AriesValueType::DECIMAL, 1 };
// //    AriesDataBufferSPtr columnC = make_shared< AriesDataBuffer >( AriesColumnType { dataType2, false, false } );
// //    columnC->AttachBuffer( ( int8_t* )quantity.release_data(), len * repeat );
// //
// //
// //    context.timer_begin();
// //    AriesDataBufferSPtr aggResult = AggregateColumnData( columnC, AriesAggFunctionType::SUM, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// ////    for( int i = 0; i < groupCount; ++i )
// ////    {
// ////        printf( "data[%d]=%s\n", i, aggResult->GetDecimalAsString( i ).c_str() );
// ////    }
// //
// //    context.timer_begin();
// //    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::MAX, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// ////    for( int i = 0; i < groupCount; ++i )
// ////    {
// ////        printf( "data[%d]=%s\n", i, aggResult->GetDecimalAsString( i ).c_str() );
// ////    }
// //
// //    context.timer_begin();
// //    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::MIN, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// ////    for( int i = 0; i < groupCount; ++i )
// ////    {
// ////        printf( "data[%d]=%s\n", i, aggResult->GetDecimalAsString( i ).c_str() );
// ////    }
// //    context.timer_begin();
// //    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::COUNT, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// ////    int64_t* p2 = ( int64_t* )aggResult->GetData();
// ////    for( int i = 0; i < groupCount; ++i )
// ////    {
// ////        printf( "data[%d]=%ld\n", i, p2[i] );
// ////    }
// //}
// //
// //TEST(engine, agg_lineitem_decimal)
// //{
// //    assert( bool( std::is_same< std::common_type< aries_acc::Decimal, int64_t >::type, aries_acc::Decimal >::value ) );
// //    standard_context_t context;
// //
// //    const char* file = "/var/rateup/data/scale_1/lineitem/lineitem8"; //L_RETURNFLAG char(1)
// //    const char* file2 = "/var/rateup/data/scale_1/lineitem/lineitem9"; //L_LINESTATUS char(1)
// //    int colSize = 1;
// //    int repeat = 5;
// //
// //    char *h_data;
// //    int alen = loadColumn( file, colSize, &h_data );
// //    managed_mem_t< char > needles( alen * repeat * colSize, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( needles.data() + i * alen * colSize, h_data, alen * colSize );
// //    }
// //
// //    free( h_data );
// //
// //    int blen = loadColumn( file2, colSize, &h_data );
// //    managed_mem_t< char > haystack( blen * repeat * colSize, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( haystack.data() + i * blen * colSize, h_data, blen * colSize );
// //    }
// //    //memcpy( haystack.data(), h_data, blen * repeat * colSize );
// //    free( h_data );
// //    managed_mem_t< char > clone_needles = needles.clone();
// //    managed_mem_t< char > clone_haystack = haystack.clone();
// //
// //    AriesDataType dataType { AriesValueType::CHAR, 1 };
// //    AriesDataBufferSPtr columnA = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
// //    AriesDataBufferSPtr columnB = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
// //
// //    columnA->AttachBuffer( ( int8_t* )clone_needles.release_data(), alen * repeat );
// //    columnB->AttachBuffer( ( int8_t* )clone_haystack.release_data(), blen * repeat );
// //
// //    std::vector< AriesDataBufferSPtr > columns;
// //    columns.push_back( columnA );
// //    columns.push_back( columnB );
// //    AriesInt32ArraySPtr outAssociated;
// //    AriesInt32ArraySPtr outGroups;
// //    context.timer_begin();
// //    int32_t groupCount = GroupColumns( columns, outAssociated, outGroups );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //    printf( "groupCount=%d\n", groupCount );
// //    ASSERT_TRUE( groupCount == 4 );
// //
// //    const char* file3 = "/var/rateup/data/scale_1/lineitem/lineitem5"; //L_EXTENDEDPRICE float
// //    aries_acc::Decimal* decimal_data;
// //    int len = loadDecimalCol( file3, 10, 2, decimal_data );
// //    cout << "len================" << len << endl;
// //    colSize = sizeof(aries_acc::Decimal);
// //    managed_mem_t< aries_acc::Decimal > quantity( len * repeat, context );
// //
// //    for( int i = 0; i < repeat; ++i )
// //    {
// //        memcpy( quantity.data() + i * len, decimal_data, len * colSize );
// //    }
// //    cudaFree( decimal_data );
// //    AriesDataType dataType2 { AriesValueType::DECIMAL, 1 };
// //    AriesDataBufferSPtr columnC = make_shared< AriesDataBuffer >( AriesColumnType { dataType2, false, false } );
// //    columnC->AttachBuffer( ( int8_t* )quantity.release_data(), len * repeat );
// //
// //    cout << "cpu---------------<<" << endl;
// //    outGroups->Dump();
// //    CPU_Timer t;
// //    t.begin();
// //
// //    vector< int > segments;
// //    for( int i = 1; i < outGroups->GetItemCount(); ++i )
// //    {
// //        segments.push_back( outGroups->GetData()[ i ] );
// //    }
// //    segments.push_back( len * repeat );
// //    columnC->PrefetchToCpu();
// //
// //    vector< aries_acc::Decimal > ret = SegmentReduce( ( aries_acc::Decimal* )columnC->GetData(), outAssociated->GetData(), len * repeat, outGroups->GetData(),
// //            outGroups->GetItemCount(), agg_sum_t< aries_acc::Decimal >(), aries_acc::Decimal() );
// //
// //    cout << "cpu time:" << t.end() << endl;
// //    for( aries_acc::Decimal d : ret )
// //    {
// //        char result[ 64 ];
// //        cout << d.GetDecimal( result ) << endl;
// //    }
// //    t.begin();
// //    ret = SegmentReduce( ( aries_acc::Decimal* )columnC->GetData(), outAssociated->GetData(), len * repeat, outGroups->GetData(), outGroups->GetItemCount(),
// //            agg_max_t< aries_acc::Decimal >(), aries_acc::Decimal( std::numeric_limits< long >::min() ) );
// //
// //    cout << "cpu time:" << t.end() << endl;
// //    for( aries_acc::Decimal d : ret )
// //    {
// //        char result[ 64 ];
// //        cout << d.GetDecimal( result ) << endl;
// //    }
// //    t.begin();
// //    ret = SegmentReduce( ( aries_acc::Decimal* )columnC->GetData(), outAssociated->GetData(), len * repeat, outGroups->GetData(), outGroups->GetItemCount(),
// //            agg_min_t< aries_acc::Decimal >(), aries_acc::Decimal( std::numeric_limits< long >::max() ) );
// //
// //    cout << "cpu time:" << t.end() << endl;
// //    for( aries_acc::Decimal d : ret )
// //    {
// //        char result[ 64 ];
// //        cout << d.GetDecimal( result ) << endl;
// //    }
// //
// //    columnC->PrefetchToGpu();
// //    outAssociated->PrefetchToGpu();
// //    outGroups->PrefetchToGpu();
// //    context.timer_begin();
// //    AriesDataBufferSPtr aggResult = AggregateColumnData( columnC, AriesAggFunctionType::SUM, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// //    aggResult->Dump();
// //
// //    context.timer_begin();
// //    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::MAX, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// //    aggResult->Dump();
// //
// //    context.timer_begin();
// //    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::MIN, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// //    aggResult->Dump();
// //    context.timer_begin();
// //    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::COUNT, outAssociated, outGroups, nullptr, true, false, aries_acc::SumStrategy::NONE );
// //    printf( "gpu time: %3.1f\n", context.timer_end() );
// //
// //    aggResult->Dump();
// //}
// //
// TEST(engine, agg_lineitem_decimal_thrust)
// {

//     //  /data/tpch/tpch218_1/lineitem.tbl

//    assert( bool( std::is_same< std::common_type< aries_acc::Decimal, int64_t >::type, aries_acc::Decimal >::value ) );
//    standard_context_t context;

//    const char* file = "/var/rateup/data/scale_1/lineitem/lineitem8"; //L_RETURNFLAG char(1)
//    const char* file2 = "/var/rateup/data/scale_1/lineitem/lineitem9"; //L_LINESTATUS char(1)
//    int colSize = 1;
//    int repeat = 5;

//    char *h_data;
//    int alen = loadColumn( file, colSize, &h_data );
//    managed_mem_t< char > needles( alen * repeat * colSize, context );

//    for( int i = 0; i < repeat; ++i )
//    {
//        memcpy( needles.data() + i * alen * colSize, h_data, alen * colSize );
//    }

//    free( h_data );

//    int blen = loadColumn( file2, colSize, &h_data );
//    managed_mem_t< char > haystack( blen * repeat * colSize, context );

//    for( int i = 0; i < repeat; ++i )
//    {
//        memcpy( haystack.data() + i * blen * colSize, h_data, blen * colSize );
//    }
//    //memcpy( haystack.data(), h_data, blen * repeat * colSize );
//    free( h_data );
//    managed_mem_t< char > clone_needles = needles.clone();
//    managed_mem_t< char > clone_haystack = haystack.clone();

//    AriesDataType dataType { AriesValueType::CHAR, 1 };
//    AriesDataBufferSPtr columnA = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );
//    AriesDataBufferSPtr columnB = make_shared< AriesDataBuffer >( AriesColumnType { dataType, false, false } );

//    columnA->AttachBuffer( ( int8_t* )clone_needles.release_data(), alen * repeat );
//    columnB->AttachBuffer( ( int8_t* )clone_haystack.release_data(), blen * repeat );

//    std::vector< AriesDataBufferSPtr > columns;
//    columns.push_back( columnA );
//    columns.push_back( columnB );
//    AriesInt32ArraySPtr outAssociated;
//    AriesInt32ArraySPtr outGroups;
//    context.timer_begin();
//    int32_t groupCount = GroupColumns( columns, outAssociated, outGroups );
//    printf( "gpu time: %3.1f\n", context.timer_end() );
//    printf( "groupCount=%d\n", groupCount );
//    ASSERT_TRUE( groupCount == 4 );

//    const char* file3 = "/var/rateup/data/scale_1/lineitem/lineitem5"; //L_EXTENDEDPRICE float
//    aries_acc::Decimal* decimal_data;
//    int len = loadDecimalCol( file3, 10, 2, decimal_data );
//    cout << "len================" << len << endl;
//    colSize = sizeof(aries_acc::Decimal);
//    managed_mem_t< aries_acc::Decimal > quantity( len * repeat, context );

//    for( int i = 0; i < repeat; ++i )
//    {
//        memcpy( quantity.data() + i * len, decimal_data, len * colSize );
//    }
//    cudaFree( decimal_data );
//    AriesDataType dataType2 { AriesValueType::DECIMAL, 1 };
//    AriesDataBufferSPtr columnC = make_shared< AriesDataBuffer >( AriesColumnType { dataType2, false, false } );
//    columnC->AttachBuffer( ( int8_t* )quantity.release_data(), len * repeat );

//    cout << "cpu---------------<<" << endl;
//    outGroups->Dump();
//    CPU_Timer t;
//    t.begin();

//    vector< int > segments;
//    for( int i = 1; i < outGroups->GetItemCount(); ++i )
//    {
//        segments.push_back( outGroups->GetData()[ i ] );
//    }
//    segments.push_back( len * repeat );
//    columnC->PrefetchToCpu();

//    vector< aries_acc::Decimal > ret = SegmentReduce( ( aries_acc::Decimal* )columnC->GetData(), outAssociated->GetData(), len * repeat, outGroups->GetData(),
//            outGroups->GetItemCount(), agg_sum_t< aries_acc::Decimal >(), aries_acc::Decimal() );

//    cout << "cpu time:" << t.end() << endl;
//    for( aries_acc::Decimal d : ret )
//    {
//        char result[ 64 ];
//        cout << d.GetDecimal( result ) << endl;
//    }
//    t.begin();
//    ret = SegmentReduce( ( aries_acc::Decimal* )columnC->GetData(), outAssociated->GetData(), len * repeat, outGroups->GetData(), outGroups->GetItemCount(),
//            agg_max_t< aries_acc::Decimal >(), aries_acc::Decimal( std::numeric_limits< long >::min() ) );

//    cout << "cpu time:" << t.end() << endl;
//    for( aries_acc::Decimal d : ret )
//    {
//        char result[ 64 ];
//        cout << d.GetDecimal( result ) << endl;
//    }
//    t.begin();
//    ret = SegmentReduce( ( aries_acc::Decimal* )columnC->GetData(), outAssociated->GetData(), len * repeat, outGroups->GetData(), outGroups->GetItemCount(),
//            agg_min_t< aries_acc::Decimal >(), aries_acc::Decimal( std::numeric_limits< long >::max() ) );

//    cout << "cpu time:" << t.end() << endl;
//    for( aries_acc::Decimal d : ret )
//    {
//        char result[ 64 ];
//        cout << d.GetDecimal( result ) << endl;
//    }

//    AriesInt32ArraySPtr groups = outAssociated->CloneWithNoContent();
//    device_ptr< int32_t > group_data = device_pointer_cast( groups->GetData() );
//    int32_t* pGroupData = outGroups->GetData();
//    int i = 1;
//    for( ; i < outGroups->GetItemCount(); ++i )
//    {
//        size_t sz = pGroupData[ i ] - pGroupData[ i - 1 ];
//        thrust::fill_n( thrust::device, group_data, sz, i );
//        group_data += sz;
//    }
//    thrust::fill_n( thrust::device, group_data, len * repeat - pGroupData[ i - 1 ], i );

//    columnC->PrefetchToGpu( 0 );
//    outAssociated->PrefetchToGpu( 0 );
//    outGroups->PrefetchToGpu( 0 );
//    groups->PrefetchToGpu( 0 );

//    AriesDataBufferSPtr tmp = columnC->CloneWithNoContent();
//    tmp->PrefetchToGpu();
//    aries_acc::shuffle_column_data( columnC->GetData(), columnC->GetDataType(), columnC->GetItemCount(), outAssociated->GetData(), tmp->GetData(),
//            context );

//    AriesInt32ArraySPtr ogroup = outGroups->CloneWithNoContent();
//    AriesDataBufferSPtr otmp = columnC->CloneWithNoContent( outGroups->GetItemCount() );
//    otmp->PrefetchToGpu();
//    thrust::pair< thrust::device_ptr< int32_t >, thrust::device_ptr< aries_acc::Decimal > > new_end;
//    thrust::equal_to< int32_t > binary_pred;
//    thrust::plus< aries_acc::Decimal > binary_op;
//    context.timer_begin();
//    new_end = thrust::reduce_by_key( thrust::device, device_pointer_cast( groups->GetData() ),
//            device_pointer_cast( groups->GetData() ) + len * repeat, device_pointer_cast( ( aries_acc::Decimal* )tmp->GetData() ),
//            device_pointer_cast( ogroup->GetData() ), device_pointer_cast( ( aries_acc::Decimal* )otmp->GetData() ), binary_pred, binary_op );

//    printf( "thrust gpu time: %3.1f\n", context.timer_end() );
//    otmp->Dump();
//    tmp->PrefetchToCpu();
//    otmp->PrefetchToCpu();
//    groups->PrefetchToCpu();
//    ogroup->PrefetchToCpu();
//    t.begin();
//    new_end = thrust::reduce_by_key( thrust::host, ( groups->GetData() ), ( groups->GetData() ) + len * repeat, ( ( aries_acc::Decimal* )tmp->GetData() ),
//            ( ogroup->GetData() ), ( ( aries_acc::Decimal* )otmp->GetData() ), binary_pred, binary_op );

//    cout << "thrust cpu time:" << t.end() << endl;

//    context.timer_begin();
//    AriesDataBufferSPtr aggResult = AggregateColumnData( columnC, AriesAggFunctionType::SUM, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );

//    printf( "gpu time: %3.1f\n", context.timer_end() );

//    aggResult->Dump();

//    context.timer_begin();
//    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::MAX, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
//    printf( "gpu time: %3.1f\n", context.timer_end() );

//    aggResult->Dump();

//    context.timer_begin();
//    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::MIN, outAssociated, outGroups, nullptr, false, false, aries_acc::SumStrategy::NONE );
//    printf( "gpu time: %3.1f\n", context.timer_end() );

//    aggResult->Dump();
//    context.timer_begin();
//    aggResult = AggregateColumnData( columnC, AriesAggFunctionType::COUNT, outAssociated, outGroups, nullptr, true, false, aries_acc::SumStrategy::NONE );
//    printf( "gpu time: %3.1f\n", context.timer_end() );

//    aggResult->Dump();
// }

