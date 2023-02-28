#include "test_common.h"
#include "AriesEngine/cpu_algorithm.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "CudaAcc/AriesEngineAlgorithm.h"
using namespace aries_acc;
static const char* DB_NAME = "scale_1";

class GPUTimer {
public:
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    virtual ~GPUTimer() { }

    template <typename Func>
    float timing(Func func) {
        float perf;

        cudaEventRecord(start);

        func();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&perf, start, stop);

        return perf;
    }
};

__global__ void co_to_non_sig( const int8_t* input_a, size_t tupleNum, size_t item_size, aries_acc::Decimal *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        aries_acc::Decimal columnId_1_( (CompactDecimal*)(input_a+i*item_size), 12, 2);
        output[i] = columnId_1_;
    }
}

__global__ void co_to_non_mlt( const int8_t* input_a, size_t tupleNum, size_t item_size, aries_acc::Decimal *output )
{
    int32_t group_thread=threadIdx.x & TPI-1;
    int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(index>=tupleNum)
        return;

    uint32_t var_1[LIMBS] = {0};
    uint8_t var_1_sign = 0;

    char *var_1_temp = (char *)(input_a+index*item_size);
    var_1_temp += 5;
    char c_1= *var_1_temp;
    var_1_sign = GET_SIGN_FROM_BIT(c_1);
    if(group_thread == 0){
        aries_memcpy(var_1, ((CompactDecimal*)( input_a+index*item_size )) + group_thread * LIMBS * 4, 6);
        char *inner_temp = (char *)(var_1);
        inner_temp += 6 - 1;
        *inner_temp = *inner_temp & 0x7f;
    }

    // aries_memcpy(output[index].v+group_thread*LIMBS, var_1,  LIMBS*4);
    for(int i=0; i<LIMBS; i++)
        output[index].v[group_thread*LIMBS+i] = var_1[i];

    if(group_thread == 0){
        output[index].sign = var_1_sign;
        output[index].prec = 12;
        output[index].frac = 2;
    }
}

__global__ void non_to_co_sig( const aries_acc::Decimal* input_a, size_t tupleNum, char *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        aries_acc::Decimal columnId_1_ = input_a[i];
        auto tmp = output + i * (size_t)6;
        columnId_1_.ToCompactDecimal(tmp, 6);
    }
}

__global__ void non_to_co_mlt( const aries_acc::Decimal* input_a, size_t tupleNum, char *output )
{
    int32_t group_thread=threadIdx.x & TPI-1;
    int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(index>=tupleNum)
        return;
    uint32_t var_1[LIMBS] = {0};
    uint8_t var_1_sign = 0;

    for(int i=0; i<LIMBS; i++)
        var_1[i] = input_a[index].v[group_thread*LIMBS+i];

    auto ans_tmp = output +  (long long)(index) * 6 + group_thread * LIMBS * 4;
    if( group_thread == 0){
        aries_memcpy(ans_tmp, var_1, 6);
    }
    if( group_thread == 0 ){
        var_1_sign = input_a[index].sign;
        char *buf = output + (long long)(index) * 6;
        SET_SIGN_BIT( buf[6-1], var_1_sign);
    }
}

void main_execute_co_to_non(int type){
        GPUTimer gpuTimer;
        float gpuPerf = 0.0;
        // 从 data 目录中读取数据 这里读取到的数据是 compactDecimal
        // 这里取到了两组数据 lineitem 的第五列 和 第七列 它们的格式都是 prec = 12 frac = 2
        standard_context_t context;
        AriesDataBufferSPtr l_quantity = ReadColumn( DB_NAME, "lineitem", 5 );

        // 两组数据的列数 和 两组数据 compactDecimal的字节数
        size_t calc_number = l_quantity->GetItemCount();
        size_t byte_per_op_unit = GetDecimalRealBytes(12, 2);
        
        // 结果需要的内存空间的大小
        size_t ans_sum_byte = calc_number * sizeof(aries_acc::Decimal);
        aries_acc::Decimal *ans_cpu, *ans_gpu;
        ans_cpu = (aries_acc::Decimal *)malloc(ans_sum_byte);
        cudaMalloc((void **)&ans_gpu, ans_sum_byte);
        FILE *fp;

        if(type == 1){
            // 调用 kernel
            int threadN = 256;
            int blockN = (calc_number - 1)/threadN + 1;
            gpuPerf = gpuTimer.timing( [&](){
                co_to_non_sig<<<blockN, threadN>>>(l_quantity->GetData(), calc_number, byte_per_op_unit, ans_gpu);
            });
            cout<<"CompactToNonCompact --> sig :"<<gpuPerf<<"ms"<<endl;

            fp = fopen("../Varify/CompactToNonCompact_sig.txt", "ab+");
        }
        else if(type == 2){
            // 调用 kernel
            int threadN = 256;
            int blockN = (calc_number*TPI - 1)/threadN + 1;
            gpuPerf = gpuTimer.timing( [&](){
                co_to_non_mlt<<<blockN, threadN>>>(l_quantity->GetData(), calc_number, byte_per_op_unit, ans_gpu);
            });
            cout<<"CompactToNonCompact --> mlt :"<<gpuPerf<<"ms"<<endl;

            fp = fopen("../Varify/CompactToNonCompact_mlt.txt", "ab+");
        }

        // 将数据拷回到 CPU 上并输入到文件中
        cudaMemcpy(ans_cpu, ans_gpu, ans_sum_byte, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // // decimal 输出
        // for(int i=0; i<calc_number; i++){
        //     char result[2048];
        //     for(int j=INDEX_LAST_DIG ; j>=0 ;j--){
        //             sprintf(result+(INDEX_LAST_DIG-j)*8,"%08x",ans_cpu[i].v[j]);
        //     }
        //     sprintf(result+NUM_TOTAL_DIG*8," sign = %d , frac = %d",ans_cpu[i].sign,ans_cpu[i].frac);
        //     int len = strlen(result);
        //     fwrite(result,len,1,fp);
        //     fwrite("\r\n",1,2,fp);
        // }

        fclose(fp);
        free(ans_cpu);
        cudaFree(ans_gpu);
}

void main_execute_non_to_co(int type){
        GPUTimer gpuTimer;
        float gpuPerf = 0.0;
        // 从 data 目录中读取数据 这里读取到的数据是 compactDecimal
        // 这里取到了两组数据 lineitem 的第五列 和 第七列 它们的格式都是 prec = 12 frac = 2
        standard_context_t context;
        AriesDataBufferSPtr l_quantity = ReadColumn( DB_NAME, "lineitem", 5 );

        // 两组数据的列数 和 两组数据 compactDecimal的字节数
        size_t calc_number = l_quantity->GetItemCount();
        size_t byte_per_op_unit = GetDecimalRealBytes(12, 2);
        
        // 申请 Decimal 操作数的 CPU 和 GPU 存储空间
        aries_acc::Decimal *q_cpu, *q_gpu;
        q_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
        cudaMalloc((void **)&q_gpu, calc_number * sizeof(aries_acc::Decimal));

        // 将从 存储文件中读取到的 compactDecimal 转成 Decimal 并拷贝到 gpu 上
        for(size_t i=0; i<calc_number; i++){
            q_cpu[i] = aries_acc::Decimal((CompactDecimal *)(l_quantity->GetData()+i*byte_per_op_unit), 12, 2);
        }
        cudaMemcpy(q_gpu, q_cpu, calc_number * sizeof(aries_acc::Decimal), cudaMemcpyHostToDevice);

        // 结果需要的内存空间的大小
        int byte_per_ans_unit = GetDecimalRealBytes(12, 2);
        size_t ans_sum_byte = byte_per_ans_unit * calc_number;
        char *ans_cpu, *ans_gpu;
        ans_cpu = (char *)malloc(ans_sum_byte);
        cudaMalloc((void **)&ans_gpu, ans_sum_byte);
        FILE *fp;

        if(type == 1){
            // 调用 kernel
            int threadN = 256;
            int blockN = (calc_number - 1)/threadN + 1;
            gpuPerf = gpuTimer.timing( [&](){
                non_to_co_sig<<<blockN, threadN>>>(q_gpu, calc_number, ans_gpu);
            });
            cout<<"NonCompactToCompact --> sig :"<<gpuPerf<<"ms"<<endl;

            fp = fopen("../Varify/NonCompactToCompact_sig.txt", "ab+");
        }
        else if(type == 2){
            // 调用 kernel
            int threadN = 256;
            int blockN = (calc_number*TPI - 1)/threadN + 1;
            gpuPerf = gpuTimer.timing( [&](){
                non_to_co_mlt<<<blockN, threadN>>>(q_gpu, calc_number, ans_gpu);
            });
            cout<<"NonCompactToCompact --> mlt :"<<gpuPerf<<"ms"<<endl;

            fp = fopen("../Varify/NonCompactToCompact_mlt.txt", "ab+");
        }

        // 将数据拷回到 CPU 上并输入到文件中
        cudaMemcpy(ans_cpu, ans_gpu, ans_sum_byte, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // // decimal 输出
        // aries_acc::Decimal *ans_dec_cpu;
        // ans_dec_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
        // for(size_t i=0; i<calc_number; i++){
        //         if(type == 1 || type == 2 ){
        //                 int byte_per_ans_unit = GetDecimalRealBytes(12, 2); 
        //                 ans_dec_cpu[i] = aries_acc::Decimal((CompactDecimal *)(ans_cpu+i*byte_per_ans_unit), 12, 2);
        //         }
        //         else{
        //                 int byte_per_ans_unit = GetDecimalRealBytes(24, 4); 
        //                 ans_dec_cpu[i] = aries_acc::Decimal((CompactDecimal *)(ans_cpu+i*byte_per_ans_unit), 24, 4);
        //         }       
        // }
        // for(int i=0; i<calc_number; i++){
        //         char result[2048];
        //         for(int j=INDEX_LAST_DIG ; j>=0 ;j--){
        //                 sprintf(result+(INDEX_LAST_DIG-j)*8,"%08x",ans_dec_cpu[i].v[j]);
        //         }
        //         sprintf(result+NUM_TOTAL_DIG*8," sign = %d , frac = %d",ans_dec_cpu[i].sign,ans_dec_cpu[i].frac);
        //         int len = strlen(result);
        //         fwrite(result,len,1,fp);
        //         fwrite("\r\n",1,2,fp);
        // }

        fclose(fp);
        free(q_cpu);
        free(ans_cpu);
        cudaFree(q_gpu);
        cudaFree(ans_gpu);
}

TEST(CompactToNonCompact, sig)
{
        main_execute_co_to_non(1);
}

TEST(CompactToNonCompact, mlt)
{
        main_execute_co_to_non(2);
}

TEST(NonCompactToCompact, sig)
{
        main_execute_non_to_co(1);
}

TEST(NonCompactToCompact, mlt)
{
        main_execute_non_to_co(2);
}