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

__global__ void non_to_non_sig_add( const aries_acc::Decimal* input_a, const aries_acc::Decimal* input_b, size_t tupleNum, aries_acc::Decimal *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        output[i] = input_a[i] + input_b[i];
    }
}

__global__ void non_to_non_sig_mul( const aries_acc::Decimal* input_a, const aries_acc::Decimal* input_b, size_t tupleNum, aries_acc::Decimal *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        output[i] = input_a[i] * input_b[i];
    }
}

__global__ void non_to_non_mlt_add( const aries_acc::Decimal* input_a, const aries_acc::Decimal* input_b, size_t tupleNum, aries_acc::Decimal *output )
{
    int32_t group_thread=threadIdx.x & TPI-1;
    int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(index>=tupleNum)
        return;
    uint32_t var_1[LIMBS] = {0};
    uint8_t var_1_sign = 0;
    uint32_t var_2[LIMBS] = {0};
    uint8_t var_2_sign = 0;

    var_1_sign = input_a[index].sign;
    for(int i=0; i<LIMBS; i++)
        var_1[i] = input_a[index].v[group_thread*LIMBS+i];

    var_2_sign = input_b[index].sign;
    for(int i=0; i<LIMBS; i++)
        var_2[i] = input_b[index].v[group_thread*LIMBS+i];

    uint32_t var_0[LIMBS] = {0};
    uint8_t var_0_sign = 0;
    var_0_sign = aries_acc::operator_add(var_0, var_1, var_2, 0, var_1_sign, var_2_sign);
    
    for(int i=0; i<LIMBS; i++)
        output[index].v[group_thread*LIMBS+i] = var_0[i];
    
    // uint32_t var_0_sign = aries_acc::operator_add(output[index].v+group_thread*LIMBS, input_a[index].v+group_thread*LIMBS, input_b[index].v+group_thread*LIMBS, 0, input_a[index].sign, input_b[index].sign);

    if(group_thread==0){
        output[index].sign = var_0_sign;
        output[index].prec = 12;
        output[index].frac = 2;
    }
}

__global__ void non_to_non_mlt_mul( const aries_acc::Decimal* input_a, const aries_acc::Decimal* input_b, size_t tupleNum, aries_acc::Decimal *output )
{
    int32_t group_thread=threadIdx.x & TPI-1;
    int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(index>=tupleNum)
        return;
    uint32_t var_1[LIMBS] = {0};
    uint8_t var_1_sign = 0;
    uint32_t var_2[LIMBS] = {0};
    uint8_t var_2_sign = 0;
   
    var_1_sign = input_a[index].sign;
    for(int i=0; i<LIMBS; i++)
        var_1[i] = input_a[index].v[group_thread*LIMBS+i];

    var_2_sign = input_b[index].sign;
    for(int i=0; i<LIMBS; i++)
        var_2[i] = input_b[index].v[group_thread*LIMBS+i];

    uint32_t var_0[LIMBS] = {0};
    uint8_t var_0_sign = 0;
    aries_acc::operator_mul(var_0, var_1, var_2);
    var_0_sign = var_1_sign ^ var_2_sign;

    for(int i=0; i<LIMBS; i++)
        output[index].v[group_thread*LIMBS+i] = var_0[i];
    
    // aries_acc::operator_mul(output[index].v+group_thread*LIMBS, input_a[index].v+group_thread*LIMBS, input_b[index].v+group_thread*LIMBS);

    if(group_thread==0){
        output[index].sign = var_0_sign;
        output[index].prec = 12;
        output[index].frac = 4;
    }
}

void main_execute_non_to_non(int type)
{

    GPUTimer gpuTimer;
    float gpuPerf = 0.0;
    // 从 data 目录中读取数据 这里读取到的数据是 compactDecimal
    // 这里取到了两组数据 lineitem 的第五列 和 第七列 它们的格式都是 prec = 12 frac = 2
    standard_context_t context;
    AriesDataBufferSPtr l_quantity = ReadColumn( DB_NAME, "lineitem", 5 );
    AriesDataBufferSPtr l_discount = ReadColumn( DB_NAME, "lineitem", 7 );
    
    // 两组数据的列数 和 两组数据 compactDecimal的字节数
    size_t calc_number = l_quantity->GetItemCount();
    size_t byte_per_op_unit = GetDecimalRealBytes(12, 2);
    
    // 申请 Decimal 操作数的 CPU 和 GPU 存储空间
    aries_acc::Decimal *q_cpu, *q_gpu, *d_cpu, *d_gpu;
    q_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
    d_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
    cudaMalloc((void **)&q_gpu, calc_number * sizeof(aries_acc::Decimal));
    cudaMalloc((void **)&d_gpu, calc_number * sizeof(aries_acc::Decimal));

    // 将从 存储文件中读取到的 compactDecimal 转成 Decimal 并拷贝到 gpu 上
    for(size_t i=0; i<calc_number; i++){
        q_cpu[i] = aries_acc::Decimal((CompactDecimal *)(l_quantity->GetData()+i*byte_per_op_unit), 12, 2);
        d_cpu[i] = aries_acc::Decimal((CompactDecimal *)(l_discount->GetData()+i*byte_per_op_unit), 12, 2);
    }
    cudaMemcpy(q_gpu, q_cpu, calc_number * sizeof(aries_acc::Decimal), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu, d_cpu, calc_number * sizeof(aries_acc::Decimal), cudaMemcpyHostToDevice);

    // 申请结果在 CPU 和 GPU 上的空间
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
            non_to_non_sig_add<<<blockN, threadN>>>(q_gpu, d_gpu, calc_number, ans_gpu);
        });
        cout<<"NonCompactToNonCompact --> sig_add :"<<gpuPerf<<"ms"<<endl;
        fp = fopen("../Varify/NonCompactToNonCompact_sig_add.txt", "ab+");
    }
    else if(type == 2){
        // 调用 kernel
        int threadN = 256;
        int blockN = (calc_number*TPI - 1)/threadN + 1;
        gpuPerf = gpuTimer.timing( [&](){
            non_to_non_mlt_add<<<blockN, threadN>>>(q_gpu, d_gpu, calc_number, ans_gpu);
        });
        cout<<"NonCompactToNonCompact --> mlt_add :"<<gpuPerf<<"ms"<<endl;
        fp = fopen("../Varify/NonCompactToNonCompact_mlt_add.txt", "ab+");
    }
    else if(type == 3){
        // 调用 kernel
        int threadN = 256;
        int blockN = (calc_number - 1)/threadN + 1;
        gpuPerf = gpuTimer.timing( [&](){
            non_to_non_sig_mul<<<blockN, threadN>>>(q_gpu, d_gpu, calc_number, ans_gpu);
        });
        cout<<"NonCompactToNonCompact --> sig_mul :"<<gpuPerf<<"ms"<<endl;
        fp = fopen("../Varify/NonCompactToNonCompact_sig_mul.txt", "ab+");
    }
    else if(type == 4){
        // 调用 kernel
        int threadN = 256;
        int blockN = (calc_number*TPI - 1)/threadN + 1;
        gpuPerf = gpuTimer.timing( [&](){
            non_to_non_mlt_mul<<<blockN, threadN>>>(q_gpu, d_gpu, calc_number, ans_gpu);
        });
        cout<<"NonCompactToNonCompact --> mlt_mul :"<<gpuPerf<<"ms"<<endl;
        fp = fopen("../Varify/NonCompactToNonCompact_mlt_mul.txt", "ab+");
    }

    // 将数据拷回到 CPU 上并输入到文件中
    cudaMemcpy(ans_cpu, ans_gpu, ans_sum_byte, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // // compact 输出
    // fseek(fp, 0, SEEK_SET);
    // fwrite(ans_cpu ,1 , ans_sum_byte, fp);

    // decimal 输出
    for(int i=0; i<calc_number; i++){
        char result[2048];
        for(int j=INDEX_LAST_DIG ; j>=0 ;j--){
                sprintf(result+(INDEX_LAST_DIG-j)*8,"%08x",ans_cpu[i].v[j]);
        }
        // sprintf(result+NUM_TOTAL_DIG*8," sign = %d , frac = %d",ans_cpu[i].sign,ans_cpu[i].frac);
        int len = strlen(result);
        fwrite(result,len,1,fp);
        fwrite("\r\n",1,2,fp);
    }

    fclose(fp);

    free(q_cpu);
    free(d_cpu);
    free(ans_cpu);
    cudaFree(q_gpu);
    cudaFree(d_gpu);
    cudaFree(ans_gpu);
}

TEST(NonCompactToNonCompact, sig_add)
{
    main_execute_non_to_non(1);
}

TEST(NonCompactToNonCompact, mlt_add)
{
    main_execute_non_to_non(2);
}

TEST(NonCompactToNonCompact, sig_mul)
{
    main_execute_non_to_non(3);
}

TEST(NonCompactToNonCompact, mlt_mul)
{
    main_execute_non_to_non(4);
}