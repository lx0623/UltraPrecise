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

__global__ void co_to_co_sig_add_align( const int8_t* input_a, const int8_t* input_b, size_t tupleNum, size_t item_size, char *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        aries_acc::Decimal columnId_1_( (CompactDecimal*)(input_a+i*item_size), 17, 2);
        aries_acc::Decimal columnId_2_( (CompactDecimal*)(input_b+i*item_size), 17, 2);
        aries_acc::Decimal columnId_3_ = columnId_1_ + columnId_2_;
        auto tmp = output + i * (size_t)8;
        columnId_3_.ToCompactDecimal(tmp, 8);
    }
}

__global__ void co_to_co_sig_mul_align( const int8_t* input_a, const int8_t* input_b, size_t tupleNum, size_t item_size, char *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for( int64_t i = tid; i < tupleNum; i += stride )
    {
        aries_acc::Decimal columnId_1_( (CompactDecimal*)(input_a+i*item_size), 17, 2);
        aries_acc::Decimal columnId_2_( (CompactDecimal*)(input_b+i*item_size), 17, 2);
        aries_acc::Decimal columnId_3_ = columnId_1_ * columnId_2_;
        auto tmp = output + i * (size_t)16;
        columnId_3_.ToCompactDecimal(tmp, 16);
    }
}

__global__ void co_to_co_mlt_add_align( const int8_t* input_a, const int8_t* input_b, size_t tupleNum, size_t item_size, char *output )
{
    int32_t group_thread=threadIdx.x & TPI-1;
    int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(index>=tupleNum)
        return;
    uint32_t var_1[LIMBS] = {0};
    uint8_t var_1_sign = 0;
    uint32_t var_2[LIMBS] = {0};
    uint8_t var_2_sign = 0;
    char *var_1_temp = (char *)(input_a+index*item_size);
    var_1_temp += 5;
    char c_1= *var_1_temp;
    var_1_sign = GET_SIGN_FROM_BIT(c_1);
    if(group_thread == 0){
            aries_memcpy(var_1, ((CompactDecimal*)( input_a+index*item_size )) + group_thread * LIMBS * 4, 8);
            char *inner_temp = (char *)(var_1);
            inner_temp += 8 - 1;
            *inner_temp = *inner_temp & 0x7f;
    }
    char *var_2_temp = (char *)( input_b+index*item_size );
    var_2_temp += 5;
    char c_2= *var_2_temp;
    var_2_sign = GET_SIGN_FROM_BIT(c_2);
    if(group_thread == 0){
            aries_memcpy(var_2, ((CompactDecimal*)( input_b+index*item_size )) + group_thread * LIMBS * 4, 8);
            char *inner_temp = (char *)(var_2);
            inner_temp += 8 - 1;
            *inner_temp = *inner_temp & 0x7f;
    }
    uint32_t var_0[LIMBS] = {0};
    uint8_t var_0_sign = 0;
    var_0_sign = aries_acc::operator_add(var_0, var_1, var_2, 0, var_1_sign, var_2_sign);
    auto ans_tmp = output +  (long long)(index) * 8 + group_thread * LIMBS * 4;
    if(group_thread==0){
            aries_memcpy(ans_tmp, var_0, 8);
            char *buf = output + (long long)(index) * 8;
            SET_SIGN_BIT( buf[6-1], var_0_sign);
       }
}

__global__ void co_to_co_mlt_mul_align( const int8_t* input_a, const int8_t* input_b, size_t tupleNum, size_t item_size, char *output )
{
    int32_t group_thread=threadIdx.x & TPI-1;
    int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(index>=tupleNum)
        return;
    uint32_t var_1[LIMBS] = {0};
    uint8_t var_1_sign = 0;
    uint32_t var_2[LIMBS] = {0};
    uint8_t var_2_sign = 0;
    char *var_1_temp = (char *)( input_a+index*item_size );
    var_1_temp += 5;
    char c_1= *var_1_temp;
    var_1_sign = GET_SIGN_FROM_BIT(c_1);
    if(group_thread == 0){
            aries_memcpy(var_1, ((CompactDecimal*)( input_a+index*item_size )) + group_thread * LIMBS * 4, 8);
            char *inner_temp = (char *)(var_1);
            inner_temp += 8 - 1;
            *inner_temp = *inner_temp & 0x7f;
    }
    char *var_2_temp = (char *)( input_b+index*item_size );
    var_2_temp += 5;
    char c_2= *var_2_temp;
    var_2_sign = GET_SIGN_FROM_BIT(c_2);
    if(group_thread == 0){
            aries_memcpy(var_2, ((CompactDecimal*)( input_b+index*item_size )) + group_thread * LIMBS * 4, 8);
            char *inner_temp = (char *)(var_2);
            inner_temp += 8 - 1;
            *inner_temp = *inner_temp & 0x7f;
    }
    uint32_t var_0[LIMBS] = {0};
    uint8_t var_0_sign = 0;
    aries_acc::operator_mul(var_0, var_1, var_2);
    var_0_sign = var_1_sign ^ var_2_sign;
    auto ans_tmp = output +  (long long)(index) * 16 + group_thread * LIMBS * 4;
    if(group_thread==0){
            aries_memcpy(ans_tmp, var_0, 16);
            char *buf = output + (long long)(index) * 16;
            SET_SIGN_BIT( buf[16-1], var_0_sign);
    }
}

void main_execute_co_to_co_align(int type){
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

        // 将 compactDecimal 压缩成 4字节 向上对齐的 compactDecimal
        // 申请 Decimal 操作数的 CPU 和 GPU 存储空间
        aries_acc::Decimal *q_cpu, *q_gpu, *d_cpu, *d_gpu;
        q_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
        d_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
        cudaMalloc((void **)&q_gpu, calc_number * sizeof(aries_acc::Decimal));
        cudaMalloc((void **)&d_gpu, calc_number * sizeof(aries_acc::Decimal));
        
        // 将从 存储文件中读取到的 compactDecimal 转成 Decimal
        for(size_t i=0; i<calc_number; i++){
            q_cpu[i] = aries_acc::Decimal((CompactDecimal *)(l_quantity->GetData()+i*byte_per_op_unit), 12, 2);
            d_cpu[i] = aries_acc::Decimal((CompactDecimal *)(l_discount->GetData()+i*byte_per_op_unit), 12, 2);
        }

        // 申请 向上对齐的compactDecimal 空间 重新压缩
        // 6 向上取整
        int new_byte_per_op_unit = GetDecimalRealBytes(17, 2);
        int8_t *c_q_gpu, *c_d_gpu;
        int8_t *c_q_cpu = (int8_t *)malloc(calc_number * new_byte_per_op_unit);
        int8_t *c_d_cpu = (int8_t *)malloc(calc_number * new_byte_per_op_unit);
        cudaMalloc((void **)&c_q_gpu, calc_number * new_byte_per_op_unit);
        cudaMalloc((void **)&c_d_gpu, calc_number * new_byte_per_op_unit);
        // 重新压缩
        for(size_t i=0; i<calc_number; i++){
            auto q_tmp = c_q_cpu + i * (size_t)new_byte_per_op_unit;
            q_cpu[i].ToCompactDecimal((char *)q_tmp, new_byte_per_op_unit);
            auto d_tmp = c_d_cpu + i * (size_t)new_byte_per_op_unit;
            d_cpu[i].ToCompactDecimal((char *)d_tmp, new_byte_per_op_unit);
        }
        // 将数据拷贝到 gpu 上
        cudaMemcpy(c_q_gpu, c_q_cpu, calc_number * new_byte_per_op_unit, cudaMemcpyHostToDevice);
        cudaMemcpy(c_d_gpu, c_d_cpu, calc_number * new_byte_per_op_unit, cudaMemcpyHostToDevice);
        
        char *ans_cpu, *ans_gpu;
        // 结果需要的内存空间的大小
        size_t ans_sum_byte;
        // 输出的文件
        FILE *fp;
        if(type == 1 || type == 2){
                // 计算 l_quantity + l_discount 列 它们的结果类型 ans.prec = 13 ans.frac = 2
                int byte_per_ans_unit = GetDecimalRealBytes(18, 2);
                ans_sum_byte = byte_per_ans_unit * calc_number;

                // 申请结果在 CPU 和 GPU 上的空间
                ans_cpu = (char *)malloc(ans_sum_byte);
                cudaMalloc((void **)&ans_gpu, ans_sum_byte);

                if(type == 1){
                        // 调用 kernel
                        int threadN = 256;
                        size_t blockN = (calc_number - 1)/threadN + 1;
                        gpuPerf = gpuTimer.timing( [&](){
                                co_to_co_sig_add_align<<<blockN, threadN>>>(c_q_gpu, c_d_gpu, calc_number, new_byte_per_op_unit, ans_gpu);
                        });
                        cout<<"CompactToCompactAlign --> sig_add :"<<gpuPerf<<"ms"<<endl;
                        fp = fopen("../Varify/CompactToCompactAlign_sig_add.txt", "ab+");
                }
                else{
                        // 调用 kernel
                        int threadN = 256;
                        size_t blockN = (calc_number*TPI - 1)/threadN + 1;
                        gpuPerf = gpuTimer.timing( [&](){
                                co_to_co_mlt_add_align<<<blockN, threadN>>>(c_q_gpu, c_d_gpu, calc_number, new_byte_per_op_unit, ans_gpu);
                        });
                        cout<<"CompactToCompactAlign --> mlt_add :"<<gpuPerf<<"ms"<<endl;
                        fp = fopen("../Varify/CompactToCompactAlign_mlt_add.txt", "ab+");
                }
        }

        if(type == 3 || type == 4){
                // 计算 l_quantity * l_discount 列 它们的结果类型 ans.prec = 24 ans.frac = 4
                int byte_per_ans_unit = GetDecimalRealBytes(34, 4);
                ans_sum_byte = byte_per_ans_unit * calc_number;

                // 申请结果在 CPU 和 GPU 上的空间
                ans_cpu = (char *)malloc(ans_sum_byte);
                cudaMalloc((void **)&ans_gpu, ans_sum_byte);

                if(type == 3){
                        // 调用 kernel
                        int threadN = 256;
                        size_t blockN = (calc_number - 1)/threadN + 1;
                        gpuPerf = gpuTimer.timing( [&](){
                                co_to_co_sig_mul_align<<<blockN, threadN>>>(c_q_gpu, c_d_gpu, calc_number, new_byte_per_op_unit, ans_gpu);
                        });
                        cout<<"CompactToCompactAlign --> sig_mul :"<<gpuPerf<<"ms"<<endl;
                        fp = fopen("../Varify/CompactToCompactAlign_sig_mul.txt", "ab+");
                }
                else{
                        // 调用 kernel
                        int threadN = 256;
                        size_t blockN = (calc_number*TPI - 1)/threadN + 1;
                        gpuPerf = gpuTimer.timing( [&](){
                                co_to_co_mlt_mul_align<<<blockN, threadN>>>(c_q_gpu, c_d_gpu, calc_number, new_byte_per_op_unit, ans_gpu);
                        });
                        cout<<"CompactToCompactAlign --> mlt_mul :"<<gpuPerf<<"ms"<<endl;
                        fp = fopen("../Varify/CompactToCompactAlign_mlt_mul.txt", "ab+");
                }
        }

        // 将数据拷回到 CPU 上并输入到文件中
        cudaMemcpy(ans_cpu, ans_gpu, ans_sum_byte, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // // compact 输出
        // fseek(fp, 0, SEEK_SET);
        // fwrite(ans_cpu ,1 , ans_sum_byte, fp);

        // decimal 输出
        aries_acc::Decimal *ans_dec_cpu;
        ans_dec_cpu = (aries_acc::Decimal *)malloc(calc_number * sizeof(aries_acc::Decimal));
        for(size_t i=0; i<calc_number; i++){
                if(type == 1 || type == 2 ){
                        int byte_per_ans_unit = GetDecimalRealBytes(18, 2); 
                        ans_dec_cpu[i] = aries_acc::Decimal((CompactDecimal *)(ans_cpu+i*byte_per_ans_unit), 18, 2);
                }
                else{
                        int byte_per_ans_unit = GetDecimalRealBytes(37, 4); 
                        ans_dec_cpu[i] = aries_acc::Decimal((CompactDecimal *)(ans_cpu+i*byte_per_ans_unit), 37, 4);
                }       
        }
        for(int i=0; i<calc_number; i++){
                char result[2048];
                for(int j=INDEX_LAST_DIG ; j>=0 ;j--){
                        sprintf(result+(INDEX_LAST_DIG-j)*8,"%08x",ans_dec_cpu[i].v[j]);
                }
                // sprintf(result+NUM_TOTAL_DIG*8," sign = %d , frac = %d",ans_dec_cpu[i].sign,ans_dec_cpu[i].frac);
                int len = strlen(result);
                fwrite(result,len,1,fp);
                fwrite("\r\n",1,2,fp);
        }

        fclose(fp);

        free(ans_cpu);
        cudaFree(ans_gpu);
}

TEST(CompactToCompactAlign, sig_add)
{
        main_execute_co_to_co_align(1);
}

TEST(CompactToCompactAlign, mlt_add)
{
        main_execute_co_to_co_align(2);
}

// TEST(CompactToCompact, sig_mul)
// {
//         main_execute_co_to_co(3);
// }

// TEST(CompactToCompact, mlt_mul)
// {
//         main_execute_co_to_co(4);
// }