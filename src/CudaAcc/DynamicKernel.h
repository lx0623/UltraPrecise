#ifndef DYNAMIC_KERNEL_H_
#define DYNAMIC_KERNEL_H_
#include <vector>
#include <mutex>
#include <deque>
#include <map>
#include "nvrtc.h"
#include "AriesException.h"
#include "AriesEngineDef.h"
#include "datatypes/AriesSqlFunctions.hxx"

BEGIN_ARIES_ACC_NAMESPACE
using namespace std;

struct AriesColumnDataIterator;
using CUmoduleSPtr = std::shared_ptr<CUmodule>;

#define CU_SAFE_CALL(x)                                         \
      do {                                                            \
        CUresult result = x;                                          \
        if (result != CUDA_SUCCESS) {                                 \
          const char *msg;                                            \
          cuGetErrorName(result, &msg);                               \
          LOG(INFO) << "\nerror: " #x " failed with error "           \
                    << msg << '\n';                                   \
          throw cu_exception_t( result );                           \
        }                                                             \
      } while(0)

#define NVRTC_SAFE_CALL(x)                                        \
      do {                                                            \
        nvrtcResult result = x;                                       \
        if (result != NVRTC_SUCCESS) {                                \
          LOG(ERROR) << "\nerror: " #x " failed with error "          \
                    << nvrtcGetErrorString(result) << '\n';           \
          throw nvrtc_exception_t( result );                          \
        }                                                             \
      } while(0)

struct AriesCUModuleInfo
{
    map< string, string > FunctionKeyNameMapping;
    vector< CUmoduleSPtr > Modules;
};
using AriesCUModuleInfoSPtr = std::shared_ptr< AriesCUModuleInfo >;

class AriesDynamicKernelManager
{
public:
    static AriesDynamicKernelManager &GetInstance()
    {
        static AriesDynamicKernelManager instance;
        return instance;
    }

    AriesCUModuleInfoSPtr FindFunction( const string& functionKey ) const;

    AriesCUModuleInfoSPtr CompileKernels( const AriesDynamicCodeInfo& code );

    AriesCUModuleInfoSPtr FindModule( const string& code );

    // filter
    void CallKernel( const vector< CUmoduleSPtr >& modules,
                     const char *functionName,
                     const AriesColumnDataIterator *input,
                     int tupleNum,
                     const int8_t** constValues,
                     const vector< AriesDynamicCodeComparator >& items,
                     int8_t *output ) const;
    // join
    void CallKernel( const vector< CUmoduleSPtr >& modules,
                     const char* functionName,
                     const AriesColumnDataIterator *input,
                     const index_t *leftIndices,
                     const index_t *rightIndices,
                     int tupleNum,
                     const int8_t** constValues,
                     const vector< AriesDynamicCodeComparator >& items,
                     int8_t *output ) const;

    // 迪卡尔集
    void CallKernel( const vector< CUmoduleSPtr >& modules,
                     const char* functionName,
                     const AriesColumnDataIterator *input,
                     size_t leftCount,
                     size_t rightCount,
                     size_t tupleNum,
                     int* left_unmatched_flag,
                     int* right_unmatched_flag,
                     const int8_t** constValues,
                     const vector< AriesDynamicCodeComparator >& items,
                     int *left_output,
                     int *right_output,
                     unsigned long long int* output_count ) const;

    // self join
    void CallKernel( const vector< CUmoduleSPtr >& modules, const char* functionName, const AriesColumnDataIterator *input,
            const int32_t* associated, const int32_t* groups, const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum, const int8_t** constValues,
            const vector< AriesDynamicCodeComparator >& items, int8_t *output ) const;
private:
    AriesDynamicKernelManager(){}
    ~AriesDynamicKernelManager(){}
    AriesDynamicKernelManager( const AriesDynamicKernelManager& );
    AriesDynamicKernelManager &operator=( const AriesDynamicKernelManager& src );

    void DestroyComparators( AriesArraySPtr< CallableComparator* > comparators ) const;
    AriesArraySPtr< CallableComparator* > CreateInComparators( CUmoduleSPtr module, const vector< AriesDynamicCodeComparator >& items ) const;
    void SetModuleAccessed( const string& code );
    void AddModule( const string& code, AriesCUModuleInfoSPtr moduleInfo );
    void RemoveOldModulesIfNecessary();

private:
    mutex m_mutex;
    deque< string > m_moduleCodesLru;
    map< string, AriesCUModuleInfoSPtr > m_modules;
    static int const LRU_COUNT = 100;
};

END_ARIES_ACC_NAMESPACE

#endif
