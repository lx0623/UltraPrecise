/*
 * AriesSqlOperator_common.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_COMMON_H_
#define ARIESSQLOPERATOR_COMMON_H_
#include <string>
#include <vector>
#include <nvrtc.h>
#include "AriesException.h"

#include "CudaAcc/AriesEngineDef.h"

using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE


    class AriesCommonExpr;
    using AriesCommonExprUPtr = std::unique_ptr<AriesCommonExpr>;
    class AriesTableBlock;
    using AriesTableBlockUPtr = std::unique_ptr< AriesTableBlock >;
    class AriesColumn;
    using AriesColumnSPtr = std::shared_ptr< AriesColumn >;
    class AriesColumnReference;
    using AriesColumnReferenceSPtr = std::shared_ptr< AriesColumnReference >;

    using AriesVariantIndicesArray = AriesDataBuffer;
    using AriesVariantIndicesArraySPtr = AriesDataBufferSPtr;
    using AriesVariantIndices = AriesColumn;
    using AriesVariantIndicesSPtr = AriesColumnSPtr;

END_ARIES_ENGINE_NAMESPACE

BEGIN_ARIES_ACC_NAMESPACE

    enum LOCALE_LANGUAGE
    : int32_t;
    enum interval_type
    : int32_t;
    enum SumStrategy : int32_t
    {
        NONE = 0,
        SOLE_TEMP_SUM,  //每个BLOCK都有自己的中间结果，完成后选thread 0加到最后结果中
        SHARE_TEMP_SUM,  //某几个BLOCK共用一个中间结果,所有block完成后再加到中间结果中
        NO_TEMP_SUM  //每个BLOCK都没有自己的中间结果，直接加到最后结果中
    };
    struct AriesInterval;
    struct AriesColumnDataIterator;
    using CUmoduleSPtr = std::shared_ptr<CUmodule>;

    class standard_context_t;
    class CallableComparator;
    class KernelComparator;
    struct JoinDynamicCodeParams
    {
        std::vector< std::shared_ptr< CUmodule > > cuModules;
        std::string functionName;
        const AriesColumnDataIterator *input;
        std::vector< AriesDataBufferSPtr > constValues;
        std::vector< AriesDynamicCodeComparator > comparators;
    };

#define CUDA_SAFE_CALL(x)                                         \
      do {                                                            \
        CUresult result = x;                                          \
        if (result != CUDA_SUCCESS) {                                 \
          const char *msg;                                            \
          cuGetErrorName(result, &msg);                               \
          LOG(INFO) << "\nerror: " #x " failed with error "           \
                    << msg << '\n';                                   \
          throw aries::cu_exception_t( result );                      \
        }                                                             \
      } while(0)

    class standard_context_t;
    class AriesSqlOperatorContext
    {
    public:
        static AriesSqlOperatorContext &GetInstance()
        {
            static AriesSqlOperatorContext instance;
            return instance;
        }

        std::shared_ptr< standard_context_t > GetContext() const
        {
            return m_ctx;
        }

    private:
        AriesSqlOperatorContext();
        ~AriesSqlOperatorContext();
        AriesSqlOperatorContext( const AriesSqlOperatorContext & );
        AriesSqlOperatorContext &operator=( const AriesSqlOperatorContext &src );

    private:
        std::shared_ptr< standard_context_t > m_ctx;
    };

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_COMMON_H_ */
