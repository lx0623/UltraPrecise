/*
 * AriesSqlOperator_common.cu
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#include "AriesSqlOperator_common.h"
#include "algorithm/context.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    AriesSqlOperatorContext::AriesSqlOperatorContext()
    {
        m_ctx = std::make_shared< standard_context_t >();
        CUDA_SAFE_CALL( cuInit( 0 ) );
    }

    AriesSqlOperatorContext::~AriesSqlOperatorContext()
    {
    }

END_ARIES_ACC_NAMESPACE
