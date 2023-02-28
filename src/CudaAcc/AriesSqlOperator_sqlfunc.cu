#include "AriesSqlOperator_sqlfunc.h"
#include "AriesEngineAlgorithm.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

    AriesBoolArraySPtr IsNull( const AriesDataBufferSPtr& column )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        return is_null( column->GetData(), column->GetItemCount(), column->GetDataType(), *ctx );
    }

    AriesBoolArraySPtr IsNotNull( const AriesDataBufferSPtr& column )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        return is_not_null( column->GetData(), column->GetItemCount(), column->GetDataType(), *ctx );
    }

END_ARIES_ACC_NAMESPACE
