#pragma once

#include "AriesOpNode.h"
#include "AriesCommonExpr.h"
#include "AriesCalcTreeGenerator.h"
#include "transaction/AriesTransaction.h"
#include "transaction/AriesMvccTable.h"

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesConstantNode : public AriesOpNode
{
public:
    AriesConstantNode( const std::string& dbName, const std::string& tableName );
    ~AriesConstantNode();

    void SetCuModule( const vector< CUmoduleSPtr >& modules ) override
    {
    }

    std::string GetCudaKernelCode() const override
    {
        return std::string();
    }

    int SetColumnData( const std::vector<std::vector<AriesCommonExprUPtr>> &data,
                       const std::vector<int> &columnIds,
                       string& errorMsg );

    virtual bool Open() override final;
    virtual AriesOpResult GetNext() override final;
    virtual void Close() override final;
    JSON GetProfile() const override final;

private:
    std::vector< AriesDataBufferSPtr > m_buffers;
    bool m_hasMoreData = false;
    std::string m_dbName;
    std::string m_tableName;
};

END_ARIES_ENGINE_NAMESPACE
