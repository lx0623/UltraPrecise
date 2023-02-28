#pragma once

#include <vector>

#include "AriesOpNode.h"
#include "AriesCommonExpr.h"
#include "AriesExprCalcNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesUpdateCalcNode: public AriesOpNode {

public:
    AriesUpdateCalcNode();
    ~AriesUpdateCalcNode();

    void SetCalcExprs( const std::vector< AriesCommonExprUPtr >& exprs );
    void SetColumnIds( const std::vector< int >& columnIds );

    virtual void SetCuModule( const vector< CUmoduleSPtr >& modules ) override
    {
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
        for( const auto& expr : m_exprs )
        {
            if ( expr )
            {
                expr->SetCuModule( modules );
            }
        }
        m_dataSource->SetCuModule( modules );
    }

    virtual string GetCudaKernelCode() const override
    {
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
        std::string kernel_code;
        for( const auto& expr : m_exprs )
        {
            if ( expr )
            {
                kernel_code += expr->GetCudaKernelCode();
            }
        }
        return m_dataSource->GetCudaKernelCode() + kernel_code;
    }

public:
    bool Open() override final;
    AriesOpResult GetNext() override final;
    void Close() override final;
    JSON GetProfile() const override final;

private:
    std::vector< AEExprNodeUPtr > m_exprs;
    std::vector< AriesCommonExprUPtr > m_originExprs;
    std::vector< int > m_columnIds;
};
using AriesUpdateCalcNodeSPtr = shared_ptr< AriesUpdateCalcNode >;

END_ARIES_ENGINE_NAMESPACE
