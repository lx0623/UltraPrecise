/*
 * AriesSortNode.h
 *
 *  Created on: Sep 26, 2018
 *      Author: lichi
 */

#pragma once
#include <vector>
#include "AriesOpNode.h"
#include "AriesCalcTreeGenerator.h"
#include "AriesColumnType.h"
#include "AriesCommonExpr.h"
#include "CudaAcc/AriesEngineException.h"

using std::vector;

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesSortNode: public AriesOpNode
    {
    public:
        AriesSortNode();
        ~AriesSortNode();
        void SetOutputColumnIds( const vector< int >& columnIds );
        void SetOrderbyExprs( const vector< AriesCommonExprUPtr >& exprs );
        void SetOrderbyType( const vector< AriesOrderByType >& orders );
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules )
        {
            ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr");
            for( const auto& expr : m_exprs )
            {
                expr->SetCuModule( modules );
            }
            m_leftSource->SetCuModule( modules );
        }
        virtual string GetCudaKernelCode() const
        {
            ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr");
            std::string kernel_code;
            for( const auto& expr : m_exprs )
            {
                kernel_code += expr->GetCudaKernelCode();
            }
            return m_leftSource->GetCudaKernelCode() + kernel_code;
        }

        virtual AriesTableBlockUPtr GetEmptyTable() const override final;

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;
        // string GetProfile() const override final;

    private:
        void UpdateOutColumns(AriesTableBlockUPtr &table);
    private:
        vector< AEExprNodeUPtr > m_exprs;
        vector< AriesOrderByType > m_orders;
        vector< int > m_outputColumnIds;
    };

    using AriesSortNodeSPtr = shared_ptr< AriesSortNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
