/*
 * AriesColumnNode.h
 *
 *  Created on: Oct 11, 2018
 *      Author: lichi
 */

#pragma once

#include "AriesCalcTreeGenerator.h"
#include "AriesOpNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesColumnNode: public AriesOpNode
    {
    public:
        AriesColumnNode();
        ~AriesColumnNode();
        void SetColumnExprs( const vector< AriesCommonExprUPtr >& exprs );
        void SetExecutionMode( int mode );
        void SetOutputColumnIds( const vector< int >& columnIds );
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
        virtual AriesTableBlockUPtr GetEmptyTable() const override final;

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;

    private:
        AriesCalcTreeGenerator m_calcTreeGen;
        vector< AEExprNodeUPtr > m_rootOps;
        vector< int > m_outputColumnIds;
        vector< std::pair<AriesExprType, AriesExprContent> > m_LiteralContents;
        // 0: we need to output by the_select_exprs;
        // 1: we only do a forward by m_outputColumnIds;
        int m_mode;
    };

    using AriesColumnNodeSPtr = shared_ptr< AriesColumnNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

