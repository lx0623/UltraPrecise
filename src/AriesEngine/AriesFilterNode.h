//
// Created by tengjp on 19-11-1.
//

#ifndef AIRES_ARIESFILTERNODEV2_H
#define AIRES_ARIESFILTERNODEV2_H

#include "CudaAcc/AriesSqlOperator.h"
#include "AriesDataDef.h"
#include "AriesAssert.h"
#include "AriesOpNode.h"
#include "AriesCommonExpr.h"
#include "AriesCalcTreeGenerator.h"
#include <deque>
BEGIN_ARIES_ENGINE_NAMESPACE

class AriesFilterNode : public AriesOpNode {
public:
    AriesFilterNode();
    ~AriesFilterNode();

    void SetCondition( const AriesCommonExprUPtr& condition );
    void SetOutputColumnIds( const vector< int >& columnIds );
    void SetCuModule( const vector< CUmoduleSPtr >& modules );
    string GetCudaKernelCode() const;

    bool Open() override final;
    AriesOpResult GetNext() override final;
    void Close() override final;

    virtual AriesTableBlockUPtr GetEmptyTable() const override final;

private:
    AriesOpResult ReadAllData();

private:
    AriesCalcTreeGenerator m_calcTreeGen;
    vector< int > m_outputColumnIds;
    AEExprNodeUPtr m_rootOp;
    deque< AriesOpResult > m_outputTables;
    deque< AriesTableBlockUPtr > m_inputTables;
    bool m_bAllDataReceived;
};

using AriesFilterNodeSPtr = std::shared_ptr<AriesFilterNode>;

END_ARIES_ENGINE_NAMESPACE
#endif //AIRES_ARIESFILTERNODEV2_H
