//
// Created by 胡胜刚 on 2019-07-23.
//

#pragma once

#include "AriesEngine/AriesCommonExpr.h"
#include "frontend/CommonBiaodashi.h"
#include "datatypes/AriesTime.hxx"
#include "AriesEngine/transaction/AriesTransaction.h"
#include "ExpressionSimplifier.h"

using namespace aries_acc;
using namespace aries;

BEGIN_ARIES_ENGINE_NAMESPACE
class AriesExprBridge : protected DisableOtherConstructors {
public:
    AriesExprBridge();

    AriesCommonExprUPtr Bridge(const BiaodashiPointer &expression) const;

    std::vector<AriesOrderByType> ConvertToAriesOrderType(const std::vector<OrderbyDirection> &directions) const;

    AriesJoinType ConvertToAriesJoinType(JoinType type) const;

    AriesSetOpType ConvertToAriesSetOpType( SetOperationType type ) const;

private:
    AriesValueType convertValueType(BiaodashiValueType type) const;

    AriesAggFunctionType getAggFuncType(const string &arg_value) const;

    AriesCalculatorOpType getCalcType(CalcType arg_value) const;

    AriesComparisonOpType getCompType(ComparisonType arg_value) const;

    AriesLogicOpType getLogicType( LogicType arg_value ) const;

    void bridgeChildren(ExpressionSimplifier &exprSimplifier, AriesCommonExprUPtr& self, CommonBiaodashi* origin) const;

    AriesCommonExprUPtr
    BridgeDictEncodedColumnComparison( CommonBiaodashi* origin ) const;

    AriesCommonExprUPtr
    BridgeDictEncodedColumnInOp( CommonBiaodashi* origin, AriesExprType op ) const;

private:
    string m_defaultDb;
    THD* m_thd;
    aries_engine::AriesTransactionPtr m_tx;
};

END_ARIES_ENGINE_NAMESPACE // namespace aries
