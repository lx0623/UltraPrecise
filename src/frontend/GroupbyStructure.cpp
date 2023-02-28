#include "GroupbyStructure.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"

namespace aries {

GroupbyStructure::GroupbyStructure() {
}

void GroupbyStructure::SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd )
{
    size_t exprCount = groupby_exprs.size();
    for ( size_t i = 0; i < exprCount; ++i )
    {
        auto commonExpr = ( CommonBiaodashi* )groupby_exprs[ i ].get();
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            groupby_exprs[ i ] = simplifiedExpr;
    }

    if ( having_expr )
    {
        auto commonExpr = ( CommonBiaodashi* )having_expr.get();
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            having_expr = simplifiedExpr;
    }
}
std::vector<BiaodashiPointer> GroupbyStructure::GetGroupbyExprs() {
    return this->groupby_exprs;
}

void GroupbyStructure::AddGroupbyExpr(BiaodashiPointer arg_groupby_expr) {
    this->groupby_exprs.push_back((arg_groupby_expr));
}

void GroupbyStructure::SetGroupbyExprs(std::vector<BiaodashiPointer> arg_vbp) {
    this->groupby_exprs = (arg_vbp);
}

void GroupbyStructure::SetHavingExpr(BiaodashiPointer arg_having_expr) {
    assert(arg_having_expr != nullptr);

    this->having_having = true;
    this->having_expr = (arg_having_expr);
}

void GroupbyStructure::DeleteHavingExpr() {
    assert(this->having_expr != nullptr);
    this->having_having = false;
    this->having_expr = nullptr;
}

std::string GroupbyStructure::ToString() {
    std::string result = "";
    for (size_t i = 0; i != this->groupby_exprs.size(); i++) {
        result += std::to_string(i) + ":   " + this->groupby_exprs[i]->ToString() + "   ,";
    }

    if (this->having_having) {
        result += " HAVING " + this->having_expr->ToString();
    }

    return result;
}


size_t GroupbyStructure::GetGroupbyExprCount() {
    return this->groupby_exprs.size();
}

BiaodashiPointer GroupbyStructure::GetGroupbyExpr(size_t index) {
    assert(index >= 0 && index < this->groupby_exprs.size());
    return this->groupby_exprs[index];
}

BiaodashiPointer GroupbyStructure::GetHavingExpr() {
    return this->having_expr;
}

void GroupbyStructure::SetAdditionalExprsForSelect(std::vector<BiaodashiPointer> arg_vbp) {
    this->additional_exprs = arg_vbp;
}

std::vector<BiaodashiPointer> GroupbyStructure::GetAdditionalExprsForSelect() {
    return this->additional_exprs;
}

void GroupbyStructure::SetAggExprsInHaving(std::vector<BiaodashiPointer> arg_value) {
    this->agg_exprs_in_having = arg_value;
}

std::vector<BiaodashiPointer> GroupbyStructure::GetAggExprsInHaving() {
    return this->agg_exprs_in_having;
}


void GroupbyStructure::SetAggExprsLocationMap(std::map<BiaodashiPointer, int> arg_map) {
    this->agg_exprs_location_map = arg_map;
}

std::map<BiaodashiPointer, int> GroupbyStructure::GetAggExprsLocationMap() {
    return this->agg_exprs_location_map;
}


void GroupbyStructure::SetAllPlaceHolderExprs(std::vector<BiaodashiPointer> arg_value) {
    this->all_placeholder_exprs = arg_value;
}

std::vector<BiaodashiPointer> GroupbyStructure::GetAllPlaceHolderExprs() {
    return this->all_placeholder_exprs;
}
}
