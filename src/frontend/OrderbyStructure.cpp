#include "OrderbyStructure.h"
#include "CommonBiaodashi.h"

namespace aries {

OrderbyStructure::OrderbyStructure() {
}

void OrderbyStructure::AddOrderbyItem(BiaodashiPointer arg_oe, OrderbyDirection arg_direction) {
    this->orderby_expr_count += 1;
    this->orderby_expr_array.push_back((arg_oe));
    this->orderby_direction_array.push_back(arg_direction);
}

size_t OrderbyStructure::GetOrderbyItemCount() {
    return this->orderby_expr_array.size();
}

void OrderbyStructure::SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd )
{
    size_t exprCount = orderby_expr_array.size();
    for ( size_t i = 0; i < exprCount; ++i )
    {
        auto commonExpr = ( CommonBiaodashi* )orderby_expr_array[ i ].get();
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            orderby_expr_array[ i ] = simplifiedExpr;
    }
}

BiaodashiPointer OrderbyStructure::GetOrderbyItem(size_t index) {
    assert(index >= 0 && index < this->orderby_expr_array.size());
    return this->orderby_expr_array[index];
}

void OrderbyStructure::SetOrderbyItem( BiaodashiPointer expr, size_t index )
{
    assert(index >= 0 && index < this->orderby_expr_array.size());
    this->orderby_expr_array[index] = expr;
}

OrderbyDirection OrderbyStructure::GetOrderbyDirection(size_t index) {
    assert(index >= 0 && index < this->orderby_expr_array.size());
    return this->orderby_direction_array[index];
}

std::string OrderbyStructure::ToString() {
    std::string result = "";

    for (int i = 0; i < this->orderby_expr_count; i++) {
        result += "(";
        result += this->orderby_expr_array[i]->ToString();
        result += ")";

        if (this->orderby_direction_array[i] == OrderbyDirection::ASC) {
            result += " ASC";
        } else {
            result += " DESC";
        }

        result += ", ";
    }

    return result;
}

const std::vector<BiaodashiPointer>& OrderbyStructure::GetOrderbyExprs() {
    return orderby_expr_array;
}

}
