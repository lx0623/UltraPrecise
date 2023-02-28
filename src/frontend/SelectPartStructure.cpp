#include "SelectPartStructure.h"
#include "CommonBiaodashi.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"

namespace aries {
SelectPartStructure::SelectPartStructure() {
}

void SelectPartStructure::SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd )
{
    size_t exprCount = select_expr_array.size();
    for ( size_t i = 0; i < exprCount; ++i )
    {
        auto commonExpr = ( CommonBiaodashi* )select_expr_array[ i ].get();
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            select_expr_array[ i ] = simplifiedExpr;
    }

    if ( distinct_expr )
    {
        auto commonExpr = ( CommonBiaodashi* )distinct_expr.get();
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            distinct_expr = simplifiedExpr;
    }
}
void SelectPartStructure::SetDistinctExpr(BiaodashiPointer arg_de) {
    this->have_distinct = true;
    this->distinct_expr = (arg_de);
}

bool SelectPartStructure::DoIHaveDistinct() {
    return this->have_distinct;
}

BiaodashiPointer SelectPartStructure::GetDistinctExpr() {
    return this->distinct_expr;
}

void SelectPartStructure::AddSelectExpr(BiaodashiPointer arg_se, std::shared_ptr<std::string> arg_alias) {
    this->select_expr_count += 1;
    this->select_expr_array.push_back((arg_se));
    this->select_alias_array.push_back((arg_alias));
}

void SelectPartStructure::ResetSelectExprsAndAlias(const std::vector<BiaodashiPointer>& exprs, const std::vector<std::shared_ptr<std::string>>& alias) {
    assert(exprs.size() == alias.size());

    select_expr_count = exprs.size();
    select_expr_array = exprs;
    select_alias_array = alias;
}

void SelectPartStructure::ResetSelectExprsAndClearAlias( std::vector<BiaodashiPointer>&& exprs )
{
    select_expr_count = exprs.size();
    select_expr_array = std::move( exprs );
    select_alias_array.clear();
}

//unavailable after query plan. since they use checked_expr_array
size_t SelectPartStructure::GetSelectItemCount() {
    return this->select_expr_count;
}

BiaodashiPointer SelectPartStructure::GetSelectExpr(size_t index) {
    assert(index < this->select_expr_array.size());
    return this->select_expr_array[index];
}

std::shared_ptr<std::string> SelectPartStructure::GetSelectAlias(size_t index) {
    if( select_alias_array.empty() )
        return nullptr;
    assert(index < this->select_alias_array.size());
    return this->select_alias_array[index];
}

std::string SelectPartStructure::GetName(size_t index) {
    BiaodashiPointer abp = GetSelectExpr(index);
    CommonBiaodashi *rawpointer = (CommonBiaodashi *) abp.get();
    return rawpointer->GetOrigName();
}

std::string SelectPartStructure::ToString() {
    std::string result = "";


    /*after check, we should use checked_array*/

    if (this->checked_expr_array.size() >= 1) {
        if (this->have_distinct) {
            result += "DISTINCT ";
            assert(this->distinct_expr != nullptr);
            result += this->distinct_expr->ToString();
        }

        result += " ";

        for (size_t i = 0; i < this->checked_expr_array.size(); i++) {
            result += "(";
            result += this->checked_expr_array[i]->ToString();
            result += ")";

            if (this->checked_alias_array[i] != nullptr) {
                result += " AS ";
                /*here we don't use the shared_ptr*/
                result += *(this->checked_alias_array[i].get());
            }

            result += "\n, ";
        }

    } else {

        if (this->have_distinct) {
            result += "DISTINCT ";
            assert(this->distinct_expr != nullptr);
            result += this->distinct_expr->ToString();
        }


        result += " ";

        for (size_t i = 0; i < this->select_expr_count; i++) {
            result += "(";
            result += this->select_expr_array[i]->ToString();
            result += ")";
            if( !select_alias_array.empty() ) {
                assert( select_alias_array.size() == select_expr_count );
                if (this->select_alias_array[i] != nullptr) {
                    result += " AS ";
                    /*here we don't use the shared_ptr*/
                    result += *(this->select_alias_array[i].get());
                }
            }
            result += "\n, ";
        }

    }
    return result;
}


void SelectPartStructure::AddCheckedExpr(BiaodashiPointer arg_expr) {
    this->checked_expr_array.push_back(arg_expr);
}

void SelectPartStructure::AddCheckedAlias(std::shared_ptr<std::string> arg_alias_p) {
    this->checked_alias_array.push_back(arg_alias_p);
}

/*for exist (select ... )!*/
void SelectPartStructure::ChangeEverythingToNothing() {
    this->have_distinct = false;
    this->distinct_expr = nullptr;
    this->select_expr_count = 0;
    this->select_expr_array.clear();
    this->select_alias_array.clear();

    auto new_expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, 0);
    new_expr->SetValueType(BiaodashiValueType::INT);
    this->AddSelectExpr(new_expr, nullptr);
}

void SelectPartStructure::ResetCheckedExpr(std::vector<BiaodashiPointer> arg_value) {
    this->checked_expr_array = arg_value;
}

void SelectPartStructure::ResetCheckedExprAlias(std::vector<std::shared_ptr<std::string>> arg_value) {
    this->checked_alias_array = arg_value;
}


std::vector<BiaodashiPointer> SelectPartStructure::GetAllExprs() {
    return this->checked_expr_array;
}

std::vector<std::shared_ptr<std::string>> SelectPartStructure::GetALlAliasPointers() {
    return this->checked_alias_array;
}

void SelectPartStructure::SetTheOnlyExprAlias(std::string arg_value) {
    assert(this->select_alias_array.size() == 1);
    this->select_alias_array[0] = std::make_shared<std::string>(arg_value);

    assert(this->checked_alias_array.size() == 1);
    this->checked_alias_array[0] = std::make_shared<std::string>(arg_value);
}


int SelectPartStructure::LocateExprInSelectList(BiaodashiPointer arg_expr) {
    int ret = -1;

    CommonBiaodashi *expr_left = (CommonBiaodashi *) (arg_expr.get());

    for (size_t i = 0; i < this->checked_expr_array.size(); i++) {

        CommonBiaodashi *expr_right = (CommonBiaodashi *) ((this->checked_expr_array[i]).get());

        if (CommonBiaodashi::__compareTwoExprs(expr_left, expr_right) == true) {
            ret = i;
            break;
        }
    }

    return ret;
}


int SelectPartStructure::GetAllExprCount() {
    return this->checked_expr_array.size();
}

void SelectPartStructure::ResetAliasAtIndex(std::shared_ptr<std::string> alias, size_t index) {
    select_alias_array[index] = alias;
}

}
