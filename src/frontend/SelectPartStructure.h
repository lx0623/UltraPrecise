#ifndef ARIES_SELECT_PART_STRUCTURE
#define ARIES_SELECT_PART_STRUCTURE


#include <iostream>
#include <memory>
#include <vector>

#include "AbstractBiaodashi.h"
#include "server/mysql/include/mysqld.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"


namespace aries {

/***************************************************************************
 *
 *                          SelectPartStructure
 *
 **************************************************************************/



class SelectPartStructure {

private:

    bool have_distinct = false;
    BiaodashiPointer distinct_expr = nullptr;


    size_t select_expr_count = 0;
    std::vector<BiaodashiPointer> select_expr_array;
    std::vector<std::shared_ptr<std::string>> select_alias_array;

    SelectPartStructure(const SelectPartStructure &arg);

    SelectPartStructure &operator=(const SelectPartStructure &arg);

    std::vector<BiaodashiPointer> checked_expr_array;
    std::vector<std::shared_ptr<std::string>> checked_alias_array;


public:


    SelectPartStructure();

    void SetDistinctExpr(BiaodashiPointer arg_de);

    void SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd );

    void AddSelectExpr(BiaodashiPointer arg_se, std::shared_ptr<std::string> arg_alias);

    void ResetSelectExprsAndClearAlias( std::vector<BiaodashiPointer>&& exprs );

    void ResetSelectExprsAndAlias(const std::vector<BiaodashiPointer>& exprs, const std::vector<std::shared_ptr<std::string>>& alias);

    size_t GetSelectItemCount();

    BiaodashiPointer GetSelectExpr(size_t index);

    std::shared_ptr<std::string> GetSelectAlias(size_t index);

    std::string GetName(size_t index);

    std::string ToString();

    bool DoIHaveDistinct();

    BiaodashiPointer GetDistinctExpr();

    void AddCheckedExpr(BiaodashiPointer arg_expr);

    void AddCheckedAlias(std::shared_ptr<std::string> arg_alias_p);

    void ChangeEverythingToNothing();


    void ResetCheckedExpr(std::vector<BiaodashiPointer> arg_value);

    void ResetCheckedExprAlias(std::vector<std::shared_ptr<std::string>> arg_value);


    std::vector<BiaodashiPointer> GetAllExprs();

    std::vector<std::shared_ptr<std::string>> GetALlAliasPointers();


    void SetTheOnlyExprAlias(std::string arg_value);

    int LocateExprInSelectList(BiaodashiPointer arg_expr);

    int GetAllExprCount();

    void ResetAliasAtIndex(std::shared_ptr<std::string> alias, size_t index);

};

typedef std::shared_ptr<SelectPartStructure> SelectPartStructurePointer;


}//namespace

#endif
