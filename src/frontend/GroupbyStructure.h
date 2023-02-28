#ifndef ARIES_GROUPBY_STRUCTURE
#define ARIES_GROUPBY_STRUCTURE

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>


#include "CommonBiaodashi.h"
#include "server/mysql/include/mysqld.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"


namespace aries {


/***************************************************************************
 *
 *                          GroupbyStructure
 *
 **************************************************************************/


class GroupbyStructure {
private:

    std::vector<BiaodashiPointer> groupby_exprs;

    bool having_having = false;

    BiaodashiPointer having_expr = nullptr;

    //to store those agg exprs in having but not in select
    std::vector<BiaodashiPointer> additional_exprs;

    //to store all agg exprs in having
    std::vector<BiaodashiPointer> agg_exprs_in_having;

    //to store the map-to-location map
    std::map<BiaodashiPointer, int> agg_exprs_location_map;

    std::vector<BiaodashiPointer> all_placeholder_exprs;

    GroupbyStructure(const GroupbyStructure &arg);

    GroupbyStructure &operator=(const GroupbyStructure &arg);


public:

    GroupbyStructure();

    void SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd );
    std::vector<BiaodashiPointer> GetGroupbyExprs();

    void AddGroupbyExpr(BiaodashiPointer arg_groupby_expr);

    void SetGroupbyExprs(std::vector<BiaodashiPointer> arg_vbp);

    void SetHavingExpr(BiaodashiPointer arg_having_expr);

    void DeleteHavingExpr();

    size_t GetGroupbyExprCount();

    BiaodashiPointer GetGroupbyExpr(size_t index);

    BiaodashiPointer GetHavingExpr();


    std::string ToString();

    void SetAdditionalExprsForSelect(std::vector<BiaodashiPointer> arg_vbp);

    std::vector<BiaodashiPointer> GetAdditionalExprsForSelect();

    void SetAggExprsInHaving(std::vector<BiaodashiPointer> arg_value);

    std::vector<BiaodashiPointer> GetAggExprsInHaving();

    void SetAggExprsLocationMap(std::map<BiaodashiPointer, int> arg_map);

    std::map<BiaodashiPointer, int> GetAggExprsLocationMap();

    void SetAllPlaceHolderExprs(std::vector<BiaodashiPointer> arg_value);

    std::vector<BiaodashiPointer> GetAllPlaceHolderExprs();

};


typedef std::shared_ptr<GroupbyStructure> GroupbyStructurePointer;

}//namespace

#endif
