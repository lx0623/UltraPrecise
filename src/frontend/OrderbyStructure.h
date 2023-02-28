#ifndef ARIES_ORDERBY_STRUCTURE
#define ARIES_ORDERBY_STRUCTURE

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "VariousEnum.h"

#include "AbstractBiaodashi.h"
#include "server/mysql/include/mysqld.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"


namespace aries {

/***************************************************************************
 *
 *                          OrderbyStructure
 *
 **************************************************************************/


class OrderbyStructure {

private:

    int orderby_expr_count = 0;
    std::vector<BiaodashiPointer> orderby_expr_array;

    std::vector<OrderbyDirection> orderby_direction_array;

    OrderbyStructure(const OrderbyStructure &arg);

    OrderbyStructure &operator=(const OrderbyStructure &arg);


public:


    OrderbyStructure();

    void AddOrderbyItem(BiaodashiPointer arg_oe, OrderbyDirection arg_direction);

    std::string ToString();

    size_t GetOrderbyItemCount();

    void SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd );
    BiaodashiPointer GetOrderbyItem(size_t index);

    void SetOrderbyItem( BiaodashiPointer expr, size_t index );

    OrderbyDirection GetOrderbyDirection(size_t index);

    const std::vector<BiaodashiPointer>& GetOrderbyExprs();
};

typedef std::shared_ptr<OrderbyStructure> OrderbyStructurePointer;


}

#endif
