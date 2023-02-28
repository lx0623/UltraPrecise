#ifndef ARIES_EXPR_CONTEXT
#define ARIES_EXPR_CONTEXT

#include <memory>
#include <vector>

#include "AbstractBiaodashi.h"
#include "VariousEnum.h"


namespace aries {

/*------------------------------------------------------------
 *------------------------------------------------------------
 *-----------------------------------------------------------*/

class ColumnShell;

class QueryContext;

class ExprContext {

private:
    ExprContext(const ExprContext &arg);

    ExprContext &operator=(const ExprContext &arg);

    std::weak_ptr< ExprContext > parent_context;

    std::weak_ptr< QueryContext > query_context;

public:

    ExprContextType type;

    //std::shared_ptr<AbstractBiaodashi> expr;
    std::weak_ptr< AbstractBiaodashi > expr;

    int index; /*which table_array is mine? -- for join on condition!*/



    bool query_one_row_requirement = false; /*for subquery, Can it return only one row or multiple rows?*/

    int check_serial = 0;


    std::vector<std::shared_ptr<ColumnShell>> referenced_column_array; /*all columns so far found*/
    std::vector<bool> referenced_column_agg_status_array; /*am i in agg*/

    bool see_agg_func = false;


    /*boundary for exists*/
    bool exist_begin_mark = false;
    bool exist_end_mark = false;

    bool not_orginal_group_expr = false;
    bool not_orginal_select_expr = false;

    std::shared_ptr< ExprContext > GetParent() const;
    std::shared_ptr< QueryContext > GetQueryContext() const;

    ExprContext(ExprContextType arg_type,
                BiaodashiPointer arg_expr,
                std::shared_ptr<QueryContext> arg_query_context,
                std::shared_ptr<ExprContext> arg_parent_context,
                int index);

};


typedef std::shared_ptr<ExprContext> ExprContextPointer;

}//namespace

#endif
