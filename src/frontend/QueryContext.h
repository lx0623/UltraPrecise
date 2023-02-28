#ifndef ARIES_QUERY_CONTEXT
#define ARIES_QUERY_CONTEXT

#include <memory>
#include <vector>

#include "AbstractQuery.h"

#include "VariousEnum.h"

#include "ColumnShell.h"

namespace aries {

class SelectStructure; //forward declaration
class ExprContext; //forward declaration


class QueryContext : public std::enable_shared_from_this<QueryContext> {


private:
    QueryContext(const QueryContext &arg);

    QueryContext &operator=(const QueryContext &arg);

    /*the query within this context*/
    std::weak_ptr< AbstractQuery > select_structure;

    /*parent_query*/
    std::weak_ptr< QueryContext > parent_context;

public:

    bool HasOuterColumn() const;

    QueryContextType type;
    //SelectStructure* select_structure; //Fuck smart pointer!! Should we use raw pointer? reference? or fucking boost::enable_shared_from_this? FUCK!!!


    /*the expr where I am */
    std::shared_ptr<ExprContext> expr_context;

    int query_level;

    std::vector<std::shared_ptr<ColumnShell>> outer_column_array;
    std::vector<bool> outer_column_agg_status_array;


    std::vector<std::shared_ptr<ColumnShell>> unsolved_ID_array; /*this is hold those names in groupby/orderby -- but actually they are select alias*/

    std::vector<std::shared_ptr<QueryContext>> subquery_context_array; /*all the subqueries of mine -- just one layer - no deeper!*/

    AbstractQueryPointer GetSelectStructure() const;
    std::shared_ptr< QueryContext > GetParent() const;

    QueryContext(QueryContextType arg_type,
                 int arg_query_level,
                 std::shared_ptr<AbstractQuery> arg_select_structure,
            //SelectStructure* arg_select_structure,
                 std::shared_ptr<QueryContext> arg_parent_context,
                 std::shared_ptr<ExprContext> arg_expr_context);

};

typedef std::shared_ptr<QueryContext> QueryContextPointer;

}//namespace

#endif
