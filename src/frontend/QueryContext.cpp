#include "ExprContext.h"
#include "QueryContext.h"

namespace aries {

QueryContext::QueryContext(QueryContextType arg_type,
                           int arg_query_level,
                           std::shared_ptr<AbstractQuery> arg_select_structure,
        //SelectStructure* arg_select_structure,
                           std::shared_ptr<QueryContext> arg_parent_context,
                           std::shared_ptr<ExprContext> arg_expr_context) {

    this->type = arg_type;
    this->query_level = arg_query_level;
    this->select_structure = arg_select_structure;
    this->parent_context = arg_parent_context;
    this->expr_context = arg_expr_context;


}

AbstractQueryPointer QueryContext::GetSelectStructure() const
{
    // assert( !select_structure.expired() );
    return select_structure.lock();
}

QueryContextPointer QueryContext::GetParent() const
{
    return parent_context.lock();
}

bool QueryContext::HasOuterColumn() const
{
    if( !outer_column_array.empty() )
        return true;
    for( auto& child : subquery_context_array )
    {
        if( !child->outer_column_array.empty() )
            return true;
    }
    return false;
}


}
