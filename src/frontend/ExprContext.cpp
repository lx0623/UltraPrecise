#include "ExprContext.h"

namespace aries {


ExprContext::ExprContext(ExprContextType arg_type,
                         BiaodashiPointer arg_expr,
                         std::shared_ptr<QueryContext> arg_query_context,
                         std::shared_ptr<ExprContext> arg_parent_context,
                         int index) {

    this->type = arg_type;
    this->expr = arg_expr;
    this->query_context = arg_query_context;
    this->parent_context = arg_parent_context;
    this->index = index;

}

std::shared_ptr< ExprContext > ExprContext::GetParent() const
{
    return parent_context.lock();
}

std::shared_ptr< QueryContext > ExprContext::GetQueryContext() const
{
    return query_context.lock();
}

}
