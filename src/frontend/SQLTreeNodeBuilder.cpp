#include "SQLTreeNodeBuilder.h"

namespace aries {
SQLTreeNodeBuilder::SQLTreeNodeBuilder(AbstractQueryPointer arg_query) {
    this->query = arg_query;
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Table(BasicRelPointer arg_basic_rel) {
    return SQLTreeNode::makeTreeNode_Table(query.lock(), arg_basic_rel);
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_BinaryJoin(JoinType arg_join_type, BiaodashiPointer arg_equal_expr, BiaodashiPointer arg_other_expr, bool is_not_in) {
    return SQLTreeNode::makeTreeNode_BinaryJoin(query.lock(), arg_join_type, arg_equal_expr, arg_other_expr, is_not_in );
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Filter(BiaodashiPointer arg_expr) {
    return SQLTreeNode::makeTreeNode_Filter(query.lock(), arg_expr);
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Group() {
    return SQLTreeNode::makeTreeNode_Group(query.lock());
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Sort() {
    return SQLTreeNode::makeTreeNode_Sort(query.lock());
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Column() {
    return SQLTreeNode::makeTreeNode_Column(query.lock());
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_SetOperation(SetOperationType arg_set_type) {
    return SQLTreeNode::makeTreeNode_SetOperation(query.lock(), arg_set_type);
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Limit(int64_t offset, int64_t size) {
    return SQLTreeNode::makeTreeNode_Limit(query.lock(), offset, size);
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_SelfJoin( int joinColumnId, CommonBiaodashiPtr mainFilter, const vector< HalfJoinInfo >& joinInfo )
{
    return SQLTreeNode::makeTreeNode_SelfJoin(query.lock(), joinColumnId, mainFilter, joinInfo );
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_StarJoin()
{
    return SQLTreeNode::makeTreeNode_StarJoin(query.lock() );
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_InnerJoin()
{
    return SQLTreeNode::makeTreeNode_InnerJoin(query.lock() );
}

SQLTreeNodePointer SQLTreeNodeBuilder::makeTreeNode_Exchange()
{
    return SQLTreeNode::makeTreeNode_Exchange(query.lock() );
}

}
