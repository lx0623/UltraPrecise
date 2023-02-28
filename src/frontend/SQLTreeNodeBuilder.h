#ifndef ARIES_SQL_TREE_NODE_BUILDER
#define ARIES_SQL_TREE_NODE_BUILDER

#include "AbstractQuery.h"
#include "AbstractBiaodashi.h"
#include "BasicRel.h"
#include "SQLTreeNode.h"

namespace aries {
class SQLTreeNodeBuilder {

private:
    std::weak_ptr< AbstractQuery > query;

    SQLTreeNodeBuilder(const SQLTreeNodeBuilder &arg);

    SQLTreeNodeBuilder &operator=(const SQLTreeNodeBuilder &arg);

public:
    SQLTreeNodeBuilder(AbstractQueryPointer arg_query);

    SQLTreeNodePointer makeTreeNode_Table(BasicRelPointer arg_basic_rel);

    SQLTreeNodePointer makeTreeNode_BinaryJoin(JoinType arg_join_type, BiaodashiPointer arg_equal_expr, BiaodashiPointer arg_other_expr = nullptr, bool is_not_in = false );

    SQLTreeNodePointer makeTreeNode_Filter(BiaodashiPointer arg_expr);

    SQLTreeNodePointer makeTreeNode_Group();

    SQLTreeNodePointer makeTreeNode_Sort();

    SQLTreeNodePointer makeTreeNode_Column();

    SQLTreeNodePointer makeTreeNode_SetOperation(SetOperationType arg_set_type);

    SQLTreeNodePointer makeTreeNode_Limit(int64_t offset, int64_t size);

    SQLTreeNodePointer makeTreeNode_SelfJoin( int joinColumnId, CommonBiaodashiPtr mainFilter, const vector< HalfJoinInfo >& joinInfo );
    SQLTreeNodePointer makeTreeNode_StarJoin();

    SQLTreeNodePointer makeTreeNode_InnerJoin();
    SQLTreeNodePointer makeTreeNode_Exchange();
};


typedef std::shared_ptr<SQLTreeNodeBuilder> SQLTreeNodeBuilderPointer;
}

#endif
