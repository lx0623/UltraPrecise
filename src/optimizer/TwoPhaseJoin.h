#ifndef ARIES_TWO_PHASE_JOIN
#define ARIES_TWO_PHASE_JOIN


#include "QueryOptimizationPolicy.h"

namespace aries {

class TwoPhaseJoin : public QueryOptimizationPolicy {
public:

    TwoPhaseJoin();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;

    void twophasejoin_single_query(SQLTreeNodePointer arg_input);

    void twophasejoin_handling_node(SQLTreeNodePointer arg_input);


    void HandlingBinaryJoinNode(SQLTreeNodePointer arg_input);

    bool CheckEqualJoinCondition(BiaodashiPointer arg_expr, SQLTreeNodePointer arg_input);

private:
    bool CheckTablesInTables( std::vector<BasicRelPointer> ref, std::vector<BasicRelPointer> tables );
    void handleInnerJoinNode( SQLTreeNodePointer node );

};

}//namespace


#endif
