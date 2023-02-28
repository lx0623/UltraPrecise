#ifndef ARIES_QUERY_OPTIMIZER
#define ARIES_QUERY_OPTIMIZER

#include <vector>
#include <memory>

#include "frontend/SQLTreeNode.h"
#include "QueryOptimizationPolicy.h"

namespace aries {

class QueryOptimizer {

private:

    QueryOptimizer(const QueryOptimizer &arg);

    QueryOptimizer &operator=(const QueryOptimizer &arg);


    std::vector<QueryOptimizationPolicyPointer> policy_array;

public:

    QueryOptimizer();

    void RegisterPolicy(QueryOptimizationPolicyPointer arg_policy);

    SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input);

    static std::shared_ptr<QueryOptimizer> GetQueryOptimizer();
};

typedef std::shared_ptr<QueryOptimizer> QueryOptimizerPointer;

}
#endif
