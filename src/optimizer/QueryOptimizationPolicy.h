#ifndef ARIES_QUERY_OPTIMIZATION_POLICY
#define ARIES_QUERY_OPTIMIZATION_POLICY

#include <string>

#include "frontend/SQLTreeNode.h"

namespace aries {

class QueryOptimizationPolicy {
public:
    virtual std::string ToString() = 0;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) = 0;

};

typedef std::shared_ptr<QueryOptimizationPolicy> QueryOptimizationPolicyPointer;

}
#endif


