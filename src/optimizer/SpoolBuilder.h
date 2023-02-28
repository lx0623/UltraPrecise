#pragma once

#include "frontend/SelectStructure.h"

#include "QueryOptimizationPolicy.h"

namespace aries
{

class SpoolBuilder : public QueryOptimizationPolicy
{
private:
    int next_spool_id;

public:
    SpoolBuilder();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree( SQLTreeNodePointer node ) override;

private:
    void handleSubQueries( const std::vector< AbstractQueryPointer >& sub_queries );
    int getNextSpoolId();
};

} // namespace aries