#ifndef ARIES_QUERY_OPTIMIZATION_PARTITION_CONDITION
#define ARIES_QUERY_OPTIMIZATION_PARTITION_CONDITION

#include "QueryOptimizationPolicy.h"
#include "schema/TablePartition.h"

namespace aries {

class PartitionConditionOptimizer : public QueryOptimizationPolicy
{
public:
    struct PartitionInfo
    {
        int32_t ColumnIndex;
        std::string ColumnName;
        std::vector< schema::TablePartitionSPtr > Partions;
    };

public:
    PartitionConditionOptimizer();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;

private:
    void processNode( const SQLTreeNodePointer& node );
    void processFilterNode( const SQLTreeNodePointer& node );

    /**
     * @brief 返回 table 的 partition 信息
     * @return < columnIndex, columnName >
     */
    PartitionInfo getTablePartitionColumn( const SQLTreeNodePointer& node );

};


}
#endif
