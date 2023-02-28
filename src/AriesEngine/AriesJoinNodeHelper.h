#pragma once

#include <boost/variant.hpp>

#include "CudaAcc/AriesSqlOperator.h"
#include "AriesDataDef.h"
#include "AriesAssert.h"
#include "AriesCalcTreeGenerator.h"
#include "frontend/SQLTreeNode.h"
#include "AriesColumnDataIterator.hxx"
#include "AriesOpNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesJoinNode;
class AriesJoinNodeHelper
{

private:
    AriesJoinNode *m_joinNode;
    AEExprNodeUPtr left_node_of_equal_condition;
    AEExprNodeUPtr right_node_of_equal_condition;

    AriesJoinType join_type;

    DynamicCodeParams dynamic_code_params;

    bool is_cartesian_product;
    JoinConditionConstraintType equal_condtion_constraint_type;

    std::vector< int > unique_columns_id;
    std::vector< AEExprNodeUPtr > hash_join_nodes;
    AEExprNodeUPtr other_condition_as_filter_node;

    std::vector< int > required_columns_id_in_left;
    std::vector< int > required_columns_id_in_right;
    int m_nodeId;
    bool m_isNotIn;
public:
    AriesJoinNodeHelper( AriesJoinNode *joinNode,
                         const AriesCommonExprUPtr& equal_condition,
                         const AriesCommonExprUPtr& other_conditionconst,
                         AriesJoinType type, int nodeId );
    bool IsCartesianProduct() const
    {
        return is_cartesian_product;
    }

    std::string GetDynamicCode() const;

    void SetCUModule( const vector< CUmoduleSPtr >& modules );

    void SetJoinEqualConditionConstraintType( const JoinConditionConstraintType& type )
    {
        equal_condtion_constraint_type = type;
    }

    void SetIsNotInFlag( bool isNotIn )
    {
        m_isNotIn = isNotIn;
    }

    bool IsNotIn() const { return m_isNotIn; }

    void InitColumnsIdInConditions( const std::vector< int >& columns_id );

    AriesJoinResult Process( const AriesTableBlockUPtr& left_table, const AriesTableBlockUPtr& right_table );

    AriesJoinResult ProcessWithMaterializedBuffer(
        const AriesTableBlockUPtr& left_table,
        const AriesTableBlockUPtr& right_table,
        const AriesDataBufferSPtr& left_buffer,
        const AriesDataBufferSPtr& right_buffer );

    AriesOpResult ProcessGracePartitioned( const AriesTableBlockUPtr& left_table, const AriesTableBlockUPtr& right_table );

    AriesJoinResult ProcessHashLeftJoin( const AriesHashTableUPtr& hash_table,
                                         const AriesTableBlockUPtr& left_table,
                                         const AriesIndicesArraySPtr& left_table_indices,
                                         const AriesTableBlockUPtr& right_table,
                                         const AriesIndicesArraySPtr& right_table_indices,
                                         int column_id,
                                         bool can_use_dict,
                                         bool left_as_hash );

    AriesJoinResult ProcessHashLeftJoin( const AriesHashTableMultiKeysUPtr& hash_table,
                                          const AriesTableBlockUPtr& left_table,
                                          const AriesIndicesArraySPtr& left_table_indices,
                                          const AriesTableBlockUPtr& right_table,
                                          const AriesIndicesArraySPtr& right_table_indices,
                                          const std::vector< int > column_ids,
                                          const std::vector< bool >& can_use_dict,
                                          bool left_as_hash );

    AriesJoinResult ProcessHashFullJoin( const AriesHashTableUPtr& hash_table,
                                         const AriesTableBlockUPtr& left_table,
                                         const AriesIndicesArraySPtr& left_table_indices,
                                         const AriesTableBlockUPtr& right_table,
                                         const AriesIndicesArraySPtr& right_table_indices,
                                         const int column_id,
                                         bool can_use_dict,
                                         bool needToSwap = false );

    AriesJoinResult ProcessHalfJoinLeftAsHash( const AriesHashTableUPtr& hash_table,
                                               const AriesTableBlockUPtr& left_table,
                                               const AriesIndicesArraySPtr& left_table_indices,
                                               const AriesTableBlockUPtr& right_table,
                                               int column_id,
                                               bool can_use_dict );

    AriesJoinResult ProcessHalfJoinRightAsHash( const AriesHashTableUPtr& hash_table,
                                                const AriesTableBlockUPtr& left_table,
                                                const AriesIndicesArraySPtr& left_table_indices,
                                                const AriesTableBlockUPtr& right_table,
                                                int column_id,
                                                bool can_use_dict );

    void HandleUniqueKeys( const std::vector< std::vector< int > >& left_unique_keys,
                           const std::vector< std::vector< int > >& right_unique_keys );
    const DynamicCodeParams& GetDynamicCodeParams() const
    {
        return dynamic_code_params;
    }

    const AEExprNodeUPtr& GetOtherCondtionAsFilterNode() const
    {
        return other_condition_as_filter_node;
    }

    const std::vector< int >& GetLeftRequiredColumnsId() const;
    const std::vector< int >& GetRightRequiredColumnsId() const;

    void PrepareForHashJoin( const HashJoinInfo& hash_join_info );

    /**
     * 在需要将 right join 转换为 left join 时调用该方法
     */
    void SwapRightJoinToLeft();

    size_t GetLeftHashJoinKernelUsage( const size_t left_row_count, const size_t right_row_count ) const;
    size_t GetFullHashJoinKernelUsage( const size_t left_row_count, const size_t right_row_count ) const;

    /**
     * @brief 物化 left 和 right 表中需要参与到 other condition 运算的 列
     */
    void MaterializeColumns( const AriesTableBlockUPtr& left, const AriesTableBlockUPtr& right );

private:
    void SortColumnsForJoin( const AriesTableBlockUPtr& left_table, const AriesTableBlockUPtr& right_table, AriesDataBufferSPtr& left_buffer,
            AriesInt32ArraySPtr& left_associated, AriesDataBufferSPtr& right_buffer, AriesInt32ArraySPtr& right_associated );

    void setOtherCondition( const AriesCommonExprUPtr& condition );
    void resetOtherCondition( const AriesCommonExprUPtr& condition );
    void getJoinDynamicInputs( AriesManagedArray< AriesColumnDataIterator >& columns,
                               vector< AriesColumnDataIteratorHelper >& columnHelpers,
                               const AriesTableBlockUPtr& left_table,
                               const AriesTableBlockUPtr& right_table ) const;

    size_t GetSortSemiAntiJoinPartitionCount(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );

    size_t GetSortInnerJoinPartitionCount(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff ) const;

    AriesOpResult SortSemiAntiJoinGracePartitioned(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );

    AriesOpResult SortInnerJoinGracePartitioned(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );
    size_t GetSortLeftJoinPartitionCount(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );
    AriesOpResult SortLeftJoinGracePartitioned(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );

    size_t GetSortFullJoinPartitionCount(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );

    AriesOpResult SortFullJoinGracePartitioned(
        const AriesTableBlockUPtr& leftTable,
        const AriesTableBlockUPtr& rightTable,
        const AriesDataBufferSPtr& leftColumnBuff,
        const AriesDataBufferSPtr& rightColumnBuff );

private:
    static void generateDynamicCode( int nodeId,
                                     const AriesCommonExprUPtr& other_condition,
                                     std::string& function_name,
                                     std::string& code,
                                     std::map< string, AriesCommonExprUPtr >& agg_functions,
                                     std::vector< AriesDynamicCodeParam >& params,
                                     std::vector< AriesDataBufferSPtr >& constValues,
                                     std::vector< AriesDynamicCodeComparator >& comparators,
                                     bool need_swap = false,
                                     bool is_cartesian = false );
};

END_ARIES_ENGINE_NAMESPACE
