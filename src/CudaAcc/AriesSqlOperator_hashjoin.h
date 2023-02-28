/*
 * AriesSqlOperator_hashjoin.h
 *
 *  Created on: Aug 31, 2020
 *      Author: lichi
 */

#ifndef ARIESSQLOPERATOR_HASHJOIN_H_
#define ARIESSQLOPERATOR_HASHJOIN_H_

#include "AriesSqlOperator_common.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesHashTableMultiKeysUPtr BuildHashTable( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id, const vector< bool >& can_use_dict );

    AriesHashTableUPtr BuildHashTable( const aries_engine::AriesTableBlockUPtr& table_block, int column_id, bool can_use_dict );

    void ReleaseHashTable( AriesHashTableMultiKeysUPtr& table );

    void ReleaseHashTable( AriesHashTableUPtr& table );

    JoinPair InnerJoinWithHash( const AriesHashTableUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
            const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, int column_id, bool can_use_dict );

    AriesJoinResult InnerJoinWithHash( const AriesHashTableMultiKeysUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
            const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, const std::vector< int >& columns_id, const vector< bool >& can_use_dict );

    AriesStarJoinResult StarInnerJoinWithHash( const std::vector< AriesHashTableWrapper >& tables,
            const std::vector< AriesIndicesArraySPtr >& dimension_tables_indices, const std::vector< AriesHashJoinDataWrapper >& datas,
            const AriesIndicesArraySPtr& fact_table_indices, size_t row_count );

    JoinPair FullJoinWithHash( const AriesHashTableUPtr& hash_table,
                               const AriesIndicesArraySPtr& hash_table_indices,
                               const aries_engine::AriesTableBlockUPtr& table_block,
                               const AriesIndicesArraySPtr& table_indices,
                               int column_id,
                               const JoinDynamicCodeParams* joinDynamicCodeParams,
                               bool can_use_dict,
                               const bool needToSwap = false );

    JoinPair LeftJoinWithHash( const AriesHashTableUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
            const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, int column_id,
            const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool left_as_hash );

    JoinPair LeftJoinWithHash( const AriesHashTableMultiKeysUPtr& hash_table, const AriesIndicesArraySPtr& hash_table_indices,
            const aries_engine::AriesTableBlockUPtr& table_block, const AriesIndicesArraySPtr& table_indices, const std::vector< int >& column_ids,
            const JoinDynamicCodeParams* joinDynamicCodeParams, const vector< bool >& can_use_dict, bool left_as_hash );

    AriesInt32ArraySPtr HalfJoinWithLeftHashInternal( const AriesHashTableUPtr& left_hash_table, const aries_engine::AriesTableBlockUPtr& table,
            int column_id, const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool isSemiJoin, bool isNotIn );

    AriesInt32ArraySPtr HalfJoinWithLeftHash( AriesJoinType joinType, const AriesHashTableUPtr& left_hash_table, const AriesIndicesArraySPtr& indices,
            const aries_engine::AriesTableBlockUPtr& right_table, int column_id, const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool isNotIn );

    AriesInt32ArraySPtr HalfJoinWithRightHash( AriesJoinType joinType, const AriesHashTableUPtr& right_hash_table,
            const AriesIndicesArraySPtr& indices, const aries_engine::AriesTableBlockUPtr& left_table, int column_id,
            const JoinDynamicCodeParams* joinDynamicCodeParams, bool can_use_dict, bool isNotIn );


    /**
     * @brief 计算在 hash join 时，创建hash table所需要的显存大小
     * @param table_block 用来创建hash table的表
     * @param columns_id 用来创建hash table的列
     * @return 总共占用显存大小
     */
    size_t EstimateBuildHashTableMemOccupancy( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id );

    /**
     * @brief 计算在 hash join 时，每行数据创建 hash table 所需要的显存大小
     * @param table_block 用来创建hash table的表
     * @param columns_id 用来创建hash table的列
     * @return 总共占用显存大小
     */
    size_t EstimateBuildHashTableMemOccupancyPerRow( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id );

    size_t GetHashTableSizePerRow( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id );

    /**
     * @brief 获取 left hash join 对于每一行左表数据，需要占用多少显存
     * @param[in] hash_tabel_row_count 左表的总行数，用以计算左表和右表的行数比例
     * @param[in] value_table_row_count 右表的总行数，用以计算左表和右表的行数比例
     * @param[in] value_table 右表数据
     * @param[in] columns_id 右表参与 join 的列
     * @param[in] has_indices 右表是否包含 indices （未物化的列）
     */
    size_t GetLeftHashJoinUsage(
        const size_t hash_tabel_row_count,
        const size_t value_table_row_count,
        const aries_engine::AriesTableBlockUPtr& value_table,
        const std::vector< int >& columns_id,
        bool has_indices );

    /**
     * @brief 获取 build hash table 所需要的显存大小
     * @param[in] originTable 用于 build hash table 的数据
     * @param[in] columnIds 用于 build hash table 的 column 的 id
     * @param[in] partRowCount `originTable` 中有多少行数据会用于 build hash table
     * @details 该方法用于在分块 build hash table 之前，判断是否需要将数据物化之后放入显存，
     *          如果空间足够，直接将原始数据和 indices 放入显存再构建 hash table 是更快的方式，
     *          否则需要先将数据物化，再将物化后的数据放入显存
     */
    size_t GetBuildHashTableUsage(
        const aries_engine::AriesTableBlockUPtr& originTable,
        const std::vector< int32_t >& columnIds,
        const size_t partRowCount
    );

    size_t GetHashTableSize(
        const aries_engine::AriesTableBlockUPtr& originTable,
        const std::vector< int32_t >& columnIds,
        const size_t partRowCount
    );

    /**
     * @brief 计算在 hash inner join 时，取value table一行数据进行扫描所需显存大小
     * @param table_block 进行扫描的表（ value table ) 
     * @param columns_id 用来扫描的列
     * @return 扫描一行占用显存大小
     */
    size_t EstimateHashInnerJoinPerRowMemOccupancy( const aries_engine::AriesTableBlockUPtr& table_block, const std::vector< int >& columns_id );

END_ARIES_ACC_NAMESPACE

#endif /* ARIESSQLOPERATOR_HASHJOIN_H_ */
