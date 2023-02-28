//
// Created by david shen on 2019-07-23.
//

#pragma once

#include <deque>
#include "AriesOpNode.h"
#include "AriesJoinNodeHelper.h"
#include "cpu_algorithm.h"
#include "CudaAcc/AriesEngineException.h"
#include "frontend/SQLTreeNode.h"

using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

#define SUBTABLE_COUNT ARIES_DATA_BLOCK_ROW_SIZE
#define SEMI_ANTI_SBUTABLE_COUNT (SUBTABLE_COUNT * 3)
    class AriesJoinNode: public AriesOpNode
    {
    public:
        enum class HashJoinType : uint8_t
        {
            None,
            LeftAsHash,
            RightAsHash
        };
    public:
        AriesJoinNode();
        ~AriesJoinNode();
        void SetCondition( AriesCommonExprUPtr equalCondition, AriesCommonExprUPtr otherCondition, AriesJoinType type );
        void SetOutputColumnIds( const vector< int >& columnIds );
        void SetJoinHint( int joinHint, bool bIntact );

        void SetJoinConditionConstraintType( const JoinConditionConstraintType& type )
        {
            m_joinConditionConstraintType = type;
            m_joinHelper->SetJoinEqualConditionConstraintType( type );
        }

        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules )
        {
            ARIES_ASSERT(m_leftSource && m_rightSource,
                                "m_leftSource is nullptr: " + to_string(!!m_rightSource) + ", m_rightSource is nullptr: " +
                                to_string(!!m_rightSource));
            ARIES_ASSERT( m_joinHelper, "need call SetCondition function firstly" );
            m_leftSource->SetCuModule( modules );
            m_rightSource->SetCuModule( modules );
            m_joinHelper->SetCUModule( modules );
        }

        virtual string GetCudaKernelCode() const
        {
            ARIES_ASSERT(m_leftSource && m_rightSource,
                                "m_leftSource is nullptr: " + to_string(!!m_leftSource) + ", m_rightSource is nullptr: " +
                                to_string(!!m_rightSource));
            ARIES_ASSERT( m_joinHelper, "need call SetCondition function firstly" );
            return m_leftSource->GetCudaKernelCode() + m_rightSource->GetCudaKernelCode() + m_joinHelper->GetDynamicCode();
        }

        virtual void SetSourceNode( AriesOpNodeSPtr leftSource, AriesOpNodeSPtr rightSource = nullptr ) override;

        void SetHashJoinType( const HashJoinType& type );
        void SetHashJoinInfo( const HashJoinInfo& info );
        void SetUniqueKeys( const std::vector< int >& keys );
        void SetHashValueKeys( const std::vector< int >& keys );
        void SetIsNotIn( bool isNotIn );
    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;
        JSON GetProfile() const override final;
        virtual AriesTableBlockUPtr GetEmptyTable() const override final;
        bool IsConstFalseCondition();

        AriesTableBlockUPtr GenerateEmptyTable();
        AriesTableBlockUPtr GenerateTableWithRowCountOnly(size_t count);

        const std::vector< int > &GetOutputColumnIds() const
        {
            return m_outputColumnIds;
        }
        const std::vector< int > &GetLeftOutputColumnIds() const
        {
            return m_leftIds;
        }
        const std::vector< int > &GetRightOutputColumnIds() const
        {
            return m_rightIds;
        }
        const std::map<index_t, index_t> &GetLeftOutColumnIdMap() const
        {
            return m_leftOutColumnIdMap;
        }
        const std::map<index_t, index_t> &GetRightOutColumnIdMap() const
        {
            return m_rightOutColumnIdMap;
        }

    private:
        AriesCommonExprUPtr m_equalCondition;
        AriesCommonExprUPtr m_otherCondition;
        std::vector< int > m_leftIds, m_rightIds;
        std::map<index_t, index_t> m_leftOutColumnIdMap;
        std::map<index_t, index_t> m_rightOutColumnIdMap;
        JoinConditionConstraintType m_joinConditionConstraintType;

        void SplitOutputColumnIds();

        void PrepareHashTable( AriesOpNodeSPtr& value_source, size_t& hash_table_row_count );

        // AriesOpResult InnerJoinGetNextPartitioned();
        AriesOpResult InnerJoinWithHashSimplePartitioned();
        AriesOpResult InnerJoinWithHashGracePartitioned();
        AriesOpResult SortBasedInnerJoinWithGracePartitioned();
        AriesOpResult InnerJoinWithHash();
        AriesOpResult InnerJoinGetNext();
        AriesOpResult InnerJoinGetNextV2();
        AriesOpResult InnerJoinOneBlock();
        AriesOpResult InnerJoinOneBlockV2();
        AriesOpResult InnerJoinOneSubTable(AriesTableBlockUPtr &leftTable, AriesTableBlockUPtr &rightTable);
        bool CacheAllToSubTables( deque<AriesTableBlockUPtr>& cache,
                                  int64_t& totalRowCount,
                                  AriesOpNodeSPtr& source,
                                  AriesTableBlockUPtr &emptyTable,
                                  std::size_t rowCount,
                                  AriesTableBlockStats& tableStats );
        bool CacheAllLeftToSubTables(std::size_t rowCount);
        bool CacheAllRightToSubTables(std::size_t rowCount);

        bool CacheAllToSubTables( deque<AriesTableBlockUPtr>& cache,
                                  int64_t& totalRowCount,
                                  AriesOpNodeSPtr& source,
                                  AriesTableBlockUPtr &emptyTable );
        bool CacheAllLeftToSubTables();
        bool CacheAllRightToSubTables();

        AriesOpResult LeftJoinWithHashGracePartitioned();
        AriesOpResult FullJoinWithHashGracePartitioned();

        AriesOpResult LeftJoinGetNext();
        AriesOpResult LeftJoinGracePartitioned();
        AriesOpResult RightJoinGetNext();
        AriesOpResult FullJoinGetNext();
        AriesOpResult FullJoinGracePartitioned();

        AriesOpResult SemiOrAntiJoinReadData();
        AriesOpResult SemiOrAntiJoinCheckShortcut();
        AriesOpResult HashSemiOrAntiJoinGracePartitionTable(
            uint32_t seed,
            size_t &hashTablePartitionCount,
            vector< AriesTableBlockUPtr > &leftSubTables,
            vector< AriesTableBlockUPtr > &rightSubTables );

        AriesOpResult SemiOrAntiJoinGetNext();
        AriesOpResult SemiOrAntiJoinGracePartitioned();
        AriesOpResult SemiOrAntiHashJoinGetNext();
        AriesOpResult HashSemiOrAntiJoinGracePartitioned();

        AriesTableBlockUPtr ReadAllData( AriesOpNodeSPtr dataSource,
                                         AriesTableBlockUPtr &emptyTable,
                                         AriesTableBlockStats& tableStats );
        void CacheAllLeftTable();
        void CacheAllRightTable();
        AriesTableBlockUPtr GetNextCachedSubTable(deque<AriesTableBlockUPtr> &cache);

        AriesOpResult GetFullJoinResultOfFalseCondition();

        size_t GetHashTablePartitionCount() const;
        size_t GetPartitionCountForLeftHashJoin() const;

        void SwapRightJoinToLeft();
        vector< bool > CheckHashJoinConditionForDict( const aries_engine::AriesTableBlockUPtr& hashTable, const std::vector< int >& unique_keys, 
                                            const aries_engine::AriesTableBlockUPtr& valueTable, const std::vector< int >& value_keys );

    private:
        std::shared_ptr<AriesJoinNodeHelper> m_joinHelper;
        vector< int > m_outputColumnIds;
        AriesJoinType m_joinType;

        deque<AriesTableBlockUPtr> m_leftSubTablesCache;
        deque<AriesTableBlockUPtr> m_rightSubTablesCache;
        bool m_isLeftSubTableCached;
        bool m_isRightSubTableCached;
        int64_t m_rightHandledOffset;
        int64_t m_rightOneBlockSize;

        AriesTableBlockUPtr m_leftDataTable;
        AriesTableBlockUPtr m_rightDataTable;

        AriesTableBlockUPtr m_leftEmptyTable;
        AriesTableBlockUPtr m_rightEmptyTable;

        AriesHashTableUPtr m_hashTable;
        AriesHashTableMultiKeysUPtr m_hashTableMultiKeys;
        AriesTableBlockUPtr m_hashTableBlock;

        bool m_isLeftDataAllRead;
        bool m_isRightDataAllRead;

        long m_filterOpTime;
        bool m_usedHashJoin;

        HashJoinType m_hashJoinType;
        HashJoinInfo m_hashJoinInfo;
        std::vector< int > m_uniqueKeys;
        std::vector< int > m_hashValueKeys;

        AriesTableBlockUPtr m_hashTableData;
        AriesTableBlockUPtr m_valueTableData;
    };

    using AriesJoinNodeSPtr = shared_ptr< AriesJoinNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
