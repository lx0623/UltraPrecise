/*
 * AriesMvccScanNode.h
 *
 *  Created on: Mar 24, 2020
 *      Author: tengjianping
 */

#pragma once

#include "AriesDataDef.h"
#include "AriesOpNode.h"
#include "transaction/AriesMvccTable.h"
using namespace aries;

BEGIN_ARIES_ENGINE_NAMESPACE

#define USE_DATA_CACHE

    class AriesMvccScanNode: public AriesOpNode
    {
    public:
        AriesMvccScanNode( const AriesTransactionPtr& tx,
                           const string& dbName,
                           const string& tableName );
        ~AriesMvccScanNode();

    public:
        void SetOutputColumnIds( const vector< int >& columnIds );

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        AriesOpResult GetNextRange();
        void Close() override final;
        void ReleaseData() override;

        // split the total data into totalSliceCount slices,
        // and return the sliceIdx slice
        void SetRange( int totalSliceCount, int sliceIdx )
        {
            ARIES_ASSERT( totalSliceCount > 0,
                          "invalid slice count: " + std::to_string( totalSliceCount ) );
            ARIES_ASSERT( sliceIdx >= 0 && sliceIdx < totalSliceCount,
                          "invalid slice index: " + std::to_string( sliceIdx ) );
            m_totalSliceCount = totalSliceCount;
            m_sliceIdx = sliceIdx;
        }

        void AddPartitionCondition( AriesCommonExprUPtr condition );

    private:
        string m_dbName;
        string m_tableName;
        AriesTransactionPtr m_tx;

        vector< int > m_outputColumnIds;
        AriesTableBlockUPtr m_outputTable;
        int64_t m_readRowCount;

        int m_totalSliceCount;
        int m_sliceIdx;

        vector< int64_t > m_dataBlockPrefixSumArray;
        size_t m_blockIndex;

        std::vector< AriesCommonExprUPtr > m_partitionConditions;
    };

    using AriesMvccScanNodeSPtr = shared_ptr< AriesMvccScanNode >;

END_ARIES_ENGINE_NAMESPACE
