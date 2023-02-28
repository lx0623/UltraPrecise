/*
 * AriesOpNode.h
 *
 *  Created on: Oct 31, 2019
 *      Author: lichi
 */

#ifndef ARIESOPNODE_H_
#define ARIESOPNODE_H_

#include "CudaAcc/AriesSqlOperator.h"
#include "AriesDataDef.h"
#include "AriesAssert.h"
#include "AriesSpoolCacheManager.h"
#include "frontend/nlohmann/json.hpp"
#include "transaction/AriesInitialTable.h"

bool IsCurrentThdKilled();
void SendKillMessage();

using JSON = nlohmann::json;

BEGIN_ARIES_ENGINE_NAMESPACE

    enum class AriesOpNodeStatus
    {
        CONTINUE, END, ERROR
    };

    struct AriesOpResult
    {
        AriesOpResult()
        {
            Status = AriesOpNodeStatus::ERROR;
        }
        AriesOpResult( AriesOpNodeStatus status, AriesTableBlockUPtr tableBlock )
        {
            Status = status;
            TableBlock = std::move( tableBlock );
        }
        AriesOpNodeStatus Status;
        AriesTableBlockUPtr TableBlock; //!< set to null only when Status is ERROR. If the result is an empty table, it's a table block with no content.
    };

   // static constexpr int64_t ARIES_DATA_BLOCK_ROW_SIZE = 20971520;
    static constexpr int64_t ARIES_DATA_BLOCK_ROW_SIZE = ARIES_BLOCK_FILE_ROW_COUNT;

    class AriesOpNode;
    using AriesOpNodeSPtr = shared_ptr< AriesOpNode >;
    class AriesOpNode
    {
    public:
        AriesOpNode() : m_opTime( 0 ), m_nodeId( -1 ), m_rowCount( 0 ),
                        m_leftRowCount( 0 ), m_rightRowCount( 0 ), m_spoolId( -1 )
        {
        }
        
        virtual ~AriesOpNode()
        {
            m_outputColumnTypes.clear();
        }
        virtual bool Open() = 0;
        virtual AriesOpResult GetNext() = 0;
        virtual void Close() = 0;
        virtual void ReleaseData();
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules )
        {
        }
        virtual string GetCudaKernelCode() const
        {
            return string();
        }
        virtual JSON GetProfile() const;
        virtual void SetSourceNode( AriesOpNodeSPtr leftSource, AriesOpNodeSPtr rightSource = nullptr )
        {
            m_dataSource = leftSource;
            m_leftSource = leftSource;
            m_rightSource = rightSource;
        }

        //该函数暂不使用
        //接收数据的下级OpNode调用此函数，表示ids中的column自己不会使用（在自己的处理过程中属于打酱油字段）
        void SetTrivalColumnIds( set< int32_t >&& ids )
        {
            m_trivalColumnIds = ids;
        }

        void SetNodeId( int id )
        {
            m_nodeId = id;
        }

        void SetUniqueColumnsId( const std::vector< std::vector< int > >& columns_id )
        {
            unique_columns_id.assign( columns_id.cbegin(), columns_id.cend() );
        }

        const std::vector< std::vector< int > >& GetUniqueColumnsId()
        {
            return unique_columns_id;
        }

        virtual AriesTableBlockUPtr GetEmptyTable() const;
        int GetSpoolId() const
        {
            return m_spoolId;
        }

        void SetSpoolCache( const int spoolId, const AriesSpoolCacheManagerSPtr& manager );

    protected:
        bool IsOutputColumnsEmpty() const {
            return m_outputColumnTypes.empty();
        }

        bool NeedCacheSpool()
        {
            return ( m_spoolId > -1 ) && ( nullptr != m_spool_cache_manager );
        }
        void CacheNodeData( const AriesTableBlockUPtr& tableBlock );
        AriesOpResult GetCachedResult() const;

    protected:
        AriesOpNodeSPtr m_leftSource;
        AriesOpNodeSPtr m_dataSource;
        AriesOpNodeSPtr m_rightSource;
        set< int32_t > m_trivalColumnIds;//数据接收方中的打酱油字段
        vector< AriesColumnType > m_outputColumnTypes;

        // for performance statistics
        AriesTableBlockStats m_tableStats;
        AriesTableBlockStats m_leftTableStats;
        AriesTableBlockStats m_rightTableStats;
        long m_opTime; // 操作耗时
        int m_nodeId; // 节点id
        string m_opName; // 打印日志需要
        string m_opParam; //打印日志需要
        atomic< int64_t > m_rowCount; // 处理的数据总条数
        int64_t m_leftRowCount; // join操作左表数据条数
        int64_t m_rightRowCount; // join操作右表数据条数
        std::vector< std::vector< int > > unique_columns_id;
        int m_spoolId;
        AriesSpoolCacheManagerSPtr m_spool_cache_manager;
    };

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESOPNODE_H_ */
