/*
 * AriesSetOperationNode.cpp
 *
 *  Created on: Sep 25, 2019
 *      Author: lichi
 */

#include "AriesSetOperationNode.h"
#include "AriesAssert.h"
#include "AriesUtil.h"
#include "CudaAcc/AriesEngineException.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesSetOperationNode::AriesSetOperationNode()
    {
        // TODO Auto-generated constructor stub

    }

    AriesSetOperationNode::~AriesSetOperationNode()
    {
        // TODO Auto-generated destructor stub
    }

    void AriesSetOperationNode::SetOpType( AriesSetOpType opType )
    {
        m_opType = opType;
    }

    bool AriesSetOperationNode::Open()
    {
        ARIES_ASSERT(m_leftSource && m_rightSource,
                            "m_leftSource is nullptr: " + to_string(!!m_leftSource) + ", m_rightSource: " +
                            to_string(!!m_rightSource));
        return m_leftSource->Open() && m_rightSource->Open();
    }

    AriesOpResult AriesSetOperationNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;
        AriesOpResult result
        { AriesOpNodeStatus::ERROR, nullptr };
        switch( m_opType )
        {
            case AriesSetOpType::UNION_ALL:
                return UnionAllGetNext();
            case AriesSetOpType::UNION:
                return UnionGetNext();
            default:
                //FIXME need support other join types;
                assert( 0 );
                ARIES_ENGINE_EXCEPTION(ER_NOT_SUPPORTED_YET, "set Operation type " + GetAriesSetOpTypeName (m_opType));
                break;
        }
        CacheNodeData( result.TableBlock );
        return result;
    }

    AriesOpResult  AriesSetOperationNode::UnionGetNext()
    {
        AriesOpResult result = UnionAllGetNext();
        if( result.Status == AriesOpNodeStatus::END )
        {
            //remove duplicate rows
            map< int32_t, AriesColumnSPtr > columns = result.TableBlock->GetAllColumns();
            vector< AriesDataBufferSPtr > groupByColumns;
            for( auto& col : columns )
                groupByColumns.push_back( col.second->GetDataBuffer() );
            AriesInt32ArraySPtr outAssociated;
            AriesInt32ArraySPtr outGroups;
            aries_acc::GroupColumns( groupByColumns, outAssociated, outGroups );
            vector< AriesDataBufferSPtr > distinctBuffers = aries_acc::GatherGroupedColumnData( groupByColumns, outAssociated, outGroups );
            AriesTableBlockUPtr table = make_unique< AriesTableBlock >();
            int index = 0;
            for( auto& col : columns )
            {
                AriesColumnSPtr newCol = std::make_shared< AriesColumn >();
                newCol->AddDataBuffer( distinctBuffers[ index++ ] );
                table->AddColumn( col.first, newCol );
            }

            result.TableBlock = std::move( table );
        }

        return result;
    }

    AriesOpResult  AriesSetOperationNode::UnionAllGetNext()
    {
        AriesOpResult result
        { AriesOpNodeStatus::ERROR, nullptr };
        auto dataBlock = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        result.TableBlock = move( dataBlock.TableBlock );

        while ( dataBlock.Status == AriesOpNodeStatus::CONTINUE && !IsCurrentThdKilled() )
        {
            dataBlock = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if ( dataBlock.TableBlock && dataBlock.TableBlock->GetRowCount() > 0 )
                result.TableBlock->AddBlock( std::move( dataBlock.TableBlock ) );
#ifdef ARIES_PROFILE            
            m_opTime += t.end();
#endif
        }

        if ( IsCurrentThdKilled() )
            SendKillMessage();

        if ( dataBlock.Status == AriesOpNodeStatus::ERROR )
        {
            result.TableBlock = nullptr;
            return result;
        }

        dataBlock = m_rightSource->GetNext();
#ifdef ARIES_PROFILE
        t.begin();
#endif
        if ( dataBlock.TableBlock && dataBlock.TableBlock->GetRowCount() > 0 )
            result.TableBlock->AddBlock( std::move( dataBlock.TableBlock ) );
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        while ( dataBlock.Status == AriesOpNodeStatus::CONTINUE && !IsCurrentThdKilled() )
        {
            dataBlock = m_rightSource->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if ( dataBlock.TableBlock && dataBlock.TableBlock->GetRowCount() > 0 )
                result.TableBlock->AddBlock( std::move( dataBlock.TableBlock ) );
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }

        if ( IsCurrentThdKilled() )
            SendKillMessage();
        result.Status = dataBlock.Status;

        if ( result.TableBlock )
            m_rowCount += result.TableBlock->GetRowCount();

        return result;
    }

    void AriesSetOperationNode::Close()
    {
        ARIES_ASSERT(m_leftSource && m_rightSource,
                            "m_leftSource is nullptr: " + to_string(!!m_leftSource) + ", m_rightSource: " +
                            to_string(!!m_rightSource));
        m_leftSource->Close();
        m_rightSource->Close();
    }

    JSON AriesSetOperationNode::GetProfile() const
    {
        JSON stat = this->AriesOpNode::GetProfile();
        stat["type"] = "AriesSetOperationNode";
        return stat;
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */
