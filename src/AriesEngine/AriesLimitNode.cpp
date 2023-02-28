/*
 * AriesLimitNode.cpp
 *
 *  Created on: Sep 4, 2019
 *      Author: lichi
 */

#include <algorithm>

#include "AriesLimitNode.h"
#include "CudaAcc/AriesEngineException.h"
#include "CpuTimer.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesLimitNode::AriesLimitNode()
            : m_offset( 0 ), m_size( 0 ), m_pos( 0 ), m_outputCount( 0 )
    {
    }

    AriesLimitNode::~AriesLimitNode()
    {
    }

    void AriesLimitNode::SetLimitInfo( int64_t offset, int64_t size )
    {
        m_offset = offset;
        m_size = size;
    }

    bool AriesLimitNode::IsValidSize() const {
        return m_size > 0;
    }

    void AriesLimitNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
        m_dataSource->SetCuModule( modules );
    }

    string AriesLimitNode::GetCudaKernelCode() const
    {
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
        return m_dataSource->GetCudaKernelCode();
    }

    bool AriesLimitNode::Open()
    {
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
        m_pos = 0;
        m_outputCount = 0;
        return m_dataSource->Open();
    }

    AriesOpResult AriesLimitNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        if( m_pos < m_offset + m_size && m_size > 0 )
        {
            auto data = m_dataSource->GetNext();

            if (m_emptyTablePtr == nullptr)
            {
                m_emptyTablePtr = data.TableBlock->CloneWithNoContent();
            }
            while( data.Status != AriesOpNodeStatus::ERROR && !IsCurrentThdKilled() )
            {
#ifdef ARIES_PROFILE
                t.begin();
#endif
                const AriesTableBlockUPtr & dataBlock = data.TableBlock;
                int tupleNum = dataBlock->GetRowCount();
                if( tupleNum > 0 )
                {
                    m_rowCount += tupleNum;

                    if( m_pos + tupleNum > m_offset )
                    {
                        int64_t startPos = std::max( m_offset - m_pos, 0l );
                        int64_t outputTupleNum = std::min( tupleNum - startPos, m_size - m_outputCount );
                        ARIES_ASSERT( outputTupleNum > 0, "outputTupleNum: " + to_string(outputTupleNum));
                        m_pos += tupleNum;
                        m_outputCount += outputTupleNum;
                        if ( m_outputCount == m_size )
                        {
                            data.Status = AriesOpNodeStatus::END;
                        }
                        if( startPos == 0 && outputTupleNum == tupleNum )
                        {
#ifdef ARIES_PROFILE
                            m_opTime += t.end();
#endif
                            CacheNodeData( data.TableBlock );
                            return data;
                        }
                        else
                        {
                            auto subTable = dataBlock->GetSubTable( startPos, outputTupleNum, true );
#ifdef ARIES_PROFILE
                            m_opTime += t.end();
#endif
                            CacheNodeData( subTable );
                            return {data.Status, std::move( subTable )};
                        }
                    } else {
                        m_pos += tupleNum;
                    }
                }
                if( data.Status == AriesOpNodeStatus::END )
                {
#ifdef ARIES_PROFILE
                    m_opTime += t.end();
#endif
                    // no any data selected
                    if ( m_pos <= m_offset )
                    {
                        return {AriesOpNodeStatus::END, m_emptyTablePtr->CloneWithNoContent()};
                    }
                    CacheNodeData( data.TableBlock );
                    return data;
                }
#ifdef ARIES_PROFILE
                m_opTime += t.end();
#endif
                data = m_dataSource->GetNext();
            }
            if ( IsCurrentThdKilled() )
            {
                LOG(INFO) << "thread was killed in AriesLimitNode::GetNext";
                SendKillMessage();
            }
            if( data.Status == AriesOpNodeStatus::ERROR )
            {
                LOG(ERROR) << "read error in AriesLimitNode::GetNext";
                data.TableBlock = nullptr;
                return data;
            }
        }
        else
        {
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if (m_emptyTablePtr == nullptr) {
                m_emptyTablePtr = m_dataSource->GetNext().TableBlock->CloneWithNoContent();
            }
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
            return {   AriesOpNodeStatus::END, m_emptyTablePtr->CloneWithNoContent()};
        }

        return {   AriesOpNodeStatus::ERROR, m_emptyTablePtr->CloneWithNoContent()};
    }

    void AriesLimitNode::Close()
    {
        ARIES_ASSERT( m_dataSource, "m_dataSource is nullptr");
        m_pos = 0;
        m_outputCount = 0;
        m_dataSource->Close();
    }
    JSON AriesLimitNode::GetProfile() const
    {
        JSON stat = this->AriesOpNode::GetProfile();
        stat["type"] = "AriesLimitNode";
        return stat;
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */
