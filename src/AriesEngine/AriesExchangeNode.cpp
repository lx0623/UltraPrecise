/*
 * AriesExchangeNode.cpp
 *
 *  Created on: Sep 1, 2020
 *      Author: lichi
 */

#include "AriesExchangeNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesExchangeNode::AriesExchangeNode()
            : m_dstDevice( -1 )
    {
        m_opName = "exchange";

    }

    AriesExchangeNode::~AriesExchangeNode()
    {
    }

    void AriesExchangeNode::SetDispatchInfo( int dstDevice, const vector< int >& srcDevices )
    {
        m_dstDevice = dstDevice;
        m_srcDevices = srcDevices;
        int index = 0;
        for( int device: srcDevices )
            m_deviceNodeMapping.insert( { device, index++ } );
    }

    void AriesExchangeNode::AddSourceNode( AriesOpNodeSPtr node )
    {
        assert( node );
        m_sourceNodes.push_back( node );
    }

    void AriesExchangeNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        assert( !m_sourceNodes.empty() );
        for( auto& node : m_sourceNodes )
            node->SetCuModule( modules );
    }

    string AriesExchangeNode::GetCudaKernelCode() const
    {
        assert( !m_sourceNodes.empty() );
        string code;
        for( auto& node : m_sourceNodes )
            code += node->GetCudaKernelCode();
        return code;
    }

    AriesOpResult AriesExchangeNode::GetNext()
    {
        assert( !m_sourceNodes.empty() && m_dstDevice != -1 );
        assert( m_sourceNodes.size() == m_srcDevices.size() );
        //return GetNextGently();
        return GetNextAggressively();
    }

    AriesOpResult AriesExchangeNode::GetNextSimply()
    {
        AriesOpResult ret
        { AriesOpNodeStatus::CONTINUE, nullptr };
        vector< future< OpResult > > allThreads;
        int index = 0;
        for( auto deviceId : m_srcDevices )
        {
            if( m_sourceStatus[index] == AriesOpNodeStatus::CONTINUE )
            {
                allThreads.push_back( std::async( std::launch::async, [&]( int idx )
                {
                    cudaSetDevice( deviceId );
                    return OpResult
                    {   idx, m_sourceNodes[ idx ]->GetNext()};
                }, index ) );
            }
            ++index;
        }

        // collect output data from each gpu
        bool endStatus = true;
        for( auto& t : allThreads )
        {
            auto retVal = t.get();
            AriesOpNodeStatus status = retVal.second.Status;
            m_sourceStatus[retVal.first] = status;
            if( status != AriesOpNodeStatus::ERROR )
            {
                if( status == AriesOpNodeStatus::CONTINUE )
                    endStatus = false;
                if( ret.TableBlock )
                    ret.TableBlock->AddBlock( std::move( retVal.second.TableBlock ) );
                else
                    ret.TableBlock = std::move( retVal.second.TableBlock );
            }
            else
            {
                ret.Status = AriesOpNodeStatus::ERROR;
                ret.TableBlock = nullptr;
                break;
            }
        }
        if( ret.Status != AriesOpNodeStatus::ERROR && endStatus )
            ret.Status = AriesOpNodeStatus::END;

        cudaSetDevice( m_dstDevice );
        return ret;
    }

    bool AriesExchangeNode::Open()
    {
        assert( !m_sourceNodes.empty() && m_sourceStatus.empty() );
        m_receivedEndCount = 0;
        m_receivedDataBlocks.clear();
        m_sourceStatus.clear();
        m_deviceInfo.clear();
        m_finishedDevices.clear();
        m_deviceThreads.clear();
        m_bHasError = false;
        m_bKilled = false;
        bool bRet = true;
        for( auto& node : m_sourceNodes )
        {
            m_sourceStatus.push_back( AriesOpNodeStatus::CONTINUE );
            bRet = bRet && node->Open();
        }
        m_srcRowCount.resize( m_sourceNodes.size() );
        m_srcTableStats.resize( m_sourceNodes.size() );
        return bRet;
    }

    void AriesExchangeNode::Close()
    {
        assert( !m_sourceNodes.empty() );
        for( auto& node : m_sourceNodes )
            node->Close();
    }

    AriesOpResult AriesExchangeNode::GetNextAggressively()
    {
        if( IsCurrentThdKilled() )
            m_bKilled = true;
        if( m_deviceThreads.empty() )
        {
            int nodeIndex = 0;
            for( auto deviceId : m_srcDevices )
            {
                m_deviceThreads.insert(
                { deviceId, std::async( std::launch::async, [&](int id, int srcNodeIndex )
                {
                    cudaSetDevice( id );
                    AriesOpResult output;
                    do
                    {
                        output = m_sourceNodes[ srcNodeIndex ]->GetNext();
                        if( output.Status == AriesOpNodeStatus::ERROR )
                            m_bHasError = true;
                        else
                        {
                            unique_lock< mutex > lock( m_mutex );
                            m_srcRowCount[ srcNodeIndex ] = m_srcRowCount[ srcNodeIndex ] + output.TableBlock->GetRowCount();
                            m_srcTableStats[ srcNodeIndex ] = m_srcTableStats[ srcNodeIndex ] += output.TableBlock->GetStats();
                            output.TableBlock->SetDeviceId( id );
                            m_receivedDataBlocks.push_back( std::move( output.TableBlock ) );
                        }
                        m_cv.notify_one();
                    }while( !m_bKilled && !m_bHasError && output.Status == AriesOpNodeStatus::CONTINUE );
                    if( output.Status == AriesOpNodeStatus::END )
                        ++m_receivedEndCount;
                }, deviceId, nodeIndex ) } );
                ++nodeIndex;
            }
        }
        auto itDst = m_deviceThreads.find( m_dstDevice );
        if( itDst != m_deviceThreads.end() )
            itDst->second.wait();
        AriesOpResult ret = ProcessExistingDataBlocks();
        if( ret.Status == AriesOpNodeStatus::CONTINUE )
        {
            if( !ret.TableBlock )
            {
                //　等待处理
                {
                    unique_lock< mutex > lock( m_mutex );
                    m_cv.wait( lock, [ & ]
                    {   return !m_receivedDataBlocks.empty() || m_bHasError || m_bKilled;} );
                }
                ret = ProcessExistingDataBlocks();
            }
        }
        return ret;
    }

    AriesOpResult AriesExchangeNode::ProcessExistingDataBlocks()
    {
        AriesOpResult ret
        { AriesOpNodeStatus::CONTINUE, nullptr };
        if( m_bHasError )
        {
            ret.Status = AriesOpNodeStatus::ERROR;
            ret.TableBlock = nullptr;
            WaitAllJobDone();
        }
        else
        {
            if( m_receivedEndCount == m_srcDevices.size() )
            {
                WaitAllJobDone();
                ret.Status = AriesOpNodeStatus::END;
            }

            vector< AriesTableBlockUPtr > dataBlocks;
            {
                unique_lock< mutex > lock( m_mutex );
                std::swap( dataBlocks, m_receivedDataBlocks );
            }
            for( auto& data : dataBlocks )
            {
                data->MoveIndicesToDevice( m_dstDevice );
                if( ret.TableBlock )
                    ret.TableBlock->AddBlock( std::move( data ) );
                else
                    ret.TableBlock = std::move( data );
            }
        }
        return ret;
    }

    AriesOpResult AriesExchangeNode::GetNextGently()
    {
        AriesOpResult ret = ProcessExistingDeviceInfo();
        if( ret.Status == AriesOpNodeStatus::CONTINUE )
        {
            map< int, DeviceInfo > tmpInfo;
            for( auto deviceId : m_srcDevices )
            {
                auto it = m_deviceInfo.find( deviceId );
                if( it == m_deviceInfo.end() || ( it->second.Status == AriesOpNodeStatus::CONTINUE && !it->second.Result.valid() ) )
                {
                    tmpInfo.insert(
                    { deviceId, DeviceInfo
                    { AriesOpNodeStatus::CONTINUE, std::async( std::launch::async, [&]( int id, int srcNodeIndex )
                    {
                        cudaSetDevice( id );
                        auto output = m_sourceNodes[ srcNodeIndex ]->GetNext();
                        {
                            unique_lock< mutex > lock( m_mutex );
                            m_finishedDevices.push_back( id );
                            m_cv.notify_one();
                        }
                        return OpResult
                        {   id, std::move( output )};
                    }, deviceId, m_deviceNodeMapping[ deviceId ] ) } } );
                }
            }

            for( auto & it : tmpInfo )
                m_deviceInfo[it.first] = std::move( it.second );

            if( !ret.TableBlock )
            {
                //　等待处理
                {
                    unique_lock< mutex > lock( m_mutex );
                    m_cv.wait( lock, [ & ]
                    {   return !m_finishedDevices.empty();} );
                }
                ret = ProcessExistingDeviceInfo();
            }
        }

        cudaSetDevice( m_dstDevice );
        return ret;
    }

    AriesOpResult AriesExchangeNode::ProcessExistingDeviceInfo()
    {
        AriesOpResult ret
        { AriesOpNodeStatus::CONTINUE, nullptr };
        if( m_deviceInfo.empty() )
            return ret;

        bool hasError = false;
        set< int > endDevices;
        map< int, DeviceInfo > tmpInfo;
        for( auto& info : m_deviceInfo )
        {
            switch( info.second.Status )
            {
                case AriesOpNodeStatus::ERROR:
                {
                    hasError = true;
                    break;
                }
                case AriesOpNodeStatus::CONTINUE:
                {
                    if( !hasError )
                    {
                        vector< int > tmpFinished;
                        {
                            unique_lock< mutex > lock( m_mutex );
                            std::swap( tmpFinished, m_finishedDevices );
                        }
                        for( int deviceId : tmpFinished )
                        {
                            assert( m_deviceInfo[deviceId].Result.valid() );
                            auto retVal = m_deviceInfo[deviceId].Result.get();
                            AriesOpNodeStatus status = retVal.second.Status;
                            m_deviceInfo[deviceId].Status = status;
                            if( status != AriesOpNodeStatus::ERROR )
                            {
                                if( ret.TableBlock )
                                    ret.TableBlock->AddBlock( std::move( retVal.second.TableBlock ) );
                                else
                                    ret.TableBlock = std::move( retVal.second.TableBlock );

                                // empty future as placeholder
                                tmpInfo.insert(
                                { retVal.first, DeviceInfo
                                { retVal.second.Status, future< OpResult >() } } );

                                if( status == AriesOpNodeStatus::END )
                                    endDevices.insert( deviceId );
                            }
                            else
                            {
                                hasError = true;
                                break;
                            }
                        }
                    }
                    break;
                }
                case AriesOpNodeStatus::END:
                {
                    endDevices.insert( info.first );
                    break;
                }
            }
        }

        if( hasError )
        {
            ret.Status = AriesOpNodeStatus::ERROR;
            ret.TableBlock = nullptr;
            WaitCurrentJobDone();
        }
        else
        {
            for( auto & it : tmpInfo )
                m_deviceInfo[it.first] = std::move( it.second );

            //　如果已经收到所有source的End信息，可以给下一级节点返回End
            if( endDevices.size() == m_srcDevices.size() )
                ret.Status = AriesOpNodeStatus::END;
        }
        return ret;
    }

    void AriesExchangeNode::WaitCurrentJobDone()
    {
        for( auto& info : m_deviceInfo )
        {
            if( info.second.Status == AriesOpNodeStatus::CONTINUE && info.second.Result.valid() )
                info.second.Result.wait();
        }
    }

    void AriesExchangeNode::WaitAllJobDone()
    {
        for( auto& info : m_deviceThreads )
            info.second.wait();
    }

    JSON AriesExchangeNode::GetProfile() const
    {
        JSON stats = {
            {"type", m_opName},
            {"param", m_opParam},
            {"time", m_opTime}
        };
        
        JSON m_srcTableStats_json;
        for ( std::size_t i = 0; i < m_srcTableStats.size(); i++ )
            m_srcTableStats_json.push_back( JSON::parse( m_srcTableStats[i].ToJson( m_srcRowCount[i] ) ) );
        stats["memory"] = m_srcTableStats_json;
        if(m_spoolId != -1)
            stats["spool_id"] = m_spoolId;
        JSON children_json;
        for ( const auto& source : m_sourceNodes )
            children_json.push_back(source->GetProfile());
        stats["children"] = children_json;
        return stats;
    }

END_ARIES_ENGINE_NAMESPACE
