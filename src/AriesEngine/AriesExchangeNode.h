/*
 * AriesExchangeNode.h
 *
 *  Created on: Sep 1, 2020
 *      Author: lichi
 */

#ifndef ARIESEXCHANGENODE_H_
#define ARIESEXCHANGENODE_H_

#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <memory>

#include "AriesOpNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesExchangeNode: public AriesOpNode
    {
        using OpResult = std::pair< int, AriesOpResult >;
        struct DeviceInfo
        {
            AriesOpNodeStatus Status;
            future< OpResult > Result;
        };
    public:
        AriesExchangeNode();
        virtual ~AriesExchangeNode();

        void SetDispatchInfo( int dstDevice, const vector< int >& srcDevices );
        void AddSourceNode( AriesOpNodeSPtr node );

        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;
        JSON GetProfile() const override final;

    private:
        //simple mode
        AriesOpResult GetNextSimply();

        //gently mode
        AriesOpResult GetNextGently();
        AriesOpResult ProcessExistingDeviceInfo();
        void WaitCurrentJobDone();

        //aggressively mode
        AriesOpResult GetNextAggressively();
        AriesOpResult ProcessExistingDataBlocks();
        void WaitAllJobDone();

    private:
        //simple mode
        vector< AriesOpNodeStatus > m_sourceStatus;

        condition_variable m_cv;
        mutex m_mutex;

        //gently mode
        vector< int > m_finishedDevices;
        map< int, DeviceInfo > m_deviceInfo;
        map< int, int > m_deviceNodeMapping;

        //aggressively mode
        vector< AriesTableBlockUPtr > m_receivedDataBlocks;
        atomic< std::size_t > m_receivedEndCount;
        atomic< bool > m_bHasError;
        atomic< bool > m_bKilled;
        map< int, future< void > > m_deviceThreads;

        int m_dstDevice;
        vector< int > m_srcDevices;
        vector< AriesOpNodeSPtr > m_sourceNodes;
        vector< size_t > m_srcRowCount;
        vector< AriesTableBlockStats > m_srcTableStats;
    };

    using AriesExchangeNodeSPtr = std::shared_ptr< AriesExchangeNode >;

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESEXCHANGENODE_H_ */
