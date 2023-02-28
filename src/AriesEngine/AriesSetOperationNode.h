/*
 * AriesSetOperationNode.h
 *
 *  Created on: Sep 25, 2019
 *      Author: lichi
 */

#ifndef ARIESSETOPERATIONNODE_H_
#define ARIESSETOPERATIONNODE_H_

#include "AriesOpNode.h"
#include "CudaAcc/AriesEngineException.h"
using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesSetOperationNode: public AriesOpNode
    {
    public:
        AriesSetOperationNode();
        ~AriesSetOperationNode();
        void SetOpType( AriesSetOpType opType );
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules )
        {
            ARIES_ASSERT(m_leftSource && m_rightSource,
                                "m_leftSource is nullptr: " + to_string(!!m_leftSource) + ", m_rightSource is nullptr: " +
                                to_string(!!m_rightSource));
            m_leftSource->SetCuModule( modules );
            m_rightSource->SetCuModule( modules );
        }

        virtual string GetCudaKernelCode() const
        {
            ARIES_ASSERT(m_leftSource && m_rightSource,
                                "m_leftSource is nullptr: " + to_string(!!m_leftSource) + ", m_rightSource is nullptr: " +
                                to_string(!!m_rightSource));
            return m_leftSource->GetCudaKernelCode() + m_rightSource->GetCudaKernelCode();
        }

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;
        JSON GetProfile() const override final;

    private:
        AriesOpResult UnionAllGetNext();
        AriesOpResult UnionGetNext();

    private:
        AriesSetOpType m_opType;
    };

    using AriesSetOperationNodeSPtr = std::shared_ptr< AriesSetOperationNode >;

END_ARIES_ENGINE_NAMESPACE /* namespace aries_engine */

#endif /* ARIESSETOPERATIONNODE_H_ */
