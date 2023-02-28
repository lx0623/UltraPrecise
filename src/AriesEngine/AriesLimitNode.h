/*
 * AriesLimitNode.h
 *
 *  Created on: Sep 4, 2019
 *      Author: lichi
 */

#ifndef ARIESLIMITNODE_H_
#define ARIESLIMITNODE_H_

#include "AriesOpNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesLimitNode: public AriesOpNode
    {
    public:
        AriesLimitNode();
        ~AriesLimitNode();
        void SetLimitInfo( int64_t offset, int64_t size );
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;
        JSON GetProfile() const override final;

    public:
        bool IsValidSize() const;
    private:
        int64_t m_offset;
        int64_t m_size;
        int64_t m_pos;
        int64_t m_outputCount;
        AriesTableBlockUPtr m_emptyTablePtr;
    };

    using AriesLimitNodeSPtr = shared_ptr< AriesLimitNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */

#endif /* ARIESLIMITNODE_H_ */
