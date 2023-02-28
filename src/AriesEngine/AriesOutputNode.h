/*
 * AriesOutputNode.h
 *
 *  Created on: Mar 16, 2019
 *      Author: lichi
 */

#pragma once

#include "AriesOpNode.h"
#include "AriesUtil.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesOutputNode: protected DisableOtherConstructors
    {
    public:
        AriesOutputNode();
        ~AriesOutputNode();

    public:
        void SetSourceOpNode( AriesOpNodeSPtr source );
        string GetPipelineKernelCode() const;
        void AttachCuModuleToPipeline( const vector< CUmoduleSPtr >& modules );
        // pair< DataSourceStatus, AriesDataBlockUPtr > GetResult();
        AriesTableBlockUPtr GetResult();

    private:
        AriesOpNodeSPtr m_sourceOpNode;
    };

    using AriesOutputNodeSPtr = shared_ptr< AriesOutputNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngineV2 */

