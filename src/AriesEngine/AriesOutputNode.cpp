/*
 * AriesOutputNode.cpp
 *
 *  Created on: Mar 16, 2019
 *      Author: lichi
 */

#include "AriesOutputNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesOutputNode::AriesOutputNode()
    {
        // TODO Auto-generated constructor stub

    }

    AriesOutputNode::~AriesOutputNode()
    {
        // TODO Auto-generated destructor stub
    }

    void AriesOutputNode::SetSourceOpNode( AriesOpNodeSPtr source )
    {
        m_sourceOpNode = source;
    }

    string AriesOutputNode::GetPipelineKernelCode() const
    {
        ARIES_ASSERT( m_sourceOpNode, "m_sourceOpNode is nullptr");
        string code = m_sourceOpNode->GetCudaKernelCode();
        if( !code.empty() )
        {
            code = R"(#include "functions.hxx"
#include "AriesDateFormat.hxx"
#include "aries_char.hxx"
#include "decimal.hxx"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesIntervalTime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"
#include "AriesYear.hxx"
#include "AriesTimeCalc.hxx"
#include "AriesSqlFunctions.hxx"
#include "AriesColumnDataIterator.hxx"
#include "AriesDecimal.hxx"
using namespace aries_acc;

)" + code;
        }
        return code;
    }

    void AriesOutputNode::AttachCuModuleToPipeline( const vector< CUmoduleSPtr >& modules )
    {
        ARIES_ASSERT( m_sourceOpNode, "m_sourceOpNode is nullptr");
        m_sourceOpNode->SetCuModule( modules );
    }

    AriesTableBlockUPtr AriesOutputNode::GetResult()
    {
        ARIES_ASSERT( m_sourceOpNode, "m_sourceOpNode is nullptr");
        AriesTableBlockUPtr tableBlock;
        if( m_sourceOpNode->Open() )
        {
            AriesOpResult result = m_sourceOpNode->GetNext();
            tableBlock = std::move( result.TableBlock );
            while( result.Status == AriesOpNodeStatus::CONTINUE )
            {
                result = m_sourceOpNode->GetNext();
                if( result.Status == AriesOpNodeStatus::ERROR )
                {
                    break;
                }
                tableBlock->AddBlock( std::move( result.TableBlock ) );
            }
            m_sourceOpNode->Close();
            if( result.Status == AriesOpNodeStatus::ERROR )
            {
                return nullptr;
            }
        }
        else
        {
            LOG(ERROR) << "open source failed";
        }
        return std::move( tableBlock );
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
