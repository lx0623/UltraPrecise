//
// Created by david.shen on 2019/11/19.
//

#include "nodetest_framework.h"

/**
 * for UnittestFrameWork
 * */
NodeTestFrameWork::NodeTestFrameWork() : m_openStatus( false ), m_openStatusSet( false ), m_testOpNodeSPtr( nullptr ), m_leftSource( nullptr ),
                                         m_rightSource( nullptr )
{
    m_preFakeNodeBlocksInfo.clear();
}

NodeTestFrameWork::~NodeTestFrameWork()
{
    m_preFakeNodeBlocksInfo.clear();
}

void NodeTestFrameWork::SetTestNodeOpNode(const AriesOpNodeSPtr &opNodeSPtr)
{
    m_testOpNodeSPtr = opNodeSPtr;
}

void NodeTestFrameWork::SetTestNodeSource(PreFakeOpNodeSPtr leftSource, PreFakeOpNodeSPtr rightSource)
{
    m_leftSource = leftSource;
    m_rightSource = rightSource;
}

void NodeTestFrameWork::SetVerifyDataInfo(VerifyDataFunc verifyDataFunc)
{
    m_verifyDataFunc = verifyDataFunc;
}

void NodeTestFrameWork::SetOpenStatus(bool fakeStatus)
{
    m_openStatusSet = true;
    m_openStatus = fakeStatus;
}
bool NodeTestFrameWork::SetBlocksInfo(vector<BlockInfo> &info)
{
    if (!info.empty() && info[info.size() - 1].status != AriesOpNodeStatus::CONTINUE)
    {
        m_preFakeNodeBlocksInfo.assign(info.cbegin(), info.cend());
        return true;
    }
    return false;
}

bool NodeTestFrameWork::Run()
{
    try
    {
        // set all parameters
        if (m_openStatusSet) {
            if (m_leftSource != nullptr)
            {
                m_leftSource->SetOpenStatus(m_openStatus);
            }
            if (m_rightSource != nullptr)
            {
                m_rightSource->SetOpenStatus(m_openStatus);
            }
        }

        if (!m_preFakeNodeBlocksInfo.empty())
        {
            if (m_leftSource != nullptr)
            {
                m_leftSource->SetBlocksInfo(m_preFakeNodeBlocksInfo);
            }
            if (m_rightSource != nullptr)
            {
                m_rightSource->SetBlocksInfo(m_preFakeNodeBlocksInfo);
            }
        }
        // set pre-fake node
        if (m_leftSource != nullptr)
        {
            if (m_rightSource != nullptr)
            {
                m_testOpNodeSPtr->SetSourceNode(m_leftSource, m_rightSource);
            }
            else
            {
                m_testOpNodeSPtr->SetSourceNode(m_leftSource);
            }
        }

        bool opened = m_testOpNodeSPtr->Open();
        if (m_openStatusSet)
        {
            return opened == m_openStatus;
        }
        if (!opened)
        {
            cout << "Open Failed!" << endl;
            return false;
        }

        auto opResult = m_testOpNodeSPtr->GetNext();
        auto table = move(opResult.TableBlock);
        while (opResult.Status == AriesOpNodeStatus::CONTINUE)
        {
            opResult = m_testOpNodeSPtr->GetNext();
            if (opResult.Status == AriesOpNodeStatus::ERROR)
            {
                break;
            }
            table->AddBlock(move(opResult.TableBlock));
        }
        if (opResult.Status == AriesOpNodeStatus::ERROR)
        {
            return m_verifyDataFunc(opResult);
        }
        else
        {
            return m_verifyDataFunc({opResult.Status, move(table)});
        }
    }
    catch (AriesException &exception)
    {
        cout << "Error: " << exception.errMsg << endl;
    }
    catch (...)
    {
        cout << "Error: unknown error" << endl;
    }
    return false;
}
