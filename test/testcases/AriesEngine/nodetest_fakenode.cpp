//
// Created by david.shen on 2019/11/26.
//

#include "nodetest_fakenode.h"
/**
 * for TestFakeOpNode
 * */

bool PreFakeOpNode::Open()
{
    return  m_openStatus;
}

AriesOpResult PreFakeOpNode::GetNext()
{
    if (!m_openStatus)
    {
        throw AriesException( ERR_UNITTEST_EXCEPTION_CODE, "Call GetNext after Open failed");
    }
    else if (m_blocksInfo.empty())
    {
        throw AriesException( ERR_UNITTEST_EXCEPTION_CODE, "Block info of Data should be set");
    }
    else if (m_usedBlockIndex >= m_blocksInfo.size())
    {
        throw AriesException( ERR_UNITTEST_EXCEPTION_CODE, "Call GetNext after all Blocks sent");
    }
    else if (m_dataEnd)
    {
        throw AriesException( ERR_UNITTEST_EXCEPTION_CODE, "Call GetNext after data End");
    }
    auto &info = m_blocksInfo[m_usedBlockIndex++];
    switch (info.status)
    {
        case AriesOpNodeStatus::END:
            m_dataEnd = true;
        case AriesOpNodeStatus::CONTINUE:
        {
            auto table = m_tableBlockGenFunc(m_usedBlockIndex - 1, info.rowCount);
            if (table == nullptr)
            {
                throw AriesException( ERR_UNITTEST_EXCEPTION_CODE, "Generate Table is null");
            }
            return {info.status, move(table)};
        }
        case AriesOpNodeStatus::ERROR:
            m_dataEnd = true;
            return {info.status, nullptr};
        default:
            throw AriesException( ERR_UNITTEST_EXCEPTION_CODE, "Unsupported AriesOpNodeStatus: " + to_string(( int)info.status));
    }
}

bool PreFakeOpNode::SetBlocksInfo(vector<BlockInfo> &info)
{
    if (!info.empty() && info[info.size() - 1].status != AriesOpNodeStatus::CONTINUE)
    {
        m_blocksInfo.assign(info.cbegin(), info.cend());
        return true;
    }
    return false;
}

void PreFakeOpNode::SetGenDataInfo(GenDataFunc genDataFun)
{
    m_tableBlockGenFunc = genDataFun;
}

void PreFakeOpNode::SetOpenStatus(bool fakeStatus)
{
    m_openStatus = fakeStatus;
}
