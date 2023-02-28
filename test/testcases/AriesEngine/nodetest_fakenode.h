//
// Created by david.shen on 2019/11/26.
//

#ifndef AIRES_NODETEST_FAKENODE_H
#define AIRES_NODETEST_FAKENODE_H

#include "AriesEngine/AriesCommonExpr.h"
#include "AriesEngine/AriesOpNode.h"

using namespace aries_engine;
using namespace std;

struct BlockInfo {
    BlockInfo()
    {
        rowCount = -1;
        status = AriesOpNodeStatus::ERROR;
    }
    BlockInfo(int count, AriesOpNodeStatus fakeStatus)
    {
        rowCount = count;
        status = fakeStatus;
    }
    int rowCount;
    AriesOpNodeStatus status;
};

#define ERR_UNITTEST_EXCEPTION_CODE 99999999
/**
 * 定义数据产生回调函数类型
 * 使用者必须需要按照此函数方式定义各自测试node的相应函数
 *
 * */
typedef AriesTableBlockUPtr (* GenDataFunc)(int blockId, int rowCount);

/** @class PreFakeOpNode
 *  模拟被测的前一个Node
 * */
class PreFakeOpNode : public AriesOpNode
{
public:
    PreFakeOpNode()
    {
        m_openStatus = true;
        m_usedBlockIndex = 0;
        m_dataEnd = false;
        m_blocksInfo.clear();
    }

    ~PreFakeOpNode() final
    {
        m_blocksInfo.clear();
    }

    AriesOpResult GetNext() final;

    bool Open() final;
    void Close() final
    {
    }

    /**
     * SetGenDataInfo
     * 设置各列的AriesColumnType, columnId，以及各列数据生成的回调函数
     * 数据生成回调函数：使用者可以为各列定义一个GenDataFunc函数用于产生测试节点的前一节点数据
     * */
    void SetGenDataInfo(GenDataFunc genDataFun);

public:
    /**
     * 以下属性设置方法可以在unittest_framework里统一调用
     * */
    bool SetBlocksInfo(vector<BlockInfo> &info);
    void SetOpenStatus(bool fakeStatus);
private:
    bool m_openStatus;
    bool m_dataEnd;
    int m_usedBlockIndex;
    vector<BlockInfo> m_blocksInfo;
    AriesTableBlockUPtr m_tableUPtr;
    GenDataFunc m_tableBlockGenFunc;
};

using PreFakeOpNodeSPtr = shared_ptr<PreFakeOpNode>;
#endif //AIRES_NODETEST_FAKENODE_H
