//
// Created by david.shen on 2019/11/19.
//

#ifndef AIRES_NODETEST_FRAMEWORK_H
#define AIRES_NODETEST_FRAMEWORK_H

#include "AriesEngine/AriesCommonExpr.h"
#include "AriesEngine/AriesOpNode.h"
#include "nodetest_fakenode.h"

using namespace aries_engine;
using namespace std;

/**
 * 定义数据验证的回调函数类型
 * 使用者必须需要按照此函数方式定义各自测试node的相应函数
 * */
typedef bool (* VerifyDataFunc)(const AriesOpResult &opResult);

/**
 * UnittestFrameWork
 * 有如下逻辑：
 * 1, 模拟设置被unittest opNode的前一个节点 @PreFakeOpNode
 * 2, 设置模拟情况
 *
 * */
class NodeTestFrameWork
{
public:
    NodeTestFrameWork();
    ~NodeTestFrameWork();
public:
    /**
     * 设置需要测试的Node
     * */
    void SetTestNodeOpNode(const AriesOpNodeSPtr &opNodeSPtr);
    /**
     * 设置需要测试的Node
     * */
    void SetTestNodeSource(PreFakeOpNodeSPtr leftSource, PreFakeOpNodeSPtr rightSource = nullptr);
    /**
     * SetVerifyDataInfo
     * 数据验证回调函数：使用者为一个Table定义VerifyDataFunc函数用于验证通过测试节点计算后的数据结果是否正确
     * */

    void SetVerifyDataInfo(VerifyDataFunc verifyDataFunc);
    /**
     * SetOpenStatus
     * 设置测试节点前置节点的Open方法返回值
     * */
    void SetOpenStatus(bool fakeStatus);
    bool SetBlocksInfo(vector<BlockInfo> &info);
    /**
     * Run
     * 运行测试用例
     * 返回值：true，表示用例测试成功
     * 返回值：true，表示用例测试失败
     * */
    bool Run();

private:
    AriesOpNodeSPtr m_testOpNodeSPtr;
    PreFakeOpNodeSPtr m_leftSource;
    PreFakeOpNodeSPtr m_rightSource;
    VerifyDataFunc m_verifyDataFunc;
    vector<BlockInfo> m_preFakeNodeBlocksInfo;
    bool m_openStatus;
    bool m_openStatusSet;
};

using UnittestTestFrameWorkSPtr = shared_ptr<NodeTestFrameWork>;

#endif //AIRES_NODETEST_FRAMEWORK_H
