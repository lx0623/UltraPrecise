//
// Created by david.shen on 2019/11/14.
//

#include <gtest/gtest.h>
#include "nodetest_framework.h"
#include "AriesEngine/AriesSortNode.h"

AriesTableBlockUPtr GenIntData(int blockId, int rowCount)
{
    auto table = make_unique<AriesTableBlock>();
    if ( blockId == 0 )
    {
        auto column = make_shared<AriesColumn>();
        int source[] = { 9, 10, -1, 7, 0, -6, 5, 8, -3, 100 };
        AriesColumnType columnType(AriesDataType(AriesValueType::INT32, 1), false, false);
        auto dataBuf = make_shared<AriesDataBuffer>( columnType, rowCount );
        memcpy( ( void * ) dataBuf->GetData(), ( void * ) source, dataBuf->GetTotalBytes() );
        column->AddDataBuffer( dataBuf );
        table->AddColumn(1, column);
        return table;
    }
    return nullptr;
}

bool VerifyIntData(const AriesOpResult &opResult)
{
    if (opResult.Status == AriesOpNodeStatus::END)
    {
        int result[] = {-6, -3, -1, 0, 5, 7, 8, 9, 10, 100};
        auto columnBuf = opResult.TableBlock->GetColumnBuffer(1);
        return memcmp(columnBuf->GetData(), result, columnBuf->GetTotalBytes()) == 0;
    }
    return false;
}

TEST( sortnode, t_normal_int )
{
    AriesSortNodeSPtr sortNodeSPtr = std::make_shared<AriesSortNode>();
    vector<AriesOrderByType> orders;
    orders.push_back(AriesOrderByType::ASC);
    sortNodeSPtr->SetOrderbyType(orders);

    AriesExprType type = AriesExprType::COLUMN_ID;
    AriesExpressionContent content = 1;
    aries::AriesDataType data_type{AriesValueType::INT32, 1};
    AriesColumnType value_type{data_type, false, false};
    vector<AriesCommonExprUPtr> exprs;
    exprs.push_back(move(AriesCommonExpr::Create(type, content, value_type)));
    sortNodeSPtr->SetOrderbyExprs(exprs);

    auto preNodeSPtr = make_shared<PreFakeOpNode>();
    preNodeSPtr->SetGenDataInfo(GenIntData);
    auto frameWorkSPtr = make_shared<NodeTestFrameWork>();
    frameWorkSPtr->SetTestNodeOpNode( sortNodeSPtr );
    frameWorkSPtr->SetTestNodeSource( preNodeSPtr );
    vector<BlockInfo> temp;
    temp.emplace_back(10, AriesOpNodeStatus::END);
    frameWorkSPtr->SetBlocksInfo(temp);
    frameWorkSPtr->SetVerifyDataInfo(VerifyIntData);

    ASSERT_TRUE(frameWorkSPtr->Run());
}

AriesTableBlockUPtr GenIntData4Error(int blockId, int rowCount)
{
    return nullptr;
}

bool VerifyIntData4Error(const AriesOpResult &opResult)
{
    return opResult.Status == AriesOpNodeStatus::ERROR;
}

TEST( sortnode, t_error )
{
    AriesSortNodeSPtr sortNodeSPtr = std::make_shared<AriesSortNode>();
    vector<AriesOrderByType> orders;
    orders.push_back(AriesOrderByType::ASC);
    sortNodeSPtr->SetOrderbyType(orders);
    AriesExprType type = AriesExprType::COLUMN_ID;
    AriesExpressionContent content = 1;
    aries::AriesDataType data_type{AriesValueType::INT32, 1};
    AriesColumnType value_type{data_type, false, false};
    vector<AriesCommonExprUPtr> exprs;
    exprs.push_back(move(AriesCommonExpr::Create(type, content, value_type)));
    sortNodeSPtr->SetOrderbyExprs(exprs);

    auto preNodeSPtr = make_shared<PreFakeOpNode>();
    preNodeSPtr->SetGenDataInfo(GenIntData4Error);
    auto frameWorkSPtr = make_shared<NodeTestFrameWork>();
    frameWorkSPtr->SetTestNodeOpNode( sortNodeSPtr );
    frameWorkSPtr->SetTestNodeSource( preNodeSPtr );
    vector<BlockInfo> temp;
    temp.emplace_back(-1, AriesOpNodeStatus::ERROR);
    frameWorkSPtr->SetBlocksInfo(temp);
    frameWorkSPtr->SetVerifyDataInfo(VerifyIntData4Error);

    ASSERT_TRUE(frameWorkSPtr->Run());
}

AriesTableBlockUPtr GenIntData4Empty(int blockId, int rowCount)
{
    auto table = make_unique<AriesTableBlock>();
    if ( blockId == 0 )
    {
        auto column = make_shared<AriesColumn>();
        AriesColumnType columnType(AriesDataType(AriesValueType::INT32, 1), false, false);
        auto dataBuf = make_shared<AriesDataBuffer>( columnType, rowCount );
        table->AddColumn(1, column);
        return table;
    }
    return nullptr;
}

bool VerifyIntData4Empty(const AriesOpResult &opResult)
{
    if (opResult.Status == AriesOpNodeStatus::END)
    {
        return opResult.TableBlock->GetRowCount() == 0;
    }
}

TEST( sortnode, t_empty )
{
    AriesSortNodeSPtr sortNodeSPtr = std::make_shared<AriesSortNode>();
    vector<AriesOrderByType> orders;
    orders.push_back(AriesOrderByType::ASC);
    sortNodeSPtr->SetOrderbyType(orders);
    AriesExprType type = AriesExprType::COLUMN_ID;
    AriesExpressionContent content = 1;
    aries::AriesDataType data_type{AriesValueType::INT32, 1};
    AriesColumnType value_type{data_type, false, false};
    vector<AriesCommonExprUPtr> exprs;
    exprs.push_back(move(AriesCommonExpr::Create(type, content, value_type)));
    sortNodeSPtr->SetOrderbyExprs(exprs);

    auto preNodeSPtr = make_shared<PreFakeOpNode>();
    preNodeSPtr->SetGenDataInfo(GenIntData4Empty);
    auto frameWorkSPtr = make_shared<NodeTestFrameWork>();
    frameWorkSPtr->SetTestNodeOpNode( sortNodeSPtr );
    frameWorkSPtr->SetTestNodeSource( preNodeSPtr );
    vector<BlockInfo> temp;
    temp.emplace_back(0, AriesOpNodeStatus::END);
    frameWorkSPtr->SetBlocksInfo(temp);
    frameWorkSPtr->SetVerifyDataInfo(VerifyIntData4Empty);

    ASSERT_TRUE(frameWorkSPtr->Run());
}

TEST( sortnode, t_openfailed )
{
    AriesSortNodeSPtr sortNodeSPtr = std::make_shared<AriesSortNode>();

    auto preNodeSPtr = make_shared<PreFakeOpNode>();
    auto frameWorkSPtr = make_shared<NodeTestFrameWork>();
    frameWorkSPtr->SetTestNodeOpNode( sortNodeSPtr );
    frameWorkSPtr->SetTestNodeSource( preNodeSPtr );
    frameWorkSPtr->SetOpenStatus( false );

    ASSERT_TRUE(frameWorkSPtr->Run());
}