/*
 * AriesNodeManager.h
 *
 *  Created on: Mar 10, 2019
 *      Author: lichi
 */

#pragma once

// #include "AriesScanNode.h"
#include "AriesMvccScanNode.h"
#include "AriesFilterNode.h"
#include "AriesGroupNode.h"
#include "AriesSortNode.h"
#include "AriesJoinNode.h"
#include "AriesColumnNode.h"
#include "AriesOutputNode.h"
#include "AriesLimitNode.h"
#include "AriesSetOperationNode.h"
#include "AriesUpdateCalcNode.h"
#include "AriesUpdateNode.h"
#include "AriesInsertNode.h"
#include "AriesDeleteNode.h"
#include "AriesSelfJoinNode.h"
#include "AriesExchangeNode.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesNodeManager: protected DisableOtherConstructors
    {
    public:
        AriesNodeManager();
        ~AriesNodeManager();

    public:
        // static AriesScanNodeSPtr MakeScanNode( int nodeId, const string& dbName, PhysicalTablePointer arg_table, const vector< int >& arg_columns_id );
        static AriesMvccScanNodeSPtr MakeMvccScanNode( int nodeId, const AriesTransactionPtr& tx,
                                                       const string& dbName,
                                                       const string& tableName,
                                                       const vector< int >& arg_columns_id );
        // static AriesRdbScanNodeSPtr MakeRdbScanNode( int nodeId, const string& dbName, PhysicalTablePointer arg_table, const vector< int >& arg_columns_id );
        static AriesFilterNodeSPtr MakeFilterNode( int nodeId, const AriesCommonExprUPtr& condition, const vector< int >& arg_columns_id );
        static AriesGroupNodeSPtr MakeGroupNode( int nodeId, const vector< AriesCommonExprUPtr >& groups, const vector< AriesCommonExprUPtr >& sels );
        static AriesSortNodeSPtr MakeSortNode( int nodeId, const vector< AriesCommonExprUPtr >& exprs, const vector< AriesOrderByType >& orders, const vector< int >& arg_columns_id );
        static AriesJoinNodeSPtr MakeJoinNode( int nodeId, AriesCommonExprUPtr equalCondition, AriesCommonExprUPtr otherCondition, AriesJoinType type, int joinHint, bool bIntact,
                const vector< int >& columnIds );
        static AriesJoinNodeSPtr MakeJoinNodeComplex( int nodeId, AriesCommonExprUPtr equalCondition, AriesCommonExprUPtr otherCondition, AriesJoinType type, const vector< int >& columnIds );
        static AriesColumnNodeSPtr MakeColumnNode( int nodeId, const vector< AriesCommonExprUPtr >& exprs, int mode, const vector< int >& columnIds );
        static AriesUpdateCalcNodeSPtr MakeUpdateCalcNode( int nodeId, const vector< AriesCommonExprUPtr >& exprs, const vector< int >& columnIds );
        static AriesOutputNodeSPtr MakeOutputNode();
        static AriesLimitNodeSPtr MakeLimitNode( int nodeId, int64_t offset, int64_t size );
        static AriesSetOperationNodeSPtr MakeSetOpNode( int nodeId, AriesSetOpType type );
        static AriesSelfJoinNodeSPtr MakeSelfJoinNode( int nodeId, int joinColumnId, const SelfJoinParams& joinParams, const vector< int >& arg_columns_id );
        static AriesExchangeNodeSPtr MakeExchangeNode( int nodeId, int dstDeviceId, const vector< int >& srcDeviceId );
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

