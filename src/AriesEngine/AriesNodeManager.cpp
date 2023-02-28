/*
 * AriesNodeManager.cpp
 *
 *  Created on: Mar 10, 2019
 *      Author: lichi
 */

#include "AriesNodeManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesNodeManager::AriesNodeManager()
    {
        // TODO Auto-generated constructor stub

    }

    AriesNodeManager::~AriesNodeManager()
    {
        // TODO Auto-generated destructor stub
    }

    AriesOutputNodeSPtr AriesNodeManager::MakeOutputNode()
    {
        return make_shared< AriesOutputNode >();
    }

    // AriesScanNodeSPtr AriesNodeManager::MakeScanNode( int nodeId, const string& dbName, PhysicalTablePointer arg_table, const vector< int >& arg_columns_id )
    // {
    //     AriesScanNodeSPtr node = make_shared< AriesScanNode >( dbName );
    //     node->SetNodeId( nodeId );
    //     node->SetPhysicalTable( arg_table );
    //     node->SetOutputColumnIds( arg_columns_id );
    //     return node;
    // }

    AriesMvccScanNodeSPtr AriesNodeManager::MakeMvccScanNode( int nodeId, const AriesTransactionPtr& tx, const string& dbName, const string& tableName,
            const vector< int >& arg_columns_id )
    {
        AriesMvccScanNodeSPtr node = make_shared< AriesMvccScanNode >( tx, dbName, tableName );
        node->SetNodeId( nodeId );
        node->SetOutputColumnIds( arg_columns_id );
        return node;
    }

    AriesFilterNodeSPtr AriesNodeManager::MakeFilterNode( int nodeId, const AriesCommonExprUPtr& condition, const vector< int >& arg_columns_id )
    {
        AriesFilterNodeSPtr node = make_shared< AriesFilterNode >();
        node->SetNodeId( nodeId );
        node->SetCondition( condition );
        node->SetOutputColumnIds( arg_columns_id );
        return node;
    }

    AriesGroupNodeSPtr AriesNodeManager::MakeGroupNode( int nodeId, const vector< AriesCommonExprUPtr >& groups, const vector< AriesCommonExprUPtr >& sels )
    {
        AriesGroupNodeSPtr node = make_shared< AriesGroupNode >();
        node->SetNodeId( nodeId );
        node->SetGroupByExprs( groups );
        node->SetSelectionExprs( sels );
        return node;
    }

    AriesSortNodeSPtr AriesNodeManager::MakeSortNode( int nodeId, const vector< AriesCommonExprUPtr >& exprs, const vector< AriesOrderByType >& orders,
            const vector< int >& arg_columns_id )
    {
        AriesSortNodeSPtr node = make_shared< AriesSortNode >();
        node->SetNodeId( nodeId );
        node->SetOrderbyExprs( exprs );
        node->SetOrderbyType( orders );
        node->SetOutputColumnIds( arg_columns_id );
        return node;
    }

    AriesJoinNodeSPtr AriesNodeManager::MakeJoinNode( int nodeId, AriesCommonExprUPtr equalCondition, AriesCommonExprUPtr otherCondition,
            AriesJoinType type, int joinHint, bool bIntact, const vector< int >& columnIds )
    {
        AriesJoinNodeSPtr node = make_shared< AriesJoinNode >();
        node->SetNodeId( nodeId );
        node->SetCondition( std::move( equalCondition ), std::move( otherCondition ), type );
        node->SetOutputColumnIds( columnIds );
        node->SetJoinHint( joinHint, bIntact );
        return node;
    }

    AriesSetOperationNodeSPtr AriesNodeManager::MakeSetOpNode( int nodeId, AriesSetOpType type )
    {
        AriesSetOperationNodeSPtr node = make_shared< AriesSetOperationNode >();
        node->SetNodeId( nodeId );
        node->SetOpType( type );
        return node;
    }

    AriesJoinNodeSPtr AriesNodeManager::MakeJoinNodeComplex( int nodeId, AriesCommonExprUPtr equalCondition, AriesCommonExprUPtr otherCondition,
            AriesJoinType type, const vector< int >& columnIds )
    {
        AriesJoinNodeSPtr node = make_shared< AriesJoinNode >();
        node->SetNodeId( nodeId );
        node->SetCondition( std::move( equalCondition ), std::move( otherCondition ), type );
        node->SetOutputColumnIds( columnIds );
        return node;
    }

    AriesColumnNodeSPtr AriesNodeManager::MakeColumnNode( int nodeId, const vector< AriesCommonExprUPtr >& exprs, int mode, const vector< int >& columnIds )
    {
        AriesColumnNodeSPtr node = make_shared< AriesColumnNode >();
        node->SetNodeId( nodeId );
        node->SetColumnExprs( exprs );
        node->SetOutputColumnIds( columnIds );
        node->SetExecutionMode( mode );
        return node;
    }

    AriesUpdateCalcNodeSPtr AriesNodeManager::MakeUpdateCalcNode( int nodeId, const vector< AriesCommonExprUPtr >& exprs, const vector< int >& columnIds )
    {
        AriesUpdateCalcNodeSPtr node = make_shared< AriesUpdateCalcNode >();
        node->SetNodeId( nodeId );
        node->SetCalcExprs( exprs );
        node->SetColumnIds( columnIds );
        return node;
    }

    AriesLimitNodeSPtr AriesNodeManager::MakeLimitNode( int nodeId, int64_t offset, int64_t size )
    {
        AriesLimitNodeSPtr node = make_shared< AriesLimitNode >();
        node->SetNodeId( nodeId );
        node->SetLimitInfo( offset, size );
        return node;
    }

    AriesSelfJoinNodeSPtr AriesNodeManager::MakeSelfJoinNode( int nodeId, int joinColumnId, const SelfJoinParams& joinParams, const vector< int >& arg_columns_id )
    {
        AriesSelfJoinNodeSPtr node = make_shared< AriesSelfJoinNode >();
        node->SetNodeId( nodeId );
        node->SetOutputColumnIds( arg_columns_id );
        node->SetJoinInfo( joinColumnId, joinParams );
        return node;
    }

    AriesExchangeNodeSPtr AriesNodeManager::MakeExchangeNode( int nodeId, int dstDeviceId, const vector< int >& srcDeviceId )
    {
        AriesExchangeNodeSPtr node = make_shared< AriesExchangeNode >();
        node->SetNodeId( nodeId );
        node->SetDispatchInfo( dstDeviceId, srcDeviceId );
        return node;
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
