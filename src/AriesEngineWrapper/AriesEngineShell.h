/*
 * AriesEngineShell.h
 *
 *  Created on: Mar 13, 2019
 *      Author: lichi
 */

#pragma once

#include "../frontend/PhysicalTable.h"
#include "../frontend/ColumnShell.h"
#include "../frontend/AbstractBiaodashi.h"
#include "../AriesEngine/AriesNodeManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesEngineShell: protected DisableOtherConstructors
    {
    public:
        AriesEngineShell();
        ~AriesEngineShell();
    public:
        // static AriesScanNodeSPtr MakeScanNode( int nodeId, const string& dbName, PhysicalTablePointer arg_table, const vector< int >& arg_columns_id );
        static AriesMvccScanNodeSPtr MakeMvccScanNode( int nodeId, const AriesTransactionPtr& tx,
                                                       const string& dbName,
                                                       const string& tableName,
                                                       const vector< int >& arg_columns_id );
        static AriesFilterNodeSPtr MakeFilterNode( int nodeId, BiaodashiPointer arg_filter_expr, const vector< int >& arg_columns_id );
        static AriesGroupNodeSPtr MakeGroupNode( int nodeId, const vector< BiaodashiPointer >& arg_group_by_exprs,
                const vector< BiaodashiPointer >& arg_select_exprs );
        static AriesSortNodeSPtr MakeSortNode( int nodeId, const vector< BiaodashiPointer >& arg_order_by_exprs,
                const vector< OrderbyDirection >& arg_order_by_directions, const vector< int >& arg_columns_id );
        static AriesJoinNodeSPtr MakeJoinNode( int nodeId, BiaodashiPointer equal_join_expr, BiaodashiPointer other_join_expr, int arg_join_hint,
                bool arg_join_hint_2, const vector< int >& arg_columns_id );
        static AriesJoinNodeSPtr MakeJoinNodeComplex( int nodeId, BiaodashiPointer eqaul_join_expr, BiaodashiPointer other_join_expr,
                JoinType arg_join_type, const vector< int >& arg_columns_id );
        static AriesColumnNodeSPtr MakeColumnNode( int nodeId, const vector< BiaodashiPointer >& arg_select_exprs, const vector< int >& arg_columns_id,
                int arg_mode );
        static AriesUpdateCalcNodeSPtr MakeUpdateCalcNode( int nodeId, const vector< BiaodashiPointer >& arg_select_exprs, const vector< int >& arg_columns_id );
        static AriesOutputNodeSPtr MakeOutputNode();

        static AriesLimitNodeSPtr MakeLimitNode( int nodeId, int64_t offset, int64_t size );

        static AriesSetOperationNodeSPtr MakeSetOpNode( int nodeId, SetOperationType type );

        static AriesSelfJoinNodeSPtr MakeSelfJoinNode( int nodeId, int joinColumnId, CommonBiaodashiPtr filter_expr, const vector< HalfJoinInfo >& join_info, const vector< int >& arg_columns_id );

        static AriesExchangeNodeSPtr MakeExchangeNode( int nodeId, int dstDeviceId, const vector< int >& srcDeviceId );
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

