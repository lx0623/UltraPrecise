/*
 * AriesEngineByMaterialization.h
 *
 *  Created on: Mar 13, 2019
 *      Author: lichi
 */

#ifndef ARIESENGINEBYMATERIALIZATION_H_
#define ARIESENGINEBYMATERIALIZATION_H_

#include "SQLTreeNode.h"
#include "AbstractQueryEngine.h"
#include "../AriesEngineWrapper/AriesEngineShell.h"
#include "../AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngine/AriesStarJoinNode.h"
#include "AriesEngine/AriesExchangeNode.h"
#include "frontend/nlohmann/json.hpp"

using namespace aries_engine;
using JSON = nlohmann::json;

namespace aries {

class AriesEngineByMaterialization : public AbstractQueryEngine {
public:
    AriesEngineByMaterialization( bool isUpdate = false );

    ~AriesEngineByMaterialization();

public:
    AbstractMemTablePointer ExecuteQueryTree(const AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName) override final;
    AbstractMemTablePointer ExecuteUpdateTree( const aries_engine::AriesTransactionPtr& tx,
                                               SQLTreeNodePointer arg_query_tree,
                                               const string& targetDbName,
                                               const std::string& defaultDbName );
    AbstractMemTablePointer ExecuteInsert( const aries_engine::AriesTransactionPtr& tx,
                                           const string& dbName,
                                           const string& tableName,
                                           vector< int >& insertColumnIds,
                                           VALUES_LIST& insertValuesList,
                                           vector< BiaodashiPointer >& optUpdateColumnExprs,
                                           vector< BiaodashiPointer >& optUpdateValueExprs,
                                           SQLTreeNodePointer queryPlanTree );
    AbstractMemTablePointer ExecuteDeleteTree( const aries_engine::AriesTransactionPtr& tx,
                                               const string& dbName,
                                               const string& tableName,
                                               SQLTreeNodePointer arg_query_tree,
                                               const string& defaultDbName );

    string ToString() override final;

private:
    AriesMemTableSPtr ExecuteQueryPipeline( AriesOpNodeSPtr pipeline );
    AbstractMemTablePointer ExecuteQuery(const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName);

    void ExecuteUncorrelatedSubQueries(const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName);

private:
    AriesOpNodeSPtr CreateOpNode(const aries_engine::AriesTransactionPtr& tx,
                                 SQLTreeNodePointer arg_input_node,
                                 const std::string& dbName,
                                 int& nodeId );

    AriesOpNodeSPtr CreateScanNode(const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_input_node, const std::string& defaultDbName, int nodeId );

    AriesFilterNodeSPtr CreateFilterNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesGroupNodeSPtr CreateGroupNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesSortNodeSPtr CreateSortNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesJoinNodeSPtr CreateJoinNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesColumnNodeSPtr CreateColumnNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesUpdateCalcNodeSPtr CreateUpdateCalcNode( SQLTreeNodePointer arg_input_node, int nodeId );

    AriesLimitNodeSPtr CreateLimitNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesSetOperationNodeSPtr CreateSetOpNode(SQLTreeNodePointer arg_input_node, int nodeId);

    AriesOutputNodeSPtr CreateOutputNode(AriesOpNodeSPtr dataSource);

    AriesSelfJoinNodeSPtr CreateSelfJoinNode( SQLTreeNodePointer arg_input_node, int nodeId );
    AriesStarJoinNodeSPtr CreateStarJoinNode( const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_input_node, const std::string& dbName, int& nodeId );

    AriesExchangeNodeSPtr CreateExchangeNode( const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_input_node, const std::string& db_name, int nodeId );

private:
    void ProcessExprByUsingColumnIndex(BiaodashiPointer arg_expr, SQLTreeNodePointer arg_input_node);

    void ProcessColumn(CommonBiaodashi *p_expr, SQLTreeNodePointer arg_input_node);

    void checkJoinConditions( BiaodashiPointer expr, SQLTreeNodePointer arg_input_node, bool b_check_equal_condition );

private:
    bool m_isUpdate;
    bool m_isDelete;
    // target database and table name for update or delete
    string m_targetDbName;
    string m_targetTableName;

#ifdef ARIES_PROFILE
    JSON m_perfStat;
#endif

    AriesSpoolCacheManagerSPtr spool_cache_manager;
};

} /* namespace aries */

#endif /* ARIESENGINEBYMATERIALIZATION_H_ */
