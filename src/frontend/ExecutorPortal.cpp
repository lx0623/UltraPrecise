#include "ExecutorPortal.h"

namespace aries {

ExecutorPortal::ExecutorPortal( AbstractQueryEnginePointer arg_engine )
    : the_engine( arg_engine )
{
}

std::string ExecutorPortal::ToString() {
    return "ExecutorPortal";
}

AbstractMemTablePointer ExecutorPortal::ExecuteQuery(const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName) {
    return this->the_engine->ExecuteQueryTree(tx, arg_query_tree, dbName);
}

AbstractMemTablePointer ExecutorPortal::ExecuteInsert( const aries_engine::AriesTransactionPtr& tx,
                                                       const string& dbName,
                                                       const string& tableName,
                                                       vector< int >& insertColumnIds,
                                                       VALUES_LIST& insertValuesList,
                                                       vector< BiaodashiPointer >& optUpdateColumns,
                                                       vector< BiaodashiPointer >& optUpdateValues,
                                                       SQLTreeNodePointer queryPlanTree )
{
    return this->the_engine->ExecuteInsert( tx,
                                            dbName,
                                            tableName,
                                            insertColumnIds,
                                            insertValuesList,
                                            optUpdateColumns,
                                            optUpdateValues,
                                            queryPlanTree );
}
AbstractMemTablePointer ExecutorPortal::ExecuteUpdate( const aries_engine::AriesTransactionPtr& tx,
                                                       SQLTreeNodePointer arg_query_tree,
                                                       const string& targetDbName,
                                                       const std::string& defaultDbName )
{
    return this->the_engine->ExecuteUpdateTree(tx, arg_query_tree, targetDbName, defaultDbName);
}
AbstractMemTablePointer ExecutorPortal::ExecuteDelete( const aries_engine::AriesTransactionPtr& tx,
                                                       const string& dbName,
                                                       const string& tableName,
                                                       SQLTreeNodePointer arg_query_tree,
                                                       const string& defaultDbName )
{
    return this->the_engine->ExecuteDeleteTree( tx, dbName, tableName, arg_query_tree, defaultDbName );
}

}
