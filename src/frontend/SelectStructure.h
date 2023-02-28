#ifndef ARIES_SELECT_STRUCTURE
#define ARIES_SELECT_STRUCTURE

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

#include "TroubleHandler.h"
#include "VariousEnum.h"
#include "ShowUtility.h"

#include "AbstractQuery.h"
#include "AbstractBiaodashi.h"

#include "QueryContext.h"

#include "SchemaAgent.h"

#include "CommonBiaodashi.h"
#include "BasicRel.h"

#include "GroupbyStructure.h"
#include "JoinStructure.h"
#include "FromPartStructure.h"
#include "SelectPartStructure.h"
#include "OrderbyStructure.h"
#include "LimitStructure.h"

#include "SQLTreeNodeBuilder.h"


namespace aries {



/***************************************************************************
 *
 *                          SelectStructure
 *
 **************************************************************************/



/*This represent the select query*/

class SelectStructure : public AbstractQuery {

private:


    SelectStructure(const SelectStructure &arg);

    SelectStructure &operator=(const SelectStructure &arg);


    QueryContextPointer query_context;

    RelationStructurePointer rel_structure;

    std::unordered_map<std::string, bool> schema_agents_need_rowid_column;
    std::unordered_map<std::string, SchemaAgentPointer> schema_agents;
    SchemaAgentPointer default_schema_agent;
    SQLTreeNodeBuilderPointer node_builder;

/*Two cases: (1) SetQuery: query setop query ; (2) SimpleQuery: just a select*/
    bool is_set_query = false;
    bool is_group_or_sort_node_removable = false;

    /*for SetQuery*/
    SetOperationType set_operation_type;
    AbstractQueryPointer left_part;
    AbstractQueryPointer right_part;


    /*for SimpleQuery*/

    SelectPartStructurePointer select_part;

    FromPartStructurePointer from_part;

    BiaodashiPointer where_part;

    GroupbyStructurePointer groupby_part;

    OrderbyStructurePointer orderby_part;


    /*array and map for all tables*/
    std::vector<BasicRelPointer> from_table_array;
    std::map<std::string, BasicRelPointer> from_table_map;


    /*given an expr, where I can check -- a list of tables*/
    std::map<int, std::vector<BasicRelPointer>> map_expr_tables;


    std::vector<BiaodashiPointer> checked_expr_array;
    std::vector<std::shared_ptr<std::string>> checked_alias_array;

    std::vector<ColumnShellPointer> ambiguous_column_array;
    std::vector<ColumnShellPointer> ambiguous_alias_column_array;

    /*for building query plan tre*/
    SQLTreeNodePointer query_plan_tree = nullptr;
    // std::map<BasicRelPointer, SQLTreeNodePointer> rel_node_map;
    SQLTreeNodePointer tree_node_from = nullptr;

    bool selectpart_has_been_processed = false;

    /*do I have group by explicility or implicitly (ie.e, select sum() from...)?*/
    bool need_groupby_operation = false;

    std::vector<ColumnShellPointer> referenced_column_array_in_select;
    std::vector<ColumnShellPointer> referenced_column_array_in_groupby;

    std::vector<ColumnShellPointer> referenced_column_array_in_orderby;

    bool agg_in_select = false;

    bool am_i_in_exist = false;


    bool do_i_still_exist = true;

    bool is_distinct = false;


    /*a query can have multiple init_query_plan_trees, for uncorrelated subquery*/

    std::vector< std::shared_ptr< SelectStructure > > init_queries;
    std::vector<BiaodashiPointer> init_replace_exprs;
    std::vector< AbstractQueryPointer > sub_queries;

    std::map< std::string, std::string > spool_alias_map;

    SQLTreeNodePointer inner_join_node = nullptr;

    LimitStructurePointer limit_structure = nullptr;

    SQLTreeNodePointer BuildTreeNodeForJoinStructure( int join_structure_index, BiaodashiJoinTreeNodePointer expr_node, int& expr_index );

public:

    SelectStructure();

    void SimplifyExprs();

    ~SelectStructure();

    void SetQueryContext(QueryContextPointer arg);

    QueryContextPointer GetQueryContext();


    std::vector<BasicRelPointer> GetFromTableArray();

    std::vector<BiaodashiPointer> GetGroupbyList();


    void SetRelationStructure(RelationStructurePointer arg);

    RelationStructurePointer GetRelationStructure();

    bool IsSetQuery();

    void init_set_query(SetOperationType arg_type,
                        AbstractQueryPointer arg_left_part,
                        AbstractQueryPointer arg_right_part);


    void init_simple_query(SelectPartStructurePointer arg_select_part,
                           FromPartStructurePointer arg_from_part,
                           BiaodashiPointer arg_where_part,
                           GroupbyStructurePointer arg_groupby_part,
                           OrderbyStructurePointer arg_orderby_part);

    bool DoIHaveGroupBy();

    std::string ToString();

    void CheckQueryGate(SchemaAgentPointer arg_agent, SQLTreeNodeBuilderPointer arg_node_builder,
                        QueryContextPointer arg_query_context);

    bool CheckQueryGate2(SchemaAgentPointer arg_agent, SQLTreeNodeBuilderPointer arg_node_builder,
                         QueryContextPointer arg_query_context);

    bool CheckQuery();


    bool CheckSetQuery();

    bool CheckSimpleQuery();


    bool CheckQueryParts();


    bool CheckFromPart();


    void CheckExprGate(BiaodashiPointer arg_expr, ExprContextPointer arg_expr_context);

    void SetMapExprTables(int arg_index, std::vector<BasicRelPointer> arg_rel_array);

    std::vector<BasicRelPointer> GetFromMapExprTables(int arg_index);


    void SetDefaultSchemaAgent(SchemaAgentPointer arg_schema_agent);

    SchemaAgentPointer GetDefaultSchemaAgent();
    SchemaAgentPointer GetSchemaAgent(const string& name);
    bool IsSchemaAgentNeedRowIdColumn(const string& name);
    void SetSchemaAgentNeedRowIdColumn(const string& name, bool b);

    /*Two things: setup relation_structure and create a tree node.*/
    void CheckBasicRel(BasicRelPointer arg_rel, QueryContextPointer arg_query_context);

    PhysicalTablePointer GetTableByName(const BasicRelPointer& arg_rel);

    bool CheckWherePart();

    bool CheckGroupPart();

    bool CheckOrderPart();

    bool CheckSelectPart();

    bool BuildQueryPlanTreeMain();

    bool AddFromTable(BasicRelPointer arg_table);


    void CheckSelectExprGate(BiaodashiPointer arg_expr, std::shared_ptr<std::string> arg_alias_p);

    std::vector<BiaodashiPointer> GetAllColumnsFromGivenTable(std::string arg_table_name);

    std::vector<BiaodashiPointer> GetAllColumnsFromBasicRel(BasicRelPointer arg_brp);


    GroupbyStructurePointer GetGroupbyPart();

    OrderbyStructurePointer GetOrderbyPart();

    SelectPartStructurePointer GetSelectPart();


    void SetSQLTreeNodeBuilder(SQLTreeNodeBuilderPointer arg_node_builder);

    SQLTreeNodeBuilderPointer GetSQLTreeNodeBuilder();

    SQLTreeNodePointer GetQueryPlanTree();

    void SetQueryPlanTree(SQLTreeNodePointer arg_node);

    void BuildFromTreeAddInnerJoinNode( SQLTreeNodePointer arg_new_node, JoinType arg_join_type, BiaodashiPointer arg_expr );

    SQLTreeNodePointer BuildSubTreeAddNode( SQLTreeNodePointer arg_root_node, SQLTreeNodePointer arg_new_node, JoinType arg_join_type, BiaodashiPointer arg_expr, bool b_new_node_as_left_child = false );

    std::string GetTableNameOfSelectItemIfColumn(size_t i);

    bool GetSelectpartProcessed();

    void SetSelectpartProcessed(bool arg_value);


    bool GetNeedGroupbyOperation();

    void AddReferencedColumnIntoSelectPartArray();

    void AddReferencedColumnIntoGroupbyPartArray();

    void AddReferencedColumnIntoOrderbyPartArray();

    std::vector<ColumnShellPointer> GetReferencedColumnInSelect();

    std::vector<ColumnShellPointer> GetReferencedColumnInGroupby();

    std::vector<ColumnShellPointer> GetReferencedColumnInOrderby();

    void SetIAmInExist(bool arg_value);

    bool GetIAmInExist();

    std::vector<BiaodashiPointer> GetAllSelectExprs();

    std::vector<std::shared_ptr<std::string>> GetAllSelectAlias();

    bool DoIStillExist();

    void SetDoIStillExist(bool arg_value);


    void AddInitQueryAndExpr( const std::shared_ptr< SelectStructure >& arg_ss, BiaodashiPointer arg_bp );

    int GetInitQueryCount();

    std::shared_ptr< SelectStructure > GetInitQueryByIndex(int arg_index);

    BiaodashiPointer GetReplaceExprByIndex(int arg_index);

    void AddExtraExpr(BiaodashiPointer arg_expr);

    void AddAliasForTheOnlyExpr(std::string arg_value);


    int LocateExprInSelectList(BiaodashiPointer arg_expr);

    int GetAllExprCount();

    const LimitStructurePointer &GetLimitStructure() const;

    void SetLimitStructure(const LimitStructurePointer &limit_structure);

    FromPartStructurePointer& GetFromPart() {
        return from_part;
    }
    void SetDistinct(bool value);

    bool IsDistinct() const;

    void SetOrderbyPart(const OrderbyStructurePointer& orderby_part);

    SelectPartStructurePointer& GetOriginalSelectPart();
    // ambiguous_expr_array
    void AddAmbiguousColumn(const ColumnShellPointer& column);

    void AddAmbiguousAliasColumn(const ColumnShellPointer& column);

    void AddSubQuery( const AbstractQueryPointer& query );

    const std::vector< AbstractQueryPointer >& GetSubQueries() const;

    BiaodashiPointer GetWhereCondition();

    void SetSpoolAlias( const std::string left, const std::string right );
    const std::map< std::string, std::string >& GetSpoolAlias() const;

private:
    void handleAmbiguousAlias(const std::vector<BiaodashiPointer>& exprs);
};

typedef std::shared_ptr<SelectStructure> SelectStructurePointer;


} //namespace aries



#endif
