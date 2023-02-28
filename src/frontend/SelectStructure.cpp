#include <stdio.h>
#include <glog/logging.h>
#include <server/mysql/include/derror.h>
#include <server/mysql/include/mysqld_error.h>
#include "server/mysql/include/mysqld.h"
#include <schema/SchemaManager.h>
#include <stack>
#include "AriesAssert.h"
#include "SelectStructure.h"
#include "AriesAssert.h"
#include "SchemaBuilder.h"
#include "ViewManager.h"
#include "BiaodashiAuxProcessor.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"


namespace aries {

SelectStructure::SelectStructure() {
}

SelectStructure::~SelectStructure() {
}

void SelectStructure::SimplifyExprs()
{
    THD* thd = current_thd;
    aries_engine::ExpressionSimplifier exprSimplifier( true );
    if ( from_part )
    {
        std::vector<JoinStructurePointer> fp = this->from_part->GetFromList();
        for (size_t i = 0; i < fp.size(); i++ )
            fp[i]->SimplifyExprs( exprSimplifier, thd );
    }

    if ( where_part )
    {
        auto commonExpr = ( CommonBiaodashi* )where_part.get();
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            where_part = simplifiedExpr;
    }
    if ( groupby_part )
        groupby_part->SimplifyExprs( exprSimplifier, thd );
    
    if ( orderby_part )
        orderby_part->SimplifyExprs( exprSimplifier, thd );
    
    select_part->SimplifyExprs( exprSimplifier, thd );
}

void SelectStructure::SetQueryContext(QueryContextPointer arg) {
    this->query_context = arg;
}

QueryContextPointer SelectStructure::GetQueryContext() {
    return this->query_context;
}


std::vector<BasicRelPointer> SelectStructure::GetFromTableArray() {
    return this->from_table_array;
}

std::vector<BiaodashiPointer> SelectStructure::GetGroupbyList() {
    std::vector<BiaodashiPointer> gbl;
    if (this->groupby_part) {
        gbl = this->groupby_part->GetGroupbyExprs();
    }
    return gbl;
}


void SelectStructure::SetRelationStructure(RelationStructurePointer arg) {
    this->rel_structure = arg;
}

RelationStructurePointer SelectStructure::GetRelationStructure() {
    return this->rel_structure;
}


bool SelectStructure::IsSetQuery() {
    return this->is_set_query;
}

void SelectStructure::init_set_query(SetOperationType arg_type,
                                     AbstractQueryPointer arg_left_part,
                                     AbstractQueryPointer arg_right_part) {
    this->is_set_query = true;

    this->set_operation_type = arg_type;
    this->left_part = (arg_left_part);
    this->right_part = (arg_right_part);

}


void SelectStructure::init_simple_query(SelectPartStructurePointer arg_select_part,
                                        FromPartStructurePointer arg_from_part,
                                        BiaodashiPointer arg_where_part,
                                        GroupbyStructurePointer arg_groupby_part,
                                        OrderbyStructurePointer arg_orderby_part) {

    this->is_set_query = false;

    ARIES_ASSERT(arg_select_part != nullptr, "select part should not be null");
    this->select_part = (arg_select_part);

    this->from_part = (arg_from_part);
    this->where_part = (arg_where_part);
    this->groupby_part = (arg_groupby_part);
    this->orderby_part = (arg_orderby_part);
}

bool SelectStructure::DoIHaveGroupBy() {
    return (this->groupby_part != nullptr);
}

std::string SelectStructure::ToString() {

    if (this->query_plan_tree != nullptr)
        return this->query_plan_tree->ToString(0);


    std::string result = "";

    if (this->is_set_query == true) {
        result += this->left_part->ToString();
        result += "\nSET_OPERATION: " + std::to_string(int(this->set_operation_type));
        result += this->right_part->ToString();
    } else {

        result += "SELECT\n\t";
        ARIES_ASSERT(this->select_part != nullptr, "select part should not be empty");
        result += this->select_part->ToString();

        if (this->from_part != nullptr) {
            result += "\nFROM\n\t";
            result += this->from_part->ToString();
        }

        if (this->where_part != nullptr) {
            result += "\nWHERE\n\t";
            result += this->where_part->ToString();
        }

        if (this->groupby_part != nullptr) {
            result += "\nGROUP BY\n\t";
            result += this->groupby_part->ToString();
        }

        if (this->orderby_part != nullptr) {
            result += "\nORDER BY\n\t";
            result += this->orderby_part->ToString();
        }

        if (limit_structure) {
            result += "\nLimit " + std::to_string(limit_structure->Offset) + ", " + std::to_string(limit_structure->Limit);
        }
    }

    //std::cout << "ss->ToString(): " << result << "\n";

    return result;
}


//-----------------------------------------------------------------------------------------
void SelectStructure::CheckQueryGate(SchemaAgentPointer arg_agent, SQLTreeNodeBuilderPointer arg_node_builder,
                                     QueryContextPointer arg_query_context) {

    this->CheckQueryGate2(arg_agent, arg_node_builder, arg_query_context);

}


bool SelectStructure::CheckQueryGate2(SchemaAgentPointer arg_agent, SQLTreeNodeBuilderPointer arg_node_builder,
                                      QueryContextPointer arg_query_context) {

    this->SetDefaultSchemaAgent(arg_agent);
    this->SetSQLTreeNodeBuilder(arg_node_builder);
    this->SetQueryContext(arg_query_context);

    this->CheckQuery();

    return true;

}

bool SelectStructure::CheckQuery() {
    return this->IsSetQuery() ? this->CheckSetQuery() : this->CheckSimpleQuery();
}

bool SelectStructure::CheckSetQuery() {
    // now we have to check both left part and right part, and compare their outputs.

    //std::cout << "SelectStructure::CheckSetQuery begin\n";


    //check left
    //std::cout << "CheckSetQuery(): left part check begin\n";
    SelectStructure *ss_left_p = (SelectStructure *) ((this->left_part).get());
    SQLTreeNodeBuilderPointer left_node_builder = std::make_shared<SQLTreeNodeBuilder>(this->left_part);
    QueryContextPointer left_query_context = std::make_shared<QueryContext>(QueryContextType::TheTopQuery,
                                                                            0,
                                                                            this->left_part, //query
                                                                            nullptr, //parent
                                                                            nullptr //possible expr
    );

    ss_left_p->CheckQueryGate2(this->default_schema_agent, left_node_builder, left_query_context);
    //std::cout << "CheckSetQuery(): left part check done\n";

    //check right
    //std::cout << "CheckSetQuery(): right part check begin\n";
    SelectStructure *ss_right_p = (SelectStructure *) ((this->right_part).get());
    SQLTreeNodeBuilderPointer right_node_builder = std::make_shared<SQLTreeNodeBuilder>(this->right_part);
    QueryContextPointer right_query_context = std::make_shared<QueryContext>(QueryContextType::TheTopQuery,
                                                                             0,
                                                                             this->right_part, //query
                                                                             nullptr, //parent
                                                                             nullptr //possible expr
    );

    ss_right_p->CheckQueryGate2(this->default_schema_agent, right_node_builder, right_query_context);
    //std::cout << "CheckSetQuery(): right part check done\n";


    //compare left&right
    RelationStructurePointer rsp_left = ss_left_p->GetRelationStructure();
    RelationStructurePointer rsp_right = ss_right_p->GetRelationStructure();

    ARIES_ASSERT(rsp_left != nullptr, "left relation should not be null");
    ARIES_ASSERT(rsp_right != nullptr, "right relation should not be null");

    bool rs_matching = true;
    if (rsp_left->GetColumnCount() == rsp_right->GetColumnCount()) {
        for (size_t i = 0; i < rsp_left->GetColumnCount(); i++) {
            bool see_matching = false;
            if (rsp_left->GetColumn(i) != nullptr && rsp_right->GetColumn(i) != nullptr) {
                if (rsp_left->GetColumn(i)->GetValueType() == rsp_right->GetColumn(i)->GetValueType()) {
                    see_matching = true;
                }
            }

            if (!see_matching) {
                rs_matching = false;
                break;
            }


        }
    } else {
        rs_matching = false;
    }

    if (!rs_matching) {
        ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "different types from the two sides of UNION/INTERSECT/EXCEPT!" );
    }



    //setup my own things!
    this->rel_structure = rsp_left;

    //build the query plan
    SQLTreeNodePointer left_tree = ss_left_p->GetQueryPlanTree();
    SQLTreeNodePointer right_tree = ss_right_p->GetQueryPlanTree();

    SQLTreeNodePointer set_node = this->node_builder->makeTreeNode_SetOperation(this->set_operation_type);
    set_node->AddChild(left_tree);
    left_tree->SetParent( set_node );
    set_node->AddChild(right_tree);
    right_tree->SetParent( set_node );

    SQLTreeNodePointer column_node = this->node_builder->makeTreeNode_Column();
    column_node->SetTheChild(set_node);
    set_node->SetParent( column_node );

    this->query_plan_tree = column_node;

    //std::cout << "I have built my query plan tree -- Set Operatioin query\n";


    return true;
}

bool SelectStructure::CheckSimpleQuery() {

    SimplifyExprs();
    this->CheckQueryParts();
    this->BuildQueryPlanTreeMain();

    return true;
}

bool SelectStructure::CheckQueryParts() {

    /*First, we need to process the FROM part*/

    this->CheckFromPart();

    /*where cannot see those in select*/

    this->CheckWherePart();


    this->CheckGroupPart();


    this->CheckOrderPart();

    this->CheckSelectPart();


    return true;
}

SQLTreeNodePointer SelectStructure::BuildTreeNodeForJoinStructure( int join_structure_index, BiaodashiJoinTreeNodePointer expr_node, int& expr_index )
{
    SQLTreeNodePointer result;
    BiaodashiPointer the_expr = expr_node->self;
    std::vector< BasicRelPointer > join_rel_array = expr_node->ref_rel_array;
    if( the_expr )
    {
        assert( !join_rel_array.empty() );
        int the_expr_index = join_structure_index * 1000 + expr_index;
        this->SetMapExprTables( the_expr_index, join_rel_array );
        //cout<<"BuildTreeNodeForJoinStructure the_expr:"<<the_expr->ToString()<<", index:"<<the_expr_index<<endl;
        ExprContextPointer on_expr_context = std::make_shared< ExprContext >(
                ExprContextType::JoinOnExpr, /*exprcontext type*/
                the_expr, /*myself*/
                this->query_context, /*querycontext type*/
                this->query_context->expr_context, /*the parent expr context == my quest contxt's expr context ???REALLY USEFUL???*/
                the_expr_index);

        this->CheckExprGate( the_expr, on_expr_context );
        result = this->BuildSubTreeAddNode( BuildTreeNodeForJoinStructure( join_structure_index, expr_node->left, ++expr_index ), BuildTreeNodeForJoinStructure( join_structure_index, expr_node->right, ++expr_index ), expr_node->join_type, 
                                                            the_expr, false );
    }
    else 
    {
        assert( join_rel_array.size() == 1 );
        result = join_rel_array[ 0 ]->GetMyRelNode();
    }
    return result;
}

bool SelectStructure::CheckFromPart() 
{
    if( !this->from_part )
        return true;

    std::vector< JoinStructurePointer > fp = this->from_part->GetFromList();
    BiaodashiAuxProcessor processor;
    for( size_t i = 0; i < fp.size(); ++i ) 
    {
        JoinStructurePointer a_from_item = fp[ i ];
        assert( a_from_item->GetRelCount() > 0 );
        for( int index = 0; index < a_from_item->GetRelCount(); ++index ) 
        {
            BasicRelPointer a_join_rel = a_from_item->GetJoinRel( index );
            this->CheckBasicRel( a_join_rel, this->query_context );
            this->AddFromTable( a_join_rel );
        }
        int expr_index = 0;
        SQLTreeNodePointer new_node = this->BuildTreeNodeForJoinStructure( i, a_from_item->GetJoinExprTree(), expr_index );
        //cout<<new_node->ToString(0)<<endl;
        this->BuildFromTreeAddInnerJoinNode( new_node, JoinType::InnerJoin, processor.make_biaodashi_boolean( true ) );
    }

    return true;
}


void SelectStructure::CheckExprGate(BiaodashiPointer arg_expr, ExprContextPointer arg_expr_context) {
    ARIES_ASSERT(arg_expr != nullptr, "expression to check should not be null");

    std::shared_ptr<CommonBiaodashi> expr = std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr);

    expr->CheckExpr(arg_expr_context);

    /*post-work*/
    expr->CheckExprPostWork(arg_expr_context);

}


void SelectStructure::SetMapExprTables(int arg_index, std::vector<BasicRelPointer> arg_rel_array) {

    this->map_expr_tables.insert(std::make_pair(arg_index, arg_rel_array));

}

std::vector<BasicRelPointer> SelectStructure::GetFromMapExprTables(int arg_index) {
    //todo handling exception

    //try to find and then report error

    return this->map_expr_tables[arg_index];
}


void SelectStructure::SetDefaultSchemaAgent(SchemaAgentPointer arg_schema_agent) {
    // assert(arg_schema_agent != nullptr);
    default_schema_agent = arg_schema_agent;
    if ( arg_schema_agent )
    {
        schema_agents[ arg_schema_agent->schema->GetName() ] = default_schema_agent;
    }
}

SchemaAgentPointer SelectStructure::GetDefaultSchemaAgent() {
    return default_schema_agent;
}
void SelectStructure::SetSchemaAgentNeedRowIdColumn(const string& name, bool b)
{
    schema_agents_need_rowid_column[ name ] = b;
}
bool SelectStructure::IsSchemaAgentNeedRowIdColumn(const string& name)
{
    auto it = schema_agents_need_rowid_column.find( name );
    if ( schema_agents_need_rowid_column.end() == it )
        return false;
    return it->second;
}
SchemaAgentPointer SelectStructure::GetSchemaAgent(const string& schemaName) {
    auto it = schema_agents.find(schemaName);
    if (schema_agents.end() == it) {
        auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(schemaName);
        if (database) {
            auto schema_p = SchemaBuilder::BuildFromDatabase( database.get(), IsSchemaAgentNeedRowIdColumn( schemaName ) );
            SchemaAgentPointer schema_agent = std::make_shared<SchemaAgent>();
            schema_agent->SetDatabaseSchema(schema_p);
            schema_agents[schemaName] = schema_agent;
            return schema_agent;
        }
        return nullptr;
    } else {
        return it->second;
    }
}

/*Two things: setup relation_structure and create a tree node.*/
void SelectStructure::CheckBasicRel(BasicRelPointer arg_rel, QueryContextPointer arg_query_context) {
    ARIES_ASSERT(arg_rel != nullptr, "expression to check should not be empty");


    //std::cout << "OK: " << arg_rel->ToString() << "\n";

    if (arg_rel->IsSubquery() == false) {


        if ( arg_rel->GetDb().empty() )
        {
            if ( nullptr == default_schema_agent ) {
                ARIES_EXCEPTION(ER_NO_DB_ERROR);
            }
            arg_rel->SetDb( default_schema_agent->schema->GetName() );
        }

        PhysicalTablePointer physical_table = this->GetTableByName(arg_rel);

        if (physical_table) {
            /*we cannot directly use the RelationStructure of the underlying table --
            *because it could be used multiple times in a single query, for example a seft join!
            *So we get a simpleclone */

            RelationStructurePointer rel_structure = physical_table->GetRelationStructure()->SimpleClone();

            /* we must re-set the tablestructure name to the alias name*/
            if (arg_rel->GetAliasNamePointer() != nullptr) {
                rel_structure->SetName((*(arg_rel->GetAliasNamePointer())));
            }

            arg_rel->SetRelationStructure(rel_structure);
            arg_rel->SetUnderlyingTable(physical_table);

            /*The first parameter is the query, and the second one is the table_structure*/
            //arg_basic_rel.tree_node = SQLTreeBuilder.makeTreeNode_Table(arg_basic_rel, arg_basic_rel.my_table_structure);
            //arg_basic_rel.tree_node = SQLTreeBuilder.makeTreeNode_Table(arg_context.select_structure, arg_basic_rel);

            //this->rel_node_map[arg_rel] = this->node_builder->makeTreeNode_Table(arg_rel);
            arg_rel->SetMyRelNode(this->node_builder->makeTreeNode_Table(arg_rel));
        } else {

            std::string db_name = arg_rel->GetDb();
            if (db_name.empty()) {
                if (default_schema_agent) {
                    db_name = default_schema_agent->schema->GetName();
                }
                else
                {
                    ARIES_EXCEPTION(ER_NO_DB_ERROR);
                }
            }
            std::string view_name = arg_rel->GetID();

            auto view_node = ViewManager::GetInstance().GetViewNode(view_name, db_name);

            if (view_node) {
                view_node = view_node->Clone();
                auto s = (SelectStructure*)(view_node->GetMyQuery().get());
                RelationStructurePointer rel_structure = s->GetRelationStructure();
                rel_structure->SetName(arg_rel->GetID());
                arg_rel->SetRelationStructure(rel_structure);
                arg_rel->SetMyRelNode(view_node);
                arg_rel->ResetAlias(view_name);
                arg_rel->ResetToSubQuery(view_node->GetMyQuery());
                view_node->SetIsTopNodeOfAFromSubquery(true);
                view_node->SetBasicRel(arg_rel);
            } else {
                ARIES_EXCEPTION(ER_NO_SUCH_TABLE, arg_rel->GetDb().data(), arg_rel->GetID().data());
            }
        }

        //std::cout << "In CheckBasicRel: arg_rel =  " + arg_rel->ToString() << "\n";

    } else {

        ExprContextPointer a_virtual_expr_context = std::make_shared<ExprContext>(ExprContextType::VirtualExpr, //type
                                                                                  nullptr, //the expr
                                                                                  this->query_context, //query context
                                                                                  this->query_context->expr_context, //parent context
                                                                                  0);


        AbstractQueryPointer subquery = arg_rel->GetSubQuery();

        /*now we have to convert subquery into a raw pointer?!*/
        //SelectStructure* subquery_raw_pointer = (SelectStructure *)(subquery.get());

        QueryContextPointer new_query_context = std::make_shared<QueryContext>(QueryContextType::FromSubQuery,
                                                                               this->query_context->query_level + 1,
                                                                               subquery,
                                                                               this->query_context,
                                                                               a_virtual_expr_context);

        this->query_context->subquery_context_array.push_back(new_query_context);


        std::shared_ptr<SelectStructure> real_subquery = std::dynamic_pointer_cast<SelectStructure>(subquery);

        /*I need my own node_builder*/
        SQLTreeNodeBuilderPointer new_node_builder = std::make_shared<SQLTreeNodeBuilder>(subquery);

        real_subquery->CheckQueryGate2(this->default_schema_agent, new_node_builder, new_query_context);


        RelationStructurePointer rel_structure = real_subquery->GetRelationStructure();
        ARIES_ASSERT(arg_rel->GetAliasNamePointer() != nullptr, "alias pointer should not be null");

        const auto& columns_alias = arg_rel->GetColumnsAlias();
        if ( !columns_alias.empty() )
        {
            rel_structure = std::make_shared< RelationStructure >();
            for ( size_t i = 0; i < columns_alias.size(); i++ )
            {
                // auto column_name = columns_alias[ i ]->GetName();
                // rel_structure->GetColumn( i )->ResetName( columns_alias[ i ]->GetName() );
                rel_structure->AddColumn( columns_alias[ i ] );
            }
            real_subquery->SetRelationStructure( rel_structure );
        }

        rel_structure->SetName((*(arg_rel->GetAliasNamePointer())));


        arg_rel->SetRelationStructure(rel_structure);

        //arg_basic_rel.tree_node = arg_basic_rel.subquery.query_plan_tree;
        //this->rel_node_map[arg_rel] = real_subquery->GetQueryPlanTree();
        arg_rel->SetMyRelNode(real_subquery->GetQueryPlanTree());
        real_subquery->GetQueryPlanTree()->SetIsTopNodeOfAFromSubquery(true);
        real_subquery->GetQueryPlanTree()->SetBasicRel(arg_rel);

    }


}

PhysicalTablePointer SelectStructure::GetTableByName(const BasicRelPointer& arg_rel) {

    //std::cout << "In GetTableByName: + " << arg_table_name + "\n";
    std::string arg_table_name = arg_rel->GetID();
    std::string arg_db_name = arg_rel->GetDb();
    SchemaAgentPointer schemaAgentPointer = nullptr;
    if (arg_db_name.empty()) {
        if (nullptr == default_schema_agent) {
            ARIES_EXCEPTION(ER_NO_DB_ERROR);
        }
        arg_db_name = default_schema_agent->schema->GetName();
        schemaAgentPointer = default_schema_agent;
    } else {
        schemaAgentPointer = GetSchemaAgent(arg_db_name);
    }
    if (!schemaAgentPointer) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, arg_db_name.data() );
    }

    return schemaAgentPointer->FindPhysicalTable(arg_table_name);
}


bool SelectStructure::CheckWherePart() {
    /**
     * 如果 group-by 只是用来承载 having clause（sql 语句中本身没有 group-by clause）
     * 那么视情况（having clause 不包含聚合函数）将 having clause 移动到 where clause 中
     */
    if (groupby_part && groupby_part->GetGroupbyExprCount() == 0 && groupby_part->GetHavingExpr()) {
        auto having_expr = (std::dynamic_pointer_cast<CommonBiaodashi>(groupby_part->GetHavingExpr()))->Clone();
        ExprContextPointer having_expr_context = std::make_shared<ExprContext>(
                ExprContextType::HavingExpr,
                having_expr,
                query_context,
                query_context->expr_context,
                0);

        this->CheckExprGate(having_expr, having_expr_context);

        if (!having_expr_context->see_agg_func) {
            BiaodashiAuxProcessor processor;
            std::vector<BiaodashiPointer> exprs;
            exprs.emplace_back(groupby_part->GetHavingExpr());
            if (where_part) {
                exprs.emplace_back(where_part);
            }

            where_part = processor.make_biaodashi_from_and_list(exprs);
            groupby_part = nullptr;
        }
    }

    if (this->where_part == nullptr)
        return true;

    ExprContextPointer where_expr_context = std::make_shared<ExprContext>(
            ExprContextType::WhereExpr,
            this->where_part,
            this->query_context,
            this->query_context->expr_context,
            0);

    this->CheckExprGate(this->where_part, where_expr_context);

    BiaodashiAuxProcessor processor;
    auto conditions = processor.generate_and_list( where_part );
    bool has_changed = false;
    for ( size_t i = 0; i < conditions.size(); i++ )
    {
        std::vector< BiaodashiPointer > or_conditions_list;
        processor.generate_or_list( conditions[ i ], or_conditions_list );
        if ( or_conditions_list.size() < 2 )
        {
            continue;
        }

        std::vector< std::vector< BiaodashiPointer > > new_and_list( or_conditions_list.size() );
        int total_count = 1;
        for ( size_t j = 0; j < or_conditions_list.size(); j++ )
        {
            auto& condition = or_conditions_list[ j ];
            auto and_list = processor.generate_and_list( condition );
            std::vector< BiaodashiPointer > multi_table_conditions;
            std::map< std::string, std::vector< BiaodashiPointer > > conditions_by_table;
            for ( const auto& child : and_list )
            {
                auto expr = std::dynamic_pointer_cast< CommonBiaodashi >( child );
                expr->ObtainReferenceTableInfo();
                auto tables = expr->GetInvolvedTableList();
                bool has_decimal = false;

                if ( tables.size() == 1 )
                {
                    auto columns = expr->GetAllReferencedColumns();
                    for ( const auto& column : columns )
                    {
                        if ( column->GetValueType() == BiaodashiValueType::DECIMAL )
                        {
                            has_decimal = true;
                            break;
                        }
                    }
                }

                if ( tables.size() != 1 || has_decimal )
                {
                    multi_table_conditions.emplace_back( child );
                }
                else
                {
                    auto& table = tables[ 0 ];
                    conditions_by_table[ table->GetMyOutputName() ].emplace_back( child );
                }
            }
            if ( !multi_table_conditions.empty() )
            {
                new_and_list[ j ].emplace_back( processor.make_biaodashi_from_and_list( multi_table_conditions ) );
            }

            for ( const auto it : conditions_by_table )
            {
                new_and_list[ j ].emplace_back( processor.make_biaodashi_from_and_list( it.second ) );
            }
            total_count *= new_and_list[ j ].size();
        }

        std::vector< std::vector< BiaodashiPointer > > new_or_list( total_count );
        int repeat = total_count;
        for ( size_t j = 0; j < new_and_list.size(); j++ )
        {
            auto this_count = new_and_list[ j ].size();
            repeat = repeat / this_count;
            for ( int h = 0; h < total_count; h += repeat * this_count )
            {
                for ( size_t k = 0; k < this_count; k++ )
                {
                    for ( int l = 0; l < repeat; l++ )
                    {
                        auto index = h + repeat * k + l;
                        new_or_list[ index ].emplace_back( new_and_list[ j ][ k ] );
                    }
                }   
            }
        }

        std::vector< BiaodashiPointer > new_and_array;
        for ( const auto& list : new_or_list )
        {
            new_and_array.emplace_back ( processor.make_biaodashi_from_or_list( list ) );
        }

        auto new_condition = processor.make_biaodashi_from_and_list( new_and_array );
        has_changed = true;
        conditions[ i ] = new_condition;
    }

    if ( has_changed )
    {
        where_part = processor.make_biaodashi_from_and_list( conditions );
    }

    return true;
}

bool SelectStructure::CheckGroupPart() {
    if (this->groupby_part == nullptr)
        return true;

    this->need_groupby_operation = true;

    for (size_t i = 0; i != this->groupby_part->GetGroupbyExprCount(); i++) {
        BiaodashiPointer a_gb_expr = this->groupby_part->GetGroupbyExpr(i);

        auto group_by_expr = std::dynamic_pointer_cast<CommonBiaodashi>(a_gb_expr);
        if (group_by_expr->GetType() == BiaodashiType::Zhengshu) {
            auto index = boost::get<int>(group_by_expr->GetContent());

            if (index < 1 || ( size_t )index > select_part->GetSelectItemCount()) {
                ARIES_EXCEPTION(ER_BAD_FIELD_ERROR, 
                                std::to_string(index).c_str(), 
                                "group-by clause");
            }

            auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(select_part->GetSelectExpr(index - 1));

            group_by_expr->Clone(expr);
        }

        ExprContextPointer a_gb_expr_context = std::make_shared<ExprContext>(
                ExprContextType::GroupbyExpr,
                a_gb_expr,
                this->query_context,
                this->query_context->expr_context,
                0);

        this->CheckExprGate(a_gb_expr, a_gb_expr_context);
    }


    /*now we check the having part*/
    /*According to postgresql, having part cannot see the column alias in select part*/
    auto havingExpr = std::dynamic_pointer_cast<CommonBiaodashi>(groupby_part->GetHavingExpr());
    if (havingExpr != nullptr)
    {
        //收集select列表里,有别名并且不包含聚合函数的复杂表达式.
        vector<CommonBiaodashi *> exprsWithAlias;
        for (size_t i = 0; i < select_part->GetSelectItemCount(); ++i)
        {
            auto tmpExpr = (CommonBiaodashi *)(select_part->GetSelectExpr(i).get());
            if (tmpExpr->GetType() != BiaodashiType::Biaoshifu 
                && select_part->GetSelectAlias(i) )
                exprsWithAlias.push_back(tmpExpr);
        }
        if (havingExpr->GetType() == BiaodashiType::Biaoshifu)
        {
            //如果a_ob_expr是exprsWithAlias中的别名,直接替换成对应的表达式
            for (auto &expr : exprsWithAlias)
            {
                if (expr->GetOrigName() == havingExpr->GetOrigName())
                {
                    havingExpr = expr->Clone();
                    groupby_part->SetHavingExpr(havingExpr);
                    break;
                }
            }
        }
        else
        {
            //将havingExpr中包含的别名进行替换
            if (!exprsWithAlias.empty())
                std::dynamic_pointer_cast<CommonBiaodashi>(havingExpr)->ReplaceBiaoshifu(exprsWithAlias);
        }

        ExprContextPointer having_expr_context = std::make_shared<ExprContext>(
            ExprContextType::HavingExpr,
            this->groupby_part->GetHavingExpr(),
            this->query_context,
            this->query_context->expr_context,
            0);

        this->CheckExprGate(this->groupby_part->GetHavingExpr(),
                            having_expr_context);
    }

    this->AddReferencedColumnIntoGroupbyPartArray();

    return true;
}

bool SelectStructure::CheckOrderPart() {
    if (this->orderby_part == nullptr)
        return true;

    //收集select列表里,有别名并且不包含聚合函数的复杂表达式.
    vector< CommonBiaodashi* > exprsWithAlias;
    for( size_t i = 0; i < select_part->GetSelectItemCount(); ++i ) 
    {
        auto tmpExpr = ( CommonBiaodashi* )( select_part->GetSelectExpr( i ).get() );
        if( tmpExpr->GetType() != BiaodashiType::Biaoshifu 
            && select_part->GetSelectAlias( i )
            && !tmpExpr->IsAggFunction() 
            && !tmpExpr->ContainsAggFunction() )
            exprsWithAlias.push_back( tmpExpr );
    }

    for (size_t i = 0; i != this->orderby_part->GetOrderbyItemCount(); i++) {
        BiaodashiPointer a_ob_expr = this->orderby_part->GetOrderbyItem(i);

        auto order_by_expr = (CommonBiaodashi*) (a_ob_expr.get());

        if (order_by_expr->GetType() == BiaodashiType::Zhengshu) {
            continue;
        }

        //如果a_ob_expr是exprsWithAlias中的别名,直接替换成对应的表达式
        for( auto& expr : exprsWithAlias )
        {
            if( expr->GetOrigName() == order_by_expr->GetOrigName()  )
            {
                a_ob_expr = expr->Clone();
                this->orderby_part->SetOrderbyItem( a_ob_expr, i );
                order_by_expr = ( CommonBiaodashi* ) ( a_ob_expr.get() );
                break;
            }
        }

        if (order_by_expr->GetType() != BiaodashiType::Biaoshifu) {

            /**
             * 当 order-by 表达式是聚合函数或者查询中包含 group-by 时，需要在 group-by 节点来计算该表达式
             */
            if (need_groupby_operation || order_by_expr->IsAggFunction() || order_by_expr->ContainsAggFunction()){
                std::shared_ptr<std::string> alias_ptr = nullptr;
                auto random_alias = std::make_shared<std::string>("ORDER_BY_" + std::to_string(i) + "_" + a_ob_expr->GetName());
                for (size_t j = 0; j < select_part->GetSelectItemCount(); j ++) {
                    auto select = (CommonBiaodashi*) (select_part->GetSelectExpr(j).get());
                    if (!select->IsSameAs(order_by_expr)) {
                        continue;
                    }

                    alias_ptr = select_part->GetSelectAlias(j);
                    if (!alias_ptr) {
                        alias_ptr = random_alias;
                        select_part->ResetAliasAtIndex(random_alias, j);
                        select->SetNeedShowAlias(false);
                    }
                }

                if (!alias_ptr) {
                    alias_ptr = random_alias;
                    auto new_expr = order_by_expr->Clone();
                    new_expr->SetIsVisibleInResult(false);
                    select_part->AddSelectExpr(new_expr, random_alias);
                }

                auto ident = std::make_shared<SQLIdent>("", "", *alias_ptr);
                order_by_expr->SetType(BiaodashiType::Biaoshifu);
                order_by_expr->SetContent(ident);
                order_by_expr->SetOrigName("", *alias_ptr);
                order_by_expr->SetName(*alias_ptr);
                order_by_expr->ClearChildren();
            }
            else
            {
                //将a_ob_expr中包含的别名进行替换
                if( !exprsWithAlias.empty() )
                    std::dynamic_pointer_cast< CommonBiaodashi >( a_ob_expr )->ReplaceBiaoshifu( exprsWithAlias );
            }
        }

        ExprContextPointer a_ob_expr_context = std::make_shared<ExprContext>(
                ExprContextType::OrderbyExpr,
                a_ob_expr,
                this->query_context,
                this->query_context->expr_context,
                0);

        this->CheckExprGate(a_ob_expr, a_ob_expr_context);

    }

    return true;
}


std::vector<BiaodashiPointer> SelectStructure::GetAllColumnsFromBasicRel(BasicRelPointer arg_brp) {
    ARIES_ASSERT(arg_brp != nullptr, "arg should not be null");
    return arg_brp->GetAllColumnsAsExpr();
}

/*if the input table name is empty, then we need get all columns of all tables in the from part!*/
std::vector<BiaodashiPointer> SelectStructure::GetAllColumnsFromGivenTable(std::string arg_table_name) {
    std::vector<BiaodashiPointer> ret_column_array;

    if ( !arg_table_name.empty() ) {
        auto it = this->from_table_map.find(arg_table_name);
        if (it == this->from_table_map.end()) {
            string msg = format_mysql_err_msg( ER_BAD_TABLE_ERROR, arg_table_name.data() );
            ARIES_EXCEPTION_SIMPLE( ER_BAD_TABLE_ERROR, "GetAllColumnsFromGivenTable: " + msg );
        } else {
            ret_column_array = this->GetAllColumnsFromBasicRel(it->second);
        }

    } else {
        std::vector<BiaodashiPointer> temp_bpa;
        for (size_t i = 0; i < this->from_table_array.size(); i++) {
            temp_bpa = this->GetAllColumnsFromBasicRel(this->from_table_array[i]);

            ret_column_array.insert(ret_column_array.end(), temp_bpa.begin(), temp_bpa.end());
        }
    }

    return ret_column_array;
}

void SelectStructure::CheckSelectExprGate(BiaodashiPointer arg_expr, std::shared_ptr<std::string> arg_alias_p) {

    ExprContextPointer select_expr_context = std::make_shared<ExprContext>(
            ExprContextType::SelectExpr,
            arg_expr,
            this->query_context,
            this->query_context->expr_context,
            0);

    auto expr = (CommonBiaodashi*) (arg_expr.get());
    if (arg_alias_p && expr->NeedShowAlias()) {
        arg_expr->SetName(*arg_alias_p);
    }

    this->CheckExprGate(arg_expr, select_expr_context);

    /*todo: these two parts are the same. we should remove one*/
    this->checked_expr_array.push_back(arg_expr);
    this->checked_alias_array.push_back(arg_alias_p);

    this->select_part->AddCheckedExpr(arg_expr);
    this->select_part->AddCheckedAlias(arg_alias_p);

    if (select_expr_context->see_agg_func == true) {
        this->agg_in_select = true;
    }
}

/*tough!*/
bool SelectStructure::CheckSelectPart() {

    /*step 0: special handling for exist(select ... from ...)*/

    if (this->am_i_in_exist == true) {
        this->select_part->ChangeEverythingToNothing();
    }

    /*Step 1: Check each expr*/

    bool changed = false;
    std::vector<BiaodashiPointer> new_select_exprs;
    std::vector<std::shared_ptr<std::string>> new_alias;
    
    int sp_length = this->select_part->GetSelectItemCount();
    for (int i = 0; i < sp_length; i++) {
        BiaodashiPointer abp = this->select_part->GetSelectExpr(i);
        CommonBiaodashi *rawpointer = (CommonBiaodashi *) abp.get();

        if (rawpointer->GetType() == BiaodashiType::Star) {
            /*it must be a * or a table_name.* */

            std::string table_name = rawpointer->ContentToString();
            std::vector<BiaodashiPointer> vbp = this->GetAllColumnsFromGivenTable(table_name);

            changed = true;
            for (size_t ac_index = 0; ac_index < vbp.size(); ac_index++) {
                BiaodashiPointer ac = vbp[ac_index];

                if (((CommonBiaodashi*)(ac.get()))->GetOrigName() == schema::DBEntry::ROWID_COLUMN_NAME) {
                    continue;
                }

                new_select_exprs.emplace_back(ac);
                new_alias.emplace_back(nullptr);

                this->CheckSelectExprGate(ac, nullptr);
            }
        } else {
            new_select_exprs.emplace_back(abp);
            new_alias.emplace_back(select_part->GetSelectAlias(i));
            this->CheckSelectExprGate(abp, this->select_part->GetSelectAlias(i));
        }
    }

    if (changed) {
        select_part->ResetSelectExprsAndAlias(new_select_exprs, new_alias);
    }


    /*step 2:*/
    /*we need to check whether some alias names are being used by group by.*/

    for (size_t uiai = 0; uiai < this->query_context->unsolved_ID_array.size(); uiai++) {
        ColumnShellPointer p_column_shell = this->query_context->unsolved_ID_array[uiai];

        std::string colname = p_column_shell->GetColumnName();

        bool found = false;
        for (size_t caai = 0; caai < this->checked_alias_array.size(); caai++) {
            std::shared_ptr<std::string> alias_p = this->checked_alias_array[caai];

            if (alias_p != nullptr && (*alias_p) == colname) {
                found = true;
                /*todo: maybe should combine this with relationstructure*/
                /*or we can allow a ColumnShell expr has a children expr!*/
                p_column_shell->SetTableName("!-!SELECT_ALIAS!-!");
                p_column_shell->SetAliasExprIndex(caai);
                p_column_shell->SetExpr4Alias(this->checked_expr_array[caai]);

                break;
            }

        }

        if (!found) {
            ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, "unsolved name in group by or order by: " + colname);
        }

    }

    /**
     * 在 order-by 和 group-by 中使用到的别名有可能和真实的 column name 相同
     * 这里需要特殊处理这种情况，以别名优先
     */
    if (groupby_part) {
        handleAmbiguousAlias(groupby_part->GetGroupbyExprs());
    }

    if (orderby_part) {
        handleAmbiguousAlias(orderby_part->GetOrderbyExprs());
    }

    /*step 3:*/
    /*old js: now we need to put all the columns in my schema array*/
    /*new c++: now we need create the RelationStruction of this query*/

    this->rel_structure = std::make_shared<RelationStructure>();
    this->rel_structure->SetName("nameless"); /*it will be reset in basicrel*/

    for (size_t ceai = 0; ceai < this->checked_expr_array.size(); ceai++) {
        CommonBiaodashi *rawpointer = (CommonBiaodashi *) ((this->checked_expr_array[ceai]).get());

        /*we have a column*/
        std::string column_name =
                (this->checked_alias_array[ceai]) ?
                *(this->checked_alias_array[ceai])
                                                  : rawpointer->GetName();
        if("nameless" == column_name)
            column_name += std::to_string(ceai);
        BiaodashiValueType column_type = rawpointer->GetValueType();
        int column_length = rawpointer->GetLength();
        bool column_allow_null = rawpointer->IsNullable();
        bool column_is_primary = false;

        ColumnStructurePointer csp = std::make_shared<ColumnStructure>(column_name,
                                                                       column_type,
                                                                       column_length,
                                                                       column_allow_null,
                                                                       column_is_primary);
        if (column_type == BiaodashiValueType::DECIMAL) {
            csp->SetPresision(column_length);
            csp->SetScale(rawpointer->GetAssociatedLength());
        }

        if( rawpointer->GetType() == BiaodashiType::Lie )
        {
            ColumnShellPointer tmpCsp = boost::get<ColumnShellPointer>( rawpointer->GetContent() );
            csp->SetEncodeType( tmpCsp->GetColumnStructure()->GetEncodeType() );
            csp->SetEncodedIndexType( tmpCsp->GetColumnStructure()->GetEncodedIndexType() );
        }

        this->rel_structure->AddColumn(csp);
    }

    for (const auto& column : ambiguous_alias_column_array) {
        auto found = false;
        for (size_t caai = 0; caai < this->checked_alias_array.size(); caai++) {
            std::shared_ptr<std::string> alias_p = this->checked_alias_array[caai];
            if (alias_p && column->GetColumnName() == *(alias_p)) {
                found = true;
                column->SetTableName("!-!SELECT_ALIAS!-!");
                column->SetAliasExprIndex(caai);
                column->SetExpr4Alias(this->checked_expr_array[caai]);
                break;
             }
         }

         if (!found) {
             ARIES_EXCEPTION(ER_BAD_FIELD_ERROR, 
                            column->GetColumnName().c_str(), 
                            (column->GetTableName().empty() ? "field list" : column->GetTableName().c_str()));
         }
    }

    for (const auto& column : ambiguous_column_array) {
        int found_times = 0;
        for (const auto& item : checked_expr_array) {
            auto expr = (CommonBiaodashi*)(item.get());
            std::cout << expr->ToString() << std::endl;
            if (expr->GetName() == column->GetColumnName()) {
                if (expr->GetType() == BiaodashiType::Lie) {
                    auto csp = boost::get<ColumnShellPointer>(expr->GetContent());
                    column->SetTableName(csp->GetTableName());
                    column->SetTable(csp->GetTable());
                }
                found_times ++;
            }
        }

        if (found_times != 1) {
            ARIES_EXCEPTION( ER_NON_UNIQ_ERROR, column->GetColumnName().data(), " field list" );
        }
    }

    if (orderby_part) {
        AddReferencedColumnIntoOrderbyPartArray();
    }

    /*Finally, we create an array to store all columns used by select part -- ignoring those outer columns and remove duplicated!*/
    this->AddReferencedColumnIntoSelectPartArray();

    return true;
}

void SelectStructure::handleAmbiguousAlias(const std::vector<BiaodashiPointer>& exprs) {
    for (const auto& expr : exprs) {
        if (!expr) {
            continue;
        }

        auto expr_item = std::dynamic_pointer_cast<CommonBiaodashi>(expr);
        if (expr_item->GetType() != BiaodashiType::Lie) {
            continue;
        }

        auto column_shell = boost::get<ColumnShellPointer>(expr_item->GetContent());

        if (!column_shell->GetTableName().empty()) {
            continue;
        }

        for (size_t caai = 0; caai < this->checked_alias_array.size(); caai++) {
            const auto& alias = checked_alias_array[caai];
            if (!alias) {
                continue;
            }

            if (column_shell->GetColumnName() == *(alias)) {
                column_shell->SetTableName("!-!SELECT_ALIAS!-!");
                column_shell->SetAliasExprIndex(caai);
                column_shell->SetExpr4Alias(this->checked_expr_array[caai]);
            }
        }
    }
}


bool SelectStructure::BuildQueryPlanTreeMain() {

    /*now we build the query plan tree -- we have ready built the from node when checking the from part*/
    if (this->tree_node_from != nullptr) {
        this->query_plan_tree = this->tree_node_from;
    }

    /*filter*/
    if (this->where_part != nullptr) {
        SQLTreeNodePointer tmp_filter_node = this->node_builder->makeTreeNode_Filter(this->where_part);
        if ( !query_plan_tree )
        {
            query_plan_tree = tmp_filter_node;
        }
        else
        {
            SQLTreeNode::SetTreeNodeChild( tmp_filter_node, this->query_plan_tree );
        }

        this->query_plan_tree = tmp_filter_node;
    }

    if (groupby_part == nullptr && is_distinct) {
        groupby_part = std::make_shared<GroupbyStructure>();

        std::vector<BiaodashiPointer> group_by_items;
        for (size_t i = 0; i < select_part->GetSelectItemCount(); i ++) {
            auto select_item = select_part->GetSelectExpr(i);
            if( !std::dynamic_pointer_cast< CommonBiaodashi >( select_item )->ContainsAggFunction() )
                group_by_items.emplace_back(select_item);
            else 
                ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "distinct agg functions" );
        }

        groupby_part->SetGroupbyExprs(group_by_items);
    }

    /*group by*/
    if (!is_group_or_sort_node_removable)
    {
        if (this->groupby_part != nullptr)
        {
            SQLTreeNodePointer tmp_group_node = this->node_builder->makeTreeNode_Group();
            SQLTreeNode::SetTreeNodeChild(tmp_group_node, this->query_plan_tree);
            this->query_plan_tree = tmp_group_node;
        }
        else
        {
            /*todo: create a groupby node for those agg functions without explicit group by clause*/
            if (this->agg_in_select == true)
            {
                /*create a GroupbyStructure*/
                this->groupby_part = std::make_shared<GroupbyStructure>();
                this->need_groupby_operation = true;

                SQLTreeNodePointer tmp_group_node = this->node_builder->makeTreeNode_Group();
                tmp_group_node->SetACreatedGroupNode(true);
                SQLTreeNode::SetTreeNodeChild(tmp_group_node, this->query_plan_tree);
                tmp_group_node->SetGroupForAggInSelect(true);
                this->query_plan_tree = tmp_group_node;
            }
        }
    }

/*sort*/
    if (this->orderby_part && !is_group_or_sort_node_removable) {
        if( is_distinct )
        {
            for (size_t i = 0; i < this->orderby_part->GetOrderbyItemCount(); i++) 
            {
                if( LocateExprInSelectList( orderby_part->GetOrderbyItem(i) ) == -1 )
                    ARIES_EXCEPTION( ER_FIELD_IN_ORDER_NOT_SELECT, orderby_part->GetOrderbyItem(i)->GetName(), orderby_part->GetOrderbyItem(i)->GetName().c_str(), "DISTINCT" );
            }
        }
        SQLTreeNodePointer tmp_orderby_node = this->node_builder->makeTreeNode_Sort();
        SQLTreeNode::SetTreeNodeChild( tmp_orderby_node, this->query_plan_tree );
        this->query_plan_tree = tmp_orderby_node;
    }

    if (limit_structure) {
        auto limit_node = node_builder->makeTreeNode_Limit(limit_structure->Offset, limit_structure->Limit);

        if (query_plan_tree) {
            SQLTreeNode::SetTreeNodeChild( limit_node, query_plan_tree );
        }

        query_plan_tree = limit_node;
    }

    /*column*/
    if (true)//this->groupby_part == nullptr)
    {
        SQLTreeNodePointer tmp_column_node = this->node_builder->makeTreeNode_Column();
        if (this->query_plan_tree != nullptr) {
            SQLTreeNode::SetTreeNodeChild( tmp_column_node, this->query_plan_tree );
        }
        this->query_plan_tree = tmp_column_node;

    }

    /*now we have created the query plan*/
    return true;
}


bool SelectStructure::AddFromTable(BasicRelPointer arg_table) {
    /*Two tables must have different names*/

    RelationStructurePointer the_rel = arg_table->GetRelationStructure();
    std::string rel_name = the_rel ? the_rel->GetName() : arg_table->GetID();

    //std::cout << rel_name << "|OK\n";
    /*before add this rel, we have to detect duplication*/

    auto it = this->from_table_map.find(rel_name);
    if (it != this->from_table_map.end()) {
        /*duplicated!*/

        ARIES_EXCEPTION( ER_NONUNIQ_TABLE, rel_name.data() );

        return false;

    } else {

        this->from_table_map.insert(std::make_pair(rel_name, arg_table));
        this->from_table_array.push_back(arg_table);

    }

    return true;
}


GroupbyStructurePointer SelectStructure::GetGroupbyPart() {
    return this->groupby_part;
}

OrderbyStructurePointer SelectStructure::GetOrderbyPart() {
    return this->orderby_part;
}

SelectPartStructurePointer SelectStructure::GetSelectPart() {

    if (this->IsSetQuery()) {
        SelectStructure *ss_left_p = (SelectStructure *) ((this->left_part).get());
        return ss_left_p->GetSelectPart();
    }


    //todo: why?
    this->select_part->ResetCheckedExpr(this->checked_expr_array);
    this->select_part->ResetCheckedExprAlias(this->checked_alias_array);

    return this->select_part;
}

void SelectStructure::SetSQLTreeNodeBuilder(SQLTreeNodeBuilderPointer arg_node_builder) {
    this->node_builder = arg_node_builder;
}


SQLTreeNodeBuilderPointer SelectStructure::GetSQLTreeNodeBuilder() {
    return this->node_builder;
}

SQLTreeNodePointer SelectStructure::GetQueryPlanTree() {
    return this->query_plan_tree;
}

void SelectStructure::SetQueryPlanTree(SQLTreeNodePointer arg_node) {
    this->query_plan_tree = arg_node;
}

SQLTreeNodePointer SelectStructure::BuildSubTreeAddNode( SQLTreeNodePointer arg_root_node, SQLTreeNodePointer arg_new_node, JoinType arg_join_type, 
                                                            BiaodashiPointer arg_expr, bool b_new_node_as_left_child )
{
    ARIES_ASSERT(arg_new_node != nullptr, "new node should not be null");
    if( !arg_root_node ) 
        return arg_new_node;
    else 
    {
        SQLTreeNodePointer new_join_node = arg_root_node;
        if( arg_join_type == JoinType::InnerJoin )
        {
            BiaodashiAuxProcessor processor;
            auto list = processor.generate_and_list( arg_expr );
            SQLTreeNodeType rootNodeType = arg_root_node->GetType();
            if( rootNodeType == SQLTreeNodeType::InnerJoin_NODE )
            {
                SQLTreeNode::AddTreeNodeChild( arg_root_node, arg_new_node );

                auto& conditions = arg_root_node->GetInnerJoinConditions();
                conditions.insert( conditions.end(), list.cbegin(), list.cend() );
            }
            else 
            {
                new_join_node = node_builder->makeTreeNode_InnerJoin();
                SQLTreeNode::AddTreeNodeChild( new_join_node, arg_root_node );
                SQLTreeNode::AddTreeNodeChild( new_join_node, arg_new_node );

                auto& conditions = new_join_node->GetInnerJoinConditions();
                conditions.insert( conditions.end(), list.cbegin(), list.cend() );
            }
        }
        else
        {
            new_join_node = node_builder->makeTreeNode_BinaryJoin( arg_join_type, arg_expr );
            if( b_new_node_as_left_child )
            {
                SQLTreeNode::AddTreeNodeChild( new_join_node, arg_new_node );
                SQLTreeNode::AddTreeNodeChild( new_join_node, arg_root_node );
            }
            else 
            {
                SQLTreeNode::AddTreeNodeChild( new_join_node, arg_root_node );
                SQLTreeNode::AddTreeNodeChild( new_join_node, arg_new_node );
            }
        }
        return new_join_node;
    }
}

/*setup the from node; */
void SelectStructure::BuildFromTreeAddInnerJoinNode(SQLTreeNodePointer arg_new_node, JoinType arg_join_type, BiaodashiPointer arg_expr ) 
{
    assert( arg_join_type == JoinType::InnerJoin );
    ARIES_ASSERT(arg_new_node != nullptr, "new node should not be null");
    if( !tree_node_from ) 
        tree_node_from = arg_new_node;
    else 
    {
        BiaodashiAuxProcessor processor;
        auto list = processor.generate_and_list( arg_expr );
        SQLTreeNodeType rootNodeType = tree_node_from->GetType();
        if( rootNodeType == SQLTreeNodeType::InnerJoin_NODE )
        {
            SQLTreeNode::AddTreeNodeChild( tree_node_from, arg_new_node );

            auto& conditions = tree_node_from->GetInnerJoinConditions();
            conditions.insert( conditions.end(), list.cbegin(), list.cend() );
        }
        else 
        {
            SQLTreeNodePointer new_join_node = node_builder->makeTreeNode_InnerJoin();
            SQLTreeNode::AddTreeNodeChild( new_join_node, tree_node_from );
            SQLTreeNode::AddTreeNodeChild( new_join_node, arg_new_node );

            auto& conditions = new_join_node->GetInnerJoinConditions();
            conditions.insert( conditions.end(), list.cbegin(), list.cend() );
            tree_node_from = new_join_node;
        }
    }
}

/*If the ith expr in the checked_expr_array is a ColumnShell, return its table name!*/
std::string SelectStructure::GetTableNameOfSelectItemIfColumn(size_t i) {
    ARIES_ASSERT(i < this->checked_expr_array.size(), "out of boundary");

    CommonBiaodashi *rawpointer = (CommonBiaodashi *) ((this->checked_expr_array[i]).get());

    return rawpointer->GetTableName();
}


bool SelectStructure::GetSelectpartProcessed() {
    return this->selectpart_has_been_processed;
}

void SelectStructure::SetSelectpartProcessed(bool arg_value) {
    this->selectpart_has_been_processed = arg_value;
}

bool SelectStructure::GetNeedGroupbyOperation() {
    return this->need_groupby_operation;
}


/*this is for the Select part. If I found a columnshell used there, then it will be processed here.*/
void SelectStructure::AddReferencedColumnIntoSelectPartArray() {

    this->referenced_column_array_in_select.clear();

    std::vector<ColumnShellPointer> tmp_vector;

    for (size_t ceai = 0; ceai < this->checked_expr_array.size(); ceai++) {
        CommonBiaodashi *rawpointer = (CommonBiaodashi *) ((this->checked_expr_array[ceai]).get());

        std::vector<ColumnShellPointer> tmp_csp_array;

        if (rawpointer->GetExprContext()->not_orginal_select_expr) {
            tmp_csp_array = rawpointer->GetAllReferencedColumns_NoQuery();
        } else {
            tmp_csp_array = rawpointer->GetExprContext()->referenced_column_array;
        }

        //= rawpointer->GetExprContext()->referenced_column_array;

        tmp_vector.insert(
                tmp_vector.end(),
                tmp_csp_array.begin(),
                tmp_csp_array.end()
        );

    }

    /*check and use each one!*/
    for (size_t i = 0; i < tmp_vector.size(); i++) {
        ColumnShellPointer csp = tmp_vector[i];

        /*is it an outercolumn? If so, ignore it!*/
        if (csp->GetAbsoluteLevel() != this->query_context->query_level) {
            continue;
        }

        bool found = false;

        for (size_t l = 0; l < this->referenced_column_array_in_select.size(); l++) {
            //if(csp == this->referenced_column_array_in_select[l])
            if (csp->GetTableName() == this->referenced_column_array_in_select[l]->GetTableName()
                &&
                csp->GetColumnName() == this->referenced_column_array_in_select[l]->GetColumnName()) {
                found = true;
                break;
            }
        }

        if (!found) {
            this->referenced_column_array_in_select.push_back(csp);
        }

    }

    is_group_or_sort_node_removable = !from_part || from_part->GetFromList().empty();
    /**
     * IF selct count(*)
     */
    if (referenced_column_array_in_select.size() == 0) {
        if (agg_in_select && from_part ) {
            for (const auto& part : from_part->GetFromList()) {
                auto rel = part->GetLeadingRel();

                if (rel->IsSubquery()) {
                    auto sub_query = std::dynamic_pointer_cast<SelectStructure>(rel->GetSubQuery());
                    auto select_item = sub_query->GetSelectPart()->GetSelectExpr(0);
                    auto alias = sub_query->GetSelectPart()->GetSelectAlias(0);

                    auto column_name = alias ? *alias : select_item->GetName();

                    auto column_shell = std::make_shared<ColumnShell>(rel->GetMyOutputName(), column_name);
                    column_shell->SetTable( rel );
                    referenced_column_array_in_select.emplace_back(column_shell);
                } else {
                    auto table = rel->GetPhysicalTable();
                    ColumnStructurePointer column;
                    std::string table_name;
                    if (table) {
                        column = table->GetRelationStructure()->GetColumn(0);
                        table_name = rel->GetMyOutputName();
                    } else {
                        std::string db_name(rel->GetDb());
                        if (db_name.empty()) {
                            if (default_schema_agent) {
                                db_name = default_schema_agent->schema->GetName();
                            }
                        }
                        table_name = rel->GetMyOutputName();
                        auto view_node = ViewManager::GetInstance().GetViewNode(table_name, db_name);
                        column = ((SelectStructure*)(view_node->GetMyQuery().get()))->GetRelationStructure()->GetColumn(0);
                    }

                    auto column_shell = std::make_shared<ColumnShell>(table_name, column->GetName());
                    column_shell->SetTable( rel );
                    referenced_column_array_in_select.emplace_back(column_shell);
                }
                break;
            }
        }
    }

    /*for debug*/
//	for(int di = 0; di < this->referenced_column_array_in_select.size(); di++)
//	{
//	    //std::cout << "AddReferencedColumnIntoSelectPartArray\t" << this->referenced_column_array_in_select[di]->ToString() << "\n";
//	}

}


void SelectStructure::AddReferencedColumnIntoGroupbyPartArray() {

    /* Make sure the result vector is empty when begin!
       It works for even the function is re-executed during query optimizations -- JoinSubqueryRemoving!  
    */
    this->referenced_column_array_in_groupby.clear();


    /* we check every groupby expr and the having expr,
       and then form an array to hold all ColumnShells used.
       But this way ignores those ColumnShells that are essentially Select Alias
       because they are not included into the referenced_column_array.
       However, we don't need to include them because they have already been processed in the SelectPart!
    */

    std::vector<ColumnShellPointer> tmp_vector;

    for (size_t i = 0; i < this->groupby_part->GetGroupbyExprCount(); i++) {
        CommonBiaodashi *a_gb_expr = (CommonBiaodashi *) ((this->groupby_part->GetGroupbyExpr(i)).get());

        //std::cout << i << "\t" << a_gb_expr->ToString() << "\n";

        std::vector<ColumnShellPointer> tmp_agb_array;
        if (a_gb_expr->GetExprContext()->not_orginal_group_expr) {
            tmp_agb_array = a_gb_expr->GetAllReferencedColumns_NoQuery();
        } else {
            tmp_agb_array = a_gb_expr->GetExprContext()->referenced_column_array;
        }

        //std::cout << "tmp_agb_array:\t" << tmp_agb_array.size() << "\n";

        tmp_vector.insert(
                tmp_vector.end(),
                tmp_agb_array.begin(),
                tmp_agb_array.end()
        );

    }

    if (this->groupby_part->GetHavingExpr() != nullptr) {
        CommonBiaodashi *the_having_expr = (CommonBiaodashi *) ((this->groupby_part->GetHavingExpr()).get());

        std::vector<ColumnShellPointer> tmp_the_array = the_having_expr->GetExprContext()->referenced_column_array;

        tmp_vector.insert(
                tmp_vector.end(),
                tmp_the_array.begin(),
                tmp_the_array.end()
        );


    }

    /*check and use each one!*/
    for (size_t i = 0; i < tmp_vector.size(); i++) {
        ColumnShellPointer csp = tmp_vector[i];

        /*is it an outercolumn? If so, ignore it!*/
        if (csp->GetAbsoluteLevel() != this->query_context->query_level) {
            continue;
        }

        bool found = false;

        for (size_t l = 0; l < this->referenced_column_array_in_groupby.size(); l++) {
            //if(csp == this->referenced_column_array_in_groupby[l])
            if (csp->GetTableName() == this->referenced_column_array_in_groupby[l]->GetTableName()
                &&
                csp->GetColumnName() == this->referenced_column_array_in_groupby[l]->GetColumnName()) {
                found = true;
                break;
            }
        }

        if (!found) {
            this->referenced_column_array_in_groupby.push_back(csp);
        }

    }

    /*for debug*/
    // std::cout << "\n\t---    void SelectStructure::AddReferencedColumnIntoGroupbyPartArray()\n";
    // for(int gi = 0; gi < this->referenced_column_array_in_groupby.size(); gi++)
    // {
    //     std::cout << this->referenced_column_array_in_groupby[gi]->ToString() << "\n";
    // }
    // std::cout << "\n\n";
}


void SelectStructure::AddReferencedColumnIntoOrderbyPartArray() {
    std::vector<ColumnShellPointer> tmp_vector;

    for (size_t i = 0; i < this->orderby_part->GetOrderbyItemCount(); i++) {
        auto a_ob_expr = std::dynamic_pointer_cast<CommonBiaodashi>(orderby_part->GetOrderbyItem(i));

        if (a_ob_expr->GetType() == BiaodashiType::Zhengshu) {
            auto index = boost::get<int>(a_ob_expr->GetContent());

            if (index < 1 || ( size_t )index > select_part->GetSelectItemCount()) {
                ARIES_EXCEPTION(ER_BAD_FIELD_ERROR, std::to_string(index).c_str(), "order clause");
            }

            auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(select_part->GetSelectExpr(index - 1));

            if ( groupby_part ) {
                auto alias = select_part->GetSelectAlias(index - 1);
                if (!alias) {
                    alias = std::make_shared<std::string>("ORDER_BY_" + std::to_string(i) + "_" + a_ob_expr->GetName());
                    select_part->ResetAliasAtIndex(alias, index - 1);
                    expr->SetNeedShowAlias(false);
                }

                auto column_shell = std::make_shared<ColumnShell>("!-!SELECT_ALIAS!-!", *alias);

                column_shell->SetAliasExprIndex(index - 1);
                column_shell->SetExpr4Alias(expr);

                auto ident = std::make_shared<SQLIdent>("", "", *alias);
                a_ob_expr->SetType(BiaodashiType::Lie);
                a_ob_expr->SetContent(column_shell);
                continue;
            }
            else
            {
                orderby_part->SetOrderbyItem( expr, i );
                a_ob_expr = expr;
            }
        }

        ExprContextPointer context = a_ob_expr->GetExprContext();

        std::vector<ColumnShellPointer> tmp_aob_array = context->referenced_column_array;

        tmp_vector.insert(
                tmp_vector.end(),
                tmp_aob_array.begin(),
                tmp_aob_array.end()
        );

    }


    /*check and use each one!*/
    for (size_t i = 0; i < tmp_vector.size(); i++) {
        ColumnShellPointer csp = tmp_vector[i];

        /*is it an outercolumn? If so, ignore it!*/
        if (csp->GetAbsoluteLevel() != this->query_context->query_level) {
            continue;
        }

        bool found = false;

        for (size_t l = 0; l < this->referenced_column_array_in_orderby.size(); l++) {
            //if(csp == this->referenced_column_array_in_orderby[l])
            if (csp->GetTableName() == this->referenced_column_array_in_orderby[l]->GetTableName()
                &&
                csp->GetColumnName() == this->referenced_column_array_in_orderby[l]->GetColumnName()) {
                found = true;
                break;
            }
        }

        if (!found) {
            this->referenced_column_array_in_orderby.push_back(csp);
        }

    }

}

std::vector<ColumnShellPointer> SelectStructure::GetReferencedColumnInSelect() {
    return this->referenced_column_array_in_select;
}

std::vector<ColumnShellPointer> SelectStructure::GetReferencedColumnInGroupby() {
    return this->referenced_column_array_in_groupby;
}

std::vector<ColumnShellPointer> SelectStructure::GetReferencedColumnInOrderby() {
    return this->referenced_column_array_in_orderby;
}


void SelectStructure::SetIAmInExist(bool arg_value) {
    this->am_i_in_exist = true;
}

bool SelectStructure::GetIAmInExist() {
    return this->am_i_in_exist;
}


std::vector<BiaodashiPointer> SelectStructure::GetAllSelectExprs() {

    if (this->IsSetQuery()) {
        SelectStructure *ss_left_p = (SelectStructure *) ((this->left_part).get());
        return ss_left_p->GetAllSelectExprs();
    }

    return this->checked_expr_array;
}

std::vector<std::shared_ptr<std::string>> SelectStructure::GetAllSelectAlias() {
    return this->checked_alias_array;
}


bool SelectStructure::DoIStillExist() {
    return this->do_i_still_exist;
}

void SelectStructure::SetDoIStillExist(bool arg_value) {
    this->do_i_still_exist = arg_value;
}


void SelectStructure::AddInitQueryAndExpr( const SelectStructurePointer& arg_ss, BiaodashiPointer arg_bp) {
    this->init_queries.push_back(arg_ss);
    this->init_replace_exprs.push_back(arg_bp);
}

int SelectStructure::GetInitQueryCount() {
    return this->init_queries.size();
}

SelectStructurePointer SelectStructure::GetInitQueryByIndex(int arg_index) {
    return this->init_queries[arg_index];
}

BiaodashiPointer SelectStructure::GetReplaceExprByIndex(int arg_index) {
    return this->init_replace_exprs[arg_index];
}


//this happens when JoinSubqueryRemoving add a column
void SelectStructure::AddExtraExpr(BiaodashiPointer arg_expr) {
    this->checked_expr_array.push_back(arg_expr);
    this->checked_alias_array.push_back(nullptr);
}

void SelectStructure::AddAliasForTheOnlyExpr(std::string arg_value) {
    ARIES_ASSERT(this->checked_alias_array.size() == 1, "checked alias array size should be 1");
    this->checked_alias_array[0] = std::make_shared<std::string>(arg_value);
}


int SelectStructure::LocateExprInSelectList(BiaodashiPointer arg_expr) {
    return this->select_part->LocateExprInSelectList(arg_expr);
}

int SelectStructure::GetAllExprCount() {
    return this->select_part->GetAllExprCount();
}

const LimitStructurePointer &SelectStructure::GetLimitStructure() const {
    return limit_structure;
}

void SelectStructure::SetLimitStructure(const LimitStructurePointer &limit_structure) {
    SelectStructure::limit_structure = limit_structure;
}

void SelectStructure::SetDistinct(bool value) {
    is_distinct = value;
}

bool SelectStructure::IsDistinct() const {
    return is_distinct;
}

void SelectStructure::SetOrderbyPart(const OrderbyStructurePointer& orderby_part) {
    SelectStructure::orderby_part = orderby_part;
}

SelectPartStructurePointer& SelectStructure::GetOriginalSelectPart() {
    return select_part;
}

void SelectStructure::AddAmbiguousColumn(const ColumnShellPointer& column) {
    ambiguous_column_array.emplace_back(column);
}

void SelectStructure::AddAmbiguousAliasColumn(const ColumnShellPointer& column) {
    ambiguous_alias_column_array.emplace_back(column);
}

void SelectStructure::AddSubQuery( const AbstractQueryPointer& query )
{
    sub_queries.emplace_back( query );
}

const std::vector< AbstractQueryPointer >& SelectStructure::GetSubQueries() const
{
    return sub_queries;
}

BiaodashiPointer SelectStructure::GetWhereCondition()
{
    return where_part;
}

void SelectStructure::SetSpoolAlias( const std::string left, const std::string right )
{
    spool_alias_map[ left ] = right;
}

const std::map< std::string, std::string >& SelectStructure::GetSpoolAlias() const
{
    return spool_alias_map;
}

}
