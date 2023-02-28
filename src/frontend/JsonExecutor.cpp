#include "JsonExecutor.h"
#include "nlohmann/json.hpp"
#include "SQLTreeNode.h"
#include <fstream>
#include "SelectStructure.h"
#include "AriesEngineByMaterialization.h"
#include "AriesEngine/transaction/AriesTransManager.h"

using namespace aries_engine;

using json = nlohmann::json;

namespace aries
{

struct parse_context
{
    json data;
    std::map< std::string, BiaodashiPointer > expression_references;
    std::map< std::string, ColumnShellPointer> column_references;
    std::map< std::string, BasicRelPointer > basic_rel_references;
    std::map< std::string, AbstractQueryPointer > query_references;
    std::map< std::string, ExprContextPointer > expr_context_references;
    std::map< std::string, QueryContextPointer > query_context_references;
};

static BiaodashiPointer parse_expression_reference( const json& ref_data, parse_context& context );
static void parse_expression_reference( const std::string& key, parse_context& context );
static void parse_expression_reference( const std::string& key, const json& data, parse_context& context );
static BiaodashiPointer parse_expression( const json& data, parse_context& context );

static ColumnShellPointer parse_column_reference( const json& ref_data, parse_context& context );
static void parse_column_reference( const std::string& key, parse_context& context );
static void parse_column_reference( const std::string& key, const json& data, parse_context& context );
static ColumnShellPointer parse_column_shell( const json& data, parse_context& context );

static BasicRelPointer parse_basic_rel_reference( const json& ref_data, parse_context& context );
static void parse_basic_rel_reference( const std::string& key, parse_context& context );
static void parse_basic_rel_reference( const std::string& key, const json& data, parse_context& context );
static BasicRelPointer parse_basic_rel( const json& data, parse_context& context );

static AbstractQueryPointer parse_query_reference( const json& ref_data, parse_context& context );
static void parse_query_reference( const std::string& key, parse_context& context );
static void parse_query_reference( const std::string& key, const json& data, parse_context& context );
static AbstractQueryPointer parse_query( const json& data, parse_context& context );

static ExprContextPointer parse_expr_context_reference( const json& ref_data, parse_context& context );
static void parse_expr_context_reference( const std::string& key, parse_context& context );
static void parse_expr_context_reference( const std::string& key, const json& data, parse_context& context );
static ExprContextPointer parse_expr_context( const json& data, parse_context& context );

static QueryContextPointer parse_query_context_reference( const json& ref_data, parse_context& context );
static void parse_query_context_reference( const std::string& key, parse_context& context );
static void parse_query_context_reference( const std::string& key, const json& data, parse_context& context );
static QueryContextPointer parse_query_context( const json& data, parse_context& context );

static void parse_tree_node( const json& data, SQLTreeNodePointer node, parse_context& context );

static inline bool is_ref( const json& data )
{
    return data.is_object() && data.value< bool >( "is_ref", false );
}

template < typename type_t >
static void parse_vector( const json& data, std::vector< type_t >& array )
{
    if ( !data.is_array() )
    {
        LOG( ERROR ) << "invalid vector json type, should be array";
        return;
    }

    for ( const auto& item : data.items() )
    {
        array.emplace_back( item.value().get< type_t >() );
    }
}

// static auto ColumnShellVectorHandler = []( const json& data, parse_context& context )
// {
//     return parse_column_shell( data, context );
// };

template < typename type_t, typename Handler >
static void parse_vector( const json& data, std::vector< type_t >& array, parse_context& context, Handler handler )
{
    if ( !data.is_array() )
    {
        return;
    }

    for ( const auto& item : data.items() )
    {
        array.emplace_back( handler( item.value(), context ) );
    }
}

static auto ColumnShellPositionHandler = []( ColumnShellPointer& key, int& position, const json& data, parse_context& context )
{
    key = parse_column_shell( data[ "key" ], context );
    position = data[ "position" ].get< int >();
};

template < typename type_a, typename type_b, typename Handler >
static void parser_map( const json& data, std::map< type_a, type_b >& map, parse_context& context, Handler handler )
{
    if ( !data.is_array() )
    {
        LOG( ERROR ) << "invalid vector json type, should be array";
        return;
    }

    for ( const auto& item : data.items() )
    {
        type_a key;
        type_b value;

        handler( key, value, item.value(), context );
        map[ key ] = value;
    }
}

static void parse_children( const json& data, std::vector< SQLTreeNodePointer >& children, parse_context& context )
{
    if ( !data.is_array() )
    {
        LOG( ERROR ) << "invalid children json type, should be array";
        return;
    }

    for ( const auto& item : data.items() )
    {
        SQLTreeNodePointer child = std::make_shared< SQLTreeNode >( SQLTreeNodeType::Table_NODE );
        parse_tree_node( item.value(), child, context );
        children.emplace_back( child );
    }
}

static ColumnStructurePointer parse_column_structure( const json& data )
{
    auto name = data.value< std::string >( "name", "" );
    auto type = data.value< int >( "type", 0 );
    auto length = data.value< int >( "length", 0 );
    auto allow_null = data.value< bool >( "allow_null", false );
    auto is_primary = data.value< bool >( "is_primary", false );
    auto is_fk = data.value< bool >( "is_fk", 0 );
    auto numeric_precision = data.value< int >( "numeric_precision", -1 );
    auto numeric_scale = data.value< int >( "numeric_scale", -1 );

    auto column = std::make_shared< ColumnStructure >( name, static_cast< ColumnValueType >( type ), length, allow_null, is_primary );
    column->SetIsFk( is_fk );
    column->SetPresision( numeric_precision );
    column->SetScale( numeric_scale );
    return column;
}

static ColumnShellPointer parse_column_shell( const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        return parse_column_reference( data, context );
    }

    auto table_name = data.value< std::string >( "table_name", "" );
    auto column_name = data.value< std::string >( "column_name", "" );
    auto numeric_precision = data.value< int >( "numeric_precision", -1 );
    auto numeric_scale = data.value< int >( "numeric_scale", -1 );

    auto column_shell = std::make_shared< ColumnShell >( table_name, column_name );
    column_shell->SetPresision( numeric_precision );
    column_shell->SetScale( numeric_scale );
    column_shell->SetPlaceholderMark( data.value< bool >( "is_placeholder", false ) );
    column_shell->SetAbsoluteLevel( data.value< int >( "absolute_level", -1 ) );

    if ( column_shell->GetPlaceholderMark() )
    {
        column_shell->SetMyOwnValueType( static_cast< BiaodashiValueType >( data.value< int >( "my_own_value_type", 0 ) ) );
    }

    if ( data.contains( "table") )
    {
        column_shell->SetTable( parse_basic_rel( data[ "table" ], context ) );
    }

    if ( data.contains( "column_structure" ) )
    {
        column_shell->SetColumnStructure( parse_column_structure( data[ std::string( "column_structure" ) ] ) );
    }

    return column_shell;
}

const std::string REF_KEY_EXPRESSION = "expression_references";
const std::string REF_KEY_COLUMN = "column_references";
const std::string REF_KEY_BASIC_REL = "basic_rel_references";
const std::string REF_KEY_QUERY = "query_references";
const std::string REF_KEY_QUERY_CONTEXT = "query_context_references";
const std::string REF_KEY_EXPR_CONTEXT = "expr_context_references";

#define REF_TYPE_EXPRESSION "expression"
#define REF_TYPE_COLUMN "column"
#define REF_TYPE_BASIC_REL "basic_rel"
#define REF_TYPE_QUERY "query"
#define REF_TYPE_QUERY_CONTEXT "query_context"
#define REF_TYPE_EXPR_CONTEXT "expr_context"
#define REF_KEY_TYPE "type"
#define REF_KEY_VALUE "value"

static BiaodashiContent parse_expression_content( BiaodashiType type, const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        auto ref_type = data.value< std::string >(REF_KEY_TYPE, "" );
        auto key = data.value< std::string >( REF_KEY_VALUE, "" );
        if ( ref_type == REF_TYPE_COLUMN )
        {
            return parse_column_reference( data, context );
        }
        else
        {
            assert( 0 );
        }
    }
    switch ( type )
    {
        case BiaodashiType::Lie:
            return parse_column_shell( data, context );
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Andor:
        case BiaodashiType::Bijiao:
        case BiaodashiType::Yunsuan:
            return data.get< int >();
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Star:
        case BiaodashiType::Decimal:
        {
            auto str = data.get< std::string >();
            return str;
        }
        case BiaodashiType::SQLFunc:
        {
            auto function_name = data.value< std::string >( "name", "" );
            auto function = std::make_shared< SQLFunction >( function_name );
            auto value_type = static_cast< BiaodashiValueType >( data.value< int >( "value_type", 0 ) );
            function->SetValueType( value_type );
            return function;
        }
        default:
#ifndef NDEBUG
            std::cout << "unhandled type of expression: " << static_cast< int >( type ) << std::endl;
#endif
            return 0;
    }
}

static BiaodashiPointer parse_expression( const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        return parse_expression_reference( data, context );
    }

    if ( data.is_null() )
    {
        return nullptr;
    }

    auto type = static_cast< BiaodashiType >( data.value< int >( "type", 0 ) );
    BiaodashiContent content;
    if ( data.contains( "content" ) )
    {
        auto& content_obj = data[ std::string( "content" ) ];
        content = parse_expression_content( type, content_obj, context );
    }
    auto expr = std::make_shared< CommonBiaodashi >( type, content );
    expr->SetValueType( static_cast< BiaodashiValueType >( data.value< int >( "value_type", 0 ) ) );
    expr->SetIsNullable( data.value< bool >( "nullable", false ) );
    expr->SetLength( data.value< int >( "length", 1 ) );
    if ( expr->GetValueType() == BiaodashiValueType::DECIMAL )
    {
        expr->SetAssociatedLength( data.value< int >( "associated_length", -1 ) );
    }

    if ( data.contains( "children" ) )
    {
        for ( const auto& item : data[ std::string( "children" ) ].items() )
        {
            auto child = parse_expression( item.value(), context );
            expr->AddChild( child );
        }
    }
    if ( data.contains( "expr_context" ) ){
        expr->SetExprContext( parse_expr_context( data[ "expr_context" ], context ) );
    }

    expr->ObtainReferenceTableInfo();
    return expr;
}

static RelationStructurePointer parse_relation_structure( const json& data )
{
    RelationStructurePointer relation = std::make_shared< RelationStructure >();

    if ( data.contains( "name" ) )
    {
        relation->SetName( data[ std::string( "name" ) ].get< std::string >() );
    }

    for ( const auto& item : data[ "columns" ].items() )
    {
        relation->AddColumn( parse_column_structure( item.value() ) );
    }

    return relation;
}

static BasicRelPointer parse_basic_rel( const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        return parse_basic_rel_reference( data, context );
    }
    auto id = data.value< std::string >( "rel_id", "" );
    auto db_name = data.value< std::string >( "db_name", "" );
    auto is_subquery = data.value< bool >( "isSubquery", false );

    std::shared_ptr< std::string > alias_name_pointer;
    if ( data.contains( "rel_alias_name" ) )
    {
        auto alias_name = data.value< std::string >( "rel_alias_name", "" );
        if ( !alias_name.empty() )
        {
            alias_name_pointer = std::make_shared< std::string >( alias_name );
        }
    }

    PhysicalTablePointer physical_table = nullptr;

    if ( data.contains( "underlying_table" ) )
    {
        auto physical_table_name = data[ std::string("underlying_table") ].value< std::string >( "table_name", "" );
        physical_table = std::make_shared< PhysicalTable >( physical_table_name );
    }

    AbstractQueryPointer selecture = nullptr;
    if ( is_subquery )
    {
        selecture = parse_query( data[ std::string( "subquery" ) ], context );
    }

    RelationStructurePointer relation;
    if ( data.contains( "relation_structure" ) )
    {
        relation = parse_relation_structure( data[ std::string( "relation_structure" ) ] );
    }

    auto rel = std::make_shared< BasicRel >( is_subquery, id, alias_name_pointer, selecture );
    rel->SetDb( db_name );
    rel->SetUnderlyingTable( physical_table );
    rel->SetRelationStructure( relation );
    rel->SetIsExpression( data.value< bool >( "isExpression", false ) );
    return rel;
}

static QueryContextPointer parse_query_context( const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        return parse_query_context_reference( data, context );
    }

    auto type = static_cast< QueryContextType >( data.value< int >( "type", 0 ) );
    auto query_level = data.value< int >( "query_level", 0 );
    BiaodashiPointer expr;
    SelectStructurePointer select_structure;// = parse_query( data[ "select_structure" ], context );
    QueryContextPointer parent_context;
    return std::make_shared< QueryContext >( type, query_level, select_structure, parent_context, nullptr );
}

static ExprContextPointer parse_expr_context( const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        return parse_expr_context_reference( data, context );
    }

    auto type = static_cast< ExprContextType >( data.value< int >( "type", 0 ) );
    auto index = data.value< int >( "index", 0 );
    BiaodashiPointer expr;

    auto query_context = parse_query_context( data[ "query_context" ], context );
    ExprContextPointer parent_context;
    return std::make_shared< ExprContext >( type, expr, query_context, parent_context, index );
}

static AbstractQueryPointer parse_query( const json& data, parse_context& context )
{
    if ( is_ref( data ) )
    {
        return parse_query_reference( data, context );
    }

    GroupbyStructurePointer groupby_part = std::make_shared< GroupbyStructure >();
    if ( data.contains( "groupby_part" ) && data[ std::string( "groupby_part" ) ].is_object() )
    {
        if ( data[ std::string( "groupby_part" ) ].contains( std::string( "groupby_exprs" ) ) )
        {
            for ( const auto& item : data[ std::string( "groupby_part" ) ][ "groupby_exprs" ].items() )
            {
                auto expr = parse_expression( item.value(), context );
                groupby_part->AddGroupbyExpr( expr );
            }
        }

        if ( data[ std::string( "groupby_part" ) ].contains( std::string( "additional_exprs" ) ) )
        {
            std::vector< BiaodashiPointer > exprs;
            for ( const auto& item : data[ std::string( "groupby_part" ) ][ "additional_exprs" ].items() )
            {
                auto expr = parse_expression( item.value(), context );
                exprs.emplace_back( expr );
            }

            groupby_part->SetAdditionalExprsForSelect( exprs );
        }
    }

    SelectPartStructurePointer select_part;
    if ( data.contains( "select_part" ) )
    {
        select_part = std::make_shared< SelectPartStructure >();
        for ( const auto& item : data[ std::string( "select_part" ) ][ "checked_expr_array"].items() )
        {
            auto expr = parse_expression( item.value(), context );
            select_part->AddCheckedExpr( expr );
        }
    }

    OrderbyStructurePointer orderby_part = std::make_shared< OrderbyStructure >();
    if ( data.contains( "orderby_part" ) )
    {
        const auto& orderby_expressions = data[ std::string( "orderby_part" ) ][ "orderby_expr_array" ];
        const auto& directions = data[ std::string( "orderby_part" ) ][ "orderby_direction_array" ];
        assert( orderby_expressions.size() == directions.size() );
        for ( size_t i = 0; i < orderby_expressions.size(); i++ )
        {
            const auto& item = orderby_expressions.at( i );
            auto expression = parse_expression( item, context );
            const auto& direction_item = directions.at( i );
            auto direction = static_cast< OrderbyDirection >( direction_item.get< int >() );
            orderby_part->AddOrderbyItem( expression, direction );
        }
    }

    auto query = std::make_shared< SelectStructure >();
    query->init_simple_query( select_part, nullptr, nullptr, groupby_part, orderby_part );

    if ( data.contains( "checked_expr_array" ) )
    {
        for ( const auto& item : data[ std::string( "checked_expr_array" ) ].items() )
        {
            auto expr = parse_expression( item.value(), context );
            query->AddExtraExpr( expr );
        }
    }
    return query;
}

static BiaodashiPointer parse_expression_reference( const json& ref_data, parse_context& context )
{
    auto ref_type = ref_data.value< std::string >(REF_KEY_TYPE, "" );
    auto key = ref_data.value< std::string >( REF_KEY_VALUE, "" );
    assert( ref_type == REF_TYPE_EXPRESSION );

    parse_expression_reference( key, context );
    if ( context.expression_references.find( key ) != context.expression_references.cend() )
    {
        return context.expression_references[ key ];
    }
    else
    {
#ifndef NDEBUG
        std::cout << "cannot find object for expression reference: " << key << std::endl;
#endif
        assert( 0 );
        return nullptr;
    }
}

static ColumnShellPointer parse_column_reference( const json& ref_data, parse_context& context )
{
    auto ref_type = ref_data.value< std::string >(REF_KEY_TYPE, "" );
    auto key = ref_data.value< std::string >( REF_KEY_VALUE, "" );
    assert( ref_type == REF_TYPE_COLUMN );

    parse_column_reference( key, context );
    if ( context.column_references.find( key ) != context.column_references.cend() )
    {
        return context.column_references[ key ];
    }
    else
    {
#ifndef NDEBUG
        std::cout << "cannot find object for column reference: " << key << std::endl;
#endif
        assert( 0 );
        return nullptr;
    }
}

static BasicRelPointer parse_basic_rel_reference( const json& ref_data, parse_context& context )
{
    auto ref_type = ref_data.value< std::string >(REF_KEY_TYPE, "" );
    auto key = ref_data.value< std::string >( REF_KEY_VALUE, "" );
    assert( ref_type == REF_TYPE_BASIC_REL );

    parse_basic_rel_reference( key, context );
    if ( context.basic_rel_references.find( key ) != context.basic_rel_references.cend() )
    {
        return context.basic_rel_references[ key ];
    }
    else
    {
#ifndef NDEBUG
        std::cout << "cannot find object for basic_rel reference: " << key << std::endl;
#endif
        assert( 0 );
        return nullptr;
    }
}

static QueryContextPointer parse_query_context_reference( const json& ref_data, parse_context& context )
{
    auto ref_type = ref_data.value< std::string >(REF_KEY_TYPE, "" );
    auto key = ref_data.value< std::string >( REF_KEY_VALUE, "" );
    assert( ref_type == REF_TYPE_QUERY_CONTEXT );

    parse_query_context_reference( key, context );
    if ( context.query_context_references.find( key ) != context.query_context_references.cend() )
    {
        return context.query_context_references[ key ];
    }
    else
    {
#ifndef NDEBUG
        std::cout << "cannot find object for query_references: " << key << std::endl;
#endif
        assert( 0 );
        return nullptr;
    }
}

static ExprContextPointer parse_expr_context_reference( const json& ref_data, parse_context& context )
{
    auto ref_type = ref_data.value< std::string >(REF_KEY_TYPE, "" );
    auto key = ref_data.value< std::string >( REF_KEY_VALUE, "" );
    assert( ref_type == REF_TYPE_EXPR_CONTEXT );

    parse_expr_context_reference( key, context );
    if ( context.expr_context_references.find( key ) != context.expr_context_references.cend() )
    {
        return context.expr_context_references[ key ];
    }
    else
    {
#ifndef NDEBUG
        std::cout << "cannot find object for query_references: " << key << std::endl;
#endif
        assert( 0 );
        return nullptr;
    }
}

static AbstractQueryPointer parse_query_reference( const json& ref_data, parse_context& context )
{
    auto ref_type = ref_data.value< std::string >(REF_KEY_TYPE, "" );
    auto key = ref_data.value< std::string >( REF_KEY_VALUE, "" );
    assert( ref_type == REF_TYPE_QUERY );

    parse_query_reference( key, context );
    if ( context.query_references.find( key ) != context.query_references.cend() )
    {
        return context.query_references[ key ];
    }
    else
    {
#ifndef NDEBUG
        std::cout << "cannot find object for query_references: " << key << std::endl;
#endif
        assert( 0 );
        return nullptr;
    }
}

static void parse_expression_reference( const std::string& key, const json& data, parse_context& context )
{
    if ( context.expression_references.find( key ) != context.expression_references.cend() )
    {
        return;
    }

    auto expression = parse_expression( data, context );
    context.expression_references[ key ] = expression;
}

static void parse_column_reference( const std::string& key, const json& data, parse_context& context )
{
    if ( context.column_references.find( key ) != context.column_references.cend() )
    {
        return;
    }

    auto column = parse_column_shell( data, context );
    context.column_references[ key ] = column;
}

static void parse_basic_rel_reference( const std::string& key, const json& data, parse_context& context )
{
    if ( context.basic_rel_references.find( key ) != context.basic_rel_references.cend() )
    {
        return;
    }

    auto basic_rel = parse_basic_rel( data, context );
    context.basic_rel_references[ key ] = basic_rel;
}

static void parse_query_context_reference( const std::string& key, const json& data, parse_context& context )
{
    if ( context.query_context_references.find( key ) != context.query_context_references.cend() )
    {
        return;
    }

    auto query_context = parse_query_context( data, context );
    context.query_context_references[ key ] = query_context;
}

static void parse_expr_context_reference( const std::string& key, const json& data, parse_context& context )
{
    if ( context.expr_context_references.find( key ) != context.expr_context_references.cend() )
    {
        return;
    }

    auto expr_context = parse_expr_context( data, context );
    context.expr_context_references[ key ] = expr_context;
}

static void parse_query_reference( const std::string& key, const json& data, parse_context& context )
{
    if ( context.query_references.find( key ) != context.query_references.cend() )
    {
        return;
    }

    auto query = parse_query( data, context );
    context.query_references[ key ] = query;
}

static void parse_expression_reference( const std::string& key, parse_context& context )
{
    if ( context.expression_references.find( key ) != context.expression_references.cend() )
    {
        return;
    }

    if ( !context.data[ REF_KEY_EXPRESSION ].contains( key ) )
    {
#ifndef NDEBUG
        std::cout << "cannot find expression ref key: " << key << std::endl;
#endif
        return;
    }

    parse_expression_reference( key, context.data[ REF_KEY_EXPRESSION ][ key ], context );
}

static void parse_column_reference( const std::string& key, parse_context& context )
{
    if ( context.column_references.find( key ) != context.column_references.cend() )
    {
        return;
    }

    if ( !context.data[ REF_KEY_COLUMN ].contains( key ) )
    {
#ifndef NDEBUG
        std::cout << "cannot find column ref key: " << key << std::endl;
#endif
        return;
    }

    parse_column_reference( key, context.data[ REF_KEY_COLUMN ][ key ], context );
}

static void parse_basic_rel_reference( const std::string& key, parse_context& context )
{
    if ( context.basic_rel_references.find( key ) != context.basic_rel_references.cend() )
    {
        return;
    }

    if ( !context.data[ REF_KEY_BASIC_REL ].contains( key ) )
    {
#ifndef NDEBUG
        std::cout << "cannot find column ref key: " << key << std::endl;
#endif
        return;
    }

    parse_basic_rel_reference( key, context.data[ REF_KEY_BASIC_REL ][ key ], context );
}

static void parse_query_context_reference( const std::string& key, parse_context& context )
{
    if ( context.query_context_references.find( key ) != context.query_context_references.cend() )
    {
        return;
    }

    if ( !context.data[ REF_KEY_QUERY_CONTEXT ].contains( key ) )
    {
#ifndef NDEBUG
        std::cout << "cannot find query context ref key: " << key << std::endl;
#endif
        return;
    }

    parse_query_context_reference( key, context.data[ REF_KEY_QUERY_CONTEXT ][ key ], context );
}

static void parse_expr_context_reference( const std::string& key, parse_context& context )
{
    if ( context.expr_context_references.find( key ) != context.expr_context_references.cend() )
    {
        return;
    }

    if ( !context.data[ REF_KEY_EXPR_CONTEXT ].contains( key ) )
    {
#ifndef NDEBUG
        std::cout << "cannot find expr context ref key: " << key << std::endl;
#endif
        return;
    }

    parse_expr_context_reference( key, context.data[ REF_KEY_EXPR_CONTEXT ][ key ], context );
}

static void parse_query_reference( const std::string& key, parse_context& context )
{
    if ( context.query_references.find( key ) != context.query_references.cend() )
    {
        return;
    }

    if ( !context.data[ REF_KEY_QUERY ].contains( key ) )
    {
#ifndef NDEBUG
        std::cout << "cannot find column ref key: " << key << std::endl;
#endif
        return;
    }

    parse_query_reference( key, context.data[ REF_KEY_QUERY ][ key ], context );
}

static void parse_references( parse_context& context )
{
    for ( const auto& item : context.data[ REF_KEY_EXPRESSION ].items() )
    {
        parse_expression_reference( item.key(), item.value(), context );
    }

    for ( const auto& item : context.data[ REF_KEY_COLUMN ].items() )
    {
        parse_column_reference( item.key(), item.value(), context );
    }

    for ( const auto& item : context.data[ REF_KEY_BASIC_REL ].items() )
    {
        parse_basic_rel_reference( item.key(), item.value(), context );
    }
}

static void parse_column_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    node->SetColumnNodeRemovable( data.value< bool >( "column_node_removable", false ) );
    node->SetForwardMode4ColumnNode( data.value< bool >( "ForwardMode4ColumnNode", false ) );

    if ( data.contains( "exprs_for_column_node" ) )
    {
        std::vector< BiaodashiPointer > exprs;
        for ( const auto& item : data[ std::string( "exprs_for_column_node" ) ].items() )
        {
            exprs.emplace_back( parse_expression( item.value(), context ) );
        }

        node->SetExprs4ColumnNode( exprs);
    }

    if ( data.contains( "column_pos_map_by_name" ) )
    {
        auto& map = data[ std::string( "column_pos_map_by_name" ) ];
        for ( const auto& item : map.items() )
        {
            auto pos = item.key().find( '.' );
            auto table_name = item.key().substr( 0, pos );
            auto column_name = item.key().substr( pos + 1 );
            auto column_shell = std::make_shared< ColumnShell >( table_name, column_name );
            node->SetPositionForReferencedColumn( column_shell, item.value().get< int >() );
        }
    }
}

static void parse_group_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    GroupbyStructurePointer groupby_part;
    if ( data.contains( "groupby_structure" ) )
    {
        groupby_part = std::make_shared< GroupbyStructure >();
        for ( const auto& item : data[ std::string( "groupby_structure" ) ][ "groupby_exprs" ].items() )
        {
            auto expr = parse_expression( item.value(), context );
            groupby_part->AddGroupbyExpr( expr );
        }

        if ( data[ std::string( "groupby_structure" ) ].contains( "additional_exprs" ) )
        {
            std::vector< BiaodashiPointer > exprs;
            for ( const auto& item : data[ std::string( "groupby_structure" ) ][ std::string( "additional_exprs" ) ].items() )
            {
                auto expr = parse_expression( item.value(), context );
                exprs.emplace_back( expr );
            }
            groupby_part->SetAdditionalExprsForSelect( exprs );
        }
        node->SetMyGroupbyStructure( groupby_part );
    }


    SelectPartStructurePointer select_part;
    if ( data.contains( "select_part_structure" ) )
    {
        select_part = std::make_shared< SelectPartStructure >();
        for ( const auto& item : data[ std::string( "select_part_structure" ) ][ "checked_expr_array"].items() )
        {
            auto expr = parse_expression( item.value(), context );
            select_part->AddCheckedExpr( expr );
        }
        node->SetMySelectPartStructure( select_part );
    }
}

static void parse_exchange_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    std::vector< int > source_devices_id;
    parse_vector( data[ "source_devices_id" ], source_devices_id );
    node->SetSourceDevicesId( source_devices_id );

    node->SetTargetDeviceId( data[ "target_device_id" ].get< int >() );
}

static void parse_filter_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    if ( data.contains( "my_filter_structure" ) )
    {
        auto expression = parse_expression( data[ std::string( "my_filter_structure" ) ], context );
        node->SetFilterStructure( expression );
    }
}

static void parse_join_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    if ( data.contains( "my_join_condition" ) )
    {
        node->SetJoinCondition( parse_expression( data[ std::string( "my_join_condition" ) ], context ) );
    }

    if ( data.contains( "my_join_other_condition") )
    {
        node->SetJoinOtherCondition( parse_expression( data[ std::string( "my_join_other_condition" ) ], context ) );
    }

    node->SetJoinType( static_cast< JoinType >( data.value< int >( "my_join_type", 0 ) ) );
}

static void parse_star_join_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    for ( const auto& item : data[ "star_join_conditions"].items() )
    {
        std::vector< CommonBiaodashiPtr > conditions;
        for ( const auto& child : item.value().items() )
        {
            auto condition = parse_expression( child.value(), context );
            conditions.emplace_back( std::dynamic_pointer_cast< CommonBiaodashi >( condition ) );
        }
        node->AddStarJoinCondition( conditions );
    }
}

static void parse_self_join_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{

}

static void parse_sort_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    OrderbyStructurePointer orderby_part;
    if ( data.contains( "orderby_structure" ) )
    {
        orderby_part = std::make_shared< OrderbyStructure >();
        const auto& orderby_expressions = data[ std::string( "orderby_structure" ) ][ "orderby_expr_array" ];
        const auto& directions = data[ std::string( "orderby_structure" ) ][ "orderby_direction_array" ];
        assert( orderby_expressions.size() == directions.size() );
        for ( size_t i = 0; i < orderby_expressions.size(); i++ )
        {
            const auto& item = orderby_expressions.at( i );
            auto expression = parse_expression( item, context );
            const auto& direction_item = directions.at( i );
            auto direction = static_cast< OrderbyDirection >( direction_item.get< int >() );
            orderby_part->AddOrderbyItem( expression, direction );
        }
    }

    node->SetMyOrderbyStructure( orderby_part );
}

static void parse_table_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    auto rel = parse_basic_rel( data[ std::string("my_basic_rel"  ) ], context );
    node->SetBasicRel( rel );

    node->SetSliceCount( data.value< int >( "slice_count", 0 ) );
    node->SetSliceIndex( data.value< int >( "slice_index", 0 ) );
}

static void parse_tree_node( const json& data, SQLTreeNodePointer node, parse_context& context )
{
    auto type = data.value< int >( "type", 0 );

    switch ( type )
    {
        case SQLTreeNodeType::Column_NODE:
            parse_column_node( data, node, context );
            break;
        case SQLTreeNodeType::Table_NODE:
            parse_table_node( data, node, context );
            break;
        case SQLTreeNodeType::Filter_NODE:
            parse_filter_node( data, node, context );
            break;
        case SQLTreeNodeType::Group_NODE:
            parse_group_node( data, node, context );
            break;
        case SQLTreeNodeType::Exchange_NODE:
            parse_exchange_node( data, node, context );
            break;
        case SQLTreeNodeType::BinaryJoin_NODE:
            parse_join_node( data, node, context );
            break;
        case SQLTreeNodeType::StarJoin_NODE:
            parse_star_join_node( data, node, context );
            break;
        case SQLTreeNodeType::SELFJOIN_NODE:
            parse_self_join_node( data, node, context );
            break;
        case SQLTreeNodeType::Sort_NODE:
            parse_sort_node( data, node, context );
            break;
        default:
            LOG( ERROR ) << "invalid tree node type: " << type;
            break;
    }

    node->ResetType( static_cast< SQLTreeNodeType >( type ) );

    if ( data.contains( "query" ) )
    {
        auto selecture = parse_query( data[ std::string( "query" ) ], context );
        node->SetMyQuery( selecture );
    }

    if ( data.contains( "children" ) )
    {
        std::vector< SQLTreeNodePointer > children;
        // auto& d =  data[ std::forward< std::string >( "children" ) ];
        parse_children( data[ std::string( "children" ) ], children, context );
        
        for ( const auto& child : children )
        {
            node->AddChild( child );
            child->SetParent( node );
        }
    }

    if ( data.contains( "referenced_column_position_map") )
    {
        std::map< ColumnShellPointer, int > columns_position_map;
        parser_map( data[ std::string( "referenced_column_position_map" ) ], columns_position_map, context, ColumnShellPositionHandler );
        for ( const auto& item : columns_position_map )
        {
            node->SetPositionForReferencedColumn( item.first, item.second );
        }
    }

    if ( data.contains( "column_output_sequence" ) )
    {
        std::vector< int > output_columns_id;
        parse_vector( data[ std::string( "column_output_sequence" ) ], output_columns_id );
        node->SetColumnOutputSequence( output_columns_id );
    }

    if ( data.contains( "required_column_array" ) )
    {
        std::vector< ColumnShellPointer > columns;
        parse_vector( data[ std::string( "required_column_array" ) ], columns, context, parse_column_shell );
        node->SetRequiredColumnArray( columns );
    }

    if ( data.contains( "referenced_column_array" ) )
    {
        std::vector< ColumnShellPointer > columns;
        parse_vector( data[ std::string( "referenced_column_array" ) ], columns, context, parse_column_shell );
        node->SetReferencedColumnArray( columns );
    }

    node->SetSpoolId( data.value< int >( "spool_id", -1 ) );

    std::vector< std::vector< int > > unique_keys_array;

    if ( data.contains( "unique_columns" ) )
    {
        for ( const auto& item : data[ std::string( "unique_columns" ) ].items() )
        {
            std::vector< int > keys;
            parse_vector( item.value(), keys );
            unique_keys_array.emplace_back( keys );
        }
    }

    if ( data.contains( "involved_table_list" ) )
    {
        std::vector< BasicRelPointer > tables;
        parse_vector( data[ std::string( "involved_table_list" ) ], tables, context, parse_basic_rel );
    }

    node->SetUniqueKeys( unique_keys_array );
}

bool JsonExecutor::Load( const std::string& path )
{
    ifstream f( path );
    std::string str((std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());

    SQLTreeNodePointer root = std::make_shared< SQLTreeNode >( SQLTreeNodeType::Table_NODE );
    parse_context context;
    context.data = json::parse( str );

    parse_references( context );
    parse_tree_node( context.data[ std::string( "tree" ) ], root, context );
    // std::cout << data.dump() << std::endl;
    // auto v = data.value<int>( "a", 0 );
    // data["a"].get< int >();
    // std::cout << "value: " << v << std::endl;
    // std::cout << root->ToString( 0 ) << std::endl;
    auto engine = std::make_shared<AriesEngineByMaterialization>();
    auto tx = AriesTransManager::GetInstance().NewTransaction();
    auto result = engine->ExecuteQueryTree( tx, root, "tpch_100" );
    // result = engine->ExecuteQueryTree( tx, root, "tpch_100" );
    auto table = std::dynamic_pointer_cast< AriesMemTable >( result );
    auto table_block = table->GetContent();

    std::vector< AriesDataBufferSPtr > columns;
    for ( int i = 1; i < table_block->GetColumnCount() + 1; i++ )
    {
        columns.emplace_back( table_block->GetColumnBuffer( i ) );
    }

    for ( int i = 0; i < table_block->GetRowCount(); i++ )
    {
#ifndef NDEBUG
        std::cout << i << ": ";
#endif
        for ( size_t j = 0; j < columns.size(); j++ )
        {
            auto& column = columns[ j ];
            auto data_type = column->GetDataType();
            if ( data_type.DataType.ValueType == AriesValueType::CHAR )
            {
                if ( column->GetDataType().HasNull )
                {
#ifndef NDEBUG
                    std::cout << "!\t!" << column->GetNullableString( i ) << "[N]";
#endif
                }
                else
                {
#ifndef NDEBUG
                    std::cout << "!\t!" << column->GetString( i );
#endif
                }
            }
            else if ( data_type.DataType.ValueType == AriesValueType::DECIMAL ||
                      data_type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
            {
#ifndef NDEBUG
                std::cout << "!\t!" << column->GetDecimalAsString( i );
#endif
                if ( column->GetDataType().HasNull )
                {
#ifndef NDEBUG
                    std::cout << "[N]";
#endif
                    continue;
                }
            }
            else if ( data_type.DataType.ValueType == AriesValueType::INT64 )
            {
#ifndef NDEBUG
                std::cout << "!\t!" << column->GetInt64AsString( i );
#endif
                if ( column->GetDataType().HasNull )
                {
#ifndef NDEBUG
                    std::cout << "[N]";
#endif
                    continue;
                }
            }
        }
#ifndef NDEBUG
        std::cout << std::endl;
#endif
    }

    AriesTransManager::GetInstance().EndTransaction( tx, TransactionStatus::COMMITTED );
    return true;
}

AbstractMemTablePointer JsonExecutor::Run()
{
    return nullptr;
}

}