#include "PartitionConditionOptimizer.h"

#include "frontend/BiaodashiAuxProcessor.h"
#include "schema/SchemaManager.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"
#include "server/mysql/include/sql_class.h"
#include "utils/string_util.h"
#include "datatypes/AriesDatetimeTrans.h"

namespace aries
{

PartitionConditionOptimizer::PartitionConditionOptimizer()
{

}

std::string PartitionConditionOptimizer::ToString()
{
    return "PartitionConditionOptimizer";
}

SQLTreeNodePointer PartitionConditionOptimizer::OptimizeTree( const SQLTreeNodePointer arg_input )
{
    processNode( arg_input );
    
    return arg_input;
}

PartitionConditionOptimizer::PartitionInfo PartitionConditionOptimizer::getTablePartitionColumn( const SQLTreeNodePointer& node )
{
    assert( node->GetType() == SQLTreeNodeType::Table_NODE );
    auto rel = node->GetBasicRel();
    if ( rel->IsSubquery() )
    {
        return { -1, "" };
    }

    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    auto db = schema->GetDatabaseByName( rel->GetDb() );
    auto table = db->GetTableByName( rel->GetID() );

    assert( db && table );

    if ( !table->IsPartitioned() )
    {
        return { -1, "" };
    }
    
    auto column_index = table->GetPartitionColumnIndex();
    auto column = table->GetColumnById( column_index + 1 );
    return { column_index, column->GetName(), table->GetPartitions() };
}

void PartitionConditionOptimizer::processNode( const SQLTreeNodePointer& node )
{
    if ( node->GetType() == SQLTreeNodeType::Filter_NODE && node->GetTheChild()->GetType() == SQLTreeNodeType::Table_NODE )
    {

        processFilterNode( node );
        return;
    }

    for ( size_t i = 0; i < node->GetChildCount(); i++ )
    {
        processNode( node->GetChildByIndex( i ) );
    }
}

void PartitionConditionOptimizer::processFilterNode( const SQLTreeNodePointer& node )
{
    assert( node->GetType() == SQLTreeNodeType::Filter_NODE );

    auto table_node = node->GetTheChild();
    assert( table_node->GetType() == SQLTreeNodeType::Table_NODE );

    auto partition_info = getTablePartitionColumn( table_node );
    if ( partition_info.ColumnIndex == -1 )
    {
        return;
    }

    DLOG(INFO) << "table(" + table_node->GetBasicRel()->GetID() + ") is partitioned!";

    BiaodashiAuxProcessor processor;
    auto condition = node->GetFilterStructure();
    auto condition_array = processor.generate_and_list( condition );

    std::vector< BiaodashiPointer > partition_conditions;
    std::vector< BiaodashiPointer > non_partition_conditions;
    for ( size_t i = 0; i < condition_array.size(); i++ )
    {
        auto expression = std::dynamic_pointer_cast< CommonBiaodashi >( condition_array[ i ] );
        if ( expression->GetType() != BiaodashiType::Bijiao )
        {
            non_partition_conditions.emplace_back( condition_array[ i ]  );
            continue;
        }

        auto left = std::dynamic_pointer_cast< CommonBiaodashi >(  expression->GetChildByIndex( 0 ) );
        auto right = std::dynamic_pointer_cast< CommonBiaodashi >(  expression->GetChildByIndex( 1 ) );

        CommonBiaodashiPtr column_expression;
        CommonBiaodashiPtr value_expression;
        if ( left->GetType() == BiaodashiType::Lie )
        {
            column_expression = left;
        }
        else if ( left->IsCalculable() )
        {
            value_expression = left;
        }

        if ( right->GetType() == BiaodashiType::Lie )
        {
            if ( column_expression )
            {
                non_partition_conditions.emplace_back( condition_array[ i ]  );
                continue;
            }

            column_expression = right;
        }
        else if ( right->IsCalculable() )
        {
            value_expression = right;
        }

        if ( !column_expression || !value_expression )
        {
            continue;
        }

        auto column = boost::get< ColumnShellPointer >( column_expression->GetContent() );
        if ( column->GetColumnName() != partition_info.ColumnName )
        {
            non_partition_conditions.emplace_back( condition_array[ i ]  );
            continue;
        }

        aries_engine::ExpressionSimplifier simplifier( true );
        auto simplified = simplifier.Simplify( value_expression.get(), current_thd );
        if ( !simplified )
        {
            DLOG( INFO ) << "here cannot simplify expression: " << value_expression->ToString() << std::endl;
            non_partition_conditions.emplace_back( condition_array[ i ]  );
            continue;
        }

        auto comparison_type = static_cast< ComparisonType >( boost::get< int >( expression->GetContent() ) );
        table_node->AddPartitionCondition( processor.make_biaodashi_compare( column_expression, value_expression, comparison_type ) );
        switch ( comparison_type )
        {
            case ComparisonType::DaYuDengYu:
                break;
            case ComparisonType::XiaoYu:
                break;
            default:
                non_partition_conditions.emplace_back( condition_array[ i ]  );
                continue;
        }

        int64_t value = 0;
        switch( simplified->GetType() )
        {
            case AriesExprType::DATE:
            {
                value = boost::get< aries_acc::AriesDate >( simplified->GetContent() ).toTimestamp();
                break;
            }
            case AriesExprType::DATE_TIME:
            {
                value = boost::get< aries_acc::AriesDatetime >( simplified->GetContent() ).toTimestamp();
                break;
            }
            case AriesExprType::INTEGER:
            {
                auto content = simplified->GetContent();
                if ( CHECK_VARIANT_TYPE( content, int32_t ) )
                {
                    value = boost::get< int32_t >( content );
                }
                else if ( CHECK_VARIANT_TYPE( content, int64_t ) )
                {
                    value = boost::get< int64_t >( content );
                }
                else
                {
                    DLOG( INFO ) << "unhandled int type for partition condition: " << content.which() << std::endl;
                    non_partition_conditions.emplace_back( condition_array[ i ]  );
                    continue;
                }
                break;
            }
            case AriesExprType::STRING:
            {
                auto content = boost::get< std::string >( simplified->GetContent() );
                switch ( column->GetValueType() )
                {
                    case BiaodashiValueType::DATE:
                    {
                        value = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( content ).toTimestamp();
                        break;
                    }
                    case BiaodashiValueType::DATE_TIME:
                    {
                        value = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( content ).toTimestamp();
                        break;
                    }
                    default:
                    {
                        DLOG( INFO ) << "unhandled column expresstion type: " << static_cast< int >( column->GetValueType() );
                        non_partition_conditions.emplace_back( condition_array[ i ]  );
                        break;
                    }
                }
                break;
            }
            default:
            {
                DLOG( INFO ) << "unhandled type of value expression for partition condition: " << static_cast< int >( simplified->GetType() );
                non_partition_conditions.emplace_back( condition_array[ i ]  );
                continue;
            }
        }

        bool matched = false;
        for ( const auto& partition : partition_info.Partions )
        {
            if ( !partition )
            {
                DLOG( INFO ) << "here should not be nullptr";
                continue;
            }

            if ( partition->m_value == value )
            {
                matched = true;
                partition_conditions.emplace_back( condition_array[ i ]  );
                break;
            }
        }

        if ( !matched )
        {
            non_partition_conditions.emplace_back( condition_array[ i ]  );
        }
    }

    if ( non_partition_conditions.empty() )
    {
        node->GetParent()->CompletelyResetAChild_WithoutReCalculateInvolvedTableList( node, node->GetTheChild() );
        DLOG( INFO ) << "here remove filter node" << std::endl;
    }
    else
    {
        node->SetFilterStructure( processor.make_biaodashi_from_and_list( non_partition_conditions ) );
    }
}

} // namespace aries
