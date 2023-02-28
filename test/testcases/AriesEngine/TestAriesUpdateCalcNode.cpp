#include <gtest/gtest.h>

#include "AriesEngine/AriesUpdateCalcNode.h"
#include "../../TestUtils.h"
#include "CudaAcc/DynamicKernel.h"

using namespace aries_engine;
using namespace aries_acc;

#define ROW_COUNT 10
#define BLOCK_COUNT 3

class FakeSourceNode : public AriesOpNode
{
public:
    virtual bool Open()
    {
        block_count = BLOCK_COUNT;
        return true;
    }

    virtual void Close()
    {
    }

    virtual AriesOpResult GetNext()
    {
        block_count --;
        if ( block_count >= 0 )
        {
            auto table = std::make_unique< AriesTableBlock >();

            table->AddColumn( 1, generateColumn< int, AriesValueType::INT32 >( 10 ) );
            table->AddColumn( 2, generateColumn< int, AriesValueType::INT32 >( 20 ) );
            table->AddColumn( 3, generateColumn< int, AriesValueType::INT32 >( 30 ) );
            table->AddColumn( 4, generateColumn< int, AriesValueType::INT32 >( 40 ) );
            table->AddColumn( 5, generateColumn< int, AriesValueType::INT32 >( 50 ) );
            table->AddColumn( 6, generateColumn< aries_acc::Decimal, AriesValueType::DECIMAL >( aries_acc::Decimal(1) ) );
            table->AddColumn( 7, generateColumn< aries_acc::AriesDate, AriesValueType::DATE >( aries_acc::AriesDate( 2019, 10, 9 ) ) );

            return { block_count == 0 ? AriesOpNodeStatus::END : AriesOpNodeStatus::CONTINUE, std::move( table ) };
        }

        return { AriesOpNodeStatus::END, nullptr };
    }

private:

    template< typename T, AriesValueType value_type >
    AriesColumnSPtr generateColumn( T value )
    {
        auto column = std::make_shared< AriesColumn >();
        auto buffer = std::make_shared< AriesDataBuffer >( AriesColumnType{ { value_type }, false, false }, 10 );

        auto* p = ( T* )buffer->GetData();
        for ( int i = 0; i < ROW_COUNT; i++ )
        {
            p[ i ] = value;
        }

        column->AddDataBuffer( buffer );
        return column;
    }

    int block_count = BLOCK_COUNT;
};

/**
 * set col1=col1 + 2, col3=col1, col4=col3+4, col5=100, col6=1.7, col7=date('2019-10-10')
 */
ARIES_UNIT_TEST( AriesUpdateCalcNode, GetNext )
{
    auto node = std::make_shared< AriesUpdateCalcNode >();
    node->SetSourceNode( std::make_shared< FakeSourceNode >() );
    node->SetNodeId( 0 );

    std::vector< int > column_ids;
    column_ids.emplace_back( 1 );
    column_ids.emplace_back( 3 );
    column_ids.emplace_back( 4 );
    column_ids.emplace_back( 5 );
    column_ids.emplace_back( 6 );
    column_ids.emplace_back( 7 );

    node->SetColumnIds( column_ids );

    int exprId = 0;

    auto col1 = AriesCommonExpr::Create( AriesExprType::COLUMN_ID, 1, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto col3 = AriesCommonExpr::Create( AriesExprType::COLUMN_ID, 3, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto col4 = AriesCommonExpr::Create( AriesExprType::COLUMN_ID, 4, AriesColumnType{ { AriesValueType::INT32 }, false, false } );

    auto const1 = AriesCommonExpr::Create( AriesExprType::INTEGER, 2, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto const2 = AriesCommonExpr::Create( AriesExprType::INTEGER, 4, AriesColumnType{ { AriesValueType::INT32 }, false, false } );

    auto const3 = AriesCommonExpr::Create( AriesExprType::INTEGER, 100, AriesColumnType{ { AriesValueType::INT32 }, false, false } );
    auto const4 = AriesCommonExpr::Create( AriesExprType::DECIMAL, aries_acc::Decimal( "1.7" ), AriesColumnType{ { AriesValueType::DECIMAL }, false, false } );
    auto const5 = AriesCommonExpr::Create( AriesExprType::DATE, aries_acc::AriesDate( 2019, 10, 10 ), AriesColumnType{ { AriesValueType::DATE }, false, false } );

    auto calc1 = AriesCommonExpr::Create( AriesExprType::CALC,
                                          static_cast< int >( AriesCalculatorOpType::ADD ),
                                          AriesColumnType{ { AriesValueType::INT32 }, false, false }
                                        );
    calc1->AddChild( std::move( col1->Clone() ) );
    calc1->AddChild( std::move( const1 ) );
    calc1->SetId( ++exprId );

    auto calc2 = AriesCommonExpr::Create( AriesExprType::CALC, 
                                          static_cast< int >( AriesCalculatorOpType::ADD ),
                                          AriesColumnType{ { AriesValueType::INT32 }, false, false }
                                        );

    calc2->AddChild( std::move( col3 ) );
    calc2->AddChild( std::move( const2 ) );
    calc2->SetId( ++exprId );

    std::vector< AriesCommonExprUPtr > exprs;
    exprs.emplace_back( std::move( calc1 ) );
    exprs.emplace_back( std::move( col1 ) );
    exprs.emplace_back( std::move( calc2 ) );
    exprs.emplace_back( std::move( const3 ) );
    exprs.emplace_back( std::move( const4 ) );
    exprs.emplace_back( std::move( const5 ) );

    node->SetCalcExprs( exprs );

    auto code = R"(#include "functions.hxx"
#include "AriesDateFormat.hxx"
#include "aries_char.hxx"
#include "decimal.hxx"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesIntervalTime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"
#include "AriesYear.hxx"
#include "AriesTimeCalc.hxx"
#include "AriesSqlFunctions.hxx"
#include "AriesColumnDataIterator.hxx"
using namespace aries_acc;

)" + node->GetCudaKernelCode();

    AriesDynamicCodeInfo codeInfo;
    codeInfo.KernelCode = code;
    auto modules = AriesDynamicKernelManager::GetInstance().CompileKernels( codeInfo );
    node->SetCuModule( modules->Modules );

    ASSERT_TRUE( !modules->Modules.empty() );
    do
    {
        auto result = node->GetNext();
        ASSERT_NE( result.Status, AriesOpNodeStatus::ERROR );

        if ( result.TableBlock )
        {
            auto& table = result.TableBlock;

            ASSERT_EQ( table->GetColumnCount(), 6 );
            ASSERT_EQ( table->GetRowCount(), ROW_COUNT );

            for ( int i = 0; i < ROW_COUNT; i++ )
            {
                ASSERT_EQ( table->GetColumnBuffer( 1 )->GetInt32AsString( i ), "12" );
                // std::cout << table->GetColumnBuffer( 2 )->GetInt32AsString( i ) << " : " << table->GetColumnBuffer( 3 )->GetInt32AsString( i ) << std::endl;
                ASSERT_EQ( table->GetColumnBuffer( 2 )->GetInt32AsString( i ), "12" );
                ASSERT_EQ( table->GetColumnBuffer( 3 )->GetInt32AsString( i ), "16" );
                ASSERT_EQ( table->GetColumnBuffer( 4 )->GetInt32AsString( i ), "100" );
                ASSERT_EQ( table->GetColumnBuffer( 5 )->GetDecimalAsString( i ), "1.7" );
                ASSERT_EQ( table->GetColumnBuffer( 6 )->GetDateAsString( i ), "2019-10-10" );
            }
        }

        if ( result.Status == AriesOpNodeStatus::END )
        {
            break;
        }
    } while ( true );
}
