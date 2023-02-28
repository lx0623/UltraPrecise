/*
 * TestAriesExchangeNode.cpp
 *
 *  Created on: Sep 1, 2020
 *      Author: lichi
 */
#include <gtest/gtest.h>
#include "AriesEngine/AriesExchangeNode.h"
#include "AriesEngine/AriesCalcTreeGenerator.h"
#include "CudaAcc/AriesSqlOperator.h"

namespace
{
    static const int TOTAL_ROW_COUNT = 1000000;
    static const int MAX_VALUE = 3000;
    static const int BLOCK_COUNT = 20;
}

class FakeExchangeSourceNode: public AriesOpNode
{
public:
    FakeExchangeSourceNode( bool retError = false )
            : m_bReturnError( retError )
    {
    }
    virtual bool Open()
    {
        auto compare = AriesCommonExpr::Create( AriesExprType::COMPARISON, static_cast< int >( AriesComparisonOpType::LT ), AriesColumnType
        {
        { AriesValueType::INT32 }, false, false } );
        auto col = AriesCommonExpr::Create( AriesExprType::COLUMN_ID, 1, AriesColumnType
        {
        { AriesValueType::INT32 }, false, false } );
        auto value = AriesCommonExpr::Create( AriesExprType::INTEGER, MAX_VALUE, AriesColumnType
        {
        { AriesValueType::INT32 }, false, false } );
        compare->AddChild( std::move( col ) );
        compare->AddChild( std::move( value ) );

        AriesCalcTreeGenerator gen;
        m_rootOp = gen.ConvertToCalcTree( compare, 0 );
        return true;
    }

    virtual void Close()
    {
    }

    virtual AriesOpResult GetNext()
    {
        auto table = std::make_unique< AriesTableBlock >();
        table->AddColumn( 1, generateColumn< int, AriesValueType::INT32 >() );

        AriesBoolArraySPtr associated = boost::get< AriesBoolArraySPtr >( m_rootOp->Process( table ) );
        auto outIndex = FilterAssociated( associated );
        table->UpdateIndices( outIndex );
        if( --blockCount > 0 )
            return
            {   AriesOpNodeStatus::CONTINUE, std::move( table )};
        else
        {
            if( m_bReturnError )
                return
                {   AriesOpNodeStatus::ERROR, std::move( table )};
            else
                return
                {   AriesOpNodeStatus::END, std::move( table )};
        }
    }

private:
    template< typename T, AriesValueType value_type >
    AriesColumnSPtr generateColumn()
    {
        auto column = std::make_shared< AriesColumn >();
        auto buffer = std::make_shared< AriesDataBuffer >( AriesColumnType
        {
        { value_type }, false, false }, TOTAL_ROW_COUNT );

        auto* p = ( T* )buffer->GetData();
        for( int i = 0; i < TOTAL_ROW_COUNT; i++ )
        {
            p[i] = i;
        }

        column->AddDataBuffer( buffer );
        return column;
    }
    bool m_bReturnError;
    AEExprNodeUPtr m_rootOp;
    int blockCount = BLOCK_COUNT;
};

TEST(UT_TestAriesExchangeNode, filter)
{
    int deviceCount;
    cudaGetDeviceCount( &deviceCount );

    vector< int > deviceIds;

    AriesExchangeNode node;
    for( int i = 0; i < deviceCount; ++i )
    {
        deviceIds.push_back( i );
        node.AddSourceNode( std::make_shared< FakeExchangeSourceNode >() );
    }
    node.SetDispatchInfo( 0, deviceIds );

    ASSERT_TRUE( node.Open() );
    AriesOpResult result
    { AriesOpNodeStatus::CONTINUE, std::make_unique< AriesTableBlock >() };

    AriesOpResult block;
    do
    {
        block = node.GetNext();
        result.TableBlock->AddBlock( std::move( block.TableBlock ) );
    } while( block.Status != AriesOpNodeStatus::END );

    ASSERT_TRUE( result.TableBlock->GetRowCount() == MAX_VALUE * deviceCount * BLOCK_COUNT );
}

TEST(UT_TestAriesExchangeNode, filter_error)
{
    int deviceCount;
    cudaGetDeviceCount( &deviceCount );

    vector< int > deviceIds;

    AriesExchangeNode node;
    for( int i = 0; i < deviceCount; ++i )
    {
        deviceIds.push_back( i );
        node.AddSourceNode( std::make_shared< FakeExchangeSourceNode >( i == 0 ) );
    }
    node.SetDispatchInfo( 0, deviceIds );

    ASSERT_TRUE( node.Open() );
    AriesOpResult result
    { AriesOpNodeStatus::CONTINUE, std::make_unique< AriesTableBlock >() };

    AriesOpResult block;
    do
    {
        block = node.GetNext();
        if( block.TableBlock )
            result.TableBlock->AddBlock( std::move( block.TableBlock ) );
    } while( block.Status != AriesOpNodeStatus::END && block.Status != AriesOpNodeStatus::ERROR );

    ASSERT_TRUE( block.TableBlock == nullptr );
    ASSERT_TRUE( block.Status == AriesOpNodeStatus::ERROR );
}

