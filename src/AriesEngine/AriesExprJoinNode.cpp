//
// Created by david shen on 2019-07-23.
//

#include <fstream>
#include "AriesExprJoinNode.h"
#include "../datatypes/AriesDatetimeTrans.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"

BEGIN_ARIES_ENGINE_NAMESPACE

using Decimal = aries_acc::Decimal;

/*
 * AriesExprJoinNode start
 */
    AriesExprJoinNode::AriesExprJoinNode( AriesJoinType type )
            : m_joinType( type )
    {
    }

    AriesExprJoinNode::~AriesExprJoinNode()
    {
        m_children.clear();
    }

    void AriesExprJoinNode::SetJoinType( AriesJoinType joinType )
    {
        m_joinType = joinType;
        for( auto & child : m_children )
        {
            child->SetJoinType( joinType );
        }
    }

    void AriesExprJoinNode::AddChild( AriesExprJoinNodeUPtr child )
    {
        m_children.push_back( std::move( child ) );
    }

    size_t AriesExprJoinNode::GetChildCount() const
    {
        return m_children.size();
    }

    const AriesExprJoinNode* AriesExprJoinNode::GetRawChild( int index ) const
    {
        ARIES_ASSERT( index >= 0 && std::size_t( index ) < m_children.size(),
                "index: " + to_string( index ) + ", m_children.size(): " + to_string( m_children.size() ) );
        return m_children[index].get();
    }

    bool AriesExprJoinNode::IsFromLeftTable() const
    {
        for( const auto & child : m_children )
        {
            if( child->IsFromLeftTable() )
            {
                return true;
            }
        }
        return false;
    }

    bool AriesExprJoinNode::IsLiteral( const AriesExprJoinNodeResult& value ) const
    {
        const auto& type = value.type();
        return type == typeid(bool) || type == typeid(int32_t) || type == typeid(int64_t) || type == typeid(float) || type == typeid(double)
                || type == typeid(string) || type == typeid(Decimal) || type == typeid(AriesDate) || type == typeid(AriesDatetime);
    }

    bool AriesExprJoinNode::IsIntArray( const AriesExprJoinNodeResult& value ) const
    {
        return value.type() == typeid(AriesInt32ArraySPtr);
    }

    bool AriesExprJoinNode::IsDataBuffer( const AriesExprJoinNodeResult& value ) const
    {
        return value.type() == typeid(AriesDataBufferSPtr);
    }

    bool AriesExprJoinNode::IsJoinKeyPair( const AriesExprJoinNodeResult& value ) const
    {
        return value.type() == typeid(JoinPair);
    }

    AriesDataBufferSPtr AriesExprJoinNode::ConvertLiteralToBuffer( const AriesExprJoinNodeResult& value, AriesColumnType columnType ) const
    {
        ARIES_ASSERT( IsLiteral( value ), "value type: " + string( value.type().name() ) );
        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( columnType );
        result->AllocArray( 1, true );
        switch( value.which() )
        {
            case 0:
                // bool: true or false -> int32: 1 or 0
                InitValueBuffer( result->GetData(), boost::get< bool >( value ) ? 1 : 0 );
                break;
            case 1:
                // int32
                InitValueBuffer( result->GetData(), boost::get< int32_t >( value ) );
                break;
            case 2:
                // int64
                InitValueBuffer( result->GetData(), boost::get< int64_t >( value ) );
                break;
            case 3:
                // float
                InitValueBuffer( result->GetData(), boost::get< float >( value ) );
                break;
            case 4:
                // double
                InitValueBuffer( result->GetData(), boost::get< double >( value ) );
                break;
            case 5:
                // decimal
                InitValueBuffer( result->GetData(), boost::get< Decimal >( value ) );
                break;
            case 6:
                // AriesDate
                InitValueBuffer( result->GetData(), boost::get< AriesDate >( value ) );
                break;
            case 7:
                // AriesDatetime
                InitValueBuffer( result->GetData(), boost::get< AriesDatetime >( value ) );
                break;
            case 8:
            {
                //if the param is longer than column's, we just cut it down to the columnType's size. otherwise, fill 0 until the size matches.
                string param = boost::get< string >( value );
                param.resize( columnType.GetDataTypeSize(), 0 );
                InitValueBuffer( result->GetData(), columnType, param.c_str() );
                break;
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "converting data type " + string( value.type().name() ) + " for JOIN expression" );
                break;
        }
        return result;
    }

    /*
     * AriesExprJoinComparisonNode start
     */

    unique_ptr< AriesExprJoinComparisonNode > AriesExprJoinComparisonNode::Create( AriesJoinType type, AriesComparisonOpType opType )
    {
        return unique_ptr< AriesExprJoinComparisonNode >( new AriesExprJoinComparisonNode( type, opType ) );
    }
    AriesExprJoinComparisonNode::AriesExprJoinComparisonNode( AriesJoinType type, AriesComparisonOpType opType )
            : AriesExprJoinNode( type ), m_opType( opType )
    {
    }

    AriesExprJoinComparisonNode::~AriesExprJoinComparisonNode()
    {
    }

    AriesExprJoinNodeResult AriesExprJoinComparisonNode::Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const
    {
        ARIES_ASSERT( m_children.size() == 2, "m_children.size(): " + to_string( m_children.size() ) );
        AriesExprJoinNodeResult result;
        AriesExprJoinNodeResult leftRes = m_children[0]->Process( leftTable, rightTable );
        AriesExprJoinNodeResult rightRes = m_children[1]->Process( leftTable, rightTable );

        if( !m_children[0]->IsFromLeftTable() )
        {
            ARIES_ASSERT( m_children[1]->IsFromLeftTable(), "m_children[1]->IsFromLeftTable() error" );
            std::swap( leftRes, rightRes );
        }

        ARIES_ASSERT( IsDataBuffer( leftRes ) && IsDataBuffer( rightRes ),
                "leftRes type: " + string( leftRes.type().name() ) + ", rightRes type: " + string( rightRes.type().name() ) );

        if( m_joinType == AriesJoinType::SEMI_JOIN || m_joinType == AriesJoinType::ANTI_JOIN )
        {
            AriesDataBufferSPtr leftData = boost::get< AriesDataBufferSPtr >( leftRes );
            AriesDataBufferSPtr rightData = boost::get< AriesDataBufferSPtr >( rightRes );

            if( m_opType != AriesComparisonOpType::EQ )
            {
                // TODO
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "non-EQ operator for ANTI-JOIN expression" );
            }
            if( m_joinType == AriesJoinType::SEMI_JOIN )
            {
                // result = SemiJoin( leftData->Clone(), rightData->Clone(), nullptr );
            }
            else
            {
                // result = AntiJoin( leftData->Clone(), rightData->Clone(), nullptr );
            }
        }
        else
        {
            switch( m_opType )
            {
                case AriesComparisonOpType::EQ:
                    result = ProcessEqual( leftRes, rightRes );
                    break;
                default:
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "non-EQ operator for JOIN expression" );
                    break;
            }
        }
        return result;
    }

    AriesExprJoinNodeResult AriesExprJoinComparisonNode::ProcessEqual( const AriesExprJoinNodeResult& leftData,
            const AriesExprJoinNodeResult& rightData ) const
    {
        const AriesDataBufferSPtr leftBuf = boost::get< AriesDataBufferSPtr >( leftData );
        const AriesDataBufferSPtr rightBuf = boost::get< AriesDataBufferSPtr >( rightData );
        JoinPair result;
        switch( m_joinType )
        {
            case AriesJoinType::LEFT_JOIN:
            {
                // result = LeftJoin( leftBuf->Clone(), rightBuf->Clone() );
                break;
            }
            case AriesJoinType::INNER_JOIN:
            {
                // result = InnerJoin( leftBuf->Clone(), rightBuf->Clone() );
                //result = ProcessInnerJoin( leftData, rightData );
                break;
            }
            case AriesJoinType::RIGHT_JOIN:
            {
                // result = RightJoin( leftBuf->Clone(), rightBuf->Clone() );
                break;
            }
            case AriesJoinType::FULL_JOIN:
            {
                // result = FullJoin( leftBuf->Clone(), rightBuf->Clone() );
                break;
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "type " + GetAriesJoinTypeName( m_joinType ) + "for JOIN EQ expression" );
                break;
        }

        return result;
    }

    JoinPair AriesExprJoinComparisonNode::ProcessInnerJoin( const AriesExprJoinNodeResult& leftData, const AriesExprJoinNodeResult& rightData ) const
    {
        const AriesDataBufferSPtr leftBuf = boost::get< AriesDataBufferSPtr >( leftData );
        const AriesDataBufferSPtr rightBuf = boost::get< AriesDataBufferSPtr >( rightData );
        auto itLeft = m_sortedBuffers.find( leftBuf->GetId() );
        if( itLeft == m_sortedBuffers.end() )
            itLeft = m_sortedBuffers.insert(
            { leftBuf->GetId(), SortOneColumn( leftBuf, AriesOrderByType::ASC ) } ).first;
        auto itRight = m_sortedBuffers.find( rightBuf->GetId() );
        if( itRight == m_sortedBuffers.end() )
            itRight = m_sortedBuffers.insert(
            { rightBuf->GetId(), SortOneColumn( rightBuf, AriesOrderByType::ASC ) } ).first;
        return InnerJoin( itLeft->second.first, itRight->second.first, itLeft->second.second,
                itRight->second.second );
    }

    AriesComparisonOpType AriesExprJoinComparisonNode::GetComparisonType() const
    {
        return m_opType;
    }

    string AriesExprJoinComparisonNode::ToString() const
    {
        ARIES_ASSERT( m_children.size() == 2, "m_children.size(): " + to_string( m_children.size() ) );
        string ret = "(";
        ret += m_children[0]->ToString();
        ret += " " + ComparisonOpToString( m_opType ) + " ";
        ret += m_children[1]->ToString();
        ret += ")";
        return ret;
    }

    /*
     * AriesExprJoinColumnIdNode
     */
    unique_ptr< AriesExprJoinColumnIdNode > AriesExprJoinColumnIdNode::Create( AriesJoinType type, int columnId )
    {
        return unique_ptr< AriesExprJoinColumnIdNode >( new AriesExprJoinColumnIdNode( type, columnId ) );
    }

    AriesExprJoinColumnIdNode::AriesExprJoinColumnIdNode( AriesJoinType type, int columnId )
            : AriesExprJoinNode( type ), m_columnId( columnId )
    {
        ARIES_ASSERT( columnId != 0, "columnId: " + to_string( columnId ) );
    }

    AriesExprJoinColumnIdNode::~AriesExprJoinColumnIdNode()
    {

    }

    bool AriesExprJoinColumnIdNode::IsFromLeftTable() const
    {
        ARIES_ASSERT( m_columnId != 0, "m_columnId: " + to_string( m_columnId ) );
        return m_columnId > 0;
    }

    int AriesExprJoinColumnIdNode::GetId() const
    {
        return m_columnId;
    }

    void AriesExprJoinColumnIdNode::ReverseColumnId()
    {
        m_columnId = -m_columnId;
    }

    AriesExprJoinNodeResult AriesExprJoinColumnIdNode::Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const
    {
        ARIES_ASSERT( m_columnId != 0, "m_columnId: " + to_string( m_columnId ) );
        const AriesTableBlockUPtr* table = &leftTable;
        int columnId = m_columnId;
        if( columnId < 0 )
        {
            columnId = -columnId;
            table = &rightTable;
            //cout << "right table columnid:" << m_columnId << endl;
        }
//        else
//        {
//            cout << "left table columnid:" << m_columnId << endl;
//        }
//        std::cout << ( *table )->GetColumnCount() << "\t" << columnId << "\n";
        ARIES_ASSERT( columnId > 0 && columnId <= ( *table )->GetColumnCount(),
                "columnId: " + to_string( columnId ) + ", ( *table )->GetColumnCount(): " + to_string( ( *table )->GetColumnCount() ) );
        AriesDataBufferSPtr result = ( *table )->GetColumnBuffer( columnId );
        result->PrefetchToGpu();
        return result;
    }

    string AriesExprJoinColumnIdNode::ToString() const
    {
        ARIES_ASSERT( m_columnId != 0, "m_columnId: " + to_string( m_columnId ) );
        if( m_columnId > 0 )
            return "columnid_" + std::to_string( m_columnId ) + "_left_";
        else
            return "columnid_" + std::to_string( -m_columnId ) + "_right_";
    }

    /*
     * AriesExprJoinAndOrNode start
     */
    unique_ptr< AriesExprJoinAndOrNode > AriesExprJoinAndOrNode::Create( AriesJoinType type, AriesLogicOpType opType )
    {
        return unique_ptr< AriesExprJoinAndOrNode >( new AriesExprJoinAndOrNode( type, opType ) );
    }

    AriesExprJoinAndOrNode::AriesExprJoinAndOrNode( AriesJoinType type, AriesLogicOpType opType )
            : AriesExprJoinNode( type ), m_opType( opType )
    {

    }

    AriesExprJoinAndOrNode::~AriesExprJoinAndOrNode()
    {

    }

    void AriesExprJoinAndOrNode::GetAllComparisonNodes( const AriesExprJoinNode* root, vector< const AriesExprJoinComparisonNode* > &nodes,
            int& eqNodeIndex ) const
    {
        ARIES_ASSERT( root, "root is nullptr" );
        auto node = dynamic_cast< const AriesExprJoinComparisonNode* >( root );
        if( node )
        {
            nodes.push_back( node );
            if( node->GetComparisonType() == AriesComparisonOpType::EQ && eqNodeIndex < 0 )
            {
                eqNodeIndex = nodes.size() - 1;
            }
        }
        else
        {
            int count = root->GetChildCount();
            for( int i = 0; i < count; ++i )
            {
                GetAllComparisonNodes( root->GetRawChild( i ), nodes, eqNodeIndex );
            }
        }
    }

    void AriesExprJoinAndOrNode::InitColumnPair( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable,
            const vector< const AriesExprJoinComparisonNode* >& nodes, int skipIndex, std::vector< ColumnsToCompare > &columnPairs ) const
    {
        ARIES_ASSERT( !nodes.empty(), "nodes is empty" );

        size_t count = nodes.size();
        ARIES_ASSERT( skipIndex < 0 || std::size_t( skipIndex ) < count, "skipIndex: " + to_string( skipIndex ) + ", count: " + to_string( count ) );
        if( skipIndex >= 0 )
            columnPairs.resize( count - 1 );
        else
            columnPairs.resize( count );
        const AriesExprJoinComparisonNode* node;
        int index = 0;
        for( std::size_t i = 0; i < count; ++i )
        {
            if( skipIndex < 0 || i != std::size_t( skipIndex ) )
            {
                node = nodes[i];
                ColumnsToCompare& pair = columnPairs[index++];
                FillColumnPairValue( leftTable, rightTable, node, pair );
            }
            else
                continue;
        }
    }

    static AriesDataBufferSPtr generateDataBuffer( AriesExprJoinNodeResult source, size_t count )
    {
        if( source.type() == typeid(int32_t) )
        {
            return CreateDataBufferWithValue( boost::get< int32_t >( source ), count );
        }
        else if( source.type() == typeid(aries_acc::Decimal) )
        {
            return CreateDataBufferWithValue( boost::get< aries_acc::Decimal >( source ), count );
        }
        else if( source.type() == typeid(aries_acc::AriesDate) )
        {
            return CreateDataBufferWithValue( boost::get< aries_acc::AriesDate >( source ), count );
        }
        else if( source.type() == typeid(aries_acc::AriesDatetime) )
        {
            return CreateDataBufferWithValue( boost::get< aries_acc::AriesDatetime >( source ), count );
        }
        else if( source.type() == typeid(string) )
        {
            return CreateDataBufferWithValue( boost::get< string >( source ), count );
        }
        else
        {
            return nullptr;
        }
    }

    void AriesExprJoinAndOrNode::FillColumnPairValue( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable,
            const AriesExprJoinComparisonNode* node, ColumnsToCompare& pair ) const
    {
        ARIES_ASSERT( node->GetChildCount() == 2, "node->GetChildCount(): " + to_string( node->GetChildCount() ) );
        AriesExprJoinNodeResult leftRes = node->GetRawChild( 0 )->Process( leftTable, rightTable );
        AriesExprJoinNodeResult rightRes = node->GetRawChild( 1 )->Process( leftTable, rightTable );
        bool bNeedSwap = false;

        // TODO: 常量之间的比较可以在前端优化掉
        if( !node->GetRawChild( 0 )->IsFromLeftTable() )
        {
            if( node->GetRawChild( 1 )->IsFromLeftTable() )
            {
                std::swap( leftRes, rightRes );
                bNeedSwap = true;
            }
        }

        if ( !IsDataBuffer( rightRes ) )
        {
            LOG(INFO) << "here generate constant column buffer for right table, count: " << rightTable->GetRowCount();
            auto data = generateDataBuffer( rightRes, rightTable->GetRowCount() );   
            ARIES_ASSERT( data, "invalid right data type" );
            rightRes = data;
        }

        if ( !IsDataBuffer( leftRes ) )
        {
            LOG(INFO) << "here generate constant column buffer for left table, count: " << rightTable->GetRowCount();
            auto data = generateDataBuffer( leftRes, leftTable->GetRowCount() );
            ARIES_ASSERT( data, "invalid left data type" );
            leftRes = data;
        }

        ARIES_ASSERT(IsDataBuffer(leftRes) && IsDataBuffer(rightRes),
                     "leftRes: " + string(leftRes.type().name()) + ", rightRes: " + string(rightRes.type().name()));
        pair.LeftColumn = boost::get< AriesDataBufferSPtr >( leftRes );
        pair.RightColumn = boost::get< AriesDataBufferSPtr >( rightRes );
        pair.OpType = bNeedSwap ? SwapComparisonType( node->GetComparisonType() ) : node->GetComparisonType();
    }

    AriesExprJoinNodeResult AriesExprJoinAndOrNode::Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const
    {
        ARIES_ASSERT( leftTable && rightTable && m_children.size() == 2,
                "leftTable is nullptr: " + to_string( leftTable == nullptr ) + "rightTable is nullptr: " + to_string( rightTable == nullptr )
                        + ", m_children.size(): " + to_string( m_children.size() ) );
        AriesInt32ArraySPtr result;
        vector< const AriesExprJoinComparisonNode* > nodes;
        int eqNodeIndex = -1;
        GetAllComparisonNodes( this, nodes, eqNodeIndex );
        if( -1 == eqNodeIndex )
        {
            ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "Join expression without equal condition" );
        }
        std::vector< ColumnsToCompare > columnCompares;
        InitColumnPair( leftTable, rightTable, nodes, eqNodeIndex, columnCompares );

        AriesDataBufferSPtr leftData;
        AriesDataBufferSPtr rightData;
        if( eqNodeIndex != -1 )
        {
            ARIES_ASSERT( eqNodeIndex >= 0 && std::size_t( eqNodeIndex ) < nodes.size(),
                    "eqNodeIndex: " + to_string( eqNodeIndex ) + "nodes.size(): " + to_string( nodes.size() ) );
            const AriesExprJoinComparisonNode* node = nodes[eqNodeIndex];
            ARIES_ASSERT( node->GetChildCount() == 2, "node->GetChildCount(): " + to_string( node->GetChildCount() ) );
            AriesExprJoinNodeResult leftRes = node->GetRawChild( 0 )->Process( leftTable, rightTable );
            AriesExprJoinNodeResult rightRes = node->GetRawChild( 1 )->Process( leftTable, rightTable );
            if( !node->GetRawChild( 0 )->IsFromLeftTable() )
            {
                ARIES_ASSERT( node->GetRawChild( 1 )->IsFromLeftTable(), "node->GetRawChild( 1 )->IsFromLeftTable() error" );
                std::swap( leftRes, rightRes );
            }
            ARIES_ASSERT( IsDataBuffer( leftRes ) && IsDataBuffer( rightRes ),
                    "leftRes type: " + string( leftRes.type().name() ) + ", rightRes: " + string( rightRes.type().name() ) );
            leftData = boost::get< AriesDataBufferSPtr >( leftRes );
            rightData = boost::get< AriesDataBufferSPtr >( rightRes );
        }
        switch( m_joinType )
        {
            case AriesJoinType::SEMI_JOIN:
            {
                // return SemiJoin( leftData->Clone(), rightData->Clone(), &columnCompares );
                break;
            }
            case AriesJoinType::ANTI_JOIN:
            {
                // return AntiJoin( leftData->Clone(), rightData->Clone(), &columnCompares );
                break;
            }
            case AriesJoinType::LEFT_JOIN:
            {
                // return LeftJoin( leftData->Clone(), rightData->Clone(), &columnCompares );
                break;
            }
            case AriesJoinType::INNER_JOIN:
            {
                // return InnerJoin( leftData->Clone(), rightData->Clone(), &columnCompares );
                break;
            }
            case AriesJoinType::RIGHT_JOIN:
            {
                // return RightJoin( leftData->Clone(), rightData->Clone(), &columnCompares );
                break;
            }
            case AriesJoinType::FULL_JOIN:
            {
                // return FullJoin( leftData->Clone(), rightData->Clone(), &columnCompares );
                break;
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "type " + GetAriesJoinTypeName( m_joinType ) + "for AND or OR expression" );
                break;
        }
        return result;
    }

    string AriesExprJoinAndOrNode::ToString() const
    {
        ARIES_ASSERT( m_children.size() == 2, "m_children.size(): " + to_string( m_children.size() ) );
        string ret = "(";
        ret += m_children[0]->ToString();
        ret += " " + LogicOpToString( m_opType ) + " ";
        ret += m_children[1]->ToString();
        ret += ")";
        return ret;
    }

    /*
     * AriesExprJoinLiteralNode start
     */
    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, int32_t value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, int64_t value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }
    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, float value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }
    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, double value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }
    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, Decimal value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, const string& value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, aries_acc::AriesDate value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }

    unique_ptr< AriesExprJoinLiteralNode > AriesExprJoinLiteralNode::Create( AriesJoinType type, aries_acc::AriesDatetime value )
    {
        return unique_ptr< AriesExprJoinLiteralNode >( new AriesExprJoinLiteralNode( type, value ) );
    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, int32_t value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, int64_t value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, float value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, Decimal value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, double value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, const string& value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, AriesDate value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::AriesExprJoinLiteralNode( AriesJoinType type, AriesDatetime value )
            : AriesExprJoinNode( type ), m_value( value )
    {

    }

    AriesExprJoinLiteralNode::~AriesExprJoinLiteralNode()
    {

    }

    AriesExprJoinNodeResult AriesExprJoinLiteralNode::Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const
    {
        return m_value;
    }

    string AriesExprJoinLiteralNode::ToString() const
    {
        switch( m_value.which() )
        {
            case 0:
                return std::to_string( boost::get< int32_t >( m_value ) );
            case 1:
                return std::to_string( boost::get< int64_t >( m_value ) );
            case 2:
                return std::to_string( boost::get< float >( m_value ) );
            case 3:
                return std::to_string( boost::get< double >( m_value ) );
            case 4:
            {
                Decimal d = boost::get< Decimal >( m_value );
                char decimal[64];
                return d.GetDecimal( decimal );
            }
            case 5:
                return "'" + boost::get< string >( m_value ) + "'";
            case 6:
            {
                AriesDate date = boost::get< AriesDate >( m_value );
                return AriesDatetimeTrans::GetInstance().ToString( date );
            }
            case 7:
            {
                AriesDatetime date = boost::get< AriesDatetime >( m_value );
                return AriesDatetimeTrans::GetInstance().ToString( date );
            }
            default:
                ARIES_ASSERT( 0, "NEED support more JoinLiteralNode result type: " + to_string( m_value.which() ) );
                break;
        }

        return "wrong literal value";
    }

    unique_ptr< AriesExprJoinCartesianProductNode > AriesExprJoinCartesianProductNode::Create( AriesJoinType type )
    {
        return unique_ptr< AriesExprJoinCartesianProductNode >( new AriesExprJoinCartesianProductNode( type ) );
    }

    AriesExprJoinCartesianProductNode::AriesExprJoinCartesianProductNode( AriesJoinType type )
            : AriesExprJoinNode( type )
    {

    }

    AriesExprJoinCartesianProductNode::~AriesExprJoinCartesianProductNode()
    {

    }

    AriesExprJoinNodeResult AriesExprJoinCartesianProductNode::Process( const AriesTableBlockUPtr& leftTable,
            const AriesTableBlockUPtr& rightTable ) const
    {
        return CartesianProductJoin( leftTable->GetRowCount(), rightTable->GetRowCount() );
    }

    string AriesExprJoinCartesianProductNode::ToString() const
    {
        return "AriesExprJoinCartesianProductNode";
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
