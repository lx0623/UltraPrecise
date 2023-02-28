//
// Created by david shen on 2019-07-23.
//

#pragma once

#include <boost/variant.hpp>
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesUtil.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"
#include "AriesDataDef.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesExprJoinNode;
    using AriesExprJoinNodeResult = boost::variant< bool, int32_t, int64_t, float, double, aries_acc::Decimal, AriesDate, AriesDatetime, string, AriesInt32ArraySPtr, AriesDataBufferSPtr, JoinPair >;
    using AriesExprJoinNodeUPtr = unique_ptr< AriesExprJoinNode >;
    class AriesExprJoinNode: protected DisableOtherConstructors
    {
    public:
        virtual ~AriesExprJoinNode();
        virtual AriesExprJoinNodeResult Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const = 0;
        void AddChild( AriesExprJoinNodeUPtr child );
        virtual string ToString() const = 0;
        virtual bool IsFromLeftTable() const;
        size_t GetChildCount() const;
        const AriesExprJoinNode* GetRawChild( int index ) const;
        void SetJoinType( AriesJoinType joinType );

    protected:
        AriesExprJoinNode( AriesJoinType type );
        bool IsLiteral( const AriesExprJoinNodeResult& value ) const;
        bool IsIntArray( const AriesExprJoinNodeResult& value ) const;
        bool IsDataBuffer( const AriesExprJoinNodeResult& value ) const;
        bool IsJoinKeyPair( const AriesExprJoinNodeResult& value ) const;
        AriesDataBufferSPtr ConvertLiteralToBuffer( const AriesExprJoinNodeResult& value, AriesColumnType columnType ) const;

        template< typename T >
        void InitValueBuffer( int8_t *dstBuf, T value ) const
        {
            *reinterpret_cast< T * >( dstBuf ) = value;
        }

        template< typename T >
        void InitValueBuffer( int8_t *dstBuf, AriesColumnType columnType, T *value ) const
        {
            AriesDataType dstType = columnType.DataType;
            switch( dstType.ValueType )
            {
                case AriesValueType::CHAR:
                    memcpy( dstBuf, value, dstType.Length );
                    break;
                default:
                    assert( 0 );
                    ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "initialization of non-CHAR type " + GetValueTypeAsString( columnType ) );
            }
        }

    protected:
        vector< AriesExprJoinNodeUPtr > m_children;
        AriesJoinType m_joinType;
    };

    class AriesExprJoinComparisonNode: public AriesExprJoinNode
    {
    public:
        static unique_ptr< AriesExprJoinComparisonNode > Create( AriesJoinType type, AriesComparisonOpType opType );
        virtual ~AriesExprJoinComparisonNode();
        virtual AriesExprJoinNodeResult Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const;
        virtual string ToString() const;
        AriesComparisonOpType GetComparisonType() const;

    protected:
        AriesExprJoinComparisonNode( AriesJoinType type, AriesComparisonOpType opType );

    private:
        AriesExprJoinNodeResult ProcessEqual( const AriesExprJoinNodeResult& leftData, const AriesExprJoinNodeResult& rightData ) const;
        JoinPair ProcessInnerJoin( const AriesExprJoinNodeResult& leftData, const AriesExprJoinNodeResult& rightData ) const;

    private:
        AriesComparisonOpType m_opType;
        mutable map< uint32_t, pair< AriesDataBufferSPtr, AriesInt32ArraySPtr > > m_sortedBuffers;
    };

    class AriesExprJoinColumnIdNode: public AriesExprJoinNode
    {
    public:
        static unique_ptr< AriesExprJoinColumnIdNode > Create( AriesJoinType type, int columnId );
        virtual ~AriesExprJoinColumnIdNode();
        virtual AriesExprJoinNodeResult Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const;
        virtual string ToString() const;
        virtual bool IsFromLeftTable() const;
        virtual void ReverseColumnId();
        int GetId() const;
    protected:
        AriesExprJoinColumnIdNode( AriesJoinType type, int columnId );

    private:
        int m_columnId;
    };

    class AriesExprJoinAndOrNode: public AriesExprJoinNode
    {
    public:
        static unique_ptr< AriesExprJoinAndOrNode > Create( AriesJoinType type, AriesLogicOpType opType );
        virtual ~AriesExprJoinAndOrNode();
        virtual AriesExprJoinNodeResult Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const;
        virtual string ToString() const;

    protected:
        AriesExprJoinAndOrNode( AriesJoinType type, AriesLogicOpType opType );

    private:
        void GetAllComparisonNodes( const AriesExprJoinNode* root, vector< const AriesExprJoinComparisonNode* > &nodes, int& eqNodeIndex ) const;
        void InitColumnPair( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable,
                const vector< const AriesExprJoinComparisonNode* >& nodes, int skipIndex, std::vector< ColumnsToCompare > &columnPairs ) const;
        void FillColumnPairValue( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable, const AriesExprJoinComparisonNode* node,
                ColumnsToCompare& pair ) const;

    private:
        AriesLogicOpType m_opType;
    };

    class AriesExprJoinLiteralNode: public AriesExprJoinNode
    {
    public:
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, int32_t value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, int64_t value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, float value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, double value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, aries_acc::Decimal value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, const string& value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, aries_acc::AriesDate value );
        static unique_ptr< AriesExprJoinLiteralNode > Create( AriesJoinType type, aries_acc::AriesDatetime value );

        virtual ~AriesExprJoinLiteralNode();
        virtual AriesExprJoinNodeResult Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const;
        virtual string ToString() const;

    protected:
        AriesExprJoinLiteralNode( AriesJoinType type, int32_t value );
        AriesExprJoinLiteralNode( AriesJoinType type, int64_t value );
        AriesExprJoinLiteralNode( AriesJoinType type, float value );
        AriesExprJoinLiteralNode( AriesJoinType type, double value );
        AriesExprJoinLiteralNode( AriesJoinType type, aries_acc::Decimal value );
        AriesExprJoinLiteralNode( AriesJoinType type, const string& value );
        AriesExprJoinLiteralNode( AriesJoinType type, aries_acc::AriesDate value );
        AriesExprJoinLiteralNode( AriesJoinType type, aries_acc::AriesDatetime value );

    private:
        boost::variant< int32_t, int64_t, float, double, aries_acc::Decimal, string, aries_acc::AriesDate, aries_acc::AriesDatetime > m_value;
    };

    class AriesExprJoinCartesianProductNode: public AriesExprJoinNode
    {
    public:
        static unique_ptr< AriesExprJoinCartesianProductNode > Create( AriesJoinType type );
        virtual ~AriesExprJoinCartesianProductNode();
        virtual AriesExprJoinNodeResult Process( const AriesTableBlockUPtr& leftTable, const AriesTableBlockUPtr& rightTable ) const;
        virtual string ToString() const;

    protected:
        AriesExprJoinCartesianProductNode( AriesJoinType type );

    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
