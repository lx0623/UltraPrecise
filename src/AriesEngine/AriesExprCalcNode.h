/*
 * AriesExprCalcNode.h
 *
 *  Created on: Sep 13, 2018
 *      Author: lichi
 */

#pragma once

#include <boost/variant.hpp>
#include <string>
#include <set>
#include "AriesUtil.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesAssert.h"
#include "AriesUtil.h"
#include "AriesDataDef.h"

using namespace aries_acc;
using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE

    class AEExprNode;
    class AEExprAggFunctionNode;
    using AEExprNodeResult = boost::variant< bool, int32_t, int64_t, float, double, aries_acc::Decimal, string, AriesDate, AriesDatetime, AriesInterval, AriesNull, AriesTime, AriesTimestamp, AriesYear, AriesInt32ArraySPtr, AriesDataBufferSPtr, AriesBoolArraySPtr>;
    using AEExprNodeUPtr = unique_ptr< AEExprNode >;

    void DumpExprNodeResult( const AEExprNodeResult& result );

    class AEExprNode: protected DisableOtherConstructors
    {
    public:
        virtual ~AEExprNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const = 0;
        void AddChild( AEExprNodeUPtr child );
        virtual string ToString() const
        {
            return string();
        }

        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules )
        {
        }
        virtual string GetCudaKernelCode() const
        {
            return string();
        }
        virtual void GetAllParams( map< string, AEExprAggFunctionNode * > &result )
        {
        }

    protected:
        AEExprNode( int nodeId = 0, int exprIndex = -1 );
        bool IsNullValue( const AEExprNodeResult& value ) const;
        bool IsLiteral( const AEExprNodeResult& value ) const;
        bool IsIntArray( const AEExprNodeResult& value ) const;
        bool IsDataBuffer( const AEExprNodeResult& value ) const;
        bool IsString( const AEExprNodeResult& value ) const;
        bool IsInt( const AEExprNodeResult& value ) const;
        bool IsLong( const AEExprNodeResult& value ) const;
        bool IsInterval( const AEExprNodeResult& value ) const;
        bool IsAriesDate( const AEExprNodeResult& value ) const;
        bool IsAriesDatetime( const AEExprNodeResult& value ) const;
        bool IsAriesTime( const AEExprNodeResult& value ) const;
        bool IsBoolArray( const AEExprNodeResult& value ) const;
        bool IsDecimal( const AEExprNodeResult& value ) const;
        AriesDataBufferSPtr ConvertLiteralToBuffer( const AEExprNodeResult& value, AriesColumnType columnType ) const;

        template< typename T >
        void InitValueBuffer( int8_t* dstBuf, T value ) const
        {
            *reinterpret_cast< T* >( dstBuf ) = value;
        }

        template< typename T >
        void InitValueBuffer( int8_t* dstBuf, AriesColumnType columnType, T* value ) const
        {
            AriesDataType dstType = columnType.DataType;
            switch( dstType.ValueType )
            {
                case AriesValueType::CHAR:
                    memcpy( dstBuf, value, dstType.Length );
                    break;
                default:
                    ARIES_ASSERT( 0, "dstType.ValueType: " + GetValueTypeAsString( columnType ) );
            }
        }

    protected:
        vector< CUmoduleSPtr > m_cuModules;
        vector< AEExprNodeUPtr > m_children;
        int64_t m_exprId;
    };

    class AEExprDynKernelNode: public AEExprNode
    {
    public:
        AEExprDynKernelNode( int nodeId, int exprIndex,
                             vector< AriesDynamicCodeParam > && params,
                             vector< AriesDataBufferSPtr > && constValues,
                             vector< AriesDynamicCodeComparator > && comparators,
                             const string& expr,
                             const AriesColumnType& valueType );
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual AEExprNodeResult RunKernelFunction( const map< string, AriesDataBufferSPtr > &params ) const;

    protected:
        size_t GetPartitionCount( const map< string, AriesDataBufferSPtr >& params ) const;
        size_t GetPartitionCount( const AriesTableBlockUPtr& refTable ) const;
        virtual size_t GetResultItemSize() const = 0;

    protected:
        string m_cudaFunction;
        string m_cudaFunctionName;
        vector< AriesDynamicCodeParam > m_ariesParams;
        vector< AriesDataBufferSPtr > m_constValues;
        vector< AriesDynamicCodeComparator > m_ariesComparators;
        string m_expr; // the original expr;
        AriesColumnType m_valueType; //the result value type

    };

    class AEExprAndOrNode: public AEExprDynKernelNode
    {
    public:
        static unique_ptr< AEExprAndOrNode > Create( int nodeId,
                                                     int exprIndex,
                                                     AriesLogicOpType opType,
                                                     vector< AriesDynamicCodeParam > && params,
                                                     vector< AriesDataBufferSPtr > && constValues,
                                                     vector< AriesDynamicCodeComparator > && comparators,
                                                     const string& expr );
        virtual ~AEExprAndOrNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        AEExprNodeResult RunKernelFunction( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;

    protected:
        AEExprAndOrNode( int nodeId,
                         int exprIndex,
                         AriesLogicOpType opType,
                         vector< AriesDynamicCodeParam > && params,
                         vector< AriesDataBufferSPtr > && constValues,
                         vector< AriesDynamicCodeComparator > && comparators,
                         const string& expr );
        size_t GetResultItemSize() const { return sizeof( AriesBool ); }
    private:
        string GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const;
        pair< string, string > GenerateCudaFunctionEx( const vector< AriesDynamicCodeParam >& params, const string& expr ) const;
        pair< string, string > GenerateLoadDataCodeEx( const AriesDynamicCodeParam& param, int index ) const;
        string GenerateSaveDataCode( const string& paramName ) const;

    private:
        AriesLogicOpType m_opType;
    };

    class AEExprComparisonNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprComparisonNode > Create( AriesComparisonOpType opType );
        virtual ~AEExprComparisonNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;

    protected:
        AEExprComparisonNode( AriesComparisonOpType opType );

    private:
        AriesComparisonOpType m_opType;
    };

    class AEExprLiteralNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprLiteralNode > Create( int32_t value );
        static unique_ptr< AEExprLiteralNode > Create( int64_t value );
        static unique_ptr< AEExprLiteralNode > Create( float value );
        static unique_ptr< AEExprLiteralNode > Create( double value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::Decimal value );
        static unique_ptr< AEExprLiteralNode > Create( const string& value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::AriesDate value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::AriesDatetime value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::AriesTime value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::AriesTimestamp value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::AriesYear value );
        static unique_ptr< AEExprLiteralNode > Create( aries_acc::AriesNull value );
        virtual ~AEExprLiteralNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;

    protected:
        AEExprLiteralNode( int32_t value );
        AEExprLiteralNode( int64_t value );
        AEExprLiteralNode( float value );
        AEExprLiteralNode( double value );
        AEExprLiteralNode( aries_acc::Decimal value );
        AEExprLiteralNode( const string& value );
        AEExprLiteralNode( aries_acc::AriesDate value );
        AEExprLiteralNode( aries_acc::AriesDatetime value );
        AEExprLiteralNode( aries_acc::AriesTime value );
        AEExprLiteralNode( aries_acc::AriesTimestamp value );
        AEExprLiteralNode( aries_acc::AriesYear value );
        AEExprLiteralNode( aries_acc::AriesNull value );

    private:
        boost::variant< int32_t, int64_t, float, double, aries_acc::Decimal, string, aries_acc::AriesDate, aries_acc::AriesDatetime,
                aries_acc::AriesNull, aries_acc::AriesTime, aries_acc::AriesTimestamp, aries_acc::AriesYear > m_value;
    };

    class AEExprAggFunctionNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprAggFunctionNode > Create( AriesAggFunctionType functionType, bool bDistinct );
        virtual ~AEExprAggFunctionNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
        virtual void GetAllParams( map< string, AEExprAggFunctionNode * > &result );

    public:
        pair< AEExprNodeResult, AEExprNodeResult > RunKernelFunction( const AriesTableBlockUPtr& refTable, const AriesInt32ArraySPtr& associated,
                const AriesInt32ArraySPtr& groups, const AriesInt32ArraySPtr& groupFlags, const AriesDataBufferSPtr& itemCountInGroups ) const;
        string GetFunctionCode() const;
        AriesAggFunctionType GetFunctionType() const;
        bool IsDistinct() const;

    protected:
        AEExprAggFunctionNode( AriesAggFunctionType functionType, bool bDistinct );
    private:
        AriesAggFunctionType m_functionType;
        bool m_bDistinct;
    };

    class AEExprCalcNode: public AEExprDynKernelNode
    {
    public:
        static unique_ptr< AEExprCalcNode > Create( int nodeId,
                                                    int exprIndex,
                                                    map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                                                    vector< AriesDynamicCodeParam > && params,
                                                    vector< AriesDataBufferSPtr > && constValues,
                                                    vector< AriesDynamicCodeComparator > && comparators,
                                                    const string& expr,
                                                    const AriesColumnType& valueType );
        static unique_ptr< AEExprCalcNode > CreateForXmp( int nodeId,
                                                    int exprIndex,
                                                    map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                                                    vector< AriesDynamicCodeParam > && params,
                                                    vector< AriesDataBufferSPtr > && constValues,
                                                    vector< AriesDynamicCodeComparator > && comparators,
                                                    const string& expr,
                                                    const AriesColumnType& valueType,
                                                    int ansLEN,
                                                    int ansTPI );
        virtual ~AEExprCalcNode();
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    public:
        virtual void GetAllParams( map< string, AEExprAggFunctionNode * > &result );

    protected:
        AEExprCalcNode( int nodeId,
                        int exprIndex,
                        map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                        vector< AriesDynamicCodeParam > && params,
                        vector< AriesDataBufferSPtr > && constValues,
                        vector< AriesDynamicCodeComparator > && comparators,
                        const string& expr,
                        const AriesColumnType& valueType,
                        int ansLEN = 0,
                        int ansTPI = 0,
                        bool isXmp = false );
        size_t GetResultItemSize() const { return m_valueType.GetDataTypeSize(); }

    private:
        string GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const;
        pair< string, string > GenerateCudaFunction( const vector< AriesDynamicCodeParam >& params, const string& expr,
                const AriesColumnType& value ) const;
        pair< string, string > GenerateCudaFunctionXmp( const vector< AriesDynamicCodeParam >& params, const string& expr,
                const AriesColumnType& value, int ansLEN, int ansTPI ) const;
        string GenerateLoadDataCode( const AriesDynamicCodeParam& param, int index, int len ) const;
        string GenerateSaveDataCode( const AriesColumnType& type, const string& paramName ) const;

    private:
        map< string, unique_ptr< AEExprAggFunctionNode > > m_aggFunctions;
    };

    class AEExprColumnIdNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprColumnIdNode > Create( int columnId );
        virtual ~AEExprColumnIdNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void GetAllParams( map< string, AEExprAggFunctionNode * > &result );

    public:
        int GetId() const;
    protected:
        AEExprColumnIdNode( int columnId );

    private:
        int m_columnId;
    };

    class AEExprBetweenNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprBetweenNode > Create();
        virtual ~AEExprBetweenNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    protected:
        AEExprBetweenNode();
    };

    class AEExprInNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprInNode > Create( bool bHasNot = false );
        virtual ~AEExprInNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    protected:
        AEExprInNode( bool bHasNot );

    private:
        string ChildrenToString_SkipFirstOne() const;
        AriesDataBufferSPtr ConvertToDataBuffer( AriesColumnType dataType, const AriesTableBlockUPtr& refTable ) const;
        AriesDataBufferSPtr ConvertToDataBuffer( AriesColumnType dataType, const AriesDataBufferSPtr& buffer ) const;
        bool m_bHasNot;
    };

    class AEExprNotNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprNotNode > Create();
        virtual ~AEExprNotNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    protected:
        AEExprNotNode();
    };

    class AEExprLikeNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprLikeNode > Create();
        virtual ~AEExprLikeNode();
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    protected:
        AEExprLikeNode();
    };

    class AEExprSqlFunctionNode: public AEExprDynKernelNode
    {
    public:
        static unique_ptr< AEExprSqlFunctionNode >
        Create( int nodeId,
                int exprIndex,
                AriesSqlFunctionType functionType,
                map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                vector< AriesDynamicCodeParam > && params,
                vector< AriesDataBufferSPtr > && constValues,
                vector< AriesDynamicCodeComparator > && comparators,
                const string& expr,
                const AriesColumnType& valueType );
        virtual ~AEExprSqlFunctionNode();
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;

    public:
        virtual void GetAllParams( map< string, AEExprAggFunctionNode * > &result );

    protected:
        AEExprSqlFunctionNode( int nodeId,
                               int exprIndex,
                               AriesSqlFunctionType functionType,
                               map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                               vector< AriesDynamicCodeParam > && params,
                               vector< AriesDataBufferSPtr > && constValues,
                               vector< AriesDynamicCodeComparator > && comparators,
                               const string& expr,
                               const AriesColumnType& valueType );

    private:
        size_t GetResultItemSize() const { return m_valueType.GetDataTypeSize(); }
        string GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const;
        pair< string, string > GenerateCudaFunction( const vector< AriesDynamicCodeParam >& params, const string& expr,
                const AriesColumnType& value ) const;
        string GenerateLoadDataCode( const AriesDynamicCodeParam& param, int index ) const;
        string GenerateSaveDataCode( const AriesColumnType& type, const string& paramName ) const;

    private:
        AriesSqlFunctionType m_functionType;
        map< string, unique_ptr< AEExprAggFunctionNode > > m_aggFunctions;
    };

    class AEExprCaseNode: public AEExprDynKernelNode
    {
    public:
        static unique_ptr< AEExprCaseNode >
        Create( int nodeId,
                int exprIndex,
                map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                vector< AriesDynamicCodeParam > && params,
                vector< AriesDataBufferSPtr > && constValues,
                vector< AriesDynamicCodeComparator > && comparators,
                const string& expr,
                const AriesColumnType& valueType );
        virtual ~AEExprCaseNode();
        virtual string ToString() const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;
    protected:
        AEExprCaseNode( int nodeId,
                        int exprIndex,
                        map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                        vector< AriesDynamicCodeParam > && params,
                        vector< AriesDataBufferSPtr > && constValues,
                        vector< AriesDynamicCodeComparator > && comparators,
                        const string& expr,
                        const AriesColumnType& valueType );

    public:
        virtual void GetAllParams( map< string, AEExprAggFunctionNode * > &result );
    private:
        size_t GetResultItemSize() const { return m_valueType.GetDataTypeSize(); }
        string GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const;
        pair< string, string > GenerateCudaFunction( const vector< AriesDynamicCodeParam >& params, const string& expr,
                const AriesColumnType& value ) const;
        string GenerateLoadDataCode( const AriesDynamicCodeParam& param, int index ) const;
        string GenerateSaveDataCode( const AriesColumnType& type, const string& paramName ) const;

    private:
        map< string, unique_ptr< AEExprAggFunctionNode > > m_aggFunctions;
    };

    class AEExprStarNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprStarNode > Create()
        {
            return unique_ptr< AEExprStarNode >( new AEExprStarNode() );
        }
        virtual ~AEExprStarNode()
        {
        }
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;

    protected:
        AEExprStarNode()
        {
        }
    };

    class AEExprIsNullNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprIsNullNode > Create( bool bHasNot )
        {
            return unique_ptr< AEExprIsNullNode >( new AEExprIsNullNode( bHasNot ) );
        }

        virtual ~AEExprIsNullNode()
        {
        }

        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;

    protected:
        AEExprIsNullNode( bool bHasNot )
                : m_bHasNot( bHasNot )
        {
        }

    private:
        bool m_bHasNot;
    };

    class AEExprTrueFalseNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprTrueFalseNode > Create( bool bTrue )
        {
            return unique_ptr< AEExprTrueFalseNode >( new AEExprTrueFalseNode( bTrue ) );
        }

        virtual ~AEExprTrueFalseNode()
        {
        }

        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;

    protected:
        AEExprTrueFalseNode( bool bTrue )
                : m_bTrue( bTrue )
        {
        }

    private:
        bool m_bTrue;
    };

    class AEExprIntervalNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprIntervalNode > Create( string unitType )
        {
            return unique_ptr< AEExprIntervalNode >( new AEExprIntervalNode( unitType ) );
        }

        virtual ~AEExprIntervalNode()
        {
        }

        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;

    protected:
        AEExprIntervalNode( string unitType )
                : m_unitType( unitType )
        {
        }
    private:
        string m_unitType;
    };

    class AEExprBufferNode: public AEExprNode
    {
    public:
        static unique_ptr< AEExprBufferNode > Create( const AriesDataBufferSPtr& buffer )
        {
            return unique_ptr< AEExprBufferNode >( new AEExprBufferNode( buffer ) );
        }
        virtual ~AEExprBufferNode()
        {
        }
        virtual AEExprNodeResult Process( const AriesTableBlockUPtr& refTable ) const;
        virtual string ToString() const;

    protected:
        AEExprBufferNode( const AriesDataBufferSPtr& buffer )
                : m_buffer( buffer )
        {
        }

    private:
        AriesDataBufferSPtr m_buffer;
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

