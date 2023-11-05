/*
 * AriesExprCalcNode.cpp
 *
 *  Created on: Sep 13, 2018
 *      Author: lichi
 */
#include <random>
#include <thread>
#include <cuda_runtime.h>
#include "utils/utils.h"
#include "AriesExprCalcNode.h"
#include "AriesUtil.h"
#include "../datatypes/AriesDatetimeTrans.h"
#include "AriesAssert.h"
#include "CudaAcc/AriesEngineException.h"
#include "CudaAcc/DynamicKernel.h"
#include "CudaAcc/AriesDecimalAlgorithm.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesColumnDataIterator.hxx"


using namespace aries_acc;
BEGIN_ARIES_ENGINE_NAMESPACE

    using Decimal = aries_acc::Decimal;
    /*
     * helper functions
     */
    void DumpExprNodeResult( const AEExprNodeResult& result )
    {
        switch( result.which() )
        {
            case 0:
                LOG( INFO )<< boost::get< bool >( result ) << endl;
                break;
                case 1:
                LOG( INFO ) << boost::get< int32_t >( result ) << endl;
                break;
                case 2:
                LOG( INFO ) << boost::get< int64_t >( result ) << endl;
                break;
                case 3:
                LOG( INFO ) << boost::get< float >( result ) << endl;
                break;
                case 4:
                LOG( INFO ) << boost::get< double >( result ) << endl;
                break;
                case 5:
                {
                    char dec[64];
                    Decimal d = boost::get< Decimal >( result );
                    LOG( INFO ) << d.GetDecimal( dec ) << endl;
                    break;
                }
                case 6:
                LOG( INFO ) << "'" + boost::get< string >( result ) + "'" << endl;
                break;
                case 7:
                //AriesDate
                break;
                case 8:
                //AriesDatetime
                break;
                case 14:
                boost::get< AriesInt32ArraySPtr >( result )->Dump();
                break;
                case 15:
                boost::get< AriesDataBufferSPtr >( result )->Dump();
                break;
                default:
                LOG( INFO ) << "wrong AEExprNodeResult value!!!" << endl;
                break;
            }
        }

    string GenSaveDataCode( const AriesColumnType &type, const string &paramName )
    {
        // FIXME the paramName's size should be limited by frontend.
        ARIES_ASSERT( paramName.size() < 512,
                "the paramName's size should be limited by frontend. paramName.size(): " + to_string( paramName.size() ) );
        char buf[1024];
        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            auto p = buf;
            sprintf( p, "        auto tmp = output + i * (size_t)%lu;\n", type.GetDataTypeSize() );
            p = buf + strlen( buf );
            if( type.HasNull )
            {
                sprintf( p, "        if ((*tmp++ = %s.flag) == 0) continue;\n", paramName.c_str() );
                p = buf + strlen( buf );
                sprintf( p, "        Decimal(%u, %u).cast(%s.value).ToCompactDecimal( tmp, %d);\n", type.DataType.Precision, type.DataType.Scale,
                        paramName.c_str(), type.DataType.Length );
            }
            else
            {
                sprintf( p, "        %s.ToCompactDecimal( tmp, %d);\n",
                        paramName.c_str(), type.DataType.Length );
            }
        }
        else
        {
            string dataType = GenerateParamType( type );
            sprintf( buf, "        *( ( %s* )( output + i * sizeof( %s ) ) ) = %s;\n", dataType.c_str(), dataType.c_str(), paramName.c_str() );
        }
        return buf;
    }

    /*
     * AEExprNode start
     */
    AEExprNode::AEExprNode( int nodeId, int exprIndex )
            : m_exprId( ( ( int64_t )nodeId << 32 ) | ( int64_t )exprIndex )
    {
    }

    AEExprNode::~AEExprNode()
    {
        m_children.clear();
    }

    void AEExprNode::AddChild( AEExprNodeUPtr child )
    {
        m_children.push_back( std::move( child ) );
    }

    bool AEExprNode::IsLiteral( const AEExprNodeResult& value ) const
    {
        return value.which() < 14;
    }

    bool AEExprNode::IsNullValue( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesNull);
    }

    bool AEExprNode::IsString( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(string);
    }

    bool AEExprNode::IsInt( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(int32_t);
    }

    bool AEExprNode::IsLong( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(int64_t);
    }

    bool AEExprNode::IsIntArray( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesInt32ArraySPtr);
    }

    bool AEExprNode::IsBoolArray( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesBoolArraySPtr);
    }

    bool AEExprNode::IsDataBuffer( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesDataBufferSPtr);
    }

    bool AEExprNode::IsInterval( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesInterval);
    }

    bool AEExprNode::IsAriesDate( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesDate);
    }

    bool AEExprNode::IsAriesDatetime( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesDatetime);
    }

    bool AEExprNode::IsAriesTime( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(AriesTime);
    }

    bool AEExprNode::IsDecimal( const AEExprNodeResult& value ) const
    {
        return value.type() == typeid(aries_acc::Decimal);
    }

    AriesDataBufferSPtr AEExprNode::ConvertLiteralToBuffer( const AEExprNodeResult& value, AriesColumnType columnType ) const
    {
        ARIES_ASSERT( IsLiteral( value ), "value type: " + string( value.type().name() ) );
        AriesDataBufferSPtr result;
        switch( value.which() )
        {
            case 0:
                // bool: true or false -> int32: 1 or 0
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::INT32, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< bool >( value ) ? 1 : 0 );
                break;
            case 1:
                // int32
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::INT32, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< int32_t >( value ) );
                break;
            case 2:
                // int64
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::INT64, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< int64_t >( value ) );
                break;
            case 3:
                // float
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::FLOAT, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< float >( value ) );
                break;
            case 4:
                // double
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::DOUBLE, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< double >( value ) );
                break;
            case 5:
                // decimal
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::DECIMAL, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< Decimal >( value ) );
                break;
            case 6:
            {
                // string
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::CHAR, columnType.DataType.Length }, false, false ) );
                result->AllocArray( 1 );
                //if the param is longer than column's, we just cut it down to the columnType's size. otherwise, fill 0 until the size matches.
                string param = boost::get< string >( value );
                param.resize( columnType.DataType.Length, 0 );
                InitValueBuffer( result->GetData(), columnType, param.c_str() );
                break;
            }
            case 7:
                // AriesDate
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::DATE, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< AriesDate >( value ) );
                break;
            case 8:
                // AriesDatetime
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::DATETIME, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< AriesDatetime >( value ) );
                break;
            case 11:
                // Ariestime
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::TIME, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< AriesTime >( value ) );
                break;
            case 12:
                // AriesTimestamp
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::TIMESTAMP, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< AriesTimestamp >( value ) );
                break;
            case 13:
                // AriesYear
                result = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::YEAR, 1 }, false, false ) );
                result->AllocArray( 1 );
                InitValueBuffer( result->GetData(), boost::get< AriesYear >( value ) );
                break;
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "converting expression node result type: " + string( value.type().name() ) );
                break;
        }
        return result;
    }

    /**
     * @brief 将数据分成若干部分分别处理
     */
    template< typename CreateResultFunc >
    static AEExprNodeResult RunKernelByBlock(
        const AriesTableBlockUPtr& refTable,
        const size_t partitionCount,
        const std::vector< AriesDynamicCodeParam >& params,
        const std::vector< CUmoduleSPtr >& cuModules,
        const std::string& functionName,
        const std::vector< AriesDataBufferSPtr >& constantValues,
        const std::vector< AriesDynamicCodeComparator >& comparators,
        const CreateResultFunc& create_result_func,
        const AriesColumnType& valueType,
        const size_t& returnTypeSize )
    {
        auto paramNum = params.size();
        const auto totalRowCount = size_t( refTable->GetRowCount() );
        auto result = create_result_func( valueType, totalRowCount );

        vector< int > columnIds;
        for( size_t i = 0; i < params.size(); i++ )
        {
            const auto& param = params[i];
            int columnId = param.ColumnIndex;
            columnIds.push_back( columnId );

            if ( refTable->IsColumnUnMaterilized( columnId ) )
            {
                refTable->MaterilizeColumns( { columnId } );
            }
        }

        auto refTable2 = refTable->MakeTableByColumns( columnIds, false );

        vector< AriesTableBlockUPtr > subTables;
        if( partitionCount == 1 )
            subTables.push_back( refTable2->Clone( false ) );
        else
        {
            vector<size_t> partitionRowCounts;
            partitionItems( totalRowCount, partitionCount, partitionRowCounts );
            size_t offset = 0;
            for ( auto partitionRowCount : partitionRowCounts )
            {
                if ( 0 == partitionRowCount )
                    break;
                subTables.push_back( refTable2->GetSubTable( offset, partitionRowCount, true ) );
                offset += partitionRowCount;
            }
        }

        auto constValues = make_shared< AriesManagedArray< int8_t* > >( constantValues.size() );
        for ( std::size_t i = 0; i < constantValues.size(); ++i )
            ( *constValues )[ i ] = constantValues[ i ]->GetData();
        constValues->PrefetchToGpu();   // 提前拷贝到 GPU 上

        size_t handledRowCount = 0;
        for ( auto& subTable : subTables )
        {
            size_t subTableRowCount = subTable->GetRowCount();

            AriesManagedArray< AriesColumnDataIterator > columns( paramNum );
            vector< AriesColumnDataIteratorHelper > columnHelpers( paramNum );
            for( size_t i = 0; i < paramNum; i++ )
            {
                const auto& param = params[i];
                auto& iter = columns[i];
                auto& iterHelper = columnHelpers[i];
                int columnId = param.ColumnIndex;
                GetAriesColumnDataIteratorInfo( iter, iterHelper, subTable, columnId, param.Type, param.UseDictIndex );
            }
            result->PrefetchToGpu( handledRowCount, subTableRowCount ); //数据拷贝到 GPU 上？
            columns.PrefetchToGpu();

            AriesDynamicKernelManager::GetInstance().CallKernel( cuModules,
                                                                 functionName.c_str(),
                                                                 columns.GetData(),
                                                                 subTableRowCount,
                                                                 ( const int8_t** )constValues->GetData(),
                                                                 comparators,
                                                                 ( int8_t* )( result->GetData() + returnTypeSize * handledRowCount ) );
            handledRowCount += subTableRowCount;
        }
        return result;
    }

    template< typename CreateResultFunc >
    static AEExprNodeResult RunKernelByBlock(
        const size_t totalRowCount,
        const size_t partitionCount,
        const map< string, AriesDataBufferSPtr >& params,
        const std::vector< AriesDynamicCodeParam >& dynCodeParams,
        const std::vector< CUmoduleSPtr >& cuModules,
        const std::string& functionName,
        const std::vector< AriesDataBufferSPtr >& constantValues,
        const std::vector< AriesDynamicCodeComparator >& comparators,
        const CreateResultFunc& create_result_func,
        const AriesColumnType& valueType,
        const size_t& returnTypeSize )
    {
        auto paramNum = dynCodeParams.size();
        auto result = create_result_func( valueType, totalRowCount );

        vector< AriesDataBufferSPtr > columnBuffers;
        AriesManagedArray< AriesColumnDataIterator > columns( paramNum );
        vector< AriesColumnDataIteratorHelper > columnHelpers( paramNum );
        for( size_t i = 0; i < paramNum; i++ )
        {
            const auto& param = dynCodeParams[i];
            auto& iter = columns[i];
            auto& iterHelper = columnHelpers[i];

            iter.m_nullData = AriesNullValueProvider::GetInstance().GetNullValue( param.Type );
            iter.m_perItemSize = param.Type.GetDataTypeSize();
            iter.m_dataBlockCount = 1;
            iter.m_indices = nullptr;

            auto dataBlockSizePrefixSum = std::make_shared< AriesInt64Array >( 1 );
            dataBlockSizePrefixSum->SetValue( 0, 0 );
            iterHelper.m_dataBlockSizePrefixSumArray = dataBlockSizePrefixSum;
            iter.m_dataBlockSizePrefixSum = dataBlockSizePrefixSum->GetData();

            AriesDataBufferSPtr columnBuffer;
            int columnId = param.ColumnIndex;
            if( columnId > 0 )
            {
                //数据来自于原始的table
                columnBuffer = params.at( std::to_string( columnId ) );
            }
            else
            {
                //数据来自于聚合函数
                columnBuffer = params.at( dynCodeParams[i].ParamName );
            }
            columnBuffers.emplace_back( columnBuffer );
        }

        auto constValues = make_shared< AriesManagedArray< int8_t* > >( constantValues.size() );
        for ( std::size_t i = 0; i < constantValues.size(); ++i )
            ( *constValues )[ i ] = constantValues[ i ]->GetData();
        constValues->PrefetchToGpu();

        vector<size_t> partitionRowCounts;
        partitionItems( totalRowCount, partitionCount, partitionRowCounts );

        size_t handledRowCount = 0;

        int8_t *pResult = result->GetData();
        for ( auto partItemCount : partitionRowCounts )
        {
            if ( 0 == partItemCount )
                break;
            result->PrefetchToGpu( handledRowCount, partItemCount );

            AriesDataBufferSPtr columnBuffer;
            for( size_t i = 0; i < paramNum; i++ )
            {
                auto& iter = columns[i];
                auto& iterHelper = columnHelpers[i];

                columnBuffer = columnBuffers[ i ];
                columnBuffer->PrefetchToGpu( handledRowCount, partItemCount );

                auto dataBlocks = make_shared< AriesManagedArray< int8_t* > >( 1 );
                dataBlocks->PrefetchToGpu();
                ( *dataBlocks )[0] = columnBuffer->GetData( handledRowCount );

                iterHelper.m_dataBlockPtrs = dataBlocks;
                iter.m_data = dataBlocks->GetData();
            }
            columns.PrefetchToGpu();

            AriesDynamicKernelManager::GetInstance().CallKernel( cuModules,
                                                                 functionName.c_str(),
                                                                 columns.GetData(),
                                                                 partItemCount,
                                                                 ( const int8_t** )constValues->GetData(),
                                                                 comparators,
                                                                 pResult );
            pResult += returnTypeSize * partItemCount; 

            handledRowCount += partItemCount;
        }

        return result;
    }

    const AriesDataBufferSPtr create_result_func( AriesColumnType valueType, const size_t& rowCount )
    {
        return std::make_shared< AriesDataBuffer >( valueType, rowCount );
    }

    AEExprDynKernelNode::AEExprDynKernelNode( int nodeId, int exprIndex,
                                              vector< AriesDynamicCodeParam > && params,
                                              vector< AriesDataBufferSPtr > && constValues,
                                              vector< AriesDynamicCodeComparator > && comparators,
                                              const string& expr,
                                              const AriesColumnType& valueType )
        : AEExprNode( nodeId, exprIndex ),
          m_ariesParams( std::move( params ) ),
          m_constValues( std::move( constValues ) ),
          m_ariesComparators( std::move( comparators ) ),
          m_expr( expr ),
          m_valueType( valueType )
    {

    }
    size_t AEExprDynKernelNode::GetPartitionCount( const AriesTableBlockUPtr& refTable ) const
    {
        size_t partitionCount = 1;
        size_t rowCount = refTable->GetRowCount();
        size_t totalMemNeed = GetResultItemSize() * rowCount;
        for( size_t i = 0; i < m_ariesParams.size(); i++ )
        {
            const auto& param = m_ariesParams[i];
            totalMemNeed += param.Type.GetDataTypeSize() * rowCount;
        }

        size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        double currentRatio = totalMemNeed / available;
        if( currentRatio > MAX_RATIO )
            partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

        return partitionCount;
    }
    size_t AEExprDynKernelNode::GetPartitionCount( const map< string, AriesDataBufferSPtr >& params ) const
    {
        size_t partitionCount = 1;
        size_t rowCount = params.cbegin()->second->GetItemCount();
        size_t totalMemNeed = GetResultItemSize() * rowCount;
        for ( const auto& pair : params )
        {
            totalMemNeed += pair.second->GetTotalBytes();
        }
        size_t available = AriesDeviceProperty::GetInstance().GetMemoryCapacity();

        const double MAX_RATIO = 0.8;//不能超过空闲区域的80%

        double currentRatio = totalMemNeed / available;
        if( currentRatio > MAX_RATIO )
            partitionCount = size_t( currentRatio / MAX_RATIO ) + 1;

        return partitionCount;
    }

    AEExprNodeResult AEExprDynKernelNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( !m_expr.empty(), "m_expr is empty(), no dynamic kernel exits!" );
        assert( m_exprId != -1 );

        size_t partitionCount = GetPartitionCount( refTable );

        return RunKernelByBlock( refTable,
                                 partitionCount,
                                 m_ariesParams,
                                 m_cuModules,
                                 m_cudaFunctionName,
                                 m_constValues,
                                 m_ariesComparators,
                                 create_result_func,
                                 m_valueType,
                                 m_valueType.GetDataTypeSize() );
    }

    AEExprNodeResult AEExprDynKernelNode::RunKernelFunction( const map< string, AriesDataBufferSPtr >& params ) const
    {
        assert( m_exprId != -1 );

        const auto total_row_count = params.cbegin()->second->GetItemCount();
        size_t partitionCount = GetPartitionCount( params );
        return RunKernelByBlock( total_row_count,
                                 partitionCount,
                                 params,
                                 m_ariesParams,
                                 m_cuModules,
                                 m_cudaFunctionName,
                                 m_constValues,
                                 m_ariesComparators,
                                 create_result_func,
                                 m_valueType,
                                 m_valueType.GetDataTypeSize() );
    }


    /*
     * AEExprAndOrNode start
     */
    unique_ptr< AEExprAndOrNode > AEExprAndOrNode::Create( int nodeId,
                                                           int exprIndex,
                                                           AriesLogicOpType opType,
                                                           vector< AriesDynamicCodeParam > && params,
                                                           vector< AriesDataBufferSPtr > && constValues,
                                                           vector< AriesDynamicCodeComparator > && comparators,
                                                           const string& expr,
                                                           const AriesColumnType& valueType )
    {
        return unique_ptr< AEExprAndOrNode >( new AEExprAndOrNode( nodeId,
                                                                   exprIndex,
                                                                   opType,
                                                                   std::move( params ),
                                                                   std::move( constValues ),
                                                                   std::move( comparators ),
                                                                   expr,
                                                                   valueType ) );
    }

    AEExprAndOrNode::AEExprAndOrNode( int nodeId,
                                      int exprIndex,
                                      AriesLogicOpType opType,
                                      vector< AriesDynamicCodeParam > && params,
                                      vector< AriesDataBufferSPtr > && constValues,
                                      vector< AriesDynamicCodeComparator > && comparators,
                                      const string& expr,
                                      const AriesColumnType& valueType )
            : AEExprDynKernelNode( nodeId, exprIndex, std::move( params ), std::move( constValues ), std::move( comparators ), expr, AriesColumnType() ),
              m_opType( opType )
    {
        pair< string, string > cudaFunction = GenerateCudaFunctionEx( m_ariesParams, m_expr, valueType );
        m_cudaFunctionName = cudaFunction.first;
        m_cudaFunction = cudaFunction.second;
    }

    AEExprAndOrNode::~AEExprAndOrNode()
    {

    }

    string AEExprAndOrNode::GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const
    {
        if( !comparator.TempName.empty() )
        {
            char buf[1024];
            AriesColumnType type = comparator.Type;
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s;\n", typeName.c_str(), comparator.TempName.c_str() );
            return buf;
        }
        else
        {
            return string();
        }
    }

    pair< string, string > AEExprAndOrNode::GenerateLoadDataCodeEx( const AriesDynamicCodeParam& param, int index, int len ) const
    {
        char buf[1024];
        AriesColumnType type = param.Type;

        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            if( type.HasNull )
            {
                sprintf( buf, "( nullable_type< Decimal >( *(int8_t*)( input[%d][i] ), Decimal( (CompactDecimal*)( input[%d][i] + 1 ), %u, %u ) ) )",
                        index, index, type.DataType.Precision, type.DataType.Scale );
            }
            else
            {
                sprintf( buf, "        AriesDecimal<%u> ( (CompactDecimal*)( input[%d][i] ), %u, %u )", len, index, type.DataType.Precision, type.DataType.Scale );
            }
        }
        else
        {
            sprintf( buf, "(*( ( %s* )( input[%d][i] ) ) )", GenerateParamType( type ).c_str(), index );
        }

        return
        {   param.ParamName, buf};
    }

    pair< string, string > AEExprAndOrNode::GenerateCudaFunctionEx( const vector< AriesDynamicCodeParam >& params, const string& expr, const AriesColumnType& resultDataType ) const
    {
        ARIES_ASSERT( !expr.empty() && !params.empty(),
                "expr empty: " + to_string( expr.empty() ) + ", params empty: " + to_string( params.empty() ) );
        pair< string, string > result;
        string name = "and_or_expr_" + std::to_string( m_exprId );
        string code = "extern \"C\"  __global__ void ";
        code += name;
        code += "( const AriesColumnDataIterator *input, int tupleNum, const int8_t** constValues, const CallableComparator** comparators, char *output )\n"
                "{\n"
                "    int stride = blockDim.x * gridDim.x;\n"
                "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    for( int i = tid; i < tupleNum; i += stride )\n"
                "    {\n";
        for( const auto& comp : m_ariesComparators )
        {
            code += GenerateTempVarCode( comp );
        }
        string resultValName = "Cuda_Dyn_resultValueName";
        int index = 0;
        vector< pair< string, string > > loadParamCodes;
        for( const auto& param : params )
        {
            loadParamCodes.push_back( GenerateLoadDataCodeEx( param, index++, resultDataType.DataType.AdaptiveLen) );
        }

        code += "        AriesBool " + resultValName + " = " + expr + ";\n";
        code += GenerateSaveDataCode( resultValName );
        code += "    }\n"
                "}\n";
        for( const auto& param : loadParamCodes )
        {
            ReplaceString( code, param.first, param.second );
        }
        return
        {   name, code};
    }

    string AEExprAndOrNode::GenerateSaveDataCode( const string& paramName ) const
    {
        //FIXME the paramName's size should be limited by frontend.
        ARIES_ASSERT( paramName.size() < 512,
                "the paramName's size should be limited by frontend. paramName.size(): " + to_string( paramName.size() ) );
        char buf[1024];
        sprintf( buf, "        *( ( AriesBool* )( output + i * sizeof( AriesBool ) ) ) = %s;\n", paramName.c_str() );
        return buf;
    }

    AEExprNodeResult AEExprAndOrNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( !m_expr.empty(), "m_expr is empty(), no dynamic kernel exits!" );
        assert( m_exprId != -1 );
        return RunKernelFunction( refTable );
    }

    AEExprNodeResult AEExprAndOrNode::RunKernelFunction( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( !m_expr.empty(), "m_expr is empty" );
        assert( m_exprId != -1 );

        size_t partitionCount = GetPartitionCount( refTable );
        const auto andor_create_result_func = []( const AriesColumnType& valueType, const size_t& row_count ) -> AriesBoolArraySPtr
        {
            return std::make_shared< AriesArray< AriesBool > >( row_count, false );
        };

        return RunKernelByBlock( refTable,
                                 partitionCount,
                                 m_ariesParams,
                                 m_cuModules,
                                 m_cudaFunctionName,
                                 m_constValues,
                                 m_ariesComparators,
                                 andor_create_result_func,
                                 AriesColumnType(),
                                 sizeof( AriesBool ) );
    }

    void AEExprAndOrNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        m_cuModules = modules;
    }

    string AEExprAndOrNode::GetCudaKernelCode() const
    {
        return m_cudaFunction;
    }

    string AEExprAndOrNode::ToString() const
    {
        return m_expr;
    }

    /*
     * AEExprComparisonNode start
     */
    unique_ptr< AEExprComparisonNode > AEExprComparisonNode::Create( AriesComparisonOpType opType )
    {
        return unique_ptr< AEExprComparisonNode >( new AEExprComparisonNode( opType ) );
    }

    AEExprComparisonNode::AEExprComparisonNode( AriesComparisonOpType opType )
            : m_opType( opType )
    {

    }

    AEExprComparisonNode::~AEExprComparisonNode()
    {

    }

    AEExprNodeResult AEExprComparisonNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( m_children.size() == 2, "m_children.size(): " + to_string( m_children.size() ) );
        AriesBoolArraySPtr result;
        size_t tupleNum = refTable->GetRowCount();
        AriesComparisonOpType opType = m_opType;

        AEExprColumnIdNode* leftColumnNode = dynamic_cast< AEExprColumnIdNode * >( m_children[0].get() );
        AEExprLiteralNode* rightLiteralNode = dynamic_cast< AEExprLiteralNode * >( m_children[1].get() );
        if( leftColumnNode && rightLiteralNode && ( AriesComparisonOpType::EQ == opType || AriesComparisonOpType::NE == opType ) )
        {
            auto columnId = leftColumnNode->GetId();
            auto colEncodeType = refTable->GetColumnEncodeType( columnId );
            AEExprNodeResult rightRes = m_children[1]->Process( refTable );

            // 对字典做过滤
            if( EncodeType::DICT == colEncodeType && rightRes.which() == 1 )
            {
                LOG( INFO )<< "compare dict encoded column";

                /*
                AriesColumnType indexDataType;
                if ( refTable->IsColumnUnMaterilized( columnId ) )
                {
                    auto columnReference = refTable->GetUnMaterilizedColumn( columnId );
                    auto refferedColumn = columnReference->GetReferredColumn();
                    auto dictColumn = std::dynamic_pointer_cast< AriesDictEncodedColumn >( refferedColumn );
                    indexDataType = dictColumn->GetIndices()->GetColumnType();
                }
                else
                {
                    auto dictColumn = refTable->GetDictEncodedColumn( columnId );
                    indexDataType = dictColumn->GetIndices()->GetColumnType();
                }

                AriesManagedArray< AriesColumnDataIterator > columnIters( 1 );
                AriesColumnDataIteratorHelper columnIterHelper;

                auto& iter = columnIters[ 0 ];
                GetAriesColumnDataIteratorInfo( iter, columnIterHelper, refTable, columnId, indexDataType, true );

                AriesDataBufferSPtr rightBuf = ConvertLiteralToBuffer( rightRes, indexDataType );

                return CompareColumn( columnIters.GetData(), opType, rightBuf );
                */

                AriesDataBufferSPtr indiceBuff = refTable->GetDictEncodedColumnIndiceBuffer( columnId );

                AriesDataBufferSPtr rightBuf = ConvertLiteralToBuffer( rightRes, indiceBuff->GetDataType() );

                return CompareColumn( indiceBuff, opType, rightBuf );
            }
        }

        AEExprNodeResult leftRes = m_children[0]->Process( refTable );
        AEExprNodeResult rightRes = m_children[1]->Process( refTable );
        //FIXME 如果表达式两边均为常量，希望前端能进行化简
        ARIES_ASSERT( !IsLiteral( leftRes ) || !IsLiteral( rightRes ), "AEExprComparisonNode, 如果表达式两边均为常量，希望前端能进行化简" );

        if ( typeid(AriesBoolArraySPtr) == leftRes.type() )
        {
            leftRes = aries_acc::ConvertToDataBuffer( boost::get< AriesBoolArraySPtr >( leftRes ) );
        }
        if ( typeid(AriesBoolArraySPtr) == rightRes.type() )
        {
            rightRes = aries_acc::ConvertToDataBuffer( boost::get< AriesBoolArraySPtr >( rightRes ) );
        }
        ARIES_ASSERT( IsDataBuffer( leftRes ) || IsLiteral( leftRes ),
                "leftRes type: " + string( leftRes.type().name() ) );
        ARIES_ASSERT( IsDataBuffer( rightRes ) || IsLiteral( rightRes ),
                "rightRes type: " + string( rightRes.type().name() ) );

        if( IsLiteral( leftRes ) )
        {
            // 左右交换，让操作符左边为Databuf，右边为常量
            std::swap( leftRes, rightRes );
            switch( opType )
            {
                case AriesComparisonOpType::GT:
                    opType = AriesComparisonOpType::LT;
                    break;
                case AriesComparisonOpType::GE:
                    opType = AriesComparisonOpType::LE;
                    break;
                case AriesComparisonOpType::LT:
                    opType = AriesComparisonOpType::GT;
                    break;
                case AriesComparisonOpType::LE:
                    opType = AriesComparisonOpType::GE;
                    break;
                default:
                    break;
            }
        }

        // 操作符左边一定为databuf
        const AriesDataBufferSPtr leftBuf = boost::get< AriesDataBufferSPtr >( leftRes );
        leftBuf->PrefetchToGpu();
        tupleNum = leftBuf->GetItemCount();
        ARIES_ASSERT( tupleNum > 0, "leftBuf tupleNum : " + to_string( tupleNum ) );
        if( IsLiteral( rightRes ) )
        {
            if( !IsNullValue( rightRes ) )
            {
                AriesColumnType leftColumnType = leftBuf->GetDataType();
                //操作符右边为常量
                AriesDataBufferSPtr rightBuf = ConvertLiteralToBuffer( rightRes, leftColumnType );
                if( rightRes.type() == typeid(string) )
                {
                    //对字符串比较要特殊处理
                    bool bHandled = false;
                    if( boost::get< string >( rightRes ).size() > leftColumnType.GetDataTypeSize() )
                    {
                        switch( opType )
                        {
                            case AriesComparisonOpType::EQ:
                            {
                                // 对于字符串相等比较，如果常量的长度大于目标类型长度，则不可能有相等值，直接返回 false
                                result = make_shared< AriesBoolArray >( tupleNum );
                                FillWithValue( result, AriesBool::ValueType::False );
                                bHandled = true;
                                break;
                            }
                            case AriesComparisonOpType::NE:
                            {
                                if( !leftColumnType.HasNull )
                                {
                                    // 对于字符串不等比较，如果常量的长度大于目标类型长度，则不等于全部成立，直接返回 true
                                    result = make_shared< AriesBoolArray >( tupleNum );
                                    FillWithValue( result, AriesBool::ValueType::True );
                                    bHandled = true;
                                }
                                break;
                            }
                            case AriesComparisonOpType::GE:
                            {
                                //对于>=，由于对常量进行了截断，=条件不可能满足。所以转化为GT
                                opType = AriesComparisonOpType::GT;
                                break;
                            }
                            case AriesComparisonOpType::LT:
                            {
                                //对于<，由于对常量进行了截断，=条件也满足最终结果。所以转化为LE
                                opType = AriesComparisonOpType::LE;
                                break;
                            }
                            default:
                                break;
                        }
                    }
                    if( !bHandled )
                    {
                        result = CompareColumn( leftBuf, opType, rightBuf );
                    }
                }
                else
                {
                    result = CompareColumn( leftBuf, opType, rightBuf );
                }
            }
            else
            {
                // compare with null always return false;
                result = make_shared< AriesBoolArray >( tupleNum );
                FillWithValue( result, AriesBool::ValueType::False );
            }
        }
        else
        {
            // rightRes is DataBuffer
            AriesDataBufferSPtr rightBuf = std::move( boost::get< AriesDataBufferSPtr >( rightRes ) );
            rightBuf->PrefetchToGpu();
            ARIES_ASSERT( rightBuf->GetItemCount() == tupleNum,
                    "rightBuf->GetItemCount(): " + to_string( rightBuf->GetItemCount() ) + ", tupleNum: " + to_string( tupleNum ) );
            result = CompareTowColumns( leftBuf, opType, rightBuf );
        }

        return AEExprNodeResult
        { std::move( result ) };
    }

    void AEExprComparisonNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprComparisonNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprComparisonNode::ToString() const
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
     * AEExprLiteralNode start
     */
    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( int32_t value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( int64_t value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( float value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( double value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( Decimal value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( const string& value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( AriesDate value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( AriesDatetime value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( AriesTime value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( AriesTimestamp value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( AriesYear value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    unique_ptr< AEExprLiteralNode > AEExprLiteralNode::Create( AriesNull value )
    {
        return unique_ptr< AEExprLiteralNode >( new AEExprLiteralNode( value ) );
    }

    AEExprLiteralNode::AEExprLiteralNode( int32_t value )
            : m_value( value )
    {
    }

    AEExprLiteralNode::AEExprLiteralNode( int64_t value )
            : m_value( value )
    {
    }

    AEExprLiteralNode::AEExprLiteralNode( float value )
            : m_value( value )
    {
    }

    AEExprLiteralNode::AEExprLiteralNode( double value )
            : m_value( value )
    {
    }

    AEExprLiteralNode::AEExprLiteralNode( Decimal value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( const string& value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( AriesDate value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( AriesDatetime value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( AriesTime value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( AriesTimestamp value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( AriesYear value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::AEExprLiteralNode( AriesNull value )
            : m_value( value )
    {

    }

    AEExprLiteralNode::~AEExprLiteralNode()
    {

    }

    AEExprNodeResult AEExprLiteralNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        return m_value;
    }

    string AEExprLiteralNode::ToString() const
    {
        //int32_t, int64_t, float, double, aries_acc::Decimal, string, AriesDate, AriesDatetime
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
            case 8:
            {
                return "NULL";
            }
            default:
                break;
        }

        return "wrong literal value";
    }

    /*
     * AEExprCalcNode start
     */
    unique_ptr< AEExprCalcNode >
    AEExprCalcNode::Create( int nodeId,
                            int exprIndex,
                            map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                            vector< AriesDynamicCodeParam > && params,
                            vector< AriesDataBufferSPtr > && constValues,
                            vector< AriesDynamicCodeComparator > && comparators,
                            const string& expr,
                            const AriesColumnType& valueType )
    {
        return unique_ptr< AEExprCalcNode >(
                new AEExprCalcNode( nodeId,
                                    exprIndex,
                                    std::move( aggFunctions ),
                                    std::move( params ),
                                    std::move( constValues ),
                                    std::move( comparators ),
                                    expr,
                                    valueType ) );
    }

    unique_ptr< AEExprCalcNode >
    AEExprCalcNode::CreateForXmp( int nodeId,
                            int exprIndex,
                            map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                            vector< AriesDynamicCodeParam > && params,
                            vector< AriesDataBufferSPtr > && constValues,
                            vector< AriesDynamicCodeComparator > && comparators,
                            const string& expr,
                            const AriesColumnType& valueType,
                            int ansLEN,
                            int ansTPI )
    {
        return unique_ptr< AEExprCalcNode >(
                new AEExprCalcNode( nodeId,
                                    exprIndex,
                                    std::move( aggFunctions ),
                                    std::move( params ),
                                    std::move( constValues ),
                                    std::move( comparators ),
                                    expr,
                                    valueType,
                                    ansLEN,
                                    ansTPI,
                                    true) );
    }

    AEExprCalcNode::AEExprCalcNode( int nodeId,
                                    int exprIndex,
                                    map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                                    vector< AriesDynamicCodeParam > && params,
                                    vector< AriesDataBufferSPtr > && constValues,
                                    vector< AriesDynamicCodeComparator > && comparators,
                                    const string& expr,
                                    const AriesColumnType& valueType,
                                    int ansLEN,
                                    int ansTPI,
                                    bool isXmp )
        : AEExprDynKernelNode( nodeId, exprIndex, std::move( params ), std::move( constValues ), std::move( comparators ), expr, valueType ),
          m_aggFunctions( std::move( aggFunctions ) )
    {
        pair< string, string > cudaFunction;
        // 普通表达式生成常规单线程单实例的代码 decimal 表达式则生成多线程协作的代码
        if(isXmp == true)
            cudaFunction = GenerateCudaFunctionXmp( m_ariesParams, m_expr, m_valueType, ansLEN, ansTPI );
        else
            cudaFunction = GenerateCudaFunction( m_ariesParams, m_expr, m_valueType );
        m_cudaFunctionName = cudaFunction.first;
        m_cudaFunction = cudaFunction.second;
    }

    AEExprCalcNode::~AEExprCalcNode()
    {
    }

    void AEExprCalcNode::GetAllParams( map< string, AEExprAggFunctionNode* >& result )
    {
        //获取agg参数
        for( const auto& agg : m_aggFunctions )
        {
            result[agg.first] = agg.second.get();
        }

        //获取普通column参数
        for( const auto& param : m_ariesParams )
        {
            if( param.ColumnIndex > 0 )
            {
                result[std::to_string( param.ColumnIndex )] = nullptr;
            }
        }
    }

    string AEExprCalcNode::GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const
    {
        if( !comparator.TempName.empty() )
        {
            char buf[1024];
            AriesColumnType type = comparator.Type;
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s;\n", typeName.c_str(), comparator.TempName.c_str() );
            return buf;
        }
        else
        {
            return string();
        }
    }

    // 生成动态代码
    pair< string, string > AEExprCalcNode::GenerateCudaFunction( const vector< AriesDynamicCodeParam >& params, const string& expr,
            const AriesColumnType& resultDataType ) const
    {
        ARIES_ASSERT( !expr.empty() && !params.empty(),
                "expr.empty(): " + to_string( expr.empty() ) + ", params.empty():" + to_string( params.empty() ) );
        pair< string, string > result;
        string name = "calc_expr_" + std::to_string( m_exprId );
        string code = "extern \"C\"  __global__ void ";
        code += name;

        code += "( const AriesColumnDataIterator *input, int tupleNum, const int8_t** constValues, const CallableComparator** comparators, char *output )\n"
                "{\n"
                "    int stride = blockDim.x * gridDim.x;\n"
                "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    for( int i = tid; i < tupleNum; i += stride )\n"
                "    {\n";
        for( const auto& comp : m_ariesComparators )
        {
            code += GenerateTempVarCode( comp );
        }
        const char* resultValName = "Cuda_Dyn_resultValueName";
        int index = 0;
        for( const auto& param : params )
        {
            code += GenerateLoadDataCode( param, index++, resultDataType.DataType.AdaptiveLen );
        }
        code += "        " + GenerateParamType( resultDataType ) + " " + resultValName + " = " + expr + ";\n";
        code += GenerateSaveDataCode( resultDataType, resultValName );
        code += "    }\n"
                "}\n";
        return
        {   name, code};
    }

    // 生成动态代码
    pair< string, string > AEExprCalcNode::GenerateCudaFunctionXmp( const vector< AriesDynamicCodeParam >& params, const string& expr,
            const AriesColumnType& resultDataType, int ansLEN, int ansTPI ) const
    {
        ARIES_ASSERT( !expr.empty() && !params.empty(),
                "expr.empty(): " + to_string( expr.empty() ) + ", params.empty():" + to_string( params.empty() ) );
        pair< string, string > result;

        string str_var_tpi = "TPI";
        string str_var_limbs = "";

        if(ansTPI == 4)
            str_var_tpi = "TPI_ONE";
        else if(ansTPI == 8)
            str_var_tpi = "TPI_TWO";
        else if(ansTPI == 16)
            str_var_tpi = "TPI_THR";
        else if(ansTPI == 32)
            str_var_tpi = "TPI_FOR";

        int ansLIMBS = ansLEN/ansTPI;
        if(ansLIMBS == 1)
            str_var_limbs = "LIMBS_ONE";
        else if(ansLIMBS == 2)
            str_var_limbs = "LIMBS_TWO";
        else if(ansLIMBS == 4)
            str_var_limbs = "LIMBS_THR";
        else if(ansLIMBS == 8)
            str_var_limbs = "LIMBS_FOR";

        string name = str_var_tpi + "xmp_calc_expr_" + std::to_string( m_exprId );
        string code = "extern \"C\"  __global__ void ";
        code += name;

        // xmp
        code += "( const AriesColumnDataIterator *input, int tupleNum, const int8_t** constValues, const CallableComparator** comparators, char *output )\n"
                "{\n"
                "       int32_t group_thread=threadIdx.x & "+ str_var_tpi +"-1;\n"
                "       int32_t index = ((long long)blockIdx.x*blockDim.x + threadIdx.x)/ "+ str_var_tpi +";\n"
                "       if(index>=tupleNum)\n"
                "           return;\n";
        
        // 声明参数的变量
        for(size_t i=0; i<params.size(); i++){
            code += "       uint32_t " + params[i].ParamName + "["+ str_var_limbs +"] = {0};\n";
            code += "       uint8_t " + params[i].ParamName + "_sign = 0;\n";
        }

        code += expr;

        uint32_t per_size = ansLIMBS*sizeof(uint32_t);
        string str_per_size = to_string(per_size);
        uint32_t ans_len = resultDataType.GetDataTypeSize();
        uint32_t ans_div = ans_len / per_size;
        uint32_t ans_mod = ans_len % per_size;
        string str_ans_div = to_string(ans_div);
        string str_ans_len = to_string(ans_len);

        // 将结果放入到 compact 中
        code += "       auto ans_tmp = output +  (long long)(index) * "+ str_ans_len +" + group_thread * "+ str_per_size +";\n";
        if(ans_mod == 0){
            code += "       if(group_thread < "+ str_ans_div +"){\n"
                    "              aries_memcpy(ans_tmp, var_0, "+ str_per_size +");\n"
                    "       }\n";
        }
        else{
            string str_ans_mod = to_string(ans_mod);
            code += "       if(group_thread < "+ str_ans_div +"){\n"
                    "              aries_memcpy(ans_tmp, var_0, "+ str_per_size +");\n"
                    "       }\n"
                    "       if(group_thread == "+ str_ans_div +"){\n"
                    "              aries_memcpy(ans_tmp, var_0, "+ str_ans_mod +");\n"
                    "       }\n";
        }
        code += "       if(group_thread==0){\n"
                "              char *buf = output + (long long)(index) * "+ str_ans_len +";\n"
                "              SET_SIGN_BIT( buf["+ str_ans_len +"-1], var_0_sign);\n"
                "       }\n"
                "}\n";
        return
        {   name, code};
    }

    string AEExprCalcNode::GenerateLoadDataCode( const AriesDynamicCodeParam& param, int index, int len ) const
    {
        char buf[1024];
        AriesColumnType type = param.Type;
        type.DataType.AdaptiveLen = len;
        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            if( type.HasNull )
            {
                sprintf( buf,
                        "        nullable_type< Decimal > %s( *(int8_t*)( input[%d][i] ), Decimal( (CompactDecimal*)( input[%d][i] + 1 ), %u, %u ) );\n",
                        param.ParamName.c_str(), index, index, type.DataType.Precision, type.DataType.Scale );
            }
            else
            {
                sprintf( buf, "        AriesDecimal<%u> %s( (CompactDecimal*)( input[%d][i] ), %u, %u );\n", len, param.ParamName.c_str(), index,
                        type.DataType.Precision, type.DataType.Scale );
            }
        }
        else
        {
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s = *( ( %s* )( input[%d][i] ) );\n", typeName.c_str(), param.ParamName.c_str(), typeName.c_str(), index );
        }

        return buf;
    }

    string AEExprCalcNode::GenerateSaveDataCode( const AriesColumnType& type, const string& paramName ) const
    {
        return GenSaveDataCode( type, paramName );
    }

    void AEExprCalcNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        m_cuModules = modules;
        for( const auto& agg : m_aggFunctions )
        {
            agg.second->SetCuModule( modules );
        }
    }

    string AEExprCalcNode::GetCudaKernelCode() const
    {
        string code = m_cudaFunction;
        for( const auto& agg : m_aggFunctions )
        {
            code += agg.second->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprCalcNode::ToString() const
    {
        return m_expr;
    }
    /*
     * AEExprColumnIdNode
     */
    unique_ptr< AEExprColumnIdNode > AEExprColumnIdNode::Create( int columnId )
    {
        ARIES_ASSERT( columnId != 0, "columnId: 0" );
        return unique_ptr< AEExprColumnIdNode >( new AEExprColumnIdNode( abs( columnId ) ) );
    }

    AEExprColumnIdNode::AEExprColumnIdNode( int columnId )
            : m_columnId( columnId )
    {
    }

    AEExprColumnIdNode::~AEExprColumnIdNode()
    {
    }

    int AEExprColumnIdNode::GetId() const
    {
        return m_columnId;
    }

    AEExprNodeResult AEExprColumnIdNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        return refTable->GetColumnBuffer( m_columnId );
    }

    void AEExprColumnIdNode::GetAllParams( map< string, AEExprAggFunctionNode * > &result )
    {
        result.insert(
        { to_string( m_columnId ), nullptr } );
    }

    string AEExprColumnIdNode::ToString() const
    {
        return "columnid_" + std::to_string( m_columnId ) + "_";
    }

    /*
     * AEExprBetweenNode
     */
    unique_ptr< AEExprBetweenNode > AEExprBetweenNode::Create()
    {
        return unique_ptr< AEExprBetweenNode >( new AEExprBetweenNode() );
    }

    AEExprBetweenNode::AEExprBetweenNode()
    {
    }

    AEExprBetweenNode::~AEExprBetweenNode()
    {
    }

    AEExprNodeResult AEExprBetweenNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( m_children.size() == 3, "m_children.size(): " + to_string( m_children.size() ) );
        AEExprNodeResult data = m_children[0]->Process( refTable );
        AEExprNodeResult minData = m_children[1]->Process( refTable );
        AEExprNodeResult maxData = m_children[2]->Process( refTable );
        ARIES_ASSERT( IsDataBuffer( data ), "data type: " + string( data.type().name() ) );

        //FIXME, the code is duplicate as comparisonNode and AndOrNode. ugly for now, should improve for big kernal
        AriesDataBufferSPtr dataBuf = boost::get< AriesDataBufferSPtr >( data );
        dataBuf->PrefetchToGpu();
        size_t tupleNum = dataBuf->GetItemCount();
        ARIES_ASSERT( tupleNum > 0, "tupleNum: " + to_string( tupleNum ) );
        AriesBoolArraySPtr minResult;
        if( IsLiteral( minData ) )
        {
            // minData is literal
            AriesDataBufferSPtr minValue = ConvertLiteralToBuffer( minData, dataBuf->GetDataType() );
            minResult = CompareColumn( dataBuf, AriesComparisonOpType::GE, minValue );
        }
        else
        {
            // minData is DataBuffer
            assert( 0 ); //FIXME
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "min data of BETWEEN must be a Literal" );
        }

        AriesBoolArraySPtr maxResult;
        if( IsLiteral( maxData ) )
        {
            // maxData is literal
            AriesDataBufferSPtr maxValue = ConvertLiteralToBuffer( maxData, dataBuf->GetDataType() );
            maxResult = CompareColumn( dataBuf, AriesComparisonOpType::LE, maxValue );
        }
        else
        {
            // maxData is DataBuffer
            assert( 0 ); //FIXME
            ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "max data of BETWEEN must be a Literal" );
        }

        MergeAssociates( minResult, maxResult, AriesLogicOpType::AND );
        return AEExprNodeResult
        { std::move( minResult ) };
    }

    void AEExprBetweenNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprBetweenNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprBetweenNode::ToString() const
    {
        ARIES_ASSERT( m_children.size() == 3, "m_children.size(): " + to_string( m_children.size() ) );

        string ret = m_children[0]->ToString();
        ret += " between ";
        ret += m_children[1]->ToString();
        ret += " and ";
        ret += m_children[2]->ToString();
        return ret;
    }

    /*
     * AEExprAggFunctionNode start
     */
    unique_ptr< AEExprAggFunctionNode > AEExprAggFunctionNode::Create( AriesAggFunctionType functionType, bool bDistinct )
    {
        return unique_ptr< AEExprAggFunctionNode >( new AEExprAggFunctionNode( functionType, bDistinct ) );
    }

    AEExprAggFunctionNode::AEExprAggFunctionNode( AriesAggFunctionType functionType, bool bDistinct )
            : m_functionType( functionType ), m_bDistinct( bDistinct )
    {
    }

    AEExprAggFunctionNode::~AEExprAggFunctionNode()
    {
    }

    AriesAggFunctionType AEExprAggFunctionNode::GetFunctionType() const
    {
        return m_functionType;
    }

    bool AEExprAggFunctionNode::IsDistinct() const
    {
        return m_bDistinct;
    }

    AEExprNodeResult AEExprAggFunctionNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        // only for compiler, we shouldn't use call this function!
        ARIES_ASSERT( 0, "only for compiler, we shouldn't use call this function" );
        return false;
    }

    pair< AEExprNodeResult, AEExprNodeResult > AEExprAggFunctionNode::RunKernelFunction( const AriesTableBlockUPtr& refTable,
            const AriesInt32ArraySPtr& associated, const AriesInt32ArraySPtr& groups, const AriesInt32ArraySPtr& groupFlags,
            const AriesDataBufferSPtr& itemCountInGroups ) const
    {
        ARIES_ASSERT( refTable && associated && groups && groups,
                "refTable is nullptr: " + to_string( refTable == nullptr ) + "associated is nullptr: " + to_string( associated == nullptr )
                        + "groups is nullptr: " + to_string( groups == nullptr ) + "groups is nullptr: " + to_string( groups == nullptr ) );
        AriesDataBufferSPtr result1;
        AriesDataBufferSPtr result2;
        auto tupleNum = refTable->GetRowCount();
        if( tupleNum > 0 )
        {
            ARIES_ASSERT( m_children.size() == 1, "m_children.size(): " + to_string( m_children.size() ) );
            AEExprNodeResult data = m_children[0]->Process( refTable );
            AriesDataBufferSPtr res;
            if( IsDataBuffer( data ) )
            {
                res = boost::get< AriesDataBufferSPtr >( data );
                res->PrefetchToGpu();
            }
            else if( IsInt( data ) )
            {
                ARIES_ASSERT( m_functionType == AriesAggFunctionType::COUNT, "m_functionType: " + GetAriesAggFunctionTypeName( m_functionType ) );
                res = make_shared< AriesDataBuffer >( AriesColumnType(
                { AriesValueType::INT32, 1 }, false, false ), 1 );
            }
            else if( IsDecimal( data ) )
            {
                auto val = boost::get< aries_acc::Decimal >( data );
                res = CreateDataBufferWithValue( val, associated->GetItemCount() );
            }
            else
            {
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "agg function result type: " + string( data.type().name() ) );
            }
            SumStrategy strategy;
            if( m_functionType == AriesAggFunctionType::AVG )
            {
                // SUM
                //strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( res, AriesAggFunctionType::SUM, groups->GetItemCount() );
                result1 = AggregateColumnData( res, AriesAggFunctionType::SUM, associated, groups, groupFlags, m_bDistinct, false, SumStrategy::NONE );
                // COUNT
                if( res->GetDataType().HasNull || m_bDistinct )
                {
                    // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( res, AriesAggFunctionType::COUNT, groups->GetItemCount() );
                    result2 = AggregateColumnData( res, AriesAggFunctionType::COUNT, associated, groups, groupFlags, m_bDistinct, true, SumStrategy::NONE );
                }
                else
                {
                    assert( itemCountInGroups );
                    result2 = itemCountInGroups;
                }
            }
            else if( m_functionType == AriesAggFunctionType::COUNT )
            {
                if( res->GetDataType().HasNull || m_bDistinct )
                {
                    // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( res, AriesAggFunctionType::SUM, groups->GetItemCount() );
                    result1 = AggregateColumnData( res, m_functionType, associated, groups, groupFlags, m_bDistinct, true,  SumStrategy::NONE  );
                }
                else
                {
                    assert( itemCountInGroups );
                    result1 = itemCountInGroups;
                }
            }
            else
            {
                // strategy = AriesDecimalAlgorithm::GetInstance().GetRuntimeStrategy( res, AriesAggFunctionType::SUM, groups->GetItemCount() );
                result1 = AggregateColumnData( res, m_functionType, associated, groups, groupFlags, m_bDistinct, false,  SumStrategy::NONE  );
            }
        }
        return
        {   AEExprNodeResult
            {   result1},AEExprNodeResult
            {   result2}};
    }

    void AEExprAggFunctionNode::GetAllParams( map< string, AEExprAggFunctionNode * > &result )
    {
        for( auto &child : m_children )
        {
            child->GetAllParams( result );
        }
    }

    void AEExprAggFunctionNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        // agg's children might be calc or case expr. they need CUmoduleSPtr.
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprAggFunctionNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprAggFunctionNode::ToString() const
    {
        return "AEExprAggFunctionNode";
    }
    /*
     * AEExprInNode
     */
    unique_ptr< AEExprInNode > AEExprInNode::Create( bool bHasNot )
    {
        return unique_ptr< AEExprInNode >( new AEExprInNode( bHasNot ) );
    }

    AEExprInNode::AEExprInNode( bool bHasNot )
            : m_bHasNot( bHasNot )
    {

    }

    AEExprInNode::~AEExprInNode()
    {

    }

    AEExprNodeResult AEExprInNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( m_children.size() > 1, "m_children.size(): " + to_string( m_children.size() ) );
        AriesBoolArraySPtr result;

        AEExprColumnIdNode* leftColumnNode = dynamic_cast< AEExprColumnIdNode * >( m_children[0].get() );
        if( leftColumnNode )
        {
            auto columnId = leftColumnNode->GetId();
            auto colEncodeType = refTable->GetColumnEncodeType( columnId );
            if( EncodeType::DICT == colEncodeType )
            {
                /*
                AriesManagedArray< AriesColumnDataIterator > columnIters( 1 );
                AriesColumnDataIteratorHelper columnIterHelper;

                AriesValueType leftValueType = AriesValueType::INT32;
                aries::AriesDataType leftDataType
                { leftValueType, 1 };
                AriesColumnType leftColumnType
                { leftDataType, false, false };
                auto& iter = columnIters[0];
                GetAriesColumnDataIteratorInfo( iter, columnIterHelper, refTable, columnId, leftColumnType, true );
                */

                AriesDataBufferSPtr indiceBuff = refTable->GetDictEncodedColumnIndiceBuffer( columnId );
                AriesColumnType indexDataType = indiceBuff->GetDataType();

                AriesDataBufferSPtr inData;
                //find out where other children data come from
                AEExprNodeResult otherChildData = m_children[1]->Process( refTable );
                if( IsDataBuffer( otherChildData ) )
                {
                    // the data may comes from a sub query
                    ARIES_ASSERT( m_children.size() == 2, "m_children.size(): " + to_string( m_children.size() ) );
                    inData = boost::get< AriesDataBufferSPtr >( otherChildData );
                    // the data type should match perfectly
                    inData = ConvertToDataBuffer( indexDataType, inData );
                }
                else
                {
                    // the data is a literal
                    ARIES_ASSERT( IsLiteral( otherChildData ), "otherChildData type: " + string( otherChildData.type().name() ) );
                    // FIXME 需要根据实际数据类型来产生inData，而非使用目前的srcData->GetDataType()
                    inData = ConvertToDataBuffer( indexDataType, refTable );
                }
                inData->PrefetchToGpu();
                AriesDataBufferSPtr params = SortData( inData, AriesOrderByType::ASC );
                if( m_bHasNot )
                    // result = CompareColumn( columnIters.GetData(), AriesComparisonOpType::NOTIN, params );
                    result = CompareColumn( indiceBuff, AriesComparisonOpType::NOTIN, params );
                else
                    // result = CompareColumn( columnIters.GetData(), AriesComparisonOpType::IN, params );
                    result = CompareColumn( indiceBuff, AriesComparisonOpType::IN, params );
                return result;
            }
        }

        AEExprNodeResult data = m_children[0]->Process( refTable );
        ARIES_ASSERT( IsDataBuffer( data ), "data type: " + string( data.type().name() ) );
        AriesDataBufferSPtr srcData = boost::get< AriesDataBufferSPtr >( data );
        srcData->PrefetchToGpu();
        AriesDataBufferSPtr inData;
        //find out where other children data come from
        // FIXME 需要根据实际数据类型来产生inData，而非使用目前的srcData->GetDataType()
        inData = ConvertToDataBuffer( srcData->GetDataType(), refTable );
        inData->PrefetchToGpu();
        AriesDataBufferSPtr params = SortData( inData, AriesOrderByType::ASC );
        if( m_bHasNot )
            result = CompareColumn( srcData, AriesComparisonOpType::NOTIN, params );
        else
            result = CompareColumn( srcData, AriesComparisonOpType::IN, params );
        return result;
    }

    AriesDataBufferSPtr AEExprInNode::ConvertToDataBuffer( AriesColumnType dataType, const AriesDataBufferSPtr& buffer ) const
    {
        AriesDataBufferSPtr result = buffer;
        AriesColumnType type = buffer->GetDataType();
        if( type.DataType.Length > 1 && type.DataType.ValueType == AriesValueType::CHAR )
        {
            ARIES_ASSERT( dataType.DataType.ValueType == AriesValueType::CHAR, "dataType.DataType.ValueType: " + GetValueTypeAsString( dataType ) );
            size_t paramCount = buffer->GetItemCount();
            dataType.HasNull = false;
            result = std::make_shared< AriesDataBuffer >( dataType );
            result->AllocArray( paramCount, true );
            size_t srcItemSize = buffer->GetItemSizeInBytes();
            int8_t* src = buffer->GetData();
            size_t dstItemSize = result->GetItemSizeInBytes();
            int8_t* dst = result->GetData();
            size_t validItemCount = 0;
            bool isNullable = type.HasNull;
            size_t srcDataSize = srcItemSize - isNullable;
            for( std::size_t i = 0; i < paramCount; ++i )
            {
                src += i * srcItemSize;
                if( isNullable )
                {
                    if( !*src )
                    {
                        if( m_bHasNot )
                        {
                            validItemCount = 0;
                            break;
                        }
                        else
                            continue;
                    }
                    ++src;
                }
                if( srcDataSize <= dstItemSize )
                {
                    memcpy( dst, src, srcDataSize );
                    ++validItemCount;
                    dst += dstItemSize;
                }
                else if( !*( src + dstItemSize ) )
                {
                    memcpy( dst, src, dstItemSize );
                    ++validItemCount;
                    dst += dstItemSize;
                }
            }
            result->SetItemCount( validItemCount );
        }
        return result;
    }

/*
mysql> select * from t1;
+------+
| f1   |
+------+
|    1 |
|    2 |
|    3 |
|    3 |
+------+

mysql> select * from t2 where f1 in (10, ( select * from t1 where f1 = 2 ), ( select * from t1 where f1 = 1 )  );
+------+
| f1   |
+------+
|    1 |
|    2 |
+------+

mysql> select * from t2 where f1 in (10, ( select * from t1 where f1 = 3 ) );
ERROR 1242 (21000): Subquery returns more than 1 row
mysql> select * from t2 where f1 in (10, ( select * from t1 where f1 = 2 ), ( select * from t1 where f1 = 3 )  );
ERROR 1242 (21000): Subquery returns more than 1 row

mysql> select * from t2 where f1 in ( select f1 as a, f1 as b from t1 where f1 = 2 );
ERROR 1241 (21000): Operand should contain 1 column(s)
mysql> select * from t2 where f1 in (10, ( select f1 as a, f1 as b from t1 where f1 = 2 ) );
ERROR 1241 (21000): Operand should contain 1 column(s)

*/
    AriesDataBufferSPtr AEExprInNode::ConvertToDataBuffer( AriesColumnType dataType, const AriesTableBlockUPtr& refTable ) const
    {
        size_t childCount = m_children.size();

        AriesDataBufferSPtr result = make_shared< AriesDataBuffer >( AriesColumnType( dataType.DataType, false, false ) );
        vector<AriesDataBufferSPtr> children_buffers;
        bool children_buffer_hasNull = false;
        vector<AEExprNodeResult> otherChildDatas;
        size_t literal_child_count = 0;
        for( std::size_t i = 1; i < childCount; ++i )
        {
            AEExprNodeResult otherChildData = m_children[i]->Process( refTable );
            otherChildDatas.push_back(otherChildData);
            if ( IsLiteral( otherChildData ) ){
                literal_child_count += 1;
            }
        }
        if(literal_child_count)
            result->AllocArray( literal_child_count );
        size_t validItemCount = 0;

        // eg. expr: a in (x,y,z), m_children[1] is x, m_children[2] is y, ...
        for( std::size_t i = 0; i < otherChildDatas.size(); ++i )
        {
            AEExprNodeResult otherChildData = otherChildDatas[i];
            if ( IsLiteral( otherChildData ) ){
                if( IsNullValue( otherChildData ) )
                {
                    if( m_bHasNot )
                    {
                        //如果not in的集合中包含null，那没有任何记录会满足条件
                        validItemCount = 0;
                        break;
                    }
                    else
                    {
                        //如果 in集合中包含null,则剔除null值，以简化opt_in的操作符的实现。
                        continue;
                    }
                }
                const type_info& type = otherChildData.type();
                switch( dataType.DataType.ValueType )
                {
                    case AriesValueType::COMPACT_DECIMAL:
                    {
                        if( type == typeid(Decimal) )
                        {
                            auto value =  (boost::get< Decimal >( otherChildData ));
                            //这里为保证cast不改变精度信息
                            if( dataType.DataType.Precision >= (value.prec) && dataType.DataType.Scale >= value.frac )
                            {
                                auto dec = aries_acc::Decimal(dataType.DataType.Precision, dataType.DataType.Scale, ARIES_MODE_STRICT_ALL_TABLES);
                                dec.cast( value );

                                if ( dec.GetError() != ERR_OVER_FLOW && dec.ToCompactDecimal( (char*)(result->GetItemDataAt( validItemCount )), dataType.DataType.Length) )
                                    ++validItemCount;
                            }
                        }
                        break;
                    }
                    case AriesValueType::DECIMAL:
                    {
                        if( type == typeid(Decimal) )
                        {
                            *reinterpret_cast< Decimal* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< Decimal >( otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::DATE:
                    {
                        if( type == typeid(AriesDate) )
                        {
                            *reinterpret_cast< AriesDate* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesDate >( otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::DATETIME:
                    {
                        if( type == typeid(AriesDatetime) )
                        {
                            *reinterpret_cast< AriesDatetime* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesDatetime >(
                                    otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::TIMESTAMP:
                    {
                        if( type == typeid(AriesTimestamp) )
                        {
                            *reinterpret_cast< AriesTimestamp* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesTimestamp >(
                                    otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::TIME:
                    {
                        if( type == typeid(AriesTime) )
                        {
                            *reinterpret_cast< AriesTime* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesTime >( otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::YEAR:
                    {
                        if( type == typeid(AriesYear) )
                        {
                            *reinterpret_cast< AriesYear* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< AriesYear >( otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::DOUBLE:
                    {
                        if( type == typeid(double) )
                        {
                            *reinterpret_cast< double* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< double >( otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::FLOAT:
                    {
                        if( type == typeid(double) )
                        {
                            *reinterpret_cast< float* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< float >( boost::get< double >(
                                    otherChildData ) );
                        }
                        break;
                    }
                    case AriesValueType::INT8:
                    {
                        if( type == typeid(int32_t) )
                        {
                            int32_t val = boost::get< int32_t >( otherChildData );
                            if( val >= numeric_limits< signed char >::min() && val <= numeric_limits< signed char >::max() )
                            {
                                *reinterpret_cast< signed char* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< signed char >( val );
                            }
                        }
                        else if( type == typeid(int64_t) )
                        {
                            int64_t val = boost::get< int64_t >( otherChildData );
                            if( val >= numeric_limits< signed char >::min() && val <= numeric_limits< signed char >::max() )
                            {
                                *reinterpret_cast< signed char* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< signed char >( val );
                            }
                        }
                        break;
                    }
                    case AriesValueType::INT16:
                    {
                        if( type == typeid(int32_t) )
                        {
                            int32_t val = boost::get< int32_t >( otherChildData );
                            if( val >= numeric_limits< short >::min() && val <= numeric_limits< short >::max() )
                            {
                                *reinterpret_cast< short* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< short >( val );
                            }
                        }
                        else if( type == typeid(int64_t) )
                        {
                            int64_t val = boost::get< int64_t >( otherChildData );
                            if( val >= numeric_limits< short >::min() && val <= numeric_limits< short >::max() )
                            {
                                *reinterpret_cast< short* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< short >( val );
                            }
                        }
                        break;
                    }
                    case AriesValueType::INT32:
                    {
                        if( type == typeid(int) )
                        {
                            int value = boost::get< int >( otherChildData );
                            *reinterpret_cast< int* >( result->GetItemDataAt( validItemCount++ ) ) = value;
                        }
                        else if( type == typeid(int64_t) )
                        {
                            int64_t val = boost::get< int64_t >( otherChildData );
                            if( val >= numeric_limits< int >::min() && val <= numeric_limits< int >::max() )
                            {
                                *reinterpret_cast< int* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< int >( val );
                            }
                        }

                        break;
                    }
                    case AriesValueType::INT64:
                    {
                        int64_t value;
                        if( type == typeid(int64_t) )
                        {
                            value = boost::get< int64_t >( otherChildData );
                        }
                        else if( type == typeid(int) )
                        {
                            value = boost::get< int >( otherChildData );
                        }
                        else
                        {
                            break;
                        }

                        *reinterpret_cast< long long* >( result->GetItemDataAt( validItemCount++ ) ) = value;
                        break;
                    }
                    case AriesValueType::UINT8:
                    {
                        if( type == typeid(int) )
                        {
                            int val = boost::get< int >( otherChildData );
                            if( val >= numeric_limits< unsigned char >::min() && val <= numeric_limits< unsigned char >::max() )
                            {
                                *reinterpret_cast< unsigned char* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< unsigned char >( val );
                            }
                        }
                        break;
                    }
                    case AriesValueType::UINT16:
                    {
                        if( type == typeid(int) )
                        {
                            int val = boost::get< int >( otherChildData );
                            if( val >= 0 && static_cast< unsigned int >( val ) >= numeric_limits< unsigned short >::min() && static_cast< unsigned int >( val ) <= numeric_limits< unsigned short >::max() )
                            {
                                *reinterpret_cast< unsigned short* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< unsigned short >( val );
                            }
                        }
                        break;
                    }
                    case AriesValueType::UINT32:
                    {
                        if( type == typeid(int) )
                        {
                            int val = boost::get< int >( otherChildData );
                            if( val >= 0 && static_cast< unsigned int >( val ) >= numeric_limits< unsigned int >::min() && static_cast< unsigned int >( val ) <= numeric_limits< unsigned int >::max() )
                            {
                                *reinterpret_cast< unsigned int* >( result->GetItemDataAt( validItemCount++ ) ) = static_cast< unsigned int >( val );
                            }
                        }
                        break;
                    }
                    case AriesValueType::UINT64:
                    {
                        if( type == typeid(int) )
                        {
                            *reinterpret_cast< unsigned long long* >( result->GetItemDataAt( validItemCount++ ) ) = boost::get< int >( otherChildData );
                        }
                        break;
                    }
                    case AriesValueType::CHAR:
                    {
                        if( type == typeid(string) )
                        {
                            string val = boost::get< string >( otherChildData );
                            size_t len = dataType.DataType.Length;
                            if( val.size() <= len )
                            {
                                val.resize( len, 0 );
                                memcpy( result->GetItemDataAt( validItemCount++ ), val.c_str(), len );
                            }
                        }
                        break;
                    }
                    default:
                    {
                        assert( 0 );
                        ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "converting data type " + GetValueTypeAsString( dataType ) + " for IN expression" );
                        break;
                    }
                }
            }
            else if (IsDataBuffer(otherChildData))
            {
                auto children_buffer = boost::get< AriesDataBufferSPtr >( otherChildData );

                if ( literal_child_count > 0 )
                {
                    //与mysql行为一致，in expr 内的子查询结果只能为1行
                    if (children_buffer->GetItemCount() != 1)
                        ARIES_EXCEPTION( ER_SUBQUERY_NO_1_ROW );
                }

                auto children_buffer_columnType = children_buffer->GetDataType();
                if (!(dataType.DataType.ValueType == children_buffer_columnType.DataType.ValueType)){
                    // TODO:如果类型之间可以相互转换，这里应转换，目前暂时不支持
                    ThrowNotSupportedException("different data type in IN expr");
                }

                if(children_buffer_columnType.HasNull)
                    children_buffer_hasNull = true;
                
                children_buffers.push_back(children_buffer);
            }
            else
            {
                ThrowNotSupportedException("unsupported datatype for in expr");
            }
        }
        result->SetItemCount( validItemCount );

        //try to merge children_buffer and the origin result into a new result
        //check databuffer's datatype, if it's not same with ohter literal, throw not support exception
        //if same, check if databuffer's hasNull is same with origin result's hasNull, they should be unified 
        if(!children_buffers.empty()){
            AriesDataBufferSPtr tmp;
            size_t total_itemCount = 0;

            vector<AriesDataBufferSPtr> children_buffers2;
            result = children_buffer_hasNull ? ConvertToNullableType(result) : result;
            total_itemCount += result->GetItemCount();
            for (auto children_buffer : children_buffers){
                children_buffer = children_buffer_hasNull ? ConvertToNullableType(children_buffer) : children_buffer;
                children_buffers2.push_back(children_buffer);
                total_itemCount += children_buffer->GetItemCount();
            }
            tmp = std::make_shared< AriesDataBuffer >( AriesColumnType( dataType.DataType, children_buffer_hasNull, false ), total_itemCount );

            size_t copied_itemCount = 0;
            if ( result->GetItemCount() > 0 )
            {
                AriesMemAllocator::MemCopy( tmp->GetData(), result->GetData(), result->GetTotalBytes() );
                copied_itemCount = result->GetItemCount();
            }
            for(auto children_buffer2 : children_buffers2){
                if ( children_buffer2->GetItemCount() > 0 )
                {
                    AriesMemAllocator::MemCopy( tmp->GetData() + tmp->GetItemSizeInBytes() * copied_itemCount,
                                                children_buffer2->GetData(),
                                                children_buffer2->GetTotalBytes() );
                    copied_itemCount += children_buffer2->GetItemCount();
                }
            }
            assert(copied_itemCount==total_itemCount);

            return tmp;
        }
        
        return result;
    }

    void AEExprInNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprInNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprInNode::ToString() const
    {
        ARIES_ASSERT( m_children.size() > 1, "m_children.size(): " + to_string( m_children.size() ) );

        string ret = this->m_children[0]->ToString();
        ret += " in (";
        ret += this->ChildrenToString_SkipFirstOne();
        ret += ")";

        return ret;
    }

    string AEExprInNode::ChildrenToString_SkipFirstOne() const
    {
        string ret;
        int count = m_children.size();
        for( int i = 1; i < count; ++i )
        {
            ARIES_ASSERT( m_children[i], "i: " + to_string( i ) );
            ret += m_children[i]->ToString();
            ret += std::string( ", " );
        }
        ARIES_ASSERT( !ret.empty(), "ret is empty" );
        return ret.substr( 0, ret.length() - 2 );
    }

    /*
     * AEExprNotNode
     */
    unique_ptr< AEExprNotNode > AEExprNotNode::Create()
    {
        return unique_ptr< AEExprNotNode >( new AEExprNotNode() );
    }

    AEExprNotNode::AEExprNotNode()
    {

    }

    AEExprNotNode::~AEExprNotNode()
    {

    }

    AEExprNodeResult AEExprNotNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        AEExprNodeResult data = m_children[0]->Process( refTable );
        ARIES_ASSERT( IsBoolArray( data ), "data type: " + string( data.type().name() ) );
        AriesBoolArraySPtr result = boost::get< AriesBoolArraySPtr >( data );

        FlipAssociated( result );
        return result;
    }

    void AEExprNotNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprNotNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprNotNode::ToString() const
    {
        ARIES_ASSERT( m_children.size() == 1, "m_children.size(): " + to_string( m_children.size() ) );
        return "not (" + this->m_children[0]->ToString() + ")";
    }

    /*
     * AEExprLikeNode
     */
    unique_ptr< AEExprLikeNode > AEExprLikeNode::Create()
    {
        return unique_ptr< AEExprLikeNode >( new AEExprLikeNode() );
    }

    AEExprLikeNode::AEExprLikeNode()
    {

    }

    AEExprLikeNode::~AEExprLikeNode()
    {

    }

    AEExprNodeResult AEExprLikeNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        AEExprNodeResult leftRes = m_children[0]->Process( refTable );
        AEExprNodeResult rightRes = m_children[1]->Process( refTable );
        ARIES_ASSERT( IsDataBuffer( leftRes ), "leftRes type: " + string( leftRes.type().name() ) );
        ARIES_ASSERT( rightRes.type() == typeid(string), "rightRes.type(): " + string( rightRes.type().name() ) );

        const AriesDataBufferSPtr leftBuf = boost::get< AriesDataBufferSPtr >( leftRes );
        leftBuf->PrefetchToGpu();
        size_t tupleNum = leftBuf->GetItemCount();
        ARIES_ASSERT( tupleNum > 0, "tupleNum > 0 error, tupleNum: " + to_string( tupleNum ) );
        string value = boost::get< string >( rightRes );

        AriesDataBufferSPtr rightBuf = make_shared< AriesDataBuffer >( AriesColumnType(
        { AriesValueType::CHAR, ( int )value.size() + 1 }, false, false ) );
        rightBuf->AllocArray( 1, true );
        memcpy( rightBuf->GetData(), value.data(), value.size() );
        return CompareColumn( leftBuf, AriesComparisonOpType::LIKE, rightBuf );
    }

    void AEExprLikeNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprLikeNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprLikeNode::ToString() const
    {
        ARIES_ASSERT( m_children.size() == 2, "m_children.size(): " + to_string( m_children.size() ) );
        string ret = m_children[0]->ToString();
        ret += " like ";
        ret += m_children[1]->ToString();
        return ret;
    }

    /*
     * AEExprSqlFunctionNode
     */
    unique_ptr< AEExprSqlFunctionNode >
    AEExprSqlFunctionNode::Create( int nodeId,
                                   int exprIndex, AriesSqlFunctionType functionType,
                                   map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                                   vector< AriesDynamicCodeParam > && params,
                                   vector< AriesDataBufferSPtr > && constValues,
                                   vector< AriesDynamicCodeComparator > && comparators,
                                   const string& expr,
                                   const AriesColumnType& valueType )
    {
        return unique_ptr< AEExprSqlFunctionNode >(
                new AEExprSqlFunctionNode( nodeId,
                                           exprIndex,
                                           functionType,
                                           std::move( aggFunctions ),
                                           std::move( params ),
                                           std::move( constValues ),
                                           std::move( comparators ),
                                           expr,
                                           valueType ) );
    }

    AEExprSqlFunctionNode::AEExprSqlFunctionNode( int nodeId,
                                                  int exprIndex,
                                                  AriesSqlFunctionType functionType,
                                                  map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                                                  vector< AriesDynamicCodeParam > && params,
                                                  vector< AriesDataBufferSPtr > && constValues,
                                                  vector< AriesDynamicCodeComparator > && comparators,
                                                  const string& expr,
                                                  const AriesColumnType& valueType )
            : AEExprDynKernelNode( nodeId, exprIndex, std::move( params ), std::move( constValues ), std::move( comparators ), expr, valueType ),
              m_functionType( functionType ),
              m_aggFunctions( std::move( aggFunctions ) )
    {
        pair< string, string > cudaFunction = GenerateCudaFunction( m_ariesParams, m_expr, m_valueType );
        m_cudaFunctionName = cudaFunction.first;
        m_cudaFunction = cudaFunction.second;
    }

    AEExprSqlFunctionNode::~AEExprSqlFunctionNode()
    {

    }

    void AEExprSqlFunctionNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        m_cuModules = modules;
        for( const auto& agg : m_aggFunctions )
        {
            agg.second->SetCuModule( modules );
        }
    }

    string AEExprSqlFunctionNode::GetCudaKernelCode() const
    {
        string code = m_cudaFunction;
        for( const auto& agg : m_aggFunctions )
        {
            code += agg.second->GetCudaKernelCode();
        }
        return code;
    }

    void AEExprSqlFunctionNode::GetAllParams( map< string, AEExprAggFunctionNode* >& result )
    {
        //获取agg参数
        for( const auto& agg : m_aggFunctions )
        {
            result[agg.first] = agg.second.get();
        }

        //获取普通column参数
        for( const auto& param : m_ariesParams )
        {
            if( param.ColumnIndex != 0 )
            {
                result[std::to_string( param.ColumnIndex )] = nullptr;
            }
        }
    }

    string AEExprSqlFunctionNode::GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const
    {
        if( !comparator.TempName.empty() )
        {
            char buf[1024];
            AriesColumnType type = comparator.Type;
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s;\n", typeName.c_str(), comparator.TempName.c_str() );
            return buf;
        }
        else
        {
            return string();
        }
    }

    pair< string, string > AEExprSqlFunctionNode::GenerateCudaFunction( const vector< AriesDynamicCodeParam >& params, const string& expr,
            const AriesColumnType& resultDataType ) const
    {
        /* ANY_VALUE 不需要动态代码 */
        if ( expr.empty() )
        {
            return { "", "" };
        }

        pair< string, string > result;
        string name = "sql_function_expr_" + std::to_string( m_exprId );
        string code = "extern \"C\"  __global__ void ";
        code += name;
        code += "( const AriesColumnDataIterator *input, int tupleNum, const int8_t** constValues, const CallableComparator** comparators, char *output )\n"
                "{\n"
                "    int stride=blockDim.x * gridDim.x;\n"
                "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    for( int i = tid; i < tupleNum; i += stride )\n"
                "    {\n";
        for( const auto& comp : m_ariesComparators )
        {
            code += GenerateTempVarCode( comp );
        }
        const char* resultValName = "Cuda_Dyn_resultValueName";
        int index = 0;
        for( const auto& param : params )
        {
            code += GenerateLoadDataCode( param, index++ );
        }

        code += "        " + GenerateParamType( resultDataType ) + " " + resultValName + " = " + expr + ";\n";
        code += GenerateSaveDataCode( resultDataType, resultValName );
        code += "    }\n"
                "}\n";
        return
        {   name, code};
    }

    string AEExprSqlFunctionNode::GenerateLoadDataCode( const AriesDynamicCodeParam& param, int index ) const
    {
        char buf[1024];
        AriesColumnType type = param.Type;

        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            if( type.HasNull )
            {
                sprintf( buf,
                        "        nullable_type< Decimal > %s( *(int8_t*)( input[%d][i] ), Decimal( (CompactDecimal*)( input[%d][i] + 1 ), %u, %u ) );\n",
                        param.ParamName.c_str(), index, index, type.DataType.Precision, type.DataType.Scale );
            }
            else
            {
                sprintf( buf, "        Decimal %s( (CompactDecimal*)( input[%d][i] ), %u, %u );\n", param.ParamName.c_str(), index,
                        type.DataType.Precision, type.DataType.Scale );
            }
        }
        else
        {
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s = *( ( %s* )( input[%d][i] ) );\n", typeName.c_str(), param.ParamName.c_str(), typeName.c_str(), index );
        }
        return buf;
    }

    string AEExprSqlFunctionNode::GenerateSaveDataCode( const AriesColumnType& type, const string& paramName ) const
    {
        return GenSaveDataCode( type, paramName );
    }

    string AEExprSqlFunctionNode::ToString() const
    {
        string result;
        map< AriesSqlFunctionType, string > sqlFuncToStr =
        {
        { AriesSqlFunctionType::SUBSTRING, "sub_string" },
        { AriesSqlFunctionType::EXTRACT, "extract" },
        { AriesSqlFunctionType::ST_VOLUMN, "ST_Volumn" },
        { AriesSqlFunctionType::DATE, "DATE" },
        { AriesSqlFunctionType::NOW, "NOW" },
        { AriesSqlFunctionType::DATE_SUB, "DATE_SUB" },
        { AriesSqlFunctionType::DATE_ADD, "DATE_ADD" },
        { AriesSqlFunctionType::ABS, "abs" } };
        result = sqlFuncToStr[m_functionType];
        return result;
    }

    /*
     * AEExprCaseNode start
     */
    unique_ptr< AEExprCaseNode >
    AEExprCaseNode::Create( int nodeId,
                            int exprIndex,
                            map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                            vector< AriesDynamicCodeParam > && params,
                            vector< AriesDataBufferSPtr > && constValues,
                            vector< AriesDynamicCodeComparator > && comparators,
                            const string& expr,
                            const AriesColumnType& valueType )
    {
        return unique_ptr< AEExprCaseNode >(
                new AEExprCaseNode( nodeId,
                                    exprIndex,
                                    std::move( aggFunctions ),
                                    std::move( params ),
                                    std::move( constValues ),
                                    std::move( comparators ),
                                    expr,
                                    valueType ) );
    }

    AEExprCaseNode::AEExprCaseNode( int nodeId,
                                    int exprIndex,
                                    map< string, unique_ptr< AEExprAggFunctionNode > > && aggFunctions,
                                    vector< AriesDynamicCodeParam > && params,
                                    vector< AriesDataBufferSPtr > && constValues,
                                    vector< AriesDynamicCodeComparator > && comparators,
                                    const string& expr,
                                    const AriesColumnType& valueType )
            : AEExprDynKernelNode( nodeId, exprIndex, std::move( params ), std::move( constValues ), std::move( comparators ), expr, valueType ),
              m_aggFunctions( std::move( aggFunctions ) )
    {
        pair< string, string > cudaFunction = GenerateCudaFunction( m_ariesParams, m_expr, m_valueType );
        m_cudaFunctionName = cudaFunction.first;
        m_cudaFunction = cudaFunction.second;
    }

    AEExprCaseNode::~AEExprCaseNode()
    {
    }

    void AEExprCaseNode::GetAllParams( map< string, AEExprAggFunctionNode* >& result )
    {
        //获取agg参数
        for( const auto& agg : m_aggFunctions )
        {
            result[agg.first] = agg.second.get();
        }

        //获取普通column参数
        for( const auto& param : m_ariesParams )
        {
            if( param.ColumnIndex > 0 )
            {
                result[std::to_string( param.ColumnIndex )] = nullptr;
            }
        }
    }

    string AEExprCaseNode::GenerateTempVarCode( const AriesDynamicCodeComparator& comparator ) const
    {
        if( !comparator.TempName.empty() )
        {
            char buf[1024];
            AriesColumnType type = comparator.Type;
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s;\n", typeName.c_str(), comparator.TempName.c_str() );
            return buf;
        }
        else
        {
            return string();
        }
    }

    pair< string, string > AEExprCaseNode::GenerateCudaFunction( const vector< AriesDynamicCodeParam >& params, const string& expr,
            const AriesColumnType& resultDataType ) const
    {
        ARIES_ASSERT( !expr.empty(), "expr is empty" );
        pair< string, string > result;
        string name = "case_expr_" + std::to_string( m_exprId );
        string code = "extern \"C\"  __global__ void ";
        code += name;
        code += "( const AriesColumnDataIterator *input, int tupleNum, const int8_t** constValues, const CallableComparator** comparators, char *output )\n"
                "{\n"
                "    int stride=blockDim.x * gridDim.x;\n"
                "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    for( int i = tid; i < tupleNum; i += stride )\n"
                "    {\n";
        for( const auto& comp : m_ariesComparators )
        {
            code += GenerateTempVarCode( comp );
        }
        const char* resultValName = "Cuda_Dyn_resultValueName";
        int index = 0;
        for( const auto& param : params )
        {
            code += GenerateLoadDataCode( param, index++, resultDataType.DataType.AdaptiveLen );
        }

        code += "        " + GenerateParamType( resultDataType ) + " " + resultValName + " = " + expr + ";\n";
        code += GenerateSaveDataCode( resultDataType, resultValName );
        code += "    }\n"
                "}\n";
        return
        {   name, code};
    }

    string AEExprCaseNode::GenerateLoadDataCode( const AriesDynamicCodeParam& param, int index, int len ) const
    {
        char buf[1024];
        AriesColumnType type = param.Type;

        if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
        {
            if( type.HasNull )
            {
                sprintf( buf,
                        "        nullable_type< Decimal > %s( *(int8_t*)( input[%d][i] ), Decimal( (CompactDecimal*)( input[%d][i] + 1 ), %u, %u ) );\n",
                        param.ParamName.c_str(), index, index, type.DataType.Precision, type.DataType.Scale );
            }
            else
            {
                sprintf( buf, "        AriesDecimal<%u> %s( (CompactDecimal*)( input[%d][i] ), %u, %u );\n", len, param.ParamName.c_str(), index, type.DataType.Precision, type.DataType.Scale );
            }
        }
        else
        {
            string typeName = GenerateParamType( type );
            sprintf( buf, "        %s %s = *( ( %s* )( input[%d][i] ) );\n", typeName.c_str(), param.ParamName.c_str(), typeName.c_str(), index );
        }
        return buf;
    }

    string AEExprCaseNode::GenerateSaveDataCode( const AriesColumnType& type, const string& paramName ) const
    {
        return GenSaveDataCode( type, paramName );
    }

    void AEExprCaseNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        m_cuModules = modules;
        for( const auto& agg : m_aggFunctions )
        {
            agg.second->SetCuModule( modules );
        }
    }

    string AEExprCaseNode::GetCudaKernelCode() const
    {
        string code = m_cudaFunction;
        for( const auto& agg : m_aggFunctions )
        {
            code += agg.second->GetCudaKernelCode();
        }
        return code;
    }

    string AEExprCaseNode::ToString() const
    {
        return m_expr;
    }

    AEExprNodeResult AEExprStarNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        // create a dummy column for count(*)
        return make_shared< AriesDataBuffer >( AriesColumnType(
        { AriesValueType::INT32, 1 }, false, false ), 1 );
    }

    AEExprNodeResult AEExprIsNullNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        AEExprNodeResult data = m_children[0]->Process( refTable );
        ARIES_ASSERT( IsDataBuffer( data ), "data type: " + string( data.type().name() ) );
        AriesDataBufferSPtr result = boost::get< AriesDataBufferSPtr >( data );

        if( m_bHasNot )
            return IsNotNull( result );
        else
            return IsNull( result );
    }

    void AEExprIsNullNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        for( auto & child : m_children )
        {
            child->SetCuModule( modules );
        }
    }

    string AEExprIsNullNode::GetCudaKernelCode() const
    {
        string code;
        for( auto & child : m_children )
        {
            code += child->GetCudaKernelCode();
        }
        return code;
    }

    AEExprNodeResult AEExprTrueFalseNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( m_children.size() == 0, "m_children.size(): " + to_string( m_children.size() ) );
        AriesBoolArraySPtr result = std::make_shared< AriesBoolArray >();
        size_t count = refTable->GetRowCount();
        if( count > 0 )
        {
            result->AllocArray( count );
            FillWithValue( result, m_bTrue ? AriesBool::ValueType::True : AriesBool::ValueType::False );
        }
        return result;
    }

    AEExprNodeResult AEExprIntervalNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        ARIES_ASSERT( m_children.size() == 1, "m_children.size(): " + to_string( m_children.size() ) );
        AEExprNodeResult data = m_children[0]->Process( refTable );
        string value;
        if( IsInt( data ) )
        {
            value = std::to_string( boost::get< int32_t >( data ) );
        }
        else
        {
            ARIES_ASSERT( IsString( data ), "data type: " + string( data.type().name() ) );
            value = boost::get< string >( data );
        }
        string unit = m_unitType;
        interval_type type = get_interval_type( unit );
        INTERVAL interval = getIntervalValue( value, type );
        return AriesInterval
        { type, interval };
    }

    string AEExprIntervalNode::ToString() const
    {
        return m_unitType;
    }

    AEExprNodeResult AEExprBufferNode::Process( const AriesTableBlockUPtr& refTable ) const
    {
        return m_buffer;
    }

    string AEExprBufferNode::ToString() const
    {
        return "AEExprBufferNode";
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
