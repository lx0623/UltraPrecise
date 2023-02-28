/*
 * AriesSelfJoinNode.cpp
 *
 *  Created on: Jun 29, 2020
 *      Author: lichi
 */
#include <regex>
#include <set>
#include "AriesSelfJoinNode.h"
#include "CudaAcc/DynamicKernel.h"
#include "AriesUtil.h"
#include "CpuTimer.h"
#include "AriesCommonExpr.h"
#include "AriesSpoolCacheManager.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesSelfJoinNode::AriesSelfJoinNode()
            : m_joinColumnId( -1 )
    {
        m_opName="self";
        m_semiTemplate =
                R"(
            if( !anti_matched )
            {
                semi_matched = 0;
                for( int32_t pos = 0; pos < group_size; ++pos )
                {
                    sibling_index = associated[ group_start_pos + pos ];
                    if( #condition# )
                    {
                        semi_matched = 1;
                        break;
                    }
                }
                if( !semi_matched )
                    continue;
            }
)";
        m_antiTemplate =
                R"(
            if( semi_matched )
            {
                anti_matched = 0;
                for( int32_t pos = 0; pos < group_size; ++pos )
                {
                    sibling_index = associated[ group_start_pos + pos ];
                    if( #condition# )
                    {
                        anti_matched = 1;
                        break;
                    }
                }
                if( anti_matched )
                    continue;
            }
)";
        m_filterTemplate =
                R"(if( #condition# ))";
    }

    void AriesSelfJoinNode::SetJoinInfo( int joinColumnId, const SelfJoinParams& joinParams )
    {
        m_joinColumnId = joinColumnId;
        m_kernelParams = GenerateDynamicCode( joinParams );
    }

    AriesSelfJoinNode::~AriesSelfJoinNode()
    {
        // TODO Auto-generated destructor stub
    }

    void AriesSelfJoinNode::SetOutputColumnIds( const vector< int >& columnIds )
    {
        m_outputColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void AriesSelfJoinNode::SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        m_kernelParams.CUModules = modules;
        m_leftSource->SetCuModule( modules );
    }

    string AriesSelfJoinNode::GetCudaKernelCode() const
    {
        return m_leftSource->GetCudaKernelCode() + m_kernelParams.code;
    }

    bool AriesSelfJoinNode::Open()
    {
        assert( m_joinColumnId != -1 );
        return m_leftSource->Open();
    }

    AriesOpResult AriesSelfJoinNode::GetNext()
    {
        AriesOpResult cachedResult = GetCachedResult();
        if ( AriesOpNodeStatus::END == cachedResult.Status )
            return cachedResult;
        ARIES_ASSERT( m_leftSource, "m_leftSource is nullptr" );
        AriesOpResult allData;

        // read all data
        AriesOpResult data = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
#endif
        allData.TableBlock = move( data.TableBlock );
        if ( allData.TableBlock )
            allData.TableBlock->ResetAllStats();

        while( data.Status == AriesOpNodeStatus::CONTINUE && !IsCurrentThdKilled() )
        {
            data = m_leftSource->GetNext();
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if( data.Status == AriesOpNodeStatus::ERROR )
            {
                break;
            }
            allData.TableBlock->AddBlock( move( data.TableBlock ) );
#ifdef ARIES_PROFILE
            m_opTime += t.end();
#endif
        }
        allData.Status = data.Status;

        if ( allData.TableBlock )
            m_rowCount += allData.TableBlock->GetRowCount();
#ifdef ARIES_PROFILE
        t.begin();
#endif
        if ( IsCurrentThdKilled() )
        {
            LOG(INFO) << "thread was kill in AriesSortNode::GetNext";
            SendKillMessage();
        }

        if( allData.Status == AriesOpNodeStatus::END )
        {
            int outTupleNum = 0;
            auto tupleNum = allData.TableBlock->GetRowCount();
            if( tupleNum > 0 )
            {
                vector< AriesDataBufferSPtr > groupByColumns;
                AriesInt32ArraySPtr outAssociated;
                AriesInt32ArraySPtr outGroups;
                AriesInt32ArraySPtr outGroupFlags;
                auto joinColumn = allData.TableBlock->GetColumnBuffer( m_joinColumnId );
                groupByColumns.push_back( joinColumn );

                int32_t groupCount = aries_acc::GroupColumns( groupByColumns, outAssociated, outGroups );

                AriesInt32ArraySPtr groupsPrefixSum;
                int64_t joinCount = aries_acc::GetSelfJoinGroupSizePrefixSum( outGroups, tupleNum, groupsPrefixSum );

                AriesManagedArray< AriesColumnDataIterator > columns( m_kernelParams.params.size() );
                vector< AriesColumnDataIteratorHelper > columnHelpers( m_kernelParams.params.size() );
                for( std::size_t i = 0; i < m_kernelParams.params.size(); i++ )
                {
                    const auto& param = m_kernelParams.params[i];
                    auto& iter = columns[i];
                    auto& iterHelper = columnHelpers[ i ];
                    int columnId = abs( param.ColumnIndex );
                    GetAriesColumnDataIteratorInfo( iter, iterHelper, allData.TableBlock, columnId, param.Type, param.UseDictIndex );
                }

                AriesManagedArray< int8_t* > constantPtrs( m_kernelParams.constantValues.size() );
                for ( std::size_t i = 0; i < m_kernelParams.constantValues.size(); i++ )
                {
                    const auto& value = m_kernelParams.constantValues[ i ];
                    constantPtrs.GetData()[ i ] = value->GetData();
                }

                AriesInt8ArraySPtr output = std::make_shared< AriesInt8Array >( tupleNum, true );
                int8_t* pOutput = output->GetData();
                columns.PrefetchToGpu();
                AriesDynamicKernelManager::GetInstance().CallKernel( m_kernelParams.CUModules, m_kernelParams.functionName.c_str(), columns.GetData(), outAssociated->GetData(),
                        outGroups->GetData(), groupsPrefixSum->GetData(), groupCount, joinCount, ( const int8_t** )( constantPtrs.GetData() ), m_kernelParams.items, pOutput );
                AriesIndicesArraySPtr indices = aries_acc::FilterFlags( output );
                outTupleNum = indices->GetItemCount();
                if( outTupleNum > 0 )
                {
                    if( !m_outputColumnIds.empty() )
                    {
                        allData.TableBlock = allData.TableBlock->MakeTableByColumns( m_outputColumnIds, false );
                        map< int, int > idToUpdate;
                        int outputId = 0;
                        for ( const auto& id : m_outputColumnIds )
                            idToUpdate[ ++outputId ] = id;
                        allData.TableBlock->UpdateColumnIds( idToUpdate );
                    }
                        
                    allData.TableBlock->ResetAllStats();
                    allData.TableBlock->UpdateIndices( indices );
                }

                const auto& tableStats = allData.TableBlock->GetStats();
                tableStats.Print( "AriesSelfJoinNode::GetNext, process expr" );
                m_tableStats += tableStats;
            }
            if( outTupleNum == 0 )
            {
                if( !m_outputColumnIds.empty() )
                {
                    auto outputColumnTypes = allData.TableBlock->GetColumnTypes( m_outputColumnIds );
                    allData.TableBlock = allData.TableBlock->CreateTableWithNoRows( outputColumnTypes );
                }
                else
                {
                    allData.TableBlock = std::make_unique< AriesTableBlock >();
                }
            }
        }
#ifdef ARIES_PROFILE
        m_opTime += t.end();
#endif
        CacheNodeData( allData.TableBlock );

        return allData;
    }

    void AriesSelfJoinNode::Close()
    {
        m_leftSource->Close();
    }

    DynamicCodeParams AriesSelfJoinNode::GenerateDynamicCode( const SelfJoinParams& joinParams )
    {
        DynamicCodeParams kernelParams;

        kernelParams.functionName = "self_join_fun_" + std::to_string( m_nodeId );

        std::string functionCodeTemplate =
                R"(
extern "C" __global__ void #function_name#( const AriesColumnDataIterator *input, const int32_t* associated, const int32_t* groups,
        const int32_t *group_size_prefix_sum, int32_t group_count, int32_t tupleNum, const int8_t** constValues, const CallableComparator** comparators, int8_t *output )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int self_index;
    int sibling_index;
    int group_index;
    int group_size;
    int group_start_pos;
    int semi_matched = 1;
    int anti_matched = 0;
    
    for( int32_t i = tid; i < tupleNum; i += stride )
    {
        group_index = binary_search< bounds_upper >( group_size_prefix_sum, group_count, i ) - 1;
        group_start_pos = groups[group_index];
        if( group_index < group_count - 1 )
            group_size = groups[group_index + 1] - group_start_pos;
        else
            group_size = tupleNum - groups[group_index];
        self_index = associated[i];

        #filter_condition#
        {
            #join_condition#

            output[self_index] = semi_matched && !anti_matched;
        }
    }
}
)"
        ;
        string code = ReplaceString( functionCodeTemplate, "#function_name#", kernelParams.functionName );
        SelfJoinExprCode exprCode = GenerateSelfJoinExprCode( joinParams );
        code = ReplaceString( code, "#filter_condition#", exprCode.FilterExpr );
        code = ReplaceString( code, "#join_condition#", exprCode.JoinExpr );

        kernelParams.code = std::move( code );
        kernelParams.items = std::move( exprCode.AriesComparators );
        kernelParams.params = std::move( exprCode.AriesParams );
        kernelParams.constantValues = std::move( exprCode.ConstantValues );

        return kernelParams;
    }

    AriesSelfJoinNode::SelfJoinExprCode AriesSelfJoinNode::GenerateSelfJoinExprCode( const SelfJoinParams& joinParams ) const
    {
        SelfJoinExprCode result;

        set< AriesDynamicCodeParam, AriesParamsComparator > referencedColumns;
        vector< AriesDynamicCodeParam > allAriesParams;

        // handle filter
        if( joinParams.CollectedFilterConditionExpr )
        {
            map< string, AriesCommonExprUPtr > aggFunctions;
            vector< AriesDynamicCodeParam > ariesParams;
            vector< AriesDataBufferSPtr > constValues;
            vector< AriesDynamicCodeComparator > ariesComparators;

            string expr = joinParams.CollectedFilterConditionExpr->StringForDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators );
            expr = AdjustComparators( expr, result.AriesComparators.size() );
            expr = AdjustConstantExprs( expr, result.ConstantValues.size() );

            referencedColumns.insert( ariesParams.begin(), ariesParams.end() );
            std::copy( ariesComparators.begin(), ariesComparators.end(), back_inserter( result.AriesComparators ) );
            std::copy( ariesParams.begin(), ariesParams.end(), back_inserter( allAriesParams ) );
            std::copy( constValues.cbegin(), constValues.cend(), back_inserter( result.ConstantValues ) );

            result.FilterExpr = m_filterTemplate;
            ReplaceString( result.FilterExpr, "#condition#", expr );

            result.FunctionKey += joinParams.CollectedFilterConditionExpr->ToString() + "_Filter";
        }

        // handle half join
        for( const auto& joinInfo : joinParams.HalfJoins )
        {
            map< string, AriesCommonExprUPtr > aggFunctions;
            vector< AriesDynamicCodeParam > ariesParams;
            vector< AriesDataBufferSPtr > constValues;
            vector< AriesDynamicCodeComparator > ariesComparators;

            string expr = joinInfo.JoinConditionExpr->StringForDynamicCode( aggFunctions, ariesParams, constValues, ariesComparators );
            expr = AdjustComparators( expr, result.AriesComparators.size() );
            expr = AdjustConstantExprs( expr, result.ConstantValues.size() );

            referencedColumns.insert( ariesParams.begin(), ariesParams.end() );
            std::copy( ariesComparators.begin(), ariesComparators.end(), back_inserter( result.AriesComparators ) );
            std::copy( ariesParams.begin(), ariesParams.end(), back_inserter( allAriesParams ) );
            std::copy( constValues.cbegin(), constValues.cend(), back_inserter( result.ConstantValues ) );

            string keySuffix;
            string joinExpr;
            switch( joinInfo.JoinType )
            {
                case AriesJoinType::SEMI_JOIN:
                    joinExpr = m_semiTemplate;
                    keySuffix = "_SEMI";
                    break;
                case AriesJoinType::ANTI_JOIN:
                    joinExpr = m_antiTemplate;
                    keySuffix = "_ANTI";
                    break;
                default:
                    assert( 0 );
                    break;
            }
            ReplaceString( joinExpr, "#condition#", expr );
            result.JoinExpr += joinExpr;

            result.FunctionKey += joinInfo.JoinConditionExpr->ToString() + keySuffix;
        }

        // handle input column params
        result.AriesParams.assign( referencedColumns.begin(), referencedColumns.end() );

        map< int, int > idIndexMapping;
        int index = 0;
        for( const auto& param : result.AriesParams )
            idIndexMapping[abs( param.ColumnIndex )] = index++;

        for( const auto& param : allAriesParams )
        {
            assert( idIndexMapping.find( abs( param.ColumnIndex ) ) != idIndexMapping.end() );
            int index = idIndexMapping[abs( param.ColumnIndex )];

            char buf[1024];
            AriesColumnType type = param.Type;
            if( type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL )
            {
                if( type.HasNull )
                {
                    sprintf( buf,
                            "( nullable_type< Decimal >( *(int8_t*)( input[%d][%s] ), Decimal( (CompactDecimal*)( input[%d][%s] + 1 ), %u, %u ) ) )",
                            index, param.ColumnIndex > 0 ? "self_index" : "sibling_index", index,
                            param.ColumnIndex > 0 ? "self_index" : "sibling_index", type.DataType.Precision, type.DataType.Scale );
                }
                else
                {
                    sprintf( buf, "( Decimal( (CompactDecimal*)( input[%d][%s] ), %u, %u ) )", index,
                            param.ColumnIndex > 0 ? "self_index" : "sibling_index", type.DataType.Precision, type.DataType.Scale );
                }
            }
            else
            {
                ::sprintf( buf, "(*( ( %s* )( input[%d][%s] ) ) )", GenerateParamType( type ).c_str(), index,
                        param.ColumnIndex > 0 ? "self_index" : "sibling_index" );
            }

            ReplaceString( result.FilterExpr, param.ParamName, std::string( buf ) );
            ReplaceString( result.JoinExpr, param.ParamName, std::string( buf ) );
        }

        return result;
    }

    string AriesSelfJoinNode::AdjustConstantExprs( const string& code, int offset ) const
    {
        string kernelCode = code;
        if( offset > 0 )
        {
            regex words_regex( "constValues\\[\\s+(\\d+)\\s+\\]" );
            auto words_begin = sregex_iterator( code.begin(), code.end(), words_regex );
            auto words_end = sregex_iterator();
            for( sregex_iterator i = words_begin; i != words_end; ++i )
            {
                smatch match = *i;
                kernelCode.replace( match.position(), match.length(), "constValues[ " + std::to_string( stoi( match[1] ) + offset ) + " ]" );
            }
        }
        return kernelCode;
    }

    string AriesSelfJoinNode::AdjustComparators( const string& code, int offset ) const
    {
        string kernelCode = code;
        if( offset > 0 )
        {
            regex words_regex( "comparators\\[\\s+(\\d+)\\s+\\]" );
            auto words_begin = sregex_iterator( code.begin(), code.end(), words_regex );
            auto words_end = sregex_iterator();
            for( sregex_iterator i = words_begin; i != words_end; ++i )
            {
                smatch match = *i;
                kernelCode.replace( match.position(), match.length(), "comparators[ " + std::to_string( stoi( match[1] ) + offset ) + " ]" );
            }
        }
        return kernelCode;
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */
