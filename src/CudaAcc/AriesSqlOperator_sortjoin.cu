#include "AriesSqlOperator_sortjoin.h"
#include "AriesSqlOperator_helper.h"
#include "AriesSqlOperator_sort.h"
#include "filter_column.h"
#include "AriesEngineAlgorithm.h"
#include "../AriesEngine/AriesDataCache.h"
#include "CpuTimer.h"

using namespace std;

BEGIN_ARIES_ACC_NAMESPACE

    const int32_t MAX_JOIN_RESULT_COUNT = 536870911;

    AriesJoinResult Join( AriesJoinType joinType, AriesDataBufferSPtr leftData, AriesDataBufferSPtr rightData,
            const DynamicCodeParams* dynamicCodeParams, const AriesColumnDataIterator *input, bool isNotIn )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        JoinDynamicCodeParams joinDynamicCodeParams
        { dynamicCodeParams->CUModules, dynamicCodeParams->functionName, input, dynamicCodeParams->constValues, dynamicCodeParams->items };
        switch( joinType )
        {
            case AriesJoinType::SEMI_JOIN:
            {
                return SemiJoin( leftData->Clone(), rightData->Clone(), &joinDynamicCodeParams );
            }
            case AriesJoinType::ANTI_JOIN:
            {
                return AntiJoin( leftData->Clone(), rightData->Clone(), &joinDynamicCodeParams, nullptr, nullptr, isNotIn );
            }
            case AriesJoinType::LEFT_JOIN:
            {
                return LeftJoin( leftData->Clone(), rightData->Clone(), &joinDynamicCodeParams );
            }
            case AriesJoinType::INNER_JOIN:
            {
                return InnerJoin( leftData->Clone(), rightData->Clone(), &joinDynamicCodeParams );
            }
            case AriesJoinType::RIGHT_JOIN:
            {
                return RightJoin( leftData->Clone(), rightData->Clone(), &joinDynamicCodeParams );
            }
            case AriesJoinType::FULL_JOIN:
            {
                return FullJoin( leftData->Clone(), rightData->Clone(), &joinDynamicCodeParams );
            }
            default:
                assert( 0 );
                ARIES_ENGINE_EXCEPTION( ER_NOT_SUPPORTED_YET, "type " + GetAriesJoinTypeName( joinType ) + "for AND or OR expression" );
        }
    }

    AriesBoolArraySPtr SemiJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams,
            const AriesInt32ArraySPtr leftAssociated, const AriesInt32ArraySPtr rightAssociated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();
        if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
        {
            AriesColumnType type = leftType;
            type.HasNull = leftType.HasNull || rightType.HasNull;
            type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
            leftColumn = ConvertStringColumn( leftColumn, type );
            rightColumn = ConvertStringColumn( rightColumn, type );
        }
        int32_t* pLeft = nullptr;
        int32_t* pRight = nullptr;
        if( leftAssociated )
            pLeft = leftAssociated->GetData();
        if( rightAssociated )
            pRight = rightAssociated->GetData();

        auto result = sort_based_semi_join( leftColumn, rightColumn, joinDynamicCodeParams, *ctx, pLeft, pRight );
        LOG(INFO)<< "SemiJoin gpu time: " << ctx->timer_end();
        return result;

    }

    AriesBoolArraySPtr AntiJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams,
            const AriesInt32ArraySPtr leftAssociated, const AriesInt32ArraySPtr rightAssociated, bool isNotIn )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();
        if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
        {
            AriesColumnType type = leftType;
            type.HasNull = leftType.HasNull || rightType.HasNull;
            type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
            leftColumn = ConvertStringColumn( leftColumn, type );
            rightColumn = ConvertStringColumn( rightColumn, type );
        }
        int32_t* pLeft = nullptr;
        int32_t* pRight = nullptr;
        if( leftAssociated )
            pLeft = leftAssociated->GetData();
        if( rightAssociated )
            pRight = rightAssociated->GetData();
        auto result = sort_based_anti_join( leftColumn, rightColumn, joinDynamicCodeParams, *ctx, pLeft, pRight, isNotIn );
        LOG(INFO)<< "AntiJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair InnerJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const AriesInt32ArraySPtr leftAssociated,
            const AriesInt32ArraySPtr rightAssociated )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();
        if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
        {
            AriesColumnType type = leftType;
            type.HasNull = leftType.HasNull || rightType.HasNull;
            type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
            leftColumn = ConvertStringColumn( leftColumn, type );
            rightColumn = ConvertStringColumn( rightColumn, type );
        }

        JoinPair result = sort_based_inner_join( leftColumn, rightColumn, *ctx, leftAssociated, rightAssociated );
        LOG(INFO)<< "InnerJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair InnerJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        AriesColumnType leftType = leftColumn->GetDataType();
        AriesColumnType rightType = rightColumn->GetDataType();
        if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
        {
            AriesColumnType type = leftType;
            type.HasNull = leftType.HasNull || rightType.HasNull;
            type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
            leftColumn = ConvertStringColumn( leftColumn, type );
            rightColumn = ConvertStringColumn( rightColumn, type );
        }

        JoinPair result = sort_based_inner_join( leftColumn, rightColumn, *ctx, joinDynamicCodeParams );
        LOG(INFO)<< "InnerJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair LeftJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        JoinPair result;
        if( rightColumn->GetItemCount() > 0 )
        {
            AriesColumnType leftType = leftColumn->GetDataType();
            AriesColumnType rightType = rightColumn->GetDataType();
            if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
            {
                AriesColumnType type = leftType;
                type.HasNull = leftType.HasNull || rightType.HasNull;
                type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
                leftColumn = ConvertStringColumn( leftColumn, type );
                rightColumn = ConvertStringColumn( rightColumn, type );
            }

            result = sort_based_left_join( leftColumn, rightColumn, *ctx, joinDynamicCodeParams );
        }
        else
        {
            result.JoinCount = leftColumn->GetItemCount();
            result.LeftIndices = make_shared< AriesInt32Array >( result.JoinCount );
            result.RightIndices = make_shared< AriesInt32Array >( result.JoinCount );
            InitSequenceValue( result.LeftIndices );
            FillWithValue( result.RightIndices, -1 );
        }
        LOG(INFO)<< "LeftJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair RightJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        JoinPair result;
        if( leftColumn->GetItemCount() > 0 )
        {
            AriesColumnType leftType = leftColumn->GetDataType();
            AriesColumnType rightType = rightColumn->GetDataType();
            if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
            {
                AriesColumnType type = leftType;
                type.HasNull = leftType.HasNull || rightType.HasNull;
                type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
                leftColumn = ConvertStringColumn( leftColumn, type );
                rightColumn = ConvertStringColumn( rightColumn, type );
            }

            result = sort_based_right_join( leftColumn, rightColumn, *ctx, joinDynamicCodeParams );
        }
        else
        {
            result.JoinCount = rightColumn->GetItemCount();
            result.LeftIndices = make_shared< AriesInt32Array >( result.JoinCount );
            result.RightIndices = make_shared< AriesInt32Array >( result.JoinCount );
            InitSequenceValue( result.RightIndices );
            FillWithValue( result.LeftIndices, -1 );
        }
        LOG(INFO)<< "RightJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair FullJoin( AriesDataBufferSPtr leftColumn, AriesDataBufferSPtr rightColumn, const JoinDynamicCodeParams* joinDynamicCodeParams )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        JoinPair result;
        if( leftColumn->GetItemCount() > 0 && rightColumn->GetItemCount() > 0 )
        {
            AriesColumnType leftType = leftColumn->GetDataType();
            AriesColumnType rightType = rightColumn->GetDataType();
            if( leftType.DataType.ValueType == AriesValueType::CHAR && rightType.DataType.ValueType == AriesValueType::CHAR )
            {
                AriesColumnType type = leftType;
                type.HasNull = leftType.HasNull || rightType.HasNull;
                type.DataType.Length = std::max( leftType.DataType.Length, rightType.DataType.Length );
                leftColumn = ConvertStringColumn( leftColumn, type );
                rightColumn = ConvertStringColumn( rightColumn, type );
            }

            result = sort_based_full_join( leftColumn, rightColumn, *ctx, joinDynamicCodeParams );
        }
        else if( leftColumn->GetItemCount() > 0 )
        {
            result.JoinCount = leftColumn->GetItemCount();
            result.LeftIndices = make_shared< AriesInt32Array >( result.JoinCount );
            result.RightIndices = make_shared< AriesInt32Array >( result.JoinCount );
            InitSequenceValue( result.LeftIndices );
            FillWithValue( result.RightIndices, -1 );
        }
        else if( rightColumn->GetItemCount() > 0 )
        {
            result.JoinCount = rightColumn->GetItemCount();
            result.LeftIndices = make_shared< AriesInt32Array >( result.JoinCount );
            result.RightIndices = make_shared< AriesInt32Array >( result.JoinCount );
            InitSequenceValue( result.RightIndices );
            FillWithValue( result.LeftIndices, -1 );
        }
        LOG(INFO)<< "FullJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair CartesianProductJoin( size_t leftCount, size_t rightCount )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        ctx->timer_begin();
        JoinPair result = create_cartesian_product( leftCount, rightCount, *ctx );
        LOG(INFO)<< "CartesianProductJoin gpu time: " << ctx->timer_end();
        return result;
    }

    JoinPair CartesianJoin( AriesJoinType joinType, size_t leftCount, size_t rightCount, const DynamicCodeParams& dynamicCodeParams,
            const AriesColumnDataIterator *input )
    {
        auto ctx = AriesSqlOperatorContext::GetInstance().GetContext();
        JoinPair result;
        JoinDynamicCodeParams* joinDynamicCodeParams = nullptr;
        auto params = std::make_shared< JoinDynamicCodeParams >();
        if( !dynamicCodeParams.functionName.empty() )
        {
            joinDynamicCodeParams = params.get();
            joinDynamicCodeParams->comparators = dynamicCodeParams.items;
            joinDynamicCodeParams->cuModules = dynamicCodeParams.CUModules;
            joinDynamicCodeParams->functionName = dynamicCodeParams.functionName;
            joinDynamicCodeParams->input = input;
            joinDynamicCodeParams->constValues = dynamicCodeParams.constValues;;
        }
        join_pair_t< int > pair;
        pair.count = 0;
        switch( joinType )
        {
            case AriesJoinType::INNER_JOIN:
            {
                pair = cartesian_join_wrapper( leftCount, rightCount, false, false, *ctx, joinDynamicCodeParams );
                break;
            }
            case AriesJoinType::LEFT_JOIN:
            {
                pair = cartesian_join_wrapper( leftCount, rightCount, true, false, *ctx, joinDynamicCodeParams );
                break;
            }
            case AriesJoinType::RIGHT_JOIN:
            {
                pair = cartesian_join_wrapper( leftCount, rightCount, false, true, *ctx, joinDynamicCodeParams );
                break;
            }
            case AriesJoinType::FULL_JOIN:
            {
                pair = cartesian_join_wrapper( leftCount, rightCount, true, true, *ctx, joinDynamicCodeParams );
                break;
            }
            default:
                ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "cartesian join for other join type is not supportted yet." );
                // break;
        }

        if( pair.count > 0 )
        {
            result.LeftIndices = std::make_shared< AriesInt32Array >();
            result.RightIndices = std::make_shared< AriesInt32Array >();
            result.JoinCount = pair.count;

            result.LeftIndices->AttachBuffer( pair.left_indices.release_data(), pair.count );
            result.RightIndices->AttachBuffer( pair.right_indices.release_data(), pair.count );
        }
        return result;
    }

END_ARIES_ACC_NAMESPACE
