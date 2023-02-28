/*
 * AriesSelfJoinNode.h
 *
 *  Created on: Jun 29, 2020
 *      Author: lichi
 */

#ifndef ARIESSELFJOINNODE_H_
#define ARIESSELFJOINNODE_H_

#include "AriesOpNode.h"
using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

// for self join
    struct HalfJoinCondition
    {
        AriesJoinType JoinType;
        AriesCommonExprUPtr JoinConditionExpr;
    };

    struct SelfJoinParams
    {
        AriesCommonExprUPtr CollectedFilterConditionExpr;
        vector< HalfJoinCondition > HalfJoins;
    };

    class AriesSelfJoinNode: public AriesOpNode
    {
        struct SelfJoinExprCode
        {
            string FilterExpr;
            string JoinExpr;
            string FunctionKey;
            vector< AriesDynamicCodeParam > AriesParams;
            vector< AriesDynamicCodeComparator > AriesComparators;
            vector< AriesDataBufferSPtr > ConstantValues;
        };

        struct AriesParamsComparator
        {
            bool operator()( const AriesDynamicCodeParam& a, const AriesDynamicCodeParam& b ) const
            {
                return abs( a.ColumnIndex ) < abs( b.ColumnIndex );
            }
        };

    public:
        AriesSelfJoinNode();
        ~AriesSelfJoinNode();
        void SetJoinInfo( int joinColumnId, const SelfJoinParams& joinParams );
        void SetOutputColumnIds( const vector< int >& columnIds );
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules );
        virtual string GetCudaKernelCode() const;

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;

    private:
        DynamicCodeParams GenerateDynamicCode( const SelfJoinParams& joinParams );
        string AdjustComparators( const string& code, int offset ) const;
        string AdjustConstantExprs( const string& code, int offset ) const;
        SelfJoinExprCode GenerateSelfJoinExprCode( const SelfJoinParams& joinParams ) const;

    private:
        vector< int > m_outputColumnIds;

        int m_joinColumnId;
        DynamicCodeParams m_kernelParams;
        string m_semiTemplate;
        string m_antiTemplate;
        string m_filterTemplate;
    };

    using AriesSelfJoinNodeSPtr = std::shared_ptr<AriesSelfJoinNode>;

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */

#endif /* ARIESSELFJOINNODE_H_ */
