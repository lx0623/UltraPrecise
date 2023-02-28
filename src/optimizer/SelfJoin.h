/*
 * SelfJoin.h
 *
 *  Created on: Jun 28, 2020
 *      Author: lichi
 */

#ifndef SELFJOIN_H_
#define SELFJOIN_H_

#include "QueryOptimizationPolicy.h"
using namespace std;

namespace aries
{

    class SelfJoin: public QueryOptimizationPolicy
    {
    public:
        SelfJoin();

        ~SelfJoin();

    public:
        virtual string ToString() override;

        virtual SQLTreeNodePointer OptimizeTree( SQLTreeNodePointer arg_input ) override;

    private:
        //indicate what nodes are allowed during search self join process.
        enum class AllowedNodeTypes
        {
            HALFJOIN, HALFJOIN_FILTER_TABLE, FILTER_TABLE, NONE
        };

        // the search context is used to keep all needed info to check if we can find a self join case.
        struct SelfJoinSearchContext
        {
            AllowedNodeTypes AllowedTypes = AllowedNodeTypes::HALFJOIN;// seach for half join node initially
            BasicRelPointer SourceTable;// the one and only table used in self join.
            int ColumnLocation = -1;// the join key column.
            void Reset()
            {
                AllowedTypes = AllowedNodeTypes::HALFJOIN;
                SourceTable.reset();
                ColumnLocation = -1;
            }
        };

        struct SelfJoinSearchResult
        {
            SQLTreeNodePointer RootNode;// the root node which will be replaced by a self join node.
            SelfJoinSearchContext Context;
        };

        SelfJoinSearchResult SearchSelfJoin( SQLTreeNodePointer root ) const;

        bool IsSelfJoin( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const;

        bool IsJoinConditionMatched( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const;

        bool IsFilterConditionMatched( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const;

        bool IsTableMatched( SQLTreeNodePointer arg_input, SelfJoinSearchContext& context ) const;

        bool IsHalfJoinNode( SQLTreeNodePointer arg_input ) const;

        bool IsFilterNode( SQLTreeNodePointer arg_input ) const
        {
            return arg_input->GetType() == SQLTreeNodeType::Filter_NODE;
        }

        bool IsTableNode( SQLTreeNodePointer arg_input ) const
        {
            return arg_input->GetType() == SQLTreeNodeType::Table_NODE;
        }

        void CollectSelfJoinInfo( SQLTreeNodePointer node, int activeJoinInfo, vector< HalfJoinInfo >& joinInfo ) const;

        CommonBiaodashiPtr ExtractFilterCondition( SQLTreeNodePointer node ) const;

        bool AreTheSamePhysicalTable( BasicRelPointer rel1, BasicRelPointer rel2 ) const
        {
            return !rel1->IsSubquery() && !rel2->IsSubquery() && rel1->GetDb() == rel2->GetDb() && rel1->GetID() == rel2->GetID();
        }

        void AdjustColumnPositionInChildTable( const map< int, int >& idMapping, const vector< ColumnShellPointer >& columns ) const;
    };

} /* namespace aries */

#endif /* SELFJOIN_H_ */
