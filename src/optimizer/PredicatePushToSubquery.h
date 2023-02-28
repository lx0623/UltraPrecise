/*
 * PredicatePushToSubquery.h
 *
 *  Created on: Jul 20, 2020
 *      Author: lichi
 */

#ifndef PREDICATEPUSHTOSUBQUERY_H_
#define PREDICATEPUSHTOSUBQUERY_H_

#include "QueryOptimizationPolicy.h"
#include "frontend/SelectStructure.h"
#include <map>

using namespace std;

namespace aries
{

    class PredicatePushToSubquery: public QueryOptimizationPolicy
    {
        struct BasicRelComparator
        {
            bool operator()( const BasicRelPointer& a, const BasicRelPointer& b ) const
            {
                return a->GetDb() < b->GetDb() ? true : a->GetDb() > b->GetDb() ? false : a->GetID() < b->GetID();
            }
        };

        struct ColumnShellComparator
        {
            bool operator()( const ColumnShellPointer& a, const ColumnShellPointer& b ) const
            {
                return a->GetTableName() < b->GetTableName() ? true : a->GetTableName() > b->GetTableName() ? false : a->GetColumnName() < b->GetColumnName();
            }
        };

        struct FilterNodeInfo
        {
            vector< CommonBiaodashiPtr > ExprUsingOneTable; //all the expr will use only 1 table
            vector< CommonBiaodashiPtr > ExprUsingTwoTable; //all the expr used 2 tables
            vector< SelectStructurePointer > Subqueries; //all subquerys
            map< CommonBiaodashiPtr, vector< ColumnShellPointer > > ExprReferencedColumns; //the expr referenced columns
        };

        struct ExprToPush
        {
            vector< CommonBiaodashiPtr > FilterExpr;
            vector< CommonBiaodashiPtr > JoinExpr;
        };

    public:
        PredicatePushToSubquery();

        ~PredicatePushToSubquery();

    public:
        virtual string ToString() override;

        virtual SQLTreeNodePointer OptimizeTree( SQLTreeNodePointer arg_input ) override;

    private:
        void FindFilterNodes( const SQLTreeNodePointer& node, vector< SQLTreeNodePointer >& output ) const;

        FilterNodeInfo ProcessMainFilterNode( const SQLTreeNodePointer& filterNode ) const;

        vector< CommonBiaodashiPtr > ExtractExprOfFilterNode( const SQLTreeNodePointer& filterNode ) const;

        void PushConditionToFilterNode( SQLTreeNodePointer& filterNode, const vector< ColumnShellPointer >& outerColumns,
                const FilterNodeInfo& filterInfo, const vector< CommonBiaodashiPtr >& subqueryExprs ) const;

        bool IsTwoColumnEqualExpr( const CommonBiaodashiPtr& expr ) const;

        bool IsExprExistsInSubquery( const CommonBiaodashiPtr& expr, const vector< CommonBiaodashiPtr >& subqueryExprs ) const;

        bool AreFromSameTable( ColumnShellPointer col1, ColumnShellPointer col2 ) const;

        pair< int, vector< bool > > FindFilterToPush( const ColumnShellPointer& col, const FilterNodeInfo& filterInfo,
                const vector< CommonBiaodashiPtr >& subqueryExprs, const vector< BasicRelPointer >& subqueryTables ) const;

        vector< CommonBiaodashiPtr > FindOtherConditionToPush( const ColumnShellPointer& col, const FilterNodeInfo& filterInfo,
                vector< bool >& pushedFilterFlags, const vector< CommonBiaodashiPtr >& subqueryExprs,
                const vector< BasicRelPointer >& subqueryTables ) const;

        ExprToPush FindExprsToPushDown( const vector< ColumnShellPointer >& outerColumns, const FilterNodeInfo& filterInfo,
                const vector< CommonBiaodashiPtr >& subqueryExprs, const vector< BasicRelPointer >& subqueryTables ) const;

        void MergeFlags( vector< bool >& inout, const vector< bool >& input ) const;

        bool IsTableExistsInSubquery( const BasicRelPointer& table, const vector< BasicRelPointer >& subqueryTables ) const;

        BasicRelPointer CloneBasicRelUsingAlias( const BasicRelPointer& table, const string& alias ) const;

        string GenerateRandomTableAlias( const string& tableId ) const;

        bool AreSameColumnShell( const ColumnShellPointer& col1, const ColumnShellPointer& col2 ) const;

        void SimplifyOuterColumnExpr( vector< CommonBiaodashiPtr >& exprs, const ColumnShellPointer& outerColumn ) const;
    };

} /* namespace aries */

#endif /* PREDICATEPUSHTOSUBQUERY_H_ */
