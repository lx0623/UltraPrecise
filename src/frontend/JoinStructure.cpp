#include "server/mysql/include/mysqld.h"
#include "ShowUtility.h"
#include "JoinStructure.h"
#include "CommonBiaodashi.h"

namespace aries {
JoinStructure::JoinStructure() {
}

void JoinStructure::SetLeadingRel(BasicRelPointer arg_br) {
    this->leading_rel = (arg_br);
    expr_root_node = std::make_shared< BiaodashiJoinTreeNode >();
    expr_root_node->ref_rel_array.push_back( arg_br );
}

BasicRelPointer JoinStructure::GetLeadingRel() {
    return this->leading_rel;
}

int JoinStructure::GetRelCount() 
{
    assert( expr_root_node );
    return expr_root_node->ref_rel_array.size();
}

void JoinStructure::SimplifyExprsInternal( BiaodashiJoinTreeNodePointer node, aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd )
{
    assert( node );
    if( node->self )
    {
        auto commonExpr = ( CommonBiaodashi* )( node->self.get() );
        auto simplifiedExpr = exprSimplifier.SimplifyAsCommonBiaodashi( commonExpr, thd );
        if ( simplifiedExpr )
            node->self = simplifiedExpr;
        assert( node->left && node->right );
        SimplifyExprsInternal( node->left, exprSimplifier, thd );
        SimplifyExprsInternal( node->right, exprSimplifier, thd );
    }
}

void JoinStructure::SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd )
{
    assert( expr_root_node );
    SimplifyExprsInternal( expr_root_node, exprSimplifier, thd );
}

BasicRelPointer JoinStructure::GetJoinRel( int i ) 
{
    assert( expr_root_node );
    assert( i >= 0 && ( size_t )i < expr_root_node->ref_rel_array.size() );
    return expr_root_node->ref_rel_array[ i ];
}

BiaodashiJoinTreeNodePointer JoinStructure::GetJoinExprTree()
{
    return expr_root_node;
}

void JoinStructure::AddJoinRel( JoinType arg_jointype, JoinStructurePointer other, BiaodashiPointer arg_bp )
{
    assert( expr_root_node );
    auto old_root_node = expr_root_node;
    expr_root_node = std::make_shared< BiaodashiJoinTreeNode >();
    expr_root_node->join_type = arg_jointype;
    expr_root_node->self = arg_bp;
    expr_root_node->left = old_root_node;
    expr_root_node->right = other->GetJoinExprTree();
    expr_root_node->ref_rel_array.assign( expr_root_node->left->ref_rel_array.begin(), expr_root_node->left->ref_rel_array.end() );
    expr_root_node->ref_rel_array.insert( expr_root_node->ref_rel_array.end(), expr_root_node->right->ref_rel_array.begin(), expr_root_node->right->ref_rel_array.end() );
}

std::string JoinStructure::ToString() {
    std::string result = "";

    assert(this->leading_rel != nullptr);
    result += this->leading_rel->ToString();

    result += expr_root_node->ToString();

    result += " ";

    return result;
}

std::string BiaodashiJoinTreeNode::ToString() const
{
    std::string result;
    if( self )
    {
        result += left->ToString() + " ";
        result += ShowUtility::GetInstance()->GetTextFromJoinType( join_type ) + " on ";
        result += self->ToString() + " ";
        result += right->ToString() + " ";
    }
    return result;
}

}
