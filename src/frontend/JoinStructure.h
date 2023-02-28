#ifndef ARIES_JOIN_STRUCTURE
#define ARIES_JOIN_STRUCTURE


#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "AbstractBiaodashi.h"
#include "BasicRel.h"
#include "server/mysql/include/mysqld.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"


namespace aries {

/***************************************************************************
 *
 *                          JoinStructure
 *
 **************************************************************************/
struct BiaodashiJoinTreeNode;
typedef std::shared_ptr<BiaodashiJoinTreeNode> BiaodashiJoinTreeNodePointer;
struct BiaodashiJoinTreeNode
{
    JoinType join_type;
    BiaodashiPointer self;
    BiaodashiJoinTreeNodePointer left;
    BiaodashiJoinTreeNodePointer right;
    std::vector<BasicRelPointer> ref_rel_array;
    std::string ToString() const;
};

class JoinStructure;
typedef std::shared_ptr<JoinStructure> JoinStructurePointer;
/*A JoinStructure means [a_rel (jointype a_rel on expr)*] */
class JoinStructure {
private:

    BiaodashiJoinTreeNodePointer expr_root_node;

    BasicRelPointer leading_rel;

    JoinStructure(const JoinStructure &arg);

    JoinStructure &operator=(const JoinStructure &arg);

    void SimplifyExprsInternal( BiaodashiJoinTreeNodePointer node, aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd );

public:

    JoinStructure();

    void SetLeadingRel(BasicRelPointer arg_br);

    BasicRelPointer GetLeadingRel();

    int GetRelCount();

    void SimplifyExprs( aries_engine::ExpressionSimplifier &exprSimplifier, THD *thd );

    BasicRelPointer GetJoinRel(int i);

    void AddJoinRel( JoinType arg_jointype, JoinStructurePointer other, BiaodashiPointer arg_bp );

    BiaodashiJoinTreeNodePointer GetJoinExprTree();

    std::string ToString();
};

}//namespace
#endif
