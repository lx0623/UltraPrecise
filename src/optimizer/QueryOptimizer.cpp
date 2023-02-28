#include "QueryOptimizer.h"

#include "server/Configuration.h"

#include "PredicatePushdown.h"
#include "SetPrimaryForeignJoin.h"
#include "SubqueryUnnesting.h"
#include "UncorrelatedSubqueryHandling.h"
#include "JoinSubqueryRemoving.h"
#include "TwoPhaseJoin.h"
#include "HavingToFilter.h"
#include "QueryTreeFormation.h"
#include "JoinReorganization.h"
#include "SelfJoin.h"
#include "GroupByColumnsSimplify.h"
#include "StarJoinBuilder.h"
#include "PredicatePushToSubquery.h"
#include "SpoolBuilder.h"
#include "ExchangeBuilder.h"
#include "PartitionConditionOptimizer.h"

namespace aries {
QueryOptimizer::QueryOptimizer() {
}

void QueryOptimizer::RegisterPolicy(QueryOptimizationPolicyPointer arg_policy) {
    this->policy_array.push_back(arg_policy);
}


SQLTreeNodePointer QueryOptimizer::OptimizeTree(SQLTreeNodePointer arg_input) {
    SQLTreeNodePointer tree = arg_input;

    for (size_t i = 0; i < this->policy_array.size(); i++) {
        std::string policy_name = this->policy_array[i]->ToString();
        LOG(INFO) << "optimization policy begin: " << policy_name << "\n";
        tree = this->policy_array[i]->OptimizeTree(tree);
        LOG(INFO) << "optimization policy end: " << policy_name << " \n";
    }

    arg_input->SetOptimized( true );

    return tree;

}

QueryOptimizerPointer QueryOptimizer::GetQueryOptimizer() {
    QueryOptimizerPointer qo = std::make_shared<QueryOptimizer>();

    qo->RegisterPolicy(std::make_shared<PredicatePushToSubquery>());

    // qo->RegisterPolicy(std::make_shared<JoinReorganization>());

    qo->RegisterPolicy(std::make_shared<PredicatePushdown>());

    qo->RegisterPolicy(std::make_shared<SetPrimaryForeignJoin>());

    qo->RegisterPolicy(std::make_shared<SubqueryUnnesting>());

    qo->RegisterPolicy(std::make_shared<UncorrelatedSubqueryHandling>());

    qo->RegisterPolicy(std::make_shared<JoinSubqueryRemoving>());

    // qo->RegisterPolicy( std::make_shared< SpoolBuilder >() );

    qo->RegisterPolicy(std::make_shared<TwoPhaseJoin>());

    //we give new join nodes an opportunity to set primary/foreign join mark!
    qo->RegisterPolicy(std::make_shared<SetPrimaryForeignJoin>());

    qo->RegisterPolicy(std::make_shared<HavingToFilter>());

    qo->RegisterPolicy(std::make_shared<GroupByColumnsSimplify>());

    qo->RegisterPolicy(std::make_shared<QueryTreeFormation>());

    qo->RegisterPolicy(std::make_shared<SelfJoin>());

    // qo->RegisterPolicy( std::make_shared< StarJoinBuilder >() );

    if ( Configuartion::GetInstance().IsExchangeEnabled() )
    {
        qo->RegisterPolicy( std::make_shared< ExchangeBuilder >() );
    }

    qo->RegisterPolicy( std::make_shared< PartitionConditionOptimizer >() );

    return qo;
}

}
