#ifndef ARIES_SET_PRIMARY_FOREIGN_JOIN
#define ARIES_SET_PRIMARY_FOREIGN_JOIN

#include "QueryOptimizationPolicy.h"

namespace aries {

class SetPrimaryForeignJoin : public QueryOptimizationPolicy {

private:

    bool see_unsafe = false;

public:

    SetPrimaryForeignJoin();

    virtual std::string ToString() override;

    virtual SQLTreeNodePointer OptimizeTree(SQLTreeNodePointer arg_input) override;


    void set_primary_foreign_join_single_query(SQLTreeNodePointer arg_input);

    void set_primary_foreign_join_handling_node(SQLTreeNodePointer arg_input);

    bool process_join_node(SQLTreeNodePointer arg_input);


};


}//namespace


#endif
