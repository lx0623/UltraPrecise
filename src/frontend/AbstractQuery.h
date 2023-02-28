#ifndef ARIES_ABSTARCT_QUERY
#define ARIES_ABSTARCT_QUERY

#include <string>
#include <memory>

namespace aries {

/*A SQL query will be compiled into an AbstractQuery!
     *No need to use this class or its pointer typedef.
     *The real implementation is SelectStructure! Use it!
     */

class AbstractQuery {
public:
    virtual ~AbstractQuery() = default;
    virtual std::string ToString() = 0;
};

typedef std::shared_ptr<AbstractQuery> AbstractQueryPointer;

}//namespace

#endif

