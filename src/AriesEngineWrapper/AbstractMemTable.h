#ifndef ARIES_ABSTRACT_MEM_TABLE
#define ARIES_ABSTRACT_MEM_TABLE

#include <memory>

namespace aries
{
    class AbstractMemTable
    {
    public:
    virtual ~AbstractMemTable() = default;
    };

    typedef std::shared_ptr<AbstractMemTable> AbstractMemTablePointer;
}


#endif
