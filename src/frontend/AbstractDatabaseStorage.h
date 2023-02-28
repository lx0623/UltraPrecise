#ifndef ARIES_ABSTRACT_DATABASE_STORAGE
#define ARIES_ABSTRACT_DATABASE_STORAGE

#include <string>
#include <memory>

namespace aries {

class AbstractDatabaseStorage {
public:
    virtual ~AbstractDatabaseStorage() = default;

    virtual std::string ToString() = 0;
};

typedef std::shared_ptr<AbstractDatabaseStorage> AbstractDatabaseStoragePointer;


}//namespace

#endif
