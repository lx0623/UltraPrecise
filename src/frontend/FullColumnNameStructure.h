//
// Created by tengjp on 19-8-23.
//

#ifndef AIRES_FULLCOLUMNNAMESTRUCTURE_H
#define AIRES_FULLCOLUMNNAMESTRUCTURE_H

#include <string>
#include <memory>
#include "AriesDefinition.h"

using std::string;


NAMESPACE_ARIES_START
class FullColumnNameStructure {
public:
    string dbName;
    string tableName;
    string columnName;
    string fullName;
};
using FullColumnNameStructurePointer = std::shared_ptr<FullColumnNameStructure>;
NAMESPACE_ARIES_END // namespace aries


#endif //AIRES_FULLCOLUMNNAMESTRUCTURE_H
