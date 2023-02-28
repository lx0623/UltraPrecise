//
// Created by tengjp on 19-8-16.
//

#ifndef AIRES_TABLENAMESTRUCTURE_H
#define AIRES_TABLENAMESTRUCTURE_H

#include <string>
#include <memory>
#include "AriesDefinition.h"

using std::string;
NAMESPACE_ARIES_START
class TableNameStructure {
public:
    TableNameStructure() = default;
    TableNameStructure(const string& dbNameArg, const string& tableNameArg) :
            dbName(dbNameArg),
            tableName(tableNameArg) {}
    string dbName;
    string tableName;
};
using TableNameStructureSPtr = std::shared_ptr<TableNameStructure>;

NAMESPACE_ARIES_END // namespace aries

#endif //AIRES_TABLENAMESTRUCTURE_H
